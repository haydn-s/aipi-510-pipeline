import hashlib
import json
import logging
from pathlib import Path

import mlflow
import mlflow.sklearn
import pytest
from mlflow import MlflowClient
from sklearn.datasets import load_iris
from structlog.contextvars import bound_contextvars, get_contextvars

from aipi_pipeline.config import Settings
from aipi_pipeline.demo import run_demo
from aipi_pipeline.run_config import RunConfig
from aipi_pipeline.logging import close_logging, configure_logging, flush_logs, get_logger
from aipi_pipeline.tracking import pipeline_run, pipeline_step


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_LOGGING", "false")
    original_uri = mlflow.get_tracking_uri()
    config = Settings(
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        experiment_name="test-pipeline",
        run_dir=tmp_path / "runs",
        log_format="json",
    )
    yield config
    close_logging()
    while mlflow.active_run() is not None:
        mlflow.end_run()
    mlflow.set_tracking_uri(original_uri)


def run_config(name):
    return RunConfig({"name": name, "model": {"type": "test"}})


def run_directory(settings, run_id):
    return settings.run_dir / mlflow.get_run(run_id).data.tags["pipeline.run_hash"]


def read_events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_success_tracks_steps_and_uploads_correlated_logs(settings):
    with pipeline_run(run_config("success"), settings) as parent:
        with pipeline_step("ingest") as child:
            get_logger("test").info("loaded", rows=10)
            mlflow.log_metric("rows", 10)
        assert mlflow.active_run().info.run_id == parent.info.run_id
        assert get_contextvars()["step"] == "pipeline"

    client = MlflowClient(tracking_uri=settings.tracking_uri)
    child_record = client.get_run(child.info.run_id)
    assert client.get_run(parent.info.run_id).info.status == "FINISHED"
    assert child_record.info.status == "FINISHED"
    assert child_record.data.tags["mlflow.parentRunId"] == parent.info.run_id
    assert child_record.data.metrics["rows"] == 10
    assert child_record.data.metrics["duration_seconds"] >= 0
    assert mlflow.active_run() is None

    log_file = run_directory(settings, parent.info.run_id) / "events.jsonl"
    events = read_events(log_file)
    loaded = next(event for event in events if event["event"] == "loaded")
    assert loaded["run_id"] == child.info.run_id
    assert loaded["parent_run_id"] == parent.info.run_id
    assert loaded["step"] == "ingest"
    assert loaded["pipeline"] == "success"
    assert loaded["level"] == "info"
    assert loaded["timestamp"]
    parent_record = client.get_run(parent.info.run_id)
    assert loaded["run_hash"] == parent_record.data.tags["pipeline.run_hash"]
    assert loaded["config_hash"] == run_config("success").config_hash
    assert child_record.data.tags["pipeline.config_hash"] == loaded["config_hash"]
    config_file = client.download_artifacts(parent.info.run_id, "config/config.json")
    assert hashlib.sha256(Path(config_file).read_bytes()).hexdigest() == loaded["config_hash"]
    manifest_file = client.download_artifacts(parent.info.run_id, "config/manifest.json")
    manifest = json.loads(Path(manifest_file).read_text())
    assert manifest["status"] == "FINISHED"
    assert manifest["run_hash"] == loaded["run_hash"]
    assert manifest["mlflow_run_id"] == parent.info.run_id
    archived = client.download_artifacts(parent.info.run_id, f"logs/{log_file.name}")
    assert Path(archived).read_bytes() == log_file.read_bytes()


def test_failure_marks_runs_failed_and_restores_context(settings):
    error = ValueError("invalid training data")
    with bound_contextvars(request_id="outer"):
        with pytest.raises(ValueError) as raised:
            with pipeline_run(run_config("failure"), settings) as parent:
                with pipeline_step("validate") as child:
                    raise error
        assert raised.value is error
        assert get_contextvars() == {"request_id": "outer"}

    client = MlflowClient(tracking_uri=settings.tracking_uri)
    assert client.get_run(parent.info.run_id).info.status == "FAILED"
    assert client.get_run(child.info.run_id).info.status == "FAILED"
    assert mlflow.active_run() is None
    events = read_events(run_directory(settings, parent.info.run_id) / "events.jsonl")
    failures = [event for event in events if event["event"].endswith("_failed")]
    assert {event["event"] for event in failures} == {"step_failed", "pipeline_failed"}
    assert all("ValueError: invalid training data" in event["exception"] for event in failures)
    assert client.list_artifacts(parent.info.run_id, "logs")


def test_telemetry_cleanup_cannot_replace_pipeline_error(settings, monkeypatch):
    def backend_unavailable(*args, **kwargs):
        raise ConnectionError("tracking backend unavailable")

    monkeypatch.setattr(mlflow, "log_metric", backend_unavailable)
    error = RuntimeError("training failed")
    with pytest.raises(RuntimeError) as raised:
        with pipeline_run(run_config("backend-failure"), settings) as parent:
            monkeypatch.setattr(mlflow, "log_artifact", backend_unavailable)
            with pipeline_step("train"):
                raise error
    assert raised.value is error
    events = read_events(run_directory(settings, parent.info.run_id) / "events.jsonl")
    assert any(event["event"] == "log_artifact_upload_failed" for event in events)


def test_step_requires_parent_run():
    with pytest.raises(RuntimeError, match="inside pipeline_run"):
        with pipeline_step("orphan"):
            pytest.fail("An orphan step must not execute")


def test_repeated_runs_share_config_hash_but_have_separate_artifacts(settings):
    config = run_config("repeat")
    with pipeline_run(config, settings) as first:
        pass
    with pipeline_run(config, settings) as second:
        pass
    first_tags = mlflow.get_run(first.info.run_id).data.tags
    second_tags = mlflow.get_run(second.info.run_id).data.tags
    assert first_tags["pipeline.config_hash"] == second_tags["pipeline.config_hash"]
    assert first_tags["pipeline.run_hash"] != second_tags["pipeline.run_hash"]
    assert len(list(settings.run_dir.iterdir())) == 2
    assert mlflow.active_run() is None


def test_backend_startup_failure_leaves_local_config_and_failed_manifest(settings, monkeypatch):
    def unavailable(*args, **kwargs):
        raise ConnectionError("backend unavailable")

    monkeypatch.setattr(mlflow, "set_experiment", unavailable)
    config = run_config("startup")
    with pytest.raises(ConnectionError):
        with pipeline_run(config, settings):
            pytest.fail("Pipeline must not execute without tracking")
    directory, = settings.run_dir.iterdir()
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == "FAILED"
    assert manifest["mlflow_run_id"] is None
    assert manifest["config_hash"] == config.config_hash
    assert RunConfig.load(directory / "config.json") == config


def test_logging_reconfiguration_does_not_duplicate_events(tmp_path, capsys):
    root_handlers = logging.getLogger().handlers[:]
    log_file = tmp_path / "events.jsonl"
    try:
        configure_logging(log_file, log_format="json")
        configure_logging(log_file, log_format="json")
        get_logger("test").info("once")
        logging.getLogger("aipi_pipeline.legacy").warning("legacy event")
        flush_logs()
    finally:
        close_logging()
    assert [event["event"] for event in read_events(log_file)] == ["once", "legacy event"]
    assert len(capsys.readouterr().err.splitlines()) == 2
    assert logging.getLogger().handlers == root_handlers


def test_demo_exports_loadable_model_and_all_six_steps(settings):
    run_id = run_demo(RunConfig.load(Path(__file__).resolve().parents[1] / "configs/iris-logistic.json"), settings)
    client = MlflowClient(tracking_uri=settings.tracking_uri)
    parent = client.get_run(run_id)
    assert parent.info.status == "FINISHED"
    assert 0 <= parent.data.metrics["accuracy"] <= 1
    assert 0 <= parent.data.metrics["f1_macro"] <= 1
    children = client.search_runs(
        [parent.info.experiment_id], filter_string=f"tags.mlflow.parentRunId = '{run_id}'",
    )
    assert {run.data.tags["pipeline.step"] for run in children} == {
        "ingest", "validate", "split", "train", "evaluate", "export",
    }
    assert all(run.info.status == "FINISHED" for run in children)
    model = mlflow.sklearn.load_model(parent.data.tags["pipeline.model_uri"])
    sample = load_iris(as_frame=True).data.head(3)
    assert len(model.predict(sample)) == len(sample)


def test_configuration_reads_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("LOG_LEVEL", "debug")
    monkeypatch.setenv("LOG_FORMAT", "JSON")
    monkeypatch.setenv("PIPELINE_RUN_DIR", str(tmp_path))
    assert Settings.from_env().log_level == "DEBUG"
    assert Settings.from_env().log_format == "json"
    assert Settings.from_env().run_dir == tmp_path


@pytest.mark.parametrize("values", [
    {"log_level": "LOUD"}, {"log_format": "xml"},
    {"tracking_uri": ""}, {"experiment_name": " "},
])
def test_invalid_configuration_is_rejected(values):
    with pytest.raises(ValueError):
        Settings(**values)
