"""Sequential pipeline and step contexts combining MLflow with structured logs."""

from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter

import mlflow
from structlog.contextvars import bound_contextvars, get_contextvars

from .config import Settings
from .logging import close_logging, configure_logging, flush_logs, get_logger
from .run_config import RunConfig
from .run_record import RunRecord

log = get_logger("tracking")


@contextmanager
def _timed_scope(kind: str) -> Iterator[None]:
    started = perf_counter()
    log.info(f"{kind}_started")
    try:
        yield
    except BaseException:
        log.exception(f"{kind}_failed", duration_seconds=perf_counter() - started)
        raise
    else:
        log.info(f"{kind}_completed", duration_seconds=perf_counter() - started)
    finally:
        # Telemetry cleanup must not hide an exception raised by the pipeline.
        try:
            mlflow.log_metric("duration_seconds", perf_counter() - started)
        except Exception:
            log.exception("duration_tracking_failed")


@contextmanager
def pipeline_run(config: RunConfig, settings: Settings | None = None) -> Iterator[mlflow.ActiveRun]:
    """Save a config snapshot and uniquely identify each pipeline invocation."""
    if mlflow.active_run() is not None:
        raise RuntimeError("Finish the active MLflow run before starting a pipeline")
    settings = settings or Settings.from_env()
    record = RunRecord.create(config, settings.run_dir)
    values = config.to_dict()
    tags = {
        "pipeline.name": values["name"],
        "pipeline.run_hash": record.run_hash,
        "pipeline.config_hash": config.config_hash,
        "pipeline.model_type": values["model"]["type"],
    }
    run_id = None
    try:
        configure_logging(record.log_file, settings.log_level, settings.log_format)
        with bound_contextvars(
            pipeline=values["name"], run_hash=record.run_hash,
            config_hash=config.config_hash, step="pipeline",
        ):
            try:
                mlflow.set_tracking_uri(settings.tracking_uri)
                mlflow.set_experiment(settings.experiment_name)
                with mlflow.start_run(
                    run_name=f"{values['name']}-{record.run_hash[:12]}", tags=tags,
                ) as run:
                    run_id = run.info.run_id
                    record.update(status="RUNNING", mlflow_run_id=run_id)
                    with bound_contextvars(run_id=run_id):
                        with _timed_scope("pipeline"):
                            mlflow.log_artifact(str(record.directory / "config.json"), "config")
                            mlflow.log_params({
                                "model.type": values["model"]["type"],
                                **{f"split.{key}": value for key, value in values["split"].items()},
                            })
                            yield run
            except BaseException as error:
                try:
                    record.update(status="FAILED", error_type=type(error).__name__)
                except OSError:
                    log.exception("manifest_update_failed")
                if run_id is None:
                    log.exception("pipeline_start_failed")
                raise
            else:
                record.update(status="FINISHED")
            finally:
                if run_id is not None:
                    # Explicit IDs prevent artifact uploads from starting an implicit run.
                    # Upload after MLflow closes so the saved manifest has final status.
                    flush_logs()
                    with bound_contextvars(run_id=run_id):
                        for path, destination in (
                            (record.directory / "manifest.json", "config"),
                            (record.log_file, "logs"),
                        ):
                            try:
                                mlflow.log_artifact(str(path), artifact_path=destination, run_id=run_id)
                            except Exception:
                                log.exception("log_artifact_upload_failed", artifact=str(path))
    finally:
        close_logging()


@contextmanager
def pipeline_step(name: str) -> Iterator[mlflow.ActiveRun]:
    """Time a step, record its status, and bind its run ID to every step log."""
    parent = mlflow.active_run()
    if parent is None or "pipeline" not in get_contextvars():
        raise RuntimeError("pipeline_step must be used inside pipeline_run")
    with mlflow.start_run(
        run_name=name,
        nested=True,
        tags={
            "pipeline.step": name,
            "pipeline.run_hash": get_contextvars()["run_hash"],
            "pipeline.config_hash": get_contextvars()["config_hash"],
        },
    ) as run:
        with bound_contextvars(
            step=name, run_id=run.info.run_id, parent_run_id=parent.info.run_id,
        ):
            with _timed_scope("step"):
                yield run
