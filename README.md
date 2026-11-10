# AIPI 510 ML pipeline

MLflow tracks experiments, parameters, metrics, artifacts, and models. `structlog`
adds readable console logs and a JSON-lines file for each run. Shared contexts
instrument ingestion, validation, splitting, training, evaluation, and export in
the included example; use the same contexts for additional pipeline steps.

## Run locally

From the repository root, using the existing virtual environment:

```bash
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m aipi_pipeline --config configs/iris-logistic.json
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

For a uv-managed environment without pip, use `uv pip install -r requirements-dev.txt`.
The example uses scikit-learn's bundled Iris data and requires no dataset download.
Open <http://127.0.0.1:5000> and select the `aipi-510-pipeline` experiment.
Expand the parent run to see each step's status, duration, and logged outputs.
The parent run includes headline evaluation metrics and a `pipeline.model_uri`
tag pointing to the exported model, including its fitted scaler.

## Run configurations and hashes

Copy a JSON file in `configs/` for each experiment. Set `name`, `model.type`,
`model.params`, `data`, `split`, `preprocessing`, and arbitrary JSON `metadata`
(for example owner, description, dataset version, or labels). The example supports
`logistic_regression` and `random_forest`. Unsupported model types and malformed
configs raise errors instead of silently falling back to a default.

```bash
python -m aipi_pipeline --config configs/iris-random-forest.json
```

The example constructs the requested estimator and expands its default parameters
before starting the tracked run. The saved config contains the effective model
parameters, split seed, test fraction, preprocessing settings, and metadata.
Scaling, when enabled, is fitted only on training data and saved with the model.
The example currently loads only Iris; `data` records its description/source.
Additional pipelines can use `RunConfig` and `pipeline_run` with their own models
and data-loading code.

Each execution produces:

```text
runs/<run_hash>/
  config.json      # Immutable snapshot of the resolved configuration
  manifest.json    # Both hashes, MLflow ID, timestamps, and execution status
  events.jsonl     # Logs carrying both hashes and step/run IDs
```

- `config_hash` is SHA-256 of the saved config's UTF-8 bytes. JSON keys are sorted,
  whitespace is removed, and defaults are filled in. Reordering JSON keys or
  changing formatting does not change this hash. Changing a model parameter,
  metadata, or another resolved value does. Numbers must be finite; duplicate
  JSON keys and unknown schema fields are rejected.
- `run_hash` is SHA-256 of canonical JSON containing `schema_version`,
  `config_hash`, a UTC `created_at`, and a random UUID `nonce`. These fields are
  retained in the manifest so the hash can be verified. Running the same config
  twice creates separate directories with different run hashes and the same
  config hash. Prior runs are never overwritten.

MLflow parent and step runs carry `pipeline.run_hash` and `pipeline.config_hash`
tags. The parent also records `pipeline.model_type` and saves `config/config.json`,
`config/manifest.json`, and `logs/events.jsonl` artifacts. Search by either tag to
find one execution or compare executions with identical resolved configurations.
MLflow retains its own native run ID, which is also stored in the manifest.

The local config is written before tracking startup, so a tracking connection
failure still leaves a config and a `FAILED` manifest. Unhandled pipeline errors
also mark the manifest as failed. A hard process termination can leave `RUNNING`
or `CREATED` behind. The config hash identifies settings; it does not hash the
dataset contents, source code, or dependency environment. Populate data/version
metadata when you need to distinguish those inputs.

## Runtime settings

Set environment variables before running. `.env` files are not loaded automatically.

| Variable | Default | Purpose |
| --- | --- | --- |
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | Local SQLite database or HTTP tracking server |
| `MLFLOW_EXPERIMENT_NAME` | `aipi-510-pipeline` | Experiment grouping |
| `PIPELINE_RUN_DIR` | `runs` | Local directories named by each execution's run hash |
| `LOG_LEVEL` | `INFO` | DEBUG, INFO, WARNING, ERROR, or CRITICAL |
| `LOG_FORMAT` | `console` | `console` or `json` for stderr output |

Paths are relative to the working directory. Run commands from the repository
root so the pipeline and UI use the same database and artifact locations.
For an existing remote tracking server, set `MLFLOW_TRACKING_URI` to its URL and
use that server's UI. Local database files, artifacts, logs, and environments
are excluded from Git.

## Instrument another pipeline

```python
import mlflow
from aipi_pipeline.logging import get_logger
from aipi_pipeline.run_config import RunConfig
from aipi_pipeline.tracking import pipeline_run, pipeline_step

log = get_logger("training")

config = RunConfig({
    "name": "my-pipeline",
    "model": {"type": "custom_model", "params": {"seed": 42}},
    "metadata": {"owner": "researcher", "description": "Initial baseline"},
})

with pipeline_run(config):
    with pipeline_step("prepare"):
        log.info("preparation_started", source="training-data")
        # Prepare data here.
    with pipeline_step("train"):
        # Fit your model and log its metrics or artifacts here.
        log.info("training_completed")
```

Each step is a nested MLflow run. Logs include UTC time, severity, pipeline name,
step name, both hashes, and MLflow run IDs. Unhandled exceptions produce tracebacks and mark
both the step and its parent run as failed. The local JSON log is uploaded to the
parent's `logs/` artifacts on success or failure. Duration recording and final
manifest/log uploads are best effort: failures are logged locally and do not
replace an exception from pipeline code. The initial config upload is required;
if it fails, the pipeline body does not execute and its local record is retained.

The helpers configure the `aipi_pipeline` logger namespace. Other application
logging handlers are preserved. `LOG_LEVEL=WARNING` or higher suppresses normal
start/completion events; use the default `INFO` for full step logs.
This initial implementation supports one sequential pipeline per process.
Use separate processes for parallel pipelines; the helpers configure process-wide
MLflow and structlog state.

## Tests

```bash
python -m pytest -q
```

Tests use temporary SQLite stores and log directories. They cover run lifecycle,
failure propagation, log correlation, repeated logger setup, model loading,
configuration validation, hash stability, snapshot integrity, and distinct
execution records for repeated configurations.

The config/persistence tests have no third-party dependencies. The model tests
need only the existing scikit-learn environment:

```bash
python -m unittest discover -s tests -p test_run_config.py -v
python -m unittest discover -s tests -p test_models.py -v
```

Implementation references: [MLflow tracking API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html),
[SQLite backend](https://mlflow.org/docs/latest/tracking/backend-stores/), and
[structlog standard-library integration](https://www.structlog.org/en/stable/standard-library.html).
