"""An offline Iris example showing instrumentation across the entire pipeline."""

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from .config import Settings
from .logging import get_logger
from .models import resolve_model
from .run_config import RunConfig
from .tracking import pipeline_run, pipeline_step

log = get_logger("demo")


def run_demo(config: RunConfig, settings: Settings | None = None) -> str:
    config, model = resolve_model(config)
    values = config.to_dict()
    with pipeline_run(config, settings) as run:
        mlflow.log_params({f"model.{key}": value for key, value in values["model"]["params"].items()})

        with pipeline_step("ingest"):
            if values["data"].get("name") != "iris":
                raise ValueError("The example pipeline supports only data.name='iris'")
            data = load_iris(as_frame=True)
            X, y = data.data, data.target
            mlflow.log_params({"dataset": "iris", "rows": len(X), "features": X.shape[1]})
            log.info("data_loaded", rows=len(X), features=X.shape[1])

        with pipeline_step("validate"):
            if X.isna().any().any() or y.isna().any() or len(X) != len(y):
                raise ValueError("Features and targets must be aligned and contain no missing values")
            log.info("data_validated", classes=int(y.nunique()))

        with pipeline_step("split"):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, **values["split"], stratify=y,
            )
            log.info("data_split", train_rows=len(X_train), test_rows=len(X_test))

        with pipeline_step("train"):
            # Fit scaling only on training data, and persist it with the estimator.
            mlflow.log_params({"estimator": values["model"]["type"], **values["model"]["params"]})
            model.fit(X_train, y_train)
            log.info("model_trained")

        with pipeline_step("evaluate"):
            predictions = model.predict(X_test)
            metrics = {
                "accuracy": float(accuracy_score(y_test, predictions)),
                "f1_macro": float(f1_score(y_test, predictions, average="macro")),
            }
            mlflow.log_metrics(metrics)
            mlflow.log_dict(
                classification_report(y_test, predictions, output_dict=True),
                "classification_report.json",
            )
            log.info("model_evaluated", **metrics)
        # Keep headline metrics on the parent run for experiment comparisons.
        mlflow.log_metrics(metrics)

        with pipeline_step("export"):
            model_info = mlflow.sklearn.log_model(
                model,
                name="model",
                signature=infer_signature(X_train, model.predict(X_train)),
                input_example=X_train.head(3),
            )
            log.info("model_saved", model_uri=model_info.model_uri)
        mlflow.set_tag("pipeline.model_uri", model_info.model_uri)
        return run.info.run_id
