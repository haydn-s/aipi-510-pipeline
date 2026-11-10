"""Resolve supported example estimators and record their effective defaults."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .run_config import RunConfig

MODEL_TYPES = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
}


def resolve_model(config: RunConfig) -> tuple[RunConfig, object]:
    values = config.to_dict()
    model_type = values["model"]["type"]
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unsupported model.type {model_type!r}; choose from {sorted(MODEL_TYPES)}")
    parameters = {"random_state": values["split"]["random_state"], **values["model"]["params"]}
    estimator = MODEL_TYPES[model_type](**parameters)
    values["model"]["params"] = estimator.get_params(deep=False)
    model = make_pipeline(StandardScaler(), estimator) if values["preprocessing"]["standardize"] else estimator
    return RunConfig(values), model
