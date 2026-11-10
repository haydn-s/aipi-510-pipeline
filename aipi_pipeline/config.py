"""Environment-based configuration, loaded when a pipeline starts."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "aipi-510-pipeline"
    run_dir: Path = Path("runs")
    log_level: str = "INFO"
    log_format: str = "console"

    def __post_init__(self) -> None:
        if self.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("LOG_LEVEL must be DEBUG, INFO, WARNING, ERROR, or CRITICAL")
        if self.log_format not in {"console", "json"}:
            raise ValueError("LOG_FORMAT must be console or json")
        if not self.tracking_uri.strip() or not self.experiment_name.strip():
            raise ValueError("Tracking URI and experiment name cannot be empty")

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
            experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "aipi-510-pipeline"),
            run_dir=Path(os.getenv("PIPELINE_RUN_DIR", "runs")),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            log_format=os.getenv("LOG_FORMAT", "console").lower(),
        )
