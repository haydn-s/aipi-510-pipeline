"""Run with: python -m aipi_pipeline --config configs/iris-logistic.json."""

import argparse

from .run_config import RunConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a configured ML pipeline")
    parser.add_argument("--config", required=True, help="Path to a JSON run configuration")
    arguments = parser.parse_args()
    try:
        config = RunConfig.load(arguments.config)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    from .demo import run_demo
    run_demo(config)

if __name__ == "__main__":
    main()
