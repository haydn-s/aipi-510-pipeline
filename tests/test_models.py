import unittest
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from aipi_pipeline.models import resolve_model
from aipi_pipeline.run_config import RunConfig

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


class ModelConfigTests(unittest.TestCase):
    def test_both_configs_control_training_and_record_defaults(self):
        X, y = load_iris(return_X_y=True, as_frame=True)
        for path in CONFIGS.glob("*.json"):
            with self.subTest(config=path.name):
                config, model = resolve_model(RunConfig.load(path))
                values = config.to_dict()
                X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, **values["split"])
                model.fit(X_train, y_train)
                self.assertEqual(len(model.predict(X_test)), len(y_test))
                estimator = model.steps[-1][1] if values["preprocessing"]["standardize"] else model
                self.assertEqual(values["model"]["params"], estimator.get_params(deep=False))
                self.assertEqual(estimator.random_state, values["split"]["random_state"])
                resolved_again, _ = resolve_model(config)
                self.assertEqual(resolved_again.config_hash, config.config_hash)

    def test_unknown_model_type_does_not_silently_select_default(self):
        with self.assertRaisesRegex(ValueError, "Unsupported model.type"):
            resolve_model(RunConfig({"model": {"type": "unknown"}}))

    def test_explicit_model_seed_overrides_split_seed(self):
        config, estimator = resolve_model(RunConfig({
            "model": {"type": "random_forest", "params": {"random_state": 7, "n_estimators": 3}},
            "preprocessing": {"standardize": False},
        }))
        self.assertEqual(estimator.random_state, 7)
        self.assertEqual(estimator.n_estimators, 3)
        self.assertEqual(config.to_dict()["split"]["random_state"], 42)


if __name__ == "__main__":
    unittest.main()
