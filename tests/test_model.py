"""Regression checks for reproducibility, leakage, scoring and persistence."""

import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import adjusted_rand_score

from data import load_data
from model import evaluate_model, main, train_model


class ModelingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.report = train_model()
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = load_data(scale=False)

    def test_scaler_fits_only_training_features(self):
        scaler = self.model.named_steps["scaler"]
        np.testing.assert_allclose(scaler.mean_, self.X_train.mean().to_numpy())
        self.assertEqual(scaler.n_samples_seen_, len(self.X_train))
        self.assertNotIn("target", self.model.feature_names_in_)
        self.assertEqual(self.model.n_features_in_, 13)
        self.assertFalse(set(self.report["split"]["train_indices"]) &
                         set(self.report["split"]["test_indices"]))

    def test_same_seed_reproduces_model_and_report(self):
        repeated, report = train_model()
        np.testing.assert_allclose(
            self.model.named_steps["kmeans"].cluster_centers_,
            repeated.named_steps["kmeans"].cluster_centers_,
        )
        np.testing.assert_array_equal(self.model.predict(self.X_test),
                                      repeated.predict(self.X_test))
        self.assertEqual(self.report, report)

    def test_evaluation_uses_holdout_without_refitting(self):
        centers = self.model.named_steps["kmeans"].cluster_centers_.copy()
        means = self.model.named_steps["scaler"].mean_.copy()
        metrics = evaluate_model(self.model, self.X_test, self.y_test)
        expected = adjusted_rand_score(self.y_test, self.model.predict(self.X_test))
        self.assertAlmostEqual(metrics["adjusted_rand_score"], expected)
        self.assertEqual(sum(metrics["cluster_counts"].values()), len(self.X_test))
        np.testing.assert_array_equal(centers, self.model.named_steps["kmeans"].cluster_centers_)
        np.testing.assert_array_equal(means, self.model.named_steps["scaler"].mean_)

    def test_single_cluster_silhouette_is_null(self):
        _, report = train_model(n_clusters=1)
        self.assertIsNone(report["train"]["silhouette_score"])
        self.assertIsNone(report["test"]["silhouette_score"])
        json.dumps(report, allow_nan=False)

    def test_cli_artifacts_round_trip_with_raw_features(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            main(["--output-dir", str(output)])
            restored = joblib.load(output / "pipeline.joblib")
            np.testing.assert_array_equal(self.model.predict(self.X_test),
                                          restored.predict(self.X_test))
            report = json.loads((output / "metrics.json").read_text())
            self.assertEqual(report, self.report)


if __name__ == "__main__":
    unittest.main()
