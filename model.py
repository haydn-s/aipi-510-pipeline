"""Reproducible K-Means training on the sklearn wine dataset."""

import argparse
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data import load_data


def build_model(n_clusters=3, random_state=42):
    """Build an unfitted pipeline that accepts raw wine feature DataFrames."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=n_clusters, random_state=random_state,
                          n_init=10, algorithm="lloyd")),
    ])


def evaluate_model(model, X, y):
    """Evaluate without refitting; labels are used only for external scoring.

    Cluster IDs are arbitrary, so classification accuracy is inappropriate.
    Silhouette is undefined with one cluster or one cluster per sample.
    """
    clusters = model.predict(X)
    scaled = model.named_steps["scaler"].transform(X)
    n_labels = len(np.unique(clusters))
    silhouette = (float(silhouette_score(scaled, clusters))
                  if 1 < n_labels < len(X) else None)
    return {
        "adjusted_rand_score": float(adjusted_rand_score(y, clusters)),
        "silhouette_score": silhouette,
        "cluster_counts": {
            str(i): int(np.sum(clusters == i))
            for i in range(model.named_steps["kmeans"].n_clusters)
        },
    }


def train_model(n_clusters=3, random_state=42, test_size=0.2):
    """Return a fitted raw-input pipeline and a JSON-serializable run report."""
    X_train, X_test, y_train, y_test = load_data(
        test_size=test_size, random_state=random_state, scale=False
    )
    model = build_model(n_clusters=n_clusters, random_state=random_state)
    model.fit(X_train)
    report = {
        "dataset": "sklearn.datasets.load_wine",
        "config": {"n_clusters": n_clusters, "random_state": random_state,
                   "test_size": test_size, "n_init": 10, "algorithm": "lloyd"},
        "features": list(X_train.columns),
        "split": {"train_indices": X_train.index.tolist(),
                  "test_indices": X_test.index.tolist()},
        "versions": {"python": platform.python_version(),
                     "scikit-learn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__,
                     "joblib": joblib.__version__},
        "train": evaluate_model(model, X_train, y_train),
        "test": evaluate_model(model, X_test, y_test),
        "train_inertia": float(model.named_steps["kmeans"].inertia_),
        "n_iter": int(model.named_steps["kmeans"].n_iter_),
    }
    return model, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-clusters", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/model"))
    args = parser.parse_args(argv)
    if args.n_clusters < 1:
        parser.error("--n-clusters must be at least 1")
    if not 0 < args.test_size < 1:
        parser.error("--test-size must be between 0 and 1")
    model, report = train_model(args.n_clusters, args.random_state, args.test_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output_dir / "pipeline.joblib")
    report_json = json.dumps(report, indent=2, allow_nan=False) + "\n"
    (args.output_dir / "metrics.json").write_text(report_json, encoding="utf-8")
    print(json.dumps({"train": report["train"], "test": report["test"]}, indent=2))
    print(f"Saved model and run report to {args.output_dir}")


if __name__ == "__main__":
    main()
