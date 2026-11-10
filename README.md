# Wine ML pipeline

A small class project using the bundled scikit-learn wine dataset: 178 samples,
13 numeric features, and three known wine classes. No dataset download is needed.

## Run the modeling step

The modeling-only environment below pins the packages used for verification
(Python 3.10.0). The team's broader `requirements.txt` is maintained separately;
its different package versions were not used to validate this modeling step.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-model.txt
python model.py
python -m unittest discover -s tests -v
```

Configuration is explicit and can be changed from the command line:

```bash
python model.py --n-clusters 3 --random-state 42 --test-size 0.2 --output-dir artifacts/experiment-1
```

The default run creates `artifacts/model/pipeline.joblib` and
`artifacts/model/metrics.json`. Reusing an output directory overwrites its files;
use a new directory to retain an experiment. Generated artifacts are ignored by Git.

## Modeling design

- Reuse `data.load_data(scale=False)` for the existing cleaned, stratified 80/20
  split (142 training rows and 36 test rows by default).
- Fit a `StandardScaler` and then K-Means together in one sklearn `Pipeline`.
  Both fit only on training features. Scaling matters because wine features have
  very different units and K-Means uses Euclidean distances.
- Use three clusters to match the dataset's three known classes, a fixed seed of
  42, ten initializations, and the Lloyd algorithm. This is a fixed baseline,
  with no tuning on the held-out test data.
- Labels are used by the existing stratified split and for evaluation, never for
  fitting the scaler or K-Means. Cluster numbers are arbitrary, not class predictions.
- Report adjusted Rand index (ARI) for agreement with the known labels, silhouette
  for separation in scaled feature space, cluster counts, and training inertia.
  ARI is invariant to cluster numbering; ordinary classification accuracy is not
  appropriate. Silhouette is JSON `null` when fewer than two clusters are present
  or every sample has its own cluster.

The run report also records feature order, split row indices, configuration, and
runtime versions. Fixed seeds and pinned dependencies support repeatable runs;
floating-point results can still vary slightly across platforms.

The verified default run achieved test ARI **0.9088** and test silhouette
**0.2721**. These describe clustering performance, not classification accuracy.

## Integrate with other pipeline steps

```python
from model import train_model

pipeline, report = train_model(random_state=42)
print(report["test"])
```

For inference, load the entire saved pipeline and supply unscaled features with
the same columns and order used during training. Do not scale a second time or
pass the target column. Only load joblib files from trusted sources, using the
same dependency versions as training.

```python
import joblib
from data import load_data

pipeline = joblib.load("artifacts/model/pipeline.joblib")
_, X_test, _, _ = load_data(scale=False)
cluster_ids = pipeline.predict(X_test)
```

`build_model()` exposes an unfitted pipeline for other orchestration code, and
`evaluate_model(pipeline, X, y)` evaluates raw feature inputs without refitting.
Tests cover repeatability, training-only preprocessing, holdout scoring, undefined
silhouette handling, and artifact loading with identical predictions.
