import argparse
import json
import urllib.request

import numpy as np
import pandas as pd

from scipy.stats import ks_2samp
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
)

from common import (
    DATA,
    REPORTS,
    MODELS,
    FEATURES,
    NUM_FEATURES,
    CAT_FEATURES,
    TARGET,
    init_mlflow,
    load_split,
)

from fraudsim import generate

from drift_utils import (
    psi_numeric,
    psi_categorical,
)


parser = argparse.ArgumentParser()

parser.add_argument(
    "--batch",
    choices=["control", "drift"],
    required=True,
)

args = parser.parse_args()

name = args.batch

batch_path = DATA / f"batch_{name}.csv"

if not batch_path.exists():
    batch = generate(
        n=5000,
        seed=7 if name == "drift" else 8,
        drift=(name == "drift"),
    )

    batch.to_csv(batch_path, index=False)

batch = pd.read_csv(batch_path)

X_ref, _ = load_split("train")

X_batch = batch[FEATURES]
y_batch = batch[TARGET].astype(int)

meta = json.loads(
    (MODELS / "model_meta.json").read_text()
)

base_url = "http://127.0.0.1:8000"

with urllib.request.urlopen(
    f"{base_url}/metadata", timeout=30
) as response:
    api_meta = json.load(response)

assert str(api_meta["model_version"]) == str(
    meta["model_version"]
)

assert api_meta["run_id"] == meta["run_id"]

print("Batch:", name)
print("Rows:", len(batch))
print("Fraud rate:", float(y_batch.mean()))
print("Model version:", meta["model_version"])

drift = {}

print("\n========== NUMERIC FEATURE DRIFT ==========")

for feature in NUM_FEATURES:
    ref = X_ref[feature]
    cur = X_batch[feature]

    value = psi_numeric(ref, cur)

    ks = ks_2samp(
        ref.dropna(),
        cur.dropna(),
    )

    drift[feature] = {
        "psi": float(value),
        "ks_pvalue": float(ks.pvalue),
    }

    print(
        f"{feature}: PSI={value:.6f}, "
        f"KS p-value={ks.pvalue:.6g}"
    )

print("\n========== CATEGORICAL FEATURE DRIFT ==========")

for feature in CAT_FEATURES:
    value = psi_categorical(
        X_ref[feature],
        X_batch[feature],
    )

    drift[feature] = {
        "psi": float(value),
        "ks_pvalue": None,
    }

    print(f"{feature}: PSI={value:.6f}")

print("\n========== SCORE BATCH THROUGH API ==========")

probabilities = []
predictions = []

for start in range(0, len(X_batch), 500):
    chunk = X_batch.iloc[start:start + 500]

    records = (
        chunk.astype(object)
        .where(chunk.notna(), None)
        .to_dict(orient="records")
    )

    payload = json.dumps(
        {"transactions": records},
        default=lambda obj: obj.item(),
        allow_nan=False,
    ).encode("utf-8")

    request = urllib.request.Request(
        f"{base_url}/predict",
        data=payload,
        headers={
            "Content-Type": "application/json"
        },
        method="POST",
    )

    with urllib.request.urlopen(
        request, timeout=60
    ) as response:
        result = json.load(response)

    assert str(result["model_version"]) == str(
        meta["model_version"]
    )

    probabilities.extend(
        row["fraud_probability"]
        for row in result["predictions"]
    )

    predictions.extend(
        int(row["is_fraud"])
        for row in result["predictions"]
    )

    print(
        f"Scored {len(probabilities)}/{len(X_batch)}",
        flush=True,
    )

proba = np.asarray(probabilities)
pred = np.asarray(predictions)

assert len(proba) == len(y_batch)
assert np.isfinite(proba).all()

pr_auc = float(
    average_precision_score(y_batch, proba)
)

reference_pr_auc = float(meta["test_pr_auc"])

pr_auc_drop = reference_pr_auc - pr_auc

recall = float(
    recall_score(y_batch, pred, zero_division=0)
)

precision = float(
    precision_score(y_batch, pred, zero_division=0)
)

reasons = []

for feature, metrics in drift.items():
    if metrics["psi"] >= 0.25:
        reasons.append(
            f"{feature}: PSI={metrics['psi']:.4f} >= 0.25"
        )

if pr_auc_drop >= 0.15:
    reasons.append(
        f"PR-AUC drop={pr_auc_drop:.4f} >= 0.15"
    )

decision = "RETRAIN" if reasons else "OK"

report = {
    "batch": name,
    "rows": len(batch),
    "fraud_rate": float(y_batch.mean()),
    "drift": drift,
    "pr_auc": pr_auc,
    "reference_pr_auc": reference_pr_auc,
    "pr_auc_drop": pr_auc_drop,
    "recall": recall,
    "precision": precision,
    "decision": decision,
    "reasons": reasons,
    "model_version": str(meta["model_version"]),
}

REPORTS.mkdir(parents=True, exist_ok=True)

report_path = REPORTS / f"drift_{name}.json"

report_path.write_text(
    json.dumps(report, indent=2)
)

mlflow = init_mlflow()

with mlflow.start_run(
    run_name=f"monitor_{name}"
):
    mlflow.set_tags({
        "task": "monitoring",
        "batch": name,
        "decision": decision,
    })

    mlflow.log_param(
        "model_version",
        meta["model_version"],
    )

    mlflow.log_metrics({
        **{
            f"psi_{feature}": metrics["psi"]
            for feature, metrics in drift.items()
        },
        "batch_pr_auc": pr_auc,
        "pr_auc_drop": pr_auc_drop,
        "batch_recall": recall,
        "batch_precision": precision,
    })

    mlflow.log_artifact(str(report_path))

print("\n========== MONITORING RESULTS ==========")
print(json.dumps(report, indent=2))
print("\nDecision:", decision)
print("PASS: Monitoring report saved")
print("PASS: MLflow monitoring run logged")
