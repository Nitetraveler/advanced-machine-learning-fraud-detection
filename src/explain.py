import json

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.base import clone
from sklearn.calibration import (
    CalibratedClassifierCV,
    calibration_curve,
)
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
)

from common import (
    SEED,
    FEATURES,
    MODEL_NAME,
    REPORTS,
    init_mlflow,
    load_split,
)

from explain_utils import aggregate_contributions


mlflow = init_mlflow()

client = mlflow.tracking.MlflowClient()

champion = client.get_model_version_by_alias(
    MODEL_NAME,
    "champion",
)

pipe = mlflow.sklearn.load_model(
    f"models:/{MODEL_NAME}@champion"
)

X_train, y_train = load_split("train")
X_test, y_test = load_split("test")

raw_proba = pipe.predict_proba(X_test)[:, 1]

print("Champion version:", champion.version)
print("Champion run ID:", champion.run_id)

print("\n========== PERMUTATION IMPORTANCE ==========")

permutation = permutation_importance(
    pipe,
    X_test,
    y_test,
    scoring="average_precision",
    n_repeats=5,
    random_state=SEED,
    n_jobs=1,
)

perm_scores = dict(sorted(
    {
        name: float(value)
        for name, value in zip(
            FEATURES,
            permutation.importances_mean,
        )
    }.items(),
    key=lambda item: item[1],
    reverse=True,
))

for name, score in perm_scores.items():
    print(f"{name}: {score:.6f}")

print("\n========== NATIVE XGBOOST SHAP ==========")

prep = pipe.named_steps["prep"]
model = pipe.named_steps["model"]

X_transformed = prep.transform(X_test)
feature_names = prep.get_feature_names_out()

booster = model.get_booster()

matrix = xgb.DMatrix(X_transformed)

contributions = booster.predict(
    matrix,
    pred_contribs=True,
)

margin = booster.predict(
    matrix,
    output_margin=True,
)

assert contributions.shape[1] == len(feature_names) + 1

additivity_ok = bool(np.allclose(
    contributions.sum(axis=1),
    margin,
    atol=1e-3,
))

max_error = float(np.max(np.abs(
    contributions.sum(axis=1) - margin
)))

print("Contribution matrix shape:", contributions.shape)
print("Maximum additivity error:", max_error)
print("SHAP additivity check:", additivity_ok)

assert additivity_ok

print("PASS: SHAP contributions reconstruct model margins")

shap_global = aggregate_contributions(
    contributions[:, :-1],
    feature_names,
)

print("\n========== GLOBAL SHAP IMPORTANCE ==========")

for name, score in shap_global.items():
    print(f"{name}: {score:.6f}")

print("\n========== LOCAL HIGH-RISK EXPLANATION ==========")

idx = int(np.argmax(raw_proba))

local_values = contributions[idx, :-1]

top_indices = np.argsort(
    np.abs(local_values)
)[::-1][:6]

local_explanation = [
    {
        "feature": str(feature_names[i]),
        "contribution": float(local_values[i]),
    }
    for i in top_indices
]

print("Test row:", idx)
print("Fraud probability:", float(raw_proba[idx]))
print("Actual label:", int(y_test.iloc[idx]))
print("Bias contribution:", float(contributions[idx, -1]))
print("Raw model margin:", float(margin[idx]))

for item in local_explanation:
    print(
        f"{item['feature']}: "
        f"{item['contribution']:+.6f}"
    )

print("\n========== ISOTONIC CALIBRATION ==========")

calibrated = CalibratedClassifierCV(
    estimator=clone(pipe),
    method="isotonic",
    cv=3,
    n_jobs=1,
)

calibrated.fit(X_train, y_train)

cal_proba = calibrated.predict_proba(X_test)[:, 1]

brier_raw = float(
    brier_score_loss(y_test, raw_proba)
)

brier_calibrated = float(
    brier_score_loss(y_test, cal_proba)
)

pr_auc_raw = float(
    average_precision_score(y_test, raw_proba)
)

pr_auc_calibrated = float(
    average_precision_score(y_test, cal_proba)
)

raw_observed, raw_predicted = calibration_curve(
    y_test,
    raw_proba,
    n_bins=5,
    strategy="quantile",
)

cal_observed, cal_predicted = calibration_curve(
    y_test,
    cal_proba,
    n_bins=5,
    strategy="quantile",
)

print("Brier score raw:", brier_raw)
print("Brier score calibrated:", brier_calibrated)
print("PR-AUC raw:", pr_auc_raw)
print("PR-AUC calibrated:", pr_auc_calibrated)
print("Mean raw probability:", float(raw_proba.mean()))
print("Mean calibrated probability:", float(cal_proba.mean()))
print("Observed fraud rate:", float(y_test.mean()))

print("\n========== RAW CALIBRATION CURVE ==========")

print(pd.DataFrame({
    "mean_predicted": raw_predicted,
    "observed_fraction": raw_observed,
}).to_string(index=False))

print("\n========== ISOTONIC CALIBRATION CURVE ==========")

print(pd.DataFrame({
    "mean_predicted": cal_predicted,
    "observed_fraction": cal_observed,
}).to_string(index=False))

report = {
    "model_version": str(champion.version),
    "run_id": champion.run_id,
    "permutation_importance": perm_scores,
    "shap_global": shap_global,
    "additivity_check": additivity_ok,
    "additivity_max_error": max_error,
    "brier_raw": brier_raw,
    "brier_calibrated": brier_calibrated,
    "mean_pred_raw": float(raw_proba.mean()),
    "mean_pred_calibrated": float(cal_proba.mean()),
    "observed_rate": float(y_test.mean()),
    "pr_auc_raw": pr_auc_raw,
    "pr_auc_calibrated": pr_auc_calibrated,
    "local_explanation": {
        "test_row": idx,
        "fraud_probability": float(raw_proba[idx]),
        "actual_label": int(y_test.iloc[idx]),
        "bias": float(contributions[idx, -1]),
        "margin": float(margin[idx]),
        "top_contributions": local_explanation,
    },
    "calibration_curve_raw": {
        "mean_predicted": raw_predicted.tolist(),
        "observed_fraction": raw_observed.tolist(),
    },
    "calibration_curve_calibrated": {
        "mean_predicted": cal_predicted.tolist(),
        "observed_fraction": cal_observed.tolist(),
    },
}

REPORTS.mkdir(parents=True, exist_ok=True)

report_path = REPORTS / "explain_report.json"

report_path.write_text(
    json.dumps(report, indent=2)
)

with mlflow.start_run(
    run_name="champion_explain_calibration"
):
    mlflow.set_tag("task", "explain")

    mlflow.log_param(
        "champion_version",
        champion.version,
    )

    mlflow.log_metrics({
        "brier_raw": brier_raw,
        "brier_calibrated": brier_calibrated,
        "additivity_ok": float(additivity_ok),
        "pr_auc_raw": pr_auc_raw,
        "pr_auc_calibrated": pr_auc_calibrated,
    })

    mlflow.log_artifact(str(report_path))

print("\nPASS: Permutation importance")
print("PASS: Global SHAP explanation")
print("PASS: Local SHAP explanation")
print("PASS: SHAP additivity verification")
print("PASS: Probability calibration")
print("PASS: Explanation report saved")
print("PASS: MLflow explanation run logged")
