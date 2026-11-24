import json

import numpy as np
import optuna
import pandas as pd

from sklearn.metrics import average_precision_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from xgboost import XGBClassifier

from common import (
    LAB,
    DATA,
    REPORTS,
    FEATURES,
    TARGET,
    SEED,
    MODEL_NAME,
    init_mlflow,
    load_split,
    make_pipeline,
)

from costs import expected_cost, best_threshold


mlflow = init_mlflow()
client = mlflow.tracking.MlflowClient()

champion = client.get_model_version_by_alias(
    MODEL_NAME, "champion"
)

champion_run = client.get_run(champion.run_id)

champion_threshold = float(
    champion_run.data.metrics["decision_threshold"]
)

champion_model = mlflow.sklearn.load_model(
    f"models:/{MODEL_NAME}/{champion.version}"
)

X_train, y_train = load_split("train")
X_test, y_test = load_split("test")

drift = pd.read_csv(DATA / "batch_drift.csv")

drift_train, drift_holdout = train_test_split(
    drift,
    test_size=0.5,
    random_state=SEED,
    stratify=drift[TARGET],
)

assert len(drift_train) == 2500
assert len(drift_holdout) == 2500

assert set(drift_train["transaction_id"]).isdisjoint(
    drift_holdout["transaction_id"]
)

X_holdout = drift_holdout[FEATURES]
y_holdout = drift_holdout[TARGET].astype(int)

X_combined = pd.concat(
    [X_train, drift_train[FEATURES]],
    ignore_index=True,
)

y_combined = pd.concat(
    [y_train, drift_train[TARGET].astype(int)],
    ignore_index=True,
)

assert len(X_combined) == 26500

print("========== RETRAINING DATA ==========")
print("Original training rows:", len(X_train))
print("Drift training rows:", len(drift_train))
print("Combined training rows:", len(X_combined))
print("Independent drift holdout:", len(X_holdout))
print("Original test rows:", len(X_test))
print("PASS: Drift training and holdout are disjoint")

study = optuna.load_study(
    study_name="xgb-fraud",
    storage=f"sqlite:///{LAB / 'optuna.db'}",
)

params = study.best_params

challenger = make_pipeline(
    XGBClassifier(
        **params,
        tree_method="hist",
        eval_metric="aucpr",
        n_jobs=2,
        random_state=SEED,
    ),
    scale=False,
)

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=SEED,
)

print("\n========== CHALLENGER OOF THRESHOLD ==========")

oof_proba = cross_val_predict(
    challenger,
    X_combined,
    y_combined,
    cv=cv,
    method="predict_proba",
    n_jobs=1,
)[:, 1]

challenger_threshold, oof_cost = best_threshold(
    y_combined,
    oof_proba,
)

print("Champion threshold:", champion_threshold)
print("Challenger threshold:", challenger_threshold)
print("Challenger OOF cost:", oof_cost)

print("\n========== TRAIN CHALLENGER ==========")

challenger.fit(X_combined, y_combined)


def evaluate(model, X, y, threshold):
    proba = model.predict_proba(X)[:, 1]

    return {
        "pr_auc": float(
            average_precision_score(y, proba)
        ),
        "cost": float(
            expected_cost(y, proba, threshold)
        ),
    }


champion_holdout = evaluate(
    champion_model,
    X_holdout,
    y_holdout,
    champion_threshold,
)

challenger_holdout = evaluate(
    challenger,
    X_holdout,
    y_holdout,
    challenger_threshold,
)

champion_test = evaluate(
    champion_model,
    X_test,
    y_test,
    champion_threshold,
)

challenger_test = evaluate(
    challenger,
    X_test,
    y_test,
    challenger_threshold,
)

old_cost = champion_holdout["cost"]
new_cost = challenger_holdout["cost"]

cost_reduction = (
    (old_cost - new_cost) / old_cost
    if old_cost > 0 else 0.0
)

pr_auc_drop = (
    champion_test["pr_auc"]
    - challenger_test["pr_auc"]
)

cost_gate = bool(
    old_cost > 0 and cost_reduction >= 0.05
)

pr_auc_gate = bool(pr_auc_drop <= 0.02)

promote = cost_gate and pr_auc_gate

print("\n========== PROMOTION EVALUATION ==========")
print("Champion drift holdout:", champion_holdout)
print("Challenger drift holdout:", challenger_holdout)
print("Champion original test:", champion_test)
print("Challenger original test:", challenger_test)
print(f"Holdout cost reduction: {cost_reduction:.2%}")
print(f"Original test PR-AUC drop: {pr_auc_drop:.6f}")
print("Cost gate:", cost_gate)
print("PR-AUC gate:", pr_auc_gate)

metrics = {
    "decision_threshold": float(challenger_threshold),
    "test_pr_auc": challenger_test["pr_auc"],
    "test_cost": challenger_test["cost"],
    "drift_holdout_pr_auc": challenger_holdout["pr_auc"],
    "drift_holdout_cost": challenger_holdout["cost"],
    "champion_drift_holdout_pr_auc": champion_holdout["pr_auc"],
    "champion_drift_holdout_cost": champion_holdout["cost"],
    "champion_test_pr_auc": champion_test["pr_auc"],
    "champion_test_cost": champion_test["cost"],
    "holdout_cost_reduction": float(cost_reduction),
    "original_test_pr_auc_drop": float(pr_auc_drop),
    "oof_best_cost": float(oof_cost),
    "cost_gate": float(cost_gate),
    "pr_auc_gate": float(pr_auc_gate),
}

print("\n========== REGISTER CHALLENGER ==========")

with mlflow.start_run(run_name="xgb_drift_challenger") as run:
    run_id = run.info.run_id

    mlflow.set_tags({
        "task": "retrain",
        "promotion_decision": (
            "PROMOTE" if promote else "RETAIN"
        ),
    })

    mlflow.log_params({
        **params,
        "original_train_rows": len(X_train),
        "drift_train_rows": len(drift_train),
        "drift_holdout_rows": len(drift_holdout),
        "combined_train_rows": len(X_combined),
        "previous_champion_version": champion.version,
    })

    mlflow.log_metrics(metrics)

    mlflow.sklearn.log_model(
        challenger,
        artifact_path="model",
        registered_model_name=MODEL_NAME,
        input_example=X_combined.dropna().head(5),
    )

versions = [
    v for v in client.search_model_versions(
        f"name='{MODEL_NAME}'"
    )
    if v.run_id == run_id
]

assert versions

version = max(
    versions,
    key=lambda v: int(v.version),
)

assert int(version.version) > int(champion.version)

client.set_registered_model_alias(
    MODEL_NAME,
    "challenger",
    version.version,
)

if promote:
    client.set_registered_model_alias(
        MODEL_NAME,
        "champion",
        version.version,
    )

    print("PROMOTE: Both acceptance criteria satisfied")

else:
    print("RETAIN: Original champion remains active")

    if not cost_gate:
        print("Reason: Insufficient drift holdout cost reduction")

    if not pr_auc_gate:
        print("Reason: Original test PR-AUC degradation exceeds limit")

active = client.get_model_version_by_alias(
    MODEL_NAME, "champion"
)

assert str(active.version) == str(
    version.version if promote else champion.version
)

report = {
    "previous_champion_version": str(champion.version),
    "challenger_version": str(version.version),
    "active_champion_version": str(active.version),
    "challenger_run_id": run_id,
    "original_train_rows": len(X_train),
    "drift_train_rows": len(drift_train),
    "drift_holdout_rows": len(drift_holdout),
    "combined_train_rows": len(X_combined),
    "champion_threshold": champion_threshold,
    "challenger_threshold": float(challenger_threshold),
    "champion_drift_holdout": champion_holdout,
    "challenger_drift_holdout": challenger_holdout,
    "champion_original_test": champion_test,
    "challenger_original_test": challenger_test,
    "holdout_cost_reduction": float(cost_reduction),
    "original_test_pr_auc_drop": float(pr_auc_drop),
    "cost_gate": cost_gate,
    "pr_auc_gate": pr_auc_gate,
    "promoted": promote,
    "decision": "PROMOTE" if promote else "RETAIN",
}

REPORTS.mkdir(parents=True, exist_ok=True)

report_path = REPORTS / "retrain_report.json"

report_path.write_text(
    json.dumps(report, indent=2)
)

with mlflow.start_run(run_id=run_id):
    mlflow.log_artifact(str(report_path))

print("\n========== FINAL CHALLENGE DECISION ==========")
print(json.dumps(report, indent=2))

print("PASS: Challenger trained and registered")
print("PASS: Promotion criteria evaluated")
print("PASS: Champion alias verified")
print("PASS: Retraining report saved")
