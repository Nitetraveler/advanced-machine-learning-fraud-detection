import argparse
import json

import numpy as np
import optuna

from optuna.trial import TrialState
from sklearn.metrics import average_precision_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
)
from xgboost import XGBClassifier

from common import (
    LAB,
    REPORTS,
    SEED,
    MODEL_NAME,
    init_mlflow,
    load_split,
    make_pipeline,
    eval_metrics,
)

from costs import expected_cost, best_threshold


parser = argparse.ArgumentParser()
parser.add_argument("--trials", type=int, default=30)
args = parser.parse_args()

X_train, y_train = load_split("train")
X_test, y_test = load_split("test")

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=SEED,
)


def build_model(params):
    return make_pipeline(
        XGBClassifier(
            **params,
            tree_method="hist",
            eval_metric="aucpr",
            n_jobs=2,
            random_state=SEED,
        ),
        scale=False,
    )


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int(
            "n_estimators", 100, 500, step=50
        ),
        "max_depth": trial.suggest_int(
            "max_depth", 2, 8
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.3, log=True
        ),
        "subsample": trial.suggest_float(
            "subsample", 0.6, 1.0
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.5, 1.0
        ),
        "min_child_weight": trial.suggest_int(
            "min_child_weight", 1, 20
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", 0.001, 10.0, log=True
        ),
        "scale_pos_weight": trial.suggest_float(
            "scale_pos_weight", 1.0, 30.0, log=True
        ),
    }

    scores = []

    for step, (train_idx, valid_idx) in enumerate(
        cv.split(X_train, y_train)
    ):
        pipe = build_model(params)

        pipe.fit(
            X_train.iloc[train_idx],
            y_train.iloc[train_idx],
        )

        proba = pipe.predict_proba(
            X_train.iloc[valid_idx]
        )[:, 1]

        score = average_precision_score(
            y_train.iloc[valid_idx],
            proba,
        )

        scores.append(float(score))

        trial.report(float(np.mean(scores)), step)

        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(scores))


study = optuna.create_study(
    study_name="xgb-fraud",
    storage=f"sqlite:///{LAB / 'optuna.db'}",
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=SEED),
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=1,
    ),
    load_if_exists=True,
)

print("========== OPTUNA STUDY ==========", flush=True)
print("Existing trials:", len(study.trials), flush=True)
print("Requested new trials:", args.trials, flush=True)

study.optimize(
    objective,
    n_trials=args.trials,
    n_jobs=1,
    gc_after_trial=True,
    show_progress_bar=False,
)

complete = [
    t for t in study.trials
    if t.state == TrialState.COMPLETE
]

pruned = [
    t for t in study.trials
    if t.state == TrialState.PRUNED
]

print("\n========== TUNING RESULTS ==========")
print("Total trials:", len(study.trials))
print("Completed trials:", len(complete))
print("Pruned trials:", len(pruned))
print("Best trial:", study.best_trial.number)
print("Best CV PR-AUC:", study.best_value)
print("Best parameters:")
print(json.dumps(study.best_params, indent=2))

params = study.best_params
pipe = build_model(params)

print("\n========== OOF THRESHOLD OPTIMISATION ==========")

oof_proba = cross_val_predict(
    pipe,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=1,
)[:, 1]

threshold, oof_cost = best_threshold(
    y_train,
    oof_proba,
)

print("Decision threshold:", threshold)
print("Optimised OOF cost:", oof_cost)

print("\n========== TRAIN FINAL XGBOOST ==========")

pipe.fit(X_train, y_train)

test_proba = pipe.predict_proba(X_test)[:, 1]

test_metrics = eval_metrics(
    y_test,
    test_proba,
    threshold=threshold,
)

test_cost = expected_cost(
    y_test,
    test_proba,
    threshold,
)

test_cost_default = expected_cost(
    y_test,
    test_proba,
    0.5,
)

metrics = {
    "cv_pr_auc": float(study.best_value),
    **{
        f"test_{k}": float(v)
        for k, v in test_metrics.items()
    },
    "decision_threshold": float(threshold),
    "test_cost": float(test_cost),
    "test_cost_at_0_5": float(test_cost_default),
}

print(json.dumps(metrics, indent=2))

mlflow = init_mlflow()

print("\n========== REGISTER MODEL ==========")

with mlflow.start_run(run_name="xgb_optuna_best") as run:
    run_id = run.info.run_id

    mlflow.set_tag("task", "tuning")

    mlflow.log_params({
        **params,
        "study_trials_total": len(study.trials),
    })

    mlflow.log_metrics(metrics)

    mlflow.sklearn.log_model(
        pipe,
        artifact_path="model",
        registered_model_name=MODEL_NAME,
        input_example=X_train.dropna().head(5),
    )

client = mlflow.tracking.MlflowClient()

versions = [
    v for v in client.search_model_versions(
        f"name='{MODEL_NAME}'"
    )
    if v.run_id == run_id
]

assert versions, "Registered model version not found"

version = max(versions, key=lambda v: int(v.version))

client.set_registered_model_alias(
    MODEL_NAME,
    "champion",
    version.version,
)

champion = client.get_model_version_by_alias(
    MODEL_NAME,
    "champion",
)

assert champion.run_id == run_id

print("Registered model:", MODEL_NAME)
print("Champion version:", champion.version)
print("Champion run ID:", champion.run_id)

importance = optuna.importance.get_param_importances(study)

report = {
    "study_name": study.study_name,
    "total_trials": len(study.trials),
    "completed_trials": len(complete),
    "pruned_trials": len(pruned),
    "best_trial": study.best_trial.number,
    "best_params": params,
    "metrics": metrics,
    "oof_best_cost": float(oof_cost),
    "parameter_importance": importance,
    "model_name": MODEL_NAME,
    "model_version": str(champion.version),
    "run_id": run_id,
}

REPORTS.mkdir(parents=True, exist_ok=True)

(REPORTS / "tuning_report.json").write_text(
    json.dumps(report, indent=2)
)

print("\n========== PARAMETER IMPORTANCE ==========")

for name, value in importance.items():
    print(f"{name}: {value:.6f}")

print("\nPASS: Optuna tuning completed")
print("PASS: Champion registered")
print("PASS: Tuning report saved")
