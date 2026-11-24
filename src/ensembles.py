import time

import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
)
from xgboost import XGBClassifier

from common import (
    SEED,
    init_mlflow,
    load_split,
    make_pipeline,
    eval_metrics,
)

X_train, y_train = load_split("train")
X_test, y_test = load_split("test")

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=SEED,
)

lr = make_pipeline(
    LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    )
)

rf = make_pipeline(
    RandomForestClassifier(
        n_estimators=150,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=SEED,
    ),
    scale=False,
)

hgb = make_pipeline(
    HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.08,
        random_state=SEED,
    ),
    scale=False,
)

xgb = make_pipeline(
    XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        eval_metric="aucpr",
        n_jobs=2,
        random_state=SEED,
    ),
    scale=False,
)

stack = StackingClassifier(
    estimators=[
        ("lr", lr),
        ("rf", rf),
        ("xgb", xgb),
    ],
    final_estimator=LogisticRegression(max_iter=1000),
    stack_method="predict_proba",
    cv=3,
    n_jobs=1,
)

candidates = {
    "logreg_balanced": lr,
    "random_forest": rf,
    "hist_gradient_boosting": hgb,
    "xgboost": xgb,
    "stacking_lr_rf_xgb": stack,
}

mlflow = init_mlflow()
rows = []

for i, (name, estimator) in enumerate(
    candidates.items(), start=1
):
    print(
        f"\n========== MODEL {i}/5: {name} ==========",
        flush=True,
    )

    started = time.perf_counter()

    print("Running 3-fold cross-validation...", flush=True)

    scores = cross_val_score(
        estimator,
        X_train,
        y_train,
        cv=cv,
        scoring="average_precision",
        n_jobs=1,
        error_score="raise",
    )

    print(
        f"CV PR-AUC: {scores.mean():.6f} "
        f"+/- {scores.std():.6f}",
        flush=True,
    )

    print("Training on full training dataset...", flush=True)

    estimator.fit(X_train, y_train)

    train_seconds = time.perf_counter() - started

    proba = estimator.predict_proba(X_test)[:, 1]
    test_metrics = eval_metrics(y_test, proba, threshold=0.5)

    metrics = {
        "cv_pr_auc_mean": float(scores.mean()),
        "cv_pr_auc_std": float(scores.std()),
        "test_pr_auc": test_metrics["pr_auc"],
        "test_roc_auc": test_metrics["roc_auc"],
        "train_seconds": float(train_seconds),
    }

    with mlflow.start_run(run_name=name):
        mlflow.set_tag("task", "ensembles")
        mlflow.log_param("estimator", name)
        mlflow.log_param("cv_folds", 3)
        mlflow.log_metrics(metrics)

    rows.append({"model": name, **metrics})

    print(
        f"Test PR-AUC: {metrics['test_pr_auc']:.6f}",
        flush=True,
    )
    print(
        f"Test ROC-AUC: {metrics['test_roc_auc']:.6f}",
        flush=True,
    )
    print(
        f"Training seconds: {train_seconds:.2f}",
        flush=True,
    )
    print(f"PASS: {name} completed", flush=True)

results = pd.DataFrame(rows).sort_values(
    "cv_pr_auc_mean",
    ascending=False,
)

print("\n========== ENSEMBLE COMPARISON ==========")
print(results.to_string(
    index=False,
    float_format=lambda x: f"{x:.6f}",
))

print("\nBest model by CV PR-AUC:", results.iloc[0]["model"])
print("PASS: All five ensemble experiments completed")
