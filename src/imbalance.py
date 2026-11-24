import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

from common import (
    SEED, build_preprocessor, init_mlflow,
    load_split, make_pipeline,
)

X, y = load_split("train")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED,
)

scoring = {
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    "precision": "precision",
    "recall": "recall",
}

candidates = {
    "lr_plain": make_pipeline(
        LogisticRegression(max_iter=1000)
    ),
    "lr_balanced": make_pipeline(
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        )
    ),
    "lr_smote": ImbPipeline([
        ("prep", build_preprocessor()),
        ("smote", SMOTE(random_state=SEED)),
        ("model", LogisticRegression(max_iter=1000)),
    ]),
}

mlflow = init_mlflow()
rows = []


def evaluate(name, estimator, features, labels, leaky=False):
    scores = cross_validate(
        estimator,
        features,
        labels,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        error_score="raise",
    )

    metrics = {}

    for key in scoring:
        values = scores[f"test_{key}"]
        metrics[f"cv_{key}_mean"] = float(values.mean())
        metrics[f"cv_{key}_std"] = float(values.std())

    with mlflow.start_run(run_name=name):
        mlflow.set_tag("task", "imbalance")
        mlflow.set_tag("leaky", str(leaky))
        mlflow.log_param("cv_folds", 5)
        mlflow.log_metrics(metrics)

    rows.append({
        "model": name,
        "leaky": leaky,
        **metrics,
    })

    print(
        f"{name}: "
        f"PR-AUC={metrics['cv_pr_auc_mean']:.4f} "
        f"ROC-AUC={metrics['cv_roc_auc_mean']:.4f} "
        f"Recall={metrics['cv_recall_mean']:.4f}",
        flush=True,
    )


for name, estimator in candidates.items():
    evaluate(name, estimator, X, y)

# Intentional leakage demonstration.
# Never use this approach for production model selection.
X_prepared = build_preprocessor().fit_transform(X)

X_leaky, y_leaky = SMOTE(
    random_state=SEED
).fit_resample(X_prepared, y)

evaluate(
    "lr_smote_LEAKY",
    LogisticRegression(max_iter=1000),
    X_leaky,
    y_leaky,
    leaky=True,
)

print()
print(pd.DataFrame(rows).to_string(
    index=False,
    float_format=lambda x: f"{x:.4f}",
))
