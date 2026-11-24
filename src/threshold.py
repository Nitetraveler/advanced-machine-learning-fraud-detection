from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold, cross_val_predict,
)

from common import (
    SEED, init_mlflow, load_split, make_pipeline,
)
from costs import (
    COST_FN, COST_FP, expected_cost, best_threshold,
)

X, y = load_split("train")

pipe = make_pipeline(
    LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    )
)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED,
)

oof_proba = cross_val_predict(
    pipe,
    X,
    y,
    cv=cv,
    method="predict_proba",
    n_jobs=1,
)[:, 1]

default_cost = expected_cost(y, oof_proba, 0.5)
threshold, minimum_cost = best_threshold(y, oof_proba)

bayes_threshold = COST_FP / (COST_FN + COST_FP)

print(f"OOF cost at threshold 0.50: {default_cost:.2f}")
print(f"Optimal OOF threshold: {threshold:.4f}")
print(f"Minimum OOF cost: {minimum_cost:.2f}")
print(f"Theoretical Bayes threshold: {bayes_threshold:.4f}")

if default_cost > 0:
    reduction = 100 * (
        default_cost - minimum_cost
    ) / default_cost
    print(f"Cost reduction: {reduction:.2f}%")

mlflow = init_mlflow()

with mlflow.start_run(
    run_name="lr_balanced_cost_threshold"
):
    mlflow.set_tag("task", "threshold")

    mlflow.log_params({
        "cost_fn": COST_FN,
        "cost_fp": COST_FP,
    })

    mlflow.log_metrics({
        "oof_cost_at_0_5": default_cost,
        "oof_best_cost": minimum_cost,
        "best_threshold": threshold,
    })

assert minimum_cost <= default_cost

print("PASS: OOF threshold optimisation")
