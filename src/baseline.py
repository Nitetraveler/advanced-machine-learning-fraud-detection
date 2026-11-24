import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from common import eval_metrics, init_mlflow, load_split, make_pipeline

X_train, y_train = load_split("train")
X_test, y_test = load_split("test")

mlflow = init_mlflow()

candidates = {
    "dummy_majority": DummyClassifier(strategy="most_frequent"),
    "logreg_baseline": LogisticRegression(max_iter=1000),
}

rows = []

for name, estimator in candidates.items():
    pipe = make_pipeline(estimator)
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    metrics = eval_metrics(y_test, proba, threshold=0.5)

    with mlflow.start_run(run_name=name):
        mlflow.set_tag("task", "baseline")
        mlflow.log_param("estimator", type(estimator).__name__)
        mlflow.log_metrics({
            f"test_{k}": v for k, v in metrics.items()
        })

    rows.append({"model": name, **metrics})

print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
