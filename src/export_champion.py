import json
from datetime import datetime, timezone

import joblib

from common import (
    FEATURES,
    MODEL_NAME,
    MODELS,
    init_mlflow,
)

mlflow = init_mlflow()
client = mlflow.tracking.MlflowClient()

champion = client.get_model_version_by_alias(
    MODEL_NAME, "champion"
)

run = client.get_run(champion.run_id)

pipe = mlflow.sklearn.load_model(
    f"models:/{MODEL_NAME}@champion"
)

MODELS.mkdir(parents=True, exist_ok=True)

model_path = MODELS / "fraud_model.joblib"

joblib.dump(pipe, model_path)

meta = {
    "model_name": MODEL_NAME,
    "model_version": str(champion.version),
    "run_id": champion.run_id,
    "threshold": float(
        run.data.metrics["decision_threshold"]
    ),
    "test_pr_auc": float(
        run.data.metrics["test_pr_auc"]
    ),
    "features": FEATURES,
    "exported_at": datetime.now(timezone.utc).isoformat(),
}

(MODELS / "model_meta.json").write_text(
    json.dumps(meta, indent=2)
)

print(json.dumps(meta, indent=2))
print("PASS: Champion model exported")
print("PASS: Model metadata exported")
