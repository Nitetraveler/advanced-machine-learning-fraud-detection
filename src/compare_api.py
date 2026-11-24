import json
import urllib.request

import joblib
import numpy as np

from common import MODELS, load_split

model = joblib.load(
    MODELS / "fraud_model.joblib"
)

X_test, _ = load_split("test")

X = X_test.head(20)

local = model.predict_proba(X)[:, 1]

records = (
    X.astype(object)
    .where(X.notna(), None)
    .to_dict(orient="records")
)

payload = json.dumps(
    {"transactions": records},
    default=lambda obj: obj.item(),
    allow_nan=False,
).encode("utf-8")

request = urllib.request.Request(
    "http://127.0.0.1:8000/predict",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)

with urllib.request.urlopen(request, timeout=30) as response:
    result = json.load(response)

remote = np.array([
    row["fraud_probability"]
    for row in result["predictions"]
])

assert len(local) == len(remote) == 20

difference = float(
    np.max(np.abs(local - remote))
)

print("Compared predictions:", len(local))
print("Maximum absolute difference:", difference)

assert difference < 1e-6

print("MATCH: Local and API predictions")
