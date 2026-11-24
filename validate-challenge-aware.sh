#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source env.sh

echo "========== MAIN LAB VALIDATION =========="

bash validate.sh

echo
echo "========== VERIFY ALL TASK LOGS =========="

for i in $(seq 1 8); do
    grep -Fq "TASK $i ALL CHECKS PASSED" \
        "evidence/task${i}-validation.log"

    echo "PASS: Task $i"
done

echo
echo "========== VERIFY CHALLENGE =========="

PYTHONPATH=src python - <<'PY'
import json
import math

import mlflow
import optuna
import requests

from common import MODEL_NAME

with open("reports/retrain_report.json") as f:
    r = json.load(f)

assert r["original_train_rows"] == 24000
assert r["drift_train_rows"] == 2500
assert r["drift_holdout_rows"] == 2500
assert r["combined_train_rows"] == 26500

print("PASS: Challenge data split")

old = r["champion_drift_holdout"]["cost"]
new = r["challenger_drift_holdout"]["cost"]

reduction = (old - new) / old if old > 0 else 0

drop = (
    r["champion_original_test"]["pr_auc"]
    - r["challenger_original_test"]["pr_auc"]
)

assert math.isclose(
    reduction,
    r["holdout_cost_reduction"],
    abs_tol=1e-10,
)

assert math.isclose(
    drop,
    r["original_test_pr_auc_drop"],
    abs_tol=1e-10,
)

promote = old > 0 and reduction >= .05 and drop <= .02

assert r["promoted"] == promote

assert r["decision"] == (
    "PROMOTE" if promote else "RETAIN"
)

print("PASS: Official promotion gates enforced")

client = mlflow.tracking.MlflowClient()

champion = client.get_model_version_by_alias(
    MODEL_NAME, "champion"
)

challenger = client.get_model_version_by_alias(
    MODEL_NAME, "challenger"
)

assert int(challenger.version) > 1
assert str(challenger.version) == r["challenger_version"]

expected = (
    r["challenger_version"]
    if promote else r["previous_champion_version"]
)

assert str(champion.version) == expected

print("PASS: Champion and challenger aliases")

run = client.get_run(challenger.run_id)

assert run.info.status == "FINISHED"
assert run.data.tags["task"] == "retrain"

print("PASS: Retraining experiment")

with open("models/model_meta.json") as f:
    meta = json.load(f)

assert meta["model_version"] == expected
assert meta["run_id"] == champion.run_id

response = requests.get(
    "http://127.0.0.1:8000/health",
    timeout=10,
)

response.raise_for_status()

assert response.json()["model_version"] == expected

print("PASS: Registry, export and API versions consistent")

study = optuna.load_study(
    study_name="xgb-fraud",
    storage="sqlite:///optuna.db",
)

assert len(study.trials) >= 30

print("PASS: Full 30-trial tuning target")

print("Champion:", champion.version)
print("Challenger:", challenger.version)
print(f"Cost reduction: {reduction:.2%}")
print(f"PR-AUC drop: {drop:.6f}")
print("Promotion decision:", r["decision"])
PY

echo
echo "========== VERIFY DATA INTEGRITY =========="

sha256sum -c evidence/task1-data-SHA256SUMS

echo
echo "========== VERIFY PRESERVED EVIDENCE =========="

sha256sum -c evidence/pre-challenge/SHA256SUMS

echo
echo "========== VERIFY DEPLOYED MODEL =========="

if jq -e '.promoted == false' \
    reports/retrain_report.json >/dev/null; then

    sha256sum -c evidence/task7-model-SHA256SUMS

    cmp models/fraud_model.joblib \
        evidence/pre-challenge/fraud_model.joblib

    echo "PASS: Original champion preserved"
fi

python src/compare_api.py

echo
echo "========== VERIFY DEPENDENCIES =========="

python -m pip check

echo
echo "========== VERIFY PYTHON SYNTAX =========="

python -m compileall -q src serving

echo "PASS: Python source syntax"

echo
echo "=============================================="
echo " MAIN LAB AND CHALLENGE ALL CHECKS PASSED"
echo "=============================================="
