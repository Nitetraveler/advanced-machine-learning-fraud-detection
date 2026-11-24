#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")"
source env.sh

PASSED=0
FAILED=0

check() {
    local label="$1"
    shift

    if "$@" >/dev/null 2>&1; then
        printf 'PASS: %s\n' "$label"
        PASSED=$((PASSED + 1))
    else
        printf 'FAIL: %s\n' "$label"
        FAILED=$((FAILED + 1))
    fi
}

pyok() {
    PYTHONPATH=src python -c "$1"
}

runs_ge() {
    local task="$1"
    local minimum="$2"

    mlq "tags.task = '$task'" |
        jq -e --argjson n "$minimum" '
            [.runs[] |
                select(.info.status == "FINISHED")
            ] | length >= $n
        '
}

alias_ok() {
    curl -fs \
        "$MLFLOW_TRACKING_URI/api/2.0/mlflow/registered-models/alias?name=fraud-detector&alias=champion" |
        jq -e '.model_version.status == "READY"'
}

container_ok() {
    test "$(docker inspect \
        --format '{{.State.Health.Status}}' fraud-api)" = healthy
}

predict_ok() {
    curl -fs \
        -H 'Content-Type: application/json' \
        --data-binary @serving/sample.json \
        http://127.0.0.1:8000/predict |
        jq -e '(.predictions | length) == 2'
}

invalid_ok() {
    local status

    status=$(jq '.transactions[0].hour = 27' serving/sample.json |
        curl -s \
            -o /dev/null \
            -w '%{http_code}' \
            -H 'Content-Type: application/json' \
            --data-binary @- \
            http://127.0.0.1:8000/predict)

    test "$status" = "422"
}

export_ok() {
    test -s models/fraud_model.joblib &&
    test -s models/model_meta.json
}

parity_ok() {
    python src/compare_api.py | grep -q MATCH
}

echo "=============================================="
echo " ADVANCED MACHINE LEARNING LAB VALIDATION"
echo "=============================================="

check "01 - Dataset split: 24000 train / 6000 test" \
    pyok 'import pandas as p; assert len(p.read_csv("data/train.csv")) == 24000; assert len(p.read_csv("data/test.csv")) == 6000'

check "02 - MLflow server healthy" \
    curl -fs http://127.0.0.1:5000/health

check "03 - Dataset experiment logged" \
    runs_ge data 1

check "04 - Both baseline models logged" \
    runs_ge baseline 2

check "05 - Four imbalance experiments logged" \
    runs_ge imbalance 4

check "06 - Expected cost and threshold utilities" \
    pyok 'import numpy as n; from costs import expected_cost,best_threshold; y=n.array([1,0,1,0]); p=n.array([.9,.6,.2,.1]); assert expected_cost(y,p,.5)==205; t,c=best_threshold(y,p); assert n.isclose(t,.11) and c==5'

check "07 - Cost-sensitive threshold experiment logged" \
    runs_ge threshold 1

check "08 - Five ensemble experiments logged" \
    runs_ge ensembles 5

check "09 - Optuna study contains at least 20 trials" \
    pyok 'import optuna; s=optuna.load_study(study_name="xgb-fraud",storage="sqlite:///optuna.db"); assert len(s.trials)>=20'

check "10 - Tuned model experiment logged" \
    runs_ge tuning 1

check "11 - Registered champion alias available" \
    alias_ok

check "12 - Explanation aggregation utilities" \
    pyok 'import numpy as n; from explain_utils import aggregate_contributions; r=aggregate_contributions(n.array([[1,.6,-.2],[-1,-.6,.2]]),["num__amount","cat__channel_web","cat__channel_mobile"]); assert n.isclose(r["amount"],1) and n.isclose(r["channel"],.4)'

check "13 - SHAP additivity verified" \
    jq -e '.additivity_check == true' \
    reports/explain_report.json

check "14 - Exported model and metadata available" \
    export_ok

check "15 - Docker fraud API healthy" \
    container_ok

check "16 - API returns two predictions" \
    predict_ok

check "17 - Invalid hour rejected with HTTP 422" \
    invalid_ok

check "18 - Local and API prediction parity" \
    parity_ok

check "19 - PSI utility numerical test" \
    pyok 'import numpy as n; from drift_utils import psi_categorical; assert n.isclose(psi_categorical(["a"]*50+["b"]*50,["a"]*80+["b"]*20),.416,atol=.001)'

check "20 - Control batch decision OK" \
    jq -e '.decision == "OK"' \
    reports/drift_control.json

check "21 - Drift batch decision RETRAIN" \
    jq -e '.decision == "RETRAIN"' \
    reports/drift_drift.json

check "22 - Both monitoring experiments logged" \
    runs_ge monitoring 2

echo
echo "=============================================="
echo " VALIDATION SUMMARY"
echo "=============================================="
echo "PASSED: $PASSED"
echo "FAILED: $FAILED"
echo "TOTAL:  $((PASSED + FAILED))"

if [ "$PASSED" -eq 22 ] && [ "$FAILED" -eq 0 ]; then
    echo "ALL 22 CHECKS PASSED"
    exit 0
else
    echo "VALIDATION FAILED"
    exit 1
fi
