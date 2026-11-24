# Usage: source ~/ml-lab/env.sh
export LAB="$HOME/ml-lab"
export MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
export GIT_PYTHON_REFRESH=quiet
source "$LAB/.venv/bin/activate"

# Query MLflow experiment runs through the REST API.
mlq() {
  local exp_id
  exp_id=$(curl -s "$MLFLOW_TRACKING_URI/api/2.0/mlflow/experiments/get-by-name?experiment_name=fraud-detection" \
           | jq -r '.experiment.experiment_id')
  jq -n --arg id "$exp_id" --arg f "$1" '{experiment_ids: [$id], filter: $f, max_results: 200}' \
    | curl -s -X POST "$MLFLOW_TRACKING_URI/api/2.0/mlflow/runs/search" \
           -H 'Content-Type: application/json' -d @-
}

# Lab localhost proxy bypass
export NO_PROXY="127.0.0.1,localhost,::1${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
