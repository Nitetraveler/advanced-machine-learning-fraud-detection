# Advanced Machine Learning — Fraud Detection

## Author: Tariq Hussain

**GitHub:** [Nitetraveler](https://github.com/Nitetraveler)

**Course:** Alnafi — Lab 43  
**Module:** Advanced Topics in Machine Learning  
**Project:** From Imbalanced Data to a Monitored Fraud-Detection Service

## Project Overview

An end-to-end machine learning fraud detection system covering
synthetic data generation, imbalanced classification, cost-sensitive
decision-making, ensemble learning, hyperparameter optimization,
experiment tracking, explainability, probability calibration,
containerized inference, drift monitoring and controlled retraining.

## Implementation Status

| Task | Description | Status |
|---|---|---|
| 1 | Synthetic fraud data and MLflow tracking | PASS |
| 2 | Baseline classification models | PASS |
| 3 | Imbalanced learning and cost optimization | PASS |
| 4 | Ensemble learning | PASS |
| 5 | Optuna tuning and model registry | PASS |
| 6 | SHAP explainability and calibration | PASS |
| 7 | FastAPI and Docker deployment | PASS |
| 8 | Data and performance drift monitoring | PASS |
| Challenge | Champion-challenger retraining | PASS |

## Technology Stack

- Ubuntu 24.04 LTS
- Python 3.12
- NumPy, Pandas and SciPy
- Scikit-learn and imbalanced-learn
- XGBoost CPU
- Optuna
- MLflow
- FastAPI, Pydantic and Uvicorn
- Docker
- SQLite
- Git and GitHub

## Dataset

All transaction data is synthetically generated.

- Total original transactions: 30,000
- Training transactions: 24,000
- Test transactions: 6,000
- Fraud transactions: 925
- Fraud prevalence: approximately 3.08%
- Control monitoring batch: 5,000
- Drift monitoring batch: 5,000

## Champion Model Results

| Metric | Result |
|---|---:|
| Model | Tuned XGBoost |
| Registry version | 1 |
| Optuna trials | 30 |
| Completed trials | 20 |
| Pruned trials | 10 |
| Cross-validation PR-AUC | 0.170032 |
| Test PR-AUC | 0.231612 |
| Test ROC-AUC | 0.842468 |
| Test recall | 0.778378 |
| Decision threshold | 0.23 |
| Default threshold cost | 23,145 |
| Optimized threshold cost | 15,795 |
| Cost reduction | 31.76% |

## Explainability and Calibration

Native XGBoost SHAP contributions were verified through
an additivity test. Permutation importance was also calculated.

| Metric | Result |
|---|---:|
| SHAP additivity | PASS |
| Raw Brier score | 0.063075 |
| Calibrated Brier score | 0.026639 |
| Brier score improvement | 57.77% |

The original champion and its matching optimized threshold
were retained for serving.

## Model Serving

The fraud detection API uses FastAPI and a non-root Docker container.

- Container image: fraud-api:1.0
- Container name: fraud-api
- Binding: 127.0.0.1:8000
- Memory limit: 1 GiB
- Restart policy: unless-stopped
- Health check: enabled

Endpoints:

- GET /health
- GET /metadata
- POST /predict

Invalid inputs are rejected with HTTP 422.

Local and API prediction parity was verified on 20 transactions,
with a maximum absolute probability difference of 0.0.

## Drift Monitoring

Monitoring uses Population Stability Index (PSI),
Kolmogorov-Smirnov tests and predictive performance metrics.

| Monitoring batch | Decision |
|---|---|
| Control | OK |
| Drift | RETRAIN |

## Challenge Extension

The challenger was trained using 24,000 original training
transactions and 2,500 drift transactions.

An independent 2,500-transaction drift holdout was used
for promotion evaluation.

| Metric | Champion v1 | Challenger v2 |
|---|---:|---:|
| Drift holdout PR-AUC | 0.163935 | 0.175150 |
| Drift holdout cost | 9,685 | 9,380 |
| Original test PR-AUC | 0.231612 | 0.241825 |
| Original test cost | 15,795 | 15,765 |

Promotion requires both:

1. Drift holdout cost reduction of at least 5%.
2. Original test PR-AUC degradation no greater than 0.02.

The challenger achieved a 3.15% holdout cost reduction
and improved original test PR-AUC.

**Final decision: RETAIN champion version 1.**

Challenger version 2 remains registered for future evaluation.
The production deployment was not replaced because the
minimum cost reduction requirement was not met.

## Project Structure

    src/          Machine learning implementation
    data/         Synthetic datasets
    models/       Exported champion and metadata
    reports/      Evaluation and monitoring reports
    serving/      FastAPI application and Dockerfile
    evidence/     Validation logs and integrity checksums
    mlartifacts/  MLflow experiment artifacts
    mlflow.db     MLflow tracking and registry database
    optuna.db     Persistent optimization study

## Environment Setup

    python3 -m venv .venv
    source .venv/bin/activate

    python -m pip install \
        pip==24.2 wheel==0.44.0 setuptools==72.1.0

    python -m pip install \
        -r requirements.txt -c constraints.txt

    source env.sh
    ./start_mlflow.sh

SQLAlchemy 2.0.32 is pinned in constraints.txt to maintain
compatibility with MLflow 2.15.1.

The environment script configures localhost proxy bypass
for local MLflow and API communication.

The environment and MLflow startup scripts expect the
project to be located at ~/ml-lab. Stored MLflow artifact
paths may require migration when restoring on another host.

## Build and Start the API

    docker build -f serving/Dockerfile -t fraud-api:1.0 .

    docker run -d \
        --name fraud-api \
        -p 127.0.0.1:8000:8000 \
        --memory 1g \
        --restart unless-stopped \
        fraud-api:1.0

## Validation

With MLflow and the Docker API running:

    bash validate.sh
    bash validate-challenge-aware.sh

Verified local results:

    PASSED: 22
    FAILED: 0
    TOTAL: 22

    MAIN LAB AND CHALLENGE ALL CHECKS PASSED

    Comprehensive validation exit status: 0

The local validator reconstructs the 22 lab acceptance checks.
These results are not an independently confirmed external
AI auto-grader score.

## Evidence and Reproducibility

The repository preserves synthetic datasets, trained models,
experiment records, monitoring reports, challenge results,
validation logs and SHA-256 integrity manifests.

## Author

**Tariq Hussain**

GitHub: https://github.com/Nitetraveler

Alnafi — Advanced Topics in Machine Learning

Completed: 24 November 2026
