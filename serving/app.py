import json
import os
from pathlib import Path
from typing import Annotated, Literal, Optional

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/model"))

meta = json.loads(
    (MODEL_DIR / "model_meta.json").read_text()
)

model = joblib.load(
    MODEL_DIR / "fraud_model.joblib"
)

FEATURES = meta["features"]

NUM_FEATURES = [
    "amount",
    "hour",
    "account_age_days",
    "txn_count_24h",
    "distance_from_home_km",
    "avg_amount_30d",
]

CAT_FEATURES = [
    "merchant_category",
    "channel",
]

THRESHOLD = float(
    os.getenv("DECISION_THRESHOLD", str(meta["threshold"]))
)

if not np.isfinite(THRESHOLD) or not 0 <= THRESHOLD <= 1:
    raise ValueError("Invalid decision threshold")

app = FastAPI(
    title="fraud-detector",
    version=str(meta["model_version"]),
)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request,
    exc: RequestValidationError,
):
    # Avoid echoing invalid non-finite JSON values.
    details = [
        {
            "loc": list(error["loc"]),
            "msg": error["msg"],
            "type": error["type"],
        }
        for error in exc.errors()
    ]

    return JSONResponse(
        status_code=422,
        content={"detail": details},
    )


class Transaction(BaseModel):
    amount: float = Field(
        gt=0, allow_inf_nan=False
    )

    hour: int = Field(ge=0, le=23)

    merchant_category: Optional[str] = None

    channel: Literal["pos", "web", "mobile"]

    account_age_days: int = Field(ge=0)

    txn_count_24h: int = Field(ge=0)

    distance_from_home_km: Optional[
        Annotated[
            float,
            Field(ge=0, allow_inf_nan=False),
        ]
    ] = None

    avg_amount_30d: float = Field(
        gt=0, allow_inf_nan=False
    )


class Batch(BaseModel):
    transactions: list[Transaction] = Field(
        min_length=1,
        max_length=1000,
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_name": meta["model_name"],
        "model_version": meta["model_version"],
    }


@app.get("/metadata")
def metadata():
    return {
        **meta,
        "active_threshold": THRESHOLD,
    }


@app.post("/predict")
def predict(batch: Batch):
    frame = pd.DataFrame([
        row.model_dump()
        for row in batch.transactions
    ])[FEATURES]

    frame[NUM_FEATURES] = (
        frame[NUM_FEATURES].astype(float)
    )

    frame[CAT_FEATURES] = (
        frame[CAT_FEATURES]
        .where(frame[CAT_FEATURES].notna(), np.nan)
    )

    probabilities = model.predict_proba(frame)[:, 1]

    return {
        "model_version": meta["model_version"],
        "threshold": THRESHOLD,
        "predictions": [
            {
                "fraud_probability": float(p),
                "is_fraud": bool(p >= THRESHOLD),
            }
            for p in probabilities
        ],
    }
