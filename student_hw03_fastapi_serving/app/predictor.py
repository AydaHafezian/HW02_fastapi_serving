from __future__ import annotations

from typing import Iterable, List

import pandas as pd
from fastapi import HTTPException, status

from . import config
from .schemas import ListingFeatures, PredictionResponse


def records_to_dataframe(records: Iterable[ListingFeatures]) -> pd.DataFrame:
    """Convert validated API payloads into the exact DataFrame expected by the model."""
    rows = [record.model_dump() for record in records]
    df = pd.DataFrame(rows)

    missing_cols = [c for c in config.EXPECTED_FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "Missing required feature fields.",
                "missing_fields": missing_cols,
            },
        )

    forbidden_cols = [c for c in config.FORBIDDEN_FIELDS if c in df.columns]
    if forbidden_cols:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "Forbidden leakage fields are not allowed.",
                "forbidden_fields": forbidden_cols,
            },
        )

    extra_cols = [c for c in df.columns if c not in config.EXPECTED_FEATURE_COLUMNS]
    if extra_cols:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "Unexpected or forbidden fields found.",
                "extra_fields": extra_cols,
            },
        )

    return df[config.EXPECTED_FEATURE_COLUMNS]


def predict_records(model, records: List[ListingFeatures]) -> List[PredictionResponse]:
    """Run model prediction and return API responses."""
    X = records_to_dataframe(records).rename(columns=config.MODEL_COLUMN_RENAMES)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        positive_proba = proba[:, 1]
        preds = (positive_proba >= config.PREDICTION_THRESHOLD).astype(int)
    else:
        positive_proba = None
        preds = model.predict(X)

    responses: List[PredictionResponse] = []
    for i, pred in enumerate(preds):
        probability = float(positive_proba[i]) if positive_proba is not None else None
        responses.append(
            PredictionResponse(
                prediction=int(pred),
                prediction_label=config.POSITIVE_LABEL if int(pred) == 1 else config.NEGATIVE_LABEL,
                probability=probability,
                threshold=config.PREDICTION_THRESHOLD,
            )
        )

    return responses