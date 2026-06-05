from __future__ import annotations

import logging
import logging.config
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"],
    },
})

logger = logging.getLogger(__name__)

from . import config
from .model_loader import ModelService
from .predictor import predict_records
from .schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ListingFeatures,
    ModelInfoResponse,
    PredictionResponse,
)

model_service = ModelService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading model...")
    model_service.load()
    if model_service.state.loaded:
        logger.info("Model ready.")
    else:
        logger.error("Model failed to load: %s", model_service.state.error)
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title=config.APP_TITLE,
    version=config.APP_VERSION,
    description="HW03 FastAPI service. Use Swagger at /docs.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["service"])
def root() -> dict:
    return {
        "message": "QBC12 Listing Availability Prediction API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health() -> HealthResponse:
    if model_service.state.loaded and model_service.state.model is not None:
        return HealthResponse(status="ok", model_loaded=True)

    return HealthResponse(
        status="error",
        model_loaded=False,
        error=model_service.state.error or "Model is not loaded.",
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["model"])
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(**model_service.model_info())


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(payload: ListingFeatures) -> PredictionResponse:
    logger.info("POST /predict payload=%s", payload.model_dump())
    try:
        model = model_service.require_model()
    except RuntimeError as exc:
        logger.error("Model unavailable: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model is not loaded: {exc}",
        ) from exc

    result = predict_records(model, [payload])[0]
    logger.info("POST /predict result=%s", result)
    return result


@app.post("/predict-batch", response_model=BatchPredictionResponse, tags=["prediction"])
def predict_batch(payload: BatchPredictionRequest) -> BatchPredictionResponse:
    logger.info("POST /predict-batch records=%d", len(payload.records))
    try:
        model = model_service.require_model()
    except RuntimeError as exc:
        logger.error("Model unavailable: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model is not loaded: {exc}",
        ) from exc

    predictions = predict_records(model, payload.records)
    logger.info("POST /predict-batch returned %d predictions", len(predictions))
    return BatchPredictionResponse(count=len(predictions), predictions=predictions)