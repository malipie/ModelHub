"""FastAPI serving application for the E-Commerce Churn Predictor.

Endpoints
---------
POST /predict              → A/B-routed prediction (champion or challenger)
POST /predict/champion     → always uses champion model
POST /predict/challenger   → always uses challenger model (503 if unavailable)
GET  /health               → liveness / readiness probe
GET  /model/info           → loaded model versions and stages
GET  /metrics              → Prometheus metrics (text/plain)
PUT  /ab/config            → live traffic-split reconfiguration
POST /model/reload         → hot-reload models from MLflow Registry

Usage:
    uvicorn serving.src.main:app --port 8000
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from serving.src.ab_router import ABRouter
from serving.src.database import prediction_logger
from serving.src.metrics import (
    model_load_time_seconds,
    prediction_counter,
    prediction_errors_total,
    prediction_latency,
)
from serving.src.model_loader import ModelLoader
from serving.src.models import (
    ABConfig,
    PredictionRequest,
    PredictionResponse,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (from environment variables)
# ---------------------------------------------------------------------------

MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
MODEL_NAME: str = os.getenv("MODEL_NAME", "ecommerce-churn-predictor")
CHAMPION_PCT: int = int(os.getenv("CHAMPION_PCT", "80"))

# ---------------------------------------------------------------------------
# Module-level singletons — created at import time, populated at startup
# ---------------------------------------------------------------------------

loader = ModelLoader(model_name=MODEL_NAME, tracking_uri=MLFLOW_TRACKING_URI)
router = ABRouter(champion_pct=CHAMPION_PCT)


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    """Load models on startup; flush DB queue on shutdown."""
    start = time.time()
    loader.load()
    elapsed = time.time() - start
    model_load_time_seconds.labels(model_name=MODEL_NAME).set(elapsed)
    logger.info("Startup complete — models loaded in %.2fs", elapsed)
    prediction_logger.start()
    yield
    await prediction_logger.stop()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ML Model Serving API",
    description=(
        "Champion/Challenger A/B testing for the E-Commerce Churn Predictor. "
        "See `/docs` for interactive API explorer."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _request_to_df(request: PredictionRequest) -> pd.DataFrame:
    """Convert a PredictionRequest to a single-row DataFrame.

    Excludes ``request_id`` which is not a model feature.
    """
    data = request.model_dump(exclude={"request_id"})
    return pd.DataFrame([data])


def _run_prediction(model: Any, df: pd.DataFrame) -> tuple[int, float]:
    """Call model.predict_proba and return (label, churn_probability)."""
    churn_prob: float = float(model.predict_proba(df)[0, 1])
    label: int = int(churn_prob >= 0.5)
    return label, churn_prob


def _predict(
    request: PredictionRequest,
    request_id: str,
    model_role: str,
) -> PredictionResponse:
    """Core prediction logic shared across all /predict endpoints."""
    info = loader.get_info()

    if model_role == "challenger":
        model = loader.get_challenger()
        if model is None:
            # Graceful fallback: no challenger available → serve champion
            logger.warning("No challenger loaded; falling back to champion for %s", request_id)
            model = loader.get_champion()
            model_role = "champion"
            version = info["champion"]["version"]
        else:
            version = info["challenger"]["version"]
    else:
        model = loader.get_champion()
        version = info["champion"]["version"]

    df = _request_to_df(request)
    t0 = time.perf_counter()

    try:
        label, probability = _run_prediction(model, df)
    except Exception as exc:
        prediction_errors_total.inc()
        logger.exception("Prediction failed for request %s", request_id)
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc

    elapsed = time.perf_counter() - t0
    prediction_latency.labels(model_name=model_role).observe(elapsed)
    prediction_counter.labels(model_name=model_role, prediction=str(label)).inc()

    prediction_logger.enqueue(
        request_id=request_id,
        model_name=model_role,
        model_version=version,
        input_features=request.model_dump(exclude={"request_id"}),
        prediction=label,
        probability=round(probability, 4),
        latency_ms=round(elapsed * 1000, 2),
    )

    return PredictionResponse(
        prediction=label,
        probability=round(probability, 4),
        model_version=version,
        model_name=model_role,
        request_id=request_id,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", tags=["ops"])
def health() -> dict[str, Any]:
    """Liveness and readiness probe.

    Returns 503 if the champion model is not loaded.
    """
    info = loader.get_info()
    if not info["champion"]["loaded"]:
        raise HTTPException(status_code=503, detail="Champion model not loaded")
    return {"status": "ok", "models": info, "ab_config": router.get_config()}


@app.get("/model/info", tags=["ops"])
def model_info() -> dict[str, Any]:
    """Return information about currently loaded model versions."""
    return loader.get_info()


@app.get("/metrics", tags=["ops"], response_class=PlainTextResponse)
def metrics() -> PlainTextResponse:
    """Prometheus metrics endpoint."""
    return PlainTextResponse(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(request: PredictionRequest) -> PredictionResponse:
    """A/B-routed prediction.

    Routes each request to champion or challenger based on a deterministic
    hash of ``request_id`` and the current traffic split configuration.
    """
    variant = router.route(request.request_id)
    return _predict(request, request.request_id, variant)


@app.post("/predict/champion", response_model=PredictionResponse, tags=["prediction"])
def predict_champion(request: PredictionRequest) -> PredictionResponse:
    """Always predict using the champion (Production) model."""
    return _predict(request, request.request_id, "champion")


@app.post("/predict/challenger", response_model=PredictionResponse, tags=["prediction"])
def predict_challenger(request: PredictionRequest) -> PredictionResponse:
    """Always predict using the challenger (Staging) model.

    Returns 503 if no challenger is available.
    """
    if loader.get_challenger() is None:
        raise HTTPException(status_code=503, detail="No challenger model available")
    return _predict(request, request.request_id, "challenger")


@app.put("/ab/config", tags=["ops"])
def update_ab_config(config: ABConfig) -> dict[str, int]:
    """Update the champion / challenger traffic split at runtime."""
    router.set_split(config.champion_pct)
    return router.get_config()


@app.post("/model/reload", tags=["ops"])
def reload_models() -> dict[str, Any]:
    """Hot-reload both models from the MLflow Registry without downtime."""
    start = time.time()
    loader.reload()
    elapsed = time.time() - start
    model_load_time_seconds.labels(model_name=MODEL_NAME).set(elapsed)
    return {"status": "reloaded", "elapsed_s": round(elapsed, 3), "models": loader.get_info()}
