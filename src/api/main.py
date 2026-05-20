"""FastAPI app — health, predict, predict/latest, train, /metrics (Prometheus)."""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import date, timedelta

import mlflow
import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator

from src.api.inference import Predictor, load_predictor
from src.api.metrics import (
    INFERENCE_DURATION,
    INPUT_WINDOW_MEAN,
    INPUT_WINDOW_STD,
    LAST_TRAIN_METRIC,
    LAST_TRAIN_TS,
    MODEL_LOADED,
    PREDICTED_VALUE,
    PREDICTION_COUNT,
    TRAIN_COUNT,
    TRAIN_DURATION,
)
from src.api.middleware import AccessLogMiddleware
from src.api.schemas import (
    HealthResponse,
    LatestPredictResponse,
    PredictRequest,
    PredictResponse,
    TrainRequest,
    TrainResponse,
)
from src.config import DATA, MODEL, MODEL_PATH
from src.model.train import train as train_fn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger(__name__)


_state: dict[str, Predictor | None] = {"predictor": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _state["predictor"] = load_predictor()
        MODEL_LOADED.set(1)
        log.info("Model loaded.")
    except FileNotFoundError as e:
        log.warning("Predictor not available: %s", e)
        _state["predictor"] = None
        MODEL_LOADED.set(0)
    yield
    _state["predictor"] = None
    MODEL_LOADED.set(0)


app = FastAPI(
    title="BABA Stock Prediction API",
    version="0.1.0",
    description="LSTM next-day close-price prediction for Alibaba (BABA).",
    lifespan=lifespan,
)
app.add_middleware(AccessLogMiddleware)

# Prometheus HTTP instrumentation. Exposes /metrics in Prometheus text format.
# Auto-tracks http_requests_total + http_request_duration_seconds (histogram)
# plus the standard process collectors (memory, CPU, GC).
Instrumentator(
    excluded_handlers=["/metrics"],          # don't measure the scrape endpoint itself
    should_group_status_codes=False,         # keep 200 / 422 / 503 distinct
).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


def _require_predictor() -> Predictor:
    p = _state.get("predictor")
    if p is None:
        raise HTTPException(status_code=503,
                            detail="Model not loaded. Train first via POST /train.")
    return p


def _observe_window(closes: list[float]) -> None:
    """Push input-window summary stats into the drift histograms."""
    arr = np.asarray(closes, dtype=np.float64)
    INPUT_WINDOW_MEAN.observe(float(arr.mean()))
    INPUT_WINDOW_STD.observe(float(arr.std()))


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=_state.get("predictor") is not None,
        model_path=str(MODEL_PATH),
        symbol=DATA.symbol,
        lookback=MODEL.lookback,
    )


@app.post("/predict", response_model=PredictResponse, tags=["predict"])
def predict(body: PredictRequest) -> PredictResponse:
    predictor = _require_predictor()
    try:
        value, elapsed_ms = predictor.predict_next(body.closes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    PREDICTION_COUNT.labels(endpoint="/predict").inc()
    INFERENCE_DURATION.labels(endpoint="/predict").observe(elapsed_ms / 1000.0)
    PREDICTED_VALUE.observe(value)
    _observe_window(body.closes)
    return PredictResponse(predicted_close=value, inference_ms=round(elapsed_ms, 2))


@app.post("/predict/latest", response_model=LatestPredictResponse, tags=["predict"])
def predict_latest() -> LatestPredictResponse:
    predictor = _require_predictor()

    # Pull a few extra days to be safe across weekends/holidays.
    end = date.today()
    start = end - timedelta(days=MODEL.lookback * 2 + 30)
    df = yf.download(DATA.symbol, start=start.isoformat(), end=end.isoformat(),
                     auto_adjust=False, progress=False)
    if df.empty:
        raise HTTPException(status_code=502, detail="yfinance returned no data")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    closes = df[DATA.target_col].astype(float).values[-MODEL.lookback:]
    if len(closes) < MODEL.lookback:
        raise HTTPException(status_code=502,
                            detail=f"Need {MODEL.lookback} closes, got {len(closes)}")
    window_dates = df.index[-MODEL.lookback:]
    value, elapsed_ms = predictor.predict_next(closes.tolist())

    PREDICTION_COUNT.labels(endpoint="/predict/latest").inc()
    INFERENCE_DURATION.labels(endpoint="/predict/latest").observe(elapsed_ms / 1000.0)
    PREDICTED_VALUE.observe(value)
    _observe_window(closes.tolist())

    return LatestPredictResponse(
        predicted_close=value,
        inference_ms=round(elapsed_ms, 2),
        as_of=end,
        window_start=window_dates[0].date(),
        window_end=window_dates[-1].date(),
    )


@app.post("/train", response_model=TrainResponse, tags=["train"])
def train_model(body: TrainRequest | None = None) -> TrainResponse:
    """Run a fresh training pass, refresh the in-process model.

    Blocks for the duration of training (~1 min on CPU with defaults).
    FastAPI offloads sync handlers to a threadpool, so concurrent /predict
    calls are not blocked.
    """
    body = body or TrainRequest()
    kwargs = body.merged_with_defaults()

    log.info("POST /train kicked off with kwargs=%s", kwargs)
    t0 = time.perf_counter()
    try:
        metrics = train_fn(**kwargs)
    except FileNotFoundError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except mlflow.exceptions.MlflowException as e:
        raise HTTPException(status_code=503, detail=f"MLflow unreachable: {e}")
    duration = time.perf_counter() - t0

    try:
        _state["predictor"] = load_predictor()
        MODEL_LOADED.set(1)
        log.info("Predictor reloaded after training.")
    except FileNotFoundError as e:
        log.warning("Trained model not found on reload: %s", e)

    # Update training-observability metrics.
    TRAIN_COUNT.inc()
    TRAIN_DURATION.observe(duration)
    LAST_TRAIN_TS.set(time.time())
    for k, v in metrics.items():
        LAST_TRAIN_METRIC.labels(metric=k).set(float(v))

    last_run = mlflow.last_active_run()
    run_id = last_run.info.run_id if last_run is not None else ""

    return TrainResponse(
        status="completed",
        run_id=run_id,
        test_metrics=metrics,
        model_path=str(MODEL_PATH),
        duration_seconds=round(duration, 1),
        mlflow_tracking_uri=os.environ.get(
            "MLFLOW_TRACKING_URI", f"file:{MODEL_PATH.parent.parent / 'mlruns'}"
        ),
    )
