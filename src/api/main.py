"""FastAPI app — health, predict, predict/latest, metrics."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import date, timedelta

import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException

from src.api.inference import Predictor, load_predictor
from src.api.middleware import METRICS, AccessLogMiddleware
from src.api.schemas import (
    HealthResponse,
    LatestPredictResponse,
    MetricsResponse,
    PredictRequest,
    PredictResponse,
)
from src.config import DATA, MODEL, MODEL_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger(__name__)


_state: dict[str, Predictor | None] = {"predictor": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _state["predictor"] = load_predictor()
        log.info("Model loaded.")
    except FileNotFoundError as e:
        log.warning("Predictor not available: %s", e)
        _state["predictor"] = None
    yield
    _state["predictor"] = None


app = FastAPI(
    title="BABA Stock Prediction API",
    version="0.1.0",
    description="LSTM next-day close-price prediction for Alibaba (BABA).",
    lifespan=lifespan,
)
app.add_middleware(AccessLogMiddleware)


def _require_predictor() -> Predictor:
    p = _state.get("predictor")
    if p is None:
        raise HTTPException(status_code=503,
                            detail="Model not loaded. Train first: python -m src.model.train")
    return p


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=_state.get("predictor") is not None,
        model_path=str(MODEL_PATH),
        symbol=DATA.symbol,
        lookback=MODEL.lookback,
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["meta"])
def metrics() -> MetricsResponse:
    return MetricsResponse(**METRICS.snapshot())


@app.post("/predict", response_model=PredictResponse, tags=["predict"])
def predict(body: PredictRequest) -> PredictResponse:
    predictor = _require_predictor()
    try:
        value, elapsed_ms = predictor.predict_next(body.closes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
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
    return LatestPredictResponse(
        predicted_close=value,
        inference_ms=round(elapsed_ms, 2),
        as_of=end,
        window_start=window_dates[0].date(),
        window_end=window_dates[-1].date(),
    )
