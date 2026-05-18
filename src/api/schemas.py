"""Pydantic request/response models for the prediction API."""
from __future__ import annotations

from datetime import date
from typing import List

from pydantic import BaseModel, Field, field_validator

from src.config import DATA, MODEL


class PredictRequest(BaseModel):
    closes: List[float] = Field(
        ...,
        description=f"Closing prices for the last {MODEL.lookback} trading days, oldest first.",
        min_length=MODEL.lookback,
        max_length=MODEL.lookback,
    )

    @field_validator("closes")
    @classmethod
    def _finite_positive(cls, v: List[float]) -> List[float]:
        import math
        for i, x in enumerate(v):
            if not math.isfinite(x):
                raise ValueError(f"closes[{i}] is not finite")
            if x <= 0:
                raise ValueError(f"closes[{i}] must be > 0")
        return v


class PredictResponse(BaseModel):
    symbol: str = DATA.symbol
    predicted_close: float
    inference_ms: float


class LatestPredictResponse(PredictResponse):
    as_of: date
    window_start: date
    window_end: date


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    symbol: str
    lookback: int


class MetricsResponse(BaseModel):
    total_requests: int
    by_route: dict[str, int]
    latency_ms: dict[str, float]  # p50, p95, p99, mean
