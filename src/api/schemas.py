"""Pydantic request/response models for the prediction API."""
from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional

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


class TrainRequest(BaseModel):
    """Hyperparameter overrides for POST /train. All optional, defaults from MODEL."""

    epochs: Optional[int] = Field(default=None, gt=0)
    batch_size: Optional[int] = Field(default=None, gt=0)
    units: Optional[int] = Field(default=None, gt=0)
    dropout: Optional[float] = Field(default=None, ge=0.0, lt=1.0)
    lookback: Optional[int] = Field(default=None, gt=0)
    learning_rate: Optional[float] = Field(default=None, gt=0.0)
    patience: Optional[int] = Field(default=None, ge=0)
    seed: Optional[int] = Field(default=None, ge=0)

    def merged_with_defaults(self) -> dict:
        """Return a kwargs dict with `None` slots filled from MODEL config."""
        return {
            "epochs": self.epochs if self.epochs is not None else MODEL.epochs,
            "batch_size": self.batch_size if self.batch_size is not None else MODEL.batch_size,
            "units": self.units if self.units is not None else MODEL.lstm_units,
            "dropout": self.dropout if self.dropout is not None else MODEL.dropout,
            "lookback": self.lookback if self.lookback is not None else MODEL.lookback,
            "learning_rate": self.learning_rate if self.learning_rate is not None else MODEL.learning_rate,
            "patience": self.patience if self.patience is not None else MODEL.early_stop_patience,
            "seed": self.seed if self.seed is not None else MODEL.random_seed,
        }


class TrainResponse(BaseModel):
    status: str
    run_id: str
    test_metrics: Dict[str, float]
    model_path: str
    duration_seconds: float
    mlflow_tracking_uri: str
