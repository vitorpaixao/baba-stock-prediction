"""Model loading + prediction helpers used by FastAPI handlers."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

from src.config import MODEL, MODEL_PATH, SCALER_PATH

log = logging.getLogger(__name__)


@dataclass
class Predictor:
    model: object
    scaler: MinMaxScaler
    lookback: int = MODEL.lookback

    def predict_next(self, closes: Sequence[float]) -> tuple[float, float]:
        """Return (predicted_close, inference_ms)."""
        if len(closes) != self.lookback:
            raise ValueError(f"Expected {self.lookback} closes, got {len(closes)}")
        arr = np.asarray(closes, dtype=np.float32).reshape(-1, 1)
        scaled = self.scaler.transform(arr).reshape(1, self.lookback, 1)
        t0 = time.perf_counter()
        scaled_pred = self.model.predict(scaled, verbose=0).ravel()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        unscaled = self.scaler.inverse_transform(scaled_pred.reshape(-1, 1)).ravel()
        return float(unscaled[0]), float(elapsed_ms)


def load_predictor(model_path: Path = MODEL_PATH,
                   scaler_path: Path = SCALER_PATH) -> Predictor:
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact missing: {model_path}. "
                                "Call POST /train on the API first.")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler missing: {scaler_path}.")
    log.info("Loading model %s and scaler %s", model_path, scaler_path)
    model = load_model(model_path)
    scaler: MinMaxScaler = joblib.load(scaler_path)
    return Predictor(model=model, scaler=scaler)
