"""End-to-end inference path with a tiny synthetic-trained model."""
from __future__ import annotations

import numpy as np
import pytest

from src.api.inference import Predictor
from src.data.preprocess import build_splits
from src.model.architecture import build_lstm


@pytest.fixture(scope="module")
def tiny_predictor(synthetic_df) -> Predictor:
    splits = build_splits(df=synthetic_df, lookback=20, train_frac=0.7, val_frac=0.15)
    model = build_lstm(lookback=20, units=8, dropout=0.0, learning_rate=1e-2)
    model.fit(splits.X_train, splits.y_train, epochs=2, batch_size=32, verbose=0)
    return Predictor(model=model, scaler=splits.scaler, lookback=20)


def test_predict_next_returns_finite_scalar(tiny_predictor, synthetic_df) -> None:
    window = synthetic_df["Close"].astype(float).values[-20:].tolist()
    pred, ms = tiny_predictor.predict_next(window)
    assert isinstance(pred, float) and np.isfinite(pred)
    assert ms >= 0.0
    # Prediction in plausible neighbourhood of the input range.
    lo, hi = min(window), max(window)
    assert lo * 0.3 <= pred <= hi * 3.0


def test_predict_next_wrong_length_raises(tiny_predictor) -> None:
    with pytest.raises(ValueError):
        tiny_predictor.predict_next([1.0] * 19)
