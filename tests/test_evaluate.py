"""Sanity checks on metric functions."""
from __future__ import annotations

import numpy as np

from src.model.evaluate import all_metrics, mae, mape, rmse


def test_perfect_prediction_zero_error() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert mae(y, y) == 0.0
    assert rmse(y, y) == 0.0
    assert mape(y, y) == 0.0


def test_known_values() -> None:
    y = np.array([10.0, 20.0])
    yhat = np.array([12.0, 18.0])
    m = all_metrics(y, yhat)
    assert m["mae"] == 2.0
    assert m["rmse"] == 2.0
    # MAPE = mean(|2/10|, |2/20|) * 100 = mean(20, 10) = 15
    assert abs(m["mape"] - 15.0) < 1e-9
