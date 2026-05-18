"""Shared fixtures: synthetic OHLCV dataframe for fast deterministic tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    """Sine-wave close prices on business-day index, long enough for windowing."""
    rng = pd.date_range("2018-01-01", periods=800, freq="B")
    t = np.arange(len(rng), dtype=np.float64)
    close = 100 + 20 * np.sin(t / 30.0) + 0.05 * t  # trend + cycle
    return pd.DataFrame({
        "Date": rng,
        "Open": close - 0.5,
        "High": close + 1.0,
        "Low": close - 1.0,
        "Close": close,
        "Adj Close": close,
        "Volume": np.full_like(close, 1_000_000),
    })
