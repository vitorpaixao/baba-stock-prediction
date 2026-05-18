"""Preprocessing: window shapes, scaler round-trip, no train/test leak."""
from __future__ import annotations

import numpy as np

from src.data.preprocess import _make_windows, build_splits


def test_make_windows_shapes() -> None:
    series = np.arange(100, dtype=np.float32)
    X, y = _make_windows(series, lookback=10)
    assert X.shape == (90, 10, 1)
    assert y.shape == (90,)
    # First window: [0..9] predicts series[10]
    assert np.allclose(X[0, :, 0], np.arange(10))
    assert y[0] == 10.0


def test_build_splits_chronological_and_scaler(synthetic_df) -> None:
    splits = build_splits(df=synthetic_df, lookback=30, train_frac=0.7, val_frac=0.15)

    assert splits.X_train.shape[1:] == (30, 1)
    assert splits.X_val.shape[1:] == (30, 1)
    assert splits.X_test.shape[1:] == (30, 1)
    # All scaled values must be in [0,1] for train (fit there).
    assert splits.X_train.min() >= 0.0 - 1e-6
    assert splits.X_train.max() <= 1.0 + 1e-6

    # Scaler round-trip
    sample = synthetic_df["Close"].astype(np.float32).values[:5].reshape(-1, 1)
    rt = splits.scaler.inverse_transform(splits.scaler.transform(sample))
    assert np.allclose(rt.ravel(), sample.ravel(), atol=1e-4)


def test_split_target_counts(synthetic_df) -> None:
    """Train loses `lookback` targets to warm-up; val/test get warm-up windows."""
    lookback = 30
    splits = build_splits(df=synthetic_df, lookback=lookback,
                          train_frac=0.7, val_frac=0.15)
    n = len(synthetic_df)
    expected = n - lookback  # only train block loses warm-up targets
    total = splits.y_train.size + splits.y_val.size + splits.y_test.size
    assert total == expected
