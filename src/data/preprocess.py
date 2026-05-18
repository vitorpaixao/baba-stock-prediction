"""Scale close prices and build sliding-window sequences for LSTM training."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.config import DATA, MODEL, PROCESSED_NPZ, RAW_PARQUET, SCALER_PATH

log = logging.getLogger(__name__)


@dataclass
class Splits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scaler: MinMaxScaler


def _make_windows(series: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """Slide a window of `lookback` over a 1-D series → (X, y)."""
    X, y = [], []
    for i in range(lookback, len(series)):
        X.append(series[i - lookback : i])
        y.append(series[i])
    X = np.asarray(X, dtype=np.float32).reshape(-1, lookback, 1)
    y = np.asarray(y, dtype=np.float32)
    return X, y


def build_splits(df: pd.DataFrame | None = None,
                 lookback: int = MODEL.lookback,
                 train_frac: float = DATA.train_frac,
                 val_frac: float = DATA.val_frac) -> Splits:
    """Chronological train/val/test split + fit scaler on train only."""
    if df is None:
        df = pd.read_parquet(RAW_PARQUET)

    closes = df[DATA.target_col].astype(np.float32).values.reshape(-1, 1)
    n = len(closes)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_raw = closes[:n_train]
    val_raw = closes[n_train : n_train + n_val]
    test_raw = closes[n_train + n_val :]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_raw)

    train_s = scaler.transform(train_raw).ravel()
    val_s = scaler.transform(val_raw).ravel()
    test_s = scaler.transform(test_raw).ravel()

    # To predict the first val/test point, we need the last `lookback` train points
    # as the warm-up window. Concatenate then window.
    val_input = np.concatenate([train_s[-lookback:], val_s])
    test_input = np.concatenate([val_s[-lookback:], test_s])

    X_train, y_train = _make_windows(train_s, lookback)
    X_val, y_val = _make_windows(val_input, lookback)
    X_test, y_test = _make_windows(test_input, lookback)

    log.info("Shapes — train %s val %s test %s", X_train.shape, X_val.shape, X_test.shape)
    return Splits(X_train, y_train, X_val, y_val, X_test, y_test, scaler)


def save_splits(splits: Splits, npz_path: Path = PROCESSED_NPZ,
                scaler_path: Path = SCALER_PATH) -> None:
    np.savez_compressed(npz_path,
                        X_train=splits.X_train, y_train=splits.y_train,
                        X_val=splits.X_val, y_val=splits.y_val,
                        X_test=splits.X_test, y_test=splits.y_test)
    joblib.dump(splits.scaler, scaler_path)
    log.info("Wrote splits → %s, scaler → %s", npz_path, scaler_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    splits = build_splits()
    save_splits(splits)


if __name__ == "__main__":
    main()
