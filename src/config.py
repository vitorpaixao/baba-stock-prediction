"""Central configuration: paths, hyperparameters, data range."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
MLRUNS_DIR = ROOT / "mlruns"

for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DataConfig:
    symbol: str = "BABA"
    start_date: str = "2015-01-01"
    end_date: str = date.today().isoformat()
    target_col: str = "Close"
    train_frac: float = 0.70
    val_frac: float = 0.15
    # test_frac = 1 - train - val


@dataclass(frozen=True)
class ModelConfig:
    lookback: int = 60
    lstm_units: int = 50
    dropout: float = 0.2
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    early_stop_patience: int = 10
    random_seed: int = 42


DATA = DataConfig()
MODEL = ModelConfig()

RAW_PARQUET = RAW_DIR / f"{DATA.symbol.lower()}.parquet"
PROCESSED_NPZ = PROCESSED_DIR / f"{DATA.symbol.lower()}_seq.npz"
MODEL_PATH = MODELS_DIR / f"lstm_{DATA.symbol.lower()}.keras"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

MLFLOW_TRACKING_URI = f"file:{MLRUNS_DIR.as_posix()}"
MLFLOW_EXPERIMENT = "baba-lstm"
MODEL_REGISTRY_NAME = "lstm-baba"
