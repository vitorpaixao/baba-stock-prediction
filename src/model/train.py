"""Train LSTM, log to MLflow, persist model + scaler for inference.

Usage:
    python -m src.model.train
    python -m src.model.train --epochs 100 --units 64
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
from dataclasses import asdict

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import mlflow
import mlflow.tensorflow

from src.config import MODEL, MODEL_PATH, MODEL_REGISTRY_NAME, RAW_PARQUET
from src.data.fetch import fetch
from src.data.preprocess import build_splits, save_splits
from src.model.architecture import build_lstm
from src.model.evaluate import all_metrics
from src.monitoring.mlflow_utils import configure as mlflow_configure

log = logging.getLogger(__name__)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train(epochs: int = MODEL.epochs,
          batch_size: int = MODEL.batch_size,
          units: int = MODEL.lstm_units,
          dropout: float = MODEL.dropout,
          lookback: int = MODEL.lookback,
          learning_rate: float = MODEL.learning_rate,
          patience: int = MODEL.early_stop_patience,
          seed: int = MODEL.random_seed) -> dict[str, float]:
    _seed(seed)

    if not RAW_PARQUET.exists():
        log.info("Raw data missing — fetching.")
        fetch()

    splits = build_splits(lookback=lookback)
    save_splits(splits)
    scaler = splits.scaler

    model = build_lstm(lookback=lookback, units=units, dropout=dropout,
                       learning_rate=learning_rate)
    model.summary(print_fn=log.info)

    mlflow_configure()
    with mlflow.start_run() as run:
        mlflow.log_params({
            **asdict(MODEL),
            "epochs": epochs,
            "batch_size": batch_size,
            "units": units,
            "dropout": dropout,
            "lookback": lookback,
            "learning_rate": learning_rate,
            "patience": patience,
            "seed": seed,
        })

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=patience,
                          restore_best_weights=True),
            ModelCheckpoint(filepath=str(MODEL_PATH), monitor="val_loss",
                            save_best_only=True),
        ]
        history = model.fit(
            splits.X_train, splits.y_train,
            validation_data=(splits.X_val, splits.y_val),
            epochs=epochs, batch_size=batch_size,
            callbacks=callbacks, verbose=2,
        )

        for k, v in history.history.items():
            for ep, val in enumerate(v):
                mlflow.log_metric(k, float(val), step=ep)

        # Evaluate in ORIGINAL price scale (inverse transform).
        y_pred_scaled = model.predict(splits.X_test, verbose=0).ravel()
        y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        y_true = scaler.inverse_transform(splits.y_test.reshape(-1, 1)).ravel()
        metrics = all_metrics(y_true, y_pred)
        log.info("Test metrics: %s", metrics)
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})

        # Persist for the API.
        model.save(MODEL_PATH)
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="model")

        try:
            mlflow.tensorflow.log_model(model, artifact_path="keras_model",
                                        registered_model_name=MODEL_REGISTRY_NAME)
        except Exception as e:  # registry unavailable on file:// store — non-fatal.
            log.warning("Model registry skipped: %s", e)

        log.info("MLflow run %s — model saved to %s", run.info.run_id, MODEL_PATH)

    return metrics


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=MODEL.epochs)
    p.add_argument("--batch-size", type=int, default=MODEL.batch_size)
    p.add_argument("--units", type=int, default=MODEL.lstm_units)
    p.add_argument("--dropout", type=float, default=MODEL.dropout)
    p.add_argument("--lookback", type=int, default=MODEL.lookback)
    p.add_argument("--lr", type=float, default=MODEL.learning_rate)
    p.add_argument("--patience", type=int, default=MODEL.early_stop_patience)
    p.add_argument("--seed", type=int, default=MODEL.random_seed)
    args = p.parse_args(argv)

    metrics = train(epochs=args.epochs, batch_size=args.batch_size, units=args.units,
                    dropout=args.dropout, lookback=args.lookback,
                    learning_rate=args.lr, patience=args.patience, seed=args.seed)
    print(metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
