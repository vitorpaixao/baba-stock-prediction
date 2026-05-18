"""Thin wrapper around MLflow tracking config."""
from __future__ import annotations

import os

import mlflow

from src.config import MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI


def configure() -> None:
    """Use MLFLOW_TRACKING_URI env var if set (e.g. http://mlflow:5000 in compose),
    otherwise fall back to local file store."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
