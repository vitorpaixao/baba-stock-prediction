"""FastAPI TestClient — health, predict happy path, validation error."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api import main as api_main
from src.api.inference import Predictor
from src.config import MODEL
from src.data.preprocess import build_splits
from src.model.architecture import build_lstm


@pytest.fixture(scope="module")
def client(synthetic_df) -> TestClient:
    splits = build_splits(df=synthetic_df, lookback=MODEL.lookback,
                          train_frac=0.7, val_frac=0.15)
    model = build_lstm(lookback=MODEL.lookback, units=8, dropout=0.0, learning_rate=1e-2)
    model.fit(splits.X_train, splits.y_train, epochs=1, batch_size=32, verbose=0)
    predictor = Predictor(model=model, scaler=splits.scaler, lookback=MODEL.lookback)
    with TestClient(api_main.app) as c:
        # Lifespan runs at context entry and may fail to load (no real model on disk).
        # Inject the test predictor after startup.
        api_main._state["predictor"] = predictor
        yield c
    api_main._state["predictor"] = None


def test_health_ok(client) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["lookback"] == MODEL.lookback


def test_predict_happy_path(client, synthetic_df) -> None:
    closes = synthetic_df["Close"].astype(float).values[-MODEL.lookback:].tolist()
    r = client.post("/predict", json={"closes": closes})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "predicted_close" in body
    assert body["predicted_close"] > 0
    assert body["inference_ms"] >= 0


def test_predict_wrong_length_422(client) -> None:
    r = client.post("/predict", json={"closes": [100.0, 101.0, 102.0]})
    assert r.status_code == 422


def test_predict_non_positive_422(client) -> None:
    bad = [1.0] * (MODEL.lookback - 1) + [0.0]  # zero not allowed
    r = client.post("/predict", json={"closes": bad})
    assert r.status_code == 422


def test_metrics_after_traffic(client, synthetic_df) -> None:
    client.get("/health")
    closes = synthetic_df["Close"].astype(float).values[-MODEL.lookback:].tolist()
    client.post("/predict", json={"closes": closes})

    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.json()
    assert body["total_requests"] >= 2
    assert "/health" in body["by_route"]
    assert body["latency_ms"]["mean"] >= 0
