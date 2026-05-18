# BABA Stock Prediction — Tech Challenge Fase 4

LSTM model that predicts next-day **BABA (Alibaba, NYSE)** closing price, served via a FastAPI REST API, with training tracked in MLflow and structured request logs in production.

> Pós-Graduação MLET — Fase 4 (Deep Learning e IA).

---

## Results

Trained on BABA daily close from **2015-01-01** through current date (2,859 trading days, last refresh 2026-05-15).

| Metric (test split, original USD scale) | Value   |
| --------------------------------------- | ------- |
| MAE                                     | **4.44** |
| RMSE                                    | **5.92** |
| MAPE                                    | **3.50 %** |

Test split: chronological 15% tail (≈ last 14 months). Naive baseline (yesterday-as-tomorrow) for comparison is computed in the EDA notebook.

---

## Architecture

```
yfinance ──► data/raw/baba.parquet ──► MinMaxScaler + 60-day windows
                                                │
                                                ▼
                                       LSTM(50) → Dropout(0.2)
                                       LSTM(50) → Dropout(0.2)
                                       Dense(1)
                                                │
                              ┌─────────────────┴─────────────────┐
                              ▼                                   ▼
                  models/lstm_baba.keras                   MLflow tracking
                  models/scaler.pkl                        (params, metrics, artifact)
                              │
                              ▼
                       FastAPI service
              /health · /predict · /predict/latest · /metrics
                              │
                              ▼
                      Structured JSON logs
                  + in-memory latency histogram
```

---

## Quickstart — Docker

Requires a trained model on disk (`models/lstm_baba.keras`, `models/scaler.pkl`). If not present, run the local training step first (see below).

```bash
docker compose up --build
# API     → http://localhost:8000/docs
# MLflow  → http://localhost:5000
```

The compose stack runs two services:

| Service  | Port | Purpose                                                |
| -------- | ---- | ------------------------------------------------------ |
| `api`    | 8000 | FastAPI prediction service                             |
| `mlflow` | 5000 | MLflow tracking + model registry UI (sqlite-backed)    |

---

## Quickstart — Local

```bash
# 1. Python 3.11 or 3.12 (TF 2.18 requirement)
python -m venv .venv
.venv\Scripts\activate                # Windows
# source .venv/bin/activate           # Linux/macOS

# 2. Install
pip install -e ".[dev]"

# 3. Fetch data
python -m src.data.fetch              # idempotent; --force to refresh

# 4. Train
python -m src.model.train             # logs to ./mlruns, writes ./models/*

# 5. Serve
uvicorn src.api.main:app --reload     # http://localhost:8000/docs
```

---

## API

OpenAPI/Swagger UI auto-generated at `GET /docs`.

### `GET /health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_path": ".../models/lstm_baba.keras",
  "symbol": "BABA",
  "lookback": 60
}
```

### `POST /predict`

Provide the last 60 closing prices (oldest first):

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"closes": [100.1, 100.4, ..., 132.59]}'   # length == 60
```

Response:

```json
{
  "symbol": "BABA",
  "predicted_close": 139.79,
  "inference_ms": 149.5
}
```

### `POST /predict/latest`

API fetches the last 60 days from yfinance and predicts the next close.

```bash
curl -X POST http://localhost:8000/predict/latest
```

```json
{
  "symbol": "BABA",
  "predicted_close": 139.79,
  "inference_ms": 87.4,
  "as_of": "2026-05-18",
  "window_start": "2026-02-20",
  "window_end": "2026-05-15"
}
```

### `GET /metrics`

In-process counters + latency percentiles (last 1,000 requests):

```json
{
  "total_requests": 142,
  "by_route": {"/health": 12, "/predict": 130},
  "latency_ms": {"p50": 11.4, "p95": 92.7, "p99": 178.0, "mean": 27.1}
}
```

---

## Monitoring

**Training** — every `python -m src.model.train` run is logged to MLflow with params (epochs, lookback, units, ...), per-epoch loss/MAE curves, test MAE/RMSE/MAPE, and the saved Keras artifact. Models are also registered under the name `lstm-baba` (when registry backend supports it).

Open MLflow UI:

```bash
mlflow ui --backend-store-uri ./mlruns      # local
# or visit http://localhost:5000 with docker compose up
```

**Inference** — `AccessLogMiddleware` emits one JSON log line per request:

```json
{"request_id": "abc...", "method": "POST", "path": "/predict",
 "status": 200, "latency_ms": 12.34}
```

Latencies feed an in-memory ring buffer (1k entries) exposed at `/metrics` (p50/p95/p99/mean + per-route counts).

---

## Tests

```bash
pytest -q
```

12 tests covering windowing/scaler invariants, metric correctness, end-to-end inference on a tiny synthetic-trained model, and FastAPI handlers via `TestClient`.

---

## Repository layout

```
src/
├── config.py                # symbol, dates, hyperparams, paths
├── data/{fetch,preprocess}.py
├── model/{architecture,train,evaluate}.py
├── api/{main,schemas,inference,middleware}.py
└── monitoring/mlflow_utils.py
notebooks/01_eda_lstm_baba.ipynb
models/                      # lstm_baba.keras, scaler.pkl (gitignored)
data/raw/baba.parquet        # gitignored
tests/                       # 12 tests
Dockerfile · docker-compose.yml
plan/                        # original challenge spec + plan
```

---

## Limitations & honest caveats

- Univariate close-price LSTM. Volume + macro features (multivariate) would likely help.
- Predicts a single step ahead. Multi-day forecasts compound error fast.
- Stock prices are non-stationary; predicting **returns** (Δp/p) is generally more rigorous than predicting prices, even if MAPE looks pretty.
- Test split is chronological (no leakage) but a single hold-out window — for production use, walk-forward / expanding-window backtesting is the standard.
