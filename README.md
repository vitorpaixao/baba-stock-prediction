<div align="center">

# BABA Stock Prediction

**LSTM next-day close-price prediction for Alibaba (BABA), served as a containerized FastAPI service with MLflow experiment tracking and structured request logging.**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.16-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://docs.docker.com/compose/)
[![Tests](https://img.shields.io/badge/tests-12%20passing-brightgreen)](#tests)

</div>

---

## Why this exists

This is the deliverable for **Tech Challenge Phase 4** of the MLET postgraduate program (worth 90% of the phase grade). The goal is to go end-to-end on a deep-learning system: **collect → preprocess → train an LSTM → save the model → serve it through a REST API → monitor it in production**.

---

## At a glance

| | |
| --- | --- |
| **Ticker** | BABA (Alibaba Group, NYSE) |
| **Data** | 2015-01-01 → today, daily OHLCV via `yfinance` (~2,860 trading days) |
| **Model** | 2-layer LSTM(50) + Dropout(0.2), 60-day look-back, univariate Close |
| **Test MAE / RMSE / MAPE** | **$4.44 · $5.92 · 3.50 %** (chronological tail hold-out) |
| **Serving** | FastAPI (`/health`, `/predict`, `/predict/latest`, `/metrics`) |
| **Monitoring** | MLflow (training) + JSON access logs + in-process latency histogram (serving) |
| **Tests** | 12 pytest cases, all green |

---

## Architecture

```
┌────────────┐   ┌──────────────────────┐   ┌────────────────────────┐
│  yfinance  │──▶│ data/raw/baba.parquet│──▶│ MinMaxScaler (fit train)│
└────────────┘   └──────────────────────┘   │ 60-day sliding windows  │
                                            └────────────┬───────────┘
                                                         ▼
                                       ┌───────────────────────────────┐
                                       │  LSTM(50) → Dropout(0.2)      │
                                       │  LSTM(50) → Dropout(0.2)      │
                                       │  Dense(1)                     │
                                       │  loss = MSE, optimizer = Adam │
                                       └───────────────┬───────────────┘
                                                       ▼
                          ┌──────────────────────────────────────────────┐
                          │ models/lstm_baba.keras + models/scaler.pkl   │
                          │ MLflow run: params · metrics · artifact      │
                          │ MLflow registry: lstm-baba v1                │
                          └─────────────────────┬────────────────────────┘
                                                ▼
                                      ┌──────────────────────┐
                                      │     FastAPI app      │
                                      │  /health · /predict  │
                                      │  /predict/latest     │
                                      │  /metrics            │
                                      └──────────┬───────────┘
                                                 ▼
                            JSON access logs · p50/p95/p99 latency
```

---

## How to run this project

Three independent flows. Pick the one matching your goal:

| Flow | What it does | MLflow runs created? | Best for |
| --- | --- | --- | --- |
| **1. Notebook demo** | Runs the entire pipeline interactively in Jupyter, saves model + scaler. | ❌ No | Understanding the project, video defense |
| **2. Local (script)** | `python -m src.model.train` from the terminal with full MLflow logging to `./mlruns/`. | ✅ Yes (file backend) | Iterating on hyperparameters, reproducible training |
| **3. Docker compose** | Containerized API + dedicated MLflow server, training logs to the dockerized MLflow. | ✅ Yes (SQLite server) | Demonstrating MLOps maturity, deployment |

All three share the same one-time Python setup below.

---

## One-time setup (all flows)

```powershell
# Python 3.11 or 3.12 (TensorFlow 2.18 requirement)
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux/macOS

# Install in editable mode with dev extras (jupyter, pytest, matplotlib, ...)
pip install -e ".[dev]"

# Fetch BABA history once (idempotent: skips if data/raw/baba.parquet exists)
python -m src.data.fetch
```

> **Install with `pip`, not `uv`.** On Windows, `uv sync` mishandles the `tensorflow` → `tensorflow-intel` meta-package redirect and leaves TensorFlow non-importable. `pip install -e ".[dev]"` is the supported path. If you've already run `uv sync` and broken the env, recover with:
>
> ```powershell
> pip install --force-reinstall tensorflow==2.18.0
> ```

---

## Flow 1: Notebook demo

The Jupyter notebook at [`notebooks/01_eda_lstm_baba.ipynb`](notebooks/01_eda_lstm_baba.ipynb) is the **primary demo** of the project. It runs the entire pipeline end-to-end in ~2 minutes, covering Requirements 1, 2 and 3 of the challenge spec by importing the same functions used by the production code in `src/`. No duplication; the notebook is a narrated wrapper around the modular pipeline.

> ⚠️ **This flow does NOT generate MLflow runs.** The notebook calls `model.fit(...)` directly without wrapping it in `mlflow.start_run()`. If you want MLflow tracking, use **Flow 2** or **Flow 3** instead. The notebook still saves `models/lstm_baba.keras` + `models/scaler.pkl` to disk.

**Launch it:**

```powershell
.venv\Scripts\activate
jupyter lab notebooks/01_eda_lstm_baba.ipynb
```

**What's inside (run cells top-to-bottom, ~2 minutes total):**

| # | Section | What it shows |
| - | --- | --- |
| 1 | **Imports + config** | Loads `src/config.py` constants (ticker, lookback, hyperparams). |
| 2 | **Requirement 1.1: Data collection** | Reads `data/raw/baba.parquet` (auto-fetches if missing). 2,800+ trading days. |
| 3 | **EDA: Price + volume** | Two-panel chart: BABA Close (top) and Volume (bottom). |
| 4 | **EDA: Daily returns** | Close + 60-day rolling mean, plus daily returns. Visual proof prices are non-stationary, returns are not. |
| 5 | **Naive baseline** | MAE / RMSE / MAPE for "predict yesterday's close". The floor any model must beat. |
| 6 | **Requirement 1.2: Preprocessing** | Calls `build_splits()` from `src/data/preprocess.py`; shows shapes and scaler bounds. |
| 7 | **Requirement 2.1: Model construction** | Calls `build_lstm()` from `src/model/architecture.py`; renders `model.summary()`. |
| 8 | **Requirement 2.2: Training** | Real training loop (~30 epochs, ~1 min on CPU) with `EarlyStopping` + `ModelCheckpoint`. Plots loss/val_loss + mae/val_mae curves. |
| 9 | **Requirement 2.3: Evaluation** | Inverse-transforms predictions, computes MAE/RMSE/MAPE in USD via `all_metrics()` from `src/model/evaluate.py`, prints LSTM-vs-naive comparison. |
| 10 | **Actual vs predicted plot** | Overlays prediction series on real series for the test window. |
| 11 | **Requirement 3: Saving** | `model.save()` + `joblib.dump(scaler)` to `models/`. |
| 12 | **Requirement 4: Deploy (reference)** | Pointer card to `src/api/` and Docker setup. Not executed in the notebook. |
| 13 | **Requirement 5: Monitoring (reference)** | Pointer to MLflow setup + serving middleware. |

**Tips:**

- Register the venv as a Jupyter kernel once: `python -m ipykernel install --user --name baba-stock-prediction --display-name "BABA venv"`.
- For a clean re-run during a demo, use **Kernel → Restart & Run All** (takes ~2 minutes).

---

## Flow 2: Local script with MLflow file backend

For iteration and reproducible training with full MLflow tracking, run the training script directly. Runs land in `./mlruns/` (file backend).

```powershell
# Train + log to ./mlruns/ + save models/lstm_baba.keras + scaler.pkl
python -m src.model.train

# Tune via CLI flags
python -m src.model.train --epochs 100 --units 64 --batch-size 64 --lr 5e-4

# Serve the API locally
uvicorn src.api.main:app --reload
# → http://localhost:8000/docs
```

**View MLflow runs in the browser:**

```powershell
mlflow ui --backend-store-uri ./mlruns --port 5050
# → http://localhost:5050
```

> Use port `5050` to avoid colliding with the compose MLflow on `5000` (in case Flow 3 is also up).

Each `python -m src.model.train` invocation logs:

- **Params**: `epochs`, `batch_size`, `units`, `dropout`, `lookback`, `learning_rate`, `patience`, `seed`.
- **Per-epoch metrics**: `loss`, `val_loss`, `mae`, `val_mae`.
- **Test metrics (USD scale)**: `test_mae`, `test_rmse`, `test_mape`.
- **Artifact**: the saved Keras model.
- **Registry entry**: `lstm-baba`, auto-incremented version.

Training takes ~1 minute on CPU.

---

## Flow 3: Docker compose (full stack)

The `Dockerfile` and `docker-compose.yml` package the API and a dedicated MLflow server as a portable stack, demonstrating deployment + monitoring maturity (Requirements 4 and 5 of the spec). Requires Docker Desktop running.

### Prerequisite: trained model on disk

The Docker image bakes in `models/lstm_baba.keras` + `models/scaler.pkl` at build time. If those don't exist yet, run **Flow 1** or **Flow 2** first to produce them.

### Bring up the stack

```powershell
docker compose up -d --build
docker compose ps     # both services should be `Up (healthy)` after ~15s
```

| Service       | Host port | Container port | Purpose                                |
| ------------- | --------- | -------------- | -------------------------------------- |
| `baba-api`    | **8010**  | 8000           | FastAPI prediction service             |
| `baba-mlflow` | **5000**  | 5000           | MLflow tracking server (SQLite-backed) |

> Host port for the API is `8010` (not `8000`) to avoid colliding with other dev containers. The container itself still listens on `8000`.

### Smoke-test the API

```powershell
curl http://localhost:8010/health
curl -X POST http://localhost:8010/predict/latest
```

### Logging a training run into the compose MLflow

By default, `python -m src.model.train` writes to the local file backend (`./mlruns/`). To send a run to the **dockerized** MLflow server instead, set two env vars **before** invoking training:

```powershell
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
$env:PYTHONIOENCODING = "utf-8"   # avoids a cp1252 emoji crash on Windows
python -m src.model.train
# then clear when done:
Remove-Item Env:\MLFLOW_TRACKING_URI
```

Refresh `http://localhost:5000` after the run finishes to see it under experiment `baba-lstm`.

> **Why `PYTHONIOENCODING` and not a `.env` file:** MLflow's client prints emoji to stdout. On Windows, the default cp1252 codec crashes on them. `PYTHONIOENCODING` is read by the CPython interpreter **before** any Python code runs, so a `.env` loaded via `python-dotenv` is too late. The env var must be set in the shell.
>
> **Note on the model registry:** the compose image is `ghcr.io/mlflow/mlflow:v2.16.2` while the client is `mlflow>=3.x`. The `registered_model_name=` call returns 404 against the older server, gets caught by a `try/except`, and is logged as a warning. Run + metrics + artifact still log fine. Bump the compose image to `ghcr.io/mlflow/mlflow:v3.12.0` if you want registry write-through.

### Stop the stack

```powershell
docker compose stop   # pauses, keeps volumes
docker compose down   # removes containers (keeps volumes by default)
```

---

## API reference

OpenAPI/Swagger UI auto-generated at **`GET /docs`** (ReDoc at `/redoc`).

### `GET /health`

Liveness + readiness probe. Used by the Docker healthcheck.

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_path": "/app/models/lstm_baba.keras",
  "symbol": "BABA",
  "lookback": 60
}
```

### `POST /predict`

You provide the last 60 closing prices (oldest first); you get back the predicted next close.

```bash
curl -X POST http://localhost:8010/predict \
  -H "Content-Type: application/json" \
  -d '{"closes": [100.12, 100.40, ...60 floats..., 132.59]}'
```

```json
{
  "symbol": "BABA",
  "predicted_close": 139.79,
  "inference_ms": 12.4
}
```

**Validation errors return HTTP 422:**

- Wrong length (≠ 60) → `422`
- Non-positive value (≤ 0) → `422`
- Non-finite value → `422`

### `POST /predict/latest`

Convenience endpoint: the API pulls the last 60 trading days from yfinance for you, then predicts.

```bash
curl -X POST http://localhost:8010/predict/latest
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

In-process counters and latency percentiles over the last 1,000 requests.

```json
{
  "total_requests": 142,
  "by_route": {"/health": 12, "/predict": 130},
  "latency_ms": {"p50": 11.4, "p95": 92.7, "p99": 178.0, "mean": 27.1}
}
```

---

## Serving-time monitoring: JSON logs + `/metrics`

Each request emits one line of JSON to stdout (parseable with `jq`, Loki, ELK, Datadog):

```json
{"request_id": "9f2c...", "method": "POST", "path": "/predict",
 "status": 200, "latency_ms": 12.34}
```

`AccessLogMiddleware` also feeds a 1,000-entry ring buffer that backs the `/metrics` endpoint (p50/p95/p99/mean + per-route counts). Lightweight, no external dependencies. Easy to swap for Prometheus later.

---

## Repository layout

```
baba-stock-prediction/
├── src/
│   ├── config.py                       # symbol, dates, hyperparams, paths
│   ├── data/{fetch,preprocess}.py      # yfinance → parquet → scaled windows
│   ├── model/{architecture,train,evaluate}.py
│   ├── api/{main,schemas,inference,middleware}.py
│   └── monitoring/mlflow_utils.py
├── notebooks/01_eda_lstm_baba.ipynb    # full pipeline demo, start here
├── tests/                              # 12 pytest cases
├── models/                             # lstm_baba.keras, scaler.pkl (gitignored)
├── data/raw/baba.parquet               # (gitignored)
├── Dockerfile                          # python:3.12-slim, healthcheck
├── docker-compose.yml                  # api + mlflow services
└── pyproject.toml                      # dependencies
```

---

## Tests

```bash
pytest -q
```

12 tests covering:

- Sliding-window shape invariants and scaler round-trip (`tests/test_preprocess.py`)
- Metric correctness on synthetic inputs (`tests/test_evaluate.py`)
- End-to-end inference on a tiny synthetic-trained model (`tests/test_inference.py`)
- FastAPI routes via `TestClient`: `/health`, `/predict` happy + validation errors, `/metrics` (`tests/test_api.py`)

Tests run in ~30 seconds without hitting yfinance or training a real model (they use a synthetic sine-wave dataframe; see `tests/conftest.py`).

---

## Honest caveats

- **Univariate.** Close only. Adding `Volume` and macro features (multivariate input) is the natural next step.
- **One-step-ahead.** Multi-day rollouts compound error fast.
- **Price vs. returns.** Stock prices are non-stationary; predicting **returns** (Δp/p) is more rigorous than predicting prices, even when MAPE looks pretty.
- **Single hold-out.** A walk-forward / expanding-window backtest gives a much more robust estimate of out-of-sample error.
- **Regression accuracy ≠ trading utility.** For real money you'd care about *direction accuracy* and *Sharpe of the strategy on top*, not MAPE.

---

<div align="center">

Tech Challenge Phase 4 · MLET postgraduate program · Built with TensorFlow, FastAPI, MLflow, Docker.

</div>
