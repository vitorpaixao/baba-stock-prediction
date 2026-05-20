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

Two paths. Pick the one matching your goal:

| Path | What it does | MLflow runs created? | Best for |
| --- | --- | --- | --- |
| **A. Notebook demo** | Runs the entire pipeline interactively in Jupyter, saves model + scaler. | ❌ No | Understanding the project, walkthroughs |
| **B. Docker Compose (full pipeline)** | Containerized API + dedicated MLflow server. Training is fired from the API via `POST /train` and logged to the dockerized MLflow. | ✅ Yes (SQLite-backed server) | Reproducible training, serving, deployment, monitoring |

Both share the same one-time Python setup below.

---

## One-time setup (both paths)

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

> The `fetch` step is only required for **Path A** (the notebook). **Path B** triggers the fetch automatically inside the container the first time `POST /train` is called.

> **Install with `pip`, not `uv`.** On Windows, `uv sync` mishandles the `tensorflow` → `tensorflow-intel` meta-package redirect and leaves TensorFlow non-importable. `pip install -e ".[dev]"` is the supported path. If you've already run `uv sync` and broken the env, recover with:
>
> ```powershell
> pip install --force-reinstall tensorflow==2.18.0
> ```

---

## Path A: Notebook demo

The Jupyter notebook at [`notebooks/01_eda_lstm_baba.ipynb`](notebooks/01_eda_lstm_baba.ipynb) is the **primary demo** of the project. It runs the entire pipeline end-to-end in ~2 minutes, importing the same functions used by the production code in `src/`. No duplication; the notebook is a narrated wrapper around the modular pipeline.

> ⚠️ **This path does NOT generate MLflow runs.** The notebook calls `model.fit(...)` directly without wrapping it in `mlflow.start_run()`. For MLflow tracking, use **Path B**. The notebook still saves `models/lstm_baba.keras` + `models/scaler.pkl` to disk.

**Launch it:**

```powershell
.venv\Scripts\activate
jupyter lab notebooks/01_eda_lstm_baba.ipynb
```

**What's inside (run cells top-to-bottom, ~2 minutes total):**

| # | Section | What it shows |
| - | --- | --- |
| 1 | **Imports + config** | Loads `src/config.py` constants (ticker, lookback, hyperparams). |
| 2 | **Data collection** | Reads `data/raw/baba.parquet` (auto-fetches if missing). 2,800+ trading days. |
| 3 | **EDA: Price + volume** | Two-panel chart: BABA Close (top) and Volume (bottom). |
| 4 | **EDA: Daily returns** | Close + 60-day rolling mean, plus daily returns. Visual proof prices are non-stationary, returns are not. |
| 5 | **Naive baseline** | MAE / RMSE / MAPE for "predict yesterday's close". The floor any model must beat. |
| 6 | **Preprocessing** | Calls `build_splits()` from `src/data/preprocess.py`; shows shapes and scaler bounds. |
| 7 | **Why an LSTM?** | Short narrative on RNNs, vanishing gradients, and how LSTM gates solve it. |
| 8 | **Model architecture** | Calls `build_lstm()` from `src/model/architecture.py`; renders `model.summary()`. |
| 9 | **Training** | Real training loop (~30 epochs, ~1 min on CPU) with `EarlyStopping` + `ModelCheckpoint`. Plots loss/val_loss + mae/val_mae curves. |
| 10 | **Evaluation** | Inverse-transforms predictions, computes MAE/RMSE/MAPE in USD via `all_metrics()` from `src/model/evaluate.py`, prints LSTM-vs-naive comparison. |
| 11 | **Actual vs predicted plot** | Overlays prediction series on real series for the test window. |
| 12 | **Saving** | `model.save()` + `joblib.dump(scaler)` to `models/`. |
| 13 | **Serving (reference)** | Pointer card to `src/api/` and Docker setup. Not executed in the notebook. |
| 14 | **Monitoring (reference)** | Pointer to MLflow setup + serving middleware. |

**Tips:**

- Register the venv as a Jupyter kernel once: `python -m ipykernel install --user --name baba-stock-prediction --display-name "BABA venv"`.
- For a clean re-run during a demo, use **Kernel → Restart & Run All** (takes ~2 minutes).

---

## Path B: Docker Compose (full pipeline)

The `Dockerfile` and `docker-compose.yml` package the API + a dedicated MLflow server as a portable stack. All MLflow-tracked training happens here, fired through the API's `POST /train` endpoint. Requires Docker Desktop running.

### 1. Bring up the stack

```powershell
docker compose up -d --build
docker compose ps     # both services should be Up (healthy) after ~15s
```

| Service       | Host port | Container port | Purpose                                |
| ------------- | --------- | -------------- | -------------------------------------- |
| `baba-api`    | **8010**  | 8000           | FastAPI prediction service             |
| `baba-mlflow` | **5000**  | 5000           | MLflow tracking server (SQLite-backed) |

> Host port for the API is `8010` (not `8000`) to avoid colliding with other dev containers. The container itself still listens on `8000`.

Two bind mounts in `docker-compose.yml` make the runtime data persist on the host:
- `./models:/app/models` — keeps the trained `lstm_baba.keras` + `scaler.pkl` between container restarts.
- `./data/raw:/app/data/raw` — caches `baba.parquet` so the first `/train` doesn't re-download from yfinance every time.

### 2. Train the model via the API

```powershell
# Train with all defaults (epochs=50, units=50, lookback=60, ...)
curl -X POST http://localhost:8010/train -H "Content-Type: application/json" -d '{}'

# Override any hyperparameter
curl -X POST http://localhost:8010/train -H "Content-Type: application/json" `
     -d '{"epochs": 30, "units": 64, "learning_rate": 5e-4}'
```

The endpoint blocks for ~1 minute and returns the run id, test metrics, model path, and the MLflow tracking URI used. After it returns, the API has already reloaded the freshly trained model in memory — `POST /predict` and `POST /predict/latest` start using it immediately.

Each `POST /train` invocation logs to MLflow:

- **Params**: `epochs`, `batch_size`, `units`, `dropout`, `lookback`, `learning_rate`, `patience`, `seed`.
- **Per-epoch metrics**: `loss`, `val_loss`, `mae`, `val_mae`.
- **Test metrics (USD scale)**: `test_mae`, `test_rmse`, `test_mape`.
- **Artifact**: the saved Keras model.
- **Registry entry**: `lstm-baba`, auto-incremented version (when the server supports it).

### 3. Browse MLflow

Open `http://localhost:5000` in the browser. The `baba-lstm` experiment lists every run, with metrics, params, and the model artifact.

> **Note on the model registry:** the compose image is `ghcr.io/mlflow/mlflow:v2.16.2` while the client is `mlflow>=3.x`. The `registered_model_name=` call returns 404 against the older server, gets caught by a `try/except` inside `src/model/train.py`, and is logged as a warning. Run + metrics + artifact still log fine. Bump the compose image to `ghcr.io/mlflow/mlflow:v3.12.0` if you want registry write-through.

### 4. Smoke-test the prediction endpoints

```powershell
curl http://localhost:8010/health
curl -X POST http://localhost:8010/predict/latest
```

### 5. Stop the stack

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

### `POST /train`

Kicks off a training run. Synchronous: blocks for the duration of training (~1 minute with defaults on CPU). On success, the API automatically reloads its in-process model so subsequent `/predict` calls use the freshly trained weights.

All hyperparameters are optional. Pass an empty body to train with defaults from `src/config.py:MODEL`:

```bash
curl -X POST http://localhost:8010/train \
  -H "Content-Type: application/json" \
  -d '{}'
```

Override anything you like:

```bash
curl -X POST http://localhost:8010/train \
  -H "Content-Type: application/json" \
  -d '{"epochs": 30, "units": 64, "batch_size": 64, "learning_rate": 5e-4, "dropout": 0.3}'
```

Response:

```json
{
  "status": "completed",
  "run_id": "02defbd59a01418b8e60b61b4e35dc96",
  "test_metrics": {"mae": 4.44, "rmse": 5.92, "mape": 3.50},
  "model_path": "/app/models/lstm_baba.keras",
  "duration_seconds": 67.2,
  "mlflow_tracking_uri": "http://mlflow:5000"
}
```

**Validation errors (HTTP 422):** any hyperparameter outside its bounds (e.g. `epochs <= 0`, `dropout >= 1`).

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
