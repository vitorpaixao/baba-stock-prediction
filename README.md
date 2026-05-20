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

This is the deliverable for **Tech Challenge Fase 4** of the MLET pós-graduação (worth 90% of the phase grade). The goal is to go end-to-end on a deep-learning system: **collect → preprocess → train an LSTM → save the model → serve it through a REST API → monitor it in production**.

Spec: [`plan/plan.md`](plan/plan.md) · Implementation check: [`plan/challenge_check.md`](plan/challenge_check.md) · Concepts & defense notes: [`plan/challenge_process.md`](plan/challenge_process.md)

---

## At a glance

| | |
| --- | --- |
| **Ticker** | BABA — Alibaba Group (NYSE) |
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

## Quickstart — Local (recommended for demo)

The **fastest path** is local Python. The notebook at [`notebooks/01_eda_lstm_baba.ipynb`](notebooks/01_eda_lstm_baba.ipynb) runs the entire pipeline end-to-end in ~2 minutes — coleta, modelo LSTM, treino, avaliação, salvamento. See [Exploring with the notebook](#exploring-with-the-notebook) below.

For a video defense, follow [`plan/demo_script.md`](plan/demo_script.md) — an ~11 min step-by-step that uses the notebook as the centerpiece.

---

## Quickstart — Docker Compose (deployment artifact)

The `Dockerfile` and `docker-compose.yml` package the API + MLflow as a portable stack — useful to demonstrate MLOps maturity for the spec (Fase 4 / 5), though for a live demo the local path above is more reliable. Requires Docker Desktop running. The trained model + scaler must already exist locally under `models/`; if not, run the local training step (one command) below first.

```bash
docker compose up -d --build

# API           → http://localhost:8010/docs
# MLflow UI     → http://localhost:5000

# smoke test
curl http://localhost:8010/health
curl -X POST http://localhost:8010/predict/latest
```

| Service       | Host port | Container port | Purpose                                 |
| ------------- | --------- | -------------- | --------------------------------------- |
| `baba-api`    | **8010**  | 8000           | FastAPI prediction service              |
| `baba-mlflow` | **5000**  | 5000           | MLflow tracking server (SQLite-backed)  |

> The host port for the API is `8010` (not `8000`) to avoid colliding with other dev containers. The container itself still listens on 8000.

---

## Quickstart — Local

```bash
# 1. Python 3.11 or 3.12 (TensorFlow 2.18 requirement)
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # Linux/macOS

# 2. Install
pip install -e ".[dev]"

# 3. Fetch BABA history into data/raw/baba.parquet (idempotent)
python -m src.data.fetch

# 4. Train the LSTM, log to ./mlruns, save models/lstm_baba.keras + scaler.pkl
python -m src.model.train

# 5. Serve
uvicorn src.api.main:app --reload
# → http://localhost:8000/docs
```

Training takes ~1 minute on CPU. Tune via CLI flags:

```bash
python -m src.model.train --epochs 100 --units 64 --batch-size 64 --lr 5e-4
```

---

## Exploring with the notebook

The Jupyter notebook at [`notebooks/01_eda_lstm_baba.ipynb`](notebooks/01_eda_lstm_baba.ipynb) is the **primary demo of the project**. It runs the entire pipeline end-to-end — covering **Fases 1, 2 and 3** of the challenge spec — by importing the same functions used by the production code in `src/`. No duplication; the notebook is a narrated wrapper around the modular pipeline.

Use it as the starting point if you want to understand the project, **and** as the script for the video defense (see [`plan/demo_script.md`](plan/demo_script.md)).

**Launch it:**

```bash
.venv\Scripts\activate
jupyter lab notebooks/01_eda_lstm_baba.ipynb
```

**What's inside (run cells top-to-bottom — ~2 minutes total):**

| # | Section | What it shows |
| - | --- | --- |
| 1 | **Imports + config** | Loads `src/config.py` constants (ticker, lookback, hyperparams). |
| 2 | **Fase 1.1 — Coleta** | Reads `data/raw/baba.parquet` (auto-fetches if missing). 2,800+ trading days. |
| 3 | **EDA — Preço + volume** | Two-panel chart: BABA Close (top) and Volume (bottom). |
| 4 | **EDA — Retornos diários** | Close + 60-day rolling mean, plus daily returns. Visual proof prices are non-stationary, returns are not. |
| 5 | **Baseline ingênuo** | MAE / RMSE / MAPE for "predict yesterday's close". The floor any model must beat. |
| 6 | **Fase 1.2 — Pré-processamento** | Calls `build_splits()` from `src/data/preprocess.py`; shows shapes and scaler bounds. |
| 7 | **Fase 2.1 — Construção** | Calls `build_lstm()` from `src/model/architecture.py`; renders `model.summary()`. |
| 8 | **Fase 2.2 — Treinamento** | Real training loop (~30 epochs, ~1 min on CPU) with `EarlyStopping` + `ModelCheckpoint`. Plots loss/val_loss + mae/val_mae curves. |
| 9 | **Fase 2.3 — Avaliação** | Inverse-transforms predictions, computes MAE/RMSE/MAPE in USD via `all_metrics()` from `src/model/evaluate.py`, prints LSTM-vs-naive comparison. |
| 10 | **Actual vs predicted plot** | Overlays prediction series on real series for the test window. |
| 11 | **Fase 3 — Salvamento** | `model.save()` + `joblib.dump(scaler)` to `models/`. |
| 12 | **Fase 4 — Deploy (referência)** | Pointer card to `src/api/` and Docker setup. Not executed in the notebook. |
| 13 | **Fase 5 — Monitoring (referência)** | Pointer to MLflow setup + serving middleware. |

**Tips:**

- Install the package first (`pip install -e ".[dev]"`) and register the venv as a Jupyter kernel: `python -m ipykernel install --user --name baba-stock-prediction --display-name "BABA venv"`.
- For a clean re-run during a demo, use **Kernel → Restart & Run All** — takes ~2 minutes.
- Want MLflow tracking on top? Run `python -m src.model.train` in a terminal — that's the same model but with full MLflow logging to `./mlruns/`.

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

## Monitoring

### Training-time — MLflow

Every `python -m src.model.train` invocation creates a run with:

- **Params** — `epochs`, `batch_size`, `units`, `dropout`, `lookback`, `learning_rate`, `patience`, `seed`.
- **Per-epoch metrics** — `loss`, `val_loss`, `mae`, `val_mae`.
- **Test metrics (USD scale)** — `test_mae`, `test_rmse`, `test_mape`.
- **Artifact** — the saved Keras model.
- **Registry entry** — `lstm-baba`, auto-incremented version.

Open the UI:

```bash
# local file backend (recommended for demo/dev):
mlflow ui --backend-store-uri ./mlruns --port 5050
# → http://localhost:5050

# or, if the compose stack is up:
# → http://localhost:5000 (separate sqlite-backed server)
```

> **Tip:** use port `5050` for the local UI to avoid colliding with the compose MLflow on `5000`.

### Serving-time — JSON logs + `/metrics`

Each request emits one line of JSON to stdout (parseable with `jq`, Loki, ELK, Datadog):

```json
{"request_id": "9f2c...", "method": "POST", "path": "/predict",
 "status": 200, "latency_ms": 12.34}
```

`AccessLogMiddleware` also feeds a 1,000-entry ring buffer that backs the `/metrics` endpoint (p50/p95/p99/mean + per-route counts). Lightweight, no external dependencies — easy to swap for Prometheus later.

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
├── notebooks/01_eda_lstm_baba.ipynb    # full pipeline demo — start here
├── tests/                              # 12 pytest cases
├── models/                             # lstm_baba.keras, scaler.pkl (gitignored)
├── data/raw/baba.parquet               # (gitignored)
├── Dockerfile                          # python:3.12-slim, healthcheck
├── docker-compose.yml                  # api + mlflow services
├── pyproject.toml                      # dependencies
└── plan/                               # spec + delivery check + concept guide
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

Tests run in ~30 seconds without hitting yfinance or training a real model (they use a synthetic sine-wave dataframe — see `tests/conftest.py`).

---

## Honest caveats

- **Univariate.** Close only. Adding `Volume` and macro features (multivariate input) is the natural next step.
- **One-step-ahead.** Multi-day rollouts compound error fast.
- **Price vs. returns.** Stock prices are non-stationary; predicting **returns** (Δp/p) is more rigorous than predicting prices, even when MAPE looks pretty.
- **Single hold-out.** A walk-forward / expanding-window backtest gives a much more robust estimate of out-of-sample error.
- **Regression accuracy ≠ trading utility.** For real money you'd care about *direction accuracy* and *Sharpe of the strategy on top*, not MAPE.

---

## Further reading inside this repo

| Doc | What it's for |
| --- | --- |
| [`plan/plan.md`](plan/plan.md) | Original tech-challenge spec (Portuguese). |
| [`plan/challenge_check.md`](plan/challenge_check.md) | Spec → delivered mapping with file pointers. Use this for grading. |
| [`plan/challenge_process.md`](plan/challenge_process.md) | Concepts, decisions, and metric definitions — defense notes for the board. |
| [`plan/demo_script.md`](plan/demo_script.md) | Step-by-step ~11 min script for the video defense (notebook-first, Docker as backup). |
| [`plan/podcast_script.md`](plan/podcast_script.md) | 4-episode podcast script (PT-BR) covering the project end-to-end. |

---

<div align="center">

Tech Challenge Fase 4 — MLET pós-graduação · Built with TensorFlow, FastAPI, MLflow, Docker.

</div>
