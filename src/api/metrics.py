"""Prometheus metric singletons for the BABA prediction API.

HTTP-level metrics (request count, duration, in-flight) come from
``prometheus_fastapi_instrumentator`` automatically. The metrics defined here
are domain-specific: prediction throughput, inference latency, drift watch,
and training observability.
"""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# --- Tier 2: serving --------------------------------------------------------

PREDICTION_COUNT = Counter(
    "baba_predictions_total",
    "Total predictions served.",
    ["endpoint"],  # /predict, /predict/latest
)

INFERENCE_DURATION = Histogram(
    "baba_inference_duration_seconds",
    "Pure TF forward-pass time, excluding request overhead.",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

PREDICTED_VALUE = Histogram(
    "baba_predicted_close_usd",
    "Distribution of predicted close prices.",
    buckets=(50, 75, 100, 125, 150, 175, 200, 250, 300, 400),
)

MODEL_LOADED = Gauge(
    "baba_model_loaded",
    "1 if the predictor is in memory, else 0.",
)

# --- Tier 3: drift watch ----------------------------------------------------

INPUT_WINDOW_MEAN = Histogram(
    "baba_input_window_mean_usd",
    "Mean of the 60-day input window passed to /predict.",
    buckets=(50, 75, 100, 125, 150, 175, 200, 250, 300, 400),
)

INPUT_WINDOW_STD = Histogram(
    "baba_input_window_std_usd",
    "Std of the 60-day input window passed to /predict.",
    buckets=(1, 2.5, 5, 10, 20, 40, 80),
)

# --- Tier 4: training observability -----------------------------------------

TRAIN_COUNT = Counter(
    "baba_train_runs_total",
    "Number of POST /train invocations that completed successfully.",
)

TRAIN_DURATION = Histogram(
    "baba_train_duration_seconds",
    "Wall-clock duration of POST /train.",
    buckets=(30, 60, 120, 300, 600, 1200, 3600),
)

LAST_TRAIN_TS = Gauge(
    "baba_last_train_timestamp_seconds",
    "Unix timestamp of the last successful training run.",
)

LAST_TRAIN_METRIC = Gauge(
    "baba_last_train_metric",
    "Test-set metric of the last training run (mae/rmse in USD, mape in %).",
    ["metric"],
)
