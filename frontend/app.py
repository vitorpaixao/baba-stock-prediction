"""Streamlit UI for the BABA next-day close prediction API."""
from __future__ import annotations

import os
from datetime import datetime, timezone

import requests
import streamlit as st
from prometheus_client.parser import text_string_to_metric_families
from streamlit_autorefresh import st_autorefresh

API_INTERNAL = os.getenv("API_BASE_INTERNAL", "http://localhost:8000")
API_BROWSER = os.getenv("API_BASE_BROWSER", "http://localhost:8010")
MLFLOW_URL = os.getenv("MLFLOW_URL_BROWSER", "http://localhost:5000")
GRAFANA_URL = os.getenv("GRAFANA_URL_BROWSER", "http://localhost:3001")
PROM_URL = os.getenv("PROMETHEUS_URL_BROWSER", "http://localhost:9090")

st.set_page_config(page_title="BABA Stock Prediction", page_icon="📈", layout="wide")


# --- API helpers -----------------------------------------------------------

@st.cache_data(ttl=15, show_spinner=False)
def fetch_health() -> dict:
    r = requests.get(f"{API_INTERNAL}/health", timeout=5)
    r.raise_for_status()
    return r.json()


def fetch_predict_latest() -> dict:
    r = requests.post(f"{API_INTERNAL}/predict/latest", timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_predict_custom(closes: list[float]) -> dict:
    r = requests.post(f"{API_INTERNAL}/predict", json={"closes": closes}, timeout=10)
    r.raise_for_status()
    return r.json()


def fire_training(payload: dict) -> dict:
    r = requests.post(f"{API_INTERNAL}/train", json=payload, timeout=600)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=15, show_spinner=False)
def fetch_last_train_metrics() -> dict:
    """Parse Prometheus /metrics, pull last-train gauges."""
    r = requests.get(f"{API_INTERNAL}/metrics", timeout=5)
    r.raise_for_status()
    out: dict = {"mae": None, "rmse": None, "mape": None, "trained_at": None}
    for fam in text_string_to_metric_families(r.text):
        if fam.name == "baba_last_train_metric":
            for s in fam.samples:
                key = s.labels.get("metric")
                if key in out:
                    out[key] = s.value
        elif fam.name == "baba_last_train_timestamp_seconds":
            for s in fam.samples:
                if s.value > 0:
                    out["trained_at"] = datetime.fromtimestamp(s.value, tz=timezone.utc)
                break
    return out


# --- sidebar ---------------------------------------------------------------

with st.sidebar:
    st.header("Status")
    st_autorefresh(interval=15_000, key="health_poll")
    try:
        h = fetch_health()
        st.success(f"🟢 API healthy")
    except Exception as e:
        st.error(f"🔴 API unreachable\n\n{e}")
    st.caption(f"Last check: {datetime.now().strftime('%H:%M:%S')}")
    st.divider()
    st.header("Links")
    st.markdown(f"- [API docs ↗]({API_BROWSER}/docs)")
    st.markdown(f"- [MLflow ↗]({MLFLOW_URL})")
    st.markdown(f"- [Grafana ↗]({GRAFANA_URL}) (admin / admin)")
    st.markdown(f"- [Prometheus ↗]({PROM_URL})")


# --- hero: next-day prediction ---------------------------------------------

st.title("📈 BABA next-day close prediction")

hero = st.container(border=True)
with hero:
    if "prediction" not in st.session_state:
        try:
            st.session_state["prediction"] = fetch_predict_latest()
        except Exception as e:
            st.session_state["prediction"] = {"error": str(e)}

    p = st.session_state["prediction"]
    if "error" in p:
        st.error(f"Could not fetch prediction: {p['error']}")
    else:
        col1, col2, col3 = st.columns([2, 1, 1])
        col1.metric(
            label="Predicted close (next trading day)",
            value=f"${p['predicted_close']:.2f}",
        )
        col2.caption(f"window\n{p['window_start']} → {p['window_end']}")
        col3.caption(f"inference\n{p['inference_ms']} ms")

    if st.button("↻ Refresh prediction"):
        try:
            st.session_state["prediction"] = fetch_predict_latest()
        except Exception as e:
            st.session_state["prediction"] = {"error": str(e)}
        st.rerun()


# --- last-train metrics card -----------------------------------------------

st.subheader("Last training metrics")
try:
    m = fetch_last_train_metrics()
    if m["mae"] is None:
        st.info("No training metrics yet. Train a model first using the form below.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE (USD)", f"${m['mae']:.2f}")
        c2.metric("RMSE (USD)", f"${m['rmse']:.2f}")
        c3.metric("MAPE (%)", f"{m['mape']:.2f}%")
        if m["trained_at"]:
            c4.metric("Trained at (UTC)", m["trained_at"].strftime("%Y-%m-%d %H:%M"))
except Exception as e:
    st.warning(f"Could not read /metrics: {e}")


# --- train form ------------------------------------------------------------

with st.expander("Train a new model", expanded=False):
    with st.form("train_form"):
        c1, c2 = st.columns(2)
        epochs = c1.number_input("epochs", min_value=1, max_value=500, value=50)
        units = c2.number_input("units", min_value=1, max_value=512, value=50)
        batch_size = c1.number_input("batch_size", min_value=1, max_value=512, value=32)
        learning_rate = c2.number_input(
            "learning_rate", min_value=1e-5, max_value=1.0, value=1e-3, format="%.5f"
        )
        dropout = c1.slider("dropout", min_value=0.0, max_value=0.9, value=0.2, step=0.05)
        lookback = c2.number_input("lookback", min_value=10, max_value=240, value=60)
        patience = c1.number_input("patience", min_value=0, max_value=100, value=10)
        seed = c2.number_input("seed", min_value=0, max_value=100000, value=42)
        submitted = st.form_submit_button("▶ Start training (~1 min)")

        if submitted:
            payload = dict(
                epochs=int(epochs),
                units=int(units),
                batch_size=int(batch_size),
                learning_rate=float(learning_rate),
                dropout=float(dropout),
                lookback=int(lookback),
                patience=int(patience),
                seed=int(seed),
            )
            with st.spinner("Training in progress, this blocks for ~60 s..."):
                try:
                    res = fire_training(payload)
                    st.success(
                        f"Done in {res['duration_seconds']}s. "
                        f"Run id: `{res['run_id']}` · "
                        f"Logged to {res['mlflow_tracking_uri']}"
                    )
                    st.json(res["test_metrics"])
                    fetch_last_train_metrics.clear()
                except Exception as e:
                    st.error(f"Training failed: {e}")


# --- manual /predict -------------------------------------------------------

with st.expander("Manual prediction with a custom 60-day window", expanded=False):
    raw = st.text_area(
        "Paste 60 comma-separated closing prices (oldest first):",
        height=120,
        placeholder="100.12, 100.40, 101.05, ...",
    )
    if st.button("Predict from this window"):
        try:
            closes = [
                float(x.strip())
                for x in raw.replace("\n", ",").split(",")
                if x.strip()
            ]
            if len(closes) != 60:
                st.error(f"Need exactly 60 values, got {len(closes)}.")
            else:
                res = fetch_predict_custom(closes)
                st.metric("Predicted close", f"${res['predicted_close']:.2f}")
                st.caption(f"inference: {res['inference_ms']} ms")
        except ValueError as e:
            st.error(f"Parse error: {e}")
        except Exception as e:
            st.error(f"API error: {e}")
