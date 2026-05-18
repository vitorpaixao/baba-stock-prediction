"""Keras LSTM architecture for univariate close-price prediction."""
from __future__ import annotations

from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

from src.config import MODEL


def build_lstm(lookback: int = MODEL.lookback,
               units: int = MODEL.lstm_units,
               dropout: float = MODEL.dropout,
               learning_rate: float = MODEL.learning_rate) -> Sequential:
    model = Sequential([
        Input(shape=(lookback, 1)),
        LSTM(units, return_sequences=True),
        Dropout(dropout),
        LSTM(units),
        Dropout(dropout),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])
    return model
