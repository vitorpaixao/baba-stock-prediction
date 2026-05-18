"""Download historical OHLCV data from Yahoo Finance and persist as Parquet.

Usage:
    python -m src.data.fetch              # idempotent, skip if file exists
    python -m src.data.fetch --force      # always re-download
"""
from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd
import yfinance as yf

from src.config import DATA, RAW_PARQUET

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def fetch(symbol: str = DATA.symbol,
          start: str = DATA.start_date,
          end: str = DATA.end_date,
          force: bool = False) -> pd.DataFrame:
    if RAW_PARQUET.exists() and not force:
        log.info("Cached file at %s — use --force to refetch.", RAW_PARQUET)
        return pd.read_parquet(RAW_PARQUET)

    log.info("Downloading %s from %s to %s", symbol, start, end)
    df = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError(f"Empty dataframe for {symbol} ({start}..{end})")

    # yfinance may return MultiIndex columns (e.g. ('Close','BABA')) — flatten.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df.index.name = "Date"
    df = df.reset_index()
    df.to_parquet(RAW_PARQUET, index=False)
    log.info("Saved %d rows → %s", len(df), RAW_PARQUET)
    return df


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fetch BABA historical OHLCV from yfinance.")
    p.add_argument("--symbol", default=DATA.symbol)
    p.add_argument("--start", default=DATA.start_date)
    p.add_argument("--end", default=DATA.end_date)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    df = fetch(args.symbol, args.start, args.end, force=args.force)
    print(df.tail())
    return 0


if __name__ == "__main__":
    sys.exit(main())
