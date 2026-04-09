"""
Data loader for Binance Vision OHLCV CSV files.

Binance Vision daily kline columns (no header row):
  0  open_time               Unix timestamp (candle open) — see note below
  1  open                    Open price (USDT)
  2  high                    High price (USDT)
  3  low                     Low price (USDT)
  4  close                   Close price (USDT)
  5  volume                  BTC volume traded
  6  close_time              Unix timestamp (candle close)
  7  quote_volume            USDT volume traded
  8  trades                  Number of trades
  9  taker_buy_base_volume   BTC bought by takers
  10 taker_buy_quote_volume  USDT spent by takers
  11 ignore                  Unused (always 0)

Only columns 0-5 are retained after loading.

Timestamp format change (Binance Vision):
  Files up to 2024-12  → 13-digit milliseconds  (e.g. 1733011200000)
  Files from 2025-01   → 16-digit microseconds  (e.g. 1735689600000000)
  The loader detects this automatically and normalises to milliseconds.
"""
import pandas as pd
from pathlib import Path
from typing import Union, List


# Binance Vision column names (positional, no header in file)
_ALL_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
]

# Only these columns are kept for the project
OHLCV_COLUMNS = ["open_time", "open", "high", "low", "close", "volume"]


def load_raw_csv(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load a single Binance Vision daily kline CSV.

    Returns a DataFrame with columns: open_time (DatetimeIndex),
    open, high, low, close, volume — sorted ascending by date.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"CSV not found: {filepath}")

    df = pd.read_csv(
        filepath,
        header=None,
        names=_ALL_COLUMNS,
        usecols=OHLCV_COLUMNS,
        dtype={
            "open_time": "int64",
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
        },
    )

    # Normalise timestamp: Binance switched from ms (13 digits) to
    # µs (16 digits) starting from 2025-01 files.
    if df["open_time"].iloc[0] > 1e15:
        df["open_time"] = df["open_time"] // 1000  # µs → ms

    # Convert Unix ms timestamp to UTC datetime and use as index
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time").sort_index()

    return df


def load_multiple(filepaths: List[Union[str, Path]]) -> pd.DataFrame:
    """
    Load and concatenate multiple monthly CSV files in chronological order.

    Args:
        filepaths: List of CSV paths, e.g. from experiment_config.yaml raw_files.

    Returns:
        Single concatenated DataFrame sorted by date with no duplicates.
    """
    frames = [load_raw_csv(fp) for fp in filepaths]
    df = pd.concat(frames).sort_index()

    # Drop any accidental duplicate timestamps (e.g. month boundary overlap)
    df = df[~df.index.duplicated(keep="first")]

    return df


def validate_continuity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check for missing dates in the daily time series.

    Prints a warning for each gap found. Daily crypto markets run 24/7
    so every calendar day should have a candle.

    Returns the original DataFrame unchanged (gaps are reported, not dropped).
    """
    expected = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="D",
        tz="UTC",
    )
    missing = expected.difference(df.index)

    if missing.empty:
        print(f"Continuity check passed: {len(df)} rows, no missing dates.")
    else:
        print(f"WARNING: {len(missing)} missing date(s) detected:")
        for d in missing:
            print(f"  {d.date()}")

    return df
