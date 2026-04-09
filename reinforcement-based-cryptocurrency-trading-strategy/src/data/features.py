"""
Technical indicator computation: SMA, RSI, momentum.

Design notes:
- SMA-50 requires 50 days of look-back. With separate train/val/test files,
  val and test would produce NaN for their first rows if computed in isolation.
- Strategy: compute features on the FULL combined dataset (train+val+test),
  then split by date. This ensures every row has valid indicator values.
- After feature computation, drop the warm-up rows (first max(sma_long - 1, ...)
  NaN rows) from the front of the training set only.
"""
import pandas as pd
import numpy as np


def add_sma(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Add a Simple Moving Average column over `close` price.

    Column added: sma_{window}  (e.g. sma_10, sma_50)
    First (window - 1) rows will be NaN — handle with drop_warmup().
    """
    df = df.copy()
    df[f"sma_{window}"] = df["close"].rolling(window=window).mean()
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Add RSI (Relative Strength Index) over `close` price.

    Column added: rsi
    Range: 0–100.  Values above 70 → overbought, below 30 → oversold.
    First (period) rows will be NaN — handle with drop_warmup().

    Uses Wilder's smoothed method (standard RSI definition).
    """
    df = df.copy()
    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder smoothing (equivalent to EWM with alpha = 1/period)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def add_momentum(df: pd.DataFrame, period: int = 5) -> pd.DataFrame:
    """
    Add price momentum: percentage change of `close` over `period` days.

    Column added: momentum_{period}
    Positive → price rising over the window; negative → falling.
    First (period) rows will be NaN — handle with drop_warmup().
    """
    df = df.copy()
    df[f"momentum_{period}"] = df["close"].pct_change(periods=period) * 100
    return df


def drop_warmup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop leading rows that contain NaN from indicator warm-up periods.

    Should only be called on the training set — val/test rows are fully
    populated because features are computed on the combined dataset first.
    """
    n_before = len(df)
    df = df.dropna()
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"drop_warmup: removed {n_dropped} warm-up row(s) from the front.")
    return df


def build_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Compute all technical indicators on a DataFrame and return enriched copy.

    Expected config keys (from experiment_config.yaml):
        sma_short   : int  (e.g. 10)
        sma_long    : int  (e.g. 50)
        rsi_period  : int  (e.g. 14)

    Momentum period is fixed at 5 days (rate of change over 1 week).

    IMPORTANT: Call this on the full combined dataset (train+val+test) so
    that SMA-50 warm-up rows are in the training block and val/test rows
    all have valid values. Then split by date afterwards.
    """
    sma_short  = config.get("sma_short", 10)
    sma_long   = config.get("sma_long", 50)
    rsi_period = config.get("rsi_period", 14)

    df = add_sma(df, window=sma_short)
    df = add_sma(df, window=sma_long)
    df = add_rsi(df, period=rsi_period)
    df = add_momentum(df, period=5)

    feature_cols = [f"sma_{sma_short}", f"sma_{sma_long}", "rsi", "momentum_5"]
    n_nan = df[feature_cols].isna().any(axis=1).sum()
    print(
        f"build_features: added {feature_cols}  |  "
        f"{n_nan} warm-up row(s) contain NaN (drop from train before use)"
    )

    return df
