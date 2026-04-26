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


def add_returns_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add scale-invariant return and deviation features derived from OHLCV.

    These features are stationary and do not depend on the absolute price
    level, so they generalise to price regimes unseen during training.

    Columns added:
        close_return  — per-step % price change; NaN on row 0
        open_gap      — (open − prev_close) / prev_close; NaN on row 0
        high_dev      — (high − close) / close; always >= 0
        low_dev       — (low  − close) / close; always <= 0
        log_volume    — log1p(volume); log-compresses the heavy right tail
    """
    df = df.copy()
    df["close_return"] = df["close"].pct_change()
    df["open_gap"]     = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)
    df["high_dev"]     = (df["high"] - df["close"]) / df["close"]
    df["low_dev"]      = (df["low"]  - df["close"]) / df["close"]
    df["log_volume"]   = np.log1p(df["volume"])
    return df


def add_sma_deviations(
    df: pd.DataFrame,
    sma_short: int = 10,
    sma_long: int = 50,
) -> pd.DataFrame:
    """
    Add SMA deviation features: (close − sma) / close.

    Must be called after add_sma() so sma_{window} columns already exist.

    Columns added: sma_{sma_short}_dev, sma_{sma_long}_dev
    Positive → close above the moving average (bullish); negative → below.
    """
    df = df.copy()
    df[f"sma_{sma_short}_dev"] = (df["close"] - df[f"sma_{sma_short}"]) / df["close"]
    df[f"sma_{sma_long}_dev"]  = (df["close"] - df[f"sma_{sma_long}"])  / df["close"]
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
    df = add_returns_features(df)
    df = add_sma_deviations(df, sma_short=sma_short, sma_long=sma_long)

    all_indicator_cols = [
        f"sma_{sma_short}", f"sma_{sma_long}", "rsi", "momentum_5",
        "close_return", "open_gap", "high_dev", "low_dev", "log_volume",
        f"sma_{sma_short}_dev", f"sma_{sma_long}_dev",
    ]
    n_nan = df[all_indicator_cols].isna().any(axis=1).sum()
    print(
        f"build_features: added {len(all_indicator_cols)} indicator columns  |  "
        f"{n_nan} warm-up row(s) contain NaN (drop from train before use)"
    )

    return df
