"""
Data preprocessing: cleaning, train/val/test splitting, and normalization.

Design notes:
- Split is always chronological (no shuffling) to prevent data leakage.
- Normalization is fit ONLY on the training set, then applied to val/test.
  This is critical: fitting on the full dataset would leak future price
  information into the training process.
- The scalers dict is returned so it can be saved and reused at inference time.
"""
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler


# Return-based features: MinMaxScaler fit on training fractional returns
RETURN_COLS = ["close_return", "open_gap", "high_dev", "low_dev"]

# Log-volume: MinMaxScaler fit on training log1p(volume)
LOG_VOL_COL = ["log_volume"]

# SMA deviations: clip to symmetric range then shift to [0, 1] — no scaler needed
# key = column name, value = half-range for clipping (e.g. 0.2 → clip to ±0.2)
SMA_DEV_PARAMS = {
    "sma_10_dev": 0.2,
    "sma_50_dev": 0.3,
}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean the raw OHLCV DataFrame.

    Checks performed:
    - Drop rows with any NaN values (should be none for Binance data)
    - Assert OHLCV sanity: high >= low, high >= open/close, low <= open/close
    - Report extreme close-price outliers (>5x IQR) — flagged but not removed

    Returns a cleaned copy; raises ValueError on critical OHLCV violations.
    """
    df = df.copy()

    # 1. Drop NaN rows
    n_before = len(df)
    df = df.dropna()
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"clean: dropped {n_dropped} row(s) with NaN values.")

    # 2. Sanity check OHLCV relationships
    violations = (
        (df["high"] < df["low"]) |
        (df["high"] < df["open"]) |
        (df["high"] < df["close"]) |
        (df["low"] > df["open"]) |
        (df["low"] > df["close"])
    )
    if violations.any():
        bad_dates = df.index[violations].tolist()
        raise ValueError(
            f"OHLCV sanity check failed on {len(bad_dates)} row(s): {bad_dates}"
        )

    # 3. Flag extreme close-price outliers (informational, not dropped)
    q1 = df["close"].quantile(0.25)
    q3 = df["close"].quantile(0.75)
    iqr = q3 - q1
    outliers = df[
        (df["close"] < q1 - 5 * iqr) | (df["close"] > q3 + 5 * iqr)
    ]
    if not outliers.empty:
        print(
            f"clean: {len(outliers)} potential close-price outlier(s) flagged (not removed):"
        )
        for d in outliers.index:
            print(f"  {d.date()}  close={outliers.loc[d, 'close']:.2f}")

    print(f"clean: {len(df)} rows retained.")
    return df


def split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological train / validation / test split.

    The remaining ratio after train + val becomes the test set.
    No shuffling — temporal order is always preserved.

    Returns:
        (train_df, val_df, test_df)
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    print(
        f"split: train={len(train)} rows "
        f"({train.index[0].date()} → {train.index[-1].date()})  "
        f"val={len(val)} rows "
        f"({val.index[0].date()} → {val.index[-1].date()})  "
        f"test={len(test)} rows "
        f"({test.index[0].date()} → {test.index[-1].date()})"
    )

    return train, val, test


def normalize(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Normalize return-based and log-volume features to [0, 1].

    All scalers are fit exclusively on the training set to prevent leakage,
    then applied identically to val and test.

    Features normalized:
        RETURN_COLS   — fractional return features; MinMaxScaler on training
        LOG_VOL_COL   — log1p(volume); MinMaxScaler on training log-volume
        SMA_DEV_PARAMS — SMA deviations; clip to symmetric range, shift to [0,1]

    Raw OHLCV columns (open, high, low, close, volume) and indicator columns
    (rsi, momentum_5) are left unchanged in the output.

    Returns:
        (train_norm, val_norm, test_norm, scalers)

        scalers is a dict {"returns": MinMaxScaler, "log_volume": MinMaxScaler}
    """
    train = train.copy()
    val = val.copy()
    test = test.copy()

    scalers = {}

    # Return-based features (close_return, open_gap, high_dev, low_dev)
    present_return = [c for c in RETURN_COLS if c in train.columns]
    if present_return:
        returns_scaler = MinMaxScaler()
        train[present_return] = returns_scaler.fit_transform(train[present_return])
        val[present_return]   = returns_scaler.transform(val[present_return])
        test[present_return]  = returns_scaler.transform(test[present_return])
        scalers["returns"] = returns_scaler

    # Log-volume
    present_log_vol = [c for c in LOG_VOL_COL if c in train.columns]
    if present_log_vol:
        log_vol_scaler = MinMaxScaler()
        train[present_log_vol] = log_vol_scaler.fit_transform(train[present_log_vol])
        val[present_log_vol]   = log_vol_scaler.transform(val[present_log_vol])
        test[present_log_vol]  = log_vol_scaler.transform(test[present_log_vol])
        scalers["log_volume"] = log_vol_scaler

    # SMA deviations: clip to ±half_range then shift to [0, 1]
    for col, half_range in SMA_DEV_PARAMS.items():
        if col in train.columns:
            for ds in [train, val, test]:
                ds[col] = (ds[col].clip(-half_range, half_range) + half_range) / (2 * half_range)

    print("normalize: return features and log-volume scaled to [0, 1] using training-set statistics.")
    return train, val, test, scalers
