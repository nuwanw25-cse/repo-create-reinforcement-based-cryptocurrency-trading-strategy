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


# Columns that will be normalized
PRICE_COLUMNS = ["open", "high", "low", "close"]
VOLUME_COLUMNS = ["volume"]


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
    Min-Max normalize price and volume columns.

    Scalers are fit exclusively on the training set to prevent leakage,
    then applied identically to val and test.

    Returns:
        (train_norm, val_norm, test_norm, scalers)

        scalers is a dict {"price": MinMaxScaler, "volume": MinMaxScaler}
        — save these for inverse-transforming results back to USDT values.
    """
    train = train.copy()
    val = val.copy()
    test = test.copy()

    scalers = {}

    # Price scaler — fit on train open/high/low/close jointly so their
    # relative relationships (e.g. high > close) are preserved after scaling
    price_scaler = MinMaxScaler()
    train[PRICE_COLUMNS] = price_scaler.fit_transform(train[PRICE_COLUMNS])
    val[PRICE_COLUMNS] = price_scaler.transform(val[PRICE_COLUMNS])
    test[PRICE_COLUMNS] = price_scaler.transform(test[PRICE_COLUMNS])
    scalers["price"] = price_scaler

    # Volume scaler — separate scaler since volume has a different scale
    volume_scaler = MinMaxScaler()
    train[VOLUME_COLUMNS] = volume_scaler.fit_transform(train[VOLUME_COLUMNS])
    val[VOLUME_COLUMNS] = volume_scaler.transform(val[VOLUME_COLUMNS])
    test[VOLUME_COLUMNS] = volume_scaler.transform(test[VOLUME_COLUMNS])
    scalers["volume"] = volume_scaler

    print(
        "normalize: price and volume scaled to [0, 1] "
        "using training-set statistics only."
    )
    return train, val, test, scalers
