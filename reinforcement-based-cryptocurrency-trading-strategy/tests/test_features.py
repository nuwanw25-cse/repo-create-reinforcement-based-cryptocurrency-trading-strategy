"""Unit tests for technical indicator computation (src/data/features.py)."""
import numpy as np
import pandas as pd
import pytest

from src.data.features import add_sma, add_rsi, add_momentum, build_features, drop_warmup


# --------------------------------------------------------------------------- #
# Fixture: synthetic OHLCV DataFrame
# --------------------------------------------------------------------------- #
@pytest.fixture
def ohlcv_df():
    """100-row synthetic OHLCV DataFrame with a simple linearly rising close."""
    n = 100
    close = np.linspace(50_000, 60_000, n)
    return pd.DataFrame({
        "open":   close * 0.999,
        "high":   close * 1.001,
        "low":    close * 0.998,
        "close":  close,
        "volume": np.random.default_rng(42).uniform(100, 1000, n),
    }, index=pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"))


@pytest.fixture
def flat_df():
    """50-row OHLCV with completely flat close — RSI edge case."""
    n = 50
    close = np.full(n, 60_000.0)
    return pd.DataFrame({
        "open": close, "high": close, "low": close, "close": close,
        "volume": np.ones(n),
    }, index=pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"))


# --------------------------------------------------------------------------- #
# SMA tests
# --------------------------------------------------------------------------- #

def test_sma_column_exists(ohlcv_df):
    df = add_sma(ohlcv_df.copy(), window=10)
    assert "sma_10" in df.columns


def test_sma_values(ohlcv_df):
    """SMA-10 at index 9 should equal the mean of rows 0–9."""
    df = add_sma(ohlcv_df.copy(), window=10)
    expected = ohlcv_df["close"].iloc[:10].mean()
    assert df["sma_10"].iloc[9] == pytest.approx(expected, rel=1e-9)


def test_sma_warmup_is_nan(ohlcv_df):
    """First (window-1) rows of SMA should be NaN."""
    window = 10
    df = add_sma(ohlcv_df.copy(), window=window)
    assert df["sma_10"].iloc[:window - 1].isna().all()
    assert not np.isnan(df["sma_10"].iloc[window - 1])


def test_sma_monotone_on_rising_price(ohlcv_df):
    """SMA should be monotonically non-decreasing on a rising price series."""
    df = add_sma(ohlcv_df.copy(), window=10)
    valid = df["sma_10"].dropna()
    assert (valid.diff().dropna() >= 0).all()


# --------------------------------------------------------------------------- #
# RSI tests
# --------------------------------------------------------------------------- #

def test_rsi_column_exists(ohlcv_df):
    df = add_rsi(ohlcv_df.copy(), period=14)
    assert "rsi" in df.columns


def test_rsi_range(ohlcv_df):
    """RSI must always be in [0, 100] for all non-NaN values."""
    df = add_rsi(ohlcv_df.copy(), period=14)
    valid = df["rsi"].dropna()
    assert (valid >= 0).all(), f"RSI below 0: {valid.min()}"
    assert (valid <= 100).all(), f"RSI above 100: {valid.max()}"


def test_rsi_rising_market_above_50(ohlcv_df):
    """On a steadily rising price series RSI should settle above 50."""
    df = add_rsi(ohlcv_df.copy(), period=14)
    # Drop the warm-up rows and check the settled part
    settled = df["rsi"].iloc[30:]
    assert (settled.dropna() > 50).all(), "RSI should be > 50 on rising prices"


def test_rsi_flat_market(flat_df):
    """On a flat price series (no change) RSI should be NaN or 50."""
    df = add_rsi(flat_df.copy(), period=14)
    valid = df["rsi"].dropna()
    # With no price change, gains=losses=0. Implementation may produce NaN or 50.
    if len(valid) > 0:
        assert ((valid == 50) | valid.isna()).all() or valid.isna().all()


# --------------------------------------------------------------------------- #
# Momentum tests
# --------------------------------------------------------------------------- #

def test_momentum_column_exists(ohlcv_df):
    df = add_momentum(ohlcv_df.copy(), period=5)
    assert "momentum_5" in df.columns


def test_momentum_sign(ohlcv_df):
    """Momentum should be positive on a rising price series."""
    df = add_momentum(ohlcv_df.copy(), period=5)
    valid = df["momentum_5"].dropna()
    assert (valid > 0).all(), "Momentum should be positive on rising prices"


def test_momentum_is_pct_change(ohlcv_df):
    """momentum_5[i] should equal (close[i]/close[i-5] - 1) * 100."""
    df = add_momentum(ohlcv_df.copy(), period=5)
    i = 20
    expected = (ohlcv_df["close"].iloc[i] / ohlcv_df["close"].iloc[i - 5] - 1) * 100
    assert df["momentum_5"].iloc[i] == pytest.approx(expected, rel=1e-9)


# --------------------------------------------------------------------------- #
# build_features / drop_warmup
# --------------------------------------------------------------------------- #

def test_build_features_adds_all_columns(ohlcv_df):
    config = {"sma_short": 10, "sma_long": 50, "rsi_period": 14}
    df = build_features(ohlcv_df.copy(), config)
    for col in ["sma_10", "sma_50", "rsi", "momentum_5"]:
        assert col in df.columns, f"Expected column {col!r} not found"


def test_drop_warmup_removes_nan_rows(ohlcv_df):
    config = {"sma_short": 10, "sma_long": 50, "rsi_period": 14}
    df = build_features(ohlcv_df.copy(), config)
    df_clean = drop_warmup(df)
    assert not df_clean.isnull().any().any(), "No NaN should remain after drop_warmup"


def test_drop_warmup_preserves_tail():
    """Tail rows should be identical before and after drop_warmup.
    Uses 200 rows of realistic (noisy) prices so RSI has both gains and losses."""
    n = 200
    rng = np.random.default_rng(42)
    # Brownian-motion-like price: mix of up and down days → RSI has valid values
    daily_returns = rng.normal(0.001, 0.015, n)
    close = 50_000 * np.cumprod(1 + daily_returns)
    df200 = pd.DataFrame({
        "open": close * 0.999, "high": close * 1.01,
        "low":  close * 0.99,  "close": close,
        "volume": np.ones(n),
    }, index=pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"))

    config = {"sma_short": 10, "sma_long": 50, "rsi_period": 14}
    df_feat  = build_features(df200, config)
    df_clean = drop_warmup(df_feat)
    assert len(df_clean) > 0, "drop_warmup left an empty DataFrame"
    assert (df_clean.iloc[-1] == df_feat.iloc[-1]).all()
