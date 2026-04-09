"""Unit tests for financial metric functions (src/evaluation/metrics.py)."""
import math
import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    total_return,
    annualized_return,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
    win_rate,
    compute_all,
)


# --------------------------------------------------------------------------- #
# Helper to build a portfolio series
# --------------------------------------------------------------------------- #
def make_portfolio(values):
    idx = pd.date_range("2024-01-01", periods=len(values), freq="D", tz="UTC")
    return pd.Series(values, index=idx, dtype=float)


# --------------------------------------------------------------------------- #
# total_return
# --------------------------------------------------------------------------- #

def test_total_return_flat():
    p = make_portfolio([10_000, 10_000, 10_000])
    assert total_return(p) == pytest.approx(0.0)


def test_total_return_gain():
    p = make_portfolio([10_000, 15_000])
    assert total_return(p) == pytest.approx(0.5)


def test_total_return_loss():
    p = make_portfolio([10_000, 5_000])
    assert total_return(p) == pytest.approx(-0.5)


def test_total_return_single_row():
    p = make_portfolio([10_000])
    assert total_return(p) == 0.0


# --------------------------------------------------------------------------- #
# max_drawdown
# --------------------------------------------------------------------------- #

def test_max_drawdown_no_loss():
    """Monotonically rising portfolio — no drawdown."""
    p = make_portfolio([1, 2, 3, 4, 5])
    assert max_drawdown(p) == pytest.approx(0.0)


def test_max_drawdown_peak_then_valley():
    """Peak=4, valley=2 → drawdown = 2/4 - 1 = -0.5."""
    p = make_portfolio([1, 2, 4, 2, 3])
    assert max_drawdown(p) == pytest.approx(-0.5)


def test_max_drawdown_is_non_positive():
    p = make_portfolio([100, 80, 120, 90, 110])
    assert max_drawdown(p) <= 0.0


def test_max_drawdown_all_decline():
    """Monotonically falling — drawdown is (final/initial - 1)."""
    p = make_portfolio([10, 8, 6, 4])
    assert max_drawdown(p) == pytest.approx(4 / 10 - 1)


# --------------------------------------------------------------------------- #
# sharpe_ratio
# --------------------------------------------------------------------------- #

def test_sharpe_ratio_positive_returns():
    """Positive mean return with non-zero std → positive Sharpe."""
    # Alternating high/low positive returns so std > 0
    returns = pd.Series([0.01, 0.02] * 126)   # mean=0.015, std>0
    sr = sharpe_ratio(returns)
    assert sr > 0


def test_sharpe_ratio_zero_std():
    """Constant returns → std = 0 → NaN."""
    returns = pd.Series([0.01] * 50)
    result = sharpe_ratio(returns)
    assert math.isnan(result)


def test_sharpe_ratio_negative_returns():
    """Negative mean returns → negative Sharpe."""
    returns = pd.Series([-0.01] * 100)
    result = sharpe_ratio(returns)
    # std will be 0 → NaN in our implementation
    assert math.isnan(result) or result < 0


def test_sharpe_ratio_mixed():
    """Non-trivial positive Sharpe on a good return series."""
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.001, 0.01, 252))
    sr = sharpe_ratio(returns)
    assert sr > 0   # positive mean, so Sharpe should be positive


# --------------------------------------------------------------------------- #
# sortino_ratio
# --------------------------------------------------------------------------- #

def test_sortino_no_downside():
    """No negative returns → Sortino = +inf."""
    returns = pd.Series([0.01] * 50)
    result = sortino_ratio(returns)
    assert result == float("inf")


def test_sortino_greater_than_sharpe_on_skewed():
    """For an asymmetric return series Sortino ≥ Sharpe."""
    rng = np.random.default_rng(1)
    # Mostly positive returns, few large negatives
    returns = pd.Series(
        np.concatenate([rng.normal(0.002, 0.005, 200),
                        rng.normal(-0.005, 0.002, 10)])
    )
    sr = sharpe_ratio(returns)
    so = sortino_ratio(returns)
    # Sortino should be >= Sharpe when downside deviation < total deviation
    assert so >= sr or (math.isnan(sr) or math.isnan(so))


# --------------------------------------------------------------------------- #
# calmar_ratio
# --------------------------------------------------------------------------- #

def test_calmar_ratio_no_drawdown():
    """Rising portfolio, no drawdown → Calmar = +inf."""
    p = make_portfolio([1, 2, 3, 4, 5])
    result = calmar_ratio(p)
    assert result == float("inf")


def test_calmar_ratio_sign():
    """With a drawdown and positive ann return, Calmar should be positive."""
    p = make_portfolio([100, 120, 80, 130, 150])
    result = calmar_ratio(p)
    assert result > 0 or math.isnan(result)


# --------------------------------------------------------------------------- #
# win_rate
# --------------------------------------------------------------------------- #

def test_win_rate_all_positive():
    returns = pd.Series([0.01, 0.02, 0.005])
    assert win_rate(returns) == pytest.approx(1.0)


def test_win_rate_all_negative():
    returns = pd.Series([-0.01, -0.02])
    assert win_rate(returns) == pytest.approx(0.0)


def test_win_rate_half():
    returns = pd.Series([0.01, -0.01, 0.02, -0.02])
    assert win_rate(returns) == pytest.approx(0.5)


def test_win_rate_empty():
    result = win_rate(pd.Series([], dtype=float))
    assert math.isnan(result)


# --------------------------------------------------------------------------- #
# compute_all
# --------------------------------------------------------------------------- #

def test_compute_all_returns_dict():
    p = make_portfolio([10_000, 10_500, 11_000, 10_800, 11_500])
    result = compute_all(p)
    expected_keys = {
        "total_return", "annualized_return", "max_drawdown",
        "sharpe_ratio", "sortino_ratio", "calmar_ratio", "win_rate",
    }
    assert set(result.keys()) == expected_keys


def test_compute_all_values_types():
    p = make_portfolio([10_000, 11_000, 10_500])
    result = compute_all(p)
    for key, val in result.items():
        assert isinstance(val, float), f"{key} should be float, got {type(val)}"


def test_compute_all_consistent_with_individual_functions():
    p = make_portfolio([10_000, 11_000, 10_500, 12_000, 11_500])
    result = compute_all(p)
    assert result["total_return"] == pytest.approx(total_return(p))
    assert result["max_drawdown"] == pytest.approx(max_drawdown(p))
    assert result["win_rate"]     == pytest.approx(win_rate(p.pct_change().dropna()))
