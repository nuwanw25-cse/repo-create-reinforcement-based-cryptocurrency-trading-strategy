"""
Financial performance metrics for strategy evaluation.

All functions accept a ``portfolio`` pd.Series of portfolio values at each
timestep, or a ``returns`` pd.Series of per-step percentage returns.

1H candle data is assumed throughout (``periods_per_year=8760``, i.e. 365 × 24).

Metrics implemented:
    total_return      — overall gain/loss as a fraction
    annualized_return — CAGR scaled to one year
    max_drawdown      — largest peak-to-trough drop (as a negative fraction)
    sharpe_ratio      — risk-adjusted return (excess return / std dev)
    sortino_ratio     — downside risk-adjusted return (excess return / downside std)
    calmar_ratio      — annualized return / |max drawdown|
    win_rate          — fraction of periods with positive return

Usage:
    from src.evaluation.metrics import compute_all
    import pandas as pd

    result = pd.read_csv("results/dqn_test.csv", index_col=0, parse_dates=True)
    metrics = compute_all(result["portfolio_value"])
    for k, v in metrics.items():
        print(f"  {k:<25s}: {v:.4f}")
"""
import numpy as np
import pandas as pd


def _to_returns(portfolio: pd.Series) -> pd.Series:
    """Convert a portfolio-value series to per-step fractional returns."""
    return portfolio.pct_change().dropna()


# --------------------------------------------------------------------------- #
# Individual metrics
# --------------------------------------------------------------------------- #

def total_return(portfolio: pd.Series) -> float:
    """
    Total return over the evaluation period.

    Returns:
        float — e.g. 0.487 means +48.7%.  Negative values indicate a loss.
    """
    if len(portfolio) < 2:
        return 0.0
    return float(portfolio.iloc[-1] / portfolio.iloc[0] - 1.0)


def annualized_return(portfolio: pd.Series, periods_per_year: int = 8760) -> float:
    """
    Compound annualised growth rate (CAGR).

    Formula:  (end / start) ^ (periods_per_year / n_periods) - 1

    Returns:
        float — annualised return fraction.
    """
    n = len(portfolio)
    if n < 2:
        return 0.0
    tr = portfolio.iloc[-1] / portfolio.iloc[0]
    if tr <= 0:
        return -1.0
    return float(tr ** (periods_per_year / (n - 1)) - 1.0)


def max_drawdown(portfolio: pd.Series) -> float:
    """
    Maximum peak-to-trough drawdown.

    Returns:
        float — e.g. -0.23 means the worst drawdown was -23%.
                Always <= 0.  Returns 0.0 for flat/always-rising portfolios.
    """
    if len(portfolio) < 2:
        return 0.0
    rolling_max = portfolio.cummax()
    drawdowns = portfolio / rolling_max - 1.0
    return float(drawdowns.min())


def sharpe_ratio(
    returns: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = 8760,
) -> float:
    """
    Annualised Sharpe ratio.

    Formula:  (mean_excess_return / std_return) * sqrt(periods_per_year)

    Args:
        returns:         Per-step fractional returns (e.g. from pct_change()).
        risk_free:       Risk-free rate per step (default 0 — no adjustment).
        periods_per_year: Trading periods in a year (8760 for 1H candles).

    Returns:
        float — Sharpe ratio.  NaN if std is zero.
    """
    returns = returns.dropna()
    if len(returns) < 2:
        return float("nan")
    excess = returns - risk_free
    std = excess.std(ddof=1)
    if std == 0:
        return float("nan")
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = 8760,
) -> float:
    """
    Annualised Sortino ratio.

    Uses only downside deviation (negative excess returns) in the denominator.

    Returns:
        float — Sortino ratio.  NaN if downside std is zero (no losing days).
    """
    returns = returns.dropna()
    if len(returns) < 2:
        return float("nan")
    excess = returns - risk_free
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf")   # No losing periods — theoretically infinite
    downside_std = downside.std(ddof=1)
    if downside_std == 0:
        return float("nan")
    return float(excess.mean() / downside_std * np.sqrt(periods_per_year))


def calmar_ratio(portfolio: pd.Series, periods_per_year: int = 8760) -> float:
    """
    Calmar ratio: annualised return divided by absolute max drawdown.

    Returns:
        float — Calmar ratio.  NaN if max drawdown is zero (no drawdown at all).
    """
    ann_ret = annualized_return(portfolio, periods_per_year)
    mdd = max_drawdown(portfolio)
    if mdd == 0.0:
        return float("inf")
    return float(ann_ret / abs(mdd))


def win_rate(returns: pd.Series) -> float:
    """
    Fraction of periods (steps) with a positive return.

    Returns:
        float in [0, 1].
    """
    returns = returns.dropna()
    if len(returns) == 0:
        return float("nan")
    return float((returns > 0).sum() / len(returns))


# --------------------------------------------------------------------------- #
# Convenience wrapper
# --------------------------------------------------------------------------- #

def compute_all(
    portfolio: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = 8760,
) -> dict:
    """
    Compute and return all metrics as an ordered dictionary.

    Args:
        portfolio:       Series of portfolio values (one per trading step).
        risk_free:       Per-step risk-free rate (default 0).
        periods_per_year: Trading periods in a year (default 8760 for 1H candles).

    Returns:
        dict with keys:
            total_return        (float)
            annualized_return   (float)
            max_drawdown        (float, <= 0)
            sharpe_ratio        (float)
            sortino_ratio       (float)
            calmar_ratio        (float)
            win_rate            (float, in [0,1])
    """
    returns = _to_returns(portfolio)
    return {
        "total_return":      total_return(portfolio),
        "annualized_return": annualized_return(portfolio, periods_per_year),
        "max_drawdown":      max_drawdown(portfolio),
        "sharpe_ratio":      sharpe_ratio(returns, risk_free, periods_per_year),
        "sortino_ratio":     sortino_ratio(returns, risk_free, periods_per_year),
        "calmar_ratio":      calmar_ratio(portfolio, periods_per_year),
        "win_rate":          win_rate(returns),
    }
