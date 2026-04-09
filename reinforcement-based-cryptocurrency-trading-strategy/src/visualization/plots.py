"""
Visualization utilities for portfolio analysis and training diagnostics.
"""
import pandas as pd
import matplotlib.pyplot as plt


def plot_portfolio(portfolios: dict[str, pd.Series], title: str = "Portfolio Performance"):
    """Overlay portfolio value curves for multiple strategies."""
    raise NotImplementedError


def plot_signals(df: pd.DataFrame, actions: pd.Series):
    """Plot BTC price with buy/sell signals overlaid."""
    raise NotImplementedError


def plot_learning_curve(log_path: str):
    """Plot episode reward progression from training logs."""
    raise NotImplementedError


def plot_drawdown(portfolio: pd.Series):
    """Plot drawdown over time."""
    raise NotImplementedError


def plot_action_distribution(actions: pd.Series):
    """Bar chart of Hold / Buy / Sell action counts."""
    raise NotImplementedError
