"""
Backtesting pipeline: run all strategies and compare their performance.

The backtester provides two main workflows:

1. Single-period evaluation (most common):
   Run each strategy over a single DataFrame (train, val, or test) and
   collect a metrics table.

2. Walk-forward validation:
   Roll a fixed-size training window forward across the dataset, retrain the
   DQN at each step, then evaluate on the following out-of-sample window.
   Useful for robustness analysis but computationally expensive.

Key design:
  - All strategies return a ``pd.DataFrame`` with a ``portfolio_value`` column.
  - ``backtest_agent``   handles the DQN agent (via ``dqn_agent.run_episode``).
  - ``backtest_baseline`` handles B&H and MA Crossover (plain functions).
  - ``compare_strategies`` turns a dict of metric dicts into a tidy table.

Usage:
    from src.evaluation.backtester import run_all, compare_strategies
    import pandas as pd, yaml

    cfg = yaml.safe_load(open("configs/dqn_config.yaml"))
    df_test      = pd.read_csv("data/features/test.csv",    index_col=0, parse_dates=True)
    df_test_norm = pd.read_csv("data/normalized/test.csv",  index_col=0, parse_dates=True)

    from stable_baselines3 import DQN
    from src.environment.trading_env import TradingEnv
    env_test = TradingEnv(df_test_norm, cfg["environment"])
    agent    = DQN.load("models/best/best_model", env=env_test)

    results, table = run_all(df_test, df_test_norm, agent, cfg["environment"])
    print(table.to_string())
"""
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import numpy as np

from src.evaluation.metrics import compute_all
from src.agents import buy_and_hold, ma_crossover
from src.agents.dqn_agent import run_episode
from src.environment.trading_env import TradingEnv


# --------------------------------------------------------------------------- #
# Per-strategy backtesting helpers
# --------------------------------------------------------------------------- #

def backtest_agent(agent, env: TradingEnv) -> pd.DataFrame:
    """
    Run a trained DQN agent through one full episode and record results.

    Args:
        agent: Trained ``stable_baselines3.DQN`` instance.
        env:   ``TradingEnv`` wrapping the evaluation dataset
               (will be reset before use).

    Returns:
        DataFrame with columns [portfolio_value, cash, btc_held, position, action]
        indexed by the environment's trading day index.
    """
    return run_episode(agent, env)


def backtest_baseline(
    strategy: str,
    df: pd.DataFrame,
    initial_balance: float = 10_000.0,
    transaction_cost: float = 0.001,
) -> pd.DataFrame:
    """
    Run a baseline strategy over a DataFrame.

    Args:
        strategy:   "buy_and_hold" or "ma_crossover".
        df:         Feature DataFrame (raw or normalised — returns are scale-invariant).
        initial_balance: Starting cash.
        transaction_cost: Fractional cost per trade.

    Returns:
        DataFrame with columns [portfolio_value, cash, btc_held, position, action].
    """
    kwargs = dict(initial_balance=initial_balance, transaction_cost=transaction_cost)
    if strategy == "buy_and_hold":
        return buy_and_hold.run(df, **kwargs)
    elif strategy == "ma_crossover":
        return ma_crossover.run(df, **kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}. Use 'buy_and_hold' or 'ma_crossover'.")


# --------------------------------------------------------------------------- #
# Convenience: run all strategies and compute metrics
# --------------------------------------------------------------------------- #

def run_all(
    df_raw: pd.DataFrame,
    df_norm: pd.DataFrame,
    agent,
    env_config: dict,
    initial_balance: float = 10_000.0,
    transaction_cost: Optional[float] = None,
    save_dir: Optional[str] = None,
) -> tuple:
    """
    Run all three strategies and return their per-step traces and metrics.

    Args:
        df_raw:          Un-normalised feature DataFrame (for B&H and MA crossover).
        df_norm:         Normalised feature DataFrame (for DQN via TradingEnv).
        agent:           Trained DQN agent (or None to skip DQN).
        env_config:      Dict with ``initial_balance`` and ``transaction_cost``.
        initial_balance: Overrides env_config initial_balance if provided.
        transaction_cost: Overrides env_config transaction_cost if provided.
        save_dir:        If set, saves per-strategy portfolio traces as CSVs.

    Returns:
        (results, metrics_table) where:
            results       — dict mapping strategy name → portfolio trace DataFrame
            metrics_table — pd.DataFrame with strategies as columns, metrics as rows
    """
    ib = initial_balance or env_config.get("initial_balance", 10_000.0)
    tc = transaction_cost if transaction_cost is not None else env_config.get("transaction_cost", 0.001)

    results: Dict[str, pd.DataFrame] = {}

    # B&H
    results["Buy-and-Hold"] = backtest_baseline("buy_and_hold", df_raw, ib, tc)

    # MA Crossover
    results["MA-Crossover"] = backtest_baseline("ma_crossover", df_raw, ib, tc)

    # DQN
    if agent is not None:
        env_eval = TradingEnv(df_norm, {"initial_balance": ib, "transaction_cost": tc})
        results["DQN"] = backtest_agent(agent, env_eval)

    # Compute metrics for each strategy
    metrics_per_strategy = {}
    for name, trace in results.items():
        portfolio = trace["portfolio_value"]
        # Scale DQN portfolio from normalized-price units to initial_balance USD
        # (percentage returns are identical, so metrics are scale-invariant)
        metrics_per_strategy[name] = compute_all(portfolio)

    # Optionally save traces
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        for name, trace in results.items():
            fname = name.lower().replace("-", "_").replace(" ", "_") + ".csv"
            trace.to_csv(save_path / fname)
            print(f"Saved {name} trace → {save_path / fname}")

    metrics_table = compare_strategies(metrics_per_strategy)
    return results, metrics_table


# --------------------------------------------------------------------------- #
# Comparison table
# --------------------------------------------------------------------------- #

def compare_strategies(results: dict) -> pd.DataFrame:
    """
    Build a formatted comparison table from a dict of metric dicts.

    Args:
        results:
            {strategy_name: metrics_dict, ...}
            where metrics_dict comes from ``metrics.compute_all()``.

    Returns:
        pd.DataFrame with strategies as columns and metric names as rows.
        Values are formatted to 4 decimal places.

    Example:
        >>> compare_strategies({
        ...     "DQN":          {"total_return": 0.15, "sharpe_ratio": 1.2, ...},
        ...     "Buy-and-Hold": {"total_return": 0.48, "sharpe_ratio": 1.8, ...},
        ... })
    """
    if not results:
        return pd.DataFrame()

    table = pd.DataFrame(results).round(4)

    # Friendly row labels
    label_map = {
        "total_return":      "Total Return",
        "annualized_return": "Ann. Return",
        "max_drawdown":      "Max Drawdown",
        "sharpe_ratio":      "Sharpe Ratio",
        "sortino_ratio":     "Sortino Ratio",
        "calmar_ratio":      "Calmar Ratio",
        "win_rate":          "Win Rate",
    }
    table.index = [label_map.get(k, k) for k in table.index]
    return table


# --------------------------------------------------------------------------- #
# Walk-forward validation
# --------------------------------------------------------------------------- #

def walk_forward(
    df_norm: pd.DataFrame,
    agent_builder,
    agent_model_config: dict,
    training_config: dict,
    env_config: dict,
    train_window: int = 60,
    test_window: int = 20,
) -> "list[dict]":
    """
    Walk-forward validation: retrain DQN on a rolling window, evaluate next window.

    Args:
        df_norm:           Full normalised feature DataFrame.
        agent_builder:     Callable(env, config) → DQN  (e.g. dqn_agent.build_agent).
        agent_model_config: Model hyperparameters for agent_builder.
        training_config:   Training config for trainer.train().
        env_config:        TradingEnv config dict.
        train_window:      Number of rows in each training window.
        test_window:       Number of rows in each test window.

    Returns:
        List of dicts, one per fold:
            {fold, train_start, train_end, test_start, test_end, metrics}
    """
    from src.training.trainer import train as train_agent

    n = len(df_norm)
    folds = []
    fold = 0
    start = 0

    while start + train_window + test_window <= n:
        train_slice = df_norm.iloc[start : start + train_window]
        test_slice  = df_norm.iloc[start + train_window : start + train_window + test_window]

        env_train = TradingEnv(train_slice, env_config)
        env_val   = TradingEnv(test_slice,  env_config)

        agent = agent_builder(env_train, agent_model_config)
        agent = train_agent(env_train, env_val, agent, training_config)

        trace   = backtest_agent(agent, env_val)
        metrics = compute_all(trace["portfolio_value"])

        folds.append({
            "fold":        fold,
            "train_start": train_slice.index[0],
            "train_end":   train_slice.index[-1],
            "test_start":  test_slice.index[0],
            "test_end":    test_slice.index[-1],
            "metrics":     metrics,
        })

        fold  += 1
        start += test_window   # advance by test_window (rolling)

    return folds
