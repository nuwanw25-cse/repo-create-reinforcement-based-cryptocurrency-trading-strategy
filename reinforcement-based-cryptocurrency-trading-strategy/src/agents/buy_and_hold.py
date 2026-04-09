"""
Buy-and-Hold baseline strategy.

Purchases BTC at the very first close price and holds until the end of the
evaluation period.  A single buy transaction cost is deducted up front; no
sell cost is deducted at the end (position is implicitly liquidated at the
last close price for portfolio reporting only).

Works with either raw or normalized DataFrames — only the `close` column is
used, so percentage returns are identical in both cases.

Usage:
    from src.agents.buy_and_hold import run
    import pandas as pd

    df = pd.read_csv("data/features/train.csv", index_col=0, parse_dates=True)
    result = run(df, initial_balance=10_000.0, transaction_cost=0.001)
    print(result.tail())
"""
import pandas as pd
import numpy as np


def run(
    df: pd.DataFrame,
    initial_balance: float = 10_000.0,
    transaction_cost: float = 0.001,
) -> pd.DataFrame:
    """
    Simulate the Buy-and-Hold strategy over the given DataFrame.

    Args:
        df:
            DataFrame with at least a ``close`` column.
            Index should be a DatetimeIndex.
        initial_balance:
            Starting cash in USDT (or in whatever unit ``close`` is in).
        transaction_cost:
            Fractional cost deducted on the buy (e.g. 0.001 = 0.1%).

    Returns:
        DataFrame indexed like ``df`` with columns:
            portfolio_value  — total value (cash + BTC) at each step
            position         — 1 = holding BTC throughout
            action           — 1 on the first row (Buy), 0 thereafter (Hold)
            cash             — cash held at each step (0 after buying)
            btc_held         — BTC units held at each step
    """
    if "close" not in df.columns:
        raise ValueError("buy_and_hold.run: DataFrame must contain a 'close' column.")
    if len(df) == 0:
        raise ValueError("buy_and_hold.run: DataFrame is empty.")

    close = df["close"].values
    n = len(close)

    # ------------------------------------------------------------------ #
    # Buy at step 0 — spend all cash minus transaction cost
    # ------------------------------------------------------------------ #
    buy_price = close[0]
    trade_cost = initial_balance * transaction_cost
    spend = initial_balance - trade_cost
    btc = spend / buy_price if buy_price > 0 else 0.0

    # ------------------------------------------------------------------ #
    # Portfolio value at every step
    # ------------------------------------------------------------------ #
    portfolio_values = btc * close          # shape (n,)
    cash_trace = np.zeros(n)
    btc_trace = np.full(n, btc)
    position_trace = np.ones(n, dtype=int)
    action_trace = np.zeros(n, dtype=int)
    action_trace[0] = 1                     # Buy on first day

    return pd.DataFrame(
        {
            "portfolio_value": portfolio_values,
            "cash": cash_trace,
            "btc_held": btc_trace,
            "position": position_trace,
            "action": action_trace,
        },
        index=df.index,
    )
