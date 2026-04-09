"""
Moving Average Crossover baseline strategy.

Signal logic:
    Golden cross  — SMA-10 crosses *above* SMA-50 → Buy (go all-in)
    Death cross   — SMA-10 crosses *below* SMA-50 → Sell (go all-out)
    Otherwise     — Hold current position

The strategy starts in cash (position = 0).  If the first row already has
SMA-10 > SMA-50 a Buy is triggered immediately so we don't miss an uptrend
that started before our data window.

Works with either raw or normalized DataFrames — only the relative ordering
of ``sma_10`` and ``sma_50`` matters for signal generation.

Usage:
    from src.agents.ma_crossover import generate_signals, run
    import pandas as pd

    df = pd.read_csv("data/features/train.csv", index_col=0, parse_dates=True)
    signals = generate_signals(df)   # Series of 0/1/2
    result  = run(df, initial_balance=10_000.0, transaction_cost=0.001)
    print(result.tail())
"""
import pandas as pd
import numpy as np


def generate_signals(df: pd.DataFrame) -> pd.Series:
    """
    Generate a trade-signal series from SMA crossovers.

    Args:
        df:
            DataFrame with ``sma_10`` and ``sma_50`` columns.

    Returns:
        pd.Series of int (same index as df):
            0 = Hold
            1 = Buy  (golden cross, or immediate entry on first bar)
            2 = Sell (death cross)
    """
    required = {"sma_10", "sma_50"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"ma_crossover.generate_signals: missing columns {missing}. "
            f"Available: {list(df.columns)}"
        )

    sma_short = df["sma_10"].values
    sma_long = df["sma_50"].values
    n = len(df)

    signals = np.zeros(n, dtype=int)

    # Is SMA-10 above SMA-50?  Boolean array.
    above = sma_short > sma_long

    for i in range(n):
        if i == 0:
            # First bar: enter if already in bullish regime
            signals[i] = 1 if above[i] else 0
        else:
            if above[i] and not above[i - 1]:
                signals[i] = 1   # Golden cross — Buy
            elif not above[i] and above[i - 1]:
                signals[i] = 2   # Death cross — Sell
            else:
                signals[i] = 0   # Hold

    return pd.Series(signals, index=df.index, name="signal")


def run(
    df: pd.DataFrame,
    initial_balance: float = 10_000.0,
    transaction_cost: float = 0.001,
) -> pd.DataFrame:
    """
    Simulate the MA Crossover strategy over the given DataFrame.

    Args:
        df:
            DataFrame with ``close``, ``sma_10``, and ``sma_50`` columns.
            Index should be a DatetimeIndex.
        initial_balance:
            Starting cash in USDT (or in whatever unit ``close`` is in).
        transaction_cost:
            Fractional cost deducted on each Buy and Sell (e.g. 0.001 = 0.1%).

    Returns:
        DataFrame indexed like ``df`` with columns:
            portfolio_value  — total value (cash + BTC) at each step
            position         — 0 = cash, 1 = holding BTC
            action           — 0=Hold, 1=Buy, 2=Sell
            cash             — cash held at each step
            btc_held         — BTC units held at each step
    """
    if "close" not in df.columns:
        raise ValueError("ma_crossover.run: DataFrame must contain a 'close' column.")
    if len(df) == 0:
        raise ValueError("ma_crossover.run: DataFrame is empty.")

    signals = generate_signals(df)
    close = df["close"].values
    n = len(close)

    # ------------------------------------------------------------------ #
    # Simulate step by step
    # ------------------------------------------------------------------ #
    cash = initial_balance
    btc = 0.0
    position = 0

    portfolio_values = np.zeros(n)
    cash_trace = np.zeros(n)
    btc_trace = np.zeros(n)
    position_trace = np.zeros(n, dtype=int)
    action_trace = signals.values.copy()

    for i in range(n):
        price = close[i]
        action = signals.iloc[i]

        if action == 1 and position == 0 and cash > 0:
            # Buy — go all-in
            trade_cost = cash * transaction_cost
            spend = cash - trade_cost
            btc = spend / price if price > 0 else 0.0
            cash = 0.0
            position = 1

        elif action == 2 and position == 1 and btc > 0:
            # Sell — liquidate all
            gross = btc * price
            trade_cost = gross * transaction_cost
            cash = gross - trade_cost
            btc = 0.0
            position = 0

        # Record state after action
        portfolio_values[i] = cash + btc * price
        cash_trace[i] = cash
        btc_trace[i] = btc
        position_trace[i] = position

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
