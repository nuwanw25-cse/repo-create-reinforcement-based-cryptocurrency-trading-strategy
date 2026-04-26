"""
Custom Gymnasium trading environment for BTC/USDT daily trading.

State (11D, all in [0, 1] or bounded):
    Market features (from normalized feature CSV):
        close_return, open_gap, high_dev, low_dev  — return-based, MinMax-scaled
        log_volume                                 — log1p then MinMax-scaled
        sma_10_dev, sma_50_dev                     — (close−sma)/close, clip-shifted
        rsi                                        — divided by 100 → [0, 1]
        momentum_5                                 — clipped and normalized → [0, 1]
    Agent state:
        position      — 0 (cash) or 1 (holding BTC)
        portfolio_pct — portfolio value relative to initial balance,
                        clipped to [0, 2] then normalized to [0, 1]

Actions:
    0 = Hold    (do nothing)
    1 = Buy     (go all-in: buy BTC with all available cash)
    2 = Sell    (go all-out: sell all BTC for cash)

Reward:
    Percentage change in total portfolio value between steps.
    A transaction cost (default 0.1%) is deducted on Buy and Sell.

Episode:
    Starts at step 0 (first row of df).
    Ends after the last row has been processed.
    Terminal step gives a reward based on liquidating the final position.

Usage:
    from src.environment.trading_env import TradingEnv
    import pandas as pd

    df = pd.read_csv("data/features/train.csv", index_col=0, parse_dates=True)
    env = TradingEnv(df, config={"initial_balance": 10000, "transaction_cost": 0.001})
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(1)  # Buy
"""
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


# Observation columns — all normalized to [0, 1] by the data pipeline (except rsi/momentum
# which are transformed in _get_obs).  A ``close`` column must also be present in the
# DataFrame for portfolio value tracking, but it is not part of the observation.
MARKET_COLS = [
    "close_return", "open_gap", "high_dev", "low_dev", "log_volume",
    "sma_10_dev", "sma_50_dev",
    "rsi", "momentum_5",
]


class TradingEnv(gym.Env):
    """
    Single-asset (BTC/USDT) daily trading environment.

    The agent can hold at most 1 unit of BTC at a time (all-in / all-out).
    Fractional positions are not supported — this keeps the action space
    simple (Discrete 3) and the reward signal clear.
    """

    metadata = {"render_modes": ["human"]}

    # ---------------------------------------------------------------------- #
    # Initialisation
    # ---------------------------------------------------------------------- #

    def __init__(self, df: pd.DataFrame, config: dict):
        """
        Args:
            df:
                Feature-enriched, normalized DataFrame produced by the data
                pipeline.  Must contain the columns listed in MARKET_COLS.
                Index should be a DatetimeIndex (UTC).
            config:
                Dict with keys:
                    initial_balance   (float) — starting cash in USDT
                    transaction_cost  (float) — fractional cost per trade
                                               (e.g. 0.001 = 0.1%)
        """
        super().__init__()

        # Validate required columns: MARKET_COLS for observation + close for price tracking
        required = MARKET_COLS + ["close"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"TradingEnv: DataFrame is missing required columns: {missing}\n"
                f"Available columns: {list(df.columns)}"
            )

        self._original_index = df.index          # preserve DatetimeIndex for trace alignment
        self._df = df.reset_index(drop=True)     # integer index for fast iloc
        self._prices = df["close"].values        # raw close prices for portfolio arithmetic
        self._n_steps = len(df)

        self._initial_balance: float = float(config.get("initial_balance", 10_000.0))
        self._transaction_cost: float = float(config.get("transaction_cost", 0.001))

        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)

        # Observation space: 11D vector in [0, 1]
        # (market features: 9D) + (position: 1D) + (portfolio_pct: 1D)
        n_obs = len(MARKET_COLS) + 2  # +2 for position, portfolio_pct
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(n_obs,),
            dtype=np.float32,
        )

        # Mutable episode state (initialised in reset)
        self._current_step: int = 0
        self._cash: float = self._initial_balance
        self._btc_held: float = 0.0        # BTC units (0 or >0)
        self._position: int = 0            # 0=cash, 1=holding
        self._portfolio_value: float = self._initial_balance
        self._prev_portfolio_value: float = self._initial_balance

    # ---------------------------------------------------------------------- #
    # Gymnasium API
    # ---------------------------------------------------------------------- #

    def reset(self, *, seed=None, options=None):
        """
        Reset environment to the start of the episode.

        Returns:
            observation (np.ndarray): initial state vector
            info (dict): auxiliary information (empty at reset)
        """
        super().reset(seed=seed)

        self._current_step = 0
        self._cash = self._initial_balance
        self._btc_held = 0.0
        self._position = 0
        self._portfolio_value = self._initial_balance
        self._prev_portfolio_value = self._initial_balance

        return self._get_obs(), {}

    def step(self, action: int):
        """
        Execute one trading step.

        Args:
            action: 0=Hold, 1=Buy, 2=Sell

        Returns:
            observation  (np.ndarray)
            reward       (float)     — portfolio % change this step
            terminated   (bool)      — True when last row is reached
            truncated    (bool)      — always False (no time limit)
            info         (dict)      — diagnostic data
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"

        current_price_norm = self._prices[self._current_step]

        # We need the actual (unnormalized) price for portfolio arithmetic.
        # Since we only use close prices for P&L tracking, we work with the
        # normalized price directly — the initial_balance is also in this
        # normalized "unit space".  The profit/loss percentages are identical
        # regardless of the absolute price scale.
        price = current_price_norm  # dimensionless: consistent with portfolio tracking

        # ------------------------------------------------------------------ #
        # Execute action
        # ------------------------------------------------------------------ #
        trade_cost = 0.0

        if action == 1 and self._position == 0 and self._cash > 0:
            # Buy: spend all cash on BTC
            trade_cost = self._cash * self._transaction_cost
            spend = self._cash - trade_cost
            self._btc_held = spend / price if price > 0 else 0.0
            self._cash = 0.0
            self._position = 1

        elif action == 2 and self._position == 1 and self._btc_held > 0:
            # Sell: liquidate all BTC
            gross = self._btc_held * price
            trade_cost = gross * self._transaction_cost
            self._cash = gross - trade_cost
            self._btc_held = 0.0
            self._position = 0

        # Hold (action == 0) or invalid trade (e.g. sell when no position): no-op

        # ------------------------------------------------------------------ #
        # Update portfolio value
        # ------------------------------------------------------------------ #
        btc_value = self._btc_held * price
        self._portfolio_value = self._cash + btc_value

        # ------------------------------------------------------------------ #
        # Reward: percentage change in portfolio value this step
        # ------------------------------------------------------------------ #
        if self._prev_portfolio_value > 0:
            reward = (self._portfolio_value - self._prev_portfolio_value) / self._prev_portfolio_value
        else:
            reward = 0.0

        self._prev_portfolio_value = self._portfolio_value

        # ------------------------------------------------------------------ #
        # Advance step
        # ------------------------------------------------------------------ #
        self._current_step += 1
        terminated = self._current_step >= self._n_steps

        # If episode ends and we're still holding, the final portfolio value
        # already reflects the last close price — no extra sell needed.

        info = {
            "step": self._current_step,
            "action": action,
            "price": price,
            "cash": self._cash,
            "btc_held": self._btc_held,
            "portfolio_value": self._portfolio_value,
            "trade_cost": trade_cost,
            "position": self._position,
        }

        return self._get_obs(), float(reward), terminated, False, info

    def render(self, mode="human"):
        """Print a one-line summary of the current state."""
        print(
            f"Step {self._current_step:4d} | "
            f"Price: {self._prices[max(0, self._current_step - 1)]:.4f} | "
            f"Cash: {self._cash:10.2f} | "
            f"BTC: {self._btc_held:.6f} | "
            f"Portfolio: {self._portfolio_value:10.2f} | "
            f"Position: {'LONG' if self._position else 'CASH'}"
        )

    # ---------------------------------------------------------------------- #
    # Internal helpers
    # ---------------------------------------------------------------------- #

    def _get_obs(self) -> np.ndarray:
        """
        Build the observation vector for the current step.

        Returns a float32 array of shape (11,):
            [close_return, open_gap, high_dev, low_dev, log_volume,  # already in [0,1]
             sma_10_dev, sma_50_dev,                                  # already in [0,1]
             rsi_norm,                                                # rsi / 100 → [0,1]
             momentum_norm,                                           # (mom+100)/200 → [0,1]
             position,                                                # 0 or 1
             portfolio_pct]                                           # portfolio/(2*initial)
        """
        idx = min(self._current_step, self._n_steps - 1)
        row = self._df.iloc[idx]

        market = np.array([
            np.clip(row["close_return"], 0.0, 1.0),
            np.clip(row["open_gap"],     0.0, 1.0),
            np.clip(row["high_dev"],     0.0, 1.0),
            np.clip(row["low_dev"],      0.0, 1.0),
            np.clip(row["log_volume"],   0.0, 1.0),
            np.clip(row["sma_10_dev"],   0.0, 1.0),
            np.clip(row["sma_50_dev"],   0.0, 1.0),
            np.clip(row["rsi"] / 100.0, 0.0, 1.0),
            np.clip((row["momentum_5"] + 100.0) / 200.0, 0.0, 1.0),
        ], dtype=np.float32)

        # Agent state features
        portfolio_pct = np.clip(
            self._portfolio_value / (2.0 * self._initial_balance), 0.0, 1.0
        )
        agent_state = np.array([float(self._position), portfolio_pct], dtype=np.float32)

        return np.concatenate([market, agent_state])

    # ---------------------------------------------------------------------- #
    # Utility
    # ---------------------------------------------------------------------- #

    @property
    def n_steps(self) -> int:
        """Number of rows (trading days) in this environment's dataset."""
        return self._n_steps

    @property
    def total_return(self) -> float:
        """
        Cumulative return of the current episode as a percentage.
        (portfolio_value / initial_balance - 1) * 100
        """
        return (self._portfolio_value / self._initial_balance - 1.0) * 100.0
