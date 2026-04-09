"""Unit tests for the TradingEnv Gymnasium environment."""
import numpy as np
import pandas as pd
import pytest
from stable_baselines3.common.env_checker import check_env

from src.environment.trading_env import TradingEnv


# --------------------------------------------------------------------------- #
# Fixture: minimal synthetic DataFrame (no real data dependency)
# --------------------------------------------------------------------------- #
@pytest.fixture
def simple_df():
    """10-row synthetic normalised feature DataFrame with steadily rising price."""
    n = 10
    close = np.linspace(0.1, 0.5, n)   # price goes from 0.1 to 0.5
    data = {
        "open":       close * 0.99,
        "high":       close * 1.01,
        "low":        close * 0.98,
        "close":      close,
        "volume":     np.full(n, 0.5),
        "sma_10":     close * 0.97,
        "sma_50":     close * 0.95,
        "rsi":        np.full(n, 60.0),       # 0–100 range; env divides by 100
        "momentum_5": np.full(n, 5.0),         # env applies (x+100)/200 clip
    }
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(data, index=idx)


@pytest.fixture
def env_config():
    return {"initial_balance": 10_000.0, "transaction_cost": 0.001}


@pytest.fixture
def env(simple_df, env_config):
    return TradingEnv(simple_df, env_config)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_sb3_env_check(env):
    """Environment must pass Stable-Baselines3's API checker."""
    check_env(env, warn=True)


def test_env_reset(env, env_config):
    obs, info = env.reset()
    assert obs.shape == (11,), "Observation should be 11-dimensional"
    assert (obs >= 0).all() and (obs <= 1).all(), "All obs must be in [0, 1]"
    assert info == {}, "reset() info should be empty dict"
    assert env._portfolio_value == env_config["initial_balance"]
    assert env._position == 0
    assert env._cash == env_config["initial_balance"]


def test_obs_bounds_full_episode(env):
    """Observations must stay in [0, 1] for every step of a random episode."""
    obs, _ = env.reset()
    all_obs = [obs]
    done = False
    while not done:
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        done = terminated or truncated
        all_obs.append(obs)
    arr = np.array(all_obs)
    assert arr.min() >= 0.0, f"Obs below 0: {arr.min()}"
    assert arr.max() <= 1.0, f"Obs above 1: {arr.max()}"


def test_env_step_hold(env, env_config):
    """Hold action should not change cash or BTC held."""
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)  # Hold
    assert info["cash"] == env_config["initial_balance"], "Cash unchanged on Hold"
    assert info["btc_held"] == 0.0, "No BTC bought on Hold"
    assert info["position"] == 0


def test_env_step_buy(env, env_config):
    """Buy action should move all cash into BTC and deduct transaction cost."""
    env.reset()
    obs, reward, terminated, truncated, info = env.step(1)  # Buy
    assert info["cash"] == pytest.approx(0.0, abs=1e-9), "Cash should be 0 after Buy"
    assert info["btc_held"] > 0, "BTC units should be positive after Buy"
    assert info["position"] == 1
    # Transaction cost deducted — portfolio value slightly below initial
    assert info["portfolio_value"] < env_config["initial_balance"]


def test_env_step_buy_sell(env, env_config):
    """Buy then Sell should return a portfolio value close to initial (minus 2× cost)."""
    env.reset()
    env.step(1)   # Buy
    _, _, _, _, info_sell = env.step(2)  # Sell
    # After Buy + Sell at same price step, we lose ~0.2% (two 0.1% costs)
    # but the price may have changed between step 0 and step 1
    assert info_sell["position"] == 0
    assert info_sell["btc_held"] == pytest.approx(0.0, abs=1e-9)
    assert info_sell["cash"] > 0


def test_transaction_cost_applied(env_config):
    """Transaction cost must be strictly deducted on Buy and Sell."""
    n = 5
    # Flat price — portfolio return should equal exactly -2 * transaction_cost
    close = np.full(n, 0.3)
    data = {
        "open": close, "high": close, "low": close, "close": close,
        "volume": np.full(n, 0.5),
        "sma_10": close, "sma_50": close,
        "rsi": np.full(n, 50.0), "momentum_5": np.zeros(n),
    }
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    df_flat = pd.DataFrame(data, index=idx)

    e = TradingEnv(df_flat, env_config)
    e.reset()
    e.step(1)   # Buy  at step 0
    e.step(0)   # Hold at step 1
    _, _, _, _, info = e.step(2)  # Sell at step 2

    tc = env_config["transaction_cost"]
    expected_return = (1 - tc) * (1 - tc) - 1   # ≈ -0.002 for tc=0.001
    actual_return = info["portfolio_value"] / env_config["initial_balance"] - 1
    assert actual_return == pytest.approx(expected_return, rel=1e-6)


def test_episode_terminates(env):
    """Episode must terminate exactly after n_steps steps."""
    env.reset()
    steps = 0
    done = False
    while not done:
        _, _, terminated, truncated, _ = env.step(0)
        done = terminated or truncated
        steps += 1
    assert steps == env.n_steps


def test_invalid_sell_noop(env):
    """Sell when not holding should be a no-op (no error, no change in cash)."""
    env.reset()
    obs, reward, terminated, truncated, info = env.step(2)  # Sell without position
    assert info["cash"] == env._initial_balance   # unchanged
    assert info["position"] == 0


def test_double_buy_noop(env):
    """Second Buy while already holding should be a no-op."""
    env.reset()
    env.step(1)   # Buy
    _, _, _, _, info_before = env.step(0)   # Hold — record BTC held
    btc_before = info_before["btc_held"]
    env.step(1)   # Buy again — should be ignored
    _, _, _, _, info_after = env.step(0)    # Hold
    assert info_after["btc_held"] == pytest.approx(btc_before, rel=1e-9)
