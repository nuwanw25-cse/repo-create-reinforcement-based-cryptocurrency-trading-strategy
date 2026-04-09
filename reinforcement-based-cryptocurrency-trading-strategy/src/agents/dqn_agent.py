"""
DQN agent using Stable-Baselines3.

This module provides thin wrappers around SB3's DQN class so the rest of the
project only needs to know about ``build_agent`` and ``load_agent``.  All
hyperparameters come from ``configs/dqn_config.yaml`` via the ``config`` dict.

Usage:
    from src.agents.dqn_agent import build_agent, load_agent
    from src.environment.trading_env import TradingEnv
    import pandas as pd, yaml

    df  = pd.read_csv("data/normalized/train.csv", index_col=0, parse_dates=True)
    cfg = yaml.safe_load(open("configs/dqn_config.yaml"))

    env   = TradingEnv(df, cfg["environment"])
    agent = build_agent(env, cfg["model"])

    agent.learn(total_timesteps=cfg["training"]["total_timesteps"])
    agent.save("models/dqn_final")

    agent2 = load_agent("models/dqn_final", env)
"""
from pathlib import Path

import pandas as pd
from gymnasium import Env
from stable_baselines3 import DQN
from stable_baselines3.common.type_aliases import MaybeCallback


def build_agent(env: Env, config: dict) -> DQN:
    """
    Instantiate a DQN agent from a config dictionary.

    Args:
        env:
            A Gymnasium environment (typically ``TradingEnv``).
        config:
            Dict matching the ``model`` block in ``configs/dqn_config.yaml``.
            Recognised keys (all optional — SB3 defaults apply otherwise):
                policy, learning_rate, buffer_size, learning_starts,
                batch_size, tau, gamma, train_freq, gradient_steps,
                target_update_interval, exploration_fraction,
                exploration_initial_eps, exploration_final_eps,
                verbose, seed, policy_kwargs.

    Returns:
        Configured (but untrained) ``stable_baselines3.DQN`` instance.
    """
    # policy_kwargs may contain net_arch etc — pass through directly
    policy_kwargs = config.get("policy_kwargs", None)

    agent = DQN(
        policy=config.get("policy", "MlpPolicy"),
        env=env,
        learning_rate=config.get("learning_rate", 1e-4),
        buffer_size=config.get("buffer_size", 50_000),
        learning_starts=config.get("learning_starts", 1_000),
        batch_size=config.get("batch_size", 32),
        tau=config.get("tau", 1.0),
        gamma=config.get("gamma", 0.99),
        train_freq=config.get("train_freq", 4),
        gradient_steps=config.get("gradient_steps", 1),
        target_update_interval=config.get("target_update_interval", 1_000),
        exploration_fraction=config.get("exploration_fraction", 0.3),
        exploration_initial_eps=config.get("exploration_initial_eps", 1.0),
        exploration_final_eps=config.get("exploration_final_eps", 0.05),
        policy_kwargs=policy_kwargs,
        verbose=config.get("verbose", 1),
        seed=config.get("seed", 42),
    )
    return agent


def load_agent(path: str, env: Env) -> DQN:
    """
    Load a saved DQN model from disk.

    Args:
        path:
            Path to the saved model (with or without ``.zip`` extension).
        env:
            Environment to attach to the loaded model (used for subsequent
            inference / evaluation).

    Returns:
        ``stable_baselines3.DQN`` instance loaded from ``path``.
    """
    return DQN.load(path, env=env)


def run_episode(agent: DQN, env: Env) -> pd.DataFrame:
    """
    Run one full deterministic episode with a trained agent.

    Args:
        agent: Trained DQN agent.
        env:   Environment to evaluate on (will be reset internally).

    Returns:
        DataFrame with one row per step, columns:
            portfolio_value, cash, btc_held, position, action
        Indexed by the environment's underlying DataFrame index.
    """
    obs, _ = env.reset()
    done = False

    records = []
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        records.append({
            "portfolio_value": info["portfolio_value"],
            "cash": info["cash"],
            "btc_held": info["btc_held"],
            "position": info["position"],
            "action": info["action"],
        })

    # Align with the environment's original DatetimeIndex
    index = env._original_index[:len(records)]
    return pd.DataFrame(records, index=index)
