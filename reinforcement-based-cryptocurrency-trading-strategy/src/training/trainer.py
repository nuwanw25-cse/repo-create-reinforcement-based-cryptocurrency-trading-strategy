"""
Training orchestration for the DQN agent.

Wraps Stable-Baselines3 training with:
  - EvalCallback   — evaluates on the validation environment every N steps and
                     saves the best model automatically.
  - StopTrainingOnNoModelImprovement — early stopping if the val reward does
                                        not improve for K consecutive evals.
  - CheckpointCallback — periodic checkpoint saves.

Usage:
    from src.training.trainer import train
    import pandas as pd, yaml
    from src.environment.trading_env import TradingEnv
    from src.agents.dqn_agent import build_agent

    with open("configs/dqn_config.yaml") as f:
        cfg = yaml.safe_load(f)

    df_train = pd.read_csv("data/normalized/train.csv", index_col=0, parse_dates=True)
    df_val   = pd.read_csv("data/normalized/val.csv",   index_col=0, parse_dates=True)

    env_train = TradingEnv(df_train, cfg["environment"])
    env_val   = TradingEnv(df_val,   cfg["environment"])
    agent     = build_agent(env_train, cfg["model"])

    agent = train(env_train, env_val, agent, cfg["training"])
    agent.save("models/dqn_final")
"""
from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium import Env


def train(
    env_train: Env,
    env_val: Env,
    agent: DQN,
    config: dict,
) -> DQN:
    """
    Train the DQN agent with validation-based early stopping.

    Args:
        env_train:
            Gymnasium environment wrapping the training dataset.
        env_val:
            Gymnasium environment wrapping the validation dataset.
            Used only for evaluation — never for learning.
        agent:
            Untrained (or partially trained) SB3 DQN instance.
        config:
            Dict matching the ``training`` block in ``configs/dqn_config.yaml``.
            Keys (all optional — defaults shown):
                total_timesteps        (int)   100_000
                eval_freq              (int)   5_000   — steps between evals
                n_eval_episodes        (int)   1
                save_freq              (int)   10_000  — checkpoint interval
                patience               (int)   5       — evals with no improvement
                                                         before early stopping
                best_model_save_path   (str)   "models/best"
                checkpoint_save_path   (str)   "models/checkpoints"
                log_path               (str)   "results/logs"
                verbose                (int)   1

    Returns:
        The trained DQN agent (same object, mutated in place by SB3).
    """
    total_timesteps = config.get("total_timesteps", 100_000)
    eval_freq       = config.get("eval_freq", 5_000)
    n_eval_episodes = config.get("n_eval_episodes", 1)
    save_freq       = config.get("save_freq", 10_000)
    patience        = config.get("patience", 5)
    verbose         = config.get("verbose", 1)

    best_model_path  = Path(config.get("best_model_save_path", "models/best"))
    checkpoint_path  = Path(config.get("checkpoint_save_path", "models/checkpoints"))
    log_path         = Path(config.get("log_path", "results/logs"))

    best_model_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)

    # SB3 EvalCallback requires the eval env to be a VecEnv
    eval_vec_env = DummyVecEnv([lambda: env_val])

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #

    # Early-stopping: stop if val reward doesn't improve for `patience` evals
    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=patience,
        min_evals=patience,          # don't stop before at least this many evals
        verbose=verbose,
    )

    # Eval + best-model save
    eval_callback = EvalCallback(
        eval_env=eval_vec_env,
        n_eval_episodes=n_eval_episodes,
        eval_freq=eval_freq,
        log_path=str(log_path),
        best_model_save_path=str(best_model_path),
        callback_after_eval=stop_callback,
        deterministic=True,
        render=False,
        verbose=verbose,
    )

    # Periodic checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(checkpoint_path),
        name_prefix="dqn_checkpoint",
        verbose=0,
    )

    callbacks = CallbackList([eval_callback, checkpoint_callback])

    # ------------------------------------------------------------------ #
    # Train
    # ------------------------------------------------------------------ #
    if verbose:
        print(f"Training for up to {total_timesteps:,} timesteps "
              f"(eval every {eval_freq:,} steps, patience={patience} evals).")

    agent.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        reset_num_timesteps=True,
    )

    if verbose:
        print(f"Training complete.  Best model saved → {best_model_path}/best_model.zip")

    return agent
