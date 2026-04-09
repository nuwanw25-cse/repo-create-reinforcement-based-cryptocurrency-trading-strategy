"""
Hyperparameter search (grid or random) for the DQN agent.
"""


def grid_search(param_grid: dict, env_train, env_val):
    """Exhaustive grid search over param_grid combinations."""
    raise NotImplementedError


def random_search(param_distributions: dict, n_iter: int, env_train, env_val):
    """Random sampling from param_distributions."""
    raise NotImplementedError
