# src/barrier_option.py

import numpy as np
from .config import BarrierOptionParams


def barrier_hit(path: np.ndarray, params: BarrierOptionParams) -> bool:
    """
    Check if the barrier has been hit along the simulated price path.
    path: array of stock prices along time [S_0, S_1, ..., S_T]
    """
    B = params.barrier
    if params.barrier_type == "down-and-out":
        return np.any(path <= B)
    elif params.barrier_type == "up-and-out":
        return np.any(path >= B)
    else:
        raise ValueError(f"Unknown barrier_type: {params.barrier_type}")


def vanilla_payoff(ST: float, params: BarrierOptionParams) -> float:
    """Vanilla European payoff (no barrier)."""
    if params.option_type == "call":
        return max(ST - params.K, 0.0)
    elif params.option_type == "put":
        return max(params.K - ST, 0.0)
    else:
        raise ValueError(f"Unknown option_type: {params.option_type}")


def barrier_payoff_from_path(path: np.ndarray, params: BarrierOptionParams) -> float:
    """
    Barrier option payoff given a full path of S_t.
    Barrier is monitored continuously at time grid points of the simulation.
    """
    if barrier_hit(path, params):
        return 0.0  # knocked out
    ST = path[-1]
    return vanilla_payoff(ST, params)