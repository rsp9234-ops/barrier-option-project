# src/monte_carlo.py

import time
import numpy as np
from typing import Tuple
from .config import BarrierOptionParams
from .barrier_option import barrier_payoff_from_path


def simulate_gbm_paths(
    params: BarrierOptionParams,
    n_paths: int,
    n_steps: int,
    antithetic: bool = True,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate GBM paths for the underlying asset using Euler discretization in log-space.
    Returns an array of shape (n_paths, n_steps + 1).
    """
    if seed is not None:
        np.random.seed(seed)

    S0, r, sigma, T = params.S0, params.r, params.sigma, params.T
    dt = T / n_steps

    # number of *base* paths (we will double them with antithetic if True)
    base_paths = n_paths if not antithetic else (n_paths + 1) // 2

    # Normal shocks for base paths
    Z = np.random.normal(size=(base_paths, n_steps))
    if antithetic:
        Z = np.vstack([Z, -Z])
    Z = Z[:n_paths, :]

    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)

    # log S paths
    log_S = np.zeros((n_paths, n_steps + 1))
    log_S[:, 0] = np.log(S0)

    for t in range(1, n_steps + 1):
        log_S[:, t] = log_S[:, t - 1] + drift + diffusion * Z[:, t - 1]

    S_paths = np.exp(log_S)
    return S_paths


def price_barrier_monte_carlo(
    params: BarrierOptionParams,
    n_paths: int,
    n_steps: int,
    antithetic: bool = True,
    seed: int | None = None,
) -> Tuple[float, float, float]:
    """
    Monte Carlo price for a barrier option.

    Returns:
        price: MC estimate of option price
        std_error: standard error of the estimate
        runtime: wall-clock time in seconds
    """
    start_time = time.time()

    S_paths = simulate_gbm_paths(params, n_paths, n_steps, antithetic=antithetic, seed=seed)

    # Compute discounted payoffs
    payoffs = np.zeros(n_paths)
    for i in range(n_paths):
        payoffs[i] = barrier_payoff_from_path(S_paths[i, :], params)

    discount_factor = np.exp(-params.r * params.T)
    discounted_payoffs = discount_factor * payoffs

    price = discounted_payoffs.mean()
    std_error = discounted_payoffs.std(ddof=1) / np.sqrt(n_paths)
    runtime = time.time() - start_time

    return price, std_error, runtime