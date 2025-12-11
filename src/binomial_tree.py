# src/binomial_tree.py

import math
import time
from typing import Tuple
from .config import BarrierOptionParams
from .barrier_option import vanilla_payoff


def price_barrier_binomial_tree(
    params: BarrierOptionParams,
    n_steps: int,
) -> Tuple[float, float]:
    """
    Price a barrier option using a Cox-Ross-Rubinstein binomial tree.

    Barrier is checked at each node (discrete monitoring).
    Returns:
        price: tree price
        runtime: wall-clock time in seconds
    """
    start_time = time.time()

    S0, K, r, sigma, T = params.S0, params.K, params.r, params.sigma, params.T
    B = params.barrier

    dt = T / n_steps
    disc = math.exp(-r * dt)
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    a = math.exp(r * dt)
    p = (a - d) / (u - d)

    # stock prices and option values at maturity
    stock_prices = [0.0] * (n_steps + 1)
    option_values = [0.0] * (n_steps + 1)

    for j in range(n_steps + 1):
        # S_T(j) = S0 * u^j * d^(N-j)
        S_T = S0 * (u**j) * (d ** (n_steps - j))
        stock_prices[j] = S_T

        # At maturity, if barrier already crossed at this node, approximate knockout
        if (params.barrier_type == "down-and-out" and S_T <= B) or \
           (params.barrier_type == "up-and-out" and S_T >= B):
            option_values[j] = 0.0
        else:
            option_values[j] = vanilla_payoff(S_T, params)

    # Backward induction
    for i in range(n_steps - 1, -1, -1):
        for j in range(i + 1):
            # Stock price at node (i, j)
            S_ij = S0 * (u**j) * (d ** (i - j))

            # If barrier violated at this node, knock out
            if (params.barrier_type == "down-and-out" and S_ij <= B) or \
               (params.barrier_type == "up-and-out" and S_ij >= B):
                option_values[j] = 0.0
            else:
                # risk-neutral expected discounted value
                option_values[j] = disc * (p * option_values[j + 1] + (1 - p) * option_values[j])

    price = option_values[0]
    runtime = time.time() - start_time

    return price, runtime
