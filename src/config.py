# src/config.py

from dataclasses import dataclass

@dataclass
class BarrierOptionParams:
    S0: float        # Initial stock price
    K: float         # Strike price
    r: float         # Risk-free rate (annual, cont. compounding)
    sigma: float     # Volatility (annual)
    T: float         # Maturity in years
    barrier: float   # Barrier level
    barrier_type: str = "down-and-out"  # "down-and-out" or "up-and-out"
    option_type: str = "call"           # "call" or "put"


# A default parameter set for the project
DEFAULT_PARAMS = BarrierOptionParams(
    S0=100.0,
    K=100.0,
    r=0.05,
    sigma=0.2,
    T=1.0,
    barrier=90.0,
    barrier_type="down-and-out",
    option_type="call",
)
