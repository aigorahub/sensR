import numpy as np
from scipy.stats import norm

__all__ = ["beta_binomial_power"]

def beta_binomial_power(n: int, alpha: float, beta: float, dprime: float) -> float:
    """Very rough power estimate for beta-binomial model."""
    mu = alpha / (alpha + beta)
    sd = np.sqrt(mu * (1 - mu) / n)
    pc = norm.cdf(dprime / np.sqrt(2))
    z = (pc - mu) / sd
    return norm.cdf(z)
