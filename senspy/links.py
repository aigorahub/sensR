import numpy as np
from scipy.stats import norm

__all__ = ["psyfun", "psyinv", "psyderiv", "rescale"]

def psyfun(dprime: float) -> float:
    """Map d-prime to proportion correct using the Gaussian model."""
    return norm.cdf(dprime / np.sqrt(2.0))

def psyinv(p: float) -> float:
    """Map proportion correct to d-prime."""
    return np.sqrt(2.0) * norm.ppf(p)

def psyderiv(dprime: float) -> float:
    """Derivative of :func:`psyfun` with respect to d-prime."""
    x = dprime / np.sqrt(2.0)
    return norm.pdf(x) / np.sqrt(2.0)

def rescale(x: float, from_scale: str, to_scale: str) -> float:
    """Rescale between proportion correct (pc), proportion discriminated (pd), and d-prime (dp)."""
    scale = {"dp": "dp", "pc": "pc", "pd": "pd"}
    if from_scale == to_scale:
        return x
    # convert input to dp
    if from_scale == "pc":
        dp = psyinv(x)
    elif from_scale == "pd":
        dp = psyinv((x + 1) / 2)
    else:
        dp = x
    # convert dp to target
    if to_scale == "pc":
        return psyfun(dp)
    elif to_scale == "pd":
        return 2 * psyfun(dp) - 1
    else:
        return dp
