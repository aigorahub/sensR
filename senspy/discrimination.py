import numpy as np
from scipy.stats import norm

__all__ = ["two_afc", "duotrio_pc"]

def two_afc(dprime: float) -> float:
    """Proportion correct in a 2-AFC task for a given d-prime."""
    return norm.cdf(dprime / np.sqrt(2))


def duotrio_pc(dprime: float) -> float:
    """Proportion correct in a duo-trio test for a given d-prime."""
    if dprime <= 0:
        return 0.5
    a = norm.cdf(dprime / np.sqrt(2.0))
    b = norm.cdf(dprime / np.sqrt(6.0))
    return 1 - a - b + 2 * a * b
