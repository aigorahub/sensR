import numpy as np
from scipy.stats import norm

__all__ = ["two_afc"]

def two_afc(dprime: float) -> float:
    """Proportion correct in a 2-AFC task for a given d-prime."""
    return norm.cdf(dprime / np.sqrt(2))
