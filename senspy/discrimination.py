import numpy as np
from scipy.stats import norm

__all__ = ["two_afc", "duotrio_pc", "discrim_2afc"]

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


def discrim_2afc(correct: int, total: int) -> dict:
    """Estimate d-prime and standard error for a 2-AFC test."""
    if total <= 0:
        raise ValueError("total must be positive")
    if correct < 0 or correct > total:
        raise ValueError("correct must be between 0 and total")

    pc = correct / total
    dp = np.sqrt(2.0) * norm.ppf(pc)
    se_pc = np.sqrt(pc * (1 - pc) / total)
    deriv = norm.pdf(dp / np.sqrt(2.0)) / np.sqrt(2.0)
    se_dp = se_pc / deriv
    return {"d_prime": dp, "se": se_dp}
