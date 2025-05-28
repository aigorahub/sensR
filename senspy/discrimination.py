import numpy as np
from scipy.stats import norm

__all__ = [
    "two_afc",
    "duotrio_pc",
    "discrim_2afc",
    "pc2pd",
    "pd2pc",
    "get_pguess",
]

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
    """Estimate d-prime from 2-AFC data using the Gaussian model.

    Parameters
    ----------
    correct : int
        Number of correct responses.
    total : int
        Total number of trials.

    Returns
    -------
    dict
        Dictionary containing ``d_prime`` and ``se`` (standard error).
    """
    if total <= 0 or correct < 0 or correct > total:
        raise ValueError("invalid counts")
    pc = correct / total
    # avoid 0 or 1 which lead to infinite estimates
    eps = 0.5 / total
    pc = min(max(pc, eps), 1 - eps)
    d_prime = np.sqrt(2.0) * norm.ppf(pc)
    # standard error via delta method
    se_p = np.sqrt(pc * (1 - pc) / total)
    deriv = np.sqrt(2.0) / norm.pdf(norm.ppf(pc))
    se = se_p * deriv
    return {"d_prime": d_prime, "se": se}


def get_pguess(method: str, double: bool = False) -> float:
    """Return guessing probability for a discrimination protocol."""
    method = method.lower()
    mapping = {
        "duotrio": 0.5,
        "twoafc": 0.5,
        "threeafc": 1 / 3,
        "triangle": 1 / 3,
        "tetrad": 1 / 3,
        "hexad": 1 / 10,
        "twofive": 1 / 10,
        "twofivef": 2 / 5,
    }
    if method not in mapping:
        raise ValueError(f"unknown method '{method}'")
    pg = mapping[method]
    if double:
        if method in {"hexad", "twofive", "twofivef"}:
            raise NotImplementedError("double version not implemented")
        return pg ** 2
    return pg


def pc2pd(pc: float, p_guess: float) -> float:
    """Convert proportion correct to proportion discriminated."""
    if not 0 <= p_guess <= 1:
        raise ValueError("Pguess must be between 0 and 1")
    if not 0 <= pc <= 1:
        raise ValueError("pc must be between 0 and 1")
    pd = (pc - p_guess) / (1 - p_guess)
    return 0.0 if pc <= p_guess else pd


def pd2pc(pd: float, p_guess: float) -> float:
    """Convert proportion discriminated to proportion correct."""
    if not 0 <= p_guess <= 1:
        raise ValueError("Pguess must be between 0 and 1")
    if not 0 <= pd <= 1:
        raise ValueError("pd must be between 0 and 1")
    return p_guess + pd * (1 - p_guess)
