import numpy as np
from scipy.stats import norm, ncf
from scipy.integrate import quad

__all__ = [
    "two_afc",
    "duotrio_pc",
    "three_afc_pc",
    "triangle_pc",
    "tetrad_pc",
    "hexad_pc",
    "twofive_pc",
    "get_pguess",
    "pc2pd",
    "pd2pc",
    "discrim_2afc",
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


def three_afc_pc(dprime: float) -> float:
    """Proportion correct in a 3-AFC test for a given d-prime."""
    if dprime <= 0:
        return 1.0 / 3.0

    def integrand(x: float) -> float:
        return norm.pdf(x - dprime) * norm.cdf(x) ** 2

    val, _ = quad(integrand, -np.inf, np.inf)
    return min(max(val, 1.0 / 3.0), 1.0)


def triangle_pc(dprime: float) -> float:
    """Proportion correct in a triangle test for a given d-prime."""
    if dprime <= 0:
        return 1.0 / 3.0
    val = ncf.sf(3.0, 1, 1, dprime ** 2 * 2.0 / 3.0)
    return min(max(val, 1.0 / 3.0), 1.0)


def tetrad_pc(dprime: float) -> float:
    """Proportion correct in a tetrad test for a given d-prime."""
    if dprime <= 0:
        return 1.0 / 3.0

    def integrand(z: float) -> float:
        c1 = norm.cdf(z)
        c2 = norm.cdf(z - dprime)
        return norm.pdf(z) * (2 * c1 * c2 - c2 ** 2)

    val, _ = quad(integrand, -np.inf, np.inf)
    res = 1.0 - 2.0 * val
    return min(max(res, 1.0 / 3.0), 1.0)


_HEXAD_COEFFS = [
    0.0977646147,
    0.0319804414,
    0.0656128284,
    0.1454153496,
    -0.0994639381,
    0.0246960778,
    -0.0027806267,
    0.0001198169,
]


def hexad_pc(dprime: float) -> float:
    """Polynomial approximation for the hexad test."""
    if dprime <= 0:
        return 0.1
    if dprime >= 5.368:
        return 1.0
    x = dprime
    val = 0.0
    for i, c in enumerate(_HEXAD_COEFFS):
        val += c * x ** i
    return min(max(val, 0.1), 1.0)


_TWOFIVE_COEFFS = [
    0.0988496065454,
    0.0146108899965,
    0.0708075379445,
    0.0568876949069,
    -0.0424936635277,
    0.0114595626175,
    -0.0016573180506,
    0.0001372413489,
    -0.0000061598395,
    0.0000001166556,
]


def twofive_pc(dprime: float) -> float:
    """Polynomial approximation for the two-out-of-five test."""
    if dprime <= 0:
        return 0.1
    if dprime >= 9.28:
        return 1.0
    x = dprime
    val = 0.0
    for i, c in enumerate(_TWOFIVE_COEFFS):
        val += c * x ** i
    return min(max(val, 0.1), 1.0)


def get_pguess(method: str, double: bool = False) -> float:
    """Return chance performance for a given method."""
    method = method.lower()
    mapping = {
        "duotrio": 0.5,
        "twoafc": 0.5,
        "threeafc": 1.0 / 3.0,
        "triangle": 1.0 / 3.0,
        "tetrad": 1.0 / 3.0,
        "hexad": 0.1,
        "twofive": 0.1,
        "twofivef": 0.4,
    }
    p = mapping.get(method, 0.5)
    return p ** 2 if double else p


def pc2pd(pc: float, pguess: float) -> float:
    """Convert proportion correct to proportion discriminated."""
    if pc <= pguess:
        return 0.0
    return (pc - pguess) / (1.0 - pguess)


def pd2pc(pd: float, pguess: float) -> float:
    """Convert proportion discriminated to proportion correct."""
    if pd <= 0:
        return pguess
    return pguess + pd * (1.0 - pguess)


def discrim_2afc(correct: int, total: int) -> tuple[float, float]:
    """Estimate d-prime from 2-AFC data.

    Returns a tuple of (estimate, standard_error).
    """
    pc = correct / total
    pc = max(min(pc, 1 - 1e-8), 0.5 + 1e-8)
    dprime = np.sqrt(2.0) * norm.ppf(pc)
    se = np.sqrt(pc * (1 - pc) / total) * np.sqrt(2.0) / norm.pdf(norm.ppf(pc))
    return dprime, se
