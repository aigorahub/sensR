"""A-Not-A (AnotA) protocol for sensory discrimination.

This module implements the A-Not-A discrimination protocol, which is a
signal detection method where assessors are presented with samples and
must identify whether each is "A" or "Not-A".

The protocol involves:
- A samples: n1 presentations, x1 correctly identified as "A"
- Not-A samples: n2 presentations, x2 correctly identified as "Not-A"

The d-prime is estimated using probit regression.

References
----------
Ennis, J.M. (1993). The power of sensory discrimination methods.
Journal of Sensory Studies, 8(4), 353-370.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats
from scipy.stats import fisher_exact


@dataclass
class ANotAResult:
    """Result from A-Not-A analysis.

    Attributes
    ----------
    d_prime : float
        Estimated d-prime (discriminability).
    se_d_prime : float
        Standard error of d-prime.
    p_value : float
        P-value from Fisher's exact test.
    hit_rate : float
        Proportion of A samples correctly identified (x1/n1).
    false_alarm_rate : float
        Proportion of Not-A samples incorrectly identified as A (1 - x2/n2).
    data : dict
        Input data: x1, n1, x2, n2.
    """

    d_prime: float
    se_d_prime: float
    p_value: float
    hit_rate: float
    false_alarm_rate: float
    data: dict


def anota(
    x1: int,
    n1: int,
    x2: int,
    n2: int,
) -> ANotAResult:
    """Fit A-Not-A discrimination model.

    The A-Not-A protocol presents assessors with samples that are either
    "A" or "Not-A", and they must identify which type each sample is.

    Parameters
    ----------
    x1 : int
        Number of A samples correctly identified as "A" (hits).
    n1 : int
        Total number of A samples presented.
    x2 : int
        Number of Not-A samples correctly identified as "Not-A".
    n2 : int
        Total number of Not-A samples presented.

    Returns
    -------
    ANotAResult
        Analysis results including d-prime estimate.

    Raises
    ------
    ValueError
        If inputs are invalid.

    Examples
    --------
    >>> result = anota(x1=80, n1=100, x2=70, n2=100)
    >>> print(f"d-prime: {result.d_prime:.3f}")
    >>> print(f"p-value: {result.p_value:.4f}")

    Notes
    -----
    d-prime is computed using probit regression:
    - Hit rate (H) = x1/n1
    - False alarm rate (F) = (n2-x2)/n2 = 1 - x2/n2
    - d' = z(H) - z(F)

    The p-value is from Fisher's exact test for testing whether there
    is a difference in discrimination ability.
    """
    # Validate inputs
    for name, val in [("x1", x1), ("n1", n1), ("x2", x2), ("n2", n2)]:
        if not isinstance(val, (int, np.integer)):
            if isinstance(val, float) and val == int(val):
                val = int(val)
            else:
                raise ValueError(f"{name} must be a positive integer")
        if val <= 0:
            raise ValueError(f"{name} must be a positive integer")

    x1, n1, x2, n2 = int(x1), int(n1), int(x2), int(n2)

    if x1 > n1:
        raise ValueError("x1 cannot exceed n1")
    if x2 > n2:
        raise ValueError("x2 cannot exceed n2")

    # Compute hit rate and false alarm rate
    hit_rate = x1 / n1
    false_alarm_rate = (n2 - x2) / n2  # FA = proportion of Not-A called "A"

    # Compute d-prime using probit (z-scores)
    # d' = z(H) - z(FA)
    # Need to handle edge cases where rates are 0 or 1
    def safe_probit(p, n):
        """Compute probit with adjustment for extreme values."""
        # Apply 1/(2n) correction (Macmillan & Kaplan, 1985)
        # This avoids infinite z-scores at p=0 or p=1
        if p <= 0:
            p = 0.5 / n
        elif p >= 1:
            p = (n - 0.5) / n
        return stats.norm.ppf(p)

    z_hit = safe_probit(hit_rate, n1)
    z_fa = safe_probit(false_alarm_rate, n2)

    d_prime = z_hit - z_fa

    # Compute standard error using delta method
    # Var(d') ≈ Var(z_H) + Var(z_FA)
    # Var(z(p)) ≈ p(1-p) / [n * φ(z(p))^2] where φ is std normal pdf
    def var_probit(p, n):
        """Variance of probit-transformed proportion."""
        if p <= 0 or p >= 1:
            p = np.clip(p, 0.01, 0.99)
        z = stats.norm.ppf(p)
        phi = stats.norm.pdf(z)
        if phi < 1e-10:
            return np.inf
        return p * (1 - p) / (n * phi**2)

    var_z_hit = var_probit(hit_rate, n1)
    var_z_fa = var_probit(false_alarm_rate, n2)

    se_d_prime = np.sqrt(var_z_hit + var_z_fa)

    # Fisher's exact test
    # Contingency table:
    #           A    Not-A
    # "A"       x1   n2-x2
    # "Not-A"   n1-x1  x2
    table = np.array([[x1, n2 - x2], [n1 - x1, x2]])
    _, p_value = fisher_exact(table, alternative="greater")

    return ANotAResult(
        d_prime=d_prime,
        se_d_prime=se_d_prime,
        p_value=p_value,
        hit_rate=hit_rate,
        false_alarm_rate=false_alarm_rate,
        data={"x1": x1, "n1": n1, "x2": x2, "n2": n2},
    )
