"""ROC curves, AUC, and Signal Detection Theory functions.

This module provides functions for:
- SDT (Signal Detection Theory) d-prime computation from contingency tables
- ROC (Receiver Operating Characteristic) curve generation
- AUC (Area Under the Curve) computation with confidence intervals

References
----------
Macmillan, N.A. & Creelman, C.D. (2005). Detection Theory: A User's Guide.
    2nd ed. Lawrence Erlbaum Associates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats


def sdt(
    table: ArrayLike,
    method: Literal["probit", "logit"] = "probit",
) -> NDArray[np.float64]:
    """Compute Signal Detection Theory d-prime from a contingency table.

    Performs the empirical probit or logit transform on a 2 x J table of
    observations to compute d-prime values at each response criterion.

    Parameters
    ----------
    table : array_like
        A 2 x J matrix where rows represent signal/noise conditions and
        columns represent J ordered response categories. Row 1 is typically
        the "signal" distribution, row 2 is the "noise" distribution.
    method : {"probit", "logit"}, default "probit"
        The transform method to use:
        - "probit": Uses the normal quantile function (qnorm)
        - "logit": Uses the logistic transform

    Returns
    -------
    result : ndarray
        A (J-1) x 3 matrix with columns:
        - z(Hit rate): Transformed cumulative hit rates
        - z(False alarm rate): Transformed cumulative false alarm rates
        - d-prime: The difference z(Hit) - z(FA) at each criterion

    Raises
    ------
    ValueError
        If table is not a 2-row matrix or method is not recognized.

    Examples
    --------
    >>> import numpy as np
    >>> from senspy import sdt
    >>> # Example: rating scale data with 5 categories
    >>> table = np.array([[10, 20, 30, 25, 15],   # Signal trials
    ...                   [30, 25, 20, 15, 10]])  # Noise trials
    >>> result = sdt(table)
    >>> print(result)

    Notes
    -----
    The d-prime at each criterion represents the separation between the
    signal and noise distributions at that point on the ROC curve.
    """
    table = np.asarray(table, dtype=np.float64)

    if table.ndim != 2 or table.shape[0] != 2:
        raise ValueError("table must be a 2 x J matrix (2 rows, J columns)")

    if method not in ("probit", "logit"):
        raise ValueError(f"method '{method}' not recognized; use 'probit' or 'logit'")

    def transform_row(x: NDArray) -> NDArray:
        """Apply cumulative transform to a row."""
        cs = np.cumsum(x)
        total = cs[-1]

        if method == "probit":
            # Cumulative proportions (excluding last point which is always 1)
            cp = cs[:-1] / total
            return stats.norm.ppf(cp)
        else:  # logit
            return np.log(cs[:-1] / (total - cs[:-1]))

    # Apply transform to each row
    z_signal = transform_row(table[0, :])  # z(Hit rate)
    z_noise = transform_row(table[1, :])  # z(False alarm rate)

    # Compute d-prime at each criterion
    d_prime = z_signal - z_noise

    # Build result matrix
    n_criteria = table.shape[1] - 1
    result = np.column_stack([z_signal, z_noise, d_prime])

    return result


@dataclass
class AUCResult:
    """Result from AUC computation.

    Attributes
    ----------
    value : float
        The Area Under the ROC Curve.
    lower : float or None
        Lower bound of confidence interval (if se_d provided).
    upper : float or None
        Upper bound of confidence interval (if se_d provided).
    ci_alpha : float or None
        Alpha level used for confidence interval.
    """

    value: float
    lower: float | None = None
    upper: float | None = None
    ci_alpha: float | None = None


def auc(
    d: float,
    se_d: float | None = None,
    scale: float = 1.0,
    ci_alpha: float = 0.05,
) -> AUCResult:
    """Compute Area Under the ROC Curve from d-prime.

    Calculates the AUC analytically from the d-prime value, optionally
    with confidence intervals based on the standard error.

    Parameters
    ----------
    d : float
        The d-prime (discriminability) value. Can be negative.
    se_d : float, optional
        Standard error of d-prime. If provided, confidence intervals
        are computed.
    scale : float, default 1.0
        Scale parameter (ratio of noise to signal standard deviation).
        Must be positive.
    ci_alpha : float, default 0.05
        Significance level for confidence interval (e.g., 0.05 for 95% CI).

    Returns
    -------
    AUCResult
        Result containing AUC value and optional confidence bounds.

    Raises
    ------
    ValueError
        If inputs are invalid.

    Examples
    --------
    >>> from senspy import auc
    >>> result = auc(d=1.5, se_d=0.2)
    >>> print(f"AUC: {result.value:.3f}")
    AUC: 0.856
    >>> print(f"95% CI: [{result.lower:.3f}, {result.upper:.3f}]")
    95% CI: [0.793, 0.905]

    Notes
    -----
    The AUC is computed as:
        AUC = Φ(d / √(1 + scale²))

    where Φ is the standard normal CDF. This formula assumes equal-variance
    Gaussian signal detection theory when scale=1.

    For unequal variance models (scale ≠ 1), the ROC curve is asymmetric
    and the formula accounts for the variance ratio.
    """
    # Validate inputs
    if not np.isfinite(d):
        raise ValueError("d must be a finite number")
    if scale <= 0:
        raise ValueError("scale must be positive")
    if not (0 < ci_alpha < 1):
        raise ValueError("ci_alpha must be between 0 and 1")

    if se_d is not None:
        if not np.isfinite(se_d) or se_d < 0:
            raise ValueError("se_d must be a non-negative finite number")

    # Compute AUC
    scale_factor = np.sqrt(1 + scale**2)
    auc_value = float(stats.norm.cdf(d / scale_factor))

    # Compute confidence interval if se_d provided
    lower = None
    upper = None
    result_ci_alpha = None

    if se_d is not None:
        tol = se_d * stats.norm.ppf(1 - ci_alpha / 2)
        lower = float(stats.norm.cdf((d - tol) / scale_factor))
        upper = float(stats.norm.cdf((d + tol) / scale_factor))
        result_ci_alpha = ci_alpha

    return AUCResult(
        value=auc_value,
        lower=lower,
        upper=upper,
        ci_alpha=result_ci_alpha,
    )


@dataclass
class ROCResult:
    """Result from ROC curve computation.

    Attributes
    ----------
    fpr : ndarray
        False positive rates (x-axis of ROC curve).
    tpr : ndarray
        True positive rates (y-axis of ROC curve).
    lower : ndarray or None
        Lower confidence bound for TPR (if se_d provided).
    upper : ndarray or None
        Upper confidence bound for TPR (if se_d provided).
    d_prime : float
        The d-prime value used.
    scale : float
        The scale parameter used.
    """

    fpr: NDArray[np.float64]
    tpr: NDArray[np.float64]
    lower: NDArray[np.float64] | None = None
    upper: NDArray[np.float64] | None = None
    d_prime: float = 0.0
    scale: float = 1.0


def roc(
    d: float,
    se_d: float | None = None,
    scale: float = 1.0,
    n_points: int = 1000,
    se_type: Literal["CI", "SE"] = "CI",
    ci_alpha: float = 0.05,
) -> ROCResult:
    """Compute ROC curve from d-prime.

    Generates the theoretical ROC curve for a given d-prime value,
    optionally with confidence bands.

    Parameters
    ----------
    d : float
        The d-prime (discriminability) value.
    se_d : float, optional
        Standard error of d-prime. If provided, confidence bands
        are computed.
    scale : float, default 1.0
        Scale parameter (ratio of noise to signal standard deviation).
    n_points : int, default 1000
        Number of points to generate on the ROC curve.
    se_type : {"CI", "SE"}, default "CI"
        Type of error band:
        - "CI": Confidence interval (se_d * z_alpha)
        - "SE": Standard error (se_d)
    ci_alpha : float, default 0.05
        Significance level for confidence interval (only used if se_type="CI").

    Returns
    -------
    ROCResult
        Result containing FPR, TPR arrays and optional confidence bounds.

    Raises
    ------
    ValueError
        If inputs are invalid.

    Examples
    --------
    >>> from senspy import roc
    >>> result = roc(d=1.5, se_d=0.2)
    >>> print(f"AUC ≈ {np.trapz(result.tpr, result.fpr):.3f}")

    Notes
    -----
    The ROC curve is computed using:
        TPR = Φ((Φ⁻¹(FPR) + d) / scale)

    where Φ is the standard normal CDF and Φ⁻¹ is the quantile function.

    For equal-variance SDT (scale=1), this simplifies to:
        TPR = Φ(Φ⁻¹(FPR) + d)
    """
    # Validate inputs
    if not np.isfinite(d):
        raise ValueError("d must be a finite number")
    if scale <= 0:
        raise ValueError("scale must be positive")
    if n_points < 2:
        raise ValueError("n_points must be at least 2")
    if se_type not in ("CI", "SE"):
        raise ValueError("se_type must be 'CI' or 'SE'")

    if se_d is not None:
        if not np.isfinite(se_d) or se_d < 0:
            raise ValueError("se_d must be a non-negative finite number")

    # Generate FPR values
    fpr = np.linspace(0, 1, n_points)

    # Compute z-scores of FPR (avoiding infinities at 0 and 1)
    # Use small offset to avoid -inf and +inf
    fpr_safe = np.clip(fpr, 1e-10, 1 - 1e-10)
    z_fpr = stats.norm.ppf(fpr_safe)

    # Compute TPR
    tpr = stats.norm.cdf((z_fpr + d) / scale)

    # Compute confidence bands if se_d provided
    lower = None
    upper = None

    if se_d is not None:
        if se_type == "CI":
            tol = se_d * stats.norm.ppf(1 - ci_alpha / 2)
        else:  # SE
            tol = se_d

        lower = stats.norm.cdf((z_fpr + d - tol) / scale)
        upper = stats.norm.cdf((z_fpr + d + tol) / scale)

    return ROCResult(
        fpr=fpr,
        tpr=tpr,
        lower=lower,
        upper=upper,
        d_prime=d,
        scale=scale,
    )
