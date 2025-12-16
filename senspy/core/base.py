"""Base classes for sensPy result objects.

This module defines base classes for result objects returned by statistical
functions. The design follows the statsmodels pattern where functions return
result objects with methods for inference.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from senspy.core.types import Protocol, Statistic


@dataclass
class DiscrimResult:
    """Result from a discrimination analysis.

    This class holds the results from the `discrim()` function, including
    point estimates, standard errors, confidence intervals, and test results.

    Attributes
    ----------
    d_prime : float
        Estimated d-prime (sensitivity measure).
    pc : float
        Proportion correct (estimated).
    pd : float
        Proportion discriminators.
    se_d_prime : float
        Standard error of d-prime estimate.
    se_pc : float
        Standard error of proportion correct.
    se_pd : float
        Standard error of proportion discriminators.
    p_value : float
        P-value from the significance test.
    statistic : float
        Value of the test statistic.
    stat_type : Statistic
        Type of test statistic used.
    method : Protocol
        Discrimination protocol used.
    correct : int
        Number of correct responses.
    total : int
        Total number of trials.
    conf_level : float
        Confidence level for intervals (default 0.95).
    loglik : float | None
        Log-likelihood at the MLE (if computed).

    Examples
    --------
    >>> from senspy import discrim
    >>> result = discrim(correct=80, total=100, method="triangle")
    >>> result.d_prime
    2.164...
    >>> result.confint()
    array([1.65..., 2.75...])
    """

    d_prime: float
    pc: float
    pd: float
    se_d_prime: float
    se_pc: float
    se_pd: float
    p_value: float
    statistic: float
    stat_type: Statistic
    method: Protocol
    correct: int
    total: int
    conf_level: float = 0.95
    loglik: float | None = None
    _ci_d_prime: NDArray[np.floating] | None = field(default=None, repr=False)
    _ci_pc: NDArray[np.floating] | None = field(default=None, repr=False)
    _ci_pd: NDArray[np.floating] | None = field(default=None, repr=False)

    @property
    def coefficients(self) -> dict[str, dict[str, float]]:
        """Return coefficients as a dictionary.

        Returns
        -------
        dict
            Dictionary with 'pc', 'pd', 'd_prime' keys, each containing
            'estimate' and 'se' values.

        Examples
        --------
        >>> result.coefficients['d_prime']['estimate']
        2.164...
        """
        return {
            "pc": {"estimate": self.pc, "se": self.se_pc},
            "pd": {"estimate": self.pd, "se": self.se_pd},
            "d_prime": {"estimate": self.d_prime, "se": self.se_d_prime},
        }

    def confint(
        self, level: float | None = None, parameter: str = "d_prime"
    ) -> NDArray[np.floating]:
        """Compute confidence intervals.

        Parameters
        ----------
        level : float, optional
            Confidence level (0 < level < 1). If None, uses the level
            from the original analysis.
        parameter : str, default "d_prime"
            Which parameter to compute CI for: "d_prime", "pc", or "pd".

        Returns
        -------
        ndarray
            Array of shape (2,) with [lower, upper] bounds.

        Examples
        --------
        >>> result.confint()
        array([1.65..., 2.75...])
        >>> result.confint(level=0.99)
        array([1.45..., 2.95...])
        """
        from scipy import stats

        if level is None:
            level = self.conf_level

        alpha = 1 - level
        z = stats.norm.ppf(1 - alpha / 2)

        if parameter == "d_prime":
            estimate, se = self.d_prime, self.se_d_prime
            lower_bound = 0.0  # d-prime cannot be negative
        elif parameter == "pc":
            estimate, se = self.pc, self.se_pc
            lower_bound = self.method.p_guess
        elif parameter == "pd":
            estimate, se = self.pd, self.se_pd
            lower_bound = 0.0
        else:
            raise ValueError(f"Unknown parameter: {parameter}")

        lower = max(lower_bound, estimate - z * se)
        upper = estimate + z * se

        return np.array([lower, upper])

    def summary(self) -> str:
        """Return a formatted summary of the results.

        Returns
        -------
        str
            A formatted string with the analysis results.

        Examples
        --------
        >>> print(result.summary())
        Discrimination Analysis (triangle)
        ...
        """
        ci = self.confint()
        lines = [
            f"Discrimination Analysis ({self.method.value})",
            "=" * 45,
            f"Data: {self.correct} correct out of {self.total} trials",
            "",
            "Estimates:",
            f"  {'Parameter':<12} {'Estimate':>10} {'Std.Err':>10}",
            f"  {'-' * 12} {'-' * 10} {'-' * 10}",
            f"  {'pc':<12} {self.pc:>10.4f} {self.se_pc:>10.4f}",
            f"  {'pd':<12} {self.pd:>10.4f} {self.se_pd:>10.4f}",
            f"  {'d.prime':<12} {self.d_prime:>10.4f} {self.se_d_prime:>10.4f}",
            "",
            f"{self.conf_level:.0%} Confidence Interval for d-prime:",
            f"  [{ci[0]:.4f}, {ci[1]:.4f}]",
            "",
            f"Test of H0: d-prime = 0 ({self.stat_type.value} test)",
            f"  Statistic = {self.statistic:.4f}, p-value = {self.p_value:.4g}",
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        """Return a brief string representation."""
        return (
            f"DiscrimResult(method={self.method.value!r}, "
            f"d_prime={self.d_prime:.4f}, p_value={self.p_value:.4g})"
        )

    def __repr__(self) -> str:
        """Return a detailed representation."""
        return self.__str__()


@dataclass
class RescaleResult:
    """Result from rescaling between pc, pd, and d-prime.

    Attributes
    ----------
    pc : float | NDArray
        Proportion correct.
    pd : float | NDArray
        Proportion discriminators.
    d_prime : float | NDArray
        Sensitivity measure (d-prime).
    se_pc : float | NDArray | None
        Standard error of pc (if provided).
    se_pd : float | NDArray | None
        Standard error of pd (if provided).
    se_d_prime : float | NDArray | None
        Standard error of d-prime (if provided).
    method : Protocol
        The protocol used for conversion.
    """

    pc: float | NDArray[np.floating]
    pd: float | NDArray[np.floating]
    d_prime: float | NDArray[np.floating]
    method: Protocol
    se_pc: float | NDArray[np.floating] | None = None
    se_pd: float | NDArray[np.floating] | None = None
    se_d_prime: float | NDArray[np.floating] | None = None

    @property
    def coefficients(self) -> dict[str, float | NDArray[np.floating]]:
        """Return coefficients as a dictionary."""
        return {
            "pc": self.pc,
            "pd": self.pd,
            "d_prime": self.d_prime,
        }

    @property
    def std_err(self) -> dict[str, float | NDArray[np.floating] | None]:
        """Return standard errors as a dictionary."""
        return {
            "pc": self.se_pc,
            "pd": self.se_pd,
            "d_prime": self.se_d_prime,
        }

    def __str__(self) -> str:
        """Return a brief string representation."""
        return (
            f"RescaleResult(method={self.method.value!r}, "
            f"pc={self.pc}, pd={self.pd}, d_prime={self.d_prime})"
        )
