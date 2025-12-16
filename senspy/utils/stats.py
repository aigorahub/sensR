"""Statistical utility functions for sensPy.

This module provides basic statistical utilities used throughout the package,
including functions for computing p-values and finding critical values.

These functions correspond to utilities in sensR's utils.R file.
"""

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats


def delimit(
    x: ArrayLike,
    lower: float | None = None,
    upper: float | None = None,
) -> NDArray[np.floating]:
    """Constrain values to be within specified bounds.

    Sets values below `lower` to `lower` and values above `upper` to `upper`.

    Parameters
    ----------
    x : array_like
        Input values to constrain.
    lower : float, optional
        Lower bound. Values below this are set to `lower`.
    upper : float, optional
        Upper bound. Values above this are set to `upper`.

    Returns
    -------
    ndarray
        Array with values constrained to [lower, upper].

    Raises
    ------
    ValueError
        If both bounds are specified and lower >= upper.

    Notes
    -----
    Corresponds to `delimit()` in sensR's utils.R.

    Examples
    --------
    >>> delimit([0.1, 0.5, 0.9], lower=0.2, upper=0.8)
    array([0.2, 0.5, 0.8])
    >>> delimit(-0.5, lower=0)
    array([0.])
    """
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))

    if lower is not None and upper is not None:
        if lower >= upper:
            raise ValueError(f"lower ({lower}) must be less than upper ({upper})")

    result = x.copy()

    if lower is not None:
        result = np.maximum(result, lower)
    if upper is not None:
        result = np.minimum(result, upper)

    return result


def normal_pvalue(
    statistic: ArrayLike,
    alternative: Literal["two.sided", "greater", "less"] = "two.sided",
) -> NDArray[np.floating]:
    """Compute p-value for a standard normal test statistic.

    Parameters
    ----------
    statistic : array_like
        Test statistic(s) following a standard normal distribution under H0.
    alternative : {"two.sided", "greater", "less"}, default "two.sided"
        Type of alternative hypothesis:
        - "two.sided": H1: parameter != 0
        - "greater": H1: parameter > 0
        - "less": H1: parameter < 0

    Returns
    -------
    ndarray
        P-value(s) corresponding to the test statistic(s).

    Notes
    -----
    Corresponds to `normalPvalue()` in sensR's utils.R.

    Examples
    --------
    >>> normal_pvalue(1.96, alternative="two.sided")
    array([0.05...])
    >>> normal_pvalue(1.645, alternative="greater")
    array([0.05...])
    """
    statistic = np.atleast_1d(np.asarray(statistic, dtype=np.float64))

    if alternative == "greater":
        p_value = stats.norm.sf(statistic)  # 1 - CDF (upper tail)
    elif alternative == "less":
        p_value = stats.norm.cdf(statistic)  # lower tail
    elif alternative == "two.sided":
        p_value = 2 * stats.norm.sf(np.abs(statistic))
    else:
        raise ValueError(
            f"Unknown alternative: {alternative!r}. "
            "Must be 'two.sided', 'greater', or 'less'."
        )

    return p_value


def find_critical(
    sample_size: int,
    alpha: float = 0.05,
    p0: float = 0.5,
    pd0: float = 0.0,
    test: Literal["difference", "similarity"] = "difference",
) -> int:
    """Find the critical value for a one-tailed binomial test.

    Parameters
    ----------
    sample_size : int
        Number of trials.
    alpha : float, default 0.05
        Significance level.
    p0 : float, default 0.5
        Guessing probability (chance level).
    pd0 : float, default 0.0
        Proportion of discriminators under H0.
    test : {"difference", "similarity"}, default "difference"
        Type of test:
        - "difference": H1 is that the true proportion is greater than H0
        - "similarity": H1 is that the true proportion is less than H0

    Returns
    -------
    int
        Critical value. For a "difference" test, reject H0 if
        observed >= critical. For a "similarity" test, reject H0 if
        observed <= critical.

    Notes
    -----
    Corresponds to `findcr()` in sensR's utils.R.

    Examples
    --------
    >>> find_critical(sample_size=100, alpha=0.05, p0=0.5)
    59
    >>> find_critical(sample_size=100, alpha=0.05, p0=1/3)
    43
    """
    if sample_size != int(sample_size) or sample_size <= 0:
        raise ValueError("sample_size must be a positive integer")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1 (exclusive)")
    if not 0 < p0 < 1:
        raise ValueError("p0 must be between 0 and 1 (exclusive)")
    if not 0 <= pd0 <= 1:
        raise ValueError("pd0 must be between 0 and 1 (inclusive)")

    sample_size = int(sample_size)

    # Compute pc (proportion correct) under H0
    pc = pd0 + p0 * (1 - pd0)

    if test == "difference":
        # Find smallest x such that P(X >= x) <= alpha
        # This is equivalent to finding x where P(X <= x-1) >= 1-alpha
        for x in range(sample_size + 2):
            p_upper = 1 - stats.binom.cdf(x - 1, sample_size, pc)
            if p_upper <= alpha:
                return x
        return sample_size + 1  # No rejection possible

    elif test == "similarity":
        # Find largest x such that P(X <= x) <= alpha
        for x in range(sample_size, -2, -1):
            p_lower = stats.binom.cdf(x, sample_size, pc)
            if p_lower <= alpha:
                return x
        return -1  # No rejection possible

    else:
        raise ValueError(
            f"Unknown test: {test!r}. Must be 'difference' or 'similarity'."
        )


def test_critical(
    xcr: int,
    sample_size: int,
    p_correct: float = 0.5,
    alpha: float = 0.05,
    test: Literal["difference", "similarity"] = "difference",
) -> bool:
    """Test if xcr is the critical value for a binomial test.

    Parameters
    ----------
    xcr : int
        Candidate critical value.
    sample_size : int
        Number of trials.
    p_correct : float, default 0.5
        Probability of correct response under H0.
    alpha : float, default 0.05
        Significance level.
    test : {"difference", "similarity"}, default "difference"
        Type of test.

    Returns
    -------
    bool
        True if xcr is the critical value.

    Notes
    -----
    Corresponds to `test.crit()` in sensR's utils.R.
    """
    if test in ("difference", "greater"):
        # xcr is critical if P(X >= xcr) <= alpha and P(X >= xcr-1) > alpha
        p_at_xcr = 1 - stats.binom.cdf(xcr - 1, sample_size, p_correct)
        p_at_xcr_minus_1 = 1 - stats.binom.cdf(xcr - 2, sample_size, p_correct)
        return p_at_xcr <= alpha < p_at_xcr_minus_1

    elif test in ("similarity", "less"):
        # xcr is critical if P(X <= xcr) <= alpha and P(X <= xcr+1) > alpha
        p_at_xcr = stats.binom.cdf(xcr, sample_size, p_correct)
        p_at_xcr_plus_1 = stats.binom.cdf(xcr + 1, sample_size, p_correct)
        return p_at_xcr <= alpha < p_at_xcr_plus_1

    else:
        raise ValueError(f"Unknown test: {test!r}")
