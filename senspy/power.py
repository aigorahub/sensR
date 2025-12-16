"""Power analysis and sample size calculations for discrimination tests.

This module provides functions for computing statistical power and
required sample sizes for sensory discrimination experiments.

Corresponds to sensR's power.R, newPower.R, and sample.size.R.
"""

import numpy as np
from scipy import stats

from senspy.core.types import Protocol, parse_protocol
from senspy.links import psy_fun
from senspy.utils import pd_to_pc, find_critical, delimit


def _normal_power(
    pd_a: float,
    pd_0: float,
    sample_size: int,
    alpha: float,
    p_guess: float,
    test: str,
    continuity: bool = False,
) -> float:
    """Normal approximation to exact binomial power.

    Parameters
    ----------
    pd_a : float
        True proportion of discriminators (alternative hypothesis).
    pd_0 : float
        Null hypothesis proportion of discriminators.
    sample_size : int
        Number of trials.
    alpha : float
        Significance level.
    p_guess : float
        Guessing probability for the protocol.
    test : str
        "difference" or "similarity".
    continuity : bool
        Whether to apply continuity correction.

    Returns
    -------
    float
        Statistical power.
    """
    pc_0 = pd_to_pc(pd_0, p_guess)[0]
    pc_a = pd_to_pc(pd_a, p_guess)[0]

    sigma_0 = np.sqrt(pc_0 * (1 - pc_0) * sample_size)
    sigma_a = np.sqrt(pc_a * (1 - pc_a) * sample_size)

    if test == "difference":
        lam = (stats.norm.ppf(1 - alpha) * sigma_0 + sample_size * (pc_0 - pc_a)) / sigma_a
        if continuity:
            lam = lam + 0.5 / sigma_a
        power = stats.norm.sf(lam)
    else:  # similarity
        lam = (stats.norm.ppf(alpha) * sigma_0 + sample_size * (pc_0 - pc_a)) / sigma_a
        if continuity:
            lam = lam - 0.5 / sigma_a
        power = stats.norm.cdf(lam)

    return float(power)


def _exact_power(
    pd_a: float,
    pd_0: float,
    sample_size: int,
    alpha: float,
    p_guess: float,
    test: str,
) -> float:
    """Exact binomial power.

    Parameters
    ----------
    pd_a : float
        True proportion of discriminators (alternative hypothesis).
    pd_0 : float
        Null hypothesis proportion of discriminators.
    sample_size : int
        Number of trials.
    alpha : float
        Significance level.
    p_guess : float
        Guessing probability for the protocol.
    test : str
        "difference" or "similarity".

    Returns
    -------
    float
        Statistical power.
    """
    pc_a = pd_to_pc(pd_a, p_guess)[0]

    # Get critical value
    crit = find_critical(
        sample_size=sample_size,
        alpha=alpha,
        p0=p_guess,
        pd0=pd_0,
        test=test,
    )

    # Compute power from critical value
    if test == "difference":
        crit = delimit(crit, lower=1, upper=sample_size + 1)[0]
        power = 1 - stats.binom.cdf(crit - 1, sample_size, pc_a)
    else:  # similarity
        crit = delimit(crit, lower=0, upper=sample_size)[0]
        power = stats.binom.cdf(crit, sample_size, pc_a)

    return float(power)


def discrim_power(
    pd_a: float,
    sample_size: int,
    *,
    pd_0: float = 0.0,
    alpha: float = 0.05,
    p_guess: float = 0.5,
    test: str = "difference",
    statistic: str = "exact",
) -> float:
    """Compute power for a discrimination test.

    Parameters
    ----------
    pd_a : float
        True proportion of discriminators (alternative hypothesis).
        Must be between 0 and 1.
    sample_size : int
        Number of trials in the experiment.
    pd_0 : float, default 0.0
        Null hypothesis proportion of discriminators.
    alpha : float, default 0.05
        Significance level (Type I error rate).
    p_guess : float, default 0.5
        Guessing probability for the protocol (e.g., 1/3 for triangle).
    test : str, default "difference"
        Type of test: "difference" (H1: pd > pd_0) or
        "similarity" (H1: pd < pd_0).
    statistic : str, default "exact"
        Method for computing power:
        - "exact": Exact binomial test
        - "normal": Normal approximation
        - "cont.normal": Normal approximation with continuity correction

    Returns
    -------
    float
        Statistical power (probability of rejecting H0 when H1 is true).

    Examples
    --------
    >>> # Power for triangle test with 30% discriminators, n=100
    >>> discrim_power(pd_a=0.3, sample_size=100, p_guess=1/3)
    0.869...

    >>> # Power for similarity test
    >>> discrim_power(pd_a=0.1, sample_size=100, pd_0=0.3,
    ...               p_guess=1/3, test="similarity")
    0.78...

    Notes
    -----
    Corresponds to `discrimPwr()` in sensR.
    """
    # Validate inputs
    if not 0 <= pd_a <= 1:
        raise ValueError("'pd_a' must be between 0 and 1")
    if not 0 <= pd_0 <= 1:
        raise ValueError("'pd_0' must be between 0 and 1")
    if not isinstance(sample_size, (int, np.integer)) or sample_size <= 0:
        raise ValueError("'sample_size' must be a positive integer")
    if not 0 < alpha < 1:
        raise ValueError("'alpha' must be between 0 and 1")
    if not 0 <= p_guess < 1:
        raise ValueError("'p_guess' must be between 0 and 1")

    sample_size = int(sample_size)

    # Validate test type
    test = test.lower()
    if test not in ("difference", "similarity"):
        raise ValueError("'test' must be 'difference' or 'similarity'")

    # Validate pd_a vs pd_0 for test type
    if test == "difference" and pd_a < pd_0:
        raise ValueError("'pd_a' must be >= 'pd_0' for difference tests")
    if test == "similarity" and pd_a > pd_0:
        raise ValueError("'pd_a' must be <= 'pd_0' for similarity tests")

    # Validate statistic
    statistic = statistic.lower().replace(".", "").replace("_", "")
    if statistic not in ("exact", "normal", "contnormal"):
        raise ValueError(
            "'statistic' must be 'exact', 'normal', or 'cont.normal'"
        )

    # Compute power
    if statistic == "normal":
        return _normal_power(pd_a, pd_0, sample_size, alpha, p_guess, test, False)
    elif statistic == "contnormal":
        return _normal_power(pd_a, pd_0, sample_size, alpha, p_guess, test, True)
    else:  # exact
        return _exact_power(pd_a, pd_0, sample_size, alpha, p_guess, test)


def dprime_power(
    d_prime_a: float,
    sample_size: int,
    method: str | Protocol = "triangle",
    *,
    d_prime_0: float = 0.0,
    alpha: float = 0.05,
    test: str = "difference",
    statistic: str = "exact",
) -> float:
    """Compute power for a discrimination test using d-prime.

    This is a convenience wrapper around `discrim_power()` that converts
    d-prime values to proportion of discriminators.

    Parameters
    ----------
    d_prime_a : float
        True d-prime value (alternative hypothesis). Must be non-negative.
    sample_size : int
        Number of trials in the experiment.
    method : str or Protocol, default "triangle"
        Discrimination protocol.
    d_prime_0 : float, default 0.0
        Null hypothesis d-prime value.
    alpha : float, default 0.05
        Significance level (Type I error rate).
    test : str, default "difference"
        Type of test: "difference" or "similarity".
    statistic : str, default "exact"
        Method for computing power: "exact", "normal", or "cont.normal".

    Returns
    -------
    float
        Statistical power.

    Examples
    --------
    >>> # Power for triangle test with d'=1.5, n=100
    >>> dprime_power(d_prime_a=1.5, sample_size=100, method="triangle")
    0.86...

    >>> # Power for 2-AFC test
    >>> dprime_power(d_prime_a=1.0, sample_size=50, method="twoafc")
    0.74...

    Notes
    -----
    Corresponds to `d.primePwr()` in sensR.
    """
    # Validate d-prime values
    if d_prime_a < 0:
        raise ValueError("'d_prime_a' must be non-negative")
    if d_prime_0 < 0:
        raise ValueError("'d_prime_0' must be non-negative")

    # Parse protocol
    protocol = parse_protocol(method)
    p_guess = protocol.p_guess

    # Convert d-prime to pc, then to pd
    pc_a = psy_fun(d_prime_a, method=protocol)[0]
    pc_0 = psy_fun(d_prime_0, method=protocol)[0]

    pd_a = (pc_a - p_guess) / (1 - p_guess)
    pd_0 = (pc_0 - p_guess) / (1 - p_guess)

    return discrim_power(
        pd_a=pd_a,
        sample_size=sample_size,
        pd_0=pd_0,
        alpha=alpha,
        p_guess=p_guess,
        test=test,
        statistic=statistic,
    )


def _normal_sample_size(
    pd_a: float,
    pd_0: float,
    target_power: float,
    alpha: float,
    p_guess: float,
    test: str,
    continuity: bool = False,
) -> int:
    """Normal approximation to required sample size.

    Parameters
    ----------
    pd_a : float
        True proportion of discriminators.
    pd_0 : float
        Null hypothesis proportion.
    target_power : float
        Desired power level.
    alpha : float
        Significance level.
    p_guess : float
        Guessing probability.
    test : str
        "difference" or "similarity".
    continuity : bool
        Whether to apply continuity correction.

    Returns
    -------
    int
        Required sample size.
    """
    pc_0 = pd_to_pc(pd_0, p_guess)[0]
    pc_a = pd_to_pc(pd_a, p_guess)[0]

    # For similarity tests, swap and transform
    if test == "similarity":
        pc_a, pc_0 = pc_0, pc_a
        alpha, target_power = 1 - target_power, 1 - alpha

    # Solve quadratic for n
    # Power equation: Phi((z_alpha * sigma_0 + n * (pc_0 - pc_a)) / sigma_a) = 1 - beta
    a = pc_0 - pc_a
    b = (
        stats.norm.ppf(1 - alpha) * np.sqrt(pc_0 * (1 - pc_0))
        - stats.norm.ppf(1 - target_power) * np.sqrt(pc_a * (1 - pc_a))
    )
    c = 0.5 if continuity else 0.0

    # Solve ax^2 + bx + c = 0 for x = sqrt(n)
    discriminant = max(0, b**2 - 4 * a * c)
    x = (-b - np.sqrt(discriminant)) / (2 * a)
    n = max(1, round(x**2))

    return int(n)


def _exact_sample_size(
    pd_a: float,
    pd_0: float,
    target_power: float,
    alpha: float,
    p_guess: float,
    test: str,
    stable: bool = False,
) -> int:
    """Find exact sample size by searching.

    Parameters
    ----------
    pd_a : float
        True proportion of discriminators.
    pd_0 : float
        Null hypothesis proportion.
    target_power : float
        Desired power level.
    alpha : float
        Significance level.
    p_guess : float
        Guessing probability.
    test : str
        "difference" or "similarity".
    stable : bool
        If True, return stable sample size (power stays above target).
        If False, return first sample size that achieves target.

    Returns
    -------
    int
        Required sample size.
    """
    # Get bounds from normal approximation
    n_lower = _normal_sample_size(
        pd_a, pd_0, target_power, alpha, p_guess, test, continuity=True
    )
    # Adjust lower bound down
    n_lower = max(1, n_lower - 10)

    n_upper = _normal_sample_size(
        pd_a, pd_0, target_power, alpha, p_guess, test, continuity=True
    )
    # Adjust upper bound up
    n_upper = n_upper + 20

    # Check bounds
    lower_power = _exact_power(pd_a, pd_0, n_lower, alpha, p_guess, test)
    upper_power = _exact_power(pd_a, pd_0, n_upper, alpha, p_guess, test)

    # Expand bounds if needed
    while lower_power >= target_power and n_lower > 1:
        n_lower = max(1, n_lower - 10)
        lower_power = _exact_power(pd_a, pd_0, n_lower, alpha, p_guess, test)

    while upper_power <= target_power:
        n_upper = n_upper + 20
        upper_power = _exact_power(pd_a, pd_0, n_upper, alpha, p_guess, test)
        if n_upper > 100000:
            raise ValueError(
                "Sample size > 100000; use 'normal' or 'cont.normal' statistic"
            )

    if not stable:
        # Find first n where power > target
        for n in range(n_lower, n_upper + 1):
            power = _exact_power(pd_a, pd_0, n, alpha, p_guess, test)
            if power > target_power:
                return n
    else:
        # Find stable n: step down from upper until power < target
        for n in range(n_upper, n_lower - 1, -1):
            power = _exact_power(pd_a, pd_0, n, alpha, p_guess, test)
            if power < target_power:
                return n + 1

    return n_upper


def discrim_sample_size(
    pd_a: float,
    *,
    pd_0: float = 0.0,
    target_power: float = 0.90,
    alpha: float = 0.05,
    p_guess: float = 0.5,
    test: str = "difference",
    statistic: str = "exact",
) -> int:
    """Compute required sample size for a discrimination test.

    Parameters
    ----------
    pd_a : float
        True proportion of discriminators (alternative hypothesis).
    pd_0 : float, default 0.0
        Null hypothesis proportion of discriminators.
    target_power : float, default 0.90
        Desired statistical power.
    alpha : float, default 0.05
        Significance level.
    p_guess : float, default 0.5
        Guessing probability for the protocol.
    test : str, default "difference"
        Type of test: "difference" or "similarity".
    statistic : str, default "exact"
        Method for computing sample size:
        - "exact": First n where exact power exceeds target
        - "stable.exact": Stable n (power stays above target)
        - "normal": Normal approximation
        - "cont.normal": Normal approximation with continuity correction

    Returns
    -------
    int
        Required sample size.

    Examples
    --------
    >>> # Sample size for triangle test with 30% discriminators
    >>> discrim_sample_size(pd_a=0.3, p_guess=1/3)
    85

    >>> # Using normal approximation
    >>> discrim_sample_size(pd_a=0.3, p_guess=1/3, statistic="normal")
    83

    Notes
    -----
    Corresponds to `discrimSS()` in sensR.

    The "exact" method finds the first sample size where power exceeds
    the target. Due to the discrete nature of the binomial distribution,
    power may drop below target for slightly larger n. Use "stable.exact"
    if you need power to remain above target for all larger sample sizes.
    """
    # Validate inputs
    if not 0 < pd_a <= 1:
        raise ValueError("'pd_a' must be between 0 and 1 (exclusive of 0)")
    if not 0 <= pd_0 < 1:
        raise ValueError("'pd_0' must be between 0 and 1")
    if not 0 < target_power < 1:
        raise ValueError("'target_power' must be between 0 and 1")
    if not 0 < alpha < 1:
        raise ValueError("'alpha' must be between 0 and 1")
    if not 0 <= p_guess < 1:
        raise ValueError("'p_guess' must be between 0 and 1")

    # Validate test type
    test = test.lower()
    if test not in ("difference", "similarity"):
        raise ValueError("'test' must be 'difference' or 'similarity'")

    # Validate pd_a vs pd_0 for test type
    if test == "difference" and pd_a <= pd_0:
        raise ValueError("'pd_a' must be > 'pd_0' for difference tests")
    if test == "similarity" and pd_a >= pd_0:
        raise ValueError("'pd_a' must be < 'pd_0' for similarity tests")

    # Validate statistic
    statistic = statistic.lower().replace(".", "_").replace("-", "_")
    if statistic not in ("exact", "stable_exact", "normal", "cont_normal"):
        raise ValueError(
            "'statistic' must be 'exact', 'stable.exact', 'normal', or 'cont.normal'"
        )

    # Compute sample size
    if statistic == "normal":
        return _normal_sample_size(
            pd_a, pd_0, target_power, alpha, p_guess, test, continuity=False
        )
    elif statistic == "cont_normal":
        return _normal_sample_size(
            pd_a, pd_0, target_power, alpha, p_guess, test, continuity=True
        )
    elif statistic == "exact":
        return _exact_sample_size(
            pd_a, pd_0, target_power, alpha, p_guess, test, stable=False
        )
    else:  # stable_exact
        return _exact_sample_size(
            pd_a, pd_0, target_power, alpha, p_guess, test, stable=True
        )


def dprime_sample_size(
    d_prime_a: float,
    method: str | Protocol = "triangle",
    *,
    d_prime_0: float = 0.0,
    target_power: float = 0.90,
    alpha: float = 0.05,
    test: str = "difference",
    statistic: str = "exact",
) -> int:
    """Compute required sample size using d-prime.

    This is a convenience wrapper around `discrim_sample_size()` that
    converts d-prime values to proportion of discriminators.

    Parameters
    ----------
    d_prime_a : float
        True d-prime value (alternative hypothesis).
    method : str or Protocol, default "triangle"
        Discrimination protocol.
    d_prime_0 : float, default 0.0
        Null hypothesis d-prime value.
    target_power : float, default 0.90
        Desired statistical power.
    alpha : float, default 0.05
        Significance level.
    test : str, default "difference"
        Type of test: "difference" or "similarity".
    statistic : str, default "exact"
        Method: "exact", "stable.exact", "normal", or "cont.normal".

    Returns
    -------
    int
        Required sample size.

    Examples
    --------
    >>> # Sample size for triangle test with d'=1.5
    >>> dprime_sample_size(d_prime_a=1.5, method="triangle")
    62

    >>> # Sample size for 2-AFC test with 80% power
    >>> dprime_sample_size(d_prime_a=1.0, method="twoafc", target_power=0.80)
    47

    Notes
    -----
    Corresponds to `d.primeSS()` in sensR.
    """
    # Validate d-prime values
    if d_prime_a <= 0:
        raise ValueError("'d_prime_a' must be positive")
    if d_prime_0 < 0:
        raise ValueError("'d_prime_0' must be non-negative")

    # Parse protocol
    protocol = parse_protocol(method)
    p_guess = protocol.p_guess

    # Convert d-prime to pc, then to pd
    pc_a = psy_fun(d_prime_a, method=protocol)[0]
    pc_0 = psy_fun(d_prime_0, method=protocol)[0]

    pd_a = (pc_a - p_guess) / (1 - p_guess)
    pd_0 = (pc_0 - p_guess) / (1 - p_guess)

    return discrim_sample_size(
        pd_a=pd_a,
        pd_0=pd_0,
        target_power=target_power,
        alpha=alpha,
        p_guess=p_guess,
        test=test,
        statistic=statistic,
    )
