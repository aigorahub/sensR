"""Protocol-specific power analysis functions.

This module provides power analysis for specific discrimination protocols
that require specialized methods beyond the general discrim_power function.

For the general discrimination power function, see senspy.power.discrim_power.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from senspy.simulation import samediff_sim
from senspy.samediff import samediff


def samediff_power(
    n: int = 1000,
    tau: float = 1.0,
    delta: float = 1.0,
    Ns: int = 10,
    Nd: int = 10,
    alpha: float = 0.05,
    random_state: int | np.random.Generator | None = None,
) -> float:
    """Compute power for same-different test via simulation.

    Computes the power for a same-different discrimination experiment
    with a no-difference null hypothesis via Monte Carlo simulation.

    Parameters
    ----------
    n : int, default=1000
        Number of simulations to run. More simulations give higher precision
        but take longer to compute.
    tau : float, default=1.0
        Decision criterion (tau parameter) under the alternative hypothesis.
    delta : float, default=1.0
        Sensory difference (d-prime) under the alternative hypothesis.
        Must be non-negative.
    Ns : int, default=10
        Number of same-sample pairs per experiment.
    Nd : int, default=10
        Number of different-sample pairs per experiment.
    alpha : float, default=0.05
        Type I error rate (significance level).
    random_state : int, Generator, or None, default=None
        Random number generator seed or instance for reproducibility.

    Returns
    -------
    float
        Estimated power of the test.

    Notes
    -----
    The power is computed via simulation: n datasets are simulated from the
    same-different model with the specified parameters, and for each dataset
    a significance test is performed. The power is the fraction of times
    the p-value is less than alpha.

    Under some parameter combinations, the MLE of delta may not be defined
    and the p-value cannot be computed. Such undefined p-values are ignored
    in the power calculation.

    The estimated power may vary between runs, especially when power is close
    to 0 or 1. Using more simulations (larger n) provides higher precision.

    Examples
    --------
    >>> from senspy import samediff_power
    >>> # Power for detecting delta=2 with tau=1 and sample size 2x10
    >>> samediff_power(n=100, tau=1, delta=2, Ns=10, Nd=10, random_state=42)
    0.82

    References
    ----------
    Christensen, R.H.B., Brockhoff, P.B. (2009). Estimation and inference in
        the same-different test. Food, Quality and Preference, 20, pp. 514-520.
    """
    if n <= 0:
        raise ValueError("'n' must be a positive integer")
    if delta < 0:
        raise ValueError("'delta' must be non-negative")
    if Ns <= 0 or Nd <= 0:
        raise ValueError("'Ns' and 'Nd' must be positive integers")
    if not 0 < alpha < 1:
        raise ValueError("'alpha' must be between 0 and 1")

    # Simulate n datasets
    sim_data = samediff_sim(n, tau, delta, Ns, Nd, random_state=random_state)
    data_array = sim_data.to_array()

    # Compute p-values for each simulated dataset
    p_values = np.full(n, np.nan)

    for i in range(n):
        try:
            result = samediff(
                nsamesame=int(data_array[i, 0]),
                ndiffsame=int(data_array[i, 1]),
                nsamediff=int(data_array[i, 2]),
                ndiffdiff=int(data_array[i, 3]),
            )

            # Check if delta estimate is valid
            delta_hat = result.delta
            if np.isnan(delta_hat):
                continue  # Skip if delta is not estimable

            if delta_hat == 0:
                p_values[i] = 1.0
                continue

            # Compute p-value using profile likelihood
            # Under H0: delta = 0, test H1: delta > 0
            # Use the signed likelihood root statistic
            ll_max = result.log_likelihood

            # Profile likelihood at delta=0
            # When delta=0, optimize over tau only
            from scipy import optimize

            def nll_delta_zero(t):
                if t <= 0:
                    return np.inf
                sqrt2 = np.sqrt(2)
                p_ss = 2 * stats.norm.cdf(t / sqrt2) - 1
                p_sd = 2 * stats.norm.cdf(t / sqrt2) - 1  # When delta=0, same formula
                ss, ds = data_array[i, 0], data_array[i, 1]
                sd, dd = data_array[i, 2], data_array[i, 3]
                probs = np.array([p_ss, 1 - p_ss, p_sd, 1 - p_sd])
                probs = np.maximum(probs, 1e-300)
                counts = np.array([ss, ds, sd, dd])
                return -np.sum(counts * np.log(probs))

            try:
                res = optimize.minimize_scalar(nll_delta_zero, bounds=(1e-6, 20), method="bounded")
                ll_null = -res.fun
            except Exception:
                continue

            # Likelihood ratio statistic
            lr_stat = 2 * (ll_max - ll_null)
            if lr_stat < 0:
                lr_stat = 0  # Numerical precision

            # One-sided p-value (testing delta > 0)
            lroot = np.sqrt(lr_stat) if delta_hat > 0 else -np.sqrt(lr_stat)
            p_values[i] = stats.norm.sf(lroot)

        except Exception:
            # Skip datasets where estimation fails
            continue

    # Power = fraction of p-values < alpha (ignoring NaN)
    valid_pvals = p_values[~np.isnan(p_values)]
    if len(valid_pvals) == 0:
        return 0.0

    power = np.mean(valid_pvals < alpha)
    return float(power)


@dataclass
class TwoACPowerResult:
    """Result from exact 2-AC power computation.

    Attributes
    ----------
    power : float
        Computed power.
    actual_alpha : float
        Actual size of the test (may differ from nominal alpha due to discreteness).
    samples : int
        Total number of possible outcomes for this sample size.
    discarded : int
        Number of outcomes not evaluated (probability mass < tol).
    kept : int
        Number of outcomes evaluated.
    p : NDArray[np.float64]
        Probability vector of the multinomial distribution.
    """

    power: float
    actual_alpha: float
    samples: int
    discarded: int
    kept: int
    p: NDArray[np.float64]


def twoac_power(
    tau: float,
    d_prime: float,
    size: int,
    d_prime_0: float = 0.0,
    alpha: float = 0.05,
    tol: float = 1e-5,
    alternative: Literal["two.sided", "less", "greater"] = "two.sided",
) -> TwoACPowerResult:
    """Compute exact power for the 2-AC discrimination protocol.

    Computes the exact power using the signed likelihood root statistic.
    Power is computed by enumerating all possible data outcomes and
    computing the p-value for each.

    Parameters
    ----------
    tau : float
        Threshold parameter under the alternative hypothesis. Must be > 0.
    d_prime : float
        D-prime value under the alternative hypothesis.
    size : int
        Sample size. Must be between 2 and 5000.
    d_prime_0 : float, default=0.0
        D-prime value under the null hypothesis.
    alpha : float, default=0.05
        Significance level. Must be between 0 and 1.
    tol : float, default=1e-5
        Precision for power computation. Lower values give higher precision
        but longer computation times. Use tol=0 for exact computation.
    alternative : str, default="two.sided"
        Type of alternative hypothesis: "two.sided", "less", or "greater".

    Returns
    -------
    TwoACPowerResult
        Result object with power, actual alpha, and diagnostic information.

    Notes
    -----
    The algorithm enumerates all possible data outcomes and computes the
    likelihood ratio p-value for each. Outcomes with very small probability
    (controlled by `tol`) are ignored to improve computation time.

    For large sample sizes, this can be slow since the number of outcomes
    grows as O(n^2). For sample sizes > 200, consider using simulation-based
    methods instead.

    Examples
    --------
    >>> from senspy import twoac_power
    >>> # Exact power computation
    >>> result = twoac_power(tau=0.5, d_prime=0.7, size=50, tol=0)
    >>> print(f"Power: {result.power:.4f}")

    >>> # Power with tolerance for faster computation
    >>> result = twoac_power(tau=0.5, d_prime=0.7, size=50, tol=1e-5)
    >>> print(f"Power: {result.power:.4f}")

    References
    ----------
    Christensen R.H.B., Lee H-S and Brockhoff P.B. (2012). Estimation of
        the Thurstonian model for the 2-AC protocol. Food Quality and
        Preference, 24(1), pp.119-128.
    """
    from senspy.twoac import _estimate_2ac, _nll_2ac
    from scipy import optimize

    # Validate inputs
    if tau <= 0:
        raise ValueError("'tau' must be positive")
    if not isinstance(size, int) or size < 2 or size > 5000:
        raise ValueError("'size' must be an integer between 2 and 5000")
    if not 0 < alpha < 1:
        raise ValueError("'alpha' must be between 0 and 1")
    if not 0 <= tol < 1:
        raise ValueError("'tol' must be between 0 and 1")
    if alternative not in ("two.sided", "less", "greater"):
        raise ValueError("'alternative' must be 'two.sided', 'less', or 'greater'")

    n = size

    # Generate all possible outcomes: (n_A, n_no_pref, n_B) where sum = n
    # For n observations, enumerate all (y1, y2, y3) with y1 + y2 + y3 = n
    outcomes = []
    for j in range(n + 1):
        for k in range(j + 1):
            outcomes.append([n - j, k, j - k])
    Y = np.array(outcomes)

    # Compute probability vector from tau and d_prime
    sqrt2 = np.sqrt(2)
    p1 = stats.norm.cdf(-tau, loc=d_prime, scale=sqrt2)
    p2 = stats.norm.cdf(tau, loc=d_prime, scale=sqrt2) - p1
    p3 = 1 - p1 - p2
    pvec = np.array([p1, p2, p3])

    # Compute multinomial probabilities for each outcome
    from scipy.special import gammaln

    def log_dmultinom(x, prob):
        """Log multinomial probability."""
        n = np.sum(x)
        log_coef = gammaln(n + 1) - np.sum(gammaln(x + 1))
        # Handle zero probabilities
        log_prob = np.zeros_like(x, dtype=float)
        mask = x > 0
        log_prob[mask] = x[mask] * np.log(prob[mask])
        return log_coef + np.sum(log_prob)

    dmult = np.array([np.exp(log_dmultinom(y, pvec)) for y in Y])

    # Discard outcomes with very low probability
    sdmult = np.sort(dmult)
    cumsum_sorted = np.cumsum(sdmult)
    no_discard = np.sum(cumsum_sorted < tol)

    if no_discard == 0:
        keep = np.ones(len(dmult), dtype=bool)
    else:
        threshold = sdmult[no_discard]
        keep = dmult > threshold

    dmult_kept = dmult[keep]
    Y_kept = Y[keep]

    # Compute p-values for kept outcomes
    def lr_pvalue(data, d_prime_0, alternative):
        """Compute likelihood ratio p-value for a single data point."""
        x = data.astype(float)

        # Get MLE
        est = _estimate_2ac(x, compute_vcov=False)
        d_prime_hat = est["coefficients"][1, 0]
        log_lik_max = est["log_likelihood"]

        # Compute log-likelihood under null
        def nll_tau(tau_val):
            return _nll_2ac(tau_val, d_prime_0, x)

        # Handle special cases
        if x[1] == 0:
            if x[0] > 0 and x[2] > 0:
                result = optimize.minimize_scalar(nll_tau, bounds=(1e-10, 10), method="bounded")
                nll_0 = result.fun
            else:
                nll_0 = nll_tau(1e-10)  # Approximate for boundary
        elif x[0] == 0 and x[2] == 0:
            nll_0 = 0
        else:
            result = optimize.minimize_scalar(nll_tau, bounds=(1e-10, 10), method="bounded")
            nll_0 = result.fun

        # Signed likelihood root
        lr_stat = 2 * (log_lik_max + nll_0)
        if lr_stat < 0:
            lr_stat = 0
        lroot = np.sign(d_prime_hat - d_prime_0) * np.sqrt(lr_stat)

        # P-value
        if alternative == "greater":
            pval = stats.norm.sf(lroot)
        elif alternative == "less":
            pval = stats.norm.cdf(lroot)
        else:  # two.sided
            pval = 2 * stats.norm.sf(np.abs(lroot))

        return pval

    pvals = np.array([lr_pvalue(y, d_prime_0, alternative) for y in Y_kept])

    # Compute power from p-value distribution
    # Sort by p-value to get cumulative distribution
    idx = np.argsort(pvals)
    pvals_sorted = pvals[idx]
    dmult_sorted = dmult_kept[idx]
    dist = np.cumsum(dmult_sorted)

    if np.any(pvals_sorted <= alpha):
        power = np.max(dist[pvals_sorted <= alpha])
        actual_alpha = np.max(pvals_sorted[pvals_sorted <= alpha])
    else:
        power = 0.0
        actual_alpha = 0.0

    return TwoACPowerResult(
        power=float(power),
        actual_alpha=float(actual_alpha),
        samples=len(Y),
        discarded=int(no_discard),
        kept=int(len(Y) - no_discard),
        p=np.round(pvec, 4),
    )
