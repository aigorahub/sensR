"""Discrimination analysis for sensory testing.

This module implements the core `discrim()` function for analyzing
discrimination test data. It estimates d-prime (sensitivity) from
the number of correct responses in forced-choice trials.

Corresponds to sensR's discrim() function.
"""

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from senspy.core.types import Protocol, Statistic, Alternative, parse_protocol
from senspy.core.base import DiscrimResult
from senspy.links import psy_fun, psy_inv, psy_deriv
from senspy.utils import pc_to_pd, pd_to_pc, delimit


def _profile_binom(
    x: int, n: int, n_points: int = 100
) -> tuple[NDArray, NDArray, float, float]:
    """Compute binomial profile likelihood.

    Parameters
    ----------
    x : int
        Number of successes.
    n : int
        Number of trials.
    n_points : int
        Number of points for the profile.

    Returns
    -------
    p_seq : ndarray
        Sequence of probability values.
    l_root : ndarray
        Signed likelihood root statistics.
    log_lik : float
        Log-likelihood at MLE.
    p_hat : float
        MLE of probability.
    """
    p_hat = x / n
    log_lik = stats.binom.logpmf(x, n, p_hat) if p_hat > 0 and p_hat < 1 else 0.0

    p_seq = np.linspace(1e-8, 1 - 1e-8, n_points)
    if p_hat not in p_seq and 0 < p_hat < 1:
        p_seq = np.sort(np.append(p_seq, p_hat))

    ll = stats.binom.logpmf(x, n, p_seq)
    sign = 2 * (p_seq > p_hat).astype(float) - 1
    l_root = sign * np.sqrt(2) * np.sqrt(np.maximum(-ll + log_lik, 0))

    return p_seq, l_root, log_lik, p_hat


def _confint_profile(
    p_seq: NDArray, l_root: NDArray, level: float = 0.95
) -> tuple[float, float]:
    """Compute confidence interval from profile likelihood.

    Parameters
    ----------
    p_seq : ndarray
        Sequence of probability values.
    l_root : ndarray
        Signed likelihood root statistics.
    level : float
        Confidence level.

    Returns
    -------
    lower, upper : float
        Confidence interval bounds.
    """
    from scipy.interpolate import interp1d

    alpha = 1 - level
    cutoff = stats.norm.ppf([alpha / 2, 1 - alpha / 2])

    # Interpolate on logit scale for numerical stability
    p_logit = np.log(p_seq / (1 - p_seq))

    try:
        interp = interp1d(l_root, p_logit, kind="linear", fill_value="extrapolate")
        ci_logit = interp(cutoff)
        ci = 1 / (1 + np.exp(-ci_logit))
        return float(ci[0]), float(ci[1])
    except Exception:
        return 0.0, 1.0


def discrim(
    correct: int,
    total: int,
    method: str | Protocol = "triangle",
    *,
    d_prime0: float | None = None,
    pd0: float | None = None,
    conf_level: float = 0.95,
    statistic: str | Statistic = "exact",
    test: str = "difference",
) -> DiscrimResult:
    """Analyze discrimination test data.

    Estimates d-prime (sensitivity) from the number of correct responses
    in a forced-choice discrimination test.

    Parameters
    ----------
    correct : int
        Number of correct responses.
    total : int
        Total number of trials.
    method : str or Protocol, default "triangle"
        Discrimination protocol: "duotrio", "triangle", "twoafc", "threeafc",
        "tetrad", "hexad", "twofive", "twofivef".
    d_prime0 : float, optional
        Null hypothesis value for d-prime (for similarity tests).
    pd0 : float, optional
        Null hypothesis value for proportion discriminators.
        Only one of d_prime0 or pd0 can be specified.
    conf_level : float, default 0.95
        Confidence level for intervals.
    statistic : str or Statistic, default "exact"
        Test statistic: "exact", "likelihood", "wald", "score".
    test : str, default "difference"
        Type of test: "difference" (H1: d' > d'0) or "similarity" (H1: d' < d'0).

    Returns
    -------
    DiscrimResult
        Result object with estimates, standard errors, confidence intervals,
        and test results.

    Examples
    --------
    >>> result = discrim(correct=80, total=100, method="triangle")
    >>> result.d_prime
    2.164...
    >>> result.confint()
    array([1.65..., 2.75...])

    >>> # Similarity test
    >>> result = discrim(correct=40, total=100, method="triangle",
    ...                  d_prime0=1.0, test="similarity")
    >>> result.p_value
    0.02...

    Notes
    -----
    Corresponds to `discrim()` in sensR.

    The function supports four test statistics:

    - **exact**: Exact binomial test (default). P-value from binomial
      distribution, CI from Clopper-Pearson method.
    - **likelihood**: Likelihood ratio test. P-value from signed likelihood
      root statistic, CI from profile likelihood.
    - **wald**: Wald test. P-value and CI based on normal approximation.
    - **score**: Score test. P-value from Pearson chi-square, CI from
      Wilson score interval.
    """
    # Validate inputs
    if not isinstance(correct, (int, np.integer)):
        if correct != int(correct):
            raise ValueError("'correct' must be a non-negative integer")
        correct = int(correct)
    if correct < 0:
        raise ValueError("'correct' must be a non-negative integer")

    if not isinstance(total, (int, np.integer)):
        if total != int(total):
            raise ValueError("'total' must be a positive integer")
        total = int(total)
    if total <= 0:
        raise ValueError("'total' must be a positive integer")

    if correct > total:
        raise ValueError("'correct' cannot be larger than 'total'")

    if not 0 < conf_level < 1:
        raise ValueError("'conf_level' must be between 0 and 1")

    # Parse method
    protocol = parse_protocol(method)
    p_guess = protocol.p_guess

    # Parse statistic
    if isinstance(statistic, str):
        stat_lower = statistic.lower()
        stat_type = {
            "exact": Statistic.EXACT,
            "likelihood": Statistic.LIKELIHOOD,
            "wald": Statistic.WALD,
            "score": Statistic.SCORE,
        }.get(stat_lower)
        if stat_type is None:
            raise ValueError(
                f"Unknown statistic: {statistic!r}. "
                "Valid options: 'exact', 'likelihood', 'wald', 'score'"
            )
    else:
        stat_type = statistic

    # Parse test type
    test_lower = test.lower()
    if test_lower not in ("difference", "similarity"):
        raise ValueError(
            f"Unknown test: {test!r}. Valid options: 'difference', 'similarity'"
        )

    # Handle null hypothesis specification
    if d_prime0 is not None and pd0 is not None:
        raise ValueError("Only specify one of 'd_prime0' and 'pd0'")

    if test_lower == "similarity" and d_prime0 is None and pd0 is None:
        raise ValueError(
            "Either 'd_prime0' or 'pd0' must be specified for a similarity test"
        )

    # Set null hypothesis values
    if pd0 is not None:
        if not 0 <= pd0 <= 1:
            raise ValueError("'pd0' must be between 0 and 1")
        pc0 = pd_to_pc(pd0, p_guess)[0]
    elif d_prime0 is not None:
        if d_prime0 < 0:
            raise ValueError("'d_prime0' must be non-negative")
        pc0 = psy_fun(d_prime0, method=protocol)[0]
    else:
        pd0 = 0.0
        pc0 = p_guess

    # Compute estimates
    pc_hat = correct / total
    se_pc = np.sqrt(pc_hat * (1 - pc_hat) / total) if 0 < pc_hat < 1 else 0.0

    # Convert to pd and d-prime
    if pc_hat <= p_guess:
        pd_hat = 0.0
        d_prime_hat = 0.0
        se_pd = np.nan
        se_d_prime = np.nan
    else:
        pd_hat = pc_to_pd(pc_hat, p_guess)[0]
        d_prime_hat = psy_inv(pc_hat, method=protocol)[0]

        # Standard errors via delta method
        if pc_hat < 1:
            # SE for pd: pd = (pc - p_guess) / (1 - p_guess)
            se_pd = se_pc / (1 - p_guess)

            # SE for d-prime: use derivative of inverse link
            deriv = psy_deriv(d_prime_hat, method=protocol)[0]
            if deriv > 0:
                se_d_prime = se_pc / deriv
            else:
                se_d_prime = np.nan
        else:
            se_pd = np.nan
            se_d_prime = np.nan

    # Compute p-value and confidence interval based on statistic
    if stat_type == Statistic.EXACT:
        # Exact binomial test
        if test_lower == "difference":
            p_value = 1 - stats.binom.cdf(correct - 1, total, pc0)
        else:
            p_value = stats.binom.cdf(correct, total, pc0)

        # Clopper-Pearson CI
        ci_result = stats.binomtest(correct, total)
        ci_pc = ci_result.proportion_ci(confidence_level=conf_level)
        ci_lower_pc, ci_upper_pc = ci_pc.low, ci_pc.high
        stat_value = float(correct)

    elif stat_type == Statistic.LIKELIHOOD:
        # Likelihood ratio test
        p_seq, l_root, log_lik_max, _ = _profile_binom(correct, total)

        log_lik_null = stats.binom.logpmf(correct, total, pc0)
        stat_value = np.sign(pc_hat - pc0) * np.sqrt(
            2 * max(log_lik_max - log_lik_null, 0)
        )

        if test_lower == "difference":
            p_value = stats.norm.sf(stat_value)
        else:
            p_value = stats.norm.cdf(stat_value)

        # Profile likelihood CI
        ci_lower_pc, ci_upper_pc = _confint_profile(p_seq, l_root, conf_level)

    elif stat_type == Statistic.WALD:
        # Wald test
        if se_pc > 0:
            stat_value = (pc_hat - pc0) / np.sqrt(pc_hat * (1 - pc_hat) / total)
        else:
            stat_value = 0.0

        if test_lower == "difference":
            p_value = stats.norm.sf(stat_value)
        else:
            p_value = stats.norm.cdf(stat_value)

        # Wald CI
        alpha = 1 - conf_level
        z = stats.norm.ppf(1 - alpha / 2)
        ci_lower_pc = pc_hat - z * se_pc
        ci_upper_pc = pc_hat + z * se_pc

    elif stat_type == Statistic.SCORE:
        # Score test (Wilson)
        pc0_bounded = delimit(pc0, lower=1e-8, upper=1 - 1e-8)[0]

        if test_lower == "difference":
            # One-sided greater
            stat_value = (pc_hat - pc0_bounded) / np.sqrt(
                pc0_bounded * (1 - pc0_bounded) / total
            )
            p_value = stats.norm.sf(stat_value)
        else:
            stat_value = (pc_hat - pc0_bounded) / np.sqrt(
                pc0_bounded * (1 - pc0_bounded) / total
            )
            p_value = stats.norm.cdf(stat_value)

        # Wilson score CI
        alpha = 1 - conf_level
        z = stats.norm.ppf(1 - alpha / 2)
        denom = 1 + z**2 / total
        center = (pc_hat + z**2 / (2 * total)) / denom
        margin = z * np.sqrt(pc_hat * (1 - pc_hat) / total + z**2 / (4 * total**2)) / denom
        ci_lower_pc = center - margin
        ci_upper_pc = center + margin

    # Bound CI to valid range
    ci_lower_pc = max(0, ci_lower_pc)
    ci_upper_pc = min(1, ci_upper_pc)

    # Convert CI to d-prime scale
    if ci_lower_pc <= p_guess:
        ci_lower_d = 0.0
    else:
        ci_lower_d = psy_inv(ci_lower_pc, method=protocol)[0]

    if ci_upper_pc <= p_guess:
        ci_upper_d = 0.0
    elif ci_upper_pc >= 1:
        ci_upper_d = np.inf
    else:
        ci_upper_d = psy_inv(ci_upper_pc, method=protocol)[0]

    # Build result
    result = DiscrimResult(
        d_prime=d_prime_hat,
        pc=pc_hat,
        pd=pd_hat,
        se_d_prime=se_d_prime if not np.isnan(se_d_prime) else 0.0,
        se_pc=se_pc,
        se_pd=se_pd if not np.isnan(se_pd) else 0.0,
        p_value=float(p_value),
        statistic=float(stat_value),
        stat_type=stat_type,
        method=protocol,
        correct=correct,
        total=total,
        conf_level=conf_level,
        _ci_d_prime=np.array([ci_lower_d, ci_upper_d]),
        _ci_pc=np.array([ci_lower_pc, ci_upper_pc]),
        _ci_pd=np.array([
            pc_to_pd(ci_lower_pc, p_guess)[0] if ci_lower_pc > p_guess else 0.0,
            pc_to_pd(ci_upper_pc, p_guess)[0] if ci_upper_pc > p_guess else 0.0,
        ]),
    )

    return result
