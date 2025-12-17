"""D-prime hypothesis testing functions.

This module implements functions for testing hypotheses about d-prime
values from discrimination experiments, including:
- dprime_test: Test if a common d-prime equals a specified value
- dprime_compare: Test if all d-primes are equal (any-difference test)
- posthoc: Post-hoc pairwise comparisons of d-primes

These functions work with data from multiple discrimination experiments,
potentially using different protocols (triangle, duotrio, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize
from scipy.special import comb
from scipy.stats import norm

from senspy.core.types import Protocol, parse_protocol
from senspy.links import get_link, psy_deriv, psy_fun, psy_inv
from senspy.utils.stats import delimit, normal_pvalue
from senspy.utils.transforms import pc_to_pd


@dataclass
class DprimeTableRow:
    """A row in the d-prime table."""

    correct: int
    total: int
    protocol: str
    p_hat: float
    se_p_hat: float
    dprime: float
    se_dprime: float


@dataclass
class DprimeTestResult:
    """Result from dprime_test function.

    Attributes
    ----------
    d_prime : float
        Estimated common d-prime.
    se_d_prime : float
        Standard error of the common d-prime.
    conf_int : tuple[float, float]
        Confidence interval for common d-prime.
    conf_level : float
        Confidence level used.
    conf_method : str
        Method used for confidence interval.
    stat_value : float
        Value of the test statistic.
    p_value : float
        P-value for the hypothesis test.
    statistic : str
        Type of test statistic used.
    alternative : str
        Alternative hypothesis.
    dprime0 : float
        Null hypothesis value for d-prime.
    estim : str
        Estimation method used ("ML" or "weighted_avg").
    data : list[DprimeTableRow]
        The input data with per-group estimates.
    """

    d_prime: float
    se_d_prime: float
    conf_int: tuple[float, float]
    conf_level: float
    conf_method: str
    stat_value: float
    p_value: float
    statistic: str
    alternative: str
    dprime0: float
    estim: str
    data: list[DprimeTableRow]


@dataclass
class DprimeCompareResult:
    """Result from dprime_compare function.

    Attributes
    ----------
    d_prime : float
        Estimated common d-prime.
    se_d_prime : float
        Standard error of the common d-prime.
    conf_int : tuple[float, float]
        Confidence interval for common d-prime.
    conf_level : float
        Confidence level used.
    conf_method : str
        Method used for confidence interval.
    stat_value : float
        Chi-square test statistic.
    df : int
        Degrees of freedom.
    p_value : float
        P-value for the any-difference test.
    statistic : str
        Type of test statistic used.
    estim : str
        Estimation method used.
    data : list[DprimeTableRow]
        The input data with per-group estimates.
    """

    d_prime: float
    se_d_prime: float
    conf_int: tuple[float, float]
    conf_level: float
    conf_method: str
    stat_value: float
    df: int
    p_value: float
    statistic: str
    estim: str
    data: list[DprimeTableRow]


@dataclass
class PosthocResult:
    """Result from posthoc function.

    Attributes
    ----------
    posthoc : list[dict]
        Post-hoc comparison results.
    test : str
        Type of post-hoc test performed.
    alternative : str
        Alternative hypothesis.
    padj_method : str
        P-value adjustment method.
    letters : dict[str, str] | None
        Letter display for pairwise comparisons (if applicable).
    base_result : DprimeCompareResult | DprimeTestResult
        The original comparison result.
    """

    posthoc: list[dict]
    test: str
    alternative: str
    padj_method: str
    letters: dict[str, str] | None
    base_result: DprimeCompareResult | DprimeTestResult


def _get_p_guess(protocol: str) -> float:
    """Get guessing probability for a protocol."""
    link = get_link(protocol)
    return link.p_guess


def dprime_table(
    correct: ArrayLike,
    total: ArrayLike,
    protocol: ArrayLike,
    restrict_above_guess: bool = True,
) -> list[DprimeTableRow]:
    """Create a table of d-prime estimates for each group.

    Parameters
    ----------
    correct : ArrayLike
        Number of correct responses in each group.
    total : ArrayLike
        Total number of trials in each group.
    protocol : ArrayLike
        Protocol name for each group.
    restrict_above_guess : bool
        If True, restrict p_hat to be at least p_guess. Default is True.

    Returns
    -------
    list[DprimeTableRow]
        List of rows with per-group estimates.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    correct = np.asarray(correct, dtype=int)
    total = np.asarray(total, dtype=int)
    protocol = np.asarray(protocol, dtype=str)

    # Validate inputs
    if not (len(correct) == len(total) == len(protocol)):
        raise ValueError("correct, total, and protocol must have the same length")
    if len(correct) < 1:
        raise ValueError("Need at least one group")
    if np.any(correct < 0) or np.any(total <= 0):
        raise ValueError("Invalid counts: need correct >= 0 and total > 0")
    if np.any(correct > total):
        raise ValueError("correct cannot exceed total")

    valid_protocols = [
        "triangle", "duotrio", "threeAFC", "twoAFC", "tetrad",
        "hexad", "twofive", "twofiveF"
    ]
    for p in protocol:
        if p not in valid_protocols:
            raise ValueError(f"Invalid protocol: {p}. Must be one of {valid_protocols}")

    rows = []
    for i in range(len(correct)):
        x = correct[i]
        n = total[i]
        prot = protocol[i]
        p_guess = _get_p_guess(prot)

        # Compute p_hat
        p_hat = x / n
        if restrict_above_guess and p_hat < p_guess:
            p_hat = p_guess

        se_p_hat = np.sqrt(p_hat * (1 - p_hat) / n)

        # Compute d-prime
        dprime = psy_inv(p_hat, prot)

        # Compute se(d-prime) using delta method
        deriv = psy_deriv(dprime, prot)
        if np.isfinite(deriv) and deriv > 0:
            se_dprime = se_p_hat / deriv
        else:
            se_dprime = np.nan

        rows.append(
            DprimeTableRow(
                correct=x,
                total=n,
                protocol=prot,
                p_hat=p_hat,
                se_p_hat=se_p_hat,
                dprime=dprime,
                se_dprime=se_dprime,
            )
        )

    return rows


def _dprime_nll(dp: float, data: list[DprimeTableRow]) -> float:
    """Negative log-likelihood under common d-prime model.

    Parameters
    ----------
    dp : float
        Common d-prime value.
    data : list[DprimeTableRow]
        Data table from dprime_table().

    Returns
    -------
    float
        Negative log-likelihood.
    """
    nll = 0.0
    for row in data:
        p = psy_fun(dp, row.protocol)
        # Binomial log-likelihood (up to constant)
        log_lik = row.correct * np.log(p) + (row.total - row.correct) * np.log(1 - p)
        nll -= log_lik
    return nll


def _dprime_estim(
    data: list[DprimeTableRow],
    estim: Literal["ML", "weighted_avg"] = "ML",
) -> tuple[float, float, float | None]:
    """Estimate common d-prime and its standard error.

    Parameters
    ----------
    data : list[DprimeTableRow]
        Data table from dprime_table().
    estim : str
        Estimation method: "ML" or "weighted_avg".

    Returns
    -------
    tuple[float, float, float | None]
        (d_prime_estimate, se_estimate, nll_at_estimate or None)
    """
    if estim == "ML":
        # Maximum likelihood estimation
        result = optimize.minimize_scalar(
            lambda dp: _dprime_nll(dp, data),
            bounds=(0, 10),
            method="bounded",
        )
        d_exp = result.x
        nll_exp = result.fun

        # Compute standard error from Hessian
        if d_exp < 1e-4:
            se_exp = np.nan
        else:
            eps = 1e-5
            h = (
                _dprime_nll(d_exp + eps, data)
                - 2 * _dprime_nll(d_exp, data)
                + _dprime_nll(d_exp - eps, data)
            ) / eps**2
            if h > 0:
                se_exp = np.sqrt(1 / h)
            else:
                se_exp = np.nan

        return d_exp, se_exp, nll_exp

    else:  # weighted_avg
        dprimes = np.array([row.dprime for row in data])
        se_dprimes = np.array([row.se_dprime for row in data])

        if not np.all(np.isfinite(np.concatenate([dprimes, se_dprimes]))):
            raise ValueError(
                "Boundary cases occurred: use 'estim = ML' instead"
            )

        # Inverse variance weighting
        # variance = se^2, weights = 1/variance
        variance = se_dprimes**2
        sum_inv_var = np.sum(1 / variance)
        d_exp = np.sum(dprimes / variance) / sum_inv_var
        se_exp = np.sqrt(1 / sum_inv_var)

        return d_exp, se_exp, None


def dprime_test(
    correct: ArrayLike,
    total: ArrayLike,
    protocol: ArrayLike,
    conf_level: float = 0.95,
    dprime0: float = 0.0,
    statistic: Literal["likelihood", "Wald"] = "likelihood",
    alternative: Literal[
        "difference", "similarity", "two.sided", "less", "greater"
    ] = "difference",
    estim: Literal["ML", "weighted_avg"] = "ML",
) -> DprimeTestResult:
    """Test if a common d-prime equals a specified value.

    This function estimates a common d-prime from multiple groups (potentially
    using different protocols) and tests whether it equals a specified null
    hypothesis value.

    Parameters
    ----------
    correct : ArrayLike
        Number of correct responses in each group.
    total : ArrayLike
        Total number of trials in each group.
    protocol : ArrayLike
        Protocol name for each group (e.g., "triangle", "duotrio").
    conf_level : float
        Confidence level for interval. Default is 0.95.
    dprime0 : float
        Null hypothesis value for d-prime. Default is 0.
    statistic : str
        Test statistic: "likelihood" or "Wald". Default is "likelihood".
    alternative : str
        Alternative hypothesis:
        - "difference" or "greater": d-prime > dprime0 (default)
        - "similarity" or "less": d-prime < dprime0
        - "two.sided": d-prime != dprime0
    estim : str
        Estimation method: "ML" or "weighted_avg". Default is "ML".

    Returns
    -------
    DprimeTestResult
        Test results including estimate, confidence interval, and p-value.

    Raises
    ------
    ValueError
        If inputs are invalid.

    Examples
    --------
    >>> result = dprime_test(
    ...     correct=[60, 45, 55],
    ...     total=[100, 100, 100],
    ...     protocol=["triangle", "duotrio", "twoAFC"]
    ... )
    >>> print(f"Common d-prime: {result.d_prime:.3f}")
    >>> print(f"p-value: {result.p_value:.4f}")
    """
    # Map alternative names
    if alternative == "difference":
        alternative = "greater"
    elif alternative == "similarity":
        alternative = "less"

    # Validate
    if not 0 < conf_level < 1:
        raise ValueError("conf_level must be between 0 and 1")
    if dprime0 < 0:
        raise ValueError("dprime0 must be non-negative")
    if np.isclose(dprime0, 0.0) and alternative not in ["difference", "greater"]:
        raise ValueError(
            "'alternative' has to be 'difference'/'greater' if 'dprime0' is 0"
        )

    # Get data table
    data = dprime_table(correct, total, protocol)

    # Estimate common d-prime
    d_exp, se_exp, nll_exp = _dprime_estim(data, estim)

    # Compute test statistic
    if statistic == "likelihood":
        nll_0 = _dprime_nll(dprime0, data)
        if nll_exp is None:
            nll_exp = _dprime_nll(d_exp, data)
        LR = 2 * (nll_0 - nll_exp)
        # Signed likelihood root statistic
        stat_value = np.sign(d_exp - dprime0) * np.sqrt(abs(LR))
    else:  # Wald
        if not np.all(np.isfinite([d_exp, se_exp])):
            raise ValueError(
                "Boundary cases occurred: use 'statistic = likelihood' instead"
            )
        stat_value = (d_exp - dprime0) / se_exp

    # Compute p-value
    p_value = normal_pvalue(stat_value, alternative)

    # Compute Wald confidence interval
    alpha = (1 - conf_level) / 2
    z = norm.ppf(1 - alpha)
    ci_lower = max(0, d_exp - z * se_exp) if np.isfinite(se_exp) else np.nan
    ci_upper = d_exp + z * se_exp if np.isfinite(se_exp) else np.nan
    conf_int = (ci_lower, ci_upper)

    return DprimeTestResult(
        d_prime=d_exp,
        se_d_prime=se_exp,
        conf_int=conf_int,
        conf_level=conf_level,
        conf_method="Wald",
        stat_value=stat_value,
        p_value=p_value,
        statistic=statistic,
        alternative=alternative,
        dprime0=dprime0,
        estim=estim,
        data=data,
    )


def dprime_compare(
    correct: ArrayLike,
    total: ArrayLike,
    protocol: ArrayLike,
    conf_level: float = 0.95,
    statistic: Literal["likelihood", "Pearson", "Wald.p", "Wald.d"] = "likelihood",
    estim: Literal["ML", "weighted_avg"] = "ML",
) -> DprimeCompareResult:
    """Test if all d-primes are equal (any-difference test).

    This function tests the null hypothesis that all groups share a common
    d-prime value against the alternative that at least two differ.

    Parameters
    ----------
    correct : ArrayLike
        Number of correct responses in each group.
    total : ArrayLike
        Total number of trials in each group.
    protocol : ArrayLike
        Protocol name for each group.
    conf_level : float
        Confidence level for interval. Default is 0.95.
    statistic : str
        Test statistic:
        - "likelihood": Likelihood ratio test (default)
        - "Pearson": Pearson chi-square
        - "Wald.p": Wald test on proportions
        - "Wald.d": Wald test on d-primes
    estim : str
        Estimation method: "ML" or "weighted_avg". Default is "ML".

    Returns
    -------
    DprimeCompareResult
        Test results including chi-square statistic, df, and p-value.

    Examples
    --------
    >>> result = dprime_compare(
    ...     correct=[60, 45, 55],
    ...     total=[100, 100, 100],
    ...     protocol=["triangle", "duotrio", "twoAFC"]
    ... )
    >>> print(f"Chi-square: {result.stat_value:.3f}, df={result.df}")
    >>> print(f"p-value: {result.p_value:.4f}")
    """
    # Validate
    if not 0 < conf_level < 1:
        raise ValueError("conf_level must be between 0 and 1")

    # Get data table
    data = dprime_table(correct, total, protocol)
    n_groups = len(data)

    # Estimate common d-prime and get test result for conf interval
    d_exp, se_exp, _ = _dprime_estim(data, estim)

    # Get confidence interval from dprime_test
    test_result = dprime_test(
        correct, total, protocol,
        conf_level=conf_level,
        dprime0=0,
        statistic="likelihood",
        estim=estim,
    )

    # Compute chi-square test statistic
    x = np.array([row.correct for row in data])
    n = np.array([row.total for row in data])
    O = np.concatenate([x, n - x])

    protocols = [row.protocol for row in data]

    if statistic == "likelihood":
        # Expected counts under common d-prime
        # psy_fun returns arrays, so we extract scalar values
        p_exp = np.array([psy_fun(d_exp, p).item() for p in protocols])
        E = np.concatenate([n * p_exp, n * (1 - p_exp)])
        # Avoid log(0)
        O_safe = np.maximum(O, 1e-10)
        E_safe = np.maximum(E, 1e-10)
        X = 2 * np.sum(O_safe * np.log(O_safe / E_safe))

    elif statistic == "Pearson":
        p_exp = np.array([psy_fun(d_exp, p).item() for p in protocols])
        E = np.concatenate([n * p_exp, n * (1 - p_exp)])
        E_safe = np.maximum(E, 1e-10)
        X = np.sum((O - E) ** 2 / E_safe)

    elif statistic == "Wald.p":
        p_exp = np.array([psy_fun(d_exp, p).item() for p in protocols])
        p_hat = np.array([row.p_hat for row in data])
        var_p = p_hat * (1 - p_hat) / n
        var_p_safe = np.maximum(var_p, 1e-10)
        X = np.sum((p_hat - p_exp) ** 2 / var_p_safe)

    elif statistic == "Wald.d":
        dprimes = np.array([row.dprime for row in data])
        se_dprimes = np.array([row.se_dprime for row in data])
        if not np.all(np.isfinite(np.concatenate([dprimes, se_dprimes]))):
            raise ValueError(
                "Boundary cases occurred: use 'likelihood' or 'Pearson' instead"
            )
        X = np.sum(((dprimes - d_exp) / se_dprimes) ** 2)

    else:
        raise ValueError(f"Unknown statistic: {statistic}")

    # Degrees of freedom
    df = n_groups - 1

    # P-value from chi-square distribution
    if df >= 1:
        from scipy.stats import chi2
        p_value = chi2.sf(X, df)
    else:
        p_value = np.nan

    return DprimeCompareResult(
        d_prime=d_exp,
        se_d_prime=se_exp,
        conf_int=test_result.conf_int,
        conf_level=conf_level,
        conf_method=test_result.conf_method,
        stat_value=X,
        df=df,
        p_value=p_value,
        statistic=statistic,
        estim=estim,
        data=data,
    )


def _p_adjust(pvals: NDArray, method: str = "holm") -> NDArray:
    """Adjust p-values for multiple testing.

    Parameters
    ----------
    pvals : NDArray
        Array of p-values.
    method : str
        Adjustment method: "holm", "bonferroni", or "none".

    Returns
    -------
    NDArray
        Adjusted p-values.
    """
    pvals = np.asarray(pvals)
    n = len(pvals)

    if method == "none":
        return pvals
    elif method == "bonferroni":
        return np.minimum(pvals * n, 1.0)
    elif method == "holm":
        # Holm-Bonferroni method
        order = np.argsort(pvals)
        ranks = np.argsort(order)
        adjusted = np.minimum(pvals * (n - ranks), 1.0)
        # Ensure monotonicity
        sorted_adj = adjusted[order]
        for i in range(1, n):
            sorted_adj[i] = max(sorted_adj[i], sorted_adj[i - 1])
        return sorted_adj[np.argsort(order)]
    else:
        raise ValueError(f"Unknown adjustment method: {method}")


def _get_letters(signifs: dict[str, bool]) -> dict[str, str]:
    """Generate compact letter display for pairwise comparisons.

    Uses greedy graph coloring: groups that are significantly different
    get different letters, groups that are not significantly different
    may share the same letter.

    Parameters
    ----------
    signifs : dict[str, bool]
        Dictionary mapping comparison names (e.g., "group1 - group2")
        to significance (True if different).

    Returns
    -------
    dict[str, str]
        Dictionary mapping group names to letter codes.
    """
    # Extract group names
    groups = set()
    for name in signifs.keys():
        parts = name.split(" - ")
        groups.update(parts)
    groups = sorted(groups)
    n = len(groups)

    if n == 0:
        return {}

    # Build conflict matrix (True if significantly different)
    conflicts = [[False] * n for _ in range(n)]
    for comp_name, is_sig in signifs.items():
        if is_sig:
            parts = comp_name.split(" - ")
            i = groups.index(parts[0])
            j = groups.index(parts[1])
            conflicts[i][j] = True
            conflicts[j][i] = True

    # Greedy graph coloring: assign each group the lowest letter
    # not used by any significantly different (conflicting) group
    colors = [-1] * n
    for i in range(n):
        # Find colors used by conflicting groups
        used_colors = {colors[j] for j in range(n) if conflicts[i][j] and colors[j] != -1}
        # Assign lowest available color
        color = 0
        while color in used_colors:
            color += 1
        colors[i] = color

    # Convert colors to letters
    return {groups[i]: chr(ord("a") + colors[i]) for i in range(n)}


def posthoc(
    result: DprimeCompareResult | DprimeTestResult,
    alpha: float = 0.05,
    test: Literal["pairwise", "common", "base", "zero"] | float = "pairwise",
    base: int = 1,
    alternative: Literal["two.sided", "less", "greater"] = "two.sided",
    statistic: Literal["likelihood", "Wald"] = "likelihood",
    padj_method: Literal["holm", "bonferroni", "none"] = "holm",
) -> PosthocResult:
    """Perform post-hoc comparisons of d-primes.

    Parameters
    ----------
    result : DprimeCompareResult | DprimeTestResult
        Result from dprime_compare() or dprime_test().
    alpha : float
        Significance level for letter display. Default is 0.05.
    test : str | float
        Type of comparison:
        - "pairwise": All pairwise differences
        - "common": Compare each to common d-prime
        - "base": Compare each to base group
        - "zero": Compare each to zero
        - float: Compare each to this value
    base : int
        Index of base group (1-indexed) for "base" test. Default is 1.
    alternative : str
        Alternative hypothesis. Default is "two.sided".
    statistic : str
        Test statistic. Default is "likelihood".
    padj_method : str
        P-value adjustment method. Default is "holm".

    Returns
    -------
    PosthocResult
        Post-hoc comparison results.
    """
    data = result.data
    n = len(data)

    # Determine dprime0 for testing
    if isinstance(test, (int, float)) and not isinstance(test, bool):
        dprime0 = float(test)
        test_type = "value"
    elif test == "zero":
        dprime0 = 0.0
        test_type = "zero"
    elif test == "base":
        dprime0 = data[base - 1].dprime  # 1-indexed
        test_type = "base"
    else:
        dprime0 = 0.0
        test_type = test

    posthoc_results = []
    stat_values = []

    if test_type in ["pairwise", "base"]:
        # Pairwise or Dunnett-style comparisons
        if test_type == "pairwise":
            # All pairs
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        else:
            # Compare to base
            pairs = [(base - 1, j) for j in range(n) if j != base - 1]

        for i, j in pairs:
            diff = data[i].dprime - data[j].dprime
            se_diff = np.sqrt(data[i].se_dprime**2 + data[j].se_dprime**2)

            if statistic == "likelihood":
                # Likelihood ratio for pairwise test
                # Fit common model to just these two groups
                sub_correct = [data[i].correct, data[j].correct]
                sub_total = [data[i].total, data[j].total]
                sub_protocol = [data[i].protocol, data[j].protocol]
                sub_data = dprime_table(sub_correct, sub_total, sub_protocol)

                # NLL under null (common d-prime)
                nll_0 = optimize.minimize_scalar(
                    lambda dp: _dprime_nll(dp, sub_data),
                    bounds=(0, 10),
                    method="bounded",
                ).fun

                # NLL under alternative (separate d-primes)
                nll_i = -data[i].correct * np.log(data[i].p_hat) - (
                    data[i].total - data[i].correct
                ) * np.log(1 - data[i].p_hat)
                nll_j = -data[j].correct * np.log(data[j].p_hat) - (
                    data[j].total - data[j].correct
                ) * np.log(1 - data[j].p_hat)
                nll_alt = nll_i + nll_j

                LR = -2 * (nll_alt - nll_0)
                stat = np.sign(diff) * np.sqrt(abs(LR))
            else:
                stat = diff / se_diff if np.isfinite(se_diff) else np.nan

            stat_values.append(stat)
            name = f"group{i+1} - group{j+1}"
            posthoc_results.append({
                "name": name,
                "estimate": diff,
                "se": se_diff,
                "stat_value": stat,
            })

    else:
        # Compare each group to dprime0 (common, zero, or value)
        for i in range(n):
            diff = data[i].dprime - dprime0
            se = data[i].se_dprime

            if statistic == "likelihood" and test_type == "common":
                # Compare to common d-prime using likelihood
                sub_data = [data[i]]
                d_common = result.d_prime
                nll_0 = _dprime_nll(d_common, sub_data)
                nll_alt = _dprime_nll(data[i].dprime, sub_data)
                LR = -2 * (nll_alt - nll_0)
                stat = np.sign(diff) * np.sqrt(abs(LR))
            else:
                stat = diff / se if np.isfinite(se) else np.nan

            stat_values.append(stat)
            posthoc_results.append({
                "name": f"group{i+1}",
                "estimate": data[i].dprime,
                "se": se,
                "stat_value": stat,
            })

    # Compute p-values
    pvals = np.array([normal_pvalue(s, alternative) for s in stat_values])
    adjusted_pvals = _p_adjust(pvals, padj_method)

    for i, res in enumerate(posthoc_results):
        res["p_value"] = adjusted_pvals[i]

    # Generate letter display for pairwise comparisons
    letters = None
    if test_type == "pairwise" and alternative == "two.sided":
        signifs = {
            res["name"]: res["p_value"] < alpha for res in posthoc_results
        }
        letters = _get_letters(signifs)

    return PosthocResult(
        posthoc=posthoc_results,
        test=test_type,
        alternative=alternative,
        padj_method=padj_method,
        letters=letters,
        base_result=result,
    )
