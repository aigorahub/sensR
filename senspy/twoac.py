"""2-AC (2-Alternative Constant) protocol for discrimination and preference testing.

The 2-AC protocol is equivalent to a 2-AFC protocol with a "no-difference" option,
or a paired preference test with a "no-preference" option.

Data format: (count_A, count_no_preference, count_B)
"""

import warnings
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike
from scipy import interpolate, optimize, stats

from senspy.utils import normal_pvalue


@dataclass
class TwoACResult:
    """Result from 2-AC protocol analysis.

    Attributes
    ----------
    coefficients : np.ndarray
        2x2 matrix with estimates and standard errors for (tau, d_prime).
    vcov : np.ndarray | None
        2x2 variance-covariance matrix, or None if not computable.
    log_likelihood : float
        Log-likelihood at the MLE.
    data : np.ndarray
        The input data (count_A, count_no_pref, count_B).
    d_prime_0 : float
        Null hypothesis value of d-prime.
    alternative : str
        Alternative hypothesis type.
    statistic : str
        Test statistic type ('likelihood' or 'wald').
    conf_level : float
        Confidence level for intervals.
    stat_value : float | None
        Test statistic value.
    p_value : float | None
        P-value from significance test.
    confint : np.ndarray | None
        Confidence interval for d-prime.
    """

    coefficients: np.ndarray
    vcov: np.ndarray | None
    log_likelihood: float
    data: np.ndarray
    d_prime_0: float
    alternative: str
    statistic: str
    conf_level: float
    stat_value: float | None = None
    p_value: float | None = None
    confint: np.ndarray | None = None

    @property
    def tau(self) -> float:
        """Threshold parameter estimate."""
        return self.coefficients[0, 0]

    @property
    def d_prime(self) -> float:
        """D-prime estimate."""
        return self.coefficients[1, 0]

    @property
    def se_tau(self) -> float | None:
        """Standard error of tau."""
        se = self.coefficients[0, 1]
        return None if np.isnan(se) else se

    @property
    def se_d_prime(self) -> float | None:
        """Standard error of d-prime."""
        se = self.coefficients[1, 1]
        return None if np.isnan(se) else se

    def __str__(self) -> str:
        """Format result as string."""
        lines = []
        lines.append(f"Results for the 2-AC protocol with data {self.data}:")
        lines.append("")

        # Coefficients table
        lines.append(f"{'':12} {'Estimate':>12} {'Std. Error':>12}")
        tau_se = f"{self.se_tau:.4f}" if self.se_tau is not None else "NA"
        dp_se = f"{self.se_d_prime:.4f}" if self.se_d_prime is not None else "NA"
        lines.append(f"{'tau':<12} {self.tau:>12.4f} {tau_se:>12}")
        lines.append(f"{'d_prime':<12} {self.d_prime:>12.4f} {dp_se:>12}")

        # Confidence interval
        if self.confint is not None:
            lines.append("")
            stat_name = "likelihood root" if self.statistic == "likelihood" else "Wald"
            lines.append(
                f"Two-sided {self.conf_level * 100:.1f}% confidence interval for d-prime "
                f"based on the {stat_name} statistic:"
            )
            lines.append(f"  Lower: {self.confint[0]:.4f}  Upper: {self.confint[1]:.4f}")

        # Significance test
        if self.stat_value is not None and self.p_value is not None:
            lines.append("")
            lines.append("Significance test:")
            stat_name = "Likelihood root" if self.statistic == "likelihood" else "Wald"
            lines.append(f"  {stat_name} statistic = {self.stat_value:.4f}, p-value = {self.p_value:.4g}")
            alt_text = {
                "two.sided": "different from",
                "less": "less than",
                "greater": "greater than",
            }[self.alternative]
            lines.append(f"  Alternative hypothesis: d-prime is {alt_text} {self.d_prime_0}")

        return "\n".join(lines)


def _nll_2ac(tau: float, d_prime: float, data: np.ndarray) -> float:
    """Negative log-likelihood for 2-AC protocol.

    Parameters
    ----------
    tau : float
        Threshold parameter (must be > 0 for valid model).
    d_prime : float
        D-prime parameter.
    data : array
        Counts (n_A, n_no_pref, n_B).

    Returns
    -------
    float
        Negative log-likelihood.
    """
    sqrt2 = np.sqrt(2)
    p1 = stats.norm.cdf(-tau, loc=d_prime, scale=sqrt2)
    p12 = stats.norm.cdf(tau, loc=d_prime, scale=sqrt2)
    prob = np.array([p1, p12 - p1, 1 - p12])

    # Avoid log(0)
    prob = np.maximum(prob, 1e-300)

    return -np.sum(data * np.log(prob))


def _estimate_2ac(data: np.ndarray, compute_vcov: bool = True) -> dict:
    """Estimate parameters of the 2-AC model.

    Parameters
    ----------
    data : array
        Counts (n_A, n_no_pref, n_B).
    compute_vcov : bool
        Whether to compute variance-covariance matrix.

    Returns
    -------
    dict
        Dictionary with 'coefficients', 'vcov', 'log_likelihood'.
    """
    x = data.astype(float)

    # Handle special cases
    # Case 1: x1>0, x2=0, x3=0 -> d'=-inf, tau=0
    if x[0] > 0 and x[1] == 0 and x[2] == 0:
        tau, d_prime = 0.0, -np.inf
    # Case 2: x1=0, x2>0, x3=0 -> d'=NA, tau=NA
    elif x[0] == 0 and x[1] > 0 and x[2] == 0:
        tau, d_prime = np.nan, np.nan
    # Case 3: x1=0, x2=0, x3>0 -> d'=+inf, tau=0
    elif x[0] == 0 and x[1] == 0 and x[2] > 0:
        tau, d_prime = 0.0, np.inf
    # Case 4: x1>0, x2>0, x3=0 -> d'=-inf, tau=NA
    elif x[0] > 0 and x[1] > 0 and x[2] == 0:
        tau, d_prime = np.nan, -np.inf
    # Case 5: x1=0, x2>0, x3>0 -> d'=+inf, tau=NA
    elif x[0] == 0 and x[1] > 0 and x[2] > 0:
        tau, d_prime = np.nan, np.inf
    # Case 0 and 6: General case with closed-form solution
    else:
        prob = x / np.sum(x)
        gamma = np.cumsum(prob)[:2]  # Cumulative probabilities for categories 1 and 1+2
        z = stats.norm.ppf(gamma) * np.sqrt(2)
        tau = (z[1] - z[0]) / 2
        d_prime = -z[0] - tau

    # Log-likelihood at MLE (saturated model)
    prob = x / np.sum(x)
    prob = np.maximum(prob, 1e-300)  # Avoid log(0)
    log_lik_max = np.sum(x * np.log(prob))

    # Coefficient table
    coef = np.full((2, 2), np.nan)
    coef[:, 0] = [tau, d_prime]

    result = {
        "coefficients": coef,
        "vcov": None,
        "log_likelihood": log_lik_max,
    }

    # Compute vcov and standard errors
    if compute_vcov and np.isfinite(tau) and np.isfinite(d_prime) and tau > 0:
        try:
            hess = _compute_hessian_2ac(tau, d_prime, data)
            vcov = np.linalg.inv(hess)
            result["vcov"] = vcov
            result["coefficients"][:, 1] = np.sqrt(np.diag(vcov))
        except (np.linalg.LinAlgError, ValueError):
            pass

    return result


def _compute_hessian_2ac(tau: float, d_prime: float, data: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Compute Hessian of negative log-likelihood numerically.

    Uses central differences with bounds checking to ensure tau remains positive
    during finite difference steps.
    """
    # Ensure eps is small enough that tau - eps remains positive
    # Use one-sided differences if tau is too small
    tau_eps = min(eps, tau / 2) if tau > 2 * eps else tau / 2
    if tau_eps < 1e-10:
        raise ValueError("tau too small for stable Hessian computation")

    def nll(params):
        # Ensure tau stays positive
        t, d = params[0], params[1]
        if t <= 0:
            return np.inf
        return _nll_2ac(t, d, data)

    x = np.array([tau, d_prime])
    n = 2
    hess = np.zeros((n, n))

    # Use different step sizes for tau (index 0) and d_prime (index 1)
    eps_vec = np.array([tau_eps, eps])

    for i in range(n):
        for j in range(i, n):
            x_pp = x.copy()
            x_pm = x.copy()
            x_mp = x.copy()
            x_mm = x.copy()

            x_pp[i] += eps_vec[i]
            x_pp[j] += eps_vec[j]
            x_pm[i] += eps_vec[i]
            x_pm[j] -= eps_vec[j]
            x_mp[i] -= eps_vec[i]
            x_mp[j] += eps_vec[j]
            x_mm[i] -= eps_vec[i]
            x_mm[j] -= eps_vec[j]

            nll_pp = nll(x_pp)
            nll_pm = nll(x_pm)
            nll_mp = nll(x_mp)
            nll_mm = nll(x_mm)

            # Check for invalid values
            if any(np.isinf([nll_pp, nll_pm, nll_mp, nll_mm])):
                raise ValueError("Hessian computation encountered invalid likelihood values")

            hess[i, j] = (nll_pp - nll_pm - nll_mp + nll_mm) / (4 * eps_vec[i] * eps_vec[j])
            hess[j, i] = hess[i, j]

    return hess


def _lr_test_2ac(
    data: np.ndarray,
    d_prime_0: float = 0.0,
    alternative: str = "two.sided",
) -> tuple[float, float]:
    """Compute likelihood ratio test for 2-AC.

    Returns
    -------
    tuple[float, float]
        (p_value, lroot) where lroot is the signed likelihood root statistic.
    """
    x = data.astype(float)

    # Get MLE d-prime
    est = _estimate_2ac(data, compute_vcov=False)
    d_prime_hat = est["coefficients"][1, 0]
    log_lik_max = est["log_likelihood"]

    # Compute log-likelihood under null (d' = d_prime_0)
    # Need to optimize over tau
    def nll_tau(tau):
        return _nll_2ac(tau, d_prime_0, x)

    # Handle special cases based on data pattern
    if x[1] == 0 and x[0] > 0 and x[2] > 0:
        # Case 6: x2 = 0, optimize tau
        result = optimize.minimize_scalar(nll_tau, bounds=(1e-10, 10), method="bounded")
        nll_0 = result.fun
    elif x[1] == 0:
        # Cases 1, 3 with x2 = 0
        nll_0 = nll_tau(0)
    elif x[0] == 0 and x[2] == 0:
        # Case 2: x1 = x3 = 0, x2 > 0
        nll_0 = 0  # Perfect fit under null
    else:
        # General case: optimize tau
        result = optimize.minimize_scalar(nll_tau, bounds=(1e-10, 10), method="bounded")
        nll_0 = result.fun

    # Signed likelihood root statistic
    lr_stat = 2 * (log_lik_max + nll_0)
    if lr_stat < 0:
        lr_stat = 0  # Numerical precision issue
    lroot = np.sign(d_prime_hat - d_prime_0) * np.sqrt(lr_stat)

    # P-value based on alternative
    if alternative == "greater":
        p_value = stats.norm.sf(lroot)
    elif alternative == "less":
        p_value = stats.norm.cdf(lroot)
    else:  # two.sided
        p_value = 2 * stats.norm.sf(np.abs(lroot))

    return p_value, lroot


def _wald_confint_2ac(
    d_prime: float,
    se_d_prime: float,
    level: float = 0.95,
) -> np.ndarray:
    """Compute Wald confidence interval for d-prime."""
    alpha = (1 - level) / 2
    z = stats.norm.ppf(1 - alpha)
    return np.array([d_prime - z * se_d_prime, d_prime + z * se_d_prime])


def _profile_confint_2ac(
    data: np.ndarray,
    log_lik_max: float,
    d_prime_hat: float,
    se_d_prime: float | None = None,
    level: float = 0.95,
    n_steps: int = 100,
) -> np.ndarray:
    """Compute profile likelihood confidence interval for d-prime.

    Parameters
    ----------
    data : np.ndarray
        Counts (n_A, n_no_pref, n_B).
    log_lik_max : float
        Log-likelihood at the MLE.
    d_prime_hat : float
        MLE of d-prime.
    se_d_prime : float | None
        Pre-calculated standard error of d-prime (to avoid redundant computation).
    level : float
        Confidence level.
    n_steps : int
        Number of steps for profile likelihood grid.

    Returns
    -------
    np.ndarray
        Confidence interval [lower, upper].
    """
    # Determine search range using SE if available
    if se_d_prime is not None and np.isfinite(se_d_prime) and se_d_prime > 0:
        d_range = [d_prime_hat - 4 * se_d_prime, d_prime_hat + 4 * se_d_prime]
    else:
        # Default range
        d_range = [d_prime_hat - 3, d_prime_hat + 3]

    # Profile likelihood
    d_seq = np.linspace(d_range[0], d_range[1], n_steps)
    nll_profile = np.zeros(n_steps)

    for i, d in enumerate(d_seq):
        # Optimize tau for each d-prime
        def nll_tau(tau, d_val=d):
            return _nll_2ac(tau, d_val, data)

        result = optimize.minimize_scalar(nll_tau, bounds=(1e-10, 10), method="bounded")
        nll_profile[i] = result.fun

    # Signed likelihood root statistic
    lroot = np.sign(d_prime_hat - d_seq) * np.sqrt(np.maximum(0, 2 * (log_lik_max + nll_profile)))

    # Find CI by interpolation
    # Sort by lroot for interpolation (lroot should be monotonically decreasing with d_seq)
    try:
        # Sort by lroot value
        idx = np.argsort(lroot)
        lroot_sorted = lroot[idx]
        d_sorted = d_seq[idx]

        # Cutoff values for the CI
        alpha = (1 - level) / 2
        lower_cutoff = stats.norm.ppf(alpha)  # negative value
        upper_cutoff = stats.norm.ppf(1 - alpha)  # positive value

        # Interpolate: given lroot value, find d_prime
        spline = interpolate.interp1d(lroot_sorted, d_sorted, kind="linear", fill_value="extrapolate")

        # Lower bound: where lroot = lower_cutoff (negative, so smaller d_prime)
        # Upper bound: where lroot = upper_cutoff (positive, so larger d_prime)
        lower = float(spline(lower_cutoff))
        upper = float(spline(upper_cutoff))

        # Ensure lower < upper
        if lower > upper:
            lower, upper = upper, lower

        return np.array([lower, upper])
    except (ValueError, RuntimeError, IndexError) as e:
        warnings.warn(f"Profile CI interpolation failed: {e}")
        return np.array([np.nan, np.nan])


def twoac(
    data: ArrayLike,
    d_prime_0: float = 0.0,
    conf_level: float = 0.95,
    statistic: str = "likelihood",
    alternative: str = "two.sided",
) -> TwoACResult:
    """Analyze data from the 2-AC (2-Alternative Constant) protocol.

    The 2-AC protocol is equivalent to a 2-AFC protocol with a "no-difference"
    option, or a paired preference test with a "no-preference" option.

    Parameters
    ----------
    data : array-like
        A non-negative integer vector of length 3 with counts in the form
        (prefer_A, no_preference, prefer_B). If prefer_B > prefer_A, the
        estimate of d-prime is positive.
    d_prime_0 : float, default 0.0
        Value of d-prime under the null hypothesis.
    conf_level : float, default 0.95
        Confidence level for confidence intervals.
    statistic : str, default "likelihood"
        Test statistic type: "likelihood" or "wald".
    alternative : str, default "two.sided"
        Alternative hypothesis: "two.sided", "less", or "greater".

    Returns
    -------
    TwoACResult
        Result object with estimates, confidence intervals, and test results.

    Examples
    --------
    >>> from senspy import twoac
    >>> # Simple discrimination test
    >>> result = twoac([2, 2, 6])
    >>> print(f"d-prime = {result.d_prime:.3f}")

    >>> # Discrimination-difference test
    >>> result = twoac([2, 5, 8], d_prime_0=0, alternative="greater")
    >>> print(f"p-value = {result.p_value:.4f}")
    """
    # Validate inputs
    data = np.asarray(data, dtype=float)
    if len(data) != 3:
        raise ValueError("'data' must be a vector of length 3")
    if not np.allclose(np.round(data), data):
        raise ValueError("'data' must contain integer values")
    if np.any(data < 0):
        raise ValueError("'data' must contain non-negative values")

    data = np.round(data).astype(int)

    statistic = statistic.lower()
    if statistic not in ("likelihood", "wald"):
        raise ValueError("'statistic' must be 'likelihood' or 'wald'")

    alternative = alternative.lower().replace("-", ".").replace("_", ".")
    if alternative not in ("two.sided", "less", "greater"):
        raise ValueError("'alternative' must be 'two.sided', 'less', or 'greater'")

    if not 0 < conf_level < 1:
        raise ValueError("'conf_level' must be between 0 and 1")

    # Get ML estimates
    est = _estimate_2ac(data, compute_vcov=True)
    d_prime = est["coefficients"][1, 0]
    se_d_prime = est["coefficients"][1, 1]

    # Initialize result
    result = TwoACResult(
        coefficients=est["coefficients"],
        vcov=est["vcov"],
        log_likelihood=est["log_likelihood"],
        data=data,
        d_prime_0=d_prime_0,
        alternative=alternative,
        statistic=statistic,
        conf_level=conf_level,
    )

    # Compute test statistic
    if statistic == "likelihood":
        p_value, lroot = _lr_test_2ac(data, d_prime_0, alternative)
        result.stat_value = lroot
        result.p_value = p_value
    elif statistic == "wald" and est["vcov"] is not None:
        wald_stat = (d_prime - d_prime_0) / se_d_prime
        result.stat_value = wald_stat
        p_val = normal_pvalue(wald_stat, alternative)
        result.p_value = float(p_val) if np.ndim(p_val) == 0 else float(p_val[0])

    # Compute confidence interval
    if est["vcov"] is not None:
        if statistic == "wald":
            result.confint = _wald_confint_2ac(d_prime, se_d_prime, conf_level)
        else:  # likelihood
            result.confint = _profile_confint_2ac(
                data, est["log_likelihood"], d_prime, se_d_prime, conf_level
            )

    return result
