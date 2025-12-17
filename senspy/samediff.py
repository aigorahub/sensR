"""Same-Different protocol for sensory discrimination testing.

The Same-Different protocol presents participants with pairs of samples that are
either the same (identical products) or different (two different products).
Participants judge each pair as "same" or "different".

Data format: (same_same, diff_same, same_diff, diff_diff)
- same_same: Same response to same samples
- diff_same: Different response to same samples
- same_diff: Same response to different samples
- diff_diff: Different response to different samples
"""

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import optimize, stats


@dataclass
class SameDiffResult:
    """Result from Same-Different protocol analysis.

    Attributes
    ----------
    coefficients : np.ndarray
        2x2 matrix with estimates and standard errors for (tau, delta).
    vcov : np.ndarray | None
        2x2 variance-covariance matrix, or None if not computable.
    log_likelihood : float
        Log-likelihood at the MLE.
    data : np.ndarray
        The input data (same_same, diff_same, same_diff, diff_diff).
    case : float
        Case indicator for boundary conditions.
    convergence : int | None
        Convergence indicator from optimization (0 = converged).
    """

    coefficients: np.ndarray
    vcov: np.ndarray | None
    log_likelihood: float
    data: np.ndarray
    case: float
    convergence: int | None = None

    @property
    def tau(self) -> float:
        """Threshold parameter estimate."""
        return self.coefficients[0, 0]

    @property
    def delta(self) -> float:
        """Delta (d-prime) estimate."""
        return self.coefficients[1, 0]

    @property
    def se_tau(self) -> float | None:
        """Standard error of tau."""
        se = self.coefficients[0, 1]
        return None if np.isnan(se) else se

    @property
    def se_delta(self) -> float | None:
        """Standard error of delta."""
        se = self.coefficients[1, 1]
        return None if np.isnan(se) else se

    def __str__(self) -> str:
        """Format result as string."""
        lines = []
        lines.append("Same-Different Test Results")
        lines.append(f"Data: ss={self.data[0]}, ds={self.data[1]}, sd={self.data[2]}, dd={self.data[3]}")
        lines.append("")

        # Coefficients table
        lines.append(f"{'':12} {'Estimate':>12} {'Std. Error':>12}")
        tau_se = f"{self.se_tau:.4f}" if self.se_tau is not None else "NA"
        delta_se = f"{self.se_delta:.4f}" if self.se_delta is not None else "NA"

        tau_str = f"{self.tau:.4f}" if np.isfinite(self.tau) else ("Inf" if self.tau > 0 else "-Inf")
        delta_str = f"{self.delta:.4f}" if np.isfinite(self.delta) else ("Inf" if self.delta > 0 else "NA" if np.isnan(self.delta) else "-Inf")

        lines.append(f"{'tau':<12} {tau_str:>12} {tau_se:>12}")
        lines.append(f"{'delta':<12} {delta_str:>12} {delta_se:>12}")

        lines.append("")
        lines.append(f"Log-likelihood: {self.log_likelihood:.4f}")

        return "\n".join(lines)


def _ll_samediff(tau: float, delta: float, ss: float, ds: float, sd: float, dd: float) -> float:
    """Log-likelihood for Same-Different model.

    Parameters
    ----------
    tau : float
        Threshold parameter (must be > 0).
    delta : float
        D-prime parameter (must be > 0).
    ss, ds, sd, dd : float
        Data counts.

    Returns
    -------
    float
        Log-likelihood value.
    """
    if tau <= 0 or delta <= 0:
        return -np.inf

    sqrt2 = np.sqrt(2)

    # Probabilities
    p_ss = 2 * stats.norm.cdf(tau / sqrt2) - 1  # P(same | same)
    p_ds = 1 - p_ss  # P(diff | same)
    p_sd = stats.norm.cdf((tau - delta) / sqrt2) - stats.norm.cdf((-tau - delta) / sqrt2)  # P(same | diff)
    p_dd = 1 - p_sd  # P(diff | diff)

    # Avoid log(0)
    probs = np.array([p_ss, p_ds, p_sd, p_dd])
    data = np.array([ss, ds, sd, dd])

    # Handle zero probabilities
    probs = np.maximum(probs, 1e-300)

    return np.sum(data * np.log(probs))


def _ll_delta_inf(tau: float, ss: float, ds: float) -> float:
    """Log-likelihood when delta = Inf (only same-samples data).

    Parameters
    ----------
    tau : float
        Threshold parameter.
    ss, ds : float
        Same-sample data counts.

    Returns
    -------
    float
        Log-likelihood value.
    """
    if tau <= 0:
        return -np.inf

    sqrt2 = np.sqrt(2)
    p_ss = 2 * stats.norm.cdf(tau / sqrt2) - 1
    p_ds = 1 - p_ss

    # Avoid log(0)
    p_ss = max(p_ss, 1e-300)
    p_ds = max(p_ds, 1e-300)

    return ss * np.log(p_ss) + ds * np.log(p_ds)


def _ll_ds0(params: np.ndarray, sd: float, dd: float) -> float:
    """Log-likelihood when ds = 0 (optimize over delta and tau).

    Parameters
    ----------
    params : array
        [delta, tau] parameters.
    sd, dd : float
        Different-sample data counts.

    Returns
    -------
    float
        Log-likelihood value.
    """
    delta, tau = params[0], params[1]
    if delta <= 0 or tau <= 0:
        return -np.inf

    sqrt2 = np.sqrt(2)
    p_sd = stats.norm.cdf((tau - delta) / sqrt2) - stats.norm.cdf((-tau - delta) / sqrt2)
    p_dd = 1 - p_sd

    p_sd = max(p_sd, 1e-300)
    p_dd = max(p_dd, 1e-300)

    return sd * np.log(p_sd) + dd * np.log(p_dd)


def _compute_tau(ss: float, ds: float) -> float:
    """Compute tau from same-sample data.

    Parameters
    ----------
    ss, ds : float
        Same-sample counts.

    Returns
    -------
    float
        Tau estimate.
    """
    frac = (2 * ss + ds) / (2 * (ds + ss))
    tau = np.sqrt(2) * stats.norm.ppf(frac)
    return tau


def _delta_root(delta: float, tau: float, psd: float) -> float:
    """Root function for finding delta.

    Parameters
    ----------
    delta : float
        D-prime parameter.
    tau : float
        Threshold parameter.
    psd : float
        Observed proportion of same responses to different samples.

    Returns
    -------
    float
        Difference from target (should be 0 at root).
    """
    sqrt2 = np.sqrt(2)
    p_sd_model = stats.norm.cdf((tau - delta) / sqrt2) - stats.norm.cdf((-tau - delta) / sqrt2)
    return p_sd_model - psd


def _fisher_info_11(tau: float, delta: float, ss: float, ds: float, sd: float, dd: float) -> float:
    """Compute I_11 element of Fisher information matrix."""
    sqrt2 = np.sqrt(2)

    E1 = (dd + sd) ** 3 / (2 * dd * sd)
    E2 = (stats.norm.pdf((tau - delta) / sqrt2) + stats.norm.pdf((-tau - delta) / sqrt2)) ** 2
    E3 = 2 * (ds + ss) ** 3 / (ds * ss)
    E4 = stats.norm.pdf(tau / sqrt2) ** 2

    return E1 * E2 + E3 * E4


def _fisher_info_12(tau: float, delta: float, sd: float, dd: float) -> float:
    """Compute I_12 = I_21 element of Fisher information matrix."""
    sqrt2 = np.sqrt(2)

    E1 = -(dd + sd) ** 3 / (2 * dd * sd)
    E2 = stats.norm.pdf((tau - delta) / sqrt2) ** 2 - stats.norm.pdf((-tau - delta) / sqrt2) ** 2

    return E1 * E2


def _fisher_info_22(tau: float, delta: float, sd: float, dd: float) -> float:
    """Compute I_22 element of Fisher information matrix."""
    sqrt2 = np.sqrt(2)

    E1 = (dd + sd) ** 3 / (2 * dd * sd)
    E2 = (stats.norm.pdf((tau - delta) / sqrt2) - stats.norm.pdf((-tau - delta) / sqrt2)) ** 2

    return E1 * E2


def samediff(
    nsamesame: int | None = None,
    ndiffsame: int | None = None,
    nsamediff: int | None = None,
    ndiffdiff: int | None = None,
    data: ArrayLike | None = None,
    vcov: bool = True,
) -> SameDiffResult:
    """Analyze data from the Same-Different protocol.

    The Same-Different protocol presents participants with pairs of samples
    that are either the same or different. Participants judge each pair as
    "same" or "different".

    Parameters
    ----------
    nsamesame : int, optional
        Number of same-answers on same-samples.
    ndiffsame : int, optional
        Number of different-answers on same-samples.
    nsamediff : int, optional
        Number of same-answers on different-samples.
    ndiffdiff : int, optional
        Number of different-answers on different-samples.
    data : array-like, optional
        Alternative input: array of [nsamesame, ndiffsame, nsamediff, ndiffdiff].
    vcov : bool, default True
        Whether to compute variance-covariance matrix.

    Returns
    -------
    SameDiffResult
        Result object with estimates and inference.

    Examples
    --------
    >>> from senspy import samediff
    >>> # 8 same-same, 5 diff-same, 4 same-diff, 9 diff-diff
    >>> result = samediff(8, 5, 4, 9)
    >>> print(f"tau = {result.tau:.3f}, delta = {result.delta:.3f}")
    """
    # Handle input
    if data is not None:
        data = np.asarray(data, dtype=float)
        if len(data) != 4:
            raise ValueError("'data' must be a vector of length 4")
        ss, ds, sd, dd = data
    elif all(x is not None for x in [nsamesame, ndiffsame, nsamediff, ndiffdiff]):
        ss, ds, sd, dd = float(nsamesame), float(ndiffsame), float(nsamediff), float(ndiffdiff)
        data = np.array([ss, ds, sd, dd])
    else:
        raise ValueError("Either provide all four counts or use 'data' parameter")

    # Validate data
    if not all(x == int(x) and x >= 0 for x in [ss, ds, sd, dd]):
        raise ValueError("Data must be non-negative integers")

    zero_count = sum(1 for x in [ss, ds, sd, dd] if x == 0)
    if zero_count > 2:
        raise ValueError("Not enough information in data when more than two entries are zero")

    # Initialize result containers
    vcov_mat = np.full((2, 2), np.nan)
    se = np.array([np.nan, np.nan])
    conv = None
    case = 0.0

    # Check if simple estimation is possible
    # delta is estimable if P(same|same) > P(same|diff)
    is_delta = (ss / (ds + ss) > sd / (sd + dd)) if (ds + ss > 0 and sd + dd > 0) else False

    # Estimation based on case
    if is_delta and all(x > 0 for x in [ss, ds, sd, dd]):
        # Case 0: General case - all data positive and delta estimable
        tau = _compute_tau(ss, ds)
        psd = sd / (sd + dd)

        # Find delta by root-finding
        try:
            result = optimize.brentq(_delta_root, 0, 10, args=(tau, psd))
            delta = result
        except ValueError:
            # Root not found in interval, try larger range
            try:
                result = optimize.brentq(_delta_root, 0, 100, args=(tau, psd))
                delta = result
            except ValueError:
                delta = np.nan

        if np.isfinite(delta) and vcov:
            # Compute Fisher information and vcov
            try:
                i11 = _fisher_info_11(tau, delta, ss, ds, sd, dd)
                i12 = _fisher_info_12(tau, delta, sd, dd)
                i22 = _fisher_info_22(tau, delta, sd, dd)

                fisher = np.array([[i11, i12], [i12, i22]])
                vcov_mat = np.linalg.inv(fisher)
                se = np.sqrt(np.diag(vcov_mat))
            except (np.linalg.LinAlgError, ValueError):
                pass

        log_lik = _ll_samediff(tau, delta, ss, ds, sd, dd)
        case = 0.0

    elif ss == 0 and sd == 0:
        # Case 0.1: No same responses -> tau = 0, delta = NA
        tau = 0.0
        delta = np.nan
        log_lik = 0.0
        case = 0.1

    elif ds == 0 and dd == 0:
        # Case 1: No different responses -> tau = Inf, delta = NA
        tau = np.inf
        delta = np.nan
        log_lik = 0.0
        case = 1.0

    elif ds == 0 and sd == 0:
        # Case 1.2: tau = Inf, delta = Inf
        tau = np.inf
        delta = np.inf
        log_lik = 0.0
        case = 1.2

    elif ss == 0 and ds == 0:
        # Case 1.12: No same-sample data -> optimize over tau, delta
        tau = np.inf
        delta = np.nan

        # Optimize likelihood
        result = optimize.minimize(
            lambda p: -_ll_ds0(p, sd, dd),
            x0=[5.0, 5.0],
            method="L-BFGS-B",
            bounds=[(1e-4, 100), (1e-4, 100)],
        )
        log_lik = -result.fun
        conv = result.status
        case = 1.12

    elif sd == 0 and dd == 0:
        # Case 1.3: No different-sample data -> optimize tau, delta = NA
        delta = np.nan

        # Optimize tau
        result = optimize.minimize(
            lambda t: -_ll_delta_inf(t[0], ss, ds),
            x0=[1.0],
            method="BFGS",
        )
        tau = result.x[0]
        log_lik = _ll_delta_inf(tau, ss, ds)
        conv = 0 if result.success else 1

        if vcov:
            # Compute Hessian for SE
            eps = 1e-5
            ll_p = _ll_delta_inf(tau + eps, ss, ds)
            ll_m = _ll_delta_inf(tau - eps, ss, ds)
            ll_0 = _ll_delta_inf(tau, ss, ds)
            hess = (ll_p - 2 * ll_0 + ll_m) / (eps ** 2)
            if hess < 0:
                vcov_mat[0, 0] = -1 / hess
                se[0] = np.sqrt(vcov_mat[0, 0])

        case = 1.3

    elif ds == 0:
        # Case 1.22: ds = 0 -> tau = Inf, delta = Inf
        tau = np.inf
        delta = np.inf

        result = optimize.minimize(
            lambda p: -_ll_ds0(p, sd, dd),
            x0=[5.0, 5.0],
            method="L-BFGS-B",
            bounds=[(1e-4, 100), (1e-4, 100)],
        )
        log_lik = -result.fun
        conv = result.status
        case = 1.22

    elif sd == 0:
        # Case 3: sd = 0 -> delta = Inf, optimize tau
        delta = np.inf

        result = optimize.minimize(
            lambda t: -_ll_delta_inf(t[0], ss, ds),
            x0=[1.0],
            method="BFGS",
        )
        tau = result.x[0]
        log_lik = _ll_delta_inf(tau, ss, ds)
        conv = 0 if result.success else 1

        if vcov:
            eps = 1e-5
            ll_p = _ll_delta_inf(tau + eps, ss, ds)
            ll_m = _ll_delta_inf(tau - eps, ss, ds)
            ll_0 = _ll_delta_inf(tau, ss, ds)
            hess = (ll_p - 2 * ll_0 + ll_m) / (eps ** 2)
            if hess < 0:
                vcov_mat[0, 0] = -1 / hess
                se[0] = np.sqrt(vcov_mat[0, 0])

        case = 3.0

    elif dd == 0 or ss == 0 or not is_delta:
        # Case 2: delta = 0, optimize tau
        delta = 0.0

        def neg_ll_delta_zero(t):
            return -_ll_samediff(t[0], 1e-4, ss, ds, sd, dd)

        result = optimize.minimize(neg_ll_delta_zero, x0=[1.0], method="BFGS")
        tau = result.x[0]
        log_lik = _ll_samediff(tau, 1e-4, ss, ds, sd, dd)
        conv = 0 if result.success else 1

        if vcov:
            eps = 1e-5
            ll_p = _ll_samediff(tau + eps, 1e-4, ss, ds, sd, dd)
            ll_m = _ll_samediff(tau - eps, 1e-4, ss, ds, sd, dd)
            ll_0 = _ll_samediff(tau, 1e-4, ss, ds, sd, dd)
            hess = (ll_p - 2 * ll_0 + ll_m) / (eps ** 2)
            if hess < 0:
                vcov_mat[0, 0] = -1 / hess
                se[0] = np.sqrt(vcov_mat[0, 0])

        case = 2.0

    else:
        raise RuntimeError("Unexpected case in samediff estimation")

    # Build coefficient matrix
    coef = np.full((2, 2), np.nan)
    coef[0, 0] = tau
    coef[1, 0] = delta
    coef[:, 1] = se

    return SameDiffResult(
        coefficients=coef,
        vcov=vcov_mat if vcov else None,
        log_likelihood=log_lik,
        data=data.astype(int),
        case=case,
        convergence=conv,
    )
