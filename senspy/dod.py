"""Degree of Difference (DOD) model for sensory discrimination.

This module implements the Thurstonian model for degree-of-difference (DOD)
data. The DOD method asks assessors to rate the degree of difference between
pairs of samples on an ordinal scale. Same-pairs and different-pairs are
presented, and the responses are used to estimate d' (d-prime).

The model estimates:
- d-prime: the discriminability parameter
- tau: boundary parameters defining the rating scale cutoffs

References
----------
Christensen, R.H.B. et al. (2011). A Thurstonian model for the
degree of difference method with multiple response categories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize
from scipy.stats import norm

from senspy.utils.stats import normal_pvalue


@dataclass
class DODControl:
    """Control parameters for DOD model fitting.

    Attributes
    ----------
    grad_tol : float
        Tolerance for gradient convergence check. Default is 1e-4.
    integer_tol : float
        Tolerance for checking if counts are integers. Default is 1e-8.
    get_vcov : bool
        Whether to compute variance-covariance matrix. Default is True.
    get_grad : bool
        Whether to compute and check gradient. Default is True.
    test_args : bool
        Whether to test argument validity. Default is True.
    do_warn : bool
        Whether to emit warnings. Default is True.
    opt_options : dict
        Additional options passed to scipy.optimize.minimize. Default is empty dict.
    """

    grad_tol: float = 1e-4
    integer_tol: float = 1e-8
    get_vcov: bool = True
    get_grad: bool = True
    test_args: bool = True
    do_warn: bool = True
    opt_options: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.grad_tol <= 0 or not np.isfinite(self.grad_tol):
            raise ValueError("grad_tol must be positive and finite")
        if self.integer_tol < 0 or not np.isfinite(self.integer_tol):
            raise ValueError("integer_tol must be non-negative and finite")


@dataclass
class DODFitResult:
    """Result from dod_fit function.

    Attributes
    ----------
    d_prime : float
        Estimated d-prime (discriminability).
    tau : NDArray
        Estimated boundary parameters.
    log_lik : float
        Log-likelihood at the estimates.
    coefficients : NDArray
        Combined coefficient vector [tau1, tau2, ..., d_prime].
    vcov : NDArray | None
        Variance-covariance matrix of coefficients.
    gradient : NDArray | None
        Gradient at convergence.
    hessian : NDArray | None
        Hessian matrix at convergence.
    data : NDArray
        The 2 x k data matrix (same-pairs, diff-pairs).
    convergence : int
        Optimizer convergence code (0 = success).
    """

    d_prime: float
    tau: NDArray
    log_lik: float
    coefficients: NDArray
    vcov: NDArray | None
    gradient: NDArray | None
    hessian: NDArray | None
    data: NDArray
    convergence: int


@dataclass
class DODResult:
    """Result from dod function.

    Attributes
    ----------
    d_prime : float
        Estimated d-prime (discriminability).
    tau : NDArray
        Estimated boundary parameters.
    log_lik : float
        Log-likelihood at the estimates.
    se_d_prime : float
        Standard error of d-prime estimate.
    se_tau : NDArray
        Standard errors of tau estimates.
    conf_int : tuple[float, float]
        Confidence interval for d-prime.
    conf_level : float
        Confidence level used for interval.
    conf_method : str
        Method used for confidence interval ("profile likelihood" or "Wald").
    stat_value : float
        Value of the test statistic.
    p_value : float
        P-value for the hypothesis test.
    statistic : str
        Type of test statistic used.
    alternative : str
        Alternative hypothesis ("greater", "less", "two.sided").
    d_prime0 : float
        Null hypothesis value for d-prime.
    data : NDArray
        The 2 x k data matrix (same-pairs, diff-pairs).
    vcov : NDArray | None
        Variance-covariance matrix of coefficients.
    coefficients : NDArray
        Combined coefficient vector [tau1, tau2, ..., d_prime].
    convergence : int
        Optimizer convergence code (0 = success).
    """

    d_prime: float
    tau: NDArray
    log_lik: float
    se_d_prime: float
    se_tau: NDArray
    conf_int: tuple[float, float]
    conf_level: float
    conf_method: str
    stat_value: float
    p_value: float
    statistic: str
    alternative: str
    d_prime0: float
    data: NDArray
    vcov: NDArray | None
    coefficients: NDArray
    convergence: int


def par2prob_dod(tau: NDArray, d_prime: float) -> NDArray:
    """Convert DOD parameters to probability matrix.

    Computes the probability of each response category for same-pairs
    and different-pairs given the boundary parameters tau and d-prime.

    Parameters
    ----------
    tau : NDArray
        Boundary parameters (must be positive and increasing).
    d_prime : float
        Discriminability parameter (must be non-negative).

    Returns
    -------
    NDArray
        A 2 x (len(tau)+1) matrix where:
        - Row 0: probabilities for same-pairs
        - Row 1: probabilities for different-pairs

    Raises
    ------
    ValueError
        If parameters are invalid.
    """
    tau = np.asarray(tau, dtype=float)
    if d_prime < 0:
        raise ValueError("d_prime must be non-negative")
    if len(tau) < 1:
        raise ValueError("tau must have at least one element")
    if np.any(tau <= 0):
        raise ValueError("tau values must be positive")
    if len(tau) > 1 and np.any(np.diff(tau) <= 0):
        raise ValueError("tau values must be strictly increasing")

    return _par2prob_dod_internal(tau, d_prime)


def _par2prob_dod_internal(tau: NDArray, d_prime: float) -> NDArray:
    """Internal function to compute DOD probabilities without validation."""
    sqrt2 = np.sqrt(2)

    # Cumulative probabilities
    gamma_same = 2 * norm.cdf(tau / sqrt2) - 1
    gamma_diff = norm.cdf((tau - d_prime) / sqrt2) - norm.cdf(
        (-tau - d_prime) / sqrt2
    )

    # Non-cumulative probabilities
    p_same = np.concatenate([gamma_same, [1.0]]) - np.concatenate([[0.0], gamma_same])
    p_diff = np.concatenate([gamma_diff, [1.0]]) - np.concatenate([[0.0], gamma_diff])

    # Normalize to ensure sum = 1
    p_same = p_same / np.sum(p_same)
    p_diff = p_diff / np.sum(p_diff)

    return np.vstack([p_same, p_diff])


def _dod_nll_internal(
    tau: NDArray, d_prime: float, same: NDArray, diff: NDArray
) -> float:
    """Compute negative log-likelihood for DOD model.

    Parameters
    ----------
    tau : NDArray
        Boundary parameters.
    d_prime : float
        Discriminability parameter.
    same : NDArray
        Counts for same-pairs in each category.
    diff : NDArray
        Counts for different-pairs in each category.

    Returns
    -------
    float
        Negative log-likelihood. Returns inf if parameters are inadmissible.
    """
    prob = _par2prob_dod_internal(tau, d_prime)

    if not np.all(np.isfinite(prob)) or np.any(prob <= 0):
        return np.inf

    nll = -np.sum(same * np.log(prob[0, :])) - np.sum(diff * np.log(prob[1, :]))
    return nll


def _dod_nll_all(par: NDArray, same: NDArray, diff: NDArray) -> float:
    """NLL expressed as function of all parameters c(tau, d_prime)."""
    npar = len(par)
    tau = par[: npar - 1]
    d_prime = par[npar - 1]
    return _dod_nll_internal(tau, d_prime, same, diff)


def _dod_null_tau_internal(same: NDArray, diff: NDArray) -> NDArray:
    """Compute tau for DOD null model (d_prime=0)."""
    all_data = same + diff
    all_data = all_data[::-1]  # Reverse
    nlev = len(all_data)
    cum_data = np.cumsum(all_data)
    total = np.sum(all_data)

    tau = np.array(
        [
            -np.sqrt(2) * norm.ppf(cum_data[lev] / (2 * total))
            for lev in range(nlev - 1)
        ]
    )
    return tau[::-1]  # Reverse back


def _init_tau(ncat: int = 4) -> NDArray:
    """Initialize tau values for optimization."""
    if ncat < 2:
        raise ValueError("ncat must be at least 2")
    return np.array([1.0] + [3.0 / (ncat - 1)] * (ncat - 2))


def _validate_dod_data(
    same: ArrayLike, diff: ArrayLike, integer_tol: float = 1e-8
) -> tuple[NDArray, NDArray]:
    """Validate and clean DOD data.

    Parameters
    ----------
    same : ArrayLike
        Counts for same-pairs.
    diff : ArrayLike
        Counts for different-pairs.
    integer_tol : float
        Tolerance for integer check.

    Returns
    -------
    tuple[NDArray, NDArray]
        Cleaned same and diff arrays.

    Raises
    ------
    ValueError
        If data is invalid.
    """
    same = np.asarray(same, dtype=float)
    diff = np.asarray(diff, dtype=float)

    if len(same) != len(diff):
        raise ValueError("same and diff must have the same length")
    if np.any(same < 0) or np.any(diff < 0):
        raise ValueError("Counts must be non-negative")
    if len(same) < 2:
        raise ValueError("Need at least 2 response categories")

    # Remove empty categories (where both same and diff are 0)
    total = same + diff
    mask = total > 0
    same = same[mask]
    diff = diff[mask]

    if len(same) < 2:
        raise ValueError("Need counts in more than one response category")

    # Check for near-integer values
    if np.any(np.abs(same - np.round(same)) > integer_tol):
        import warnings

        warnings.warn("non-integer counts in 'same'")
    if np.any(np.abs(diff - np.round(diff)) > integer_tol):
        import warnings

        warnings.warn("non-integer counts in 'diff'")

    return same, diff


def optimal_tau(
    d_prime: float,
    d_prime0: float = 0.0,
    ncat: int = 3,
    method: Literal["equi_prob", "LR_max", "se_min"] = "equi_prob",
    tau_start: NDArray | None = None,
    equi_tol: float = 1e-4,
    grad_tol: float = 1e-2,
    do_warn: bool = True,
) -> dict:
    """Estimate optimal boundary parameters tau.

    Parameters
    ----------
    d_prime : float
        True d-prime value (must be non-negative).
    d_prime0 : float
        Null hypothesis d-prime value. Default is 0.
    ncat : int
        Number of response categories. Default is 3.
    method : str
        Optimization criterion:
        - "equi_prob": Equal category probabilities (averaged over same/diff)
        - "LR_max": Maximum likelihood ratio statistic
        - "se_min": Minimum standard error of d-prime
    tau_start : NDArray | None
        Starting values for tau. Default uses init_tau().
    equi_tol : float
        Tolerance for equi_prob method. Default is 1e-4.
    grad_tol : float
        Gradient tolerance. Default is 1e-2.
    do_warn : bool
        Whether to emit warnings. Default is True.

    Returns
    -------
    dict
        Dictionary with keys: tau, prob, tau_start, gradient, method
    """
    if d_prime < 0:
        raise ValueError("d_prime must be non-negative")
    if ncat < 2:
        raise ValueError("ncat must be at least 2")

    ncat = int(round(ncat))

    def tau2tpar(tau):
        """Convert tau to optimization parameterization."""
        return np.concatenate([[tau[0]], np.diff(tau)])

    def tpar2tau(tpar):
        """Convert optimization parameterization to tau."""
        return np.cumsum(tpar)

    # Define objective functions
    def equi_prob_obj(tpar):
        """Objective for equal category probabilities."""
        tau = tpar2tau(tpar)
        prob = _par2prob_dod_internal(tau, d_prime)
        avg_prob = np.sum(prob, axis=0) / 2
        target = 1.0 / ncat
        return 1 + np.sum(((avg_prob - target) * 1e3) ** 2)

    def se_min_obj(tpar):
        """Objective for minimum standard error of d-prime."""
        if not np.all(np.isfinite(tpar)):
            return np.inf
        tau = tpar2tau(tpar)
        if np.any(tau <= 0):
            return np.inf

        # Generate "data" from probabilities
        data = _par2prob_dod_internal(tau, d_prime) * 100

        # Compute Hessian numerically
        par = np.concatenate([tau, [d_prime]])

        def nll_func(p):
            return _dod_nll_all(p, data[0, :], data[1, :])

        try:
            h = _numerical_hessian(nll_func, par)
            if not np.all(np.isfinite(h)):
                return np.inf
            vcov = np.linalg.inv(h)
            if not np.all(np.isfinite(vcov)):
                return np.inf
            return np.sqrt(vcov[-1, -1])
        except (np.linalg.LinAlgError, ValueError):
            return np.inf

    def lr_max_obj(tpar):
        """Objective for maximum LR statistic (negative)."""
        tau = tpar2tau(tpar)
        if np.any(tau <= 0):
            return np.inf

        # Limiting distribution of data given c(tau, d_prime)
        data = _par2prob_dod_internal(tau, d_prime) * 100

        # Log-lik at d_prime and at d_prime0
        log_lik = -_dod_nll_internal(tau, d_prime, data[0, :], data[1, :])

        if np.isclose(d_prime0, 0.0):
            tau0 = _dod_null_tau_internal(data[0, :], data[1, :])
            log_lik0 = -_dod_nll_internal(tau0, 0.0, data[0, :], data[1, :])
        else:
            # Need to fit model at d_prime0
            fit0 = dod_fit(
                data[0, :],
                data[1, :],
                d_prime=d_prime0,
                control=DODControl(
                    test_args=False, do_warn=False, get_grad=False, get_vcov=False
                ),
            )
            log_lik0 = fit0.log_lik

        return -(log_lik - log_lik0)  # Negative LR/2 statistic

    # Select objective function
    obj_funcs = {
        "equi_prob": equi_prob_obj,
        "LR_max": lr_max_obj,
        "se_min": se_min_obj,
    }
    objfun = obj_funcs[method]

    # Get starting values
    if tau_start is not None:
        tau_start = np.asarray(tau_start)
        if len(tau_start) != ncat - 1:
            raise ValueError(f"tau_start must have length {ncat - 1}")
        if np.any(tau_start <= 0):
            raise ValueError("tau_start values must be positive")
        if len(tau_start) > 1 and np.any(np.diff(tau_start) <= 0):
            raise ValueError("tau_start must be strictly increasing")
        start = tau2tpar(tau_start)
    else:
        start = _init_tau(ncat)

    # Optimize
    result = optimize.minimize(
        objfun,
        start,
        method="L-BFGS-B",
        bounds=[(1e-4, None)] * len(start),
    )

    tau = tpar2tau(result.x)
    prob = _par2prob_dod_internal(tau, d_prime)

    # Check convergence for equi_prob method
    if method == "equi_prob":
        avg_prob = np.sum(prob, axis=0) / 2
        diffs = np.abs(avg_prob - 1.0 / ncat)
        if np.max(diffs) > equi_tol and do_warn:
            import warnings

            warnings.warn(
                f"Estimation of tau failed with max(diffs) = {np.max(diffs):.2g} "
                f"(equi_tol = {equi_tol:.2g})"
            )

    # Compute gradient
    grad = _numerical_gradient(objfun, result.x)
    if np.max(np.abs(grad)) > grad_tol and do_warn:
        import warnings

        warnings.warn(
            f"Estimation of tau failed with max(gradient) = {np.max(np.abs(grad)):.2g} "
            f"(grad_tol = {grad_tol:.2g})"
        )

    return {
        "tau": tau,
        "prob": prob,
        "tau_start": np.cumsum(start),
        "gradient": grad,
        "method": method,
    }


def _numerical_gradient(func, x, eps: float = 1e-6) -> NDArray:
    """Compute numerical gradient using central differences."""
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        grad[i] = (func(x_plus) - func(x_minus)) / (2 * eps)
    return grad


def _numerical_hessian(func, x, eps: float = 1e-5) -> NDArray:
    """Compute numerical Hessian using central differences."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    hess = np.zeros((n, n))

    f0 = func(x)

    for i in range(n):
        for j in range(i, n):
            x_pp = x.copy()
            x_pm = x.copy()
            x_mp = x.copy()
            x_mm = x.copy()

            x_pp[i] += eps
            x_pp[j] += eps
            x_pm[i] += eps
            x_pm[j] -= eps
            x_mp[i] -= eps
            x_mp[j] += eps
            x_mm[i] -= eps
            x_mm[j] -= eps

            hess[i, j] = (func(x_pp) - func(x_pm) - func(x_mp) + func(x_mm)) / (
                4 * eps * eps
            )
            hess[j, i] = hess[i, j]

    return hess


def dod_fit(
    same: ArrayLike,
    diff: ArrayLike,
    tau: NDArray | None = None,
    d_prime: float | None = None,
    control: DODControl | None = None,
) -> DODFitResult:
    """Fit DOD model (low-level function).

    This is the lower-level fitting function. Use `dod()` for the full analysis
    including hypothesis testing.

    Parameters
    ----------
    same : ArrayLike
        Counts for same-pairs in each response category.
    diff : ArrayLike
        Counts for different-pairs in each response category.
    tau : NDArray | None
        Fixed boundary parameters. If None, estimated from data.
    d_prime : float | None
        Fixed d-prime value. If None, estimated from data.
    control : DODControl | None
        Control parameters. If None, uses defaults.

    Returns
    -------
    DODFitResult
        Fitting results including estimates and diagnostics.
    """
    if control is None:
        control = DODControl()

    # Validate data
    if control.test_args:
        same, diff = _validate_dod_data(same, diff, control.integer_tol)
    else:
        same = np.asarray(same, dtype=float)
        diff = np.asarray(diff, dtype=float)

    nlev = len(same)
    data = np.vstack([same, diff])

    # Initialize result variables
    vcov = None
    gradient = None
    hessian = None
    convergence = 0

    # CASE 1: d_prime = 0 and tau = None
    if tau is None and d_prime is not None and np.isclose(d_prime, 0.0):
        est_tau = _dod_null_tau_internal(same, diff)
        est_d_prime = d_prime

    # CASE 2: Need to optimize tau, d_prime, or both
    elif tau is None or d_prime is None:
        if d_prime is None and tau is None:
            # Estimate both tau and d_prime
            start = np.concatenate([np.cumsum(_init_tau(nlev)), [1.0]])
            par_names = [f"tau{i+1}" for i in range(nlev - 1)] + ["d_prime"]
            npar = len(start)

            def objfun(par):
                return _dod_nll_internal(par[: npar - 1], par[npar - 1], same, diff)

            case = "both"

        elif tau is None and d_prime is not None:
            # Estimate tau only
            start = np.cumsum(_init_tau(nlev))
            par_names = [f"tau{i+1}" for i in range(nlev - 1)]

            def objfun(par):
                return _dod_nll_internal(par, d_prime, same, diff)

            case = "tau"

        else:  # tau is not None and d_prime is None
            # Estimate d_prime only
            start = np.array([1.0])
            par_names = ["d_prime"]

            def objfun(par):
                return _dod_nll_internal(tau, par[0], same, diff)

            case = "d_prime"

        # Optimization - use small positive lower bound to avoid numerical issues
        result = optimize.minimize(
            objfun,
            start,
            method="L-BFGS-B",
            bounds=[(1e-6, None)] * len(start),
            options=control.opt_options,
        )

        par = result.x
        convergence = 0 if result.success else result.status

        # Compute gradient and Hessian if requested
        if np.all(par > 1e-2) and control.get_grad:
            gradient = _numerical_gradient(objfun, par)

            if not np.all(np.isfinite(gradient)):
                if control.do_warn:
                    import warnings

                    warnings.warn("Cannot assess convergence: non-finite gradient")
            else:
                if np.max(np.abs(gradient)) > control.grad_tol and control.do_warn:
                    import warnings

                    warnings.warn(
                        f"Estimation failed with max(gradient) = {np.max(np.abs(gradient)):.2g} "
                        f"(grad_tol = {control.grad_tol:.2g})"
                    )

                if control.get_vcov:
                    try:
                        hessian = _numerical_hessian(objfun, par)
                        if not np.all(np.isfinite(hessian)):
                            if control.do_warn:
                                import warnings

                                warnings.warn("unable to compute Hessian")
                        else:
                            # Check positive definiteness via Cholesky
                            try:
                                np.linalg.cholesky(hessian)
                                vcov = np.linalg.inv(hessian)
                            except np.linalg.LinAlgError:
                                if control.do_warn:
                                    import warnings

                                    warnings.warn(
                                        "Model is ill-defined and may not have converged"
                                    )
                    except Exception:
                        if control.do_warn:
                            import warnings

                            warnings.warn("unable to compute Hessian")

        # Extract parameters based on case
        if case == "both":
            est_tau = par[: npar - 1]
            est_d_prime = par[npar - 1]
        elif case == "tau":
            est_tau = par
            est_d_prime = d_prime
        else:  # case == "d_prime"
            est_tau = tau
            est_d_prime = par[0]

    # CASE 3: Both tau and d_prime provided - just evaluate likelihood
    else:
        est_tau = np.asarray(tau, dtype=float)
        est_d_prime = d_prime

    # Compute log-likelihood
    log_lik = -_dod_nll_internal(est_tau, est_d_prime, same, diff)

    # Build coefficients vector
    coefficients = np.concatenate([est_tau, [est_d_prime]])

    return DODFitResult(
        d_prime=est_d_prime,
        tau=est_tau,
        log_lik=log_lik,
        coefficients=coefficients,
        vcov=vcov,
        gradient=gradient,
        hessian=hessian,
        data=data,
        convergence=convergence,
    )


def _profile_ci_dod(
    same: NDArray,
    diff: NDArray,
    fit: DODFitResult,
    level: float = 0.95,
    tol_d_prime: float = 1e-3,
) -> tuple[float, float]:
    """Compute profile likelihood confidence interval for d-prime."""
    ctrl = DODControl(get_vcov=False, get_grad=False, do_warn=False, test_args=False)
    alpha = 1 - level
    lim = norm.ppf(1 - alpha / 2)

    mle = fit.d_prime
    log_lik_max = fit.log_lik

    if not np.isfinite(mle):
        return (np.nan, np.nan)

    def lroot_fun(d_prime_val, target):
        """Profile likelihood root statistic minus target."""
        ll = dod_fit(same, diff, d_prime=d_prime_val, control=ctrl).log_lik
        lr = 2 * (log_lik_max - ll)
        lr = max(0, lr)  # Handle numerical issues
        lroot_val = np.sign(mle - d_prime_val) * np.sqrt(lr)
        return lroot_val - target

    # Find lower bound
    if mle < tol_d_prime:
        lower = 0.0
    else:
        # Check if 0 is in CI
        lr0 = 2 * (log_lik_max + _dod_nll_internal(_dod_null_tau_internal(same, diff), 0.0, same, diff))
        if lr0 < lim**2:
            lower = 0.0
        else:
            # Search for lower bound
            try:
                result = optimize.brentq(lroot_fun, 0, mle, args=(lim,))
                lower = result
            except ValueError:
                lower = 0.0

    # Find upper bound
    if mle == np.inf:
        upper = np.nan
    else:
        # Search for upper bound
        upper_search = 10.0
        if mle > upper_search:
            upper = np.nan
        else:
            ll_upper = dod_fit(same, diff, d_prime=upper_search, control=ctrl).log_lik
            lr_upper = 2 * (log_lik_max - ll_upper)
            if lr_upper < lim**2:
                upper = np.nan
            else:
                try:
                    result = optimize.brentq(lroot_fun, mle, upper_search, args=(-lim,))
                    upper = result
                except ValueError:
                    upper = np.nan

    return (lower, upper)


def _wald_ci_dod(
    fit: DODFitResult, level: float = 0.95
) -> tuple[float, float]:
    """Compute Wald confidence interval for d-prime."""
    alpha = 1 - level
    z = norm.ppf(1 - alpha / 2)

    if fit.vcov is None or not np.all(np.isfinite(fit.vcov)):
        return (np.nan, np.nan)

    se = np.sqrt(fit.vcov[-1, -1])
    if not np.isfinite(se):
        return (np.nan, np.nan)

    lower = max(0, fit.d_prime - z * se)
    upper = fit.d_prime + z * se

    return (lower, upper)


def dod(
    same: ArrayLike,
    diff: ArrayLike,
    d_prime0: float = 0.0,
    conf_level: float = 0.95,
    statistic: Literal["likelihood", "Pearson", "Wilcoxon", "Wald"] = "likelihood",
    alternative: Literal[
        "difference", "similarity", "two.sided", "less", "greater"
    ] = "difference",
    control: DODControl | None = None,
) -> DODResult:
    """Fit DOD model and perform hypothesis test.

    The Degree-of-Difference (DOD) model is a Thurstonian model for sensory
    discrimination where assessors rate the degree of difference between
    pairs of samples on an ordinal scale.

    Parameters
    ----------
    same : ArrayLike
        Counts for same-pairs in each response category.
    diff : ArrayLike
        Counts for different-pairs in each response category.
    d_prime0 : float
        Null hypothesis value for d-prime. Default is 0.
    conf_level : float
        Confidence level for interval. Default is 0.95.
    statistic : str
        Test statistic to use:
        - "likelihood": Likelihood ratio test (default)
        - "Pearson": Pearson chi-square test
        - "Wilcoxon": Wilcoxon rank-sum test (only for d_prime0=0)
        - "Wald": Wald test
    alternative : str
        Alternative hypothesis:
        - "difference" or "greater": d-prime > d_prime0 (default)
        - "similarity" or "less": d-prime < d_prime0
        - "two.sided": d-prime != d_prime0
    control : DODControl | None
        Control parameters. If None, uses defaults.

    Returns
    -------
    DODResult
        Complete analysis results including estimates, confidence interval,
        and hypothesis test.

    Raises
    ------
    ValueError
        If arguments are invalid or incompatible.

    Examples
    --------
    >>> same = [20, 30, 25, 25]
    >>> diff = [10, 20, 35, 35]
    >>> result = dod(same, diff)
    >>> print(f"d-prime: {result.d_prime:.3f}")
    >>> print(f"p-value: {result.p_value:.4f}")
    """
    if control is None:
        control = DODControl()

    # Map alternative names
    if alternative == "difference":
        alternative = "greater"
    elif alternative == "similarity":
        alternative = "less"

    # Validate arguments
    if control.test_args:
        if not (0 < conf_level < 1):
            raise ValueError("conf_level must be between 0 and 1")
        if d_prime0 < 0:
            raise ValueError("d_prime0 must be non-negative")
        if np.isclose(d_prime0, 0.0) and alternative not in ["difference", "greater"]:
            raise ValueError(
                "'alternative' has to be 'difference' or 'greater' if 'd_prime0' is 0"
            )
        if not np.isclose(d_prime0, 0.0) and statistic == "Wilcoxon":
            raise ValueError("Wilcoxon statistic only available with d_prime0 = 0")

    # Fit the model
    fit = dod_fit(same, diff, control=control)

    # Get cleaned data
    same_arr = fit.data[0, :]
    diff_arr = fit.data[1, :]
    nlev = len(same_arr)
    npar = len(fit.coefficients)

    # Compute standard errors
    se_d_prime = np.nan
    se_tau = np.full(nlev - 1, np.nan)
    if fit.d_prime < 0.01 and control.get_vcov and control.do_warn:
        import warnings

        warnings.warn("d_prime < 0.01: standard errors are unavailable")
    if fit.vcov is not None and np.all(np.isfinite(fit.vcov)):
        se_all = np.sqrt(np.diag(fit.vcov))
        se_tau = se_all[:-1]
        se_d_prime = se_all[-1]

    # Compute confidence interval
    if statistic == "Wald":
        conf_int = _wald_ci_dod(fit, conf_level)
        conf_method = "Wald"
    else:
        conf_int = _profile_ci_dod(same_arr, diff_arr, fit, conf_level)
        conf_method = "profile likelihood"

    # Compute test statistic and p-value
    stat_value = np.nan
    p_value = np.nan

    if statistic == "Wilcoxon":
        # Wilcoxon rank-sum test
        from scipy.stats import mannwhitneyu

        # Create pseudo-data for Wilcoxon test
        # Expand rating categories by their counts for each pair type
        diff_pair_ratings = np.repeat(np.arange(1, nlev + 1), diff_arr.astype(int))
        same_pair_ratings = np.repeat(np.arange(1, nlev + 1), same_arr.astype(int))

        if alternative == "two.sided":
            alt_mwu = "two-sided"
        elif alternative == "greater":
            alt_mwu = "greater"
        else:
            alt_mwu = "less"

        try:
            # Test if diff-pair ratings are stochastically greater than same-pair ratings
            mwu_result = mannwhitneyu(diff_pair_ratings, same_pair_ratings, alternative=alt_mwu)
            stat_value = mwu_result.statistic
            p_value = mwu_result.pvalue
        except ValueError:
            pass

    elif statistic == "Wald":
        if np.isfinite(se_d_prime):
            stat_value = (fit.d_prime - d_prime0) / se_d_prime
            p_value = normal_pvalue(stat_value, alternative)

    elif statistic in ["likelihood", "Pearson"]:
        # Fit model under null hypothesis
        ctrl0 = DODControl(
            get_vcov=False, get_grad=False, test_args=False, do_warn=False
        )
        fit0 = dod_fit(same_arr, diff_arr, d_prime=d_prime0, control=ctrl0)
        log_lik0 = fit0.log_lik

        # Check convergence
        if fit.log_lik < log_lik0 and abs(fit.log_lik - log_lik0) > 1e-6:
            if control.do_warn:
                import warnings

                warnings.warn(
                    "Estimation of DOD model failed: likelihood did not increase"
                )

        if statistic == "likelihood":
            LR = 2 * (fit.log_lik - log_lik0)
            if LR < 0 and abs(LR) < 1e-4:
                LR = 0
            if LR >= 0:
                stat_value = np.sign(fit.d_prime - d_prime0) * np.sqrt(LR)
                p_value = normal_pvalue(stat_value, alternative)

        else:  # Pearson
            # Expected frequencies under alternative
            freq = _par2prob_dod_internal(fit.tau, fit.d_prime) * np.array(
                [[np.sum(same_arr)], [np.sum(diff_arr)]]
            )
            # Expected frequencies under null
            freq0 = _par2prob_dod_internal(fit0.tau, fit0.d_prime) * np.array(
                [[np.sum(same_arr)], [np.sum(diff_arr)]]
            )

            X2 = np.sum((freq - freq0) ** 2 / freq0)
            stat_value = np.sign(fit.d_prime - d_prime0) * np.sqrt(X2)
            p_value = normal_pvalue(stat_value, alternative)

    return DODResult(
        d_prime=fit.d_prime,
        tau=fit.tau,
        log_lik=fit.log_lik,
        se_d_prime=se_d_prime,
        se_tau=se_tau,
        conf_int=conf_int,
        conf_level=conf_level,
        conf_method=conf_method,
        stat_value=stat_value,
        p_value=p_value,
        statistic=statistic,
        alternative=alternative,
        d_prime0=d_prime0,
        data=fit.data,
        vcov=fit.vcov,
        coefficients=fit.coefficients,
        convergence=fit.convergence,
    )


def dod_sim(
    d_prime: float,
    ncat: int = 4,
    sample_size: int | tuple[int, int] = 100,
    method_tau: Literal["equi_prob", "LR_max", "se_min", "user_defined"] = "equi_prob",
    tau: NDArray | None = None,
    d_prime0: float = 0.0,
    random_state: int | np.random.Generator | None = None,
) -> NDArray:
    """Simulate DOD data.

    Parameters
    ----------
    d_prime : float
        True d-prime value (must be non-negative).
    ncat : int
        Number of response categories. Default is 4.
    sample_size : int | tuple[int, int]
        Sample size. If int, same size for both same-pairs and diff-pairs.
        If tuple, (same_size, diff_size). Default is 100.
    method_tau : str
        Method for determining tau values:
        - "equi_prob": Equal category probabilities
        - "LR_max": Maximum likelihood ratio
        - "se_min": Minimum standard error
        - "user_defined": Use provided tau values
    tau : NDArray | None
        Boundary parameters (required if method_tau="user_defined").
    d_prime0 : float
        Null hypothesis d-prime for optimal_tau. Default is 0.
    random_state : int | np.random.Generator | None
        Random state for reproducibility.

    Returns
    -------
    NDArray
        A 2 x ncat matrix with row 0 = same-pairs, row 1 = diff-pairs.

    Examples
    --------
    >>> np.random.seed(42)
    >>> data = dod_sim(d_prime=1.0, sample_size=100)
    >>> print(data)
    """
    if d_prime < 0:
        raise ValueError("d_prime must be non-negative")
    if d_prime0 < 0:
        raise ValueError("d_prime0 must be non-negative")

    # Handle random state
    if random_state is None:
        rng = np.random.default_rng()
    elif isinstance(random_state, int):
        rng = np.random.default_rng(random_state)
    else:
        rng = random_state

    # Get tau
    if method_tau != "user_defined":
        if tau is not None:
            import warnings

            warnings.warn(
                f"'tau' is ignored when method_tau != 'user_defined' "
                f"(method_tau was '{method_tau}')"
            )
        # Map method names
        method_map = {
            "equi_prob": "equi_prob",
            "LR_max": "LR_max",
            "se_min": "se_min",
        }
        tau_result = optimal_tau(
            d_prime=d_prime,
            d_prime0=d_prime0,
            ncat=ncat,
            method=method_map[method_tau],
            do_warn=False,
        )
        tau = tau_result["tau"]
    else:
        if tau is None:
            raise ValueError("tau must be provided when method_tau='user_defined'")
        tau = np.asarray(tau, dtype=float)

    # Handle sample size
    if isinstance(sample_size, int):
        size_same = size_diff = sample_size
    else:
        size_same, size_diff = sample_size

    # Get probabilities and simulate data
    prob = par2prob_dod(tau, d_prime)

    same_counts = rng.multinomial(size_same, prob[0, :])
    diff_counts = rng.multinomial(size_diff, prob[1, :])

    return np.vstack([same_counts, diff_counts])


@dataclass
class DODPowerResult:
    """Result from dod_power function.

    Attributes
    ----------
    power : float
        Estimated power (proportion of simulations with p < alpha).
    se_power : float
        Standard error of the power estimate.
    n_used : int
        Number of simulations used (excluding failures).
    d_primeA : float
        True d-prime value (alternative hypothesis).
    d_prime0 : float
        Null hypothesis d-prime.
    sample_size : tuple[int, int]
        Sample sizes (same_pairs, diff_pairs).
    nsim : int
        Total number of simulations requested.
    alpha : float
        Significance level.
    statistic : str
        Test statistic used.
    alternative : str
        Alternative hypothesis.
    tau : NDArray
        Boundary parameters used for simulation.
    """

    power: float
    se_power: float
    n_used: int
    d_primeA: float
    d_prime0: float
    sample_size: tuple[int, int]
    nsim: int
    alpha: float
    statistic: str
    alternative: str
    tau: NDArray


def dod_power(
    d_primeA: float,
    d_prime0: float = 0.0,
    ncat: int = 4,
    sample_size: int | tuple[int, int] = 100,
    nsim: int = 1000,
    alpha: float = 0.05,
    method_tau: Literal["equi_prob", "LR_max", "se_min", "user_defined"] = "LR_max",
    statistic: Literal["likelihood", "Wilcoxon", "Pearson", "Wald"] = "likelihood",
    alternative: Literal[
        "difference", "similarity", "two.sided", "less", "greater"
    ] = "difference",
    tau: NDArray | None = None,
    random_state: int | np.random.Generator | None = None,
) -> DODPowerResult:
    """Compute power for DOD discrimination test via simulation.

    This function estimates the power of a DOD (Degree-of-Difference) test
    by simulating data under the alternative hypothesis and computing the
    proportion of simulations that would reject the null hypothesis.

    Parameters
    ----------
    d_primeA : float
        True d-prime value (alternative hypothesis).
    d_prime0 : float
        Null hypothesis d-prime value. Default is 0.
    ncat : int
        Number of response categories. Default is 4.
    sample_size : int | tuple[int, int]
        Sample size. If int, same size for both same-pairs and diff-pairs.
        If tuple, (same_size, diff_size). Default is 100.
    nsim : int
        Number of simulations. Default is 1000.
    alpha : float
        Significance level. Default is 0.05.
    method_tau : str
        Method for determining tau values:
        - "LR_max": Maximum likelihood ratio (default)
        - "equi_prob": Equal category probabilities
        - "se_min": Minimum standard error
        - "user_defined": Use provided tau values
    statistic : str
        Test statistic:
        - "likelihood": Likelihood ratio test (default)
        - "Pearson": Pearson chi-square
        - "Wilcoxon": Wilcoxon rank-sum test (requires d_prime0=0)
        - "Wald": Wald test
    alternative : str
        Alternative hypothesis:
        - "difference" or "greater": d-prime > d_prime0 (default)
        - "similarity" or "less": d-prime < d_prime0
        - "two.sided": d-prime != d_prime0
    tau : NDArray | None
        Boundary parameters (required if method_tau="user_defined").
    random_state : int | np.random.Generator | None
        Random state for reproducibility.

    Returns
    -------
    DODPowerResult
        Power analysis results including power estimate and standard error.

    Raises
    ------
    ValueError
        If inputs are invalid or incompatible.

    Examples
    --------
    >>> result = dod_power(d_primeA=1.0, sample_size=100, nsim=500)
    >>> print(f"Power: {result.power:.3f} (SE: {result.se_power:.3f})")

    >>> # With specific tau values
    >>> result = dod_power(
    ...     d_primeA=1.5,
    ...     method_tau="user_defined",
    ...     tau=np.array([0.5, 1.0, 1.5])
    ... )
    """
    # Validate inputs
    if not (isinstance(d_primeA, (int, float)) and d_primeA >= 0 and np.isfinite(d_primeA)):
        raise ValueError("d_primeA must be a finite non-negative number")
    if not (isinstance(d_prime0, (int, float)) and d_prime0 >= 0 and np.isfinite(d_prime0)):
        raise ValueError("d_prime0 must be a finite non-negative number")
    if nsim <= 0 or not isinstance(nsim, (int, float)):
        raise ValueError("nsim must be a positive number")
    if not (0 <= alpha <= 1):
        raise ValueError("alpha must be between 0 and 1")

    nsim = int(round(nsim))

    # Handle sample size
    if isinstance(sample_size, int):
        size = (sample_size, sample_size)
    else:
        size = (int(round(sample_size[0])), int(round(sample_size[1])))

    # Map alternative names
    alt = alternative
    if alt == "difference":
        alt = "greater"
    elif alt == "similarity":
        alt = "less"

    # Check d_primeA, d_prime0 consistency with alternative
    if alt == "greater" and d_primeA < d_prime0:
        raise ValueError(
            f"Need d_primeA >= d_prime0 when alternative is '{alternative}'"
        )
    if alt == "less" and d_primeA > d_prime0:
        raise ValueError(
            f"Need d_primeA <= d_prime0 when alternative is '{alternative}'"
        )

    # Wilcoxon requires d_prime0 = 0
    if statistic == "Wilcoxon" and not np.isclose(d_prime0, 0.0):
        raise ValueError("Wilcoxon statistic only available with d_prime0=0")

    # Get tau values
    if method_tau != "user_defined":
        method_map = {
            "equi_prob": "equi_prob",
            "LR_max": "LR_max",
            "se_min": "se_min",
        }
        tau_result = optimal_tau(
            d_prime=d_primeA,
            d_prime0=d_prime0,
            ncat=ncat,
            method=method_map[method_tau],
            do_warn=False,
        )
        tau_arr = tau_result["tau"]
    else:
        if tau is None:
            raise ValueError("tau must be provided when method_tau='user_defined'")
        tau_arr = np.asarray(tau, dtype=float)

    # Handle random state
    if random_state is None:
        rng = np.random.default_rng()
    elif isinstance(random_state, int):
        rng = np.random.default_rng(random_state)
    else:
        rng = random_state

    # Run simulations
    pvals = np.full(nsim, np.nan)
    ctrl = DODControl(get_vcov=statistic == "Wald", get_grad=False, do_warn=False)

    for i in range(nsim):
        # Simulate data under alternative
        data = dod_sim(
            d_prime=d_primeA,
            sample_size=size,
            method_tau="user_defined",
            tau=tau_arr,
            random_state=rng,
        )
        same_sim = data[0, :]
        diff_sim = data[1, :]

        if statistic == "Wilcoxon":
            # Special case: Wilcoxon test without DOD fitting
            from scipy.stats import mannwhitneyu

            nlev = data.shape[1]
            # Expand rating categories by their counts for each pair type
            diff_pair_ratings = np.repeat(np.arange(1, nlev + 1), diff_sim.astype(int))
            same_pair_ratings = np.repeat(np.arange(1, nlev + 1), same_sim.astype(int))

            if alt == "two.sided":
                alt_mwu = "two-sided"
            elif alt == "greater":
                alt_mwu = "greater"
            else:
                alt_mwu = "less"

            try:
                # Test if diff-pair ratings are stochastically greater than same-pair ratings
                mwu_result = mannwhitneyu(diff_pair_ratings, same_pair_ratings, alternative=alt_mwu)
                pvals[i] = mwu_result.pvalue
            except ValueError:
                pass

        elif statistic == "Wald":
            # Wald test
            try:
                fit = dod_fit(same_sim, diff_sim, control=ctrl)
                if fit.vcov is not None and np.all(np.isfinite(fit.vcov)):
                    std_err = np.sqrt(fit.vcov[-1, -1])
                    if np.isfinite(std_err) and std_err > 0:
                        stat_value = (fit.d_prime - d_prime0) / std_err
                        p_val = normal_pvalue(stat_value, alt)
                        if hasattr(p_val, 'item'):
                            p_val = p_val.item()
                        pvals[i] = p_val
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                # Optimization or numerical errors - skip this simulation
                pass

        else:  # likelihood or Pearson
            try:
                result = dod(
                    same_sim, diff_sim,
                    d_prime0=d_prime0,
                    alternative=alt,
                    statistic=statistic,
                    control=ctrl,
                )
                if result.convergence == 0:
                    # p_value may be array, extract scalar
                    p_val = result.p_value
                    if hasattr(p_val, 'item'):
                        p_val = p_val.item()
                    pvals[i] = p_val
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                # Optimization or numerical errors - skip this simulation
                pass

    # Compute power
    valid_pvals = pvals[~np.isnan(pvals)]
    n_used = len(valid_pvals)

    if n_used == 0:
        power = np.nan
        se_power = np.nan
    else:
        power = np.mean(valid_pvals < alpha)
        if power == 0 or power == 1 or not np.isfinite(power):
            se_power = np.nan
        else:
            se_power = np.sqrt(power * (1 - power) / n_used)

    return DODPowerResult(
        power=power,
        se_power=se_power,
        n_used=n_used,
        d_primeA=d_primeA,
        d_prime0=d_prime0,
        sample_size=size,
        nsim=nsim,
        alpha=alpha,
        statistic=statistic,
        alternative=alt,
        tau=tau_arr,
    )
