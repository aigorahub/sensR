"""Beta-binomial models for over-dispersed discrimination data.

This module implements the beta-binomial model and chance-corrected
beta-binomial model for replicated discrimination tests with overdispersion.
"""

import warnings
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike
from scipy import optimize, special, stats

from senspy.core.types import Protocol, parse_protocol
from senspy.links import get_link
from senspy.utils import rescale


@dataclass
class BetaBinomialResult:
    """Result from beta-binomial model fitting.

    Attributes
    ----------
    coefficients : dict
        Dictionary with 'mu' and 'gamma' parameter estimates.
    vcov : np.ndarray | None
        2x2 variance-covariance matrix of (mu, gamma), or None if not computed.
    log_likelihood : float
        Log-likelihood at the MLE.
    log_lik_null : float
        Log-likelihood under null model (p = p_guess).
    log_lik_mu : float
        Log-likelihood under binomial model (p = observed proportion).
    data : np.ndarray
        The input data matrix (successes, trials).
    method : str
        The discrimination protocol used.
    corrected : bool
        Whether the chance-corrected model was fit.
    convergence : int
        Optimizer convergence status (0 = converged).
    message : str
        Optimizer message.
    n_iter : int
        Number of optimizer iterations.
    """

    coefficients: dict[str, float]
    vcov: np.ndarray | None
    log_likelihood: float
    log_lik_null: float
    log_lik_mu: float
    data: np.ndarray
    method: str
    corrected: bool
    convergence: int
    message: str
    n_iter: int
    _p_guess: float = field(repr=False)

    @property
    def mu(self) -> float:
        """Mean parameter estimate."""
        return self.coefficients["mu"]

    @property
    def gamma(self) -> float:
        """Overdispersion parameter estimate."""
        return self.coefficients["gamma"]

    def se(self) -> dict[str, float] | None:
        """Standard errors of mu and gamma."""
        if self.vcov is None:
            return None
        se_vals = np.sqrt(np.diag(self.vcov))
        return {"mu": se_vals[0], "gamma": se_vals[1]}

    def lr_overdispersion(self) -> tuple[float, float]:
        """Likelihood ratio test for overdispersion.

        Returns
        -------
        tuple[float, float]
            (test statistic G^2, p-value) with 1 df.
            Note: The true df may be between 0.5 and 1 since gamma is tested
            at the boundary.
        """
        g2 = 2 * (self.log_likelihood - self.log_lik_mu)
        p_value = stats.chi2.sf(g2, df=1)
        return g2, p_value

    def lr_association(self) -> tuple[float, float]:
        """Likelihood ratio test for association (difference from null).

        Returns
        -------
        tuple[float, float]
            (test statistic G^2, p-value) with 2 df.
        """
        g2 = 2 * (self.log_likelihood - self.log_lik_null)
        p_value = stats.chi2.sf(g2, df=2)
        return g2, p_value

    def summary(self, level: float = 0.95) -> "BetaBinomialSummary":
        """Generate summary with confidence intervals.

        Parameters
        ----------
        level : float
            Confidence level (default 0.95).

        Returns
        -------
        BetaBinomialSummary
            Summary object with estimates, SEs, and CIs on multiple scales.
        """
        return BetaBinomialSummary.from_result(self, level=level)


@dataclass
class BetaBinomialSummary:
    """Summary of beta-binomial model fit."""

    estimates: dict[str, float]
    std_errors: dict[str, float | None]
    ci_lower: dict[str, float | None]
    ci_upper: dict[str, float | None]
    level: float
    log_likelihood: float
    lr_overdispersion: tuple[float, float]
    lr_association: tuple[float, float]
    method: str
    corrected: bool

    @classmethod
    def from_result(
        cls, result: BetaBinomialResult, level: float = 0.95
    ) -> "BetaBinomialSummary":
        """Create summary from BetaBinomialResult."""
        mu = result.mu
        gamma = result.gamma
        se_dict = result.se()

        # Initialize estimates
        estimates = {"mu": mu, "gamma": gamma, "pc": np.nan, "pd": np.nan, "d_prime": np.nan}
        std_errors: dict[str, float | None] = {k: None for k in estimates}
        ci_lower: dict[str, float | None] = {k: None for k in estimates}
        ci_upper: dict[str, float | None] = {k: None for k in estimates}

        # Fill in mu and gamma SE/CI
        if se_dict is not None:
            std_errors["mu"] = se_dict["mu"]
            std_errors["gamma"] = se_dict["gamma"]

            alpha = (1 - level) / 2
            z = stats.norm.ppf(1 - alpha)

            # Wald CI for mu
            mu_lower = max(0.0, mu - z * se_dict["mu"])
            mu_upper = min(1.0, mu + z * se_dict["mu"])
            ci_lower["mu"] = mu_lower
            ci_upper["mu"] = mu_upper

            # Wald CI for gamma
            ci_lower["gamma"] = max(0.0, gamma - z * se_dict["gamma"])
            ci_upper["gamma"] = min(1.0, gamma + z * se_dict["gamma"])

        # Transform mu to pc, pd, d_prime scales
        if not np.isnan(mu):
            if result.corrected:
                # mu is on pd scale
                obj = rescale(pd=mu, method=result.method)
            else:
                # mu is on pc scale
                obj = rescale(pc=mu, method=result.method)

            estimates["pc"] = obj.pc
            estimates["pd"] = obj.pd
            estimates["d_prime"] = obj.d_prime

            # Transform SE using delta method (via rescale)
            if se_dict is not None and se_dict["mu"] is not None:
                if result.corrected:
                    obj_se = rescale(pd=mu, se=se_dict["mu"], method=result.method)
                else:
                    obj_se = rescale(pc=mu, se=se_dict["mu"], method=result.method)

                std_errors["pc"] = obj_se.se_pc
                std_errors["pd"] = obj_se.se_pd
                std_errors["d_prime"] = obj_se.se_d_prime

            # Transform CI limits
            if ci_lower["mu"] is not None and ci_upper["mu"] is not None:
                if result.corrected:
                    obj_lower = rescale(pd=mu_lower, method=result.method)
                    obj_upper = rescale(pd=mu_upper, method=result.method)
                else:
                    obj_lower = rescale(pc=mu_lower, method=result.method)
                    obj_upper = rescale(pc=mu_upper, method=result.method)

                ci_lower["pc"] = obj_lower.pc
                ci_lower["pd"] = obj_lower.pd
                ci_lower["d_prime"] = obj_lower.d_prime

                ci_upper["pc"] = obj_upper.pc
                ci_upper["pd"] = obj_upper.pd
                ci_upper["d_prime"] = obj_upper.d_prime

        return cls(
            estimates=estimates,
            std_errors=std_errors,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            level=level,
            log_likelihood=result.log_likelihood,
            lr_overdispersion=result.lr_overdispersion(),
            lr_association=result.lr_association(),
            method=result.method,
            corrected=result.corrected,
        )

    def __str__(self) -> str:
        """Format summary as string."""
        lines = []
        model_type = "Chance-corrected beta-binomial" if self.corrected else "Beta-binomial"
        lines.append(f"{model_type} model for the {self.method} protocol")
        lines.append(f"with {self.level * 100:.1f}% confidence intervals")
        lines.append("")

        # Header
        lines.append(f"{'Parameter':<10} {'Estimate':>10} {'Std.Err':>10} {'Lower':>10} {'Upper':>10}")
        lines.append("-" * 55)

        # Parameters
        for param in ["mu", "gamma", "pc", "pd", "d_prime"]:
            est = self.estimates[param]
            se = self.std_errors[param]
            lo = self.ci_lower[param]
            hi = self.ci_upper[param]

            est_str = f"{est:10.4f}" if not np.isnan(est) else f"{'NA':>10}"
            se_str = f"{se:10.4f}" if se is not None else f"{'NA':>10}"
            lo_str = f"{lo:10.4f}" if lo is not None else f"{'NA':>10}"
            hi_str = f"{hi:10.4f}" if hi is not None else f"{'NA':>10}"

            lines.append(f"{param:<10} {est_str} {se_str} {lo_str} {hi_str}")

        lines.append("")
        lines.append(f"log-likelihood: {self.log_likelihood:.4f}")

        g2_od, p_od = self.lr_overdispersion
        lines.append(f"LR-test of over-dispersion: G^2={g2_od:.4f}, df=1, p-value={p_od:.4g}")

        g2_assoc, p_assoc = self.lr_association
        lines.append(f"LR-test of association: G^2={g2_assoc:.4f}, df=2, p-value={p_assoc:.4g}")

        return "\n".join(lines)


def _mu_gamma_to_alpha_beta(mu: float, gamma: float) -> tuple[float, float]:
    """Convert (mu, gamma) parameterization to (alpha, beta)."""
    alpha = mu * (1 - gamma) / gamma
    beta = (1 - gamma) * (1 - mu) / gamma
    return alpha, beta


def _log_likelihood_standard(
    params: np.ndarray, x: np.ndarray, n: np.ndarray
) -> float:
    """Negative log-likelihood for standard beta-binomial model.

    Parameters
    ----------
    params : array
        [mu, gamma] parameters in (0, 1).
    x : array
        Number of successes for each observation.
    n : array
        Number of trials for each observation.

    Returns
    -------
    float
        Negative log-likelihood (for minimization).
    """
    mu, gamma = params
    alpha, beta = _mu_gamma_to_alpha_beta(mu, gamma)

    # Log-likelihood: sum of log(Beta(alpha+x, beta+n-x) / Beta(alpha, beta))
    # Plus the binomial coefficient terms (constant)
    N = len(x)
    ll = -N * special.betaln(alpha, beta) + np.sum(
        special.betaln(alpha + x, beta + n - x)
    )
    return -ll  # Return negative for minimization


def _logsumexp(log_values: np.ndarray) -> float:
    """Compute log(sum(exp(log_values))) in a numerically stable way.

    This is equivalent to scipy.special.logsumexp but inlined for clarity.
    """
    if len(log_values) == 0:
        return -np.inf
    max_val = np.max(log_values)
    if np.isinf(max_val):
        return max_val
    return max_val + np.log(np.sum(np.exp(log_values - max_val)))


def _log_likelihood_corrected(
    params: np.ndarray, x: np.ndarray, n: np.ndarray, p_guess: float
) -> float:
    """Negative log-likelihood for chance-corrected beta-binomial model.

    This computes the kernel of the log-likelihood without constant factors
    (lchoose and pGuess terms), which are added back via the Factor constant.

    The formula follows sensR's nllAux pattern where the (1-pGuess)^(n-x) * pGuess^x
    terms are factored out and accounted for separately in Factor.

    All computations are done in log-space to prevent numerical underflow
    for small gamma values (large alpha/beta).

    Parameters
    ----------
    params : array
        [mu, gamma] parameters in (0, 1).
    x : array
        Number of successes for each observation.
    n : array
        Number of trials for each observation.
    p_guess : float
        Guessing probability for the protocol.

    Returns
    -------
    float
        Negative log-likelihood kernel (for minimization).
    """
    mu, gamma = params
    alpha, beta = _mu_gamma_to_alpha_beta(mu, gamma)

    ll = 0.0
    N = len(x)

    # Log of ratio for probability calculation: log((1-pGuess)/pGuess)
    log_ratio = np.log(1 - p_guess) - np.log(p_guess)

    for j in range(N):
        xj, nj = int(x[j]), int(n[j])
        # Sum over possible true correct counts (i = number of true discriminations)
        # Using log-space: log(choose(x,i)) + i*log(ratio) + log(beta(a+i, b+n-x))
        # where log(beta(a,b)) = betaln(a,b)
        log_terms = np.zeros(xj + 1)
        for i in range(xj + 1):
            log_coeff = special.gammaln(xj + 1) - special.gammaln(i + 1) - special.gammaln(xj - i + 1)
            log_prob = i * log_ratio
            log_beta = special.betaln(alpha + i, nj - xj + beta)
            log_terms[i] = log_coeff + log_prob + log_beta

        # Use logsumexp for numerically stable summation
        ll += _logsumexp(log_terms)

    # Subtract the common Beta(alpha, beta) term (N times)
    ll -= N * special.betaln(alpha, beta)

    return -ll  # Return negative for minimization


def betabin(
    data: ArrayLike,
    method: str | Protocol = "duotrio",
    corrected: bool = True,
    start: tuple[float, float] = (0.5, 0.5),
    vcov: bool = True,
    grad_tol: float = 1e-4,
) -> BetaBinomialResult:
    """Fit beta-binomial model to over-dispersed discrimination data.

    Parameters
    ----------
    data : array-like
        Matrix or array with shape (N, 2) where first column is number of
        successes (correct responses) and second column is number of trials.
        Each row represents an independent observation (e.g., a panelist).
    method : str or Protocol
        The sensory discrimination protocol. One of 'triangle', 'duotrio',
        'twoafc', 'threeafc', 'tetrad', 'hexad', 'twofive', 'twofivef'.
    corrected : bool, default True
        If True, fit the chance-corrected model where mu represents Pd
        (probability of discrimination). If False, fit the standard model
        where mu represents Pc (probability correct).
    start : tuple[float, float], default (0.5, 0.5)
        Starting values for (mu, gamma) optimization. Must be in (0, 1).
    vcov : bool, default True
        Whether to compute the variance-covariance matrix.
    grad_tol : float, default 1e-4
        Gradient tolerance for convergence warning.

    Returns
    -------
    BetaBinomialResult
        Fitted model result with coefficients, vcov, log-likelihood, etc.

    Raises
    ------
    ValueError
        If data has wrong shape or start values are out of bounds.

    Examples
    --------
    >>> import numpy as np
    >>> from senspy import betabin
    >>> # Data: successes and total trials per panelist
    >>> data = np.array([[3, 10], [5, 10], [7, 10], [4, 10], [6, 10]])
    >>> result = betabin(data, method="triangle", corrected=True)
    >>> print(f"mu={result.mu:.3f}, gamma={result.gamma:.3f}")
    """
    # Validate and convert data
    data = np.asarray(data)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("'data' must have 2 columns (successes, trials)")
    if data.shape[0] < 3:
        raise ValueError("'data' must have at least 3 rows")

    x = data[:, 0].astype(float)
    n = data[:, 1].astype(float)

    if np.any(x < 0) or np.any(x > n):
        raise ValueError("Successes must be between 0 and trials")

    # Validate start values
    if any(s <= 1e-3 or s >= 1 - 1e-3 for s in start):
        raise ValueError("start values must be in the open interval (0, 1)")

    # Get protocol info
    protocol = parse_protocol(method)
    method_str = protocol.value
    link = get_link(method_str)
    p_guess = link.p_guess

    # Set up bounds (log-space arithmetic allows small gamma values)
    lower = np.array([1e-6, 1e-6])
    upper = np.array([1 - 1e-6, 1 - 1e-6])

    # Define objective function
    if corrected:
        def objective(params):
            return _log_likelihood_corrected(params, x, n, p_guess)
    else:
        def objective(params):
            return _log_likelihood_standard(params, x, n)

    # Optimize
    result = optimize.minimize(
        objective,
        x0=np.array(start),
        method="L-BFGS-B",
        bounds=list(zip(lower, upper)),
        options={"ftol": 1e-10, "gtol": 1e-8},
    )

    coef = result.x
    mu, gamma = coef

    # Check gradient at optimum
    if all(coef > lower) and all(coef < upper):
        # Compute gradient numerically
        eps = 1e-7
        grad = np.zeros(2)
        for i in range(2):
            params_plus = coef.copy()
            params_plus[i] += eps
            params_minus = coef.copy()
            params_minus[i] -= eps
            grad[i] = (objective(params_plus) - objective(params_minus)) / (2 * eps)

        if np.max(np.abs(grad)) > grad_tol:
            warnings.warn(
                f"Optimizer terminated with max|gradient|: {np.max(np.abs(grad)):.2e}",
                stacklevel=2,
            )
    else:
        warnings.warn("Parameters at boundary occurred", stacklevel=2)

    # Compute variance-covariance matrix
    vcov_matrix = None
    if vcov and all(coef > lower) and all(coef < upper):
        # Compute Hessian numerically
        hess = _compute_hessian(objective, coef)
        try:
            vcov_matrix = np.linalg.inv(hess)
        except np.linalg.LinAlgError:
            vcov_matrix = None

    # Compute log-likelihoods for tests
    # Factor (binomial coefficients) - constant term
    factor = np.sum(special.gammaln(n + 1) - special.gammaln(x + 1) - special.gammaln(n - x + 1))
    if corrected:
        factor += np.sum((n - x) * np.log(1 - p_guess)) + np.sum(x * np.log(p_guess))

    log_lik = -result.fun + factor

    # Null model: binomial with p = p_guess
    log_lik_null = np.sum(stats.binom.logpmf(x.astype(int), n.astype(int), p_guess))

    # Mu-only model: binomial with p = observed proportion
    p_obs = np.sum(x) / np.sum(n)
    log_lik_mu = np.sum(stats.binom.logpmf(x.astype(int), n.astype(int), p_obs))

    return BetaBinomialResult(
        coefficients={"mu": mu, "gamma": gamma},
        vcov=vcov_matrix,
        log_likelihood=log_lik,
        log_lik_null=log_lik_null,
        log_lik_mu=log_lik_mu,
        data=data,
        method=method_str,
        corrected=corrected,
        convergence=0 if result.success else 1,
        message=result.message,
        n_iter=result.nit,
        _p_guess=p_guess,
    )


def _compute_hessian(func, x, eps=1e-5):
    """Compute Hessian matrix numerically using finite differences."""
    n = len(x)
    hess = np.zeros((n, n))

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

            hess[i, j] = (func(x_pp) - func(x_pm) - func(x_mp) + func(x_mm)) / (4 * eps * eps)
            hess[j, i] = hess[i, j]

    return hess
