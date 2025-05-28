import numpy as np
from scipy.stats import norm, ncf
from scipy.integrate import quad

__all__ = [
    "two_afc",
    "duotrio_pc",
    "three_afc_pc",
    "triangle_pc",
    "tetrad_pc",
    "hexad_pc",
    "twofive_pc",
    "get_pguess",
    "pc2pd",
    "pd2pc",
    "discrim_2afc", 
    "discrim",      
    "twoAC",        # Add new function
]

import scipy.optimize # Ensure this is imported for minimize

# Try to import scipy.derivative, fallback for older scipy or if not available
try:
    from scipy.derivative import derivative as numerical_derivative
except ImportError:
    try:
        from scipy.misc import derivative as numerical_derivative # Older scipy
    except ImportError:
        # Fallback simple finite difference derivative
        def numerical_derivative(func, x0, dx=1e-6, n=1, order=3):
            if n == 1: # only first derivative needed
                # Central difference of order 2 accuracy if order >=3 not specified
                # For order=3 (default in scipy.misc.derivative), it uses more points.
                # This is a simpler central difference: (f(x+h) - f(x-h)) / 2h
                # For better accuracy, especially if func is complex, a more robust method
                # like from numdifftools would be better, but keeping it simple here.
                return (func(x0 + dx) - func(x0 - dx)) / (2 * dx)
            raise NotImplementedError("Only first derivative (n=1) is implemented in fallback.")


def two_afc(dprime: float) -> float:
    """Proportion correct in a 2-AFC task for a given d-prime."""
    return norm.cdf(dprime / np.sqrt(2))


def duotrio_pc(dprime: float) -> float:
    """Proportion correct in a duo-trio test for a given d-prime."""
    if dprime <= 0:
        return 0.5
    a = norm.cdf(dprime / np.sqrt(2.0))
    b = norm.cdf(dprime / np.sqrt(6.0))
    return 1 - a - b + 2 * a * b


def three_afc_pc(dprime: float) -> float:
    """Proportion correct in a 3-AFC test for a given d-prime."""
    if dprime <= 0:
        return 1.0 / 3.0

    def integrand(x: float) -> float:
        return norm.pdf(x - dprime) * norm.cdf(x) ** 2

    val, _ = quad(integrand, -np.inf, np.inf)
    return min(max(val, 1.0 / 3.0), 1.0)


def triangle_pc(dprime: float) -> float:
    """Proportion correct in a triangle test for a given d-prime."""
    if dprime <= 0:
        return 1.0 / 3.0
    val = ncf.sf(3.0, 1, 1, dprime ** 2 * 2.0 / 3.0)
    return min(max(val, 1.0 / 3.0), 1.0)


def tetrad_pc(dprime: float) -> float:
    """Proportion correct in a tetrad test for a given d-prime."""
    if dprime <= 0:
        return 1.0 / 3.0

    def integrand(z: float) -> float:
        c1 = norm.cdf(z)
        c2 = norm.cdf(z - dprime)
        return norm.pdf(z) * (2 * c1 * c2 - c2 ** 2)

    val, _ = quad(integrand, -np.inf, np.inf)
    res = 1.0 - 2.0 * val
    return min(max(res, 1.0 / 3.0), 1.0)


_HEXAD_COEFFS = [
    0.0977646147,
    0.0319804414,
    0.0656128284,
    0.1454153496,
    -0.0994639381,
    0.0246960778,
    -0.0027806267,
    0.0001198169,
]


def hexad_pc(dprime: float) -> float:
    """Polynomial approximation for the hexad test."""
    if dprime <= 0:
        return 0.1
    if dprime >= 5.368:
        return 1.0
    x = dprime
    val = 0.0
    for i, c in enumerate(_HEXAD_COEFFS):
        val += c * x ** i
    return min(max(val, 0.1), 1.0)


_TWOFIVE_COEFFS = [
    0.0988496065454,
    0.0146108899965,
    0.0708075379445,
    0.0568876949069,
    -0.0424936635277,
    0.0114595626175,
    -0.0016573180506,
    0.0001372413489,
    -0.0000061598395,
    0.0000001166556,
]


def twofive_pc(dprime: float) -> float:
    """Polynomial approximation for the two-out-of-five test."""
    if dprime <= 0:
        return 0.1
    if dprime >= 9.28:
        return 1.0
    x = dprime
    val = 0.0
    for i, c in enumerate(_TWOFIVE_COEFFS):
        val += c * x ** i
    return min(max(val, 0.1), 1.0)


def get_pguess(method: str, double: bool = False) -> float:
    """Return chance performance for a given method."""
    method = method.lower()
    mapping = {
        "duotrio": 0.5,
        "twoafc": 0.5,
        "threeafc": 1.0 / 3.0,
        "triangle": 1.0 / 3.0,
        "tetrad": 1.0 / 3.0,
        "hexad": 0.1,
        "twofive": 0.1,
        "twofivef": 0.4,
    }
    p = mapping.get(method, 0.5)
    return p ** 2 if double else p


def pc2pd(pc: float, pguess: float) -> float:
    """Convert proportion correct to proportion discriminated."""
    if pc <= pguess:
        return 0.0
    return (pc - pguess) / (1.0 - pguess)


def pd2pc(pd: float, pguess: float) -> float:
    """Convert proportion discriminated to proportion correct."""
    if pd <= 0:
        return pguess
    return pguess + pd * (1.0 - pguess)


def discrim_2afc(correct: int, total: int) -> tuple[float, float]:
    """Estimate d-prime from 2-AFC data.

    Returns a tuple of (estimate, standard_error).
    """
    pc = correct / total
    pc = max(min(pc, 1 - 1e-8), 0.5 + 1e-8)
    dprime = np.sqrt(2.0) * norm.ppf(pc)
    se = np.sqrt(pc * (1 - pc) / total) * np.sqrt(2.0) / norm.pdf(norm.ppf(pc))
    return dprime, se


def discrim(correct: int, total: int, method: str, conf_level: float = 0.95, statistic: str = "Wald") -> dict:
    """
    Estimate d-prime and related statistics for various discrimination methods.

    Args:
        correct (int): Number of correct responses.
        total (int): Total number of trials.
        method (str): Discrimination method (e.g., "2afc", "triangle", "duotrio").
        conf_level (float, optional): Confidence level for the CI. Defaults to 0.95.
        statistic (str, optional): Type of statistic to compute. Currently only "Wald" is fully supported.
                                   Defaults to "Wald".

    Returns:
        dict: A dictionary containing d-prime estimate, standard error, confidence interval,
              p-value, method, and statistic type.
    """
    if statistic.lower() != "wald":
        raise NotImplementedError(f"Statistic '{statistic}' not yet implemented. Only 'Wald' is supported.")

    method_lc = method.lower()

    pc_funcs = {
        "2afc": two_afc,
        "two_afc": two_afc,
        "triangle": triangle_pc,
        "duotrio": duotrio_pc,
        "3afc": three_afc_pc,
        "three_afc": three_afc_pc,
        "tetrad": tetrad_pc,
        "hexad": hexad_pc,
        "2outoffive": twofive_pc, # common alias
        "twofive": twofive_pc,
    }

    if method_lc not in pc_funcs:
        raise ValueError(f"Unknown method: {method}. Supported methods are: {list(pc_funcs.keys())}")

    pc_func = pc_funcs[method_lc]
    pguess = get_pguess(method_lc)

    if correct < 0 or total <= 0 or correct > total:
        raise ValueError("Invalid 'correct' or 'total' values.")

    pc_obs = correct / total

    # Adjust pc_obs for edge cases to avoid issues with ppf or log(0) in d' estimation
    # Using a common correction: 1/(2N) for 0 correct, 1 - 1/(2N) for N correct
    # Ensure pc_obs stays within (pguess, 1.0) for sensible d'
    epsilon_lower = 1.0 / (2 * total) if total > 0 else 1e-8
    epsilon_upper = 1.0 / (2 * total) if total > 0 else 1e-8
    
    pc_clipped = np.clip(pc_obs, pguess + epsilon_lower, 1.0 - epsilon_upper)
    
    # If pc_obs is at or below pguess, d-prime is effectively 0 (or undefined negative for some funcs)
    if pc_obs <= pguess:
        dprime_est = 0.0
    # If pc_obs is 1.0 (and pguess < 1.0), d-prime is very large / infinity
    elif pc_obs == 1.0 and pguess < 1.0:
        # For practical purposes, use the clipped value that results in a large d-prime
        # Or, could return np.inf or a pre-defined large value.
        # brentq might struggle if pc_func(large_d_prime) is still < pc_clipped (if pc_clipped is extremely close to 1)
        # Let's try to estimate with the clipped value.
        # A very large d-prime like 10-15 often gives pc near 1.0 for most methods.
        # If pc_func(15) is still less than pc_clipped, brentq might fail.
        # Fallback: if pc_obs is effectively 1, dprime is very large.
        # However, brentq should handle one side of interval being inf if function behaves.
        pass # Will use pc_clipped for brentq

    # Estimate d-prime using brentq
    # Search interval for d-prime. Most d-primes of interest are 0-10.
    # Lower bound can be 0 as d'<0 is not usually meaningful for these %correct functions.
    # Some functions might be defined for d'<0, but pc(d'<0) typically <= pguess.
    dprime_lower_bound = -5.0 # Allow slightly negative for robustness if pc_obs is near pguess
    dprime_upper_bound = 15.0 

    if pc_obs <= pguess: # Already handled: dprime_est = 0.0
        pass
    elif pc_obs == 1.0 and pguess < 1.0: # Handle perfect score
        # If pc_func can reach 1.0, brentq might find a root with pc_clipped.
        # If pc_func asymptotes below 1.0, this indicates an issue or need for very high dprime.
        # For now, let dprime_est be found by brentq with pc_clipped.
        # If brentq fails (e.g. pc_clipped is too high for pc_func(dprime_upper_bound)),
        # it might indicate dprime is effectively infinity or outside search range.
        try:
            # Check if the function can even reach pc_clipped
            if pc_func(dprime_upper_bound) < pc_clipped:
                 dprime_est = dprime_upper_bound # Or np.inf, or raise warning
            else:
                dprime_est = scipy.optimize.brentq(lambda d: pc_func(d) - pc_clipped, dprime_lower_bound, dprime_upper_bound, xtol=1e-6, rtol=1e-6)
        except ValueError: # brentq fails if f(a) and f(b) must have different signs
            # This can happen if pc_clipped is outside range [pc_func(lower), pc_func(upper)]
            # or if pc_func is flat.
            if pc_func(dprime_lower_bound) > pc_clipped : # pc_clipped is too low
                dprime_est = dprime_lower_bound 
            elif pc_func(dprime_upper_bound) < pc_clipped : # pc_clipped is too high
                dprime_est = dprime_upper_bound
            else: # Other brentq failure (e.g. flat function, no sign change)
                dprime_est = np.nan # Or handle error appropriately
    else: # pc_obs is between pguess and 1.0 (exclusive of 1.0)
        try:
            dprime_est = scipy.optimize.brentq(lambda d: pc_func(d) - pc_clipped, dprime_lower_bound, dprime_upper_bound, xtol=1e-6, rtol=1e-6)
        except ValueError:
            # Fallback for difficult cases, could indicate pc_clipped is too extreme
            if pc_func(dprime_lower_bound) > pc_clipped: dprime_est = dprime_lower_bound
            elif pc_func(dprime_upper_bound) < pc_clipped: dprime_est = dprime_upper_bound
            else: dprime_est = np.nan


    # Wald Standard Error
    se_dprime = np.nan
    if pc_obs == 0.0 or pc_obs == 1.0:
        se_dprime = np.inf
    elif np.isfinite(dprime_est):
        # Numerical derivative of pc_func at dprime_est
        dx_deriv = max(abs(dprime_est) * 1e-4, 1e-6) 
        deriv_val = numerical_derivative(pc_func, dprime_est, dx=dx_deriv)

        if deriv_val is not None and abs(deriv_val) > 1e-9: # Avoid division by zero or tiny derivative
            # Use original pc_obs for variance calculation, not pc_clipped, for binomial variance part
            variance_pc = pc_obs * (1 - pc_obs) / total
            se_dprime = np.sqrt(variance_pc) / deriv_val
        else: # Derivative is zero or too small, SE is undefined or very large
            se_dprime = np.inf
    
    # Wald Confidence Interval
    z_crit = norm.ppf(1 - (1 - conf_level) / 2)
    lower_ci = dprime_est - z_crit * se_dprime
    upper_ci = dprime_est + z_crit * se_dprime

    # P-value (Wald test for dprime=0)
    if np.isfinite(se_dprime) and se_dprime > 1e-9: # Avoid division by zero for wald_z
        wald_z = dprime_est / se_dprime
        p_value = 2 * norm.sf(np.abs(wald_z)) # Two-tailed
    else: # If SE is inf or zero, p-value is problematic
        if dprime_est == 0: p_value = 1.0
        elif np.isinf(se_dprime) and dprime_est != 0 : p_value = np.nan # SE is inf, cannot determine
        else: p_value = np.nan # Default for other problematic SE cases

    return {
        "dprime": dprime_est,
        "se_dprime": se_dprime,
        "lower_ci": lower_ci,
        "upper_ci": upper_ci,
        "p_value": p_value,
        "conf_level": conf_level,
        "correct": correct,
        "total": total,
        "pc_obs": pc_obs,
        "pguess": pguess,
        "method": method,
        "statistic": statistic,
    }


# Helper function for twoAC log-likelihood
def _loglik_twoAC(params, h, f, n_signal, n_noise, epsilon=1e-12):
    """
    Log-likelihood for the 2AC (Yes/No or Same/Different with bias) model.
    params: [delta, tau]
    h: hits
    f: false alarms
    n_signal: number of signal trials
    n_noise: number of noise trials
    epsilon: small value to prevent log(0)
    """
    delta, tau = params
    
    pHit = norm.cdf(delta / 2 - tau)
    pFA = norm.cdf(-delta / 2 - tau)

    # Clip probabilities to avoid log(0)
    pHit = np.clip(pHit, epsilon, 1 - epsilon)
    pFA = np.clip(pFA, epsilon, 1 - epsilon)

    loglik = (
        h * np.log(pHit) +
        (n_signal - h) * np.log(1 - pHit) +
        f * np.log(pFA) +
        (n_noise - f) * np.log(1 - pFA)
    )
    return loglik


def twoAC(x: list[int], n: list[int], method: str = "ml") -> dict:
    """
    Thurstonian model for 2-Alternative Choice (Yes/No) with response bias.

    Estimates sensitivity (delta) and bias (tau) using maximum likelihood.

    Args:
        x (list[int]): A list or 2-element array `[hits, false_alarms]`.
        n (list[int]): A list or 2-element array `[n_signal_trials, n_noise_trials]`.
        method (str, optional): Estimation method. Currently only "ml" (maximum likelihood)
                                is supported. Defaults to "ml".

    Returns:
        dict: A dictionary containing estimated parameters (delta, tau), standard errors,
              log-likelihood, variance-covariance matrix, convergence status, and input data.
    """
    if method.lower() != "ml":
        raise NotImplementedError("Only 'ml' (maximum likelihood) method is currently supported.")

    if len(x) != 2 or len(n) != 2:
        raise ValueError("'x' and 'n' must both be lists or arrays of length 2.")

    h = x[0]  # hits
    f = x[1]  # false alarms
    n_signal = n[0]  # number of signal trials
    n_noise = n[1]   # number of noise trials

    if not (0 <= h <= n_signal and 0 <= f <= n_noise):
        raise ValueError("Number of hits/false_alarms must be between 0 and respective trial counts.")
    if n_signal <= 0 or n_noise <= 0:
        raise ValueError("Number of signal and noise trials must be positive.")

    # Objective function for minimization (negative log-likelihood)
    def objective_func(params):
        return -_loglik_twoAC(params, h, f, n_signal, n_noise)

    # Initial guesses
    # d' from P(Hit) and P(FA) ignoring bias, tau=0
    # pH_obs = h/n_signal; pFA_obs = f/n_noise
    # delta_init_approx = norm.ppf(np.clip(pH_obs, 1e-5, 1-1e-5)) - norm.ppf(np.clip(pFA_obs, 1e-5, 1-1e-5))
    # tau_init_approx = -0.5 * (norm.ppf(np.clip(pH_obs,1e-5,1-1e-5)) + norm.ppf(np.clip(pFA_obs,1e-5,1-1e-5)))
    # Using simpler fixed initial guesses for now
    delta_init = 0.0 
    tau_init = 0.0
    initial_params = np.array([delta_init, tau_init])
    
    # Bounds for parameters: delta >= 0, tau is unbounded
    bounds = [(0, None), (None, None)]

    result = scipy.optimize.minimize(
        objective_func,
        initial_params,
        method="L-BFGS-B",
        bounds=bounds,
        hess='2-point' # Request Hessian approximation for vcov
    )

    delta_est, tau_est = result.x
    loglik_val = -result.fun
    convergence_status = result.success
    
    vcov_matrix = np.full((2, 2), np.nan)
    se_delta, se_tau = np.nan, np.nan

    if convergence_status:
        if hasattr(result, 'hess_inv') and result.hess_inv is not None:
            try:
                # For L-BFGS-B, hess_inv is an OptimizeResultLBFGSB specific attribute
                # which is an approximation of the inverse Hessian.
                vcov_matrix = result.hess_inv.todense()
                if vcov_matrix[0,0] >=0 : se_delta = np.sqrt(vcov_matrix[0, 0])
                if vcov_matrix[1,1] >=0 : se_tau = np.sqrt(vcov_matrix[1, 1])
            except Exception: # Catch any error during vcov processing (e.g. not positive definite)
                warnings.warn("Could not compute standard errors from Hessian.", RuntimeWarning)
        else:
             warnings.warn("Hessian information not available from optimizer for SE calculation.", RuntimeWarning)


    return {
        "delta": delta_est,
        "tau": tau_est,
        "se_delta": se_delta,
        "se_tau": se_tau,
        "loglik": loglik_val,
        "vcov": vcov_matrix,
        "convergence_status": convergence_status,
        "hits": h,
        "false_alarms": f,
        "n_signal_trials": n_signal,
        "n_noise_trials": n_noise,
    }
