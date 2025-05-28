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
    "twoAC",        
    "par2prob_dod", 
    "dod_nll",      
    "dod",          
    "samediff_nll", 
    "samediff",     
    "dprime_test",  
    "dprime_compare", 
    "SDT",            
    "AUC",            # Add new function
]

import scipy.optimize 
from scipy.stats import binom, chi2 # For binomial logpmf and chi2.sf
from senspy.links import psyfun 
import warnings # For handling warnings in par2prob_dod

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


def par2prob_dod(tau: np.ndarray, d_prime: float) -> np.ndarray:
    """
    Calculate category probabilities for "same" and "different" pairs in a DoD task.

    Args:
        tau (np.ndarray): 1D array of K-1 increasing positive boundary parameters.
        d_prime (float): Sensitivity index (must be >= 0).

    Returns:
        np.ndarray: A 2xK array of probabilities.
                    Row 0: Probabilities for "same" pairs.
                    Row 1: Probabilities for "different" pairs.
    """
    # Input validation
    if not isinstance(tau, np.ndarray) or tau.ndim != 1:
        raise ValueError("tau must be a 1D NumPy array.")
    if not np.all(tau > 0):
        raise ValueError("All elements of tau must be positive.")
    if not np.all(np.diff(tau) > 0):
        raise ValueError("Elements of tau must be strictly increasing.")
    if not (isinstance(d_prime, (float, int)) and d_prime >= 0):
        raise ValueError("d_prime must be a non-negative scalar.")

    # Calculate gamma values
    gamma_same = 2 * norm.cdf(tau / np.sqrt(2)) - 1
    gamma_diff = norm.cdf((tau - d_prime) / np.sqrt(2)) - norm.cdf((-tau - d_prime) / np.sqrt(2))

    # Calculate probabilities for each category
    # Ensure endpoints are 0 and 1 for np.diff to work correctly
    # np.diff calculates the difference between subsequent elements
    p_same = np.diff(np.concatenate(([0.0], gamma_same, [1.0])))
    p_diff = np.diff(np.concatenate(([0.0], gamma_diff, [1.0])))
    
    # Probabilities should ideally sum to 1. Numerical precision might cause tiny deviations.
    # Clipping to ensure valid probabilities (e.g. >=0) and normalization can be added if necessary,
    # but the formulas should mathematically ensure this if inputs are valid and cdf outputs are in [0,1].
    # Let's add a small clip and re-normalize just in case of extreme floating point issues,
    # though this is often handled by how these probabilities are used (e.g., in log-likelihood).
    epsilon = 1e-12
    p_same = np.clip(p_same, epsilon, 1.0)
    p_diff = np.clip(p_diff, epsilon, 1.0)

    p_same /= np.sum(p_same)
    p_diff /= np.sum(p_diff)
    
    return np.vstack((p_same, p_diff))


def dod_nll(params: np.ndarray, same_counts: np.ndarray, diff_counts: np.ndarray) -> float:
    """
    Negative log-likelihood for the Degree of Difference (DoD) model.

    Args:
        params (np.ndarray): 1D array where params[:-1] are tau values and params[-1] is d_prime.
        same_counts (np.ndarray): Observed counts for "same" pairs in each category.
        diff_counts (np.ndarray): Observed counts for "different" pairs in each category.

    Returns:
        float: Negative log-likelihood.
    """
    num_categories = len(same_counts)
    if len(params) != num_categories: # K-1 taus + 1 d_prime = K params
        raise ValueError(f"Length of params ({len(params)}) must be equal to number of categories ({num_categories}).")

    tau = params[:-1]
    d_prime = params[-1]

    # Basic validation for optimizer (optimizer should handle bounds, but good for direct calls)
    if d_prime < 0:
        return np.inf # Or a very large number, d_prime must be non-negative
    if np.any(tau <= 0) or np.any(np.diff(tau) <= 0): # Taus must be positive and increasing
        return np.inf

    if len(diff_counts) != num_categories:
        raise ValueError("same_counts and diff_counts must have the same length (number of categories).")

    # Calculate probabilities
    # par2prob_dod expects K-1 tau values for K categories.
    # If params contains K elements (K-1 taus, 1 dprime), then tau = params[:-1] has K-1 elements.
    prob = par2prob_dod(tau, d_prime) # prob is 2xK

    # Calculate log-likelihood
    # np.log will produce -inf if prob is 0. This is desired as it makes NLL +inf.
    # We rely on par2prob_dod to clip probabilities slightly above 0 if needed.
    loglik_same = np.sum(same_counts * np.log(prob[0, :]))
    loglik_diff = np.sum(diff_counts * np.log(prob[1, :]))
    
    total_loglik = loglik_same + loglik_diff

    if not np.isfinite(total_loglik):
        return np.inf # Return large positive for NLL if loglik was -inf

    return -total_loglik


def dod(same_counts: np.ndarray, diff_counts: np.ndarray, 
        initial_tau: np.ndarray | None = None, 
        initial_d_prime: float | None = None,
        method: str = "ml", # For future extensions like 'clm'
        conf_level: float = 0.95) -> dict:
    """
    Degree of Difference (DoD) analysis. (Placeholder)

    Estimates d-prime and boundary parameters (tau) from DoD data.

    Args:
        same_counts (np.ndarray): 1D array of observed counts for "same" pairs.
        diff_counts (np.ndarray): 1D array of observed counts for "different" pairs.
        initial_tau (np.ndarray | None, optional): Initial guess for tau parameters.
        initial_d_prime (float | None, optional): Initial guess for d-prime.
        method (str, optional): Estimation method. Defaults to "ml".
        conf_level (float, optional): Confidence level for intervals. Defaults to 0.95.

    Returns:
        dict: Results including d-prime, tau, SEs, CIs, log-likelihood, etc.
    """
    # This is a placeholder for the full fitting logic to be implemented later.
    # For now, it doesn't do anything.
    # The full implementation will use dod_nll with scipy.optimize.minimize.
    
    # Example of what would be returned eventually:
    # return {
    #     "d_prime": None, "se_d_prime": None, "ci_d_prime": (None, None),
    #     "tau": None, "se_tau": None, "ci_tau": None,
    #     "loglik": None, "vcov": None, "convergence_status": None,
    #     "method": method, "conf_level": conf_level,
    #     "same_counts": same_counts, "diff_counts": diff_counts
    # }
    if method.lower() != "ml":
        raise NotImplementedError("Only 'ml' (maximum likelihood) method is currently supported.")

    # --- Input Validation ---
    same_counts = np.asarray(same_counts, dtype=np.int32)
    diff_counts = np.asarray(diff_counts, dtype=np.int32)

    if same_counts.ndim != 1 or diff_counts.ndim != 1:
        raise ValueError("same_counts and diff_counts must be 1D arrays.")
    if len(same_counts) != len(diff_counts):
        raise ValueError("same_counts and diff_counts must have the same length.")
    
    num_categories = len(same_counts)
    if num_categories < 2:
        raise ValueError("Number of categories must be at least 2.")

    if np.any(same_counts < 0) or np.any(diff_counts < 0):
        raise ValueError("Counts must be non-negative.")
    if np.sum(same_counts) == 0 and np.sum(diff_counts) == 0:
        raise ValueError("Total counts for both same and different pairs cannot be zero.")

    # --- Initial Parameter Guesses (Optimizing tpar and d_prime) ---
    # tpar are the increments of tau: tau_i = cumsum(tpar_i)
    
    if initial_tau is not None:
        initial_tau = np.asarray(initial_tau, dtype=float)
        if len(initial_tau) != num_categories - 1:
            raise ValueError(f"initial_tau must have length {num_categories - 1}")
        if not (np.all(initial_tau > 0) and np.all(np.diff(initial_tau) > 0)):
            raise ValueError("initial_tau must be positive and strictly increasing.")
        tpar_init = np.concatenate(([initial_tau[0]], np.diff(initial_tau)))
        if np.any(tpar_init <= 0):
             warnings.warn("Derived initial_tpar from initial_tau contains non-positive values. Check initial_tau. Defaulting tpar_init.", UserWarning)
             tpar_init = _init_tpar_sensr(num_categories) # Fallback
    else:
        tpar_init = _init_tpar_sensr(num_categories)

    d_prime_init = initial_d_prime if initial_d_prime is not None else 1.0
    if d_prime_init < 0:
        warnings.warn("initial_d_prime is negative. Using 0.0 as initial guess for d_prime.", UserWarning)
        d_prime_init = 0.0 # Optimizer bound will also enforce this

    initial_params_optim = np.concatenate((tpar_init, [d_prime_init]))

    # --- Parameter Bounds for Optimization ---
    # tpar_i >= epsilon, d_prime >= 0 (or epsilon for strict positivity if model requires)
    epsilon_bound = 1e-5 
    bounds_tpar = [(epsilon_bound, None)] * (num_categories - 1)
    bounds_d_prime = (epsilon_bound, None) # d_prime strictly positive
    bounds_for_optimizer = bounds_tpar + [bounds_d_prime]

    # --- Objective Function Wrapper ---
    def _objective_func_dod_wrapper(params_optim, s_counts, d_counts):
        tpar_optim = params_optim[:-1]
        d_prime_optim_val = params_optim[-1]
        
        # Transform tpar to tau for dod_nll
        # tau values must be positive and strictly increasing for par2prob_dod
        # The bounds on tpar should enforce tpar_i > 0, so cumsum(tpar) will be increasing.
        # The first tau (tpar[0]) is also > 0.
        tau_optim = np.cumsum(tpar_optim)
        
        # dod_nll expects (K-1) taus and 1 d_prime, total K parameters.
        params_for_nll = np.concatenate((tau_optim, [d_prime_optim_val]))
        return dod_nll(params_for_nll, s_counts, d_counts)

    # --- Optimization ---
    optim_result = scipy.optimize.minimize(
        _objective_func_dod_wrapper,
        initial_params_optim,
        args=(same_counts, diff_counts),
        method="L-BFGS-B",
        bounds=bounds_for_optimizer,
        hess='2-point' 
    )

    # --- Results Extraction ---
    optim_params_final = optim_result.x
    tpar_final = optim_params_final[:-1]
    tau_final = np.cumsum(tpar_final)
    d_prime_final = optim_params_final[-1]
    
    loglik = -optim_result.fun
    convergence_status = optim_result.success
    
    vcov_optim_params = np.full((len(initial_params_optim), len(initial_params_optim)), np.nan)
    se_tpar = np.full(len(tpar_final), np.nan)
    se_d_prime = np.nan

    if convergence_status:
        if hasattr(optim_result, 'hess_inv') and optim_result.hess_inv is not None:
            try:
                vcov_optim_params = optim_result.hess_inv.todense()
                se_all_optim_params = np.sqrt(np.diag(vcov_optim_params))
                se_tpar = se_all_optim_params[:-1]
                se_d_prime = se_all_optim_params[-1]
            except Exception as e:
                warnings.warn(f"Could not compute standard errors from Hessian: {e}", RuntimeWarning)
        else:
             warnings.warn("Hessian information not available from optimizer for SE calculation.", RuntimeWarning)
    
    # Note: se_tau would require delta method transformation from se_tpar and vcov_tpar.
    # For this step, returning se_tpar is sufficient.

    return {
        "d_prime": d_prime_final,
        "tau": tau_final, # Derived from optimized tpar
        "se_d_prime": se_d_prime,
        "tpar": tpar_final, # Optimized tpar values
        "se_tpar": se_tpar, # Standard errors for tpar
        "loglik": loglik,
        "vcov_optim_params": vcov_optim_params, # VCOV of (tpar, d_prime)
        "convergence_status": convergence_status,
        "initial_params_optim": initial_params_optim, # tpar_init and d_prime_init
        "optim_result": optim_result, # Full scipy result object for inspection
        "same_counts": same_counts,
        "diff_counts": diff_counts,
        "method": method,
        "conf_level": conf_level # Retained from signature, though CIs are not yet computed here
    }

def _init_tpar_sensr(num_categories: int) -> np.ndarray:
    """
    Helper to generate initial tpar values based on sensR's initTau logic.
    tpar are the increments for tau: tau_k = sum_{i=1 to k} tpar_i.
    num_categories (K) is the number of response categories.
    This means there are K-1 tau values, and thus K-1 tpar values.
    """
    if num_categories < 2:
        raise ValueError("Number of categories must be at least 2 for DoD.")
    
    num_tpar = num_categories - 1
    
    if num_tpar == 1: # K=2 categories, 1 tau, 1 tpar
        return np.array([1.0])
    else: # K > 2 categories
        # sensR: initTau <- c(1, rep(3/(ncat - 1), ncat - 2))
        # This gives K-1 tpar values.
        tpar_values = [1.0] + [3.0 / (num_tpar)] * (num_tpar - 1)
        # Correction: sensR's ncat in rep(3/(ncat-1), ncat-2) seems to refer to K-1 (number of taus)
        # Let's re-interpret based on sensR code or typical behavior.
        # If K = num_categories, then number of taus = K-1.
        # sensR's init_tau(ncat) where ncat = K-1 (number of taus)
        # If K=3 (2 taus), init_tau(2) = c(1). This is 1 tpar.
        # If K=4 (3 taus), init_tau(3) = c(1, 3/(3-1)) = c(1, 1.5). These are 2 tpars.
        # If K=5 (4 taus), init_tau(4) = c(1, 3/(4-1), 3/(4-1)) = c(1, 1, 1). These are 3 tpars.
        # This means the formula `3.0 / (num_tpar - 1)` should be used for the `rep` part if `num_tpar > 1`.
        
        if num_tpar == 1: # e.g. K=2 categories, so 1 tau, 1 tpar
             tpar_values = [1.0]
        else: # num_tpar > 1
             tpar_values = [1.0] + [3.0 / (num_tpar -1 + 1e-9)] * (num_tpar - 1) # Add epsilon for num_tpar=1 case if it got here
             # Corrected based on understanding sensR's initTau(ncat = K-1)
             # initTau for K-1 thresholds. tpar has K-1 elements.
             # First tpar is 1. Remaining K-2 tpars are 3/((K-1)-1) = 3/(K-2) if K-2 > 0
             if num_tpar == 1: # K=2, one tau, so one tpar
                 tpar_values = np.array([1.0])
             else: # K > 2, num_tpar = K-1 > 1
                 # tpar_0 = 1.0
                 # tpar_1, ..., tpar_{K-2} are 3.0 / ( (K-1) -1 ) = 3.0 / (K-2)
                 # Example: K=3, num_tpar=2. tpar_0=1.0. One more tpar: 3.0/(3-2)=3.0. So [1.0, 3.0] - This is wrong.
                 # sensR initTau(ncat) where ncat is number of thresholds (K-1)
                 # initTau(N) -> c(1, rep(3/(N-1), N-2))
                 # So for K-1 thresholds (num_tpar = K-1):
                 if num_tpar == 1: # K=2
                     tpar_values = np.array([1.0])
                 else: # num_tpar > 1 (K > 2)
                     tpar_values = np.concatenate( ([1.0] , np.full(num_tpar - 1, 3.0 / (num_tpar -1 + 1e-9) )) )
                     # The above was sensR's initTau for ncat = num_tpar.
                     # Example K=3, num_tpar=2. initTau(2) = c(1, rep(3/1, 0)) = c(1). So tpar=[1.0]. This is wrong.
                     # initTau(ncat = K-1)
                     # if K=2, ncat=1. initTau(1) should be [1]. length 1.
                     # if K=3, ncat=2. initTau(2) should be [1, 3/(2-1)] = [1,3]. length 2.
                     # if K=4, ncat=3. initTau(3) should be [1, 3/(3-1), 3/(3-1)] = [1, 1.5, 1.5]. length 3.
                     # The logic for `rep` means `N-2` elements.
                     # So, tpar_0 = 1.0.
                     # The remaining `num_tpar - 1` elements are `3.0 / (num_tpar - 1)` if `num_tpar - 1 > 0`.
                     # This means `num_tpar - 1` should be used in the denominator.
                     if num_tpar == 1: # K=2
                        return np.array([1.0])
                     else: # num_tpar > 1 (K > 2)
                        return np.concatenate(([1.0], np.full(num_tpar - 1, 3.0 / (num_tpar - 1.0 + 1e-9) )))


        return np.asarray(tpar_values)


    p = mapping.get(method, 0.5)
    return p ** 2 if double else p


def AUC(d_prime: float, scale: float = 1.0) -> float:
    """
    Calculate the Area Under the ROC Curve (AUC) from d-prime.

    This calculation is based on the assumptions of Signal Detection Theory (SDT)
    with equal variance for the noise and signal distributions when scale=1.
    The formula corresponds to `pnorm(d_prime / sqrt(1 + scale^2))` as used in `sensR`.

    Args:
        d_prime (float): The sensitivity index (d-prime). Can be any real number.
        scale (float, optional): A scale parameter, typically representing the ratio of
                                 standard deviations (sd_signal / sd_noise) if they differ.
                                 Must be strictly positive. Defaults to 1.0 (equal variance).

    Returns:
        float: The calculated AUC value, ranging from 0 to 1.
    """
    # Input Validation
    if not isinstance(d_prime, (float, int, np.number)): # np.number for numpy float/int types
        raise TypeError("d_prime must be a numeric scalar.")
    if not isinstance(scale, (float, int, np.number)):
        raise TypeError("scale must be a numeric scalar.")
    if scale <= 0:
        raise ValueError("scale must be strictly positive.")

    # AUC Calculation
    auc_value = norm.cdf(d_prime / np.sqrt(1 + scale**2))
    
    return auc_value


def SDT(counts_table: np.ndarray, method: str = "probit") -> np.ndarray:
    """
    Signal Detection Theory analysis for rating scales.

    Calculates z(Hit rate), z(False alarm rate), and d-prime for each criterion
    based on a 2xJ table of observations.

    Args:
        counts_table (np.ndarray): A 2xJ NumPy array.
            Row 0: Counts for "noise" distribution (e.g., false alarms per category).
            Row 1: Counts for "signal" distribution (e.g., hits per category).
            J is the number of rating categories (J >= 2).
        method (str, optional): Transformation method. Either "probit" (default) or "logit".

    Returns:
        np.ndarray: A (J-1)x3 NumPy array:
            Column 0: Transformed values for noise distribution (zFA or logitFA).
            Column 1: Transformed values for signal distribution (zH or logitH).
            Column 2: d-prime (or equivalent for logit) for each criterion.
    """
    # --- Input Validation ---
    if not isinstance(counts_table, np.ndarray):
        raise TypeError("counts_table must be a NumPy array.")
    if counts_table.ndim != 2:
        raise ValueError("counts_table must be a 2D array.")
    if counts_table.shape[0] != 2:
        raise ValueError("counts_table must have 2 rows (noise and signal distributions).")
    if counts_table.shape[1] < 2:
        raise ValueError("counts_table must have at least 2 columns (categories).")
    if not np.issubdtype(counts_table.dtype, np.integer):
        # Allow float if they are whole numbers, but prefer integer.
        if not np.all(counts_table == np.floor(counts_table)):
             raise ValueError("counts_table must contain integer counts.")
        counts_table = counts_table.astype(np.int32) # Convert if they are whole floats
    if np.any(counts_table < 0):
        raise ValueError("All counts in counts_table must be non-negative.")

    method_lc = method.lower()
    if method_lc not in ["probit", "logit"]:
        raise ValueError("method must be either 'probit' or 'logit'.")

    num_categories = counts_table.shape[1]
    num_criteria = num_categories - 1
    
    results_transformed = np.full((2, num_criteria), np.nan) # For zH/zFA or logitH/logitFA

    for i in range(2): # 0 for noise, 1 for signal
        row_counts = counts_table[i, :]
        total_count_i = np.sum(row_counts)

        if total_count_i == 0:
            warnings.warn(f"Row {i} (noise/signal) in counts_table has zero total count. Results for this row will be NaN.", UserWarning)
            continue # results_transformed[i,:] will remain NaN

        cumulative_counts_i = np.cumsum(row_counts)
        # We need J-1 cumulative proportions for the J-1 criteria
        # These are P(response >= category_j+1) or P(response > category_j)
        # Or, if using left-to-right cumulation: P(response <= category_j)
        # sensR's SDT uses P(X > c_j), which is 1 - P(X <= c_j)
        # If counts are [c1, c2, c3, c4], J=4. Criteria c1, c2, c3.
        # P(X > c1) = (c2+c3+c4)/N = 1 - c1/N
        # P(X > c2) = (c3+c4)/N   = 1 - (c1+c2)/N
        # P(X > c3) = c4/N         = 1 - (c1+c2+c3)/N
        # So, use cumulative proportions from the left, take 1-that, for J-1 points.
        # Or, cumulate from right: cum_prop_right = np.cumsum(row_counts[::-1])[::-1] / total_count_i
        
        cumulative_proportions_i = cumulative_counts_i[:-1] / total_count_i # Take only first J-1

        # Apply corrections for proportions of 0 or 1
        # sensR uses correction: p_adj = (p * (N-1) + 0.5) / N to avoid exactly 0 or 1
        # which is equivalent to (count + 0.5) / (N+1) if p = count/N.
        # A simpler common one is 0.5/N and (N-0.5)/N
        corrected_proportions = np.zeros_like(cumulative_proportions_i, dtype=float)
        for j in range(num_criteria):
            p = cumulative_proportions_i[j]
            if p == 0.0:
                corrected_proportions[j] = 0.5 / total_count_i
            elif p == 1.0:
                corrected_proportions[j] = (total_count_i - 0.5) / total_count_i
            else:
                corrected_proportions[j] = p
        
        if method_lc == "probit":
            results_transformed[i, :] = norm.ppf(corrected_proportions)
        elif method_lc == "logit":
            # Ensure corrected_proportions are not exactly 0 or 1 for logit
            # The previous correction should handle this, but clip again for safety before logit
            logit_input = np.clip(corrected_proportions, 1e-9, 1.0 - 1e-9)
            results_transformed[i, :] = np.log(logit_input / (1.0 - logit_input))
            
    zFA_or_logitFA = results_transformed[0, :] # Noise distribution transformed values
    zH_or_logitH = results_transformed[1, :]   # Signal distribution transformed values

    # Calculate d-prime (or equivalent for logit scale)
    # d_prime_j = zH_j - zFA_j (for probit)
    # For logit, this difference is on the log-odds scale.
    d_prime_values = zH_or_logitH - zFA_or_logitFA

    # Assemble results into (J-1)x3 array
    # Column 0: zFA / logitFA
    # Column 1: zH / logitH
    # Column 2: d_prime_j / logit_diff_j
    output_array = np.vstack((zFA_or_logitFA, zH_or_logitH, d_prime_values)).T
    
    return output_array


# --- Same-Different (SD) Method Core Components ---

def _get_samediff_probs(tau: float, delta: float, epsilon: float = 1e-12) -> tuple[float, float, float, float]:
    """
    Calculate response probabilities for the Same-Different method.

    Args:
        tau (float): Decision criterion parameter (> 0).
        delta (float): Sensitivity parameter (d-prime, >= 0).
        epsilon (float): Small value for clipping probabilities to avoid log(0).

    Returns:
        tuple[float, float, float, float]: (Pss, Pds, Psd, Pdd)
    """
    # Pss (Prob "same" response | "same" stimulus pair)
    Pss = 2 * norm.cdf(tau / np.sqrt(2)) - 1
    
    # Pds (Prob "different" response | "same" stimulus pair)
    Pds = 1 - Pss
    
    # Psd (Prob "same" response | "different" stimulus pair)
    Psd = norm.cdf((tau - delta) / np.sqrt(2)) - norm.cdf((-tau - delta) / np.sqrt(2))
    
    # Pdd (Prob "different" response | "different" stimulus pair)
    Pdd = 1 - Psd

    # Clip probabilities
    Pss = np.clip(Pss, epsilon, 1.0 - epsilon)
    Pds = np.clip(Pds, epsilon, 1.0 - epsilon)
    Psd = np.clip(Psd, epsilon, 1.0 - epsilon)
    Pdd = np.clip(Pdd, epsilon, 1.0 - epsilon)
    
    # Normalize probabilities for each stimulus condition (row-wise) to sum to 1
    # This is important if clipping significantly altered the sum.
    sum_same_stim = Pss + Pds
    Pss /= sum_same_stim
    Pds /= sum_same_stim
    
    sum_diff_stim = Psd + Pdd
    Psd /= sum_diff_stim
    Pdd /= sum_diff_stim

    return Pss, Pds, Psd, Pdd


def samediff_nll(params: np.ndarray, nsamesame: int, ndiffsame: int, nsamediff: int, ndiffdiff: int) -> float:
    """
    Negative log-likelihood for the Same-Different (SD) model.

    Args:
        params (np.ndarray): 1D NumPy array `[tau, delta]`.
        nsamesame (int): Count of "same" responses to "same" stimuli.
        ndiffsame (int): Count of "different" responses to "same" stimuli.
        nsamediff (int): Count of "same" responses to "different" stimuli.
        ndiffdiff (int): Count of "different" responses to "different" stimuli.

    Returns:
        float: Negative log-likelihood.
    """
    tau, delta = params

    # Input validation for parameters (from optimizer)
    if tau <= 0 or delta < 0:
        return np.inf # Penalize invalid parameter values

    Pss, Pds, Psd, Pdd = _get_samediff_probs(tau, delta)

    # Calculate log-likelihood
    # np.log will produce -inf if any probability is exactly 0 (after clipping in helper).
    loglik = (
        nsamesame * np.log(Pss) +
        ndiffsame * np.log(Pds) +
        nsamediff * np.log(Psd) +
        ndiffdiff * np.log(Pdd)
    )

    if not np.isfinite(loglik):
        return np.inf # Should not happen if _get_samediff_probs clips correctly
    
    return -loglik


def samediff(nsamesame: int, ndiffsame: int, nsamediff: int, ndiffdiff: int,
             initial_tau: float | None = None, 
             initial_delta: float | None = None,
             method: str = "ml", # For future extensions
             conf_level: float = 0.95) -> dict:
    """
    Same-Different (SD) method analysis. (Placeholder for full implementation)

    Estimates sensitivity (delta) and bias/criterion (tau) from SD data.

    Args:
        nsamesame (int): Count of "same" responses to "same" stimuli.
        ndiffsame (int): Count of "different" responses to "same" stimuli.
        nsamediff (int): Count of "same" responses to "different" stimuli.
        ndiffdiff (int): Count of "different" responses to "different" stimuli.
        initial_tau (float | None, optional): Initial guess for tau.
        initial_delta (float | None, optional): Initial guess for delta.
        method (str, optional): Estimation method. Defaults to "ml".
        conf_level (float, optional): Confidence level for intervals. Defaults to 0.95.

    Returns:
        dict: Results including delta, tau, SEs, CIs, log-likelihood, etc.
    """
    # Placeholder - full fitting logic to be added in a subsequent step.
    # Example structure:
    # return {
    #     "delta": None, "se_delta": None, "ci_delta": (None, None),
    #     "tau": None, "se_tau": None, "ci_tau": (None, None),
    #     "loglik": None, "vcov": None, "convergence_status": None,
    #     "method": method, "conf_level": conf_level,
    #     "nsamesame": nsamesame, "ndiffsame": ndiffsame, 
    #     "nsamediff": nsamediff, "ndiffdiff": ndiffdiff
    # }
    if method.lower() != "ml":
        raise NotImplementedError("Only 'ml' (maximum likelihood) method is currently supported.")

    # --- Input Validation ---
    counts = np.array([nsamesame, ndiffsame, nsamediff, ndiffdiff], dtype=np.int32)
    if np.any(counts < 0):
        raise ValueError("All counts must be non-negative integers.")
    if not all(isinstance(c, (int, np.integer)) for c in [nsamesame, ndiffsame, nsamediff, ndiffdiff]):
         raise ValueError("All counts must be integers.")

    if np.sum(counts) <= 0:
        raise ValueError("The sum of counts must be positive.")
    
    if np.sum(counts > 0) < 2:
        raise ValueError("Not enough information in data: at least two of the four counts must be non-zero.")

    # --- Initial Parameter Guesses ---
    tau_init_val = initial_tau if initial_tau is not None and initial_tau > 0 else 1.0
    delta_init_val = initial_delta if initial_delta is not None and initial_delta >= 0 else 1.0
    
    initial_params_optim = np.array([tau_init_val, delta_init_val])

    # --- Parameter Bounds for Optimization ---
    epsilon_bound = 1e-5 # For strict positivity of tau
    bounds = [(epsilon_bound, None),  # tau > 0
              (0, None)]              # delta >= 0

    # --- Optimization ---
    optim_result = scipy.optimize.minimize(
        samediff_nll,
        initial_params_optim,
        args=(nsamesame, ndiffsame, nsamediff, ndiffdiff),
        method="L-BFGS-B",
        bounds=bounds,
        hess='2-point' 
    )

    # --- Results Extraction ---
    tau_final, delta_final = optim_result.x
    loglik = -optim_result.fun
    convergence_status = optim_result.success
    
    vcov_matrix = np.full((2, 2), np.nan)
    se_tau, se_delta = np.nan, np.nan

    if convergence_status:
        if hasattr(optim_result, 'hess_inv') and optim_result.hess_inv is not None:
            try:
                vcov_matrix = optim_result.hess_inv.todense()
                if vcov_matrix[0,0] >=0 : se_tau = np.sqrt(vcov_matrix[0, 0])
                if vcov_matrix[1,1] >=0 : se_delta = np.sqrt(vcov_matrix[1, 1])
            except Exception as e:
                warnings.warn(f"Could not compute standard errors from Hessian: {e}", RuntimeWarning)
        else:
             warnings.warn("Hessian information not available from optimizer for SE calculation.", RuntimeWarning)

    return {
        "tau": tau_final,
        "delta": delta_final,
        "se_tau": se_tau,
        "se_delta": se_delta,
        "loglik": loglik,
        "vcov": vcov_matrix,
        "convergence_status": convergence_status,
        "initial_params": initial_params_optim, # What was fed to optimizer
        "optim_result": optim_result, # Full scipy result object
        "nsamesame": nsamesame,
        "ndiffsame": ndiffsame,
        "nsamediff": nsamediff,
        "ndiffdiff": ndiffdiff,
        "method": method,
        "conf_level": conf_level # Retained, though CIs not computed in this step
    }


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
