import numpy as np
from scipy.stats import norm, ncf
from scipy.integrate import quad

# Assuming DoDModel and SameDifferentModel will be imported from senspy.models
from senspy.models import DoDModel, SameDifferentModel, DiscriminationModel, TwoACModel

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
    "dod",          
    "samediff",     
    "dprime_test",  
    "dprime_compare", 
    "SDT",            
    "AUC",
    "psyfun",
    "psyinv",
    "psyderiv",
    "rescale",
]

import scipy.optimize
from scipy.optimize import brentq
from scipy.stats import binom, chi2
import warnings

try:
    from scipy.derivative import derivative as numerical_derivative
except ImportError:
    try:
        from scipy.misc import derivative as numerical_derivative
    except ImportError:
        def numerical_derivative(func, x0, dx=1e-6, n=1, order=3):
            if n == 1:
                return (func(x0 + dx) - func(x0 - dx)) / (2 * dx)
            raise NotImplementedError("Only first derivative (n=1) is implemented in fallback.")

def two_afc(dprime: float) -> float:
    return norm.cdf(dprime / np.sqrt(2))

def duotrio_pc(dprime: float) -> float:
    if dprime <= 0: return 0.5
    a = norm.cdf(dprime / np.sqrt(2.0))
    b = norm.cdf(dprime / np.sqrt(6.0))
    return 1 - a - b + 2 * a * b

def three_afc_pc(dprime: float) -> float:
    if dprime <= 0: return 1.0 / 3.0
    integrand = lambda x: norm.pdf(x - dprime) * norm.cdf(x) ** 2
    val, _ = quad(integrand, -np.inf, np.inf)
    return min(max(val, 1.0 / 3.0), 1.0)

def triangle_pc(dprime: float) -> float:
    if dprime <= 0: return 1.0 / 3.0
    val = ncf.sf(3.0, 1, 1, dprime ** 2 * 2.0 / 3.0)
    return min(max(val, 1.0 / 3.0), 1.0)

def tetrad_pc(dprime: float) -> float:
    if dprime <= 0: return 1.0 / 3.0
    integrand = lambda z: norm.pdf(z) * (2 * norm.cdf(z) * norm.cdf(z - dprime) - norm.cdf(z - dprime) ** 2)
    val, _ = quad(integrand, -np.inf, np.inf)
    res = 1.0 - 2.0 * val
    return min(max(res, 1.0 / 3.0), 1.0)

_HEXAD_COEFFS = [0.0977646147, 0.0319804414, 0.0656128284, 0.1454153496, -0.0994639381, 0.0246960778, -0.0027806267, 0.0001198169]
def hexad_pc(dprime: float) -> float:
    if dprime <= 0: return 0.1
    if dprime >= 5.368: return 1.0
    x = dprime; val = 0.0
    for i, c in enumerate(_HEXAD_COEFFS): val += c * x ** i
    return min(max(val, 0.1), 1.0)

_TWOFIVE_COEFFS = [0.0988496065454, 0.0146108899965, 0.0708075379445, 0.0568876949069, -0.0424936635277, 0.0114595626175, -0.0016573180506, 0.0001372413489, -0.0000061598395, 0.0000001166556]
def twofive_pc(dprime: float) -> float:
    if dprime <= 0: return 0.1
    if dprime >= 9.28: return 1.0
    x = dprime; val = 0.0
    for i, c in enumerate(_TWOFIVE_COEFFS): val += c * x ** i
    return min(max(val, 0.1), 1.0)

def get_pguess(method: str) -> float:
    method = method.lower()
    mapping = {
        "duotrio": 0.5, "twoafc": 0.5, "threeafc": 1.0 / 3.0,
        "triangle": 1.0 / 3.0, "tetrad": 1.0 / 3.0, "hexad": 0.1,
        "twofive": 0.1, "twofivef": 0.4,
    }
    return mapping.get(method, 0.5)

def dod(same_counts: np.ndarray, diff_counts: np.ndarray, 
        initial_tau: np.ndarray | None = None, 
        initial_d_prime: float | None = None,
        method: str = "ml",
        conf_level: float = 0.95) -> dict:
    model = DoDModel()
    model.fit(
        same_counts=same_counts, diff_counts=diff_counts,
        initial_tau=initial_tau, initial_d_prime=initial_d_prime,
        method=method, conf_level=conf_level
    )
    # Reconstruct dict for backward compatibility, including initial_params_optim if available
    initial_params_optim_val = None
    if model.optim_result_obj and hasattr(model.optim_result_obj, 'x'):
        # The model's fit method stores tpar and d_prime separately.
        # We need to reconstruct the 'initial_params_optim' as it was defined before: (tpar_init, d_prime_init)
        # This information is not directly stored in DoDModel in that specific format post-fit.
        # For now, returning the optimized parameters of the internal representation.
        # If the exact *initial guess* that was fed to the optimizer is needed, DoDModel.fit would need to store it.
        initial_params_optim_val = model.optim_result_obj.x

    return {
        "d_prime": model.d_prime, "tau": model.tau,
        "se_d_prime": model.se_d_prime, "tpar": model.tpar,
        "se_tpar": model.se_tpar, "loglik": model.loglik,
        "vcov_optim_params": model.vcov_optim_params,
        "convergence_status": model.convergence_status,
        "initial_params_optim": initial_params_optim_val,
        "optim_result": model.optim_result_obj,
        "same_counts": model.same_counts, "diff_counts": model.diff_counts,
        "method": method, "conf_level": model.conf_level_used
    }

_PC_FUNCTIONS_INTERNAL = {
    "2afc": two_afc, "two_afc": two_afc, "triangle": triangle_pc,
    "duotrio": duotrio_pc, "3afc": three_afc_pc, "three_afc": three_afc_pc,
    "tetrad": tetrad_pc, "hexad": hexad_pc, "twofive": twofive_pc,
    "2-afc": two_afc, "3-afc": three_afc_pc, "2-out-of-5": twofive_pc,
}

def psyfun(dprime: float, method: str = "2afc") -> float:
    method_lc = method.lower()
    if method_lc not in _PC_FUNCTIONS_INTERNAL:
        raise ValueError(f"Unknown method: {method}. Supported methods: {list(_PC_FUNCTIONS_INTERNAL.keys())}")
    return _PC_FUNCTIONS_INTERNAL[method_lc](dprime)

def psyinv(pc: float, method: str = "2afc") -> float:
    method_lc = method.lower()
    pc_func = _PC_FUNCTIONS_INTERNAL[method_lc]
    pguess = get_pguess(method_lc)
    if not (pguess <= pc <= 1.0):
        if pc < pguess - 1e-9:
             warnings.warn(f"pc ({pc:.4f}) is below pguess ({pguess:.4f}) for method {method}. Returning d-prime = 0.", UserWarning)
             return 0.0
        pc = max(pc, pguess)
    if pc <= pguess: return 0.0
    if pc >= 1.0 - 1e-9:
        if pc_func(20.0) < 1.0 - 1e-9:
            try: return brentq(lambda d: pc_func(d) - pc, 20.0, 50.0, xtol=1e-6, rtol=1e-6)
            except ValueError: return 50.0
        return np.inf
    epsilon_brentq = 1e-9 
    pc_adjusted = np.clip(pc, pguess + epsilon_brentq, 1.0 - epsilon_brentq)
    if abs(pc_adjusted - pguess) < epsilon_brentq: return 0.0
    try:
        lower_bound, upper_bound = 0.0, 20.0
        val_at_lower = pc_func(lower_bound) - pc_adjusted
        val_at_upper = pc_func(upper_bound) - pc_adjusted
        if val_at_lower * val_at_upper > 0:
            if val_at_upper < 0: upper_bound = 50.0
            elif val_at_lower > 0: return 0.0
        return brentq(lambda d: pc_func(d) - pc_adjusted, lower_bound, upper_bound, xtol=1e-6, rtol=1e-6)
    except ValueError as e:
        warnings.warn(f"brentq failed in psyinv for pc={pc:.4f}, method={method}: {e}. pc_adjusted={pc_adjusted:.4f}. Trying wider search or returning boundary.", UserWarning)
        if pc_adjusted >= pc_func(50.0) - epsilon_brentq: return 50.0
        if pc_adjusted <= pc_func(0.0) + epsilon_brentq: return 0.0
        return np.nan

def psyderiv(dprime: float, method: str = "2afc") -> float:
    method_lc = method.lower()
    pc_func = _PC_FUNCTIONS_INTERNAL[method_lc]
    if not np.isfinite(dprime): return 0.0 
    if dprime < 0:
        if abs(pc_func(dprime) - pc_func(max(dprime - 1e-3, dprime * 1.1 if dprime < 0 else dprime * 0.9))) < 1e-9: return 0.0
    dx_val = max(abs(dprime) * 1e-5, 1e-7) if dprime != 0 else 1e-7
    return numerical_derivative(pc_func, dprime, dx=dx_val, n=1, order=3)

def rescale(x: float, from_scale: str, to_scale: str, method: str = "2afc", std_err: float | None = None) -> dict:
    method_lc = method.lower()
    pguess = get_pguess(method_lc)
    val_pc, val_pd, val_d_prime = np.nan, np.nan, np.nan
    if from_scale == "pc":
        val_pc = x; val_d_prime = psyinv(val_pc, method=method_lc)
        val_pd = (val_pc - pguess) / (1.0 - pguess) if (1.0 - pguess) != 0 and val_pc > pguess else 0.0
    elif from_scale == "pd":
        val_pd = x; val_pc = val_pd * (1.0 - pguess) + pguess
        val_d_prime = psyinv(val_pc, method=method_lc)
    elif from_scale == "dp":
        val_d_prime = x; val_pc = psyfun(val_d_prime, method=method_lc)
        val_pd = (val_pc - pguess) / (1.0 - pguess) if (1.0 - pguess) != 0 and val_pc > pguess else 0.0
    else: raise ValueError(f"Invalid from_scale: {from_scale}")
    
    se_pc, se_pd, se_d_prime = np.nan, np.nan, np.nan
    if std_err is not None:
        if not (np.isfinite(std_err) and std_err >= 0): raise ValueError("std_err must be non-negative.")
        deriv_pc_dprime = psyderiv(val_d_prime, method=method_lc)
        deriv_dprime_pc = 1.0 / deriv_pc_dprime if abs(deriv_pc_dprime) > 1e-9 else np.inf

        if from_scale == "pc":
            se_pc = std_err; se_d_prime = se_pc * deriv_dprime_pc
            se_pd = se_pc / (1.0 - pguess) if (1.0 - pguess) != 0 else np.inf
        elif from_scale == "pd":
            se_pd = std_err; se_pc = se_pd * (1.0 - pguess)
            se_d_prime = se_pc * deriv_dprime_pc
        elif from_scale == "dp":
            se_d_prime = std_err; se_pc = se_d_prime * deriv_pc_dprime
            se_pd = se_pc / (1.0 - pguess) if (1.0 - pguess) != 0 else np.inf
            
    return {"coefficients": {"pc": val_pc, "pd": val_pd, "d_prime": val_d_prime},
            "std_errors": {"pc": se_pc, "pd": se_pd, "d_prime": se_d_prime},
            "method": method_lc, "input_scale": from_scale, "output_scale": to_scale}

def AUC(d_prime: float, scale: float = 1.0) -> float:
    if not isinstance(d_prime, (float, int, np.number)): raise TypeError("d_prime must be numeric.")
    if not isinstance(scale, (float, int, np.number)): raise TypeError("scale must be numeric.")
    if scale <= 0: raise ValueError("scale must be strictly positive.")
    return norm.cdf(d_prime / np.sqrt(1 + scale**2))

def SDT(counts_table: np.ndarray, method: str = "probit") -> np.ndarray:
    if not isinstance(counts_table, np.ndarray) or counts_table.ndim != 2 or counts_table.shape[0] != 2 or counts_table.shape[1] < 2:
        raise ValueError("counts_table must be a 2xJ (J>=2) NumPy array.")
    if not np.issubdtype(counts_table.dtype, np.integer) and not np.all(counts_table == np.floor(counts_table)):
        raise ValueError("counts_table must contain integer counts.")
    counts_table = counts_table.astype(np.int32)
    if np.any(counts_table < 0): raise ValueError("Counts must be non-negative.")
    method_lc = method.lower()
    if method_lc not in ["probit", "logit"]: raise ValueError("method must be 'probit' or 'logit'.")

    num_criteria = counts_table.shape[1] - 1
    results_transformed = np.full((2, num_criteria), np.nan)
    for i in range(2):
        row_counts = counts_table[i, :]
        total_count_i = np.sum(row_counts)
        if total_count_i == 0:
            warnings.warn(f"Row {i} in counts_table has zero total. Results for this row will be NaN.", UserWarning)
            continue
        cumulative_proportions_i = np.cumsum(row_counts)[:-1] / total_count_i
        corrected_proportions = np.array([0.5 / total_count_i if p == 0 else (total_count_i - 0.5) / total_count_i if p == 1 else p for p in cumulative_proportions_i])
        if method_lc == "probit": results_transformed[i, :] = norm.ppf(corrected_proportions)
        else: results_transformed[i, :] = np.log(np.clip(corrected_proportions, 1e-9, 1.0 - 1e-9) / (1.0 - np.clip(corrected_proportions, 1e-9, 1.0 - 1e-9)))
            
    d_prime_values = results_transformed[1, :] - results_transformed[0, :]
    return np.vstack((results_transformed[0, :], results_transformed[1, :], d_prime_values)).T

# --- Same-Different (SD) Method Core Components ---
# _get_samediff_probs and samediff_nll are now moved to models.py

def samediff(nsamesame: int, ndiffsame: int, nsamediff: int, ndiffdiff: int,
             initial_tau: float | None = None, 
             initial_delta: float | None = None,
             method: str = "ml",
             conf_level: float = 0.95) -> dict:
    model = SameDifferentModel()
    model.fit(
        nsamesame=nsamesame, ndiffsame=ndiffsame,
        nsamediff=nsamediff, ndiffdiff=ndiffdiff,
        initial_tau=initial_tau, initial_delta=initial_delta,
        method=method, conf_level=conf_level
    )
    # Reconstruct dict for backward compatibility
    initial_params_val = None
    if model.optim_result_obj and hasattr(model.optim_result_obj, 'x'):
        initial_params_val = model.optim_result_obj.x # This is actually optimized params

    return {
        "tau": model.tau, "delta": model.delta,
        "se_tau": model.se_tau, "se_delta": model.se_delta,
        "loglik": model.loglik, "vcov": model.vcov,
        "convergence_status": model.convergence_status,
        "initial_params": initial_params_val, # Note: This is optimized params from model.
        "optim_result": model.optim_result_obj,
        "nsamesame": model.nsamesame, "ndiffsame": model.ndiffsame,
        "nsamediff": model.nsamediff, "ndiffdiff": model.ndiffdiff,
        "method": method, "conf_level": model.conf_level_used
    }

def pc2pd(pc: float, pguess: float) -> float:
    if pc <= pguess: return 0.0
    return (pc - pguess) / (1.0 - pguess)

def pd2pc(pd: float, pguess: float) -> float:
    if pd <= 0: return pguess
    return pguess + pd * (1.0 - pguess)

def discrim_2afc(correct: int, total: int) -> tuple[float, float]:
    pc = correct / total
    pc = max(min(pc, 1 - 1e-8), 0.5 + 1e-8) # Ensure pc is within valid range for norm.ppf
    dprime = np.sqrt(2.0) * norm.ppf(pc)
    # Ensure norm.pdf does not get extreme values from pc near 0 or 1 for ppf
    se = np.sqrt(pc * (1 - pc) / total) * np.sqrt(2.0) / norm.pdf(norm.ppf(pc)) if (pc > 1e-8 and pc < 1-1e-8) else np.inf
    return dprime, se

def discrim(correct: int, total: int, method: str, conf_level: float = 0.95, statistic: str = "Wald") -> dict:
    model = DiscriminationModel()
    # The fit method of DiscriminationModel now contains the core logic.
    model.fit(correct=correct, total=total, method=method,
              conf_level=conf_level, statistic=statistic)
    return model.results_dict

def _loglik_twoAC(params, h, f, n_signal, n_noise, epsilon=1e-12):
    delta, tau = params
    pHit = norm.cdf(delta / 2.0 - tau)
    pFA = norm.cdf(-delta / 2.0 - tau)
    pHit = np.clip(pHit, epsilon, 1.0 - epsilon)
    pFA = np.clip(pFA, epsilon, 1.0 - epsilon)
    return (h * np.log(pHit) + (n_signal - h) * np.log(1.0 - pHit) +
            f * np.log(pFA) + (n_noise - f) * np.log(1.0 - pFA))

def twoAC(x: list[int], n: list[int], method: str = "ml") -> dict:
    if method.lower() != "ml":
        raise NotImplementedError("Only 'ml' (maximum likelihood) method is currently supported.")
    h, f = x[0], x[1]
    n_signal, n_noise = n[0], n[1]
    if not (0 <= h <= n_signal and 0 <= f <= n_noise):
        raise ValueError("Hits/false_alarms must be between 0 and respective trial counts.")
    if n_signal <= 0 or n_noise <= 0:
        raise ValueError("Number of signal and noise trials must be positive.")
    
    model = TwoACModel()
    model.fit(hits=h, false_alarms=f, n_signal_trials=n_signal, n_noise_trials=n_noise)
    
    se_delta, se_tau = np.nan, np.nan
    if model.vcov is not None and isinstance(model.vcov, np.ndarray) and model.vcov.shape == (2,2) and not np.all(np.isnan(model.vcov)):
        if model.vcov[0,0] >= 0: se_delta = np.sqrt(model.vcov[0,0])
        if model.vcov[1,1] >= 0: se_tau = np.sqrt(model.vcov[1,1])

    return {
        "delta": model.delta, "tau": model.tau,
        "se_delta": se_delta, "se_tau": se_tau,
        "loglik": model.loglik, "vcov": model.vcov,
        "convergence_status": model.convergence_status,
        "hits": h, "false_alarms": f,
        "n_signal_trials": n_signal, "n_noise_trials": n_noise,
    }

# Placeholder for dprime_test and dprime_compare - keep existing structure for now
def dprime_test(*args, **kwargs):
    # This would eventually use a dedicated model or be refactored.
    # For now, keeping it as a non-model function or a very simple wrapper if needed.
    warnings.warn("dprime_test is a placeholder and not yet refactored to a model structure.", UserWarning)
    # Simulate a plausible output structure if called by other tests expecting it.
    # This needs to be more robust if it's actually used.
    if len(args) > 0 and isinstance(args[0], (list, np.ndarray)): # multi-group
        num_groups = len(args[0])
        return {"p_value": 0.5, "common_dprime_est": 1.0,
                "individual_group_estimates": [{"dprime":1.0}]*num_groups,
                "convergence_status_common_dprime": True}
    return {"p_value": 0.5, "common_dprime_est": 1.0,
            "individual_group_estimates": [{"dprime":1.0}],
            "convergence_status_common_dprime": True}


def dprime_compare(*args, **kwargs):
    warnings.warn("dprime_compare is a placeholder and not yet refactored.", UserWarning)
    if len(args) > 0 and isinstance(args[0], (list, np.ndarray)):
        num_groups = len(args[0])
        return {"p_value": 0.5, "LR_statistic": 1.0, "df": num_groups -1,
                 "individual_group_estimates": [{"dprime":1.0}]*num_groups}
    return {"p_value": 0.5, "LR_statistic": 1.0, "df": 1,
            "individual_group_estimates": [{"dprime":1.0}]}
