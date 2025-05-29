import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from senspy.discrimination import (
    two_afc,
    triangle_pc,
    duotrio_pc,
    three_afc_pc,
    tetrad_pc,
    hexad_pc,
    twofive_pc,
    get_pguess,
)

__all__ = ["psyfun", "psyinv", "psyderiv", "rescale"]

# Numerical derivative helper (consistent with senspy.discrimination)
try:
    from scipy.derivative import derivative as numerical_derivative_scipy
except ImportError:
    try:
        from scipy.misc import derivative as numerical_derivative_scipy # Older scipy
    except ImportError:
        numerical_derivative_scipy = None

def _numerical_derivative_fallback(func, x0, dx=1e-6):
    """Simple finite difference if scipy.derivative is not available."""
    return (func(x0 + dx) - func(x0 - dx)) / (2 * dx)

def _get_derivative(func, x0, dx=1e-6):
    if numerical_derivative_scipy:
        return numerical_derivative_scipy(func, x0, dx=dx, n=1, order=3) # order=3 is common default
    return _numerical_derivative_fallback(func, x0, dx=dx)


PC_FUNCTIONS = {
    "2afc": two_afc,
    "two_afc": two_afc, # alias
    "triangle": triangle_pc,
    "duotrio": duotrio_pc,
    "3afc": three_afc_pc,
    "three_afc": three_afc_pc, # alias
    "tetrad": tetrad_pc,
    "hexad": hexad_pc,
    "twofive": twofive_pc,
    "2-afc": two_afc, # sensR alias
    "3-afc": three_afc_pc, # sensR alias
    "2-out-of-5": twofive_pc, # sensR alias
}

def psyfun(dprime: float, method: str = "2afc") -> float:
    """
    Map d-prime to proportion correct for a given discrimination method.

    Args:
        dprime (float): Sensitivity index.
        method (str, optional): Discrimination method. Defaults to "2afc".
                                Supported: "2afc", "triangle", "duotrio", "3afc", 
                                "tetrad", "hexad", "twofive".

    Returns:
        float: Proportion correct.
    """
    method_lc = method.lower()
    if method_lc not in PC_FUNCTIONS:
        raise ValueError(f"Unknown method: {method}. Supported methods: {list(PC_FUNCTIONS.keys())}")
    
    pc_func = PC_FUNCTIONS[method_lc]
    # Most pc_func implementations handle dprime <= 0 by returning pguess.
    return pc_func(dprime)


def psyinv(pc: float, method: str = "2afc") -> float:
    """
    Map proportion correct to d-prime for a given discrimination method.

    Args:
        pc (float): Proportion correct.
        method (str, optional): Discrimination method. Defaults to "2afc".

    Returns:
        float: Sensitivity index (d-prime).
    """
    method_lc = method.lower()
    if method_lc not in PC_FUNCTIONS:
        raise ValueError(f"Unknown method: {method}. Supported methods: {list(PC_FUNCTIONS.keys())}")

    pc_func = PC_FUNCTIONS[method_lc]
    pguess = get_pguess(method_lc)

    if not (pguess <= pc <= 1.0):
        # Allow pc slightly below pguess due to sampling, but clip for dprime calc
        if pc < pguess - 1e-9 : # If significantly below pguess
             warnings.warn(f"pc ({pc:.4f}) is below pguess ({pguess:.4f}) for method {method}. Returning d-prime = 0.", UserWarning)
             return 0.0
        # Clip pc to be at least pguess for calculations if it's very slightly below due to float precision
        pc = max(pc, pguess)


    if pc <= pguess: # After potential adjustment above
        return 0.0
    if pc >= 1.0 - 1e-9: # Handle pc effectively being 1.0
        # Check if pc_func can actually reach 1.0. If not, dprime could be very large but finite.
        # For simplicity, return np.inf, or a large number if pc_func has an asymptote < 1.
        # Test with a large dprime.
        if pc_func(20.0) < 1.0 - 1e-9 : # if function asymptotes below 1
            return 20.0 # Return a large finite dprime if pc_func doesn't reach 1
        return np.inf


    # Clip pc for brentq to avoid issues exactly at pguess or 1.0 after checks above
    epsilon_brentq = 1e-9 # Small offset for brentq stability
    pc_adjusted = np.clip(pc, pguess + epsilon_brentq, 1.0 - epsilon_brentq)
    
    # If after adjustment, pc_adjusted is effectively pguess, dprime is 0
    if abs(pc_adjusted - pguess) < epsilon_brentq : # handles cases where pc was extremely close to pguess
        return 0.0

    try:
        # Search interval for d-prime. Most d-primes of interest are 0-15.
        # Ensure the interval brackets the root.
        lower_bound, upper_bound = 0.0, 20.0 # Increased upper bound
        val_at_lower = pc_func(lower_bound) - pc_adjusted
        val_at_upper = pc_func(upper_bound) - pc_adjusted

        if val_at_lower * val_at_upper > 0:
            # Try to expand search if no root in initial interval
            if val_at_upper < 0: # pc_adjusted is higher than pc_func(upper_bound)
                upper_bound = 30.0 
            elif val_at_lower > 0: # pc_adjusted is lower than pc_func(lower_bound)
                 # This case should be handled by pc <= pguess check if lower_bound is 0
                 # but if lower_bound was < 0, this might be relevant.
                 # Since we start search at 0, this means pc_func(0) > pc_adjusted, which implies pc_adjusted < pguess
                 # This should have been caught by pc <= pguess.
                 warnings.warn(f"Cannot bracket root for psyinv: pc_func(0)={pc_func(0):.4f}, pc_adjusted={pc_adjusted:.4f}. Method {method}", UserWarning)
                 return 0.0 # Or handle as error


            dprime_est = brentq(lambda d: pc_func(d) - pc_adjusted, lower_bound, upper_bound, xtol=1e-6, rtol=1e-6)
        else:
            dprime_est = brentq(lambda d: pc_func(d) - pc_adjusted, lower_bound, upper_bound, xtol=1e-6, rtol=1e-6)
        
        return dprime_est
    except ValueError as e:
        # Handle cases where brentq fails (e.g., pc_adjusted is out of achievable range for pc_func)
        warnings.warn(f"brentq failed in psyinv for pc={pc:.4f}, method={method}: {e}. pc_adjusted={pc_adjusted:.4f}. pc_func(0)={pc_func(0):.4f}, pc_func(20)={pc_func(20):.4f}", UserWarning)
        # Fallback: if pc is very high, return large dprime; if very low, return 0.
        if pc_adjusted >= pc_func(upper_bound) - epsilon_brentq : return upper_bound
        if pc_adjusted <= pc_func(lower_bound) + epsilon_brentq : return lower_bound # Should be 0 if lower_bound is 0
        return np.nan # Or raise error


def psyderiv(dprime: float, method: str = "2afc") -> float:
    """
    Derivative of psyfun(dprime, method) with respect to d-prime.

    Args:
        dprime (float): Sensitivity index.
        method (str, optional): Discrimination method. Defaults to "2afc".

    Returns:
        float: Derivative of the psychometric function.
    """
    method_lc = method.lower()
    if method_lc not in PC_FUNCTIONS:
        raise ValueError(f"Unknown method: {method}. Supported methods: {list(PC_FUNCTIONS.keys())}")

    pc_func = PC_FUNCTIONS[method_lc]

    if not np.isfinite(dprime):
        return 0.0 # Derivative at infinity is 0
    if dprime < 0: 
        # Many pc_funcs are flat (equal to pguess) for dprime < 0, so derivative is 0.
        # Or, if they are defined differently, numerical derivative will handle.
        # For safety, if pc_func(0) == pc_func(-0.1) (example), then derivative is 0.
        # This also handles dprime=0 case if the function is flat around 0 for d'<0.
        if abs(pc_func(dprime) - pc_func(max(dprime - 1e-3, dprime*1.1 if dprime<0 else dprime*0.9))) < 1e-9 : # Check if flat
             return 0.0

    # Use numerical derivative
    # dx should be small relative to dprime, but not too small to cause precision loss.
    dx_val = max(abs(dprime) * 1e-5, 1e-7) if dprime != 0 else 1e-7
    
    derivative = _get_derivative(pc_func, dprime, dx=dx_val)
    return derivative


def rescale(x: float, from_scale: str, to_scale: str, method: str = "2afc", std_err: float | None = None) -> dict:
    """
    Rescale a value between proportion correct (pc), proportion discriminated (pd), 
    and d-prime (dp) scales for a given discrimination method. 
    Optionally transforms standard error using the delta method.

    Args:
        x (float): The input value to rescale.
        from_scale (str): The scale of the input value `x`. Must be one of "pc", "pd", "dp".
        to_scale (str): The target scale to convert `x` to. Must be one of "pc", "pd", "dp".
                        (Note: the returned dict contains values on all three scales).
        method (str, optional): The discrimination method context. Defaults to "2afc".
        std_err (float | None, optional): Standard error of the input `x`. 
                                          If provided, SEs for all scales are computed. Defaults to None.

    Returns:
        dict: A dictionary containing:
            - "coefficients": {"pc": value_pc, "pd": value_pd, "d_prime": value_d_prime}
            - "std_errors": {"pc": se_pc, "pd": se_pd, "d_prime": se_d_prime} (if std_err was input, else NaNs)
            - "method": The method used for conversions.
            - "input_scale": The original from_scale.
            - "output_scale": The original to_scale (primarily for context, as all scales are returned).
    """
    method_lc = method.lower()
    if method_lc not in PC_FUNCTIONS:
        raise ValueError(f"Unknown method: {method}. Supported methods: {list(PC_FUNCTIONS.keys())}")

    valid_scales = {"dp", "pc", "pd"}
    if from_scale not in valid_scales or to_scale not in valid_scales:
        raise ValueError(f"Invalid scale. Choose from {list(valid_scales)}")

    pguess = get_pguess(method_lc)

    # Step 1: Convert input `x` to d_prime (dp_intermediate)
    # Also calculate pc and pd values based on input x
    val_pc, val_pd, val_d_prime = np.nan, np.nan, np.nan

    if from_scale == "pc":
        val_pc = x
        val_d_prime = psyinv(val_pc, method=method_lc)
        if val_pc <= pguess: val_pd = 0.0
        else: val_pd = (val_pc - pguess) / (1.0 - pguess) if (1.0 - pguess) != 0 else np.inf
    elif from_scale == "pd":
        val_pd = x
        val_pc = val_pd * (1.0 - pguess) + pguess
        val_d_prime = psyinv(val_pc, method=method_lc)
    elif from_scale == "dp":
        val_d_prime = x
        val_pc = psyfun(val_d_prime, method=method_lc)
        if val_pc <= pguess: val_pd = 0.0
        else: val_pd = (val_pc - pguess) / (1.0 - pguess) if (1.0 - pguess) != 0 else np.inf
    
    # Ensure intermediate d_prime is finite for derivative calculation
    # If d_prime is inf (e.g. from pc=1), derivative is 0, SEs will be inf or handled
    d_prime_for_deriv = val_d_prime 

    # Step 2: Calculate standard errors if std_err is provided
    se_pc, se_pd, se_d_prime = np.nan, np.nan, np.nan

    if std_err is not None:
        if not np.isfinite(std_err) or std_err < 0:
            raise ValueError("Input std_err must be a non-negative finite number.")

        deriv_val = psyderiv(d_prime_for_deriv, method=method_lc)
        
        # Handle derivative being zero or very small (leads to infinite SEs)
        # This often happens at d_prime = 0 for symmetric functions or d_prime = inf
        if not np.isfinite(deriv_val) or abs(deriv_val) < 1e-9:
            se_pc_from_dp = np.inf
            se_d_prime_from_pc = np.inf
        else:
            se_pc_from_dp = deriv_val # This is actually se_pc if se_dprime is 1
            se_d_prime_from_pc = 1.0 / deriv_val # This is se_dprime if se_pc is 1

        if from_scale == "pc":
            se_pc = std_err
            se_d_prime = se_pc * se_d_prime_from_pc # se_pc / deriv_val
            se_pd = se_pc / (1.0 - pguess) if (1.0 - pguess) != 0 else np.inf
        elif from_scale == "pd":
            se_pd = std_err
            se_pc = se_pd * (1.0 - pguess)
            se_d_prime = se_pc * se_d_prime_from_pc # se_pc / deriv_val
        elif from_scale == "dp":
            se_d_prime = std_err
            se_pc = se_d_prime * se_pc_from_dp # se_dprime * deriv_val
            se_pd = se_pc / (1.0 - pguess) if (1.0 - pguess) != 0 else np.inf
            
    result_val_on_to_scale = {
        "pc": val_pc, "pd": val_pd, "dp": val_d_prime
    }.get(to_scale)

    return {
        # "value_on_to_scale": result_val_on_to_scale, # Retaining this for clarity if needed by caller
        "coefficients": {"pc": val_pc, "pd": val_pd, "d_prime": val_d_prime},
        "std_errors": {"pc": se_pc, "pd": se_pd, "d_prime": se_d_prime},
        "method": method_lc, # Store the method used
        "input_scale": from_scale,
        "output_scale": to_scale 
    }
