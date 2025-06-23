import numpy as np
from scipy.stats import binom, norm
import warnings
from senspy.discrimination import psyfun
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize
from collections import namedtuple
from typing import Union, Optional, Dict, Any

# Define PowerResult namedtuple
PowerResult = namedtuple("PowerResult", ["power", "n_trials", "alpha_level", "method_type", "details"])
"""
A namedtuple to store results from power calculations.

Attributes:
    power (float): The calculated statistical power (0.0 to 1.0).
    n_trials (int): The number of trials used for the power calculation.
    alpha_level (float): The significance level (alpha) used.
    method_type (str): A string indicating the method used for power calculation
                       (e.g., "exact_binomial", "discrim_exact_binomial", "discrim_normal_approx").
    details (dict): A dictionary containing other relevant parameters and intermediate
                    calculations specific to the power method used.
"""

__all__ = ["find_critical_binomial_value", "exact_binomial_power", 
           "sample_size_for_binomial_power", "power_discrim",
           "power_discrim_normal_approx", "PowerResult", "sample_size_discrim"]

def find_critical_binomial_value(n_trials: int, p_null: float, alpha_level: float, alternative: str = "greater") -> int:
    """
    Determines the critical number of successes (k) required to reject the null 
    hypothesis in a one-sample binomial test.

    This function is analogous to `sensR::findcr`.

    Args:
        n_trials (int): The total number of trials. Must be a positive integer.
        p_null (float): The proportion of successes under the null hypothesis.
                        Must be between 0 and 1.
        alpha_level (float): The desired significance level (Type I error rate).
                             Must be between 0 and 1.
        alternative (str, optional): Specifies the alternative hypothesis.
                                     Must be 'greater' (default) or 'less'.
                                     'two.sided' is not currently implemented.

    Returns:
        int: The critical number of successes. If the calculated critical value
             is outside the possible range of successes (0 to n_trials), it may
             return values like n_trials + 1 (for 'greater' alternative if no k
             achieves alpha) or -1 (for 'less' alternative).

    Raises:
        ValueError: If inputs are invalid (e.g., n_trials <= 0, p_null or alpha_level
                    outside [0,1], invalid alternative).
        NotImplementedError: If `alternative` is 'two.sided'.
    """
    if not isinstance(n_trials, int) or n_trials <= 0:
        raise ValueError("n_trials must be a positive integer.")
    if not (isinstance(p_null, (float, int, np.number)) and 0 <= p_null <= 1):
        raise ValueError("p_null must be a float or int between 0 and 1 (inclusive).")
    if p_null == 0 and alternative.lower() == "greater": return 1 if alpha_level > 0 else n_trials + 1 
    if p_null == 1 and alternative.lower() == "less": return n_trials - 1 if alpha_level > 0 else -1
    if p_null == 1 and alternative.lower() == "greater": return n_trials + 1 # Cannot be greater than 1
    if p_null == 0 and alternative.lower() == "less": return -1 # Cannot be less than 0
    if not (isinstance(alpha_level, float) and 0 < alpha_level < 1):
        raise ValueError("alpha_level must be a float strictly between 0 and 1.")
    alternative = alternative.lower()
    if alternative not in ["greater", "less"]:
        if alternative == "two.sided": raise NotImplementedError("Alternative 'two.sided' is not implemented.")
        else: raise ValueError("alternative must be 'greater' or 'less'.")

    critical_value: int
    if alternative == "greater":
        for k_val in range(n_trials + 2):
            if k_val == 0: current_pval = 1.0
            else: current_pval = binom.sf(k_val - 1, n_trials, p_null)
            if current_pval <= alpha_level:
                critical_value = k_val; break
        else: critical_value = n_trials + 1
    else: # less
        for k_val in range(n_trials, -2, -1):
            if k_val < 0: prob_le_k = 0.0
            elif k_val >= n_trials: prob_le_k = 1.0
            else: prob_le_k = binom.cdf(k_val, n_trials, p_null)
            if prob_le_k <= alpha_level:
                critical_value = k_val; break
        else: critical_value = -1
    return critical_value

def exact_binomial_power(n_trials: int, p_alt: float, alpha_level: float, p_null: float = 0.5, alternative: str = "greater") -> PowerResult:
    """
    Calculates the exact power for a one-sample binomial test.

    Args:
        n_trials (int): The total number of trials.
        p_alt (float): The proportion of successes under the alternative hypothesis.
        alpha_level (float): The significance level (Type I error rate).
        p_null (float, optional): The proportion of successes under the null
                                 hypothesis. Defaults to 0.5.
        alternative (str, optional): The alternative hypothesis ('greater' or 'less').
                                     Defaults to "greater".

    Returns:
        PowerResult: A namedtuple containing the calculated power, n_trials,
                     alpha_level, method_type ("exact_binomial"), and a details
                     dictionary with p_null, p_alt, and alternative.

    Raises:
        ValueError: If inputs are invalid.
    """
    if not isinstance(n_trials, int) or n_trials <= 0: raise ValueError("n_trials must be a positive integer.")
    if not (0 <= p_alt <= 1): raise ValueError("p_alt must be between 0 and 1.")
    if not (0 < alpha_level < 1): raise ValueError("alpha_level must be between 0 and 1.")
    if not (0 <= p_null <= 1): raise ValueError("p_null must be between 0 and 1.")
    alternative_lc = alternative.lower()
    if alternative_lc not in ["greater", "less"]: raise ValueError("alternative must be 'greater' or 'less'.")
    
    calculated_power: float
    # Handle edge cases for p_null to avoid issues with find_critical_binomial_value
    if p_null == 0 and alternative_lc == "greater":
        # If p_null is 0, any success (k>=1) leads to rejection if alpha > 0.
        # Power = P(X>=1 | p_alt) = 1 - P(X=0 | p_alt) = 1 - (1-p_alt)^n
        calculated_power = 1.0 - (1.0 - p_alt)**n_trials if alpha_level > 0 else 0.0
    elif p_null == 1 and alternative_lc == "less":
        # If p_null is 1, any non-success (k<n) leads to rejection if alpha > 0.
        # Power = P(X < n | p_alt) = 1 - P(X=n | p_alt) = 1 - p_alt^n
        calculated_power = 1.0 - p_alt**n_trials if alpha_level > 0 else 0.0
    elif p_null == 1 and alternative_lc == "greater": # Cannot be greater than 1
        calculated_power = 0.0
    elif p_null == 0 and alternative_lc == "less": # Cannot be less than 0
        calculated_power = 0.0
    else:
        crit_val = find_critical_binomial_value(n_trials, p_null, alpha_level, alternative_lc)
        if alternative_lc == "greater":
            if crit_val > n_trials: calculated_power = 0.0 # Critical value is impossible to reach
            elif crit_val <= 0: calculated_power = 1.0
            else: calculated_power = binom.sf(crit_val - 1, n_trials, p_alt)
        else: # less
            if crit_val < 0: calculated_power = 0.0 # Critical value is impossible to reach
            elif crit_val >= n_trials: calculated_power = 1.0
            else: calculated_power = binom.cdf(crit_val, n_trials, p_alt)

    return PowerResult(
        power=float(calculated_power),
        n_trials=n_trials,
        alpha_level=alpha_level,
        method_type="exact_binomial",
        details={"p_null": p_null, "p_alt": p_alt, "alternative": alternative_lc}
    )

def power_discrim_normal_approx(
    d_prime_alt: float,
    n_trials: Optional[int],
    method: str,
    d_prime_null: float = 0.0,
    alpha_level: float = 0.05,
    alternative: str = "greater",
    power_target: Optional[float] = None
) -> Union[PowerResult, int, None, float]: # Added float for np.nan case
    """
    Calculates power or sample size for a discrimination task using normal approximation.

    Args:
        d_prime_alt (float): d-prime under the alternative hypothesis.
        n_trials (Optional[int]): Number of trials. Required if calculating power.
                                 Ignored if `power_target` is set.
        method (str): The sensory discrimination method (e.g., "2afc", "triangle").
        d_prime_null (float, optional): d-prime under the null hypothesis. Defaults to 0.0.
        alpha_level (float, optional): Significance level. Defaults to 0.05.
        alternative (str, optional): Alternative hypothesis ('greater', 'less', 'two-sided').
                                   Defaults to "greater".
        power_target (Optional[float], optional): Desired power. If set, calculates sample size.
                                               Defaults to None (calculates power).

    Returns:
        Union[PowerResult, int, None, float]:
        - If `power_target` is None: A `PowerResult` namedtuple containing the
          calculated power, n_trials, alpha_level, method_type ("discrim_normal_approx"),
          and a details dictionary.
        - If `power_target` is not None: The required sample size (int), or np.nan/np.inf
          if not achievable or error.

    Raises:
        ValueError: If inputs are invalid.
    """
    if not isinstance(d_prime_alt, (float, int, np.number)) or d_prime_alt < 0:
        raise ValueError("d_prime_alt must be a non-negative numeric value.")
    if not isinstance(d_prime_null, (float, int, np.number)) or d_prime_null < 0:
        raise ValueError("d_prime_null must be a non-negative numeric value.")
    if not (isinstance(alpha_level, float) and 0 < alpha_level < 1):
        raise ValueError("alpha_level must be a float strictly between 0 and 1.")
    
    alternative_lc = alternative.lower()
    if alternative_lc not in ["greater", "less", "two-sided"]:
        raise ValueError("alternative must be 'greater', 'less', or 'two-sided'.")

    if power_target is not None:
        if not (isinstance(power_target, float) and 0 < power_target < 1):
            raise ValueError("power_target must be a float strictly between 0 and 1.")
        if n_trials is not None:
            warnings.warn("n_trials is ignored when power_target is specified.", UserWarning)
            n_trials = None
    else: 
        if not (isinstance(n_trials, int) and n_trials > 0):
            raise ValueError("n_trials must be a positive integer for power calculation.")

    pc_alt = psyfun(d_prime_alt, method=method)
    pc_null = psyfun(d_prime_null, method=method)

    pc_alt_clipped = np.clip(pc_alt, 1e-9, 1.0 - 1e-9)
    pc_null_clipped = np.clip(pc_null, 1e-9, 1.0 - 1e-9)
    
    effect_size = proportion_effectsize(pc_alt_clipped, pc_null_clipped)
    calculated_power_value: Optional[float] = None # Explicitly Optional[float]

    if not np.isfinite(effect_size) or abs(effect_size) < 1e-9 :
        is_zero_effect = abs(effect_size) < 1e-9
        warn_msg = f"Effect size is {'zero' if is_zero_effect else 'non-finite'} ({effect_size:.4f}). pc_alt={pc_alt:.4f}, pc_null={pc_null:.4f}."
        warnings.warn(warn_msg, UserWarning)
        if power_target is not None: # Sample size calculation
            return np.inf if is_zero_effect and power_target > alpha_level else np.nan 
        # Power calculation
        if is_zero_effect: calculated_power_value = alpha_level
        else: calculated_power_value = np.nan

    alternative_sm_map = {"greater": "larger", "less": "smaller", "two-sided": "two-sided"}
    alternative_sm = alternative_sm_map[alternative_lc]

    if power_target is None: # Calculate power
        if calculated_power_value is None: # if not set by zero/non-finite effect_size block
            calculated_power_value = NormalIndPower().power(
                effect_size=effect_size, nobs1=n_trials, alpha=alpha_level,
                ratio=0, alternative=alternative_sm
            )
        return PowerResult(
            power=float(calculated_power_value) if calculated_power_value is not None else np.nan,
            n_trials=n_trials,
            alpha_level=alpha_level,
            method_type="discrim_normal_approx",
            details={
                "d_prime_alt": d_prime_alt, "d_prime_null": d_prime_null,
                "pc_alt": pc_alt, "pc_null": pc_null,
                "effect_size": effect_size, "method_protocol": method,
                "alternative": alternative_lc
            }
        )
    else: # Calculate sample size
        try:
            n_calculated = NormalIndPower().solve_power(
                effect_size=effect_size, nobs1=None, alpha=alpha_level, 
                power=power_target, ratio=0, alternative=alternative_sm
            )
            if not np.isfinite(n_calculated):
                warnings.warn(f"Sample size calculation resulted in non-finite value ({n_calculated}).", UserWarning)
                return np.nan 
            return np.ceil(n_calculated).astype(int)
        except Exception as e:
            warnings.warn(f"statsmodels solve_power failed: {e}. Check parameters.", UserWarning)
            return np.nan

def sample_size_for_binomial_power(p_alt: float, target_power: float, alpha_level: float, 
                                 p_null: float = 0.5, alternative: str = "greater", 
                                 min_n: int = 5, max_n: int = 10000) -> Optional[int]:
    """
    Finds the smallest sample size (n_trials) for a desired statistical power
    in a one-sample binomial test using an iterative search.

    Args:
        p_alt (float): Proportion of successes under the alternative hypothesis.
        target_power (float): Desired statistical power (e.g., 0.80 for 80%).
        alpha_level (float): Significance level (Type I error rate).
        p_null (float, optional): Proportion of successes under the null hypothesis.
                                 Defaults to 0.5.
        alternative (str, optional): Alternative hypothesis ('greater' or 'less').
                                     Defaults to "greater".
        min_n (int, optional): Minimum number of trials to check. Defaults to 5.
        max_n (int, optional): Maximum number of trials to check. Defaults to 10000.

    Returns:
        Optional[int]: The smallest sample size (n_trials) that achieves the target
                       power, or None if not achieved within `max_n`.

    Raises:
        ValueError: If inputs are invalid.
    """
    if not (0 <= p_alt <= 1): raise ValueError("p_alt must be between 0 and 1.")
    if not (0 < target_power < 1): raise ValueError("target_power must be between 0 and 1.")
    if not (0 < alpha_level < 1): raise ValueError("alpha_level must be between 0 and 1.")
    if not (0 <= p_null <= 1): raise ValueError("p_null must be between 0 and 1.")
    alternative_lc = alternative.lower()
    if alternative_lc not in ["greater", "less"]: raise ValueError("alternative must be 'greater' or 'less'.")

    if (alternative_lc == "greater" and p_alt <= p_null) or \
       (alternative_lc == "less" and p_alt >= p_null):
        warnings.warn(f"p_alt ({p_alt}) not more extreme than p_null ({p_null}) for alternative='{alternative_lc}'. Target power may not be achievable.", UserWarning)

    for n_trials_current in range(min_n, max_n + 1):
        power_result = exact_binomial_power(n_trials_current, p_alt, alpha_level, p_null, alternative_lc)
        if power_result.power >= target_power:
            return n_trials_current
    warnings.warn(f"Target power {target_power} not reached within max_n {max_n} trials.", UserWarning)
    return None

def power_discrim(d_prime_alt: float, n_trials: int, method: str, 
                  d_prime_null: float = 0.0, alpha_level: float = 0.05, 
                  alternative: str = "greater", **kwargs: Any) -> PowerResult:
    """
    Calculates statistical power for detecting d-prime in a discrimination task.

    This function uses an exact binomial test on the proportion correct scale.

    Args:
        d_prime_alt (float): d-prime under the alternative hypothesis.
        n_trials (int): Total number of trials.
        method (str): Sensory discrimination method (e.g., "2afc", "triangle").
        d_prime_null (float, optional): d-prime under the null hypothesis. Defaults to 0.0.
        alpha_level (float, optional): Significance level. Defaults to 0.05.
        alternative (str, optional): Alternative hypothesis ('greater' or 'less').
                                     Defaults to "greater".
        **kwargs: Additional keyword arguments (currently ignored).

    Returns:
        PowerResult: A namedtuple containing the calculated power, n_trials,
                     alpha_level, method_type ("discrim_exact_binomial"), and a
                     details dictionary including d-primes, pc_values, and method info.

    Raises:
        ValueError: If inputs are invalid.
    """
    if d_prime_alt < 0: raise ValueError("d_prime_alt must be non-negative.")
    if d_prime_null < 0: raise ValueError("d_prime_null must be non-negative.")
    if not isinstance(n_trials, int) or n_trials <= 0: raise ValueError("n_trials must be a positive integer.")
    if not (0 < alpha_level < 1): raise ValueError("alpha_level must be between 0 and 1.")
    alternative_lc = alternative.lower()
    if alternative_lc not in ["greater", "less"]: raise ValueError("alternative must be 'greater' or 'less'.")

    pc_alt = psyfun(d_prime_alt, method=method)
    pc_null = psyfun(d_prime_null, method=method)

    if (alternative_lc == "greater" and pc_alt <= pc_null) or \
       (alternative_lc == "less" and pc_alt >= pc_null):
        warnings.warn(f"For alternative='{alternative_lc}', pc_alt ({pc_alt:.4f}) is not more extreme than pc_null ({pc_null:.4f}). Power may be low.", UserWarning)

    binomial_power_result = exact_binomial_power(n_trials, pc_alt, alpha_level, pc_null, alternative_lc)

    updated_details = {
        "d_prime_alt": d_prime_alt, "d_prime_null": d_prime_null,
        "pc_alt": pc_alt, "pc_null": pc_null,
        "method_protocol": method, "alternative": alternative_lc,
        "original_binomial_details": binomial_power_result.details
    }

    return PowerResult(
        power=binomial_power_result.power,
        n_trials=binomial_power_result.n_trials,
        alpha_level=binomial_power_result.alpha_level,
        method_type="discrim_exact_binomial",
        details=updated_details
    )

def sample_size_discrim(d_prime_alt: float, target_power: float, method: str,
                        d_prime_null: float = 0.0, alpha_level: float = 0.05,
                        alternative: str = "greater", min_n: int = 5, max_n: int = 10000) -> Optional[int]:
    """
    Calculates sample size for a discrimination task to achieve target power.

    This function converts d-primes to proportions correct and then uses an
    exact binomial method to find the required sample size.

    Args:
        d_prime_alt (float): d-prime under the alternative hypothesis.
        target_power (float): Desired statistical power (e.g., 0.80).
        method (str): Sensory discrimination method (e.g., "2afc", "triangle").
        d_prime_null (float, optional): d-prime under the null hypothesis. Defaults to 0.0.
        alpha_level (float, optional): Significance level. Defaults to 0.05.
        alternative (str, optional): Alternative hypothesis ('greater' or 'less').
                                     Defaults to "greater".
        min_n (int, optional): Minimum number of trials to consider. Defaults to 5.
        max_n (int, optional): Maximum number of trials to consider. Defaults to 10000.

    Returns:
        Optional[int]: The smallest sample size (n_trials) that achieves target power,
                       or None if not achieved within `max_n`.

    Raises:
        ValueError: If inputs are invalid (propagated from `psyfun` or
                    `sample_size_for_binomial_power`).
    """
    pc_alt = psyfun(d_prime_alt, method=method)
    pc_null = psyfun(d_prime_null, method=method)

    return sample_size_for_binomial_power(
        p_alt=pc_alt,
        target_power=target_power,
        alpha_level=alpha_level,
        p_null=pc_null,
        alternative=alternative,
        min_n=min_n,
        max_n=max_n
    )
