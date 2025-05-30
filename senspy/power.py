import numpy as np
from scipy.stats import binom, norm # Added norm for power_discrim_normal_approx if needed by z-scores
import warnings
from senspy.discrimination import psyfun # Assuming psyfun is in discrimination.py
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize
from collections import namedtuple

# Define PowerResult namedtuple
PowerResult = namedtuple("PowerResult", ["power", "n_trials", "alpha_level", "method_type", "details"])
# details can be a dict for p_null, p_alt, alternative, etc.

__all__ = ["find_critical_binomial_value", "exact_binomial_power", 
           "sample_size_for_binomial_power", "power_discrim",
           "power_discrim_normal_approx", "PowerResult", "sample_size_discrim"]

def find_critical_binomial_value(n_trials: int, p_null: float, alpha_level: float, alternative: str = "greater") -> int:
    """
    Determines the critical number of successes (k) required to reject the null 
    hypothesis in a one-sample binomial test. Analogous to sensR::findcr.
    """
    if not isinstance(n_trials, int) or n_trials <= 0:
        raise ValueError("n_trials must be a positive integer.")
    if not (isinstance(p_null, (float, int, np.number)) and 0 <= p_null <= 1):
        raise ValueError("p_null must be a float or int between 0 and 1 (inclusive).")
    if p_null == 0 and alternative.lower() == "greater": return 1 if alpha_level > 0 else n_trials + 1 
    if p_null == 1 and alternative.lower() == "less": return n_trials - 1 if alpha_level > 0 else -1
    if p_null == 1 and alternative.lower() == "greater": return n_trials + 1
    if p_null == 0 and alternative.lower() == "less": return -1
    if not (isinstance(alpha_level, float) and 0 < alpha_level < 1):
        raise ValueError("alpha_level must be a float strictly between 0 and 1.")
    alternative = alternative.lower()
    if alternative not in ["greater", "less"]:
        if alternative == "two.sided": raise NotImplementedError("Alternative 'two.sided' is not implemented.")
        else: raise ValueError("alternative must be 'greater' or 'less'.")

    critical_value: int
    if alternative == "greater":
        # Smallest k such that P(X >= k | p_null) <= alpha_level
        # which is binom.sf(k-1, n, p_null) <= alpha_level
        for k_val in range(n_trials + 2): # k can range from 0 to n_trials+1 (for power=0 case)
            if k_val == 0: # P(X >= 0) is 1 unless n_trials=0 (handled)
                current_pval = 1.0
            else:
                current_pval = binom.sf(k_val - 1, n_trials, p_null)

            if current_pval <= alpha_level:
                critical_value = k_val
                break
        else: # Should not be reached if logic is correct, implies alpha_level too small or n_trials too small
            critical_value = n_trials + 1
    else: # less
        # Smallest k such that P(X <= k | p_null) <= alpha_level
        for k_val in range(n_trials, -2, -1): # k can range from n_trials down to -1 (for power=0 case)
            if k_val < 0: prob_le_k = 0.0 # P(X <= -1) is 0
            elif k_val >= n_trials: prob_le_k = 1.0 # P(X <= n_trials) is 1
            else: prob_le_k = binom.cdf(k_val, n_trials, p_null)

            if prob_le_k <= alpha_level:
                critical_value = k_val
                break
        else: # Should not be reached
            critical_value = -1
    return critical_value

def exact_binomial_power(n_trials: int, p_alt: float, alpha_level: float, p_null: float = 0.5, alternative: str = "greater") -> PowerResult:
    if not isinstance(n_trials, int) or n_trials <= 0: raise ValueError("n_trials must be a positive integer.")
    if not (0 <= p_alt <= 1): raise ValueError("p_alt must be between 0 and 1.")
    if not (0 < alpha_level < 1): raise ValueError("alpha_level must be between 0 and 1.")
    if not (0 <= p_null <= 1): raise ValueError("p_null must be between 0 and 1.")
    alternative_lc = alternative.lower()
    if alternative_lc not in ["greater", "less"]: raise ValueError("alternative must be 'greater' or 'less'.")
    
    calculated_power: float
    if p_null == 0 and alternative_lc == "greater":
        calculated_power = 1.0 - (1 - p_alt)**n_trials if alpha_level > 0 else 0.0
    elif p_null == 1 and alternative_lc == "less":
        calculated_power = 1.0 - p_alt**n_trials if alpha_level > 0 else 0.0
    elif p_null == 1 and alternative_lc == "greater":
        calculated_power = 0.0
    elif p_null == 0 and alternative_lc == "less":
        calculated_power = 0.0
    else:
        crit_val = find_critical_binomial_value(n_trials, p_null, alpha_level, alternative_lc)
        if alternative_lc == "greater":
            if crit_val > n_trials: calculated_power = 0.0
            elif crit_val <= 0: calculated_power = 1.0 # If k_crit is 0 (or less), any success means rejection.
            else: calculated_power = binom.sf(crit_val - 1, n_trials, p_alt)
        else: # less
            if crit_val < 0: calculated_power = 0.0 # If k_crit is < 0, no result leads to rejection
            elif crit_val >= n_trials: calculated_power = 1.0 # If k_crit is n (or more), any non-perfect result means rejection
            else: calculated_power = binom.cdf(crit_val, n_trials, p_alt)

    return PowerResult(
        power=calculated_power,
        n_trials=n_trials,
        alpha_level=alpha_level,
        method_type="exact_binomial",
        details={"p_null": p_null, "p_alt": p_alt, "alternative": alternative_lc}
    )

def power_discrim_normal_approx(d_prime_alt: float, n_trials: int | None, method: str, 
                                d_prime_null: float = 0.0, alpha_level: float = 0.05, 
                                alternative: str = "greater", power_target: float | None = None) -> PowerResult | int | None:
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
            n_trials = None # Ensure n_trials is None if power_target is given
    else: 
        if not (isinstance(n_trials, int) and n_trials > 0):
            raise ValueError("n_trials must be a positive integer for power calculation.")

    pc_alt = psyfun(d_prime_alt, method=method)
    pc_null = psyfun(d_prime_null, method=method)

    pc_alt_clipped = np.clip(pc_alt, 1e-9, 1.0 - 1e-9)
    pc_null_clipped = np.clip(pc_null, 1e-9, 1.0 - 1e-9)
    
    effect_size = proportion_effectsize(pc_alt_clipped, pc_null_clipped)
    calculated_power: float | None = None

    if not np.isfinite(effect_size) or abs(effect_size) < 1e-9 :
        is_zero_effect = abs(effect_size) < 1e-9
        warn_msg = f"Effect size is {'zero' if is_zero_effect else 'non-finite'} ({effect_size:.4f}). pc_alt={pc_alt:.4f}, pc_null={pc_null:.4f}."
        warnings.warn(warn_msg, UserWarning)
        if power_target is not None:
            return np.inf if is_zero_effect and power_target > alpha_level else np.nan 
        if is_zero_effect:
            calculated_power = alpha_level # Power is alpha if effect size is zero
        else:
            calculated_power = np.nan

    alternative_sm_map = {"greater": "larger", "less": "smaller", "two-sided": "two-sided"}
    alternative_sm = alternative_sm_map[alternative_lc]
    power_calculator = NormalIndPower()

    if power_target is None: # Calculate power
        if calculated_power is None: # if not set by zero/non-finite effect_size block
            calculated_power = power_calculator.power(
                effect_size=effect_size, nobs1=n_trials, alpha=alpha_level,
                ratio=0, alternative=alternative_sm # ratio=0 for one-sample
            )
        return PowerResult(
            power=calculated_power,
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
            n_calculated = power_calculator.solve_power(
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
                                 min_n: int = 5, max_n: int = 10000) -> int | None:
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
        # exact_binomial_power now returns a PowerResult object
        power_result = exact_binomial_power(n_trials_current, p_alt, alpha_level, p_null, alternative_lc)
        if power_result.power >= target_power:
            return n_trials_current
    warnings.warn(f"Target power {target_power} not reached within max_n {max_n} trials.", UserWarning)
    return None

def power_discrim(d_prime_alt: float, n_trials: int, method: str, 
                  d_prime_null: float = 0.0, alpha_level: float = 0.05, 
                  alternative: str = "greater", **kwargs) -> PowerResult:
    if d_prime_alt < 0: raise ValueError("d_prime_alt must be non-negative.")
    if d_prime_null < 0: raise ValueError("d_prime_null must be non-negative.")
    if n_trials <= 0: raise ValueError("n_trials must be positive.")
    if not (0 < alpha_level < 1): raise ValueError("alpha_level must be between 0 and 1.")
    alternative_lc = alternative.lower()
    if alternative_lc not in ["greater", "less"]: raise ValueError("alternative must be 'greater' or 'less'.")

    pc_alt = psyfun(d_prime_alt, method=method)
    pc_null = psyfun(d_prime_null, method=method)

    if (alternative_lc == "greater" and pc_alt <= pc_null) or \
       (alternative_lc == "less" and pc_alt >= pc_null):
        warnings.warn(f"For alternative='{alternative_lc}', pc_alt ({pc_alt:.4f}) is not more extreme than pc_null ({pc_null:.4f}). Power may be low.", UserWarning)

    # Call exact_binomial_power which returns a PowerResult
    binomial_power_result = exact_binomial_power(n_trials, pc_alt, alpha_level, pc_null, alternative_lc)

    # Adapt the details for discrimination context
    updated_details = {
        "d_prime_alt": d_prime_alt, "d_prime_null": d_prime_null,
        "pc_alt": pc_alt, "pc_null": pc_null,
        "method_protocol": method, "alternative": alternative_lc,
        "original_binomial_details": binomial_power_result.details # Nest original details
    }

    return PowerResult(
        power=binomial_power_result.power,
        n_trials=binomial_power_result.n_trials,
        alpha_level=binomial_power_result.alpha_level,
        method_type="discrim_exact_binomial", # More specific type
        details=updated_details
    )

def sample_size_discrim(d_prime_alt: float, target_power: float, method: str,
                        d_prime_null: float = 0.0, alpha_level: float = 0.05,
                        alternative: str = "greater", min_n: int = 5, max_n: int = 10000) -> int | None:
    """
    Calculates the sample size required for a discrimination task to achieve a target power.
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
