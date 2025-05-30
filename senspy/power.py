import numpy as np
from scipy.stats import binom
import warnings
from senspy.discrimination import psyfun
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

__all__ = ["find_critical_binomial_value", "exact_binomial_power", 
           "sample_size_for_binomial_power", "power_discrim",
           "power_discrim_normal_approx"]

def find_critical_binomial_value(n_trials: int, p_null: float, alpha_level: float, alternative: str = "greater") -> int:
    """
    Determines the critical number of successes (k) required to reject the null 
    hypothesis in a one-sample binomial test. Analogous to sensR::findcr.
    """
    # Input Validation (omitted for brevity, assumed correct from previous state)
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
        for k_val in range(n_trials + 2): 
            if binom.sf(k_val - 1, n_trials, p_null) <= alpha_level:
                critical_value = k_val; break
        else: critical_value = n_trials + 1 
    else: # less
        for k_val in range(n_trials, -2, -1): 
            if k_val < 0: prob_le_k = 0.0
            else: prob_le_k = binom.cdf(k_val, n_trials, p_null)
            if prob_le_k <= alpha_level:
                critical_value = k_val; break
        else: critical_value = -1 
    return critical_value

def exact_binomial_power(n_trials: int, p_alt: float, alpha_level: float, p_null: float = 0.5, alternative: str = "greater") -> float:
    """
    Calculates the exact power for a one-sample binomial test.
    """
    # Validations omitted for brevity, assumed correct
    if not isinstance(n_trials, int) or n_trials <= 0: raise ValueError("n_trials must be a positive integer.")
    if not (0 <= p_alt <= 1): raise ValueError("p_alt must be between 0 and 1.")
    if not (0 < alpha_level < 1): raise ValueError("alpha_level must be between 0 and 1.")
    if not (0 <= p_null <= 1): raise ValueError("p_null must be between 0 and 1.")
    alternative = alternative.lower()
    if alternative not in ["greater", "less"]: raise ValueError("alternative must be 'greater' or 'less'.")
    
    if p_null == 0 and alternative == "greater": return 1.0 - (1 - p_alt)**n_trials if alpha_level > 0 else 0.0
    if p_null == 1 and alternative == "less": return 1.0 - p_alt**n_trials if alpha_level > 0 else 0.0
    if p_null == 1 and alternative == "greater": return 0.0
    if p_null == 0 and alternative == "less": return 0.0

    crit_val = find_critical_binomial_value(n_trials, p_null, alpha_level, alternative)
    if alternative == "greater":
        if crit_val > n_trials: return 0.0
        elif crit_val <= 0: return 1.0
        else: return binom.sf(crit_val - 1, n_trials, p_alt)
    else: # less
        if crit_val < 0: return 0.0
        elif crit_val >= n_trials: return 1.0
        else: return binom.cdf(crit_val, n_trials, p_alt)

def power_discrim_normal_approx(d_prime_alt: float, n_trials: int | None, method: str, 
                                d_prime_null: float = 0.0, alpha_level: float = 0.05, 
                                alternative: str = "greater", power_target: float | None = None) -> float | int | None:
    if not isinstance(d_prime_alt, (float, int, np.number)) or d_prime_alt < 0:
        raise ValueError("d_prime_alt must be a non-negative numeric value.")
    if not isinstance(d_prime_null, (float, int, np.number)) or d_prime_null < 0:
        raise ValueError("d_prime_null must be a non-negative numeric value.")
    if not (isinstance(alpha_level, float) and 0 < alpha_level < 1):
        raise ValueError("alpha_level must be a float strictly between 0 and 1.")
    
    alternative = alternative.lower()
    if alternative not in ["greater", "less", "two-sided"]:
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
    
    # Using Cohen's h as per original implementation plan
    effect_size = proportion_effectsize(pc_alt_clipped, pc_null_clipped)

    if not np.isfinite(effect_size) or abs(effect_size) < 1e-9 : # Also check for near-zero effect
        is_zero_effect = abs(effect_size) < 1e-9
        warn_msg = f"Effect size is {'zero' if is_zero_effect else 'non-finite'} ({effect_size:.4f}). pc_alt={pc_alt:.4f}, pc_null={pc_null:.4f}."
        warnings.warn(warn_msg, UserWarning)
        if power_target is not None:
            return np.inf if is_zero_effect and power_target > alpha_level else np.nan 
        # For power calculation:
        if is_zero_effect:
            if alternative == "two-sided": return alpha_level
            if alternative == "greater" and pc_alt_clipped <= pc_null_clipped: return alpha_level
            if alternative == "less" and pc_alt_clipped >= pc_null_clipped: return alpha_level
            # If effect is zero but one-sided test is in a "favorable" direction (e.g. greater and pc_alt > pc_null, though effect_size is ~0)
            # this case implies pc_alt is extremely close to pc_null. Power should be alpha.
            return alpha_level 
        return np.nan # Non-finite, non-zero effect_size

    alternative_sm_map = {"greater": "larger", "less": "smaller", "two-sided": "two-sided"}
    alternative_sm = alternative_sm_map[alternative]
    power_calculator = NormalIndPower()

    if power_target is None:
        power = power_calculator.power(
            effect_size=effect_size, nobs1=n_trials, alpha=alpha_level, 
            ratio=0, alternative=alternative_sm
        )
        return power
    else:
        try:
            # Using ratio=0 for solve_power as well, assuming it's for a one-sample context.
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
    """
    Finds the smallest sample size (n_trials) required to achieve a desired 
    statistical power for a one-sample binomial test using an iterative search.
    """
    # Validations omitted for brevity
    if not (0 <= p_alt <= 1): raise ValueError("p_alt must be between 0 and 1.")
    if not (0 < target_power < 1): raise ValueError("target_power must be between 0 and 1.")
    if not (0 < alpha_level < 1): raise ValueError("alpha_level must be between 0 and 1.")
    if not (0 <= p_null <= 1): raise ValueError("p_null must be between 0 and 1.")
    alternative = alternative.lower()
    if alternative not in ["greater", "less"]: raise ValueError("alternative must be 'greater' or 'less'.")

    if (alternative == "greater" and p_alt <= p_null) or \
       (alternative == "less" and p_alt >= p_null):
        warnings.warn(f"p_alt ({p_alt}) not more extreme than p_null ({p_null}) for alternative='{alternative}'. Target power may not be achievable.", UserWarning)

    for n_trials_current in range(min_n, max_n + 1):
        current_power = exact_binomial_power(n_trials_current, p_alt, alpha_level, p_null, alternative)
        if current_power >= target_power:
            return n_trials_current
    warnings.warn(f"Target power {target_power} not reached within max_n {max_n} trials.", UserWarning)
    return None

def power_discrim(d_prime_alt: float, n_trials: int, method: str, 
                  d_prime_null: float = 0.0, alpha_level: float = 0.05, 
                  alternative: str = "greater", **kwargs) -> float:
    """
    Calculates the statistical power for detecting a given d-prime in a sensory
    discrimination task.
    """
    # Validations omitted for brevity
    if d_prime_alt < 0: raise ValueError("d_prime_alt must be non-negative.")
    if d_prime_null < 0: raise ValueError("d_prime_null must be non-negative.")
    if n_trials <= 0: raise ValueError("n_trials must be positive.")
    if not (0 < alpha_level < 1): raise ValueError("alpha_level must be between 0 and 1.")
    alternative = alternative.lower()
    if alternative not in ["greater", "less"]: raise ValueError("alternative must be 'greater' or 'less'.")

    pc_alt = psyfun(d_prime_alt, method=method)
    pc_null = psyfun(d_prime_null, method=method)

    if (alternative == "greater" and pc_alt <= pc_null) or \
       (alternative == "less" and pc_alt >= pc_null):
        warnings.warn(f"For alternative='{alternative}', pc_alt ({pc_alt:.4f}) is not more extreme than pc_null ({pc_null:.4f}). Power may be low.", UserWarning)

    return exact_binomial_power(n_trials, pc_alt, alpha_level, pc_null, alternative)
