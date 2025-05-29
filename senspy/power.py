import numpy as np
from scipy.stats import binom
import warnings
from senspy.links import psyfun # Added for power_discrim

__all__ = ["find_critical_binomial_value", "exact_binomial_power", "sample_size_for_binomial_power", "power_discrim"]

def find_critical_binomial_value(n_trials: int, p_null: float, alpha_level: float, alternative: str = "greater") -> int:
    """
    Determines the critical number of successes (k) required to reject the null 
    hypothesis in a one-sample binomial test. Analogous to sensR::findcr.

    Args:
        n_trials (int): The total number of trials. Must be a positive integer.
        p_null (float): The probability of success under the null hypothesis (H0). 
                        Must be between 0 and 1.
        alpha_level (float): The significance level (Type I error rate). 
                             Must be between 0 and 1.
        alternative (str, optional): Specifies the alternative hypothesis. 
                                     Must be one of "greater" (default) or "less".
                                     "two.sided" is not currently implemented for this helper.

    Returns:
        int: The critical number of successes.
             For "greater", this is the smallest k such that P(X >= k | H0) <= alpha.
             It can be n_trials + 1 if significance cannot be achieved even with all successes.
             For "less", this is the largest k such that P(X <= k | H0) <= alpha.
             It can be -1 if significance cannot be achieved even with zero successes.

    Raises:
        ValueError: If input parameters are invalid.
        NotImplementedError: If alternative is "two.sided".
    """
    # Input Validation
    if not isinstance(n_trials, int) or n_trials <= 0:
        raise ValueError("n_trials must be a positive integer.")
    if not (isinstance(p_null, (float, int, np.number)) and 0 <= p_null <= 1):
        raise ValueError("p_null must be a float or int between 0 and 1 (inclusive).")
    if p_null == 0 and alternative.lower() == "greater":
        # If p_null is 0, P(X >= k) is 0 for k > 0, and 1 for k=0.
        # To have P(X >= k) <= alpha (alpha < 1), k must be at least 1.
        # Or, if alpha is also 0, then k must be > n_trials (impossible).
        # This case is tricky, sensR::findcr might have specific handling.
        # For p_null=0, P(X=0)=1. P(X>=1)=0. Smallest k for P(X>=k)<=alpha is 1 (unless alpha=0).
        return 1 if alpha_level > 0 else n_trials + 1 
    if p_null == 1 and alternative.lower() == "less":
        # If p_null is 1, P(X <= k) is 0 for k < n_trials, and 1 for k=n_trials.
        # To have P(X <= k) <= alpha (alpha < 1), k must be n_trials - 1.
        return n_trials - 1 if alpha_level > 0 else -1
    if p_null == 1 and alternative.lower() == "greater":
        # P(X >= k | p_null=1) means P(X=n_trials). This is 1 if k <= n_trials, 0 if k > n_trials.
        # We need P(X >= k) <= alpha.
        # If alpha < 1, then k must be n_trials + 1 (impossible to achieve significance by rejecting).
        return n_trials + 1
    if p_null == 0 and alternative.lower() == "less":
        # P(X <= k | p_null=0) means P(X=0). This is 1 if k >= 0, 0 if k < 0.
        # We need P(X <= k) <= alpha.
        # If alpha < 1, then k must be -1 (impossible to achieve significance by rejecting).
        return -1


    if not (isinstance(alpha_level, float) and 0 < alpha_level < 1):
        raise ValueError("alpha_level must be a float strictly between 0 and 1.")
    
    alternative = alternative.lower()
    if alternative not in ["greater", "less"]:
        if alternative == "two.sided":
            raise NotImplementedError("Alternative 'two.sided' is not implemented for find_critical_binomial_value. "
                                      "The main power function might handle this by calling this helper twice with alpha/2.")
        else:
            raise ValueError("alternative must be 'greater' or 'less'.")

    critical_value: int
    if alternative == "greater":
        # We want the smallest k such that P(X >= k) <= alpha_level
        # This is P(X > k-1) <= alpha_level
        # Which is 1 - P(X <= k-1) <= alpha_level  (i.e., binom.sf(k-1, ...))
        # So, P(X <= k-1) >= 1 - alpha_level
        # k-1 = binom.ppf(1 - alpha_level, n_trials, p_null)
        # k = binom.ppf(1 - alpha_level, n_trials, p_null) + 1
        
        # Iterate to find k, as ppf might not be exact for discrete distributions.
        # Start from k=0 up to n_trials+1
        for k_val in range(n_trials + 2): # Test k_val from 0 to n_trials+1
            prob_ge_k = binom.sf(k_val - 1, n_trials, p_null) # P(X >= k_val)
            if prob_ge_k <= alpha_level:
                critical_value = k_val
                break
        else: # Should not happen if loop goes to n_trials+1, as P(X >= n_trials+1) = 0
            critical_value = n_trials + 1 
            
    elif alternative == "less":
        # We want the largest k such that P(X <= k) <= alpha_level
        # k = binom.ppf(alpha_level, n_trials, p_null)
        # Iterate downwards to find k, as ppf behavior at exact alpha can be tricky.
        for k_val in range(n_trials, -2, -1): # Test k_val from n_trials down to -1
            if k_val < 0: # P(X <= k) is 0 if k < 0
                prob_le_k = 0.0
            else:
                prob_le_k = binom.cdf(k_val, n_trials, p_null)
            
            if prob_le_k <= alpha_level:
                critical_value = k_val
                break
        else: # Should not happen, loop includes k_val = -1 where P(X<=-1) = 0
            critical_value = -1 
            
    return critical_value


def exact_binomial_power(n_trials: int, p_alt: float, alpha_level: float, p_null: float = 0.5, alternative: str = "greater") -> float:
    """
    Calculates the exact power for a one-sample binomial test.

    Power is the probability of correctly rejecting the null hypothesis (H0)
    when the alternative hypothesis (H_A) is true.

    Args:
        n_trials (int): The total number of trials.
        p_alt (float): The probability of success under the alternative hypothesis (H_A).
                       Must be between 0 and 1 (inclusive).
        alpha_level (float): The significance level (Type I error rate).
                             Must be strictly between 0 and 1.
        p_null (float, optional): The probability of success under the null hypothesis (H0).
                                  Defaults to 0.5. Must be between 0 and 1 (inclusive).
        alternative (str, optional): Specifies the alternative hypothesis.
                                     Must be one of "greater" (default) or "less".
                                     "two.sided" is not currently implemented.

    Returns:
        float: The exact power of the test (probability between 0 and 1).

    Raises:
        ValueError: If input parameters are invalid.
        NotImplementedError: If alternative is "two.sided".
    """
    # Input Validation
    if not isinstance(n_trials, int) or n_trials <= 0:
        raise ValueError("n_trials must be a positive integer.")
    if not (isinstance(p_alt, (float, int, np.number)) and 0 <= p_alt <= 1):
        raise ValueError("p_alt must be a numeric value between 0 and 1 (inclusive).")
    if not (isinstance(alpha_level, float) and 0 < alpha_level < 1):
        raise ValueError("alpha_level must be a float strictly between 0 and 1.")
    if not (isinstance(p_null, (float, int, np.number)) and 0 <= p_null <= 1):
        raise ValueError("p_null must be a numeric value between 0 and 1 (inclusive).")

    alternative = alternative.lower()
    if alternative not in ["greater", "less"]:
        if alternative == "two.sided":
            raise NotImplementedError("Alternative 'two.sided' is not implemented for exact_binomial_power.")
        else:
            raise ValueError("alternative must be 'greater' or 'less'.")

    # Determine the critical value(s) based on H0
    # Note: find_critical_binomial_value already validates p_null if it's not 0 or 1.
    # Handle edge cases for p_null = 0 or 1 before calling find_critical_binomial_value,
    # or ensure find_critical_binomial_value handles them robustly for power calculation.
    # The current find_critical_binomial_value has specific returns for p_null=0 or 1.
    
    # Special handling for p_null extremes if they make find_critical_binomial_value's logic tricky for power
    if p_null == 0 and alternative == "greater":
        # H0: p=0. Crit_val is 1 (reject if X>=1) if alpha > 0.
        # Power = P(X>=1 | p_alt) = 1 - P(X=0 | p_alt) = 1 - (1-p_alt)^n_trials
        return 1.0 - (1 - p_alt)**n_trials if alpha_level > 0 else 0.0
    if p_null == 1 and alternative == "less":
        # H0: p=1. Crit_val is n_trials-1 (reject if X <= n_trials-1) if alpha > 0.
        # Power = P(X <= n_trials-1 | p_alt) = 1 - P(X=n_trials | p_alt) = 1 - p_alt^n_trials
        return 1.0 - p_alt**n_trials if alpha_level > 0 else 0.0
    if p_null == 1 and alternative == "greater":
        # H0: p=1. Crit_val is n_trials+1 (never reject). Power = 0.
        return 0.0
    if p_null == 0 and alternative == "less":
        # H0: p=0. Crit_val is -1 (never reject). Power = 0.
        return 0.0

    crit_val = find_critical_binomial_value(n_trials, p_null, alpha_level, alternative)
    power: float

    if alternative == "greater":
        # Power = P(X >= crit_val | p_alt)
        if crit_val > n_trials: # Rejection region is impossible (e.g. k > n_trials)
            power = 0.0
        elif crit_val <= 0: # Rejection region includes all possible outcomes (k <= 0)
            power = 1.0
        else:
            power = binom.sf(crit_val - 1, n_trials, p_alt)
    elif alternative == "less":
        # Power = P(X <= crit_val | p_alt)
        if crit_val < 0: # Rejection region is impossible (e.g. k < 0)
            power = 0.0
        elif crit_val >= n_trials: # Rejection region includes all possible outcomes
            power = 1.0
        else:
            power = binom.cdf(crit_val, n_trials, p_alt)
    
    return power


def sample_size_for_binomial_power(p_alt: float, target_power: float, alpha_level: float, 
                                 p_null: float = 0.5, alternative: str = "greater", 
                                 min_n: int = 5, max_n: int = 10000) -> int | None:
    """
    Finds the smallest sample size (n_trials) required to achieve a desired 
    statistical power for a one-sample binomial test using an iterative search.

    Args:
        p_alt (float): The probability of success under the alternative hypothesis (H_A).
                       Must be between 0 and 1 (inclusive).
        target_power (float): The desired power level (e.g., 0.80 for 80% power).
                              Must be strictly between 0 and 1.
        alpha_level (float): The significance level (Type I error rate).
                             Must be strictly between 0 and 1.
        p_null (float, optional): The probability of success under the null hypothesis (H0).
                                  Defaults to 0.5. Must be between 0 and 1 (inclusive).
        alternative (str, optional): Specifies the alternative hypothesis.
                                     Must be one of "greater" (default) or "less".
                                     "two.sided" is not currently implemented.
        min_n (int, optional): Minimum sample size to consider in the search. Defaults to 5.
        max_n (int, optional): Maximum sample size to search up to. Defaults to 10000.

    Returns:
        int | None: The estimated integer sample size required to achieve the target_power.
                    Returns None if target_power cannot be achieved within max_n.

    Raises:
        ValueError: If input parameters are invalid or target_power is unachievable 
                    (e.g., p_alt is not more extreme than p_null in the direction of alternative).
        NotImplementedError: If alternative is "two.sided".
    """
    # Input Validation
    if not (isinstance(p_alt, (float, int, np.number)) and 0 <= p_alt <= 1):
        raise ValueError("p_alt must be a numeric value between 0 and 1 (inclusive).")
    if not (isinstance(target_power, float) and 0 < target_power < 1):
        raise ValueError("target_power must be a float strictly between 0 and 1.")
    if not (isinstance(alpha_level, float) and 0 < alpha_level < 1):
        raise ValueError("alpha_level must be a float strictly between 0 and 1.")
    if not (isinstance(p_null, (float, int, np.number)) and 0 <= p_null <= 1):
        raise ValueError("p_null must be a numeric value between 0 and 1 (inclusive).")
    if not (isinstance(min_n, int) and min_n > 0):
        raise ValueError("min_n must be a positive integer.")
    if not (isinstance(max_n, int) and max_n >= min_n):
        raise ValueError("max_n must be a positive integer greater than or equal to min_n.")

    alternative = alternative.lower()
    if alternative not in ["greater", "less"]:
        if alternative == "two.sided":
            raise NotImplementedError("Alternative 'two.sided' is not implemented for sample_size_for_binomial_power.")
        else:
            raise ValueError("alternative must be 'greater' or 'less'.")

    # Check if target power is achievable in principle
    if alternative == "greater" and p_alt <= p_null:
        warnings.warn(f"p_alt ({p_alt}) must be > p_null ({p_null}) for alternative='greater'. Target power may not be achievable.", UserWarning)
        # Power will likely be <= alpha, so if target_power > alpha, it's impossible.
        # We let the loop run to see if it can find a solution by chance for very small n, but it's unlikely.
    if alternative == "less" and p_alt >= p_null:
        warnings.warn(f"p_alt ({p_alt}) must be < p_null ({p_null}) for alternative='less'. Target power may not be achievable.", UserWarning)

    # Iterative search for sample size
    for n_trials_current in range(min_n, max_n + 1):
        current_power = exact_binomial_power(
            n_trials_current, 
            p_alt, 
            alpha_level, 
            p_null, 
            alternative
        )
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

    This function serves as a general power calculation tool for various
    discrimination methods supported by `senspy.links.psyfun`.

    Args:
        d_prime_alt (float): The d-prime value under the alternative hypothesis 
                             (the d-prime to be detected). Must be non-negative.
        n_trials (int): The number of trials in the sensory test. Must be positive.
        method (str): The discrimination method string (e.g., "2afc", "triangle").
        d_prime_null (float, optional): The d-prime value under the null hypothesis. 
                                        Defaults to 0.0. Must be non-negative.
        alpha_level (float, optional): The significance level (Type I error rate). 
                                       Defaults to 0.05. Must be strictly between 0 and 1.
        alternative (str, optional): Specifies the alternative hypothesis relative to d_prime_null.
                                     Must be one of "greater" (default) or "less".
                                     "two.sided" is not currently implemented by underlying functions.
        **kwargs: Additional keyword arguments (currently not used but included for future flexibility).

    Returns:
        float: The statistical power (probability of correctly rejecting H0).

    Raises:
        ValueError: If input parameters are invalid.
        NotImplementedError: If alternative is "two.sided".
    """
    # Input Validation
    if not isinstance(d_prime_alt, (float, int, np.number)) or d_prime_alt < 0:
        raise ValueError("d_prime_alt must be a non-negative numeric value.")
    if not isinstance(d_prime_null, (float, int, np.number)) or d_prime_null < 0:
        raise ValueError("d_prime_null must be a non-negative numeric value.")
    if not isinstance(n_trials, int) or n_trials <= 0:
        raise ValueError("n_trials must be a positive integer.")
    # `method` validation will be implicitly handled by psyfun
    if not (isinstance(alpha_level, float) and 0 < alpha_level < 1):
        raise ValueError("alpha_level must be a float strictly between 0 and 1.")
    
    alternative = alternative.lower()
    if alternative not in ["greater", "less"]:
        if alternative == "two.sided":
            # exact_binomial_power and find_critical_binomial_value do not currently support "two.sided"
            raise NotImplementedError("Alternative 'two.sided' is not yet supported for power_discrim.")
        else:
            raise ValueError("alternative must be 'greater' or 'less'.")

    # Convert d-prime values to probabilities of correct response (Pc)
    try:
        pc_alt = psyfun(d_prime_alt, method=method)
        pc_null = psyfun(d_prime_null, method=method)
    except ValueError as e: # Catches unknown method from psyfun
        raise ValueError(f"Invalid method '{method}' or d-prime value for psyfun: {e}")


    # Validate relationship between pc_alt and pc_null based on alternative
    if alternative == "greater":
        if pc_alt <= pc_null:
            warnings.warn(f"For alternative='greater', pc_alt ({pc_alt:.4f} from d_prime_alt={d_prime_alt}) "
                          f"should be > pc_null ({pc_null:.4f} from d_prime_null={d_prime_null}). "
                          "Power may be low or equal to alpha.", UserWarning)
            # If pc_alt is not greater, power will be alpha or less.
            # exact_binomial_power will calculate this; if crit_val leads to rejecting H0,
            # then P(reject H0 | H_A is true) will be calculated.
            # If p_alt <= p_null for "greater", power will be <= alpha.
    elif alternative == "less":
        if pc_alt >= pc_null:
            warnings.warn(f"For alternative='less', pc_alt ({pc_alt:.4f} from d_prime_alt={d_prime_alt}) "
                          f"should be < pc_null ({pc_null:.4f} from d_prime_null={d_prime_null}). "
                          "Power may be low or equal to alpha.", UserWarning)
            # If pc_alt >= pc_null for "less", power will be <= alpha.

    # Calculate power using the exact binomial test framework
    power = exact_binomial_power(
        n_trials=n_trials,
        p_alt=pc_alt,
        alpha_level=alpha_level,
        p_null=pc_null,
        alternative=alternative
    )
    
    return power
