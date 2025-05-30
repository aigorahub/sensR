import pytest
import numpy as np
from scipy.stats import binom
import warnings

from senspy.power import (
    find_critical_binomial_value,
    exact_binomial_power,
    sample_size_for_binomial_power,
    power_discrim,
    power_discrim_normal_approx # Added new function
)
from senspy.discrimination import psyfun # Corrected import

# --- Tests for find_critical_binomial_value ---

@pytest.mark.parametrize("n_trials, p_null, alpha, alt, expected_k", [
    (10, 0.5, 0.05, "greater", 8),
    (10, 0.5, 0.055, "greater", 8),
    (10, 0.5, 0.054, "greater", 9),
    (10, 0.5, 0.05, "less", 2),
    (10, 0.5, 0.055, "less", 2),
    (10, 0.5, 0.054, "less", 1),
    (50, 1/3, 0.05, "greater", 23),
    (50, 1/3, 0.01, "greater", 26),
    (20, 0.1, 0.1, "greater", 4),
    (20, 0.9, 0.1, "less", 16),
    (10, 0.0, 0.05, "greater", 1),
    (10, 1.0, 0.05, "less", 9),
    (10, 1.0, 0.05, "greater", 11),
    (10, 0.0, 0.05, "less", -1),
    (10, 0.00001, 0.05, "greater", 1),
    (10, 0.99999, 0.05, "less", 9),
])
def test_find_critical_binomial_value_cases(n_trials, p_null, alpha, alt, expected_k):
    assert find_critical_binomial_value(n_trials, p_null, alpha, alternative=alt) == expected_k

def test_find_critical_binomial_value_two_sided():
    with pytest.raises(NotImplementedError):
        find_critical_binomial_value(10, 0.5, 0.05, alternative="two.sided")

def test_find_critical_binomial_value_invalid_inputs():
    with pytest.raises(ValueError): find_critical_binomial_value(0, 0.5, 0.05)
    with pytest.raises(ValueError): find_critical_binomial_value(10, 1.5, 0.05)
    with pytest.raises(ValueError): find_critical_binomial_value(10, -0.5, 0.05)
    with pytest.raises(ValueError): find_critical_binomial_value(10, 0.5, 1.5)
    with pytest.raises(ValueError): find_critical_binomial_value(10, 0.5, -0.5)
    with pytest.raises(ValueError): find_critical_binomial_value(10, 0.5, 0.05, "other")

# --- Tests for exact_binomial_power ---

@pytest.mark.parametrize("n, p_alt, alpha, p_null, alt, expected_power_approx", [
    (10, 0.8, 0.05, 0.5, "greater", binom.sf(8-1, 10, 0.8)),
    (10, 0.2, 0.05, 0.5, "less", binom.cdf(2, 10, 0.2)),
    (50, 0.5, 0.05, 1/3, "greater", binom.sf(23-1, 50, 0.5)),
    (20, 0.5, 0.05, 0.5, "greater", binom.sf(find_critical_binomial_value(20,0.5,0.05,"greater")-1, 20,0.5)),
    (20, 0.5, 0.05, 0.5, "less", binom.cdf(find_critical_binomial_value(20,0.5,0.05,"less"), 20,0.5)),
    (10, 0.8, 0.05, 0.0, "greater", 1.0 - (1-0.8)**10),
    (10, 0.2, 0.05, 1.0, "less", 1.0 - 0.2**10),
    (10, 0.8, 0.05, 1.0, "greater", 0.0),
    (10, 0.2, 0.05, 0.0, "less", 0.0),
])
def test_exact_binomial_power_cases(n, p_alt, alpha, p_null, alt, expected_power_approx):
    if (alt == "greater" and p_alt <= p_null) or \
       (alt == "less" and p_alt >= p_null):
        with pytest.warns(UserWarning, match="Power may be low or equal to alpha"):
            power = exact_binomial_power(n, p_alt, alpha, p_null, alternative=alt)
    else:
        power = exact_binomial_power(n, p_alt, alpha, p_null, alternative=alt)
    np.testing.assert_allclose(power, expected_power_approx, atol=1e-6)

def test_exact_binomial_power_two_sided():
    with pytest.raises(NotImplementedError):
        exact_binomial_power(10, 0.8, 0.05, 0.5, alternative="two.sided")

# --- Tests for sample_size_for_binomial_power ---

@pytest.mark.parametrize("p_alt, target_power, alpha, p_null, alt, max_n, expected_n_range", [
    (0.8, 0.8, 0.05, 0.5, "greater", 50, (12, 15)),
    (0.2, 0.8, 0.05, 0.5, "less", 50, (12, 15)),
    (0.5, 0.9, 0.01, 1/3, "greater", 100, (60, 70)),
    (0.6, 0.95, 0.05, 0.5, "greater", 20, None),
    (0.5, 0.8, 0.05, 0.5, "greater", 50, None),
])
def test_sample_size_for_binomial_power_cases(p_alt, target_power, alpha, p_null, alt, max_n, expected_n_range):
    if (alt == "greater" and p_alt <= p_null) or \
       (alt == "less" and p_alt >= p_null) or \
       (expected_n_range is None and target_power > alpha):
        with pytest.warns(UserWarning):
            n_required = sample_size_for_binomial_power(p_alt, target_power, alpha, p_null, alt, max_n=max_n)
    else:
        n_required = sample_size_for_binomial_power(p_alt, target_power, alpha, p_null, alt, max_n=max_n)

    if expected_n_range is None:
        assert n_required is None
    else:
        assert n_required is not None, f"Expected sample size in {expected_n_range}, but got None"
        assert expected_n_range[0] <= n_required <= expected_n_range[1]

def test_sample_size_for_binomial_power_two_sided():
    with pytest.raises(NotImplementedError):
        sample_size_for_binomial_power(0.8, 0.8, 0.05, 0.5, alternative="two.sided")

# --- Tests for power_discrim ---

@pytest.mark.parametrize("d_alt, n, method, d_null, alpha, alt, expected_power_range", [
    (1.0, 50, "2afc", 0.0, 0.05, "greater", (0.7, 0.9)),
    (1.5, 30, "triangle", 0.0, 0.05, "greater", (0.6, 0.85)),
    (0.5, 50, "2afc", 0.2, 0.05, "greater", (0.3, 0.5)),
    (0.3, 50, "2afc", 0.8, 0.05, "less", (0.6, 0.8)),
    (0.5, 50, "2afc", 0.5, 0.05, "greater", (0.0, 0.1)),
])
def test_power_discrim_cases(d_alt, n, method, d_null, alpha, alt, expected_power_range):
    pc_alt = psyfun(d_alt, method=method)
    pc_null = psyfun(d_null, method=method)

    if (alt == "greater" and pc_alt <= pc_null) or \
       (alt == "less" and pc_alt >= pc_null):
        with pytest.warns(UserWarning, match="Power may be low or equal to alpha"):
            power = power_discrim(d_alt, n, method, d_null, alpha, alt)
    else:
        power = power_discrim(d_alt, n, method, d_null, alpha, alt)
    
    assert expected_power_range[0] <= power <= expected_power_range[1], f"Power {power} out of range {expected_power_range}"

def test_power_discrim_two_sided():
    with pytest.raises(NotImplementedError):
        power_discrim(1.0, 50, "2afc", 0.0, 0.05, alternative="two.sided")

def test_power_discrim_invalid_method():
    with pytest.raises(ValueError, match="Invalid method 'unknown_method'"):
        power_discrim(1.0, 50, "unknown_method")

# --- Tests for power_discrim_normal_approx ---

@pytest.mark.parametrize("d_alt, n_trials, method, d_null, alpha, alt, expected_power_range", [
    (1.0, 50, "2afc", 0.0, 0.05, "greater", (0.75, 0.95)),
    (0.5, 100, "triangle", 0.0, 0.05, "greater", (0.30, 0.55)),
    (1.0, 50, "2afc", 0.0, 0.05, "two-sided", (0.70, 0.90)),
    (0.0, 50, "2afc", 0.0, 0.05, "greater", (0.04, 0.06)),
])
def test_power_discrim_normal_approx_power_calc(d_alt, n_trials, method, d_null, alpha, alt, expected_power_range):
    """Test power calculation using normal approximation."""
    if d_alt == d_null and alt != "two-sided":
         with warnings.catch_warnings(record=True) as w:
            power = power_discrim_normal_approx(d_prime_alt=d_alt, n_trials=n_trials, method=method, 
                                                d_prime_null=d_null, alpha_level=alpha, alternative=alt)
            assert any("Effect size is 0" in str(warn.message) or "power is alpha_level" in str(warn.message) for warn in w)
    else:
        power = power_discrim_normal_approx(d_prime_alt=d_alt, n_trials=n_trials, method=method, 
                                            d_prime_null=d_null, alpha_level=alpha, alternative=alt)

    assert expected_power_range[0] <= power <= expected_power_range[1], \
        f"Calculated power {power:.4f} not in expected range {expected_power_range} for {alt} test."

@pytest.mark.parametrize("d_alt, method, target_power, d_null, alpha, alt, expected_n_range", [
    (1.0, "2afc", 0.80, 0.0, 0.05, "greater", (40, 60)),
    (1.5, "triangle", 0.90, 0.0, 0.01, "two-sided", (60, 90)),
    (0.2, "2afc", 0.50, 0.0, 0.10, "greater", (200, 250)),
])
def test_power_discrim_normal_approx_sample_size_calc(d_alt, method, target_power, d_null, alpha, alt, expected_n_range):
    """Test sample size calculation using normal approximation."""
    n_required = power_discrim_normal_approx(d_prime_alt=d_alt, n_trials=None, method=method, 
                                            power_target=target_power, d_prime_null=d_null, 
                                            alpha_level=alpha, alternative=alt)
    
    assert n_required is not None and np.isfinite(n_required), "Sample size should be a finite number."
    assert isinstance(n_required, (int, np.integer)), "Sample size should be an integer or numpy integer." # Corrected type check
    assert expected_n_range[0] <= n_required <= expected_n_range[1], \
        f"Calculated n_trials {n_required} not in expected range {expected_n_range}."

def test_power_discrim_normal_approx_edge_cases():
    """Test edge cases like zero effect size."""
    power_at_alpha = power_discrim_normal_approx(d_prime_alt=0.5, n_trials=100, method="2afc", 
                                                 d_prime_null=0.5, alpha_level=0.05, alternative="greater")
    np.testing.assert_allclose(power_at_alpha, 0.05, atol=0.01, err_msg="Power for zero effect (greater) should be approx. alpha.")

    power_at_alpha_ts = power_discrim_normal_approx(d_prime_alt=0.5, n_trials=100, method="2afc", 
                                                 d_prime_null=0.5, alpha_level=0.05, alternative="two-sided")
    np.testing.assert_allclose(power_at_alpha_ts, 0.05, atol=0.01, err_msg="Power for zero effect (two-sided) should be approx. alpha.")

    with warnings.catch_warnings(record=True) as w:
        power_low = power_discrim_normal_approx(d_prime_alt=0.1, n_trials=100, method="2afc", 
                                                d_prime_null=0.5, alpha_level=0.05, alternative="greater")
        assert any("effect size is negative" in str(warn.message).lower() or "power will be small" in str(warn.message).lower() for warn in w)
    assert power_low < 0.051, "Power when effect is in opposite direction (greater) should be <= alpha."

    with warnings.catch_warnings(record=True) as w:
        n_for_zero_effect = power_discrim_normal_approx(d_prime_alt=0.5, method="2afc", power_target=0.8,
                                                        d_prime_null=0.5, alpha_level=0.05, alternative="greater")
    assert n_for_zero_effect is np.nan or n_for_zero_effect > 10**6, \
        "Sample size for zero effect should be NaN or extremely large."

def test_power_discrim_normal_approx_invalid_inputs():
    with pytest.raises(ValueError, match="d_prime_alt must be a non-negative numeric value"):
        power_discrim_normal_approx(d_prime_alt=-1.0, n_trials=50, method="2afc")
    with pytest.raises(ValueError, match="n_trials must be a positive integer for power calculation"):
        power_discrim_normal_approx(d_prime_alt=1.0, n_trials=0, method="2afc")
    with pytest.raises(ValueError, match="power_target must be a float strictly between 0 and 1"):
        power_discrim_normal_approx(d_prime_alt=1.0, n_trials=None, method="2afc", power_target=1.5) # Added n_trials=None
    with pytest.raises(ValueError, match="alternative must be 'greater', 'less', or 'two-sided'"):
        power_discrim_normal_approx(d_prime_alt=1.0, n_trials=50, method="2afc", alternative="invalid_alt")
    with pytest.raises(ValueError, match="Invalid method 'unknown_method'"):
        power_discrim_normal_approx(d_prime_alt=1.0, n_trials=50, method="unknown_method")
