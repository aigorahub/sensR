import pytest
import numpy as np
from scipy.stats import binom
import warnings

from senspy.power import (
    find_critical_binomial_value,
    exact_binomial_power,
    sample_size_for_binomial_power,
    power_discrim
)
from senspy.links import psyfun

# --- Tests for find_critical_binomial_value ---

@pytest.mark.parametrize("n_trials, p_null, alpha, alt, expected_k", [
    (10, 0.5, 0.05, "greater", 8),      # P(X>=8|10,0.5) = 0.0546875
    (10, 0.5, 0.055, "greater", 8),     # Should still be 8 as P(X>=8) is smallest <= 0.055
    (10, 0.5, 0.054, "greater", 9),     # P(X>=9|10,0.5) = 0.01074
    (10, 0.5, 0.05, "less", 2),         # P(X<=2|10,0.5) = 0.0546875
    (10, 0.5, 0.055, "less", 2),        # Should still be 2
    (10, 0.5, 0.054, "less", 1),        # P(X<=1|10,0.5) = 0.01074
    (50, 1/3, 0.05, "greater", 23),     # P(X>=23|50,1/3) approx 0.043
    (50, 1/3, 0.01, "greater", 26),     # P(X>=26|50,1/3) approx 0.007
    (20, 0.1, 0.1, "greater", 4),       # P(X>=4|20,0.1) approx 0.043 (sf(3)=0.043)
    (20, 0.9, 0.1, "less", 16),         # P(X<=16|20,0.9) approx 0.043 (cdf(16)=0.043)
    # Edge cases for p_null
    (10, 0.0, 0.05, "greater", 1),      # Must get at least 1 success if null is 0
    (10, 1.0, 0.05, "less", 9),         # Must get less than 10 successes if null is 1
    (10, 1.0, 0.05, "greater", 11),     # Impossible to be "more" than all successes
    (10, 0.0, 0.05, "less", -1),        # Impossible to be "less" than 0 successes
    (10, 0.00001, 0.05, "greater", 1),  # Test near p_null=0
    (10, 0.99999, 0.05, "less", 9),   # Test near p_null=1
])
def test_find_critical_binomial_value_cases(n_trials, p_null, alpha, alt, expected_k):
    assert find_critical_binomial_value(n_trials, p_null, alpha, alternative=alt) == expected_k

def test_find_critical_binomial_value_two_sided():
    with pytest.raises(NotImplementedError):
        find_critical_binomial_value(10, 0.5, 0.05, alternative="two.sided")

def test_find_critical_binomial_value_invalid_inputs():
    with pytest.raises(ValueError): find_critical_binomial_value(0, 0.5, 0.05) # n_trials <=0
    with pytest.raises(ValueError): find_critical_binomial_value(10, 1.5, 0.05) # p_null > 1
    with pytest.raises(ValueError): find_critical_binomial_value(10, -0.5, 0.05) # p_null < 0
    with pytest.raises(ValueError): find_critical_binomial_value(10, 0.5, 1.5) # alpha > 1
    with pytest.raises(ValueError): find_critical_binomial_value(10, 0.5, -0.5) # alpha < 0
    with pytest.raises(ValueError): find_critical_binomial_value(10, 0.5, 0.05, "other")


# --- Tests for exact_binomial_power ---

@pytest.mark.parametrize("n, p_alt, alpha, p_null, alt, expected_power_approx", [
    (10, 0.8, 0.05, 0.5, "greater", binom.sf(8-1, 10, 0.8)), # crit_val=8
    (10, 0.2, 0.05, 0.5, "less", binom.cdf(2, 10, 0.2)),    # crit_val=2
    (50, 0.5, 0.05, 1/3, "greater", binom.sf(23-1, 50, 0.5)), # crit_val=23
    # Cases with p_alt not more extreme (expect power approx alpha)
    (20, 0.5, 0.05, 0.5, "greater", binom.sf(find_critical_binomial_value(20,0.5,0.05,"greater")-1, 20,0.5)),
    (20, 0.5, 0.05, 0.5, "less", binom.cdf(find_critical_binomial_value(20,0.5,0.05,"less"), 20,0.5)),
    # Edge p_null cases (these are handled by direct returns in exact_binomial_power)
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
    (0.8, 0.8, 0.05, 0.5, "greater", 50, (12, 15)), # sensR::discrimSS gives 13
    (0.2, 0.8, 0.05, 0.5, "less", 50, (12, 15)),    # sensR::discrimSS gives 13 (symmetric)
    (0.5, 0.9, 0.01, 1/3, "greater", 100, (60, 70)), # sensR::discrimSS gives 65 for 2afc with similar params
    # Target power not reached
    (0.6, 0.95, 0.05, 0.5, "greater", 20, None), # Needs more than 20 trials
    # p_alt not more extreme
    (0.5, 0.8, 0.05, 0.5, "greater", 50, None), # p_alt = p_null
])
def test_sample_size_for_binomial_power_cases(p_alt, target_power, alpha, p_null, alt, max_n, expected_n_range):
    if (alt == "greater" and p_alt <= p_null) or \
       (alt == "less" and p_alt >= p_null) or \
       (expected_n_range is None and target_power > alpha): # If power is not achievable
        with pytest.warns(UserWarning): # Expect "not reached" or "p_alt must be..."
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
    (1.0, 50, "2afc", 0.0, 0.05, "greater", (0.7, 0.9)), # pc_alt~0.76, pc_null=0.5. Power should be decent.
    (1.5, 30, "triangle", 0.0, 0.05, "greater", (0.6, 0.85)), # pc_alt~0.63, pc_null~0.33
    (0.5, 50, "2afc", 0.2, 0.05, "greater", (0.3, 0.5)), # Small diff in dprime
    (0.3, 50, "2afc", 0.8, 0.05, "less", (0.6, 0.8)), # Test "less" alternative
    # p_alt not more extreme
    (0.5, 50, "2afc", 0.5, 0.05, "greater", (0.0, 0.1)), # Expect power around alpha
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

```
