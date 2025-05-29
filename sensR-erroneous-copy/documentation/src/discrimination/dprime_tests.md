---
layout: layouts/base.njk
title: "d-prime Hypothesis Tests"
permalink: /discrimination/dprime-tests/
tags: api_doc
---

## Introduction

This page details functions in `senspy.discrimination` used for performing statistical hypothesis tests involving d-prime (d') values. These functions allow you to test a common d-prime against a specific value or compare d-prime values across multiple groups.

These functionalities are inspired by and aim to provide similar capabilities as those found in `sensR` (specifically from `R/d.primeTest.R`).

---

## `dprime_test`: Test a common d-prime

### Purpose

The `dprime_test` function estimates a common d-prime value from one or more groups of sensory discrimination data. It then performs a hypothesis test to determine if this common d-prime is significantly different from a specified value (`dprime0`). This is useful for testing, for example, if a product is discriminable from a control (i.e., d-prime > 0) or if a panel's sensitivity meets a certain target.

Currently, estimation of the common d-prime is done via Maximum Likelihood ("ML"), and the hypothesis test uses a Wald statistic.

### Function Signature

```python
dprime_test(
    correct: list[int] | int, 
    total: list[int] | int, 
    protocol: list[str] | str, 
    dprime0: float = 0.0, 
    conf_level: float = 0.95, 
    statistic: str = "Wald", 
    alternative: str = "two.sided", 
    estim_method: str = "ML"
) -> dict
```

### Parameters

*   **`correct`** (list[int] | int): The number of correct responses. Can be a single integer for one group or a list of integers for multiple groups.
*   **`total`** (list[int] | int): The total number of trials. Can be a single integer or a list corresponding to `correct`.
*   **`protocol`** (list[str] | str): The discrimination protocol(s) used (e.g., `"2afc"`, `"triangle"`). Can be a single string or a list corresponding to `correct`.
*   **`dprime0`** (float, optional): The hypothesized value of d-prime under the null hypothesis. Defaults to `0.0`.
*   **`conf_level`** (float, optional): The desired confidence level for the confidence interval of the common d-prime. Defaults to `0.95`.
*   **`statistic`** (str, optional): The type of test statistic to use. Currently, only `"Wald"` is implemented. Defaults to `"Wald"`.
*   **`alternative`** (str, optional): Specifies the alternative hypothesis. Must be one of `"two.sided"` (default), `"less"` (common d-prime < dprime0), or `"greater"` (common d-prime > dprime0).
*   **`estim_method`** (str, optional): The method for estimating the common d-prime. Currently, only `"ML"` (Maximum Likelihood) is implemented. Defaults to `"ML"`.

### Return Value

The function returns a dictionary containing:

*   **`"common_dprime_est"`** (float): The Maximum Likelihood estimate of the common d-prime.
*   **`"se_common_dprime_est"`** (float): The standard error of the common d-prime estimate.
*   **`"conf_int_common_dprime"`** (tuple): The confidence interval `(lower_bound, upper_bound)` for the common d-prime.
*   **`"statistic_value"`** (float): The value of the Wald Z-statistic for the test against `dprime0`.
*   **`"p_value"`** (float): The p-value for the hypothesis test.
*   **`"dprime0"`** (float): The hypothesized d-prime value used in the test.
*   **`"alternative"`** (str): The alternative hypothesis used.
*   **`"conf_level"`** (float): The confidence level used.
*   **`"estim_method"`** (str): The estimation method used (e.g., "ML").
*   **`"statistic_type"`** (str): The type of statistic used (e.g., "WALD").
*   **`"individual_group_estimates"`** (list[dict]): A list of dictionaries, where each dictionary contains the individual d-prime estimate and other statistics for each group (from an internal call to `senspy.discrimination.discrim`).
*   **`"loglik_common_dprime"`** (float): The maximized log-likelihood value for the model assuming a common d-prime.
*   **`"convergence_status_common_dprime"`** (bool | None): `True` if the optimization for the common d-prime converged, `False` or `None` otherwise.

### Usage Examples

```python
from senspy.discrimination import dprime_test
import numpy as np

# Example 1: Single group, testing if d-prime > 0
result_single = dprime_test(correct=35, total=50, protocol="2afc", dprime0=0, alternative="greater")
print("--- Single Group d-prime_test (d' > 0) ---")
print(f"Common d-prime estimate: {result_single['common_dprime_est']:.4f}")
print(f"SE: {result_single['se_common_dprime_est']:.4f}")
print(f"P-value: {result_single['p_value']:.4f}")
print(f"CI for common d-prime: {result_single['conf_int_common_dprime']}")

# Example 2: Multiple groups, testing if common d-prime is different from 0.5
correct_counts = [30, 35, 32]
total_counts = [50, 50, 50]
protocols = ["2afc", "triangle", "2afc"] # Note: Using different protocols is supported

result_multi = dprime_test(correct_counts, total_counts, protocols, dprime0=0.5, alternative="two.sided")
print("\\n--- Multiple Groups d-prime_test (d' != 0.5) ---")
print(f"Common d-prime estimate: {result_multi['common_dprime_est']:.4f}")
print(f"SE: {result_multi['se_common_dprime_est']:.4f}")
print(f"Wald Z-statistic: {result_multi['statistic_value']:.4f}")
print(f"P-value: {result_multi['p_value']:.4f}")
# print("\\nIndividual group estimates:")
# for i, res_group in enumerate(result_multi['individual_group_estimates']):
#     print(f"  Group {i+1} ({res_group['method']}): d-prime = {res_group['dprime']:.4f}")
```

#### Further Examples for `dprime_test`

##### Example 3: Different Protocols in a Multi-Group Test
The `dprime_test` can estimate a common d-prime even when the groups used different sensory protocols, as d-prime is a common scale.

```python
correct_diff_proto = [30, 25]
total_diff_proto = [50, 40] # Different number of trials too
protocol_diff_proto = ['2afc', 'triangle']
dprime0_val = 0.5

result_diff_proto = dprime_test(correct_diff_proto, total_diff_proto, protocol_diff_proto, 
                                dprime0=dprime0_val, alternative="two.sided")

print(f"\\n--- dprime_test with Different Protocols (vs d'={dprime0_val}) ---")
print(f"Common d-prime estimate: {result_diff_proto['common_dprime_est']:.4f}")
print(f"P-value: {result_diff_proto['p_value']:.4f}")
# The common d-prime is estimated by finding a single d-prime value that best fits 
# all group data simultaneously, considering each group's specific protocol.
```

##### Example 4: Interpreting `alternative` options more explicitly
Let's use a single group with `correct=40, total=50, protocol='2afc'`. 
The d-prime for this (from `discrim(40,50,'2afc')`) is approximately 1.68.

```python
# Data: correct=40, total=50, protocol='2afc' (d' approx 1.68)

# Test 1: Is d-prime greater than 1.5?
res_alt1 = dprime_test(40, 50, '2afc', dprime0=1.5, alternative='greater')
print(f"\\n--- Alternative Test (d' > 1.5) ---")
print(f"P-value (d' > 1.5): {res_alt1['p_value']:.4f}")
# Expected: p-value might be low/moderate if 1.68 is sufficiently > 1.5 given SE.

# Test 2: Is d-prime less than 1.5?
res_alt2 = dprime_test(40, 50, '2afc', dprime0=1.5, alternative='less')
print(f"\\n--- Alternative Test (d' < 1.5) ---")
print(f"P-value (d' < 1.5): {res_alt2['p_value']:.4f}")
# Expected: p-value should be high, as 1.68 is not less than 1.5.

# Test 3: Is d-prime greater than 2.0?
res_alt3 = dprime_test(40, 50, '2afc', dprime0=2.0, alternative='greater')
print(f"\\n--- Alternative Test (d' > 2.0) ---")
print(f"P-value (d' > 2.0): {res_alt3['p_value']:.4f}")
# Expected: p-value should be high, as 1.68 is not greater than 2.0.
```
**Explanation:**
*   **Test 1 (`alternative='greater'`, `dprime0=1.5`):** We're testing if the true d-prime is significantly greater than 1.5. Since our estimate is ~1.68, the p-value tells us the probability of observing a d-prime this high or higher if the true d-prime were actually 1.5.
*   **Test 2 (`alternative='less'`, `dprime0=1.5`):** We're testing if the true d-prime is significantly less than 1.5. Since our estimate is ~1.68, it's unlikely to be significantly less than 1.5, so we expect a large p-value.
*   **Test 3 (`alternative='greater'`, `dprime0=2.0`):** We're testing if the true d-prime is significantly greater than 2.0. Since our estimate is ~1.68, it's unlikely to be significantly greater than 2.0, so we expect a large p-value.

##### Example 5: Case where `common_dprime_est` is close to `dprime0`
If the true common d-prime is very close to `dprime0`, the p-value for a two-sided test should be large, indicating no significant difference from `dprime0`.

```python
# Data: Two groups with similar, low performance.
# For 2AFC, 35/100 correct -> pc=0.35. pguess=0.5. dprime from discrim is 0.
# The common dprime ML estimate should also be 0.
result_close = dprime_test(correct=[35,36], total=[100,100], 
                           protocol=['2afc','2afc'], dprime0=0.0, 
                           alternative='two.sided')
print(f"\\n--- dprime_test (d' close to dprime0=0) ---")
print(f"Common d-prime estimate: {result_close['common_dprime_est']:.4f}")
print(f"P-value when d-prime close to dprime0: {result_close['p_value']:.3f}")
# Expected: p-value should be large (close to 1.0).
```

---

## `dprime_compare`: Compare d-primes from multiple groups

### Purpose

The `dprime_compare` function tests the hypothesis that d-prime values from two or more groups are all equal. This is useful for determining if there are significant differences in sensitivity across different conditions, products, or panels.

Currently, the comparison uses a Likelihood Ratio Test, where the full model allows each group to have its own d-prime, and the reduced model assumes a single common d-prime for all groups.

### Function Signature

```python
dprime_compare(
    correct: list[int], 
    total: list[int], 
    protocol: list[str], 
    conf_level: float = 0.95, # Retained for signature consistency
    estim_method: str = "ML", 
    statistic_method: str = "LikelihoodRatio"
) -> dict
```

### Parameters

*   **`correct`** (list[int]): A list of correct responses for each group. Must have length >= 2.
*   **`total`** (list[int]): A list of total trials for each group. Must have the same length as `correct`.
*   **`protocol`** (list[str]): A list of discrimination protocols for each group. Must have the same length as `correct`.
*   **`conf_level`** (float, optional): Confidence level, retained for signature consistency but not directly used by the Likelihood Ratio Test itself. Defaults to `0.95`.
*   **`estim_method`** (str, optional): Method for estimating parameters. Currently, only `"ML"` (Maximum Likelihood) is implemented. Defaults to `"ML"`.
*   **`statistic_method`** (str, optional): Method for the comparison statistic. Currently, only `"LikelihoodRatio"` is implemented. Defaults to `"LikelihoodRatio"`.

### Return Value

The function returns a dictionary containing:

*   **`"LR_statistic"`** (float): The Likelihood Ratio (LR) test statistic value (often denoted as G² or -2ΔLL).
*   **`"df"`** (int): The degrees of freedom for the LR test (`num_groups - 1`).
*   **`"p_value"`** (float): The p-value associated with the LR statistic.
*   **`"loglik_full_model"`** (float): The log-likelihood of the full model (where each group has its own d-prime).
*   **`"loglik_reduced_model"`** (float): The log-likelihood of the reduced model (where all groups share a common d-prime).
*   **`"common_dprime_H0_est"`** (float): The estimated common d-prime under the null hypothesis (H0).
*   **`"individual_group_estimates"`** (list[dict]): A list of dictionaries from `senspy.discrimination.discrim` for each group, representing the full model estimates.
*   **`"estim_method"`** (str): The estimation method used.
*   **`"statistic_method"`** (str): The statistic method used.
*   **`"conf_level"`** (float): The confidence level specified.

### Usage Examples

```python
from senspy.discrimination import dprime_compare
import numpy as np

# Example 1: Two groups, potentially different
correct_g1 = [30, 45] 
total_g1 = [50, 50]
protocol_g1 = ["2afc", "2afc"]
result_compare_diff = dprime_compare(correct_g1, total_g1, protocol_g1)
print("--- d-prime Comparison (Likely Different Groups) ---")
print(f"LR Statistic: {result_compare_diff['LR_statistic']:.4f}")
print(f"df: {result_compare_diff['df']}")
print(f"P-value: {result_compare_diff['p_value']:.4f}")
# print("\\nIndividual estimates for comparison:")
# for i, res_group in enumerate(result_compare_diff['individual_group_estimates']):
#     print(f"  Group {i+1} ({res_group['method']}): d-prime = {res_group['dprime']:.4f}")
# print(f"Common d-prime under H0: {result_compare_diff['common_dprime_H0_est']:.4f}")


# Example 2: Two groups, likely similar (from original examples)
correct_g2_orig = [30, 32]
total_g2_orig = [50, 50]
protocol_g2_orig = ["triangle", "triangle"]
result_compare_sim_orig = dprime_compare(correct_g2_orig, total_g2_orig, protocol_g2_orig)
print("\\n--- d-prime Comparison (Likely Similar Groups - Original Example) ---")
print(f"LR Statistic: {result_compare_sim_orig['LR_statistic']:.4f}")
print(f"df: {result_compare_sim_orig['df']}")
print(f"P-value: {result_compare_sim_orig['p_value']:.4f}")

```

#### Further Examples for `dprime_compare`

##### Example 3: Comparing Three Groups
This demonstrates comparing more than two groups. The degrees of freedom will be `num_groups - 1`.

```python
correct_3gr = [30, 35, 40]
total_3gr = [50, 50, 50]
protocol_3gr = ['2afc', '2afc', '2afc']
result_3gr = dprime_compare(correct_3gr, total_3gr, protocol_3gr)

print(f"\\n--- Comparing Three 2AFC Groups ---")
print(f"LR Statistic: {result_3gr['LR_statistic']:.4f}")
print(f"df: {result_3gr['df']}") # Should be 3 - 1 = 2
print(f"P-value: {result_3gr['p_value']:.4f}")
# A significant p-value would suggest at least one group's d-prime differs.
```

##### Example 4: Comparing Groups with Different Protocols
The comparison is still valid as d-prime is the common scale.

```python
correct_mix_proto = [30, 28] # 2AFC: 30/50 (Pc=0.6), Triangle: 28/40 (Pc=0.7)
total_mix_proto = [50, 40]
protocol_mix_proto = ['2afc', 'triangle']
result_mix_proto = dprime_compare(correct_mix_proto, total_mix_proto, protocol_mix_proto)

print(f"\\n--- Comparing Groups with Different Protocols (2AFC vs Triangle) ---")
print(f"LR Statistic: {result_mix_proto['LR_statistic']:.4f}")
print(f"df: {result_mix_proto['df']}") # Should be 2 - 1 = 1
print(f"P-value: {result_mix_proto['p_value']:.4f}")
```
**Comments:**
*   The Likelihood Ratio Test compares the fit of a model where all groups share a common d-prime against a model where each group has its own d-prime. This comparison is valid even if the protocols (and thus the relationship between d-prime and Pc) differ between groups, because d-prime itself is considered the underlying measure of sensory difference on a common scale.

##### Example 5: Clear Difference vs. No Clear Difference in d-primes
Illustrating how p-values reflect the evidence for/against the null hypothesis (H0: all d-primes are equal).

```python
# Case A: No clear difference expected
correct_nodiff = [30, 32] # Fairly similar performance (Pc=0.6, Pc=0.64 for 2AFC)
total_nodiff = [50, 50]
protocol_nodiff = ['2afc', '2afc']
res_nodiff = dprime_compare(correct_nodiff, total_nodiff, protocol_nodiff)
print(f"\\n--- Comparison: No Clear Difference Expected ---")
print(f"P-value: {res_nodiff['p_value']:.4f}") # Expect high p-value

# Case B: Clear difference expected
correct_clear_diff = [20, 45] # Very different performance (Pc=0.4, Pc=0.9 for 2AFC)
total_clear_diff = [50, 50]
protocol_clear_diff = ['2afc', '2afc']
res_clear_diff = dprime_compare(correct_clear_diff, total_clear_diff, protocol_clear_diff)
print(f"\\n--- Comparison: Clear Difference Expected ---")
print(f"P-value: {res_clear_diff['p_value']:.4f}") # Expect low p-value
```
**Comments:**
*   A high p-value (e.g., > 0.05) from `dprime_compare` suggests that there is no statistically significant evidence to reject the hypothesis that all groups have the same underlying d-prime.
*   A low p-value suggests that the d-prime values are not all equal across the groups.
