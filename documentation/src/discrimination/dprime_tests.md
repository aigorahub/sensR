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


# Example 2: Two groups, likely similar
correct_g2 = [30, 32]
total_g2 = [50, 50]
protocol_g2 = ["triangle", "triangle"]
result_compare_sim = dprime_compare(correct_g2, total_g2, protocol_g2)
print("\\n--- d-prime Comparison (Likely Similar Groups) ---")
print(f"LR Statistic: {result_compare_sim['LR_statistic']:.4f}")
print(f"df: {result_compare_sim['df']}")
print(f"P-value: {result_compare_sim['p_value']:.4f}")

```
