---
layout: layouts/base.njk
title: "discrim: General d-prime Estimation"
permalink: /discrimination/discrim/
tags: api_doc
---

## Introduction

The `discrim` function provides a general tool for estimating d-prime (d') from various sensory discrimination methods. It uses Maximum Likelihood principles by finding the d-prime value that makes the theoretical proportion correct (pc) match the observed proportion correct for the chosen method. Standard errors and confidence intervals are based on Wald statistics.

This function is a general d-prime estimation tool, similar in purpose to `sensR::discrim()` (from `R/discrim.R`) but with its own specific implementation for parameter estimation and supported methods within `sensPy`.

## Function Signature

```python
discrim(
    correct: int, 
    total: int, 
    method: str, 
    conf_level: float = 0.95, 
    statistic: str = "Wald"
) -> dict
```

## Parameters

*   **`correct`** (int): The number of correct responses observed.
*   **`total`** (int): The total number of trials conducted.
*   **`method`** (str): The discrimination protocol used. Supported methods include:
    *   `"2afc"` (or `"two_afc"`, `"2-afc"`)
    *   `"triangle"`
    *   `"duotrio"`
    *   `"3afc"` (or `"three_afc"`, `"3-afc"`)
    *   `"tetrad"`
    *   `"hexad"`
    *   `"twofive"` (or `"2outoffive"`)
*   **`conf_level`** (float, optional): The desired confidence level for the confidence interval (e.g., 0.95 for 95% CI). Defaults to `0.95`.
*   **`statistic`** (str, optional): The method used for calculating standard errors, confidence intervals, and p-values. Currently, only `"Wald"` is implemented. Defaults to `"Wald"`.

## Return Value

The function returns a dictionary containing the following key-value pairs:

*   **`"dprime"`** (float): The estimated d-prime value.
*   **`"se_dprime"`** (float): The standard error of the estimated d-prime. Can be `np.inf` if `pc_obs` is 0 or 1, or if the derivative of the psychometric function is zero.
*   **`"lower_ci"`** (float): The lower bound of the confidence interval for d-prime.
*   **`"upper_ci"`** (float): The upper bound of the confidence interval for d-prime.
*   **`"p_value"`** (float): The p-value for the Wald test of the hypothesis that d-prime is equal to 0.
*   **`"conf_level"`** (float): The confidence level used for the CI.
*   **`"correct"`** (int): The number of correct responses provided as input.
*   **`"total"`** (int): The total number of trials provided as input.
*   **`"pc_obs"`** (float): The observed proportion correct (`correct / total`).
*   **`"pguess"`** (float): The chance performance (proportion correct by guessing) for the specified method.
*   **`"method"`** (str): The discrimination method used for the calculation.
*   **`"statistic"`** (str): The type of statistic used (e.g., "Wald").

## Usage Examples

```python
import numpy as np
from senspy.discrimination import discrim

# Example 1: 2-AFC task
result_2afc = discrim(correct=40, total=50, method="2afc")
print("--- 2-AFC Example ---")
print(f"d-prime: {result_2afc['dprime']:.4f}")
print(f"SE(d-prime): {result_2afc['se_dprime']:.4f}")
print(f"95% CI: ({result_2afc['lower_ci']:.4f}, {result_2afc['upper_ci']:.4f})")
print(f"P-value (d'=0): {result_2afc['p_value']:.4f}")
# print(result_2afc) # To see the full dictionary

# Example 2: Triangle task
result_triangle = discrim(correct=25, total=50, method="triangle")
print("\\n--- Triangle Example ---")
print(f"d-prime: {result_triangle['dprime']:.4f}")
print(f"Observed Pc: {result_triangle['pc_obs']:.4f}, Pguess: {result_triangle['pguess']:.4f}")
# print(result_triangle)

# Example 3: Duo-trio task
result_duotrio = discrim(correct=30, total=50, method="duotrio")
print("\\n--- Duo-Trio Example ---")
print(f"d-prime: {result_duotrio['dprime']:.4f}")
# print(result_duotrio)

# Example 4: Perfect score in 3-AFC (d-prime and SE might be large/inf)
result_3afc_perfect = discrim(correct=50, total=50, method="3afc")
print("\\n--- 3-AFC Perfect Score Example ---")
print(f"d-prime: {result_3afc_perfect['dprime']}") # Could be a large number or inf
print(f"SE(d-prime): {result_3afc_perfect['se_dprime']}") # Expected to be np.inf
# print(result_3afc_perfect)
```

## Notes on Calculation

*   The d-prime value is estimated by numerically finding the root of the equation `pc_func(dprime) - pc_obs_clipped = 0`, where `pc_func` is the psychometric function for the chosen `method` and `pc_obs_clipped` is the observed proportion correct (adjusted slightly away from 0, `pguess`, or 1 for numerical stability). This root-finding is performed using `scipy.optimize.brentq`.
*   Standard Errors (SEs) are Wald-type estimates. They are calculated using the formula: `SE(d') = sqrt(Var(Pc_obs)) / |dPc/dd'|`, where `Var(Pc_obs)` is the variance of a binomial proportion (`Pc_obs * (1-Pc_obs) / total`), and `dPc/dd'` is the derivative of the psychometric function with respect to d-prime, evaluated at the estimated d-prime. This derivative is computed numerically.
*   If `pc_obs` is 0 or 1, `se_dprime` is set to `np.inf`.

```python
# Example of how to access the full dictionary
# result = discrim(40,50,"2afc")
# for key, value in result.items():
# print(f" {key}: {value}")
```
