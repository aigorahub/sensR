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

### More Examples for Different Methods

Here are examples for other supported discrimination methods:

```python
# Example for "3afc" method
result_3afc = discrim(correct=20, total=50, method="3afc")
print("\\n--- 3-AFC Example (20/50 correct) ---")
# print(f"3AFC Results (20/50): {result_3afc}") # Full dictionary
print(f"d-prime: {result_3afc['dprime']:.4f}, Pc: {result_3afc['pc_obs']:.2f}, Pguess: {result_3afc['pguess']:.2f}")


# Example for "tetrad" method
result_tetrad = discrim(correct=22, total=50, method="tetrad")
print("\\n--- Tetrad Example (22/50 correct) ---")
# print(f"Tetrad Results: {result_tetrad}")
print(f"d-prime: {result_tetrad['dprime']:.4f}, Pc: {result_tetrad['pc_obs']:.2f}, Pguess: {result_tetrad['pguess']:.2f}")

# Example for "hexad" method (uses polynomial approximation)
result_hexad = discrim(correct=15, total=50, method="hexad")
print("\\n--- Hexad Example (15/50 correct) ---")
# print(f"Hexad Results: {result_hexad}")
print(f"d-prime: {result_hexad['dprime']:.4f}, Pc: {result_hexad['pc_obs']:.2f}, Pguess: {result_hexad['pguess']:.2f}")

# Example for "twofive" (2-out-of-5) method (uses polynomial approximation)
result_twofive = discrim(correct=18, total=50, method="twofive")
print("\\n--- 2-out-of-5 Example (18/50 correct) ---")
# print(f"2-out-of-5 Results: {result_twofive}")
print(f"d-prime: {result_twofive['dprime']:.4f}, Pc: {result_twofive['pc_obs']:.2f}, Pguess: {result_twofive['pguess']:.2f}")
```

### Deeper Interpretation of Results

Let's revisit the triangle test example and explore how to interpret the p-value and confidence interval.

```python
# Using result_triangle from a previous example:
# result_triangle = discrim(correct=25, total=50, method="triangle") 
# Ensure this variable is defined if running standalone, or use fresh data:
result_triangle_interpret = discrim(correct=25, total=50, method="triangle")

print("\\n--- Deeper Interpretation (Triangle Test: 25/50 correct) ---")
print(f"Estimated d-prime: {result_triangle_interpret['dprime']:.2f}")

if result_triangle_interpret['p_value'] < 0.05:
    print(f"The result is statistically significant (p={result_triangle_interpret['p_value']:.3f}), "
          f"suggesting a discernible difference between the samples.")
else:
    print(f"The result is not statistically significant (p={result_triangle_interpret['p_value']:.3f}), "
          f"meaning there isn't strong evidence to conclude a discernible difference based on this test.")

print(f"The 95% confidence interval for d-prime is "
      f"[{result_triangle_interpret['lower_ci']:.2f}, {result_triangle_interpret['upper_ci']:.2f}].")
print(f"Observed Pc: {result_triangle_interpret['pc_obs']:.2f}, "
      f"Chance Pc (pguess): {result_triangle_interpret['pguess']:.2f}")
```

**Explanation:**
*   **P-value**: The p-value helps determine if the observed result is likely due to chance or if there's a real sensory difference. A small p-value (e.g., < 0.05) indicates that the estimated d-prime is significantly different from zero.
*   **Confidence Interval (CI)**: The CI provides a range of plausible values for the true d-prime. If the CI does not include zero, it's another way to infer statistical significance (that d-prime is likely not zero). A wide CI might suggest more uncertainty in the d-prime estimate, often due to a smaller number of trials or results close to chance.

### Edge Case: Performance at Chance Level

What happens if the number of correct responses is exactly what we'd expect by chance?

```python
# Example for Triangle test (pguess = 1/3)
# For 50 trials, chance performance is approx. 50/3 = 16.67. Let's use 17.
result_chance = discrim(correct=17, total=50, method="triangle")
print("\\n--- Edge Case: Performance at Chance (Triangle, 17/50 correct) ---")
# print(f"Chance Performance Results: {result_chance}")
print(f"d-prime: {result_chance['dprime']:.4f}")
print(f"P-value: {result_chance['p_value']:.4f}")
print(f"Observed Pc: {result_chance['pc_obs']:.2f}, Pguess: {result_chance['pguess']:.2f}")
```
**Comments:**
*   When performance is at or very near chance level, the estimated `dprime` should be close to 0.
*   The `p_value` should be high (e.g., > 0.05), indicating no significant evidence of discriminability.
*   The confidence interval for d-prime will likely include 0.

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
