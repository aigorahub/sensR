---
layout: layouts/base.njk
title: "Psychometric Link Utilities"
permalink: /links/utilities/
tags: api_doc
---

## Introduction

The `senspy.links` module provides core psychometric utility functions essential for sensory data analysis. These functions allow for conversions between d-prime (d', a measure of sensitivity) and proportion correct (Pc) or proportion discriminated (Pd) for various sensory discrimination methods. Additionally, a function is provided for rescaling values and their associated standard errors across these different measurement scales.

These utilities are fundamental to many of the estimation and testing procedures within `sensPy` and are analogous to functions found in `R/links.R` and some utilities in `R/utils.R` of the `sensR` package.

---

## `psyfun`: d-prime to Proportion Correct

### Purpose

The `psyfun` function converts a given d-prime value to the expected proportion of correct responses (Pc) for a specified sensory discrimination method. It essentially evaluates the psychometric function at a given d-prime.

### Function Signature

```python
psyfun(dprime: float, method: str = "2afc") -> float
```

### Parameters

*   **`dprime`** (float): The sensitivity index (d-prime).
*   **`method`** (str, optional): The discrimination protocol used. Defaults to `"2afc"`. Supported methods include:
    *   `"2afc"` (or `"two_afc"`, `"2-afc"`)
    *   `"triangle"`
    *   `"duotrio"`
    *   `"3afc"` (or `"three_afc"`, `"3-afc"`)
    *   `"tetrad"`
    *   `"hexad"` (polynomial approximation)
    *   `"twofive"` (or `"2outoffive"`, polynomial approximation)

### Return Value

*   (float): The expected proportion correct (Pc) for the given d-prime and method.

### Example

```python
from senspy.links import psyfun

# Calculate Pc for d-prime = 1.5 in a triangle task
pc_value = psyfun(dprime=1.5, method="triangle")
print(f"Proportion correct for d'=1.5 (triangle): {pc_value:.4f}")

# Calculate Pc for d-prime = 1.0 in a 2AFC task
pc_2afc = psyfun(dprime=1.0, method="2afc")
print(f"Proportion correct for d'=1.0 (2AFC): {pc_2afc:.4f}")
```

#### Illustrative Psychometric Function (2AFC)
<!-- TODO: Add illustrative_psychometric_function_2afc.png here once script execution is resolved -->
The `psyfun` for the "2afc" method shows a characteristic S-shaped curve, starting from Pc=0.5 (chance) at d'=0 and asymptotically approaching Pc=1.0 for large d-prime values.

#### Further Examples for `psyfun`, `psyinv`, and `psyderiv`

##### Example 1: Cross-Method Comparison for `psyfun`
The proportion correct (Pc) for the same d-prime value varies across different sensory discrimination methods due to differences in their underlying Thurstonian models and chance performance levels (`pguess`).

```python
from senspy.links import psyfun, psyinv, psyderiv # Ensure all are imported
import numpy as np # For np.nan, np.isclose

d_val = 1.0
pc_2afc = psyfun(d_val, method="2afc")
pc_triangle = psyfun(d_val, method="triangle")
pc_duotrio = psyfun(d_val, method="duotrio")
print(f"For d'={d_val}: Pc(2afc)={pc_2afc:.3f}, Pc(triangle)={pc_triangle:.3f}, Pc(duotrio)={pc_duotrio:.3f}")
```
**Comments:**
*   For `d'=1.0`, Pc(2afc) will be `norm.cdf(1/sqrt(2)) approx 0.760`.
*   Pc(triangle) and Pc(duotrio) will be lower for the same d-prime because their chance levels (1/3 and 1/2 respectively, before mapping to d') are different, and the Thurstonian models are distinct.

##### Example 2: Inverse Relationship (`psyinv(psyfun(...))` Check)
The `psyinv` function should ideally return the original d-prime if you convert `d-prime -> Pc -> d-prime`.

```python
d_original = 1.5
method_example = "tetrad" # Using tetrad as an example

pc_val = psyfun(d_original, method=method_example)
d_recalculated = psyinv(pc_val, method=method_example)

print(f"Original d': {d_original}, Pc from psyfun: {pc_val:.4f}")
print(f"Recalculated d' from psyinv(Pc): {d_recalculated:.4f} for {method_example}")
assert np.isclose(d_original, d_recalculated), "Recalculated d-prime should match original."
```
**Comments:**
*   `d_recalculated` should be very close to `d_original`. Minor differences might occur due to floating-point precision and the numerical root-finding in `psyinv`.

##### Example 3: `psyderiv` for Different d-primes and Methods
The derivative `dPc/dd'` indicates the slope of the psychometric function. It's often maximal at an intermediate d-prime (or Pc) and smaller at very low or very high d-primes.

```python
deriv_low_d_2afc = psyderiv(dprime=0.5, method="2afc")
deriv_high_d_2afc = psyderiv(dprime=2.0, method="2afc")
deriv_low_d_3afc = psyderiv(dprime=0.5, method="3afc") # Different method

print(f"psyderiv(dprime=0.5, method='2afc'): {deriv_low_d_2afc:.4f}")
print(f"psyderiv(dprime=2.0, method='2afc'): {deriv_high_d_2afc:.4f}")
print(f"psyderiv(dprime=0.5, method='3afc'): {deriv_low_d_3afc:.4f}")
```
**Comments:**
*   For many methods like "2afc", the psychometric function is steepest (derivative is maximal) around `d_prime=0` if `pc_func(0)=pguess` and `pguess=0.5`. The derivative generally decreases as `d_prime` moves away from the steepest point.
*   The magnitude of the derivative also depends on the specific method.

---

## `psyinv`: Proportion Correct to d-prime

### Purpose

The `psyinv` function is the inverse of `psyfun`. It converts an observed or theoretical proportion of correct responses (Pc) back to the corresponding d-prime value for a specified discrimination method. This is often used in d-prime estimation.

### Function Signature

```python
psyinv(pc: float, method: str = "2afc") -> float
```

### Parameters

*   **`pc`** (float): The proportion of correct responses. Must be between the chance level (`pguess`) for the method and 1.0.
*   **`method`** (str, optional): The discrimination protocol used. Defaults to `"2afc"`. Supported methods are the same as for `psyfun`.

### Return Value

*   (float): The estimated d-prime value. Returns `0.0` if `pc <= pguess`, and `np.inf` if `pc` is effectively 1.0 (and the method allows for perfect performance). Can return `np.nan` if `pc` is invalid or `brentq` fails.

### Example

```python
from senspy.links import psyinv

# Calculate d-prime for Pc = 0.8 in a duo-trio task
d_prime_value = psyinv(pc=0.8, method="duotrio")
print(f"d-prime for Pc=0.8 (duo-trio): {d_prime_value:.4f}")

# Calculate d-prime for Pc = 0.65 in a 3AFC task
d_prime_3afc = psyinv(pc=0.65, method="3afc")
print(f"d-prime for Pc=0.65 (3AFC): {d_prime_3afc:.4f}")
```

*(See "Further Examples for `psyfun`, `psyinv`, and `psyderiv`" above for more examples)*

---

## `psyderiv`: Derivative of the Psychometric Function

### Purpose

The `psyderiv` function calculates the derivative of the psychometric function (`dPc/dd'`) at a given d-prime value for a specified discrimination method. This derivative is a crucial component in calculating the standard error of d-prime estimates, particularly when using the delta method.

### Function Signature

```python
psyderiv(dprime: float, method: str = "2afc") -> float
```

### Parameters

*   **`dprime`** (float): The sensitivity index (d-prime) at which to evaluate the derivative.
*   **`method`** (str, optional): The discrimination protocol used. Defaults to `"2afc"`. Supported methods are the same as for `psyfun`.

### Return Value

*   (float): The value of the derivative `dPc/dd'` at the specified d-prime. Returns `0.0` if `dprime` is non-finite or if the function is flat at that point.

### Example

```python
from senspy.links import psyderiv

# Calculate the derivative for d-prime = 1.0 in a 2AFC task
derivative_value = psyderiv(dprime=1.0, method="2afc")
print(f"Derivative at d'=1.0 (2AFC): {derivative_value:.4f}")

# Calculate the derivative for d-prime = 1.5 in a triangle task
derivative_triangle = psyderiv(dprime=1.5, method="triangle")
print(f"Derivative at d'=1.5 (triangle): {derivative_triangle:.4f}")
```

*(See "Further Examples for `psyfun`, `psyinv`, and `psyderiv`" above for more examples)*

---

## `rescale`: Rescale Values and Standard Errors

### Purpose

The `rescale` function converts a sensory discrimination measure (and optionally its standard error) from one scale (d-prime, proportion correct Pc, or proportion discriminated Pd) to the others. It is method-aware, meaning the conversions depend on the specified discrimination protocol.

### Function Signature

```python
rescale(
    x: float, 
    from_scale: str, 
    to_scale: str, 
    method: str = "2afc", 
    std_err: float | None = None
) -> dict
```

### Parameters

*   **`x`** (float): The input value to be rescaled.
*   **`from_scale`** (str): The scale of the input `x`. Must be one of `"dp"` (d-prime), `"pc"` (proportion correct), or `"pd"` (proportion discriminated).
*   **`to_scale`** (str): The target scale to which `x` should be primarily converted (though values on all scales are returned). Must be one of `"dp"`, `"pc"`, or `"pd"`.
*   **`method`** (str, optional): The discrimination protocol context for the conversions. Defaults to `"2afc"`. Supported methods are the same as for `psyfun`.
*   **`std_err`** (float | None, optional): The standard error of the input value `x`. If provided, standard errors for all three scales (`se_dp`, `se_pc`, `se_pd`) are calculated using the delta method. Defaults to `None`.

### Return Value

A dictionary containing the rescaled values and their standard errors:

*   **`"coefficients"`** (dict): A dictionary with the input value `x` expressed on all three scales:
    *   `"pc"`: Proportion correct.
    *   `"pd"`: Proportion discriminated.
    *   `"d_prime"`: d-prime.
*   **`"std_errors"`** (dict): A dictionary with the standard errors for each scale if `std_err` was provided, otherwise `np.nan` for all:
    *   `"pc"`: Standard error for Pc.
    *   `"pd"`: Standard error for Pd.
    *   `"d_prime"`: Standard error for d-prime.
*   **`"method"`** (str): The discrimination method used for the conversions.
*   **`"input_scale"`** (str): The `from_scale` provided.
*   **`"output_scale"`** (str): The `to_scale` provided (mainly for context, as all scales are in `"coefficients"`).

### Usage Examples

```python
from senspy.links import rescale

# Example 1: Convert Pc to d-prime and Pd for a triangle task
pc_val = 0.75
result1 = rescale(x=pc_val, from_scale="pc", to_scale="dp", method="triangle")
print(f"--- Rescaling Pc={pc_val} (triangle) ---")
print(f"  d-prime: {result1['coefficients']['d_prime']:.4f}")
print(f"  Pd: {result1['coefficients']['pd']:.4f}")

# Example 2: Convert d-prime to Pc and Pd for a 2AFC task, with standard error
d_prime_val = 1.2
se_d_prime = 0.15
result2 = rescale(x=d_prime_val, from_scale="dp", to_scale="pc", method="2afc", std_err=se_d_prime)
print(f"\\n--- Rescaling d'={d_prime_val} +/- {se_d_prime} (2AFC) ---")
print(f"  Pc: {result2['coefficients']['pc']:.4f} +/- {result2['std_errors']['pc']:.4f}")
print(f"  Pd: {result2['coefficients']['pd']:.4f} +/- {result2['std_errors']['pd']:.4f}")
print(f"  d-prime: {result2['coefficients']['d_prime']:.4f} +/- {result2['std_errors']['d_prime']:.4f}")
```

#### Further Examples for `rescale`

##### Example 3: Rescale `pd` to `dprime` and `pc` with `std_err`
This example shows how to convert a Proportion Discriminated (Pd) value and its standard error to the d-prime and Pc scales for a "duotrio" task.

```python
pd_val = 0.6
se_pd = 0.05
method_rescale_duo = "duotrio"

rescaled_from_pd = rescale(x=pd_val, from_scale="pd", to_scale="dp", 
                           method=method_rescale_duo, std_err=se_pd)

print(f"\\n--- Rescaled from Pd={pd_val} +/- {se_pd} ({method_rescale_duo}) ---")
print(f"  d-prime: {rescaled_from_pd['coefficients']['d_prime']:.3f} +/- {rescaled_from_pd['std_errors']['d_prime']:.3f}")
print(f"  Pc:      {rescaled_from_pd['coefficients']['pc']:.3f} +/- {rescaled_from_pd['std_errors']['pc']:.3f}")
print(f"  Pd:      {rescaled_from_pd['coefficients']['pd']:.3f} +/- {rescaled_from_pd['std_errors']['pd']:.3f}")
```

##### Example 4: Rescale from `dprime` for a method with lower `pguess` (e.g., "hexad")
Methods like "hexad" have a lower chance performance level (`pguess`). This affects the relationship between d-prime, Pc, and Pd.

```python
dp_val_hex = 2.0
se_dp_hex = 0.2
method_hexad = "hexad" # pguess for hexad is 0.1

rescaled_hexad = rescale(x=dp_val_hex, from_scale="dp", to_scale="pc", 
                         method=method_hexad, std_err=se_dp_hex)

print(f"\\n--- Rescaled from d'={dp_val_hex} +/- {se_dp_hex} ({method_hexad}) ---")
print(f"  Pc:      {rescaled_hexad['coefficients']['pc']:.3f} +/- {rescaled_hexad['std_errors']['pc']:.3f}")
print(f"  Pd:      {rescaled_hexad['coefficients']['pd']:.3f} +/- {rescaled_hexad['std_errors']['pd']:.3f}")
print(f"  d-prime: {rescaled_hexad['coefficients']['d_prime']:.3f} +/- {rescaled_hexad['std_errors']['d_prime']:.3f}")
```
**Comments:**
*   For a given d-prime, methods with lower `pguess` values will generally yield lower Pc values compared to methods like "2afc" (where `pguess=0.5`). The Pd value, which corrects for chance, provides a more standardized measure of discriminability across methods.

##### Example 5: `to_scale` parameter's role
The `to_scale` parameter specifies the primary scale of interest for the output, but the `rescale` function in `sensPy` always returns values on all three scales (`pc`, `pd`, `d_prime`) in the `coefficients` dictionary.

```python
result_all_scales = rescale(x=0.75, from_scale="pc", to_scale="dp", method="triangle")

print("\\n--- 'to_scale' parameter and accessing all coefficients ---")
print(f"Input: Pc=0.75, Method=triangle, to_scale='dp'")
print(f"  Coefficients returned:")
print(f"    d-prime: {result_all_scales['coefficients']['d_prime']:.4f}")
print(f"    Pc:      {result_all_scales['coefficients']['pc']:.4f}") # This is the input Pc
print(f"    Pd:      {result_all_scales['coefficients']['pd']:.4f}")
# The 'to_scale' argument is mainly for context or if a single value was expected,
# but sensPy's rescale provides all three for convenience.
```
**Comments:**
*   This behavior differs slightly from `sensR`'s `rescale` which typically returns a single numeric value corresponding to `to_scale`. `sensPy`'s version returns a dictionary with all conversions, making `to_scale` less critical for data retrieval but still useful for indicating the primary conversion of interest.
