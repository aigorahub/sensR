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
