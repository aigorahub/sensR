---
layout: layouts/base.njk
title: "Getting Started with sensPy"
permalink: /tutorials/getting-started/
tags: tutorial
---

## Introduction

Welcome to `sensPy`! This library provides tools for sensory discrimination analysis, offering Python-based alternatives to many functions found in the R package `sensR`. This tutorial will walk you through a basic analysis of a sensory test to get you started.

We'll cover:
1.  Installing `sensPy`.
2.  Estimating d-prime (sensitivity) from raw count data using `senspy.discrimination.discrim`.
3.  Understanding the model-based approach using `senspy.models.DiscriminationModel`.
4.  Interpreting the results, including d-prime, p-value, and confidence intervals.
5.  Converting d-prime to other common scales like Proportion Correct (Pc) and Proportion Discriminated (Pd) using `senspy.discrimination.rescale`.

## Installation

You can install `sensPy` directly from its GitHub repository using pip:

```bash
pip install git+https://github.com/aigorahub/sensPy.git
```

Make sure you have Python 3.8 or newer and pip installed.

## Scenario

Imagine you've conducted a **triangle test** to see if a new product formulation (Test Product) is noticeably different from a current one (Control). 
*   You had **50 participants**.
*   **25** of them correctly identified the odd sample.

Let's analyze this data using `sensPy`.

## Using `discrim` to Estimate d-prime (Functional Approach)

The `discrim` function in `senspy.discrimination` allows for a quick estimation of d-prime. This function now internally uses the `DiscriminationModel` (see next section) but returns a dictionary for backward compatibility.

First, let's import the function and define our data:

```python
from senspy.discrimination import discrim
import numpy as np # For printing arrays nicely if needed

# Our scenario data
correct_responses = 25
total_trials = 50
test_method = "triangle"

# Perform the analysis
result_dict = discrim(correct=correct_responses, total=total_trials, method=test_method)

# Let's look at the key results
d_prime = result_dict['dprime']
p_value = result_dict['p_value']
ci_lower = result_dict['lower_ci'] # Wald CI by default
ci_upper = result_dict['upper_ci'] # Wald CI by default
pc_observed = result_dict['pc_obs']
pguess = result_dict['pguess']

print(f"--- Results from discrim() function ---")
print(f"Observed Proportion Correct (Pc): {pc_observed:.4f}")
print(f"Chance Performance (Pguess) for {test_method}: {pguess:.4f}")
print(f"Estimated d-prime: {d_prime:.4f}")
print(f"95% Wald Confidence Interval for d-prime: ({ci_lower:.4f}, {ci_upper:.4f})")
print(f"P-value (for H0: d-prime = 0): {p_value:.4f}")
```

**Interpretation of Results:**
The interpretation remains the same as described previously. The `discrim` function provides a direct way to get these key metrics.

## Using `DiscriminationModel` (Object-Oriented Approach)

`sensPy` now emphasizes an object-oriented approach using model classes available in `senspy.models`. This provides more flexibility and additional methods like `summary()` and `confint()` for different CI types.

Let's re-analyze the same data using `DiscriminationModel`:

```python
from senspy.models import DiscriminationModel

# Our scenario data remains the same
# correct_responses = 25
# total_trials = 50
# test_method = "triangle"

# Create and fit the model
model = DiscriminationModel()
model.fit(correct=correct_responses, total=total_trials, method=test_method)

# Get a summary of the model
print(f"\n--- Results from DiscriminationModel ---")
print(model.summary())

# Get confidence intervals (profile likelihood by default, if available)
# Wald CIs are also available via model.results_dict or by specifying method_ci='wald'
conf_intervals = model.confint(level=0.95, method_ci='profile') # Or 'wald'
dprime_ci = conf_intervals.get('dprime', (np.nan, np.nan))

print(f"Profile Likelihood 95% CI for d-prime: ({dprime_ci[0]:.4f}, {dprime_ci[1]:.4f})")
# The model.dprime attribute holds the MLE estimate, also available in model.results_dict['dprime']
# print(f"MLE d-prime from model: {model.dprime:.4f}")
```

**Advantages of the Model-Based Approach:**
*   **Rich Output**: The `model.summary()` method provides a formatted summary of key results.
*   **Flexible Confidence Intervals**: The `model.confint()` method allows specifying different CI calculation methods (e.g., 'profile' vs. 'wald'). Profile likelihood CIs are generally considered more accurate, especially for smaller sample sizes or when parameters are near boundaries.
*   **Object State**: The fitted model object (`model`) retains all parameters, input data, and results as attributes for easy access.

## Using `rescale` to Convert d-prime

The `rescale` function, now in `senspy.discrimination`, can be used with the d-prime obtained from either method.

```python
from senspy.discrimination import rescale # Updated import

# Assuming 'd_prime' is the value obtained from the 'discrim' result or model.dprime
# and 'test_method' is "triangle"

# Convert d-prime to Pc
rescaled_to_pc = rescale(x=d_prime, from_scale="dp", to_scale="pc", method=test_method)
pc_estimate_from_dprime = rescaled_to_pc['coefficients']['pc']

# Convert d-prime to Pd
rescaled_to_pd = rescale(x=d_prime, from_scale="dp", to_scale="pd", method=test_method)
pd_estimate_from_dprime = rescaled_to_pd['coefficients']['pd']

print(f"\n--- Rescaled Values for d-prime = {d_prime:.4f} ({test_method} method) ---")
print(f"Equivalent Proportion Correct (Pc): {pc_estimate_from_dprime:.4f}")
print(f"Equivalent Proportion Discriminated (Pd): {pd_estimate_from_dprime:.4f}")
```

**Interpretation of Rescaled Values:**
This remains the same. Pc should be close to your observed proportion correct, and Pd represents the proportion of discriminators adjusted for chance.

## Power Analysis

`sensPy` also includes functions for power analysis in `senspy.power`. These functions, like `exact_binomial_power` and `power_discrim`, now return a `PowerResult` object (a namedtuple) which bundles the power, number of trials, alpha level, method type, and detailed parameters used in the calculation. For sample size calculations, functions like `sample_size_discrim` and `power_discrim_normal_approx` (when `power_target` is set) return the required number of trials.

## Conclusion

This tutorial covered:
1.  Installation of `sensPy`.
2.  Estimating d-prime using the functional approach (`senspy.discrimination.discrim`).
3.  Utilizing the object-oriented model-based approach (`senspy.models.DiscriminationModel`) for richer analysis and more robust confidence intervals.
4.  Converting d-prime to other scales using `senspy.discrimination.rescale`.

`sensPy` offers a growing suite of tools for sensory analysis, including various discrimination models (`BetaBinomial`, `TwoACModel`, `DoDModel`, `SameDifferentModel` in `senspy.models`), power calculation functions, and other utilities. Explore the documentation for other modules and functions to see how `sensPy` can assist in your sensory analyses!
