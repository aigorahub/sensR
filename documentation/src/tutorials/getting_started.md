---
layout: layouts/base.njk
title: "Getting Started with sensPy"
permalink: /tutorials/getting-started/
tags: tutorial
---

## Introduction

Welcome to `sensPy`! This library provides tools for sensory discrimination analysis, offering Python-based alternatives to many functions found in the R package `sensR`. This tutorial will walk you through a basic analysis of a sensory test to get you started.

We'll cover:
1.  Estimating d-prime (sensitivity) from raw count data using the `discrim` function.
2.  Interpreting the results, including d-prime, p-value, and confidence intervals.
3.  Converting d-prime to other common scales like Proportion Correct (Pc) and Proportion Discriminated (Pd) using the `rescale` function.

## Scenario

Imagine you've conducted a **triangle test** to see if a new product formulation (Test Product) is noticeably different from a current one (Control). 
*   You had **50 participants**.
*   **25** of them correctly identified the odd sample.

Let's analyze this data using `sensPy`.

## Using `discrim` to Estimate d-prime

The `discrim` function in `senspy.discrimination` allows us to estimate d-prime from this kind of data.

First, let's import the function and our data:

```python
from senspy.discrimination import discrim
import numpy as np # For printing arrays nicely if needed

# Our scenario data
correct_responses = 25
total_trials = 50
test_method = "triangle"

# Perform the analysis
result = discrim(correct=correct_responses, total=total_trials, method=test_method)

# Let's look at the key results
d_prime = result['dprime']
p_value = result['p_value']
ci_lower = result['lower_ci']
ci_upper = result['upper_ci']
pc_observed = result['pc_obs']
pguess = result['pguess']

print(f"Observed Proportion Correct (Pc): {pc_observed:.4f}")
print(f"Chance Performance (Pguess) for {test_method}: {pguess:.4f}")
print(f"Estimated d-prime: {d_prime:.4f}")
print(f"95% Confidence Interval for d-prime: ({ci_lower:.4f}, {ci_upper:.4f})")
print(f"P-value (for H0: d-prime = 0): {p_value:.4f}")
```

**Interpretation of Results:**

*   **Observed Proportion Correct (Pc)**: This is simply `correct_responses / total_trials`. In our case, 25/50 = 0.50.
*   **Chance Performance (Pguess)**: For a triangle test, the probability of correctly identifying the odd sample purely by chance is 1/3 (approximately 0.3333). Our observed Pc (0.50) is above this.
*   **Estimated d-prime**: This is the primary measure of sensitivity. A larger d-prime indicates better discriminability between the Test Product and Control. The output will show the estimated value.
*   **Confidence Interval for d-prime**: This provides a range within which the true d-prime value is likely to lie (with 95% confidence by default).
*   **P-value**: This tests the null hypothesis that d-prime is zero (i.e., no discriminable difference). 
    *   If the p-value is small (typically < 0.05), we reject the null hypothesis and conclude that there is a statistically significant difference between the products.
    *   If the p-value is larger, we don't have enough evidence to say they are different.

Running the code above would give you specific numerical values for these interpretations.

## Using `rescale` to Convert d-prime

Sometimes, d-prime is not the most intuitive scale. We can use the `rescale` function from `senspy.links` to convert our estimated d-prime to Proportion Correct (Pc) or Proportion Discriminated (Pd). Pc is the overall proportion correct expected for that d-prime, and Pd is the proportion of the population that can discriminate the products, adjusted for guessing.

```python
from senspy.links import rescale

# Assuming 'd_prime' is the value obtained from the 'discrim' result above
# and 'test_method' is "triangle"

# Convert d-prime to Pc
rescaled_to_pc = rescale(x=d_prime, from_scale="dp", to_scale="pc", method=test_method)
pc_estimate_from_dprime = rescaled_to_pc['coefficients']['pc']

# Convert d-prime to Pd
rescaled_to_pd = rescale(x=d_prime, from_scale="dp", to_scale="pd", method=test_method)
pd_estimate_from_dprime = rescaled_to_pd['coefficients']['pd']

print(f"\\n--- Rescaled Values for d-prime = {d_prime:.4f} ({test_method} method) ---")
print(f"Equivalent Proportion Correct (Pc): {pc_estimate_from_dprime:.4f}")
print(f"Equivalent Proportion Discriminated (Pd): {pd_estimate_from_dprime:.4f}")
```

**Interpretation of Rescaled Values:**

*   **Equivalent Proportion Correct (Pc)**: This value should be close to our original `pc_obs` (0.50 in this scenario), as `discrim` finds the d-prime that best matches this observed Pc. Small differences can occur due to internal adjustments for numerical stability in `discrim` and `psyinv`.
*   **Equivalent Proportion Discriminated (Pd)**: This tells us the proportion of discriminators in the population after accounting for chance success. For example, if Pd is 0.25, it means 25% of the population can distinguish the products beyond mere guessing.

## Conclusion

This tutorial covered a basic workflow for analyzing data from a triangle test using `sensPy`:
1.  Estimating d-prime and its statistical significance using `senspy.discrimination.discrim`.
2.  Converting d-prime to more interpretable scales like Pc and Pd using `senspy.links.rescale`.

`sensPy` offers many more functions for various discrimination methods, model fitting (like `BetaBinomial` for overdispersed data), and advanced d-prime tests. Explore the documentation for other modules and functions to see how `sensPy` can assist in your sensory analyses!
