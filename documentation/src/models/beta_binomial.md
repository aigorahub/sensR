---
layout: layouts/base.njk
title: "BetaBinomial Model"
permalink: /models/beta-binomial/
tags: api_doc
---

## Introduction

The `BetaBinomial` model is used for analyzing binomial-type data that exhibits overdispersion, meaning there is more variability in the data than would be expected from a standard binomial distribution. It assumes that the probability of success in each binomial trial itself follows a beta distribution.

This model is equivalent to `sensR::betaBin()` (from `R/betaBin.R`) in the R package `sensR`.

## Initialization and Fitting

The `BetaBinomial` model is typically initialized without specific parameters, which are then estimated from data using the `fit` method.

```python
from senspy.models import BetaBinomial
import numpy as np

# Initialize a model instance
model = BetaBinomial() 
# Default initial alpha and beta are 1.0, but these are primarily starting points for the fit.
```

### `fit(self, x, n, corrected=False, p_guess=0.5)`

This method fits the Beta-Binomial model to the provided data using Maximum Likelihood (ML) estimation.

**Parameters:**

*   `x` (array-like): Observed number of "successes" in each group/trial. Can be a single integer or a list/NumPy array of integers.
*   `n` (array-like): Total number of trials in each group. Must correspond to `x` (scalar if `x` is scalar, or same length array).
*   `corrected` (bool, optional): If `True`, applies a chance-correction to the model. Defaults to `False`.
*   `p_guess` (float, optional): The chance probability (guess rate) to use if `corrected` is `True`. Must be between 0 and 1 (exclusive in current implementation if corrected). Defaults to 0.5.

The fitting process involves optimizing the model's parameters (alpha and beta) to maximize the likelihood of observing the given data.

## Attributes

After a successful fit, the following attributes are populated on the model instance:

*   `alpha` (float): The estimated alpha parameter of the beta distribution.
*   `beta` (float): The estimated beta parameter of the beta distribution.
*   `loglik` (float): The maximized log-likelihood value of the fitted model.
*   `vcov` (np.ndarray): A 2x2 variance-covariance matrix for the *logarithms* of alpha and beta (`log_alpha`, `log_beta`). This is derived from the Hessian matrix obtained during optimization.
*   `params` (np.ndarray): A NumPy array containing the optimized parameters on the log scale: `[log_alpha, log_beta]`.
*   `convergence_status` (bool): `True` if the optimizer successfully converged, `False` otherwise.
*   `x_obs_arr` (np.ndarray): The input `x` data stored as a NumPy array.
*   `n_trials_arr` (np.ndarray): The input `n` data stored as a NumPy array.
*   `corrected` (bool): Whether chance correction was applied.
*   `p_guess` (float | None): The guess rate used if `corrected` was `True`, otherwise `None`.
*   `n_obs` (int): The number of observation groups (e.g., length of `x_obs_arr`).
*   `log_terms_buffer` (list | None): Internal buffer used for corrected model calculations; `None` if not corrected.

## Methods

### `summary()`

Returns a string summarizing the fitted model.

**Content of the Summary:**
*   Model type ("BetaBinomial Model Summary").
*   Estimated coefficients (alpha, beta).
*   Standard Errors (SE) for alpha and beta (derived from `vcov`).
*   Maximized Log-Likelihood.
*   Number of observations.
*   Convergence status of the optimizer.

### `confint(self, parm=None, level=0.95, method='profile')`

Calculates confidence intervals for the model parameters (alpha and/or beta).

**Parameters:**

*   `parm` (list[str] | None, optional): A list of parameter names to calculate CIs for (e.g., `['alpha', 'beta']`). If `None`, CIs for all supported parameters are calculated. Defaults to `None`.
*   `level` (float, optional): The desired confidence level (e.g., 0.95 for 95% CI). Defaults to 0.95.
*   `method` (str, optional): The method for calculating confidence intervals. Currently, only `'profile'` (profile likelihood) is supported. Defaults to `'profile'`.

**Returns:**

*   `dict`: A dictionary where keys are parameter names (e.g., "alpha", "beta") and values are tuples `(lower_bound, upper_bound)` for the confidence interval.

## Usage Examples

```python
import numpy as np
from senspy.models import BetaBinomial

# Example Data
x_data = np.array([20, 25, 30])
n_data = np.array([50, 55, 60])

# --- Standard Beta-Binomial Model ---
print("\\n--- Standard Model ---")
std_model = BetaBinomial()
std_model.fit(x_data, n_data)

print(f"Fitted Alpha: {std_model.alpha:.4f}")
print(f"Fitted Beta: {std_model.beta:.4f}")
print(f"Log-Likelihood: {std_model.loglik:.4f}")
print(f"Convergence: {std_model.convergence_status}")

# Get and print summary
print("\\nModel Summary:")
print(std_model.summary())

# Get and print confidence intervals
print("\\nConfidence Intervals (95%):")
std_ci = std_model.confint()
print(f"  Alpha: ({std_ci['alpha'][0]:.4f}, {std_ci['alpha'][1]:.4f})")
print(f"  Beta:  ({std_ci['beta'][0]:.4f}, {std_ci['beta'][1]:.4f})")


# --- Chance-Corrected Beta-Binomial Model ---
print("\\n--- Chance-Corrected Model (p_guess=0.25) ---")
# Using different data for illustration if needed, or same data with correction
x_corr_data = np.array([35, 40]) # Example: higher success counts
n_corr_data = np.array([50, 50])
p_guess_rate = 0.25

corr_model = BetaBinomial()
corr_model.fit(x_corr_data, n_corr_data, corrected=True, p_guess=p_guess_rate)

print(f"Fitted Alpha (corrected): {corr_model.alpha:.4f}")
print(f"Fitted Beta (corrected): {corr_model.beta:.4f}")
print(f"Log-Likelihood (corrected): {corr_model.loglik:.4f}")
print(f"Convergence (corrected): {corr_model.convergence_status}")

# Get and print summary for corrected model
print("\\nCorrected Model Summary:")
print(corr_model.summary())

# Get and print confidence intervals for corrected model
print("\\nCorrected Model Confidence Intervals (95%):")
corr_ci = corr_model.confint()
# Note: CI calculation can be sensitive for corrected models if data is sparse or near boundaries
if corr_ci['alpha'] is not None and not any(np.isnan(corr_ci['alpha'])):
    print(f"  Alpha: ({corr_ci['alpha'][0]:.4f}, {corr_ci['alpha'][1]:.4f})")
else:
    print("  Alpha: CI could not be reliably computed.")
if corr_ci['beta'] is not None and not any(np.isnan(corr_ci['beta'])):
    print(f"  Beta:  ({corr_ci['beta'][0]:.4f}, {corr_ci['beta'][1]:.4f})")
else:
    print("  Beta:  CI could not be reliably computed.")

```

### Example 1: Fitting with a Different Dataset

Let's try a dataset that might exhibit more overdispersion or a different underlying mean proportion. Here, the number of trials (`n`) is constant.

```python
# Example 1: Different Dataset
x_new = np.array([5, 8, 12, 15]) # Successes out of 20 trials each
n_new = np.array([20, 20, 20, 20]) # Could also be n_new = 20

model_new = BetaBinomial()
model_new.fit(x_new, n_new)

print("\\n--- Fit with New Dataset ---")
print(model_new.summary())
```

**Comments:**
*   Depending on the data's characteristics (e.g., if proportions `x_new/n_new` vary significantly), the estimated `alpha` and `beta` values will change.
*   Lower `alpha` and `beta` values generally indicate more overdispersion. Higher values suggest the data is closer to a standard binomial distribution.

### Example 2: Interpreting `confint()` for Specific Parameters

You can request confidence intervals for specific parameters.

```python
# Example 2: Confidence Interval for a single parameter
# Using the 'std_model' from the first example
if std_model.convergence_status: # Ensure model converged before getting CIs
    print("\\n--- CI for Alpha only (Standard Model) ---")
    ci_alpha_only = std_model.confint(parm=['alpha'])
    alpha_ci = ci_alpha_only.get('alpha', (np.nan, np.nan)) # Safely get the tuple
    if not any(np.isnan(alpha_ci)):
        print(f"  Alpha (95% CI): ({alpha_ci[0]:.4f}, {alpha_ci[1]:.4f})")
    else:
        print("  Alpha: CI could not be reliably computed.")
    # The output is still a dictionary, but only contains the requested parameter.
else:
    print("\\nStandard model did not converge, skipping specific CI example.")
```

### Example 3: Chance-Corrected Model with a High Guess Rate

This example demonstrates a chance-corrected model where the probability of guessing correctly is unusually high (e.g., 0.75).

```python
# Example 3: Corrected Model with High Guess Rate
x_high_pg = np.array([30, 35]) # Number of correct identifications
n_high_pg = np.array([50, 50]) # Total trials
p_guess_high = 0.75 # e.g. a 4-AFC task where 3 choices are very obviously wrong

model_high_pg = BetaBinomial()
try:
    model_high_pg.fit(x_high_pg, n_high_pg, corrected=True, p_guess=p_guess_high)
    print("\\n--- Corrected Model (High p_guess=0.75) ---")
    print(model_high_pg.summary())
    
    print("\\nConfidence Intervals (High p_guess):")
    ci_high_pg = model_high_pg.confint()
    alpha_ci_hpg = ci_high_pg.get('alpha', (np.nan, np.nan))
    beta_ci_hpg = ci_high_pg.get('beta', (np.nan, np.nan))

    if not any(np.isnan(alpha_ci_hpg)):
        print(f"  Alpha: ({alpha_ci_hpg[0]:.4f}, {alpha_ci_hpg[1]:.4f})")
    else:
        print("  Alpha: CI could not be reliably computed.")
    if not any(np.isnan(beta_ci_hpg)):
        print(f"  Beta:  ({beta_ci_hpg[0]:.4f}, {beta_ci_hpg[1]:.4f})")
    else:
        print("  Beta:  CI could not be reliably computed.")
except ValueError as e:
    print(f"\\nError fitting high p_guess model: {e}")

```
**Comments:**
*   A high `p_guess` means a significant portion of correct responses could be due to chance.
*   This can lead to different `alpha` and `beta` estimates compared to a non-corrected model or one with a lower `p_guess`.
*   The confidence intervals might become wider, reflecting increased uncertainty about the true underlying parameters once the high chance of guessing is accounted for. If `p_guess` is very close to or exceeds the observed proportion correct, the model may not be identifiable, leading to convergence issues or extremely wide/unreliable CIs. The current `fit` method restricts `p_guess` to be strictly between 0 and 1 when `corrected=True`.

### Example 4: Conceptual Access to `vcov` for Standard Errors

The `summary()` method conveniently provides standard errors for `alpha` and `beta`. However, it's useful to understand that `model.vcov` stores the variance-covariance matrix of the *logarithms* of these parameters (`log_alpha`, `log_beta`), as these are often what is estimated during optimization for stability.

```python
# Example 4: Conceptual SEs from vcov
# Using the 'std_model' from the first example, assuming it converged.
if std_model.convergence_status and std_model.vcov is not None:
    print("\\n--- Conceptual SEs from vcov (Standard Model) ---")
    log_params_vcov = std_model.vcov 
    
    # SE for log_alpha and log_beta are sqrt of diagonal elements
    # Ensure diagonal elements are non-negative before sqrt
    se_log_alpha = np.sqrt(log_params_vcov[0,0]) if log_params_vcov[0,0] >= 0 else np.nan
    se_log_beta = np.sqrt(log_params_vcov[1,1]) if log_params_vcov[1,1] >= 0 else np.nan
    
    print(f"  Raw vcov of [log_alpha, log_beta]:")
    print(log_params_vcov)
    print(f"  SE for log_alpha (from vcov): {se_log_alpha:.4f}")
    print(f"  SE for log_beta (from vcov): {se_log_beta:.4f}")
    
    print("\\n  Note: The `summary()` method already provides standard errors for alpha and beta")
    print("  on their natural scale, transformed from the log-scale vcov via the delta method.")
else:
    print("\\nStandard model did not converge or vcov not available, skipping vcov example.")
```
This example is primarily for understanding the underlying components. For practical use, the SEs in the `summary()` output are generally sufficient.
