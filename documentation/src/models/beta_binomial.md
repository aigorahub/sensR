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
