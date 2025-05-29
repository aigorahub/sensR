---
layout: layouts/base.njk
title: "dod: Degree of Difference Model"
permalink: /discrimination/dod/
tags: api_doc
---

## Introduction

The `dod` function fits the Thurstonian model for the Degree of Difference (DoD) method. This method is used when observers rate the perceived difference between pairs of stimuli, some of which are physically identical ("same" pairs) and some of which are physically different ("different" pairs). The model estimates d-prime (d', representing sensory sensitivity to the difference) and a set of category boundary parameters (`tau`) using Maximum Likelihood estimation.

This function replaces `sensR::dod()` (from `R/dod.R`) in the R package `sensR`.

## Function Signature

```python
dod(
    same_counts: np.ndarray, 
    diff_counts: np.ndarray, 
    initial_tau: np.ndarray | None = None, 
    initial_d_prime: float | None = None, 
    method: str = "ml", 
    conf_level: float = 0.95
) -> dict
```

## Parameters

*   **`same_counts`** (np.ndarray): A 1D NumPy array of observed counts for "same" stimulus pairs, distributed across `K` response categories.
*   **`diff_counts`** (np.ndarray): A 1D NumPy array of observed counts for "different" stimulus pairs, distributed across `K` response categories. Must be the same length as `same_counts`.
*   **`initial_tau`** (np.ndarray | None, optional): Initial guesses for the `K-1` tau (category boundary) parameters. If provided, must be a 1D array of positive, strictly increasing values. Defaults to `None`, in which case internal defaults based on `sensR`'s `initTau` logic are used.
*   **`initial_d_prime`** (float | None, optional): Initial guess for the d-prime parameter. Defaults to `1.0`.
*   **`method`** (str, optional): Estimation method. Currently, only `"ml"` (Maximum Likelihood) is implemented and supported. Defaults to `"ml"`.
*   **`conf_level`** (float, optional): The desired confidence level for confidence intervals (currently retained for future use, as CIs are not yet part of the return value for `dod`). Defaults to `0.95`.

## Model Details

The DoD model estimates `d_prime` (sensitivity) and `K-1` category boundary parameters, denoted as `tau_1, tau_2, ..., tau_{K-1}`, for `K` response categories. These `tau` values represent the criteria on an internal sensory continuum that separate the response categories. They are constrained to be positive and strictly increasing: `0 < tau_1 < tau_2 < ... < tau_{K-1}`.

The probability of responding in a particular category `k` depends on whether the stimulus pair was "same" or "different", and is calculated based on areas under normal distributions defined by `d_prime` and `tau`. Specifically, cumulative probabilities (`gamma`) are first determined:

*   For "same" pairs: `gamma_same_j = 2 * Phi(tau_j / sqrt(2)) - 1`
*   For "different" pairs: `gamma_diff_j = Phi((tau_j - d_prime) / sqrt(2)) - Phi((-tau_j - d_prime) / sqrt(2))`

Where `Phi` is the standard normal CDF, and `tau_j` is the j-th boundary. The probability for each category `k` (e.g., `p_same_k`) is then the difference between these cumulative probabilities at successive boundaries (`gamma_k - gamma_{k-1}`, with `gamma_0 = 0` and `gamma_K = 1`). The `senspy.discrimination.par2prob_dod` function handles these calculations.

The parameters are estimated by maximizing the multinomial log-likelihood of observing the given `same_counts` and `diff_counts`. Internally, the optimization is performed on `d_prime` and `tpar` (increments of `tau`, where `tau_k = cumsum(tpar_k)`), with bounds ensuring `tpar_i > 0` and `d_prime > 0`.

## Return Value

The function returns a dictionary containing detailed results of the model fitting:

*   **`"d_prime"`** (float): The estimated sensitivity parameter (d').
*   **`"tau"`** (np.ndarray): A 1D array of the estimated `K-1` category boundary parameters. These are derived from the optimized `tpar` values (`tau = np.cumsum(tpar)`).
*   **`"se_d_prime"`** (float): The standard error of the estimated `d_prime`.
*   **`"tpar"`** (np.ndarray): The optimized `tpar` values (increments of `tau`). These are the parameters directly optimized, alongside `d_prime`.
*   **`"se_tpar"`** (np.ndarray): A 1D array of standard errors for the `tpar` increments.
*   **`"loglik"`** (float): The maximized log-likelihood value of the fitted model.
*   **`"vcov_optim_params"`** (np.ndarray): The variance-covariance matrix for the optimized parameters (`tpar`s and `d_prime`).
*   **`"convergence_status"`** (bool): `True` if the optimization algorithm successfully converged, `False` otherwise.
*   **`"initial_params_optim"`** (np.ndarray): The initial parameters `[tpar_init_1, ..., tpar_init_{K-1}, d_prime_init]` used to start the optimization.
*   **`"optim_result"`** (scipy.optimize.OptimizeResult): The full result object from `scipy.optimize.minimize` for detailed inspection.
*   **`"same_counts"`** (np.ndarray): The input `same_counts` data.
*   **`"diff_counts"`** (np.ndarray): The input `diff_counts` data.
*   **`"method"`** (str): The estimation method used (e.g., "ml").
*   **`"conf_level"`** (float): The confidence level specified (retained for context).

## Usage Examples

```python
import numpy as np
from senspy.discrimination import dod

# Example Data for a 3-category scale (K=3, so K-1=2 tau values)
same_counts_data = np.array([10, 20, 70])  # Counts for "same" pairs in categories 1, 2, 3
diff_counts_data = np.array([70, 20, 10])  # Counts for "different" pairs in categories 1, 2, 3

# Fit the DoD model
result_dod = dod(same_counts_data, diff_counts_data)

# Print the full result dictionary
print("--- Full DoD Result ---")
for key, value in result_dod.items():
    if isinstance(value, np.ndarray):
        print(f"  {key}: {np.array2string(value, precision=4, suppress_small=True)}")
    elif isinstance(value, float):
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value}")

# Access specific key estimates
print("\\n--- Key Estimates ---")
if result_dod["convergence_status"]:
    print(f"Estimated d-prime: {result_dod['d_prime']:.4f} (SE: {result_dod['se_d_prime']:.4f})")
    # Tau values are derived from tpar
    print(f"Estimated tau values: {np.array2string(result_dod['tau'], precision=4, suppress_small=True)}")
    # tpar are the directly optimized increments for tau
    # print(f"Estimated tpar values: {np.array2string(result_dod['tpar'], precision=4, suppress_small=True)}")
    # print(f"SE for tpar values: {np.array2string(result_dod['se_tpar'], precision=4, suppress_small=True)}")
    print(f"Log-Likelihood: {result_dod['loglik']:.4f}")
else:
    print("DoD model optimization did not converge. Estimates may be unreliable.")

# Example with user-provided initial guesses (K=3, so 2 tau values, 2 tpar values)
# initial_tau_guess = np.array([0.5, 1.5]) # Must be positive and increasing
# initial_d_prime_guess = 0.8
# result_dod_custom_init = dod(same_counts_data, diff_counts_data, 
#                              initial_tau=initial_tau_guess, 
#                              initial_d_prime=initial_d_prime_guess)
# print("\\n--- DoD Result with Custom Initial Guesses ---")
# print(f"Estimated d-prime: {result_dod_custom_init['d_prime']:.4f}")
# print(f"Estimated tau values: {np.array2string(result_dod_custom_init['tau'], precision=4, suppress_small=True)}")

```

## Notes on Calculation

*   The function uses `scipy.optimize.minimize` with the "L-BFGS-B" method to find the Maximum Likelihood Estimates.
*   It optimizes `d_prime` and `tpar` (increments of `tau`), where `tau_k = sum(tpar_i up to k)`. Constraints are `tpar_i > 0` and `d_prime > 0`.
*   Standard errors are derived from the inverse of the Hessian matrix approximated by the optimizer for the `tpar` and `d_prime` parameters.
```
