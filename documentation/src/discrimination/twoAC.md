---
layout: layouts/base.njk
title: "twoAC: 2-Alternative Choice Model"
permalink: /discrimination/twoAC/
tags: api_doc
---

## Introduction

The `twoAC` function estimates parameters for the Thurstonian model of a 2-Alternative Choice (2AC) task, often referred to as a Yes/No or Same-Different task with response bias. It uses Maximum Likelihood estimation to determine the sensitivity (`delta`, a d-prime like measure) and the response bias (`tau`, representing the criterion placement).

This function replaces `sensR::twoAC()` (from `R/twoAC.R`) in the R package `sensR`.

## Function Signature

```python
twoAC(
    x: list[int], 
    n: list[int], 
    method: str = "ml"
) -> dict
```

## Parameters

*   **`x`** (list[int] | np.ndarray): A list or 2-element array `[hits, false_alarms]`.
    *   `hits`: Number of times the signal was correctly identified ("yes" response to signal present).
    *   `false_alarms`: Number of times a signal was reported when none was present ("yes" response to noise only).
*   **`n`** (list[int] | np.ndarray): A list or 2-element array `[n_signal_trials, n_noise_trials]`.
    *   `n_signal_trials`: Total number of trials where a signal was present.
    *   `n_noise_trials`: Total number of trials where only noise was present.
*   **`method`** (str, optional): The estimation method to be used. Currently, only `"ml"` (Maximum Likelihood) is implemented and supported. Defaults to `"ml"`.

## Return Value

The function returns a dictionary containing the following key-value pairs:

*   **`"delta"`** (float): The estimated sensitivity parameter, analogous to d-prime. It represents the difference between the means of the signal and noise distributions in the Thurstonian model.
*   **`"tau"`** (float): The estimated response bias or criterion placement. A value of 0 typically indicates no bias. Positive values might indicate a bias towards responding "no" (or "different"), and negative values a bias towards "yes" (or "same"), depending on the specific model interpretation.
*   **`"se_delta"`** (float): The standard error of the estimated `delta`.
*   **`"se_tau"`** (float): The standard error of the estimated `tau`.
*   **`"loglik"`** (float): The maximized log-likelihood value of the fitted model.
*   **`"vcov"`** (np.ndarray): A 2x2 variance-covariance matrix for `delta` and `tau`. Derived from the Hessian of the log-likelihood function.
*   **`"convergence_status"`** (bool): `True` if the optimization algorithm successfully converged, `False` otherwise.
*   **`"hits"`** (int): The number of hits provided as input.
*   **`"false_alarms"`** (int): The number of false alarms provided as input.
*   **`"n_signal_trials"`** (int): The total number of signal trials provided.
*   **`"n_noise_trials"`** (int): The total number of noise trials provided.
*   **`"initial_params"`** (np.ndarray): The initial parameters `[tau_init, delta_init]` used for optimization.
*   **`"optim_result"`** (scipy.optimize.OptimizeResult): The full result object from `scipy.optimize.minimize` for detailed inspection.

## Model Details

The underlying Thurstonian model for the 2AC (Yes/No) task assumes that the observer's internal response to a "noise" trial is drawn from a normal distribution N(0,1), and to a "signal" trial from N(d',1) (where d' here corresponds to `delta`). The observer responds "yes" (signal present) if their internal response exceeds a criterion, `c`.

The parameters estimated are:
*   **`delta`**: The difference between the means of the signal and noise distributions (d').
*   **`tau`**: The criterion `c` in the model. (Note: some SDT formulations use `c` and `d'`, where `tau` might be related to `c/d'` or other forms. Here, `tau` is directly the criterion `c`).

The probabilities of Hits and False Alarms are given by:
*   `P(Hit) = P("yes" | signal) = Phi(delta / 2 - tau)`
*   `P(False Alarm) = P("yes" | noise) = Phi(-delta / 2 - tau)`

Where `Phi` is the cumulative distribution function (CDF) of the standard normal distribution. The parameters `delta` and `tau` are estimated by maximizing the binomial likelihood of observing the given number of hits and false alarms.

## Usage Examples

```python
import numpy as np
from senspy.discrimination import twoAC

# Example Data
hits = 70
false_alarms = 15
n_signal_trials = 100
n_noise_trials = 100

# Perform the 2AC analysis
result_2ac = twoAC(x=[hits, false_alarms], n=[n_signal_trials, n_noise_trials])

# Print the full result dictionary
print("--- Full 2AC Result ---")
for key, value in result_2ac.items():
    if isinstance(value, np.ndarray):
        # Print arrays in a more readable format if desired
        print(f"  {key}:")
        print(value)
    else:
        print(f"  {key}: {value}")

# Access specific values
print("\\n--- Key Estimates ---")
if result_2ac["convergence_status"]:
    print(f"Estimated delta (sensitivity): {result_2ac['delta']:.4f} (SE: {result_2ac['se_delta']:.4f})")
    print(f"Estimated tau (criterion): {result_2ac['tau']:.4f} (SE: {result_2ac['se_tau']:.4f})")
    print(f"Log-Likelihood: {result_2ac['loglik']:.4f}")
else:
    print("Optimization did not converge. Estimates may be unreliable.")

# Example with different data
hits_2 = 20
false_alarms_2 = 5
n_signal_2 = 30
n_noise_2 = 30

result_2ac_2 = twoAC(x=[hits_2, false_alarms_2], n=[n_signal_2, n_noise_2])
print("\\n--- Second Example ---")
print(f"Estimated delta: {result_2ac_2['delta']:.4f}")
print(f"Estimated tau: {result_2ac_2['tau']:.4f}")

```

## Notes on Calculation

*   The function uses `scipy.optimize.minimize` with the "L-BFGS-B" method to find the Maximum Likelihood Estimates of `delta` and `tau`.
*   `delta` is constrained to be non-negative (`>=0`). `tau` is unconstrained.
*   Standard errors are derived from the inverse of the Hessian matrix approximated by the optimizer.
*   Probabilities are clipped internally (e.g., to `[1e-12, 1.0 - 1e-12]`) during log-likelihood calculation to prevent `log(0)` errors if observed hit/false alarm rates are 0 or 1.
```
