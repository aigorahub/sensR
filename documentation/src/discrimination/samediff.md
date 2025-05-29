---
layout: layouts/base.njk
title: "samediff: Same-Different Model"
permalink: /discrimination/samediff/
tags: api_doc
---

## Introduction

The `samediff` function fits the Thurstonian model for the Same-Different task. In this task, observers are presented with pairs of stimuli and judge whether they are the "same" or "different". The model estimates two key parameters using Maximum Likelihood:

*   **`delta`**: The sensitivity (d-prime) to the difference between stimuli when they are indeed different.
*   **`tau`**: A parameter related to the decision criterion or response bias. Specifically, it's linked to the criterion placement on the difference axis.

This function replaces `sensR::samediff()` (from `R/samediff.R`) in the R package `sensR`.

## Function Signature

```python
samediff(
    nsamesame: int, 
    ndiffsame: int, 
    nsamediff: int, 
    ndiffdiff: int,
    initial_tau: float | None = None, 
    initial_delta: float | None = None, 
    method: str = "ml", 
    conf_level: float = 0.95
) -> dict
```

## Parameters

*   **`nsamesame`** (int): The number of trials where the stimuli were "same" and the observer responded "same".
*   **`ndiffsame`** (int): The number of trials where the stimuli were "same" but the observer responded "different".
*   **`nsamediff`** (int): The number of trials where the stimuli were "different" but the observer responded "same".
*   **`ndiffdiff`** (int): The number of trials where the stimuli were "different" and the observer responded "different".
*   **`initial_tau`** (float | None, optional): Initial guess for the `tau` parameter. If `None` or not positive, defaults to `1.0`.
*   **`initial_delta`** (float | None, optional): Initial guess for the `delta` (d-prime) parameter. If `None` or negative, defaults to `1.0` (though `delta` will be estimated as non-negative).
*   **`method`** (str, optional): The estimation method. Currently, only `"ml"` (Maximum Likelihood) is implemented and supported. Defaults to `"ml"`.
*   **`conf_level`** (float, optional): The desired confidence level for confidence intervals. This parameter is retained for future use (e.g., if CIs are added to the return) but is not directly used in the current fitting process for SE calculation. Defaults to `0.95`.

## Model Details (Briefly)

The model assumes that the perceived difference for "same" pairs follows N(0,1) and for "different" pairs follows N(delta,1). A single criterion `tau` is used on this difference axis. The probabilities for the four response types are:

*   **`Pss`** (Prob "same" response | "same" pair): `2 * Phi(tau / sqrt(2)) - 1`
*   **`Pds`** (Prob "different" response | "same" pair): `1 - Pss`
*   **`Psd`** (Prob "same" response | "different" pair): `Phi((tau - delta) / sqrt(2)) - Phi((-tau - delta) / sqrt(2))`
*   **`Pdd`** (Prob "different" response | "different" pair): `1 - Psd`

Where `Phi` is the cumulative distribution function (CDF) of the standard normal distribution. The internal helper `_get_samediff_probs` calculates these, including clipping for numerical stability. The parameters `delta` and `tau` are estimated by maximizing the multinomial log-likelihood based on the observed counts.

## Return Value

The function returns a dictionary containing:

*   **`"tau"`** (float): The estimated criterion parameter `tau`. Must be positive.
*   **`"delta"`** (float): The estimated sensitivity parameter `delta` (d-prime). Must be non-negative.
*   **`"se_tau"`** (float): The standard error of the estimated `tau`.
*   **`"se_delta"`** (float): The standard error of the estimated `delta`.
*   **`"loglik"`** (float): The maximized log-likelihood value of the fitted model.
*   **`"vcov"`** (np.ndarray): A 2x2 variance-covariance matrix for `tau` and `delta`.
*   **`"convergence_status"`** (bool): `True` if the optimization algorithm successfully converged, `False` otherwise.
*   **`"initial_params"`** (np.ndarray): The initial parameters `[tau_init, delta_init]` used to start the optimization.
*   **`"optim_result"`** (scipy.optimize.OptimizeResult): The full result object from `scipy.optimize.minimize` for detailed inspection.
*   **`"nsamesame"`** (int): Input count.
*   **`"ndiffsame"`** (int): Input count.
*   **`"nsamediff"`** (int): Input count.
*   **`"ndiffdiff"`** (int): Input count.
*   **`"method"`** (str): The estimation method used (e.g., "ml").
*   **`"conf_level"`** (float): The confidence level specified (retained for context).

## Usage Examples

```python
import numpy as np
from senspy.discrimination import samediff

# Example Data
nsamesame_val = 70
ndiffsame_val = 30
nsamediff_val = 25
ndiffdiff_val = 75

# Fit the Same-Different model
result_sd = samediff(
    nsamesame=nsamesame_val, 
    ndiffsame=ndiffsame_val, 
    nsamediff=nsamediff_val, 
    ndiffdiff=ndiffdiff_val
)

# Print the full result dictionary
print("--- Full Same-Different Result ---")
for key, value in result_sd.items():
    if isinstance(value, np.ndarray):
        print(f"  {key}: {np.array2string(value, precision=4, suppress_small=True)}")
    elif isinstance(value, float):
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value}")

# Access specific key estimates
print("\\n--- Key Estimates ---")
if result_sd["convergence_status"]:
    print(f"Estimated delta (sensitivity): {result_sd['delta']:.4f} (SE: {result_sd['se_delta']:.4f})")
    print(f"Estimated tau (criterion): {result_sd['tau']:.4f} (SE: {result_sd['se_tau']:.4f})")
    print(f"Log-Likelihood: {result_sd['loglik']:.4f}")
else:
    print("Same-Different model optimization did not converge. Estimates may be unreliable.")

# Example with initial guesses
# result_sd_init = samediff(
#     nsamesame_val, ndiffsame_val, nsamediff_val, ndiffdiff_val,
#     initial_tau=0.8, initial_delta=1.2
# )
# print("\\n--- Result with Initial Guesses ---")
# print(f"Estimated delta: {result_sd_init['delta']:.4f}")
# print(f"Estimated tau: {result_sd_init['tau']:.4f}")

```

## Notes on Calculation

*   The function uses `scipy.optimize.minimize` with the "L-BFGS-B" method to find the Maximum Likelihood Estimates of `tau` and `delta`.
*   Constraints: `tau > 0` (strictly positive) and `delta >= 0` (non-negative).
*   Standard errors are derived from the inverse of the Hessian matrix approximated by the optimizer.
*   Input counts are validated: must be non-negative integers, sum of counts must be positive, and at least two of the four counts must be non-zero.
```
