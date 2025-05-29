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

### Further Examples

#### Example 1: Data Suggesting High vs. Low Sensitivity (`delta`)

Sensitivity (`delta`) reflects how well the observer can distinguish between "same" and "different" pairs.

```python
# High delta: Strong distinction between "same" and "different" responses
# High P(Same|Same) and P(Diff|Diff)
result_high_delta = samediff(nsamesame=80, ndiffsame=20,   # 80% "same" for same pairs
                             nsamediff=15, ndiffdiff=85)   # 85% "different" for diff pairs
print("\\n--- High Sensitivity Example ---")
if result_high_delta["convergence_status"]:
    print(f"  Delta: {result_high_delta['delta']:.4f}, Tau: {result_high_delta['tau']:.4f}")
else:
    print("  High sensitivity model did not converge.")

# Low delta: Poor distinction
# P(Same|Same) and P(Diff|Diff) are closer to 50%, or responses are mixed
result_low_delta = samediff(nsamesame=55, ndiffsame=45,    # 55% "same" for same pairs
                            nsamediff=40, ndiffdiff=60)    # 60% "different" for diff pairs
print("\\n--- Low Sensitivity Example ---")
if result_low_delta["convergence_status"]:
    print(f"  Delta: {result_low_delta['delta']:.4f}, Tau: {result_low_delta['tau']:.4f}")
else:
    print("  Low sensitivity model did not converge.")
```
**Comments:**
*   For the "High Sensitivity" data, `delta` should be relatively large. `tau`'s value will depend on any response bias.
*   For the "Low Sensitivity" data, `delta` should be small (closer to 0).

#### Example 2: Data Suggesting Different Biases (`tau`)

Response bias (`tau`) reflects the observer's tendency to respond "same" or "different" overall, regardless of the actual stimulus pair.

```python
# Bias towards "same": Observer tends to say "same" more often
# P(Same|Same) is high, P(Same|Diff) is also relatively high
result_bias_same = samediff(nsamesame=70, ndiffsame=30,  # High "same" for same
                            nsamediff=60, ndiffdiff=40) # Also high "same" for different
print("\\n--- Bias Towards 'Same' Example ---")
if result_bias_same["convergence_status"]:
    print(f"  Delta: {result_bias_same['delta']:.4f}, Tau: {result_bias_same['tau']:.4f}") # Expect tau < default
else:
    print("  Bias 'same' model did not converge.")


# Bias towards "different": Observer tends to say "different" more often
# P(Diff|Same) is high, P(Diff|Diff) is also very high
result_bias_diff = samediff(nsamesame=40, ndiffsame=60,  # High "different" for same
                            nsamediff=30, ndiffdiff=70) # High "different" for different
print("\\n--- Bias Towards 'Different' Example ---")
if result_bias_diff["convergence_status"]:
    print(f"  Delta: {result_bias_diff['delta']:.4f}, Tau: {result_bias_diff['tau']:.4f}") # Expect tau > default
else:
    print("  Bias 'different' model did not converge.")
```
**Comments:**
*   `tau` is related to the criterion `c`. A smaller `tau` might indicate a bias towards saying "same" (criterion shifted to pick up more "same" responses, even for different pairs). A larger `tau` might indicate a bias towards "different". The exact interpretation can depend on the model parameterization details. In this model, `tau` is directly the criterion `c`. A liberal criterion to say "same" (many "same" responses) would mean `tau` is larger. A strict criterion to say "same" (few "same" responses) would mean `tau` is smaller.
*   The prompt's interpretation of `tau` (positive for "no"/different, negative for "yes"/same) is for the `twoAC` task where `tau` is `c`. Here, `tau` is also `c` but on the difference axis. `Pss` increases with `tau`. `Psd` decreases with `tau` (for fixed delta).
    *   Bias towards "same": High `Pss` and high `Psd`. This implies `tau` is relatively large.
    *   Bias towards "different": Low `Pss` and low `Psd`. This implies `tau` is relatively small.
    *   The example data for "Bias towards 'same'" (`nsamesame=70, nsamediff=60`) should yield a larger `tau`.
    *   The example data for "Bias towards 'different'" (`nsamesame=40, nsamediff=30`) should yield a smaller `tau`.

#### Example 3: Impact of an Edge Case (e.g., one count is zero)

If one of the response counts is zero, it can affect parameter estimates and their standard errors.

```python
# Edge Case: No "same" responses to "different" items (nsamediff = 0)
# This suggests very good performance if ndiffdiff is high.
result_edge = samediff(nsamesame=80, ndiffsame=20, 
                       nsamediff=0, ndiffdiff=100) 
print("\\n--- Edge Case (nsamediff = 0) ---")
if result_edge["convergence_status"]:
    print(f"  Delta: {result_edge['delta']:.4f} (SE: {result_edge['se_delta']:.4f})")
    print(f"  Tau:   {result_edge['tau']:.4f} (SE: {result_edge['se_tau']:.4f})")
else:
    print("  Edge case model did not converge.")
```
**Comments:**
*   When `nsamediff = 0`, it implies that if stimuli were different, the observer never mistakenly called them "same". This could lead to a very high `delta` estimate, potentially pushing it towards infinity.
*   Standard errors might become very large or `np.nan` if the estimate is at the boundary of the parameter space or if the likelihood surface is flat in that region.
*   The function includes input validation that requires at least two of the four counts to be non-zero. If, for example, `nsamesame`, `ndiffsame`, and `nsamediff` were all zero, the function would raise an error.

#### Example 4: User-Provided Initial Guesses

Providing initial guesses for `tau` and `delta` can be useful if the default starting points (tau=1.0, delta=1.0) lead to slow convergence or convergence to a non-optimal local maximum, though "L-BFGS-B" is generally robust.

```python
# Using data from the first main example
result_custom_init_sd = samediff(nsamesame=70, ndiffsame=30, 
                                 nsamediff=25, ndiffdiff=75, 
                                 initial_tau=0.5, initial_delta=1.5)
print("\\n--- Using Initial Guesses (70,30,25,75) ---")
if result_custom_init_sd["convergence_status"]:
    print(f"  Delta: {result_custom_init_sd['delta']:.4f}, Tau: {result_custom_init_sd['tau']:.4f}")
    print(f"  Initial params used by optimizer: {result_custom_init_sd['initial_params']}")
else:
    print("  Optimization did not converge with custom initial guesses.")

```
**Comments:**
*   The `initial_params` field in the returned dictionary shows the actual starting values used by the optimizer after applying defaults or validating user inputs.

## Notes on Calculation

*   The function uses `scipy.optimize.minimize` with the "L-BFGS-B" method to find the Maximum Likelihood Estimates of `tau` and `delta`.
*   Constraints: `tau > 0` (strictly positive) and `delta >= 0` (non-negative).
*   Standard errors are derived from the inverse of the Hessian matrix approximated by the optimizer.
*   Input counts are validated: must be non-negative integers, sum of counts must be positive, and at least two of the four counts must be non-zero.
```
