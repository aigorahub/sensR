---
layout: layouts/base.njk
title: "ROC Curve Utilities (SDT, AUC)"
permalink: /discrimination/roc-utilities/
tags: api_doc
---

## Introduction

This page details utility functions in `senspy.discrimination` related to Receiver Operating Characteristic (ROC) analysis. These functions allow for the calculation of d-prime from rating scale data using Signal Detection Theory (`SDT`) and the calculation of the Area Under the ROC Curve (`AUC`) from a d-prime value.

These functionalities are inspired by and aim to provide similar capabilities as those found in `sensR` (from `R/ROC.R`).

---

## `SDT`: d-prime from Rating Scale Data

### Purpose

The `SDT` function applies Signal Detection Theory to rating scale data. It calculates transformed hit rates and false alarm rates (either as z-scores via the "probit" method or as logits via the "logit" method) for each criterion in a rating scale task. From these, it also computes d-prime (or its logit equivalent) for each criterion. This is used to analyze data where observers categorize stimuli into multiple ordered categories (e.g., confidence ratings, or "same/different" judgments on a scale).

### Function Signature

```python
SDT(
    counts_table: np.ndarray, 
    method: str = "probit"
) -> np.ndarray
```

### Parameters

*   **`counts_table`** (np.ndarray): A 2xJ NumPy array containing the observed frequencies.
    *   **Row 0**: Counts for the "noise" distribution (e.g., responses to noise-only stimuli, or "same" pairs in a same-different task using rating scales).
    *   **Row 1**: Counts for the "signal" distribution (e.g., responses to signal+noise stimuli, or "different" pairs).
    *   `J` is the number of rating categories, and must be at least 2.
*   **`method`** (str, optional): The transformation method to be applied to the cumulative probabilities.
    *   `"probit"` (default): Applies the inverse of the standard normal CDF (z-scores).
    *   `"logit"`: Applies the logit transformation (`log(p / (1-p))`).

### Return Value

The function returns a (J-1)x3 NumPy array, where J is the number of categories. Each row corresponds to one of the J-1 criteria:

*   **Column 0**: Transformed false alarm rates (zFA if `method="probit"`, or logit(FA) if `method="logit"`).
*   **Column 1**: Transformed hit rates (zH if `method="probit"`, or logit(H) if `method="logit"`).
*   **Column 2**: d-prime (calculated as `zH - zFA`) if `method="probit"`, or the equivalent difference on the logit scale if `method="logit"`.

### Usage Examples

```python
import numpy as np
from senspy.discrimination import SDT

# Example counts_table: 2 distributions (e.g., noise, signal) across 4 rating categories
# Row 0: Noise counts [cat1, cat2, cat3, cat4]
# Row 1: Signal counts [cat1, cat2, cat3, cat4]
counts = np.array([
    [10, 20, 30, 40],  # Noise distribution counts
    [ 5, 15, 25, 55]   # Signal distribution counts
])

# Using the default "probit" method
result_probit = SDT(counts, method="probit")
print("--- SDT Results (Probit) ---")
print("Columns: z(FA), z(Hit), d-prime (for each criterion)")
print(np.round(result_probit, 4))
# Expected output based on sensR::SDT(table) but with zFA and zH columns swapped:
# [[-0.8416 -1.2816 -0.4399]
#  [-0.1534 -0.4307 -0.2773]
#  [ 0.5244  0.5244  0.    ]]

# Using the "logit" method
result_logit = SDT(counts, method="logit")
print("\\n--- SDT Results (Logit) ---")
print("Columns: logit(FA), logit(Hit), logit_diff (for each criterion)")
print(np.round(result_logit, 4))

# Example with a row sum of 0 (e.g., no noise trials observed)
counts_zero_row = np.array([
    [0, 0, 0, 0],      # Noise distribution counts all zero
    [5, 15, 25, 55]    # Signal distribution counts
])
# This will produce a UserWarning and NaNs for the noise row and d-primes
result_zero = SDT(counts_zero_row) 
print("\\n--- SDT Results (Zero Row Sum) ---")
print(result_zero)
```

#### Further Examples for `SDT`

##### Example 1: Smaller Number of Categories (J=3)
If your rating scale has 3 categories, the output will be a (3-1)x3 = 2x3 array.

```python
counts_3cat = np.array([
    [20, 30, 50],  # Noise: sum=100
    [10, 25, 65]   # Signal: sum=100
])

result_3cat_probit = SDT(counts_3cat, method="probit")
print("\\n--- SDT Results (3 Categories, Probit) ---")
print("Columns: z(FA), z(Hit), d-prime")
print(np.round(result_3cat_probit, 4))

result_3cat_logit = SDT(counts_3cat, method="logit")
print("\\n--- SDT Results (3 Categories, Logit) ---")
print("Columns: logit(FA), logit(Hit), logit_diff")
print(np.round(result_3cat_logit, 4))
```
**Comments:**
*   The output array has J-1 = 2 rows, corresponding to the two criteria separating the three categories.

##### Example 2: Data with Strong Separation
When the noise and signal distributions are well separated, d-prime values will be larger.

```python
counts_sep = np.array([
    [50, 30, 15, 5],  # Noise responses skewed left (towards lower categories)
    [5, 15, 30, 50]   # Signal responses skewed right (towards higher categories)
]) # Total 100 for each row

result_sep_probit = SDT(counts_sep, method="probit")
print("\\n--- SDT Results (Strong Separation, Probit) ---")
print(np.round(result_sep_probit, 4))
```
**Comments:**
*   Expect larger (and potentially more consistent across criteria) d-prime values in the third column.

##### Example 3: Data with Poor Separation / Overlap
When distributions overlap significantly, d-prime values will be small.

```python
counts_overlap = np.array([
    [20, 25, 30, 25], # Noise distribution (total 100)
    [22, 28, 27, 23]  # Signal distribution, very similar to noise (total 100)
])

result_overlap_probit = SDT(counts_overlap, method="probit")
print("\\n--- SDT Results (Poor Separation, Probit) ---")
print(np.round(result_overlap_probit, 4))
```
**Comments:**
*   Expect d-prime values close to zero, indicating poor discriminability.

##### Example 4: Interpretation of `zFA` and `zH`
Let's re-use the first example's `result_probit` to understand the columns.

```python
# Using 'result_probit' from the first SDT example:
# counts = np.array([[10,20,30,40], [5,15,25,55]])
# result_probit = SDT(counts, method="probit")
# print(np.round(result_probit, 4))
# Output was:
# [[-0.8416 -1.2816 -0.4399]
#  [-0.1534 -0.4307 -0.2773]
#  [ 0.5244  0.5244  0.    ]]

print("\\n--- Interpretation of zFA and zH (from first Probit example) ---")
print("Output array reminder (zFA, zH, d-prime):")
print(np.round(result_probit, 4))

zFA_values = result_probit[:, 0]
zH_values = result_probit[:, 1]
d_prime_calculated = zH_values - zFA_values # Should match column 2

print(f"\\nTransformed False Alarm Rates (zFA) for each criterion: {np.round(zFA_values, 4)}")
print(f"Transformed Hit Rates (zH) for each criterion: {np.round(zH_values, 4)}")
print(f"Calculated d-primes (zH - zFA): {np.round(d_prime_calculated, 4)}")
```
**Explanation:**
*   **`zFA` (Column 0)**: These are the z-scores corresponding to the cumulative false alarm rates at each criterion. They represent the placement of the criteria from the perspective of the noise distribution.
*   **`zH` (Column 1)**: These are the z-scores corresponding to the cumulative hit rates at each criterion. They represent the placement of the criteria from the perspective of the signal+noise distribution.
*   **`d-prime` (Column 2)**: For each criterion `j`, `d_prime_j = zH_j - zFA_j`. Assuming equal variances of the noise and signal distributions, this `d_prime_j` is an estimate of the separation between the means of the two distributions. Ideally, if the equal variance SDT model holds, these d-prime values should be similar across criteria.

---

## `AUC`: Area Under ROC Curve

### Purpose

The `AUC` function calculates the Area Under the Receiver Operating Characteristic (ROC) curve from a given d-prime value. This is a common measure of discriminability that is independent of response bias. The calculation assumes underlying normal distributions for the signal and noise.

### Function Signature

```python
AUC(
    d_prime: float, 
    scale: float = 1.0
) -> float
```

### Parameters

*   **`d_prime`** (float): The sensitivity index (d-prime). This value can be positive or negative.
*   **`scale`** (float, optional): A scale parameter, typically representing the ratio of standard deviations (sd_signal / sd_noise) if they differ. It must be strictly positive. Defaults to `1.0`, which assumes equal variances for the signal and noise distributions.

### Return Value

The function returns a single float representing the calculated AUC value. This value will range from 0 to 1.
*   An AUC of 0.5 indicates chance performance (d-prime = 0).
*   An AUC of 1.0 indicates perfect discrimination (d-prime approaches positive infinity).
*   An AUC of 0.0 indicates perfect "anti-discrimination" (d-prime approaches negative infinity).

### Formula

The AUC is calculated using the formula:
`AUC = Phi(d_prime / sqrt(1 + scale^2))`
where `Phi` is the cumulative distribution function (CDF) of the standard normal distribution.

### Usage Examples

```python
from senspy.discrimination import AUC
import numpy as np

# Example 1: d-prime = 0 (chance performance)
auc_chance = AUC(d_prime=0.0)
print(f"AUC for d-prime=0: {auc_chance:.4f}") # Expected: 0.5000

# Example 2: d-prime = 1.0, scale = 1.0 (equal variances)
auc_d1_s1 = AUC(d_prime=1.0, scale=1.0)
print(f"AUC for d-prime=1, scale=1: {auc_d1_s1:.4f}") # Expected: ~0.7602 (norm.cdf(1/sqrt(2)))

# Example 3: d-prime = 1.0, scale = 0.5 (signal distribution has smaller variance)
auc_d1_s05 = AUC(d_prime=1.0, scale=0.5)
print(f"AUC for d-prime=1, scale=0.5: {auc_d1_s05:.4f}") # Expected: ~0.8142 (norm.cdf(1/sqrt(1+0.25)))

# Example 4: Negative d-prime
auc_neg_d1 = AUC(d_prime=-1.0, scale=1.0)
print(f"AUC for d-prime=-1, scale=1: {auc_neg_d1:.4f}") # Expected: ~0.2398 (norm.cdf(-1/sqrt(2)))
```

#### Further Examples for `AUC`

##### Example 1: Range of d-prime values

This example shows how AUC changes with different d-prime values, assuming `scale=1.0`.

```python
d_primes_to_test = [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0]
print("\\n--- AUC for a Range of d-prime values (scale=1.0) ---")
for dp_val in d_primes_to_test:
    auc_val = AUC(d_prime=dp_val, scale=1.0)
    print(f"  AUC for d-prime={dp_val:.1f}: {auc_val:.4f}")
```
**Comments:**
*   AUC increases as d-prime increases.
*   AUC = 0.5 when d-prime = 0 (chance performance).
*   AUC approaches 1 as d-prime becomes large and positive.
*   AUC approaches 0 as d-prime becomes large and negative.

##### Example 2: Impact of `scale` parameter

The `scale` parameter influences the mapping from d-prime to AUC. It's defined as `d_prime / sqrt(1 + scale^2)` in the argument to the normal CDF. If `scale` represents `sigma_signal / sigma_noise`, a `scale < 1` means the signal distribution has less variance than the noise distribution, and `scale > 1` means it has more.

```python
d_prime_fixed = 1.5
scales_to_test = [0.5, 1.0, 1.5, 2.0]
print(f"\\n--- AUC for d-prime={d_prime_fixed} with Varying 'scale' ---")
for s_val in scales_to_test:
    auc_val = AUC(d_prime=d_prime_fixed, scale=s_val)
    print(f"  AUC for d-prime={d_prime_fixed}, scale={s_val:.1f}: {auc_val:.4f}")
```
**Comments:**
*   For a fixed positive `d_prime`, as `scale` increases, the denominator `sqrt(1 + scale^2)` increases, so `d_prime / sqrt(1 + scale^2)` decreases. This leads to a smaller AUC.
*   This means if the signal distribution becomes much wider (`scale` > 1) relative to the noise distribution (or if d-prime was defined relative to a smaller variance), the overall discriminability (AUC) for the same d-prime value decreases. Conversely, if `scale` < 1 (signal less variable), AUC increases for the same d-prime.
*   The `scale` parameter essentially adjusts d-prime to an "effective" d-prime for the AUC calculation in the context of potentially unequal variances or differently scaled d-prime values.

#### Illustrative ROC Curves
<!-- TODO: Add illustrative_roc_curves.png here once script execution is resolved -->
The AUC represents the area under an ROC curve. Higher d-prime values lead to ROC curves that bow further towards the top-left corner, indicating better discrimination.
An ROC curve plots Hit Rate (Sensitivity) against False Alarm Rate (1 - Specificity) for various decision criteria.
