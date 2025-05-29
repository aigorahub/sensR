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
