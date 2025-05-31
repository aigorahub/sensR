# Guide to Switching from R (sensR) to Python (sensPy)

This guide is intended to help users familiar with the R package `sensR` transition to using its Python port, `sensPy`.

## Introduction

`sensPy` aims to replicate the core functionality of `sensR` while leveraging the Python scientific computing ecosystem. While many concepts are similar, there are differences in API, data handling, and typical workflows.

## Installation

**sensR (R):**
Typically installed from CRAN or GitHub:
```R
# install.packages("sensR") # From CRAN
# devtools::install_github("aigorahub/sensR") # From GitHub
library(sensR)
```

**sensPy (Python):**
Currently installed from GitHub. Ensure you have Python (>=3.8) and pip installed.
```bash
pip install git+https://github.com/aigorahub/sensPy.git
```
In your Python script or notebook:
```python
import senspy.models as spm
import senspy.discrimination as spd
import senspy.power as spp
import numpy as np
import pandas as pd # For data handling, if preferred
```

## Key Differences at a Glance

| Feature          | sensR (R)                                   | sensPy (Python)                                                                 |
|------------------|---------------------------------------------|---------------------------------------------------------------------------------|
| **Primary API**  | Functional (e.g., `discrim()`, `betaBin()`) | Object-Oriented (`Model.fit()`, `Model.summary()`, `Model.confint()`) and functional wrappers |
| **Data Input**   | Vectors, data.frames                        | NumPy arrays, lists. Pandas DataFrames can be used for data preparation.        |
| **Output**       | List objects, S3/S4 class objects           | Dictionaries (from wrapper functions), Model instances with attributes/methods.   |
| **Plotting**     | Base R plots, ggplot2                       | Matplotlib, Plotly (plotting features in `sensPy` are under development)        |

## Functionality Mapping

This section details the mapping of common `sensR` functions to their `sensPy` equivalents. Refer to the `functionality_inventory.csv` in the repository for a more exhaustive list.

### 1. Core Discrimination (`discrim`)

**sensR:**
```R
# result_2afc <- discrim(correct = 75, total = 100, method = "2afc")
# print(result_2afc$dprime)
```

**sensPy:**
`sensPy` offers two ways: a direct functional equivalent and an object-oriented approach.

*Functional Wrapper (returns a dictionary):*
```python
results_dict = spd.discrim(correct=75, total=100, method="2afc")
print(results_dict["dprime"])
```

*Object-Oriented Model:*
```python
model = spm.DiscriminationModel()
model.fit(correct=75, total=100, method="2afc")
print(model.dprime)
print(model.summary())
ci_profile = model.confint(method_ci='profile') # Or 'wald'
print(f"D-prime: {model.dprime:.4f}")
print(f"Profile CI: {ci_profile['dprime']}")
```
Supported methods for `discrim` (and `DiscriminationModel`) include: `"2afc"`, `"triangle"`, `"duotrio"`, `"3afc"`, `"tetrad"`, `"hexad"`, `"twofive"`.

### 2. Beta-Binomial Model (`betaBin`)

**sensR:**
```R
# x_r <- c(20, 25)
# n_r <- c(50, 50)
# bb_model_r <- betaBin(x_r, n_r, corrected = FALSE)
# print(summary(bb_model_r))
# print(confint(bb_model_r))
```

**sensPy:**
```python
x_py = np.array([20, 25])
n_py = np.array([50, 50])

bb_model_py = spm.BetaBinomial()
bb_model_py.fit(x_py, n_py, corrected=False)

print(bb_model_py.summary())
# Profile CIs are default
conf_intervals = bb_model_py.confint(parm=['alpha', 'beta'], level=0.95)
print(conf_intervals)

# Access parameters
print(f"Alpha: {bb_model_py.alpha}, Beta: {bb_model_py.beta}")
print(f"LogLik: {bb_model_py.loglik}")
```
The `corrected` and `p_guess` parameters are available in `sensPy`'s `fit` method.

### 3. Two-Alternative Choice (`twoAC`)

**sensR:**
```R
# twoac_res_r <- twoAC(x = c(70, 15), N = c(100, 100)) # x = c(hits, false_alarms)
# print(twoac_res_r)
```

**sensPy:**
*Functional Wrapper:*
```python
twoac_dict_py = spd.twoAC(x=[70, 15], n=[100, 100])
print(f"Delta (functional): {twoac_dict_py['delta']:.4f}, Tau: {twoac_dict_py['tau']:.4f}")
```

*Object-Oriented Model:*
```python
twoac_model_py = spm.TwoACModel()
twoac_model_py.fit(hits=70, false_alarms=15, n_signal_trials=100, n_noise_trials=100)

print(twoac_model_py.summary())
ci_profile_twoac = twoac_model_py.confint(method_ci='profile') # Or 'wald'
print(f"Profile CIs: {ci_profile_twoac}")
print(f"Delta (model): {twoac_model_py.delta:.4f}, Tau (model): {twoac_model_py.tau:.4f}")
```

### 4. Degree of Difference (`dod`)

**sensR:**
```R
# same_counts_r <- c(10, 20, 70)
# diff_counts_r <- c(70, 20, 10)
# dod_res_r <- dod(same_counts_r, diff_counts_r)
# print(dod_res_r)
```

**sensPy:**
*Functional Wrapper:*
```python
same_counts_py = np.array([10, 20, 70])
diff_counts_py = np.array([70, 20, 10])
dod_dict_py = spd.dod(same_counts_py, diff_counts_py)
print(f"d-prime (functional): {dod_dict_py['d_prime']:.4f}")
print(f"Tau (functional): {dod_dict_py['tau']}")
```

*Object-Oriented Model:*
```python
dod_model_py = spm.DoDModel()
dod_model_py.fit(same_counts_py, diff_counts_py)

print(dod_model_py.summary())
# confint for DoDModel provides Wald CIs for d_prime and tpar by default
# Profile CIs are available via method_ci='profile'
ci_dod_profile = dod_model_py.confint(parm=['d_prime', 'tpar_0', 'tpar_1'], method_ci='profile')
print(f"Profile CIs for d_prime: {ci_dod_profile.get('d_prime')}")
print(f"Optimized d-prime: {dod_model_py.d_prime}, Tau values: {dod_model_py.tau}")
```

### 5. Same-Different (`samediff`)

**sensR:**
```R
# sd_res_r <- samediff(samesame = 70, diffsame = 30, samediff = 25, diffdiff = 75)
# print(sd_res_r)
```

**sensPy:**
*Functional Wrapper:*
```python
sd_dict_py = spd.samediff(nsamesame=70, ndiffsame=30, nsamediff=25, ndiffdiff=75)
print(f"Delta (functional): {sd_dict_py['delta']:.4f}, Tau: {sd_dict_py['tau']:.4f}")
```

*Object-Oriented Model:*
```python
sd_model_py = spm.SameDifferentModel()
sd_model_py.fit(nsamesame=70, ndiffsame=30, nsamediff=25, ndiffdiff=75)

print(sd_model_py.summary())
# confint for SameDifferentModel provides Wald CIs by default
# Profile CIs are available via method_ci='profile'
ci_sd_profile = sd_model_py.confint(parm=['delta', 'tau'], method_ci='profile')
print(f"Profile CIs: {ci_sd_profile}")
print(f"Delta (model): {sd_model_py.delta:.4f}, Tau (model): {sd_model_py.tau:.4f}")
```

### 6. Psychometric Link Functions (`psyfun`, `psyinv`, `psyderiv`, `rescale`)

These functions are available in `senspy.discrimination` (as they were moved from the original `senspy.links`).

**sensR:** (Typically `psyfun()`, `psyinv()`, etc. are called directly)
```R
# pc <- psyfun(dprime = 1.0, method = "triangle")
# d_prime_val <- psyinv(pc = 0.6, method = "triangle")
```

**sensPy:**
```python
pc = spd.psyfun(dprime=1.0, method="triangle")
print(f"Pc for d'=1 (triangle): {pc:.4f}")

d_prime_val = spd.psyinv(pc=0.6, method="triangle")
print(f"d' for Pc=0.6 (triangle): {d_prime_val:.4f}")

deriv = spd.psyderiv(dprime=1.0, method="triangle")
print(f"Derivative at d'=1 (triangle): {deriv:.4f}")

rescaled_vals = spd.rescale(x=pc, from_scale="pc", to_scale="dp", method="triangle", std_err=0.05)
print(f"Rescaled values: {rescaled_vals}")
```

### 7. ROC Utilities (`SDT`, `AUC`)

**sensR:** (From `sensR/R/ROC.R`)
```R
# counts_r <- matrix(c(50,30,15,5, 5,15,30,50), nrow=2, byrow=TRUE)
# sdt_res_r <- SDT(counts_r)
# auc_val_r <- AUC(d.prime = 1.5)
```

**sensPy:**
```python
counts_py = np.array([[50,30,15,5], [5,15,30,50]]) # Noise, Signal
sdt_results_py = spd.SDT(counts_py, method="probit")
print("SDT Results (zFA, zH, d-prime per criterion):")
print(sdt_results_py)

auc_val_py = spd.AUC(d_prime=1.5)
print(f"AUC for d'=1.5: {auc_val_py:.4f}")
```

### 8. Power and Sample Size

`sensR` has functions like `discrimPwr`, `discrimSS`. `sensPy` provides equivalents in `senspy.power`.

**sensR Example (Conceptual):**
```R
# Power for d'=0.8, 50 trials, 2AFC
# pwr_r <- discrimPwr(d.prime = 0.8, n = 50, method = "2afc")
```

**sensPy Example:**
```python
# Power (exact binomial method)
power_result_exact = spp.power_discrim(
    d_prime_alt=0.8, n_trials=50, method="2afc",
    d_prime_null=0.0, alpha_level=0.05, alternative="greater"
)
print(f"Exact Power: {power_result_exact.power:.4f} (Details: {power_result_exact.details})")

# Power (normal approximation)
power_result_approx = spp.power_discrim_normal_approx(
    d_prime_alt=0.8, n_trials=50, method="2afc",
    d_prime_null=0.0, alpha_level=0.05, alternative="greater"
)
if power_result_approx: # Check if it's not None
    print(f"Approx Power: {power_result_approx.power:.4f} (Details: {power_result_approx.details})")

# Sample Size (exact binomial method)
n_exact = spp.sample_size_discrim(
    d_prime_alt=0.8, target_power=0.8, method="2afc",
    d_prime_null=0.0, alpha_level=0.05, alternative="greater"
)
print(f"Required N (Exact): {n_exact}")

# Sample Size (normal approximation)
n_approx = spp.power_discrim_normal_approx(
    d_prime_alt=0.8, power_target=0.8, method="2afc",
    d_prime_null=0.0, alpha_level=0.05, alternative="greater"
)
print(f"Required N (Approx): {n_approx}")

# The PowerResult object provides structured output:
# power_result_exact.power
# power_result_exact.n_trials
# power_result_exact.alpha_level
# power_result_exact.method_type
# power_result_exact.details (dictionary with more context)
```

### 9. d-prime Hypothesis Testing (`dprime_test`, `dprime_compare`)

These are available in `senspy.discrimination`.

**sensR:**
```R
# dprime_test(correct=35, total=50, protocol="2afc", dprime0=0.5, alternative="greater")
# dprime_compare(correct=c(30,40), total=c(50,50), protocol=c("2afc", "2afc"))
```

**sensPy:**
```python
test_res = spd.dprime_test(correct=35, total=50, protocol='2afc', dprime0=0.5, alternative="greater")
print(f"dprime_test p-value: {test_res['p_value']:.4f}")

compare_res = spd.dprime_compare(correct=[30,40], total=[50,50], protocol=['2afc','2afc'])
print(f"dprime_compare p-value: {compare_res['p_value']:.4f}")
```

## Data Handling

*   **Vectors/Arrays**: R vectors often map to 1D NumPy arrays.
*   **Data Frames**: R `data.frame` objects are similar to Pandas `DataFrame`. If your data is in a CSV, you can load it with Pandas:
    ```python
    # Assuming data.csv has columns 'correct', 'total', 'subject_id'
    # df = pd.read_csv("data.csv")
    # correct_counts = df['correct'].values # NumPy array
    # total_trials = df['total'].values   # NumPy array
    ```

## Conclusion

Transitioning from `sensR` to `sensPy` involves adopting Python's syntax and object-oriented patterns for some functionalities. However, `sensPy` provides functional wrappers for many common tasks to ease this transition. The core statistical methodologies are preserved.

We encourage you to explore the examples in the `sensPy` documentation and notebooks. If you find discrepancies or have suggestions, please open an issue on the [sensPy GitHub repository](https://github.com/aigorahub/sensPy).
