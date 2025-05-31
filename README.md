# sensPy

A Python port of the R package `sensR`. This project is in active development, with many core features of `sensR` now available. `sensPy` aims to provide a comprehensive suite for sensory discrimination analysis in Python.

## Features

`sensPy` provides a growing suite of tools for sensory discrimination analysis, including:

**1. Core Psychometric Models:**
*   **`BetaBinomial`**: Fit Beta-Binomial models for overdispersed binomial data (e.g., from panel studies). Includes chance-correction, parameter estimation, summary, and confidence intervals. (Replaces `sensR::betaBin`)
    ```python
    from senspy.models import BetaBinomial
    import numpy as np
    # Example data
    x = np.array([20, 25]); n = np.array([50, 50])
    model = BetaBinomial() # Instantiate before fit
    model.fit(x, n)
    print(model.summary())
    # print(model.confint()) # confint() can be computationally intensive
    ```
*   **General Discrimination (`discrim`)**: Estimate d-prime (and SE, CI, p-value) from various discrimination methods like "2afc", "triangle", "duotrio", "3afc", "tetrad", "hexad", "twofive". (General replacement for `sensR::discrim`)
    ```python
    from senspy.discrimination import discrim
    result = discrim(correct=30, total=50, method="triangle")
    print(f"d-prime (triangle): {result['dprime']:.2f}, p-value: {result['p_value']:.3f}")
    ```
*   **2-Alternative Choice Model (`twoAC`)**: Estimates sensitivity (`delta`) and bias (`tau`) for Yes/No tasks. (Replaces `sensR::twoAC`)
    ```python
    from senspy.discrimination import twoAC
    # hits, false_alarms, n_signal_trials, n_noise_trials
    res_2ac = twoAC(x=[70, 15], n=[100, 100])
    print(f"2AC Delta: {res_2ac['delta']:.2f}, Tau: {res_2ac['tau']:.2f}")
    ```
*   **Degree of Difference (`dod`)**: Thurstonian model for degree of difference data, estimating d-prime and category boundary parameters (`tau`). (Replaces `sensR::dod`)
    ```python
    from senspy.discrimination import dod
    import numpy as np # Ensure numpy is imported for dod example
    same_counts = np.array([10, 20, 70])
    diff_counts = np.array([30, 40, 30])
    res_dod = dod(same_counts, diff_counts)
    print(f"DoD d-prime: {res_dod['d_prime']:.2f}, Tau: {np.round(res_dod['tau'], 2)}")
    ```
*   **Same-Different Model (`samediff`)**: Thurstonian model for same-different tasks, estimating sensitivity (`delta`) and criterion (`tau`). (Replaces `sensR::samediff`)
    ```python
    from senspy.discrimination import samediff
    res_sd = samediff(nsamesame=70, ndiffsame=30, nsamediff=25, ndiffdiff=75)
    print(f"Same-Different Delta: {res_sd['delta']:.2f}, Tau: {res_sd['tau']:.2f}")
    ```

**2. d-prime Hypothesis Testing:**
*   **`dprime_test`**: Test a common d-prime (from one or more groups) against a specified value. (Replaces `sensR::dprime_test`)
    ```python
    from senspy.discrimination import dprime_test
    # Test if d-prime from a 2AFC task (35 correct / 50 trials) is greater than 0.5
    res_test = dprime_test(correct=35, total=50, protocol="2afc", dprime0=0.5, alternative="greater")
    print(f"dprime_test P-value: {res_test['p_value']:.3f}")
    ```
*   **`dprime_compare`**: Compare d-primes from multiple groups for equality using a Likelihood Ratio Test. (Replaces `sensR::dprime_compare`)
    ```python
    from senspy.discrimination import dprime_compare
    # Compare two groups
    res_comp = dprime_compare(correct=[30, 40], total=[50, 50], protocol=['2afc', '2afc'])
    print(f"dprime_compare P-value: {res_comp['p_value']:.3f}")
    ```

**3. ROC Utilities:**
*   **`SDT`**: Calculate d-prime and criteria from rating scale data (Signal Detection Theory). (Replaces `sensR::SDT` from `ROC.R`)
    ```python
    from senspy.discrimination import SDT
    import numpy as np # Ensure numpy is imported for SDT example
    counts = np.array([[50,30,15,5], [5,15,30,50]]) # Noise, Signal
    sdt_results = SDT(counts, method="probit")
    # print(sdt_results) # Prints (J-1)x3 array of zFA, zH, d_prime_j
    print(f"SDT d-primes per criterion: {np.round(sdt_results[:, 2], 2)}")
    ```
*   **`AUC`**: Calculate Area Under the ROC Curve from d-prime. (Replaces `sensR::AUC` from `ROC.R`)
    ```python
    from senspy.discrimination import AUC
    auc_val = AUC(d_prime=1.5)
    print(f"AUC for d'=1.5: {auc_val:.3f}")
    ```

**4. Psychometric Link Functions (`senspy.links`):**
*   **`psyfun`**: Convert d-prime to proportion correct for various methods.
*   **`psyinv`**: Convert proportion correct to d-prime for various methods.
*   **`psyderiv`**: Calculate the derivative of the psychometric function.
*   **`rescale`**: Convert values (and standard errors) between d-prime, Pc, and Pd scales.
    ```python
    from senspy.links import psyfun, rescale
    import numpy as np # Ensure numpy is imported for link examples
    pc_triangle = psyfun(dprime=1.0, method="triangle")
    rescaled_vals = rescale(x=pc_triangle, from_scale="pc", to_scale="dp", method="triangle")
    print(f"Pc for d'=1 (triangle): {pc_triangle:.3f}, rescaled d': {np.round(rescaled_vals['coefficients']['d_prime'], 3)}")
    ```

**5. Power and Sample Size (`senspy.power`):**
*   **`power_discrim`**: Calculate statistical power for detecting a given d-prime in various discrimination tasks.
*   **`sample_size_for_binomial_power`**: Estimate sample size needed to achieve a target power for a binomial test.
*   **`exact_binomial_power`** and **`find_critical_binomial_value`**: Core binomial test utilities.
    ```python
    from senspy.power import power_discrim
    # Power to detect d'=0.8 in a 2AFC task with 50 trials (alpha=0.05, vs d'=0)
    pwr = power_discrim(d_prime_alt=0.8, n_trials=50, method="2afc")
    print(f"Power for d'=0.8 (2AFC, N=50): {pwr:.3f}")
    ```

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management.
To install the development version:

```bash
poetry install
```

## Contributing

Contributions are welcome! Please submit pull requests via GitHub. See
`CONTRIBUTING.md` for guidelines.

## License

This project is licensed under the terms of the GNU General Public License
version 2.0 or later. See [LICENSE](LICENSE) for details.
```
