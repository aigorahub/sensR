# sensPy

A Python port of the R package `sensR` for Thurstonian models in sensory discrimination analysis.

[![CI](https://github.com/aigorahub/sensPy/actions/workflows/pytest.yml/badge.svg)](https://github.com/aigorahub/sensPy/actions/workflows/pytest.yml)
[![codecov](https://codecov.io/gh/aigorahub/sensPy/graph/badge.svg)](https://codecov.io/gh/aigorahub/sensPy)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v2](https://img.shields.io/badge/License-GPL%20v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)

---

## About

sensPy provides a comprehensive Python implementation of [sensR](https://cran.r-project.org/package=sensR), the R package for sensory discrimination methods. This project ensures:

- **Numerical parity** with sensR (validated against R outputs)
- **Modern Python practices** (type hints, dataclasses, NumPy/SciPy integration)
- **Full test coverage** with 500+ tests

## Installation

```bash
# Install with pip
pip install senspy

# Or install from source
git clone https://github.com/aigorahub/sensPy.git
cd sensPy
pip install -e .
```

## Features

### Discrimination Protocols

All standard protocols with single and double variants:

| Protocol | Single | Double |
|----------|--------|--------|
| Triangle | ✅ | ✅ |
| Duo-Trio | ✅ | ✅ |
| 2-AFC | ✅ | ✅ |
| 3-AFC | ✅ | ✅ |
| Tetrad | ✅ | ✅ |
| Hexad | ✅ | - |
| 2-out-of-5 | ✅ | - |

### Statistical Models

- **`discrim()`** - D-prime estimation with confidence intervals and p-values
- **`betabin()`** - Beta-binomial model for overdispersed data
- **`twoac()`** - 2-Alternative Certainty model
- **`samediff()`** - Same-different model
- **`dod()`** - Degree-of-difference model
- **`anota()`** - A-not-A signal detection model

### Power & Sample Size

- **`discrim_power()`**, **`dprime_power()`** - Power analysis
- **`discrim_sample_size()`**, **`dprime_sample_size()`** - Sample size calculation
- **`samediff_power()`** - Same-different power (simulation-based)
- **`twoac_power()`** - 2-AC exact power
- **`dod_power()`** - DOD power analysis

### Inference & Comparison

- **`dprime_test()`** - Test d-prime hypotheses
- **`dprime_compare()`** - Compare d-primes across groups
- **`posthoc()`** - Post-hoc pairwise comparisons

### ROC & Signal Detection

- **`roc()`** - ROC curve computation
- **`auc()`** - Area under the ROC curve
- **`sdt()`** - Signal detection theory transforms

### Simulation

- **`discrim_sim()`** - Simulate discrimination experiments
- **`samediff_sim()`** - Simulate same-different data
- **`dod_sim()`** - Simulate DOD data

### Visualization

- **`plot_psychometric()`** - Psychometric functions
- **`plot_roc()`** - ROC curves
- **`plot_power_curve()`** - Power curves
- **`plot_profile_likelihood()`** - Profile likelihood

## Quick Start

```python
from senspy import discrim, discrim_power

# Analyze a Triangle test result (80 correct out of 100)
result = discrim(correct=80, total=100, method="triangle")
print(f"d-prime: {result.d_prime:.3f}")
print(f"95% CI: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
print(f"p-value: {result.p_value:.4g}")

# Calculate power for a future test
power = discrim_power(d_prime=1.5, sample_size=100, method="triangle")
print(f"Power: {power:.1%}")
```

```python
from senspy import samediff, dprime_compare

# Same-different analysis
sd = samediff(nsamesame=45, ndiffsame=5, nsamediff=20, ndiffdiff=30)
print(f"delta: {sd.delta:.3f}, tau: {sd.tau:.3f}")

# Compare d-primes across products
result = dprime_compare(
    correct=[80, 65, 90],
    total=[100, 100, 100],
    method="triangle"
)
print(f"Chi-square: {result.statistic:.2f}, p={result.p_value:.4f}")
```

## License

This project is licensed under the GNU General Public License version 2.0 or later. See [LICENSE](LICENSE) for details.

## Citation

If you use sensPy in your research, please cite:

```bibtex
@software{senspy,
  title = {sensPy: Python port of sensR for Thurstonian models},
  author = {Aigora},
  year = {2025},
  url = {https://github.com/aigorahub/sensPy}
}
```

## Acknowledgments

This package is a port of [sensR](https://cran.r-project.org/package=sensR) by Per Bruun Brockhoff and Rune Haubo Bojesen Christensen.
