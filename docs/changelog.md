# Changelog

All notable changes to sensPy will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial release of sensPy
- Complete port of sensR (R package v1.5-3) to Python
- Psychometric link functions for all major protocols:
  - Triangle, Duo-trio, 2-AFC, 3-AFC, Tetrad, Hexad, Twofive, TwofiveF
- Core discrimination analysis (`discrim()`)
- Beta-binomial model for replicated data (`betabin()`)
- Two-Alternative Certainty model (`twoac()`)
- Same-Different protocol (`samediff()`)
- Degree-of-Difference model (`dod()`, `dod_fit()`, `dod_sim()`)
- A-Not-A protocol (`anota()`)
- Power analysis functions:
  - `discrim_power()`, `dprime_power()`
  - `dod_power()` with Monte Carlo simulation
- Sample size calculation:
  - `discrim_sample_size()`, `dprime_sample_size()`
- D-prime hypothesis testing:
  - `dprime_test()` - single group tests
  - `dprime_compare()` - multiple group comparison
  - `posthoc()` - pairwise comparisons with CLD
- ROC and signal detection:
  - `sdt()` - SDT transform for rating data
  - `roc()` - ROC curve computation
  - `auc()` - Area under curve with CI
- Interactive Plotly visualizations:
  - `plot_roc()` - ROC curves with confidence bands
  - `plot_psychometric()` - psychometric functions
  - `plot_psychometric_comparison()` - protocol comparison
  - `plot_sdt_distributions()` - signal/noise distributions
  - `plot_profile_likelihood()` - profile likelihood plots
  - `plot_power_curve()` - power analysis curves
  - `plot_sample_size_curve()` - sample size planning

### Changed

- N/A (initial release)

### Deprecated

- N/A (initial release)

### Removed

- N/A (initial release)

### Fixed

- N/A (initial release)

### Security

- N/A (initial release)

## Comparison with sensR

sensPy aims for numerical parity with sensR v1.5-3. Key differences:

| Feature | sensR (R) | sensPy (Python) |
|---------|-----------|-----------------|
| Plotting | Base R graphics | Plotly (interactive) |
| Type hints | No | Yes (full typing) |
| Result objects | S3 classes | Dataclasses |
| CI methods | Profile likelihood | Profile likelihood |
| Optimization | `optim()` | `scipy.optimize` |

Results should match within floating-point tolerance (typically 1e-10 for coefficients).
