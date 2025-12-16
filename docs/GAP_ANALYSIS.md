# sensR to sensPy: Gap Analysis

**Version:** 1.0
**Date:** 2025-12-16
**Scope:** Complete rewrite (Delete & Rewrite approach)

---

## Executive Summary

This document provides a comprehensive gap analysis comparing the sensR R package (v1.5-3) against the current sensPy implementation. Following the decision to **delete and rewrite** the existing Python code, all functions are currently unimplemented.

**Overall Status:** 0 of 127 functions implemented (0%)

---

## Table of Contents

1. [Implementation Status Overview](#1-implementation-status-overview)
2. [Detailed Function Gap Analysis](#2-detailed-function-gap-analysis)
3. [Feature Parity Checklist](#3-feature-parity-checklist)
4. [Priority Implementation Queue](#4-priority-implementation-queue)

---

## 1. Implementation Status Overview

### 1.1 Category Summary

| Category | Total Functions | Implemented | Missing | % Complete |
|----------|-----------------|-------------|---------|------------|
| Link Functions | 13 | 0 | 13 | 0% |
| Double Link Functions | 5 | 0 | 5 | 0% |
| Discrimination | 4 | 0 | 4 | 0% |
| Statistical Models | 12 | 0 | 12 | 0% |
| Power Analysis | 9 | 0 | 9 | 0% |
| Sample Size | 4 | 0 | 4 | 0% |
| Inference/Testing | 8 | 0 | 8 | 0% |
| ROC/AUC | 6 | 0 | 6 | 0% |
| Utilities | 14 | 0 | 14 | 0% |
| S3 Methods | 52 | 0 | 52 | 0% |
| **TOTAL** | **127** | **0** | **127** | **0%** |

### 1.2 Visual Progress

```
Link Functions:        [░░░░░░░░░░░░░░░░░░░░]   0%
Double Links:          [░░░░░░░░░░░░░░░░░░░░]   0%
Discrimination:        [░░░░░░░░░░░░░░░░░░░░]   0%
Models:                [░░░░░░░░░░░░░░░░░░░░]   0%
Power Analysis:        [░░░░░░░░░░░░░░░░░░░░]   0%
Sample Size:           [░░░░░░░░░░░░░░░░░░░░]   0%
Inference:             [░░░░░░░░░░░░░░░░░░░░]   0%
ROC/AUC:               [░░░░░░░░░░░░░░░░░░░░]   0%
Utilities:             [░░░░░░░░░░░░░░░░░░░░]   0%
S3 Methods:            [░░░░░░░░░░░░░░░░░░░░]   0%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OVERALL:               [░░░░░░░░░░░░░░░░░░░░]   0%
```

---

## 2. Detailed Function Gap Analysis

### 2.1 Link Functions (`links.R`)

| R Function | Python Target | Status | Priority | Notes |
|------------|---------------|--------|----------|-------|
| `duotrio()` | `DuoTrioLink` | ❌ Missing | P0 | Core - most used |
| `triangle()` | `TriangleLink` | ❌ Missing | P0 | Core - most used |
| `twoAFC()` | `TwoAFCLink` | ❌ Missing | P0 | Core - most used |
| `threeAFC()` | `ThreeAFCLink` | ❌ Missing | P0 | |
| `tetrad()` | `TetradLink` | ❌ Missing | P0 | |
| `hexad()` | `HexadLink` | ❌ Missing | P1 | Less common |
| `twofive()` | `TwoFiveLink` | ❌ Missing | P1 | Less common |
| `twofiveF()` | `TwoFiveFLink` | ❌ Missing | P1 | Less common |

**Required Methods per Link:**
- `linkinv(eta)` - d-prime to probability
- `linkfun(mu)` - probability to d-prime
- `mu.eta(eta)` - derivative for delta method

### 2.2 Double Link Functions (`doublelinks.R`)

| R Function | Python Target | Status | Priority | Notes |
|------------|---------------|--------|----------|-------|
| `doubleduotrio()` | `DoubleDuoTrioLink` | ❌ Missing | P2 | |
| `doubletriangle()` | `DoubleTriangleLink` | ❌ Missing | P2 | |
| `doubletwoAFC()` | `DoubleTwoAFCLink` | ❌ Missing | P2 | |
| `doublethreeAFC()` | `DoubleThreeAFCLink` | ❌ Missing | P2 | |
| `doubletetrad()` | `DoubleTetradLink` | ❌ Missing | P2 | |

### 2.3 Discrimination Analysis (`discrim.R`)

| R Function | Python Target | Status | Priority | Notes |
|------------|---------------|--------|----------|-------|
| `discrim()` | `discrim()` | ❌ Missing | P0 | Main entry point |
| `AnotA()` | `anota()` | ❌ Missing | P1 | Specialized protocol |
| `discrimSim()` | `discrim_sim()` | ❌ Missing | P2 | Simulation |
| `profBinom()` | `_prof_binom()` | ❌ Missing | P1 | Internal helper |

**Required Features:**
- All statistics: exact, likelihood, Wald, score
- Confidence intervals: Wald, profile likelihood
- Profile computation for plotting

### 2.4 Beta-Binomial Model (`betaBin.R`)

| R Function | Python Target | Status | Priority | Notes |
|------------|---------------|--------|----------|-------|
| `betabin()` | `BetaBinomial.fit()` | ❌ Missing | P0 | Main fitting |
| `print.betabin` | `BetaBinomial.__str__()` | ❌ Missing | P1 | |
| `summary.betabin` | `BetaBinomial.summary()` | ❌ Missing | P1 | |
| `vcov.betabin` | `BetaBinomial.vcov` | ❌ Missing | P1 | |
| `logLik.betabin` | `BetaBinomial.loglik` | ❌ Missing | P1 | |

### 2.5 Two-Alternative Certainty Model (`twoAC.R`)

| R Function | Python Target | Status | Priority | Notes |
|------------|---------------|--------|----------|-------|
| `twoAC()` | `TwoAC.fit()` | ❌ Missing | P1 | Main fitting |
| `twoACpwr()` | `twoac_power()` | ❌ Missing | P2 | |
| `estimate.2AC()` | `TwoAC._estimate()` | ❌ Missing | P1 | Internal |
| `nll.2AC()` | `TwoAC._nll()` | ❌ Missing | P1 | Internal |
| `LRtest.2AC()` | `TwoAC.lr_test()` | ❌ Missing | P2 | |
| `profile.twoAC` | `TwoAC.profile()` | ❌ Missing | P2 | |
| `confint.twoAC` | `TwoAC.confint()` | ❌ Missing | P1 | |
| `clm2twoAC()` | `clm_to_twoac()` | ❌ Missing | P3 | Integration |

### 2.6 Same-Different Model (`samediff.R`)

| R Function | Python Target | Status | Priority | Notes |
|------------|---------------|--------|----------|-------|
| `samediff()` | `SameDiff.fit()` | ❌ Missing | P1 | Main fitting |
| `samediffSim()` | `samediff_sim()` | ❌ Missing | P2 | |
| `samediffPwr()` | `samediff_power()` | ❌ Missing | P2 | |
| `profile.samediff` | `SameDiff.profile()` | ❌ Missing | P2 | |
| `confint.samediff` | `SameDiff.confint()` | ❌ Missing | P1 | |
| `summary.samediff` | `SameDiff.summary()` | ❌ Missing | P1 | |

### 2.7 Degree-of-Difference Model (`dod.R`)

| R Function | Python Target | Status | Priority | Notes |
|------------|---------------|--------|----------|-------|
| `dod()` | `DOD.fit()` | ❌ Missing | P1 | Main fitting |
| `dod_fit()` | `DOD._fit()` | ❌ Missing | P1 | Internal |
| `dodControl()` | `DODControl` | ❌ Missing | P1 | Config dataclass |
| `dodSim()` | `dod_sim()` | ❌ Missing | P2 | |
| `optimal_tau()` | `optimal_tau()` | ❌ Missing | P2 | |
| `par2prob_dod()` | `_par_to_prob_dod()` | ❌ Missing | P1 | Internal |
| `dod_nll()` | `_dod_nll()` | ❌ Missing | P1 | Internal |
| `confint.dod_fit` | `DOD.confint()` | ❌ Missing | P1 | |

### 2.8 Power Analysis (`power.R`, `newPower.R`, `dod_power.R`)

| R Function | Python Target | Status | Priority | Notes |
|------------|---------------|--------|----------|-------|
| `discrimPwr()` | `discrim_power()` | ❌ Missing | P1 | Main power |
| `d.primePwr()` | `dprime_power()` | ❌ Missing | P1 | |
| `d.primePwr2()` | `dprime_power2()` | ❌ Missing | P2 | Alternative |
| `normalPwr()` | `_normal_power()` | ❌ Missing | P1 | Internal |
| `pdPwr()` | `_pd_power()` | ❌ Missing | P1 | Internal |
| `binomPwr()` | `_binom_power()` | ❌ Missing | P2 | Internal |
| `dodPwr()` | `dod_power()` | ❌ Missing | P2 | DOD specific |
| `dodPwr_internal()` | `_dod_power_internal()` | ❌ Missing | P2 | Internal |
| `testPwrArgs()` | `_test_power_args()` | ❌ Missing | P2 | Validation |

### 2.9 Sample Size (`sample.size.R`)

| R Function | Python Target | Status | Priority | Notes |
|------------|---------------|--------|----------|-------|
| `discrimSS()` | `discrim_sample_size()` | ❌ Missing | P1 | |
| `d.primeSS()` | `dprime_sample_size()` | ❌ Missing | P1 | |
| `normalSS()` | `_normal_sample_size()` | ❌ Missing | P1 | Internal |
| `pdSS()` | `_pd_sample_size()` | ❌ Missing | P1 | Internal |

### 2.10 Inference & Testing (`d.primeTest.R`)

| R Function | Python Target | Status | Priority | Notes |
|------------|---------------|--------|----------|-------|
| `dprime_test()` | `dprime_test()` | ❌ Missing | P1 | |
| `dprime_compare()` | `dprime_compare()` | ❌ Missing | P1 | |
| `posthoc()` | `posthoc()` | ❌ Missing | P2 | |
| `dprime_table()` | `_dprime_table()` | ❌ Missing | P1 | Internal |
| `dprime_estim()` | `_dprime_estim()` | ❌ Missing | P1 | Internal |
| `dprime_nll()` | `_dprime_nll()` | ❌ Missing | P1 | Internal |
| `dprime_testStat()` | `_dprime_test_stat()` | ❌ Missing | P1 | Internal |
| `getPosthoc()` | `_get_posthoc()` | ❌ Missing | P2 | Internal |

### 2.11 ROC & AUC (`ROC.R`)

| R Function | Python Target | Status | Priority | Notes |
|------------|---------------|--------|----------|-------|
| `SDT()` | `sdt_transform()` | ❌ Missing | P2 | |
| `ROC()` | `roc_curve()` | ❌ Missing | P2 | Generic |
| `ROC.default()` | `roc_default()` | ❌ Missing | P2 | |
| `ROC.anota()` | `roc_anota()` | ❌ Missing | P2 | |
| `AUC()` | `auc()` | ❌ Missing | P2 | Generic |
| `AUC.default()` | `auc_default()` | ❌ Missing | P2 | |

### 2.12 Utilities (`utils.R`)

| R Function | Python Target | Status | Priority | Notes |
|------------|---------------|--------|----------|-------|
| `getPguess()` | `get_p_guess()` | ❌ Missing | P0 | |
| `getFamily()` | `get_link()` | ❌ Missing | P0 | |
| `rescale()` | `rescale()` | ❌ Missing | P0 | Core utility |
| `pc2pd()` | `pc_to_pd()` | ❌ Missing | P0 | |
| `pd2pc()` | `pd_to_pc()` | ❌ Missing | P0 | |
| `psyfun()` | `psy_fun()` | ❌ Missing | P0 | |
| `psyinv()` | `psy_inv()` | ❌ Missing | P0 | |
| `psyderiv()` | `psy_deriv()` | ❌ Missing | P0 | |
| `findcr()` | `find_critical()` | ❌ Missing | P1 | |
| `test.crit()` | `_test_crit()` | ❌ Missing | P1 | Internal |
| `delimit()` | `delimit()` | ❌ Missing | P0 | |
| `normalPvalue()` | `normal_pvalue()` | ❌ Missing | P0 | |
| `lrp_binom()` | `_lr_pvalue_binom()` | ❌ Missing | P2 | Internal |
| `Waldp_binom()` | `_wald_pvalue_binom()` | ❌ Missing | P2 | Internal |

---

## 3. Feature Parity Checklist

### 3.1 Protocol Support

| Protocol | Single | Double | Notes |
|----------|--------|--------|-------|
| Duo-trio | ❌ | ❌ | |
| Triangle | ❌ | ❌ | |
| Tetrad | ❌ | ❌ | |
| 2-AFC | ❌ | ❌ | |
| 3-AFC | ❌ | ❌ | |
| Hexad | ❌ | N/A | No double version in R |
| 2/5 | ❌ | N/A | No double version in R |
| 2/5F | ❌ | N/A | No double version in R |

### 3.2 Statistical Methods

| Feature | discrim | betabin | twoAC | samediff | dod |
|---------|---------|---------|-------|----------|-----|
| ML Estimation | ❌ | ❌ | ❌ | ❌ | ❌ |
| Wald CI | ❌ | ❌ | ❌ | ❌ | ❌ |
| Profile CI | ❌ | ❌ | ❌ | ❌ | ❌ |
| LR Test | ❌ | ❌ | ❌ | ❌ | ❌ |
| Wald Test | ❌ | ❌ | ❌ | ❌ | ❌ |
| Score Test | ❌ | N/A | N/A | N/A | N/A |

### 3.3 Analysis Features

| Feature | Status | Notes |
|---------|--------|-------|
| Power Analysis | ❌ | All methods |
| Sample Size | ❌ | All methods |
| Simulation | ❌ | discrimSim, dodSim, etc. |
| d-prime Tests | ❌ | dprime_test, dprime_compare |
| Post-hoc | ❌ | With letter displays |
| ROC/AUC | ❌ | SDT transform |

### 3.4 Visualization

| Feature | Status | Notes |
|---------|--------|-------|
| Profile plots | ❌ | plot.discrim, etc. |
| Psychometric curves | ❌ | |
| ROC curves | ❌ | |
| Diagnostic plots | ❌ | |

### 3.5 Integration Features

| Feature | Status | Notes |
|---------|--------|-------|
| Pandas integration | ❌ | DataFrames for input/output |
| NumPy arrays | ❌ | Vector operations |
| matplotlib plots | ❌ | Publication quality |
| Numba acceleration | ❌ | Performance critical paths |

---

## 4. Priority Implementation Queue

### 4.1 Phase 0 (Foundation) - Priority P0

Must be implemented first as dependencies for everything else:

```
1. senspy/core/types.py          # Type definitions
2. senspy/core/base.py           # Base classes
3. senspy/utils/stats.py         # delimit, normalPvalue
4. senspy/utils/transforms.py    # pc2pd, pd2pc, rescale
5. senspy/links/psychometric.py  # All 8 single link functions
6. senspy/__init__.py            # Public API
```

**Estimated Test Cases:** ~150 unit tests, ~100 golden data tests

### 4.2 Phase 1 (Core Models) - Priority P1

Core discrimination analysis and statistical models:

```
7. senspy/discrimination/discrim.py   # discrim()
8. senspy/models/betabinomial.py      # BetaBinomial
9. senspy/power/discrim_power.py      # discrimPwr, d.primePwr
10. senspy/power/sample_size.py       # discrimSS, d.primeSS
11. senspy/inference/dprime_tests.py  # dprime_test, dprime_compare
```

**Estimated Test Cases:** ~200 unit tests, ~150 golden data tests

### 4.3 Phase 2 (Advanced Models) - Priority P2

Specialized models and advanced features:

```
12. senspy/links/double.py            # 5 double link functions
13. senspy/models/twoac.py            # TwoAC
14. senspy/models/samediff.py         # SameDiff
15. senspy/models/dod.py              # DOD
16. senspy/power/dod_power.py         # dodPwr
17. senspy/discrimination/anota.py    # AnotA
18. senspy/roc/                       # SDT, ROC, AUC
```

**Estimated Test Cases:** ~300 unit tests, ~200 golden data tests

### 4.4 Phase 3 (Polish) - Priority P3

Visualization and integration:

```
19. senspy/plotting/psychometric.py   # Psychometric curves
20. senspy/plotting/profile.py        # Profile plots
21. senspy/plotting/roc.py            # ROC curves
22. senspy/datasets/                  # Example datasets
23. Documentation                     # Sphinx docs
24. Tutorials                         # Jupyter notebooks
```

**Estimated Test Cases:** ~100 integration tests, visual comparison tests

---

## Appendix A: Mapping Reference

### A.1 R File to Python Module Mapping

| R Source File | Python Module | Functions |
|---------------|---------------|-----------|
| `links.R` | `senspy.links.psychometric` | 8 |
| `doublelinks.R` | `senspy.links.double` | 5 |
| `discrim.R` | `senspy.discrimination.discrim` | 4 |
| `betaBin.R` | `senspy.models.betabinomial` | 5 |
| `twoAC.R` | `senspy.models.twoac` | 8 |
| `samediff.R` | `senspy.models.samediff` | 6 |
| `dod.R` | `senspy.models.dod` | 8+ |
| `dod_power.R` | `senspy.power.dod_power` | 3 |
| `ROC.R` | `senspy.roc` | 6 |
| `power.R` | `senspy.power.discrim_power` | 4 |
| `newPower.R` | `senspy.power.discrim_power` | 5 |
| `sample.size.R` | `senspy.power.sample_size` | 4 |
| `d.primeTest.R` | `senspy.inference.dprime_tests` | 8 |
| `utils.R` | `senspy.utils` | 14 |

### A.2 R Function Name to Python Name Conventions

| R Convention | Python Convention | Example |
|--------------|-------------------|---------|
| `camelCase` | `snake_case` | `getPguess` → `get_p_guess` |
| `dot.separated` | `snake_case` | `d.prime` → `d_prime` |
| `S3.method` | `class.method()` | `print.discrim` → `Discrim.__str__()` |
| `internal_` | `_private` | `dod_nll_internal` → `_dod_nll` |

---

## Appendix B: Test Case Estimates

| Category | Unit Tests | Golden Tests | Property Tests | Total |
|----------|------------|--------------|----------------|-------|
| Links | 80 | 60 | 30 | 170 |
| Discrimination | 40 | 50 | 10 | 100 |
| Models | 120 | 100 | 20 | 240 |
| Power | 60 | 80 | 10 | 150 |
| Inference | 40 | 50 | 10 | 100 |
| ROC | 20 | 30 | 5 | 55 |
| Utilities | 60 | 30 | 20 | 110 |
| Integration | 30 | - | - | 30 |
| **TOTAL** | **450** | **400** | **105** | **955** |

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-16 | Claude | Initial gap analysis |
