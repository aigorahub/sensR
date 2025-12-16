# sensR to sensPy: Master Porting Plan

**Version:** 1.0
**Date:** 2025-12-16
**Status:** Approved
**Scope:** Complete rewrite of senspy from scratch

---

## Executive Summary

This document defines the authoritative roadmap for porting the **sensR** R package (v1.5-3) to **sensPy**, a modern Python library for Thurstonian models in sensory discrimination. The port will achieve 1:1 numerical parity with R while leveraging Python best practices: type hints, NumPy/SciPy integration, Numba optimization, and comprehensive testing.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Design](#2-architecture-design)
3. [Module Structure](#3-module-structure)
4. [Function Inventory](#4-function-inventory)
5. [Implementation Phases](#5-implementation-phases)
6. [Testing Strategy](#6-testing-strategy)
7. [Quality Standards](#7-quality-standards)
8. [Dependencies](#8-dependencies)

---

## 1. Project Overview

### 1.1 Source Package: sensR

The sensR package provides methods for sensory discrimination:

- **Protocols:** duotrio, tetrad, triangle, 2-AFC, 3-AFC, A-not A, same-different, 2-AC, degree-of-difference (DOD)
- **Capabilities:** d-prime estimation, standard errors, confidence intervals, power analysis, sample size calculations
- **Statistical Methods:** Maximum likelihood, profile likelihood, Wald/likelihood ratio tests

### 1.2 Target Package: sensPy

sensPy will be a complete Python port maintaining:
- Identical numerical results (within floating-point tolerance)
- Pythonic API design following scikit-learn conventions
- Modern Python features (type hints, dataclasses, protocols)
- Performance optimization via Numba where beneficial

### 1.3 Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Existing code | Delete & Rewrite | Fresh start ensures clean architecture |
| Testing | Hybrid RPy2/Static | RPy2 for golden data generation, static fixtures for CI |
| Scope | Full (functions + plots + data) | Complete feature parity |
| Min Python | 3.9+ | Modern type hints, dataclasses |
| Optimization | Numba @njit for hot paths | 10-100x speedup for likelihood loops |

---

## 2. Architecture Design

### 2.1 Package Layout

```
senspy/
├── __init__.py              # Public API exports
├── _version.py              # Version info
├── core/
│   ├── __init__.py
│   ├── types.py             # Type definitions, protocols
│   └── base.py              # Base classes (StatsModel, Protocol)
├── links/
│   ├── __init__.py
│   ├── psychometric.py      # Link functions (duotrio, triangle, etc.)
│   └── double.py            # Double protocol links
├── discrimination/
│   ├── __init__.py
│   ├── discrim.py           # discrim() main function
│   ├── anota.py             # A-Not-A protocol
│   └── protocols.py         # Protocol enumeration and helpers
├── models/
│   ├── __init__.py
│   ├── betabinomial.py      # BetaBinomial class
│   ├── twoac.py             # TwoAC class
│   ├── samediff.py          # SameDiff class
│   └── dod.py               # DOD class
├── power/
│   ├── __init__.py
│   ├── discrim_power.py     # discrimPwr, d.primePwr
│   ├── sample_size.py       # discrimSS, d.primeSS
│   └── dod_power.py         # dodPwr
├── inference/
│   ├── __init__.py
│   ├── dprime_tests.py      # dprime_test, dprime_compare, posthoc
│   └── confidence.py        # Profile likelihood CI utilities
├── roc/
│   ├── __init__.py
│   ├── sdt.py               # SDT transform
│   └── auc.py               # ROC, AUC functions
├── utils/
│   ├── __init__.py
│   ├── transforms.py        # pc2pd, pd2pc, rescale
│   ├── critical.py          # findcr, test.crit
│   └── stats.py             # normalPvalue, delimit
├── plotting/
│   ├── __init__.py
│   ├── psychometric.py      # Psychometric function plots
│   ├── profile.py           # Profile likelihood plots
│   └── roc.py               # ROC curve plots
├── datasets/
│   ├── __init__.py
│   └── data.py              # Built-in example datasets
└── _numba/
    ├── __init__.py
    └── likelihoods.py       # JIT-compiled likelihood functions
```

### 2.2 Class Hierarchy

```
StatsModel (ABC)
├── BetaBinomial
├── TwoAC
├── SameDiff
└── DOD

ProtocolLink (Protocol)
├── DuoTrioLink
├── TriangleLink
├── TetradLink
├── TwoAFCLink
├── ThreeAFCLink
├── HexadLink
├── TwoFiveLink
└── TwoFiveFLink

Result (ABC)
├── DiscrimResult
├── BetaBinomialResult
├── TwoACResult
├── SameDiffResult
└── DODResult
```

### 2.3 Core Interfaces

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, Optional
import numpy as np
from numpy.typing import NDArray

class StatsModel(ABC):
    """Base class for all statistical models."""

    @abstractmethod
    def fit(self, data: NDArray) -> "StatsModel":
        """Fit the model to data."""
        ...

    @abstractmethod
    def summary(self) -> str:
        """Return model summary."""
        ...

    @property
    @abstractmethod
    def coefficients(self) -> NDArray:
        """Model coefficients."""
        ...

    @property
    @abstractmethod
    def vcov(self) -> Optional[NDArray]:
        """Variance-covariance matrix."""
        ...

    @abstractmethod
    def confint(self, level: float = 0.95) -> NDArray:
        """Confidence intervals."""
        ...

class ProtocolLink(Protocol):
    """Protocol for psychometric link functions."""

    @property
    def name(self) -> str: ...

    @property
    def p_guess(self) -> float: ...

    def linkinv(self, eta: NDArray) -> NDArray:
        """d-prime to probability (pc)."""
        ...

    def linkfun(self, mu: NDArray) -> NDArray:
        """Probability (pc) to d-prime."""
        ...

    def mu_eta(self, eta: NDArray) -> NDArray:
        """Derivative of linkinv (for delta method)."""
        ...
```

---

## 3. Module Structure

### 3.1 R-to-Python Module Mapping

| R File | Python Module | Description |
|--------|---------------|-------------|
| `links.R` | `senspy.links.psychometric` | Single protocol psychometric functions |
| `doublelinks.R` | `senspy.links.double` | Double protocol psychometric functions |
| `discrim.R` | `senspy.discrimination.discrim` | Main discrimination analysis |
| `betaBin.R` | `senspy.models.betabinomial` | Beta-binomial overdispersion model |
| `twoAC.R` | `senspy.models.twoac` | Two-alternative certainty model |
| `samediff.R` | `senspy.models.samediff` | Same-different protocol model |
| `dod.R` | `senspy.models.dod` | Degree-of-difference model |
| `dod_power.R` | `senspy.power.dod_power` | DOD power analysis |
| `ROC.R` | `senspy.roc` | ROC curves and AUC |
| `power.R` | `senspy.power.discrim_power` | Discrimination power analysis |
| `newPower.R` | `senspy.power.discrim_power` | Alternative power implementations |
| `sample.size.R` | `senspy.power.sample_size` | Sample size calculations |
| `d.primeTest.R` | `senspy.inference.dprime_tests` | d-prime comparison tests |
| `utils.R` | `senspy.utils` | Utility functions |
| `clls.R` | `senspy.inference` | (deprecated, skip) |
| `discrimR.R` | - | (deprecated, skip) |
| `warning_functions.R` | - | (R-specific, skip) |

---

## 4. Function Inventory

### 4.1 Core Discrimination Functions

| R Function | Python Function | Module | Priority |
|------------|-----------------|--------|----------|
| `discrim()` | `discrim()` | `discrimination.discrim` | P0 |
| `AnotA()` | `anota()` | `discrimination.anota` | P1 |
| `discrimSim()` | `discrim_sim()` | `discrimination.discrim` | P2 |
| `rescale()` | `rescale()` | `utils.transforms` | P0 |

### 4.2 Link Functions (P0 - Highest Priority)

| R Function | Python Function | Module |
|------------|-----------------|--------|
| `duotrio()` | `DuoTrioLink` | `links.psychometric` |
| `triangle()` | `TriangleLink` | `links.psychometric` |
| `twoAFC()` | `TwoAFCLink` | `links.psychometric` |
| `threeAFC()` | `ThreeAFCLink` | `links.psychometric` |
| `tetrad()` | `TetradLink` | `links.psychometric` |
| `hexad()` | `HexadLink` | `links.psychometric` |
| `twofive()` | `TwoFiveLink` | `links.psychometric` |
| `twofiveF()` | `TwoFiveFLink` | `links.psychometric` |
| `doubleduotrio()` | `DoubleDuoTrioLink` | `links.double` |
| `doubletriangle()` | `DoubleTriangleLink` | `links.double` |
| `doubletwoAFC()` | `DoubleTwoAFCLink` | `links.double` |
| `doublethreeAFC()` | `DoubleThreeAFCLink` | `links.double` |
| `doubletetrad()` | `DoubleTetradLink` | `links.double` |

### 4.3 Statistical Models (P0-P1)

| R Function | Python Class | Module | Priority |
|------------|--------------|--------|----------|
| `betabin()` | `BetaBinomial` | `models.betabinomial` | P0 |
| `twoAC()` | `TwoAC` | `models.twoac` | P1 |
| `samediff()` | `SameDiff` | `models.samediff` | P1 |
| `dod()` | `DOD` | `models.dod` | P1 |

### 4.4 Power & Sample Size (P1)

| R Function | Python Function | Module |
|------------|-----------------|--------|
| `discrimPwr()` | `discrim_power()` | `power.discrim_power` |
| `d.primePwr()` | `dprime_power()` | `power.discrim_power` |
| `discrimSS()` | `discrim_sample_size()` | `power.sample_size` |
| `d.primeSS()` | `dprime_sample_size()` | `power.sample_size` |
| `twoACpwr()` | `twoac_power()` | `power.twoac_power` |
| `dodPwr()` | `dod_power()` | `power.dod_power` |

### 4.5 Inference & Testing (P1)

| R Function | Python Function | Module |
|------------|-----------------|--------|
| `dprime_test()` | `dprime_test()` | `inference.dprime_tests` |
| `dprime_compare()` | `dprime_compare()` | `inference.dprime_tests` |
| `posthoc()` | `posthoc()` | `inference.dprime_tests` |

### 4.6 ROC & AUC (P2)

| R Function | Python Function | Module |
|------------|-----------------|--------|
| `SDT()` | `sdt_transform()` | `roc.sdt` |
| `ROC()` | `roc_curve()` | `roc.roc` |
| `AUC()` | `auc()` | `roc.auc` |

### 4.7 Utility Functions (P0)

| R Function | Python Function | Module |
|------------|-----------------|--------|
| `getPguess()` | `get_p_guess()` | `utils.stats` |
| `getFamily()` | `get_link()` | `links` |
| `pc2pd()` | `pc_to_pd()` | `utils.transforms` |
| `pd2pc()` | `pd_to_pc()` | `utils.transforms` |
| `psyfun()` | `psy_fun()` | `utils.transforms` |
| `psyinv()` | `psy_inv()` | `utils.transforms` |
| `psyderiv()` | `psy_deriv()` | `utils.transforms` |
| `findcr()` | `find_critical()` | `utils.critical` |
| `delimit()` | `delimit()` | `utils.stats` |
| `normalPvalue()` | `normal_pvalue()` | `utils.stats` |

---

## 5. Implementation Phases

### Phase 0: Foundation (Week 1-2)

**Goal:** Core infrastructure and fundamental functions

**Deliverables:**
- [ ] Package skeleton with pyproject.toml
- [ ] Type definitions (`core/types.py`)
- [ ] Base classes (`core/base.py`)
- [ ] All psychometric link functions (`links/`)
- [ ] Utility transforms (`utils/transforms.py`)
- [ ] `rescale()` function
- [ ] Basic test infrastructure with golden data

**Acceptance Criteria:**
- All link functions produce identical results to R (tolerance: 1e-12)
- `rescale()` matches R output exactly

### Phase 1: Core Models (Week 3-5)

**Goal:** Main statistical models and discrimination analysis

**Deliverables:**
- [ ] `discrim()` function with all statistics (exact, likelihood, Wald, score)
- [ ] `BetaBinomial` class with fit/summary/confint
- [ ] `TwoAC` class with profile likelihood
- [ ] `SameDiff` class
- [ ] Power functions (`discrim_power`, `dprime_power`)
- [ ] Sample size functions (`discrim_sample_size`, `dprime_sample_size`)

**Acceptance Criteria:**
- Coefficients match R within 1e-10
- Log-likelihoods match R within 1e-9
- Confidence intervals match R within 1e-6

### Phase 2: Advanced Models (Week 6-8)

**Goal:** DOD model and d-prime comparison tests

**Deliverables:**
- [ ] `DOD` class with all fitting options
- [ ] `dod_power()` with simulation
- [ ] `dprime_test()` function
- [ ] `dprime_compare()` function
- [ ] `posthoc()` with letter displays
- [ ] `A-Not-A` protocol

**Acceptance Criteria:**
- All test statistics match R output
- P-values match within 1e-8
- Letter displays identical to R

### Phase 3: Visualization & Polish (Week 9-10)

**Goal:** Plotting functions and documentation

**Deliverables:**
- [ ] Profile likelihood plots
- [ ] Psychometric function plots
- [ ] ROC curves
- [ ] Complete API documentation
- [ ] Tutorial notebooks
- [ ] Performance benchmarks

**Acceptance Criteria:**
- Plots visually match R output
- Documentation 100% complete
- All benchmarks pass

---

## 6. Testing Strategy

See [TESTING_STRATEGY.md](TESTING_STRATEGY.md) for complete details.

### 6.1 Test Types

| Type | Purpose | Tool |
|------|---------|------|
| Unit | Individual function correctness | pytest |
| Golden Data | 1:1 R parity validation | pytest + fixtures |
| Property | Mathematical invariants | hypothesis |
| Integration | End-to-end workflows | pytest |
| Benchmark | Performance regression | pytest-benchmark |

### 6.2 R Validation Approach (Hybrid)

**Development Phase:**
- Use RPy2 to call sensR directly and compare outputs
- Generate golden data fixtures from RPy2 calls
- Store fixtures as JSON for CI portability

**CI Phase:**
- Load static JSON fixtures
- No R dependency required
- Fast execution (~30s total)

### 6.3 Tolerances

| Metric | Tolerance | Rationale |
|--------|-----------|-----------|
| Coefficients | 1e-10 | Numerical precision of optimization |
| Log-likelihood | 1e-9 | Accumulated floating-point error |
| P-values | 1e-8 | Distribution function precision |
| Confidence intervals | 1e-6 | Profile likelihood interpolation |
| Standard errors | 1e-8 | Hessian inversion precision |

---

## 7. Quality Standards

### 7.1 Code Standards

- **Type hints:** Required on all public functions
- **Docstrings:** NumPy style, required on all public API
- **Line length:** 88 characters (Black default)
- **Imports:** isort with black profile
- **Linting:** ruff with strict settings

### 7.2 Documentation Standards

- Every public function has:
  - Parameter descriptions
  - Return type documentation
  - At least one example
  - Reference to corresponding R function
- Module-level docstrings explain purpose and usage

### 7.3 Test Coverage Requirements

- Minimum 90% line coverage
- 100% coverage for core numerical functions
- Every R function parity test

---

## 8. Dependencies

### 8.1 Runtime Dependencies

```toml
[project]
dependencies = [
    "numpy>=1.20",
    "scipy>=1.7",
    "pandas>=1.3",
    "numba>=0.55",
    "matplotlib>=3.4",
]
```

### 8.2 Development Dependencies

```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "hypothesis>=6.0",
    "rpy2>=3.5",  # For golden data generation
    "black>=23.0",
    "ruff>=0.1",
    "mypy>=1.0",
    "sphinx>=6.0",
    "sphinx-rtd-theme>=1.0",
]
```

---

## Appendix A: sensR Function Reference

### A.1 Exported Functions (NAMESPACE)

```
# Core discrimination
discrim, AnotA, discrimSim

# Link functions
duotrio, triangle, tetrad, twoAFC, threeAFC, hexad, twofive, twofiveF
doubleduotrio, doubletriangle, doubletetrad, doubletwoAFC, doublethreeAFC

# Models
betabin, twoAC, samediff, dod, dod_fit

# Power & sample size
discrimPwr, d.primePwr, discrimSS, d.primeSS
twoACpwr, dodPwr, samediffPwr

# Inference
dprime_test, dprime_compare, posthoc

# ROC
SDT, ROC, AUC

# Utilities
rescale, getPguess, getFamily
pc2pd, pd2pc, psyfun, psyinv, psyderiv
findcr, delimit, normalPvalue

# S3 methods
print.*, summary.*, plot.*, confint.*, profile.*, vcov.*, logLik.*
```

### A.2 Internal Functions (Not Exported)

These may be implemented as private functions or skipped:

```
# Likelihood functions
nll.2AC, llSameDiff, dod_nll_internal, dprime_nll

# Helpers
profBinom, bbEnvir, estimate.2AC, LRtest.2AC
getPosthoc, dprime_table, dprime_estim

# Deprecated
discrimOld, discrimR
```

---

## Appendix B: R Test Cases

Priority test cases for validation:

```r
# Link functions
library(sensR)
duotrio()$linkinv(1.5)  # Expected: 0.7659283
triangle()$linkfun(0.6)  # Expected: 1.225537

# discrim
discrim(correct=80, total=100, method="triangle")
# d.prime: 2.165, pc: 0.8, pd: 0.7

# betabin
data <- matrix(c(3,6,5,8,9,5,4,7,8,10,6,6,5,5,6,7), ncol=2)
betabin(data, method="triangle")

# twoAC
twoAC(c(25, 35, 40))

# power
discrimPwr(pdA=0.3, sample.size=100, pGuess=1/3)
d.primeSS(d.primeA=1.0, method="triangle", target.power=0.8)
```

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-16 | Claude | Initial comprehensive plan |
