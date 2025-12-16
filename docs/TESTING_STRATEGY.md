# sensR to sensPy: Testing Strategy

**Version:** 1.0
**Date:** 2025-12-16
**Approach:** Hybrid (RPy2 development + Static fixtures for CI)

---

## Overview

This document defines the testing strategy for ensuring 1:1 numerical parity between sensPy (Python) and sensR (R). The hybrid approach uses RPy2 for development-time validation and golden data generation, while CI tests run against pre-generated static fixtures for portability.

---

## Table of Contents

1. [Testing Philosophy](#1-testing-philosophy)
2. [Test Types](#2-test-types)
3. [Golden Data System](#3-golden-data-system)
4. [Tolerance Standards](#4-tolerance-standards)
5. [Directory Structure](#5-directory-structure)
6. [RPy2 Usage Guide](#6-rpy2-usage-guide)
7. [CI Configuration](#7-ci-configuration)
8. [Test Examples](#8-test-examples)

---

## 1. Testing Philosophy

### 1.1 Core Principles

1. **R is the reference:** All numerical results must match sensR output
2. **Reproducibility:** Tests must produce identical results across runs
3. **Isolation:** Tests should not depend on external state
4. **Speed:** CI tests must complete in under 60 seconds
5. **Portability:** CI should not require R installation

### 1.2 Testing Pyramid

```
         /\
        /  \      Integration Tests (10%)
       /----\     End-to-end workflows
      /      \
     /--------\   Golden Data Tests (30%)
    /          \  R parity validation
   /------------\
  /              \  Unit Tests (60%)
 /________________\ Individual functions
```

---

## 2. Test Types

### 2.1 Unit Tests

**Purpose:** Test individual functions in isolation

**Characteristics:**
- Fast execution (<10ms each)
- No external dependencies
- Test edge cases and error handling
- Located in `tests/unit/`

**Example domains:**
- Link function calculations at specific d-prime values
- Probability transforms (pc2pd, pd2pc)
- Utility functions (delimit, findcr)

### 2.2 Golden Data Tests

**Purpose:** Validate numerical parity with R

**Characteristics:**
- Compare Python output against pre-computed R results
- Use static JSON fixtures in CI
- Can regenerate fixtures with RPy2 locally
- Located in `tests/golden/`

**Coverage:**
- All exported R functions
- Multiple parameter combinations per function
- Edge cases (boundary conditions, extreme values)

### 2.3 Property-Based Tests

**Purpose:** Test mathematical invariants

**Characteristics:**
- Use Hypothesis for input generation
- Test properties that must always hold
- Located in `tests/properties/`

**Example properties:**
- `linkfun(linkinv(x)) == x` for valid d-prime
- `linkinv(x)` always in [p_guess, 1]
- `confint` always contains point estimate

### 2.4 Integration Tests

**Purpose:** Test end-to-end workflows

**Characteristics:**
- Test realistic analysis scenarios
- May combine multiple functions
- Located in `tests/integration/`

**Example workflows:**
- Complete discrimination analysis pipeline
- Power analysis followed by sample size calculation
- Model fitting with confidence intervals and plots

---

## 3. Golden Data System

### 3.1 Architecture

```
tests/
├── golden/
│   ├── fixtures/              # Static JSON fixtures (committed)
│   │   ├── links/
│   │   │   ├── duotrio.json
│   │   │   ├── triangle.json
│   │   │   └── ...
│   │   ├── discrim/
│   │   │   └── discrim.json
│   │   ├── models/
│   │   │   ├── betabinomial.json
│   │   │   ├── twoac.json
│   │   │   └── ...
│   │   └── power/
│   │       └── discrim_power.json
│   ├── generate/              # RPy2 scripts to generate fixtures
│   │   ├── generate_links.py
│   │   ├── generate_discrim.py
│   │   └── ...
│   └── test_*.py              # Golden data test modules
└── conftest.py                # Pytest fixtures for loading golden data
```

### 3.2 Fixture Format

Each fixture file is a JSON array of test cases:

```json
{
  "metadata": {
    "r_version": "4.3.1",
    "sensr_version": "1.5-3",
    "generated": "2025-12-16T10:00:00Z",
    "generator": "generate_links.py"
  },
  "test_cases": [
    {
      "id": "duotrio_linkinv_basic",
      "description": "Basic duotrio linkinv at d.prime=1.5",
      "function": "duotrio()$linkinv",
      "inputs": {
        "eta": 1.5
      },
      "expected": {
        "value": 0.7659283383632449,
        "type": "numeric"
      }
    },
    {
      "id": "duotrio_linkinv_vector",
      "description": "Duotrio linkinv with vector input",
      "function": "duotrio()$linkinv",
      "inputs": {
        "eta": [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
      },
      "expected": {
        "value": [0.5, 0.5871962, 0.6729767, 0.7533881, 0.8246459, 0.9293309],
        "type": "numeric_vector"
      }
    }
  ]
}
```

### 3.3 Generating Fixtures

Fixtures are generated using RPy2 scripts that call sensR:

```python
#!/usr/bin/env python3
"""Generate golden data fixtures for link functions."""

import json
from datetime import datetime
from pathlib import Path

import rpy2.robjects as ro
from rpy2.robjects.packages import importr

# Load sensR
sensR = importr("sensR")

def generate_link_fixtures():
    """Generate fixtures for all link functions."""
    fixtures = {
        "metadata": {
            "r_version": str(ro.r("R.version.string")[0]),
            "sensr_version": str(ro.r('packageVersion("sensR")')[0]),
            "generated": datetime.utcnow().isoformat() + "Z",
            "generator": "generate_links.py"
        },
        "test_cases": []
    }

    # Test cases for duotrio
    test_values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

    # linkinv
    link = sensR.duotrio()
    linkinv = link.rx2("linkinv")
    for val in test_values:
        result = linkinv(val)[0]
        fixtures["test_cases"].append({
            "id": f"duotrio_linkinv_{val}",
            "function": "duotrio()$linkinv",
            "inputs": {"eta": val},
            "expected": {"value": result, "type": "numeric"}
        })

    # linkfun
    linkfun = link.rx2("linkfun")
    pc_values = [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    for val in pc_values:
        result = linkfun(val)[0]
        fixtures["test_cases"].append({
            "id": f"duotrio_linkfun_{val}",
            "function": "duotrio()$linkfun",
            "inputs": {"mu": val},
            "expected": {"value": result, "type": "numeric"}
        })

    # ... repeat for other link functions

    return fixtures


if __name__ == "__main__":
    output_path = Path(__file__).parent.parent / "fixtures" / "links" / "duotrio.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fixtures = generate_link_fixtures()
    with open(output_path, "w") as f:
        json.dump(fixtures, f, indent=2)

    print(f"Generated {len(fixtures['test_cases'])} test cases")
```

### 3.4 Running Golden Data Generation

```bash
# Generate all fixtures (requires R + sensR installed)
make generate-golden

# Generate specific fixture
python tests/golden/generate/generate_links.py

# Verify fixtures are up to date
make verify-golden
```

---

## 4. Tolerance Standards

### 4.1 Numerical Tolerances

| Metric Type | Absolute Tolerance | Relative Tolerance | Rationale |
|-------------|-------------------|-------------------|-----------|
| Coefficients (d-prime, pc, pd) | 1e-10 | 1e-10 | Optimization convergence |
| Log-likelihood | 1e-9 | 1e-9 | Accumulated FP error |
| Standard errors | 1e-8 | 1e-8 | Hessian inversion |
| P-values | 1e-8 | 1e-6 | Distribution tail precision |
| Confidence intervals | 1e-6 | 1e-6 | Profile interpolation |
| Power calculations | 1e-6 | 1e-6 | Binomial summation |
| Sample sizes | 0 (exact match) | - | Integer result |

### 4.2 Comparison Functions

```python
import numpy as np
from numpy.testing import assert_allclose

def assert_coefficient_match(actual, expected, name="coefficient"):
    """Assert coefficient matches R output."""
    assert_allclose(
        actual, expected,
        atol=1e-10, rtol=1e-10,
        err_msg=f"{name} mismatch"
    )

def assert_loglik_match(actual, expected):
    """Assert log-likelihood matches R output."""
    assert_allclose(
        actual, expected,
        atol=1e-9, rtol=1e-9,
        err_msg="Log-likelihood mismatch"
    )

def assert_pvalue_match(actual, expected):
    """Assert p-value matches R output."""
    assert_allclose(
        actual, expected,
        atol=1e-8, rtol=1e-6,
        err_msg="P-value mismatch"
    )

def assert_confint_match(actual, expected):
    """Assert confidence interval matches R output."""
    assert_allclose(
        actual, expected,
        atol=1e-6, rtol=1e-6,
        err_msg="Confidence interval mismatch"
    )
```

### 4.3 Handling Edge Cases

Some edge cases require special handling:

```python
def compare_with_edge_handling(actual, expected, tolerance):
    """Compare values with edge case handling."""
    # Handle infinities
    if np.isinf(expected):
        assert np.isinf(actual) and np.sign(actual) == np.sign(expected)
        return

    # Handle NaN
    if np.isnan(expected):
        assert np.isnan(actual)
        return

    # Handle very small values (near zero)
    if abs(expected) < 1e-15:
        assert abs(actual) < 1e-10
        return

    # Standard comparison
    assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)
```

---

## 5. Directory Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── unit/                    # Unit tests
│   ├── test_links.py
│   ├── test_transforms.py
│   ├── test_utils.py
│   └── ...
├── golden/                  # Golden data tests
│   ├── fixtures/            # JSON fixture files
│   │   ├── links/
│   │   ├── discrim/
│   │   ├── models/
│   │   └── power/
│   ├── generate/            # Fixture generation scripts
│   │   ├── generate_links.py
│   │   ├── generate_discrim.py
│   │   └── ...
│   ├── test_links_golden.py
│   ├── test_discrim_golden.py
│   └── ...
├── properties/              # Property-based tests
│   ├── test_link_properties.py
│   └── ...
├── integration/             # Integration tests
│   ├── test_workflows.py
│   └── ...
└── benchmarks/              # Performance benchmarks
    ├── bench_links.py
    └── ...
```

---

## 6. RPy2 Usage Guide

### 6.1 Setup

```bash
# Install R (Ubuntu/Debian)
sudo apt-get install r-base r-base-dev

# Install sensR in R
R -e 'install.packages("sensR", repos="https://cloud.r-project.org")'

# Install rpy2
pip install rpy2
```

### 6.2 Basic Usage

```python
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri

# Enable numpy conversion
numpy2ri.activate()

# Import sensR
sensR = importr("sensR")

# Call discrim
result = sensR.discrim(correct=80, total=100, method="triangle")

# Extract values
d_prime = result.rx2("coefficients")[2, 0]  # d.prime estimate
std_err = result.rx2("coefficients")[2, 1]  # std.err
```

### 6.3 Helper Module

Create a helper module for consistent R interaction:

```python
# tests/rpy2_helper.py
"""Helper functions for RPy2 interaction with sensR."""

from contextlib import contextmanager
from typing import Any, Dict, Optional

import numpy as np

try:
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from rpy2.robjects import numpy2ri

    numpy2ri.activate()
    HAS_RPY2 = True
except ImportError:
    HAS_RPY2 = False


def requires_rpy2(func):
    """Decorator to skip tests if RPy2 is not available."""
    import pytest

    return pytest.mark.skipif(
        not HAS_RPY2,
        reason="RPy2 not available"
    )(func)


@contextmanager
def r_context():
    """Context manager for R operations."""
    if not HAS_RPY2:
        raise RuntimeError("RPy2 not installed")

    sensR = importr("sensR")
    yield sensR


def call_discrim(correct: int, total: int, method: str, **kwargs) -> Dict[str, Any]:
    """Call sensR::discrim and return results as dict."""
    with r_context() as sensR:
        result = sensR.discrim(
            correct=correct,
            total=total,
            method=method,
            **kwargs
        )

        coef = np.array(result.rx2("coefficients"))
        return {
            "pc": {"estimate": coef[0, 0], "std_err": coef[0, 1]},
            "pd": {"estimate": coef[1, 0], "std_err": coef[1, 1]},
            "d_prime": {"estimate": coef[2, 0], "std_err": coef[2, 1]},
            "loglik": float(result.rx2("logLik")[0]),
            "p_value": float(result.rx2("p.value")[0]),
        }
```

### 6.4 Generating Test Data Interactively

```python
# Interactive fixture development
from tests.rpy2_helper import call_discrim

# Test various scenarios
scenarios = [
    {"correct": 80, "total": 100, "method": "triangle"},
    {"correct": 50, "total": 100, "method": "twoAFC"},
    {"correct": 90, "total": 100, "method": "tetrad"},
    {"correct": 34, "total": 100, "method": "threeAFC"},  # Edge: near guessing
]

for scenario in scenarios:
    result = call_discrim(**scenario)
    print(f"Scenario: {scenario}")
    print(f"  d_prime: {result['d_prime']['estimate']:.10f}")
    print(f"  std_err: {result['d_prime']['std_err']:.10f}")
    print()
```

---

## 7. CI Configuration

### 7.1 GitHub Actions Workflow

```yaml
# .github/workflows/test.yml
name: Tests

on:
  push:
    branches: [main, master]
  pull_request:
    branches: [main, master]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11', '3.12']

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"

    - name: Run tests (excluding RPy2)
      run: |
        pytest tests/ -v --cov=senspy --cov-report=xml \
          --ignore=tests/golden/generate/ \
          -m "not rpy2"

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        files: ./coverage.xml

  golden-data-validation:
    # Only run on main branch, with R installed
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
    - uses: actions/checkout@v4

    - name: Set up R
      uses: r-lib/actions/setup-r@v2

    - name: Install sensR
      run: |
        Rscript -e 'install.packages("sensR", repos="https://cloud.r-project.org")'

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'

    - name: Install dependencies with RPy2
      run: |
        pip install -e ".[dev]"

    - name: Validate golden data freshness
      run: |
        python -c "from tests.golden.validate import validate_all; validate_all()"
```

### 7.2 Pytest Configuration

```toml
# pyproject.toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --strict-markers"
markers = [
    "rpy2: tests that require RPy2 and R",
    "slow: tests that take >1 second",
    "benchmark: performance benchmark tests",
]
filterwarnings = [
    "ignore::DeprecationWarning:rpy2.*",
]
```

### 7.3 Marker Usage

```python
import pytest

@pytest.mark.rpy2
def test_discrim_against_r():
    """Test discrim matches R output (requires RPy2)."""
    ...

@pytest.mark.slow
def test_dod_power_simulation():
    """Test DOD power with many simulations."""
    ...
```

---

## 8. Test Examples

### 8.1 Unit Test Example

```python
# tests/unit/test_links.py
"""Unit tests for psychometric link functions."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from senspy.links import DuoTrioLink, TriangleLink


class TestDuoTrioLink:
    """Tests for the duo-trio link function."""

    @pytest.fixture
    def link(self):
        return DuoTrioLink()

    def test_linkinv_at_zero(self, link):
        """linkinv(0) should equal p_guess = 0.5."""
        assert link.linkinv(0.0) == 0.5

    def test_linkinv_at_infinity(self, link):
        """linkinv(inf) should approach 1."""
        assert_allclose(link.linkinv(20.0), 1.0, atol=1e-6)

    def test_linkinv_monotonic(self, link):
        """linkinv should be monotonically increasing."""
        x = np.linspace(0, 10, 100)
        y = link.linkinv(x)
        assert np.all(np.diff(y) >= 0)

    def test_linkfun_inverse_of_linkinv(self, link):
        """linkfun should be the inverse of linkinv."""
        d_primes = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
        pc = link.linkinv(d_primes)
        d_prime_recovered = link.linkfun(pc)
        assert_allclose(d_prime_recovered, d_primes, atol=1e-10)

    def test_mu_eta_positive(self, link):
        """Derivative should be positive for d.prime > 0."""
        x = np.linspace(0.1, 10, 100)
        deriv = link.mu_eta(x)
        assert np.all(deriv > 0)

    @pytest.mark.parametrize("d_prime,expected_pc", [
        (0.0, 0.5),
        (1.0, 0.6729767),
        (2.0, 0.8246459),
    ])
    def test_linkinv_known_values(self, link, d_prime, expected_pc):
        """Test linkinv against known reference values."""
        assert_allclose(link.linkinv(d_prime), expected_pc, atol=1e-6)
```

### 8.2 Golden Data Test Example

```python
# tests/golden/test_links_golden.py
"""Golden data tests for link functions."""

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from senspy.links import get_link

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "links"


def load_fixture(name: str) -> dict:
    """Load a fixture file."""
    with open(FIXTURES_DIR / f"{name}.json") as f:
        return json.load(f)


class TestLinksGolden:
    """Golden data tests for link functions."""

    @pytest.fixture(scope="class")
    def duotrio_fixture(self):
        return load_fixture("duotrio")

    def test_duotrio_linkinv_golden(self, duotrio_fixture):
        """Test duotrio linkinv against R golden data."""
        link = get_link("duotrio")

        for case in duotrio_fixture["test_cases"]:
            if "linkinv" not in case["function"]:
                continue

            eta = case["inputs"]["eta"]
            expected = case["expected"]["value"]

            if isinstance(eta, list):
                actual = link.linkinv(np.array(eta))
                assert_allclose(
                    actual, expected,
                    atol=1e-10, rtol=1e-10,
                    err_msg=f"Failed for case {case['id']}"
                )
            else:
                actual = link.linkinv(eta)
                assert_allclose(
                    actual, expected,
                    atol=1e-10, rtol=1e-10,
                    err_msg=f"Failed for case {case['id']}"
                )


@pytest.mark.parametrize("method", [
    "duotrio", "triangle", "tetrad", "twoAFC", "threeAFC",
    "hexad", "twofive", "twofiveF"
])
class TestAllLinksGolden:
    """Parameterized golden data tests for all link functions."""

    def test_linkinv_golden(self, method):
        """Test linkinv for each method."""
        fixture = load_fixture(method)
        link = get_link(method)

        linkinv_cases = [
            c for c in fixture["test_cases"]
            if "linkinv" in c["function"]
        ]

        for case in linkinv_cases:
            eta = case["inputs"]["eta"]
            expected = case["expected"]["value"]
            actual = link.linkinv(np.atleast_1d(eta))

            assert_allclose(
                actual, np.atleast_1d(expected),
                atol=1e-10, rtol=1e-10,
                err_msg=f"Method {method}, case {case['id']}"
            )
```

### 8.3 Property-Based Test Example

```python
# tests/properties/test_link_properties.py
"""Property-based tests for link functions."""

import numpy as np
from hypothesis import given, assume, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from senspy.links import get_link


@given(st.sampled_from([
    "duotrio", "triangle", "tetrad", "twoAFC", "threeAFC"
]))
@settings(max_examples=100)
def test_linkinv_output_range(method):
    """linkinv output should always be in [p_guess, 1]."""
    link = get_link(method)
    d_primes = np.linspace(0, 10, 50)

    pc = link.linkinv(d_primes)

    assert np.all(pc >= link.p_guess - 1e-10)
    assert np.all(pc <= 1.0 + 1e-10)


@given(
    st.sampled_from(["duotrio", "triangle", "tetrad", "twoAFC", "threeAFC"]),
    st.floats(min_value=0.01, max_value=10.0, allow_nan=False)
)
def test_linkfun_linkinv_roundtrip(method, d_prime):
    """linkfun(linkinv(x)) should equal x."""
    link = get_link(method)

    pc = link.linkinv(d_prime)
    d_prime_recovered = link.linkfun(pc)

    np.testing.assert_allclose(d_prime_recovered, d_prime, rtol=1e-8)


@given(
    st.sampled_from(["duotrio", "triangle", "tetrad", "twoAFC", "threeAFC"]),
    arrays(
        dtype=np.float64,
        shape=st.integers(min_value=2, max_value=100),
        elements=st.floats(min_value=0.0, max_value=10.0, allow_nan=False)
    )
)
def test_linkinv_monotonicity(method, d_primes):
    """linkinv should be monotonically increasing."""
    assume(len(d_primes) >= 2)

    link = get_link(method)
    d_primes_sorted = np.sort(d_primes)

    pc = link.linkinv(d_primes_sorted)

    # Allow small numerical tolerance
    assert np.all(np.diff(pc) >= -1e-10)
```

### 8.4 Integration Test Example

```python
# tests/integration/test_workflows.py
"""Integration tests for complete analysis workflows."""

import numpy as np
import pytest

from senspy import discrim, discrim_power, discrim_sample_size, rescale


class TestDiscriminationWorkflow:
    """Test complete discrimination analysis workflow."""

    def test_triangle_analysis_workflow(self):
        """Test complete triangle test analysis."""
        # 1. Analyze experimental data
        result = discrim(correct=80, total=100, method="triangle")

        # 2. Verify estimates are reasonable
        assert result.d_prime > 0
        assert 0 < result.pc < 1
        assert 0 < result.pd < 1

        # 3. Check confidence interval contains estimate
        ci = result.confint(level=0.95)
        assert ci[0] <= result.d_prime <= ci[1]

        # 4. Compute power for detected effect
        power = discrim_power(
            pd_a=result.pd,
            sample_size=100,
            p_guess=1/3
        )
        assert power > 0.5  # Should have decent power at observed effect

        # 5. Compute sample size for 90% power
        n_required = discrim_sample_size(
            pd_a=result.pd,
            target_power=0.9,
            p_guess=1/3
        )
        assert n_required > 0

    def test_rescale_consistency(self):
        """Test that rescale produces consistent conversions."""
        # Start with d-prime
        d_prime = 1.5
        method = "triangle"

        # Convert d-prime -> pc -> pd -> d-prime (roundtrip)
        result1 = rescale(d_prime=d_prime, method=method)
        result2 = rescale(pc=result1.pc, method=method)
        result3 = rescale(pd=result2.pd, method=method)

        # Should recover original d-prime
        np.testing.assert_allclose(
            result3.d_prime, d_prime,
            atol=1e-10,
            err_msg="Rescale roundtrip failed"
        )
```

---

## Appendix: Makefile Targets

```makefile
# Makefile
.PHONY: test test-unit test-golden test-all generate-golden verify-golden

test:
	pytest tests/ -v --ignore=tests/golden/generate/ -m "not rpy2"

test-unit:
	pytest tests/unit/ -v

test-golden:
	pytest tests/golden/ -v --ignore=tests/golden/generate/

test-all:
	pytest tests/ -v

test-rpy2:
	pytest tests/ -v -m "rpy2"

generate-golden:
	python tests/golden/generate/generate_all.py

verify-golden:
	python tests/golden/generate/verify_fixtures.py

coverage:
	pytest tests/ --cov=senspy --cov-report=html --cov-report=term-missing
```

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-16 | Claude | Initial testing strategy |
