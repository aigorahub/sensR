# Repository Manifest

## Project Structure

This repository follows a standard Python package structure managed by Poetry. It is a port of the R package `sensR`.

### Top-Level Directories

- **`senspy/`**: The main Python package source code.
- **`sensR/`**: The original R package source (reference implementation).
- **`tests/`**: Pytest test suite, including unit tests and golden data validation.
- **`docs/`**: Planning and status documentation (`PORTING_PLAN.md`, `GAP_ANALYSIS.md`).
- **`.github/`**: CI/CD workflows for testing and code review.
- **`scripts/`**: Utility scripts (e.g., R scripts to generate golden data).

### Key Source Modules (`senspy/`)

#### Core & Infrastructure
- **`core/`**: Type definitions and base classes.
  - `types.py`: Defines `Protocol` enum, `Statistic` enum.
  - `base.py`: Base classes for results (`DiscrimResult`, `BetaBinomialResult`).
- **`utils/`**: Statistical and transformation utilities.
  - `stats.py`: P-value computations, critical values.
  - `transforms.py`: Conversions between `pc`, `pd`, and `d_prime`.

#### logic & Models
- **`links/`**: Psychometric link functions (the mathematical core).
  - `psychometric.py`: Implements `psy_fun`, `psy_inv`, `psy_deriv` for all protocols.
- **`discrim.py`**: Main entry point for simple discrimination tests (`discrim()`).
- **`betabin.py`**: Beta-binomial models for overdispersion (`betabin()`).
- **`power.py`**: Power analysis and sample size calculations.
- **`twoac.py`**: 2-AC protocol implementation.

### Test Structure (`tests/`)

- **`conftest.py`**: Fixtures, particularly for loading golden JSON data.
- **`test_*.py`**: Unit tests mirroring the module structure.
- **`fixtures/`**: Static JSON files containing validated outputs from R (`golden_sensr.json`).

### Configuration Files

- **`pyproject.toml`**: Dependencies, build config, and tool settings (Ruff, Mypy, Pytest).
- **`poetry.lock`** / **`uv.lock`**: Dependency lock files.
