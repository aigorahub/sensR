# Contributor Guide

## Prerequisites

- **Python**: 3.10+
- **Package Manager**: Poetry (or `uv` as seen in workflows)
- **R**: Optional, but required if you need to regenerate golden data using `rpy2`.

## Installation

```bash
# Clone the repository
git clone https://github.com/aigorahub/sensPy.git
cd sensPy

# Install with Poetry
poetry install

# OR install with pip/uv in editable mode
pip install -e ".[dev]"
```

## Development Workflow

1.  **Select a Task**: Check `docs/GAP_ANALYSIS.md` for unimplemented functions.
2.  **Implement**: Write Python code in `senspy/`.
3.  **Validate**: Ensure numerical results match `sensR` exactly.

## Testing

Tests are crucial because this is a port requiring numerical parity.

```bash
# Run all tests
pytest

# Run tests excluding those that require R installed
pytest -m "not rpy2"

# Run specific module tests
pytest tests/test_discrim.py
```

### Golden Data
Tests in `tests/test_sensr_validation.py` compare Python outputs against `tests/fixtures/golden_sensr.json`. If you implement new logic that changes results, you may need to verify against R.

## Code Style & Linting

Settings are defined in `pyproject.toml`.

- **Formatting**: `black` style (via `ruff format` or `black`).
- **Linting**: `ruff check`.
- **Type Checking**: `mypy`.

```bash
# Check style
ruff check .

# Format code
ruff format .
```

## Contribution Process

1.  Create a branch.
2.  Add tests covering the ported functionality.
3.  Update documentation.
4.  Submit PR.
