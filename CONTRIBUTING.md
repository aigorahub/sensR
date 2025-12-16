# Contributing to sensPy

Thank you for considering contributing to sensPy! This project is a Python port of the R package sensR.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/sensPy.git`
3. Install dependencies: `poetry install`
4. Create a feature branch: `git checkout -b feature/your-feature`

## Development Guidelines

### Code Style
- Follow PEP 8 style guidelines
- Use type hints on all public functions
- Use NumPy-style docstrings
- Format code with `black` and lint with `ruff`

### Testing
- All new functions must have corresponding tests
- Tests must validate against sensR output (see `docs/TESTING_STRATEGY.md`)
- Run tests with: `pytest tests/`

### Documentation
- Update docstrings for any API changes
- Reference the corresponding sensR function in docstrings

## Project Documentation

- [PORTING_PLAN.md](docs/PORTING_PLAN.md) - Implementation roadmap
- [TESTING_STRATEGY.md](docs/TESTING_STRATEGY.md) - Testing approach
- [GAP_ANALYSIS.md](docs/GAP_ANALYSIS.md) - What needs to be implemented

## Submitting Changes

1. Ensure all tests pass
2. Update documentation as needed
3. Submit a pull request with a clear description

## Questions?

Open an issue to discuss ideas before submitting pull requests.
