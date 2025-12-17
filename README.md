# sensPy

A Python port of the R package `sensR` for Thurstonian models in sensory discrimination analysis.

**Status:** Beta (Core functionality implemented)

---

## About

sensPy aims to provide a comprehensive Python implementation of [sensR](https://github.com/aigorahub/sensR), the R package for sensory discrimination methods. This project ensures:

- **1:1 numerical parity** with sensR (validated against R outputs)
- **Modern Python practices** (type hints, dataclasses, NumPy/SciPy integration)
- **Performance optimization**

## Features Implemented

### Discrimination Protocols
- Core protocols supported: Triangle, Duo-Trio, 2-AFC, 3-AFC, Tetrad, Hexad, 2-out-of-5.
- **Function**: `discrim()` provides d-prime estimation, p-values, and confidence intervals.

### Statistical Models
- **Beta-Binomial**: `betabin()` model for over-dispersed data (replicated tests).
- **2-AC**: `twoac()` model for 2-Alternative Certainty protocols.

### Power & Sample Size
- **Power**: `discrim_power()` and `dprime_power()`.
- **Sample Size**: `discrim_sample_size()` and `dprime_sample_size()`.

### Utilities
- **Links**: Full suite of psychometric link functions (forward and inverse).
- **Transforms**: `rescale()`, `pc_to_pd()`, `pd_to_pc()`.

## Installation

```bash
# Clone the repository
git clone https://github.com/aigorahub/sensPy.git
cd sensPy

# Install with Poetry
poetry install

# Or with pip (development mode)
pip install -e .
```

## Development Status

```
Phase 0: Foundation       [████████████████████] 100% (Links, Utilities)
Phase 1: Core Models      [██████████████████░░]  90% (discrim, BetaBinomial, Power)
Phase 2: Advanced Models  [████████░░░░░░░░░░░░]  40% (2-AC implemented, DOD/SameDiff pending)
Phase 3: Visualization    [░░░░░░░░░░░░░░░░░░░░]   0% (Plotting pending)
```

See `docs/GAP_ANALYSIS.md` for detailed function-by-function status.

## Usage Example

```python
from senspy import discrim, discrim_power

# Analyze a Triangle test result (80 correct out of 100)
result = discrim(correct=80, total=100, method="triangle")
print(f"d-prime: {result.d_prime:.3f}")
print(f"p-value: {result.p_value:.4g}")

# Calculate power for a future test
power = discrim_power(pd_a=0.3, sample_size=100, p_guess=1/3)
print(f"Power: {power:.2f}")
```

## Documentation

- [manifest.md](manifest.md): Codebase structure.
- [contributor.md](contributor.md): Setup and testing guide.
- [architecture.md](architecture.md): System design.

## License

This project is licensed under the GNU General Public License version 2.0 or later. See [LICENSE](LICENSE) for details.
