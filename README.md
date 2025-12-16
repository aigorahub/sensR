# sensPy

A Python port of the R package `sensR` for Thurstonian models in sensory discrimination analysis.

**Status:** Complete rewrite in progress (see [Porting Plan](docs/PORTING_PLAN.md))

---

## About

sensPy aims to provide a comprehensive Python implementation of [sensR](https://github.com/aigorahub/sensR), the R package for sensory discrimination methods. This project is undergoing a complete architectural redesign to ensure:

- **1:1 numerical parity** with sensR (validated against R outputs)
- **Modern Python practices** (type hints, dataclasses, NumPy/SciPy integration)
- **Performance optimization** via Numba for computationally intensive functions
- **Complete feature coverage** including all protocols, models, and plotting

## Planned Features

When complete, sensPy will provide:

### Discrimination Protocols
- **Forced-choice:** 2-AFC, 3-AFC, duo-trio, triangle, tetrad, hexad, 2/5, 2/5F
- **Double protocols:** All of the above in double versions
- **Rating scale:** A-Not-A, same-different, 2-AC, degree-of-difference (DOD)

### Statistical Analysis
- d-prime estimation with standard errors
- Multiple confidence interval methods (Wald, profile likelihood)
- Hypothesis testing (exact, likelihood ratio, Wald, score)
- d-prime comparison tests across groups

### Power & Sample Size
- Power analysis for all discrimination protocols
- Sample size calculations for target power

### Visualization
- Psychometric function plots
- ROC curves
- Profile likelihood plots

## Project Documentation

| Document | Description |
|----------|-------------|
| [PORTING_PLAN.md](docs/PORTING_PLAN.md) | Master roadmap for the porting project |
| [TESTING_STRATEGY.md](docs/TESTING_STRATEGY.md) | Testing approach for R validation |
| [GAP_ANALYSIS.md](docs/GAP_ANALYSIS.md) | Current implementation status |

## Installation

The package is not yet available on PyPI. For development:

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
Phase 0: Foundation       [░░░░░░░░░░░░░░░░░░░░]  0%  (Links, Utilities)
Phase 1: Core Models      [░░░░░░░░░░░░░░░░░░░░]  0%  (discrim, BetaBinomial)
Phase 2: Advanced Models  [░░░░░░░░░░░░░░░░░░░░]  0%  (DOD, twoAC, samediff)
Phase 3: Visualization    [░░░░░░░░░░░░░░░░░░░░]  0%  (Plotting, Datasets)
```

See [GAP_ANALYSIS.md](docs/GAP_ANALYSIS.md) for detailed function-by-function status.

## Architecture

The planned package structure:

```
senspy/
├── core/           # Base classes and type definitions
├── links/          # Psychometric link functions
├── discrimination/ # Main discrimination analysis
├── models/         # Statistical models (BetaBinomial, TwoAC, etc.)
├── power/          # Power and sample size calculations
├── inference/      # d-prime tests and comparisons
├── roc/            # ROC curves and AUC
├── utils/          # Utility functions
└── plotting/       # Visualization
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Key areas where help is needed:
- Implementing psychometric link functions
- Writing golden data tests against R
- Performance benchmarking
- Documentation

## License

This project is licensed under the GNU General Public License version 2.0 or later. See [LICENSE](LICENSE) for details.

## Citation

If you use sensPy in your research, please cite:

```bibtex
@software{senspy,
  title = {sensPy: Thurstonian Models for Sensory Discrimination in Python},
  author = {Aigora},
  year = {2025},
  url = {https://github.com/aigorahub/sensPy}
}
```

## Acknowledgments

sensPy is a port of [sensR](https://github.com/aigorahub/sensR), developed by:
- Rune Haubo Bojesen Christensen
- Per Bruun Brockhoff

The original sensR methodology is described in:

> Brockhoff, P.B. and Christensen, R.H.B. (2010). Thurstonian models for sensory discrimination tests as generalized linear models. *Food Quality and Preference*, 21(3), 330-338. [doi:10.1016/j.foodqual.2009.04.003](https://doi.org/10.1016/j.foodqual.2009.04.003)
