# sensPy Documentation

sensPy is a Python port of the R package `sensR` for Thurstonian models in sensory discrimination analysis.

**Status:** Complete rewrite in progress

## Project Documentation

| Document | Description |
|----------|-------------|
| [PORTING_PLAN.md](PORTING_PLAN.md) | Master roadmap for porting sensR to Python |
| [TESTING_STRATEGY.md](TESTING_STRATEGY.md) | Hybrid testing approach (RPy2 + static fixtures) |
| [GAP_ANALYSIS.md](GAP_ANALYSIS.md) | Function-by-function implementation status |

## Quick Links

- **GitHub Repository:** [aigorahub/sensPy](https://github.com/aigorahub/sensPy)
- **Original R Package:** [sensR](https://cran.r-project.org/package=sensR)

## Implementation Phases

1. **Phase 0 - Foundation:** Core types, link functions, utilities
2. **Phase 1 - Core Models:** discrim, BetaBinomial, power analysis
3. **Phase 2 - Advanced Models:** DOD, TwoAC, SameDiff, inference tests
4. **Phase 3 - Visualization:** Plotting functions, datasets, documentation

See [PORTING_PLAN.md](PORTING_PLAN.md) for detailed phase descriptions and deliverables.
