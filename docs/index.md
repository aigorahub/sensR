# sensPy

**Thurstonian Models for Sensory Discrimination in Python**

sensPy is a Python port of the R package [sensR](https://cran.r-project.org/package=sensR), providing statistical methods for analyzing sensory discrimination tests.

## Features

- **Discrimination Analysis**: Analyze triangle, duo-trio, 2-AFC, 3-AFC, tetrad, and other protocols
- **Power Analysis**: Calculate statistical power and required sample sizes
- **Advanced Models**: Beta-binomial, Same-Different, Degree-of-Difference (DOD), 2-AC
- **Group Comparisons**: Compare d-prime across multiple groups with post-hoc tests
- **ROC Analysis**: Compute ROC curves and AUC with confidence intervals
- **Interactive Plots**: Beautiful Plotly visualizations

## Quick Example

```python
import senspy as sp

# Analyze a triangle test: 80 correct out of 100
result = sp.discrim(correct=80, total=100, method="triangle")
print(f"d-prime: {result.d_prime:.3f}")
print(f"p-value: {result.p_value:.4f}")

# Calculate power
power = sp.discrim_power(d_prime=1.0, n=100, method="triangle")
print(f"Power: {power:.1%}")

# Interactive ROC curve
fig = sp.plot_roc(d_prime=1.5, se_d=0.2)
fig.show()
```

## Installation

```bash
pip install senspy
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv pip install senspy
```

## Documentation

- [Getting Started](getting-started/installation.md) - Installation and quick start
- [User Guide](guide/protocols.md) - Detailed protocol and analysis guides
- [API Reference](api/index.md) - Complete function reference
- [Changelog](changelog.md) - Version history

## Supported Protocols

| Protocol | Method | Guessing Prob | Double |
|----------|--------|---------------|--------|
| Triangle | `"triangle"` | 1/3 | ✓ |
| Duo-trio | `"duotrio"` | 1/2 | ✓ |
| 2-AFC | `"twoAFC"` | 1/2 | ✓ |
| 3-AFC | `"threeAFC"` | 1/3 | ✓ |
| Tetrad | `"tetrad"` | 1/3 | ✓ |
| Hexad | `"hexad"` | 1/10 | - |
| 2-out-of-5 | `"twofive"` | 1/10 | - |

## License

GPL-2.0-or-later (same as sensR)

## Links

- [GitHub Repository](https://github.com/aigorahub/sensPy)
- [Original sensR Package](https://cran.r-project.org/package=sensR)
