# Quick Start

This guide walks you through the basic usage of sensPy for sensory discrimination analysis.

## Basic Discrimination Analysis

The `discrim()` function is the primary tool for analyzing discrimination test data:

```python
from senspy import discrim

# Triangle test: 80 correct out of 100 trials
result = discrim(correct=80, total=100, method="triangle")

print(f"d-prime: {result.d_prime:.3f}")
print(f"Proportion correct: {result.pc:.3f}")
print(f"Proportion discriminating: {result.pd:.3f}")
print(f"P-value: {result.p_value:.4f}")
```

## Supported Protocols

sensPy supports all major discrimination protocols:

| Protocol | Function | Guessing Probability |
|----------|----------|---------------------|
| Triangle | `method="triangle"` | 1/3 |
| Duo-trio | `method="duotrio"` | 1/2 |
| 2-AFC | `method="twoAFC"` | 1/2 |
| 3-AFC | `method="threeAFC"` | 1/3 |
| Tetrad | `method="tetrad"` | 1/3 |
| Hexad | `method="hexad"` | 1/10 |

## Power Analysis

Calculate the power of a discrimination test:

```python
from senspy import discrim_power

power = discrim_power(
    d_prime=1.0,
    n=100,
    method="triangle",
    alpha=0.05
)
print(f"Power: {power:.3f}")
```

## Sample Size Calculation

Determine the required sample size:

```python
from senspy import discrim_sample_size

n = discrim_sample_size(
    d_prime=1.0,
    method="triangle",
    power=0.80,
    alpha=0.05
)
print(f"Required N: {n}")
```

## Interactive Plots

sensPy uses Plotly for interactive visualizations:

```python
from senspy import plot_psychometric, plot_roc

# Compare psychometric functions
fig = plot_psychometric_comparison()
fig.show()

# ROC curve with confidence band
fig = plot_roc(d_prime=1.5, se_d=0.2)
fig.show()
```

## Next Steps

- [Discrimination Protocols](../guide/protocols.md) - Detailed protocol guide
- [Power Analysis](../guide/power.md) - Power and sample size planning
- [Plotting](../guide/plotting.md) - Visualization options
- [API Reference](../api/index.md) - Complete function reference
