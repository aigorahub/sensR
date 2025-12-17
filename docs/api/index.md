# API Reference

Complete reference for all sensPy functions and classes.

## Module Overview

| Module | Description |
|--------|-------------|
| [Core Functions](core.md) | Link functions, utilities, transforms |
| [Discrimination](discrimination.md) | `discrim()`, `anota()` |
| [Models](models.md) | `betabin()`, `twoac()`, `samediff()`, `dod()` |
| [Power & Sample Size](power.md) | Power analysis and sample size calculation |
| [ROC & AUC](roc.md) | Signal detection theory functions |
| [Plotting](plotting.md) | Interactive Plotly visualizations |

## Quick Reference

### Basic Analysis

```python
from senspy import discrim

result = discrim(correct=80, total=100, method="triangle")
```

### Power Analysis

```python
from senspy import discrim_power, discrim_sample_size

power = discrim_power(d_prime=1.0, n=100, method="triangle")
n = discrim_sample_size(d_prime=1.0, method="triangle", power=0.8)
```

### Plotting

```python
from senspy import plot_roc, plot_psychometric

fig = plot_roc(d_prime=1.5)
fig = plot_psychometric(method="triangle")
```

## Conventions

- All d-prime values are on the standard normal scale
- Proportions are in [0, 1], not percentages
- Angles (where applicable) are in radians
- P-values are two-tailed unless otherwise specified
