# Power Analysis

Statistical power is the probability of correctly detecting a true difference when one exists. Proper power analysis is essential for designing effective sensory studies.

## Power Calculation

Calculate the power of a planned study:

```python
from senspy import discrim_power, dprime_power

# Power for triangle test
power = discrim_power(
    d_prime=1.0,      # Expected effect size
    n=100,            # Sample size
    method="triangle",
    alpha=0.05        # Significance level
)
print(f"Power: {power:.1%}")
```

### Using Proportion Discriminating

You can also specify the effect size as proportion discriminating (Pd):

```python
power = discrim_power(
    pd=0.25,          # 25% truly discriminating
    n=100,
    method="triangle",
    alpha=0.05
)
```

## Sample Size Calculation

Determine the required sample size to achieve a target power:

```python
from senspy import discrim_sample_size, dprime_sample_size

# Required N for 80% power
n = discrim_sample_size(
    d_prime=1.0,
    method="triangle",
    power=0.80,
    alpha=0.05
)
print(f"Required N: {n}")
```

## Visualizing Power

```python
import numpy as np
from senspy import discrim_power, plot_power_curve

# Calculate power across effect sizes
d_values = np.linspace(0, 3, 50)
powers = [discrim_power(d_prime=d, n=100, method="triangle")
          for d in d_values]

fig = plot_power_curve(d_values, powers, target_power=0.8)
fig.show()
```

## Sample Size vs Power Curve

```python
from senspy import discrim_sample_size, plot_sample_size_curve

# Calculate sample sizes for various power levels
powers = np.linspace(0.5, 0.95, 20)
sample_sizes = [discrim_sample_size(d_prime=1.0, method="triangle", power=p)
                for p in powers]

fig = plot_sample_size_curve(powers, sample_sizes)
fig.show()
```

## Protocol Comparison

Different protocols require different sample sizes for the same power:

```python
from senspy import discrim_sample_size

methods = ["triangle", "duotrio", "twoAFC", "tetrad"]
d_prime = 1.0

for method in methods:
    n = discrim_sample_size(d_prime=d_prime, method=method, power=0.8)
    print(f"{method:12s}: N = {n}")
```

**Typical results:**
```
triangle    : N = 92
duotrio     : N = 136
twoAFC      : N = 64
tetrad      : N = 78
```

2-AFC is the most efficient, while duo-trio requires the largest samples.

## Advanced: DOD Power Analysis

For Degree-of-Difference (DOD) studies, use Monte Carlo simulation:

```python
from senspy import dod_power

result = dod_power(
    d_prime=1.5,
    n=50,
    tau=[1, 2, 3, 4],
    n_sim=1000,
    alpha=0.05
)
print(f"DOD Power: {result.power:.1%}")
```

## Guidelines

| Target Power | Typical Use |
|--------------|-------------|
| 0.80 | Standard research |
| 0.90 | Confirmatory studies |
| 0.95 | Critical decisions |

**Tips:**
- Always conduct power analysis before collecting data
- Consider the smallest meaningful effect size
- Account for potential dropouts (add 10-20% to calculated N)
- Use pilot data to estimate expected effect size
