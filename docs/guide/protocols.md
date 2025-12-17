# Discrimination Protocols

sensPy implements all major sensory discrimination protocols used in the food, beverage, and consumer products industries.

## Overview

Discrimination tests measure an assessor's ability to detect sensory differences between products. The result is quantified as **d-prime (d')**, a measure of discriminability that is independent of the specific protocol used.

## Forced-Choice Protocols

### Triangle Test

The triangle test presents three samples (two identical, one different). The assessor must identify the odd sample.

```python
from senspy import discrim

# 25 correct out of 50 triangle tests
result = discrim(correct=25, total=50, method="triangle")
print(f"d-prime: {result.d_prime:.3f}")
```

**Properties:**
- Guessing probability: 1/3
- Cognitively demanding
- Widely used in industry

### Duo-Trio Test

Two samples presented with a reference. The assessor identifies which sample matches the reference.

```python
result = discrim(correct=35, total=50, method="duotrio")
```

**Properties:**
- Guessing probability: 1/2
- Less demanding than triangle
- Reference-based comparison

### 2-AFC (Two-Alternative Forced Choice)

Two samples presented; assessor chooses which is stronger on a specified attribute.

```python
result = discrim(correct=60, total=100, method="twoAFC")
```

**Properties:**
- Guessing probability: 1/2
- Attribute-specific
- Most statistically efficient

### 3-AFC (Three-Alternative Forced Choice)

Three samples (two identical controls, one target); assessor identifies the different sample based on a specified attribute.

```python
result = discrim(correct=45, total=100, method="threeAFC")
```

**Properties:**
- Guessing probability: 1/3
- Attribute-specific
- Combines benefits of AFC and triangle

### Tetrad Test

Four samples (two pairs); assessor groups into matching pairs.

```python
result = discrim(correct=40, total=100, method="tetrad")
```

**Properties:**
- Guessing probability: 1/3
- More powerful than triangle
- Good for small differences

### Hexad Test

Six samples (two groups of three); assessor groups into matching triads.

```python
result = discrim(correct=20, total=100, method="hexad")
```

**Properties:**
- Guessing probability: 1/10
- Very sensitive
- Cognitively demanding

## Comparing Protocols

Different protocols have different psychometric functions:

```python
from senspy import plot_psychometric_comparison

fig = plot_psychometric_comparison(
    methods=["triangle", "duotrio", "twoAFC", "threeAFC", "tetrad"]
)
fig.show()
```

At the same d-prime, protocols with lower guessing probability require higher discriminability to achieve the same proportion correct.

## Protocol Selection Guidelines

| Situation | Recommended Protocol |
|-----------|---------------------|
| General difference testing | Triangle, Tetrad |
| Specific attribute | 2-AFC, 3-AFC |
| Less experienced panel | Duo-trio |
| Detecting small differences | Tetrad, Hexad |
| Large sample sizes available | 2-AFC |

## Converting Between Scales

Convert between proportion correct (Pc), proportion discriminating (Pd), and d-prime:

```python
from senspy import psy_fun, psy_inv, pc_to_pd, pd_to_pc

# d-prime to proportion correct
pc = psy_fun(d_prime=1.5, method="triangle")

# proportion correct to d-prime
d_prime = psy_inv(pc=0.6, method="triangle")

# proportion correct to proportion discriminating
pd = pc_to_pd(pc=0.6, method="triangle")
```
