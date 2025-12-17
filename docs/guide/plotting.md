# Plotting

sensPy uses [Plotly](https://plotly.com/python/) for interactive visualizations that work seamlessly in Jupyter notebooks and can be exported to various formats.

## ROC Curves

Plot Receiver Operating Characteristic curves for signal detection analysis:

```python
from senspy import plot_roc

# Basic ROC curve
fig = plot_roc(d_prime=1.5)
fig.show()

# With confidence bands
fig = plot_roc(d_prime=1.5, se_d=0.2)
fig.show()
```

### Customization

```python
fig = plot_roc(
    d_prime=1.5,
    se_d=0.2,
    ci_alpha=0.05,        # 95% CI
    title="My ROC Curve",
    show_diagonal=True,   # Show chance line
    show_auc=True,        # Display AUC value
    width=700,
    height=600
)
```

## Psychometric Functions

Visualize the relationship between d-prime and proportion correct:

```python
from senspy import plot_psychometric

# Single protocol
fig = plot_psychometric(method="triangle")
fig.show()
```

### Compare Multiple Protocols

```python
from senspy import plot_psychometric_comparison

fig = plot_psychometric_comparison(
    methods=["triangle", "duotrio", "twoAFC", "threeAFC", "tetrad"]
)
fig.show()
```

## Signal Detection Distributions

Visualize the underlying signal detection theory:

```python
from senspy import plot_sdt_distributions

fig = plot_sdt_distributions(
    d_prime=1.5,
    show_criterion=True,
    criterion=0.5  # Optional: custom criterion location
)
fig.show()
```

This shows:
- Noise distribution (green)
- Signal distribution (red)
- Decision criterion
- Hit rate and false alarm rate

## Profile Likelihood

Plot profile likelihood for confidence interval visualization:

```python
import numpy as np
from senspy import plot_profile_likelihood

# Example: simulate profile likelihood data
d_values = np.linspace(0.5, 2.5, 100)
ll_values = -50 * (d_values - 1.5)**2  # Parabolic approximation

fig = plot_profile_likelihood(
    d_values,
    ll_values,
    levels=(0.95, 0.99)  # Show 95% and 99% CI
)
fig.show()
```

## Power Curves

Visualize statistical power:

```python
import numpy as np
from senspy import discrim_power, plot_power_curve

d_values = np.linspace(0, 3, 50)
powers = [discrim_power(d_prime=d, n=100, method="triangle")
          for d in d_values]

fig = plot_power_curve(d_values, powers, target_power=0.8)
fig.show()
```

## Sample Size Curves

```python
from senspy import plot_sample_size_curve

powers = np.linspace(0.5, 0.95, 30)
n_values = [50 / (1 - p) for p in powers]  # Example relationship

fig = plot_sample_size_curve(powers, n_values, target_power=0.8)
fig.show()
```

## Exporting Plots

### To HTML

```python
fig.write_html("my_plot.html")
```

### To Static Image

Requires kaleido:

```bash
pip install kaleido
```

```python
fig.write_image("my_plot.png", scale=2)  # High resolution
fig.write_image("my_plot.pdf")           # Vector format
fig.write_image("my_plot.svg")           # SVG format
```

### Inline in Jupyter

Plots display automatically in Jupyter. For other environments:

```python
fig.show()
```

## Theming

Plotly figures can be customized with templates:

```python
import plotly.io as pio

# Use a different template
pio.templates.default = "plotly_white"

# Or per-figure
fig.update_layout(template="plotly_dark")
```

Available templates: `plotly`, `plotly_white`, `plotly_dark`, `ggplot2`, `seaborn`, `simple_white`
