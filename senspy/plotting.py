"""Interactive plotting functions using Plotly.

This module provides visualization functions for sensory discrimination analysis:
- ROC curves with confidence bands
- Psychometric functions showing discrimination protocols
- Profile likelihood plots for confidence intervals
- Distribution plots for signal detection theory

All plots are interactive Plotly figures that can be displayed in Jupyter
notebooks, exported to HTML, or saved as static images.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from .roc import ROCResult

# Color palette for consistent styling
COLORS = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "signal": "#d62728",
    "noise": "#2ca02c",
    "ci_fill": "rgba(31, 119, 180, 0.2)",
    "reference": "#7f7f7f",
}


def plot_roc(
    roc_result: ROCResult | None = None,
    d_prime: float | None = None,
    se_d: float | None = None,
    scale: float = 1.0,
    n_points: int = 1000,
    ci_alpha: float = 0.05,
    title: str = "ROC Curve",
    show_diagonal: bool = True,
    show_auc: bool = True,
    width: int = 600,
    height: int = 500,
) -> go.Figure:
    """Plot ROC curve with optional confidence bands.

    Parameters
    ----------
    roc_result : ROCResult, optional
        Pre-computed ROC result from `roc()` function. If provided,
        d_prime and se_d are ignored.
    d_prime : float, optional
        D-prime value to compute ROC curve. Required if roc_result not provided.
    se_d : float, optional
        Standard error of d-prime for confidence bands.
    scale : float, default 1.0
        Scale parameter for unequal variance model.
    n_points : int, default 1000
        Number of points on the curve.
    ci_alpha : float, default 0.05
        Alpha level for confidence interval.
    title : str, default "ROC Curve"
        Plot title.
    show_diagonal : bool, default True
        Show the chance diagonal line.
    show_auc : bool, default True
        Display AUC value in the plot.
    width : int, default 600
        Figure width in pixels.
    height : int, default 500
        Figure height in pixels.

    Returns
    -------
    go.Figure
        Plotly figure object.

    Examples
    --------
    >>> from senspy.plotting import plot_roc
    >>> fig = plot_roc(d_prime=1.5, se_d=0.2)
    >>> fig.show()
    """
    from .roc import auc as compute_auc
    from .roc import roc as compute_roc

    # Get or compute ROC data
    if roc_result is not None:
        fpr = roc_result.fpr
        tpr = roc_result.tpr
        lower = roc_result.lower
        upper = roc_result.upper
        d = roc_result.d_prime
        scale = roc_result.scale
    elif d_prime is not None:
        result = compute_roc(
            d=d_prime,
            se_d=se_d,
            scale=scale,
            n_points=n_points,
            ci_alpha=ci_alpha,
        )
        fpr = result.fpr
        tpr = result.tpr
        lower = result.lower
        upper = result.upper
        d = d_prime
    else:
        raise ValueError("Either roc_result or d_prime must be provided")

    fig = go.Figure()

    # Add confidence band if available
    if lower is not None and upper is not None:
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([fpr, fpr[::-1]]),
                y=np.concatenate([upper, lower[::-1]]),
                fill="toself",
                fillcolor=COLORS["ci_fill"],
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name=f"{int((1 - ci_alpha) * 100)}% CI",
                showlegend=True,
            )
        )

    # Add ROC curve
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"ROC (d' = {d:.2f})",
            line=dict(color=COLORS["primary"], width=2),
            hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
        )
    )

    # Add diagonal reference line
    if show_diagonal:
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Chance",
                line=dict(color=COLORS["reference"], width=1, dash="dash"),
                hoverinfo="skip",
            )
        )

    # Compute and display AUC
    if show_auc:
        auc_result = compute_auc(d=d, se_d=se_d, scale=scale, ci_alpha=ci_alpha)
        auc_text = f"AUC = {auc_result.value:.3f}"
        if auc_result.lower is not None:
            auc_text += f"<br>[{auc_result.lower:.3f}, {auc_result.upper:.3f}]"

        fig.add_annotation(
            x=0.95,
            y=0.05,
            xref="paper",
            yref="paper",
            text=auc_text,
            showarrow=False,
            font=dict(size=12),
            align="right",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor=COLORS["primary"],
            borderwidth=1,
            borderpad=4,
        )

    fig.update_layout(
        title=title,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        width=width,
        height=height,
        xaxis=dict(range=[0, 1], constrain="domain"),
        yaxis=dict(range=[0, 1], scaleanchor="x", scaleratio=1),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
        hovermode="closest",
    )

    return fig


def plot_psychometric(
    method: str = "triangle",
    d_prime_range: tuple[float, float] = (0, 4),
    n_points: int = 200,
    show_guessing: bool = True,
    title: str | None = None,
    width: int = 700,
    height: int = 500,
) -> go.Figure:
    """Plot psychometric function for a discrimination protocol.

    Shows the relationship between d-prime and proportion correct (Pc)
    for a given sensory discrimination protocol.

    Parameters
    ----------
    method : str, default "triangle"
        Protocol name: "triangle", "duotrio", "twoAFC", "threeAFC",
        "tetrad", "hexad", "twofive", "twofiveF".
    d_prime_range : tuple, default (0, 4)
        Range of d-prime values to plot.
    n_points : int, default 200
        Number of points on the curve.
    show_guessing : bool, default True
        Show horizontal line at guessing probability.
    title : str, optional
        Plot title. If None, auto-generated from method.
    width : int, default 700
        Figure width in pixels.
    height : int, default 500
        Figure height in pixels.

    Returns
    -------
    go.Figure
        Plotly figure object.

    Examples
    --------
    >>> from senspy.plotting import plot_psychometric
    >>> fig = plot_psychometric(method="triangle")
    >>> fig.show()
    """
    from .links import get_link

    link = get_link(method)
    d_values = np.linspace(d_prime_range[0], d_prime_range[1], n_points)
    pc_values = link.linkinv(d_values)

    if title is None:
        title = f"Psychometric Function: {method.capitalize()} Protocol"

    fig = go.Figure()

    # Add guessing line
    if show_guessing:
        fig.add_hline(
            y=link.p_guess,
            line_dash="dash",
            line_color=COLORS["reference"],
            annotation_text=f"Guessing (Pg = {link.p_guess:.3f})",
            annotation_position="bottom right",
        )

    # Add psychometric curve
    fig.add_trace(
        go.Scatter(
            x=d_values,
            y=pc_values,
            mode="lines",
            name=method.capitalize(),
            line=dict(color=COLORS["primary"], width=2.5),
            hovertemplate="d' = %{x:.2f}<br>Pc = %{y:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="d-prime (d')",
        yaxis_title="Proportion Correct (Pc)",
        width=width,
        height=height,
        xaxis=dict(range=d_prime_range),
        yaxis=dict(range=[0, 1]),
        showlegend=False,
        hovermode="x unified",
    )

    return fig


def plot_psychometric_comparison(
    methods: list[str] | None = None,
    d_prime_range: tuple[float, float] = (0, 4),
    n_points: int = 200,
    title: str = "Psychometric Functions Comparison",
    width: int = 800,
    height: int = 500,
) -> go.Figure:
    """Compare psychometric functions across multiple protocols.

    Parameters
    ----------
    methods : list of str, optional
        Protocols to compare. Default is common protocols.
    d_prime_range : tuple, default (0, 4)
        Range of d-prime values.
    n_points : int, default 200
        Number of points per curve.
    title : str, default "Psychometric Functions Comparison"
        Plot title.
    width : int, default 800
        Figure width in pixels.
    height : int, default 500
        Figure height in pixels.

    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    from .links import get_link

    if methods is None:
        methods = ["triangle", "duotrio", "twoAFC", "threeAFC", "tetrad"]

    # Plotly's default color sequence
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ]

    d_values = np.linspace(d_prime_range[0], d_prime_range[1], n_points)

    fig = go.Figure()

    for i, method in enumerate(methods):
        link = get_link(method)
        pc_values = link.linkinv(d_values)
        color = colors[i % len(colors)]

        fig.add_trace(
            go.Scatter(
                x=d_values,
                y=pc_values,
                mode="lines",
                name=f"{method} (Pg={link.p_guess:.2f})",
                line=dict(color=color, width=2),
                hovertemplate=f"{method}<br>d' = %{{x:.2f}}<br>Pc = %{{y:.3f}}<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="d-prime (d')",
        yaxis_title="Proportion Correct (Pc)",
        width=width,
        height=height,
        xaxis=dict(range=d_prime_range),
        yaxis=dict(range=[0, 1]),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
        hovermode="x unified",
    )

    return fig


def plot_sdt_distributions(
    d_prime: float = 1.5,
    show_criterion: bool = True,
    criterion: float | None = None,
    title: str | None = None,
    width: int = 700,
    height: int = 400,
) -> go.Figure:
    """Plot Signal Detection Theory distributions.

    Shows the noise and signal distributions with optional criterion
    line and shaded hit/false alarm regions.

    Parameters
    ----------
    d_prime : float, default 1.5
        Separation between distributions (d-prime).
    show_criterion : bool, default True
        Show decision criterion line.
    criterion : float, optional
        Criterion location. Default is midpoint (d'/2).
    title : str, optional
        Plot title.
    width : int, default 700
        Figure width in pixels.
    height : int, default 400
        Figure height in pixels.

    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    if criterion is None:
        criterion = d_prime / 2

    if title is None:
        title = f"Signal Detection Theory (d' = {d_prime:.2f})"

    x = np.linspace(-4, d_prime + 4, 500)
    noise_dist = stats.norm.pdf(x, loc=0)
    signal_dist = stats.norm.pdf(x, loc=d_prime)

    fig = go.Figure()

    # Noise distribution
    fig.add_trace(
        go.Scatter(
            x=x,
            y=noise_dist,
            mode="lines",
            name="Noise",
            line=dict(color=COLORS["noise"], width=2),
            fill="tozeroy",
            fillcolor="rgba(44, 160, 44, 0.2)",
        )
    )

    # Signal distribution
    fig.add_trace(
        go.Scatter(
            x=x,
            y=signal_dist,
            mode="lines",
            name="Signal",
            line=dict(color=COLORS["signal"], width=2),
            fill="tozeroy",
            fillcolor="rgba(214, 39, 40, 0.2)",
        )
    )

    # Criterion line
    if show_criterion:
        fig.add_vline(
            x=criterion,
            line_dash="dash",
            line_color=COLORS["primary"],
            line_width=2,
            annotation_text=f"c = {criterion:.2f}",
            annotation_position="top",
        )

        # Add hit rate and FA rate annotations
        hit_rate = 1 - stats.norm.cdf(criterion, loc=d_prime)
        fa_rate = 1 - stats.norm.cdf(criterion, loc=0)

        fig.add_annotation(
            x=criterion + 0.5,
            y=max(noise_dist) * 0.7,
            text=f"Hit = {hit_rate:.2f}<br>FA = {fa_rate:.2f}",
            showarrow=False,
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.8)",
        )

    fig.update_layout(
        title=title,
        xaxis_title="Sensory Magnitude",
        yaxis_title="Probability Density",
        width=width,
        height=height,
        legend=dict(x=0.02, y=0.98),
        hovermode="x unified",
    )

    return fig


def plot_profile_likelihood(
    d_prime_values: ArrayLike,
    log_likelihood_values: ArrayLike,
    levels: tuple[float, ...] = (0.95, 0.99),
    title: str = "Profile Likelihood",
    relative: bool = True,
    width: int = 600,
    height: int = 450,
) -> go.Figure:
    """Plot profile likelihood with confidence levels.

    Parameters
    ----------
    d_prime_values : array_like
        D-prime values at which likelihood was evaluated.
    log_likelihood_values : array_like
        Log-likelihood values (will be normalized to relative likelihood).
    levels : tuple of float, default (0.95, 0.99)
        Confidence levels to display as horizontal lines.
    title : str, default "Profile Likelihood"
        Plot title.
    relative : bool, default True
        Plot relative likelihood (max = 1) vs raw log-likelihood.
    width : int, default 600
        Figure width in pixels.
    height : int, default 450
        Figure height in pixels.

    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    d_values = np.asarray(d_prime_values)
    ll_values = np.asarray(log_likelihood_values)

    # Compute relative likelihood
    max_ll = np.max(ll_values)
    if relative:
        rel_ll = np.exp(ll_values - max_ll)
        y_values = rel_ll
        yaxis_title = "Relative Likelihood"
        y_range = [0, 1.05]
    else:
        y_values = ll_values
        yaxis_title = "Log-Likelihood"
        y_range = None

    fig = go.Figure()

    # Add profile curve
    fig.add_trace(
        go.Scatter(
            x=d_values,
            y=y_values,
            mode="lines",
            name="Profile",
            line=dict(color=COLORS["primary"], width=2),
            hovertemplate="d' = %{x:.3f}<br>L = %{y:.4f}<extra></extra>",
        )
    )

    # Add confidence level lines
    for level in levels:
        cutoff = np.exp(-stats.chi2.ppf(level, df=1) / 2)
        if relative:
            fig.add_hline(
                y=cutoff,
                line_dash="dot",
                line_color=COLORS["reference"],
                annotation_text=f"{int(level * 100)}% CI",
                annotation_position="right",
            )

    # Mark MLE
    mle_idx = np.argmax(ll_values)
    mle_d = d_values[mle_idx]
    fig.add_vline(
        x=mle_d,
        line_dash="dash",
        line_color=COLORS["secondary"],
        line_width=1,
        annotation_text=f"MLE = {mle_d:.3f}",
        annotation_position="top",
    )

    fig.update_layout(
        title=title,
        xaxis_title="d-prime (d')",
        yaxis_title=yaxis_title,
        width=width,
        height=height,
        yaxis=dict(range=y_range) if y_range else {},
        showlegend=False,
        hovermode="x unified",
    )

    return fig


def plot_power_curve(
    d_prime_values: ArrayLike,
    power_values: ArrayLike,
    target_power: float = 0.8,
    title: str = "Power Curve",
    width: int = 600,
    height: int = 450,
) -> go.Figure:
    """Plot power as a function of effect size (d-prime).

    Parameters
    ----------
    d_prime_values : array_like
        D-prime values.
    power_values : array_like
        Corresponding power values.
    target_power : float, default 0.8
        Target power level to highlight.
    title : str, default "Power Curve"
        Plot title.
    width : int, default 600
        Figure width in pixels.
    height : int, default 450
        Figure height in pixels.

    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    d_values = np.asarray(d_prime_values)
    p_values = np.asarray(power_values)

    fig = go.Figure()

    # Add power curve
    fig.add_trace(
        go.Scatter(
            x=d_values,
            y=p_values,
            mode="lines",
            name="Power",
            line=dict(color=COLORS["primary"], width=2.5),
            hovertemplate="d' = %{x:.2f}<br>Power = %{y:.3f}<extra></extra>",
        )
    )

    # Add target power line
    fig.add_hline(
        y=target_power,
        line_dash="dash",
        line_color=COLORS["secondary"],
        annotation_text=f"Target = {target_power}",
        annotation_position="right",
    )

    # Add alpha line
    fig.add_hline(
        y=0.05,
        line_dash="dot",
        line_color=COLORS["reference"],
        annotation_text="α = 0.05",
        annotation_position="right",
    )

    fig.update_layout(
        title=title,
        xaxis_title="d-prime (d')",
        yaxis_title="Power",
        width=width,
        height=height,
        yaxis=dict(range=[0, 1]),
        showlegend=False,
        hovermode="x unified",
    )

    return fig


def plot_sample_size_curve(
    power_values: ArrayLike,
    sample_sizes: ArrayLike,
    target_power: float = 0.8,
    title: str = "Sample Size vs Power",
    width: int = 600,
    height: int = 450,
) -> go.Figure:
    """Plot sample size as a function of target power.

    Parameters
    ----------
    power_values : array_like
        Target power values.
    sample_sizes : array_like
        Required sample sizes.
    target_power : float, default 0.8
        Highlight this power level.
    title : str, default "Sample Size vs Power"
        Plot title.
    width : int, default 600
        Figure width in pixels.
    height : int, default 450
        Figure height in pixels.

    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    p_values = np.asarray(power_values)
    n_values = np.asarray(sample_sizes)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=p_values,
            y=n_values,
            mode="lines",
            name="Sample Size",
            line=dict(color=COLORS["primary"], width=2.5),
            hovertemplate="Power = %{x:.2f}<br>N = %{y:.0f}<extra></extra>",
        )
    )

    # Mark target power
    if target_power in p_values or True:
        # Interpolate to find sample size at target power
        target_n = np.interp(target_power, p_values, n_values)
        fig.add_vline(
            x=target_power,
            line_dash="dash",
            line_color=COLORS["secondary"],
        )
        fig.add_annotation(
            x=target_power,
            y=target_n,
            text=f"N = {target_n:.0f}",
            showarrow=True,
            arrowhead=2,
            ax=40,
            ay=-40,
        )

    fig.update_layout(
        title=title,
        xaxis_title="Power",
        yaxis_title="Sample Size (N)",
        width=width,
        height=height,
        xaxis=dict(range=[0, 1]),
        showlegend=False,
        hovermode="x unified",
    )

    return fig
