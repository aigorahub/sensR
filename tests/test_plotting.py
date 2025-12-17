"""Tests for Plotly-based plotting functions."""

import numpy as np
import pytest
import plotly.graph_objects as go

from senspy import roc
from senspy.plotting import (
    plot_roc,
    plot_psychometric,
    plot_psychometric_comparison,
    plot_sdt_distributions,
    plot_profile_likelihood,
    plot_power_curve,
    plot_sample_size_curve,
)


class TestPlotROC:
    """Tests for ROC curve plotting."""

    def test_plot_roc_with_d_prime(self):
        """Test ROC plot with d-prime value."""
        fig = plot_roc(d_prime=1.5)
        assert isinstance(fig, go.Figure)
        # Should have at least ROC curve and diagonal
        assert len(fig.data) >= 2

    def test_plot_roc_with_result(self):
        """Test ROC plot with pre-computed result."""
        roc_result = roc(d=1.5, se_d=0.2)
        fig = plot_roc(roc_result=roc_result)
        assert isinstance(fig, go.Figure)
        # Should have confidence band, ROC curve, and diagonal
        assert len(fig.data) >= 3

    def test_plot_roc_with_ci(self):
        """Test ROC plot with confidence bands."""
        fig = plot_roc(d_prime=1.5, se_d=0.2)
        assert isinstance(fig, go.Figure)
        # Should have more traces with CI
        assert len(fig.data) >= 3

    def test_plot_roc_no_diagonal(self):
        """Test ROC plot without diagonal line."""
        fig = plot_roc(d_prime=1.5, show_diagonal=False)
        # Should have fewer traces
        assert isinstance(fig, go.Figure)

    def test_plot_roc_no_auc(self):
        """Test ROC plot without AUC annotation."""
        fig = plot_roc(d_prime=1.5, show_auc=False)
        assert isinstance(fig, go.Figure)

    def test_plot_roc_custom_dimensions(self):
        """Test custom figure dimensions."""
        fig = plot_roc(d_prime=1.5, width=800, height=600)
        assert fig.layout.width == 800
        assert fig.layout.height == 600

    def test_plot_roc_requires_input(self):
        """Test error when no d-prime or result provided."""
        with pytest.raises(ValueError, match="Either roc_result or d_prime"):
            plot_roc()


class TestPlotPsychometric:
    """Tests for psychometric function plotting."""

    def test_plot_psychometric_default(self):
        """Test default psychometric plot (triangle)."""
        fig = plot_psychometric()
        assert isinstance(fig, go.Figure)
        assert "Triangle" in fig.layout.title.text

    @pytest.mark.parametrize("method", [
        "triangle", "duotrio", "twoAFC", "threeAFC", "tetrad"
    ])
    def test_plot_psychometric_methods(self, method):
        """Test psychometric plot for various methods."""
        fig = plot_psychometric(method=method)
        assert isinstance(fig, go.Figure)

    def test_plot_psychometric_range(self):
        """Test custom d-prime range."""
        fig = plot_psychometric(d_prime_range=(0, 6))
        assert isinstance(fig, go.Figure)

    def test_plot_psychometric_no_guessing(self):
        """Test plot without guessing line."""
        fig = plot_psychometric(show_guessing=False)
        assert isinstance(fig, go.Figure)

    def test_plot_psychometric_custom_title(self):
        """Test custom title."""
        fig = plot_psychometric(title="Custom Title")
        assert fig.layout.title.text == "Custom Title"


class TestPlotPsychometricComparison:
    """Tests for psychometric comparison plotting."""

    def test_comparison_default(self):
        """Test default comparison plot."""
        fig = plot_psychometric_comparison()
        assert isinstance(fig, go.Figure)
        # Should have one trace per default method
        assert len(fig.data) >= 4

    def test_comparison_custom_methods(self):
        """Test comparison with custom methods."""
        methods = ["triangle", "twoAFC"]
        fig = plot_psychometric_comparison(methods=methods)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == len(methods)

    def test_comparison_custom_dimensions(self):
        """Test custom figure dimensions."""
        fig = plot_psychometric_comparison(width=1000, height=600)
        assert fig.layout.width == 1000
        assert fig.layout.height == 600


class TestPlotSDTDistributions:
    """Tests for SDT distribution plotting."""

    def test_sdt_default(self):
        """Test default SDT distribution plot."""
        fig = plot_sdt_distributions()
        assert isinstance(fig, go.Figure)
        # Should have noise and signal distributions
        assert len(fig.data) >= 2

    def test_sdt_custom_dprime(self):
        """Test with custom d-prime."""
        fig = plot_sdt_distributions(d_prime=2.5)
        assert isinstance(fig, go.Figure)
        assert "2.50" in fig.layout.title.text

    def test_sdt_no_criterion(self):
        """Test without criterion line."""
        fig = plot_sdt_distributions(show_criterion=False)
        assert isinstance(fig, go.Figure)

    def test_sdt_custom_criterion(self):
        """Test with custom criterion."""
        fig = plot_sdt_distributions(d_prime=1.5, criterion=1.0)
        assert isinstance(fig, go.Figure)


class TestPlotProfileLikelihood:
    """Tests for profile likelihood plotting."""

    def test_profile_basic(self):
        """Test basic profile likelihood plot."""
        d_values = np.linspace(0.5, 2.5, 50)
        # Simulate a parabolic log-likelihood (normal approximation)
        ll_values = -0.5 * (d_values - 1.5) ** 2 / 0.04
        fig = plot_profile_likelihood(d_values, ll_values)
        assert isinstance(fig, go.Figure)

    def test_profile_custom_levels(self):
        """Test with custom confidence levels."""
        d_values = np.linspace(0.5, 2.5, 50)
        ll_values = -0.5 * (d_values - 1.5) ** 2 / 0.04
        fig = plot_profile_likelihood(d_values, ll_values, levels=(0.90, 0.95))
        assert isinstance(fig, go.Figure)

    def test_profile_not_relative(self):
        """Test raw log-likelihood plot."""
        d_values = np.linspace(0.5, 2.5, 50)
        ll_values = -100 - 0.5 * (d_values - 1.5) ** 2 / 0.04
        fig = plot_profile_likelihood(d_values, ll_values, relative=False)
        assert isinstance(fig, go.Figure)


class TestPlotPowerCurve:
    """Tests for power curve plotting."""

    def test_power_curve_basic(self):
        """Test basic power curve plot."""
        d_values = np.linspace(0, 3, 50)
        # Simulate power curve (monotonic increasing)
        power_values = 1 - np.exp(-d_values)
        fig = plot_power_curve(d_values, power_values)
        assert isinstance(fig, go.Figure)

    def test_power_curve_custom_target(self):
        """Test with custom target power."""
        d_values = np.linspace(0, 3, 50)
        power_values = 1 - np.exp(-d_values)
        fig = plot_power_curve(d_values, power_values, target_power=0.9)
        assert isinstance(fig, go.Figure)


class TestPlotSampleSizeCurve:
    """Tests for sample size curve plotting."""

    def test_sample_size_basic(self):
        """Test basic sample size plot."""
        power_values = np.linspace(0.5, 0.99, 50)
        # Simulate sample size curve (increasing with power)
        sample_sizes = 20 / (1 - power_values)
        fig = plot_sample_size_curve(power_values, sample_sizes)
        assert isinstance(fig, go.Figure)

    def test_sample_size_custom_target(self):
        """Test with custom target power."""
        power_values = np.linspace(0.5, 0.99, 50)
        sample_sizes = 20 / (1 - power_values)
        fig = plot_sample_size_curve(
            power_values, sample_sizes, target_power=0.9
        )
        assert isinstance(fig, go.Figure)


class TestPlotIntegration:
    """Integration tests for plotting functions."""

    def test_all_plots_return_figure(self):
        """All plot functions should return go.Figure."""
        figures = [
            plot_roc(d_prime=1.5),
            plot_psychometric(),
            plot_psychometric_comparison(),
            plot_sdt_distributions(),
            plot_profile_likelihood(
                np.linspace(0, 2, 20),
                -np.linspace(0, 2, 20) ** 2
            ),
            plot_power_curve(
                np.linspace(0, 2, 20),
                np.linspace(0, 0.99, 20)
            ),
            plot_sample_size_curve(
                np.linspace(0.5, 0.99, 20),
                np.linspace(50, 500, 20)
            ),
        ]
        for fig in figures:
            assert isinstance(fig, go.Figure)

    def test_figures_are_serializable(self):
        """All figures should be JSON serializable (for export)."""
        fig = plot_roc(d_prime=1.5)
        # to_json should work without error
        json_str = fig.to_json()
        assert isinstance(json_str, str)
        assert len(json_str) > 0
