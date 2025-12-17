"""Tests for ROC, AUC, and SDT functions."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from senspy import sdt, roc, auc, ROCResult, AUCResult


class TestSDT:
    """Tests for Signal Detection Theory transform."""

    def test_basic_probit(self):
        """Test basic probit SDT computation."""
        # Simple 2x3 table
        table = np.array([
            [10, 30, 60],  # Signal: cumulative = [10, 40, 100]
            [60, 30, 10],  # Noise: cumulative = [60, 90, 100]
        ])
        result = sdt(table, method="probit")

        assert result.shape == (2, 3)  # J-1 criteria, 3 columns
        # d-prime should be z(hit) - z(fa)
        assert_allclose(result[:, 2], result[:, 0] - result[:, 1])

    def test_probit_vs_logit(self):
        """Test probit and logit give different results."""
        table = np.array([[20, 30, 50], [50, 30, 20]])
        probit_result = sdt(table, method="probit")
        logit_result = sdt(table, method="logit")

        # Results should be different
        assert not np.allclose(probit_result, logit_result)

    def test_invalid_table_shape(self):
        """Test error for non-2-row table."""
        with pytest.raises(ValueError, match="2 x J matrix"):
            sdt(np.array([[1, 2], [3, 4], [5, 6]]))

    def test_invalid_method(self):
        """Test error for invalid method."""
        table = np.array([[10, 20], [20, 10]])
        with pytest.raises(ValueError, match="not recognized"):
            sdt(table, method="invalid")

    def test_equal_distributions(self):
        """Test d-prime near zero for equal distributions."""
        # Symmetric distribution should give d' ~ 0
        table = np.array([[20, 30, 50], [20, 30, 50]])
        result = sdt(table, method="probit")
        assert_allclose(result[:, 2], 0, atol=1e-10)


class TestAUC:
    """Tests for Area Under the ROC Curve."""

    def test_auc_d_zero(self):
        """AUC should be 0.5 when d' = 0."""
        result = auc(d=0)
        assert_allclose(result.value, 0.5)

    def test_auc_positive_d(self):
        """AUC should be > 0.5 for positive d'."""
        result = auc(d=1.5)
        assert result.value > 0.5
        assert result.value < 1.0

    def test_auc_negative_d(self):
        """AUC should be < 0.5 for negative d'."""
        result = auc(d=-1.0)
        assert result.value < 0.5

    def test_auc_with_ci(self):
        """Test confidence interval computation."""
        result = auc(d=1.5, se_d=0.2)
        assert result.lower is not None
        assert result.upper is not None
        assert result.ci_alpha == 0.05
        assert result.lower < result.value < result.upper

    def test_auc_scale_effect(self):
        """Test scale parameter affects AUC."""
        result_scale1 = auc(d=1.5, scale=1.0)
        result_scale2 = auc(d=1.5, scale=2.0)
        # Different scales should give different AUC
        assert result_scale1.value != result_scale2.value

    def test_auc_known_value(self):
        """Test against known analytical value."""
        # AUC = Phi(d / sqrt(2)) for scale=1
        # d = 1.0: AUC = Phi(1/sqrt(2)) = Phi(0.707) ≈ 0.760
        from scipy import stats
        d = 1.0
        expected_auc = stats.norm.cdf(d / np.sqrt(2))
        result = auc(d=d, scale=1.0)
        assert_allclose(result.value, expected_auc)

    def test_auc_invalid_inputs(self):
        """Test validation of inputs."""
        with pytest.raises(ValueError):
            auc(d=np.inf)
        with pytest.raises(ValueError):
            auc(d=1.0, scale=-1)
        with pytest.raises(ValueError):
            auc(d=1.0, ci_alpha=0)
        with pytest.raises(ValueError):
            auc(d=1.0, se_d=-0.1)


class TestROC:
    """Tests for ROC curve computation."""

    def test_roc_basic(self):
        """Test basic ROC curve generation."""
        result = roc(d=1.5)
        assert isinstance(result, ROCResult)
        assert len(result.fpr) == 1000
        assert len(result.tpr) == 1000
        assert result.lower is None
        assert result.upper is None

    def test_roc_endpoints(self):
        """Test ROC curve starts at (0,0) and ends at (1,1)."""
        result = roc(d=1.5)
        assert_allclose(result.fpr[0], 0)
        assert_allclose(result.fpr[-1], 1)
        # TPR should be close to 0 at FPR=0 and close to 1 at FPR=1
        assert result.tpr[0] < 0.1
        assert result.tpr[-1] > 0.9

    def test_roc_with_ci(self):
        """Test ROC curve with confidence bands."""
        result = roc(d=1.5, se_d=0.2)
        assert result.lower is not None
        assert result.upper is not None
        assert len(result.lower) == len(result.fpr)
        # Confidence bands should bracket TPR
        assert np.all(result.lower <= result.tpr + 1e-10)
        assert np.all(result.upper >= result.tpr - 1e-10)

    def test_roc_d_zero(self):
        """ROC should be diagonal when d' = 0."""
        result = roc(d=0)
        # TPR should approximately equal FPR
        assert_allclose(result.tpr, result.fpr, atol=0.01)

    def test_roc_monotonic(self):
        """ROC curve should be monotonically increasing."""
        result = roc(d=1.5)
        assert np.all(np.diff(result.tpr) >= -1e-10)

    def test_roc_n_points(self):
        """Test custom number of points."""
        result = roc(d=1.5, n_points=100)
        assert len(result.fpr) == 100

    def test_roc_invalid_inputs(self):
        """Test validation of inputs."""
        with pytest.raises(ValueError):
            roc(d=np.nan)
        with pytest.raises(ValueError):
            roc(d=1.5, scale=0)
        with pytest.raises(ValueError):
            roc(d=1.5, n_points=1)


class TestROCIntegration:
    """Integration tests for ROC functionality."""

    def test_auc_from_roc_curve(self):
        """AUC computed from ROC curve should match analytical AUC."""
        d = 1.5
        roc_result = roc(d=d, n_points=10000)
        auc_result = auc(d=d)

        # Numerical integration using trapezoidal rule
        numerical_auc = np.trapezoid(roc_result.tpr, roc_result.fpr)
        assert_allclose(numerical_auc, auc_result.value, atol=0.001)

    def test_roc_auc_symmetry(self):
        """AUC(d) + AUC(-d) should equal 1."""
        d = 1.5
        auc_pos = auc(d=d).value
        auc_neg = auc(d=-d).value
        assert_allclose(auc_pos + auc_neg, 1.0)
