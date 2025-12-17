"""Tests for A-Not-A protocol."""

import numpy as np
import pytest

from senspy.anota import anota, ANotAResult


class TestAnotA:
    """Tests for anota function."""

    def test_basic_anota(self):
        """Test basic A-Not-A analysis."""
        result = anota(x1=80, n1=100, x2=70, n2=100)

        assert isinstance(result, ANotAResult)
        assert result.d_prime > 0
        assert result.se_d_prime > 0
        assert 0 <= result.p_value <= 1
        assert result.hit_rate == 0.8
        assert result.false_alarm_rate == 0.3

    def test_high_discrimination(self):
        """Test with high discrimination (large d-prime)."""
        result = anota(x1=95, n1=100, x2=95, n2=100)

        assert result.d_prime > 2.0
        assert result.p_value < 0.001

    def test_low_discrimination(self):
        """Test with low discrimination (small d-prime)."""
        result = anota(x1=55, n1=100, x2=55, n2=100)

        assert result.d_prime < 0.5
        assert result.p_value > 0.01

    def test_equal_hit_and_fa(self):
        """Test when hit rate equals false alarm rate (d-prime near 0)."""
        result = anota(x1=50, n1=100, x2=50, n2=100)

        assert np.abs(result.d_prime) < 0.5
        assert result.hit_rate == 0.5
        assert result.false_alarm_rate == 0.5

    def test_asymmetric_sample_sizes(self):
        """Test with different sample sizes."""
        result = anota(x1=40, n1=50, x2=70, n2=100)

        assert isinstance(result, ANotAResult)
        assert np.isfinite(result.d_prime)
        assert np.isfinite(result.se_d_prime)

    def test_data_attribute(self):
        """Test that data is correctly stored."""
        result = anota(x1=80, n1=100, x2=70, n2=100)

        assert result.data["x1"] == 80
        assert result.data["n1"] == 100
        assert result.data["x2"] == 70
        assert result.data["n2"] == 100


class TestAnotAValidation:
    """Tests for input validation."""

    def test_x1_exceeds_n1_raises(self):
        """Test that x1 > n1 raises error."""
        with pytest.raises(ValueError, match="x1 cannot exceed n1"):
            anota(x1=101, n1=100, x2=70, n2=100)

    def test_x2_exceeds_n2_raises(self):
        """Test that x2 > n2 raises error."""
        with pytest.raises(ValueError, match="x2 cannot exceed n2"):
            anota(x1=80, n1=100, x2=101, n2=100)

    def test_negative_values_raise(self):
        """Test that negative values raise error."""
        with pytest.raises(ValueError, match="positive integer"):
            anota(x1=-1, n1=100, x2=70, n2=100)

    def test_zero_values_raise(self):
        """Test that zero values raise error."""
        with pytest.raises(ValueError, match="positive integer"):
            anota(x1=0, n1=100, x2=70, n2=100)


class TestAnotAEdgeCases:
    """Tests for edge cases."""

    def test_perfect_hit_rate(self):
        """Test perfect hit rate (x1 = n1)."""
        result = anota(x1=100, n1=100, x2=70, n2=100)

        assert result.hit_rate == 1.0
        assert result.d_prime > 2.0
        assert np.isfinite(result.d_prime)

    def test_perfect_correct_rejection(self):
        """Test perfect correct rejection rate (x2 = n2)."""
        result = anota(x1=80, n1=100, x2=100, n2=100)

        assert result.false_alarm_rate == 0.0
        assert result.d_prime > 2.0
        assert np.isfinite(result.d_prime)

    def test_perfect_discrimination(self):
        """Test perfect discrimination (x1=n1 and x2=n2)."""
        result = anota(x1=100, n1=100, x2=100, n2=100)

        assert result.hit_rate == 1.0
        assert result.false_alarm_rate == 0.0
        assert result.d_prime > 4.0
        assert np.isfinite(result.d_prime)

    def test_near_perfect_discrimination(self):
        """Test near-perfect hit and correct rejection rates."""
        result = anota(x1=99, n1=100, x2=99, n2=100)

        assert result.d_prime > 3.0
        assert np.isfinite(result.se_d_prime)

    def test_small_sample(self):
        """Test with small sample sizes."""
        result = anota(x1=8, n1=10, x2=7, n2=10)

        assert np.isfinite(result.d_prime)
        assert np.isfinite(result.p_value)

    def test_float_integer_inputs(self):
        """Test that float integers are accepted."""
        result = anota(x1=80.0, n1=100.0, x2=70.0, n2=100.0)

        assert isinstance(result, ANotAResult)
        assert result.data["x1"] == 80
