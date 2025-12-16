"""Tests for senspy.utils module - utility functions."""

import numpy as np
import pytest

from senspy.utils import (
    delimit,
    normal_pvalue,
    find_critical,
    pc_to_pd,
    pd_to_pc,
    rescale,
)
from senspy.core.types import Protocol


class TestDelimit:
    """Tests for delimit function."""

    def test_delimit_lower_only(self):
        """Test delimiting with only lower bound."""
        result = delimit([-0.5, 0.5, 1.5], lower=0)
        expected = np.array([0, 0.5, 1.5])
        np.testing.assert_array_equal(result, expected)

    def test_delimit_upper_only(self):
        """Test delimiting with only upper bound."""
        result = delimit([-0.5, 0.5, 1.5], upper=1)
        expected = np.array([-0.5, 0.5, 1])
        np.testing.assert_array_equal(result, expected)

    def test_delimit_both_bounds(self):
        """Test delimiting with both bounds."""
        result = delimit([0.1, 0.5, 0.9], lower=0.2, upper=0.8)
        expected = np.array([0.2, 0.5, 0.8])
        np.testing.assert_array_equal(result, expected)

    def test_delimit_scalar_input(self):
        """Test that scalar input is handled correctly."""
        result = delimit(-0.5, lower=0)
        assert result[0] == 0

    def test_delimit_invalid_bounds_raises(self):
        """Test that lower >= upper raises ValueError."""
        with pytest.raises(ValueError, match="lower.*must be less than upper"):
            delimit([0.5], lower=1, upper=0.5)


class TestNormalPvalue:
    """Tests for normal_pvalue function."""

    def test_two_sided_z_1_96(self):
        """Test two-sided p-value for z = 1.96."""
        result = normal_pvalue(1.96, alternative="two.sided")
        assert result[0] == pytest.approx(0.05, rel=0.01)

    def test_two_sided_z_0(self):
        """Test two-sided p-value for z = 0."""
        result = normal_pvalue(0, alternative="two.sided")
        assert result[0] == pytest.approx(1.0)

    def test_greater_z_1_645(self):
        """Test one-sided (greater) p-value for z = 1.645."""
        result = normal_pvalue(1.645, alternative="greater")
        assert result[0] == pytest.approx(0.05, rel=0.01)

    def test_less_z_negative_1_645(self):
        """Test one-sided (less) p-value for z = -1.645."""
        result = normal_pvalue(-1.645, alternative="less")
        assert result[0] == pytest.approx(0.05, rel=0.01)

    def test_invalid_alternative_raises(self):
        """Test that invalid alternative raises ValueError."""
        with pytest.raises(ValueError, match="Unknown alternative"):
            normal_pvalue(1.0, alternative="invalid")


class TestFindCritical:
    """Tests for find_critical function."""

    def test_difference_test_n100_p05(self):
        """Test critical value for difference test, n=100, p0=0.5, alpha=0.05."""
        result = find_critical(sample_size=100, alpha=0.05, p0=0.5)
        # For n=100, p=0.5, alpha=0.05, critical value should be around 59
        assert result == 59

    def test_difference_test_n100_p033(self):
        """Test critical value for triangle test scenario."""
        result = find_critical(sample_size=100, alpha=0.05, p0=1 / 3)
        # For n=100, p=1/3, alpha=0.05, critical value should be around 42-43
        assert result in (42, 43)  # Boundary value depends on exact computation

    def test_similarity_test(self):
        """Test critical value for similarity test."""
        result = find_critical(
            sample_size=100, alpha=0.05, p0=0.5, test="similarity"
        )
        # For similarity, we're looking for values where H0 can be rejected
        assert isinstance(result, int)
        assert result < 50  # Should be below the mean

    def test_invalid_sample_size_raises(self):
        """Test that invalid sample size raises ValueError."""
        with pytest.raises(ValueError):
            find_critical(sample_size=0)
        with pytest.raises(ValueError):
            find_critical(sample_size=-10)

    def test_invalid_alpha_raises(self):
        """Test that invalid alpha raises ValueError."""
        with pytest.raises(ValueError):
            find_critical(sample_size=100, alpha=0)
        with pytest.raises(ValueError):
            find_critical(sample_size=100, alpha=1)


class TestPcToPd:
    """Tests for pc_to_pd function."""

    def test_pc_to_pd_at_chance(self):
        """Test that pc=p_guess gives pd=0."""
        result = pc_to_pd(0.5, p_guess=0.5)
        assert result[0] == pytest.approx(0.0)

    def test_pc_to_pd_at_perfect(self):
        """Test that pc=1 gives pd=1."""
        result = pc_to_pd(1.0, p_guess=0.5)
        assert result[0] == pytest.approx(1.0)

    def test_pc_to_pd_midpoint(self):
        """Test pc_to_pd at midpoint."""
        # For p_guess=0.5, pc=0.75 should give pd=0.5
        result = pc_to_pd(0.75, p_guess=0.5)
        assert result[0] == pytest.approx(0.5)

    def test_pc_below_p_guess_gives_zero(self):
        """Test that pc below p_guess gives pd=0."""
        result = pc_to_pd(0.3, p_guess=0.5)
        assert result[0] == 0.0

    def test_invalid_p_guess_raises(self):
        """Test that invalid p_guess raises ValueError."""
        with pytest.raises(ValueError):
            pc_to_pd(0.5, p_guess=1.5)
        with pytest.raises(ValueError):
            pc_to_pd(0.5, p_guess=-0.1)


class TestPdToPc:
    """Tests for pd_to_pc function."""

    def test_pd_to_pc_at_zero(self):
        """Test that pd=0 gives pc=p_guess."""
        result = pd_to_pc(0.0, p_guess=0.5)
        assert result[0] == pytest.approx(0.5)

    def test_pd_to_pc_at_one(self):
        """Test that pd=1 gives pc=1."""
        result = pd_to_pc(1.0, p_guess=0.5)
        assert result[0] == pytest.approx(1.0)

    def test_pd_to_pc_midpoint(self):
        """Test pd_to_pc at midpoint."""
        result = pd_to_pc(0.5, p_guess=0.5)
        assert result[0] == pytest.approx(0.75)

    def test_roundtrip_consistency(self):
        """Test that pd_to_pc(pc_to_pd(pc)) ≈ pc."""
        pc_values = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        p_guess = 0.5
        pd = pc_to_pd(pc_values, p_guess)
        pc_roundtrip = pd_to_pc(pd, p_guess)
        np.testing.assert_allclose(pc_roundtrip, pc_values, rtol=1e-10)


class TestRescale:
    """Tests for rescale function."""

    def test_rescale_from_d_prime(self):
        """Test rescaling from d-prime to pc and pd."""
        result = rescale(d_prime=1.5, method="triangle")
        assert hasattr(result, "pc")
        assert hasattr(result, "pd")
        assert hasattr(result, "d_prime")
        assert result.d_prime == pytest.approx(1.5)
        assert result.pc > 1 / 3  # Above chance
        assert result.pd > 0

    def test_rescale_from_pc(self):
        """Test rescaling from pc to d-prime and pd."""
        result = rescale(pc=0.6, method="triangle")
        assert result.pc == pytest.approx(0.6)
        assert result.d_prime > 0
        assert result.pd > 0

    def test_rescale_from_pd(self):
        """Test rescaling from pd to pc and d-prime."""
        result = rescale(pd=0.5, method="triangle")
        assert result.pd == pytest.approx(0.5)
        assert result.pc > 1 / 3
        assert result.d_prime > 0

    def test_rescale_with_se(self):
        """Test rescaling with standard error propagation."""
        result = rescale(d_prime=1.5, se=0.2, method="triangle")
        assert result.se_d_prime == pytest.approx(0.2)
        assert result.se_pc is not None
        assert result.se_pd is not None

    def test_rescale_requires_one_input(self):
        """Test that exactly one of pc, pd, d_prime must be provided."""
        with pytest.raises(ValueError, match="Exactly one"):
            rescale()  # No input
        with pytest.raises(ValueError, match="Exactly one"):
            rescale(pc=0.6, d_prime=1.5)  # Two inputs

    def test_rescale_accepts_protocol_enum(self):
        """Test that Protocol enum is accepted."""
        result = rescale(d_prime=1.5, method=Protocol.TRIANGLE)
        assert result.method == Protocol.TRIANGLE

    def test_rescale_roundtrip_consistency(self):
        """Test rescaling roundtrip consistency."""
        d_prime_orig = 1.5
        result1 = rescale(d_prime=d_prime_orig, method="triangle")
        result2 = rescale(pc=result1.pc, method="triangle")
        assert result2.d_prime == pytest.approx(d_prime_orig, rel=1e-6)
