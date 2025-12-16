"""Tests for senspy.power module - power and sample size calculations."""

import numpy as np
import pytest

from senspy import (
    discrim_power,
    dprime_power,
    discrim_sample_size,
    dprime_sample_size,
)
from senspy.core.types import Protocol


class TestDiscrimPower:
    """Tests for discrim_power function."""

    def test_basic_power_calculation(self):
        """Test basic power calculation."""
        power = discrim_power(pd_a=0.3, sample_size=100, p_guess=1/3)
        assert 0 < power <= 1
        # With 30% discriminators and n=100, power should be high
        assert power > 0.9

    def test_power_increases_with_sample_size(self):
        """Test that power increases with sample size."""
        power_50 = discrim_power(pd_a=0.3, sample_size=50, p_guess=1/3)
        power_100 = discrim_power(pd_a=0.3, sample_size=100, p_guess=1/3)
        power_200 = discrim_power(pd_a=0.3, sample_size=200, p_guess=1/3)

        assert power_50 < power_100 < power_200

    def test_power_increases_with_effect_size(self):
        """Test that power increases with larger effect size."""
        power_low = discrim_power(pd_a=0.1, sample_size=100, p_guess=1/3)
        power_med = discrim_power(pd_a=0.3, sample_size=100, p_guess=1/3)
        power_high = discrim_power(pd_a=0.5, sample_size=100, p_guess=1/3)

        assert power_low < power_med < power_high

    def test_similarity_test(self):
        """Test similarity test power calculation."""
        power = discrim_power(
            pd_a=0.1, sample_size=100, pd_0=0.3,
            p_guess=1/3, test="similarity"
        )
        assert 0 < power <= 1

    def test_different_statistics(self):
        """Test different statistical methods give similar results."""
        exact = discrim_power(pd_a=0.3, sample_size=100, p_guess=1/3, statistic="exact")
        normal = discrim_power(pd_a=0.3, sample_size=100, p_guess=1/3, statistic="normal")
        cont = discrim_power(pd_a=0.3, sample_size=100, p_guess=1/3, statistic="cont.normal")

        # All should be in reasonable range
        assert 0.9 < exact < 1.0
        assert 0.9 < normal < 1.0
        assert 0.9 < cont < 1.0

    def test_invalid_pd_a_raises(self):
        """Test that invalid pd_a raises error."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            discrim_power(pd_a=1.5, sample_size=100, p_guess=1/3)

    def test_invalid_sample_size_raises(self):
        """Test that invalid sample_size raises error."""
        with pytest.raises(ValueError, match="positive integer"):
            discrim_power(pd_a=0.3, sample_size=0, p_guess=1/3)

    def test_pd_a_less_than_pd_0_for_difference_raises(self):
        """Test that pd_a < pd_0 raises for difference test."""
        with pytest.raises(ValueError, match="must be >="):
            discrim_power(pd_a=0.1, sample_size=100, pd_0=0.3, p_guess=1/3, test="difference")

    def test_pd_a_greater_than_pd_0_for_similarity_raises(self):
        """Test that pd_a > pd_0 raises for similarity test."""
        with pytest.raises(ValueError, match="must be <="):
            discrim_power(pd_a=0.5, sample_size=100, pd_0=0.3, p_guess=1/3, test="similarity")


class TestDprimePower:
    """Tests for dprime_power function."""

    def test_basic_power_calculation(self):
        """Test basic power calculation with d-prime."""
        power = dprime_power(d_prime_a=1.5, sample_size=100, method="triangle")
        assert 0 < power <= 1

    def test_different_methods(self):
        """Test power for different discrimination methods."""
        for method in ["triangle", "twoafc", "duotrio", "threeafc"]:
            power = dprime_power(d_prime_a=1.0, sample_size=50, method=method)
            assert 0 < power <= 1

    def test_protocol_enum_accepted(self):
        """Test that Protocol enum is accepted."""
        power = dprime_power(d_prime_a=1.5, sample_size=100, method=Protocol.TRIANGLE)
        assert 0 < power <= 1

    def test_invalid_d_prime_raises(self):
        """Test that negative d-prime raises error."""
        with pytest.raises(ValueError, match="non-negative"):
            dprime_power(d_prime_a=-1.0, sample_size=100, method="triangle")


class TestDiscrimSampleSize:
    """Tests for discrim_sample_size function."""

    def test_basic_sample_size_calculation(self):
        """Test basic sample size calculation."""
        n = discrim_sample_size(pd_a=0.3, p_guess=1/3)
        assert n > 0
        assert isinstance(n, int)

    def test_sample_size_increases_with_power(self):
        """Test that required sample size increases with target power."""
        n_80 = discrim_sample_size(pd_a=0.3, p_guess=1/3, target_power=0.80)
        n_90 = discrim_sample_size(pd_a=0.3, p_guess=1/3, target_power=0.90)
        n_95 = discrim_sample_size(pd_a=0.3, p_guess=1/3, target_power=0.95)

        assert n_80 < n_90 < n_95

    def test_sample_size_decreases_with_effect_size(self):
        """Test that required sample size decreases with larger effect."""
        n_large = discrim_sample_size(pd_a=0.2, p_guess=1/3)
        n_small = discrim_sample_size(pd_a=0.4, p_guess=1/3)

        assert n_large > n_small

    def test_different_statistics(self):
        """Test different statistical methods."""
        n_exact = discrim_sample_size(pd_a=0.3, p_guess=1/3, statistic="exact")
        n_normal = discrim_sample_size(pd_a=0.3, p_guess=1/3, statistic="normal")
        n_cont = discrim_sample_size(pd_a=0.3, p_guess=1/3, statistic="cont.normal")

        # All should be reasonable
        assert 30 < n_exact < 200
        assert 30 < n_normal < 200
        assert 30 < n_cont < 200

    def test_achieved_power_meets_target(self):
        """Test that computed n achieves target power."""
        target = 0.90
        pd_a = 0.3
        p_guess = 1/3

        n = discrim_sample_size(pd_a=pd_a, p_guess=p_guess, target_power=target)
        achieved = discrim_power(pd_a=pd_a, sample_size=n, p_guess=p_guess)

        # Achieved power should be at least target
        assert achieved >= target - 0.01  # Small tolerance for discrete nature

    def test_invalid_pd_a_raises(self):
        """Test that pd_a = 0 raises error."""
        with pytest.raises(ValueError):
            discrim_sample_size(pd_a=0.0, p_guess=1/3)

    def test_pd_a_less_than_pd_0_raises(self):
        """Test that pd_a <= pd_0 raises for difference test."""
        with pytest.raises(ValueError, match="must be >"):
            discrim_sample_size(pd_a=0.2, pd_0=0.3, p_guess=1/3, test="difference")


class TestDprimeSampleSize:
    """Tests for dprime_sample_size function."""

    def test_basic_sample_size_calculation(self):
        """Test basic sample size calculation with d-prime."""
        n = dprime_sample_size(d_prime_a=1.5, method="triangle")
        assert n > 0
        assert isinstance(n, int)

    def test_different_methods(self):
        """Test sample size for different discrimination methods."""
        for method in ["triangle", "twoafc", "duotrio"]:
            n = dprime_sample_size(d_prime_a=1.0, method=method, target_power=0.80)
            assert n > 0

    def test_achieved_power_meets_target(self):
        """Test that computed n achieves target power."""
        target = 0.90
        d_prime_a = 1.5
        method = "triangle"

        n = dprime_sample_size(d_prime_a=d_prime_a, method=method, target_power=target)
        achieved = dprime_power(d_prime_a=d_prime_a, sample_size=n, method=method)

        assert achieved >= target - 0.01

    def test_invalid_d_prime_raises(self):
        """Test that d_prime <= 0 raises error."""
        with pytest.raises(ValueError, match="positive"):
            dprime_sample_size(d_prime_a=0.0, method="triangle")


class TestPowerValidation:
    """Validation tests comparing against sensR golden data."""

    def test_discrim_power_matches_sensr(self, golden_power_data):
        """Test discrim_power matches sensR discrimPwr."""
        if golden_power_data is None:
            pytest.skip("Golden data not available")

        for case in golden_power_data["discrim_power"]:
            inp = case["input"]
            expected = case["power"]

            # Test exact statistic
            result_exact = discrim_power(
                pd_a=inp["pd_a"],
                sample_size=inp["sample_size"],
                pd_0=inp["pd_0"],
                p_guess=inp["p_guess"],
                test=inp["test"],
                statistic="exact",
            )
            assert result_exact == pytest.approx(expected["exact"], rel=1e-3), \
                f"exact power mismatch for {inp}"

            # Test normal statistic
            result_normal = discrim_power(
                pd_a=inp["pd_a"],
                sample_size=inp["sample_size"],
                pd_0=inp["pd_0"],
                p_guess=inp["p_guess"],
                test=inp["test"],
                statistic="normal",
            )
            assert result_normal == pytest.approx(expected["normal"], rel=1e-3), \
                f"normal power mismatch for {inp}"

    def test_dprime_power_matches_sensr(self, golden_power_data):
        """Test dprime_power matches sensR d.primePwr."""
        if golden_power_data is None:
            pytest.skip("Golden data not available")

        for case in golden_power_data["dprime_power"]:
            inp = case["input"]
            expected = case["power"]

            result_exact = dprime_power(
                d_prime_a=inp["d_prime_a"],
                sample_size=inp["sample_size"],
                method=inp["method"],
                d_prime_0=inp["d_prime_0"],
                test=inp["test"],
                statistic="exact",
            )
            assert result_exact == pytest.approx(expected["exact"], rel=1e-3), \
                f"exact power mismatch for {inp}"

    def test_discrim_sample_size_matches_sensr(self, golden_sample_size_data):
        """Test discrim_sample_size matches sensR discrimSS."""
        if golden_sample_size_data is None:
            pytest.skip("Golden data not available")

        for case in golden_sample_size_data["discrim_sample_size"]:
            inp = case["input"]
            expected = case["sample_size"]

            # Test exact statistic
            result_exact = discrim_sample_size(
                pd_a=inp["pd_a"],
                pd_0=inp["pd_0"],
                target_power=inp["target_power"],
                p_guess=inp["p_guess"],
                test=inp["test"],
                statistic="exact",
            )
            # Sample size should match exactly or be within 1-2 due to search algorithm
            assert abs(result_exact - expected["exact"]) <= 2, \
                f"exact sample size mismatch for {inp}: got {result_exact}, expected {expected['exact']}"

            # Test normal statistic
            result_normal = discrim_sample_size(
                pd_a=inp["pd_a"],
                pd_0=inp["pd_0"],
                target_power=inp["target_power"],
                p_guess=inp["p_guess"],
                test=inp["test"],
                statistic="normal",
            )
            assert abs(result_normal - expected["normal"]) <= 2, \
                f"normal sample size mismatch for {inp}: got {result_normal}, expected {expected['normal']}"

    def test_dprime_sample_size_matches_sensr(self, golden_sample_size_data):
        """Test dprime_sample_size matches sensR d.primeSS."""
        if golden_sample_size_data is None:
            pytest.skip("Golden data not available")

        for case in golden_sample_size_data["dprime_sample_size"]:
            inp = case["input"]
            expected = case["sample_size"]

            result_exact = dprime_sample_size(
                d_prime_a=inp["d_prime_a"],
                method=inp["method"],
                d_prime_0=inp["d_prime_0"],
                target_power=inp["target_power"],
                test=inp["test"],
                statistic="exact",
            )
            assert abs(result_exact - expected["exact"]) <= 2, \
                f"exact sample size mismatch for {inp}: got {result_exact}, expected {expected['exact']}"
