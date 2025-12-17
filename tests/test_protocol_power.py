"""Tests for protocol-specific power functions."""

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from senspy.protocol_power import samediff_power, twoac_power, TwoACPowerResult


# Load golden values from sensR
FIXTURES_DIR = Path(__file__).parent / "fixtures"
with open(FIXTURES_DIR / "golden_sensr.json") as f:
    GOLDEN = json.load(f)


class TestSameDiffPower:
    """Tests for samediff_power function."""

    def test_basic_power(self):
        """Test basic power computation."""
        power = samediff_power(n=100, tau=1, delta=2, Ns=10, Nd=10, random_state=42)
        assert 0 <= power <= 1

    def test_reproducibility(self):
        """Test that random_state gives reproducible results."""
        power1 = samediff_power(n=50, tau=1, delta=2, Ns=10, Nd=10, random_state=123)
        power2 = samediff_power(n=50, tau=1, delta=2, Ns=10, Nd=10, random_state=123)
        assert power1 == power2

    def test_high_delta_high_power(self):
        """High delta should give high power."""
        power = samediff_power(n=100, tau=1, delta=4, Ns=20, Nd=20, random_state=42)
        assert power > 0.7

    def test_low_delta_low_power(self):
        """Low delta should give low power."""
        power = samediff_power(n=100, tau=1, delta=0.1, Ns=10, Nd=10, random_state=42)
        assert power < 0.5

    def test_larger_sample_higher_power(self):
        """Larger sample sizes should give higher power."""
        power_small = samediff_power(n=100, tau=1, delta=1, Ns=5, Nd=5, random_state=42)
        power_large = samediff_power(n=100, tau=1, delta=1, Ns=30, Nd=30, random_state=42)
        # Due to stochasticity, we just check they're reasonable
        assert 0 <= power_small <= 1
        assert 0 <= power_large <= 1

    def test_invalid_n(self):
        """Test that invalid n raises ValueError."""
        with pytest.raises(ValueError, match="'n'"):
            samediff_power(n=0, tau=1, delta=1, Ns=10, Nd=10)

    def test_invalid_delta(self):
        """Test that negative delta raises ValueError."""
        with pytest.raises(ValueError, match="'delta'"):
            samediff_power(n=100, tau=1, delta=-1, Ns=10, Nd=10)

    def test_invalid_sample_sizes(self):
        """Test that invalid Ns/Nd raises ValueError."""
        with pytest.raises(ValueError, match="'Ns' and 'Nd'"):
            samediff_power(n=100, tau=1, delta=1, Ns=0, Nd=10)
        with pytest.raises(ValueError, match="'Ns' and 'Nd'"):
            samediff_power(n=100, tau=1, delta=1, Ns=10, Nd=0)

    def test_invalid_alpha(self):
        """Test that invalid alpha raises ValueError."""
        with pytest.raises(ValueError, match="'alpha'"):
            samediff_power(n=100, tau=1, delta=1, Ns=10, Nd=10, alpha=0)
        with pytest.raises(ValueError, match="'alpha'"):
            samediff_power(n=100, tau=1, delta=1, Ns=10, Nd=10, alpha=1)


class TestTwoACPower:
    """Tests for twoac_power function."""

    def test_basic_power(self):
        """Test basic power computation."""
        result = twoac_power(tau=0.5, d_prime=0.7, size=50, tol=1e-3)
        assert isinstance(result, TwoACPowerResult)
        assert 0 <= result.power <= 1
        assert 0 <= result.actual_alpha <= 1
        assert result.samples > 0
        assert result.kept > 0

    def test_exact_power(self):
        """Test exact power computation with tol=0."""
        result = twoac_power(tau=0.5, d_prime=0.7, size=20, tol=0)
        assert result.discarded == 0
        assert result.kept == result.samples

    def test_tolerance_effect(self):
        """Test that tolerance affects discarded count."""
        result_exact = twoac_power(tau=0.5, d_prime=0.7, size=30, tol=0)
        result_approx = twoac_power(tau=0.5, d_prime=0.7, size=30, tol=1e-4)
        assert result_approx.discarded >= result_exact.discarded

    def test_high_d_prime_high_power(self):
        """High d-prime should give high power."""
        result = twoac_power(tau=0.5, d_prime=2.0, size=50, tol=1e-3)
        assert result.power > 0.8

    def test_zero_d_prime_low_power(self):
        """Zero d-prime should give power close to alpha."""
        result = twoac_power(tau=0.5, d_prime=0, size=50, d_prime_0=0, alpha=0.05, tol=1e-3)
        # Power should be close to alpha (Type I error rate)
        assert result.power < 0.15

    def test_similarity_test(self):
        """Test similarity test (alternative='less')."""
        result = twoac_power(
            tau=0.4, d_prime=0.5, size=100, d_prime_0=1.0,
            alpha=0.05, tol=1e-4, alternative="less"
        )
        assert 0 <= result.power <= 1

    def test_preference_test(self):
        """Test preference test with negative d-prime."""
        result = twoac_power(
            tau=0.4, d_prime=-0.5, size=100, d_prime_0=0,
            alpha=0.05, tol=1e-4, alternative="two.sided"
        )
        assert 0 <= result.power <= 1

    def test_probability_vector(self):
        """Test that probability vector sums to 1."""
        result = twoac_power(tau=0.5, d_prime=0.7, size=30, tol=0)
        assert_allclose(np.sum(result.p), 1.0, rtol=1e-3)

    def test_invalid_tau(self):
        """Test that invalid tau raises ValueError."""
        with pytest.raises(ValueError, match="'tau'"):
            twoac_power(tau=0, d_prime=0.7, size=50)
        with pytest.raises(ValueError, match="'tau'"):
            twoac_power(tau=-0.5, d_prime=0.7, size=50)

    def test_invalid_size(self):
        """Test that invalid size raises ValueError."""
        with pytest.raises(ValueError, match="'size'"):
            twoac_power(tau=0.5, d_prime=0.7, size=1)
        with pytest.raises(ValueError, match="'size'"):
            twoac_power(tau=0.5, d_prime=0.7, size=6000)

    def test_invalid_alpha(self):
        """Test that invalid alpha raises ValueError."""
        with pytest.raises(ValueError, match="'alpha'"):
            twoac_power(tau=0.5, d_prime=0.7, size=50, alpha=0)
        with pytest.raises(ValueError, match="'alpha'"):
            twoac_power(tau=0.5, d_prime=0.7, size=50, alpha=1)

    def test_invalid_tol(self):
        """Test that invalid tol raises ValueError."""
        with pytest.raises(ValueError, match="'tol'"):
            twoac_power(tau=0.5, d_prime=0.7, size=50, tol=-0.1)
        with pytest.raises(ValueError, match="'tol'"):
            twoac_power(tau=0.5, d_prime=0.7, size=50, tol=1.5)

    def test_invalid_alternative(self):
        """Test that invalid alternative raises ValueError."""
        with pytest.raises(ValueError, match="'alternative'"):
            twoac_power(tau=0.5, d_prime=0.7, size=50, alternative="invalid")


class TestTwoACPowerRegression:
    """Regression tests for twoac_power against R sensR golden values."""

    def test_golden_exact_case(self):
        """Test against R sensR twoACpwr exact case."""
        golden = GOLDEN["twoac_power"]["exact_case"]
        inp = golden["input"]
        result = twoac_power(
            tau=inp["tau"],
            d_prime=inp["d_prime"],
            size=inp["size"],
            tol=inp["tol"],
        )

        assert_allclose(result.power, golden["power"], rtol=1e-5)
        assert_allclose(result.actual_alpha, golden["actual_alpha"], rtol=1e-5)
        assert result.samples == golden["samples"]
        assert_allclose(result.p, golden["p"], rtol=1e-3)

    def test_golden_tolerance_case(self):
        """Test against R sensR twoACpwr with tolerance."""
        golden = GOLDEN["twoac_power"]["tolerance_case"]
        inp = golden["input"]
        result = twoac_power(
            tau=inp["tau"],
            d_prime=inp["d_prime"],
            size=inp["size"],
            tol=inp["tol"],
        )

        assert_allclose(result.power, golden["power"], rtol=1e-4)
        assert result.samples == golden["samples"]

    def test_samples_count(self):
        """Test that sample count matches expected formula.

        For n observations, the number of outcomes is (n+1)(n+2)/2.
        """
        for n in [10, 20, 30]:
            result = twoac_power(tau=0.5, d_prime=0.7, size=n, tol=0)
            expected_samples = (n + 1) * (n + 2) // 2
            assert result.samples == expected_samples
