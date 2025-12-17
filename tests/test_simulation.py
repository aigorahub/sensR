"""Tests for simulation functions."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from senspy.simulation import discrim_sim, samediff_sim, SameDiffSimResult


class TestDiscrimSim:
    """Tests for discrim_sim function."""

    def test_basic_simulation(self):
        """Test basic simulation output."""
        result = discrim_sim(
            sample_size=10, replicates=3, d_prime=2,
            method="triangle", random_state=42
        )
        assert result.shape == (10,)
        assert result.dtype == np.int_
        assert np.all(result >= 0)
        assert np.all(result <= 3)

    def test_reproducibility(self):
        """Test that random_state produces reproducible results."""
        result1 = discrim_sim(
            sample_size=10, replicates=5, d_prime=1,
            method="twoAFC", random_state=123
        )
        result2 = discrim_sim(
            sample_size=10, replicates=5, d_prime=1,
            method="twoAFC", random_state=123
        )
        assert_array_equal(result1, result2)

    def test_d_prime_zero(self):
        """At d'=0, results should cluster around guessing."""
        result = discrim_sim(
            sample_size=1000, replicates=100, d_prime=0,
            method="triangle", random_state=42
        )
        # For triangle, guessing = 1/3 ≈ 33.3%
        mean_correct = result.mean()
        assert 30 < mean_correct < 40  # Should be around 33.3

    def test_high_d_prime(self):
        """High d' should produce near-perfect performance."""
        result = discrim_sim(
            sample_size=100, replicates=10, d_prime=5,
            method="twoAFC", random_state=42
        )
        # With d'=5, performance should be very high
        assert result.mean() > 9.5

    def test_individual_variability(self):
        """Test that sd_indiv increases variability."""
        result_no_var = discrim_sim(
            sample_size=100, replicates=10, d_prime=2,
            method="triangle", sd_indiv=0, random_state=42
        )
        result_with_var = discrim_sim(
            sample_size=100, replicates=10, d_prime=2,
            method="triangle", sd_indiv=2, random_state=42
        )
        # With individual variability, variance should be higher
        var_no = result_no_var.var()
        var_with = result_with_var.var()
        assert var_with > var_no

    @pytest.mark.parametrize("method", [
        "duotrio", "triangle", "twoAFC", "threeAFC", "tetrad", "hexad", "twofive", "twofiveF"
    ])
    def test_all_methods(self, method):
        """Test that all methods work."""
        result = discrim_sim(
            sample_size=5, replicates=3, d_prime=1,
            method=method, random_state=42
        )
        assert result.shape == (5,)
        assert np.all(result >= 0)
        assert np.all(result <= 3)

    def test_double_protocol(self):
        """Test double protocol simulation."""
        result = discrim_sim(
            sample_size=10, replicates=5, d_prime=2,
            method="triangle", double=True, random_state=42
        )
        assert result.shape == (10,)
        assert np.all(result >= 0)
        assert np.all(result <= 5)

    def test_double_lower_probability(self):
        """Double protocol should have lower correct rate than single."""
        result_single = discrim_sim(
            sample_size=1000, replicates=20, d_prime=1,
            method="twoAFC", double=False, random_state=42
        )
        result_double = discrim_sim(
            sample_size=1000, replicates=20, d_prime=1,
            method="twoAFC", double=True, random_state=42
        )
        # Double should have lower mean correct
        assert result_double.mean() < result_single.mean()

    def test_invalid_sample_size(self):
        """Test that invalid sample_size raises ValueError."""
        with pytest.raises(ValueError, match="sample_size"):
            discrim_sim(sample_size=0, replicates=3, d_prime=1)
        with pytest.raises(ValueError, match="sample_size"):
            discrim_sim(sample_size=-1, replicates=3, d_prime=1)

    def test_invalid_replicates(self):
        """Test that invalid replicates raises ValueError."""
        with pytest.raises(ValueError, match="replicates"):
            discrim_sim(sample_size=10, replicates=-1, d_prime=1)

    def test_invalid_d_prime(self):
        """Test that negative d_prime raises ValueError."""
        with pytest.raises(ValueError, match="d_prime"):
            discrim_sim(sample_size=10, replicates=3, d_prime=-1)

    def test_invalid_sd_indiv(self):
        """Test that negative sd_indiv raises ValueError."""
        with pytest.raises(ValueError, match="sd_indiv"):
            discrim_sim(sample_size=10, replicates=3, d_prime=1, sd_indiv=-0.5)

    def test_invalid_method(self):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Invalid method"):
            discrim_sim(sample_size=10, replicates=3, d_prime=1, method="invalid")

    def test_double_not_implemented(self):
        """Test that double for certain methods raises ValueError."""
        for method in ["hexad", "twofive", "twofiveF"]:
            with pytest.raises(ValueError, match="double"):
                discrim_sim(sample_size=10, replicates=3, d_prime=1,
                           method=method, double=True)


class TestSameDiffSim:
    """Tests for samediff_sim function."""

    def test_basic_simulation(self):
        """Test basic simulation output."""
        result = samediff_sim(n=10, tau=1, delta=1, Ns=10, Nd=10, random_state=42)
        assert isinstance(result, SameDiffSimResult)
        assert result.ss.shape == (10,)
        assert result.ds.shape == (10,)
        assert result.sd.shape == (10,)
        assert result.dd.shape == (10,)

    def test_reproducibility(self):
        """Test that random_state produces reproducible results."""
        result1 = samediff_sim(n=10, tau=1, delta=1, Ns=10, Nd=10, random_state=123)
        result2 = samediff_sim(n=10, tau=1, delta=1, Ns=10, Nd=10, random_state=123)
        assert_array_equal(result1.ss, result2.ss)
        assert_array_equal(result1.ds, result2.ds)
        assert_array_equal(result1.sd, result2.sd)
        assert_array_equal(result1.dd, result2.dd)

    def test_sums_to_n(self):
        """Test that ss + ds = Ns and sd + dd = Nd."""
        Ns, Nd = 15, 20
        result = samediff_sim(n=100, tau=1, delta=1, Ns=Ns, Nd=Nd, random_state=42)
        assert np.all(result.ss + result.ds == Ns)
        assert np.all(result.sd + result.dd == Nd)

    def test_to_array(self):
        """Test to_array method."""
        result = samediff_sim(n=5, tau=1, delta=1, Ns=10, Nd=10, random_state=42)
        arr = result.to_array()
        assert arr.shape == (5, 4)
        assert_array_equal(arr[:, 0], result.ss)
        assert_array_equal(arr[:, 1], result.ds)
        assert_array_equal(arr[:, 2], result.sd)
        assert_array_equal(arr[:, 3], result.dd)

    def test_delta_zero(self):
        """With delta=0, same and different pairs should behave similarly."""
        result = samediff_sim(n=1000, tau=1, delta=0, Ns=100, Nd=100, random_state=42)
        # With delta=0, P(same|same) should equal P(same|different)
        p_same_same = result.ss.mean() / 100
        p_same_diff = result.sd.mean() / 100
        assert abs(p_same_same - p_same_diff) < 0.1

    def test_high_delta(self):
        """With high delta, discriminability should be clear."""
        result = samediff_sim(n=1000, tau=1, delta=5, Ns=100, Nd=100, random_state=42)
        # With high delta, P(same|different) should be low
        p_same_diff = result.sd.mean() / 100
        assert p_same_diff < 0.2

    def test_tau_effect(self):
        """Higher tau should increase P(same|same)."""
        result_low_tau = samediff_sim(n=1000, tau=0.5, delta=1, Ns=100, Nd=100, random_state=42)
        result_high_tau = samediff_sim(n=1000, tau=2.0, delta=1, Ns=100, Nd=100, random_state=42)
        p_same_same_low = result_low_tau.ss.mean() / 100
        p_same_same_high = result_high_tau.ss.mean() / 100
        assert p_same_same_high > p_same_same_low


class TestSimulationRegressionR:
    """Regression tests against R sensR values."""

    def test_discrimsim_r_values(self):
        """Test against R sensR discrimSim results.

        R code:
        set.seed(1)
        a <- discrimSim(sample.size=10, replicates=3, d.prime=2,
                        method="triangle", sd.indiv=1)
        # [1] 3 3 2 3 3 1 3 3 1 2
        """
        # Note: NumPy RNG is different from R, so we test properties
        # rather than exact values
        result = discrim_sim(
            sample_size=10, replicates=3, d_prime=2,
            method="triangle", sd_indiv=1, random_state=1
        )
        assert result.shape == (10,)
        assert np.all(result >= 0)
        assert np.all(result <= 3)
