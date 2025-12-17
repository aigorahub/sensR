"""Tests for DOD (Degree of Difference) model."""

import numpy as np
import pytest

from senspy.dod import (
    DODControl,
    DODFitResult,
    DODResult,
    _dod_nll_internal,
    _dod_null_tau_internal,
    _init_tau,
    _par2prob_dod_internal,
    dod,
    dod_fit,
    dod_sim,
    optimal_tau,
    par2prob_dod,
)


class TestPar2ProbDOD:
    """Tests for par2prob_dod function."""

    def test_basic_computation(self):
        """Test basic probability computation."""
        tau = np.array([1.0, 2.0, 3.0])
        d_prime = 1.0
        prob = par2prob_dod(tau, d_prime)

        # Should return 2 x 4 matrix
        assert prob.shape == (2, 4)
        # Rows should sum to 1
        assert np.allclose(prob.sum(axis=1), [1.0, 1.0])
        # All probabilities should be positive
        assert np.all(prob > 0)

    def test_d_prime_zero(self):
        """Test that d_prime=0 gives same probabilities for both rows."""
        tau = np.array([1.0, 2.0, 3.0])
        prob = par2prob_dod(tau, 0.0)

        # With d_prime=0, same-pairs and diff-pairs should have same distribution
        assert np.allclose(prob[0, :], prob[1, :])

    def test_invalid_d_prime(self):
        """Test that negative d_prime raises error."""
        tau = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="d_prime must be non-negative"):
            par2prob_dod(tau, -1.0)

    def test_invalid_tau_not_positive(self):
        """Test that non-positive tau raises error."""
        with pytest.raises(ValueError, match="tau values must be positive"):
            par2prob_dod(np.array([0.0, 1.0, 2.0]), 1.0)
        with pytest.raises(ValueError, match="tau values must be positive"):
            par2prob_dod(np.array([-1.0, 1.0, 2.0]), 1.0)

    def test_invalid_tau_not_increasing(self):
        """Test that non-increasing tau raises error."""
        with pytest.raises(ValueError, match="tau values must be strictly increasing"):
            par2prob_dod(np.array([1.0, 1.0, 2.0]), 1.0)
        with pytest.raises(ValueError, match="tau values must be strictly increasing"):
            par2prob_dod(np.array([2.0, 1.0, 3.0]), 1.0)


class TestDODNLL:
    """Tests for DOD negative log-likelihood."""

    def test_basic_nll(self):
        """Test basic NLL computation."""
        tau = np.array([1.0, 2.0, 3.0])
        d_prime = 1.0
        same = np.array([20, 30, 30, 20])
        diff = np.array([10, 20, 35, 35])

        nll = _dod_nll_internal(tau, d_prime, same, diff)

        assert np.isfinite(nll)
        assert nll > 0

    def test_nll_minimum_at_mle(self):
        """Test that NLL is minimized at MLE."""
        tau = np.array([0.5, 1.0, 1.5])
        d_prime_true = 1.0

        # Generate data from model
        np.random.seed(42)
        data = dod_sim(d_prime=d_prime_true, tau=tau, method_tau="user_defined")

        # Fit model
        fit = dod_fit(data[0, :], data[1, :])

        # NLL at MLE should be lower than at true values
        nll_mle = _dod_nll_internal(fit.tau, fit.d_prime, data[0, :], data[1, :])
        nll_true = _dod_nll_internal(tau, d_prime_true, data[0, :], data[1, :])

        # MLE should have lower or equal NLL
        assert nll_mle <= nll_true + 1e-6


class TestDODNullTau:
    """Tests for DOD null model tau computation."""

    def test_null_tau_symmetric_data(self):
        """Test null tau with symmetric data."""
        same = np.array([25, 25, 25, 25])
        diff = np.array([25, 25, 25, 25])

        tau = _dod_null_tau_internal(same, diff)

        # Should have 3 tau values for 4 categories
        assert len(tau) == 3
        # Should be positive and increasing
        assert np.all(tau > 0)
        assert np.all(np.diff(tau) > 0)

    def test_null_tau_skewed_data(self):
        """Test null tau with skewed data."""
        same = np.array([10, 20, 30, 40])
        diff = np.array([10, 20, 30, 40])

        tau = _dod_null_tau_internal(same, diff)

        assert len(tau) == 3
        assert np.all(tau > 0)


class TestDODFit:
    """Tests for dod_fit function."""

    def test_basic_fit(self):
        """Test basic model fitting."""
        same = np.array([20, 30, 25, 25])
        diff = np.array([10, 20, 35, 35])

        result = dod_fit(same, diff)

        assert isinstance(result, DODFitResult)
        assert result.d_prime >= 0
        assert len(result.tau) == 3
        assert np.isfinite(result.log_lik)
        assert result.convergence == 0

    def test_fit_with_fixed_d_prime(self):
        """Test fitting with fixed d_prime."""
        same = np.array([20, 30, 25, 25])
        diff = np.array([10, 20, 35, 35])

        result = dod_fit(same, diff, d_prime=1.0)

        assert np.isclose(result.d_prime, 1.0)
        assert len(result.tau) == 3

    def test_fit_with_fixed_tau(self):
        """Test fitting with fixed tau."""
        same = np.array([20, 30, 25, 25])
        diff = np.array([10, 20, 35, 35])
        tau = np.array([0.5, 1.0, 1.5])

        result = dod_fit(same, diff, tau=tau)

        assert np.allclose(result.tau, tau)
        assert result.d_prime >= 0

    def test_fit_d_prime_zero(self):
        """Test fitting under null hypothesis (d_prime=0)."""
        same = np.array([25, 25, 25, 25])
        diff = np.array([25, 25, 25, 25])

        result = dod_fit(same, diff, d_prime=0.0)

        assert np.isclose(result.d_prime, 0.0)
        assert len(result.tau) == 3

    def test_fit_returns_vcov(self):
        """Test that fit returns variance-covariance matrix."""
        same = np.array([20, 30, 25, 25])
        diff = np.array([10, 20, 35, 35])

        result = dod_fit(same, diff)

        assert result.vcov is not None
        assert result.vcov.shape == (4, 4)  # 3 tau + 1 d_prime
        # vcov should be symmetric
        assert np.allclose(result.vcov, result.vcov.T)


class TestDOD:
    """Tests for main dod function."""

    def test_basic_dod(self):
        """Test basic DOD analysis."""
        same = np.array([20, 30, 25, 25])
        diff = np.array([10, 20, 35, 35])

        result = dod(same, diff)

        assert isinstance(result, DODResult)
        assert result.d_prime >= 0
        assert len(result.tau) == 3
        assert 0 <= result.p_value <= 1
        assert result.conf_level == 0.95
        assert result.alternative == "greater"

    def test_dod_likelihood_statistic(self):
        """Test DOD with likelihood statistic."""
        same = np.array([20, 30, 25, 25])
        diff = np.array([10, 20, 35, 35])

        result = dod(same, diff, statistic="likelihood")

        assert result.statistic == "likelihood"
        assert np.isfinite(result.stat_value)
        assert 0 <= result.p_value <= 1

    def test_dod_wald_statistic(self):
        """Test DOD with Wald statistic."""
        same = np.array([20, 30, 25, 25])
        diff = np.array([10, 20, 35, 35])

        result = dod(same, diff, statistic="Wald")

        assert result.statistic == "Wald"
        assert result.conf_method == "Wald"

    def test_dod_pearson_statistic(self):
        """Test DOD with Pearson statistic."""
        same = np.array([20, 30, 25, 25])
        diff = np.array([10, 20, 35, 35])

        result = dod(same, diff, statistic="Pearson")

        assert result.statistic == "Pearson"
        assert np.isfinite(result.stat_value)

    def test_dod_wilcoxon_statistic(self):
        """Test DOD with Wilcoxon statistic."""
        same = np.array([20, 30, 25, 25])
        diff = np.array([10, 20, 35, 35])

        result = dod(same, diff, statistic="Wilcoxon")

        assert result.statistic == "Wilcoxon"
        assert 0 <= result.p_value <= 1

    def test_dod_wilcoxon_requires_d_prime0_zero(self):
        """Test that Wilcoxon requires d_prime0=0."""
        same = np.array([20, 30, 25, 25])
        diff = np.array([10, 20, 35, 35])

        with pytest.raises(ValueError, match="Wilcoxon.*d_prime0 = 0"):
            dod(same, diff, statistic="Wilcoxon", d_prime0=1.0)

    def test_dod_alternative_difference(self):
        """Test DOD with difference alternative."""
        same = np.array([20, 30, 25, 25])
        diff = np.array([10, 20, 35, 35])

        result = dod(same, diff, alternative="difference")

        assert result.alternative == "greater"

    def test_dod_alternative_similarity(self):
        """Test DOD with similarity alternative."""
        same = np.array([20, 30, 25, 25])
        diff = np.array([10, 20, 35, 35])

        result = dod(same, diff, alternative="similarity", d_prime0=2.0)

        assert result.alternative == "less"

    def test_dod_alternative_two_sided(self):
        """Test DOD with two-sided alternative."""
        same = np.array([20, 30, 25, 25])
        diff = np.array([10, 20, 35, 35])

        result = dod(same, diff, alternative="two.sided", d_prime0=1.0)

        assert result.alternative == "two.sided"

    def test_dod_d_prime0_zero_requires_greater_alternative(self):
        """Test that d_prime0=0 requires greater/difference alternative."""
        same = np.array([20, 30, 25, 25])
        diff = np.array([10, 20, 35, 35])

        with pytest.raises(ValueError, match="'alternative' has to be 'difference'"):
            dod(same, diff, alternative="two.sided", d_prime0=0.0)

    def test_dod_confidence_interval(self):
        """Test DOD confidence interval."""
        same = np.array([20, 30, 25, 25])
        diff = np.array([10, 20, 35, 35])

        result = dod(same, diff, conf_level=0.90)

        assert result.conf_level == 0.90
        assert result.conf_int[0] >= 0
        assert result.conf_int[1] > result.conf_int[0]


class TestDODControl:
    """Tests for DODControl."""

    def test_default_control(self):
        """Test default control parameters."""
        ctrl = DODControl()

        assert ctrl.grad_tol == 1e-4
        assert ctrl.integer_tol == 1e-8
        assert ctrl.get_vcov is True
        assert ctrl.get_grad is True
        assert ctrl.test_args is True
        assert ctrl.do_warn is True

    def test_invalid_grad_tol(self):
        """Test invalid grad_tol."""
        with pytest.raises(ValueError, match="grad_tol must be positive"):
            DODControl(grad_tol=-1)
        with pytest.raises(ValueError, match="grad_tol must be positive"):
            DODControl(grad_tol=0)
        with pytest.raises(ValueError, match="grad_tol must be positive"):
            DODControl(grad_tol=np.inf)

    def test_control_no_vcov(self):
        """Test control with get_vcov=False."""
        same = np.array([20, 30, 25, 25])
        diff = np.array([10, 20, 35, 35])
        ctrl = DODControl(get_vcov=False)

        result = dod_fit(same, diff, control=ctrl)

        assert result.vcov is None


class TestOptimalTau:
    """Tests for optimal_tau function."""

    def test_equi_prob_method(self):
        """Test equi_prob method for optimal tau."""
        result = optimal_tau(d_prime=1.0, ncat=4, method="equi_prob")

        assert "tau" in result
        assert len(result["tau"]) == 3
        assert np.all(result["tau"] > 0)
        assert np.all(np.diff(result["tau"]) > 0)

        # Check that probabilities are roughly equal
        avg_prob = np.sum(result["prob"], axis=0) / 2
        assert np.allclose(avg_prob, 0.25, atol=0.05)

    def test_se_min_method(self):
        """Test se_min method for optimal tau."""
        result = optimal_tau(d_prime=1.0, ncat=4, method="se_min")

        assert "tau" in result
        assert len(result["tau"]) == 3

    def test_lr_max_method(self):
        """Test LR_max method for optimal tau."""
        result = optimal_tau(d_prime=1.0, ncat=4, method="LR_max")

        assert "tau" in result
        assert len(result["tau"]) == 3

    def test_invalid_d_prime(self):
        """Test invalid d_prime."""
        with pytest.raises(ValueError, match="d_prime must be non-negative"):
            optimal_tau(d_prime=-1.0, ncat=4)

    def test_invalid_ncat(self):
        """Test invalid ncat."""
        with pytest.raises(ValueError, match="ncat must be at least 2"):
            optimal_tau(d_prime=1.0, ncat=1)


class TestDODSim:
    """Tests for dod_sim function."""

    def test_basic_simulation(self):
        """Test basic DOD data simulation."""
        np.random.seed(42)
        data = dod_sim(d_prime=1.0, ncat=4, sample_size=100)

        assert data.shape == (2, 4)
        assert data.sum() == 200  # 100 same + 100 diff

    def test_simulation_with_different_sizes(self):
        """Test simulation with different sample sizes."""
        np.random.seed(42)
        data = dod_sim(d_prime=1.0, ncat=4, sample_size=(50, 100))

        assert data[0, :].sum() == 50
        assert data[1, :].sum() == 100

    def test_simulation_user_defined_tau(self):
        """Test simulation with user-defined tau."""
        np.random.seed(42)
        tau = np.array([0.5, 1.0, 1.5])
        data = dod_sim(d_prime=1.0, tau=tau, method_tau="user_defined")

        assert data.shape == (2, 4)

    def test_simulation_reproducibility(self):
        """Test simulation reproducibility with seed."""
        data1 = dod_sim(d_prime=1.0, ncat=4, sample_size=100, random_state=42)
        data2 = dod_sim(d_prime=1.0, ncat=4, sample_size=100, random_state=42)

        assert np.array_equal(data1, data2)


class TestDODValidation:
    """Tests for DOD data validation."""

    def test_mismatched_lengths(self):
        """Test that mismatched same/diff lengths raise error."""
        same = np.array([20, 30, 25])
        diff = np.array([10, 20, 35, 35])

        with pytest.raises(ValueError, match="same and diff must have the same length"):
            dod(same, diff)

    def test_negative_counts(self):
        """Test that negative counts raise error."""
        same = np.array([20, 30, -25, 25])
        diff = np.array([10, 20, 35, 35])

        with pytest.raises(ValueError, match="Counts must be non-negative"):
            dod(same, diff)

    def test_single_category(self):
        """Test that single category raises error."""
        same = np.array([100])
        diff = np.array([100])

        with pytest.raises(ValueError, match="Need at least 2 response categories"):
            dod(same, diff)


class TestDODGoldenValidation:
    """Tests validating DOD against sensR golden data."""

    def test_golden_simple_case(self, golden_dod_data, tolerance):
        """Test DOD against golden data for simple case."""
        if golden_dod_data is None:
            pytest.skip("Golden DOD data not available")

        case = golden_dod_data.get("simple_case")
        if case is None:
            pytest.skip("simple_case not in golden data")

        same = np.array(case["input"]["same"])
        diff = np.array(case["input"]["diff"])

        result = dod(same, diff)

        # Compare d_prime
        assert np.isclose(
            result.d_prime, case["d_prime"], rtol=tolerance["coefficients"]
        )

        # Compare tau values
        expected_tau = np.array(case["tau"])
        assert np.allclose(result.tau, expected_tau, rtol=tolerance["coefficients"])

        # Compare log-likelihood
        assert np.isclose(result.log_lik, case["logLik"], rtol=tolerance["strict"])

        # Compare p-value
        assert np.isclose(result.p_value, case["p_value"], rtol=tolerance["p_values"])

    def test_golden_large_d_prime(self, golden_dod_data, tolerance):
        """Test DOD against golden data for large d-prime case."""
        if golden_dod_data is None:
            pytest.skip("Golden DOD data not available")

        case = golden_dod_data.get("large_d_prime")
        if case is None:
            pytest.skip("large_d_prime not in golden data")

        same = np.array(case["input"]["same"])
        diff = np.array(case["input"]["diff"])

        result = dod(same, diff)

        assert np.isclose(
            result.d_prime, case["d_prime"], rtol=tolerance["coefficients"]
        )

    def test_golden_wald_case(self, golden_dod_data, tolerance):
        """Test DOD Wald statistic against golden data."""
        if golden_dod_data is None:
            pytest.skip("Golden DOD data not available")

        case = golden_dod_data.get("wald_case")
        if case is None:
            pytest.skip("wald_case not in golden data")

        same = np.array(case["input"]["same"])
        diff = np.array(case["input"]["diff"])

        result = dod(same, diff, statistic="Wald")

        assert np.isclose(
            result.d_prime, case["d_prime"], rtol=tolerance["coefficients"]
        )
        assert np.isclose(
            result.stat_value, case["stat_value"], rtol=tolerance["coefficients"]
        )


class TestDODPower:
    """Tests for dod_power function."""

    def test_basic_power(self):
        """Test basic power computation."""
        from senspy.dod import dod_power

        result = dod_power(d_primeA=1.0, sample_size=100, nsim=50, random_state=42)

        assert hasattr(result, 'power')
        assert hasattr(result, 'se_power')
        assert hasattr(result, 'n_used')
        assert 0 <= result.power <= 1
        assert result.d_primeA == 1.0
        assert result.sample_size == (100, 100)

    def test_power_with_different_statistics(self):
        """Test power with different test statistics."""
        from senspy.dod import dod_power

        for stat in ['likelihood', 'Wilcoxon']:
            result = dod_power(
                d_primeA=1.0,
                sample_size=50,
                nsim=20,
                statistic=stat,
                random_state=42
            )
            assert hasattr(result, 'power')

    def test_power_invalid_alternative(self):
        """Test that invalid d_primeA/alternative combination raises error."""
        from senspy.dod import dod_power

        with pytest.raises(ValueError, match='Need d_primeA'):
            dod_power(d_primeA=0.5, d_prime0=1.0, alternative='difference')

    def test_power_wilcoxon_requires_dprime0_zero(self):
        """Test that Wilcoxon requires d_prime0=0."""
        from senspy.dod import dod_power

        with pytest.raises(ValueError, match='Wilcoxon'):
            dod_power(d_primeA=1.0, d_prime0=0.5, statistic='Wilcoxon')

    def test_power_user_defined_tau(self):
        """Test power with user-defined tau."""
        from senspy.dod import dod_power

        tau = np.array([0.5, 1.0, 1.5])
        result = dod_power(
            d_primeA=1.0,
            method_tau='user_defined',
            tau=tau,
            nsim=20,
            random_state=42
        )
        assert np.array_equal(result.tau, tau)

    def test_power_reproducibility(self):
        """Test power reproducibility with seed."""
        from senspy.dod import dod_power

        result1 = dod_power(d_primeA=1.0, sample_size=50, nsim=30, random_state=42)
        result2 = dod_power(d_primeA=1.0, sample_size=50, nsim=30, random_state=42)

        assert result1.power == result2.power

