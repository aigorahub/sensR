"""Tests for senspy.betabin module - beta-binomial models."""

import numpy as np
import pytest

from senspy import betabin, BetaBinomialResult, BetaBinomialSummary
from senspy.core.types import Protocol


# Test data from sensR documentation
TEST_X = np.array([3, 2, 6, 8, 3, 4, 6, 0, 9, 9, 0, 2, 1, 2, 8, 9, 5, 7])
TEST_N = np.array([10, 9, 8, 9, 8, 6, 9, 10, 10, 10, 9, 9, 10, 10, 10, 10, 9, 10])
TEST_DATA = np.column_stack([TEST_X, TEST_N])


class TestBetabinBasic:
    """Basic tests for betabin function."""

    def test_corrected_model_fits(self):
        """Test that chance-corrected model fits without error."""
        result = betabin(TEST_DATA, method="duotrio", corrected=True)
        assert isinstance(result, BetaBinomialResult)
        assert 0 < result.mu < 1
        assert 0 < result.gamma < 1

    def test_uncorrected_model_fits(self):
        """Test that standard model fits without error."""
        result = betabin(TEST_DATA, method="duotrio", corrected=False)
        assert isinstance(result, BetaBinomialResult)
        assert 0 < result.mu < 1
        assert 0 < result.gamma < 1

    def test_different_methods(self):
        """Test that different discrimination methods work."""
        for method in ["triangle", "duotrio", "twoafc", "threeafc", "tetrad"]:
            result = betabin(TEST_DATA, method=method, corrected=True)
            assert isinstance(result, BetaBinomialResult)
            assert 0 < result.mu < 1

    def test_protocol_enum_accepted(self):
        """Test that Protocol enum is accepted."""
        result = betabin(TEST_DATA, method=Protocol.TRIANGLE, corrected=True)
        assert isinstance(result, BetaBinomialResult)

    def test_returns_vcov_by_default(self):
        """Test that variance-covariance matrix is computed by default."""
        result = betabin(TEST_DATA, method="duotrio", corrected=True)
        assert result.vcov is not None
        assert result.vcov.shape == (2, 2)

    def test_vcov_false_returns_none(self):
        """Test that vcov=False returns None for vcov."""
        result = betabin(TEST_DATA, method="duotrio", corrected=True, vcov=False)
        assert result.vcov is None

    def test_standard_errors_available(self):
        """Test that standard errors are available when vcov is computed."""
        result = betabin(TEST_DATA, method="duotrio", corrected=True)
        se = result.se()
        assert se is not None
        assert "mu" in se
        assert "gamma" in se
        assert se["mu"] > 0
        assert se["gamma"] > 0


class TestBetabinValidation:
    """Input validation tests."""

    def test_data_must_have_two_columns(self):
        """Test that data with wrong number of columns raises error."""
        with pytest.raises(ValueError, match="2 columns"):
            betabin(np.array([[1, 2, 3], [4, 5, 6]]), method="duotrio")

    def test_data_must_have_at_least_three_rows(self):
        """Test that data with too few rows raises error."""
        with pytest.raises(ValueError, match="at least 3 rows"):
            betabin(np.array([[5, 10], [6, 10]]), method="duotrio")

    def test_successes_must_not_exceed_trials(self):
        """Test that x > n raises error."""
        bad_data = np.array([[5, 10], [15, 10], [6, 10], [7, 10]])
        with pytest.raises(ValueError, match="between 0 and trials"):
            betabin(bad_data, method="duotrio")

    def test_invalid_start_values_raises(self):
        """Test that start values outside (0,1) raise error."""
        with pytest.raises(ValueError, match="open interval"):
            betabin(TEST_DATA, method="duotrio", start=(0.0, 0.5))


class TestBetabinSensRValidation:
    """Validation tests comparing against sensR golden values."""

    def test_corrected_duotrio_coefficients(self):
        """Test corrected duotrio model matches sensR coefficients."""
        result = betabin(TEST_DATA, method="duotrio", corrected=True)

        # Expected from sensR: mu = 0.176050917183022, gamma = 0.504272625540766
        assert result.mu == pytest.approx(0.176051, rel=1e-4)
        assert result.gamma == pytest.approx(0.504273, rel=1e-4)

    def test_corrected_duotrio_summary_values(self):
        """Test corrected duotrio summary matches sensR."""
        result = betabin(TEST_DATA, method="duotrio", corrected=True)
        summary = result.summary()

        # Expected from sensR test
        assert summary.estimates["pc"] == pytest.approx(0.588025, rel=1e-3)
        assert summary.estimates["pd"] == pytest.approx(0.176051, rel=1e-3)
        assert summary.estimates["d_prime"] == pytest.approx(1.0372, rel=1e-3)

    def test_uncorrected_duotrio_coefficients(self):
        """Test un-corrected duotrio model matches sensR coefficients."""
        result = betabin(TEST_DATA, method="duotrio", corrected=False)

        # Expected from sensR: mu = 0.493812345858779, gamma = 0.31442578341548
        assert result.mu == pytest.approx(0.493812, rel=1e-4)
        assert result.gamma == pytest.approx(0.314426, rel=1e-4)

    def test_corrected_triangle_fits(self):
        """Test corrected triangle model fits."""
        # Different test data for triangle (p_guess = 1/3)
        result = betabin(TEST_DATA, method="triangle", corrected=True)
        assert 0 < result.mu < 1
        assert 0 < result.gamma < 1


class TestBetabinLRTests:
    """Tests for likelihood ratio tests."""

    def test_lr_overdispersion_returns_tuple(self):
        """Test LR overdispersion test returns (G^2, p-value)."""
        result = betabin(TEST_DATA, method="duotrio", corrected=True)
        g2, pval = result.lr_overdispersion()
        assert isinstance(g2, float)
        assert isinstance(pval, float)
        assert 0 <= pval <= 1

    def test_lr_association_returns_tuple(self):
        """Test LR association test returns (G^2, p-value)."""
        result = betabin(TEST_DATA, method="duotrio", corrected=True)
        g2, pval = result.lr_association()
        assert isinstance(g2, float)
        assert isinstance(pval, float)
        assert 0 <= pval <= 1

    def test_overdispersion_detected(self):
        """Test that overdispersion is detected in test data."""
        result = betabin(TEST_DATA, method="duotrio", corrected=True)
        _, pval = result.lr_overdispersion()
        # The test data has significant overdispersion
        assert pval < 0.05

    def test_association_detected(self):
        """Test that association is detected in test data."""
        result = betabin(TEST_DATA, method="duotrio", corrected=True)
        _, pval = result.lr_association()
        # The test data has significant association
        assert pval < 0.05


class TestBetabinSummary:
    """Tests for BetaBinomialSummary."""

    def test_summary_has_all_parameters(self):
        """Test that summary includes all parameter estimates."""
        result = betabin(TEST_DATA, method="duotrio", corrected=True)
        summary = result.summary()

        for param in ["mu", "gamma", "pc", "pd", "d_prime"]:
            assert param in summary.estimates
            assert param in summary.std_errors
            assert param in summary.ci_lower
            assert param in summary.ci_upper

    def test_summary_ci_contains_estimate(self):
        """Test that CI contains the point estimate."""
        result = betabin(TEST_DATA, method="duotrio", corrected=True)
        summary = result.summary()

        for param in ["mu", "gamma"]:
            est = summary.estimates[param]
            lower = summary.ci_lower[param]
            upper = summary.ci_upper[param]
            if lower is not None and upper is not None:
                assert lower <= est <= upper

    def test_summary_different_levels(self):
        """Test that different confidence levels produce different CIs."""
        result = betabin(TEST_DATA, method="duotrio", corrected=True)
        summary_90 = result.summary(level=0.90)
        summary_95 = result.summary(level=0.95)
        summary_99 = result.summary(level=0.99)

        # Wider level should give wider CI
        width_90 = summary_90.ci_upper["mu"] - summary_90.ci_lower["mu"]
        width_95 = summary_95.ci_upper["mu"] - summary_95.ci_lower["mu"]
        width_99 = summary_99.ci_upper["mu"] - summary_99.ci_lower["mu"]

        assert width_90 < width_95 < width_99

    def test_summary_str_output(self):
        """Test that summary string output is formatted correctly."""
        result = betabin(TEST_DATA, method="duotrio", corrected=True)
        summary = result.summary()
        output = str(summary)

        assert "Chance-corrected beta-binomial" in output
        assert "duotrio" in output
        assert "mu" in output
        assert "gamma" in output
        assert "log-likelihood" in output
        assert "LR-test" in output


class TestBetabinEdgeCases:
    """Edge case tests."""

    def test_high_success_rate_data(self):
        """Test with data having high success rate."""
        x = np.array([9, 10, 8, 9, 10, 9, 8, 10, 9, 10])
        n = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10])
        data = np.column_stack([x, n])

        result = betabin(data, method="triangle", corrected=True)
        assert result.mu > 0.5  # High discrimination

    def test_low_success_rate_data(self):
        """Test with data having low success rate (near chance)."""
        x = np.array([3, 4, 3, 3, 4, 3, 4, 3, 4, 3])
        n = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10])
        data = np.column_stack([x, n])

        result = betabin(data, method="triangle", corrected=True)
        # Should still fit, mu might be close to 0
        assert 0 <= result.mu <= 1

    def test_dataframe_input(self):
        """Test that DataFrame-like input works."""
        # Use a dict with array-like values (similar to DataFrame behavior)
        import pandas as pd

        df = pd.DataFrame({"successes": TEST_X, "trials": TEST_N})
        result = betabin(df.values, method="duotrio", corrected=True)
        assert isinstance(result, BetaBinomialResult)


class TestBetabinLogLikelihood:
    """Tests for log-likelihood computation."""

    def test_log_likelihood_is_finite(self):
        """Test that log-likelihood is finite."""
        result = betabin(TEST_DATA, method="duotrio", corrected=True)
        assert np.isfinite(result.log_likelihood)

    def test_log_likelihood_null_less_than_fitted(self):
        """Test that null model has lower log-likelihood than fitted."""
        result = betabin(TEST_DATA, method="duotrio", corrected=True)
        assert result.log_lik_null <= result.log_likelihood

    def test_log_likelihood_ordering(self):
        """Test log-likelihood ordering: null <= mu-only <= fitted (usually)."""
        result = betabin(TEST_DATA, method="duotrio", corrected=True)
        # The fitted model should have highest log-likelihood
        assert result.log_lik_null <= result.log_likelihood
        # Note: log_lik_mu might be higher or lower than fitted depending on data


class TestBetabinGoldenValidation:
    """Validation tests comparing against sensR golden data."""

    def test_corrected_duotrio_matches_sensr(self, golden_betabin_data):
        """Test corrected duotrio matches sensR exactly."""
        if golden_betabin_data is None:
            pytest.skip("Golden data not available")

        expected = golden_betabin_data["corrected_duotrio"]
        result = betabin(TEST_DATA, method="duotrio", corrected=True)

        # Check coefficients
        assert result.mu == pytest.approx(expected["coefficients"]["mu"], rel=1e-4)
        assert result.gamma == pytest.approx(expected["coefficients"]["gamma"], rel=1e-4)

        # Check log-likelihood
        assert result.log_likelihood == pytest.approx(expected["log_likelihood"], rel=1e-3)

        # Check summary values
        summary = result.summary()
        assert summary.estimates["pc"] == pytest.approx(expected["summary"]["pc"], rel=1e-3)
        assert summary.estimates["pd"] == pytest.approx(expected["summary"]["pd"], rel=1e-3)
        assert summary.estimates["d_prime"] == pytest.approx(expected["summary"]["d_prime"], rel=1e-3)

    def test_uncorrected_duotrio_matches_sensr(self, golden_betabin_data):
        """Test un-corrected duotrio matches sensR."""
        if golden_betabin_data is None:
            pytest.skip("Golden data not available")

        expected = golden_betabin_data["uncorrected_duotrio"]
        result = betabin(TEST_DATA, method="duotrio", corrected=False)

        assert result.mu == pytest.approx(expected["coefficients"]["mu"], rel=1e-4)
        assert result.gamma == pytest.approx(expected["coefficients"]["gamma"], rel=1e-4)
        assert result.log_likelihood == pytest.approx(expected["log_likelihood"], rel=1e-3)

    def test_corrected_triangle_matches_sensr(self, golden_betabin_data):
        """Test corrected triangle matches sensR."""
        if golden_betabin_data is None:
            pytest.skip("Golden data not available")

        expected = golden_betabin_data["corrected_triangle"]
        result = betabin(TEST_DATA, method="triangle", corrected=True)

        assert result.mu == pytest.approx(expected["coefficients"]["mu"], rel=1e-4)
        assert result.gamma == pytest.approx(expected["coefficients"]["gamma"], rel=1e-4)
