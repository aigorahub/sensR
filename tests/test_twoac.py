"""Tests for senspy.twoac module - 2-AC protocol analysis."""

import numpy as np
import pytest

from senspy import twoac, TwoACResult


class TestTwoACBasic:
    """Basic tests for twoac function."""

    def test_simple_case_returns_result(self):
        """Test that simple case returns TwoACResult."""
        result = twoac([2, 2, 6])
        assert isinstance(result, TwoACResult)

    def test_estimates_are_finite(self):
        """Test that estimates are finite for typical data."""
        result = twoac([2, 2, 6])
        assert np.isfinite(result.tau)
        assert np.isfinite(result.d_prime)

    def test_standard_errors_available(self):
        """Test that SEs are available for typical data."""
        result = twoac([2, 2, 6])
        assert result.se_tau is not None
        assert result.se_d_prime is not None
        assert result.se_tau > 0
        assert result.se_d_prime > 0

    def test_vcov_is_2x2_matrix(self):
        """Test that vcov is 2x2 matrix when available."""
        result = twoac([2, 2, 6])
        assert result.vcov is not None
        assert result.vcov.shape == (2, 2)

    def test_positive_d_prime_when_prefer_b_greater(self):
        """Test that d-prime is positive when prefer_B > prefer_A."""
        result = twoac([2, 2, 6])
        assert result.d_prime > 0

    def test_negative_d_prime_when_prefer_a_greater(self):
        """Test that d-prime is negative when prefer_A > prefer_B."""
        result = twoac([6, 2, 2])
        assert result.d_prime < 0

    def test_log_likelihood_is_finite(self):
        """Test that log-likelihood is finite."""
        result = twoac([2, 2, 6])
        assert np.isfinite(result.log_likelihood)


class TestTwoACStatistics:
    """Tests for different test statistics."""

    def test_likelihood_statistic_default(self):
        """Test that likelihood is the default statistic."""
        result = twoac([2, 2, 6])
        assert result.statistic == "likelihood"

    def test_wald_statistic(self):
        """Test Wald statistic option."""
        result = twoac([2, 2, 6], statistic="wald")
        assert result.statistic == "wald"
        assert result.stat_value is not None
        assert result.p_value is not None

    def test_likelihood_gives_p_value(self):
        """Test that likelihood statistic gives p-value."""
        result = twoac([2, 2, 6], statistic="likelihood")
        assert result.p_value is not None
        assert 0 <= result.p_value <= 1

    def test_wald_gives_p_value(self):
        """Test that Wald statistic gives p-value."""
        result = twoac([2, 2, 6], statistic="wald")
        assert result.p_value is not None
        assert 0 <= result.p_value <= 1


class TestTwoACAlternatives:
    """Tests for different alternative hypotheses."""

    def test_two_sided_alternative(self):
        """Test two-sided alternative."""
        result = twoac([2, 2, 6], alternative="two.sided")
        assert result.alternative == "two.sided"

    def test_greater_alternative(self):
        """Test greater alternative."""
        result = twoac([2, 2, 6], alternative="greater")
        assert result.alternative == "greater"
        # p-value should be smaller for one-sided when d' > 0
        result_two = twoac([2, 2, 6], alternative="two.sided")
        assert result.p_value < result_two.p_value

    def test_less_alternative(self):
        """Test less alternative."""
        result = twoac([2, 2, 6], alternative="less")
        assert result.alternative == "less"

    def test_alternative_formats_accepted(self):
        """Test that different alternative formats are accepted."""
        # All these should work
        twoac([2, 2, 6], alternative="two.sided")
        twoac([2, 2, 6], alternative="two-sided")
        twoac([2, 2, 6], alternative="two_sided")
        twoac([2, 2, 6], alternative="greater")
        twoac([2, 2, 6], alternative="less")


class TestTwoACConfidenceIntervals:
    """Tests for confidence intervals."""

    def test_confint_available_for_typical_data(self):
        """Test that CI is available for typical data."""
        result = twoac([2, 2, 6])
        assert result.confint is not None
        assert len(result.confint) == 2

    def test_confint_lower_less_than_upper(self):
        """Test that lower CI bound is less than upper."""
        result = twoac([2, 2, 6])
        assert result.confint[0] < result.confint[1]

    def test_confint_contains_estimate(self):
        """Test that CI contains the point estimate."""
        result = twoac([2, 2, 6])
        assert result.confint[0] <= result.d_prime <= result.confint[1]

    def test_wald_confint(self):
        """Test Wald confidence interval."""
        result = twoac([2, 2, 6], statistic="wald")
        assert result.confint is not None
        assert result.confint[0] < result.d_prime < result.confint[1]

    def test_different_conf_levels(self):
        """Test different confidence levels."""
        result_95 = twoac([2, 2, 6], conf_level=0.95, statistic="wald")
        result_90 = twoac([2, 2, 6], conf_level=0.90, statistic="wald")
        result_99 = twoac([2, 2, 6], conf_level=0.99, statistic="wald")

        # Wider level should give wider CI
        width_90 = result_90.confint[1] - result_90.confint[0]
        width_95 = result_95.confint[1] - result_95.confint[0]
        width_99 = result_99.confint[1] - result_99.confint[0]

        assert width_90 < width_95 < width_99


class TestTwoACBoundaryCases:
    """Tests for boundary cases."""

    def test_case1_all_prefer_a(self):
        """Case 1: x1>0, x2=0, x3=0 -> d'=-inf, tau=0."""
        result = twoac([5, 0, 0])
        assert result.d_prime == -np.inf
        assert result.tau == 0

    def test_case2_all_no_preference(self):
        """Case 2: x1=0, x2>0, x3=0 -> d'=NA, tau=NA."""
        result = twoac([0, 5, 0])
        assert np.isnan(result.d_prime)
        assert np.isnan(result.tau)

    def test_case3_all_prefer_b(self):
        """Case 3: x1=0, x2=0, x3>0 -> d'=+inf, tau=0."""
        result = twoac([0, 0, 5])
        assert result.d_prime == np.inf
        assert result.tau == 0

    def test_case4_prefer_a_or_no_pref(self):
        """Case 4: x1>0, x2>0, x3=0 -> d'=-inf."""
        result = twoac([5, 4, 0])
        assert result.d_prime == -np.inf

    def test_case5_prefer_b_or_no_pref(self):
        """Case 5: x1=0, x2>0, x3>0 -> d'=+inf."""
        result = twoac([0, 4, 5])
        assert result.d_prime == np.inf

    def test_case6_no_middle_response(self):
        """Case 6: x1>0, x2=0, x3>0 -> finite estimates."""
        result = twoac([5, 0, 15])
        assert np.isfinite(result.d_prime)
        # tau should be 0 for this symmetric case
        assert result.tau == 0

    def test_boundary_case_gives_p_value(self):
        """Test that boundary cases still give p-value with likelihood test."""
        result = twoac([5, 0, 15])
        assert result.p_value is not None
        assert 0 <= result.p_value <= 1


class TestTwoACValidation:
    """Input validation tests."""

    def test_data_must_be_length_3(self):
        """Test that data must have length 3."""
        with pytest.raises(ValueError, match="length 3"):
            twoac([1, 2])

    def test_data_must_be_non_negative(self):
        """Test that data must be non-negative."""
        with pytest.raises(ValueError, match="non-negative"):
            twoac([2, -1, 3])

    def test_data_must_be_integers(self):
        """Test that data must be integers."""
        with pytest.raises(ValueError, match="integer"):
            twoac([2.5, 2, 6])

    def test_invalid_statistic_raises(self):
        """Test that invalid statistic raises error."""
        with pytest.raises(ValueError, match="statistic"):
            twoac([2, 2, 6], statistic="invalid")

    def test_invalid_alternative_raises(self):
        """Test that invalid alternative raises error."""
        with pytest.raises(ValueError, match="alternative"):
            twoac([2, 2, 6], alternative="invalid")

    def test_invalid_conf_level_raises(self):
        """Test that invalid conf_level raises error."""
        with pytest.raises(ValueError, match="conf_level"):
            twoac([2, 2, 6], conf_level=1.5)


class TestTwoACNullHypothesis:
    """Tests for different null hypothesis values."""

    def test_d_prime_0_default_is_zero(self):
        """Test that default d_prime_0 is 0."""
        result = twoac([2, 2, 6])
        assert result.d_prime_0 == 0.0

    def test_nonzero_d_prime_0(self):
        """Test non-zero d_prime_0."""
        result = twoac([15, 15, 20], d_prime_0=0.5)
        assert result.d_prime_0 == 0.5

    def test_similarity_test(self):
        """Test similarity test (alternative='less')."""
        # Typical similarity test: d-prime is less than some threshold
        result = twoac([15, 15, 20], d_prime_0=0.5, alternative="less")
        assert result.alternative == "less"
        assert result.d_prime_0 == 0.5


class TestTwoACOutput:
    """Tests for output formatting."""

    def test_str_output(self):
        """Test string representation."""
        result = twoac([2, 2, 6])
        output = str(result)
        assert "2-AC" in output
        assert "d_prime" in output
        assert "tau" in output

    def test_coefficients_matrix_shape(self):
        """Test coefficients matrix shape."""
        result = twoac([2, 2, 6])
        assert result.coefficients.shape == (2, 2)


class TestTwoACGoldenValidation:
    """Validation tests comparing against sensR golden data."""

    def test_simple_case_matches_sensr(self, golden_twoac_data):
        """Test simple case matches sensR."""
        if golden_twoac_data is None:
            pytest.skip("Golden data not available")

        expected = golden_twoac_data["simple_case"]
        result = twoac([2, 2, 6])

        assert result.tau == pytest.approx(expected["tau"], rel=1e-3)
        assert result.d_prime == pytest.approx(expected["d_prime"], rel=1e-3)

    def test_se_matches_sensr(self, golden_twoac_data):
        """Test standard errors match sensR."""
        if golden_twoac_data is None:
            pytest.skip("Golden data not available")

        expected = golden_twoac_data["simple_case"]
        result = twoac([2, 2, 6])

        assert result.se_tau == pytest.approx(expected["se_tau"], rel=1e-2)
        assert result.se_d_prime == pytest.approx(expected["se_d_prime"], rel=1e-2)
