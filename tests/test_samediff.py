"""Tests for senspy.samediff module - Same-Different protocol analysis."""

import numpy as np
import pytest

from senspy import samediff, SameDiffResult


class TestSameDiffBasic:
    """Basic tests for samediff function."""

    def test_simple_case_returns_result(self):
        """Test that simple case returns SameDiffResult."""
        result = samediff(8, 5, 4, 9)
        assert isinstance(result, SameDiffResult)

    def test_estimates_are_finite(self):
        """Test that estimates are finite for typical data."""
        result = samediff(8, 5, 4, 9)
        assert np.isfinite(result.tau)
        assert np.isfinite(result.delta)

    def test_standard_errors_available(self):
        """Test that SEs are available for typical data."""
        result = samediff(8, 5, 4, 9)
        assert result.se_tau is not None
        assert result.se_delta is not None
        assert result.se_tau > 0
        assert result.se_delta > 0

    def test_vcov_is_2x2_matrix(self):
        """Test that vcov is 2x2 matrix when available."""
        result = samediff(8, 5, 4, 9)
        assert result.vcov is not None
        assert result.vcov.shape == (2, 2)

    def test_positive_tau(self):
        """Test that tau is positive for typical data."""
        result = samediff(8, 5, 4, 9)
        assert result.tau > 0

    def test_positive_delta(self):
        """Test that delta is positive when estimable."""
        result = samediff(8, 5, 4, 9)
        assert result.delta > 0

    def test_log_likelihood_is_finite(self):
        """Test that log-likelihood is finite."""
        result = samediff(8, 5, 4, 9)
        assert np.isfinite(result.log_likelihood)

    def test_data_input_alternative(self):
        """Test that data parameter works."""
        result1 = samediff(8, 5, 4, 9)
        result2 = samediff(data=[8, 5, 4, 9])
        assert result1.tau == result2.tau
        assert result1.delta == result2.delta


class TestSameDiffBoundaryCases:
    """Tests for boundary cases."""

    def test_case_01_no_same_responses(self):
        """Case 0.1: ss=0, sd=0 -> tau=0, delta=NA."""
        result = samediff(0, 5, 0, 9)
        assert result.tau == 0
        assert np.isnan(result.delta)
        assert result.case == 0.1

    def test_case_1_no_diff_responses(self):
        """Case 1: ds=0, dd=0 -> tau=Inf, delta=NA."""
        result = samediff(8, 0, 4, 0)
        assert result.tau == np.inf
        assert np.isnan(result.delta)
        assert result.case == 1.0

    def test_case_12_ds_sd_zero(self):
        """Case 1.2: ds=0, sd=0 -> tau=Inf, delta=Inf."""
        result = samediff(8, 0, 0, 9)
        assert result.tau == np.inf
        assert result.delta == np.inf
        assert result.case == 1.2

    def test_case_3_sd_zero(self):
        """Case 3: sd=0 -> delta=Inf."""
        result = samediff(8, 5, 0, 9)
        assert result.delta == np.inf
        assert result.case == 3.0

    def test_case_2_delta_not_estimable(self):
        """Case 2: delta=0 when not estimable."""
        # When P(same|same) <= P(same|diff)
        result = samediff(4, 8, 9, 5)
        assert result.delta == 0
        assert result.case == 2.0


class TestSameDiffValidation:
    """Input validation tests."""

    def test_data_must_be_length_4(self):
        """Test that data must have length 4."""
        with pytest.raises(ValueError, match="length 4"):
            samediff(data=[1, 2, 3])

    def test_data_must_be_non_negative(self):
        """Test that data must be non-negative."""
        with pytest.raises(ValueError, match="non-negative"):
            samediff(8, -1, 4, 9)

    def test_data_must_be_integers(self):
        """Test that data must be integers."""
        with pytest.raises(ValueError, match="non-negative integers"):
            samediff(8.5, 5, 4, 9)

    def test_too_many_zeros_raises(self):
        """Test that more than 2 zeros raises error."""
        with pytest.raises(ValueError, match="more than two"):
            samediff(0, 0, 0, 9)

    def test_requires_all_counts_or_data(self):
        """Test that either all counts or data must be provided."""
        with pytest.raises(ValueError):
            samediff(8, 5, 4)  # Missing one count


class TestSameDiffOutput:
    """Tests for output formatting."""

    def test_str_output(self):
        """Test string representation."""
        result = samediff(8, 5, 4, 9)
        output = str(result)
        assert "Same-Different" in output
        assert "tau" in output
        assert "delta" in output

    def test_coefficients_matrix_shape(self):
        """Test coefficients matrix shape."""
        result = samediff(8, 5, 4, 9)
        assert result.coefficients.shape == (2, 2)

    def test_vcov_disabled(self):
        """Test that vcov=False returns None."""
        result = samediff(8, 5, 4, 9, vcov=False)
        assert result.vcov is None


class TestSameDiffNumericalStability:
    """Tests for numerical stability."""

    def test_large_counts(self):
        """Test with large counts."""
        result = samediff(800, 500, 400, 900)
        assert np.isfinite(result.tau)
        assert np.isfinite(result.delta)
        assert np.isfinite(result.log_likelihood)

    def test_small_counts(self):
        """Test with small counts."""
        result = samediff(2, 1, 1, 2)
        assert np.isfinite(result.tau)
        # delta may or may not be finite depending on data

    def test_asymmetric_data(self):
        """Test with highly asymmetric data."""
        result = samediff(50, 2, 3, 45)
        assert np.isfinite(result.tau)


class TestSameDiffGoldenValidation:
    """Validation tests comparing against sensR golden data."""

    def test_simple_case_matches_sensr(self, golden_samediff_data):
        """Test simple case matches sensR."""
        if golden_samediff_data is None:
            pytest.skip("Golden data not available")

        expected = golden_samediff_data["simple_case"]
        result = samediff(8, 5, 4, 9)

        assert result.tau == pytest.approx(expected["tau"], rel=1e-3)
        assert result.delta == pytest.approx(expected["delta"], rel=1e-3)

    def test_se_matches_sensr(self, golden_samediff_data):
        """Test standard errors match sensR."""
        if golden_samediff_data is None:
            pytest.skip("Golden data not available")

        expected = golden_samediff_data["simple_case"]
        result = samediff(8, 5, 4, 9)

        if expected.get("se_tau") is not None:
            assert result.se_tau == pytest.approx(expected["se_tau"], rel=1e-2)
        if expected.get("se_delta") is not None:
            assert result.se_delta == pytest.approx(expected["se_delta"], rel=1e-2)

    def test_log_likelihood_matches_sensr(self, golden_samediff_data):
        """Test log-likelihood matches sensR."""
        if golden_samediff_data is None:
            pytest.skip("Golden data not available")

        expected = golden_samediff_data["simple_case"]
        result = samediff(8, 5, 4, 9)

        assert result.log_likelihood == pytest.approx(expected["log_likelihood"], rel=1e-3)

    def test_boundary_case_matches_sensr(self, golden_samediff_data):
        """Test boundary case matches sensR."""
        if golden_samediff_data is None:
            pytest.skip("Golden data not available")

        if "boundary_sd_zero" not in golden_samediff_data:
            pytest.skip("Boundary case data not available")

        expected = golden_samediff_data["boundary_sd_zero"]
        result = samediff(8, 5, 0, 9)

        assert result.delta == np.inf
        assert result.tau == pytest.approx(expected["tau"], rel=1e-2)
