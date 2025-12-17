"""Tests for dprime_test, dprime_compare, and posthoc functions."""

import numpy as np
import pytest

from senspy.dprime_tests import (
    DprimeCompareResult,
    DprimeTestResult,
    PosthocResult,
    dprime_compare,
    dprime_table,
    dprime_test,
    posthoc,
)


class TestDprimeTable:
    """Tests for dprime_table function."""

    def test_basic_table(self):
        """Test basic table creation."""
        correct = [60, 45, 55]
        total = [100, 100, 100]
        protocol = ["triangle", "duotrio", "twoAFC"]

        table = dprime_table(correct, total, protocol)

        assert len(table) == 3
        assert table[0].correct == 60
        assert table[0].total == 100
        assert table[0].protocol == "triangle"
        assert table[0].p_hat > 0
        assert table[0].dprime >= 0

    def test_restrict_above_guess(self):
        """Test that p_hat is restricted above guessing."""
        correct = [30]  # Below chance for triangle (1/3)
        total = [100]
        protocol = ["triangle"]

        table = dprime_table(correct, total, protocol, restrict_above_guess=True)

        assert table[0].p_hat >= 1 / 3
        assert table[0].dprime == 0  # Should be 0 at chance

    def test_no_restrict_above_guess(self):
        """Test without restriction above guessing."""
        correct = [30]
        total = [100]
        protocol = ["triangle"]

        table = dprime_table(correct, total, protocol, restrict_above_guess=False)

        assert table[0].p_hat == 0.3
        # d-prime undefined below chance - should handle gracefully

    def test_invalid_protocol(self):
        """Test that invalid protocol raises error."""
        with pytest.raises(ValueError, match="Invalid protocol"):
            dprime_table([60], [100], ["invalid_protocol"])

    def test_mismatched_lengths(self):
        """Test that mismatched lengths raise error."""
        with pytest.raises(ValueError, match="same length"):
            dprime_table([60, 70], [100], ["triangle"])


class TestDprimeTest:
    """Tests for dprime_test function."""

    def test_basic_test(self):
        """Test basic dprime_test."""
        correct = [60, 45, 55]
        total = [100, 100, 100]
        protocol = ["triangle", "duotrio", "twoAFC"]

        result = dprime_test(correct, total, protocol)

        assert isinstance(result, DprimeTestResult)
        assert result.d_prime > 0
        assert 0 <= result.p_value <= 1
        assert result.alternative == "greater"
        assert result.dprime0 == 0.0

    def test_likelihood_statistic(self):
        """Test dprime_test with likelihood statistic."""
        correct = [60, 45, 55]
        total = [100, 100, 100]
        protocol = ["triangle", "duotrio", "twoAFC"]

        result = dprime_test(correct, total, protocol, statistic="likelihood")

        assert result.statistic == "likelihood"
        assert np.isfinite(result.stat_value)

    def test_wald_statistic(self):
        """Test dprime_test with Wald statistic."""
        correct = [60, 45, 55]
        total = [100, 100, 100]
        protocol = ["triangle", "duotrio", "twoAFC"]

        result = dprime_test(correct, total, protocol, statistic="Wald")

        assert result.statistic == "Wald"
        assert np.isfinite(result.stat_value)

    def test_alternative_similarity(self):
        """Test dprime_test with similarity alternative."""
        correct = [60, 45, 55]
        total = [100, 100, 100]
        protocol = ["triangle", "duotrio", "twoAFC"]

        result = dprime_test(
            correct, total, protocol, alternative="similarity", dprime0=2.0
        )

        assert result.alternative == "less"
        assert result.dprime0 == 2.0

    def test_alternative_two_sided(self):
        """Test dprime_test with two-sided alternative."""
        correct = [60, 45, 55]
        total = [100, 100, 100]
        protocol = ["triangle", "duotrio", "twoAFC"]

        result = dprime_test(
            correct, total, protocol, alternative="two.sided", dprime0=1.0
        )

        assert result.alternative == "two.sided"

    def test_dprime0_zero_requires_greater(self):
        """Test that dprime0=0 requires greater/difference alternative."""
        correct = [60, 45, 55]
        total = [100, 100, 100]
        protocol = ["triangle", "duotrio", "twoAFC"]

        with pytest.raises(ValueError, match="'alternative' has to be"):
            dprime_test(correct, total, protocol, alternative="two.sided", dprime0=0.0)

    def test_weighted_avg_estimation(self):
        """Test dprime_test with weighted average estimation."""
        correct = [60, 55, 58]
        total = [100, 100, 100]
        protocol = ["triangle", "triangle", "triangle"]

        result = dprime_test(correct, total, protocol, estim="weighted_avg")

        assert result.estim == "weighted_avg"
        assert np.isfinite(result.d_prime)


class TestDprimeCompare:
    """Tests for dprime_compare function."""

    def test_basic_compare(self):
        """Test basic dprime_compare."""
        correct = [60, 45, 55]
        total = [100, 100, 100]
        protocol = ["triangle", "duotrio", "twoAFC"]

        result = dprime_compare(correct, total, protocol)

        assert isinstance(result, DprimeCompareResult)
        assert result.d_prime > 0
        assert result.df == 2  # n_groups - 1
        assert 0 <= result.p_value <= 1

    def test_likelihood_statistic(self):
        """Test dprime_compare with likelihood statistic."""
        correct = [60, 45, 55]
        total = [100, 100, 100]
        protocol = ["triangle", "duotrio", "twoAFC"]

        result = dprime_compare(correct, total, protocol, statistic="likelihood")

        assert result.statistic == "likelihood"
        assert result.stat_value >= 0

    def test_pearson_statistic(self):
        """Test dprime_compare with Pearson statistic."""
        correct = [60, 45, 55]
        total = [100, 100, 100]
        protocol = ["triangle", "duotrio", "twoAFC"]

        result = dprime_compare(correct, total, protocol, statistic="Pearson")

        assert result.statistic == "Pearson"
        assert result.stat_value >= 0

    def test_wald_p_statistic(self):
        """Test dprime_compare with Wald.p statistic."""
        correct = [60, 45, 55]
        total = [100, 100, 100]
        protocol = ["triangle", "duotrio", "twoAFC"]

        result = dprime_compare(correct, total, protocol, statistic="Wald.p")

        assert result.statistic == "Wald.p"

    def test_wald_d_statistic(self):
        """Test dprime_compare with Wald.d statistic."""
        correct = [60, 55, 58]
        total = [100, 100, 100]
        protocol = ["triangle", "triangle", "triangle"]

        result = dprime_compare(correct, total, protocol, statistic="Wald.d")

        assert result.statistic == "Wald.d"


class TestPosthoc:
    """Tests for posthoc function."""

    def test_pairwise_posthoc(self):
        """Test pairwise post-hoc comparisons."""
        correct = [60, 45, 55]
        total = [100, 100, 100]
        protocol = ["triangle", "duotrio", "twoAFC"]

        compare_result = dprime_compare(correct, total, protocol)
        result = posthoc(compare_result, test="pairwise")

        assert isinstance(result, PosthocResult)
        assert result.test == "pairwise"
        assert len(result.posthoc) == 3  # 3 choose 2 pairs

    def test_zero_posthoc(self):
        """Test post-hoc comparison to zero."""
        correct = [60, 45, 55]
        total = [100, 100, 100]
        protocol = ["triangle", "duotrio", "twoAFC"]

        compare_result = dprime_compare(correct, total, protocol)
        result = posthoc(compare_result, test="zero")

        assert result.test == "zero"
        assert len(result.posthoc) == 3

    def test_common_posthoc(self):
        """Test post-hoc comparison to common d-prime."""
        correct = [60, 45, 55]
        total = [100, 100, 100]
        protocol = ["triangle", "duotrio", "twoAFC"]

        compare_result = dprime_compare(correct, total, protocol)
        result = posthoc(compare_result, test="common")

        assert result.test == "common"
        assert len(result.posthoc) == 3

    def test_letter_display(self):
        """Test letter display for pairwise comparisons."""
        correct = [60, 45, 55]
        total = [100, 100, 100]
        protocol = ["triangle", "duotrio", "twoAFC"]

        compare_result = dprime_compare(correct, total, protocol)
        result = posthoc(compare_result, test="pairwise", alternative="two.sided")

        # Letters should be present for pairwise two-sided
        assert result.letters is not None

    def test_p_adjust_holm(self):
        """Test p-value adjustment with Holm method."""
        correct = [60, 45, 55]
        total = [100, 100, 100]
        protocol = ["triangle", "duotrio", "twoAFC"]

        compare_result = dprime_compare(correct, total, protocol)
        result = posthoc(compare_result, test="pairwise", padj_method="holm")

        assert result.padj_method == "holm"

    def test_p_adjust_bonferroni(self):
        """Test p-value adjustment with Bonferroni method."""
        correct = [60, 45, 55]
        total = [100, 100, 100]
        protocol = ["triangle", "duotrio", "twoAFC"]

        compare_result = dprime_compare(correct, total, protocol)
        result = posthoc(compare_result, test="pairwise", padj_method="bonferroni")

        assert result.padj_method == "bonferroni"


class TestDprimeTestsGoldenValidation:
    """Tests validating dprime_test/dprime_compare against sensR golden data."""

    def test_golden_dprime_test(self, golden_dprime_tests_data, tolerance):
        """Test dprime_test against golden data."""
        if golden_dprime_tests_data is None:
            pytest.skip("Golden dprime_tests data not available")

        case = golden_dprime_tests_data.get("dprime_test_simple")
        if case is None:
            pytest.skip("dprime_test_simple not in golden data")

        correct = case["input"]["correct"]
        total = case["input"]["total"]
        protocol = case["input"]["protocol"]

        result = dprime_test(correct, total, protocol, statistic="likelihood")

        assert np.isclose(
            result.d_prime, case["d_prime"], rtol=tolerance["coefficients"]
        )
        assert np.isclose(
            result.stat_value, case["stat_value"], rtol=tolerance["coefficients"]
        )
        assert np.isclose(result.p_value, case["p_value"], rtol=tolerance["p_values"])

    def test_golden_dprime_compare(self, golden_dprime_tests_data, tolerance):
        """Test dprime_compare against golden data."""
        if golden_dprime_tests_data is None:
            pytest.skip("Golden dprime_tests data not available")

        case = golden_dprime_tests_data.get("dprime_compare_simple")
        if case is None:
            pytest.skip("dprime_compare_simple not in golden data")

        correct = case["input"]["correct"]
        total = case["input"]["total"]
        protocol = case["input"]["protocol"]

        result = dprime_compare(correct, total, protocol, statistic="likelihood")

        assert np.isclose(
            result.d_prime, case["d_prime"], rtol=tolerance["coefficients"]
        )
        assert result.df == case["df"]
        assert np.isclose(
            result.stat_value, case["stat_value"], rtol=tolerance["coefficients"]
        )
        assert np.isclose(result.p_value, case["p_value"], rtol=tolerance["p_values"])
