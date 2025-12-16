"""Tests for senspy.discrim module - discrimination analysis."""

import numpy as np
import pytest

from senspy import discrim
from senspy.core.types import Protocol, Statistic


class TestDiscrimBasic:
    """Basic tests for discrim function."""

    def test_basic_triangle(self):
        """Test basic triangle discrimination analysis."""
        result = discrim(correct=80, total=100, method="triangle")

        assert result.pc == 0.8
        assert result.pd == pytest.approx(0.7, rel=1e-6)  # (0.8 - 1/3) / (1 - 1/3)
        assert result.d_prime > 0
        assert result.method == Protocol.TRIANGLE
        assert result.correct == 80
        assert result.total == 100

    def test_basic_twoafc(self):
        """Test basic 2-AFC discrimination analysis."""
        result = discrim(correct=75, total=100, method="twoafc")

        assert result.pc == 0.75
        assert result.pd == pytest.approx(0.5, rel=1e-6)  # (0.75 - 0.5) / (1 - 0.5)
        assert result.d_prime > 0
        assert result.method == Protocol.TWOAFC

    def test_at_chance_level(self):
        """Test when performance is at chance level."""
        result = discrim(correct=33, total=100, method="triangle")

        assert result.pc == 0.33
        assert result.pd == 0.0
        assert result.d_prime == 0.0

    def test_below_chance_level(self):
        """Test when performance is below chance level."""
        result = discrim(correct=25, total=100, method="triangle")

        assert result.pc == 0.25
        assert result.pd == 0.0
        assert result.d_prime == 0.0

    def test_perfect_performance(self):
        """Test with perfect performance (100% correct)."""
        result = discrim(correct=100, total=100, method="triangle")

        assert result.pc == 1.0
        assert result.pd == 1.0
        assert result.d_prime > 0  # Should be very high

    @pytest.mark.parametrize("stat", ["exact", "wald", "likelihood", "score"])
    def test_perfect_performance_all_statistics(self, stat):
        """Test that all statistics return significant p-value for perfect performance."""
        result = discrim(correct=100, total=100, method="triangle", statistic=stat)

        assert result.pc == 1.0
        # Perfect performance should always be significant
        assert result.p_value < 0.05, f"{stat} should be significant for perfect performance"


class TestDiscrimStatistics:
    """Tests for different test statistics."""

    def test_exact_statistic(self):
        """Test exact binomial statistic."""
        result = discrim(80, 100, "triangle", statistic="exact")

        assert result.stat_type == Statistic.EXACT
        assert 0 <= result.p_value <= 1
        assert result.statistic == 80  # For exact, statistic is correct count

    def test_wald_statistic(self):
        """Test Wald statistic."""
        result = discrim(80, 100, "triangle", statistic="wald")

        assert result.stat_type == Statistic.WALD
        assert 0 <= result.p_value <= 1
        assert result.statistic > 0  # z-score should be positive for 80/100

    def test_likelihood_statistic(self):
        """Test likelihood ratio statistic."""
        result = discrim(80, 100, "triangle", statistic="likelihood")

        assert result.stat_type == Statistic.LIKELIHOOD
        assert 0 <= result.p_value <= 1

    def test_score_statistic(self):
        """Test score statistic."""
        result = discrim(80, 100, "triangle", statistic="score")

        assert result.stat_type == Statistic.SCORE
        assert 0 <= result.p_value <= 1

    def test_all_statistics_consistent(self):
        """Test that all statistics give consistent results."""
        results = {}
        for stat in ["exact", "wald", "likelihood", "score"]:
            results[stat] = discrim(80, 100, "triangle", statistic=stat)

        # All should agree on point estimates
        for stat, r in results.items():
            assert r.d_prime == pytest.approx(results["exact"].d_prime, rel=1e-6)
            assert r.pc == pytest.approx(results["exact"].pc, rel=1e-6)

        # All p-values should be very small for 80/100 triangle
        for stat, r in results.items():
            assert r.p_value < 0.001, f"{stat} p-value should be < 0.001"


class TestDiscrimConfidenceIntervals:
    """Tests for confidence intervals."""

    def test_default_ci_level(self):
        """Test default 95% confidence level."""
        result = discrim(80, 100, "triangle")

        assert result.conf_level == 0.95
        ci = result.confint()
        assert len(ci) == 2
        assert ci[0] < result.d_prime < ci[1]

    def test_ci_different_parameters(self):
        """Test CI for different parameters."""
        result = discrim(80, 100, "triangle")

        ci_d = result.confint(parameter="d_prime")
        ci_pc = result.confint(parameter="pc")
        ci_pd = result.confint(parameter="pd")

        assert ci_d[0] >= 0  # d-prime lower bound
        assert ci_pc[0] >= 1/3  # pc lower bound for triangle
        assert ci_pd[0] >= 0  # pd lower bound

        assert ci_pc[1] <= 1  # pc upper bound
        assert ci_pd[1] <= 1  # pd upper bound

    def test_ci_custom_level(self):
        """Test CI with custom confidence level."""
        result = discrim(80, 100, "triangle")

        ci_95 = result.confint(level=0.95)
        ci_99 = result.confint(level=0.99)
        ci_90 = result.confint(level=0.90)

        # Higher confidence = wider interval
        assert (ci_99[1] - ci_99[0]) > (ci_95[1] - ci_95[0])
        assert (ci_95[1] - ci_95[0]) > (ci_90[1] - ci_90[0])


class TestDiscrimSimilarityTest:
    """Tests for similarity tests."""

    def test_similarity_test_with_pd0(self):
        """Test similarity test with pd0 null hypothesis."""
        result = discrim(
            correct=40, total=100, method="triangle",
            pd0=0.5, test="similarity"
        )

        # For similarity test, we test if d' < d'0
        assert 0 <= result.p_value <= 1

    def test_similarity_test_with_d_prime0(self):
        """Test similarity test with d_prime0 null hypothesis."""
        result = discrim(
            correct=40, total=100, method="triangle",
            d_prime0=2.0, test="similarity"
        )

        assert 0 <= result.p_value <= 1

    def test_similarity_requires_null(self):
        """Test that similarity test requires null hypothesis."""
        with pytest.raises(ValueError, match="must be specified"):
            discrim(40, 100, "triangle", test="similarity")


class TestDiscrimInputValidation:
    """Tests for input validation."""

    def test_correct_must_be_nonnegative(self):
        """Test that correct must be non-negative."""
        with pytest.raises(ValueError, match="non-negative"):
            discrim(correct=-1, total=100, method="triangle")

    def test_total_must_be_positive(self):
        """Test that total must be positive."""
        with pytest.raises(ValueError, match="positive"):
            discrim(correct=50, total=0, method="triangle")

    def test_correct_cannot_exceed_total(self):
        """Test that correct cannot exceed total."""
        with pytest.raises(ValueError, match="cannot be larger"):
            discrim(correct=101, total=100, method="triangle")

    def test_conf_level_must_be_valid(self):
        """Test that conf_level must be between 0 and 1."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            discrim(80, 100, "triangle", conf_level=1.5)

    def test_invalid_method_raises(self):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="Unknown protocol"):
            discrim(80, 100, "invalid_method")

    def test_invalid_statistic_raises(self):
        """Test that invalid statistic raises error."""
        with pytest.raises(ValueError, match="Unknown statistic"):
            discrim(80, 100, "triangle", statistic="invalid")

    def test_invalid_test_raises(self):
        """Test that invalid test type raises error."""
        with pytest.raises(ValueError, match="Unknown test"):
            discrim(80, 100, "triangle", test="invalid")

    def test_both_pd0_and_d_prime0_raises(self):
        """Test that specifying both pd0 and d_prime0 raises error."""
        with pytest.raises(ValueError, match="Only specify one"):
            discrim(80, 100, "triangle", pd0=0.5, d_prime0=1.0)


class TestDiscrimProtocols:
    """Tests for different protocols."""

    @pytest.mark.parametrize("method", [
        "duotrio", "triangle", "twoafc", "threeafc",
        "tetrad", "hexad", "twofive", "twofivef"
    ])
    def test_all_protocols_work(self, method):
        """Test that all protocols work."""
        result = discrim(correct=80, total=100, method=method)

        assert result.pc == 0.8
        assert result.d_prime >= 0
        assert 0 <= result.p_value <= 1

    def test_protocol_enum_accepted(self):
        """Test that Protocol enum is accepted."""
        result = discrim(80, 100, method=Protocol.TRIANGLE)
        assert result.method == Protocol.TRIANGLE


class TestDiscrimSummary:
    """Tests for summary output."""

    def test_summary_returns_string(self):
        """Test that summary returns a string."""
        result = discrim(80, 100, "triangle")
        summary = result.summary()

        assert isinstance(summary, str)
        assert "triangle" in summary.lower()
        assert "80" in summary
        assert "100" in summary

    def test_str_representation(self):
        """Test string representation."""
        result = discrim(80, 100, "triangle")
        s = str(result)

        assert "DiscrimResult" in s
        assert "triangle" in s
