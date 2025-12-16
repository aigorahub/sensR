"""Tests for senspy.core.types module."""

import pytest

from senspy.core.types import Protocol, Statistic, Alternative, parse_protocol


class TestProtocol:
    """Tests for the Protocol enum."""

    def test_protocol_values(self):
        """Test that protocol values match expected strings."""
        assert Protocol.TRIANGLE.value == "triangle"
        assert Protocol.TWOAFC.value == "twoafc"
        assert Protocol.THREEAFC.value == "threeafc"
        assert Protocol.DUOTRIO.value == "duotrio"
        assert Protocol.TETRAD.value == "tetrad"
        assert Protocol.HEXAD.value == "hexad"
        assert Protocol.TWOFIVE.value == "twofive"
        assert Protocol.TWOFIVEF.value == "twofivef"

    def test_p_guess_values(self):
        """Test guessing probabilities for each protocol."""
        assert Protocol.TWOAFC.p_guess == 0.5
        assert Protocol.DUOTRIO.p_guess == 0.5
        assert Protocol.TRIANGLE.p_guess == pytest.approx(1 / 3)
        assert Protocol.THREEAFC.p_guess == pytest.approx(1 / 3)
        assert Protocol.TETRAD.p_guess == pytest.approx(1 / 3)
        assert Protocol.HEXAD.p_guess == pytest.approx(0.1)
        assert Protocol.TWOFIVE.p_guess == pytest.approx(0.1)
        assert Protocol.TWOFIVEF.p_guess == pytest.approx(0.4)

    def test_protocol_is_string(self):
        """Test that Protocol values work as strings."""
        assert Protocol.TRIANGLE == "triangle"
        assert Protocol.TWOAFC.value == "twoafc"


class TestParseProtocol:
    """Tests for the parse_protocol function."""

    def test_parse_enum_passthrough(self):
        """Test that Protocol enums are returned unchanged."""
        assert parse_protocol(Protocol.TRIANGLE) == Protocol.TRIANGLE
        assert parse_protocol(Protocol.TWOAFC) == Protocol.TWOAFC

    def test_parse_lowercase_string(self):
        """Test parsing lowercase protocol strings."""
        assert parse_protocol("triangle") == Protocol.TRIANGLE
        assert parse_protocol("twoafc") == Protocol.TWOAFC
        assert parse_protocol("duotrio") == Protocol.DUOTRIO

    def test_parse_case_insensitive(self):
        """Test case-insensitive parsing."""
        assert parse_protocol("TRIANGLE") == Protocol.TRIANGLE
        assert parse_protocol("Triangle") == Protocol.TRIANGLE
        assert parse_protocol("TwoAFC") == Protocol.TWOAFC

    def test_parse_common_aliases(self):
        """Test parsing common aliases."""
        assert parse_protocol("2afc") == Protocol.TWOAFC
        assert parse_protocol("2AFC") == Protocol.TWOAFC
        assert parse_protocol("3afc") == Protocol.THREEAFC
        assert parse_protocol("3AFC") == Protocol.THREEAFC
        assert parse_protocol("duo") == Protocol.DUOTRIO
        assert parse_protocol("tri") == Protocol.TRIANGLE

    def test_parse_with_separators(self):
        """Test parsing with dashes and underscores."""
        assert parse_protocol("two_afc") == Protocol.TWOAFC
        assert parse_protocol("two-afc") == Protocol.TWOAFC
        assert parse_protocol("three_afc") == Protocol.THREEAFC

    def test_parse_invalid_raises(self):
        """Test that invalid protocols raise ValueError."""
        with pytest.raises(ValueError, match="Unknown protocol"):
            parse_protocol("invalid")
        with pytest.raises(ValueError, match="Unknown protocol"):
            parse_protocol("xyz")


class TestStatistic:
    """Tests for the Statistic enum."""

    def test_statistic_values(self):
        """Test statistic values match expected strings."""
        assert Statistic.EXACT.value == "exact"
        assert Statistic.LIKELIHOOD.value == "likelihood"
        assert Statistic.WALD.value == "wald"
        assert Statistic.SCORE.value == "score"


class TestAlternative:
    """Tests for the Alternative enum."""

    def test_alternative_values(self):
        """Test alternative values match expected strings."""
        assert Alternative.TWO_SIDED.value == "two.sided"
        assert Alternative.GREATER.value == "greater"
        assert Alternative.LESS.value == "less"
