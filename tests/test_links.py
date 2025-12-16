"""Tests for senspy.links module - psychometric link functions."""

import numpy as np
import pytest

from senspy.links import (
    psy_fun,
    psy_inv,
    psy_deriv,
    get_link,
    twoafc_link,
    triangle_link,
    duotrio_link,
    threeafc_link,
    tetrad_link,
    hexad_link,
    twofive_link,
    twofivef_link,
)
from senspy.core.types import Protocol


class TestPsyFun:
    """Tests for psy_fun (d-prime to proportion correct)."""

    @pytest.mark.xfail(reason="Golden data needs validation against sensR via RPy2")
    def test_twoafc_golden_data(self, golden_links_data, tolerance):
        """Test twoafc link function against sensR golden data."""
        if golden_links_data is None:
            pytest.skip("Golden data not available")

        data = golden_links_data["psy_fun"]["twoafc"]
        d_prime = np.array(data["d_prime"])
        expected = np.array(data["expected_pc"])

        result = psy_fun(d_prime, method="twoafc")
        np.testing.assert_allclose(result, expected, rtol=tolerance["probabilities"])

    @pytest.mark.xfail(reason="Golden data needs validation against sensR via RPy2")
    def test_triangle_golden_data(self, golden_links_data, tolerance):
        """Test triangle link function against sensR golden data."""
        if golden_links_data is None:
            pytest.skip("Golden data not available")

        data = golden_links_data["psy_fun"]["triangle"]
        d_prime = np.array(data["d_prime"])
        expected = np.array(data["expected_pc"])

        result = psy_fun(d_prime, method="triangle")
        np.testing.assert_allclose(result, expected, rtol=tolerance["probabilities"])

    @pytest.mark.xfail(reason="Golden data needs validation against sensR via RPy2")
    def test_duotrio_golden_data(self, golden_links_data, tolerance):
        """Test duotrio link function against sensR golden data."""
        if golden_links_data is None:
            pytest.skip("Golden data not available")

        data = golden_links_data["psy_fun"]["duotrio"]
        d_prime = np.array(data["d_prime"])
        expected = np.array(data["expected_pc"])

        result = psy_fun(d_prime, method="duotrio")
        np.testing.assert_allclose(result, expected, rtol=tolerance["probabilities"])

    @pytest.mark.xfail(reason="Golden data needs validation against sensR via RPy2")
    def test_threeafc_golden_data(self, golden_links_data, tolerance):
        """Test threeafc link function against sensR golden data."""
        if golden_links_data is None:
            pytest.skip("Golden data not available")

        data = golden_links_data["psy_fun"]["threeafc"]
        d_prime = np.array(data["d_prime"])
        expected = np.array(data["expected_pc"])

        result = psy_fun(d_prime, method="threeafc")
        np.testing.assert_allclose(result, expected, rtol=tolerance["probabilities"])

    def test_d_prime_zero_returns_p_guess(self):
        """Test that d-prime=0 returns guessing probability."""
        assert psy_fun(0, method="twoafc")[0] == pytest.approx(0.5)
        assert psy_fun(0, method="triangle")[0] == pytest.approx(1 / 3, rel=1e-3)
        assert psy_fun(0, method="threeafc")[0] == pytest.approx(1 / 3, rel=1e-3)
        assert psy_fun(0, method="duotrio")[0] == pytest.approx(0.5)

    def test_negative_d_prime_raises(self):
        """Test that negative d-prime raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            psy_fun(-1, method="triangle")

    def test_accepts_protocol_enum(self):
        """Test that Protocol enum is accepted."""
        result = psy_fun(1.0, method=Protocol.TRIANGLE)
        assert result[0] > 1 / 3  # Should be above chance

    def test_scalar_input_returns_array(self):
        """Test that scalar input returns array."""
        result = psy_fun(1.0, method="twoafc")
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)

    def test_array_input_preserves_shape(self):
        """Test that array input preserves shape."""
        d_prime = np.array([0.5, 1.0, 1.5])
        result = psy_fun(d_prime, method="twoafc")
        assert result.shape == d_prime.shape


class TestPsyInv:
    """Tests for psy_inv (proportion correct to d-prime)."""

    @pytest.mark.xfail(reason="Golden data needs validation against sensR via RPy2")
    def test_twoafc_golden_data(self, golden_links_data, tolerance):
        """Test twoafc inverse against sensR golden data."""
        if golden_links_data is None:
            pytest.skip("Golden data not available")

        data = golden_links_data["psy_inv"]["twoafc"]
        pc = np.array(data["pc"])
        expected = np.array(data["expected_d_prime"])

        result = psy_inv(pc, method="twoafc")
        np.testing.assert_allclose(result, expected, rtol=tolerance["coefficients"])

    @pytest.mark.xfail(reason="Golden data needs validation against sensR via RPy2")
    def test_triangle_golden_data(self, golden_links_data, tolerance):
        """Test triangle inverse against sensR golden data."""
        if golden_links_data is None:
            pytest.skip("Golden data not available")

        data = golden_links_data["psy_inv"]["triangle"]
        pc = np.array(data["pc"])
        expected = np.array(data["expected_d_prime"])

        result = psy_inv(pc, method="triangle")
        np.testing.assert_allclose(result, expected, rtol=tolerance["coefficients"])

    def test_p_guess_returns_zero(self):
        """Test that pc=p_guess returns d-prime=0."""
        assert psy_inv(0.5, method="twoafc")[0] == pytest.approx(0.0, abs=1e-6)
        assert psy_inv(1 / 3, method="triangle")[0] == pytest.approx(0.0, abs=1e-6)

    def test_pc_below_p_guess_returns_zero(self):
        """Test that pc below guessing probability returns 0."""
        assert psy_inv(0.25, method="triangle")[0] == pytest.approx(0.0, abs=1e-6)

    def test_roundtrip_consistency(self):
        """Test that psy_fun(psy_inv(pc)) ≈ pc."""
        pc_values = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        d_prime = psy_inv(pc_values, method="twoafc")
        pc_roundtrip = psy_fun(d_prime, method="twoafc")
        np.testing.assert_allclose(pc_roundtrip, pc_values, rtol=1e-6)


class TestPsyDeriv:
    """Tests for psy_deriv (derivative of psychometric function)."""

    @pytest.mark.xfail(reason="Golden data needs validation against sensR via RPy2")
    def test_twoafc_golden_data(self, golden_links_data, tolerance):
        """Test twoafc derivative against sensR golden data."""
        if golden_links_data is None:
            pytest.skip("Golden data not available")

        data = golden_links_data["psy_deriv"]["twoafc"]
        d_prime = np.array(data["d_prime"])
        expected = np.array(data["expected_deriv"])

        result = psy_deriv(d_prime, method="twoafc")
        np.testing.assert_allclose(result, expected, rtol=tolerance["derivatives"])

    def test_derivative_is_positive(self):
        """Test that derivative is non-negative for all protocols."""
        d_prime = np.array([0.5, 1.0, 2.0, 3.0])
        for protocol in Protocol:
            result = psy_deriv(d_prime, method=protocol)
            assert np.all(result >= 0), f"Derivative should be >= 0 for {protocol}"

    def test_negative_d_prime_raises(self):
        """Test that negative d-prime raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            psy_deriv(-1, method="triangle")


class TestGetLink:
    """Tests for get_link function."""

    def test_returns_link_object(self):
        """Test that get_link returns a Link object."""
        link = get_link("triangle")
        assert hasattr(link, "linkinv")
        assert hasattr(link, "linkfun")
        assert hasattr(link, "mu_eta")
        assert hasattr(link, "p_guess")
        assert hasattr(link, "name")

    def test_all_protocols_have_links(self):
        """Test that all protocols have corresponding links."""
        for protocol in Protocol:
            link = get_link(protocol)
            assert link.name == protocol.value
            assert link.p_guess == protocol.p_guess


class TestLinkObjects:
    """Tests for individual link objects."""

    @pytest.mark.parametrize(
        "link,expected_p_guess",
        [
            (twoafc_link, 0.5),
            (triangle_link, 1 / 3),
            (duotrio_link, 0.5),
            (threeafc_link, 1 / 3),
            (tetrad_link, 1 / 3),
            (hexad_link, 0.1),
            (twofive_link, 0.1),
            (twofivef_link, 0.4),
        ],
    )
    def test_link_p_guess(self, link, expected_p_guess):
        """Test that link objects have correct guessing probabilities."""
        assert link.p_guess == pytest.approx(expected_p_guess)

    def test_linkinv_at_zero(self):
        """Test that linkinv(0) returns p_guess."""
        d_zero = np.array([0.0])
        for link in [twoafc_link, triangle_link, duotrio_link, threeafc_link]:
            result = link.linkinv(d_zero)
            assert result[0] == pytest.approx(link.p_guess, rel=1e-3)
