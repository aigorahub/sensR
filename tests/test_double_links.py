"""Tests for double protocol link functions."""

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from senspy.links.double import (
    get_double_link,
    double_twoafc_link,
    double_duotrio_link,
    double_triangle_link,
    double_threeafc_link,
    double_tetrad_link,
)


# Load golden values from sensR
FIXTURES_DIR = Path(__file__).parent / "fixtures"
with open(FIXTURES_DIR / "golden_sensr.json") as f:
    GOLDEN = json.load(f)


class TestDoubleTwoAFC:
    """Tests for double 2-AFC link function."""

    def test_guessing_probability(self):
        """Test guessing probability is 0.25."""
        link = double_twoafc_link()
        assert link.p_guess == 0.25

    def test_linkinv_at_zero(self):
        """At d'=0, probability should be guessing."""
        link = double_twoafc_link()
        assert_allclose(link.linkinv(0), 0.25)

    def test_linkinv_positive(self):
        """Probability should increase with d'."""
        link = double_twoafc_link()
        probs = [link.linkinv(d) for d in [0, 1, 2, 3]]
        assert all(probs[i] < probs[i + 1] for i in range(len(probs) - 1))

    def test_linkinv_bounds(self):
        """Probability should be in [p_guess, 1]."""
        link = double_twoafc_link()
        for d in [0, 0.5, 1, 2, 5, 10]:
            p = link.linkinv(d)
            assert 0.25 <= p <= 1.0

    def test_linkfun_inverse(self):
        """linkfun should be inverse of linkinv."""
        link = double_twoafc_link()
        for d in [0.5, 1.0, 2.0, 3.0]:
            p = link.linkinv(d)
            d_recovered = link.linkfun(p)
            assert_allclose(d_recovered, d, rtol=1e-4)

    def test_mu_eta_positive(self):
        """Derivative should be non-negative."""
        link = double_twoafc_link()
        for d in [0, 0.5, 1, 2, 3]:
            deriv = link.mu_eta(d)
            assert np.all(deriv >= 0)

    def test_known_value(self):
        """Test against known analytical value."""
        # For double 2-AFC: P = [Phi(d/sqrt(2))]^2
        # At d=sqrt(2), Phi(1) ≈ 0.8413, so P ≈ 0.8413^2 ≈ 0.7078
        link = double_twoafc_link()
        from scipy import stats
        d = np.sqrt(2)
        expected = stats.norm.cdf(1) ** 2
        assert_allclose(link.linkinv(d), expected, rtol=1e-6)


class TestDoubleDuoTrio:
    """Tests for double duo-trio link function."""

    def test_guessing_probability(self):
        """Test guessing probability is 0.25."""
        link = double_duotrio_link()
        assert link.p_guess == 0.25

    def test_linkinv_at_zero(self):
        """At d'=0, probability should be guessing."""
        link = double_duotrio_link()
        assert_allclose(link.linkinv(0), 0.25)

    def test_linkinv_positive(self):
        """Probability should increase with d'."""
        link = double_duotrio_link()
        probs = [link.linkinv(d) for d in [0, 1, 2, 3]]
        assert all(probs[i] < probs[i + 1] for i in range(len(probs) - 1))

    def test_linkfun_inverse(self):
        """linkfun should be inverse of linkinv."""
        link = double_duotrio_link()
        for d in [0.5, 1.0, 2.0]:
            p = link.linkinv(d)
            d_recovered = link.linkfun(p)
            assert_allclose(d_recovered, d, rtol=1e-3)


class TestDoubleTriangle:
    """Tests for double triangle link function."""

    def test_guessing_probability(self):
        """Test guessing probability is 1/9."""
        link = double_triangle_link()
        assert_allclose(link.p_guess, 1 / 9)

    def test_linkinv_at_zero(self):
        """At d'=0, probability should be guessing."""
        link = double_triangle_link()
        assert_allclose(link.linkinv(0), 1 / 9)

    def test_linkinv_positive(self):
        """Probability should increase with d'."""
        link = double_triangle_link()
        probs = [link.linkinv(d) for d in [0, 1, 2, 3]]
        assert all(probs[i] < probs[i + 1] for i in range(len(probs) - 1))

    def test_linkfun_inverse(self):
        """linkfun should be inverse of linkinv."""
        link = double_triangle_link()
        for d in [0.5, 1.0, 2.0]:
            p = link.linkinv(d)
            d_recovered = link.linkfun(p)
            assert_allclose(d_recovered, d, rtol=1e-3)


class TestDoubleThreeAFC:
    """Tests for double 3-AFC link function."""

    def test_guessing_probability(self):
        """Test guessing probability is 1/9."""
        link = double_threeafc_link()
        assert_allclose(link.p_guess, 1 / 9)

    def test_linkinv_at_zero(self):
        """At d'=0, probability should be guessing."""
        link = double_threeafc_link()
        assert_allclose(link.linkinv(0), 1 / 9, rtol=1e-3)

    def test_linkinv_positive(self):
        """Probability should increase with d'."""
        link = double_threeafc_link()
        probs = [link.linkinv(d) for d in [0, 1, 2, 3]]
        assert all(probs[i] < probs[i + 1] for i in range(len(probs) - 1))

    @pytest.mark.slow
    def test_linkfun_inverse(self):
        """linkfun should be inverse of linkinv."""
        link = double_threeafc_link()
        for d in [1.0, 2.0]:
            p = link.linkinv(d)
            d_recovered = link.linkfun(p)
            assert_allclose(d_recovered, d, rtol=1e-2)


class TestDoubleTetrad:
    """Tests for double tetrad link function."""

    def test_guessing_probability(self):
        """Test guessing probability is 1/9."""
        link = double_tetrad_link()
        assert_allclose(link.p_guess, 1 / 9)

    def test_linkinv_at_zero(self):
        """At d'=0, probability should be guessing."""
        link = double_tetrad_link()
        # At d=0, single tetrad probability is 1/3, so double is 1/9
        assert_allclose(link.linkinv(0), 1 / 9, rtol=1e-3)

    def test_linkinv_positive(self):
        """Probability should increase with d'."""
        link = double_tetrad_link()
        probs = [link.linkinv(d) for d in [0, 1, 2, 3]]
        assert all(probs[i] < probs[i + 1] for i in range(len(probs) - 1))

    @pytest.mark.slow
    def test_linkfun_inverse(self):
        """linkfun should be inverse of linkinv."""
        link = double_tetrad_link()
        for d in [1.0, 2.0]:
            p = link.linkinv(d)
            d_recovered = link.linkfun(p)
            assert_allclose(d_recovered, d, rtol=1e-2)


class TestGetDoubleLink:
    """Tests for get_double_link factory function."""

    @pytest.mark.parametrize("method", [
        "double_twoAFC", "doubletwoAFC", "double-twoAFC", "double2afc",
        "double_duotrio", "doubleduotrio",
        "double_triangle", "doubletriangle",
        "double_threeAFC", "doublethreeAFC", "double3afc",
        "double_tetrad", "doubletetrad",
    ])
    def test_valid_methods(self, method):
        """Test that valid method names work."""
        link = get_double_link(method)
        assert link is not None
        assert hasattr(link, "linkinv")
        assert hasattr(link, "linkfun")
        assert hasattr(link, "mu_eta")

    def test_invalid_method(self):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown double method"):
            get_double_link("invalid")


class TestDoubleLinksConsistency:
    """Cross-protocol consistency tests."""

    def test_all_links_have_same_interface(self):
        """All double links should have the same interface."""
        links = [
            double_twoafc_link(),
            double_duotrio_link(),
            double_triangle_link(),
            double_threeafc_link(),
            double_tetrad_link(),
        ]
        for link in links:
            assert hasattr(link, "linkinv")
            assert hasattr(link, "linkfun")
            assert hasattr(link, "mu_eta")
            assert hasattr(link, "p_guess")
            assert hasattr(link, "name")

    def test_vectorized_linkinv(self):
        """linkinv should work with arrays."""
        link = double_twoafc_link()
        d_values = np.array([0, 1, 2, 3])
        probs = link.linkinv(d_values)
        assert probs.shape == d_values.shape
        assert np.all(probs >= link.p_guess)
        assert np.all(probs <= 1.0)


class TestDoubleLinksGoldenValidation:
    """Golden validation tests against R sensR values."""

    @pytest.mark.parametrize("method,link_func", [
        ("double_twoAFC", double_twoafc_link),
        ("double_duotrio", double_duotrio_link),
        ("double_triangle", double_triangle_link),
        ("double_threeAFC", double_threeafc_link),
        ("double_tetrad", double_tetrad_link),
    ])
    def test_golden_values(self, method, link_func):
        """Test against R sensR golden values."""
        golden = GOLDEN["links"]["double_links"][method]
        link = link_func()

        for d, expected_p in zip(golden["d_prime"], golden["expected_pc"]):
            computed_p = link.linkinv(d)
            assert_allclose(
                computed_p, expected_p, rtol=1e-6,
                err_msg=f"{method} at d'={d}: expected {expected_p}, got {computed_p}"
            )
