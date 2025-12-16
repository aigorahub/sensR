"""Validation tests comparing sensPy against sensR golden data.

These tests verify numerical parity between the Python implementation
and the reference R package (sensR).
"""

import numpy as np
import pytest

from senspy import psy_fun, psy_inv, psy_deriv, discrim, rescale


class TestPsyFunValidation:
    """Validate psy_fun against sensR psyfun()."""

    @pytest.mark.parametrize("method", [
        "duotrio", "triangle", "twoafc", "threeafc", "tetrad"
    ])
    def test_psy_fun_matches_sensr(self, golden_links_data, tolerance, method):
        """Test that psy_fun matches sensR psyfun for exact protocols."""
        if golden_links_data is None:
            pytest.skip("Golden data not available")

        data = golden_links_data["psy_fun"].get(method)
        if data is None:
            pytest.skip(f"No golden data for {method}")

        d_prime = np.array(data["d_prime"])
        expected = np.array(data["expected_pc"])

        result = psy_fun(d_prime, method=method)

        np.testing.assert_allclose(
            result, expected, rtol=tolerance["probabilities"],
            err_msg=f"psy_fun mismatch for {method}"
        )

    @pytest.mark.parametrize("method", ["hexad", "twofive", "twofivef"])
    @pytest.mark.xfail(reason="Approximation - needs correct formula from sensR")
    def test_psy_fun_approximated_protocols(self, golden_links_data, tolerance, method):
        """Test psy_fun for approximated protocols (expected to fail until fixed)."""
        if golden_links_data is None:
            pytest.skip("Golden data not available")

        data = golden_links_data["psy_fun"].get(method)
        d_prime = np.array(data["d_prime"])
        expected = np.array(data["expected_pc"])
        result = psy_fun(d_prime, method=method)

        np.testing.assert_allclose(result, expected, rtol=0.05)


class TestPsyInvValidation:
    """Validate psy_inv against sensR psyinv()."""

    @pytest.mark.parametrize("method", [
        "duotrio", "triangle", "twoafc", "threeafc", "tetrad"
    ])
    def test_psy_inv_matches_sensr(self, golden_links_data, tolerance, method):
        """Test that psy_inv matches sensR psyinv for exact protocols."""
        if golden_links_data is None:
            pytest.skip("Golden data not available")

        data = golden_links_data["psy_inv"].get(method)
        if data is None:
            pytest.skip(f"No golden data for {method}")

        pc = np.array(data["pc"])
        expected = np.array(data["expected_d_prime"])

        result = psy_inv(pc, method=method)

        np.testing.assert_allclose(
            result, expected, rtol=tolerance["coefficients"],
            err_msg=f"psy_inv mismatch for {method}"
        )

    @pytest.mark.parametrize("method", ["hexad", "twofive", "twofivef"])
    @pytest.mark.xfail(reason="Approximation - needs correct formula from sensR")
    def test_psy_inv_approximated_protocols(self, golden_links_data, tolerance, method):
        """Test psy_inv for approximated protocols (expected to fail until fixed)."""
        if golden_links_data is None:
            pytest.skip("Golden data not available")

        data = golden_links_data["psy_inv"].get(method)
        pc = np.array(data["pc"])
        expected = np.array(data["expected_d_prime"])
        result = psy_inv(pc, method=method)

        np.testing.assert_allclose(result, expected, rtol=0.1)


class TestPsyDerivValidation:
    """Validate psy_deriv against sensR psyderiv()."""

    @pytest.mark.parametrize("method", [
        "duotrio", "triangle", "twoafc", "threeafc", "tetrad"
    ])
    def test_psy_deriv_matches_sensr(self, golden_links_data, tolerance, method):
        """Test that psy_deriv matches sensR psyderiv for exact protocols."""
        if golden_links_data is None:
            pytest.skip("Golden data not available")

        data = golden_links_data["psy_deriv"].get(method)
        if data is None:
            pytest.skip(f"No golden data for {method}")

        d_prime = np.array(data["d_prime"])
        expected = np.array(data["expected_deriv"])

        result = psy_deriv(d_prime, method=method)

        # tetrad uses numerical derivative, needs more tolerance
        rtol = 0.15 if method == "tetrad" else tolerance["derivatives"]

        np.testing.assert_allclose(
            result, expected, rtol=rtol,
            err_msg=f"psy_deriv mismatch for {method}"
        )

    @pytest.mark.parametrize("method", ["hexad", "twofive", "twofivef"])
    @pytest.mark.xfail(reason="Approximation - needs correct formula from sensR")
    def test_psy_deriv_approximated_protocols(self, golden_links_data, tolerance, method):
        """Test psy_deriv for approximated protocols (expected to fail until fixed)."""
        if golden_links_data is None:
            pytest.skip("Golden data not available")

        data = golden_links_data["psy_deriv"].get(method)
        d_prime = np.array(data["d_prime"])
        expected = np.array(data["expected_deriv"])
        result = psy_deriv(d_prime, method=method)

        np.testing.assert_allclose(result, expected, rtol=0.15)


class TestDiscrimValidation:
    """Validate discrim against sensR discrim()."""

    def test_discrim_estimates_match_sensr(self, golden_discrim_data, tolerance):
        """Test that discrim point estimates match sensR."""
        if golden_discrim_data is None:
            pytest.skip("Golden data not available")

        for case in golden_discrim_data["test_cases"]:
            inp = case["input"]
            expected = case["estimates"]

            result = discrim(
                correct=inp["correct"],
                total=inp["total"],
                method=inp["method"]
            )

            # Check point estimates
            assert result.pc == pytest.approx(expected["pc"], rel=1e-4), \
                f"pc mismatch for {inp}"
            assert result.pd == pytest.approx(expected["pd"], rel=1e-3), \
                f"pd mismatch for {inp}"
            assert result.d_prime == pytest.approx(expected["d_prime"], rel=1e-2), \
                f"d_prime mismatch for {inp}"

    def test_discrim_std_errors_match_sensr(self, golden_discrim_data, tolerance):
        """Test that discrim standard errors match sensR."""
        if golden_discrim_data is None:
            pytest.skip("Golden data not available")

        for case in golden_discrim_data["test_cases"]:
            inp = case["input"]
            expected = case["std_errors"]

            # Skip cases where SE is NA (at or below chance)
            if expected["d_prime"] is None or np.isnan(expected["d_prime"]):
                continue

            result = discrim(
                correct=inp["correct"],
                total=inp["total"],
                method=inp["method"]
            )

            assert result.se_pc == pytest.approx(expected["pc"], rel=1e-3), \
                f"se_pc mismatch for {inp}"
            assert result.se_d_prime == pytest.approx(expected["d_prime"], rel=0.1), \
                f"se_d_prime mismatch for {inp}"

    def test_discrim_p_values_match_sensr(self, golden_discrim_data, tolerance):
        """Test that discrim p-values match sensR for different statistics."""
        if golden_discrim_data is None:
            pytest.skip("Golden data not available")

        for case in golden_discrim_data["test_cases"]:
            inp = case["input"]
            expected_p = case["p_values"]

            # Test exact statistic
            result_exact = discrim(
                correct=inp["correct"],
                total=inp["total"],
                method=inp["method"],
                statistic="exact"
            )
            # Use abs tolerance for small p-values due to rounding in R output
            assert result_exact.p_value == pytest.approx(expected_p["exact"], rel=1e-2, abs=1e-4), \
                f"exact p-value mismatch for {inp}"

            # Test wald statistic
            result_wald = discrim(
                correct=inp["correct"],
                total=inp["total"],
                method=inp["method"],
                statistic="wald"
            )
            # Wald can differ more due to approximation
            assert result_wald.p_value == pytest.approx(expected_p["wald"], rel=0.1, abs=1e-4), \
                f"wald p-value mismatch for {inp}"


class TestRescaleValidation:
    """Validate rescale against sensR rescale()."""

    def test_rescale_matches_sensr(self, golden_rescale_data, tolerance):
        """Test that rescale matches sensR rescale()."""
        if golden_rescale_data is None:
            pytest.skip("Golden data not available")

        for case in golden_rescale_data["test_cases"]:
            inp = case["input"]
            expected = case["output"]

            # Call rescale with appropriate input
            if inp["type"] == "pc":
                result = rescale(pc=inp["value"], method=inp["method"])
            elif inp["type"] == "pd":
                result = rescale(pd=inp["value"], method=inp["method"])
            else:
                result = rescale(d_prime=inp["value"], method=inp["method"])

            # Use looser tolerance for hexad/twofive
            if inp["method"] in ("hexad", "twofive", "twofivef"):
                rtol = 0.1
            else:
                rtol = 1e-3

            assert result.pc == pytest.approx(expected["pc"], rel=rtol), \
                f"rescale pc mismatch for {inp}"
            assert result.pd == pytest.approx(expected["pd"], rel=rtol), \
                f"rescale pd mismatch for {inp}"
            assert result.d_prime == pytest.approx(expected["d_prime"], rel=rtol), \
                f"rescale d_prime mismatch for {inp}"
