"""Pytest configuration and fixtures for sensPy test suite."""

import json
from pathlib import Path

import pytest


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def golden_links_data():
    """Load golden data for link function tests from sensR.

    This fixture provides validated reference values from the sensR R package
    for testing psy_fun, psy_inv, and psy_deriv functions.
    """
    filepath = FIXTURES_DIR / "golden_links.json"
    if filepath.exists():
        with open(filepath) as f:
            return json.load(f)
    return None


@pytest.fixture
def golden_rescale_data():
    """Load golden data for rescale function tests from sensR."""
    filepath = FIXTURES_DIR / "golden_rescale.json"
    if filepath.exists():
        with open(filepath) as f:
            return json.load(f)
    return None


@pytest.fixture
def tolerance():
    """Standard numerical tolerances for tests.

    Returns a dict with tolerances for different types of values:
    - coefficients: 1e-6 (point estimates)
    - probabilities: 1e-6 (pc, pd values)
    - derivatives: 1e-5 (less accurate due to numerical computation)
    """
    return {
        "coefficients": 1e-6,
        "probabilities": 1e-6,
        "derivatives": 1e-5,
        "p_values": 1e-8,
    }
