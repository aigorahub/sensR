"""Pytest configuration and fixtures for sensPy test suite."""

import json
from pathlib import Path

import pytest


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def golden_sensr():
    """Load comprehensive golden data from sensR.

    This fixture provides validated reference values from the sensR R package
    for testing all sensPy functions.
    """
    filepath = FIXTURES_DIR / "golden_sensr.json"
    if filepath.exists():
        with open(filepath) as f:
            return json.load(f)
    return None


@pytest.fixture
def golden_links_data(golden_sensr):
    """Load golden data for link function tests from sensR."""
    if golden_sensr is not None:
        return golden_sensr.get("links")
    # Fallback to old file
    filepath = FIXTURES_DIR / "golden_links.json"
    if filepath.exists():
        with open(filepath) as f:
            return json.load(f)
    return None


@pytest.fixture
def golden_discrim_data(golden_sensr):
    """Load golden data for discrim function tests from sensR."""
    if golden_sensr is not None:
        return golden_sensr.get("discrim")
    return None


@pytest.fixture
def golden_rescale_data(golden_sensr):
    """Load golden data for rescale function tests from sensR."""
    if golden_sensr is not None:
        return golden_sensr.get("rescale")
    return None


@pytest.fixture
def tolerance():
    """Standard numerical tolerances for tests.

    Returns a dict with tolerances for different types of values.
    Note: Some protocols (hexad, twofive) use approximations so need looser tolerance.
    """
    return {
        "coefficients": 1e-3,  # d-prime estimates
        "probabilities": 1e-3,  # pc, pd values
        "derivatives": 1e-2,  # less accurate due to numerical computation
        "p_values": 1e-4,
        "strict": 1e-6,  # for exact protocols like twoafc
    }
