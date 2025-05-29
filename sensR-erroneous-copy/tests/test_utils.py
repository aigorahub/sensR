from importlib.metadata import version as pkg_version

import senspy
from senspy import has_jax


def test_has_jax_boolean():
    assert isinstance(has_jax(), bool)


def test_version_matches_metadata():
    assert senspy.version() == pkg_version("senspy")
