"""Psychometric link functions for sensory discrimination protocols.

This module provides functions to convert between d-prime (sensitivity)
and proportion correct for various discrimination protocols.
"""

from senspy.links.psychometric import (
    psy_fun,
    psy_inv,
    psy_deriv,
    get_link,
    duotrio_link,
    triangle_link,
    twoafc_link,
    threeafc_link,
    tetrad_link,
    hexad_link,
    twofive_link,
    twofivef_link,
)

from senspy.links.double import (
    get_double_link,
    double_twoafc_link,
    double_duotrio_link,
    double_triangle_link,
    double_threeafc_link,
    double_tetrad_link,
    DoubleLinkResult,
)

__all__ = [
    # Single protocols
    "psy_fun",
    "psy_inv",
    "psy_deriv",
    "get_link",
    "duotrio_link",
    "triangle_link",
    "twoafc_link",
    "threeafc_link",
    "tetrad_link",
    "hexad_link",
    "twofive_link",
    "twofivef_link",
    # Double protocols
    "get_double_link",
    "double_twoafc_link",
    "double_duotrio_link",
    "double_triangle_link",
    "double_threeafc_link",
    "double_tetrad_link",
    "DoubleLinkResult",
]
