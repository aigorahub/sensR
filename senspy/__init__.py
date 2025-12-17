"""
sensPy: Thurstonian Models for Sensory Discrimination in Python

A Python port of the R package sensR.

This package provides:
- Psychometric link functions for sensory discrimination protocols
- Utility functions for converting between pc, pd, and d-prime
- Statistical utilities for binomial tests

See docs/PORTING_PLAN.md for the full implementation roadmap.
"""

__version__ = "0.0.1-dev"
__author__ = "Aigora"

# Core types and result classes
from senspy.core.types import Protocol, Statistic, Alternative, parse_protocol
from senspy.core.base import DiscrimResult, RescaleResult

# Psychometric link functions
from senspy.links import (
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

# Utility functions
from senspy.utils import (
    delimit,
    normal_pvalue,
    find_critical,
    pc_to_pd,
    pd_to_pc,
    rescale,
)

# Main analysis functions
from senspy.discrim import discrim

# Power and sample size functions
from senspy.power import (
    discrim_power,
    dprime_power,
    discrim_sample_size,
    dprime_sample_size,
)

# Beta-binomial model
from senspy.betabin import betabin, BetaBinomialResult, BetaBinomialSummary

# 2-AC protocol
from senspy.twoac import twoac, TwoACResult

# Same-Different protocol
from senspy.samediff import samediff, SameDiffResult

# Degree of Difference (DOD) model
from senspy.dod import (
    dod,
    dod_fit,
    dod_sim,
    dod_power,
    DODResult,
    DODFitResult,
    DODControl,
    DODPowerResult,
    par2prob_dod,
    optimal_tau,
)

# D-prime hypothesis tests
from senspy.dprime_tests import (
    dprime_test,
    dprime_compare,
    posthoc,
    dprime_table,
    DprimeTestResult,
    DprimeCompareResult,
    PosthocResult,
)

# A-Not-A protocol
from senspy.anota import anota, ANotAResult

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Core types
    "Protocol",
    "Statistic",
    "Alternative",
    "parse_protocol",
    # Result classes
    "DiscrimResult",
    "RescaleResult",
    # Link functions
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
    # Utilities
    "delimit",
    "normal_pvalue",
    "find_critical",
    "pc_to_pd",
    "pd_to_pc",
    "rescale",
    # Analysis functions
    "discrim",
    # Power and sample size
    "discrim_power",
    "dprime_power",
    "discrim_sample_size",
    "dprime_sample_size",
    # Beta-binomial model
    "betabin",
    "BetaBinomialResult",
    "BetaBinomialSummary",
    # 2-AC protocol
    "twoac",
    "TwoACResult",
    # Same-Different protocol
    "samediff",
    "SameDiffResult",
    # DOD model
    "dod",
    "dod_fit",
    "dod_sim",
    "dod_power",
    "DODResult",
    "DODFitResult",
    "DODControl",
    "DODPowerResult",
    "par2prob_dod",
    "optimal_tau",
    # D-prime hypothesis tests
    "dprime_test",
    "dprime_compare",
    "posthoc",
    "dprime_table",
    "DprimeTestResult",
    "DprimeCompareResult",
    "PosthocResult",
    # A-Not-A protocol
    "anota",
    "ANotAResult",
]
