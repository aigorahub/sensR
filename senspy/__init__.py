"""
sensPy: Thurstonian Models for Sensory Discrimination in Python

.. warning::
    **BETA SOFTWARE** - This package is currently in Beta and under active
    development. While we strive for numerical parity with the R package sensR,
    results should be independently validated before use in production systems
    or critical decision-making. We welcome testers to compare outputs against
    sensR and report any discrepancies.

    Contact info@aigora.com to connect with the Aigora team.

A Python port of the R package sensR providing:
- Discrimination analysis (discrim, betabin, twoac, samediff, dod)
- Psychometric link functions for all standard protocols
- Power analysis and sample size calculation
- D-prime inference and comparison tools
- ROC/AUC analysis and visualization
"""

__version__ = "0.1.0"
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
    # Double protocols
    get_double_link,
    double_twoafc_link,
    double_duotrio_link,
    double_triangle_link,
    double_threeafc_link,
    double_tetrad_link,
    DoubleLinkResult,
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

# ROC, AUC, and Signal Detection Theory
from senspy.roc import sdt, roc, auc, ROCResult, AUCResult

# Plotting functions (Plotly-based)
from senspy.plotting import (
    plot_roc,
    plot_psychometric,
    plot_psychometric_comparison,
    plot_sdt_distributions,
    plot_profile_likelihood,
    plot_power_curve,
    plot_sample_size_curve,
)

# Simulation functions
from senspy.simulation import discrim_sim, samediff_sim, SameDiffSimResult

# Protocol-specific power functions
from senspy.protocol_power import samediff_power, twoac_power, TwoACPowerResult

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
    # Double protocol links
    "get_double_link",
    "double_twoafc_link",
    "double_duotrio_link",
    "double_triangle_link",
    "double_threeafc_link",
    "double_tetrad_link",
    "DoubleLinkResult",
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
    # ROC, AUC, SDT
    "sdt",
    "roc",
    "auc",
    "ROCResult",
    "AUCResult",
    # Plotting
    "plot_roc",
    "plot_psychometric",
    "plot_psychometric_comparison",
    "plot_sdt_distributions",
    "plot_profile_likelihood",
    "plot_power_curve",
    "plot_sample_size_curve",
    # Simulation
    "discrim_sim",
    "samediff_sim",
    "SameDiffSimResult",
    # Protocol-specific power
    "samediff_power",
    "twoac_power",
    "TwoACPowerResult",
]
