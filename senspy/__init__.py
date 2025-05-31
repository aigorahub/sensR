"""sensPy - Python port of sensR."""

# psyfun, psyinv, psyderiv, rescale are now in discrimination.py
# from .links import psyfun, psyinv, psyderiv, rescale 
from .models import BetaBinomial, TwoACModel # Added TwoACModel here
from .discrimination import (
    two_afc,
    # Add psyfun, psyinv, psyderiv, rescale to this import list
    psyfun,
    psyinv,
    psyderiv,
    rescale,
    duotrio_pc,
    three_afc_pc,
    triangle_pc,
    tetrad_pc,
    hexad_pc,
    twofive_pc,
    get_pguess,
    pc2pd,
    pd2pc,
    discrim_2afc,
    discrim,
    twoAC,
    dod,
    samediff,
    # dprime_test, # Definition is missing in discrimination.py
    # dprime_compare, # Definition is missing in discrimination.py
    SDT,
    AUC,
    par2prob_dod,
    dod_nll,
    samediff_nll,
)
# beta_binomial_power is not defined in senspy.power.py
from .power import find_critical_binomial_value, exact_binomial_power, sample_size_for_binomial_power, power_discrim
from .plotting import plot_psychometric
from .utils import has_jax, version

__all__ = [
    "psyfun",
    "psyinv",
    "psyderiv",
    "rescale",
    "BetaBinomial",
    "TwoACModel", # Added TwoACModel here
    "two_afc",
    "duotrio_pc",
    "three_afc_pc",
    "triangle_pc",
    "tetrad_pc",
    "hexad_pc",
    "twofive_pc",
    "get_pguess",
    "pc2pd",
    "pd2pc",
    "discrim_2afc",
    "discrim",
    "twoAC",
    "dod",
    "samediff",
    # "dprime_test", # Definition is missing in discrimination.py
    # "dprime_compare", # Definition is missing in discrimination.py
    "SDT",
    "AUC",
    "par2prob_dod",
    "dod_nll",
    "samediff_nll",
    # "beta_binomial_power", # Not defined in senspy.power.py and removed from import above
    "find_critical_binomial_value",
    "exact_binomial_power",
    "sample_size_for_binomial_power",
    "power_discrim",
    "plot_psychometric",
    "has_jax",
    "version",
]
