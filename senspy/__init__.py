"""sensPy - Python port of sensR."""

from .links import psyfun, psyinv, psyderiv, rescale
from .models import BetaBinomial
from .discrimination import (
    two_afc,
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
)
from .power import beta_binomial_power
from .plotting import plot_psychometric
from .utils import has_jax, version

__all__ = [
    "psyfun",
    "psyinv",
    "psyderiv",
    "rescale",
    "BetaBinomial",
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
    "beta_binomial_power",
    "plot_psychometric",
    "has_jax",
    "version",
]
