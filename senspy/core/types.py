"""Type definitions and enumerations for sensPy.

This module defines the core types used throughout the package, including
protocol enumerations and type aliases for numerical computations.
"""

from enum import Enum
from typing import Union

import numpy as np
from numpy.typing import ArrayLike, NDArray


# Type aliases
FloatArray = NDArray[np.floating]
FloatOrArray = Union[float, FloatArray]


class Protocol(str, Enum):
    """Sensory discrimination protocols.

    Each protocol corresponds to a specific psychometric function that maps
    d-prime (sensitivity) to proportion correct.

    Attributes
    ----------
    DUOTRIO : str
        Duo-trio test. Guessing probability = 1/2.
    TRIANGLE : str
        Triangle test. Guessing probability = 1/3.
    TWOAFC : str
        Two-alternative forced choice. Guessing probability = 1/2.
    THREEAFC : str
        Three-alternative forced choice. Guessing probability = 1/3.
    TETRAD : str
        Tetrad (unspecified) test. Guessing probability = 1/3.
    HEXAD : str
        Hexad test. Guessing probability = 1/10.
    TWOFIVE : str
        Two-out-of-five test. Guessing probability = 1/10.
    TWOFIVEF : str
        Two-out-of-five test (specified). Guessing probability = 2/5.

    Examples
    --------
    >>> from senspy.core.types import Protocol
    >>> Protocol.TRIANGLE
    <Protocol.TRIANGLE: 'triangle'>
    >>> Protocol.TRIANGLE.value
    'triangle'
    """

    DUOTRIO = "duotrio"
    TRIANGLE = "triangle"
    TWOAFC = "twoafc"
    THREEAFC = "threeafc"
    TETRAD = "tetrad"
    HEXAD = "hexad"
    TWOFIVE = "twofive"
    TWOFIVEF = "twofivef"

    @property
    def p_guess(self) -> float:
        """Return the guessing probability for this protocol.

        Returns
        -------
        float
            The probability of a correct response by chance alone.

        Examples
        --------
        >>> Protocol.TRIANGLE.p_guess
        0.3333333333333333
        >>> Protocol.TWOAFC.p_guess
        0.5
        """
        guessing_probs = {
            Protocol.DUOTRIO: 1 / 2,
            Protocol.TRIANGLE: 1 / 3,
            Protocol.TWOAFC: 1 / 2,
            Protocol.THREEAFC: 1 / 3,
            Protocol.TETRAD: 1 / 3,
            Protocol.HEXAD: 1 / 10,
            Protocol.TWOFIVE: 1 / 10,
            Protocol.TWOFIVEF: 2 / 5,
        }
        return guessing_probs[self]


class Statistic(str, Enum):
    """Test statistics for hypothesis testing.

    Attributes
    ----------
    EXACT : str
        Exact binomial test.
    LIKELIHOOD : str
        Likelihood ratio test.
    WALD : str
        Wald test (based on standard errors).
    SCORE : str
        Score test (Rao test).
    """

    EXACT = "exact"
    LIKELIHOOD = "likelihood"
    WALD = "wald"
    SCORE = "score"


class Alternative(str, Enum):
    """Alternative hypotheses for statistical tests.

    Attributes
    ----------
    TWO_SIDED : str
        Two-sided alternative (different from).
    GREATER : str
        One-sided alternative (greater than).
    LESS : str
        One-sided alternative (less than).
    """

    TWO_SIDED = "two.sided"
    GREATER = "greater"
    LESS = "less"


def parse_protocol(method: str | Protocol) -> Protocol:
    """Parse a protocol string or enum into a Protocol enum.

    Parameters
    ----------
    method : str or Protocol
        The protocol name (case-insensitive) or Protocol enum.

    Returns
    -------
    Protocol
        The corresponding Protocol enum value.

    Raises
    ------
    ValueError
        If the protocol name is not recognized.

    Examples
    --------
    >>> parse_protocol("triangle")
    <Protocol.TRIANGLE: 'triangle'>
    >>> parse_protocol("2AFC")
    <Protocol.TWOAFC: 'twoafc'>
    >>> parse_protocol(Protocol.DUOTRIO)
    <Protocol.DUOTRIO: 'duotrio'>
    """
    if isinstance(method, Protocol):
        return method

    # Normalize the input string
    normalized = method.lower().replace("-", "").replace("_", "")

    # Handle common aliases
    aliases = {
        "2afc": Protocol.TWOAFC,
        "3afc": Protocol.THREEAFC,
        "duo": Protocol.DUOTRIO,
        "tri": Protocol.TRIANGLE,
        "2outof5": Protocol.TWOFIVE,
        "2of5": Protocol.TWOFIVE,
        "2/5": Protocol.TWOFIVE,
        "2/5f": Protocol.TWOFIVEF,
    }

    if normalized in aliases:
        return aliases[normalized]

    # Try direct enum lookup
    try:
        return Protocol(normalized)
    except ValueError:
        valid = ", ".join(p.value for p in Protocol)
        raise ValueError(
            f"Unknown protocol: {method!r}. Valid protocols are: {valid}"
        ) from None
