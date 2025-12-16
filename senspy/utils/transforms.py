"""Transformation utilities for sensPy.

This module provides functions for converting between different scales
used in sensory discrimination:
- pc: proportion correct
- pd: proportion of discriminators
- d_prime: sensitivity (d')

These functions correspond to utilities in sensR's utils.R file.
"""

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from senspy.core.types import Protocol, parse_protocol
from senspy.core.base import RescaleResult


def pc_to_pd(
    pc: ArrayLike,
    p_guess: float,
) -> NDArray[np.floating]:
    """Convert proportion correct to proportion of discriminators.

    Parameters
    ----------
    pc : array_like
        Proportion correct. Values should be in [p_guess, 1].
    p_guess : float
        Guessing probability (chance level).

    Returns
    -------
    ndarray
        Proportion of discriminators. Values in [0, 1].

    Notes
    -----
    The relationship is: pd = (pc - p_guess) / (1 - p_guess)

    Values of pc below p_guess are mapped to pd = 0.

    Corresponds to `pc2pd()` in sensR's utils.R.

    Examples
    --------
    >>> pc_to_pd(0.8, p_guess=0.5)
    array([0.6])
    >>> pc_to_pd(0.6, p_guess=1/3)
    array([0.4])
    """
    pc = np.atleast_1d(np.asarray(pc, dtype=np.float64))

    if not 0 <= p_guess <= 1:
        raise ValueError(f"p_guess must be in [0, 1], got {p_guess}")

    pd = (pc - p_guess) / (1 - p_guess)
    pd = np.maximum(pd, 0.0)  # pd cannot be negative

    return pd


def pd_to_pc(
    pd: ArrayLike,
    p_guess: float,
) -> NDArray[np.floating]:
    """Convert proportion of discriminators to proportion correct.

    Parameters
    ----------
    pd : array_like
        Proportion of discriminators. Values should be in [0, 1].
    p_guess : float
        Guessing probability (chance level).

    Returns
    -------
    ndarray
        Proportion correct. Values in [p_guess, 1].

    Notes
    -----
    The relationship is: pc = p_guess + pd * (1 - p_guess)

    Corresponds to `pd2pc()` in sensR's utils.R.

    Examples
    --------
    >>> pd_to_pc(0.6, p_guess=0.5)
    array([0.8])
    >>> pd_to_pc(0.4, p_guess=1/3)
    array([0.6])
    """
    pd = np.atleast_1d(np.asarray(pd, dtype=np.float64))

    if not 0 <= p_guess <= 1:
        raise ValueError(f"p_guess must be in [0, 1], got {p_guess}")

    pc = p_guess + pd * (1 - p_guess)

    return pc


def rescale(
    pc: ArrayLike | None = None,
    pd: ArrayLike | None = None,
    d_prime: ArrayLike | None = None,
    se: ArrayLike | None = None,
    method: str | Protocol = "triangle",
) -> RescaleResult:
    """Convert between pc, pd, and d-prime scales.

    Provide exactly one of `pc`, `pd`, or `d_prime`, and optionally
    a standard error. The function will compute the other two scales.

    Parameters
    ----------
    pc : array_like, optional
        Proportion correct.
    pd : array_like, optional
        Proportion of discriminators.
    d_prime : array_like, optional
        Sensitivity (d').
    se : array_like, optional
        Standard error of the provided parameter. If given, standard
        errors for the other parameters will be computed via the
        delta method.
    method : str or Protocol, default "triangle"
        Discrimination protocol. Determines the psychometric function
        used for conversion.

    Returns
    -------
    RescaleResult
        Result object with pc, pd, d_prime, and optionally standard errors.

    Notes
    -----
    Corresponds to `rescale()` in sensR's utils.R.

    The conversion uses the psychometric function for the given protocol:
    - pc = psyfun(d_prime, method)
    - d_prime = psyinv(pc, method)
    - pd = (pc - p_guess) / (1 - p_guess)

    Standard errors are propagated using the delta method.

    Examples
    --------
    >>> result = rescale(d_prime=1.5, method="triangle")
    >>> result.pc
    0.628...
    >>> result.pd
    0.442...

    >>> result = rescale(pc=0.8, method="twoafc")
    >>> result.d_prime
    1.683...

    >>> result = rescale(d_prime=1.5, se=0.2, method="triangle")
    >>> result.se_pc
    0.058...
    """
    # Import here to avoid circular imports
    from senspy.links import psy_fun, psy_inv, psy_deriv

    # Validate inputs: exactly one of pc, pd, d_prime must be provided
    provided = sum(x is not None for x in [pc, pd, d_prime])
    if provided != 1:
        raise ValueError(
            "Exactly one of pc, pd, or d_prime must be provided. "
            f"Got {provided} arguments."
        )

    protocol = parse_protocol(method)
    p_guess = protocol.p_guess

    # Convert to arrays
    if pc is not None:
        pc = np.atleast_1d(np.asarray(pc, dtype=np.float64))
        if se is not None:
            se = np.atleast_1d(np.asarray(se, dtype=np.float64))
            se_pc = se
    elif pd is not None:
        pd = np.atleast_1d(np.asarray(pd, dtype=np.float64))
        if se is not None:
            se = np.atleast_1d(np.asarray(se, dtype=np.float64))
            se_pd = se
    else:  # d_prime is not None
        d_prime = np.atleast_1d(np.asarray(d_prime, dtype=np.float64))
        if se is not None:
            se = np.atleast_1d(np.asarray(se, dtype=np.float64))
            se_d_prime = se

    # Initialize standard errors as None
    se_pc_out = None
    se_pd_out = None
    se_d_prime_out = None

    # Compute conversions
    if pc is not None:
        # pc -> pd, d_prime
        # Restrict pc to be >= p_guess
        pc_adj = np.maximum(pc, p_guess)
        pd_out = pc_to_pd(pc_adj, p_guess)
        d_prime_out = psy_inv(pc_adj, method=protocol)
        pc_out = pc_adj

        if se is not None:
            se_pc_out = se.copy()
            # Where pc was below p_guess, SE is undefined
            se_pc_out[pc < p_guess] = np.nan
            se_pd_out = se_pc_out / (1 - p_guess)
            # SE of d_prime via delta method: se_d = se_pc / |d(pc)/d(d')|
            deriv = psy_deriv(d_prime_out, method=protocol)
            se_d_prime_out = se_pc_out / deriv

    elif pd is not None:
        # pd -> pc, d_prime
        pc_out = pd_to_pc(pd, p_guess)
        d_prime_out = psy_inv(pc_out, method=protocol)
        pd_out = pd

        if se is not None:
            se_pd_out = se.copy()
            se_pc_out = se_pd_out * (1 - p_guess)
            deriv = psy_deriv(d_prime_out, method=protocol)
            se_d_prime_out = se_pc_out / deriv

    else:  # d_prime is not None
        # d_prime -> pc, pd
        d_prime = np.maximum(d_prime, 0.0)  # d_prime cannot be negative
        pc_out = psy_fun(d_prime, method=protocol)
        pd_out = pc_to_pd(pc_out, p_guess)
        d_prime_out = d_prime

        if se is not None:
            se_d_prime_out = se.copy()
            deriv = psy_deriv(d_prime_out, method=protocol)
            se_pc_out = se_d_prime_out * deriv
            se_pd_out = se_pc_out / (1 - p_guess)

    # Return scalar if input was scalar
    if pc_out.size == 1:
        pc_out = float(pc_out[0])
        pd_out = float(pd_out[0])
        d_prime_out = float(d_prime_out[0])
        if se_pc_out is not None:
            se_pc_out = float(se_pc_out[0])
            se_pd_out = float(se_pd_out[0])
            se_d_prime_out = float(se_d_prime_out[0])

    return RescaleResult(
        pc=pc_out,
        pd=pd_out,
        d_prime=d_prime_out,
        method=protocol,
        se_pc=se_pc_out,
        se_pd=se_pd_out,
        se_d_prime=se_d_prime_out,
    )
