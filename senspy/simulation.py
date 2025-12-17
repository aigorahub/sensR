"""Simulation functions for sensory discrimination tests.

This module provides functions to simulate the outcomes of sensory discrimination
tests, useful for power analysis via simulation and methodological research.

References
----------
Brockhoff, P.B. and Christensen, R.H.B. (2010). Thurstonian models for sensory
    discrimination tests as generalized linear models. Food Quality and Preference,
    21, pp. 330-338.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from senspy.links import psy_fun, get_double_link
from senspy.utils import delimit


MethodType = Literal[
    "duotrio", "triangle", "twoAFC", "threeAFC", "tetrad", "hexad", "twofive", "twofiveF"
]


def discrim_sim(
    sample_size: int,
    replicates: int,
    d_prime: float,
    method: MethodType = "triangle",
    sd_indiv: float = 0.0,
    double: bool = False,
    random_state: int | np.random.Generator | None = None,
) -> NDArray[np.int_]:
    """Simulate replicated difference tests.

    Simulates the outcome of sensory difference tests for a given d-prime value
    and optional overdispersion (individual variability).

    Parameters
    ----------
    sample_size : int
        Number of subjects/assessors.
    replicates : int
        Number of replications per subject.
    d_prime : float
        The true d-prime value. Must be non-negative.
    method : str, default="triangle"
        The discrimination protocol. One of: "duotrio", "triangle", "twoAFC",
        "threeAFC", "tetrad", "hexad", "twofive", "twofiveF".
    sd_indiv : float, default=0.0
        Standard deviation of individual d-prime values around the mean.
        A value of 0 corresponds to complete independence (no overdispersion).
    double : bool, default=False
        Use the 'double' variant of the protocol. Not implemented for
        "twofive", "twofiveF", and "hexad".
    random_state : int, Generator, or None, default=None
        Random number generator seed or instance for reproducibility.

    Returns
    -------
    NDArray[np.int_]
        Array of length `sample_size` with the number of correct answers
        for each subject.

    Raises
    ------
    ValueError
        If parameters are invalid.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> discrim_sim(sample_size=10, replicates=3, d_prime=2, method="triangle")
    array([2, 2, 3, 2, 2, 1, 2, 2, 2, 2])

    >>> # With individual variability
    >>> discrim_sim(sample_size=10, replicates=3, d_prime=2,
    ...             method="triangle", sd_indiv=1)
    array([2, 3, 2, 3, 3, 2, 3, 2, 3, 0])

    Notes
    -----
    The d-prime for each subject is drawn from a normal distribution with
    mean `d_prime` and standard deviation `sd_indiv`. All negative values
    are clipped to zero.
    """
    # Validate inputs
    if not isinstance(sample_size, int) or sample_size <= 0:
        raise ValueError("'sample_size' must be a positive integer")
    if not isinstance(replicates, int) or replicates < 0:
        raise ValueError("'replicates' must be a non-negative integer")
    if d_prime < 0:
        raise ValueError("'d_prime' must be non-negative")
    if sd_indiv < 0:
        raise ValueError("'sd_indiv' must be non-negative")

    valid_methods = [
        "duotrio", "triangle", "twoAFC", "threeAFC", "tetrad",
        "hexad", "twofive", "twofiveF"
    ]
    if method not in valid_methods:
        raise ValueError(f"Invalid method '{method}'. Must be one of {valid_methods}")

    if double and method in ["hexad", "twofive", "twofiveF"]:
        raise ValueError(
            f"'double' method for '{method}' is not implemented"
        )

    # Set up random number generator
    rng = np.random.default_rng(random_state)

    # Individual deviations in d-prime
    d_indiv = rng.normal(loc=0, scale=sd_indiv, size=sample_size)
    q = delimit(d_prime + d_indiv, lower=0)  # Individual d-primes, clipped to >= 0

    # Compute probability of correct answer for each subject
    if double:
        double_link = get_double_link(f"double_{method}")
        prob = np.array([double_link.linkinv(d) for d in q])
    else:
        prob = psy_fun(q, method=method)

    # Simulate number of correct answers
    n_correct = rng.binomial(n=replicates, p=prob, size=sample_size)

    return n_correct


@dataclass
class SameDiffSimResult:
    """Result from same-different simulation.

    Attributes
    ----------
    ss : NDArray[np.int_]
        Same-answers to same-samples.
    ds : NDArray[np.int_]
        Different-answers to same-samples.
    sd : NDArray[np.int_]
        Same-answers to different-samples.
    dd : NDArray[np.int_]
        Different-answers to different-samples.
    """

    ss: NDArray[np.int_]
    ds: NDArray[np.int_]
    sd: NDArray[np.int_]
    dd: NDArray[np.int_]

    def to_array(self) -> NDArray[np.int_]:
        """Return as (n, 4) array with columns [ss, ds, sd, dd]."""
        return np.column_stack([self.ss, self.ds, self.sd, self.dd])


def samediff_sim(
    n: int,
    tau: float,
    delta: float,
    Ns: int,
    Nd: int,
    random_state: int | np.random.Generator | None = None,
) -> SameDiffSimResult:
    """Simulate data from same-different tests.

    Simulates the outcome of n same-different experiments with given
    tau (decision criterion) and delta (d-prime) values.

    Parameters
    ----------
    n : int
        Number of experiments to simulate.
    tau : float
        Decision criterion (tau parameter).
    delta : float
        Sensitivity parameter (d-prime).
    Ns : int
        Number of same-sample pairs per experiment.
    Nd : int
        Number of different-sample pairs per experiment.
    random_state : int, Generator, or None, default=None
        Random number generator seed or instance for reproducibility.

    Returns
    -------
    SameDiffSimResult
        Result object with attributes ss, ds, sd, dd representing the
        four response categories.

    Examples
    --------
    >>> import numpy as np
    >>> result = samediff_sim(n=10, tau=1, delta=1, Ns=10, Nd=10, random_state=42)
    >>> result.ss
    array([9, 9, 6, 7, 8, 8, 6, 7, 6, 9])
    >>> result.to_array()
    array([[ 9,  1,  5,  5],
           [ 9,  1,  4,  6],
           ...])

    Notes
    -----
    The probabilities are computed as:
    - P(same|same) = 2 * Phi(tau/sqrt(2)) - 1
    - P(same|different) = Phi((tau-delta)/sqrt(2)) - Phi((-tau-delta)/sqrt(2))

    References
    ----------
    Christensen, R.H.B., Brockhoff, P.B. (2009). Estimation and inference in
        the same-different test. Food, Quality and Preference, 20, pp. 514-520.
    """
    # Set up random number generator
    rng = np.random.default_rng(random_state)

    # Probability of "same" answer to same-samples
    sqrt_2 = np.sqrt(2)
    pss = 2 * stats.norm.cdf(tau / sqrt_2) - 1

    # Simulate same-answers to same-samples
    ss = rng.binomial(n=Ns, p=pss, size=n)
    ds = Ns - ss

    # Probability of "same" answer to different-samples
    psd = stats.norm.cdf((tau - delta) / sqrt_2) - stats.norm.cdf((-tau - delta) / sqrt_2)

    # Simulate same-answers to different-samples
    sd = rng.binomial(n=Nd, p=psd, size=n)
    dd = Nd - sd

    return SameDiffSimResult(ss=ss, ds=ds, sd=sd, dd=dd)
