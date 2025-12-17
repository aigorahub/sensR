"""Double protocol psychometric link functions.

Double protocols involve two independent presentations of the same discrimination
task. The probability of two correct responses is the square of the single
protocol probability, but accounting for the specific structure of each protocol.

References
----------
Ennis, J.M. & Jesionka, V. (2011). The power of sensory discrimination methods
    revisited. Journal of Sensory Studies, 26(5), 371-382.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy import integrate, optimize, stats


@dataclass
class DoubleLinkResult:
    """Result from double link function operations."""

    linkinv: Callable[[NDArray | float], NDArray | float]
    linkfun: Callable[[NDArray | float], NDArray | float]
    mu_eta: Callable[[NDArray | float], NDArray | float]
    p_guess: float
    name: str


def _make_double_link(
    name: str,
    p_guess: float,
    linkinv_func: callable,
    mu_eta_func: callable,
    linkfun_bounds: tuple[float, float] = (0, 15),
) -> DoubleLinkResult:
    """Factory function to create double link objects with consistent interface."""

    def linkfun(mu: NDArray | float) -> NDArray | float:
        """Inverse link: probability to d-prime."""
        mu = np.atleast_1d(np.asarray(mu, dtype=np.float64))
        result = np.zeros_like(mu)

        eps = 1e-8
        ok = (mu > p_guess) & (mu < 1 - eps)
        result[mu <= p_guess] = 0.0
        result[mu >= 1 - eps] = np.inf

        if np.any(ok):
            for i in np.where(ok)[0]:

                def root_func(d: float, target: float) -> float:
                    p = linkinv_func(d)
                    # Handle both scalar and array returns
                    if isinstance(p, np.ndarray):
                        p = float(p.flat[0])
                    return p - target

                try:
                    result[i] = optimize.brentq(
                        root_func, linkfun_bounds[0], linkfun_bounds[1],
                        args=(mu[i],)
                    )
                except ValueError:
                    result[i] = np.nan

        return float(result[0]) if result.size == 1 else result

    return DoubleLinkResult(
        linkinv=linkinv_func,
        linkfun=linkfun,
        mu_eta=mu_eta_func,
        p_guess=p_guess,
        name=name,
    )


def double_twoafc_link() -> DoubleLinkResult:
    """Double 2-AFC link function.

    The double 2-AFC protocol has two independent 2-AFC trials.
    Guessing probability is 0.5^2 = 0.25.

    Returns
    -------
    DoubleLinkResult
        Link function object with linkinv, linkfun, and mu_eta methods.
    """
    p_guess = 0.25

    def linkinv(eta: NDArray | float) -> NDArray | float:
        """d-prime to probability for double 2-AFC."""
        eta = np.atleast_1d(np.asarray(eta, dtype=np.float64))
        result = np.full_like(eta, p_guess)

        ok = eta > 0
        if np.any(ok):
            result[ok] = stats.norm.cdf(eta[ok] / np.sqrt(2)) ** 2

        result = np.clip(result, p_guess, 1.0)
        return float(result[0]) if result.size == 1 else result

    def mu_eta(eta: NDArray | float) -> NDArray | float:
        """Derivative of linkinv for double 2-AFC."""
        eta = np.atleast_1d(np.asarray(eta, dtype=np.float64))
        result = np.zeros_like(eta)

        ok = eta > 0
        if np.any(ok):
            sqrt_2 = np.sqrt(2)
            eta_ok = eta[ok]
            pnorm_val = stats.norm.cdf(eta_ok / sqrt_2)
            dnorm_val = stats.norm.pdf(eta_ok / sqrt_2)
            result[ok] = sqrt_2 * pnorm_val * dnorm_val

        return np.maximum(result, 0.0)

    return _make_double_link("double_twoAFC", p_guess, linkinv, mu_eta)


def double_duotrio_link() -> DoubleLinkResult:
    """Double duo-trio link function.

    The double duo-trio protocol has two independent duo-trio trials.
    Guessing probability is 0.5^2 = 0.25.

    Returns
    -------
    DoubleLinkResult
        Link function object with linkinv, linkfun, and mu_eta methods.
    """
    p_guess = 0.25

    def linkinv(eta: NDArray | float) -> NDArray | float:
        """d-prime to probability for double duo-trio."""
        eta = np.atleast_1d(np.asarray(eta, dtype=np.float64))
        result = np.full_like(eta, p_guess)

        ok = (eta > 0) & (eta < 20)
        result[eta >= 20] = 1.0

        if np.any(ok):
            eta_ok = eta[ok]
            pnorm_eta_2 = stats.norm.cdf(eta_ok * np.sqrt(0.5))
            pnorm_eta_6 = stats.norm.cdf(eta_ok * np.sqrt(1 / 6))
            # Single duo-trio probability
            p_single = 1 - pnorm_eta_2 - pnorm_eta_6 + 2 * pnorm_eta_2 * pnorm_eta_6
            result[ok] = p_single**2

        result = np.clip(result, p_guess, 1.0)
        return float(result[0]) if result.size == 1 else result

    def mu_eta(eta: NDArray | float) -> NDArray | float:
        """Derivative of linkinv for double duo-trio."""
        eta = np.atleast_1d(np.asarray(eta, dtype=np.float64))
        result = np.zeros_like(eta)

        ok = eta > 0
        result[eta >= 20] = 0.0

        if np.any(ok):
            eta_ok = eta[ok]
            sqrt_2 = np.sqrt(0.5)
            sqrt_6 = np.sqrt(1 / 6)

            pnorm_eta_2 = stats.norm.cdf(eta_ok * sqrt_2)
            pnorm_eta_6 = stats.norm.cdf(eta_ok * sqrt_6)

            A = stats.norm.pdf(eta_ok * sqrt_2) * sqrt_2
            B = stats.norm.pdf(eta_ok * sqrt_6) * sqrt_6
            C = stats.norm.pdf(eta_ok * sqrt_2)
            D = pnorm_eta_6 * sqrt_2
            E = pnorm_eta_2

            p_single = 1 - pnorm_eta_2 - pnorm_eta_6 + 2 * pnorm_eta_2 * pnorm_eta_6
            dp_single = -A - B + 2 * (C * D + E * B)

            result[ok] = 2 * p_single * dp_single

        return np.maximum(result, 0.0)

    return _make_double_link("double_duotrio", p_guess, linkinv, mu_eta)


def double_triangle_link() -> DoubleLinkResult:
    """Double triangle link function.

    The double triangle protocol has two independent triangle trials.
    Guessing probability is (1/3)^2 = 1/9.

    Returns
    -------
    DoubleLinkResult
        Link function object with linkinv, linkfun, and mu_eta methods.
    """
    p_guess = 1 / 9

    def linkinv(eta: NDArray | float) -> NDArray | float:
        """d-prime to probability for double triangle."""
        eta = np.atleast_1d(np.asarray(eta, dtype=np.float64))
        result = np.full_like(eta, p_guess)

        ok = (eta > 0) & (eta < 20)
        result[eta >= 20] = 1.0

        if np.any(ok):
            eta_ok = eta[ok]
            # Using non-central F distribution
            # P(single triangle) = 1 - F(3; df1=1, df2=1, ncp=d^2*2/3)
            ncp = eta_ok**2 * 2 / 3
            p_single = stats.ncf.sf(3, dfn=1, dfd=1, nc=ncp)
            result[ok] = p_single**2

        result = np.clip(result, p_guess, 1.0)
        return float(result[0]) if result.size == 1 else result

    def mu_eta(eta: NDArray | float) -> NDArray | float:
        """Derivative of linkinv for double triangle."""
        eta = np.atleast_1d(np.asarray(eta, dtype=np.float64))
        result = np.zeros_like(eta)

        ok = (eta > 0) & (eta < 20)
        # Derivative is 0 at boundaries (flat regions of the psychometric function)
        # result already initialized to 0 for eta <= 0 and eta >= 20

        if np.any(ok):
            d = eta[ok]
            ncp = d**2 * 2 / 3
            p_single = stats.ncf.sf(3, dfn=1, dfd=1, nc=ncp)

            sqrt_2_3 = np.sqrt(2 / 3)
            dnorm_d_sqrt6 = stats.norm.pdf(d / np.sqrt(6))
            pnorm_diff = stats.norm.cdf(d / np.sqrt(2)) - stats.norm.cdf(
                -d / np.sqrt(2)
            )

            dp_single = sqrt_2_3 * dnorm_d_sqrt6 * pnorm_diff
            result[ok] = 2 * p_single * dp_single

        return np.maximum(result, 0.0)

    return _make_double_link("double_triangle", p_guess, linkinv, mu_eta)


def double_threeafc_link() -> DoubleLinkResult:
    """Double 3-AFC link function.

    The double 3-AFC protocol has two independent 3-AFC trials.
    Guessing probability is (1/3)^2 = 1/9.

    Returns
    -------
    DoubleLinkResult
        Link function object with linkinv, linkfun, and mu_eta methods.

    Notes
    -----
    This function uses numerical integration and may be slower than
    closed-form alternatives.
    """
    p_guess = 1 / 9

    def _threeafc_integrand(x: float, d: float) -> float:
        """Integrand for single 3-AFC probability."""
        return stats.norm.pdf(x - d) * stats.norm.cdf(x) ** 2

    def _threeafc_deriv_integrand(x: float, d: float) -> float:
        """Integrand for derivative of 3-AFC probability."""
        return (x - d) * stats.norm.pdf(x - d) * stats.norm.cdf(x) ** 2

    def linkinv(eta: NDArray | float) -> NDArray | float:
        """d-prime to probability for double 3-AFC."""
        eta = np.atleast_1d(np.asarray(eta, dtype=np.float64))
        result = np.full_like(eta, p_guess)

        ok = (eta > 0) & (eta < 9)
        result[eta >= 9] = 1.0

        if np.any(ok):
            for i in np.where(ok)[0]:
                p_single, _ = integrate.quad(
                    _threeafc_integrand, -np.inf, np.inf, args=(eta[i],)
                )
                result[i] = p_single**2

        result = np.clip(result, p_guess, 1.0)
        return float(result[0]) if result.size == 1 else result

    def mu_eta(eta: NDArray | float) -> NDArray | float:
        """Derivative of linkinv for double 3-AFC."""
        eta = np.atleast_1d(np.asarray(eta, dtype=np.float64))
        result = np.zeros_like(eta)

        ok = (eta > 0) & (eta < 9)

        if np.any(ok):
            for i in np.where(ok)[0]:
                p_single, _ = integrate.quad(
                    _threeafc_integrand, -np.inf, np.inf, args=(eta[i],)
                )
                dp_single, _ = integrate.quad(
                    _threeafc_deriv_integrand, -np.inf, np.inf, args=(eta[i],)
                )
                result[i] = 2 * p_single * dp_single

        return np.maximum(result, 0.0)

    return _make_double_link("double_threeAFC", p_guess, linkinv, mu_eta, (0, 9))


def double_tetrad_link() -> DoubleLinkResult:
    """Double tetrad link function.

    The double tetrad protocol has two independent tetrad trials.
    Guessing probability is (1/3)^2 = 1/9.

    Returns
    -------
    DoubleLinkResult
        Link function object with linkinv, linkfun, and mu_eta methods.

    Notes
    -----
    This function uses numerical integration and may be slower than
    closed-form alternatives.
    """
    p_guess = 1 / 9

    def _tetrad_integrand(z: float, delta: float) -> float:
        """Integrand for single tetrad probability."""
        pnorm_z = stats.norm.cdf(z)
        pnorm_z_delta = stats.norm.cdf(z - delta)
        return stats.norm.pdf(z) * (2 * pnorm_z * pnorm_z_delta - pnorm_z_delta**2)

    def linkinv(eta: NDArray | float) -> NDArray | float:
        """d-prime to probability for double tetrad."""
        eta = np.atleast_1d(np.asarray(eta, dtype=np.float64))
        result = np.full_like(eta, p_guess)

        eps = 1e-8
        ok = (eta > eps) & (eta < 9)
        result[eta >= 9] = 1.0

        if np.any(ok):
            for i in np.where(ok)[0]:
                integral, _ = integrate.quad(
                    _tetrad_integrand, -np.inf, np.inf, args=(eta[i],)
                )
                p_single = 1 - 2 * integral
                result[i] = p_single**2

        result = np.clip(result, p_guess, 1.0)
        return float(result[0]) if result.size == 1 else result

    def mu_eta(eta: NDArray | float) -> NDArray | float:
        """Derivative of linkinv for double tetrad (numerical)."""
        eta = np.atleast_1d(np.asarray(eta, dtype=np.float64))
        result = np.zeros_like(eta)

        eps = 1e-8
        ok = (eta > eps) & (eta < 9)

        if np.any(ok):
            # Use numerical differentiation
            h = 1e-6
            for i in np.where(ok)[0]:
                f_plus = linkinv(eta[i] + h)
                f_minus = linkinv(eta[i] - h)
                result[i] = (f_plus - f_minus) / (2 * h)

        return np.maximum(result, 0.0)

    return _make_double_link("double_tetrad", p_guess, linkinv, mu_eta, (0, 9))


# Convenience accessors
def get_double_link(method: str) -> DoubleLinkResult:
    """Get a double link function by name.

    Parameters
    ----------
    method : str
        Protocol name: "double_twoAFC", "double_duotrio", "double_triangle",
        "double_threeAFC", "double_tetrad". Also accepts variants like
        "doubletwoAFC", "double-twoAFC", etc.

    Returns
    -------
    DoubleLinkResult
        Link function object.

    Raises
    ------
    ValueError
        If method is not recognized.
    """
    # Normalize method name
    method_lower = method.lower().replace("-", "").replace("_", "")

    link_map = {
        "doubletwoafc": double_twoafc_link,
        "double2afc": double_twoafc_link,
        "doubleduotrio": double_duotrio_link,
        "doubletriangle": double_triangle_link,
        "doublethreeafc": double_threeafc_link,
        "double3afc": double_threeafc_link,
        "doubletetrad": double_tetrad_link,
    }

    if method_lower not in link_map:
        valid = ["double_twoAFC", "double_duotrio", "double_triangle",
                 "double_threeAFC", "double_tetrad"]
        raise ValueError(f"Unknown double method '{method}'. Valid: {valid}")

    return link_map[method_lower]()
