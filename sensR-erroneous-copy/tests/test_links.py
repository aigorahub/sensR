import numpy as np
from senspy import psyfun, psyinv, psyderiv


def test_psyfun_psyinv_roundtrip():
    x = np.linspace(-3, 3, 7)
    p = psyfun(x)
    x_back = psyinv(p)
    assert np.allclose(x, x_back)


def test_psyderiv_positive():
    assert psyderiv(0.0) > 0
