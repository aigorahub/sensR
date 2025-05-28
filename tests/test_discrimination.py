import numpy as np
from senspy import duotrio_pc, discrim_2afc


def test_duotrio_half_at_zero():
    assert duotrio_pc(0.0) == 0.5


def test_duotrio_monotonic():
    low = duotrio_pc(0.1)
    high = duotrio_pc(1.0)
    assert low < high < 1.0


def test_discrim_2afc_estimates():
    res = discrim_2afc(correct=25, total=50)
    assert res["d_prime"] > 0
    assert res["se"] > 0
