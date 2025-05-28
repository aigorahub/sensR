import numpy as np
from senspy import (
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


def test_duotrio_half_at_zero():
    assert duotrio_pc(0.0) == 0.5


def test_duotrio_monotonic():
    low = duotrio_pc(0.1)
    high = duotrio_pc(1.0)
    assert low < high < 1.0


def test_triangle_baseline():
    assert triangle_pc(0.0) == 1 / 3


def test_three_afc_baseline():
    assert three_afc_pc(0.0) == 1 / 3


def test_get_pguess_mapping():
    assert get_pguess("twoAFC") == 0.5
    assert get_pguess("triangle") == 1 / 3


def test_pc_pd_roundtrip():
    pguess = 0.5
    pc = 0.8
    pd = pc2pd(pc, pguess)
    pc_back = pd2pc(pd, pguess)
    assert abs(pc - pc_back) < 1e-12


def test_discrim_2afc_positive():
    est, se = discrim_2afc(correct=30, total=50)
    assert est > 0 and se > 0
