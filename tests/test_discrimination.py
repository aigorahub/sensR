import numpy as np
from senspy import (
    duotrio_pc,
    discrim_2afc,
    get_pguess,
    pc2pd,
    pd2pc,
)

def test_duotrio_half_at_zero():
    assert duotrio_pc(0.0) == 0.5


def test_duotrio_monotonic():
    low = duotrio_pc(0.1)
    high = duotrio_pc(1.0)
    assert low < high < 1.0

def test_discrim_2afc_basic():
    res = discrim_2afc(correct=30, total=50)
    assert res["d_prime"] > 0
    assert res["se"] > 0


def test_pc2pd_pd2pc_roundtrip():
    pg = get_pguess("twoafc")
    pc = 0.75
    pd = pc2pd(pc, pg)
    pc_back = pd2pc(pd, pg)
    assert abs(pc - pc_back) < 1e-12
