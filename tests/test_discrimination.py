import numpy as np
from senspy import duotrio_pc


def test_duotrio_half_at_zero():
    assert duotrio_pc(0.0) == 0.5


def test_duotrio_monotonic():
    low = duotrio_pc(0.1)
    high = duotrio_pc(1.0)
    assert low < high < 1.0
