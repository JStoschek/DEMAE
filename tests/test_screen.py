"""The exact-zero land / zero-fill pixel screen (ADR-0001 §3, Amendment 2)."""

from __future__ import annotations

import numpy as np
import pytest

from dataset import screen


def test_zero_fraction():
    a = np.array([[0.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    assert screen.zero_fraction(a) == pytest.approx(0.5)


def test_is_zero_fill_threshold():
    # exactly at the 0.75 threshold -> dropped
    a = np.zeros((4, 4), dtype=np.float32)
    a[0] = 5.0  # 4/16 land -> 0.75 zero-fraction
    assert screen.zero_fraction(a) == pytest.approx(0.75)
    assert screen.is_zero_fill(a) is True
    a[1] = 5.0  # now 0.5 zero-fraction -> kept (coastline)
    assert screen.is_zero_fill(a) is False


def test_negative_land_is_not_fill():
    # Sub-sea-level land is negative, never the exact-zero fill.
    a = np.full((4, 4), -12.0, dtype=np.float32)
    assert screen.zero_fraction(a) == 0.0
    assert screen.is_zero_fill(a) is False
