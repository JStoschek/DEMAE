"""The exact-zero land / zero-fill pixel screen (ADR-0001 §3, Amendment 2).

Copernicus paints ocean — and sea-level rivers and lakes — to exactly ``0.0``. This is
the pixel-level screen that measures how much of a window is that fill and decides whether
it survives. Two callers, at two resolutions: :mod:`dataset.catalog` measures the
exact-zero fraction of each x8-overview window at scoring time (it becomes the
``exact_zero_frac`` column the gates read), and :mod:`dataset.normalize` re-applies the
same predicate at full resolution before a tile is stored.

It lived in the retired TRI-era ``stratify.py`` and moved here intact: the zero-fill
screen is live and referenced by the gate work, and it is not stratification — it is the
one arithmetic definition of "how much of this window is fill", owned in one place so the
scoring pass and the materialize pass cannot drift.
"""

from __future__ import annotations

import numpy as np

from . import config


def zero_fraction(window: np.ndarray) -> float:
    """Fraction of pixels equal to Copernicus's exact ocean fill (``0.0``)."""
    return float(np.mean(window == config.OCEAN_FILL_VALUE))


def is_zero_fill(window: np.ndarray) -> bool:
    """Whether a window is mostly zero-fill and should be dropped (§3).

    True iff the exact-zero fraction reaches :data:`config.LAND_ZERO_FRAC_DROP` (0.75).
    Coastlines (well under 75% sea) are kept; genuine sub-sea-level land is negative, not
    zero, so it is never mistaken for fill.

    **Not named ``is_ocean``** (ADR-0001 §3 Amendment 2, issue #34). A *pixel* value of
    ``0.0`` really is Copernicus's ocean fill — ``WBM == 1`` was measured to be a strict
    subset of exact-zero, 0 counter-examples in 144.5 M pixels. But a *window* that is 75%
    zeros is not necessarily ocean: Copernicus paints sea-level rivers and lakes (WBM 2/3)
    to exactly ``0.0`` too, so this predicate fires on Bangladesh delta and Scheldt polder
    water. It screens zero-fill, and the catalog books its attrition under ``zero_fill`` —
    not ``ocean``.
    """
    return zero_fraction(window) >= config.LAND_ZERO_FRAC_DROP
