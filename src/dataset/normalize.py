"""Full-resolution tile preparation (ADR-0001 §6-7, ADR-0002 §2).

Turns a raw 224x224 float32 window into what actually gets stored: a plausibility-
clipped **int16-metre** array plus the per-tile land-only ``mean``/``std`` the
loader will normalize with. Normalization is deliberately *not* applied to the
pixels here — storing raw + stats keeps the per-tile-vs-global normalization
ablation a loader-config flip (ADR-0002). Degenerate near-constant windows are
dropped so uniform int16 quantisation cannot band a kept tile.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config
from .screen import is_zero_fill  # zero-fill screen (ADR-0001 §3); re-applied here at full res


@dataclass(frozen=True)
class PreparedTile:
    """A tile ready for the shard writer + manifest."""

    raw_i16: np.ndarray  # (224, 224) int16 metres, plausibility-clipped
    mean: float  # per-tile land-only mean (metres) — loader normalization
    std: float  # per-tile land-only std (metres)
    land_frac: float  # fraction of non-ocean pixels — QC / auditing


def plausibility_clip(a: np.ndarray) -> np.ndarray:
    """Clamp to the absolute plausibility range ``[-500, 9000]`` m (ADR-0001 §6).

    This is the *only* clip — no percentile clip — so real high-relief peaks (the
    reason the corpus exists) survive; it just fences out fill/artifact garbage.
    """
    return np.clip(a, config.CLIP_MIN_M, config.CLIP_MAX_M)


def land_mask(a: np.ndarray) -> np.ndarray:
    """Boolean land mask: ``True`` where elevation is not the exact-zero ocean
    fill. Sub-sea-level land is negative, so it stays land."""
    return a != config.OCEAN_FILL_VALUE


def land_stats(a: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Population mean and std over land pixels only (exact-zero excluded)."""
    land = a[mask]
    return float(land.mean()), float(land.std())


def to_int16(a: np.ndarray) -> np.ndarray:
    """Round to the nearest metre and cast to int16 (the storage dtype).

    Assumes the input is already plausibility-clipped, so it sits inside the
    int16 range. int16 metres is effectively lossless w.r.t. Copernicus's +/-4 m
    accuracy and halves storage vs fp32 (ADR-0002 §2).
    """
    return np.rint(a).astype(np.int16)


def prepare_tile(window: np.ndarray) -> tuple[PreparedTile | None, str]:
    """Prepare a full-res window for storage, or reject it.

    Returns ``(PreparedTile, "ok")`` on success, or ``(None, reason)`` where
    reason is one of ``"zero_fill"`` (>=75% exact-zero at full res, ADR-0001 §3;
    renamed from ``"ocean"`` by Amendment 2 — it also catches sea-level river/lake),
    ``"no_land"`` (no land pixels to compute stats from), or ``"degenerate"``
    (land-std < 3 m, ADR-0002 §2's tightened epsilon).

    The mask, land fraction, and per-tile ``mean``/``std`` are all computed on the
    **stored int16 array**, so they describe exactly the bytes the loader reads
    and recomputes its ``raw != 0`` mask from (ADR-0003 §4) — the per-tile and
    global normalization stats then share one quantisation.
    """
    raw_i16 = to_int16(plausibility_clip(window))

    if is_zero_fill(raw_i16):
        return None, "zero_fill"

    mask = land_mask(raw_i16)
    land_frac = float(mask.mean())
    if land_frac == 0.0:
        return None, "no_land"

    mean, std = land_stats(raw_i16.astype(np.float32), mask)
    if std < config.DEGENERATE_STD_EPS_M:
        return None, "degenerate"

    return (
        PreparedTile(
            raw_i16=raw_i16,
            mean=mean,
            std=std,
            land_frac=land_frac,
        ),
        "ok",
    )


__all__ = [
    "PreparedTile",
    "plausibility_clip",
    "land_mask",
    "land_stats",
    "to_int16",
    "prepare_tile",
]
