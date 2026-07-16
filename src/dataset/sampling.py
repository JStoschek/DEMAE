"""The shared deterministic global COG sampler.

Draws ``n`` of the released 1-degree COGs, spread across latitude bands and 10-degree
geo-cells, reproducibly, so the Phase-A samples stay comparable: A1's mask-prevalence
sample (:func:`prevalence_sample`), A3's overview-vs-full-res calibration sample
(:func:`calibration_sample`), and later Phase B1's pilot frame.

Selection is two-level — an explicit per-latitude-band quota outside, ADR-0001 §5's
round-robin across 10-degree geo-cells inside. The band quota makes each band's sample
size a chosen number rather than one inherited from how many geo-cells the band happens
to touch, and how it is apportioned depends on what the sample is for: see the two
consumer functions. Issue #15 records the measured frame behind that split.

Everything here is a pure function of COG *names* — no bytes are read. That is what
makes the tables re-runnable, and it bounds what the sampler can screen: mostly-ocean
and off-terrain cells can only be found in pixels, so callers over-draw and filter
after reading. The result is ordered to make that cheap — any prefix of it is itself a
spread sample.

Ordering within a geo-cell is a keyed BLAKE2b hash of the COG stem, not
``Generator.permutation``: NumPy guarantees its bit-generator streams across releases
but not the distributions layered on them, and these draws must reproduce from a
committed table for the life of the project. Hashing the stem also makes the draw
independent of the order the frame arrives from :mod:`dataset.discovery`.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

from . import config
from .geometry import CogRef, geocell_id, lat_band

Allocation = Literal["proportional", "balanced"]

_ALLOCATIONS: tuple[Allocation, ...] = ("proportional", "balanced")

# Sample sizes the curation plan fixes for Phase A (§A1, §A3).
PREVALENCE_SAMPLE_N: int = 500
CALIBRATION_SAMPLE_N: int = 100


# --- Name-level keys ----------------------------------------------------------


def _center(ref: CogRef) -> tuple[float, float]:
    """Centre ``(lon, lat)`` of a COG's 1-degree cell.

    Keyed on the centre, not the south-west origin the name encodes, so a cell is
    binned by where its mass actually is: N49 spans [49, 50] and belongs to the 0-50
    band, while its origin sits exactly on the edge and its north edge does not.
    """
    return ref.origin_lon + 0.5, ref.origin_lat + 0.5


def band_of(ref: CogRef) -> str:
    """Latitude-band label of a COG (poleward-symmetric, e.g. ``"50-60"``)."""
    return lat_band(_center(ref)[1])


def geocell_of(ref: CogRef) -> str:
    """10-degree geo-cell id of a COG (e.g. ``"+000_+040"``)."""
    lon, lat = _center(ref)
    return geocell_id(lon, lat)


def _shuffle_key(stem: str, seed: int) -> bytes:
    """Deterministic per-COG sort key — a keyed hash, stable across machines."""
    return hashlib.blake2b(f"{seed}:{stem}".encode(), digest_size=16).digest()


# --- Band quotas --------------------------------------------------------------


def band_quotas(
    band_counts: Mapping[str, int],
    n: int,
    *,
    allocation: Allocation = "proportional",
) -> dict[str, int]:
    """Apportion ``n`` draws across latitude bands holding ``band_counts`` cells.

    ``allocation`` weights the bands: ``"proportional"`` by population, ``"balanced"``
    equally. No band is ever allotted more cells than it has; the surplus from a band
    that hits its ceiling redistributes to bands with headroom, so the total is ``n``
    whenever the frame can supply it — and every available cell when it cannot.
    """
    if allocation not in _ALLOCATIONS:
        raise ValueError(f"unknown allocation {allocation!r}; expected one of {_ALLOCATIONS}")

    caps = {b: c for b, c in band_counts.items() if c > 0}
    weights = {b: (float(caps[b]) if allocation == "proportional" else 1.0) for b in caps}
    alloc = dict.fromkeys(caps, 0)

    # Largest-remainder apportionment, re-run over the bands that still have headroom
    # until n is placed or the frame is exhausted. Each pass places at least one draw,
    # so the loop cannot spin: every open band has headroom by construction, and the
    # leftover pass below hands a draw to one of them whenever the ideal shares round
    # down to nothing.
    while (remaining := n - sum(alloc.values())) > 0:
        open_bands = [b for b in sorted(caps) if alloc[b] < caps[b]]
        if not open_bands:
            break
        total_w = sum(weights[b] for b in open_bands)
        ideal = {b: remaining * weights[b] / total_w for b in open_bands}
        add = {b: min(int(ideal[b]), caps[b] - alloc[b]) for b in open_bands}
        # Hand out what rounding dropped, largest fractional part first; the band name
        # breaks ties so the result never depends on dict ordering.
        leftover = remaining - sum(add.values())
        for b in sorted(open_bands, key=lambda b: (-(ideal[b] % 1.0), b)):
            if leftover <= 0:
                break
            if alloc[b] + add[b] < caps[b]:
                add[b] += 1
                leftover -= 1
        for b, extra in add.items():
            alloc[b] += extra

    return alloc


# --- The sampler --------------------------------------------------------------


def _draw_band(refs: Sequence[CogRef], quota: int, seed: int) -> list[CogRef]:
    """Take ``quota`` cells from one band, round-robin across its geo-cells.

    Each geo-cell yields one cell per turn and goes to the back of the queue, so the
    draw spreads before it deepens. Cells are consumed inside a geo-cell in hash order
    and the queue is walked in sorted geo-cell order, so nothing depends on the order
    ``refs`` arrived in.

    This is deliberately a second copy of ``stratify.select_quota``'s geo-cell rotation
    rather than a shared primitive: that module is slated for replacement in Phase D,
    and the two differ (it rotates scored *windows* under a per-COG cap; this rotates
    whole COGs, which cannot repeat). Phase D's sampler inherits the plan's round-robin
    fill and is the point at which one rotation should be extracted for both.
    """
    buckets: dict[str, list[CogRef]] = defaultdict(list)
    for ref in refs:
        buckets[geocell_of(ref)].append(ref)
    for gc in buckets:
        buckets[gc].sort(key=lambda r: _shuffle_key(r.stem, seed))

    cursor = dict.fromkeys(buckets, 0)
    active = deque(sorted(buckets))
    picked: list[CogRef] = []
    while len(picked) < quota and active:
        gc = active.popleft()
        picked.append(buckets[gc][cursor[gc]])
        cursor[gc] += 1
        if cursor[gc] < len(buckets[gc]):
            active.append(gc)
    return picked


def _interleave(by_band: Mapping[str, Sequence[CogRef]]) -> list[CogRef]:
    """Merge the per-band draws into one priority-ordered list.

    A band's i-th pick is laid at position ``(i + 0.5) / <that band's draw length>`` on
    a shared unit line, and everything merges by position, so each band is spent at a
    rate proportional to its own draw. The consequence is the over-draw contract: the
    first ``m`` of the result carry ``draw_b * m / n`` from every band, to within one
    cell, so a truncated read is still a band-spread sample. The positions are exact
    IEEE-754 quotients and the band name breaks ties, so the merge is deterministic.
    """
    laid = [
        ((i + 0.5) / len(picks), band, i, ref)
        for band, picks in by_band.items()
        for i, ref in enumerate(picks)
    ]
    laid.sort(key=lambda t: t[:3])
    return [t[3] for t in laid]


def sample_cogs(
    refs: Iterable[CogRef],
    n: int,
    *,
    allocation: Allocation = "proportional",
    seed: int = config.RANDOM_SEED,
) -> list[CogRef]:
    """Draw ``n`` COGs from ``refs``, spread across latitude bands and geo-cells.

    Returns fewer than ``n`` only when ``refs`` cannot supply them. The result is in
    priority order, not sorted: any prefix of it is itself a spread sample, so a caller
    who must discard cells after reading them (mostly-ocean, wrong terrain) can
    over-draw and stop early. Prefer :func:`prevalence_sample` or
    :func:`calibration_sample` — they carry Phase A's decided arguments.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    refs = list(refs)
    if not refs:
        raise ValueError("cannot sample an empty frame")

    by_band: dict[str, list[CogRef]] = defaultdict(list)
    for ref in refs:
        by_band[band_of(ref)].append(ref)

    quotas = band_quotas({b: len(v) for b, v in by_band.items()}, n, allocation=allocation)
    drawn = {
        band: _draw_band(by_band[band], quota, seed)
        for band, quota in sorted(quotas.items())
        if quota > 0
    }
    return _interleave(drawn)


# --- Phase-A consumers --------------------------------------------------------


def prevalence_sample(
    refs: Iterable[CogRef],
    n: int = PREVALENCE_SAMPLE_N,
    *,
    seed: int = config.RANDOM_SEED,
) -> list[CogRef]:
    """A1's mask-prevalence sample — corpus-shaped, ~500 cells (plan §A1).

    Apportioned ``"proportional"``, i.e. roughly uniform over cells, because A1 is
    estimating a global figure and wants the corpus's own shape.

    ⚠️ Uniform over *cells* is not uniform over *pixels*. A 1-degree cell at 85 degrees
    holds a fraction of the pixels of one at the equator — Copernicus drops columns
    poleward, which is what the latitude bands *are* — and the cells are not
    equal-area either: above 50 degrees is 55% of the corpus by cell and under a third
    of it by ground area. Pool raw pixel counts and that corrects itself, since each
    cell carries its own weight; average per-cell fractions instead and the prevalence
    table inherits the polar over-weight.
    """
    return sample_cogs(refs, n, allocation="proportional", seed=seed)


def calibration_sample(
    refs: Iterable[CogRef],
    n: int = CALIBRATION_SAMPLE_N,
    *,
    seed: int = config.RANDOM_SEED,
) -> list[CogRef]:
    """A3's overview-vs-full-res calibration sample — ~100 cells (plan §A3).

    Apportioned ``"balanced"``, so every populated band lands the same number of cells.
    Gate A3 reports a rank correlation *per metric per band* and the artifact it is
    hunting is band-local — anisotropic pixels and the sampling discontinuities above
    50 degrees (plan §11.4) — so the sample needs equal power in each band, not
    corpus-shaped proportions. This is what guarantees the high-latitude cells the gate
    cannot be decided without; a corpus-shaped draw would spend most of itself where
    the risk is not.

    Terrain spread — the other half of A3's "spanning latitude bands and terrain types"
    — is *not* handled here and cannot be: terrain type is defined by the very metrics
    A3 exists to calibrate, and no COG name predicts it. Take this draw over-sized,
    read the x8 overviews, stratify on what they show, and keep the ``n`` wanted; the
    priority ordering keeps every prefix band-spread while you do.
    """
    return sample_cogs(refs, n, allocation="balanced", seed=seed)
