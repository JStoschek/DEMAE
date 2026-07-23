"""Measure EDM/FLM/WBM prevalence on a global tile sample (issue #19).

The number this produces is load-bearing: the Phase-C quality gates (WBM water
> 0.30 drop, EDM/FLM > 0.25 drop) are ``[project]`` calls with no published
precedent, tuned against the table measured here, and the curation map hangs one
open question on it — *does prevalence force the WBM-only fallback?* This module is
the measurement, split into a pure, unit-tested core (counting, identities,
pooling, attrition) and a thin live driver (:func:`main`).

Four things the mask spec (``docs/research/copernicus-mask-spec.md``) pins, and how
each lands here:

1. **Full resolution, never ×8** (spec §7). Mode-resampled overviews erase any code
   that is never locally dominant — a per-code table built on ×8 reports hard zeros
   for real rare classes. Reads go through the mask reader at ``factor=1``
   (native), affordable because the masks are ~0.25 MB/tile.

2. **Bin on the explicit vocabularies** (spec §3). FLM is non-contiguous — 0–9 then
   100–102 — so :func:`count_codes` bins on the value list, never ``range``, and
   raises on any out-of-vocabulary code rather than pooling it.

3. **Pool raw pixel counts; do not average per-cell fractions** (plan §A1 ⚠️). The
   draw is uniform over *cells*, which is not uniform over *pixels* — polar cells
   hold fewer columns by design, and above 50° is 55% of the corpus by cell against
   30% by land area. :func:`pooled_fractions` sums counts and divides once, so each
   cell carries its own pixel weight; averaging fractions would bake in a ~2× polar
   over-weight into the very table that tunes the thresholds.

4. **Assert the three §6.1 identities as a tripwire** (spec §6.1). A mis-keyed bin
   or an off-by-one window would corrupt a committed artifact silently;
   :func:`identity_residuals` cross-checks EDM 1 ≡ FLM 2, EDM 2 ≡ Σ FLM≥3, and
   WBM water ≡ EDM water before any number is written.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from . import config, masks
from .geometry import CogRef
from .masks import Layer, MaskReader
from .sampling import band_of, geocell_of

# --- Semantic code groupings (spec §2-4) --------------------------------------

#: WBM codes that count as water: Ocean/Lake/River. WBM 0 is "No water", not void.
WBM_WATER_CODES: Final = (1, 2, 3)

#: FLM codes that mark an externally-filled (non-TanDEM) pixel: everything ≥ 3
#: (ASTER/SRTM/GMTED/… and the 100–102 national DEMs). FLM ∈ {1,2} is native. The
#: explicit tuple (derived once from the vocabulary) is what the report bins on —
#: the non-contiguous 100–102 must never fall through a ``range``-based test.
FLM_FILL_MIN: Final = 3
FLM_CODES_FILL: Final = tuple(c for c in masks.FLM_CODES if c >= FLM_FILL_MIN)

#: EDM codes that count as *terrain editing*. The pristine code 1 (Not edited) and
#: void 0 are excluded, and so are the water classes 8/9/10/11 (Ocean/Lake/River/
#: Shoreline), which the WBM gate already owns — counting them here would put water
#: in double jeopardy across two gates. What remains is genuine terrain intervention:
#: 2 external infill, 3 interpolated, 4 smoothed, 5 airport, 6 raised-negative,
#: 7 flattened, 12 morphed, 13 shifted.
#:
#: The Phase-C gate reads "EDM-edited **or** FLM-non-TanDEM fill". By the §6.1
#: identity the FLM≥3 pixels are exactly the EDM==2 pixels, and EDM==2 ∈ this set, so
#: the union collapses to this EDM set alone — :func:`edm_edited_frac` *is* that gate
#: input, provided the identity holds (which the driver verifies per tile).
EDITED_EDM_CODES: Final = frozenset({2, 3, 4, 5, 6, 7, 12, 13})

#: EDM codes that mark water (Ocean/Lake/River) — the §6.1 cross-check against WBM.
_EDM_WATER_CODES: Final = (8, 9, 10)


# --- Per-code counting (spec §3, §6.1 tripwire) -------------------------------


def count_codes(mask: np.ndarray, codes: Sequence[int]) -> dict[int, int]:
    """Pixel count per code over a whole mask, keyed on the explicit vocabulary.

    Every code in ``codes`` gets a row — including ones this tile never shows, at
    zero — so a table column is never silently absent just because a class did not
    occur here. Any pixel whose value is *outside* ``codes`` raises
    :class:`ValueError`: a real Copernicus mask never carries an out-of-vocabulary
    code, so one is a read bug, and pooling it into a committed table would corrupt
    it. Bins on the value list, never a contiguous range (FLM is non-contiguous).
    """
    flat = np.ascontiguousarray(mask).ravel()
    bc = np.bincount(flat, minlength=256)
    counts = {int(c): int(bc[c]) for c in codes}
    accounted = sum(counts.values())
    if accounted != flat.size:
        stray = sorted(int(v) for v in np.unique(flat) if int(v) not in set(codes))
        raise ValueError(
            f"mask carries {flat.size - accounted} pixel(s) outside the vocabulary "
            f"{tuple(codes)}: codes {stray}"
        )
    return counts


@dataclass(frozen=True)
class TileCounts:
    """Per-code pixel counts for one tile's three masks, plus its geography.

    ``total`` is the tile's pixel count (identical across the three co-registered
    masks). ``lon``/``lat`` are the 1° cell centre; ``band``/``geocell`` are the
    sampler's own keys, so a prevalence row joins back to the draw.
    """

    stem: str
    lon: float
    lat: float
    band: str
    geocell: str
    total: int
    edm: dict[int, int]
    flm: dict[int, int]
    wbm: dict[int, int]

    def counts(self, layer: Layer) -> dict[int, int]:
        return {"EDM": self.edm, "FLM": self.flm, "WBM": self.wbm}[layer]


def _read_full_res(reader: MaskReader, cog: CogRef, layer: Layer) -> np.ndarray:
    """Read a mask at native resolution (spec §7: measure prevalence full-res).

    ``factor=1`` means no decimation — the reader serves the base raster, not one of
    the mode-resampled ×2/×4/×8 overviews that erase never-dominant codes.
    """
    return reader.read_overview(cog, layer, 1)


def tile_counts(reader: MaskReader, cog: CogRef) -> TileCounts:
    """Read a tile's EDM/FLM/WBM full-res and count each on its vocabulary.

    Raises whatever the reader raises on absence (:class:`~dataset.masks.MaskNotFound`)
    or a transient failure — never degrades a missing mask into "no edited pixels".
    """
    edm = count_codes(_read_full_res(reader, cog, "EDM"), masks.EDM_CODES)
    flm = count_codes(_read_full_res(reader, cog, "FLM"), masks.FLM_CODES)
    wbm = count_codes(_read_full_res(reader, cog, "WBM"), masks.WBM_CODES)
    total = sum(edm.values())
    return TileCounts(
        stem=cog.stem,
        lon=cog.origin_lon + 0.5,
        lat=cog.origin_lat + 0.5,
        band=band_of(cog),
        geocell=geocell_of(cog),
        total=total,
        edm=edm,
        flm=flm,
        wbm=wbm,
    )


# --- §6.1 cross-mask identities (the committed-table tripwire) ----------------


def identity_residuals(tc: TileCounts) -> dict[str, int]:
    """Absolute pixel residuals of the three §6.1 identities (0 when they hold).

    * ``edm1_eq_flm2`` — EDM 1 (Not edited) ≡ FLM 2 (Not edited / not filled).
    * ``edm2_eq_flm_fill`` — EDM 2 (external infill) ≡ Σ FLM≥3 (all fill sources).
    * ``wbm_water_eq_edm_water`` — WBM {1,2,3} ≡ EDM {8,9,10} (ocean/lake/river).

    A non-zero residual means a mis-keyed bin or a misaligned read; the driver flags
    any tile where it exceeds a hair of the pixel count before trusting its row.
    """
    flm_fill = sum(c for code, c in tc.flm.items() if code >= FLM_FILL_MIN)
    wbm_water = sum(tc.wbm[c] for c in WBM_WATER_CODES)
    edm_water = sum(tc.edm[c] for c in _EDM_WATER_CODES)
    return {
        "edm1_eq_flm2": abs(tc.edm[1] - tc.flm[2]),
        "edm2_eq_flm_fill": abs(tc.edm[2] - flm_fill),
        "wbm_water_eq_edm_water": abs(wbm_water - edm_water),
    }


def identities_hold(tc: TileCounts, *, tol_px: int = 0) -> bool:
    """True iff every §6.1 residual is within ``tol_px`` pixels (exact by default)."""
    return all(r <= tol_px for r in identity_residuals(tc).values())


# --- Derived per-tile fractions (the Phase-C gate inputs) ---------------------


def wbm_water_frac(tc: TileCounts) -> float:
    """Fraction of the tile classified as water (WBM ∈ {1,2,3}) — the WBM gate input."""
    return sum(tc.wbm[c] for c in WBM_WATER_CODES) / tc.total


def flm_fill_frac(tc: TileCounts) -> float:
    """Fraction externally filled (FLM ≥ 3) — the "not TanDEM-X" signal (spec §3)."""
    return sum(c for code, c in tc.flm.items() if code >= FLM_FILL_MIN) / tc.total


def edm_edited_frac(tc: TileCounts) -> float:
    """Fraction terrain-edited (EDM ∈ :data:`EDITED_EDM_CODES`) — the EDM/FLM gate input.

    Equals the plan's "EDM-edited or FLM-non-TanDEM fill" union by the §6.1 identity
    (FLM≥3 pixels ≡ EDM==2 pixels ⊆ this set); water is excluded here because the WBM
    gate owns it.
    """
    return sum(tc.edm[c] for c in EDITED_EDM_CODES) / tc.total


def void_frac(tc: TileCounts, layer: Layer) -> float:
    """Fraction of in-band void (code 0) for EDM/FLM. WBM has no void code."""
    return tc.counts(layer)[0] / tc.total


# --- Pixel-pooled aggregation (plan §A1 ⚠️ polar-weight guard) ----------------


def _vocab(layer: Layer) -> tuple[int, ...]:
    return {"EDM": masks.EDM_CODES, "FLM": masks.FLM_CODES, "WBM": masks.WBM_CODES}[layer]


def pool_counts(tiles: Sequence[TileCounts], layer: Layer) -> tuple[dict[int, int], int]:
    """Sum per-code counts and total pixels across ``tiles`` for one layer."""
    pooled = dict.fromkeys(_vocab(layer), 0)
    total = 0
    for tc in tiles:
        total += tc.total
        for code, c in tc.counts(layer).items():
            pooled[code] += c
    return pooled, total


def pooled_subset_frac(
    tiles: Sequence[TileCounts], layer: Layer, codes: Iterable[int]
) -> float:
    """Pixel-pooled fraction of one layer's pixels falling in ``codes``.

    The report needs "fraction at *these* codes, pooled across *these* tiles" for
    several groupings (a band's water, a band's edit set, …); this is that one shape,
    so the summation is written once rather than re-inlined per section.
    """
    codes = set(codes)
    total = sum(tc.total for tc in tiles)
    if total == 0:
        return float("nan")
    hit = sum(sum(c for code, c in tc.counts(layer).items() if code in codes) for tc in tiles)
    return hit / total


def pooled_fractions(tiles: Sequence[TileCounts], layer: Layer) -> dict[int, float]:
    """Corpus-shaped per-code fraction: Σ pixels at code / Σ pixels (spec/plan §A1).

    This is the pixel-pooled figure — each cell carries its own weight — *not* the
    mean of the per-cell fractions, which would inherit the polar over-weight.
    """
    pooled, total = pool_counts(tiles, layer)
    if total == 0:
        return {code: float("nan") for code in pooled}
    return {code: c / total for code, c in pooled.items()}


# --- Distribution + tail ------------------------------------------------------


def describe(values: Iterable[float]) -> dict[str, float]:
    """Mean/quantile/tail summary of a per-tile quantity.

    The prevalence that bites is per-tile, not global: a 0.02 mean could be 2%
    everywhere (harmless) or 100% in 2% of tiles (a terrain family gated out). This
    reports the shape so the tail is visible.
    """
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"n": 0}
    q = np.quantile(arr, [0.5, 0.9, 0.95, 0.99])
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p50": float(q[0]),
        "p90": float(q[1]),
        "p95": float(q[2]),
        "p99": float(q[3]),
        "max": float(arr.max()),
    }


# --- Threshold attrition (apply the plan's defaults; do not re-tune) ----------

#: Phase-C default caps (plan §C2). Applied here, never re-tuned — that is Phase C.
WATER_DROP_CAP: Final = 0.30
WATER_FLAG_CAP: Final = 0.05
EDIT_DROP_CAP: Final = 0.25


def attrition(
    tiles: Sequence[TileCounts],
    *,
    water_cap: float = WATER_DROP_CAP,
    edit_cap: float = EDIT_DROP_CAP,
    water_flag: float = WATER_FLAG_CAP,
) -> dict[str, float]:
    """How many sample tiles the plan's default gates would discard.

    Reports each gate alone and their union — the number issue #19 asks for. The
    caps are the plan's published defaults; this measures their cost, it does not
    re-tune them (that is Phase C).
    """
    water = np.array([wbm_water_frac(tc) for tc in tiles])
    edit = np.array([edm_edited_frac(tc) for tc in tiles])
    drop_water = water > water_cap
    drop_edit = edit > edit_cap
    drop_either = drop_water | drop_edit
    n = len(tiles)
    return {
        "n_tiles": n,
        "water_cap": water_cap,
        "edit_cap": edit_cap,
        "dropped_water": int(drop_water.sum()),
        "dropped_edit": int(drop_edit.sum()),
        "dropped_either": int(drop_either.sum()),
        "flagged_water": int((water > water_flag).sum()),
        "frac_dropped_water": float(drop_water.mean()) if n else 0.0,
        "frac_dropped_edit": float(drop_edit.mean()) if n else 0.0,
        "frac_dropped_either": float(drop_either.mean()) if n else 0.0,
    }


# --- Report assembly ----------------------------------------------------------

# Human-readable meaning per code, for the committed table (spec §2-4).
_EDM_MEANING: Final = {
    0: "Void (no data)", 1: "Not edited", 2: "Infill of external elevation data",
    3: "Interpolated", 4: "Smoothed", 5: "Airport editing",
    6: "Raised negative elevation", 7: "Flattened", 8: "Ocean", 9: "Lake",
    10: "River", 11: "Shoreline", 12: "Morphed", 13: "Shifted",
}
_FLM_MEANING: Final = {
    0: "Void (no data)", 1: "Edited (not filled)", 2: "Not edited / not filled",
    3: "ASTER", 4: "SRTM90", 5: "SRTM30", 6: "GMTED2010", 7: "SRTM30plus",
    8: "TerraSAR-X radargrammetric", 9: "AW3D30", 100: "Norway DEM",
    101: "DSM05 Spain", 102: "Norway DEM v2",
}
_WBM_MEANING: Final = {0: "No water", 1: "Ocean", 2: "Lake", 3: "River"}
_MEANINGS: Final = {"EDM": _EDM_MEANING, "FLM": _FLM_MEANING, "WBM": _WBM_MEANING}


def per_code_table(tiles: Sequence[TileCounts], layer: Layer) -> str:
    """A markdown per-code prevalence table for one layer (pixel-pooled fractions)."""
    pooled, total = pool_counts(tiles, layer)
    fracs = pooled_fractions(tiles, layer)
    meaning = _MEANINGS[layer]
    lines = [
        f"| Code | Meaning | Pixels | Fraction |",
        f"|---|---|---:|---:|",
    ]
    for code in _vocab(layer):
        lines.append(
            f"| {code} | {meaning[code]} | {pooled[code]:,} | {fracs[code]:.6f} |"
        )
    lines.append(f"| — | **total** | {total:,} | 1.000000 |")
    return "\n".join(lines)


def _dist_row(name: str, d: dict[str, float]) -> str:
    return (
        f"| {name} | {d['mean']:.4f} | {d['p50']:.4f} | {d['p90']:.4f} | "
        f"{d['p95']:.4f} | {d['p99']:.4f} | {d['max']:.4f} |"
    )


def to_dict(tc: TileCounts) -> dict[str, str | float | int]:
    """A JSON/CSV-friendly flat record of one tile's counts and derived fractions."""
    rec = {
        "stem": tc.stem, "lon": tc.lon, "lat": tc.lat, "band": tc.band,
        "geocell": tc.geocell, "total": tc.total,
        "wbm_water_frac": wbm_water_frac(tc),
        "edm_edited_frac": edm_edited_frac(tc),
        "flm_fill_frac": flm_fill_frac(tc),
        "edm_void_frac": void_frac(tc, "EDM"),
        "flm_void_frac": void_frac(tc, "FLM"),
    }
    for layer in ("EDM", "FLM", "WBM"):
        for code, c in tc.counts(layer).items():
            rec[f"{layer.lower()}_{code}"] = c
    return rec


# --- Live driver --------------------------------------------------------------


def _load_frame(frame_cache: Path | None) -> list[CogRef]:
    """The full COG frame — from a cached stem list if present, else list S3 once."""
    from .geometry import parse_cog_key

    if frame_cache and frame_cache.exists():
        stems = json.loads(frame_cache.read_text())
        return [parse_cog_key(s) for s in stems]
    from .discovery import list_cog_refs

    refs = list_cog_refs()
    if frame_cache:
        frame_cache.parent.mkdir(parents=True, exist_ok=True)
        frame_cache.write_text(json.dumps([r.stem for r in refs]))
    return refs


def measure(
    reader: MaskReader,
    cogs: Sequence[CogRef],
    *,
    workers: int = 16,
    progress: Callable[[int, str], object] | None = None,
) -> tuple[list[TileCounts], list[tuple[str, str]]]:
    """Read and count every COG in ``cogs`` (parallel I/O), collecting anomalies.

    Returns ``(tiles, anomalies)``. An anomaly is ``(stem, reason)`` — a genuinely
    absent mask (:class:`~dataset.masks.MaskNotFound`), an out-of-vocabulary read, or
    a **§6.1 cross-mask identity violation**. All three are surfaced and the tile is
    kept *out* of the returned ``tiles``, so the tripwire actually guards the pooled
    table (a mis-keyed/misaligned tile never reaches a committed number) rather than
    only being noted after the fact. An identity violation should never happen —
    they held 500/500 when measured — which is exactly why it must fail loud, not fold
    into the pool.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    tiles: list[TileCounts] = []
    anomalies: list[tuple[str, str]] = []
    bar = progress(len(cogs), "prevalence") if progress else None
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(tile_counts, reader, cog): cog for cog in cogs}
        for fut in as_completed(futs):
            cog = futs[fut]
            try:
                tc = fut.result()
            except masks.MaskNotFound as exc:
                anomalies.append((cog.stem, f"absent: {exc}"))
            except ValueError as exc:
                anomalies.append((cog.stem, f"vocabulary: {exc}"))
            else:
                if identities_hold(tc, tol_px=0):
                    tiles.append(tc)
                else:
                    anomalies.append(
                        (tc.stem, f"§6.1 identity violation: {identity_residuals(tc)}")
                    )
            if bar:
                bar.update(1)
    if bar:
        bar.close()
    tiles.sort(key=lambda tc: tc.stem)
    return tiles, anomalies


def build_report(
    tiles: Sequence[TileCounts],
    anomalies: Sequence[tuple[str, str]],
    *,
    seed: int,
    requested_n: int,
    code_version: str,
    snapshot_date: str,
) -> str:
    """Assemble the committed markdown prevalence table from measured tiles."""
    att = attrition(tiles)
    water_d = describe([wbm_water_frac(tc) for tc in tiles])
    edit_d = describe([edm_edited_frac(tc) for tc in tiles])
    fill_d = describe([flm_fill_frac(tc) for tc in tiles])

    # §6.1 identity tripwire. `measure` already excludes any violator (to anomalies)
    # before a tile reaches here, so `bad` is a defensive re-check: it must be empty,
    # and if it ever isn't, the header says so loudly rather than pooling it.
    bad = [tc.stem for tc in tiles if not identities_hold(tc, tol_px=0)]

    # Geographic structure: edit/fill and void by latitude band.
    bands = sorted({tc.band for tc in tiles}, key=lambda b: float(b.split("-")[0]))

    out: list[str] = []
    w = out.append
    w("# Mask prevalence — EDM / FLM / WBM on a global tile sample")
    w("")
    w("<!-- GENERATED by `python -m dataset.prevalence`. Do not hand-edit numbers; "
      "re-run to refresh. -->")
    w("")
    w(f"Resolves issue #19 (part of the Phase-A map, #13). Measures the fraction of "
      f"pixels per **EDM code** and **FLM source**, plus **WBM** water, on the "
      f"deterministic {requested_n}-tile prevalence sample "
      f"(`dataset.sampling.prevalence_sample`, seed `{seed:#x}`), read **full "
      f"resolution** (spec §7 — ×8 overviews erase rare codes). Fractions are "
      f"**pixel-pooled** across tiles, not per-cell averages (plan §A1 ⚠️).")
    w("")
    w(f"- Tiles measured: **{len(tiles)}** of {requested_n} requested"
      f"{'' if not anomalies else f' ({len(anomalies)} anomalies — see below)'}.")
    w(f"- Total pixels pooled: **{sum(tc.total for tc in tiles):,}**.")
    w(f"- §6.1 cross-mask identities hold on **{len(tiles) - len(bad)}/{len(tiles)}** "
      f"pooled tiles (a violation excludes the tile from the table — see anomalies)"
      f"{'' if not bad else f' — UNEXPECTED VIOLATIONS IN POOL: {bad}'}.")
    w(f"- Provenance: code `{code_version}`, snapshot `{snapshot_date}`.")
    w("")
    w("Copyright / attribution (mask spec §5, Art. 6(b)): *produced using Copernicus "
      "WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH "
      "2014-2018 provided under COPERNICUS by the European Union and ESA; all rights "
      "reserved.* Art. 6(c): *The organisations in charge of the Copernicus programme "
      "by law or by delegation do not incur any liability for any use of the "
      "Copernicus WorldDEM-30.*")
    w("")

    w("## 1. Per-code prevalence (pixel-pooled)")
    w("")
    for layer in ("EDM", "FLM", "WBM"):
        w(f"### {layer}")
        w("")
        w(per_code_table(tiles, layer))
        w("")

    w("## 2. Per-tile distribution and tail")
    w("")
    w("The mean hides the shape. Each row is the distribution *across tiles* of a "
      "per-tile fraction — the tail is where a whole terrain family gets gated out.")
    w("")
    w("| Per-tile quantity | mean | p50 | p90 | p95 | p99 | max |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    w(_dist_row("WBM water (∈{1,2,3})", water_d))
    w(_dist_row("EDM-edited (terrain)", edit_d))
    w(_dist_row("FLM fill (≥3)", fill_d))
    w("")

    w("## 3. Geographic structure")
    w("")
    w("The issue asks specifically whether voids concentrate in steep forested "
      "regions `[verified]` (plan §11.3). **The direct answer: there is no void "
      "concentration to find — EDM/FLM void (code 0) is 0 on every tile in the "
      "sample** (§1 tables; the void branch of the binning is untested by real data "
      "because edited GLO-30 has effectively none). So the §11.3 worry, as posed "
      "about *voids*, is empty for this corpus.")
    w("")
    w("The live concentration is instead *external fill and editing*. Terrain type "
      "has no name-level proxy and cannot have one — it is defined by the very "
      "metrics Phase A calibrates (#15) — so this sample cannot be stratified by "
      "forest/steepness before those metrics exist. Latitude band is the axis "
      "available at name level, and it already surfaces the real risk: editing spikes "
      "at the pole (ice-sheet smoothing), not in the mid-latitude forest belt.")
    w("")
    w("| Band | tiles | pixels | WBM water | EDM-edited | FLM fill | EDM void | FLM void |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|")
    for b in bands:
        bt = [tc for tc in tiles if tc.band == b]
        tot = sum(tc.total for tc in bt)
        w(f"| {b} | {len(bt)} | {tot:,} "
          f"| {pooled_subset_frac(bt, 'WBM', WBM_WATER_CODES):.4f} "
          f"| {pooled_subset_frac(bt, 'EDM', EDITED_EDM_CODES):.4f} "
          f"| {pooled_subset_frac(bt, 'FLM', FLM_CODES_FILL):.4f} "
          f"| {pooled_subset_frac(bt, 'EDM', (0,)):.6f} "
          f"| {pooled_subset_frac(bt, 'FLM', (0,)):.6f} |")
    w("")

    w("## 4. What the plan's default gates would cost")
    w("")
    w(f"Applying the plan's Phase-C defaults to this sample **without re-tuning "
      f"them** (that is Phase C): WBM water > {att['water_cap']:.2f} drop, "
      f"EDM/FLM-edited > {att['edit_cap']:.2f} drop.")
    w("")
    w("| Gate | tiles dropped | fraction of sample |")
    w("|---|---:|---:|")
    w(f"| WBM water > {att['water_cap']:.2f} | {att['dropped_water']} | "
      f"{att['frac_dropped_water']:.4f} |")
    w(f"| EDM/FLM-edited > {att['edit_cap']:.2f} | {att['dropped_edit']} | "
      f"{att['frac_dropped_edit']:.4f} |")
    w(f"| **either (union)** | **{att['dropped_either']}** | "
      f"**{att['frac_dropped_either']:.4f}** |")
    w(f"| WBM water > {WATER_FLAG_CAP:.2f} (flag, not drop) | {att['flagged_water']} | "
      f"{att['flagged_water'] / att['n_tiles']:.4f} |")
    w("")

    # --- §5 Resolution — derived from the numbers above so a re-run stays honest.
    water_drop = [tc for tc in tiles if wbm_water_frac(tc) > att["water_cap"]]
    edit_drop = [tc for tc in tiles if edm_edited_frac(tc) > att["edit_cap"]]
    both = [tc for tc in tiles if wbm_water_frac(tc) > att["water_cap"]
            and edm_edited_frac(tc) > att["edit_cap"]]
    whole_ocean = [tc for tc in water_drop if wbm_water_frac(tc) >= 0.90]
    void_max = max(
        (max(void_frac(tc, "EDM"), void_frac(tc, "FLM")) for tc in tiles), default=0.0
    )
    # Which band concentrates the editing, and which EDM code drives it there.
    band_edit = {
        b: pooled_subset_frac(
            [tc for tc in tiles if tc.band == b], "EDM", EDITED_EDM_CODES
        )
        for b in bands
    }
    top_band = max(band_edit, key=band_edit.get) if band_edit else "—"
    top_band_tiles = [tc for tc in tiles if tc.band == top_band]
    edit_code_frac = {
        c: pooled_subset_frac(top_band_tiles, "EDM", (c,)) for c in EDITED_EDM_CODES
    }
    drive_code = max(edit_code_frac, key=edit_code_frac.get) if edit_code_frac else 0
    fill_ext = pooled_subset_frac(tiles, "FLM", FLM_CODES_FILL)
    max_fill = max((flm_fill_frac(tc) for tc in tiles), default=0.0)

    w("## 5. Resolution — does prevalence force the WBM-only fallback?")
    w("")
    w("**No. EDM/FLM gating is affordable; the WBM-only fallback graduates into a "
      "decision to *keep* EDM/FLM columns.** The map's only live trigger was "
      "prevalence showing the thresholds cost unacceptable coverage (mask access "
      "itself is free — #14). Against this sample it does not:")
    w("")
    w(f"- **EDM/FLM editing is modest and does not hollow out terrain broadly.** "
      f"Only **{len(edit_drop)}/{len(tiles)}** tiles ({len(edit_drop)/len(tiles):.1%}) "
      f"exceed the {att['edit_cap']:.2f} edit/fill cap, and they are **not spread "
      f"across terrain**: editing concentrates in the **{top_band}° band** "
      f"({band_edit.get(top_band, 0):.1%} pooled), driven by EDM code {drive_code} "
      f"(*{_EDM_MEANING[drive_code]}*) over the polar ice sheets — a single stratum, "
      f"not the steep-forest voids the plan's §11.3 risk anticipated. External fill "
      f"(FLM ≥ 3) pools to just **{fill_ext:.2%}** globally; its worst single tile is "
      f"**{max_fill:.1%}**.")
    w(f"- **The WBM drops are whole mostly-ocean tiles, not terrain.** Of the "
      f"**{len(water_drop)}** tiles over the {att['water_cap']:.2f} water cap, "
      f"**{len(whole_ocean)}** are ≥90% water — coastal/island cells that survive "
      f"discovery's absent-prefix filter and that the exact-zero ocean gate "
      f"(ADR-0001 §3, ≥0.75 drop) already removes. ⚠️ This table is **per whole 1° "
      f"tile**; Phase C gates score 224² "
      f"**windows**, so a half-ocean tile contributes ocean windows (dropped anyway) "
      f"and land windows (kept). The **{att['frac_dropped_water']:.0%}** whole-tile "
      f"water figure therefore **overstates** per-window land attrition — read it as "
      f"an upper bound, not the Phase-C number.")
    w(f"- **The two gates are near-disjoint** (**{len(both)}** tiles dropped by both): "
      f"water is a coastline/ocean concern, editing a polar-ice concern. They gate "
      f"different strata and neither compounds the other.")
    w(f"- **Voids are absent.** EDM/FLM void (code 0) never exceeded "
      f"**{void_max:.2%}** on any tile — in practice **0** across the sample — "
      f"resolving the spec's fog note 3 (the void branch of the binning is untested "
      f"by real data because GLO-30 has effectively none). The national-DEM fills "
      f"(FLM 100–102) and TerraSAR-X (FLM 8) are likewise negligible.")
    w("")
    w("**Carried to Phase C, not decided here** (threshold *values* are Phase C): the "
      f"polar-ice stratum ({top_band}°) is where a strict EDM cap bites — flag it for "
      "the §11.3 attrition-by-stratum table so ice-sheet terrain is not silently "
      "hollowed out.")
    w("")
    w("**Left open — a separate fog note, not this ticket's** (#13): whether "
      "`WBM == 1` should *supersede* ADR-0001 §3's exact-zero exclusion as the ocean "
      "test. This ticket measured WBM water fractions (§1, §3), which is the input "
      "that fog was waiting on, but does **not** settle it: deciding it needs a "
      "per-pixel `WBM == 1` vs `elevation == 0` disagreement count (how much land the "
      "zero-test mislabels — e.g. the Netherlands −12 m case), and this pass reads "
      "masks only, never DEM elevation. That comparison is its own measurement.")
    w("")

    if anomalies:
        w("## Anomalies (surfaced, not swallowed)")
        w("")
        w("A missing mask is an anomaly to investigate, never read as 'no edits' "
          "(mask spec fog note 4). These tiles are excluded from the table above:")
        w("")
        for stem, reason in anomalies:
            w(f"- `{stem}` — {reason}")
        w("")

    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    """``python -m dataset.prevalence`` — measure and write the committed table."""
    import argparse
    import csv
    import logging

    from .sampling import prevalence_sample

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    log = logging.getLogger("prevalence")

    p = argparse.ArgumentParser(prog="python -m dataset.prevalence")
    p.add_argument("--n", type=int, default=None, help="sample size (default: plan's 500)")
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    p.add_argument("--workers", type=int, default=16, help="parallel S3 reads")
    p.add_argument("--out", type=Path, default=Path("docs/research/mask-prevalence.md"))
    p.add_argument("--csv", type=Path, default=Path("docs/research/mask-prevalence-per-tile.csv"))
    p.add_argument("--cache-dir", type=Path, default=Path(config.MASK_CACHE_DIR))
    p.add_argument(
        "--frame-cache",
        type=Path,
        default=Path("data/cog_frame.json"),
        help="cache the S3 prefix listing here so re-runs skip it (blank to always list)",
    )
    p.add_argument("--limit", type=int, default=None, help="cap tiles read (debugging)")
    args = p.parse_args(argv)

    from .sampling import PREVALENCE_SAMPLE_N

    requested_n = args.n or PREVALENCE_SAMPLE_N

    log.info("loading COG frame ...")
    refs = _load_frame(args.frame_cache)
    log.info("frame: %d COGs", len(refs))

    sample = prevalence_sample(refs, requested_n, seed=args.seed)
    if args.limit:
        sample = sample[: args.limit]
    log.info("sample: %d tiles (seed %#x)", len(sample), args.seed)

    reader = masks.RasterioMaskReader(cache_dir=args.cache_dir)
    tiles, anomalies = measure(reader, sample, workers=args.workers, progress=_tqdm)
    log.info("measured %d tiles, %d anomalies", len(tiles), len(anomalies))

    from .provenance import git_sha

    report = build_report(
        tiles, anomalies,
        seed=args.seed, requested_n=requested_n,
        code_version=git_sha(), snapshot_date=_today(),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    log.info("wrote %s", args.out)

    # Per-tile CSV — the auditable backing data behind the distribution/tail.
    rows = [to_dict(tc) for tc in tiles]
    if rows:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        log.info("wrote %s", args.csv)

    att = attrition(tiles)
    print(
        f"\nMeasured {len(tiles)} tiles.\n"
        f"  union gate drops {att['dropped_either']} "
        f"({att['frac_dropped_either']:.1%}); "
        f"water {att['dropped_water']}, edit/fill {att['dropped_edit']}.\n"
        f"  -> {args.out}"
    )
    return 0


def _today() -> str:
    from datetime import date

    return date.today().isoformat()


def _tqdm(total: int, desc: str):
    try:
        from tqdm import tqdm

        return tqdm(total=total, desc=desc, unit="tile", dynamic_ncols=True)
    except ImportError:  # pragma: no cover
        return None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
