"""The hand-picked flat-wet / flat-dry exemplar set — Phase B3 (issue #25).

B3 asks whether the surviving metrics *separate* wet from dry, and that question
needs labelled exemplars **the algorithm did not choose for itself**. The pilot
frame (:mod:`dataset.pilot`, #24) is a representative draw — it cannot answer a
separation question because it carries no ground-truth class. This module supplies
that class: a curated list of named low-relief locations, each tagged ``flat-wet``
or ``flat-dry``, resolved to a specific production window and scored through the
**identical B1 frame** so the exemplars live in the same descriptor space as the
pilot.

**Same frame, hand-picked draw.** Where :mod:`dataset.pilot` wraps
:func:`dataset.sampling.sample_cogs` for a cell-balanced draw, this resolves each
named location to its COG (:func:`dataset.geometry.cog_at`, floor to the SW corner)
and its **nearest production grid window** (:func:`nearest_window`), then scores that
one window with the pilot's own :func:`dataset.pilot.score_window` and
:func:`dataset.pilot.window_mask_fractions`. Every metric therefore reads at its
Gate A3 production resolution (#21) and every mask fraction uses the prevalence
groupings — the scores are directly comparable to the pilot's, cell for cell.

**The classes (the plan's named hard cases, #25).**

*flat-wet* — low-relief terrain where flattened water / saturated ground reads as
spurious zero-texture: a floodplain, a wetland, a playa, a delta.

*flat-dry* — flat plateaus of comparable low relief but no surface water, the
control the separation question needs: is the algorithm's zero-texture reading
*water*, or just *flatness*? If a surviving metric cannot tell the two apart, B3's
TPI re-entry (#22) is triggered.

The read path mirrors :mod:`dataset.pilot`: a pure, unit-tested core
(:func:`nearest_window`, :func:`score_exemplar`, :func:`to_rows`,
:func:`build_report`) and a thin live driver (:func:`main`) that reads each COG
from S3 and writes the committed CSV + markdown cover sheet.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from . import config
from .geometry import (
    CogRef,
    cog_at,
    grid_windows,
    lat_band,
    window_center_lonlat,
)
from .masks import MaskReader
from .metrics import METRICS
from .pilot import (
    MASK_FRACTIONS,
    OVERVIEW_FACTOR,
    WindowRow,
    _read_mask_windows,
    _resolution_of,
    _stat_cells,
    score_window,
)
from .sampling import geocell_of
from .tiling import CogReader, Overview

#: The two exemplar classes. ``flat-wet`` is the hard case (spurious zero-texture
#: over water/saturated ground); ``flat-dry`` is the control of comparable relief.
FLAT_WET: Final = "flat-wet"
FLAT_DRY: Final = "flat-dry"


@dataclass(frozen=True)
class Exemplar:
    """A named location tagged with its ground-truth flat-wet / flat-dry class.

    ``lon``/``lat`` are a representative point *inside* the feature; the scored
    window is the production grid window nearest that point (:func:`nearest_window`),
    so the exemplar id is a real window the pipeline would tile, not a bespoke crop.
    ``note`` records why the location is a hard case, for the committed cover sheet.
    """

    name: str
    klass: str
    lon: float
    lat: float
    note: str = ""

    @property
    def cog(self) -> CogRef:
        """The source COG whose 1° cell contains the point (floor to SW corner)."""
        return cog_at(self.lon, self.lat)


#: The curated exemplar set (#25). Four flat-wet hard cases the plan names — a
#: floodplain, a wetland, a playa, a delta — and four flat-dry plateaus of
#: comparable low relief but no surface water. Points are interior to each feature;
#: :func:`nearest_window` snaps them to the production grid.
EXEMPLARS: Final[tuple[Exemplar, ...]] = (
    # --- flat-wet: flattened water / saturated ground -> spurious zero-texture ---
    Exemplar(
        "Mississippi Alluvial Plain", FLAT_WET, lon=-90.9, lat=33.4,
        note="Mississippi floodplain near Greenville MS — flat, saturated alluvium.",
    ),
    Exemplar(
        "Pantanal", FLAT_WET, lon=-57.3, lat=-17.6,
        note="World's largest tropical wetland (Brazil) — seasonally flooded flats.",
    ),
    Exemplar(
        "Etosha Pan", FLAT_WET, lon=16.4, lat=-18.85,
        note="Endorheic salt playa (Namibia) — vast flat pan, seasonally flooded clay.",
    ),
    Exemplar(
        "Ganges–Brahmaputra delta", FLAT_WET, lon=89.9, lat=22.4,
        note="Sundarbans delta (Bangladesh) — near-sea-level, river/tidal water-laced.",
    ),
    # --- flat-dry: comparable low relief, no surface water (the control) ---------
    Exemplar(
        "Nullarbor Plain", FLAT_DRY, lon=129.4, lat=-31.2,
        note="Treeless limestone plateau (Australia) — extremely flat, no drainage.",
    ),
    Exemplar(
        "Llano Estacado", FLAT_DRY, lon=-102.5, lat=34.2,
        note="High Plains mesa (Texas/NM) — flat semi-arid tableland, no water.",
    ),
    Exemplar(
        "Kalahari sand plain", FLAT_DRY, lon=22.6, lat=-22.6,
        note="Semi-arid sand plain (Botswana) — flat, dry, internally drained.",
    ),
    Exemplar(
        "Tanezrouft Basin", FLAT_DRY, lon=1.5, lat=24.5,
        note="Hyper-arid Sahara plain (Algeria) — 'land of terror', flat, no water.",
    ),
)


# --- Resolving a point to its production window -------------------------------


def nearest_window(
    full: Overview, cog: CogRef, lon: float, lat: float, tile: int = config.TILE_PX
) -> tuple[int, int]:
    """The ``(win_row0, win_col0)`` production grid window nearest a target point.

    Lays the same non-overlapping window grid production tiles the COG with
    (:func:`~dataset.geometry.grid_windows`, ADR-0001 §2) and returns the origin of the
    window whose *centre* is closest to ``(lon, lat)`` in degrees. Nearest-centre, not
    containing-window, so a point that lands in the flushed trailing window's overlap
    resolves deterministically to a single window. ``full`` is the native-resolution
    :class:`~dataset.tiling.Overview`, whose :attr:`~dataset.tiling.Overview.full_w` /
    :attr:`~dataset.tiling.Overview.full_h` carry the COG's degree span and
    poleward-narrowed column count, so the centre coordinates match what the frame
    records.
    """
    full_w, full_h = full.full_w, full.full_h

    def dist2(origin: tuple[int, int]) -> float:
        wlon, wlat = window_center_lonlat(
            origin_lon=cog.origin_lon,
            origin_lat=cog.origin_lat,
            width=full_w,
            height=full_h,
            win_row0=origin[0],
            win_col0=origin[1],
            tile=tile,
        )
        return (wlon - lon) ** 2 + (wlat - lat) ** 2

    return min(grid_windows(full_w, full_h, tile), key=dist2)


# --- One scored exemplar ------------------------------------------------------


@dataclass(frozen=True)
class ExemplarRow:
    """One scored exemplar: its class label joined to a pilot :class:`WindowRow`.

    The :attr:`window` is scored by exactly the pilot's core, so its metrics, mask
    fractions and :attr:`~WindowRow.window_id` are the same objects the pilot frame
    holds — the exemplar simply carries a ground-truth :attr:`~Exemplar.klass` the
    representative draw cannot.
    """

    exemplar: Exemplar
    window: WindowRow


def score_exemplar(
    cog_reader: CogReader,
    mask_reader: MaskReader,
    exemplar: Exemplar,
    *,
    tile: int = config.TILE_PX,
    factor: int = OVERVIEW_FACTOR,
) -> tuple[ExemplarRow, tuple[str, str] | None]:
    """Resolve, read and score one exemplar's window through the B1 frame.

    Two DEM reads (full-res + ×``factor``) then the three masks at the single
    resolved origin — the pilot's per-COG shape narrowed to one window. Returns
    ``(row, mask_anomaly)``; ``mask_anomaly`` is ``None`` unless a mask was
    absent/unreadable, in which case the mask columns are NaN but the metrics stand
    (mask spec fog note 4, as in :func:`dataset.pilot.tile_rows`).
    """
    cog = exemplar.cog
    full = cog_reader.read_overview(cog, 1)
    ov = cog_reader.read_overview(cog, factor)

    win_row0, win_col0 = nearest_window(full, cog, exemplar.lon, exemplar.lat, tile)
    wlon, wlat = window_center_lonlat(
        origin_lon=cog.origin_lon,
        origin_lat=cog.origin_lat,
        width=full.full_w,
        height=full.full_h,
        win_row0=win_row0,
        win_col0=win_col0,
        tile=tile,
    )
    scores = score_window(
        full.array,
        ov.array,
        win_row0=win_row0,
        win_col0=win_col0,
        full_h=full.full_h,
        full_w=full.full_w,
        ov_h=ov.ov_h,
        ov_w=ov.ov_w,
        lat=wlat,
        tile=tile,
    )
    mask_fracs, anomaly = _read_mask_windows(
        mask_reader, cog, [(win_row0, win_col0)], tile
    )
    window = WindowRow(
        stem=cog.stem,
        band=lat_band(wlat),
        geocell=geocell_of(cog),
        lon=wlon,
        lat=wlat,
        win_row0=win_row0,
        win_col0=win_col0,
        metrics=scores,
        masks=mask_fracs[0],
    )
    return ExemplarRow(exemplar=exemplar, window=window), anomaly


def measure(
    cog_reader: CogReader,
    mask_reader: MaskReader,
    exemplars: Sequence[Exemplar] = EXEMPLARS,
    *,
    tile: int = config.TILE_PX,
    factor: int = OVERVIEW_FACTOR,
    progress: Callable[[int, str], object] | None = None,
) -> tuple[list[ExemplarRow], list[tuple[str, str]]]:
    """Score every exemplar in order, surfacing (never swallowing) read failures.

    Serial by design: the set is a dozen named windows, not a 15k-window frame, so
    the pilot's thread-pool machinery is unwarranted. A COG that cannot be read at
    all is recorded in ``anomalies`` and dropped from the rows; an absent *mask*
    keeps the row with NaN mask columns and its own anomaly (:func:`score_exemplar`).
    """
    bar = progress(len(exemplars), "exemplars") if progress else None
    rows: list[ExemplarRow] = []
    anomalies: list[tuple[str, str]] = []
    for ex in exemplars:
        try:
            row, anomaly = score_exemplar(
                cog_reader, mask_reader, ex, tile=tile, factor=factor
            )
        except Exception as exc:  # noqa: BLE001 - surfaced, never swallowed
            anomalies.append((ex.name, f"{type(exc).__name__}: {exc}"))
            if bar:
                bar.update(1)
            continue
        if anomaly is not None:
            anomalies.append(anomaly)
        rows.append(row)
        if bar:
            bar.update(1)
    if bar:
        bar.close()
    return rows, anomalies


# --- Report + rows ------------------------------------------------------------


def to_rows(rows: Sequence[ExemplarRow]) -> list[dict[str, float | str | int]]:
    """Flat per-exemplar records for the committed CSV: name -> window id -> scores.

    One row per exemplar: the ground-truth label and location, the resolved
    ``window_id`` and its centre, then the nine metrics (each at its production
    resolution) and the three mask fractions — the **same score columns** as the pilot
    CSV (the label/location columns are extra), so a row's descriptor vector drops
    straight into the pilot's descriptor space.
    """
    out: list[dict[str, float | str | int]] = []
    for r in rows:
        ex, w = r.exemplar, r.window
        rec: dict[str, float | str | int] = {
            "name": ex.name,
            "klass": ex.klass,
            "window_id": w.window_id,
            "stem": w.stem,
            "band": w.band,
            "geocell": w.geocell,
            "target_lon": ex.lon,
            "target_lat": ex.lat,
            "win_lon": w.lon,
            "win_lat": w.lat,
            "win_row0": w.win_row0,
            "win_col0": w.win_col0,
        }
        for metric in METRICS:
            rec[metric] = w.metrics[metric]
        for frac in MASK_FRACTIONS:
            rec[frac] = w.masks[frac]
        rec["note"] = ex.note
        out.append(rec)
    return out


def _class_cells(rows: Sequence[ExemplarRow], klass: str, metric: str) -> str:
    """The distribution cells for one metric within one class (:func:`_stat_cells`)."""
    return _stat_cells([r.window.metrics[metric] for r in rows if r.exemplar.klass == klass])


def build_report(
    rows: Sequence[ExemplarRow],
    anomalies: Sequence[tuple[str, str]],
    *,
    factor: int,
    code_version: str,
    snapshot_date: str,
) -> str:
    """Assemble the committed markdown cover sheet (the CSV holds the per-row data)."""
    wet = [r for r in rows if r.exemplar.klass == FLAT_WET]
    dry = [r for r in rows if r.exemplar.klass == FLAT_DRY]
    _res_label = {"full": "full-res", "overview": f"×{factor} overview", "box": "2× box (ov)"}

    out: list[str] = []
    w = out.append
    w("# Phase-B flat-wet / flat-dry exemplar set — ground-truth contrast for B3")
    w("")
    w("<!-- GENERATED by `python -m dataset.exemplars`. Do not hand-edit numbers; "
      "re-run to refresh. The per-exemplar CSV is the artifact; this is its cover sheet. -->")
    w("")
    w(f"Resolves issue #25 (part of the Phase-B map, #23). A hand-picked set of "
      f"**{len(wet)} flat-wet** and **{len(dry)} flat-dry** locations, each resolved to "
      f"a specific production window and scored through the **same B1 frame** the pilot "
      f"uses (:mod:`dataset.pilot`): every metric at its Gate A3 resolution (#21) plus "
      f"the three EDM/FLM/WBM mask fractions. B3 asks whether the surviving metrics "
      f"*separate* wet from dry; this is the labelled contrast set that question needs — "
      f"windows the algorithm did **not** choose for itself.")
    w("")
    w(f"- Exemplars: **{len(rows)}** scored"
      f"{'' if not anomalies else f' ({len(anomalies)} anomalies — see below)'}. "
      f"Each is the production grid window nearest a point interior to the named feature.")
    w(f"- Data: the per-exemplar scores are committed alongside this sheet in "
      f"[`exemplars-per-window.csv`](exemplars-per-window.csv) — the artifact B3 reads; "
      f"this sheet is its cover.")
    w(f"- Provenance: code `{code_version}`, snapshot `{snapshot_date}`, "
      f"overview ×{factor}.")
    w("")
    w("Copyright / attribution (mask spec §5, Art. 6(b)): *produced using Copernicus "
      "WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH "
      "2014-2018 provided under COPERNICUS by the European Union and ESA; all rights "
      "reserved.* Art. 6(c): *The organisations in charge of the Copernicus programme "
      "by law or by delegation do not incur any liability for any use of the "
      "Copernicus WorldDEM-30.*")
    w("")

    w("## 1. The exemplars (name → class → window id)")
    w("")
    w("| Name | class | window id | centre (lon, lat) | note |")
    w("|---|---|---|---|---|")
    for r in rows:
        ex, wr = r.exemplar, r.window
        w(f"| {ex.name} | `{ex.klass}` | `{wr.window_id}` | "
          f"{wr.lon:.3f}, {wr.lat:.3f} | {ex.note} |")
    w("")

    w("## 2. Per-class metric distributions (the side-by-side B3 will read)")
    w("")
    w("Per-metric distribution within each class — the descriptor space B3's "
      "separation test operates on. This cover sheet renders **no verdict** (that is "
      "B3's decision, #23); it only lays the labelled columns out. Each cell is "
      "`n | mean | std | min | p50 | p90 | max` over the class's exemplars (NaN "
      "dropped); a metric that separates would show non-overlapping wet vs dry centres.")
    w("")
    w("| Metric | resolution | class | n | mean | std | min | p50 | p90 | max |")
    w("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for metric in METRICS:
        res = _res_label[_resolution_of(metric)]
        w(f"| `{metric}` | {res} | wet | {_class_cells(rows, FLAT_WET, metric)} |")
        w(f"| | | dry | {_class_cells(rows, FLAT_DRY, metric)} |")
    w("")

    w("## 3. Mask fractions by class")
    w("")
    w("The wet cases should carry water (`wbm_water_frac`) the dry controls do not — a "
      "sanity check that the labels reach the masks, not only the terrain metrics.")
    w("")
    w("| Mask fraction | class | n | mean | std | min | p50 | p90 | max |")
    w("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for frac in MASK_FRACTIONS:
        for label, group in ((FLAT_WET, wet), (FLAT_DRY, dry)):
            cells = _stat_cells([r.window.masks[frac] for r in group])
            w(f"| `{frac}` | {label} | {cells} |")
    w("")

    if anomalies:
        w("## Anomalies (surfaced, not swallowed)")
        w("")
        for name, reason in anomalies:
            w(f"- `{name}` — {reason}")
        w("")

    return "\n".join(out) + "\n"


# --- Live driver --------------------------------------------------------------


def _today() -> str:
    from datetime import date

    return date.today().isoformat()


def _tqdm(total: int, desc: str):
    try:
        from tqdm import tqdm

        return tqdm(total=total, desc=desc, unit="ex", dynamic_ncols=True)
    except ImportError:  # pragma: no cover
        return None


def main(argv: list[str] | None = None) -> int:
    """``python -m dataset.exemplars`` — score the curated set and write the artifacts."""
    import argparse
    import csv
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    log = logging.getLogger("exemplars")

    p = argparse.ArgumentParser(prog="python -m dataset.exemplars")
    p.add_argument("--factor", type=int, default=OVERVIEW_FACTOR, help="overview decimation")
    p.add_argument("--out", type=Path, default=Path("docs/research/exemplars.md"))
    p.add_argument("--csv", type=Path,
                   default=Path("docs/research/exemplars-per-window.csv"))
    p.add_argument("--cache-dir", type=Path, default=Path(config.MASK_CACHE_DIR))
    args = p.parse_args(argv)

    from .masks import RasterioMaskReader
    from .tiling import RasterioCogReader

    cog_reader = RasterioCogReader()
    mask_reader = RasterioMaskReader(cache_dir=args.cache_dir)

    log.info("scoring %d exemplars ...", len(EXEMPLARS))
    rows, anomalies = measure(
        cog_reader, mask_reader, EXEMPLARS, factor=args.factor, progress=_tqdm
    )
    log.info("scored %d exemplars, %d anomalies", len(rows), len(anomalies))

    from .build import git_sha

    report = build_report(
        rows,
        anomalies,
        factor=args.factor,
        code_version=git_sha(),
        snapshot_date=_today(),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    log.info("wrote %s", args.out)

    flat = to_rows(rows)
    if flat:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(flat[0].keys()))
            wr.writeheader()
            wr.writerows(flat)
        log.info("wrote %s", args.csv)

    print(
        f"\nScored {len(rows)} exemplars ({len(anomalies)} anomalies).\n"
        f"  -> {args.out}\n  -> {args.csv}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
