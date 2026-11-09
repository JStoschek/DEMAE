"""#186: how many of Task 1's committed test tiles survive plan §3.2's majority rule.

For each tile in ``artifacts/task1/test_tiles.csv`` this computes the fraction of its
footprint covered by its most common Iwahashi et al. (2018) terrain group, reading the
distributor's 7.5 arc-second GeoTIFF (15 groups, uint8 values 1..15, nodata 255) through a
small window per tile, never whole.

Footprint: a tile is a 224x224 full-resolution window of a 1-degree Copernicus GLO-30 COG
(EPSG:4326), so its bounds follow from the COG stem's SW corner, ``win_row0/win_col0`` and the
COG's rows/cols per degree (3600 rows; columns per ``dataset.geometry.COLS_PER_DEGREE``).
Each label pixel is weighted by the area of its overlap with that rectangle (in degrees; the
cos-latitude factor is constant to <0.1% across a 0.06-degree tile and cancels in a fraction).

Two fractions per tile, both reported:

- ``frac_footprint``: majority-group area / whole footprint. Nodata (water, the map's edge
  south of 59.46 S) counts against the majority. This is §3.2's rule as written.
- ``frac_valid``: majority-group area / labelled area only.

Run:  uv run python scripts/p186_iwahashi_survival.py \
          --tif data/iwahashi2018/7.5_arc_second_compressedGeoTIFF/GlobalTerrainClassification_Iwahashi_etal_2018.tif
Writes the per-tile table to data/iwahashi2018/p186_test_tile_fractions.csv (gitignored) and
prints the summary tables the note quotes.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from benchmark import voids
from dataset.config import TILE_PX
from dataset.geometry import COLS_PER_DEGREE, ROWS_PER_DEGREE, lat_band, parse_cog_key

GROUPS = ["1a", "1b", "2a", "2b", "3a", "3b", "4", "5", "6", "7", "8", "9", "10", "11", "12"]
NODATA = 255
THRESHOLDS = (0.60, 0.50)


def footprint(source_cog: str, row0: int, col0: int, lat: float) -> tuple[float, float, float, float]:
    """(west, south, east, north) of a tile window, in degrees."""
    ref = parse_cog_key(source_cog)
    width = COLS_PER_DEGREE[lat_band(lat)]
    west = ref.origin_lon + col0 / width
    east = ref.origin_lon + (col0 + TILE_PX) / width
    north = ref.origin_lat + 1 - row0 / ROWS_PER_DEGREE
    south = ref.origin_lat + 1 - (row0 + TILE_PX) / ROWS_PER_DEGREE
    return west, south, east, north


def overlap_weights(lo: float, hi: float, start: float, step: float, n: int) -> np.ndarray:
    """Length of overlap of [lo, hi] with each of n cells [start + i*step, start + (i+1)*step]."""
    edges = start + step * np.arange(n + 1)
    return np.clip(np.minimum(edges[1:], hi) - np.maximum(edges[:-1], lo), 0.0, None)


def group_areas(src, bounds) -> np.ndarray:
    """Area (deg^2) of each value 0..255 inside the footprint."""
    west, south, east, north = bounds
    t = src.transform
    c0 = math.floor((west - t.c) / t.a)
    c1 = math.ceil((east - t.c) / t.a)
    r0 = math.floor((north - t.f) / t.e)
    r1 = math.ceil((south - t.f) / t.e)
    areas = np.zeros(256)
    total = (east - west) * (north - south)
    # Rows/cols outside the raster (south of 59.46 S) are nodata by construction.
    rr0, rr1 = max(r0, 0), min(r1, src.height)
    cc0, cc1 = max(c0, 0), min(c1, src.width)
    if rr1 > rr0 and cc1 > cc0:
        arr = src.read(1, window=Window(cc0, rr0, cc1 - cc0, rr1 - rr0))
        wx = overlap_weights(west, east, t.c + cc0 * t.a, t.a, cc1 - cc0)
        # Rows run north to south: the cell edges in latitude descend, so work in -lat.
        wy = overlap_weights(-north, -south, -(t.f + rr0 * t.e), -t.e, rr1 - rr0)
        w = np.outer(wy, wx)
        np.add.at(areas, arr.ravel(), w.ravel())
    areas[NODATA] += max(total - areas.sum(), 0.0)
    return areas / total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tif", required=True)
    ap.add_argument("--tiles", default="artifacts/task1/test_tiles.csv")
    ap.add_argument("--corpus", default="data/corpus_500k")
    ap.add_argument("--out", default="data/iwahashi2018/p186_test_tile_fractions.csv")
    args = ap.parse_args()

    with open(args.tiles) as f:
        tile_ids = [r["tile_id"] for r in csv.DictReader(f)]
    cols = voids.read_manifest_columns(
        args.corpus,
        ["tile_id", "source_cog", "win_row0", "win_col0", "lon", "lat", "split",
         "land_frac", "water_affected", "slope_mean"],
    )
    index = {t: i for i, t in enumerate(cols["tile_id"])}
    bands = voids.Bands.from_manifest(args.corpus)

    rows = []
    with rasterio.open(args.tif) as src:
        for tid in sorted(tile_ids, key=lambda t: -cols["lat"][index[t]]):
            i = index[tid]
            assert cols["split"][i] == "test", tid
            b = footprint(str(cols["source_cog"][i]), int(cols["win_row0"][i]),
                          int(cols["win_col0"][i]), float(cols["lat"][i]))
            # The manifest's lon/lat is the window centre; a footprint that missed it is a bug.
            assert abs((b[0] + b[2]) / 2 - cols["lon"][i]) < 1e-6, tid
            assert abs((b[1] + b[3]) / 2 - cols["lat"][i]) < 1e-6, tid
            a = group_areas(src, b)
            labelled = a[1:16].sum()
            g = int(np.argmax(a[1:16])) + 1
            rows.append(dict(
                tile_id=tid,
                lat=float(cols["lat"][i]),
                band=bands.label(bands.of(float(cols["slope_mean"][i]))),
                clean=bool(voids.eligible(np.array([cols["land_frac"][i]]),
                                          np.array([cols["water_affected"][i]]))[0]),
                majority_group=GROUPS[g - 1] if labelled > 0 else "none",
                frac_footprint=float(a[g]),
                frac_valid=float(a[g] / labelled) if labelled > 0 else 0.0,
                nodata_frac=float(a[NODATA]),
                other_frac=float(1 - a[1:16].sum() - a[NODATA]),
            ))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def summary(pop, name):
        print(f"\n## {name}: {len(pop)} tiles")
        print(f"other (non-1..15, non-nodata) area, max over tiles: {max(r['other_frac'] for r in pop):.4f}")
        print(f"tiles with any nodata: {sum(r['nodata_frac'] > 0 for r in pop)}; "
              f"all-nodata: {sum(r['majority_group'] == 'none' for r in pop)}")
        for key in ("frac_footprint", "frac_valid"):
            fr = np.array([r[key] for r in pop])
            print(f"\n{key}: median {np.median(fr):.3f}, quartiles "
                  f"{np.quantile(fr, 0.25):.3f}/{np.quantile(fr, 0.75):.3f}")
            print("| | n | >=60% | >=50% |")
            print("|---|---:|---:|---:|")
            print(f"| all | {len(pop)} | {(fr >= .6).sum()} | {(fr >= .5).sum()} |")
            for band in ("R1", "R2", "R3", "R4"):
                sel = np.array([r["band"] == band for r in pop])
                print(f"| {band} | {sel.sum()} | {(fr[sel] >= .6).sum()} | {(fr[sel] >= .5).sum()} |")
            print("\nper majority group:")
            print("| group | n (majority) | >=60% | >=50% |")
            print("|---|---:|---:|---:|")
            for grp in GROUPS + ["none"]:
                sel = np.array([r["majority_group"] == grp for r in pop])
                if sel.any():
                    print(f"| {grp} | {sel.sum()} | {(fr[sel] >= .6).sum()} | {(fr[sel] >= .5).sum()} |")

    summary(rows, "all committed test tiles")
    summary([r for r in rows if r["clean"]], "clean test tiles (voids.eligible, #150's filter)")


if __name__ == "__main__":
    main()
