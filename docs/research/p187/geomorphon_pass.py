"""#187: time one buffered geomorphon pass on real tiles.

Run from the repo root with PYTHONPATH=src. Reads buffered windows off s3://copernicus-dem-30m,
checks the centre crop against the shard tile, runs WhiteboxTools Geomorphons, times each stage.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.transform import Affine

from benchmark.evaluate import TileStore
from dataset import config
from dataset.geometry import COLS_PER_DEGREE, format_cog_stem, lat_band, parse_cog_key

WBT = sys.argv[1]
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
T = 224
FRAME = set(json.load(open("data/cog_frame.json")))


def cog_path(stem: str) -> str:
    return f"/vsis3/{config.COPERNICUS_GLO30_BUCKET}/{stem}/{stem}.tif"


def parse(tile_id: str):
    stem, r0, c0 = tile_id.rsplit("_", 2)
    return stem, int(r0), int(c0)


def width_of(origin_lat: int) -> int:
    return COLS_PER_DEGREE[lat_band(origin_lat + 0.5)]


def needed_neighbours(stem, r0, c0, buf):
    c = parse_cog_key(stem)
    W = width_of(c.origin_lat)
    dr = [0] + ([-1] if r0 < buf else []) + ([1] if r0 + T + buf > 3600 else [])
    dc = [0] + ([-1] if c0 < buf else []) + ([1] if c0 + T + buf > W else [])
    return [(a, b, format_cog_stem(c.origin_lon + b, c.origin_lat - a)) for a in dr for b in dc]


def read_buffered(tile_id: str, buf: int):
    """(H, W) float32 window on the tile COG's own grid, +buf px each side. Absent COGs -> 0 m."""
    stem, r0, c0 = parse(tile_id)
    c = parse_cog_key(stem)
    W = width_of(c.origin_lat)
    res_x, res_y = 1.0 / W, 1.0 / 3600
    # COGs are pixel-is-point: the grid origin sits half a pixel west/north of the degree line.
    west = c.origin_lon - 0.5 * res_x + (c0 - buf) * res_x
    north = c.origin_lat + 1 + 0.5 * res_y - (r0 - buf) * res_y
    n = T + 2 * buf
    bounds = (west, north - n * res_y, west + n * res_x, north)
    nbs = needed_neighbours(stem, r0, c0, buf)
    present = [s for _, _, s in nbs if s in FRAME]
    with rasterio.Env(**config.AWS_NO_SIGN_ENV, GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        dss = [rasterio.open(cog_path(s)) for s in present]
        try:
            arr, tr = merge(dss, bounds=bounds, res=(res_x, res_y), nodata=-32767,
                            resampling=Resampling.nearest, dtype="float32")
        finally:
            for d in dss:
                d.close()
    a = arr[0][:n, :n]
    a = np.where(a == -32767, 0.0, a).astype(np.float32)
    meta = dict(n_cogs=len(nbs), absent=len(nbs) - len(present),
                crosses_band=any(width_of(c.origin_lat - a_) != W for a_, _, _ in nbs))
    return a, Affine(res_x, 0, west, 0, -res_y, north), meta


def write_tif(path, arr, transform, crs):
    with rasterio.open(path, "w", driver="GTiff", width=arr.shape[1], height=arr.shape[0], count=1,
                       dtype="float32", crs=crs, transform=transform, nodata=-32768) as d:
        d.write(arr.astype(np.float32), 1)


def geomorphons(arr, transform, crs, L, tmp: Path, tag: str):
    src, dst = tmp / f"{tag}_dem.tif", tmp / f"{tag}_gm.tif"
    write_tif(src, arr, transform, crs)
    t0 = time.perf_counter()
    subprocess.run([WBT, "-r=Geomorphons", f"--dem={src}", f"--output={dst}", f"--search={L}",
                    "--threshold=1.0", "--skip=0", "--fdist=0", "--forms"],
                   check=True, capture_output=True)
    dt = time.perf_counter() - t0
    with rasterio.open(dst) as d:
        return d.read(1), dt


def metric_transform(lat, W):
    """The same grid with ground-metre spacing, in a projected CRS, as a control."""
    dy = 111_320.0 / 3600
    dx = 111_320.0 * np.cos(np.radians(lat)) / W
    return Affine(dx, 0, 0, 0, -dy, 0), CRS.from_epsg(3857), (dy, dx)


def pick_tiles():
    import hashlib
    rows = [r for f in ("head_train_tiles", "val_tiles", "test_tiles")
            for r in csv.DictReader(open(f"artifacts/task1/{f}.csv"))]
    rows.sort(key=lambda r: hashlib.blake2b(r["tile_id"].encode(), digest_size=8).digest())
    by_stratum: dict[str, str] = {}
    for r in rows:
        stem, r0, c0 = parse(r["tile_id"])
        nbs = needed_neighbours(stem, r0, c0, 50)
        if len(nbs) == 1:
            by_stratum.setdefault(r["stratum_id"], r["tile_id"])
    picks = list(by_stratum.values())[:16]
    want = {"corner": None, "band": None, "absent": None, "edge": None}
    for r in rows:
        stem, r0, c0 = parse(r["tile_id"])
        nbs = needed_neighbours(stem, r0, c0, 50)
        c = parse_cog_key(stem)
        if len(nbs) == 4 and not want["corner"]:
            want["corner"] = r["tile_id"]
        if any(width_of(c.origin_lat - a) != width_of(c.origin_lat) for a, _, _ in nbs) and not want["band"]:
            want["band"] = r["tile_id"]
        if any(s not in FRAME for _, _, s in nbs) and not want["absent"]:
            want["absent"] = r["tile_id"]
        if len(nbs) == 2 and not want["edge"]:
            want["edge"] = r["tile_id"]
    return picks + [v for v in want.values() if v]


def main():
    store = TileStore("data/corpus_500k")
    tiles = pick_tiles()
    log = []
    tmp = Path(tempfile.mkdtemp(dir=OUT))
    for i, tid in enumerate(tiles):
        rec = {"tile_id": tid}
        c = parse_cog_key(parse(tid)[0])
        W = width_of(c.origin_lat)
        mtr, mcrs, (dy, dx) = metric_transform(c.origin_lat + 0.5, W)
        rec["dy_dx_m"] = [dy, dx]
        for buf, L in [(50, 50), (50, 25), (100, 100)]:
            t0 = time.perf_counter()
            arr, tr, meta = read_buffered(tid, buf)
            rec[f"read_s_buf{buf}_L{L}"] = time.perf_counter() - t0
            if buf == 50 and L == 50:
                rec.update(meta)
                shard = store.read(tid).raw_i16.astype(np.float32)
                ctr = arr[buf:buf + T, buf:buf + T]
                rec["centre_rounded_equal"] = bool((np.round(ctr) == shard).all())
            gm, dt = geomorphons(arr, mtr, mcrs, L, tmp, f"t{i}_L{L}")
            rec[f"wbt_s_L{L}"] = dt
            np.save(OUT / f"t{i}_L{L}_buffered.npy", gm[buf:buf + T, buf:buf + T].astype(np.uint8))
            if L == 50:
                bare, dtb = geomorphons(arr[buf:buf + T, buf:buf + T], mtr, mcrs, L, tmp, f"t{i}_bare")
                rec["wbt_s_bare"] = dtb
                np.save(OUT / f"t{i}_L50_bare.npy", bare.astype(np.uint8))
                geo, _ = geomorphons(arr, tr, CRS.from_epsg(4326), L, tmp, f"t{i}_geo")
                np.save(OUT / f"t{i}_L50_geo.npy", geo[buf:buf + T, buf:buf + T].astype(np.uint8))
                iso = Affine(dy, 0, 0, 0, -dy, 0)
                gi, _ = geomorphons(arr, iso, mcrs, L, tmp, f"t{i}_iso")
                np.save(OUT / f"t{i}_L50_iso.npy", gi[buf:buf + T, buf:buf + T].astype(np.uint8))
        log.append(rec)
        print(json.dumps(rec), flush=True)
    json.dump(log, open(OUT / "records.json", "w"), indent=1)


if __name__ == "__main__":
    main()
