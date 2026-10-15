"""#188: geomorphon parameter sweep — flatness x fdist x L, class shares per ruggedness band.

Run from the repo root: PYTHONPATH=src python p188_sweep.py <whitebox_tools> <out_dir>
Reads each tile once at the widest anisotropic buffer (rows L, cols ceil(L*dy/dx), L = 50),
crops for L = 25. Header-only metric CRS, per-tile WGS84 ground spacing at the centre latitude.
Ocean (exact-zero) pixels of the 224 crop are excluded from every share.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Geod
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.transform import Affine

from benchmark import voids
from dataset import config
from dataset.geometry import COLS_PER_DEGREE, format_cog_stem, lat_band, parse_cog_key

WBT, OUT = sys.argv[1], Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
T = 224
FRAME = set(json.load(open("data/cog_frame.json")))
FORMS = ["flat", "peak", "ridge", "shoulder", "spur", "slope", "hollow", "footslope", "valley", "pit"]
FLATS = [1.0, 0.5, 0.2, 0.1]
LS = [25, 50]
GEOD = Geod(ellps="WGS84")


def parse(tile_id):
    stem, r0, c0 = tile_id.rsplit("_", 2)
    return stem, int(r0), int(c0)


def width_of(origin_lat):
    return COLS_PER_DEGREE[lat_band(origin_lat + 0.5)]


def spacing(tile_id):
    """Ground (dy, dx) in metres at the tile centre, WGS84 ellipsoid."""
    stem, r0, c0 = parse(tile_id)
    c = parse_cog_key(stem)
    W = width_of(c.origin_lat)
    lat = c.origin_lat + 1 - (r0 + T / 2) / 3600
    lon = c.origin_lon + (c0 + T / 2) / W
    _, _, dy = GEOD.inv(lon, lat - 0.5 / 3600, lon, lat + 0.5 / 3600)
    _, _, dx = GEOD.inv(lon - 0.5 / W, lat, lon + 0.5 / W, lat)
    return dy, dx


def read_buffered(tile_id, br, bc):
    stem, r0, c0 = parse(tile_id)
    c = parse_cog_key(stem)
    W = width_of(c.origin_lat)
    rx, ry = 1.0 / W, 1.0 / 3600
    west = c.origin_lon - 0.5 * rx + (c0 - bc) * rx
    north = c.origin_lat + 1 + 0.5 * ry - (r0 - br) * ry
    h, w = T + 2 * br, T + 2 * bc
    bounds = (west, north - h * ry, west + w * rx, north)
    stems = set()
    for lat in (bounds[1], bounds[3]):
        for lon in (bounds[0], bounds[2]):
            stems.add(format_cog_stem(math.floor(lon), math.floor(lat - 1e-9)))
    present = [s for s in stems if s in FRAME]
    with rasterio.Env(**config.AWS_NO_SIGN_ENV, GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        dss = [rasterio.open(f"/vsis3/{config.COPERNICUS_GLO30_BUCKET}/{s}/{s}.tif") for s in present]
        try:
            arr, _ = merge(dss, bounds=bounds, res=(rx, ry), nodata=-32767,
                           resampling=Resampling.nearest, dtype="float32")
        finally:
            for d in dss:
                d.close()
    a = arr[0][:h, :w]
    return np.where(a == -32767, 0.0, a).astype(np.float32), len(stems) - len(present)


def run_wbt(arr, dy, dx, L, flat, fdist, tmp: Path):
    src, dst = tmp / "dem.tif", tmp / "gm.tif"
    with rasterio.open(src, "w", driver="GTiff", width=arr.shape[1], height=arr.shape[0], count=1,
                       dtype="float32", crs=CRS.from_epsg(3857),
                       transform=Affine(dx, 0, 0, 0, -dy, 0), nodata=-32768) as d:
        d.write(arr, 1)
    subprocess.run([WBT, "-r=Geomorphons", f"--dem={src}", f"--output={dst}", f"--search={L}",
                    f"--threshold={flat}", "--skip=0", f"--fdist={fdist}", "--forms"],
                   check=True, capture_output=True)
    with rasterio.open(dst) as d:
        return d.read(1)


def pick_tiles(per_band=10, flat_extra=10):
    rows = list(csv.DictReader(open("artifacts/task1/test_tiles.csv")))
    rows.sort(key=lambda r: hashlib.blake2b(r["tile_id"].encode(), digest_size=8).digest())
    cols = voids.read_manifest_columns("data/corpus_500k", ["tile_id", "slope_mean", "land_frac"])
    slope = dict(zip(cols["tile_id"].astype(str), cols["slope_mean"]))
    land = dict(zip(cols["tile_id"].astype(str), cols["land_frac"]))
    bands = voids.Bands.from_manifest("data/corpus_500k", split="test")
    picks, n_band, n_flat = [], defaultdict(int), 0
    for r in rows:
        t = r["tile_id"]
        if land[t] < 0.999:
            continue
        b = bands.of(slope[t])
        if n_band[b] < per_band:
            n_band[b] += 1
            picks.append((t, b, int(r["stratum_id"])))
        elif int(r["stratum_id"]) in (2, 5, 7, 17) and n_flat < flat_extra:
            n_flat += 1
            picks.append((t, b, int(r["stratum_id"])))
    return picks, bands


def main():
    picks, bands = pick_tiles()
    tmp = Path(tempfile.mkdtemp(dir=OUT))
    recs = []
    for i, (tid, band, stratum) in enumerate(picks):
        dy, dx = spacing(tid)
        br, bc = 50, math.ceil(50 * dy / dx)
        arr, absent = read_buffered(tid, br, bc)
        ocean = arr[br:br + T, bc:bc + T] == 0.0
        for L, flat, fd in itertools.product(LS, FLATS, ("0", "half")):
            fdist = 0 if fd == "0" else L // 2
            cr, cc = br - L, bc - math.ceil(L * dy / dx)
            sub = arr[cr:arr.shape[0] - cr, cc:arr.shape[1] - cc]
            gm = run_wbt(sub, dy, dx, L, flat, fdist, tmp)
            ctr = gm[L:L + T, math.ceil(L * dy / dx):math.ceil(L * dy / dx) + T].astype(int)
            v = ctr[~ocean]
            counts = np.bincount(v[(v >= 1) & (v <= 10)] - 1, minlength=10)
            recs.append(dict(tile_id=tid, band=band, stratum=stratum, L=L, flat=flat, fdist=fd,
                             dy=dy, dx=dx, absent=absent, counts=counts.tolist()))
        print(i, tid, band, stratum, flush=True)
    json.dump(dict(edges=list(bands.edges), recs=recs), open(OUT / "sweep.json", "w"))


if __name__ == "__main__":
    main()
