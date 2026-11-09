"""Geomorphon labels under ADR-0014: one WBT call, a buffered re-read, a crop, one ignore rule (#201).

Promoted from the code #187 and #188 measured with (``docs/research/p187/geomorphon_pass.py``,
``docs/research/p188/p188_sweep.py``), because ADR-0014's figures came from it. What is lifted is
the tile-centre WGS84 spacing (:func:`spacing`), the pixel-is-point mosaic on the tile COG's own
grid (:func:`read_buffered`), the GTiff-with-a-projected-header WBT call (:func:`call_wbt`), and
"read once at the widest L, crop per variant" (:meth:`Window.crop`). What is fixed is everything
ADR-0014 settled after those scripts were written:

* **Absent data is not 0 m.** p188 zero-filled any pixel no COG covered. Here it is masked
  (:attr:`Window.absent`), written to WBT as nodata, and every label pixel whose search window
  reaches it is 255 (:func:`label_tile`). Only a COG on the committed verified-ocean list reads as
  0 m, and a present pixel bordering it that is not 0 m refuses the read: the list was wrong.
* **The antimeridian wraps** for the COGs on the committed wrap list (S85/S86 E179). p188 had no
  wrap, so the east buffer of those tiles was zero-filled.
* **The column buffer is anisotropic** (p187's was square): L rows and ⌈L·dy/dx⌉ columns.
* **Nearest is per pixel across a column-width band.** p187/p188 mosaicked with
  ``rasterio.merge``, which resamples a neighbour in a coarser band by rounding whole windows: on
  a fixture crossing 50°N it read ~half of the coarser band's pixels one source column off and
  left the window's last column at nodata. Same-band pixels are identical either way, and so is
  every tile centre (#187's check); only band-crossing buffers differ.

## Why this module is mostly refusals

Every failure ADR-0014 names returns a plausible uint8 raster and raises nothing. A bare 224 px
tile, an EPSG:4326 header (#187: WBT reads degrees as metres and loses flat, shoulder and
footslope), a WBT default standing in for a registered parameter, a window off the shard grid
by a pixel, a different WBT build — each one is a wrong label set that trains normally. So:

* :class:`SpacingHeader` cannot hold a geographic CRS, and :func:`call_wbt` checks again before it
  writes a byte, so an EPSG:4326 window cannot reach the binary.
* :func:`call_wbt` passes all four of :class:`WbtParams` on every call and refuses a binary whose
  sha256 is not in the config's pin list. The binary's *path* is environment (outside every hash);
  its *sha256* is recipe.
* :func:`read_buffered` raises on any read it cannot complete, and :func:`label_tile` raises when
  the window's centre is not the shard tile. **There is no ring-mask fallback**: a failed buffered
  read stops the pass rather than being scored on the centre 124².

The pass that drives this over the committed tile lists — resume, progress, fingerprints, the
census, the class weights and the fixed-sample rebuild — is :mod:`benchmark.geomorphon_pass`.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Final, Protocol

import numpy as np

from dataset import config as dataset_config
from dataset import normalize
from dataset.geometry import COLS_PER_DEGREE, ROWS_PER_DEGREE, format_cog_stem, lat_band, parse_cog_key

#: The benchmark tile edge, in pixels. The label is the centre ``TILE x TILE`` of every window.
TILE: Final = 224

#: What :func:`call_wbt` writes for absent data, and declares as the GTiff's nodata (p188's value).
WBT_NODATA: Final = -32768.0

#: The ten classes' WBT ``--forms`` codes. 0 in :func:`call_wbt`'s output is WBT's nodata.
WBT_FORMS: Final = tuple(range(1, 11))

#: The projected CRS the header claims. Only its units matter to WBT — the cell size *is* the
#: ground spacing — and EPSG:3857 is what #187 and #188 measured with.
HEADER_CRS: Final = "EPSG:3857"

# --- The three rules the config hash covers (ADR-0014 D8) ----------------------------------------
#
# Written out as the sentences the code implements, so that moving the code without moving the
# sentence is visible in review, and moving the sentence moves the geomorphon config hash (and so
# the labels directory and every committed fingerprint). Change one only with the code beside it.

BUFFER_RULE: Final = (
    "L rows and ceil(L*dy/dx) columns of buffer per side, mosaicked on the tile COG's own grid "
    "(pixel-is-point, nearest across a column-width band); a tile is read once at the widest L "
    "its variants need and each variant crops its input from that read; the centre 224x224 is "
    "cropped after WBT runs; no ring mask"
)
SPACING_RULE: Final = (
    "unresampled elevations under a projected EPSG:3857 header whose cell size is the WGS84 "
    "(pyproj.Geod) ground spacing (dx, dy) of one pixel at the tile's centre; never EPSG:4326"
)
IGNORE_RULE: Final = (
    "255 where the shard tile is exact-zero ocean (normalize.land_mask), or where the "
    "(2L+1)-row x (2*ceil(L*dy/dx)+1)-column rectangle centred on the pixel reaches data absent "
    "from the GLO-30 release; absent COGs on verified_ocean read as 0 m; COGs on "
    "antimeridian_wrap read across 180 degrees; every other absent COG must be on absent_land"
)


# --- The four parameters --------------------------------------------------------------------------


@dataclass(frozen=True)
class WbtParams:
    """WBT ``Geomorphons``' four parameters — all four, every call, never a WBT default (D2)."""

    search: int
    threshold: float
    skip: int
    fdist: int

    def __post_init__(self) -> None:
        if self.search <= 0 or self.threshold <= 0 or self.skip < 0 or self.fdist < 0:
            raise ValueError(f"{self} is not a geomorphon parameter set ADR-0014 can mean")

    def args(self) -> list[str]:
        """The four flags, each explicit. :func:`call_wbt` passes exactly these."""
        return [
            f"--search={self.search}",
            f"--threshold={self.threshold}",
            f"--skip={self.skip}",
            f"--fdist={self.fdist}",
        ]


@dataclass(frozen=True)
class Variant:
    """One label set: a name, its parameters, and the splits it labels (D7)."""

    name: str
    params: WbtParams
    splits: tuple[str, ...]


#: Where each split's tiles come from in Task 1's committed lists.
SPLITS: Final = ("train", "val", "test")


def variants(g) -> tuple[Variant, ...]:
    """The primary set (every split) and D7's four sensitivity variants (test only), in that order.

    Each variant moves one parameter and holds the rest at the primary. ``g`` is
    :class:`benchmark.landform_config.Geomorphon`.
    """
    primary = WbtParams(search=g.search, threshold=g.threshold, skip=g.skip, fdist=g.fdist)
    out = [Variant("primary", primary, SPLITS)]
    for search in g.variant_search:
        out.append(Variant(f"L{search}", _moved(primary, search=search), ("test",)))
    for threshold in g.variant_threshold:
        out.append(Variant(f"flat{threshold:g}", _moved(primary, threshold=threshold), ("test",)))
    return tuple(out)


def _moved(p: WbtParams, **change) -> WbtParams:
    return WbtParams(**{**p.__dict__, **change})


# --- The spacing header ---------------------------------------------------------------------------


def _is_geographic(crs: str) -> bool:
    from pyproj import CRS

    return CRS.from_user_input(crs).is_geographic


def refuse_geographic(crs: str) -> None:
    """D3: WBT is never handed a geographic CRS. On EPSG:4326 it reads degrees as metres (#187)."""
    if _is_geographic(crs):
        raise ValueError(
            f"a window under {crs!r} cannot reach WBT: on a geographic CRS it reads the cell "
            f"size in degrees as metres, every line of sight clears the flatness threshold, and "
            f"flat, shoulder and footslope never appear (#187, ADR-0014 D3). Write the window "
            f"under a projected header carrying its WGS84 ground spacing"
        )


@dataclass(frozen=True)
class SpacingHeader:
    """What WBT is told about the grid: a projected CRS and the tile-centre ground spacing (D3)."""

    #: Ground metres per pixel along a row (east-west) and a column (north-south).
    dx: float
    dy: float
    crs: str = HEADER_CRS

    def __post_init__(self) -> None:
        refuse_geographic(self.crs)
        if not (self.dx > 0 and self.dy > 0):
            raise ValueError(f"spacing (dx {self.dx}, dy {self.dy}) must be positive metres")

    def column_buffer(self, search: int) -> int:
        """D4: WBT reaches ``L*dy`` metres, which along a row is ⌈L·dy/dx⌉ cells."""
        return math.ceil(search * self.dy / self.dx)


def width_of(origin_lat: int) -> int:
    """Columns per degree of the COG whose south edge is ``origin_lat`` (p188's ``width_of``)."""
    return COLS_PER_DEGREE[lat_band(origin_lat + 0.5)]


def parse_tile_id(tile_id: str) -> tuple[str, int, int]:
    """``(stem, row0, col0)`` of a corpus tile id (``manifest.make_tile_id``'s inverse)."""
    stem, r0, c0 = tile_id.rsplit("_", 2)
    return stem, int(r0), int(c0)


@lru_cache(maxsize=1)
def _geod():
    from pyproj import Geod

    return Geod(ellps="WGS84")


def spacing(tile_id: str) -> SpacingHeader:
    """WGS84 ground spacing of one pixel at the tile centre — p188's ``spacing``, verbatim math.

    Not :func:`dataset.geometry.pixel_spacing_m`, which is spherical and right for a ranking;
    ADR-0014 D3 names the ellipsoid.
    """
    stem, r0, c0 = parse_tile_id(tile_id)
    cog = parse_cog_key(stem)
    w = width_of(cog.origin_lat)
    lat = cog.origin_lat + 1 - (r0 + TILE / 2) / ROWS_PER_DEGREE
    lon = cog.origin_lon + (c0 + TILE / 2) / w
    geod = _geod()
    _, _, dy = geod.inv(lon, lat - 0.5 / ROWS_PER_DEGREE, lon, lat + 0.5 / ROWS_PER_DEGREE)
    _, _, dx = geod.inv(lon - 0.5 / w, lat, lon + 0.5 / w, lat)
    return SpacingHeader(dx=float(dx), dy=float(dy))


# --- The window -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Window:
    """A buffered elevation window with its spacing header — what the one WBT call takes.

    ``elevation`` is float32 metres with :data:`WBT_NODATA` wherever ``absent`` is true. The
    tile is ``elevation[rows:rows+224, cols:cols+224]``.
    """

    tile_id: str
    elevation: np.ndarray
    absent: np.ndarray
    header: SpacingHeader
    rows: int
    cols: int

    def __post_init__(self) -> None:
        if not isinstance(self.header, SpacingHeader):
            raise TypeError(f"a window's header is a SpacingHeader, not {type(self.header)}")
        want = (TILE + 2 * self.rows, TILE + 2 * self.cols)
        if self.elevation.shape != want or self.absent.shape != want:
            raise ValueError(
                f"{self.tile_id}: a window buffered by {self.rows} rows and {self.cols} columns "
                f"is {want}, not {self.elevation.shape} (absent mask {self.absent.shape})"
            )

    @property
    def centre(self) -> tuple[slice, slice]:
        return slice(self.rows, self.rows + TILE), slice(self.cols, self.cols + TILE)

    def buffer_for(self, search: int) -> tuple[int, int]:
        """D4's ``(rows, cols)`` buffer for search radius ``search``."""
        return search, self.header.column_buffer(search)

    def crop(self, search: int) -> Window:
        """The input a variant with search radius ``search`` would have read — D7's crop.

        Valid because ``fdist = 0`` makes no label depend on the window's extent beyond L.
        """
        rows, cols = self.buffer_for(search)
        if rows > self.rows or cols > self.cols:
            raise ValueError(
                f"{self.tile_id} was read at a {self.rows}x{self.cols} buffer and L = {search} "
                f"needs {rows}x{cols}: read the tile at its widest variant's L, once (D4)"
            )
        dr, dc = self.rows - rows, self.cols - cols
        sl = (slice(dr, self.elevation.shape[0] - dr), slice(dc, self.elevation.shape[1] - dc))
        return Window(
            tile_id=self.tile_id,
            elevation=self.elevation[sl],
            absent=self.absent[sl],
            header=self.header,
            rows=rows,
            cols=cols,
        )


# --- The one narrow WBT call ----------------------------------------------------------------------


@lru_cache(maxsize=8)
def _sha256(path: str, size: int, mtime_ns: int, inode: int) -> str:
    """The binary's sha256, cached on its stat so 47k calls hash a 20 MB file once."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binary_sha256(binary: Path | str) -> str:
    """The sha256 of the file at ``binary``, the thing ADR-0014 D1 pins."""
    st = os.stat(binary)
    return _sha256(str(Path(binary).resolve()), st.st_size, st.st_mtime_ns, st.st_ino)


class WbtUnavailable(FileNotFoundError):
    """WhiteboxTools is not on this box: a declared external (ADR-0014 D1), not a broken one.

    Distinct from an unpinned binary (a :class:`ValueError`), which is a different build and
    never skipped. ``freeze`` catches only this, to run the stages that read no label (#202).
    """


def check_pinned(binary: Path | str | None, pins: Iterable[str]) -> str:
    """Refuse a WBT binary that is missing or whose sha256 is not pinned; return the sha256."""
    if binary is None:
        raise WbtUnavailable(
            "paths.wbt_binary is not set. WhiteboxTools v2.4.0 is a declared external, not a "
            "Python dependency (ADR-0014 D1): fetch a build whose sha256 is in "
            "geomorphon.wbt_sha256 and point paths.wbt_binary at it (#202)"
        )
    if not Path(binary).is_file():
        raise WbtUnavailable(f"paths.wbt_binary {binary} is not a file (#202)")
    sha = binary_sha256(binary)
    pins = tuple(pins)
    if sha not in pins:
        raise ValueError(
            f"WBT binary {binary} has sha256 {sha}, which is not in geomorphon.wbt_sha256 "
            f"{list(pins)}. A different build can produce a different label set silently "
            f"(ADR-0014 D1); pin this one only after its labels pass the byte-identical "
            f"fixed-sample check against the committed fingerprints"
        )
    return sha


def call_wbt(
    window: Window,
    params: WbtParams,
    *,
    binary: Path | str | None,
    pins: Iterable[str],
) -> np.ndarray:
    """**The** WBT call: a buffered window with its header and the four parameters, to uint8.

    Returns WBT's ``--forms`` codes over the whole window: 1..10 for the ten forms, 0 where WBT
    wrote nodata. Cropping, the 0-9 relabel and the ignore rule are :func:`label_tile`'s, so
    this stays the narrowest thing that can touch the binary. In order, before anything runs:
    the header is projected (D3), the binary is pinned (D1); then all four parameters go on
    the command line (D2).
    """
    refuse_geographic(window.header.crs)
    check_pinned(binary, pins)
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import Affine

    h = window.header
    with tempfile.TemporaryDirectory(prefix="demae-wbt-") as tmp:
        src, dst = Path(tmp) / "dem.tif", Path(tmp) / "forms.tif"
        elevation = np.where(window.absent, WBT_NODATA, window.elevation).astype(np.float32)
        with rasterio.open(
            src, "w", driver="GTiff", width=elevation.shape[1], height=elevation.shape[0],
            count=1, dtype="float32", crs=CRS.from_user_input(h.crs),
            transform=Affine(h.dx, 0, 0, 0, -h.dy, 0), nodata=WBT_NODATA,
        ) as d:
            d.write(elevation, 1)
        cmd = [str(binary), "-r=Geomorphons", f"--dem={src}", f"--output={dst}",
               *params.args(), "--forms"]
        done = subprocess.run(cmd, capture_output=True, text=True)
        if done.returncode != 0 or not dst.is_file():
            raise RuntimeError(
                f"WBT Geomorphons failed on {window.tile_id} (exit {done.returncode}): "
                f"{(done.stderr or done.stdout).strip()[-500:]}"
            )
        with rasterio.open(dst) as d:
            out = d.read(1)
            nodata = d.nodata
    return _forms_to_uint8(out, nodata, window.tile_id)


def _forms_to_uint8(out: np.ndarray, nodata: float | None, tile_id: str) -> np.ndarray:
    if nodata is None or np.isnan(nodata):
        valid = ~np.isnan(out)
    else:
        valid = out != nodata
    codes = out[valid]
    if not np.isin(codes, WBT_FORMS).all():
        bad = np.unique(codes[~np.isin(codes, WBT_FORMS)])[:5]
        raise ValueError(f"WBT returned values {bad.tolist()} on {tile_id}, outside --forms' 1..10")
    forms = np.zeros(out.shape, dtype=np.uint8)
    forms[valid] = codes.astype(np.uint8)
    return forms


#: Substitution point A (spec #197): what labels a window. The real one is :func:`call_wbt`
#: with the binary and pins bound; tests swap in a deterministic function of the window and the
#: four parameters.
Labeller = Callable[[Window, WbtParams], np.ndarray]


@dataclass(frozen=True)
class WbtLabeller:
    """:func:`call_wbt` with the binary and its pin list bound. ``build`` is the binary's sha256."""

    binary: str | None
    pins: tuple[str, ...]
    build: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "build", check_pinned(self.binary, self.pins))

    def __call__(self, window: Window, params: WbtParams) -> np.ndarray:
        return call_wbt(window, params, binary=self.binary, pins=self.pins)


# --- One tile's label -----------------------------------------------------------------------------


def reaches_absent(absent: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """``(224, 224)`` bool: the centre pixels whose search rectangle holds an absent pixel.

    The rectangle is (2·rows+1) x (2·cols+1) — D5's "reaches" read conservatively, since WBT's
    diagonal reach (~L/√2) is shorter than its axial one. ``absent`` is the window cropped to
    exactly that buffer, so a summed-area table answers every pixel at once.
    """
    s = np.pad(absent.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    r, c = 2 * rows + 1, 2 * cols + 1
    n = s[r:r + TILE, c:c + TILE] - s[:TILE, c:c + TILE] - s[r:r + TILE, :TILE] + s[:TILE, :TILE]
    return n > 0


def check_centre(window: Window, shard_raw_i16: np.ndarray) -> None:
    """The buffered window's centre, stored the way the corpus stores a tile, is the shard tile.

    #187's ``centre_rounded_equal``, made a refusal: a mosaic off the shard grid by one pixel, or
    a COG revised under the corpus, labels terrain the encoders never see.
    """
    centre = window.elevation[window.centre]
    if window.absent[window.centre].any():
        raise ValueError(f"{window.tile_id}: the tile's own pixels came back absent from the read")
    stored = normalize.to_int16(normalize.plausibility_clip(centre))
    if not np.array_equal(stored, shard_raw_i16):
        n = int((stored != shard_raw_i16).sum())
        raise ValueError(
            f"{window.tile_id}: the buffered window's centre differs from the shard tile at "
            f"{n:,} of {TILE * TILE:,} pixels. The read is off the shard grid or the source COG "
            f"moved under the corpus; labels from it would describe terrain no encoder reads"
        )


def label_tile(
    window: Window,
    params: WbtParams,
    labeller: Labeller,
    *,
    shard_raw_i16: np.ndarray,
    ignore_label: int = 255,
) -> tuple[np.ndarray, str]:
    """One variant's ``(224, 224)`` uint8 label (0-9, 255 = ignore) and its input hash.

    ``window`` is the tile's one read; it is cropped to this variant's buffer, labelled, and its
    centre cropped after the labeller runs (D4). The ignore rule is D5's.
    """
    check_centre(window, shard_raw_i16)
    w = window.crop(params.search)
    forms = np.asarray(labeller(w, params))
    if forms.shape != w.elevation.shape or forms.dtype != np.uint8:
        raise ValueError(
            f"{window.tile_id}: the labeller returned {forms.dtype} {forms.shape} for a "
            f"{w.elevation.shape} window; the WBT call returns uint8 over the whole window"
        )
    centre = forms[w.centre]
    ignore = ~normalize.land_mask(shard_raw_i16) | reaches_absent(w.absent, w.rows, w.cols)
    if ((centre == 0) & ~ignore).any():
        n = int(((centre == 0) & ~ignore).sum())
        raise ValueError(
            f"{window.tile_id}: WBT wrote nodata at {n:,} pixels the ignore rule keeps — a label "
            f"there would be invented, so the pass stops"
        )
    label = np.where(ignore, ignore_label, centre.astype(np.int16) - 1).astype(np.uint8)
    return label, input_hash(w)


def content_hash(label: np.ndarray) -> str:
    """The label raster's fingerprint: 64-bit BLAKE2b of its uint8 bytes, in hex.

    :func:`dataset.shards.content_hash`'s construction (ADR-0002 §3), on uint8 so no byte order
    enters it.
    """
    b = np.ascontiguousarray(label, dtype=np.uint8).tobytes()
    return hashlib.blake2b(b, digest_size=8).hexdigest()


def input_hash(window: Window) -> str:
    """The fingerprint of what WBT read: the window's float32 LE elevations, nodata applied, and
    its spacing. A moved input hash means upstream data moved; a moved label hash alone means
    the pipeline did (D8)."""
    h = hashlib.blake2b(digest_size=8)
    elevation = np.where(window.absent, WBT_NODATA, window.elevation)
    h.update(np.ascontiguousarray(elevation, dtype="<f4").tobytes())
    h.update(json.dumps([window.header.crs, window.header.dx, window.header.dy]).encode())
    return h.hexdigest()


def class_counts(label: np.ndarray, *, classes: int = 10, ignore_label: int = 255) -> list[int]:
    """Per-class pixel counts of one label, then the ignored count — ``classes + 1`` integers."""
    counts = np.bincount(label[label != ignore_label].ravel(), minlength=classes)
    if counts.size != classes:
        raise ValueError(f"a label holds values outside 0..{classes - 1} and {ignore_label}")
    return [*map(int, counts), int((label == ignore_label).sum())]


# --- The buffered re-read (substitution point B) ---------------------------------------------------


class CogSource(Protocol):
    """Where source COGs are read from: which stems the release has, and a rasterio path for each.

    The sibling of :class:`dataset.tiling.CogReader` for reads that one COG's in-bounds square
    windows cannot answer: a mosaic of up to four COGs, across a column-width band, across the
    antimeridian. The live source is the public bucket; a fixture is a directory of small
    GeoTIFFs on the same geographic grid.
    """

    def frame(self) -> frozenset[str]: ...

    def path(self, stem: str) -> str: ...


@dataclass(frozen=True)
class RasterioCogSource:
    """The GLO-30 release as ``data/cog_frame.json`` lists it, read from ``root`` or the bucket.

    ``root=None`` is ``/vsis3/copernicus-dem-30m/{stem}/{stem}.tif`` unsigned, as #187 and #188
    read it; a directory is ``{root}/{stem}/{stem}.tif`` with the same layout.
    """

    frame_path: str
    root: str | None = None

    def frame(self) -> frozenset[str]:
        return _load_frame(self.frame_path)

    def path(self, stem: str) -> str:
        if self.root is None:
            return f"/vsis3/{dataset_config.COPERNICUS_GLO30_BUCKET}/{stem}/{stem}.tif"
        return str(Path(self.root) / stem / f"{stem}.tif")


@lru_cache(maxsize=4)
def _load_frame(frame_path: str) -> frozenset[str]:
    path = Path(frame_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is not on this box. It lists the 26,450 COG stems the GLO-30 release has, "
            f"which is how a buffered read tells an absent neighbour from a present one; set "
            f"paths.cog_frame to it"
        )
    return frozenset(json.loads(path.read_text()))


@dataclass(frozen=True)
class Exceptions:
    """D5's committed lists, by COG stem. Everything absent and on none of them refuses."""

    #: Absent COGs verified to be open ocean, read as 0 m.
    verified_ocean: frozenset[str] = frozenset()
    #: Absent COGs on land, whose reach is ignored (the default D5 rule, listed so it is checked).
    absent_land: frozenset[str] = frozenset()
    #: Tile COGs whose windows wrap across the antimeridian rather than reaching ±180°+.
    antimeridian_wrap: frozenset[str] = frozenset()

    @classmethod
    def of(cls, g) -> Exceptions:
        return cls(frozenset(g.verified_ocean), frozenset(g.absent_land),
                   frozenset(g.antimeridian_wrap))


def _wrap_k(lon_centre: np.ndarray) -> np.ndarray:
    """-1, 0 or +1: how many times a pixel centre's longitude is past ±180°."""
    return np.floor((lon_centre + 180.0) / 360.0).astype(np.int64)


@dataclass(frozen=True)
class _Source:
    """The window pixels one source COG supplies, and which of its pixels each one reads."""

    stem: str
    #: Window row and column indices (an ``np.ix_`` block: a COG covers a rectangle of them).
    rows: np.ndarray
    cols: np.ndarray
    #: Each window pixel's source pixel centre, in degrees (lon per column, lat per row).
    lon: np.ndarray
    lat: np.ndarray


def window_sources(tile_id: str, rows: int, cols: int, ex: Exceptions) -> tuple[
    list[_Source], bool
]:
    """Which COG, and which pixel of it, every pixel of the buffered window reads — nearest.

    The tile COG's grid is pixel-is-point: row ``r`` is centred at ``origin_lat + 1 - r/3600``
    and column ``c`` at ``origin_lon + c/W``. Rows are 1" everywhere, so a window row reads the
    source row at its own latitude. Columns are 1/W of a degree, and W changes across a
    column-width band, so each window column reads the source column whose centre is nearest
    (``round(lon * W_source)``) — D4's "nearest across a column-width band", done per pixel. A
    source centre on a degree line belongs to the cell east of it, and a row centre on one to the
    cell south of it. Columns past ±180° of a tile COG on the wrap list read across the
    antimeridian. Returns the sources and whether any column wrapped.
    """
    stem, r0, c0 = parse_tile_id(tile_id)
    cog = parse_cog_key(stem)
    w = width_of(cog.origin_lat)
    h_px, w_px = TILE + 2 * rows, TILE + 2 * cols
    # Global row centre, in 1" steps north of the equator.
    row_index = (cog.origin_lat + 1) * ROWS_PER_DEGREE - (r0 - rows + np.arange(h_px))
    lon_centre = cog.origin_lon + (c0 - cols + np.arange(w_px)) / w
    k = _wrap_k(lon_centre) if stem in ex.antimeridian_wrap else np.zeros(w_px, dtype=np.int64)
    lon_eff = lon_centre - 360.0 * k
    stem_lat = -((-row_index) // ROWS_PER_DEGREE) - 1  # ceil(row / 3600) - 1
    out = []
    for la in np.unique(stem_lat):
        rr = np.flatnonzero(stem_lat == la)
        ws = width_of(int(la))
        col_index = np.round(lon_eff * ws).astype(np.int64)  # nearest source column, globally
        stem_lon = col_index // ws
        for lo in np.unique(stem_lon):
            cc = np.flatnonzero(stem_lon == lo)
            out.append(_Source(
                stem=format_cog_stem(int(lo), int(la)),
                rows=rr,
                cols=cc,
                lon=col_index[cc] / ws,
                lat=row_index[rr] / ROWS_PER_DEGREE,
            ))
    return out, bool(k.any())


def stems_reached(tile_id: str, rows: int, cols: int, ex: Exceptions) -> set[str]:
    """Every COG stem a window buffered by ``rows`` x ``cols`` reads — the census's question."""
    return {s.stem for s in window_sources(tile_id, rows, cols, ex)[0]}


def _read_source(path: str, src: _Source, tile_id: str) -> tuple[np.ndarray, np.ndarray]:
    """``(values, nodata)`` of ``src``'s pixels from the COG at ``path``, located by its own
    geotransform — so a fixture holding a block of a COG and the full COG read alike. A pixel
    the file does not hold is a failed read, not an absence."""
    import rasterio
    from rasterio.windows import Window as RWindow

    with rasterio.open(path) as ds:
        inv = ~ds.transform
        fc = np.floor(np.array([(inv * (x, src.lat[0]))[0] for x in src.lon])).astype(np.int64)
        fr = np.floor(np.array([(inv * (src.lon[0], y))[1] for y in src.lat])).astype(np.int64)
        if fc.min() < 0 or fr.min() < 0 or fc.max() >= ds.width or fr.max() >= ds.height:
            raise ValueError(
                f"{tile_id}: {src.stem} does not hold every pixel the window reads (file "
                f"{ds.width}x{ds.height}, needs columns {fc.min()}..{fc.max()} and rows "
                f"{fr.min()}..{fr.max()}). A truncated buffer is not labelled: no ring mask"
            )
        c0, r0 = int(fc.min()), int(fr.min())
        block = ds.read(1, window=RWindow(c0, r0, int(fc.max()) - c0 + 1, int(fr.max()) - r0 + 1))
        values = block[np.ix_(fr - r0, fc - c0)].astype(np.float32)
        nodata = (values == ds.nodata) if ds.nodata is not None else np.zeros(values.shape, bool)
    return values, nodata


def read_buffered(
    source: CogSource, tile_id: str, rows: int, cols: int, ex: Exceptions, header: SpacingHeader
) -> Window:
    """The tile re-read with ``rows`` x ``cols`` of buffer per side, mosaicked (D4, D5).

    p188's ``read_buffered``, with its ``rasterio.merge`` replaced by an explicit per-pixel
    nearest read (:func:`window_sources`): ``merge`` resamples a coarser neighbour band by
    rounding whole windows, which leaves the last window column of a band crossing at nodata —
    harmless to p188, which zero-filled nodata, and an invented absence under D5. Then ADR-0014's
    corrections: pixels in a COG the release lacks are *absent* (not 0 m), except inside a
    verified-ocean COG, which reads 0 m; columns past ±180° of a wrap-list tile read the COG
    across the antimeridian. Any read that fails raises — there is no fallback window.
    """
    import rasterio

    stem, _, _ = parse_tile_id(tile_id)
    frame = source.frame()
    if stem not in frame:
        raise ValueError(f"{tile_id}: its own COG {stem} is not in the release frame")
    h_px, w_px = TILE + 2 * rows, TILE + 2 * cols
    out = np.zeros((h_px, w_px), dtype=np.float32)
    absent = np.ones((h_px, w_px), dtype=bool)
    filled = np.zeros((h_px, w_px), dtype=bool)
    sources, _ = window_sources(tile_id, rows, cols, ex)
    env = {**dataset_config.AWS_NO_SIGN_ENV, "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR"}
    with rasterio.Env(**env):
        for src in sources:
            block = np.ix_(src.rows, src.cols)
            if src.stem in frame:
                values, nodata = _read_source(source.path(src.stem), src, tile_id)
                out[block] = values
                absent[block] = nodata
            elif src.stem in ex.verified_ocean:
                filled[block] = True
    if filled.any():
        absent &= ~filled
        _check_ocean_border(tile_id, out, absent, filled, sources, ex)
    out[absent] = WBT_NODATA
    return Window(tile_id=tile_id, elevation=out, absent=absent, header=header, rows=rows,
                  cols=cols)


def _check_ocean_border(tile_id, out, absent, filled, sources, ex: Exceptions) -> None:
    """A verified-ocean COG borders only exact-zero ocean in the data that was read.

    "Each entry checked, never assumed" (D5), checked on every read that reaches one: if a
    present pixel touching the 0 m fill is not 0 m, land runs into the "ocean", the list entry
    is wrong, and the labels near it would be computed against an invented coastline.
    """
    grown = filled.copy()
    grown[1:, :] |= filled[:-1, :]
    grown[:-1, :] |= filled[1:, :]
    grown[:, 1:] |= filled[:, :-1]
    grown[:, :-1] |= filled[:, 1:]
    border = grown & ~filled & ~absent
    if (out[border] != 0.0).any():
        names = sorted({s.stem for s in sources if s.stem in ex.verified_ocean})
        raise ValueError(
            f"{tile_id}: geomorphon.verified_ocean lists {names}, but "
            f"{int((out[border] != 0.0).sum())} present pixels touching it are land. An absence "
            f"is never assumed to be ocean (ADR-0014 D5); move the entry to absent_land"
        )
