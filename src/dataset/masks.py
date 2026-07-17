"""Resolve, read, and cache the three Copernicus quality masks (EDM/FLM/WBM).

The acquisition story is already settled — see ``docs/research/copernicus-mask-spec.md``
(issue #14). This module does not re-investigate it; it implements it. The four
things that story pinned, and how each lands here:

1. **Resolution** (``mask_key``). The masks live in each tile's *own* prefix in
   the DEM bucket — ``s3://copernicus-dem-30m/{stem}/AUXFILES/{mask_stem}.tif`` —
   so a mask key is derived from the DEM stem by one trailing-token substitution
   plus an ``AUXFILES/`` segment. No manifest, no separate resolver. Two naming
   traps the substitution has to respect are documented on :func:`mask_key`.

2. **Reading** (:class:`MaskReader`). EDM/FLM/WBM are **bit-exactly
   co-registered** with their DEM — byte-identical affine/shape/CRS/bounds/overview
   pyramid, including the poleward column-shrink (spec §6). So a mask window read
   with the *same* :mod:`~dataset.geometry` window origins as the DEM is aligned by
   construction: this module reuses ``grid_windows``' ``(row0, col0)`` origins and
   invents no second indexing scheme. The reader mirrors
   :class:`~dataset.tiling.CogReader` — ``read_overview`` + ``read_windows`` — but
   keeps the raster **uint8**: masks are categorical (an averaged "EDM 6.5" is
   meaningless), and their ``[2,4,8]`` overviews are mode-resampled and
   categorical-safe, so ×8 reads are valid for gating (spec §7).

3. **Caching** (:class:`MaskCache`). The COG path (:mod:`dataset.tiling`) has **no
   local cache** — it streams the 42 MB DEM from ``/vsis3`` on every open, by
   design. Masks are the opposite case: ~0.25 MB for all three, read repeatedly,
   so this is a **separate** I/O path that localizes each mask file whole on first
   read (:data:`config.MASK_CACHE_DIR`) and serves every later window/overview from
   disk. Nothing to share with the COG reads because they cache nothing.

4. **Absence** (:class:`MaskNotFound`). A missing mask and a *fetch failure* must
   never be conflated: a failure silently read as "no edited pixels" would corrupt
   the prevalence table and every gate tuned against it. The invariant is strong —
   216/216 sampled tiles carry all three masks — but it is sampled, not proven over
   all 26,450 prefixes, so this module **fails loudly** either way. There is no
   code path that returns data on failure: a genuinely-absent object raises
   :class:`MaskNotFound`; any other read error propagates its native exception. An
   all-zeros array is only ever a *successfully read* all-zeros array.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import Final, Literal, Protocol

import numpy as np

from . import config
from .geometry import CogRef

#: A quality-mask layer name — the closed vocabulary of :data:`MASK_LAYERS`.
Layer = Literal["EDM", "FLM", "WBM"]

# --- Layers and code vocabularies (spec §2-4) ---------------------------------

#: The three 8-bit thematic masks in scope. ``HEM`` (float32, ~47 MB, larger than
#: the DEM) and the KML layers are deliberately excluded (spec §1.1).
MASK_LAYERS: Final = config.MASK_LAYERS

# The complete, normative pixel vocabularies from Product Handbook i5.0 Tables
# 6/7/8 — the value sets a valid read may contain. Binning the prevalence table on
# these is issue #19's job; they live here because this is where a mask read is
# given meaning, and two traps ride on them:
#   * FLM is NON-CONTIGUOUS — 0-9 then 100-102 — so never bin it on range(0, 10).
#   * "Void" is in-band code 0 for EDM/FLM (nodata=None on the file), but WBM has
#     NO void code: its 0 is the substantive class "No water". Absence of a mask is
#     never code 0 — it is a raised exception (see :class:`MaskNotFound`).
EDM_CODES: Final = tuple(range(0, 14))  # 0..13
FLM_CODES: Final = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 100, 101, 102)
WBM_CODES: Final = (0, 1, 2, 3)


# --- Absence (spec §1.1 fog note (4)) -----------------------------------------


class MaskNotFound(FileNotFoundError):
    """A mask object is genuinely absent from the bucket — a loud, distinct signal.

    Raised only when the store reports the key does not exist. It is *not* raised
    for transient fetch failures (network, throttling, permissions): those keep
    their native exception type and propagate unchanged. Both are loud; neither
    ever degrades into an empty or all-zeros mask. Per the spec's sampled-not-proven
    invariant, an absent mask is an anomaly to investigate, never "this cell has no
    edited pixels".
    """

    def __init__(self, bucket: str, key: str) -> None:
        self.bucket = bucket
        self.key = key
        super().__init__(f"mask object not found: s3://{bucket}/{key}")


# --- Key resolution (spec §1.1) -----------------------------------------------

_DEM_SUFFIX: Final = "_DEM"


def _mask_stem(dem_stem: str, layer: Layer) -> str:
    """Mask filename stem for a DEM stem and layer, respecting both naming traps.

    Trap 1 (the parent dir keeps ``_DEM``) is handled in :func:`mask_key`, which
    uses ``dem_stem`` verbatim for the prefix. Trap 2, handled here: the mask
    filename swaps only the *trailing* ``_DEM`` token for ``_EDM``/``_FLM``/``_WBM``
    and **retains ``COG_10``** in the stem (unlike the sidecar XML, which drops it).
    """
    if layer not in MASK_LAYERS:
        raise ValueError(f"unknown mask layer {layer!r}; expected one of {MASK_LAYERS}")
    if not dem_stem.endswith(_DEM_SUFFIX):
        raise ValueError(f"not a Copernicus DEM stem (no trailing _DEM): {dem_stem!r}")
    return dem_stem[: -len(_DEM_SUFFIX)] + f"_{layer}"


def mask_key(dem_stem: str, layer: Layer) -> str:
    """Bucket-relative key of a tile's mask, derived from its DEM stem.

    ``Copernicus_DSM_COG_10_N45_00_E007_00_DEM``, ``"EDM"`` ->
    ``Copernicus_DSM_COG_10_N45_00_E007_00_DEM/AUXFILES/Copernicus_DSM_COG_10_N45_00_E007_00_EDM.tif``.

    The parent directory keeps the ``_DEM`` suffix (trap 1); only the filename's
    trailing token changes (trap 2, :func:`_mask_stem`).
    """
    return f"{dem_stem}/AUXFILES/{_mask_stem(dem_stem, layer)}.tif"


# --- Local cache (spec §1.1: masks are small, read repeatedly) ----------------

#: Fetch a bucket object's bytes by key, or raise :class:`MaskNotFound` if it is
#: genuinely absent. Injectable so the cache is exercised with zero network.
Fetch = Callable[[str], bytes]


def s3_fetch(bucket: str = config.COPERNICUS_GLO30_BUCKET) -> Fetch:
    """Anonymous S3 object fetch, translating a missing key to :class:`MaskNotFound`.

    A "no such key" response becomes :class:`MaskNotFound` (genuine absence); every
    other :class:`~botocore.exceptions.ClientError` — throttling, transient network,
    a permissions surprise — **propagates unchanged**, so a fetch failure can never
    be mistaken for an absent mask (and thus never for "no edits").
    """
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    from botocore.exceptions import ClientError

    client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    missing = {"NoSuchKey", "NoSuchBucket", "404", "NotFound"}

    def fetch(key: str) -> bytes:
        try:
            resp = client.get_object(Bucket=bucket, Key=key)
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code in missing:
                raise MaskNotFound(bucket, key) from exc
            raise  # transient / auth failure — loud, never swallowed
        return resp["Body"].read()

    return fetch


class MaskCache:
    """A local file cache that localizes each mask object whole on first read.

    Keyed by the bucket-relative mask key, mirrored under ``cache_dir`` so the
    on-disk layout matches the bucket. The whole (tiny) mask file is fetched once
    and written atomically; every later window/overview read opens the local file.
    ``fetch`` is injectable (defaults to anonymous S3), which is what lets the whole
    read path run with zero network in tests.
    """

    def __init__(self, cache_dir: Path | str, fetch: Fetch) -> None:
        self.cache_dir = Path(cache_dir)
        self._fetch = fetch

    def path(self, dem_stem: str, layer: Layer) -> Path:
        """Local path a mask caches to (whether or not it is present yet)."""
        return self.cache_dir / mask_key(dem_stem, layer)

    def ensure(self, dem_stem: str, layer: Layer) -> Path:
        """Local path to the cached mask, fetching it once if not already present.

        Raises :class:`MaskNotFound` if the object is genuinely absent; any other
        fetch error propagates. On success the file is written via a temp-file
        rename so a crashed or concurrent fetch never leaves a half-written mask
        that a later read would trust.
        """
        dest = self.path(dem_stem, layer)
        if dest.exists():
            return dest
        data = self._fetch(mask_key(dem_stem, layer))
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f"{dest.name}.{os.getpid()}.tmp")
        try:
            tmp.write_bytes(data)
            os.replace(tmp, dest)  # atomic within the cache dir
        finally:
            if tmp.exists():
                tmp.unlink()
        return dest


# --- Reader interface (mirrors dataset.tiling.CogReader) -----------------------


class MaskReader(Protocol):
    """The mask-access surface, parallel to :class:`~dataset.tiling.CogReader`.

    ``layer`` is one of :data:`MASK_LAYERS`; reads return **uint8** categorical
    rasters. Window origins are the DEM's own ``grid_windows`` origins — aligned by
    construction (spec §6). Absence raises :class:`MaskNotFound`.
    """

    def read_overview(self, cog: CogRef, layer: Layer, factor: int) -> np.ndarray: ...

    def read_windows(
        self, cog: CogRef, layer: Layer, origins: list[tuple[int, int]], tile: int
    ) -> list[np.ndarray]: ...


class RasterioMaskReader:
    """Read EDM/FLM/WBM from ``s3://copernicus-dem-30m`` via a local file cache.

    Each mask is localized whole on first touch (:class:`MaskCache`) and every read
    opens the cached file with rasterio. ``fetch`` is injectable for tests; it
    defaults to anonymous S3 (:func:`s3_fetch`) and is only constructed on first
    use, so importing this module never requires boto3.
    """

    def __init__(
        self,
        bucket: str = config.COPERNICUS_GLO30_BUCKET,
        cache_dir: Path | str = config.MASK_CACHE_DIR,
        fetch: Fetch | None = None,
    ) -> None:
        self.bucket = bucket
        self.cache = MaskCache(cache_dir, fetch or s3_fetch(bucket))

    def read_overview(self, cog: CogRef, layer: Layer, factor: int) -> np.ndarray:
        """Read the whole mask decimated to its ×``factor`` overview, as uint8.

        Mirrors :meth:`~dataset.tiling.RasterioCogReader.read_overview`: ``out_shape``
        makes GDAL serve the stored (mode-resampled, categorical-safe) overview
        rather than resampling full-res. The dtype is kept uint8 — averaging a
        categorical raster is meaningless.
        """
        import rasterio

        path = self.cache.ensure(cog.stem, layer)
        with rasterio.open(path) as ds:
            oh = math.ceil(ds.height / factor)
            ow = math.ceil(ds.width / factor)
            return ds.read(1, out_shape=(oh, ow))

    def read_windows(
        self, cog: CogRef, layer: Layer, origins: list[tuple[int, int]], tile: int
    ) -> list[np.ndarray]:
        """Read ``tile``×``tile`` mask windows at ``origins`` (open once, read many).

        ``origins`` are the DEM's ``grid_windows`` ``(row0, col0)`` pairs; the mask
        slice is the same window geometry by co-registration (spec §6).
        """
        import rasterio
        from rasterio.windows import Window

        path = self.cache.ensure(cog.stem, layer)
        out: list[np.ndarray] = []
        with rasterio.open(path) as ds:
            for row0, col0 in origins:
                win = Window(col_off=col0, row_off=row0, width=tile, height=tile)
                out.append(ds.read(1, window=win))
        return out


class SyntheticMaskReader:
    """In-memory reader over fabricated uint8 masks — the zero-network test seam.

    The mask analogue of :class:`~dataset.tiling.SyntheticCogReader`: overviews are
    plain strided decimations (enough to exercise the flow deterministically; not
    the mode resampling GDAL applies to the real overviews), windows are array
    slices at the given origins. An absent stem or layer raises
    :class:`MaskNotFound`, so the absence contract holds here too.
    """

    def __init__(self, masks: dict[str, dict[str, np.ndarray]]) -> None:
        # {dem_stem: {"EDM": arr, "FLM": arr, "WBM": arr}}, coerced to uint8.
        self._masks = {
            stem: {layer: np.asarray(a, dtype=np.uint8) for layer, a in layers.items()}
            for stem, layers in masks.items()
        }

    def _get(self, stem: str, layer: Layer) -> np.ndarray:
        if layer not in MASK_LAYERS:
            raise ValueError(f"unknown mask layer {layer!r}; expected one of {MASK_LAYERS}")
        try:
            return self._masks[stem][layer]
        except KeyError as exc:
            raise MaskNotFound(config.COPERNICUS_GLO30_BUCKET, mask_key(stem, layer)) from exc

    def read_overview(self, cog: CogRef, layer: Layer, factor: int) -> np.ndarray:
        a = self._get(cog.stem, layer)
        return a[::factor, ::factor].astype(np.uint8)

    def read_windows(
        self, cog: CogRef, layer: Layer, origins: list[tuple[int, int]], tile: int
    ) -> list[np.ndarray]:
        a = self._get(cog.stem, layer)
        return [a[r : r + tile, c : c + tile].astype(np.uint8) for r, c in origins]
