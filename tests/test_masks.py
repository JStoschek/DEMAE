"""Resolve/read/cache the EDM/FLM/WBM masks — fixtures only, zero network.

The live-S3 path is one opt-in smoke test (``DEMAE_S3_TEST=1``), mirroring
``test_tiling``. Everything else runs against fabricated masks: in-memory arrays
for :class:`~dataset.masks.SyntheticMaskReader`, and a real on-disk uint8 GeoTIFF
(built here, no bucket) fed to :class:`~dataset.masks.RasterioMaskReader` through an
injected fetch — the ``synthetic.py`` fabricated-bucket precedent for testing the
real read path without touching S3.
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import pytest

from dataset import config, masks
from dataset.geometry import grid_windows, parse_cog_key

STEM = "Copernicus_DSM_COG_10_N45_00_E007_00_DEM"


# --- Key resolution: the two naming traps (spec §1.1) -------------------------


def test_mask_key_keeps_dem_dir_and_cog10_swaps_only_trailing_token():
    key = masks.mask_key(STEM, "EDM")
    assert key == (
        "Copernicus_DSM_COG_10_N45_00_E007_00_DEM/AUXFILES/"
        "Copernicus_DSM_COG_10_N45_00_E007_00_EDM.tif"
    )
    # Trap 1: parent dir keeps the _DEM suffix.
    assert key.startswith(f"{STEM}/AUXFILES/")
    # Trap 2: the filename retains COG_10 and swaps only the trailing token.
    assert "COG_10" in key.rsplit("/", 1)[1]
    assert key.endswith("_EDM.tif")


def test_mask_key_covers_all_three_layers():
    for layer in masks.MASK_LAYERS:
        assert masks.mask_key(STEM, layer).endswith(f"_{layer}.tif")


def test_mask_key_rejects_unknown_layer_and_non_dem_stem():
    with pytest.raises(ValueError):
        masks.mask_key(STEM, "HEM")  # out of scope, not a thematic layer
    with pytest.raises(ValueError):
        masks.mask_key("Copernicus_DSM_COG_10_N45_00_E007_00", "EDM")  # no _DEM


# --- Vocabularies: the non-contiguity trap (spec §3) --------------------------


def test_flm_vocabulary_is_non_contiguous():
    # range(0, 10) would silently drop the three national-DEM fill sources.
    assert set(masks.FLM_CODES) == set(range(0, 10)) | {100, 101, 102}
    assert masks.EDM_CODES == tuple(range(0, 14))
    assert masks.WBM_CODES == (0, 1, 2, 3)  # no void code — 0 is "No water"


# --- Synthetic reader: alignment, decimation, absence -------------------------


def _synthetic(h: int = 448, w: int = 448) -> masks.SyntheticMaskReader:
    rng = np.random.default_rng(0)
    layers = {
        "EDM": rng.integers(0, 14, size=(h, w), dtype=np.uint8),
        "FLM": rng.integers(0, 10, size=(h, w), dtype=np.uint8),
        "WBM": rng.integers(0, 4, size=(h, w), dtype=np.uint8),
    }
    return masks.SyntheticMaskReader({STEM: layers})


def test_synthetic_windows_match_the_dem_grid_slices():
    reader = _synthetic(448, 448)
    cog = parse_cog_key(STEM)
    edm = reader._masks[STEM]["EDM"]
    origins = grid_windows(width=448, height=448)  # the DEM's own window origins
    wins = reader.read_windows(cog, "EDM", origins, config.TILE_PX)
    assert len(wins) == 4
    for (r0, c0), win in zip(origins, wins):
        assert win.dtype == np.uint8
        assert win.shape == (config.TILE_PX, config.TILE_PX)
        assert np.array_equal(win, edm[r0 : r0 + 224, c0 : c0 + 224])


def test_synthetic_overview_is_decimated_uint8():
    reader = _synthetic(448, 448)
    cog = parse_cog_key(STEM)
    wbm = reader._masks[STEM]["WBM"]
    ov = reader.read_overview(cog, "WBM", 8)
    assert ov.dtype == np.uint8
    assert ov.shape == (56, 56)
    assert np.array_equal(ov, wbm[::8, ::8])


def test_synthetic_absent_stem_raises_mask_not_found():
    reader = _synthetic()
    other = parse_cog_key("Copernicus_DSM_COG_10_N00_00_E000_00_DEM")
    with pytest.raises(masks.MaskNotFound):
        reader.read_windows(other, "EDM", [(0, 0)], config.TILE_PX)


def test_synthetic_unknown_layer_is_a_value_error_not_absence():
    # A bad layer name is a caller bug (ValueError), distinct from a missing object.
    reader = _synthetic()
    with pytest.raises(ValueError):
        reader.read_overview(parse_cog_key(STEM), "HEM", 8)


# --- Cache: download-once, atomic, absence (spec §1.1) ------------------------


class _CountingFetch:
    """A fetch stub over an in-memory {key: bytes} bucket; counts calls."""

    def __init__(self, store: dict[str, bytes]) -> None:
        self.store = store
        self.calls: list[str] = []

    def __call__(self, key: str) -> bytes:
        self.calls.append(key)
        try:
            return self.store[key]
        except KeyError as exc:
            raise masks.MaskNotFound(config.COPERNICUS_GLO30_BUCKET, key) from exc


def test_cache_fetches_once_and_mirrors_the_bucket_layout(tmp_path):
    key = masks.mask_key(STEM, "FLM")
    fetch = _CountingFetch({key: b"raster-bytes"})
    cache = masks.MaskCache(tmp_path, fetch)

    p1 = cache.ensure(STEM, "FLM")
    p2 = cache.ensure(STEM, "FLM")  # second read must hit disk, not the fetch

    assert p1 == p2 == tmp_path / key  # on-disk layout mirrors the bucket key
    assert p1.read_bytes() == b"raster-bytes"
    assert fetch.calls == [key]  # fetched exactly once
    assert not list(tmp_path.rglob("*.tmp"))  # no temp file left behind


def test_cache_absence_raises_and_writes_nothing(tmp_path):
    fetch = _CountingFetch({})  # empty bucket
    cache = masks.MaskCache(tmp_path, fetch)
    with pytest.raises(masks.MaskNotFound):
        cache.ensure(STEM, "EDM")
    # A failed fetch must never leave a file a later read would trust as "no edits".
    assert not cache.path(STEM, "EDM").exists()
    assert not list(tmp_path.rglob("*"))


# --- Rasterio reader against a fabricated on-disk GeoTIFF (no network) --------


def _geotiff_bytes(arr: np.ndarray) -> bytes:
    """A tiled, LZW, uint8 GeoTIFF with the DEM's [2,4,8] mode-resampled overviews."""
    import rasterio
    from affine import Affine
    from rasterio.enums import Resampling

    h, w = arr.shape
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "mask.tif")
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=1,
            dtype="uint8",
            crs="EPSG:4326",
            transform=Affine(1.0 / w, 0, 7, 0, -1.0 / h, 46),
            compress="LZW",
            tiled=True,
            blockxsize=128,
            blockysize=128,
        ) as ds:
            ds.write(arr, 1)
        with rasterio.open(path, "r+") as ds:
            ds.build_overviews([2, 4, 8], Resampling.mode)
        with open(path, "rb") as fh:
            return fh.read()


def _rasterio_reader(tmp_path, arr):
    """A RasterioMaskReader whose fetch serves ``arr`` as GeoTIFF bytes, no S3."""
    store = {masks.mask_key(STEM, "EDM"): _geotiff_bytes(arr)}
    fetch = _CountingFetch(store)
    reader = masks.RasterioMaskReader(cache_dir=tmp_path, fetch=fetch)
    return reader, fetch


def test_rasterio_windows_read_the_expected_uint8_slice(tmp_path):
    arr = np.zeros((448, 448), np.uint8)
    arr[:224] = 1
    arr[224:] = 2
    arr[100:130, 300:330] = 8  # a small distinctive patch
    reader, fetch = _rasterio_reader(tmp_path, arr)
    cog = parse_cog_key(STEM)

    origins = grid_windows(width=448, height=448)
    wins = reader.read_windows(cog, "EDM", origins, config.TILE_PX)
    for (r0, c0), win in zip(origins, wins):
        assert win.dtype == np.uint8
        assert win.shape == (config.TILE_PX, config.TILE_PX)
        assert np.array_equal(win, arr[r0 : r0 + 224, c0 : c0 + 224])
    # Opened once, and every window came from the single cached fetch.
    assert fetch.calls == [masks.mask_key(STEM, "EDM")]


def test_rasterio_overview_is_categorical_safe_uint8(tmp_path):
    arr = np.zeros((448, 448), np.uint8)
    arr[:224] = 1
    arr[224:] = 2
    arr[:60, :60] = 9  # a locally-dominant block so it survives mode resampling
    reader, _ = _rasterio_reader(tmp_path, arr)

    ov = reader.read_overview(parse_cog_key(STEM), "EDM", 8)
    assert ov.dtype == np.uint8
    assert ov.shape == (56, 56)
    # Mode resampling never invents an out-of-vocabulary code (spec §7).
    assert set(np.unique(ov)).issubset(set(arr.ravel().tolist()))


def test_rasterio_reader_caches_across_reads(tmp_path):
    arr = np.zeros((448, 448), np.uint8)
    arr[224:] = 2
    reader, fetch = _rasterio_reader(tmp_path, arr)
    cog = parse_cog_key(STEM)

    reader.read_windows(cog, "EDM", [(0, 0)], config.TILE_PX)
    reader.read_overview(cog, "EDM", 8)
    reader.read_windows(cog, "EDM", [(224, 224)], config.TILE_PX)
    assert fetch.calls == [masks.mask_key(STEM, "EDM")]  # fetched once, reused


def test_rasterio_reader_propagates_absence(tmp_path):
    reader = masks.RasterioMaskReader(cache_dir=tmp_path, fetch=_CountingFetch({}))
    with pytest.raises(masks.MaskNotFound):
        reader.read_windows(parse_cog_key(STEM), "EDM", [(0, 0)], config.TILE_PX)


# --- Picklable: the reader has to cross a process boundary (#38) --------------


def test_default_reader_pickles_and_keeps_its_bucket_and_cache_dir(tmp_path):
    """A default (live-S3) reader must survive a trip to a worker process.

    :func:`~dataset.catalog.scan` scores on a process pool, so every argument it
    hands a worker is pickled. The default ``s3_fetch`` closure is not picklable, so
    it is built on first use rather than in ``__init__`` — which is also what makes
    each spawned worker build its own boto3 client instead of inheriting one.
    """
    reader = masks.RasterioMaskReader(cache_dir=tmp_path)
    revived = pickle.loads(pickle.dumps(reader))
    assert revived.bucket == reader.bucket
    assert revived.cache.cache_dir == tmp_path


def test_pickling_a_used_reader_does_not_carry_its_s3_client(tmp_path):
    """Even after the default fetch has been built, the reader still pickles.

    Constructing it lazily would be no help if the built closure then stuck to the
    instance: a reader that scored one COG in the parent would stop being sendable.
    """
    reader = masks.RasterioMaskReader(cache_dir=tmp_path)
    built = reader.cache._resolve_fetch()  # what a first read would trigger
    assert built is not None
    revived = pickle.loads(pickle.dumps(reader))
    assert revived.cache._default_fetch is None  # rebuilt in the worker, not shipped


def test_pickling_preserves_an_injected_fetch(tmp_path):
    """An injected (picklable) fetch travels — only the lazy default is dropped."""
    key = masks.mask_key(STEM, "FLM")
    reader = masks.RasterioMaskReader(
        cache_dir=tmp_path, fetch=_CountingFetch({key: b"raster-bytes"})
    )
    revived = pickle.loads(pickle.dumps(reader))
    assert revived.cache.ensure(STEM, "FLM").read_bytes() == b"raster-bytes"


# --- Opt-in live smoke (mirrors test_tiling) ----------------------------------


@pytest.mark.skipif(
    os.environ.get("DEMAE_S3_TEST") != "1",
    reason="set DEMAE_S3_TEST=1 to hit the live Copernicus bucket",
)
def test_rasterio_reader_reads_real_masks(tmp_path):
    reader = masks.RasterioMaskReader(cache_dir=tmp_path)
    cog = parse_cog_key(STEM)
    for layer, codes in (
        ("EDM", masks.EDM_CODES),
        ("FLM", masks.FLM_CODES),
        ("WBM", masks.WBM_CODES),
    ):
        win = reader.read_windows(cog, layer, [(0, 0)], config.TILE_PX)[0]
        assert win.shape == (config.TILE_PX, config.TILE_PX)
        assert win.dtype == np.uint8
        assert set(np.unique(win)).issubset(set(codes))
