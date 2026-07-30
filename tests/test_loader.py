"""Read path: Normalizer strategies (numpy) and the torch TileDataset."""

from __future__ import annotations

import numpy as np
import pytest

from dataset import loader, manifest, normalize, shards


# --- Normalizer strategies (no torch needed) ----------------------------------


def test_per_tile_normalizer_standardizes():
    x = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    out = loader.PerTileNormalizer()(x, mean=20.0, std=np.std(x))
    assert out.mean() == pytest.approx(0.0, abs=1e-6)
    assert out.std() == pytest.approx(1.0, abs=1e-6)


def test_global_normalizer_uses_corpus_stats():
    x = np.array([100.0, 200.0], dtype=np.float32)
    out = loader.GlobalNormalizer(global_mean=150.0, global_std=50.0)(
        x, mean=999.0, std=999.0  # per-tile stats ignored in global mode
    )
    assert out.tolist() == pytest.approx([-1.0, 1.0])


def test_make_normalizer_selects_and_validates():
    assert isinstance(loader.make_normalizer("per_tile"), loader.PerTileNormalizer)
    g = loader.make_normalizer("global", {"global_mean": 0.0, "global_std": 1.0})
    assert isinstance(g, loader.GlobalNormalizer)
    with pytest.raises(ValueError):
        loader.make_normalizer("bogus")
    with pytest.raises(ValueError):
        loader.make_normalizer("global")  # missing meta


# --- TileDataset (torch) ------------------------------------------------------


@pytest.fixture
def corpus(tmp_path):
    """A tiny materialized corpus (shards + manifest + meta), built straight from the
    ADR-0002 storage primitives — the loader read path needs a corpus on disk, not the
    retired TRI selection pipeline that used to produce one."""
    rng = np.random.default_rng(0)
    specs = [
        ("Copernicus_DSM_COG_10_N45_00_E006_00_DEM", 1500.0, 6.5, 45.5, "0-50", "+000_+040"),
        ("Copernicus_DSM_COG_10_S01_00_E036_00_DEM", 800.0, 36.5, -0.5, "0-50", "+030_+000"),
    ]
    rows: list[manifest.TileRecord] = []
    gsum = gsumsq = gcount = 0.0
    with shards.ShardWriter(tmp_path / "shards") as writer:
        for stem, base, lon, lat, band, gc in specs:
            for j in range(6):
                win = (base + rng.normal(0, 200, size=(224, 224))).astype(np.float32)
                tile, reason = normalize.prepare_tile(win)
                assert tile is not None, reason
                ref = writer.add(tile.raw_i16)
                vals = tile.raw_i16[normalize.land_mask(tile.raw_i16)].astype(np.float64)
                gsum += float(vals.sum())
                gsumsq += float(np.square(vals).sum())
                gcount += vals.size
                r0 = j * 224
                rows.append(
                    manifest.TileRecord(
                        tile_id=manifest.make_tile_id(stem, r0, 0),
                        shard_id=ref.shard_id,
                        row_index=ref.row_index,
                        source_cog=f"{stem}/{stem}.tif",
                        win_row0=r0,
                        win_col0=0,
                        lon=lon,
                        lat=lat,
                        lat_band=band,
                        geocell_id=gc,
                        mean=tile.mean,
                        std=tile.std,
                        land_frac=tile.land_frac,
                        stratum_id=j % 20,
                        provenance="base",
                        slope_mean=0.1,
                        local_relief=30.0,
                        texture=0.5,
                        elevation_relief_ratio=0.4,
                        mean_elevation=float(base),
                        geomorphon_entropy=1.8,
                        edm_edited_frac=0.0,
                        flm_fill_frac=0.0,
                        wbm_water_frac=0.0,
                        water_affected=False,
                        coastline=False,
                        content_hash=ref.content_hash,
                    )
                )
    gmean = gsum / gcount
    gstd = float(np.sqrt(max(gsumsq / gcount - gmean * gmean, 0.0)))
    manifest.write_manifest(rows, tmp_path / "manifest.parquet")
    manifest.write_corpus_meta(
        tmp_path / "corpus_meta.json",
        rows=rows,
        global_mean=gmean,
        global_std=gstd,
        snapshot_date="test",
        build_timestamp="2026-07-14T00:00:00Z",
        code_version="test",
        selection_list_md5="test-selection-md5",
        catalog_version="test-catalog",
        centroids_md5="test-centroids-md5",
        scaler_md5="test-scaler-md5",
    )
    return tmp_path


def test_dataset_len_matches_manifest(corpus):
    import pyarrow.parquet as pq

    ds = loader.TileDataset(corpus)
    assert len(ds) == pq.read_table(corpus / "manifest.parquet").num_rows
    assert len(ds) > 0


def test_sample_schema(corpus):
    import torch

    ds = loader.TileDataset(corpus)
    s = ds[0]
    assert set(s) == {"image", "land_mask", "mean", "std", "tile_id", "stratum_id"}
    assert s["image"].shape == (1, 224, 224)  # C=1 channel-ready
    assert s["image"].dtype == torch.float32
    assert s["land_mask"].shape == (224, 224)
    assert s["land_mask"].dtype == torch.bool
    assert isinstance(s["tile_id"], str)
    assert isinstance(s["stratum_id"], int)


def test_per_tile_normalization_zero_centers_land(corpus):
    ds = loader.TileDataset(corpus, normalization="per_tile")
    s = ds[0]
    land = s["image"][0][s["land_mask"]]
    assert float(land.mean()) == pytest.approx(0.0, abs=1e-3)
    assert float(land.std()) == pytest.approx(1.0, abs=1e-2)


def test_global_and_per_tile_differ(corpus):
    per = loader.TileDataset(corpus, normalization="per_tile")[0]["image"]
    glob = loader.TileDataset(corpus, normalization="global")[0]["image"]
    assert not np.allclose(per.numpy(), glob.numpy())


def test_memmaps_are_lazy(corpus):
    ds = loader.TileDataset(corpus)
    assert ds._memmaps == {}  # nothing opened at construction
    _ = ds[0]
    assert len(ds._memmaps) >= 1  # opened on first access


def test_dataloader_batches_cleanly(corpus):
    from torch.utils.data import DataLoader

    ds = loader.TileDataset(corpus)
    batch = next(iter(DataLoader(ds, batch_size=4, shuffle=True)))
    assert batch["image"].shape[1:] == (1, 224, 224)
    assert batch["image"].shape[0] <= 4
    assert isinstance(batch["tile_id"], list)
    assert batch["mean"].shape[0] == batch["image"].shape[0]


def test_build_dataloader_sets_adr_perf_flags(corpus):
    ds = loader.TileDataset(corpus)
    dl = loader.build_dataloader(ds, batch_size=4, num_workers=0)
    assert dl.pin_memory is True
    assert dl.batch_size == 4
    batch = next(iter(dl))
    assert batch["image"].shape[1:] == (1, 224, 224)


def test_the_dataset_survives_a_pickle_round_trip(corpus):
    """What a DataLoader worker needs under the `spawn` start method (macOS's default).

    The Dataset class is built inside a function so that importing this module never imports
    torch (ADR-0003), and a class defined inside a function is not picklable by reference
    unless the module can resolve its name. Fork copies the class object and never notices;
    spawn dies with "Can't pickle local object" — at `num_workers > 0` only, which is why this
    survived until #93 ran the loop against the real corpus.
    """
    import pickle

    ds = loader.TileDataset(corpus)
    restored = pickle.loads(pickle.dumps(ds))
    assert len(restored) == len(ds)
    assert restored[0]["tile_id"] == ds[0]["tile_id"]


def test_it_reads_through_worker_processes(corpus):
    ds = loader.TileDataset(corpus)
    dl = loader.build_dataloader(ds, batch_size=2, num_workers=2, shuffle=False, pin_memory=False)
    tiles = [t for batch in dl for t in batch["tile_id"]]
    assert sorted(tiles) == sorted(ds._tile_id)
