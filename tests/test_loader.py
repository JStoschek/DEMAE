"""Read path: Normalizer strategies (numpy) and the torch TileDataset."""

from __future__ import annotations

import numpy as np
import pytest

from dataset import loader, pipeline
from dataset.tiling import SyntheticCogReader


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
    rng = np.random.default_rng(0)
    tiles = {
        "Copernicus_DSM_COG_10_N45_00_E006_00_DEM": (
            1500 + rng.normal(0, 200, size=(672, 672))
        ).astype(np.float32),
        "Copernicus_DSM_COG_10_S01_00_E036_00_DEM": (
            800 + rng.normal(0, 120, size=(672, 672))
        ).astype(np.float32),
    }
    pipeline.run_pipeline(
        SyntheticCogReader(tiles),
        tmp_path,
        per_stratum_quota=6,
        snapshot_date="test",
        code_version="test",
        build_timestamp="2026-07-14T00:00:00Z",
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
    assert set(s) == {"image", "land_mask", "mean", "std", "tile_id", "stratum"}
    assert s["image"].shape == (1, 224, 224)  # C=1 channel-ready
    assert s["image"].dtype == torch.float32
    assert s["land_mask"].shape == (224, 224)
    assert s["land_mask"].dtype == torch.bool
    assert isinstance(s["tile_id"], str)
    assert isinstance(s["stratum"], int)


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
