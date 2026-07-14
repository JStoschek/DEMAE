"""Parquet tile-manifest, corpus_meta.json, and the dataset_version fingerprint."""

from __future__ import annotations

import json

import pyarrow.parquet as pq
import pytest

from dataset import config, manifest


def _row(i, *, cog="Copernicus_DSM_COG_10_N45_00_E006_00_DEM", chash=None):
    return manifest.TileRecord(
        tile_id=manifest.make_tile_id(cog, i * 224, 0),
        shard_id=i // 4,
        row_index=i % 4,
        source_cog=f"{cog}/{cog}.tif",
        win_row0=i * 224,
        win_col0=0,
        lon=6.5,
        lat=45.5,
        lat_band="0-50",
        geocell_id="+000_+040",
        tri=float(i),
        stratum=i % config.N_STRATA,
        mean=1000.0 + i,
        std=50.0,
        land_frac=1.0,
        content_hash=(chash if chash is not None else 111 * (i + 1)),
    )


def test_make_tile_id_is_deterministic():
    assert (
        manifest.make_tile_id("Copernicus_DSM_COG_10_N45_00_E006_00_DEM", 224, 448)
        == "Copernicus_DSM_COG_10_N45_00_E006_00_DEM_224_448"
    )


def test_manifest_round_trips_with_typed_schema(tmp_path):
    rows = [_row(i) for i in range(6)]
    path = tmp_path / "manifest.parquet"
    manifest.write_manifest(rows, path)

    table = pq.read_table(path)
    assert table.num_rows == 6
    # schema types are pinned per ADR-0002 §3
    assert table.schema.field("content_hash").type == "uint64"
    assert table.schema.field("shard_id").type == "int32"
    assert table.schema.field("stratum").type == "int32"
    assert table.schema.field("split").type == "string"

    back = table.to_pylist()
    assert back[0]["tile_id"] == rows[0].tile_id
    assert back[3]["content_hash"] == rows[3].content_hash
    assert back[0]["split"] is None  # Phase-2 fills the geographic hold-out


def test_manifest_has_every_adr_field(tmp_path):
    path = tmp_path / "m.parquet"
    manifest.write_manifest([_row(0)], path)
    names = set(pq.read_table(path).schema.names)
    assert names == {
        "tile_id", "shard_id", "row_index", "source_cog", "win_row0", "win_col0",
        "lon", "lat", "lat_band", "geocell_id", "tri", "stratum", "mean", "std",
        "land_frac", "content_hash", "split",
    }


def test_dataset_version_is_order_independent(tmp_path):
    rows = [_row(i) for i in range(5)]
    params = {"seed": 0, "clip": [-500, 9000]}
    v1 = manifest.dataset_version(rows, params)
    v2 = manifest.dataset_version(list(reversed(rows)), params)
    assert v1 == v2


def test_dataset_version_changes_with_content_and_params():
    rows = [_row(i) for i in range(5)]
    base = manifest.dataset_version(rows, {"seed": 0})
    # a different tile hash -> different fingerprint
    mutated = rows[:-1] + [_row(4, chash=999999)]
    assert manifest.dataset_version(mutated, {"seed": 0}) != base
    # a different build param -> different fingerprint
    assert manifest.dataset_version(rows, {"seed": 1}) != base


def test_corpus_meta_records_card_and_global_stats(tmp_path):
    rows = [_row(i) for i in range(5)]
    path = tmp_path / "corpus_meta.json"
    manifest.write_corpus_meta(
        path,
        rows=rows,
        global_mean=812.3,
        global_std=623.1,
        snapshot_date="2026-07",
        build_timestamp="2026-07-14T00:00:00Z",
        code_version="abc123",
    )
    meta = json.loads(path.read_text())
    assert meta["normalization"] == config.NORMALIZATION
    assert meta["tile_px"] == config.TILE_PX
    assert meta["storage_dtype"] == "int16"
    assert meta["clip_bounds"] == [config.CLIP_MIN_M, config.CLIP_MAX_M]
    assert meta["degenerate_std_eps"] == config.DEGENERATE_STD_EPS_M
    assert meta["per_stratum_quota"] == config.PER_STRATUM_QUOTA
    assert meta["seed"] == config.RANDOM_SEED
    assert meta["n_tiles"] == 5
    # ADR-0003 handoff: corpus-wide land μ/σ for the global-normalization mode
    assert meta["global_mean"] == pytest.approx(812.3)
    assert meta["global_std"] == pytest.approx(623.1)
    # reproducibility fingerprint present and stable
    assert meta["dataset_version"] == manifest.dataset_version(rows, meta["build_params"])
