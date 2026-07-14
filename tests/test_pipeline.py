"""End-to-end corpus build against an in-memory synthetic Copernicus bucket."""

from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq
import pytest

from dataset import config, pipeline, shards
from dataset.tiling import SyntheticCogReader

TILE = config.TILE_PX


def _terrain(stem_seed, *, size=672, ruggedness=80.0, ocean_rows=0):
    rng = np.random.default_rng(stem_seed)
    base = rng.normal(1500, 300, size=(size, size))
    rough = rng.normal(0, ruggedness, size=(size, size))
    arr = (base + rough).astype(np.float32)
    if ocean_rows:
        arr[:ocean_rows] = 0.0  # a band of ocean fill
    return arr


def _synthetic_bucket():
    # Distinct 1-degree cells (distinct geocells) with varied ruggedness.
    return SyntheticCogReader(
        {
            "Copernicus_DSM_COG_10_N45_00_E006_00_DEM": _terrain(1, ruggedness=200.0),
            "Copernicus_DSM_COG_10_S01_00_E036_00_DEM": _terrain(2, ruggedness=90.0),
            "Copernicus_DSM_COG_10_N20_00_W100_00_DEM": _terrain(3, ruggedness=30.0),
            "Copernicus_DSM_COG_10_N55_00_E010_00_DEM": _terrain(
                4, ruggedness=120.0, ocean_rows=300
            ),
        }
    )


def _run(tmp_path, **kw):
    return pipeline.run_pipeline(
        _synthetic_bucket(),
        tmp_path,
        per_stratum_quota=kw.pop("per_stratum_quota", 3),
        per_tile_cap=kw.pop("per_tile_cap", 64),
        snapshot_date="2026-07",
        code_version="testsha",
        build_timestamp="2026-07-14T00:00:00Z",
        **kw,
    )


def test_pipeline_writes_shards_manifest_and_meta(tmp_path):
    res = _run(tmp_path)
    assert res.n_candidates > 0
    assert res.n_written > 0
    assert res.manifest_path.exists()
    assert res.corpus_meta_path.exists()
    assert (res.shard_dir).exists()
    # manifest row count matches what was written
    assert pq.read_table(res.manifest_path).num_rows == res.n_written


def test_pipeline_manifest_joins_to_shard_content(tmp_path):
    res = _run(tmp_path)
    table = pq.read_table(res.manifest_path).to_pylist()
    for row in table[:10]:
        mm = np.load(shards.shard_path(res.shard_dir, row["shard_id"]), mmap_mode="r")
        tile = mm[row["row_index"]]
        # the stored content hash matches the actual stored bytes
        assert shards.content_hash(tile) == row["content_hash"]
        # land-only mean roughly matches the recomputed raw mean
        land = tile[tile != 0].astype(float)
        assert row["mean"] == pytest.approx(land.mean(), abs=1.0)


def test_pipeline_stratifies_and_spreads(tmp_path):
    res = _run(tmp_path, per_stratum_quota=3)
    rows = pq.read_table(res.manifest_path).to_pylist()
    strata = {r["stratum"] for r in rows}
    # multiple strata represented; geographic spread across >1 source cell
    assert len(strata) >= 2
    assert len({r["geocell_id"] for r in rows}) >= 2


def test_pipeline_meta_has_global_stats_and_version(tmp_path):
    res = _run(tmp_path)
    meta = json.loads(res.corpus_meta_path.read_text())
    assert meta["global_std"] > 0
    assert meta["n_tiles"] == res.n_written
    assert meta["dataset_version"] == res.dataset_version
    assert meta["code_version"] == "testsha"


def test_pipeline_is_deterministic(tmp_path):
    a = _run(tmp_path / "a")
    b = _run(tmp_path / "b")
    assert a.dataset_version == b.dataset_version
    assert a.n_written == b.n_written


def test_pipeline_limit_cogs_smoke(tmp_path):
    res = _run(tmp_path, limit_cogs=1)
    assert res.timings.n_cogs_scored == 1
    assert res.n_written > 0


def test_pipeline_records_timings(tmp_path):
    res = _run(tmp_path)
    t = res.timings
    assert t.n_cogs_scored == 4
    assert t.n_windows_read > 0
    assert t.per_window_read_s >= 0.0
    assert t.per_cog_overview_s >= 0.0


def test_pipeline_parallel_matches_sequential(tmp_path):
    # The whole safety claim of the thread pool: workers>1 must produce a
    # byte-identical corpus to workers=1 (reads consumed in COG order).
    seq = _run(tmp_path / "seq", per_stratum_quota=5, per_tile_cap=1000, workers=1)
    par = _run(tmp_path / "par", per_stratum_quota=5, per_tile_cap=1000, workers=4)

    assert par.dataset_version == seq.dataset_version
    assert par.n_written == seq.n_written

    seq_rows = pq.read_table(seq.manifest_path).to_pylist()
    par_rows = pq.read_table(par.manifest_path).to_pylist()
    assert par_rows == seq_rows  # same rows, same shard/row assignment, same order
    assert json.loads(par.corpus_meta_path.read_text())["global_mean"] == pytest.approx(
        json.loads(seq.corpus_meta_path.read_text())["global_mean"]
    )


def test_pipeline_parallel_scores_all_cogs(tmp_path):
    res = _run(tmp_path, workers=4)
    assert res.timings.n_cogs_scored == 4  # every COG scored under the pool
    assert res.n_written > 0


def test_pipeline_drives_progress_bars(tmp_path):
    events: list[tuple[str, int]] = []

    class _Bar:
        def __init__(self, total, desc):
            events.append((desc, total))
            self.n = 0

        def update(self, n=1):
            self.n += n

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    res = _run(tmp_path, progress=lambda total, desc: _Bar(total, desc))
    descs = [d for d, _ in events]
    # both long stages open a bar, sized to their real work
    assert any("scoring" in d for d in descs)
    assert any("materializing" in d for d in descs)
    score_total = next(t for d, t in events if "scoring" in d)
    mat_total = next(t for d, t in events if "materializing" in d)
    assert score_total == res.timings.n_cogs_scored
    assert mat_total == res.n_selected


def test_pipeline_progress_defaults_silent(tmp_path):
    # No factory -> the no-op bar; build still succeeds and writes tiles.
    res = _run(tmp_path)
    assert res.n_written > 0


def test_pipeline_ocean_windows_dropped(tmp_path):
    # the N55 cell has a 300-row ocean band; those windows must not be written
    res = _run(tmp_path, per_stratum_quota=50, per_tile_cap=1000)
    rows = pq.read_table(res.manifest_path).to_pylist()
    for r in rows:
        if r["source_cog"].startswith("Copernicus_DSM_COG_10_N55"):
            # any surviving N55 window is below the ocean band (row0 >= 224-ish)
            assert r["land_frac"] >= 0.25
