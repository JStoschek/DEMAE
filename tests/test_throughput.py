"""Throughput projection + the cloud-burst verdict (synthetic, deterministic)."""

from __future__ import annotations

import numpy as np

from dataset import throughput
from dataset.tiling import SyntheticCogReader


def _bucket():
    rng = np.random.default_rng(0)
    tiles = {}
    cells = [
        "Copernicus_DSM_COG_10_N45_00_E006_00_DEM",
        "Copernicus_DSM_COG_10_S01_00_E036_00_DEM",
        "Copernicus_DSM_COG_10_N20_00_W100_00_DEM",
        "Copernicus_DSM_COG_10_N55_00_E010_00_DEM",
    ]
    for i, stem in enumerate(cells):
        tiles[stem] = (
            1200 + rng.normal(0, 60 + 40 * i, size=(672, 672))
        ).astype(np.float32)
    return SyntheticCogReader(tiles)


def test_report_has_rates_and_projection(tmp_path):
    r = throughput.measure_throughput(
        _bucket(), tmp_path, sample_cogs=4, per_stratum_quota=6
    )
    assert r.n_tiles_measured > 0
    assert r.per_window_read_s >= 0.0
    assert r.projected_total_single_thread_s >= r.projected_total_parallel_s
    assert r.recommendation in {"local", "cloud-burst"}
    # projection reflects the full workload, not the smoke sample
    assert r.projected_materialize_s == (
        throughput.FULL_N_TILES
        * (r.per_window_read_s + r.per_window_process_s)
    )


def test_generous_budget_recommends_local(tmp_path):
    r = throughput.measure_throughput(
        _bucket(), tmp_path, sample_cogs=4, per_stratum_quota=6, budget_hours=10_000
    )
    assert r.recommendation == "local"


def test_zero_budget_recommends_cloud_burst(tmp_path):
    r = throughput.measure_throughput(
        _bucket(), tmp_path, sample_cogs=4, per_stratum_quota=6, budget_hours=0.0
    )
    assert r.recommendation == "cloud-burst"


def test_more_workers_shortens_projection(tmp_path):
    slow = throughput.measure_throughput(
        _bucket(), tmp_path / "a", sample_cogs=4, per_stratum_quota=6, workers=1
    )
    fast = throughput.measure_throughput(
        _bucket(), tmp_path / "b", sample_cogs=4, per_stratum_quota=6, workers=16
    )
    assert fast.projected_total_parallel_s < slow.projected_total_parallel_s


def test_format_report_is_readable(tmp_path):
    r = throughput.measure_throughput(
        _bucket(), tmp_path, sample_cogs=4, per_stratum_quota=6
    )
    text = throughput.format_report(r)
    assert "RECOMMENDATION" in text
    assert r.recommendation.upper() in text
