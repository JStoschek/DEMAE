"""Phase-E1 materialization driver (issue #51).

Every test runs against the in-memory :class:`~dataset.tiling.SyntheticCogReader`
and a hand-built gated-catalog fixture, so the whole driver — selection load,
catalog join, screening, shard write, ledger, checkpoint/resume, determinism —
is exercised end to end in CI with no network. ``synthetic.py`` (the fabricated
S3 bucket the retired TRI pipeline used) is gone; the ``CogReader`` Protocol it
sat behind is what makes that unnecessary.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from dataset import catalog as catalog_mod
from dataset import config, manifest, materialize, shards
from dataset.tiling import SyntheticCogReader

# Four 672x672 COGs — a 3x3 window grid each, so a whole "corpus" is 36 windows.
_STEMS = [
    "Copernicus_DSM_COG_10_N45_00_E006_00_DEM",
    "Copernicus_DSM_COG_10_N46_00_E006_00_DEM",
    "Copernicus_DSM_COG_10_S01_00_W123_00_DEM",
    "Copernicus_DSM_COG_10_N07_00_E007_00_DEM",
]
_DIM = 672
_TILE = config.TILE_PX


def _hilly(n: int, seed: int = 0) -> np.ndarray:
    """Terrain with enough relief that every window clears the degenerate screen."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    base = 1200 + 400 * np.sin(x / 37.0) + 300 * np.cos(y / 29.0)
    return (base + rng.normal(0, 25, size=(n, n))).astype(np.float32)


def _arrays() -> dict[str, np.ndarray]:
    """The synthetic bucket: three healthy COGs plus one carrying both drop causes."""
    a = {stem: _hilly(_DIM, seed=i) for i, stem in enumerate(_STEMS)}
    doomed = a[_STEMS[3]]
    doomed[0:_TILE, 0:_TILE] = 0.0  # window (0,0) -> zero_fill
    doomed[0:_TILE, _TILE : 2 * _TILE] = 800.0  # window (0,224) -> degenerate (std 0)
    return a


def _selected_windows() -> list[tuple[str, int, str]]:
    """``(window_id, stratum_id, provenance)`` — the fixture's selection list.

    Deliberately *not* grouped by COG and not in COG order: the driver has to do
    the grouping itself, and a test that hands it pre-grouped rows would not check
    that. Strata and provenance are spread so the per-stratum roll-up has something
    to count and the coastal floor line is exercised.
    """
    rows = []
    for i, stem in enumerate(_STEMS):
        for j, (r0, c0) in enumerate([(0, 0), (0, _TILE), (_TILE, _TILE)]):
            prov = "floor:coastline" if (i + j) % 5 == 0 else "base"
            rows.append((f"{stem}:{r0}:{c0}", (i + j) % 3, prov))
    # sorted by (stratum_id, window_id), as dataset.select emits it
    return sorted(rows, key=lambda t: (t[1], t[0]))


def _write_selection(path: Path, rows: list[tuple[str, int, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["window_id,stratum_id,provenance"]
    lines += [f"{w},{s},{p}" for w, s, p in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


def _write_catalog(path: Path, rows: list[tuple[str, int, str]]) -> Path:
    """A gated-catalog stand-in holding exactly the columns the manifest copies."""
    import pyarrow as pa

    n = len(rows)
    ids = [w for w, _, _ in rows]
    cols = {
        "window_id": ids,
        "stem": [w.rsplit(":", 2)[0] for w in ids],
        "band": ["0-50"] * n,
        "geocell": ["+000_+040"] * n,
        "lon": [6.0 + i * 0.01 for i in range(n)],
        "lat": [45.0 + i * 0.01 for i in range(n)],
        "slope_mean": [0.10 + i for i in range(n)],
        "local_relief": [30.0 + i for i in range(n)],
        "texture": [0.5 + i for i in range(n)],
        "elevation_relief_ratio": [0.4 + i for i in range(n)],
        "mean_elevation": [1000.0 + i for i in range(n)],
        "geomorphon_entropy": [1.8 + i for i in range(n)],
        "edm_edited_frac": [0.01] * n,
        "flm_fill_frac": [0.02] * n,
        "wbm_water_frac": [0.03] * n,
        "water_affected": [i % 2 == 0 for i in range(n)],
        "coastline": [p == "floor:coastline" for _, _, p in rows],
        "keep": [True] * n,
    }
    parts = path / catalog_mod.PARTS_DIR
    parts.mkdir(parents=True, exist_ok=True)
    table = pa.table(cols)
    # two part files, because the real catalog is partitioned and the loader has to
    # concatenate rather than assume one
    pq.write_table(table.slice(0, n // 2), parts / "part-00000.parquet")
    pq.write_table(table.slice(n // 2), parts / "part-00001.parquet")
    (path / catalog_mod.META_FILE).write_text(json.dumps({"n_rows": n, "gated": True}))
    return path


@pytest.fixture
def build(tmp_path):
    """Everything a driver run needs, wired and returned as a small namespace."""
    rows = _selected_windows()
    selection = _write_selection(tmp_path / "selection.csv", rows)
    catalog_dir = _write_catalog(tmp_path / "catalog_gated", rows)
    reader = SyntheticCogReader(_arrays())

    class _Build:
        pass

    b = _Build()
    b.rows = rows
    b.selection = selection
    b.catalog_dir = catalog_dir
    b.reader = reader
    b.out_dir = tmp_path / "corpus"
    b.ledger = tmp_path / "ledger.parquet"
    # Stand-ins for the committed Phase-D artifacts the corpus card fingerprints, so
    # no test depends on the working directory being the repo root.
    b.centroids = tmp_path / "centroids.json"
    b.centroids.write_text('{"k": 20}')
    b.scaler = tmp_path / "scaler.json"
    b.scaler.write_text('{"mu": [0.0], "sigma": [1.0]}')
    b.tmp = tmp_path
    return b


def _run(b, *, reader=None, out_dir=None, ledger=None, **kw):
    return materialize.run(
        cog_reader=reader if reader is not None else b.reader,
        selection_path=b.selection,
        catalog_dir=b.catalog_dir,
        out_dir=out_dir or b.out_dir,
        ledger_path=ledger or b.ledger,
        centroids_path=b.centroids,
        scaler_path=b.scaler,
        rows_per_shard=kw.pop("rows_per_shard", 4),
        **kw,
    )


def _ledger(path):
    return pq.read_table(path).to_pylist()


def _manifest(out_dir):
    return pq.read_table(Path(out_dir) / "manifest.parquet").to_pylist()


# --- Selection load -----------------------------------------------------------


def test_selection_is_grouped_by_cog_in_a_stable_order(build):
    jobs = materialize.load_selection(build.selection)
    assert [j.stem for j in jobs] == sorted(_STEMS)
    # windows within a COG are sorted by pixel origin, so one open serves them all
    for job in jobs:
        origins = [(w.win_row0, w.win_col0) for w in job.windows]
        assert origins == sorted(origins)
    assert sum(len(j.windows) for j in jobs) == len(build.rows)


def test_selection_carries_stratum_and_provenance_through(build):
    jobs = materialize.load_selection(build.selection)
    by_id = {w.window_id: w for j in jobs for w in j.windows}
    for window_id, stratum_id, provenance in build.rows:
        assert by_id[window_id].stratum_id == stratum_id
        assert by_id[window_id].provenance == provenance


def test_selection_rejects_a_duplicated_window(build, tmp_path):
    dup = _write_selection(tmp_path / "dup.csv", build.rows + build.rows[:1])
    with pytest.raises(ValueError, match="duplicate"):
        materialize.load_selection(dup)


# --- The end-to-end build -----------------------------------------------------


def test_build_writes_shards_manifest_ledger_and_meta(build):
    result = _run(build)

    assert (build.out_dir / "manifest.parquet").exists()
    assert (build.out_dir / "corpus_meta.json").exists()
    assert build.ledger.exists()
    assert result.n_shards == len(list((build.out_dir / "shards").glob("shard_*.npy")))
    assert result.n_selected == len(build.rows)
    assert result.n_tiles == len(_manifest(build.out_dir))


def test_ledger_has_one_row_per_selected_window(build):
    _run(build)
    led = _ledger(build.ledger)
    assert len(led) == len(build.rows)
    assert {r["window_id"] for r in led} == {w for w, _, _ in build.rows}
    assert {r["status"] for r in led} <= set(materialize.LEDGER_STATUSES)


def test_ledger_ok_rows_equal_the_manifest_exactly(build):
    """#50's invariant: the ledger's ``ok`` set *is* the materialized set."""
    _run(build)
    ok = {r["window_id"] for r in _ledger(build.ledger) if r["status"] == "ok"}
    written = {materialize.window_id_of(r) for r in _manifest(build.out_dir)}
    assert ok == written


def test_screen_drops_are_booked_by_cause_and_not_written(build):
    _run(build)
    by_id = {r["window_id"]: r for r in _ledger(build.ledger)}
    doomed = _STEMS[3]
    assert by_id[f"{doomed}:0:0"]["status"] == "zero_fill"
    assert by_id[f"{doomed}:0:{_TILE}"]["status"] == "degenerate"
    assert by_id[f"{doomed}:{_TILE}:{_TILE}"]["status"] == "ok"
    written = {materialize.window_id_of(r) for r in _manifest(build.out_dir)}
    assert f"{doomed}:0:0" not in written


def test_dropped_windows_get_no_null_shard_id_manifest_row(build):
    # #50 item 2: dropped windows live only in the ledger. A null-shard_id manifest
    # row would be an ADR-0003 read-path change, which is out of scope.
    _run(build)
    for row in _manifest(build.out_dir):
        assert row["shard_id"] is not None
        assert row["row_index"] is not None


def test_manifest_denormalizes_the_catalog_columns(build):
    _run(build)
    cat = {}
    for part in sorted((build.catalog_dir / catalog_mod.PARTS_DIR).glob("*.parquet")):
        for row in pq.read_table(part).to_pylist():
            cat[row["window_id"]] = row
    for row in _manifest(build.out_dir):
        src = cat[materialize.window_id_of(row)]
        assert row["lat_band"] == src["band"]
        assert row["geocell_id"] == src["geocell"]
        for col in ("lon", "lat", *catalog_mod.FROZEN_METRICS):
            assert row[col] == pytest.approx(src[col])
        for col in ("edm_edited_frac", "flm_fill_frac", "wbm_water_frac"):
            assert row[col] == pytest.approx(src[col])
        assert row["water_affected"] == src["water_affected"]
        assert row["coastline"] == src["coastline"]


def test_manifest_copies_stratum_and_provenance_from_the_selection(build):
    _run(build)
    sel = {w: (s, p) for w, s, p in build.rows}
    for row in _manifest(build.out_dir):
        stratum_id, provenance = sel[materialize.window_id_of(row)]
        assert row["stratum_id"] == stratum_id
        assert row["provenance"] == provenance


def test_manifest_conforms_to_the_v2_schema(build):
    _run(build)
    schema = pq.read_schema(build.out_dir / "manifest.parquet")
    assert schema.names == manifest.MANIFEST_SCHEMA.names
    assert schema.types == manifest.MANIFEST_SCHEMA.types


def test_tiles_land_where_the_manifest_says_they_do(build):
    _run(build)
    arrays = _arrays()
    for row in _manifest(build.out_dir):
        shard = np.load(
            shards.shard_path(build.out_dir / "shards", row["shard_id"]), mmap_mode="r"
        )
        stored = shard[row["row_index"]]
        window = arrays[row["source_cog"]][
            row["win_row0"] : row["win_row0"] + _TILE,
            row["win_col0"] : row["win_col0"] + _TILE,
        ]
        expected = materialize.normalize.to_int16(
            materialize.normalize.plausibility_clip(window)
        )
        assert np.array_equal(stored, expected)
        assert row["content_hash"] == shards.content_hash(expected)


def test_corpus_meta_carries_the_attrition_contract_and_fingerprints(build):
    result = _run(build)
    meta = json.loads((build.out_dir / "corpus_meta.json").read_text())
    # #47's E2 notebook reads this string to pick the live branch of the gate bar
    assert meta["attrition_contract"] == "accept_shortfall"
    assert meta["n_tiles"] == result.n_tiles
    assert meta["dataset_version"] == result.dataset_version
    params = meta["build_params"]
    assert params["selection_list_md5"] == materialize.file_md5(build.selection)
    assert params["centroids_md5"] and params["scaler_md5"] and params["catalog_version"]
    assert "seed" not in params


def test_corpus_meta_names_the_normalization_stat_source_outside_build_params(build):
    """#70: the card says which corpus ``global_mean``/``global_std`` describe.

    Top-level, not in ``build_params`` — #69's Gate-F bar reads
    ``corpus_meta.json:normalization_stat_source`` to pick the live F8 branch, and a
    ``build_params`` key would move every corpus's ``dataset_version``.
    """
    _run(build)
    meta = json.loads((build.out_dir / "corpus_meta.json").read_text())
    assert meta["normalization_stat_source"] == "tier_specific"
    assert "normalization_stat_source" not in meta["build_params"]


def test_global_stats_are_the_corpus_wide_land_mean_and_std(build):
    """ADR-0003's ``global`` mode: one streaming pass, not a second read."""
    result = _run(build)
    land = []
    for row in _manifest(build.out_dir):
        shard = np.load(
            shards.shard_path(build.out_dir / "shards", row["shard_id"]), mmap_mode="r"
        )
        tile = np.asarray(shard[row["row_index"]], dtype=np.float64)
        land.append(tile[tile != config.OCEAN_FILL_VALUE])
    allland = np.concatenate(land)
    assert result.global_mean == pytest.approx(allland.mean(), rel=1e-9)
    assert result.global_std == pytest.approx(allland.std(), rel=1e-6)


def test_loader_reads_the_built_corpus(build):
    pytest.importorskip("torch")
    from dataset import loader

    _run(build)
    ds = loader.TileDataset(build.out_dir)
    assert len(ds) == len(_manifest(build.out_dir))
    sample = ds[0]
    assert sample["image"].shape == (1, _TILE, _TILE)
    assert isinstance(sample["stratum_id"], int)


def test_result_reports_the_counts_the_ticket_asks_for(build):
    result = _run(build)
    assert result.status_counts["ok"] == result.n_tiles
    assert sum(result.status_counts.values()) == result.n_selected
    assert sum(result.per_stratum.values()) == result.n_tiles
    assert result.shard_bytes > 0
    assert result.n_cogs == len(_STEMS)
    # every number the ticket's closing line asks for reaches the printed report
    printed = "\n".join(result.summary_lines())
    for stratum_id, count in result.per_stratum.items():
        assert f"{stratum_id}:{count}" in printed
    assert f"{result.n_shards} shard" in printed
    assert "s/COG" in printed  # the unit E1's projection is stated in


def test_wall_clock_accumulates_across_a_resumed_run(build):
    """The ticket compares wall clock against E1's projection; a resumed run that
    reported only its final leg would flatter the comparison."""
    with pytest.raises(KeyboardInterrupt):
        _run(build, reader=_DyingReader(build.reader, after=2), checkpoint_every=1)
    first = json.loads((build.out_dir / "build_state.json").read_text())["elapsed_s"]
    assert first > 0

    result = _run(build)
    assert result.wall_s > first


# --- Read failures ------------------------------------------------------------


class _FlakyReader:
    """Wraps a reader and fails ``read_windows`` for chosen stems.

    ``fail_times`` makes the failure transient (the S3 blip the driver retries);
    leaving it ``None`` makes it permanent (the COG that never comes back).
    """

    def __init__(self, inner, stems, fail_times=None):
        self.inner = inner
        self.stems = set(stems)
        self.fail_times = fail_times
        self.calls: dict[str, int] = {}

    def list_cogs(self):
        return self.inner.list_cogs()

    def read_overview(self, cog, factor):
        return self.inner.read_overview(cog, factor)

    def read_windows(self, cog, origins, tile):
        n = self.calls.get(cog.stem, 0)
        self.calls[cog.stem] = n + 1
        if cog.stem in self.stems and (self.fail_times is None or n < self.fail_times):
            raise OSError("simulated /vsis3 read failure")
        return self.inner.read_windows(cog, origins, tile)


def test_a_transient_read_failure_is_retried_inside_the_run(build):
    reader = _FlakyReader(build.reader, [_STEMS[0]], fail_times=1)
    result = _run(build, reader=reader, read_attempts=3)
    assert result.status_counts.get("read_error", 0) == 0
    assert reader.calls[_STEMS[0]] == 2


def test_a_permanent_read_failure_is_booked_with_its_cause(build):
    reader = _FlakyReader(build.reader, [_STEMS[0]])
    result = _run(build, reader=reader, read_attempts=2)
    failed = [r for r in _ledger(build.ledger) if r["status"] == "read_error"]
    assert len(failed) == 3  # every selected window of that COG
    assert all(r["window_id"].startswith(_STEMS[0]) for r in failed)
    assert all("simulated" in r["detail"] for r in failed)
    # the failed COG's windows leave the corpus, on top of the fixture's screen drops
    assert result.status_counts["read_error"] == 3
    assert result.n_tiles == result.status_counts["ok"] == len(build.rows) - 3 - 2


def test_a_read_error_cog_is_retryable_on_a_later_run(build):
    """``accept_shortfall`` forbids backfilling a *different* window — it does not ask
    us to abandon a selected one to a transient blip. So the failure is booked, and
    the COG that produced it stays re-runnable."""
    reader = _FlakyReader(build.reader, [_STEMS[0]])
    first = _run(build, reader=reader, read_attempts=1)
    assert first.status_counts["read_error"] == 3

    healed = _run(build, retry_read_errors=True)
    assert healed.status_counts.get("read_error", 0) == 0
    assert healed.n_cogs == 1  # only the failed COG is re-read
    ids = [r["window_id"] for r in _ledger(build.ledger)]
    assert len(ids) == len(set(ids)) == len(build.rows)
    assert healed.n_tiles == len(_manifest(build.out_dir))


def test_retry_is_opt_in(build):
    _run(build, reader=_FlakyReader(build.reader, [_STEMS[0]]), read_attempts=1)
    again = _run(build)
    assert again.n_cogs == 0
    assert again.status_counts["read_error"] == 3


def test_a_healed_corpus_still_matches_the_ledger(build):
    _run(build, reader=_FlakyReader(build.reader, [_STEMS[0]]), read_attempts=1)
    result = _run(build, retry_read_errors=True)
    ok = {r["window_id"] for r in _ledger(build.ledger) if r["status"] == "ok"}
    assert ok == {materialize.window_id_of(r) for r in _manifest(build.out_dir)}
    assert result.n_tiles == len(ok)


def test_the_ledger_is_sorted_so_a_retry_run_reproduces_it(build, tmp_path):
    """The manifest's row order is the shard's and cannot move; the ledger's can, so
    it is sorted — a corpus healed by a retry ships the same ledger as a clean one."""
    clean = _run(build, out_dir=tmp_path / "c", ledger=tmp_path / "c.parquet")

    _run(
        build,
        reader=_FlakyReader(build.reader, [_STEMS[0]]),
        read_attempts=1,
        out_dir=tmp_path / "h",
        ledger=tmp_path / "h.parquet",
    )
    healed = _run(build, out_dir=tmp_path / "h", ledger=tmp_path / "h.parquet",
                  retry_read_errors=True)

    assert _ledger(tmp_path / "c.parquet") == _ledger(tmp_path / "h.parquet")
    assert healed.n_tiles == clean.n_tiles


def test_detail_is_null_for_everything_but_a_read_error(build):
    _run(build, reader=_FlakyReader(build.reader, [_STEMS[0]]), read_attempts=1)
    for row in _ledger(build.ledger):
        assert (row["detail"] is None) == (row["status"] != "read_error")


def test_a_wholesale_read_outage_aborts_instead_of_booking_the_corpus_dead(build):
    reader = _FlakyReader(build.reader, _STEMS)
    with pytest.raises(materialize.ReadOutage):
        _run(build, reader=reader, read_attempts=1, max_consecutive_read_errors=2)


# --- Resume -------------------------------------------------------------------


class _DyingReader(_FlakyReader):
    """Raises a non-``Exception`` after N COGs — a kill, not a read failure."""

    def __init__(self, inner, after):
        super().__init__(inner, [])
        self.after = after
        self.seen = 0

    def read_windows(self, cog, origins, tile):
        self.seen += 1
        if self.seen > self.after:
            raise KeyboardInterrupt("simulated kill")
        return self.inner.read_windows(cog, origins, tile)


def test_a_killed_run_resumes_where_it_stopped(build):
    with pytest.raises(KeyboardInterrupt):
        _run(build, reader=_DyingReader(build.reader, after=2), checkpoint_every=1)

    partial = len(_ledger(build.ledger))
    assert 0 < partial < len(build.rows)

    result = _run(build)
    assert result.n_skipped == 2
    assert result.n_cogs == len(_STEMS) - 2
    assert len(_ledger(build.ledger)) == len(build.rows)


def test_a_resumed_build_is_identical_to_an_uninterrupted_one(build, tmp_path):
    straight = _run(build, out_dir=tmp_path / "straight", ledger=tmp_path / "a.parquet")

    with pytest.raises(KeyboardInterrupt):
        _run(
            build,
            reader=_DyingReader(build.reader, after=2),
            out_dir=tmp_path / "resumed",
            ledger=tmp_path / "b.parquet",
            checkpoint_every=1,
        )
    resumed = _run(build, out_dir=tmp_path / "resumed", ledger=tmp_path / "b.parquet")

    assert resumed.dataset_version == straight.dataset_version
    assert _manifest(tmp_path / "straight") == _manifest(tmp_path / "resumed")
    assert _ledger(tmp_path / "a.parquet") == _ledger(tmp_path / "b.parquet")
    assert resumed.global_mean == pytest.approx(straight.global_mean)
    for shard_id in range(straight.n_shards):
        a = np.load(shards.shard_path(tmp_path / "straight" / "shards", shard_id))
        b = np.load(shards.shard_path(tmp_path / "resumed" / "shards", shard_id))
        assert np.array_equal(a, b)


def test_resume_discards_rows_written_after_the_last_checkpoint(build):
    """The crash window: a COG finishes, then the process dies before the commit.

    Its rows are in the shard and may be in the parquet files, but the state file
    never claimed them — so the resumed run re-reads that COG and must not end up
    with the window booked twice.
    """
    with pytest.raises(KeyboardInterrupt):
        _run(build, reader=_DyingReader(build.reader, after=3), checkpoint_every=99)
    assert not build.ledger.exists() or len(_ledger(build.ledger)) == 0

    result = _run(build)
    ids = [r["window_id"] for r in _ledger(build.ledger)]
    assert len(ids) == len(set(ids)) == len(build.rows)
    assert result.n_tiles == len(_manifest(build.out_dir))


def test_a_finished_build_re_run_is_a_no_op(build):
    first = _run(build)
    again = _run(build)
    assert again.n_cogs == 0
    assert again.n_skipped == len(_STEMS)
    assert again.dataset_version == first.dataset_version


def test_resume_refuses_a_different_selection_list(build, tmp_path):
    with pytest.raises(KeyboardInterrupt):
        _run(build, reader=_DyingReader(build.reader, after=2), checkpoint_every=1)
    moved = _write_selection(tmp_path / "moved.csv", build.rows[:-1])
    with pytest.raises(ValueError, match="selection"):
        materialize.run(
            cog_reader=build.reader,
            selection_path=moved,
            catalog_dir=build.catalog_dir,
            out_dir=build.out_dir,
            ledger_path=build.ledger,
            centroids_path=build.centroids,
            scaler_path=build.scaler,
            rows_per_shard=4,
        )


def test_restart_throws_the_previous_attempt_away(build):
    with pytest.raises(KeyboardInterrupt):
        _run(build, reader=_DyingReader(build.reader, after=2), checkpoint_every=1)
    result = _run(build, restart=True)
    assert result.n_skipped == 0
    assert result.n_cogs == len(_STEMS)
    assert len(_ledger(build.ledger)) == len(build.rows)


# --- Determinism (#44's bar) --------------------------------------------------


def test_two_runs_reproduce_the_manifest_and_the_fingerprint(build, tmp_path):
    a = _run(build, out_dir=tmp_path / "a", ledger=tmp_path / "a.parquet")
    b = _run(build, out_dir=tmp_path / "b", ledger=tmp_path / "b.parquet")
    assert a.dataset_version == b.dataset_version
    assert _manifest(tmp_path / "a") == _manifest(tmp_path / "b")
    assert (tmp_path / "a" / "manifest.parquet").read_bytes() == (
        tmp_path / "b" / "manifest.parquet"
    ).read_bytes()


def test_shard_order_is_fixed_by_the_driver_not_by_the_executor(build, tmp_path):
    """#51 item 8's honest limit, closed: ``ordered_futures`` + a serial writer."""
    serial = _run(build, out_dir=tmp_path / "s", ledger=tmp_path / "s.parquet", workers=1)
    fanned = _run(
        build,
        out_dir=tmp_path / "p",
        ledger=tmp_path / "p.parquet",
        workers=4,
        executor="thread",
    )
    assert [r["tile_id"] for r in _manifest(tmp_path / "s")] == [
        r["tile_id"] for r in _manifest(tmp_path / "p")
    ]
    assert serial.dataset_version == fanned.dataset_version


def test_the_process_pool_path_pickles_and_agrees(build, tmp_path):
    """The live executor, on the synthetic bucket.

    Threads would never catch it if the task graph stopped pickling — and the live
    pass runs on processes, because ``rasterio.open`` is GIL-serialized (E1). This
    exercises the real submission path and checks it lands on the same corpus.
    """
    serial = _run(build, out_dir=tmp_path / "s", ledger=tmp_path / "s.parquet", workers=1)
    spawned = _run(
        build,
        out_dir=tmp_path / "p",
        ledger=tmp_path / "p.parquet",
        workers=2,
        executor="process",
    )
    assert spawned.dataset_version == serial.dataset_version
    assert _manifest(tmp_path / "p") == _manifest(tmp_path / "s")


# --- Guards -------------------------------------------------------------------


def test_a_window_missing_from_the_catalog_is_refused(build, tmp_path):
    rows = build.rows + [(f"{_STEMS[0]}:448:0", 0, "base")]
    sel = _write_selection(tmp_path / "extra.csv", rows)
    with pytest.raises(KeyError, match="not in the gated catalog"):
        materialize.run(
            cog_reader=build.reader,
            selection_path=sel,
            catalog_dir=build.catalog_dir,
            out_dir=build.out_dir,
            ledger_path=build.ledger,
            centroids_path=build.centroids,
            scaler_path=build.scaler,
            rows_per_shard=4,
        )


def test_limit_cogs_runs_a_prefix_without_claiming_a_full_build(build):
    result = _run(build, limit_cogs=2)
    assert result.n_cogs == 2
    assert result.complete is False
    assert not (build.out_dir / "corpus_meta.json").exists()


def test_no_land_is_declared_but_structurally_unreachable():
    # prepare_tile can only report no_land when land_frac == 0, i.e. every pixel is
    # the exact-zero fill — which trips the zero_fill screen first at any threshold
    # below 1.0. The status stays in the vocabulary #50 fixed; this records why it
    # will read zero on the shipped ledger.
    assert "no_land" in materialize.LEDGER_STATUSES
    assert config.LAND_ZERO_FRAC_DROP < 1.0


def test_cli_builds_a_corpus(build, capsys):
    rc = materialize.main(
        [
            "--selection", str(build.selection),
            "--catalog-dir", str(build.catalog_dir),
            "--out-dir", str(build.out_dir),
            "--ledger", str(build.ledger),
            "--centroids", str(build.centroids),
            "--scaler", str(build.scaler),
            "--workers", "1",
            "--rows-per-shard", "4",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "selected" in out.lower()
    assert (build.out_dir / "corpus_meta.json").exists()
