"""Shared fixtures: a tiny materialized corpus with a split column (issue #89).

`test_loader.py` builds its own corpus inline and predates this; the Phase-1 tests need the
same thing but with *geography and splits they control* — a test geocell, a val COG, and the
ability to contaminate the assignment on purpose. That is a builder, not a fixture, so it
lives here where both `test_pretrain_fingerprint.py` and `test_pretrain_data.py` reach it.
"""

from __future__ import annotations

import numpy as np
import pytest

from dataset import manifest, normalize, shards


def _tile(rng: np.random.Generator, base: float) -> np.ndarray:
    """A plausible 224x224 elevation window: land everywhere, non-degenerate relief."""
    return (base + rng.normal(0, 200, size=(224, 224))).astype(np.float32)


def build_corpus(
    root,
    cogs,
    *,
    tiles_per_cog: int = 2,
    splits: dict[str, str] | None = None,
    tile_fn=None,
    metrics_fn=None,
):
    """Materialize a corpus at ``root`` from ``cogs`` — ``[(stem, geocell_id, split), ...]``.

    Every tile of a COG takes that COG's ``split`` label, so the blocking the real assignment
    is drawn at (whole geocells for ``test``, whole COGs for ``val``) is expressible here.
    ``splits`` overrides individual tiles by id, which is how a *contaminated* assignment —
    one train tile inside a held-out block — gets built for the test that must reject it.

    ``tile_fn(rng, base) -> (224, 224) float array`` replaces the default noise window. The
    default is right for every test that only reads a sample; #93's training tests pass a
    *smooth* generator instead, because a model cannot reduce a reconstruction loss on noise
    and "the loss falls" would then be an untestable claim.

    ``metrics_fn(cog_index, tile_index) -> dict`` overrides the frozen metric vector and the
    ``stratum_id`` beside it. The default is one constant row, which is right for every test
    that never reads those columns and wrong for #99's, whose kNN regression targets *are*
    ``slope_mean`` and ``geomorphon_entropy`` — a constant target makes a no-skill baseline
    score perfectly and the probe untestable.
    """
    rng = np.random.default_rng(0)
    tile_fn = tile_fn or _tile
    rows: list[manifest.TileRecord] = []
    total = sum_ = sumsq = 0.0
    with shards.ShardWriter(root / "shards") as writer:
        for c, (stem, geocell, split) in enumerate(cogs):
            for j in range(tiles_per_cog):
                tile, reason = normalize.prepare_tile(tile_fn(rng, 500.0 + 100 * c))
                assert tile is not None, reason
                ref = writer.add(tile.raw_i16)
                vals = tile.raw_i16[normalize.land_mask(tile.raw_i16)].astype(np.float64)
                sum_ += float(vals.sum())
                sumsq += float(np.square(vals).sum())
                total += vals.size
                tile_id = manifest.make_tile_id(stem, j * 224, 0)
                metrics = {
                    "stratum_id": j % 20,
                    "slope_mean": 0.1,
                    "local_relief": 30.0,
                    "texture": 0.5,
                    "elevation_relief_ratio": 0.4,
                    "mean_elevation": 500.0 + 100 * c,
                    "geomorphon_entropy": 1.8,
                }
                metrics.update(metrics_fn(c, j) if metrics_fn is not None else {})
                rows.append(
                    manifest.TileRecord(
                        tile_id=tile_id,
                        shard_id=ref.shard_id,
                        row_index=ref.row_index,
                        source_cog=stem,  # the bare stem, as the shipped manifest holds it
                        win_row0=j * 224,
                        win_col0=0,
                        lon=float(c),
                        lat=float(j),
                        lat_band="0-50",
                        geocell_id=geocell,
                        mean=tile.mean,
                        std=tile.std,
                        land_frac=tile.land_frac,
                        provenance="base",
                        **metrics,
                        edm_edited_frac=0.0,
                        flm_fill_frac=0.0,
                        wbm_water_frac=0.0,
                        water_affected=False,
                        coastline=False,
                        content_hash=ref.content_hash,
                        split=(splits or {}).get(tile_id, split),
                    )
                )
    mean = sum_ / total
    manifest.write_manifest(rows, root / "manifest.parquet")
    manifest.write_corpus_meta(
        root / "corpus_meta.json",
        rows=rows,
        global_mean=mean,
        global_std=float(np.sqrt(max(sumsq / total - mean * mean, 0.0))),
        snapshot_date="test",
        build_timestamp="2026-07-28T00:00:00Z",
        code_version="test",
        selection_list_md5="test-selection-md5",
        catalog_version="test-catalog",
        centroids_md5="test-centroids-md5",
        scaler_md5="test-scaler-md5",
    )
    return root


#: Three geocells: one held out whole as ``test``, one holding a ``val`` COG beside a train
#: COG, one all train. The smallest shape that can express both blocking units at once.
SPLIT_COGS = [
    ("Copernicus_DSM_COG_10_N45_00_E006_00_DEM", "+000_+040", "train"),
    ("Copernicus_DSM_COG_10_N45_00_E007_00_DEM", "+000_+040", "val"),
    ("Copernicus_DSM_COG_10_N55_00_E010_00_DEM", "+010_+050", "test"),
    ("Copernicus_DSM_COG_10_N56_00_E011_00_DEM", "+010_+050", "test"),
    ("Copernicus_DSM_COG_10_S01_00_E036_00_DEM", "+030_+000", "train"),
]


@pytest.fixture
def corpus_factory():
    """:func:`build_corpus` itself, for tests that need geography of their own."""
    return build_corpus


@pytest.fixture
def set_split():
    """Rewrite a built corpus's ``split`` column — how a contaminated assignment is staged.

    Deliberately a raw column write rather than anything in `dataset.splits`: the point of
    the tests that use it is that the *reader* refuses a bad split, whoever wrote it.
    """

    def _set(corpus, labels: list[str | None]):
        import pyarrow as pa
        import pyarrow.parquet as pq

        from dataset.manifest import MANIFEST_SCHEMA

        path = corpus / "manifest.parquet"
        table = pq.read_table(path)
        assert len(labels) == table.num_rows, f"{len(labels)} labels for {table.num_rows} rows"
        patched = table.set_column(
            table.schema.get_field_index("split"), "split", pa.array(labels, type=pa.string())
        )
        pq.write_table(patched.cast(MANIFEST_SCHEMA), path)
        return corpus

    return _set


@pytest.fixture
def write_assignment():
    """Write the committed ``tile_id,split`` CSV from a corpus's own manifest column.

    Hand-rolled rather than routed through `dataset.splits.write_assignment`, for the reason
    #89 exists before #77's driver is ever run again: what these tests exercise is a *reader*
    of a committed artifact, and the artifact's format is two columns and a header.
    """

    def _write(path, corpus, labels: list[str] | None = None):
        import pyarrow.parquet as pq

        table = pq.read_table(corpus / "manifest.parquet", columns=["tile_id", "split"])
        rows = zip(table["tile_id"].to_pylist(), labels or table["split"].to_pylist())
        path.parent.mkdir(parents=True, exist_ok=True)
        body = [f"{t},{s}" for t, s in sorted(rows)]
        path.write_text("\n".join(["tile_id,split"] + body) + "\n")
        return path

    return _write


@pytest.fixture
def split_corpus(tmp_path):
    """A corpus whose ``split`` column is filled the way #77 fills the real one."""
    return build_corpus(tmp_path, SPLIT_COGS)


@pytest.fixture
def unsplit_corpus(tmp_path):
    """The same corpus before #77 lands: ``split`` declared and null on every row."""
    return build_corpus(tmp_path, [(stem, cell, None) for stem, cell, _ in SPLIT_COGS])
