"""Tile manifest (Parquet) + corpus card (``corpus_meta.json``), ADR-0002 §3
(schema v2 addendum, Phase E / issue #48).

A single typed Parquet file indexes every tile — the loader loads it whole and
joins to the shards by ``(shard_id, row_index)``. A JSON sidecar holds the
corpus-level constants, the ADR-0003 corpus-wide land ``mean``/``std`` (so the
loader's ``global`` normalization mode works), and a ``dataset_version``
fingerprint that pins this exact build for reproducibility and the model card.

Schema v2 (issue #48) redraws the TRI-era row for the metric-vector corpus: it
drops ``tri`` (nothing computes it since #44 retired the TRI path), renames
``stratum`` → ``stratum_id`` (the k=20 k-means label #44 shipped, not a
ruggedness quintile), and **denormalizes** the frozen metric vector, the
mask/water fractions, and the selection ``provenance`` straight from the gated
catalog and the Phase-D selection list. The manifest is thus self-contained: a
corpus consumer or the model card needs no second file. See ADR-0002 §3a.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from . import config

# Typed per-row schema (ADR-0002 §3, v2 addendum §3a). Order here defines column
# order on disk. Source of each column is noted: materialization computes it from
# the full-res read, "catalog" copies it from ``data/catalog_gated/`` by window id,
# "selection" copies it from the Phase-D selection list.
MANIFEST_SCHEMA = pa.schema(
    [
        ("tile_id", pa.string()),
        ("shard_id", pa.int32()),
        ("row_index", pa.int32()),
        ("source_cog", pa.string()),
        ("win_row0", pa.int32()),
        ("win_col0", pa.int32()),
        ("lon", pa.float64()),
        ("lat", pa.float64()),
        ("lat_band", pa.string()),
        ("geocell_id", pa.string()),
        ("mean", pa.float64()),  # per-tile land stats — computed at full res
        ("std", pa.float64()),
        ("land_frac", pa.float64()),
        ("stratum_id", pa.int32()),  # selection: k=20 k-means label (was `stratum`)
        ("provenance", pa.string()),  # selection: {base, floor:coastline} (#42)
        # Frozen metric vector (ADR-0004), raw values copied from the catalog —
        # the exact numbers the sampler clustered on, so E2.1a compares like-for-like.
        ("slope_mean", pa.float64()),
        ("local_relief", pa.float64()),
        ("texture", pa.float64()),
        ("elevation_relief_ratio", pa.float64()),
        ("mean_elevation", pa.float64()),
        ("geomorphon_entropy", pa.float64()),
        # Mask/water facts, copied from the catalog (gates are columns, not drops).
        ("edm_edited_frac", pa.float64()),
        ("flm_fill_frac", pa.float64()),
        ("wbm_water_frac", pa.float64()),
        ("water_affected", pa.bool_()),
        ("coastline", pa.bool_()),
        ("content_hash", pa.uint64()),
        ("split", pa.string()),  # null this tier; Phase-2 fills the geographic hold-out
    ]
)


@dataclass(frozen=True)
class TileRecord:
    """One manifest row — the full provenance + storage join for a tile.

    Field order mirrors ``MANIFEST_SCHEMA`` (v2, issue #48). ``split`` keeps its
    default so it stays optional; every other field is required.
    """

    tile_id: str
    shard_id: int
    row_index: int
    source_cog: str
    win_row0: int
    win_col0: int
    lon: float
    lat: float
    lat_band: str
    geocell_id: str
    mean: float
    std: float
    land_frac: float
    stratum_id: int
    provenance: str
    slope_mean: float
    local_relief: float
    texture: float
    elevation_relief_ratio: float
    mean_elevation: float
    geomorphon_entropy: float
    edm_edited_frac: float
    flm_fill_frac: float
    wbm_water_frac: float
    water_affected: bool
    coastline: bool
    content_hash: int
    split: str | None = None


def make_tile_id(cog_stem: str, win_row0: int, win_col0: int) -> str:
    """Stable per-tile id ``{cog_stem}_{row0}_{col0}`` (ADR-0002 §3)."""
    return f"{cog_stem}_{win_row0}_{win_col0}"


def write_manifest(rows: list[TileRecord], path: Path | str) -> None:
    """Write the tile manifest to a single Parquet file."""
    cols = {name: [] for name in MANIFEST_SCHEMA.names}
    for r in rows:
        d = asdict(r)
        for name in cols:
            cols[name].append(d[name])
    table = pa.table(cols, schema=MANIFEST_SCHEMA)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def dataset_version(rows: list[TileRecord], build_params: dict) -> str:
    """Blake2b fingerprint over schema version + sorted ``(tile_id, content_hash)``
    + build params (ADR-0002 §3).

    Independent of row order (the pairs are sorted), so two builds that produced
    the same tiles from the same recipe fingerprint identically; any change in
    tile content or recipe params flips it.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update(str(config.MANIFEST_SCHEMA_VERSION).encode())
    for tid, chash in sorted((r.tile_id, r.content_hash) for r in rows):
        h.update(tid.encode())
        h.update(int(chash).to_bytes(8, "little"))
    h.update(json.dumps(build_params, sort_keys=True, default=str).encode())
    return h.hexdigest()


def build_params(
    *,
    selection_list_md5: str,
    catalog_version: str,
    centroids_md5: str,
    scaler_md5: str,
) -> dict:
    """The recipe parameters that define a corpus build (feed the fingerprint).

    The **seed is no longer the contract** (#43/#44): the persisted Phase-D
    selection list is. So the fingerprint pins the selection-list md5 — which alone
    fixes *which* windows, their ``stratum_id`` and their ``provenance``, since all
    three columns are copied from it — plus the ``catalog_version`` the denormalized
    metric/mask columns were copied from (nothing else in the fingerprint covers
    those). ``centroids_md5`` / ``scaler_md5`` are redundant on the hash (the
    selection list already pins the labels) but are recorded so ``corpus_meta.json``
    documents the full Phase-D provenance chain for the model card.
    """
    return {
        "schema_version": config.MANIFEST_SCHEMA_VERSION,
        "tile_px": config.TILE_PX,
        "normalization": config.NORMALIZATION,
        "clip_bounds": [config.CLIP_MIN_M, config.CLIP_MAX_M],
        "degenerate_std_eps": config.DEGENERATE_STD_EPS_M,
        "land_zero_frac_drop": config.LAND_ZERO_FRAC_DROP,
        "geocell_size_deg": config.GEOCELL_SIZE_DEG,
        "selection_list_md5": selection_list_md5,
        "catalog_version": catalog_version,
        "centroids_md5": centroids_md5,
        "scaler_md5": scaler_md5,
    }


def write_corpus_meta(
    path: Path | str,
    *,
    rows: list[TileRecord],
    global_mean: float,
    global_std: float,
    snapshot_date: str,
    build_timestamp: str,
    code_version: str,
    selection_list_md5: str,
    catalog_version: str,
    centroids_md5: str,
    scaler_md5: str,
) -> dict:
    """Write ``corpus_meta.json`` — the corpus card + reproducibility fingerprint.

    Returns the meta dict as written (handy for logging). ``global_mean`` /
    ``global_std`` are the corpus-wide land-pixel stats the pipeline accumulates,
    consumed by the loader's ``global`` normalization mode (ADR-0003 §3). The four
    Phase-D fingerprints feed ``build_params`` (see its docstring) and so ride into
    both the ``dataset_version`` hash and the documented provenance chain.
    """
    params = build_params(
        selection_list_md5=selection_list_md5,
        catalog_version=catalog_version,
        centroids_md5=centroids_md5,
        scaler_md5=scaler_md5,
    )
    meta = {
        "normalization": config.NORMALIZATION,
        "tile_px": config.TILE_PX,
        "storage_dtype": "int16",
        "clip_bounds": [config.CLIP_MIN_M, config.CLIP_MAX_M],
        "degenerate_std_eps": config.DEGENERATE_STD_EPS_M,
        "n_tiles": len(rows),
        "global_mean": global_mean,
        "global_std": global_std,
        "snapshot_date": snapshot_date,
        "build_timestamp": build_timestamp,
        "code_version": code_version,
        "build_params": params,
        "dataset_version": dataset_version(rows, params),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(meta, indent=2, sort_keys=True))
    return meta
