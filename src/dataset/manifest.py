"""Tile manifest (Parquet) + corpus card (``corpus_meta.json``), ADR-0002 §3.

A single typed Parquet file indexes every tile — the loader loads it whole and
joins to the shards by ``(shard_id, row_index)``. A JSON sidecar holds the
corpus-level constants, the ADR-0003 corpus-wide land ``mean``/``std`` (so the
loader's ``global`` normalization mode works), and a ``dataset_version``
fingerprint that pins this exact build for reproducibility and the model card.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from . import config

# Typed per-row schema (ADR-0002 §3). Order here defines column order on disk.
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
        ("tri", pa.float64()),
        ("stratum", pa.int32()),
        ("mean", pa.float64()),
        ("std", pa.float64()),
        ("land_frac", pa.float64()),
        ("content_hash", pa.uint64()),
        ("split", pa.string()),  # null in Phase 0; Phase-2 fills the hold-out
    ]
)


@dataclass(frozen=True)
class TileRecord:
    """One manifest row — the full provenance + storage join for a tile."""

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
    tri: float
    stratum: int
    mean: float
    std: float
    land_frac: float
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


def build_params() -> dict:
    """The recipe parameters that define a corpus build (feed the fingerprint)."""
    return {
        "schema_version": config.MANIFEST_SCHEMA_VERSION,
        "tile_px": config.TILE_PX,
        "normalization": config.NORMALIZATION,
        "clip_bounds": [config.CLIP_MIN_M, config.CLIP_MAX_M],
        "degenerate_std_eps": config.DEGENERATE_STD_EPS_M,
        "land_zero_frac_drop": config.LAND_ZERO_FRAC_DROP,
        "n_strata": config.N_STRATA,
        "per_stratum_quota": config.PER_STRATUM_QUOTA,
        "per_tile_window_cap": config.PER_TILE_WINDOW_CAP,
        "tri_overview_factor": config.TRI_OVERVIEW_FACTOR,
        "geocell_size_deg": config.GEOCELL_SIZE_DEG,
        "seed": config.RANDOM_SEED,
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
) -> dict:
    """Write ``corpus_meta.json`` — the corpus card + reproducibility fingerprint.

    Returns the meta dict as written (handy for logging). ``global_mean`` /
    ``global_std`` are the corpus-wide land-pixel stats the pipeline accumulates,
    consumed by the loader's ``global`` normalization mode (ADR-0003 §3).
    """
    params = build_params()
    meta = {
        "normalization": config.NORMALIZATION,
        "tile_px": config.TILE_PX,
        "storage_dtype": "int16",
        "clip_bounds": [config.CLIP_MIN_M, config.CLIP_MAX_M],
        "degenerate_std_eps": config.DEGENERATE_STD_EPS_M,
        "per_stratum_quota": config.PER_STRATUM_QUOTA,
        "n_strata": config.N_STRATA,
        "seed": config.RANDOM_SEED,
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
