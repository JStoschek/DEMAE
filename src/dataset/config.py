"""Locked Phase-0 constants — decided by the plan or the accepted ADRs.

Every value here is *settled*. ADR 0001 (corpus-construction recipe) and ADR 0002
(storage format + manifest) fixed the numbers that ADR-era stubs left open; they
are gathered here so the pipeline reads its recipe from one place and a reviewer
can diff behaviour against the ADRs line by line.

References are cited inline as ``ADR-000N §M`` so each constant traces to its decision.
"""

from __future__ import annotations

from typing import Final

# --- Source corpus (ADR 0001; recon docs/research/copernicus-glo30-recon.md) --
# Public AWS bucket, no credentials required. Cloud-Optimized GeoTIFFs (COGs),
# 1x1-degree cells, EPSG:4326, float32 metres over the EGM2008 geoid.
COPERNICUS_GLO30_BUCKET: Final = "copernicus-dem-30m"  # bare name (boto3 / s3://)
COPERNICUS_GLO30_RES_M: Final = 30.0  # nominal N-S ground sample distance, metres
# Anonymous access: GDAL/rasterio need this in the environment (recon §1).
AWS_NO_SIGN_ENV: Final = {"AWS_NO_SIGN_REQUEST": "YES"}

# --- Tile geometry (plan; MAE-native 224x224 @ patch 16) ----------------------
TILE_PX: Final = 224
PATCH_PX: Final = 16
TOKENS_PER_TILE: Final = (TILE_PX // PATCH_PX) ** 2  # 196

# --- Channels (ADR 0001; elevation-only now, channel-ready later) -------------
INPUT_CHANNELS: Final = ("elevation",)

# --- Land masking (ADR 0001 §3) -----------------------------------------------
# Ocean is Copernicus's exact-0.0 fill. A window is dropped iff its exact-zero
# fraction reaches this threshold (keeps coastlines, drops mostly-sea windows).
OCEAN_FILL_VALUE: Final = 0.0
LAND_ZERO_FRAC_DROP: Final = 0.75

# --- Overview downsample factor (ADR 0001 §4) ---------------------------------
# The x8 overview every window is scored on. Named for its TRI-era origin, but the
# factor itself is live: the ADR-0004 metric pass (catalog), the pilot frame, and
# the A3 calibration all still read their windows off the x8 overview.
TRI_OVERVIEW_FACTOR: Final = 8

# --- Geographic spread (ADR 0001 §5) ------------------------------------------
GEOCELL_SIZE_DEG: Final = 10  # 10x10-degree lon/lat cell for round-robin spread
# Latitude bands where Copernicus changes longitude sampling (recon §6, ADR-0002
# lat_band). Upper bound is exclusive; the band label is "<lo>-<hi>".
LAT_BAND_EDGES: Final = (0, 50, 60, 70, 80, 85, 90)

# --- Normalization / clipping (ADR 0001 §6, ADR 0002 §2) ----------------------
# Absolute plausibility clip only (NO percentile clip); land-only stats; drop
# degenerate near-constant windows. ADR-0002 tightened the degenerate epsilon to
# >=3 m so int16-metre quantisation cannot band a kept tile.
NORMALIZATION: Final = "per_tile_standardize"
CLIP_MIN_M: Final = -500.0
CLIP_MAX_M: Final = 9000.0
DEGENERATE_STD_EPS_M: Final = 3.0

# --- Storage format (ADR 0002 §1-2) -------------------------------------------
# Sharded memmap NPY of raw int16 metres (single channel), normalization deferred
# to the loader. Shards target ~1-2 GB; one int16 224x224 tile is ~100 KB, so a
# ~1.5 GB shard holds ~15k tiles.
import numpy as _np

STORAGE_DTYPE: Final = _np.int16
_TILE_BYTES: Final = TILE_PX * TILE_PX * 2  # int16 -> 2 bytes/px
SHARD_TARGET_BYTES: Final = 1_500_000_000
ROWS_PER_SHARD: Final = SHARD_TARGET_BYTES // _TILE_BYTES  # ~14,946

# --- Corpus tiers & quotas ----------------------------------------------------
# The debug tier is 50,000 windows. Stratification, allocation and selection are
# ADR-0004's metric-basis sampler (dataset.frame / dataset.centroids /
# dataset.select), not ADR-0001's TRI quintiles: the budget is an argument to
# dataset.select, the per-stratum quotas come from the committed Neyman table
# (docs/research/d2-allocation-nh.json), and the per-COG cap was retired as inert
# (#42). The three TRI-era constants that lived here — N_STRATA, PER_STRATUM_QUOTA,
# PER_TILE_WINDOW_CAP — went with dataset.stratify.

# --- Phase-D handoff (issue #44) ----------------------------------------------
# The committed selection list: the 50,000 windows the debug tier materializes, and
# *the* determinism contract — not the seed (#43/#44). `dataset.select` writes it and
# `dataset.materialize` reads it, so the path is named once here rather than in both.
SELECTION_PATH: Final = "artifacts/phase_d/selection_debug_50k.csv"

# --- Materialization attrition (issue #50) ------------------------------------
# What happens when a *selected* window fails at full res. `accept_shortfall`: it is
# recorded in the materialization ledger and never backfilled, so the corpus is
# "50,000 selected, N materialized". Backfill was rejected because it would write
# window_ids absent from the committed selection list — and that list, not the seed,
# is the determinism contract (#43/#44). The string rides into corpus_meta.json
# because #47's Gate-E bar reads it to pick which pre-committed branch to evaluate.
ATTRITION_CONTRACT: Final = "accept_shortfall"

# --- Global normalization statistic (issue #70, ADR-0003 §3 Amendment 1) -------
# Which corpus ADR-0003's `global` normalization mode reads its mean/std from.
# `tier_specific`: every tier accumulates its own land-pixel mu/sigma from its own
# tiles, so the per-tile-vs-global ablation is a *within-tier* comparison only —
# never compare a global-mode run across corpora, the denominators differ. The
# rejected alternative was `inherit_debug` (freeze the 50k tier's values project-
# wide); it buys a cross-tier comparison ADR-0005 §6 already disavowed, at the
# price of normalizing each corpus against a statistic taken from a different one.
# Recorded top-level in corpus_meta.json and *outside* build_params — same slot and
# same reason as ATTRITION_CONTRACT above: #69's Gate-F bar reads it to pick which
# pre-committed F8 branch to evaluate, and adding a build_params key would change
# the debug tier's dataset_version, which four committed Gate-E artifacts quote.
NORMALIZATION_STAT_SOURCE: Final = "tier_specific"

# --- Quality masks (docs/research/copernicus-mask-spec.md; masks.py) ----------
# EDM/FLM/WBM ship in each tile's own AUXFILES/ prefix in the DEM bucket above.
# The cache-locally-not-stream policy (and why it differs from the COG reads) is
# argued in dataset.masks; these are just the knobs it reads.
MASK_LAYERS: Final = ("EDM", "FLM", "WBM")
MASK_CACHE_DIR: Final = "data/mask_cache"

# --- Reproducibility (ADR 0001 §Consequences, ADR 0002 §3) --------------------
RANDOM_SEED: Final = 0xDE_A_E  # fixed seed for all tie-breaking / round-robin
MANIFEST_SCHEMA_VERSION: Final = 2  # v2: metric-vector schema (issue #48, ADR-0002 §3a)
