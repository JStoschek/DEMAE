"""Constants already settled by the plan or while charting the Phase-0 map.

Only values that are *decided* live here. Anything still an open ticket (tiling
geometry, TRI thresholds, storage format, manifest schema) is intentionally
absent — it belongs in the ticket that resolves it, not in a premature constant.
"""

from __future__ import annotations

# --- Source corpus (decided: Copernicus GLO-30 only for Phase 0) --------------
# Public AWS bucket, no credentials required. Cloud-Optimized GeoTIFFs (COGs).
COPERNICUS_GLO30_BUCKET = "s3://copernicus-dem-30m"
COPERNICUS_GLO30_RES_M = 30.0  # nominal ground sample distance, metres (1 arc-second)

# --- Tile geometry (decided: MAE-native 224x224 @ patch 16) -------------------
TILE_PX = 224
PATCH_PX = 16
TOKENS_PER_TILE = (TILE_PX // PATCH_PX) ** 2  # 196

# --- Channels (decided: elevation-only now, channel-ready later) --------------
# Phase 0 emits a single elevation channel. Derived channels (slope/curvature)
# are a Phase-3 ablation; the pipeline is built to add them via config, not code.
INPUT_CHANNELS = ("elevation",)

# --- Normalization (decided: per-tile standardization is the default) ---------
# Subtract tile mean, divide by tile std; store mean+std so reconstruction
# metrics can be inverted. Clipping percentile / std~=0 handling are still open
# (resolved by "Decide the corpus-construction recipe").
NORMALIZATION = "per_tile_standardize"

# --- Corpus tiers -------------------------------------------------------------
# Phase 0 ships only the debug tier. Smoke tier size is still fog on the map.
DEBUG_CORPUS_TILES = 50_000
