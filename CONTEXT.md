# DEMAE — Context

DEMAE is an open-weights **ViT masked-autoencoder** pretrained self-supervised on
global **Copernicus DEM GLO-30** elevation, released with a small reproducible
fine-tuning benchmark. Full framing, literature positioning, and phased plan live
in [`DEMAE_Project_Plan.md`](DEMAE_Project_Plan.md).

This file is the project's **ubiquitous language** — the vocabulary every issue,
module, and ADR should use. It grows lazily as terms and decisions get resolved
(see `docs/agents/domain.md`). Phase-0 work is charted on the wayfinder map,
GitHub issue #1.

## Environment

- **Package/venv manager:** `uv`. `uv sync` builds the venv from `uv.lock`;
  `uv run <cmd>` runs inside it. Python pinned to 3.12 (`.python-version`).
- **Geospatial stack:** `rasterio` (with **bundled GDAL** via wheels — no system
  GDAL needed), `numpy`, `pyproj`, `boto3`. A Dockerfile pinning this is the
  closing Phase-0 artifact, so the rented vast.ai box reproduces the local run.

## Glossary

- **DEM** — Digital Elevation Model; a raster of ground elevation. DEMAE's sole
  pretraining modality (elevation-only).
- **Copernicus GLO-30** — the primary corpus: global 30 m DEM, public AWS bucket
  `s3://copernicus-dem-30m`, Cloud-Optimized GeoTIFFs, 1°×1° tiles. Phase-0's
  only data source.
- **COG** — Cloud-Optimized GeoTIFF; supports **windowed reads** (fetch a pixel
  window without downloading the whole file), which is how tiles are sampled.
- **Tile** — a 224×224-pixel elevation crop (≈6.7 km across at 30 m), the unit of
  pretraining. 196 patches of 16 px. *(Distinct from a "source tile" = a 1°×1°
  Copernicus COG; when ambiguous, say "COG" for the source and "tile" for the crop.)*
- **Window** — a 224×224 pixel region read from a COG before it becomes a tile.
- **Land mask** — dropping ocean / zero-elevation areas so the model doesn't waste
  capacity reconstructing flat sea.
- **TRI** — Terrain Ruggedness Index; a cheap per-window ruggedness score. Phase 0
  ([ADR-0001](docs/adr/0001-corpus-construction-recipe.md) §4) used it as the *single*
  stratification axis; **[ADR-0004](docs/adr/0004-frozen-metric-vector.md) supersedes
  that** — stratification now runs on the frozen **metric vector** (below), not TRI.
- **Metric vector** — the frozen 6-metric per-window descriptor the corpus stratifies
  on: `{slope_mean, local_relief, texture, elevation_relief_ratio, mean_elevation,
  geomorphon_entropy}` (log-transform `slope_mean`, `local_relief`). Chosen by
  measurement — a Spearman correlation prune over a 15,000-window pilot frame — and
  frozen at Gate B ([ADR-0004](docs/adr/0004-frozen-metric-vector.md)). A terrain-diverse
  corpus beats a geographically uniform one; the vector is how "diverse" is measured.
- **Candidate catalog** — the raw, **ungated** frame holding one row per window of
  every COG: the frozen metric vector, `exact_zero_frac`, `land_std`, the EDM/FLM/WBM
  fractions, and the geography keys. Built once by `dataset.catalog` (Phase C) and
  reused by every corpus tier without re-scoring, so it is the project's *sampling
  frame* — distinct from the **manifest**, which indexes the tiles actually selected
  and materialized. Every fraction is stored raw: **quality gates are filter columns
  over the catalog, never drops during scoring**, so a threshold can be re-tuned
  without touching S3.
- **Per-tile standardization** — the default normalization: subtract the tile
  mean, divide by the tile std, so the model learns *terrain shape* invariant to
  absolute altitude. Mean+std are stored so reconstructions can be inverted.
- **Manifest** — the per-tile index (id, source COG, lon/lat + region, stratum,
  mean+std). The stratum's metric basis was the single TRI axis in
  [ADR-0001](docs/adr/0001-corpus-construction-recipe.md); it is now the **metric
  vector** ([ADR-0004](docs/adr/0004-frozen-metric-vector.md)), and the final manifest
  schema follows Phase D's allocation design. **Region is recorded** so a geographic
  hold-out split is possible in Phase 2 (the split itself is not a Phase-0 decision).
- **Shard / storage format** — how normalized tiles are persisted for the Phase-1
  dataloader; chosen for streaming-read throughput (the plan's #1 risk: I/O
  starving the GPU).
- **Corpus tiers** — **smoke** (a few hundred tiles, end-to-end pipeline check) →
  **debug** (~50k, the Phase-0 deliverable) → **scale** (200–500k, Phase 3).
- **Channel-ready** — the pipeline emits a single elevation channel now but is
  architected so slope/curvature channels are a Phase-3 config flag, not a rewrite.
- **GSD** — Ground Sample Distance (metres/pixel); 30 m for GLO-30. Multi-resolution
  / GSD-aware encoding is Phase 4, out of scope now.

## Decisions

Architectural decisions are recorded as ADRs under [`docs/adr/`](docs/adr/), created
lazily as wayfinder decision tickets resolve. If your work contradicts an ADR,
surface it rather than silently overriding.
