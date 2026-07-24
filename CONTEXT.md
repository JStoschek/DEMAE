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
- **Quality gates** — the predicates over the candidate catalog's stored fractions: four
  that **drop** (`zero_fill`, `degenerate`, `water`, `edited`) and two that only **tag**
  (`water_affected`, the `coastline` stratum). Thresholds locked by measurement in
  [`gate-attrition.md`](docs/research/gate-attrition.md); applied by `dataset.gates` as
  **boolean columns appended to the frame** — a *gated catalog* is the same rows plus a
  `keep` column, never a smaller frame. Re-running the gates after a threshold moves
  costs seconds; re-scoring would cost a six-hour global pass.
- **`masks_missing`** — the guard beside those gates: a window whose EDM/FLM/WBM
  fractions could not be read has NaN there, and `NaN > 0.30` is false, so it would
  clear the water and edited caps *unevaluated*. Such windows are flagged and excluded
  from `keep` — unknown quality is not passing quality. Zero windows in the shipped
  catalog; it exists so a future scoring run cannot seed the frame silently.
- **Stratum** (`stratum_id`) — one of the **k = 20** k-means cells the kept catalog is
  partitioned into, in the z-scored space of the frozen metric vector. It is a **label, not
  a rank**: unlike the TRI-era `stratum` (a ruggedness quintile) it carries no ordering, and
  the two columns are not comparable — which is why [ADR-0002 §3a](docs/adr/0002-storage-format-and-manifest.md)
  renamed rather than reused it. k=20 is set at the one measured boundary, the seed-stability
  cliff at k=20→30; the descriptor space is a **continuum, not clumps**, so no interior
  optimum exists to find, and *distinguishability between* adjacent strata is explicitly
  disavowed. A stratum is **terrain-meaningful** — the Gate E gallery judged all 20 coherent.
  Frozen by [ADR-0005](docs/adr/0005-sampling-design.md).
- **Selection list** — the committed list of `window_id, stratum_id, provenance` triples the
  sampler emits: the windows chosen for a tier, *before* any of them is read at full
  resolution. It is **the determinism contract** — not the seed, and not the sampler's code
  path. Phase E consumes the file rather than regenerating it, because multithreaded Lloyd is
  not reproducible even at a fixed seed. Distinct from the **manifest**, which indexes what
  actually survived materialization ([ADR-0005 §5](docs/adr/0005-sampling-design.md#5-determinism--the-persisted-artifact-is-the-contract-never-the-seed)).
- **Neyman allocation** — the rule splitting a tier's budget across strata, `n_h ∝ N_h·σ_h`
  (stratum size × mean distance to its centroid), apportioned to the exact budget by largest
  remainder. It deliberately over-samples rare, varied terrain — a **1.8× per-row premium** —
  which is precisely why the corpus is **terrain-diverse, not geo-uniform**: per-region counts
  read as representativeness will mislead. The only addition is a within-stratum **coastal
  frame-rate floor** at 1.337%, the rate at which coastal windows occur in the frame.
- **FSCS** — feature-space coverage sampling: the within-stratum selector. Sub-k-means at
  `k = quota`, taking the window nearest each sub-centroid, so a stratum's picks *cover* its
  descriptor space instead of sampling it randomly. Ties resolve by a **10° geo-cell
  round-robin that falls through** to the next-nearest window — without the fall-through the
  quota under-fills, because two sub-centroids in one dense pocket resolve to the same row.
- **Per-tile standardization** — the default normalization: subtract the tile
  mean, divide by the tile std, so the model learns *terrain shape* invariant to
  absolute altitude. Mean+std are stored so reconstructions can be inverted.
- **Manifest** — the per-tile index: one row per tile **actually on disk**, holding the
  storage join (`shard_id`, `row_index`), the per-tile land `mean`/`std`, the geography,
  the `stratum_id` and `provenance` copied from the selection list, and the frozen metric
  vector and mask fractions **denormalized in** from the gated catalog. The stratum's
  metric basis was the single TRI axis in
  [ADR-0001](docs/adr/0001-corpus-construction-recipe.md); it is now the **metric
  vector** ([ADR-0004](docs/adr/0004-frozen-metric-vector.md)), and the v2 schema
  ([ADR-0002 §3a](docs/adr/0002-storage-format-and-manifest.md)) follows Phase D's
  allocation design, now recorded as [ADR-0005](docs/adr/0005-sampling-design.md) — the
  column is `stratum_id` (a k=20 label), not the TRI-era `stratum` (a ruggedness
  quintile), and `tri` is gone. Denormalizing makes the manifest
  **self-contained**: reading the corpus needs no second file. **Region is recorded** so a
  geographic hold-out split is possible in Phase 2 (the split itself is not a Phase-0
  decision).
- **Materialization** — the pass that turns the selection list into the corpus: read each
  selected window at full resolution, screen it, store it as int16, index it. Run by
  `dataset.materialize` (Phase E1), which is the only stage that reads full-res pixels in
  bulk. Distinct from **scoring** (`dataset.catalog`), which measures every window in the
  world but stores none.
- **Attrition contract** — what happens when a *selected* window fails at full res.
  Settled as **`accept_shortfall`**: the failure is recorded and **never backfilled**, so
  the corpus is "50,000 selected, N materialized" rather than 50,000 by construction.
  Backfilling was rejected because it would write window ids absent from the committed
  selection list — and that list, not the seed, is the determinism contract. The string
  is recorded in `corpus_meta.json` because Gate E's pre-registered bar branches on it.
- **Materialization ledger** — one row per **selected** window (all 50,000) recording what
  became of it: `ok`, or the cause it was dropped (`zero_fill`, `no_land`, `degenerate`,
  `read_error`). The manifest holds the survivors; the ledger holds the account. It is
  **committed** because, unlike the shards and the manifest, it cannot be re-derived from
  committed inputs — knowing which windows failed requires the full-res read. Its
  `status == 'ok'` set is exactly the manifest's tile set, by asserted invariant.
- **Pre-registered bar** — Gate E's pass/fail criteria, committed to
  [`e2-gate-bar.json`](docs/research/e2-gate-bar.json) *before* the corpus, the E2 report or
  the gallery existed, so the review cannot rationalize whatever the numbers turn out to be.
  Two failure classes, pinned per criterion: a **Defect** (the corpus does not implement its
  own committed inputs — fix, re-run, stay in the map) versus a **Finding** (the corpus is
  faithful and the *design* is wrong — record, stop, fresh map on the implicated knob).
  `dataset.validate` carries each registered number as a module constant beside the bar
  phrase it was registered in, and **refuses to load a bar whose prose no longer states
  it** — so a threshold cannot move in the committed file without the report failing
  loudly. Choosing a denominator, population, or tolerance *after* seeing which choice
  fires is the peeking the whole arrangement exists to prevent; where such a choice is
  genuinely open, the alternative is published as a sensitivity, never as a second verdict.
- **Would-be stratum** — the stratum a *gate-dropped* window would have joined, obtained by
  scoring it against the committed `centroids.json` + `scaler.json`. k-means was fit on
  survivors only, so dropped rows carry full metrics and no `stratum_id`; retroactive
  assignment is what makes a per-(gate × stratum) attrition table possible at all (E2.5).
  Exact for kept rows too, since `centroids.assign` is the same pure argmin the sampler
  labelled them with. A window with a NaN metric — no land to measure — has no would-be
  stratum and is counted outside the table rather than imputed into a terrain family.
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
