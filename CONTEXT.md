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
  Two are committed and **neither contains the other** — the debug tier's
  `artifacts/phase_d/selection_debug_50k.csv` and the scale tier's
  `artifacts/phase_f/selection_scale_500k.csv` are *independent draws* at two budgets
  ([#45](https://github.com/JStoschek/DEMAE/issues/45)); their 30,998-window overlap is
  incidental, not a nesting rule ([`f-scale-draw.md`](docs/research/f-scale-draw.md)).
- **Split assignment** — the committed `tile_id, split` list that fills the manifest's
  `split` column, plus the block lists it was drawn as
  (`artifacts/phase1/split_scale_500k.csv` + `split_blocks_500k.json`,
  [#77](https://github.com/JStoschek/DEMAE/issues/77)). Third artifact in the same
  contract family as the selection list and the ledger: **the file is what later phases
  re-read**, not this code re-run. Drawn once per tier and **never randomly** — the corpus
  card's §4.6 autocorrelation warning is binding — as `test` (whole 10° geocells, the
  Phase-2 evaluation regions, untouched by pretraining), `val` (whole source COGs from
  non-test cells) and `train` (everything else). A **block** is whatever unit is held out
  whole — a geocell for `test`, a COG for `val` — and the unit is the honesty claim: the
  corpus card allows nothing finer than a COG, because two windows from one COG are
  neighbours, not independent samples. Both hold-outs are chosen to minimize
  **relative stratum shortfall** against the corpus composition, a scale-free measure that
  prices a *missing* terrain family at a whole unit however rare it is — which is why
  total-variation distance is reported but never optimized. Filling the column does not
  move `dataset_version`: that hashes `(tile_id, content_hash)` and `build_params` alone.
  What the blocking does and does not buy is measured, not asserted
  ([`p77-split.md`](docs/research/p77-split.md) §3).
- **Split containment** — the invariant a Phase-1 Dataset asserts **at construction**: no tile
  of one split sits inside a block held out for another — no train tile in a `test` geocell, none
  in a `val` COG ([#89](https://github.com/JStoschek/DEMAE/issues/89), plan §15.2). Checked
  against the union of two block lists, because neither subsumes the other: the blocks *derived*
  from the manifest's own `split` column (which catches a relabelled tile inside a still-held-out
  region) and the *committed* block lists (which catch a whole test geocell relabelled `train`,
  where nothing left in the column looks held out). It fails loudly rather than warning: a leak is
  invisible in the loss, the recon grids and the probe, and surfaces only in Phase 2 as an
  evaluation on regions the encoder has already seen — unrepairable without re-pretraining.
- **Headroom** (`N_h / n_h`) — kept windows a stratum holds per window it must yield. The
  one quantity the *budget* moves: Neyman shares are budget-invariant, so ten times the
  budget is a ten times shallower pool. Across all 20 strata it runs 43.9–79.0× at the
  debug tier and **4.39–7.90×** at the scale tier; over the four flat strata (2/5/7/17)
  [ADR-0007](docs/adr/0007-accept-degenerate-flatness-gate-flat-strata.md) leg 3 leans on,
  its **≥53×** becomes **≥5.35×**. Attrition *rates* reproduce across tiers by
  construction; headroom does not, which is what that ADR's Phase-F revisit trigger on
  strata 2/17 is written against.
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
- **Realized share** (`r_h`) and **exhaustion** (`e_h`) — the pair that says *why* a terrain
  family came out thin, and they must be read together. `r_h` is the share of a stratum's
  **shipped** tiles that belong to the family; `e_h` is the share of the **surviving pool** of
  that family the draw actually consumed. `r_h` near the kept-pool proportional share means the
  sampler drew proportionally and declined an available tail — a **sampler** limit. `r_h` near
  `e_h`'s ceiling means the survivors ran out — a **gate** limit. The distinction is not
  cosmetic: it decides which knob a Finding sends to a fresh map, and at Gate F it inverted the
  obvious answer — `F3` fired on the flatness gate's revisit trigger, but `e_h` of 0.185 / 0.213
  showed the gate was **not** the binding constraint
  ([#74](https://github.com/JStoschek/DEMAE/issues/74),
  [`f-gate-verdict.md`](docs/research/f-gate-verdict.md)). Note they are not independent:
  `r_h = e_h · tail_h / n_h`, which is why a bar conjoining "deficient **and** exhausted" is
  unsatisfiable once `n_h` approaches `tail_h`, and why the decomposition is *published*
  rather than gated.
- **Per-tile standardization** — the default normalization: subtract the tile
  mean, divide by the tile std, so the model learns *terrain shape* invariant to
  absolute altitude. Mean+std are stored so reconstructions can be inverted.
- **Global normalization** — the *other* arm of the plan's normalization ablation
  ([ADR-0003 §3](docs/adr/0003-dataloader-design.md)): `(x − μ)/σ` against one corpus-wide
  land-pixel pair, so absolute altitude is **kept** rather than discarded. The statistic is
  **tier-specific** — each tier accumulates its own μ/σ from its own tiles, recorded as
  `normalization_stat_source: "tier_specific"` top-level in `corpus_meta.json` (outside
  `build_params`, so it never moves `dataset_version`). Two consequences worth knowing before
  Phase 1 spends a run: the ablation is a **within-tier** comparison only — two tiers normalize
  against different constants — and the mode puts roughly **42×** more variance in between-tile
  altitude than in within-tile relief, so a normalized tile with `std ≈ 0.02` is correct
  behaviour, not a loader bug ([ADR-0003 Amendment 1](docs/adr/0003-dataloader-design.md#amendments)).
  **Third consequence, for whoever runs the ablation:** the shipped scale-tier pair
  (955.67 / 1012.16 m) was accumulated over *every* tile, including the 4.98% of land pixels
  the **split assignment** later held out, so it is contaminated by `val` and `test`.
  Measured rather than bounded ([`p77-split.md`](docs/research/p77-split.md) §5): the
  train-only pair is **952.75 / 1013.43 m**, a −0.31% / +0.13% shift. `corpus_meta.json`
  keeps the shipped pair — moving it would move a number four committed artifacts quote for
  a third of a percent — so a train-only ablation should read the train-only pair from
  `p77-split.json` and say which one it used.
- **Corpus card** — the project's one artifact written for a reader **outside** it:
  [`f-honesty-notes.md`](docs/research/f-honesty-notes.md), the plan's E3 honesty notes
  recorded against the shipped 500 k tier. Not a ticket report — it is organized by *what a
  consumer could get wrong* rather than by what Phase F measured, and a Phase-4 model card is
  meant to lift from it directly. Its numbers are **derived, never typed**: the driver
  re-computes them from committed artifacts into `f-honesty-notes.json` and the prose quotes
  that, because a stale number in a card is one no downstream reader can catch. Caveats with
  no measurable size (the withheld-country gap) are pinned as explicit nulls with a stated
  reason, since an absent entry and a forgotten one look identical.
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
  geographic hold-out split is possible (the split itself is not a Phase-0 decision; the
  scale tier's `split` column was filled in Phase 1 by
  [#77](https://github.com/JStoschek/DEMAE/issues/77), the debug tier's is still null).
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
- **Materialization ledger** — one row per **selected** window recording what became of it:
  `ok`, or the cause it was dropped (`zero_fill`, `no_land`, `degenerate`, `read_error`).
  The manifest holds the survivors; the ledger holds the account. It is **committed**
  because, unlike the shards and the manifest, it cannot be re-derived from committed
  inputs — knowing which windows failed requires the full-res read. Its `status == 'ok'`
  set is exactly the manifest's tile set, by asserted invariant. **One per tier, and they
  are not interchangeable** — `artifacts/phase_e/materialization_ledger.parquet` books the
  debug tier's 50,000 and `artifacts/phase_f/`'s books the scale tier's 500,000. Because
  neither is re-derivable, writing one tier's run into the other's ledger is unrecoverable;
  `dataset.materialize.Tier` is what makes a tier's three paths move as a set, and
  `_assert_tier_paths_agree` refuses a run that mixes two tiers' artifacts.
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
  **debug** (50,000 selected / 49,998 materialized, the Phase-0 deliverable) → **scale**
  (500,000 selected / 499,934 materialized, `data/corpus_500k`, built by
  [#72](https://github.com/JStoschek/DEMAE/issues/72)). Both realized tiers are the *same*
  `dataset.materialize` driver over two independent selection lists — not a nesting
  ([#45](https://github.com/JStoschek/DEMAE/issues/45)) — and each carries its own ledger,
  corpus card and `dataset_version`. "smoke" is a way of running the driver
  (`--limit-cogs` into a scratch directory), not a third shipped tier.
- **Run config** — the frozen object holding every Phase-1 recipe knob from plan §2-§8 at its
  cited default (`pretrain.config.RunConfig`, [#89](https://github.com/JStoschek/DEMAE/issues/89)).
  Every run R0-R5 *is this object with a knob moved*, which is what makes an ablation a diff:
  nothing downstream reads a recipe number from anywhere else. It round-trips to a plain dict —
  a partial dict fills the rest from the plan, so an ablation's config file names only what it
  changes — and hashes to a **config hash**, the W&B tag plan §15.7 asks for. That hash covers
  the *recipe* and deliberately not the four **environment** fields — the run's label and the
  three paths saying where the corpus and the split artifacts are mounted: §11 resumes on a
  fresh vast.ai instance where the corpus lives at another path, and a hash that moved with the
  mount point would read every resume as a config mismatch. `init.checkpoint` is a path and is
  *not* environment: which checkpoint a continual arm warm-starts from is the arm.
- **Data fingerprint** — the block every checkpoint and every W&B run carries, saying *which
  corpus, which selection, which split, which code*: `dataset_version`, the selection-list md5
  (quoted from `corpus_meta.json`, since that is the value the corpus was actually built from),
  the **split md5**, and the git SHA. All four are content hashes of committed artifacts, never
  descriptions. A split that is not there is recorded as an explicit **absent** *with a stated
  reason* rather than raised — the fingerprint shipped before the split did — and "committed"
  requires both halves: the assignment artifact exists *and* the manifest column is filled on
  every row. A partly filled column is the dangerous shape and reads as absent.
- **Visible tokens** — the 49 of 196 patches the encoder actually sees at 75% masking
  (`pretrain.model`, [#90](https://github.com/JStoschek/DEMAE/issues/90)). This is the ~4x
  pretraining discount the whole Phase-1 budget rests on, and it is why the **mask count**,
  not just the ratio, is fixed at model construction: a count that varied per batch would make
  every tensor shape in the model data-dependent and defeat `torch.compile(dynamic=False)`.
  Masking lives in the *model*, never in the loader (ADR-0003), is drawn per sample and afresh
  every step, and takes an optional generator so the val loss can pin its mask and stay
  comparable across epochs. The ratio itself is a config knob — R2 ablates 90%.
- **Channel-ready** — the pipeline emits a single elevation channel now but is
  architected so slope/curvature channels are a Phase-3 config flag, not a rewrite.
- **GSD** — Ground Sample Distance (metres/pixel); 30 m for GLO-30. Multi-resolution
  / GSD-aware encoding is Phase 4, out of scope now.

## Decisions

Architectural decisions are recorded as ADRs under [`docs/adr/`](docs/adr/), created
lazily as wayfinder decision tickets resolve. If your work contradicts an ADR,
surface it rather than silently overriding.
