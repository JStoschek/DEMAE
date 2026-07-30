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
  capacity reconstructing flat sea. Derived per tile *before* normalization (ADR-0003 §4) and
  carried beside the image in the sample dict — not static metadata: anything that moves the
  image geometrically has to move the mask the same way (see **Dihedral augmentation**).
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
- **Dihedral augmentation** — the whole of Phase 1's augmentation policy: one of the 8 elements
  of the dihedral group (four 90° rotations × {identity, mirror}) per sample, and nothing else
  (`pretrain.augment`, [#91](https://github.com/JStoschek/DEMAE/issues/91), plan §6). Overhead
  terrain has no canonical orientation, so the group is free label-preserving variety; **no**
  RandomResizedCrop (rescaling changes the effective GSD of a modality where scale is
  physically meaningful — scale augmentation waits for GSDPE in Phase 4) and **no** intensity
  jitter (harmful in MAE's own ablation, and meaningless on standardized elevation). Both
  omissions are deliberate deviations from the MAE default, so a config asking for them is
  refused rather than ignored. It is a **wrapper around** the read path, never an edit to it
  (ADR-0003), and the load-bearing property is that **one draw carries both the image and its
  land mask**: the two drifting apart does not crash and does not show in a loss curve, it
  simply points the ocean weighting at a rotated copy of where the ocean used to be.
  The element is a **pure function of `(run seed, epoch, sample index)`** — a keyed hash, not a
  draw from any generator — so it is **not an RNG stream** and there is nothing about it for a
  checkpoint to carry. That is what makes **exact resume** hold at *any* worker count: a
  sequential stream has a position, position is process-local, and at the shipped
  `num_workers = 8` it lived in worker processes the parent could neither read nor restore, so a
  resumed run silently redrew every dihedral element from the kill point on
  ([#104](https://github.com/JStoschek/DEMAE/issues/104) closed that by deleting the stream;
  [#94](https://github.com/JStoschek/DEMAE/issues/94) had documented it as a limitation). The
  epoch reaches a worker — which is handed sample indices and nothing else — on an `EpochIndex`,
  the `int` subclass `EpochShuffleSampler` yields; without it every epoch would apply the same
  transform to the same tile, which is a fixed relabelling of the corpus rather than
  augmentation.
- **norm_pix** — per-patch target normalization: the reconstruction target for a patch is that
  patch standardized by its **own** mean and variance, so the model is scored on internal relief
  rather than on where the patch sits within the tile (`pretrain.loss`,
  [#92](https://github.com/JStoschek/DEMAE/issues/92), plan §4). On elevation it is local
  detrending, and with `per_tile` input normalization on top of it the model neither sees nor
  reconstructs absolute altitude anywhere — the intended Phase-1 objective, terrain shape only.
  It is ON by default and **R3 ablates it OFF**, which is a genuinely open question rather than
  a formality, so the OFF path is characterized and not merely reachable.
- **Variance floor** — the lower bound on that per-patch variance before it is divided by, 1e-4
  in per-tile-standardized units against the reference implementation's 1e-6. It exists because
  the corpus's degenerate-flatness screen runs per **tile**, so a near-flat *patch* — playa,
  lake surface, residual ocean — survives curation, and dividing it by its own near-zero
  variance amplifies quantization noise into the loss. The floor bounds that at 100× and
  inverts the shape of the failure: below the floor, a flatter patch now scores *less*, so
  near-flat patches sink to the bottom of the per-patch loss distribution instead of dominating
  its top. The **fp32 guard** is the same concern from the other side — in bf16 a patch with
  genuine relief can measure as exactly flat — and it binds the caller too: a target arriving
  narrower than fp32 is refused, because no upcast returns a variance already rounded away.
  Both are config knobs, and §4's standing instruction is to read the per-patch distribution
  before concluding anything about the model.
- **Ocean weighting** — the pixel-level layer under the patch-level loss mask: ocean and void
  pixels carry **zero weight**, each patch renormalizes by its own **valid-pixel count**, and the
  `norm_pix` mean and variance compute over valid pixels only (`pretrain.loss`,
  [#96](https://github.com/JStoschek/DEMAE/issues/96), plan §4). Each of those has its own
  reason: without the renormalization a half-ocean patch is scored as if it were half as wrong as
  it is, and without valid-only statistics the "local relief" a patch is judged on is really the
  land/ocean step — the largest gradient in the tile and no part of terrain. The weights come from
  the loader's `land_mask` (`raw != 0`, derived before normalization, ADR-0003 §4). Precedent is
  thin and the honest framing is that the published pattern is **curation, not weighting**
  (MAESSTRO drops any patch containing a missing value); this corpus already curated hard in
  Phase 0, so this is a correctness patch on the residual rather than a novelty claim.
- **Minimum valid patch fraction** — a masked patch with under **25%** valid pixels leaves the
  loss entirely, rather than being weighted down. Below that there is too little terrain left to
  learn from and the renormalization amplifies whatever noise remains; an all-ocean patch
  therefore contributes exactly zero, not a NaN out of the 0/0. It is a config knob, and 0
  switches the threshold off without switching off the requirement that a scored patch contain
  something. Note what this makes the loss's returned per-patch `weight` grid: not just which
  patches were masked, but **how much of a batch the ocean took out of the objective**.
- **Image / mask drift** — the named silent failure of ocean weighting, and the reason
  `pretrain.loss` carries an alignment check rather than a comment. The weights and their targets
  reach the loss by two routes (`patchify(land_mask)` and `patchify(image)`) and are held in step
  only by the dihedral draw being used twice. Drift produces 0/1 weights, plausible per-patch
  valid counts, a descending loss and no crash — it simply scores a rearranged copy of where the
  ocean was. It is detectable because ocean is the *exact-zero* fill and both normalizers are
  affine, so every invalid pixel in a tile leaves normalization carrying one identical value; a
  mask that no longer selects exactly that set has drifted. The check is **off the hot path** (it
  syncs the device and breaks the compile graph) and belongs to the R0 pre-flight and the tests.
- **Continual init** — warm-starting R4's ViT-B encoder from an externally trained ImageNet MAE
  instead of from scratch (`pretrain.init`, [#102](https://github.com/JStoschek/DEMAE/issues/102),
  plan §8). It is **a config flag, not a code path**: `init.mode` is `scratch` for R1/R2/R3/R5 and
  `imagenet_mae` for R4, and the scratch runs go through the same call, which does nothing. The
  checkpoint is OpenMMLab's Apache-2.0 reproduction, not Meta's official release — [P101](docs/research/p101-mae-checkpoint-licence.md)
  found the official file's licence ambiguous in a way that cannot be resolved by asking, and an
  unresolvable ambiguity under a released artifact is the wrong kind of risk to carry. Three
  things happen: the RGB patch-embed kernel is collapsed to one channel by **summing** (not
  averaging — summing reproduces the 3-channel convolution exactly on a matched input, averaging
  reproduces a third of it, so the pretrained blocks downstream receive the activation scale they
  were trained on), the decoder is left fresh, and every encoder parameter is asserted to have
  been populated by name. No `strict=False` anywhere: a load that silently skips half the weights
  trains fine, descends, and produces an R4 that performs like R1 — which is the one question the
  arm exists to answer.
- **Layout convention** — the answer to "which tensor goes in which slot, and in what order",
  for a warm start. The rule, set by [#103](https://github.com/JStoschek/DEMAE/issues/103) and
  discharged by [P102](docs/research/p102-mmselfsup-mae-layout.md): **read it from the source
  implementation and write it down as a checked claim with a link — never infer it from tensor
  shapes.** "Their `qkv` is fused q,k,v, untransposed, per `<link>`" is checkable; "a (2304, 768)
  tensor fits a (2304, 768) slot" is not evidence of anything. The class of bug this guards is
  weights landing in right-shaped wrong slots: every name populates, every shape matches, every
  assertion passes, and the encoder computes garbage. It is not hypothetical — mmselfsup's fixed
  sin-cos positional embedding turned out to be the **row/column transpose** of MAE's, because
  `torch.meshgrid` defaults to `'ij'` and `np.meshgrid` to `'xy'`. From scratch the two orders are
  a harmless relabelling no test would surface; warm-starting, the weights learned one of them.
  Where an oracle exists it beats a claim — the loader reconstructs the checkpoint's own
  `pos_embed` and compares 151,296 floats before copying a single weight — and where one does
  not, the honest position is that a **behavioural** check has measured power and usually less
  than hoped (P102 §6.1: two of eight scrambles caught).
- **Micro-batch** — one batch as the DataLoader hands it over, and what a forward/backward
  actually runs on: `batch_size` samples, sized by what fits in 32 GB
  (`pretrain.train`, [#93](https://github.com/JStoschek/DEMAE/issues/93)). It is *not* the unit
  the recipe is written in — **effective batch** is — and the distinction is the whole of why
  gradient accumulation is not a recipe change.
- **Effective batch** — physical batch × gradient accumulation: `accum_iter` micro-batches
  backwarded at 1/N before one optimizer step, and the *only* batch number the recipe reads
  (plan §6). §6's linear scaling rule is `lr = blr × eff_batch / 256`, so 1024×2 (ViT-S) and
  512×4 (ViT-B) both land on **2048 → 1.2e-3** and the two runs stay comparable: the physical
  batch is what fits in 32 GB, the effective one is the recipe. The gradient of N accumulated
  micro-batches at 1/N *is* the gradient of one batch N× the size, which is asserted rather than
  assumed — and it is why a **short group at an epoch boundary is dropped rather than stepped
  on**: half an effective batch at the LR a whole one earns is a step outside the recipe.
- **No-decay group** — the AdamW parameter group at weight decay 0, holding exactly the biases
  and the normalization parameters (`pretrain.optim`, §6's "excluded on bias/norm params").
  Everything else, the two class tokens included, decays at 0.05 — they are `(1, 1, D)` and the
  reference rule is a dimensionality test, which is the one place "tokens are like biases" would
  quietly change the recipe. Getting the grouping wrong is invisible: it trains, it descends,
  and it produces a slightly worse encoder that reads as an unlucky run. The two readings of the
  rule — by module type, and the reference's `ndim <= 1 or .bias` — are both computed and must
  **agree**, so a future 1-D learnable scale (timm's LayerScale `gamma`) raises rather than
  landing in whichever group the code happened to pick.
- **Warmup and cosine** — the LR curve: linear from **0** over 20 epochs (5% of the 400-epoch
  schedule, matching MAE's 40/800), then cosine to **exactly 0** at the final epoch, evaluated at
  the fractional epoch so it moves every step rather than every epoch. Past the end it is clamped
  at 0, because `cos` climbs again after π and R5's 400→800 extension is a config change, not a
  loop that kept running. Like the no-decay group, every wrong version of this trains.
- **bf16 without a scaler** — the precision policy (plan §7.1): bf16 autocast over the encoder
  and decoder, and **no `GradScaler` anywhere**. The absence is the point — fp16 needs a scaler
  because its dynamic range cannot hold the gradients, and that machinery is the documented NaN
  failure mode of the official MAE repo; bf16 has fp32's exponent range, so there is nothing to
  scale. fp16 is refused by the config rather than offered. The loss and the `norm_pix`
  statistics stay fp32 regardless (see **Variance floor**), so what autocast actually buys is the
  matmuls.
- **Silent lever** — the shared failure mode of plan §7's efficiency levers, and the reason
  `pretrain.runtime` is mostly checks: **compile and SDPA fail by being absent**. A model whose
  attention fell back to the unfused kernel, or whose graph is recompiled every step, trains the
  *identical* model more slowly. No loss curve, gradient norm, checkpoint or reconstruction grid
  differs by a bit. The only evidence is a throughput number against a baseline nobody has, on a
  box rented by the hour — so every §7 lever is read off the **built modules** (`attention_path`,
  Dynamo's frame counter) and compared against the config that asked for it, before the first
  step rather than after the invoice. Distinct from the invisible-*recipe* class (**No-decay
  group**, **Warmup and cosine**), where the wrong version trains a slightly worse model: a
  silent lever changes nothing but the bill.
- **Loader wait fraction** — of one step's wall clock, the share spent **blocked inside the
  loader's `next`** (`pretrain.throughput`). Not the time the loader spent working: with 8
  workers prefetching, most of that happens during the previous step's backward pass and costs
  nothing. What is counted is the residual the prefetch failed to cover, which is the only part
  that makes a step slower — and §7.5's stop-and-fix threshold (~10%) is written about it,
  because the dataloader is the plan's **#1 named risk** (DailyMAE got 5.8× end to end from IO
  alone). Reported per step and per epoch, always by **summing seconds and samples and dividing
  once**: a mean of per-step rates flatters the shape every real run has, where step 1 pays for
  the compile and the loader's first fill.
- **Launch entrypoint** — `python -m pretrain.launch`, the one documented way to start a run
  ([#95](https://github.com/JStoschek/DEMAE/issues/95)). A run is *the plan's defaults plus the
  knobs it moves*, and the entrypoint is where that sentence becomes a command: a preset, an
  optional partial config file, the environment paths, then repeatable `--set group.knob=value`
  overrides — in that order, the general escape hatch last. A typo'd knob stops the run rather
  than defaulting, for the reason the config refuses unknown keys at all: a run that reports one
  recipe and trains another is worse than a run that fails. `--dry-run` assembles the whole
  thing and spends nothing, which is the form the §15 pre-flight checklist is walked in.
- **Exact resume** — the property plan §11 makes a *gate* rather than a nicety, and §15's
  checklist item 4 measures: a run killed at step k and restarted from its checkpoint produces
  a **bit-identical** loss trajectory to the uninterrupted one. vast.ai instances vanish
  permanently rather than pausing, so this is what stands between a multi-day run and starting
  over. It needs four things restored together — weights, optimizer moments, the schedule's
  step count, and every **RNG stream** — and the failure of any one of them looks like nothing:
  the run resumes, the loss descends, and the recipe quietly changed
  ([#94](https://github.com/JStoschek/DEMAE/issues/94)). Claimed and tested **at the worker
  count that ships**, not only at `num_workers = 0` — #94's gate held at 0 while the default was
  8, because the **dihedral augmentation** drew from a stream living in the worker processes;
  [#104](https://github.com/JStoschek/DEMAE/issues/104) made that draw a pure function instead, and
  the kill-and-resume test now runs with real workers. The general rule it leaves behind: state
  that must survive a resume belongs in a *pure function of position*, and only what genuinely
  cannot — optimizer moments, the mask draw's generator — belongs in the checkpoint payload.
- **RNG stream** — one independent source of randomness a run draws from, of which there are
  five and not one: the ambient torch generator (the in-model mask draw), CUDA's per-device
  generators, NumPy's legacy global, Python's `random`, and the DataLoader's own
  `torch.Generator`. They are named as a list (`pretrain.checkpoint.RNG_STREAMS`) precisely
  because a stream captured in one place and restored in neither of the other two is a stream
  that silently is not restored. The augmentation was a sixth until #104 removed the stream
  rather than the restore; a pre-#104 checkpoint still carries the dead key and still resumes,
  which is why the checkpoint format version is unchanged.
- **Epoch order** — the sequence a training epoch reads its tiles in, derived as a pure
  function of `(seed, epoch)` (`pretrain.train.EpochShuffleSampler`) rather than drawn from a
  rolling generator. The distinction is what makes a **mid-epoch resume** continue the order it
  was interrupted in instead of drawing a fresh one — a bug that trains perfectly well while
  showing some tiles twice in an epoch and others never. A generator's state at epoch 300
  depends on every draw before it, including how many DataLoader iterators were constructed,
  which differs between a first launch and a resumed one; a pure function has no such history.
  What the sampler yields is an `EpochIndex` — an `int` that also carries the epoch it was drawn
  for, which is how the epoch reaches a worker process for the **dihedral augmentation** to key
  on; everything between the two treats it as the plain index it is.
  **This supersedes [ADR-0003](docs/adr/0003-dataloader-design.md) §1's global
  `DataLoader(shuffle=True)`** for the *training* loader only — the val loader still reads in
  manifest order, and the shuffle is still global and still seeded from the run config.
- **Milestone set** — the epochs whose end-of-epoch checkpoint is **protected from retention**,
  read off `Schedule.milestones` and never hardcoded. §11 keeps the last 3 checkpoints plus
  this set; §14 keeps the set so Phase 2 can plot downstream performance against pretraining
  compute. Read from the config because R5 ([#87](https://github.com/JStoschek/DEMAE/issues/87))
  may extend to 800 epochs: against a fixed 50/100/200/400 the 800 checkpoint would be an
  ordinary one, and last-3 retention would delete the endpoint of the very plot the milestones
  exist for. R2/R3 stop at 200, so their sets are shorter for the same reason.
- **Atomic checkpoint write** — tmp file, `fsync`, then `os.replace`, so that at every instant
  the checkpoint name refers to a whole file. The failure it prevents is not a lost checkpoint
  but a **half-written one that loads**: a resume from it comes back with weights partly from
  the previous step, and nothing about the run looks wrong. The temporary name is hidden and
  outside the discovery glob, because a hard kill leaves it behind with no chance to clean up.
- **Fixed val mask seed** — the committed seed the val pass masks under
  (`pretrain.config.VAL_MASK_SEED`, plan §10.1 and §15's checklist item 3). MAE's masking is a
  fresh random draw per sample per step, so a val loss scored under fresh masks moves for
  reasons that have nothing to do with the model, and comparing two epochs compares two
  measurements. Pinning the draw is what makes the number comparable epoch to epoch, across
  runs, and across ablation arms — and it is half of what §10's standing decision rule
  adjudicates on (the kNN probe trend is the other half). Deliberately **not** `runtime.seed`:
  re-seeding a run to sample the same recipe twice must not move the ruler. The pass takes an
  *explicit* generator rather than seeding the ambient RNG, so measuring a run cannot perturb
  the masks its next training step draws.
- **Per-patch loss distribution** — plan §4's log (the issue calls it §10.4; §10.4 is the throughput dashboard), and the only thing that can answer §4's
  variance-floor rule: *if the top tail is dominated by near-flat patches, raise the floor before
  concluding anything about the model.* That is a question about the **joint** distribution of
  loss and flatness, so `pretrain.monitor.LossHistogram` cross-tabs the two — a run can have 40%
  near-flat patches whose tail is all real terrain, or 4% that are the whole tail, and those call
  for opposite actions. Binned logarithmically because the structure is at both ends, and
  accumulated over the val pass only: reading per-patch tensors in the training loop is a device
  sync per step for a diagnostic read once an epoch. The bottom bin is load-bearing on its own —
  a **broken `norm_pix` target** puts every patch in it, which is the one collapse the alarms
  structurally cannot catch (see **Alarm**).
- **Alarm** — a §10.1 failure condition wired to fire, as opposed to written down
  (`pretrain.alarms`). Four: *collapse toward zero* (the broken-`norm_pix` symptom), *val rising
  while train falls* (overfitting, plausible here because the corpus card §4.6 warns the
  effective dataset is smaller than 500k), *a plateau at high loss inside the first 20 epochs*
  (an LR or normalization bug, adjudicated where §6's warmup reaches peak LR), and the
  **loader wait fraction** over ~10%. Each carries an epoch, a message with the numbers that
  fired it, and an action — `stop` or `watch` — which is advice: nothing kills a run on its own
  reading of a threshold. Thresholds are fractions of the run's **own first epoch**, never
  absolutes, because the loss scale is a recipe knob (R3 ablates `norm_pix` off and scores
  something else entirely). The stated cost of that choice: a run already broken at epoch 0 has
  a broken baseline and no ratio against it fires — which is what the per-patch distribution's
  bottom bin covers instead.
- **Metrics sink** — `pretrain.sink`, whose interface is `RunLogger` (`MemoryLogger` in
  process, `WandbLogger` against W&B, `open_run_logger` to build one): the one seam every
  monitoring payload goes through
  (plan §11: *the box holds nothing unique*). Payload-typed — a number, a `Histogram` or an
  `Image` — because two of its three consumers do not emit scalars: #98's reconstruction grids
  and #99's probe trends write into it beside #97's losses, on one shared **step axis, which is
  the epoch**. Built against W&B's **offline** mode by default: a local run directory, no
  account, no network, `wandb sync` later — which decouples the code from the ops account and is
  the better test target anyway. Online is the same code with one environment field moved. Every
  run carries §15.7's four tags — git SHA, config hash, selection-list md5, split md5 — derived
  from the config and the fingerprint rather than typed, and an uncommitted split tags as
  `split:absent` rather than going missing, because the two are indistinguishable afterwards and
  the absent case is the dangerous one.
- **Val panel** — the committed 40 val tiles §10.2's reconstruction grids render, **2 from each
  of the 20 strata** (`artifacts/phase1/val_panel_40.csv`, drawn by `pretrain.panel` under
  `PANEL_SEED`; plan §15's checklist item 3). Third artifact in the same contract family as the
  selection list and the split assignment — *the committed file is what later runs re-read* —
  and committed for the reason a grid is never read on its own: it is read against the last
  epoch's, against the other ablation arm's, against the same tiles at a different mask ratio.
  A panel that re-drew itself would keep producing perfectly good pictures of a *different* 40
  tiles, and nothing in the picture would say so. Stratified because the finding it exists to
  support is about **terrain families** ("stratum 14 smears"), val-only because a train tile
  here would show the model reconstructing its own training data in the one panel built to
  judge generalization. Re-checked against the manifest at every load — a tile the corpus no
  longer holds, one whose split has moved, or one whose stratum has been re-fit is refused
  rather than rendered.
- **Reconstruction grid** — the §10.2 triptych, `pretrain.recon`: **masked → predicted → truth**
  for each panel tile, every 10 epochs, in **metres**, plus a **hillshade** of each. Metres
  because the model works in two nested normalizations — §4's per-patch `norm_pix` and the
  loader's per-tile standardization — and neither is an elevation; both are undone, the second
  by the very `Normalizer` instance that applied it. The stated limit of that inversion: with
  `norm_pix` on, a predicted patch's *level* is the target's and only its *shape* is the
  model's, because §5's design means the model is never asked to predict absolute altitude.
  The **hillshade is the part that earns the ticket** — drainage texture lives in the derivative
  at amplitudes a colour ramp cannot show, and low-relief drainage is precisely where the corpus
  card §3.1 predicts (and [ADR-0008](docs/adr/0008-accept-f3-drainage-shortfall-strata-2-17.md)
  accepts) this corpus is thinnest. Unlike Gate E's sheets it is **unstretched** and shares the
  truth's vertical exaggeration, because a per-panel contrast stretch restores a smeared
  prediction to full range and hides the one failure the grids are built to catch. The ramp and
  the exaggeration come from the truth alone, so the picture does not rescale itself on the
  epoch the model moves; and non-finite predictions are **painted**, not propagated, because
  §15's item 5 requires the whole pipeline to render on a randomly initialized model and a NaN
  reaching W&B reads as a reconstruction failure rather than the renderer bug it is.
- **Probe set** — the committed ~5,000 val tiles §10.3's kNN probe scores, drawn stratified over
  `stratum_id` (`artifacts/phase1/probe_set_500k.csv`, by `pretrain.probe`; 5,000 tiles over 477
  source COGs, half the shipped val split). Fourth artifact in the same contract family as the
  selection list, the split assignment and the val panel, and committed for the family's reason
  sharpened: **every probe number is a comparison** — this checkpoint against the last, this
  ablation arm against the release recipe — so a probe set that re-drew itself would make each
  of those a comparison between two different tile populations. Re-checked against the manifest
  at every load: a tile the corpus does not hold, a stratum that has moved under the committed
  label, or a tile outside `val` is refused rather than scored.
- **kNN probe** — §10.3's cheap proxy for representation quality, `pretrain.probe`, run **every
  20 epochs** (epoch 0 included, as the untrained baseline). A **frozen** encoder produces one
  mean-pooled feature per probe tile, and a kNN over those features classifies `stratum_id` and
  regresses `slope_mean` and `geomorphon_entropy` — supervision the manifest already carries, so
  it costs nothing. Half of what #85 adjudicates the release recipe on, jointly with the
  fixed-mask val loss, and the only in-flight signal that reports on the *encoder* rather than on
  the decoder's inpainting. Three properties are load-bearing:
  - **Frozen is asserted, not assumed** — eval mode restored, no gradient reached, no parameter
    moved, each checked on the way out. And the probe's loader carries an explicit generator:
    a `DataLoader` without one draws its base seed from the ambient RNG, which is the stream the
    masking draws from, so measuring the run would otherwise change it.
  - **The bank excludes a query's own source COG.** Corpus card §4.6: two windows of one COG are
    neighbours, not independent samples. A plain leave-one-out kNN answers most queries with the
    overlapping window next door and reports a high accuracy for an encoder that knows nothing
    transferable — the corpus's geography read back as representation quality, at its most
    flattering. Same blocking unit the val split was drawn at, one level down.
  - **The SSL4EO caveat is emitted, not documented.** SSL4EO-S12 measured MAE representations as
    weak under linear and kNN probes relative to contrastive methods, on benchmarks where MAE
    fine-tuning ultimately wins, so these numbers compare **trends and never set an absolute
    bar**. The caveat is logged beside the numbers at every probe point and printed under every
    line, each accuracy is reported against its own leave-one-COG-out no-skill baseline, and the
    headline series is the **delta against the epoch-0 baseline** — so the natural thing to read
    off a chart is movement rather than level.
- **Channel-ready** — the pipeline emits a single elevation channel now but is
  architected so slope/curvature channels are a Phase-3 config flag, not a rewrite.
- **GSD** — Ground Sample Distance (metres/pixel); 30 m for GLO-30. Multi-resolution
  / GSD-aware encoding is Phase 4, out of scope now.

## Decisions

Architectural decisions are recorded as ADRs under [`docs/adr/`](docs/adr/), created
lazily as wayfinder decision tickets resolve. If your work contradicts an ADR,
surface it rather than silently overriding.
