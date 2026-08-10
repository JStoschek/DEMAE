# Task 1 Execution Plan: The DEM Void-Filling Benchmark

**Planning memo, August 2026.** Turns §2 of
`downstream-task-selection.md` into a runnable experiment: comparison arms,
the DEMAE / TerraMind reconciliation, dataset construction, training setup,
metrics, and the fairness checklist. Written for a < $50 residual budget on
rented 4090-class GPUs, in the repo's committed-artifact, byte-reproducible
style.

## 0. What this benchmark must show

Three claims, each falsifiable, each mapped to a specific table row. "Worth" is
defined before any number is seen:

- **C1 (pretraining works):** DEMAE-frozen features beat ImageNet-MAE-frozen
  features and TerraMind-tokenizer-frozen features at void filling *under an
  identical trained head*, on the same tiles, same masks, same seeds. This is
  the representation claim, and the fairest instrument for it.
- **C2 (the model is useful):** fine-tuned DEMAE beats classical interpolation
  (the operational status quo) in metres, slope, and drainage metrics on rugged
  terrain at every void size, and stays at least even on **flat terrain**, where
  classical fill is already at the corpus's resolution floor. The parity
  carve-out is by ruggedness, not by void size: p119 measured classical fill at
  0.87 m MAE on small voids and 1.83 m on large ones in the flattest quartile —
  against 12.7 m and 38.6 m in the roughest — so "small voids where splines are
  strong" was the wrong exemption to grant
  ([`p119-fill-headroom.md`](p119-fill-headroom.md)).
- **C3 (label efficiency):** DEMAE-frozen reaches a given RMSE with far fewer
  head-training tiles than the UNet-from-scratch baseline (the 1k / 5k / 20k
  sweep). This is the foundation-model value proposition.

If C2 partially fails (e.g., Diff-DEM wins on texture), that is reported, not
hidden; PANGAEA-style candor is part of the deliverable. C1 and C3 are the
claims DEMAE's worth actually rests on, because they isolate the pretrained
representation from decoder capacity.

## 1. Comparison arms

| # | Arm | Encoder | Trained part | Input convention |
|---|-----|---------|--------------|------------------|
| A1 | Bilinear/spline fill | none | none | classical `gdal_fillnodata`-class interpolation; also rasterio `fillnodata`, and a thin-plate-spline variant |
| A2 | DEMAE zero-shot | DEMAE frozen (R1 ViT-S; R4 ViT-B if usable) | none | native MAE inpainting: void-touching patches dropped, decoder reconstructs, non-void pixels composited back |
| A3 | DEMAE fine-tuned | DEMAE, all weights | encoder + decoder | native token dropping on void-shaped masks (§4.2) |
| A4 | DEMAE frozen + head | DEMAE frozen | head only | voided tile pre-filled by A1 + void mask channel at the head (§2.4) |
| A5 | TerraMind frozen + head | TerraMind DEM tokenizer ViT-B encoder, 768-d `feats` (pre-`quant_proj`), frozen | head only (identical to A4's) | same pre-filled input, TerraMind's own normalization (§2.1) |
| A6 | ImageNet-MAE frozen + head | timm/MAE ViT-S (and ViT-B) frozen, 1-channel patch embed collapsed (same recipe as `pretrain.init`) | head only (identical) | same pre-filled input |
| A7 | UNet from scratch | none (UNet) | everything | voided tile + mask channel, residual over the A1 fill |
| A8 | Diff-DEM released checkpoint | theirs | none | their protocol, their data (cross-eval, §3.4) |
| A9 | TerraMind tokenizer pass-through (reference row) | full tokenizer: encode, FSQ, diffusion decode | none | A1-pre-filled tile through the full VQ pipeline; shows the 5-d bottleneck's cost |

A4 vs A5 vs A6 is the heart of the benchmark: identical head, identical data,
identical masks and seeds, only the frozen representation differs. A2/A3 are
DEMAE-only capabilities (the MAE *is* an inpainter; the tokenizer is not), and
that asymmetry is itself a finding to state: reconstruction-through-a-bottleneck
and masked-prediction are different pretraining objectives, and only one of them
hands you an inpainter for free.

A9 is cheap (inference only) and instructive: TerraMind's published DEM
reconstruction error (MAE 170.7 m, RMSE 737.2 on an SSL4EO-S12 subset, paper
appendix Table 10) suggests the pass-through will not be competitive; measuring
it on this benchmark's tiles grounds the "the tokenizer was never meant as an
inpainter" caveat in a number instead of an assertion.

## 2. Reconciling DEMAE and TerraMind

The two encoders differ on four axes. Each gets an explicit decision, not an
implicit default.

### 2.1 Normalization: feed each model its native diet

- DEMAE: per-tile standardization, mean/std computed from **non-void pixels
  only** (a voided tile's true stats are unknowable at inference; using
  full-tile stats would leak). A unit test asserts fill(identity-mask) == input.
- TerraMind: its global z-score, `(x - 670.665) / 951.272`
  (`v1_pretraining_mean/std["tok_dem@224"]` in terratorch), NaNs pre-filled,
  because the tokenizer does not normalize internally and TerraMesh trained it
  NaN-free.

Each encoder at its best is the fair default. One ablation row: TerraMind fed
per-tile-standardized input, to measure how much of its representation is
absolute-altitude passthrough. The head sees each encoder's features as-is
(LayerNorm at the head input equalizes scale), and **loss and metrics are always
computed in metres after inverting DEMAE's per-tile transform with the non-void
stats**, so no arm is scored in its own normalized space.

### 2.2 GSD: primary regime at 30 m, secondary at 10 m

TerraMind's DEM pixels were GLO-30 bilinearly resampled to the 10 m Sentinel-2
grid (TerraMesh §3.2); DEMAE's are native 30 m. Decisions:

- **Primary regime (all arms): 224x224 at native 30 m.** The task is defined
  operationally on GLO-30, DEMAE's home turf and honestly labeled as such;
  TerraMind runs out-of-distribution here and the writeup says so. Both ViTs
  produce a 14x14 token grid, so the head is byte-identical. **Measured, and it
  co-registers by resampling:** TerraMind's DEM tokenizer was trained at 256 px,
  its `pos_emb` is natively `[1, 768, 16, 16]`, and `ViTEncoder.forward`
  bicubically interpolates it to 14x14 on every 224 px pass
  ([`p118-terramind-encoder-probe.md`](p118-terramind-encoder-probe.md) §6). The
  premise holds patch-for-patch; what it costs is that TerraMind runs 12.5% off
  its native grid while DEMAE runs at its own. That is a one-sided handicap, and
  **whether this regime keeps 224 px and names it, quantifies it, or moves to
  256 px is [#133](https://github.com/JStoschek/DEMAE/issues/133)** — open at the
  time of writing, so this bullet is not yet final. Distinct from the secondary
  regime below, which is about GSD, not grid.
- **Secondary regime (A5 only): TerraMind-native check.** The same geographic
  windows bilinearly resampled to 10 m, center-cropped to 224 px (covering
  2.24 km instead of 6.72 km), TerraMind + head retrained. If TerraMind-at-10m
  still loses A4 vs A5, the GSD objection is closed. Run this once, at the 20k
  training size only.

### 2.3 Capacity: report it, and bracket it

ViT-S (22M) vs ViT-B (90M encoder) is a 4x gap. Mitigations: parameter counts
printed in every table; if the R4 ViT-B continual-init DEMAE arm is usable, it
runs in A2/A3/A4 so there is a same-capacity DEMAE-vs-TerraMind cell; A6 runs at
both ViT-S and ViT-B for the same reason. If DEMAE-S beats TerraMind-B on frozen
probes, the capacity gap strengthens the claim rather than weakening it.

### 2.4 The void-input convention for frozen encoders

Neither frozen encoder can be told "these pixels are missing" (1-channel input,
no mask token in inference mode). Convention, applied identically to A4/A5/A6/A7:

1. Fill voids with the A1 classical interpolation (bilinear `fillnodata`).
2. Encode the filled tile with the frozen encoder.
3. Concatenate the binary void mask (downsampled to the token grid, and at full
   resolution inside the head's upsampling path) to the head input.
4. The head predicts a **residual over the A1 fill**, not raw elevation.
   Residual formulation is current best practice in the DEM-correction
   literature (GrounDiff, FathomDEM-style pipelines) and means every learned
   arm starts from the classical baseline and must *improve* it; a head that
   learns nothing scores exactly A1, which makes the table legible.

DEMAE's native arms (A2/A3) instead use token dropping, which is the honest
representation of "MAE inpaints natively"; both conventions are documented and
neither is used to score the other's arm.

## 3. Dataset construction

### 3.1 Tiles and splits

Source: the existing `corpus_500k` shards and the committed geographic split
(`artifacts/phase1/split_scale_500k.csv`). No new pixel reads needed for the
core benchmark.

- **Head-train set:** 20,000 tiles from *train* geocells, stratified by the
  corpus's k=20 strata (proportional draw, seeded, committed as a
  `tile_id` list in the existing `pretrain.artifact` format).
- **Val set:** 2,000 tiles from *val* geocells (early stopping, head LR pick).
- **Test set:** 5,000 tiles from *test* geocells, never touched until the final
  pass, stratified over strata so flat and rugged terrain are both represented.
- **Label-efficiency subsets:** nested 1k ⊂ 5k ⊂ 20k draws (nesting, so the
  sweep measures data volume, not draw luck).

All four lists are committed artifacts; the benchmark's determinism contract is
the same as the corpus's: two runs from committed inputs reproduce the tables.

### 3.2 Synthetic void masks

A seeded `benchmark.masks` generator, one committed mask list per eval tile
(train-time masks are drawn on the fly from the same generator, seeded by
`(tile_id, epoch)`).

- **Geometry families:** (i) random-walk blobs dilated to target area, the
  standard proxy for radar-shadow voids; (ii) axis-aligned rectangles
  (sensor-dropout proxy); (iii) transplanted real SRTM void shapes (§3.3).
- **Size buckets:** small 16 to 48 px, medium 64 to 96 px, large 128 to 160 px
  (the two upper buckets match Diff-DEM's published protocol so numbers are
  directly comparable). One void per tile per eval instance; every eval tile
  appears once per bucket **for the two synthetic families (i) and (ii)**, so
  bucket scores are computed over identical tile sets.
- **The real-void family (iii) is exempt from bucket balancing.** Real SRTM
  voids are ~94% small and the 128 to 160 px bucket is empty in practice — two
  cells measured at true 1 arc-second returned 139 and 175 components in
  16 to 48 px, 2 and 6 in 64 to 96, and **zero** in 128 to 160. Forcing
  667-per-bucket would mean either fabricating large voids, which destroys the
  only thing the family contributes, or silently reweighting, which makes it
  non-comparable anyway. It is drawn at its natural size distribution and
  reported as its own row instead (§3.3).
- **Constraints:** voids fully interior (≥ 8 px from tile edge); void may not
  cover > 40% of the tile; the same mask list is reused verbatim by every arm.

### 3.3 Real SRTM void geometry

Scoped and measured in [`p125-srtm-void-geometry.md`](p125-srtm-void-geometry.md)
on 17 real SRTM tiles rather than inferred from published summaries. That study
corrects this section on four points, each recorded below.

- **Source and predicate.** Pull SRTMGL1N ("number of scenes") granules for
  void-rich regions — **the Himalaya/Karakoram and the Andes** — one
  3601x3601 `uint8` class raster per 1-degree cell, behind a free Earthdata
  Login that a human must click through once. The void predicate is
  **`num > 10 AND num < 201`**: SRTM-observed pixels carry the swath count
  **201 to 224**, water is 1/2/5, and land values above 10 but below 201 are
  fill from some other source, i.e. void. **There is no value 0 in this
  product's class table** (SRTM Collection User Guide V3, Table 2, p. 6); 0
  means void in `SRTMIMGM`'s differently-encoded `.num`, which is a different
  product. The earlier "pixels with scene count 0 are true voids" rule produces
  an **empty mask**. Two values inside the void range are arguable — 11 (very
  small SRTMv1 voids that NGA interpolated) and 25 (GDEM fill that was itself
  SRTM-derived); both are rare, and excluding them for a strict "never observed
  by radar" mask is a documented flag on the extractor, not a silent choice.
- **Not the Atlas.** Measured void fractions: Himalaya 3.7 to 22.5%, Andes
  0.03 to 2.0%, Atlas 0.03 to 0.16%. Atlas cells yield 4 to 24 usable
  components each against the Himalaya's 174 to 689 — two orders of magnitude
  poorer. It is dropped from the void-rich region list.
- **Not NASADEM_SHHP.** NASADEM's reprocessing filled **52 to 63%** of SRTM's
  voids (NASADEM User Guide V1, Table 3), so an SHHP mask is the population that
  survived a better phase unwrapper: a smaller, differently-shaped phenomenon,
  at ~150x the bytes. It is the right source only if the family's claim changes
  from "real SRTM void geometry" to "voids that defeat modern InSAR", and the
  writeup would have to say so.
- **The fallback is degraded, not equivalent.** SRTM v2.1 is **3 arc-second
  only** in these regions — its 1 arc-second release was United States only
  (`SRTM_Topo.pdf` §2.2) — and the archive.org item is one 13.7 GB tarball with
  no per-tile access. If a no-login fallback is kept for resilience, point it at
  the Wayback per-tile capture of
  `dds.cr.usgs.gov/srtm/version2_1/SRTM3/<Continent>/<CELL>.hgt.zip`, ~2 MB per
  tile. Either way, label it degraded: measured on the only cells where both
  resolutions exist, the upsampled 3 arc-second mask scores **IoU 0.60** against
  1 arc-second truth and misses **39% of void area**, because a 3 arc-second
  post is void only when all nine of its 1 arc-second parents are — an erosion
  that shaves off exactly the ragged filigree the family exists to test.
- **Extract, do not resample.** Connected void components between 16 and 160 px
  equivalent diameter, then **transplant onto clean GLO-30 test tiles of
  matching ruggedness stratum** (truth exists there; scoring on actual SRTM
  voids is impossible since the ground truth under them was never observed by
  SRTM). There is no "rasterize to 30 m" step: SRTM 1 arc-second and Copernicus
  GLO-30 below 50 degrees latitude are the *same lattice*, so the mask is
  cropped and pasted, never interpolated (P125 §10.1). The benchmark code
  asserts `|lat| < 50` rather than discovering it, since GLO-30 widens its
  longitude spacing poleward.
- **The fourth eval family, at its natural size distribution.** ~2,000
  instances drawn **as measured — roughly 94% small, 5% medium, 1% large** —
  reported as its own row, with the size histogram published beside the score,
  and exempt from §3.2's bucket balancing. This is deliberate: the family is not
  a fourth point of comparison against the bucketed synthetic families, it is a
  **distribution-shift check** — do methods tuned on synthetic blobs hold up on
  real void geometry, at the sizes real voids actually come in? Framed that way
  the small-void skew is the experiment rather than a defect. Real voids are
  strikingly non-compact (perimeter ~2.0 to 2.2x that of an equal-area disc) and
  concentrate in steep terrain and radar shadow; the stratified transplant
  preserves that difficulty rather than averaging it away.

### 3.4 Diff-DEM cross-eval (transfer test)

Their released Norway 10 m dataset, their mask lists, their checkpoint. Two
directions: Diff-DEM's model on its own data vs DEMAE arms on the same data
(off-distribution for a GLO-30 model: different country, 3x finer GSD); and
optionally their model on this benchmark's 30 m tiles (off-distribution for
them, same caveat, stated symmetrically). This section is labeled *transfer*,
not headline, in every table.

### 3.5 Committed artifacts

`benchmark/` module in the repo's stage-per-module style: tile lists, mask
lists (parquet: `tile_id, family, bucket, mask_rle, seed`), the mask
generator's config hash, and a dataset card documenting all of the above plus
the licenses (SRTM public domain, Diff-DEM's data per their repo terms).

## 4. Training setup

### 4.1 The shared head (A4 / A5 / A6)

One head spec, frozen across arms; only the input channel width differs
(384 vs 768), which the first projection absorbs.

- **Architecture:** LayerNorm on patch tokens, 1x1 conv to 256, then a light
  convolutional decoder: 4 stages of [3x3 conv, GroupNorm, GELU, 2x bilinear
  upsample] (256, 128, 64, 32), full-resolution void mask concatenated at each
  stage's input, 1-channel residual output added to the A1 fill. ~3M params.
  No UperNet here: multi-scale semantic decoding is overkill for single-image
  regression, and a smaller head keeps the comparison about the features.
- **Loss:** masked L1 on void pixels in metres (L1 over L2: robust to the
  heavy-tailed elevation-error distribution; consistent with current DEM
  correction practice), plus 0.1x a slope-consistency term (L1 on Sobel
  gradients inside the void) to discourage flat-patch cheating.
- **Optimizer:** AdamW, lr 1e-3, wd 0.05, cosine to 0, 5% warmup, bf16,
  batch 128, 50 epochs on the 20k set, early stop on val RMSE. These knobs are
  tuned **once on A6 (the neutral encoder) using val only**, then frozen for
  every arm; tuning on the arm you hope wins is the classic thumb on the scale.
- **Seeds:** 3 seeds per cell for A4/A5/A6/A7 at 20k; 2 seeds for the
  label-efficiency subsets. Report mean ± std.
- **Features are *not* precomputed** *(amended 2026-08-10, §7 —
  [#120](https://github.com/JStoschek/DEMAE/issues/120))*. The frozen encoder runs inline
  in the head step, and the void mask is redrawn every epoch per §3.2. The cache this
  bullet used to specify — fp16 memmaps, 301 KB/tile, 5.6 GiB per encoder per 20k set —
  was measured to save **$0.50** across the whole benchmark, because a frozen ViT-B
  forward pass costs 0.179 ms. Against that it fixed the A1-filled input, and therefore
  the mask, for all 50 epochs: the sentence this bullet used to end with ("precomputation
  fixes the A1-filled input per (tile, mask), which is exactly the eval-time contract")
  is true at eval time and is a **frozen training mask** at training time, which §3.2 does
  not intend. The *eval* side is unchanged — the committed mask list fixes each test
  instance's input exactly as before.

### 4.2 DEMAE native fine-tune (A3)

- Init from the release checkpoint (R1; R4 arm if usable). Continue MAE
  training with the mask sampler swapped: instead of 75% random patches, drop
  the set of patches overlapping a sampled synthetic void (16 to 160 px
  families), plus 25% random patches so the encoder keeps seeing global
  context and the effective mask ratio stays near pretraining's.
- **Loss:** masked MSE in per-tile-standardized space with `norm_pix` **off**
  (void filling needs absolute per-patch scale; norm-pix targets discard it).
  Eval always re-inverts to metres with non-void stats.
- **Schedule:** 20 epochs over the 20k head-train set, lr 1e-4 (encoder) with
  0.75 layer-wise decay, 1.5e-3 (decoder), cosine, bf16; this is a short
  adaptation, not a new pretraining.
- **Inference composite:** predicted patches only where voided, original pixels
  elsewhere, then a 16 px cross-fade at patch boundaries inside the void.
  For voids near tile scale, 4 shifted encodings (0 and 112 px offsets in x/y)
  averaged, which suppresses patch-grid seams (standard test-time practice).

### 4.3 Zero-shot (A2)

No training. Mask = patches touching the void; the known caveat is that MAE was
pretrained on *random* masks and contiguous block masks are a distribution
shift, so zero-shot numbers are expected to be modest, improving under A3.
Plotting zero-shot vs fine-tuned per bucket quantifies precisely what the
task-adaptation bought, which is a figure in itself.

### 4.4 UNet baseline (A7)

Standard 4-down/4-up UNet (~17M params, comparable order to ViT-S+head),
2-channel input (A1-filled tile, void mask), residual output, same loss, same
schedule length, same seeds. Trained from scratch at each label-efficiency
size. This is the PANGAEA bar: if A7 matches A4 at 20k tiles, the honest
conclusion is that pretraining pays only in the low-label regime, and C3 is
where DEMAE must show it.

### 4.5 Order of operations

1. Freeze mask lists and tile lists (CPU only, local).
2. A1 baseline numbers + unit tests on norm inversion and mask bookkeeping.
3. Precompute frozen features (one GPU-hour class).
4. Tune head hyperparams on A6/val; freeze.
5. Run A4, A5, A6, A7 at 20k; then the 1k/5k sweep; then A3; then A2/A9
   (inference only); then A8 cross-eval; secondary 10 m regime last.

## 5. Metrics and statistics

All metrics computed **only inside the void**, in metres, per (family, bucket),
over the fixed test mask lists. The real-void family is the one exception: it
carries no bucket axis (§3.2), so it is a single row per arm, published with its
size histogram so the reader can see what distribution the number is over.

- **Elevation error:** MAE, RMSE, NMAD (robust spread, standard in DEM QA), and
  the 95th percentile absolute error (tail behavior matters for hazards).
- **Terrain derivatives:** slope RMSE (degrees), aspect error (circular
  distance, slope-weighted so flat pixels do not dominate), profile-curvature
  RMSE. Derivative fidelity is more meaningful than pixel RMSE for terrain,
  and it is where blurry fills get caught.
- **Hydrological plausibility:** fill each test DEM's void, compute D8 flow
  accumulation (pysheds or WhiteboxTools) on truth vs filled, report log-flow
  correlation inside and 2 px around the void, plus extracted stream-network
  IoU at a fixed accumulation threshold. Drainage continuity is the
  operational reason void quality matters.
- **Perceptual/structural:** hillshade SSIM inside the void (hillshade, not raw
  elevation, is what a human QA analyst looks at). LPIPS optional, not a
  headline (its ImageNet features are off-domain for hillshades).
- **Stratification:** every metric also reported by the corpus's ruggedness
  strata (collapsed to 4 groups for table size). A method that wins on plains
  and loses in mountains must be visible as such. This is **load-bearing, not
  cosmetic**: p119 measured the classical baseline's error varying ~40× across
  these four bands but only ~3× across the size buckets
  ([`p119-fill-headroom.md`](p119-fill-headroom.md)), so a pooled per-arm number
  is dominated by the rugged tail and hides the fact that the flattest quartile
  is already at the resolution floor. Where table space forces a choice, drop a
  size bucket before dropping a ruggedness band.
- **Statistics:** paired bootstrap over test tiles (10k resamples) for each arm
  pair on RMSE and slope RMSE; report 95% CIs and mark table cells where the
  CI excludes zero. Three-seed spread reported separately from bootstrap CI
  (seed variance vs data variance are different quantities).
- **Efficiency footnote:** params trained, params total, and single-tile
  inference time per arm (A8's diffusion sampler vs A3's single forward pass is
  a real operational difference worth one honest row).

## 6. Fairness checklist (goes in the writeup verbatim)

1. Identical tiles, masks, seeds, head, schedule for every frozen-encoder arm;
   head hyperparams tuned once on the neutral encoder (A6), then frozen.
2. Each encoder fed its native normalization; scoring always in metres;
   per-tile stats from non-void pixels only; identity-mask unit test committed.
3. GSD stated per regime; primary regime is DEMAE-native and says so;
   TerraMind gets a retrained head in its own 10 m regime before any
   conclusion is drawn about it.
4. Parameter counts in every table; ViT-B cells (R4, ImageNet-B) bracket the
   capacity confound.
5. The tokenizer encoder was trained for reconstruction, not representation;
   A9 quantifies the bottleneck; conclusions are phrased as "under identical
   probes" not "TerraMind is worse."
6. Geographic holdout only; the test geocells are the corpus's committed test
   split; Diff-DEM data is labeled transfer, both directions.
7. Negative or null results (e.g., A7 ties A4 at 20k) are reported with the
   same prominence as wins; C3's low-label sweep is the pre-registered fallback
   claim.
8. Pre-registration: this memo is committed before the first head trains;
   success criteria C1-C3 are defined above the results, not after them.

## 7. Compute budget

**Measured, 2026-08-10 — [`p120-primitive-costs.md`](p120-primitive-costs.md)
([#120](https://github.com/JStoschek/DEMAE/issues/120)).** The table below replaces an
estimate that was high by 8 to 13x and was denominated in the wrong card at the wrong
price: this project rents an **RTX 5090** at a real **$0.4016/hr**, not a 4090 at $0.34
(see [`phase1-pretraining-plan.md`](../phase1-pretraining-plan.md)'s hardware line and
§12). Every line is a measured primitive times a count, and the primitives are published so
any new arm can be costed without a fresh run.

Costed for the design §4.1 now specifies — **frozen encoder inline, no feature cache**:

| Item | Count | 5090-min | Notes |
|---|---|---|---|
| **A7 UNet, 3 seeds + label sweep** | 3 x 50 ep x 20k, + 2 x 50 ep x 6k | **46.1** | priciest single arm; UNet epoch is 15.4 s against the head's 1.0 |
| Head tuning on A6 | 12 configs x 50 ep x 20k | 45.7 | |
| Head training A4/A5/A6 | 5 cells x 3 seeds x 50 ep x 20k | 45.1 | |
| Label-efficiency sweep (1k, 5k; 2 seeds) | 5 cells x 2 seeds x 50 ep x 6k | 9.0 | |
| Inference over the eval instances | 47 passes x 32k | 4.5 | frozen forward is **0.179 ms/tile** at ViT-B; prices an encoder pass, so A2/A3's decode is not in it |
| A3 fine-tune (ViT-B + ViT-S) | 20 ep x 20k | 3.1 | fine-tune epoch is **6.66 s**; measured equal to a pretraining epoch |
| **GPU total** | | **153.6 min = 2.56 h** | **$1.03** at $0.4016/hr |
| **A9 (TerraMind full tokenizer)** | 32k instances | **unpriced** | diffusion decode; at 1 s/instance it is 8.9 GPU-h, 3.5x the whole table above. **§7 is not final until this is measured.** |
| Eval passes (CPU, local, single core) | **48** arm-seed passes x 32k instances | **1.99 h** | 4.6 ms/instance; **the largest single line in the benchmark** (118 min of it is scoring) |

The 48 scored passes are 24 at 20k — A1, A2 and A3 at both capacities, A9, the five frozen
cells at 3 seeds, A7 at 3 seeds — plus 24 for the label-efficiency sweep, which §6 item 7
makes a headline claim and therefore scores rather than merely trains.

**The budget is not a constraint and should stop being treated as one.** At $1.03 of GPU
against a $25 target, no line here justifies a spot-instance strategy or a resume design;
#117's standing note that *attention and wall-clock are the real budget* is now measured
rather than asserted. The one number that could still move the total is A9's.

**Why no feature cache** (the §4.1 amendment, priced): precomputing frozen features is
*cheaper in GPU seconds* — 1.31 h against 2.56 h — but the whole difference is **$0.50**,
and against that it costs 30.3 GiB, 7.9 TB of re-reads, and a design constraint §4.1 did
not name: a cached feature fixes the A1-filled input, hence the mask, so head training
would see **one void per tile for all 50 epochs**, contradicting §3.2's on-the-fly
`(tile_id, epoch)` masks. Run the encoder inline and redraw the mask each epoch. §9.4's
Task-2 reuse constraint is unaffected: it binds the frozen-feature *contract*, not whether
features were written to disk.

**§4.1's "~3M params" is wrong** — the head as specified there measures **1.08M** (384-d)
and **1.18M** (768-d). Cost is barely affected, but §6 item 4 puts parameter counts in
every table, so the corrected figures are the ones to print.

## 8. Risks

- **Norm-inversion bugs** silently flatter or sabotage an arm: the identity
  test plus a golden-tile fixture (known void, hand-checked fill) gate every
  arm's eval.
- **A1 pre-fill quality ceiling: measured, and the saturating axis is ruggedness,
  not void size** ([`p119-fill-headroom.md`](p119-fill-headroom.md)). This row
  used to read "if bilinear fill is too good on small voids, learned residuals
  look like noise; the small-void bucket exists to show this honestly." The
  measurement — 1,600 real corpus tiles, 48,000 scored instances — says the
  bucket axis is the wrong one:
  - **Small voids are not saturated.** Pooled, the best classical variant leaves
    **MAE 5.2 m** on the small bucket, against GLO-30's own σ ≈ 2.43 m
    (LE90 4 m / 1.645). Only 42% of small-void instances beat σ. There is ample
    residual signal there.
  - **Flat terrain is saturated, at every size.** On the flattest ruggedness
    quartile (R1, median slope 1.0°, relief 40 m) a *small* void fills to
    **0.87 m MAE** with 96% of instances under σ, and even a *large* void fills
    to **1.83 m**. At that point the fill is within a factor of two of the
    int16-metre storage quantisation and there is nothing left to learn.
  - **Rugged terrain has the headroom.** On R4 (12.9°, 583 m relief) a small void
    already costs **12.7 m** and a large one **38.6 m**, with 0% beating σ. The
    spread across ruggedness bands is ~40×; across size buckets, ~3×.

  Consequences, applied elsewhere in this plan: §5's ruggedness stratification is
  promoted from a table-completeness nicety to the axis the headline result lives
  on, and C2's parity carve-out (§0) is retargeted from the small-void bucket to
  flat terrain. Which classical fill is chosen is *not* a risk —
  TPS beats bilinear `fillnodata` by a median of 0.2 m and wins only 55% of
  paired instances, so A1's identity barely moves the ceiling.
- **TerraMind feature extraction: run, not hypothetical**
  ([`p118-terramind-encoder-probe.md`](p118-terramind-encoder-probe.md)). The
  path works end to end in 36 s on the laptop, on terratorch 1.2.10 /
  diffusers 0.39.0, and the weights are public Apache-2.0 with **no login**.
  Three of this row's original four pitfalls did not survive the run:
  - **The `diffusers==0.30.0` pin does not exist.** terratorch declares a bare
    `diffusers`; `0.30.0` appears only in the text of an import-failure warning.
    The real hazard is the opposite one — nothing is pinned, so a future
    `diffusers` can break the tokenizer import with no lockfile signal. Pin
    `diffusers` in the benchmark's own environment, at the version it is
    verified on.
  - **The 14x14 grid is confirmed** on 224 px input, with the resampling caveat
    now recorded in §2.2.
  - **Global-vs-per-tile normalization is the live one, and it fails silently.**
    DEMAE's loader emits per-tile-standardized tiles; TerraMind wants one global
    affine over metres. The encoder accepts both and returns a plausible
    `[1, 768, 14, 14]` either way, with no error. The TerraMind arms therefore
    cannot consume the pretraining loader's output — see §9.4.
  - NaN filling before the z-score is untested here and stands as written.
- **Contiguous-mask shift sinks A2:** expected and framed as the zero-shot vs
  fine-tuned delta, not buried.
- **Spot-instance reclaim:** heads checkpoint every epoch to the synced
  directory like every other run in this repo; features are memmapped so
  nothing long-running is unresumable.

## 9. Deliverables

1. `benchmark/` module + committed tile/mask/split artifacts (reproduces every
   table from the repo).
2. The results table (arms x buckets x strata, CIs marked) and three figures:
   RMSE vs void size per arm; the label-efficiency curve (C3); a hillshade
   panel grid (truth, voided, A1, A3, A5, A8) on two rugged and one flat tile.
3. A short results memo in `docs/research/` answering C1, C2, C3 with numbers,
   plus the fairness checklist and the TerraMind caveats.
4. If C1 holds, the same frozen-feature cache feeds Task 2's probes unchanged,
   which is why this ordering is efficient. **Two constraints on that contract
   are now measured** ([`p118-terramind-encoder-probe.md`](p118-terramind-encoder-probe.md)):
   the TerraMind arms take **raw metres under TerraMind's global affine**, never
   the pretraining loader's per-tile-standardized output (§7 there) — so a cache
   shared with the DEMAE arms has to distinguish the two normalizations somehow,
   and how is a design call this ticket does not make; and the extractor needs
   only `encoder` + `quant_proj` — **90.2M parameters, 361 MB fp32, 31.5% of the
   1.15 GB checkpoint** — since the remaining 196M is the diffusion decoder that
   no in-scope arm calls.

## 10. Key sources

Diff-DEM (github.com/kylelo/Diff-DEM; IEEE GRSL 2024) · SRTMGL1N
(earthdata.nasa.gov, `.num` class raster; predicate and geometry measured in
`p125-srtm-void-geometry.md`) · SRTM Collection User Guide V3 (Table 2, the
`.NUM` fill values) · SRTM v2.1 (3 arc-second, Wayback per-tile capture of
`dds.cr.usgs.gov`; degraded fallback only) · TerraMind (arXiv 2504.11171; tokenizer model card, Apache-2.0) · TerraMesh
(arXiv 2504.11172, 10 m DEM resampling) · terratorch v1.2.10 source (encoder
extraction path, normalization constants) · PANGAEA (arXiv 2412.04204, frozen
protocol + baseline bar) · MAE (arXiv 2111.06377) · GrounDiff (arXiv 2511.10391,
residual formulation) · FathomDEM (ERL 2025, stratified DEM eval practice) ·
Kornblith et al. (arXiv 1905.00414) for the CKA follow-up if features get
compared directly.
