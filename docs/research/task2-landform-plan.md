# Task 2 Execution Plan: The Landform-Classification Benchmark

**Planning memo, August 2026.** Turns §3 of `downstream-task-selection.md` into a
runnable experiment, in the same shape as `task1-void-fill-plan.md`: claims,
arms, the DEMAE / TerraMind reconciliation, label and dataset construction,
training setup, the representation-comparison suite, metrics, fairness
checklist, and budget. Task 2 is where the low-label money plot lives and where
the TerraMind representation comparison is run in full.

## 0. What this benchmark must show

- **C1 (representation quality):** under identical frozen-encoder probes
  (linear and UperNet), DEMAE features classify landforms better than
  ImageNet-MAE features and TerraMind-tokenizer-encoder features, on the same
  tiles, labels, seeds, and heads.
- **C2 (label efficiency, the money plot):** DEMAE-frozen at 1% of labeled
  geocells beats or matches the from-scratch UNet at 10%, i.e. the pretraining
  buys roughly an order of magnitude of labels. This is the plot the whole
  project is remembered by, and the claim most likely to survive even if C1's
  margins are thin.
- **C3 (independence and practicality):** the advantage holds on the
  human-calibrated Iwahashi & Pike labels (closing the circularity objection),
  and a full fine-tune at 100% labels is at least competitive with the UNet
  (the PANGAEA bar).

Pre-registered before the first head trains. If the UNet ties the fine-tune at
100% labels, that is the expected PANGAEA outcome and is reported as such; C2
is the claim DEMAE's worth rests on.

## 1. Comparison arms

| # | Arm | Encoder | Trained part |
|---|-----|---------|--------------|
| B1 | DEMAE frozen + linear probe | R1 ViT-S frozen (R4 ViT-B if usable) | 1x1 conv on last-layer tokens |
| B2 | DEMAE frozen + UperNet | same | UperNet over 4 intermediate layers |
| B3 | DEMAE full fine-tune | DEMAE, all weights | encoder + UperNet, 100% labels only |
| B4 | TerraMind frozen + linear probe | tokenizer ViT-B encoder, 768-d `feats` (pre-`quant_proj`), frozen | identical linear probe |
| B5 | TerraMind frozen + UperNet | same, 4 intermediate layers via `return_intermediates=True` | identical UperNet |
| B6 | ImageNet-MAE frozen (linear + UperNet) | timm MAE ViT-S and ViT-B, 1-channel patch embed collapsed (the `pretrain.init` recipe) | identical heads |
| B7 | UNet from scratch | none | everything |
| B8 | TerraMind 5-d latent probe (ablation row) | post-`quant_proj`, pre-FSQ | linear probe |
| B9 | TerraMind backbone, optional | `terramind_v1_base` with raw DEM input (`untok_dem@224`), frozen | identical heads |

B1/B4/B6 (linear) and B2/B5/B6 (UperNet) are the core grid; every cell runs at
label fractions 1% / 10% / 100%. B8 exists to show, in one row, why the 5-d
pre-FSQ bottleneck is not "TerraMind's representation" and would have been a
strawman as the primary target. B9 is the strongest version of the competition
(the actual foundation-model encoder, which ingests raw DEM directly) and runs
if budget allows after the core grid; if it beats the tokenizer encoder, that
is the right TerraMind number to foreground in the writeup.

## 2. Reconciling DEMAE and TerraMind

Same four axes as Task 1 (§2 of that plan), with the task-specific deltas:

1. **Normalization.** Native diets again: DEMAE per-tile standardization
   (full-tile stats, no voids here), TerraMind global z-score (670.665 /
   951.272). The confound matters *more* in Task 2 than Task 1: landform
   classes correlate with absolute altitude (peaks are high, valleys low
   within a tile, but also mountains are high in general), and TerraMind's
   features can carry absolute elevation while DEMAE's cannot. Two controls:
   the geomorphon label is locally defined (a ridge at 500 m and at 5,000 m is
   the same class), which blunts the confound at the pixel level; and the
   ablation row feeding TerraMind per-tile-standardized input measures the
   altitude-passthrough contribution directly. Report it next to B4/B5.
2. **GSD.** Primary regime: everything at native 30 m, 224x224, both ViTs on a
   14x14 grid, heads byte-identical. Secondary regime (TerraMind only, once, at
   100% labels): 10 m-resampled input replicating TerraMesh preprocessing, with
   labels regenerated on the resampled grid, so the "TerraMind was
   out-of-distribution" objection has a measured answer.
3. **Capacity.** ViT-S 22M vs ViT-B 90M. B6 runs at both sizes, and R4 (if
   usable) gives a same-capacity DEMAE-B vs TerraMind-B cell. Additionally,
   Task 2 gets the depth sweep (§5.3): probe accuracy per encoder layer, which
   turns the capacity question into a curve instead of a single contested
   point.
4. **Objective mismatch.** The tokenizer encoder was trained for reconstruction
   through a 5-d bottleneck; reconstruction pressure keeps geometry but has no
   reason to linearize *classes*. Landform probing is therefore close to a
   best-case test for MAE-style pretraining and a hard test for a tokenizer
   encoder; the writeup phrases conclusions as "under identical probes," and B9
   exists so TerraMind is also represented by the model IBM actually intends
   as a representation encoder.

## 3. Labels and dataset

### 3.1 Primary labels: geomorphons

Generated with WhiteboxTools `Geomorphons` (pin the binary version in the
benchmark config; the algorithm is deterministic given parameters), 10 classes:
flat, peak, ridge, shoulder, spur, slope, hollow, footslope, valley, pit.

- **Parameters, fixed and committed:** outer search radius **L = 50 cells**
  (1.5 km at 30 m, the scale of hillslope-to-small-massif landforms), skip
  radius 0, flatness threshold **1.0°**. These are the standard
  literature-default settings; they go in the dataset card verbatim because
  geomorphon maps are parameter-sensitive.
- **Parameter-sensitivity ablation (CPU-only, cheap):** regenerate test-set
  labels at L = 25 and L = 100 and re-score the *already-trained* B1/B4 probes
  against them. This shows whether the encoder ranking is an artifact of one
  label scale; no GPU retraining needed for the check.
- **The edge problem, solved at generation time:** a geomorphon at pixel p
  needs terrain up to L cells away, so labels computed on a bare 224 px tile
  are wrong in a 50 px ring (60% of the tile's area). Fix: re-read each
  benchmark tile as a **324x324 buffered window** from the source COGs using
  the existing `dataset.tiling` machinery (same window center, +50 px on each
  side), compute geomorphons on the buffer, crop the center 224x224. Labels are
  then valid everywhere. Cost: ~27k windowed COG reads, CPU + network, the
  exact operation the corpus pipeline is built for; label rasters are uint8
  (~50 KB/tile, ~1.4 GB total), stored like shards (gitignored, rebuildable,
  fingerprinted). Fallback if the re-read is unwanted: compute on the 224 tile
  and mask the 50 px ring out of loss and metrics, at the price of scoring only
  the central 124x124.
- **Class imbalance is structural:** slope and flat dominate globally; pit and
  peak are rare. Handled in the loss (inverse-sqrt-frequency class weights)
  and in the metrics (macro-F1 and mIoU as headline numbers, never overall
  accuracy alone), and reported per ruggedness stratum, where the class mix
  shifts drastically.

### 3.2 Secondary labels: Iwahashi & Pike 2018 (independence check)

Global terrain-classification GeoTIFF (gisstar.gsi.go.jp/terrain2018, CC-BY
4.0), 280 m, 40 classes grouped to the published 15 terrain groups. At 280 m a
224 px GLO-30 tile spans ~24x24 label pixels, too coarse for dense
segmentation, so this is a **tile-level task**: majority group over the tile
footprint = the tile's label, predicted from the pooled embedding (mean-pooled
patch tokens, linear classifier; kNN variant in §5.2). Tiles whose majority
group covers < 60% of the footprint are dropped from this label set (mixed
tiles make the label meaningless); the drop rule and the resulting counts go in
the dataset card. Because the labels derive from a *different* DEM lineage and
human calibration, this is the answer to "you trained on the label-generating
function": nothing in the Iwahashi map is a function of GLO-30.

### 3.3 Tiles, splits, fractions

Same committed lists as Task 1 (20k train / 2k val / 5k test, geographic
geocell splits, stratified over the k=20 strata). No features carry over from Task 1:
every Task 1 encoder input is an A1-filled voided tile, so no clean-tile
feature was ever computed, and Task 1 runs its encoders inline with no cache
(#121). What carries over is the ADR-0012 extractor, which Task 2 also runs
inline (#184). Label
fractions are drawn by **geocell, not by tile** (1% of geocells, 10% of
geocells, nested), because a real low-label regime means whole regions
unlabeled, not a thin uniform scatter; the nested draw is seeded and committed.

### 3.4 Committed artifacts

`benchmark/` additions: geomorphon config (tool version, L, flatness, buffer),
label fingerprints, the Iwahashi tile-label table with the majority-fraction
column, fraction draws, and the dataset card covering licenses (WhiteboxTools
labels derived from GLO-30, attribution per Copernicus licence; Iwahashi
CC-BY 4.0).

## 4. Training setup

### 4.1 Linear probes (B1 / B4 / B6 / B8): the strict instrument

LayerNorm on frozen last-layer patch tokens, a single 1x1 conv (d -> 10
classes), bilinear upsample to 224, weighted cross-entropy. No hidden layers,
no augmentation beyond the pretraining-native dihedral group applied jointly to
tile and label raster. AdamW, lr 1e-3, wd 0, cosine, 30 epochs, batch 256,
with the frozen encoder run inline through the ADR-0012 extractor and reading
its `output`. Inline is what keeps the dihedral augmentation possible: the
feature of a rotated tile is not the rotated feature of the tile, so a cache
would have to hold all eight orientations or drop the augmentation. 3 seeds
per cell. The linear
probe is the headline instrument for C1 because no decoder capacity can mask
representation differences.

### 4.2 UperNet probes (B2 / B5 / B6): the PANGAEA instrument

UperNet over 4 intermediate layers (indices 3, 6, 9, 12 of 12 for every
encoder, stated explicitly since layer choice is a known sensitivity), frozen
encoder, ~30M-param head, weighted CE, AdamW lr 6e-4, 40 epochs, batch 64.
The encoder runs inline and the head reads the extractor's `intermediates` at
those indices; nothing is cached (#184). A cache would cost 32.5 GB per ViT-B
encoder (27k tiles x 4 layers x 196 tokens x 768 x fp16), ~146 GB for the five
frozen encoders, against roughly an hour of GPU forward for the whole UperNet
and linear-probe grid run inline.
Hyperparameters tuned once on B6 (the neutral encoder) on val, then frozen for
all arms, same discipline as Task 1.

### 4.3 Full fine-tune (B3) and UNet (B7)

- **B3:** encoder + UperNet, 100% labels only (budget), lr 1e-4 with 0.75
  layer-wise decay, head lr 1e-3, 50 epochs, dihedral augmentation, bf16.
  One seed plus a second if variance on val looks material.
- **B7:** the same 4-down/4-up UNet spec as Task 1's A7 (~17M params),
  1-channel input, from scratch at every label fraction, 3 seeds, same
  schedule length as B2 for comparability. This is the bar: PANGAEA says it is
  often not beaten, and the table must be able to say so plainly if it is not.

### 4.4 Order of operations

1. Generate and fingerprint geomorphon labels (CPU + network, no GPU): buffer
   re-read, WhiteboxTools pass, sensitivity variants for test only.
2. Build the Iwahashi tile-label table (one global GeoTIFF, zonal majority per
   tile footprint; CPU).
3. Pin every frozen encoder in the ADR-0012 extractor registry (no feature
   caching step: encoders run inline, #184).
4. Tune both heads on B6/val; freeze the recipes.
5. Core grid: B1/B2/B4/B5/B6 x {1%, 10%, 100%} x seeds; B7 x fractions.
6. B3 fine-tune; B8 ablation; TerraMind normalization and 10 m ablations.
7. Representation suite (§5): one inline pass per analysis over a fixed tile
   set.
8. B9 if budget remains.

## 5. The representation-comparison suite

Each analysis is one inline pass of the ADR-0012 extractor over a fixed tile
set, holding in memory only what it needs and writing no features to disk
(#184): ridge (§5.1) and CKA (§5.4) accumulate their covariances batch by
batch; kNN (§5.2) and MNN hold mean-pooled embeddings (~83 MB per encoder at
27k tiles x 768 fp32); the depth sweep (§5.3) and the concatenation row are
linear probes run inline as in §4.1. Nearly free; this is the part of the
project with no published precedent (nobody has probed the TerraMind DEM
tokenizer encoder as a representation model).

### 5.1 Terrain-derivative regression probes

Ridge regression (closed-form, CPU) from frozen per-patch tokens to per-patch
mean slope, aspect (sin/cos pair), and profile curvature, computed from the
buffered tiles so derivatives are edge-clean. R² per target per encoder.
Derivatives are invariant to absolute altitude, so this probe is immune to the
normalization confound and is the cleanest single number for "how much terrain
geometry lives in the features."

### 5.2 kNN probes (DINOv2-style)

Mean-pooled, L2-normalized tile embeddings; cosine 20-NN classification on
(a) the corpus's own k=20 stratum labels (ready-made, zero cost) and (b) the
Iwahashi 15-group tile labels. No training at all. Runs on Task 2's own
committed lists (train tiles as the neighbour bank, test tiles as queries),
since the Iwahashi table is built over those and not over `pretrain.probe`'s
5k corpus-val probe set. Reuses `pretrain.probe`'s `neighbour_bank` and
`knn_classify` only; features come from the ADR-0012 extractor, not
`pretrain.probe.extract_features`, which is DEMAE-only (#184).

### 5.3 Depth sweep

Linear probe (§4.1 recipe, 1 seed) at layers 2, 4, 6, 8, 10, 12 of each
encoder. Answers two things at once: where each model's most linearly useful
features live (MAE features typically peak before the final layer; tokenizer
encoders may peak earlier still), and whether the DEMAE-vs-TerraMind gap is a
last-layer artifact. Plot: probe mIoU vs depth, one line per encoder.

### 5.4 Geometry comparison: CKA and mutual nearest neighbors

Linear CKA between DEMAE and TerraMind token representations on identical
tiles in the co-registered 30 m regime (per-layer CKA heatmap), plus
mutual-nearest-neighbor alignment on pooled embeddings (Platonic-style,
dimension-agnostic). These do not say who is better; they say whether the two
models learned similar or complementary geometry, which motivates the obvious
future-work line (concatenated features) that a reviewer will ask about.
Run the concatenation probe (DEMAE ⊕ TerraMind tokens -> linear probe) as one
extra row: if it beats both, the representations are complementary and that is
an interesting, honest finding.

## 6. Metrics and statistics

- **Dense (geomorphons):** mIoU and macro-F1 headline; per-class IoU; overall
  accuracy reported but never headlined; confusion matrices (the
  ridge/spur/shoulder confusions are the geomorphologically interesting ones);
  boundary-F1 at a 2 px tolerance as a secondary number, since landform
  boundaries are where segmentation quality is visible.
- **Tile-level (Iwahashi):** accuracy and macro-F1 over the 15 groups.
- **Regression probes:** R² per target.
- **Stratification:** all dense metrics also per collapsed ruggedness stratum
  (4 groups); a model that only classifies mountains well must be visible.
- **Statistics:** 3 seeds (mean ± std) for probe cells; paired bootstrap over
  test tiles (10k resamples) for the key pairs (B1 vs B4, B1 vs B6, B2 vs B7)
  with 95% CIs; CI-excludes-zero marked in tables. Seed spread and bootstrap
  CI reported separately.
- **Efficiency footnote:** params (frozen + trained) per arm; inline
  frozen-forward cost per arm stated so the compute story is honest.

## 7. Fairness checklist (verbatim into the writeup)

1. Identical tiles, labels, fraction draws, seeds, heads, schedules across
   frozen arms; head recipes tuned once on the neutral encoder (B6), frozen.
2. Native normalization per encoder; the altitude-passthrough ablation
   quantifies the confound instead of arguing about it.
3. GSD: primary regime is DEMAE-native and labeled as such; TerraMind gets a
   retrained head in its 10 m regime, with labels regenerated on that grid,
   before conclusions.
4. Capacity bracketed by ViT-B cells and the depth sweep; parameter counts in
   every table.
5. The tokenizer encoder is scored as what it is; B8 shows the bottleneck row;
   B9 (the intended representation model) is included when budget allows and
   foregrounded if stronger.
6. Geomorphon circularity stated plainly: the primary label measures
   representation quality of the input's own geometry; Iwahashi is the
   independent check; both must point the same way for C1 to be claimed.
7. Label-generation parameters committed; the L-sensitivity ablation shows the
   encoder ranking is not an artifact of one geomorphon scale.
8. Geographic geocell splits only; fraction draws by geocell; nested subsets.
9. Pre-registration: this memo is committed before the first head trains;
   C1-C3 defined above any result.

## 8. Compute budget

| Item | 4090-hours | Notes |
|---|---|---|
| Label generation (buffered re-reads + WhiteboxTools + Iwahashi zonal) | CPU/network only | local or a cheap CPU box |
| Linear-probe grid (arms x fractions x 3 seeds) | 1-2 | encoders inline, no cache (#184); re-priced by #191 |
| UperNet grid (B2/B5/B6 x fractions x 3 seeds) | 6-9 | the main line item |
| B7 UNet (fractions x 3 seeds) | 3-5 | |
| B3 fine-tune | 2-4 | 100% labels only |
| Ablations (B8, altitude-passthrough, 10 m regime, L-sensitivity rescore) | 2-3 | |
| Representation suite (5.1 to 5.4 + concatenation probe) | 1-2 | mostly CPU |
| B9 TerraMind backbone (optional) | 2-4 | only if budget remains |
| **Total** | **17-28 core (+2-4 optional)** | **~$6-10 at $0.34/hr** |

Combined with Task 1's 20-33 hours, the two-task program lands at roughly
$13-22, inside the < $50 envelope with the mandated slack.

## 9. Risks

- **Buffered re-read friction:** 27k windowed COG reads is a solved problem in
  this repo, but transatlantic latency is not; the Dockerfile cloud-burst path
  exists for exactly this. Fallback (ring masking) is specified and costs only
  eval area, not correctness.
- **Class collapse at 1% labels:** with few geocells some rare classes may
  vanish from training. Report which classes were present per fraction draw;
  do not silently re-draw until it looks nice (the draw is committed first).
- **Iwahashi label coarseness:** the 60% majority rule discards mixed tiles;
  if too many test tiles drop, relax to 50% and report both counts.
- **UperNet layer-choice sensitivity:** indices fixed at 3/6/9/12 and stated;
  the depth sweep doubles as the sensitivity check.
- **TerraMind extraction pitfalls:** same as Task 1 (diffusers pin, NaN
  handling, grid check); the verified code path is in
  `downstream-task-selection.md` §3.4.
- **The PANGAEA outcome** (B7 ties B3 at 100%): expected, pre-registered, and
  C2 is the claim that survives it.

## 10. Deliverables

1. `benchmark/` module extensions + committed label configs, fraction draws,
   and fingerprints; labels rebuildable byte-for-byte from committed inputs.
2. The core results table (arms x fractions, CIs marked) and four figures:
   the money plot (mIoU vs label fraction, one line per encoder + UNet);
   probe mIoU vs depth; the derivative-regression R² bar chart; a qualitative
   panel (tile hillshade, geomorphon truth, B1/B4/B7 predictions) on two
   rugged and one flat tile.
3. A results memo in `docs/research/` answering C1, C2, C3 with numbers, the
   fairness checklist, and the CKA/MNN geometry findings including the
   concatenation row.
4. The ADR-0012 extractor plus the probe harness: this is also the
   infrastructure any future task (bare-earth correction, Task 3) plugs into.

## 11. Key sources

Geomorphons (Jasiewicz & Stepinski 2013, Geomorphology; WhiteboxTools /
GRASS r.geomorphon implementations) · Iwahashi & Pike lineage 2018
(gisstar.gsi.go.jp/terrain2018, CC-BY 4.0; PEPS doi:10.1186/s40645-017-0157-2)
· Geomorpho90m (Sci Data 2020) as cross-reference · PANGAEA (arXiv 2412.04204,
frozen UperNet protocol, the baseline bar) · TerraMind (arXiv 2504.11171) and
TerraMesh (arXiv 2504.11172) · terratorch v1.2.10 source (extraction path,
normalization constants, `return_intermediates`) · MAE (arXiv 2111.06377,
linear-probe lineage) · DINOv2 (arXiv 2304.07193, kNN protocol) · CKA
(Kornblith et al., arXiv 1905.00414) · mutual-nearest-neighbor alignment
(arXiv 2405.07987) · GEO-Bench (arXiv 2306.03831) · Self-Supervised Masked DEM
Encoding (arXiv 2309.03367, the low-label DEM-segmentation template) ·
GeomorPM (doi:10.1080/13658816.2024.2414409, task-level comparability).
