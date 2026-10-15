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
  ImageNet-MAE features and TerraMind features, on the same tiles, labels,
  seeds, and heads. "DEMAE" here is the R5 release encoder and nothing else;
  "TerraMind" is whichever of the tokenizer encoder (B4/B5) and the backbone
  (B9) scores higher, so the claim is read against TerraMind's stronger
  encoder (#190). Thresholds are the register's.
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

Every encoder is named by its registry entry in `benchmark.arms.ENCODERS` and
pinned by md5 there; an unpinned encoder is unrunnable, so an arm whose pin
reads "to pin" is planned, not runnable, until its pin lands (#190).

| # | Arm | Encoder (registry name, md5) | Trained part | Fractions | Role |
|---|-----|------------------------------|--------------|-----------|------|
| B1 | DEMAE frozen + linear probe | `demae-r5`, the Phase-1 release ViT-B (ADR-0010), `3ec5ebc7…` | 1x1 conv on `output` | 1/10/100% | core, C1 |
| B2 | DEMAE frozen + UperNet | same | UperNet over 4 intermediate layers | 1/10/100% | core, C1 |
| B3 | DEMAE full fine-tune | `demae-r5`, all weights | encoder + UperNet | 100% | C3 |
| B4 | TerraMind frozen + linear probe | `terramind`, tokenizer ViT-B encoder, 768-d `feats` (pre-`quant_proj`), `66b7fbcb…` | identical linear probe | 1/10/100% | core, C1 |
| B5 | TerraMind frozen + UperNet | same, 4 intermediate layers via `return_intermediates=True` | identical UperNet | 1/10/100% | core, C1 |
| B6 | ImageNet-MAE frozen (linear + UperNet) | `imagenet-mae`, mmselfsup MAE ViT-B, 1-channel patch embed collapsed (the `pretrain.init` recipe), `91ecc034…` | identical heads | 1/10/100% | core, C1 |
| B7 | UNet from scratch | none | everything | 1/10/100% | C2, C3 |
| B8 | TerraMind 5-d latent probe | new entry: post-`quant_proj`, pre-FSQ, same weights and md5 as `terramind` | linear probe | 100% | ablation |
| B9 | TerraMind backbone, frozen (linear + UperNet) | new entry: `terramind_v1_base` with raw DEM input (`untok_dem@224`), to pin | identical heads | 1/10/100% | core, C1 |
| B1-R4 / B2-R4 | DEMAE R4, frozen | `demae-r4`, ImageNet-warm-started release ViT-B, `59e8346c…` | identical heads | 1/10/100% | extra row, never C1's number |
| B1-S / B2-S | DEMAE ViT-S, frozen | `demae-vits`, exported from R1, `a1e50e5c…` | identical heads | 1/10/100% | appendix, no bootstrap pair |
| B4-ALT / B5-ALT | TerraMind fed per-tile-standardized input | new entry: same weights and md5 as `terramind`, DEMAE's diet | identical heads | 100% | altitude-passthrough control (§2.1) |

All frozen rows run 3 seeds. B1/B2, B4/B5, B6 and B9 are the core grid. **R5
is C1's DEMAE number unconditionally**, the rule Task 1's register made R3: an
R4 win is reported prominently and never becomes C1's number. R4 earns its row
because it is the only one that separates ImageNet init plus DEM pretraining
from DEM pretraining alone. The ViT-S row has no ViT-S competitor (B6 is
ViT-B only), so it is a within-DEMAE scaling row, run at every fraction because
whether scale matters more at 1% labels than at 100% is something no other row
answers. B8 exists to show, in one row, why the 5-d pre-FSQ bottleneck is not
"TerraMind's representation" and would have been a strawman as the primary
target. B9 is the actual foundation-model encoder, which ingests raw DEM
directly; it is a planned arm, not a budget-permitting one, and C1 is read
against the stronger of B4/B5 and B9. The altitude-passthrough row and B8 are
registry entries of their own rather than flags, on the `terramind-bicubic`
precedent: results replace rows by arm, so a flag would let the control delete
the row it sits beside.

**Not arms.** The concatenation probe (R5 ⊕ TerraMind tokens) is a §5.4 result,
not a row of this table. GeomorPM's pretrained encoder is not benchmarked; §2.3
says why.

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
   ablation row feeding TerraMind per-tile-standardized input (B4-ALT/B5-ALT)
   measures the altitude-passthrough contribution directly. Report it next to
   B4/B5.
2. **GSD.** One regime: everything at native 30 m, 224x224, every ViT on a
   14x14 grid, heads byte-identical. There is no 10 m regime. TerraMind's
   10 m pixels are GLO-30 upsampled 3x (#138), so a 10 m regime adds a
   bandwidth shift and no information, and geomorphon labels regenerated on
   that grid would be computed from an interpolant of the same 30 m posts
   (#139). Task 1's pre-registered symmetric caveat stands in for it: a
   bandwidth shift, not a resolution handicap.
3. **Capacity.** Every C1 pair is ViT-B/16 against ViT-B/16, matched to
   within 5.8% in parameters (the table below), and the depth sweep (§5.3) is
   what checks C1 is not a last-layer artifact.
4. **Objective mismatch.** The tokenizer encoder was trained for reconstruction
   through a 5-d bottleneck; reconstruction pressure keeps geometry but has no
   reason to linearize *classes*. Landform probing is therefore close to a
   best-case test for MAE-style pretraining and a hard test for a tokenizer
   encoder; the writeup phrases conclusions as "under identical probes," and B9
   exists so TerraMind is also represented by the model IBM actually intends
   as a representation encoder.

### Capacity, in full (item 3)

This item used to hedge a 4x gap, "ViT-S 22M vs ViT-B 90M", the same stale
premise #122 removed from Task 1's §2.3. It is not the comparison being made:
every C1 pair is ViT-B/16 at 224 px on a 14x14 token grid, so capacity is one
table and the section's real subject is what is left.

| Encoder | Arms | Params | Role |
|---|---|---:|---|
| DEMAE ViT-B R5, the Phase-1 release encoder | B1, B2, B3 | 85,254,144 | C1's DEMAE number |
| DEMAE ViT-B R4, ImageNet warm start | B1-R4, B2-R4 | 85,254,144 | extra row |
| ImageNet-MAE ViT-B | B6 | 85,405,440 | C1 comparator |
| TerraMind DEM tokenizer ViT-B encoder | B4, B5, B4-ALT, B5-ALT, B8 | 90,172,416 | C1 comparator |
| TerraMind backbone `terramind_v1_base` | B9 | counted when pinned | C1 comparator |
| DEMAE ViT-S, exported from R1 | B1-S, B2-S | 21,393,408 | appendix |

Counts are `benchmark.arms.ENCODERS`' and are printed in every results table.
The largest C1 gap is TerraMind's 5.8% over R5, at identical depth, patch size
and token grid; B9's count joins the table with its pin. The live asymmetry is
pretraining data, and Task 1's §2.3 already measured it for the two encoders
both tasks share (matched optimization steps to 4.1%, TerraMind covering ~3x
the ground area at 10 m); it is cited, not restated.

The depth sweep's job is narrower than it was. With capacity matched, it is
not a capacity curve but the check that a C1 win is not an artifact of which
layer the linear probe reads. What a best-layer reversal (a comparator's best
layer beating DEMAE's best while DEMAE wins at the last layer) does to C1 is
the register's rule, not this section's.

**GeomorPM is not benchmarked.** It is the one published DEM-pretrained
encoder and technically viable frozen (#185): 14x14 tokens of 1024-d, all 24
blocks reachable. Joining would cost an ADR-0012 amendment (the contract fixes
12 layers), an extractor that bypasses the released `forward_encoder`'s token
shuffle, and an encoder-only load from a 4.66 GB Lightning checkpoint. What it
would buy is a number confounded three ways: 324.9M frozen parameters (3.81x
R5, ~11x its per-tile compute), a ViT-L ImageNet MAE init, and an unreleased
pretraining corpus that cannot be checked for overlap with this benchmark's
test geocells. A GeomorPM loss would be unattributable and a GeomorPM win
expected, so the row would print a number the writeup must disclaim on every
axis. Ruled out of Task 2's scope (#190), not deferred.

## 3. Labels and dataset

### 3.1 Primary labels: geomorphons

The label contract is [ADR-0014](../adr/0014-geomorphon-label-contract.md) (#188), and this section
only summarizes it. WhiteboxTools v2.4.0 `Geomorphons`, pinned by binary sha256, produces 10
classes: flat, peak, ridge, shoulder, spur, slope, hollow, footslope, valley, pit.

- **Parameters, fixed and committed:** outer search radius **L = 50 cells** (1.5 km at 30 m),
  skip 0, fdist 0, flatness threshold **1.0°**, all passed explicitly. These are **a measured
  choice on GLO-30, not a literature default.** L = 50 is WBT's default and 1.0° is GRASS's.
  #188's sweep found that thresholds under 0.5° erase flat, shoulder and footslope on GLO-30's
  noise, and that 1.0° is the only tested value that keeps all ten classes while holding the
  flattest ruggedness band under half flat. They go in the dataset card verbatim because
  geomorphon maps are parameter-sensitive.
- **Spacing:** WBT reads the unresampled tile array with a projected-CRS header carrying the
  tile's WGS84 ground spacing (dx, dy). It never sees EPSG:4326, where it reads degrees as metres
  and loses three classes (#187).
- **Parameter-sensitivity ablation (CPU-only, cheap):** regenerate test-set labels at
  **L = 25 and L = 100**, and at **flatness 0.75° and 1.25°**, one parameter at a time, and
  re-score the *already-trained* B1/B4 probes against them. This shows whether the encoder ranking
  is an artifact of one label setting; no GPU retraining is needed. Flatness is in the set because
  the labels are more sensitive to it than to L.
- **The edge problem, solved at generation time:** a geomorphon at pixel p needs terrain L·dy
  metres away, so labels computed on a bare 224 px tile are wrong over half the tile (#187). Fix:
  re-read each benchmark tile from the source COGs with **L rows and ⌈L·dy/dx⌉ columns** of buffer
  on each side (the column buffer is 58 px at the median and 224 px at worst for L = 50), compute
  geomorphons on the buffer, and crop the centre 224×224. A test tile is read once at the widest
  variant's buffer. Cost: ~1.5 h of laptop wall-clock, network-bound (#187). Label rasters are
  uint8, stored like shards (gitignored, rebuildable, fingerprinted). There is no ring-mask
  fallback.
- **Ignore label (255):** ocean pixels (exact-zero fill) and pixels whose search window reaches
  data missing from the GLO-30 release are excluded from loss, metrics and class weights. Verified
  ocean gaps read as 0 m, and the two polar tiles at E179 wrap across the antimeridian. The tile
  lists stay identical to Task 1's.
- **Class imbalance is structural:** slope and flat dominate globally; pit and peak are rare.
  Handled in the loss (inverse-sqrt-frequency class weights, computed once over the full train
  list's non-ignore pixels and used unchanged at every label fraction) and in the metrics (macro-F1
  and mIoU as headline numbers, never overall accuracy alone), and reported per ruggedness stratum,
  where the class mix shifts drastically. Steep bands contain no flat or footslope at all.

### 3.2 Secondary labels: Iwahashi & Pike 2018 (independence check)

Global terrain-classification GeoTIFF (gisstar.gsi.go.jp/terrain2018; licence
as §3.4 states it), a rasterization of the authors' polygons at 7.5″ (~232 m at
the equator) holding the published **15 terrain groups** directly. The 15 are
not a lookup from the 40 classes: 40 → 12 by table, then mountains and hills
split a/b by texture (#186). A 224 px GLO-30 tile spans ~30 label rows and
~30–60 label columns (more poleward, where a tile is wider in degrees), far too
coarse for dense segmentation, so this is a **tile-level task**: majority group
over the tile footprint = the tile's label, predicted from the pooled embedding
(mean-pooled patch tokens, linear classifier; kNN variant in §5.2). It is
trained **at 100% only** (#189): the 1% rung's 200 tiles leave about 90 that
pass the 60% rule, under 6 per group across 15 groups.

**The tile-label rule (#193), fixed before any probe is scored:**

- **Fraction:** majority-group area over the **whole footprint**, area-weighted
  per label pixel; nodata (255) and the unlegended value 0 count against the
  majority.
- **Threshold:** **≥ 60% is primary**, and C3 is read there only. **≥ 50% is a
  pre-registered sensitivity row**, reported beside it and never used to decide
  C3. The same rule labels train, val and test.
- **Counts** (test, after #150's land filter): 1,985 of 4,523 tiles at 60%,
  2,712 at 50%. 622 can never be labelled: Antarctica lies south of the map's
  59.46° S edge, and Arctic ice reads 0. The 60% rule then drops more mixed
  rugged tiles, so survival by band is roughly even (R1 48%, R2 40%, R3 45%,
  R4 42%). **No re-weighting**: every arm is scored on the same surviving
  tiles, so the loss cannot favour one; it limits coverage, which the dataset
  card states (no Antarctica, no polar ice).
- **Grouping:** all **15 groups** are the headline; the 5-way overview level
  (bedrock mountain / hills / large highland slope / plateau-terrace / plain)
  is a secondary table. The a/b subgroups are **not pooled**: pooling rescues
  only 1b and 2b, not the thinnest groups (4, 5, 7, at 10–29 test tiles), and
  the a/b texture split is where encoders are most likely to differ.

Because the labels derive from a *different* DEM lineage (MERIT: SRTM, AW3D,
Viewfinder Panoramas; no TanDEM-X) and human calibration, this is the answer to
"you trained on the label-generating function": nothing in the Iwahashi map is
a function of GLO-30. Independent is not uncorrelated: both DEMs sample the
same ground, and the claim is only that the label is not a function of the
encoder's input.

### 3.3 Tiles, splits, fractions

Same committed lists as Task 1 (20k train / 2k val / 5k test, geographic
geocell splits, stratified over the k=20 strata). No features carry over from Task 1:
every Task 1 encoder input is an A1-filled voided tile, so no clean-tile
feature was ever computed, and Task 1 runs its encoders inline with no cache
(#121). What carries over is the ADR-0012 extractor, which Task 2 also runs
inline (#184).

**Label fractions: the geocell ladder (#189).** A real low-label regime leaves
whole regions unlabelled, not a thin uniform scatter, so labels are withheld by
10° geocell. Task 1's label-efficiency draw (1k ⊂ 5k ⊂ 20k, by tile,
stratified) is not reused. What #189 measured on the head-train list:

- It spans **376 train geocells** at 1–175 tiles each (median 51), so a literal
  "1% of geocells" is about 4 cells holding **25–506 tiles** (median 210;
  103–331 for the middle 90% of 5,000 simulated draws). Ten percent is 38 cells
  and 1,710–2,353 tiles. The rung's data volume would be draw luck.
- **No class vanishes at 1%.** In every simulated draw all ten classes were
  present, with ≥129k pit and ≥137k peak pixels at the 5th percentile. A
  geomorphon is a per-pixel local form; pit and peak occur in every ruggedness
  band. The classes that do drop out of whole tiles are flat, shoulder and
  footslope on steep ground, and a 1% draw reaches all four bands 99.6% of the
  time. (Estimated by resampling #188's 50 labelled tiles within each band, so
  within-cell correlation is not modelled.)
- Terrain coverage is what thins: a 1% draw covers 18 of 20 strata at the
  median, 14 at the 5th percentile, 4 at worst.

The rule:

- **Rungs are tile budgets, cells are the unit.** **1% (200 tiles) ⊂ 10%
  (2,000 tiles) ⊂ 100% (20,000 tiles)**, written that way rather than "1% of
  geocells". Geocells are taken whole in keyed order until the next one would
  pass the budget; that next cell contributes just enough of its own tiles, in
  keyed order, to land on the budget exactly. A rung therefore holds at most one
  partial cell, which is still one contiguous region, and its id is recorded.
  Stopping at the first cell that reaches the budget instead would overshoot to
  202–292 tiles (up to 374).
- **Nested by construction, asserted anyway.** Each ladder is one walk over
  tiles (cell rank, then tile rank within the cell), and a rung is a prefix of it.
- **One ladder per seed.** Ladders 0, 1 and 2 go with seeds 0, 1 and 2, and every
  arm trains on the same three, so pairs stay paired. §6's seed spread at 1% and
  10% then includes which regions were labelled, which the paired bootstrap over
  test tiles cannot see. At 100% every ladder is the whole list.
- **Not stratified.** This departs from Task 1's rule that every draw is
  proportional over the k=20 strata, on purpose: four or five cells cannot be
  proportional over 20 strata, and choosing cells to minimise stratum shortfall
  (as the test-geocell selection does) would be picking the best-looking regions
  ahead of time. Each draw's strata and band coverage is printed beside it
  instead.
- **Order** is `tie_rank` keyed with ladder *i*'s draw seed: over geocell ids,
  then over tile ids within a cell. No RNG, so a redraw is byte-identical.
- **Committed first, never re-drawn.** Each draw also reports its per-class
  train-pixel counts. There is no merge-a-class or change-the-fraction rule,
  because no simulated draw needed one.

### 3.4 Committed artifacts

`benchmark/` additions: geomorphon config (WBT binary sha256, L, flatness, skip, fdist, the
buffer formula, the spacing rule, the ignore rule, and the frozen class weights), label
fingerprints per ADR-0014 (per-tile label and input-window hashes, a set fingerprint per
variant, a config hash), the Iwahashi rebuild script with the source zip's md5
(`5ae98dc7782c13384a1df4d8d609418f`) and a fingerprint of the tile-label table
it produces, the three geocell ladders (§3.3; one file each, md5 in the run
record), and the dataset card covering licenses
(WhiteboxTools labels derived from GLO-30, attribution per Copernicus licence).

**The Iwahashi tile labels themselves are not committed (#193).** The
distributor's grant is hedged and the upstream MERIT DEM is CC BY-NC 4.0 /
ODbL 1.0, whose share-alike clause would reach a committed derived table if it
applies at all. Shipping the script instead sidesteps that at no cost: the zip
downloads without a login and the table rebuilds in about a minute. The dataset
card quotes the distributor's licence sentence **verbatim** rather than stating
"CC BY 4.0" as fact, carries the Iwahashi et al. (2018) citation, states
MERIT's upstream terms beside it, and notes that the GeoTIFF is Hengl's
rasterization of the polygons.

## 4. Training setup

### 4.1 Linear probes (B1 / B4 / B6 / B8 / B9 and the B1/B4 extra rows): the strict instrument

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

### 4.2 UperNet probes (B2 / B5 / B6 / B9 and the B2/B5 extra rows): the PANGAEA instrument

UperNet over 4 intermediate layers (indices 3, 6, 9, 12 of 12 for every
encoder, stated explicitly since layer choice is a known sensitivity), frozen
encoder, ~30M-param head, weighted CE, AdamW lr 6e-4, 40 epochs, batch 64.
The encoder runs inline and the head reads the extractor's `intermediates` at
those indices; nothing is cached (#184). A cache would cost 32.5 GB per ViT-B
encoder (27k tiles x 4 layers x 196 tokens x 768 x fp16), ~146 GB for five
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

1. Generate and fingerprint geomorphon labels per ADR-0014 (CPU + network, no GPU):
   anisotropic buffer re-read, WhiteboxTools pass, ignore mask, class weights from the train
   list, sensitivity variants (L 25/100, flatness 0.75°/1.25°) for test only, and a
   byte-identical rebuild check on a fixed sample.
2. Build the Iwahashi tile-label table (one global GeoTIFF, area-weighted
   majority per tile footprint under §3.2's rule; CPU) and commit its
   fingerprint, not the table.
3. Pin every frozen encoder in the ADR-0012 extractor registry (no feature
   caching step: encoders run inline, #184).
4. Tune both heads on B6/val; freeze the recipes.
5. Core grid: B1/B2/B4/B5/B6/B9 x {1%, 10%, 100%} x seeds; B7 x fractions;
   the B1-R4/B2-R4 and B1-S/B2-S extra rows x fractions.
6. B3 fine-tune; B8 and B4-ALT/B5-ALT at 100%.
7. Representation suite (§5): one inline pass per analysis over a fixed tile
   set.

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

Linear probe (§4.1 recipe, 1 seed) at blocks 2, 4, 6, 8, 10, 12 of each
encoder, read from the extractor's `intermediates`, plus DEMAE's `output` as
its own point: ADR-0012 makes DEMAE's `intermediates[11]` and its `output`
different tensors (the latter is post-norm), so block 12 of the sweep is not
the tensor B1 reads, and the sweep has to show both. Its job, with capacity
matched (§2, item 3), is to check that a C1 win is not a last-layer artifact;
it also shows where each model's most linearly useful features live (MAE
features typically peak before the final layer; tokenizer encoders may peak
earlier still). What a best-layer reversal does to C1 is the register's rule.
Plot: probe mIoU vs depth, one line per encoder.

### 5.4 Geometry comparison: CKA and mutual nearest neighbors

Linear CKA between DEMAE and TerraMind token representations on identical
tiles in the co-registered 30 m regime (per-layer CKA heatmap), plus
mutual-nearest-neighbor alignment on pooled embeddings (Platonic-style,
dimension-agnostic). These do not say who is better; they say whether the two
models learned similar or complementary geometry, which motivates the obvious
future-work line (concatenated features) that a reviewer will ask about.
Run the concatenation probe (R5 ⊕ TerraMind tokenizer tokens -> linear probe,
100% labels, 3 seeds) as a result of this section, not as an arm: it sits
outside the core table and outside C1 (#190). If it beats both, the
representations are complementary and that is an interesting, honest
finding.

## 6. Metrics and statistics

- **Dense (geomorphons):** mIoU and macro-F1 headline; per-class IoU; overall
  accuracy reported but never headlined; confusion matrices (the
  ridge/spur/shoulder confusions are the geomorphologically interesting ones);
  boundary-F1 at a 2 px tolerance as a secondary number, since landform
  boundaries are where segmentation quality is visible.
- **Tile-level (Iwahashi):** accuracy and macro-F1 over the 15 groups at the
  60% threshold; the 5-way overview level and the 50% sensitivity row reported
  beside it (§3.2). Also per collapsed ruggedness stratum.
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
3. GSD: one regime, native 30 m, labelled DEMAE-native. TerraMind's 10 m
   pixels are GLO-30 upsampled 3x (#138), so a 10 m regime would add a
   bandwidth shift and no information. Task 1's symmetric caveat stands in for
   it, and no 10 m head is trained.
4. Capacity: every C1 pair is ViT-B/16 on a 14x14 grid, within 5.8% in
   parameters, with parameter counts in every table. The depth sweep shows
   whether C1 is a last-layer artifact. GeomorPM (324.9M, ViT-L ImageNet init,
   unreleased corpus) is not benchmarked, for the reasons given in §2.
5. The tokenizer encoder is scored as what it is. B8 shows the 5-d bottleneck
   row, B9 (the intended representation model) is a planned arm, and C1 is
   read against the stronger of the two TerraMind encoders.
6. Geomorphon circularity stated plainly: the primary label measures
   representation quality of the input's own geometry; Iwahashi is the
   independent check; both must point the same way for C1 to be claimed.
7. Label-generation parameters committed (ADR-0014); the L- and flatness-sensitivity
   ablations show whether the encoder ranking is an artifact of one geomorphon setting.
8. Geographic geocell splits only; label fractions by geocell, as exact tile
   budgets, one nested ladder per seed (§3.3).
9. Pre-registration: this memo is committed before the first head trains;
   C1-C3 defined above any result.

## 8. Compute budget

| Item | 4090-hours | Notes |
|---|---|---|
| Label generation (buffered re-reads + WhiteboxTools + Iwahashi zonal) | CPU/network only | local or a cheap CPU box |
| Linear-probe grid (arms x fractions x 3 seeds) | 1-2 | encoders inline, no cache (#184); re-priced by #191 |
| UperNet grid (B2/B5/B6/B9 + B2-R4/B2-S x fractions x 3 seeds) | 6-9 | the main line item |
| B7 UNet (fractions x 3 seeds) | 3-5 | |
| B3 fine-tune | 2-4 | 100% labels only |
| Ablations (B8, B4-ALT/B5-ALT at 100%, L- and flatness-sensitivity rescore) | 2-3 | no 10 m regime (#190) |
| Representation suite (5.1 to 5.4 + concatenation probe) | 1-2 | mostly CPU |
| B9 TerraMind backbone | 2-4 | a planned arm (#190); its linear and UperNet cells are in the grid lines above once re-priced |
| **Total** | **19-32** | **~$6-11 at $0.34/hr**; every line re-priced by #191 |

Combined with Task 1's 20-33 hours, the two-task program lands at roughly
$13-22, inside the < $50 envelope with the mandated slack.

## 9. Risks

- **Buffered re-read friction:** #187 measured it: ~1.5 h of laptop wall-clock over a home
  network, at 16 threads. There is no ring-mask fallback (ADR-0014). The remaining risk is
  upstream drift in the public GLO-30 bucket, which the per-tile input hash detects.
- **Draw variance at 1% labels:** #189 found no class vanishes at 1% (pit and
  peak are in every band), so class collapse is not the risk. The risk is which
  ~4 regions a draw labels: a 1% rung covers as few as 14 of 20 strata at the
  5th percentile. Handled in §3.3: exact tile budgets remove volume luck, one
  ladder per seed puts region luck into the reported spread, and each draw's
  coverage and per-class pixel counts are printed. A committed draw is never
  re-drawn until it looks nice.
- **Iwahashi label coarseness:** the 60% majority rule keeps 1,985 of 4,523
  clean test tiles, and five groups have under 30 test tiles each. Settled in
  §3.2 before scoring (#193): 60% is primary and 50% a fixed sensitivity row,
  so there is no relax-if-too-many-drop choice left to make after looking.
  Thin groups widen the macro-F1 CI; the paired bootstrap carries that rather
  than a pooling choice hiding it.
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
