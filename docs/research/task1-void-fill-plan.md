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
  the representation claim, and the fairest instrument for it. **"DEMAE" here
  means the R5 ViT-B release encoder, fixed before results** — the R4
  continual-init arm is a pre-registered secondary and never C1's number (§2.3).
- **C2 (the model is useful):** fine-tuned DEMAE beats classical interpolation
  (the operational status quo) in metres, slope, and **seam continuity** on
  rugged terrain at every void size, and stays at least even on **flat terrain**,
  where classical fill is already at the corpus's resolution floor. The parity
  carve-out is by ruggedness, not by void size: p119 measured classical fill at
  0.87 m MAE on small voids and 1.83 m on large ones in the flattest quartile —
  against 12.7 m and 38.6 m in the roughest — so "small voids where splines are
  strong" was the wrong exemption to grant
  ([`p119-fill-headroom.md`](p119-fill-headroom.md)). **"Seam continuity"
  replaces the original "drainage metrics" clause**, which fell with the
  hydrological metrics: a 224 px window has no upstream context, so D8
  accumulation near a tile edge is an artifact of the crop. The operational
  argument is not dropped, only re-sited — an operator's complaint about a bad
  fill is that it does not join up with the terrain around it, and §5's seam gap
  measures exactly that ([`p123-metric-discrimination.md`](p123-metric-discrimination.md)).
- **C3 (label efficiency):** DEMAE-frozen reaches a given RMSE with far fewer
  head-training tiles than the UNet-from-scratch baseline (the 1k / 5k / 20k
  sweep). This is the foundation-model value proposition. As with C1, the curve
  is **R5's**; R4 runs at 20k only and does not enter the sweep, so C3 is one
  DEMAE curve rather than two (§2.3).
- **C4 (the ranking survives real geometry):** the *ordering* of arms on real
  SRTM void shapes, **within a ruggedness band**, does not contradict the
  ordering on synthetic masks. This is deliberately an ordering claim and **not**
  a metric-delta claim: the real-void family is drawn at its natural size
  distribution and the synthetic families are bucket-balanced, so a difference
  in metres between the two would confound void *shape* with void *size* (§3.3).
  Ordering is confound-free because every arm is scored on the identical
  instance set. C4 fails if an arm pair swaps rank between the synthetic and
  real-void rows within the same band — which would put C1 in question, since
  C1's comparison is made entirely on synthetic masks.

If C2 partially fails (e.g., Diff-DEM wins on texture), that is reported, not
hidden; PANGAEA-style candor is part of the deliverable. C1 and C3 are the
claims DEMAE's worth actually rests on, because they isolate the pretrained
representation from decoder capacity.

## 1. Comparison arms

| # | Arm | Encoder | Trained part | Input convention |
|---|-----|---------|--------------|------------------|
| A1 | Bilinear/spline fill | none | none | classical `gdal_fillnodata`-class interpolation; also rasterio `fillnodata`, and a thin-plate-spline variant |
| A2 | DEMAE zero-shot | DEMAE frozen (R5 ViT-B; R4 ViT-B secondary) | none | native MAE inpainting: void-touching patches dropped, decoder reconstructs, non-void pixels composited back |
| A3 | DEMAE fine-tuned | DEMAE, all weights (R5 ViT-B; R4 ViT-B secondary) | encoder + decoder | native token dropping on void-shaped masks (§4.2) |
| A4 | DEMAE frozen + head | DEMAE frozen (R5 ViT-B; R4 ViT-B secondary; ViT-S as the §2.3 appendix row) | head only | voided tile pre-filled by A1 + void mask channel at the head (§2.4) |
| A5 | TerraMind frozen + head | TerraMind DEM tokenizer ViT-B encoder, 768-d `feats` (pre-`quant_proj`), frozen | head only (identical to A4's) | same pre-filled input, TerraMind's own normalization (§2.1) |
| A6 | ImageNet-MAE frozen + head | timm/MAE ViT-B frozen (ViT-S as the §2.3 appendix row), 1-channel patch embed collapsed (same recipe as `pretrain.init`) | head only (identical) | same pre-filled input |
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
  NaN-free. **Three different DEM constant pairs ship across IBM's artifacts** —
  651.663/928.168 in the dataset loader and 435.726/560.326 in the v0.1
  tokenizer, besides the v1 pair above ([`p138-terramind-corpus.md`](p138-terramind-corpus.md)).
  The v1 pair is the one that matches the v1 checkpoint this plan benchmarks;
  the build asserts it by name rather than by value, since a silent swap is a
  wrong-diet arm that still runs.

**Neither normalization is applied before the extractor** *(amended 2026-08-11 —
[#137](https://github.com/JStoschek/DEMAE/issues/137),
[ADR-0012](../adr/0012-frozen-feature-extractor-contract.md))*. The extractor
takes **raw int16 metres plus a valid-pixel mask** (valid = land ∧ ¬void) and
applies the arm's declared normalization itself, so the pretraining loader's
per-tile-standardized output is not a thing it can be handed. p118 §7 measured
why that matters: the TerraMind encoder accepts either normalization silently and
returns a well-shaped, plausible `[1, 768, 14, 14]` from an input distribution it
has never seen. The mask is also what makes "non-void pixels only" one code path
rather than a convention — and Task 2 reuses it unchanged by passing a land-only
mask.

Each encoder at its best is the fair default. One ablation row: TerraMind fed
per-tile-standardized input, to measure how much of its representation is
absolute-altitude passthrough. The head sees each encoder's features as-is
(LayerNorm at the head input equalizes scale), and **loss and metrics are always
computed in metres after inverting DEMAE's per-tile transform with the non-void
stats**, so no arm is scored in its own normalized space.

### 2.2 GSD: primary regime at 30 m, secondary at 10 m

TerraMind's DEM pixels were GLO-30 bilinearly resampled to the 10 m Sentinel-2
grid (TerraMesh §3.2 — the sources write "Copernicus DEM 30m" / "Copernicus
WorldDEM-30", never "GLO-30", so quote them that way in the writeup); DEMAE's
are native 30 m. The source is single and unblended — no FABDEM, no NASADEM, no
DSM/DEM switch relative to DEMAE ([`p138-terramind-corpus.md`](p138-terramind-corpus.md)).
Decisions:

- **Primary regime (all arms): 224x224 at native 30 m.** The task is defined
  operationally on GLO-30, DEMAE's home turf and honestly labeled as such;
  TerraMind runs out-of-distribution here and the writeup says so. Both ViTs
  produce a 14x14 token grid, so the head is byte-identical. **Measured, and it
  co-registers patch-for-patch:** TerraMind's DEM tokenizer was trained at 256 px
  and its `pos_emb` is natively `[1, 768, 16, 16]`
  ([`p118-terramind-encoder-probe.md`](p118-terramind-encoder-probe.md) §6),
  now double-sourced by terratorch's `input_size=256` independently of that
  checkpoint probe. It **strengthens ADR-0011**: the tokenizer was itself fed
  random 256 px crops of a *larger* 264 px corpus tile, so cropping — not
  resizing — is the native relationship between its inputs and its positional
  grid ([`p138-terramind-corpus.md`](p138-terramind-corpus.md)). The
  next bullet is how that premise is made to hold without costing TerraMind
  anything; it settles [#133](https://github.com/JStoschek/DEMAE/issues/133) and
  supersedes the provisional text this bullet carried while #133 was open.
- **TerraMind's position embedding is re-derived at 14x14, centre-anchored, not
  bicubically resized** (ADR-0011, [#133](https://github.com/JStoschek/DEMAE/issues/133)).
  The tokenizer was trained at 256 px, so `ViTEncoder.pos_emb` is natively
  `[1, 768, 16, 16]`, and terratorch's `forward` resizes it with
  `F.interpolate(..., mode="bicubic")` on every 224 px pass. That resize preserves
  *normalized* position and therefore stretches the token index spacing to
  **1.1429** where a crop's spacing is exactly **1.0** — the drift accumulates past
  a full patch width by the far edge, and only **20 of 196** tokens still rank their
  own correct embedding nearest. The embedding is not learned
  (`sincos_pos_emb=True, learnable_pos_emb=False`), so it is a closed-form function
  of integer grid index and can be re-derived exactly. A 224 px tile at 30 m *is* a
  crop of a 256 px scene at 30 m, so the faithful embedding is the crop, not the
  resize. We use the **centre** anchor, `pos_emb[:, :, 1:15, 1:15]` — equivalently
  indices 1-14 — which removes the stretch without also introducing a translation.
  This makes both encoders in-distribution at 224 px simultaneously: TerraMind sees
  indices 1-14, a subset of its trained 0-15, and DEMAE sees its own native 0-13.
  Neither is handicapped, so §2.2 no longer owes the reader a caveat here.
  Distinct from the secondary regime below, which is about GSD, not grid.
- **Secondary regime (A5 only): TerraMind-native check.** The same geographic
  windows bilinearly resampled to 10 m, center-cropped to 224 px (covering
  2.24 km instead of 6.72 km), TerraMind + head retrained. If TerraMind-at-10m
  still loses A4 vs A5, the GSD objection is closed. Run this once, at the 20k
  training size only.
- **Ablation row (A5 only, 20k training size): bicubic vs re-derived `pos_emb`.**
  One row reporting A5's metrics under terratorch's default resize against the
  centre-anchored re-derivation, so the ADR-0011 intervention is evidenced in the
  results rather than asserted in prose. Pre-registered here, with the drop condition
  fixed in ADR-0011 before any number existed — and **the row stays**:
  [#134](https://github.com/JStoschek/DEMAE/issues/134)'s materiality check measured a
  mean per-token cosine of **0.9708** between the two feature sets over 300 tiles and
  58,800 tokens, below the 0.99 that would have made the choice immaterial
  ([`p134-posemb-materiality.md`](p134-posemb-materiality.md)). Two constraints on how
  the row is reported come from that run. It is a **tail, not a shift** — the median
  token is at 0.993 and the 1st percentile at 0.525 — and linear CKA is a kinder 0.981,
  so a pooled or rotation-tolerant readout would understate the effect. Report it on
  this section's positional head, not on a summary statistic of the features.
- **Not considered: moving the primary regime to 256 px.** It looks symmetric and is
  not. Both encoders use fixed sin-cos, so a 256 px regime would give TerraMind its
  trained 0-15 while DEMAE saw indices **14 and 15 for the first time** — an
  extrapolation beyond its training range that re-derivation cannot fix, because you
  cannot crop into indices that do not exist. It would also retile the corpus to
  7.68 km windows, reopening the frozen Phase-0 tiling, strata and split. This is
  distinct from the out-of-scope secondary 10 m regime, which is about GSD, not grid.

### 2.3 Capacity and pretraining data: both near-matched

This section used to hedge a 4x capacity gap — "ViT-S (22M) vs ViT-B (90M)" —
and spent itself on mitigations for it. That is not the comparison being made.
Both sides of the primary regime are ViT-B/16 at 224 px on a 14x14 token grid,
so capacity is reported here in one table and the section's real subject is what
is left.

| Encoder | Params | Source |
|---|---|---|
| DEMAE ViT-B — the Phase-1 release encoder | **85,254,144** | built from `Architecture(encoder_dim=768, encoder_depth=12)`; matches the 149-tensor export check in [`p88-r4-r5-readout.md`](p88-r4-r5-readout.md) §4 |
| TerraMind DEM tokenizer ViT-B encoder | **90,172,416** | [`p118-terramind-encoder-probe.md`](p118-terramind-encoder-probe.md) |
| DEMAE ViT-S — appendix row only, below | 21,393,408 | same builder at 384-d |

**A 5.8% difference**, at identical depth, patch size and token grid.
TerraMind's extra 4.92M is *not* attributed to a module here: terratorch was not
installed in any live worktree when this was written and the attribution did not
justify a reinstall. It is reported, not explained.

**The live asymmetry is pretraining data — and it is far smaller than a patch
count suggests.**

| | DEMAE | TerraMind DEM tokenizer |
|---|---:|---:|
| Tiles per epoch | 473,088 (231 steps x 2048) | 9,089,536 |
| Epochs | 400 | 20 |
| Effective batch size | 2048 | 2048 |
| Optimizer steps | 92,400 | 88,765 (derived) |
| **Tile presentations** | **189.2M** | **181.8M** |
| Tile footprint | 224 px @ 30 m = 45.2 km² | 264 px @ 10 m = 6.97 km² |
| **Ground area covered** | **21.4M km²** | **63.4M km²** |

All figures on the TerraMind side are sourced in
[`p138-terramind-corpus.md`](p138-terramind-corpus.md); they were provisional
until it landed and are not any more.

Two things follow, and both cut against the obvious reading. **Optimization
compute is matched to within 5%** — this was never a compute-scale confound.
DEMAE sees **4.1% more** presentations, and the match is tighter than a
presentation count alone would show: both runs used an effective batch of
**2048**, so the step counts match to the same 4.1%. Read it as *matched sample
exposure and step count* and nothing more — the objectives differ (masked
autoencoder against an FSQ autoencoder with a diffusion decoder), no source
gives FLOPs for either side, and so no FLOP claim is made.

And the data gap is **~3x in ground area, not the ~19x the tile counts imply**
(the tile ratio is 19.2x exactly), because TerraMind's DEM tiles are resampled
onto the 10 m Sentinel-2 grid and therefore cover 6.5x less ground each. Since
TerraMesh's DEM modality *is* Copernicus GLO-30 bilinearly resampled (§2.2),
that ~3x is more ground of the **same product at the same true resolution** —
not more information, and not a finer view of it. TerraMesh singles DEM out for
*bilinear* resampling where Sentinel-2's own bands get nearest, so the 3x
upsampling is deliberate smoothing and demonstrably adds no high-frequency
content: "not more information" is affirmatively supported, not merely
unfalsified.

**Where the ~3x comes from, and the one correction that applies to a single
side.** 63.4M / 21.4M = **2.97x** on stated inputs. TerraMesh's own sources
imply within-corpus redundancy that this does not price — MajorTOM's 10.68 km
cells sit on a 10 km lattice (1.111x), and SSL4EO contributes four seasons per
location, which for a static DEM are exact duplicates — putting unique ground
nearer 55M km² and the ratio nearer **2.6x**. TerraMesh states **no**
deduplication or minimum-separation policy. DEMAE's side needs no such
correction: ADR-0001 grids each COG into a **non-overlapping** window lattice,
so its 21.4M km² is already unique ground. The honest claim is therefore
**2.6-3.0x**, and this plan prints **~3x** — the stated-inputs figure, which is
the one that does not require trusting an estimate. The earlier "~2.8x" was
right by accident: a footprint 6% too small (256 px is the tokenizer's *crop*;
the corpus tile is 264 px) cancelling an omitted overlap correction.

**The paper says 100 epochs, and a reader will check.** [TMind] §8.1 gives 100
epochs for its tokenizers, but that figure is the S-2/S-1 tokenizers'. The
per-modality training records shipped verbatim in terratorch's
`tokenizer_register.py` show `epochs=20` for `terramind_v1_tokenizer_dem`
(alongside `lulc` and `ndvi`) against `epochs=100` for `s2l2a`, `s1rtc` and
`s1grd`; IBM's own DEM tokenizer model card agrees — "pre-trained for 20 epochs
on nine million DEM images". Taking the paper at its word would have inflated
TerraMind's budget 5x and destroyed a claim that is in fact correct. Cite the
per-modality evidence wherever this number is stated, or it reads as a
misreading of the paper.

**The caveat is pre-registered, and it runs in both directions.** A data
advantage to TerraMind is only ever cited as an explanation of a DEMAE *loss*,
which makes it an escape hatch unless it is also priced as a discount on a DEMAE
*win*. It is stated once, here, before any number exists: if DEMAE wins C1 on
~3x less ground of the same product at matched compute, that is a stronger
result than a bare win, and if DEMAE loses, this is the one pre-registered
alternative explanation. Reaching for it afterwards is not available.

**ViT-S is one appendix row, not a capacity bracket.** A4-S and A6-S at 20k, one
seed each, answering "did the extra 4x buy anything" under the frozen probe.
No claim in §0 depends on it and it is not a co-primary: it is cut from A2/A3
and from the label-efficiency sweep, and it has no TerraMind counterpart at all
(no ViT-S DEM tokenizer is published), so the row is a DEMAE-vs-ImageNet-MAE
capacity observation rather than a three-way. Its price is why it is one row:
in GPU minutes ViT-S is ~$0.15, but at full seeds it would have been **16 of the
48 scored eval passes** this plan carried before this section was rewritten — a
third of §7's largest line — plus eight extra rows in a table whose
combinatorics are already the binding constraint. Cut to two single-seed cells,
it costs 2 passes and keeps the finding.

**The R4 continual-init arm stays, on a completely different justification.**
Its old one — "a same-capacity DEMAE-vs-TerraMind cell" — is gone, because R5 is
already same-capacity. What replaces it is a result from Phase 1:
[`p88-r4-r5-readout.md`](p88-r4-r5-readout.md) found that R4 and R5 are the one
pair where DEMAE's two instruments genuinely oppose — R5 out-*represents* R4 at
all 19 matched probe points, while R4 ends **0.56% better on masked-reconstruction
val loss**. Void filling is the reconstruction-flavoured downstream task, so
whether that disagreement transfers is a real, pre-registered question, and this
benchmark is the instrument for it. R4 runs in A2/A3/A4 at 20k.

- **Build precondition, not a detail.** R4 and R5 do **not** share a `pos_embed`
  axis order — R5 is `sincos2d`, R4 is `sincos2d_rowfirst` (P102), a difference
  measured as 6.7e-7 against 2.0 on the same tensor. `pos_embed` is registered
  non-persistent and is absent from `state_dict()`, so a `strict=False` load
  silently succeeds with the wrong embedding. Each arm's `pos_embed` is
  materialized from **its own** run config, and the load is asserted strict.
  "Running both costs minutes" is true in GPU minutes and false in correctness
  risk.

**Which DEMAE is the headline: R5, unconditionally.** C1 and C3 are evaluated on
the R5 release encoder alone; R4 is a pre-registered secondary row at 20k only,
and does not enter the label-efficiency sweep. If R4 beats R5 materially at void
filling, that is reported prominently as a finding about init transfer — and
explicitly as a challenge to Phase 1's probe-based release criterion — but it
does not become C1's number. Letting the winner claim the headline would leave
two DEMAE identities and a free choice made after seeing both, which is exactly
the forking path pre-registration exists to close.

**The plan names the encoder by property, never by file:** the Phase-1 release
ViT-B encoder under the ADR-0010 recipe, with init provenance recorded. R5 is
that encoder. This wording is unchanged by everything above and was written to
survive R4-vs-R5 landing either way; it did.

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

A seeded `benchmark.masks` generator producing **two committed artifacts**: one
mask list per eval tile, and a **train-time mask bank** of K = 50 draws per
head-train tile (§3.5). Training masks are *not* drawn on the fly; the bank is
read by every arm, seed and hyperparameter config, so the mask sequence a run
sees is a committed input rather than a reproduction of the generator's RNG.

- **Geometry families:** (i) random-walk blobs dilated to target area, the
  standard proxy for radar-shadow voids; (ii) axis-aligned rectangles
  (sensor-dropout proxy); (iii) transplanted real SRTM void shapes (§3.3).
- **Size buckets, as equivalent diameter:** small 16 to 48 px, medium 64 to
  96 px, large 128 to **158** px. The unit is `d = 2·√(area/π)`, and it is
  stated because it is load-bearing: both families are area-matched through
  `d`, so a blob and a rectangle in the same bucket cover the same pixel count
  and the bucket axis means one thing. Read instead as a *side length* the
  large bucket would demand 160² = 51% of a 224 px tile and collide with the
  40% cap at 141 px — which is the contradiction
  [p124](p124-voidfill-comparability.md) §9 reported and
  [#121](https://github.com/JStoschek/DEMAE/issues/121) measured away. The cap
  does bind, just barely: it permits `d ≤ 159.86 px`, so the published top of
  160 overshot by **0.14 px**. The top is lowered to 158 rather than clamped,
  so no draw is ever silently trimmed and the two constraints are satisfiable
  with margin. The two upper buckets still bracket Diff-DEM's published
  protocol; the change is 1.2% of the large bucket's diameter range and no
  measured quantity moves. One void per tile per eval instance; every eval tile
  appears once per bucket **for the two synthetic families (i) and (ii)**, so
  bucket scores are computed over identical tile sets.
- **The large bucket is positionally cramped, and the table should not read it
  as free variation.** Measured over 300 draws per cell: in the large bucket's
  top decile a blob's bounding box leaves a median of **950** legal placements
  inside the 208 px interior (min 150), against **43,264** for a point — so the
  biggest voids are close to centred, and instance-to-instance variation in
  that bucket is terrain, not placement. Blob draws also retry **1.44×** there
  against 1.00 in every other cell. Neither is fatal, and both are reasons the
  large bucket carries less independent information per instance than its
  instance count suggests.
- **The real-void family (iii) carries no bucket axis.** Two cells measured at
  true 1 arc-second returned 139 and 175 components in 16 to 48 px, 2 and 6 in
  64 to 96, and **zero** in 128 to 160. Forcing 667-per-bucket would mean either
  fabricating large voids, which destroys the only thing the family contributes,
  or silently reweighting, which makes it non-comparable anyway. It is drawn at
  its natural size distribution and reported as its own rows instead (§3.3).
  The large bucket survives for the *synthetic* families — the pressure this
  bullet used to hedge against was a units error, settled in the size-bucket
  bullet above. **The real family's bands stay at 128 to 160 px** even though
  the synthetic large bucket now tops at 158: those bands describe sizes real
  voids *have*, measured on SRTM, and are not draw targets the 40% cap can
  constrain — so renaming them to match a synthetic bucket would misdescribe
  the histogram (§3.3, §5).
- **Constraints:** voids fully interior (≥ 8 px from tile edge); void may not
  cover > 40% of the tile; the same mask list is reused verbatim by every arm.
  All three are **checked, not assumed**: over 300 draws per (family, bucket)
  cell the generator produced zero cap violations, zero interiority violations,
  and a maximum bounding-box side of 204 px against the 208 px interior
  ([#121](https://github.com/JStoschek/DEMAE/issues/121)).

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
  poorer. It is dropped from the void-rich region list. **The Andes stay, and
  the reason is diversity, not volume**: at ~148 small components per cell they
  contribute perhaps a fifth of the Himalaya's yield and ~1 large component per
  cell, so on volume alone they are marginal. They are kept because a family
  whose entire claim is "real void geometry" is weaker if every void comes from
  one massif, under one radar look-direction, in one climate. The 7 Andes
  granules cost ~8 MB of the fetch's 28.4 MB.
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
  longitude spacing poleward. **This protocol is not an invention** — mask
  extraction plus stratum-matched relocation onto void-free terrain is
  **Reuter et al. 2007**'s validation design, the method behind CGIAR-CSI's
  SRTM v4 fill, and it is cited as such rather than presented as new (§10).
- **Stratum assignment comes from GLO-30, not from SRTM.** Matching a void to a
  ruggedness stratum requires knowing the terrain the void sits on — which SRTM,
  by definition, never observed there. Two ways out, and only one is unbiased:
  inferring the stratum from the observed SRTM pixels *surrounding* the void
  measures the wrong terrain and biases systematically flat, because radar
  shadow occurs precisely where the surroundings are less extreme than the hole.
  So the stratum is computed from **GLO-30 at the void's own coordinates** —
  void-free there, on the same lattice below 50 degrees, and through the same
  code path that assigns strata to every other tile in the corpus. The SRTM
  granule contributes the void's *shape* and nothing else.
- **The size distribution is bimodal, and the ceiling is this benchmark's tile,
  not nature.** Within the 16 to 160 px window the plan scores, the draw is
  roughly **94% small, 5% medium, 1% large**. That window is not the whole
  story and the writeup must not imply it is: real voids are abundant *below*
  160 px and abundant again far *above* it — `N30E080`'s largest component is
  ~1,289 px equivalent diameter and every void-rich cell holds several over
  400 px — while 128 to 160 px sits in a genuine trough, big enough to be rare,
  small enough that the giants overshoot it (P125 §11.2). The giants are
  excluded because a 400 px void cannot fit inside a 224 px tile under §3.2's
  40%-area cap, i.e. **by this benchmark's tile geometry, not by scarcity**.
  The published size histogram therefore shows the excluded >160 px tail
  alongside the scored range, so a reader can see what the tile size cut.
- **Reported as two size sub-rows plus a count.** Instance budget per §3.2's
  eval-instance contract, split as drawn:
  - **real-small** (16 to 48 px) — the headline row. Its honest comparison
    partner is the *synthetic small bucket within the same ruggedness band*,
    not the synthetic pooled number, and tables state that explicitly.
  - **real-medium+** (64 to 96 px) — reported separately so the headline row
    stays size-comparable to synthetic small.
  - **128 to 160 px** — reported as an **instance count with no metric
    attached**. At ~1% of the draw split across ruggedness bands this is single
    digits per cell; publishing a mean over it would invite a significance the
    number cannot carry.
- **Stratified by ruggedness, with a population floor.** Every sub-row is broken
  out by the corpus's four collapsed ruggedness groups (§5). This is what makes
  the family readable at all: p119 measured the classical baseline varying ~40x
  across those bands against ~3x across size, so a pooled real-void number is a
  statement about which terrain the voids happened to land on. **Any band with
  fewer than 100 instances is reported as a count, not a metric.** Concentration
  in the rugged bands (R3/R4) is *expected*, not a surprise — radar shadow is a
  steep-terrain phenomenon — and a thin or empty R1 is a predicted outcome of
  this design rather than a defect in it. The realised histogram is checked
  against that expectation once the granules land, and this section amended if
  it is badly wrong.
- **What the family is for.** It is not a fourth point of metric comparison
  against the bucketed synthetic families — with a 94%-small draw against
  bucket-balanced synthetics, a difference in metres would confound shape with
  size. It carries **C4**, an *ordering* claim: does the arm ranking established
  on synthetic blobs still hold when the void shapes are real? Ordering is
  immune to the size confound because every arm sees the identical instances.
  Real voids are strikingly non-compact (perimeter ~2.0 to 2.2x that of an
  equal-area disc) and concentrate in steep terrain and radar shadow; the
  stratum-matched transplant preserves that difficulty rather than averaging it
  away. **Do not upgrade C4 into a metric-delta claim without first adding a
  size- and tile-matched synthetic control** — that control was considered and
  deliberately not built, because the ordering claim is what this draw supports.

### 3.4 Diff-DEM cross-eval (transfer test)

Their released Norway 10 m dataset, their mask lists, their checkpoint. Two
directions: Diff-DEM's model on its own data vs DEMAE arms on the same data
(off-distribution for a GLO-30 model: different country, 3x finer GSD); and
optionally their model on this benchmark's 30 m tiles (off-distribution for
them, same caveat, stated symmetrically). This section is labeled *transfer*,
not headline, in every table.

### 3.5 Committed artifacts

`benchmark/` module in the repo's stage-per-module style: tile lists, the two
mask artifacts, the mask generator's config hash, and a dataset card
documenting all of the above plus the licenses (SRTM public domain, Diff-DEM's
data per their repo terms).

- **Eval mask list** (parquet: `tile_id, family, bucket, mask_rle, seed`) — one
  row per eval instance, 30,000 synthetic rows plus the real-void family (§5).
- **Train mask bank** (parquet: `tile_id, epoch, family, bucket, mask_rle,
  seed`) — K = 50 draws per head-train tile, 1,000,000 rows, **740 MB** of RLE
  measured on the generator's own output. Generated once in **33 min** of one
  CPU core, then read by every arm, seed and config.
- **Extractor spec** (`extractor_spec.json`, [ADR-0012](../adr/0012-frozen-feature-extractor-contract.md),
  [#137](https://github.com/JStoschek/DEMAE/issues/137)) — the frozen-feature
  contract as a file rather than as prose: per-arm normalization, the emitted
  layer list, the reference and compute dtypes, the token layout, the encoder
  identity and checkpoint md5, and ADR-0011's `pos_emb` derivation. Committed for
  the reason the whole family is: **what Task 2 re-reads is the file**, and an
  extractor re-derived from a description keeps producing perfectly good features
  of a slightly different kind, with nothing in the output saying so. It
  re-checks itself at load and refuses rather than warns (§9.4).

The two mask artifacts carry the same generator and the same config hash, so the
determinism contract is one contract: two runs from committed inputs reproduce
the tables, and the training masks are now inside that guarantee rather than
beside it.

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
  label-efficiency subsets. Report mean ± std. **One exception:** §2.3's ViT-S
  appendix cells (A4-S, A6-S) run at 20k with a **single seed**, and are
  reported as a single observation without a ± — they answer a capacity-scaling
  aside that no claim in §0 rests on, and buying them error bars would cost more
  eval passes than the primary comparison's entire label sweep.
- **Features are *not* precomputed; masks are** *(amended 2026-08-10, §7 —
  [#120](https://github.com/JStoschek/DEMAE/issues/120); mask half amended 2026-08-11 —
  [#121](https://github.com/JStoschek/DEMAE/issues/121))*. The frozen encoder runs inline
  in the head step, and the void mask is read from §3.2's bank, epoch `e` of 50. The
  feature cache this bullet used to specify — fp16 memmaps, 301 KB/tile, 5.6 GiB per
  encoder per 20k set — was measured to save **$0.50** across the whole benchmark,
  because a frozen ViT-B forward pass costs 0.179 ms. Against that it fixed the A1-filled
  input, and therefore the mask, for all 50 epochs: the sentence this bullet used to end
  with ("precomputation fixes the A1-filled input per (tile, mask), which is exactly the
  eval-time contract") is true at eval time and is a **frozen training mask** at training
  time, which §3.2 does not intend.
- **What replaces the cache is a mask bank, not nothing, and it is 40× smaller.**
  Deleting the feature cache made mask generation a per-epoch cost that nobody had
  priced. Measured, a draw averages **2.00 ms** — but that mean spans 240×: a rectangle
  is 0.037 ms and a large blob is 8.86 ms (30.9 ms worst case), because the blob family
  dilates a random walk to a target area. §7's table implies **34.4M training tile
  presentations**; drawn on the fly that is **19.1 CPU-hours**, against the program's
  2.56 GPU-hours and the 1.6–4.1 h eval line §7 already calls its largest. (That 34.4M
  was counted against §7's table *before* §2.3's rewrite cut the ViT-S cells to one seed
  and took R4 out of the label sweep — it is an upper bound, not a current figure.) Per epoch over
  20k tiles it is **40 s** of one core against a **4.6 s** GPU step (head 1.0 s + inline
  ViT-B 3.58 s) — the loader running **8.7×** the GPU it feeds, and still at parity with
  eight workers. The bank collapses 34.4M draws to **1M**: 33 minutes once, 740 MB, and
  per-epoch variation is preserved *exactly*, since 50 committed masks per tile is what
  50 on-the-fly epochs would have produced. It is the same trade that killed the feature
  cache, read the other way round — cache the artifact that is cheap to store and
  expensive to recompute, not the one that is the reverse.
- The *eval* side is unchanged throughout — the committed mask list fixes each test
  instance's input exactly as it always did.
- **The inline forward runs the §9.4 extractor contract, not an ad-hoc call**
  *(amended 2026-08-11 — [#137](https://github.com/JStoschek/DEMAE/issues/137),
  [ADR-0012](../adr/0012-frozen-feature-extractor-contract.md))*. The head reads
  the contract's `output` field — each encoder's own final tensor, which for
  DEMAE means post-`LayerNorm` and therefore *not* `intermediates[11]`. Two
  consequences for this section. The extractor is handed **raw metres and a
  valid-pixel mask**, so §2.1's normalization happens inside it and the A1 fill
  happens before it. And the extractor emits twelve intermediates that no arm
  here reads, because Task 2 does — which is why §9.4 makes the intermediates
  path Task 1's to *test*, even though it is Task 2's to use.

### 4.2 DEMAE native fine-tune (A3)

- Init from the **R5 ViT-B release checkpoint** (§2.3's encoder-by-property
  wording), with the **R4 continual-init checkpoint as the pre-registered
  secondary**; §2.3's `pos_embed` axis-order precondition applies to both loads.
  The old text here read "the release checkpoint (R1; R4 arm if usable)", which
  was stale twice over — R1 is the **ViT-S** run, and R4 is no longer
  conditional. Continue MAE
  training with the mask sampler swapped: instead of 75% random patches, drop
  the set of patches overlapping a synthetic void **read from §3.2's train mask
  bank** (16 to 158 px families, epoch `e` of 20), plus 25% random patches so
  the encoder keeps seeing global context and the effective mask ratio stays
  near pretraining's. A3 reads the same bank as the heads — its 20 epochs are
  the bank's first 20 — so the void geometry A3 adapts to and the geometry the
  frozen arms train against are the same draws, not merely the same
  distribution.
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

Standard 4-down/4-up UNet (~17M params), 2-channel input (A1-filled tile, void
mask), residual output, same loss, same schedule length, same seeds. Trained
from scratch at each label-efficiency size.

This bullet used to justify the UNet's size as "comparable order to
ViT-S+head", an anchor that goes with ViT-S's demotion in §2.3. The honest
statement under a ViT-B primary is the stronger one: **A7 trains ~17M
parameters against the 1.18M trainable in every frozen arm**, an order of
magnitude more, so a totals-only column (17M against A4's 86.4M) reads the
baseline as handicapped when it is the opposite. §6 item 4 prints both counts
for this reason. This is the PANGAEA bar, and it is a hard one: if A7 matches
A4 at 20k tiles, the honest conclusion is that pretraining pays only in the
low-label regime, and C3 is where DEMAE must show it.

### 4.5 Order of operations

1. Freeze tile lists, the eval mask list, and the train mask bank (CPU only,
   local; the bank is the 33-minute line, §3.5).
2. A1 baseline numbers + unit tests on norm inversion and mask bookkeeping.
3. *(was: precompute frozen features — deleted, §4.1. There is no feature
   precompute step; the encoder runs inline.)*
4. Tune head hyperparams on A6/val; freeze.
5. Run A4, A5, A6, A7 at 20k; then the 1k/5k sweep; then A3; then A2/A9
   (inference only); then A8 cross-eval; secondary 10 m regime last.

## 5. Metrics and statistics

All metrics computed **only inside the void**, in metres, per (family, bucket),
over the fixed test mask lists. The real-void family is the one exception: it
carries no bucket axis (§3.2), so instead of one row per bucket it reports
**two size sub-rows per arm — real-small (16 to 48 px) and real-medium+
(64 to 96 px) — plus a bare instance count for 128 to 160 px** (§3.3). Both
sub-rows are published with the size histogram, including its excluded >160 px
tail, so the reader can see what distribution the number is over and what the
224 px tile cut off. Comparisons drawn against this family are against the
synthetic **small** bucket within the same ruggedness band, never against a
synthetic pooled number, and carry C4's ordering claim rather than a metric
delta.

The set below is **pre-registered and graded**: five headline columns per cell,
four appendix ones, and nothing else. The grading is measured rather than
argued — [`p123-metric-discrimination.md`](p123-metric-discrimination.md) scored
every candidate over 24,000 A1 instances for **separation** (spread across the
A1 variants over the within-cell instance spread; below ~1.0 a metric cannot
rank arms above noise) and **redundancy** (within-cell Spearman against the rest
of the set, never pooled, since terrain drives every metric at once). The rule
applied is the one that motivated the measurement: a metric that cannot separate
A1 from truth cannot separate the arms either.

- **Elevation error (headline): RMSE and NMAD.** RMSE because C3 and §4.1's
  early stop are already worded on it; NMAD because it separated best of all ten
  candidates (2.51, worst cell 1.89) *and* is less redundant with RMSE (0.92)
  than the 95th percentile is (0.99). **MAE and p95 move to the appendix** —
  p95 is the weakest of the three and fails to separate the A1 variants at all
  in its worst cell (0.86).
- **Systematic offset (headline): mean per-instance |bias|** — the signed error
  averaged within an instance, then its magnitude averaged over instances.
  **The pooled mean signed error is deliberately not reported.** Measured, it
  cancels to 0.003–0.9 m against MAE of 0.87–38.6 m, so it would print a null
  for every arm; per-instance |bias| runs at roughly *half* of MAE, and 42–50%
  of instances carry an offset exceeding half their own MAE. This column is a
  **guard, not a discriminator**: its separation is modest (0.95) because its
  job is to catch a fill sitting at the wrong absolute level, a failure mode no
  other column sees. It is not to be dropped for ranking poorly.
- **Terrain derivatives (headline): slope RMSE and slope-weighted aspect error**,
  both from the **Zevenbergen–Thorne** quadratic surface — which profile
  curvature needs anyway, so one kernel serves all three. ZT's first derivatives
  are central differences and therefore *numerically identical* to the corpus's
  own `dataset.metrics.slope_deg` (max divergence 0.0006° over 12,000
  instances), which keeps benchmark slope and the `slope_mean` defining the
  ruggedness bands the same quantity. **Horn was measured and rejected**: ~7%
  different in level, 0.98 identical in ranking, and it would have bought that
  nothing at the cost of the alignment. **Aspect is headline because it is the
  only independent signal in the set** (within-cell Spearman −0.07 vs RMSE,
  −0.02 vs NMAD, +0.26 vs slope) and separates well (1.46). **Profile curvature
  moves to the appendix**: 0.72 redundant with slope, and unable to separate
  variants in its worst cell (0.28).
- **Derivatives score on the full void mask, not an eroded one.** A 3×3 kernel
  one pixel inside the void does read ground truth, and that was expected to
  hand an idle arm a free ride. Measured, the effect runs the other way: the
  boundary ring is where a *discontinuous* fill gets caught, so eroding it
  **halves** slope's separation (1.89 → 0.91). The free ride is real but worth
  −0.005 to −0.107° on the smooth interpolators, against 17% of the scored
  pixels on a small void. The eroded-mask slope is kept as an appendix
  contamination control; the rule itself is stated in §6.
- **Seam continuity (headline): the full-minus-eroded slope gap.** Free, since
  both masks are already computed, and it sign-separates a fill that breaks at
  the void boundary (+1.33° on the flat-fill control) from every smooth
  interpolator (−0.005 to −0.107°). **This is what carries the operational half
  of C2 now that drainage is gone**, and what makes §4.2's cross-fade auditable
  rather than assumed.
- **Perceptual/structural: nothing scored.** Hillshade SSIM was measured and
  cut — weakest separator of all ten candidates (0.56) *and* among the most
  redundant (−0.84 with slope, −0.42 with RMSE). LPIPS is cut with it, on this
  section's own former hedge that ImageNet features are off-domain for
  hillshades. The "what a QA analyst actually looks at" argument is carried by
  §9's hillshade **panel figure**, which is where it belongs: a picture, not a
  number nobody can act on.
- **Stratification:** every metric also reported by ruggedness band, where the
  four bands are the **quartiles of `slope_mean` over the eligible test pool**,
  cut on the pool itself — not a collapse of the corpus's k-means strata. The
  earlier wording ("the corpus's ruggedness strata, collapsed to 4 groups") did
  not describe a defined operation: those strata are k-means clusters over a
  six-metric descriptor vector (ADR-0004) with no ordering along a ruggedness
  axis, so there is no such thing as collapsing them to four. The quartile cut
  is what p119 and p123 both measured on, and every ruggedness number this plan
  cites is on it. A method that wins on plains
  and loses in mountains must be visible as such. This is **load-bearing, not
  cosmetic**: p119 measured the classical baseline's error varying ~40× across
  these four bands but only ~3× across the size buckets
  ([`p119-fill-headroom.md`](p119-fill-headroom.md)), so a pooled per-arm number
  is dominated by the rugged tail and hides the fact that the flattest quartile
  is already at the resolution floor. Where table space forces a choice, drop a
  size bucket before dropping a ruggedness band. **Population floor:** a
  (row, band) cell with fewer than 100 instances is reported as a count with no
  metric rather than pooled into a neighbouring band. This binds hardest on the
  real-void family, which takes whatever distribution SRTM gives it and is
  expected to concentrate in R3/R4 (§3.3). **The synthetic rows clear it with
  room** — 5,000 test tiles × 2 families × 3 buckets is **30,000** instances,
  1,250 per (row, band). **The real-medium+ sub-row probably does not, and its
  rule is therefore fixed here, before the counts exist.** p125 measured 2 and 6
  components in 64–96 px on two 1-degree cells at true 1 arc-second; scaled by
  [#127](https://github.com/JStoschek/DEMAE/issues/127)'s 24 granules that
  projects to roughly **96 instances in total** — at the floor before
  stratifying, and ~24 per band after. (real-small projects to ~3,800, which
  clears comfortably.) So: **real-medium+ is reported unstratified — one pooled
  number with its instance count — unless the realised histogram clears 100 per
  band**, which [#135](https://github.com/JStoschek/DEMAE/issues/135) checks
  against exactly this threshold. The projection rests on two cells and may be
  wrong in either direction; what is pre-registered is the threshold, not the
  outcome. One knock-on: §7's **32k eval instances is provisional**, since it
  predates #126's reshaping of the family — at the natural distribution the
  total is nearer 34k, which moves the two eval lines by ~6% and no conclusion.
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
   Derivative metrics score the **full** void mask, including the boundary ring
   whose 3x3 kernel partly reads true pixels. That ring is included knowingly:
   it is where a discontinuous fill is caught, and eroding it halves slope's
   ability to separate arms. The free ride it grants a *smooth* fill was
   measured at −0.005 to −0.107° and is published beside the eroded-mask control
   rather than argued away ([`p123-metric-discrimination.md`](p123-metric-discrimination.md)).
3. GSD stated per regime; primary regime is DEMAE-native and says so;
   TerraMind gets a retrained head in its own 10 m regime before any
   conclusion is drawn about it.
4. Capacity and pretraining data are **reported rather than bracketed**, because
   both are near-matched and there is nothing left to control. Parameter counts
   in every table, **total and trainable**: DEMAE ViT-B **85.25M** against
   TerraMind's DEM tokenizer encoder **90.17M** — same depth, same 16 px patch,
   same 14x14 token grid — and 1.18M trainable in every frozen-encoder arm
   against the UNet baseline's 17M, so the from-scratch bar is not read as
   handicapped by a totals-only column. Pretraining differs in data, not
   compute: **189.2M tile presentations against 181.8M** at the same effective
   batch of 2048, over **21.4M km²** of GLO-30 against **63.4M km²** of the same
   product resampled to 10 m. That **~3x** ground-area advantage to TerraMind is
   recorded here, before any result, and is not available as a post-hoc
   explanation of a DEMAE loss that was not also offered as a discount on a
   DEMAE win. TerraMind's 20 DEM-tokenizer epochs are the per-modality figure
   from terratorch and IBM's model card, **not** the 100 the TerraMind paper
   states for its S-2/S-1 tokenizers ([`p138-terramind-corpus.md`](p138-terramind-corpus.md)).
5. The tokenizer encoder was trained for reconstruction, not representation;
   A9 quantifies the bottleneck; conclusions are phrased as "under identical
   probes" not "TerraMind is worse."
6. Geographic holdout only; the test geocells are the corpus's committed test
   split; Diff-DEM data is labeled transfer, both directions.
7. Negative or null results (e.g., A7 ties A4 at 20k) are reported with the
   same prominence as wins; C3's low-label sweep is the pre-registered fallback
   claim.
8. Pre-registration: this memo is committed before the first head trains;
   success criteria C1-C4 are defined above the results, not after them.

## 7. Compute budget

**Measured, 2026-08-10 — [`p120-primitive-costs.md`](p120-primitive-costs.md)
([#120](https://github.com/JStoschek/DEMAE/issues/120)).** The table below replaces an
estimate that was high by 8 to 13x and was denominated in the wrong card at the wrong
price: this project rents an **RTX 5090** at a real **$0.4016/hr**, not a 4090 at $0.34
(see [`phase1-pretraining-plan.md`](../phase1-pretraining-plan.md)'s hardware line and
§12). Every line is a measured primitive times a count, and the primitives are published so
any new arm can be costed without a fresh run.

Costed for the design §4.1 now specifies — **frozen encoder inline, no feature cache,
training masks read from the committed bank** ([#121](https://github.com/JStoschek/DEMAE/issues/121)):

| Item | Count | 5090-min | Notes |
|---|---|---|---|
| **A7 UNet, 3 seeds + label sweep** | 3 x 50 ep x 20k, + 2 x 50 ep x 6k | **46.1** | priciest single arm; UNet epoch is 15.4 s against the head's 1.0 |
| Head tuning on A6 | 12 configs x 50 ep x 20k | 45.7 | |
| Head training A4/A5/A6 | 4 cells x 3 seeds + 2 appendix cells x 1 seed, x 50 ep x 20k | 42.1 | 14 cell-seeds at the 3.01 min/cell-seed this line already prices |
| Label-efficiency sweep (1k, 5k; 2 seeds) | 3 cells x 2 seeds x 50 ep x 6k | 5.4 | R4 and ViT-S are both out of the sweep (§2.3) |
| Inference over the eval instances | 38 passes x 32k | 3.6 | frozen forward is **0.179 ms/tile** at ViT-B; prices an encoder pass, so A2/A3's decode is not in it |
| A3 fine-tune (R5 ViT-B + R4 ViT-B) | 20 ep x 20k | 4.4 | fine-tune epoch is **6.66 s**; measured equal to a pretraining epoch. **Derived, not re-measured** — the old 3.1 was ViT-B + ViT-S, re-split at Phase 1's measured 3,001 / 7,291 samples/s |
| **GPU total** | | **147.3 min = 2.46 h** | **$0.99** at $0.4016/hr |
| **A9 (TerraMind full tokenizer)** | 32k instances | **unpriced** | diffusion decode; at 1 s/instance it is 8.9 GPU-h, 3.5x the whole table above. **§7 is not final until this is measured.** |
| Eval passes (CPU, local, single core) | **39** arm-seed passes x 32k instances | **1.6-4.1 h** | 4.6 ms/instance scoring-only, 12.1 ms end-to-end as [p119](p119-fill-headroom.md) measured it; **the largest single line in the benchmark** |
| Train mask bank, generated once (CPU, local, single core) | 20k tiles x K=50 | **33 min** | 2.00 ms/draw averaged over the six (family, bucket) cells — 0.037 ms for a rectangle, 8.86 ms for a large blob. **Drawn on the fly instead this line would be 19.1 CPU-h**, since §7's own counts total 34.4M training tile presentations ([#121](https://github.com/JStoschek/DEMAE/issues/121)). **The 34.4M predates §2.3's rewrite** and is not re-derived here — see the note below the table |

The 39 scored passes are 23 at 20k — A1; A2 and A3 at R5 and R4; A9; the four primary frozen
cells (A4-R5, A4-R4, A5, A6) at 3 seeds; the two ViT-S appendix cells at 1 seed; A7 at 3 seeds
— plus 16 for the label-efficiency sweep (3 cells x 2 seeds x 2 sizes, plus A7), which §6
item 7 makes a headline claim and therefore scores rather than merely trains.

**This was 48 passes before §2.3's rewrite**, and the 19% cut is the point worth reading: the
GPU bill barely moved ($1.03 → $0.99) while the benchmark's largest line dropped by nine
passes, *even though R4 was added*. ViT-S gave back more than R4 took. #117's standing note
that attention and wall-clock are the real budget is what this line measures; the dollar column
is not.

**The budget is not a constraint and should stop being treated as one.** At $0.99 of GPU
against a $25 target — the CPU hours are the project's own laptop, not rent — no line here justifies a spot-instance strategy or a resume design;
#117's standing note that *attention and wall-clock are the real budget* is now measured
rather than asserted. The one number that could still move the total is A9's.

**Why no feature cache** (the §4.1 amendment, priced): precomputing frozen features is
*cheaper in GPU seconds* — 1.31 h against 2.56 h — but the whole difference is **$0.50**,
and against that it costs 30.3 GiB, 7.9 TB of re-reads, and a design constraint §4.1 did
not name: a cached feature fixes the A1-filled input, hence the mask, so head training
would see **one void per tile for all 50 epochs**, contradicting §3.2's per-epoch masks.
Run the encoder inline.

**Why a mask bank instead** ([#121](https://github.com/JStoschek/DEMAE/issues/121)):
deleting the feature cache made mask generation a per-epoch cost, and it had never been
priced. It is **2.00 ms/draw**, so the 34.4M presentations this table's own counts imply
come to **19.1 CPU-h** — larger than the eval line above and 7.5× the whole GPU program.
Per epoch over 20k tiles that is 40 s of one core against a 4.6 s GPU step, so on-the-fly
masks make head training **loader-bound**, which is the exact property the deleted cache
was supposed to buy. Banking K = 50 draws per tile cuts 34.4M draws to 1M — **33 min once,
740 MB** — with per-epoch variation preserved exactly, because 50 committed masks per tile
is what 50 on-the-fly epochs produce. Both decisions are the same trade read consistently:
cache what is cheap to store and expensive to recompute, not the reverse.

**Open, from the #121/#122 merge (2026-08-11):** the 34.4M presentation count above was
derived from §7's table *as it stood before* §2.3's rewrite, which cut the ViT-S appendix
cells to a single seed and took R4 out of the label-efficiency sweep. Every 34.4M / 19.1
CPU-h / 1M / 33 min / 740 MB figure in this section is therefore an **upper bound** and
wants re-deriving against the current cell counts. The bank-vs-on-the-fly conclusion is
not in doubt — the margin is 7.5× — but the printed numbers are stale.

§9.4's Task-2 reuse constraint is **not** merely unaffected by these two decisions, as the
#120 amendment said — it was independently wrong, and #121 rewrote it. Task 1's encoder
never sees a clean tile, so Task 2 inherits an extractor contract, not an artifact.

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
- **Spot-instance reclaim: not a risk this benchmark has.** Heads checkpoint
  every epoch to the synced directory like every other run in this repo, but
  §7's measured total is 2.56 GPU-hours with a 46-minute longest job, so there
  is nothing long-running to lose. *(The clause that used to close this bullet —
  "features are memmapped so nothing long-running is unresumable" — went with
  the feature cache in §4.1.)*

## 9. Deliverables

1. `benchmark/` module + committed tile/mask/split artifacts (reproduces every
   table from the repo).
2. The results table (arms x buckets x strata, CIs marked) and three figures:
   RMSE vs void size per arm; the label-efficiency curve (C3); a hillshade
   panel grid (truth, voided, A1, A3, A5, A8) on two rugged and one flat tile.
   The hillshade panel is **load-bearing, not decorative**: §5 scores no
   perceptual metric, so this figure is the whole of the visual-quality
   evidence and is not to be cut for space.
3. A short results memo in `docs/research/` answering C1, C2, C3 with numbers
   and C4 with a rank comparison, plus the fairness checklist and the TerraMind
   caveats.
4. **What Task 2 inherits is the frozen-feature *extractor contract*, not a
   feature cache** *(amended 2026-08-11 —
   [#121](https://github.com/JStoschek/DEMAE/issues/121))*. This item used to
   read "the same frozen-feature cache feeds Task 2's probes unchanged, which is
   why this ordering is efficient." That was never true, and the
   [#120](https://github.com/JStoschek/DEMAE/issues/120) decision to delete the
   cache is not why. **Every Task 1 arm's encoder input is an A1-filled *voided*
   tile; Task 2 probes *clean* tiles.** Task 1 therefore computes no clean-tile
   feature at any point, so there was never an artifact for Task 2 to reuse —
   with a cache or without one. The ordering is still the right one, but for the
   reason below rather than for a shared file.

   What genuinely transfers, and what Task 1 must therefore pin down and
   document, is the **extractor**: the code path that turns a tile into frozen
   patch tokens, the per-arm normalization, the layer indices, and the dtype. If
   Task 1 fixes those, Task 2 re-runs them over its own clean tiles and the two
   tasks' C1 claims are about the same features. The extractor needs only
   `encoder` + `quant_proj` — **90.2M parameters, 361 MB fp32, 31.5% of the
   1.15 GB checkpoint**
   ([`p118-terramind-encoder-probe.md`](p118-terramind-encoder-probe.md)) —
   since the remaining 196M is the diffusion decoder that no in-scope arm calls.

   **The contract's fields are now fixed, and it is a committed file**
   *(amended 2026-08-11 — [#137](https://github.com/JStoschek/DEMAE/issues/137),
   recorded as [ADR-0012](../adr/0012-frozen-feature-extractor-contract.md))*.
   #121 named these fields and fixed none of their values; that is what this
   amendment does. In brief, with the reasoning and the rejected alternatives in
   the ADR:

   - **Raw int16 metres + a valid-pixel mask in, never a pre-normalized
     tensor.** The extractor applies its own declared normalization, so the
     pretraining loader's per-tile-standardized output is not something it can be
     handed. p118 §7's trap — TerraMind's encoder accepts either normalization
     silently and returns a plausible `[1, 768, 14, 14]` both ways — stops being
     expressible rather than being caught by a test. **Valid = land ∧ ¬void**, so
     Task 1's non-void stats (§2.1) and Task 2's clean-tile stats are the same
     code path.
   - **Statistics are recomputed from that mask, never read from the
     manifest** — one path, no branch that only one task exercises. Measured over
     500 corpus tiles, recomputing reproduces the manifest's stored `mean`/`std`
     to **3.6e-4 m** and **1.1e-7 relative**, and that equivalence ships as a
     test rather than as a claim.
   - **All twelve block outputs are emitted, plus a separate `output` field.**
     The head and any linear probe read `output`; a UperNet reads
     `intermediates` and picks its own indices. **DEMAE's `intermediates[11]` is
     not its `output`** — `MaskedAutoencoder._transform` applies a final
     `LayerNorm` that TerraMind's encoder has no counterpart for (p118 §5.3), so
     the two are named fields rather than one shared index. Task 1 reads only
     `output`, and therefore owes the intermediates path a test on shapes,
     ordering and count; an unexercised path shipped with a contract's authority
     is *the composition*'s failure with a document attached.
   - **Canonical layout `(N, 196, D)`, row-major, CLS optional and DEMAE-family
     only.** Row-major is not a coin flip: DEMAE tokenizes through timm's
     `PatchEmbed` and TerraMind's `[N, 768, 14, 14]` flattens identically against
     the grid its `pos_emb` is built on. CLS never enters a cross-encoder
     comparison, because TerraMind structurally has none.
   - **fp32 is the reference tensor; compute dtype is declared per task; storage
     dtype is outside the contract.** Measured on the R5 release encoder over
     12,544 tokens, fp32 against bf16 autocast agrees to mean per-token cosine
     **0.999968** — against the ≥ 0.99 bar
     [#134](https://github.com/JStoschek/DEMAE/issues/134) fixed for materiality,
     and the 0.9708 that failed it. **Measured on the DEMAE family only:** the
     TerraMind arm's figure does not exist yet and the build session measures it
     on first run rather than inheriting this one.
   - **`extractor_spec.json` is committed** and joins the family whose defining
     property is that the committed file is what later phases re-read (§3.5). It
     re-checks itself at load — encoder identity and checkpoint md5, TerraMind's
     constants (the **named** v1 lookup §2.1 requires, *and* its value against
     `670.665 / 951.272`: the name is what picks the right pair of the three that
     ship, the value is what catches a terratorch retune), shape and layer count,
     and ADR-0011's `pos_emb` derivation as a declared field — and its md5 goes in
     the benchmark's run record, not in Phase 1's frozen four-field data
     fingerprint.

   **[`task2-landform-plan.md`](task2-landform-plan.md) is stale on this in
   three places** and is *not* amended here, because Task 2 is out of this map's
   scope ([#117](https://github.com/JStoschek/DEMAE/issues/117)) and gets its own
   map: §3.3 ("the frozen feature caches from Task 1 are reused directly for the
   clean tiles"), §4.2 (intermediate-layer features "cached once per encoder",
   ~25 GB), and §4.4 step 3 ("Extend Task 1's feature caches with intermediate
   layers"). Task 2's own caching decision is open and may well go the other way
   — its UperNet probes read 4 layers per tile over 30 epochs, which is a very
   different read pattern from Task 1's single inline forward. The point for
   Task 2's map is only that it inherits an extractor, not a file.

## 10. Key sources

Diff-DEM (github.com/kylelo/Diff-DEM; IEEE GRSL 2024) · SRTMGL1N
(earthdata.nasa.gov, `.num` class raster; predicate and geometry measured in
`p125-srtm-void-geometry.md`) · SRTM Collection User Guide V3 (Table 2, the
`.NUM` fill values) · **Reuter, Nelson & Jarvis 2007** (*Int. J. Geogr. Inf.
Sci.* 21(9), 983–1008; the void-fill validation design this plan's §3.3
transplant protocol follows — mask extraction plus stratum-matched relocation
onto void-free terrain — and the method behind CGIAR-CSI's SRTM v4 fill) ·
SRTM v2.1 (3 arc-second, Wayback per-tile capture of
`dds.cr.usgs.gov`; degraded fallback only) · TerraMind (arXiv 2504.11171; tokenizer model card, Apache-2.0) · TerraMesh
(arXiv 2504.11172, 10 m DEM resampling; 9,089,536 DEM tiles at 264 px) ·
terratorch v1.2.10 source (encoder extraction path, normalization constants,
`tokenizer_register.py`'s per-modality `epochs=20` for DEM) · Major TOM (arXiv
2402.12095, the 10 km grid whose 10.68 km cells overlap) · PANGAEA (arXiv 2412.04204, frozen
protocol + baseline bar) · MAE (arXiv 2111.06377) · GrounDiff (arXiv 2511.10391,
residual formulation) · FathomDEM (ERL 2025, stratified DEM eval practice) ·
Kornblith et al. (arXiv 1905.00414) for the CKA follow-up if features get
compared directly.
