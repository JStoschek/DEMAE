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
  ([`p119-fill-headroom.md`](p119-fill-headroom.md)). **The "at every void size"
  half is read off §9's RMSE-vs-void-size figure, not off the table**, which
  pools the size buckets ([#136](https://github.com/JStoschek/DEMAE/issues/136))
  — which is why that figure is load-bearing. **"Seam continuity"
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
  C1's comparison is made entirely on synthetic masks. **The band is R4, and that
  is measured rather than chosen**: real SRTM voids land 99.94% in the roughest
  quartile ([#135](https://github.com/JStoschek/DEMAE/issues/135)), so C4 is
  evaluable in R4 and nowhere else. That narrows the claim and states the
  narrowing up front — and R4 is also where the arms have the most room to
  separate, so it is the band in which a rank swap says the most about C1. **It
  prints as its own rank block** — synthetic blob small, real-small and
  real-medium+ in R4, ranked on RMSE and slope RMSE
  ([#136](https://github.com/JStoschek/DEMAE/issues/136)) — so that the ordering
  claim is what the reader is handed, rather than three columns of metres they
  would be free to subtract.

If C2 partially fails (e.g., A7 wins on seam continuity while losing on metres),
that is reported, not
hidden; PANGAEA-style candor is part of the deliverable. C1 and C3 are the
claims DEMAE's worth actually rests on, because they isolate the pretrained
representation from decoder capacity.

**This table is an internally controlled comparison, and there is nothing
external to stand it beside. Goes in the writeup.** No public DEM void-filling
benchmark exists that independent methods report against: published numbers in
this task are each tied to a private tile set, a private mask set and a private
scoring script. The field's most-cited evaluation, **Reuter et al. 2007**,
publishes mean *rank* of RMSE and never metres; the most recent work,
**Dfilled (2025)**, retrained its competitors rather than citing them. Every
arm here sees the same tiles, masks, seeds and head, and every number was
produced by one scoring function in this repository.

**Both candidates for an external contact point were investigated, and neither
yields a comparable number.** This is a checked claim, not a hedge:

- **Diff-DEM.** Its committed `tif_metric.py` de-normalises to metres using the
  ground-truth tile's own extrema, **void interior included** — the exact leak
  §2.1 forbids for DEMAE — so their metres and ours are different quantities and
  could never share a column
  ([#124](https://github.com/JStoschek/DEMAE/issues/124)).
- **GeomorPM.** Its CC BY 4.0 Figshare deposit releases *pretrained* weights
  only; there is **no void-filling checkpoint**, its VF fine-tune config is
  absent from the deposit, and its paper is paywalled. A cross-eval would mean
  reproducing an unpublished recipe and then being unable to check the
  reproduction ([#128](https://github.com/JStoschek/DEMAE/issues/128)).

So the honest statement is the strong one — **incomparable, and we looked** —
rather than "incomparable except at the transfer blocks". That sentence is
available only because both were opened and read, and it is worth more than the
two weak transfer rows it replaces.

## 1. Comparison arms

| # | Arm | Encoder | Trained part | Input convention |
|---|-----|---------|--------------|------------------|
| A1 | Bilinear/spline fill | none | none | classical `gdal_fillnodata`-class interpolation, in **two** variants: rasterio `fillnodata` (the default) and a thin-plate spline — see the amendment below |
| A2 | DEMAE zero-shot | DEMAE frozen (R5 ViT-B; R4 ViT-B secondary) | none | native MAE inpainting: void-touching patches dropped, decoder reconstructs, non-void pixels composited back |
| A3 | DEMAE fine-tuned | DEMAE, all weights (R5 ViT-B; R4 ViT-B secondary) | encoder + decoder | native token dropping on void-shaped masks (§4.2) |
| A4 | DEMAE frozen + head | DEMAE frozen (R5 ViT-B; R4 ViT-B secondary; ViT-S as the §2.3 appendix row) | head only | voided tile pre-filled by A1 + void mask channel at the head (§2.4) |
| A5 | TerraMind frozen + head | TerraMind DEM tokenizer ViT-B encoder, 768-d `feats` (pre-`quant_proj`), frozen | head only (identical to A4's) | same pre-filled input, TerraMind's own normalization (§2.1) |
| A6 | ImageNet-MAE frozen + head | timm/MAE ViT-B frozen, 1-channel patch embed collapsed (same recipe as `pretrain.init`) | head only (identical) | same pre-filled input |
| A7 | UNet from scratch | none (UNet) | everything | voided tile + mask channel, residual over the A1 fill |
| A9 | TerraMind tokenizer pass-through (reference row) | full tokenizer: encode, FSQ, diffusion decode | none | A1-pre-filled tile through the full VQ pipeline; shows the 5-d bottleneck's cost |

**A1's variants are two methods, not three** *(amended 2026-08-13 —
[#158](https://github.com/JStoschek/DEMAE/issues/158))*. This row read
"classical `gdal_fillnodata`-class interpolation; also rasterio `fillnodata`,
and a thin-plate-spline variant", which names three things and describes two:
`rasterio.fill.fillnodata` **is** a binding of GDAL's own `GDALFillNodata`, so
"the `gdal_fillnodata`-class fill" is the class of method and "rasterio
`fillnodata`" is this project's way of calling it. `benchmark.fill.VARIANTS`
therefore registers `fillnodata` and `tps`, and no third entry: two names on one
code path would give the results parquet a variant column whose two values are
bit-identical, which reads as a comparison and is not one. The choice between the
two remains unimportant to the ceiling — §9's risk row measured TPS ahead of
`fillnodata` by a median of 0.2 m, winning 55% of paired instances — so the
default is `fillnodata` and the spline is a config knob (`fill.variant`) rather
than a second committed baseline. No registered rule moves: R1 grades metrics and
R5 fixes the printed table, neither of which mentions A1's method.

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

### 2.2 GSD: one regime at 30 m, and what TerraMind's tokenizer was fed

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
  This is about the token grid, not GSD; the GSD asymmetry is the separate,
  still-live matter of the last three bullets, and #133 did not touch it.
- **The GSD asymmetry is a bandwidth shift, not a resolution handicap.** State it
  this way and no other. TerraMind's DEM tokenizer was trained on GLO-30
  bilinearly upsampled 3x onto the 10 m grid (§2.1, [`p138-terramind-corpus.md`](p138-terramind-corpus.md)),
  so its inputs carried real detail only up to the 30 m source — roughly a third
  of that grid's pixel Nyquist — and each 16 px patch spanned 160 m of ground.
  At this benchmark's native 30 m it sees full-bandwidth terrain across a 480 m
  patch. **It is fed sharper inputs than it ever trained on, at 3x the ground span
  per patch, not coarser ones.** The older phrasing — "evaluated below its native
  10 m GSD" — reads as resolution starvation and is wrong on the mechanism, since
  those 10 m pixels were upsampled 30 m of the same product DEMAE uses. It also
  invites the "you crippled the baseline" objection that
  [#133](https://github.com/JStoschek/DEMAE/issues/133) spent a whole ticket
  killing. This is a real out-of-distribution condition and the writeup states it;
  it is simply not the one the plan used to claim.
- **That caveat is pre-registered and explicitly symmetric** — *registered rules
  **R9** and **R9b**, asserted in
  [ADR-0013](../adr/0013-task1-pre-registration-register.md) and indexed at §6
  item 8; what follows is the reasoning, not a separate commitment* — under the
  rule [#122](https://github.com/JStoschek/DEMAE/issues/122) fixed for the
  corpus-size caveat: the GSD shift is **not available as a post-hoc explanation
  of a TerraMind loss unless it is also priced as a discount on a TerraMind
  win.** Deliberately **no threshold and no drop condition** are attached, unlike
  [#134](https://github.com/JStoschek/DEMAE/issues/134)'s materiality check. #134
  could fix 0.99 because cosine has a familiar scale; the statistics below have
  none, so any threshold would be invented to look principled — and a threshold
  would imply this instrument can settle whether the shift *costs* TerraMind
  anything, which it cannot. Refusing a statistic that would overstate what it
  decides is the same call §5 makes against pooled mean signed error and §3.3
  makes against upgrading C4 to a metric delta.
- **The shift is measured, not asserted — as an input-side diagnostic on the
  tokenizer's own quantizer.** TerraMind quantizes with **FSQ: `quant_proj`
  (Conv2d 768->5, 3,845 params) to a 5-d latent, then a parameter-free rounding
  onto a `8-8-8-6-5` grid, 15,360 codes** ([`p118-terramind-encoder-probe.md`](p118-terramind-encoder-probe.md) §5.2).
  So the **distance from the pre-quantization latent to its assigned code**, and
  the **code-usage entropy** over those 15,360 codes (against a 13.91-bit
  ceiling), read directly on whether an input is off-distribution for *this*
  tokenizer — with no target, no head and no training, and therefore comparable
  across GSDs in a way metres are not. Native 30 m was measured against **two**
  references, because each closes the other's hole: the same geographic windows
  upsampled to 10 m and centre-cropped to 224 px, which fixes the geography and
  varies bandwidth; and a sample of real TerraMesh DEM tiles, which fixes
  bandwidth and varies geography. *(That description of the first reference was
  incomplete as pre-registered, and #140 says so rather than quietly rewriting
  it: at a fixed 224 px, "upsample to 10 m and crop" also cuts the ground extent
  from 6.72 km to 2.24 km, so the arm varies extent as well as bandwidth and the
  two references do not in fact close each other's holes. That is what the fourth
  arm below exists to fix.)* It is encoder-only — 90.2M of the 1.15 GB
  checkpoint, counted in the record and matching #118 §8 — the
  same parts A5 and A9 already load — so it costs no new regime, no retrained
  head and no scope reopen. **Its limit is stated wherever it is reported:** it
  measures whether 30 m inputs are off-distribution for the tokenizer, *never*
  what that costs at void filling. Only a retrained head answers the second
  question, and that is what the next bullet declines to build.
- **Measured, before any arm ran: the shift is not visible in the quantizer.**
  [#140](https://github.com/JStoschek/DEMAE/issues/140) ran the diagnostic over
  300 tiles per arm, 58,800 tokens each
  ([`p140-gsd-quantizer.md`](p140-gsd-quantizer.md)). **Native 30 m is
  indistinguishable from TerraMesh's own val DEM tiles**: the code-entropy gap is
  **−0.018 bits, 95% CI [−0.175, +0.136]** against a 13.907-bit ceiling, and the
  quantization residual — in FSQ level units, where each axis's grid spacing is 1
  — is marginally *smaller* for native 30 m (0.4660) than for TerraMesh (0.4676).
  A **fourth** arm was added to separate bandwidth from the ground extent it is
  confounded with at a fixed pixel count, and it is what makes the reading safe:
  **the bandwidth component alone is 0.0093 bits**, against 0.323 for extent and
  0.315 for geography — 35× smaller than either thing the two references confound
  it with. **And the direction runs the other way from the intuition the old
  phrasing carried:** putting these windows onto TerraMind's native 10 m grid —
  TerraMesh's own recipe — *lowers* code entropy by 0.332 bits and *raises* the
  residual, so the next bullet's out-of-scope 10 m regime is not a correction the
  plan is passing up. Two things keep all of this small, and travel with the
  numbers wherever they are quoted. **The limit:** this reads on whether 30 m
  inputs are off-distribution for the tokenizer, *never* on what that costs at
  void filling. **The scale:** every arm, TerraMesh's included, reaches only ~7.1
  of 13.907 bits with two thirds of latent dimensions pinned to a terminal level,
  so no arm's absolute level means anything on its own and only the differences
  between them are read. Re-run under ADR-0011's re-derived `pos_emb` as well as
  the checkpoint's own, because the arms drive the latter and that is a departure
  from the ADR: the levels shift ~0.10 bits and every comparison above survives
  (`p140-gsd-quantizer.md` §6).
- **The caveat stays, and stays symmetric.** A measurement that finds no shift is
  not a licence to drop it — it is what stops the caveat being used later, in
  either direction, as an unpriced explanation. #122's rule still governs: not
  available to explain a TerraMind loss unless also priced as a discount on a
  TerraMind win. **No threshold was set before the run and none is set after it.**
- **Not considered: a secondary 10 m TerraMind-native regime.** Ruled out of
  scope at this map's charting, and reaffirmed here on evidence stronger than the
  ruling had. It would be built by bilinearly resampling the same geographic
  windows to 10 m and center-cropping to 224 px — 2.24 km instead of 6.72 km —
  with TerraMind's head retrained, run once at 20k. It fails the test A8 and A10
  failed: **it closes an objection, it does not carry a claim**
  ([#128](https://github.com/JStoschek/DEMAE/issues/128)). Three costs the GPU
  bill hides, none of them compute — one extra scored pass is ~$0.03 against §7's
  $0.99. It would **add no information**, since those 10 m tiles would be GLO-30
  upsampled, TerraMesh's own recipe, so the fill target inside the void would be a
  smooth interpolant of the very 30 m posts the primary regime scores against, and
  its metres would not be comparable to any other row in the table. It would put
  **DEMAE** off its native regime in order to put TerraMind on its own. And it
  would need a 10 m tile variant, masks redrawn at that geometry, and §3.4's mask
  bank rebuilt for it — 104 MB and 21 minutes for the 30 m geometry alone. What
  replaces it is the caveat and diagnostic above. Settles
  [#139](https://github.com/JStoschek/DEMAE/issues/139).
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

**The caveat is pre-registered, and it runs in both directions.** *(Registered
rule **R2**, asserted in
[ADR-0013](../adr/0013-task1-pre-registration-register.md); the per-modality
citation requirement above is **R4**.)* A data
advantage to TerraMind is only ever cited as an explanation of a DEMAE *loss*,
which makes it an escape hatch unless it is also priced as a discount on a DEMAE
*win*. It is stated once, here, before any number exists: if DEMAE wins C1 on
~3x less ground of the same product at matched compute, that is a stronger
result than a bare win, and if DEMAE loses, this is the one pre-registered
alternative explanation. Reaching for it afterwards is not available.

**ViT-S is one appendix row, not a capacity bracket.** A4-S at 20k, one seed,
asking whether DEMAE's extra 4x bought anything under the frozen probe.
No claim in §0 depends on it and it is not a co-primary: it is cut from A2/A3
and from the label-efficiency sweep, and it has no TerraMind counterpart at all
(no ViT-S DEM tokenizer is published). Its price is why it is one row:
in GPU minutes ViT-S is ~$0.15, but at full seeds it would have been **16 of the
48 scored eval passes** this plan carried before this section was rewritten — a
third of §7's largest line — plus eight extra rows in a table whose
combinatorics are already the binding constraint. Cut to a single-seed cell,
it costs 1 pass and keeps the finding.

**A6-S is dropped, and not because the checkpoint is missing** *(amended
2026-08-12 — [#147](https://github.com/JStoschek/DEMAE/issues/147); evidence in
[`p147-vit-s-mae-checkpoint.md`](p147-vit-s-mae-checkpoint.md))*. This paragraph
used to register a **pair** — A4-S and A6-S — making the row a
DEMAE-vs-ImageNet-MAE capacity observation. The build found that
`pretrain.config.Init` asserts "no ViT-S MAE checkpoint exists at any
provenance", checked it, and found the assertion **too strong**: one exists,
`vit-s_imagenet_mae.pth` from `lambert-x/medical_mae` (WACV 2023), Apache-2.0,
with a timm-format mirror. So A6-S was runnable. It is dropped on a different
ground. Its publisher states no epoch count, no confirmation of which ImageNet
(the paper's Table 1 says "ImageNet (14M)", the mirror says `in1k`), and no
quality number on any axis but a chest-X-ray fine-tune. A6's own init was chosen
in [`p101`](p101-mae-checkpoint-licence.md) §6.1 *precisely* on the
epochs-and-linear-probe axis, at a priced 2.9-point cost — so an A6-vs-A6-S
delta would confound capacity with trainer, schedule and possibly corpus, which
is not the capacity observation this paragraph asked for. A4-S stands: it is
DEMAE's own ViT-S with a full training record. **The ViT-S appendix is one cell,
not two.** Re-adding A6-S costs one config line if a later session wants it with
the caveat printed; nothing here is hard to reverse.

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

**Which DEMAE is the headline: R5, unconditionally.** *(Registered rule **R3**,
asserted in [ADR-0013](../adr/0013-task1-pre-registration-register.md).)* C1 and
C3 are evaluated on
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
4. The head predicts a **residual over the A1 fill**, not raw elevation. This
   means every learned arm starts from the classical baseline and must
   *improve* it; a head that learns nothing scores exactly A1, which makes the
   table legible.

   **The precedent has two halves, and they are different objects**
   *(amended 2026-08-11 — [#129](https://github.com/JStoschek/DEMAE/issues/129);
   this bullet used to call the residual "current best practice in the
   DEM-correction literature (GrounDiff, FathomDEM-style pipelines)", which
   attributes to GrounDiff both a claim it never makes and a base surface it
   does not use, [#124](https://github.com/JStoschek/DEMAE/issues/124) §6.1)*:

   - **Residual over an *interpolated* base is the classical void-filling
     convention**, and it predates the learned literature by ~20 years:
     Grohman, Kroenung & Strebeck's delta surface fill (*PE&RS* 72(3), 2006) and
     **Luedeling, Siebert & Buerkert's TIN delta surface** (*ISPRS J.* 62(4),
     2007), whose own §3.1 describes this construction in words — "adding this
     difference surface to the SRTM base surface results in an elevation model
     that accounts for the varying vertical bias within the void". Substitute
     "A1 fill" for "SRTM base surface" and "head output" for "difference
     surface" and it is this bullet. That is the citation this plan wants: a
     void-filling paper, with a base surface interpolated across the void.
   - **Residual over the *input* surface is what modern learned DEM correction
     does**, which is a different base. GrounDiff's residual is the nDSM over
     the observed DSM (that paper's §3.2: **Eq. 4** emits the residual `r̂` and
     per-pixel confidence logits `ℓ`, and **Eq. 5** gates the ungated DTM
     branch `s − r̂` against the DSM `s` itself), and FathomDEM's output
     is "the per-pixel error in the COPDEM values". The evidence worth taking
     from that lineage is a measurement rather than a convention — GrounDiff's
     **Table 4 ablation**, "predicting residuals (nDSM) instead of absolute
     elevations improves accuracy by 17%".
   - **Neither modern paper does void filling at all.** GrounDiff is DSM→DTM
     ground extraction over complete rasters and gap filling appears nowhere in
     it; FathomDEM corrects an observed surface it never has to invent. That is
     precisely why their base differs from this plan's: **inside a void there is
     no input to be a residual to**, so A1 has to manufacture one. The two
     halves are cited for different things — the classical line for the
     construction, the learned line for the 17%.

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
head-train tile (§3.4). Training masks are *not* drawn on the fly; the bank is
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
  with margin; the change is 1.2% of the large bucket's diameter range and no
  *reported* quantity moves. (It does move two measured ones, both descriptive
  and both in the next bullet: the blob retry rate and the top decile's
  placement freedom were counted against the 160 px top and are re-measured
  there — [#150](https://github.com/JStoschek/DEMAE/issues/150).) **The boundaries are set by the 40% area cap and the
  equivalent-diameter geometry, and by nothing external.** This bullet used to
  add that the two upper buckets "still bracket Diff-DEM's published protocol";
  that clause is deleted with the A8 arm
  ([#128](https://github.com/JStoschek/DEMAE/issues/128)) and is deliberately
  *not* replaced by another competitor's protocol. Nothing about the parameter
  changes — only its stated reason, which was resting on a comparability §0
  now denies. **For the record of what that clause was worth even on its own
  terms** ([#124](https://github.com/JStoschek/DEMAE/issues/124) §3.2,
  [#129](https://github.com/JStoschek/DEMAE/issues/129)): the match it asserted
  was in *file pixels only*. Diff-DEM buckets 64–96 / 96–128 / 128–160 on a
  256 px tile at 10 m, so this plan shares two of its three ranges even
  nominally — 64–96 exactly, 128–160 only up to the 158 top set above — while
  **the middle one, 96–128, is omitted** and a small bucket (16–48) is added
  below the whole range. And a shared pixel boundary is not a shared void: at
  30 m the same pixel count is **3× the ground extent**. What the pixel match
  approximately preserves is the void's size *relative to its tile* — 25–62% of
  tile width there against 29–71% here — which is a **design convenience**, a
  reason the two protocols probe voids of the same relative severity and never
  a reason their metres could share a column. Why the size axis is worth having
  at all is a terrain argument,
  measured in [#119](https://github.com/JStoschek/DEMAE/issues/119) and
  [#126](https://github.com/JStoschek/DEMAE/issues/126), not a comparability
  one. One void per tile per eval instance; every eval tile
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
  instance count suggests. **Both figures were taken at the 160 px top the
  size-bucket bullet above then lowered to 158**, so re-measured on the shipped
  generator at the settled boundaries they are a median of **1,100** placements
  (min 100) and **1.337×** retries in the top decile, 1.158× over the large
  bucket as a whole ([#150](https://github.com/JStoschek/DEMAE/issues/150)).
  The conclusion is the one stated here and is unchanged; what moved is the
  size-bucket bullet's claim that *no* measured quantity moved with the top.
- **The real-void family (iii) carries no bucket axis.** Two cells measured at
  true 1 arc-second returned 139 and 175 components in 16 to 48 px, 2 and 6 in
  64 to 96, and **zero** in 128 to 160. The full 24-granule draw is steeper still
  in its skew — **8,824 / 438 / 11** across the three sub-rows
  ([#135](https://github.com/JStoschek/DEMAE/issues/135)) — so 128 to 160 is not
  empty, just three orders of magnitude below the headline row.
  Forcing 667-per-bucket would mean either
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
  granule contributes the void's *shape* and nothing else. The window is
  **224 px centred on the component**, which is the ground the void actually came
  from; p135 §3.3 shows the answer is indifferent to that choice. That code path
  is checked rather than asserted: recomputing `slope_mean` through it reproduces
  five committed catalog rows of `N28E086` to **0.0**, and every COG is asserted
  to be 3600^2 at exactly 1/3600 degree with its pixel centres on whole
  arc-second multiples before it is read. **Scoring the voids in place is not an
  option, and the reason is the split, not the truth:** GLO-30 is complete at
  those coordinates, but none of the 24 source cells falls in a committed test
  geocell, so an in-place score would be a score on train-side ground.
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
  eval-instance contract, split as drawn. Counts below are **measured** over
  #127's 24 granules ([`p135-void-ruggedness.md`](p135-void-ruggedness.md)), not
  projected — 9,978 scored components of 10,291 at `d ≥ 16 px`, after a 182-component
  `d > 160` exclusion and 1.3% attrition on §3.2's placement tests:
  - **real-small** (16 to 48 px) — the headline row, **8,824 instances**. Its
    honest comparison partner is the *synthetic small bucket within the same
    ruggedness band*, not the synthetic pooled number, and tables state that
    explicitly.
  - **real-medium+** (64 to 96 px) — **438 instances**, reported separately so the
    headline row stays size-comparable to synthetic small.
  - **128 to 160 px** — **11 instances**, reported as an **instance count with no
    metric attached**. Publishing a mean over eleven would invite a significance the
    number cannot carry.
  - **The sub-rows do not tile the 16 to 160 px range, and 7.1% of the draw falls
    in the gaps** — 604 components at 48 to 64 px and 101 at 96 to 128 px. They are
    **excluded, not silently dropped**: the boundaries exist to keep real-small
    size-comparable to synthetic *small* (16 to 48) and real-medium+ to synthetic
    *medium* (64 to 96), so widening them to cover the gaps would destroy the
    comparability the split exists for. The published histogram states the excluded
    count beside the scored ones.
- **Not stratified by ruggedness after all — it is an R4 family, measured.** The
  design was to break every sub-row out by §5's four ruggedness bands, on the
  reasoning that p119 measured the classical baseline varying ~40x across them
  against ~3x across size, so a pooled real-void number would be a statement
  about which terrain the voids happened to land on. That reasoning stands and
  the stratification is nonetheless vacuous, because **the realised histogram is
  99.94% R4**: of 9,978 scored components, R4 holds **9,972**, R3 holds **3**,
  and R1 and R2 hold **3** between them
  ([`p135-void-ruggedness.md`](p135-void-ruggedness.md), all 24 granules, both
  regions, every size sub-row). §5's **100-instance floor** therefore fires on
  three bands of four, and what it produces is not a table with a thin band but a
  **single-band table with a footnote** reading `R1 2, R2 1, R3 3`. This section
  previously predicted "concentration in R3/R4" with "a thin or empty R1" as the
  designed outcome; the prediction was right in direction and too narrow in
  degree — R3 is as empty as R1 — and it is replaced here by the measurement
  rather than left standing as an expectation in a pre-registered document.
  Neither the window choice nor the predicate's open flag can account for it: the
  corpus's own grid window and a 448 px window move **six of 9,978** components
  between them, and excluding the arguable class values 11 and 25 moves **one**.
  The consequence for the claims is that **C4 is an R4 claim** (§0), because R4 is
  the only band with real instances to rank arms in.
- **What the family is for.** It is not a fourth point of metric comparison
  against the bucketed synthetic families — with a 94%-small draw against
  bucket-balanced synthetics, a difference in metres would confound shape with
  size. It carries **C4**, an *ordering* claim: does the arm ranking established
  on synthetic blobs still hold when the void shapes are real? Ordering is
  immune to the size confound because every arm sees the identical instances.
  Real voids are strikingly non-compact (perimeter ~2.0 to 2.2x that of an
  equal-area disc) and concentrate in steep terrain and radar shadow. **The
  transplant preserves the void's shape and its band, and not its native
  ruggedness** — an earlier draft of this bullet claimed it "preserves that
  difficulty rather than averaging it away", and that is measurably false at the
  top of the range. The voids come from terrain with a median `slope_mean` of
  **31.7 degrees**; the eligible test pool's R4 band runs **8.35 to 33.08 degrees**
  with a median of **12.95**, and **34% of the voids are steeper than the
  ruggedest tile in the whole pool** (p135 §5). Matching tighter than the band is
  not available: 156 pool tiles sit at or above 25 degrees against 8,824
  real-small instances, i.e. 57x reuse against 2.6x for the band as a whole. So
  band matching is the only feasible rule, it is coarse at exactly the end of the
  range this family lives at, and the writeup says so rather than claiming a
  fidelity the draw does not have. **Do not upgrade C4 into a metric-delta claim without first adding a
  size- and tile-matched synthetic control** — that control was considered and
  deliberately not built, because the ordering claim is what this draw supports.

### 3.4 Committed artifacts

`benchmark/` module in the repo's stage-per-module style: tile lists, the two
mask artifacts, the mask generator's config hash, and a dataset card
documenting all of the above plus the terms below.

*(A "Diff-DEM cross-eval (transfer test)" section stood here until
[#128](https://github.com/JStoschek/DEMAE/issues/128). It described the A8 arm,
which the map ruled out of scope at charting; the section simply outlived the
ruling. Both external cross-evals are now closed off — see §0.)*

**Data terms, as the distributors state them.** This is a record of what the
distributors publish, not legal advice, following the precedent
[`p101-mae-checkpoint-licence.md`](p101-mae-checkpoint-licence.md) set for this
repo. **Nothing here is a redistribution question**, because the module commits
**tile lists and masks, never tiles** — a property worth stating rather than
leaving as a happy accident, since it is what keeps the artifact freely
shippable under all three sets of terms.

- **Copernicus GLO-30** (every benchmark tile) — available "on a free basis for
  the general public" under the ESA Copernicus Contributing Missions user
  licence, which requires acceptance and carries an attribution requirement. It
  is *not* a standard open licence such as CC BY. This is the primary data
  behind every arm, and until [#128](https://github.com/JStoschek/DEMAE/issues/128)
  the plan recorded no terms for it at all.
- **SRTMGL1N v003** (the real-void family's mask shapes, §3.3) — LP DAAC states
  the dataset "is openly shared, without restriction, in accordance with" NASA's
  EOSDIS data use policy. The plan previously called this "public domain", which
  is a legal conclusion the distributor does not make.
- **The synthetic masks, the mask bank and the tile lists** — this repository's
  own, under its licence.

**The two mask artifacts.**

- **Eval mask list** (parquet: `tile_id, family, bucket, mask_rle, seed`) — one
  row per eval instance, 27,138 synthetic rows plus the real-void family's
  **9,273**, so ~36.4k rows. *(Amended 2026-09-02 — the synthetic count read
  "30,000" until §3.2 was filtered to the **4,523** test tiles that are clean
  ground. A synthetic void is placed by geometry alone and the generator never
  opened the tile, so 400 of the 30,000 held painted ocean fill and 8 were
  entirely ocean — which leaves every derivative metric undefined, and one NaN
  takes a whole cell of the table with it. The 477 part-ocean tiles are dropped
  whole; §3.1's test draw is unchanged at 5,000. §3.3's hosts were always the
  clean tiles, so this also makes both halves of the artifact one population.)*
  *(Amended 2026-08-13 —
  [#153](https://github.com/JStoschek/DEMAE/issues/153). This line read "9,978"
  until the build wrote the file. **9,978 is §3.3's scored-component count**, and
  it includes the 705 components that placed cleanly and fell in the gaps between
  the sub-rows — which the very next clause of §3.3, and §5's histogram sentence,
  **exclude** from the published rows. The rows are the published instances,
  8,824 + 438 + 11 = **9,273**, and the 705 are recorded as a count in the
  artifact's metadata rather than as rows in it. Nothing moves but the arithmetic:
  no registered rule states either figure, and §3.3's two counts were already two
  names. This is the same conflation `benchmark.voids` warns about in its own
  docstring — one word, "scored", for two questions.)* The real rows repeat
  `tile_id`: every one of them needs a host in its own band and the test draw
  holds **1,111 clean R4 tiles** (measured; ~1,250 was the estimate), so a host
  carries **8.3 real void shapes on average** — median 8, at most 23, since each
  void picks its host off its own keyed seed rather than being dealt round-robin.
  §5's bootstrap resamples tiles for this reason.
- **Train mask bank** (parquet: `tile_id, epoch, family, bucket, mask_rle,
  seed`) — K = 50 draws per head-train tile, 1,000,000 rows, **104 MB** as the
  zstd parquet `freeze` writes, measured on the shipped generator's own output.
  (This bullet used to say 740 MB, which is the *uncompressed* RLE payload rather
  than the artifact — [#151](https://github.com/JStoschek/DEMAE/issues/151). The
  rule that keeps it out of git is the corpus shards' rule and not a size
  threshold, so nothing turns on which figure it is.) Generated once in
  **21.1 min** of one CPU core, then read by every arm, seed and config.
  Each tile's K draws are **apportioned** over §3.2's six (family, bucket) cells
  rather than sampled across them, and ordered per tile — so no tile trains on a
  lopsided geometry mix, and no epoch trains every tile on one cell
  ([#151](https://github.com/JStoschek/DEMAE/issues/151)). It is the one committed
  input that is committed **as a hash rather than as bytes**: 104 MB cannot go in
  git, so `freeze` writes `train_mask_bank.json` beside the tile lists carrying the
  bank's md5, a parquet-independent hash of the draw itself and the generator config
  hash, and `run` refuses a bank whose md5 has moved rather than reading it. The
  determinism contract is preserved — the mask sequence a run sees is still a
  committed input, its *hash* rather than its bytes.
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
  appendix cell (A4-S) runs at 20k with a **single seed**, and is
  reported as a single observation without a ± — it answers a capacity-scaling
  aside that no claim in §0 rests on, and buying it error bars would cost more
  eval passes than the primary comparison's entire label sweep. *(Was a pair
  until [#147](https://github.com/JStoschek/DEMAE/issues/147) dropped A6-S —
  §2.3.)*
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
  priced. Measured on the shipped generator, a draw averages **0.947 ms** — but that mean
  spans two orders of magnitude: a rectangle is a fraction of it and a large blob dominates
  it, because the blob family dilates a random walk to a target area. §7's table implies
  **31.2M training tile presentations**; drawn on the fly that is **8.2 CPU-hours**, against
  the program's 5.09 GPU-hours and the 1.9–5.1 h eval line, which §7 called its largest until
  [#175](https://github.com/JStoschek/DEMAE/issues/175) measured A7 at 3.44 h and put a GPU line
  inside that range.
  (Every figure in this bullet was re-derived by
  [#151](https://github.com/JStoschek/DEMAE/issues/151): the count was counted against §7's
  table *before* §2.3's rewrite and [#147](https://github.com/JStoschek/DEMAE/issues/147), and
  the 2.00 ms/draw primitive beside it was p121's prototype rather than the shipped generator.
  The old readings were 34.4M, 19.1 CPU-h, 2.56 GPU-h and a 1.6–4.1 h eval line.) Per
  epoch over 20k tiles it is **19 s** of one core against a **4.6 s** GPU step (head 1.0 s +
  inline ViT-B 3.58 s) — the loader running **4.1×** the GPU it feeds, and still at parity with
  eight workers. The bank collapses 31.2M draws to **1M**: 21.1 minutes once,
  104 MB, and per-epoch variation is preserved *exactly*, since 50 committed masks
  per tile is what 50 on-the-fly epochs would have produced. It is the same trade that killed
  the feature cache, read the other way round — cache the artifact that is cheap to store and
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

No training. Mask = patches touching the void; the decoder reconstructs them
and the non-void pixels are composited back.

**MAE was pretrained on *random* masks and this benchmark scores *contiguous
block* masks, which is a distribution shift.** That is a statement about the
mechanism and not a prediction. The sentence it replaces — "zero-shot numbers
are expected to be modest, improving under A3" — survived every possible
outcome, which is precisely what disqualifies it: A2 far behind is "modest",
A2 close is "modest", and A2 *ahead* of A3 went unaddressed. A falsifiable
version would need a number, and this plan declines to invent one for the same
reason §2.2 declines a threshold on the FSQ diagnostic (**R9b**) — no scale is
available that was not chosen to look principled.

**How the A2 row is read is pre-registered instead** *(registered rule
**R12**, asserted in
[ADR-0013](../adr/0013-task1-pre-registration-register.md) —
[#142](https://github.com/JStoschek/DEMAE/issues/142))*. Three parts, and the
first two apply to every outcome rather than to a tail:

- **A2 carries no claim in §0, and it is insulated from C1 in both
  directions.** A2 is decoder-driven with no trained head; C1 compares frozen
  *features* under an identical head (A4 vs A5 vs A6). A zero-shot failure is
  evidence that the *decoder* does not transfer to block masks, not that the
  representation is weak — and a zero-shot success is not extra evidence for
  C1 either. Insulating in one direction only is the escape-hatch shape §2.3's
  corpus caveat was made symmetric to close
  ([#122](https://github.com/JStoschek/DEMAE/issues/122)). This does not
  retract §1's asymmetry finding, which is about A2/A3 *existing* for an MAE
  and not for a tokenizer — a statement about capability, not about A2's
  numbers.
- **The A2→A3 difference is a joint quantity and is never attributed to task
  adaptation alone.** A3 changes two things at once relative to A2: it adapts
  to the task, and it adapts to the contiguous-mask distribution. The arm that
  would separate them — DEMAE fine-tuned on *random* masks — does not exist and
  is not proposed. So the clause this replaces, "quantifies precisely what the
  task-adaptation bought", is not true of any outcome, good or bad. What the
  difference measures is the cost of the distribution shift and the value of
  task adaptation, summed, and it is reported in those words.
- **The contingency: if A2 does not beat A1 in a majority of the four
  ruggedness bands on RMSE, the row is reported as a negative result** — zero-shot
  MAE inpainting does not transfer from random-mask pretraining to contiguous
  voids — with the prominence §6 item 7 requires, rather than narrated as the
  baseline half of a delta. The trigger is an **ordering against A1**, not a
  threshold in metres, because the gap has no single scale: p119 measured A1
  itself at 0.87 m in the flattest quartile against 38.6 m in the roughest, so
  one number in metres would be four different bars
  ([`p119-fill-headroom.md`](p119-fill-headroom.md)). A1 is the operational
  status quo and is scored in every cell A2 is, so "is zero-shot usable at all
  without training?" is answerable with no invented constant. Majority-of-bands
  rather than the R4 band alone buys robustness by **replication**, which is
  what keeps A2-vs-A1 out of §5's four marked bootstrap pairs (**R6**): A2 is
  single-pass, and the pair is computed like the other 50 and left unmarked.

**There is no zero-shot-vs-fine-tuned figure**, and this section used to
promise one "per bucket". §9 prints three figures and
[#136](https://github.com/JStoschek/DEMAE/issues/136) fixed that set at three,
in the same amendment that pooled the size buckets off the page (**R5**) — so
the promise was owed at a resolution the table no longer has. A fourth figure
for a row carrying no claim also fails the test #136 applied to the rectangle
family. The A2 and A3 rows print side by side across 4 bands × 6 columns,
which is where the difference is read.

### 4.4 UNet baseline (A7)

Standard 4-down/4-up UNet (~17M params), 2-channel input (A1-filled tile, void
mask), residual output, same loss, same schedule length, same seeds. Trained
from scratch at each label-efficiency size.

*(Amended 2026-09-01 — [#162](https://github.com/JStoschek/DEMAE/issues/162).
Two things this bullet left implicit, resolved by the build rather than
invented by it.*

***The width.** "~17M" does not name a base width, and the count is quadratic
in it: the 4-down/4-up shape lands **7.8M at base 32, 17.5M at 48 and 31.0M at
64**. The build ships **base 48 — 17,465,377 parameters**, which is the only
one of the three that reads as "~17M", and the number is derived arithmetically
and asserted against the built module so the printed count is a measurement.
GroupNorm at 16 groups rather than §4.1's 32, because 32 does not divide 48;
GroupNorm rather than the textbook UNet's BatchNorm, because running statistics
would make an eval fill depend on the batch it was scored in and would put a
non-trainable number in the state dict, directly across the one column §6 item
4 exists to print.*

***The input scaling.** This bullet names the two input channels and says
nothing about their units, and §2.1's `normalization` field cannot answer it:
that field is the **frozen encoder's declared diet**, and A7's registry row
declares `none` because it has no encoder to feed. A from-scratch trunk handed
raw metres — the corpus spans roughly -400 to 8,800 — trains badly for reasons
that have nothing to do with representation quality, and a baseline crippled by
its input scaling is not the hard bar this section asks for. So A7 standardizes
the A1 fill per tile over the valid pixels (`land ∧ ¬void`, ADR-0012 §1) and
multiplies its residual back by the same per-tile std, which keeps §4.1's loss
and §2.4's residual convention in the metres they are written in and keeps the
zero-initialized output exactly equal to A1 at step 0. The arithmetic is
`benchmark.extract.normalize`'s `per_tile` and is deliberately not a call to
it — same formula, different question — but the std floor is the same 1e-6 this
repo standardizes by everywhere.)*

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
   local; the bank is the 33-minute line, §3.4).
2. A1 baseline numbers + unit tests on norm inversion and mask bookkeeping.
3. *(was: precompute frozen features — deleted, §4.1. There is no feature
   precompute step; the encoder runs inline.)*
4. Tune head hyperparams on A6/val; freeze.
5. Run the **four primary frozen cells at three seeds each — A4-R5, A4-R4, A5,
   A6 — plus A7 at three seeds, and the ViT-S appendix cell (A4-S)
   at one seed**, all at 20k; then the 1k/5k sweep (R4 and ViT-S are both out of
   it); then A3; then A2/A9 (inference only). **That is the whole run list** —
   the A8 cross-eval and the secondary 10 m regime were both ruled out of scope
   and are no longer tail items waiting for spare budget
   ([#128](https://github.com/JStoschek/DEMAE/issues/128)). *(This step read
   "Run A4, A5, A6, A7 at 20k" until
   [#129](https://github.com/JStoschek/DEMAE/issues/129), which is the arm list
   from before §2.3's rewrite split A4 by init and cut ViT-S to single-seed
   appendix cells; §7's pass count has been on the current shape since
   [#122](https://github.com/JStoschek/DEMAE/issues/122) and the order of
   operations had not caught up. **§7's pass breakdown is the authority on cell
   and seed counts**; this step orders them and should be re-read against it
   rather than maintained beside it, which is how it drifted in the first
   place.)*

## 5. Metrics and statistics

All metrics computed **only inside the void**, in metres, per (family, bucket),
over the fixed test mask lists. **Computed and printed are different questions**,
and this section settles both: every metric is computed for every
(arm, family, bucket, band) cell and committed to the results parquet, while the
*printed* table is a deliberate subset fixed in advance under **The printed
table** below ([#136](https://github.com/JStoschek/DEMAE/issues/136)). Nothing
below that collapses an axis for print discards a number.
The real-void family is the one exception: it
carries no bucket axis (§3.2), so instead of one row per bucket it reports
**two size sub-rows per arm — real-small (16 to 48 px) and real-medium+
(64 to 96 px) — plus a bare instance count for 128 to 160 px** (§3.3). Both
sub-rows are published with the size histogram, including its excluded >160 px
tail and the 705 components that fall in the gaps between sub-rows, so the
reader can see what distribution the number is over and what the 224 px tile and
the sub-row boundaries cut off. Both sub-rows are **R4 rows**: the realised
histogram is 99.94% R4 (§3.3), so the family has one ruggedness column and a
footnote, not four. Comparisons drawn against this family are against the
synthetic **small** bucket within the same ruggedness band, never against a
synthetic pooled number, and carry C4's ordering claim rather than a metric
delta.

The set below is **pre-registered and graded**: **ten candidates were graded**,
of which five are headline columns per cell and four are appendix ones, and
nothing else enters the grading. The printed table is **six columns wide**: those
five, **plus the seam gap**, which is a *derived* column — full-mask minus
eroded-mask slope, free by construction from two numbers already computed — and
was therefore never one of the ten. *(Registered rule **R1**, asserted in
[ADR-0013](../adr/0013-task1-pre-registration-register.md).)*

**Five graded plus one derived, settled 2026-08-12 —
[#146](https://github.com/JStoschek/DEMAE/issues/146).** This section marked six
things headline while R1 and this section's own printed-table arithmetic said
five, and the register cannot gate a build on a count that contradicts itself.
The five/six split is not a disagreement about *what to print*: it is the graded
set being counted in one place and the printed set in another.
[`p123-metric-discrimination.md`](p123-metric-discrimination.md) §3 grades
exactly ten candidates and the seam gap is not among them — it cannot be, since
it is the difference of two rows of that table — and both p123 and
[#136](https://github.com/JStoschek/DEMAE/issues/136) state the outcome as "five
headline columns **plus** the seam gap". No document ever decided the seam gap
does not print; the arithmetic below simply multiplied by five and dropped the
derived column. **The count was corrected, not the metric set**, and it was
corrected while no arm had run — which is the only condition under which this
was correctable at all. R1's roster is now six names, and §5, R1, R5 and every
copy of the arithmetic were amended in one change. The grading is
measured rather than
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

  **NMAD and p95 both come from one source, and it should be named**
  *(amended 2026-08-11 — [#129](https://github.com/JStoschek/DEMAE/issues/129);
  every reading in this paragraph is
  [`p124-voidfill-comparability.md`](p124-voidfill-comparability.md) §7.4's,
  which read the paper in full)*:
  Höhle & Höhle 2009 (*ISPRS J.* 64(4), 398–406) **Table 2**, "Proposed accuracy
  measures for DEMs", whose argument is that DEM error distributions are
  heavy-tailed so the Gaussian measures are the wrong ones. That table lists
  four — median error, **NMAD** (its Eq. 2, `1.4826 · median|Δh − m|`), the
  68.3% quantile of |Δh|, and the **95% quantile of |Δh|**. This plan took two
  of the four without saying so, and the paper gives p95 a better justification
  than the bare "tail behaviour matters" this section used to carry: 95% of
  errors lie in `[0, Q̂(0.95)]` and the remaining 5% "can be of any value",
  which is what makes it robust to up to 5% blunders. The grading above is why
  those two land where they do; the citation is why they were candidates at all.
  Two further notes from the same paper, both load-bearing elsewhere here:
  non-parametric measures are already required for *non-open terrain* by the
  ASPRS Lidar Committee (2004), which is the field's own precedent for robust
  reporting exactly where terrain is hard; and the paper's own suggested
  extension is "modelling the p-quantile as a function of terrain classes" —
  i.e. the stratification bullet below is this paper's generalisation, not an
  invention of this plan.
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
- **Seam continuity (headline, and the sixth printed column): the
  full-minus-eroded slope gap.** **Derived rather than graded** — it is the
  difference of the full-mask and eroded-mask slope rows above, so it was never
  one of the ten candidates p123 scored, which is why R1's roster reads five
  graded plus this one ([#146](https://github.com/JStoschek/DEMAE/issues/146)).
  Free, since
  both masks are already computed, and it sign-separates a fill that breaks at
  the void boundary (+1.33° on the flat-fill control) from every smooth
  interpolator (−0.005 to −0.107°). **This is what carries the operational half
  of C2 now that drainage is gone**, and what makes §4.2's cross-fade auditable
  rather than assumed.

  **Drainage metrics stay out, and the family splits rather than falling
  together** *(closed 2026-08-12 —
  [#143](https://github.com/JStoschek/DEMAE/issues/143))*. The ruling at
  charting was that a 224 px window has no upstream context. That is right
  about **accumulation** and says nothing about **direction**, and the
  imprecision is why the question stayed open. Taken apart: D8 accumulation
  counts upstream pixels, and 224 px at 30 m is **6.72 km across**, so anywhere
  but a local divide the count is bounded by the crop rather than by the
  terrain — which holds *inside* the void exactly as it does at the tile edge,
  so scoring only the void does not rescue it. D8 **direction** does not
  inherit that objection at all: it reads a 3×3 neighbourhood and needs no
  upstream context. It is excluded on the other criterion — it is **aspect**.
  D8 direction is steepest descent quantized to eight neighbours; the aspect
  column above is steepest descent, continuous, off the same ZT derivatives and
  the same pixels. Adding it would re-import the redundancy the
  24,000-instance grading exists to remove, against the one metric measured
  **orthogonal** to the rest of the set (−0.07 vs RMSE, −0.02 vs NMAD).
  **MERIT Hydro's precedent is real and inapplicable here**: basin-shape CSI —
  §5's IoU under another name — needs the whole basin, and this benchmark's
  unit of evaluation is a 6.72 km tile
  ([#124](https://github.com/JStoschek/DEMAE/issues/124)). **Stated plainly:
  the direction half rests on construction, not on a measured correlation** —
  nobody has scored D8 direction against aspect on the p123 harness, and if
  that is ever run and the two prove less redundant than they read, this
  paragraph is what should be reopened. **R1 is unamended**: nothing enters or
  leaves the metric set, only its stated reason changes, the same disposition
  [#128](https://github.com/JStoschek/DEMAE/issues/128) gave §3.2's bucket
  boundaries.

  **Printed as signed degrees to 3 dp, with the sign carried as a visual channel
  rather than left to be read off the digits, and against a threshold fixed here
  before any arm runs: a gap above +0.15° is a seam break; |gap| ≤ 0.15° is
  not** ([#136](https://github.com/JStoschek/DEMAE/issues/136); *registered rule
  **R7**, asserted in
  [ADR-0013](../adr/0013-task1-pre-registration-register.md)*). The scale is
  the reason both halves are needed. It is wildly asymmetric — the break side
  sits an order of magnitude above the intact side, and the intact side is a
  cluster straddling zero — so a raw signed column reads as noise and a bare
  `> 0` rule would fire on that scatter. **+0.15° sits in an empty region**,
  above the measured smooth-interpolator floor of −0.107° and far below the
  flat-fill control's +1.33°, so it is placed by the gap in the measurements
  rather than tuned to a result. The eroded-mask slope §5 keeps as a
  contamination control needs no column of its own: the gap is defined as
  full-minus-eroded, so printing full slope beside the gap makes the eroded
  number recoverable by subtraction.
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
  and loses in mountains must be visible as such. **The precedent is
  FathomDEM Table 1 *and*, closer to home, Luedeling et al. 2007 Table 3**
  *(added 2026-08-11 — [#129](https://github.com/JStoschek/DEMAE/issues/129))*.
  FathomDEM reports ME/MAE/RMSE across 23 landscape and land-cover masks and
  reads them for meaning — its worst accuracy is in steep slopes, and its
  Bare-class stratum is used to show the model correctly leaves already-good
  pixels alone. Luedeling's Table 3 is the same practice **on this exact task**:
  ∆GPS error against 187,500 differential-GPS points, **inside vs outside the
  void**, in four slope classes (0–10°, 10–20°, 20–30°, 30–60°), and its
  headline is a stratification result — the delta-filled surface's ∆GPS spread
  inside the voids runs **58.8 m in the flattest class against 106.4 m in the
  steepest**. Those two endpoints are what
  [`p124-voidfill-comparability.md`](p124-voidfill-comparability.md) §6.3
  records from the paper, and they are what this citation rests on; the two
  intermediate classes were not read across, so **nothing here claims the rise
  is monotonic**. The sentence above is that table, in 2007, and a
  void-filling paper is the better of the two citations for it. This is
  **load-bearing, not
  cosmetic**: p119 measured the classical baseline's error varying ~40× across
  these four bands but only ~3× across the size buckets
  ([`p119-fill-headroom.md`](p119-fill-headroom.md)), so a pooled per-arm number
  is dominated by the rugged tail and hides the fact that the flattest quartile
  is already at the resolution floor. Where table space forces a choice, drop a
  size bucket before dropping a ruggedness band — **a rule that has now been
  applied rather than held in reserve**: the printed table keeps all four bands
  and pools the buckets away entirely
  ([#136](https://github.com/JStoschek/DEMAE/issues/136)). **Population floor:** a
  (row, band) cell with fewer than 100 instances is reported as a count with no
  metric rather than pooled into a neighbouring band *(registered rule **R8**,
  asserted in [ADR-0013](../adr/0013-task1-pre-registration-register.md))*.
  This binds hardest on the
  real-void family, which takes whatever distribution SRTM gives it. **The
  synthetic rows clear it with room** — the 4,523 clean test tiles × 2 families
  × 3 buckets is **27,138** instances, 1,111–1,158 per (row, band) once the
  part-ocean tiles are dropped (see §3.4's amendment), against a floor of 100. **The real rows have now been
  measured against it, and the rule fires on three bands of four**
  ([#135](https://github.com/JStoschek/DEMAE/issues/135),
  [`p135-void-ruggedness.md`](p135-void-ruggedness.md)): the realised histogram
  is **99.94% R4** — R1 2, R2 1, R3 3, R4 9,972 — so every real sub-row is
  reported in R4 with the other three bands as a footnoted count. The
  pre-registered rule for real-medium+ (*unstratified unless the realised
  histogram clears 100 per band*) is **satisfied and vacuous at the same time**:
  438 instances, all of them in R4, so the stratified table and the pooled number
  are the same number, and it is labelled R4 rather than presented as a pooled
  one. Both projections behind that rule were low — real-small was projected at
  ~3,800 and measured at **8,824**, real-medium+ at ~96 and measured at **438** —
  because p125's per-cell yields came from two Cascades cells tamer than the draw
  #126 settled on. What was pre-registered was the threshold, not the outcome,
  and that is what it was for. **Bootstrap over tiles, not instances**: 8,824
  real-small instances land on **1,111 clean R4 test tiles** (measured — #153), so
  ~8 instances share each tile's terrain and a per-instance resample would
  overstate precision by up to √8. §5's bootstrap already resamples test tiles,
  which is the correct cluster; this is a note against changing it. One knock-on:
  §7's **32k eval instances** predates #126's reshaping and #135's counts — the
  total is **30,000 synthetic + 9,273 real = ~39.3k** (the real half is the
  published instances, not the 9,978 scored components — see "The two mask
  artifacts" above), which moves the two eval lines by ~23% and no conclusion.
- **The printed table** *(pre-registered 2026-08-11 —
  [#136](https://github.com/JStoschek/DEMAE/issues/136); registered rule **R5**,
  asserted in [ADR-0013](../adr/0013-task1-pre-registration-register.md))*.
  Printed in full, the
  grid is **1,716 headline numbers** — 11 arm rows (A1; A2, A3 and A4 each split
  R5/R4; A5; A6; A7; A9) × 26 cells (2 synthetic families × 3 buckets × 4 bands,
  plus the real family's two sub-rows in R4 alone) × 6 headline columns (the five
  graded plus the derived seam gap) — and
  another 1,144 in the appendix set. Two axes come off the page:

  - **The size buckets are pooled**, one number per (arm, band). p119 measured
    error varying **~40× across the bands and only ~3× across the buckets**, so
    the bucket axis is the low-information one, and the pool is well defined
    rather than an artifact of sampling because §3.2 balances the buckets by
    construction (1,250 instances per row-band per bucket). Size does not
    vanish: it moves to §9's RMSE-vs-void-size figure, which C2's "at every void
    size" is read off, and which §9 now marks load-bearing for exactly that
    reason. A monotonicity claim reads better off a curve than off three columns.
  - **The blob family is the headline; the rectangle family is not printed.**
    Blobs are §3.2's "standard proxy for radar-shadow voids", which is what real
    SRTM voids are; rectangles are a sensor-dropout proxy, and **no C-claim
    mentions them** — C1 through C4 were checked one by one against this. They
    are still drawn, still scored and still in the parquet, and carry a one-line
    robustness statement in the memo. A family carrying no claim should not cost
    half the page width.

  What prints is **396 numbers on one page**: 11 arms × 4 bands × 6 columns for
  the blob family, plus 11 arms × 2 sub-rows × 6 columns for the real family in
  R4 — 66 (row, band) cells, six columns each. Parameter counts total and
  trainable ride along per §6 item 4. The sixth column is the seam gap, printed
  as signed degrees to 3 dp against R7's +0.15° bar; **its 66 cells are the same
  66 the C4 rank block has, and the two are unrelated** — the coincidence is that
  both are 11 arms × 6, and nothing should be read across them.

  **C4 gets its own block, and it prints ranks rather than metres.** Pooling the
  buckets would otherwise have deleted C4's comparator, since the real family is
  compared against the synthetic **small** bucket within band (§3.3) and that row
  is now inside the pool. The block restores it: the three R4 instance sets —
  synthetic blob small, real-small, real-medium+ — for all 11 arms on RMSE and
  slope RMSE, **66 rank cells**. Ranks, not metres, because §0 makes C4 an
  *ordering* claim precisely so that a natural-distribution family is never
  differenced against a bucket-balanced one; three columns of metres side by side
  would invite that subtraction whatever the caption said, and a rank swap — the
  thing C4 actually tests — is visible at a glance instead. Restoring the
  comparator by refusing to pool R4 was considered and rejected: it would make one
  band structurally unlike the other three, so the band column would stop meaning
  one thing.

  **The four appendix metrics live in the parquet and are printed nowhere.** MAE,
  p95 and profile curvature were graded into the appendix *because* they were
  redundant or could not separate the A1 variants, so printing them would
  re-import the exact noise the 24,000-instance measurement was run to remove.
  The fourth, eroded-mask slope, is the contamination control this section
  promises to publish — and it is published, by arithmetic rather than by column,
  since the seam gap is full-minus-eroded and the full slope is headline. The
  memo states where the parquet is and what is in it.

- **Statistics:** paired bootstrap over test tiles (10k resamples) on RMSE and
  slope RMSE; report 95% CIs and mark table cells where the CI excludes zero.
  **The marked pairs are four, pre-registered here, not all of them**
  ([#136](https://github.com/JStoschek/DEMAE/issues/136); *registered rule
  **R6**, asserted in
  [ADR-0013](../adr/0013-task1-pre-registration-register.md)*):
  **A4-R5 vs A5** and **A4-R5 vs A6** (C1's two comparisons), **A3 vs A1**
  (C2), and **A4-R5 vs A7 at 20k** (C3's endpoint). Every other pair is computed
  into the parquet and simply left unmarked in print. This section used to
  commit to "each arm pair", which is C(11,2) = **55 pairs × 26 cells × 2
  metrics = 2,860 CIs** at 10k resamples — unprintable, and a
  multiple-comparisons surface nobody had priced. Naming the four before any
  number exists is the pre-registration doing its actual job: which comparisons
  matter is fixed by the claims, not by which ones happen to clear zero.
  Three-seed spread is reported separately from bootstrap CI (seed variance vs
  data variance are different quantities) and lives **in the parquet plus one
  printed line giving the maximum spread observed across the headline table**.
  It is deliberately *not* a ± in every cell: only 5 of the 11 arm rows carry
  three seeds at all (A4-R5, A4-R4, A5, A6, A7 — A1/A2/A3/A9 are single-pass and
  the ViT-S appendix cell is single-seed per §2.3), so a per-cell ± would
  imply a seed structure most rows do not have, and reads as a CI besides. A
  spread large enough to threaten a claim earns a paragraph, not a decoration in
  396 cells.
- **Efficiency footnote:** params trained, params total, and single-tile
  inference time per arm (A9's diffusion decode vs A3's single forward pass is
  a real operational difference worth one honest row — the contrast used to be
  drawn against A8's sampler, and A9 makes the identical point with an arm that
  actually runs).

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
3. There is **one regime, 224 px at native 30 m**, it is DEMAE's home turf, and
   the writeup says so rather than leaving it to be inferred. TerraMind runs
   out-of-distribution here, and the shift is **not** that it is starved of
   resolution: its DEM tokenizer trained on the same Copernicus product
   bilinearly upsampled 3x to 10 m, so it is fed **sharper** inputs than it ever
   saw, across 3x the ground span per patch. That caveat is pre-registered and
   symmetric — not available to explain a TerraMind loss unless also priced as a
   discount on a TerraMind win — and it is **measured on the tokenizer's own FSQ
   quantizer rather than asserted** (§2.2). No retrained 10 m head is run, and
   the writeup does not promise one: that regime would close an objection
   without carrying a claim, and its metres would not be comparable to any row
   in the table (§2.2).
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
   split. **There is no transfer block and no external arm**: both candidates
   the field offers were investigated and neither yields a comparable number
   (§0). Every row in every table is an internal comparison, and the writeup
   says so rather than leaving it to be inferred.
7. Negative or null results (e.g., A7 ties A4 at 20k) are reported with the
   same prominence as wins; C3's low-label sweep is the pre-registered fallback
   claim.
8. Pre-registration: this memo is committed before the first head trains;
   success criteria C1-C4 are defined above the results, not after them. **So are
   twelve rules governing how the evidence may be used**, and this item is their
   index — each is stated in full beside the topic it governs, and all twelve are
   asserted together in
   [ADR-0013](../adr/0013-task1-pre-registration-register.md), which this list
   quotes rather than independently commits. A rule is on it when a *different*
   choice, made after seeing the numbers, could change which conclusion the table
   supports:

   - **R1** — ten candidates were graded; the metric set is **five headline
     columns plus the derived seam gap — six printed — and four appendix ones**,
     nothing else, graded by measured separation (§5).
   - **R2** — the pretraining-corpus caveat is symmetric (item 4 above).
   - **R3** — **R5 is the headline unconditionally**; an R4 win is reported
     prominently but never becomes C1's number (§0, §2.3).
   - **R4** — TerraMind's **20** DEM-tokenizer epochs carry their per-modality
     citation wherever stated, not the paper's 100 (item 4 above).
   - **R5** — which axes the headline table prints: the size buckets pool, the
     blob family is headline, and C4 prints as a rank block (§5, §9).
   - **R6** — exactly **four** arm pairs carry a bootstrap CI mark, named in
     advance, of the 55 computed (§5, §9).
   - **R7** — a seam gap above **+0.15°** counts as a seam break (§5).
   - **R8** — real-medium+ is unstratified unless the realised histogram clears
     **100 per band** (§3.3, §5).
   - **R9** — the GSD caveat is symmetric and is stated as a *bandwidth* shift,
     not a resolution handicap (item 3 above).
   - **R9b** — *a refusal:* the FSQ diagnostic carries **no threshold**, because
     the statistic has no natural scale and a threshold would imply it settles
     what the shift costs (§2.2).
   - **R10** — C3's low-label sweep is the fallback claim, fixed here (item 7
     above).
   - **R11** — the bicubic-vs-re-derived ablation row, A5 at 20k, dropped iff the
     materiality check returns **≥ 0.99**
     ([ADR-0011](../adr/0011-terramind-posemb-rederived-at-14x14.md) D4-5; it
     returned 0.9708, so the row stays).
   - **R12** — the A2 row carries **no C-claim** and is insulated from C1 in
     *both* directions; the A2→A3 difference is a **joint** quantity (task
     adaptation plus mask-distribution adaptation) and is never attributed to
     task adaptation alone; and A2 is reported as a **negative result** if it
     does not beat A1 in a majority of the four bands (§4.3).

   Each is a choice that would be worth nothing decided after the numbers existed
   — a table shape picked to flatter a result, a pair set chosen from whichever
   comparisons cleared zero, a threshold slid to put an arm on the right side of
   it. **Two have already fired against their own projections and held**: R8's
   floor was fixed at ~96 projected instances and met a measured 438
   ([#135](https://github.com/JStoschek/DEMAE/issues/135)), and R11 returned
   0.9708 against a bar of 0.99 ([#134](https://github.com/JStoschek/DEMAE/issues/134)).
   Neither outcome was available to argue with, which is the only property that
   made either worth registering. The numeric entries are additionally committed
   to `task1-prereg-bar.json`, which refuses to load if the phrase it was
   registered in no longer appears above.

## 7. Compute budget

**Measured, 2026-08-10 — [`p120-primitive-costs.md`](p120-primitive-costs.md)
([#120](https://github.com/JStoschek/DEMAE/issues/120)).** The table below replaces an
estimate that was high by 8 to 13x and was denominated in the wrong card at the wrong
price: this project rents an **RTX 5090** at a real **$0.4016/hr**, not a 4090 at $0.34
(see [`phase1-pretraining-plan.md`](../phase1-pretraining-plan.md)'s hardware line and
§12). Every line is a primitive times a count, and the primitives are published so
any new arm can be costed without a fresh run.

**Read the Notes column for which kind of primitive a line carries**
([#175](https://github.com/JStoschek/DEMAE/issues/175), 2026-09-14). There are three kinds here,
not two, and the middle one is why the damage is bounded:

1. **Measured on the rented card.** Only **A7**, timed on the real arm.
2. **Transferred, but anchored on the same architecture.** **A3** (read off Phase 1's own 5090
   rates for a step p120 §3 shows is structurally identical, agreeing to 1.2%) and the
   **inference line** (a ViT-B forward, carried at the ratio measured on a ViT-B). Same-family
   anchor, and the one that held.
3. **Transferred across architectures.** The **head primitive** and **A9's decode** — convnets
   priced by a ratio anchored on a transformer. This is the kind that broke.

A7 was kind 3 and turned out **4.5x low** — the single largest correction this section has taken.
The failure mode is not arithmetic, it is a transferred number reading as a measured one, so each
line now says which kind it is.

Costed for the design §4.1 now specifies — **frozen encoder inline, no feature cache,
training masks read from the committed bank** ([#121](https://github.com/JStoschek/DEMAE/issues/121)):

| Item | Count | 5090-min | Notes |
|---|---|---|---|
| **A7 UNet, 3 seeds + label sweep** | 3 x 50 ep x 20k, + 2 x 50 ep x 6k | **206.6** | priciest single arm, and now by much more than this table used to say. **Measured 2026-09-03 on the rented 5090 ([#175](https://github.com/JStoschek/DEMAE/issues/175))** — the only line here timed on a real arm instead of transferred from the laptop. A UNet epoch over 20k tiles is **68.9 s at base 48** (3 seeds: 68.1 / 69.6 / 69.0; the 118 per-epoch lines the committed logs hold span 66.4-71.0 s, so the epoch time is flat, not warming up), against the head's 1.0 and against **the 15.4 s this row carried until now — 4.5x**. Activations cost a measured **0.24 GiB/sample**, a fact about §4.4's architecture rather than about the box, and the reason the declared batch of 128 needs ~30.7 GiB and does not fit a 32 GB card: `runtime.micro_batch` splits it with the effective batch unchanged ([#162](https://github.com/JStoschek/DEMAE/issues/162)) |
| Head tuning on A6 | 12 configs x 50 ep x 20k | 45.7 | |
| Head training A4/A5/A6 | 4 cells x 3 seeds + 1 appendix cell x 1 seed, x 50 ep x 20k | 39.1 | 13 cell-seeds at the 3.01 min/cell-seed this line already prices *(was 14 at 42.1 until [#147](https://github.com/JStoschek/DEMAE/issues/147) dropped A6-S)* |
| Label-efficiency sweep (1k, 5k; 2 seeds) | 3 cells x 2 seeds x 50 ep x 6k | 5.4 | R4 and ViT-S are both out of the sweep (§2.3) |
| Inference over the eval instances | 37 passes x **40k** | 4.4 | frozen forward is **0.179 ms/tile** at ViT-B; prices an encoder pass, so A2/A3's decode is not in it. 37 is the 38 scored passes below minus A1, which runs no encoder |
| A3 fine-tune (R5 ViT-B + R4 ViT-B) | 20 ep x 20k | 4.4 | fine-tune epoch is **6.66 s**; measured equal to a pretraining epoch. **Derived, not re-measured** — the old 3.1 was ViT-B + ViT-S, re-split at Phase 1's measured 3,001 / 7,291 samples/s |
| **GPU total** | | **589.8 min = 9.83 h** | **$3.95** at $0.4016/hr — **305.6 min / $2.05 for the lines above, plus the two pass-through rows' measured 284.2 min / $1.90 below**, which was outside this total for as long as it was unpriced *(was 163.7 min / $1.09 until [#175](https://github.com/JStoschek/DEMAE/issues/175) measured A7, and 148.2 min / $0.99 before [#147](https://github.com/JStoschek/DEMAE/issues/147), still excluding A9)* |
| **A9 (TerraMind full tokenizer)** | **36,411** instances | **142.2** | **Measured 2026-09-16 by the pass itself, on the rented card — [#168](https://github.com/JStoschek/DEMAE/issues/168).** 8,534 s for 36,411 rows = **0.2344 s/instance**, **2.37 GPU-h = $0.95**. Supersedes p163's transferred **18.6 min / $0.12**, which was **7.6x low**, and #163's own UNet-factor exposure of 83 min / $0.56, which was still **1.7x low**. A capacity pilot on the same card read 0.2320 s/instance *before* the pass was committed to, so this line stopped being a transfer at the moment it stopped being a projection. Part of the residual gap is the card and not the ratio: this is a **5080**, where every other line in this table prices a 5090. Prices the *decode only* — the encoder forward is already in the inference line above, and A9 is one of its 38 passes |
| **A9-NOTHRESH (the same pass, sampler thresholding off)** | **36,411** instances | **142.0** | **#168's fair fight, and the same bill to 0.17%**: 8,519.9 s against A9's 8,534.5 for the same instances, because dynamic thresholding is a clamp applied *on* each denoising step and not a step. Registered because A9's row measures terratorch's published DDIM config more than it measures the tokenizer: that config rescales every decode into [-1, 1], which under TerraMind's own DEM affine is **[-280.6 m, +1621.9 m]**, and **37.7%** of the committed instances sit on tiles above it. Measured over the full pass, mean RMSE is **32.6 m** against the published path's **345.7**, and the split at that ceiling is the whole argument: A9 scores **19.11 below / 753.57 above**, this row **18.73 / 19.47** — flat across a line the published sampler cannot cross |
| Eval passes (CPU, local, single core) | **38** arm-seed passes x **40k** instances | **1.9-5.1 h** | 4.6 ms/instance scoring-only, 12.1 ms end-to-end as [p119](p119-fill-headroom.md) measured it. **No longer unambiguously the largest single line**: A7 measures **3.44 h** ([#175](https://github.com/JStoschek/DEMAE/issues/175)), which sits inside this line's range, so this is the largest only at the end-to-end rate and A7 is the largest at the scoring-only one. It stays the largest *CPU* line, which is the claim that carried the argument |
| Train mask bank, generated once (CPU, local, single core) | 20k tiles x K=50 | **21.1 min** | **Measured by `freeze` itself**, 2026-08-13: 1,000,000 rows, **104 MB**, 21.1 min of one core, and two independent freezes produced the same md5, draw hash and byte count. The draw is **0.947 ms** on the shipped generator against the 2.00 ms p121 measured on the prototype, [#150](https://github.com/JStoschek/DEMAE/issues/150)'s one-pass dilation having roughly halved it. **Drawn on the fly instead this line would be 8.2 CPU-h** at that primitive (17.3 at p121's), since this table's own counts total **31.2M** training tile presentations — both re-derived below ([#151](https://github.com/JStoschek/DEMAE/issues/151)) |

The 38 scored passes are 22 at 20k — A1; A2 and A3 at R5 and R4; A9; the four primary frozen
cells (A4-R5, A4-R4, A5, A6) at 3 seeds; the ViT-S appendix cell at 1 seed; A7 at 3 seeds
— plus 16 for the label-efficiency sweep (3 cells x 2 seeds x 2 sizes, plus A7), which §6
item 7 makes a headline claim and therefore scores rather than merely trains.

**This was 48 passes before §2.3's rewrite** and 39 after it; [#147](https://github.com/JStoschek/DEMAE/issues/147)'s
drop of A6-S takes it to 38. The 21% cut is the point worth reading: the
GPU bill barely moved ($1.03 → $0.97) while the benchmark's largest line dropped by ten
passes, *even though R4 was added*. ViT-S gave back more than R4 took. #117's standing note
that attention and wall-clock are the real budget is what this line measures; the dollar column
is not.

**Why the A7 line was wrong, measured rather than guessed**
([#175](https://github.com/JStoschek/DEMAE/issues/175), 2026-09-14). Not the base width, which
is the first thing to suspect and is wrong: [p120](p120-primitive-costs.md) built the UNet at
**17,465,377 parameters** — byte-for-byte the count [#162](https://github.com/JStoschek/DEMAE/issues/162)
later shipped at base 48 — so the measured architecture and the priced architecture were the
same one. The error is entirely in the **cross-device transfer**. p120 timed the UNet on this
laptop's MPS at batch 32 (13.375 samples/s, so **1,495 s** for an epoch over 20k) and carried it
to the 5090 at the **~97x** device ratio anchored on the MAE pretraining step. The real ratio for
this architecture is **21.7x**. The transfer was overstated by 4.5x, and that factor *is* the
whole of the A7 error.

**p120 named this exact risk and named it in the right direction**, which is why the correction
costs a re-derivation rather than a re-measurement: its §3 says that for the head and the UNet
"there is no same-architecture anchor", that the UNet's transfer is "the weaker of the two", and
that convnets do not gain from `torch.compile` what transformers do. A transformer-anchored ratio
therefore overstates the 5090's advantage on a convnet — the direction this measurement confirms.
p120's own sensitivity table priced a 2x overstatement at 4.99 h / $2.01; the measured outcome,
with only the UNet moving, is 5.40 h / $2.17.

**The other lines, checked rather than assumed.** The rest of the GPU table was derived the same
way, and the lines do not share the exposure equally (A9 is the third case and is taken below,
with the paragraph that prices it):

- **A3 fine-tune (6.66 s/epoch) is safe, and for a stated reason.** It needs no transfer at all:
  p120 §3's internal check establishes that an A3 epoch *is* a pretraining epoch, so the line is
  read straight off R0's measured **3,001.2 samples/s on the 5090 itself**. Transferred and direct
  agreed to **1.2%** — a same-architecture, same-device anchor, which is precisely what the UNet
  lacked.
- **The head primitive (1.00 s/epoch) carries the same defect class and remains unmeasured — but
  it is the smaller half of its own lines.** The head is a convnet with no same-architecture
  anchor, so nothing in this measurement clears it. What *does* bound it is that a head line is
  not the head primitive times a count: §4.1 runs the frozen encoder **inline**, so each 4.6 s
  step is **1.0 s of head on top of 3.58 s of ViT-B forward** — and that forward is kind 2, the
  anchor that held. Only **22%** of head tuning and **27%** of head training and the label sweep
  is the exposed primitive. Scaling *that* share by the UNet's 4.5x and leaving the encoder
  alone: head tuning 45.7 -> **80.6**, head training 39.1 -> **76.3**, label sweep 5.4 ->
  **10.5**, and the GPU total reaches **8.75 h / $3.51**. That is the honest worst case and it
  clears the $25 target by 7.1x. *(Scaling the whole line instead gives 12.7 h / $5.11 — wrong,
  because it applies a convnet-specific error to the transformer forward that is three quarters
  of the line, which is exactly the conflation this section is about.)* **Both figures moved with
  A9's measurement** ([#168](https://github.com/JStoschek/DEMAE/issues/168)): they were 6.69 h /
  $2.69 and 10.6 h / $4.27 against a base that priced A9 at $0.12, and the head deltas they add
  are unchanged — what moved underneath them is the floor. **The first head cell
  should print its per-epoch seconds and this line should be re-derived from it**, the way this
  one just was.

The general lesson is p120's own, now with a measurement behind it: a *transferred* primitive is
not a measured one, and §7 should say which of the two each line is. A7 is measured; A3 and the
inference line are anchored on the same architecture; the head primitive and A9's decode are
transferred across architectures and still owe a measurement.

**The budget is not a constraint and should stop being treated as one.** At $2.17 of GPU
against a $25 target — the CPU hours are the project's own laptop, not rent — no line here justifies a spot-instance strategy or a resume design;
#117's standing note that *attention and wall-clock are the real budget* is now measured
rather than asserted.

**Every line is now costed, and two of them are measured on the rented card**
([#163](https://github.com/JStoschek/DEMAE/issues/163), 2026-08-13;
[#175](https://github.com/JStoschek/DEMAE/issues/175), 2026-09-14;
[#168](https://github.com/JStoschek/DEMAE/issues/168), 2026-09-16). *This paragraph read "every
line is now measured" until #175, which is the distinction that failed: A7 was costed, not
measured, and it was 4.5x out.* A9 was the last **unpriced** line, then for a month the last
**transferred** one. It closes at **$0.95** — **32%** of the GPU program, and the single largest
GPU line in the table, ahead of A7's 3.44 h only in dollars and not in hours. Its 1 s/instance
placeholder would still have been **2.5x** the whole table. What does *not* survive the
measurement is the reassurance that followed: the claim that p120's transfer would have to
overstate the 5090 by **201x** before A9 alone reached $25 was arithmetic on a number that was
itself 7.6x low, and the honest version is that A9 came in at 8x its transferred price and is
still $24 short of the bar. See [p163](p163-a9-decode-cost.md) §8 for the sensitivity table that
bracketed this correctly.

**That exposure was real, and it was still not large enough.** A9's decode was transferred
across the ratio whose UNet case #175 measured as **4.5x overstated**, and A9's decoder is the
one other convnet in the table; this section predicted **83 min / $0.56** at the UNet's measured
factor. The pass ran at **142.2 min / $0.95** — so the direction was right, the mechanism was
right, and the magnitude was *still* 1.7x short even after correcting for the thing that was
known to be wrong. Some of that residual is the card (a 5080 here against the 5090 this table
prices) rather than the ratio. The lesson the third correction teaches that the first two did
not: a transferred primitive corrected by another transferred primitive is not a measurement,
and the only line in this section that has ever been right about a convnet is the one that ran
it. **The head primitive is now the last transferred line**, and it is a convnet too.

Two things [p163](p163-a9-decode-cost.md) found that are **not** budget items and belong to
whoever runs the pass: `decode_quant` sizes its output raster from the decoder's 256 px training
resolution and takes no hint from the 14x14 latent, so the default call returns a **256 px raster
for a 224 px tile**, silently — and the decode is **stochastic unless a `generator` is passed**,
which for a single-pass, no-seed arm means unreproducible rather than merely noisy.

**Why no feature cache** (the §4.1 amendment, priced): precomputing frozen features is
*cheaper in GPU seconds* — 1.31 h against 2.56 h — but the whole difference is **$0.50**,
*(both counted before [#147](https://github.com/JStoschek/DEMAE/issues/147); the no-cache
total is now 5.09 h and the cached figure beside it was not re-derived, so this pair is
flagged rather than half-corrected —
[#151](https://github.com/JStoschek/DEMAE/issues/151). The $0.50 the bullet turns on moves by
cents either way, and [#175](https://github.com/JStoschek/DEMAE/issues/175)'s A7 correction does
not touch it at all: A7 runs no encoder, so its 206.6 min sits on **both** sides of this
comparison and cancels out of the difference.)*
and against that it costs 30.3 GiB, 7.9 TB of re-reads, and a design constraint §4.1 did
not name: a cached feature fixes the A1-filled input, hence the mask, so head training
would see **one void per tile for all 50 epochs**, contradicting §3.2's per-epoch masks.
Run the encoder inline.

**Why a mask bank instead** ([#121](https://github.com/JStoschek/DEMAE/issues/121)):
deleting the feature cache made mask generation a per-epoch cost, and it had never been
priced. On the shipped generator it is **0.947 ms/draw**, so the 31.2M presentations this
table's own counts imply come to **8.2 CPU-h** — still larger than the low end of the eval
line above and 1.6× the whole GPU program at that total's present 5.09 h *(it read 3.4× against
the 2.42 h this total stood at before [#175](https://github.com/JStoschek/DEMAE/issues/175)
measured A7)*. Per epoch over 20k tiles that is 19 s of one core
against a 4.6 s GPU step, so on-the-fly masks make head training **loader-bound** by 4.1×,
which is the exact property the deleted cache was supposed to buy. Banking K = 50 draws per
tile cuts 31.2M draws to 1M — **21.1 min once, 104 MB** — with per-epoch variation preserved
exactly, because 50 committed masks per tile is what 50 on-the-fly epochs produce. Both
decisions are the same trade read consistently: cache what is cheap to store and expensive to
recompute, not the reverse.

**Re-derived 2026-08-13 ([#151](https://github.com/JStoschek/DEMAE/issues/151)), closing the
#121/#122 merge's open note.** The 34.4M presentation count this section used to print was
derived from §7's table *as it stood before* §2.3's rewrite, which cut the ViT-S appendix
cells to a single seed and took R4 out of the label-efficiency sweep, and before
[#147](https://github.com/JStoschek/DEMAE/issues/147) dropped A6-S. Against the table above
it is **31.2M**, and the arithmetic is published for the reason p121 published the original —
so the next rewrite of §2.3 can re-derive it rather than re-measure it:

| line | presentations | against p121's count |
|---|---|---|
| Head tuning on A6 (12 configs x 50 ep x 20k) | 12,000,000 | unchanged |
| Head training A4/A5/A6 (4 cells x 3 seeds + 1 appendix cell x 1 seed, x 50 ep x 20k) | 13,000,000 | **15,000,000** — 15 cell-seeds to 13 |
| A7 UNet (3 seeds x 50 ep x 20k, + 2 x 50 ep x 6k) | 3,600,000 | unchanged |
| Label-efficiency sweep (3 cells x 2 seeds x 50 ep x 6k) | 1,800,000 | **3,000,000** — 5 cells to 3 |
| A3 fine-tune (R5 ViT-B + R4 ViT-B, 20 ep x 20k) | 800,000 | unchanged |
| **total** | **31,200,000** | **34,400,000** |

At p121's 2.00 ms/draw that is **17.3 CPU-h** against its 19.1, and **3.4×** the 5.09 h GPU
program rather than 7.5× *(it read 7.2× against the 2.42 h that total stood at before
[#175](https://github.com/JStoschek/DEMAE/issues/175))*. Both ratios name the total they divide
by, because that total has now moved three times and a bare multiple is how this section went
stale the first time.

**The count was not the only stale half — the primitive moved too, and further.** §7 costs
every line as a measured primitive times a count, and this ticket was chartered against the
count alone. Re-measured by `freeze` on the shipped generator, a draw is **0.947 ms** (30,000
eval draws in 28.4 s over the same six cells), not p121's 2.00 ms:
[#150](https://github.com/JStoschek/DEMAE/issues/150) replaced the prototype's iterated 3x3
dilation with one distance transform. So the honest on-the-fly figure is **8.2 CPU-h**, and the
per-epoch loader cost is **19 s** against the 4.6 s GPU step rather than 40 s — **4.1×
loader-bound rather than 8.7×**. Both halves of a "primitive times count" line went stale
together, which is the general lesson worth carrying to the rest of this table: `freeze` and
`run` should print the primitives they actually spend.

**The bank's own row count does not move with any of this.** It is 20,000 head-train tiles
times K = 50 whatever the pass count does, so 1,000,000 was never a stale number — only the
ratio it is quoted against, which is **31.2×** in draws and **23×** in CPU time at the shipped
primitive. Its measured cost moved downwards on both axes: **21.1 min** against the projected
33, and **104 MB** against 740 — the latter never having been the artifact's size at all, but
the uncompressed RLE payload before zstd.

**The conclusion is unchanged and was never in doubt**, at 3.4× or at 1.6×: a loader running
four times the GPU it feeds is loader-bound, and 21 minutes once against 8.2 CPU-h every time
is the same trade the feature cache was deleted over, read the other way round. What this
re-derivation buys is that the *numbers* on the page are now the ones the command prints.

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
- **Contiguous-mask shift sinks A2:** stated as a mechanism rather than a
  prediction, and how the row is then read is pre-registered — no C-claim, C1
  insulated in both directions, the A2→A3 difference reported as a *joint*
  quantity, and a negative result if A2 does not beat A1 in a majority of the
  four bands (§4.3, registered rule **R12**). This bullet used to read
  "expected and framed as the zero-shot vs fine-tuned delta, not buried", which
  named the framing the register now fixes without committing to it
  ([#142](https://github.com/JStoschek/DEMAE/issues/142)).
- **Spot-instance reclaim: not a risk this benchmark has.** Heads checkpoint
  every epoch to the synced directory like every other run in this repo, but
  §7's total is 5.09 GPU-hours with a **58-minute** longest job — one A7 seed, measured
  ([#175](https://github.com/JStoschek/DEMAE/issues/175)); the 46 minutes this bullet used to
  quote was never a job, it was A7's whole 3-seed line at the old primitive, and that line is now
  206.6 min. The longest *job* barely moved, which is the number this argument turns on, so there
  is nothing long-running to lose. *(2.56 until
  [#147](https://github.com/JStoschek/DEMAE/issues/147) dropped A6-S; corrected
  here by [#151](https://github.com/JStoschek/DEMAE/issues/151), which found the
  same stale total quoted in §4.1.)* *(The clause that used to close this bullet —
  "features are memmapped so nothing long-running is unresumable" — went with
  the feature cache in §4.1.)*

## 9. Deliverables

1. `benchmark/` module + committed tile/mask/split artifacts (reproduces every
   table from the repo).
2. The results table and three figures. **The table's shape is pre-registered in
   §5 under "The printed table"** *(amended 2026-08-11 —
   [#136](https://github.com/JStoschek/DEMAE/issues/136))*; this item used to say
   "arms x buckets x strata, CIs marked", which is 1,716 headline numbers with
   CIs on 55 arm pairs and was never printable. What prints is **11 arms × 4
   ruggedness bands × 6 columns for the blob family, plus the real family's two
   sub-rows in R4 — 396 numbers** — with the size buckets pooled, the rectangle
   family in the parquet only, CI marks on **four pre-registered arm pairs**, and
   a separate **C4 rank block** of 66 cells. Then three figures:

   - **RMSE vs void size per arm.** **Load-bearing, and newly so:** #136 pooled
     the size buckets out of the table, making this figure the only place void
     size appears and therefore the whole of C2's "at every void size" evidence.
     Not to be cut for space.
   - **The label-efficiency curve (C3).** The whole of C3 — the sweep's 16 scored
     passes (§7) print here and nowhere else, their numbers in the parquet.
   - **A hillshade panel grid** (truth, voided, A1, A3, A5, A7) on two rugged and
     one flat tile. **Load-bearing, not decorative**: §5 scores no perceptual
     metric, so this figure is the whole of the visual-quality evidence and is
     not to be cut for space.

   **There is no fourth figure for ruggedness, and the reason is the table's
   shape.** The bands are the printed table's column groups at full resolution
   across all five metrics, so p119's ~40× is already the most visible thing on
   the page; a figure would restate it. Dropping the buckets is what bought the
   space to keep the bands whole.
3. A short results memo in `docs/research/` answering C1, C2, C3 with numbers
   and C4 with a rank comparison, plus the fairness checklist and the TerraMind
   caveats.

   **The memo opens on §0's incomparability paragraph, not on a comparison**
   *(amended 2026-08-11 — [#129](https://github.com/JStoschek/DEMAE/issues/129))*.
   §0's "internally controlled comparison, and there is nothing external to
   stand it beside" is a deliverable, not a caveat: it is what stops a reader
   arriving at the table expecting a competitor column and reading its absence
   as an omission. It goes in the memo in the same load-bearing position §6
   gives the fairness checklist. The reason this needs saying is that void
   filling was **selected** partly on the opposite premise —
   [`downstream-task-selection.md`](downstream-task-selection.md) §1 criterion 4
   ranked it because open competitor code "lands next to something". That reason
   did not survive [#124](https://github.com/JStoschek/DEMAE/issues/124) and
   [#128](https://github.com/JStoschek/DEMAE/issues/128), and is struck in the
   memo that carries it. **The task remains the right first choice on the
   reasons that did survive** — the MAE decoder is natively an inpainter, it is
   the cheapest of the candidate tasks, and the money plot (before/after
   hillshades) is strong — so nothing about this plan's existence rests on the
   struck one. Writing that down is the difference between a selection that was
   re-checked and one that was never revisited.
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
     property is that the committed file is what later phases re-read (§3.4). It
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

Diff-DEM (github.com/kylelo/Diff-DEM; IEEE GRSL 2024 — **cited as evidence, not
as an arm**: reading its committed `tif_metric.py` is what established that the
table is externally incomparable, §0) · GeomorPM (doi:10.1080/13658816.2024.2414409;
Figshare deposit doi:10.6084/m9.figshare.25756806.v1, CC BY 4.0 — the second
candidate, closed off in [#128](https://github.com/JStoschek/DEMAE/issues/128))
· SRTMGL1N
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
protocol + baseline bar) · MAE (arXiv 2111.06377) · **Grohman, Kroenung &
Strebeck 2006** (*Photogramm. Eng. Remote Sens.* 72(3), 213–216; the Delta
Surface Fill method — residual over an interpolated base, §2.4 — *not reached in
full text, characterised through Luedeling et al. 2007 who describe it in
detail*) · **Luedeling, Siebert & Buerkert 2007** (*ISPRS J. Photogramm. Remote
Sens.* 62(4), 283–294, doi:10.1016/j.isprsjprs.2007.05.004; the TIN delta
surface — §2.4's construction in words, and **Table 3**, void-fill error by
slope class inside vs outside the void, which is §5's stratification on this
task) · **Höhle & Höhle 2009** (*ISPRS J. Photogramm. Remote Sens.* 64(4),
398–406, doi:10.1016/j.isprsjprs.2009.02.003; **Table 2**, the robust DEM
accuracy measures §5's NMAD and p95 are two of) · GrounDiff (arXiv 2511.10391;
residual over the **input DSM**, and the **Table 4** 17% ablation — cited for
the measurement, not for a convention, and it does no void filling, §2.4) ·
FathomDEM (ERL 2025; residual over the input COPDEM, and **Table 1**'s
mask-stratified reporting) ·
Kornblith et al. (arXiv 1905.00414) for the CKA follow-up if features get
compared directly.
