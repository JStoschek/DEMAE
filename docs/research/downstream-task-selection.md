# Downstream Task Selection for DEMAE

**Research memo, August 2026.** Decides which downstream tasks the pretrained DEMAE
encoder is evaluated on, and specifies the TerraMind encoder comparison protocol.
Supersedes plan §5's four-task sketch: super-resolution is dropped (per project
decision), landslide susceptibility is demoted to an optional side-table, and the
remaining budget (< $50, roughly 100 to 140 RTX-4090-hours at ~$0.34/hr, of which
20 to 30% should stay slack) is spent on two core tasks plus one cross-cutting
representation study.

## 0. Recommendation

**Task 1: DEM void filling** (synthetic masks + real SRTM voids, head-to-head
against the open Diff-DEM baseline). The MAE decoder *is* an inpainter, so this is
the cheapest possible task and the cleanest pretraining-to-task story, and it is
GeomorPM's headline task (task-level comparability).

**Task 2: Landform classification** (rule-generated geomorphons as the primary
label set, the independent Iwahashi & Pike 2018 map as the second). Frozen-encoder
probes plus the low-label-fraction curve: the canonical SSL evaluation and the
portfolio's money plot.

**Cross-cutting study: DEMAE vs. TerraMind's DEM tokenizer encoder** at the
representation level (frozen features, identical probes), run *inside* Task 2's
harness plus a small terrain-derivative regression probe, so it costs almost
nothing extra. This comparison has no published prior art (§3.6): nobody has
probed TerraMind's DEM tokenizer encoder as a standalone representation model.

**Optional Task 3 (only if budget survives Tasks 1 and 2): bare-earth correction**,
predicting the FABDEM (bare-earth) residual from Copernicus GLO-30, validated
against LiDAR. Scientifically the most interesting and the one task GeomorPM never
touched, but the most expensive of the three and the labels carry a
CC BY-NC-SA license. Treat as stretch, not core.

Everything else investigated (landslide detection/susceptibility, super-resolution,
sinkholes, avalanche terrain, hydrography, glaciers, soil/permafrost, canopy
height) is rejected with reasons in §4.

## 1. Constraints and selection criteria

1. **Budget.** < $50 of GPU time remains. Every task must fit in single-digit to
   low-tens of 4090-hours including its baselines. This kills anything needing a
   large decoder trained to convergence on a big corpus.
2. **The PANGAEA bar** (arXiv 2412.04204). Geospatial FMs often fail to beat a
   supervised UNet. Every task reports: DEMAE frozen (linear probe and UperNet),
   DEMAE fine-tuned, ImageNet-MAE-init, and a from-scratch UNet. Geographic
   holdout splits only (the corpus's committed geocell split machinery already
   provides this), never random tile splits.
3. **Signal plausibility.** Elevation alone must plausibly carry the task's
   signal at 30 m. This is the criterion that kills landslide detection and
   sinkholes.
4. **Comparability.** Prefer tasks GeomorPM evaluated (void filling, landform
   classification) or where an open competitor ships code + data + weights
   (Diff-DEM), so numbers land next to something.
5. **Portfolio value.** Each task needs a money plot: before/after hillshades for
   void filling, the accuracy-vs-label-fraction curve for landforms.

New prior art found in this pass that sharpens positioning: **FathomDEM**
(ERL 2025, closed weights, commercial) does GLO-30 bare-earth correction with a
hybrid ViT; **GrounDiff** (arXiv 2511.10391) does diffusion DSM-to-DTM. Nothing
released 2025 to mid-2026 closes the "open-weights DEM-only MAE" niche.

## 2. Task 1: DEM void filling

### 2.1 Why

Void filling is MAE's native objective wearing a different mask shape. Pretraining
already reconstructed 75% of every tile from the remaining 25%; the task is to swap
random patch dropout for contiguous void geometries and evaluate in metres. It is
also GeomorPM's flagship task, real SRTM voids make it more than a toy, and there
is an open, directly runnable competitor.

### 2.2 Data and labels

- **Synthetic voids on held-out corpus tiles.** Use the existing test-split
  geocells. Mask geometries in two families: random contiguous blobs (morphological
  dilation of random walks) and rectangles, in size buckets matching Diff-DEM's
  protocol (64 to 96 px, 128 to 160 px) plus a small-void bucket (16 to 48 px)
  where classical interpolation is strong and learned methods must prove they do
  not lose.
- **Real SRTM void geometry.** SRTMGL1N ("number of scenes", NASA Earthdata,
  free login, but the one-time OAuth consent is a human step) gives a per-pixel
  class raster at 1 arc-second. Voids are **`num > 10 AND num < 201`**:
  SRTM-observed pixels carry the swath count 201 to 224, water is 1/2/5, and
  **there is no value 0 in this product** (SRTM User Guide V3, Table 2) — 0 is
  `SRTMIMGM`'s encoding, a different product. SRTM v2.1 is the non-void-filled
  release, but at 3 arc-second outside the US, so it is a *degraded* fallback
  rather than the same geometry (an upsampled v2.1 mask scores IoU 0.60 against
  1 arc-second truth, missing 39% of void area); prefer the per-tile Wayback
  capture of the retired `dds.cr.usgs.gov` tree over the 13.7 GB archive.org
  tarball if it is used at all. NASADEM_SHHP is not a substitute either: its
  reprocessing filled 52 to 63% of SRTM's voids, so it measures the population
  that survived modern InSAR, not SRTM's. Source regions are the
  Himalaya/Karakoram and the Andes; the Atlas is too void-poor (0.03 to 0.16%)
  to contribute. Protocol: take real void masks, transfer them onto clean GLO-30
  tiles (where truth exists, and on the same lattice, so no resampling), fill,
  score. Real voids concentrate in steep terrain and radar shadow, so stratify
  results by relief or they will look deceptively easy — and they are ~94%
  small, so this family is reported at its natural size distribution rather than
  balanced across the synthetic size buckets. Measured in
  [`p125-srtm-void-geometry.md`](p125-srtm-void-geometry.md).
- **Cross-eval on Diff-DEM's data.** Diff-DEM (IEEE GRSL 2024,
  github.com/kylelo/Diff-DEM) releases its Norway 10 m dataset, mask lists, and a
  pretrained checkpoint. Run their released model and DEMAE on the same tiles both
  ways. Their 10 m Norwegian data is off-distribution for a GLO-30-trained model;
  that is a transfer test, and the writeup should say so.

### 2.3 Method

1. **Zero-shot:** the pretrained MAE as-is, visible patches = non-void pixels'
   patches, reconstruct the rest. Establishes what pretraining alone bought.
2. **Fine-tuned filler:** continue training encoder + decoder briefly with
   blob/rectangle masking instead of random patch masking, loss in un-normalized
   space (voids need metres, and `norm_pix` targets throw away per-patch scale).
   Handle the two known seams: per-tile standardization must be inverted using
   statistics computed from *non-void* pixels only, and patch-boundary seams are
   suppressed with overlapping-tile blending at inference.
3. **Baselines:** bicubic/spline interpolation (the classical operational answer,
   e.g. `gdal_fillnodata`-class methods), a from-scratch UNet inpainter with a
   partial-convolution or mask-channel input, ImageNet-MAE-init variant of (2),
   and the released Diff-DEM checkpoint.

### 2.4 Metrics

MAE and RMSE in metres inside the void, stratified by void-size bucket and by
terrain ruggedness stratum (the corpus's own strata are already computed);
slope and aspect error inside the void (terrain-derivative fidelity matters more
than pixel RMSE); drainage-continuity spot checks and hillshade panels
(masked, filled, truth) for the qualitative figure.

### 2.5 Cost and risks

Roughly 10 to 20 4090-hours total including baselines (the fine-tune is short; the
UNet baseline is the biggest single line item). Risks: norm-inversion bookkeeping
errors (write a unit test that fills a zero-void tile and demands identity);
MAE reconstructions look blurry at high frequency compared to a diffusion model,
so expect Diff-DEM to win on texture and DEMAE to compete on low-frequency
accuracy and speed; report both and say which matters for hydrology.

## 3. Task 2: Landform classification, and the TerraMind comparison inside it

### 3.1 Why

Per-pixel landform classification is the canonical dense probing task for a
terrain encoder, it is GeomorPM-comparable, labels are free and unlimited, and the
low-label-fraction curve (1% / 10% / 100%) is the single most persuasive
foundation-model plot. It is also the natural harness for the TerraMind
representation comparison, because probes are exactly the fair instrument for
comparing frozen encoders of different widths.

### 3.2 Labels, two sets

- **Primary: geomorphons** (Jasiewicz & Stepinski 2013), generated with GRASS
  `r.geomorphon` or WhiteboxTools on the corpus's own tiles. Ten classes, zero
  label noise, unlimited quantity, perfect control of geographic splits. Known
  objection: geomorphons are a deterministic function of the DEM, so this measures
  representation quality, not new science. State that plainly; it is a feature for
  a probing benchmark. Fix and document the algorithm's search radius and flatness
  threshold, since the classes are scale-sensitive.
- **Secondary: Iwahashi & Pike lineage global terrain classification (2018)**
  (gisstar.gsi.go.jp/terrain2018, CC-BY 4.0, global GeoTIFF, 280 m, 40 classes
  grouped into 15). Human-calibrated and independent of the input DEM, which
  answers the circularity objection. Coarse resolution means heavy label smoothing
  at 224x224 GLO-30 tiles; treat as tile-level (majority class) classification
  rather than dense segmentation.

### 3.3 Protocol (PANGAEA-shaped)

For each encoder and each label fraction (1%, 10%, 100% of training geocells):

1. **Linear probe on frozen patch tokens:** a single 1x1 conv on the last-layer
   token grid, bilinearly upsampled to pixel resolution. The
   strictest, decoder-capacity-free measure.
2. **UperNet on frozen features** from 4 intermediate layers (PANGAEA's standard
   dense protocol, also what the TerraMind paper itself uses).
3. **Full fine-tune** of DEMAE + UperNet at 100% labels only (budget).
4. **Baselines:** from-scratch UNet, ImageNet-MAE frozen + same heads.

Metrics: overall accuracy, macro-F1, mIoU, confusion matrices; the money plot is
mIoU vs. label fraction with one line per encoder.

### 3.4 The TerraMind comparison: what exactly to compare

Target model: `ibm-esa-geospatial/TerraMind-1.0-Tokenizer-DEM`. Verified
architecture (from the terratorch source, v1.2.10, and the ICCV 2025 paper):
a **ViT-B/16 encoder** (768-d tokens, 12 layers, ~90M params, fixed 2D sin-cos
positional embeddings interpolated at runtime), followed by a 1x1 conv
(`quant_proj`) down to a **5-dimensional latent**, then **FSQ quantization** with
levels [8,8,8,6,5] (codebook 15,360), then a ~196M-param conditional diffusion
UNet decoder. Trained ~20 epochs on TerraMesh's ~9M DEM tiles at 256 px crops.
Weights are Apache-2.0, so benchmarking and publishing results is unambiguously
fine (attribute the model card and arXiv 2504.11171 / 2504.11172).

"Encoder output before tokenization" has two readings, and the choice matters:

- **`feats`, the 768-d ViT-B output before `quant_proj`: use this as the primary
  comparison target.** It is the representation-rich tensor, structurally
  comparable to DEMAE's ViT-S patch tokens (same 14x14 grid on a 224 px input).
- **`latent`, the 5-d pre-FSQ projection:** literally "before quantization" but an
  extreme bottleneck built for the generative pipeline. A probe on 5 dims loses by
  construction; using it as TerraMind's representation would be a strawman.
  Include it only as an ablation row, labeled as the bottleneck.

Extraction is direct submodule calls (there is no public continuous-latent flag).
Install `terratorch>=1.2.5` plus `diffusers==0.30.0` (required even encode-only):

```python
import torch
from terratorch.registry import FULL_MODEL_REGISTRY
from terratorch.models.backbones.terramind.model.terramind_register import (
    v1_pretraining_mean, v1_pretraining_std,
)

model = FULL_MODEL_REGISTRY.build("terramind_v1_tokenizer_dem", pretrained=True)
model.eval()

MEAN = v1_pretraining_mean["tok_dem@224"][0]   # 670.665 m
STD  = v1_pretraining_std["tok_dem@224"][0]    # 951.272 m

x = (dem_metres - MEAN) / STD                  # global z-score, NOT per-tile; NaNs filled first
with torch.no_grad():
    x = model.prepare_input(x)                 # no-op for DEM
    feats  = model.encoder(x)                  # [B, 768, H/16, W/16]  <- primary target
    latent = model.quant_proj(feats)           # [B, 5, H/16, W/16]    pre-FSQ bottleneck
    # intermediate layers for UperNet probes:
    layers = model.encoder(x, return_intermediates=True)   # 12 x [B, 768, h, w]
```

### 3.5 Fairness pitfalls (name all four in the writeup)

1. **GSD mismatch, the big one.** TerraMind's DEM pixels are GLO-30 *bilinearly
   resampled to the 10 m Sentinel-2 grid* (TerraMesh §3.2): a 224 px TerraMind
   tile covers 2.24 km, a DEMAE tile covers 6.72 km. Run both regimes and label
   them: (a) *same pixel grid*, both encoders get identical 224x224 @ 30 m tiles
   (native for DEMAE, out-of-distribution for TerraMind), which is the only regime
   where token grids co-register for patch-level metrics; (b) *near-native for
   each*, TerraMind gets 10 m-resampled tiles (replicating TerraMesh
   preprocessing) over the same center area, comparison at image level with pooled
   features only.
2. **Normalization confound.** DEMAE's per-tile standardization discards absolute
   elevation; TerraMind's global z-score keeps it. Any target correlated with
   altitude is unfairly easy for TerraMind. Slope-like and shape-like targets are
   fair. Ablation: feed TerraMind per-tile-standardized input too, to measure how
   much of its probe accuracy is absolute-altitude passthrough.
3. **Capacity confound.** ViT-B (90M) vs ViT-S (22M). Report parameter counts next
   to every number; if the ViT-B continual-init DEMAE arm (R4) is usable, include
   it to partially control capacity. Optionally plot probe accuracy vs. encoder
   depth (TerraMind's 12 layers are all extractable) instead of a single point.
4. **Objective mismatch.** MAE features are trained for masked prediction;
   the tokenizer encoder is trained for reconstruction through a 5-d bottleneck
   and was never meant as a representation model. The honest headline is
   "representation content under identical probes," not "which model is better."
   A defensible extension if time allows: also probe the actual TerraMind
   *backbone* (`terramind_v1_base`, which accepts raw DEM via `untok_dem@224`) as
   the third encoder, since that is arguably TerraMind's true DEM representation.

### 3.6 Comparison experiments, in priority order

1. **Probe suite (inside Task 2):** linear probe + UperNet on frozen features,
   geomorphon and Iwahashi labels, both encoders, label-fraction sweep. One table,
   one plot.
2. **Terrain-derivative regression probes:** linear regression from frozen patch
   tokens to per-patch slope, aspect (sin/cos), and curvature. Pure
   representation-content measure, invariant to the normalization confound
   (derivatives are altitude-invariant), and nearly free to run.
3. **kNN retrieval probe:** mean-pooled, L2-normalized tile embeddings, cosine
   kNN by stratum label (the corpus's own k=20 strata are ready-made classes),
   DINOv2-style. Reuses the existing `pretrain.probe` machinery.
4. **CKA (linear) between the two token grids** on identical tiles in regime (a),
   plus mutual-nearest-neighbor alignment on pooled embeddings. This is the
   "how similar are the geometries" figure, dimension-agnostic.

No published work probes this tokenizer's encoder as a representation model (the
TerraMind paper only reports its *reconstruction* quality: DEM MAE 170.7 m,
RMSE 737.2, SSIM 0.974 on an SSL4EO-S12 subset, appendix Table 10). This
comparison is a novel, citable artifact on its own.

### 3.7 Cost

Feature extraction for ~10k probe tiles is under an hour on a 4090 for both
encoders. The full probe suite including UperNet heads, label fractions, and
baselines: roughly 15 to 25 4090-hours. The comparison experiments add
low-single-digit hours on top.

## 4. Optional Task 3, and what was rejected

### 4.1 Bare-earth correction (GLO-30 to FABDEM residual): the stretch task

Predict the per-pixel residual between Copernicus GLO-30 (a surface model:
includes canopy and buildings) and FABDEM v1-2 (bare earth, data.bris.ac.uk,
global 30 m). Head: light UNet-style or UperNet decoder with a 1-channel
regression output. Validate against *LiDAR* DTMs (USGS 3DEP 1 m via
OpenTopography, Norway hoydedata.no, Dutch AHN), never only against FABDEM,
because FABDEM is itself a gradient-boosted model's output and matching it only
proves distance to somebody else's random forest. Metrics: MAE/RMSE vs. LiDAR
stratified by canopy cover and slope, % pixels within ±2 m, plus the
before/after hillshade of a forested valley (a spectacular figure).

Why it is compelling: it is the one task family GeomorPM never evaluated, it has
hot 2025-2026 prior art whose best model is closed (FathomDEM, ERL 2025;
GrounDiff, arXiv 2511.10391), so an open DEM-MAE-based version is genuinely
differentiated, and labels are free, global, and per-pixel.

Why it is third: it is the most expensive (real decoder training plus LiDAR
evaluation wrangling, realistically 25 to 40 4090-hours done properly, which
does not fit alongside Tasks 1 and 2 in < $50); a DEM-only encoder will honestly
trail FathomDEM's multi-predictor setup in dense forest (fine to report, but
temper expectations); and FABDEM's CC BY-NC-SA license means any released
weights fine-tuned on it inherit NC-SA, which needs a clear README note kept
separate from the Apache-clean core release. Decision rule: only start it if
Tasks 1 and 2 complete under ~35 hours combined.

### 4.2 Rejected

- **Super-resolution:** dropped per project decision; also needs the GSDPE
  multi-resolution machinery (plan Phase 4) that pretraining did not include.
- **Landslide detection / susceptibility:** landslide scars at 10 m are an
  optical signature; a pre-event 30 m DEM carries weak detection signal.
  Landslide4Sense is imagery-centric (12 of 14 bands are Sentinel-2), only its
  train split has public labels, and its DEM/slope bands are ALOS-derived at a
  different source and grid than GLO-30. Susceptibility mapping has no standard
  benchmark and endemically leaky evaluations. If a cheap gesture at this
  literature is wanted later: a one-day DEM-only ablation row on L4S bands 13-14,
  clearly framed as an ablation, not a task.
- **Karst sinkhole detection:** sinkholes are 10 to 100 m features mapped from
  1 to 1.5 m LiDAR; at GLO-30 they are sub-pixel to a few pixels. Fatal
  resolution mismatch.
- **Avalanche terrain (ATES):** DEM-appropriate signal, but the ML-ATES (NHESS
  2025) labels are unreleased and harvesting Norwegian ATES maps is a project in
  itself. Good future-work line.
- **Hydrography / stream extraction:** labels exist (NHDPlus HR) but flow routing
  is largely deterministic from the DEM (circularity without geomorphons'
  cleanliness) and NHD streams were mapped from higher-resolution sources, so
  30 m registration noise pollutes the labels.
- **Glaciers / rock glaciers, soil, permafrost, canopy height:** either the
  signal is primarily spectral, the labels are themselves coarse model outputs,
  or (canopy height) the task collapses into Task 3's residual.

## 5. Budget plan

| Item | 4090-hours | ~$ |
|---|---|---|
| Task 1: void filling (zero-shot, fine-tune, UNet + ImageNet + spline baselines, Diff-DEM cross-eval) | 10-20 | 3-7 |
| Task 2: landform probes (both label sets, label fractions, UperNet + linear, UNet baseline, one full fine-tune) | 15-25 | 5-9 |
| TerraMind comparison (feature extraction, probes, derivative regression, kNN, CKA) | 3-6 | 1-2 |
| Slack (reruns, spot reclaims, debugging) | ~15 | ~5 |
| **Core total** | **~45-65** | **~15-22** |
| Optional Task 3: bare-earth correction | 25-40 | 9-14 |

Core plan lands well under $50 even with slack; Task 3 fits only if Tasks 1 and 2
come in near the low end.

## 6. Deliverables checklist

1. Results tables: per-task, every encoder and baseline, geographic-holdout only.
2. Money plots: mIoU vs. label fraction (Task 2); RMSE vs. void size with
   hillshade panels (Task 1); probe-accuracy bar chart DEMAE vs. TerraMind with
   parameter counts annotated (comparison).
3. A `benchmark/` module in-repo mirroring the pipeline's stage-per-module style,
   with committed split artifacts and mask lists so every number reproduces.
4. Writeup sections: honest PANGAEA-style framing, the four fairness caveats of
   §3.5 stated explicitly, licenses noted (Apache-2.0 TerraMind, CC-BY Iwahashi,
   CC BY-NC-SA FABDEM if Task 3 runs).

## 7. Sources

Prior art and positioning: GeomorPM (doi:10.1080/13658816.2024.2414409) ·
Self-Supervised Masked DEM Encoding (arXiv 2309.03367) · EvaMAE (SIGSPATIAL 2025,
doi:10.1145/3748636.3762739) · TerraMind (ICCV 2025, arXiv 2504.11171) ·
TerraMesh (arXiv 2504.11172) · PANGAEA (arXiv 2412.04204) · FathomDEM (ERL 2025,
doi:10.1088/1748-9326/ada972) · GrounDiff (arXiv 2511.10391) · TerraMind-vs-THOR
(arXiv 2607.18504).

Task 1 data: Diff-DEM (github.com/kylelo/Diff-DEM; IEEE GRSL 2024,
ieeexplore.ieee.org/document/10535979) · SRTMGL1N `.num` class raster
(earthdata.nasa.gov/data/catalog/lpcloud-srtmgl1n-003) · SRTM Collection User
Guide V3, Table 2 (lpdaac.usgs.gov/documents/179/SRTM_User_Guide_V3.pdf) · SRTM
v2.1 non-void-filled, 3 arc-second, degraded fallback only
(archive.org/details/srtm3v2.1; per-tile via the Wayback capture of
dds.cr.usgs.gov/srtm/version2_1/) · NASADEM_SHHP, a different void population
(earthdata.nasa.gov/data/catalog/lpcloud-nasadem-shhp-001).

Task 2 labels: geomorphons (Jasiewicz & Stepinski 2013, Geomorphology; GRASS
r.geomorphon / WhiteboxTools) · Iwahashi & Pike 2018 terrain classification
(gisstar.gsi.go.jp/terrain2018, CC-BY 4.0; PEPS doi:10.1186/s40645-017-0157-2) ·
Geomorpho90m (Sci Data 2020, doi:10.1038/s41597-020-0479-6).

Task 3 data: FABDEM v1-2
(data.bris.ac.uk/data/dataset/s5hqmjcdj8yo2ibzi9b4ew3sn, CC BY-NC-SA 4.0) ·
USGS 3DEP via OpenTopography · AHN4 (Netherlands) · hoydedata.no (Norway).

TerraMind tokenizer specifics: model card
(huggingface.co/ibm-esa-geospatial/TerraMind-1.0-Tokenizer-DEM, Apache-2.0) ·
IBM/terramind repo and tokenizer reconstruction notebook · terratorch v1.2.10
source (`tokenizer_register.py`, `vqvae.py`, `quantize_finite_scalar.py`,
`terramind_register.py`); architecture facts in §3.4 verified by instantiating
those classes.

Comparison methodology: MAE (arXiv 2111.06377) · DINOv2 kNN protocol
(arXiv 2304.07193) · CKA (Kornblith et al., arXiv 1905.00414) ·
mutual-nearest-neighbor alignment (arXiv 2405.07987) · GEO-Bench
(arXiv 2306.03831).
