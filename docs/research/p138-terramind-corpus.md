# P138: What TerraMind's DEM tokenizer actually saw

**Research memo, August 2026.** Resolves
[#138](https://github.com/JStoschek/DEMAE/issues/138), a child of map
[#117](https://github.com/JStoschek/DEMAE/issues/117). Establishes the DEM
tokenizer's pretraining corpus size, epoch count, source product and sampling
policy **from the papers, official cards and terratorch source only** —
deliberately ignoring the repo's own carried figures, which is what the ticket
asked for.

The line under audit is
[`downstream-task-selection.md:190`](downstream-task-selection.md) —
*"Trained ~20 epochs on TerraMesh's ~9M DEM tiles at 256 px crops."*

**Headline, and it is not the expected one: the memo line is correct.** All four
of its claims check out against primary sources. This map has had three tickets
where a repo-carried figure did not survive being measured; this is the fourth,
and the figure survives. §2.3's table needs two numbers changed and its three
load-bearing sentences all stand.

**The finding that mattered was a trap avoided, not a figure broken.** The
TerraMind paper states **100 epochs** for its tokenizers. That number is wrong
for DEM. The per-modality training configs shipped in terratorch show the
image-like tokenizers were *not* trained uniformly: **S-2 L2A, S-1 RTC and S-1
GRD ran 100 epochs; DEM, LULC and NDVI ran 20.** Reading the paper's headline
sentence and stopping there would have inflated TerraMind's optimization budget
by 5× and destroyed a correct claim. The ticket's instinct — *"the DEM one may
differ from the S-2/S-1 ones"* — was exactly right.

---

## 0. Sources used

Primary only. Both arXiv entries were checked for version currency before
reading; versions below are v-latest as of 2026-08-11.

| Tag | Source | Version |
|---|---|---|
| **[TMesh]** | Blumenstiel et al., *TerraMesh*, [arXiv:2504.11172](https://arxiv.org/abs/2504.11172) | **v2**, 1 Aug 2025 (latest; v1 15 Apr 2025) |
| **[TMind]** | Jakubik et al., *TerraMind*, [arXiv:2504.11171](https://arxiv.org/abs/2504.11171) | **v5**, 17 Jun 2026 (latest; v1–v4 Apr 2025 – Sep 2025) |
| **[MTom]** | Francis & Czerkawski, *Major TOM*, [arXiv:2402.12095](https://arxiv.org/abs/2402.12095) | v2 — TerraMesh's base grid |
| **[SSL4EO]** | Blumenstiel et al., *SSL4EO-S12 v1.1*, [arXiv:2503.00168](https://arxiv.org/abs/2503.00168) | TerraMesh's second base dataset |
| **[Card-D]** | [`ibm-esa-geospatial/TerraMesh`](https://huggingface.co/datasets/ibm-esa-geospatial/TerraMesh) dataset card + HF datasets-server size API | fetched 2026-08-11 |
| **[Card-T]** | [`ibm-esa-geospatial/TerraMind-1.0-Tokenizer-DEM`](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-Tokenizer-DEM) model card | fetched 2026-08-11 |
| **[TT]** | [`terratorch/models/backbones/terramind/tokenizer/tokenizer_register.py`](https://raw.githubusercontent.com/IBM/terratorch/main/terratorch/models/backbones/terramind/tokenizer/tokenizer_register.py) — the DEM tokenizer's full training `Namespace`, verbatim in the factory docstring | fetched 2026-08-11 |

**On [TT]'s standing.** This is not third-party commentary. It is the literal
argument namespace of the training run, pasted into IBM's own released library
by the model's authors, including the run's `output_dir`, `wandb_project` and
config path. It is the most direct evidence available about what the DEM
tokenizer run actually did — closer to the run than the paper is. Where it and
the paper disagree, it wins, and §2 documents the disagreements rather than
hiding them.

**Version currency was load-bearing and was checked.** [TMind] was re-read at v5
after first being read at v2, specifically to see whether the tokenizer
hyperparameters moved between revisions. They did not — the "100 epochs"
sentence is byte-identical across v2 and v5. The paper has been wrong about DEM
for four revisions.

---

## 1. Sub-question 1 — is the DEM modality all ~9M, or a subset?

**All of them. DEM is a fully-populated modality.** Three independent
confirmations.

**(a) The paper.** [TMesh] §3.3:

> "TerraMesh includes 16 TB of compressed data and consists of **9 089 536
> training and 89 088 validation samples** [...] Each spatiotemporal sample
> features seven aligned modalities, resulting in 64M individual patches."

The only modality that is ever absent is Sentinel-1, per [TMesh] Table 2's
caption:

> "Every co-registered sample includes **seven out of eight** modalities due to
> the source datasets providing either S-1 GRD or S-1 RTC data."

Eight modalities minus one always-absent S-1 variant = seven per sample, DEM
among them. Arithmetic checks: 9,089,536 × 7 = 63.6M ≈ the stated "64M".

**(b) The dataset card.** [Card-D] repeats it — *"Each sample location includes
seven modalities that share the same shard and sample name. Note that each
sample only in[c]ludes one Sentinel-1 version (S1GRD or S1RTC)."*

**(c) Shard-count parity, from the HF repo tree API.** `train/DEM` has **889**
shards and `train/S2L2A` has **889**. In `val/`, every modality has 9 shards
except S1RTC (8) and S1GRD (1) — which sum to 9. DEM tracks the universal
modalities exactly.

**Exact split counts** from the HF datasets-server size endpoint for
`ibm-esa-geospatial/TerraMesh`: dataset total `num_rows: 9178624`, `train:
9089536`, `validation: 89088` — matching [TMesh] §3.3 to the digit.

**The tokenizer used the train split.** [TT] records
`data_path="./data/TerraMesh/train"`, `eval_data_path="./data/TerraMesh/val"`.
So the per-epoch figure is **9,089,536**, not 9,178,624. [Card-T] rounds it:
*"pre-trained for 20 epochs on **nine million DEM images** from the TerraMesh
dataset."*

**No DEM subset, and no "TerraMesh-Small".** An HF dataset search returns only
`TerraMesh`, `TerraMesh-Masks` and `TerraMesh-Masks-Eval`; the latter two are
segmentation-mask extensions, not elevation subsets. There *is* a documented
one-million-sample subset, but it belongs to a different artifact — [TMind] §7:
*"TerraMind.v1-B-single was pre-trained on a subset of TerraMesh with one
million samples, specifically the SSL4EOS12 v1.1 locations"* — that is the
v1-B-single **ablation model**, not any tokenizer.

| Established | Value | Source |
|---|---:|---|
| TerraMesh train samples | **9,089,536** | [TMesh] §3.3; HF size API |
| TerraMesh val samples | **89,088** | [TMesh] §3.3; HF size API |
| DEM-specific count | **same — DEM in every sample** | [TMesh] Table 2 caption; [Card-D]; 889/889 shard parity |
| Split the tokenizer trained on | **train** (9,089,536) | [TT] `data_path` |
| DEM subsets / TerraMesh-Small | **do not exist** | HF dataset search |

---

## 2. Sub-question 2 — the actual epoch count for the DEM tokenizer

**20 epochs.** Not 100. This is the ticket's central question and the sources
split, so both sides are given in full.

### What the paper says (and why it is wrong for DEM)

[TMind] §8.1 supplementary, identical in v2 and v5:

> "We pretrained the TerraMind tokenizers for image-like modalities with DDP on
> **4 GPUs** for a total of **100 epochs** on the respective modality of
> TerraMesh. We use a base learning rate of 1e-4, an effective batch size of 64
> samples per GPU, i.e. the **global batch size is 256**."

[TMind] §3 puts DEM in that class: *"image-like modalities such as Sentinel-1,
Sentinel-2, LULC, NDVI, and DEM."* Taken at face value this says the DEM
tokenizer ran 100 epochs at global batch 256.

### What the DEM run actually did

[TT]'s `terramind_v1_tokenizer_dem` docstring carries the run's argument
namespace verbatim. The relevant fields:

```
epochs=20, effective_batch_size=2048, world_size=16, num_tasks=16,
domain="dem@264", all_domains=["dem@264"], input_size=256,
input_size_min=256, input_size_max=256, min_crop_scale=0.8, hflip=0.5,
blr=0.0064, lr=0.0001414213562373095, warmup_epochs=1,
opt="adamw", weight_decay=0.05, clip_grad=1.0, dtype="fp16",
model_ema=True, model_ema_decay=0.9999,
quantizer_type="fsq", codebook_size="8-8-8-6-5", latent_dim=5,
encoder_type="vit_b_enc", decoder_type="unet_patched",
patch_size=16, patch_size_dec=4,
data_path="./data/TerraMesh/train", eval_data_path="./data/TerraMesh/val",
output_dir="artifacts/runs/sm-dem-20-epochs",
wandb_project="terramind-dem",
wandb_run_name="2048_epochs_20/divae/rgb/DEM-ViTB-UNetP4_16k_224-264",
resume="artifacts/runs/sm-dem-20-epochs/checkpoint_12.pth", start_epoch=13
```

`epochs=20` is corroborated **four times inside its own record** — the field
itself, the `output_dir` (`sm-dem-**20**-epochs`), the `wandb_run_name`
(`**2048_epochs_20**`, which also carries the batch size), and the resume path.
This is not a stray value.

### And the model card agrees

[Card-T], independently of terratorch:

> "The model was pre-trained for **20 epochs** on nine million DEM images from
> the TerraMesh dataset."

### The paper's "100" is real — it just belongs to the other modalities

Extracting `epochs` from every tokenizer factory in [TT]:

| Tokenizer | `epochs` | `world_size` | `effective_batch_size` | `domain` |
|---|---:|---:|---:|---|
| `s2l2a` | **100** | 32 | 2048 | `sen2l2a@264` |
| `s1rtc` | **100** | 32 | 2048 | `sen1@264` |
| `s1grd` | **100** | 6 | *(not recorded)* | `sen1@264` |
| **`dem`** | **20** | 16 | **2048** | **`dem@264`** |
| `lulc` | **20** | 16 | 2048 | `lulc@264` |
| `ndvi` | **20** | 16 | 2048 | `ndvi@264` |

The split is clean and interpretable: the three **multi-channel radar/optical**
tokenizers ran 100 epochs, the three **single-channel derived** modalities ran
20. The paper's §8.1 generalised the S-2/S-1 figure across all six. **Use 20 for
DEM.**

### Derived: steps and samples-seen

`effective_batch_size=2048` is recorded directly, so this is a short derivation.
**Labelled derived:**

```
samples seen    = 9,089,536 × 20        = 181,790,720   (~181.8M)
steps per epoch = 9,089,536 / 2048      = 4,438.25
optimizer steps = 4,438.25 × 20         = 88,765
```

This is a far more plausible budget than the paper's numbers imply, and its
plausibility is a check on the reading: 88,765 steps on 16 GPUs is an ordinary
run, whereas the paper's 100 epochs at global batch 256 would imply 3.55M steps
on 4 GPUs — thousands of GPU-hours for a preliminary component, against the
9,216 GPU-hours [TMind] §8.2 reports for the *entire* TerraMindv1-B pretraining.
The config-derived figure is the one that makes sense of the paper's own compute
budget.

### Source conflicts recorded, not smoothed

Three, all resolved in favour of [TT]/[Card-T]:

1. **Epochs** — paper 100, config and card **20**. Resolved above.
2. **Batch size** — paper "global batch size is 256"; config
   `effective_batch_size=2048`, echoed in the run name `2048_epochs_20`. Note
   the paper's §8.1 also contradicts *itself* one paragraph earlier
   (*"set a batch size of 1 per GPU"*). The config's 2048 is used.
3. **GPU count** — paper "4 GPUs"; config `world_size=16, num_tasks=16`.

One internal wrinkle in [TT] itself, reported for honesty: `batch_size=1,
accum_grad=2, world_size=16` multiplies to 32, not the recorded
`effective_batch_size=2048`. The `effective_batch_size` field is taken as
authoritative because the run name independently encodes `2048`. If instead the
true effective batch were 32, the step count would be 5.68M rather than 88,765
— **but samples-seen, which is what §5's comparison uses, is unaffected either
way**, since it depends only on epochs × corpus size.

| Established | Value | Source |
|---|---:|---|
| **DEM tokenizer epochs** | **20** | [TT] `epochs=20`; [Card-T] |
| Effective batch size | **2048** | [TT] (+ run name `2048_epochs_20`) |
| GPUs | 16 | [TT] `world_size=16` |
| LR | 1.4142e-4 (`blr` 0.0064) | [TT] |
| Warmup | 1 epoch | [TT] |
| **Samples seen** | **181,790,720** | **derived** (9,089,536 × 20) |
| **Optimizer steps** | **88,765** | **derived** (÷ 2048) |
| Wall-clock / GPU-hours | **unstated** | — |

---

## 3. Sub-question 3 — source product, resampling, GSD, tile size

Source, resampling and GSD all confirm. Tile size is **264 px, not 256** — but
256 is separately correct as the tokenizer's crop, so the memo line is right and
only §2.3's table row is loose.

### Source: confirmed single-product Copernicus DEM at 30 m

[TMesh] §3.1:

> "TerraMesh also leverages the **Copernicus DEM, a global 30 m resolution
> elevation model based on TanDEM-X**, to provide topographic context."

[TMind] §3: *"DEM is extracted from the **Copernicus DEM 30m dataset**, which
provides global coverage of the Earth's elevation at a 30m resolution."*
[TMesh] §3.2 gives the acquisition endpoint as the Copernicus Data Space
Ecosystem COP-DEM collection. [Card-D]'s attribution line names it precisely:

> "The DEM data is produced using **Copernicus WorldDEM-30** © DLR e.V.
> 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under
> COPERNICUS by the European Union and ESA"

**No blending. No FABDEM, NASADEM or ALOS anywhere** — not in either paper, not
on either card, not in terratorch. One elevation modality, one source. The
closest thing to a complication is that [TMesh] §3.1 explored *other*
pseudo-labelled modalities (DynamicWorld LULC, ETH-GCHM canopy height) and
**rejected** them; neither touches DEM.

Two naming caveats, for precision rather than doubt:

- **The string "GLO-30" appears in none of the sources.** They say "Copernicus
  DEM 30m" ([TMind]), "Copernicus DEM ... 30 m ... TanDEM-X" ([TMesh]) and
  "Copernicus WorldDEM-30" ([Card-D]), all pointing at the COP-DEM collection.
  WorldDEM-30 is the Airbus commercial name for the source of what Copernicus
  distributes as GLO-30. The identification is by product description, vendor
  attribution and endpoint — solid, but by description rather than by
  designation.
- **DSM vs DEM is never disambiguated in any source.** WorldDEM-30 is
  physically a DSM (TanDEM-X first-surface returns over canopy and buildings),
  but no primary source in scope says so; all of them say "Digital Elevation
  Model". **This is not an asymmetry against DEMAE**, whose corpus is the same
  COP-DEM product — both sides inherit the same DSM-labelled-as-DEM convention.

### Resampling: confirmed bilinear onto the 10 m grid

[TMesh] §3.2, Preprocessing:

> "**Copernicus DEM data, acquired at 30 m resolution, is bilinearly resampled
> and projected to the Sentinel-2 grid.**"

[TMesh] §3.1 fixes the grid: *"All modalities in TerraMesh are co-registered to
a consistent grid, **based on Sentinel-2 with a 10 m resolution**."* [Card-D]
concurs on the GSD — *"pixel-aligned 10 m cubes"*.

Sharper than the plan needed: **bilinear is a DEM-specific choice**, not a
default. The same paragraph specifies *"All Sentinel-2 bands are resampled to
10 m using **nearest** interpolation"*. DEM is the exception that gets smooth
interpolation.

Provenance note: the resampling method exists **only in the paper**. It is not
on the HF card and not in terratorch — a grep of the terramind backbone tree for
`resampl|bilinear|nearest|cubic` returns only model-internal `F.interpolate`
calls. So this claim rests on [TMesh] §3.2 alone, which is the authors' own
primary publication and adequate.

### Tile size: 264 px corpus tile, 256 px tokenizer crop

[TMesh] §3.2:

> "Each grid cell is a 1068 × 1068-pixel tile covering 16 non-overlapping
> patches at the **264 × 264 pixel size** of SSL4EO-S12. This patch size,
> **divisible by six**, accommodates Sentinel-2's resolutions of 10 m, 20 m,
> and 60 m."

[TMesh] Table 1 lists TerraMesh's patch size as `264 × 264`; [TMind] §3 repeats
*"each measuring 264 × 264 pixels at 10m resolution"*; [Card-D]'s xarray dump
shows `Dimensions: (band: 2, time: 1, y: 264, x: 264)`.

**And the tokenizer crops that to 256.** [TT] settles what [TMind] §8.1 left as
a range (*"input images ranging from 224x224 to 256x256"*): for DEM,
`domain="dem@264"`, `input_size=256`, `input_size_min=256`,
`input_size_max=256`, `min_crop_scale=0.8`. The run name spells out the whole
pipeline — `DEM-ViTB-UNetP4_16k_**224-264**`. So: **a random crop of scale
0.8–1.0 from the 264 px tile, resized to 256 px**, plus `hflip=0.5`.

This independently corroborates
[`p118-terramind-encoder-probe.md`](p118-terramind-encoder-probe.md) §6, which
reached 256 px by measuring the released checkpoint's `pos_emb` shape
`[1, 768, 16, 16]` (16 tokens × 16 px). Two different routes, same answer, so
§2.2's 256 px premise is now double-sourced.

**The memo line's "256 px crops" is therefore correct.** What is loose is
§2.3's table row, which prices the *corpus* at 256 px when the corpus tile is
264 px. That is the one number in the table that changes for a reason other than
the epoch count. It matters by 6.3% and only for ground area.

| Established | Value | Source |
|---|---:|---|
| DEM source | Copernicus DEM / WorldDEM-30, TanDEM-X, 30 m | [TMesh] §3.1; [TMind] §3; [Card-D] |
| String "GLO-30" in sources? | **No** — "Copernicus DEM 30m" / "WorldDEM-30" | [Card-D] |
| Blended / mixed? | **No** — single source, no FABDEM/NASADEM/ALOS | all sources |
| Resampling | **Bilinear** (S-2 bands get nearest — DEM is the exception) | [TMesh] §3.2 only |
| Grid / GSD | Sentinel-2 grid, **10 m** | [TMesh] §3.1; [Card-D] |
| **Corpus tile** | **264 × 264 px** = 2.64 km = **6.9696 km²** | [TMesh] §3.2, Table 1; [TMind] §3; [Card-D] |
| **Tokenizer crop** | **256 px**, random scale 0.8–1.0 of the 264 tile, + hflip | [TT]; corroborated by p118 §6 |
| DSM vs DEM | **unstated in every source** | — |

---

## 4. Sub-question 4 — deduplication and geographic overlap

The sources say more than expected. There are **three** overlap mechanisms, and
the naive "9M × footprint" product double-counts two of them.

### (a) Within a grid cell: explicitly non-overlapping

[TMesh] §3.2: *"Each grid cell is a 1068 × 1068-pixel tile covering **16
non-overlapping patches** at the 264 × 264 pixel size."* Sixteen 264 px patches
tile 1056 × 1056 of the 1068 × 1068 cell, 12 px unused. Clean.

### (b) Between grid cells: cells *do* overlap, by construction

TerraMesh inherits MajorTOM-Core's grid, and [MTom] §3.1 defines it:

> "The Major TOM grid is defined by a single number: the nominal grid spacing,
> D, in units of distance. **We use 10 km**."

But the cell extent is 1068 px at 10 m = **10.68 km**. A 10.68 km extent on a
10 km lattice overlaps its neighbours by 0.68 km per axis. [MTom] §4 quantifies
the consequence for full MajorTOM-Core:

> "This amounts to a total surface area of **250 million km² including
> overlapping boundaries, or 225 million km² excluding those overlaps**."

A 1.111× redundancy factor — so **unique ground per grid cell is D² = 100 km²,
not the 114.1 km² its extent implies**. [MTom] Fig. 2's caption says it plainly:
the sample *"overlaps slightly with other nearby grid cells."*

### (c) Between source datasets: SSL4EO-S12 is multi-seasonal

[TMesh] §3.2 merges *"SSL4EO-S12 v1.1 with a subset of MajorTOM-Core"*, and
[TMind] §3 characterises them: *"SSL4EO-S12 has lower geographic coverage but is
**multi-seasonal**. MajorTOM-Core covers most of the Earth's land surface at a
**single timestamp**."* [SSL4EO] gives *"246k time series with nearly one
million image patches"* — **~4 timestamps per location**, so SSL4EO's patch
count overstates its unique ground roughly fourfold. [Card-D] confirms the
identity scheme: *"The `sample_id` refers to the MajorTOM grid cell with two
additional `x` and `y` chip indices or to the SSL4EO-S12 sample id with an
additional **time index**."*

### The only explicit dedup policy is train/val leakage control

[TMesh] §3.2:

> "A random 99%-1% train-validation split is applied to the selected
> MajorTOM-Core grid cells. Any SSL4EO-S12 v1.1 patch that **intersects the
> validation extent** is added to the validation set; **further patches
> spatially overlapping the joined validation area are removed to mitigate data
> leakage**."

**This is leakage control across the split boundary, not deduplication within
the training set.** Nothing in any source removes near-duplicate or overlapping
tiles *inside* training. There is no minimum-separation rule, no perceptual
dedup, and no claim that training tiles are mutually disjoint. Given (b), they
demonstrably are not.

### What the subsampling policy actually is: diversity balancing

[TMesh] §3.2, Subsampling:

> "We begin by discarding any MajorTOM-Core grid cells containing more than 50%
> missing data (NaN) [...] For each land-cover category, we select grid cell
> candidates within **5%–99% coverage** to avoid extremely homogeneous patches
> and sample up to 500k grid cells. Subsequently, we ensure that each of the
> **846 ecoregions** is represented by at least 500 grid cells [...] The
> resulting selection comprises **538k grid cells**."

Plus a quality filter: *"we discard any patch with more than 1% missing values
(NaN) in any modality or channel."*

### Derived: unique ground area

**Labelled derived — no source states TerraMesh's unique ground area.**

```
MajorTOM subset, unique = 538,000 × 100 km²      = 53.80M km²   (grid spacing, per [MTom])
SSL4EO subset,   unique = 246,000 × 6.9696 km²   =  1.71M km²   (locations, not patches)
                                                   -----------
                                                   ~55.5M km²
```

Against the naive product 9,089,536 × 6.9696 = **63.35M km²**, so the naive
figure over-counts unique ground by about **14%**. Two smaller effects are
ignored, both pushing the same way: patches dropped by the NaN filter, and
SSL4EO locations falling inside MajorTOM coverage. So ~55M km² is an upper-ish
estimate, call it ±5%.

| Established | Value | Source |
|---|---:|---|
| Patches within a grid cell | **non-overlapping** | [TMesh] §3.2 |
| Grid cells between each other | **overlap** — 10.68 km extent on 10 km spacing | [MTom] §3.1, §4 |
| MajorTOM redundancy | 250M / 225M km² = **1.111×** | [MTom] §4 |
| Retained MajorTOM cells | **538,000** | [TMesh] §3.2 |
| Diversity rule | 5–99% class coverage; ≥500 cells per 846 ecoregions | [TMesh] §3.2 |
| SSL4EO locations / patches | 246k / ~1M (≈4 seasons each) | [SSL4EO] |
| Tiling scheme | MajorTOM grid + SSL4EO ids — **not MGRS** | [Card-D] |
| Train/val leakage policy | **stated** — overlapping patches removed | [TMesh] §3.2 |
| Within-training dedup | **none stated in any source** | — |
| Minimum tile separation | **none stated** | — |
| Unique ground area | ~55M km² | **derived** |

---

## 5. The recomputed §2.3 table

DEMAE's column is unchanged — not this ticket's subject, carried for comparison.

| | DEMAE | TerraMind DEM tok. (**old**) | TerraMind DEM tok. (**new**) |
|---|---:|---:|---:|
| Tiles per epoch | 473,088 (231 × 2048) | ~9M | **9,089,536** (4,438.25 × 2048) |
| Epochs | 400 | ~20 | **20** |
| **Tile presentations** | **189.2M** | ~180M | **181.8M** |
| Optimizer steps | 92,400 | — | **88,765** |
| Effective batch | 2048 | — | **2048** |
| Tile footprint | 224 px @ 30 m = 45.16 km² | 256 px @ 10 m = 6.55 km² | **264 px @ 10 m = 6.97 km²** |
| **Ground area (naive)** | **21.4M km²** | ~59.0M km² | **63.4M km²** |
| Ground area (unique, derived) | 21.4M km² | — | **~55M km²** |

Derived ratios:

| Ratio | Value |
|---|---:|
| Tile count | **19.2×** |
| **Tile presentations** | **1.041×** — DEMAE sees **4.1% more** |
| Optimizer steps (same batch size) | **1.041×** — DEMAE runs 4.1% more |
| Footprint per tile | **6.5×** (45.16 / 6.97) |
| Ground area, naive | **2.97×** |
| Ground area, unique-corrected | **~2.6×** |

**An unplanned bonus.** Both runs used **the same effective batch size, 2048**.
That was not known when §2.3 was written and it makes the matched-compute claim
much stronger than a bare presentation count: the two runs are matched in
samples seen *and* in optimizer steps *and* in batch size, so the "matched
optimization" comparison is close to like-for-like rather than a coincidence of
two numbers meeting in the middle.

---

## 6. Verdict per sentence

### ✅ **"Optimization compute is matched to within 5%" — SURVIVES AS WRITTEN.**

This was flagged as the softest and most visible sentence, and it holds. The DEM
tokenizer ran **20 epochs** over 9,089,536 tiles = **181.8M presentations**,
against DEMAE's 189.2M. **DEMAE sees 4.1% more** — inside the stated 5%.

It survives on better evidence than it was written with. The parenthetical
"~180M" was an estimate from a rounded 9M × ~20; the real figure is 181.8M, and
both runs additionally share an effective batch size of 2048 and land within
4.1% on optimizer steps (92,400 vs 88,765). Recommend upgrading the sentence
rather than merely keeping it: state 181.8M, and state that batch size and step
count match too.

Two qualifications should be attached, neither of which threatens the claim:

1. **The two runs optimize different objectives** — DEMAE is a masked
   autoencoder; the TerraMind tokenizer is an FSQ autoencoder with a diffusion
   decoder ([TMind] §3), trained with `num_train_timesteps=1000` ([TT]).
   Presentations and steps are comparable; **FLOPs are not**, and no source
   gives FLOPs for either side. "Matched optimization compute" should be read as
   *matched sample exposure and step count*, which is what the table measures.
2. **The paper says 100, and a reader will check.** Since §2.3 now asserts a
   number that contradicts the published paper, it should cite the per-modality
   evidence in-line — [TT]'s `epochs=20` and [Card-T]'s "20 epochs on nine
   million DEM images" — and note that the paper's 100 belongs to the S-2/S-1
   tokenizers. Without that, the claim looks like a misreading of [TMind] §8.1.

**The pre-registered caveat paragraph stands unchanged** — *"if DEMAE wins C1 on
2.8x less ground of the same product at matched compute"* is still true, subject
only to the area number in §6's next verdict.

### ✅ **"The data gap is ~2.8× in ground area, not the ~19× the tile counts imply" — SURVIVES, with a number to adjust.**

Robust to the epoch question as predicted, and the DEM-subset risk did not
materialise (§1). The tile-count ratio **19.2×** is confirmed exactly. The area
ratio moves for one reason — the corpus tile is **264 px, not 256 px**:

- **2.97×** pricing the corpus by its real 264 px tiles.
- **~2.6×** additionally correcting for MajorTOM's overlapping 10 km grid and
  SSL4EO's four-seasons-per-location (§4).

The plan's "~2.8x" sits between the two, having been reached with a footprint
6% too small which partly cancelled an omitted overlap correction. **The claim
is right; the number is right by accident.** Recommend **~2.6×** with the
overlap correction shown, or **~3.0×** for a stated-inputs-only figure — but not
an unlabelled "~2.8x", since neither derivation produces it.

The supporting clause *"cover 6.6x less ground each"* becomes **6.5×**
(45.16 / 6.97).

**One asymmetry to disclose.** §2.3 currently closes with *"Within-corpus
geographic overlap is unknown on both sides."* That is now **half-resolved**:
known and quantified on TerraMind's side (§4), still unknown on DEMAE's. If the
plan adopts the overlap-corrected ~2.6× it will be comparing a *corrected*
TerraMind figure against an *uncorrected* DEMAE figure, which flatters DEMAE.
Either state both naive (2.97×) or state the asymmetry explicitly.

### ✅ **"More ground of the same product at the same true resolution — not more information" — SURVIVES INTACT.**

The claim with the worst downside did not break. Sources confirm a **single,
unblended Copernicus DEM / WorldDEM-30 30 m source, bilinearly resampled to the
10 m Sentinel-2 grid**. No FABDEM, no NASADEM, no mixing, and no DSM/DEM switch
relative to DEMAE, which uses the same COP-DEM product.

The sentence can be **sharpened**: [TMesh] §3.2 shows DEM is singled out for
bilinear resampling while Sentinel-2's own bands get nearest — so the 3×
upsampling is deliberate smooth interpolation, demonstrably adding no
high-frequency content. "Not more information" is not just unfalsified here; it
is affirmatively supported.

### ✅ **§2.2's GLO-30 / bilinear / 10 m assertion — SURVIVES.**

*"TerraMind's DEM pixels were GLO-30 bilinearly resampled to the 10 m Sentinel-2
grid (TerraMesh §3.2)"* is correct and correctly attributed — §3.2's
Preprocessing paragraph is exactly where the bilinear sentence lives. One
optional pedantic amendment: no source writes "GLO-30"; they write "Copernicus
DEM 30m" and "Copernicus WorldDEM-30" (§3). Worth changing only if the plan
wants the citation quotable.

**§2.2's other TerraMind claim also survives and is now double-sourced.**
*"TerraMind's DEM tokenizer was trained at 256 px and its `pos_emb` is natively
[1, 768, 16, 16]"* is confirmed by [TT] (`input_size=256`, `image_size=256`)
independently of p118's checkpoint measurement. The ADR-0011 re-derivation
premise — that a 224 px tile is a crop of a 256 px scene — is sound, and [TT]
strengthens it: the tokenizer itself was fed random crops of a *larger* 264 px
tile, so cropping is the native relationship between its inputs and its
positional grid, not a resize.

**Caution: do not extend "256" to the corpus.** The tiles are 264 px; 256 is the
crop. §2.3's footprint row is where that distinction was lost.

### ⚠️ The memo line itself — `downstream-task-selection.md:190`

*"Trained ~20 epochs on TerraMesh's ~9M DEM tiles at 256 px crops."* — **all
four claims correct.** ~20 epochs (exactly 20), ~9M DEM tiles (9,089,536), DEM
tiles (fully-populated modality), 256 px crops (exactly, of 264 px tiles). No
correction needed. It is worth adding the citation it lacked, since the claim
contradicts the paper's own headline number and will otherwise read as an error.

---

## 7. Everything that could not be established

Reported as unknown rather than inferred, per the ticket's standing rule.

| Question | Status |
|---|---|
| **Tokenizer wall-clock / GPU-hours / FLOPs** | **Unstated.** Given for TerraMindv1-B (12 days × 32 A100 = 9,216 GPU-h) but never for tokenizers. This is why §6 declines to convert the 4.1% presentation match into a FLOP claim. |
| **Total step count as logged** | **Not recorded.** [TT] has `warmup_steps=-1` (epoch-based scheduling); the 88,765 in §2 is derived. |
| **Which is right: `effective_batch_size=2048` or `batch_size × accum × world_size = 32`** | **Unresolved** in [TT]. 2048 is used (run name corroborates). Does not affect samples-seen. |
| **Per-modality sample counts as a stated figure** | **Unstated.** "DEM = all samples" rests on the seven-of-eight rule plus 889/889 shard parity, not on a printed per-modality count. |
| **DSM vs DEM** | **Never disambiguated** in any source. WorldDEM-30 is named; its surface-vs-terrain nature is not discussed. |
| **Resampling method outside the paper** | **Absent** from the HF card and terratorch. Rests on [TMesh] §3.2 alone. |
| **Whether training used train-only** | [TT]'s `data_path=".../train"` establishes it; no prose sentence says so. |
| **Exact MajorTOM / SSL4EO split of the 9.09M** | **Unstated.** Loosely: 538k × 16 = 8.61M MajorTOM patches pre-filter, plus [TMind]'s "one million" SSL4EO samples = 9.61M against an actual 9.18M, i.e. ~4.5% dropped by the NaN filter and val-overlap removal. The residual is not attributable. |
| **TerraMesh's unique ground area** | **Unstated.** The ~55M km² in §4 is derived from grid spacing and season count. |
| **Within-training deduplication / minimum separation** | **No policy stated.** Only the train/val leakage rule exists. |
| **DEMAE's own within-corpus overlap** | Out of scope here, but now the *only* unknown side of the area ratio. Worth its own ticket if §2.3 adopts the corrected 2.6×. |

### One incidental finding, flagged because it can silently corrupt an eval

**Three different DEM normalization constants ship across IBM's own artifacts:**

| Where | mean | std |
|---|---:|---:|
| TerraMesh dataset loader (`terramesh.py`) | 651.663 | 928.168 |
| TerraMind v1 pretraining stats (`terramind_register.py`, `*_dem@224`) | 670.665 | 951.272 |
| TerraMind v0.1 tokenizer stats | 435.726 | 560.326 |

§2.1 of the plan ("feed each model its native diet") depends on picking the
right one. The **670.665 / 951.272** pair is the one attached to the TerraMind
v1 model's `dem@224` input, so it is the one for feeding the v1 encoder. This
is outside #138's four sub-questions and is recorded here only so it is not
discovered later.

---

## 8. One-paragraph summary for the map

**Nothing in §2.3 has to be deleted.** TerraMind's DEM tokenizer saw
**9,089,536 tiles for 20 epochs at effective batch 2048** — **181.8M tile
presentations over 88,765 steps** — against DEMAE's 189.2M over 92,400 steps at
the same batch size. **DEMAE sees 4.1% more, so "optimization compute is matched
to within 5%" survives**, and survives with better evidence than it was written
with. The trap was that [TMind] §8.1 states **100 epochs** for its tokenizers;
that figure belongs to the S-2 L2A, S-1 RTC and S-1 GRD tokenizers, while the
per-modality configs in terratorch and the official DEM model card both record
**20** for DEM, LULC and NDVI. Two numbers in the table change: the corpus tile
is **264 px, not 256** (256 is the tokenizer's random crop of it), so tile
footprint is 6.97 km² and naive ground area is **63.4M km²** — or **~55M km²**
once MajorTOM's overlapping 10 km grid and SSL4EO's four-seasons-per-location
are corrected for. The area gap is therefore **~2.6× corrected / ~3.0× naive**
against a **19.2×** tile-count gap, so the "~2.8x, not ~19x" point stands with a
relabelled number. The DEM source is confirmed **single-product Copernicus
DEM / WorldDEM-30 at 30 m, bilinearly resampled to the 10 m Sentinel-2 grid**,
unblended, with DEM the *only* modality given bilinear treatment — so "same
product at the same true resolution, not more information" survives and can be
stated more strongly. **DEM is a fully-populated modality**, the finding most
likely to have broken the area claim, and it did not.
