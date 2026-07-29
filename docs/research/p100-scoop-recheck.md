# P100 — scoop re-check: is DEMAE still the first open-weights elevation-only encoder?

Ticket [#100](https://github.com/JStoschek/DEMAE/issues/100) · plan §13 (Risks), §14 (Interface to Phase 2), §16 (Key sources) · downstream [#86](https://github.com/JStoschek/DEMAE/issues/86), [#88](https://github.com/JStoschek/DEMAE/issues/88)

> **What this note is.** A fresh, independent search — not a diff against the 2026-07-28 pass. Everything below is either a URL with a retrieval date or an inference explicitly marked as one. Where I could not reach a source, §7 says so by name. All retrievals are **2026-07-29** unless stated otherwise.

## 1. The claim under test

Plan §13's Risks table, row "Scooped by a new DEM FM":

> Scoop check on 2026-07-28 found no open elevation-only pretrained encoder (GeomorPM remains closed; Copernicus-FM/TerraMind use DEM as one of many modalities); re-check before release

DEMAE is an open-weights ViT-B/16 MAE pretrained on 500k single-channel Copernicus GLO-30 tiles at 224 px. The novelty claim resting on §13 is: **first open-weights encoder pretrained on elevation alone.**

The §13 check was one day old, so a brand-new scoop was never the likely outcome. The question worth asking is whether the 2026-07-28 pass was *thorough enough* — whether something that already existed was missed. It was.

## 2. Verdict

**No. The §13 claim does not hold as worded, and it does not fail on a technicality.**

An open-weights, Apache-2.0, **ViT-B/16 encoder pretrained on single-channel DEM alone** has been downloadable from Hugging Face since **2025-02-20**: [`ibm-esa-geospatial/TerraMind-1.0-Tokenizer-DEM`](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-Tokenizer-DEM). Its architecture is `encoder_type="vit_b_enc"`, `patch_size=16`, `n_channels=1`, `image_size=256`, trained 20 epochs on ~9 M Copernicus DEM patches. That is DEMAE's exact architecture and modality at roughly **18× the corpus**. §4.1 has the evidence.

The 2026-07-28 pass got the *backbone* right and the *tokenizer* wrong. TerraMind's ViT backbone genuinely is multimodal — but TerraMind also ships **seven separate per-modality tokenizers**, one of which is DEM-only, and that one is a ViT-B/16 on single-channel elevation. "TerraMind uses DEM as one of many modalities" is true of the thing §13 was looking at and false of the thing sitting next to it.

Three further candidates the 2026-07-28 pass appears to have missed, each weaker but each real:

- **DINO-CV** — DINO self-distillation pretrained on 188,006 unlabelled DEM-derivative tiles; ResNet-50 / WRN-50-2 / ViT-S/16 backbones; weights downloadable (verified, 745 MB), CC BY-NC 4.0 (§4.2).
- **`xandergos/terrain-diffusion-30m`** — MIT-licensed generative diffusion stack trained on **the Copernicus GLO-30 dataset**, i.e. DEMAE's own corpus, released 2025-12-17 (§4.3).
- **`hwh-datascience/divae-tokenizer-ahn-448`** — Apache-2.0 ViT-B/16 DiVAE on 2-channel AHN6 elevation; a third-party TerraMind-methodology clone (§4.4).

**What survives, and it is narrower than §13.** Nothing I found is an encoder pretrained with a **masked-reconstruction objective** on **raw single-channel elevation**, **released and evaluated as a transferable representation backbone**, with probe or transfer numbers to support that use. TerraMind's DEM tokenizer is a *tokenizer*: its documented job is to encode DEM into FSQ codes for the multimodal backbone and decode them back, and IBM publishes no linear-probe, kNN, or transfer result for it as a feature extractor. DEMAE can still be first at the thing it is actually doing. It cannot be first at "an open encoder trained on elevation alone" — that sentence is now false and must be retired.

**Recommendation:** amend §13, narrow the novelty sentence wherever it appears, and add a related-work paragraph naming the four candidates above. #88 needs to change (§6.3). None of this touches the experiment design, the corpus, or the run plan — it is a framing and release-writeup problem, caught before release, which is exactly what the §13 re-check trigger existed to do.

## 3. What I searched, and with what

Stated in enough detail that a later reader can judge coverage and re-run it.

### 3.1 Hugging Face — model ID search (`/api/models?search=`)

This endpoint matches **model-ID substrings only**, not card text, which is why it is nearly useless here and why I did not stop at it. Queries run: `dem`, `elevation`, `terrain`, `topograph`, `bathymetry`, `glo-30`, `copernicus-dem`, `dtm`, `lidar`, `geomorph`. `dem` returned 100 hits dominated by `demucs` and `wav2vec2-…-demo-colab`; `bathymetry` returned **0**; `glo-30` returned 100 hits all of which were `glove`/`global_step`. **Nothing usable came from this endpoint.** Recording it because a searcher who only ran this — which is the obvious first move — would correctly conclude "nothing exists," and that is very likely how the 2026-07-28 pass reached its answer.

### 3.2 Hugging Face — full-text card search (`/api/search/full-text?type=model`)

This is the endpoint that actually works. 37 distinct queries:

> `digital elevation model` · `elevation model pretrained` · `DEM tiles pretraining` · `masked autoencoder DEM` · `elevation foundation model` · `terrain foundation model` · `DEM foundation model` · `Copernicus DEM` · `GLO-30` · `SRTM` · `ALOS AW3D30` · `ALOS World 3D` · `bathymetry` · `bathymetry foundation` · `digital terrain model` · `hillshade` · `geomorphology` · `geomorphometric` · `landform classification pretrained` · `DEM super-resolution` · `DEM void filling` · `topographic self-supervised` · `topographic foundation model` · `elevation encoder ViT` · `single channel DEM` · `heightmap ViT` · `geomorphon` · `slope aspect curvature pretrained` · `masked autoencoder elevation tiles` · `ViT pretrained elevation raster` · `MAE pretrained terrain encoder` · `self-supervised DEM encoder` · `DEM only pretraining` · `elevation backbone timm` · `height map foundation model` · `Copernicus GLO-30 pretrained` · `MERIT DEM` · `GEBCO`

`geomorphometric` returned **0 hits**. `GLO-30` returned **2**, both `xandergos/terrain-diffusion-*`. `geomorphology` returned 12, `geomorphon` 15, `bathymetry` 17 — all small enough to read in full, and none contained a pretrained elevation encoder.

### 3.3 Hugging Face — tag enumeration and author enumeration

Tag filters `dem` (12 models), `elevation` (1), `terrain` (4), `topography` (**0**), `bathymetry` (**0**), `geospatial` (100, read in full). Author listings enumerated completely for `ibm-esa-geospatial` (14 models), `ibm-nasa-geospatial` (15), `made-with-clay` (2), `Major-TOM` (**0 models** — Major TOM publishes datasets, not model weights), `hwh-datascience` (3).

### 3.4 Model registries

- **TorchGeo** — read [`torchgeo/models/__init__.py`](https://github.com/microsoft/torchgeo/blob/main/torchgeo/models/__init__.py) in full. The registry is Aurora, Copernicus-FM, CROMA, DOFA, EarthLoc, OlmoEarth, Panopticon, Presto, ResNet-18/50/152, Scale-MAE, Swin, Tessera, TileNet, U-Net, ViT (incl. DINOv2). **No DEM or elevation entry of any kind.**
- **TerraTorch** — read [`tokenizer_register.py`](https://github.com/IBM/terratorch/blob/main/terratorch/models/backbones/terramind/tokenizer/tokenizer_register.py). This is where the DEM tokenizer surfaced; see §4.1.
- **Prithvi / IBM-NASA** — full author listing; Prithvi-EO 1.0/2.0 and Prithvi-WxC only. No DEM-only variant.
- **Clay** — `made-with-clay/Clay` ships one file, `v1.5/clay-v1.5.ckpt`, Apache-2.0. No DEM-only variant.

### 3.5 Literature

arXiv API (`export.arxiv.org/api/query`), sorted by submission date descending, 14 structured queries:

> `all:"digital elevation model" AND all:"self-supervised"` (6 hits) · `all:"digital elevation model" AND all:"foundation model"` (3) · `all:"DEM" AND all:"masked autoencoder"` (**1** — only arXiv 2309.03367, already in plan §16) · `abs:"terrain" AND abs:"foundation model"` (25) · `abs:"elevation" AND abs:"pretrained encoder"` (1) · `all:"geomorphometric" AND all:"deep learning"` (**0**) · `abs:"bathymetry" AND abs:"foundation model"` (**0**) · `abs:"topographic" AND abs:"self-supervised pretraining"` (**0**) · `cat:cs.CV AND abs:"elevation"` · `cat:eess.IV AND abs:"DEM"` · `cat:physics.geo-ph AND abs:"deep learning" AND abs:"terrain"` · `abs:"DEM" AND abs:"pretraining"` · `abs:"digital elevation" AND abs:"transformer"` · `abs:"terrain" AND abs:"masked"`

The date-sorted sweeps reached **2026-07-28** (`cat:cs.CV AND abs:"elevation"`, top hit 2607.25324) and **2026-07-27** (`abs:"terrain" AND abs:"masked"`, top hit 2607.25148), so the window since the last check is covered. Nothing new landed in it.

Also queried Semantic Scholar's Graph API by DOI for GeomorPM and EvaMAE, and ran GitHub repo search (`gh search repos`) for `GeomorPM`, `VQMAE-DEM`, `DEM pretrained model` — all **0 results** — plus `gh search code GeomorPM`, which returned exactly one hit, a citation in an unrelated awesome-list README.

### 3.6 What I did **not** cover

Named here rather than buried in §7: I did **not** search ISPRS Journal, IJGIS, Remote Sensing of Environment, or Computers & Geosciences directly. Those are paywalled and my fetches to `tandfonline.com` and `dl.acm.org` were both refused with HTTP 403. I reached that literature only indirectly, via arXiv preprints, Semantic Scholar metadata, and GitHub. A closed-access DEM foundation-model paper published in one of those journals **without** an arXiv preprint, a GitHub repo, or a Hugging Face release would not have been found — though note that such a release, by construction, would have no downloadable weights and so would not be a scoop under this note's scope.

## 4. The candidates, in order of force

### 4.1 TerraMind 1.0 DEM Tokenizer — this is the one that breaks the claim

| | |
|---|---|
| Repo | [`ibm-esa-geospatial/TerraMind-1.0-Tokenizer-DEM`](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-Tokenizer-DEM) |
| Created | **2025-02-20**, last modified 2025-11-03 |
| Licence | **`apache-2.0`** (card front-matter) |
| Weights | `TerraMind_Tokenizer_DEM.pt`, **1,145,511,376 B**, HTTP 200 verified 2026-07-29 |
| Downloads | 381 |
| Publisher | IBM + ESA + Forschungszentrum Jülich |

The model card states, in full on the training question:

> This tokenizer encodes and decodes Digital Elevation Model (DEM) maps for the TerraMind model. … The model was pre-trained for 20 epochs on nine million DEM images from the TerraMesh dataset.

The architecture is not left to inference. TerraTorch's registration function carries the complete training `Namespace` in its docstring:

```python
def terramind_v1_tokenizer_dem(**kwargs):
    """
    model args: Namespace(patch_size=16, input_size_min=256, input_size_max=256,
    encoder_type="vit_b_enc", decoder_type="unet_patched", quantizer_type="fsq",
    codebook_size="8-8-8-6-5", latent_dim=5, epochs=20, domain="dem@264",
    all_domains=["dem@264"], effective_batch_size=2048, world_size=16,
    wandb_project="terramind-dem", ...)
    """
    tokenizer = build_vqvae(variant="terramind_v1_tokenizer_dem", image_size=256,
                            n_channels=1, encoder_type="vit_b_enc", ...)
```

— [`terratorch/models/backbones/terramind/tokenizer/tokenizer_register.py`](https://github.com/IBM/terratorch/blob/main/terratorch/models/backbones/terramind/tokenizer/tokenizer_register.py), lines ~236–258

Read that against DEMAE's spec:

| | TerraMind DEM tokenizer | DEMAE |
|---|---|---|
| Encoder | ViT-B, patch 16 | ViT-B/16 |
| Input channels | **1** | **1** |
| Input size | 256 | 224 |
| Pretraining modality | DEM only (`all_domains=["dem@264"]`) | DEM only |
| Corpus | ~9 M patches (TerraMesh) | 500 k tiles |
| Objective | FSQ-VAE, diffusion (UNet) decoder | MAE, masked reconstruction |
| Licence | Apache-2.0 | (planned open) |

**Training data.** TerraMesh ([`ibm-esa-geospatial/TerraMesh`](https://huggingface.co/datasets/ibm-esa-geospatial/TerraMesh), CC-BY-SA-4.0, arXiv [2504.11172](https://arxiv.org/abs/2504.11172)) is a multimodal corpus of >9 M co-registered patches at 10 m, 9.09 M train / 89.1 k val, whose DEM modality the dataset card names as **Copernicus DEM**. Every sample carries all modalities, so the DEM count equals the sample count.

**Is the encoder actually extractable?** Yes — a plain `encoder.*` subtree, and it is a timm-style ViT. I confirmed this by range-reading the safetensors header of the architecturally identical AHN clone (§4.4) without downloading the payload:

```
total tensors: 601
prefixes: {'decoder': 446, 'encoder': 153, 'quant_proj': 2}
sample encoder keys: encoder.blocks.0.attn.qkv.weight, encoder.blocks.0.mlp.fc1.weight,
                     encoder.blocks.0.norm1.weight, ...
encoder params: 90.8M
```

90.8 M is ViT-B-shaped. *(Inference, marked: I verified this on the AHN clone, whose card says it follows "the TerraMind paper methodology (Section 8.1)" and whose constructor arguments match TerraTorch's `terramind_v1_tokenizer_dem` exactly apart from `n_channels` and `image_size`. I did **not** open IBM's own 1.15 GB `.pt`, which is not safetensors and cannot be header-read. The inference is strong but it is an inference.)*

**What is genuinely different about DEMAE, and it is not nothing.**

1. **Objective.** FSQ-VAE tokenization reconstructs the whole image through a quantization bottleneck; MAE reconstructs masked patches from visible context. These produce different representations, and the MIM-vs-VAE distinction is the substance of DEMAE's method.
2. **Framing and evidence.** IBM publishes **no** probe, kNN, linear-eval, or downstream-transfer number for this tokenizer as a feature extractor. Its card documents `encode` → tokens and `decode_tokens` → reconstruction, and nothing else. Nobody has shown it is a good representation encoder; nobody has claimed it. DEMAE proposes to answer that question for its own encoder, which is a contribution the tokenizer does not make.
3. **Sampling.** TerraMesh's DEM is co-registered with Sentinel-1/2 and resampled to the 10 m optical grid — the corpus is sampled where Sentinel imagery is, not stratified on terrain. DEMAE's corpus is terrain-stratified at native 30 m (§9, ADR-0005). That is a defensible and *measurable* difference, and it is the one DEMAE should lead with.

**But be honest about the direction of the pressure.** 9 M > 500 k by 18×, it is Apache-2.0, it is on the hub, and it is the same architecture on the same modality from IBM/ESA. A reviewer will find it. The framing has to name it first, not be caught by it.

### 4.2 DINO-CV — self-supervised, DEM-derivative, weights genuinely downloadable

Paper: "Mapping Hidden Heritage: Self-supervised Pre-training on High-Resolution LiDAR DEM Derivatives for Archaeological Stone Wall Detection", Huang, Islam, Armstrong, Bell, Khoshelham, Tomko. arXiv [2510.17644](https://arxiv.org/abs/2510.17644) (submitted 2025-10-20, latest version 2026-01-17); journal version at [ScienceDirect S2666017226000106](https://www.sciencedirect.com/science/article/pii/S2666017226000106). Code and weights: [`MLinArcheaomatics/BudjBimStoneWall-DINO-CV`](https://github.com/MLinArcheaomatics/BudjBimStoneWall-DINO-CV) (repo created 2024-09-11, last push 2026-01-23).

**Pretraining.** The README's own dataset table: `BudjBimLandscape`, **188,006 tiles**, 301 km², 10 cm DEM resolution, unlabelled, views = MHS (multi-directional hillshade) and VAT (Visualization for Archaeological Topography), 400×400 px resized to 256×256. The method is DINO-style cross-view self-distillation between the two DEM-derived views. Backbones pretrained: **ResNet-50, WideResNet-50-2, ViT-S/16.**

**Weights exist and download.** The README links pretrained backbones and three fine-tuned checkpoints on the University of Melbourne Mediaflux share. I followed the JS redirect to the direct endpoint and confirmed the payload:

```
http=200 size=745586637 type=application/zip
```

— `pretrained_weights.zip`, 745 MB, verified 2026-07-29. This is a real, downloadable, self-supervised encoder whose pretraining modality is elevation.

**Licence: CC BY-NC 4.0.** The repo `LICENSE` is the verbatim `Attribution-NonCommercial 4.0 International` legal code, and the README adds its own conditions block ("By downloading or using any part of this project, you agree to the following conditions (CC BY-NC 4.0)… Selling or licensing the data or derived works"). GitHub cannot classify it (`spdx_id: NOASSERTION`). So "open weights" holds only in the non-commercial sense — the same shape of constraint P101 analysed for the ImageNet MAE checkpoint.

**Why it is a near-miss rather than the scoop.** It pretrains on **DEM derivatives** — hillshade and VAT are 3-channel visualizations, not raw elevation, and the whole point of DEMAE is that you can skip the derivative step. It covers **one 301 km² site** in Victoria, Australia at 10 cm, so it is regional and at a resolution three orders of magnitude off DEMAE's. The objective is DINO, not MAE. And ViT-S/16 is the only ViT.

It is nonetheless the closest thing to "an open self-supervised encoder pretrained on elevation" in the literature, and §13's sentence does not survive contact with it either.

### 4.3 `xandergos/terrain-diffusion-30m` — trained on DEMAE's own corpus

| | |
|---|---|
| Repo | [`xandergos/terrain-diffusion-30m`](https://huggingface.co/xandergos/terrain-diffusion-30m) |
| Created | **2025-12-17**, last modified 2026-02-22 |
| Licence | **`mit`** |
| Weights | `base_model/`, `coarse_model/`, `decoder_model/`; base model 1,014,772,076 B, HTTP 200 verified |
| Likes / downloads | 17 / 0 |

Model card, in full: *"Terrain Diffusion Pipeline trained on the 30m Copernicus GLO-30 dataset."* Paper: arXiv [2512.08309](https://arxiv.org/abs/2512.08309), "InfiniteDiffusion: Bridging Learned Fidelity and Procedural Utility for Open-World Terrain Generation", Alexander Goslin, submitted 2025-12-09, v4 2026-05-03. Siblings: `terrain-diffusion-90m` and `terrain-diffusion-90m-pool2` on MERIT DEM, plus consistency-model variants under the same account.

**Not a scoop under this note's scope** — it is a hierarchical generative diffusion stack for terrain synthesis, not an encoder, and the released artifacts are diffusion UNets. Nothing in it is offered as a transferable backbone.

**But it does damage one specific sentence.** Any DEMAE framing of the form "the first open-weights model trained on Copernicus GLO-30 elevation alone" is false: this is MIT-licensed, on the hub since 2025-12-17, and trained on exactly that corpus. The word that has to be doing the work is **encoder**, and even that now needs the qualifiers in §2.

### 4.4 `hwh-datascience/divae-tokenizer-ahn-448` — a third-party elevation-only ViT-B/16

| | |
|---|---|
| Repo | [`hwh-datascience/divae-tokenizer-ahn-448`](https://huggingface.co/hwh-datascience/divae-tokenizer-ahn-448) |
| Created | **2026-03-26** |
| Licence | **`apache-2.0`** |
| Weights | `tokenizer.safetensors` **1,148,037,244 B** + `tokenizer.pt`, HTTP 200 verified |
| Downloads | 1 |

Card: *"A DiVAE (Diffusion VQ-VAE) tokenizer for **Fused AHN6 DSM + DTM elevation (2-channel, float32)** trained on Dutch national geospatial data. Part of the TerraVision-NL project."* Encoder ViT-B (`vit_b_enc`), patch 16, 448 px, FSQ codebook `8-8-8-6-5`, source AHN6 at 7.5 cm, normalization minmax clip [-20, 80].

Elevation-only pretraining modality, Apache-2.0, weights downloadable, ViT-B/16 — it ticks the literal boxes. What it is in practice: a **single-checkpoint hobby/project release** (`ahn-best-epoch-0002.ckpt` — epoch 2), one download, no paper, no evaluation, Netherlands-only, two channels rather than one. I record it because the brief asked for honest coverage and because it is the same DiVAE recipe as §4.1; it is not a competitive release and no reviewer will raise it. Its practical value to this note was as the safetensors-readable proxy that let me confirm the encoder subtree in §4.1.

Its siblings `divae-tokenizer-rgb-448` and `divae-tokenizer-cir-448` are the optical modalities of the same project — confirming the per-modality tokenizer pattern rather than any elevation-specific intent.

### 4.5 COP-GEN-Beta VAEs — a DEM-only VAE inside a multimodal generative model

[`mespinosami/copgen-vaes`](https://huggingface.co/mespinosami/copgen-vaes), created 2026-01-09, licence `cc-by-4.0`, arXiv 2603.03239, dataset `Major-TOM/COP-GEN-Benchmark`. The file listing includes a directory `DEM_64x64_DEM_latent_8/` alongside `S1RTC_…`, `S2L1C_…`, `S2L2A_…`, `LULC_…`, and the card states each VAE "is trained independently on its respective modality."

So: a **DEM-only KL-regularised VAE**, open weights, CC-BY-4.0, at 64×64 with 8 latent channels. Same structural pattern as §4.1 — a modality-specific autoencoder released as a component of a multimodal generative system. It is a convolutional VAE at low resolution, not a ViT and not a representation encoder, so it is the weakest of the five. Recorded for completeness; the card does not state the DEM source, scale, or resolution, which I could not resolve without the paper.

## 5. What did **not** change — negative evidence, stated explicitly

Each of these was checked and each came back clean. This is the part of §13 that survives.

- **GeomorPM remains closed.** Semantic Scholar by DOI (`10.1080/13658816.2024.2414409`) returns `openAccessPdf: {"url": "", "status": "CLOSED"}`. Authors Yang, Xu, Zhu, Liu, Zhou; published 2024-10-13; IJGIS 39(2). `gh search repos` for `GeomorPM` and `VQMAE-DEM` returns **0 results**; `gh search code GeomorPM` returns exactly one hit, a bibliography line in an unrelated awesome-list. My direct fetches to `tandfonline.com` were refused (**HTTP 403**, both WebFetch and curl with a browser UA), so I could not read the paper's data-availability statement — see §7. On every reachable signal, **no weights have appeared**. Plan §16's "(closed)" annotation is still correct.
- **Copernicus-FM has no DEM-only variant.** Full file listing of [`wangyi111/Copernicus-FM`](https://huggingface.co/wangyi111/Copernicus-FM) (licence `cc-by-4.0`): `CopernicusFM_ViT_base_varlang_e100.pth`, `CopernicusFM_ViT_large_varlang_e100.pth`, a `deprecated/` folder of DOFA-S checkpoints, and a `varname_embed/` folder. **No elevation-only checkpoint of any kind.** §13's characterisation of Copernicus-FM was and remains accurate.
- **TerraMind's backbone has no DEM-only variant.** Full author listing of `ibm-esa-geospatial`: TerraMind 1.0 tiny/small/base/large, six per-modality tokenizers plus Coords, Llama3-MS-CLIP-base, and two fine-tuned task models. The backbone is multimodal exactly as §13 says. The tokenizer is the miss, not the backbone.
- **EvaMAE is DEM-plus-RGB, not DEM-only, and its weights are not actually there.** Abstract (via Semantic Scholar; SIGSPATIAL 2025, published 2025-11-03, Gold OA CC-BY): *"introducing EvaMAE, a Masked Autoencoder (MAE) architecture that integrates elevation data for pre-training… uses cross-attention to combine DEM and RGB features."* Out of scope by design — DEM as an auxiliary channel to imagery. Worth one extra line: the abstract claims *"All our code, pre-trained models, and the EvaFlood dataset are available at https://github.com/saugatadhikari/EvaMAE"*, but the repo's top level is `README.md`, `env.yml`, `finetuning/`, `pretraining/` — **no weights, and no licence file at all** (`license: null`). Last push 2025-06-08. A paper-claimed release that did not happen.
- **MAESSTRO is ocean, not land.** [os.copernicus.org/articles/20/1309/2024](https://os.copernicus.org/articles/20/1309/2024/) — "Masked Autoencoders for Sea Surface Temperature Reconstruction under Occlusion." ViT-tiny, patch 4, 128×128 **single-channel SST**, ~750 k tiles from the LLC4320 ocean simulation; code on Zenodo (`10.5281/zenodo.13799966`). It is the closest *methodological* analogue to DEMAE anywhere — single-channel geoscientific raster, MAE, ~750 k tiles — and it contains **no elevation data**. It stays in §16 as a recipe precedent, not as prior art on the modality.
- **No DEM entry in TorchGeo's weight registry.** Read in full; listed in §3.4.
- **No DEM-only Prithvi or Clay variant.** Full author listings; §3.3.
- **Nothing new in the literature window.** The arXiv sweeps reach 2026-07-28 and 2026-07-27. The only July-2026 hit that looked threatening was arXiv [2607.08711](https://arxiv.org/abs/2607.08711), "LTM: Large-scale Terrain Model for Wildfire-prone Landscapes" (2026-07-09) — the name is alarming and the content is not: it is a multi-modal 3D **reconstruction** framework that uses legacy DEMs as geometric priors for image-based terrain reconstruction. Not a pretrained model, no weights mentioned, paper under CC-BY-NC-SA 4.0. **Not a scoop.**
- **arXiv `all:"DEM" AND all:"masked autoencoder"` returns exactly one paper**, [2309.03367](https://arxiv.org/abs/2309.03367), which plan §16 already cites. Three years on, DEMAE's specific combination is still nearly unoccupied in the preprint literature. That is a real and reportable finding, and it is the strongest single piece of evidence that the *narrowed* novelty claim holds.

## 6. What this changes

### 6.1 Plan §13 (Risks)

The row must be rewritten. It is not a likelihood change — the risk did not materialise between 2026-07-28 and today; it was **already realised and not seen**. Proposed replacement:

> Scooped by a new DEM FM | Realised in part, absorbed | Re-check 2026-07-29 (research/p100-scoop-recheck.md) found TerraMind's per-modality **DEM tokenizer** is an Apache-2.0 ViT-B/16 pretrained on ~9 M single-channel Copernicus DEM patches — so "first open encoder on elevation alone" is retired. What holds: first **masked-reconstruction** encoder on raw single-channel elevation released and evaluated as a transferable backbone, on a terrain-stratified corpus. Related work must name TerraMind-Tokenizer-DEM, DINO-CV, terrain-diffusion-30m. Re-check before release.

The re-check trigger should stay, and should specify **full-text card search plus registry source-reading**, not model-ID search — §3.1 is the trap.

### 6.2 Plan §14 (Interface to Phase 2)

§14's artifact list is untouched. Everything it promises — encoder-only timm-loadable weights, full MAE checkpoint, config, `dataset_version`/split fingerprints, training curves — is still exactly the right deliverable, and nothing found here makes any of it un-shippable.

One addition, and it costs nothing: §14 should require a **positioning line** in the release, naming the nearest open elevation-pretrained artifacts. A release that says "first open encoder on elevation" and ships next to a better-resourced IBM/ESA artifact doing the same thing is a credibility problem; a release that says "first *masked-reconstruction* encoder on raw elevation, and here is how it differs from TerraMind's DEM tokenizer" is a stronger claim precisely because it is checkable.

### 6.3 #88 — "Package the Phase-1 release artifacts" — **yes, this needs to change**

Plainly: **#88's release writeup needs to change, and the change is not cosmetic.**

1. **The novelty sentence must be narrowed.** Wherever #88's writeup says or implies "first open-weights encoder pretrained on elevation alone," that is now false and must be replaced with the §2 formulation. This is a one-paragraph edit, but it is load-bearing — it is the release's headline claim.
2. **#88 must carry a related-work section it currently does not have.** Minimum four entries with links, licences, and a sentence each on what differs: TerraMind-1.0-Tokenizer-DEM (§4.1), DINO-CV (§4.2), terrain-diffusion-30m (§4.3), COP-GEN-Beta VAEs (§4.5). Naming the strongest competitor first is the only framing that survives review.
3. **#88 should state the comparison it is *not* making.** The obvious reviewer question is "how does your ViT-B/16 compare to TerraMind's ViT-B/16 on the same modality?" That is a Phase-2 benchmark, not a Phase-1 deliverable, and #88 should say so explicitly rather than leave the gap silent.

### 6.4 #86 and the scratch-vs-continual headline — unaffected as an experiment, but with a live new option

**The R4-vs-R5 comparison is internal and nothing here touches it.** It compares two DEMAE runs; no external release changes what that measures or how it reports.

One consequential side-effect, marked as inference and flagged for the map rather than decided here: **P101 concluded that R4's ImageNet MAE init is not safe to build a shipped artifact on and recommended switching to OpenMMLab's Apache-2.0 reproduction, at a cost of ~2.9 pt ImageNet linear probe.** §4.1 puts a *third* option on the table that did not exist when P101 was written — an **Apache-2.0, in-domain, DEM-pretrained ViT-B/16 encoder** with a cleanly separable `encoder.*` subtree. Warm-starting from an in-domain elevation encoder is a different experiment from warm-starting from ImageNet, so it does **not** substitute for R4; but "continual pretraining from a 9 M-image DEM tokenizer encoder" is a cheap, licence-clean, and genuinely interesting arm, and it would answer the §6.3(3) reviewer question directly. *(Inference and a suggestion, not a finding: it is scope for the map to accept or reject, and it should not be smuggled into Phase 1 by this note.)*

## 7. Limits — what I could not reach or verify

- **Paywalled journals were not searched directly.** `tandfonline.com` (IJGIS) and `dl.acm.org` both returned **HTTP 403** to every fetch I attempted, including with a browser user-agent. I did not search ISPRS Journal, IJGIS, Remote Sensing of Environment, or Computers & Geosciences through their own interfaces. Coverage of that literature is indirect only — arXiv, Semantic Scholar metadata, GitHub, Hugging Face. §3.6 states the residual exposure.
- **GeomorPM's data-availability statement is unread.** I could not open the paper. The conclusion "still closed" rests on absent-evidence signals (no OA PDF per Semantic Scholar, no GitHub repo, no HF repo, no hub hits under any of 37 full-text queries), not on reading what the authors wrote about availability. That is a weaker basis than I would like, and if it matters, someone with institutional access should read the statement directly.
- **I did not download or open IBM's DEM tokenizer checkpoint.** The `encoder.*` structure and 90.8 M parameter count in §4.1 were verified on the architecturally identical AHN clone via a safetensors header range-read. IBM's file is a 1.15 GB `.pt`, which cannot be header-read. The inference is well-supported by TerraTorch's explicit `encoder_type="vit_b_enc"` registration but it is an inference, and nobody has demonstrated the extracted encoder works as a standalone backbone.
- **The TerraMind DEM tokenizer has no published transfer evaluation, and I did not run one.** Whether it is actually a good frozen feature extractor for DEM tasks is unknown to me, to IBM's card, and — as far as I can tell — to the literature. §2's argument that DEMAE retains a contribution depends on that gap staying open; it is an empirical question that Phase 2 could settle in either direction.
- **TerraMesh's DEM source is "Copernicus DEM" per the dataset card; I did not confirm GLO-30 vs GLO-90.** The card gives 10 m resolution (the Sentinel-2 grid), which implies resampling from a 30 m or 90 m source rather than a native product. I did not read arXiv 2504.11172 to pin it down.
- **COP-GEN-Beta's DEM VAE is under-characterised.** The card does not state the DEM source, scale, or resolution, and I did not read arXiv 2603.03239. Given it is a 64×64 convolutional VAE, resolving this would not change the verdict.
- **Non-English and non-arXiv preprint venues were not covered** — no search of CNKI, SSRN, TechRxiv, or EarthArXiv. GeomorPM's authors are at the Chinese Academy of Sciences, so a Chinese-language follow-up release is a specific blind spot.
- **The Hugging Face full-text index is not guaranteed complete**, and a model whose card mentions none of my 37 query terms — plausible for a terse or non-English card — would be invisible to §3.2. §3.3's tag and author enumerations partly compensate but only where a maintainer applied a tag.

## Sources

**The candidates**
- TerraMind DEM tokenizer card, weights, licence: https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-Tokenizer-DEM
- TerraTorch registration with full training `Namespace`: https://github.com/IBM/terratorch/blob/main/terratorch/models/backbones/terramind/tokenizer/tokenizer_register.py
- TerraMesh dataset (>9 M patches, 10 m, Copernicus DEM, CC-BY-SA-4.0): https://huggingface.co/datasets/ibm-esa-geospatial/TerraMesh · paper https://arxiv.org/abs/2504.11172
- TerraMind paper: https://arxiv.org/abs/2504.11171
- DINO-CV paper: https://arxiv.org/abs/2510.17644 · journal version https://www.sciencedirect.com/science/article/pii/S2666017226000106 · code + weights + CC BY-NC 4.0 LICENSE https://github.com/MLinArcheaomatics/BudjBimStoneWall-DINO-CV
- `xandergos/terrain-diffusion-30m` (MIT, Copernicus GLO-30): https://huggingface.co/xandergos/terrain-diffusion-30m · paper https://arxiv.org/abs/2512.08309
- `hwh-datascience/divae-tokenizer-ahn-448`: https://huggingface.co/hwh-datascience/divae-tokenizer-ahn-448
- COP-GEN-Beta VAEs (DEM-only VAE among them): https://huggingface.co/mespinosami/copgen-vaes

**The negatives**
- GeomorPM (closed; Semantic Scholar `openAccessPdf.status = CLOSED`): https://doi.org/10.1080/13658816.2024.2414409 · https://api.semanticscholar.org/graph/v1/paper/DOI:10.1080/13658816.2024.2414409
- Copernicus-FM full file listing (no DEM-only checkpoint): https://huggingface.co/wangyi111/Copernicus-FM
- TorchGeo model registry (no elevation entry): https://github.com/microsoft/torchgeo/blob/main/torchgeo/models/__init__.py
- Clay: https://huggingface.co/made-with-clay/Clay
- EvaMAE (DEM+RGB; repo has no weights, no licence): https://dl.acm.org/doi/10.1145/3748636.3762739 · https://github.com/saugatadhikari/EvaMAE
- MAESSTRO (SST, not elevation): https://os.copernicus.org/articles/20/1309/2024/
- LTM (reconstruction framework, not a pretrained model): https://arxiv.org/abs/2607.08711
- The one DEM+MAE preprint, already in plan §16: https://arxiv.org/abs/2309.03367

**Search endpoints used**
- HF model-ID search: `https://huggingface.co/api/models?search=…`
- HF full-text card search: `https://huggingface.co/api/search/full-text?q=…&type=model`
- HF tag/author filters: `https://huggingface.co/api/models?filter=…` / `?author=…`
- arXiv: `http://export.arxiv.org/api/query?search_query=…&sortBy=submittedDate`
- Semantic Scholar Graph API: `https://api.semanticscholar.org/graph/v1/paper/DOI:…`
- GitHub: `gh search repos`, `gh search code`, `gh api repos/…`

All URLs retrieved 2026-07-29.
