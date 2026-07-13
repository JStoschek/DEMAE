# DEMAE: A Foundation Model for Terrain

### A self-supervised ViT-MAE pretrained on global elevation data, fine-tuned for terrain analysis tasks

**Author:** Jonathan Stoschek
**Type:** Solo portfolio / applied-research project
**Compute:** vast.ai, ~$100–300 total budget
**Timeline:** 2–3 months, part-time
**Status:** Planning (v1, July 2026)

---

## 0. TL;DR

Almost all deep learning on Digital Elevation Models (DEMs) is still task-specific supervised learning: a bespoke CNN for super-resolution here, another for void-filling there. Meanwhile, satellite *imagery* has a healthy ecosystem of self-supervised foundation models (SatMAE, Scale-MAE, Prithvi, DOFA, Clay, TerraMind). There is no widely released, open-weights foundation model pretrained *directly and solely on elevation rasters*.

This project builds one: **DEMAE**, a Vision Transformer masked autoencoder pretrained self-supervised on hundreds of thousands of global DEM tiles, then fine-tuned with minimal task-specific data on two core downstream tasks (void filling and landform classification) plus two stretch tasks (super-resolution and landslide susceptibility). The deliverable is a released model + weights + a clean fine-tuning benchmark, positioned explicitly against the small amount of prior art that already touches this space.

The research below confirms the niche is *nearly* open, not *entirely* open, and that framing matters for how the project is positioned honestly.

---

## 1. Where this sits in the literature (and the honest gap)

The single most important thing to get right for a portfolio project is an **accurate novelty claim**. Overclaiming ("the first foundation model for terrain") is easy to falsify and reads as naive; a precise claim reads as competent. Here is the verified landscape as of mid-2026.

### 1.1 The one piece of real prior art: GeomorPM

**GeomorPM** (Zhou et al., *International Journal of Geographical Information Science*, online Oct 2024) is a peer-reviewed, self-supervised model pretrained *directly on massive DEM data*. Critically, it is fine-tuned on almost exactly the tasks in this plan — **DEM void filling, DEM super-resolution, and landform classification**. This means the headline "first foundation model for terrain" is **not available**.

However, GeomorPM differs from this project in ways that leave a genuine, defensible contribution:

- **Architecture:** It is a hybrid "VQMAE" — a convolution-based VQ-VAE fused with a Transformer MAE — not a clean ViT-MAE. That makes it heavier and less directly comparable to the imagery-FM lineage.
- **No public release:** Searches found no code or weights on GitHub or HuggingFace. It is a single-group model, not a reusable foundation model the community can build on.
- **Not positioned in the imagery-FM design space:** It does not adopt or ablate the design conventions (GSD-aware positional encodings, frequency-decomposed loss) that the satellite-imagery MAE literature has converged on.

**Your defensible framing:** *"An open-weights, pure ViT-MAE foundation model pretrained solely on elevation, released with a reproducible fine-tuning benchmark — trained on a hobbyist compute budget."* That is true, useful, and not covered by GeomorPM.

### 1.2 Everything else treats elevation as auxiliary

- **arXiv 2309.03367** (Sept 2023) — the closest "MAE on DEM" title — actually fine-tunes an *ImageNet*-pretrained MAE (frozen backbone + UperNet head) for building/road segmentation from DEMs. It reaches ~82% IoU and beats a UNet, proving the MAE-encoder-on-DEM pairing works, but it explicitly does **not** pretrain on elevation and lists "pre-train an MAE on a large corpus of DEM datasets" as *future work*. This is essentially a written invitation to do this project.
- **EvaMAE** (SIGSPATIAL 2025) integrates DEM as an *auxiliary conditioning modality* alongside RGB via cross-attention (encoder + decoder) during pretraining, and via ControlNet at fine-tuning. Evaluated on flood/landslide segmentation. DEM is a helper, not the subject. Code and the EvaFlood dataset are released — a useful reference implementation.
- **TerraMind** (IBM/ESA, ICCV 2025) is the flagship open multimodal EO foundation model. It includes Copernicus DEM as **one of nine** tokenized modalities, but its recipe is cross-modal masked-token *classification* (modality-specific FSQ tokenizers, cross-entropy loss, Dirichlet-sampled modality mixes) — not pixel-reconstruction MAE, and elevation is never the sole target.
- **Scale-MAE** (ICCV 2023), **Prithvi-EO-2.0**, **SatMAE**, **GFM**, **DOFA**, **CROMA**, **SSL4EO** — all pretrain on optical/SAR bands. **None uses elevation as a pretraining modality.** A comprehensive, community-maintained catalog of remote-sensing foundation models (checked to its May 2026 update) contains **no** entry described as DEM/elevation/terrain-pretrained.

### 1.3 The sobering benchmark warning: PANGAEA

**PANGAEA** (arXiv 2412.04204) evaluated ~9 popular geospatial foundation models and found they **do not consistently beat simple supervised baselines** (UNet, vanilla ViT), including under limited-label (10%/50%) conditions. Individual models win on individual tasks, but "foundation model → automatic win" is false in this domain.

**Two implications baked into this plan:**
1. Every downstream evaluation *must* include a strong from-scratch supervised baseline (UNet) and an ImageNet-init baseline. Beating those is the actual bar. If DEMAE only ties them, that is still a legitimate, publishable finding — and honestly reporting it is what separates a portfolio project from a hype piece.
2. PANGAEA includes **no elevation modality or task** — its 11 datasets are all optical/SAR. So there is no ready-made terrain benchmark; assembling a clean one is itself part of the contribution.

### 1.4 One-line positioning statement for your README/portfolio

> DEMAE is an open-weights ViT masked-autoencoder pretrained self-supervised on ~global Copernicus DEM tiles, released with a small reproducible benchmark for void-filling, landform classification, super-resolution, and landslide susceptibility. Unlike GeomorPM (hybrid VQ-VAE, unreleased) or TerraMind (elevation as 1-of-9 modalities), it is a clean, reproducible, elevation-only MAE trainable end-to-end for under $300 of rented GPU time — with supervised baselines reported honestly per the PANGAEA critique.

---

## 2. Data strategy

### 2.1 Primary pretraining corpus: Copernicus DEM GLO-30

**Why:** Global, free, permissively usable, cloud-native access, and it is the same elevation source TerraMind uses — which lets you gesture at compatibility.

- **Resolution:** 30 m (1 arc-second).
- **Access:** Public AWS bucket, no credentials: `aws s3 ls --no-sign-request s3://copernicus-dem-30m/`. Cloud-Optimized GeoTIFFs (COGs), so you can read windows without downloading whole tiles. STAC endpoint available for programmatic discovery.
- **License:** Free under the Copernicus DEM licence; attribution required. Fine for a portfolio/research release. (A small set of tiles over specific countries is restricted/withheld — just skip them.)
- **Structure:** 1°×1° tiles; ocean areas have no tiles (treated as zero elevation) — you will explicitly mask these out.

### 2.2 Secondary / specialty datasets

| Dataset | Res | Role in project | License note |
|---|---|---|---|
| **FABDEM** (Forests And Buildings removed) | 30 m | Bare-earth variant; optional second pretraining channel or a "clean terrain" fine-tune | **CC BY-NC-SA 4.0** — non-commercial only. Fine for a portfolio, but keep it out of anything you'd commercialize, and note it. |
| **SRTM 30 m** | 30 m | Real-world voids (its known data holes) give you *authentic* void-filling test cases, not just synthetic masks | Public domain (US Gov) |
| **USGS 3DEP 1 m lidar** | 1 m | High-res targets for the super-resolution stretch task; also a domain-shift test | Public domain |
| **ALOS AW3D30** | 30 m | Optional cross-sensor generalization check | Free, attribution |

### 2.3 Sampling: the part everyone gets wrong

Global DEM data is dominated by **flat, low-information terrain** (oceans, plains, deserts). Uniform random tiling wastes compute reconstructing near-constant patches and produces a model that has mostly learned "the world is flat." Mitigations:

1. **Land mask first** — drop ocean/zero tiles.
2. **Terrain-ruggedness stratification** — compute a cheap Terrain Ruggedness Index (TRI) or elevation std per candidate tile and *oversample* rugged terrain (mountains, canyons, coastlines, dunes) while keeping a representative tail of flat terrain so the model still handles it. Aim for a deliberately terrain-diverse corpus rather than a geographically uniform one — GFM showed a curated ~600k-sample set can outperform much larger homogeneous corpora.
3. **Geographic spread** — sample across climate/tectonic regimes (glaciated, fluvial, aeolian, tectonic, karst) so representations generalize.
4. **Target corpus size:** ~200k–500k tiles of 224×224 px (at 30 m ≈ 6.7 km across) is a sensible starting scale — comparable to GFM's GeoPile and well within budget. Start smaller (~50k) to debug the pipeline end-to-end before scaling.

### 2.4 Normalization — the single most important elevation-specific design choice

This is where DEM differs fundamentally from RGB and where the research explicitly flagged an open question. Raw elevations span roughly −400 m to 8,849 m globally, but what matters locally is **relief**, not absolute height. Options, in recommended order:

1. **Per-tile standardization (recommended default):** subtract the tile mean, divide by the tile std. This makes the model learn *terrain shape* invariant to absolute altitude — a Himalayan ridge and a small hill look structurally similar. Store the mean/std so you can invert for reconstruction metrics.
2. **Per-patch `norm_pix_loss`** (from the original MAE): compute the reconstruction loss against per-patch-normalized targets. He et al. found this improves representation quality; it composes naturally with per-tile standardization.
3. **Derived-channel augmentation (experiment):** feed slope and/or curvature (or a Laplacian of elevation) as extra input channels. Terrain analysis has long known slope/curvature carry the geomorphological signal; giving the model these directly may accelerate learning. Treat as an ablation, not the baseline.
4. **Robust clipping:** clip to a per-tile percentile range (e.g. 1st–99th) before standardizing to tame lidar spikes and DEM artifacts.

**Explicitly test** per-tile standardization vs. global scaling as an early ablation — it is cheap and likely your highest-leverage decision.

---

## 3. Model architecture

Start from the **official Facebook MAE codebase** (well-documented, supports gradient accumulation via `--accum_iter`) and modify minimally. Resist the urge to invent architecture; the novelty here is *domain + release + honest benchmark*, not a new block.

### 3.1 Backbone

- **ViT-Small/16 or ViT-Base/16** encoder. **Recommendation: start with ViT-S/16**, scale to B/16 only if the budget and results justify it. ViT-Large (the Scale-MAE / MAE-paper default) is out of scope for $300.
- **Input:** single-channel (elevation) 224×224, patch size 16 → 196 tokens. (If you add slope/curvature channels, input becomes 3-channel and you keep the RGB-style patch embed almost unchanged.)
- **Masking ratio:** **0.75** — confirmed optimal in both the original MAE and Scale-MAE ablations; no reason to deviate initially.
- **Decoder:** lightweight (MAE default is 8 blocks; Scale-MAE showed depth ~3 works well). Start with the MAE default, try shallow later.

### 3.2 Two elevation-motivated design borrows from Scale-MAE

Both are verified as directly reusable and cheap to add:

1. **GSD-aware positional encoding (GSDPE):** scale the sinusoidal positional encoding by the ratio of the tile's ground sample distance to a reference GSD, so *ground area covered* — not pixel count — sets positional scale. This is what will let a single DEMAE ingest 30 m *and* 1 m tiles coherently (crucial for the super-resolution stretch task and for multi-resolution generality). Add this once single-resolution training works.
2. **Frequency-decomposed reconstruction loss:** reconstruct a **low-frequency** component with **L2** loss (gets the broad landform right) and a **high-frequency** component with **L1** loss (sharpens ridges, valleys, edges). This maps beautifully onto terrain: low-freq = regional slope/basins, high-freq = drainage networks and breaklines. Implement as a stretch/ablation after the vanilla per-patch-MSE baseline works.

### 3.3 Recommended build order (de-risked)

1. Vanilla single-channel ViT-S/16 MAE, per-tile standardized, per-patch-norm MSE loss, 75% mask. **Get this training and reconstructing recognizable terrain first.**
2. Add derived-channel and normalization ablations.
3. Add GSDPE + multi-resolution tiles.
4. Add frequency-decomposed loss.

---

## 4. Pretraining recipe

### 4.1 Hyperparameters (starting point, adapted from MAE/Scale-MAE)

- **Effective batch size:** 4096 is the canonical MAE target. You will hit this with **gradient accumulation** (`--accum_iter`), *not* 64 GPUs. E.g. per-GPU batch 128 × accum 16 on a single 4090 ≈ 2048; two GPUs or accum 32 → 4096. A smaller effective batch (1024–2048) is fine at this scale and trains faster per step; tune down if memory-bound.
- **LR:** base LR 1.5e-4, scaled by the linear rule `lr = blr × eff_batch / 256`. With eff_batch 2048 → ~1.2e-3. Warmup 40 epochs (or ~5–10% of total).
- **Epochs:** the MAE paper uses 800; at small scale you will not afford that on the full corpus. Plan **~200–400 epochs on a ~50k–200k tile subset**, and watch the reconstruction-loss / linear-probe curves for diminishing returns rather than fixating on epoch count.
- **Precision:** **bf16** mixed precision (4090/5090/A6000 all support it) + FlashAttention. Free ~2× throughput and memory.
- **Optimizer:** AdamW, weight decay 0.05, betas (0.9, 0.95) — MAE defaults.

### 4.2 Efficiency levers (all verified, all worth using)

- **MAE's structural free lunch:** the encoder only sees the **25% visible tokens**, so MAE pretraining is intrinsically ~4× cheaper than a dense ViT. This is *why* this project is affordable.
- **bf16 + FlashAttention** as above.
- **DailyMAE**-style pipeline optimization (efficient data loading, avoiding CPU bottlenecks) got a ViT-B MAE to ~1 day of training; a COG-backed streaming loader with a local SSD cache on the vast.ai box is the practical equivalent.
- **Continual pretraining option (big lever):** GFM cut geospatial FM pretraining from ~768 to ~93 V100 GPU-hours (>8×) by initializing from ImageNet weights instead of from scratch. **Strongly consider initializing DEMAE's encoder from a public ImageNet-pretrained ViT-S/B MAE** (adapting the single-channel patch embed by averaging/replicating the RGB weights). This could be the difference between "one solid run" and "several ablation runs" within budget. Report both from-scratch and continual variants if you can afford it — that comparison is itself an interesting result.

### 4.3 Compute budget (the numbers)

Live vast.ai rates (July 2026): **RTX 4090 ≈ $0.34/hr**, **RTX A6000 ≈ $0.39/hr** (interruptible; on-demand somewhat higher). Your $100–300 budget therefore buys roughly:

| Budget | 4090 GPU-hours (@ ~$0.34) | What it realistically funds |
|---|---|---|
| $100 | ~290 hrs | 1 solid ViT-S pretraining run + a few fine-tunes + debugging headroom |
| $200 | ~580 hrs | ViT-S pretraining, 2–3 ablations, both core + one stretch task, baselines |
| $300 | ~880 hrs | ViT-B pretraining or from-scratch-vs-continual comparison, all 4 tasks, fuller ablation grid |

**Anchors that make this credible:** a full MAE reproduction + one extension was done in **~150 GPU-hours total**; SatMAE-from-scratch was ~768 V100-hrs but continual-pretraining (GFM) was ~93. A ViT-S MAE on a curated few-hundred-k-tile corpus for a few hundred epochs sits comfortably inside the $200 tier, especially with bf16 + FlashAttention + ImageNet init. **Budget ~20–30% of funds as slack** for crashed instances, data-loader debugging, and re-runs — this is a real cost on spot GPUs.

**Practical vast.ai tips:** pick instances with fast local NVMe and good download bandwidth (your bottleneck will be streaming COGs, not FLOPs); pre-stage a tiled, pre-normalized dataset to a cheap object store or the instance's disk; checkpoint every epoch to persistent storage because interruptible instances *will* get reclaimed; prefer a single strong GPU over multi-GPU until the pipeline is proven (avoids distributed-training debugging on rented time).

---

## 5. Downstream evaluation suite

Per your scope: **2 core tasks** (self-labelable, no external label hunting) + **2 stretch tasks**. Every task reports DEMAE (linear-probe *and* fine-tune) against a **from-scratch UNet** and an **ImageNet-init** baseline — this is the PANGAEA-mandated bar.

### 5.1 Core task A — Void filling *(self-supervised labels, trivial to generate)*

- **Setup:** mask regions of clean DEM tiles and reconstruct; this is literally MAE's native objective, so it is the most natural showcase. Evaluate on (a) **synthetic** voids (random blobs/rectangles) and (b) **real** SRTM voids for authenticity.
- **Metric:** RMSE / MAE of filled elevation vs. ground truth; visual plausibility of drainage continuity.
- **Why core:** zero labeling effort, directly leverages pretraining, produces compelling before/after visuals for a portfolio.

### 5.2 Core task B — Landform / geomorphon classification *(rule-based labels = free)*

- **Setup:** generate per-pixel landform classes with the classical **geomorphons** algorithm (a deterministic, well-established method — available in GRASS GIS / `richdem`-adjacent tooling) to create "free" labels, then fine-tune DEMAE to predict them from raw elevation. Tests whether pretrained features encode geomorphology.
- **Metric:** per-class IoU / accuracy vs. geomorphon ground truth; low-label regime (train on 1%, 10%, 50%) to demonstrate the foundation-model value proposition.
- **Why core:** GeomorPM used exactly this task (direct comparability), labels are free, and the low-label curve is the single most persuasive foundation-model plot.

### 5.3 Stretch task C — DEM super-resolution

- **Setup:** downsample 30 m (or 3DEP 1 m) tiles and reconstruct high-res; GSDPE + frequency-decomposed loss make DEMAE naturally suited. Public reference: the 2025 multi-modal DEM-SR benchmark (ScienceDirect S1569843225005126) and OpenDEM's SR resources.
- **Metric:** RMSE, PSNR, plus terrain-derivative fidelity (slope/aspect error — more meaningful than pixel PSNR for terrain).
- **Why stretch:** visually spectacular, but needs the multi-resolution machinery working first.

### 5.4 Stretch task D — Landslide susceptibility

- **Setup:** fine-tune on a public labeled benchmark. Candidates: **Landslide4Sense** (Zenodo 10463239 / iarai GitHub — multi-sensor, includes a slope layer) and the **USGS landslide-susceptibility-zonation benchmark** (ScienceDirect S0012825224002551). This is the one task needing external labels, hence stretch.
- **Metric:** AUROC / AUPRC for susceptibility; compare terrain-only DEMAE features vs. multi-modal baselines.
- **Why stretch:** highest real-world relevance and clearest "so what," but the most data-wrangling and the labels are noisiest.

### 5.5 Evaluation hygiene

- **Geographic hold-out:** split train/test by region (not random tiles) so you measure genuine generalization, not spatial autocorrelation leakage — a classic and damning mistake in geospatial ML.
- **Report linear probe AND fine-tune:** linear probe isolates representation quality; fine-tune shows practical performance.
- **Always show the baselines losing OR winning honestly.** If UNet ties DEMAE on a task, say so and analyze why (probably the task is easy / label-rich). That candor is a portfolio asset.

---

## 6. Phased timeline (2–3 months, part-time)

The plan front-loads a *presentable result* so there is always something to show, then adds depth. Each phase ends with a concrete artifact.

**Phase 0 — Foundations & data pipeline (Weeks 1–2)**
Set up repo, environment, and a *reproducible* data pipeline: COG streaming from `s3://copernicus-dem-30m`, land masking, TRI-stratified tiling, per-tile normalization, and a fast local-cache dataloader. Ship ~50k tiles as a debug corpus. **Artifact:** a documented `dataset/` module + a notebook visualizing sampled terrain tiles and the normalization.

**Phase 1 — Minimal viable pretraining (Weeks 3–4)**
Vanilla ViT-S/16 MAE on the debug corpus, on a single 4090, bf16 + FlashAttention. Goal: reconstructions that look like terrain, sane loss curves, checkpointing that survives instance reclaim. **Artifact:** reconstruction grids (masked → predicted → truth) + a training-curve writeup. This is your first shareable result.

**Phase 2 — Core downstream tasks + baselines (Weeks 5–6)**
Void filling and landform classification, each with UNet + ImageNet-init baselines and the low-label curves. **Artifact:** a results table and the money plot (accuracy vs. label fraction). At this point the project is already portfolio-complete.

**Phase 3 — Scale up & normalization/architecture ablations (Weeks 7–8)**
Scale corpus to ~200k tiles; run the key ablations (per-tile vs. global norm; derived channels; from-scratch vs. ImageNet-continual init). Possibly bump to ViT-B if budget allows. **Artifact:** an ablation section + the released, cleaned checkpoint(s).

**Phase 4 (stretch) — SR and/or landslide + multi-resolution (Weeks 9–10)**
Add GSDPE + frequency loss, wire up 3DEP 1 m, tackle super-resolution and/or Landslide4Sense. **Artifact:** SR before/after visuals and/or susceptibility maps.

**Phase 5 — Release & write-up (Weeks 11–12)**
Publish weights to HuggingFace, code to GitHub with a strong README, and a blog post / short technical report framing the contribution honestly against GeomorPM/EvaMAE/TerraMind and reporting the PANGAEA-style baseline comparison. **Artifact:** the public release — the actual portfolio centerpiece.

> If time compresses, Phases 0–2 + 5 alone constitute a complete, credible project. Phases 3–4 are depth, not prerequisites.

---

## 7. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| **Foundation model doesn't beat UNet** (the PANGAEA outcome) | Medium–High | Reframe as an honest finding; emphasize low-label regime where FMs help most; the *negative result, reported well* is still a strong portfolio piece. Include the low-label curve where pretraining advantage is largest. |
| **Data-loader is the bottleneck, GPU sits idle** | High | Pre-tile and pre-normalize to a local NVMe/webdataset shard format *before* renting GPU time; profile the loader on a cheap instance first. Never debug I/O on expensive GPU hours. |
| **Spot instance reclaimed mid-run** | High (interruptible) | Checkpoint every epoch to persistent storage; make training resumable from the start; keep a 20–30% budget buffer. |
| **Budget overrun** | Medium | Start with ViT-S and the debug corpus; only scale after the pipeline is proven; use ImageNet-continual init to cut pretraining cost ~8×. |
| **GeomorPM/others make the novelty look thin** | Medium | Lean into the *open-weights + reproducible + hobbyist-budget + clean ViT-MAE* framing, which none of them offer; do not claim "first." |
| **Absolute-elevation normalization sabotages learning** | Medium | Make per-tile standardization the default and ablate it early — it's cheap and high-leverage. |
| **FABDEM's NC license** | Low | Keep FABDEM optional and clearly labeled; base the released model on Copernicus GLO-30 (permissive) so the release is clean. |
| **Field moves under you** (new DEM-FM appears) | Low–Medium | Re-check arXiv/HuggingFace at project start and before release; the reproducible-open-budget angle survives most new entrants. |

---

## 8. Portfolio deliverables (what a reviewer actually sees)

1. **GitHub repo** — clean, documented, reproducible: data pipeline, pretraining, fine-tuning, eval. This is the primary artifact for an ML/engineering portfolio.
2. **HuggingFace model card + weights** — DEMAE-S (and -B if trained). Discoverable, reusable, signals you can ship.
3. **A written report / blog post** — the narrative: the gap, the design decisions (especially the elevation-specific normalization and loss), the honest baseline comparison, the money plots. Frames you as someone who reasons about the literature, not just trains models.
4. **Compelling visuals** — masked→reconstructed terrain grids, void-fill before/after, low-label accuracy curves, (stretch) SR and susceptibility maps. These are what get remembered.
5. **A short "model card of limitations"** — where it fails, the PANGAEA caveat, geographic-generalization notes. Demonstrates maturity.

---

## 9. Key references

**Prior art / niche:**
- GeomorPM — *IJGIS* 2024, doi:[10.1080/13658816.2024.2414409](https://www.tandfonline.com/doi/full/10.1080/13658816.2024.2414409) (the one to differentiate against)
- Self-Supervised Masked DEM Encoding — arXiv [2309.03367](https://arxiv.org/abs/2309.03367) (MAE-on-DEM proof of concept; names this project as future work)
- EvaMAE — SIGSPATIAL 2025, doi:[10.1145/3748636.3762739](https://dl.acm.org/doi/10.1145/3748636.3762739) (DEM as auxiliary modality)
- TerraMind — ICCV 2025, arXiv [2504.11171](https://arxiv.org/abs/2504.11171) (DEM as 1-of-9 modalities)

**Design lineage:**
- MAE — He et al., CVPR 2022; official code + recipe: [github.com/facebookresearch/mae](https://github.com/facebookresearch/mae/blob/main/PRETRAIN.md)
- Scale-MAE — ICCV 2023, arXiv [2212.14532](https://arxiv.org/abs/2212.14532) (GSDPE + frequency-decomposed loss)
- GFM (continual pretraining) — ICCV 2023 (the ~8× compute-cut result)
- DailyMAE — arXiv [2404.00509](https://arxiv.org/html/2404.00509v1) (MAE in ~1 day)

**Benchmark caution:**
- PANGAEA — arXiv [2412.04204](https://arxiv.org/abs/2412.04204) (FMs don't always beat UNet; no terrain tasks)

**Data:**
- Copernicus DEM GLO-30 — [registry.opendata.aws/copernicus-dem](https://registry.opendata.aws/copernicus-dem/) (`s3://copernicus-dem-30m`, COG, free)
- FABDEM — [gee-community-catalog.org/projects/fabdem](https://gee-community-catalog.org/projects/fabdem/) (bare-earth, CC BY-NC-SA)
- Multi-modal DEM-SR benchmark 2025 — ScienceDirect [S1569843225005126](https://www.sciencedirect.com/science/article/pii/S1569843225005126)
- Landslide4Sense — Zenodo [10463239](https://zenodo.org/records/10463239) / [iarai/Landslide4Sense-2022](https://github.com/iarai/Landslide4Sense-2022)
- USGS landslide susceptibility benchmark — ScienceDirect [S0012825224002551](https://www.sciencedirect.com/science/article/pii/S0012825224002551)

**Compute:**
- vast.ai live pricing — [vast.ai/pricing](https://vast.ai/pricing) (4090 ≈ $0.34/hr, A6000 ≈ $0.39/hr as of July 2026)

---

*Plan v1 — assembled from a verified literature review (mid-2026). Re-check arXiv/HuggingFace for new DEM foundation models before public release, as the field is moving quickly.*
