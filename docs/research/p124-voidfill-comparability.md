# P124 — what published void-filling numbers this table can stand beside

Ticket [#124](https://github.com/JStoschek/DEMAE/issues/124) · map [#117](https://github.com/JStoschek/DEMAE/issues/117) · plan `docs/research/task1-void-fill-plan.md` §1, §2.1, §2.4.4, §3.2, §3.3, §3.4, §3.5, §5, §9 · memo `docs/research/downstream-task-selection.md` §1.4, §2.2, §2.4

> Everything below is either a quote/figure from a primary source with a URL, a value computed here from arithmetic that is shown, or an inference explicitly marked as one. Where a source could not be reached, that is stated rather than papered over with a secondary account.
>
> **Provenance note.** The Diff-DEM repository, Gavriil, GrounDiff, FathomDEM, Luedeling, Höhle & Höhle, Reuter et al., ET-SDE, the Copernicus DEM Product Handbook and ASPRS Edition 2 were opened and quoted directly. The Bielski, MERIT Hydro, FathomDEM-supplementary, Dfilled, EuroSDR-60 and GeomorPM-Figshare material was gathered by parallel search agents; those quotes carry their URLs and were not re-opened here, so treat their exact wording as one remove from the page even though the sources themselves are primary.

## 1. What was asked, and why it matters

Plan §9 promises "the results table (arms x buckets x strata, CIs marked)". Memo §1.4 makes comparability a *selection criterion* — tasks are preferred "where an open competitor ships code + data + weights (Diff-DEM), so numbers land next to something" — and §2.1 calls void filling "GeomorPM's flagship task". Plan §3.2 then matches two void-size buckets to Diff-DEM's "so numbers are directly comparable".

That is three separate load-bearing claims about external comparability, and none had been checked against the sources. If they are wrong, the benchmark still works — it is an internally controlled comparison with a shared head, shared masks and shared seeds, and that is the instrument C1 and C3 actually need — but §9's table gets one fewer column and the writeup needs a paragraph it does not currently have.

The ticket also asks that two citation-backed design claims be *verified* rather than inherited: that residual-over-classical-fill is current best practice (plan §2.4.4, citing GrounDiff), and that stratified DEM evaluation is standard practice (plan §5, citing FathomDEM).

## 2. Bottom line

**There is no published number this benchmark's headline table can be placed beside. The honest framing is "internally controlled comparison, externally incomparable", and it should be written into the plan.** That is the finding; it is not a placeholder for a better answer.

Concretely, each evidenced below:

1. **Diff-DEM's numbers are unreachable and, from its own committed code, would not be commensurable even if reached.** The paper is closed-access with no preprint (§3.1). Its benchmark is 10 m Norwegian laser-derived terrain on 256 px tiles; the model actually runs on a 2×2 **max-pooled** 128 px grid and the prediction is quantised to 257 levels before being scored (§3.3). Most decisively, the affine map that converts predictions back to metres is fitted on the **ground-truth tile's own min and max, including the pixels inside the void** (§3.5) — precisely the leak plan §2.1 forbids for DEMAE. Numbers produced under a leaking de-normalisation cannot share a column with numbers produced under a non-leaking one.
2. **The one real comparison available is the one the plan already has: A8, run by us.** Diff-DEM's checkpoint, dataset and mask lists are live and downloadable (§3.6). Running their model and ours on the same tiles under *one* scoring function is a genuine head-to-head. It is a transfer row, not a headline row, and the number in it is one we compute — not one we cite.
3. **Gavriil et al. (2019) is the only open, quotable number in this lineage, and its own authors disclaim it**: "Complete recovery of the ground truth is an unrealistic goal, so these results provide limited insight" (§4). Its metrics are MSE (m²) and an Earth-Mover distance on intensity histograms — neither is in plan §5's metric set.
4. **GeomorPM's "task-level comparability" does not survive inspection — but its artifacts do.** The paper is paywalled and its numbers were not read. From its own released code and data: the grid is EPSG:3395 World Mercator, so its "30 m" is ~22–24 m of true ground distance; the source DEMs are AW3D30/SRTM, not GLO-30; and its void difficulty axis is *fraction of tile occluded* in two coarse regimes (10.6–18.6 %, 40.5–59.6 %) with scattered multi-component masks, so plan §3.2's size buckets have no counterpart at all (§5.3). **However, GeomorPM released its weights, its fine-tuning tiles and its void masks under CC BY 4.0** — a cleaner licence and a closer regime than Diff-DEM's — which makes a second, better cross-eval arm runnable. §5.4 proposes it as **A10**, and it is the most valuable actionable thing in this note.
5. **The canonical evaluation of DEM void filling publishes no metres.** Reuter et al. (2007) — 883 citations, the study behind CGIAR-CSI's hole-filled SRTM v4 — reports **mean *rank* of RMSE**, never RMSE in metres (§7.1). Its protocol (8 log-spaced void-size classes × 6 terrain units, real void shapes transplanted onto matching terrain, error over the whole void area) is plan §3.2/§3.3/§5 almost exactly, independently re-derived; the plan should cite it. But there is nothing in it to stand beside either.
6. **Correction to plan §2.4.4.** GrounDiff's residual is the nDSM — a residual over the **input DSM**, not over a classical interpolation fill — and GrounDiff does not do void filling at all. FathomDEM's residual is over **COPDEM**, likewise the input. Neither paper claims residual prediction is "current best practice"; GrounDiff supports it with an ablation instead, which is stronger. The plan is right about the design and wrong about who says so. The correct citations are 20 years older: Grohman et al.'s delta surface fill (2006) and Luedeling et al.'s TIN delta surface (2007) are residual-over-a-base-fill *for void filling specifically* (§6.3).
7. **Plan §5's stratification claim is correct and under-cited.** FathomDEM does stratify — 23 land-cover/slope masks, Table 1 — but it is a bare-earth-correction paper. Luedeling et al. (2007) Table 3 stratifies *void-fill* error by slope class, inside vs outside the void, and is the better citation (§6.2, §6.3).
8. **Two internal problems found in passing.** Plan §3.2's bucket-matching claim overreaches — matching pixel counts preserves the fraction of tile occluded but triples the ground extent (§3.7) — and plan §3.2's own "void may not cover > 40% of the tile" constraint **forbids the upper half of its own large bucket** (§9).

## 3. Diff-DEM, read from its committed code

### 3.1 The paper is closed access, and there is no preprint

Lo, K. S.-H. and Peters, J., "Diff-DEM: A Diffusion Probabilistic Approach to Digital Elevation Model Void Filling", *IEEE Geoscience and Remote Sensing Letters* 21, 1–5 (2024), doi:[10.1109/LGRS.2024.3403835](https://doi.org/10.1109/LGRS.2024.3403835), IEEE Xplore document [10535979](https://ieeexplore.ieee.org/document/10535979).

Semantic Scholar's record for the DOI reports `"openAccessPdf": {"url": "", "status": "CLOSED", "license": null}` (retrieved 2026-08-06 via `api.semanticscholar.org/graph/v1/paper/DOI:10.1109/LGRS.2024.3403835`). There is no arXiv entry, no author-hosted copy found, and ResearchGate carries only a "Request PDF" stub. **Table I and Table II of Diff-DEM — the actual numbers — were not reached.** Everything in §3.2–§3.7 is derived from the repository, which is the authors' own released implementation, plus the abstract.

The abstract (verbatim, from the Semantic Scholar record of the DOI) fixes the regime:

> "The comparison is on Gavriil's and on our benchmark that expands Gavriil's dataset from 63 to 217 full-size (5051 × 5051) 10-m GeoTIFF images sourced from the Norwegian Mapping Authority; and from 50 DEMs to three groups of 1 k each of increasing void size."

### 3.2 The regime

| | Diff-DEM | DEMAE plan §3.1–§3.2 |
|---|---|---|
| Product | Norwegian Mapping Authority laser-derived DTM (hoydedata.no) | Copernicus GLO-30 (radar DSM) |
| Extent | Norway | global, corpus test geocells |
| GSD | 10 m | 30 m (primary regime) |
| Tile | 256 × 256 px (2.56 km) | 224 × 224 px (6.72 km) |
| Effective model grid | **128 × 128 px @ 20 m** (§3.3) | 224 × 224 px @ 30 m |
| Eval instances | 3 buckets × 1 000 | 5 000 test tiles × 3 buckets × 3 families + ~2 000 real-void |
| Void geometry | axis-aligned rectangles only | blobs, rectangles, transplanted real SRTM shapes |
| Buckets (file px) | 64–96, **96–128**, 128–160 | 16–48, 64–96, 128–160 |

Sources: `README.md` and `evaluation.sh` (`difficulty=("64-96" "96-128" "128-160")`) at <https://github.com/kylelo/Diff-DEM>; `config/dem_completion.json` (`"image_size": [256, 256]` for test, `[128, 128]` for train); the abstract above.

Note the plan's bucket set drops Diff-DEM's middle bucket (96–128) and adds a small one (16–48) below Diff-DEM's range. Only two of three buckets are shared even nominally.

Void geometry is rectangles-only in Diff-DEM's *training* sampler, and the benchmark mask directories are named by a width range rather than a shape family, so there is no blob or real-void family to compare against:

```python
# data/dataset.py, DEMDataset.get_crop_bbox
bbox_width  = np.random.randint(32, 81)
bbox_height = np.random.randint(32, 81)
```

(That draw is in *model-grid* pixels, i.e. 64–160 in file pixels after the 2× pooling of §3.3 — consistent with the docstring's "size ranging from 64 to 160 pixels".)

### 3.3 The preprocessing pipeline, which the paper's metrics inherit

`data/dataset.py` composes the ground-truth transform as:

```python
self.gt_tfs = transforms.Compose([
    transforms.ToTensor(), ToFloat32(),
    transforms.RandomCrop(256),
    DEMNormalize(),
    MaxPooling2DTransform(kernel_size=2, stride=2),
])
```

Three consequences, none visible from the abstract:

1. **`DEMNormalize` is per-tile min–max over the *complete* tile.** From `data/util/dem_transform.py`: `min_val, max_val = torch.min(height_meter), torch.max(height_meter)`, then rescaled to [-1, 1]. It is applied to `gt_img` *before* the void is stamped in (`cond_img = gt_img.clone()`; `cond_img[mask > 0] = -1`), so the normalisation constants are computed from pixels that are then declared missing.
2. **The DEM is 2×2 max-pooled.** Not averaged — `nn.MaxPool2d(kernel_size=2, stride=2)`. So the network sees a 128 × 128 grid at an effective 20 m spacing, built by taking the maximum of each 2 × 2 block. A max filter on elevation is upward-biased and preferentially preserves ridges.
3. **Scoring is at 256 × 256 against the unpooled truth.** `data/util/tif_metric.py`: `if image.shape != (256,256): image = resize(image, gt_image.shape, mode='reflect', anti_aliasing=False)`. So the reported error includes one downsample-by-max and one upsample-by-interpolation that are properties of the harness, not of the method.

There is also a quantisation step. `core/util.py`'s `tensor2img` writes uint16 output as `((img_np+1) / 2 * scale_factor * 256).round()`, and `evaluation.sh` passes `--scale_factor 1`. Predictions are therefore stored on **257 integer levels spanning the tile's full elevation range** before being converted back to metres. For a tile of relief *R* metres that is a quantisation floor of roughly *R*/886 m RMSE (uniform-quantisation step *R*/256, divided by √12) — around 1 m on a 1 000 m-relief Norwegian tile. Small relative to the errors involved, but it is a floor the DEMAE arms do not have, and it is one more reason the two harnesses are not interchangeable.

### 3.4 The metrics

`data/util/tif_metric.py` prints exactly four numbers:

```
MAE: ... RMSE: ... MS-SSIM: ... PSNR: ...
```

- **MAE and RMSE** are computed strictly inside the void (`diff = gt_image[mask > 0] - image[mask > 0]`), in whatever unit the uint16 GeoTIFF encodes — metres for a 10 m Norwegian DTM. **This matches plan §5's "all metrics computed only inside the void, in metres"** — the reporting *shape* is shared even though the numbers are not.
- **MS-SSIM and PSNR are scored on the mask's *bounding box*, not the mask.** `crop_to_mask` crops both rasters to the bounding box of `mask > 0` and scores the whole crop. For Diff-DEM's own masks this is harmless — they are axis-aligned rectangles (`get_crop_bbox` draws `bbox_width`/`bbox_height` independently; ET-SDE, §7, describes the protocol as "Diff-DEM assumes voids appearing in rectangular patches"), so bounding box and void coincide. It silently stops being void-only the moment the mask is not a rectangle, which is three of plan §3.2's four geometry families. **This is a concrete reason the plan cannot simply reuse their script on our masks.**
- Both use a per-image `data_range = max(gt.max(), pred.max())` over that crop, and both clamp negatives to zero. A per-image, data-dependent dynamic range makes PSNR non-poolable across tiles; the script averages it anyway.
- MS-SSIM uses three scales only (`scale_weights = torch.tensor([0.0448, 0.2856, 0.3001])`, unnormalised) — a truncation of the standard five-scale weights. Three scales with piq's default 11-px kernel also imposes a minimum crop of ~41 px, so plan §3.2's small bucket (16–48 px) would fail outright in this harness. Another reason to score everything with our own function.

Separately, `models/metric.py`'s `masked_mae`/`masked_rmse` — the metrics named in `config/dem_completion.json` and therefore what appears in the training logs — are a different and much cruder thing: they rescale by a hardcoded `scale_factor=80`, clip at `max_thres=80`, multiply by 1e3, and mask on `target > valid_thres` (i.e. by *elevation value*, not by the void mask). These are not the published metrics, but anyone reading the repo's tensorboard curves as if they were "MAE in metres" will be misled.

**Table II is bucketed by void size.** The commented-out argument block in `tif_metric.py` is labelled `# Table II` and enumerates exactly three `--mask_dir` values, `64-96`, `96-128`, `128-160`, each with its own `--algo_dir`. So error-in-metres-inside-the-void, reported per void-size bucket, is Diff-DEM's own reporting shape — confirming that half of the ticket's convention question ("is this the field's normal reporting shape?") is answered yes by this source.

### 3.5 The de-normalisation is fitted on the ground truth

This is the finding that settles the same-table question.

```python
# data/util/tif_metric.py, main()
if normalize: # Only for Diff-DEM!!
    cond_min, cond_max = np.min(gt_image), np.max(gt_image)
    image /= 256
    image *= (cond_max - cond_min)
    image += cond_min
```

The model's output lives in a normalised space; to score it in metres it must be mapped back by an affine transform. The two constants of that transform are taken from `gt_image` — the ground-truth tile, whole, **including the void interior**. The variable names (`cond_min`, `cond_max`, for the *conditioning* image) suggest the intent was the visible part; the code uses the answer.

This is not an isolated slip in the eval script — it is consistent with the pipeline, because `DEMNormalize` computed the same constants from the same complete tile at training time (§3.3). The leak is at the pipeline level, and it applies to the `--normalize` path, which the README and `evaluation.sh` both flag as "Only for Diff-DEM" / "Only required by Diff-DEM".

Two consequences:

- **At inference on a genuinely voided DEM this information does not exist.** If the void covers the tile's summit, the tile maximum is unknown. The published protocol supplies it.
- **It is exactly what plan §2.1 rules out for DEMAE**: "mean/std computed from non-void pixels only (a voided tile's true stats are unknowable at inference; using full-tile stats would leak)". Placing a leaking number and a non-leaking number in adjacent cells of one table, with no column separating them, would be the single most misleading thing this benchmark could do.

*(Inference, marked: the magnitude of the leak is not quantifiable from the repo alone — it depends on how often a void contains the tile extremum, which needs the dataset. The dataset is downloadable (§3.6), so this is measurable if anyone wants the number. An upper bound on the benefit is the difference between fitting the affine map on the full tile versus on the visible pixels, which is zero for tiles whose extrema are outside the void and can be tens of metres for tiles whose extrema are inside it.)*

Note the asymmetry this creates in the baseline rows too: the classical baselines (`spline`, `void_fill`, `kriging`, `pm_diff`, `generative_model` in the commented block) are scored with `normalize=False` — they output metres directly and need no rescaling. So Diff-DEM's own comparison is between one method that is handed the truth's dynamic range and several that are not. *(Inference, marked: this argues the direction of any bias favours Diff-DEM in its own table.)*

### 3.6 Reproduction hazards in the committed code

Relevant to plan §3.4 and §4.5's A8 step, and to plan §8's risk list:

- **The dataset and checkpoint are live.** The Google Drive folder linked from the README returned HTTP 200 on 2026-08-06. `pretrained/760_Network.pth` and `dataset/norway_dem/benchmark/` (with `benchmark_gt.flist`, `mask_64-96.flist`, and `mask/{64-96,96-128,128-160}/`) are what A8 needs. **A8 is feasible.**
- **`--normalize` cannot be turned off from the command line.** `parser.add_argument('--normalize', default=True, action='store_true')` — with `store_true` and `default=True`, the flag is `True` whether or not it is passed. The README's "otherwise do not use" is not executable as written. The paper's baseline rows were evidently produced by editing the `default=` in `__main__` (the commented-out blocks each carry `default=False`), so this is a hazard for *reproducers*, not necessarily a defect in the published table. Anyone re-scoring A1-class baselines with this script must edit the source, not the CLI.
- **The committed `evaluation.sh` runs one diffusion step**, `step=1`, against the README's `--n_timestep 512`. A third-party evaluation (ET-SDE, §7) reports that Diff-DEM needs 128–512 steps depending on difficulty. Running `evaluation.sh` as committed reproduces nothing.
- **The repository carries no licence.** `gh api repos/kylelo/Diff-DEM --jq .license` returns `null`; there is no LICENSE file in the tree. The Gavriil-derived portion is separately restricted: `konstantg/dem-fill`'s README states "CC 4.0 Attribution-NonCommercial International — The software is for educational and academic research purpose only", and the underlying elevation data is "courtesy Norwegian Mapping Authorities (www.hoydedata.no), copyright Kartverket (CC BY 4.0)". **Plan §3.5's "Diff-DEM's data per their repo terms" names terms that do not exist.** The dataset card must say: Diff-DEM — no stated licence, used for evaluation only; Gavriil-derived subset — CC BY-NC 4.0, academic use only; Kartverket source data — CC BY 4.0.

### 3.7 So: same table, or prose?

**Prose, plus one clearly-labelled transfer block containing numbers we computed.** The specific reasons, in order of force:

1. The de-normalisation leak (§3.5) makes their published numbers and ours measurements of different quantities.
2. The numbers are behind a paywall, so they cannot be quoted with a table/row reference anyway (§3.1).
3. Different product (laser DTM vs radar DSM), different country, 3× GSD, different tile extent, rectangles-only geometry, and a max-pool/upsample harness that is part of the measurement (§3.2–§3.3).
4. **The bucket match does not do what plan §3.2 claims.** Matching pixel counts across a 10 m / 256 px grid and a 30 m / 224 px grid:

   | Bucket | Diff-DEM ground extent | Plan ground extent | Diff-DEM % of tile width | Plan % of tile width |
   |---|---|---|---|---|
   | 64–96 px | 640–960 m | 1 920–2 880 m | 25.0–37.5 % | 28.6–42.9 % |
   | 128–160 px | 1 280–1 600 m | 3 840–4 800 m | 50.0–62.5 % | 57.1–71.4 % |

   (Arithmetic: bucket px × GSD, and bucket px ÷ tile px.)

   The honest reading cuts both ways and the plan should say both. Matching pixel counts *does* approximately preserve the fraction of the tile occluded — which is the axis that governs how much context an inpainter has left, and is arguably the right difficulty axis. It does **not** preserve ground extent, which is off by 3×, and it therefore does not preserve the number of terrain autocorrelation lengths spanned. A physically identical radar-shadow void lands in different buckets in the two benchmarks. "Matched so numbers are directly comparable" is too strong; "matched on fraction-of-tile-occluded, which is the difficulty axis; ground extents differ by 3× and the metres are not comparable" is what is true.

## 4. Gavriil et al. — the only open numbers in this lineage

Gavriil, K., Muntingh, G., Barrowclough, O. J. D., "Void Filling of Digital Elevation Models with Deep Generative Models", *IEEE GRSL* 16(10), 1645–1649 (2019), doi:[10.1109/LGRS.2019.2902222](https://doi.org/10.1109/LGRS.2019.2902222). Preprint: <https://arxiv.org/abs/1811.12693> (open). Code and data: <https://github.com/konstantg/dem-fill>.

This is the dataset Diff-DEM's Table I extends, so it is the root of the lineage. Its protocol, from §IV of the preprint:

- **G1** trained on 2 m DEMs of Oslo, Trondheim, Bergen ("urban"); **G2** on 10 m DEMs of Western and Eastern Norway ("rural"). Input 256 × 256. Voids "range from 64×64 up to 128×128 (not necessarily rectangular) and randomly placed" (repo README).
- Evaluation is on **50 urban + 50 rural** randomly selected DEMs, versus IDW (`gdal_fillnodata.py` with two 3×3 smoothing passes) and LR B-spline interpolation, with no post-processing.

Table I in full:

| | | G1 | G2 | IDW | LR B-spline |
|---|---|---|---|---|---|
| Urban (2 m) | MSE | 28.76 | — | 22.10 | **17.08** |
| | EM | 10.28 | — | **8.21** | 12.99 |
| Rural (10 m) | MSE | — | **809.09** | 1079.01 | 868.13 |
| | EM | — | **8.55** | 9.37 | 11.32 |

Two things matter here. First, **the classical methods win on urban MSE** — LR B-spline 17.08 vs the GAN's 28.76 — which is direct published support for plan §0's C2 hedge that splines are strong on the easier regime, and for §8's "A1 pre-fill quality ceiling" risk.

Second, **the authors disclaim their own table**, §IV:

> "Complete recovery of the ground truth is an unrealistic goal, so these results provide limited insight. However, quantitative measures, while being less subjective, may not directly correspond to how humans perceive and judge generated images [19], which is reflected in the table."

And in §V they list "using other evaluation metrics more suitable to this use case" as future work.

**Can these numbers stand beside ours? No.** MSE is in m² (RMSE ≈ 28.4 m rural, 5.4 m urban), which plan §5 does not report; "EM" is an Earth-Mover distance between *histograms of intensities*, a distributional metric with no counterpart in plan §5 at all; the strata are "urban"/"rural" rather than ruggedness; n = 50 per cell; and the void size range (64–128 px at 10 m = 640–1 280 m) overlaps only part of one of the plan's buckets. They are quotable — usefully, as prior art on the classical-vs-learned question — but only in prose.

## 5. GeomorPM — the claim fails, but the artifact is a gift

Yang, J., Xu, J., Zhu, Y., Liu, Z., Zhou, C., "GeomorPM: a geomorphic pretrained model integrating convolution and Transformer architectures based on DEM data", *Int. J. Geogr. Inf. Sci.* 39(2), 422–451, published online 13 Oct 2024, doi:[10.1080/13658816.2024.2414409](https://doi.org/10.1080/13658816.2024.2414409). All authors at LREIS / IGSNRR, Chinese Academy of Sciences.

### 5.1 The paper is paywalled; the numbers were not reached

Taylor & Francis returns HTTP 403 to automated fetches and USD 64 / 48 h to a browser. OpenAlex reports `is_oa: false`, `oa_status: closed`; Unpaywall reports `oa_locations: []`, `has_repository_copy: false`; Europe PMC and CORE return zero hits; there is no arXiv or institutional copy. Crossref carries no abstract (`abstract: null`) and no licence (`license: null`).

**Sections 4–5 — the experimental design, the baselines, and every numeric result — were not read.** Nothing below claims to report GeomorPM's published numbers.

What *was* verified: the abstract (identically from the Semantic Scholar API and the publisher's access page), which confirms the task list —

> "GeomorPM, based on the VQMAE architecture, was pretrained on massive DEM data and fine-tuned for three specific tasks: DEM void filling, DEM superresolution, and landform classification. GeomorPM outperformed the traditional and other deep learning methods in all three tasks"

— and the data-availability statement: *"The data and codes that support the findings of this study are available on Figshare at https://doi.org/10.6084/m9.figshare.25756806.v1."*

### 5.2 The Figshare deposit is open, and it is the real finding

Figshare item "DEM foundation model", published 2024-05-06, **CC BY 4.0**, doi:[10.6084/m9.figshare.25756806.v1](https://doi.org/10.6084/m9.figshare.25756806.v1). It contains the pretrained weights (`vqmae.ckpt`, 4.66 GB; `vq-f4.ckpt`, 723 MB), the pretraining and per-task fine-tuning code, and `Areas.zip` (6.70 GB): the five fine-tuning study areas at four resolutions, 85,565 clipped tiles, and the void masks. Everything in §5.3 is read off those files, not off the paper.

The pretraining *corpus* is not released — the README states "The pre-training datasets are not provided in this repository due to their large size. If you need access to the pre-training datasets, please contact the author directly." **Whether GeomorPM's pretraining was global or China-only therefore could not be determined**, and the memo should not assume either way. The only primary signal is the two DEM products footnoted in the paper, ALOS AW3D30 and SRTM 1 Arc-Second Global (USGS doi:10.5066/F7PR7TFT), both global.

### 5.3 Does "task-level comparability" survive inspection? No

Memo §1.4 prefers "tasks GeomorPM evaluated (void filling, landform classification)". At the level of *task name*, that is true — void filling is one of its three fine-tuning tasks. At the level of anything you could put in a table, it fails on every axis:

| | GeomorPM | DEMAE plan |
|---|---|---|
| Source DEM | ALOS AW3D30 / SRTM 1-arcsec | Copernicus GLO-30 |
| Grid | **EPSG:3395 (World Mercator)**, 30/90/120/240 *projected* m | geographic, native GLO-30 |
| True ground GSD of the "30 m" grid | **~22–24 m** (23.8 m at 37.5° N, 22.2 m at 42.5° N) | 30 m |
| Tile | 224 × 224 px, `int16`, nodata 32767 | 224 × 224 px |
| Extent | five 5° × 5° blocks — three in the western USA (Great Basin, Nevada/Idaho, Wyoming/Colorado Rockies), two in the Tian Shan | global, corpus test geocells |
| Void definition | **fraction of tile voided**, two fixed regimes: `mask-1` 10.6–18.6 %, `mask-2` 40.5–59.6 %; 10 pre-generated PNG masks each; irregular, many disconnected components | **void size buckets**, one void per tile per instance |
| Metric | MAE over void pixels only | MAE, RMSE, NMAD, p95, derivatives, hydrology |
| Normalisation | global fixed affine `(h + 1000) / 10000`, clipped to [−1000, 9000] m | per-tile, non-void pixels only |

Three consequences worth stating plainly:

1. **There is no bucket structure to match.** GeomorPM's void difficulty axis is *percentage of tile occluded*, in two coarse regimes, with masks that are scattered multi-component patterns rather than one contiguous void. Plan §3.2's size buckets have no counterpart, and neither does its "one void per tile" rule. `dem_VF_vqmae_finetune.py`'s `main()` hardcodes `mask_mode = 'mask-2'`, i.e. the published void-filling result is the 40–60 %-occluded regime — roughly plan §3.2's large bucket in area terms, and well past the plan's own 40 % cap.
2. **The "30 m" grid is not 30 m.** Because the rasters are in World Mercator, the 30 m pixel size is a projected metre, and at 35–45° latitude the true ground sample distance is about 22–24 m. Any metres-per-pixel comparison to a GLO-30 benchmark is off by ~25 % before anything else is considered. *(This is arithmetic on the Mercator scale factor sec(φ), not a claim from the paper.)*
3. **Its normalisation, unlike Diff-DEM's, does not leak.** `(h + 1000) / 10000` is a fixed global affine with no dependence on the tile, let alone on the void interior. On the specific axis that disqualifies Diff-DEM (§3.5), GeomorPM's protocol is clean. That makes the *inability* to read its numbers the only thing blocking a comparison — which is exactly the case §5.4 is about.

One incidental note, flagged as **unverified**: the last two lines of `dem_VF_vqmae_finetune.py` are trailing comments reading `# 398.474217M 6.53` and `# 398.474217M 0.002108982764184475`. 398.47 M is the model's parameter count. If the second figure is a test loss in the code's own metric and units, it converts to ≈ 21.1 m MAE. It is a comment in a source file, not a published number, it cannot be matched to any table, and it should not be cited as GeomorPM's result. It is recorded only because it is the sole quantitative signal that exists outside the paywall.

### 5.4 The recommendation this produces: an A10 arm

**GeomorPM released its weights, its fine-tuning tiles and its void masks under CC BY 4.0.** That is a cleaner licence than Diff-DEM's (which has none, §3.6) and a materially closer regime than Diff-DEM's (224 px tiles at ~22–24 m against DEMAE's 224 px at 30 m, versus Diff-DEM's 256 px at 10 m max-pooled to 20 m). It is also the *most relevant possible comparator for DEMAE specifically*: a DEM-only self-supervised pretrained model, fine-tuned to void filling, which is the thing DEMAE is.

So the plan should add a tenth arm, alongside A8:

> **A10 — GeomorPM released checkpoint, cross-eval (transfer).** Run `vqmae.ckpt` + their VF fine-tune head on their released areas and masks, and run the DEMAE arms on the same tiles and masks, scored by *our* function. Inference only.

Four things must be said in the same breath: (a) GeomorPM is **398 M parameters** against DEMAE-S's 22 M, an 18× capacity gap, which plan §2.3's "report it, and bracket it" policy already covers and which must appear in the table; (b) their architecture is a VQ-GAN f4 encoder–decoder plus a `MaskedAutoencoderViT(img_size=56, patch_size=4, embed_dim=1024, depth=24)` operating in a 56 × 56 latent, so it is a two-stage model and not a drop-in ViT; (c) their fine-tune loss combines L1 + Sobel (α = 0.3) + Laplacian (β = 0.6) + LPIPS, which is worth noting beside plan §4.1's L1-plus-slope loss as independent convergence on the same idea; (d) their five areas are rugged (Great Basin, Rockies, Tian Shan) and are **not** a representative global sample, so it is a transfer row, labelled as such, exactly like A8.

**This is the closest thing to a real external comparison that exists for this benchmark, and it still is not a number we can cite — it is a number we would compute.** That is the whole answer to the ticket in one sentence.

### 5.5 Landform classification (for map #117 and Task 2)

Out of scope here, but recorded since it was in the same deposit and Task 2 will need it: **the landform label rasters and split files are absent from the Figshare deposit**, despite the README claiming all three tasks are covered. From `dem_LF_vqmae_finetune.py`, the head is `nn.Conv2d(128, 5, kernel_size=1)` — **five classes** — labels decoded from a digit-coded raster (`label = (v // 10) % 10`, class 5 merged into class 4, values > 5 ignored), metrics **pixel accuracy and mean IoU**, with an 18-pixel border excluded and focal loss (α = 1, γ = 2). The source landform map could not be identified. Memo §3.1's "it is GeomorPM-comparable" claim for Task 2 needs its own check, and this note is not it.

## 6. Verifying the plan's two citation-backed design claims

### 6.1 GrounDiff (arXiv 2511.10391) — the plan's characterisation is wrong on two counts

Dhaouadi, O., Meier, J., Kaiser, J., Cremers, D., "GrounDiff: Diffusion-Based Ground Surface Generation from Digital Surface Models", arXiv:[2511.10391](https://arxiv.org/abs/2511.10391), submitted 13 Nov 2025.

Plan §2.4.4 says:

> "The head predicts a **residual over the A1 fill**, not raw elevation. Residual formulation is current best practice in the DEM-correction literature (GrounDiff, FathomDEM-style pipelines)…"

What GrounDiff actually does:

- **The residual is over the input DSM, not over a classical fill.** §3.2, Eq. 4: given corrupted DTM g_t, DSM s and diffusion timestep t, "the model outputs residual correction r̂ and per-pixel confidence logits ℓ", combined by the gate of Eq. 5, `G(r̂,ℓ,s) = σ(ℓ)⊙s + (1−σ(ℓ))⊙(s−r̂)`. The residual r̂ *is the nDSM* — the normalised height of non-ground structures above ground — and the DTM is `s − r̂`. The base surface is the observed DSM. There is no classical filter in the loop, and the paper's framing of the alternative is regression to absolute height, not regression over a filter's output: "Regression formulations directly predict terrain heights, avoiding interpolation bottlenecks."
- **GrounDiff does not do void filling.** Its task is DSM→DTM ground extraction from complete rasters; its benchmarks are ALS2DTM (DALES, NB), USGS/OpenTopography tiles (SU-I/II/III, RT, KW) and GeRoD; its metrics are RMSE, MAE, Type I / Type II / total error rates in %, MED and MAD. Gap filling appears nowhere. PrioStitch is a tiling strategy, not an inpainter.
- **It does not claim residual prediction is best practice.** No such assertion appears. What it offers instead is an ablation, Table 4: "Second, predicting residuals (nDSM) instead of absolute elevations improves accuracy by 17%".

**The correction to record:** the plan's design decision is sound, and GrounDiff's ablation is *better* evidence for it than an appeal to best practice would be — a measured 17% is worth more than a citation to convention. But the sentence as written attributes to GrounDiff (a) a claim it does not make, and (b) a base surface it does not use. Residual-over-a-*classical-fill* is a different object from residual-over-the-*input*, and in void filling the distinction is not cosmetic: inside a void there is no input to be residual to, which is exactly why the plan has to manufacture a base with A1. GrounDiff never faces that problem.

### 6.2 FathomDEM (ERL 2025) — the stratification claim is correct

Uhe, P., Lucas, C., Hawker, L., Brine, M., Wilkinson, H., Cooper, A., Saoulis, A. A., Savage, J., Sampson, C., "FathomDEM: an improved global terrain map using a hybrid vision transformer model", *Environmental Research Letters* 20(3), 034002, 11 Feb 2025, doi:[10.1088/1748-9326/ada972](https://doi.org/10.1088/1748-9326/ada972). Open access; full text read.

**Stratification: confirmed, and strongly.** §3.1:

> "We further categorize our evaluation metrics by 23 masks (supplementary table S4) that categorize a range of landscapes and landcover types. Broadly, we use masks that delineate buildings, trees, bare areas, floodplains, and terrain slope."

Table 1 gives ME / MAE / RMSE per mask against LiDAR; Table 2 repeats the same mask set over the low-elevation coastal zone. The paper reads its own strata for meaning rather than just tabulating them —

> "The worst accuracy is in steep slope areas, which aligns with other assessments of DEMs that always find DEM accuracy to be worst in areas of steep slopes"

— and uses a *stratum where a method should do nothing* as a diagnostic: "errors for the Bare class are similar between COPDEM and FathomDEM, showing that our model was able to reliably identify locations which did not need bias removed, in contrast to FABDEM which has a higher RMSE for Bare pixels." That is a pattern plan §5 should copy directly: the flattest ruggedness stratum is where A2/A3/A4 must *not* damage the input, and reporting it is how you show the model knows when to leave well alone.

Also relevant, and not currently used by the plan: FathomDEM's metric set is **ME, MAE, RMSE** — mean error included, explicitly because "ME informs any bias in the data, and is thus useful in determining whether there is a systematic error in our predictions" — plus **slope error and roughness error** (supplementary S1), plus a downstream-task metric (flood-inundation CSI). Plan §5 has MAE, RMSE, NMAD, p95, slope, aspect, curvature, hydrology, hillshade SSIM. **It does not have ME.** For a residual-over-A1 head, mean error is the cheapest possible check for a systematic over- or under-fill and it costs nothing to add.

**On the residual, FathomDEM is closer to the plan than GrounDiff is, but still not the same object.** §2.1: "The model outputs a 224 × 224 image representing the per-pixel error in the COPDEM values compared to a reference DTM." That is a residual over the input DEM. The base is an observed surface, not an interpolated fill. Worth noting for honesty: FathomDEM's *second* stage is not residual — "The refinement model targets the absolute DEM elevations instead of the error in COPDEM" — so even the paper the plan leans on switches away from the residual formulation for its refinement pass.

(Incidental but useful: FathomDEM trains on **224 × 224 patches at 30 m**, "∼6 km × 6 km", chosen as "a good balance between capturing context from the surrounding terrain … and computational efficiency". That is DEMAE's exact tile geometry, independently arrived at, and is worth one sentence in the writeup.)

### 6.3 What the plan should cite instead

Residual-over-a-**base-fill** is not new in the DEM-correction literature; it is the standing convention in the **void-filling** literature specifically, and predates both cited papers by ~20 years.

- **Delta Surface Fill (DSF)**: Grohman, G., Kroenung, G., Strebeck, J., "Filling SRTM Voids: The Delta Surface Fill Method", *Photogrammetric Engineering & Remote Sensing* 72(3), 213–216 (2006). *(Not reached in full text — no open copy located; characterised below via Luedeling et al. 2007, who describe it in detail and are open.)*
- **TIN delta surface**: Luedeling, E., Siebert, S., Buerkert, A., "Filling the voids in the SRTM elevation model — A TIN-based delta surface approach", *ISPRS Journal of Photogrammetry and Remote Sensing* 62(4), 283–294 (2007), doi:[10.1016/j.isprsjprs.2007.05.004](https://doi.org/10.1016/j.isprsjprs.2007.05.004). Author-hosted copy at uni-kassel.de, read in full.

Luedeling et al. describe DSF (§1) as exactly the plan's construction:

> "A more promising approach is the delta surface fill method, proposed by Grohman et al. (2006). The delta surface in this method is used to remove the vertical bias of the auxiliary DEM. For the central area of a void, the delta surface is assigned a mean value representing the mean vertical difference between the SRTM and the fill surface."

and their own method (§3.1) is a residual added to a base surface spanning the void:

> "Subtracting the Russian base surface from the Russian elevation model then yields the relative elevation of the Russian model above the base surface spanning the void. Adding this difference surface to the SRTM base surface results in an elevation model that accounts for the varying vertical bias within the void and can seamlessly be merged with the SRTM model"

Substituting "A1 spline fill" for "SRTM base surface" and "learned head output" for "difference surface" gives plan §2.4.4 verbatim. **This is the citation §2.4.4 wants** — a void-filling paper, with a base surface interpolated across the void, and a residual added on top — and it makes the plan's decision look like continuity with 20 years of operational practice rather than a borrowing from an adjacent task.

The same paper also settles §6.2's stratification point better than FathomDEM does, because it is a void-filling paper. Its **Table 3** — "Mean values and standard deviation of the ∆GPS values, according to steepness of slope, for the four surfaces" — reports error against 187 500 differential-GPS points, **inside vs outside the void**, in four slope classes (0–10°, 10–20°, 20–30°, 30–60°). Its headline result is a stratification result: std. dev. of ∆GPS inside voids rises from 58.8 m (low slope) to 106.4 m (very steep) for the delta-filled product. Plan §5's "A method that wins on plains and loses in mountains must be visible as such" is this table, in 2007.

## 7. Convention survey — what the field reports

### 7.1 The canonical protocol is Reuter et al. 2007, and it publishes no metres

Reuter, H. I., Nelson, A., Jarvis, A., "An evaluation of void-filling interpolation methods for SRTM data", *Int. J. Geogr. Inf. Sci.* 21(9), 983–1008 (2007), doi:[10.1080/13658810601169899](https://doi.org/10.1080/13658810601169899). 883 citations; this is the study that produced CGIAR-CSI's hole-filled SRTM v4. Taylor & Francis returns HTTP 403 and Semantic Scholar reports the abstract itself "elided by the publisher" — but **the authors' own manuscript is hosted by CGIAR-CSI** at `srtm.csi.cgiar.org/wp-content/uploads/files/Reuteretal2007.pdf` (49 pp., tables included), which is primary and was read in full. Its internal pagination is 1–49, not 983–1008.

**It is the direct precedent for plan §3.2 and §5, and it validates both.** From the abstract:

> "Based on a sample of 1,304 artificial but realistic voids across six terrain types and eight void size classes, we found that the choice of void filling algorithm is dependent on both the size and terrain type of the void."

- **Void-size buckets: eight classes**, cut on the natural log of void pixel count, (min, mean, max) pixels: A (1, 10, 25), B (26, 50, 80), C (81, 120, 140), D (141, 240, 400), E (401, 600, 800), F (801, 1100, 2500), G (2501, 4000, 8000), H (8001, 10000, 1267052).
- **Terrain: six units** aggregated from Meybeck et al. (2001)'s 15-class typology, built from mean elevation plus "relief roughness … the range of SRTM30 elevation values in the pixel divided by half the pixel length" — PLAINS, PLATEAUS, HIGH PLATEAUS, HILLS, MOUNTAINS, HIGH MOUNTAINS.
- **Results are reported on the cross**, 6 × 8 = 48 void typologies (§4.3).
- **Error is computed over the whole void area**, not at scattered control points: "We choose an evaluation based on the total area of the void similar to Grohman et al. (2006) in contrast to a selected number of GCP".
- **The masks are real void shapes transplanted to matching terrain.** 15 real African voids per terrain unit, drawn in proportion to the size distribution, then duplicated and relocated to non-overlapping same-terrain-unit neighbours so the underlying real SRTM is the withheld truth — 1,304 voids in total.

**That last point deserves emphasis: plan §3.3's real-SRTM-void transplant protocol is Reuter et al.'s protocol, independently re-derived.** The plan should cite it. It is also the answer to anyone who calls the transplant a hack — it is what the field's canonical void-filling evaluation did in 2007, for the same reason (truth under a real void was never observed).

**And here is the fact that closes the comparability question.** Reuter et al. report **mean *rank* of RMSE, not RMSE in metres.** Each filled patch is "ranked from 1 (lowest RMSE) to 7 — or 21 for a comparison across all variations — (highest RMSE)", and Tables 3, 4 and 5 tabulate mean ranks. **No absolute RMSE in metres appears anywhere in the paper.** The paper even trips on its own convention in §4.1 — "Differences between VF methods in RMSE can triple (e.g. see first row differences between KR (6.04 and SP 17.40)" — where 6.04 and 17.40 are mean ranks, not metres. So the single most-cited quantitative evaluation of DEM void filling publishes nothing this benchmark could stand beside, by construction.

Two more things it supplies:

- **Mean error was in the canonical metric set from the start.** They computed RMSE, ρ, mean difference, sum difference and difference SD, and then added ME and the error standard deviation S on Fisher & Tate (2006)'s recommendation that "the RMSE is not necessarily a good estimator of the error". They report RMSE only because "ME showed diverse rankings of different algorithms" — a ranking-scheme problem, not a reason to omit the quantity. Third independent argument for §7.4's add-ME recommendation.
- **Their stated limitation is plan §5's design.** Verbatim: "We are aware that a global statistic was used to compare the filled voids against the truth surface, rather than evaluation methods which take the spatial pattern of errors into account or which identify the different factors which led to that error." Plan §5's terrain-derivative, hydrology and hillshade metrics are exactly the spatial-pattern evaluation Reuter et al. said was missing. That is a sentence worth putting in the writeup.

### 7.2 Is error-in-metres-inside-the-void, bucketed by void size, the normal shape?

**Yes, and it is Diff-DEM's own shape.** Evidence, all primary:

| Source | Scope of the error metric | Buckets |
|---|---|---|
| Reuter et al. Tables 3–5 | RMSE over the whole void area — but reported as mean **rank**, never metres | 8 log-spaced size classes × 6 terrain units |
| Diff-DEM `tif_metric.py` | MAE, RMSE strictly on `mask > 0`, in file units (metres) | three void-width buckets, `# Table II` block |
| ET-SDE (arXiv 2407.01908) Tables 5–6 | RMSE (scope not stated) | six void configurations, three of them Diff-DEM's width buckets |
| Gavriil et al. Table I | MSE (m²) | urban / rural, **not** void size |
| Luedeling et al. Table 3 | mean and std. dev. of ∆GPS, in metres | **inside vs outside the void**, × four slope classes |

So plan §5's core reporting shape is conventional. Two refinements the sources suggest:

- **Report the inside/outside pair, not just inside.** Luedeling et al.'s table is built around it, and it is the direct measure of whether a fill damages the surrounding surface — the seam problem plan §4.2 handles with cross-fading and shifted encodings. Plan §5 scores "only inside the void" and so cannot see a seam at all. A 2 px collar around the void, scored separately, costs nothing and is the standard place this failure shows up. (Plan §5 already computes drainage correlation "inside and 2 px around the void", so the collar exists in the hydrology metric and just needs extending to elevation and slope.)
- **ET-SDE's tables do not state whether RMSE is void-only or whole-patch.** That ambiguity is exactly why an external number cannot be lifted into our table, and it is worth one sentence in the writeup as an example of why.

### 7.3 What else is standard, and what plan §5 is missing

**Mean error (ME) — missing, and it should not be.** FathomDEM reports ME/MAE/RMSE and gives the reason: ME "informs any bias in the data, and is thus useful in determining whether there is a systematic error in our predictions". Luedeling et al. report mean and standard deviation of ∆GPS, which is the same pair under different names, and their headline SRTM finding is a *bias* finding (+84.3 m mean over the void areas), invisible to MAE or RMSE. For a head that predicts a residual over an A1 fill, a systematic non-zero mean is the specific failure mode; MAE and RMSE both hide it. **Add ME to plan §5.**

**Image-quality metrics on raw elevation — omitted, and the omission is defensible but should be deliberate.** The learned void-filling literature reports them as a matter of course: Diff-DEM prints MS-SSIM and PSNR beside MAE/RMSE; ET-SDE's every table is PSNR / SSIM / RMSE. Plan §5 has hillshade SSIM (a better metric for terrain, and correctly justified) but nothing a reader of that literature will recognise. Two honest considerations pull opposite ways: those metrics as implemented are weak — §3.4 shows Diff-DEM's PSNR uses a per-image, data-dependent `data_range`, which makes averaging across tiles meaningless — but a table with none of the columns that community expects will read as evasive. **Suggested resolution:** compute MS-SSIM and PSNR on raw elevation as a clearly-labelled secondary block with a *fixed, documented* data range, state that a fixed range is why our values are not comparable to published ones, and keep hillshade SSIM as the headline perceptual metric.

**Terrain-derivative fidelity — established, but a house style rather than a universal convention, and plan §5 should know which.** Two strong precedents, and one instructive counter-example.

- **FathomDEM** analyses slope and roughness in supplementary §S1 — Table S1 ("As per Table 1, except for errors in surface slope") and Table S2 (roughness), across the same 23 masks including nine slope bins. Its stated reason is the plan's reason: "As derivatives of elevation that also considers neighboring pixels, these measurements are more sensitive to nonsystematic DEM elevation error than elevation alone." It reads the *sign* diagnostically — "Slightly negative mean error indicates that FathomDEM is marginally smoother than the reference terrain… COPDEM has a positive mean error, indicating it is too rough" — which is a use of ME (§7.4) that MAE and RMSE cannot support.
- **Bielski et al. (2024)** goes further, making slope and roughness co-equal ranking criteria rather than an appendix: "We used a suite of criteria relating to the differences in the elevation, slope, and roughness distributions compared to reference DEMs", justified because "These parameters were chosen because they measure different aspects of the DEM and the underlying terrain, and are not closely correlated." Its 15 criteria are the cross of {elevation, slope, roughness} × {STD, AVD, RMSE, MAE, LE90}. Bielski, C. *et al.*, "Novel approach for ranking DEMs: Copernicus DEM improves one arc second open global topography", *IEEE Trans. Geosci. Remote Sens.* 62, 1–22 (2024), doi:[10.1109/TGRS.2024.3368015]; IEEE paywalled, read via the author manuscript arXiv:[2302.08425](https://arxiv.org/abs/2302.08425)v6, so table numbers may differ from the version of record.
- **The counter-example matters for framing.** FABDEM — Hawker, L. *et al.*, "A 30 m global map of elevation with forests and buildings removed", *Environ. Res. Lett.* 17 024016 (2022), doi:[10.1088/1748-9326/ac4d4f](https://doi.org/10.1088/1748-9326/ac4d4f) — reports **elevation only** in both its tables. FathomDEM (2025) *added* derivative reporting where its own direct predecessor had not. So plan §5's derivative suite is a defensible best practice with recent precedent, **not** something reviewers will already expect. Say it as a strengthening, and justify it, rather than assuming it reads as standard.

**Downstream-task evaluation — plan §5's hydrology metrics are the recognised move, and there is a precise precedent for each one.** See §7.6.

**Ground/non-ground classification rates (Type I / Type II / total error, in %) — not applicable.** They are GrounDiff's headline metrics and are standard for DSM→DTM filtering, but there is no ground/non-ground decision in void filling. Noted only so nobody re-derives the question.

### 7.4 Robust and tolerance-based statistics — plan §5's NMAD claim, verified

Plan §5 asserts "NMAD (robust spread, standard in DEM QA)" without a citation. **The claim is correct**, and the source is Höhle, J. and Höhle, M., "Accuracy assessment of digital elevation models by means of robust statistical methods", *ISPRS Journal of Photogrammetry and Remote Sensing* 64(4), 398–406 (2009), doi:[10.1016/j.isprsjprs.2009.02.003](https://doi.org/10.1016/j.isprsjprs.2009.02.003). Green open access, author's institutional copy at `vbn.aau.dk/ws/files/17874509/sdarticle.pdf`, read in full.

The paper's whole argument is that DEM error distributions are heavy-tailed, so the Gaussian measures are the wrong ones. Its **Table 1** ("accuracy measures for the normal distribution") is RMSE, mean error, standard deviation. Its **Table 2** ("Proposed accuracy measures for DEMs") replaces them with four:

| Accuracy measure | Error type | Expression |
|---|---|---|
| Median (50% quantile) | Δh | Q̂_Δh(0.5) = m_Δh |
| Normalized median absolute deviation | Δh | NMAD = 1.4826 · median_j(\|Δh_j − m_Δh\|) *(Eq. 2)* |
| 68.3% quantile | \|Δh\| | Q̂_\|Δh\|(0.683) |
| 95% quantile | \|Δh\| | Q̂_\|Δh\|(0.95) |

The abstract's conclusion, verbatim:

> "It is concluded that measures such as median, normalized median absolute deviation, and sample quantiles should be used in the accuracy assessment of such DEMs."

**Plan §5 already has two of the four** — NMAD and "the 95th percentile absolute error (tail behavior matters for hazards)" — and the paper gives that second one a better justification than the plan does:

> "the 95% sample quantile of |Δh| literally means that 95% of the errors have a magnitude within the interval [0, Q̂_|Δh|(0.95)]. The remaining 5% of the errors can be of any value making the measure robust against up to 5% blunders."

**What plan §5 is missing from Table 2 is the median error**, the robust counterpart of the mean error §7.3 also asks for. Höhle & Höhle prefer it to ME explicitly: it "is less sensitive to outliers in the data than the mean error and provides a better distributional summary for skew distributions". So the combined recommendation from §7.3 and here is: **add both mean error and median error** — ME because FathomDEM's bias argument is the right one for a residual head, median error because it is the one that survives the heavy tails this paper is about. Together they cost two lines and give the "is the fill systematically high or low?" question an answer that MAE, RMSE, NMAD and p95 all structurally cannot give.

Two further notes from the same source, both directly useful:

- **Non-parametric accuracy measures are already in a standard, for exactly the difficult-terrain case.** §1: "So far, only the Lidar committee of the American Society for Photogrammetry and Remote Sensing deals with it and requires non-parametric accuracy measures for non-open terrain (ASPRS Lidar Committee, 2004)." That is the field's own precedent for reporting robust measures specifically where terrain is hard — which is where plan §5's rugged strata live.
- **The paper anticipates stratification.** Its quantile-regression extension "allows e.g. modelling the p-quantile as a function of terrain classes" — i.e. reporting a robust accuracy measure per terrain stratum is the paper's own suggested generalisation, not an invention of plan §5.

### 7.5 There is no single "DEM accuracy standard" — there are two, and they disagree

This is worth knowing before the writeup says "the standard".

**Copernicus mandates LE90 and never uses RMSE.** *Copernicus DEM Product Handbook*, doc `geo1988-copernicusdem-spe-002_producthandbook`, **v5.0, 29.11.2022**, [dataspace.copernicus.eu](https://dataspace.copernicus.eu/sites/default/files/media/files/2024-06/geo1988-copernicusdem-spe-002_producthandbook_i5.0.pdf). Tables 1 and 2 state the requirement for GLO-30 — **the plan's own source product** — as "< 4m (90% linear error)" absolute vertical, "< 2m (slope ≤20%) / < 4m (slope > 20%)" relative, "< 6m (90% circular error)" horizontal. §2.1: the error "is expressed as linear error at a 90 percent confidence level". And the relative accuracy metric *is* a percentage-within-tolerance by construction (§2.2): "The relative, vertical accuracy value is provided in percent and represents the accumulated area of fulfilled relative, vertical accuracy (2m for slope ≤ 20%; 4m for slope > 20%) compared to the overall area of a geotile."

Note two things. First, GLO-30's own spec is **slope-stratified** (a different tolerance above and below 20 % slope) — so plan §5's ruggedness stratification is what the source product's own quality statement does. Second, Table 12 tabulates LE90 by tolerance band across geotiles (< 2 m: 64.7 % of landmass; 2–5 m: 18.6 %; 5–10 m: 4.6 %; > 10 m: 0.8 %), which is the "percentage of pixels within a tolerance" shape the ticket asked about, done on area.

**ASPRS mandates RMSE and never uses LE90.** ASPRS *Positional Accuracy Standards for Digital Geospatial Data*, Edition 1 v1.0 (2014) and **Edition 2 v1.0.0 (2023)**. Edition 1 §7.1 required "RMSEz statistics in non-vegetated terrain and 95th percentile statistics in vegetated terrain" — i.e. land-cover-stratified reporting (NVA vs VVA), which is again the stratification argument. **Edition 2 changed course:** §7.4, "Vertical accuracy is to be expressed as RMSEV in both non-vegetated terrain and vegetated terrain", with VVA reported "As found" and no pass/fail. The official change list (*PERS* 89(10), Oct 2023, doi:10.14358/PERS.89.10.589) records "Eliminated references to the 95% confidence level as an accuracy measure" and "Removed the pass/fail requirement for Vegetated Vertical Accuracy".

**Two things follow for plan §5.** (a) **Edition 2 §7.15 explicitly requires median error**: "related statistical quantities should be computed and reported, including the following quantities: Residual errors at each checkpoint • Maximum error • Minimum error • Mean error • **Median error** • Standard deviation • RMSE". That is a fourth independent argument for §7.4's recommendation. (b) **Honest caveat on the 95th percentile:** plan §5 has it and Höhle & Höhle endorse it, but ASPRS Edition 2 deliberately dropped 95 %-level reporting as a source of user confusion. Keep p95 — the hazard-tail argument is good and the robust-statistics literature backs it — but do not describe it as what the standards require, because one of the two standards just removed it.

*(Both standards documents were read as verbatim copies hosted by ASPRS Florida Region and AAGS respectively, because asprs.org and my.asprs.org return 403; the Edition 2 change list was confirmed from the official PERS article, which does load. Nothing above depends on the mirrors being byte-identical to the ASPRS originals, but the mirroring should be stated.)*

### 7.6 Hydrological evaluation is established, and plan §5's three metrics each have a named precedent

Plan §5 proposes D8 flow accumulation, log-flow correlation inside and 2 px around the void, and stream-network IoU at a fixed accumulation threshold. All three are recognised moves, and the canonical example is itself a *void-filled-DEM* case.

**MERIT Hydro** — Yamazaki, D. *et al.*, "MERIT Hydro: A High-Resolution Global Hydrography Map Based on Latest Topography Dataset", *Water Resources Research* 55, 5053–5073 (2019), doi:[10.1029/2019WR024873](https://doi.org/10.1029/2019WR024873). (Wiley returns HTTP 402; the CC BY-NC-ND version of record was read from the University of Bristol repository.) MERIT Hydro is built on MERIT DEM — a multi-error-*removed* DEM — and its §3.3 validation is entirely drainage-based, in four tests that map almost one-to-one onto plan §5:

1. **Flow accumulation area against reference hydrography** (Figure 9a) — "For each pixel with >1,000-km² flow accumulation area in the new hydrography map, the flow accumulation area of HydroSHEDS pixels within 2-arc min distance (i.e., 5 × 5 pixels) was checked to find the minimum relative error." Plan §5's log-flow correlation is the same quantity, correlated rather than differenced.
2. **Basin-shape agreement via Critical Success Index** (Figure 9b) — "CSI = N∩H / N∪H … The CSI is 1 when the shape of the considering river basin is exactly same in the two data sets, while CSI is zero when there is no overlap." That is plan §5's IoU, under its hydrology name. Their own caveat is worth inheriting: "As the shapes of river basins are highly affected by the treatment of inland depressions and by the specific land/ocean mask used, it is difficult to compare the accuracy of the river basin shape only using the CSI."
3. **Stream centreline location against GRWL** (Figure 10) — error normalised by channel width.
4. **Flow accumulation against gauge-reported drainage area** (Figure 11) — 5,795 GRDC stations.

**Bielski et al. §4.3** does the same thing in the shape plan §5 needs — as a *check on a ranking*. SAGA's "Sink Removal" and "Channel network" modules on reference DTMs, rasterised, filtered to `PLOT_ORDER>1`; Table 5 is a 2 × 2 contingency table (correct channel / underprediction / overprediction / correct inter-channel) reduced to a hit percentage, "Correct channel / (Correct channel + Underprediction)". Their conclusion is the one to quote in the writeup: "This hydrology experiment verifies the applicability of the global wine contest rankings beyond having better RMSE or LE90 statistical values." Their flagged weakness — "Flat areas are a known problem in channel extraction" — is why plan §5 should report the hydrology metrics *per ruggedness stratum* rather than pooled, and expect them to be uninformative in the flattest one.

### 7.7 Is there a shared, open benchmark for DEM void filling?

**The clearest single piece of evidence is what the one paper that tried to compare against Diff-DEM actually did.** Zhang, Zuo & Li, "Efficient Terrain Stochastic Differential Equations for Multipurpose Digital Elevation Model Restoration", arXiv:[2407.01908](https://arxiv.org/abs/2407.01908) (v2, 24 Nov 2024; arXiv preprint only, no journal reference in the arXiv metadata as of 2026-08-06), §4.3. It selects "the state-of-the-art void filling method Diff-DEM (Lo and Peters, 2024)" as its benchmark — and then **does not use Diff-DEM's benchmark.** It builds its own: 2 m Pyrenees DEMs in 96 × 96 patches and ASTER-derived Mount Tai DEMs in 256 × 256 patches, with its own random-walk void generator (Table 3) and its own rectangular masks (Table 4), and re-runs Diff-DEM's released checkpoint on them at various sampling steps.

Its Table 6 is the nearest published thing to DEMAE's regime — 256 px patches, rectangular voids on Diff-DEM's own bucket boundaries, on data the paper says is "obtained from the Advanced Spaceborne Thermal Emission and Reflection Radiometer (ASTER)". *(The Mount Tai GSD is never stated; ASTER GDEM is a 1-arcsecond product, so ~30 m is an inference, marked, not a reported figure.)*

| | 64 × 64 | | 64 × 96 | | 96 × 128 | |
|---|---|---|---|---|---|---|
| | Diff-DEM | ET-SDE | Diff-DEM | ET-SDE | Diff-DEM | ET-SDE |
| PSNR | 15.62 | 24.21 | 14.92 | 22.97 | 14.37 | 21.81 |
| SSIM | 0.47 | 0.81 | 0.43 | 0.77 | 0.42 | 0.73 |
| RMSE | 8.97 | 7.02 | 9.21 | 7.36 | 12.04 | 7.94 |

**These numbers cannot be borrowed either**, and the reasons are worth listing because they are the general reasons: the column headings (`64 × 64`, `64 × 96`, `96 × 128`) do not match the paper's own Table 4 notation (`V-64-96`, `V-96-128`, `V-128-160`), so it is not determinable which masks produced which column; the RMSE scope (void-only or whole-patch) is never stated; both models were retrained on Mount Tai, so the values measure a locality, not a method; and the paper describes Diff-DEM as showing "great capability in void-filling in urban areas" and refers to "the urban area introduced by Diff-DEM", which mischaracterises a benchmark that is 10 m Norwegian landscape (the urban 2 m tiles are Gavriil's, §4). *(Inference, marked: a paper that garbles the provenance of the benchmark it is comparing against is itself evidence that no shared, well-specified benchmark exists — if there were one, this would be harder to get wrong.)*

So the observed pattern across the four learned void-filling works reached here is: **every one builds its own tiles, its own masks and its own scoring script.** Gavriil's 50 + 50 evaluation DEMs are the only test set reused by a second paper (Diff-DEM's Table I), and even there the second paper re-scored the baselines under different metrics rather than quoting the first paper's numbers.

**Searched exhaustively, and the answer is no.** No submission server, no held-out test set with an official protocol, no ranking that independent groups report against.

- **Papers with Code no longer exists** — `paperswithcode.com/search?q=DEM+void+filling` 302-redirects to `huggingface.co/papers/trending`. There is no PwC leaderboard for this task, or for anything.
- **ISPRS benchmarks** (`isprs.org/resources/datasets/benchmarks/`; the older `/education/benchmarks.aspx` is a 404) lists AGI2P, RTK-SLAM, WHU-TLS, H3D, UAVid, dense image matching, urban classification, aerial image matching. None concerns elevation gap filling; the nearest relatives are DSM-generation-from-stereo, a different task.
- **IEEE GRSS Data Fusion Contest**, every year 2006–2026: no year has been about void filling. The nearest are DFC2019 (single-view height, stereo disparity) and DFC2023 Track 2 (building height) — height *inference from imagery*, not completion of an existing raster.
- **GitHub** returns **four** repositories total for "DEM void filling" (`kylelo/Diff-DEM` 20 stars, `sayantanonfire/CRPAG-Code-…` 4, `jkhansell/TerraGen` 1, `Haranvs/HMA-DEM-Void-Filling` 0); the topic `dem-void-filling` has zero. **Zenodo**: 5 hits, none a benchmark. **Hugging Face** datasets: zero for "DEM void", "DEM digital elevation" and "void filling". **EarthNets**, **PANGAEA**, **GEO-Bench**: classification, detection, segmentation, change detection — no terrain completion. **CodaLab / Codabench / Kaggle / EvalAI**: nothing.

**The strongest positive evidence for the negative is that five recent learned void-filling papers use five different corpora:** Gavriil 2019 (Norway 10 m + 2 m); Diff-DEM 2024 (Norway again, but a different and larger curation); GeomorPM 2024 (AW3D30/SRTM over the western USA and Tian Shan, World Mercator, fraction-voided masks); ET-SDE 2024 (Pyrenees 2 m and Mount Tai, author-synthesised random-walk masks); and Dfilled 2025 (arXiv:[2501.15440](https://arxiv.org/abs/2501.15440) — 0.5 m Pléiades-1B stereo over Ho Chi Minh City and Dushanbe plus Swisstopo lidar, test on Dushanbe only, metrics RMSE/NMAD/MedAE, and the Pléiades licence makes the test set non-redistributable). **Dfilled had to retrain Diff-DEM and RSAGAN on its own data to compare at all** — which is precisely the move §5.4 recommends for A10, and the field's own admission that there is nothing to compare against otherwise.

*(Two incidental confirmations for plan §5: Dfilled's metric set is RMSE / NMAD / median absolute error — NMAD in a void-filling paper, and the median-family measure §7.4 recommends, in the most recent work in the area.)*

**Implication:** a common evaluation surface for DEM void filling has to be built, not adopted. The Gavriil/Diff-DEM Norway pair is the only asset with any prior cross-paper reuse, and GeomorPM's Figshare deposit (§5.2) is the only one under a licence that makes reuse comfortable.

## 8. The honest framing

The ticket asks for one of two answers. This is the second one.

**"Internally controlled comparison, externally incomparable" is the true description of the headline table, and it should be stated in the plan rather than discovered by a reviewer.**

The argument, assembled:

- **No shared benchmark exists.** DEM void filling has no ImageNet, no Cityscapes, no leaderboard, no held-out test set that independent groups report against (§7.7). Every paper builds its own tiles, its own masks and its own scoring script, and — as §3.3–§3.5 shows for the one case where the scoring script is public — the script is a substantial part of the measurement.
- **The canonical evaluation does not publish metres at all.** Reuter et al. (2007) reports mean *rank* of RMSE (§7.1). The most-cited quantitative study of this task contains, by construction, no number anyone could stand beside.
- **The nearest thing to a shared test set is Gavriil's 50+50 tiles**, and it is shared by exactly two papers, in different metrics (§4 vs §3.4), one of which is paywalled.
- **Even the metric names collide misleadingly.** "RMSE in metres inside the void" appears in Diff-DEM, in ET-SDE and in plan §5, and means three different things: void-only at 10 m after a max-pool/upsample round trip with a truth-fitted rescale; unspecified scope on Pyrenees/Mount Tai; and void-only at 30 m with non-void-derived normalisation. GeomorPM's code computes something it calls RMSE that is arithmetically MAE (§5.3). A reader who sees four numbers in one column will assume they are the same measurement. They are not.
- **The field itself behaves as if this is true.** Dfilled (2025), the most recent work found, retrained both Diff-DEM and RSAGAN on its own data rather than citing their published numbers (§7.7). That is the same move A8 and A10 make, arrived at independently.
- **What survives is what the benchmark was actually built to do.** Plan §0's C1 and C3 are *within-table* claims: identical tiles, identical masks, identical seeds, identical head, only the frozen representation differs. Those claims do not need an external number and are not weakened by the absence of one. C2 (beat classical interpolation) is also internal — A1 is in the table.

So the framing to write into §9 is roughly:

> No public DEM void-filling benchmark exists that independent methods report against; published numbers in this task are each tied to a private tile set, a private mask set and a private scoring script, and are not commensurable with each other, let alone with ours — the field's most-cited evaluation publishes only ranks, and the most recent work retrains its competitors rather than citing them. This table is therefore an internally controlled comparison: every arm sees the same tiles, the same masks, the same seeds and the same head, and every number in it was produced by one scoring function in this repository. The external contact points are the two transfer blocks, §3.4 (Diff-DEM) and the proposed A10 (GeomorPM), where released competitor checkpoints and the DEMAE arms are run on the competitors' own released data and scored by *our* function — comparisons we compute, not numbers we cite.

**Two things this framing buys, and they are worth more than a borrowed column.** First, credibility: PANGAEA-style candor is already plan §6's stated posture, and "we checked, and there is nothing to compare to" is a stronger sentence than a table with an incommensurable row in it. Second, it removes a real failure mode — a reader who spots the de-normalisation leak of §3.5 in a table where our numbers sit next to Diff-DEM's will discount the whole table, not just that row.

**The one qualification.** "Externally incomparable" is a claim about *numbers in a table*, not about the literature. §4's Gavriil result (classical splines beating the GAN on the urban set), §7.1's Reuter protocol, and §6.3's Luedeling stratification are directly relevant prior art and belong in the writeup's prose, as context for C2, for §3.2/§3.3's design, and for §5's stratification respectively.

## 9. Amendments to the plan and the memo

Each of these is a text change, not a design change. The benchmark as designed is sound; what needs fixing is what it claims about the outside world.

**Plan §3.2 — the bucket-matching sentence.** "the two upper buckets match Diff-DEM's published protocol so numbers are directly comparable" should become something like: *the two upper buckets match Diff-DEM's bucket boundaries in pixels, which approximately preserves the fraction of tile occluded (25–62% of tile width there, 29–71% here) but not the ground extent, which is 3× larger at 30 m; the numbers are not directly comparable and the match is a design convenience, not a comparability claim.* Diff-DEM's third bucket (96–128) is omitted and it is worth one clause saying so.

**Plan §3.2 — an internal contradiction, unrelated to comparability but found while checking the arithmetic.** The constraints are "voids fully interior (≥ 8 px from tile edge); void may not cover > 40% of the tile", and the large bucket is 128–160 px on a 224 px tile. A square void satisfies the 40% area cap only up to side √(0.40 × 224²) = **141 px**. So for the rectangle family, the upper half of the large bucket (142–160 px) is unreachable: the two constraints are jointly unsatisfiable. Either the cap rises (Diff-DEM's own top bucket reaches 39.1% of tile area, so 40% was presumably calibrated against a 256 px tile and not re-derived for 224), or the large bucket's definition needs to be in *area* rather than in *side length*, or the bucket top drops to ~140. This must be resolved before the mask lists are frozen in §4.5 step 1, because the mask generator will otherwise either silently reject draws or silently violate a stated constraint.

**Plan §2.4.4 — the residual citation.** Replace "Residual formulation is current best practice in the DEM-correction literature (GrounDiff, FathomDEM-style pipelines)" with an accurate two-part statement: (a) residual-over-an-interpolated-base is the *classical* void-filling convention — Grohman et al.'s delta surface fill (2006) and Luedeling et al.'s TIN delta surface (2007), which is literally "adding this difference surface to the SRTM base surface"; (b) modern learned DEM-correction models predict a residual over the *input* surface — GrounDiff's nDSM over the DSM, FathomDEM's per-pixel error in COPDEM — and GrounDiff reports a 17% accuracy gain for residual over absolute prediction, which is the actual evidence. Note explicitly that neither modern paper does void filling, so the base surface differs, and that this is why A1 has to manufacture one.

**Plan §5 — add mean error and median error; cite Höhle & Höhle for NMAD/p95; cite Luedeling for the stratification.** Every metric in §5 is a magnitude, so a fill that is systematically 3 m high scores identically to one that is 3 m low. FathomDEM's stated reason for ME — it "informs any bias in the data, and is thus useful in determining whether there is a systematic error in our predictions" — applies with extra force to a residual head, and Höhle & Höhle's median error is the heavy-tail-resistant version of the same quantity (§7.3). Two lines. While there, §5's uncited "NMAD (robust spread, standard in DEM QA)" should cite Höhle & Höhle 2009 Table 2, which is where NMAD *and* the 95th-percentile choice both come from — the plan already picked two of that table's four measures without saying so. And §5's stratification sentence should cite Luedeling et al. 2007 Table 3 (void-fill error by slope class, inside vs outside void) alongside FathomDEM Table 1, because the former is the same task.

**Plan §5 — score a collar as well as the void.** §5 computes elevation and derivative metrics "only inside the void", so seam artefacts at the void boundary — the exact thing §4.2's cross-fade and shifted-encoding machinery exists to suppress — are invisible to every headline metric. The hydrology metric already uses "inside and 2 px around the void"; extend the same collar to elevation and slope. Luedeling et al.'s inside/outside pairing (§7.1) is the precedent.

**Plan §5 — borrow FathomDEM's "stratum where nothing should happen" diagnostic.** Its Bare-class result is used to show the model correctly leaves already-good pixels alone. The flattest ruggedness stratum plays that role here and reporting it that way is free.

**Plan §3.5 and §9 — the licence line.** "Diff-DEM's data per their repo terms" names terms that do not exist: the repo has no LICENSE and `gh api` reports `license: null`. The dataset card should record: Diff-DEM repo — no stated licence; the Gavriil-derived subset — CC BY-NC 4.0, "educational and academic research purpose only" per `konstantg/dem-fill`; underlying elevation — Kartverket, CC BY 4.0. Since §3.4's cross-eval only *evaluates* on this data and redistributes nothing, this is a citation-and-attribution obligation rather than a blocker, but it should be written down (cf. P101's finding that unstated licences are the expensive kind).

**Plan §3.4 and §4.5 — A8 execution notes.** Three concrete hazards from §3.6: `--normalize` cannot be disabled from the CLI (edit `__main__`, or better, score everything with *our* function); the committed `evaluation.sh` runs `n_timestep=1` against the README's 512, so it reproduces nothing as-is; and Diff-DEM's harness max-pools 2× and rescales with ground-truth extrema, so **A8 must be scored by our function on our terms in both directions, and the fact that their published numbers were not produced that way must be stated.** Budget note: the third-party ET-SDE evaluation runs Diff-DEM at 128–512 sampling steps, which bounds §7's "A8 diffusion sampling is the slow one" line.

**Plan §1 and §3 — add arm A10, the GeomorPM cross-eval.** Spelled out in §5.4. Inference only; a released CC BY 4.0 checkpoint plus released tiles and masks at 224 px and ~22–24 m; the closest external comparator DEMAE has, because it is the only other DEM-only self-supervised pretrained model with an open void-filling checkpoint. Report the 398 M vs 22 M parameter gap in the same row, label it transfer, and score it with our function on both sides. This is the one place where the answer to #124 is "yes, there is something" — and even there, the number is computed rather than cited.

**Plan §3.3 — cite Reuter et al. 2007 for the real-void transplant.** §3.3's protocol (extract real void geometry, transplant onto clean tiles of matching ruggedness stratum, score against the truth that exists there) is Reuter et al.'s 2007 construction, down to the stratum-matched relocation. It is currently presented as an invention. Citing it costs a line and converts a design that looks improvised into one with the field's canonical evaluation behind it. The same paper's 8-class log-spaced void-size scheme is worth a sentence beside §3.2's three buckets, if only to note that three is a coarsening of eight.

**Plan §5 — expect the hydrology metrics to be uninformative on flat terrain, and say so up front.** Bielski et al. §4.3 flags "Flat areas are a known problem in channel extraction". Reporting drainage metrics per ruggedness stratum (which §5 already does for everything) will surface this, and pre-registering the expectation stops it reading as a null result later.

**Plan §5 — do not claim the derivative suite is standard.** FABDEM (2022) reports elevation only; FathomDEM (2025) added slope and roughness. §7.3's precedents are real but recent. Frame the derivative metrics as a deliberate strengthening with a stated rationale, not as convention.

**Plan §9 and memo §1.4 — the framing.** §9's "lands next to something" and memo §1.4's comparability criterion should both be amended per §8. Memo §1.4 ranked void filling partly *because* it promised external comparability; that reason did not survive, and the honest note is that the task is still the right first choice for the other reasons given (the MAE decoder is natively an inpainter, it is the cheapest task, the money plot is strong) — just not for that one.

## 10. What we checked that did not answer the question

Recorded so it is not re-litigated, and so the shape of the gap is visible:

- **IEEE Xplore document 10535979 / doi:10.1109/LGRS.2024.3403835** — closed access. Semantic Scholar's `openAccessPdf` is `{"url": "", "status": "CLOSED"}`. No arXiv entry. ResearchGate carries a "Request PDF" stub only. **Diff-DEM's Table I and Table II were not read**, and nothing in this note claims to report their values.
- **`kylelo/Diff-DEM` repository** — full tree listed via `gh api repos/kylelo/Diff-DEM/git/trees/main?recursive=1`; 9 top-level entries, no `LICENSE`, no `results/`, no committed numbers. The tables are not in the repo either.
- **Gavriil et al. arXiv 1811.12693** — read in full. Gives Table I (§4). Its own authors call it of "limited insight".
- **`konstantg/dem-fill`** — README read; licence, data provenance, and void size range (64×64 to 128×128) obtained. The 50+50 evaluation DEMs are on Google Drive, not in git.
- **GrounDiff arXiv 2511.10391** — abstract page and full HTML read. Confirms the residual is the nDSM over the DSM and that void filling is not addressed.
- **FathomDEM, ERL 20 034002** — open access; PDF retrieved and read in full. Confirms stratification (Table 1, 23 masks) and the residual-over-COPDEM formulation, and supplies the ME rationale.
- **Luedeling et al. 2007** — author-hosted copy at uni-kassel.de, read in full. Supplies both the delta-surface residual construction and the slope-stratified void-fill evaluation (Table 3).
- **Grohman et al. 2006 (PE&RS 72(3) 213–216)** — **not reached.** No open copy located; PE&RS is not open for that volume. It is characterised here only through Luedeling et al.'s description of it, and that is said explicitly at the point of use (§6.3).
- **ET-SDE arXiv 2407.01908** — read in full. It re-runs Diff-DEM on its own Pyrenees/Mount Tai data rather than reporting Diff-DEM's published numbers, so it does not give a route around the paywall (§7.7).
- **Höhle & Höhle 2009** — green OA copy read in full; confirms plan §5's uncited NMAD claim and supplies the missing median-error measure (§7.4).
- **Reuter, Nelson & Jarvis 2007** — publisher (Taylor & Francis and the ACM mirror) returns 403 and Semantic Scholar shows the abstract "elided by the publisher", but **the authors' own 49-page manuscript is hosted by CGIAR-CSI and was read in full** (§7.1). Pagination differs from the typeset article; section and table numbering match. Note there is no separate "Hole-filled SRTM v4" methodology PDF — that string is a citation form (Jarvis, Reuter, Nelson & Guevara 2008), not a document, and the old `cgiar-csi.org` data page now 403s.
- **GeomorPM, doi:10.1080/13658816.2024.2414409** — **full text not reached.** Routes tried and their results: tandfonline `/doi/full/`, `/doi/abs/`, `/doi/epdf/` → 403 to automated fetches, and a "Get Access" paywall (USD 64 / 48 h) in a browser; Semantic Scholar → abstract, `openAccessPdf` CLOSED; OpenAlex → `is_oa: false`, `oa_status: closed`; Unpaywall → `oa_locations: []`, `has_repository_copy: false`; Europe PMC → 0 hits; CORE → 0 hits; OpenAIRE → metadata only; BASE → access denied; GitHub and Gitee → no repository; arXiv → nothing. **Sections 4–5, the baselines and every numeric result were not read.** §5.3's protocol table is derived entirely from the CC BY 4.0 Figshare deposit's own files.
- **The GeomorPM pretraining corpus** — not released ("please contact the author directly"), so whether the pretraining was global or China-only is **undetermined**. Memo §2.1 and §3.1 should not assume.
- **Guth et al. (2025), *Remote Sens.* 17:3919** — MDPI returns 403 on every route. Likely the best additional slope/roughness precedent for §7.3; **nothing from it is used here**.
- **Höhle & Höhle 2009 via the publisher** — ScienceDirect and Elsevier linkinghub 403. The green OA copy at Aalborg is served from `vbn.aau.dk/ws/files/…` (the `/files/…` path 403s), and that is what was read.
- **ASPRS standards via asprs.org / my.asprs.org** — 403. Read from verbatim copies hosted by ASPRS Florida Region (Ed. 1) and AAGS (Ed. 2); the Edition 2 change list was confirmed independently from the official *PERS* 89(10) article, which does load.
- **MERIT Hydro via Wiley** (402), **HydroSHEDS/Lehner 2008 via Wiley** (402, not used), **Bielski et al. via IEEE** (paywalled; author manuscript used), **PeerJ GEDTM30** (403), **OptiGDM RSE 2026** (403), **ISPRS IJGI 9(12):734** (403 despite being open access).
- **PeerJ GEDTM30 (peerj.com/articles/19673)** — HTTP 403 to automated fetch of both the article page and the PDF; not read here.
- **Attempts to find any shared open void-filling benchmark** — Papers with Code (site gone), ISPRS benchmarks, every IEEE GRSS Data Fusion Contest 2006–2026, GitHub search and topics, Zenodo, Hugging Face datasets, EarthNets, PANGAEA, GEO-Bench, CodaLab, Codabench, Kaggle, EvalAI. Full detail in §7.7. The answer is no.

**What would actually resolve the residual uncertainty** and was not done, because it is out of scope for a research ticket:

1. Obtaining the Diff-DEM and GeomorPM PDFs through an institutional subscription. Note that even with both in hand, §3.5's leak and §5.3's regime mismatch still block a shared table — the PDFs would confirm the numbers, not make them commensurable.
2. **Downloading Diff-DEM's 3 × 1 000 benchmark and measuring how often a void contains the tile extremum**, which converts §3.5's leak from a protocol argument into a number. This is cheap — the data is live and it is a few lines — and would make §8's framing unarguable rather than merely well-founded. Recommended.
3. Downloading GeomorPM's Figshare deposit and running §5.4's A10 arm. That is the plan amendment, not a research task, but it is the thing that turns "externally incomparable" into "externally incomparable, and here is the one comparison we built anyway".

## Sources

**Primary — Diff-DEM and its lineage**
- Repository (README, `config/dem_completion.json`, `data/dataset.py`, `data/util/dem_transform.py`, `data/util/tif_metric.py`, `models/metric.py`, `core/util.py`, `run.py`, `evaluation.sh`): <https://github.com/kylelo/Diff-DEM>
- Publisher record (closed access): <https://ieeexplore.ieee.org/document/10535979> · doi:[10.1109/LGRS.2024.3403835](https://doi.org/10.1109/LGRS.2024.3403835)
- Abstract and access status: `https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/LGRS.2024.3403835?fields=title,abstract,openAccessPdf,externalIds`
- Gavriil, Muntingh & Barrowclough (2019), preprint: <https://arxiv.org/abs/1811.12693> · repo, licence and data: <https://github.com/konstantg/dem-fill>
- Zhang, Zuo & Li, ET-SDE: <https://arxiv.org/abs/2407.01908> (arXiv preprint only; `export.arxiv.org/api/query?id_list=2407.01908` returns no `journal_ref`)

**Primary — the plan's two cited design claims**
- GrounDiff (Dhaouadi, Meier, Kaiser, Cremers): <https://arxiv.org/abs/2511.10391> · full text <https://arxiv.org/html/2511.10391v1>
- FathomDEM (Uhe et al., *Environ. Res. Lett.* 20 034002, 2025): doi:[10.1088/1748-9326/ada972](https://doi.org/10.1088/1748-9326/ada972) · <https://iopscience.iop.org/article/10.1088/1748-9326/ada972>

**Primary — void-filling and DEM-accuracy convention**
- Luedeling, Siebert & Buerkert (2007), *ISPRS J. Photogramm. Remote Sens.* 62(4) 283–294, doi:[10.1016/j.isprsjprs.2007.05.004](https://doi.org/10.1016/j.isprsjprs.2007.05.004); author-hosted copy at uni-kassel.de, read in full
- Höhle & Höhle (2009), *ISPRS J. Photogramm. Remote Sens.* 64(4) 398–406, doi:[10.1016/j.isprsjprs.2009.02.003](https://doi.org/10.1016/j.isprsjprs.2009.02.003); green OA copy `vbn.aau.dk/ws/files/17874509/sdarticle.pdf`, read in full (Table 1, Table 2, Eq. 2)
- Grohman, Kroenung & Strebeck (2006), *PE&RS* 72(3) 213–216 — **not reached**; described via Luedeling et al. §1

**Primary — GeomorPM**
- Paper (paywalled): doi:[10.1080/13658816.2024.2414409](https://doi.org/10.1080/13658816.2024.2414409) · Crossref record `api.crossref.org/works/10.1080/13658816.2024.2414409` (*Int. J. Geogr. Inf. Sci.* 39(2) 422–451, published 2024-10-13, `abstract: null`, `license: null`) · abstract verified via `api.semanticscholar.org/graph/v1/paper/DOI:10.1080/13658816.2024.2414409` and the publisher access page
- **Figshare deposit "DEM foundation model", CC BY 4.0**: doi:[10.6084/m9.figshare.25756806.v1](https://doi.org/10.6084/m9.figshare.25756806.v1) — `vqmae.ckpt`, `vq-f4.ckpt`, `Areas.zip`, `dem_VF_vqmae_finetune.py`, `dem_LF_vqmae_finetune.py`, `dem_mae_models.py`, README. Everything in §5.2–§5.5 is read from these files.
- Cited source DEMs: ALOS AW3D30 (JAXA) · SRTM 1 Arc-Second Global, USGS doi:10.5066/F7PR7TFT

**Primary — convention survey**
- Reuter, Nelson & Jarvis (2007) author manuscript, hosted by CGIAR-CSI: `srtm.csi.cgiar.org/wp-content/uploads/files/Reuteretal2007.pdf`
- *Copernicus DEM Product Handbook* v5.0 (29.11.2022), Tables 1, 2, 12, 13, §2.1, §2.2: <https://dataspace.copernicus.eu/sites/default/files/media/files/2024-06/geo1988-copernicusdem-spe-002_producthandbook_i5.0.pdf>
- ASPRS *Positional Accuracy Standards for Digital Geospatial Data*, Ed. 1 v1.0 (2014) §7.1/§7.4/§7.12 and Ed. 2 v1.0.0 (2023) §7.2/§7.4/§7.15 — read from ASPRS Florida Region and AAGS copies respectively; change list confirmed from *PERS* 89(10) 589–592 (2023), doi:10.14358/PERS.89.10.589
- Bielski, C. *et al.* (2024), *IEEE Trans. Geosci. Remote Sens.* 62, 1–22, doi:10.1109/TGRS.2024.3368015 — read via author manuscript <https://arxiv.org/abs/2302.08425> (v6)
- Hawker, L. *et al.* (2022) FABDEM, *Environ. Res. Lett.* 17 024016, doi:[10.1088/1748-9326/ac4d4f](https://doi.org/10.1088/1748-9326/ac4d4f) — the elevation-only counter-example
- Yamazaki, D. *et al.* (2019) MERIT Hydro, *Water Resour. Res.* 55, 5053–5073, doi:[10.1029/2019WR024873](https://doi.org/10.1029/2019WR024873) — read from the University of Bristol repository copy of the version of record
- FathomDEM supplementary §S1 (Tables S1, S2, Figure S1, mask definitions in Table S4): <https://iopscience.iop.org/article/10.1088/1748-9326/ada972/data>
- Dfilled (2025): <https://arxiv.org/abs/2501.15440>
- Höhle, J. & Potuckova, M. (2011), *Assessment of the Quality of Digital Terrain Models*, EuroSDR Official Publication No. 60 — same lead author as the 2009 paper, restates its Table 2 scheme and adds the "apply robust measures when the normality check fails" decision rule: <https://www.eurosdr.net/sites/default/files/uploaded_files/eurosdr_publication_ndeg_60.pdf>

**Repo documents**
- `docs/research/task1-void-fill-plan.md` §1, §2.1, §2.4.4, §3.2, §3.4, §3.5, §4.5, §5, §6, §8, §9
- `docs/research/downstream-task-selection.md` §0, §1.4, §2.1, §2.2, §2.4, §2.5
- `docs/research/p101-mae-checkpoint-licence.md` (precedent for the unstated-licence finding)

All URLs retrieved 2026-08-06.
