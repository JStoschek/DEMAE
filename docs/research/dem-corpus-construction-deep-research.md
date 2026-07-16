# Constructing the DEMAE pretraining corpus: what matters, and what published work says

Deep-research report on the question: *what matters most in dataset construction for a
global DEM foundation model (ViT-MAE on Copernicus GLO-30), how to guarantee a good
distribution of terrain types, and whether the 50k–500k-tile plan is feasible.* Builds on
(and does not re-litigate) the descriptor choices in
[`terrain-representativeness-metrics.md`](terrain-representativeness-metrics.md) and the
read-budget analysis in [`terrain-metric-feasibility.md`](terrain-metric-feasibility.md).

Method: 5 search angles → 24 primary sources fetched → 119 claims extracted → top 25
adversarially verified (3-vote panels). 24 confirmed, 1 refuted. Findings below carry
their vote and confidence.

---

## 1. Verdict

**Curation beats raw size.** Every comparable Earth-observation foundation model that
published its corpus recipe abandoned uniform random sampling in favor of deliberately
engineered coverage — stratification, rare-class oversampling, and near-duplicate
filtering — and the one published corpus-size ablation (SSL4EO-S12) shows downstream
gains plateau around ~750K patches. DEMAE's 50k–500k tile plan (each 1° tile yielding
many 224² crops) is comfortably inside the published operating range, and storage budgets
of comparable corpora are single-digit terabytes. The strongest project-specific results:
terrain class frequencies are extremely long-tailed (8 geomorphons cover 50% of pixels in
the method's own benchmark), feature-space-coverage/k-means sampling on terrain
descriptors beats random and Latin-hypercube sampling on both accuracy *and* variance,
and slope/convexity/texture descriptors demonstrably fail to separate flat-terrain
subtypes without a relative-elevation axis — independently validating the already-chosen
descriptor set including the large-window TPI proxy. The main new findings that should
*change* plans: (a) the EDM/FLM/WBM per-pixel quality masks are the right filtering
instrument but are **not in the AWS COG bucket** and must be fetched from ESA separately;
(b) **vegetation cover deserves consideration as a sampling/quality axis** because land
cover drives GLO-30 elevation error more than slope does (GLO-30 is a canopy-top surface
model, up to ~12 m RMSE in rugged forest); and (c) longitude grid spacing coarsens in
bands above 50° latitude, so a pipeline assuming uniform 30 m square pixels systematically
distorts high-latitude terrain and its derived descriptors.

---

## 2. Corpus size, scaling behavior, and feasibility

**Diminishing returns arrive well within DEMAE's planned range.** *(high confidence, 3-0)*

- SSL4EO-S12 (IEEE GRSM): 251,079 locations → ~3M patches, ~3.7 TB uncompressed. Its
  ablation: "While 50% (500K) or less pre-training data yields significant performance
  drops, there's little diminishing gaps from 75% (750K) on" — only +1 point linear-probe
  accuracy from 750K to 1M patches.
- Prithvi-EO-2.0: 4.2M training samples of 256×256 at 30 m GSD (NASA HLS).
- Scale-MAE: ViT-Large MAE pretrained on ~363.6k images (FMoW RGB) for 800 epochs — squarely
  inside DEMAE's scale tier.

**Unit care:** these papers count fixed-size *crops*; DEMAE's tiers count 1° *tiles* that
each yield many crops, and the Phase-0 "~50k" is 50k *windows/crops*. Comparing
crop-to-crop, the scale tier (200–500k crops) sits between Scale-MAE's corpus and
SSL4EO's plateau point — a sound, even generous, plan. Storage at these sizes is trivially
feasible (SSL4EO's whole multimodal corpus is ~3.7 TB; single-band 224² float tiles are
far smaller per sample).

**Caveats:** the SSL4EO scale ablation used MoCo-v2 on ResNet50 with linear probing, not
MAE/ViT, and the paper notes saturation depends on model size. One tempting counter-claim
was **refuted 0-3** in verification: the alleged SelectiveMAE result that a curated 4M
subset *outperformed* the full 13M corpus. Do not cite non-monotonic returns; the safe
statement is SSL4EO's 750K plateau.

## 3. Sampling strategy: the published precedent is engineered coverage, not uniform

**Leading EO foundation models deliberately engineer geographic and semantic coverage.**
*(high confidence, 3-0 / 2-1)*

- **SSL4EO-S12**: population-weighted sampling — uniform over the top-10k cities, then a
  Gaussian (σ = 50 km) around each — with explicit overlap/near-duplicate filtering
  (<1–3% residual overlap between patches).
- **Prithvi-EO-2.0**: stratified by land-cover class (100 tiles per LULC class),
  oversampled urban areas (1,000 tiles) and high-LULC-entropy heterogeneous tiles
  (1,000), and enforced representation of all 846 global ecoregions above a 5% tile-area
  threshold (712 ecoregions present in 3+ tiles). *(ecoregion detail: 2-1 vote, strong
  verifier evidence)*

Neither stratifies on terrain descriptors — LULC/ecoregions are the imagery-world
analogue — so the precedent is at the *strategy* level: stratify on the semantic axes that
matter for your modality, oversample the rare and the heterogeneous, and dedupe. DEMAE's
terrain-descriptor stratification is the correct translation of this playbook to
elevation data.

**Terrain classes are extremely long-tailed, so uniform sampling *will* miss rare
terrain.** *(high confidence, 3-0)* In the geomorphon method's own benchmark (North
Carolina, 10 m DEM): 8 geomorphon patterns cover 50% of pixels, 36 cover 90%, and only
352 of 498 possible patterns occur at all — "the vast majority of terrain … consists of a
small number of standard landform types, while more exotic landform types are exceedingly
rare" (Stepinski & Jasiewicz). Stratified or importance sampling on landform class is
necessary just to get rare terrain into the corpus.

**Feature-space coverage sampling (k-means on descriptors, sample near centroids) is the
best-supported mechanism.** *(high confidence, 3-0)* Ma et al. 2020 (Geoderma): FSCS
outperformed conditioned Latin hypercube and simple random sampling on mean accuracy
across three prediction models and two study areas, and — as important for a one-shot
corpus build — sharply reduced run-to-run variance at equal sample size. Caveat: this is
a supervised soil-mapping simulation, so the transfer to SSL pretraining is analogical.

**A global k-means terrain classification on a handful of DEM descriptors is an
established published template.** *(high, 3-0 / 2-1)* Iwahashi et al. 2018 (log-slope /
local convexity / texture → 40 clusters → 15 terrain groups, globally); Iwahashi &
Yamazaki 2022 (90 m MERIT: log-slope, log-HAND, texture → 15/40 clusters, global and
per-basin), with clusters mapping onto interpretable landforms (low HAND+low texture+low
slope = plains; high HAND+high slope = mountains; low texture+high HAND =
plateaus/terraces/dunes). This is a direct template for DEMAE's stratification: cluster
the per-window descriptor vectors, then sample per cluster (with a floor for small
clusters) rather than hand-designing strata boundaries.

**The flat-wet axis is independently validated.** *(high, 3-0)* Iwahashi et al. 2018's
self-reported limitation: geometric signatures fail "to detect narrow valley bottom
plains, metropolitan areas, and slight rises in gentle plains" — and the 2022 follow-up
added HAND (a relative-height measure directly analogous to a large-window TPI)
*specifically* to fix this. This is exactly the gap the TPI-proxy decision in the prior
research notes addresses; the descriptor set stands as chosen.

## 4. Copernicus GLO-30 quality pitfalls that should shape filtering

**GLO-30 is a heavily edited Digital Surface Model, and every edited/filled pixel is
flagged in shipped per-pixel masks.** *(high confidence, 3-0, verified against Product
Handbook i5.0)*

- Water bodies are flattened, rivers flow-enforced, shorelines/airports/"implausible
  terrain structures" edited. EDM (Editing Mask) codes 0–13 distinguish void, infill,
  interpolated, smoothed, airport-edited, flattened, ocean/lake/river, shoreline, morphed,
  shifted pixels.
- Voids are filled from **ten heterogeneous external DEMs** recorded per-pixel in the
  Filling Mask (FLM): ASTER, SRTM90/30/30plus, GMTED2010, TerraSAR-X radargrammetric,
  AW3D30, Norway DEM v1/v2, DSM05 Spain. Filled pixels are not native TanDEM-X signal —
  they have different noise/resolution characteristics.
- **Operational catch: the AWS Open Data COG bucket ships only the DEM GeoTIFFs.** The
  EDM/FLM/WBM/HEM masks must be fetched separately from the ESA/Copernicus distribution.
  The EDM records only the *last* edit applied per pixel.

**The DSM-vs-DTM issue is the dominant quality axis for terrain-morphology use.**
*(medium confidence — direction unanimous, one figure paywall-blocked)* X-band radar
records near-canopy-top elevations: RMSE ranges from ~1.55 m (bare/flat) to ~11.91 m
(rugged forested), and ICESat-2 validation over China found land cover drives elevation
error *more strongly than slope* (ranking scoped to that study area; other regions show
slope dominating above ~15°). Both GLO-30 and AW3D30 "have poor performances in steep
slopes, with a limited capability to describe fine-scale morphology" (Alpine lidar
validation). The biases are large enough that ML-corrected bare-earth derivatives exist
specifically to remove them (FABDEM: dense-canopy median error 12.95 m → 0.45 m).
**Implication:** dense-canopy crops carry systematic canopy structure, not ground
morphology — vegetation cover is worth tracking as a manifest column / sampling axis, or
at minimum acknowledged as a known corpus bias.

**Grid geometry is latitude-dependent.** *(high confidence, 3-0, Handbook §1.2.3/Table 3)*
Latitude spacing is fixed at 1.0″, but longitude spacing coarsens by band: 1.0″ (0–50°),
1.5″ (50–60°), 2.0″ (60–70°), 3.0″ (70–80°), 5.0″ (80–85°), 10.0″ (85–90°) — tile columns
drop from 3600 to 2400 → 360. Even below 50°, ground distance per longitude pixel shrinks
as cos(lat) (~30.9 m → ~19.9 m within the 1.0″ band). A pipeline treating all tiles as
uniform square 30 m grids misrepresents high-latitude terrain and distorts slope/relief
computed in pixel units — consistent with the anisotropic-dx/dy caveat already flagged in
[`terrain-metric-feasibility.md`](terrain-metric-feasibility.md), but note the *band
discontinuities* above 50° additionally change the raster width itself.

**GLO-30 is nonetheless the right source.** *(high confidence, 3-0)* Among the three
newest 1-arcsec global DEMs it shows the best terrain representation and detail (ahead of
AW3D30 and NASADEM), corroborated independently by Purinton & Bookhagen 2021 and the
DEMIX benchmark ranking. Vertical accuracy among the three is near-tied (~6.6–6.7 m
RMSE); the detail advantage is what matters for morphology pretraining.

## 5. Redundancy and low-entropy content

**Remote-sensing imagery is demonstrably redundancy-heavy; dedup/selection is standard
curation.** *(medium confidence, 3-0)* SelectiveMAE (NeurIPS 2024 D&B): ~40% of RS image
patches suffice to train a comparable MIM model (HOG-based patch selection, >2×
pretraining speedup), and OpticalRS-13M was built with an explicit
exclusion/slicing/deduplication pipeline. General vision-SSL corpus work (SemDeDup,
DINOv2) treats semantic dedup as a standard stage. For DEMAE this supports screening or
downweighting low-entropy crops (flat deserts, ice sheets, edited-water flats) — the TRI/
descriptor stratification already does most of this implicitly by capping flat strata.
Transfer caveat: the 40% figure is specific to HOG selection on optical imagery; no
DEM-specific ablation exists, and whether flat terrain is wasted compute or necessary
coverage depends on downstream tasks (a model meant to embed *all* terrain arguably needs
some ice sheet and playa in-distribution).

## 6. What this changes for DEMAE (actionable)

1. **Fetch the EDM/FLM/WBM masks from ESA** (they are not in the AWS COG bucket) and
   record per-crop edited/filled-pixel fractions in the manifest; filter or cap crops
   dominated by flattened water, airport edits, or heterogeneous fill. Quantifying the
   global prevalence of edited pixels first (open question below) tells you whether
   strict exclusion is cheap or would gut coverage.
2. **Prefer k-means/FSCS-style cluster sampling in descriptor space** over hand-binned
   strata when the corpus recipe is finalized — it is the best-evidenced mechanism and
   the Iwahashi work is a published global template on nearly the same descriptor set.
3. **Consider a vegetation-cover column** (e.g., from a global canopy-cover product) in
   the manifest — as a quality flag if not a sampling axis; dense canopy = canopy-top
   signal, not ground morphology.
4. **Handle the >50° longitude-spacing bands explicitly** (raster width changes per band;
   pixel ground distances are anisotropic everywhere) in both descriptor computation and
   any resampling decision.
5. **The 50k debug / 200–500k scale tiers need no upward revision.** Published plateau
   evidence and comparable corpora say this range is adequate; spend the margin on
   curation, not more tiles.
6. **Don't over-prune flat terrain.** Redundancy evidence supports capping, not
   eliminating, low-entropy strata — downstream tasks may include exactly those terrains.

## 7. Open questions (no verified evidence exists)

- Does terrain-class balancing actually improve *downstream* performance for MAE
  pretraining on elevation, and by how much? No paper A/B-tests stratified vs uniform for
  a DEM (or even EO) MAE end-to-end.
- What is the MAE/ViT-specific corpus-size scaling curve for single-band terrain data?
  The 750K plateau came from contrastive learning on a CNN.
- Should low-entropy DEM crops be excluded, downweighted, or kept with a higher masking
  ratio? Unresolved; SelectiveMAE's patch-selection result is optical-only.
- What fraction of global land pixels are edited/filled per EDM/FLM? Nothing verified
  quantifies this; it determines whether strict mask-based filtering is cheap or would
  hollow out steep forested regions where voids concentrate.

## 8. Sources (primary unless noted)

**Foundation-model corpora & scaling**
- SSL4EO-S12 — Wang et al., IEEE GRSM. <https://arxiv.org/abs/2211.07044>
- Prithvi-EO-2.0 — Szwarcman et al. <https://arxiv.org/abs/2412.02732>
- Scale-MAE — Reed et al., ICCV 2023. <https://arxiv.org/pdf/2212.14532>
- SelectiveMAE / OpticalRS-13M — NeurIPS 2024 D&B. <https://arxiv.org/abs/2406.11933>

**Sampling & geomorphometry**
- Ma, Brus, Zhu, Zhang & Scholten 2020, Geoderma (FSCS vs cLHS vs SRS).
  <https://www.sciencedirect.com/science/article/pii/S001670612030135X>
- Stepinski & Jasiewicz 2013, Geomorphology (geomorphon frequency long tail).
  <https://www.sciencedirect.com/science/article/abs/pii/S0169555X12005028>
- Iwahashi et al. 2018, PEPS (global k-means terrain classification).
  <https://progearthplanetsci.springeropen.com/articles/10.1186/s40645-017-0157-2>
- Iwahashi & Yamazaki 2022, PEPS (k-means with HAND on MERIT).
  <https://link.springer.com/article/10.1186/s40645-022-00487-2>

**GLO-30 quality**
- Copernicus DEM Product Handbook i5.0 (Airbus/ESA, 2024) — EDM/FLM codes, grid bands.
  <https://dataspace.copernicus.eu/sites/default/files/media/files/2024-06/geo1988-copernicusdem-spe-002_producthandbook_i5.0.pdf>
- Li et al. 2022, Int. J. Digital Earth (ICESat-2 validation; land-cover-driven error).
  <https://www.tandfonline.com/doi/pdf/10.1080/17538947.2022.2094002>
- Marchi et al. 2023, Environ. Earth Sci. (Alpine lidar validation).
  <https://link.springer.com/article/10.1007/s12665-023-10882-7>
- FABDEM-lineage bare-earth correction review.
  <https://www.sciencedirect.com/science/article/pii/S2666544125000371>
- Digital Earth Africa COP DEM specs (secondary).
  <https://docs.digitalearthafrica.org/en/latest/data_specs/COP_DEM_specs.html>

**SSL data curation**
- SemDeDup — Abbas et al. 2023. <https://arxiv.org/abs/2303.09540>
- Beyond neural scaling laws (data pruning) — Sorscher et al. 2022. <https://arxiv.org/abs/2206.14486>
- DINOv2 (curation pipeline) — Oquab et al. 2023. <https://arxiv.org/abs/2304.07193>
- Data-scaling for MIM — <https://arxiv.org/abs/2305.15248>

**Refuted in verification (do not cite):** SelectiveMAE's alleged non-monotonic result
(4M curated subset beating the full 13M corpus at fixed compute) — 0-3 against.
