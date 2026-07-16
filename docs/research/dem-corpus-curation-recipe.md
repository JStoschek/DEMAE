# DEMAE corpus curation recipe: descriptors, strata, allocation, dedup, validation

Deep-research follow-up to
[`dem-corpus-construction-deep-research.md`](dem-corpus-construction-deep-research.md).
That report settled the *strategy* (GLO-30 + EDM/FLM/WBM masks; stratification +
rare-class oversampling + near-duplicate filtering; no uniform random sampling). This one
pins down the *operational detail*: exact descriptor engineering, cluster/strata counts,
allocation rules, dedup mechanics, and validation — each step grounded in verified
primary sources.

Method: 5 search angles → 20 sources fetched → 87 claims extracted → top 25
adversarially verified (3-vote panels). 24 confirmed, 1 refuted. Votes and confidence
noted per finding.

---

## 1. The recipe (direct steps)

The pipeline order below follows ADR-0001's score-on-overview-then-materialize shape;
the literature adds the *content* of each step, not a different ordering (no verified
source contradicts this ordering — see §7 on gaps).

**Step 1 — Enumerate candidate windows and hard-filter.**
For every land-bearing 1° COG, enumerate non-overlapping 224² candidate windows on the
×8 overview. Drop windows failing the land mask. Compute per-window fractions of
WBM water and EDM edited / FLM filled pixels; record them in the manifest. *(Thresholds
are a project decision — no published precedent survived verification; see §6.)*

**Step 2 — Compute a small, transformed, standardized descriptor vector per window.**
- **ln(slope)** — Iwahashi 2018 uses logarithmic slope as the classifying parameter.
- **ln(TPI-proxy + 1)** — Iwahashi & Yamazaki 2022 use lnHAND = ln(HAND+1) explicitly to
  emphasize distinctions among gentle/low-relief terrain; apply the same transform to the
  large-window TPI proxy (the project's HAND analog).
- **Combined surface texture at two scales** — Iwahashi 2018 computes texture at 3×3 and
  13×13 windows and takes the per-unit *minimum of normalized values* to handle scale
  dependence. (Their pixels are 280 m; on the ×8 overview of GLO-30, ~240 m/px, the same
  3×3/13×13 pairing is nearly scale-equivalent.)
- **Elevation** and **local relief** stay as manifest columns and stratification aids
  (relief may substitute for texture if redundant — check correlation on the smoke tier).
- **Keep the vector at 3–5 variables.** Iwahashi & Pike 2007 classified all global
  topography with only 3; no published global terrain classification in this lineage uses
  more than ~4 clustering inputs.
- **Z-score standardize every descriptor before clustering** (Iwahashi & Yamazaki 2022;
  Brus). Standardization, not whitening — no published precedent decorrelates.

**Step 3 — Build coarse terrain strata: k-means with k in 15–40.**
Run k-means over the standardized descriptor vectors of *all* candidate windows.
Published global precedent: 15 clusters "for a quick overview," 40 "so that users can
group data samples according to their needs" (Iwahashi & Yamazaki 2022; k-means,
convergence criterion 0, max 999 iterations); 8/12/16 classes in Iwahashi & Pike 2007.
Practical choice: **k = 40** for the stratum layer, with a coarser 15-ish grouping for
reporting. Persist the cluster id per window in the manifest (this is the `stratum`
column).

**Step 4 — Allocate the crop budget across strata: Neyman, with floors and boosts.**
- **Base allocation: Neyman** — n_h ∝ N_h·σ_h (stratum size × within-stratum descriptor
  variability). This is the classical variance-minimizing rule and the published
  middle ground between proportional (rare terrain starves) and equal (huge flat strata
  waste budget); it automatically up-weights heterogeneous rare terrain.
- **Minimum-presence floor + fixed quotas for rare classes**, following Prithvi-EO-2.0's
  verified template: fixed per-class quotas (they took 100 tiles per LULC class from the
  500 tiles with the highest class proportion), targeted boosts for special classes
  (1,000 urban + 1,000 high-entropy tiles), a 5% area threshold for a class to count as
  present, and an implicit ~3-tiles-per-stratum minimum. DEMAE analog: a floor of
  N_min crops per stratum (so no terrain family is absent), quota top-ups for known-rare
  strata (high-relief, flat-wet/valley-bottom), and a boost pool for
  high-descriptor-entropy windows (heterogeneous terrain).
- Cap the largest flat strata implicitly via Neyman (low σ_h ⇒ few samples) rather than
  by deleting them — prior research: cap, don't eliminate.

**Step 5 — Select windows within each stratum by feature-space coverage, not randomly.**
Within each stratum, pick the n_h windows that best cover the stratum's descriptor
space. Brus's canonical FSCS mechanics: set the number of sub-clusters equal to the
number of samples wanted and take the unit with the shortest scaled Euclidean distance
to each centroid. FSCS beat cLHS and random on both accuracy and run-to-run variance
(Ma et al. 2020; independently, Wadoux et al. 2019 found cLHS underperforms and
recommended spreading samples uniformly in covariate space).

**Step 6 — Near-duplicate control: spatial-overlap exclusion with an R-tree.**
The verified SSL-corpus precedent is geometric, not perceptual: SSL4EO-S12 rejects a
candidate whose center is closer than **¾ of the patch width** to an accepted one
(>25% overlap), and SSL4EO-L replaces the lon/lat-grid heuristic with an **O(N log N)
R-tree**, eliminating the residual 1–3% overlap. DEMAE analog: enforce center spacing
≥ ¾ × 6.72 km ≈ **5 km** between accepted crops (zero overlap is guaranteed at ≥ 6.72 km;
the ¾ rule permits ≤25% overlap if window supply in a rare stratum is tight). No verified
precedent exists for perceptual hashing, SemDeDup-style embedding dedup, or
variogram-derived spacing on elevation rasters — the descriptor-space coverage step
(Step 5) already suppresses redundant same-looking terrain within strata.

**Step 7 — Materialize (full-res window reads), then validate representativeness.**
Fetch the selected windows at full resolution, apply per-tile standardization, write
shards + manifest. Then validate the corpus the way the sampling literature does
(Brungard & Boettinger 2010):
- **Continuous descriptors:** graphical comparison of the corpus's descriptor
  distributions against the full candidate-pool distributions (histograms/ECDFs
  per descriptor).
- **Categorical strata:** chi-square test of per-stratum corpus counts against the
  intended allocation, plus a hard check that every stratum ≥ its floor.
- Record region per crop (already planned) so a Phase-2 geographic holdout is possible —
  note no EO-benchmark holdout-design claim survived verification, so region-split
  design remains a project decision.

## 2. Descriptor engineering details (verified)

| Choice | Published precedent | Vote |
|---|---|---|
| ln(slope) | Iwahashi 2018: "logarithmic converted slope gradient (LN Slope) was used as a classifying parameter" | 3-0 |
| ln(relative height + 1) | Iwahashi & Yamazaki 2022: lnHAND = ln(HAND+1), "to emphasize the topography of gentle slopes" | 3-0 |
| Z-score before k-means | I&Y 2022 ("standardized values of lnSLOPE, lnHAND, TEXTURE"); Brus (center + scale to mean 0/sd 1); Iwahashi 2018 (8-bit min-max, volume-motivated) | 3-0, 3-0, 2-1 |
| 3–5 descriptors max | Iwahashi & Pike 2007 "three-part geometric signature" classified the globe; no lineage paper uses > ~4 clustering inputs | 3-0 |
| Two-scale texture, min-of-normalized | Iwahashi 2018: 3×3 and 13×13 windows, "minimum value of the normalized … surface texture values" | 3-0 |

**Geomorphon folding — open.** No verified precedent for mixing a categorical landform
class into k-means coverage sampling surfaced. Practical options (project decision):
per-window geomorphon class-proportion histogram appended to the vector, an entropy
scalar, or geomorphons as a *separate categorical stratification layer* crossed with the
k-means strata. The cleanest evidence-adjacent route: keep k-means on the continuous
descriptors and use geomorphon-derived rarity only for the quota/boost step (Step 4),
mirroring how Prithvi used LULC classes for quotas without clustering on them.

## 3. Cluster and allocation details (verified)

- **15/40 cluster precedent** (I&Y 2022, global, k-means, convergence 0, ≤999 iter);
  8/12/16 nested-means (I&P 2007). *(3-0)* One rationale was **refuted 0-3**: the claim
  that Iwahashi 2018 chose 40 as deliberate over-segmentation later merged to 15 by
  hierarchical clustering — do not cite that; the documented framing is the 2022 paper's
  15-overview/40-regroupable.
- **FSCS mechanics** (Brus): #clusters = #samples, select nearest-to-centroid unit. *(3-0)*
- **FSCS > cLHS > SRS** on accuracy and stability (Ma 2020; Wadoux 2019). *(3-0 ×4)*
  Domain is digital soil mapping (n = 20–900); transfer to a 200–500k-crop SSL corpus is
  extrapolation.
- **Neyman allocation** (n_h ∝ N_h·σ_h) is the variance-minimizing published choice.
  *(3-0 for the rule; the companion PcLHS-beats-FSCS empirical result was 2-1, a single
  soil study — take the allocation rule, not the method ranking.)*
- **Prithvi quota template**: 100 tiles/class from top-500 by class proportion; 1,000
  urban + 1,000 high-entropy boosts; 5% area threshold; 712 ecoregions in 3+ tiles.
  *(3-0, verified verbatim)*

## 4. Near-duplicate filtering details (verified)

- SSL4EO-S12: centers closer than ¾ patch width ⇒ >25% overlap ⇒ reject and resample;
  1°-grid bucketing left ~3% (Tokyo) / 1.5% (Chicago) / <1% residual overlap. *(3-0)*
- SSL4EO-L: O(N log N) R-tree replaces the grid heuristic and removes the residual. *(3-0)*
- Nothing survived verification for hashing/embedding dedup or variogram-based spacing on
  elevation data. SemDeDup remains an *optional* extra pass if embedding infrastructure
  exists later (e.g., dedup flat-plain crops that are far apart geographically), but it is
  unproven for DEMs and not required by precedent.

## 5. Validation details (verified)

Brungard & Boettinger 2010: optimal sample size was determined by how well the sample
reproduced covariate distributions — "graphical analysis for continuous data and
chi-square analysis of categorical data" (their 0.05–0.1% ratio is site-specific; the
*method* generalizes, the ratio does not). *(3-0)* This is the only representativeness-
validation precedent that survived; KL-divergence-style corpus validation and
region-holdout design produced no verified claims.

## 6. The evidence gap: EDM/FLM/WBM thresholds

Research question (4) — published thresholds for how much edited/filled/water area
disqualifies a GLO-30 window, and how FABDEM / forest-height / DEM-super-resolution
datasets filtered on the masks — produced **zero surviving claims**. Any threshold DEMAE
adopts is a project decision without direct precedent. Suggested defensible defaults
(explicitly *not* literature-backed): record exact fractions in the manifest; exclude
crops > ~20–30% EDM-edited or FLM-filled from non-TanDEM-X sources; treat WBM ocean
pixels via the existing land-mask rule; keep coastline crops as their own reportable
stratum rather than silently dropping them, since the exact-0 ocean fill creates the
artificial-cliff artifact already documented in
[`terrain-metric-feasibility.md`](terrain-metric-feasibility.md). Measure the global
prevalence of edited/filled pixels on the smoke tier first — that number decides whether
strict exclusion is affordable.

## 7. Open questions

- EDM/FLM/WBM disqualification thresholds and coastline-crop handling (exclude,
  mask-in-loss, or deliberate stratum) — no precedent (§6).
- Variogram/autocorrelation ranges for slope/relief/TPI at 30 m–1 km scales — would a
  variogram-derived minimum spacing differ meaningfully from the ¾-width rule?
- Does embedding-space dedup add value over spatial-overlap exclusion for elevation
  rasters (visually identical flat terrain at large geographic separation)?
- How to fold the geomorphon 10-class output into the descriptor vector (histogram vs
  entropy vs separate stratum) — no verified precedent for categorical inputs to FSCS.
- Pipeline-ordering and geographic-holdout design have no verified published precedent;
  the ADR-0001 ordering stands on its own engineering logic.

## 8. Sources (primary unless noted)

**Geomorphometry descriptor & cluster design**
- Iwahashi & Pike 2007, Geomorphology (three-part signature, nested means, 8/12/16).
  <https://www.sciencedirect.com/science/article/abs/pii/S0169555X06004375>
- Iwahashi et al. 2018, PEPS (LN slope, two-scale combined texture, global k-means).
  <https://progearthplanetsci.springeropen.com/articles/10.1186/s40645-017-0157-2>
- Iwahashi & Yamazaki 2022, PEPS (lnSLOPE/lnHAND/TEXTURE, z-scored, 15/40 clusters).
  <https://link.springer.com/article/10.1186/s40645-022-00487-2>

**Sampling design & allocation**
- Brus, *Spatial Sampling with R*, ch. 18 (covariate-space coverage sampling).
  <https://dickbrus.github.io/SpatialSamplingwithR/kmeans.html>
- Ma, Brus, Zhu, Zhang & Scholten 2020, Geoderma (FSCS vs cLHS vs SRS).
  <https://www.researchgate.net/publication/340560502>
- Wadoux, Brus & Heuvelink 2019, Geoderma (cLHS underperforms for RF mapping).
  <https://www.sciencedirect.com/science/article/abs/pii/S0016706119306792>
- Brungard & Boettinger 2010 (cLHS sample-size determination, distribution matching).
  <https://link.springer.com/chapter/10.1007/978-90-481-8863-5_6>
- PcLHS 2025 (Neyman allocation within regionalized cLHS; 2-1 on empirics).
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC11997126/>

**SSL corpus pipelines**
- SSL4EO-S12 — Wang et al., IEEE GRSM (¾-width overlap rule, grid bucketing).
  <https://arxiv.org/pdf/2211.07044>
- SSL4EO-L — NeurIPS 2023 D&B (R-tree dedup). <https://proceedings.neurips.cc/paper_files/paper/2023/file/bbf7ee04e2aefec136ecf60e346c2e61-Paper-Datasets_and_Benchmarks.pdf>
- Prithvi-EO-2.0 (quota + boost + threshold oversampling template).
  <https://arxiv.org/html/2412.02732v3>
- SemDeDup — Abbas et al. 2023 (embedding dedup; no DEM precedent).
  <https://arxiv.org/abs/2303.09540>

**GLO-30 masks**
- Copernicus DEM Product Handbook i5.0. <https://dataspace.copernicus.eu/sites/default/files/media/files/2024-06/geo1988-copernicusdem-spe-002_producthandbook_i5.0.pdf>
- FABDEM (Hawker et al. 2022, ERL) + GEE community catalog (secondary) — fetched; no
  mask-threshold claims survived verification.

**Refuted in verification (do not cite):** Iwahashi 2018 allegedly choosing 40 clusters
as deliberate over-segmentation merged to 15 via hierarchical clustering (0-3 against).
