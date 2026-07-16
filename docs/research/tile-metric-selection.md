# Which per-window metrics for the DEMAE corpus: minimal set, redundancy, scale-fit

Deep-research report answering: *of the candidate metrics (local relief, slope mean+std,
hypsometric integral, dissection/drainage texture, VRM-like roughness, anisotropy, mean
elevation, TPI, geomorphon dominant class), what is the smallest set that best spans
terrain variability for 224×224 @ 30 m (~6.7 km) windows, which are redundant, and which
don't work at this window size?* Supersedes parts of
[`terrain-representativeness-metrics.md`](terrain-representativeness-metrics.md) where
noted (§5). Constraint context: metrics computed per window from the full-res crop or
the ×8 overview; basin-scale hydrology excluded (full-res scoring on an in-region box is
an accepted fallback per project decision 2026-07-16, though nothing below requires it).

Method: 5 angles → 24 sources → 107 claims extracted → top 25 adversarially verified
(3-vote panels). 22 confirmed, **3 refuted** — and the refutations matter as much as the
confirmations here.

---

## 1. The recommendation

**Core set (5 metrics), each a few lines of numpy/scipy, all computable on the ×8
overview:**

| # | Metric | Axis | Formula sketch |
|---|---|---|---|
| 1 | **Local relief** | amplitude | `z.max() - z.min()` |
| 2 | **Slope std** (keep mean too in the manifest; prune one after the correlation check, §4) | amplitude spread / heterogeneity | std of per-pixel gradient magnitude (anisotropic dx/dy) |
| 3 | **Elevation-relief ratio** (≡ hypsometric integral) | distribution shape | `(z.mean() - z.min()) / (z.max() - z.min())`, flat-window guard |
| 4 | **Iwahashi–Pike surface texture** | dissection / frequency | pit+peak density: `median_filter(z, k) - z` → sign-threshold → `uniform_filter` count in ~10-cell-radius circular kernel |
| 5 | **Mean elevation** | absolute altitude | `z.mean()` (already stored for standardization) |

**Optional 6th:** spherical standard deviation of surface normals (the well-behaved
member of the VRM family) as a slope-decoupled roughness axis — include *only if* it
survives the corpus correlation/distribution check (§4); raw VRM at 3×3 is documented to
degenerate.

**Drop:** geomorphon **dominant class** (uninformative mode at 6.7 km; if geomorphons are
kept at all, use class entropy/histogram — and their claimed orthogonality to relief was
refuted, so even entropy must pass the correlation check). **TPI** (no verified evidence
on window-size behavior survived; it overlaps ERR conceptually — both encode vertical
position of the window mean, just against different references). **Anisotropy** (zero
verified evidence at this scale either way — defer rather than ship untested).
**Basin-style drainage texture** (excluded by the locality constraint; the I&P texture
metric *is* the hydrology-free replacement for this axis).

## 2. Why these five: the axis evidence

- **Multiple axes are justified in principle.** The land surface "is not (uni-)fractal
  but varies along multiple statistical dimensions" (Evans & Cox 1999), and
  altitude–slope correlation is non-monotonic, flipping sign by terrain type (negative in
  dissected plateaus) — so mean elevation is demonstrably *not* a ruggedness proxy and
  earns its own axis. *(3-0, 3-0)*
- **Three windowed variables already stratify the globe.** Iwahashi & Pike 2007 classify
  world terrain unsupervised with only slope, local convexity, and surface texture, at
  1 km (global 16-class, Hammond-like) and 270 m (Japan, matches airphoto landform maps).
  The 270 m demonstration is essentially the ×8-overview regime (~240 m/px). Direct
  precedent that a handful of window statistics is enough — though I&P omit elevation and
  anisotropy, so three proves sufficiency for *their* map, not completeness. *(3-0 ×2)*
- **Hypsometric integral = elevation-relief ratio, and square windows are fine.**
  Pike & Wilson 1971 *mathematically prove* the identity — not an approximation — and
  verify agreement "for arbitrarily bounded topographic samples," i.e. square windows,
  not just drainage basins. Iwahashi & Pike use it in square moving windows as
  established practice. One line of numpy. *(3-0 ×5, 2-1)* Caveats: HI values are only
  comparable at a fixed window size (satisfied — all windows are 6.7 km), and its
  independence from relief is a single-study 2-1 result (Walcott & Summerfield 2008,
  basins, contradicting earlier work) — so ERR-vs-relief correlation goes on the §4
  checklist. *(2-1, 3-0)*
- **Local relief is robust to overview computation.** Relief is strongly
  window-size-dependent (fine — window size is fixed) and differs <5% between 30 m and
  90 m DEMs vs >15% at 1 km (Lin & Oguchi 2015); ~240 m sits in a moderate-degradation
  regime — acceptable, worth a one-off calibration against full-res on the smoke tier.
  *(3-0 ×2)*
- **The texture metric is the legal version of "dissection/drainage texture".**
  I&P's surface texture extracts pits/peaks as `median-smoothed − original` and counts
  their density in a ~10-cell-radius circular kernel — no flow accumulation, no
  depression filling, no cross-tile anything; `scipy.ndimage` in a few lines
  (SAGA-GIS ta_morphometry_22 is a reference reimplementation). *(3-0)*
- **Normals-dispersion roughness is complementary *by design* but fragile in practice.**
  The ruggedness-vs-roughness framework (Lindsay et al. 2019, adopted by the Argudo 2025
  survey) separates a smooth steep slope (high ruggedness, zero roughness) from flat-but-
  complex terrain; roughness is exactly 0 on any inclined plane. But in the one verified
  empirical test (Habib 2021: 10 m DEM, 3×3 windows, one arid site) VRM decorrelated from
  slope (r = 0.29) yet classed ~97% of pixels "level" and failed to discriminate
  moderate-to-high ruggedness. Hence: spherical-std-of-normals formulation, computed at
  window scale not 3×3, admitted only after corpus validation. *(3-0 ×4)*
- **Geomorphon dominant class is a wasted column.** Frequencies are extremely
  concentrated (8 patterns cover 50% of pixels, 10 most frequent ≈ 60%), so the modal
  class of a ~50k-pixel window is nearly always the same generic form. *(3-0; the
  window-level implication is a flagged conservative inference from pixel statistics.)*

## 3. The refuted claims (these change the picture)

1. **"Slope, relief, and roughness are near-collinear (r ≈ 0.86–0.98, one PC = 82%)" —
   REFUTED 0-3.** The strongest redundancy hypothesis is *not established*. Do not
   assume the amplitude family collapses to one factor at 6.7 km; measure it.
2. **"Windowed elevation std is interchangeable with TRI (99.7% map agreement)" —
   REFUTED 1-2.** Don't cite this shortcut.
3. **"Geomorphons are by construction orthogonal to relief/slope" — REFUTED 0-3.**
   Geomorphon entropy cannot be assumed to add an independent axis; it must earn its
   place empirically. (This weakens the geomorphon recommendation in
   `terrain-representativeness-metrics.md` §1.)

## 4. The decisive step: an empirical correlation pass on the smoke tier

Because the redundancy question came back *unresolved* (refuted PCA claim; single-site
counter-evidence), the literature's own answer is: **compute everything cheap, correlate,
then prune.** Argudo et al. 2025 (Eurographics STAR, peer-reviewed, open code at
`github.com/oargudo/terrain-descriptors`) publish a cross-descriptor correlation matrix
precisely so cheap metrics can substitute for correlated ones. *(3-0)*

Concretely, on the smoke corpus (few hundred → few thousand windows):

1. Compute all candidates per window: relief, slope mean, slope std, ERR, texture,
   normals-dispersion, geomorphon entropy, mean elevation (all cheap on the overview).
2. Build the Spearman correlation matrix + PCA.
3. Prune to one representative per correlated block (|ρ| ≳ 0.9): expect to keep one of
   {relief, slope mean}, keep slope std only if it separates from both, keep ERR and
   texture and mean elevation (expected near-independent), admit normals-dispersion and
   geomorphon entropy only if they clear the bar and have non-degenerate distributions.
4. Freeze the surviving 4–6 metrics as the k-means stratification vector (log-transform
   slope/relief per the curation recipe; z-score all).

This is a half-day of work, fully unit-testable, and converts every unresolved
redundancy question into a measured answer on the actual corpus.

## 5. Relation to the earlier descriptor note

[`terrain-representativeness-metrics.md`](terrain-representativeness-metrics.md)
recommended slope + local relief + elevation + geomorphons (+ TWI→TPI proxy). This
report revises that on three points, on stronger evidence: **ERR (hypsometric integral)
enters** as a genuinely distinct distribution-shape axis with a proof of validity on
square windows; **I&P texture enters** as the dissection axis (the earlier note's
"contextual" gap-filler, geomorphons, loses its orthogonality claim to a 0-3 refutation
and its dominant-class form to the concentration argument); **TPI/flat-wet proxy is
demoted** from definite to open — ERR may already carry the vertical-position signal at
fixed window size, and no verified evidence on TPI window-size behavior survived. The
flat-wet axis concern (valley floors vs plains) remains real; whether ERR + texture
capture it is exactly what the §4 pass and a look at known flat-wet regions in the smoke
tier should decide.

## 6. Gaps and open questions (honest limits)

- **DSM-specific robustness is unaddressed by verified evidence**: nothing survived on
  how canopy-top surfaces, flattened water, or edited pixels distort texture/roughness
  (e.g., a flattened lake reads as zero-texture — the EDM/WBM water fraction column
  should gate these), or on latitude pixel-anisotropy effects on orientation metrics.
- **Anisotropy at 6.7 km**: no verified evidence either way; deferred, revisit if
  downstream tasks care about oriented fabrics (dunes, drumlins).
- **TPI window-size behavior**: targeted (Weiss 2001, multi-scale TPI) but zero claims
  survived verification.
- Single-source findings to hold loosely: HI-relief independence (2-1, basins), VRM
  degeneracy (one arid site), geomorphon concentration (one 10 m site, raw ternary
  patterns).
- No published precedent curates an ML DEM dataset by exactly these window statistics;
  the closest are Iwahashi–Pike classification and the Argudo survey. The final set
  composition is a synthesis inference, not a published result.

## 7. Sources (primary)

- Evans & Cox 1999, in *Process Modelling and Landform Evolution* (multi-dimensional
  land surface; altitude-slope non-monotonicity). <https://link.springer.com/chapter/10.1007/BFb0009718>
- Pike & Wilson 1971, GSA Bulletin (HI ≡ elevation-relief ratio, proof; square-sample
  validity). <https://pubs.geoscienceworld.org/gsa/gsabulletin/article/82/4/1079/7165>
- Walcott & Summerfield 2008, Geomorphology (HI scale-dependence; independence from
  relief, 2-1). <https://www.sciencedirect.com/science/article/abs/pii/S0169555X07003911>
- Sappington et al. 2007, J. Wildlife Mgmt (VRM). <https://wildlife.onlinelibrary.wiley.com/doi/10.2193/2005-723>
- Habib 2021 (VRM empirical degeneracy, one site). <https://onlinelibrary.wiley.com/doi/10.1155/2021/3311912>
- Iwahashi & Pike 2007, Geomorphology (3-variable global classification; texture
  algorithm). <https://www.sciencedirect.com/science/article/abs/pii/S0169555X06004375>
- Lin & Oguchi 2015, GIScience & RS (relief vs DEM resolution). <https://www.tandfonline.com/doi/full/10.1080/15481603.2015.1049577>
- Stepinski & Jasiewicz 2011 (geomorphon frequency concentration). <https://www.semanticscholar.org/paper/63d3301c150e5f02f86e4a8d422e2b834bd64df4>
- Argudo, Guérin, Schott & Galin 2025, Computer Graphics Forum STAR (descriptor
  correlation-matrix pruning workflow; open code). <https://onlinelibrary.wiley.com/doi/10.1111/cgf.70080>

**Refuted in verification (do not cite):** Habib's slope/relief/roughness
near-collinearity PCA (0-3); elevation-std ≡ TRI 99.7% interchangeability (1-2);
geomorphons' by-construction orthogonality to relief (0-3).
