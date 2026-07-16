# Terrain-descriptor selection for a representative DEM corpus

Evaluates the candidate per-window feature set the Phase-0 corpus recipe wants to use
to **stratify GLO-30 sampling so every terrain family on Earth is represented** and no
whole class is accidentally omitted. Candidate set under review:

> **TRI + slope + local relief (or raw elevation) + a multi-scale TPI / geomorphon landform class.**

Judged against primary sources on land-surface parameterisation (Evans; Wilson & Gallant;
Hengl & Reuter) and the origin papers for each index (Riley 1999; Sappington 2007; Weiss 2001;
Jasiewicz & Stepinski 2013; Beven & Kirkby 1979). Feeds the corpus-recipe ticket (#4): which
columns the manifest stores and what the sampler stratifies on. Elevation-only DEM is the sole
modality, so this note is strictly about *topographic* descriptors — the non-topographic axes of
"representative of the actual world" are flagged separately in §7.

---

## 1. Verdict

The candidate set is **broadly sensible but neither minimal nor complete as written.** It carries
one clear redundancy and one real coverage gap. **TRI is redundant** with the slope + local-relief
it already includes — on a locally planar surface TRI is essentially a function of slope, which is
the exact reason the Vector Ruggedness Measure was invented (Sappington et al. 2007). **Elevation
and local relief are *not* redundant** and should both be kept: relief is a windowed *amplitude/scale*
measure, elevation is *absolute altitude* (a climate/latitude proxy). **TPI and geomorphons overlap**
— geomorphons are a self-adapting, multi-scale superset of Weiss's two-scale TPI landform scheme, so
keep one (geomorphons). The genuine gap is the **hydrological / flatness axis**: a broad wet valley
floor, floodplain, or playa reads as low-slope, low-TRI, low-relief and is only partly rescued by the
landform class; a **Topographic Wetness Index (TWI)** term (or an explicit flat-wet stratum) is the
one real addition worth making. Aspect and explicit curvature are minor gaps for *representativeness*
purposes (see §4). Net: **drop TRI, keep slope + local relief + elevation + geomorphons, add TWI.**

## 2. The order-of-surface decomposition (the yardstick)

Geomorphometry describes a land surface as a hierarchy of derivatives of the elevation field plus two
"contextual" families that a pure point-derivative misses (Evans, in Hengl & Reuter 2009, ch. 6;
Wilson & Gallant 2000; Olaya 2009). Any stratification claiming to span "all terrain" must touch each
row, because these axes are *empirically* the near-independent ones — PCA/factor studies of DEM
derivatives repeatedly recover elevation, slope, curvature and a wetness/position term as separate
components, and find curvature "almost no correlation with slope and aspect" (Evans 1980; Florinsky
2016; PCA reviews below).

| Surface order | Parameter | What it measures | Candidate metric that sees it | Covered? |
|---|---|---|---|---|
| 0th (height) | **elevation** | absolute altitude; climate / snowline / latitude proxy | raw elevation | ✅ (if elevation kept, not just relief) |
| 1st (gradient magnitude) | **slope** | steepness of the local tangent plane | slope; also TRI, local relief | ✅ (redundantly) |
| 1st (gradient direction) | **aspect** | which way the slope faces | — | ⚠️ gap (usually a nuisance axis — see §4) |
| 2nd (curvature) | **profile / plan curvature** | convex vs concave; accelerating vs converging flow | *partially* via geomorphon / TPI shape class | ⚠️ partial |
| amplitude / scale | **local relief, roughness, TRI, VRM** | vertical range / ruggedness within a window | TRI + local relief | ✅ (redundantly) |
| contextual position | **TPI, geomorphons** | ridge / slope / valley / flat, relative to neighbourhood | multi-scale TPI *or* geomorphon | ✅ (pick one) |
| hydrological | **TWI, catchment/flow** | flat wet convergent valley bottoms | — | ❌ gap (the real one) |

Reading the table: the candidate set is **heavy on the 1st-order/amplitude band** (three metrics all
keyed to local elevation differences — slope, TRI, relief), **adequate on 0th-order and context**
(elevation + a landform class), **thin on curvature** (only implicitly, through the landform class),
and **blind on hydrology/flatness** (nothing sees a saturated valley floor as distinct from a dry
plain).

## 3. Definitions, pinned to primary sources

Precise definitions matter because two of these indices have divergent formulas in the wild.

- **TRI — Riley, DeGloria & Elliot (1999).** Original definition: for the 3×3 neighbourhood, the
  **square root of the sum of the squared elevation differences** between the centre cell and its 8
  neighbours. It was built for *wildlife habitat* (concealment/lookout for prey), with dimensionless
  ruggedness classes: 0–80 m level, 81–116 nearly level, 117–161 slightly, 162–239 intermediately,
  240–497 moderately, 498–958 highly, >959 extremely rugged (Riley et al. 1999).
  **Variant caution:** GDAL `gdaldem TRI` offers two algorithms — **Riley** (default; sqrt of
  sum-of-squares, "recommended for terrestrial use") and **Wilson** (mean of the *absolute*
  differences, "recommended for bathymetric use") (GDAL `gdaldem` docs; Wilson et al. 2007). They are
  not interchangeable; if TRI is computed, record which. On uniform slopes both reduce to a monotone
  function of slope.
- **slope** — magnitude of the elevation gradient; GDAL offers Horn (rougher terrain) and
  Zevenbergen–Thorne (smooth) neighbourhood estimators (GDAL `gdaldem` docs).
- **local relief / roughness** — windowed elevation *range*. GDAL `roughness` = "the largest
  inter-cell difference of a central pixel and its surrounding cell" (max−min in the window)
  (Wilson et al. 2007). A larger-window relief is the scale/amplitude measure the recon note's TRI
  pre-screen already gestures at.
- **TPI — Weiss (2001).** Topographic Position Index = centre-cell elevation minus the **mean** of a
  surrounding annulus (GDAL `gdaldem TPI`; Wilson et al. 2007). Weiss's contribution is the
  **multi-scale** use: standardise TPI at a small and a large radius and cross-tabulate to get
  slope-position and 10-class landform maps (canyon, midslope drainage, plain, open slope, ridge,
  mountain top, …). Ecologically, TPI matters at several scales at once (300 m / 1000 m / 2000 m in
  his Nevada vegetation example) (Weiss 2001).
- **Geomorphons — Jasiewicz & Stepinski (2013).** A *pattern-recognition* alternative to differential
  geometry: at each cell, eight **line-of-sight** zenith comparisons yield a ternary code
  (−1/0/+1, "lower/equal/higher", with a flatness threshold). The 3^8 patterns collapse to **498
  distinct geomorphons**, mapped to **10 common landform classes**: flat, summit, ridge, shoulder,
  spur, slope, hollow, footslope, valley, depression. Crucially the line-of-sight makes the window
  **self-adapting to local scale** up to a search radius — it is effectively a multi-scale TPI that
  need not be told the scale (Jasiewicz & Stepinski 2013).
- **TWI — Beven & Kirkby (1979).** ln(a / tan β), where a is specific upslope contributing area and β
  is local slope; the TOPMODEL wetness index for variable saturated source areas. High where large
  drainage converges on low slope — i.e. **exactly the flat, wet valley-bottom terrain** that slope,
  TRI and relief all read as "boring/flat" (Beven & Kirkby 1979).
- **VRM — Sappington, Longshore & Thompson (2007).** Decompose each cell's slope+aspect into a 3-D
  unit normal vector, then measure the **dispersion (1 − resultant-vector magnitude)** of those
  vectors in a window. Its entire design purpose is a ruggedness measure **decoupled from slope**
  (Sappington et al. 2007).

## 4. Redundancy analysis

**TRI ↔ slope ↔ local relief are one family, largely redundant.** All three are functions of local
elevation *differences* in a small window. On a locally planar (constant-gradient) patch, TRI and
roughness are monotone functions of slope; they only diverge from slope where the surface is *rough at
the pixel scale* (bumpy but not steep, or steep but smooth). Sappington et al. (2007) state the
general result plainly: "most methods used to quantify ruggedness are indices derived from measures of
slope and, as a result, are strongly correlated with slope," and they built VRM precisely because it
"quantifies local variation in terrain more independently of slope than other methods tested"
(Sappington et al. 2007). *(I could not extract the paper's specific per-range r/r² values from open
sources — the numeric coefficients sit behind the Journal of Wildlife Management paywall — so treat
"strongly correlated" as the well-established qualitative claim from the abstract, not a specific
number I verified.)* Practical consequence: **slope + local relief already carry almost all the
information TRI would add.** If a genuinely slope-*independent* ruggedness axis is wanted, VRM is the
right tool for it — but for *coverage stratification* it is a "nice-to-have," not a gap-filler.

**Local relief vs elevation — NOT redundant; keep both.** Relief = a windowed *range* (an amplitude /
scale descriptor: how much vertical variation over ~km). Elevation = *absolute height* (altitude, a
proxy for climate zone, snow/tree line, and — jointly with latitude — biome). A 500 m-relief window at
200 m altitude (dissected foothill) and at 4000 m altitude (alpine ridge) are different terrain the
model should see; relief alone cannot distinguish them. Keep elevation (a robust per-window statistic,
e.g. median, since the pipeline standardises per tile anyway) **and** a larger-window relief.

**TPI vs geomorphons — overlapping; keep geomorphons.** Multi-scale TPI (Weiss 2001) and geomorphons
(Jasiewicz & Stepinski 2013) both label a cell's position (ridge/slope/valley/flat). Geomorphons are
the newer, self-adapting, single-pass multi-scale classifier and subsume the two-scale TPI cross-tab;
their 10 classes already encode convex (summit/ridge/spur/shoulder), concave (valley/hollow/footslope/
depression) and neutral (flat/slope) shape — i.e. a *categorical curvature-and-position* proxy.
Running both is double-counting. Recommend **geomorphons**; if a dependency-light path is preferred,
a two-scale Weiss TPI is an acceptable substitute (both are one categorical column).

**What the landform class does and does not rescue.** Because geomorphons include *flat*, *valley* and
*depression*, they *do* give the flatness family a foothold — a broad plain won't be invisible. What
they miss is the **wetness/convergence gradient**: geomorphons cannot tell a dry flat plateau from a
converging saturated valley floor of the same shape, and their flatness threshold can label a gently
tilted floodplain as "slope." That residual is the TWI gap in §5.

**Aspect and explicit curvature — deliberately de-prioritised.** Aspect (gradient *direction*) is
largely a *nuisance* axis for a global shape-learning corpus: terrain families are not defined by
compass facing, per-tile standardisation is rotation-agnostic, and oversampling by aspect would just
inflate the corpus without adding terrain *types*. Curvature (2nd order) is the more defensible
omission-to-watch, but the geomorphon class already carries a categorical convex/concave signal, so an
explicit profile/plan-curvature channel is optional (see §6 tier).

## 5. The one real gap: hydrology / flatness (TWI)

Every amplitude metric in the candidate set (slope, TRI, relief) is near-zero over flat wet terrain,
and elevation is uninformative there. So **floodplains, deltas, playas, peatlands, glacial-outwash
plains, karst poljes and broad valley floors** — a genuine, globally extensive terrain family — sit in
the same low-everything corner of feature space as any other flat ground, and risk being collapsed
into one under-differentiated "flat" bucket or under-sampled entirely. **TWI = ln(a/tanβ)** (Beven &
Kirkby 1979) is the standard descriptor that separates *convergent, low-gradient, high-contributing-
area* land (wet valley bottoms) from *flat-but-dry* uplands. Adding a TWI term — or, more cheaply, an
explicit "low-slope high-flow-accumulation" flat-wet stratum in the sampler — is the single change
that most improves *coverage* of the candidate set. (Caveat: TWI needs flow accumulation, i.e. a
filled DEM and a larger context than a 224² window; for a per-window score it may be approximated, or
computed once at COG scale and read back per window.)

## 6. Recommended minimal set

**Core (guarantees the major terrain families are represented):**

1. **Elevation** — a robust per-window height statistic (0th order / altitude-climate axis). Nearly
   free; orthogonal to everything else.
2. **Slope** — 1st-order gradient magnitude (Horn). The steepness axis.
3. **Local relief** — windowed elevation range at a ~km window (amplitude / scale axis; distinguishes
   subdued from high-amplitude terrain independent of steepness).
4. **Geomorphon landform class** — contextual position + categorical convex/concave shape
   (ridge/valley/flat/slope/…), self-adapting scale.
5. **TWI** (or an explicit flat-wet stratum) — the hydrology/flatness axis the other four miss.

**Drop from the candidate set:** **TRI** (redundant with slope + local relief; if a slope-*independent*
ruggedness is ever wanted, prefer **VRM**), and the **separate multi-scale TPI** (subsumed by
geomorphons — keep only one landform classifier).

**Optional "add if you care about X" tier:**

- **VRM (Sappington 2007)** — if you want a ruggedness axis provably decoupled from slope (e.g. to
  distinguish "steep-and-smooth" scarps from "rough-and-broken" badlands, which slope+relief conflate).
- **Profile + plan curvature** — if the categorical geomorphon shape signal proves too coarse and you
  want a continuous 2nd-order channel.
- **Aspect (sin/cos encoded)** — only if a downstream task cares about slope orientation (insolation,
  aspect-driven microclimate); not needed for terrain-*type* coverage.

**Stratify on the joint distribution, not the margins.** Covering the *marginal* histogram of each of
the five descriptors independently does **not** guarantee coverage of the *joint* space — the rare
terrain hides in the corners (e.g. high-elevation + low-relief = an altiplano/plateau; high-relief +
flat-geomorphon interior = a canyon floor). Bin-per-axis sampling will systematically miss these
combinations. Recommend **clustering the standardised descriptor vectors** (k-means / mini-batch
k-means on a large candidate-window sample) and sampling to flatten the *cluster* histogram, so the
sampler targets the joint terrain-type manifold rather than one axis at a time. This is the mechanism
that actually delivers "no terrain family left out."

## 7. What no DEM metric can see

Representativeness of the *actual world* has axes that are **not topographic** and therefore invisible
to every index above. These are a **separate stratification against external layers**, not something to
derive from elevation, and they should be recorded/planned for explicitly rather than assumed covered:

- **Climate / Köppen zone** and **latitude** — a 1000 m-relief tropical ridge and an Arctic one are the
  same *terrain* but different *world*; elevation is only a weak proxy.
- **Land cover / vegetation** — desert vs forest vs cropland vs ice; note GLO-30 is a DSM, so canopy
  and buildings perturb the surface, but the metrics can't *label* land cover.
- **Lithology / substrate** — granite vs limestone (karst) vs loess produce different geomorphology the
  DEM shape only indirectly reflects.
- **Coastal vs interior / tectonic setting** — active-margin vs cratonic terrain.
- **Anthropogenic modification** — terraces, mines, urban fabric.

For DEMAE's pretraining-coverage goal, the honest framing is: **DEM descriptors guarantee *geomorphic*
representativeness; geographic/climatic representativeness needs a geo/Köppen split layered on top**
(the manifest already records lon/lat + region for exactly this, per the recon note and CONTEXT.md).

## 8. Sources

Primary:
- Riley, S.J., DeGloria, S.D. & Elliot, R. (1999). *A Terrain Ruggedness Index That Quantifies
  Topographic Heterogeneity.* Intermountain Journal of Sciences 5(1–4):23–27.
  https://www.researchgate.net/publication/259011943 · class thresholds via spatialEco `tri`:
  https://www.rdocumentation.org/packages/spatialEco/versions/2.0-2/topics/tri
- Sappington, J.M., Longshore, K.M. & Thompson, D.B. (2007). *Quantifying Landscape Ruggedness for
  Animal Habitat Analysis: A Case Study Using Bighorn Sheep in the Mojave Desert (VRM).* Journal of
  Wildlife Management 71(5):1419–1426. https://doi.org/10.2193/2005-723 ·
  https://pubs.usgs.gov/publication/70000586
- Weiss, A. (2001). *Topographic Position and Landforms Analysis.* Poster, ESRI User Conference, San
  Diego. https://www.jennessent.com/downloads/TPI-poster-TNC_18x22.pdf
- Jasiewicz, J. & Stepinski, T.F. (2013). *Geomorphons — a pattern recognition approach to
  classification and mapping of landforms.* Geomorphology 182:147–156.
  https://doi.org/10.1016/j.geomorph.2012.11.005
- Beven, K.J. & Kirkby, M.J. (1979). *A physically based, variable contributing area model of basin
  hydrology (TOPMODEL / TWI).* Hydrological Sciences Bulletin 24(1):43–69.
  https://doi.org/10.1080/02626667909491834
- Wilson, M.F.J., O'Connell, B., Brown, C., Guinan, J.C. & Grehan, A.J. (2007). *Multiscale terrain
  analysis of multibeam bathymetry data (TPI, roughness, TRI-Wilson variant).* Marine Geodesy 30:3–35.
  https://doi.org/10.1080/01490410701295962
- GDAL `gdaldem` documentation (authoritative slope / aspect / TRI / TPI / roughness definitions and
  the Riley-vs-Wilson TRI algorithms). https://gdal.org/programs/gdaldem.html

Reviews / land-surface-parameter independence:
- Hengl, T. & Reuter, H.I. (eds.) (2009). *Geomorphometry: Concepts, Software, Applications.* Elsevier
  — esp. ch. 6 "Basic Land-Surface Parameters" (Olaya) and Evans's parameter hierarchy.
  https://www.researchgate.net/publication/251455003
- Evans, I.S. (1980, and later). *Relations between land surface properties: altitude, slope and
  curvature* — curvature near-independent of slope/aspect.
  https://www.researchgate.net/publication/225202795
- PCA / factor studies of DEM derivatives recovering elevation, slope, curvature and a wetness/position
  term as separate axes: *Quantifying Topographic Ruggedness Using Principal Component Analysis*
  https://www.researchgate.net/publication/355858830 ; *Land-surface parameters for spatial predictive
  mapping and modeling* https://doi.org/10.1016/j.earscirev.2022.103951
- Related in-repo: `docs/research/copernicus-glo30-recon.md` (GLO-30 corpus facts; TRI pre-screen on
  the ×8 overview), `CONTEXT.md` (manifest records lon/lat + region for the geo split).
