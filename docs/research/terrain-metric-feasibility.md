# Feasibility of the five terrain descriptors within the Phase-0 read budget

Resolves the wayfinder RESEARCH ticket **"Which terrain descriptors are computable within
the Phase-0 read budget?"** (GitHub issue #10). The prior note
[`terrain-representativeness-metrics.md`](terrain-representativeness-metrics.md) *chose* the
descriptor set (Elevation, Slope, Local relief, Geomorphon landform class, TWI) and defined the
objective as coverage-balanced stratification; this note does **not** re-litigate that choice. It
asks the orthogonal engineering question: given ADR-0001's read budget — **one ×8-overview read per
1° COG for scoring** (~26,454 reads; the ×8 overview of a 3600² tile is ~450²) plus **~50,000
full-resolution 224² window reads for materialization** — which of the five are actually computable,
at what scale, and **without breaking the property ADR-0001 §4 leans on**: *"one ×8-overview read per
tile yields both the ocean signal (§3) and the ruggedness score for all of that tile's windows."*

The key structural fact this note turns on: **the scoring read is not a per-window read — it is the
entire tile's ×8 overview grid** (~450×450 at low latitude, fewer columns poleward). So any operator
that is *local and windowed* — reads only a bounded neighbourhood — can be evaluated for **every**
window of that tile from the single array already in memory, with no additional S3 traffic. An
operator that is *non-local* — needs the whole upslope catchment — cannot, because catchments cross
the 1° tile boundary and the neighbouring tiles are not read. That single distinction sorts four of
the five metrics into "free" and isolates TWI as the one that breaks the budget.

---

## Verdict (one paragraph)

**Four of the five ride for free on the read ADR-0001 already pays for; the fifth (TWI) does not, and
should be replaced by a local proxy.** Elevation, Slope, Local relief and Geomorphons are all *local
neighbourhood operators*: each window's value depends only on a bounded halo of pixels, so all four
can be computed for every window directly from the per-tile ×8 overview array that §4 already reads
for TRI — **the one-read property survives for the whole set minus TWI**. TWI = ln(a/tanβ) is
fundamentally different: its numerator `a` (specific upslope contributing area) is a **flow-accumulation
quantity that depends on the entire upslope catchment**, which routinely crosses 1° tile boundaries,
and every standard implementation additionally requires a **globally depression-filled** DEM as a
preprocessing pass. True basin-scale TWI is therefore incompatible with independent windowed COG reads
in this budget. Recommendation: **drop true TWI; approximate the flat-wet axis with a large-window
Topographic Position Index (Weiss 2001) computed on the same ×8 overview** (valley bottoms read as
strongly negative TPI), backed by the Geomorphon valley/hollow/footslope/pit classes as a categorical
signal — both are purely local and piggyback on the existing read. Two correctness caveats apply to
the local metrics regardless: the **exact-0 ocean fill** injects artificial coastal cliffs into
slope/relief/geomorphon on kept coastline windows, and the **poleward-anisotropic GSD** breaks
`gdaldem`'s single-scalar `-s` scale for slope (relief is scale-free and immune).

| Metric | Data footprint | Overview OK? | Verdict | One-read property |
|---|---|---|---|---|
| **Elevation** | single window (already the stored per-tile mean) | yes (free) | **FEASIBLE-AS-IS** | preserved (already stored) |
| **Slope** | local 3×3 | yes (~240 m); full-res if fine slope wanted | **FEASIBLE-AS-IS** (needs anisotropic dx/dy, not scalar `-s`) | preserved |
| **Local relief** | windowed max−min (~km window) | yes; scale-free, no GSD correction | **FEASIBLE-AS-IS** | preserved |
| **Geomorphon class** | local, line-of-sight to radius L (small halo) | yes on ×8 overview; halo intra-tile except at tile edge | **FEASIBLE-AS-IS** (overview; tile-edge halo caveat) | preserved (whole-tile overview already read) |
| **TWI (true, ln a/tanβ)** | **whole upslope catchment; crosses tiles; needs filled DEM** | no — non-local + global preprocess | **NEEDS-GLOBAL-PREPROCESS → RECOMMEND-PROXY** | **broken** |
| **↳ TWI proxy** (large-window TPI + geomorphon valley classes) | local windowed | yes on ×8 overview | **FEASIBLE-AS-IS** (proxy) | preserved |

*Established facts carry citations; the mapping onto DEMAE's specific tiled-COG budget is my reasoned
inference and is flagged as such throughout.*

---

## 1. Why "local vs non-local" is the whole game here

ADR-0001 §4's efficiency rests on a subtle point worth making explicit. The scoring budget is one ×8
overview read **per COG**, and that read returns the *entire* tile downsampled to ~450² (≈240 m
effective GSD; fewer columns poleward per the recon note's band table). `stratify.py` already exploits
this: `tri_riley()` runs a per-pixel 3×3 operator over the whole overview array, and `window_tri()`
then reduces it over each window's overview footprint. The cost of scoring *N* windows in a tile is one
read, amortised.

That amortisation works for **any operator whose output at a pixel depends only on a bounded
neighbourhood** — a 3×3 stencil (slope, TRI, roughness), a fixed-radius annulus (TPI), or a
line-of-sight out to a search radius L (geomorphon). For all of these, the whole-tile overview array is
sufficient; the only pixels an operator ever wants that aren't in the array are those *just past the
tile edge* (a "halo"), and even that only affects the outermost windows.

It fails for operators that are **non-local** — where a pixel's value can depend on arbitrarily distant
pixels along a flow path. Flow accumulation is the canonical example, and TWI is built on it. No amount
of overview cleverness rescues a non-local operator from needing data the budget never reads. The rest
of this note is that distinction applied metric by metric.

## 2. Elevation — FEASIBLE-AS-IS (already free)

Elevation is 0th-order: the raw height field, no neighbourhood at all. ADR-0001 §6/§8 already computes
and stores the per-tile **mean** (and std) over land pixels for normalization, and the manifest records
it (§ "Consequences"). A robust per-window height statistic (median, per the prior note §6) is a pure
reduction over pixels already in hand — **zero marginal reads**.

- **Correctness caveat (datum):** values are metres above the **EGM2008 geoid** (recon §4), not the
  ellipsoid, and the product is a **DSM** (canopy/buildings included). Absolute-altitude comparisons
  across the corpus are geoid-relative; this is a labelling note, not a compute obstacle.
- **Ocean fill:** exact-0 pixels are already excluded from the land statistics (ADR-0001 §6), so the
  stored mean is unpolluted.

*Inference:* trivially within budget; nothing to add.

## 3. Slope — FEASIBLE-AS-IS, with a GSD-correctness caveat

Slope is 1st-order gradient magnitude over a **local 3×3** neighbourhood (`gdaldem slope`, Horn 1981
estimator; GDAL docs). Being a 3×3 stencil, it is exactly as local as the TRI already computed in
`stratify.py`, so it **piggybacks on the existing ×8 overview read** and preserves the one-read
property. Whether to compute it on the ×8 overview (~240 m, captures broad steepness) or at full
resolution (~30 m, captures fine steepness) is a modelling choice, not a budget one — on the overview
it is free; at full-res it would ride the materialization reads (§8).

- **Correctness pitfall — anisotropic GSD breaks the scalar `-s`.** `gdaldem` on an EPSG:4326 raster
  needs the vertical:horizontal unit ratio via `-s`; its documented value is **`scale=111120`** for
  "vertical units metres" (GDAL docs). But `111120` is a *single scalar* that assumes **square degrees
  on the ground**. In GLO-30 that assumption is false: N–S spacing is ~30.9 m everywhere, while E–W
  spacing is `lon_spacing″ · 30.9 m · cos(lat)` and the column count itself steps down poleward (recon
  §6). Near 50° the E–W ground distance is ~1.6× the N–S, so a single-scalar slope over-/under-states
  the E–W gradient by up to ~1.6×. **Correct handling** is to feed the gradient explicit, latitude-
  dependent `dx` and `dy` in metres (i.e. compute ∂z/∂x and ∂z/∂y with true ground spacings rather than
  relying on GDAL's one scale factor). *This is my inference from the GDAL scale semantics plus the
  recon poleward-shrink table; the anisotropy is a known GLO-30 fact, the slope-error magnitude is my
  arithmetic.*
- **Ocean fill:** on kept coastline windows the 0-m sea abutting real land creates an artificial cliff,
  inflating slope at the shoreline. Mitigate by masking exact-0 pixels before the gradient (consistent
  with §6's land-only stats), or accept it as a coastal-edge signal.

*Verdict:* FEASIBLE-AS-IS on the overview (free); the correction is a formula detail, not a read.

## 4. Local relief — FEASIBLE-AS-IS (and the most robust of the five)

"Local relief" = windowed elevation **range** (max − min). Note a terminology fork the prior note
already flagged: `gdaldem roughness` computes max−min over a **3×3** only ("the largest inter-cell
difference of a central pixel and its surrounding cell"; GDAL docs), whereas "local relief" in the
geomorphometric sense means the range over a **larger (~km) window**. Both are pure order-statistic
reductions over a bounded window — trivially local, trivially cheap, and computable at whatever window
size is wanted directly from the ×8 overview array. **Preserves the one-read property.**

- **Advantage — scale-free.** Because it is a pure vertical range (metres), local relief is **immune
  to the anisotropic-GSD problem** that afflicts slope: no `dx/dy`, no `-s`, no cos(lat) correction.
  On the ×8 overview a ~km window is only a few pixels (~4–5 px at 240 m), so if a genuinely ~km-scale
  relief is wanted the overview is adequate; a finer window needs full-res.
- **Ocean fill:** same coastal caveat — a window straddling 0-m sea and high coast shows inflated
  range. Mask exact-0 first if that is undesirable.

*Verdict:* FEASIBLE-AS-IS; the safest metric to compute on the overview.

## 5. Geomorphon landform class — FEASIBLE-AS-IS on the overview (with a halo caveat)

Geomorphons (Jasiewicz & Stepinski 2013) classify each cell by eight **line-of-sight** zenith/nadir
comparisons out to a **search radius L**, yielding a ternary 8-tuple that collapses to 498 patterns →
**10 landform classes** (flat, peak, ridge, shoulder, spur, slope, hollow, footslope, valley, pit)
(GRASS `r.geomorphon` docs). Two parameters set cost and scale:

- **Search radius L** — "the maximum allowable distance for calculation of zenith and nadir angles",
  in **cells** by default or metres with `-m` (GRASS docs). L is where the **halo** comes from: the
  method reads pixels out to L along eight directions, so it needs data L pixels beyond each focal cell.
  Larger L = "terrain classification from a higher and wider perspective"; smaller L = local view
  (GRASS docs). Because line-of-sight lets a nearer high cell *occlude* farther ones, the method is
  **self-adapting to local scale up to L** — you set a ceiling, not a fixed scale (Jasiewicz & Stepinski
  2013; GRASS docs).
- **Flat threshold t** — the zenith/nadir angle difference (degrees) that counts as "flat"; default
  examples use **t = 1°**. The docs warn that at **coarse resolution the threshold should shrink**:
  "If resolution of the map is low (more than 1 km per cell) threshold should be very small ... because
  on such distance 1 degree of difference means several meters" (GRASS docs). At the ~240 m overview
  this is not yet the >1 km regime, so t≈1° is still reasonable, but it is a knob to tune, not leave.

**Why it still rides the existing read (my inference).** The crucial point is that ADR-0001 already
reads the **whole tile's** ×8 overview, so a *modest* L (say a few hundred metres → a handful of 240 m
pixels) has its entire halo **inside that same array** for every interior window. Geomorphons therefore
compute for all of a tile's windows from the single scoring read — **one-read property preserved** —
*except* for windows touching the 1° tile edge, whose halo spills into the neighbour tile that the
budget does not read. Options for the edge: (a) accept a slightly degraded class on the outermost
~L-pixel border (cheap, and the overview border is a tiny fraction of pixels), or (b) read a thin
overview halo from the (up to 8) neighbour tiles — cheap on the overview but it *does* add reads.

- **Radius sizing:** at 30 m full-res a landform-scale L of, say, 300–600 m is 10–20 cells; on the
  240 m overview the same physical L is only ~1–3 cells — i.e. the overview is the *natural, cheap*
  resolution to run geomorphons for broad landform context, and the full-res 224² window is actually
  **too small** to host a large-L geomorphon on its own without a halo. This argues for computing the
  geomorphon on the **overview**, not per full-res window.
- **Anisotropic GSD:** in `-m` (metre) mode the tool assumes square cells; the poleward E–W stretch
  means a metre-specified L is not isotropic on the ground. Using **cell units** (default) sidesteps
  the metre assumption but then L is a slightly different ground distance E–W vs N–S. Minor for a
  categorical output; note it.
- **Output is categorical (10 classes)** — the manifest stores an integer class per window (or the
  modal/plurality class over the window), not a float. Flag for the storage schema.
- **Ocean fill:** a window's 0-m sea is dead-flat and will classify as `flat`; the artificial coastal
  cliff can read as `slope`/`ridge`. Mask exact-0 before classifying if a clean land class is wanted.

*Verdict:* FEASIBLE-AS-IS on the ×8 overview; the only budget risk is an optional neighbour-tile halo
for edge windows, which is cheap-but-nonzero. Recommend overview computation with edge-border acceptance.

## 6. TWI — the crux: NEEDS-GLOBAL-PREPROCESS, hence RECOMMEND-PROXY

TWI = **ln(a / tan β)**, where `a` is **specific upslope contributing area** (upslope area per unit
contour length) and β is local slope (Beven & Kirkby 1979). The β/tan β half is local and cheap (it is
just slope, §3). **The entire difficulty is `a`.** It is a **flow-accumulation** quantity, and flow
accumulation is **non-local by construction**: a cell's contributing area is the count/weight of *every*
cell whose flow path eventually passes through it. Two independent facts make true `a` incompatible with
independent windowed COG reads:

### 6.1 Contributing area crosses tile boundaries (edge contamination)

Every authoritative flow-routing tool documents that a clipped domain gives wrong accumulation near its
edges, because upslope area drains in from outside:

- **GRASS `r.watershed`:** cells fed from outside the region get **negative accumulation** flags —
  "Negative numbers indicate that those cells possibly have surface runoff from outside of the current
  geographic region ... cannot have their surface runoff ... calculated accurately", and the remedy is
  to **"[expand] the computational region until the catchment area for the region of interest is
  completely included"** (GRASS `r.watershed` docs). For a 1° tile that means pulling in whatever
  neighbour tiles the catchment reaches — unbounded in the general case.
- **TauDEM D∞ Contributing Area** (Tarboton 1997 D-infinity flow): ships **edge-contamination checking
  on by default** because "a contributing area value may be underestimated due to grid cells outside of
  the domain not being counted ... when drainage is inwards from the boundaries", producing "streaks of
  'no data' values extending inwards from boundaries along flow paths" (TauDEM docs). Those streaks are
  precisely the cells whose TWI is unknowable from the tile alone.

A 1° GLO-30 tile spans ~110 km N–S; large rivers and their catchments (Amazon, Ganges, Mississippi)
have contributing areas orders of magnitude larger, so for valley-bottom cells — *exactly the
flat-wet terrain TWI exists to flag* — `a` is dominated by out-of-tile upslope area. A per-224²-window
or even per-1°-COG TWI is therefore **not meaningful** for the low-lying convergent cells that are the
whole point of the index. *(Established: the edge-contamination behaviour and its cause, cited above.
Inference: that this makes tile-local TWI meaningless specifically for valley bottoms in GLO-30's 1°
tiling.)*

### 6.2 Flow accumulation needs a globally depression-filled DEM (a preprocessing pass)

Independently of tiling, contributing area is only well-defined on a **hydrologically conditioned**
(depressionless) surface, so real DEMs must be pit-filled or breached first:

- **WhiteboxTools `WetnessIndex`** takes **specific contributing area (SCA) + slope** as inputs, and if
  starting from a DEM it "**must have been hydrologically corrected to remove all spurious depressions
  and flat areas ... using the BreachDepressionsLeastCost or FillDepressions tools**" (WhiteboxTools
  docs).
- **RichDEM** demonstrates `rd.FillDepressions(dem, epsilon=True)` **before** `rd.FlowAccumulation()`,
  because "epsilon depression-filling replaces a depression with a predictable, convergent flow pattern"
  (RichDEM docs).
- **GRASS `r.watershed`** avoids an explicit fill only by substituting its own **A^T least-cost search**
  over the whole surface (GRASS docs) — i.e. it still requires the entire connected surface, not a clip.
- **GRASS `r.topidx`** (the direct TWI tool) additionally **does not support lat/long** — "Lat/long
  projects are not supported" (GRASS docs) — so GLO-30's native EPSG:4326 would first need reprojection,
  which ADR-0001 §1 deliberately declined for the corpus.

Depression filling is a **global, whole-surface** operation: filling a pit can re-route flow far
downstream, so it cannot be done correctly on an isolated window or even an isolated tile. This is a
second, independent reason true TWI needs a global preprocessing pass over a mosaicked, conditioned DEM
— categorically outside "one windowed read per tile."

### 6.3 The ocean-fill trap (a third, GLO-30-specific hazard)

Even if one mosaicked tiles, GLO-30's **exact-0.0 ocean fill with no nodata sentinel** (recon §2) would
poison flow routing: a coastal 0-m sea is a vast dead-flat plain that a flow router either treats as one
enormous sink to be filled or as a flat to be resolved, dumping all coastal drainage into a spurious
attractor. Correct TWI would first have to mask the ocean as true nodata — which the pipeline can do
(the land mask exists, §7) but which underlines that TWI needs bespoke conditioning the other four
metrics don't. *(Inference from the ocean-fill fact + how flow routers handle flats/sinks.)*

### 6.4 Does a halo rescue it? Cost verdict

A finite neighbour-tile **halo** fixes edge effects only when the catchment fits within the halo. For
ridge/upper-slope cells a 1–2-tile halo would suffice; for valley bottoms it would not, because their
catchments are basin-scale. So a halo gives *correct TWI for the terrain that already reads as
non-flat and wrong TWI for the flat-wet terrain the index is for* — the worst possible failure mode for
a stratifier trying to **find** flat-wet terrain. Combined with the global depression-fill requirement,
the honest verdict is: **true basin-scale TWI is infeasible in the Phase-0 budget** (it is a
NEEDS-GLOBAL-PREPROCESS metric), and forcing it would mean building a conditioned global mosaic — a
different pipeline than ADR-0001's independent windowed reads.

### 6.5 Recommended cheap local proxy for the flat-wet axis

The prior note wants the flat-wet/hydrology axis represented; it does **not** require the physically
exact TWI to do so. The cheapest faithful proxy that stays inside the one-read budget is a **large-window
Topographic Position Index (Weiss 2001):** TPI = centre elevation − mean of a surrounding annulus
(`gdaldem TPI`; GDAL docs). Valley bottoms and floodplains read as **strongly negative TPI** at a broad
radius, which is the convergent-low-position signal TWI captures, *minus* the contributing-area weighting.
TPI is a **purely local windowed mean** — it needs only a bounded annulus, so it computes for every
window from the same ×8 overview array (one-read property preserved). A two-scale Weiss TPI (small +
large radius) additionally separates local hollows from broad valley floors.

Back it with the **Geomorphon valley / hollow / footslope / pit classes** (§5), which already give a
categorical "convergent low-position" flag for free from the same read. Together — continuous
large-window TPI + categorical geomorphon valley classes — they cover the flat-wet axis without any
flow accumulation, depression filling, mosaicking, or extra reads.

*Rejected proxy — the Downslope Index (Hjerdt et al. 2004; WhiteboxTools `DownslopeIndex`).* It is a
tempting "wetness-ish" measure (gradient to a point d metres lower along the flow path), but the docs
require a DEM "pre-processed to remove artifact topographic depressions and flat areas" **and** it
**traces the downslope flow path** across cells — so it inherits both the depression-fill preprocessing
and a (bounded but non-trivial) non-locality. It is strictly worse than TPI for this budget; not
recommended.

*Verdict:* true TWI = **NEEDS-GLOBAL-PREPROCESS**; adopt the **large-window TPI + geomorphon valley
classes proxy**, which is **FEASIBLE-AS-IS** and preserves the one-read property.

## 7. Budget-impact summary — does the one-read property survive?

**Yes, for the surviving set.** With TWI replaced by the TPI/geomorphon proxy, **every** retained
descriptor is a local neighbourhood operator computable from the per-tile ×8 overview array that
ADR-0001 §4 already reads for TRI and the ocean screen. No metric in the *recommended* set forces
(i) full-resolution reads during scoring, (ii) neighbour-tile halos (beyond the optional, cheap
geomorphon edge-border case), or (iii) a global preprocessing pass. Concretely:

- **No new scoring reads.** Elevation (stored), slope (3×3), local relief (windowed range), geomorphon
  (line-of-sight to modest L), and the TPI proxy (annulus mean) all read only the whole-tile overview
  already in memory. The §4 invariant — *one ×8-overview read = ocean screen + all scores for the
  tile's windows* — **holds for all of them.**
- **The only budget pressure is optional and cheap:** a thin overview halo from neighbour tiles if you
  want exact geomorphon/TPI classes on the outermost tile-edge windows. This is a ×8-overview read of
  ≤8 neighbours, not a full-res read, and can be skipped by accepting a degraded L-pixel border.
- **The one metric that would have blown the budget is TWI**, and it is dropped in favour of a proxy
  precisely to keep the property. Had true TWI been mandatory, it would have forced a global
  depression-filled mosaic (option iii) — a different pipeline.

**Full-resolution note:** if the team later wants slope/relief at true 30 m (not 240 m) for the
*materialized* corpus, those computations ride the **existing ~50k full-res 224² materialization reads**
(ADR-0001 "Consequences") — they are local 3×3 / windowed operators over pixels already read for
materialization, so still **no extra reads**, just extra compute on bytes already fetched. Geomorphon at
full-res is the exception: a large-L geomorphon does not fit inside a lone 224² window and would need a
halo, which is the reason §5 recommends computing it on the overview instead.

### Per-metric verdict table (restated)

| Metric | Verdict | Reads beyond ADR-0001 §4? |
|---|---|---|
| Elevation | FEASIBLE-AS-IS | none (already stored) |
| Slope | FEASIBLE-AS-IS (fix `-s` → anisotropic dx/dy) | none |
| Local relief | FEASIBLE-AS-IS (scale-free) | none |
| Geomorphon class | FEASIBLE-AS-IS on overview | none, except optional cheap edge halo |
| TWI (true) | NEEDS-GLOBAL-PREPROCESS → **drop** | would need global filled mosaic |
| TWI proxy (large-window TPI + geomorphon valley classes) | FEASIBLE-AS-IS | none |

## 8. Recommendations for ADR-0001 / the manifest

1. **Keep the five-axis intent, swap the TWI implementation for the TPI+geomorphon proxy.** Record a
   large-window TPI value (continuous) and the geomorphon class (categorical 10-way) per window; do not
   attempt true flow-accumulation TWI in Phase 0.
2. **Compute slope with explicit latitude-dependent `dx`, `dy`** (from the recon band table +
   cos(lat)), not `gdaldem -s 111120`; or accept the ×1.6 worst-case E–W error and note it. **Local
   relief needs no such correction** (scale-free) — prefer it where a GSD-robust ruggedness/amplitude
   signal is wanted.
3. **Compute the neighbourhood metrics on the ×8 overview**, matching where TRI is already computed, so
   the §4 one-read invariant is preserved verbatim. Handle tile-edge geomorphon/TPI windows by accepting
   a degraded L-pixel border (default) or an optional neighbour overview halo (if edge fidelity matters).
4. **Mask exact-0 ocean pixels before slope/relief/geomorphon/TPI** on kept coastline windows (reusing
   the §3 land mask) to avoid artificial-cliff and false-flat artifacts — consistent with §6's land-only
   normalization.
5. **Storage:** add a categorical geomorphon-class column and a continuous large-window-TPI column to
   the manifest schema (handoff to #5), alongside the existing TRI/stratum fields.

---

## References

Primary tool documentation:
- GRASS GIS. *r.watershed — Watershed basin analysis (flow accumulation, edge/negative-value handling,
  A^T least-cost search).* https://grass.osgeo.org/grass-stable/manuals/r.watershed.html
- GRASS GIS. *r.topidx — TOPMODEL topographic wetness index ln(a/tanβ); "Lat/long projects are not
  supported."* https://grass.osgeo.org/grass-stable/manuals/r.topidx.html
- GRASS GIS. *r.geomorphon — geomorphon landform classification (search radius L, skip, flat threshold,
  10 forms / 498 patterns).* https://grass.osgeo.org/grass-stable/manuals/r.geomorphon.html
- GDAL. *gdaldem — slope (Horn; `-s`/`scale=111120`), roughness (3×3 max−min), TRI (Riley/Wilson), TPI.*
  https://gdal.org/programs/gdaldem.html
- WhiteboxTools User Manual. *Hydrological analysis — WetnessIndex (SCA + slope inputs; DEM must be
  hydrologically corrected via BreachDepressions/FillDepressions), D8/DInf FlowAccumulation,
  FillDepressions, BreachDepressions.*
  https://www.whiteboxgeo.com/manual/wbt_book/available_tools/hydrological_analysis.html
- WhiteboxTools User Manual. *Geomorphometric analysis — Geomorphons, DownslopeIndex (Hjerdt 2004;
  requires depression-removed DEM, traces flow path), TPI.*
  https://www.whiteboxgeo.com/manual/wbt_book/available_tools/geomorphometric_analysis.html
- TauDEM. *D-Infinity Contributing Area (Tarboton 1997 D∞; default edge-contamination checking; "no
  data" streaks from boundaries).*
  https://hydrology.usu.edu/taudem/taudem5/help53/DInfinityContributingArea.html
- RichDEM. *Flow metrics (D8, D∞, Quinn 1991, Freeman, Holmgren) and epsilon depression-filling before
  FlowAccumulation.* https://richdem.readthedocs.io/en/latest/flow_metrics.html

Defining papers:
- Beven, K.J. & Kirkby, M.J. (1979). *A physically based, variable contributing area model of basin
  hydrology (TOPMODEL / TWI = ln a/tanβ).* Hydrological Sciences Bulletin 24(1):43–69.
  https://doi.org/10.1080/02626667909491834
- Quinn, P., Beven, K., Chevallier, P. & Planchon, O. (1991). *The prediction of hillslope flow paths
  for distributed hydrological modelling using digital terrain models (multiple-flow-direction area).*
  Hydrological Processes 5(1):59–79. https://doi.org/10.1002/hyp.3360050106
- Tarboton, D.G. (1997). *A new method for the determination of flow directions and upslope areas in
  grid digital elevation models (D∞).* Water Resources Research 33(2):309–319.
  https://doi.org/10.1029/96WR03137
- Jasiewicz, J. & Stepinski, T.F. (2013). *Geomorphons — a pattern recognition approach to
  classification and mapping of landforms.* Geomorphology 182:147–156.
  https://doi.org/10.1016/j.geomorph.2012.11.005
- Weiss, A. (2001). *Topographic Position and Landforms Analysis.* Poster, ESRI User Conference.
  https://www.jennessent.com/downloads/TPI-poster-TNC_18x22.pdf
- Horn, B.K.P. (1981). *Hill shading and the reflectance map (Horn slope estimator).* Proceedings of
  the IEEE 69(1):14–47. https://doi.org/10.1109/PROC.1981.11918
- Hjerdt, K.N., McDonnell, J.J., Seibert, J. & Rodhe, A. (2004). *A new topographic index to quantify
  downslope controls on local drainage.* Water Resources Research 40, W05602.
  https://doi.org/10.1029/2004WR003130

In-repo:
- [`docs/adr/0001-corpus-construction-recipe.md`](../adr/0001-corpus-construction-recipe.md) — read
  budget and the §4 one-read invariant this note tests.
- [`docs/research/copernicus-glo30-recon.md`](copernicus-glo30-recon.md) — tile layout, poleward column
  shrink, ocean=0 fill, overviews.
- [`docs/research/terrain-representativeness-metrics.md`](terrain-representativeness-metrics.md) — the
  prior note that chose these five descriptors (feasibility, not the choice, is this note's subject).
- [`src/dataset/stratify.py`](../../src/dataset/stratify.py) — how per-pixel TRI + the ocean screen are
  computed today on the ×8 overview (the pattern the surviving metrics extend).
