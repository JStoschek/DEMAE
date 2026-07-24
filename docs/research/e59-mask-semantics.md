# E59 — Mask semantics: is an edited/filled pixel trustworthy elevation for MAE pretraining?

**Ticket:** [What does an edited vs filled pixel mean for the elevation value? (#59)](https://github.com/JStoschek/DEMAE/issues/59) ·
**Map:** [edit/fill gate (#57)](https://github.com/JStoschek/DEMAE/issues/57) ·
**Sibling:** [E58 stratum-19 hollowing (#58)](https://github.com/JStoschek/DEMAE/issues/58) ·
**Feeds:** [loosen-vs-accept decision (#60)](https://github.com/JStoschek/DEMAE/issues/60)

Second of the two investigations feeding #60. E58 established that the dropped high-alpine
terrain is *real and wanted* but deliberately left open whether the elevation is *trustworthy*.
This doc closes that gap against primary sources: it does **not** re-derive the EDM/FLM code
vocabularies — those are pinned in [`copernicus-mask-spec.md`](copernicus-mask-spec.md) from the
Product Handbook i5.0 and taken as given here.

## Verdict in one line

A flagged pixel is **not** trustworthy as *native* TanDEM-X elevation — the flag reliably marks
exactly where the source radar failed (steep-terrain layover/shadow, snow/glacier decorrelation),
so the original pixel was garbage or void and the flag is a **real quality signal, not
cosmetic**. But editing leaves a *plausible* height, not a NaN: it measurably **improves** accuracy
(Alps 90 m RMSE **45 m → 12 m**). The severity split that matters for MAE is **external-infill
(EDM==2 ≡ FLM≥3) vs interpolation/smoothing (EDM 3/4)** — infill injects another sensor's
texture and artefacts an MAE would learn as false terrain, interpolation merely suppresses
detail — and that axis lives **inside EDM**, *not* along the EDM-vs-FLM mask split the current
gate implicitly draws. So `edited_max = 0.25` does conflate two severities, but the conflation is
**quantitatively minor in the alpine drops** (E58: 94% of drops are over *both* caps because
external infill fires in both masks at once; only 6% are non-infill-edit-only).

---

## 1. Why high-alpine terrain gets flagged — the flag marks a real source-sensor failure, not provenance

**The mechanism is intrinsic to X-band InSAR over steep and snow/ice terrain, and it is
documented in the DLR primary literature.**

- **Radar geometry in steep terrain.** SAR interferometry is *"especially challenging over steep
  and irregular mountain surfaces due to shadowing and foreshortening effects"* — steep faces are
  laid over, foreshortened, or shadowed, so no usable phase is returned. Wessel et al. (2018, the
  DLR GPS-validation primary) name the same three effects directly: *"steeper areas tend to have
  higher height errors, especially where SAR specific effects like foreshortening, layover and
  shadow influence the heights."*
- **Snow / glaciers.** The bistatic TanDEM-X pair avoids *temporal* decorrelation, but permanent
  snow and glacier ice are low-coherence volume scatterers. Wessel: *"Less coherent objects, like
  forest, snow or water, incorporate a high random height error and therefore result in a less
  accurate absolute height."* This low coherence is annotated per-pixel in the Height Error Map
  (HEM). So the exact surfaces that dominate E58's drops — glaciated High Mountain Asia and high
  northern latitudes — are the surfaces where the *raw* interferometric height is least reliable
  or absent.
- **Voids get infilled, not left.** Where layover/shadow/decorrelation produced no valid height,
  the pixel is a void that the editing chain fills from an external DEM. This is the operational
  meaning of EDM code 2 ("Infill of external elevation data"), which the mask spec established is
  *empirically identical* to `Σ FLM≥3` (all external fill sources). Rizzoli et al. (2017) note the
  acquisition strategy already threw *up to seven or eight* passes at difficult terrain to
  minimise voids before any infill — i.e. infill is the last resort after the sensor itself could
  not close the gap.

**The decisive evidence that the flag is *not* cosmetic:** an independent airborne-laser-scanning
(ALS) validation across the Alps, Carpathians and Pyrenees found the raw TanDEM-X DEM's *"large
errors ... typically found on steep ridges or in canyons ... were largely corrected in the
Copernicus DEM,"* and the Alps 90 m RMSE dropped from **~45 m (raw TanDEM-X) to ~12 m (edited
Copernicus)**. The editing/infill in high-alpine terrain is **repairing genuinely bad radar data**,
not relabelling good data for provenance. A pixel that is `EDM != 1` in this terrain almost never
carried a usable native height in the first place.

**Answer to sub-question 2:** high-alpine pixels are flagged for a reason that *would* harm MAE if
ignored — the underlying radar measurement was untrustworthy. Equivalently: the flag correctly
identifies where you *cannot* trust a native height. Whether the *replacement* value is good enough
for MAE is §2–§3.

## 2. Is the replacement value trustworthy? Fill sources vs native — and the texture problem

Editing makes the value *plausible* (§1: global RMSE improves), but "plausible height" and
"faithful terrain texture" are different bars, and a masked-reconstruction objective cares about
the second. Two failure profiles, by edit class:

**(a) External infill (EDM==2 / FLM≥3) — carries another sensor's error *and* its artefacts.**
The fill sources that dominate high-alpine / high-latitude fill are, in the mask spec's own count,
ASTER (FLM 3), SRTM (4/5/7), GMTED2010 (6), AW3D30 (9). Their mountain accuracy is materially
worse than native TanDEM-X and, critically, *slope-dependent*:

| Source | Reported vertical accuracy in mountainous / steep terrain | Artefact signature relevant to MAE |
|---|---|---|
| Native TanDEM-X (baseline) | RMSE ~3.3–8 m on slopes <5°, rising to ~8–19 m on slopes >35° | speckle; HEM-flagged low-coherence noise |
| ASTER GDEM | RMSE ~6.5–10 m (worst of the common global DEMs); most slope-sensitive | **"pit" artefacts tied to stack-number boundaries, striping, residual cloud/stacking anomalies** (NASA/METI/USGS validation report) |
| AW3D30 | RMSE ~3.5–9 m (best optical) | **diagonal striping**, pronounced in High Mountain Asia |
| SRTM (GL1) | RMSE ~3.9–8.5 m | C-band; its *own* layover/shadow voids in steep terrain |
| GMTED2010 | coarse (~7.5″/15″ aggregate) — a smoothed legacy composite | resolution far below 30 m; injects low-frequency bias |

The load-bearing point for MAE is **not** the RMSE column — it is the artefact column. ASTER's
stack-boundary pits and both ASTER's and AW3D30's striping are **structured, spatially-coherent
patterns**, not zero-mean noise. A reconstruction pretext learns structured patterns as if they
were terrain. This directly confirms the concern the mask spec handed forward (*"filled pixels
carry another sensor's texture the MAE could learn as artifacts"*): for infilled pixels it is
**verified, not merely suspected** — the fill sources have documented, named textures.

**(b) Interpolation / smoothing of native data (EDM 3 "Interpolated", 4 "Smoothed").** These are
*not* foreign-sensor injections. Interpolation closes a small void from surrounding *native*
TanDEM-X pixels; smoothing suppresses interferometric noise. The value stays in the TanDEM-X error
family and carries **no foreign texture** — its failure mode is the opposite: **loss of
high-frequency detail** (a locally over-smooth surface), which for a reconstruction target is a
*milder* corruption than a striped/pitted foreign patch. E58's gallery is consistent with this: at
the extreme high-EDM tail a minority of tiles *"show visible smoothing consistent with heavy
interpolation,"* i.e. the visible degradation is smoothing, not garbage.

**Answer to sub-question 3 (severity, established vs inferred):**
- **Established (primary):** external-infill pixels carry a different sensor's height with that
  sensor's documented artefacts; those sources are less accurate in mountains and their errors are
  structured. Interpolation/smoothing keeps the value in the native TanDEM-X family with no foreign
  texture.
- **Reasoned inference:** for an MAE reconstruction objective, **external infill (EDM==2 / FLM≥3)
  is the more corrupting class** — structured foreign texture is worse to learn than detail loss —
  so *if* severities were ranked: `external infill (worst) > interpolation ≈ smoothing (mild) >
  not-edited (clean)`. Among fill sources, ASTER (pits+striping) is the worst injection, GMTED2010
  the coarsest, AW3D30/SRTM the least-bad. This is inference from the accuracy/artefact literature,
  not a Copernicus-stated ranking.

## 3. Does `edited_max = 0.25` conflate two failure modes — and is a per-mask split the fix?

**Sub-question 1 — yes, but read the axis carefully.** The gate is
`(edm_edited_frac > 0.25) OR (flm_fill_frac > 0.25)`. Its **edit limb lumps the severe class
(EDM==2 external infill) with the mild class (EDM 3/4 interpolation/smoothing)** into one
`edited` fraction. That *is* a conflation of genuinely different severities (§2). So the premise
of #59's sub-question 1 is correct in principle.

**But the conflation does not run along the EDM-vs-FLM axis, and that is the trap for #60.** The
mask spec's identity **`EDM==2 ≡ Σ FLM≥3`** means external infill is *the same pixels* in both
masks — it fires the EDM limb and the FLM limb simultaneously. E58 measured exactly this on the
real drops:

| E58 finding (stratum-19 drops) | value | reading |
|---|---|---|
| over EDM cap | 11,787 (100%) | edit limb catches everything |
| over FLM cap | 11,036 (94%) | fill limb = the external-infill subset |
| over **both** | 11,036 (94%) | infill double-counts across both masks |
| **EDM-only** (edit fires, fill does not) | **751 (6%)** | the *non-infill* edit: interpolation/smoothing |
| FLM-only | 0 | fill is never sole cause |

So splitting the gate **EDM-vs-FLM does not separate severe from mild** — the severe class
(external infill) is present in *both* limbs at once, and the FLM limb alone would still catch 94%
of the drops. The severity split that actually matters lives **inside EDM**: `EDM==2` (external
infill, severe) vs `EDM ∈ {3,4}` (interpolation/smoothing, mild). Along *that* axis, only the 751
EDM-only windows (6%) are the mild, arguably-recoverable class; the other 94% are external infill,
which §1–§2 say is the genuinely-compromised class you probably should keep dropping.

**Answer to sub-question 3, restated for the knob:** a rule that "treats EDM and FLM differently"
as two combined fractions is the **wrong instrument** — it cannot isolate severity, because
external infill is definitionally in both. A rule that distinguishes **infill (EDM==2 / FLM≥3) from
non-infill edit (EDM 3/4)** *is* the severity-aligned split — but in the alpine drops it recovers
only ~6%, matching E58's independent conclusion that touching edit-alone recovers 751 windows and
touching fill-alone recovers nothing.

## 4. Confidence ledger

- **Established (primary sources, directly read or agency validation reports):** the layover/
  foreshortening/shadow and low-coherence-snow mechanism (Wessel 2018); infill-after-void with 7–8
  passes over difficult terrain (Rizzoli 2017); editing *improves* alpine accuracy 45 m → 12 m and
  corrects ridge/canyon errors (RSE mountain-environment study); ASTER pit/stripe artefacts and its
  worst-in-class mountain accuracy (NASA/METI/USGS ASTER GDEM validation; comparative DEM
  literature).
- **Established (this repo's prior primary work, cited not re-derived):** EDM/FLM vocabularies and
  the `EDM==2 ≡ Σ FLM≥3` identity (mask spec); the 94%/6%/0% co-location split (E58).
- **Reasoned inference (mine, flagged as such):** the MAE-specific severity ranking
  `external infill > interpolation ≈ smoothing > clean`, resting on "structured foreign texture is
  worse for a reconstruction target than detail loss." No Copernicus or MAE source ranks these
  directly; it follows from the artefact evidence plus the nature of the pretext task.

## What this hands to the decision (#60)

- **The flag is a true quality signal, so "the terrain looks fine" (E58 §3) is necessary but not
  sufficient.** The pixels are flagged because the *source radar* failed there; the surviving value
  is a repair, not a native measurement. Admitting them means training the MAE partly on repaired
  surfaces.
- **If #60 accepts (keeps dropping):** well-justified — external infill (94% of the drops) carries
  another sensor's structured artefacts (ASTER pits/stripes, AW3D30 stripes) that a reconstruction
  objective would learn as false terrain. This is the class you most want out.
- **If #60 loosens:** the *only* severity-defensible loosening is to admit the **non-infill edit**
  class (`EDM ∈ {3,4}`, interpolation/smoothing — native-family value, no foreign texture) while
  **keeping the external-infill screen** (`EDM==2 / FLM≥3`). That is a split *inside EDM*, not the
  EDM-vs-FLM combined-fraction split the current gate draws. Its ceiling is small — E58's 751
  EDM-only windows (6% of the alpine drops) — so a per-stratum cap that simply raises 0.25 (E58 §2)
  would sweep the severe infill class back in and is **not** severity-aligned.
- **Net position for the pivot #59 was created to settle:** an edited/filled pixel is a *plausible
  but non-native* height; edit and fill are materially different in severity **in the abstract**
  (infill worst, interpolation/smoothing mild), but the EDM-vs-FLM mask boundary does **not** trace
  that severity line — infill-vs-not, which cuts through EDM, does.

## Sources

- **Copernicus DEM Product Handbook i5.0** — EDM/FLM code vocabularies (Tables 6/7), infill/edit
  definitions. Cited via [`copernicus-mask-spec.md`](copernicus-mask-spec.md); primary PDF:
  https://dataspace.copernicus.eu/sites/default/files/media/files/2024-06/geo1988-copernicusdem-spe-002_producthandbook_i5.0.pdf
- **Wessel, B. et al. (2018), "Accuracy assessment of the global TanDEM-X Digital Elevation Model
  with GPS data,"** *ISPRS J. Photogramm. Remote Sens.* 139:171–182 (DLR, primary; read directly).
  Layover/foreshortening/shadow in steep terrain; snow/low-coherence high random error; HEM
  annotation; steep-slope/high-altitude points underrepresented in GPS validation (numbers
  optimistic for mountains). doi:10.1016/j.isprsjprs.2018.02.017 ·
  https://elib.dlr.de/120420/1/2018_Wessel-etal_TDX_DEM_AccuracyAssessment_ISPRS_JPRS.pdf
- **Rizzoli, P. et al. (2017), "Generation and performance assessment of the global TanDEM-X
  digital elevation model,"** *ISPRS J. Photogramm. Remote Sens.* 132:119–139. Global LE90 ≈ 3.5 m;
  difficult terrain acquired up to 7–8 times; voids from radar shadow/layover masked out before
  infill. doi:10.1016/j.isprsjprs.2017.08.008 ·
  https://www.sciencedirect.com/science/article/pii/S092427161730093X
- **"Accuracy assessment of the global TanDEM-X digital elevation model in a mountain
  environment" (2020),** *Remote Sensing of Environment* 241:111724. Alps/Carpathians/Pyrenees vs
  ALS; raw-TanDEM-X large errors on ridges/canyons corrected by Copernicus editing; Alps 90 m RMSE
  45 m → 12 m. doi:10.1016/j.rse.2020.111724 ·
  https://www.sciencedirect.com/science/article/abs/pii/S0034425720300936
- **"Vertical Accuracy of Freely Available Global DEMs (ASTER, AW3D30, MERIT, TanDEM-X, SRTM,
  NASADEM)" (2020),** *Remote Sensing* 12(21):3482. Slope-dependent RMSE; TanDEM-X ~3.3–8 m
  (<5°) → ~8–19 m (>35°); ASTER most slope-affected. https://www.mdpi.com/2072-4292/12/21/3482
- **ASTER GDEM Version 2 — Summary of Validation Results** (NASA/METI/USGS/ERSDAC, primary agency
  report). Documented "pit" artefacts tied to stack-number boundaries, striping, cloud/stacking
  residual anomalies; artefacts introduce large local elevation errors.
  https://lpdaac.usgs.gov/documents/220/Summary_GDEM2_validation_report_final.pdf
- **"Applicability of Data Acquisition Characteristics to the Identification of Local Artefacts in
  Global DEMs: Comparison of the Copernicus and TanDEM-X DEMs" (2021),** *Remote Sensing*
  13(19):3931 — Copernicus-vs-TanDEM-X artefact localisation via COV/HEM/edit layers (cited for
  scope; full text not retrieved). https://www.mdpi.com/2072-4292/13/19/3931
- **Prior repo primaries (not re-derived):** [`copernicus-mask-spec.md`](copernicus-mask-spec.md)
  (EDM/FLM vocab + `EDM==2 ≡ Σ FLM≥3`); [`e58-stratum19-hollowing.md`](e58-stratum19-hollowing.md)
  (94%/6%/0% edit-fill co-location on the real drops); ADR-0002 §2 (Copernicus ±4 m vertical
  accuracy, low-quality-tile drop rationale).
