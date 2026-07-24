# ADR 0006 — Accept the EDM/FLM gate's high-alpine hollowing of stratum 19

- **Status:** Accepted
- **Date:** 2026-07-24
- **Context ticket:** "Decide: loosen the edit/fill gate for high-alpine terrain, or accept the hollowing?" ([issue #60](https://github.com/JStoschek/DEMAE/issues/60)) — the culminating decision of the **edit/fill-gate wayfinder map** ([#57](https://github.com/JStoschek/DEMAE/issues/57))
- **Seeded by:** Gate E's **FAIL** ([#54](https://github.com/JStoschek/DEMAE/issues/54)) on the E2.5 attrition-by-stratum criterion — specifically the `edited` × stratum-19 firing (18.1% within-stratum drop, 12.1× the corpus rate), the "edited-pixel gate hollowing out a terrain family" shape plan risk #3 named in advance
- **Inputs (both primary, committed):** [`e58-stratum19-hollowing.md`](../research/e58-stratum19-hollowing.md) ([#58](https://github.com/JStoschek/DEMAE/issues/58)) — the hollowing characterization; [`e59-mask-semantics.md`](../research/e59-mask-semantics.md) ([#59](https://github.com/JStoschek/DEMAE/issues/59)) — the mask-severity semantics; [`e2-validation-report.md §5`](../research/e2-validation-report.md) — the firing record
- **Depends on:** ADR-0002 §2 (why near-constant / low-quality tiles are dropped — the sibling rationale this extends)

> **Scope in one line.** This ADR accepts **only** the EDM/FLM gate's hollowing of **stratum 19**. The four `degenerate` (flatness) firings from the same Gate E FAIL — strata 2, 5, 7, 17 — are **not** decided here and remain **open** for their own later effort ([Out of scope](#out-of-scope)). Do not read this as dispositioning Gate E's FAIL as a whole.

## Context

The `edited` quality gate — `(edm_edited_frac > 0.25) OR (flm_fill_frac > 0.25)`,
`edited_max = 0.25`, applied in Phase C — drops a window when more than a quarter of its
pixels are flagged edited (EDM) or filled (FLM) in the Copernicus masks. Gate E's E2.5
criterion ([`e2-gate-bar.json`](../research/e2-gate-bar.json)) fired on stratum 19, the
high glaciated-alpine terrain family (relief 310–1,250 m, elev 3,000–5,000 m): the gate
removes **11,787 of 65,225** would-be members (**18.1%**, **12.1×** the 1.5% corpus
`edited` rate). That firing is a *finding*, not a defect — the corpus faithfully
implements the design; the design itself is on trial — so the pre-registered consequence
was to stop the Phase-E map and open a fresh map ([#57](https://github.com/JStoschek/DEMAE/issues/57))
whose destination is exactly this decision: **loosen the gate for high-alpine terrain, or
accept the hollowing as correct?**

Two investigations were commissioned to settle it before any direction was chosen (the
pre-registration discipline that made Gate E honest carries over — a rule chosen after
seeing a re-drawn corpus is goalpost-moving):

- **E58 ([#58](https://github.com/JStoschek/DEMAE/issues/58)) — is the dropped terrain
  real, and does the loss starve selection?** The 11,787 drops are the **same
  high-alpine terrain family as the kept tiles** — if anything higher and
  higher-relief (dropped `local_relief` p50 1898 vs kept 997; `slope_mean` 32.8 vs
  22.4) — the most glaciated tail of an already-alpine stratum, not a corrupted or
  degenerate population. Edit and fill are **co-located**: 94% of drops are over *both*
  the EDM and FLM caps, 6% are EDM-only, **0% are FLM-only** — so the knob is a
  combined edit-or-fill screen, never a per-mask split. Critically, the loss does **not
  starve selection**: stratum 19 still keeps **53,437** candidates against an allocation
  of **1,106** (48.3×). The only surviving risk is *representativeness* — whether
  shaving the glaciated extreme skews which alpine terrain the 1,106 sample from — not
  availability.

- **E59 ([#59](https://github.com/JStoschek/DEMAE/issues/59)) — is an edited/filled
  pixel trustworthy elevation for an MAE reconstruction objective?** A flagged pixel is
  a **plausible but non-native** height: the flag reliably marks where the source radar
  failed (steep-terrain layover/foreshortening/shadow, snow/glacier decorrelation), so
  it is a **real quality signal, not cosmetic** — editing measurably *improves* alpine
  accuracy (Alps 90 m RMSE ~45 m → ~12 m). The severity axis that matters for an MAE is
  **external infill (`EDM==2` ≡ `FLM≥3`) vs interpolation/smoothing (`EDM ∈ {3,4}`)**:
  external infill injects *another sensor's* structured artefacts (ASTER stack-boundary
  pits and striping, AW3D30 diagonal striping) that a reconstruction pretext learns as
  **false terrain**; interpolation/smoothing merely loses high-frequency detail and
  keeps the value in the native TanDEM-X error family. That severity line runs **inside
  EDM**, *not* along the EDM-vs-FLM split the gate implicitly draws — and E58's 94% / 6%
  / 0% split places 94% of the alpine drops in the severe external-infill class, the
  class you most want out.

## Decision

**The `edited` gate stands unchanged (`edited_max = 0.25`, combined EDM/FLM). Stratum
19's 18.1% hollowing is accepted as correct data-quality behaviour. No pipeline re-run
follows from this decision.** The Gate E `edited` × 19 FAIL is thereby dispositioned:
reviewed, and the design deliberately left in place.

Three legs carry it, in the order they close the decision:

### 1. Scarcity is not in play

Stratum 19 keeps **53,437** candidates for an allocation of **1,106** — a **48.3×**
oversupply (E58). Nothing downstream is starved; the FSCS sampler draws its 1,106 from a
pool two orders of magnitude larger than it needs. Loosening cannot be justified as
"recovering tiles we lack," and the representativeness worry (leg unresolved by scarcity)
is a *thinning of an oversupplied tail*, not its removal — the kept pool still spans the
dropped tiles' own p10–p90 relief band (1,372–2,546 m). The extreme-glaciated tail is
thinner, not gone, and a 48× pool absorbs that.

### 2. The direction the evidence allows recovers almost nothing worth having

E59 shows the only severity-*defensible* loosening is to admit the **non-infill edit**
class (`EDM ∈ {3,4}`, interpolation/smoothing — native-family value, no foreign texture)
while **keeping the external-infill screen** (`EDM==2` / `FLM≥3`). E58 sizes that class at
the **751 EDM-only windows — 6%** of the alpine drops. The other **94% is external
infill**: foreign-sensor structured texture an MAE would learn as terrain (E59). So the
gate is *removing exactly the class you most want out*, and only 6% of what it removes is
even a candidate for principled recovery. A per-mask EDM-vs-FLM split cannot isolate that
6% (external infill fires both limbs by the `EDM==2 ≡ FLM≥3` identity), and a plainly
raised combined cap sweeps the severe infill class back in — neither is severity-aligned.

### 3. Even the severity-clean 6% fails cost/benefit

Admitting the 751 non-infill windows is a **+1.4%** addition to a 53,437-strong pool
feeding a 48×-oversupplied allocation — it almost certainly does not change *which* tiles
are selected. Against that trivial yield, any gate change forces a full **C→D→E
re-materialization**: re-fit the k-means on a changed keep-set ([#44](https://github.com/JStoschek/DEMAE/issues/44)),
re-stratify, re-allocate — reopening everything Phase D locked (k=20, the allocation, the
frozen metric vector). Worse, the catalog stores only the **aggregate** `edm_edited_frac`;
isolating `EDM==2` from `EDM ∈ {3,4}` likely needs a **per-code mask recompute**, i.e. new
pipeline work, to recover those 751 windows at all. Full re-materialization (possibly
gated on a mask recompute) for a +1.4% pool bump in a 48×-oversupplied stratum is not a
trade worth making.

**Net:** the terrain is real and wanted (E58), but the flag is a true quality signal and
94% of the loss is the artefact class an MAE would mislearn (E59); the pool is not
starved; and the one principled loosening recovers 6% at the cost of a full re-run. The
gate's high-alpine bite is **correct behaviour**, not a defect to repair.

## Consequences

- **The gate is unchanged and no pipeline compute is spent.** `edited_max = 0.25` and the
  combined EDM/FLM screen stand as the committed baseline going forward. The
  C→D→E execution map that "loosen" would have chartered is **moot** and is not opened.
- **Plan risk #3's `edited` limb is realized and consciously accepted.** The plan
  ([`dataset-curation-plan.md §244`](../dataset-curation-plan.md)) named "quality gates
  hollowing out a terrain family (edited-pixel gate)" as a risk with the mitigation
  "loosen EDM cap per-stratum if needed." The E2.5 table caught the hollowing exactly as
  designed; the "if needed" test was applied and the answer is **not needed**. The risk's
  `edited` branch is closed as *accepted*, annotated in place.
- **The MAE trains partly on kept alpine tiles that press up to the cap.** Kept
  stratum-19 tiles reach `edm_edited_frac` p99 0.2366 — repaired-but-plausible surfaces
  under the 25% cap are admitted by design; only windows *over* the cap (dominated by
  external infill) are excluded. This is the intended balance: some editing is tolerated,
  a terrain family kept, while the heavily-infilled artefact surfaces stay out.
- **Gate E's `edited` × 19 FAIL is dispositioned; a future re-score should not re-raise
  it.** [`e2-validation-report.md §5`](../research/e2-validation-report.md) carries a
  disposition note pointing here. The pre-registered *scoring* of Gate E is untouched
  (that bar is a fixed input, not re-litigated); what is recorded is that the design it
  flagged was reviewed and kept.
- **The flatness firings are untouched and still open** — see below. Anyone acting on
  Gate E's FAIL must treat the `degenerate` × {2,5,7,17} pairs as unresolved.

## Out of scope

- **The `degenerate` / flatness firings (strata 2, 5, 7, 17).** A different *kind* of
  question from this one: those firings are confounded (a flatness gate biting flat strata
  is partly structural — [`e2-validation-report.md §5`](../research/e2-validation-report.md),
  "Three things Gate E has to weigh", point 3), the pre-registered payload could not even
  name a knob for them (the §215–228 table has no row for the land-std screen), and
  settling them needs the flatness screen added to the knob table plus an expected-ratio
  model that may conclude "no change." They are their own later effort and are **not
  accepted or rejected here.**
- **The C→D→E re-run and a new Gate E.** Under "accept" there is nothing to re-run; this
  ADR spends no pipeline compute. Had the decision been "loosen," the execution would have
  been a separate map — that map is moot.
- **Re-litigating Gate E's pre-registered bar.** The 3× / 10% E2.5 rule and
  `edited_max = 0.25` as the *committed baseline* are fixed inputs; this decision is about
  whether the gate's *design* should change going forward, not whether Gate E scored it
  correctly.

## Alternatives considered

- **Loosen via a raised combined cap (global or per-stratum, e.g. 0.25 → ~0.35–0.50).**
  Rejected: not severity-aligned. E58's continuum shows a modest raise is *plausible*
  (3,279 drops sit in 0.25–0.35), but E59 shows a raised combined cap sweeps the severe
  external-infill class (94% of the drops) straight back in — admitting exactly the
  foreign-sensor texture an MAE mislearns. Raising the cap trades the representativeness
  worry for a data-quality one that is worse for the pretext task.
- **Loosen via a per-mask EDM-vs-FLM split (treat edit and fill differently).** Rejected:
  the wrong instrument. External infill is `EDM==2 ≡ FLM≥3` — the *same pixels* in both
  masks — so it fires both limbs at once (E58: 94% over both caps, 0% FLM-only). A per-mask
  split cannot separate severe from mild; the FLM limb alone would still catch 94% of the
  drops.
- **Loosen via the one severity-clean rule (admit non-infill `EDM ∈ {3,4}`, keep the
  `EDM==2`/`FLM≥3` infill screen).** Rejected on cost/benefit, not principle (Decision
  §3): +1.4% pool yield into a 48×-oversupplied allocation, at the cost of a full C→D→E
  re-run likely gated on a per-code mask recompute. Left explicitly on the table and ruled
  out consciously rather than by omission.
- **Accept, but document nothing (let the closed ticket be the record).** Rejected: the
  gate stays as-is *by decision*, not neglect, and the risk it was chartered against
  deserves a discoverable record. Hence this ADR plus the in-place annotations, so a future
  reader — or a Gate E re-score — sees a reviewed decision rather than an open FAIL.
