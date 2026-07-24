# ADR 0007 — Accept the `degenerate` flatness gate's concentration on flat strata 2/5/7/17

- **Status:** Accepted
- **Date:** 2026-07-24
- **Context ticket:** "Decide: loosen the degenerate flatness gate for the flat strata, or accept the concentration?" ([issue #66](https://github.com/JStoschek/DEMAE/issues/66)) — the culminating decision of the **flatness-gate wayfinder map** ([#62](https://github.com/JStoschek/DEMAE/issues/62))
- **Seeded by:** Gate E's **FAIL** ([#54](https://github.com/JStoschek/DEMAE/issues/54)) on the E2.5 attrition-by-stratum criterion — specifically the `degenerate` (flatness, `land_std < 3 m`) firing on the four flat strata **2, 5, 7, 17** (within-stratum drop 30.1% / 51.9% / 31.7% / 29.7% = 3.04–5.32× the 9.8% corpus rate), the confounded second half of the same FAIL whose `edited` × 19 half is settled by [ADR-0006](0006-accept-edm-flm-hollowing-stratum19.md)
- **Inputs (all primary, committed):** [`e63-flatness-null-bar.json`](../research/e63-flatness-null-bar.json) ([#63](https://github.com/JStoschek/DEMAE/issues/63)) — the structural-null model + fire/no-fire rule, **pinned blind** before any number was computed (the [#47](https://github.com/JStoschek/DEMAE/issues/47)-analog); [`e65-flatness-null-model.md`](../research/e65-flatness-null-model.md) ([#65](https://github.com/JStoschek/DEMAE/issues/65)) — the model run (rate-signal); [`e64-flatness-drops.md`](../research/e64-flatness-drops.md) ([#64](https://github.com/JStoschek/DEMAE/issues/64)) — the junk-vs-informative characterization; [`e2-validation-report.md §5`](../research/e2-validation-report.md) — the firing record
- **Depends on:** ADR-0002 §2 (why near-constant / low-information tiles are dropped — the sibling rationale this extends); [ADR-0006](0006-accept-edm-flm-hollowing-stratum19.md) (the `edited` × 19 half of the same FAIL — the precedent this parallels)

> **Scope in one line.** This ADR accepts **only** the `degenerate` (flatness) gate's concentration on the flat strata **2, 5, 7, 17**. The `edited` × stratum-19 firing from the same Gate E FAIL is settled separately by [ADR-0006](0006-accept-edm-flm-hollowing-stratum19.md). Together the two ADRs disposition Gate E's FAIL in full. This ADR does **not** re-litigate Gate E's pre-registered bar ([Out of scope](#out-of-scope)).

## Context

The `degenerate` quality gate — `land_std < degenerate_land_std_m` (`= 3.0 m`), applied
in Phase C — drops a window when the standard deviation of its land elevation is under
3 m: a near-constant, low-information surface an MAE reconstruction objective cannot learn
from (ADR-0002 §2, the same rationale as the `water` and `zero_fill` screens). Gate E's
E2.5 attrition-by-stratum criterion ([`e2-gate-bar.json`](../research/e2-gate-bar.json))
fired it on the four **flat / low-relief** terrain families — stratum 2 (low-relief plains
with faint drainage), 5 (near-level plains / basins), 7 (high-altitude flats / plateaus),
17 (low-relief lowlands with drainage / water) — at within-stratum drop rates of
**30.1% / 51.9% / 31.7% / 29.7%**, i.e. **3.08× / 5.32× / 3.25× / 3.04×** the 9.8% corpus
`degenerate` rate. That firing is a *finding*, not a defect — the corpus faithfully
implements the design; the design itself is on trial — so the pre-registered consequence
was to stop the Phase-E map and open a fresh map ([#62](https://github.com/JStoschek/DEMAE/issues/62))
whose destination is exactly this decision: **loosen the flatness gate for the flat strata,
or accept the concentration as correct?**

**The confound is the whole problem.** The gate keys on `land_std` (window flatness).
`land_std` is *not* one of the six frozen metrics, but it correlates with two of them
(`slope_mean`, `local_relief`), and the k=20 strata are k-means cells in that six-metric
space. So a flatness gate concentrating in the flat strata is **partly structural, not
necessarily bias** — you cannot read the raw 5.32× as excess. Separating "excess bias"
from "expected structure" needs an *expected-ratio model* that the #54 verdict explicitly
**barred from being built post-hoc at Gate E** (a model chosen after seeing which pairs it
lights up is goalpost-moving). Building it, pre-registered, is what this map existed to do.
The #54 verdict pre-resolved three calls as fixed inputs: these are **Findings, not
Defects**; the **registered whole-catalog baseline** governs (so `degenerate` × 5 at 5.32×
is the robust core, surviving both baselines, while 2/7/17 at 3.04–3.25× are lower-confidence
and vanish under the sensitivity baseline); and the firings are **real-but-confounded**.
Its guidance: *"centre on `degenerate` × 5; carry 2/7/17 as lower-confidence."*

Two investigations, plus a rule pinned before either ran, were commissioned to settle it —
the same pre-registration discipline that made Gate E honest, carried over because the
accept/loosen call is a *threshold* question ("does observed exceed structural
expectation?") and a number already seen can be argued either way:

- **The rule, pinned blind ([#63](https://github.com/JStoschek/DEMAE/issues/63),
  [`e63-flatness-null-bar.json`](../research/e63-flatness-null-bar.json)) — the
  [#47](https://github.com/JStoschek/DEMAE/issues/47)-analog.** A **stratum-blind
  expected-ratio model** `P(degenerate fire | six frozen metrics)` — deliberately
  excluding both `land_std` (the gate's own key; including it makes the residual ~0 by
  construction) and `stratum_id` (stratum-blindness *is* the test). Its capacity trap — a
  saturated model always says accept (vacuous), a too-rigid model manufactures fake excess
  — is neutralised by a **pinned capacity ladder** (L0 logistic → L1 degree-2 → L2 splines
  → L3 cross-fitted GBM) plus two bracketing anchors (A_uniform = the structure-free 0.0976
  null; A_saturated = each stratum's own rate). The **two-tier fire rule**: stratum 5 is
  the *only* stratum that can trigger a LOOSEN signal, firing iff at **every** rung the
  residual is ≥ 10 pp **and** ≥ 1.5× **and** significant; strata 2/7/17 are computed
  identically but reported as **corroboration only**, never independent triggers — directly
  implementing the #54 guidance. Committed while #65 was still blocked, so no number could
  bias it.

- **The rate run ([#65](https://github.com/JStoschek/DEMAE/issues/65),
  [`e65-flatness-null-model.md`](../research/e65-flatness-null-model.md)) — is the
  concentration excess, or structure?** It imports the pinned bar and executes it blind.
  **Rate-signal: ACCEPT.**

- **The terrain characterization ([#64](https://github.com/JStoschek/DEMAE/issues/64),
  [`e64-flatness-drops.md`](../research/e64-flatness-drops.md)) — is the dropped terrain
  featureless junk, or informative, and does the loss starve selection?** The drop is a
  **stratum-dependent mix**, and — importantly — the terrain axis points the *opposite* way
  from the rate axis across strata.

## Decision

**The `degenerate` flatness gate stands unchanged (`degenerate_land_std_m = 3.0 m`, global,
no per-stratum override). The flat-strata concentration is accepted as correct
data-quality behaviour: it is at/within what the strata geometry alone predicts, and what
loss is genuinely informative sits in the strata the pre-registration deliberately
down-weighted, is not starved, and is flagged for re-test at scale (below). No pipeline
re-run follows.** The Gate E `degenerate` × {2,5,7,17} FAIL is thereby dispositioned:
reviewed, and the design deliberately left in place. With [ADR-0006](0006-accept-edm-flm-hollowing-stratum19.md)
settling the `edited` × 19 half, Gate E's FAIL is now dispositioned in full.

Four legs carry it, in the order they close the decision:

### 1. The rate axis: the concentration is structural, and the finding is capacity-proof

The pre-registered structural-null model tested whether stratum 5's headline **5.32×**
concentration exceeds what its six-metric geometry predicts. It does not. Stratum 5
(247,681 would-be members, 128,511 drops, observed rate **0.519**) has an expected rate of
**0.519–0.526 at every rung** — residual **−0.007 pp / 0.99× at L0**, **−0.001 pp / 1.00×
at L3**. Not one rung meets even a single magnitude floor, so under the pinned two-tier
rule the stratum-5 LOOSEN signal is **NO → ACCEPT**.

Decisively, this is **not** the vacuous "saturated model always says accept" artifact the
bar warned about. The signal holds at the **rigid L0 end**, where the binding (smallest-
residual) rung actually sits: L0 *over*-predicts (expected 0.526 > observed 0.519). A
too-rigid model was the failure mode that would *manufacture* fake excess in the joint-low
slope-&-relief corner where the flat strata live — and even it finds no excess. The ACCEPT
is therefore **capacity-choice-proof**: it does not depend on how much structure the
analyst chose to credit. The corroborating strata agree — largest residual is 17 at L0
(+2.79 pp), decaying to +0.08 pp by L3; 2 and 7 sit at ~0 throughout. The significance leg
is non-binding at the debug tier exactly as the bar disclosed. **The flat-strata
concentration is structure, not bias.**

### 2. The terrain axis splits from the rate axis — and the real informative loss is in the down-weighted strata

The characterization (#64) confirms the drop is **not** one population. It is
stratum-dependent, and the terrain-quality signal runs **opposite** to the rate-excess
signal across the four strata:

- **Stratum 5** — the decider, the largest rate excess (5.32×), the only pair surviving
  both baselines — drops **genuinely low-information smooth plains**: dropped `texture`
  (0.001) is **identical** to kept (0.001); the cut is purely on overall relief, slicing a
  continuum down toward zero (34% of the loss sits in the deepest-flat `land_std < 1 m`
  band). This is the **weakest** case for "informative terrain removed" — correctly
  targeted junk.
- **Strata 2 and 17** — the *marginal* excess (3.08× / 3.04×, both vanishing under the
  sensitivity baseline) — drop the **more-dissected, drainage-bearing** low-relief tiles
  (dropped `texture` 0.015 / 0.021 **>** kept 0.006 / 0.009; visible rivers, dendritic
  drainage, and incised valleys by eye). This is the real "informative terrain removed" —
  but it is confined to the strata the pre-registration made **corroborating-only**.
- **Stratum 7** drops a uniform polar **ice-plateau** surface whose high texture/entropy
  metrics are amplified sub-metre sastrugi/noise, not learnable landform — correctly
  targeted despite the metrics.

So the strongest ACCEPT-on-terrain (stratum 5) is the strongest LOOSEN-on-rate, and
vice-versa for 2/17. The genuine informative loss does not coincide with the genuine rate
excess in any single stratum, and where it *does* exist (2/17) it is exactly the signal the
committed rule ruled corroborating-only.

### 3. Scarcity is not in play; representativeness absorbs the thinning

No stratum is starved. Each keeps its would-be pool **≥ 53×** its allocation — stratum 5:
**118,638** kept vs **1,991** allocated (**59.6×**); 2: **65.2×**; 7: **53.5×**; 17:
**57.0×**. The FSCS sampler draws each allocation from a pool two orders of magnitude
larger than it needs. The recoverable ceiling is not even the headline drop but the
**sole-cause** count (stratum 5: 118,845; stratum 17 much less — 8,722 of its 50,046 drops
also fire `water`, which a flatness loosening would not readmit). And the surviving kept
2/17 pools still carry drainage structure (kept `texture` 0.006 / 0.009, non-zero), so
coverage-balanced selection can still surface dissected tiles for the 2/17 allocations. As
with stratum 19 under ADR-0006, the only live worry is *representativeness* — a thinning of
an oversupplied population, not its removal — and a 53–65× pool absorbs that thinning.

### 4. Honoring the pre-registration *is* the decision

The bar was committed blind precisely so a confounded firing could not be talked to either
verdict once the numbers were seen. The numbers came back structural and capacity-proof.
The decision ticket's own LOOSEN definition is **conjunctive** — "the model shows excess
bias beyond structure **and** the characterization shows informative terrain is being
dropped" — and the rate model found **no excess**, so the conjunction fails on its first
term. To LOOSEN anyway would mean acting on the 2/17 **by-eye terrain signal** — from
strata the committed rule down-weighted to corroboration-only, with no pre-registered
threshold of its own — to overturn a pre-registered ACCEPT and commission an expensive
C→D→E re-materialization into unstarved 65×/57× pools. That is precisely the post-hoc
goalpost-move the pre-registration exists to forbid. Accept is the disciplined verdict, and
the #54 verdict warned in advance it "may well be no-change."

## Consequences

- **The gate is unchanged and no pipeline compute is spent.** `degenerate_land_std_m = 3.0`
  (global, no per-stratum override) stands as the committed baseline going forward. The
  C→D→E execution map that "loosen" would have chartered is **moot** and is not opened.
- **The knob-table gap this map was chartered to close is closed.** The plan's §215–228
  knob table had a row for the `WBM water cap` and the `EDM/FLM cap` but **none** for the
  `land_std` flatness screen — the gap #54 flagged (`pairs_without_a_named_knob:
  ["degenerate"]`). A row for the **degenerate flatness screen (`land_std` cap)** is now
  added (default 3.0 m global; amendment trigger = an expected-ratio model showing excess
  beyond structure **and** a characterization showing informative loss — the exact test
  this map ran and that came back negative).
- **Plan risk #3's `degenerate` limb is realized and consciously accepted.** The plan
  ([`dataset-curation-plan.md §244`](../dataset-curation-plan.md)) named "quality gates
  hollowing out a terrain family" as a risk; ADR-0006 closed its `edited` limb, and this
  ADR closes the `degenerate`/flatness limb — annotated in place. Both limbs of risk #3 are
  now realized-and-accepted.
- **Gate E's `degenerate` × {2,5,7,17} FAIL is dispositioned; a future re-score should not
  re-raise it.** [`e2-validation-report.md §5`](../research/e2-validation-report.md) carries
  a disposition note pointing here. The pre-registered *scoring* of Gate E is untouched
  (that bar is a fixed input, not re-litigated); what is recorded is that the design it
  flagged was reviewed and kept.
- **Phase-F revisit trigger — the 2/17 caveat, accepted-for-now, not accepted-forever.**
  The 2/17 drainage loss (§Decision leg 2) is a **real, consciously-accepted
  representativeness cost** at the 50k debug tier. It is accepted now on the structural
  rate-signal (leg 1) and the absence of starvation (leg 3) — but, unlike stratum 19's
  same-family relief-tail thinning, it is a *compositional* thinning (dissected drops vs
  smooth keeps) that #64 flagged by eye and did not quantify at the kept-pool level. It is
  therefore flagged for **re-test at the full-scale Phase-F tier**, where (i) the pinned
  significance leg *gains teeth* (the bar states *"Phase F can re-run the identical spec
  against its own catalog and re-registered bar"*) and (ii) the actually-selected corpus can
  be checked for drainage representation in strata 2/17. Accept here means "sub-threshold
  and unstarved at debug scale — re-verify at scale," not "fine forever."
- **The MAE trains on kept flat tiles that retain drainage structure.** Only near-constant
  surfaces (`land_std < 3 m`) are excluded, per ADR-0002 §2; the dissected low-relief tiles
  that survive the cap in strata 2/17 remain in the training pool. This is the intended
  balance: near-zero-information surfaces out, informative low-relief terrain kept.

## Out of scope

- **The baseline choice and the Finding-vs-Defect classification.** Both settled by the #54
  verdict (registered whole-catalog baseline; these are Findings). Fixed inputs here;
  re-opening the baseline after seeing which pairs it lights up is the post-hoc peeking the
  pre-registration forbids.
- **The `water` × {14, 17} pairs** that appear only under the *sensitivity* baseline — never
  part of the registered FAIL, so not this decision's business.
- **The `edited` × 19 firing** — settled by [ADR-0006](0006-accept-edm-flm-hollowing-stratum19.md).
  This ADR touches only the flatness screen.
- **The C→D→E re-run and a new Gate E.** Under "accept" there is nothing to re-run; this ADR
  spends no pipeline compute. Had the decision been "loosen," the execution would have been a
  separate map — that map is moot.
- **Re-litigating Gate E's pre-registered bar.** The 3× / 10% E2.5 rule and
  `degenerate_land_std_m = 3.0` as the *committed baseline* are fixed inputs; this decision
  is about whether the gate's *design* should change going forward, not whether Gate E scored
  it correctly.

## Alternatives considered

- **Loosen globally (lower `degenerate_land_std_m` for all strata).** Rejected: the rate
  model shows *no* excess beyond structure anywhere, so there is nothing a global cut
  corrects — and a lower global cap readmits stratum 5's genuinely low-information smooth
  plains (dropped `texture` ≡ kept), i.e. exactly the correctly-targeted junk, for no gain.
- **Loosen via a per-stratum 2/17 exemption (raise the cap for the flat drainage families
  only).** Rejected, though it is the shape #64 noted "fits the evidence better" *if* one
  were loosening — because the premise fails: the rate model found the 2/17 concentration
  **also structural** (residuals ~0 at every rung), these strata were pre-registered as
  **corroborating-only**, and acting on their by-eye terrain signal against a structural
  rate result — to force a full C→D→E re-run into unstarved 65×/57× pools — is not warranted.
  Left explicitly on the table (it is the shape the map's fog would have graduated) and ruled
  out consciously rather than by omission. The 2/17 worry is instead carried forward as the
  Phase-F revisit trigger, not acted on now.
- **Accept, but document nothing (let the closed ticket be the record).** Rejected, as in
  ADR-0006: the gate stays as-is *by decision*, not neglect, and a confounded firing this map
  was chartered to adjudicate deserves a discoverable, reasoned record — hence this ADR, the
  knob-table row, and the in-place annotations, so a future reader or a Gate E re-score sees
  a reviewed decision rather than an open FAIL.
- **Accept unconditionally (no Phase-F revisit hook).** Rejected: 2/17's *compositional* loss
  (dissected drops vs smooth keeps) is a stiffer representativeness worry than stratum 19's
  same-family relief-tail thinning, and it was a debug-tier by-eye read the pinned bar itself
  says gains statistical power at scale. Overclaiming permanence would be dishonest; hence the
  accept carries an explicit Phase-F re-test trigger.
