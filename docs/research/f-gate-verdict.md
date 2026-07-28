# Gate F — the verdict (the 500,000-window scale tier)

**Ticket:** [Gate F — review the evidence and return the verdict (#74)](https://github.com/JStoschek/DEMAE/issues/74) ·
**Map:** [Phase F (#67)](https://github.com/JStoschek/DEMAE/issues/67) ·
**Bar:** [`f-gate-bar.json`](f-gate-bar.json) / [`f-gate-bar.md`](f-gate-bar.md) ·
**Report:** [`f-validation-report.md`](f-validation-report.md) / [`.json`](f-validation-report.json) ·
**This adjudication's numbers:** [`f-gate-verdict.json`](f-gate-verdict.json) ·
**Driver:** [`notebooks/f_gate_verdict.py`](../../notebooks/f_gate_verdict.py)

# Verdict: **FAIL**

Applying the roll-up [#69](https://github.com/JStoschek/DEMAE/issues/69) pre-registered
before the corpus existed: **no Defect open**, **one Finding fired** (`F3`) → **FAIL**. No
partial credit — the roll-up was written before any number had a value precisely so a fired
Finding could not be talked away at review time.

**This is not "the corpus is mis-built."** Every Defect-class criterion that speaks to build
fidelity passes, including the two [#69](https://github.com/JStoschek/DEMAE/issues/69) added
because Gate E could not have caught what they catch: `F1d` opened the shards and confirmed
**0 of 499,934** tiles disagree with the metrics the manifest claims for them, and `F6a`
found no window drawn twice. The FAIL is one design finding: **the corpus under-represents
drainage-bearing terrain in strata 2 and 17** — the exact question
[ADR-0007](../adr/0007-accept-degenerate-flatness-gate-flat-strata.md) planted a revisit
trigger for, and the one thing the tenfold budget existed to learn.

Per map [#67](https://github.com/JStoschek/DEMAE/issues/67)'s Destination, that verdict is
recorded here and **this map stops**. The loosening rule, its blast radius and any re-draw
are a **fresh map** — the [#54](https://github.com/JStoschek/DEMAE/issues/54) → [#57](https://github.com/JStoschek/DEMAE/issues/57)/[#62](https://github.com/JStoschek/DEMAE/issues/62) handoff, repeated.

---

## 1. The scoreboard, after adjudication

Fourteen criteria gate. Three fired in [#73](https://github.com/JStoschek/DEMAE/issues/73)'s
blind run; one survives adjudication.

| criterion | class | #73 blind | after review |
|---|---|---|---|
| `F1a` metric fidelity | Defect | pass | pass |
| **`F1b` drop selectivity** | Defect | **FIRED** | **not open** — §2 |
| `F1d` pixel integrity | Defect | pass | pass |
| `F2a` selection integrity | Defect | pass | pass |
| `F2b` materialization counts | Defect | pass | pass |
| `F2c` thin stratum | Finding | pass | pass |
| `F2d` coastal floor | Finding | pass | pass |
| **`F3` drainage representation** | **Finding** | **FIRED** | **FIRED** — §4 |
| `F5` gate attrition published | published | pass | pass |
| `F6a` distinctness | Defect | pass | pass |
| `F6b` COG inventory bound | Defect | pass | pass |
| `F7` attrition expectation | Defect | pass | pass, with §3 recorded |
| `F7b` registered status set | Defect | pass | pass |
| **`F8` normalization statistic** | Defect | **FIRED** | **not open** — §3 |

**A note on what adjudication may and may not do here.** Reclassifying a fired criterion is
the move pre-registration exists to constrain, so it needs a standard. The one applied below
is [#54](https://github.com/JStoschek/DEMAE/issues/54)'s: a firing may be set aside only on
material that was **already committed before the corpus existed**, or on a measurement of
the *criterion's own machinery* rather than of the corpus. Neither disposition below moves a
threshold, and neither would have come out differently had it been reasoned through at
pre-registration time. `F3` — the one criterion whose firing is a statement about the
design — is **not** touched.

---

## 2. `F1b` — the null model is false by construction

**`F1b` asks:** were the 66 windows that failed to materialize a *terrain-neutral* sample of
the 500,000 selected? It compares dropped against retained on each frozen metric
(Wasserstein-1) and fires if any metric exceeds the 99th percentile of a null built by
dropping 66 windows at random, 10,000 times.

**It fired on 4 of 6 metrics, by wide margins** — `local_relief` 243.7 against a null p99 of
106.6, `mean_elevation` 923.8 against 330.1.

**Why the firing carries no information about build fidelity.** 62 of the 66 drops came from
the `degenerate` gate, which fires on `land_std < 3.0 m` — **a terrain property**. Windows
removed for being too flat cannot be a terrain-neutral sample of anything; the null was false
before the corpus was built, so rejecting it says nothing about whether the corpus faithfully
implements its inputs. The dropped windows sit at `land_std` 3.0–3.5 m against a retained
median of **24.2 m**, which is the gate doing exactly what it is for.

**The bar says this itself, in the same committed file.** `F7`'s derivation — written before
any of these numbers existed — describes the drops as `land_std` values falling across the
3.0 m threshold and treats the mechanism as benign. **`F1b` and `F7` cannot both be read
literally on this corpus**, and the contradiction is between two pre-registered clauses, not
between a clause and a result.

**Why it took 500 k to surface.** At Gate E the drop count was 2, and a test comparing 2
items against 499,998 detects nothing, so it passed silently. The bar disclosed the weakness
in advance — "pinned for form and to catch a gross terrain-correlated attrition" — and
expected D≈20. D came in at **66**, finally enough power to detect the mechanism the bar had
already predicted.

**The disposition does not rest on `F7` being right about the details.** `F7`'s assumed
0.001 m band turns out to be ~0.46 m wide (§3). The argument needs only that the gate is a
*flatness threshold*, which is definitional.

> **Recorded so the criterion is not simply discarded:** `F1b` carries information exactly
> when attrition's mechanism is terrain-**neutral**. The 4 `zero_fill` drops are that kind;
> the 62 flatness drops are not. A future bar wanting a live selectivity test on a corpus
> whose attrition is gate-driven has to condition the null on the drop mechanism — a
> different criterion, not this one with a different number.

**Disposition: not an open Defect.**

---

## 3. `F8` — "exact" is satisfiable, and the corpus satisfies it

**`F8` asks:** does `corpus_meta.json`'s recorded `global_mean` / `global_std` reproduce
**exactly** from the shipped tiles? The rule compares two float64s with `!=`.

**#73 measured `global_std` 4 ULP out** (4.5e-16 relative), `global_mean` bit-identical, and
handed up the right question: *can a read-back that sums in a different order satisfy `exact`
at all?* It refused to answer, because any tolerance it chose would have been chosen after
seeing the gap.

**It is not a question about float64 — the block size is a free parameter the harness picked.**
The builder folds each tile's moments into its accumulator **inside the same loop that writes
that tile to the shard** ([`materialize.py`](../../src/dataset/materialize.py) ~1095), so the
builder's addition order *is* shard order. #73's `recompute_global_stats` read back in
512-tile blocks (`SHARD_BLOCK`, chosen so per-block Python overhead vanishes against a ~51 MB
read) and reduced each block in one call. **The bar registers no granularity.**

**Measured** — both granularities in a single pass over all 47 GB, so the two see
byte-identical input:

| read-back granularity | `land_sum` / `land_count` | `land_sqsum` | `global_std` vs recorded | `F8` |
|---|---|---|---|---|
| 512-tile blocks — #73's, run as the **control** | bit-identical | 3 ULP out | **4 ULP** | fires |
| **per-tile — the builder's own order** | bit-identical | **bit-identical** | **0 ULP** | **passes** |

**The control is what makes this attributable.** A bit-identical per-tile result alone could
mean this driver differs from #73's anywhere at all. Reproducing #73's published 4-ULP
discrepancy *first*, in the same pass, leaves the block size as the only difference. The
prediction was also made from reading the builder's loop **before** the measurement ran, so
this is a confirmed prediction rather than a parameter tuned until it passed.

**Disposition: not an open Defect.** The recorded statistic does reproduce exactly from the
shipped tiles. `F8`'s registered test passes; #73's harness read at a granularity the bar
never specified.

> **Carried forward:** the guarantee is now pinned by
> `test_global_stats_reproduce_bit_identically_at_the_builders_granularity`, so a future
> read-back that quietly re-batches cannot fire a Defect on a correct corpus again.

---

## 4. `F7` — passes the rate it gates; the *prediction* is what failed

`F7` gates on the drop **rate** and separately carries a pre-registered **expectation** of
~20 drops (11–31 at ±2σ). The rate passes with room: **66 drops, 1.32e-4 against a 4e-4
cap.** The expectation misses by 3.3×. The bar registers no rule for that combination, which
is why #73 handed it up.

**The report's spread table pools two mechanisms, and the pooling is misleading.** #73
published the dropped windows' catalog `land_std` against the 3.0 m threshold as one table
over all 66, spreading out to **6.747 m** — more than twice the threshold, which would mean
the catalog and the builder disagree wildly about the same window. Split by the ledger status
that actually dropped them:

| cause | n | `land_std` min | median | **max** |
|---|---|---|---|---|
| `degenerate` (flatness) | 62 | 3.001 | 3.069 | **3.460** |
| `zero_fill` | 4 | 3.582 | — | 6.747 |

**The alarming tail is `zero_fill`, not flatness.** All four `zero_fill` drops are
water-affected windows with 18–27% of their pixels at exactly zero elevation — a different
mechanism, and one `F7`'s derivation did not anticipate at all. The flatness mechanism is
bounded at **0.46 m**, a ≤15% disagreement decaying smoothly away from the threshold.

**So why 66 and not 20?** The derivation assumed the catalog-vs-builder disagreement band was
~0.001 m wide. Measured, it is ~0.46 m — **roughly 460× wider**, and a wider band catches
proportionally more windows sitting near the line. **The derivation's band estimate is what
did not survive contact; the corpus is behaving as designed.**

**Checked for a Defect hiding here, because this was the most plausible place for one.**
`F1d` measured the same catalog-vs-pixels agreement across all 499,934 *retained* tiles:
median disagreement 8.1e-5 relative, **zero** tiles over tolerance. The 66 drops are the far
tail of that same small, known disagreement, concentrated where a tail is consequential —
against a threshold. Coherent. `F7b` separately confirms no unregistered drop status
appeared.

**Disposition: `F7` passes as registered.** The expectation miss is a flaw in the prediction,
not in the corpus.

---

## 5. `F3` — the Finding, and what it actually implicates

**The criterion.** For strata 2 and 17 (low-relief terrain with faint drainage), `r_h` is the
share of shipped tiles whose `texture` reaches that stratum's **pre-gate p75** — so the
reference is 0.25 by construction, and it is the drainage share the terrain itself has before
the flatness gate touches it. Fire if `r_h < 0.25`.

**It fired.** `r_2` = **0.2097**, `r_17` = **0.2065**.

### 5.1 The attribution — and it is not what the pre-assigned knob says

[#69](https://github.com/JStoschek/DEMAE/issues/69) published a decomposition precisely so a
fresh map could tell two very different causes apart: `r_h` near the **kept-pool share** means
the sampler drew proportionally and declined an available tail (**a sampler problem**); `r_h`
near the **exhaustion ceiling** means the survivors ran out (**a gate problem**).

| | stratum 2 | stratum 17 |
|---|---|---|
| `r_h` realized | 0.2097 | 0.2065 |
| kept-pool share (a proportional draw) | 0.1741 | 0.1703 |
| `r_h` as a multiple of proportional | **1.20×** | **1.21×** |
| multiple needed to reach 0.25 | 1.44× | 1.47× |
| `e_h` exhaustion — share of surviving tail used | **18.5%** | **21.3%** |

FSCS *did* over-draw the drainage tail — just not far enough. Stated as tiles:

| | stratum 2 | stratum 17 |
|---|---|---|
| drainage tiles shipped | 4,856 | 3,811 |
| shortfall to reach 0.25 | 933 | 804 |
| surviving drainage tail (`kept_tail`) | 26,271 | 17,914 |
| **surviving tail left untouched** | **21,415** | **14,103** |
| the unused tail could cover the shortfall | **23.0×** over | **17.5×** over |

*(These are the figures from the **gated** run, against the bar's frozen p75. #73's
published full-precision sensitivity moves stratum 17 slightly — `n_drainage` 3,814,
`kept_tail` 17,930 — and fires on the same two strata. 0.21 against 0.25 was never a
few-dozen-tile question.)*

**The gate is not the binding constraint.** The flatness screen removed roughly half of each
stratum's drainage tail, but what survived could have filled the gap seventeen times over. The
sampler did not reach that deep.

### 5.2 Why the payload names two things

The bar pre-assigned `F3`'s knob as the **`degenerate` flatness screen**, and that stands as
registered — substituting a knob after seeing the attribution is exactly the move
pre-registration blocks. But the measured constraint is **within-stratum FSCS reach**, which
has **no row in the plan's §215–228 knob table**.

That is [#54](https://github.com/JStoschek/DEMAE/issues/54)'s gap again. Its rule said "a
criterion that can't name a knob is a Defect"; #54 resolved that the honest reading is *the
knob table is incomplete*, not *the build is broken*. Same resolution here — with the
consequence that **adding the missing row is the fresh map's first task**, before it sizes
anything.

### 5.3 Finding payload

```json
{
  "implicated_knob": "degenerate flatness screen (`land_std` cap) — as pre-registered by #69. SECONDARY, and the one the evidence points at: within-stratum FSCS coverage depth, which has NO ROW in the plan §215-228 knob table and must be added before it can be sized.",
  "measured_miss": "r_2 = 0.2097 and r_17 = 0.2065 against a registered bar of 0.25, both strata gating. Attribution: r_h is 1.20x / 1.21x the kept-pool proportional share (0.1741 / 0.1703) where 1.44x / 1.47x was needed, while exhaustion e_h is only 0.185 / 0.213 — so the sampler over-drew the tail but stopped far short of using it. The shortfall is 933 / 804 tiles against 21,415 / 14,103 surviving tail tiles left untouched: the tail could cover it 23.0x / 17.5x over. The gate is NOT the binding constraint.",
  "evidence_pointer": "docs/research/f-validation-report.md#f3-drainage-representation ; docs/research/f-validation-report.json -> drainage ; docs/research/f-gate-verdict.md#5-f3--the-finding-and-what-it-actually-implicates ; figure docs/research/f-drainage.png",
  "class": "Finding"
}
```

**What a re-run would have to show** to clear `F3` next time: `r_h ≥ 0.25` in both strata 2
and 17. The measurement above says the cheapest route is the sampler, not the gate — but that
is a hypothesis for the fresh map to test, not a prescription this verdict may make. A
Finding names the knob; it does not set the new value.

---

## 6. What this FAIL does and does not certify

The map required this to be explicit, because Gate F judged a corpus built from a design that
had already been challenged and accepted:
[ADR-0006](../adr/0006-accept-edm-flm-hollowing-stratum19.md) and
[ADR-0007](../adr/0007-accept-degenerate-flatness-gate-flat-strata.md) both accepted gate
behaviour that had *fired* at Gate E.

**Certified by the evidence:**

- **The corpus is faithfully built.** Every build-fidelity Defect passes, including two Gate E
  never posed: `F1d` read the actual pixels (0 of 499,934 tiles disagree with their manifest
  metrics, against 48.1% under a deliberate mixup) and `F6a` confirmed no window drawn twice.
- **The scale draw conforms to its committed contract** — selection list md5, per-stratum
  allocation, provenance split, `realized ≤ selected`, no COG yielding more than it holds.
- **The recorded normalization statistic is reproducible** from the shipped tiles, bit-for-bit
  (§3). Phase 1's `dataset_version` contract holds.
- **ADR-0007's revisit trigger is discharged** — it asked for a re-test at scale and got one.

**Not certified, and it matters:**

- **`ADR-0007`'s accept is *not* confirmed at scale.** Its "accept-for-now" was conditional on
  this re-test, and the re-test fired. The accept does **not** silently become permanent, and
  the trigger does **not** follow the corpus into Phase 1 as discharged-and-clear — it follows
  it as **fired**.
- **A FAIL is not a statement about ADR-0006.** Stratum 19's `edited` hollowing was not
  re-tested here; no criterion addressed it. It stands exactly where ADR-0006 left it.
- **Eleven passing criteria do not average out one fired Finding.** The roll-up says so, and
  it said so before any number existed.
- **`F1b` and `F8` being set aside is not a clean bill for those criteria** — both are
  recorded as criteria whose registered form met an input the bar did not anticipate, and §2
  and §3 say what a future bar would need to do differently.

---

## 7. The two dispositions the review also owed

### 7.1 The per-COG cap [#42](https://github.com/JStoschek/DEMAE/issues/42) retired: **not restored**

Realized spread at 500 k: mean 21.8 windows per COG, median 14, p99 118, max **233**. As a
fraction of each COG's own inventory: median 15.2%, mean 22.9%, **2,788 COGs at ≥50%**, **423
drawn dry**. The debug tier had none over 0.5.

1. **A count cap is the wrong instrument.** A COG's ceiling is its own inventory (at most 256
   kept windows), and the average draw is already 23% of it. A flat 64 cannot express
   "don't over-concentrate" when the denominator varies per file.
2. **It has a measured price and no measured benefit** — [#71](https://github.com/JStoschek/DEMAE/issues/71)
   found restoring 64 would bind on 1,320 COGs and strip **40,211 windows, 8.0% of budget**.
3. **The well-posed question already gates and passes** — `F6b`, deepest draw 233 against that
   COG's own 256.
4. **It would work against `F3`'s fix.** Reaching deeper into a stratum's drainage tail means
   drawing more heavily from the files that hold it; a ceiling pushes the other way. *(Flagged
   as a tension — how spatially clustered the drainage tail is has not been measured.)*

The **spatial-redundancy residue is real and new at scale** and goes to
[#75](https://github.com/JStoschek/DEMAE/issues/75) as a documented property. If a limit is
ever wanted, the right shape is **fraction-of-inventory**, not a count.

### 7.2 The 50 k debug tier: **retained**

Free disk is **34 GB**; the scale corpus is **47 GB**, the debug tier **4.7 GB**.

- **Retiring it does not buy what it appears to.** Freeing 4.7 GB reaches 38.7 GB against a
  47 GB redraw. **The only thing that makes room for a second scale corpus is deleting the
  first.** The debug tier is not the lever.
- **A FAIL makes it more useful, not less.** The fresh map iterates on sampler reach;
  iterating against 4.7 GB rather than a ~2.5-hour 47 GB build is what a debug tier is for.
- **The existing record rests on it.** ADR-0006 and ADR-0007 were argued from its numbers and
  [`validate.py`](../../src/dataset/validate.py) was deliberately frozen against it.

**Recorded for the fresh map:** a redrawn scale corpus **cannot coexist with the current one**
at 34 GB free. Whether to keep the failed corpus as a comparison baseline is a real cost the
fresh map must plan for, not discover.

---

## 8. What the verdict binds

- **This map stops.** No loosening rule, no re-draw, no gate threshold touched here.
- **A fresh map is seeded, not created** — §5.3's payload is its starting evidence, and adding
  the FSCS-depth row to the §215–228 knob table is its first task.
- **[#75](https://github.com/JStoschek/DEMAE/issues/75) proceeds**, unblocked by closing
  [#74](https://github.com/JStoschek/DEMAE/issues/74). It is named in the map's Destination as
  one of the two records the Gate E FAIL orphaned, its own body asks it to record "how Gate F's
  re-test at scale came out", and this map already judged Gate E's identical deferral a
  mistake — the project went a whole phase with no ADR-0005 at all. The notes are written
  against the shipped corpus **with the FAIL and its attribution in them**.

Copyright / attribution (mask spec §5, Art. 6(b)): *produced using Copernicus WorldDEM-30
© DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS
by the European Union and ESA; all rights reserved.*
