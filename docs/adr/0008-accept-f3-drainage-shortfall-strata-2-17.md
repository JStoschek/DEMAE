# ADR 0008 — Accept the `F3` drainage shortfall in strata 2 and 17

- **Status:** Accepted
- **Date:** 2026-07-28
- **Context:** Taken directly on the evidence Phase F committed — **no wayfinder map**. The decision turns on a magnitude and a cost/benefit, both already measured by [#73](https://github.com/JStoschek/DEMAE/issues/73)/[#74](https://github.com/JStoschek/DEMAE/issues/74); there was no fog to chart, which is the wayfinder skill's own test for when a map is unwarranted. The two prior gate-disposition maps ([#57](https://github.com/JStoschek/DEMAE/issues/57), [#62](https://github.com/JStoschek/DEMAE/issues/62)) existed to *find* an answer with new measurement — this one did not need to.
- **Seeded by:** Gate F's **FAIL** ([#74](https://github.com/JStoschek/DEMAE/issues/74)) on `F3_drainage_representation` — the only gating criterion that survived review. `r_2` = **0.2097**, `r_17` = **0.2065** against a pre-registered **0.25**. This is [ADR-0007](0007-accept-degenerate-flatness-gate-flat-strata.md)'s Phase-F revisit trigger, fired.
- **Inputs (all primary, committed):** [`f-gate-bar.json`](../research/f-gate-bar.json) ([#69](https://github.com/JStoschek/DEMAE/issues/69)) — `F3` pinned before the corpus existed; [`f-validation-report.json`](../research/f-validation-report.json) ([#73](https://github.com/JStoschek/DEMAE/issues/73)) — the blind execution and the `r_h`/`e_h` decomposition; [`f-gate-verdict.md`](../research/f-gate-verdict.md) ([#74](https://github.com/JStoschek/DEMAE/issues/74)) — the verdict and its attribution; [`f-honesty-notes.md`](../research/f-honesty-notes.md) ([#75](https://github.com/JStoschek/DEMAE/issues/75)) — what downstream users have already been told
- **Depends on:** [ADR-0005](0005-sampling-design.md) (the sampling design `F3` tests, including Neyman allocation and `accept_shortfall`); [ADR-0007](0007-accept-degenerate-flatness-gate-flat-strata.md) (whose trigger this closes)

> **Scope in one line.** This ADR decides **what to do about a criterion that correctly fired** — and the answer is nothing. It does **not** dispute `F3`'s measurement, move its threshold, or re-open the flatness screen. Accepting a true finding and loosening a bar are different acts, and only the first is happening here.

## Context

Gate F pre-registered `F3` before the 500 k corpus existed, as ADR-0007's promised re-test of
the 2/17 drainage loss. It executed blind and fired. Nothing about that is in question: the
rule was fixed in advance, the measurement is reproducible from committed artifacts, and a
published sensitivity confirms it fires at full precision as well as at the bar's printed
one. **`F3` is right.**

What `F3` found, in the terms the bar set up:

| stratum | shipped | drainage-bearing | `r_h` | exhaustion `e_h` |
|---|---|---|---|---|
| 2 | 23,153 | 4,856 | **0.2097** | 0.185 |
| 17 | 18,459 | 3,811 | **0.2065** | 0.213 |

"Drainage-bearing" is a **proxy**: a tile whose `texture` sits at or above its own stratum's
pre-gate p75. `texture` is the fraction of land pixels departing from their 3×3 neighbourhood
median by more than 1 m — fine, busy surface, indifferent to steepness. In a low-relief
stratum that busy-ness is mostly channel networks, which is what makes it a drainage proxy.
The 0.25 bar is not invented: because the threshold *is* the pre-gate p75, the top quartile is
by construction 25% of the pre-gate population, so the criterion asks only that the corpus
match the share the terrain itself has.

[#74](https://github.com/JStoschek/DEMAE/issues/74) established the attribution: **the
flatness screen is not the binding constraint.** Exhaustion of 0.185 / 0.213 means the draw
took only about a fifth of the *surviving* drainage tail, leaving **21,415** and **14,103**
eligible tiles untouched — enough to cover the shortfall 23.0× and 17.5× over. The binding
constraint is within-stratum sampler reach, a knob that had no row in the plan's table until
Phase F added one.

So the question this ADR answers is not "is the corpus deficient" — it is, measurably — but
**"is 1,737 tiles' worth of composition worth acting on."**

## Decision

**The `F3` shortfall is accepted. No redraw, no re-materialization, no change to the sampler,
no change to any gate. The corpus ships as built, at 499,934 tiles, `dataset_version`
`bcbd6484d8116199c5c1160e21ea0284`.** Gate F's **FAIL is thereby dispositioned**: reviewed,
and the design deliberately left in place. ADR-0007's Phase-F revisit trigger is **fired,
reviewed, and closed** by this record. The within-stratum FSCS coverage-depth knob stays
**declared and unsized** (§Consequences).

Four legs carry it.

### 1. The magnitude is a third of one percent

Holding per-stratum quotas fixed, reaching 0.25 needs **933** tiles swapped in stratum 2 and
**804** in stratum 17 — **1,737 tiles, 0.347% of the corpus**. Strata 2 and 17 together are
**8.3%** of the corpus, and within them the gap is about **4 percentage points**.

This is worth stating against how the verdict reads: Gate F's roll-up gives no partial credit,
so a single Finding on 0.35% of composition returns the same word — `FAIL` — as a
catastrophe would. Every build-fidelity criterion passed, including `F1d`, the first in this
project to open a shard and read pixels (0 of 499,934 tiles over tolerance).

### 2. The pattern across the four flat strata does not look like systematic starvation

`F3` measured four strata and gated on two:

| stratum | `r_h` | status |
|---|---|---|
| 7 | 0.3008 | **over** the target |
| 2 | 0.2097 | **fired** |
| 17 | 0.2065 | **fired** |
| 5 | 0.1890 | under — but corroboration-only, never gated |

A spread of **0.189 to 0.301** around a 0.25 target reads as ordinary sampling variance
rather than as a sampler that systematically starves the drainage tail. One of the four is
*over*-represented.

**The decisive detail is stratum 5.** It sits **further from the target than either stratum
that fired**, and escaped only because [#63](https://github.com/JStoschek/DEMAE/issues/63)'s
two-tier design made strata 5 and 7 corroboration-only — a deliberate, well-reasoned choice
about statistical confidence, made before any number existed. But its consequence here is
that **"fix what fired" is not "fix what is worst."** A repair aimed at 2 and 17 would leave
the most under-represented stratum untouched and return a passing gate, which is a worse
outcome than an honest failing one: the corpus would then *claim* drainage representativeness
it does not have.

### 3. The target is a proxy, and the proxy is unverified

Nothing in the pipeline identifies a channel. `texture` responds to fine surface busy-ness of
any origin — dunes, boulder fields, agricultural furrowing, radar speckle — and no one has
visually confirmed that the top-quartile tail in strata 2 and 17 is dissection rather than one
of those. The proxy was chosen on solid grounds (it is the only frozen metric that separates
the flatness screen's drops from its keeps: 0.015 vs 0.006 in stratum 2, 0.021 vs 0.009 in
stratum 17, where `geomorphon_entropy` does not separate at all) — but "best available
separator" is not "verified drainage detector."

Acting would therefore mean spending a redraw to move a proxy whose fidelity to the thing it
proxies is itself unmeasured.

### 4. The downstream consequence is unmeasured, and plausibly nil

No evidence exists that a 4-point composition shift, in 2 of 20 strata covering 8.3% of the
corpus, changes anything a masked autoencoder learns. The probe that would show it is a
Phase-2 artifact and Phase 1 has not run. This is the leg with the least evidence behind it,
and it is stated as a judgement rather than a finding: the expected effect is small, and it
is **not zero-by-proof**. That uncertainty is what the revisit trigger below exists to carry.

## Consequences

- **Gate F's FAIL stands as the record, and the corpus ships anyway.** That combination is
  deliberate and is written down rather than allowed to drift: the project set a bar, missed
  it, reviewed the miss, and chose to proceed. The verdict is not amended, the bar is not
  moved, and `f-gate-verdict.md` stands exactly as committed.
- **Downstream users have already been told.** [`f-honesty-notes.md`](../research/f-honesty-notes.md)
  §3.1 states the shortfall, the attribution, and what a model trained on this corpus saw —
  in a document written for readers outside this project. §3.2 is updated to point here.
- **ADR-0007 is closed out, not reversed.** Its trigger genuinely fired; that fact stays in
  its Amendment 1. What changes is that the disposition is no longer open. ADR-0007's own
  accept — the flatness screen at 3.0 m — is untouched and was never the binding constraint.
- **The FSCS coverage-depth knob stays declared and unsized.** This is *not* the
  retired-inert treatment the re-clustering knob got ([#71](https://github.com/JStoschek/DEMAE/issues/71)):
  that knob is structurally incapable of changing anything, whereas this one is live, real,
  and simply not being set. It remains in the plan's §9 table with its evidence, so the next
  effort that wants to reach deeper into a sparse region knows the knob exists and knows what
  firing looks like.
- **The fix does not get more expensive with time.** Recorded so a revisit starts from facts
  rather than re-deriving them: **1,737 tiles**, roughly **174 MB** of imagery, against
  **21,415** and **14,103** untouched eligible tiles still in the pool. A full re-materialization
  is *not* required and would not fit — 47 GB against 34 GB free — but an append-and-retire
  against the manifest touches ~174 MB and deletes nothing.

### Revisit trigger — downstream evidence, not re-measurement

**Revisit if a Phase-2 probe shows the model reconstructing channel networks measurably worse
in the low-relief strata (2, 5, 7, 17) than elsewhere in the corpus.**

The form is deliberate. ADR-0007's trigger asked *"is this number still bad at a bigger
budget?"* — and it produced a number and no decision, because the number was never the crux.
Re-measuring `r_h` at another budget would repeat that. What would actually change this
answer is evidence that the composition gap reaches the model, so that is what the trigger
names.

It is written to be **cheap to evaluate**: a per-stratum breakdown of reconstruction error on
held-out tiles, which any Phase-2 probe run produces anyway, compared across strata rather
than against an absolute standard. If low-relief strata are not distinguishable, this ADR
stands and no further work is owed.

## Out of scope

- **Re-opening the `degenerate` flatness screen** ([ADR-0007](0007-accept-degenerate-flatness-gate-flat-strata.md)).
  Exhaustion of 0.185 / 0.213 says it is not the binding constraint, so loosening it would be
  sizing the wrong knob — and this ADR declines to size any knob at all.
- **Sizing the within-stratum FSCS coverage depth.** The knob is declared; setting it is work
  this decision makes unnecessary, not work it defers.
- **Re-litigating `F3`'s threshold, its two-tier structure, or Gate F's roll-up rule.** All
  three were pinned before the corpus existed. Stratum 5's exclusion from gating is *used* as
  evidence in leg 2, not challenged: the point is what it implies about repairing 2 and 17, not
  that it was wrongly decided.
- **Re-opening anything Gate E's disposition settled** — [ADR-0006](0006-accept-edm-flm-hollowing-stratum19.md)
  and ADR-0007 both stand.
- **The three §199 deferred items** — vegetation column, geographic hold-out `split`,
  anisotropy metric. Each waits on evidence that cannot exist until pretraining runs.

## Alternatives considered

- **Fix by swap — 1,737 tiles, ~174 MB, append-and-retire against the manifest.** Rejected on
  cost/benefit, **not on feasibility**: it is genuinely cheap, it holds Neyman quotas exactly,
  and it is where a revisit would start. What it buys is a passing gate on 0.35% of
  composition, with legs 2–4 unaddressed — and leg 2 says it would produce a *misleadingly*
  passing gate while stratum 5 stayed worst.
- **Fix by addition rather than swap** — add drainage tiles without removing any. Rejected:
  it pushes strata 2 and 17 past their Neyman quota, so it would fix `F3` by breaking the
  allocation rule ADR-0005 §4 commits to. `F3` is a within-stratum composition question and
  wants a within-stratum answer.
- **Fix by full redraw and re-materialization.** Rejected: 47 GB against 34 GB free, so the
  current corpus would have to be destroyed before its replacement was known to be better —
  and it buys nothing the swap does not.
- **Loosen the bar from 0.25.** Rejected, and worth stating plainly because it is the option
  this ADR is most likely to be mistaken for: moving a pre-registered threshold after seeing
  the number it failed is exactly the peeking the pre-registration exists to prevent. **The
  bar stands at 0.25 and the corpus misses it.** What is being decided is whether to act on a
  true finding, not whether the finding is true.
- **Accept with a re-measurement trigger** (ADR-0007's form — revisit at the next budget
  change). Rejected: that form has now been run once and produced a number rather than a
  decision. See the revisit trigger above.
- **Accept unconditionally, with no trigger.** Rejected as overclaiming. Leg 4 is a judgement,
  not a proof, and an unconditional accept would assert knowledge about downstream effect that
  nobody has.
