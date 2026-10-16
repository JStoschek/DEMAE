# ADR 0015 — The Task-2 pre-registration is a register of twenty rules, L1–L20, asserted here and anchored in §7

- **Status:** Accepted
- **Date:** 2026-10-16
- **Context ticket:** "Write Task 2's pre-registration register" ([issue #192](https://github.com/JStoschek/DEMAE/issues/192)), part of the Task-2 landform map ([#183](https://github.com/JStoschek/DEMAE/issues/183))
- **Evidence:** the test list's geocell count (5,000 tiles in **13** 10° geocells, 122–980 tiles each, median 234, counted from `artifacts/task1/test_tiles.csv`); §8's B3 price (181 GPU-min per seed); the rulings of the map's closed tickets, cited per row
- **Amends:** [`docs/research/task2-landform-plan.md`](../research/task2-landform-plan.md) §0 (C2's wording, and each claim's reading points here), §4.3 (B3's seed count), §6 (the statistics bullet), §7 (item 10, the register's index), §8 (B3's row and the totals)
- **Follows:** [ADR-0013](0013-task1-pre-registration-register.md), whose Decisions 1–8 are adopted for Task 2 without change and not restated

## Context

ADR-0013 settled *how* a task pre-registers: one ADR for the collection, the plan's sections mark
and point, the fairness checklist (the copy that ships verbatim) is the index and the anchor,
membership is the **tunability test**, the claims stay in the plan and out of the register, and
the numeric subset is a committed JSON bar the loader checks against the anchored section. None
of that is re-opened here. This ADR does what ADR-0013 did for Task 1: it fixes *which* rules
Task 2 has, and their thresholds, before the first head trains.

Task 2 needs rules Task 1 did not, for four reasons:

- **C2 is a non-inferiority claim.** "Beats or matches" has no reading without a margin.
- **The label-fraction rungs vary by region.** The geocell ladder ([#189](https://github.com/JStoschek/DEMAE/issues/189))
  gives each seed different labelled regions at 1% and 10%. A bootstrap over test tiles cannot
  see that variance.
- **The test set covers only 13 geocells.** A cluster bootstrap over 13 clusters gives a
  coarse, unstable CI, so the tile bootstrap stays, and its spatial-correlation caveat has to be
  written down rather than discovered.
- **C1 is read against a choice of TerraMind encoders.** [#190](https://github.com/JStoschek/DEMAE/issues/190)
  made B9 a core arm and read C1 against "the stronger" of B4/B5 and B9, and left the
  depth-sweep reversal to this ticket. Both choices could be made after seeing the numbers.

The plan also still had one rule that was decided after the numbers: §4.3's "one seed plus a
second if variance on val looks material" for B3. A seed count chosen after seeing the variance
is exactly what a register exists to rule out.

## Decision

**1. ADR-0013's form is adopted whole.** One ADR asserts the rules. The plan's sites mark and point.
§7 (Task 2's fairness checklist, which ships verbatim) carries the index as item 10, and the
numeric subset is committed as **`task2-prereg-bar.json`**, anchored in §7 the way
`task1-prereg-bar.json` is anchored in Task 1's §6. Building the bar, and generalising
`benchmark.prereg` to load either task's bar, belongs to the build map, as it did for Task 1.

**2. Rules are numbered L1–L20.** Task 1's R1–R12 already collide with the encoder names R4 and
R5. A second R-series would make "R4" mean three different things.

**3. C1–C3 stay in §0** (ADR-0013 Decision 6). The register holds how each claim is *read*
(L12, L16, L17). §0 states the claim and points at the rule.

**4. Five of Task 1's rules carry over, restated for Task 2** (L4–L8). R1, R5, R7, R8 and R11 are
void-fill instruments with no Task 2 counterpart. R9b refuses a threshold on Task 1's FSQ
diagnostic, which Task 2 does not run. R6 and R12 come back in Task 2's form as L18 and L9.

**5. One margin, δ = 1.0 mIoU point, fixed rather than derived.** A margin derived from B7's seed
spread was considered and rejected: it lets the baseline's own noise widen the margin DEMAE is
judged against, so a noisier B7 would make "matches" easier to get.

**6. B3 runs 3 seeds**, paired with B7's three. It costs about 3 h of a 5080 (~$1.20). C3's
fine-tune half then has the same seed structure as every other pair.

## The register

Twenty rules, ten of them carrying a number. "Lower" means the lower bound of the 95% paired
bootstrap CI of the difference (first arm minus second).

| # | Rule | Home | In bar file | Source |
|---|---|---|---|---|
| L1 | Claims are read on **mIoU** (geomorphons) and **macro-F1** (Iwahashi tiles), on the **pooled test set** only. Macro-F1 (dense), per-class IoU, overall accuracy, boundary-F1, per-band metrics and C1's 1%/10% rows are printed and carry no claim. A claim restricted to one band, metric or rung after the numbers exist is not made | §6 | ✅ metric names | #192 |
| L2 | Paired bootstrap over **test tiles, 10,000 resamples**. Per seed, a resample's mIoU comes from summing the resampled tiles' confusion matrices. The statistic is the seed-paired difference (arm A seed *i* against arm B seed *i*) averaged over seeds, and every seed uses the same tile resample. Caveat, registered: the test set spans **13** geocells, so the tile CI is narrower than a spatially honest one would be | §6 | ✅ `10000`, `13` | #192 |
| L3 | **3 seeds** for every core cell, **B3 and B7 included**. Seed *i* trains on ladder *i*. The seed spread prints as mean ± std, separate from the CI and never folded into it | §3.3, §4.3, §6 | ✅ `3` | #189, #192 |
| L4 | The pretraining-corpus caveat is symmetric. It can't explain a DEMAE loss unless it is also priced as a discount on a DEMAE win (Task 1 R2) | §2, §7.4 | ❌ discursive | #122 |
| L5 | R5 is C1's DEMAE number unconditionally. An R4 win is reported prominently and never becomes C1's number (Task 1 R3) | §0, §1 | ❌ discursive | #122, #190 |
| L6 | TerraMind's DEM tokenizer had **20** epochs, cited per modality, not the paper's 100 (Task 1 R4) | §2 | ✅ `20` | #138 |
| L7 | The GSD caveat is symmetric, and stated as a bandwidth shift rather than a resolution handicap (Task 1 R9) | §2, §7.3 | ❌ discursive | #139 |
| L8 | A null is reported with the prominence of a win (Task 1 R10) | §0 | ❌ discursive | plan §0 |
| L9 | **Insulated rows.** B4-ALT/B5-ALT, B8, the concatenation row, B1-S/B2-S and B1-R4/B2-R4 carry no claim and never stand in for a core row. If an ALT row beats DEMAE at 100% with lower > 0, C1's sentence must say so in the same paragraph | §1, §5.4 | ❌ discursive | #190, #192 |
| L10 | A band's macro metric averages over the classes that have ground-truth pixels in that band. **Class collapse** is defined as any run where a class present in test ground truth gets test IoU = 0. It is flagged in the run record and counted per cell, and never excluded or rerun | §6 | ❌ discursive | ADR-0014, #192 |
| L11 | **"The stronger TerraMind encoder"** is whichever of the tokenizer (B4/B5) and the backbone (B9) has the higher seed-mean **test** score, picked separately for each head and each label set. Picking on test is allowed here because it can only strengthen DEMAE's opponent | §0, §1 | ❌ discursive | #190, #192 |
| L12 | **C1** is read at the **100%** rung on **4 pairs**: B1 vs B6 and B1 vs the stronger TerraMind (linear); B2 vs B6 and B2 vs the stronger TerraMind (UperNet). It holds if all four have lower > 0. If only one head's two pairs clear, C1 is reported for that head only, in those words. **Iwahashi gate** (§7.6): C1 is claimed only if R5's Iwahashi macro-F1 point estimate exceeds both B6's and the stronger TerraMind's. Otherwise it reads "holds on geomorphons, contradicted on Iwahashi" | §0 | ✅ pairs, `100` | #190, #192 |
| L13 | **Depth-sweep reversal**: some competitor's best point (any block, or `output`) beats DEMAE's best point with a CI that excludes zero (1 seed, tile bootstrap). A reversal qualifies C1's linear half to "at the registered layer" and never withdraws it. The UperNet half is unaffected | §5.3 | ❌ discursive | #190, #192 |
| L14 | If B1 and B4 swap order (seed-mean point estimate) under any of ADR-0014's sensitivity label sets, that is stated in C1's paragraph. C1 is read on the primary labels only | §7.7 | ❌ discursive | ADR-0014, #192 |
| L15 | **δ = 1.0 mIoU point**, the non-inferiority margin for C2's "matches" and C3b's "competitive" | §0 | ✅ `1.0` | #192 |
| L16 | **C2**: B2 at the 1% rung (200 tiles) vs B7 at the 10% rung (2,000 tiles). **Beats** if lower > 0, **matches** if lower > −δ, fails otherwise. Ladder condition: the per-ladder difference must be ≥ −δ in **at least 2 of 3** ladders, otherwise C2 is "region-dependent" and not claimed. B1 at 1% vs B7 at 10%, and the **label multiplier** (B7 tiles needed to match B2 at 1%, by log-linear interpolation along B7's three rungs, printed as "> 100×" past the last rung and never extrapolated) are printed unmarked | §0 | ✅ rungs, `2` of `3` | #189, #192 |
| L17 | **C3a**: R5's Iwahashi classifier vs B6's and vs the stronger TerraMind's, macro-F1 over the 15 groups at the 60% rule. It holds if both have lower > 0. The 50% row and the overview level are printed beside it with no claim. **C3b**: B3 vs B7 at 100%, mIoU, competitive if lower > −δ. A B7 win is reported as the expected PANGAEA outcome | §0, §3.2 | ✅ `60`, `15` | #193, #192 |
| L18 | **Eight** marked bootstrap pairs, named in L12 (4), L16 (1) and L17 (3). Every other pair is computed into the results and left unmarked in print | §6 | ✅ the eight | #192 |
| L19 | **Val realism caveat**: every rung selects its checkpoint on all 2,000 val tiles, which a real 200-label user would not have. It applies equally to every arm, so it costs realism, not fairness | §4.5 | ❌ discursive | #196 |
| L20 | **Reruns**: a run may be rerun only for an infrastructure failure (crash, OOM, corrupt checkpoint, guard refusal), with the same seed and ladder, and the reason goes in the run record. A finished run is never replaced. Class collapse counts as finished. An investigation may add a disclosed row beside the original | §4.4 | ❌ discursive | #192 |

## Consequences

- **Every claim has a reading that can't be bargained with.** C1 needs four clearances plus a
  gate, C2 has three named outcomes plus a ladder condition, and C3 has two halves with
  stated bars. Every "partial" outcome already has its wording.
- **The build map inherits a bar to build.** It writes `task2-prereg-bar.json` for the ten
  numeric rules, anchors it in §7, and generalises `benchmark.prereg` so the two tasks' bars
  load through one module and can't drift apart in mechanism.
- **The tile CI is knowingly optimistic.** L2 says so up front. A claim near its threshold
  should be read with that in mind, and the memo says so wherever one lands there.
- **B3's third seed moves §8's total to 72.3 h, ~$29.**
- **A new evidence-use rule is a register row.** As in ADR-0013, any later ticket that adds
  one adds an L-row and one marked site, and the tunability test decides admission.

## Alternatives considered

- **A cluster bootstrap over test geocells.** Rejected: 13 clusters. The region-to-region
  variance that matters most, which regions were *labelled*, is carried by the seed spread
  instead (L3), because each seed's ladder labels different cells.
- **Requiring mIoU and macro-F1 to agree.** Rejected: both come from the same confusion matrix,
  so requiring both adds a second way to fail and no information.
- **δ = max(1.0, B7's seed std)**, and δ = 2.0. Rejected under Decision 5.
- **A depth-sweep reversal withdraws C1.** Rejected: the sweep is single-seed and checks
  whether C1 is a last-layer artifact. It qualifies the claim, it doesn't decide it.
- **Folding the register into ADR-0013.** Rejected: ADR-0013 is Task 1's, its rules are
  anchored in Task 1's plan, and amending it to hold a second task's rules would mix two
  sets whose content is that each *is* a set.
