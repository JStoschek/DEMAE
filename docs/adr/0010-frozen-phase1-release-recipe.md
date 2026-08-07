# ADR 0010 — The frozen Phase-1 release recipe, and the decision rule that adjudicates arms

- **Status:** Accepted — §1's physical-batch row **amended** (nothing superseded); see [Amendments](#amendments)
- **Date:** 2026-08-05
- **Context ticket:** "Decide the release recipe" ([issue #85](https://github.com/JStoschek/DEMAE/issues/85)), part of the Phase-1 pretraining map ([#76](https://github.com/JStoschek/DEMAE/issues/76))
- **Evidence:** R1 ([#82](https://github.com/JStoschek/DEMAE/issues/82)), R2 ([#83](https://github.com/JStoschek/DEMAE/issues/83)), R3 ([#84](https://github.com/JStoschek/DEMAE/issues/84)), throughput and memory from R0 ([#81](https://github.com/JStoschek/DEMAE/issues/81))
- **Amends:** [`docs/phase1-pretraining-plan.md`](../phase1-pretraining-plan.md) §3, §4, §6, §10, §12
- **Builds on:** [ADR 0009 — One shipping entry point](0009-one-shipping-entry-point.md), which is what makes a run's config the only thing that varies between arms

## Context

Plan §0's decision table deliberately left five entries unsettled, to be closed by evidence
rather than citation. Three of them are ablations, and all three have now run:

| Run | Arm | Epochs | Cost |
|---|---|---|---|
| R1 | 75% mask, `norm_pix` ON — the baseline | 400 | $2.90 |
| R2 | 90% mask, else R1 | 200 | $1.10 |
| R3 | `norm_pix` OFF, else R1 | 200 | $1.44 |

All three ran clean: no interruption, no resume, no §10.1 alarm, and `probe/set_md5`
byte-identical across the three, so every arm is scored on the same 5,000 val tiles.

At **epoch 180** — the last probe epoch all three share — R1 wins every metric:

| Metric | R1 (75%, norm_pix ON) | R2 (90% mask) | R3 (norm_pix OFF) |
|---|---|---|---|
| `stratum_id` kNN accuracy ↑ | **0.3914** | 0.3718 | 0.3874 |
| `slope_mean` kNN MAE ↓ | **2.0344** | 2.2466 (+10.4%) | 2.2680 (+11.5%) |
| `geomorphon_entropy` kNN MAE ↓ | **0.25265** | 0.25918 (+2.6%) | 0.28843 (+14.2%) |

So the surviving recipe is R1 exactly, and there is no untested knob pair for the release
run to inherit — the ablations were one-factor-at-a-time, and both factors lost, so the
winner is the cell already trained to 400 epochs with a full probe trajectory.

That much is arithmetic. What made this a decision ticket rather than a readout is that
**reading those numbers required a rule §10 does not actually supply**, and that the recipe
was proven at ViT-S while the release run is ViT-B.

## Decision

### 1. The recipe is frozen at R1's, transposed to ViT-B

R5, the release run:

| Knob | Value | Basis |
|---|---|---|
| Backbone | ViT-B/16, `in_chans=1`, patch 16, 196 tokens | §2.1 |
| Masking | **75%**, uniform random, per-sample; mask count 147 fixed at construction | §3, confirmed by R2 |
| Decoder | 4 blocks × 512, 8 heads | §2.2 — always ViT-B's value, never a ViT-S finding |
| Loss | MSE on masked patches only, **`norm_pix` ON**, fp32 target stats, variance floor 1e-4, ocean/invalid pixel weighting | §4, confirmed by R3 |
| Input normalization | `per_tile` | §5, untouched |
| Optimizer | AdamW, betas (0.9, 0.95), weight decay 0.05, fused | §6 |
| Physical batch | ~~**1024**~~ → **512** *(Amendment 1 — 1024 OOMs; 17.82 GiB was the 512 measurement)* | measured 18.16 GiB peak of 31.36 available |
| Grad accumulation | ~~2~~ → **4** → effective batch 2048 → `lr` **1.20e-3** *(Amendment 1; effective batch and `lr` unmoved)* | §6's linear scaling rule, `lr` derived not typed |
| Schedule | **400 epochs**, warmup **20**, cosine to exactly 0 | §6; warmup is the 5% rule, 20 is its output at 400 |
| Precision / kernels | bf16 autocast, no GradScaler, SDPA, `torch.compile(dynamic=False)` | §7 |
| Grad clipping | **none** | §6 permitted clip 1.0 only if bf16 instability appeared; four runs, zero alarms |
| Augmentation | full dihedral, no RandomResizedCrop | §6; nothing in R1–R3's grids argued otherwise |
| EMA / dropout / droppath | none / 0 / 0 | §6 |
| Init | `scratch` | §8 |
| Milestones | 50 / 100 / 200 / 400 | §14 |

R4 is that table with exactly three changes: `init.mode = imagenet_mae` against OpenMMLab's
Apache-2.0 checkpoint, `pos_encoding = sincos2d_rowfirst` (those weights learned mmselfsup's
transposed ordering — [P102](../research/p102-mmselfsup-mae-layout.md) §4), and the schedule
change in §4 below.

**Two hedges in §6 are now dead, and are recorded as dead rather than left to be re-litigated.**
~~"Drop to 512 for ViT-B if needed" is not needed — R0 measured ViT-B at batch 1024 peaking at
barely half the card.~~ *(**Retracted by Amendment 1.** This paragraph is wrong on its face and
is the kind of sentence a later reader trusts: 1024 OOMs, 17.82 GiB is what 512 costs, and the
argument this section made for retiring the hedge applied to the batch it was retiring in
favour of. Left visible rather than deleted, so an old checkout's reader can find the
retraction.)* And "add grad-clip 1.0 only if bf16 instability appears" never fired: four runs,
two backbones, two mask ratios, two loss targets, no loss spike and no alarm — **that half
stands.**

### 2. The mask ratio is 75%, and the bound is stated rather than closed

90% lost on all three probes, so §12's gate for the 85% arm — *"85% only if 90% ≈ 75%"* —
does not open, and no 85% arm runs.

**A below-75% arm is ruled out of Phase 1**, as a scope call rather than an evidence call.
It would cost roughly 4.5 GPU-hours and $2 (at 65% masking the encoder sees ~69 patches
rather than 49, so it is *slower* than R1, not faster), which the envelope could absorb. It
is declined because the prior is low on the plan's own citations — MAE's sweep peaks at 75%,
Scale-MAE found 70–80% indistinguishable — and because the finding already banked does not
get stronger for it. "75% beats 90% on elevation" is worth reporting precisely because it
*contradicts* the prediction: SpectralGPT's ablation rose monotonically to 90% on Sentinel-2,
per-tile-standardized elevation is smoother than Sentinel-2, so 90% should have won here and
did not. "75% also beats 65%" adds nothing to that.

**The release states the bound it actually has:** the optimum is at or below 75%; 90% was
ruled out by measurement; downward was not searched. That sentence is checkable, which is
the point.

### 3. §10's decision rule is split in two

§10's rule reads *"kNN-probe trend on `stratum_id` + val recon loss, jointly."* Read
literally, that rule gets R3 wrong. `stratum_id` was 0.3914 vs 0.3874 — 0.67 noise widths,
a tie — and val loss was 0.4466 vs 0.09316, which is not a comparison at all, because
`norm_pix` OFF changes what the loss is computed over. Both named instruments report "no
difference" on an arm decided by 11.5% and 14.2% margins on the metrics §10 lists *second*.

It is not a quirk of R3. `stratum_id` takes ~92% of its total gain by epoch 40 in **all
three** runs (92.0 / 92.3 / 90.9%), so the saturation is a property of the probe rather than
of any arm, while both regression probes improve to the end of every run. R2 is the case
that bounds the correction: there the categorical probe *did* separate the arms at 3.3 noise
widths, just least (5.0% relative against `slope_mean`'s 10.4%). So the fix is not to demote
`stratum_id` — it is to stop treating it as the lead.

The rule becomes two rules, because §10's one sentence was doing two different jobs:

- **Across arms (ablation winners, R4-vs-R5).** All three probes read **jointly at a matched
  epoch on a byte-identical probe set**, with no lead metric. **Val loss is admissible only
  when the loss target and mask ratio are identical** across the arms compared — which
  excludes it from R2-vs-R1 and R3-vs-R1 entirely, and admits it for R4-vs-R5.
- **Within one arm (the epoch-budget call).** Val loss stays valid, exactly as §10.1 always
  claimed: the fixed masking seed makes it comparable across epochs of one run. Probe trends
  are read alongside it.

Reconstruction prettiness remains not a criterion, unchanged.

### 4. R4 runs 400 epochs, on R5's schedule, and R5 is blocked by R4

Two problems fall out of applying the cross-arm rule to §12's run table as written, where
R4 is 150 epochs and R5 is 400.

**R4's final epoch is never probed.** The probe fires on `epoch % 20 == 0`
([`probe.py:942`](../../src/pretrain/probe.py#L942)). R4 stopping at 150 gives probe points
20…140 and no measurement at its own endpoint. The last epoch the two arms share is 140.

**At epoch 140 the two runs sit at incomparable points in their schedules.** R4 would be
140/150 through its cosine — nearly annealed — against R5's 140/400, still at high LR. The
anneal is where a run consolidates; R1's own 200→400 gain was concentrated in its cosine
tail. So R4 would carry an advantage having nothing to do with the warm start, in the
direction §8's hypothesis predicts. This is the same asymmetry R2-vs-R1 had, but inverted in
consequence: there it ran *against* the arm being tested, and R2 lost anyway, which is what
made it dismissible. A confound that flatters the hypothesis is not dismissible.

**So R4 runs 400 epochs with warmup 20 — R5's schedule exactly** — at a cost of roughly 11
extra GPU-hours and $4.35. Every 20-epoch probe becomes a valid paired comparison rather
than one confounded reading, and the arm's stated premise ("hold the recipe fixed, move only
the initialization") becomes literally true for the first time. GFM's efficiency claim
survives for free: it is R4's epoch-150 point against R5's epoch-400 point, off the same
pair of runs.

**R5 is blocked by R4**, as a tracker edge rather than a convention. R4 is the same recipe at
the same scale for a quarter of R5's cost, so it is a near-free canary for a recipe never yet
trained to length at ViT-B — and, per §5 below, it is the evidence R5's epoch call needs.

**R4's only early kill is a §10.1 alarm** — loss collapsing toward zero, val loss rising
while train falls, or a first-20-epoch plateau at high loss. A flat-looking probe curve
mid-cosine is **not** a kill signal: at epoch 200 of a 400-epoch schedule the LR is still 54%
of peak (6.5e-4 of 1.20e-3), R1 measured its 200→400 gains as concentrated in the tail, and a
run stopped mid-cosine yields an unannealed checkpoint that is neither comparable to R5 nor
releasable. The second half of R4 costs 8.75 GPU-hours and $3.50, which is not worth a
compromised headline comparison.

### 5. "Extend to 800 on evidence" is replaced by a pre-launch call

§6 says: run 400 epochs, extend to 800 only if the kNN probe and val loss are still improving
at 400. The schedule cannot honour that.
[`schedule.py:102-107`](../../src/pretrain/schedule.py#L102) reaches exactly 0.0 at the final
epoch and clamps to 0 beyond it. So at epoch 400 the curve has flattened *by construction* —
the rule asks a question its own schedule guarantees the answer to — and training past 400
applies zero-sized updates. A genuine 800-epoch run is a **different run**, launched from
step 1 with an 800-epoch cosine; an epoch-400 checkpoint taken from an 800-epoch run is not
the same artifact as a finished 400-epoch run, because it has not annealed.

**R5's epoch budget is 400 by default, and the 400-vs-800 choice is made before R5 launches,
not after it finishes.** The evidence for that call is R4's ViT-B trajectory, which exists
because of §4 above; R1's ViT-S curve was never going to settle a ViT-B question. R1's
measured doubling — 200→400 bought val 0.0077 (1.7%), `stratum_id` +0.014 (~2 noise widths),
`slope_mean` −7.3% — is the prior, and it argues 400. The counter-argument, recorded rather
than dismissed: 400 epochs × 473k tiles is ~148 ImageNet-epochs of samples against MAE's
800–1600, and §6's own citation has linear-probe quality climbing long after fine-tune
quality saturates, which is the axis we adjudicate on.

If 800 is chosen, §14's milestone list gains an 800 entry — a release-artifact contract
change that [#88](https://github.com/JStoschek/DEMAE/issues/88) must be told about, not a
config tweak.

## Consequences

- **The recipe is closed.** R4 and R5 differ from R1 by backbone, decoder width and
  initialization, and by nothing else. Any further deviation is a plan amendment, not a flag.
- **Two cheap facts are now load-bearing:** the probe set must stay byte-identical across
  arms (`probe/set_md5`), and arms compared across the new rule must share probe epochs.
  Both held by luck rather than by construction in R1–R3; R4 and R5 sharing a schedule makes
  the second structural.
- **§12's budget is comfortable.** ~$7 of measured spend across R0–R3 (R1 $2.90, R2 and R3
  $2.64 together, R0 the remainder); R4 at 400 (~$7.00) plus R5 at 400 (~$7.00) lands near
  $21 against a $50–70 envelope, with 800 costing ~$7 more if chosen.
- **The wall-clock is not.** R4 → R5 sequentially is ~35 GPU-hours of watched box, on a rig
  whose teardown script currently reports success without destroying the instance
  ([#111](https://github.com/JStoschek/DEMAE/issues/111)) — which is why that ticket blocks
  both runs.
- **The release carries a bound, not a claim, on mask ratio.** Anyone asking "did you look
  below 75%?" gets an honest no with the reasoning attached.
- **`stratum_id` keeps its place but loses its primacy.** Three runs' worth of evidence say
  it is the least discriminating of the three probes on this corpus while being the one §10
  led with. Phase 2's fine-tuning remains the final adjudicator (§13), unchanged.

## Alternatives considered

**Keep §10's rule and record a caution instead of amending it.** Rejected. R4-vs-R5 is a
cross-arm readout and the release's headline result; a rule that would have called R3 "no
difference" cannot be left standing in front of it on the strength of "read it generously".

**Run a 65% mask arm to bracket the optimum.** Rejected on scope, not cost (§2). The bracket
is stated as a bound instead.

**Compare R4 and R5 at epoch 140 with a caveat.** Rejected — the confound favours the
hypothesis, so an R4 win would be uninterpretable and only an R4 loss would carry
information. Paying $4.35 to get a symmetric comparison is the better trade.

**Compare R4 and R5 endpoint to endpoint (R4@150 annealed vs R5@400 annealed).** This is
closer to GFM's actual efficiency claim and both runs would be annealed, but it needs a
forced probe at epoch 150 (not a multiple of 20) and it answers only the efficiency question,
not the like-for-like one. Running R4 at 400 yields both readings from one pair of runs, so
this was subsumed rather than rejected.

**Pre-commit to 400 for R5 here and close the question.** Defensible on R1's measurement
alone, and it was the simplest option. Declined because R4's ViT-B trajectory is strictly
better evidence and arrives before R5 launches anyway, so pre-committing spends a decision
early for nothing.

**Leave §6's extension wording alone and accept that a later 800 means paying for 400 twice**
(~52.5 GPU-hours, ~$21). Rejected: the schedule makes the "decide at 400" reading impossible
rather than merely expensive, so leaving the wording would preserve a rule nobody can follow.

**Kill R4 early if its curve flattens around epoch 200.** Rejected on measurement: the LR is
still 54% of peak there, R1's gains lived in the tail, and the resulting checkpoint would be
unannealed — reintroducing, against R4, the very confound §4 spends money to remove.

## Amendments

### Amendment 1 (2026-08-06) — ViT-B's physical batch is 512, because 1024 does not fit

**Context ticket:** "ADR-0010 §1's batch 1024 does not fit: 17.82 GiB is the 512 measurement"
([issue #113](https://github.com/JStoschek/DEMAE/issues/113)). **Found by:** R4
([#86](https://github.com/JStoschek/DEMAE/issues/86)), which OOM'd on its first training step.

§1's table froze ViT-B's physical batch at 1024, citing *"R0 measured 17.82 GiB allocated of
31.35 available"*, and retired §6's drop-to-512 hedge on that basis. **The citation is a real
measurement of a different batch.**

| `RunConfig.vit_base()`, RTX 5090, 31.36 GiB | peak allocated | result |
|---|---|---|
| bs=1024, compile off | **30.53 GiB** | **OOM** |
| bs=1024, compile on | **30.54 GiB** | **OOM** |
| bs=512, compile off | 17.20 GiB | ok |
| bs=512, compile on | 16.71 GiB | ok |

Forward and backward only. Add AdamW's two fp32 moments over 98.4M parameters (~0.79 GiB) and
512 lands at ~18.0 GiB against §1's cited 17.82 — reproducing it to within 2% — while a full
step at 512 × 4 measures **18.16 GiB**. 1024 misses the cited figure by 12 GiB, and its own
real requirement is ~31.3 GiB of 31.36, so it had no path to fitting even had it not OOM'd.
Compile is not the cause; the OOM is identical to two decimal places with it off. Neither is
the warm start: this is `vit_base()`, R5's scratch preset.

**The mechanism, because it is the part worth not repeating.** R0's readout states both
backbones ran *"at physical batch 1024 × 2 accum."* `RunConfig.vit_base()` still carried the
512 × 4 contingency at that time — as §1 itself observed, when it removed it. So R0 measured
512 × 4 and labelled it 1024 × 2, and **both** of R0's ViT-B numbers, the 17.82 GiB and the
3,001.2 samples/s, are 512 × 4 figures wearing a 1024 × 2 label. §1 read the mislabel as a
stale hedge and moved the preset to match the label. Its reasoning was right and ran the wrong
way: it retired 512 for being *"a configuration R0 measured nothing at,"* and that sentence
became true of 1024 the moment it was written.

R0's own checklist ([#81](https://github.com/JStoschek/DEMAE/issues/81)) demanded exactly this
record — *"physical batch drops to 512 if 32 GB demands it — record which."* It was asked for.
It moved.

#### What changes

`RunConfig.vit_base()` ships **512 × 4**. Effective batch stays 2048, so §6's linear rule still
derives `lr` 1.20e-3 and **no knob downstream of it moves**. Throughput at 512 × 4 re-measures
at 3,130.2 samples/s — 4% *above* the figure §12's GPU-hour budget is built on — so R5 stays
~17.5 GPU-h and the budget is left on 3,001.2 rather than re-based on the cheaper number.

This is a value, not a resurrected contingency: 512 is the only split that runs, so §6's
"drop to 512 **if needed**" wording is retired in the other direction rather than reinstated.

#### What does not change, and one thing that was never true

R4 ran 400 epochs at 512 × 4 by command-line override, ahead of this decision and recorded
rather than hidden (#86). So the arms already agree, and nothing needs re-running.

**R4's readout asks that R5 match `config_hash` `f0b87f27a24e3a65` "or the arms are not
comparable." No R5 can.** `config_hash` covers the recipe and excludes only the eight
environment fields; R4 and R5 differ by design on three hashed ones — `init.mode`,
`init.checkpoint` (hashed deliberately, since which checkpoint the arm warm-starts from *is*
the arm), and `architecture.pos_encoding`. The two arms could never share a hash, before this
amendment or after it.

What comparability actually requires is weaker, and is stated here so it is not restated
wrongly: **the same effective batch and the same derived `lr`**. With no batch normalization
anywhere and §4's `norm_pix` statistics computed per patch rather than per batch, N
micro-batches accumulated at 1/N give the gradient of one batch N times the size — which
CONTEXT.md asserts and a test pins, rather than assuming. The micro-batch split is therefore
free with respect to the result. R5 runs 512 × 4 regardless, because it is the only split that
fits; that it also matches R4 is a convenience, not the requirement.

#### The check that was considered and declined

§1's failure suggests a build-time assertion — a config that cannot fit the card refusing
before the meter starts. **Declined.** On a rented box the OOM already arrives about sixty
seconds in, before anything meaningful is spent, and `--dry-run` runs on a laptop with no
5090, so "fails earlier" buys close to nothing here. The OOM *is* the check.

What a pre-check would have bought is the one thing an OOM never reports: a **margin**. A
config peaking at 30.9 GiB of 31.36 starts cleanly and dies at epoch 137 when fragmentation or
a monitoring pass adds a few hundred megabytes. That is worth having, but it is worth having
as **measurement in the record** rather than as a gate — which is also the fix aimed at what
actually failed here, since nobody's code was wrong in R0: a human copied a batch size from
the plan into a readout while the preset ran another. Peak memory and the batch split, read
off the run and logged where a readout writer will copy them, are
[#115](https://github.com/JStoschek/DEMAE/issues/115), which blocks R5.
