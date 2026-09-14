# P120 — What the benchmark's primitives actually cost

**Ticket:** [#120](https://github.com/JStoschek/DEMAE/issues/120), under map
[#117](https://github.com/JStoschek/DEMAE/issues/117).
**Artifacts:** [`p120-primitive-costs.json`](p120-primitive-costs.json) — every timing
quoted below, plus the projection derived from them.
**Amends:** [`task1-void-fill-plan.md`](task1-void-fill-plan.md) §4.1 and §7.

> **Partly superseded, 2026-09-14 — [#175](https://github.com/JStoschek/DEMAE/issues/175).**
> **Primitive 3 (the UNet epoch) measured 4.5x higher than this memo projected** — 68.9 s, not
> 15.4 s — and everything below that multiplies it is low by that factor on the A7 line. The
> live GPU total is the plan's **5.40 h / $2.17**; this memo's 2.56 h / $1.03 differs by #175
> (+160 min on A7) *and* by two later scope changes it predates (#147's counts, #163 pricing A9),
> so do not read the gap as all one correction.
>
> Specifically wrong below, beyond the A7 line itself:
> - **§1's "scoring is the biggest single line"** (118 CPU-min "against the largest GPU line's
>   46"): both halves fail — the largest GPU line is **206.6 min** and exceeds scoring.
> - **§1's "overstates by 8–13x"**: against a 5.40 h total, §7's old 20–33 GPU-h was high by
>   **3.7–6.1x**. This multiple *shrinks*; it does not scale up.
> - **§6's "the UNet by ~6x"**: §7 guessed 4–6 h and it measures **3.44 h** — high by ~1.2–1.7x,
>   not 6x.
> - **§4's closing "two and a half hours … longest single job is 46 minutes"**: the 46 min was
>   A7's whole 3-seed line, not a job. The longest job is **58 min** (one seed).
> - **§3's "both conservative … biases every derived cost up"**: this is the claim that failed,
>   and it failed in the direction §3 itself flagged one paragraph earlier.
>
> The verdict ("§7 overstates the GPU cost", "$25 is not binding") survives, as §3's own
> sensitivity table said it would at up to 5x. This memo is kept as the record of what was
> projected and why; the live numbers are the plan's §7.

---

## 1. Verdict

**§7 overstates the benchmark's GPU cost by 8–13×, and the line it never priced at all is
now the largest single block of wall clock in the benchmark.** Measured, and projected onto
the host Phase 1 actually rented:

| | §7 as written | Measured (P120) |
|---|---|---|
| GPU | 20–33 "4090-hours" | **2.56 h** |
| GPU dollars | $7–12 at $0.34/hr | **$1.03** at the real $0.4016/hr |
| CPU eval | "CPU" — no number | **2-5 h**, single core, local |

The <$25 target is not binding and never was. §7's own hedge — "inside budget with slack" —
was right for the wrong reason: the slack is an order of magnitude, not a fraction.

The headline figure is for the design §7 now recommends, in which the frozen encoder runs
**inline** and no feature cache is built (§5). The cached alternative is cheaper in GPU
seconds — 1.31 h, $0.53 — and is rejected for reasons that have nothing to do with money.

Three findings ride along:

- **Scoring is the biggest single line in the whole benchmark**: 118 minutes of CPU at
  this ticket's scoring-only rate, against the largest GPU line's 46. The CPU total is
  1.99 h scoring-only and 5.2 h at the end-to-end rate p119 measured on the same machine
  (§4) — between 78% and 2x the entire GPU program, which is not how §7's bare "CPU" reads.
- **The UNet is the priciest single arm**, 46 of the GPU program's 154 minutes. §7's guess
  that A7 is "the priciest baseline" is right; its guess of 4–6 hours is 6× high.
- **A9 is unpriced and is the largest uncertainty left.** It is the one arm this ticket
  could not measure, and a plausible rate for it exceeds every other line combined (§7).

---

## 2. The five primitives

Unit costs on the rented RTX 5090, from which any arm can be costed as unit × count
without re-running anything. §3 explains how a local measurement becomes a 5090 number and
how far that is to be trusted.

| # | Primitive | Unit | Cost |
|---|---|---|---|
| 1 | Frozen-encoder forward, one tile, ViT-B | per tile | **0.179 ms** (5,603 tiles/s) |
| 1 | Frozen-encoder forward, one tile, ViT-S | per tile | **0.060 ms** (16,630 tiles/s) |
| 2 | §4.1 head, one epoch over 20k cached features | per epoch | **1.00 s** (768-d) / 0.96 s (384-d) |
| 3 | §4.4 UNet from scratch, one epoch over 20k tiles | per epoch | ~~15.4 s~~ → **68.9 s** |
| 4 | A3 fine-tune, one epoch at ViT-B | per epoch | **6.66 s** |
| 5 | Full CPU eval pass, all 32,000 eval instances | per arm-seed | **148 s** (4.6 ms/instance) |

**Primitive 3 was superseded by measurement on 2026-09-14 —
[#175](https://github.com/JStoschek/DEMAE/issues/175).** A7 was trained for real on the rented
5090 and a UNet epoch over 20k tiles measures **68.9 s at base 48**, not 15.4 s. The struck
figure is left visible because it is what this memo actually projected and because the *reason*
is this memo's own: not the architecture (§2's table below built the UNet at 17,465,377
parameters, exactly what shipped) but §3's cross-device transfer, which carried the local
batch-32 rate across at **~97x** where the true ratio for this architecture is **21.7x**. §3
predicted this line would be the weakest and predicted the direction; it was low by 4.5x. **Do
not cost a new convnet arm off this table without reading §3 first** — the transformer
primitives (1, 4) are anchored on the 5090 and stand.

The shape of the answer is in the ratios, not the absolutes. **A head epoch costs one
second.** At the measured UNet primitive a UNet epoch costs *sixty-nine* times a head epoch and
ten times an A3 fine-tune epoch — it read as fifteen and two and a half here — which only
sharpens why the one arm with no pretraining in it is the most expensive thing in a benchmark
about pretraining. (The head's own second is itself a transferred number and shares primitive 3's
exposure; see §7 of the plan.)

#120's arithmetic was right about A3 specifically: 20 epochs × 20k tiles at 3,001.2
samples/s is **2 min 13 s**, against §7's 4–8 hours.

**Caveat on the recipe batch.** §4.1 and #120 both specify batch 128. The head and UNet
were transferred from a **batch-32** measurement, because batch 128 does not run on the
measuring device — see §3's cliff. What is transferred is a per-sample rate, which is what
the projection multiplies, but the recipe's own batch was never timed on either machine.

### Where the plan's model sizes are wrong

Measured parameter counts, building §4.1 and §4.4 literally from their prose:

| Module | Plan says | Built | |
|---|---|---|---|
| §4.1 shared head, 384-d input | ~3M | **1,082,017** | 2.8× smaller |
| §4.1 shared head, 768-d input | ~3M | **1,181,089** | 2.5× smaller |
| §4.4 UNet | ~17M | **17,465,377** | matches |

§4.1's "~3M params" does not follow from the architecture §4.1 describes — four conv
stages at (256, 128, 64, 32) behind a 1×1 projection cannot reach 3M at either feature
width. This barely moves the cost (the wall clock is in the two high-resolution stages,
whose widths the prose does fix), but §6 item 4 puts parameter counts in every table, so
the number should be corrected before it is printed rather than after.

---

## 3. How a local measurement becomes a 5090 number

No 5090 was rented for this ticket. It was not needed, because Phase 1 already measured
one primitive on it — the MAE pretraining step — at both capacities the benchmark uses:
**7,291.5 samples/s at ViT-S** (R1, whole-epoch) and **3,001.2 at ViT-B** (R0). Measuring
that same step locally gives a **device ratio**, and every other GPU primitive is carried
across by it.

(#113 re-measured ViT-B at **3,130.2** on the same batch split, 4% higher. The lower figure
is used here, as it is in [`phase1-pretraining-plan.md`](../phase1-pretraining-plan.md)
§12's budget — it is the conservative one.)

Measured ratios, `5090 ÷ this laptop's MPS`, over a batch sweep:

| local batch | ViT-S local | ViT-B local | ratio ViT-S | ratio ViT-B | spread |
|---|---|---|---|---|---|
| 16 | 72.60 | 27.01 | 100.4 | 111.1 | 1.11 |
| 32 | 78.60 | 29.43 | 92.8 | 102.0 | 1.10 |
| **64** | **83.99** | **30.27** | **86.8** | **99.1** | **1.14** |
| 128 | 85.60 | **3.84** | 85.2 | 782.0 | 9.18 |

Two things this table settles:

- **The ratio is stable where the device is.** Across 16–64 the two independent anchors
  agree to within 14%, which is the error bar on everything transferred.
- **Batch 128 is off a memory cliff on this device**, not a data point. ViT-B collapses
  from 30.27 samples/s to 3.84 — an 8× fall. (An earlier UNet run failed outright with
  `MPS backend out of memory (MPS allocated: 17.18 GiB … max allowed: 18.13 GiB)`, which
  is where the ceiling figure comes from; it is an error string, not a recorded metric, so
  it is quoted here rather than in the JSON.) Each primitive is therefore measured across
  the sweep, transferred at the batch where *it* ran fastest per sample, using the ratio
  measured at that same batch. The cliff is published rather than smoothed, because it is
  also why the first two attempts at this measurement hung.

### What the transfer is worth, stated honestly

The ratio bundles two things it cannot separate: a hardware gap, and a software gap (the
5090 ran compiled, with fused attention, at batch 512; this laptop runs eager at 32–64).
A ~90× ratio is large, and it is large partly because of the second term. **For the head
and the UNet there is no same-architecture anchor**, so their transfer is an extrapolation,
and the UNet's is the weaker of the two — convnets do not gain from `torch.compile` what
transformers do.

So the useful question is not "is 97× right" but "how wrong can it be before the verdict
moves". Re-totalling the recommended design with the head's and UNet's ratio divided:

| head+UNet ratio overstated by | GPU total | dollars |
|---|---|---|
| (as measured) | 2.56 h | $1.03 |
| 2× | 4.99 h | $2.01 |
| 5× | 12.29 h | $4.94 |
| 10× | 24.46 h | $9.82 |
| 20× | 48.79 h | $19.60 |

**The "§7 is high" verdict survives a 5× error, and the $25 target survives a 20× one.**
Only if the transfer is wrong by ~10× does §7's 20–33 hour range become reachable, and
even at 20× wrong the bill is $19.60.

Two further properties, both conservative:

1. The 5090 anchors are **whole-epoch** rates and include loader wait; the local numbers
   are compute-only on synthetic tensors. That biases every ratio down and every derived
   cost **up**.
2. `torch.compile` is off locally and was on for the anchors, so the anchors' software
   advantage is already inside the ratio — consistent for primitives that would also be
   compiled, conservative for ones that would not.

### One internal check

A3 is the only primitive measured *both* ways: transferred from this device, and readable
straight off R0. Transferred it gives **3,037.9 samples/s** against R0's **3,001.2** —
agreement to **1.2%**.

This does **not** independently validate the cross-device transfer (A3 *is* structurally
the pretraining step, so the ratio largely cancels). What it establishes is the thing §4.2
asserts and never checked: **swapping the mask sampler and turning `norm_pix` off costs
nothing**, so an A3 epoch is a pretraining epoch and can be priced with R0's number
directly. That is why primitive 4 needs no transfer.

---

## 4. §7's table, rebuilt

Every line is primitive × count. Counts come from §1's arm list as scoped by #117
(A1–A7 + A9; A8 and the 10 m regime out), §3.1's tile counts, §3.2/§3.3's mask counts, and
§4.1/§4.4's schedules. Five (encoder, capacity) pairs carry a trained head — DEMAE-B,
DEMAE-S, TerraMind-B, ImageNet-B, ImageNet-S — of which three are 768-wide and two 384.

GPU, in the recommended no-cache design:

| Item | Count | 5090-min |
|---|---|---|
| **A7 UNet, 3 seeds + label sweep** | 3 × 50 ep × 20k, + 2 × 50 ep × 6k | **46.1** |
| Head tuning on A6 | 12 configs × 50 ep × 20k | 45.7 |
| Head training A4/A5/A6 | 5 cells × 3 seeds × 50 ep × 20k | 45.1 |
| Label-efficiency sweep (1k, 5k; 2 seeds) | 5 cells × 2 seeds × 50 ep × 6k | 9.0 |
| Inference over the eval instances | 47 passes × 32k | 4.5 |
| A3 fine-tune (ViT-B + ViT-S) | 20 ep × 20k | 3.1 |
| **GPU total** | | **153.6 min = 2.56 h → $1.03** |
| **A9 (TerraMind full tokenizer)** | 32k instances | **unpriced** — see §7 |

CPU, on the local machine, single-threaded:

| Item | Count | min |
|---|---|---|
| **Scoring** | **48 arm-seed passes × 32,000 instances** | **118.2** |
| A1 pre-fill for every eval instance | 32,000 | 0.9 |
| §5's paired bootstrap | 28 pairs × 2 metrics × 10k resamples | 0.2 |
| Truth-derivative cache | 5,000 tiles | 0.2 |
| **CPU total** | | **119.5 min = 1.99 h** (2.92 h without the truth cache) |

**Where 48 scored passes comes from.** At 20k: A1 (1) + A2 zero-shot at both capacities
(2) + A3 fine-tuned at both (2) + A9 (1) + A4/A5/A6 at 5 cells × 3 seeds (15) + A7 × 3
seeds (3) = **24**. The label-efficiency sweep adds (5 cells + A7) × 2 seeds × {1k, 5k} =
**24**, because §6 item 7 makes C3's low-label sweep a pre-registered headline claim, so
those cells are scored, not just trained. Every pass but A1's also needs a GPU forward,
hence 47 inference passes.

Scoring is 4.6 ms per instance with the truth tile's derivatives cached across arms and
buckets, and 6.8 ms without; the cache is worth 32% and costs 2.5 ms per tile to build, so
it is the one optimization here that pays for itself.

**An independent read on primitive 5, and it runs the cost up.**
[`p119-fill-headroom.md`](p119-fill-headroom.md) did a real end-to-end scoring pass on this
same machine and corpus — 48,000 instances, CPU-only, **9.7 min wall clock**, i.e.
**12.1 ms per instance**. That is 2.6× this ticket's 4.6 ms, on a *narrower* metric set
(metres and slope; no curvature, aspect, NMAD, p95 or SSIM). The gap is explained rather
than explained away: p119's instance count multiplies in **five fill variants** including
thin-plate spline, which is far costlier than the bilinear `fillnodata` priced here, and
its wall clock carries shard reads and a parquet write that this primitive excludes.

The honest reading is that 4.6 ms is a floor for scoring alone, and a real pass with I/O
lands nearer 12 ms. At p119's end-to-end rate the CPU line would be **5.2 h** rather than
1.99 h. That moves no verdict — it is still not GPU dollars, and it makes scoring an even
more dominant single line — but the CPU row should be read as **2–5 h**, not as a point
estimate.

Two notes on what these numbers do *not* cover:

- The inference line prices an **encoder forward**. A2 and A3 also run the decoder, and
  §4.2 averages **4 shifted encodings** for near-tile-scale voids. Taken at face value that
  could raise the line ~5–8×, to roughly 25–35 min — still a rounding error against the
  UNet, which is why it is stated rather than re-measured.
- 32,000 eval instances = §3.2's two synthetic families × three buckets × 5,000 test tiles,
  plus §3.3's **~2,000** real-void instances at their natural size distribution.

Note what this says about *attention*, which #117 names as the real budget: the entire GPU
program is **two and a half hours**, and its longest single job is 46 minutes. Nothing in
it justifies a spot-instance strategy or a resume design; §8's spot-reclaim risk is about
a job that would have to be unlucky inside a 46-minute window.

---

## 5. The mask-and-cache decision, which this ticket was blocking

**Do not build the feature cache. Run the frozen encoder inline and draw a fresh mask
every epoch.**

The two designs, both priced from the same primitives:

| | Cache (§4.1 as written) | Inline (recommended) |
|---|---|---|
| GPU total | 1.31 h / **$0.53** | 2.56 h / **$1.03** |
| Disk | 30.3 GiB (5.6 GiB per encoder per 20k set) | none |
| Bytes re-read over the program | **7.9 TB** | none |
| Void mask during head training | **frozen for all 50 epochs** | fresh each epoch |

The cache buys **$0.50**. It costs a design constraint §4.1 never names: precomputation
"fixes the A1-filled input per (tile, mask)", and the A1-filled input *is* the encoder's
input, so a cached feature is a feature of one specific mask. §3.2 says train-time masks
are drawn on the fly seeded by `(tile_id, epoch)` — those two sentences cannot both hold.
Either the mask is redrawn per epoch and the cache is invalid, or the cache stands and each
tile is seen under exactly one void for the whole run.

At a $0.50 price difference that trade is not close. The reason the cache looked compulsory
is the reason this ticket existed: §7 believed a forward pass was expensive. It is 0.179 ms.

Two caveats, neither of which reverses it:

- §4.1's other justification is that "reruns cost nothing on the GPU". True, and worth
  0.3 minutes of precompute.
- The 7.9 TB of re-reads is arithmetic over a *measured* per-tile size (301 KB at 768-d),
  not a timing. On a ≥64 GB host one encoder's 5.6 GiB working set stays in page cache, so
  the reads are not disk-bound in practice. The point stands either way: I/O cannot make
  the cache *attractive*, only less bad.

**Task 2 reuse (§9.4) is unaffected** — that constraint is on the frozen-feature
*contract* (which encoder, which layer, which normalization), not on whether the features
were written to disk. Dropping the cache drops an artifact, not an interface.

---

## 6. What §7 got wrong, itemized

- **Wrong card.** §7 prices "4090-hours". Phase 1 ran on a rented **RTX 5090** —
  [`phase1-pretraining-plan.md`](../phase1-pretraining-plan.md) states it in its
  front-matter hardware line.
- **Wrong price.** $0.34/hr assumed; **$0.4016/hr** actually paid, recorded in that plan's
  §12.
- **Wrong magnitude, everywhere.** Every line is high: A3 by ~150×, the frozen arms by
  ~8×, the UNet by ~6×.
- **A missing row.** The CPU eval pass is listed as "CPU" with no number, and scoring is
  the largest single line in the benchmark.
- **A row that should not have been dropped.** §7's eval line reads "Eval passes, hydrology
  metrics". Hydrology is dropped (#117), so the line's *name* is stale even though its cost
  went up.

---

## 7. What this ticket did not price

**A9, the TerraMind tokenizer pass-through, is unmeasured.** It needs `terratorch`,
`diffusers==0.30.0`, and a checkpoint, none of which are local, and it is the only arm
whose cost is not a forward pass: encode → FSQ → **diffusion decode**, over 32,000 eval
instances. At 1 s/instance — modest for a diffusion sampler — A9 alone is **8.9 GPU-h**,
which exceeds the entire measured GPU program by 3.5×. At 10 s/instance it is 89 GPU-h and
it, alone, breaks the $25 target.

This changes neither the verdict on the arms that were measured nor the mask-and-cache
decision, which turns only on the frozen forward pass. It does mean **§7's total is not
final until A9 is measured**, and that measurement is a ticket, not a footnote: it needs
the terratorch path #118 was already exercising.

The other unpriced items are deliberately out of scope per #117: A8's cross-eval and the
secondary 10 m regime.

---

## 8. Method

`prototypes/p120_primitive_cost_proto.py` (gitignored per this repo's throwaway-prototype
convention; every timing it produced is in
[`p120-primitive-costs.json`](p120-primitive-costs.json)).

- **Device:** Apple MPS, torch 2.13.0, macOS 15.7.3, 8 CPU cores. bf16 autocast
  throughout, matching §4.1/§4.2/§6.
- **Shapes:** §4.1's head and §4.4's UNet built literally from their prose; the MAE from
  `pretrain.model.MaskedAutoencoder` under `RunConfig()` / `RunConfig.vit_base()`, so the
  encoder timed is the one the project ships.
- **Timing:** median of 8–12 steps for the four GPU primitives and 5 steps for the anchor
  sweep, each after 3 warmup steps, with an explicit device synchronize on both sides of
  every interval — without it MPS times the enqueue, not the work. Each primitive starts
  from a released allocator: on the first pass the UNet's high-water mark followed the next
  measurement around and halved it, which is why A3 initially read 14 samples/s where the
  identical pretraining step read 30.
- **Eval pass:** 400 real tiles from `corpus_500k/shards/shard_00000.npy`, one random-walk
  blob per §3.2 size bucket per tile, §3.2's two constraints (≥8 px interior, ≤40% area)
  asserted per mask. Metrics are §5's minus the dropped hydrology: MAE / RMSE / NMAD / p95,
  slope RMSE, slope-weighted circular aspect error, profile-curvature RMSE, and hillshade
  SSIM inside the void. SSIM is the standard uniform-window formulation over scipy —
  scikit-image is not a project dependency, so this is also what the benchmark would have
  to ship.
- **A1 pre-fill:** `rasterio.fill.fillnodata`, the arm itself and every learned arm's input.

### Threats to these numbers

1. **The head/UNet transfer is an extrapolation** (§3). Quantified: the verdict survives
   5× error, the budget 20×.
2. **The recipe batch of 128 was never timed**, on either machine — it does not run on the
   measuring device (§3). Per-sample rates are what the projection multiplies, but a
   batch-128 measurement on the 5090 would be the thing that closes this.
3. **`torch.compile` is off locally and was on for the anchors**, so the ratio carries the
   anchors' compile gain. Conservative for anything that would not be compiled.
4. **The eval pass is single-threaded and on a laptop**, and 1.99 h is the scoring-only
   floor — p119's end-to-end rate on the same machine puts it at 5.2 h (§4), so read the
   CPU row as 2-5 h. The host has 8 cores and it parallelizes trivially over instances.
5. **`_blob_mask` is a stand-in for `benchmark.masks`, which does not exist yet.** It
   produces §3.2-legal blobs of the right size, which is what the cost depends on; it is
   not the committed generator and makes no claim to be.
