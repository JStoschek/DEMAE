# P121 — the mask contract, and what replaces the feature cache

Ticket [#121](https://github.com/JStoschek/DEMAE/issues/121) · map
[#117](https://github.com/JStoschek/DEMAE/issues/117) · plan
[`task1-void-fill-plan.md`](task1-void-fill-plan.md) §3.2, §3.5, §4.1, §4.5, §5, §7, §8, §9.4

> **What this document is.** A measurement, not a proposal. The ticket asked four
> questions and assumed three of them were arguments; three turned out to be arithmetic
> the plan had never done. Every number below comes from importing the *committed* mask
> generator — `prototypes/p119_fill_headroom_proto.py`, unmodified — and timing or
> probing it, so the numbers describe the generator the plan will actually freeze rather
> than a re-implementation of it. Raw output:
> [`p121-mask-contract.json`](p121-mask-contract.json); harnesses at
> `prototypes/p121_masks.py` and `prototypes/p121_bank.py` (`prototypes/` is gitignored).

## 1. Bottom line

1. **The feature cache was already dead ([#120](https://github.com/JStoschek/DEMAE/issues/120)); what nobody priced is the thing that replaced it.** Deleting the cache turned mask generation into a per-epoch cost. A draw averages **2.00 ms**, §7's own counts imply **34.4M training tile presentations**, and the product is **19.1 CPU-hours** — larger than the eval pass §7 calls its largest line, and 7.5× the entire 2.56-hour GPU program.
2. **On the fly, head training is loader-bound — the exact property the cache was supposed to buy.** Per epoch over 20k tiles: masks **40 s** on one core against a **4.6 s** GPU step (head 1.0 s + inline ViT-B 3.58 s). That is **8.7×**, and eight dataloader workers bring it only to parity.
3. **So bank the masks, not the features.** K = 50 RLE draws per head-train tile is **1M draws, 33 minutes once, 740 MB**, read by every arm, seed and config — a **34.4× cut** that preserves per-epoch variation *exactly*, since 50 committed masks per tile is what 50 on-the-fly epochs produce. Against the cache it replaces (30.3 GiB, 7.9 TB of re-reads, **one** mask per tile forever) it is 40× smaller and strictly more faithful to §3.2's intent.
4. **The 141 px contradiction is a units error, and the large bucket survives.** The bucket is **equivalent diameter**, `d = 2·√(area/π)`, which is how the committed generator already reads it. Under that reading the 40% cap permits `d ≤ 159.86 px`, so the published top of 160 overshoots by **0.14 px** — not by 19. It is lowered to **158** so nothing is ever clamped.
5. **§9.4's cross-task reuse claim was never true, and the cache decision is not why.** Every Task 1 arm's encoder input is an A1-filled **voided** tile; Task 2 probes **clean** tiles. Task 1 computes no clean-tile feature at any point, so there was never an artifact to share. What transfers is the **extractor contract**.
6. **The synthetic eval count is 30,000 and clears §5's floor with room; one real-void sub-row probably does not.** 1,250 instances per (row, band) synthetic, against a floor of 100. #126's real-medium+ sub-row projects to **~96 in total** — under the floor before it is even stratified.

## 2. The size bucket: two readings, and only one of them is the generator's

[p124](p124-voidfill-comparability.md) §9 reported §3.2's constraints as *jointly
unsatisfiable above 141 px*, deriving 141 from `√(0.40 × 224²)` — the largest **square
side** the area cap permits. That is a correct calculation of the wrong quantity. The
committed generator reads the buckets as equivalent diameter and **area-matches both
families through it**, precisely so that a blob and a rectangle in the same bucket cover
the same pixel count.

| Reading of "160 px" | Void area | Share of a 224 px tile | Verdict against the 40% cap |
|---|---|---|---|
| **Equivalent diameter** (the generator's) | 20,106 px² | **40.07%** | over by 0.07 pp — cap binds at `d = 159.86` |
| Square side | 25,600 px² | 51.02% | over by 11 pp — cap binds at side 141.67 |

A rectangle of equivalent diameter 160 comes out roughly 142 × 141 and sits well inside
the 208 px interior. So the upper half of the large bucket is not unreachable; only its
top **0.14 px** is.

**Measured rather than argued** — 300 draws per (family, bucket) cell, both families, all
three buckets:

| cell | achieved `d` (min/mean/max) | area fraction (max) | max bbox side | cap violations | interiority violations |
|---|---|---|---|---|---|
| blob/small | 16.16 / 33.05 / 47.91 | 0.0359 | 76 | 0 | 0 |
| blob/medium | 64.02 / 80.09 / 95.99 | 0.1442 | 174 | 0 | 0 |
| blob/large | 128.05 / 143.90 / 159.86 | **0.4000** | **204** | 0 | 0 |
| rect/small | 15.64 / 31.40 / 47.87 | 0.0359 | 59 | 0 | 0 |
| rect/medium | 63.69 / 79.60 / 95.63 | 0.1431 | 119 | 0 | 0 |
| rect/large | 128.05 / 144.24 / 159.61 | 0.3988 | 195 | 0 | 0 |

Zero violations of either constraint. The generator reaches `d = 159.86` exactly because
it clamps there — which is the 0.14 px the plan was pre-registering and then quietly
breaking. **Lowering the bucket top to 158 removes the clamp**, costs 1.2% of the large
bucket's diameter range, and leaves both upper buckets bracketing Diff-DEM's protocol.

### 2.1 The large bucket is cramped, and that is worth stating

Two properties of the top of the large bucket, both measured, neither fatal:

| quantity | large-bucket top decile | elsewhere |
|---|---|---|
| blob walks per accepted mask | **1.44** | 1.00 |
| legal placements inside the interior (median) | **950** | 43,264 for a point |
| legal placements (min) | **150** | — |

The largest voids are close to centred. Instance-to-instance variation in that bucket is
therefore terrain, not placement, and the bucket carries less independent information per
instance than its count suggests. This is a reason to read the large-bucket row
cautiously, not a reason to drop it — and it compounds §5's existing tie-break rule
(drop a size bucket before a ruggedness band).

## 3. What a mask draw costs

Timed 400 draws per cell, single core, laptop.

| cell | mean | median | p95 | max |
|---|---|---|---|---|
| blob/small | 0.626 ms | 0.588 | 1.03 | 3.73 |
| blob/medium | 2.387 ms | 2.366 | 3.38 | 4.01 |
| **blob/large** | **8.861 ms** | 7.794 | 16.15 | **30.93** |
| rect/small | 0.037 ms | 0.037 | 0.038 | 0.064 |
| rect/medium | 0.037 ms | 0.037 | 0.038 | 0.059 |
| rect/large | 0.038 ms | 0.038 | 0.040 | 0.179 |

**The mean of 2.00 ms hides a 240× spread.** A rectangle is a slice assignment; a blob is
a random walk rasterized and then dilated one ring at a time until it reaches the target
area, so its cost grows with the area it must reach. Any cost model that assumes a single
per-draw number will be wrong by two orders of magnitude at one end or the other of the
family × bucket grid.

### 3.1 Against the GPU step it feeds

| per epoch, 20k tiles | seconds |
|---|---|
| head forward/backward (p120) | 1.00 |
| inline frozen ViT-B, 0.179 ms/tile (p120) | 3.58 |
| **mask draws, one core** | **40.0** |
| mask draws, eight workers | 5.0 |

### 3.2 Against the whole program

§7's amended table, expanded to tile presentations:

| line | presentations |
|---|---|
| Head training A4/A5/A6 (5 cells × 3 seeds × 50 ep × 20k) | 15,000,000 |
| Head tuning on A6 (12 configs × 50 ep × 20k) | 12,000,000 |
| A7 UNet (3 seeds × 50 ep × 20k, + 2 × 50 ep × 6k) | 3,600,000 |
| Label-efficiency sweep (5 cells × 2 seeds × 50 ep × 6k) | 3,000,000 |
| A3 fine-tune (ViT-B + ViT-S, 20 ep × 20k) | 800,000 |
| **total** | **34,400,000** |

At 2.00 ms that is **19.1 CPU-hours**. The bank does 1,000,000 draws — **33 minutes** —
and every one of the 34.4M presentations reads from it. Ratio **34.4×**.

RLE size, measured on the generator's output (int32 run boundaries):

| cell | runs | bytes |
|---|---|---|
| blob/small | 80.9 | 324 |
| blob/medium | 203.9 | 816 |
| blob/large | 359.0 | 1,436 |
| rect/small | 58.2 | 233 |
| rect/medium | 144.9 | 580 |
| rect/large | 262.7 | 1,051 |

Mean 740 B → **740 MB** for 20k tiles × K = 50.

### 3.3 Why this is the same decision #120 made, not a reversal of it

#120 deleted a 30.3 GiB cache to save $0.50 of GPU. #121 builds a 740 MB one to save 19.1
CPU-hours. The rule underneath both: **cache what is cheap to store and expensive to
recompute.** Frozen features are the reverse — expensive to store (30.3 GiB, 7.9 TB of
re-reads) and cheap to recompute (0.179 ms). Masks are 740 MB and 2.00 ms, and they are
recomputed 34 times over.

The bank also does something on-the-fly draws only *probably* do: it makes the training
mask sequence a **committed input**. Under `(tile_id, epoch)` seeding every arm sees the
same masks so long as every run enumerates epochs identically and the generator's RNG
never changes; under the bank that is a property of a file with a config hash. Fairness
was never at risk either way — the seed does not depend on the arm — but reproducibility
moves from a convention to an artifact.

**What the bank does not do** is reintroduce what killed the feature cache. That cache
fixed **one** mask per tile for all 50 epochs. The bank holds 50, one per epoch, drawn
across families and buckets exactly as §3.2 intends.

## 4. §9.4: Task 2 was never going to reuse this

§9.4 read: *"the same frozen-feature cache feeds Task 2's probes unchanged, which is why
this ordering is efficient."* This fails on the data, independently of #120:

- Every Task 1 arm's encoder input is an **A1-filled voided** tile (§2.4.4 — every learned
  arm predicts a residual over the A1 fill).
- Task 2's probes run on **clean** tiles ([`task2-landform-plan.md`](task2-landform-plan.md) §3.3).
- Task 1 therefore produces **zero** clean-tile features. There is no subset of a Task 1
  cache that Task 2 could have read.

What genuinely transfers is the **extractor contract**: the code path from tile to frozen
patch tokens, the per-arm normalization, the layer indices, the dtype. p118 already
measured why that contract is non-trivial — TerraMind takes raw metres under its global
affine and accepts the pretraining loader's per-tile-standardized output *silently*,
returning a plausible `[1, 768, 14, 14]` either way. Under a shared cache that made
"which normalization is this file?" an unsolved design question; with no shared cache it
is a per-arm config field and the question dissolves.

`task2-landform-plan.md` carries the stale claim in three places — §3.3, §4.2 (~25 GB of
intermediate-layer cache), §4.4 step 3 — and is **not** amended here: Task 2 is out of
this map's scope and gets its own map. Its caching decision is genuinely open and may go
the other way, since UperNet probes read 4 layers per tile over 30 epochs, a very
different read pattern from one inline forward.

## 5. Eval instance counts against §5's floor

The synthetic side is fixed by §3.1 and §3.2 and is not in doubt. The real-void side is
**not fixed by anything the plan currently commits** — §3.3 says "instance budget per
§3.2's eval-instance contract, split as drawn", and since #126 reshaped the family to its
natural distribution, its total is whatever #127's granules yield.

| row | instances | per (row, band) | §5 floor of 100 |
|---|---|---|---|
| synthetic, 2 families × 3 buckets × 5,000 tiles | **30,000** | 1,250 | clears with room |
| real-small (16–48 px) | ~3,800 (projected) | ~940 | clears |
| **real-medium+ (64–96 px)** | **~96 (projected)** | **~24** | **fails, and fails before stratifying** |
| real 128–160 px | 0 in both measured cells | — | already a bare count (§3.3) |

The projection is p125's per-cell component counts at true 1 arc-second — **139 and 175**
in 16–48 px, **2 and 6** in 64–96, **zero** in 128–160 — scaled by
[#127](https://github.com/JStoschek/DEMAE/issues/127)'s 24 granules. Two cells is a thin
base and the estimate may be wrong in either direction, which is why the **threshold** is
pre-registered rather than the outcome: real-medium+ reports unstratified — one pooled
number carrying its instance count — unless the realised histogram clears 100 per band.
[#135](https://github.com/JStoschek/DEMAE/issues/135) checks against exactly that number.

**A side consequence worth flagging rather than fixing here: §7's "32k eval instances" is
now provisional.** It dates from the pre-#126 §3.3 shape (~2,000 real-void instances on
top of 30,000 synthetic). At the natural distribution the real family projects to ~3,900,
which would put the total nearer 34k. The affected §7 lines are the inference row
(4.5 GPU-min) and the CPU eval row (2–5 h), so a 6% move changes no conclusion — but the
number is #135's to settle, not this ticket's to guess.

## 6. Amendments applied in this branch

| § | change |
|---|---|
| **3.2** | Buckets stated as equivalent diameter; large bucket top 160 → **158**; the "141 px" hedge replaced by the units finding; constraint compliance recorded as measured; on-the-fly training draws replaced by the **mask bank**; the large bucket's placement cramping recorded |
| **3.5** | Two committed mask artifacts, with the bank's schema (`tile_id, epoch, family, bucket, mask_rle, seed`), row count, size and generation time |
| **4.1** | The no-feature-cache bullet gains its mask half; new bullet pricing the draw and the bank |
| **4.5** | Step 1 covers the bank; step 3 ("Precompute frozen features") deleted as stale since #120 |
| **5** | Synthetic counts against the floor; **real-medium+ reporting rule pre-registered** |
| **7** | Bank line added to the cost table; "why a mask bank instead" paragraph; the #120 note that §9.4 is "unaffected" corrected |
| **8** | Spot-instance bullet's dangling "features are memmapped" clause removed |
| **9.4** | Rewritten: **extractor contract, not a shared cache**; Task 2's three stale references logged and deliberately not amended |

## 7. What this ticket did not settle

- **A9 remains unpriced** (#120's open item). Nothing here touches it.
- **Task 2's own caching decision** is left to Task 2's map, deliberately.
- **The real-void yield** is a projection from two cells until #127 lands and #135 measures
  it. Only the reporting threshold is fixed here.
