# Task 1 results: the DEM void-filling benchmark

**Results memo, September 2026.** Answers C1, C2 and C3 in numbers and C4 in
ranks, against the pre-registration committed in
[`task1-void-fill-plan.md`](task1-void-fill-plan.md) and
[ADR-0013](../adr/0013-task1-pre-registration-register.md). Every number here is
printed by `python -m benchmark.launch report` from
`data/task1/results.parquet`; none is transcribed by hand, and §6's fairness
checklist ships below in the plan's own words.

## There is nothing external to stand this table beside

**This table is an internally controlled comparison, and there is nothing
external to stand it beside.** No public DEM void-filling benchmark exists that
independent methods report against: published numbers in this task are each tied
to a private tile set, a private mask set and a private scoring script. The
field's most-cited evaluation, **Reuter et al. 2007**, publishes mean *rank* of
RMSE and never metres; the most recent work, **Dfilled (2025)**, retrained its
competitors rather than citing them. Every arm here sees the same tiles, masks,
seeds and head, and every number was produced by one scoring function in this
repository.

This paragraph opens the memo rather than closing it, because a reader who meets
the table first has already read the absent competitor column as an omission by
the time the explanation arrives. The absence is a finding, not a gap — and **the
reason is not access**. Both candidates the field offers were opened and read:

- **Diff-DEM.** Its committed `tif_metric.py` de-normalises to metres using the
  ground-truth tile's own extrema, **void interior included** — the exact leak
  §2.1 forbids for DEMAE. Their metres and ours are different quantities and
  could never share a column
  ([#124](https://github.com/JStoschek/DEMAE/issues/124)).
- **GeomorPM.** Its CC BY 4.0 Figshare deposit releases *pretrained* weights
  only; there is **no void-filling checkpoint**, its VF fine-tune config is
  absent from the deposit, and its paper is paywalled. A cross-eval would mean
  reproducing an unpublished recipe and then being unable to check the
  reproduction ([#128](https://github.com/JStoschek/DEMAE/issues/128)).

So the honest statement is the strong one — **incomparable, and we looked** —
rather than "incomparable except at the transfer blocks".

**One premise that selected this task is struck here.**
[`downstream-task-selection.md`](downstream-task-selection.md) §1 criterion 4
ranked void filling partly because open competitor code would let the result
"land next to something". That reason did not survive #124 and #128, and this is
the memo that carries it, so it is struck rather than left standing. **The task
remains the right first choice on the reasons that did survive**: the MAE decoder
is natively an inpainter, it is the cheapest of the candidate tasks, and the
money plot — before/after hillshades — is strong. Nothing in this plan's
existence rested on the struck reason.

## What ran, and what did not

The table prints eleven rows over 36,411 committed eval instances each, at 20k
head-training tiles unless the C3 sweep says otherwise. Three seeds on the five
arms whose heads train at 20k; one seed on the arms that train nothing.

| row | what it is | rows | seeds |
|---|---|---|---|
| A1 | classical interpolation, the operational status quo | 36,411 | 1 |
| A2-R5 / A2-R4 | DEMAE zero-shot, no trained head | **0** | — |
| A3-R5 / A3-R4 | DEMAE native fine-tune | 36,411 each | 1 |
| A4-R5 / A4-R4 | DEMAE-frozen + the shared head | 109,233 / 109,233 | 3 |
| A5 | TerraMind-tokenizer-frozen + the same head | 254,877 | 3 |
| A6 | ImageNet-MAE-frozen + the same head | 254,877 | 3 |
| A7 | UNet from scratch | 254,877 | 3 |
| A9 | TerraMind tokenizer pass-through | 36,411 | 1 |

There are two deviations from the pre-registered recipe, and both are recorded in
the run records rather than only here.

**A2 has no numbers, and that is a refusal rather than a failure.** §14's release
export is encoder-only, and both R5 and R4 were pretrained with `norm_pix_loss`
on, so their decoder predicts each patch standardized by *that patch's own* mean
and variance — the two numbers a void makes unknown. Recovering them from the
truth would leak the voided terrain's own level and spread into the fill, and
A2's row would beat every arm for a reason no printed column names. §4.3
registers no estimator, so `benchmark.zeroshot` refuses before scoring an
instance rather than picking one, because any choice sets a pre-registered row's
value ([#173](https://github.com/JStoschek/DEMAE/issues/173)). The arm is built
and tested; what is missing is a registered method and a decoder-bearing
checkpoint.

**A3's inference composite ran one encoding of the four §4.2 names.** The clause
asks for 4 shifted encodings at 0 and 112 px offsets, and a 224 px window of a
224 px tile displaced by 112 px is three-quarters fabricated canvas — not a view
of the tile. Measured on the trained R5 arm over 200 large-bucket instances: 45.0
m with the shifts against 21.0 m without, with A1 at 28.4 m. The 16 px cross-fade
— §4.2's other inference clause, and the one that actually moves the patch-grid
statistic — is unaffected and still runs on four grids
([#177](https://github.com/JStoschek/DEMAE/issues/177)). Every A3 row in the
parquet was scored without the three dropped encodings, and each run record says
so under `training.composite`.

### We diverge from terratorch's default inference path

**A5's positional embedding diverges from terratorch's default inference path on
purpose: ADR-0011 D2 takes the centre-anchored 14x14 crop of TerraMind's native
16x16 grid, token spacing exactly 1.0, where terratorch resizes it bicubically
and samples at spacing 1.1429 — ~14% too wide, accumulating past a full patch by
the far edge.** The embedding is fixed sin-cos, so the crop *is* the closed form
at the smaller grid and nothing is approximated. terratorch's own path is scored
too, as A5-BICUBIC, and only as R11's pre-registered ablation.

A second divergence is A9-NOTHRESH's, described under A9 below: it turns the
tokenizer's DDIM dynamic thresholding off and nothing else.

## The printed table

R5's registered shape: 11 arms × 4 ruggedness bands × 6 columns for the blob
family, plus the real family's two sub-rows in R4 — **396 numbers**, size buckets
pooled. Metres for `rmse`, `nmad`, `bias_abs`; degrees for `slope_rmse`,
`aspect_error` and `seam_gap`. `!` marks R7's seam break, a gap above +0.15°.

| arm | section | band | rmse | nmad | bias_abs | slope_rmse | aspect_error | seam_gap |
|---|---|---|---|---|---|---|---|---|
| A1 | blob | R1 | 2.061 | 1.549 | 0.705 | 1.180 | 66.094 | -0.023 |
| A1 | blob | R2 | 5.636 | 4.122 | 1.931 | 2.582 | 62.976 | -0.055 |
| A1 | blob | R3 | 13.550 | 9.700 | 4.805 | 5.032 | 61.085 | -0.118 |
| A1 | blob | R4 | 38.496 | 29.274 | 14.552 | 10.397 | 57.929 | -0.232 |
| A1 | real-small | R4 | 12.050 | 9.446 | 4.238 | 7.319 | 35.225 | -0.434 |
| A1 | real-medium+ | R4 | 24.718 | 16.878 | 6.336 | 9.519 | 48.121 | -0.385 |
| A2-R5 | all six cells | — | *not scored — see #173* | | | | | |
| A2-R4 | all six cells | — | *not scored — see #173* | | | | | |
| A3-R5 | blob | R1 | 1.508 | 1.147 | 0.485 | 1.130 | 60.758 | +0.000 |
| A3-R5 | blob | R2 | 4.656 | 3.574 | 1.443 | 2.511 | 59.852 | +0.041 |
| A3-R5 | blob | R3 | 11.341 | 8.756 | 3.511 | 4.868 | 58.479 | +0.098 |
| A3-R5 | blob | R4 | 30.388 | 25.089 | 9.613 | 9.705 | 53.756 | +0.187 ! |
| A3-R5 | real-small | R4 | 16.656 | 14.423 | 6.713 | 8.586 | 44.434 | +0.631 ! |
| A3-R5 | real-medium+ | R4 | 28.573 | 23.348 | 8.367 | 10.156 | 53.230 | +0.577 ! |
| A3-R4 | blob | R1 | 1.784 | 1.240 | 0.803 | 1.171 | 61.578 | +0.055 |
| A3-R4 | blob | R2 | 4.989 | 3.676 | 1.857 | 2.572 | 60.191 | +0.127 |
| A3-R4 | blob | R3 | 11.735 | 8.843 | 4.060 | 4.943 | 58.628 | +0.221 ! |
| A3-R4 | blob | R4 | 31.385 | 25.060 | 11.256 | 9.721 | 53.803 | +0.434 ! |
| A3-R4 | real-small | R4 | 19.831 | 14.509 | 11.136 | 9.284 | 45.726 | +1.620 ! |
| A3-R4 | real-medium+ | R4 | 29.157 | 23.624 | 9.361 | 10.076 | 53.164 | +0.772 ! |
| A4-R5 | blob | R1 | 1.533 | 1.135 | 0.427 | 1.141 | 65.808 | -0.022 |
| A4-R5 | blob | R2 | 4.532 | 3.262 | 1.286 | 2.491 | 61.490 | -0.055 |
| A4-R5 | blob | R3 | 10.890 | 7.731 | 3.075 | 4.825 | 58.668 | -0.119 |
| A4-R5 | blob | R4 | 29.291 | 22.094 | 8.350 | 9.813 | 54.223 | -0.248 |
| A4-R5 | real-small | R4 | 11.423 | 9.176 | 3.629 | 7.239 | 34.711 | -0.456 |
| A4-R5 | real-medium+ | R4 | 20.858 | 14.869 | 4.504 | 9.137 | 46.088 | -0.381 |
| A4-R4 | blob | R1 | 1.556 | 1.152 | 0.439 | 1.142 | 65.887 | -0.022 |
| A4-R4 | blob | R2 | 4.516 | 3.258 | 1.249 | 2.493 | 61.471 | -0.055 |
| A4-R4 | blob | R3 | 10.887 | 7.719 | 3.082 | 4.829 | 58.616 | -0.119 |
| A4-R4 | blob | R4 | 29.159 | 22.058 | 8.213 | 9.824 | 54.178 | -0.246 |
| A4-R4 | real-small | R4 | 11.489 | 9.218 | 3.668 | 7.253 | 34.792 | -0.445 |
| A4-R4 | real-medium+ | R4 | 20.817 | 14.864 | 4.397 | 9.155 | 46.028 | -0.382 |
| A5 | blob | R1 | 1.772 | 1.322 | 0.556 | 1.160 | 67.066 | -0.021 |
| A5 | blob | R2 | 4.984 | 3.657 | 1.535 | 2.533 | 63.104 | -0.055 |
| A5 | blob | R3 | 11.990 | 8.616 | 3.734 | 4.917 | 60.610 | -0.119 |
| A5 | blob | R4 | 33.550 | 25.529 | 11.085 | 10.083 | 56.841 | -0.245 |
| A5 | real-small | R4 | 11.756 | 9.344 | 3.883 | 7.298 | 35.290 | -0.438 |
| A5 | real-medium+ | R4 | 23.516 | 16.327 | 5.670 | 9.364 | 48.163 | -0.376 |
| A6 | blob | R1 | 1.638 | 1.212 | 0.479 | 1.148 | 66.224 | -0.021 |
| A6 | blob | R2 | 4.728 | 3.417 | 1.402 | 2.507 | 62.037 | -0.054 |
| A6 | blob | R3 | 11.545 | 8.229 | 3.493 | 4.873 | 59.514 | -0.116 |
| A6 | blob | R4 | 31.551 | 23.988 | 9.745 | 9.989 | 55.430 | -0.244 |
| A6 | real-small | R4 | 11.749 | 9.388 | 3.851 | 7.284 | 35.074 | -0.438 |
| A6 | real-medium+ | R4 | 22.175 | 15.626 | 4.961 | 9.281 | 46.978 | -0.378 |
| A7 | blob | R1 | 1.524 | 1.110 | 0.472 | 1.078 | 61.747 | -0.020 |
| A7 | blob | R2 | 4.484 | 3.161 | 1.364 | 2.336 | 56.994 | -0.051 |
| A7 | blob | R3 | 10.866 | 7.456 | 3.354 | 4.482 | 53.987 | -0.114 |
| A7 | blob | R4 | 30.215 | 22.005 | 9.787 | 9.015 | 49.664 | -0.249 |
| A7 | real-small | R4 | 10.021 | 7.892 | 3.098 | 6.452 | 31.715 | -0.438 |
| A7 | real-medium+ | R4 | 20.265 | 13.274 | 4.639 | 8.138 | 41.353 | -0.372 |
| A9 | blob | R1 | 277.582 | 2.579 | 277.287 | 10.373 | 73.252 | +9.190 ! |
| A9 | blob | R2 | 201.665 | 6.126 | 200.127 | 9.965 | 70.090 | +7.316 ! |
| A9 | blob | R3 | 200.831 | 14.453 | 195.978 | 10.664 | 69.530 | +5.305 ! |
| A9 | blob | R4 | 503.499 | 52.942 | 492.044 | 18.418 | 73.872 | +5.454 ! |
| A9 | real-small | R4 | 494.641 | 28.620 | 490.161 | 26.930 | 70.111 | +15.053 ! |
| A9 | real-medium+ | R4 | 480.064 | 50.568 | 470.393 | 21.152 | 70.041 | +8.434 ! |

Maximum seed spread anywhere in the table is **1.019** (`aspect_error`, A7
real-small R4, three seeds), reported once rather than per cell because only five
of the eleven rows carry three seeds. On `rmse` the per-cell spread is at most
0.423 (A7 real-medium+) and typically under 0.1.

## C1: pretraining works, in all twelve marked readings

C1 asks whether **DEMAE-frozen** features beat **ImageNet-MAE-frozen** and
**TerraMind-tokenizer-frozen** features under an identical trained head, on the
same tiles, masks and seeds. **R5 is the headline unconditionally**, so C1's
number is **A4-R5**'s.

**A4-R5 is the lowest of the three in every one of the six printed cells, on both
marked metrics.** On `rmse`, blob R1 through R4: 1.533 / 4.532 / 10.890 / 29.291
against A6's 1.638 / 4.728 / 11.545 / 31.551 and A5's 1.772 / 4.984 / 11.990 /
33.550; on the real family in R4, 11.423 and 20.858 against A6's 11.749 and
22.175 and A5's 11.756 and 23.516.

R6's two C1 pairs carry the bootstrap mark, and **all twelve interval readings
per pair exclude zero** — six cells × `rmse` and `slope_rmse`, paired over test
tiles, 10,000 resamples:

- **A4-R5 − A5**: from [-0.258, -0.221] at blob R1 to [-4.559, -3.968] at blob
  R4; [-3.200, -2.138] at real-medium+.
- **A4-R5 − A6**: from [-0.118, -0.093] at blob R1 to [-2.470, -2.056] at blob
  R4; [-1.622, -1.021] at real-medium+.

The margin is larger against the tokenizer than against ImageNet-MAE in every
cell, which is the ordering the plan expected. A5's encoder was trained for
reconstruction rather than representation, so this holds under identical probes
and is not a claim that TerraMind is worse.

**The capacity control holds.** A4-S is §2.3's ViT-S appendix cell: one seed,
reported without a ±. It scores 1.553 / 4.563 / 11.041 / 29.784 on blob R1–R4,
still below both A6 and A5 in every cell despite a smaller encoder. C1's margin therefore does not rest
on DEMAE being the largest encoder in the comparison.

### The R4 secondary, reported prominently and not as C1's number

**A4-R4, the continual-init secondary, edges A4-R5 on `rmse` in four of the six
printed cells**: blob R2 (4.516 against 4.532), blob R3 (10.887 against 10.890),
blob R4 (29.159 against 29.291) and real-medium+ (20.817 against 20.858). A4-R5
is ahead at blob R1 and real-small.

R3 fixed before any number existed that an R4 win is reported prominently and
**never becomes C1's number**, and it stays that way. On its own terms it would
also be the wrong number to reach for. **Every one of those six gaps is smaller
than the seed spread of the cell it sits in.** The largest, blob R4's 0.132 m, sits inside
A4-R4's own three-seed spread of 0.235 m; blob R3's 0.003 m against spreads of
0.041 and 0.043 is indistinguishable from zero. R4 neither beats nor loses to R5
here. The two inits are tied at 20k, and the pre-registration is what stopped
that tie being read in whichever direction the table happened to fall.

## C2: the model is useful only in part, and the seam clause fails

C2 asks whether fine-tuned DEMAE beats classical interpolation in metres, slope
and **seam continuity** on rugged terrain at every void size, and stays at least
even on flat terrain. It **partially fails**, in three separate ways.
Reporting that rather than averaging it away is what §0 committed to.

**It holds on metres and slope over synthetic voids, in every band.** A3-R5
against A1 on blob `rmse`: 1.508 / 4.656 / 11.341 / 30.388 against 2.061 / 5.636
/ 13.550 / 38.496. On `slope_rmse`: 1.130 / 2.511 / 4.868 / 9.705 against 1.180 /
2.582 / 5.032 / 10.397. The C2 bootstrap pair's intervals exclude zero in all
eight synthetic readings, from [-0.598, -0.510] at blob R1 to [-8.698, -7.526] at
blob R4. **The flat-terrain carve-out is beaten rather than merely met**: R1 is
the flattest quartile and A3-R5 is 27% below A1 there.

**It fails on the small bucket in two mid bands.** The pooled table hides void
size, so this is read off the figure's grain: on blob small voids, A3-R5 loses to
A1 at R2 (3.097 against 2.996) and at R3 (7.108 against 6.985), and wins at R1
and R4. It wins the medium and large buckets in all four bands, and wins all
three buckets in R4. So "at every void size" holds in the roughest band and not
across R2 and R3.

**It fails on real SRTM voids outright.** A3-R5 scores 16.656 and 28.573 against
A1's 12.050 and 24.718, and the intervals are decisive in the wrong direction:
[+4.404, +4.813] at real-small and [+2.940, +4.768] at real-medium+ on `rmse`.
§3.2's blobs and rectangles are the only geometry A3 ever adapted to, so this is
a fact about the train mask bank rather than a defect in the arm — but it is A1's
win and is recorded as one.

**It fails on seam continuity, which was the operational half of the claim.** The
seam gap is `slope_rmse` over the full void mask minus the 1-px-eroded mask, so a
positive value means the boundary ring scores *worse* than the interior — the
fill does not join up with the terrain around it. **A3-R5's gap is positive in
every printed cell and A1's is negative in every printed cell** — A3-R5 runs from
+0.0002 at blob R1 (which prints as +0.000) up to +0.631 at real-small, against
A1's -0.023 to -0.434 — and A3-R5 trips
R7's +0.15° seam break at blob R4 (+0.187), real-small (+0.631) and real-medium+
(+0.577), where A1 never does. A3-R4 is worse again (+1.620 at real-small).
Seam continuity replaced the original drainage clause because an operator's
complaint about a bad fill is that it does not join up with the terrain around
it. On the metric chosen to carry that argument, the fine-tuned arm is the one
that breaks.

**One further result the claim did not anticipate.** The frozen-encoder arm beats
the fine-tuned one in five of the six printed cells — A4-R5's 29.291 against
A3-R5's 30.388 at blob R4, and 20.858 against 28.573 at real-medium+, with 1.18M
trainable parameters against 98.39M — and A4-R5's seam gaps are negative where
A3-R5's are positive. A3-R5 is ahead at blob R1 alone (1.508 against 1.533).
Part of that gap is #177's dropped encodings, which are already a deviation in
A3's favour, since the composite that ran is the cheaper and better one. None of
it is evidence about C1, which is scored entirely on the frozen arms.

## C3: label efficiency holds at the low-label end, and closes by 20k

C3 asks whether DEMAE-frozen reaches a given RMSE with far fewer head-training
tiles than the UNet-from-scratch baseline. The curve is **R5's**; R4 runs at 20k
only and does not enter the sweep. Mean `rmse` over the blob family, buckets
pooled:

| tiles | A4-R5 | A6 | A5 | A7 (UNet) |
|---|---|---|---|---|
| 1,000 | **13.127** | 13.995 | 14.778 | 14.397 |
| 5,000 | **12.388** | 13.208 | 14.195 | 14.156 |
| 20,000 | **11.494** | 12.292 | 12.994 | 11.701 |

**The claim holds at the low-label end.** A4-R5 trained on 1,000 tiles (13.127)
is ahead of the UNet trained on 5,000 (14.156) — a five-fold label advantage
measured rather than interpolated — and the UNet does not reach A4-R5's 1k score
anywhere below 20k. At 1k the gap is 1.27 m.

**It closes to near-parity by 20k**, and that is reported with the prominence of
the win, as R10 requires. At 20k the gap is 0.21 m, and the UNet is ahead on
several readings: A4-R5 − A7 on `slope_rmse` excludes zero on the wrong side in
all six cells (from [+0.054, +0.072] at blob R1 to [+0.887, +1.117] at
real-medium+), and on `rmse` the UNet is ahead at real-small [+1.314, +1.491] and
real-medium+ [+0.116, +1.056], tied at blob R1 and R3, with A4-R5 ahead only at
blob R4 [-1.190, -0.659]. §6 item 7's worked example was "A7 ties A4 at 20k"; at
20k, on slope, A7 beats it.

C1's ordering is unchanged at every rung: A4-R5 < A6 < A5 at 1k, 5k and 20k, in
every band. The label-efficiency result and the representation result are
therefore not in tension — the pretrained representation is worth about a
five-fold label saving against from-scratch, and worth a consistent margin
against the other two pretrained encoders at every budget.

## C4: three pairs swap rank, and none of them is a C1 pair

C4 is an **ordering** claim, deliberately not a metric-delta claim: the real-void
family is drawn at its natural size distribution and the synthetic families are
bucket-balanced, so a difference in metres between the two would confound void
*shape* with void *size*. It is evaluable in **R4** and nowhere else, because
real SRTM voids land 99.94% in the roughest quartile. The block is 66 rank cells
— 3 R4 instance sets × 11 arms × 2 metrics.

| set / metric | A1 | A2-R5 | A2-R4 | A3-R5 | A3-R4 | A4-R5 | A4-R4 | A5 | A6 | A7 | A9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| blob/small / rmse | #7 | — | — | #6 | #8 | #2 | #3 | #5 | #4 | #1 | #9 |
| blob/small / slope_rmse | #7 | — | — | #6 | #8 | #2 | #3 | #5 | #4 | #1 | #9 |
| real-small / rmse | #6 | — | — | #7 | #8 | #2 | #3 | #5 | #4 | #1 | #9 |
| real-small / slope_rmse | #6 | — | — | #7 | #8 | #2 | #3 | #5 | #4 | #1 | #9 |
| real-medium+ / rmse | #6 | — | — | #7 | #8 | #3 | #2 | #5 | #4 | #1 | #9 |
| real-medium+ / slope_rmse | #6 | — | — | #8 | #7 | #2 | #3 | #5 | #4 | #1 | #9 |

`—` is an arm with no scored instances in that set.

**C4's failure condition is met.** That condition is an arm pair swapping rank
between the synthetic and real-void rows within the same band, and three pairs
do, over six swap instances:

- **A1 and A3-R5**, on both metrics and against both real sets, which is four of
  the six instances. A3-R5 is #6 and A1 #7 on synthetic blob/small, and they
  exchange places on real-small and on real-medium+. This is C2's real-void loss
  showing up as an ordering.
- **A4-R5 and A4-R4**, on `rmse`, between blob/small (#2/#3) and real-medium+
  (#3/#2). This is the same sub-seed-spread tie C1's secondary section describes,
  crossing a rank boundary because a rank has no tolerance.
- **A3-R5 and A3-R4**, on `slope_rmse`, between blob/small (#6/#8) and
  real-medium+ (#8/#7).

**What the failure does not reach is C1.** §0's reason for caring about a rank
swap is that it would put C1 in question, since C1's comparison is made entirely
on synthetic masks. C1's comparison is A4-R5 against A6 and A5, and that triple
holds the ordering **A4-R5 < A6 < A5 — #2, #4, #5 — in all three instance sets on
both metrics**, real geometry included. A7 is #1 and A9 #9 in all six rows.
Every swapping pair is either the two DEMAE inits, which R3 already refuses to
let decide anything, or the fine-tuned arm against classical fill, which is C2's
territory and is reported as C2's loss above. So both things are true: **C4 fails
by its own criterion, and the pairs that broke it are not the pairs C1 rests on.**
Both are stated because either one alone is the convenient half.

## A9: what the tokenizer bottleneck measures

The printed A9 row — 277.582 to 503.499 m RMSE — **measures terratorch's
published sampler more than it measures the tokenizer**, and must not be read as
the bottleneck figure. The DEM DiVAE is built with `thresholding=True,
sample_max_value=1.0`; dynamic thresholding rescales every decode into [-1, 1],
which under TerraMind's own DEM affine is **[-280.6 m, +1621.9 m]**. 2500 m,
3000 m and 4000 m all decode to 1621.9 m, and 6000 m wraps to -277 m. **37.7%**
of the committed instances sit on tiles above that ceiling, and for them the
overshoot predicts scored `bias_abs` at r = 0.985.

**A9-NOTHRESH** turns that one knob off and nothing else — same weights, same
md5, same diet, same 50 timesteps, same instances, at 0.17% more runtime. It is
registered with `printed=False` so R5's eleven pre-registered rows are untouched.

| | mean RMSE | median | below 1622 m | above 1622 m |
|---|---|---|---|---|
| A1 interpolation | 14.678 | 8.235 | 6.76 | 11.53 |
| A9 (published sampler) | 345.685 | 70.662 | 19.11 | **753.57** |
| A9-NOTHRESH | **32.635** | 19.175 | 18.73 | **19.47** |

A9-NOTHRESH scores the same either side of the ceiling, 18.73 below and 19.47
above, on a line the published sampler cannot cross. **The defensible reading of
the tokenizer is ~2.2x interpolation over the pass, not 24x**, and the per-band
shape says more. On blob `rmse`, A9-NOTHRESH is 36.496 against A1's 2.061 in the
flattest band (17.7x) and 45.360 against 38.496 in the roughest (1.18x). The tokenizer's reconstruction floor is *absolute*, so
it is catastrophic where the terrain is nearly flat and close to competitive
where the terrain is large-amplitude. `clip_sample=True` is a red herring:
diffusers applies dynamic thresholding *instead of* that clamp, so turning
`clip_sample` off alone changes the decode by nothing, to the digit.

Either way the conclusion is phrased under identical probes and is not "TerraMind
is worse": A9 is a pass-through of a tokenizer trained for reconstruction, and it
trains nothing here.

## R11's ablation row

A5 at 20k under ADR-0011's re-derived embedding runs against A5-BICUBIC under
terratorch's default, both at three seeds. Blob `rmse` R1–R4: 1.772 / 4.984 /
11.990 / 33.550 against 1.776 / 4.997 / 12.011 / 33.629. Across all six printed
cells the difference runs from **-0.061 m** at real-medium+, where the default
path is the *better* of the two, to **+0.103 m** at real-small. Every one of the
six is inside the seed spread of at least one of the two arms.

**The row is a null, and publishing it is the point.** #134's materiality check
returned 0.9708 against R11's bar of 0.99, so the row could not be dropped. Had
the bar been set after the numbers existed, a sub-0.1 m difference that changes
sign between cells is exactly the size that invites dropping. The ablation says
the crop-versus-resize choice did not move A5's score materially. That is a
finding about this benchmark and not a retraction of ADR-0011 D2, whose argument
is geometric rather than empirical.

## Registered rules: every disposition

All twelve, plus R9b's refusal. R8 and R11 are the two that already fired against
their own projections and held, and they are why this table is worth reading: a
pre-registration nobody can audit is a pre-registration nobody should believe.

| rule | what it fixed | disposition |
|---|---|---|
| **R1** | six printed columns — five graded headline plus the derived seam gap — and four appendix, nothing else | **Held.** The table prints exactly `rmse`, `nmad`, `bias_abs`, `slope_rmse`, `aspect_error`, `seam_gap`. `benchmark.metrics.score` gates on the bar's roster, so an unregistered metric raises rather than being computed. No column was added after the numbers existed. |
| **R2** | the pretraining-corpus caveat is symmetric | **Held, and it was owed in the direction that cost us.** TerraMind's ~3x ground-area advantage (63.4M km² against 21.4M km²) is recorded in §6 item 4 below, before any result. The comparison went DEMAE's way, so the caveat is owed here as a discount on **our own** win, and it is priced as one: TerraMind had ~3x the ground area and still lost every C1 cell, so A4-R5's margin over A5 is if anything understated by the corpus difference rather than explained by it. The caveat is not invoked anywhere to explain a DEMAE loss. |
| **R3** | R5 is the headline unconditionally; an R4 win is reported prominently but never becomes C1's number | **Fired, and held.** A4-R4 edges A4-R5 on `rmse` in four of six printed cells. It is reported prominently above under its own heading, C1's number is A4-R5's, and the memo additionally states that all six gaps are inside seed spread. |
| **R4** | TerraMind's **20** DEM-tokenizer epochs carry their per-modality citation wherever stated, not the paper's 100 | **Held.** 20 appears in §6 item 4 below with its terratorch / model-card attribution; 100 appears nowhere as this tokenizer's epoch count. |
| **R5** | the printed table's shape: buckets pool, blobs headline, 66 (row, band) cells × 6 columns = 396 printed numbers, the 66-cell C4 rank block | **Held.** 396 numbers and 66 rank cells, asserted by `report.assert_r5_shape` and `report.assert_rank_block_shape` rather than counted by hand. A2's twelve cells print as *not scored*, which keeps the shape whole rather than shrinking the table to the arms that ran. |
| **R6** | exactly four arm pairs carry a bootstrap CI mark, named in advance, of 55 computed | **Held.** Four marked (A4-R5/A5 and A4-R5/A6 for C1, A3-R5/A1 for C2, A4-R5/A7 for C3), 36 of the 55 computable on this box, every other pair computed and left unmarked. R6 registers `A3`, which §2.3 splits by init; resolved to A3-R5 under R3 rather than by choosing after the numbers. |
| **R7** | a seam gap above **+0.15°** is a seam break | **Fired.** Breaks in **13 of the 54 scored cells**: three A3-R5 (blob R4, real-small, real-medium+), four A3-R4 (blob R3 and R4, real-small, real-medium+) and all six A9. No A1, A4, A5, A6 or A7 cell breaks. The threshold was fixed long before either arm that trips it was trained, and the arms that trip it are the fine-tuned ones and the tokenizer pass-through. |
| **R8** | real-medium+ is unstratified unless the realised histogram clears **100 per band** | **Fired against its own projection, and held.** The floor was fixed at ~96 projected instances and the realised count was **438** — 4.6x the projection — but they land 99.94% in R4, so no four-band histogram clears 100 per band and the row prints unstratified in R4 alone. A threshold nobody could move to the number produced the right report. |
| **R9** | the GSD caveat is symmetric, and stated as a **bandwidth** shift rather than a resolution handicap | **Held.** §6 item 3 below states it as a bandwidth shift — TerraMind is fed *sharper* inputs than it ever saw, across 3x the ground span per patch — and it is measured on the tokenizer's own FSQ quantizer. The comparison went DEMAE's way, so the symmetry obligation binds in the direction that costs us, and the bandwidth shift is not offered as the explanation of A5's result. |
| **R9b** | *a refusal:* the FSQ diagnostic carries **no threshold** | **Held as a refusal.** No threshold was added; the diagnostic is reported as a quantity and nothing in this memo reads a pass or a fail off it. The statistic has no natural scale, and a bar would imply it settles what the shift costs. |
| **R10** | C3's low-label sweep is the pre-registered fallback claim; nulls reported with the prominence of wins | **Fired, in the direction it was written for.** C3 holds at 1k and closes to 0.21 m by 20k, where A7 beats A4-R5 on slope in all six cells. The closing and the UNet's slope win are stated in C3's own section at the same prominence as the 1k result, not relegated. |
| **R11** | bicubic-vs-re-derived ablation row, A5 at 20k, dropped iff the materiality check returns **≥ 0.99** | **Fired against its own projection, and held.** #134 returned **0.9708** against the bar of **0.99**, so the row stayed and is published above as a null: **-0.061 m to +0.103 m** across the six printed cells, each inside a seed spread, and changing sign between them. A bar set after the numbers would have dropped it. |
| **R12** | A2 carries no C-claim and is insulated from C1 both ways; the A2→A3 difference is joint; negative result if A2 does not beat A1 in a majority of the four bands | **Pending — the comparison is not computable.** A2 has no rows (#173), so 0 of the 4 registered bands hold rows from both arms and the contingency has no verdict. **Not computed yet is not "does not beat A1"**, and the negative result is therefore not claimed. The two clauses that do not need numbers are in force: A2 carries no C-claim, and nothing here attributes the A2→A3 difference to task adaptation — it is a joint quantity, the cost of the distribution shift and the value of task adaptation summed, and the arm that would separate them does not exist and is not proposed. |

## 6. Fairness checklist (goes in the writeup verbatim)

1. Identical tiles, masks, seeds, head, schedule for every frozen-encoder arm;
   head hyperparams tuned once on the neutral encoder (A6), then frozen.
2. Each encoder fed its native normalization; scoring always in metres;
   per-tile stats from non-void pixels only; identity-mask unit test committed.
   Derivative metrics score the **full** void mask, including the boundary ring
   whose 3x3 kernel partly reads true pixels. That ring is included knowingly:
   it is where a discontinuous fill is caught, and eroding it halves slope's
   ability to separate arms. The free ride it grants a *smooth* fill was
   measured at −0.005 to −0.107° and is published beside the eroded-mask control
   rather than argued away ([`p123-metric-discrimination.md`](p123-metric-discrimination.md)).
3. There is **one regime, 224 px at native 30 m**, it is DEMAE's home turf, and
   the writeup says so rather than leaving it to be inferred. TerraMind runs
   out-of-distribution here, and the shift is **not** that it is starved of
   resolution: its DEM tokenizer trained on the same Copernicus product
   bilinearly upsampled 3x to 10 m, so it is fed **sharper** inputs than it ever
   saw, across 3x the ground span per patch. That caveat is pre-registered and
   symmetric — not available to explain a TerraMind loss unless also priced as a
   discount on a TerraMind win — and it is **measured on the tokenizer's own FSQ
   quantizer rather than asserted** (§2.2). No retrained 10 m head is run, and
   the writeup does not promise one: that regime would close an objection
   without carrying a claim, and its metres would not be comparable to any row
   in the table (§2.2).
4. Capacity and pretraining data are **reported rather than bracketed**, because
   both are near-matched and there is nothing left to control. Parameter counts
   in every table, **total and trainable**: DEMAE ViT-B **85.25M** against
   TerraMind's DEM tokenizer encoder **90.17M** — same depth, same 16 px patch,
   same 14x14 token grid — and 1.18M trainable in every frozen-encoder arm
   against the UNet baseline's 17M, so the from-scratch bar is not read as
   handicapped by a totals-only column. Pretraining differs in data, not
   compute: **189.2M tile presentations against 181.8M** at the same effective
   batch of 2048, over **21.4M km²** of GLO-30 against **63.4M km²** of the same
   product resampled to 10 m. That **~3x** ground-area advantage to TerraMind is
   recorded here, before any result, and is not available as a post-hoc
   explanation of a DEMAE loss that was not also offered as a discount on a
   DEMAE win. TerraMind's 20 DEM-tokenizer epochs are the per-modality figure
   from terratorch and IBM's model card, **not** the 100 the TerraMind paper
   states for its S-2/S-1 tokenizers ([`p138-terramind-corpus.md`](p138-terramind-corpus.md)).
5. The tokenizer encoder was trained for reconstruction, not representation;
   A9 quantifies the bottleneck; conclusions are phrased as "under identical
   probes" not "TerraMind is worse."
6. Geographic holdout only; the test geocells are the corpus's committed test
   split. **There is no transfer block and no external arm**: both candidates
   the field offers were investigated and neither yields a comparable number
   (§0). Every row in every table is an internal comparison, and the writeup
   says so rather than leaving it to be inferred.
7. Negative or null results (e.g., A7 ties A4 at 20k) are reported with the
   same prominence as wins; C3's low-label sweep is the pre-registered fallback
   claim.
8. Pre-registration: this memo is committed before the first head trains;
   success criteria C1-C4 are defined above the results, not after them. **So are
   twelve rules governing how the evidence may be used**, and this item is their
   index — each is stated in full beside the topic it governs, and all twelve are
   asserted together in
   [ADR-0013](../adr/0013-task1-pre-registration-register.md), which this list
   quotes rather than independently commits. A rule is on it when a *different*
   choice, made after seeing the numbers, could change which conclusion the table
   supports:

   - **R1** — ten candidates were graded; the metric set is **five headline
     columns plus the derived seam gap — six printed — and four appendix ones**,
     nothing else, graded by measured separation (§5).
   - **R2** — the pretraining-corpus caveat is symmetric (item 4 above).
   - **R3** — **R5 is the headline unconditionally**; an R4 win is reported
     prominently but never becomes C1's number (§0, §2.3).
   - **R4** — TerraMind's **20** DEM-tokenizer epochs carry their per-modality
     citation wherever stated, not the paper's 100 (item 4 above).
   - **R5** — which axes the headline table prints: the size buckets pool, the
     blob family is headline, and C4 prints as a rank block (§5, §9).
   - **R6** — exactly **four** arm pairs carry a bootstrap CI mark, named in
     advance, of the 55 computed (§5, §9).
   - **R7** — a seam gap above **+0.15°** counts as a seam break (§5).
   - **R8** — real-medium+ is unstratified unless the realised histogram clears
     **100 per band** (§3.3, §5).
   - **R9** — the GSD caveat is symmetric and is stated as a *bandwidth* shift,
     not a resolution handicap (item 3 above).
   - **R9b** — *a refusal:* the FSQ diagnostic carries **no threshold**, because
     the statistic has no natural scale and a threshold would imply it settles
     what the shift costs (§2.2).
   - **R10** — C3's low-label sweep is the fallback claim, fixed here (item 7
     above).
   - **R11** — the bicubic-vs-re-derived ablation row, A5 at 20k, dropped iff the
     materiality check returns **≥ 0.99**
     ([ADR-0011](../adr/0011-terramind-posemb-rederived-at-14x14.md) D4-5; it
     returned 0.9708, so the row stays).
   - **R12** — the A2 row carries **no C-claim** and is insulated from C1 in
     *both* directions; the A2→A3 difference is a **joint** quantity (task
     adaptation plus mask-distribution adaptation) and is never attributed to
     task adaptation alone; and A2 is reported as a **negative result** if it
     does not beat A1 in a majority of the four bands (§4.3).

   Each is a choice that would be worth nothing decided after the numbers existed
   — a table shape picked to flatter a result, a pair set chosen from whichever
   comparisons cleared zero, a threshold slid to put an arm on the right side of
   it. **Two have already fired against their own projections and held**: R8's
   floor was fixed at ~96 projected instances and met a measured 438
   ([#135](https://github.com/JStoschek/DEMAE/issues/135)), and R11 returned
   0.9708 against a bar of 0.99 ([#134](https://github.com/JStoschek/DEMAE/issues/134)).
   Neither outcome was available to argue with, which is the only property that
   made either worth registering. The numeric entries are additionally committed
   to `task1-prereg-bar.json`, which refuses to load if the phrase it was
   registered in no longer appears above.

## Reproducing this

```bash
uv run python -m benchmark.launch report
```

Reads `data/task1/results.parquet` and re-renders the printed table, the C4 rank
block, the bootstrap intervals, the seed spread, the parameter counts and the
three figures. It re-runs no arm. The committed inputs it checks itself against
are listed in its header with their md5s; `artifacts/task1/task1-prereg-bar.json`
refuses to load if any registered phrase has moved out of §6, in this file or in
the plan.

The three figures are load-bearing and are not to be cut for space.
`rmse-vs-void-size.png` is the only place void size appears and is therefore the
whole of C2's "at every void size". `label-efficiency.png` is the whole of C3.
`hillshade-panel.png` is the whole of the visual-quality evidence, since §5
scores no perceptual metric.

**The hillshade panel is empty of its evidence on this box, and that is a gap in
the deliverable rather than a caption.** It renders the truth and voided columns
for all three tiles, and its four arm columns (A1, A3-R5, A5, A7) each read *no
fill exhibit in panel/*: no arm pass saved a filled raster, so `data/task1/panel/`
does not exist. The figure the plan calls the whole of the visual-quality
evidence therefore currently carries none of it. Nothing in this memo's readings
rests on that figure, because §5's six columns are what every claim above is
argued from, but C2's argument in particular would be easier to check with it and
the panel should be regenerated before this memo is shown to anyone outside the
project.

**Open at the time of writing:**
[#173](https://github.com/JStoschek/DEMAE/issues/173) (A2's per-patch scale, and
a decoder-bearing checkpoint) and
[#177](https://github.com/JStoschek/DEMAE/issues/177) (whether §4.2 keeps the
shifted-encoding clause at all).
