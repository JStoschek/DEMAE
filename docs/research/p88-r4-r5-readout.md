# P88 — R4 vs R5 at matched epochs, and what the release may say about it

Pulled from W&B on 2026-08-07 for [#88](https://github.com/JStoschek/DEMAE/issues/88). Runs:
R4 = `xxx2eqpl` (continual, OpenMMLab ImageNet MAE init), R5 = `7wyou7bg` (scratch, the release
run). Both ViT-B/16, 400 epochs, 92,400 steps, the ADR-0010 frozen recipe (75% masking,
`norm_pix` ON), effective batch 2048, `lr` 1.20e-3.

This is the readout [#87](https://github.com/JStoschek/DEMAE/issues/87) deliberately did not
deliver — "all three probes at a **matched epoch** on a byte-identical `probe/set_md5`" — plus
three corrections to claims already recorded on the map, and one precondition the packaging
half of #88 cannot proceed without.

## 1. The epoch axis, which is where three recorded claims went wrong

Two counters share the word "epoch" and differ by one, and every mis-statement below is that
off-by-one plus a cadence that never lands on a round number.

- **W&B `_step`** is the **0-based index of the epoch that just ended**. `train.py:411` does
  `epoch = self.epoch; self.epoch += 1` and hands that to `on_epoch`, which
  `monitor.train_epochs` logs with `step=epoch`.
- **`RunState.epoch`** — what a checkpoint filename carries — is **epochs completed**:
  *"0 during the first one; 1 once epoch 0 has been walked whole"* (`checkpoint.py:212`).

So `completed = _step + 1`. The released `checkpoint_e0400_s000092400_milestone.pt` is
`_step` 399, the last row in W&B.

The probe fires on `due(epoch) = epoch % every == 0` (`probe.py:940`, `every = 20`) against the
**0-based** index — so at `_step` 0, 20, … 380, i.e. after **1, 21, 41, … 381 completed
epochs**. It never fires at `_step` 399.

### Correction 1 — there is no epoch-400 probe point, for either run

The map's R5 entry says the full readout is blocked because *"R5's epoch-400 probe point … [is]
only in W&B"*. It is not in W&B either. **No such point was ever taken.** The last probe of both
runs is at `_step` 380 = 381 completed epochs, 19 epochs before the released weights.

### Correction 2 — "R4's epoch-400 endpoint" is R4 at 381 completed epochs

The map's R5 entry compares R5@340 against *"R4's epoch-**400** endpoint … 0.400 / 2.040 /
0.248"*. Those three numbers are R4's probe at `_step` 380 — 0.3998 / 2.0398 / 0.24813 — which
is 381 completed epochs. The comparison drawn was R5 at 341 completed vs R4 at 381 completed,
i.e. it was conservative in R5's favour by **40** epochs, not the 60 claimed, and against a
point 19 epochs short of R4's endpoint rather than at it.

The conclusion it supported survives intact and gets stronger below. Only the labels were wrong.

### Correction 3 — R4's 0.299 is not "before a single gradient step"

The map's R4 entry offers *"**0.299 stratum accuracy at epoch 0**, before a single gradient step
— the warm start's own contribution, measured, and the natural input to #103's declined
frozen-init test"*. The `_step` 0 row carries `train/steps = 231`, `train/loss = 1.045`,
`lr = 5.974e-05`: the point is taken at the **end** of epoch index 0, after a full epoch —
231 optimizer steps, ~473k tile presentations at effective batch 2048.

`probe.py:914` calls the first point *"the epoch-0 baseline"* and every delta key is
`*_vs_epoch0`, which is what seeded the misreading: the name says baseline, the schedule says
one epoch in. The number is real and useful — it is "warm start + 1 epoch" against R5's
"scratch + 1 epoch" of 0.1798 — but it is **not** the frozen-init measurement #103 declined,
and it cannot stand in for one. That measurement still does not exist.

## 2. The matched-epoch readout

`probe/set_md5` is `ec69a3c80c8356eb562db65de8bdb7c0` for **both** runs at every point, 5,000
tiles, k=20, mean pooling, 20 strata, chance 0.0682 — so the two arms are scored on the same
tiles, and ADR-0010's across-arms rule (all three probes, matched epoch, no lead metric) applies
as written.

| completed epochs | R4 stratum | R5 stratum | R4 slope MAE | R5 slope MAE | R4 geo MAE | R5 geo MAE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.2990 | 0.1798 | 3.3933 | 4.6456 | 0.33575 | 0.42675 |
| 21 | 0.3450 | 0.3580 | 2.5427 | 2.4516 | 0.28267 | 0.27989 |
| 41 | 0.3562 | 0.3768 | 2.4023 | 2.1746 | 0.27697 | 0.26000 |
| 61 | 0.3684 | 0.3874 | 2.3454 | 2.1191 | 0.26650 | 0.25350 |
| 81 | 0.3706 | 0.3884 | 2.3451 | 2.0912 | 0.26584 | 0.25477 |
| 101 | 0.3700 | 0.3904 | 2.3758 | 2.0093 | 0.26348 | 0.25037 |
| 121 | 0.3750 | 0.3932 | 2.2888 | 2.0030 | 0.26165 | 0.24893 |
| 141 | 0.3734 | 0.3912 | 2.2432 | 2.0081 | 0.25628 | 0.24700 |
| 161 | 0.3730 | 0.3880 | 2.2514 | 1.9859 | 0.25879 | 0.24605 |
| 181 | 0.3782 | 0.3946 | 2.2501 | 1.9701 | 0.25530 | 0.24061 |
| 201 | 0.3808 | 0.3956 | 2.2144 | 1.9538 | 0.25789 | 0.23990 |
| 221 | 0.3810 | 0.4004 | 2.1746 | 1.9047 | 0.25692 | 0.23880 |
| 241 | 0.3900 | 0.4018 | 2.1511 | 1.9368 | 0.25434 | 0.23728 |
| 261 | 0.3928 | 0.4032 | 2.1471 | 1.9398 | 0.25555 | 0.23674 |
| 281 | 0.3974 | 0.4020 | 2.1113 | 1.8858 | 0.25313 | 0.23549 |
| 301 | 0.3946 | 0.4048 | 2.1246 | 1.8659 | 0.25108 | 0.23574 |
| 321 | 0.4002 | 0.4090 | 2.0981 | 1.8780 | 0.25051 | 0.23360 |
| 341 | 0.3932 | 0.4102 | 2.0650 | 1.8578 | 0.24997 | 0.23541 |
| 361 | 0.3998 | 0.4080 | 2.0458 | 1.8408 | 0.24911 | 0.23342 |
| 381 | 0.3998 | 0.4110 | 2.0398 | 1.8382 | 0.24813 | 0.23234 |

**R5 wins 19 of 19 matched points, on all three probes, from 21 completed epochs onward.** The
single point R4 wins is the first, where it is one epoch of elevation MAE past an ImageNet
initialization and R5 is one epoch past random.

At the last matched point (381 completed):

- `stratum_id` 0.4110 vs 0.3998 — **+0.0112**, ~1.9× R1's ±0.006 noise width
- `slope_mean` MAE 1.8382 vs 2.0398 — **9.9% better**
- `geomorphon_entropy` MAE 0.23234 vs 0.24813 — **6.4% better**

**The margin is not a single-point result and should not be reported as one.** These are 19
consecutive points from one run per arm and are heavily autocorrelated, so this is *not* 19
independent trials and no sign-test p-value should be quoted. What it is: a margin **constant in
sign across every matched point** and roughly constant in magnitude across the whole second half
of training, on two regression probes that were still improving at the end. That is a stronger
statement than any single epoch's gap, and it is the form the release should use.

### Val loss points the other way, and is admissible here

Final epoch (400 completed): **R4 0.43879, R5 0.44125** — R4 reconstructs **0.56% better**.

Under ADR-0010's amended rule val loss is admissible across arms *only* when the loss target and
mask ratio match. R4 and R5 differ **only** in init, so it matches, and this is a real
disagreement rather than the R2/R3 artefact where the loss target had changed.

So: **the warm start reconstructs better; the scratch run represents better.** §10 says the
probes adjudicate and reconstruction is never the criterion, so the release recipe stands — but
this is the first arm where the two instruments genuinely oppose, and the write-up should say so
rather than quote only the half that wins.

## 3. The released checkpoint has never been probed

Milestones fire at `state.epoch ∈ {50, 100, 200, 400}` — epochs **completed**
(`checkpoint.py:588-590`). The probe fires at completed 1, 21, … 381.

```
{50, 100, 200, 400} ∩ {1, 21, 41, …, 381} = ∅
```

**Not one of the four released milestone checkpoints carries a probe measurement, and neither do
the released final weights.** This is [#131](https://github.com/JStoschek/DEMAE/issues/131)
("No run probes its own final weights") and it lands directly on #88, because §14 ships the
milestone set *and* the training curves as one release.

How much this costs, measured rather than assumed — R5 over the final 19 epochs
(`_step` 380 → 399):

| | 381 completed | 400 completed | Δ |
| --- | ---: | ---: | ---: |
| `lr` | 7.390e-06 | 3.843e-13 | → 0 |
| train loss | 0.44430 | 0.44422 | −0.00008 |
| val loss | 0.44136 | 0.44125 | −0.00012 |

The tail is flat: the LR has annealed to nothing and val moves by 1.2e-4. Carrying the
381-completed probe numbers forward to the released weights is therefore **defensible** — but it
is an **inference, and the release must label it as one**. The honest sentence is "the nearest
probe point is 19 epochs earlier, over which the LR annealed to zero and val loss moved by
0.0001", not a bare number attached to the shipped file.

For the milestone checkpoints the gap is wider and the same rule applies: the nearest probe to
milestone 50 is completed epoch 41 or 61, and at that stage of training the curves are still
moving fast. §14 keeps the milestones so **Phase 2** can plot downstream performance against
pretraining compute, and Phase 2 measures its own y-axis — so this is not fatal to their
purpose. It is fatal to any Phase-1 sentence that attaches a probe number to a milestone file.

## 4. Precondition for the packaging half: `pos_embed` is not in the state dict

Verified against the real presets on this laptop:

```
encoder-only tensors: 149      params: 85,254,144
timm.create_model("vit_base_patch16_224", in_chans=1, num_classes=0).load_state_dict(enc, strict=False)
  MISSING:    ['pos_embed']
  UNEXPECTED: []
```

`model.py:260-264` registers `pos_embed` with `persistent=False`, so it is **absent from
`state_dict()`**. timm's `VisionTransformer` carries `pos_embed` as a learned `nn.Parameter` and
requires it. Every other one of the 149 names matches exactly and nothing is unexpected — so a
consumer doing the obvious `strict=False` load gets a perfect-looking result and **timm's own
randomly-initialized positional embedding**, silently. Nothing in the loss, the shapes or the
key report would show it.

This is the map's recurring shape — the seam between two correct halves, and the silent lever
that trains the wrong thing without changing a printed number — on a new axis, and it is aimed
at the one artifact outside consumers actually load.

It is worse than a generic export bug because **R4 and R5 do not share an axis order**:

- R5 (`vit_base`): `pos_encoding = sincos2d` (column-then-row, MAE's)
- R4 (`vit_base_continual`): `pos_encoding = sincos2d_rowfirst` (mmselfsup's transpose, P102)

So the exported `pos_embed` must be **materialized from each run's own config**, not
regenerated with a default — a default would ship R4 with R5's ordering, which P102 measured as
the difference between 6.7e-7 and 2.0 on the same tensor.

The release export therefore owes three things: materialize `pos_embed` into the exported state
dict from the run's own `Architecture`; assert a **strict** timm load with zero missing and zero
unexpected keys; and record the axis order in the artifact's own metadata so a consumer
rebuilding it cannot pick the wrong one.

## 5. What the release should say about scratch-vs-continual

The map's fog patch — *"how the scratch-vs-continual comparison is framed in the release"* — was
waiting on which way the result went. It is:

1. **Lead with the probes, and with the consistency rather than a single epoch.** "The scratch
   arm out-represents the ImageNet-warm-started arm at every one of 19 matched probe points from
   epoch 21 onward, on all three probes, scored on a byte-identical 5,000-tile set."
2. **Report the disagreement.** The warm start ends 0.56% better on masked-reconstruction val
   loss. §10 pins the probes as the criterion, and the reason is stated in the plan, not
   invented afterwards — but a release that quotes only the probes when the two instruments
   split is not making a checkable claim.
3. **Do not present this as evidence that warm-starting cannot help.** It is one pair of runs at
   one corpus size, and the init used is an **independent reproduction** trailing the official
   weights by 0.9 pt ImageNet linear probe at the checkpoint used (#103). What the run supports
   is narrow: *at 500k tiles on this corpus, ImageNet-MAE initialization bought nothing the
   scratch run did not reach, and cost reconstruction-side nothing either* — and GFM's
   large scratch-vs-warm-start effect did not transfer to this modality at this scale.
4. **State the epoch-0 point correctly or not at all.** R4 is at 0.2990 stratum after one epoch
   against R5's 0.1798 — the warm start starts ahead and is caught by epoch 21. That is the
   interesting shape, and it is not a frozen-init measurement.

## 6. Ledger

- Both runs' full histories pulled and cached; `probe/set_md5` identical across all 40 points.
- R4 `state = finished`; R5 `state = crashed` in W&B — the box vanished after the run completed
  ([#130](https://github.com/JStoschek/DEMAE/issues/130)); 400 epoch rows are present and the
  final milestone checkpoint was already verified byte-wise in #87.
- `s3://demae-checkpoints/r4/` 326 objects, `r5/` 309, all four milestones present in both.
