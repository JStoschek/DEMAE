# P163 — What A9's diffusion decode actually costs

**Ticket:** [#163](https://github.com/JStoschek/DEMAE/issues/163), under map
[#145](https://github.com/JStoschek/DEMAE/issues/145).
**Artifacts:** [`p163-a9-decode-cost.json`](p163-a9-decode-cost.json) — every timing quoted
below, the sweeps they come from, and the projection derived from them.
**Amends:** [`task1-void-fill-plan.md`](task1-void-fill-plan.md) §7.
**Blocked:** ticket 23, the full A9 pass, which is not run here.

---

## 1. Verdict

**A9 is not the budget risk §7 feared. It is 11% of the GPU program and twelve cents.**

| | §7 as written | Measured (P163) |
|---|---|---|
| Per instance | "at 1 s/instance" | **2.72 s** on this laptop's MPS → **0.028 s** on the rented 5090 |
| Full pass, 39,978 instances | **11.1 GPU-h**, unpriced | **0.31 GPU-h** |
| Dollars | — | **$0.12** at $0.4016/hr |
| Standing | "§7 is not final until this is measured" | measured |

§7's placeholder was not merely high; it was the wrong side of the device. **1 s/instance on a
rented 5090 would make that card 2.7 times *slower* than the laptop this was measured on.** Read
as a 5090 rate it overstates the line by **36×**; the guess was a placeholder and is
labelled as one in §7, so this is a correction rather than an error found.

The one number §7 said could still move the total does not move it. Folding A9 in takes the GPU
program from **2.42 h / $0.97** to **2.73 h / $1.09** — against a **$25** target that was already
not binding. **No scope amendment against §7 is raised**, because none is warranted: nothing about
A9's design changes on this number, and §4.5's ordering can proceed as written.

**How wrong the transfer would have to be to change that: 201×.** p120 §3 asks this question of
the whole GPU program and answers it with a table; A9 can answer it exactly, because cost is
linear in the inverse ratio. p120's local-to-5090 transfer would have to overstate the 5090 by a
factor of **201** — not 2, not 20 — before A9 alone reached $25. At the most pessimistic reading
available anywhere in p120's own data (its smallest stable ratio, 86.8) the line is $0.14.

Three findings ride along, and two of them are for ticket 23 rather than for the budget:

- **`decode_quant` returns the wrong raster by default.** It sizes its output from the *decoder's*
  256 px training resolution and takes no hint from the 14×14 latent it is handed, so a 224 px
  tile comes back 256 px, silently (§5).
- **A9's pass is stochastic unless a generator is passed**, and A9 scores once with no seeds to
  average over (§6).
- **The transfer method gets an independent check it did not have.** A9's encoder forward,
  measured here and carried across by p120's ratio, lands at **0.139 ms/tile** against the
  **0.179 ms** §7 already prices from p120 — agreement to 22%, against p120's own stated 14%
  error bar plus this run's drift (§4).

---

## 2. What was measured, and on what

The full tokenizer — encode, FSQ, diffusion decode — over **40 real test tiles**, at TerraMind's
own evaluation setting of **50 DDIM timesteps**, decoding to the benchmark's **224 px** raster.
Not the full pass: #163 fixes that at ticket 23, blocked on this so the bill is known before it is
incurred.

**The tiles are p134's draw**, imported through p157's helpers rather than restated — the same
evenly-spaced, all-land, test-split rule at two per stratum over the corpus's 20 strata. Two
answers to "which tiles" is how a second sample quietly becomes a second population.

**The pass is A9's pass, and that is checked rather than asserted.** The whole tokenizer comes down
through terratorch's registry, because A9 is the one arm that needs the 196 M-parameter decoder
`benchmark.extract` deliberately declines to load. But the registry's encoder half is terratorch's
own build, carrying the native 16×16 `pos_emb` that [ADR-0011](../adr/0011-terramind-posemb-rederived-at-14x14.md)
replaces with the centre-anchored 14×14 crop — and #134 measured those two feature sets **0.9708**
apart. So the crop is substituted here too, through `benchmark.extract`'s own `derive_pos_emb`, and
the record carries the difference between this encoder's tokens and the ones
`Extractor.extract` returns for the same tiles:

**max abs difference 0.0** — not a tolerance, an equality, at fp32 on both sides.

The parameter split is p118 §8's, confirmed on the model that ran: encoder 90,126,336 (its
90,172,416 less ADR-0011's 46,080 cropped position weights), `quant_proj` 3,845, quantizer 0,
**decoder 196,140,560**. A9 is the arm that is mostly decoder, which is the whole reason it needed
its own line.

---

## 3. The measurement had to be rebuilt once, and that is worth reading

**The first run of this script measured one configuration three times and got 2.775, 3.272 and
3.597 s/instance** — a 30% spread, rising with elapsed time. Another session on this laptop was
holding a core at 100%. None of that is visible in a sweep that walks its conditions once in
order: the drift does not arrive labelled as drift, it arrives as a *result*, attributed to
whichever condition happened to run last.

The second run added a fixed control between phases and caught the same thing at **1.82×** — and
caught it being **non-monotone** (0.759 → 1.028 → 0.747 → 0.566 → 0.623), which falsified the
thermal-throttle reading and identified it as contention. It also produced a batch curve whose
*medians* rose from batch 8 to batch 16. A saturating curve cannot do that. A median over a
contended sample can.

So the estimator changed. **Interference is a one-sided error**: another process can make a timed
interval longer and can never make it shorter. The minimum over repeats is therefore the best
available estimate of the uncontended cost, the spread above it is other people's work rather than
variance in the decoder, and the pass this projects onto a rented 5090 will not be sharing that
card with anything. Every sweep point is the minimum of its repeats, the median is published beside
it, and the control's spread is published as the honest error bar: **1.10× across the run reported
here**.

`tests/test_p163_a9_decode_cost.py` pins the monotonicity of the batch curve for this reason. It
is the check that caught the contaminated run, and it is cheap to keep.

---

## 4. The batch curve, and the primitive

Per instance, seconds, minimum of five passes over the whole sweep:

| batch | seconds/instance | median of 5 | decode | encode |
|---|---|---|---|---|
| 1 | 3.905 | 4.797 | 3.879 | 0.026 |
| 2 | 3.341 | 3.609 | 3.301 | 0.026 |
| 4 | 3.027 | 3.057 | 3.009 | 0.018 |
| 8 | 2.857 | 2.914 | 2.842 | 0.015 |
| 16 | 2.765 | 2.791 | 2.750 | 0.014 |
| **32** | **2.731** | 2.887 | **2.718** | **0.014** |

The curve has flattened by 16 — batch 32 buys 1.2% over batch 16 — so 32 is the plateau and,
following p120 §3's rule, the transfer uses **p120's device ratio measured at that same batch**:
97.26, the geometric mean of its two independent anchors, whose spread there is 1.10.

**The primitive is 56 ms per instance per diffusion step** (§7 below), and the decode is 99.5% of
the arm. The encoder forward is 13.5 ms/instance locally.

**That encoder number is an independent check on the transfer.** §7 already prices a frozen ViT-B
forward at **0.179 ms/tile**, from p120. Carried across by the same ratio, this run's encode gives
**0.139 ms/tile** — 22% low, against p120's own stated 14% error bar on the ratio and this run's
1.10× drift. Two independent routes to the same primitive, by different scripts on different
phases of the same laptop, meeting within about a fifth: that is the sort of thing p120 §3's "one
internal check" was for, and it is worth having a second instance of. It is also the only evidence
in this record that the ratio itself travels, so it is reported at the precision it has and not
more.

**The per-tile spread is time, not terrain.** Forty tiles decoded one at a time span 3.92 to 5.99
s — but the fast ones arrive in *contiguous runs* (tiles 11–17 and 37–40 of the draw), which is
what intermittent contention looks like and is not what terrain would do to a draw ordered by
stratum. Read off the floor instead: the fastest quartile spans **under 7%** and is drawn from
several different strata. A fixed step count over fixed shapes has no data-dependent work in it,
and the measurement agrees.

---

## 5. `decode_quant` returns a 256 px raster for a 224 px tile

This is a correctness finding, not a cost one, and it belongs to ticket 23.

`DiVAE.decode_quant`'s `image_size` defaults to `self.model.sample_size` — the **decoder's**
training resolution, 256 — and the pipeline builds its initial noise at that size. It takes no
hint from the 14×14 latent it is conditioned on. So the naive call:

```python
dec = model.decode_quant(quant)          # -> (N, 1, 256, 256), silently
dec = model.decode_quant(quant, image_size=224)   # -> (N, 1, 224, 224)
```

A9 scored on the default would be scored on a raster 8/7 the linear size of its own input, against
a reference the same shape as the input. Nothing raises; the tensors are all valid; every metric
in §5 would compute and be wrong.

It is also not free: decoding at 256 costs **3.534 s/instance** against 224's **2.834**, a factor
of **1.247**. That is sub-linear in the 1.306 pixel ratio, which is what a fixed per-step overhead
looks like.

Pinned in the test file so a terratorch that changes the default is caught by a test rather than by
a scoring pass over silently rescaled rasters.

---

## 6. The pass is stochastic unless seeded

The diffusion pipeline draws its initial noise from the global RNG when no `generator` is passed.
Two decodes of the same latent, unseeded, differ by **4.8e-3**; with a seeded `torch.Generator`
they are **bit-identical**.

This matters more for A9 than it would for a trained arm. A9 trains nothing and §7 runs it as a
**single pass with no seeds to average over**, so an unseeded decode makes A9's row of the headline
table unreproducible — not noisy-but-averaged, simply not re-derivable. The noise is drawn on CPU
and moved to the device, so a CPU `torch.Generator` is sufficient and device-independent.

---

## 7. Cost is linear in timesteps, so the amendment lever is priced

Measured at batch 8, each step count run ascending then descending so drift enters both directions:

| timesteps | seconds/instance | ms/instance/step |
|---|---|---|
| 5 | 0.282 | 56.47 |
| 10 | 0.565 | 56.49 |
| 20 | 1.129 | 56.45 |
| **50** | **2.883** | **57.67** |

Flat to 2.2% across a tenfold range: the decode is one decoder forward per step and nothing else.
2.5× the steps costs 2.55× the seconds.

This is priced because it is the first lever any scope amendment would reach for, and the point of
pricing it is that **it should not be pulled**. Fifty is TerraMind's own `num_eval_timesteps`, out
of the DEM tokenizer's recorded training args. A9 is a *reference row* — §1 defines it as the
tokenizer as published, and its job is to ground the "the tokenizer was never meant as an
inpainter" caveat in a number. Cutting it to 20 steps would save $0.07 and would make it a row about a
cheaper configuration nobody published, and would hand any unflattering A9 result an objection
that costs more to answer than the whole line costs to run — the whole line costs twelve cents.

---

## 8. The projection, and how far it can be trusted

39,978 eval instances — §3.4's 30,000 synthetic rows plus #153's 9,978 real-void rows, which §7's
A9 row rounds to "40k".

| | |
|---|---|
| Local (this laptop's MPS, measured) | **30.18 GPU-h** |
| Transferred at p120's batch-32 ratio (97.26) | **0.31 GPU-h** |
| Dollars at $0.4016/hr | **$0.12** |
| At p120's most conservative stable ratio (86.8) | 0.348 GPU-h, **$0.14** |

**The row prices the decode alone, deliberately.** §7's inference line already prices a frozen
encoder forward for each of the 38 scored passes, and A9 is one of them — so charging A9's encode
here would count it twice. At 13.5 ms/instance it would have moved the line by 0.5% and the
verdict by nothing, but a double-count that small is exactly the kind that survives into a table.

The sensitivity table p120 §3 runs for the whole program, run for this line:

| transfer overstates the 5090 by | GPU-h | dollars | inside $25 |
|---|---|---|---|
| (as measured) | 0.31 | $0.12 | yes |
| 2× | 0.62 | $0.25 | yes |
| 5× | 1.55 | $0.62 | yes |
| 10× | 3.10 | $1.25 | yes |
| 20× | 6.21 | $2.49 | yes |
| 100× | 31.0 | $12.46 | yes |
| **201×** | **62.3** | **$25.00** | breakeven |

**What the transfer is worth, stated honestly.** p120 §3 already says this and it applies here with
one extra caveat. The ratio bundles a hardware gap with a software gap — the 5090 anchors ran
compiled with fused attention, this laptop runs eager — and the diffusion decoder is a **UNet**,
which is p120's own weaker case: convnets do not gain from `torch.compile` what transformers do.
So this transfer is an extrapolation of the same kind p120 flags for its A7 line, and it inherits
that weakness.

It does not matter here, and that is the point of quoting 201×. **The conclusion survives any
error the transfer could plausibly carry**, and it survives the honest worst case too: if the
transfer were *entirely* worthless and A9 had to run on this laptop, it is **30 hours of local
wall clock and $0**, because the laptop is not rented. That is a scheduling fact, not a budget
one — which is exactly the distinction #117's standing note draws and §7 repeats.

---

## 9. What this does not settle

- **Whether A9's output is any good.** This is a cost measurement. §1's expectation — from
  TerraMind's published DEM reconstruction error, MAE 170.7 m — is that the pass-through will not
  be competitive, and that remains a prediction until ticket 23 scores it.
- **The 5090 rate, directly.** No 5090 was rented, exactly as p120 did not rent one. What is
  measured is local; what is projected is projected, with its error bar published and its
  breakeven stated.
- **Any figure to better than ~10%.** The control moved 1.10× across this run on a shared box.
  Every number here is quoted at a precision the JSON supports and read at a precision the drift
  supports, and no conclusion in this memo turns on a difference smaller than that.
