# P157 — TerraMind's fp32-versus-bf16 agreement, measured rather than inherited

Ticket [#157](https://github.com/JStoschek/DEMAE/issues/157) · map [#117](https://github.com/JStoschek/DEMAE/issues/117) · discharges [ADR-0012](../adr/0012-frozen-feature-extractor-contract.md) §Decision 6's *"the build session measures it on first run"* · plan [`task1-void-fill-plan.md`](task1-void-fill-plan.md) §9.4 · arms A5, A9

> **What this document is.** The record of a run. Every number below came out of one
> `benchmark.extract` call over 60 real `corpus_500k` tiles on 2026-08-13, reproducible from the
> tree: [`scripts/p157_terramind_precision.py`](../../scripts/p157_terramind_precision.py) writes
> [`p157-terramind-precision.json`](p157-terramind-precision.json), and
> `tests/test_p157_terramind_precision.py` pins the headline numbers and the verdict rule against
> that JSON so neither can drift out of this prose silently.
>
> **It drives the contract's own measurement.** What is recorded is literally what
> `Extractor.precision_agreement("A5")` returns after one `extract` call — the same code path a
> real run records — rather than a parallel computation that could agree with it by luck.

## 1. Bottom line

**0.99888. Immaterial, and not for the reason the DEMAE figure is.**

Mean per-token cosine between TerraMind's features computed under `torch.autocast(bf16)` and the
fp32 reference is **0.99888** over 11,760 tokens, against [#134](https://github.com/JStoschek/DEMAE/issues/134)'s
0.99 materiality bar. So ADR-0012 §6 stands as written for this arm too: the compute dtype stays a
per-task declared field, and A5 runs at bf16 like everything else.

**What §6 was right not to assume is the shape of the disagreement.** Beside the DEMAE-family
figure §6 quotes:

| | DEMAE (R5, ADR-0012 §6) | TerraMind (this run) | ratio |
|---|---:|---:|---:|
| mean per-token cosine | 0.999968 | **0.998877** | 35× further from 1 |
| 1st percentile | 0.999809 | **0.985315** | 77× |
| worst token | 0.999431 | **0.893271** | 188× |
| relative RMS | 0.81% | **4.79%** | 5.9× |

Every column is the same verdict and a different magnitude. The one that matters is the third
row: **TerraMind's 1st-percentile token sits at 0.9853, below the 0.99 bar the mean clears
comfortably.** A reader who had inherited the DEMAE number would have carried "the worst token
we ever see is 0.9994" into a claim about this arm, and the worst token this arm sees is 0.893.
Nothing in scope reads a single token in isolation — §4.1's head consumes all 196 — so this
changes no decision. It is recorded because it is the thing the inherited number would have got
wrong, which is exactly what §6 declined to inherit.

**No explanation is offered for why the tails are wider**, and one would need per-layer probing
no in-scope arm asks for. Two candidates are visible and neither is tested here: TerraMind runs
twelve blocks with no final `LayerNorm` to contract the spread (ADR-0012 §3 — its `output` *is*
block twelve, p118 §5.3), and its residual `post_mlp` runs a `Tanh` MLP in float32 over a tensor
the autocast pass produced in bf16. Stating the candidates is as far as this run goes.

## 2. What was measured, and on what

One `extract` call, `arm=A5`, on MPS, at the contract's default `compute_dtype="bf16"` against
its fixed `reference_dtype="fp32"`. The extractor measures the pair itself on the first batch and
hands back the record; this script writes it out.

**60 tiles, three per stratum, over the corpus's 20 strata — 11,760 patch tokens.** The tile rule
is [p134](p134-posemb-materiality.md)'s, *imported* from its script rather than restated:
evenly-spaced ids from each stratum's sorted eligible list, `land_frac == 1.0`, `split == "test"`.
Balanced across strata rather than drawn proportionally, because the question is whether the
disagreement is uniform. The sample size is chosen against ADR-0012 §6's own DEMAE-side sample —
64 tiles, 12,544 tokens — so the two figures the ADR now carries side by side are measured at
comparable scale rather than one at ten times the other.

The mask passed is land-only and makes no difference to what is measured: A5's declared diet is
TerraMind's **global** affine `(metres − 670.665) / 951.272`, which has no per-tile term a mask
could enter (ADR-0012 Decision 1). It is passed anyway, so the run stays on the contract's one
code path rather than a shortcut around it.

## 3. Three things confirmed in passing

**1. The encoder that runs holds 90,126,336 parameters, not p118's 90,172,416 — and the
difference is the crop.** ADR-0011 substitutes the centre-anchored 14×14 `pos_emb` for the native
16×16 one, which is `768 × (16² − 14²)` = **46,080** fewer, all of them frozen position and none
of them trainable. The record carries the arithmetic as a checked field
(`params_delta_is_the_crop`) rather than as two numbers a reader has to reconcile. §2.3's
reported 90.2 M and §6 item 4's printed parameter counts are about the encoder as published;
this is the same encoder with one non-trainable buffer cropped.

**2. The whole path runs on MPS, and the disagreement above is measured there.** Same as
ADR-0012 §6's DEMAE figure, so the comparison in §1 is device-matched. p118 §10 already recorded
that TerraMind's CPU and MPS forward passes agree to ~1e-5.

**3. The run is deterministic.** Two invocations of the script produce byte-identical JSON,
which is what makes the committed record a fact about the contract rather than about the
afternoon.

## 4. What this changes

- **ADR-0012 §Decision 6's unmeasured half is closed.** See its Amendment 1. The decision is
  unchanged; what changes is that the TerraMind figure now exists and is this one.
- **Nothing in the arm table moves.** A5 and A9 keep the declared `bf16` compute dtype, and the
  contract keeps handing back fp32.
- **A9 is still unmeasured, deliberately.** It reads the same weights through a separate registry
  entry and gets its own figure on its own first run, for the reason this document exists.
- **What a *run* does with its own figure was open, and was settled by
  [#171](https://github.com/JStoschek/DEMAE/issues/171).** `Extractor` exposes it
  (`precision_agreement(arm)`); the run record — which already carries the spec's md5 beside the
  table it produced — now carries the agreement too, in its own `precision` block
  (`benchmark.results.precision_block`). **It records and refuses nothing.** The `immaterial`
  verdict travels on the figure, but no gate consults it: a refusal would be a registered rule,
  and registering a thirteenth in ADR-0013 for a bar no measured encoder has crossed would be
  registering a gate ahead of a case for it. The number in §3 above is why *which statistic* such
  a gate would read is the whole decision — this run's mean clears the bar and its 1st percentile
  does not.

## 5. Sources

- This run: [`p157-terramind-precision.json`](p157-terramind-precision.json), written by
  [`scripts/p157_terramind_precision.py`](../../scripts/p157_terramind_precision.py)
- The contract: [ADR-0012](../adr/0012-frozen-feature-extractor-contract.md) §Decision 6,
  [`src/benchmark/extract.py`](../../src/benchmark/extract.py)
- The threshold: [ADR-0011](../adr/0011-terramind-posemb-rederived-at-14x14.md) §Decision 5 /
  [#134](https://github.com/JStoschek/DEMAE/issues/134), read out in
  [`p134-posemb-materiality.md`](p134-posemb-materiality.md)
- The encoder: [`p118-terramind-encoder-probe.md`](p118-terramind-encoder-probe.md) §5.3, §8, §10
