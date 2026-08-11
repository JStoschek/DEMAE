# P134 — Is TerraMind's `pos_emb` resize material? Measured on 300 tiles

Ticket [#134](https://github.com/JStoschek/DEMAE/issues/134) · map [#117](https://github.com/JStoschek/DEMAE/issues/117) · decides the fate of [ADR-0011](../adr/0011-terramind-posemb-rederived-at-14x14.md) §Decision 4's pre-registered ablation row in plan [`task1-void-fill-plan.md`](task1-void-fill-plan.md) §2.2 · builds on [#118](https://github.com/JStoschek/DEMAE/issues/118) and [#133](https://github.com/JStoschek/DEMAE/issues/133)

> **What this document is.** The record of a run, and the discharge of a pre-registration. Every
> number below came out of 1,500 forward passes of `terramind_v1_tokenizer_dem` over 300 real
> `corpus_500k` tiles on 2026-08-11, reproducible from the tree:
> [`scripts/p134_posemb_materiality.py`](../../scripts/p134_posemb_materiality.py) writes
> [`p134-posemb-materiality.json`](p134-posemb-materiality.json), and
> `tests/test_p134_posemb_materiality.py` pins the headline numbers and the decision rule against
> that JSON so neither can drift out of this prose silently.
>
> **The threshold was fixed before the numbers existed.** ADR-0011 §Decision 5 wrote down
> "mean per-token cosine >= 0.99 between bicubic and centre means immaterial, drop the row" on
> 2026-08-10, one day before this run. That is the whole methodological point of the ticket, and
> it is why §2 states the rule again before §3 states the result.

## 1. Bottom line

**0.9708. The row stays.**

Mean per-token cosine between TerraMind's features under terratorch's default bicubic `pos_emb`
resize and under ADR-0011's centre-anchored crop is **0.9708** over 58,800 tokens — below the
pre-registered 0.99, so plan §2.2's ablation row survives and is what will convert ADR-0011's
geometric argument into a number when A5 runs.

Three things worth carrying out of this beyond the verdict:

1. **It is a tail, not a shift.** The *median* token moves almost not at all (cosine **0.9932**).
   The mean is dragged under the line by a minority that moves enormously: the 1st percentile is
   **0.525** and the worst token in the sample is **−0.088**, i.e. slightly *anti*-parallel to
   what it should have been. "TerraMind's features move by 3%" would be the wrong summary; "most
   tokens are fine and a few are unrecognizable" is the right one (§4).
2. **CKA disagrees with cosine, and the disagreement is the finding.** Linear CKA is **0.9815** —
   noticeably kinder than the cosine. The two arms induce nearly the same *geometry* over tokens
   while placing individual tokens in visibly different places. A head that reads a pooled or
   rotation-tolerant summary would barely notice; a head that reads tokens positionally would.
   §2.2's head is the second kind (§3.2).
3. **The margin is not as wide as three decimal places make it look.** The shortfall below the
   threshold is 0.019, and the number is sample-sensitive at roughly that scale: per-stratum means
   span 0.935 to 0.996 (§5), and an earlier draft of the *tile-selection rule* moved the headline
   by about 0.03 (§6). The verdict is robust — every sample tried lands well under 0.99, and five
   of twenty strata clearing the line individually does not come close to carrying the other
   fifteen — but nobody should quote 0.9708 as if the fourth digit meant anything.

And one thing that did **not** happen: no result here could have restored bicubic, and none did.
ADR-0011 is unchanged. Feature drift cannot adjudicate — "better" needs a metric, a metric needs
the trained head, and no arm runs on this map. This is a materiality check on a table row.

## 2. The rule, restated before the result

From ADR-0011 §Decision 5, fixed 2026-08-10:

> **At >= 0.99 bicubic-vs-centre the choice is immaterial and the ablation row is dropped; below
> that the row stays.**

Measured: **0.9708 < 0.99 → material → the row stays.** `p134-posemb-materiality.json` carries
`"threshold": 0.99`, `"verdict": "material"`, `"ablation_row": "keep"`, and
`test_verdict_follows_from_the_recorded_number` recomputes the rule from the recorded cosine
rather than trusting those fields.

## 3. What was run

### 3.1 Three arms, one tensor apart

300 tiles through `terramind_v1_tokenizer_dem`'s encoder three times, varying only
`encoder.pos_emb`:

| arm | embedding | what it is |
|---|---|---|
| `bicubic` | `F.interpolate(pos_emb, (14,14), mode="bicubic", align_corners=False)` | terratorch's default on every 224 px pass |
| `centre` | `pos_emb[:, :, 1:15, 1:15]` | ADR-0011's spec |
| `top_left` | `pos_emb[:, :, :14, :14]` | bit-identical to `build_2d_sincos_posemb(14, 14)` (#133 claim 1) |

All three are installed **at 14x14**, which matters more than it looks. `ViTEncoder.forward`
resizes whatever it finds to the grid the input implies, unconditionally — so the centre and
top-left arms still pass through `F.interpolate`. At 14->14 with `align_corners=False` the sample
index is `(i + 0.5) * 1 - 0.5 == i` exactly, the bicubic kernel lands on integer offsets, and the
resize returns its input **bit-for-bit**. Measured at run time
(`"resize_error_at_14x14": 0.0`) and pinned by a test rather than argued, because it is the
entire basis for "the three feature maps compare elementwise with no resampling confound."

**And the bicubic arm is checked against the untouched model, not just derived from reading
`forward`.** Pre-resizing to 14x14 and installing that only reproduces terratorch's default if
nothing else in `forward` keys off the embedding's spatial size — a claim about someone else's
code, and reading is a poor way to settle it. So every batch is *also* encoded with the
checkpoint's 16x16 left exactly as it came, letting `forward` do its own resize. Worst token over
all 300 tiles: cosine **0.9999982**, against a run-to-run noise floor of 0.9999984 (§3.4). The two
are the same quantity to within float32 nondeterminism, which is what "the arm is the default"
looks like when you measure it. Cost: a quarter of the compute.

Also confirmed in passing, discharging ADR-0011's tamper-evidence consequence: the `pos_emb` that
came down with the checkpoint **is** the closed form — `[1, 768, 16, 16]`, equal to a freshly
built `build_2d_sincos_posemb(16, 16)` (`"loaded_pos_emb_is_the_closed_form": true`). Cropping it
really is re-deriving it. If a terratorch release ever ships a learned embedding here, that flag
flips and both #133's argument and this one need redoing rather than quietly continuing.

### 3.2 Two metrics, because they answer different questions

**Mean per-token cosine** is what ADR-0011 pre-registered: cosine over the 768-d channel axis of
each of the 196 tokens, averaged over all 300 x 196 = 58,800 of them. It is a statement about one
fixed basis, and it is the right statement when a downstream head reads tokens positionally.

**Linear CKA** (Kornblith et al. 2019) is reported beside it because cosine punishes changes a
head would not care about. A rotation of the 768-d space leaves every linear head equally able to
read the features while dragging cosine toward zero; CKA is invariant to exactly that, plus
isotropic scaling and translation. It is computed over the same 58,800 token rows, streamed as
768x768 cross-products rather than materialized, so the sample size is a free parameter.

Where they agree, the reading is easy. Where they disagree — and here they do, 0.971 vs 0.981 —
the gap says *how much of the movement is a change of geometry versus a change of coordinates*.

### 3.3 The tiles

**300: 15 from each of the corpus's 20 strata**, drawn from the test split with
`land_frac == 1.0`. Stated rather than sampled, following #118's precedent, so the list is
re-derivable by anyone holding the manifest: within each stratum, take the eligible `tile_id`s in
sorted order and pick 15 evenly spaced indices, endpoints included.

Balanced across strata rather than proportional, because the question is whether the drift is
*uniform*: the strata are the corpus's own terrain partition (ADR-0005), so an equal quota buys
the widest terrain spread a fixed tile budget can express, where a proportional draw would spend a
third of the sample on the three largest strata.

What the sample covers, recorded in the JSON:

| | |
|---|---|
| tiles | 300 (15 x 20 strata) |
| distinct source scenes | **208** |
| latitude | −88.0° to +80.0° |
| longitude | −168.3° to +159.8° |
| local relief | 12.7 m to 1546.7 m |

Normalization is TerraMind's **global** z-score, `(metres − 670.665) / 951.272`, read out of the
installed package at run time and applied to raw int16 metres straight from the shards — never
DEMAE's per-tile standardization, whose failure here is completely silent (#118 §7). The script
reuses #118's reader and constants by module import rather than copying them, so there is no
second place for that to go wrong.

### 3.4 The noise floor

Every batch re-encodes the centre arm a second time and keeps the worst token. Two runs of the
*same* embedding land at cosine **0.9999984** — not bit-identity, because float32 reduction order
on a threaded CPU matmul is not reproducible pass to pass. So `1 − cosine` has a floor of
**1.6 x 10⁻⁶**, and the 0.029 gap the run is reporting clears it by **four orders of magnitude**.
The numbers below are the embedding and nothing else.

## 4. The result

Over 58,800 tokens, all three pairings:

| pair | mean per-token cosine | linear CKA | median | p01 | min |
|---|---:|---:|---:|---:|---:|
| **bicubic vs centre** | **0.9708** | **0.9815** | 0.9932 | 0.525 | −0.088 |
| bicubic vs top-left | 0.9258 | 0.9533 | 0.9800 | 0.223 | −0.172 |
| centre vs top-left | 0.9363 | 0.9610 | 0.9849 | 0.256 | −0.111 |

**The distribution is the story.** Mean 0.971 against median 0.993, with a standard deviation of
0.084 and a 1st percentile at 0.525: this is not a small uniform rotation of every token, it is
most tokens landing almost exactly where they were and a minority landing somewhere else
entirely. That shape is what makes the row worth a line in the results table. A uniform 3% drift
would be easy to dismiss as noise the head absorbs; a heavy tail is a claim about *specific
patches* whose features the two variants disagree about, and whether that costs anything is
precisely the question an ablation row answers and prose cannot.

**Centre is still the crop nearest current behaviour**, in feature space as it was in embedding
space (#133 claim 4): 0.9708 vs 0.9258 against bicubic, on both metrics. This is not a decision —
ADR-0011 already chose centre, and on a different argument (keep the intervention
single-purpose). It is worth recording because an *inversion* here would have put the ADR's
reading of the mechanism — "bicubic centres and stretches, and only the stretch is wrong" — in
trouble. It did not invert.

## 5. What is under the mean

Two places a verdict-by-mean could be hiding something. Both are recorded in the JSON; neither
changes the verdict, and one of them contradicts a prediction.

**Terrain: the drift is worse on flat ground, not rugged.** Per-stratum means run from **0.9349**
(stratum 2, mean relief 52 m) to **0.9961** (stratum 10, mean relief 696 m), and 5 of the 20
strata sit above 0.99 on their own. Across strata, mean cosine correlates *positively* with mean
local relief (Pearson r = **+0.59** over 20 points) — more relief, less drift. A plausible reading
is that on rugged tiles the content dominates the features and the position embedding is a small
part of what the encoder has to work with, where on near-featureless tiles it is a large part. The
correlation is moderate and has clear exceptions (stratum 7 at 60 m relief sits at 0.993), so this
is an observation, not a mechanism, and nothing downstream should lean on it. The direction is
mildly reassuring for the benchmark: void filling lives in rugged terrain, which is where this
matters least.

**Position: ADR-0011's edge-ward accumulation does not survive into feature space.** In embedding
space the index error is monotone — it grows along each row and by the far edge a token has
drifted past its neighbour outright (#133 claim 3). Averaged over tiles, the feature-space damage
does **not** look like that. It is real and structured — per-position means run **0.923 to
0.994**, four orders of magnitude above the noise floor — but the structure is almost entirely
**columnar and non-monotone**. Averaged down the grid, column means are

```
0.978 0.958 0.956 0.980 0.978 0.974 0.979 0.979 0.971 0.970 0.984 0.960 0.951 0.972
```

against row means that barely move at all (0.962 to 0.976). The worst columns are **1-2 and
11-12** — one and two in from each edge — while columns 0, 3 and 10 are among the *least*
disturbed. The four corners land at 0.958-0.980 against a grid mean of 0.971, i.e. unremarkable;
under a monotone edge-ward story one corner should have been the worst token on the tile and the
opposite one the best. Neither is: the worst is at (12, 11) and the best at (9, 3).

That is unsurprising in hindsight — twelve layers of global self-attention redistribute a
disturbance, so a monotone perturbation of the input need not produce a monotone perturbation of
the output — but it is worth stating plainly because ADR-0011's prose invites the opposite
expectation. **No claim is made here about why the pattern has the shape it has.** Establishing
that would need per-layer probing, which this ticket does not own and no in-scope arm needs.

## 6. A methods note worth more than it looks

The first draft of the tile-selection rule was #118's, scaled up: *the lowest `tile_id`s in each
stratum*. Unimpeachable for the single tile #118 needed. Wrong for 300, and silently so — a
Copernicus tile id begins with its latitude band (`..._N00_00_E030_00_...`), so sorting is a
latitude filter wearing a sort's clothing, and all 300 tiles landed in one corner of East Africa
(28 source scenes, 30–40°E, 0–8°N).

**Nothing in the metrics showed it.** That run returned mean cosine 0.9428 and CKA 0.877 — a
perfectly plausible-looking answer, the same verdict, and 0.028 away from the number this document
reports. The fix was one line (evenly spaced indices instead of the first N), the re-run cost 56
seconds, and the guard is now on the *sample* rather than on the metrics:
`test_the_sample_is_spread_over_the_globe_not_one_region` asserts the source-scene count and the
lat/lon span, because those are the things a plausible number cannot vouch for.

> **Those two pilot figures are not reproducible from this tree** and are not pinned by any test —
> the superseded rule is gone from the script, and reinstating it as a flag would be carrying dead
> code to make one paragraph checkable. They are reported as a *caution about the method*, not as
> evidence for anything. Nothing in §1 or §8 depends on their exact values; what depends on them is
> the qualitative point that the headline is sample-sensitive at the second decimal, which is
> visible again in §5's per-stratum spread (0.935 to 0.996) and does not need them.

Two consequences. First, the honest reading of the headline is "0.97, not 0.99" and not
"0.9708" — a defensible change to the sampling rule moved it by more than its distance from the
threshold. Second, anything else in this repo that scales a "lowest id" rule from one tile to a
sample inherits the same defect; #118's own use is fine, because one tile chosen by a stated rule
makes no representativeness claim.

## 7. Cost, and where it runs

| | |
|---|---|
| Full run, 300 tiles x 5 passes, CPU, warm weights | **62 s** |
| Environment install (113 packages, from the #118 lockfile) | ~2 min |
| Peak footprint | one 286 M-parameter model in fp32, ~1.1 GB RAM |

Laptop, no GPU, no rented box — consistent with #118 §10. Five passes per tile, not three: the
three arms plus the two controls (§3.1's untouched-model check and §3.4's self-consistency
re-encode). The controls are 40% of the compute and both were kept, because 25 seconds is a cheap
price for the difference between "0.97" and "0.97, against a floor of 0.0000016, with the baseline
arm demonstrated to be the library's own behaviour."

**terratorch is still not a project dependency and must not become one.** It was installed into a
throwaway venv from [`p118-terramind-requirements.txt`](p118-terramind-requirements.txt), the
exact resolved environment #118 recorded, so this run and #118's are comparable by construction:
terratorch **1.2.10**, diffusers **0.39.0**, torch **2.13.0**, timm **1.0.28**, numpy **2.5.1**.

## 8. What this changes

- **Plan §2.2's ablation row is confirmed, not dropped.** Its bullet's conditional — "may be
  dropped before any arm runs if #134 returns >= 0.99" — is discharged, and the bullet now records
  the measured 0.9708 instead of the condition.
- **ADR-0011 §Decision 5 gains its outcome.** The ADR is otherwise unchanged: nothing here could
  have restored bicubic, and nothing did.
- **The row's design gains one constraint from §4.** A pooled or rotation-tolerant readout would
  understate this — CKA 0.981 against cosine 0.971 is exactly that gap — so the ablation must be
  reported on §2.2's actual positional head, not on a summary statistic of the features.
- **Nothing changes for A9 or the secondary 10 m regime.** A9 shares the encoder up to a 1x1
  convolution (#118 §5.2) and inherits the same `pos_emb` treatment; the 10 m regime is a GSD
  question, not a grid one.

## 9. Sources

- Pre-registration: [ADR-0011](../adr/0011-terramind-posemb-rederived-at-14x14.md) §Decision 5,
  dated 2026-08-10
- The geometry this checks the materiality of:
  [`p133-posemb-geometry.json`](p133-posemb-geometry.json) and
  [`scripts/p133_posemb_geometry.py`](../../scripts/p133_posemb_geometry.py)
- The encoder, its constants and its cost:
  [`p118-terramind-encoder-probe.md`](p118-terramind-encoder-probe.md) §5–§10
- terratorch 1.2.10 source: `terratorch/models/backbones/terramind/tokenizer/models/vit_models.py`
  (`ViTEncoder.forward`'s unconditional `F.interpolate`, `build_2d_sincos_posemb`)
- CKA: Kornblith, Norouzi, Lee, Hinton, *Similarity of Neural Network Representations Revisited*,
  ICML 2019, <https://arxiv.org/abs/1905.00414> — linear CKA, eq. 6–7
- This run: [`p134-posemb-materiality.json`](p134-posemb-materiality.json),
  [`p118-terramind-requirements.txt`](p118-terramind-requirements.txt) (the resolved environment)
