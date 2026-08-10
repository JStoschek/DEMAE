# ADR 0011 — TerraMind's position embedding is re-derived at 14x14, not bicubically resized

- **Status:** Accepted
- **Date:** 2026-08-10
- **Context ticket:** "Decide the primary regime's input size, now that TerraMind's native grid is 16x16" ([issue #133](https://github.com/JStoschek/DEMAE/issues/133)), part of the Task-1 void-fill map ([#117](https://github.com/JStoschek/DEMAE/issues/117))
- **Evidence:** terratorch 1.2.10 source (`terramind/tokenizer/models/vit_models.py`); the probe run in [#118](https://github.com/JStoschek/DEMAE/issues/118) §6; the closed-form checks recorded in [Context](#context) below, which need no weights and no GPU
- **Amends:** [`docs/research/task1-void-fill-plan.md`](../research/task1-void-fill-plan.md) §2.2
- **Materiality check:** [#134](https://github.com/JStoschek/DEMAE/issues/134), pre-registered, may drop the ablation row this ADR adds

## Context

Plan §2.2 puts the primary regime at 224x224 and rests a fairness premise on it: "Both ViTs
produce a 14x14 token grid, so the head is byte-identical." [#118](https://github.com/JStoschek/DEMAE/issues/118)
confirmed that premise and, in passing, found what it costs. TerraMind's DEM tokenizer was
trained at 256 px — its recorded training args say `input_size_min=256, input_size_max=256` —
so `ViTEncoder.pos_emb` is natively `[1, 768, 16, 16]`, and `ViTEncoder.forward` reaches a
14x14 grid by resizing it:

```python
x_pos_emb = F.interpolate(self.pos_emb, size=(N_H, N_W), mode="bicubic", align_corners=False)
```

#118 read this as a small one-sided handicap — TerraMind running 12.5% off its native
resolution while DEMAE runs at its own — and routed the call here with three options: name the
handicap, quantify it, or move the primary regime to 256 px.

**All three assume the handicap is intrinsic. It is not.** Reading the terratorch source rather
than the model card:

```python
sincos_pos_emb: bool = True,
learnable_pos_emb: bool = False,
...
if sincos_pos_emb:
    self.pos_emb = build_2d_sincos_posemb(h=N_H, w=N_W, embed_dim=self.dim_tokens)
    self.pos_emb = nn.Parameter(self.pos_emb, requires_grad=learnable_pos_emb)
```

The embedding is **not learned**. It is a closed-form function of the token's *integer* grid
index — `build_2d_sincos_posemb` builds its grid from `torch.arange(w)` — and appears in the
checkpoint only because it is wrapped in an `nn.Parameter`. Two consequences follow, both
verified numerically in float64 with no weights loaded:

**1. The correct 14x14 embedding is exactly a crop of the 16x16 one.** Because the grid index
is absolute, `build_2d_sincos_posemb(14, 14)` is *bit-identical* to `pos_emb[:, :, :14, :14]` —
measured max absolute difference **0.0**. This matters because a 224 px tile at 30 m **is** a
crop of a 256 px scene at 30 m: same 16 px patch, same stride, fewer patches. The faithful
embedding for a crop is the crop.

**2. What the resize does instead is stretch the index spacing.** `align_corners=False`
preserves *normalized* position, so it samples the 16-grid at indices 0.071 … 14.929 —
spacing **1.1429** where a crop's spacing is exactly **1.0**. Against the correct embedding:

| Measure | Value |
|---|---|
| Relative RMS error | **26%** (entries lie in [-1, 1], RMS 0.707) |
| Per-token cosine to its own correct embedding | 0.909 – 0.966 |
| Cosine between *distinct* correct tokens | 0.66 – 0.986 |
| Tokens whose nearest correct embedding is themselves | **20 / 196** |

The last row is the one that bites. A 0.97 cosine sounds harmless until you notice adjacent
positions sit at 0.986 — the error is *inside* the gap between neighbouring patches. Because
the stretch is systematic rather than noisy, it accumulates monotonically: by column 10 of row
0 a token has drifted past its neighbour outright.

So the accurate statement is not "TerraMind runs 12.5% off its native grid." It is: **TerraMind
is told its patches sit 14% further apart than they do, and the error accumulates past a full
patch width by the far edge of the tile.** Bigger than #118 assumed — and removable exactly,
for free.

### The anchor is a real choice, not a detail

If the fix is "crop, don't resize," the crop needs an anchor, and DEMAE's 224 px tiles are
standalone rasters — not crops of any 256 px scene — so no anchor is ground truth. TerraMind
saw every index 0-15 in training, so all nine anchors are in-distribution. They are not
interchangeable:

| Anchor | rel RMS vs top-left | rel RMS vs bicubic | Tokens bicubic keeps identity for |
|---|---|---|---|
| (0,0) top-left | 0.0000 | 0.2625 | 20/196 |
| (1,1) **centre** | 0.2335 | **0.1357** | **72/196** |
| (2,2) | 0.4376 | 0.2625 | 24/196 |

Anchors differ from one another by up to 44% relative RMS — **more than the bicubic error being
fixed**. Centre is by a clear margin the nearest honest crop to current behaviour, which follows
from the mechanism: bicubic preserves normalized position, so it is implicitly centred and
stretched on top.

## Decision

**1. The primary regime stays at 224x224 at native 30 m.** No arm moves.

**2. TerraMind's `pos_emb` is re-derived as the centre-anchored 14x14 crop** —
`pos_emb[:, :, 1:15, 1:15]`, indices 1-14 — replacing terratorch's bicubic resize on every
frozen-encoder pass (A5, A9 and the ablations hanging off them).

Centre rather than top-left because it keeps the intervention **single-purpose**. Bicubic does
two things: it centres, and it stretches. Only the stretch is wrong. Top-left would remove the
stretch *and* add a one-token translation, leaving two changes to defend instead of one. The
appeal of top-left — that it gives both encoders the identical index range 0-13 — is
mechanically inert, since the head consumes features and never sees a position index.

**3. 256 px is rejected, and its symmetry argument is wrong.** Because both models use fixed
sin-cos, neither needs a handicap; but the index ranges are asymmetric in the direction
*opposite* to the one #133 assumed. At 224 px TerraMind sees indices 1-14, a subset of its
trained 0-15, and DEMAE sees 0-13, exactly its own range — both in-distribution. At 256 px
TerraMind would see its trained 0-15 while DEMAE saw indices **14 and 15 for the first time**.
Sin-cos yields well-defined values there, but the model has no learned response to them, and
re-derivation cannot fix it: you cannot crop into indices that do not exist. That is before
retiling the corpus to 7.68 km windows, which reopens the frozen Phase-0 tiling, strata and
split. This is a *different* question from the out-of-scope secondary 10 m regime, which is
about GSD, not grid.

**4. One ablation row is pre-registered: bicubic vs re-derived, A5 only, at the 20k training
size** — mirroring §2.2's existing precedent for the secondary regime, so the intervention is
evidenced in the results rather than asserted in prose. It may be dropped before any arm runs;
see below.

**5. The materiality check is [#134](https://github.com/JStoschek/DEMAE/issues/134), with its
threshold fixed now.** This ADR rests on geometry and a formula check, not on measured model
behaviour, and that gap is stated rather than papered over. #134 encodes a few hundred tiles
under all three variants and reports mean per-token cosine and CKA. **At >= 0.99 bicubic-vs-centre
the choice is immaterial and the ablation row is dropped; below that the row stays.** Note what
#134 cannot do: feature drift cannot pick a winner, because "better" needs a metric and a metric
needs the trained head. There is no result that restores bicubic — bicubic is geometrically wrong
whatever the drift — so the check governs the row, not the spec.

## Consequences

- **§2.2 no longer owes the reader a handicap caveat.** The head-to-head's weakest point becomes
  a point in its favour: both encoders are in-distribution at 224 px simultaneously, which is what
  C1's "identical trained head, same tiles, same masks, same seeds" fairness argument actually
  wants. C1 can no longer be dismissed with "you crippled the baseline."
- **We now diverge from the library's default inference path**, and the writeup must say so
  plainly — one sentence, with 1.1429-vs-1.0 attached. The burden shifts from "we handicapped
  TerraMind" to "we corrected how terratorch feeds TerraMind, and here is why ours is faithful."
  This is the thing a reader rediscovers in a year asking why our TerraMind does not match
  terratorch's, which is why it is an ADR and not a code comment.
- **The intervention must be tamper-evident.** Following #118's precedent — tests asserting shape
  claims against recorded JSON so prose cannot drift silently — the extraction path should assert
  `pos_emb[:, :, 1:15, 1:15]` against a freshly built `build_2d_sincos_posemb(16, 16)` slice, so a
  terratorch upgrade that changes the embedding is a test failure rather than a silent regime change.
- **The ImageNet-MAE arm needs no equivalent.** Standard MAE ViT-B/16 uses fixed 2D sin-cos at
  224 px, natively 14x14, so no resize occurs. Worth a one-line assertion when that arm is built;
  it is not a decision.
- **DEMAE's own side is unaffected but worth recording:** [`src/pretrain/model.py`](../../src/pretrain/model.py)
  registers `pos_embed` as a *non-persistent* buffer rebuilt from config, so it is re-derivable at
  any grid too. That is what makes the 256 px asymmetry an argument about *training* range rather
  than about checkpoint surgery.

## Alternatives considered

- **Name the handicap and keep 224 (#133 option 1).** Rejected: it accepts a defect that costs
  nothing to remove, and the caveat reads as damage control once the numbers exist — which is
  exactly what pre-registration is meant to prevent.
- **Quantify the handicap by extracting at 224 and 256 (#133 option 2).** Rejected as posed. It
  measures the wrong contrast: upsampling a 224 tile to 256 changes GSD *and* grid at once, so
  the number is confounded, and cropping 256 px at native GSD needs the retiling that option 3
  is rejected for. #134 measures the contrast that is actually actionable instead.
- **Move the primary regime to 256 px (#133 option 3).** Rejected — see Decision 3.
- **Top-left anchor.** Rejected — see Decision 2.
