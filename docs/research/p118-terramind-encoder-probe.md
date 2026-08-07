# P118 — TerraMind's DEM tokenizer encoder, run on a DEMAE tile

Ticket [#118](https://github.com/JStoschek/DEMAE/issues/118) · map [#117](https://github.com/JStoschek/DEMAE/issues/117) · plan [`task1-void-fill-plan.md`](task1-void-fill-plan.md) §2.2, §8, §9.4 · extraction path [`downstream-task-selection.md`](downstream-task-selection.md) §3.4 · arms A5, A9

> **What this document is.** A record of an actual run, not a reading of a model card. Every
> shape, constant and parameter count below came out of a forward pass on one real
> `corpus_500k` tile on 2026-08-07, and the run is reproducible from the tree:
> [`scripts/p118_terramind_probe.py`](../../scripts/p118_terramind_probe.py) writes
> [`p118-terramind-probe.json`](p118-terramind-probe.json), and
> `tests/test_p118_terramind_probe.py` asserts the four shape claims against that JSON so they
> cannot drift out of this prose silently. Where a claim is read from source rather than
> measured, it says so and names the file.

## 1. Bottom line

**It works, first attempt, in 36 seconds on a laptop. Nothing broke, so there is no breakage to
record — and the paper record was already largely right.** All five things #118 asked for are
confirmed. What the run changes falls into three tiers, and it is worth keeping them apart,
because most of what is interesting here is *confirmation of an unrun claim*, not correction:

**Two things the record got wrong.**

1. **`diffusers==0.30.0` is not a pin and never was.** Plan §8 and the selection memo's §3.4 both
   present it as required. terratorch declares a bare `diffusers` dependency; `0.30.0` appears
   only inside the text of a warning message that fires when the import fails. The resolver picks
   **0.39.0**, nine minor versions later, and the tokenizer builds and runs on it (§4).
2. **No account, and therefore not a human step.** `ibm-esa-geospatial/TerraMind-1.0-Tokenizer-DEM`
   is public, Apache-2.0, `gated: false`, and an anonymous HTTP request for the weights returns
   200. This is the opposite of [#127](https://github.com/JStoschek/DEMAE/issues/127)'s Earthdata
   situation, and #118's own text flags this as the fork that decides who does the work (§3).

**One thing the record had, that nobody had drawn the consequence of.** The selection memo already
says the position embeddings are "interpolated at runtime" and that training was at 256 px crops.
It does not connect that to §2.2's co-registration premise. Measured: `pos_emb` is natively
`[1, 768, 16, 16]`, so **at 224 px TerraMind runs 12.5% off its own grid while DEMAE runs at its**
— a small one-sided handicap in a head-to-head that the writeup now has to name (§6).

**One thing the record states as a comment, which this run shows is not enough.** The memo's
snippet says "global z-score, NOT per-tile". The run establishes *why* that comment cannot be the
only guard: the failure is **completely silent**. Feed the encoder DEMAE's per-tile-standardized
tiles and it returns a perfectly plausible `[1, 768, 14, 14]`, no error, no warning (§7). It is
now a test.

And one measurement for §9.4's frozen-feature contract: **the 1.15 GB checkpoint is 69% decoder.**
The encoder A5 and A9 actually need is 90.2 M parameters, 361 MB in fp32 (§8) — matching the
memo's "~90M encoder / ~196M decoder" estimate, now exact.

## 2. What was asked

#118 lists five things to establish on one real tile from `data/corpus_500k`. Answers, in the
ticket's order:

| # | Asked | Answer |
|---|---|---|
| 1 | Install works, versions pinned, on a machine we have | Yes. 113 packages, 405 MB, isolated venv, laptop. terratorch **1.2.10**, diffusers **0.39.0** (§4) |
| 2 | `FULL_MODEL_REGISTRY.build(..., pretrained=True)` resolves; where the weights come from; is a login needed | Yes. HF `ibm-esa-geospatial/TerraMind-1.0-Tokenizer-DEM`, Apache-2.0, **no login** (§3, §5) |
| 3 | `model.encoder(x)` on 224×224 returns `[B, 768, 14, 14]` | **Confirmed** — with the resampling caveat in §6 |
| 4 | `quant_proj` gives the 5-d latent; `return_intermediates=True` gives 12 layers | **Both confirmed** (§5.2, §5.3) |
| 5 | Constants read 670.665 / 951.272 from the installed source | **Confirmed**, read out of the installed package at run time (§5.4) |

## 3. The account question, settled before anything was downloaded

#118 flags this as the fork that decides who does the work: *"the model card is Apache-2.0, so no
login should be needed — confirm that, because 'needs an account' changes who does this work."*

Confirmed by probe, not by reading the card. The Hugging Face model API reports:

```
gated: false
private: false
license: apache-2.0
sha: aba0a5650a19eb82283e52ce59320c38c41f1e56
```

and an anonymous request for the weights file — no token, no cookie — 302s to the CDN and
returns **200** with `x-linked-size: 1145511376`. There is no consent interstitial and no
application step.

**This is why #118 is not shaped like [#127](https://github.com/JStoschek/DEMAE/issues/127).**
That ticket's human step was irreducible because Earthdata's URS OAuth flow requires a person to
create an account and accept terms. Nothing here does. The map's "downloads and any account step
are the human's" convention is about *account* steps and *bulk* fetches; with the account
question answered no, what remained was 1.55 GB of ordinary public artifacts, fetched with the
user's explicit go-ahead.

## 4. The install, and the `diffusers` pin that does not exist

Installed into a **throwaway venv**, not DEMAE's own, and this should stay that way. `terratorch`
resolves to 113 packages / 405 MB of wheels, including `lightning`, `lightly==1.5.22`, `torchgeo`,
`geopandas` and `opencv-python-headless` — none of which the corpus ETL has any reason to carry —
and its `numpy>=2.2` / `timm>=1.0.15` floors sit close enough to DEMAE's own to make a shared
lockfile a standing hazard. The five packages that matter, as resolved:

| package | resolved |
|---|---|
| terratorch | 1.2.10 |
| diffusers | **0.39.0** |
| torch | 2.13.0 |
| timm | 1.0.28 |
| numpy | 2.5.1 |

The full 113-package resolve is committed as
[`p118-terramind-requirements.txt`](p118-terramind-requirements.txt), so the environment half of
this run is reproducible rather than merely reported. It is a **record, not a project
dependency** — nothing in DEMAE's `pyproject.toml` should ever gain a terratorch line.

**Plan §8's risk row names a `diffusers==0.30.0` pin. There is no such pin.** Both `1.2.5` and
`1.2.10` declare a bare `diffusers` in `requires-dist`. The string `0.30.0` occurs in exactly one
place — the body of a warning raised when the VQVAE import fails, in
`terratorch/models/backbones/terramind/tokenizer/tokenizer_register.py`:

> `"Cannot import VQVAE/DiVAE from terramind. \nMake sure to install `pip install diffusers==0.30.0`."`

That is remediation advice printed on failure, not a constraint the resolver enforces. Since the
import did not fail on 0.39.0, the advice never fired. **The risk row should be rewritten**: the
hazard is not "you must pin 0.30.0", it is "terratorch pins nothing, so a future diffusers can
break the tokenizer import with no lockfile signal" — which is an argument for pinning
`diffusers` in whatever environment file the benchmark ends up with, at whatever version it is
then verified on.

## 5. What the model actually hands back

One forward pass, `torch.no_grad()`, CPU, on tile
`Copernicus_DSM_COG_10_N00_00_E030_00_DEM_1344_448` (§9).

### 5.1 The encoder

```
model.encoder(x)  →  [1, 768, 14, 14]   float32
```

Non-degenerate: mean 0.0723, std 2.6496, range [−40.71, 27.95]. 90,172,416 parameters — ViT-B
scale, as §2.2 assumes for the head-to-head against DEMAE's ViT-B.

Note the model card documents `model.encode(x)`, which returns `(_, _, tokens)` and runs the
quantizer. `model.encoder` is the sub-module beneath it, and it is the one the memo's §3.4 wants:
patch-level *features*, before quantization discards them.

The memo's snippet calls `model.prepare_input(x)` first and annotates it "no-op for DEM". **The
annotation is correct** — this model has `undo_std = False` and `cls_emb = None`, so
`prepare_input` returns its argument unchanged, verified by `torch.equal`. The probe calls it
anyway, so it drives the documented path rather than a paraphrase of it.

### 5.2 `quant_proj` and the 5-d latent (A9)

```
model.quant_proj  =  Conv2d(768, 5, kernel_size=(1, 1), stride=(1, 1))
model.quant_proj(feats)  →  [1, 5, 14, 14]
```

Confirmed, and consistent with the model card's "FSQ with five dimensions and a codebook size of
15,360": the tokenizer's own training args give `codebook_size="8-8-8-6-5", latent_dim=5`, and
8·8·8·6·5 = 15,360. A9's ablation row is a 1×1 convolution away from the features A5 uses — the
two arms share everything up to that layer, which makes them cheap to run together.

### 5.3 Intermediates (Task 2)

```
model.encoder(x, return_intermediates=True)  →  list of 12 tensors, each [1, 768, 14, 14]
```

**12, one per transformer block**, all the same shape, and the last one is bit-identical to the
plain `encoder(x)` output. Worth knowing for Task 2's layer sweep: there is no separate embedding
or post-norm entry in the list, so "layer 12" and "the encoder output" are the same tensor, and a
sweep over `[1..12]` is a sweep over blocks with no off-by-one.

### 5.4 The normalization constants, from the installed package

```python
from terratorch.models.backbones.terramind.model.terramind_register import (
    v1_pretraining_mean, v1_pretraining_std,
)
v1_pretraining_mean["tok_dem@224"]  →  [670.665]
v1_pretraining_std["tok_dem@224"]   →  [951.272]
```

**The memo was right.** The probe reads these at run time and refuses to continue if they have
moved, so a future terratorch that retunes them fails loudly rather than quietly rescaling every
arm's input.

## 6. The 14×14 grid holds — by interpolation

This is the one answer that needs a caveat rather than a yes. The *fact* is not new — the
selection memo's §3.4 already records "fixed 2D sin-cos positional embeddings interpolated at
runtime" and "256 px crops". What is new is the consequence for §2.2, which nobody had drawn.

`model.encoder.pos_emb` is **`[1, 768, 16, 16]`**. The tokenizer was trained at 256 px:
its recorded training args say `input_size_min=256, input_size_max=256, input_size=256`, and the
run name is `DEM-ViTB-UNetP4_16k_224-264`. At patch size 16 that is a **16×16 native token grid**.

`ViTEncoder.forward` handles other sizes by bicubically interpolating the position embedding to
whatever grid the input implies:

```python
x_pos_emb = F.interpolate(self.pos_emb, size=(N_H, N_W), mode="bicubic", align_corners=False)
```

So a 224 px input gives 14×14 — measured, not inferred — and feeding the same tile upsampled to
256 px gives `[1, 768, 16, 16]`, confirming the grid follows the input rather than being fixed.

**What this does and does not do to §2.2.** The co-registration premise survives intact: at 224 px
TerraMind's 14×14 grid is DEMAE's 14×14 grid, patch for patch, same 16 px stride, so every
patch-level comparison in §2.2 is well-defined. What it adds is that **TerraMind is running 12.5%
off its native resolution while DEMAE is running at its own**, and the position embedding is
resampled to make that work. That is a small, one-sided handicap in a head-to-head, and the
writeup should say so rather than have a reviewer find it. Two ways to defuse it, neither owned by
this ticket: run TerraMind at 256 px on a 256 px crop and accept that the grids no longer align,
or state the handicap and keep 224. The plan's out-of-scope "secondary 10 m TerraMind-native
regime" is a *different* question (GSD, not grid) and does not cover this.

## 7. The normalization trap, now enforced rather than annotated

The selection memo's §3.4 snippet already carries the warning, as a code comment: *"global
z-score, NOT per-tile; NaNs filled first"*. What this run adds is that **the failure mode is
completely silent**, which is what makes a comment an insufficient guard.

- **TerraMind** wants `(metres − 670.665) / 951.272` — one global affine, fixed constants.
  Absolute elevation is signal to it.
- **DEMAE's corpus** is `per_tile_standardize` (`corpus_meta.json`). Every tile leaves the loader
  at mean 0 / std 1 and its absolute elevation is **gone**.

Both produce a `[1, 1, 224, 224]` float tensor. The encoder accepts both, returns
`[1, 768, 14, 14]` for both, raises nothing and warns about nothing. One of them is features
computed from an input distribution the model has never seen.

The probe therefore reads **raw int16 metres straight out of the shards** and applies TerraMind's
affine itself, and `tests/test_p118_terramind_probe.py::test_offset_survives` fails if the two
normalizations are ever wired together: two tiles differing only by a 1589 m offset must stay
1589/951.272 apart afterwards. Whatever `benchmark/` eventually looks like, the TerraMind arms
cannot reuse the pretraining loader's normalized output — that is a §9.4 constraint on the
frozen-feature contract, and it is a genuine one.

For the record, the probe tile lands at mean **+1.176**, std 0.257 in TerraMind units — a
comfortable place in the input range, not an extrapolation.

## 8. What the 1.15 GB is made of

The selection memo estimated "~90M encoder / ~196M decoder". Counted, from `named_children`:

| module | parameters | share |
|---|---:|---:|
| `encoder` | 90,172,416 | 31.5% |
| `quant_proj` | 3,845 | 0.001% |
| `quantize` | 0 | — |
| `decoder` | 196,140,560 | 68.5% |
| **total** | **286,316,821** | |

`FULL_MODEL_REGISTRY.build` loads the whole tokenizer with `strict=True`, so the decoder comes
down whether or not anything decodes. **A5 and A9 need only `encoder` + `quant_proj`: 90.2 M
parameters, 361 MB in fp32.** If the frozen-feature contract §9.4 wants ends up shipping or
caching anything, that is the number, and it is under a third of the checkpoint. The decoder is
the diffusion half — needed to reconstruct DEMs from tokens, which is not a question any in-scope
arm asks.

## 9. The tile, and how to get the same one

`Copernicus_DSM_COG_10_N00_00_E030_00_DEM_1344_448` — the **Rwenzori Mountains**, on the
Uganda/DRC border at 30.156°E, 0.596°N. Test split, stratum 12, mean elevation 1789.1 m,
per-tile std 244.1 m, local relief 1165.5 m, `land_frac = 1.0`.

Chosen by rule rather than by seed, so it is re-derivable by anyone holding the manifest: **the
lowest `tile_id` among test-split tiles with `land_frac == 1.0` whose `local_relief` is in the top
decile of the test split.** Rugged because that is the regime void filling lives in; all-land so
the ocean mask is not doing half the work; test split so nothing here touches training geography.
`select_tile_id` implements the rule and a test asserts it still returns this tile.

Also confirmed in passing: the shards hold **raw int16 metres**, not standardized values — the
decoded tile's mean and std match the manifest's to within int16 rounding. §7's whole argument
rests on that, so it is pinned by a test rather than assumed.

## 10. Cost, and where it runs

| | |
|---|---|
| Cold run (incl. 1.15 GB weights fetch) | **35.6 s** |
| Warm run, CPU | ~16 s |
| Warm run, MPS | ~16 s |
| Peak footprint | one 286 M-parameter model in fp32, ~1.1 GB RAM |

**MPS works**, and agrees with CPU. Both runs are committed —
[`p118-terramind-probe.json`](p118-terramind-probe.json) (`"device": "cpu"`) and
[`p118-terramind-probe-mps.json`](p118-terramind-probe-mps.json) (`"device": "mps"`) — so the
claim is checkable rather than reported. Every shape field is identical; feature statistics match
to ~1e-5 (`min` −40.710289 vs −40.710278), which is float32 accumulation order, not a device bug.
No `torch.compile`, no autocast, no fallback warnings. The only warnings emitted are
`torch.jit.script` deprecations from a transitive dependency, unrelated to TerraMind.

The `autocast("cuda", enabled=False)` guard inside `ViTEncoder`'s `post_mlp` — which the DEM
tokenizer does use, since its args set `post_mlp=True` — is a no-op on a machine with no CUDA and
did not need working around.

So: #118's premise that this is "one forward pass, not training" is right, and the whole thing is
cheaper than the download. Nothing here needs a rented box.

## 11. What this changes

- **§8's risk row** should drop the `diffusers==0.30.0` pin claim and replace it with the real
  hazard: terratorch pins `diffusers` at nothing (§4).
- **§8's risk row** should also drop any implication that this path is unverified or needs an
  account — it is verified, on the versions above, and it is public (§3).
- **§2.2** should carry the resampling caveat: the grids co-register, and TerraMind reaches 14×14
  from a native 16×16 (§6).
- **§3.4 / §9.4** gain a hard constraint: the TerraMind arms cannot consume the pretraining
  loader's per-tile-standardized output (§7).
- **A5 and A9** are confirmed runnable and share everything up to a 1×1 convolution (§5.2).

## 12. Sources

- Model card and weights: <https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-Tokenizer-DEM>
  (Apache-2.0, `TerraMind_Tokenizer_DEM.pt`, 1,145,511,376 bytes, sha `aba0a56`)
- terratorch source, `main` as of 2026-08-07: `terratorch/models/backbones/terramind/` —
  `tokenizer/tokenizer_register.py` (registry, the diffusers warning string, `strict=True` load),
  `tokenizer/vqvae.py` (`encode` = `quant_proj(encoder(x))`), `tokenizer/models/vit_models.py`
  (`ViTEncoder.forward`, pos-embed interpolation, per-block intermediates),
  `model/terramind_register.py` (`v1_pretraining_mean/std`), `model/modality_info.py`
  (`tok_dem@224`)
- TerraMind paper: Jakubik et al., ICCV 2025, <https://arxiv.org/abs/2504.11171>
- This run: [`p118-terramind-probe.json`](p118-terramind-probe.json) (CPU),
  [`p118-terramind-probe-mps.json`](p118-terramind-probe-mps.json) (MPS),
  [`p118-terramind-requirements.txt`](p118-terramind-requirements.txt) (the resolved environment)
