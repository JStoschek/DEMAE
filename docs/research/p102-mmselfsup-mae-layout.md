# P102 — the mmselfsup MAE ViT-B/16 checkpoint: what is in the file, and what layout every tensor is in

Ticket [#102](https://github.com/JStoschek/DEMAE/issues/102) · map [#76](https://github.com/JStoschek/DEMAE/issues/76) · plan §8 (as amended by [#103](https://github.com/JStoschek/DEMAE/issues/103)) · licence evidence [P101](p101-mae-checkpoint-licence.md) · consumed by [`pretrain.init`](../../src/pretrain/init.py)

> **What this document is for.** #103 set the policy that #102 works under: *layout conventions are read from the source implementation and written down as checked claims with a link — never inferred from shapes.* This is that record. Every row of §3 is a claim a reviewer can check against a URL, and §4 is the one place the two implementations **disagree**, which name-and-shape coverage would never have surfaced.
>
> Two kinds of statement appear below and are kept apart: **read from the file** (a fact about the 485 MB artifact, obtained by parsing it) and **read from the source** (a fact about the code that wrote it, obtained from a pinned URL). Nothing here is inferred from a tensor's shape.

## 1. The file

`mae_vit-base-p16_8xb512-amp-coslr-1600e_in1k`, chosen by plan §8's 2026-07-29 amendment.

```
https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-1600e_in1k/mae_vit-base-p16_8xb512-fp16-coslr-1600e_in1k_20220825-f7569ca2.pth
```

| Property | Value | How it was obtained |
|---|---|---|
| `Content-Length` | 484,833,387 B (462.4 MiB) | `curl -I`, 2026-07-29 |
| `Content-MD5` | `kezANFptbs7+nFkEGEHxgA==` → md5 `91ecc0345a6d6ecefe9c59041841f180` | the same response header; matches the file after download |
| `Last-Modified` | Mon, 29 Aug 2022 03:55:17 GMT | ditto |
| Serialization | **legacy** `torch.save` (pickle protocol 2, *not* the zipfile format) | first bytes are `\x80\x02\x8a\n…`, i.e. a raw pickle stream |
| Top-level keys | `meta`, `state_dict`, `message_hub`, `param_schedulers` | parsed |
| `meta['epoch']` / `['iter']` | 1600 / 500,800 | parsed — this **is** the 1600-epoch schedule, not a mislabelled sibling |
| `meta['experiment_name']` | `mae_vit-base-p16_8xb512-fp16-coslr-1600e_in1k_20220815_103458` | parsed |

**No `optimizer` entry** — the released file carries no optimizer state, so nothing about it invites a "resume from here" reading. `message_hub` and `param_schedulers` are the training-log and LR-schedule history; both are ignored.

### 1.1 What `state_dict` actually contains — 256 entries, three groups

Read from the file, not inferred:

| Prefix | Entries | Parameters | What it is | What #102 does with it |
|---|---|---|---|---|
| `backbone.` | 150 | **85,798,656** | the MAE ViT-B/16 encoder | **loaded**, every one asserted |
| `neck.` | 104 | **26,109,184** | the 8-block, 512-wide MAE decoder | **discarded**, explicitly |
| `data_preprocessor.` | 2 | 6 | ImageNet RGB `mean` / `std`, shape `(3, 1, 1)` | **discarded**, explicitly |

85,798,656 + 26,109,184 + 6 = **111,907,846**, which matches the metafile's 111.9 M exactly.

This settles the inference P101 §6.1 flagged as an inference. The file does ship encoder **and** decoder, as P101 guessed from byte counts — but the byte arithmetic there (485 MB ÷ 4 ≈ 121 M floats) over-counted, because ~37 MB of the file is `message_hub`, not weights. The conclusion stands and the reasoning is now unnecessary: the entry list was read.

The two `data_preprocessor` tensors are worth naming rather than skipping past. They are the ImageNet channel statistics `mean=[123.675, 116.28, 103.53]`, `std=[58.395, 57.12, 57.375]` (also visible verbatim in `meta['cfg']`), i.e. the pixel normalization the encoder was trained under. DEMAE normalizes **per tile** on standardized elevation (ADR-0003 §3, plan §5), so they are not merely unneeded, they are wrong for this input, and a loader that silently ignored two 3-channel constants is a loader that would also silently ignore something that mattered.

### 1.2 The exact `backbone.` key inventory

150 tensors, and the loader's remap must cover exactly these:

```
backbone.cls_token                              (1, 1, 768)
backbone.pos_embed                              (1, 197, 768)     <- see §4
backbone.patch_embed.projection.{weight,bias}   (768, 3, 16, 16) / (768,)
backbone.layers.{0..11}.ln1.{weight,bias}       (768,)
backbone.layers.{0..11}.attn.qkv.{weight,bias}  (2304, 768) / (2304,)
backbone.layers.{0..11}.attn.proj.{weight,bias} (768, 768) / (768,)
backbone.layers.{0..11}.ln2.{weight,bias}       (768,)
backbone.layers.{0..11}.ffn.layers.0.0.{weight,bias}  (3072, 768) / (3072,)
backbone.layers.{0..11}.ffn.layers.1.{weight,bias}    (768, 3072) / (768,)
backbone.ln1.{weight,bias}                      (768,)
```

2 + 2 + 12 × 12 + 2 = 150. ✅

## 2. The model that wrote it

The checkpoint's own `meta['cfg']`, read from the file:

```python
model = dict(
    type='MAE',
    data_preprocessor=dict(mean=[123.675, 116.28, 103.53],
                           std=[58.395, 57.12, 57.375], bgr_to_rgb=True),
    backbone=dict(type='MAEViT', arch='b', patch_size=16, mask_ratio=0.75),
    neck=dict(type='MAEPretrainDecoder', patch_size=16, in_chans=3, embed_dim=768,
              decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, mlp_ratio=4.0),
    head=dict(type='MAEPretrainHead', norm_pix=True, patch_size=16,
              loss=dict(type='MAEReconstructionLoss')))
```

So the encoder is `mmselfsup.models.backbones.MAEViT(arch='b', patch_size=16)`, which subclasses `mmcls.models.VisionTransformer`, which builds `mmcls` `TransformerEncoderLayer`s over `mmcv`'s `FFN` and `PatchEmbed`. Four repositories, and the sources pinned below are the ones that define each name.

Note in passing: `norm_pix=True` on their head. Their pretraining objective is per-patch-normalized MSE, the same objective plan §4 runs (DEMAE's `pretrain.loss`, with a raised variance floor). One less thing that differs between the arm and its init.

**Pinned sources.** `mmselfsup` is archived at v1.0.0; `mmclassification` was renamed `mmpretrain`, so the class definitions are cited at `mmclassification` v1.0.0rc5 — the last tag under the old name and the one `mmselfsup` 1.x builds against. All retrieved 2026-07-29.

- `mmselfsup/models/backbones/mae_vit.py` — https://github.com/open-mmlab/mmselfsup/blob/v1.0.0/mmselfsup/models/backbones/mae_vit.py
- `mmselfsup/models/utils/position_embedding.py` — https://github.com/open-mmlab/mmselfsup/blob/v1.0.0/mmselfsup/models/utils/position_embedding.py
- `mmcls/models/backbones/vision_transformer.py` — https://github.com/open-mmlab/mmclassification/blob/v1.0.0rc5/mmcls/models/backbones/vision_transformer.py
- `mmcls/models/utils/attention.py` — https://github.com/open-mmlab/mmclassification/blob/v1.0.0rc5/mmcls/models/utils/attention.py
- `mmcv/cnn/bricks/transformer.py` (`FFN`, `PatchEmbed`) — https://github.com/open-mmlab/mmcv/blob/v2.0.0/mmcv/cnn/bricks/transformer.py

DEMAE's side is `timm` 1.0.28's `PatchEmbed` and `vision_transformer.Block`, plus `pretrain.model`.

## 3. The layout claims

Each row states **their** convention, quotes or cites the line that fixes it, and says whether ours is the same. "Same" means the tensor is copied across unchanged; there is no reshaping or permutation anywhere in the remap.

| # | Convention | Their code, verbatim | Ours | Verdict |
|---|---|---|---|---|
| L1 | **Token order is row-major over the patch grid**, so token *k* is patch (row `k // 14`, col `k % 14`) | `mmcv` `PatchEmbed.forward`: `x = self.projection(x)`, then `x = x.flatten(2).transpose(1, 2)` | timm `PatchEmbed.forward` ends in the same `flatten(2).transpose(1, 2)`; `pretrain.model.patchify` cuts row-major to match | **same** |
| L2 | **Patch embed is a `Conv2d`** with kernel `(out_ch, in_ch, kh, kw)` = `(768, 3, 16, 16)`, stride = patch | `mmcv` `PatchEmbed.__init__`: `self.projection = build_conv_layer(..., kernel_size=kernel_size, stride=stride, ...)` | timm's `patch_embed.proj`, same conv, `in_chans=1` → `(768, 1, 16, 16)` | **same up to §5's channel collapse** |
| L3 | **`attn.qkv` is one fused `Linear(D, 3D)`, split in the order q, k, v, untransposed, heads-within-each** | `mmcls` `MultiheadAttention.forward`: `qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dims).permute(2, 0, 3, 1, 4)` then `q, k, v = qkv[0], qkv[1], qkv[2]` | timm `Attention.forward`: `qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)` then `q, k, v = qkv.unbind(0)` — character-for-character the same reshape | **same** |
| L4 | `qkv` **has a bias** | `mmcls` `MultiheadAttention.__init__` defaults `qkv_bias=True`, and `VisionTransformer` passes it through | `pretrain.model` passes `qkv_bias=True` to timm's `Block` (which defaults it *off*, and says so in a comment) | **same** |
| L5 | Attention scale is `head_dim ** -0.5` | `mmcls`: `self.scale = qk_scale or self.head_dims**-0.5` | timm: `self.scale = self.head_dim ** -0.5` | **same** |
| L6 | **Blocks are pre-norm**: `x = x + attn(ln1(x))`, then `x = x + ffn(ln2(x))` | `mmcls` `TransformerEncoderLayer.forward`: `x = x + self.attn(self.norm1(x))` / `x = self.ffn(self.norm2(x), identity=x)`, and `FFN.forward` returns `identity + out` | timm `Block.forward` is `x + attn(norm1(x))` then `x + mlp(norm2(x))` | **same**; `ln1` ↔ `norm1`, `ln2` ↔ `norm2` |
| L7 | **`ffn.layers.0.0` is fc1 (D→4D) and `ffn.layers.1` is fc2 (4D→D)**, GELU between | `mmcv` `FFN.__init__` builds `Sequential(Sequential(Linear(in, ff), act, Dropout), Linear(ff, in), Dropout)`; the `act_cfg` reaching it from `TransformerEncoderLayer` is `dict(type='GELU')` | timm `Mlp.fc1` / `Mlp.fc2`, `act_layer=GELU` | **same** |
| L8 | **The final encoder norm is `backbone.ln1`** — a top-level LayerNorm, not a block's | `MAEViT` passes `final_norm=True`; `VisionTransformer.__init__` then does `self.norm1_name, norm1 = build_norm_layer(norm_cfg, self.embed_dims, postfix=1)`, and `MAEViT.forward` ends `x = self.norm1(x)` | `pretrain.model.MaskedAutoencoder.norm`, applied at the same point | **same** (the name collides confusingly with each block's `ln1`; they are different tensors and the remap keys off the full path) |
| L9 | **LayerNorm epsilon is 1e-6**, not torch's 1e-5 | `MAEViT.__init__` signature: `norm_cfg: dict = dict(type='LN', eps=1e-6)` | `pretrain.model.NORM_EPS = 1e-6` | **same** |
| L10 | **The CLS token is prepended** (position 0) and gets `pos_embed[:, :1]`; patch tokens get `pos_embed[:, 1:]` | `MAEViT.forward`: `x = x + self.pos_embed[:, 1:, :]` … `cls_token = self.cls_token + self.pos_embed[:, :1, :]` … `x = torch.cat((cls_tokens, x), dim=1)` | `pretrain.model.forward_encoder`, line for line the same order | **same** |
| L11 | **`backbone.pos_embed` is fixed sin-cos, not learned** — a `Parameter` with `requires_grad = False`, overwritten in `init_weights` | `MAEViT.__init__`: `self.pos_embed.requires_grad = False`; `init_weights`: `pos_embed = build_2d_sincos_position_embedding(...)`, `self.pos_embed.data.copy_(...)` | ours is a non-persistent **buffer** computed from the config — same values, and nothing to remap | **same in kind** — see §4 for the ordering, which is *not* the same |
| L12 | Their encoder masks after adding the PE and before the blocks, at ratio 0.75, by argsort-of-noise | `MAEViT.random_masking` — `noise = torch.rand(N, L)`, `ids_shuffle = torch.argsort(noise, dim=1)`, keep the first `len_keep` | `pretrain.model.random_masking`, the same mechanism | **same** (not load-bearing for a weight copy; recorded because it means the weights were trained under our masking regime, not a different one) |

**L3 is the row this table exists for.** "A `(2304, 768)` tensor fits a `(2304, 768)` slot" is not evidence; "their `qkv` is fused q,k,v, untransposed, per `MultiheadAttention.forward`" is. The two reshape expressions are identical, so the copy is a copy.

## 4. Where they differ: the positional embedding is the **transpose** of MAE's

This is the finding. It is exactly the failure class #103 predicted — every name populates, every shape matches, every assertion passes, and the spatial convention is wrong — and it is *real* in this checkpoint rather than hypothetical.

### 4.1 The two constructions

`mmselfsup/models/utils/position_embedding.py`, verbatim:

```python
h, w = patches_resolution
grid_w = torch.arange(w, dtype=torch.float32)
grid_h = torch.arange(h, dtype=torch.float32)
grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
...
out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w),
                     torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]
```

`facebookresearch/mae`'s `util/pos_embed.py`, which `pretrain.model.sincos2d_pos_embed` transcribes:

```python
grid = np.meshgrid(grid_w, grid_h)   # w goes first
...
emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
emb = np.concatenate([emb_h, emb_w], axis=1)
```

Both look like they encode width first. Neither variable name can be trusted, and the reason is a different one on each side:

- **MAE** calls `np.meshgrid`, whose **default is `'xy'` indexing**. With inputs `(grid_w, grid_h)` that makes `grid[0][i, j] == j` — the *column*. So MAE's first half, misleadingly named `emb_h`, encodes the **column**, and its second half the **row**. Ordering: **column-then-row**. (`pretrain.model.sincos2d_pos_embed`'s docstring already carries this argument, and `test_the_positional_encoding_matches_maes_own_construction` pins it.)
- **mmselfsup** calls `torch.meshgrid`, whose **default is `'ij'` indexing**. With the same inputs that makes the first output vary along axis 0, so after the row-major `.flatten()` its values are the **row** index. So mmselfsup's first half, named `out_w`, encodes the **row**, and its second half the **column**. Ordering: **row-then-column**.

Two libraries, two opposite meshgrid defaults, two sets of variable names that each describe the other library's behaviour.

### 4.2 Verified numerically, not argued

Running both functions verbatim at `grid=14`, `dim=768` (`torch` 2.13, `numpy` 2.x):

```
mmselfsup == MAE                     ->  False
mmselfsup == transpose(MAE)          ->  True
mmselfsup == our sincos2d_pos_embed  ->  False
mmselfsup == transpose(ours)         ->  True
sorted values identical              ->  True
```

The frequency ladder (`omega = 1 / 10000 ** (arange(dim//4) / (dim//4))`) is identical on both sides — MAE's `omega /= embed_dim / 2.` over `embed_dim // 2 = 384` is the same `/ 192` as mmselfsup's `/ pos_dim`. Only the axis assignment differs, so the two embeddings are a permutation of each other and the difference is a pure transpose of the patch grid. `test_mmselfsups_positional_embedding_is_the_transpose_of_maes` is the check.

### 4.3 Why it matters, and what #102 does about it

A PE is added to the patch embeddings; it is not a weight, so no permutation of the loaded tensors can correct it. Using MAE's ordering with these weights means the encoder reads every tile as its own transpose: `PE_ours(row=i, col=j) == PE_mmselfsup(row=j, col=i)`.

There is a real argument that this would have been *harmless*. Plan §6 augments with the full dihedral group, which contains the transpose, so the training distribution is exactly invariant under it, and terrain has no canonical orientation. **That argument is not what the code relies on.** It is a coincidence of the augmentation policy, it would weaken the moment someone ablates `dihedral`, and #103's whole point is that this arm must not rest on a convention nobody checked.

So `pretrain.model.sincos2d_pos_embed` takes an `axis_order`, `Architecture.pos_encoding` gains the value `"sincos2d_rowfirst"`, and **the loader asserts the model's `pos_embed` buffer equals the checkpoint's `backbone.pos_embed` tensor** before it copies a single weight. That assertion is worth more than the config knob it validates:

- it is an **oracle from the checkpoint's own contents** — the one thing in the file that we can independently reconstruct and compare, needing no `mmcv` and no reference model. #102 was written expecting no such oracle to exist (its two candidates were both declined); this one costs nothing.
- it catches *any* PE convention mismatch, including ones this document has not thought of, because it compares 151,296 floats rather than a claim.
- it makes the ordering reproducible **from the config**, which matters more than it looks: our `pos_embed` is a non-persistent buffer, so a resumed R4 rebuilds it from `Architecture`. Had the loader instead patched the buffer after construction, every §11 resume on a fresh vast.ai box would have silently rebuilt the MAE-ordered one and continued training under a spatial convention the first leg had not used.

R4 therefore runs with the row-first ordering and R5 with MAE's. That is a relabelling of which half of the width carries which axis — the same set of functions with the two axes swapped — so it is not a confound on the R4-vs-R5 comparison, and the transpose-invariance of the augmented corpus is what makes that sentence true rather than merely plausible.

## 5. The channel collapse

Plan §8: collapse the 3-channel patch-embed kernel to 1 by **summing**, timm's `adapt_input_conv` behaviour, because sum preserves activation scale where mean shrinks it 3×.

The claim underneath "preserves activation scale" is exact, not statistical. For a conv whose input is the same in every channel — which is what a grayscale image replicated to RGB is —

$$\sum_c W_c * x \;=\; \Bigl(\sum_c W_c\Bigr) * x$$

so the summed 1-channel kernel applied to `x` gives **bit-comparable** output to the original 3-channel kernel applied to `x` replicated 3×. The mean gives exactly ⅓ of it. `test_the_summed_kernel_reproduces_the_three_channel_convolution_exactly` and `test_the_mean_kernel_shrinks_the_activation_by_exactly_three` are the checks, and they are the reason "sum, not mean" is a verified property here rather than a citation.

The elevation tiles this model will actually see are of course not replicated grayscale photographs; the equality above is about the operator, which is the thing the plan's claim is about.

## 6. The behavioural check, and the negative result it produced

#102 asked for "a dependency-free behavioural check that a scrambled-but-name-complete load fails", suggested two statistics for it, and expected the result to catch gross scrambling while missing a permutation *within* attention heads.

**That expectation was measured against the real checkpoint and is too optimistic.** The measurement is below, because a check whose power nobody measured is exactly the kind of reassurance this ticket exists to refuse.

### 6.1 Eight scrambles, measured

Every row is the same 12-parameter ViT-B, correctly loaded, then perturbed in a way that **preserves every name and every shape** — so all of §3's assertions and the both-directions coverage check pass on all of them. `crop` is the two-crops-of-one-field margin, `noise` the fields-versus-white-noise margin, `gain` the largest multiplier any single block applies to the residual stream's norm. Ceilings and floors are `pretrain.init`'s constants.

| weights | crop | noise | gain | verdict |
|---|---|---|---|---|
| **correct load** | +0.515 | +1.92 | 2.15 | pass |
| random init (R5's own starting point) | +0.423 | +1.53 | 1.23 | pass |
| block order reversed | +0.578 | +1.96 | **3.81** | **FAIL** |
| `ln1`/`ln2` swapped in every block | +0.427 | +1.97 | **2.81** | **FAIL** |
| final norm ⇄ block 11's `ln2` | +0.411 | +1.79 | 2.16 | pass |
| `qkv` order rotated to k, v, q | +0.587 | +1.98 | 1.73 | pass |
| `attn.proj` transposed | +0.576 | +1.97 | 1.73 | pass |
| `mlp.fc1` rows permuted | +0.230 | +1.97 | 1.86 | pass |
| patch kernel spatially transposed | +0.584 | +1.92 | 2.16 | pass |
| permutation within attention heads | +0.537 | +1.97 | 1.50 | pass |

**The two statistics #102 suggested have no discriminating power here.** Six of the eight scrambles score a *higher* crop margin than the correct load, and all eight score a higher noise margin. The reasons are mundane once seen: two crops offset by 32 of 224 px overlap by 86%, so any smooth function of the input maps them close together; and any network at all separates a smooth field from white noise, because the inputs differ enormously in spectrum before a single weight is applied. Four further statistics were tried and discarded — top-1 crop retrieval, agreement between the masked and unmasked passes, feature effective rank, and raw uncentred cosines — none separated correct from scrambled either.

What the two suggested statistics *do* detect is an encoder that cannot tell one tile from another: collapsed, saturated, or NaN. That is the shape a catastrophically wrong load takes and it is worth a floor, so both are kept and gated — just not credited with more than they do.

**A random init passes too, and that is deliberate.** The table's second row is R5's own starting point, and it clears both floors. This check is not a warm-start *detector*: a from-scratch encoder is a legitimate encoder, not a broken one, and asking "did these weights come out of a file?" is the coverage assertion's job, not a feature statistic's. Worth stating because #102's headline worry is "continual pretraining mysteriously performed like scratch" — and the thing that rules that out is 149 asserted parameter names, not this.

**The third statistic is the one with teeth**, and it was found by measurement rather than by design. The correct checkpoint's per-block amplification profile is a near-constant fingerprint — 1.46 at block 1, ~1.0 for ten blocks, 2.15 at block 12 — landing between **2.138 and 2.165** across six seeds, three batch sizes, three crop shifts, three field roughnesses and even white-noise input. A 1.3% spread. The two scrambles that a wrong key map would actually produce, both same-shape key permutations, sit at 3.81 and 2.81, and a ceiling of 2.5 separates them from the correct load with 15% headroom — many multiples of the observed variation.

Caveat, recorded: 2.5 is calibrated on the **1600e** file only. The other three schedules were not measured (a second 466 MB download to bound a smoke test), so substituting one may need it re-measured. The check prints the value beside the ceiling, and it runs in #86's pre-flight, so the cost of it being wrong is a human reading a number.

### 6.2 Why the gap is acceptable, and where the real defence is

Six of the eight scrambles are undetected. What makes that acceptable is not the check — it is that **those six are unreachable by this loader.**

`pretrain.init` performs no permutation, no transposition and no reshape on any tensor. The mapping copies whole tensors from one name to another, and the single transformation anywhere in the module is the patch-embed channel sum. So a wrong `qkv` order, a transposed matrix or a within-head permutation would require code that does not exist; the assertion those conventions need is that **no** transformation is required, which is §3's claim table — read from the source, linked, and reviewable line by line. That is #103's policy, and it is doing the load-bearing work.

The failures that *are* reachable from a bug here are three, and each has a check that catches it:

| Reachable bug | What catches it |
|---|---|
| A tensor mapped to a same-shaped wrong slot (`ln1`→`norm2`, block *i*→*j*) | `test_every_mapped_key_is_the_one_p102_documents` pins all 149 pairs against §3; the amplification ceiling catches two such swaps behaviourally |
| A key missing, misspelled, or left unmapped | the both-directions coverage assertion — an unfilled destination and an unmapped source each raise |
| The positional axis order (§4) | the `pos_embed` equivalence against the checkpoint's own tensor, before any weight is copied |

So the honest summary of the acceptance criterion "a dependency-free behavioural check that a scrambled-but-name-complete load fails": **partially met**. It fails 2 of 8 arbitrary scrambles, and 2 of the 3 scrambles this loader can produce. The remaining coverage is a claim table and a 149-pair unit test rather than a behavioural test, and that is a weaker guarantee than #102 hoped for — stated here rather than smoothed over.

Two further limits, for completeness:

- **Those two catches are only tested against the real file**, behind an opt-in `DEMAE_MAE_CHECKPOINT` skip. That is not laziness: a synthetic checkpoint holds random values, so its "correct" and "scrambled" loads are both random and no statistic can separate them. Nothing in CI covers the criterion's literal wording; what CI covers is that the floors fire on a blind encoder and the ceiling on an exploding one.
- **The check reads a model, not a load.** It cannot distinguish "these weights came from the file" from "these weights are fine"; that is the coverage assertion's job.

### 6.3 What was checked and deliberately not built

- **Exact equivalence against `mmpretrain`'s own model** — declined in #103: `mmcv`'s compiled ops and version pins cost more than the bug they guard. Unchanged by anything here. §4.3's `pos_embed` comparison recovers a useful fraction of what it would have bought, for free.
- **Reconstructing a masked photo with the checkpoint's own decoder** — declined in #103, shape-blocked: their `decoder_pred` is 768-wide per patch (16×16×**3**) against our 256, and `Architecture` validates `in_chans == 1` deliberately. Unchanged.
- **Training a probe head on the frozen encoder** to measure whether it retains information a scramble destroys — the one candidate not tried. It would very likely work, and it costs training steps, a fitted head, and a stochastic threshold. Recorded as the move available if the gap in §6.2 ever needs closing.

## Sources

All retrieved 2026-07-29.

- The checkpoint itself (headers, and `meta`/`state_dict` parsed from the file): https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-1600e_in1k/mae_vit-base-p16_8xb512-fp16-coslr-1600e_in1k_20220825-f7569ca2.pth
- mmselfsup v1.0.0 `MAEViT`: https://github.com/open-mmlab/mmselfsup/blob/v1.0.0/mmselfsup/models/backbones/mae_vit.py
- mmselfsup v1.0.0 `build_2d_sincos_position_embedding`: https://github.com/open-mmlab/mmselfsup/blob/v1.0.0/mmselfsup/models/utils/position_embedding.py
- mmclassification v1.0.0rc5 `VisionTransformer` / `TransformerEncoderLayer`: https://github.com/open-mmlab/mmclassification/blob/v1.0.0rc5/mmcls/models/backbones/vision_transformer.py
- mmclassification v1.0.0rc5 `MultiheadAttention`: https://github.com/open-mmlab/mmclassification/blob/v1.0.0rc5/mmcls/models/utils/attention.py
- mmcv v2.0.0 `FFN` and `PatchEmbed`: https://github.com/open-mmlab/mmcv/blob/v2.0.0/mmcv/cnn/bricks/transformer.py
- mmpretrain MAE metafile (parameter count, training resources, licence chain): https://github.com/open-mmlab/mmpretrain/blob/main/configs/mae/metafile.yml
- `facebookresearch/mae` `util/pos_embed.py` (spec, not a dependency — plan §1): https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
- timm `adapt_input_conv` (the sum-not-mean behaviour plan §8 names): https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/_manipulate.py
