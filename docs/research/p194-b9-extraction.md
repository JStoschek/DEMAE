# P194: B9's extraction path. One public file, the v1 DEM pair, a native 14x14 grid, and a final norm the tokenizer never had

**Ticket:** [#194](https://github.com/JStoschek/DEMAE/issues/194), under map
[#183](https://github.com/JStoschek/DEMAE/issues/183).
**Script:** [`scripts/p194_b9_backbone_probe.py`](../../scripts/p194_b9_backbone_probe.py), run
2026-10-15 on the laptop's MPS with the main checkout's venv (terratorch 1.2.10, torch 2.13.0).
It takes about two minutes after the 1.52 GB download.
**Record:** [`p194-b9-backbone-probe.json`](p194-b9-backbone-probe.json).
**Feeds:** [`task2-landform-plan.md`](task2-landform-plan.md) §1 (B9 row) and §2 (capacity
table). This note does not edit them; the parent session does.

## The answer in six lines

1. **Pin:** `ibm-esa-geospatial/TerraMind-1.0-base`, file `TerraMind_v1_base.pt`, revision
   `fb96c70d0a5f68dcc44030b89cbfd8ec3fb0c67a`. It is 1,518,713,958 bytes, md5
   **`690349bd1328e50b64979847d4f362e7`**, sha256 `83c3a093…b847d7ec`. The sha256 matches the
   LFS hash HF publishes. HF publishes no md5, so the md5 is computed from the file, as it was
   for the tokenizer pin. Apache-2.0, not gated.
2. **Diet:** **670.665 / 951.272**, under the name `v1_pretraining_mean/std["untok_dem@224"]`.
   That key is B9's own input modality. The tokenizer arms use the `tok_dem@224` key, which
   resolves to the same pair. Swapping in the TerraMesh loader pair costs a mean cosine of 0.9976,
   but the 1st-percentile token falls to 0.968. Swapping in the v0.1 pair costs a mean of 0.875.
3. **Grid:** native. At 224 px the DEM patch embed gives 196 tokens on a 14x14 grid, and its
   position table is a `(1, 196, 768)` buffer equal to `build_2d_sincos_posemb(14, 14, 768)` to
   6e-8. No crop and no resize happen. ADR-0011 does not apply, so the derivation is
   `as_published`.
4. **Layers:** 12 blocks. There are no CLS or register tokens. There is a final `encoder_norm`
   (LayerNorm with a learned weight and zero bias). terratorch's returned list **overwrites entry
   12 with `encoder_norm(block 12)`**, so the raw block-12 output is never returned. Under
   ADR-0012, B9 therefore looks like DEMAE and not like the tokenizer: `output` is normed and
   `intermediates[11]` is not `output`. UperNet reads blocks **3, 6, 9, 12**, which are
   `intermediates[2, 5, 8, 11]`, all raw.
5. **Params:** **85,151,232**. That is `parameters()` of the backbone as built with the DEM
   modality only, which is the same rule that gives R5 its 85,254,144. B9 is 0.12% smaller than
   R5.
6. **ADR-0012:** no decision changes. Three sentences that were written with the tokenizer in
   mind need an amendment note, the same kind #156 added. The extractor needs a third forward
   path, because terratorch's own list cannot fill the contract (§5).

## 1. The checkpoint and the pin

terratorch's registry maps `terramind_v1_base` to this repo and file
(`terramind_register.py`, `pretrained_weights["terramind_v1_base"]`). Its loader calls
`hf_hub_download(repo_id, filename)` **with no revision**, so `pretrained=True` follows `main`.
The pin has to be held by our code and not by terratorch's.

| Field | Value | Source |
|---|---|---|
| Repo | `ibm-esa-geospatial/TerraMind-1.0-base` | terratorch 1.2.10 `terramind_register.py` |
| File | `TerraMind_v1_base.pt` | same |
| Revision (HEAD, 2025-11-03) | `fb96c70d0a5f68dcc44030b89cbfd8ec3fb0c67a` | HF API `model_info` |
| Bytes | 1,518,713,958 | HF API, matched on disk |
| sha256 | `83c3a0938067c83867a46e564443c2fa38383bf4f966d931b11cb025b847d7ec` | HF LFS metadata, matched on disk |
| md5 | `690349bd1328e50b64979847d4f362e7` | computed on disk |
| Git blob | `ab6cddda08bd687df1865665156a0b41457e02a1` | HF API |
| Licence | Apache-2.0, ungated | model card front matter |

**The weights have not changed since they were uploaded.** The repo has 22 commits. The blob
`ab6cddda…` first appears in `bb73e123` ("Added weights", 2025-02-14) under the lowercase name
`terramind_v1_base.pt`. It was renamed to `TerraMind_v1_base.pt` in `05bb410d` the same day, and
every later commit only touches the README. So any revision from `05bb410d` onward serves the same
bytes. HEAD is the one to cite.

**What the file holds.** It has 374 tensors and 465,032,448 numbers: the full any-to-any model.
That covers the decoder, the token embeddings for every modality, and the raw-input patch embeds
for S-2, S-1, RGB and DEM. B9 takes 85.3M numbers from it: the 12 encoder blocks,
`encoder_norm`, and `encoder_embeddings.untok_dem@224` (`proj.weight` 768x256, `mod_emb`,
`pos_emb`). The other numbers are loaded into CPU memory and then thrown away, so each extractor
start costs one 1.5 GB read. Re-exporting an encoder-only file would change the md5, and then the
pin would no longer be IBM's file. It is cheaper to keep pinning the published file.

**A silent failure the extractor must guard against.** terratorch's `checkpoint_filter_fn`
handles a missing or wrongly shaped key by **logging a warning and keeping the random init**.
Only after that does it call a `strict=True` load, which therefore always succeeds. At this
revision and terratorch 1.2.10, no model key is missing (`model_keys_missing_from_file: []`), and
every tensor in the built model is bit-identical to the file. The extractor should check that
equality itself and not trust the strict load, because a terratorch upgrade that renames a key
would otherwise give B9 a randomly initialized layer and raise nothing.

**Was the raw-DEM embed actually trained?** Yes, by two pieces of evidence. The model card says
the model accepts DEM as a raw input ("S2L2A, S2L1C, S1GRD, S1RTC, DEM, RGB") through
"modality-specific patch embeddings". And `untok_dem@224.proj.weight` has std 0.0297, where the
xavier-uniform init it would start from has std 0.0442. Neither is proof of how much raw DEM the
pretraining mix contained. The card and the code do not state that.

## 2. Input diet: which of #138's three pairs

| Pair | Where | Belongs to B9? |
|---|---|---|
| **670.665 / 951.272** | `v1_pretraining_mean/std["untok_dem@224"]` (and `["tok_dem@224"]`) | **Yes** |
| 651.663 / 928.168 | TerraMesh dataset loader | No |
| 435.726 / 560.326 | `v01_pretraining_mean/std["tok_dem@224"]` | No. v0.1 has no `untok_dem@224` key at all |

The v1 dictionaries in `terramind_register.py` carry an **`untok_dem@224`** entry. That is the
exact modality key the backbone builds for `modalities=["DEM"]` (`build_modality_embeddings` maps
"dem" to `untok_dem@224`). The model card points users to those same dictionaries as "the
pre-training statistics". So the name selects the pair, as ADR-0012 §7.2 wants. The backbone
itself standardizes nothing: `TerraMindViT.forward` has no `standardize` path. That only exists
in `terramind_generation.py`. The affine stays ours, which matches ADR-0012 Decision 1.

**One wording fix for the contract.** ADR-0012 §7.2 and `extract.TERRAMIND_MODALITY` name the
`tok_dem@224` key. B9's own key is `untok_dem@224`. Today both resolve to the same values, so the
value check passes either way. But the reason for a named lookup is that the *name* picks the pair
belonging to the checkpoint, so B9's spec entry should look up `untok_dem@224`. If a future
terratorch ever retunes one key and not the other, only the right name catches it.

**What a silent swap would cost**, measured on 60 tiles / 11,760 tokens at the B9 `output`,
cosine against the v1 pair:

| Wrong pair | mean | 1st pct | min |
|---|---:|---:|---:|
| TerraMesh loader (651.663 / 928.168) | 0.9976 | 0.968 | −0.32 |
| v0.1 (435.726 / 560.326) | 0.875 | 0.258 | −0.06 |

The TerraMesh pair is close on the mean, but its tail goes below #134's 0.99 bar, and some tokens
point in a different direction altogether. The v0.1 pair is a different model's input. The same
pair is used for the tokenizer arms, so B4/B5 and B9 share one diet.

## 3. Grid and position embedding

ADR-0011 existed because the DEM **tokenizer** was trained at 256 px. Its `pos_emb` is natively
16x16, and 224 px needs a 14x14 version of it. **The backbone does not have that problem.**

- `MODALITY_INFO["untok_dem@224"]` declares `input_size: 224`, `patch_size: 16`.
  `build_modality_embeddings` builds the `ImageEncoderEmbedding` with `image_size=224`.
- With `sincos_pos_emb=True`, the embed registers `pos_emb = build_2d_sincos_posemb(14, 14, 768)`
  as a **buffer**. The checkpoint ships it at `(1, 196, 768)`. Measured against the closed form,
  the largest difference is **6.0e-8**.
- `ImageEncoderEmbedding.forward` calls `interpolate_pos_encoding` (bicubic) only when
  `(H, W) != image_size`. At 224 px that branch is never taken.

So the answer to ADR-0011's question for B9 is **native**: the registry entry declares
`pos_encoding="sincos2d_terramind"`, `pos_emb_derivation="as_published"`,
`pos_emb_native_grid_px=None`. There is no ablation row to pre-register, because there is no
derivation to ablate. The forward confirms the grid: input `(4, 1, 224, 224)`, and every returned
tensor is `(4, 196, 768)`.

**Layout.** Patches are cut with `rearrange("b d (nh ph) (nw pw) -> b (nh nw) (ph pw d)")`, so
the sequence is already ADR-0012 §5's canonical `(N, 196, D)` in row-major order. There is no
`[N, D, 14, 14]` to unpack, unlike the tokenizer. One oddity: `build_2d_sincos_posemb` puts the
index it calls "w" in the outer loop, so the first half of each position vector encodes the row.
This is how the model was trained, and it only matters to someone who rebuilds the table by hand.
The extractor uses terratorch's own embed module, so nothing has to be rebuilt.

## 4. Layers, under ADR-0012

**Structure** (`terramind_v1_base` in `terramind_register.py`, and the built model): 12 blocks,
dim 768, 12 heads, SwiGLU MLP (`gated_mlp=True`, hidden 2048), no qkv/proj/MLP biases,
`num_register_tokens=0`, no CLS token, and `encoder_norm = LayerNorm(768, eps=1e-6, bias=False)`.
The 25 LayerNorm biases are zero buffers, and the file's copies of them are zero too.

**What terratorch returns.** `TerraMindViT.forward` appends each block's output to a list, then
runs `out[-1] = self.encoder_norm(x)`. I hooked every block and compared on 60 tiles:

| Check | Result |
|---|---|
| Returned entries 1-11 equal the hooked raw block outputs | bit-identical, all 15 batches |
| Returned entry 12 minus `encoder_norm(raw block 12)` | max abs **0.0** |
| Cosine, returned entry 12 against raw block 12 | mean **0.958**, 1st pct 0.942, min 0.661 |
| RMS, raw block 12 against returned entry 12 | **8.09** against 1.22 |

So **the raw block-12 tensor is not in terratorch's output at all.** The norm also changes the
tensor substantially, not by a rounding error. This is the opposite of the tokenizer, where p118
§5.3 found no post-norm and the 12th entry is `encoder(x)` itself.

**What ADR-0012's fields mean for B9**, using DEMAE's convention, which is the one B9 now
matches:

| Field | B9 | DEMAE (for comparison) | Tokenizer B4/B5 |
|---|---|---|---|
| `intermediates[0..11]` | raw outputs of blocks 1-12 | raw outputs of blocks 1-12 | blocks 1-12 (no norm exists) |
| `output` | `encoder_norm(block 12)` | `norm(block 12)` | = `intermediates[11]` |
| `intermediates[11] == output`? | **no** | no | yes |
| `output_is_normed` | **True** | True | False |
| `has_cls` | False | True | False |
| layout | `(N, 196, 768)` native | sequence, CLS dropped | `[N, 768, 14, 14]` reshaped |

**UperNet** reads blocks **3, 6, 9, 12**, which are `intermediates[2, 5, 8, 11]`. All four are
un-normed for B9, just as they are for DEMAE. That is ADR-0012's "index 12 is not the linear
probe's input" asymmetry, and B9 now carries it as well. The **linear probe** reads `output`, the
normed tensor IBM's own decoder was trained on.

**Depth sweep.** Indices 1-12 come from `intermediates`, and B9, like DEMAE, has a separate
`output` point. #190 added DEMAE's `output` point to the sweep. The same reasoning applies to
B9's, which is 0.958 cosine from its block 12, so the sweep should plot it too. Whether it does is
the plan's call (§5.3), not this note's.

## 5. What the extractor has to do differently

`extract._forward_terratorch` calls the tokenizer's `ViTEncoder` and takes its list as
`intermediates`, with `output = intermediates[-1]`. **Reusing that path for B9 would be wrong
without any error being raised.** It would put the *normed* tensor into `intermediates[11]`, so
B9's UperNet would read a normed block 12 while DEMAE's reads a raw one. That is exactly the
learned-affine asymmetry ADR-0012 rejected. B9 needs its own forward, which:

1. builds the backbone with `pretrained=False, modalities=["DEM"], merge_method=None`, loads the
   pinned file, and checks every model tensor against the file (§1);
2. runs `encoder_embeddings["untok_dem@224"]`, adds `x + emb`, then runs the 12 blocks, keeping
   each raw output. This is unrolled the way `_forward_blocks` unrolls timm, or done with forward
   hooks as the probe does;
3. returns `output = encoder_norm(raw block 12)`, with no CLS side field.

A test should pin the unrolled pass bit-identical to terratorch's returned list for entries 1-11,
and pin `output` to entry 12, as the probe does. Then a terratorch upgrade that changes either
side is caught by the test.

`merge_method=None` matters too. The default is `"mean"`, which averages over image modalities.
With a single modality that is numerically a no-op, but it is a step whose meaning changes the
moment anyone adds a second modality.

## 6. Parameters for §2's capacity table

| Count | Number | What it covers |
|---|---:|---|
| **`parameters()`, as built (DEM only)** | **85,151,232** | encoder 84,953,088 + `untok_dem` embed 197,376 (`proj` 196,608 + `mod_emb` 768) + `encoder_norm` weight 768 |
| + fixed sin-cos `pos_emb` buffer | 85,301,760 | adds 150,528 |
| `state_dict`, with zero LN-bias buffers | 85,320,960 | adds 19,200 zeros |

**The rule is the registry's existing one:** each encoder's `parameters()` as built. R5's
85,254,144 excludes its fixed `pos_embed` because DEMAE registers it as a buffer. That is exactly
85,405,440 − 197 × 768, where 85,405,440 is the timm module with the position table as a
Parameter, which is how ImageNet-MAE's count arises. The tokenizer's 90,172,416 includes its
`pos_emb` because terratorch made that one a Parameter. B9's `pos_emb` is a buffer, as R5's is,
so **85,151,232** is the like-for-like number.

Against R5, B9 is 102,912 parameters smaller, **−0.12%**. The whole gap is biases: R5 has
103,680 bias parameters (patch embed, qkv, proj, MLP, LayerNorm) that TerraMind leaves out, and B9
adds a 768-wide `mod_emb`. The gated MLP's three 2048-wide matrices hold exactly as many weights
as the plain MLP's two 3072-wide ones. **So a C1 pair read against B9 is matched to 0.12%.**
§2's "largest C1 gap is 5.8%" remains the tokenizer's figure.

## 7. Does ADR-0012 need amending?

**No decision changes.** B9 fits every decision as written: raw metres in with a declared diet
(D1), twelve intermediates plus a separate `output` (D3), `(N, 196, D)` row-major with no CLS
(D5), a pinned md5 checked at load (D7.1), a named constant pair checked by value (D7.2), twelve
layers (D7.3), and a declared `pos_emb` derivation (D7.4, here `as_published`).

**Three sentences were written for the tokenizer and read wrongly for B9.** They need an
amendment note of the kind #156 added under Decision 5, not a new decision:

1. **Decision 3:** `output` is "identical to block 12 for TerraMind". That is true of the
   tokenizer and false for B9, whose `output` is `encoder_norm(block 12)`, 0.958 cosine away. The
   field-level note "DEMAE's `intermediates[11]` is not its `output`" now also applies to B9.
   Whether `output` is normed is the per-encoder `Encoder.output_is_normed` field, not a
   per-family rule.
2. **Decision 5:** "TerraMind emits `[N, 768, 14, 14]` spatial". That is the tokenizer. The
   backbone emits the sequence natively. The CLS clause ("TerraMind structurally has none") holds
   for B9.
3. **Decision 7.2:** it names `tok_dem@224`. B9's checkpoint is selected by `untok_dem@224`, and
   both resolve to 670.665 / 951.272.

Context fact 1 ("TerraMind's `return_intermediates=True` yields twelve tensors with no separate
post-norm entry") is also tokenizer-specific. It is historical context, so it can stay as written
if the note says so.

## 8. Not settled here

- **The precision figure for B9.** ADR-0012 Amendment 1's bf16 measurement (0.99888) is the
  tokenizer's. B9 is a different network, and its bf16-against-fp32 agreement does not exist
  until the build measures it, as #157 did for A5.
- **How much raw DEM the v1 pretraining actually fed the `untok_dem` embed.** Neither the model
  card nor terratorch says. The weight statistics show it was trained, not for how long.
- **Wall-clock.** Not measured. B9 runs the same 12 ViT-B blocks over 196 tokens as the other
  ViT-B arms, plus a one-time 1.5 GB checkpoint read per process.

## Sources

- terratorch 1.2.10 (installed in the main checkout's `.venv`):
  `models/backbones/terramind/model/terramind_register.py` (`pretrained_weights`,
  `v1/v01_pretraining_mean/std`, `terramind_v1_base`, `build_terrammind_vit`,
  `checkpoint_filter_fn`), `terramind_vit.py` (`TerraMindViT.forward`, `encoder_norm`),
  `encoder_embeddings.py` (`ImageEncoderEmbedding`), `modality_info.py` (`untok_dem@224`),
  `tm_utils.py` (`build_2d_sincos_posemb`, `interpolate_pos_encoding`, `LayerNorm`).
- Hugging Face `ibm-esa-geospatial/TerraMind-1.0-base` at `fb96c70d`: the model card
  (`README.md`), file metadata and commit history via the HF API.
- The checkpoint itself, measured by the script above.
- [#138](https://github.com/JStoschek/DEMAE/issues/138) / [`p138-terramind-corpus.md`](p138-terramind-corpus.md)
  §7 for the three constant pairs; [ADR-0011](../adr/0011-terramind-posemb-rederived-at-14x14.md),
  [ADR-0012](../adr/0012-frozen-feature-extractor-contract.md);
  `src/benchmark/arms.py` and `src/benchmark/extract.py` for the registry conventions.
