# P147 — Does an ImageNet MAE ViT-S/16 checkpoint exist, and can A6-S run?

Ticket [#147](https://github.com/JStoschek/DEMAE/issues/147) · plan [§1](task1-void-fill-plan.md) row A6, [§2.3](task1-void-fill-plan.md) · precedent [`p101-mae-checkpoint-licence.md`](p101-mae-checkpoint-licence.md) · code `src/benchmark/arms.py`, `src/pretrain/config.py`

> **This is not legal advice and contains no legal conclusion.** It reports what each distributor published, in their own words, with URLs and a retrieval date; where the published record is silent; and where the record contradicts itself. Inferences are marked as such. What DEMAE runs is a judgment call for the project, made with this evidence in hand.

## 1. What was asked

Plan §1's A6 row and §2.3's appendix register **A6-S** — "ImageNet-MAE ViT-S frozen + head" — as one of two single-seed ViT-S appendix cells. `src/benchmark/arms.py` registered it as `ARMS["A6-S"]` on encoder `imagenet-mae-vits`, with a comment flagging that it may not be runnable, because `src/pretrain/config.py`'s `Init.mode` field records the opposite fact:

> `imagenet_mae` is R4's continual arm (#86/#102), **ViT-B only — no ViT-S MAE checkpoint exists at any provenance**.
> — `src/pretrain/config.py:443`

P101 §7 says the same thing about #86's parenthetical: "OpenMMLab publishes MAE at ViT-B, ViT-L and ViT-H only — no ViT-S. The arm stays at B for the same reason."

Two claims are stacked there and only one of them is true.

## 2. Bottom line

**A ViT-S/16 ImageNet MAE checkpoint does exist.** The repo's standing claim — "no ViT-S MAE checkpoint exists at *any* provenance" — is too strong and should be corrected wherever it appears.

What exists is exactly one file, and it is not from any of the provenances the repo has been checking:

| | |
|---|---|
| File | `vit-s_imagenet_mae.pth` |
| Publisher | `lambert-x/medical_mae` — Xiao, Bai, Yuille, Zhou (Johns Hopkins), WACV 2023 |
| Distribution | Google Drive, linked from the repo README's Table-1 checkpoint table |
| Repo licence | **Apache 2.0**, stated in the README and re-confirmed by the owner in an issue thread (§4.2) |
| timm-format mirror | [`1aurent/vit_small_patch16_224.medmae_in1k_mae`](https://huggingface.co/1aurent/vit_small_patch16_224.medmae_in1k_mae) — `license: apache-2.0`, 86,676,336 B safetensors, `num_features: 384`, `num_classes: 0` |

So A6-S is **runnable**, on a permissively-licensed artifact, at a materially lower licence risk than the ViT-B row's own history carried (p101).

**But its pretraining is undocumented in a way A6's is not**, and that is the finding that actually decides the row. The publisher never states the epoch count, never states which ImageNet, has no training record for it, and its two available data labels disagree with each other — the paper's Table 1 groups it under "ImageNet (14M)", the mirror names it `in1k` (§5). A6 exists to be *the neutral encoder*, and p101 spent a whole memo establishing that a 2.9-point ImageNet linear-probe difference in that encoder's init was a confound worth paying to avoid. This checkpoint publishes no probe number at all.

**Decision, 2026-08-12: A6-S is dropped; A4-S stands.** Reasons in §7. Landed as a plan amendment against [#147](https://github.com/JStoschek/DEMAE/issues/147), not a silent edit, and the amendment says "a checkpoint exists and was not used, for this reason" rather than "none exists" — a stronger record, and one that stops this being re-litigated.

## 3. What does not exist — the negative, evidenced

### 3.1 The official MAE release: B/L/H only

`facebookresearch/mae`'s README checkpoint table publishes ViT-Base, ViT-Large and ViT-Huge and nothing smaller. The repo is archived (last commit 2022-04-20, p101 §3.4), so the table is final. The code agrees: `models_mae.py` defines `mae_vit_base_patch16_dec512d8b`, `mae_vit_large_patch16_dec512d8b` and `mae_vit_huge_patch14_dec512d8b` — there is no ViT-S builder, so the official codebase could not have produced one without an edit.

### 3.2 timm 1.0.28: three MAE ViT tags, none of them small

Enumerated locally rather than read off a web page, so it is reproducible:

```bash
uv run --no-project --with timm python -c "import timm; print(timm.list_pretrained('*mae*'))"
```

The ViT entries are exactly three — `vit_base_patch16_224.mae`, `vit_large_patch16_224.mae`, `vit_huge_patch14_224.mae`, all three still carrying p101's `license='cc-by-nc-4.0'` annotation and all three resolving to `dl.fbaipublicfiles.com`. The other twelve `*mae*` tags are **Hiera** (`hiera_tiny/small/base/base_plus/large/huge_224.mae`), also `cc-by-nc-4.0`. Hiera has small and tiny sizes, and they are not usable here: Hiera is a hierarchical, multi-scale backbone, so there is no 14×14 single-scale token grid, no 196-token canonical layout and no `pos_embed` of the kind ADR-0011 and ADR-0012 are written about.

`timm.list_pretrained('vit_small_patch16*')` returns AugReg (supervised), DINO, DINOv3 and Naver's RoPE variants. **No `vit_small_patch16_224.mae`.** Retrieved 2026-08-12, timm 1.0.28.

### 3.3 mmselfsup / mmpretrain: B/L/H only

`configs/mae/metafile.yml` lists eight MAE entries: ViT-B at 300/400/800/1600 epochs, ViT-L at 400/800/1600, ViT-H at 1600. **No `vit-small-p16` entry.** This confirms p101 §7's statement about OpenMMLab, which was the one part of the standing claim that was correct.

Source: https://github.com/open-mmlab/mmpretrain/blob/main/configs/mae/metafile.yml

### 3.4 MAE-Lite: Tiny only

`wangsr126/mae-lite` (ICML 2023, the lightweight-ViT MAE study) is Apache-2.0 — "This repo is released under the Apache 2.0 license" — and is the most obvious place a ViT-S MAE would live. It releases **ViT-Tiny only**: MAE-Tiny, MAE-Tiny-FT, MAE-Tiny-FT-RPE, MAE-Tiny-Distill-D²-FT-RPE. No Small, no Base.

Source: https://github.com/wangsr126/mae-lite

## 4. The one candidate that does exist

### 4.1 What it is and where it comes from

`lambert-x/medical_mae` is the official implementation of *Delving into Masked Autoencoders for Multi-Label Thorax Disease Classification* (Xiao, Bai, Yuille, Zhou; WACV 2023; [arXiv 2210.12843](https://arxiv.org/abs/2210.12843)). Its README ships the Table-1 checkpoint table, whose header is `Model | Pretrained Dataset | Method | Pretrained | Finetuned (NIH Chest X-ray) | mAUC`. The relevant row, verbatim:

> `| ViT-Small/16 | ImageNet | MAE | [google drive](https://drive.google.com/file/d/1QeAIWJWuNcccF09502xcnQ_2tbc2jRQG/view?usp=share_link) | [google drive](…) | 78.6 |`

The Drive object behind that link is named **`vit-s_imagenet_mae.pth`** (page title and embedded filename, retrieved 2026-08-12).

**The authors' own code can produce it, and the official code cannot.** `medical_mae/models_mae.py` defines a ViT-S MAE family the upstream repo does not have:

```python
def mae_vit_small_patch16_dec128d2b(**kwargs):   # patch_size=16, embed_dim=384, depth=12, num_heads=6
def mae_vit_small_patch16_dec512d8b(**kwargs):
def mae_vit_small_patch16_dec512d2b(**kwargs):
mae_vit_small_patch16 = mae_vit_small_patch16_dec128d2b
```

*(Inference, marked: the row links their own Drive copy where the neighbouring DeiT row links `dl.fbaipublicfiles.com` directly — so they cite upstream when an upstream exists — and their fork carries the ViT-S builder the upstream lacks. The best explanation is that they pretrained this file themselves. They never say so.)*

### 4.2 The licence — the distributor's own words

README §License, in full:

> This repo is under Apache 2.0 license.

Source: https://github.com/lambert-x/medical_mae#license

Unlike p101's subject, **the publisher was asked and answered.** [Issue #4 "License"](https://github.com/lambert-x/medical_mae/issues/4), opened 2022 by `ieee8023` (Joseph Cohen, TorchXRayVision), asks:

> Can you tell me what the license is for this code **and model**? I would like to add it to the [TorchXRayVision](https://github.com/mlmed/torchxrayvision) library which is under an Apache License.

The repo owner (`lambert-x`, `OWNER`) replied and closed it:

> License updated. This repo uses Apache 2.0 license.

**Be honest about the symmetry, exactly as p101 §6.1 was.** That is still the "this repo" construction p101 §3.2 spent a section calling ambiguous, and the answer says "repo" where the question said "code and model". Two things make it stronger than the MAE case anyway, and neither is decisive: the question that prompted it *named the models* and named a redistribution use, so the answer was given in a context about weights; and the repo is live (last push 2024-03-14, not archived), so unlike `facebookresearch/mae` there is still a channel to ask. *(Inference, marked.)*

Note also that the repo's headers carry Meta's copyright notice ("Copyright (c) Meta Platforms, Inc. and affiliates") because it is derived from MAE-official's *code* — that is the code-lineage question p101 already treated separately, and issue #4's commenter raised and traced it to DeiT/BEiT.

### 4.3 The timm-format mirror

[`1aurent/vit_small_patch16_224.medmae_in1k_mae`](https://huggingface.co/1aurent/vit_small_patch16_224.medmae_in1k_mae), last modified 2023-11-28, part of `1aurent`'s "MedMAE models" collection which points at `github.com/lambert-x/medical_mae`. Card frontmatter is four lines — `tags: [image-classification, timm]`, `library_name: timm`, `license: apache-2.0` — and there is no body: no provenance sentence, no citation, no epoch count.

`config.json`:

```json
{"architecture": "vit_small_patch16_224", "num_classes": 0, "num_features": 384,
 "global_pool": "token",
 "pretrained_cfg": {"tag": "augreg_in1k", "custom_load": true, "input_size": [3,224,224],
                    "mean": [0.5,0.5,0.5], "std": [0.5,0.5,0.5], "crop_pct": 0.9, …}}
```

`model.safetensors` is 86,676,336 B. *(Inference from the byte count: ÷4 B/float ≈ 21.67 M fp32 parameters, which matches ViT-S/16 encoder-only with a 3-channel patch embed and no classifier — consistent with DEMAE's own 384-d ViT-S at 21,393,408, the ~0.27 M difference being the 3-channel patch embed this project collapses to 1 anyway, same recipe as `pretrain.init`.)*

**One flag on the mirror, and it is the p101 §4(b) shape.** `pretrained_cfg.tag` is `augreg_in1k` and the preprocessing block (mean/std 0.5, `crop_pct` 0.9, bicubic) is AugReg's, not MAE's ImageNet mean/std. That is a timm default carried over from the wrong sibling entry, not a statement about how these weights were trained. It costs DEMAE nothing directly — the arm collapses to one channel and applies §2.1's per-tile normalization regardless — but it means **the mirror's metadata is not evidence about the file's provenance**, and the `in1k` in the model name is part of that same metadata.

Whether the mirror is bit-identical to `vit-s_imagenet_mae.pth` is **not verified** — no md5 is published on either side, and the mirror is a format conversion, so byte equality is not expected. It is asserted by the collection's pointer and nothing else.

## 5. What is not documented, and where the record contradicts itself

This is the part that decides the row, so it is listed rather than summarized.

1. **No epoch count, anywhere.** The paper's §4 "Implementation details" gives the recipe for *their X-ray* pretraining — AdamW, β=(0.9, 0.95), wd 0.05, lr 1.5e-4, batch 2048, 20-epoch warmup, cosine, "The pre-training stage takes 800 epochs in total." That paragraph is about the 266,340 chest X-rays. **No corresponding sentence exists for the ImageNet MAE ViT-S.** For comparison, p101 could put OpenMMLab's ViT-B on an epoch axis (300/400/800/1600) and pick 1600 deliberately.
2. **`PRETRAIN.md` points at a repo that cannot do it.** Verbatim: "For MAE pretraining for ViT on ImageNet, please refer to the [MAE Official](https://github.com/facebookresearch/mae/blob/main/PRETRAIN.md)." The official recipe has no ViT-S configuration (§3.1). So the repo's own instructions do not reproduce the file it distributes.
3. **The two data labels disagree.** The paper's Table 1 groups this row under **"ImageNet (14M)"** — the paper's framing throughout is "ViT pre-trained on ImageNet (14M data & labels)", i.e. the ~14 M-image ImageNet-21k, not the 1.28 M ImageNet-1k. The mirror is named **`medmae_in1k_mae`**. One of these is wrong and no source resolves it. (The table's "(14M)" grouping is loose in the other direction too — the row above it is the official DeiT-S, which is ImageNet-**1k** supervised.)
4. **No quality number on the axis that matters.** The only figure published for it is 78.6 mAUC after fine-tuning on NIH ChestX-ray14. There is no ImageNet linear probe, no ImageNet fine-tune, no reconstruction loss — nothing comparable to the 65.1/67.1 probe numbers p101 §6.1 used to price the ViT-B substitution.
5. **No footnote claiming external provenance.** Table 1's dagger legend attributes the *CNN* self-supervised rows to Ericsson et al.; the ViT-S ImageNet MAE row carries no dagger and no citation. *(Inference: consistent with §4.1's reading that they trained it. It is the absence of a marker, not a statement.)*

## 6. Everything else checked and rejected

Recorded so the sweep is not repeated. Hugging Face was enumerated through the model API (`/api/models?search=…`, up to 1,000 results per query, for `vit_small`, `vit-small`, `vits`, `mae_vit`, `mae-vit-small`, plus `author=timm&search=mae`), filtered to ids containing both an MAE marker and a small-size marker. Retrieved 2026-08-12.

| Candidate | Why not |
|---|---|
| `timm/vit_base/large/huge_*.mae` | Right objective and data, wrong capacity — this is A6, not A6-S |
| `hiera_small_224.mae`, `hiera_tiny_224.mae` | Hierarchical multi-scale backbone: no 14×14 grid, no 196-token layout, no `pos_embed` of ADR-0011's kind. Also `cc-by-nc-4.0` |
| mmselfsup / mmpretrain MAE | B/L/H only (§3.3) |
| `wangsr126/mae-lite` | ViT-Tiny only (§3.4) |
| `eminorhan/mae_s_vits16`, `mae_say_vits16`, `mae_a_vits16`, `mae_y_vits16` | MAE ViT-S/16, MIT — but pretrained on **SAYCam infant head-camera video**, not ImageNet. Same rejection p101 §6.2 already made for the ViT-B sibling |
| `STomoya/vit_small_patch16_224.st_mae_sb1k` | MAE ViT-S/16, Apache-2.0 — pretrained on `sb1k`, an illustration/booru corpus, not ImageNet. Card publishes only a reconstruction MSE |
| `moham741/mae-vit-small-s2-100k` | Sentinel-2 imagery. Wrong data — and an EO-pretrained encoder is a different arm than "the neutral encoder", which is what A6 is for |
| `mwalmsley/euclid_encoder_mae_zoobot_vit_small_patch8_224` | Euclid astronomy imagery, and patch **8**, not 16 |
| `Dhi-Technologies/thermal-mae-vits-pilot` | Thermal imagery, "pilot", no card |
| `rooty2020/mae-vits-16` | No model card at all ("Entry not found"), no licence tag, no stated pretraining data. Unusable as evidence of anything |
| `1aurent/vit_small_patch16_224.medmae_CXR_mae` | The X-ray-pretrained sibling of §4's file. Wrong data |
| DINO / DINOv3 / iBOT / AugReg / DeiT ViT-S/16 | Real, permissive, ViT-S/16 — but not MAE. Would make A6-S "ImageNet ViT-S with a different objective", which is a different row than the one §2.3 registered |

**SSL4EO-S12's MAE ViT-S/16 deserves its own line** because `pretrain.config` cites SSL4EO-S12 at `src/pretrain/config.py:148` as the benchmark for this scale, so someone will reach for it: it is Sentinel-2 pretrained, not ImageNet, so it fails A6's "neutral encoder" purpose in the same way `moham741` does.

## 7. What this means for the plan

**The plan's A6-S row is runnable.** The premise the build session flagged — that it cannot be run as specified — is wrong, and `arms.py`'s comment and `pretrain/config.py:443`'s "no ViT-S MAE checkpoint exists at any provenance" both need correcting regardless of what the plan decides to do with the row.

**The row was nevertheless dropped.** A6-S serves two readings and the checkpoint breaks the one §2.3 actually names:

- *§2.3's stated reading* — "did the extra 4× buy anything", read down a column as A6 vs A6-S. **Broken.** A6's init is OpenMMLab's 1600-epoch ImageNet-1k MAE ViT-B, chosen in p101 §6.1 *specifically* on the epoch/probe axis; A6-S's init is a different lab's ViT-S at an unknown epoch count on an unstated ImageNet variant. The A6→A6-S delta would confound capacity with trainer, schedule and possibly corpus. That is not a capacity observation.
- *The cross reading* — A4-S vs A6-S, DEMAE against ImageNet-MAE at matched small capacity. **Survives, but weakly.** A DEMAE win at ViT-S is discountable in one sentence ("your ImageNet baseline was an undocumented side-artifact of a chest-X-ray paper"), and there is no published quality number to rebut it with. p101 paid real integration cost to avoid exactly this class of objection at ViT-B.

The row is cheap to run (~$0.15, 1 of 39 scored passes) and cheap to drop — §2.3 already demotes ViT-S to "a capacity-scaling aside that no claim in §0 depends on", it is out of A2/A3 and out of the label-efficiency sweep, and it is `printed=False`. When a row is cheap either way, the tiebreak is whether it produces a number anyone can read, and this one does not.

**A4-S stands unchanged.** It is DEMAE's own ViT-S, trained by this project (R1), with a full training record. §2.3's ViT-S appendix becomes **one cell, not a pair**.

**No registered rule is touched.** R5 fixes the *printed* table — 11 printed arm rows × 6 columns × 66 (row, band) cells = 396 numbers — and both ViT-S cells sit outside it at `printed=False`. R1, R6, R7, R8 and R11 are about metrics, pairs, thresholds and the A5 ablation. Nothing in ADR-0013 names A6-S, so this amendment needs no ADR amendment; it needs the ticket the build spec's "When the build finds the plan wrong" requires.

**If a future session wants the row anyway**, the cost is one config line (`paths.imagenet_mae_vits_checkpoint`) plus re-registering the arm, and this section is the caveat it must print beside the number. Nothing about dropping it is hard to reverse — which is the other reason to drop rather than run-and-caveat.

### Sites amended, 2026-08-12

- **Plan §1**, A6 row — "(ViT-S as the §2.3 appendix row)" removed; it is now false for A6. A4's row keeps its ViT-S clause.
- **Plan §2.3** — the "ViT-S is one appendix row" paragraph now registers A4-S alone, followed by a new "A6-S is dropped, and not because the checkpoint is missing" paragraph carrying this memo's ground.
- **Plan §4.1** (seeds), **§4.5** (order of operations), **§5** (the seed-spread parenthetical) — every "two ViT-S appendix cells" is now one.
- **Plan §7** — the pass arithmetic re-derived at the line the drop actually moves: head training **14 → 13 cell-seeds** (42.1 → **39.1** min at the 3.01 min/cell-seed already priced), inference **38 → 37** passes (4.5 → **4.4** min), GPU total **148.2 → 145.1 min = 2.42 h**, **$0.99 → $0.97**, eval passes **39 → 38** (2.0–5.2 → **1.9–5.1** CPU-h), and the 23-at-20k decomposition → 22. The 34.4M presentation count and everything derived from it was *already* flagged stale and is now one cell-seed staler; it is not re-derived here, because the build spec assigns that to `freeze`.
- **`src/benchmark/arms.py`** — the `imagenet-mae-vits` encoder and the `A6-S` arm removed, absent rather than disabled, on the same rule as A8/A10; the module docstring records why.
- **`src/benchmark/config.py`** — `Paths.imagenet_mae_vits_checkpoint` removed. `Paths` is excluded from the config hash (`metadata=ENVIRONMENT`), so no hash moves.
- **`src/pretrain/config.py`** — `Init.mode`'s "no ViT-S MAE checkpoint exists at any provenance" corrected. The claim #86 actually needed is *no ViT-S MAE at a provenance this project will build on*, and it is now stated that way with this memo cited.
- **`docs/task1-build-spec.md`** — user story 42, the 39-pass and 2.47 GPU-hour figures, the order-of-operations line, and a third entry under "plan inconsistencies to settle" recording this one as found-by-the-build.
- **`tests/test_benchmark_config.py`** — the appendix-cell test now covers A4-S alone, plus a new test asserting A6-S and its encoder are absent from the registry and that `arms.get("A6-S")` refuses like any other unknown name.

**Not amended: ADR-0013.** No registered rule names A6-S, and R5 — the one that counts rows — fixes the *printed* table, which both ViT-S cells sit outside at `printed=False`. Suite green after the change: 1,824 passed, 14 skipped.

## Sources

**The candidate**
- `lambert-x/medical_mae` README, Table-1 checkpoint table and §License: https://github.com/lambert-x/medical_mae
- `models_mae.py` (the ViT-S MAE builders): https://github.com/lambert-x/medical_mae/blob/main/models_mae.py
- `PRETRAIN.md` ("please refer to the MAE Official"): https://github.com/lambert-x/medical_mae/blob/main/PRETRAIN.md
- Issue #4 "License" — the owner's answer: https://github.com/lambert-x/medical_mae/issues/4
- The Drive object `vit-s_imagenet_mae.pth`: https://drive.google.com/file/d/1QeAIWJWuNcccF09502xcnQ_2tbc2jRQG/view
- Paper (Table 1, §4 implementation details): https://arxiv.org/abs/2210.12843
- timm-format mirror and its `config.json`: https://huggingface.co/1aurent/vit_small_patch16_224.medmae_in1k_mae

**The negatives**
- `facebookresearch/mae` README checkpoint table and `models_mae.py`: https://github.com/facebookresearch/mae
- timm 1.0.28, enumerated locally via `timm.list_pretrained('*mae*')` and `('vit_small_patch16*')`
- mmpretrain MAE metafile: https://github.com/open-mmlab/mmpretrain/blob/main/configs/mae/metafile.yml
- `wangsr126/mae-lite` README and licence: https://github.com/wangsr126/mae-lite
- Hugging Face model API sweeps: `https://huggingface.co/api/models?search={vit_small,vit-small,vits,mae_vit,mae-vit-small}&limit=1000`

**Precedent this memo follows**
- [`p101-mae-checkpoint-licence.md`](p101-mae-checkpoint-licence.md) — the ViT-B version of this question
- [`p102-mmselfsup-mae-layout.md`](p102-mmselfsup-mae-layout.md) — what loading an mmselfsup MAE actually costs

All URLs retrieved 2026-08-12.
