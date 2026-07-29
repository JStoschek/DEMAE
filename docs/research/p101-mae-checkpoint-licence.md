# P101 — the ImageNet MAE checkpoint licence, and whether R4's encoder can ship

Ticket [#101](https://github.com/JStoschek/DEMAE/issues/101) · map [#76](https://github.com/JStoschek/DEMAE/issues/76) · plan §1, §8, §12 (R4), §14 · downstream [#86](https://github.com/JStoschek/DEMAE/issues/86), [#88](https://github.com/JStoschek/DEMAE/issues/88), [#102](https://github.com/JStoschek/DEMAE/issues/102)

> **This is not legal advice and contains no legal conclusion.** It reports what the licence text says, what the publisher said and did not say, where the published record contradicts itself, and where the question is genuinely open. Everything below is either a quote with a URL or an inference explicitly marked as one. A decision about what DEMAE ships is a judgment call for the project, made with this evidence in hand.

## 1. What was asked, and why it matters

Plan §8 warm-starts run **R4** — the ViT-B continual-pretraining arm — from "the official ImageNet MAE ViT-B/16 checkpoint," meaning `mae_pretrain_vit_base.pth` from `facebookresearch/mae`. The encoder's RGB patch-embed kernels are summed down to one channel, a fresh decoder is attached, and 150 epochs of MAE pretraining run on the DEM corpus. Plan §14 then ships, per released run, "encoder-only weights (timm-loadable) + full MAE checkpoint," publicly.

Plan §1 already rejected that repo as a **code** base and gave its licence as one reason:

> `facebookresearch/mae` … is **CC-BY-NC 4.0 licensed**, which is a poor foundation for an open-weights release.
> — `docs/phase1-pretraining-plan.md` §1 "Codebase", line 36

That sentence was written about the code and never re-asked about the weights. If the same licence reaches the checkpoint, then R4's output is built on non-commercial material and §14's "ship it publicly" is a different deliverable than the plan assumes. #86 calls the scratch-vs-continual comparison "a headline result of the release," and #88 packages artifacts per released run. Both assume R4 produces something shippable.

The cost of getting this wrong is asymmetric. Running R4 costs ~17 GPU-h and ~$6; that spend is fine either way, because the *comparison* can be reported regardless. What is expensive is publishing a checkpoint and later having to withdraw it.

## 2. Bottom line

**The licence status of the official MAE checkpoints is genuinely unresolved, and it is unresolved in a way that cannot be fixed by asking.** That is the finding. It is not a placeholder for a better answer.

Concretely, and each of these is evidenced below:

1. The only statement the publisher ever made is the repo README's one line: **"This project is under the CC-BY-NC 4.0 license."** Not "the code." *Project*. The repo has no model card, no release notes, and no per-checkpoint licence file. The checkpoints are hosted on a bare CDN bucket that serves no terms of any kind.
2. **Two people asked the maintainers this exact question, in 2023 and 2024. Neither got an answer.** The repo was then **archived**, so no answer is coming. The last commit is 2022-04-20.
3. **The two redistributions of the same weights carry contradictory licence tags.** Meta's own Hugging Face repo `facebook/vit-mae-base` says `license: apache-2.0`. `timm/vit_base_patch16_224.mae` says `license: cc-by-nc-4.0`. Same original file, opposite answers.
4. That contradiction does **not** resolve in Apache's favour on inspection. The `apache-2.0` tag on `facebook/vit-mae-base` was set by a Hugging Face employee, in a model card whose first revision also carried a stray `tags: - dino` — i.e. it was copy-pasted from the DINO card, which genuinely *is* Apache-2.0. The card itself says the MAE team did not write it. Meanwhile timm's `cc-by-nc-4.0` is a deliberate per-weight annotation in timm's source code, applied to the MAE and I-JEPA entries and *not* to the AugReg or DeiT-3 entries sitting beside them.
5. **If CC BY-NC 4.0 does apply, the constraint is narrower than "poison."** CC BY-NC 4.0 has **no ShareAlike term** — the string "ShareAlike" appears zero times in the licence file. A derivative does not have to be relicensed CC BY-NC. What it must do is (a) not be shared for commercial purposes, (b) carry attribution and a pointer to the licence, and (c) not be released under terms that stop recipients complying with the licence. A permissive downstream licence like Apache-2.0 would violate (c), because Apache-2.0 grants commercial use. Practically: an R4 checkpoint could be shared under a CC BY-NC-compatible non-commercial licence, but could not be shipped as Apache/MIT alongside R5.
6. **Whether any of this attaches to a weights file at all is an open legal question**, not a settled one. The U.S. Copyright Office's own 2025 report says courts "have reached varying conclusions" about model weights. Creative Commons itself recommends against using CC licences for software.

**Recommendation, stated as engineering risk rather than law:** treat the official checkpoint as **not safe to build a shipped artifact on**, and switch R4's init to OpenMMLab's independently-trained Apache-2.0 MAE ViT-B/16. That keeps the arm intact — same objective, same architecture, same pretraining data — at a measurable cost of roughly 2.9 points of ImageNet linear-probe quality in the init (§6.1). The comparison #86 wants survives; the artifact #88 wants becomes shippable; and the ambiguity above stops being load-bearing.

If the project instead decides to keep the official checkpoint, the honest framing is: "R4's init is under a licence the publisher never clarified and can no longer be asked about; we are relying on the Apache-2.0 tag Meta's own Hugging Face repo carries; we do not ship the R4 checkpoint, only the R4 numbers." That is a defensible position — it is just a decision, and it must be written down as one.

## 3. What the publisher actually said

### 3.1 The repo LICENSE file

`facebookresearch/mae/LICENSE` is 19,337 bytes and is the **verbatim, unmodified legal code of Creative Commons Attribution-NonCommercial 4.0 International**. Its first line:

> `Attribution-NonCommercial 4.0 International`

There is **no header, preamble, or scope note** added by Meta — nothing saying "this covers the code" or "this covers the models." It is the stock CC file, dropped in as-is. There are no `LICENSE_*` variant files; the repo's full top-level listing is `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, `FINETUNE.md`, `LICENSE`, `PRETRAIN.md`, `README.md`, `demo/`, seven `.py` files, and `util/`. **There is no `MODEL_CARD.md`.**

Source: `gh api repos/facebookresearch/mae/contents/LICENSE` and `.../contents`, retrieved 2026-07-29. https://github.com/facebookresearch/mae/blob/main/LICENSE

Minor but telling: GitHub's own licence detector cannot classify it. `gh api repos/facebookresearch/mae --jq .license` returns `{"key":"other","name":"Other","spdx_id":"NOASSERTION"}`.

### 3.2 The README — the one statement that exists

The README's final section, in full:

> ### License
> This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

That is the entire licence prose in the repository. **The word is "project," not "code" and not "repository."** This matters because the same lab, on the same CDN, words it differently elsewhere — `facebookresearch/deit`'s README says "This **repository** is released under the Apache 2.0 license." *(Inference, and a weak one: the word choice may be deliberate scoping or may be nothing. It is offered as a data point, not as proof.)*

The README is also where the checkpoints live. The ViT-Base row of the "pre-trained checkpoint" table links to:

> `https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth` — md5 prefix `8cad7c`

Introduced as: "The following table provides the pre-trained checkpoints used in the paper, converted from TF/TPU to PT/GPU." **No licence sentence accompanies the table.** The checkpoints are presented in the same document as the licence line, under the same "This project" umbrella, with nothing distinguishing them from the code.

Source: https://github.com/facebookresearch/mae/blob/main/README.md

### 3.3 The other repo docs say nothing

- **`CONTRIBUTING.md`** — only mentions licensing to bind contributors: "By contributing to mae, you agree that your contributions will be licensed under the LICENSE file in the root directory of this source tree." Note "source tree." That is inbound-contribution scope, not outbound-artifact scope, and it does not speak to the weights.
- **`PRETRAIN.md`** — pure recipe (hyperparameters, `submitit` invocations, the 85.5% ViT-L sanity check). No licence text.
- **`FINETUNE.md`** — lists the *fine-tuned* checkpoints (`mae_finetuned_vit_base.pth`, md5 `1b25e9`, ImageNet 83.664) on the same CDN. No licence text.

### 3.4 The two licence questions the maintainers never answered

Searching the repo's issues (`gh search issues` for `license`, `licence`, `commercial`, `commercial use`, across 219 issues) returns exactly three, and **not one has a maintainer reply**:

| Issue | Opened | Author association | Maintainer response |
|---|---|---|---|
| [#158 "License questions"](https://github.com/facebookresearch/mae/issues/158) | 2023-03-07 | `NONE` | none |
| [#171 "Creative Commons does not recommend their licenses be used for software"](https://github.com/facebookresearch/mae/issues/171) | 2023-07-19 | `NONE` | none |
| [#208 "License"](https://github.com/facebookresearch/mae/issues/208) | 2024-09-07 | `NONE` | none |

Issue #158 asks our question almost word for word:

> The github page says the project is licensed under CC-BY-NC 4.0 license. Does this license apply to just the model weights that you've trained? Or does the license apply to the source code for the Masked Autoencoder?

Issue #208, in full: "Is the license only for the model weights or the whole idea?"

**#158's single comment is from another anonymous user, not a maintainer**, and reads in full: "Maybe it applies to weight. ex) LLaMA.. :)" — that is a stranger's guess and carries no evidentiary weight. It is recorded here only so nobody later mistakes it for an answer.

`author_association` is `NONE` on all three openers, i.e. none is a Meta employee. All three are still **open**.

**And they will stay open.** `gh api repos/facebookresearch/mae` reports `"archived": true`, with the last commit `efb2a806` on **2022-04-20** ("Fix typos in instructions"). GitHub Discussions is disabled (`has_discussions: false`). There is no channel left through which the publisher could clarify this, short of contacting Meta directly.

### 3.5 The paper

arXiv **2111.06377** (He, Chen, Xie, Li, Dollár, Girshick), v3 of 2021-12-19. The arXiv listing shows the submission itself under **CC BY 4.0** — that is the licence on the *paper*, governing the PDF, and says nothing about weights distributed from a separate repository under a separate licence. The comments field is "Tech report. arXiv v2: add more transfer learning results; v3: add robustness evaluation." The only release language in the body is "We will release code along with the specific configurations" (Appendix A.3). **The paper contains no licensing statement about the models.**

Source: https://arxiv.org/abs/2111.06377

### 3.6 The CDN serves no terms

`dl.fbaipublicfiles.com` is a plain S3-behind-CloudFront bucket. A `HEAD` on the ViT-B checkpoint returns:

```
HTTP/2 200
content-type: binary/octet-stream
content-length: 343249461
last-modified: Wed, 29 Dec 2021 01:01:14 GMT
server: AmazonS3
```

No licence header, no `x-amz-meta-*` licence field, nothing. The bucket root returns `AccessDenied`; `/terms`, `/terms.html`, `/LICENSE`, and `/license.txt` all return **403**. Only `/robots.txt` exists (200). **There is no terms-of-use page on the distribution host.** Retrieved 2026-07-29.

*(Inference from the byte count: 343,249,461 B ÷ 4 B/float ≈ 85.8 M parameters, which matches ViT-B/16 encoder-only. So the "pretrain" checkpoint ships the encoder without the MAE decoder. Relevant to #102 — a fresh decoder is required regardless, so this costs nothing.)*

## 4. The redistributions contradict each other

Both repositories host the same weights, converted from the same `mae_pretrain_vit_base.pth`. They disagree.

| Repo | `license:` field | Who set it | Evidence it is the same weights |
|---|---|---|---|
| [`facebook/vit-mae-base`](https://huggingface.co/facebook/vit-mae-base) | **`apache-2.0`** | `nielsr` (Hugging Face staff) | Card: "first released in [this repository](https://github.com/facebookresearch/mae)" |
| [`timm/vit_base_patch16_224.mae`](https://huggingface.co/timm/vit_base_patch16_224.mae) | **`cc-by-nc-4.0`** | `rwightman` (timm maintainer) | Card: "**Original:** https://github.com/facebookresearch/mae"; timm's `_cfg` `url=` points at `dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth` |

The ticket's prior was that the `facebook/`-namespace tag is strong evidence of Meta's own choice. **On inspection it is not.** Three findings, in order of force:

**(a) The card was written by Hugging Face, and says so.** From the second README revision onward:

> Disclaimer: The team releasing MAE did not write a model card for this model so this model card has been written by the Hugging Face team.

**(b) The very first README revision was a copy-paste of the DINO card, licence tag included.** Commit `07a35ff` ("Create README.md", 2022-01-19T18:35:41Z, author `nielsr`) opens:

```yaml
---
license: apache-2.0
tags:
- dino
datasets:
- imagenet-1k
---
```

The body is about MAE. The `dino` tag is not — it is left over from `facebook/dino-vitb16`, whose card today carries `{'license': 'apache-2.0', 'tags': ['dino', 'vision'], ...}` and whose upstream repo `facebookresearch/dino` genuinely is Apache-2.0. The stray `dino` tag was removed 13 minutes later in `621ecfe`; **`license: apache-2.0` was left untouched and has never been revisited.** The full commit list for the repo is 11 commits, every one authored by `nielsr`, `Joao Gante`, `SFconvertbot`, or `system` — **no commit by anyone identifiable as a Meta/FAIR author.**

*(Inference, clearly marked: the sequence — template copied from an Apache-2.0 sibling model, one field from the template corrected, the licence field not — is much better explained by copy-paste than by a deliberate licensing decision. It is not proof. Meta owns the namespace and has had four and a half years to correct the tag and has not, which is a real if weak counter-argument, and it is the strongest thing anyone relying on the Apache tag has.)*

**(c) timm's `cc-by-nc-4.0` is deliberate, and selectively applied.** It is not just an HF tag; it is a field in timm's source:

```python
'vit_base_patch16_224.mae': _cfg(
    url='https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth',
    hf_hub_id='timm/',
    license='cc-by-nc-4.0',
    mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),
```

The same `license='cc-by-nc-4.0'` appears on the MAE ViT-L and ViT-H entries and on `vit_huge_patch14_gap_224.in1k_ijepa` (I-JEPA, also CC-BY-NC upstream). It does **not** appear on the neighbouring `vit_base_patch16_224.augreg_in21k_ft_in1k` or `deit3_base_patch16_224.fb_in1k` entries, whose HF cards resolve to `apache-2.0`. So the maintainer who did the port annotates weights individually and marked *these specific weights* non-commercial.

Source: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py (lines ~2473–2490, retrieved 2026-07-29)

**Neither Hugging Face repo has a discussion thread about licensing.** `facebook/vit-mae-base` has three discussions (a safetensors bot PR, a question about sequence length, and one spam post); `timm/vit_base_patch16_224.mae` has two (both pipeline-tag fixes). Nobody has raised the conflict there either.

## 5. The derivative-work question, for a continually-pretrained encoder

Two questions stack, and they have to be kept apart:

- **Q1 (prior):** do CC BY-NC 4.0's terms attach to a weights file at all?
- **Q2 (conditional):** *if* they do, what does the licence require of an encoder that was initialized from those weights and then trained for 150 more epochs on a different corpus?

Q2 is answerable from the text. Q1 is not answerable by us at all.

### 5.1 Q2 — what the licence text actually requires of an adaptation

All quotes below are from the licence file **as it sits in the repo** (`facebookresearch/mae/LICENSE`), which is the verbatim CC BY-NC 4.0 legal code. Canonical text: https://creativecommons.org/licenses/by-nc/4.0/legalcode

**"Adapted Material"**, Section 1(a):

> Adapted Material means material subject to Copyright and Similar Rights that is derived from or based upon the Licensed Material and in which the Licensed Material is translated, altered, arranged, transformed, or otherwise modified in a manner requiring permission under the Copyright and Similar Rights held by the Licensor.

Note the trailing clause — "**in a manner requiring permission**." The definition is self-referential to Q1. If no copyright subsists in the weights, nothing about further training "requires permission," and there is no Adapted Material. This is why Q1 cannot be routed around.

**"NonCommercial"**, Section 1(i):

> NonCommercial means not primarily intended for or directed towards commercial advantage or monetary compensation.

**The grant**, Section 2(a)(1):

> …the Licensor hereby grants You a worldwide, royalty-free, non-sublicensable, non-exclusive, irrevocable license to exercise the Licensed Rights in the Licensed Material to:
> a. reproduce and Share the Licensed Material, in whole or in part, for NonCommercial purposes only; and
> **b. produce, reproduce, and Share Adapted Material for NonCommercial purposes only.**

So producing an adaptation is *permitted*. Sharing it is *permitted*. The restriction is purpose, not existence. **Running R4 is fine under any reading.** The question is only about publishing the result.

### 5.2 There is no ShareAlike term, and that changes the answer

**The string "ShareAlike" appears exactly zero times in the licence file** (`grep -c -i sharealike` → `0`). CC BY-NC 4.0 is *not* a copyleft licence. This is the single most commonly misunderstood point about it and the plan's §1 framing ("a poor foundation for an open-weights release") implicitly leans on the copyleft reading.

What actually binds a derivative is Section 3 — Attribution, and one downstream-compatibility clause.

Section 3(a)(1) requires, if you Share the material "including in modified form," that you retain creator identification, a copyright notice, a notice referring to the licence, a warranty disclaimer, and a link to the material; and — 3(a)(1)(B) — that you "indicate if You modified the Licensed Material."

Section 3(a)(4) is the operative constraint on your choice of licence:

> If You Share Adapted Material You produce, the Adapter's License You apply must not prevent recipients of the Adapted Material from complying with this Public License.

Read plainly, that is **not** "you must use CC BY-NC." It is "your licence must not undercut this one." *(Inference, marked: a permissive licence such as Apache-2.0 or MIT affirmatively grants commercial use, so applying it to Adapted Material would let recipients do what CC BY-NC forbids — which reads as exactly the thing 3(a)(4) blocks. A non-commercial licence, CC BY-NC 4.0 included, would satisfy it. This is a reading of the text, not a ruling.)*

**The practical consequence, if the licence applies:** the R4 checkpoint could be released, publicly and for free, under a non-commercial licence with attribution to He et al. It could **not** be released under the same permissive terms as R5. DEMAE's release would carry two artifacts under two different licences, and the "open-weights foundation model" framing would need a footnote for one of them.

### 5.3 Q1 — whether any of this reaches weights is unsettled, and we say so

We are not able to resolve this, and neither, on the published record, is anyone else.

The **U.S. Copyright Office**, *Copyright and Artificial Intelligence, Part 3: Generative AI Training* (Pre-Publication Version, May 2025), p. 29, discussing infringement claims about weights:

> Courts that have addressed infringement claims regarding model weights have reached varying conclusions.

It goes on to contrast *Kadrey v. Meta Platforms* — where the court called the allegation that the Llama models were themselves infringing derivative works "nonsensical" (No. 23-cv-3417, 2023 WL 8039640 (N.D. Cal. Nov. 20, 2023)) — with *Andersen v. Stability AI*, where a motion to dismiss by a party that had merely downloaded an already-trained model was denied. The Office's own position is that it "turns on whether the model has retained or memorized substantial protectable expression."

Source: https://www.copyright.gov/ai/Copyright-and-Artificial-Intelligence-Part-3-Generative-AI-Training-Report-Pre-Publication-Version.pdf

**Two honest caveats on that citation.** First, that passage is about whether weights *infringe the copyright of their training data* — a different question from whether Meta holds a copyright in the MAE weights that it can license to us. Second, it is a Copyright Office report, not binding law. It is cited here for exactly one proposition: that the legal status of weights is contested rather than settled, from a source whose job is to know.

The second relevant primary source is **Creative Commons itself**, which recommends against the licence being used this way at all:

> We recommend against using Creative Commons licenses for software. … Instead, we strongly encourage you to use one of the very good software licenses which are already available.

— https://creativecommons.org/faq/ ("Can I apply a Creative Commons license to software?"). This is the point MAE issue #171 raised in 2023 and that also went unanswered. CC's stated reasons are that CC licences "do not contain specific terms about the distribution of source code" and do not address patent rights. It does note that "CC licenses may be used for software documentation, as well as for separate artistic elements such as game art or music" — the steward's own carve-out is for non-code assets bundled with code, which is arguably the closest analogue to weights, and it cuts *toward* the licence applying.

### 5.4 What is left after all that

Assembling it: the publisher said "project," never clarified when asked twice, and archived the repo; the two redistributions disagree; the Apache side of the disagreement traces to a copy-paste; the CC-BY-NC side traces to a deliberate annotation by the person who did the port; and the prior question of whether weights are licensable at all is contested by the U.S. Copyright Office's own account.

That is **not** a finding that R4's checkpoint cannot ship. It is a finding that **nobody can tell you it can**, and that the person who would know is unreachable. For an artifact whose selling point is being cleanly open, that is the wrong kind of uncertainty to carry — the cost of removing it is one line of config change plus the quality delta in §6.

## 6. Alternatives

Only one candidate keeps R4 as specified. The rest change what the arm tests, and §6.2 argues that the change is smaller than the plan's framing implies.

### 6.1 The one that preserves the arm: OpenMMLab's MAE reproduction

OpenMMLab trained MAE ViT-B/16 on ImageNet-1k themselves, on 8×A100-80G, at four schedule lengths, and published all four.

| Model | Epochs | URL |
|---|---|---|
| `mae_vit-base-p16_8xb512-amp-coslr-300e_in1k` | 300 | [`…coslr-300e-fp16_in1k_20220829-c2cf66ba.pth`](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-300e_in1k/mae_vit-base-p16_8xb512-coslr-300e-fp16_in1k_20220829-c2cf66ba.pth) |
| `mae_vit-base-p16_8xb512-amp-coslr-400e_in1k` | 400 | [`…coslr-400e-fp16_in1k_20220825-bc79e40b.pth`](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-400e_in1k/mae_vit-base-p16_8xb512-coslr-400e-fp16_in1k_20220825-bc79e40b.pth) |
| `mae_vit-base-p16_8xb512-amp-coslr-800e_in1k` | 800 | [`…coslr-800e-fp16_in1k_20220825-5d81fbc4.pth`](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-800e_in1k/mae_vit-base-p16_8xb512-coslr-800e-fp16_in1k_20220825-5d81fbc4.pth) |
| `mae_vit-base-p16_8xb512-amp-coslr-1600e_in1k` | 1600 | [`…coslr-1600e_in1k_20220825-f7569ca2.pth`](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-1600e_in1k/mae_vit-base-p16_8xb512-fp16-coslr-1600e_in1k_20220825-f7569ca2.pth) |

Both the 800e and 1600e URLs were verified live (HTTP 200, 466 MB and 485 MB) on 2026-07-29.

**Licence evidence.** `open-mmlab/mmpretrain/LICENSE` is the Apache License 2.0 verbatim, headed `Copyright (c) OpenMMLab. All rights reserved`; the README states "This project is released under the [Apache 2.0 license](LICENSE)." `open-mmlab/mmselfsup` — the repo the download path is named for, since merged into mmpretrain — carries the identical LICENSE and the identical README sentence.

**Be honest about the symmetry here.** That is the *same* "This project is released under…" construction we have just spent §3.2 calling ambiguous for MAE. The difference is which way the ambiguity cuts: for MAE, the ambiguous statement is restrictive and the question is whether it reaches further than the code; for mmpretrain the ambiguous statement is permissive, so if it does *not* reach the weights, the weights simply have no stated licence — which is worse, not better, in the abstract. Two things make it materially safer anyway. First, **OpenMMLab trained these weights themselves** (metafile: `Training Resources: 8x A100-80G GPUs`), so whatever rights exist in them are OpenMMLab's to grant, and there is no upstream non-commercial restriction anywhere in the chain. Second, nothing in the OpenMMLab record signals a non-commercial intent, whereas the MAE record signals one and then refuses to scope it. *(Inference, marked.)* The residual risk is not zero; it is a different and much smaller shape.

Sources: https://github.com/open-mmlab/mmpretrain/blob/main/LICENSE · https://github.com/open-mmlab/mmpretrain/blob/main/README.md · https://github.com/open-mmlab/mmpretrain/blob/main/configs/mae/metafile.yml · https://github.com/open-mmlab/mmselfsup/blob/main/LICENSE

**What it costs — quantified, because it is not free.** OpenMMLab's reproduction matches the official one on fine-tuning and falls short on linear probing:

| Init | Epochs | ImageNet linear probe | ImageNet fine-tune |
|---|---|---|---|
| Official MAE (paper) | 800 | **68.0** | — |
| Official MAE (paper) | 1600 | — | **83.6** |
| OpenMMLab MAE | 300 | 60.8 | 83.1 |
| OpenMMLab MAE | 400 | 62.5 | 83.3 |
| OpenMMLab MAE | 800 | **65.1** | 83.3 |
| OpenMMLab MAE | 1600 | **67.1** | 83.5 |

Official numbers from arXiv 2111.06377 (linear probe 68.0 at 800e, Table 1 default setting; fine-tune 83.6 at 1600e, Table 3) and the repo README's 83.6 for ViT-B. OpenMMLab numbers from `configs/mae/README.md`.

Fine-tuning reproduces within 0.1–0.5 pt. **Linear probing is 2.9 pt low at matched 800 epochs** (65.1 vs 68.0), closing to 0.9 pt only by spending 1600 epochs. This is exactly the axis DEMAE cares about: plan §10 and #86 adjudicate R4-vs-R5 with kNN/linear probes, so a weaker frozen-feature init is a confound sitting directly on the measurement. **Use the 1600e checkpoint**, which is the closest to the official one on the probe axis and costs nothing extra to download.

**Integration cost for #102.** OpenMMLab's checkpoint uses mmselfsup's parameter naming and, at ~485 MB versus the official ~343 MB, ships the full encoder+decoder MAE rather than encoder-only *(inference from byte counts: 485 MB ÷ 4 ≈ 121 M floats against the metafile's 111.9 M parameters, versus 343 MB ÷ 4 ≈ 85.8 M against ViT-B's 85.8 M encoder parameters)*. So #102's loader needs a key-remap step it would not have needed for the official file, and the extra decoder weights are simply discarded — R4 attaches a fresh decoder either way. This is real work, and it is the honest price of the switch. #102 must assert full encoder coverage after remapping, which it already plans to do.

### 6.2 Permissively-licensed but non-MAE ViT-B/16 inits

| Candidate | Model id | Objective / data | Licence + evidence |
|---|---|---|---|
| DINO | [`timm/vit_base_patch16_224.dino`](https://huggingface.co/timm/vit_base_patch16_224.dino) | Self-supervised self-distillation, ImageNet-1k | `apache-2.0` on card; upstream `facebookresearch/dino` Apache-2.0; **no** `license=` override in timm's `_cfg` |
| AugReg (supervised) | [`timm/vit_base_patch16_224.augreg_in21k_ft_in1k`](https://huggingface.co/timm/vit_base_patch16_224.augreg_in21k_ft_in1k) | Supervised, IN-21k → IN-1k | `apache-2.0` on card; upstream `google-research/vision_transformer` LICENSE is Apache-2.0 |
| DeiT III | [`timm/deit3_base_patch16_224.fb_in1k`](https://huggingface.co/timm/deit3_base_patch16_224.fb_in1k) | Supervised, ImageNet-1k | `apache-2.0` on card; `facebookresearch/deit` LICENSE Apache-2.0, README: "This repository is released under the Apache 2.0 license" |

**What is lost, said plainly rather than waved through.** With a non-MAE init the arm stops being *continual* pretraining in the strict sense. The encoder's weights were never shaped by a masked-reconstruction objective, so R4 becomes "MAE pretraining from a differently-objective ImageNet init," and the specific prior evidence plan §8 cites — that "a frozen ImageNet MAE already beat a scratch UNet on DEM segmentation" ([arXiv 2309.03367](https://arxiv.org/abs/2309.03367)) — is about an MAE checkpoint and stops transferring cleanly. There is also a mechanical wrinkle: DINO and AugReg/DeiT-3 checkpoints are classification backbones, so a masked-reconstruction warm start inherits features tuned for global class discrimination rather than local reconstruction, and the first epochs may partly undo them.

**But the plan's own headline citation weakens the "MAE init is essential" case.** GFM ([arXiv 2302.04476](https://arxiv.org/abs/2302.04476)), the paper §8 leans on for the whole arm, does **not** continue from a masked-autoencoder init. Its teacher is an **ImageNet-22k supervised Swin-B**; the objective is masked image modelling plus feature distillation from that frozen supervised teacher; GeoPile is ~600k images; the run is 100 epochs / 93.3 V100-hours. So the result plan §8 cites as "continual pretraining beats from-scratch at almost exactly this corpus scale" is itself an experiment in warm-starting a *masked* objective from a *supervised* ImageNet init — and across a different architecture family at that. A supervised AugReg or DeiT-3 init is arguably **closer** to what GFM actually did than the official MAE checkpoint is.

*(Inference, marked, and offered for the map to argue with rather than as a settled view: this reframes the fallback. It does not make a non-MAE init equivalent to an MAE init, and §6.1 remains the better option because it changes nothing about the arm's identity. But if OpenMMLab's checkpoint proves painful to load, "R4 warm-starts from a supervised ImageNet ViT-B" is a defensible arm with the plan's own cited precedent behind it, not a consolation prize.)*

**Not viable, checked and rejected:** `eminorhan/mae_say_vitb16` and siblings are MIT-licensed MAE ViT-B/16, but pretrained on SAYCam infant head-camera video, not ImageNet — wrong data, so the "ImageNet prior" the arm is testing is absent. `BobMcDear/vit_base_patch16_mae_in1k_224` carries no licence tag at all and its card is a one-line pointer to the `flaim` repo; it appears to be a port of the same Facebook weights *(inference)*, so it inherits the problem rather than solving it. `facebook/dinov2-base` is Apache-2.0 but is ViT-B/**14**, not /16 — a different patch grid, which changes the sequence length and the patch-embed surgery.

## 7. What this means for the plan and the tickets

### Plan §8 (Initialization)

The sentence "initialize the encoder from the official ImageNet MAE ViT-B/16 checkpoint" names a specific artifact whose licence the plan's own §1 already flagged in the adjacent breath and then did not follow through on. **This is a plan amendment or an ADR, not a silent edit** — per #101's own instruction. The amendment should name the replacement checkpoint (recommend `mae_vit-base-p16_8xb512-amp-coslr-1600e_in1k`), state the ~2.9 pt linear-probe delta of the reproduction as an accepted cost, and record that the official checkpoint was rejected on licence ambiguity rather than on quality. Plan §1's "CC-BY-NC 4.0 … is a poor foundation for an open-weights release" should also be corrected on the copyleft point: the licence has no ShareAlike term, and the real bar is the NonCommercial purpose restriction plus §3(a)(4).

### Plan §14 (Interface to Phase 2)

§14 promises, per released run, "encoder-only weights (timm-loadable) + full MAE checkpoint + config + fingerprints + training curves." If the official checkpoint is kept, R4 cannot honour that promise on the same terms as the other runs. If §6.1's substitution is adopted, §14 is unchanged and needs only a provenance line naming the init checkpoint and its licence — which §14 should arguably carry regardless, since "what was this initialized from" belongs next to `dataset_version` and the split md5 in any honest artifact.

### #86 — "Run R4: the ViT-B continual arm from ImageNet MAE"

#86's own text already anticipates this: "if the checkpoint cannot ship, this arm still runs and still reports, but #88 packages it differently and the resolution must say so." Three concrete changes:

1. **The title and body name the wrong checkpoint.** #86 says "warm-started via #102 from the ImageNet MAE ViT-B/16 checkpoint" — under the recommendation that becomes OpenMMLab's 1600e MAE ViT-B/16, and #86's pre-flight verification ("re-check it on the real checkpoint on the box") must be against that file, whose remapped key coverage is a new thing to assert.
2. **The scratch-vs-continual headline survives intact** — same objective, same architecture, same ImageNet-1k data. That is the point of choosing §6.1 over §6.2. What #86 must add is a one-line caveat when reporting: the init is an independent MAE reproduction that trails the original by ~2.9 pt ImageNet linear probe at 800e (0.9 pt at 1600e), so a *narrow* R4-vs-R5 gap should not be over-read in either direction.
3. **#86's parenthetical "No ViT-S MAE checkpoint exists; that is why this arm runs at B" still holds.** OpenMMLab publishes MAE at ViT-B, ViT-L and ViT-H only — no ViT-S. The arm stays at B for the same reason.

### #88 — "Package the Phase-1 release artifacts"

1. **Under the recommendation, #88 does not need a special case for R4.** Its "per released run: encoder-only weights (timm-loadable) + full MAE checkpoint" list applies uniformly, and the scratch-vs-continual write-up ships as planned.
2. **#88 must add an init-provenance field to the artifact set.** Every released run needs to state what its encoder was initialized from and under what licence — `scratch` for R1/R2/R3/R5, and the OpenMMLab checkpoint URL + Apache-2.0 for R4. #88 already fingerprints `dataset_version` and the split assignment; the init is the same class of fact and is currently unrecorded.
3. **If the map instead decides to keep the official checkpoint**, #88 becomes the ticket that carries the split: R4's checkpoint is withheld or released under separate non-commercial terms with attribution to He et al. per §5.1–5.2, while R4's *numbers* ship normally in the scratch-vs-continual readout. #88's release framing must then say out loud that one artifact in the release is not under the same licence as the rest — which is precisely the "materially different deliverable" #101 warned about.

## 8. What we checked that did **not** answer the question

Recorded so this is not re-litigated from scratch, and so the shape of the gap is visible:

- **`facebookresearch/mae/LICENSE`** — verbatim CC BY-NC 4.0 legal code with no Meta-authored scope note. Tells you *which* licence, not *what it covers*.
- **The repo's full file listing** — no `MODEL_CARD.md`, no `LICENSE_WEIGHTS`, no `LICENSE_*` of any kind, no release notes.
- **`CONTRIBUTING.md`** — inbound-contribution licensing only ("source tree"), silent on outbound artifacts.
- **`PRETRAIN.md` / `FINETUNE.md`** — recipes and checkpoint tables, zero licence text.
- **All 219 issues in the repo**, searched for `license`, `licence`, `commercial`, `commercial use`, and via `--match comments` — three hits, **no maintainer reply on any of them**, and the one comment that exists is an anonymous user's guess.
- **GitHub Discussions on the repo** — disabled (`has_discussions: false`).
- **The repo's live status** — archived, last commit 2022-04-20. There is no route to a clarification.
- **arXiv 2111.06377 (v1–v3)** — paper is CC BY 4.0; one sentence about releasing code; **nothing about model licensing**.
- **`dl.fbaipublicfiles.com`** — no terms page (`/terms`, `/terms.html`, `/LICENSE`, `/license.txt` all 403), no licence HTTP header, bucket root `AccessDenied`.
- **Hugging Face discussions on both `facebook/vit-mae-base` and `timm/vit_base_patch16_224.mae`** — five threads total, none about licensing.
- **The `facebook/vit-mae-base` commit history** — 11 commits, no identifiable Meta/FAIR author; the `apache-2.0` tag traces to a template copy-paste and has never been reviewed.
- **A Hugging Face model search** for other MAE ImageNet ViT-B/16 releases under Apache/MIT — found nothing usable beyond OpenMMLab (details and rejections in §6.2).
- **The U.S. Copyright Office Part 3 report** — establishes that weights' copyright status is contested, but addresses whether weights infringe *training data*, not whether a publisher holds a licensable copyright in them. Adjacent, not dispositive.
- **The Creative Commons FAQ** — CC recommends against CC licences for software, but its own carve-out for non-code assets bundled with code arguably cuts *toward* the licence reaching weights. Ambiguous either way.

**What would actually resolve it** and was not attempted, because it is out of scope for a research ticket: a direct written enquiry to Meta / FAIR, or to the paper's corresponding authors. If the map wants the official checkpoint badly enough, that is the one remaining move — and it should be made before R4 runs, not after.

## Sources

**Primary — the publisher**
- `facebookresearch/mae` LICENSE (verbatim CC BY-NC 4.0): https://github.com/facebookresearch/mae/blob/main/LICENSE
- `facebookresearch/mae` README ("This project is under the CC-BY-NC 4.0 license"; checkpoint table): https://github.com/facebookresearch/mae/blob/main/README.md
- `facebookresearch/mae` CONTRIBUTING.md: https://github.com/facebookresearch/mae/blob/main/CONTRIBUTING.md
- `facebookresearch/mae` PRETRAIN.md / FINETUNE.md: https://github.com/facebookresearch/mae/blob/main/PRETRAIN.md · https://github.com/facebookresearch/mae/blob/main/FINETUNE.md
- Unanswered licence issues: [#158](https://github.com/facebookresearch/mae/issues/158) · [#171](https://github.com/facebookresearch/mae/issues/171) · [#208](https://github.com/facebookresearch/mae/issues/208)
- Checkpoint URL and HTTP headers: https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth
- Paper: https://arxiv.org/abs/2111.06377

**Primary — the redistributions**
- `facebook/vit-mae-base` (card, `license: apache-2.0`, HF-authored disclaimer): https://huggingface.co/facebook/vit-mae-base
- First card revision `07a35ff` with stray `tags: - dino`: https://huggingface.co/facebook/vit-mae-base/raw/07a35ffdfd539ebd43f85caca2557568f3817f24/README.md
- `facebook/dino-vitb16` (the Apache-2.0 sibling the template came from): https://huggingface.co/facebook/dino-vitb16
- `timm/vit_base_patch16_224.mae` (card, `license: cc-by-nc-4.0`, "Original: facebookresearch/mae"): https://huggingface.co/timm/vit_base_patch16_224.mae
- timm source, per-weight `license='cc-by-nc-4.0'` annotation: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py

**Primary — licence text and legal status**
- CC BY-NC 4.0 legal code (definitions §1, grant §2, conditions §3): https://creativecommons.org/licenses/by-nc/4.0/legalcode
- Creative Commons FAQ, "Can I apply a Creative Commons license to software?": https://creativecommons.org/faq/
- U.S. Copyright Office, *Copyright and AI, Part 3: Generative AI Training* (Pre-Publication, May 2025), p. 29: https://www.copyright.gov/ai/Copyright-and-Artificial-Intelligence-Part-3-Generative-AI-Training-Report-Pre-Publication-Version.pdf

**Primary — alternatives**
- mmpretrain LICENSE (Apache-2.0) and README §License: https://github.com/open-mmlab/mmpretrain/blob/main/LICENSE · https://github.com/open-mmlab/mmpretrain/blob/main/README.md
- mmselfsup LICENSE and README §License: https://github.com/open-mmlab/mmselfsup/blob/main/LICENSE
- mmpretrain MAE configs — metafile (training resources, weight URLs) and README (results table): https://github.com/open-mmlab/mmpretrain/blob/main/configs/mae/metafile.yml · https://github.com/open-mmlab/mmpretrain/blob/main/configs/mae/README.md
- `facebookresearch/deit` LICENSE + README ("This repository is released under the Apache 2.0 license"): https://github.com/facebookresearch/deit
- `google-research/vision_transformer` LICENSE (Apache-2.0): https://github.com/google-research/vision_transformer/blob/main/LICENSE
- GFM (init is ImageNet-22k supervised Swin-B; ~600k images; 100 epochs; 93.3 V100-h): https://arxiv.org/abs/2302.04476
- DEM MAE transfer result cited by plan §8: https://arxiv.org/abs/2309.03367

All URLs retrieved 2026-07-29.
