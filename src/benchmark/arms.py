"""The arm registry — one entry per row of the plan's §1 table (#147).

Each arm declares four things and no behaviour: **its encoder**, **its trainable part**, **its
input convention** and **its normalization**. The ``run`` verb dispatches on it, the resolution
line prints it, and `benchmark.config` validates ``arm`` against it — so an arm that is not
here is not a thing this benchmark can be asked to score.

Declarative rather than executable on purpose. Every failure this registry exists to prevent is
silent: an encoder fed the wrong normalization returns a plausible tensor and raises nothing
(p118 §7), a ``pos_embed`` loaded with the wrong axis order trains fine (ADR-0011, P102 §4).
Those are facts about an arm, so they live where the arm is named rather than inside the code
that later loads a checkpoint.

## Eleven printed rows and one appendix cell

:data:`PRINTED_ROWS` is the table's shape, fixed by
[#136](https://github.com/JStoschek/DEMAE/issues/136) and registered as **R5**. A2, A3 and A4
each split R5/R4 — §4.2's release checkpoint and its pre-registered secondary — which is what
takes §1's eight arms to eleven rows. §2.3's ViT-S cell (``A4-S``) is registered here too and
carries ``printed=False``: it is a capacity-scaling aside no §0 claim rests on, run at one
seed, reported without a ±, and it does not widen a pre-registered table.

## A6-S was registered here and is gone

§2.3 registered a ViT-S *pair* until
[#147](https://github.com/JStoschek/DEMAE/issues/147). The build checked
``pretrain.config.Init``'s claim that no ViT-S ImageNet MAE checkpoint exists at any
provenance, found the claim wrong — ``vit-s_imagenet_mae.pth`` from ``lambert-x/medical_mae``
is real, Apache-2.0, and timm-loadable — and dropped the arm anyway, because that checkpoint
publishes no epoch count, no confirmation of which ImageNet, and no quality number on the
axis `p101` §6.1 used to choose A6's own ViT-B init. An A6-vs-A6-S delta would have
confounded capacity with trainer and schedule. Absent rather than disabled, for the reason
below; the evidence is `docs/research/p147-vit-s-mae-checkpoint.md`, and re-adding it is one
encoder entry, one arm entry and one `Paths` field.

## A8 and A10 are absent rather than disabled

Both cross-evals were ruled out of scope at charting
([#128](https://github.com/JStoschek/DEMAE/issues/128)) and §0 now carries the strong form —
incomparable, and we looked. A disabled registry entry would put a ruled-out arm back on a
surface someone reads, so there is deliberately no entry and no special-cased message: they
fail the way any other unknown name does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

# Stdlib-only itself, and it does not import this module — so R4's value is read from the
# register without putting torch on `freeze`'s or A1's import path (§3.4).
from . import prereg

# --- Vocabularies -------------------------------------------------------------

#: What an arm trains. ``nothing`` is A1's classical fill and every inference-only row; ``head``
#: is the §4.1 shared head over a frozen encoder; ``encoder+decoder`` is A3's native fine-tune;
#: ``everything`` is A7 from scratch.
TRAINABLE: Final = ("nothing", "head", "encoder+decoder", "everything")

#: How the void reaches the model. §2.4's convention for the frozen arms, native token dropping
#: for DEMAE's own arms, and the tokenizer's full encode/FSQ/decode for A9.
INPUT_CONVENTIONS: Final = (
    "classical_fill",
    "a1_fill_plus_mask",
    "token_drop",
    "a1_fill_through_tokenizer",
)

#: §2.1 — each model is fed its native diet, and which diet is a *declared* field because the
#: TerraMind encoder accepts either silently (p118 §7). ``none`` is an arm with no encoder.
NORMALIZATIONS: Final = ("none", "per_tile", "terramind_v1")

#: How the position embedding the model *runs with* is derived from the one the checkpoint
#: ships. Distinct from :attr:`Encoder.pos_encoding`, which names the closed form the shipped
#: tensor is **checked against**: two encoders can agree on the form and disagree on the
#: derivation, and ADR-0011 is exactly that case.
#:
#: ``as_published`` is the identity — the checkpoint's tensor is already at the run's grid, which
#: is true of every 224 px-native ViT here. ``rederived_14x14_crop`` is ADR-0011 Decision 2's
#: centre-anchored crop of TerraMind's native 16x16 grid. ``bicubic_resize`` is terratorch's own
#: default path, rejected at 0.9708 against #134's 0.99 bar and kept expressible **only** because
#: ADR-0011 Decision 4 pre-registers an A5-at-20k ablation row against it (ADR-0013 R11) — a
#: derivation the contract cannot express is an ablation nobody can run.
POS_EMB_DERIVATIONS: Final = ("as_published", "rederived_14x14_crop", "bicubic_resize")

#: ADR-0011's two token spacings, and the one sentence its Consequences section says the writeup
#: owes: *"we now diverge from the library's default inference path, and the writeup must say so
#: plainly — one sentence, with 1.1429-vs-1.0 attached."*
#:
#: `align_corners=False` preserves *normalized* position, so terratorch's default resize samples
#: the native 16-grid at indices 0.071 … 14.929 — spacing 1.1429 where a crop's is exactly 1.0.
#: The model is told its patches sit ~14% further apart than they do, and because the stretch is
#: systematic rather than noisy the error accumulates past a full patch width by the far edge.
BICUBIC_TOKEN_SPACING: Final = 1.1429
CROP_TOKEN_SPACING: Final = 1.0


def pos_emb_citation(encoder: Encoder) -> str | None:
    """ADR-0011's divergence sentence for one encoder, or ``None`` where there is no divergence.

    Rendered from :attr:`Encoder.pos_emb_derivation` rather than carried as a field per entry —
    :func:`benchmark.extract._encoder_note`'s rule, "written from the fact rather than copied per
    entry", and the reason R4's citation *is* a field is the reverse of it: an epoch count is a
    fact about one published artifact that nothing in this registry could derive, where this
    sentence is a function of a derivation the registry already declares.

    The two encoders say opposite things and both have to be sayable. A5's line is the divergence
    — this is not what terratorch does, and here is the arithmetic. A5-BICUBIC's line is that it
    *is* what terratorch does, run only because ADR-0011 D4 pre-registered a row against it, so a
    reader who finds its numbers cannot mistake them for the benchmark's own.

    ``None`` for ``as_published``, which is every 224 px-native ViT here: the checkpoint's tensor
    is already at the run's grid, nothing is derived, and there is no library path to diverge from.
    """
    if encoder.pos_emb_derivation == "rederived_14x14_crop":
        return (
            f"diverges from terratorch's default inference path on purpose: ADR-0011 D2 takes "
            f"the centre-anchored {encoder.pos_emb_native_grid_px - 2}x"
            f"{encoder.pos_emb_native_grid_px - 2} crop of the native "
            f"{encoder.pos_emb_native_grid_px}x{encoder.pos_emb_native_grid_px} grid, token "
            f"spacing exactly {CROP_TOKEN_SPACING}. terratorch resizes it bicubically instead, "
            f"sampling at spacing {BICUBIC_TOKEN_SPACING} — ~14% too wide, accumulating past a "
            f"full patch by the far edge. The embedding is fixed sin-cos, so the crop *is* the "
            f"closed form at the smaller grid and nothing is approximated"
        )
    if encoder.pos_emb_derivation == "bicubic_resize":
        return (
            f"terratorch's own default inference path, at token spacing "
            f"{BICUBIC_TOKEN_SPACING} against a crop's {CROP_TOKEN_SPACING}. Run **only** as "
            f"ADR-0011 D4's pre-registered ablation (R11), never as a way to score an arm — "
            f"these numbers are the thing the benchmark's own path is evidenced against"
        )
    return None


#: What the closed form a checkpoint's position tensor is **checked against** can be. A tuple
#: like its three siblings above rather than a free string, because #157 added a third value to
#: a field that had two and no vocabulary — and an unregistered order is a checkpoint nothing
#: verifies, which is the one tensor P102 §4 says must be verified rather than trusted.
#: ``sincos2d`` and ``sincos2d_rowfirst`` are :data:`pretrain.model.AXIS_ORDERS`' two;
#: ``sincos2d_terramind`` is terratorch's `build_2d_sincos_posemb`, which has its own oracle.
POS_ENCODINGS: Final = ("sincos2d", "sincos2d_rowfirst", "sincos2d_terramind")


@dataclass(frozen=True)
class Encoder:
    """A frozen encoder's identity: what it is, how wide, and which file it is loaded from.

    ``checkpoint_field`` names a field of :class:`benchmark.config.Paths` rather than holding a
    path, because *where* a checkpoint is mounted is environment (a rented box mounts it
    somewhere else) while *which* encoder an arm reads is the arm — and the arm is hashed.
    ``checkpoint_md5`` is the other half of that split and is **not** environment: it is the
    one fact that distinguishes this encoder's weights from a sibling's, so it is declared here
    and enforced by :mod:`benchmark.extract` (ADR-0012 §7.1) rather than remembered.
    """

    #: Registry key, and what the resolution line prints.
    name: str
    #: ``demae`` | ``terramind`` | ``imagenet-mae``. What CLS means, among other things: it is
    #: DEMAE-family only and never enters a cross-encoder comparison (ADR-0012 §5).
    family: str
    #: Feature width — the only thing A4/A5/A6's shared head differs by (§4.1).
    width: int
    #: §2.3's precondition, declared rather than remembered. ``sincos2d`` is MAE's own
    #: column-first order; ``sincos2d_rowfirst`` is mmselfsup's, which R4's weights learned;
    #: ``sincos2d_terramind`` is `build_2d_sincos_posemb`'s, the MoCo-v3 form TerraMind's
    #: ``[1, D, 16, 16]`` parameter is a closed form of. What the shipped tensor **is**.
    pos_encoding: str
    #: The :class:`benchmark.config.Paths` field naming this encoder's checkpoint.
    checkpoint_field: str
    #: The :class:`benchmark.config.Paths` field naming this encoder's **pretraining**
    #: checkpoint — encoder *and* decoder — or ``None`` for a family that has none.
    #:
    #: A second field rather than a second value of :attr:`checkpoint_field`, because on a given
    #: box they are two different states of two different files. §14 exports encoder-only weights
    #: and discards the decoder, so the release `encoder.pt` every other arm initializes from
    #: cannot drive A2: zero-shot inpainting *is* the pretrained decoder (§4.3). Declared here so
    #: the resolution line names the file before a run spends anything, rather than a `load` deep
    #: inside the pass discovering that 74 tensors are absent.
    mae_checkpoint_field: str | None = None
    #: One of :data:`POS_EMB_DERIVATIONS` — how the tensor the model *runs with* is derived from
    #: the one above. ADR-0011 Decision 2 is a value of this field and not a line in extraction
    #: code, because it is a fact about this encoder that a reader has to be able to find from
    #: the committed contract: the nine possible crops span 44% relative RMS, more than the
    #: bicubic error being removed, so the anchor is as load-bearing as the derivation.
    pos_emb_derivation: str = "as_published"
    #: The token grid this encoder's position tensor is **published at**, where that is not the
    #: grid a 224 px tile implies. Declared per encoder rather than being a constant a derivation
    #: implies: 16 is a fact about TerraMind's 256 px training resolution, and a second encoder
    #: reaching this grid from a third one would otherwise be silently told it was 16 too.
    #: ``None`` is the identity case and is checked against ``as_published`` below.
    pos_emb_native_grid_px: int | None = None
    #: §2.3's table, where it gives one. ``None`` where no source in this plan counts them, and
    #: left ``None`` rather than inferred: §6 item 4 prints parameter counts, and a count this
    #: registry guessed would print as a measurement.
    #:
    #: The encoder **as published**. TerraMind's is p118 §8's 90,172,416, and the model
    #: :mod:`benchmark.extract` runs holds 46,080 fewer — ADR-0011 substitutes the 14x14 crop for
    #: the native 16x16 ``pos_emb``, all of it frozen position and none of it trainable. p157
    #: records both with the arithmetic checked, so the two numbers are reconciled somewhere
    #: rather than looking like a disagreement.
    params: int | None = None
    #: Transformer blocks. ADR-0012 §3 emits one intermediate per block, so a declared count
    #: that disagrees with the checkpoint is refused rather than absorbed by a shape error.
    depth: int = 12
    #: The published architecture this checkpoint *is*, as a consumer types it into
    #: ``timm.create_model``. Quoted from :data:`pretrain.release.TIMM_MODELS` rather than
    #: rebuilt from the width, and pinned against it by a test.
    timm_model: str = "vit_base_patch16_224"
    #: How the file is laid out. ``timm`` is a §14 release export, which loads **strictly**;
    #: ``mmselfsup_rgb`` is the published ImageNet MAE, whose `backbone.`-prefixed keys and
    #: 3-channel patch embed §8's collapse translates (:mod:`pretrain.init` owns that map).
    checkpoint_format: str = "timm"
    #: The authoritative file's md5, where this plan has one. ``None`` is *unpinned*, and
    #: `freeze` then records whatever this box mounted with its source stated — see
    #: :func:`benchmark.extract.build_spec`. Never guessed.
    checkpoint_md5: str | None = None
    #: Whether the architecture has a CLS token at all. ADR-0012 §5 makes it an optional side
    #: field for exactly this reason: TerraMind structurally has none, so a statistic pooled
    #: over it would compare one model's token to the other's absence.
    #:
    #: §5 says "DEMAE-family only", written when the contract had two encoders in view;
    #: ImageNet-MAE is a ViT and structurally has one. ADR-0012's own amendment note under
    #: Decision 5 settles that (#156) — this field is what it settles *to*, and the ADR is where
    #: the reasoning lives, so the decision has one home.
    has_cls: bool = True
    #: Whether this encoder's own final tensor is its last block's output put through a norm.
    #: DEMAE norms after block twelve and TerraMind does not (ADR-0012 §3), which is the whole
    #: reason `output` and `intermediates[11]` are separately named.
    output_is_normed: bool = True
    #: How many epochs the *published* checkpoint was pretrained for, where the publisher states
    #: a figure this plan is obliged to quote. ADR-0013 **R4** is the whole reason this is a
    #: field: TerraMind's paper headlines 100 epochs, which is its S-2/S-1 figure, and the DEM
    #: modality this benchmark reads got **20** (#138). The wrong one is the one a reader reaches
    #: for, it is off by 5x, and it flatters the baseline — so it rides with the encoder and is
    #: printed wherever the arm is described, rather than being remembered by whoever writes the
    #: sentence.
    #:
    #: ``None`` is *this plan states none*, which is the honest answer for DEMAE's own encoders
    #: (their epoch count is a run record's, not a published artifact's) and for the ImageNet MAE
    #: (whose 1600 is in its filename and in `pretrain.init`, and which no registered rule
    #: governs). Never defaulted to a number — R4 is a citation, and an inherited one is not.
    #:
    #: **Deliberately not a field of** :class:`benchmark.extract.EncoderSpec`, so ADR-0012's
    #: committed contract carries this encoder's `pos_emb` derivation and not its epoch count.
    #: That contract governs what comes *out of a frozen encoder* — twelve intermediates at a
    #: declared dtype under a declared diet — and an epoch count changes none of it, where the
    #: derivation changes every tensor in it. R4's homes are §2.3, §6.4 and §10: prose a reader
    #: reads, which is what the run record and the resolution block are and the contract is not.
    pretraining_epochs: int | None = None
    #: The per-modality citation itself, in the words R4 requires it be stated in. Beside the
    #: integer rather than derived from it, because what R4 governs is the *sentence*: a bare
    #: `20` in a record is a number a reader has no way to tell from the paper's headline.
    pretraining_epochs_citation: str | None = None

    def __post_init__(self) -> None:
        if self.pos_encoding not in POS_ENCODINGS:
            raise ValueError(
                f"encoder {self.name}: unknown pos_encoding {self.pos_encoding!r}; "
                f"registered: {sorted(POS_ENCODINGS)}"
            )
        if self.pos_emb_derivation not in POS_EMB_DERIVATIONS:
            raise ValueError(
                f"encoder {self.name}: unknown pos_emb derivation {self.pos_emb_derivation!r}; "
                f"registered: {sorted(POS_EMB_DERIVATIONS)}"
            )
        # The two fields are one claim in two halves, and the halves have to agree: a derivation
        # that is the identity has no *other* grid to name, and one that is not has nothing to
        # derive from without it.
        published = self.pos_emb_derivation == "as_published"
        if published != (self.pos_emb_native_grid_px is None):
            raise ValueError(
                f"encoder {self.name}: {self.pos_emb_derivation!r} with "
                f"pos_emb_native_grid_px={self.pos_emb_native_grid_px!r}. An `as_published` "
                f"derivation names no native grid because it is the run's; every other one names "
                f"the grid it derives *from* (ADR-0011)"
            )
        # ADR-0013 R4's two halves, on the pair above's rule: what R4 governs is the *sentence*,
        # so a figure with no citation is a number a reader cannot tell from TerraMind's headline
        # and a citation with no figure states nothing. Neither half is usable alone, which is
        # what makes this a check rather than two independent optional fields.
        if (self.pretraining_epochs is None) != (self.pretraining_epochs_citation is None):
            raise ValueError(
                f"encoder {self.name}: pretraining_epochs={self.pretraining_epochs!r} with "
                f"pretraining_epochs_citation={self.pretraining_epochs_citation!r}. R4 asks for "
                f"the figure and the per-modality citation together or neither (ADR-0013)"
            )


#: The `-v2` release bundles' `encoder.pt` md5s, and the answer to "which bundle is
#: authoritative" (#156). Both R5 bundles ship a **byte-identical** `encoder.pt`, and so do both
#: R4 bundles — `-v2` re-exported the *bundle* (it adds `LICENSE` and the manifest's `licence`
#: block; #101's licence work) and not the weights. So the question the ticket asks has no
#: weights axis at all: these md5s identify the R5 and R4 encoders under either directory, and
#: `-v2` is the authoritative bundle because it is the one that carries its own terms.
#: :mod:`benchmark.config`'s `Paths` points at `-v2` for that reason.
DEMAE_R5_MD5: Final = "3ec5ebc72fa28c43fe88f67692d9b41a"
DEMAE_R4_MD5: Final = "59e8346c4b08e1ab4fedd44f0802e12b"

#: `mae_vit-base-p16_8xb512-fp16-coslr-1600e_in1k_20220825-f7569ca2.pth` — the published
#: mmselfsup ImageNet-1k MAE that A6 reads and that R4 warm-started from.
#:
#: The value of `pretrain.init.MMSELFSUP_MAE_VIT_B16_1600E.md5`, **quoted rather than imported**,
#: and pinned against it by a test — :attr:`Encoder.timm_model`'s rule, applied for
#: :attr:`Encoder.timm_model`'s reason. Importing it would put ``pretrain.init`` on this
#: module's import path, and that module imports torch at its top; this registry is read by
#: `freeze` and by A1's scoring path, which §3.4 keeps torch-free so they run on a box with no
#: CUDA stack. A test is the cheap half of that trade and the copy is the dear half.
#:
#: The spec it is quoted from is where the rest of this file's identity lives — the URL, the
#: 484,833,387 bytes, the Apache-2.0 licence and that licence's source — because P101 spent a
#: ticket establishing that the licence is why this checkpoint was chosen.
IMAGENET_MAE_MD5: Final = "91ecc0345a6d6ecefe9c59041841f180"

#: `TerraMind_Tokenizer_DEM.pt` — the whole 1,145,511,376-byte tokenizer, of which A5 and A9 read
#: the 90.2 M-parameter `encoder.` half (p118 §8). Published rather than observed, on
#: :attr:`Encoder.checkpoint_md5`'s rule: the file is public, Apache-2.0 and ungated, at
#: `ibm-esa-geospatial/TerraMind-1.0-Tokenizer-DEM` revision `aba0a56` (p118 §3), so its identity
#: is a reviewed line here rather than something a box certifies by having downloaded it. #138
#: is why that matters here more than elsewhere: **three** DEM constant pairs ship across IBM's
#: artifacts, and the v0.1 tokenizer is a different file with the same shape and a different diet.
TERRAMIND_TOKENIZER_DEM_MD5: Final = "66b7fbcb81e04684df31873623d7a3de"

#: ADR-0013 **R4**'s epoch count, read out of the register rather than re-typed here.
#:
#: `benchmark.prereg` is explicit that past it "no threshold in the benchmark is typed as a
#: literal", and R4's `20` is one of its seven numbered rules — so this is a lookup where
#: :data:`IMAGENET_MAE_MD5` and :attr:`Encoder.timm_model` are quotes-pinned-by-a-test. The
#: difference is that those two quote a module importing **torch** at its top, and §3.4 keeps
#: this registry torch-free so `freeze` and A1's scoring path run on a box with no CUDA stack.
#: `prereg` imports only stdlib and does not import this module, so there is no cost to pay and
#: no cycle to dodge: `zeroshot.py` already reads R6 and R12 exactly this way.
TERRAMIND_DEM_EPOCHS: Final = prereg.rule("R4").value

#: The sentence R4 requires the figure be stated in. Here rather than in the register because
#: what `prereg` holds is the *rule* — "carry the per-modality citation wherever stated" — and
#: what this is, is the citation being carried. The 100 is named on purpose: R4 is a
#: disambiguation, and a citation reading only "20 epochs" leaves TerraMind's 5x-larger headline
#: standing as the figure a reader supplies for themselves.
TERRAMIND_DEM_EPOCHS_CITATION: Final = (
    f"{TERRAMIND_DEM_EPOCHS} epochs over the DEM modality (TerraMind 1.0 DEM tokenizer's "
    f"recorded training args); the paper's 100 is its S-2/S-1 figure — ADR-0013 R4, #138"
)

#: The encoders §1 and §2.3 name. TerraMind's 4.92M excess over DEMAE's ViT-B is reported and
#: not attributed (§2.3): terratorch was not installed in any live worktree when it was written.
ENCODERS: Final[dict[str, Encoder]] = {
    "demae-r5": Encoder(
        name="demae-r5",
        family="demae",
        width=768,
        pos_encoding="sincos2d",
        checkpoint_field="demae_r5_checkpoint",
        mae_checkpoint_field="demae_r5_mae_checkpoint",
        params=85_254_144,
        checkpoint_md5=DEMAE_R5_MD5,
    ),
    "demae-r4": Encoder(
        name="demae-r4",
        family="demae",
        width=768,
        # R4 warm-started from mmselfsup's ImageNet MAE, whose fixed sin-cos embedding is the
        # *transpose* of MAE's (P102 §4). Every name and every shape matches under either.
        pos_encoding="sincos2d_rowfirst",
        checkpoint_field="demae_r4_checkpoint",
        mae_checkpoint_field="demae_r4_mae_checkpoint",
        params=85_254_144,
        checkpoint_md5=DEMAE_R4_MD5,
    ),
    "demae-vits": Encoder(
        name="demae-vits",
        family="demae",
        width=384,
        pos_encoding="sincos2d",
        checkpoint_field="demae_vits_checkpoint",
        params=21_393_408,
        timm_model="vit_small_patch16_224",
        # No §2.3 ViT-S run has been released, so there is no file to pin. `freeze` records the
        # md5 of whatever a box mounts, stating that it did — see `benchmark.extract`.
        checkpoint_md5=None,
    ),
    "terramind": Encoder(
        name="terramind",
        family="terramind",
        width=768,  # the 768-d `feats`, pre-`quant_proj` (§1)
        # What the checkpoint ships is `build_2d_sincos_posemb(16, 16)` — a closed form of the
        # token's *integer* grid index, in the checkpoint at all only because it is wrapped in an
        # `nn.Parameter` (ADR-0011). So it is verified against that form rather than trusted.
        pos_encoding="sincos2d_terramind",
        # ADR-0011 Decision 2: the centre-anchored 14x14 crop of that 16x16 grid, so both
        # encoders are in-distribution at 224 px simultaneously. terratorch's default bicubic
        # resize scored 0.9708 against #134's 0.99 materiality bar and was rejected; §4's
        # A5-at-20k ablation row evidences it.
        pos_emb_derivation="rederived_14x14_crop",
        # 256 px at patch 16: the DEM tokenizer's recorded training args say
        # `input_size_min=256, input_size_max=256`, so `pos_emb` ships [1, 768, 16, 16].
        pos_emb_native_grid_px=16,
        checkpoint_field="terramind_checkpoint",
        params=90_172_416,
        checkpoint_format="terratorch",
        checkpoint_md5=TERRAMIND_TOKENIZER_DEM_MD5,
        # p118 §5.3 measured it: `return_intermediates=True` yields twelve tensors with no
        # separate post-norm entry, its last bit-identical to `encoder(x)`. So "the last layer"
        # means something different here than it does for DEMAE, which is ADR-0012 §3's subject.
        output_is_normed=False,
        has_cls=False,
        pretraining_epochs=TERRAMIND_DEM_EPOCHS,
        pretraining_epochs_citation=TERRAMIND_DEM_EPOCHS_CITATION,
    ),
    # ADR-0011 Decision 4's pre-registered ablation, as an encoder rather than as a flag
    # (ADR-0013 R11). Byte-identical weights to `terramind` — same field, same md5, same diet —
    # under terratorch's own default `F.interpolate(pos_emb, size=(14, 14), mode="bicubic")`
    # instead of Decision 2's crop. #134 measured the two at 0.9708 mean per-token cosine against
    # a 0.99 bar, so the row stays and this entry is what lets it be run.
    #
    # A second registry entry rather than a knob on `A5`, and `terramind-tokenizer` below is the
    # precedent: the committed contract keys its encoders by name, `benchmark.extract.Extractor`
    # caches built models by name, and `benchmark.results.append_rows` replaces rows by
    # `(arm, size, seed)`. An override that left both cells calling themselves `A5` would have
    # the ablation *delete* the row it was registered to sit beside — silently, since a replace
    # leaves nothing behind in the parquet to say it happened.
    "terramind-bicubic": Encoder(
        name="terramind-bicubic",
        family="terramind",
        width=768,
        pos_encoding="sincos2d_terramind",
        pos_emb_derivation="bicubic_resize",
        pos_emb_native_grid_px=16,
        checkpoint_field="terramind_checkpoint",
        params=90_172_416,
        checkpoint_format="terratorch",
        checkpoint_md5=TERRAMIND_TOKENIZER_DEM_MD5,
        output_is_normed=False,
        has_cls=False,
        pretraining_epochs=TERRAMIND_DEM_EPOCHS,
        pretraining_epochs_citation=TERRAMIND_DEM_EPOCHS_CITATION,
    ),
    "imagenet-mae": Encoder(
        name="imagenet-mae",
        family="imagenet-mae",
        width=768,
        pos_encoding="sincos2d_rowfirst",
        checkpoint_field="imagenet_mae_checkpoint",
        # The published mmselfsup file, RGB and `backbone.`-prefixed. §8's collapse and P102's
        # key map turn it into the single-channel timm encoder A6 reads; `pretrain.init` owns
        # both, and `benchmark.extract` reuses them rather than writing a second map.
        checkpoint_format="mmselfsup_rgb",
        # **Measured 2026-09-01 (#159)**, on the encoder `benchmark.extract` actually builds, and
        # it is the published count minus §8's collapse rather than a number this registry
        # guessed: mmselfsup publishes 85,798,656 for the RGB backbone
        # (`pretrain.init.MMSELFSUP_MAE_VIT_B16_1600E.encoder_parameters`), and folding the
        # 3-channel patch embed down to one drops 768x(3-1)x16x16 = 393,216 of them. A test pins
        # that arithmetic, so this stays a derivation and not a transcription.
        #
        # It was `None` until A6's first real cells ran, and §6 item 4 prints parameter counts
        # **total and trainable** — so A6's row was printing an em-dash where A4's prints 86.4M,
        # which is the misreading that item exists to prevent.
        params=85_405_440,
        # **Pinned 2026-09-01 (#159).** This entry read "unpinned ... that md5 is the 485 MB
        # *full* MAE file, and what A6 reads is the encoder half of it", and the second clause
        # confuses what A6 *uses* with what a box *mounts*. `checkpoint_format` here is
        # `mmselfsup_rgb`, and `benchmark.extract._build_timm` reads the whole published file
        # and picks `backbone.` out of it (`_timm_state_from_mmselfsup`) — so the file at
        # `paths.imagenet_mae_checkpoint` **is** the 485 MB one, and this is its md5.
        #
        # The same sentence also said the value "goes in the moment a box mounts one and
        # `freeze` records what it froze against", which `benchmark.extract.build_spec` does not
        # do: it copies this field into the spec and computes an observed md5 only to *check* a
        # pin, so an unpinned encoder produces a spec whose `checkpoint_md5` is null — and
        # `Extractor._checked_path` then refuses to load it at all. A6 was unrunnable until this
        # line moved, which is what found it.
        #
        # Quoted from `pretrain.init.MMSELFSUP_MAE_VIT_B16_1600E` — the same published artifact,
        # already carrying its URL, its 484,833,387 bytes, its Apache-2.0 licence and that
        # licence's source, because P101 spent a ticket establishing that the licence is why this
        # checkpoint was chosen. Verified against a downloaded copy rather than transcribed.
        checkpoint_md5=IMAGENET_MAE_MD5,
    ),
    # A9's entry. Same file, same 90.2 M-parameter encoder, same diet — what differs is what the
    # arm does *after* the features (§1: encode, FSQ, diffusion decode), which is not something
    # the frozen-feature contract has an opinion about. That is also why `params` is a different
    # number here than on `terramind` above and not a disagreement with it: `parameter_counts`
    # reports what an arm *runs*, A5 runs the encoder half, and A9 runs the whole tokenizer.
    #
    # **Counted, where every other entry's is quoted** (#168). This line read `None` for as long
    # as A9's convention was unbuilt, on the rule that a count this registry guessed would print
    # as a measurement — and no source in this plan gives the sum. It is 286,316,821: p118 §8's
    # 90,172,416 encoder, a 196,140,560 diffusion decoder, and 3,845 across the FSQ quantizer and
    # the post-quant projection, off the strict load in `benchmark.passthrough.load` and
    # reproducible by `benchmark.passthrough.parameter_count`. The 46,080 ADR-0011's crop removes
    # is `params`' standing caveat here as above: this is the tokenizer **as published**.
    "terramind-tokenizer": Encoder(
        name="terramind-tokenizer",
        family="terramind",
        width=768,
        pos_encoding="sincos2d_terramind",
        pos_emb_derivation="rederived_14x14_crop",
        pos_emb_native_grid_px=16,
        checkpoint_field="terramind_checkpoint",
        params=286_316_821,
        checkpoint_format="terratorch",
        checkpoint_md5=TERRAMIND_TOKENIZER_DEM_MD5,
        output_is_normed=False,
        has_cls=False,
        pretraining_epochs=TERRAMIND_DEM_EPOCHS,
        pretraining_epochs_citation=TERRAMIND_DEM_EPOCHS_CITATION,
    ),
}


@dataclass(frozen=True)
class Arm:
    """One row of §1's table: what it is, what trains, what it is fed, and how."""

    name: str
    #: Key into :data:`ENCODERS`, or ``None`` for the two arms that run no encoder at all —
    #: A1's classical interpolation and A7's from-scratch UNet.
    encoder: str | None
    #: One of :data:`TRAINABLE`.
    trains: str
    #: One of :data:`INPUT_CONVENTIONS`.
    input_convention: str
    #: One of :data:`NORMALIZATIONS`. Declared per arm because §2.1's whole point is that the
    #: two encoders want different diets and neither refuses the other's.
    normalization: str
    #: What the arm is, in the words the resolution line and the report print it in.
    what: str
    #: Whether this arm is one of R5's eleven printed rows (:data:`PRINTED_ROWS`) or a §2.3
    #: appendix cell reported as a single observation without a ±.
    printed: bool = True
    #: Whether the tokenizer's diffusion sampler keeps TerraMind's **published** dynamic
    #: thresholding. Only the ``a1_fill_through_tokenizer`` arms read it; every other arm runs no
    #: sampler and leaves it at the published ``True``.
    #:
    #: A field on the *arm* rather than a second :data:`ENCODERS` entry, which is where
    #: ``terramind-bicubic`` puts its one knob. The difference is what the knob moves.
    #: ADR-0011's crop changes the encoder's own `pos_emb` tensor, and
    #: :class:`benchmark.extract.Extractor` caches built models **by encoder name**, so two arms
    #: sharing a name would share one cached model and silently score the same weights twice.
    #: Thresholding changes no weight and no tensor: it is a property of the DDIM sampler the
    #: 196 M decoder is run under, and A9 and A9-NOTHRESH really do read one encoder — which
    #: :func:`benchmark.passthrough.assert_same_encoder` goes on proving for both. Putting it on
    #: the encoder would assert a difference in the encoder that does not exist.
    #:
    #: It is likewise **not** a field of :class:`benchmark.extract.EncoderSpec`, on
    #: :attr:`Encoder.pretraining_epochs`' rule: ADR-0012's committed contract governs what comes
    #: out of a frozen *encoder*, and the sampler changes nothing in it.
    sampler_thresholding: bool = True

    def __post_init__(self) -> None:
        if self.encoder is not None and self.encoder not in ENCODERS:
            raise ValueError(f"arm {self.name}: unknown encoder {self.encoder!r}")
        if self.trains not in TRAINABLE:
            raise ValueError(f"arm {self.name}: unknown trainable part {self.trains!r}")
        if self.input_convention not in INPUT_CONVENTIONS:
            raise ValueError(f"arm {self.name}: unknown input {self.input_convention!r}")
        if self.normalization not in NORMALIZATIONS:
            raise ValueError(f"arm {self.name}: unknown normalization {self.normalization!r}")
        if not self.sampler_thresholding and self.input_convention != "a1_fill_through_tokenizer":
            raise ValueError(
                f"arm {self.name}: `sampler_thresholding` is the tokenizer's diffusion sampler "
                f"and only {'a1_fill_through_tokenizer'!r} runs one. An arm that moved it "
                f"without running a sampler would print a knob that changed nothing"
            )
        if (self.encoder is None) != (self.normalization == "none"):
            raise ValueError(
                f"arm {self.name}: an arm with an encoder declares that encoder's diet and an "
                f"arm without one declares 'none' (§2.1)"
            )

    @property
    def width(self) -> int | None:
        """The feature width the shared head's first projection absorbs — §4.1's only delta."""
        return None if self.encoder is None else ENCODERS[self.encoder].width

    @property
    def is_inference_only(self) -> bool:
        """Does this arm train anything? A1, A2 and A9 do not.

        A property rather than ``trains == "nothing"`` spelled at each site, because three
        separate questions are asked of it — what a row's trainable parameter count is, whether
        §3.4's train mask bank is an input, and whether §3.1's head-train and val draws are — and
        an inference-only arm added later has to get the same answer to all three. The string
        comparison in four places is one typo away from an arm that silently trains nothing and
        is still refused for want of a bank.
        """
        return self.trains == "nothing"

    @property
    def trains_head(self) -> bool:
        """Does this arm train §4.1's shared head? A4, A5, A6 and the ViT-S appendix cell do.

        :attr:`is_inference_only`'s sibling and its argument applied a second time (#159): five
        separate questions are asked of it — what a row's parameter pair is, whether the
        resolution line prints §4.1's frozen recipe, whether `run` checks that recipe against the
        frozen one, whether `run` trains before it scores, and what the eval pass is handed. A6
        answering four of the five is an arm that trains a head nothing checks the recipe of, and
        the string comparison spelled at each site is one typo away from exactly that.
        """
        return self.trains == "head"

    @property
    def trains_everything(self) -> bool:
        """Does this arm train §4.4's from-scratch UNet? A7, and only A7 (#162).

        :attr:`trains_head`'s sibling and its argument a third time: three questions are asked
        of it — what A7's parameter pair is, whether `run` trains before it scores, and what the
        resolution line prints about the recipe it trains under — and the string comparison
        spelled at each site is one typo away from an arm that scores an untrained UNet.
        """
        return self.trains == "everything"

    @property
    def runs_pretrained_decoder(self) -> bool:
        """Does this arm run the pretraining **decoder**, as pretrained? A2, and only A2.

        Read off two declared fields rather than off the name, on :func:`benchmark.artifacts.
        _required_for`'s rule: ``token_drop`` is the convention whose output *is* the decoder's,
        and :attr:`is_inference_only` is what says nothing was retrained on the way. A3 shares
        the convention and trains ``encoder+decoder``, so it initializes a decoder rather than
        running one, and answers ``False`` here — which is right: what a fine-tune needs from a
        checkpoint is its own ticket's question (#162), not this one's.

        The consequence is a second file. §14 exports encoder-only weights, so an arm for which
        this is true cannot be driven by the release `encoder.pt` every other arm reads
        (:attr:`Encoder.mae_checkpoint_field`).
        """
        return self.input_convention == "token_drop" and self.is_inference_only


def _arm(name: str, **kwargs) -> tuple[str, Arm]:
    return name, Arm(name=name, **kwargs)


#: §1's table, split by init where §2.3 splits it. Insertion order is the printed order.
ARMS: Final[dict[str, Arm]] = dict(
    [
        _arm(
            "A1",
            encoder=None,
            trains="nothing",
            input_convention="classical_fill",
            normalization="none",
            what="classical fill — `fillnodata`-class interpolation, the residual base every "
            "learned arm improves on (§2.4)",
        ),
        _arm(
            "A2-R5",
            encoder="demae-r5",
            trains="nothing",
            input_convention="token_drop",
            normalization="per_tile",
            what="DEMAE zero-shot — native MAE inpainting, no trained head (§4.3, R12)",
        ),
        _arm(
            "A2-R4",
            encoder="demae-r4",
            trains="nothing",
            input_convention="token_drop",
            normalization="per_tile",
            what="DEMAE zero-shot, continual init (§4.3, R12)",
        ),
        _arm(
            "A3-R5",
            encoder="demae-r5",
            trains="encoder+decoder",
            input_convention="token_drop",
            normalization="per_tile",
            what="DEMAE fine-tuned — 20 epochs on the bank's first 20 (§4.2)",
        ),
        _arm(
            "A3-R4",
            encoder="demae-r4",
            trains="encoder+decoder",
            input_convention="token_drop",
            normalization="per_tile",
            what="DEMAE fine-tuned, continual init (§4.2)",
        ),
        _arm(
            "A4-R5",
            encoder="demae-r5",
            trains="head",
            input_convention="a1_fill_plus_mask",
            normalization="per_tile",
            what="DEMAE frozen + the §4.1 head — C1's subject",
        ),
        _arm(
            "A4-R4",
            encoder="demae-r4",
            trains="head",
            input_convention="a1_fill_plus_mask",
            normalization="per_tile",
            what="DEMAE frozen + head, continual init",
        ),
        _arm(
            "A5",
            encoder="terramind",
            trains="head",
            input_convention="a1_fill_plus_mask",
            normalization="terramind_v1",
            what="TerraMind DEM tokenizer encoder, frozen + the identical head",
        ),
        _arm(
            "A6",
            encoder="imagenet-mae",
            trains="head",
            input_convention="a1_fill_plus_mask",
            normalization="per_tile",
            what="ImageNet-MAE frozen + the identical head — the neutral encoder head "
            "hyperparameters are tuned on (§4.1)",
        ),
        _arm(
            "A7",
            encoder=None,
            trains="everything",
            input_convention="a1_fill_plus_mask",
            normalization="none",
            what="UNet from scratch — ~17M trainable against every frozen arm's 1.18M (§4.4)",
        ),
        _arm(
            "A9",
            encoder="terramind-tokenizer",
            trains="nothing",
            input_convention="a1_fill_through_tokenizer",
            normalization="terramind_v1",
            what="TerraMind tokenizer pass-through — encode, FSQ, diffusion decode. §7's one "
            "unpriced line: measure per-instance cost before committing the full pass",
        ),
        # The fair fight (#168). A9 runs the tokenizer exactly as terratorch's registry builds it,
        # and that registry hands the DEM DiVAE a DDIM sampler with `thresholding=True` and
        # `sample_max_value=1.0`. Dynamic thresholding is Imagen's trick for *images*, whose data
        # really are bounded to [-1, 1]; it rescales every sample into that interval. TerraMind's
        # DEM diet is (h - 670.665) / 951.272 — zero-mean, unit-**std**, so a third of real
        # elevation sits outside |1| by construction — and the interval therefore lands at
        # [-280.6 m, +1621.9 m]. Measured on flat tiles: 2500 m, 3000 m and 4000 m all decode to
        # 1621.9 m, and 6000 m *wraps* to -277 m.
        #
        # 37.7% of the committed eval instances sit on tiles above that ceiling, and for them the
        # overshoot predicts the scored bias at r = 0.985. So A9's row measures the published
        # sampler's ceiling far more than it measures the tokenizer, and this arm is what lets the
        # table say which. The setting is the *only* difference: same weights, same md5, same
        # diet, same 50 timesteps, one knob.
        #
        # The knob is defensible rather than flattering. The DEM tokenizer's own recorded training
        # args carry `run_name="tokenization/divae/rgb/DEM-ViTB-UNetP4_16k_224-264"` — **rgb** —
        # so the recipe is the image one, and the bounded-data assumption it inherits is false for
        # elevation. Turning it off recovers capability the model already has rather than granting
        # it any: the UNet predicts `sample` and was trained against unclipped targets, and with
        # thresholding off 4000 m decodes to 3974.6 m.
        _arm(
            "A9-NOTHRESH",
            encoder="terramind-tokenizer",
            trains="nothing",
            input_convention="a1_fill_through_tokenizer",
            normalization="terramind_v1",
            sampler_thresholding=False,
            # An appendix cell, on `A5-BICUBIC`'s precedent and for its reason. R5's printed
            # table is **pre-registered** at eleven rows, and this arm was registered after A9's
            # pass had run and been read. A post-hoc ablation promoted into the pre-registered
            # table is the exact move pre-registration exists to prevent — and it would be the
            # flattering direction, which is worse. The row is reported in full beside
            # `A5-BICUBIC`; what carries the finding into the headline is §7's prose and A9's own
            # row, which now says what its ceiling costs.
            printed=False,
            what="TerraMind tokenizer pass-through, sampler thresholding off — A9's one knob, "
            "and the row that separates the published ceiling from the model",
        ),
        _arm(
            "A5-BICUBIC",
            encoder="terramind-bicubic",
            trains="head",
            input_convention="a1_fill_plus_mask",
            normalization="terramind_v1",
            what="A5 under terratorch's own bicubic `pos_emb` resize — ADR-0011 D4's "
            "pre-registered ablation row, kept by #134's 0.9708 against R11's 0.99 bar",
            printed=False,
        ),
        _arm(
            "A4-S",
            encoder="demae-vits",
            trains="head",
            input_convention="a1_fill_plus_mask",
            normalization="per_tile",
            what="DEMAE ViT-S frozen + head — §2.3 appendix, one seed, no ±",
            printed=False,
        ),
    ]
)

#: R5's eleven printed arm rows, in the order the table prints them (#136).
PRINTED_ROWS: Final[tuple[str, ...]] = tuple(
    name for name, arm in ARMS.items() if arm.printed
)


def normalization_for(encoder_name: str) -> str:
    """§2.1's declared diet for an encoder, read off the arms that eat it.

    Here rather than in :mod:`benchmark.extract` because it is a question about this table:
    ``normalization`` is an *arm* field, since §2.1's whole point is that each model is fed its
    native diet. Two arms on one encoder disagreeing about the diet is a contradiction, and one
    the contract should surface rather than pick a side of.
    """
    diets = {arm.normalization for arm in ARMS.values() if arm.encoder == encoder_name}
    if len(diets) != 1:
        raise ValueError(
            f"the arms reading {encoder_name} declare {sorted(diets)} — §2.1 feeds each model "
            f"one native diet, and a contract cannot declare two"
        )
    return diets.pop()


def parameter_counts(arm: Arm) -> dict[str, int | None]:
    """§6 item 4's two numbers for this arm: ``total`` and ``trainable``.

    Both, and never one: A7's 17M reads as handicapped beside A4's 86.4M until you know that A7
    trains **14x more** of it. That is the whole reason the plan asks for the pair, and it is why
    this returns a mapping rather than an int — a single "parameters" number is the shape of the
    misreading.

    ``None`` is *not counted here yet*, and it is deliberate on :attr:`Encoder.params`' own rule:
    a count this registry guessed would print as a measurement. Four cases are genuinely known
    and are returned as numbers:

    * **an arm with no encoder that trains nothing** — A1's classical interpolation, which has no
      parameters at all, so both counts are a true 0 rather than an unknown;
    * **an arm that trains nothing over a counted encoder** — A2 zero-shot, whose total is the
      encoder's and whose trainable half is exactly 0;
    * **the trainable half of any inference-only arm**, which is 0 whatever its encoder costs;
    * **an arm that trains §4.1's head over a counted encoder** — A4, A5 and A6, whose trainable
      half is :func:`benchmark.head.parameter_count` at that encoder's width and whose total is
      that plus the frozen encoder. #159 built the head, so these stopped being ``None``.
    * **the from-scratch arm** — A7, whose total and trainable are both
      :func:`benchmark.unet.parameter_count`, because it runs no encoder and trains every
      parameter it has. #162 built the UNet, so this stopped being ``None``.

    **A7's two numbers are equal, and that is the point rather than a degenerate case.** The pair
    §6 item 4 prints is what makes the table legible *across* arms: A7 at (17.5M, 17.5M) beside
    A4 at (86.4M, 1.18M) is the sentence §4.4 rewrote its capacity anchor into — an order of
    magnitude more trained, at a fifth of the total. A column of totals alone reverses it.

    What still waits on its own module is A3's fine-tuned decoder (#166), which reports ``None``
    rather than a plausible number nothing measured.
    """
    # Deferred so that importing this registry stays free: it is a declarative module with no
    # dependencies at all, and `benchmark.head` pulls in numpy and `dataset.config`. Everything
    # that asks for a parameter count is already paying for both.
    from . import head, unet

    encoder = None if arm.encoder is None else ENCODERS[arm.encoder]
    trainable = 0 if arm.is_inference_only else None
    if arm.trains_everything:
        # Torch-free arithmetic over §4.4's shape — see `unet.parameter_count`.
        trainable = unet.parameter_count()
    if arm.trains_head and encoder is not None:
        # Torch-free arithmetic over §4.1's four stage dimensions — see `head.parameter_count`.
        # The width is the *only* thing that differs between A4, A5 and A6 here, which is the
        # same sentence C1 rests on, arriving as a number.
        trainable = head.parameter_count(encoder.width)
    if encoder is None:
        # A1 and A7. A1 trains nothing and runs no encoder, so both its counts are a true 0;
        # A7 runs no encoder either, so its total *is* the UNet it trains.
        total = 0 if arm.is_inference_only else trainable
    elif arm.is_inference_only:
        total = encoder.params
    elif arm.trains_head and encoder.params is not None:
        total = encoder.params + trainable
    else:
        total = None
    return {"total": total, "trainable": trainable}


def get(name: str) -> Arm:
    """The registered arm, or a refusal naming the registry.

    The refusal is generic on purpose — see the module docstring on A8 and A10.
    """
    try:
        return ARMS[name]
    except KeyError:
        raise ValueError(
            f"unknown arm {name!r}; registered: {sorted(ARMS)}"
        ) from None
