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
    #: ADR-0011 / §2.3's precondition, declared rather than remembered. ``sincos2d`` is MAE's
    #: own column-first order; ``sincos2d_rowfirst`` is mmselfsup's, which R4's weights learned;
    #: ``rederived_14x14_crop`` is ADR-0011's centre-anchored crop, *not* a bicubic resize.
    pos_encoding: str
    #: The :class:`benchmark.config.Paths` field naming this encoder's checkpoint.
    checkpoint_field: str
    #: §2.3's table, where it gives one. ``None`` where no source in this plan counts them, and
    #: left ``None`` rather than inferred: §6 item 4 prints parameter counts, and a count this
    #: registry guessed would print as a measurement.
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


#: The `-v2` release bundles' `encoder.pt` md5s, and the answer to "which bundle is
#: authoritative" (#156). Both R5 bundles ship a **byte-identical** `encoder.pt`, and so do both
#: R4 bundles — `-v2` re-exported the *bundle* (it adds `LICENSE` and the manifest's `licence`
#: block; #101's licence work) and not the weights. So the question the ticket asks has no
#: weights axis at all: these md5s identify the R5 and R4 encoders under either directory, and
#: `-v2` is the authoritative bundle because it is the one that carries its own terms.
#: :mod:`benchmark.config`'s `Paths` points at `-v2` for that reason.
DEMAE_R5_MD5: Final = "3ec5ebc72fa28c43fe88f67692d9b41a"
DEMAE_R4_MD5: Final = "59e8346c4b08e1ab4fedd44f0802e12b"

#: The encoders §1 and §2.3 name. TerraMind's 4.92M excess over DEMAE's ViT-B is reported and
#: not attributed (§2.3): terratorch was not installed in any live worktree when it was written.
ENCODERS: Final[dict[str, Encoder]] = {
    "demae-r5": Encoder(
        name="demae-r5",
        family="demae",
        width=768,
        pos_encoding="sincos2d",
        checkpoint_field="demae_r5_checkpoint",
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
        # ADR-0011: the centre-anchored 14x14 crop of TerraMind's 16x16 grid, so both encoders
        # are in-distribution at 224 px simultaneously. The bicubic resize scored 0.9708 against
        # #134's 0.99 materiality bar and was rejected; §4's A5-at-20k ablation row evidences it.
        pos_encoding="rederived_14x14_crop",
        checkpoint_field="terramind_checkpoint",
        params=90_172_416,
        checkpoint_format="terratorch",
        # p118 §5.3 measured it: `return_intermediates=True` yields twelve tensors with no
        # separate post-norm entry, its last bit-identical to `encoder(x)`. So "the last layer"
        # means something different here than it does for DEMAE, which is ADR-0012 §3's subject.
        output_is_normed=False,
        has_cls=False,
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
        # Unpinned here rather than quoted from `pretrain.init.KNOWN_CHECKPOINTS`: that md5 is
        # the 485 MB *full* MAE file, and what A6 reads is the encoder half of it. The value
        # goes in the moment a box mounts one and `freeze` records what it froze against.
        checkpoint_md5=None,
    ),
    "terramind-tokenizer": Encoder(
        name="terramind-tokenizer",
        family="terramind",
        width=768,
        pos_encoding="rederived_14x14_crop",
        checkpoint_field="terramind_checkpoint",
        checkpoint_format="terratorch",
        output_is_normed=False,
        has_cls=False,
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

    def __post_init__(self) -> None:
        if self.encoder is not None and self.encoder not in ENCODERS:
            raise ValueError(f"arm {self.name}: unknown encoder {self.encoder!r}")
        if self.trains not in TRAINABLE:
            raise ValueError(f"arm {self.name}: unknown trainable part {self.trains!r}")
        if self.input_convention not in INPUT_CONVENTIONS:
            raise ValueError(f"arm {self.name}: unknown input {self.input_convention!r}")
        if self.normalization not in NORMALIZATIONS:
            raise ValueError(f"arm {self.name}: unknown normalization {self.normalization!r}")
        if (self.encoder is None) != (self.normalization == "none"):
            raise ValueError(
                f"arm {self.name}: an arm with an encoder declares that encoder's diet and an "
                f"arm without one declares 'none' (§2.1)"
            )

    @property
    def width(self) -> int | None:
        """The feature width the shared head's first projection absorbs — §4.1's only delta."""
        return None if self.encoder is None else ENCODERS[self.encoder].width


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
