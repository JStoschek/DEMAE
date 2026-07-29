"""Warm-starting the encoder from OpenMMLab's ImageNet MAE ViT-B/16 (#102, plan §8).

This is the path run **R4** takes and R1-R3/R5 do not: read an externally trained ImageNet MAE
encoder into the #90 model, collapse its RGB patch embed to one elevation channel, leave the
decoder fresh, and continue pretraining. It is selected by :attr:`Init.mode
<pretrain.config.Init>`, so the two arms differ by *configuration* and the scratch path is
untouched — :func:`initialize_from_config` is a no-op at ``scratch``.

The checkpoint is :data:`MMSELFSUP_MAE_VIT_B16_1600E`, OpenMMLab's Apache-2.0 reproduction, not
Meta's official release. Plan §8's 2026-07-29 amendment made that substitution and
[P101](../../docs/research/p101-mae-checkpoint-licence.md) is the evidence: the official file's
licence is ambiguous in a way that cannot be resolved by asking, and the ambiguity would sit
under a released artifact. Same objective, same architecture, same data; ~0.9 pt of ImageNet
linear probe, accepted and argued in the plan.

## Why this module is what it is, instead of `load_state_dict(..., strict=False)`

Because the failure it exists to prevent does not look like a failure. A partial load trains
perfectly well, descends a loss curve, and produces an R4 that performs like R1 — and the one
question R4 exists to answer is whether warm-starting helps. Three layers, in the order the
risk deserves:

1. **Names.** The state dicts do not match key-for-key: timm blocks are not mmselfsup's blocks.
   :func:`encoder_key_map` is the whole mapping, written out and **total in both directions** —
   every encoder parameter of the model has a source, and every ``backbone.`` tensor in the
   file has a destination or an explicit reason not to. A tensor that fails to populate is an
   error. There is no ``strict=False`` anywhere in this file.
2. **Contents.** What the file holds is asserted rather than inferred from its size. It carries
   an encoder *and* a decoder *and* two ImageNet channel-normalization constants; the last two
   are discarded **by name**, counted, and reported. "485 MB is about 121M floats" was a
   reasonable inference (P101 §6.1) and it was also wrong by 9M; §1.1 of
   [P102](../../docs/research/p102-mmselfsup-mae-layout.md) reads the entry list instead.
3. **Layout.** Names and shapes agreeing proves nothing about *order*. A fused-``qkv`` in a
   different order, a transposition, reversed blocks — every name populates, every shape
   matches, every assertion above passes, and the encoder computes garbage. #103 set the policy:
   layout conventions are read from the source implementation and written down as checked claims
   with a link, never inferred from shapes. P102 §3 is that record, one row per convention.

## The oracle #102 did not think it had

P102 §4 found that mmselfsup's fixed sin-cos positional embedding is the **row/column transpose
of MAE's** — `torch.meshgrid` defaults to ``'ij'`` where `np.meshgrid` defaults to ``'xy'``, and
each library's obvious code produces the other's mirror. That is not a hypothetical; it is this
checkpoint, and it is exactly the class of bug #103 warned about.

It also handed us a **free equivalence test against the real weights**. The checkpoint ships its
own `backbone.pos_embed`, which is a pure function of the grid — so we can reconstruct it and
compare 151,296 floats before copying a single weight. That is a stronger check than any claim
in a table, it needs no `mmcv` and no reference model, and #102's two candidate oracles were
both declined as too expensive. :func:`load_mmselfsup_mae_encoder` does it first, and refuses a
model whose ordering disagrees.

The remaining behavioural layer is :func:`encoder_behaviour`, and **its power was measured rather
than assumed, with a result worth stating up front: the two statistics #102 proposed do not work.**
Eight name-and-shape-preserving scrambles against six candidate statistics (P102 §6.1); six of the
eight score a *higher* crop margin than the correct load. What survived is a per-block
amplification bound, which catches 2 of the 8 — both of them the scrambles a wrong key map would
actually produce. :class:`EncoderBehaviour` carries the whole argument, including why the gap is
acceptable: the six it misses need code this module does not contain.
"""

from __future__ import annotations

import hashlib
import pickle
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import Init, RunConfig
from .model import MaskedAutoencoder, is_decoder_parameter, sincos2d_pos_embed

# --- What we are loading ------------------------------------------------------


@dataclass(frozen=True)
class CheckpointSpec:
    """The identity of an external init: what it is, where it came from, under what licence.

    Recorded as data rather than prose because two tickets downstream need to *read* it. #86
    verifies the file it fetched onto the box is this one, and #88 must stamp an init-provenance
    field into every released artifact (P101 §7: "every released run needs to state what its
    encoder was initialized from and under what licence" — `scratch` for R1/R2/R3/R5, this for
    R4). One object, so the two cannot disagree.

    Every count here was **read from the file**, not inferred from its size; P102 §1 shows how
    and why that distinction earns its keep.
    """

    #: The publisher's own model name, so the row in their metafile is findable.
    name: str
    url: str
    #: The publisher's `Content-MD5`, verified against the downloaded bytes (P102 §1).
    md5: str
    size_bytes: int
    #: Pretraining epochs on ImageNet-1k. 1600 rather than 800 because it is the closest of the
    #: four published schedules to the official weights on the linear-probe axis, which is the
    #: axis §10 adjudicates R4-vs-R5 with (plan §8's amendment).
    epochs: int
    licence: str
    licence_source: str
    #: `state_dict` entry counts per prefix, and the encoder's parameter total.
    encoder_tensors: int
    decoder_tensors: int
    preprocessor_tensors: int
    encoder_parameters: int


#: R4's init, fixed by plan §8's 2026-07-29 amendment (#103).
MMSELFSUP_MAE_VIT_B16_1600E: Final = CheckpointSpec(
    name="mae_vit-base-p16_8xb512-amp-coslr-1600e_in1k",
    url=(
        "https://download.openmmlab.com/mmselfsup/1.x/mae/"
        "mae_vit-base-p16_8xb512-fp16-coslr-1600e_in1k/"
        "mae_vit-base-p16_8xb512-fp16-coslr-1600e_in1k_20220825-f7569ca2.pth"
    ),
    md5="91ecc0345a6d6ecefe9c59041841f180",
    size_bytes=484_833_387,
    epochs=1600,
    licence="Apache-2.0",
    licence_source="https://github.com/open-mmlab/mmselfsup/blob/v1.0.0/LICENSE",
    encoder_tensors=150,
    decoder_tensors=104,
    preprocessor_tensors=2,
    encoder_parameters=85_798_656,
)

#: The three prefixes the file's ``state_dict`` uses, and what each is for. `backbone.` is the
#: encoder and the only one loaded; `neck.` is their 8-block 512-wide MAE decoder, which R4
#: replaces with a fresh one at #90's dimensions; `data_preprocessor.` is the pair of ImageNet
#: RGB `mean`/`std` constants the encoder was trained under. The last is worth discarding *by
#: name* rather than skipping past: DEMAE normalizes per tile on standardized elevation (plan §5,
#: ADR-0003 §3), so those two tensors are not merely unneeded, they are wrong for this input.
ENCODER_PREFIX: Final = "backbone."
DECODER_PREFIX: Final = "neck."
PREPROCESSOR_PREFIX: Final = "data_preprocessor."
DISCARDED_PREFIXES: Final = (DECODER_PREFIX, PREPROCESSOR_PREFIX)

#: The one encoder tensor that is **compared instead of copied** — the oracle described in the
#: module docstring. It is a fixed function of the grid on both sides, so ours is not "close
#: enough to theirs", it is reconstructible and checked.
POS_EMBED_KEY: Final = f"{ENCODER_PREFIX}pos_embed"

#: fp32 on both sides (theirs read from the file, ours built in fp64 and cast), so the only
#: slack is the cast itself: the real checkpoint agrees to 6.7e-7 and disagrees by 2.0 — the
#: full range of a sinusoid — under the wrong axis order. Nothing sits between those.
POS_EMBED_ATOL: Final = 1e-5


# --- The key map --------------------------------------------------------------

#: Their name -> ours, for everything that is not per-block. See P102 §3 for the claim behind
#: each: L2 for the patch embed's conv layout, L8 for `backbone.ln1` being the *final* norm
#: rather than a block's (the name collides with each block's `ln1`, which is why this map keys
#: off full paths and never off a suffix).
_TOP_LEVEL_KEYS: Final = {
    "cls_token": "cls_token",
    "patch_embed.projection.weight": "patch_embed.proj.weight",
    "patch_embed.projection.bias": "patch_embed.proj.bias",
    "ln1.weight": "norm.weight",
    "ln1.bias": "norm.bias",
}

#: Their per-block name -> ours, for one transformer block. Every row is a checked claim in
#: P102 §3: L6 pins that `ln1`/`ln2` are timm's `norm1`/`norm2` in a pre-norm block, L3 that
#: `attn.qkv` is one fused `Linear` split q,k,v untransposed exactly as timm splits it, and L7
#: that `ffn.layers.0.0` is fc1 and `ffn.layers.1` is fc2 (the `0.0` is an `FFN` whose first
#: entry is itself a `Sequential` of Linear, GELU, Dropout).
_BLOCK_KEYS: Final = {
    "ln1.weight": "norm1.weight",
    "ln1.bias": "norm1.bias",
    "attn.qkv.weight": "attn.qkv.weight",
    "attn.qkv.bias": "attn.qkv.bias",
    "attn.proj.weight": "attn.proj.weight",
    "attn.proj.bias": "attn.proj.bias",
    "ln2.weight": "norm2.weight",
    "ln2.bias": "norm2.bias",
    "ffn.layers.0.0.weight": "mlp.fc1.weight",
    "ffn.layers.0.0.bias": "mlp.fc1.bias",
    "ffn.layers.1.weight": "mlp.fc2.weight",
    "ffn.layers.1.bias": "mlp.fc2.bias",
}


def encoder_key_map(depth: int) -> dict[str, str]:
    """The complete mmselfsup-to-DEMAE encoder key mapping at ``depth`` blocks.

    ``depth`` rather than a hardcoded 12 so the mapping is a function of the architecture and
    the tests can exercise it on a two-block model instead of an 86M-parameter one. Excludes
    :data:`POS_EMBED_KEY`, which is verified rather than copied.
    """
    mapping = {f"{ENCODER_PREFIX}{k}": v for k, v in _TOP_LEVEL_KEYS.items()}
    for i in range(depth):
        for theirs, ours in _BLOCK_KEYS.items():
            mapping[f"{ENCODER_PREFIX}layers.{i}.{theirs}"] = f"blocks.{i}.{ours}"
    return mapping


# --- The patch-embed channel collapse -----------------------------------------


def collapse_input_channels(kernel: Tensor, *, adapt: str = "sum") -> Tensor:
    """``(out, 3, kh, kw)`` -> ``(out, 1, kh, kw)``, plan §8's patch-embed surgery.

    ``sum`` is timm's `adapt_input_conv` behaviour and what §8 specifies; ``mean`` exists only
    because the config offers it and a path the config can select must be a path that works.

    The reason §8 is emphatic about which is not a preference. For an input that is identical in
    every channel — which is exactly what a grayscale image replicated to RGB is —
    ``sum(W_c * x) == (sum W_c) * x``, so the summed kernel reproduces the original 3-channel
    convolution **exactly** and the mean reproduces a third of it. Feeding pretrained weights
    activations at a third of their trained scale is not a small perturbation this far into a
    LayerNorm-free patch embed; it is a systematic bias every downstream block inherits.
    ``test_the_summed_kernel_reproduces_the_three_channel_convolution_exactly`` is the check,
    and it is an equality rather than a correlation.
    """
    if kernel.ndim != 4:
        raise ValueError(
            f"expected a conv kernel (out, in, kh, kw), got shape {tuple(kernel.shape)}"
        )
    if adapt == "sum":
        return kernel.sum(dim=1, keepdim=True)
    if adapt == "mean":
        return kernel.mean(dim=1, keepdim=True)
    raise ValueError(f"unknown patch_embed_adapt {adapt!r} (§8: sum | mean)")


# --- Reading the file ---------------------------------------------------------


class _Dropped:
    """Stand-in for a class the checkpoint pickles that we neither need nor want to execute."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __setstate__(self, state: Any) -> None:
        pass


#: The only globals :class:`_RestrictedUnpickler` will resolve: torch's tensor-rebuilding
#: helpers and `OrderedDict`. Storage classes are allowed by pattern below.
_ALLOWED_GLOBALS: Final = {
    ("collections", "OrderedDict"),
    ("torch._utils", "_rebuild_tensor"),
    ("torch._utils", "_rebuild_tensor_v2"),
    ("torch._utils", "_rebuild_parameter"),
}


class _RestrictedUnpickler(pickle.Unpickler):
    """Resolve the globals a tensor state dict needs; make every other one inert.

    `torch.load`'s own ``weights_only=True`` **refuses this checkpoint**, and it is right to:
    the file's ``message_hub`` pickles `mmengine.logging.history_buffer.HistoryBuffer`, a class
    from a package DEMAE does not depend on and cannot allowlist because it is not importable
    here. The tempting fix is ``weights_only=False``, which disarms the check for the whole file
    and permits arbitrary code execution on 485 MB fetched over HTTP.

    So instead the log objects are *dropped*: anything outside :data:`_ALLOWED_GLOBALS` and the
    storage classes becomes :class:`_Dropped`, which does nothing when constructed or restored.
    The tensors load normally, ``message_hub`` and ``param_schedulers`` arrive as inert
    placeholders nothing reads, and the safety property is the one ``weights_only`` was giving
    us. ``meta`` survives intact — it is plain dicts and strings — which is what lets
    :func:`load_mmselfsup_mae_encoder` read the epoch back out of the file for provenance.
    """

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in _ALLOWED_GLOBALS:
            return super().find_class(module, name)
        if module == "torch" and (name.endswith("Storage") or name == "Size"):
            return super().find_class(module, name)
        return _Dropped


class _restricted_pickle:
    """A ``pickle_module`` for `torch.load`, wired to :class:`_RestrictedUnpickler`."""

    Unpickler = _RestrictedUnpickler

    @staticmethod
    def load(f: Any, **kwargs: Any) -> Any:
        return _RestrictedUnpickler(f, **kwargs).load()


def read_state_dict(checkpoint: Path | str | Mapping[str, Any]) -> tuple[dict[str, Tensor], dict]:
    """Load a checkpoint and return ``(state_dict, meta)``.

    Accepts a path, or an already-loaded mapping — the tests pass one in rather than writing
    485 MB to a temporary directory, and #86's pre-flight passes a path.

    What arrives is asserted, not assumed: a mapping carrying ``state_dict`` is unwrapped, a
    mapping that is *already* a state dict is taken as one, and anything else raises. Tensors
    only; a ``state_dict`` value that is not a tensor is a file this loader does not understand
    and saying so beats discovering it during a shape check.
    """
    if isinstance(checkpoint, (str, Path)):
        loaded = torch.load(
            Path(checkpoint), map_location="cpu", pickle_module=_restricted_pickle
        )
    else:
        loaded = checkpoint
    if not isinstance(loaded, Mapping):
        raise ValueError(f"expected a checkpoint mapping, got {type(loaded).__name__}")

    meta = loaded.get("meta") if isinstance(loaded.get("meta"), Mapping) else {}
    payload = loaded["state_dict"] if "state_dict" in loaded else loaded
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("checkpoint has no non-empty 'state_dict'")

    state = {str(k): v for k, v in payload.items()}
    non_tensors = sorted(k for k, v in state.items() if not isinstance(v, Tensor))
    if non_tensors:
        raise ValueError(f"state_dict entries are not tensors: {non_tensors[:5]}")
    return state, dict(meta)


def checkpoint_md5(path: Path | str, *, chunk_bytes: int = 1 << 20) -> str:
    """The file's md5 — the publisher's own `Content-MD5`, and what identifies it.

    Called on every load from a path (:func:`_verify_against_spec`). Hashing 485 MB costs about a
    second against a load that reads the same bytes anyway, which is a poor reason to leave the
    one fact that distinguishes this checkpoint from its three siblings unchecked.
    """
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


#: Every checkpoint DEMAE is willing to make a provenance claim about. One entry today; a list
#: because :func:`recognize` matching against it is what keeps the claim *measured*.
KNOWN_CHECKPOINTS: Final = (MMSELFSUP_MAE_VIT_B16_1600E,)


def _spec_mismatches(
    spec: CheckpointSpec, state: dict[str, Tensor], meta: dict, source: Path | str | None
) -> list[str]:
    """Every way this file differs from what ``spec`` describes, in words.

    Empty means the file *is* the artifact the spec names — as far as anything countable can say.
    Counts and the recorded epoch always; the md5 too when there is a path to hash, because the
    four published schedules are structurally identical and differ only in their weights and
    their filename, so counts alone would happily confirm the 300-epoch file as the 1600-epoch
    one.
    """
    counted = {
        "encoder_tensors": sum(1 for k in state if k.startswith(ENCODER_PREFIX)),
        "decoder_tensors": sum(1 for k in state if k.startswith(DECODER_PREFIX)),
        "preprocessor_tensors": sum(1 for k in state if k.startswith(PREPROCESSOR_PREFIX)),
        "encoder_parameters": sum(
            v.numel() for k, v in state.items() if k.startswith(ENCODER_PREFIX)
        ),
    }
    mismatches = [
        f"{what}: file has {found}, {spec.name} has {getattr(spec, what)}"
        for what, found in counted.items()
        if found != getattr(spec, what)
    ]
    if isinstance(meta.get("epoch"), int) and meta["epoch"] != spec.epochs:
        mismatches.append(f"epoch: file records {meta['epoch']}, {spec.name} is {spec.epochs}")
    if source is not None and not mismatches:
        # Only once the cheap checks agree: hashing 485 MB to reject a file whose tensor count is
        # already wrong would be a second read for an answer we have.
        if (size := Path(source).stat().st_size) != spec.size_bytes:
            mismatches.append(f"size: file is {size} B, {spec.name} is {spec.size_bytes} B")
        elif (digest := checkpoint_md5(source)) != spec.md5:
            mismatches.append(f"md5: file is {digest}, {spec.name} is {spec.md5}")
    return mismatches


def recognize(
    state: dict[str, Tensor], meta: dict, source: Path | str | None
) -> CheckpointSpec | None:
    """Which of :data:`KNOWN_CHECKPOINTS` this file *is*, or ``None`` if it is none of them.

    The provenance in an :class:`InitReport` is **recognized, never assumed**, and the difference
    is not academic. :meth:`InitReport.to_dict` is where #88 reads its init-provenance field
    from (P101 §7: "every released run needs to state what its encoder was initialized from and
    under what licence"), so a spec attached by default rather than by measurement would let a
    released artifact assert a URL, an md5 and **Apache-2.0** about a file that is none of those.
    On a licence claim, after P101 spent a whole ticket establishing that the licence is the
    reason this checkpoint was chosen, that is the worst available failure.

    An unrecognized file is not an error — warm-starting from the 800-epoch sibling is a
    legitimate thing to do — it just gets no provenance claim, and the report says so. What *is*
    an error is asking for a specific checkpoint and getting another one; that is
    :func:`load_mmselfsup_mae_encoder`'s ``expect``.
    """
    for spec in KNOWN_CHECKPOINTS:
        if not _spec_mismatches(spec, state, meta, source):
            return spec
    return None


# --- The report ---------------------------------------------------------------


@dataclass(frozen=True)
class InitReport:
    """What the warm start actually did, tensor by tensor.

    Returned rather than logged so the caller decides what to do with it, and so the assertions
    are testable as *values* instead of as the absence of an exception. #93's loop stamps it
    into the checkpoint beside the config, and #88's release artifact reads
    :attr:`spec` for its init-provenance field.
    """

    #: How the checkpoint was named to us — a path, or ``"<mapping>"`` for an in-memory one.
    source: str
    #: Its published identity, when it is a checkpoint we ship a spec for.
    spec: CheckpointSpec | None
    #: The epoch recorded *inside* the file, when it carries one. 1600 for R4's init, and worth
    #: reporting because it is the one place the schedule length is a fact rather than a
    #: filename — the four published schedules differ only in the name.
    source_epochs: int | None
    #: Our parameter names populated from the file, in load order.
    warm_started: tuple[str, ...]
    #: Tensors compared against a reconstruction instead of copied (:data:`POS_EMBED_KEY`).
    verified: tuple[str, ...]
    #: Their prefix -> how many tensors under it were thrown away.
    discarded: Mapping[str, int]
    #: Our parameters left at :meth:`MaskedAutoencoder._init_weights`'s values — the decoder,
    #: every time, and nothing else.
    fresh: tuple[str, ...]
    patch_embed_adapt: str
    #: Input channels the source kernel had. 3, asserted rather than assumed.
    source_in_chans: int
    #: How many scalars came out of the file — :attr:`spec`'s encoder total, less the two thirds
    #: of the patch-embed kernel that the channel collapse folds away.
    warm_started_parameters: int

    def to_dict(self) -> dict:
        """Plain JSON types, for the checkpoint and the W&B config."""
        return {
            "source": self.source,
            "spec": None if self.spec is None else {
                "name": self.spec.name,
                "url": self.spec.url,
                "md5": self.spec.md5,
                "epochs": self.spec.epochs,
                "licence": self.spec.licence,
                "licence_source": self.spec.licence_source,
            },
            "source_epochs": self.source_epochs,
            "patch_embed_adapt": self.patch_embed_adapt,
            "source_in_chans": self.source_in_chans,
            "warm_started_tensors": len(self.warm_started),
            "warm_started_parameters": self.warm_started_parameters,
            "verified_tensors": list(self.verified),
            "discarded_tensors": dict(self.discarded),
            "fresh_tensors": len(self.fresh),
        }

    def lines(self) -> list[str]:
        """The block a run prints at step 0, beside the config header (#89's `format_header`)."""
        name = self.spec.name if self.spec else "unrecognized checkpoint"
        licence = f"{self.spec.licence} ({self.spec.licence_source})" if self.spec else "unknown"
        discarded = ", ".join(f"{n} x {p}*" for p, n in sorted(self.discarded.items())) or "none"
        return [
            f"init             warm start from {name}",
            f"                 {self.source}",
            f"                 licence {licence}"
            + (f", {self.source_epochs} ImageNet epochs" if self.source_epochs else ""),
            f"                 {len(self.warm_started)} encoder tensors loaded, "
            f"{len(self.verified)} verified, {len(self.fresh)} left fresh (the decoder)",
            f"                 patch embed {self.source_in_chans} -> 1 channel by "
            f"{self.patch_embed_adapt}; discarded {discarded}",
        ]


# --- The load -----------------------------------------------------------------


def _assert_pos_embed_matches(model: MaskedAutoencoder, state: dict[str, Tensor]) -> None:
    """Refuse a model whose positional convention is not the checkpoint's (P102 §4).

    First, before a single weight moves, because it is the one layout error that a
    name-and-shape-complete load cannot surface and the only one we have an oracle for.
    """
    ours = model.pos_embed
    if state[POS_EMBED_KEY].shape != ours.shape:
        raise ValueError(
            f"{POS_EMBED_KEY} is {tuple(state[POS_EMBED_KEY].shape)}, this model's is "
            f"{tuple(ours.shape)}: the checkpoint was trained at a different grid or width"
        )
    # `read_state_dict` maps to CPU while the model may already be on the box's GPU (#93 builds
    # then moves). Everything here follows the model rather than the file.
    theirs = state[POS_EMBED_KEY].to(device=ours.device, dtype=ours.dtype)
    gap = (theirs - ours).abs().max().item()
    if gap > POS_EMBED_ATOL:
        alternative = sincos2d_pos_embed(
            ours.shape[-1], model.grid_px, axis_order="row_col"
        ).to(ours.device)
        transposed_gap = (theirs - alternative).abs().max().item()
        hint = (
            " The row-first ordering does match, so this is the transpose P102 §4 describes: "
            "build the model with pos_encoding='sincos2d_rowfirst' (RunConfig."
            "vit_base_continual does)."
            if transposed_gap <= POS_EMBED_ATOL
            else ""
        )
        raise ValueError(
            f"{POS_EMBED_KEY} disagrees with this model's positional embedding by {gap:.2e} "
            f"(tolerance {POS_EMBED_ATOL:.0e}). These weights learned one spatial convention "
            f"and this model computes another, which no key mapping can repair.{hint}"
        )


def _partition(state: dict[str, Tensor]) -> tuple[dict[str, Tensor], dict[str, int]]:
    """Split the file's keys into the encoder and the explicitly-discarded, refusing strangers.

    An unrecognized prefix raises. That is the difference between "we threw the decoder away"
    and "we threw away whatever we didn't recognize": the first is a decision, the second is how
    a checkpoint from a different model gets half-loaded.
    """
    encoder = {k: v for k, v in state.items() if k.startswith(ENCODER_PREFIX)}
    discarded = {
        prefix: sum(1 for k in state if k.startswith(prefix)) for prefix in DISCARDED_PREFIXES
    }
    known = (ENCODER_PREFIX, *DISCARDED_PREFIXES)
    strangers = sorted(k for k in state if not k.startswith(known))
    if strangers:
        raise ValueError(
            f"checkpoint carries {len(strangers)} tensor(s) under no prefix this loader knows "
            f"({', '.join(known)}): {strangers[:5]}"
        )
    if not encoder:
        raise ValueError(
            f"checkpoint has no {ENCODER_PREFIX}* tensors — this is not an mmselfsup MAE"
        )
    return encoder, discarded


def _assert_total_coverage(
    mapping: dict[str, str], theirs: dict[str, Tensor], ours: dict[str, Tensor]
) -> None:
    """Both directions, because each catches a different bug.

    Sources unmapped -> the file holds encoder weights we would silently drop. Sources missing
    -> the file is not the one this map was written for. Destinations unfilled -> a layer of our
    encoder would train from scratch inside a run reported as warm-started, which is the
    "continual pretraining mysteriously performed like scratch" failure in full.
    """
    expected_sources = set(mapping) | {POS_EMBED_KEY}
    missing = sorted(expected_sources - set(theirs))
    unmapped = sorted(set(theirs) - expected_sources)
    unfilled = sorted(set(ours) - set(mapping.values()))
    misdirected = sorted(set(mapping.values()) - set(ours))
    problems = [
        (missing, "the checkpoint is missing encoder tensor(s) this loader requires"),
        (unmapped, "the checkpoint carries encoder tensor(s) this loader does not map"),
        (unfilled, "encoder parameter(s) would be left un-warm-started"),
        (misdirected, "the key map targets parameter(s) this model does not have"),
    ]
    complaints = [f"{why}: {names[:8]}" for names, why in problems if names]
    if complaints:
        raise ValueError("; ".join(complaints))


def load_mmselfsup_mae_encoder(
    model: MaskedAutoencoder,
    checkpoint: Path | str | Mapping[str, Any],
    *,
    adapt: str = "sum",
    expect: CheckpointSpec | None = None,
) -> InitReport:
    """Warm-start ``model``'s encoder from an mmselfsup MAE checkpoint, in place.

    Order matters and is the argument of the module docstring: verify the positional convention,
    partition the file's keys and refuse strangers, prove the mapping total in both directions,
    check every shape, and only then copy. Nothing is loaded on the way to discovering a problem,
    so a rejected checkpoint leaves the model exactly as :meth:`MaskedAutoencoder.from_config`
    built it rather than half-overwritten.

    The decoder is not touched: every destination in :func:`encoder_key_map` is asserted to be on
    the encoder side of :func:`~pretrain.model.is_decoder_parameter`, so "the decoder is fresh"
    is a property of the mapping and not a promise in a docstring.

    ``expect`` turns "which checkpoint is this?" from a recorded observation into a **requirement**:
    given one, a file that is not measurably that artifact is refused. Left ``None`` the file is
    matched against :data:`KNOWN_CHECKPOINTS` and whatever it turns out to be — possibly nothing
    — is what the report claims. #86's pre-flight on the box is the caller that should pass it:
    "the file I fetched is the file plan §8 names" is exactly what a pre-flight is for.
    """
    state, meta = read_state_dict(checkpoint)
    is_path = isinstance(checkpoint, (str, Path))
    source = str(checkpoint) if is_path else "<mapping>"
    path = checkpoint if is_path else None

    if expect is not None:
        if mismatches := _spec_mismatches(expect, state, meta, path):
            raise ValueError(
                f"this checkpoint was required to be {expect.name} and is not, so loading it "
                f"would file the run under that URL and its {expect.licence} licence: "
                + "; ".join(mismatches)
            )
        spec = expect
    else:
        spec = recognize(state, meta, path)

    theirs, discarded = _partition(state)
    if POS_EMBED_KEY not in theirs:
        raise ValueError(
            f"checkpoint has no {POS_EMBED_KEY}: it is the one tensor this loader verifies "
            "rather than copies (P102 §4), and without it the spatial convention is unchecked"
        )
    _assert_pos_embed_matches(model, theirs)

    ours = {n: p for n, p in model.named_parameters() if not is_decoder_parameter(n)}
    mapping = encoder_key_map(model.arch.encoder_depth)
    _assert_total_coverage(mapping, theirs, ours)

    decoder_targets = sorted(t for t in mapping.values() if is_decoder_parameter(t))
    if decoder_targets:
        raise ValueError(f"the key map would write decoder parameters: {decoder_targets}")

    patch_embed_key = f"{ENCODER_PREFIX}patch_embed.projection.weight"
    source_in_chans = int(theirs[patch_embed_key].shape[1])
    if source_in_chans != 3:
        raise ValueError(
            f"{patch_embed_key} reads {source_in_chans} channels; §8's collapse is written for "
            "the 3-channel RGB kernel an ImageNet model has"
        )

    # Two passes. Everything is prepared and shape-checked before anything is written, so the
    # model is never left in a state that is neither the old one nor the new one.
    prepared: dict[str, Tensor] = {}
    for their_key, our_key in mapping.items():
        tensor = theirs[their_key]
        if their_key == patch_embed_key:
            tensor = collapse_input_channels(tensor, adapt=adapt)
        target = ours[our_key]
        if tensor.shape != target.shape:
            raise ValueError(
                f"{their_key} is {tuple(tensor.shape)} but {our_key} is "
                f"{tuple(target.shape)} — this checkpoint is not this architecture"
            )
        if not tensor.is_floating_point():
            raise ValueError(f"{their_key} is {tensor.dtype}, not a float tensor")
        prepared[our_key] = tensor

    with torch.no_grad():
        for our_key, tensor in prepared.items():
            # `copy_` casts and crosses devices, so a model already moved to the box's GPU takes
            # a CPU-mapped checkpoint without the caller having to sequence the two.
            ours[our_key].copy_(tensor)

    return InitReport(
        source=source,
        spec=spec,
        source_epochs=meta.get("epoch") if isinstance(meta.get("epoch"), int) else None,
        warm_started=tuple(prepared),
        verified=(POS_EMBED_KEY,),
        discarded={p: n for p, n in discarded.items() if n},
        fresh=tuple(n for n, _ in model.named_parameters() if is_decoder_parameter(n)),
        patch_embed_adapt=adapt,
        source_in_chans=source_in_chans,
        warm_started_parameters=sum(t.numel() for t in prepared.values()),
    )


def initialize_from_config(model: MaskedAutoencoder, config: RunConfig) -> InitReport | None:
    """Apply plan §8's initialization for ``config`` — the seam R4 and R5 differ at.

    Returns ``None`` for a scratch run, having done nothing at all: the model
    :meth:`MaskedAutoencoder.from_config` built is already R1/R5's initialization, and the
    scratch path having no branch in it is what keeps this ticket from being able to change it.
    """
    init: Init = config.init
    if init.mode == "scratch":
        return None
    if init.mode != "imagenet_mae":  # pragma: no cover — the config validates the vocabulary
        raise ValueError(f"unknown init mode {init.mode!r}")
    if init.checkpoint is None:  # pragma: no cover — RunConfig refuses this at construction
        raise ValueError("init.mode 'imagenet_mae' needs init.checkpoint")
    return load_mmselfsup_mae_encoder(
        model, init.checkpoint, adapt=init.patch_embed_adapt
    )


# --- The behavioural check ----------------------------------------------------

#: Floors calibrated on the real checkpoint, which scores +0.47 to +0.80 and +1.90 to +1.96
#: across six seeds and four batch sizes (P102 §6.1). Set well below that: what they detect is an
#: encoder that cannot tell one tile from another, which is a catastrophic rather than a marginal
#: failure, and a snug floor would only buy false alarms on the publisher's other three
#: schedules. :attr:`MIN_CROP_MARGIN` does assume :func:`encoder_behaviour`'s default
#: ``shift_px`` — at 96 px of shift the crops share too little terrain and even the correct
#: checkpoint scores −0.06, which is a fact about the stimulus and not about the weights.
MIN_CROP_MARGIN: Final = 0.15
MIN_NOISE_MARGIN: Final = 0.50
#: The blow-up guard, and the one threshold here that catches a *scramble* rather than a corpse.
#: The real checkpoint's per-block amplification profile is a remarkably stable fingerprint —
#: 1.46, then ~1.0 for ten blocks, then 2.15 — landing in **2.138 to 2.165** across six seeds,
#: three batch sizes, three crop shifts, three field roughnesses and even white-noise input. A
#: 1.3% spread, so 2.5 sits many multiples of the observed variation above it while catching the
#: two mapping scrambles a wrong :func:`encoder_key_map` would produce: reversed block order lands
#: at 3.81 and swapping each block's two LayerNorms at 2.81.
#:
#: **This number is calibrated on the 1600-epoch checkpoint.** The other three published
#: schedules were not measured — that is a second 466 MB download to bound a smoke test — so a
#: run that substitutes one may need it re-measured. The failure prints the value beside the
#: ceiling for exactly that reason, and what it costs when wrong is a human reading a number
#: during #86's pre-flight, not a bad run.
MAX_BLOCK_AMPLIFICATION: Final = 2.5


def smooth_fields(
    count: int, px: int, *, generator: torch.Generator | None = None, wavelength_px: int = 32
) -> Tensor:
    """``(count, 1, px, px)`` of smooth, terrain-like random fields.

    The check's stimulus, and synthesized rather than read from disk on purpose: it needs no
    image files on the rented box, no `PIL`, and no licence question about a photograph. Smooth
    fields are also a *better* probe than a photograph here, because they are what the encoder
    will actually be asked about — the question is whether the weights compute structure, and
    white noise is the contrast that makes "structure" measurable.

    Built by bicubic upsampling of a coarse white-noise grid, then standardized per tile exactly
    as the loader standardizes a tile (plan §5, `per_tile`), so nothing in the comparison below
    can be won on a difference of mean or scale.
    """
    coarse = max(2, -(-px // wavelength_px) + 1)
    seed = torch.randn(count, 1, coarse, coarse, generator=generator)
    field = F.interpolate(seed, size=(px, px), mode="bicubic", align_corners=False)
    return _standardize(field)


def _standardize(tiles: Tensor) -> Tensor:
    """Per-tile zero mean, unit variance — what `per_tile` normalization hands the model."""
    flat = tiles.flatten(1)
    mean = flat.mean(dim=1).view(-1, 1, 1, 1)
    std = flat.std(dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)
    return (tiles - mean) / std


@contextmanager
def _frozen(model: MaskedAutoencoder):
    """Run a measurement: `eval`, no grad, and the training mode put back afterwards.

    The restore matters more than it looks — a measurement that left the model in `eval` would
    silently disable whatever regularization the run was configured with, and #86's pre-flight
    calls this on the model it is about to train.
    """
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            yield
    finally:
        model.train(was_training)


def _pooled_features(model: MaskedAutoencoder, tiles: Tensor) -> Tensor:
    """Mean-pooled patch tokens, ``(N, encoder_dim)`` — the frozen-feature view of a tile.

    Patch tokens rather than CLS: MAE's own linear probe needs an extra normalization layer on
    CLS to be readable, and mean pooling is what the kNN probes in this lineage use. Either
    would do for a *relative* comparison; this one needs no fitted layer.
    """
    with _frozen(model):
        return model.encode(tiles)[:, 1:, :].mean(dim=1)


def _cosine(a: Tensor, b: Tensor) -> Tensor:
    return F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).T


@dataclass(frozen=True)
class EncoderBehaviour:
    """Whether an encoder's features carry structure — three measurements and a verdict.

    ## What this catches, measured rather than asserted

    #102 asked for "a dependency-free behavioural check that a scrambled-but-name-complete load
    fails", suggesting these statistics, and expected it to catch gross scrambling while missing
    a permutation *within* attention heads. **That expectation was tested against the real
    checkpoint and is too optimistic**, so this docstring says what the numbers said (the table
    is in P102 §6.1). Eight shape-preserving scrambles were measured across six candidate
    statistics. Neither statistic below separates a correct load from an arbitrary scramble —
    several scrambles score *higher* than the correct load, because two 86%-overlapping crops are
    trivially close under any smooth function, and any network at all separates smooth fields
    from white noise.

    What this check does reliably catch is an encoder that **cannot tell one tile from another**:
    collapsed, saturated, NaN, or exploding. That is the shape a catastrophically wrong load
    takes, and it is worth a gate. What it does *not* catch is a subtly wrong one.

    ## Why that is acceptable here, and where the real defence is

    The scrambles it misses — a fused-`qkv` in the wrong order, a transposed matrix, a
    within-head permutation — are **not reachable by this loader**. It performs no permutation,
    no transposition and no reshape on any tensor; the mapping copies whole tensors, so a wrong
    qkv order would require code that does not exist. That those conventions match is P102 §3's
    claim table, read from the source and linked, which is what #103's policy asks for.

    The failures that *are* reachable — a tensor landing in a same-shaped wrong slot, a block
    index off by one, the positional axis order — are covered by three things that do work: the
    both-directions coverage assertion, the `pos_embed` equivalence against the checkpoint's own
    tensor, and `test_every_mapped_key_is_the_one_p102_documents`, which pins all 149 pairs.
    """

    #: Mean cosine similarity between two overlapping crops of the *same* field, centred over
    #: the field crops only.
    within_field: float
    #: ... and between crops of *different* fields, same centring. Lower, if features track
    #: content at all. :attr:`crop_margin` is the pair that means something.
    between_fields: float
    #: Mean cosine between paired field crops, this time centred over the whole set including
    #: the noise tiles — the other axis, and the baseline :attr:`noise_margin` reads against.
    field_coherence: float
    #: ... and between field crops and white noise, same centring.
    field_versus_noise: float
    #: The largest ratio any single block applies to the residual stream's norm.
    max_block_amplification: float

    @property
    def crop_margin(self) -> float:
        """Whether features distinguish *which* field they are looking at. ~0 means every tile
        maps to the same place, which is what a dead patch embed or a saturated stack looks
        like."""
        return self.within_field - self.between_fields

    @property
    def noise_margin(self) -> float:
        """Whether "terrain-like" means anything to these weights. Both terms come from the
        same centring, so the difference is a difference."""
        return self.field_coherence - self.field_versus_noise

    @property
    def passed(self) -> bool:
        return (
            self.crop_margin >= MIN_CROP_MARGIN
            and self.noise_margin >= MIN_NOISE_MARGIN
            and self.max_block_amplification <= MAX_BLOCK_AMPLIFICATION
        )

    def lines(self) -> list[str]:
        verdict = "pass" if self.passed else "FAIL"
        return [
            f"encoder behaviour  {verdict}",
            f"  crop margin      {self.crop_margin:+.3f} "
            f"(within {self.within_field:+.3f} - between {self.between_fields:+.3f}), "
            f"floor {MIN_CROP_MARGIN:+.2f}",
            f"  noise margin     {self.noise_margin:+.3f} "
            f"(fields {self.field_coherence:+.3f} - noise {self.field_versus_noise:+.3f}), "
            f"floor {MIN_NOISE_MARGIN:+.2f}",
            f"  block gain       {self.max_block_amplification:.2f} max, "
            f"ceiling {MAX_BLOCK_AMPLIFICATION:.1f}",
        ]


def _block_amplifications(model: MaskedAutoencoder, tiles: Tensor) -> list[float]:
    """Each block's multiplier on the residual stream's mean token norm.

    Reported because it is the most *stable* thing measured here — the real checkpoint's profile
    moves by 1.3% across seeds, field roughness and white-noise input — which makes it a usable
    blow-up guard even though it is a poor scramble detector.

    Measured with forward hooks over :meth:`MaskedAutoencoder.encode` rather than by stepping the
    blocks by hand. Stepping them would mean a second copy of "embed, prepend the CLS token, add
    the positional embedding" living in a diagnostic — which is the exact duplication
    :meth:`MaskedAutoencoder.embed_tokens` was factored out to prevent, and a diagnostic that
    measured a slightly different forward pass than the model runs is worse than no diagnostic.
    """
    gains: list[float] = []

    def record(_module, inputs, output) -> None:
        before = inputs[0].norm(dim=-1).mean()
        gains.append((output.norm(dim=-1).mean() / before.clamp_min(1e-12)).item())

    handles = [block.register_forward_hook(record) for block in model.blocks]
    try:
        with _frozen(model):
            model.encode(tiles)
    finally:
        for handle in handles:
            handle.remove()
    return gains


def encoder_behaviour(
    model: MaskedAutoencoder,
    *,
    count: int = 8,
    shift_px: int = 32,
    generator: torch.Generator | None = None,
) -> EncoderBehaviour:
    """Measure whether ``model``'s encoder computes structure. See :class:`EncoderBehaviour`.

    ``shift_px`` is how far the second crop of a field is offset from the first: far enough that
    the two tiles are genuinely different inputs, near enough that they are the same terrain.

    Two different centrings, and the difference is not cosmetic. Raw cosines between pooled ViT
    features are ~0.95 for *everything*, because one shared direction dominates the geometry, so
    every number has to be read against a mean. The crop statistics centre over the field crops
    **only** — centring over the noise tiles too makes the field/noise axis eat all the variance
    and crushes within- and between-field down to a 0.03 gap. The noise statistic centres over
    the whole set, because that comparison *is* the field/noise axis.
    """
    px = model.img_px
    # The stimulus is drawn on the CPU — a `torch.Generator` belongs to one device and a seeded
    # measurement should not change with the box — then moved to wherever the model lives.
    device = next(model.parameters()).device
    fields = smooth_fields(count, px + shift_px, generator=generator)
    first = _standardize(fields[:, :, :px, :px]).to(device)
    second = _standardize(fields[:, :, shift_px:, shift_px:]).to(device)
    noise = _standardize(torch.randn(count, 1, px, px, generator=generator)).to(device)

    features = _pooled_features(model, torch.cat([first, second, noise], dim=0))
    a, b = features[:count], features[count : 2 * count]

    crop_centre = torch.cat([a, b]).mean(dim=0, keepdim=True)
    similarity = _cosine(a - crop_centre, b - crop_centre)
    off_diagonal = ~torch.eye(count, dtype=torch.bool, device=similarity.device)

    centred = features - features.mean(dim=0, keepdim=True)
    fields_versus_noise = _cosine(centred[: 2 * count], centred[2 * count :])
    field_pairs = _cosine(centred[:count], centred[count : 2 * count]).diagonal()

    return EncoderBehaviour(
        within_field=similarity.diagonal().mean().item(),
        between_fields=similarity[off_diagonal].mean().item(),
        field_coherence=field_pairs.mean().item(),
        field_versus_noise=fields_versus_noise.mean().item(),
        max_block_amplification=max(_block_amplifications(model, first)),
    )
