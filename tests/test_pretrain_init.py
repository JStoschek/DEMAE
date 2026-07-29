"""Warm-starting the encoder from an mmselfsup ImageNet MAE checkpoint (#102, plan §8).

The tests fall into five groups, and the split follows the three layers `pretrain.init`'s
docstring argues for — because each layer catches a different failure and only the first one is
the sort that makes a test suite feel thorough.

*Names* is the boring, load-bearing group: the mapping is total in both directions, an
incomplete load raises, and every one of the 149 pairs is the pair
[P102](../docs/research/p102-mmselfsup-mae-layout.md) §3 documents. This is where a
same-shaped-wrong-slot bug gets caught, since no output statistic reliably would.

*Layout* is the group #103 asked for. The claim that matters is checked against mmselfsup's own
source, transcribed here rather than imported: their fixed sin-cos positional embedding is the
**transpose** of MAE's, so a model built with MAE's ordering must be *refused*, loudly, before
any weight moves.

*Surgery* pins the patch-embed collapse as an exact equality — the summed kernel reproduces the
3-channel convolution, the mean reproduces exactly a third of it — which is what makes plan §8's
"sum preserves activation scale" a checked property rather than a citation.

*Rejection* is the deliberately-broken-checkpoint group.

*Behaviour and training* is the end-to-end group: the loss falls on a CPU fixture, the scratch
path is untouched, and the behavioural check catches an encoder that cannot tell two tiles apart.

Everything here runs on a **two-block, 32-wide** model against a synthesized checkpoint, so the
suite stays in seconds and never touches the network. The 485 MB real file is exercised by the
opt-in tests at the bottom, which is also where the numbers in P102 §6.1 come from; they skip
unless `DEMAE_MAE_CHECKPOINT` points at it.

One consequence of that split is worth stating rather than leaving to be discovered: **the
behavioural check's discriminating tests can only live in the opt-in group.** A synthetic
checkpoint holds random values, so a "correctly loaded" and a "scrambled" tiny model are both
random and there is no difference for any statistic to find. What the CPU group can prove is that
the floors fire on a blind encoder and the ceiling fires on an exploding one; that a *scramble* of
trained weights fails is
`test_the_behavioural_check_catches_the_reachable_scrambles_on_real_weights`, and it needs the
real file. Nothing in CI covers the literal wording of #102's fourth acceptance
criterion.
"""

from __future__ import annotations

import copy
import math
import os
import warnings
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from pretrain import init as pretrain_init
from pretrain.config import POS_ENCODINGS, Architecture, Init, RunConfig
from pretrain.init import (
    DECODER_PREFIX,
    ENCODER_PREFIX,
    MAX_BLOCK_AMPLIFICATION,
    MIN_CROP_MARGIN,
    MIN_NOISE_MARGIN,
    MMSELFSUP_MAE_VIT_B16_1600E,
    POS_EMBED_KEY,
    PREPROCESSOR_PREFIX,
    collapse_input_channels,
    encoder_behaviour,
    encoder_key_map,
    initialize_from_config,
    load_mmselfsup_mae_encoder,
)
from pretrain.model import (
    AXIS_ORDERS,
    MLP_RATIO,
    MaskedAutoencoder,
    is_decoder_parameter,
)

#: A tiny stand-in for ViT-B: two blocks, 32 wide, a 2x2 patch grid. The mapping is a function
#: of `encoder_depth` precisely so the whole of it can be exercised at this size — building the
#: real thing costs 86M parameters and three seconds per test.
TINY = Architecture(
    encoder_dim=32,
    encoder_depth=2,
    encoder_heads=4,
    decoder_dim=16,
    decoder_depth=1,
    decoder_heads=4,
    img_px=32,
    patch_px=16,
)

#: Their decoder's depth in the file we synthesize. Only the count matters — every `neck.` tensor
#: is discarded — but it has to be *there*, because "the decoder is discarded" is an assertion
#: about tensors that exist.
THEIR_DECODER_DEPTH = 3


class Unloadable:
    """A picklable class standing in for the `mmengine` objects the real checkpoint carries.

    Module-level because pickle resolves classes by qualified name and cannot see a local one.
    Its whole job is to be a global that `torch.load`'s allowlist does not contain.
    """

    def __init__(self, **state: int) -> None:
        self.state = state


def _generator(seed: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _tiny_config(**changes) -> RunConfig:
    return RunConfig(architecture=TINY, **changes)


def _loss_for(config: RunConfig):
    """§4's objective with the ocean weighting off, for the two training-step tests.

    Off deliberately: #96's weighting needs the loader's `land_mask`, and these tests are about
    whether a *warm-started* model trains at all. Leaving it on would mean synthesizing pixel
    weights here and coupling this file to another ticket's argument list.
    """
    from pretrain.loss import MaeLoss

    return MaeLoss.from_config(
        config.replace(loss=config.loss.replace(ocean_weighted=False))
    )


def mmselfsup_pos_embed(
    patches_resolution: int, embed_dims: int, temperature: float = 10000.0
) -> torch.Tensor:
    """`build_2d_sincos_position_embedding`, transcribed verbatim from mmselfsup v1.0.0.

    Transcribed and not imported, for the same reason `facebookresearch/mae`'s own construction
    is transcribed in `test_pretrain_model.py`: mmselfsup is *spec* here, not a dependency, and
    #103 declined an `mmcv` test dependency explicitly. The subtlety this preserves is the whole
    finding — `torch.meshgrid` defaults to ``'ij'`` indexing, so the half named `out_w` carries
    the **row** after a row-major flatten, which is the opposite of what `np.meshgrid`'s ``'xy'``
    default does for MAE.

    Source: `mmselfsup/models/utils/position_embedding.py` at v1.0.0 —
    https://github.com/open-mmlab/mmselfsup/blob/v1.0.0/mmselfsup/models/utils

    The `indexing=` argument is left off deliberately, so this stays a transcription rather than
    a correction — torch's own deprecation warning for omitting it *is* the trap, and it is
    silenced here rather than heeded because heeding it would change the tensor under test.
    """
    h = w = patches_resolution
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*indexing argument.*")
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
    pos_dim = embed_dims // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1.0 / (temperature**omega)
    out_w = torch.einsum("m,d->md", [grid_w.flatten(), omega])
    out_h = torch.einsum("m,d->md", [grid_h.flatten(), omega])
    pos_emb = torch.cat(
        [torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1
    )[None, :, :]
    return torch.cat([torch.zeros(1, 1, embed_dims), pos_emb], dim=1)


def synthetic_checkpoint(
    arch: Architecture = TINY,
    *,
    seed: int = 0,
    decoder_depth: int = THEIR_DECODER_DEPTH,
    in_chans: int = 3,
) -> dict:
    """A checkpoint shaped exactly like the real one: mmselfsup's names, at ``arch``'s sizes.

    Every key here was read off the real file (P102 §1.2), including the two
    `data_preprocessor.` constants that are neither encoder nor decoder and that the loader has
    to discard *by name* rather than by not noticing. The values are random; what these tests
    check is where they end up.
    """
    g = _generator(seed)
    dim, depth, grid = arch.encoder_dim, arch.encoder_depth, arch.img_px // arch.patch_px
    hidden = int(dim * MLP_RATIO)  # their FFN width is 4x, same constant our blocks use
    decoder_dim = dim // 2

    def r(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=g)

    state: dict[str, torch.Tensor] = {
        f"{PREPROCESSOR_PREFIX}mean": r(3, 1, 1),
        f"{PREPROCESSOR_PREFIX}std": r(3, 1, 1),
        f"{ENCODER_PREFIX}cls_token": r(1, 1, dim),
        POS_EMBED_KEY: mmselfsup_pos_embed(grid, dim),
        f"{ENCODER_PREFIX}patch_embed.projection.weight": r(
            dim, in_chans, arch.patch_px, arch.patch_px
        ),
        f"{ENCODER_PREFIX}patch_embed.projection.bias": r(dim),
        f"{ENCODER_PREFIX}ln1.weight": r(dim),
        f"{ENCODER_PREFIX}ln1.bias": r(dim),
    }
    for i in range(depth):
        p = f"{ENCODER_PREFIX}layers.{i}."
        state |= {
            f"{p}ln1.weight": r(dim), f"{p}ln1.bias": r(dim),
            f"{p}attn.qkv.weight": r(3 * dim, dim), f"{p}attn.qkv.bias": r(3 * dim),
            f"{p}attn.proj.weight": r(dim, dim), f"{p}attn.proj.bias": r(dim),
            f"{p}ln2.weight": r(dim), f"{p}ln2.bias": r(dim),
            f"{p}ffn.layers.0.0.weight": r(hidden, dim),
            f"{p}ffn.layers.0.0.bias": r(hidden),
            f"{p}ffn.layers.1.weight": r(dim, hidden), f"{p}ffn.layers.1.bias": r(dim),
        }
    state |= {
        f"{DECODER_PREFIX}mask_token": r(1, 1, decoder_dim),
        f"{DECODER_PREFIX}decoder_pos_embed": r(1, grid * grid + 1, decoder_dim),
        f"{DECODER_PREFIX}decoder_embed.weight": r(decoder_dim, dim),
        f"{DECODER_PREFIX}decoder_embed.bias": r(decoder_dim),
        f"{DECODER_PREFIX}ln1.weight": r(decoder_dim),
        f"{DECODER_PREFIX}ln1.bias": r(decoder_dim),
        f"{DECODER_PREFIX}decoder_pred.weight": r(
            arch.patch_px * arch.patch_px * in_chans, decoder_dim
        ),
        f"{DECODER_PREFIX}decoder_pred.bias": r(arch.patch_px * arch.patch_px * in_chans),
    }
    for i in range(decoder_depth):
        p = f"{DECODER_PREFIX}decoder_blocks.{i}."
        state |= {
            f"{p}ln1.weight": r(decoder_dim), f"{p}ln1.bias": r(decoder_dim),
            f"{p}attn.qkv.weight": r(3 * decoder_dim, decoder_dim),
            f"{p}attn.qkv.bias": r(3 * decoder_dim),
            f"{p}attn.proj.weight": r(decoder_dim, decoder_dim),
            f"{p}attn.proj.bias": r(decoder_dim),
            f"{p}ln2.weight": r(decoder_dim), f"{p}ln2.bias": r(decoder_dim),
            f"{p}ffn.layers.0.0.weight": r(int(decoder_dim * MLP_RATIO), decoder_dim),
            f"{p}ffn.layers.0.0.bias": r(int(decoder_dim * MLP_RATIO)),
            f"{p}ffn.layers.1.weight": r(decoder_dim, int(decoder_dim * MLP_RATIO)),
            f"{p}ffn.layers.1.bias": r(decoder_dim),
        }
    return {"meta": {"epoch": 1600}, "state_dict": state, "message_hub": {}}


@pytest.fixture
def checkpoint() -> dict:
    return synthetic_checkpoint()


@pytest.fixture
def model() -> MaskedAutoencoder:
    """A tiny model built the way R4 builds one — mmselfsup's positional ordering."""
    return MaskedAutoencoder.from_config(
        _tiny_config().replace(
            architecture=TINY.replace(pos_encoding="sincos2d_rowfirst")
        )
    )


# --- Names: the mapping, and that nothing is silently skipped ------------------


def test_the_key_map_covers_every_encoder_parameter_and_nothing_else(model, checkpoint):
    report = load_mmselfsup_mae_encoder(model, checkpoint)
    encoder_side = {n for n, _ in model.named_parameters() if not is_decoder_parameter(n)}
    # Every encoder parameter warm-started, exactly once, and no decoder parameter touched.
    assert set(report.warm_started) == encoder_side
    assert len(report.warm_started) == len(encoder_side)
    assert not [n for n in report.warm_started if is_decoder_parameter(n)]


def test_every_source_tensor_is_loaded_verified_or_deliberately_discarded(model, checkpoint):
    report = load_mmselfsup_mae_encoder(model, checkpoint)
    state = checkpoint["state_dict"]
    accounted = (
        len(report.warm_started)
        + len(report.verified)
        + sum(report.discarded.values())
    )
    # No third category. A tensor the loader neither used nor named is the failure mode the
    # whole module exists to prevent, so the arithmetic is asserted rather than the intent.
    assert accounted == len(state)
    assert report.discarded == {
        DECODER_PREFIX: 12 * THEIR_DECODER_DEPTH + 8,
        PREPROCESSOR_PREFIX: 2,
    }
    assert report.verified == (POS_EMBED_KEY,)


def test_every_mapped_key_is_the_one_p102_documents():
    """The mapping, pair by pair, against §3's claim table.

    This is the test that catches a same-shaped wrong slot — `ln1` into `norm2`, block 0 into
    block 1 — and it exists because the behavioural check demonstrably cannot: five of the six
    (768,) tensors in a block are interchangeable by shape, and the measured output statistics
    do not separate a swap from a correct load (P102 §6.1). A reviewer checking this list against
    the table is the actual defence.
    """
    mapping = encoder_key_map(depth=2)
    assert mapping == {
        # L2 — mmcv's PatchEmbed calls its conv `projection`; timm calls it `proj`.
        f"{ENCODER_PREFIX}patch_embed.projection.weight": "patch_embed.proj.weight",
        f"{ENCODER_PREFIX}patch_embed.projection.bias": "patch_embed.proj.bias",
        f"{ENCODER_PREFIX}cls_token": "cls_token",
        # L8 — the *final* norm, `ln1` at top level, not a block's `ln1`.
        f"{ENCODER_PREFIX}ln1.weight": "norm.weight",
        f"{ENCODER_PREFIX}ln1.bias": "norm.bias",
        # L6 — pre-norm block: ln1 before attention, ln2 before the FFN.
        f"{ENCODER_PREFIX}layers.0.ln1.weight": "blocks.0.norm1.weight",
        f"{ENCODER_PREFIX}layers.0.ln1.bias": "blocks.0.norm1.bias",
        f"{ENCODER_PREFIX}layers.0.ln2.weight": "blocks.0.norm2.weight",
        f"{ENCODER_PREFIX}layers.0.ln2.bias": "blocks.0.norm2.bias",
        # L3 — one fused qkv, q/k/v in that order, untransposed, copied whole.
        f"{ENCODER_PREFIX}layers.0.attn.qkv.weight": "blocks.0.attn.qkv.weight",
        f"{ENCODER_PREFIX}layers.0.attn.qkv.bias": "blocks.0.attn.qkv.bias",
        f"{ENCODER_PREFIX}layers.0.attn.proj.weight": "blocks.0.attn.proj.weight",
        f"{ENCODER_PREFIX}layers.0.attn.proj.bias": "blocks.0.attn.proj.bias",
        # L7 — `ffn.layers.0.0` is fc1 (the 0.0 is a Sequential of Linear, GELU, Dropout);
        # `ffn.layers.1` is fc2.
        f"{ENCODER_PREFIX}layers.0.ffn.layers.0.0.weight": "blocks.0.mlp.fc1.weight",
        f"{ENCODER_PREFIX}layers.0.ffn.layers.0.0.bias": "blocks.0.mlp.fc1.bias",
        f"{ENCODER_PREFIX}layers.0.ffn.layers.1.weight": "blocks.0.mlp.fc2.weight",
        f"{ENCODER_PREFIX}layers.0.ffn.layers.1.bias": "blocks.0.mlp.fc2.bias",
        f"{ENCODER_PREFIX}layers.1.ln1.weight": "blocks.1.norm1.weight",
        f"{ENCODER_PREFIX}layers.1.ln1.bias": "blocks.1.norm1.bias",
        f"{ENCODER_PREFIX}layers.1.ln2.weight": "blocks.1.norm2.weight",
        f"{ENCODER_PREFIX}layers.1.ln2.bias": "blocks.1.norm2.bias",
        f"{ENCODER_PREFIX}layers.1.attn.qkv.weight": "blocks.1.attn.qkv.weight",
        f"{ENCODER_PREFIX}layers.1.attn.qkv.bias": "blocks.1.attn.qkv.bias",
        f"{ENCODER_PREFIX}layers.1.attn.proj.weight": "blocks.1.attn.proj.weight",
        f"{ENCODER_PREFIX}layers.1.attn.proj.bias": "blocks.1.attn.proj.bias",
        f"{ENCODER_PREFIX}layers.1.ffn.layers.0.0.weight": "blocks.1.mlp.fc1.weight",
        f"{ENCODER_PREFIX}layers.1.ffn.layers.0.0.bias": "blocks.1.mlp.fc1.bias",
        f"{ENCODER_PREFIX}layers.1.ffn.layers.1.weight": "blocks.1.mlp.fc2.weight",
        f"{ENCODER_PREFIX}layers.1.ffn.layers.1.bias": "blocks.1.mlp.fc2.bias",
    }


def test_the_mapping_is_injective_at_the_real_depth():
    # Two sources landing in one destination is a load that silently drops a tensor while every
    # coverage count still adds up.
    mapping = encoder_key_map(depth=12)
    assert len(set(mapping.values())) == len(mapping) == 149


def test_the_loaded_values_are_the_files_values(model, checkpoint):
    load_mmselfsup_mae_encoder(model, checkpoint)
    state = checkpoint["state_dict"]
    for their_key, our_key in encoder_key_map(TINY.encoder_depth).items():
        ours = dict(model.named_parameters())[our_key]
        if "patch_embed" in their_key and their_key.endswith("weight"):
            continue  # the one tensor that is transformed; its own test is below
        assert torch.equal(ours, state[their_key]), our_key


def test_a_checkpoint_missing_one_encoder_tensor_is_refused(model, checkpoint):
    del checkpoint["state_dict"][f"{ENCODER_PREFIX}layers.1.attn.qkv.weight"]
    with pytest.raises(ValueError, match="missing encoder tensor"):
        load_mmselfsup_mae_encoder(model, checkpoint)


def test_a_checkpoint_with_an_extra_encoder_tensor_is_refused(model, checkpoint):
    # The dangerous direction: an unmapped source is a weight we would silently leave behind.
    checkpoint["state_dict"][f"{ENCODER_PREFIX}layers.0.attn.relative_bias"] = torch.zeros(3)
    with pytest.raises(ValueError, match="does not map"):
        load_mmselfsup_mae_encoder(model, checkpoint)


def test_a_deeper_model_than_the_checkpoint_is_refused(checkpoint):
    deeper = MaskedAutoencoder.from_config(
        _tiny_config().replace(
            architecture=TINY.replace(encoder_depth=4, pos_encoding="sincos2d_rowfirst")
        )
    )
    with pytest.raises(ValueError, match="missing encoder tensor"):
        load_mmselfsup_mae_encoder(deeper, checkpoint)


def test_nothing_is_loaded_when_the_checkpoint_is_rejected(model, checkpoint):
    """A rejected load leaves the model exactly as it was built, not half-overwritten.

    Everything is prepared and shape-checked before the first `copy_`, which is what makes a
    failed warm start recoverable rather than a model in a state that is neither.
    """
    before = {n: p.clone() for n, p in model.named_parameters()}
    del checkpoint["state_dict"][f"{ENCODER_PREFIX}ln1.weight"]
    with pytest.raises(ValueError):
        load_mmselfsup_mae_encoder(model, checkpoint)
    assert all(torch.equal(before[n], p) for n, p in model.named_parameters())


# --- Layout: the ordering hole #103 named --------------------------------------


def test_mmselfsups_positional_embedding_is_the_transpose_of_maes():
    """P102 §4, checked against both source implementations rather than against a shape.

    This is the finding the whole layout policy exists for: every name populates, every shape
    matches, and the spatial convention is nonetheless the mirror image. It is not hypothetical
    — it is the checkpoint R4 initializes from.
    """
    from pretrain.model import sincos2d_pos_embed

    grid, dim = 14, 768
    theirs = mmselfsup_pos_embed(grid, dim)
    mae_order = sincos2d_pos_embed(dim, grid, axis_order="col_row")
    row_order = sincos2d_pos_embed(dim, grid, axis_order="row_col")

    assert not torch.allclose(theirs, mae_order, atol=1e-5)
    assert torch.allclose(theirs, row_order, atol=1e-5)
    # Not "a bit different": a full sinusoid apart, so no tolerance could blur the two.
    assert (theirs - mae_order).abs().max() > 1.0


def test_a_model_with_maes_positional_ordering_is_refused_by_name(checkpoint):
    mae_ordered = MaskedAutoencoder.from_config(_tiny_config())
    assert mae_ordered.arch.pos_encoding == "sincos2d"  # the default, R1's and R5's
    with pytest.raises(ValueError, match="sincos2d_rowfirst"):
        load_mmselfsup_mae_encoder(mae_ordered, checkpoint)


def test_the_positional_check_runs_before_any_weight_is_copied(checkpoint):
    mae_ordered = MaskedAutoencoder.from_config(_tiny_config())
    before = mae_ordered.blocks[0].attn.qkv.weight.clone()
    with pytest.raises(ValueError):
        load_mmselfsup_mae_encoder(mae_ordered, checkpoint)
    assert torch.equal(before, mae_ordered.blocks[0].attn.qkv.weight)


def test_a_checkpoint_with_no_positional_embedding_is_refused(model, checkpoint):
    # It is the only oracle in the file; a checkpoint without it is one we cannot check at all,
    # which is a refusal rather than a warning.
    del checkpoint["state_dict"][POS_EMBED_KEY]
    with pytest.raises(ValueError, match="no backbone.pos_embed"):
        load_mmselfsup_mae_encoder(model, checkpoint)


def test_a_positional_embedding_at_another_grid_is_refused(model, checkpoint):
    checkpoint["state_dict"][POS_EMBED_KEY] = mmselfsup_pos_embed(7, TINY.encoder_dim)
    with pytest.raises(ValueError, match="different grid"):
        load_mmselfsup_mae_encoder(model, checkpoint)


def test_every_config_positional_encoding_has_an_axis_order():
    # Two lists in two modules — the torch-free config names the knob, the model maps it — so
    # this is what stops one of them from gaining a value the other cannot build.
    assert set(AXIS_ORDERS) == set(POS_ENCODINGS)


# --- Surgery: the patch-embed channel collapse --------------------------------


def test_the_summed_kernel_reproduces_the_three_channel_convolution_exactly():
    """Plan §8's "sum preserves activation scale", as an equality rather than a citation.

    For an input identical in every channel — a grayscale image replicated to RGB — summing the
    kernel over channels and convolving once is the *same operator* as the original 3-channel
    convolution. That is why §8 says sum and not mean.
    """
    kernel = torch.randn(8, 3, 4, 4, generator=_generator(2))
    grey = torch.randn(2, 1, 16, 16, generator=_generator(3))

    rgb_out = F.conv2d(grey.repeat(1, 3, 1, 1), kernel, stride=4)
    summed_out = F.conv2d(grey, collapse_input_channels(kernel, adapt="sum"), stride=4)
    # Exact in fp64 — it is an algebraic identity, not an approximation. In fp32 the two
    # reductions add 48 and 16 terms respectively, so they differ by summation order alone.
    assert torch.allclose(
        F.conv2d(grey.double().repeat(1, 3, 1, 1), kernel.double(), stride=4),
        F.conv2d(grey.double(), collapse_input_channels(kernel.double(), adapt="sum"), stride=4),
        atol=1e-12,
    )
    assert torch.allclose(rgb_out, summed_out, rtol=1e-4, atol=1e-4)


def test_the_mean_kernel_shrinks_the_activation_by_exactly_three():
    kernel = torch.randn(8, 3, 4, 4, generator=_generator(2))
    grey = torch.randn(2, 1, 16, 16, generator=_generator(3))
    rgb_out = F.conv2d(grey.repeat(1, 3, 1, 1), kernel, stride=4)
    mean_out = F.conv2d(grey, collapse_input_channels(kernel, adapt="mean"), stride=4)
    assert torch.allclose(rgb_out / 3.0, mean_out, rtol=1e-4, atol=1e-4)
    # And the two collapses are distinguishable — a test that only checked "sum ran" would pass
    # on a loader that quietly averaged.
    assert not torch.allclose(
        collapse_input_channels(kernel, adapt="sum"),
        collapse_input_channels(kernel, adapt="mean"),
    )


def test_the_loaded_patch_embed_is_the_sum_of_the_source_channels(model, checkpoint):
    source = checkpoint["state_dict"][f"{ENCODER_PREFIX}patch_embed.projection.weight"]
    report = load_mmselfsup_mae_encoder(model, checkpoint)
    assert report.source_in_chans == 3
    assert model.patch_embed.proj.weight.shape == (TINY.encoder_dim, 1, 16, 16)
    assert torch.allclose(
        model.patch_embed.proj.weight, source.sum(dim=1, keepdim=True), atol=1e-6
    )
    assert not torch.allclose(
        model.patch_embed.proj.weight, source.mean(dim=1, keepdim=True), atol=1e-6
    )


def test_the_adapt_mode_is_the_config_knob_it_claims_to_be(tmp_path, model, checkpoint):
    """`Init.patch_embed_adapt` reaches the kernel — driven through the config, not by hand.

    R3-style ablations move a knob in a config file and nothing else, so the knob has to arrive
    via :func:`initialize_from_config`. Passing ``adapt=`` directly would test the argument and
    leave the wire between the config and the argument untested.
    """
    source = checkpoint["state_dict"][f"{ENCODER_PREFIX}patch_embed.projection.weight"]
    path = tmp_path / "mae.pth"
    torch.save(checkpoint, path)
    cfg = _tiny_config(
        init=Init(mode="imagenet_mae", checkpoint=str(path), patch_embed_adapt="mean")
    ).replace(architecture=TINY.replace(pos_encoding="sincos2d_rowfirst"))

    report = initialize_from_config(model, cfg)
    assert report.patch_embed_adapt == "mean"
    assert torch.allclose(
        model.patch_embed.proj.weight, source.mean(dim=1, keepdim=True), atol=1e-6
    )


def test_the_activation_scale_survives_the_collapse_on_a_matched_input(model, checkpoint):
    """The property summing exists to preserve, measured through the real patch embed.

    A grayscale tile replicated to three channels is the "matched input": the collapsed embed
    must produce the same activations the 3-channel one would, so the pretrained blocks
    downstream receive what they were trained on.
    """
    source = checkpoint["state_dict"][f"{ENCODER_PREFIX}patch_embed.projection.weight"]
    bias = checkpoint["state_dict"][f"{ENCODER_PREFIX}patch_embed.projection.bias"]
    load_mmselfsup_mae_encoder(model, checkpoint)

    tile = torch.randn(2, 1, TINY.img_px, TINY.img_px, generator=_generator(4))
    with torch.no_grad():
        reference = F.conv2d(
            tile.repeat(1, 3, 1, 1), source, bias, stride=TINY.patch_px
        )
        collapsed = model.patch_embed.proj(tile)
    assert torch.allclose(reference, collapsed, rtol=1e-4, atol=1e-4)
    ratio = collapsed.std() / reference.std()
    assert ratio == pytest.approx(1.0, abs=1e-3)  # 1.0, not 1/3


def test_a_source_that_is_not_three_channel_is_refused(model):
    mono = synthetic_checkpoint(in_chans=1)
    with pytest.raises(ValueError, match="channels"):
        load_mmselfsup_mae_encoder(model, mono)


# --- Rejection: files this loader must not half-load --------------------------


def test_a_checkpoint_from_another_architecture_is_refused(model):
    wider = synthetic_checkpoint(TINY.replace(encoder_dim=64, decoder_dim=32))
    with pytest.raises(ValueError, match="not this architecture|different grid"):
        load_mmselfsup_mae_encoder(model, wider)


def test_a_foreign_checkpoint_is_refused_rather_than_partially_loaded(model, checkpoint):
    # A plain timm state dict: our own names, none of mmselfsup's. Under `strict=False` this
    # loads nothing and reports success, which is the exact failure this module exists to stop.
    timm_shaped = {"state_dict": {n: p.clone() for n, p in model.named_parameters()}}
    with pytest.raises(ValueError, match="no prefix this loader knows"):
        load_mmselfsup_mae_encoder(model, timm_shaped)


def test_an_empty_or_shapeless_checkpoint_is_refused(model):
    with pytest.raises(ValueError, match="no non-empty 'state_dict'"):
        load_mmselfsup_mae_encoder(model, {"state_dict": {}})
    with pytest.raises(ValueError, match="not tensors"):
        load_mmselfsup_mae_encoder(model, {"state_dict": {"backbone.cls_token": "nope"}})


def test_a_non_float_tensor_is_refused(model, checkpoint):
    key = f"{ENCODER_PREFIX}layers.0.ln1.weight"
    checkpoint["state_dict"][key] = torch.zeros(TINY.encoder_dim, dtype=torch.int64)
    with pytest.raises(ValueError, match="not a float tensor"):
        load_mmselfsup_mae_encoder(model, checkpoint)


def test_a_checkpoint_carrying_foreign_objects_loads_without_executing_them(
    tmp_path, model, checkpoint
):
    """The real file pickles `mmengine` log objects, and `torch.load`'s own `weights_only=True`
    refuses it for that reason.

    The tempting fix is `weights_only=False`, which disarms the check over the whole of a 485 MB
    file fetched by HTTP. Instead the foreign classes are resolved to an inert placeholder, so the
    tensors load and nothing outside the allowlist is ever constructed. `Unloadable` here stands
    in for `HistoryBuffer`: a class the loading process has no reason to trust.
    """
    checkpoint["message_hub"] = {"log_scalars": Unloadable(marker=1)}
    path = tmp_path / "with-foreign-objects.pth"
    torch.save(checkpoint, path)

    report = load_mmselfsup_mae_encoder(model, path)
    assert len(report.warm_started) == 29
    assert report.source_epochs == 1600  # `meta` is plain dicts and survives intact


def test_a_bare_state_dict_is_accepted(model, checkpoint):
    # #86's pre-flight may well hand over an already-unwrapped dict; taking one is a
    # convenience, and the prefix assertions are what keep it from being a loophole.
    report = load_mmselfsup_mae_encoder(model, checkpoint["state_dict"])
    assert len(report.warm_started) == 29
    assert report.source_epochs is None  # no `meta`, so nothing is claimed about the schedule


# --- The fresh decoder, the config flag, and a training step ------------------


def test_the_decoder_is_freshly_initialized_not_loaded(model, checkpoint):
    reference = MaskedAutoencoder.from_config(
        _tiny_config().replace(architecture=TINY.replace(pos_encoding="sincos2d_rowfirst"))
    )
    report = load_mmselfsup_mae_encoder(model, checkpoint)
    decoder = {n for n, _ in model.named_parameters() if is_decoder_parameter(n)}
    assert set(report.fresh) == decoder
    assert decoder  # a vacuous "all of nothing was fresh" would pass the loop below
    for name, param in model.named_parameters():
        if name in decoder:
            assert torch.equal(param, dict(reference.named_parameters())[name]), name


def test_the_scratch_path_does_nothing_at_all():
    cfg = _tiny_config()
    assert cfg.init.mode == "scratch"
    model = MaskedAutoencoder.from_config(cfg)
    before = {n: p.clone() for n, p in model.named_parameters()}
    assert initialize_from_config(model, cfg) is None
    assert all(torch.equal(before[n], p) for n, p in model.named_parameters())


def test_continual_versus_scratch_is_one_config_flag(tmp_path, checkpoint):
    """R4 and R5 differ by configuration, not by code path (§8, and #102's own wording)."""
    path = tmp_path / "mae.pth"
    torch.save(checkpoint, path)

    scratch_cfg = _tiny_config()
    continual_cfg = scratch_cfg.replace(
        architecture=TINY.replace(pos_encoding="sincos2d_rowfirst"),
        init=Init(mode="imagenet_mae", checkpoint=str(path)),
    )
    assert continual_cfg.config_hash != scratch_cfg.config_hash  # the arm is in the hash

    warm = MaskedAutoencoder.from_config(continual_cfg)
    report = initialize_from_config(warm, continual_cfg)
    assert report is not None and len(report.warm_started) == 29
    assert torch.allclose(
        warm.blocks[0].attn.qkv.weight,
        checkpoint["state_dict"][f"{ENCODER_PREFIX}layers.0.attn.qkv.weight"],
    )
    assert report.source_epochs == 1600  # read back out of the file, not out of its name


def test_a_warm_started_model_trains_and_the_loss_falls(tmp_path, checkpoint):
    """The end-to-end claim: warm-started weights are a *usable* starting point on CPU.

    Deliberately not a claim about how *well* it trains — that is R4's job on a real box. What
    it rules out is a warm start that produces NaNs, or an encoder wired so nothing reaches the
    patch embed.
    """
    path = tmp_path / "mae.pth"
    torch.save(checkpoint, path)
    cfg = _tiny_config(init=Init(mode="imagenet_mae", checkpoint=str(path))).replace(
        architecture=TINY.replace(pos_encoding="sincos2d_rowfirst")
    )
    model = MaskedAutoencoder.from_config(cfg)
    initialize_from_config(model, cfg)
    loss_fn = _loss_for(cfg)

    tiles = torch.randn(4, 1, TINY.img_px, TINY.img_px, generator=_generator(9))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(20):
        out = model(tiles, generator=_generator(1))
        loss = loss_fn(out.pred, model.patchify(tiles), out.mask).loss
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        assert model.patch_embed.proj.weight.grad is not None  # gradients reach the warm start
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_the_report_round_trips_to_plain_json_types(model, checkpoint):
    import json

    report = load_mmselfsup_mae_encoder(model, checkpoint)
    payload = report.to_dict()
    assert json.loads(json.dumps(payload)) == payload
    assert payload["warm_started_tensors"] == 29
    assert "warm start from" in report.lines()[0]


def test_provenance_is_recognized_and_never_assumed(model, checkpoint):
    """A 32-wide toy checkpoint must not be filed under the real one's URL and licence.

    #88's init-provenance field reads :meth:`InitReport.to_dict` (P101 §7: "every released run
    needs to state what its encoder was initialized from and under what licence"). A spec attached
    by default rather than by measurement would let a released artifact claim **Apache-2.0** and
    an md5 about a file that is neither — on the one axis P101 spent an entire ticket on. So an
    unrecognized checkpoint gets *no* claim, which is a true statement rather than a flattering
    one.
    """
    report = load_mmselfsup_mae_encoder(model, checkpoint)
    assert report.spec is None
    assert report.to_dict()["spec"] is None
    assert "unrecognized checkpoint" in report.lines()[0]


def test_requiring_a_specific_checkpoint_refuses_any_other(model, checkpoint):
    # What #86's pre-flight on the box is for: "the file I fetched is the file plan §8 names".
    with pytest.raises(ValueError, match="required to be .* and is not"):
        load_mmselfsup_mae_encoder(model, checkpoint, expect=MMSELFSUP_MAE_VIT_B16_1600E)


def test_requiring_a_checkpoint_refuses_it_before_loading_anything(model, checkpoint):
    before = {n: p.clone() for n, p in model.named_parameters()}
    with pytest.raises(ValueError):
        load_mmselfsup_mae_encoder(model, checkpoint, expect=MMSELFSUP_MAE_VIT_B16_1600E)
    assert all(torch.equal(before[n], p) for n, p in model.named_parameters())


def test_a_matching_spec_is_recorded_in_full(model, checkpoint):
    """The positive half: when the file *is* the artifact, every provenance field is carried."""
    from pretrain.init import CheckpointSpec

    state = checkpoint["state_dict"]
    toy = CheckpointSpec(
        name="toy-mae",
        url="https://example.invalid/toy.pth",
        md5="0" * 32,
        size_bytes=0,
        epochs=1600,
        licence="Apache-2.0",
        licence_source="https://example.invalid/LICENSE",
        encoder_tensors=sum(1 for k in state if k.startswith(ENCODER_PREFIX)),
        decoder_tensors=sum(1 for k in state if k.startswith(DECODER_PREFIX)),
        preprocessor_tensors=2,
        encoder_parameters=sum(
            v.numel() for k, v in state.items() if k.startswith(ENCODER_PREFIX)
        ),
    )
    report = load_mmselfsup_mae_encoder(model, checkpoint, expect=toy)
    assert report.spec is toy
    assert report.to_dict()["spec"]["licence"] == "Apache-2.0"
    assert report.to_dict()["spec"]["url"] == "https://example.invalid/toy.pth"
    assert "toy-mae" in report.lines()[0]


def test_a_spec_that_disagrees_on_the_recorded_epoch_is_refused(model, checkpoint):
    # The four published schedules are structurally identical, so the epoch and the md5 are the
    # only things that tell them apart. Getting the wrong one is a silent provenance error.
    from pretrain.init import CheckpointSpec

    state = checkpoint["state_dict"]
    wrong_schedule = CheckpointSpec(
        name="toy-mae-300e",
        url="https://example.invalid/toy.pth",
        md5="0" * 32,
        size_bytes=0,
        epochs=300,
        licence="Apache-2.0",
        licence_source="https://example.invalid/LICENSE",
        encoder_tensors=sum(1 for k in state if k.startswith(ENCODER_PREFIX)),
        decoder_tensors=sum(1 for k in state if k.startswith(DECODER_PREFIX)),
        preprocessor_tensors=2,
        encoder_parameters=sum(
            v.numel() for k, v in state.items() if k.startswith(ENCODER_PREFIX)
        ),
    )
    with pytest.raises(ValueError, match="epoch: file records 1600"):
        load_mmselfsup_mae_encoder(model, checkpoint, expect=wrong_schedule)


# --- The behavioural check ----------------------------------------------------


def _behaviour_model() -> MaskedAutoencoder:
    """A ViT-S-shaped model at full tile size — the check needs 224px crops to shift within."""
    return MaskedAutoencoder.from_config(RunConfig())


def test_the_behavioural_check_passes_a_normally_initialized_encoder():
    behaviour = encoder_behaviour(_behaviour_model(), generator=_generator(7))
    assert behaviour.passed, behaviour.lines()
    assert behaviour.crop_margin > 0  # two crops of one field, closer than crops of two fields
    assert behaviour.noise_margin > 0  # and fields nearer each other than white noise


def test_the_behavioural_check_fails_an_encoder_that_cannot_tell_tiles_apart():
    """The failure it does catch: a load that leaves the encoder blind to its input.

    Zeroing the patch embed is the extreme of it — every tile becomes the same token sequence,
    so within-field and between-field agreement collapse onto each other *and* fields stop
    sitting away from noise, which is both floors firing for one reason. This is what a
    catastrophically wrong load looks like, and P102 §6.1 is explicit that a *subtle* scramble
    is not what this statistic detects.
    """
    model = _behaviour_model()
    with torch.no_grad():
        model.patch_embed.proj.weight.zero_()
        model.patch_embed.proj.bias.zero_()
    behaviour = encoder_behaviour(model, generator=_generator(7))
    assert not behaviour.passed
    assert behaviour.crop_margin < MIN_CROP_MARGIN
    assert behaviour.noise_margin < MIN_NOISE_MARGIN


def test_the_behavioural_check_fails_an_exploding_encoder():
    model = _behaviour_model()
    with torch.no_grad():
        for block in model.blocks:
            block.mlp.fc2.weight.mul_(60.0)
    behaviour = encoder_behaviour(model, generator=_generator(7))
    assert behaviour.max_block_amplification > MAX_BLOCK_AMPLIFICATION
    assert not behaviour.passed


def test_the_behavioural_check_leaves_the_model_in_training_mode():
    # A measurement that silently switched the model to `eval` would disable whatever the run
    # was configured with, and nothing downstream would report it.
    model = _behaviour_model()
    model.train()
    encoder_behaviour(model, generator=_generator(7))
    assert model.training


def test_the_synthetic_fields_are_smooth_and_standardized():
    fields = pretrain_init.smooth_fields(4, 64, generator=_generator(1))
    assert fields.shape == (4, 1, 64, 64)
    flat = fields.flatten(1)
    assert torch.allclose(flat.mean(dim=1), torch.zeros(4), atol=1e-5)
    assert torch.allclose(flat.std(dim=1), torch.ones(4), atol=1e-3)
    # Smooth means neighbouring pixels differ far less than in white noise, which is the whole
    # reason these stand in for terrain rather than `randn` doing it.
    noise = torch.randn(4, 1, 64, 64, generator=_generator(2))
    assert fields.diff(dim=-1).abs().mean() < 0.2 * noise.diff(dim=-1).abs().mean()


# --- The real checkpoint (opt-in) --------------------------------------------

#: Point this at the 485 MB file to run the end-to-end group. #86's pre-flight on the box is the
#: same three assertions, which is why they live in the suite rather than in a notebook.
REAL_CHECKPOINT = os.environ.get("DEMAE_MAE_CHECKPOINT")
real_only = pytest.mark.skipif(
    not (REAL_CHECKPOINT and Path(REAL_CHECKPOINT).exists()),
    reason="set DEMAE_MAE_CHECKPOINT to the OpenMMLab MAE ViT-B/16 file",
)


@pytest.fixture(scope="module")
def real_vit_b() -> MaskedAutoencoder:
    cfg = RunConfig.vit_base_continual(REAL_CHECKPOINT)
    model = MaskedAutoencoder.from_config(cfg)
    initialize_from_config(model, cfg)
    return model


@real_only
def test_the_real_checkpoint_is_recognized_and_carries_its_full_provenance():
    """The positive half of `test_provenance_is_recognized_and_never_assumed`.

    Recognition has to *succeed* on the real file, or the honest-by-default behaviour would be
    honest and useless: #88's init-provenance field would read `null` for every run.
    """
    cfg = RunConfig.vit_base_continual(REAL_CHECKPOINT)
    report = initialize_from_config(MaskedAutoencoder.from_config(cfg), cfg)
    assert report.spec is MMSELFSUP_MAE_VIT_B16_1600E
    assert report.to_dict()["spec"]["licence"] == "Apache-2.0"
    assert report.to_dict()["spec"]["epochs"] == 1600


@real_only
def test_requiring_the_real_checkpoint_passes_on_the_real_checkpoint():
    # #86's pre-flight: assert the fetched file is the one plan §8 names, rather than observe it.
    cfg = RunConfig.vit_base_continual(REAL_CHECKPOINT)
    report = load_mmselfsup_mae_encoder(
        MaskedAutoencoder.from_config(cfg),
        REAL_CHECKPOINT,
        expect=MMSELFSUP_MAE_VIT_B16_1600E,
    )
    assert report.spec is MMSELFSUP_MAE_VIT_B16_1600E


@real_only
def test_the_real_checkpoint_is_the_file_p102_describes():
    from pretrain.init import checkpoint_md5, read_state_dict

    spec = MMSELFSUP_MAE_VIT_B16_1600E
    assert Path(REAL_CHECKPOINT).stat().st_size == spec.size_bytes
    assert checkpoint_md5(REAL_CHECKPOINT) == spec.md5

    state, meta = read_state_dict(REAL_CHECKPOINT)
    assert meta["epoch"] == spec.epochs
    counts = {
        prefix: sum(1 for k in state if k.startswith(prefix))
        for prefix in (ENCODER_PREFIX, DECODER_PREFIX, PREPROCESSOR_PREFIX)
    }
    assert counts == {
        ENCODER_PREFIX: spec.encoder_tensors,
        DECODER_PREFIX: spec.decoder_tensors,
        PREPROCESSOR_PREFIX: spec.preprocessor_tensors,
    }
    assert len(state) == sum(counts.values())  # no fourth group
    encoder_parameters = sum(
        v.numel() for k, v in state.items() if k.startswith(ENCODER_PREFIX)
    )
    assert encoder_parameters == spec.encoder_parameters


@real_only
def test_the_real_checkpoint_loads_with_every_scalar_accounted_for(real_vit_b):
    from pretrain.init import read_state_dict

    state, _ = read_state_dict(REAL_CHECKPOINT)
    cfg = RunConfig.vit_base_continual(REAL_CHECKPOINT)
    model = MaskedAutoencoder.from_config(cfg)
    report = initialize_from_config(model, cfg)

    assert len(report.warm_started) == 149
    assert report.discarded == {DECODER_PREFIX: 104, PREPROCESSOR_PREFIX: 2}
    # Every scalar in the file's encoder is either loaded, folded away by the channel collapse,
    # or verified — an identity, not an approximation.
    folded = state[f"{ENCODER_PREFIX}patch_embed.projection.weight"].numel() * 2 // 3
    verified = state[POS_EMBED_KEY].numel()
    assert (
        report.warm_started_parameters + folded + verified
        == MMSELFSUP_MAE_VIT_B16_1600E.encoder_parameters
    )


@real_only
def test_the_real_checkpoints_positional_embedding_is_mmselfsups_row_first_order():
    """The oracle, on the real file. This is the assertion that would have caught the bug."""
    from pretrain.init import read_state_dict
    from pretrain.model import sincos2d_pos_embed

    state, _ = read_state_dict(REAL_CHECKPOINT)
    theirs = state[POS_EMBED_KEY]
    assert torch.allclose(theirs, sincos2d_pos_embed(768, 14, axis_order="row_col"), atol=1e-5)
    assert (theirs - sincos2d_pos_embed(768, 14, axis_order="col_row")).abs().max() > 1.0
    assert torch.allclose(theirs, mmselfsup_pos_embed(14, 768), atol=1e-5)


@real_only
def test_the_real_checkpoint_is_refused_by_a_model_in_maes_positional_order():
    mae_ordered = MaskedAutoencoder.from_config(RunConfig.vit_base())
    with pytest.raises(ValueError, match="sincos2d_rowfirst"):
        load_mmselfsup_mae_encoder(mae_ordered, REAL_CHECKPOINT)


@real_only
def test_the_real_checkpoint_passes_the_behavioural_check(real_vit_b):
    behaviour = encoder_behaviour(real_vit_b, generator=_generator(7))
    assert behaviour.passed, behaviour.lines()
    # The numbers P102 §6.1 records, held loosely enough that a torch upgrade does not fail the
    # suite but tightly enough that a regression in the load would. The amplification band is
    # the tight one on purpose: it moved by 1.3% across twelve varied measurements, so it is the
    # closest thing here to a fingerprint of these particular weights.
    assert behaviour.crop_margin > 0.3
    assert behaviour.noise_margin > 1.0
    assert 2.0 < behaviour.max_block_amplification < 2.3


@real_only
@pytest.mark.parametrize("scramble", ["blocks_reversed", "norms_swapped"])
def test_the_behavioural_check_catches_the_reachable_scrambles_on_real_weights(
    real_vit_b, scramble
):
    """The two scrambles a wrong `encoder_key_map` would actually produce, on real weights.

    Both are same-shape key permutations — reversed block order, and `ln1`/`ln2` swapped within
    every block — and both push the residual stream past the ceiling. P102 §6.1 tabulates the six
    *other* scrambles measured, which this check does **not** catch; those need no defence here
    because the loader performs no permutation or transposition on any tensor, so no bug in it can
    produce them. The primary defence against a same-shape swap remains
    `test_every_mapped_key_is_the_one_p102_documents`.
    """
    scrambled = copy.deepcopy(real_vit_b)
    with torch.no_grad():
        if scramble == "blocks_reversed":
            states = [copy.deepcopy(block.state_dict()) for block in scrambled.blocks]
            for block, state in zip(scrambled.blocks, reversed(states)):
                block.load_state_dict(state)
        else:
            for block in scrambled.blocks:
                weight, bias = block.norm1.weight.clone(), block.norm1.bias.clone()
                block.norm1.weight.copy_(block.norm2.weight)
                block.norm1.bias.copy_(block.norm2.bias)
                block.norm2.weight.copy_(weight)
                block.norm2.bias.copy_(bias)
    behaviour = encoder_behaviour(scrambled, generator=_generator(7))
    assert not behaviour.passed, behaviour.lines()
    assert behaviour.max_block_amplification > MAX_BLOCK_AMPLIFICATION


@real_only
def test_the_real_warm_started_model_takes_training_steps_and_the_loss_falls(real_vit_b):
    model = copy.deepcopy(real_vit_b)
    cfg = RunConfig.vit_base_continual(REAL_CHECKPOINT)
    loss_fn = _loss_for(cfg)
    tiles = pretrain_init.smooth_fields(2, 224, generator=_generator(3))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    first = last = math.nan
    for step in range(6):
        out = model(tiles, generator=_generator(1))
        loss = loss_fn(out.pred, model.patchify(tiles), out.mask).loss
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        first = loss.item() if step == 0 else first
        last = loss.item()
    assert last < first
