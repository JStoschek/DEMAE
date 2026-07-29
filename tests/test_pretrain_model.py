"""The MAE forward pass: masking, shapes, the unshuffle, and the parameter counts (#90).

The tests fall into three groups, and the distinction is worth keeping in mind while reading.

*Masking* is where the real risk lives. Every property the rest of Phase 1 leans on is
invisible in a loss curve — a mask that repeats across steps still trains, still descends,
and quietly teaches the model 49 fixed positions; an unshuffle that scrambles the order still
produces a number. So the mask is checked for its count, its variation across two axes, and
its inverse, rather than for "the model ran".

*Shapes* pin the `torch.compile(dynamic=False)` precondition (#95): a mask **count** fixed by
construction, so nothing about the tensors moving through the model depends on a draw.

*Parameter counts* are the transcription check — that this is a ViT-S/16 and a ViT-B/16 and
not something adjacent to them, and that the patch embed reads one channel (plan §1).
"""

from __future__ import annotations

import pytest
import torch

from pretrain.config import Architecture, RunConfig
from pretrain.model import (
    MaskedAutoencoder,
    is_decoder_parameter,
    random_masking,
    sincos2d_pos_embed,
)

#: A batch small enough to run in a second on CPU and large enough that "the masks differ
#: between samples" is a real claim rather than a coin flip.
BATCH = 4


def _generator(seed: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


@pytest.fixture(scope="module")
def vit_s() -> MaskedAutoencoder:
    """The R1 baseline model, built once: init over 24M parameters is not free."""
    return MaskedAutoencoder.from_config(RunConfig())


@pytest.fixture(scope="module")
def batch() -> torch.Tensor:
    """A batch of tiles shaped as `dataset.loader` hands them over: (N, 1, 224, 224)."""
    return torch.randn(BATCH, 1, 224, 224, generator=_generator(7))


# --- The forward pass ---------------------------------------------------------


def test_forward_returns_predictions_the_mask_and_the_restore_indices(vit_s, batch):
    out = vit_s(batch)
    assert out.pred.shape == (BATCH, 196, 256)  # §1: 16 x 16 x 1 per patch
    assert out.mask.shape == (BATCH, 196)
    assert out.ids_restore.shape == (BATCH, 196)
    assert out.ids_restore.dtype == torch.int64


def test_the_mask_is_binary_over_whole_patches(vit_s, batch):
    mask = vit_s(batch).mask
    assert set(mask.unique().tolist()) == {0.0, 1.0}


def test_the_encoder_sees_only_the_49_visible_tokens(vit_s, batch):
    latent, _, _ = vit_s.forward_encoder(batch)
    # 49 visible patches + the CLS token. This is the ~4x pretraining discount §2.1 budgets on.
    assert latent.shape == (BATCH, 50, 384)


def test_the_decoder_predicts_every_patch_not_only_the_masked_ones(vit_s, batch):
    # The loss (#92) selects the masked subset; the decoder itself reconstructs all 196, which
    # is what makes the fixed-mask val grids (§10.2) renderable over a whole tile.
    assert vit_s(batch).pred.shape[1] == 196


# --- The unmasked pass (what a probe and #102's behavioural check read) --------


def test_encode_returns_every_token_with_no_masking(vit_s, batch):
    latent = vit_s.encode(batch)
    assert latent.shape == (BATCH, 197, 384)  # CLS + all 196, not CLS + 49


def test_encode_draws_no_mask_so_it_repeats_exactly(vit_s, batch):
    # `forward_encoder` deliberately varies every call; a frozen-feature reader needs the
    # opposite, and a probe whose features moved between two calls would be measuring the RNG.
    with torch.no_grad():
        assert torch.equal(vit_s.encode(batch), vit_s.encode(batch))


def test_encode_and_the_masked_encoder_share_one_embedding_step(vit_s, batch):
    # The risk in adding a second entry point is a second copy of "embed, then add the PE".
    # Both paths route through `embed_tokens`, so zeroing the encoder PE has to move both.
    saved = vit_s.pos_embed.clone()
    try:
        with torch.no_grad():
            before = vit_s.encode(batch)
            masked_before, _, _ = vit_s.forward_encoder(batch, generator=_generator(21))
            vit_s.pos_embed.zero_()
            after = vit_s.encode(batch)
            masked_after, _, _ = vit_s.forward_encoder(batch, generator=_generator(21))
    finally:
        vit_s.pos_embed.copy_(saved)
    assert not torch.allclose(before, after)
    assert not torch.allclose(masked_before, masked_after)


# --- Masking ------------------------------------------------------------------


def test_exactly_75_percent_of_patches_are_masked_on_every_sample(vit_s, batch):
    mask = vit_s(batch).mask
    assert torch.equal(mask.sum(dim=1), torch.full((BATCH,), 147.0))  # round(0.75 x 196)


def test_the_mask_count_is_identical_across_samples_and_across_steps(vit_s, batch):
    counts = {int(c) for _ in range(3) for c in vit_s(batch).mask.sum(dim=1)}
    # One value, not "close to 147": a count that drifts by one is a recompile under
    # `dynamic=False` (§7.3), and 3 steps x 4 samples is 12 independent draws of it.
    assert counts == {147}


def test_masks_differ_between_samples_in_a_batch(vit_s, batch):
    mask = vit_s(batch).mask
    rows = {tuple(row.tolist()) for row in mask}
    assert len(rows) == BATCH  # per-sample, not one pattern broadcast over the batch


def test_masks_differ_between_steps_at_the_same_input(vit_s, batch):
    first, second = vit_s(batch).mask, vit_s(batch).mask
    assert not torch.equal(first, second)  # §3: the pattern differs every epoch


def test_a_seeded_generator_reproduces_the_mask_exactly(vit_s, batch):
    # §10.1's val loss is only comparable across epochs because the val mask is fixed. That
    # makes a caller-supplied generator part of the model's contract, not a convenience.
    first = vit_s(batch, generator=_generator(1234)).mask
    second = vit_s(batch, generator=_generator(1234)).mask
    assert torch.equal(first, second)
    assert not torch.equal(first, vit_s(batch, generator=_generator(1235)).mask)


def test_the_mask_ratio_is_a_config_knob_not_a_constant(batch):
    # R2 (#83) ablates 90%, and it must move the model without touching this package.
    cfg = RunConfig()
    r2 = cfg.replace(masking=cfg.masking.replace(ratio=0.90))
    model = MaskedAutoencoder.from_config(r2)
    assert (model.n_masked, model.n_visible) == (r2.masked_tokens, r2.visible_tokens)
    assert (model.n_masked, model.n_visible) == (176, 20)
    assert torch.equal(model(batch).mask.sum(dim=1), torch.full((BATCH,), 176.0))


def test_the_model_and_the_config_agree_on_the_masked_count_at_every_ratio():
    # Two derivations of round(ratio x 196) would eventually disagree on a rounding edge; both
    # read `masked_token_count`, and this is what pins that they still do.
    cfg = RunConfig()
    for ratio in (0.25, 0.5, 0.75, 0.85, 0.875, 0.90):
        moved = cfg.replace(masking=cfg.masking.replace(ratio=ratio))
        assert MaskedAutoencoder.from_config(moved).n_masked == moved.masked_tokens


# --- Shuffle and unshuffle ----------------------------------------------------


def test_restore_indices_are_a_permutation_of_every_position(vit_s, batch):
    ids = vit_s(batch).ids_restore
    expected = torch.arange(196).expand(BATCH, -1)
    assert torch.equal(ids.sort(dim=1).values, expected)


def test_restoring_by_index_recovers_the_original_patch_order():
    # Exactly what `forward_decoder` does with the mask tokens, on values whose order is
    # checkable. If this is wrong every visible token lands under the wrong position, the
    # loss still descends, and the reconstructions are subtly, unfixably scrambled.
    n, length, dim, keep = 3, 196, 8, 49
    x = torch.randn(n, length, dim, generator=_generator(3))
    kept, mask, ids_restore = random_masking(x, keep, generator=_generator(4))
    assert kept.shape == (n, keep, dim)

    filler = torch.zeros(n, length - keep, dim)
    restored = torch.gather(
        torch.cat([kept, filler], dim=1), 1, ids_restore.unsqueeze(-1).expand(-1, -1, dim)
    )
    assert torch.equal(restored[mask == 0], x[mask == 0])  # visible tokens, in place
    assert (restored[mask == 1] == 0).all()  # and the hidden ones are the filler


def test_masking_keeps_the_tokens_the_mask_says_it_kept():
    # The mask and the gather are two views of one draw; a test that only checked counts
    # would pass on a mask that names the wrong 49 positions.
    n, length, dim, keep = 2, 196, 4, 49
    x = torch.arange(n * length, dtype=torch.float32).reshape(n, length, 1).expand(-1, -1, dim)
    kept, mask, _ = random_masking(x.contiguous(), keep, generator=_generator(5))
    for i in range(n):
        assert set(kept[i, :, 0].tolist()) == set(x[i, mask[i] == 0][:, 0].tolist())


# --- Static shapes (the #95 precondition) -------------------------------------


def test_output_shapes_are_invariant_across_steps(vit_s, batch):
    shapes = {
        (tuple(o.pred.shape), tuple(o.mask.shape), tuple(o.ids_restore.shape))
        for o in (vit_s(batch) for _ in range(3))
    }
    assert len(shapes) == 1


def test_the_visible_token_count_is_fixed_at_construction(vit_s):
    assert vit_s.n_visible == RunConfig().visible_tokens == 49
    assert vit_s.n_masked + vit_s.n_visible == vit_s.n_patches == 196


# --- Patchify -----------------------------------------------------------------


def test_patchify_unpatchify_round_trips(vit_s, batch):
    assert torch.allclose(vit_s.unpatchify(vit_s.patchify(batch)), batch)


def test_patchify_lays_out_one_patch_per_token_in_row_major_order(vit_s):
    # A tile whose every pixel carries its patch's index: patch k must come out constant at k.
    idx = torch.arange(196).reshape(14, 14).repeat_interleave(16, 0).repeat_interleave(16, 1)
    patches = vit_s.patchify(idx.to(torch.float32)[None, None])
    assert torch.equal(patches[0, :, 0], torch.arange(196, dtype=torch.float32))
    assert (patches[0] == patches[0][:, :1]).all()


def test_the_prediction_width_is_the_single_channel_patch_dim(vit_s):
    assert vit_s.patch_dim == 16 * 16 * 1 == 256  # §1, the second of three single-channel changes


# --- Positional encoding ------------------------------------------------------


def test_the_positional_encoding_separates_every_position():
    pe = sincos2d_pos_embed(384, 14)
    assert pe.shape == (1, 197, 384)
    assert torch.unique(pe[0], dim=0).shape[0] == 197
    assert (pe[0, 0] == 0).all()  # the CLS slot carries no position


def test_the_positional_encoding_matches_maes_own_construction():
    """The halves are column-then-row because MAE's are, and #102 warm-starts from its weights.

    Transcribed from `facebookresearch/mae`'s `util/pos_embed.py` rather than imported: the
    repo is spec, not a dependency (§1), and its `np.meshgrid(grid_w, grid_h)` defaults to
    'xy' indexing — which is the entire subtlety, and exactly what a reimplementation gets
    backwards while every from-scratch test still passes.
    """
    import numpy as np

    def mae_1d(dim, pos):
        omega = 1.0 / 10_000 ** (np.arange(dim // 2, dtype=float) / (dim / 2.0))
        out = np.einsum("m,d->md", pos.reshape(-1), omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    axes = np.meshgrid(np.arange(14, dtype=np.float32), np.arange(14, dtype=np.float32))
    reference = np.concatenate([mae_1d(192, axes[0]), mae_1d(192, axes[1])], axis=1)
    assert np.allclose(sincos2d_pos_embed(384, 14)[0, 1:].numpy(), reference, atol=1e-6)


def test_the_axis_order_is_a_knob_and_the_two_orders_are_one_transpose():
    """#102's finding: mmselfsup's sin-cos PE is MAE's with the two axis halves swapped.

    Recorded here as a property of *our* constructor — that the second order is reachable and
    is exactly the transpose. That mmselfsup's own function produces it is
    `tests/test_pretrain_init.py`'s job, since that is a claim about the checkpoint's source
    ([P102](../docs/research/p102-mmselfsup-mae-layout.md) §4).
    """
    grid = 14
    mae_order = sincos2d_pos_embed(384, grid)
    row_first = sincos2d_pos_embed(384, grid, axis_order="row_col")
    assert not torch.allclose(mae_order, row_first)

    patches = mae_order[:, 1:].reshape(1, grid, grid, 384).transpose(1, 2)
    transposed = torch.cat([mae_order[:, :1], patches.reshape(1, grid * grid, 384)], dim=1)
    assert torch.allclose(row_first, transposed, atol=1e-6)
    # Same functions, same frequencies — only which half carries which axis moves.
    assert torch.allclose(
        mae_order.flatten().sort().values, row_first.flatten().sort().values, atol=1e-6
    )


def test_an_unknown_axis_order_is_refused():
    with pytest.raises(ValueError, match="axis_order"):
        sincos2d_pos_embed(384, 14, axis_order="col_major")


def test_the_axis_order_reaches_both_positional_buffers_from_the_config():
    # R4 warm-starts weights that learned the row-first ordering, so the model has to be able
    # to hold that convention — and hold it in *both* buffers, or the encoder and the decoder
    # would disagree about which axis is which halfway through one forward pass.
    cfg = RunConfig.vit_base()
    row_first = cfg.replace(
        architecture=cfg.architecture.replace(pos_encoding="sincos2d_rowfirst")
    )
    model = MaskedAutoencoder.from_config(row_first)
    assert torch.equal(model.pos_embed, sincos2d_pos_embed(768, 14, axis_order="row_col"))
    assert torch.equal(
        model.decoder_pos_embed, sincos2d_pos_embed(512, 14, axis_order="row_col")
    )
    # And the default is unmoved: R1/R5 keep MAE's own ordering.
    assert torch.equal(
        MaskedAutoencoder.from_config(cfg).pos_embed, sincos2d_pos_embed(768, 14)
    )


def test_the_positional_encoding_is_fixed_not_learned(vit_s):
    # §2.3 — fixed 2D sin-cos. A PE that quietly became learnable would train fine and break
    # the "GSDPE is a Phase-4 drop-in" story, so it is asserted rather than assumed.
    learnable = {n for n, p in vit_s.named_parameters() if "pos_embed" in n}
    assert learnable == set()
    assert torch.equal(vit_s.pos_embed, sincos2d_pos_embed(384, 14))


def test_the_positional_encoding_is_derived_not_checkpointed(vit_s):
    # It is a function of the config, so a checkpoint that carried it could only ever
    # disagree with the config it was loaded under.
    assert not [k for k in vit_s.state_dict() if "pos_embed" in k]


def test_the_decoder_re_adds_positional_embeddings(vit_s, batch):
    # Scale-MAE's finding, cited in §2.2: decoder PE matters. Tested behaviourally — the
    # buffer existing proves nothing about it reaching the decoder blocks.
    saved = vit_s.decoder_pos_embed.clone()
    try:
        with torch.no_grad():
            out = vit_s(batch, generator=_generator(11))
            vit_s.decoder_pos_embed.zero_()
            flat = vit_s(batch, generator=_generator(11))
    finally:
        # The fixture is module-scoped, so a raise between the zero and the restore would
        # leave every later test running against a model with no decoder PE at all.
        vit_s.decoder_pos_embed.copy_(saved)
    assert not torch.allclose(out.pred, flat.pred)


# --- The two backbones --------------------------------------------------------


def test_the_vit_s_encoder_lands_at_about_22m_parameters(vit_s):
    counts = vit_s.parameter_counts()
    assert counts["encoder"] == pytest.approx(22e6, rel=0.05)  # §2.1
    assert counts["encoder"] + counts["decoder"] == counts["total"]
    # §2.2's whole argument: the decoder is a rounding error against the encoder it trains.
    assert counts["decoder"] < 0.2 * counts["encoder"]


def test_the_mask_token_counts_as_the_decoder_it_belongs_to(vit_s):
    # It is the one decoder tensor not named `decoder_*` (MAE's key, kept for interop), so a
    # prefix test would ship it inside §14's "encoder-only" release. Every parameter must land
    # on exactly one side of that boundary.
    names = [n for n, _ in vit_s.named_parameters()]
    assert is_decoder_parameter("mask_token")
    assert not is_decoder_parameter("cls_token")
    encoder_side = [n for n in names if not is_decoder_parameter(n)]
    assert not [n for n in encoder_side if n.startswith("decoder") or n == "mask_token"]
    assert set(encoder_side) >= {"cls_token", "patch_embed.proj.weight"}


def test_the_vit_b_encoder_lands_at_about_86m_parameters():
    counts = MaskedAutoencoder.from_config(RunConfig.vit_base()).parameter_counts()
    assert counts["encoder"] == pytest.approx(86e6, rel=0.05)  # §2.1


@pytest.mark.parametrize("cfg", [RunConfig(), RunConfig.vit_base()], ids=["vit_s", "vit_b"])
def test_both_backbone_configs_build_and_run_from_the_run_config(cfg, batch):
    model = MaskedAutoencoder.from_config(cfg)
    out = model(batch)
    assert out.pred.shape == (BATCH, cfg.architecture.n_patches, cfg.architecture.patch_dim)
    assert torch.isfinite(out.pred).all()
    assert model.decoder_dim == cfg.architecture.decoder_dim


def test_the_model_holds_no_recipe_number_of_its_own(vit_s):
    # The constructor takes `Architecture` itself rather than nine loose ints, so §2's shape
    # has exactly one home and a model whose shape the config would have refused cannot be
    # built at all. `config.py` states the rule; this is what holds the model to it.
    assert vit_s.arch == RunConfig().architecture
    with pytest.raises(ValueError):
        Architecture(encoder_dim=384, encoder_heads=7)  # 7 does not divide 384


def test_the_patch_embed_reads_one_channel(vit_s):
    assert vit_s.patch_embed.proj.in_channels == 1  # §1, the first single-channel change
    assert vit_s.patch_embed.proj.weight.shape == (384, 1, 16, 16)


@pytest.mark.parametrize(
    "shape",
    [(BATCH, 3, 224, 224), (BATCH, 1, 256, 256), (BATCH, 224, 224)],
    ids=["three_channel", "wrong_size", "no_channel_axis"],
)
def test_a_tile_that_is_not_this_model_s_is_refused_by_name(vit_s, shape):
    # The model's own message, not a conv's "expected input[4, 3, 224, 224]" or timm's bare
    # assert: whichever layer noticed first, the useful sentence names the model that refused.
    with pytest.raises(ValueError, match=r"expected tiles of shape \(N, 1, 224, 224\)"):
        vit_s(torch.randn(*shape))


# --- Initialization -----------------------------------------------------------


def test_two_models_from_one_config_start_from_identical_weights():
    # §6 — "two runs from the same config should differ only by nondeterministic kernels".
    a = MaskedAutoencoder.from_config(RunConfig())
    b = MaskedAutoencoder.from_config(RunConfig())
    for (name, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.equal(pa, pb), name


def test_building_a_model_does_not_disturb_the_global_rng():
    # Model init and the DataLoader's shuffle draw from one seeded stream in a real run
    # (§6). If construction consumed from it, adding a decoder block would silently reorder
    # the corpus.
    torch.manual_seed(99)
    before = torch.rand(4)
    torch.manual_seed(99)
    MaskedAutoencoder.from_config(RunConfig())
    assert torch.equal(before, torch.rand(4))


def test_gradients_reach_the_encoder_and_the_decoder(vit_s, batch):
    out = vit_s(batch[:1])
    out.pred.square().mean().backward()
    try:
        assert vit_s.patch_embed.proj.weight.grad is not None
        assert vit_s.decoder_pred.weight.grad is not None
        assert vit_s.mask_token.grad is not None  # the decoder's stand-in for a hidden patch
    finally:
        vit_s.zero_grad(set_to_none=True)
