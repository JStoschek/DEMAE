"""The MAE objective: masked-patch MSE, the norm_pix guards (#92), and ocean weighting (#96).

Five groups, and the risk is very unevenly spread between them.

*Selection* — that unmasked patches contribute nothing — is the cheap half. A loss that scored
every patch would still train (MAE's own ablation says it costs ~0.5%), so these tests are
about the recipe being the recipe, not about a disaster.

*The norm_pix guards* are the expensive half, and they are why this ticket exists separately
from #90. Plan §4's argument is arithmetic: a near-flat 16x16 patch — playa, lake surface,
residual ocean — has variance near zero in per-tile-standardized units, the corpus's degenerate
screen runs per *tile* so those patches survive curation, and the reference implementation's
`(var + 1e-6)**0.5` then amplifies whatever quantization noise is left by up to ~1000x. Nothing
about that is visible in a loss curve except as "the loss is oddly high and oddly spiky", which
reads as a model problem. So the floor is tested by its *magnitude* against the reference eps,
not by "the loss is finite", and the fp32 guard is tested with a target whose statistics
genuinely differ between fp32 and bf16.

*The OFF path* is tested as a live path, not as a branch that executes. R3 (#84) ablates
norm_pix off and has to get a real answer, so the tests pin what the two modes actually differ
on: local offset. ON is blind to a patch's mean; OFF is not.

*The ocean and invalid-pixel weighting* (#96) is the fourth, and its risk profile is the norm_pix
guards' rather than selection's: every one of its failures is silent. Weighted formulas that
disagree with the unweighted fast paths, a valid-fraction gate off by a boundary, a config that
records a weighting the loss did not apply — none of them crash, and all of them train. So the
weighted and unweighted paths are pinned against each other, the 25% gate is tested at exactly 3
and 4 valid pixels of 16 rather than "under" and "over", and the two ``ocean_weighted``
contradictions are tested as refusals.

*The drift check* is the fifth, and it is the one group here that does not test arithmetic. #96's
weights and its targets reach the loss by two different routes and are kept in step by §6's
augmentation (#91); if they come apart, this module scores a rotated copy of where the ocean was,
with plausible weights, a plausible loss, and no crash. These tests assume that drift is possible
rather than trusting #91 to have prevented it — they desynchronize a pair on purpose, for every
non-identity element of the group, and then point the same check at the real #91 wrapper.
"""

from __future__ import annotations

import math

import pytest
import torch

from pretrain.augment import DIHEDRAL_ORDER, DihedralDataset, EpochIndex, apply_dihedral
from pretrain.config import RunConfig
from pretrain.loss import (
    REFERENCE_VAR_EPS,
    MaeLoss,
    check_mask_alignment,
    normalized_targets,
    patch_statistics,
    pixel_weights_from_land_mask,
)
from pretrain.model import MaskedAutoencoder

#: Synthetic shapes. The loss is shape-agnostic — it never reads 196 or 256 from anywhere — so
#: the unit tests run on something small enough to reason about by hand, and one integration
#: test at the bottom runs the real (N, 196, 256) through the actual model.
BATCH, PATCHES, PATCH_DIM = 2, 6, 16


def _generator(seed: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _loss(**changes) -> MaeLoss:
    """A loss built the only way a run builds one: off a :class:`RunConfig` with a knob moved.

    ``ocean_weighted`` is flipped **off** by default here and is on in the release recipe. §4's
    weighting is a separate mechanism (#96) with its own section below, and the guards above it
    are each characterized on their own — so the tests that are not about weighting ask for the
    objective that is not weighted, rather than passing a tensor of ones to say the same thing.
    The default recipe itself is exercised where it belongs: on the real forward pass at the
    bottom of this file, which builds from an untouched :class:`RunConfig`.
    """
    changes.setdefault("ocean_weighted", False)
    cfg = RunConfig()
    return MaeLoss.from_config(cfg.replace(loss=cfg.loss.replace(**changes)))


@pytest.fixture
def pred() -> torch.Tensor:
    return torch.randn(BATCH, PATCHES, PATCH_DIM, generator=_generator(1))


@pytest.fixture
def target() -> torch.Tensor:
    return torch.randn(BATCH, PATCHES, PATCH_DIM, generator=_generator(2))


@pytest.fixture
def mask() -> torch.Tensor:
    """1 = hidden from the encoder, MAE's convention and what `model.forward` returns.

    Deliberately not the same pattern on both samples: a loss that collapsed the batch
    dimension somewhere would still pass against a mask that was.
    """
    return torch.tensor(
        [[1.0, 1.0, 0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0, 0.0, 1.0]]
    )


# --- Selection: masked patches only -------------------------------------------


def test_unmasked_patches_do_not_move_the_loss(pred, target, mask):
    loss_fn = _loss()
    before = loss_fn(pred, target, mask).loss
    moved = pred.clone()
    moved[mask == 0] += 100.0  # a catastrophic error, everywhere the encoder could see
    assert torch.equal(loss_fn(moved, target, mask).loss, before)


def test_masked_patches_do_move_the_loss(pred, target, mask):
    # The other half of the claim above: a test that only checked "unmasked changes nothing"
    # passes on a loss that is constant.
    loss_fn = _loss()
    moved = pred.clone()
    moved[mask == 1] += 1.0
    assert loss_fn(moved, target, mask).loss > loss_fn(pred, target, mask).loss


def test_the_gradient_is_exactly_zero_on_unmasked_patches(pred, target, mask):
    leaf = pred.clone().requires_grad_(True)
    _loss()(leaf, target, mask).loss.backward()
    assert (leaf.grad[mask == 0] == 0).all()
    assert (leaf.grad[mask == 1] != 0).any()


def test_scoring_every_patch_is_a_config_knob(pred, target, mask):
    # `masked_patches_only` is on the #89 config, so it has to reach something. MAE's ablation
    # prices the OFF setting at ~0.5% accuracy; what matters here is that it is not inert.
    every = _loss(masked_patches_only=False)(pred, target, mask)
    assert (every.weight == 1).all()
    leaf = pred.clone().requires_grad_(True)
    _loss(masked_patches_only=False)(leaf, target, mask).loss.backward()
    assert (leaf.grad != 0).all()


def test_the_scalar_is_the_mean_of_the_per_patch_values_over_the_masked_patches(
    pred, target, mask
):
    out = _loss()(pred, target, mask)
    assert torch.allclose(out.loss, out.per_patch[mask == 1].mean())


# --- The per-patch values §4 asks to be logged --------------------------------


def test_per_patch_losses_come_back_for_every_patch_not_only_the_masked_ones(
    pred, target, mask
):
    # §4: "log the per-patch loss distribution; if the top tail is dominated by near-flat
    # patches, raise the floor before concluding anything about the model." The monitoring
    # harness (#97) reads this, and it needs the values keyed by patch position.
    out = _loss()(pred, target, mask)
    assert out.per_patch.shape == (BATCH, PATCHES)
    assert out.per_patch.dtype == torch.float32
    assert (out.per_patch > 0).all()


def test_the_returned_weight_says_which_patches_the_scalar_counted(pred, target, mask):
    out = _loss()(pred, target, mask)
    assert torch.equal(out.weight, mask)


# --- fp32 target statistics (§4 guard 2, as `config.Loss` numbers them) -------


def test_target_statistics_are_fp32_even_under_bf16_autocast(mask):
    # A patch of 1000 +/- 0.5. In bf16 the ulp near 1000 is 4, so every value rounds to the
    # same number: computed in the autocast dtype this patch has variance 0, the normalized
    # target is all zeros, and a prediction of zero scores a perfect 0. In fp32 the variance
    # is 0.25 and the normalized target is +/-1, so the same prediction scores 1.
    #
    # This is half of §4's guard 2 in one number — the half a disabled-autocast region buys.
    # `test_a_target_that_arrives_in_bf16_is_refused` is the other half.
    flat = torch.full((BATCH, PATCHES, PATCH_DIM), 1000.0)
    flat[..., ::2] += 0.5
    flat[..., 1::2] -= 0.5

    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = _loss()(torch.zeros_like(flat), flat, mask)
    # (D-1)/D rather than 1: the squared error averages the population variance while the
    # normalization divides by MAE's unbiased one. 0.94 here at D=16, 0.996 at the real 256 —
    # the point is that it is order 1 and not order 0.
    assert out.loss.item() == pytest.approx((PATCH_DIM - 1) / PATCH_DIM, rel=1e-3)


def test_the_loss_stays_fp32_when_the_prediction_arrives_in_bf16(target, mask):
    # Under §7.1's autocast the decoder hands over bf16 predictions. The loss is fp32
    # regardless — it is the number the LR schedule and the §10.1 curves are read off.
    leaf = torch.randn_like(target, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = _loss()(leaf.to(torch.bfloat16), target, mask)
    out.loss.backward()

    assert out.loss.dtype == torch.float32
    assert out.per_patch.dtype == torch.float32
    # The cast up to fp32 is differentiable, so the disabled-autocast region is a guard on the
    # arithmetic and not a wall the gradient stops at.
    assert torch.isfinite(leaf.grad).all() and (leaf.grad != 0).any()


def test_a_target_that_arrives_in_bf16_is_refused(mask):
    # Disabling autocast stops torch from downcasting the statistics; it does nothing about a
    # target that was already rounded before it got here, and `.float()` cannot put the
    # variance back. The same 1000 +/- 0.5 patch scores a perfect 0.0 that way — the model
    # rewarded for predicting nothing, with no NaN and no crash to notice it by. It holds today
    # only because `patchify` is reshape-and-permute, which is exactly the kind of invariant
    # that stops holding when someone adds a cast to save memory.
    flat = torch.full((BATCH, PATCHES, PATCH_DIM), 1000.0)
    flat[..., ::2] += 0.5
    flat[..., 1::2] -= 0.5
    with pytest.raises(ValueError, match="fp32"):
        _loss()(torch.zeros_like(flat), flat.to(torch.bfloat16), mask)


def test_patch_statistics_are_fp32_whatever_the_input_dtype():
    # The free function upcasts rather than refusing: it is the mechanism, and #96 and the
    # §10.2 grids may want it on whatever they have. Refusing is the *loss*'s policy, above.
    stats = patch_statistics(torch.randn(2, 3, 8, generator=_generator(9)).to(torch.bfloat16))
    assert all(s.dtype == torch.float32 for s in stats)


# --- The variance floor (§4 guard 1) ------------------------------------------


def _flat_patches(std: float) -> torch.Tensor:
    """A batch of patches with a known, tiny internal relief: alternating +/- ``std``."""
    patch = torch.full((BATCH, PATCHES, PATCH_DIM), 0.0)
    patch[..., ::2] += std
    patch[..., 1::2] -= std
    return patch


def test_a_near_flat_patch_produces_a_bounded_loss(mask):
    # var = 1e-6, which is what a lake surface looks like in per-tile-standardized units.
    # Against a zero prediction the per-patch loss is var / floor once the floor binds, so
    # this patch scores 0.01 — a hundredth of the ~1.0 an unfloored division would give it,
    # and nowhere near the top of §4's per-patch distribution.
    flat = _flat_patches(1e-3)
    assert _loss()(torch.zeros_like(flat), flat, mask).loss.item() == pytest.approx(
        0.01, rel=0.05
    )


def test_the_floor_bounds_an_amplification_the_reference_eps_does_not(mask):
    # var = 1e-8, comfortably under both floors, so the two divide by their eps and nothing
    # else: exactly the 100x §4 buys by raising 1e-6 to 1e-4. Measured against
    # `REFERENCE_VAR_EPS` rather than a literal, so the claim is about the reference
    # implementation and not about a number typed twice.
    flat = _flat_patches(1e-4)
    zero = torch.zeros_like(flat)
    guarded = _loss()(zero, flat, mask).loss.item()
    reference = _loss(norm_pix_var_floor=REFERENCE_VAR_EPS)(zero, flat, mask).loss.item()
    assert reference / guarded == pytest.approx(100.0, rel=1e-3)
    assert guarded == pytest.approx(1e-4, rel=0.05)


def test_the_loss_on_a_flatter_patch_falls_rather_than_explodes(mask):
    # The property the floor buys, stated without reference to any particular scale: below the
    # floor, flatter means *less* loss. Without it, every one of these lands near 1.0 and a
    # dead-flat patch divides by zero.
    losses = [
        _loss()(torch.zeros_like(f), f, mask).loss.item()
        for f in (_flat_patches(s) for s in (1e-3, 1e-4, 1e-5, 0.0))
    ]
    assert losses == sorted(losses, reverse=True)
    assert losses[-1] == 0.0
    assert all(math.isfinite(v) and v < 1.0 for v in losses)  # under the reference's ~1.0


def test_the_variance_floor_is_a_config_knob_not_a_constant(mask):
    # §4 says to raise the floor if the top of the per-patch distribution turns out to be
    # near-flat patches, so moving it must not mean editing this package.
    flat = _flat_patches(1e-3)
    raised = _loss(norm_pix_var_floor=1e-2)(torch.zeros_like(flat), flat, mask).loss
    assert MaeLoss.from_config(RunConfig()).var_floor == 1e-4
    assert raised.item() < _loss()(torch.zeros_like(flat), flat, mask).loss.item()


def test_a_dead_flat_patch_keeps_the_gradient_finite(mask):
    flat = _flat_patches(0.0)
    leaf = torch.zeros_like(flat).requires_grad_(True)
    _loss()(leaf, flat, mask).loss.backward()
    assert torch.isfinite(leaf.grad).all()


# --- The normalization itself -------------------------------------------------


def test_normalized_targets_are_zero_mean_and_unit_variance_per_patch():
    # Far above the floor, so the guard is inert and this is plain standardization.
    raw = torch.randn(BATCH, PATCHES, PATCH_DIM, generator=_generator(11)) * 5 + 3
    normed = normalized_targets(raw, var_floor=1e-4)
    assert torch.allclose(normed.mean(-1), torch.zeros(BATCH, PATCHES), atol=1e-5)
    assert torch.allclose(normed.var(-1), torch.ones(BATCH, PATCHES), atol=1e-4)


def test_the_variance_matches_maes_own_unbiased_estimator():
    # MAE's `target.var(dim=-1, keepdim=True)` is torch's default, which is unbiased (n-1).
    # A population variance here would rescale every target by 256/255 — small, but it is a
    # silent deviation from the recipe this project claims to reproduce.
    raw = torch.randn(BATCH, PATCHES, PATCH_DIM, generator=_generator(12))
    _, var = patch_statistics(raw)
    assert torch.allclose(var, raw.var(dim=-1, keepdim=True))


# --- norm_pix OFF, as a live path ---------------------------------------------


def test_norm_pix_off_is_plain_mse_on_the_masked_patches(pred, target, mask):
    out = _loss(norm_pix_loss=False)(pred, target, mask)
    expected = ((pred - target) ** 2).mean(-1)[mask == 1].mean()
    assert torch.allclose(out.loss, expected)


def test_norm_pix_on_ignores_a_patchs_offset_and_off_does_not(pred, target, mask):
    # The detrending claim, made testable: shifting a whole patch up by a constant changes
    # its offset within the tile and nothing about its internal relief. §4 wants the ON path
    # blind to that — "each 480 m patch is judged on its internal relief" — and R3 (#84)
    # exists to find out whether that blindness helps on elevation.
    shifted = target.clone()
    shifted[:, 0, :] += 7.0

    on, off = _loss(), _loss(norm_pix_loss=False)
    assert torch.allclose(on(pred, shifted, mask).loss, on(pred, target, mask).loss)
    assert not torch.allclose(off(pred, shifted, mask).loss, off(pred, target, mask).loss)


def test_norm_pix_off_is_reachable_from_the_config_object(pred, target, mask):
    cfg = RunConfig()
    cfg = cfg.replace(loss=cfg.loss.replace(ocean_weighted=False))  # one mechanism at a time
    r3 = cfg.replace(loss=cfg.loss.replace(norm_pix_loss=False))
    assert MaeLoss.from_config(r3).norm_pix_loss is False
    assert not torch.allclose(
        MaeLoss.from_config(r3)(pred, target, mask).loss,
        MaeLoss.from_config(cfg)(pred, target, mask).loss,
    )


# --- The pixel-weighting arithmetic (#92's seam, #96's mechanism) -------------


def test_uniform_pixel_weights_reproduce_the_unweighted_loss(pred, target, mask):
    # The weighted formulas and the unweighted fast paths have to be one objective, not two —
    # otherwise half of what the tests above pin is not the objective #96 actually trains. Read
    # as a statement about the corpus: on an all-land batch, flipping §4's ocean weighting on
    # changes nothing, because all-ones is what the land mask of such a batch produces.
    ones = torch.ones_like(pred)
    for norm_pix_loss in (True, False):
        plain = _loss(norm_pix_loss=norm_pix_loss)(pred, target, mask)
        weighted = _loss(norm_pix_loss=norm_pix_loss, ocean_weighted=True)(
            pred, target, mask, pixel_weights=ones
        )
        assert torch.allclose(plain.loss, weighted.loss, atol=1e-6)
        assert torch.allclose(plain.per_patch, weighted.per_patch, atol=1e-6)


def test_a_pixel_weight_is_a_count_and_the_variance_correction_reads_it_as_one():
    # Weights are validity indicators, not relative importances: their sum is the patch's
    # valid-pixel count, and MAE's unbiased variance divides by that count minus one. So a
    # patch weighted 0.5 everywhere is "half a pixel each" and its correction genuinely
    # differs. Pinned rather than left implicit, because #96 will hand this argument a 0/1
    # land mask and the two readings only coincide there.
    raw = torch.randn(BATCH, PATCHES, PATCH_DIM, generator=_generator(15))
    _, plain = patch_statistics(raw)
    _, half = patch_statistics(raw, weights=torch.full_like(raw, 0.5))
    scale = (0.5 * PATCH_DIM - 1) / (PATCH_DIM - 1)
    assert torch.allclose(half * scale, 0.5 * plain, atol=1e-6)


@pytest.mark.parametrize("norm_pix_loss", [True, False])
def test_zero_weighted_pixels_leave_the_statistics_and_the_error_alone(norm_pix_loss, mask):
    # What §4's weighting actually claims: an ocean pixel must not reach the mean, the variance,
    # or the squared error. Half of each patch is corrupted and then zero-weighted; the answer
    # must equal the same patch built from the surviving half alone. The surviving half is 50%
    # of the patch, comfortably over the 25% gate, so this measures the arithmetic and not the
    # exclusion.
    good = torch.randn(BATCH, PATCHES, PATCH_DIM // 2, generator=_generator(13))
    guess = torch.randn(BATCH, PATCHES, PATCH_DIM // 2, generator=_generator(14))

    padded_target = torch.cat([good, torch.full_like(good, 1e4)], dim=-1)
    padded_pred = torch.cat([guess, torch.zeros_like(guess)], dim=-1)
    weights = torch.cat([torch.ones_like(good), torch.zeros_like(good)], dim=-1)

    assert torch.allclose(
        _loss(ocean_weighted=True, norm_pix_loss=norm_pix_loss)(
            padded_pred, padded_target, mask, pixel_weights=weights
        ).loss,
        _loss(norm_pix_loss=norm_pix_loss)(guess, good, mask).loss,
        atol=1e-5,
    )


def test_a_fully_zero_weighted_patch_contributes_nothing_and_stays_finite(pred, target, mask):
    # The edge an all-ocean patch hits. It must leave the average — not enter it as a zero, which
    # would quietly drag the reported loss down in proportion to how much ocean the batch
    # happened to contain — and it must not produce a NaN from the 0/0 renormalization.
    weights = torch.ones_like(pred)
    weights[0, 0] = 0.0
    out = _loss(ocean_weighted=True)(pred, target, mask, pixel_weights=weights)

    assert out.weight[0, 0] == 0.0
    assert torch.isfinite(out.loss)
    assert torch.allclose(out.loss, out.per_patch[out.weight == 1].mean(), atol=1e-6)


def test_an_all_zero_weight_batch_gives_a_finite_zero_loss(pred, target, mask):
    leaf = pred.clone().requires_grad_(True)
    out = _loss(ocean_weighted=True)(leaf, target, mask, pixel_weights=torch.zeros_like(pred))
    out.loss.backward()
    assert out.loss.item() == 0.0
    assert torch.isfinite(leaf.grad).all()


def test_mismatched_shapes_are_refused(pred, target, mask):
    loss_fn = _loss()
    with pytest.raises(ValueError, match="prediction and target"):
        loss_fn(pred, target[:, :-1], mask)
    with pytest.raises(ValueError, match="mask"):
        loss_fn(pred, target, mask[:, :-1])
    # The one worth spelling out: per-patch weights instead of per-pixel ones. That shape
    # broadcasts, so without this check #96 could weight whole patches and see a loss.
    with pytest.raises(ValueError, match="pixel weights"):
        loss_fn(pred, target, mask, pixel_weights=torch.ones(BATCH, PATCHES))


# --- #96: the minimum-valid-fraction patch gate -------------------------------


@pytest.mark.parametrize("norm_pix_loss", [True, False])
def test_a_masked_patch_below_the_valid_fraction_leaves_the_loss(
    norm_pix_loss, pred, target, mask
):
    # §4's threshold and the argument for it: under 25% valid there is not enough terrain left
    # in a patch to learn anything from, and renormalizing by the valid count amplifies
    # whatever noise remains. `PATCH_DIM` is 16 here, so the threshold lands at 4 valid pixels
    # and the boundary is checkable exactly — which is the point of testing 3 and 4 rather than
    # "a bit under" and "a bit over".
    weights = torch.ones_like(pred)
    weights[0, 0, 3:] = 0.0  # 3 of 16 valid: under the threshold, excluded
    weights[0, 1, 4:] = 0.0  # 4 of 16 valid: exactly at it, kept
    out = _loss(ocean_weighted=True, norm_pix_loss=norm_pix_loss)(
        pred, target, mask, pixel_weights=weights
    )

    assert out.weight[0, 0] == 0.0
    assert out.weight[0, 1] == 1.0


def test_the_valid_fraction_threshold_is_a_config_knob_not_a_constant(pred, target, mask):
    # 25% is a number plan §4 chose, not a law — and the same argument that makes the variance
    # floor a knob applies to it: if the per-patch distribution says the threshold is wrong,
    # moving it must not mean editing this package.
    weights = torch.ones_like(pred)
    weights[0, 0, 8:] = 0.0  # 8 of 16 valid: over 25%, under 75%

    kept = _loss(ocean_weighted=True)
    assert kept(pred, target, mask, pixel_weights=weights).weight[0, 0] == 1.0
    raised = _loss(ocean_weighted=True, min_valid_patch_frac=0.75)
    assert raised(pred, target, mask, pixel_weights=weights).weight[0, 0] == 0.0
    assert MaeLoss.from_config(RunConfig()).min_valid_patch_frac == 0.25


def test_a_zero_threshold_still_keeps_an_all_ocean_patch_out_of_the_loss(pred, target, mask):
    # `min_valid_patch_frac = 0` switches §4's threshold off, and it is a legal config value.
    # What it must not switch off is the arithmetic floor underneath: a patch with no valid
    # pixels at all has a 0/0 renormalization, and letting it into the average would trade the
    # threshold for a NaN.
    weights = torch.ones_like(pred)
    weights[0, 0] = 0.0
    out = _loss(ocean_weighted=True, min_valid_patch_frac=0.0)(
        pred, target, mask, pixel_weights=weights
    )

    assert out.weight[0, 0] == 0.0
    assert out.per_patch[0, 0] == 0.0  # "contributes exactly zero", not "contributes a NaN"
    assert torch.isfinite(out.loss)


# --- #96: the weights themselves, off the loader's land_mask -------------------


def test_pixel_weights_come_from_the_land_mask_in_the_targets_own_layout():
    # ADR-0003 ships `land_mask` as (H, W) bool per sample; the loss wants one weight per
    # predicted value, in patchify's row-major patch order. The expected patch indices below
    # are derived from that documented order by hand — 14 patches to a row at 224/16, so the
    # patch one step down the diagonal from patch 0 is patch 15 — rather than read back out of
    # the function under test.
    model = MaskedAutoencoder.from_config(RunConfig())
    land = torch.zeros(1, 224, 224, dtype=torch.bool)
    land[0, :16, :16] = True
    land[0, 16:32, 16:32] = True

    weights = pixel_weights_from_land_mask(land, model.patchify)
    counts = weights.sum(-1)[0]

    assert weights.shape == (1, 196, 256)
    assert counts[0] == 256 and counts[15] == 256
    assert counts.sum() == 2 * 256  # and nothing anywhere else
    assert sorted(weights.unique().tolist()) == [0.0, 1.0]


def test_a_zero_weight_lands_on_exactly_the_ocean_pixel_of_the_target():
    # The property that makes the weighting *correct* rather than merely correctly shaped: after
    # patchify, "weight 0" and "this pixel was ocean" have to be the same set of positions. A
    # weight grid transposed or rolled relative to the target is still 0/1, still sums to the
    # right count, and still trains.
    model = MaskedAutoencoder.from_config(RunConfig())
    raw = torch.randint(1, 100, (1, 224, 224), generator=_generator(31)).float()
    raw[0, 100:150, 30:200] = 0.0  # a bay, deliberately not square and not patch-aligned
    land = raw != 0.0

    weights = pixel_weights_from_land_mask(land, model.patchify)
    target = model.patchify(raw[:, None])

    assert torch.equal(weights == 0.0, target == 0.0)


def test_a_run_configured_for_ocean_weighting_cannot_be_scored_without_weights(
    pred, target, mask
):
    # The dishonest state #92 left behind and named in its own docstring: `ocean_weighted=True`
    # with no weights is a run that reports a weighting it does not have, and nothing in the loss
    # curve says so. Refused for the same reason `augment.make_augmented` refuses
    # `random_resized_crop` — a run must not claim a mechanism it did not use. This is also the
    # only thing standing between #93's training loop and forgetting to plumb `land_mask`.
    with pytest.raises(ValueError, match="ocean_weighted"):
        _loss(ocean_weighted=True)(pred, target, mask)


def test_weights_are_refused_when_the_run_is_configured_without_ocean_weighting(
    pred, target, mask
):
    # The same argument in the other direction: R3-style ablations turn the weighting off, and a
    # run that then weighted anyway would be filed under the wrong arm.
    with pytest.raises(ValueError, match="ocean_weighted"):
        _loss(ocean_weighted=False)(pred, target, mask, pixel_weights=torch.ones_like(pred))


# --- #96: image / land_mask drift, which nothing else in the system notices -----


def _coastline_sample(seed: int = 41) -> tuple[torch.Tensor, torch.Tensor]:
    """One sample shaped the way the loader ships it: ``(1, H, W)`` normalized elevation and its
    ``(H, W)`` land mask.

    Two properties are load-bearing and both are the loader's, not this test's. Ocean is the
    **exact-zero fill** before normalization (ADR-0003 §4), and both ADR-0003 normalizers are
    affine, so every ocean pixel comes out on one and the same value. That is what makes drift
    detectable at all.

    The coastline is a **non-square corner rectangle** on purpose. A diagonal one would be
    invariant under transposition — which is one of the eight dihedral elements — so a mask
    rotated by exactly that element would still align and the test would report a pass it had
    not earned. This region maps to eight distinct regions under the eight elements.

    Not shared with `test_pretrain_augment.py`'s similar-looking builder, and the difference is
    the point rather than an oversight: that one derives its mask from a *threshold* on the image
    (`image < 2`), which is all #91 needs, while the alignment check requires the loader's real
    invariant — ocean is the exact-zero fill, so it normalizes to one identical value.
    """
    raw = torch.randint(20, 3000, (224, 224), generator=_generator(seed)).float()
    raw[:50, :90] = 0.0
    land = raw != 0.0
    mean, std = raw[land].mean(), raw[land].std()
    return ((raw - mean) / std)[None], land


def _ocean_fill(image: torch.Tensor, land: torch.Tensor) -> torch.Tensor:
    """The one value the exact-zero ocean fill normalizes to — ``normalizer(0)``."""
    return image[0][~land][0]


@pytest.mark.parametrize("with_fill", [False, True])
def test_an_aligned_image_and_land_mask_pass_the_alignment_check(with_fill):
    image, land = _coastline_sample()
    fill = _ocean_fill(image, land) if with_fill else None
    check_mask_alignment(image[None], land[None], fill_value=fill)  # no raise


@pytest.mark.parametrize("with_fill", [False, True])
@pytest.mark.parametrize("k", range(1, DIHEDRAL_ORDER))
def test_a_desynchronized_image_and_land_mask_are_caught(k, with_fill):
    # #96's whole failure mode in one line. If augmentation ever transforms the image and its
    # land mask by different elements of the group, this module weights a rotated copy of where
    # the ocean used to be: the weights are still 0/1, the valid counts are still right, the loss
    # still descends, and nothing anywhere else in the system says a word. So the drift is
    # checked for rather than assumed away — every non-identity element of the group, since
    # "some rotations are caught" is not the claim, and on both of the check's paths.
    image, land = _coastline_sample()
    fill = _ocean_fill(image, land) if with_fill else None
    with pytest.raises(ValueError, match="land_mask"):
        check_mask_alignment(image[None], apply_dihedral(land, k)[None], fill_value=fill)


def test_a_mask_that_lost_its_ocean_is_caught_only_when_the_fill_value_is_known():
    # The one drift class the mask-derived fill cannot see, pinned as a known limit rather than
    # left for someone to discover. A mask marking every pixel valid has no invalid set to read a
    # fill value out of, so there is nothing to compare against — and `ocean_weighted` does not
    # catch it either, because such a mask still produces weights, all of them ones. Told the fill
    # value, the check has an opinion again.
    image, land = _coastline_sample()
    all_valid = torch.ones_like(land)

    check_mask_alignment(image[None], all_valid[None])  # tolerated: the documented blind spot
    with pytest.raises(ValueError, match="land_mask"):
        check_mask_alignment(
            image[None], all_valid[None], fill_value=_ocean_fill(image, land)
        )


def test_the_shipped_dihedral_augmentation_keeps_the_pair_aligned():
    # The same check pointed at the real #91 wrapper rather than at a hand-built desync. This is
    # deliberately not "trust #91's own tests": #91 asserts that one draw is used twice, and this
    # asserts the property #96 actually depends on, from #96's side, through the real read path.
    image, land = _coastline_sample()
    sample = {
        "image": image,
        "land_mask": land,
        "mean": torch.tensor(0.0),
        "std": torch.tensor(1.0),
        "tile_id": "t0",
        "stratum_id": 0,
    }
    dataset = DihedralDataset([sample], seed=7)

    # One tile across many *epochs*, not one tile read many times: since #104 the element is a
    # pure function of `(seed, epoch, index)`, so re-reading the same index is by design the
    # same transform every time. Sweeping the epoch is what varies it, and it is also the
    # honest shape of the question — this asserts that the pair stays aligned under whatever
    # element a given epoch picks, which is the property #96 depends on for the whole run.
    seen = set()
    for epoch in range(64):
        drawn = dataset[EpochIndex(0, epoch)]
        check_mask_alignment(drawn["image"][None], drawn["land_mask"][None])
        seen.add(drawn["land_mask"].numpy().tobytes())

    # Without this the test above could pass by never transforming anything.
    assert len(seen) == DIHEDRAL_ORDER


def test_a_desynchronized_mask_really_does_score_a_different_objective():
    # Why the drift is worth a check rather than a comment: a rotated mask is still 0/1 and — on
    # this tile — still sums to the same total valid count, so every shape and dtype guard in
    # this module passes it. The number it computes is simply the wrong one.
    model = MaskedAutoencoder.from_config(RunConfig())
    image, land = _coastline_sample()
    imgs, masks = image[None], land[None]

    torch.manual_seed(0)
    out = model(imgs)
    target = model.patchify(imgs)
    loss_fn = MaeLoss.from_config(RunConfig())

    weights = pixel_weights_from_land_mask(masks, model.patchify)
    aligned = loss_fn(out.pred, target, out.mask, pixel_weights=weights).loss
    drifted = loss_fn(
        out.pred,
        target,
        out.mask,
        pixel_weights=pixel_weights_from_land_mask(apply_dihedral(masks, 2), model.patchify),
    ).loss

    assert not torch.allclose(aligned, drifted)


# --- Gradients, across every path ---------------------------------------------


def _weights_for(case: str, like: torch.Tensor) -> torch.Tensor | None:
    """The four weight regimes every other path has to stay finite across.

    ``half_ocean`` is 8 valid pixels of 16 — over the 25% gate, so it is scored and not excluded.
    ``all_ocean`` is the degenerate one #96 introduces and the edge the acceptance criteria name:
    every patch fails the gate, so nothing is eligible, and both the 0/0 patch renormalization
    and the 0/0 batch average have to come back as zeros rather than NaNs. A NaN there would not
    stay local — it multiplies through the optimizer state and takes the rest of the run with it.
    """
    if case == "none":
        return None
    w = torch.ones_like(like)
    if case == "half_ocean":
        w[..., PATCH_DIM // 2 :] = 0.0
    elif case == "all_ocean":
        w.zero_()
    return w


@pytest.mark.parametrize("norm_pix_loss", [True, False])
@pytest.mark.parametrize("masked_patches_only", [True, False])
@pytest.mark.parametrize("weight_case", ["none", "uniform", "half_ocean", "all_ocean"])
@pytest.mark.parametrize("flat", [True, False])
def test_the_gradient_flows_and_is_finite_on_every_path(
    norm_pix_loss, masked_patches_only, weight_case, flat, target, mask
):
    tgt = _flat_patches(1e-6) if flat else target
    leaf = torch.randn_like(tgt, requires_grad=True)
    weights = _weights_for(weight_case, tgt)
    loss_fn = _loss(
        norm_pix_loss=norm_pix_loss,
        masked_patches_only=masked_patches_only,
        ocean_weighted=weights is not None,
    )

    out = loss_fn(leaf, tgt, mask, pixel_weights=weights)
    out.loss.backward()

    assert torch.isfinite(out.loss) and torch.isfinite(leaf.grad).all()
    if weight_case == "all_ocean":
        assert out.loss.item() == 0.0 and (leaf.grad == 0).all()
    else:
        assert out.loss.item() > 0 and (leaf.grad != 0).any()


# --- Against the real model ---------------------------------------------------


def test_the_loss_closes_the_loop_over_a_real_forward_pass():
    # Shapes, layout and the mask convention all agree with #90 or they do not; the four tensors
    # meeting for the first time is the only place that is checkable. Built from an *untouched*
    # `RunConfig`, so this is the release recipe — ocean weighting included — and not a
    # configuration assembled to suit the test.
    torch.manual_seed(0)
    model = MaskedAutoencoder.from_config(RunConfig())
    imgs = torch.randn(2, 1, 224, 224, generator=_generator(21))
    land = torch.ones(2, 224, 224, dtype=torch.bool)
    land[0, :112] = False  # the top half of the first tile is ocean

    out = model(imgs)
    result = MaeLoss.from_config(RunConfig())(
        out.pred,
        model.patchify(imgs),
        out.mask,
        pixel_weights=pixel_weights_from_land_mask(land, model.patchify),
    )
    result.loss.backward()

    # Pixel rows 0-111 are the first 7 rows of the 14x14 patch grid, so patches 0-97 of sample 0
    # hold no land at all and leave the loss whether they were masked or not. Every other patch
    # is entirely land. §3 hides 147 patches per sample; the ocean takes back the masked ones.
    excluded = out.mask[0, :98].sum().item()
    assert result.per_patch.shape == (2, 196)
    assert result.weight.sum().item() == 2 * 147 - excluded
    assert (result.weight[0, :98] == 0).all()
    assert torch.isfinite(result.loss)
    assert torch.isfinite(model.patch_embed.proj.weight.grad).all()
