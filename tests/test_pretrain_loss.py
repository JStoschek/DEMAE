"""The MAE objective: masked-patch MSE, the norm_pix guards, and the weighting seam (#92).

Four groups, and the risk is very unevenly spread between them.

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

*The pixel-weighting seam* is #96's, exercised here with uniform weights so that #96 is an
extension and not a rewrite. The single property that has to hold now is that the weighted
formulas and the unweighted fast paths agree — two implementations of one number is exactly
how #96 would end up silently scoring a different objective than the one tested here.
"""

from __future__ import annotations

import math

import pytest
import torch

from pretrain.config import RunConfig
from pretrain.loss import (
    REFERENCE_VAR_EPS,
    MaeLoss,
    normalized_targets,
    patch_statistics,
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
    """A loss built the only way a run builds one: off a :class:`RunConfig` with a knob moved."""
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
    r3 = cfg.replace(loss=cfg.loss.replace(norm_pix_loss=False))
    assert MaeLoss.from_config(r3).norm_pix_loss is False
    assert not torch.allclose(
        MaeLoss.from_config(r3)(pred, target, mask).loss,
        MaeLoss.from_config(cfg)(pred, target, mask).loss,
    )


# --- The #96 pixel-weighting seam ---------------------------------------------


def test_uniform_pixel_weights_reproduce_the_unweighted_loss(pred, target, mask):
    # The seam's only load-bearing property while #96 is unwritten: the weighted formulas and
    # the unweighted fast paths are one objective, not two. If they drift, #96 lands a loss
    # that has never been tested against anything.
    ones = torch.ones_like(pred)
    for loss_fn in (_loss(), _loss(norm_pix_loss=False)):
        plain = loss_fn(pred, target, mask)
        weighted = loss_fn(pred, target, mask, pixel_weights=ones)
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


def test_zero_weighted_pixels_leave_the_statistics_and_the_error_alone(mask):
    # What #96 actually needs: an ocean pixel must not reach the mean, the variance, or the
    # squared error. Half of each patch is corrupted and then zero-weighted; the answer must
    # equal the same patch built from the surviving half alone.
    good = torch.randn(BATCH, PATCHES, PATCH_DIM // 2, generator=_generator(13))
    guess = torch.randn(BATCH, PATCHES, PATCH_DIM // 2, generator=_generator(14))

    padded_target = torch.cat([good, torch.full_like(good, 1e4)], dim=-1)
    padded_pred = torch.cat([guess, torch.zeros_like(guess)], dim=-1)
    weights = torch.cat([torch.ones_like(good), torch.zeros_like(good)], dim=-1)

    loss_fn = _loss()
    assert torch.allclose(
        loss_fn(padded_pred, padded_target, mask, pixel_weights=weights).loss,
        loss_fn(guess, good, mask).loss,
        atol=1e-5,
    )


def test_a_fully_zero_weighted_patch_contributes_nothing_and_stays_finite(pred, target, mask):
    # The edge #96 hits on an all-ocean patch. It must leave the average — not enter it as a
    # zero, which would quietly drag the reported loss down — and it must not produce a NaN
    # from the 0/0 renormalization.
    weights = torch.ones_like(pred)
    weights[0, 0] = 0.0
    out = _loss()(pred, target, mask, pixel_weights=weights)

    assert out.weight[0, 0] == 0.0
    assert torch.isfinite(out.loss)
    assert torch.allclose(out.loss, out.per_patch[out.weight == 1].mean(), atol=1e-6)


def test_an_all_zero_weight_batch_gives_a_finite_zero_loss(pred, target, mask):
    leaf = pred.clone().requires_grad_(True)
    out = _loss()(leaf, target, mask, pixel_weights=torch.zeros_like(pred))
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


# --- Gradients, across every path ---------------------------------------------


@pytest.mark.parametrize("norm_pix_loss", [True, False])
@pytest.mark.parametrize("masked_patches_only", [True, False])
@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("flat", [True, False])
def test_the_gradient_flows_and_is_finite_on_every_path(
    norm_pix_loss, masked_patches_only, weighted, flat, target, mask
):
    tgt = _flat_patches(1e-6) if flat else target
    leaf = torch.randn_like(tgt, requires_grad=True)
    loss_fn = _loss(norm_pix_loss=norm_pix_loss, masked_patches_only=masked_patches_only)
    out = loss_fn(
        leaf, tgt, mask, pixel_weights=torch.ones_like(tgt) if weighted else None
    )
    out.loss.backward()
    assert torch.isfinite(out.loss) and out.loss.item() > 0
    assert torch.isfinite(leaf.grad).all() and (leaf.grad != 0).any()


# --- Against the real model ---------------------------------------------------


def test_the_loss_closes_the_loop_over_a_real_forward_pass():
    # Shapes, layout and the mask convention all agree with #90 or they do not; the three
    # tensors meeting for the first time is the only place that is checkable.
    torch.manual_seed(0)
    model = MaskedAutoencoder.from_config(RunConfig())
    imgs = torch.randn(2, 1, 224, 224, generator=_generator(21))

    out = model(imgs)
    result = MaeLoss.from_config(RunConfig())(out.pred, model.patchify(imgs), out.mask)
    result.loss.backward()

    assert result.per_patch.shape == (2, 196)
    assert result.weight.sum().item() == 2 * 147  # §3's 147 hidden patches per sample
    assert torch.isfinite(result.loss)
    assert torch.isfinite(model.patch_embed.proj.weight.grad).all()
