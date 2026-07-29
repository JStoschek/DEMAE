"""The MAE objective: masked-patch MSE with plan §4's two DEM-specific guards (#92).

Stock MAE's loss is four lines, and three of them are here unchanged. What is not stock is why
this module exists at all: `norm_pix_loss` divides each patch by its own standard deviation,
and on elevation that division is *load-bearing* in a way it never is on RGB.

## What per-patch normalization does to a DEM, for better and for worse

For the better: it is local detrending. Each 480 m x 480 m patch gets judged on its internal
relief rather than on its offset within the tile, which is exactly the "learn terrain shape"
objective §5 sets up (with `per_tile` input normalization on top of it, the model neither sees
nor reconstructs absolute altitude anywhere).

For the worse: a *near-flat* patch — playa, lake surface, the residual ocean pixels that
survive a <75%-ocean window filter — has variance near zero in per-tile-standardized units, and
the reference implementation's ``(var + 1e-6)**0.5`` then multiplies whatever quantization noise
is left in it by up to ~1000x. The corpus's 3 m degenerate screen runs per **tile**, not per
patch, so flat patches absolutely reach this code. §4's two guards, both implemented here and
numbered as :class:`~pretrain.config.Loss` numbers them:

1. **A variance floor of 1e-4, not 1e-6** — :data:`REFERENCE_VAR_EPS` is what MAE uses and is
   kept here so the gap is legible. It bounds the amplification at 100x instead of 1000x, and
   it changes the shape of the degenerate case: below the floor, a *flatter* patch now scores
   *less*, so near-flat patches sink to the bottom of the per-patch distribution rather than
   dominating its top. §4 asks for that distribution to be logged and the floor raised if the
   tail is still flat patches, so the floor is a config knob (:attr:`Loss.norm_pix_var_floor
   <pretrain.config.Loss>`) and this module never spells 1e-4.
2. **Target statistics in fp32 regardless of autocast.** Near 1000 the bf16 ulp is 4, so a
   patch with real relief of +/-0.5 has variance *exactly zero* in bf16: the normalized target
   is all zeros and the model is rewarded for predicting nothing. That is not "the loss is a
   bit coarse", so the guard is two-sided. :meth:`MaeLoss.forward` disables autocast over its
   whole body rather than trusting the op-by-op policy — and it *refuses* a target that
   arrives narrower than fp32, because by then the variance is already gone and no upcast
   brings it back. ``target.to(pred.dtype)`` at a call site is the one-line way to write this
   bug, and the context manager alone would not have caught it.

## Both norm_pix paths are real

R3 (#84) ablates `norm_pix_loss` OFF, and it is a genuinely open question: the small-scale MAE
reproduction found normalization unhelpful below ImageNet scale and nobody has measured it on
elevation. So OFF is a tested path with its own characterization (it is plain MSE, and unlike
ON it is sensitive to a patch's offset), not a branch that merely executes.

## The per-patch values, and who reads them

:class:`MaeLossOutput` carries the whole ``(N, n_patches)`` grid of per-patch losses and the
weight each patch entered the average with — not because the objective needs them, but because
§4's diagnosis ("is the top tail near-flat patches, or is it the model?") is unanswerable from
a scalar. #97's monitoring harness reads both. Nothing is detached: callers log with
``.detach()``, and a loss module that pre-detached its diagnostics would be making that choice
for the one caller that might want to differentiate through them.

## The pixel-weighting seam, which is #96's

``pixel_weights`` is threaded through the statistics, the squared error and the patch-level
aggregation, and every one of those reduces to the unweighted form at uniform weight. #96 fills
it from the loader's `land_mask` so ocean pixels contribute zero; until then it is exercised
only by the tests. Two things follow that are worth stating plainly:

- :attr:`Loss.ocean_weighted <pretrain.config.Loss>` and
  :attr:`Loss.min_valid_patch_frac <pretrain.config.Loss>` are **not read here**. They are
  #96's knobs, and this module honours the config's *other* four fields exactly. A run
  configured today with ``ocean_weighted=True`` gets no weighting, which is the honest state
  of the world between these two tickets rather than a bug in this one.
- The patch-level gate is :func:`_active_patches` — currently "any weight at all", which is
  what keeps the 0/0 renormalization off a NaN. #96 replaces that predicate with
  ``min_valid_patch_frac`` and touches nothing else.
"""

from __future__ import annotations

from typing import Final, NamedTuple

import torch
from torch import Tensor, nn

from .config import Loss, RunConfig

#: The reference implementation's ``(var + 1e-6)**0.5``. Not used as a default anywhere — §4
#: raises it to 1e-4 — but written down, because "the floor is 100x the reference's" is the
#: claim, and a claim about a number nobody can find is not checkable.
REFERENCE_VAR_EPS: Final = 1e-6

#: Denominator guard for the weighted reductions. Small enough to be inert on any real weight
#: sum, and its only job is to keep an all-zero-weight patch at 0/eps = 0 instead of 0/0 = NaN.
#: The patch is dropped from the average by :func:`_active_patches` either way; this stops the
#: NaN from existing in the first place, because a NaN multiplied by a zero weight is still a
#: NaN and would take the whole step's gradient with it.
_DENOM_EPS: Final = 1e-12


class MaeLossOutput(NamedTuple):
    """One loss evaluation. A :class:`~typing.NamedTuple` so it survives `torch.compile`."""

    #: The scalar the optimizer steps on: fp32, whatever precision the forward pass ran in.
    loss: Tensor
    #: ``(N, n_patches)`` per-patch loss, in patch order, for **every** patch — including the
    #: ones the scalar ignored. §4's distribution log, and #97 reads it.
    per_patch: Tensor
    #: ``(N, n_patches)`` what each patch contributed with: the mask (or all ones under
    #: ``masked_patches_only=False``), times the #96 patch gate. Zero means the patch is not in
    #: :attr:`loss` at all, which is the distinction a histogram of :attr:`per_patch` needs.
    weight: Tensor


def patch_statistics(
    target: Tensor, *, weights: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """Per-patch mean and variance over the last axis, in fp32: two ``(N, L, 1)`` tensors.

    The variance is the **unbiased** (``n - 1``) estimator, matching MAE's own
    ``target.var(dim=-1, keepdim=True)`` — torch's default, and easy to "fix" into a population
    variance that rescales every target by 256/255. Small, but a silent deviation from the
    recipe this project claims to reproduce is worth one line of comment to avoid.

    ``weights`` generalizes that to the valid-pixel counts #96 supplies: with 0/1 weights the
    formulas below *are* the mean and unbiased variance over the surviving pixels, and
    ``weights.sum(-1)`` is literally the count the ``n - 1`` comes off. That reading is the
    contract — weights are per-pixel *validity*, not relative importance, so scaling them all
    by a constant is not a no-op — and it is what makes the correction meaningful at all.

    The unweighted branch is a fast path, not a second definition — materializing a ones tensor
    at (1024, 196, 256) would cost 200 MB — and
    `test_uniform_pixel_weights_reproduce_the_unweighted_loss` is what pins the two together.

    Both denominators are clamped, so a patch with one valid pixel or none comes back with a
    finite mean and a zero variance rather than a NaN. That is a degenerate answer to a
    degenerate question; #96's 25%-valid threshold is what keeps such patches out of the loss.
    """
    t = target.float()
    if weights is None:
        return t.mean(-1, keepdim=True), t.var(-1, keepdim=True)

    w = weights.float()
    total = w.sum(-1, keepdim=True)
    mean = (w * t).sum(-1, keepdim=True) / total.clamp_min(_DENOM_EPS)
    var = (w * (t - mean) ** 2).sum(-1, keepdim=True) / (total - 1).clamp_min(1.0)
    return mean, var


def normalized_targets(
    target: Tensor, *, var_floor: float, weights: Tensor | None = None
) -> Tensor:
    """``(target - mean) / sqrt(max(var, var_floor))`` per patch, in fp32 — §4's `norm_pix`.

    The floor is a :func:`~torch.clamp_min`, not the reference's ``var + eps``: at any real
    relief the two are indistinguishable, and in the regime that matters — ``var`` below the
    floor — a true floor gives the exact bound the guard is chosen for. A zero prediction on
    such a patch scores ``var / var_floor``, so the loss *falls* as the patch gets flatter
    instead of saturating near 1.
    """
    mean, var = patch_statistics(target, weights=weights)
    return (target.float() - mean) * torch.rsqrt(var.clamp_min(var_floor))


def _active_patches(weight_sum: Tensor) -> Tensor:
    """Which patches are eligible for the average — **the predicate #96 replaces**.

    A patch with no weight at all has no target statistics and no error worth the name; it
    leaves the average entirely rather than entering it as a zero, which would quietly pull the
    reported loss down in proportion to how much ocean the batch happened to contain.

    #96 turns this into ``weight_sum >= min_valid_patch_frac * patch_dim`` and changes nothing
    else in this module.
    """
    return (weight_sum > 0).to(weight_sum.dtype)


class MaeLoss(nn.Module):
    """MSE against the (optionally normalized) target patch, over the masked patches (§4).

    Stateless — no parameters, no buffers — but an :class:`~torch.nn.Module` all the same, so
    it composes with `torch.compile` and with the training step's module tree like any other
    loss. Build it with :meth:`from_config`; the constructor takes loose numbers only so the
    guards stay testable one at a time.

    Targets arrive **already patchified**, in the same ``(N, n_patches, patch_dim)`` layout the
    decoder predicts in — :meth:`MaskedAutoencoder.patchify <pretrain.model.MaskedAutoencoder>`
    is what cuts them, so the two agree by construction and this module never learns what a
    tile is.
    """

    def __init__(
        self,
        *,
        norm_pix_loss: bool = True,
        var_floor: float = 1e-4,
        masked_patches_only: bool = True,
    ) -> None:
        super().__init__()
        if var_floor <= 0:
            raise ValueError(f"var_floor {var_floor} must be positive")
        self.norm_pix_loss = norm_pix_loss
        self.var_floor = var_floor
        self.masked_patches_only = masked_patches_only

    @classmethod
    def from_config(cls, cfg: RunConfig) -> MaeLoss:
        """The only way a run builds a loss: §4's knobs, off the config object.

        Three of :class:`~pretrain.config.Loss`'s six fields reach the constructor. Of the rest,
        ``norm_pix_stats_dtype`` is not a choice — the config already refuses anything but
        ``"fp32"``, and :meth:`forward` disables autocast unconditionally — while
        ``ocean_weighted`` and ``min_valid_patch_frac`` belong to #96 and are read nowhere in
        this module yet. The module docstring says what that costs today.
        """
        loss: Loss = cfg.loss
        return cls(
            norm_pix_loss=loss.norm_pix_loss,
            var_floor=loss.norm_pix_var_floor,
            masked_patches_only=loss.masked_patches_only,
        )

    def _check_shapes(
        self, pred: Tensor, target: Tensor, mask: Tensor, weights: Tensor | None
    ) -> None:
        """Refuse anything that cannot mean what the caller thinks it means.

        Every one of these mismatches would otherwise broadcast into a number: a target off by
        one patch, a mask transposed, weights handed over per patch instead of per pixel. All
        three train, none crashes, and the loss curve looks fine.

        The target's **dtype** is checked here for the same reason and it is guard 2's other
        half: :meth:`forward` disabling autocast keeps torch from downcasting the statistics,
        but it cannot restore a target that was already rounded to bf16 before it arrived, and
        that target measures as *perfectly flat*. Only the target is checked — ``pred`` is bf16
        under §7.1's autocast by design, and weights are 0/1 and exact in any float.

        Shapes and dtypes only, never values. `bool((weights < 0).any())` would be the natural
        next line and is deliberately absent: reading a value off a tensor is a device sync
        every step and a graph break under §7.3's `torch.compile`. Shapes and dtypes are static
        and trace out. The weight contract is documented on :meth:`forward`, and #96 derives
        its weights from a boolean land mask.
        """
        if pred.ndim != 3 or pred.shape != target.shape:
            raise ValueError(
                "prediction and target must both be (N, n_patches, patch_dim) and match; "
                f"got {tuple(pred.shape)} and {tuple(target.shape)}"
            )
        if mask.shape != pred.shape[:2]:
            raise ValueError(
                f"mask must be {tuple(pred.shape[:2])} (one value per patch), "
                f"got {tuple(mask.shape)}"
            )
        if weights is not None and weights.shape != pred.shape:
            raise ValueError(
                f"pixel weights must be {tuple(pred.shape)} (one per predicted value), "
                f"got {tuple(weights.shape)}"
            )
        if target.dtype not in (torch.float32, torch.float64):
            raise ValueError(
                f"§4 computes target statistics in fp32; a {target.dtype} target has already "
                "lost the variance this loss is guarding, and upcasting cannot restore it"
            )

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        mask: Tensor,
        *,
        pixel_weights: Tensor | None = None,
    ) -> MaeLossOutput:
        """Predictions, patchified targets and #90's mask in; §4's objective out.

        ``pixel_weights`` — #96's seam — is per-pixel **validity in [0, 1]**, in the same layout
        as ``pred``: 1 for a pixel that counts, 0 for one that does not. Its sum over a patch is
        read as that patch's valid-pixel count, by both the renormalization here and the
        variance correction in :func:`patch_statistics`. Left ``None``, every pixel counts.

        ``mask`` is MAE's convention and :attr:`MaeOutput.mask <pretrain.model.MaeOutput>`'s:
        1 where the patch was hidden from the encoder. Under ``masked_patches_only`` — §4's
        default, and MAE prices the alternative at ~0.5% accuracy — those are the only patches
        in the scalar, and the gradient reaching every other one is exactly zero.

        The whole body runs with autocast **disabled**. Not for the arithmetic's sake — the
        squared error would survive bf16 — but for §4's guard 2: the target statistics decide
        what "flat" means, and in bf16 a patch with genuine relief can measure as perfectly
        flat. Disabling around the whole computation rather than around the statistics keeps
        that from depending on which ops torch's autocast policy happens to promote today;
        :meth:`_check_shapes` covers the half of that guard a context manager cannot.
        """
        self._check_shapes(pred, target, mask, pixel_weights)

        with torch.autocast(device_type=pred.device.type, enabled=False):
            scored_target = (
                normalized_targets(target, var_floor=self.var_floor, weights=pixel_weights)
                if self.norm_pix_loss
                else target.float()
            )
            error = (pred.float() - scored_target) ** 2

            if pixel_weights is None:
                per_patch = error.mean(-1)
                weight_sum = torch.full_like(per_patch, error.shape[-1])
            else:
                w = pixel_weights.float()
                weight_sum = w.sum(-1)
                per_patch = (error * w).sum(-1) / weight_sum.clamp_min(_DENOM_EPS)

            selected = mask.float() if self.masked_patches_only else torch.ones_like(per_patch)
            weight = selected * _active_patches(weight_sum)
            # `clamp_min(1)` covers the batch in which nothing is eligible at all — every
            # masked patch excluded by the #96 gate — which gives 0/1 = 0 and a zero gradient
            # rather than a NaN that would poison the optimizer state for the rest of the run.
            loss = (per_patch * weight).sum() / weight.sum().clamp_min(1.0)

        return MaeLossOutput(loss, per_patch, weight)

    def extra_repr(self) -> str:
        return (
            f"norm_pix_loss={self.norm_pix_loss}, var_floor={self.var_floor:g}, "
            f"masked_patches_only={self.masked_patches_only}"
        )
