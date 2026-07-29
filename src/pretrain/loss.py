"""The MAE objective: masked-patch MSE, plan §4's two DEM-specific guards (#92), and its
ocean/invalid-pixel weighting (#96).

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

## Ocean and invalid-pixel weighting (#96)

``pixel_weights`` is threaded through the statistics, the squared error and the patch-level
aggregation, and every one of those reduces to the unweighted form at uniform weight — which is
what makes #96 an extension of #92 rather than a second objective. #96 fills it from the loader's
`land_mask` (:func:`pixel_weights_from_land_mask`), so ocean and void pixels contribute zero and
each patch renormalizes by its own valid-pixel count. Without that renormalization a half-ocean
patch would be scored as though it were half as wrong as it actually is.

Three parts, and each has a distinct reason to exist:

1. **Zero weight on invalid pixels, in the statistics too.** The `norm_pix` mean and variance run
   over the valid pixels only. Otherwise the "local relief" a patch is judged on is really the
   land/ocean step, which is the largest gradient in the tile and has nothing to do with terrain.
2. **A masked patch under :attr:`Loss.min_valid_patch_frac <pretrain.config.Loss>` valid leaves
   the loss entirely**, rather than being weighted down — so an all-ocean patch contributes
   exactly zero rather than a NaN. :func:`_active_patches` argues the threshold.
3. **The config knob is honoured, not merely recorded.** :attr:`Loss.ocean_weighted
   <pretrain.config.Loss>` and ``pixel_weights`` have to agree: weighting configured with no
   weights supplied is refused, and so is the reverse. A run that reported a weighted objective
   and trained an unweighted one would look identical in every curve, and #93's loop forgetting
   to plumb ``land_mask`` is the concrete way that happens.

**Precedent, honestly.** The published pattern here is curation, not weighting — MAESSTRO drops
any patch containing a missing value. This corpus already curated hard in Phase 0 (kept windows
are <75% ocean by construction), so what this is is a small correctness patch on the residual,
not a novelty claim.

## The coupling to §6's augmentation, and why there is a check for it

The weights and the target are two views of the same pixels, and they arrive by two different
routes: the target through `patchify(image)`, the weights through `patchify(land_mask)`. §6's
augmentation is what keeps those two in step, by transforming the image and the mask with one
draw used twice (#91). If they ever drift apart, this module scores a rearranged copy of where
the ocean used to be — with 0/1 weights, plausible valid counts, a descending loss and no crash.
:func:`check_mask_alignment` is the defence, and it is written from *this* module's side rather
than left to #91's tests, because this is the module the bug would be silent in. It is off the
hot path by design; the docstring says who calls it.
"""

from __future__ import annotations

from collections.abc import Callable
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
    #: ``masked_patches_only=False``), times §4's valid-fraction gate. Zero means the patch is not
    #: in :attr:`loss` at all — which is the distinction a histogram of :attr:`per_patch` needs,
    #: and now also how much of a batch the ocean took out of the objective.
    weight: Tensor


def patch_statistics(
    target: Tensor, *, weights: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """Per-patch mean and variance over the last axis, in fp32: two ``(N, L, 1)`` tensors.

    The variance is the **unbiased** (``n - 1``) estimator, matching MAE's own
    ``target.var(dim=-1, keepdim=True)`` — torch's default, and easy to "fix" into a population
    variance that rescales every target by 256/255. Small, but a silent deviation from the
    recipe this project claims to reproduce is worth one line of comment to avoid.

    ``weights`` generalizes that to the valid-pixel counts the land mask supplies: with 0/1
    weights the formulas below *are* the mean and unbiased variance over the surviving pixels, and
    ``weights.sum(-1)`` is literally the count the ``n - 1`` comes off. That reading is the
    contract — weights are per-pixel *validity*, not relative importance, so scaling them all
    by a constant is not a no-op — and it is what makes the correction meaningful at all.

    The unweighted branch is a fast path, not a second definition — materializing a ones tensor
    at (1024, 196, 256) would cost 200 MB — and
    `test_uniform_pixel_weights_reproduce_the_unweighted_loss` is what pins the two together.

    Both denominators are clamped, so a patch with one valid pixel or none comes back with a
    finite mean and a zero variance rather than a NaN. That is a degenerate answer to a
    degenerate question; §4's 25%-valid threshold (:func:`_active_patches`) is what keeps such
    patches out of the loss.
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


def pixel_weights_from_land_mask(
    land_mask: Tensor, patchify: Callable[[Tensor], Tensor]
) -> Tensor:
    """The loader's ``land_mask`` as the per-pixel weights :meth:`MaeLoss.forward` wants.

    In: ADR-0003's ``(N, H, W)`` bool mask. Out: ``(N, n_patches, patch_dim)``, 1 on land and 0 on
    the exact-zero ocean/void fill.

    ``patchify`` is passed in — :meth:`MaskedAutoencoder.patchify
    <pretrain.model.MaskedAutoencoder.patchify>`, the *same bound method* that cut the target —
    rather than the layout being recomputed here. That is the whole point of the argument. A
    weight grid that disagrees with its target by a transpose, a roll, or a row-major/column-major
    flip is still 0/1, still sums to the right valid count per patch, and still trains: it just
    scores a rearranged copy of where the ocean was. Two implementations of one permutation is
    the only way to get that wrong, so there is one.

    Channel-wise this leans on §1's elevation-only corpus (``in_chans == 1``, which
    :class:`~pretrain.config.Architecture` refuses to relax): one mask plane is unsqueezed to
    ``(N, 1, H, W)`` and patchify's ``patch_px**2 * in_chans`` layout makes it a weight per
    predicted value. A second input channel would need a mask per channel, and that is a Phase-4
    conversation rather than a broadcast that happens to work.
    """
    if land_mask.ndim != 3:
        raise ValueError(
            f"land_mask must be (N, H, W) as ADR-0003 §4 ships it, got {tuple(land_mask.shape)}"
        )
    return patchify(land_mask[:, None].float())


def check_mask_alignment(
    image: Tensor, land_mask: Tensor, *, fill_value: Tensor | float | None = None
) -> None:
    """Refuse a batch whose ``land_mask`` no longer describes its ``image`` (§4, §6).

    This is the failure mode #96 exists inside of, and it is worth stating precisely because
    nothing else in the system can see it. §6's augmentation transforms the image and the land
    mask by one draw of the dihedral group, used twice. If those two ever come apart — a second
    draw, a wrapper applied to one and not the other, a collate that reorders one — then the
    weights built by :func:`pixel_weights_from_land_mask` are still 0/1, still sum to a plausible
    valid count per patch, still pass every shape and dtype guard on :meth:`MaeLoss.forward`, and
    still produce a loss that descends. The objective is simply the wrong one: it scores a
    rotated copy of where the ocean used to be. There is no curve, no warning and no crash.

    **What makes it checkable.** Ocean is the *exact-zero* fill before normalization (ADR-0003 §4)
    and both ADR-0003 normalizers are affine, so every invalid pixel in a tile leaves
    normalization carrying one and the same value — ``normalizer(0)``. Two things then have to be
    true of an aligned pair, and the second is not implied by the first:

    - every pixel the mask calls invalid carries that one value, and
    - no pixel the mask calls valid carries it.

    Exact float comparison is correct here, not sloppy: every invalid pixel goes through the same
    single affine op on the same input, so the results are bit-identical, and land is at least one
    whole metre away from the fill before scaling — far more than a float32 ulp at these
    magnitudes, since the ETL's degenerate screen already puts a floor under the tile's std.

    **``fill_value``, and the one drift the default cannot see.** Left ``None``, that value is
    recovered from the image *through the mask* — the mask says which pixels are ocean, and the
    check asks whether they agree. That is self-contained and works under `per_tile` and `global`
    alike, and it catches every drift that moves the mask around: a rotation, a flip, a roll, a
    reordered batch. What it cannot catch is a mask with **no invalid pixels at all**, because
    then there is no set to read a fill value out of and both conditions above are vacuously
    true. A genuinely all-land tile and a mask that lost its ocean entirely look identical from
    here — and :meth:`MaeLoss.forward`'s ``ocean_weighted`` check does not help, since such a mask
    still produces weights, all of them ones.

    Passing ``fill_value`` — ``normalizer(0)``, which under `per_tile` is ``-mean / std`` from the
    sample dict — closes that gap: the invalid set is then known independently of the mask, so a
    mask claiming no ocean over an image that has some is caught by the second condition. Either a
    scalar shared by the batch or one value per sample. It is optional rather than required
    because computing it means knowing which normalizer the run used, and that is the caller's
    knowledge, not this module's.

    **Off the hot path, deliberately.** It reduces tensors to Python booleans, which is a device
    sync every call and a graph break under §7.3's `torch.compile` — exactly what
    :meth:`MaeLoss._check_inputs` refuses to do for this reason. So the training step does not
    call it. Today its callers are the tests; it is written to be the check plan §15's R0
    pre-flight (#81) runs, which has a whole rented hour to spend and is the right place to answer
    "is the corpus wired up correctly" once, loudly, before the GPU dollars start.
    """
    if image.ndim != 4 or image.shape[1] != 1:
        raise ValueError(
            f"image must be (N, 1, H, W) — §1's elevation-only corpus, one mask plane per "
            f"sample — got {tuple(image.shape)}"
        )
    if land_mask.ndim != 3 or land_mask.shape[0] != image.shape[0]:
        raise ValueError(
            f"land_mask must be (N, H, W) matching the image's batch, got "
            f"{tuple(land_mask.shape)} against {tuple(image.shape)}"
        )
    if land_mask.shape[-2:] != image.shape[-2:]:
        raise ValueError(
            f"land_mask is {tuple(land_mask.shape[-2:])} and the image is "
            f"{tuple(image.shape[-2:])}: the two cannot describe the same pixels"
        )

    x = image.reshape(image.shape[0], -1)
    valid = land_mask.reshape(land_mask.shape[0], -1)
    invalid = ~valid

    if fill_value is None:
        # `normalizer(0)`, read back through the mask. `-inf` when the mask claims no ocean, which
        # then matches no pixel — the blind spot the docstring names and `fill_value` closes.
        fill = torch.where(invalid, x, torch.full_like(x, float("-inf"))).amax(-1, keepdim=True)
    else:
        fill = torch.as_tensor(fill_value, dtype=x.dtype, device=x.device).reshape(-1, 1)
    spread = (invalid & (x != fill)).any(-1)  # a pixel called ocean that is not the fill value
    leaked = (valid & (x == fill)).any(-1)  # a pixel called land that is
    drifted = torch.nonzero(spread | leaked).flatten().tolist()
    if drifted:
        raise ValueError(
            f"land_mask does not describe image for sample(s) {drifted[:8]}"
            f"{' ...' if len(drifted) > 8 else ''}: the pixels it calls invalid and the pixels "
            "carrying the value the exact-zero ocean fill normalizes to are not the same set — "
            "either a pixel called ocean is not that value, or a pixel called land is. The pair "
            "has drifted apart: check that §6's dihedral draw is applied to the image and its "
            "land_mask together (#91) and that nothing downstream transforms one without the "
            "other. Left alone, §4's ocean weighting scores a rearranged copy of where the ocean "
            "was, and no loss curve shows it"
        )


def _active_patches(weight_sum: Tensor, *, min_valid_pixels: float) -> Tensor:
    """Which patches are eligible for the average: §4's minimum-valid-pixel gate (#96).

    A patch that fails the gate leaves the average entirely rather than entering it as a zero,
    which would quietly pull the reported loss down in proportion to how much ocean the batch
    happened to contain.

    Two conditions, and the second is not implied by the first. ``>= min_valid_pixels`` is §4's
    threshold — below ~25% valid there is too little terrain left to learn from and the
    renormalization by valid count amplifies whatever noise remains. ``> 0`` is the arithmetic
    floor underneath it, and it is what keeps the 0/0 renormalization off a NaN at
    ``min_valid_patch_frac = 0`` — a legal config value that switches §4's threshold off without
    switching off the requirement that a scored patch contain *something*.

    The comparison is exact where it matters: with 0/1 weights ``weight_sum`` is a whole number
    up to 2^24, and the default threshold is ``0.25 * 256 = 64``, so "exactly 25% valid is kept"
    is a real guarantee rather than a coin flip on the last bit.
    """
    return ((weight_sum > 0) & (weight_sum >= min_valid_pixels)).to(weight_sum.dtype)


class MaeLoss(nn.Module):
    """MSE against the (optionally normalized) target patch, over the masked patches (§4).

    Stateless — no parameters, no buffers — but an :class:`~torch.nn.Module` all the same, so
    it composes with `torch.compile` and with the training step's module tree like any other
    loss. Build it with :meth:`from_config`; the constructor takes loose numbers only so the
    guards stay testable one at a time.

    Targets arrive **already patchified**, in the same ``(N, n_patches, patch_dim)`` layout the
    decoder predicts in — :meth:`MaskedAutoencoder.patchify <pretrain.model.MaskedAutoencoder>`
    is what cuts them, so the two agree by construction and *this class* never learns what a tile
    is. Two module-level functions do work in tile space (:func:`pixel_weights_from_land_mask`
    and :func:`check_mask_alignment`), and they are deliberately not methods here: one is a
    conversion into this layout and the other is a pre-flight check, and neither belongs to
    scoring a batch.
    """

    def __init__(
        self,
        *,
        norm_pix_loss: bool = True,
        var_floor: float = 1e-4,
        masked_patches_only: bool = True,
        ocean_weighted: bool = True,
        min_valid_patch_frac: float = 0.25,
    ) -> None:
        super().__init__()
        if var_floor <= 0:
            raise ValueError(f"var_floor {var_floor} must be positive")
        if not 0.0 <= min_valid_patch_frac <= 1.0:
            raise ValueError(f"min_valid_patch_frac {min_valid_patch_frac} must lie in [0, 1]")
        self.norm_pix_loss = norm_pix_loss
        self.var_floor = var_floor
        self.masked_patches_only = masked_patches_only
        self.ocean_weighted = ocean_weighted
        self.min_valid_patch_frac = min_valid_patch_frac

    @classmethod
    def from_config(cls, cfg: RunConfig) -> MaeLoss:
        """The only way a run builds a loss: §4's knobs, off the config object.

        Five of :class:`~pretrain.config.Loss`'s six fields reach the constructor. The sixth,
        ``norm_pix_stats_dtype``, is not a choice: the config already refuses anything but
        ``"fp32"`` and :meth:`forward` disables autocast unconditionally, so honouring it would
        mean offering a setting that does not exist.
        """
        loss: Loss = cfg.loss
        return cls(
            norm_pix_loss=loss.norm_pix_loss,
            var_floor=loss.norm_pix_var_floor,
            masked_patches_only=loss.masked_patches_only,
            ocean_weighted=loss.ocean_weighted,
            min_valid_patch_frac=loss.min_valid_patch_frac,
        )

    def _check_inputs(
        self, pred: Tensor, target: Tensor, mask: Tensor, weights: Tensor | None
    ) -> None:
        """Refuse anything that cannot mean what the caller thinks it means.

        Two kinds of refusal, and both are here because the thing they catch does not crash.

        **Shapes and dtypes.** Every one of these mismatches would otherwise broadcast into a
        number: a target off by one patch, a mask transposed, weights handed over per patch
        instead of per pixel. All three train, none crashes, and the loss curve looks fine. The
        target's dtype belongs to the same list and is guard 2's other half — :meth:`forward`
        disabling autocast keeps torch from downcasting the statistics, but it cannot restore a
        target that was already rounded to bf16 before it arrived, and such a target measures as
        *perfectly flat*. Only the target is checked: ``pred`` is bf16 under §7.1's autocast by
        design, and weights are 0/1 and exact in any float.

        Never *values*, though. `bool((weights < 0).any())` would be the natural next line and is
        deliberately absent: reading a value off a tensor is a device sync every step and a graph
        break under §7.3's `torch.compile`, while shapes and dtypes are static and trace out. The
        weight contract is documented on :meth:`forward`, the weights a run actually passes come
        from a boolean land mask via :func:`pixel_weights_from_land_mask`, and the value-level
        question that *is* worth asking — do these weights still describe this image — is
        :func:`check_mask_alignment`, off the hot path.

        **Config coherence** (#96). ``ocean_weighted`` and the presence of ``pixel_weights`` have
        to agree, in both directions, because either mismatch means the config on the checkpoint
        and the W&B page describes an objective the run did not train. These read a config flag
        and whether an argument is ``None`` — both static per run — so they too trace out.
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
        if self.ocean_weighted and weights is None:
            raise ValueError(
                "loss.ocean_weighted is set and no pixel_weights arrived: §4's ocean weighting "
                "needs the loader's land_mask, via pixel_weights_from_land_mask(). Scoring "
                "anyway would train an unweighted objective under a config that reports a "
                "weighted one, and no loss curve shows the difference"
            )
        if not self.ocean_weighted and weights is not None:
            raise ValueError(
                "pixel_weights arrived but loss.ocean_weighted is unset: honouring them would "
                "file a weighted run under an unweighted arm. Set loss.ocean_weighted to weight, "
                "or pass no weights to ablate it"
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

        ``pixel_weights`` is per-pixel **validity in [0, 1]**, in the same layout as ``pred``: 1
        for a pixel that counts, 0 for one that does not. Its sum over a patch is read as that
        patch's valid-pixel count, by both the renormalization here and the variance correction in
        :func:`patch_statistics`. Left ``None``, every pixel counts — which under §4's
        ``ocean_weighted`` is a contradiction rather than a default, see :meth:`_check_inputs`.

        ``mask`` is MAE's convention and :attr:`MaeOutput.mask <pretrain.model.MaeOutput>`'s:
        1 where the patch was hidden from the encoder. Under ``masked_patches_only`` — §4's
        default, and MAE prices the alternative at ~0.5% accuracy — those are the only patches
        in the scalar, and the gradient reaching every other one is exactly zero.

        The whole body runs with autocast **disabled**. Not for the arithmetic's sake — the
        squared error would survive bf16 — but for §4's guard 2: the target statistics decide
        what "flat" means, and in bf16 a patch with genuine relief can measure as perfectly
        flat. Disabling around the whole computation rather than around the statistics keeps
        that from depending on which ops torch's autocast policy happens to promote today;
        :meth:`_check_inputs` covers the half of that guard a context manager cannot.
        """
        self._check_inputs(pred, target, mask, pixel_weights)

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
            weight = selected * _active_patches(
                weight_sum, min_valid_pixels=self.min_valid_patch_frac * error.shape[-1]
            )
            # `clamp_min(1)` covers the batch in which nothing is eligible at all — every masked
            # patch excluded by the valid-fraction gate — giving 0/1 = 0 and a zero gradient
            # rather than a NaN that would poison the optimizer state for the rest of the run.
            loss = (per_patch * weight).sum() / weight.sum().clamp_min(1.0)

        return MaeLossOutput(loss, per_patch, weight)

    def extra_repr(self) -> str:
        return (
            f"norm_pix_loss={self.norm_pix_loss}, var_floor={self.var_floor:g}, "
            f"masked_patches_only={self.masked_patches_only}, "
            f"ocean_weighted={self.ocean_weighted}, "
            f"min_valid_patch_frac={self.min_valid_patch_frac:g}"
        )
