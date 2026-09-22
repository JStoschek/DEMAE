"""A3 — §4.2's DEMAE native fine-tune, and the inference composite it is scored through (#166).

Two halves, and #166 files them as one ticket because §4.2 states them as one recipe. The
fine-tune is the arm; the composite is how the arm's output is read off the token grid. Getting
the second one wrong does not make A3 worse in a way a reader could see — it makes A3 *look*
worse in a column that is measuring the inference scheme.

## The arm: continue MAE training, with the mask sampler swapped

§4.2 is explicit that this is *"a short adaptation, not a new pretraining"*, and every piece of
it is Phase 1's: :class:`pretrain.model.MaskedAutoencoder` is the model, :class:`pretrain.loss.
MaeLoss` is the objective, and the checkpoint is the pretraining one a release was exported from
rather than the release itself. What changes is the **mask**, and one flag.

The mask is read off §3.4's committed train mask bank — *the same draws the frozen arms train
against*, which is #166's second criterion and the reason A2→A3 and A3-versus-A4 mean anything
at all. Epoch ``e`` of A3's twenty is the bank's epoch ``e``, so the void geometry A3 adapts to
is not merely drawn from the same distribution as the geometry A4's head sees: it is the same
file, the same rows, the same pixels.

The flag is ``norm_pix_loss``, and §4.2 turns it **off**: *"void filling needs absolute per-patch
scale; norm-pix targets discard it"*. That is the same fact :mod:`benchmark.zeroshot` refuses on
— A2 cannot retrain and so has no way past it — and it is why A3 is not simply A2 with better
weights. Turned off *here*, on the fine-tune's own loss, rather than read off the checkpoint: R5
and R4 were both pretrained with it on, and the whole point of the adaptation is that the arm
stops predicting in units the void hides.

## The mask budget, and why it is pretraining's

§4.2 says the void's patches are dropped *"plus 25% random patches so the encoder keeps seeing
global context and the effective mask ratio stays near pretraining's"*. Two readings of that
sentence produce two different sets of weights, and the one built here is the one that makes the
plan's own three claims true together: **mask the void's patches, then random-mask more until
exactly pretraining's count is hidden.** The encoder then sees
:attr:`~pretrain.model.MaskedAutoencoder.n_visible` tokens — 49 of 196 at R5's ratio, which is
pretraining's own number — so the ratio does not merely stay *near* pretraining's, it is
pretraining's; the survivors are scattered over the whole tile rather than clustered outside the
void, which is what "keeps seeing global context" asks for; and §7's costing, which reads A3's
epoch straight off R0's measured samples/s on the ground that *"an A3 epoch is a pretraining
epoch"*, is true of the arm rather than about it.

It is also the only reading that can be batched, which is worth saying plainly because it would
otherwise look like the reason. :func:`pretrain.model.keep_masking` refuses a ragged keep-set —
one gather has one width — and the alternative reading makes the keep count a function of the
void's area: 89 distinct values over 4,000 of the bank's draws, so a 128-tile batch would be ~80
forward passes of one or two tiles. That is affordable (it is ~45x §7's line, not 45x the
budget) and it is not what decided this. The decision is §4.2's sentence and #166's, recorded
here because nothing in the repository read ``random_patch_frac`` before this module did.

## The composite: what it is for, and what it must not do

§4.2's inference clause is *"predicted patches only where voided, original pixels elsewhere,
then a 16 px cross-fade at patch boundaries inside the void"*, plus four shifted encodings
averaged for voids near tile scale.

The artifact it removes is structural rather than incidental. The decoder predicts each 16 px
patch through its own token, so two patches meeting inside a void have no term anywhere in the
objective that makes them agree at the shared edge. What prints is a straight line on the 16 px
grid — and §5's **seam-gap** column measures exactly that shape of discontinuity. Without the
composite that column would charge A3 for its inference scheme rather than for its fill, which
is the confusion #124 asked to be made auditable and #166 asks a test to audit.

**A cross-fade needs two things to fade between**, and one decode gives one value per pixel. So
the fade is an overlap-add over four decodes on four grids: the native one and the three at half
a patch in y, x and both (:func:`crossfade_origins`). Every native patch boundary falls at the
*centre* of a shifted grid's patch, where that grid has no seam of its own, and the triangular
weights sum to exactly one at every pixel (:func:`fade_weights`). The width is §4.2's 16 px and
the offsets are not in §4.2 at all — they are the minimal set a 16 px fade is a partition of
unity over, and they are derived from the patch size rather than configured, because a shift
that was not half a patch would leave some boundary unblended.

The shifted grids are cut from a **reflect-padded** tile rather than a rolled one. A roll wraps
terrain from one edge to the other and puts a cliff through the tile; §3.2 keeps every void at
least ``edge_margin_px`` from the edge, so reflection touches an 8 px border that no void
reaches and no scored pixel is predicted from wrapped content.

**What the composite must not do is fade across the void boundary.** That discontinuity is the
fill meeting the terrain — the thing §5's seam-gap column exists to measure — and smoothing it
would be the arm marking its own homework. So the blend runs over the *predicted field only*,
and the composite's last act is to put every non-void pixel back exactly as it arrived
(:func:`composite`).

## §4.2's struck clause, and the rule that struck it

§4.2 used to end *"For voids near tile scale, 4 shifted encodings (0 and 112 px offsets in x/y)
averaged, which suppresses patch-grid seams (standard test-time practice)."* #166 built it
literally, off :attr:`benchmark.config.Finetune.shift_offsets`, and measured it; #177 struck it
from the plan on 2026-09-22, and :func:`usable_shifts` carries the reasons. Two are arithmetic
rather than empirical: 112 is seven whole patches, so a 112 px shift moves the tile onto a
**differently-indexed but identically-phased** grid — every patch boundary lands where it was,
which is precisely what a shift aimed at patch-grid seams would have to move; and the model takes
exactly one 224 px tile, so a 224 px window displaced by 112 px is three-quarters canvas that is
not the tile. The third is the measurement: **45.0 m** with the four encodings against **21.0 m**
without, on the trained R5 arm over 200 large-bucket instances. What such shifts *would* buy at a
usable offset is real but different — each puts the void at a different place in the
position-embedding grid, so the position-specific part of the error averages down — and the
fade's own four half-patch grids already are that, at the largest offset that is still a view of
the tile.

The machinery stays, and so does the rule: ``shift_offsets`` now declares the unshifted encoding
alone, and an offset re-declared later is judged on the canvas it needs rather than on the number
112. :func:`grid_energy` is what settles the seam question on real tiles, and #166's own test is
written against the cross-fade for that reason — the fade is the clause that does the work the
plan used to attribute to both.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Sequence

import numpy as np

from dataset.metrics import land_mask

from . import masks, zeroshot

#: §4.2's near-tile-scale trigger, as a name rather than a number. A void reaches it when its
#: equivalent diameter reaches §3.2's **large** bucket floor — the size axis every draw is
#: already measured on and every results row already carries, so the four shifted encodings are
#: spent on the same voids the printed table calls large. A constant here instead would be a
#: second size vocabulary for a reader to reconcile with the first.
NEAR_TILE_SCALE_BUCKET: Final = "large"


# --- The fade, as geometry ----------------------------------------------------


def padded_tile(
    normalized: np.ndarray, void: np.ndarray, pad: int, *, margin: int
) -> tuple[np.ndarray, np.ndarray]:
    """The tile and its void, edge-padded together — the canvas the shifted grids are cut from.

    A 224 px window of a 224 px tile that is shifted at all runs off the canvas by exactly as
    much as it shifted, so the shifted grids need a border to be cut from. Three ways to make one
    and only the third is both correct and non-degenerate; #166 found that by running the other
    two.

    **Not a roll.** Wrapping brings terrain from one edge to the other and puts a cliff through
    the middle of the tile — which is exactly where a near-tile-scale void is — and it moves the
    void into the border strip the fade's grids cannot cover.

    **Not a reflection.** Mirroring copies elevations, so a void within ``pad`` of the edge has
    its own true values copied into the border. Pad the mask with ``False`` there and
    :func:`benchmark.zeroshot.keep_set` hands those patches to the encoder as ordinary visible
    terrain: the arm reads the answer it is being scored on filling, and the score still looks
    plausible (measured at §3.2's minimum interior margin: 65 true void pixels in 5 visible
    patches). Mirror the mask with it and the leak closes, but at ``pad = 112 + 8`` the void's
    reflections tile the window and hide **every** token —
    :func:`pretrain.model.keep_masking` refuses, correctly, because what comes back from a
    fully-masked encoder is the decoder's unconditional prior.

    **Edge replication, and §3.2 is what makes it exact.** The outermost row and column of a tile
    are void-free — ``edge_margin_px`` keeps every draw at least 8 px inside — so replicating
    them produces a border that provably carries no void pixel at all. The mask is then padded
    ``False`` because that is *true* of the padding rather than convenient: nothing is hidden
    that should be visible, and nothing is shown that should be hidden. The refusal below is what
    turns "§3.2 guarantees it" into "this tile satisfies it", since a void that reached the edge
    would silently make the padding a leak again.
    """
    void = np.asarray(void, dtype=bool)
    if margin > 0 and (
        void[:margin].any()
        or void[-margin:].any()
        or void[:, :margin].any()
        or void[:, -margin:].any()
    ):
        raise ValueError(
            f"this void reaches within {margin} px of the tile edge, so the edge-replicated "
            f"border would carry void pixels into patches the keep-set leaves visible — the arm "
            f"would read the terrain it is being scored on filling. §3.2 keeps every draw fully "
            f"interior at {margin} px (`benchmark.masks.NotInterior`), so this is a void the "
            f"generator should not have produced"
        )
    return np.pad(normalized, pad, mode="edge"), np.pad(void, pad, mode="constant")


def crossfade_origins(patch_px: int) -> tuple[tuple[int, int], ...]:
    """The four grid origins the cross-fade blends over: ``(0,0)``, ``(0,h)``, ``(h,0)``, ``(h,h)``.

    Derived from the patch size rather than configured, and half a patch is not a tunable choice:
    it is the unique offset that puts one grid's patch **boundaries** at another's patch
    **centres**, which is the whole mechanism. An offset of 4 px would leave the fade blending
    two fields that both seam at 16 px, one of them 4 px along.
    """
    half = patch_px // 2
    return ((0, 0), (0, half), (half, 0), (half, half))


def fade_weights(*, tile_px: int, patch_px: int, crossfade_px: int) -> np.ndarray:
    """``(4, tile_px, tile_px)`` blend weights, summing to exactly 1 at every pixel.

    One weight per grid of :func:`crossfade_origins`, separable in y and x. Along an axis, a
    grid's weight ramps linearly from 0 at its own patch boundaries to 1 at distance
    ``crossfade_px / 2`` — so a field contributes nothing where it seams and everything where it
    is most confident, and at the plan's ``crossfade_px == patch_px`` the two offset ramps are
    exact complements before they are normalized at all.

    Normalized regardless, because ``crossfade_px`` is a config knob: a narrower fade leaves the
    raw weights summing to more than one in the patch interiors, and a blend that did not divide
    it out would print the grid it was built to remove as a few per cent of ripple.
    """
    if crossfade_px <= 0:
        raise ValueError(f"crossfade_px {crossfade_px} must be positive")
    if crossfade_px > patch_px:
        raise ValueError(
            f"crossfade_px {crossfade_px} outruns the {patch_px} px patch it fades across: the "
            f"fade is centred on each patch boundary and half of it has to fit inside the patch "
            f"on either side, so a wider one would reach past the next boundary (§4.2)"
        )
    half = patch_px // 2
    coordinate = np.arange(tile_px) + 0.5

    def ramp(offset: int) -> np.ndarray:
        """One axis' weight for the grid whose boundaries sit at ``offset + k * patch_px``."""
        into = (coordinate - offset) % patch_px
        # Distance to this grid's nearest patch boundary, in pixels.
        edge = np.minimum(into, patch_px - into)
        return np.clip(edge / (crossfade_px / 2.0), 0.0, 1.0)

    axes = {0: ramp(0), half: ramp(half)}
    weights = np.stack(
        [np.outer(axes[dy], axes[dx]) for dy, dx in crossfade_origins(patch_px)]
    )
    total = weights.sum(axis=0)
    if not np.all(total > 0):
        raise ValueError(
            f"a {crossfade_px} px fade leaves pixels no grid contributes to on a {patch_px} px "
            f"patch grid — every pixel must be inside some grid's ramp"
        )
    return weights / total


def crossfade(fields: np.ndarray, *, patch_px: int, crossfade_px: int) -> np.ndarray:
    """Blend the four decoded fields of :func:`crossfade_origins` into one, seamlessly.

    ``fields`` is ``(4, H, W)`` in that function's order, each already mapped back onto the
    tile's own coordinates. The result carries no discontinuity at any of the four grids' patch
    boundaries, because each field's weight is zero at its own.
    """
    fields = np.asarray(fields, dtype=np.float64)
    expected = len(crossfade_origins(patch_px))
    if fields.ndim != 3 or fields.shape[0] != expected:
        raise ValueError(
            f"the cross-fade blends {expected} fields, one per grid of `crossfade_origins`; "
            f"got {tuple(fields.shape)}"
        )
    weights = fade_weights(
        tile_px=fields.shape[-1], patch_px=patch_px, crossfade_px=crossfade_px
    )
    return np.einsum("ghw,ghw->hw", weights, fields)


def composite(
    fields: np.ndarray,
    original: np.ndarray,
    void: np.ndarray,
    *,
    patch_px: int,
    crossfade_px: int,
) -> np.ndarray:
    """§4.2's clause, end to end: cross-fade the prediction, then restore the terrain.

    The order is the argument, and it is the same one :meth:`benchmark.zeroshot.Inpainter.infill`
    makes: the fade runs over the *predicted* field, and the non-void pixels are put back
    afterwards exactly as they arrived. A fade applied after the composite would blur across the
    void boundary — the one discontinuity in the tile that is a measurement rather than an
    artifact, and the one §5's seam-gap column is for.
    """
    blended = crossfade(fields, patch_px=patch_px, crossfade_px=crossfade_px)
    return np.where(np.asarray(void, dtype=bool), blended, np.asarray(original, np.float64))


# --- The statistic #166's criterion is stated on ------------------------------


def grid_energy(field: np.ndarray, *, patch_px: int) -> float:
    """How loudly the 16 px patch grid prints in ``field``, in the field's own units.

    The mean absolute step across pixel pairs that **straddle** a patch boundary, minus the mean
    absolute step across the pairs just inside the patches. Terrain crosses a patch boundary the
    same way it crosses anything else, so the subtraction is what makes this a reading of the
    *grid* rather than of the roughness: smooth terrain and rough terrain both score ~0, and a
    per-patch offset scores its own size.

    Deliberately **not** §5's seam-gap column. #166's criterion is that the composite suppresses
    the patch grid *rather than the seam-gap column absorbing it*, and evidence for that cannot
    be a reading of the column in question. This is a direct measurement of the artifact, over
    both axes, and nothing in the scored path calls it: it is what a test and a diagnostic ask.
    """
    field = np.asarray(field, dtype=np.float64)
    straddling, interior = [], []
    for axis in (0, 1):
        step = np.abs(np.diff(field, axis=axis))
        index = np.arange(step.shape[axis])
        # A step at index i joins pixels i and i+1, so it straddles a boundary when i+1 is a
        # multiple of the patch width. Everything else is interior to some patch.
        on_boundary = ((index + 1) % patch_px) == 0
        straddling.append(np.take(step, np.flatnonzero(on_boundary), axis=axis))
        interior.append(np.take(step, np.flatnonzero(~on_boundary), axis=axis))
    across = float(np.mean(np.concatenate([part.ravel() for part in straddling])))
    within = float(np.mean(np.concatenate([part.ravel() for part in interior])))
    return max(0.0, across - within)


# --- §4.2's near-tile-scale clause --------------------------------------------


def near_tile_scale(void: np.ndarray, *, masks_cfg) -> bool:
    """Is this void near tile scale — the voids §4.2 spends four shifted encodings on?

    Read off §3.2's own size axis: the void's equivalent diameter against the **large** bucket's
    floor. That keeps one size vocabulary in the benchmark rather than two, and it answers for
    §3.3's real family as well, which carries no bucket column of its own.
    """
    area = int(np.asarray(void, dtype=bool).sum())
    if not area:
        return False
    floor = dict(masks_cfg.buckets)[NEAR_TILE_SCALE_BUCKET][0]
    return masks.equivalent_diameter(area) >= floor


def usable_shifts(recipe, *, patch_px: int) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    """§4.2's shifted encodings split into the ones that are a real view of the tile, and the rest.

    A shifted window is cut from a padded canvas (:func:`padded_tile`), so a shift of ``d`` px
    buys its view at the cost of ``d`` px of fabricated border. **The rule is that the fabricated
    part must be no wider than the patch the fade blends across.** The fade's own half-patch
    grids cost 8 px of border against a 16 px patch and pass it; §4.2's 112 px shifts cost 112
    and do not, because a 224 px window of a 224 px tile displaced by half a tile is
    three-quarters padding — there is no larger canvas for it to come from.

    **Measured, on the trained R5 arm over 200 large-bucket instances** (#166): averaging the
    four 112 px encodings scores **45.0 m** against the cross-fade alone at **21.0 m**, with A1
    at 28.4 m. So the clause as stated does not merely fail to help, it is what turns A3's large
    bucket from a win over the classical baseline into a loss of the same size — and it does it
    by averaging in predictions the model made from a context that was mostly invented.

    Two further facts sit behind the threshold rather than in it. §4.2 credits these shifts with
    *"suppress[ing] patch-grid seams"*, which they cannot do at 112: that is seven whole patches,
    so every boundary lands exactly where it was. And the clause the plan credits is built and
    works — :func:`crossfade` is what moves the grid statistic, which is what #166's test
    measures.

    Returned as a pair rather than filtered silently. What was dropped and why is recorded in the
    run record (:func:`composite_note`), because a pre-registered arm scored under fewer
    encodings than §4.2 names is a deviation a reader has to be able to find.

    #177 struck the 112 px clause from §4.2 on this rule's evidence (2026-09-22), so the declared
    offsets are now ``(0,)`` and nothing is dropped by default. The rule stays as the refusal a
    re-declared offset meets: it is about the canvas, and it was never about the number.
    """
    kept, dropped = [], []
    for origin in shift_origins(recipe):
        (kept if max(origin) <= patch_px else dropped).append(origin)
    return tuple(kept), tuple(dropped)


def composite_note(cfg) -> dict:
    """What the composite actually ran, for the run record — §4.2's clause and this box's reading.

    Written whenever A3 is scored, not only when something is dropped, so the record says what
    the composite did rather than only when it disagreed with the plan.
    """
    from dataset import config as dataset_config

    patch_px = dataset_config.PATCH_PX
    kept, dropped = usable_shifts(cfg.finetune, patch_px=patch_px)
    note = {
        "crossfade_px": cfg.finetune.crossfade_px,
        "crossfade_grids": len(crossfade_origins(patch_px)),
        "shift_offsets_declared": list(cfg.finetune.shift_offsets),
        "shifted_encodings_declared": len(shift_origins(cfg.finetune)),
        "shifted_encodings_run": len(kept),
    }
    if dropped:
        note["shifted_encodings_dropped"] = [list(o) for o in dropped]
        widths = sorted({max(origin) for origin in dropped})
        note["deviation"] = (
            f"the declared offsets include {widths} px, and a 224 px window of a 224 px tile "
            f"displaced that far is more fabricated padding than the fade blends across, so "
            f"those encodings are not a view of this tile at all. §4.2 declared 0 and 112 px "
            f"until #177 struck the clause on this measurement (trained R5 arm, 200 "
            f"large-bucket instances): 45.0 m with them against 21.0 m without, and A1 at "
            f"28.4 m. The cross-fade — the clause that actually moves the patch-grid statistic "
            f"— is unaffected and still runs on four grids."
        )
    return note


def shift_origins(recipe) -> tuple[tuple[int, int], ...]:
    """§4.2's four shifted encodings, as ``(dy, dx)`` pairs off ``finetune.shift_offsets``.

    The cross product of the declared offsets with themselves — *"0 and 112 px offsets in x/y"*
    — rather than four hand-written pairs, so a config that declared a third offset would produce
    nine encodings instead of silently running four of them.
    """
    return tuple(
        (dy, dx) for dy in recipe.shift_offsets for dx in recipe.shift_offsets
    )


# --- §4.2's recipe, and the guard that keeps it §4.2's ------------------------


def assert_declared(cfg) -> None:
    """Refuse a cell whose §4.2 recipe has drifted from the declared one (#166).

    :func:`benchmark.tune.assert_frozen`'s counterpart for the arm that does not train §4.1's
    head, and it exists because A3 fell through every gate that one guards. That check is asked
    of arms whose ``trains`` is ``head`` or ``everything``; A3's is ``encoder+decoder``, so a
    ``--set finetune.encoder_lr=3e-4`` on a cell that later reached the parquet was refused by
    nothing at all — the exact failure §4.1's guard was built to prevent, one arm along.

    The two guards differ in what they are protecting, and the difference is worth stating.
    §4.1's recipe is frozen because it was **tuned**, and a second tuning pass would be a thumb
    on the scale. §4.2's was never tuned: every number in it is stated in the plan with an
    argument beside it — 20 epochs because this is an adaptation and not a pretraining, 0.75
    because that is MAE's own ladder, ``norm_pix`` off because void filling needs the absolute
    per-patch scale. So a drifted knob here is not an extra search; it is an arm that is no
    longer the arm §1's table names, trained against a bar registered for the one it was.

    Called from :func:`benchmark.launch.resolve` rather than from the pass, beside the §4.1
    refusal and for its reason: ``--dry-run`` returns before the verb runs, so a check inside the
    pass would let ``--dry-run --arm A3-R5 --set finetune.decoder_lr=1e-2`` exit 0 with the
    resolution line printing the moved rate under §4.2's words.
    """
    from .config import DECLARED_FINETUNE

    if cfg.finetune == DECLARED_FINETUNE:
        return
    moved = sorted(
        name
        for name in type(DECLARED_FINETUNE).__dataclass_fields__
        if getattr(cfg.finetune, name) != getattr(DECLARED_FINETUNE, name)
    )
    raise ValueError(
        f"§4.2 states this arm's recipe rather than searching for it, and "
        f"{', '.join(moved)} {'has' if len(moved) == 1 else 'have'} drifted from the declared "
        f"line: "
        + ", ".join(
            f"{name} {getattr(cfg.finetune, name)!r} against {getattr(DECLARED_FINETUNE, name)!r}"
            for name in moved
        )
        + ". Every number in §4.2 is stated with an argument beside it, and none of them is one "
        "a run may move on its own — an A3 trained at another recipe is scored against a bar "
        "registered for the arm §1's table names. If the plan really has changed, "
        "`config.DECLARED_FINETUNE` is the one committed line to move."
    )


def format_declared(recipe) -> list[str]:
    """§4.2's recipe as the resolution line prints it — :func:`benchmark.tune.format_frozen`'s twin.

    Two lines rather than one because the arm has two halves that move independently: the
    schedule the encoder and decoder are adapted under, and the objective they are adapted
    against. ``norm_pix`` is on the second because it is the flag §4.2 inverts, and the one A2
    has no way past (§4.3, #173) — a reader who finds it printed ``True`` here is looking at a
    cell that is not A3.
    """
    return [
        f"{'finetune':<16} {recipe.epochs} epochs, encoder lr {recipe.encoder_lr:.1e} with "
        f"{recipe.layer_decay} layer-wise decay, decoder lr {recipe.decoder_lr:.1e}, "
        f"{recipe.decay}, {recipe.precision}",
        f"{'objective':<16} masked {recipe.loss.upper()} in per-tile-standardized space, "
        f"norm_pix {'on' if recipe.norm_pix_loss else 'off'}; composite cross-fades "
        f"{recipe.crossfade_px} px, shifts {recipe.shift_offsets} px for near-tile-scale voids",
    ]


# --- §4.2's mask sampler ------------------------------------------------------


def finetune_keep_set(
    void: np.ndarray, *, patch_px: int, n_visible: int, rng: np.random.Generator
) -> np.ndarray:
    """The patches the encoder sees during the fine-tune: ``(n_patches,)`` boolean.

    §4.2's sampler, and the module docstring argues the reading. Every patch the void touches is
    hidden — *touching*, not covering, on :func:`benchmark.zeroshot.touched_patches`' rule, since
    a patch holding one void pixel would carry the fill value through every attention layer —
    and patches are then hidden at random until exactly ``n_visible`` survive.

    ``n_visible`` is the **model's** own, off the checkpoint's mask ratio rather than a constant
    here, so the fine-tune's token count is pretraining's by construction and not by coincidence.

    Flattened in :meth:`~pretrain.model.MaskedAutoencoder.patchify`'s row-major order, which is
    the order the encoder's positional embedding and the decoder's are both cut in: a keep-set
    built column-first reconstructs a transposed tile, which renders as plausible terrain and
    scores as a number.
    """
    touched = zeroshot.touched_patches(void, patch_px=patch_px).reshape(-1)
    n_patches = touched.size
    if n_visible < 1 or n_visible > n_patches:
        raise ValueError(f"n_visible {n_visible} must lie in [1, {n_patches}]")
    free = np.flatnonzero(~touched)
    if free.size < n_visible:
        raise ValueError(
            f"this void touches {n_patches - free.size} of {n_patches} patches, more patches "
            f"than pretraining's mask budget leaves visible ({n_visible}): there is no random "
            f"top-up that reaches pretraining's ratio, and dropping below it would make this "
            f"epoch a different recipe from the last one. §3.2's {'40%'} area cap should keep "
            f"every draw far below this (the bank's largest touches 108 of 196)"
        )
    keep = np.zeros(n_patches, dtype=bool)
    # Chosen over the patches the void does not touch, uniformly, so the survivors are scattered
    # over the whole tile — §4.2's "keeps seeing global context" — rather than clustered around
    # the void, which would make this a local extrapolation problem instead of pretraining's.
    keep[rng.choice(free, size=n_visible, replace=False)] = True
    return keep


# --- §4.2's optimizer ---------------------------------------------------------


def parameter_groups(model, recipe) -> list[dict]:
    """§4.2's two learning rates and its 0.75 layer-wise decay, as AdamW groups.

    *"lr 1e-4 (encoder) with 0.75 layer-wise decay, 1.5e-3 (decoder)"*. Three facts, and the
    split between them is :func:`pretrain.model.is_decoder_parameter` — the same predicate §14's
    export uses to decide what a release throws away, so "the decoder half" means one thing in
    this repository rather than two.

    The decay is MAE's own: the patch embedding and the class token are layer 0, block *i* is
    layer *i+1*, the final encoder norm is the top, and layer *l* trains at
    ``encoder_lr * layer_decay ** (top - l)``. The lowest layers move least, which is what makes
    a 20-epoch adaptation an adaptation rather than a short, bad pretraining.

    Each group carries ``part`` and ``layer`` beside the two keys torch reads. They cost nothing
    and they are what a test can asserts the ordering on without re-deriving it.
    """
    from pretrain.model import is_decoder_parameter

    top = len(model.blocks) + 1
    groups: dict[tuple[str, int], dict] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if is_decoder_parameter(name):
            part, layer, lr = "decoder", top, recipe.decoder_lr
        else:
            part = "encoder"
            layer = _encoder_layer(name, top=top)
            lr = recipe.encoder_lr * recipe.layer_decay ** (top - layer)
        group = groups.setdefault(
            (part, layer), {"part": part, "layer": layer, "lr": lr, "params": []}
        )
        group["params"].append(parameter)
    return [groups[key] for key in sorted(groups)]


def _encoder_layer(name: str, *, top: int) -> int:
    """Which layer of the decay ladder an encoder parameter sits on."""
    if name.startswith("blocks."):
        return int(name.split(".")[1]) + 1
    # `cls_token` and `patch_embed` are the input layer; the final `norm` rides with the top
    # block, which is MAE's own convention and the one the 0.75 ladder was measured under.
    return 0 if name in ("cls_token",) or name.startswith("patch_embed.") else top


# --- The composite, driven through a model ------------------------------------


@dataclass(frozen=True)
class Compositor:
    """§4.2's inference composite over one tile — the decodes, the fade and the restore.

    One tile at a time, on :class:`benchmark.zeroshot.Inpainter`'s rule and for its reason: the
    keep-set is a fact about *this* void, so two tiles keep different patches and there is no
    single gather width for them. What is batched here instead is the **grids** — a tile's four
    (or sixteen) decodes differ only in where the patch boundaries fall, so they go through the
    encoder as one batch of four and the pass costs one forward, not four.
    """

    model: Any
    recipe: Any
    masks_cfg: Any

    def decodes_for(self, void: np.ndarray) -> int:
        """How many decodes this void costs — four, or sixteen when it is near tile scale."""
        shifts = (
            usable_shifts(self.recipe, patch_px=self.model.patch_px)[0]
            if near_tile_scale(void, masks_cfg=self.masks_cfg)
            else ((0, 0),)
        )
        return len(shifts) * len(crossfade_origins(self.model.patch_px))

    def infill(
        self, raw_i16: np.ndarray, void: np.ndarray, *, crossfade: bool = True
    ) -> np.ndarray:
        """§4.2's composite, end to end. Returns **metres**.

        The order is :meth:`benchmark.zeroshot.Inpainter.infill`'s, and for its reasons: the
        per-tile statistics are taken over ``valid = land ∧ ¬void`` (ADR-0012 §1) so nothing
        inside the void reaches the normalization, and the non-void pixels are restored last from
        the tile that was handed in.

        ``crossfade=False`` is the single-decode composite — §4.2's first clause without its
        second. Nothing in the scored path passes it; it is what #166's test measures the fade
        *against*, and a diagnostic is a better home for that than a second code path.
        """
        from . import extract, metrics

        raw_i16 = np.asarray(raw_i16)
        if raw_i16.dtype != np.int16:
            raise TypeError(
                f"{extract.ADR} §1: A3 takes **raw int16 metres** and applies §2.1's per-tile "
                f"diet itself; it was handed {raw_i16.dtype}"
            )
        void = np.asarray(void, dtype=bool)
        valid = land_mask(raw_i16) & ~void
        # Raises when the tile has no valid pixel, which is ADR-0012's own refusal.
        extract.tile_statistics(raw_i16, valid)
        normalized = extract.normalize(
            raw_i16[None], valid[None], normalization="per_tile"
        )[0]

        shifts = (
            usable_shifts(self.recipe, patch_px=self.model.patch_px)[0]
            if near_tile_scale(void, masks_cfg=self.masks_cfg)
            else ((0, 0),)
        )
        phases = crossfade_origins(self.model.patch_px) if crossfade else ((0, 0),)
        predicted = self._averaged(normalized, void, shifts, phases)
        in_metres = metrics.to_metres(predicted, raw_i16, valid)
        return np.where(void, in_metres, np.asarray(raw_i16, dtype=np.float64))

    def _averaged(self, normalized, void, shifts, phases) -> np.ndarray:
        """§4.2's shifted encodings, each cross-faded, averaged where they are defined.

        **Averaged over the shifts that cover a pixel, not over all four unconditionally**, and
        that qualification is forced rather than chosen. §4.2 asked for *"4 shifted encodings
        (0 and 112 px offsets in x/y)"* until #177 struck the clause, and a 112 px shift of a 224 px tile through a 224 px
        model cannot produce four encodings that each see the whole tile: the window runs off the
        canvas by exactly as much as it shifted. Implemented as a ``roll`` it does not run off at
        all — it wraps, which is worse, because a wrap puts a cliff of far-side terrain through
        the middle of the tile and a near-tile-scale void is exactly the void that cliff crosses.
        (#166 found this the direct way: the wrapped void landed in the 8 px border the fade's
        grids cannot cover, and the composite refused itself.)

        So the shifts are cut from the same reflect-padded tile the fade's grids are, and each
        one contributes where it reaches. The unshifted encoding reaches everywhere, so every
        pixel is predicted by at least one and the average is never empty.

        Two things about §4.2's former clause were recorded here before #177 struck it, and they
        are two of its three reasons. The first is the coverage above. The second is that
        the mechanism the plan attributes to these shifts — *"which suppresses patch-grid
        seams"* — does not follow from the offsets it names: 112 is seven whole patches, so a
        112 px shift lands on a differently-**indexed** but identically-**phased** grid and every
        patch boundary stays exactly where it was. What the shifts do buy is real and different:
        each one puts the void at a different place in the position-embedding grid, so the
        position-specific part of the error averages down. The clause that does the work §4.2
        credits here is the cross-fade, which is why #166's test is written against that one.
        """
        half = self.model.patch_px // 2
        pad = max(max(sy for sy, _ in shifts), max(sx for _, sx in shifts)) + half
        origins = [
            (sy + dy, sx + dx) for sy, sx in shifts for dy, dx in phases
        ]
        fields, covered = self._decode_grids(normalized, void, origins, pad=pad)

        per_shift, weights = [], []
        for index in range(len(shifts)):
            block = slice(index * len(phases), (index + 1) * len(phases))
            group, group_covered = fields[block], covered[block].all(axis=0)
            if len(phases) > 1:
                # The fade needs every phase grid over this shift's window; where one is missing
                # the blend is undefined, so the whole shift stands down for that pixel.
                filled = np.where(covered[block], group, group[0][None])
                faded = crossfade(
                    filled, patch_px=self.model.patch_px, crossfade_px=self.recipe.crossfade_px
                )
            else:
                faded = group[0]
            per_shift.append(faded)
            weights.append(group_covered.astype(np.float64))

        total = np.sum(weights, axis=0)
        uncovered = np.asarray(void, dtype=bool) & (total == 0)
        if uncovered.any():
            raise ValueError(
                f"{int(uncovered.sum())} void pixels are reached by no encoding at all. The "
                f"fade's grids are windows of a reflect-padded tile shifted by {half} px, so "
                f"they give up a strip at one edge — and §3.2 keeps every draw at least "
                f"{self.masks_cfg.edge_margin_px} px from the edge so that no scored pixel is "
                f"ever in it. A void that reaches the edge is one §3.2's interior constraint "
                f"should have refused"
            )
        return np.sum([f * w for f, w in zip(per_shift, weights)], axis=0) / np.maximum(total, 1.0)

    def _decode_grids(
        self,
        normalized: np.ndarray,
        void: np.ndarray,
        origins: Sequence[tuple[int, int]],
        *,
        pad: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Decode ``normalized`` on each grid of ``origins``, mapped back onto tile coordinates.

        The grids are windows of a **reflect-padded** tile rather than of a rolled one: a roll
        wraps terrain from one edge to the other and puts a cliff through the tile, while
        reflection only ever touches the border strip §3.2 keeps every void out of.

        **The void is reflected with it, and that is a leak rather than a nicety.** Reflection
        copies elevations, so a void within ``pad`` of the edge has its own true values copied
        into the padded border — and if the mask were padded with ``False`` there, `keep_set`
        would hand those patches to the encoder as ordinary visible terrain. The arm would then
        read the answer it is being scored on filling, most at exactly the sizes §4.2 spends the
        most decodes on: at ``pad = 112 + 8`` the reflected band is 120 px wide. Measured on a
        void at §3.2's minimum interior margin, padding the mask with ``False`` put 65 true void
        pixels into 5 patches the encoder could see. The mask is a property of the pixels, so it
        is copied exactly the way they are.
        """
        import torch

        patch_px = self.model.patch_px
        tile_px = normalized.shape[-1]
        padded, padded_void = padded_tile(
            normalized, void, pad, margin=self.masks_cfg.edge_margin_px
        )

        windows, keeps = [], []
        for dy, dx in origins:
            top, left = pad - dy, pad - dx
            windows.append(padded[top : top + tile_px, left : left + tile_px])
            keeps.append(
                zeroshot.keep_set(
                    padded_void[top : top + tile_px, left : left + tile_px], patch_px=patch_px
                )
            )

        # Onto the model's own device, read off a parameter rather than carried as a field: a
        # `Compositor` is built from a model that is already somewhere, and a device it was
        # *told* about is one that can disagree with where the weights actually sit. The tiles
        # come back to numpy at the end, because everything downstream of `infill` — the
        # per-tile inversion, the composite, `benchmark.metrics` — is numpy in metres.
        device = next(self.model.parameters()).device
        with torch.no_grad():
            tiles = torch.from_numpy(np.stack(windows)).unsqueeze(1).float().to(device)
            keep = torch.from_numpy(np.stack(keeps)).to(device)
            # One forward per grid rather than one batch of grids: a shift moves which patches
            # the void touches, so two grids keep different numbers of tokens and
            # `pretrain.model.keep_masking` has no single gather width for them.
            decoded = [
                self.model.unpatchify(
                    self.model.forward_decoder(*self._encode(tiles[i : i + 1], keep[i : i + 1]))
                )[0, 0].float().cpu().numpy()
                for i in range(len(windows))
            ]

        fields = np.full((len(origins), tile_px, tile_px), np.nan, dtype=np.float64)
        for index, (dy, dx) in enumerate(origins):
            # Window index i holds original index i - dy, so the window runs out at the far edge
            # by exactly the shift. Everything it does not reach stays NaN and is reported as
            # uncovered rather than filled with something plausible.
            keepable = tile_px - dy, tile_px - dx
            fields[index, : keepable[0], : keepable[1]] = decoded[index][dy:, dx:]
        covered = np.isfinite(fields)
        return np.nan_to_num(fields, nan=0.0), covered

    def _encode(self, tiles, keep):
        latent, _, ids_restore = self.model.encode_visible(tiles, keep)
        return latent, ids_restore


# --- §4.2's pass --------------------------------------------------------------


@dataclass(frozen=True)
class Adapted:
    """What A3 trains, and the three facts every consumer of it asks for.

    :class:`benchmark.train.Stack`'s counterpart for an arm that has no frozen encoder and no
    head: the whole model is the trainable part, so there is nothing to clump except the model,
    the arm it was fitted for and the device it sits on. Carried on :class:`benchmark.train.
    Trained` so the eval pass scores the object the val curve was measured on rather than a
    second load of the same file.

    No ``extractor``. A3 runs no ADR-0012 feature extraction at all — it runs its own decoder —
    so :func:`benchmark.launch._measured_agreement` finds nothing here and the run record says
    that no precision measurement was taken, which is the true reading rather than a gap.
    """

    module: Any
    arm: str
    device: str


class VoidExamples:
    """``(tile_id, mask_rle)`` pairs as raw tiles and voids — a ``torch.utils.data.Dataset``.

    :class:`benchmark.train.Examples`' sibling, and the difference is one line: this one does
    **not** run A1's fill. §2.4's fill is the frozen arms' input convention — they predict a
    residual over it — and A3's convention is ``token_drop``, which reads the tile and the void
    and nothing else. §7 measures head training as 4.1x loader-bound *because of* that fill, so
    composing one here would be paying the frozen arms' dominant per-tile cost for an arm that
    never looks at it.

    Holds paths and strings only, never an open handle, on :class:`benchmark.train.Examples`'
    rule — and carries the same split guard through the process boundary, so "a training pass
    cannot reach a test tile" stays true in the workers as well as in the parent.
    """

    def __init__(self, cfg, items, tiles, index=None) -> None:
        self.cfg = cfg
        self.items = list(items)
        self.tiles = tiles
        self.index = index
        self._store = None

    def __len__(self) -> int:
        return len(self.items)

    def store(self):
        if self._store is None:
            from .evaluate import TileStore

            self._store = TileStore(self.cfg.paths.corpus_dir, index=self.index)
        return self._store

    def __getitem__(self, index: int):
        tile_id, mask_rle = self.items[index]
        raw = self.store().read(self.tiles.allow(tile_id)).raw_i16
        return {"raw_i16": raw, "void": masks.decode_rle(mask_rle)}


def collate_voids(items) -> dict[str, np.ndarray]:
    """Stack per-tile reads into one batch. A named function so it survives pickling."""
    return {
        name: np.stack([item[name] for item in items]) for name in ("raw_i16", "void")
    }


def void_batches(cfg, items, tiles, *, batch_size: int, index=None):
    """Yield ``{raw_i16, void}`` batches over ``items``, in order, on the configured workers."""
    from . import train

    dataset = VoidExamples(cfg, items, tiles, index)
    workers = train.workers_for(len(items), batch_size, cfg.runtime.loader_workers)
    if workers <= 0:
        for start in range(0, len(dataset), batch_size):
            yield collate_voids(
                [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
            )
        return

    from torch.utils.data import DataLoader

    yield from DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_voids,
    )


def batch_masking(batch, *, model, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """§2.1's per-tile diet and §4.2's keep-sets, for one batch.

    Returns the standardized tiles ``(N, H, W)`` and the keep-sets ``(N, n_patches)``. The
    statistics are taken over ``valid = land ∧ ¬void`` (ADR-0012 §1), exactly as the eval pass
    takes them: a void's own level and spread are unknowable at inference, so a fine-tune that
    normalized by them would train an arm that cannot be run.
    """
    from . import extract

    raw, void = batch["raw_i16"], batch["void"]
    valid = land_mask(raw) & ~void
    normalized = extract.normalize(raw, valid, normalization="per_tile")
    keep = np.stack(
        [
            finetune_keep_set(
                void[i], patch_px=model.patch_px, n_visible=model.n_visible, rng=rng
            )
            for i in range(void.shape[0])
        ]
    )
    return normalized, keep


def checkpoint_path(cfg, arm) -> Path:
    """The **pretraining** checkpoint A3 initializes from, off this box's ``Paths``.

    §14's release export is encoder-only, so the ``encoder.pt`` A4 initializes from cannot start
    this arm: §4.2 continues MAE training, and there is no MAE without the decoder. The field is
    the registry's (:attr:`benchmark.arms.Encoder.mae_checkpoint_field`) — the same one A2 reads,
    for the same reason — and the resolution line names it before `run` spends anything.
    """
    from . import arms, train

    encoder = arms.ENCODERS[arm.encoder]
    field = encoder.mae_checkpoint_field
    path = None if field is None else getattr(cfg.paths, field)
    if path is None:
        raise train.TrainError(
            f"arm {arm.name} continues MAE training from {encoder.name}'s pretraining "
            f"checkpoint and this box has not mounted one — set paths.{field}. §14 exports "
            f"encoder-only weights, so the release `encoder.pt` cannot initialize an arm that "
            f"trains a decoder; the release manifest's `source_checkpoint` names the file that "
            f"can, with its md5"
        )
    return Path(path)


def load_model(cfg, arm, *, device: str):
    """A3's starting weights — encoder, decoder and ``mask_token``, loaded strictly.

    Through :func:`benchmark.zeroshot.load_inpainter` rather than beside it, because the checks
    that matter are the ones it already makes: a strict load whose *shape* of disagreement is the
    diagnosis, and the registry's declared width, depth and ``pos_embed`` axis order against what
    the file records. An encoder-only export reaches a named refusal here instead of a randomly
    initialized decoder and a plausible table.

    The ``norm_pix`` refusal :meth:`~benchmark.zeroshot.Inpainter.check_ready` makes is
    deliberately **not** asked. A2 cannot retrain and so has no way past a decoder that predicts
    in per-patch units the void hides; A3's whole recipe is that it retrains with ``norm_pix``
    off (§4.2), so R5 and R4 arriving with it on is the starting condition rather than a fault.
    """
    from . import arms

    return zeroshot.load_inpainter(
        checkpoint_path(cfg, arm), expect=arms.ENCODERS[arm.encoder]
    ).model.to(device)


def val_rmse(instances, cfg, model, *, tiles, index=None) -> float:
    """Pooled RMSE in metres over every val void — the curve A3's run record carries.

    Pooled over pixels rather than averaged over instances, which is
    :func:`benchmark.train.val_rmse`'s rule and its reason: a per-instance mean would weight a
    16 px void the same as a 158 px one, a factor of ~100 in area.

    Scored through the **single-decode** composite, not §4.2's four-grid one. The cross-fade is
    an inference-time construction whose effect #166 measures directly (:func:`grid_energy`), and
    running it here would multiply a per-epoch instrument by four for a number nothing is
    selected on — this pass keeps its last epoch, not its best. So this is the fine-tune's own
    progress, and the scored rows are the composite's.
    """
    from .evaluate import TileStore

    compositor = Compositor(model=model, recipe=cfg.finetune, masks_cfg=cfg.masks)
    # One store for the pass. Rebuilding it per tile re-opened the shard handle 2,000 times an
    # epoch for no gain — `benchmark.train.Examples` holds one per worker for the same reason.
    store = TileStore(cfg.paths.corpus_dir, index=index)
    squares, count = 0.0, 0
    for item in instances:
        raw = store.read(tiles.allow(item.tile_id)).raw_i16
        void = masks.decode_rle(item.mask_rle)
        filled = compositor.infill(raw, void, crossfade=False)
        error = filled[void] - raw.astype(np.float64)[void]
        squares += float(np.square(error).sum())
        count += int(void.sum())
    if not count:
        from . import train

        raise train.TrainError("§3.1's val draw produced no void pixels to score on")
    return math.sqrt(squares / count)


def train_finetune(cfg, arm, *, epochs: int | None = None, on_epoch=None):
    """§4.2's short adaptation for one cell — A3 (#166).

    :func:`benchmark.train.train_unet`'s sibling one arm along, and **not** a third caller of
    :func:`benchmark.train._fit`, which is the one structural difference worth arguing. That
    function is §4.1's schedule over §2.4's convention: it builds an A1 fill, adds a residual
    inside the void, and scores §4.1's L1-plus-slope loss at one learning rate. A3 shares none of
    those four — §4.2 gives it MAE's own objective over dropped tokens, two learning rates and a
    layer-wise ladder between them — so routing it through `_fit` would put four branches inside
    the one function whose whole value is that it has none. What A3 *does* share it reads from
    the same place: :func:`benchmark.train._training_inputs` opens §3.1's draws behind the split
    guard and §3.4's bank, so *"A3 reads the same draws the frozen arms train against"* is one
    call rather than two functions agreeing.

    The order:

    1. **The two draws and the bank**, behind the guard that cannot reach a test tile.
    2. **The model, from the pretraining checkpoint** — encoder *and* decoder (:func:`load_model`).
    3. **Twenty epochs, cosine, and the last one kept.** Not the best one: §4.1's head declares
       an ``early_stop_metric`` and §4.2 declares none, so A3 runs its schedule to the end and
       keeps what it produced. Selecting an epoch on val would be a second selection this plan
       does not register, and a cosine run to zero is a recipe whose last epoch *is* its answer.
       The val curve is recorded all the same, so a fine-tune that diverged is visible in the run
       record rather than only in the table.

    The schedule is :class:`benchmark.train.Cosine`'s curve applied as a **factor**. §4.2 gives
    the encoder and the decoder different rates and puts a 0.75 ladder under the encoder's, so an
    ``apply`` that wrote one learning rate onto every group — which is what the frozen arms'
    schedule does, correctly, for a recipe with one rate — would flatten the ladder on step 0 and
    leave no trace of having done it. There is no warmup, because §4.2 declares none.
    """
    import torch

    from pretrain.loss import MaeLoss, pixel_weights_from_land_mask

    from . import train

    started = time.perf_counter()
    if arm.trains != "encoder+decoder":
        raise train.TrainError(
            f"arm {arm.name} trains {arm.trains!r}; §4.2's fine-tune is for the arm that trains "
            f"the encoder and the decoder together. A4, A5 and A6 train §4.1's head and go "
            f"through `train_head`; A7 trains everything and goes through `train_unet`"
        )
    recipe = cfg.finetune
    epochs = recipe.epochs if epochs is None else epochs
    device = train.resolve_device(cfg)

    inputs = train._training_inputs(cfg)
    tiles, train_ids = inputs.tiles, inputs.train_ids
    banked, val = inputs.banked, inputs.val

    model = load_model(cfg, arm, device=device)
    # §4.2's flag, and the one thing about the objective this arm changes. Off **here**, on the
    # fine-tune's own loss, rather than read off the checkpoint: R5 and R4 were both pretrained
    # with it on, and the point of the adaptation is that A3 stops predicting in the per-patch
    # units a void hides — the fact §4.3 has no way past and #173 is open on.
    objective = MaeLoss(norm_pix_loss=recipe.norm_pix_loss)
    groups = parameter_groups(model, recipe)
    optimizer = torch.optim.AdamW([{**group, "base_lr": group["lr"]} for group in groups])
    steps_per_epoch = max(1, math.ceil(len(train_ids) / cfg.head.batch_size))
    curve = train.Cosine(
        peak_lr=1.0, total_steps=max(1, steps_per_epoch * epochs), warmup_steps=1
    )

    directory = Path(cfg.runtime.checkpoint_dir) / f"{cfg.arm}-{cfg.size}-seed{cfg.seed}"
    directory.mkdir(parents=True, exist_ok=True)

    from .evaluate import TileStore

    index = TileStore.index_for(
        cfg.paths.corpus_dir, [*train_ids, *(item.tile_id for item in val)]
    )
    order = np.random.default_rng(cfg.seed)
    # The top-up's own stream, separate from the one that orders the epoch. The *voids* come off
    # §3.4's committed bank and only the random patches are drawn here, so the two are
    # distinguishable in a way a shared generator would hide.
    sampler = np.random.default_rng(masks.stable_seed("finetune", cfg.arm, cfg.seed))

    history: list[train.Epoch] = []
    step, last = 0, None
    for epoch in range(epochs):
        epoch_started = time.perf_counter()
        model.train()
        shuffled = [
            (train_ids[i], banked.rle(train_ids[i], epoch))
            for i in order.permutation(len(train_ids))
        ]
        total_loss, batches_seen = 0.0, 0
        factor = 1.0
        for batch in void_batches(
            cfg, shuffled, tiles, batch_size=cfg.head.batch_size, index=index
        ):
            factor = _apply(optimizer, curve, step)
            optimizer.zero_grad(set_to_none=True)
            normalized, keep = batch_masking(batch, model=model, rng=sampler)
            land = land_mask(batch["raw_i16"])
            # The denominator the loss will divide this whole batch by, known before the first
            # forward — see `eligible_patches`. `clip` covers the batch in which the valid-pixel
            # gate leaves nothing eligible at all, which `MaeLoss` itself answers with a zero
            # gradient rather than a NaN.
            batch_weight = float(
                eligible_patches(keep, land, model=model).sum()
            )
            for piece, piece_keep, piece_land in _pieces_with(
                normalized, keep, land, cfg.runtime.micro_batch
            ):
                images = torch.from_numpy(piece).unsqueeze(1).to(device)
                with train._autocast(device, recipe.precision):
                    latent, mask, ids_restore = model.encode_visible(
                        images, torch.from_numpy(piece_keep).to(device)
                    )
                    predicted = model.forward_decoder(latent, ids_restore)
                # The target is the **true** tile, inside the void as well as outside: A3 trains
                # on synthetic voids over terrain the corpus holds, and the void's patches are
                # dropped from the encoder rather than from the objective. Land weighting is the
                # loss's own (§4) — ocean is not terrain to reconstruct; the void is.
                target = model.patchify(images.float())
                weights = pixel_weights_from_land_mask(
                    torch.from_numpy(piece_land).to(device), model.patchify
                )
                terms = objective(predicted, target, mask, pixel_weights=weights)
                # This piece's share of the *eligible patches*, not of the samples: the loss
                # is a mean over those, so this is what makes the accumulated gradient the
                # gradient of the unsplit batch exactly rather than approximately.
                share = (
                    float(eligible_patches(piece_keep, piece_land, model=model).sum())
                    / batch_weight
                    if batch_weight
                    else piece.shape[0] / normalized.shape[0]
                )
                (terms.loss * share).backward()
                total_loss += float(terms.loss.detach()) * share
            optimizer.step()
            step += 1
            batches_seen += 1

        model.eval()
        last = train.Epoch(
            epoch=epoch,
            loss=total_loss / max(1, batches_seen),
            # §4.1's two diagnostics are that loss's terms, and §4.2's objective has neither:
            # MAE's masked MSE is one number in standardized units. Reported as `nan` rather
            # than as 0.0, which would read as "measured, and flat".
            elevation=float("nan"),
            slope=float("nan"),
            lr=recipe.encoder_lr * factor,
            val_rmse=val_rmse(val, cfg, model, tiles=tiles, index=index),
            seconds=time.perf_counter() - epoch_started,
        )
        history.append(last)
        train.save_checkpoint(
            model, last, directory / train.EPOCH_CHECKPOINT.format(epoch=epoch)
        )
        if on_epoch is not None:
            on_epoch(last)

    if last is None:
        raise train.TrainError("§4.2's fine-tune ran no epochs, so there is nothing to score")
    # The **last** epoch, not the best — see the docstring. Written under the frozen arms'
    # `BEST_CHECKPOINT` name because it is what the pass kept and what the run record points at;
    # which epoch that is, is `best_epoch`.
    kept = train.save_checkpoint(model, last, directory / train.BEST_CHECKPOINT)
    model.eval()
    counts = model.parameter_counts()
    return train.Trained(
        module=model,
        width=None,
        parameters=int(counts["total"]),
        best_epoch=last.epoch,
        best_val_rmse=last.val_rmse,
        history=tuple(history),
        seconds=time.perf_counter() - started,
        checkpoint=str(kept),
        tiles=len(train_ids),
        sources=tiles.sources,
        stack=Adapted(module=model, arm=arm.name, device=device),
        precision_declared=recipe.precision,
        precision_used=train.precision_on(device, recipe.precision),
        kind="finetune",
    )


def _apply(optimizer, curve, step: int) -> float:
    """§4.2's cosine as a **factor** on each group's own rate. See :func:`train_finetune`."""
    factor = curve.at(step)
    for group in optimizer.param_groups:
        group["lr"] = group["base_lr"] * factor
    return factor


def eligible_patches(keep: np.ndarray, land: np.ndarray, *, model, recipe=None) -> np.ndarray:
    """Per sample, how many patches :class:`pretrain.loss.MaeLoss` will actually average over.

    The loss reduces as ``sum(per_patch * w) / sum(w)``, where ``w`` is 1 for a patch that is
    **masked** and holds at least ``min_valid_patch_frac`` of a patch's pixels on land (§4's
    gate, :func:`pretrain.loss._active_patches`). So ``sum(w)`` — not the sample count — is what
    a micro-batch contributes to the batch's mean, and it is derivable here without a forward
    pass: the mask is ``~keep`` and the valid count is the patchified land mask's row sums.

    That is what makes the split exact. :func:`benchmark.train._fit` makes the same argument for
    §4.1's loss in its own terms — *"a split whose pieces each contributed their own mean would
    silently reweight a batch by how much void happened to fall in each piece"* — and the
    quantity that does it here is ocean rather than void: a piece holding part-ocean tiles has
    fewer eligible patches than a piece of pure land, and weighting both by their sample count
    tilts the accumulated gradient toward whichever piece had less sea in it.
    """
    from pretrain.loss import MaeLoss

    floor = (MaeLoss().min_valid_patch_frac if recipe is None else 0.25) * model.patch_dim
    grid = model.grid_px
    # The land mask cut into the decoder's own (N, n_patches, patch_dim) layout, by the same
    # row-major reshape `patchify` uses — counted rather than patchified through torch, because
    # this runs before the forward and only needs a sum.
    counts = (
        np.asarray(land, dtype=bool)
        .reshape(land.shape[0], grid, model.patch_px, grid, model.patch_px)
        .sum(axis=(2, 4))
        .reshape(land.shape[0], -1)
    )
    active = (counts > 0) & (counts >= floor)
    return (active & ~np.asarray(keep, dtype=bool)).sum(axis=1)


def _pieces_with(normalized: np.ndarray, keep: np.ndarray, land: np.ndarray, size: int):
    """Split a batch into GPU-sized slices, in order. :func:`benchmark.train.micro_batches`' rule.

    Contiguous slices, so the order samples reach the model in is the order the unsplit batch
    would have used — which is what lets the accumulated gradient be the same quantity rather
    than merely a similar one. The land mask rides along because each slice's loss weights are
    cut from it.
    """
    total = normalized.shape[0]
    if size <= 0 or size >= total:
        return [(normalized, keep, land)]
    return [
        (normalized[s : s + size], keep[s : s + size], land[s : s + size])
        for s in range(0, total, size)
    ]
