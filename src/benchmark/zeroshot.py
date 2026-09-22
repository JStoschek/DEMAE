"""A2 — DEMAE zero-shot, and R12, the rule its row is read under (§4.3, #167).

Two halves, and they are one ticket because R12 makes them one. §4.3's mechanism is a single
sentence — *mask = patches touching the void; the decoder reconstructs them and the non-void
pixels are composited back* — and the register's rule for reading the resulting number is three
clauses long. Splitting them across two modules would put the declaration that **A2 carries no
C-claim** somewhere other than the code that makes it true.

## What "no trained head at all" means here

Every other learned arm in §1's table puts something trainable between the encoder and the
output: A4/A5/A6 share §4.1's head, A3 fine-tunes the encoder and decoder, A7 is a UNet. A2 has
none of them. It runs the **pretrained MAE as published** — the same encoder and the same
shallow decoder §2.2 describes — with one change, and the change is not a weight: the mask is
*read off the void* instead of drawn at 75% random. That is what
:meth:`pretrain.model.MaskedAutoencoder.encode_visible` is for, and it is why the masking helper
lives in the model rather than here (:func:`pretrain.model.keep_masking`).

It is also why A2 needs a **pretraining checkpoint** and not the release encoder. §14 exports
encoder-only weights and throws the decoder away (:func:`pretrain.model.is_decoder_parameter` is
that definition), so ``.release-work/release-R5-v2/encoder.pt`` — the file A4 and A3 initialize
from — holds no ``decoder_*`` tensor and no ``mask_token``. An arm whose whole mechanism *is* the
pretrained decoder cannot read it. :attr:`benchmark.arms.Encoder.mae_checkpoint_field` names the
second file, and it is resolved and refused like any other input rather than discovered late.

## The scale the pretrained decoder does not predict, and §4.3's registered estimate of it

R5 and R4 were both pretrained with ``norm_pix_loss`` **on**, so their decoder predicts each
patch standardized by *that patch's own* mean and variance (:func:`pretrain.loss.
normalized_targets`). Inverting that needs those two numbers, and inside a void they are exactly
what is unknown. :func:`pretrain.recon.predicted_in_tile_units` recovers them from the target,
which is right for §10.2's reconstruction grids — they are drawn *to be looked at*, beside the
truth — and would be a leak here: the fill would carry the voided terrain's own level and
spread, and A2's row would beat every arm in the table for a reason no column names.

§4.2 states the underlying fact and acts on it for A3 — *"masked MSE in per-tile-standardized
space with `norm_pix` **off** (void filling needs absolute per-patch scale; norm-pix targets
discard it)"* — but A3 gets to retrain. A2 cannot: zero-shot means the decoder as pretrained. So
#167 built this module to **refuse** a ``norm_pix`` checkpoint rather than invent an estimator,
because a method that determines a pre-registered row's value is a decision for §4.3 and not for
a loader; and #173 made that decision, before any A2 instance was scored. §4.3 now registers,
per voided patch, **the level as the mean of A1's own fill over the patch** and **the spread as
the inverse-distance-weighted mean of the per-patch standard deviation over the patches the
encoder saw** (:func:`estimate_scale`, off :class:`benchmark.config.Zeroshot`). Both read only
what the arm is handed; neither reads the void. The estimator is a recipe decision, owned by the
config hash and guarded by :func:`assert_declared`, and it is written into every A2 run record
(:func:`scale_note`) so a reader of the row knows which two numbers stood in for the unknown two.
The ``norm_pix``-off path is unchanged, so a checkpoint from R3's ablation (#84) or an A3-regime
run scores through this module as before.

## R12, and what "mechanised" means for a discursive rule

ADR-0013 files R12 as **discursive**: its trigger is an *ordering* against A1 rather than a
level, so there is no constant for `task1-prereg-bar.json` to hold, and the enforcement is the
writeup's and the reviewer's. What *can* be mechanised is not the rule but the **outcome** — so
:func:`disposition` computes the majority-of-bands comparison, states the verdict in the
register's own words, and marks the row as carrying no C-claim, all into the run record. A
reviewer then reads a disposition rather than reconstructing one from 60 printed numbers, which
is the state #167 was opened against: A2 is the entry the tunability test found where no claim
pointed at one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterable, Mapping

import numpy as np

from dataset.metrics import land_mask

from . import arms, extract, metrics, prereg, voids

#: The metric R12's contingency is stated on. Not the whole roster: §4.3 names RMSE, and a rule
#: that triggered on whichever column happened to be first would be a different rule.
COMPARISON_METRIC: Final = "rmse"

#: The results columns the comparison reads: the band it groups on, the instance key it
#: intersects on, and the metric. Named here rather than at the reader because they are R12's
#: requirements — a comparison that quietly read a column it had not asked for would be one
#: whose result depended on the roster.
COMPARISON_COLUMNS: Final = ("band", "family", "bucket", "tile_id", "instance", COMPARISON_METRIC)


# --- The void, as a keep-set over the token grid -------------------------------


def touched_patches(void: np.ndarray, *, patch_px: int) -> np.ndarray:
    """``(grid, grid)`` boolean — which patches the void reaches at all.

    *Touching*, not covering, and the difference is §4.3's. A patch holding one void pixel is a
    patch the encoder must not see: the pretrained encoder has no notion of a partially-valid
    token, so a patch admitted for its 255 good pixels would carry the void's fill value into the
    representation and out again through every attention layer.
    """
    height, width = void.shape
    if height % patch_px or width % patch_px:
        raise ValueError(
            f"a {height}x{width} tile is not a whole number of {patch_px} px patches, so the "
            f"void does not cut on the token grid the encoder embeds"
        )
    grid_h, grid_w = height // patch_px, width // patch_px
    blocks = np.asarray(void, dtype=bool).reshape(grid_h, patch_px, grid_w, patch_px)
    return blocks.any(axis=(1, 3))


def keep_set(void: np.ndarray, *, patch_px: int) -> np.ndarray:
    """The patches the encoder **sees**, flattened in :meth:`~pretrain.model.
    MaskedAutoencoder.patchify`'s row-major order.

    Row-major is not a detail: the encoder's positional embedding and the decoder's are both cut
    in that order, so a keep-set built column-first reconstructs a transposed tile — which
    renders as plausible terrain and scores as a number.
    """
    return ~touched_patches(void, patch_px=patch_px).reshape(-1)


# --- The inpainter -------------------------------------------------------------


# --- §4.3's per-patch scale, estimated without reading the void -----------------

#: The registered estimators, by knob. A name outside these is refused at load rather than
#: falling through to whichever branch is last.
LEVEL_ESTIMATORS: Final = ("a1_fill",)
SPREAD_ESTIMATORS: Final = ("visible_idw",)


def patch_view(field: np.ndarray, *, patch_px: int) -> np.ndarray:
    """``(H, W)`` -> ``(n_patches, patch_px²)`` in :meth:`~pretrain.model.MaskedAutoencoder.
    patchify`'s row-major order — the same permutation, in numpy, so a per-patch statistic taken
    here lands on the patch the decoder predicted."""
    grid = field.shape[0] // patch_px
    return (
        np.asarray(field)
        .reshape(grid, patch_px, grid, patch_px)
        .transpose(0, 2, 1, 3)
        .reshape(grid * grid, patch_px * patch_px)
    )


def unpatch(patches: np.ndarray, *, patch_px: int) -> np.ndarray:
    """:func:`patch_view`'s exact inverse."""
    grid = int(round(np.sqrt(patches.shape[0])))
    return (
        np.asarray(patches)
        .reshape(grid, grid, patch_px, patch_px)
        .transpose(0, 2, 1, 3)
        .reshape(grid * patch_px, grid * patch_px)
    )


def idw_weights(targets: np.ndarray, sources: np.ndarray, *, grid: int, power: float) -> np.ndarray:
    """``(len(targets), len(sources))`` inverse-distance weights over patch-centre distance.

    Distances are in patch units on the ``grid × grid`` token grid, so a source one patch away
    weighs ``1`` and one ``d`` patches away weighs ``d^-power``; each row sums to one. A target
    that is also a source would sit at distance zero, and that is refused rather than floored —
    the caller hands the voided patches as targets and the seen ones as sources, and the two sets
    are disjoint by construction (:func:`keep_set`).
    """
    ty, tx = np.divmod(np.asarray(targets, dtype=np.int64), grid)
    sy, sx = np.divmod(np.asarray(sources, dtype=np.int64), grid)
    d2 = (ty[:, None] - sy[None, :]) ** 2 + (tx[:, None] - sx[None, :]) ** 2
    if (d2 == 0).any():
        raise ValueError("a patch cannot be both a target and a source of the spread estimate")
    weights = np.power(d2.astype(np.float64), -power / 2.0)
    return weights / weights.sum(axis=1, keepdims=True)


def estimate_scale(
    normalized: np.ndarray,
    *,
    raw_i16: np.ndarray,
    void: np.ndarray,
    valid: np.ndarray,
    keep: np.ndarray,
    patch_px: int,
    var_floor: float,
    recipe: Any,
    fill_recipe: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """§4.3's registered ``(μ̂, σ̂)`` per patch, ``(n_patches,)`` each, in standardized units.

    Patches the encoder saw get their **own** statistics — the decoder's output there is
    restored from the input anyway, so what they carry is the exact inverse for the pixels the
    composite keeps. Patches the void touches get the estimates:

    * **level** — ``a1_fill``: the mean over the patch of A1's fill of this very void
      (:func:`benchmark.fill.fill`, the arm's own recipe), taken through the same per-tile
      standardization the encoder's input had. Every visible pixel in the patch is the truth;
      only the void pixels are the interpolant.
    * **spread** — ``visible_idw``: the inverse-distance-weighted mean of the seen patches'
      standard deviations (:func:`idw_weights`, unbiased ``n - 1`` like :func:`pretrain.loss.
      patch_statistics`), floored at ``sqrt(var_floor)`` — the loss's own clamp, so a flat
      neighbourhood cannot hand the decoder a zero to multiply by.

    Neither reads a void pixel's true value. ``keep`` is the encoder's keep-set, so a patch the
    void touches is never a source; :func:`keep_masking` has already refused a void that touches
    every patch, which is the one case with no source at all.
    """
    from . import fill as fill_module

    if recipe.level not in LEVEL_ESTIMATORS:
        raise ValueError(
            f"zeroshot.level {recipe.level!r} is not a registered level estimator "
            f"({', '.join(LEVEL_ESTIMATORS)}) — §4.3 names one, and #173 is why a loader does "
            f"not pick another"
        )
    if recipe.spread not in SPREAD_ESTIMATORS:
        raise ValueError(
            f"zeroshot.spread {recipe.spread!r} is not a registered spread estimator "
            f"({', '.join(SPREAD_ESTIMATORS)})"
        )
    keep = np.asarray(keep, dtype=bool)
    patches = patch_view(normalized, patch_px=patch_px).astype(np.float64)
    grid = normalized.shape[0] // patch_px

    mean = patches.mean(axis=1)
    std = patches.std(axis=1, ddof=1)

    touched = np.flatnonzero(~keep)
    seen = np.flatnonzero(keep)
    if touched.size:
        if not seen.size:
            raise ValueError(
                "the void touches every patch, so there is no seen patch to estimate a spread "
                "from — `keep_masking` refuses this case first, and reaching here means it did not"
            )
        # Level: A1's fill of this void, in the units the decoder saw. `tile_statistics` is the
        # same call `to_metres` inverts with, so the two transforms cannot drift apart.
        tile_mean, tile_std = extract.tile_statistics(raw_i16, valid)
        filled_m = fill_module.fill(
            np.asarray(raw_i16, dtype=np.float64), void, recipe=fill_recipe
        )
        filled_std = (filled_m - tile_mean) / max(tile_std, extract.STD_FLOOR)
        mean[touched] = patch_view(filled_std, patch_px=patch_px)[touched].mean(axis=1)
        # Spread: what the neighbourhood's roughness says, weighted by how near it is.
        weights = idw_weights(touched, seen, grid=grid, power=recipe.spread_power)
        std[touched] = weights @ std[seen]

    floor = float(np.sqrt(var_floor))
    return mean, np.maximum(std, floor)


def invert_patches(
    predicted: np.ndarray, mean: np.ndarray, std: np.ndarray, *, patch_px: int
) -> np.ndarray:
    """``pred · σ + μ`` per patch, back to one ``(H, W)`` field — :func:`pretrain.recon.
    predicted_in_tile_units`'s arithmetic with the estimates in place of the target's own."""
    patches = np.asarray(predicted, dtype=np.float64)
    restored = patches * np.asarray(std)[:, None] + np.asarray(mean)[:, None]
    return unpatch(restored, patch_px=patch_px)


def assert_declared(cfg) -> None:
    """Refuse a cell whose §4.3 estimator has drifted from the declared one (#173).

    :func:`benchmark.finetune.assert_declared`'s twin, for the same reason one arm over: the
    estimator sets A2's numbers, so an A2 scored under another one is not the row §1's table
    names, and an estimator chosen after seeing the A1 ordering is exactly the tunability R12
    exists to prevent. Called from :func:`benchmark.launch.resolve`, before ``--dry-run``
    returns.
    """
    from .config import DECLARED_ZEROSHOT

    if cfg.zeroshot == DECLARED_ZEROSHOT:
        return
    moved = sorted(
        name
        for name in type(DECLARED_ZEROSHOT).__dataclass_fields__
        if getattr(cfg.zeroshot, name) != getattr(DECLARED_ZEROSHOT, name)
    )
    raise ValueError(
        f"§4.3 registers how a voided patch's scale is estimated rather than searching for it, "
        f"and {', '.join(moved)} {'has' if len(moved) == 1 else 'have'} drifted from the "
        f"declared line: "
        + ", ".join(
            f"{name} {getattr(cfg.zeroshot, name)!r} against {getattr(DECLARED_ZEROSHOT, name)!r}"
            for name in moved
        )
        + ". The estimator sets A2's numbers and R12's verdict, so one moved on a run is an arm "
        "scored under a method the plan does not name. If §4.3 really has changed, "
        "`config.DECLARED_ZEROSHOT` is the one committed line to move."
    )


def scale_note(cfg) -> dict:
    """What stood in for the unknown per-patch scale, for the run record (§4.3, #173)."""
    return {
        "level": cfg.zeroshot.level,
        "spread": cfg.zeroshot.spread,
        "spread_power": cfg.zeroshot.spread_power,
        "fill_variant": cfg.fill.variant,
        "reads_void": False,
        "statement": (
            "the decoder predicts each patch standardized by its own mean and standard "
            "deviation; inside the void those are estimated as the mean of A1's fill over the "
            "patch and the inverse-distance-weighted standard deviation of the patches the "
            "encoder saw (§4.3, #173). Neither reads a void pixel's true value."
        ),
    }


@dataclass(frozen=True)
class Inpainter:
    """The pretrained MAE, run as an inpainter over one tile at a time.

    One tile at a time and not a batch, and that is forced rather than lazy: the keep-set is a
    fact about *this* void, so two tiles in one batch keep different numbers of tokens and there
    is no single gather width for them (:func:`pretrain.model.keep_masking` refuses it). Batching
    would mean grouping 30,000 instances by keep count, which trades the pass's simplicity for a
    speed-up on a §7 line nobody has priced.
    """

    model: Any
    #: Whether the checkpoint's own recipe had §4's per-patch target normalization on. Read off
    #: the checkpoint's recorded :class:`~pretrain.config.RunConfig` rather than assumed — the
    #: whole point of :class:`ScaleUnavailable` is that this is a property of *the file*.
    norm_pix: bool
    var_floor: float
    #: What this model was loaded from, for the refusals and the run record.
    source: str
    #: §4.3's registered estimator (:class:`benchmark.config.Zeroshot`) and A1's recipe
    #: (:class:`benchmark.config.Fill`), which the level estimate runs. ``None`` means the
    #: declared ones — the only values a scored cell may carry (:func:`assert_declared`).
    recipe: Any = None
    fill_recipe: Any = None

    def check_ready(self) -> None:
        """Refuse now, before an instance is scored, if the estimator is not a registered one.

        The same rule §8's two gates follow: a refusal that arrives after 30,000 instances is one
        that costs an eval pass to consult. A ``norm_pix`` checkpoint is no longer refused — #173
        registered the inversion in §4.3 — but an estimator name §4.3 does not carry still is.
        """
        recipe = self._recipe()
        if recipe.level not in LEVEL_ESTIMATORS or recipe.spread not in SPREAD_ESTIMATORS:
            raise ValueError(
                f"{self.source}: zeroshot recipe names an estimator §4.3 does not register — "
                f"level {recipe.level!r} (registered: {', '.join(LEVEL_ESTIMATORS)}), spread "
                f"{recipe.spread!r} (registered: {', '.join(SPREAD_ESTIMATORS)})"
            )

    def _recipe(self):
        from .config import DECLARED_ZEROSHOT

        return DECLARED_ZEROSHOT if self.recipe is None else self.recipe

    def _fill_recipe(self):
        from .config import Fill

        return Fill() if self.fill_recipe is None else self.fill_recipe

    def infill(self, raw_i16: np.ndarray, void: np.ndarray) -> np.ndarray:
        """§4.3, end to end: drop, reconstruct, composite back. Returns **metres**.

        The order is the argument. The statistics come first and are taken over ``valid = land ∧
        ¬void`` (ADR-0012 §1), so nothing inside the void reaches the normalization — a voided
        tile's true statistics are unknowable at inference, and standardizing by them would leak
        the answer into the *input* as surely as the per-patch scale would leak it into the
        output. The composite comes last and restores every pixel outside the void from the tile
        that was handed in, which is what makes :func:`benchmark.metrics.score`'s units check a
        free assertion rather than a second thing to trust.
        """
        import torch

        self.check_ready()
        if not isinstance(raw_i16, np.ndarray) or raw_i16.dtype != np.int16:
            raise TypeError(
                f"{extract.ADR} §1: the zero-shot arm takes **raw int16 metres** and applies "
                f"§2.1's per-tile diet itself; it was handed "
                f"{getattr(raw_i16, 'dtype', type(raw_i16).__name__)}"
            )
        void = np.asarray(void, dtype=bool)
        valid = land_mask(raw_i16) & ~void
        # Raises when the tile is all ocean, which is ADR-0012's own refusal and the right one:
        # a tile with no valid pixel has no per-tile statistic to normalize by.
        extract.tile_statistics(raw_i16, valid)

        normalized = extract.normalize(
            raw_i16[None], valid[None], normalization="per_tile"
        )
        keep = keep_set(void, patch_px=self.model.patch_px)

        # Onto the model's own device, read off a parameter rather than carried as a field —
        # `benchmark.finetune.Compositor`'s rule: the model is already somewhere.
        device = next(self.model.parameters()).device
        with torch.no_grad():
            tiles = torch.from_numpy(normalized).unsqueeze(1).to(device)  # (1, 1, 224, 224)
            latent, _, ids_restore = self.model.encode_visible(
                tiles, torch.from_numpy(keep).unsqueeze(0).to(device)
            )
            patches = self.model.forward_decoder(latent, ids_restore)  # (1, n_patches, p²)
        predicted_patches = patches[0].float().cpu().numpy()

        if self.norm_pix:
            # §4.3's registered inversion (#173): the decoder predicted each patch in that
            # patch's own standardized units, and inside the void the two numbers that undo it
            # are estimated from what the arm was handed — never from the void's truth.
            mean, std = estimate_scale(
                normalized[0],
                raw_i16=raw_i16,
                void=void,
                valid=valid,
                keep=keep,
                patch_px=self.model.patch_px,
                var_floor=self.var_floor,
                recipe=self._recipe(),
                fill_recipe=self._fill_recipe(),
            )
            predicted = invert_patches(
                predicted_patches, mean, std, patch_px=self.model.patch_px
            )
        else:
            # `norm_pix` off: the decoder already predicts in the per-tile standardized units
            # `normalize` produced, and §2.1's inversion is the whole of what is left.
            predicted = unpatch(predicted_patches, patch_px=self.model.patch_px)

        # `to_metres` reads its (mean, std) from the same `tile_statistics` call above rather
        # than a second copy, which is the property that keeps the transform and its inverse
        # from drifting apart.
        in_metres = metrics.to_metres(predicted, raw_i16, valid)
        return np.where(void, in_metres, np.asarray(raw_i16, dtype=np.float64))


def load_inpainter(
    path: Path | str,
    *,
    expect: arms.Encoder | None = None,
    recipe: Any = None,
    fill_recipe: Any = None,
) -> Inpainter:
    """Build the arm's model from a **pretraining** checkpoint — encoder, decoder and all.

    Strict loading, and the strictness is the check that matters: a release export handed to this
    function is missing every ``decoder_*`` tensor and ``mask_token``, and a tolerant load would
    silently give A2 a randomly-initialized decoder and a plausible table.

    ``expect`` is the registry's entry for the encoder this arm declares, and passing it turns
    three of :class:`benchmark.arms.Encoder`'s declared fields into refusals — width, depth, and
    the position-embedding axis order. The plan publishes **no md5 for a pretraining
    checkpoint** (it is not a released artifact; each release manifest names its own
    ``source_checkpoint`` and hash, but nothing in this plan reviews one), so the file's identity
    is *recorded* rather than pinned, and :class:`benchmark.artifacts.Resolved` puts its md5 in the
    run record either way. ``demae-vits`` was the standing example of that state until #164
    pinned it; every *encoder* in the registry now carries a reviewed md5, and it is the
    pretraining checkpoints alone that are recorded-not-pinned. These three checks are what is
    available in place of a pin, and the axis order is the one that matters most: the model rebuilds its sin-cos buffer from its own config, so a
    checkpoint recording MAE's column-first order loaded for R4 — whose weights learned
    mmselfsup's row-first one (P102 §4) — would place every token at the transposed position and
    reconstruct plausible, wrong terrain.
    """
    from pretrain.checkpoint import load_checkpoint
    from pretrain.model import MaskedAutoencoder

    checkpoint = load_checkpoint(path)
    if expect is not None:
        _check_against_registry(checkpoint.config.architecture, expect, path)
    model = MaskedAutoencoder.from_config(checkpoint.config)
    missing, unexpected = model.load_state_dict(checkpoint.model, strict=False)
    # `strict=False` then checked by hand, because the *shape* of the disagreement is the
    # diagnosis: an encoder-only export is missing the decoder half and nothing else, and saying
    # so beats a 150-key dump that a reader has to classify.
    decoder_missing = sorted(
        name for name in missing if _is_decoder_tensor(name)
    )
    if decoder_missing:
        raise ValueError(
            f"{path} carries no decoder: {len(decoder_missing)} tensor(s) missing, "
            f"{decoder_missing[:3]}... §14 exports encoder-only weights and throws the decoder "
            f"away, so a release `encoder.pt` cannot drive A2 — zero-shot inpainting *is* the "
            f"pretrained decoder (§4.3). Point `paths.<encoder>_mae_checkpoint` at the "
            f"pretraining checkpoint the release was exported from (its `source_checkpoint` "
            f"field names it)."
        )
    if missing or unexpected:
        raise ValueError(
            f"{path} does not match the architecture its own config declares — missing "
            f"{sorted(missing)[:3]}, unexpected {sorted(unexpected)[:3]}"
        )
    model.eval()
    loss = checkpoint.config.loss
    return Inpainter(
        model=model,
        norm_pix=loss.norm_pix_loss,
        var_floor=loss.norm_pix_var_floor,
        source=str(path),
        recipe=recipe,
        fill_recipe=fill_recipe,
    )


def _check_against_registry(arch: Any, encoder: arms.Encoder, path: Path | str) -> None:
    """The three fields the registry declares about this encoder, against what the file says."""
    for field, declared, recorded in (
        ("width", encoder.width, arch.encoder_dim),
        ("depth", encoder.depth, arch.encoder_depth),
        ("pos_encoding", encoder.pos_encoding, arch.pos_encoding),
    ):
        if declared != recorded:
            raise ValueError(
                f"{path} records {field} {recorded!r} and `benchmark.arms.ENCODERS` declares "
                f"{encoder.name} as {declared!r}. This plan publishes no md5 for a pretraining "
                f"checkpoint, so a file from a sibling run is refused on the fields the registry "
                f"does declare rather than absorbed by a shape error twelve blocks in."
            )


def _is_decoder_tensor(name: str) -> bool:
    from pretrain.model import is_decoder_parameter

    return is_decoder_parameter(name)


# --- R12: the claims table -----------------------------------------------------


def c_claim(arm: arms.Arm) -> str | None:
    """Which C-claim this arm's row is evidence for, or ``None`` — R12's first clause.

    Read off :mod:`benchmark.prereg`'s registered pairs rather than restated as a table here: R6
    names the four marked pairs and the claim each carries, so "A2 is named by none of C1–C4" is
    a fact this function *derives*, and an arm added to a claim later cannot leave a stale
    ``None`` behind. That derivation is the whole reason #167 exists — A2 is the register's
    newest entry and the one where no claim pointed at it, which is a reason to check rather than
    an exemption.
    """
    for entry in prereg.rule("R6").value["marked_pairs"]:
        if arm.name in entry["pair"]:
            return entry["claim"]
    return None


#: R12's second clause, in the register's own words. A constant rather than a sentence written at
#: the call site, because the *struck* wording — "quantifies precisely what the task adaptation
#: bought" — is what the rule exists to keep struck, and a phrasing retyped per record is one
#: that drifts back toward it.
A2_TO_A3: Final = (
    "a joint quantity — the cost of the distribution shift and the value of task adaptation, "
    "summed — and never attributed to task adaptation alone: A3 changes both at once relative "
    "to A2, and the arm that would separate them (DEMAE fine-tuned on random masks) does not "
    "exist and is not proposed (§4.3)"
)

#: R12's first clause, likewise.
INSULATION: Final = (
    "insulated from C1 in both directions: a zero-shot failure is evidence that the decoder "
    "does not transfer to block masks, not that the representation is weak — and a zero-shot "
    "success is not extra evidence for C1 either (§4.3)"
)

REPORTED_NEGATIVE: Final = "negative result"
REPORTED_ORDERING: Final = "an ordering against A1, stated"
REPORTED_PENDING: Final = "pending — the comparison is not yet computable"

#: The sentence §4.3 says a negative result is reported *as*. Quoted rather than paraphrased: R12
#: is discursive, so this string is the closest thing to an enforceable artifact it has.
NEGATIVE_STATEMENT: Final = (
    "zero-shot MAE inpainting does not transfer from random-mask pretraining to contiguous voids"
)

PROMINENCE: Final = (
    "reported with the prominence §6 item 7 requires, stated rather than narrated as the "
    "baseline half of a delta (§4.3)"
)


# --- R12: the majority-of-bands comparison -------------------------------------


def band_comparison(
    *, a1_rows: Iterable[Mapping[str, Any]], a2_rows: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    """Does A2 beat A1 in a majority of §5's four ruggedness bands on RMSE?

    §4.3 argues the shape at length and it is worth carrying here, because the shape is what
    makes the answer meaningful. The trigger is an **ordering** rather than a threshold in
    metres: p119 measured A1 itself at 0.87 m in the flattest quartile against 38.6 m in the
    roughest, so one number in metres would be four different bars. And it is a *majority of
    bands* rather than the roughest band alone, which buys robustness by replication — which is
    in turn why A2-vs-A1 is not one of §5's four marked bootstrap pairs (R6).

    Three things are computed rather than assumed, and each closes a way this could return a
    number that is not an answer:

    * **The comparison is over the instances both arms scored.** Two arms over different mask
      lists compared band-mean to band-mean is a comparison of two benchmarks, and it returns a
      float. The intersection's size is reported so a reader can see it was the whole list.
    * **A band neither arm scored is not a band A2 lost.** ``bands_compared`` is reported beside
      ``bands_won``, and "a majority of the four" is left unanswered — ``None`` — until all four
      are there. That is the distinction the memo turns on: *not computed yet* is not *did not
      beat A1*.
    * **Ties lose.** ADR-0013 repaired the register's own row to §4.3's wording for exactly this
      case: the trigger is "does **not** beat", so equality is not a win.
    """
    # Materialized first: both are walked three times below, so a generator would arrive empty
    # on the second pass and the comparison would answer "0 bands" over rows that were there.
    a1_rows, a2_rows = list(a1_rows), list(a2_rows)

    shared = _shared_instances(a1_rows, a2_rows)
    a1_shared = _mean_by_band(r for r in a1_rows if _key(r) in shared)
    a2_shared = _mean_by_band(r for r in a2_rows if _key(r) in shared)

    bands = sorted(set(a1_shared) & set(a2_shared))
    per_band = {
        band: {
            "a1": a1_shared[band],
            "a2": a2_shared[band],
            "a2_beats_a1": a2_shared[band] < a1_shared[band],
        }
        for band in bands
    }
    won = sum(1 for entry in per_band.values() if entry["a2_beats_a1"])
    registered = voids.N_BANDS

    if len(bands) < registered:
        majority: bool | None = None
        negative: bool | None = None
        why = (
            f"{len(bands)} of §5's {registered} ruggedness bands hold rows from both arms, so "
            f"\"a majority of the four\" is not yet answerable. "
            + (
                "No A1 rows are in the results file for this cell — A1 runs first and its "
                "numbers gate every learned arm (§4.5), so score it before reading A2's row."
                if not a1_rows
                else "Score both arms over the whole committed eval mask list."
            )
        )
    else:
        majority = won * 2 > registered
        negative = not majority
        why = (
            f"A2 beats A1 on {COMPARISON_METRIC} in {won} of §5's {registered} ruggedness "
            f"bands, which is {'a majority' if majority else 'not a majority'}"
        )

    return {
        "metric": COMPARISON_METRIC,
        "lower_is_better": True,
        "bands_registered": registered,
        "bands_compared": len(bands),
        "bands_won": won,
        "instances_compared": len(shared),
        "per_band": per_band,
        "beats_a1_in_a_majority": majority,
        "negative_result": negative,
        "why": why,
    }


def _key(row: Mapping[str, Any]) -> tuple:
    """What identifies one scored instance across two arms: the mask, not the arm that saw it."""
    return (row["family"], row["bucket"], row["tile_id"], row["instance"])


def _shared_instances(
    a1_rows: Iterable[Mapping[str, Any]], a2_rows: Iterable[Mapping[str, Any]]
) -> set[tuple]:
    return {_key(r) for r in a1_rows} & {_key(r) for r in a2_rows}


def _mean_by_band(rows: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    """Band → mean RMSE. Non-finite scores are dropped, and a band of them has no mean.

    Dropped rather than propagated as NaN, because a single non-finite instance would otherwise
    turn a whole band's mean into one — and a NaN compares false against everything, which is a
    band silently lost rather than a band reported as unscorable.
    """
    totals: dict[str, list[float]] = {}
    for row in rows:
        value = float(row[COMPARISON_METRIC])
        if np.isfinite(value):
            totals.setdefault(row["band"], []).append(value)
    return {band: float(np.mean(values)) for band, values in totals.items() if values}


# --- R12: the disposition that lands in the run record -------------------------


def governs(arm: arms.Arm) -> bool:
    """Is this arm R12's subject?

    Read off :attr:`benchmark.arms.Arm.runs_pretrained_decoder` rather than off the name, and
    the register's own reasoning is why that is the same question: R12 insulates A2 from C1
    because *"A2 is decoder-driven with no trained head"* (§4.3), which is exactly what that
    property says. An arm added later that runs a pretrained decoder zero-shot inherits the rule
    it would obviously be subject to, instead of missing it for want of an "A2" in its name.
    """
    return arm.runs_pretrained_decoder


def disposition(
    arm: arms.Arm,
    *,
    a1_rows: Iterable[Mapping[str, Any]],
    a2_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """R12's whole disposition for this cell, or ``None`` for an arm R12 does not govern.

    ``None`` rather than an empty block on every other row: a register entry attached to eleven
    arms is one a reviewer stops reading, and R12 is A2's rule.

    What lands in the record is a *disposition* and not a rule — the rule is ADR-0013's and is
    already in the committed bar's register. What was missing, and what #167 is about, is the
    line that says how this run's row is to be read: which claim it is evidence for (none), what
    its difference from A3 measures (a joint quantity), and whether the contingency fired.
    """
    if not governs(arm):
        return None
    comparison = band_comparison(a1_rows=a1_rows, a2_rows=a2_rows)
    if comparison["negative_result"] is None:
        reported = REPORTED_PENDING
    elif comparison["negative_result"]:
        reported = REPORTED_NEGATIVE
    else:
        reported = REPORTED_ORDERING

    return {
        "rule": "R12",
        "home": prereg.rule("R12").home,
        "arm": arm.name,
        "trains": arm.trains,
        "c_claim": c_claim(arm),
        "carries_c_claim": c_claim(arm) is not None,
        "insulation": INSULATION,
        "a2_to_a3": A2_TO_A3,
        "comparison": comparison,
        "reported_as": reported,
        "negative_result_statement": NEGATIVE_STATEMENT,
        "prominence": PROMINENCE,
    }


def format_disposition(record: Mapping[str, Any]) -> str:
    """R12's outcome on one line, for the person watching the run.

    Printed even when the contingency did not fire, on :func:`benchmark.evaluate.format_gates`'
    rule: a rule that only speaks when it triggers is one nobody can tell was consulted.
    """
    comparison = record["comparison"]
    return (
        f"R12  no C-claim; A2→A3 joint; "
        f"A2 beats A1 in {comparison['bands_won']}/{comparison['bands_compared']} of "
        f"{comparison['bands_registered']} bands on {comparison['metric']} "
        f"— reported as {record['reported_as']}"
    )
