"""§10.2's reconstruction grids: masked → predicted → truth, in metres, plus hillshade (#98).

Plan §10.2 asks for *"reconstruction grids every 10 epochs on a fixed panel: 2 val tiles from
each of the 20 strata (40 tiles), rendered masked → predicted → truth, inverted to metres via
manifest stats, plus a hillshade render of each"*. This module is the render half; which 40
tiles is :mod:`pretrain.panel`'s, and where the pictures go is :mod:`pretrain.sink`'s.

It is the visual channel of a run, and §10 opens by saying why one is needed at all: *"MAE
gives few in-flight signals"*. A falling loss says the model predicts patches better than it
did. It does not say **which terrain** it predicts badly, and that is the whole question here —
the corpus card §3.1 predicts, and
[ADR-0008](../../docs/adr/0008-accept-f3-drainage-shortfall-strata-2-17.md) accepts as a known
shortfall, that low-relief drainage is where this corpus is thinnest. A model that reconstructs
alpine ridges and smears drainage nets has one loss curve, not two, and the panel is stratified
so the difference has somewhere to show up.

## Why hillshade, and not only a heatmap

A heatmap of elevation is dominated by the low-frequency part of a tile — the broad slope —
because that is where the metres are. Drainage nets, terracing, dune fields and the smearing
that erases them live in the *derivative*, at amplitudes a hundred times smaller, and are
simply not visible on a colour ramp. Hillshading is a derivative, so it shows exactly the band
the heatmap cannot. Both are rendered, because the heatmap is what carries magnitude.

The shading here differs from :mod:`dataset.gallery`'s Gate-E sheets in one deliberate way:
**it is not contrast-stretched per panel** (``stretch=False``). The gallery stretches so each
tile is legible on its own terms, which is right for judging whether a stratum is a coherent
family and exactly wrong here — a stretched panel restores a smeared prediction to full
contrast and hides the one failure §10.2 was built to catch. For the same reason the vertical
exaggeration is computed from the **truth** and reused for all three panels: less relief has to
draw as less contrast.

## Two normalizations sit between the decoder and an elevation

Both have to be undone, and neither is visible in the picture if it is skipped:

1. **§4's `norm_pix`**, when it is on: the decoder predicts each patch in units of that
   patch's own standard deviation. Multiplying back by the *target's* per-patch mean and
   variance is the only way to place the prediction on an elevation axis at all — the model is
   never asked to predict a patch's absolute level, so it does not have one to give. That is
   §5's recorded design (per-tile input normalization plus `norm_pix` targets means the model
   "neither sees nor reconstructs absolute altitude anywhere"), and it is a real limit on what
   these grids show: **the level in a predicted patch is the truth's, and only the shape is the
   model's.** R3 (#84) ablates `norm_pix` off, and then this step does not happen — which is
   why it is read off the loss module rather than assumed.
2. **The loader's per-tile standardization**, undone by the same
   :class:`~dataset.loader.Normalizer` instance that applied it — reached through the dataset
   the samples came from, so a `global`-normalized run inverts with the corpus statistic and a
   `per_tile` one with the manifest's ``mean``/``std``, with no branch here to get wrong.

## Garbage in, garbage rendered

§15's checklist item 5 requires this to run on a **randomly initialized model**, so nothing
here may assume a sensible prediction. Three defences, and each maps to a way an untrained
decoder actually misbehaves:

- The colour ramp comes from the truth, so a prediction 10,000 sigma off **saturates** rather
  than rescaling the grid (and rather than rendering the truth as one flat block beside it).
- Non-finite values are **painted** :data:`BAD_RGB` rather than propagated. A NaN reaching
  W&B renders as a black pixel and reads as a reconstruction failure — a model problem, in a
  panel built to diagnose model problems — when it is a renderer bug.
- Every image leaves here as ``uint8`` in ``[0, 255]``, which is what
  :class:`pretrain.sink.Image` will accept.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import torch
from torch import Tensor

from dataset.gallery import OCEAN_RGB, SMOOTH_SIGMA_PX, auto_vert_exag, hillshade, smooth
from dataset.geometry import band_pixel_spacing_m

from .config import RunConfig
from .loss import normalized_targets, patch_statistics
from .panel import ResolvedPanel, read_panel, resolve_from_manifest
from .sink import Image, MetricValue

if TYPE_CHECKING:  # imported for the annotations alone — `train` pulls in the whole loop
    from dataset.loader import Normalizer

    from .train import Trainer

__all__ = [
    "BAD_RGB",
    "HEATMAP_KEY",
    "HILLSHADE_KEY",
    "MASK_RGB",
    "RAMP_PERCENTILES",
    "SEPARATOR_PX",
    "SEPARATOR_RGB",
    "ReconGrids",
    "TileReconstruction",
    "contact_sheet",
    "heat",
    "metre_ramp",
    "predict_the_target",
    "predicted_in_tile_units",
    "render_tile",
    "shade",
    "to_uint8",
    "triptych",
    "vertical_exaggeration",
]

#: Hidden patches, painted over both renders. Off the grey axis — it has chroma, so it cannot
#: be read as an elevation or a shading value — but desaturated, because at a 75% mask ratio
#: this is three quarters of the left-hand panel and a saturated colour there would be the only
#: thing anyone sees.
MASK_RGB: Final = (0.45, 0.42, 0.52)

#: Where a value was not finite. Loud on purpose: it should be impossible on a healthy run, and
#: when it happens the reader has to know it is *this pixel had no number*, not *this pixel
#: reconstructed to zero*.
BAD_RGB: Final = (0.94, 0.18, 0.55)

#: The panel separator — white, so the three panels never read as one wide image.
SEPARATOR_RGB: Final = (1.0, 1.0, 1.0)
SEPARATOR_PX: Final = 2

#: The percentiles of the **truth's** land pixels the grey ramp spans. Robust rather than
#: min/max: one spike (a void-fill artefact, a lone summit pixel) would otherwise compress the
#: entire tile into a few grey levels. Everything outside clips, which is visible as saturation
#: and is the honest rendering of "off the scale".
RAMP_PERCENTILES: Final = (2.0, 98.0)

#: Metres of ramp below which a tile is treated as flat rather than divided by. ADR-0002's
#: `degenerate` screen already drops land-std under 3 m, so this only catches a tile whose
#: *rendered* field is degenerate — an all-ocean panel, say.
_MIN_RAMP_M: Final = 1.0


# --- One tile's three fields ---------------------------------------------------


@dataclass(frozen=True)
class TileReconstruction:
    """One panel tile, in metres: what was hidden, what came back, and what was there.

    Everything here is **elevation**, on one axis, after both normalizations are undone —
    which is what lets the renderer below be a pure function of arrays and a latitude, with no
    knowledge of masks, patches or `norm_pix` anywhere in it.
    """

    tile_id: str
    stratum_id: int
    #: Tile-centre latitude, for the hillshade's true ground spacing.
    lat: float
    #: ``(H, W)`` metres, as stored.
    truth: np.ndarray
    #: ``(H, W)`` metres: the model's reconstruction at the hidden patches, the truth at the
    #: visible ones — MAE's own figure convention. The model predicts every patch, but a
    #: prediction where the encoder could see the answer is not evidence of anything, and a
    #: panel made entirely of predictions hides *where* the model was actually tested.
    predicted: np.ndarray
    #: ``(H, W)`` True where the patch was hidden from the encoder.
    hidden: np.ndarray
    #: ``(H, W)`` True on land — ADR-0003 §4's mask, derived from the stored int16.
    land: np.ndarray

    @property
    def relief_m(self) -> float:
        """The truth's land relief, in metres. What the grey ramp deliberately does not say."""
        values = self.truth[self.land & np.isfinite(self.truth)]
        return float(values.max() - values.min()) if values.size else 0.0

    @property
    def nonfinite_frac(self) -> float:
        """Share of the predicted field with no usable number in it."""
        return float((~np.isfinite(self.predicted)).mean())


# --- Inverting a prediction ----------------------------------------------------


def predicted_in_tile_units(
    pred: Tensor, target: Tensor, weights: Tensor | None, *, norm_pix: bool, var_floor: float
) -> Tensor:
    """Undo §4's `norm_pix`: patch predictions back into the units the target is in.

    The exact inverse of :func:`pretrain.loss.normalized_targets`, including its
    :func:`~torch.Tensor.clamp_min` floor and the **weights the loss itself used** — under §4's
    ocean weighting a patch's statistics are taken over its valid pixels only, and recomputing
    them unweighted would place every coastal patch on a level that includes the sea.

    With ``norm_pix`` off the decoder already predicts in the target's units and there is
    nothing to undo, which is R3's (#84) case and the reason this reads the flag.
    """
    if not norm_pix:
        return pred.float()
    mean, var = patch_statistics(target, weights=weights)
    return pred.float() * torch.sqrt(var.clamp_min(var_floor)) + mean


def predict_the_target(
    pred: Tensor, target: Tensor, weights: Tensor | None, *, norm_pix: bool, var_floor: float
) -> Tensor:
    """A stand-in "model" that predicts its own target exactly — the inversion's one oracle.

    Passed to :meth:`ReconGrids.reconstruct` as ``predict``, it replaces the decoder's output
    with what a perfect model would have emitted, under whichever units §4 is asking for. That
    is the only case where the whole two-step inversion has a **known closed-form answer** —
    the predicted panel must come back as the truth panel, to the metre — which makes it both
    §15.6's synthetic pre-flight and the sharp test of a chain whose every wrong version still
    renders a perfectly plausible landscape.

    The forward pass still happens: the mask is the model's, and only the prediction is
    replaced, so what is exercised is the real composite/unpatchify/de-normalize path.
    """
    if not norm_pix:
        return target.float()
    return normalized_targets(target, var_floor=var_floor, weights=weights)


# --- Rendering -----------------------------------------------------------------


def to_uint8(rgb: np.ndarray) -> np.ndarray:
    """``[0, 1]`` floats to the bytes an image is. Clipped, so nothing can leave the range."""
    return (np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0) * 255).round().astype(np.uint8)


def metre_ramp(tile: TileReconstruction) -> tuple[float, float]:
    """The elevation range the grey ramp spans — from the **truth's land**, and nothing else.

    Two exclusions, both load-bearing. **The prediction is not consulted**, so the picture does
    not rescale itself the epoch the model moves — which is the epoch a reader is looking for a
    change. **Ocean is not consulted**, because ADR-0001 §3's exact-zero fill would drag the low
    end of a coastal tile's ramp to sea level and render its actual terrain in the top few grey
    levels.
    """
    values = tile.truth[tile.land & np.isfinite(tile.truth)]
    if values.size == 0:
        return 0.0, _MIN_RAMP_M
    lo, hi = (float(v) for v in np.percentile(values, RAMP_PERCENTILES))
    if hi - lo < _MIN_RAMP_M:
        middle = 0.5 * (lo + hi)
        return middle - _MIN_RAMP_M / 2, middle + _MIN_RAMP_M / 2
    return lo, hi


def _fill(z: np.ndarray, land: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(field with ocean and non-finite values replaced by the land median, the bad mask)``.

    Filling before shading rather than after is :func:`dataset.gallery.shade_tile`'s own
    argument: an unfilled coastline is shaded as a several-hundred-metre cliff whose glare
    swamps the land beside it, which on a coastal tile is the terrain under review.
    """
    z = np.asarray(z, dtype=np.float64)
    bad = ~np.isfinite(z)
    usable = z[land & ~bad]
    filled = z.copy()
    filled[bad | ~land] = float(np.median(usable)) if usable.size else 0.0
    return filled, bad


def vertical_exaggeration(tile: TileReconstruction, *, sigma: float = SMOOTH_SIGMA_PX) -> float:
    """The exaggeration all three panels are shaded at, measured on the **truth**.

    The corpus spans ~3 m to ~613 m of land-std by design, so no single factor renders both
    ends (:func:`dataset.gallery.auto_vert_exag`). Measuring it per panel instead would make
    each panel legible on its own and the *comparison* meaningless: a prediction with half the
    relief would be exaggerated twice as hard and come back looking identical to the truth.
    """
    filled, _ = _fill(tile.truth, tile.land)
    filled = smooth(filled, sigma) if sigma > 0 else filled
    land_std = float(filled[tile.land].std()) if tile.land.any() else 1.0
    return auto_vert_exag(land_std)


def _paint(
    grey: np.ndarray, *, land: np.ndarray, bad: np.ndarray, hidden: np.ndarray | None
) -> np.ndarray:
    """One intensity field in ``[0, 1]`` to RGB, with the three special populations painted."""
    rgb = np.repeat(np.clip(grey, 0.0, 1.0)[..., None], 3, axis=2)
    rgb[~land] = np.asarray(OCEAN_RGB)
    rgb[bad] = np.asarray(BAD_RGB)
    if hidden is not None:
        rgb[hidden] = np.asarray(MASK_RGB)
    return rgb


def heat(
    z: np.ndarray,
    *,
    ramp: tuple[float, float],
    land: np.ndarray,
    hidden: np.ndarray | None = None,
) -> np.ndarray:
    """Elevation on a grey ramp: ``(H, W, 3)`` in ``[0, 1]``.

    Grey rather than a hypsometric colour ramp, for :mod:`dataset.gallery`'s reason: a colour
    ramp manufactures drama on a low-relief tile, and half this panel's strata are low-relief
    by construction. The magnitudes ride in the caption instead.
    """
    lo, hi = ramp
    z = np.asarray(z, dtype=np.float64)
    bad = ~np.isfinite(z)
    scaled = np.where(bad, 0.0, (z - lo) / (hi - lo))
    return _paint(scaled, land=land, bad=bad, hidden=hidden)


def shade(
    z: np.ndarray,
    *,
    land: np.ndarray,
    lat: float,
    vert_exag: float,
    hidden: np.ndarray | None = None,
    sigma: float = SMOOTH_SIGMA_PX,
) -> np.ndarray:
    """Shaded relief: ``(H, W, 3)`` in ``[0, 1]``, unstretched, at a caller-fixed exaggeration.

    ``sigma`` is the gallery's one-pixel low-pass, and it is applied here for the same reason:
    shading is a derivative, and the stored int16's 1 m quantum becomes visible stipple at the
    exaggeration a low-relief tile needs. It is applied to all three panels equally, so it
    cannot flatter one of them.
    """
    filled, bad = _fill(z, land)
    filled = smooth(filled, sigma) if sigma > 0 else filled
    dy, dx = band_pixel_spacing_m(lat)
    intensity = hillshade(filled, dy=dy, dx=dx, vert_exag=vert_exag, stretch=False)
    return _paint(intensity, land=land, bad=bad, hidden=hidden)


def triptych(panels: Sequence[np.ndarray]) -> np.ndarray:
    """Lay panels side by side with a separator between them, left to right."""
    height = panels[0].shape[0]
    gap = np.tile(np.asarray(SEPARATOR_RGB), (height, SEPARATOR_PX, 1))
    out: list[np.ndarray] = []
    for i, panel in enumerate(panels):
        if i:
            out.append(gap)
        out.append(panel)
    return np.concatenate(out, axis=1)


def render_tile(tile: TileReconstruction) -> tuple[Image, Image]:
    """One tile as its two triptychs: ``(heatmap, hillshade)``, masked → predicted → truth.

    The ramp and the exaggeration are computed once, from the truth, and shared by all six
    panels — see :func:`metre_ramp` and :func:`vertical_exaggeration` for why that is the whole
    design rather than a detail.
    """
    ramp, ve = metre_ramp(tile), vertical_exaggeration(tile)
    where = f"{tile.tile_id} · stratum {tile.stratum_id} · masked | predicted | truth"
    magnitudes = (
        f"relief {tile.relief_m:,.0f} m · ramp {ramp[0]:,.0f}-{ramp[1]:,.0f} m · land "
        f"{tile.land.mean():.0%}"
    )
    heats = [
        heat(tile.truth, ramp=ramp, land=tile.land, hidden=tile.hidden),
        heat(tile.predicted, ramp=ramp, land=tile.land),
        heat(tile.truth, ramp=ramp, land=tile.land),
    ]
    shades = [
        shade(tile.truth, land=tile.land, lat=tile.lat, vert_exag=ve, hidden=tile.hidden),
        shade(tile.predicted, land=tile.land, lat=tile.lat, vert_exag=ve),
        shade(tile.truth, land=tile.land, lat=tile.lat, vert_exag=ve),
    ]
    return (
        Image(to_uint8(triptych(heats)), caption=f"{where} · {magnitudes}"),
        Image(
            to_uint8(triptych(shades)),
            caption=f"{where} · hillshade x{ve:.3g}, unstretched · {magnitudes}",
        ),
    )


# --- The panel, rendered every N epochs ----------------------------------------

#: What :meth:`ReconGrids.render` puts in the sink. Named so the two consumers of a run's page
#: — a human scrolling it and a test asserting it — read the same keys.
HEATMAP_KEY: Final = "recon/heatmap"
HILLSHADE_KEY: Final = "recon/hillshade"


class ReconGrids:
    """The committed panel, its pixels, and the render §10.2 asks for every 10 epochs.

    Built by :meth:`from_config`, driven by :class:`pretrain.monitor.EpochMonitor`. It holds
    the panel's samples **in memory for the life of the run** — 40 tiles is 8 MB, against 40
    random reads into a 47 GB corpus every tenth epoch — which also means the pixels going into
    epoch 400's grid are byte-identical to the ones that went into epoch 0's, rather than
    identical because nothing happened to disturb them.
    """

    def __init__(
        self,
        *,
        samples: list[dict],
        panel: ResolvedPanel,
        normalizer: Normalizer,
        mask_seed: int,
        cadence: int,
        chunk: int,
        panel_md5: str | None = None,
    ) -> None:
        if len(samples) != len(panel):
            raise ValueError(f"{len(samples)} samples for a panel of {len(panel)} tiles")
        if chunk < 1:
            raise ValueError(f"a chunk of {chunk} tiles renders nothing")
        self.samples = samples
        self.panel = panel
        #: The loader's own :class:`~dataset.loader.Normalizer` **instance**, not a rebuilt one:
        #: the inverse has to be the inverse of what actually happened to these pixels.
        self.normalizer = normalizer
        self.mask_seed = mask_seed
        self.cadence = cadence
        self.chunk = chunk
        #: The committed panel file's md5, recorded in the run's summary. `resolve_panel`
        #: proves a panel describes *this* corpus; only the digest proves that two runs whose
        #: grids are being compared rendered **the same 40 tiles**, which is the property
        #: §10.2 rests on and the one nothing inside a single run can check. Same argument as
        #: the selection-list and split md5s :mod:`pretrain.fingerprint` records — and it is
        #: recorded rather than *verified* against an expected value, because the panel is
        #: drawn per tier and no single digest is right for every corpus.
        self.panel_md5 = panel_md5

    @classmethod
    def from_config(cls, cfg: RunConfig) -> ReconGrids | None:
        """The grids a run's config asks for, or ``None`` when it asks for none.

        Two ways to get ``None``, and neither is an error: the cadence is 0 (§10.2 switched
        off), or **no panel has been committed at the configured path**. The second is a
        pre-flight failure — §15's checklist item 3 — and not a reason to refuse to train: the
        run still reports every scalar §10 asks for, and a missing panel is recorded rather
        than raised, the way :mod:`pretrain.fingerprint` records an absent split.

        A panel that *is* there and no longer describes this corpus does raise, here at
        construction rather than 10 epochs into a rented hour. See
        :func:`pretrain.panel.resolve_panel` for the four ways that happens.
        """
        if cfg.monitoring.recon_every_epochs == 0:
            return None
        path = Path(cfg.data.val_panel_path)
        if not path.exists():
            return None

        from dataset.materialize import file_md5

        from .data import SplitDataset

        corpus_dir = Path(cfg.data.corpus_dir)
        panel = resolve_from_manifest(read_panel(path), corpus_dir / "manifest.parquet")
        dataset = SplitDataset(
            corpus_dir,
            cfg.data.val_split,
            blocks_path=cfg.data.split_blocks_path,
            assignment_path=cfg.data.split_assignment_path,
            normalization=cfg.data.normalization,
        )
        # The panel's rows are manifest rows; a SplitDataset is indexed within its own split.
        position = {int(row): i for i, row in enumerate(dataset.indices)}
        samples = [dataset[position[int(row)]] for row in panel.rows]
        return cls(
            samples=samples,
            panel=panel,
            normalizer=dataset.dataset.normalizer,
            mask_seed=cfg.monitoring.val_mask_seed,
            cadence=cfg.monitoring.recon_every_epochs,
            chunk=cfg.optimization.batch_size,
            panel_md5=file_md5(path),
        )

    def due(self, epoch: int) -> bool:
        """Is a grid due at this epoch?

        Epoch 0 — the **first** boundary, one epoch in — is, deliberately: it is the baseline
        every later grid is read against, and a cadence that first fired at epoch 10 would
        leave the first tenth of a run with nothing to compare to. (What §15's checklist item 5
        asks for is different and stricter: that the whole pipeline renders on a *randomly
        initialized* model, which is :meth:`render` called before any step at all —
        ``python -m pretrain.recon`` and `test_the_whole_pipeline_runs_on_a_randomly_
        initialized_model` are where that is exercised.)
        """
        return self.cadence > 0 and epoch % self.cadence == 0

    def _batches(self) -> Iterator[tuple[int, dict]]:
        """The panel, collated into batches of at most :attr:`chunk`, in panel order."""
        from torch.utils.data import default_collate

        for start in range(0, len(self.samples), self.chunk):
            yield start, default_collate(self.samples[start : start + self.chunk])

    def reconstruct(
        self, trainer: Trainer, *, predict: Callable[..., Tensor] | None = None
    ) -> list[TileReconstruction]:
        """Score the panel and invert it to metres: one :class:`TileReconstruction` per tile.

        Three properties it shares with §10.1's val pass, for the same reasons: the model is
        put in ``eval`` and restored, the whole thing runs under :func:`torch.no_grad`, and the
        masks are drawn from an **explicit** generator seeded with the committed
        :data:`~pretrain.config.VAL_MASK_SEED`. That last one is what makes two epochs' grids
        comparable at all — the same patches are hidden every time, so a difference between two
        grids is the model and never the draw — and it also keeps the render from consuming
        from the stream the next training step draws its masks from.
        """
        was_training = trainer.model.training
        trainer.model.eval()
        generator = torch.Generator(device=trainer.device).manual_seed(self.mask_seed)
        out: list[TileReconstruction] = []
        try:
            with torch.no_grad():
                for start, batch in self._batches():
                    out += self._reconstruct(trainer, batch, start, generator, predict)
        finally:
            trainer.model.train(was_training)
        return out

    def _reconstruct(
        self,
        trainer: Trainer,
        batch: dict,
        start: int,
        generator: torch.Generator,
        predict: Callable[..., Tensor] | None,
    ) -> list[TileReconstruction]:
        """One collated chunk, from the model's output to fields of metres.

        Everything comes off **one** :meth:`Trainer.score <pretrain.train.Trainer.score>` call:
        the prediction and the mask that produced it, the target the loss scored against, and
        the pixel weights that target's per-patch statistics were taken under. Running a second
        forward here to get the mask would draw a different one, and the picture would then be
        of a different masking of the tile than every number logged beside it.
        """
        model, loss_fn = trainer.model, trainer.loss_fn
        scored = trainer.score(batch, generator=generator)
        kwargs = {"norm_pix": loss_fn.norm_pix_loss, "var_floor": loss_fn.var_floor}
        args = (scored.model_out.pred, scored.target, scored.pixel_weights)

        chosen = args[0] if predict is None else predict(*args, **kwargs)
        units = predicted_in_tile_units(chosen, *args[1:], **kwargs)
        hidden = scored.model_out.mask.unsqueeze(-1).to(torch.bool)
        target = scored.target.float()

        fields = model.unpatchify(torch.where(hidden, units, target)).squeeze(1)
        truths = model.unpatchify(target).squeeze(1)
        hidden_px = model.unpatchify(hidden.expand(-1, -1, model.patch_dim)).squeeze(1)
        land = batch["land_mask"].cpu().numpy().astype(bool)
        mean, std = batch["mean"].cpu().numpy(), batch["std"].cpu().numpy()

        out = []
        for i, tile in enumerate(self.panel.tiles[start : start + len(truths)]):

            def to_metres(field: Tensor, i: int = i) -> np.ndarray:
                return self.normalizer.invert(
                    field[i].to(torch.float64).cpu().numpy(),
                    mean=float(mean[i]),
                    std=float(std[i]),
                )

            out.append(
                TileReconstruction(
                    tile_id=tile.tile_id,
                    stratum_id=tile.stratum_id,
                    lat=float(self.panel.lat[start + i]),
                    truth=to_metres(truths),
                    predicted=to_metres(fields),
                    hidden=hidden_px[i].cpu().numpy().astype(bool),
                    land=land[i],
                )
            )
        return out

    def render(self, trainer: Trainer) -> dict[str, MetricValue]:
        """The whole panel as one sink payload: two triptych galleries and their counts.

        Logged at ``step=epoch`` by the caller, under keys that do not move between runs, so
        W&B's own image-history slider walks epoch 0's grid to epoch 400's on one panel.
        """
        tiles = self.reconstruct(trainer)
        rendered = [render_tile(t) for t in tiles]
        return {
            HEATMAP_KEY: [heat for heat, _ in rendered],
            HILLSHADE_KEY: [shaded for _, shaded in rendered],
            "recon/tiles": len(tiles),
            "recon/strata": len(self.panel.strata),
            # Zero on any healthy run. Reported rather than left in the pictures because a
            # painted patch is easy to miss across 40 tiles, and a rising number is a model
            # that has started producing NaNs — which is a much louder problem than a grid.
            "recon/nonfinite_pred_frac": (
                float(np.mean([t.nonfinite_frac for t in tiles])) if tiles else 0.0
            ),
        }


# --- The entrypoint -------------------------------------------------------------


def contact_sheet(images: Sequence[Image], *, columns: int = 2) -> np.ndarray:
    """The panel's triptychs tiled into one array, in panel order — i.e. by stratum.

    A last row with a gap in it is **padded**, never cropped to fit. Cropping every row to the
    shortest one is the version of this that silently throws away half the sheet whenever the
    tile count is not a multiple of ``columns`` — which the panel's own count never is at 40
    and 2, and which ``--per-stratum 3`` makes routine.
    """
    rows = [
        np.concatenate([i.data for i in images[start : start + columns]], axis=1)
        for start in range(0, len(images), columns)
    ]
    width = max(row.shape[1] for row in rows)
    return np.concatenate(
        [
            np.pad(row, ((0, 0), (0, width - row.shape[1]), (0, 0)), constant_values=255)
            for row in rows
        ],
        axis=0,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m pretrain.recon --corpus …`` — §15's checklist item 5, as a command.

    *"Recon grid + hillshade render pipeline works on the untrained model (garbage in, garbage
    rendered)."* That is a check somebody walks through **before renting a GPU**, so it is a
    command rather than a paragraph: it builds the model the config asks for, initializes it
    and trains it for exactly zero steps, renders the committed panel, and writes the two
    contact sheets somewhere they can be looked at.

    Nothing here is a substitute for the test suite; what it adds is the half a person has to
    do, which is *look at the pictures*. A grid can pass every assertion in
    `test_pretrain_recon.py` and still be shaded from the south-east.
    """
    import argparse

    from .fingerprint import fingerprint_for
    from .header import add_config_arguments, config_from_args, format_header
    from .train import Trainer, resolve_device

    p = add_config_arguments(
        argparse.ArgumentParser(
            prog="python -m pretrain.recon",
            description="Render §10.2's panel on an untrained model — §15's item 5 (#98).",
        )
    )
    p.add_argument("--out", type=Path, default=Path("."), help="where to write the two sheets")
    p.add_argument("--device", default=None, help="cuda | cpu | … (default: cuda if present)")
    args = p.parse_args(argv)

    cfg = config_from_args(args)
    print(format_header(cfg, fingerprint_for(cfg)))
    grids = ReconGrids.from_config(cfg)
    if grids is None:
        print(
            f"\nNo panel to render: {cfg.data.val_panel_path} is absent, or the cadence is 0. "
            f"Draw one with `python -m pretrain.panel`."
        )
        return 1

    # Zero steps taken, so this is the randomly initialized model §15 asks about — plus §8's
    # warm start if the config selects one, which is the other thing worth looking at once.
    trainer = Trainer.from_config(cfg, steps_per_epoch=1, device=resolve_device(args.device))
    values = grids.render(trainer)
    print(
        f"\n{values['recon/tiles']} tiles over {values['recon/strata']} strata; "
        f"non-finite predictions {values['recon/nonfinite_pred_frac']:.3g}"
    )

    from PIL import Image as PilImage  # dev-side only: the sheets are for a human to look at

    args.out.mkdir(parents=True, exist_ok=True)
    for key, name in ((HEATMAP_KEY, "heatmap"), (HILLSHADE_KEY, "hillshade")):
        path = args.out / f"recon_panel_{name}.png"
        PilImage.fromarray(contact_sheet(values[key])).save(path)
        print(f"  {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
