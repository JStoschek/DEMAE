"""§4.1's shared conv head — the one A4, A5 and A6 differ only in the width of (#159).

LayerNorm on patch tokens → 1x1 conv to 256 → four stages of [3x3 conv, GroupNorm, GELU,
2x bilinear upsample] at (256, 128, 64, 32), the full-resolution void mask concatenated at each
stage's input, and a **1-channel residual added to the A1 fill**.

## Why the output is a residual and not a fill

§2.4 makes A1's classical interpolation the base every learned arm improves on, and this module
is where that stops being prose. The head predicts a *correction* to the A1 fill inside the
void, so a head that learns nothing scores **exactly** A1 — not approximately, not
within-noise-of, exactly, because :func:`apply_residual` adds zero to the same float32 array
`benchmark.evaluate` would have scored for A1. Two things follow that are worth having:

* the table stays legible. Every learned row is read against A1's, and a row that came out
  *worse* than A1 is then a real finding about the representation rather than an artefact of a
  decoder that had to relearn interpolation from scratch;
* the failure mode where a head silently produces garbage is bounded below by A1 rather than by
  zero metres of elevation, which is the difference between a bad number and an absurd one.

:func:`build` zero-initializes the output convolution for the same reason one level down: at
step 0 the head is A1 *by construction*, so the first thing any training curve shows is A1's own
loss, and a run whose loss starts somewhere else has a bug this module can be asked about.

## The two parameter counts, asserted rather than quoted

**1,082,017 at 384-d and 1,181,089 at 768-d** — 1.08M and 1.18M. Plan §4.1 says "~3M" and is
wrong; §7 says so, and the spec's instruction is to print the corrected figures. So they are
computed twice here and checked against each other: :func:`parameter_count` derives the number
in closed form from the recipe, and :func:`build` counts the module it just built and refuses a
disagreement. A count that only appeared as a literal would go stale the first time a stage
dimension moved, and §6 item 4 *prints* these numbers — a stale one prints as a measurement.

The whole delta between the two is the first projection: 256 x (768 - 384) weights, 98,304 of
them. Everything after the 1x1 conv is byte-identical across A4, A5 and A6, which is what makes
C1 a claim about the representation and not about the decoder attached to it.

## The mask, at four resolutions

"The full-resolution void mask concatenated at each stage's input" — and the stages run at 14,
28, 56 and 112 px, so three of the four concatenations are of a *downsampled* mask. It is
**area**-pooled (:func:`_mask_at`), not sampled: a 16 px void is 0.5% of a tile and nearest-
neighbour sampling onto the 14x14 token grid would show it to the first stage as either a whole
token or nothing at all. Area pooling hands each stage the fraction of its cell the void covers,
which is the quantity the stage can actually act on.

## Not a way to produce a result

Like :mod:`benchmark.train`, and for ADR-0009's reason. This module builds a module and computes
a loss; `python -m benchmark.launch run` is the only thing that scores anything.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from dataset import config as dataset_config

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .config import Head

#: GroupNorm's group count. 32 divides every one of §4.1's four stage dimensions, and the
#: normalization is over channels rather than over the batch on purpose: §4.1's batch is 128
#: tiles but the *last* batch of an epoch is whatever is left, and BatchNorm would make a head's
#: eval-time behaviour a function of how the 20,000 tiles happened to divide.
#:
#: It contributes nothing to the parameter count either way — GroupNorm learns 2C affine
#: parameters whatever the grouping — so this is a choice about the statistic and not about the
#: size of the head.
GROUPS: Final = 32

#: What the head's output is added to, and therefore what it is measured in. Metres, because
#: §4.1's loss is masked L1 **in metres** and :mod:`benchmark.metrics` scores in metres. Nothing
#: in this module rescales it: the encoder's diet is applied on the way *in* (§2.1,
#: :func:`benchmark.extract.normalize`) and never inverted, because the head never sees the
#: encoder's units — it sees tokens.
OUTPUT_UNITS: Final = "metres"

#: §4.1's two widths and what the head costs at each — the corrected figures §7 asks to print.
#: Checked against :func:`parameter_count` by a test rather than trusted, so this dict is a
#: *record* of the two numbers rather than a second definition of them.
PARAMETERS: Final[dict[int, int]] = {384: 1_082_017, 768: 1_181_089}

#: The plan's own figure for this head, kept so the correction has something to be a correction
#: *to*. §4.1 says "~3M"; §7 records that as wrong, and this module is where the real number is
#: produced. Quoted in :func:`format_parameters` so a build session reading the console sees the
#: amendment rather than a bare number that looks like it always said that.
PLAN_QUOTED_PARAMETERS: Final = "~3M (plan §4.1, corrected by §7)"


class HeadError(ValueError):
    """A head that cannot be built from this recipe, or cannot be handed these tokens."""


# --- The arithmetic, before any torch ----------------------------------------


def token_grid(tokens: int) -> int:
    """The square token grid ``tokens`` patch tokens form — 14 at ADR-0012's 196.

    Derived rather than declared, so a contract that moved to another patch count moves this
    head with it instead of silently reshaping 196 tokens into a grid they do not fill.
    """
    grid = math.isqrt(tokens)
    if grid * grid != tokens:
        raise HeadError(
            f"{tokens} patch tokens do not form a square grid, and §4.1's head upsamples a "
            f"square one to the tile. ADR-0012 §5's canonical layout is (N, 196, D)"
        )
    return grid


def parameter_count(width: int, recipe: Head | None = None) -> int:
    """§4.1's head at ``width``, counted in closed form — no torch, no allocation.

    Here rather than only as ``sum(p.numel() ...)`` because the count is a **claim** this ticket
    makes about the architecture, and a claim you can only check by building a 1.18M-parameter
    module on a box with torch is one the CPU-only half of this package cannot check at all.
    :func:`build` asserts the two agree, so the closed form is not a second definition that can
    drift — it is the definition, and the module is checked against it.
    """
    from .config import Head as HeadRecipe

    recipe = HeadRecipe() if recipe is None else recipe
    # LayerNorm over the token width: one scale and one shift per channel.
    total = 2 * width
    # The 1x1 projection to 256. The only term that moves with the encoder.
    total += recipe.proj_dim * width + recipe.proj_dim
    channels = recipe.proj_dim
    for out in recipe.stage_dims:
        # 3x3 conv over the previous stage's channels **plus the mask**, then GroupNorm's 2C.
        total += 9 * (channels + 1) * out + out
        total += 2 * out
        channels = out
    # The 1-channel residual: a 1x1 projection off the last stage.
    return total + channels + 1


def format_parameters(width: int, recipe: Head | None = None) -> str:
    """The corrected figure, with the figure it corrects, for the console and the run record."""
    count = parameter_count(width, recipe)
    return f"{count:,} ({count / 1e6:.2f}M) at {width}-d — plan quoted {PLAN_QUOTED_PARAMETERS}"


# --- The module ---------------------------------------------------------------


def _stage_channels(recipe: Head) -> list[tuple[int, int]]:
    """``(in, out)`` per stage, with the mask channel already counted into ``in``."""
    channels = [recipe.proj_dim, *recipe.stage_dims[:-1]]
    return [(c + 1, out) for c, out in zip(channels, recipe.stage_dims)]


def _mask_at(void, size: tuple[int, int]):
    """The void mask at one stage's resolution — see the module docstring on area pooling."""
    import torch.nn.functional as F

    if tuple(void.shape[-2:]) == size:
        return void
    return F.adaptive_avg_pool2d(void, size)


def _module():
    """`torch.nn`, imported at call time.

    :mod:`benchmark` is torch-optional — `freeze`, the scoring path and A1 all run on a box with
    no CUDA stack (§3.4) — so the import lives inside the two functions that need it rather than
    at module scope, exactly as :mod:`dataset.loader` keeps its own.
    """
    import torch.nn as nn

    return nn


def _build_class():
    """The head, defined against a torch imported at call time.

    A closure rather than a module-level ``class FillHead(nn.Module)`` for the reason above: a
    class statement subclassing ``nn.Module`` would import torch when this module is imported,
    and A1's scoring path imports this module transitively through
    :mod:`benchmark.evaluate`'s filler table.
    """
    import torch
    import torch.nn.functional as F

    nn = _module()

    class FillHead(nn.Module):
        """§4.1's head. ``forward(tokens, void) -> residual in metres``.

        ``tokens`` is ADR-0012 §5's canonical ``(N, 196, D)`` — this encoder's own final tensor,
        whatever "final" means for it (:attr:`benchmark.arms.Encoder.output_is_normed`). ``void``
        is ``(N, 1, 224, 224)`` float, 1 inside the void.

        The returned residual is the whole tile, not just the void. Masking it to the void is
        :func:`apply_residual`'s job and is stated there, because *where a prediction is allowed
        to land* is a fact about §2.4's convention rather than about this architecture.
        """

        def __init__(self, width: int, recipe: Head, *, tokens: int, tile_px: int) -> None:
            super().__init__()
            self.width = width
            self.recipe = recipe
            self.tokens = tokens
            self.grid = token_grid(tokens)
            self.tile_px = tile_px
            self.norm = nn.LayerNorm(width)
            self.project = nn.Conv2d(width, recipe.proj_dim, kernel_size=1)
            self.convs = nn.ModuleList(
                nn.Conv2d(into, out, kernel_size=3, padding=1)
                for into, out in _stage_channels(recipe)
            )
            self.norms = nn.ModuleList(
                nn.GroupNorm(GROUPS, out) for out in recipe.stage_dims
            )
            self.residual = nn.Conv2d(recipe.stage_dims[-1], 1, kernel_size=1)

        def forward(self, tokens, void):
            if tokens.ndim != 3 or tokens.shape[1] != self.tokens:
                raise HeadError(
                    f"§4.1's head reads ADR-0012 §5's (N, {self.tokens}, D) patch tokens; got "
                    f"{tuple(tokens.shape)}"
                )
            if tokens.shape[2] != self.width:
                raise HeadError(
                    f"this head was built at {self.width}-d and was handed {tokens.shape[2]}-d "
                    f"tokens. A4, A5 and A6 share one head differing **only** in this width "
                    f"(§4.1), so it is checked rather than absorbed by a reshape"
                )
            if void.shape[-2:] != (self.tile_px, self.tile_px):
                raise HeadError(
                    f"the void mask reaches the head at full resolution "
                    f"({self.tile_px}x{self.tile_px}); got {tuple(void.shape)}"
                )
            void = void.to(tokens.dtype)

            x = self.norm(tokens)
            # (N, T, D) -> (N, D, grid, grid), row-major, which is ADR-0012 §5's declared order.
            x = x.transpose(1, 2).reshape(x.shape[0], self.width, self.grid, self.grid)
            x = self.project(x)
            for conv, norm in zip(self.convs, self.norms):
                x = torch.cat([x, _mask_at(void, tuple(x.shape[-2:]))], dim=1)
                x = F.gelu(norm(conv(x)))
                x = F.interpolate(
                    x, scale_factor=2, mode="bilinear", align_corners=False
                )
            return self.residual(x)

    return FillHead


@dataclass(frozen=True)
class Built:
    """A head and the two facts §6 item 4 prints about it."""

    module: Any
    width: int
    parameters: int

    @property
    def line(self) -> str:
        return format_parameters(self.width)


def build(
    width: int,
    recipe: Head | None = None,
    *,
    seed: int = 0,
    tokens: int = dataset_config.TOKENS_PER_TILE,
    tile_px: int = dataset_config.TILE_PX,
) -> Built:
    """The one head A4, A5 and A6 instantiate — same code, same recipe, different ``width``.

    Three things happen here that are not "call the constructor", and each is an acceptance
    criterion of #159:

    1. **The upsampling has to land on the tile.** Four 2x stages from a 14x14 grid reach 224,
       and §4.1's four stage dimensions are what makes that true. A recipe with three stages
       would return an 112 px residual that broadcasts against a 224 px fill in some places and
       raises in others, so it is refused here.
    2. **The parameter count is asserted.** :func:`parameter_count` derived it before anything
       was allocated; this counts the module and refuses a disagreement. §4.1's "~3M" is the
       reason the assertion exists rather than a comment.
    3. **The output convolution is zeroed.** So the head *is* A1 at step 0 — see the module
       docstring. Everything else is torch's own default init, seeded here so that
       :attr:`benchmark.config.BenchmarkConfig.seed` is what separates this cell's three seeds
       and nothing else is.
    """
    import torch

    from .config import Head as HeadRecipe

    recipe = HeadRecipe() if recipe is None else recipe
    grid = token_grid(tokens)
    reached = grid * 2 ** len(recipe.stage_dims)
    if reached != tile_px:
        raise HeadError(
            f"§4.1's head upsamples a {grid}x{grid} token grid by 2x per stage; "
            f"{len(recipe.stage_dims)} stages reach {reached} px and the tile is {tile_px}. The "
            f"residual is added to the A1 fill pixel for pixel (§2.4), so a head that does not "
            f"land on the tile is not a head this convention can use"
        )
    for out in recipe.stage_dims:
        if out % GROUPS:
            raise HeadError(
                f"stage dimension {out} is not divisible by GroupNorm's {GROUPS} groups "
                f"(§4.1's four are {recipe.stage_dims})"
            )

    generator = torch.Generator().manual_seed(int(seed))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item()))
        module = _build_class()(width, recipe, tokens=tokens, tile_px=tile_px)

    # §2.4's residual, made exact: zero weights **and** zero bias, so the first forward returns
    # a tensor of zeros and `apply_residual` returns the A1 fill unchanged.
    torch.nn.init.zeros_(module.residual.weight)
    torch.nn.init.zeros_(module.residual.bias)

    counted = sum(p.numel() for p in module.parameters())
    expected = parameter_count(width, recipe)
    if counted != expected:
        raise HeadError(
            f"§4.1's head at {width}-d built {counted:,} parameters and this recipe derives "
            f"{expected:,}. The two are computed from the same four stage dimensions by "
            f"different means precisely so that they can disagree — §6 item 4 prints this "
            f"number, and a printed count nothing checked is a quoted one"
        )
    return Built(module=module, width=width, parameters=counted)


# --- §2.4's residual convention -----------------------------------------------


def apply_residual(a1_fill: np.ndarray, residual: np.ndarray, void: np.ndarray) -> np.ndarray:
    """§2.4's rule: the A1 fill, plus the head's correction, **inside the void only**.

    Masking to the void is not a refinement of the architecture — it is what the benchmark
    scores. Outside the void the tile is measured ground that every arm is handed and none is
    asked to predict, and :func:`benchmark.metrics.score` refuses a fill that moved it
    (``_check_scored_in_metres``). So a head whose residual leaked one metre onto untouched
    ground would fail the score rather than quietly shift a column, and the masking here is what
    keeps that from being the way anyone finds out.

    The arithmetic is float32 over the float32 fill :mod:`benchmark.fill` produced, so ``residual
    == 0`` returns an array that is **equal to** the A1 fill rather than close to it. That
    exactness is the acceptance criterion — *a zero-output head scores exactly A1* — and it is
    why this adds a masked residual rather than blending two predictions.
    """
    a1_fill = np.asarray(a1_fill)
    residual = np.asarray(residual, dtype=a1_fill.dtype).reshape(a1_fill.shape)
    return np.where(np.asarray(void, dtype=bool), a1_fill + residual, a1_fill)


# --- §4.1's loss --------------------------------------------------------------

#: Sobel's 3x3 kernel, divided by 8 so the result is a gradient per pixel rather than eight
#: times one. The scale matters here and does not usually: the slope term is weighted against an
#: elevation term in metres (:attr:`benchmark.config.Head.slope_weight` is 0.1), so an eight-fold
#: kernel would silently be a 0.8 weight.
SOBEL_SCALE: Final = 8.0


def sobel_gradients(x):
    """``(d/dy, d/dx)`` of a ``(N, 1, H, W)`` elevation raster, by 3x3 Sobel. Same shape out.

    **Per pixel, not per ground metre.** §4.1 specifies Sobel gradients and this is a consistency
    term between a prediction and its own truth on the same tile, so the tile's ground spacing is
    a common factor on both sides of every difference. Dividing by it would not change what the
    term penalizes on any one tile — it would reweight *tiles against each other* by latitude,
    which is a curriculum nothing in §4.1 asks for. The scored slope metrics are a different
    quantity and do carry the anisotropic spacing (:func:`benchmark.metrics.zt_surface`); the two
    are deliberately not the same estimator and this is the sentence that says so.

    Replicate padding, so the tile's own border does not read as a cliff down to zero metres.
    """
    import torch
    import torch.nn.functional as F

    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=x.dtype,
        device=x.device,
    ) / SOBEL_SCALE
    kernel = torch.stack([kx.t(), kx]).unsqueeze(1)  # (2, 1, 3, 3): d/dy then d/dx
    padded = F.pad(x, (1, 1, 1, 1), mode="replicate")
    both = F.conv2d(padded, kernel)
    return both[:, 0:1], both[:, 1:2]


@dataclass(frozen=True)
class Loss:
    """§4.1's two terms and their weighted sum, kept apart so a run can log both.

    Separately because they answer different questions about a failing head: an elevation term
    that falls while the slope term rises is the flat-patch cheating §4.1 added the second term
    to discourage, and a single scalar cannot show it.
    """

    #: The scalar ``backward()`` is called on — **the only one still on the graph**.
    total: Any
    elevation: Any
    slope: Any

    def detached(self) -> dict[str, float]:
        """The three numbers, as floats, for a log line.

        ``total`` is detached on the way out because it is the tensor the optimizer steps on: a
        bare ``float()`` of it is a graph-attached read, which torch warns about and which is
        the sort of thing that quietly holds an epoch's activations alive.
        """
        import torch

        total = self.total.detach() if torch.is_tensor(self.total) else self.total
        return {
            "loss": float(total),
            "l1_m": float(self.elevation),
            "slope_l1": float(self.slope),
        }


def loss(prediction, truth, void, *, slope_weight: float) -> Loss:
    """§4.1's loss: masked L1 in metres, plus ``slope_weight`` x an L1 Sobel-gradient term.

    All four tensors are ``(N, 1, 224, 224)`` and both elevation tensors are in **metres** — the
    head's residual has already been added to the A1 fill by the caller, so what is compared here
    is a whole predicted tile against a whole true one, and the mask is what restricts the
    comparison to the void.

    Both terms are averaged over **void pixels**, not over the tile: §3.2's masks span 16 to 158
    px equivalent diameter, a factor of ~100 in area, and a tile-averaged loss would weight a
    small void at a hundredth of a large one inside the same batch. The gradient term averages
    over the same pixels *and* over its two components, so ``slope_weight`` is a ratio between
    two per-pixel means rather than between a mean and a sum.

    The gradient sites are the void itself rather than the eroded void. That is the opposite of
    what :func:`benchmark.metrics.score` does and it is deliberate: erosion is a *contamination
    control for scoring* — it exists so a reported slope error reads only predicted pixels — and
    the seam between the fill and the surrounding ground is precisely the thing this term is
    supposed to teach the head to get right. Eroding it away would remove the boundary from the
    loss and leave it in the metric.
    """
    import torch

    mask = void.to(prediction.dtype)
    pixels = mask.sum().clamp_min(1.0)
    elevation = ((prediction - truth).abs() * mask).sum() / pixels

    py, px = sobel_gradients(prediction)
    ty, tx = sobel_gradients(truth)
    slope = (((py - ty).abs() + (px - tx).abs()) * mask).sum() / (2.0 * pixels)

    return Loss(
        total=elevation + float(slope_weight) * slope,
        elevation=elevation.detach() if torch.is_tensor(elevation) else elevation,
        slope=slope.detach() if torch.is_tensor(slope) else slope,
    )
