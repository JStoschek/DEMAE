"""§4.1's head with a full-resolution skip path — the variant A4-R5-SKIP, A5-SKIP and A6-SKIP
train (#182).

:class:`benchmark.head.FillHead` reads the 14x14 token grid and the void mask, and nothing else.
Detail finer than a 16 px patch therefore has to be rebuilt by four bilinear upsamples, and the
memo's C3 section says where that costs: A7's from-scratch UNet, which reads the tile at 224 px,
beats A4-R5 on ``slope_rmse`` in all six printed cells at 20k, and most of the slope error sits on
the void boundary. This module tests one hypothesis about that gap — that the head is starved of
full-resolution input, not of representation — by handing the **A1-filled tile** to the same head
through a skip path, DPT/UNet-style, and changing nothing else.

## What the skip path is

A small encoder over ``[standardized A1 fill, void]`` at the tile's own 224 px: one 3x3 conv to
:data:`SKIP_CHANNELS`, then one stride-2 3x3 conv per head stage, so the skip features exist at
224, 112, 56, 28 and 14 px. Each of §4.1's four stages reads the level at its own input
resolution, concatenated beside the mask channel it already reads; and a **refinement stage** at
224 px reads the last stage's output beside the full-resolution level before the 1-channel
residual. That last stage is the one the registered head has no room for — its output convolution
is a 1x1 off a bilinearly upsampled 112 px map — and it is where a boundary can be sharpened.

Everything the registered head has, this one has in the same place with the same name: the
LayerNorm, the projection, the four stages, the GroupNorm and the zero-initialized residual. A
state dict of this module minus the ``skip_*`` and ``refine*`` tensors is a :class:`FillHead`
state dict with wider stage inputs. That is what makes a *skip-versus-not* delta a claim about the
skip path.

## The skip input never sees the void's truth

The tile the skip path reads is **A1's fill**, which :func:`benchmark.fill.fill` interpolates from
the non-void pixels alone, and it is standardized per tile over ``valid = land ∧ ¬void`` through
:func:`benchmark.unet.condition` — the same statistic §2.1 feeds the frozen encoders and §6 item 2
commits to. So a change to the truth *inside* the void changes neither the tokens, nor the image,
nor the std, nor the output; ``tests/test_p182_skip_head.py`` proves that with a perturbed tile
rather than asserting it. The residual is scaled back to metres by that std the way A7's is
(:class:`benchmark.train.SkipStack`), because the image the trunk reads is in standardized units
and a residual predicted in those units is only in metres up to a per-tile factor.

## It is a variant, not a change to the registered head

§4.1's head is the one C1 was scored through, and every one of its rows carries a ``config_hash``
that must not move. So nothing here touches :class:`benchmark.config.Head` — the skip path's shape
is fixed in this module, the arm declares that it reads the full-resolution tile
(:attr:`benchmark.arms.Arm.head_variant`), and the new arms carry ``printed=False`` on
A9-NOTHRESH's precedent: R5's eleven printed rows were pre-registered, and a head registered after
they were read is an appendix cell however it scores. The optimizer, the schedule, the batch and
the loss are §4.1's frozen ones, read off ``cfg.head`` and gated by :func:`benchmark.tune.
assert_frozen` exactly as for A4, A5 and A6; this module has no knob of its own to tune.

## Not a way to produce a result

Like :mod:`benchmark.head`: this builds a module. `python -m benchmark.launch run` is the only
thing that scores anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from dataset import config as dataset_config

from .head import GROUPS, HeadError, _mask_at, _stage_channels, token_grid

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .config import Head

#: The skip path's width at every level. Sixteen keeps the whole path under a tenth of the
#: registered head — §6 item 4 prints the trainable count, and a variant that doubled it would
#: be testing capacity as much as resolution.
SKIP_CHANNELS: Final = 16

#: GroupNorm's group count on the skip path. §4.1's 32 does not divide :data:`SKIP_CHANNELS`,
#: and the stage GroupNorms keep §4.1's 32 — this constant governs the skip tensors alone.
SKIP_GROUPS: Final = 8

#: The variant's cost at §4.1's two widths, recorded so the number §6 item 4 would print is a
#: committed fact. Checked against :func:`parameter_count` by a test rather than trusted, on
#: :data:`benchmark.head.PARAMETERS`' rule. The registered head is 1,082,017 and 1,181,089; the
#: skip path adds 92,784 at either width, none of it in the projection.
PARAMETERS: Final[dict[int, int]] = {384: 1_174_801, 768: 1_273_873}

#: The name :attr:`benchmark.arms.Arm.head_variant` carries for this module.
VARIANT: Final = "skip"


# --- The arithmetic, before any torch ----------------------------------------


def _conv(in_ch: int, out_ch: int, k: int = 3) -> int:
    return in_ch * out_ch * k * k + out_ch


def parameter_count(width: int, recipe: Head | None = None) -> int:
    """The variant at ``width``, counted in closed form — :func:`benchmark.head.parameter_count`
    plus the skip path, so the two differ by a number a reader can check by hand."""
    from .config import Head as HeadRecipe

    recipe = HeadRecipe() if recipe is None else recipe
    total = 2 * width
    total += recipe.proj_dim * width + recipe.proj_dim
    # The skip encoder: [image, void] to SKIP_CHANNELS at 224, then one stride-2 conv per stage.
    total += _conv(2, SKIP_CHANNELS) + 2 * SKIP_CHANNELS
    total += len(recipe.stage_dims) * (_conv(SKIP_CHANNELS, SKIP_CHANNELS) + 2 * SKIP_CHANNELS)
    channels = recipe.proj_dim
    for out in recipe.stage_dims:
        # §4.1's stage over its input, the mask **and** the skip level, then GroupNorm's 2C.
        total += _conv(channels + 1 + SKIP_CHANNELS, out) + 2 * out
        channels = out
    # The refinement at 224: the last stage beside the full-resolution level, then GroupNorm.
    total += _conv(channels + SKIP_CHANNELS, channels) + 2 * channels
    return total + _conv(channels, 1, k=1)


def format_parameters(width: int, recipe: Head | None = None) -> str:
    """The console line: the variant's count, and what it adds to the registered head."""
    from . import head

    count = parameter_count(width, recipe)
    added = count - head.parameter_count(width, recipe)
    return (
        f"{count:,} ({count / 1e6:.2f}M) at {width}-d — §4.1's head plus a {added:,}-parameter "
        f"full-resolution skip path (#182)"
    )


# --- The module ---------------------------------------------------------------


def _build_class():
    """The variant, defined against a torch imported at call time — :func:`benchmark.head.
    _build_class`'s rule, for the same torch-optional reason."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class SkipFillHead(nn.Module):
        """``forward(tokens, void, image) -> residual`` in **standardized** units.

        ``tokens`` and ``void`` are :class:`benchmark.head.FillHead`'s. ``image`` is
        ``(N, 1, 224, 224)`` float: the A1 fill standardized per tile over its valid pixels
        (:func:`benchmark.unet.condition`). The caller multiplies the output by the std that
        standardized the image — see :class:`benchmark.train.SkipStack`.
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
            self.skip_in = nn.Conv2d(2, SKIP_CHANNELS, kernel_size=3, padding=1)
            self.skip_in_norm = nn.GroupNorm(SKIP_GROUPS, SKIP_CHANNELS)
            self.skip_down = nn.ModuleList(
                nn.Conv2d(SKIP_CHANNELS, SKIP_CHANNELS, kernel_size=3, stride=2, padding=1)
                for _ in recipe.stage_dims
            )
            self.skip_down_norms = nn.ModuleList(
                nn.GroupNorm(SKIP_GROUPS, SKIP_CHANNELS) for _ in recipe.stage_dims
            )
            self.convs = nn.ModuleList(
                nn.Conv2d(into + SKIP_CHANNELS, out, kernel_size=3, padding=1)
                for into, out in _stage_channels(recipe)
            )
            self.norms = nn.ModuleList(
                nn.GroupNorm(GROUPS, out) for out in recipe.stage_dims
            )
            last = recipe.stage_dims[-1]
            self.refine = nn.Conv2d(last + SKIP_CHANNELS, last, kernel_size=3, padding=1)
            self.refine_norm = nn.GroupNorm(GROUPS, last)
            self.residual = nn.Conv2d(last, 1, kernel_size=1)

        def forward(self, tokens, void, image):
            if tokens.ndim != 3 or tokens.shape[1] != self.tokens:
                raise HeadError(
                    f"the skip head reads ADR-0012 §5's (N, {self.tokens}, D) patch tokens; got "
                    f"{tuple(tokens.shape)}"
                )
            if tokens.shape[2] != self.width:
                raise HeadError(
                    f"this head was built at {self.width}-d and was handed {tokens.shape[2]}-d "
                    f"tokens. The three skip arms share one head differing **only** in this "
                    f"width, so it is checked rather than absorbed by a reshape"
                )
            full = (self.tile_px, self.tile_px)
            if void.shape[-2:] != full or image.shape[-2:] != full:
                raise HeadError(
                    f"the void mask and the A1 fill reach the skip head at full resolution "
                    f"({self.tile_px}x{self.tile_px}); got {tuple(void.shape)} and "
                    f"{tuple(image.shape)}"
                )
            void = void.to(tokens.dtype)
            image = image.to(tokens.dtype)

            # The skip levels, finest first: index k is at tile_px / 2**k.
            s = F.gelu(self.skip_in_norm(self.skip_in(torch.cat([image, void], dim=1))))
            levels = [s]
            for conv, norm in zip(self.skip_down, self.skip_down_norms):
                s = F.gelu(norm(conv(s)))
                levels.append(s)

            x = self.norm(tokens)
            x = x.transpose(1, 2).reshape(x.shape[0], self.width, self.grid, self.grid)
            x = self.project(x)
            stages = len(self.convs)
            for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
                size = tuple(x.shape[-2:])
                level = levels[stages - i]
                if tuple(level.shape[-2:]) != size:
                    raise HeadError(
                        f"stage {i} runs at {size} and its skip level is "
                        f"{tuple(level.shape[-2:])}"
                    )
                x = torch.cat([x, _mask_at(void, size), level], dim=1)
                x = F.gelu(norm(conv(x)))
                x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = torch.cat([x, levels[0]], dim=1)
            x = F.gelu(self.refine_norm(self.refine(x)))
            return self.residual(x)

    return SkipFillHead


@dataclass(frozen=True)
class Built:
    """The variant and the two facts §6 item 4 would print about it."""

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
    """The one head the three skip arms instantiate — :func:`benchmark.head.build`'s three
    guarantees, for the same reasons: the upsampling lands on the tile, the parameter count is
    asserted against the closed form, and the residual is zeroed so the head **is** A1 at step 0.

    The seed is consumed exactly as :func:`benchmark.head.build` consumes it, so a skip arm's
    three seeds are separated by :attr:`benchmark.config.BenchmarkConfig.seed` and nothing else.
    """
    import torch

    from .config import Head as HeadRecipe

    recipe = HeadRecipe() if recipe is None else recipe
    grid = token_grid(tokens)
    reached = grid * 2 ** len(recipe.stage_dims)
    if reached != tile_px:
        raise HeadError(
            f"the skip head upsamples a {grid}x{grid} token grid by 2x per stage; "
            f"{len(recipe.stage_dims)} stages reach {reached} px and the tile is {tile_px}. The "
            f"skip levels are pooled from the tile by the same factor, so a head that does not "
            f"land on it has no level to read"
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

    torch.nn.init.zeros_(module.residual.weight)
    torch.nn.init.zeros_(module.residual.bias)

    counted = sum(p.numel() for p in module.parameters())
    expected = parameter_count(width, recipe)
    if counted != expected:
        raise HeadError(
            f"the skip head at {width}-d built {counted:,} parameters and this recipe derives "
            f"{expected:,}. The two are computed by different means so that they can disagree"
        )
    return Built(module=module, width=width, parameters=counted)
