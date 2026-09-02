"""§4.4's from-scratch UNet — A7, the PANGAEA bar (#162).

The one arm with no pretraining behind it, and the reason the results table prints parameter
counts **total and trainable**: A7 trains ~17.5M against every frozen arm's 1.18M, so a
totals-only column (17M against A4's 86.4M) reads the baseline as handicapped when it is the
opposite. If A7 matches A4 at 20k tiles, the honest conclusion is that pretraining pays only in
the low-label regime — which is what makes this bar worth building properly rather than
building weakly and calling it a baseline.

## What this module does not own

Nothing about the loss, the schedule or the seeds. §4.4 says *same loss, same schedule length,
same seeds* as the frozen arms, and the way that is kept true here is that there is no second
copy of any of it: A7 trains through :func:`benchmark.train.train_unet`, which reads
:attr:`benchmark.config.BenchmarkConfig.head` — §4.1's recipe object, the one A4, A5 and A6
read — and refuses anything else. :data:`benchmark.config.SHARED_WITH_HEAD` makes the same
statement one layer down, by refusing a :class:`~benchmark.config.Unet` that grows an ``lr`` or
an ``epochs`` field at all.

So what is configurable here is *shape* — depth, base width, the two input channels — and
:class:`~benchmark.config.Unet` carries exactly those.

## The input conditioning is not §2.1's diet

A7's registry row declares ``normalization="none"``, and that is correct: §2.1's normalization
field is **the frozen encoder's declared diet**, the distribution a pretrained model was fed and
silently misreads when it is fed another (p118 §7). A7 has no encoder and no diet.

It still needs its input scaled. A from-scratch trunk handed raw metres — the corpus spans
roughly -400 to 8,800 — trains badly for reasons that have nothing to do with representation
quality, and a baseline crippled by its input scaling is not the hard bar §4.4 asks for. So
:func:`condition` standardizes the A1 fill per tile over the valid pixels, and the UNet's
residual is scaled **back into metres by the same per-tile std** before §2.4's convention
applies it.

That formula is :func:`benchmark.extract.normalize`'s ``per_tile`` and it is deliberately not a
call to it: the two are the same arithmetic answering different questions, and an arm with no
encoder reaching into the frozen-feature contract for its own conditioning would put A7 on a
seam that ADR-0012 governs for a reason that does not apply to it. What *is* shared is
:func:`benchmark.extract.tile_statistics` — the mean and std themselves, which p137 measured and
which have one code path on purpose.

## Why the residual comes back in metres

§4.1's loss is masked L1 **in metres** and §2.4's convention adds a residual to the A1 fill in
metres. A UNet emitting standardized units would make both true only up to a per-tile factor,
and the factor would be invisible: the run would converge, the val curve would look ordinary,
and every A7 row would carry a scale error that varies by tile ruggedness. Multiplying by the
same std that divided the input keeps the arm's output in the units the loss and the metric are
written in, and keeps the zero-init below exact.

## A7 is A1 at step 0, exactly

The output convolution is zeroed — weight and bias — so the first forward returns zeros, the
residual is zero, and :func:`benchmark.head.apply_residual` returns the A1 fill unchanged. Same
guarantee §4.1's head carries and for the same reason: it makes "a head that learns nothing
scores exactly A1" a property of the code rather than a hope, and it means an A7 row that comes
in at A1's numbers is a training failure that is *visible* rather than a fill wearing A7's name.

## GroupNorm at 16 groups, not §4.1's 32

§4.4's ~17M lands at a base width of 48 (:data:`PARAMETERS`), and 48 is not a multiple of
:data:`benchmark.head.GROUPS`. Rather than move the width to suit a constant from another
section, this module carries its own group count, which divides all five of A7's widths. The
choice of GroupNorm over the textbook UNet's BatchNorm is the same argument §4.1's head makes:
BatchNorm's running statistics would make an eval fill depend on the batch it was scored in, and
its affine-plus-buffer split would put a number in the state dict that is neither trainable nor
absent from the checkpoint — directly across the one column this ticket exists to print.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from .config import Unet

#: GroupNorm's group count for A7's five widths (48, 96, 192, 384, 768). See the module
#: docstring on why this is not §4.1's 32.
GROUPS: Final = 16

#: The smallest per-tile std :func:`condition` will divide by, in metres. A flat tile — a salt
#: flat, a lake margin — has a true std of 0, and the floor is what keeps its standardized image
#: finite. :data:`benchmark.extract.STD_FLOOR`'s **value**, restated rather than imported the
#: way that module restates :data:`dataset.loader._STD_FLOOR`'s: the guard is one number this
#: repo standardizes by everywhere, and A7 conditioning at a different floor from the diet every
#: other arm is fed would be a difference between arms that nothing in §4.4 asks for.
STD_FLOOR: Final = 1e-6


class UnetError(ValueError):
    """A UNet that could not be built to §4.4's shape — never one quietly reshaped."""


def stage_widths(recipe: Unet | None = None) -> list[int]:
    """The encoder's per-stage widths, doubling from ``base_width``. The decoder mirrors them."""
    recipe = Unet() if recipe is None else recipe
    return [recipe.base_width * 2**i for i in range(recipe.depth)]


def _conv(in_ch: int, out_ch: int, k: int = 3) -> int:
    return in_ch * out_ch * k * k + out_ch


def _block(in_ch: int, out_ch: int) -> int:
    """Two [3x3 conv, GroupNorm, GELU] — the UNet's repeating unit. GELU has no parameters."""
    return _conv(in_ch, out_ch) + 2 * out_ch + _conv(out_ch, out_ch) + 2 * out_ch


def parameter_count(recipe: Unet | None = None) -> int:
    """§4.4's count, derived from the shape without allocating anything.

    Torch-free so that :func:`benchmark.arms.parameter_counts` can print A7's pair without
    importing torch — :mod:`benchmark.arms` is a declarative registry and the freeze path has no
    CUDA stack. :func:`build` counts the real module and refuses a disagreement, which is what
    makes this a derivation rather than a second definition (:func:`benchmark.head.parameter_count`'s
    rule).
    """
    recipe = Unet() if recipe is None else recipe
    widths = stage_widths(recipe)
    total = 0
    channels = recipe.in_channels
    for width in widths:
        total += _block(channels, width)
        channels = width
    bottleneck = widths[-1] * 2
    total += _block(channels, bottleneck)
    channels = bottleneck
    for width in reversed(widths):
        # ConvTranspose2d(channels, width, 2, stride=2), then a block over the concatenated skip.
        total += channels * width * 2 * 2 + width
        total += _block(width * 2, width)
        channels = width
    return total + _conv(channels, 1, k=1)


#: §4.4's count at the shipped base width, recorded so §6 item 4's number is a committed fact
#: rather than whatever the arithmetic returns today. Checked against :func:`parameter_count` by
#: a test rather than trusted — :data:`benchmark.head.PARAMETERS`' rule.
PARAMETERS: Final[dict[int, int]] = {48: 17_465_377}


def format_parameters(recipe: Unet | None = None) -> str:
    """The resolution line's UNet row — the count, and the shape it is derived from.

    Pre-flight, so the count is the *derived* one: nothing has been built when the resolution
    line prints. :func:`format_counted` is its post-hoc twin, for a pass that has a real module
    and should quote what it actually counted rather than re-deriving it from a recipe the
    caller has to remember to pass.
    """
    recipe = Unet() if recipe is None else recipe
    return (
        f"§4.4 UNet, {recipe.depth}-down/{recipe.depth}-up at base {recipe.base_width}: "
        f"{parameter_count(recipe):,} parameters, all trainable"
    )


def format_counted(parameters: int) -> str:
    """The trained line's UNet row, from the count the pass measured.

    Takes the number rather than the recipe on purpose: a caller holding a trained module holds
    its real parameter count, and re-deriving one from a default recipe is how a run at
    ``--set unet.base_width=64`` prints 17,465,377 in one line and 31,042,945 in another.
    """
    return f"§4.4 UNet: {parameters:,} parameters, all trainable"


def condition(
    a1_fill: np.ndarray, a1_i16: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Standardize the A1 fill per tile over its valid pixels. Returns ``(image, std)``.

    ``std`` comes back because the caller needs it to put the residual back in metres — see the
    module docstring. All three arrays are batched ``(N, H, W)``; the image is float32 and
    ``std`` is ``(N,)``.

    The statistic is taken over ``valid = land ∧ ¬void`` (ADR-0012 §1) — so the void's *filled*
    values, which are A1's guess rather than measured ground, do not set the scale the network
    reads them at.

    Both the float fill and its int16 rounding are taken as arguments because
    :class:`benchmark.train.Composed` already carries the pair: ``a1_i16`` is the fill at the
    corpus's own quantization and is ADR-0012 §1's door — :func:`benchmark.extract.tile_statistics`
    takes int16 metres and refuses anything else — while the image the trunk reads is
    standardized from the unrounded fill. Re-deriving the rounding here would be a second
    rounding of the same array, and the one place the two could differ.
    """
    from .extract import tile_statistics

    raw = np.asarray(a1_fill, dtype=np.float32)
    quantized = np.asarray(a1_i16, dtype=np.int16)
    mask = np.asarray(valid, dtype=bool)
    image = np.empty(raw.shape, dtype=np.float32)
    stds = np.empty(raw.shape[0], dtype=np.float32)
    for i in range(raw.shape[0]):
        mean, std = tile_statistics(quantized[i], mask[i])
        std = max(float(std), STD_FLOOR)
        image[i] = (raw[i] - np.float32(mean)) / np.float32(std)
        stds[i] = np.float32(std)
    return image, stds


def _build_class():
    """§4.4's module, defined inside a function so importing this one costs no torch.

    :func:`benchmark.head._build_class`'s rule: the freeze and scoring paths import this module
    for :func:`parameter_count` alone, and a module-level ``nn.Module`` would make a torch import
    the price of printing a number.
    """
    import torch
    import torch.nn as nn

    def block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(GROUPS, out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(GROUPS, out_ch),
            nn.GELU(),
        )

    class Unet2d(nn.Module):
        """A1's fill and the void mask in, one residual out — §4.4's 4-down/4-up UNet."""

        def __init__(self, recipe: Unet) -> None:
            super().__init__()
            widths = stage_widths(recipe)
            self.down = nn.ModuleList()
            channels = recipe.in_channels
            for width in widths:
                self.down.append(block(channels, width))
                channels = width
            self.pool = nn.MaxPool2d(2)
            self.bottleneck = block(channels, widths[-1] * 2)
            channels = widths[-1] * 2
            self.up = nn.ModuleList()
            self.merge = nn.ModuleList()
            for width in reversed(widths):
                self.up.append(nn.ConvTranspose2d(channels, width, 2, stride=2))
                self.merge.append(block(width * 2, width))
                channels = width
            self.residual = nn.Conv2d(channels, 1, 1)

        def forward(self, image, void):
            # The void mask is a channel rather than a side input: §4.4's input is "the A1-filled
            # tile and the void mask", and a UNet that had to be told twice which pixels are
            # missing would be two conventions. `benchmark.config.Unet.in_channels` refuses
            # anything but the two.
            x = torch.cat([image, void], dim=1)
            skips = []
            for stage in self.down:
                x = stage(x)
                skips.append(x)
                x = self.pool(x)
            x = self.bottleneck(x)
            for up, merge, skip in zip(self.up, self.merge, reversed(skips)):
                x = merge(torch.cat([up(x), skip], dim=1))
            return self.residual(x)

    return Unet2d


@dataclass(frozen=True)
class Built:
    """A UNet and the number §6 item 4 prints about it."""

    module: Any
    parameters: int


def build(recipe: Unet | None = None, *, seed: int = 0, tile_px: int = 224) -> Built:
    """§4.4's UNet, seeded, zero-initialized at the output, and counted.

    Three things happen here that are not "call the constructor", and they are
    :func:`benchmark.head.build`'s three:

    1. **The pooling has to land back on the tile.** Four 2x downsamples and four 2x upsamples
       return to 224 only because 224 is divisible by 2^4. A depth whose pooling left an odd
       intermediate would return a residual that cannot be added to the fill pixel for pixel
       (§2.4), so it is refused here rather than discovered as a shape error mid-epoch.
    2. **The parameter count is asserted.** :func:`parameter_count` derived it before anything
       was allocated; this counts the module and refuses a disagreement.
    3. **The output convolution is zeroed**, so A7 *is* A1 at step 0 — see the module docstring.
       Everything else is torch's own default init, seeded here so that
       :attr:`benchmark.config.BenchmarkConfig.seed` is what separates this cell's three seeds
       and nothing else is.
    """
    import torch

    recipe = Unet() if recipe is None else recipe
    if tile_px % 2**recipe.depth:
        raise UnetError(
            f"§4.4's UNet pools by 2x {recipe.depth} times and the tile is {tile_px} px, which "
            f"{2 ** recipe.depth} does not divide. The residual is added to the A1 fill pixel "
            f"for pixel (§2.4), so a trunk that does not land on the tile is not one this "
            f"convention can use"
        )
    for width in [*stage_widths(recipe), stage_widths(recipe)[-1] * 2]:
        if width % GROUPS:
            raise UnetError(
                f"width {width} is not divisible by GroupNorm's {GROUPS} groups (§4.4's base "
                f"width is {recipe.base_width})"
            )

    generator = torch.Generator().manual_seed(int(seed))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item()))
        module = _build_class()(recipe)

    # §2.4's residual, made exact — zero weight *and* zero bias. See the module docstring.
    torch.nn.init.zeros_(module.residual.weight)
    torch.nn.init.zeros_(module.residual.bias)

    counted = sum(p.numel() for p in module.parameters())
    expected = parameter_count(recipe)
    if counted != expected:
        raise UnetError(
            f"§4.4's UNet built {counted:,} parameters and this recipe derives {expected:,}. "
            f"The two are computed from the same shape by different means precisely so that "
            f"they can disagree — §6 item 4 prints this number, and a printed count nothing "
            f"checked is a quoted one"
        )
    return Built(module=module, parameters=counted)
