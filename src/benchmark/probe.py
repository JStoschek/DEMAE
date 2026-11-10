"""§4.1's linear probe: LayerNorm on the extractor's `output`, a 1x1 conv to ten classes (#206).

The strict instrument, and one module for every linear arm: B1, B4, B6, B8, B9 and the B1-R4 /
B1-S rows build it from the encoder's registered width and nothing else, so two linear arms at
the same width differ only in the encoder (story 42). No hidden layer: the probe is

``LayerNorm(d)`` over each patch token -> the 14x14 token grid -> ``Conv2d(d, 10, 1)`` ->
bilinear upsample to the 224 px tile.

Its trainable count is closed-form, ``2d + 10d + 10`` (the norm's affine pair and the conv's
weight and bias): 9,226 at 768-d and 4,618 at 384-d. :func:`build` counts the module and refuses
a disagreement, on :func:`benchmark.head.build`'s rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .head import token_grid

#: WhiteboxTools' ten geomorphon forms (§3.1).
CLASSES = 10
#: The tile the logits are upsampled to.
TILE_PX = 224


def parameter_count(width: int, classes: int = CLASSES) -> int:
    """The probe's trainable parameters: LayerNorm's ``2d`` plus the 1x1 conv's ``d*k + k``."""
    return 2 * width + width * classes + classes


@dataclass(frozen=True)
class Built:
    module: Any
    width: int
    parameters: int


def _build_class():
    import torch

    class LinearProbe(torch.nn.Module):
        def __init__(self, width: int, classes: int, tile_px: int) -> None:
            super().__init__()
            self.norm = torch.nn.LayerNorm(width)
            self.classify = torch.nn.Conv2d(width, classes, kernel_size=1)
            self.tile_px = tile_px

        def forward(self, tokens):
            """``(N, T, d)`` patch tokens -> ``(N, classes, tile_px, tile_px)`` logits."""
            n, t, d = tokens.shape
            grid = token_grid(t)
            x = self.norm(tokens).transpose(1, 2).reshape(n, d, grid, grid)
            logits = self.classify(x)
            return torch.nn.functional.interpolate(
                logits, size=(self.tile_px, self.tile_px), mode="bilinear", align_corners=False
            )

    return LinearProbe


def build(width: int, *, classes: int = CLASSES, seed: int = 0, tile_px: int = TILE_PX) -> Built:
    """The probe for a ``width``-d encoder, initialized under ``seed`` and counted."""
    import torch

    generator = torch.Generator().manual_seed(int(seed))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item()))
        module = _build_class()(width, classes, tile_px)
    counted = sum(p.numel() for p in module.parameters())
    expected = parameter_count(width, classes)
    if counted != expected:
        raise ValueError(
            f"§4.1's linear probe at {width}-d built {counted:,} parameters; the closed form "
            f"gives {expected:,}"
        )
    return Built(module=module, width=width, parameters=counted)
