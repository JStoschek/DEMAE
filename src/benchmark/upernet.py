"""§4.2's UperNet: PANGAEA's decoder, verbatim, with its ViT neck (#208).

PANGAEA (`pangaea/decoders/upernet.py`) puts a ViT's four chosen blocks through
``Feature2Pyramid`` and then mmseg's UperHead:

* **The neck** turns four ``(N, d, 14, 14)`` token grids into a pyramid at rescales 4/2/1/0.5:
  two ``ConvTranspose2d(d, d, 2, 2)`` around a BatchNorm and GELU for the first level, one
  ``ConvTranspose2d`` for the second, identity for the third and a 2x2 max-pool for the fourth.
  All ``d`` channels are kept, so the levels are 56, 28, 14 and 7 px.
* **The PSP module** on the 7 px level: an adaptive average pool to each pool scale (1, 2, 3, 6),
  a 1x1 conv to ``channels``, BatchNorm and ReLU, each upsampled back to 7 px, concatenated with
  the level and fused by a 3x3 conv (the bottleneck).
* **The FPN**: 1x1 lateral convs on the other three levels, a top-down sum, 3x3 convs, every
  level upsampled to 56 px, concatenated and fused by a 3x3 conv (``fpn_bottleneck``).
* **The classifier**: ``Dropout2d(0.1)`` then a 1x1 conv to the ten geomorphon forms, and a
  bilinear upsample to the 224 px tile.

PANGAEA's convs keep their biases (they are plain ``nn.Conv2d`` before a norm). Its
``SyncBatchNorm`` is ``BatchNorm2d`` here: on one device the two compute the same thing, and
``SyncBatchNorm`` refuses to run outside a process group.

Its trainable count is closed-form (:func:`parameter_count`): 39,349,002 at 768-d, 512 channels
and pool scales (1, 2, 3, 6), which is PANGAEA Table 4's 39.34-39.35M for every 768-d encoder.
:func:`build` counts the module and refuses a disagreement, on :func:`benchmark.probe.build`'s
rule.

The head reads the extractor's ``intermediates`` at the config's ``upernet.layers`` (§4.2:
``[3, 5, 7, 11]``, blocks 4, 6, 8 and 12) and nothing else: :func:`read` is the one place that
choice is made.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

from dataset.config import TILE_PX

from .head import token_grid
from .probe import CLASSES

#: PANGAEA's ``Feature2Pyramid`` rescales, one per level, shallowest first.
RESCALES: Final = (4, 2, 1, 0.5)

#: §4.2's defaults, which are :class:`benchmark.landform_config.Upernet`'s.
CHANNELS: Final = 512
POOL_SCALES: Final = (1, 2, 3, 6)


def _conv(cin: int, cout: int, k: int) -> int:
    return cin * cout * k * k + cout


def parameter_count(width: int, *, channels: int = CHANNELS,
                    pool_scales: Sequence[int] = POOL_SCALES, classes: int = CLASSES) -> int:
    """The head's trainable parameters, neck included.

    Neck: three ``ConvTranspose2d(d, d, 2, 2)`` and one BatchNorm. Then per conv-norm-ReLU unit
    a conv with bias and BatchNorm's affine pair: one 1x1 per pool scale, the 3x3 PSP
    bottleneck over ``d + len(pool_scales) * channels``, three 1x1 laterals, three 3x3 FPN
    convs and the 3x3 fuse over ``4 * channels``; then the 1x1 classifier.
    """
    levels = len(RESCALES)
    neck = 3 * _conv(width, width, 2) + 2 * width
    unit = lambda cin, k: _conv(cin, channels, k) + 2 * channels  # noqa: E731
    psp = len(pool_scales) * unit(width, 1) + unit(width + len(pool_scales) * channels, 3)
    fpn = (levels - 1) * (unit(width, 1) + unit(channels, 3)) + unit(levels * channels, 3)
    return neck + psp + fpn + _conv(channels, classes, 1)


@dataclass(frozen=True)
class Built:
    module: Any
    width: int
    parameters: int


def read(features, layers: Sequence[int]) -> list:
    """The four blocks the head reads: ``features.intermediates`` at ``layers``, nothing else."""
    return [features.intermediates[i] for i in layers]


def _build_class():
    import torch
    from torch import nn
    from torch.nn import functional as F

    def unit(cin: int, cout: int, k: int) -> nn.Sequential:
        return nn.Sequential(nn.Conv2d(cin, cout, kernel_size=k, padding=k // 2),
                             nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    class Feature2Pyramid(nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.upsample_4x = nn.Sequential(
                nn.ConvTranspose2d(width, width, kernel_size=2, stride=2),
                nn.BatchNorm2d(width),
                nn.GELU(),
                nn.ConvTranspose2d(width, width, kernel_size=2, stride=2),
            )
            self.upsample_2x = nn.ConvTranspose2d(width, width, kernel_size=2, stride=2)
            self.identity = nn.Identity()
            self.downsample_2x = nn.MaxPool2d(kernel_size=2, stride=2)

        def forward(self, levels):
            ops = (self.upsample_4x, self.upsample_2x, self.identity, self.downsample_2x)
            return [op(x) for op, x in zip(ops, levels)]

    class UperNet(nn.Module):
        def __init__(self, width: int, channels: int, pool_scales: Sequence[int],
                     classes: int, tile_px: int) -> None:
            super().__init__()
            levels = len(RESCALES)
            self.tile_px = tile_px
            self.neck = Feature2Pyramid(width)
            self.psp_modules = nn.ModuleList(
                nn.Sequential(nn.AdaptiveAvgPool2d(scale), *unit(width, channels, 1))
                for scale in pool_scales
            )
            self.bottleneck = unit(width + len(pool_scales) * channels, channels, 3)
            self.lateral_convs = nn.ModuleList(unit(width, channels, 1)
                                               for _ in range(levels - 1))
            self.fpn_convs = nn.ModuleList(unit(channels, channels, 3)
                                           for _ in range(levels - 1))
            self.fpn_bottleneck = unit(levels * channels, channels, 3)
            self.dropout = nn.Dropout2d(0.1)
            self.conv_seg = nn.Conv2d(channels, classes, kernel_size=1)

        @staticmethod
        def _up(x, size):
            return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

        def _psp(self, x):
            pooled = [self._up(module(x), x.shape[2:]) for module in self.psp_modules]
            return self.bottleneck(torch.cat([x, *pooled], dim=1))

        def forward(self, blocks):
            """Four ``(N, T, d)`` blocks, shallowest first -> ``(N, classes, tile, tile)``."""
            grids = []
            for tokens in blocks:
                n, t, d = tokens.shape
                g = token_grid(t)
                grids.append(tokens.transpose(1, 2).reshape(n, d, g, g))
            inputs = self.neck(grids)
            laterals = [conv(inputs[i]) for i, conv in enumerate(self.lateral_convs)]
            laterals.append(self._psp(inputs[-1]))
            for i in range(len(laterals) - 1, 0, -1):
                laterals[i - 1] = laterals[i - 1] + self._up(laterals[i],
                                                             laterals[i - 1].shape[2:])
            outs = [conv(laterals[i]) for i, conv in enumerate(self.fpn_convs)]
            outs.append(laterals[-1])
            outs = [outs[0], *(self._up(o, outs[0].shape[2:]) for o in outs[1:])]
            x = self.conv_seg(self.dropout(self.fpn_bottleneck(torch.cat(outs, dim=1))))
            return self._up(x, (self.tile_px, self.tile_px))

    return UperNet


def build(width: int, *, channels: int = CHANNELS, pool_scales: Sequence[int] = POOL_SCALES,
          classes: int = CLASSES, seed: int = 0, tile_px: int = TILE_PX) -> Built:
    """The head for a ``width``-d encoder, initialized under ``seed`` and counted."""
    import torch

    generator = torch.Generator().manual_seed(int(seed))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item()))
        module = _build_class()(width, channels, tuple(pool_scales), classes, tile_px)
    counted = sum(p.numel() for p in module.parameters() if p.requires_grad)
    expected = parameter_count(width, channels=channels, pool_scales=pool_scales,
                               classes=classes)
    if counted != expected:
        raise ValueError(
            f"§4.2's UperNet at {width}-d and {channels} channels built {counted:,} parameters; "
            f"the closed form gives {expected:,}"
        )
    return Built(module=module, width=width, parameters=counted)
