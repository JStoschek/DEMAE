"""Task 2's arm table — plan §1's rows, keyed into the shared encoder registry (#198).

A **sibling** of :data:`benchmark.arms.ARMS`, not more rows in it. Task 1's table is what
Task 1's resolver, report and printed rows walk, and a B-row there would be a row every one of
them had to learn to skip. What the two tasks *do* share is the encoder registry,
:data:`benchmark.arms.ENCODERS`: an arm here names its encoder by that key, so the pin an encoder
carries there (ADR-0012, "an unpinned encoder is unrunnable") is the pin this table runs under,
and there is one place a checkpoint md5 lives.

#198 registered **B1**: it is the default cell, the way A4-R5 is Task 1's, and a config needs
a valid arm to construct. The rows a later ticket runs arrive with it — B6, B1-R4 and B1-S came
with the linear-probe cell (#206), each with the linear head only until #208 gives B6 its
UperNet; the UperNet rows arrive with #208, B7 with #209, B3 with #210, the
TerraMind rows with #211 — so no row here outruns the code that can run it.

## What an arm declares

The encoder it reads, the heads it trains, and what it trains. ``heads`` is a tuple because the
cell's ``--head`` is checked against it rather than inferred from it (see
:class:`benchmark.landform_config.LandformConfig`); an arm with one head needs no flag, and a
cell asking it for another is refused by name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from . import arms

#: The heads §4 trains: §4.1's linear probe, §4.2's UperNet (PANGAEA's, verbatim) and §4.3's
#: UNet for B7. These are also the three ``--tune`` passes §4.5 runs per rung.
HEADS: Final = ("linear", "upernet", "unet")

#: What an arm trains: a head over a frozen encoder (B1/B2/B4/B5/B6/B8/B9 and the extra rows),
#: the encoder and the head (B3), or everything from scratch (B7).
TRAINS: Final = ("head", "encoder+head", "everything")


@dataclass(frozen=True)
class LandformArm:
    """One row of plan §1's table."""

    name: str
    #: A key into :data:`benchmark.arms.ENCODERS`, or ``None`` for the arm that runs none (B7).
    encoder: str | None
    #: The heads this arm trains, the first being the one a cell gets without ``--head``.
    heads: tuple[str, ...]
    #: One of :data:`TRAINS`.
    trains: str
    #: What the resolution line says this arm is.
    what: str

    def __post_init__(self) -> None:
        if self.encoder is not None and self.encoder not in arms.ENCODERS:
            raise ValueError(
                f"arm {self.name} reads encoder {self.encoder!r}, which is not in "
                f"benchmark.arms.ENCODERS; registered: {sorted(arms.ENCODERS)}"
            )
        if not self.heads or set(self.heads) - set(HEADS):
            raise ValueError(f"arm {self.name}: heads {self.heads} must be drawn from {HEADS}")
        if self.trains not in TRAINS:
            raise ValueError(f"arm {self.name}: trains {self.trains!r} is not one of {TRAINS}")


def _arm(name: str, **kwargs) -> tuple[str, LandformArm]:
    """A table entry, keyed by its own name so the two cannot disagree."""
    return name, LandformArm(name=name, **kwargs)


ARMS: Final[dict[str, LandformArm]] = dict(
    [
        _arm(
            "B1",
            encoder="demae-r5",
            heads=("linear",),
            trains="head",
            what="DEMAE R5 frozen + §4.1's linear probe on `output` — core, C1",
        ),
        _arm(
            "B6",
            encoder="imagenet-mae",
            heads=("linear",),
            trains="head",
            what="ImageNet-MAE frozen + the identical linear probe — core, C1, and §4.5's "
            "tuning arm",
        ),
        _arm(
            "B1-R4",
            encoder="demae-r4",
            heads=("linear",),
            trains="head",
            what="DEMAE R4 (ImageNet warm start) frozen + the identical linear probe — extra "
            "row, never C1's number (L5, L9)",
        ),
        _arm(
            "B1-S",
            encoder="demae-vits",
            heads=("linear",),
            trains="head",
            what="DEMAE ViT-S frozen + the identical linear probe at 384-d — appendix, no "
            "bootstrap pair",
        ),
        _arm(
            "B7",
            encoder=None,
            heads=("unet",),
            trains="everything",
            what="Task 1's A7 UNet on one channel, from scratch at B2's batch and epochs, "
            "tuning itself — C2's and C3's bar (§4.3)",
        ),
    ]
)


def get(name: str) -> LandformArm:
    """The registered Task 2 arm, or a refusal naming the table."""
    try:
        return ARMS[name]
    except KeyError:
        raise ValueError(f"unknown Task 2 arm {name!r}; registered: {sorted(ARMS)}") from None
