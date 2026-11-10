"""Task 2's arm table — plan §1's rows, keyed into the shared encoder registry (#198).

A **sibling** of :data:`benchmark.arms.ARMS`, not more rows in it. Task 1's table is what
Task 1's resolver, report and printed rows walk, and a B-row there would be a row every one of
them had to learn to skip. What the two tasks *do* share is the encoder registry,
:data:`benchmark.arms.ENCODERS`: an arm here names its encoder by that key, so the pin an encoder
carries there (ADR-0012, "an unpinned encoder is unrunnable") is the pin this table runs under,
and there is one place a checkpoint md5 lives.

#198 registered **B1**: it is the default cell, the way A4-R5 is Task 1's, and a config needs
a valid arm to construct. The rows a later ticket runs arrive with it — B6, B1-R4 and B1-S came
with the linear-probe cell (#206); B2, B2-R4, B2-S and B6's second head, the UperNet, came with
#208; B7 arrives with #209, B3 with #210, the TerraMind rows with #211 — so no row here outruns
the code that can run it.

B6 is one row with two heads, as plan §1 has it: ``--head upernet`` names its UperNet cell, and
the parquet keys a cell by its head as well as its arm, so B6's two cells never collide.

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
    #: The rungs this arm runs at; every rung unless the plan budgets fewer (B3: 100% only).
    rungs: tuple[str, ...] = ("1%", "10%", "100%")
    #: ADR-0015 L9: an insulated row carries no claim and never stands in for a core row. It
    #: prints beside the core grid, never in it.
    insulated: bool = False

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
        if not self.rungs or set(self.rungs) - {"1%", "10%", "100%"}:
            raise ValueError(f"arm {self.name}: rungs {self.rungs} must be drawn from §3.3's")


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
            heads=("linear", "upernet"),
            trains="head",
            what="ImageNet-MAE frozen + the identical linear probe or UperNet — core, C1, and "
            "§4.5's tuning arm for both",
        ),
        _arm(
            "B1-R4",
            encoder="demae-r4",
            heads=("linear",),
            trains="head",
            what="DEMAE R4 (ImageNet warm start) frozen + the identical linear probe — extra "
            "row, never C1's number (L5, L9)",
            insulated=True,
        ),
        _arm(
            "B1-S",
            encoder="demae-vits",
            heads=("linear",),
            trains="head",
            what="DEMAE ViT-S frozen + the identical linear probe at 384-d — appendix, no "
            "bootstrap pair",
            insulated=True,
        ),
        _arm(
            "B2",
            encoder="demae-r5",
            heads=("upernet",),
            trains="head",
            what="DEMAE R5 frozen + §4.2's UperNet (PANGAEA's) on intermediates[3, 5, 7, 11] — "
            "core, C1, and C2's money plot",
        ),
        _arm(
            "B2-R4",
            encoder="demae-r4",
            heads=("upernet",),
            trains="head",
            what="DEMAE R4 (ImageNet warm start) frozen + the identical UperNet — extra row, "
            "never C1's number (L5, L9)",
            insulated=True,
        ),
        _arm(
            "B2-S",
            encoder="demae-vits",
            heads=("upernet",),
            trains="head",
            what="DEMAE ViT-S frozen + the identical UperNet at 384-d — appendix, no bootstrap "
            "pair",
            insulated=True,
        ),
        _arm(
            "B7",
            encoder=None,
            heads=("unet",),
            trains="everything",
            what="Task 1's A7 UNet on one channel, from scratch at B2's batch and epochs, "
            "tuning itself — C2's and C3's bar (§4.3)",
        ),
        _arm(
            "B3",
            encoder="demae-r5",
            heads=("upernet",),
            trains="encoder+head",
            rungs=("100%",),
            what="DEMAE R5 fine-tuned, all weights, with §4.2's UperNet under §4.3's literature "
            "recipe, untuned — C3b, against B7 at 100%",
        ),
        # The TerraMind rows (#211). B4/B5 read the DEM tokenizer, B9 the v1 base backbone; B8
        # and the ALT rows are registry entries of their own on the `terramind-bicubic`
        # precedent, so a control can never delete the row it sits beside.
        _arm(
            "B4",
            encoder="terramind",
            heads=("linear",),
            trains="head",
            what="TerraMind DEM tokenizer frozen (768-d `feats`, pre-`quant_proj`) + the "
            "identical linear probe — core, C1",
        ),
        _arm(
            "B5",
            encoder="terramind",
            heads=("upernet",),
            trains="head",
            what="TerraMind DEM tokenizer frozen + the identical UperNet on "
            "intermediates[3, 5, 7, 11] — core, C1",
        ),
        _arm(
            "B9",
            encoder="terramind-backbone",
            heads=("linear", "upernet"),
            trains="head",
            what="TerraMind v1 base backbone frozen on raw DEM + the identical linear probe "
            "(on the normed `output`) or UperNet — core, C1",
        ),
        _arm(
            "B8",
            encoder="terramind-latent",
            heads=("linear",),
            trains="head",
            what="TerraMind tokenizer's post-`quant_proj`, pre-FSQ 5-d latent + the linear probe "
            "— ablation, insulated (L9)",
            rungs=("100%",),
            insulated=True,
        ),
        _arm(
            "B4-ALT",
            encoder="terramind-per-tile",
            heads=("linear",),
            trains="head",
            what="TerraMind tokenizer fed DEMAE's per-tile diet + the identical linear probe — "
            "altitude-passthrough control, insulated (L9)",
            rungs=("100%",),
            insulated=True,
        ),
        _arm(
            "B5-ALT",
            encoder="terramind-per-tile",
            heads=("upernet",),
            trains="head",
            what="TerraMind tokenizer fed DEMAE's per-tile diet + the identical UperNet — "
            "altitude-passthrough control, insulated (L9)",
            rungs=("100%",),
            insulated=True,
        ),
    ]
)


def get(name: str) -> LandformArm:
    """The registered Task 2 arm, or a refusal naming the table."""
    try:
        return ARMS[name]
    except KeyError:
        raise ValueError(f"unknown Task 2 arm {name!r}; registered: {sorted(ARMS)}") from None
