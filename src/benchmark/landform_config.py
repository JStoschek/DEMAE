"""Task 2's frozen run object — the landform plan's knobs, serializable, hashable (#198).

The sibling of :class:`benchmark.config.BenchmarkConfig`, and a sibling rather than more fields
on it, for one measured reason: any field added to a shared Task 1 object moves every committed
Task 1 config hash (#182 found it when one ``Head`` field moved fourteen). So this is its own
frozen object with its own :attr:`LandformConfig.config_hash`, and it reuses Task 1's machinery
through functions rather than through inheritance — the frozen base, the partial-dict round
trip, the canonical hash, the override parser and the paths check all come from
:mod:`benchmark.config`, so the two tasks cannot disagree about *how* a config behaves, only
about *what* it holds.

[The Task 2 plan](../../docs/research/task2-landform-plan.md) is the source of every number
here. Structure follows its sections: §3.1's geomorphon parameters, §3.2's Iwahashi rule, §4.1
to §4.3's head recipes, §4.5's schedule and tuning discipline. **An arm is this object with a
knob moved**, as in Task 1.

## The cell

Five scalars, all recipe: ``arm``, ``rung``, ``seed``, ``head`` and ``label_set``. They are
what the Task 2 results parquet is keyed on. ``rung`` is §3.3's tile budget (1% = 200, 10% =
2,000, 100% = 20,000 tiles), not a fraction of geocells.

``head`` **follows the arm unless it is named.** Every arm declares the heads it trains
(:attr:`benchmark.landform_arms.LandformArm.heads`); a config built without a head takes the
arm's first, and one naming a head the arm does not train is refused. The head is filled in at
construction, so a cell's hash is the same whether ``--head linear`` was typed or implied — one
computation, one hash. Moving the arm without naming a head (by flag, file or ``--set``) lets
the head follow the new arm, which is what :func:`build_config` arranges.

## What is not here yet, and where it goes

The frozen per-rung lr lines (§4.5) are a committed artifact that ``run --tune`` writes (#207),
not a knob, so only the grids are here. The geomorphon class weights and the verified-ocean list
are measured by the label pass (#202) and committed in the geomorphon config file, not in this
object. The ladders' draw seeds and budgets are the ``ladders`` group (#200). Each of those
tickets adds its group to
:data:`GROUPS` and, if ``freeze`` writes from it, to :data:`ARTIFACT_GROUPS`. No Task 2 hash is
committed yet, so adding a group here moves nothing anyone has recorded.

Numbers registered by [ADR-0015](../../docs/adr/0015-task2-pre-registration-register.md) are
**not** here: they belong in ``task2-prereg-bar.json`` (#199), for Task 1's reason — a second home
for a registered number is how a threshold moves without anything failing.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from dataset import config as dataset_config

from . import landform_arms, prereg
from .config import (
    ENVIRONMENT,
    Extract,
    Frozen,
    Paths,
    Runtime,
    build_kwargs,
    canonical_hash,
    check_paths,
    config_from_dict,
    config_to_dict,
    overlay,
    parse_override,
    recipe_of,
)

#: §3.3's rungs, smallest first. Tile budgets, nested by construction (#189).
RUNGS: Final = ("1%", "10%", "100%")

#: What each rung is, in tiles: an exact budget, whole geocells plus at most one partial one.
RUNG_TILES: Final = {"1%": 200, "10%": 2_000, "100%": 20_000}

#: The two label sets: §3.1's geomorphons (primary) and §3.2's Iwahashi & Pike groups
#: (the independence check).
LABEL_SETS: Final = ("geomorphon", "iwahashi")

#: WhiteboxTools ``Geomorphons --forms``' ten classes, in its output order (1..10).
GEOMORPHON_CLASSES: Final = (
    "flat", "peak", "ridge", "shoulder", "spur", "slope", "hollow", "footslope", "valley", "pit",
)


def steps(*, epochs: int, batch_size: int, n_train: int, floor: int) -> int:
    """§4.5's run length: ``epochs x ceil(max(n_train, floor) / batch)`` optimizer steps.

    The floor is the 10% rung's 2,000 tiles, so the 1% rung trains as many steps as the 10% rung
    (ten times the epochs over its 200 tiles) and measures the representation rather than
    under-training. 240 / 240 / 2,370 for the linear probe, 1,280 / 1,280 / 12,520 for UperNet
    and B7 (§4.5's table).
    """
    return epochs * math.ceil(max(n_train, floor) / batch_size)


def _positive(where: str, **values: float) -> None:
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{where}.{name} {value} must be > 0")


def _lr_grid(where: str, grid: tuple[float, ...], points: int = 5) -> None:
    """§4.5: five log-spaced points, ascending. An edge winner widens it, so it may grow."""
    if len(grid) < points or list(grid) != sorted(set(grid)) or grid[0] <= 0:
        raise ValueError(
            f"{where}.lr_grid {grid} must be at least {points} ascending positive points (§4.5)"
        )


# --- §3.1 Geomorphon labels ---------------------------------------------------


@dataclass(frozen=True)
class Geomorphon(Frozen):
    """ADR-0014's label parameters — a measured choice on GLO-30, not a literature default.

    All four WBT parameters are passed explicitly on every call (#201), so none of these may be
    left to a WBT default.
    """

    wbt_version: str = "2.4.0"
    #: The per-platform sha256 pin list for the WBT binary; #201's one narrow call refuses a
    #: binary whose hash is not in it. ADR-0014 D1 settled the macOS M-series build's. A build
    #: for another platform is added only once its labels pass the fixed-sample check (#202).
    wbt_sha256: tuple[str, ...] = (
        "63ce77174e2abc32590df2b4868b6fb4123d8598a540290ef9669b2cb919647b",  # macOS M-series
    )
    #: Outer search radius L, in cells (1.5 km at 30 m). WBT's default, chosen by #188's sweep.
    search: int = 50
    #: Flatness threshold, degrees. #188: under 0.5° erases flat, shoulder and footslope.
    threshold: float = 1.0
    skip: int = 0
    fdist: int = 0
    #: Ocean (exact-zero fill) and windows reaching data absent from the GLO-30 release.
    ignore_label: int = 255
    #: §3.1's sensitivity variants, test tiles only, one parameter at a time.
    variant_search: tuple[int, ...] = (25, 100)
    variant_threshold: tuple[float, ...] = (0.75, 1.25)
    #: D5's named exceptions, by COG stem (#201). An absent COG verified to be open ocean reads
    #: as 0 m: S45 E172 is the absent corner neighbour #187 found for the S44 E171 tile. The
    #: census at the widest buffer extends this list, each entry checked (#202).
    verified_ocean: tuple[str, ...] = ("Copernicus_DSM_COG_10_S45_00_E172_00_DEM",)
    #: Absent COGs on land, whose reach is ignored. D5 names "the N41 neighbours of N42 E043 and
    #: N42 E046"; these are the ones `data/cog_frame.json` lacks. Listed rather than defaulted, so
    #: the census refuses any absence nobody has classified.
    absent_land: tuple[str, ...] = (
        "Copernicus_DSM_COG_10_N41_00_E043_00_DEM",
        "Copernicus_DSM_COG_10_N41_00_E044_00_DEM",
        "Copernicus_DSM_COG_10_N41_00_E045_00_DEM",
        "Copernicus_DSM_COG_10_N41_00_E046_00_DEM",
        "Copernicus_DSM_COG_10_N41_00_E047_00_DEM",
    )
    #: Tile COGs whose windows wrap across the antimeridian to W180 (D5).
    antimeridian_wrap: tuple[str, ...] = (
        "Copernicus_DSM_COG_10_S85_00_E179_00_DEM",
        "Copernicus_DSM_COG_10_S86_00_E179_00_DEM",
    )
    #: How many test tiles every `freeze` regenerates from the COGs and requires byte-identical
    #: to the committed fingerprints, every variant (story 18). The lowest in `tie_rank` order.
    fixed_sample: int = 8

    def __post_init__(self) -> None:
        _positive("geomorphon", search=self.search, threshold=self.threshold,
                  fixed_sample=self.fixed_sample)
        for name in ("verified_ocean", "absent_land", "antimeridian_wrap"):
            listed = getattr(self, name)
            if list(listed) != sorted(set(listed)):
                raise ValueError(f"geomorphon.{name} {listed} must be sorted COG stems, once each")
        if set(self.verified_ocean) & set(self.absent_land):
            raise ValueError("an absent COG is either verified ocean or land, not both")
        if self.skip < 0 or self.fdist < 0:
            raise ValueError(f"geomorphon skip {self.skip} / fdist {self.fdist} must be >= 0")
        if not len(GEOMORPHON_CLASSES) < self.ignore_label <= 255:
            raise ValueError(
                f"geomorphon.ignore_label {self.ignore_label} must be a uint8 outside the "
                f"{len(GEOMORPHON_CLASSES)} classes"
            )
        if self.search in self.variant_search or self.threshold in self.variant_threshold:
            raise ValueError("a sensitivity variant cannot equal the primary parameter it varies")

    @property
    def widest_search(self) -> int:
        """The L a test tile is read at once, every variant cropped from that read (§3.1)."""
        return max(self.search, *self.variant_search)


# --- §3.2 Iwahashi & Pike tile labels -----------------------------------------


@dataclass(frozen=True)
class Iwahashi(Frozen):
    """#193's tile-label rule: whole-footprint majority, nodata and 0 counting against it."""

    #: The distributor's zip, pinned (§3.4).
    zip_md5: str = "5ae98dc7782c13384a1df4d8d609418f"
    #: L17's 15 groups and its 60% rule, read off ADR-0015's register rather than typed here: a
    #: second home for a registered number is how one moves without anything failing (#199).
    groups: int = prereg.TASK2.rule("L17").value["iwahashi_groups"]
    #: Primary; C3 is read here only.
    threshold: float = prereg.TASK2.rule("L17").value["iwahashi_rule"]
    #: A pre-registered sensitivity row, never used to decide C3.
    sensitivity_threshold: float = 0.50
    #: #193's surviving test tiles after #150's land filter, at ``threshold`` then
    #: ``sensitivity_threshold`` (§3.2: 1,985 and 2,712 of 4,523). ``freeze`` refuses a rebuild
    #: that does not reproduce them (#203), so a footprint or land-filter bug stops there rather
    #: than in a fitted classifier. A check on the labels, so it sits beside the rule it checks.
    test_survivors: tuple[int, ...] = (1_985, 2_712)

    def __post_init__(self) -> None:
        for name in ("threshold", "sensitivity_threshold"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"iwahashi.{name} {getattr(self, name)} must be in (0, 1]")
        if self.threshold == self.sensitivity_threshold:
            raise ValueError("iwahashi's sensitivity row must differ from the primary threshold")
        if len(self.test_survivors) != 2 or any(
            not isinstance(n, int) or n < 0 for n in self.test_survivors
        ):
            raise ValueError(
                f"iwahashi.test_survivors {self.test_survivors} must be two counts, at the "
                f"primary threshold then the sensitivity row"
            )


# --- §4.1-§4.3 The head recipes -----------------------------------------------


@dataclass(frozen=True)
class Linear(Frozen):
    """§4.1's linear probe: LayerNorm on `output`, a 1x1 conv to ten classes, weighted CE."""

    optimizer: str = "adamw"
    weight_decay: float = 0.0
    batch_size: int = 256
    epochs: int = 30
    #: Task 1's grid; tuned per rung on B6 (§4.5).
    lr_grid: tuple[float, ...] = (3e-4, 1e-3, 3e-3, 1e-2, 3e-2)

    def __post_init__(self) -> None:
        _positive("linear", batch_size=self.batch_size, epochs=self.epochs)
        _lr_grid("linear", self.lr_grid)


@dataclass(frozen=True)
class Upernet(Frozen):
    """§4.2's UperNet, PANGAEA's verbatim — neck, 512 channels, pool scales, layers (#196)."""

    channels: int = 512
    pool_scales: tuple[int, ...] = (1, 2, 3, 6)
    #: ``intermediates[3, 5, 7, 11]`` under ADR-0012 — blocks 4, 6, 8 and 12 of 12.
    layers: tuple[int, ...] = (3, 5, 7, 11)
    optimizer: str = "adamw"
    weight_decay: float = 0.05
    batch_size: int = 64
    epochs: int = 40
    #: Brackets PANGAEA's 1e-4 and the plan's earlier 6e-4; tuned per rung on B6 (§4.5).
    lr_grid: tuple[float, ...] = (3e-5, 1e-4, 3e-4, 1e-3, 3e-3)

    def __post_init__(self) -> None:
        _positive("upernet", channels=self.channels, batch_size=self.batch_size, epochs=self.epochs)
        _lr_grid("upernet", self.lr_grid)
        if any(not 0 <= i < 12 for i in self.layers):
            raise ValueError(f"upernet.layers {self.layers} must index ADR-0012's 12 blocks")


@dataclass(frozen=True)
class Unet(Frozen):
    """§4.3's B7: Task 1's A7 UNet on one channel, at B2's batch and epochs, tuning itself."""

    in_channels: int = 1
    #: A7's optimizer block, Task 1's §4.1 head recipe (``benchmark.config.Head``), carried here
    #: as values so the frozen line states them (#209). The lr is tuned; these are not (§4.5).
    optimizer: str = "adamw"
    weight_decay: float = 0.05
    batch_size: int = 64
    epochs: int = 40
    #: Centred on the 3e-3 A7 shared with Task 1's head.
    lr_grid: tuple[float, ...] = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)

    def __post_init__(self) -> None:
        _positive("unet", in_channels=self.in_channels, batch_size=self.batch_size,
                  epochs=self.epochs)
        _lr_grid("unet", self.lr_grid)


@dataclass(frozen=True)
class Finetune(Frozen):
    """§4.3's B3: a literature recipe used as given and **not tuned** (#196)."""

    lr: float = 1e-4
    layer_decay: float = 0.75
    head_lr: float = 1e-3
    epochs: int = 50
    precision: str = "bf16"

    def __post_init__(self) -> None:
        _positive("finetune", lr=self.lr, head_lr=self.head_lr, epochs=self.epochs)
        if not 0 < self.layer_decay <= 1:
            raise ValueError(f"finetune.layer_decay {self.layer_decay} must be in (0, 1]")


# --- §4.5 Schedule, val and tuning --------------------------------------------


@dataclass(frozen=True)
class Schedule(Frozen):
    """§4.5, shared by every trained head (#196)."""

    #: The step floor: no rung trains fewer steps than a 2,000-tile rung would.
    step_floor: int = 2_000
    warmup_frac: float = 0.05
    decay: str = "cosine"
    #: All val tiles, read at each tenth of the steps, the last step included; best val mIoU kept.
    val_reads: int = 10
    #: Tuning: this many log-spaced lr points ...
    tune_points: int = 5
    #: ... each trained for this fraction of its rung's schedule ...
    tune_fraction: float = 0.2
    #: ... at one seed, on this ladder.
    tune_seed: int = 0
    tune_ladder: int = 0

    def __post_init__(self) -> None:
        _positive("schedule", step_floor=self.step_floor, val_reads=self.val_reads,
                  tune_points=self.tune_points)
        if not 0 <= self.warmup_frac < 1 or not 0 < self.tune_fraction <= 1:
            raise ValueError("schedule.warmup_frac must be in [0, 1), tune_fraction in (0, 1]")
        if self.decay != "cosine":
            raise ValueError(f"schedule.decay {self.decay!r}: §4.5 is warmup then cosine to zero")


# --- §3.3 The geocell ladders --------------------------------------------------


@dataclass(frozen=True)
class Ladders(Frozen):
    """§3.3's geocell ladders (#200): how many, the seed each is walked under, the rung budgets.

    **Draw seeds.** The plan says only "ladder *i*'s draw seed", so the rule is this group's:
    ladder *i* is walked under ``first_draw_seed + i``, and ``first_draw_seed`` is the project
    seed plus one (0xDEAF = 57007, so 57007 / 57008 / 57009). Not the project seed itself,
    because that key already decided two things about these very tiles: it breaks the ties in
    which geocells went to test and which COGs to val (:func:`dataset.splits.
    select_test_geocells`, :func:`dataset.splits.select_val_cogs`), and it is Task 1's
    ``tiles.draw_seed``, the order the 20,000-tile head-train list was apportioned in. A ladder
    walked under it would rank train cells by the key that sorted cells *out* of train, and
    tiles by the order they entered the list. Each ladder's coverage report records its seed,
    and the resolver checks it. The project owner ratified this rule on 2026-11-09, before any
    Task 2 result existed, and the three ladders were committed under it then (#200). Moving it
    redraws every ladder, so it needs an amendment, not a ``--set``.

    **Budgets** are §3.3's 200 / 2,000 / 20,000 tiles, one per rung in :data:`RUNGS` order, and
    knobs only so the fixture corpus can walk the same rule at a scale it holds. The top one
    must be the whole head-train list ("at 100% every ladder is the whole list"); ``freeze``
    refuses a list of any other length rather than draw a 100% rung that is not 100%.

    :attr:`count` is how many ladders ``freeze`` writes. ADR-0015 L3 registers three seeds and
    binds seed *i* to ladder *i*, so a seed at or past ``count`` has no ladder and is refused. It
    is read off the register rather than held as a knob: a second home for a registered number
    is how one moves without anything failing (#199), and a ``--set ladders.count`` would make a
    fourth seed resolvable.

    The tile budgets are not in the register: L16 registers C2's rungs by name (1% and 10%), and
    its tile counts appear only in the statement's prose, so they stay knobs here.
    """

    first_draw_seed: int = dataset_config.RANDOM_SEED + 1
    budgets: tuple[int, ...] = tuple(RUNG_TILES[rung] for rung in RUNGS)

    @property
    def count(self) -> int:
        """L3's seeds, one ladder per seed (ADR-0015 L3; L16 counts the same ladders)."""
        return prereg.TASK2.rule("L3").value["seeds"]

    def __post_init__(self) -> None:
        if len(self.budgets) != len(RUNGS):
            raise ValueError(
                f"ladders.budgets {self.budgets} must hold one tile budget per rung {list(RUNGS)}"
            )
        if self.budgets[0] < 1 or any(a >= b for a, b in zip(self.budgets, self.budgets[1:])):
            raise ValueError(
                f"ladders.budgets {self.budgets} must ascend from >= 1: §3.3's rungs are nested "
                f"prefixes of one walk"
            )

    def draw_seed(self, ladder: int) -> int:
        """The seed ladder ``ladder`` is walked under — see the class docstring."""
        if not 0 <= ladder < self.count:
            raise ValueError(
                f"there is no ladder {ladder}; freeze writes ladders 0..{self.count - 1}"
            )
        return self.first_draw_seed + ladder

    def budget(self, rung: str) -> int:
        """How many tiles ``rung`` holds: a prefix of that many tiles of the cell's ladder."""
        return self.budgets[RUNGS.index(rung)]


# --- Where it all is, on this box ---------------------------------------------


@dataclass(frozen=True)
class LandformPaths(Frozen):
    """Every path Task 2 reads. Environment throughout, as Task 1's :class:`Paths` is.

    The encoder checkpoint fields carry Task 1's names and defaults, because an encoder's
    :attr:`benchmark.arms.Encoder.checkpoint_field` is looked up on whichever task's paths group
    is in hand — one registry, one field name per checkpoint. A checkpoint only Task 2 reads
    (B9's) is a field here alone, so Task 1's group does not change for it.
    """

    corpus_dir: str = Paths.corpus_dir
    #: Task 1's committed inputs Task 2 reuses unchanged: the tile lists (with #150's land
    #: filter) and ADR-0012's extractor spec.
    task1_artifacts_dir: str = Paths.artifacts_dir
    #: Task 2's own committed inputs: the bar, the ladders, the label fingerprints, the card.
    artifacts_dir: str = "artifacts/task2"
    #: Task 2's own parquet, so Task 1's schema and its cast-on-read are not touched.
    results_path: str = "data/task2/results.parquet"
    #: The plan, whose §7 Task 2's bar is anchored in (#199).
    plan_path: str = "docs/research/task2-landform-plan.md"
    #: The distributor's zip, as downloaded from gisstar.gsi.go.jp/terrain2018 (§3.2). Read in
    #: place, never committed; its md5 is ``iwahashi.zip_md5``, which is recipe (#203).
    iwahashi_zip: str = "data/iwahashi2018/7.5_arc_second_compressedGeoTIFF.zip"
    #: The tile-label table ``freeze`` rebuilds from it. Gitignored under §3.4's licence
    #: position; only its fingerprint is committed, in ``artifacts_dir`` (#203).
    iwahashi_table: str = "data/task2/iwahashi_labels.csv"
    demae_r5_checkpoint: str | None = Paths.demae_r5_checkpoint
    demae_r4_checkpoint: str | None = Paths.demae_r4_checkpoint
    demae_vits_checkpoint: str | None = Paths.demae_vits_checkpoint
    terramind_checkpoint: str | None = Paths.terramind_checkpoint
    imagenet_mae_checkpoint: str | None = Paths.imagenet_mae_checkpoint
    #: The WhiteboxTools v2.4.0 binary (#201). Its path is where it is on this box; its sha256
    #: is recipe and lives in ``geomorphon.wbt_sha256``.
    wbt_binary: str | None = None
    #: The GLO-30 release's present COG stems (gitignored, main checkout only).
    cog_frame: str = "data/cog_frame.json"
    #: Where the source COGs are read from: ``None`` is the public bucket, a directory holds
    #: ``{stem}/{stem}.tif`` on the same grid.
    cog_root: str | None = None
    #: The label rasters and the pass's resume log: rebuilt, gitignored, fingerprinted.
    labels_dir: str = "data/task2/labels"
    #: TerraMind's v1 base model, `TerraMind_v1_base.pt` — the file B9 reads (#205), a different
    #: file from the tokenizer above under a different pin. Task 2's alone: no Task 1 arm reads
    #: it, so Task 1's :class:`Paths` has no such field.
    terramind_backbone_checkpoint: str | None = None

    def __post_init__(self) -> None:
        check_paths(self)


@dataclass(frozen=True)
class Labelling(Frozen):
    """How the label pass runs on this box (#201) — environment, outside every hash.

    #187 measured reads at 6.4 windows/s on 16 threads against ~6 h sequential, so the pass
    reads and labels tiles on a thread pool and writes them from one thread.
    """

    threads: int = 16
    #: A progress line every this many tiles.
    report_every: int = 250

    def __post_init__(self) -> None:
        _positive("labelling", threads=self.threads, report_every=self.report_every)


def _runtime() -> Runtime:
    """Task 1's runtime group, with Task 2's per-step checkpoints kept apart from Task 1's."""
    return Runtime(checkpoint_dir="data/task2/heads")


# --- The config ---------------------------------------------------------------

#: Every group, in the order ``to_dict`` writes them. A later ticket adds its group here.
GROUPS: Final = {
    "geomorphon": Geomorphon,
    "iwahashi": Iwahashi,
    "linear": Linear,
    "upernet": Upernet,
    "unet": Unet,
    "finetune": Finetune,
    "schedule": Schedule,
    "ladders": Ladders,
    "extract": Extract,
    "runtime": Runtime,
    "paths": LandformPaths,
    "labelling": Labelling,
}

#: The cell. Recipe, all five.
SCALARS: Final = ("arm", "rung", "seed", "head", "label_set")

#: The groups ``freeze``'s committed inputs are a function of, and so the ones
#: :attr:`LandformConfig.artifact_hash` covers. Not the cell, and not the head recipes.
ARTIFACT_GROUPS: Final = ("geomorphon", "iwahashi", "ladders")


@dataclass(frozen=True)
class LandformConfig(Frozen):
    """One Task 2 cell's whole recipe. Defaults are B1 at 100%, seed 0, on geomorphons."""

    arm: str = "B1"
    rung: str = "100%"
    seed: int = 0
    #: ``None`` means "the arm's first head" and is filled in at construction — see the module
    #: docstring. After ``__post_init__`` it is always a name.
    head: str | None = None
    label_set: str = "geomorphon"
    geomorphon: Geomorphon = field(default_factory=Geomorphon)
    iwahashi: Iwahashi = field(default_factory=Iwahashi)
    linear: Linear = field(default_factory=Linear)
    upernet: Upernet = field(default_factory=Upernet)
    unet: Unet = field(default_factory=Unet)
    finetune: Finetune = field(default_factory=Finetune)
    schedule: Schedule = field(default_factory=Schedule)
    ladders: Ladders = field(default_factory=Ladders)
    extract: Extract = field(default_factory=Extract)
    runtime: Runtime = field(default_factory=_runtime, metadata=ENVIRONMENT)
    paths: LandformPaths = field(default_factory=LandformPaths, metadata=ENVIRONMENT)
    labelling: Labelling = field(default_factory=Labelling, metadata=ENVIRONMENT)

    def __post_init__(self) -> None:
        arm = landform_arms.get(self.arm)  # refuses, naming the table
        if self.rung not in RUNGS:
            raise ValueError(f"unknown rung {self.rung!r} (§3.3's tile budgets: {list(RUNGS)})")
        if self.seed < 0:
            raise ValueError(f"seed {self.seed} must be >= 0")
        if self.label_set not in LABEL_SETS:
            raise ValueError(f"unknown label set {self.label_set!r}; known: {list(LABEL_SETS)}")
        head = arm.heads[0] if self.head is None else self.head
        if head not in landform_arms.HEADS:
            raise ValueError(f"unknown head {head!r}; known: {list(landform_arms.HEADS)}")
        if head not in arm.heads:
            raise ValueError(
                f"arm {arm.name} trains {list(arm.heads)}, so head {head!r} is not a cell it "
                f"has — name an arm that trains it"
            )
        object.__setattr__(self, "head", head)
        if self.rung not in arm.rungs:
            raise ValueError(
                f"arm {arm.name} runs at {list(arm.rungs)} only, so rung {self.rung} is not a "
                f"cell it has (plan §4.3's budget)"
            )

    # --- derived numbers (computed, never stored, so they cannot drift)

    @property
    def train_tiles(self) -> int:
        """How many head-train tiles this cell's rung holds — its ladder's budget for the rung."""
        return self.ladders.budget(self.rung)

    @property
    def ladder(self) -> int:
        """The ladder this cell trains on. Seed *i* trains on ladder *i* (ADR-0015 L3)."""
        return self.seed

    @property
    def recipe(self) -> Linear | Upernet | Unet:
        """The recipe group of the head this cell trains."""
        return getattr(self, self.head)

    @property
    def fine_tunes(self) -> bool:
        """Whether this cell trains its encoder (B3), under :attr:`finetune` rather than a line."""
        return landform_arms.get(self.arm).trains == "encoder+head"

    @property
    def epochs(self) -> int:
        """The cell's epochs: its head recipe's, or §4.3's 50 for the fine-tune (B3), which
        shares §4.2's UperNet and batch but not its 40 epochs."""
        return self.finetune.epochs if self.fine_tunes else self.recipe.epochs

    @property
    def steps(self) -> int:
        """This cell's run length under §4.5's floor — see :func:`steps`."""
        return steps(
            epochs=self.epochs,
            batch_size=self.recipe.batch_size,
            n_train=self.train_tiles,
            floor=self.schedule.step_floor,
        )

    # --- serialization and the hash, through Task 1's functions

    def to_dict(self) -> dict:
        """The whole config as plain JSON types — five scalars, then group by group."""
        return config_to_dict(self, scalars=SCALARS, groups=GROUPS)

    @classmethod
    def from_dict(cls, d: dict) -> LandformConfig:
        """Rebuild from :meth:`to_dict`'s output, or from any subset of it."""
        return config_from_dict(cls, d, scalars=SCALARS, groups=GROUPS)

    def recipe_dict(self) -> dict:
        """:meth:`to_dict` minus every environment field — the view the hash covers."""
        return recipe_of(self)

    @property
    def config_hash(self) -> str:
        """A stable fingerprint of this cell's recipe, Task 1's construction, its own value."""
        return canonical_hash(self.recipe_dict())

    @property
    def artifact_hash(self) -> str:
        """The generator hash of what ``freeze`` writes — it does not move with the cell."""
        recipe = self.recipe_dict()
        return canonical_hash({name: recipe[name] for name in ARTIFACT_GROUPS})


# --- The override surface -----------------------------------------------------


def _apply(cfg: LandformConfig, partial: dict) -> LandformConfig:
    """``cfg`` with a partial config merged in — the head following a moved arm unless named."""
    for name in SCALARS:
        if isinstance(partial.get(name), dict):
            # `--set head.lr=3e-4` is Task 1's knob (its `head` is a recipe group). Here `head`
            # is the cell's scalar, and merging a group into it would fail as a TypeError that
            # names neither task.
            raise ValueError(
                f"{name} is one of Task 2's cell scalars {list(SCALARS)}, not a group of knobs; "
                f"got {partial[name]!r}. Task 2's head recipes are the `linear`, `upernet` and "
                f"`unet` groups (e.g. linear.epochs=30)"
            )
    merged = overlay(cfg.to_dict(), partial)
    if "arm" in partial and "head" not in partial:
        merged["head"] = None
    return LandformConfig.from_dict(merged)


def build_config(
    *,
    config_path: Path | str | None = None,
    arm: str | None = None,
    rung: str | None = None,
    seed: int | None = None,
    head: str | None = None,
    label_set: str | None = None,
    corpus: str | None = None,
    artifacts_dir: str | None = None,
    overrides: Sequence[str] | None = None,
) -> LandformConfig:
    """The plan's defaults, a partial config file, the cell flags, then ``--set`` — Task 1's order.

    Every layer goes through :meth:`LandformConfig.from_dict`, so an unknown group or knob
    raises before anything is resolved, and a Task 1 knob (``size``, ``head.lr``) is refused as
    an unknown group here exactly as a typo is.
    """
    cfg = LandformConfig()
    if config_path is not None:
        cfg = _apply(cfg, json.loads(Path(config_path).read_text()))
    cell = {
        name: value
        for name, value in (
            ("arm", arm), ("rung", rung), ("seed", seed), ("head", head), ("label_set", label_set)
        )
        if value is not None
    }
    if cell:
        cfg = _apply(cfg, cell)
    moved: dict[str, Any] = {}
    if corpus is not None:
        moved["corpus_dir"] = corpus
    if artifacts_dir is not None:
        moved["artifacts_dir"] = artifacts_dir
    if moved:
        cfg = cfg.replace(paths=cfg.paths.replace(**moved))
    for override in overrides or ():
        cfg = _apply(cfg, parse_override(override, scalars=SCALARS))
    return cfg


def config_from_args(args: argparse.Namespace) -> LandformConfig:
    """:func:`build_config` off the namespace ``benchmark.launch`` parsed — through Task 1's one
    keyword list (:func:`benchmark.config.build_kwargs`), with Task 2's cell scalars."""
    return build_config(**build_kwargs(args, scalars=SCALARS))

