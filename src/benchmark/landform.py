"""Task 2 behind ``python -m benchmark.launch --task landform`` — its resolution and its verbs (#198).

Not a second command: there is no ``main`` here, and ADR-0009 is why. :mod:`benchmark.launch`
owns the parser and the three verbs; ``--task landform`` hands it this module's
:func:`resolve` and :data:`VERBS` in place of Task 1's, so the same command, the same
``--dry-run`` and the same "resolve, say it, then act" order carry over unchanged.

## Where each later ticket plugs in

* **A committed input ``freeze`` writes** is a :class:`FreezeStage` in :data:`FREEZE_STAGES`, in
  plan §4.4's order. Each is pending (``write`` is ``None``) until its ticket supplies the
  writer; ``freeze`` runs every built stage in order and then refuses to report success past a
  pending one, naming it and its ticket.
* **A stage this box cannot run** because WhiteboxTools is absent (a declared external, ADR-0014
  D1) is skipped, and so is every later stage that ``needs`` it; the rest still run, and
  ``freeze`` then refuses, naming what it skipped and #202. A stage declares ``needs`` only on a
  stage whose output it reads.
* **An input a cell reads** that no ticket has built yet is a :class:`Pending` line in
  :data:`PENDING_INPUTS`. The ticket that builds it deletes the line and adds a
  :class:`benchmark.artifacts.Resolved` to :func:`_inputs`, so its path and md5 print instead.
* **A refusal about the cell** (seed bound to ladder, frozen recipe line, tuning arm) goes in
  :func:`resolve`, upstream of ``--dry-run``, for Task 1's reason: a pre-flight that prints a
  drifted cell as a normal one is worse than none.
* **``run`` and ``report``** name the ticket that builds them (#206, #216) until it does.

## Why the far side raises

Task 1's rule, from the four tickets in Phase 1 that it took to learn: a verb that returns 0
having done part of its job is read as a job done. ``freeze`` writing none of its stages and
exiting 0 would be read as a frozen Task 2. So past ``--dry-run`` every verb either finishes or
names what it is missing.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from . import (
    arms,
    artifacts,
    geomorphon_pass,
    iwahashi,
    ladders,
    landform_arms,
    landform_config,
    prereg,
)
from . import geomorphon as gm
from .artifacts import Resolved
from .landform_config import LandformConfig


@dataclass(frozen=True)
class Pending:
    """An input a Task 2 cell reads that no ticket has built yet — said, not omitted."""

    key: str
    label: str
    ticket: str

    def line(self) -> str:
        return f"{self.label:<16} not built yet ({self.ticket})"


@dataclass(frozen=True)
class FreezeStage:
    """One committed input ``freeze --task landform`` writes, and the ticket that builds it.

    ``write`` takes the resolution and returns the lines to print; ``None`` is a stage whose
    ticket has not landed.
    """

    label: str
    ticket: str
    write: Callable[[LandformResolution], list[str]] | None = None
    #: The labels of earlier stages whose committed output this one reads. A stage skipped on
    #: this box skips every stage that needs it.
    needs: tuple[str, ...] = ()


def _freeze_bar(resolution: LandformResolution) -> list[str]:
    """ADR-0015's bar, checked against §7 before a byte is written (:meth:`prereg.Register.write_bar`).

    Rewritten on every freeze, as Task 1's is: a bar is a pure function of its register, so a
    re-freeze writes the same bytes or an amendment that already passed all three links.
    """
    paths = resolution.config.paths
    register = prereg.TASK2
    bar = register.write_bar(bar_path(resolution.config), plan=paths.plan_path)
    doc = register.document()
    return [
        f"wrote {bar}",
        f"  {register.adr_name}'s register: {doc['rules']} rules, {doc['mechanised']} with a "
        f"number, each anchored in {register.section} of {paths.plan_path}",
    ]


def bar_path(cfg: LandformConfig) -> Path:
    """Where Task 2's bar sits on this box."""
    return Path(cfg.paths.artifacts_dir) / prereg.TASK2.bar_name


def bar_for(cfg: LandformConfig) -> prereg.LandformBar:
    """Task 2's committed bar, refused at load if §7 no longer states what it registered."""
    return prereg.TASK2.read_bar(bar_path(cfg), plan=cfg.paths.plan_path)


#: What ``freeze --task landform`` writes, in §4.4's order: the bar first (every threshold the
#: later stages are drawn against is in it, Task 1's reason), then the labels (§4.4 steps 1-2),
#: then the ladders (their coverage reports carry per-class pixel counts, so labels come first),
#: then the card, which describes everything above it and so goes last.
#:
#: The ladders do not ``need`` the geomorphon stage yet: their per-class pixel counts are null
#: until #202 measures them (:data:`benchmark.ladders.CLASS_PIXELS_NOTE`). The change that fills
#: them in adds ``needs=("geomorphon labels",)`` to the ladder stage.
FREEZE_STAGES: Final[tuple[FreezeStage, ...]] = (
    FreezeStage("prereg bar", "#199", _freeze_bar),
    FreezeStage("geomorphon labels", "#201", geomorphon_pass.write),
    FreezeStage("iwahashi labels", "#203", iwahashi.freeze),
    FreezeStage("ladders", "#200", ladders.freeze),
    FreezeStage("dataset card", "#218"),
)

#: What a cell reads that is not built yet. Story 4's whole cell is arm, encoder pin, rung,
#: ladder, label set, frozen recipe line and bar; these are the parts with no artifact yet.
PENDING_INPUTS: Final[tuple[Pending, ...]] = (
    Pending("recipe_line", "recipe line", "#207"),
)


@dataclass(frozen=True)
class LandformResolution:
    """Everything one Task 2 command resolved: the cell, the arm, and every input by content."""

    config: LandformConfig
    arm: landform_arms.LandformArm
    #: The arm's encoder checkpoint, or ``None`` for an arm that runs no encoder.
    checkpoint: Resolved | None
    inputs: tuple[Resolved, ...]
    pending: tuple[Pending, ...] = PENDING_INPUTS

    @property
    def encoder(self) -> arms.Encoder | None:
        return None if self.arm.encoder is None else arms.ENCODERS[self.arm.encoder]

    def lines(self, *, verb: str) -> list[str]:
        """The resolution block, for the person about to spend money on this command.

        Task 1's layout and for Task 1's reasons: ``run`` names a cell and prints its config
        hash; ``freeze`` and ``report`` have no cell, so ``freeze`` prints the generator hash
        that does not move with ``--arm`` and only ``run``/``report`` name the parquet.
        """
        cfg = self.config
        if verb == "run":
            block = [
                f"DEMAE Task 2 — run   arm {cfg.arm}, rung {cfg.rung} "
                f"({cfg.train_tiles:,} tiles), seed {cfg.seed}, head {cfg.head}, "
                f"label set {cfg.label_set}   config {cfg.config_hash}",
                f"{'':<16} {self.arm.what}",
                "",
                *self.arm_lines(),
                ladders.rung_line(cfg),
                "",
            ]
        else:
            block = [
                f"DEMAE Task 2 — {verb}   artifacts {cfg.artifact_hash} "
                f"(the label generator hash)",
                "",
            ]
        block += [item.line() for item in self.inputs]
        if verb == "run":
            block += [item.line() for item in self.pending]
        if verb == "freeze":
            block += ["", *(_stage_line(stage) for stage in FREEZE_STAGES)]
        else:
            block.append(f"{'results':<16} {cfg.paths.results_path}")
        return block

    def arm_lines(self) -> list[str]:
        """Which encoder and pin, which checkpoint, and the head recipe with its step count."""
        cfg = self.config
        encoder = self.encoder
        if encoder is None:
            lines = [f"{'encoder':<16} none — this arm runs no encoder"]
        else:
            lines = [
                f"{'encoder':<16} {encoder.name}, pinned md5 {encoder.checkpoint_md5}",
                self.checkpoint.line(),
            ]
        recipe = cfg.recipe
        schedule = cfg.schedule
        lines += [
            f"{'trains':<16} {self.arm.trains} — the {cfg.head} head",
            f"{'schedule':<16} {cfg.steps:,} steps = {recipe.epochs} epochs x "
            f"ceil(max({cfg.train_tiles:,} tiles, {schedule.step_floor:,} floor) / batch "
            f"{recipe.batch_size}); {schedule.warmup_frac:.0%} warmup then {schedule.decay} to "
            f"zero; val read {schedule.val_reads}x, best val mIoU kept",
            f"{'lr':<16} tuned per rung over {list(recipe.lr_grid)} (§4.5)",
        ]
        return lines


def _stage_line(stage: FreezeStage) -> str:
    state = "built" if stage.write is not None else "not built yet"
    return f"{'freeze':<16} {stage.label} — {state} ({stage.ticket})"


def _checkpoint(arm: landform_arms.LandformArm, cfg: LandformConfig) -> Resolved | None:
    """The arm's encoder checkpoint as this box has it — or that it has not mounted one."""
    if arm.encoder is None:
        return None
    return artifacts.encoder_checkpoint(cfg.paths, arms.ENCODERS[arm.encoder].checkpoint_field)


def _ladders(cfg: LandformConfig, verb: str) -> tuple[Resolved, ...]:
    """The ladders a command reads (#200): ``run``'s cell reads its seed's one; ``freeze`` and
    ``report`` see all of them. :func:`benchmark.ladders.resolved` refuses a walk filed under
    the wrong ladder."""
    if verb == "run":
        return (ladders.resolved(cfg, ladders.bind(cfg), key="ladder", label="ladder"),)
    ladders.bind(cfg)
    return tuple(
        ladders.resolved(cfg, i, key=f"ladder_{i}", label=f"ladder {i}", verify=verb != "freeze")
        for i in range(cfg.ladders.count)
    )


def _inputs(cfg: LandformConfig) -> tuple[Resolved, ...]:
    """The committed inputs a cell reads that exist today: Task 1's, reused unchanged, and
    Task 2's own bar.

    §3.3 reuses Task 1's train, val and test lists (with #150's land filter) and ADR-0012's
    extractor contract, so they are read from Task 1's artifacts directory and frozen by Task
    1's ``freeze``. The ladders draw their rungs from the full head-train list. The bar is
    ADR-0015's, frozen by ``freeze --task landform`` into Task 2's own directory (#199).
    """
    task1 = Path(cfg.paths.task1_artifacts_dir)
    at = {
        "head_train_tiles": ("head-train list", task1 / artifacts.TILE_LIST_NAMES["20k"]),
        "val_tiles": ("val list", task1 / artifacts.VAL_TILES_NAME),
        "test_tiles": ("test list", task1 / artifacts.TEST_TILES_NAME),
        "extractor_spec": ("extractor spec", task1 / "extractor_spec.json"),
    }
    task1_inputs = tuple(
        Resolved(
            key=key,
            label=label,
            path=str(path),
            md5=artifacts.file_md5(path),
            absent_note="Task 1's, not frozen yet (`python -m benchmark.launch freeze`)",
        )
        for key, (label, path) in at.items()
    )
    bar = bar_path(cfg)
    # The geomorphon label config: its md5 moves with the label config hash, the census, every
    # variant's set fingerprint and the class weights (#201).
    geomorphon = Path(cfg.paths.artifacts_dir) / geomorphon_pass.CONFIG_NAME
    return task1_inputs + (
        Resolved(
            key="prereg_bar",
            label="prereg bar",
            path=str(bar),
            md5=artifacts.file_md5(bar),
            absent_note="not frozen yet (`python -m benchmark.launch freeze --task landform`)",
        ),
        Resolved(
            key="geomorphon_config",
            label="geomorphon",
            path=str(geomorphon),
            md5=artifacts.file_md5(geomorphon),
            absent_note="not frozen yet (`freeze --task landform`, #201/#202)",
        ),
    )


def resolve(args: argparse.Namespace) -> LandformResolution:
    """The config, then the arm, then every input — the config first, so a typo'd override
    raises before a single file is stat'd."""
    cfg = landform_config.config_from_args(args)
    if args.verb == "run" and getattr(args, "tune", False):
        raise ValueError(
            "`run --task landform --tune` is §4.5's per-head, per-rung lr tuning, which #207 "
            "builds; it is not Task 1's single A6 pass and does not run under that one's rules"
        )
    arm = landform_arms.get(cfg.arm)
    inputs = (*_inputs(cfg), *_ladders(cfg, args.verb))
    if cfg.label_set == "iwahashi":
        # #203 built this label set: its fingerprint and rebuilt table print, and `run` refuses
        # a table the fingerprint does not describe.
        inputs += iwahashi.inputs(cfg, verb=args.verb)
    return LandformResolution(
        config=cfg,
        arm=arm,
        checkpoint=_checkpoint(arm, cfg),
        inputs=inputs,
    )


def _freeze(resolution: LandformResolution, args: argparse.Namespace) -> int:
    """``freeze --task landform`` — every built stage this box can run, in order, then a refusal
    past a pending or skipped one.

    A stage that raises :class:`benchmark.geomorphon.WbtUnavailable` is skipped, with every
    later stage that ``needs`` it, and the stages that read none of its output still run: a box
    without WhiteboxTools can freeze the bar, the Iwahashi labels and the ladders. It still
    never exits 0 (ADR-0009): the refusal names what was skipped and #202.
    """
    skipped: dict[str, str] = {}  # stage label -> why
    cause: Exception | None = None
    for stage in FREEZE_STAGES:
        if stage.write is None:
            continue
        blocked = [need for need in stage.needs if need in skipped]
        if blocked:
            skipped[stage.label] = f"needs {', '.join(blocked)}"
            print(f"{'freeze':<16} {stage.label} — skipped: needs {', '.join(blocked)}")
            continue
        try:
            lines = stage.write(resolution)
        except gm.WbtUnavailable as exc:
            skipped[stage.label] = str(exc)
            cause = cause or exc
            print(f"{'freeze':<16} {stage.label} — skipped: WhiteboxTools is not on this box "
                  f"(#202)")
            continue
        for line in lines:
            print(line)
    pending = [f"{stage.label} ({stage.ticket})" for stage in FREEZE_STAGES if stage.write is None]
    if skipped:
        listing = "; ".join(f"{label}: {why}" for label, why in skipped.items())
        raise FileNotFoundError(
            f"`freeze --task landform` skipped {len(skipped)} stage(s) this box cannot run — "
            f"{listing}. It wrote the stages that do not need WhiteboxTools (#202) and stops "
            f"here rather than exit 0 as though Task 2 were frozen"
            + (f"; no writer yet for: {', '.join(pending)}" if pending else "")
        ) from cause
    if pending:
        raise NotImplementedError(
            f"`freeze --task landform` has no writer yet for: {', '.join(pending)}. It wrote "
            f"what it has and stops here rather than exit 0 as though Task 2 were frozen"
        )
    return 0


def _run(resolution: LandformResolution, args: argparse.Namespace) -> int:
    """``run --task landform`` — one (arm, rung, seed) cell. #206 builds the first one."""
    raise NotImplementedError(
        "`run --task landform` trains and scores nothing yet: the linear-probe cell, the Task 2 "
        "parquet and the run record are #206. `--dry-run` resolves the cell"
    )


def _report(resolution: LandformResolution, args: argparse.Namespace) -> int:
    """``report --task landform`` — the core table and the claims block. #216 builds it."""
    raise NotImplementedError(
        "`report --task landform` renders nothing yet: the core table and ADR-0015's claims "
        "block are #216, and there is no Task 2 parquet for it to read until #206"
    )


#: Task 2's three verbs, the same surface as :data:`benchmark.launch.VERBS`.
VERBS: Final = {"freeze": _freeze, "run": _run, "report": _report}
