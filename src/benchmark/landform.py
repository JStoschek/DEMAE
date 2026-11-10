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
* **A refusal about the cell** (seed bound to ladder, frozen recipe line, tuning arm) goes in
  :func:`resolve`, upstream of ``--dry-run``, for Task 1's reason: a pre-flight that prints a
  drifted cell as a normal one is worse than none.
* **``run``** trains and scores one cell on geomorphon labels: the linear probe (#206) or
  UperNet (#208) over a frozen encoder, or B7's from-scratch UNet (#209,
  :mod:`benchmark.landform_unet`). A pre-flight that refuses by name, then the head's trainer
  (:data:`TRAINERS`), then :mod:`benchmark.landform_results`. A head or label set no ticket has
  built names its ticket.
* **``run --tune``** is §4.5's lr pass for one head at one rung (#207): the resolver fills in
  and checks the head's tuning arm (B6 for the probes, B7 for itself) before the config is
  built, then :mod:`benchmark.landform_tune` trains the grid and freezes the winner as the
  head's recipe line, with a record of every pass beside it.
  **``report``** names #216 until it lands.

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
    landform_recipes,
    landform_results,
    landform_train,
    landform_tune,
    landform_unet,
    prereg,
    results,
)
from . import geomorphon as gm
from .artifacts import Resolved
from .landform_config import LandformConfig


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
FREEZE_STAGES: Final[tuple[FreezeStage, ...]] = (
    FreezeStage("prereg bar", "#199", _freeze_bar),
    FreezeStage("geomorphon labels", "#201", geomorphon_pass.write),
    FreezeStage("iwahashi labels", "#203", iwahashi.freeze),
    FreezeStage("ladders", "#200", ladders.freeze, needs=("geomorphon labels",)),
    FreezeStage("dataset card", "#218"),
)

@dataclass(frozen=True)
class LandformResolution:
    """Everything one Task 2 command resolved: the cell, the arm, and every input by content."""

    config: LandformConfig
    arm: landform_arms.LandformArm
    #: The arm's encoder checkpoint, or ``None`` for an arm that runs no encoder.
    checkpoint: Resolved | None
    inputs: tuple[Resolved, ...]
    #: This cell's frozen recipe line, already checked against the cell, or ``None`` while
    #: #207's tuning pass has not frozen it. ``run`` refuses the ``None``.
    recipe_line: landform_recipes.RecipeLine | None = None
    #: ``run --tune``: this resolution is §4.5's tuning pass for the cell's head and rung (#207).
    tuning: bool = False

    def input(self, key: str) -> Resolved:
        return next(item for item in self.inputs if item.key == key)

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
                f"DEMAE Task 2 — {'run --tune' if self.tuning else 'run'}   arm {cfg.arm}, rung {cfg.rung} "
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
            if self.tuning:
                block.append(
                    f"{'tuning':<16} {cfg.head} @ {cfg.rung} over {list(cfg.recipe.lr_grid)}, "
                    f"{landform_tune.tune_steps(cfg):,} steps per point "
                    f"({cfg.schedule.tune_fraction:g} of {cfg.steps:,}), better of "
                    f"{landform_tune.VAL_READS} val reads; the winner is frozen as this head's "
                    f"line at this rung (§4.5)"
                )
            else:
                block.append(
                    self.recipe_line.line() if self.recipe_line is not None else
                    f"{'recipe line':<16} {cfg.head} @ {cfg.rung} not frozen yet "
                    f"(`run --task landform --tune`, #207)"
                )
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
            f"{'lr':<16} "
            + (f"{self.recipe_line.lr:g}, the frozen line for this head and rung (§4.5)"
               if self.recipe_line is not None
               else f"tuned per rung over {list(recipe.lr_grid)} (§4.5)"),
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
        Resolved(
            key="recipe_lines",
            label="recipe lines",
            path=str(landform_recipes.path_for(cfg)),
            md5=artifacts.file_md5(landform_recipes.path_for(cfg)),
            absent_note="not frozen yet (`run --task landform --tune`, #207)",
        ),
    )


def _tuning_args(args: argparse.Namespace) -> argparse.Namespace:
    """``run --tune``'s cell flags with §4.5's tuning arm filled in, refusing any other arm.

    The head names the pass; the arm follows it (B6 for the linear probe and UperNet, B7 for
    itself), and a named arm that is not that one is refused here, before the config is built,
    so the refusal is about the tuning arm and not about which heads that arm trains today.
    """
    head, arm = args.head, args.arm
    if head is None:
        if arm is None:
            raise ValueError(
                f"`run --task landform --tune` tunes one head at one rung: name it with "
                f"--head ({list(landform_recipes.TUNING_ARMS)})"
            )
        head = landform_arms.get(arm).heads[0]
    want = landform_recipes.TUNING_ARMS.get(head)
    landform_recipes.assert_tuning_arm(want if arm is None else arm, head)
    return argparse.Namespace(**{**vars(args), "head": head, "arm": want})


def resolve(args: argparse.Namespace) -> LandformResolution:
    """The config, then the arm, then every input — the config first, so a typo'd override
    raises before a single file is stat'd. Under ``run --tune`` the tuning arm is checked even
    before the config (:func:`_tuning_args`)."""
    tuning = args.verb == "run" and getattr(args, "tune", False)
    if tuning:
        args = _tuning_args(args)
    cfg = landform_config.config_from_args(args)
    if tuning:
        landform_tune.check_tuning_knobs(cfg)
        if args.seed is None:
            cfg = cfg.replace(seed=cfg.schedule.tune_seed)
        if cfg.seed != cfg.schedule.tune_seed:
            raise ValueError(
                f"§4.5 tunes at one seed on ladder {cfg.schedule.tune_ladder} "
                f"(schedule.tune_seed {cfg.schedule.tune_seed}); --seed {cfg.seed} is not it"
            )
        frozen = landform_recipes.lookup(cfg)
        if frozen is not None:
            raise ValueError(
                f"{landform_recipes.path_for(cfg)} already freezes {cfg.head} @ {cfg.rung} at lr "
                f"{frozen.lr:g}. A frozen line is not re-tuned by running the pass again: "
                f"re-tuning is an amendment, whose first act is deleting that line"
            )
    arm = landform_arms.get(cfg.arm)
    # §4.5's frozen line, checked here rather than in `run`, for Task 1's reason: a pre-flight
    # that prints a drifted recipe as a normal one is worse than none. An absent line is said,
    # and `run` refuses it. A tuning pass has no line yet: it is what writes one.
    recipe_line = landform_recipes.lookup(cfg) if args.verb == "run" and not tuning else None
    if recipe_line is not None:
        landform_recipes.check(cfg, recipe_line)
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
        recipe_line=recipe_line,
        tuning=tuning,
    )


def _freeze(resolution: LandformResolution, args: argparse.Namespace) -> int:
    """``freeze --task landform`` — every built stage this box can run, in order, then a refusal
    past a pending or skipped one.

    A stage that raises :class:`benchmark.geomorphon.WbtUnavailable` is skipped, with every
    later stage that ``needs`` it, and the stages that read none of its output still run: a box
    without WhiteboxTools can freeze the bar and the Iwahashi labels. It still
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


def _preflight(resolution: LandformResolution, rerun_reason: str | None) -> dict:
    """Everything ``run`` checks before its first step, each refusal by name; what it found.

    The encoder's pin and checkpoint, the capacity table, the extractor contract, the bar, the
    frozen recipe line, the ladder md5, and L20's write-once rule. The label fingerprint is
    checked next, once the cell's tiles are read (:func:`benchmark.geomorphon_pass.check_labels`).
    """
    cfg = resolution.config
    line = resolution.recipe_line
    if line is None:
        landform_recipes.line_for(cfg)  # raises, naming #207
    params, spec_item, checkpoint = _preflight_encoder(resolution)
    encoder = resolution.encoder
    ladder = resolution.input("ladder")
    cell = {
        "arm": cfg.arm, "head": cfg.head, "rung": cfg.rung, "seed": cfg.seed,
        "ladder": cfg.ladder,
        "label_set": cfg.label_set, "variant": gm.PRIMARY,
    }
    attempt = landform_results.attempt_for(cfg.paths.results_path, cell, rerun_reason)
    for text in (
        *_encoder_lines(cfg, encoder, checkpoint, params, spec_item),
        f"recipe line {line.head} @ {line.rung}: lr {line.lr:g}, the recipe as frozen",
        f"ladder {cfg.ladder} md5 {ladder.md5}, as its coverage report quotes",
        f"write-once: attempt {attempt}"
        + (f", a rerun for {rerun_reason} (L20)" if rerun_reason else ", a new cell"),
    ):
        print(f"{'pre-flight':<16} {text}")
    return {"cell": cell, "params": params, "line": line, "ladder_md5": ladder.md5,
            "spec_md5": spec_item.md5}


def _encoder_lines(cfg, encoder, checkpoint, params, spec_item) -> tuple[str, ...]:
    if encoder is None:
        return (
            f"encoder none: {cfg.arm} trains everything from scratch, "
            f"{landform_unet.trunk_parameters():,} parameters (Task 1's A7 spec, one channel)",
            f"bar {bar_path(cfg)}: {prereg.TASK2.adr_name}'s, its phrases still in plan "
            f"{prereg.TASK2.section.lstrip('# ')}",
        )
    return (
        f"encoder {encoder.name}: checkpoint md5 {checkpoint.md5} is its pin; {params:,} "
        f"parameters, as §2's capacity table gives",
        f"extractor contract {spec_item.path} re-checked",
        f"bar {bar_path(cfg)}: {prereg.TASK2.adr_name}'s, its phrases still in plan "
        f"{prereg.TASK2.section.lstrip('# ')}",
    )


def _preflight_encoder(resolution: LandformResolution
                       ) -> tuple[int, Resolved, Resolved | None]:
    """The checks ``run`` and ``run --tune`` share: every input present, the encoder pinned and
    its checkpoint the pinned one, the capacity table, the extractor contract and the bar.

    An arm that runs no encoder (B7, #209) has no pin, checkpoint or capacity row to check: it
    gets the inputs and the bar, 0 encoder parameters and no checkpoint."""
    from . import extract

    cfg = resolution.config
    # A tuning pass is what writes the recipe lines, so it is the one command that may run
    # before they exist.
    missing = [
        item.label for item in resolution.inputs
        if item.required and not item.present
        and not (resolution.tuning and item.key == "recipe_lines")
    ]
    if missing:
        raise FileNotFoundError(
            f"nothing to train against — {', '.join(missing)} not on this box. Run "
            f"`python -m benchmark.launch freeze` (Task 1's lists and extractor spec) and "
            f"`freeze --task landform` first"
        )
    encoder = resolution.encoder
    if encoder is None:
        bar_for(cfg)
        return 0, resolution.input("extractor_spec"), None
    pinned = encoder.checkpoint_md5
    if pinned is None:
        raise ValueError(
            f"the {encoder.name} encoder is unpinned: `benchmark.arms.ENCODERS` publishes no md5 "
            f"for it, and an unpinned encoder is unrunnable (ADR-0012 §7.1, plan §1)"
        )
    checkpoint = resolution.checkpoint
    if not checkpoint.present:
        raise FileNotFoundError(
            f"arm {cfg.arm} reads the {encoder.name} encoder and {checkpoint.absent_note}"
        )
    if checkpoint.md5 != pinned:
        raise ValueError(
            f"the {encoder.name} checkpoint at {checkpoint.path} has md5 {checkpoint.md5}, and "
            f"its pin is {pinned}: a mismatched encoder returns a plausible tensor and raises "
            f"nothing, so it is refused (ADR-0012 §7.1)"
        )
    params = arms.check_capacity(encoder.name)
    spec_item = resolution.input("extractor_spec")
    extract.load_spec(spec_item.path).encoder_named(encoder.name)
    bar_for(cfg)
    return params, spec_item, checkpoint


def _tune(resolution: LandformResolution, args: argparse.Namespace) -> int:
    """``run --task landform --tune`` — §4.5's lr pass for one head at one rung, frozen as its
    line (#207). Scores nothing on test and appends nothing to the parquet."""
    from dataset.provenance import git_sha

    cfg = resolution.config
    if cfg.label_set != "geomorphon":
        raise NotImplementedError(
            "the Iwahashi probe is fitted, not trained, and its L2 pass is #212; `--tune` "
            "tunes the dense heads on geomorphon labels"
        )
    params, spec_item, checkpoint = _preflight_encoder(resolution)
    for text in _encoder_lines(cfg, resolution.encoder, checkpoint, params, spec_item):
        print(f"{'pre-flight':<16} {text}")
    tiles = landform_train.CellTiles.read(cfg)
    labels = geomorphon_pass.check_labels(cfg, [*tiles.train, *tiles.val])
    print(f"{'pre-flight':<16} labels: {labels['tiles_checked']:,} {labels['variant']} rasters "
          f"match their fingerprints, set {labels['set_fingerprint']}")
    tuned = landform_tune.run_pass(
        cfg, tiles=tiles,
        on_point=lambda p: print(f"{'':<16} lr {p['lr']:<10g} val mIoU {p['score']:.4f} "
                                 f"(better of {len(p['reads'])} reads)"),
    )
    line_path, record = landform_tune.freeze(tuned, git_sha=git_sha())
    print()
    for text in tuned.lines():
        print(text)
    print(f"{'froze':<16} {cfg.head} @ {cfg.rung} at lr {tuned.selected:g} in {line_path}")
    print(f"{'record':<16} {record}")
    print(f"{'scored':<16} nothing — the tuning pass reads val only; every arm's {cfg.head} "
          f"cell at {cfg.rung} now trains at this line")
    return 0


def _run(resolution: LandformResolution, args: argparse.Namespace) -> int:
    """``run --task landform`` — one (arm, rung, seed) cell: pre-flight, train, score, append.

    #206 built the linear-probe cell on geomorphon labels, #208 put UperNet through the same
    path and #209 runs B7's UNet. The Iwahashi classifier (#212) names its ticket.
    """
    from dataset.provenance import git_sha

    if resolution.tuning:
        return _tune(resolution, args)
    cfg = resolution.config
    if cfg.label_set != "geomorphon":
        raise NotImplementedError(
            "the Iwahashi tile classifier and its fitted probe are #212; `run --task landform` "
            "trains the dense geomorphon cells"
        )
    if cfg.head not in TRAINERS:
        raise NotImplementedError(
            f"the {cfg.head} head has no trainer; built: {sorted(TRAINERS)}"
        )
    rerun_reason = getattr(args, "rerun_reason", None)
    found = _preflight(resolution, rerun_reason)
    tiles = landform_train.CellTiles.read(cfg)
    labels = geomorphon_pass.check_labels(cfg, tiles.every)
    print(f"{'pre-flight':<16} labels: {labels['tiles_checked']:,} {labels['variant']} rasters "
          f"match their fingerprints, set {labels['set_fingerprint']}")
    print(f"{'trains':<16} {cfg.steps:,} steps over {len(tiles.train):,} tiles; val "
          f"{len(tiles.val):,} tiles read {cfg.schedule.val_reads}x; test {len(tiles.test):,} "
          f"tiles")

    line = found["line"]
    fitted = TRAINERS[cfg.head]()(
        cfg, lr=line.lr, tiles=tiles, on_read=lambda read: print(f"{'':<16} {read.line()}")
    )
    scored = landform_train.score_test(cfg, fitted, tiles)
    total = sum(tile.confusion for tile in scored)
    collapsed = landform_train.class_collapse(total)
    rows = [
        {
            **found["cell"],
            "tile_id": tile.tile_id,
            "band": tile.band,
            "stratum": tile.stratum,
            "confusion": tile.confusion.tolist(),
            **tile.boundary,
            "config_hash": cfg.config_hash,
        }
        for tile in scored
    ]
    record = {
        "task": "landform",
        "cell": found["cell"],
        "config_hash": cfg.config_hash,
        "artifact_hash": cfg.artifact_hash,
        "config": cfg.recipe_dict(),
        "git_sha": git_sha(),
        "encoder": None if resolution.encoder is None else {
            "name": fitted.encoder,
            "pinned_md5": resolution.encoder.checkpoint_md5,
            "checkpoint": resolution.checkpoint.to_dict(),
        },
        "extractor_spec_md5": found["spec_md5"],
        "labels": labels,
        "ladder_md5": found["ladder_md5"],
        "recipe_line": line.to_dict(),
        "inputs": {item.key: item.to_dict() for item in resolution.inputs},
        "parameters": {
            "encoder": found["params"],
            "trainable": fitted.parameters,
            "total": found["params"] + fitted.parameters,
        },
        "class_collapse": {"count": len(collapsed), "classes": collapsed},
        "timing": {
            "head": cfg.head,
            "seconds_per_step": round(fitted.seconds_per_step, 6),
            "steps": fitted.steps,
            "device": fitted.device,
        },
        "training": fitted.to_dict(),
        "precision": fitted.precision,
        "produced": {"test_miou": round(landform_train.miou(total), 6)},
    }
    path, count, record = landform_results.write(
        rows, record, cfg.paths.results_path, rerun_reason=rerun_reason
    )
    print(f"\n{'scored':<16} {len(rows):,} test tiles into {path} ({count:,} rows in the file); "
          f"test mIoU {record['produced']['test_miou']:.4f}, best val read at step "
          f"{fitted.best.step}")
    print(f"{'collapse':<16} {len(collapsed)} class(es) present in test truth at IoU 0"
          + (f": {', '.join(collapsed)} (flagged, not excluded — L10)" if collapsed else ""))
    print(f"{'params':<16} {found['params']:,} frozen + {fitted.parameters:,} trained")
    print(f"{'record':<16} run {record['run_id']} appended to "
          f"{results.run_record_path(path)}")
    return 0


#: The trainer for each built head, looked up when the cell runs (so a test's stand-in is seen).
TRAINERS: Final = {
    "linear": lambda: landform_train.train_head,
    "upernet": lambda: landform_train.train_head,
    "unet": lambda: landform_unet.train_unet,
}


def _report(resolution: LandformResolution, args: argparse.Namespace) -> int:
    """``report --task landform`` — the core table and the claims block. #216 builds it."""
    raise NotImplementedError(
        "`report --task landform` renders nothing yet: the core table and ADR-0015's claims "
        "block are #216, and there is no Task 2 parquet for it to read until #206"
    )


#: Task 2's three verbs, the same surface as :data:`benchmark.launch.VERBS`.
VERBS: Final = {"freeze": _freeze, "run": _run, "report": _report}
