"""``python -m benchmark.launch`` — the one command that freezes, runs or reports (#147).

ADR-0009's rule carried into Phase 2, for the reason it was written: *a seam that nobody is on
both sides of is a feature that does not exist*. Phase 1 discovered that the hard way — every
piece built and tested, nothing composing them, and a 400-epoch run launched by the documented
command producing no checkpoints, no W&B run and none of §10's instruments. The defence is one
command, and a test that drives ``main(argv)`` rather than one that drives each piece.

Three verbs:

```bash
uv run python -m benchmark.launch freeze              # write every committed input
uv run python -m benchmark.launch run --arm A4-R5 --size 20k --seed 0
uv run python -m benchmark.launch report              # the printed table, the C4 block, the figures
```

``benchmark.train`` and any per-stage ``main`` that appears during the build are documented as
**not a way to produce a result**, exactly as ``pretrain.train`` is.

## What is built, and what is not

#147 shipped the command surface, the frozen run config, the resolution line and ``--dry-run``:
the pre-flight a build session walks before renting anything. #148 gave ``freeze`` the first of
its five artifacts, the registered bar (:mod:`benchmark.prereg`), #149 the second, §3.1's tile
lists (:mod:`benchmark.tiles`), #156 the third, ADR-0012's extractor contract
(:mod:`benchmark.extract`), and #150 the fourth, §3.2's eval mask list
(:mod:`benchmark.masks`). #158 made ``run`` real for **A1**: §2.4's classical fill over the
committed eval mask list, both of §8's norm-inversion gates, and the rows and run record that
come out the other side (:mod:`benchmark.evaluate`, :mod:`benchmark.results`).

The two verbs that are still short of their stage raise past the work they have, and name what
has not been built — deliberately, because a verb that quietly returned 0 having done part of
its job is precisely the shape of failure this module exists to prevent, and it is the shape
that survived four tickets in Phase 1. It matters more once a verb writes real files than it did
when none did: a ``freeze`` that lands three of five artifacts and exits 0 is read as a frozen
box. ``run`` returns 0 now because it finishes what it starts *for the arm it was asked for*;
an arm whose input convention has no implementation yet names its own ticket rather than
returning quietly (:data:`benchmark.evaluate.FILLERS`).

## Why the resolution line is printed before anything is spent

Every failure it makes visible is silent. An encoder handed the wrong checkpoint returns a
plausible tensor; a head trained against a re-drawn mask bank converges normally; a table
computed against a stale eval mask list prints exactly as wide as a good one. So the run says
which encoder, which checkpoint md5, which mask bank, which spec and how many instances, before
it takes a step — and :meth:`~benchmark.artifacts.Resolution.to_dict` records the same facts
beside the table they produced.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final, NoReturn

from . import arms, artifacts, evaluate, extract, masks, prereg, results, tiles
from .artifacts import Resolution
from .config import LABEL_EFFICIENCY_SIZES, add_config_arguments, config_from_args


def resolve(args: argparse.Namespace) -> Resolution:
    """Assemble the config, the arm and every input this cell reads — in that order.

    The order is what makes *"a typo'd override stops before the corpus is touched"* true
    rather than nearly true: the config is built first, so an unknown knob raises before a
    single file is stat'd, let alone hashed.
    """
    cfg = config_from_args(args)
    return artifacts.resolve(cfg, arms.get(cfg.arm))


def format_resolution(resolution: Resolution, *, verb: str) -> str:
    """The resolution block as one string.

    The layout belongs to :meth:`~benchmark.artifacts.Resolution.lines`, beside the facts it is
    laying out; what is here is the console's join and nothing else.
    """
    return "\n".join(resolution.lines(verb=verb))


def _unbuilt(verb: str, state: str, produces: str) -> NoReturn:
    """The far side of the seam, out loud.

    Named rather than absent, and raising rather than returning 0, because the alternative is a
    command that reports success having done part of its job — which is the failure ADR-0009
    exists to close and the one this repo has a name for. It matters most now that one verb has
    started writing real artifacts: a `freeze` that lands three of five files and exits 0 is read
    as a frozen box.
    """
    raise NotImplementedError(
        f"`{verb}` {state}. What belongs here is {produces} — see the stage tickets under the "
        f"Task 1 map (#145) and `docs/task1-build-spec.md`."
    )


#: The three verbs take one resolution and return an exit code. ``run`` is the first one that
#: does (#158) — `freeze` still stops short of §3.4's train mask bank and `report` renders
#: nothing yet, so those two are annotated ``NoReturn`` and say which stage is missing. The
#: uniform parameter is what keeps :data:`VERBS` a table rather than a cascade.


def _freeze(resolution: Resolution) -> NoReturn:
    """`freeze` — write every committed input. §4.5 step 1, CPU-only and local.

    Four of the five are built, and the order is each ticket's own argument rather than the
    order they happened to land in.

    The **registered bar** goes first (#148) for the reason ADR-0013 gives: every threshold the
    later stages are drawn against is in it, and it refuses to be written at all once a rule it
    registers has drifted from plan §6. A freeze that would produce artifacts measured against a
    bar the plan no longer states should not reach the artifacts.

    **§3.1's tile lists** go next (#149), ahead of the two mask artifacts, because those are a
    function of this draw — the bank is K draws per head-train tile and the eval mask list is per
    test tile — so a stage that fails later leaves behind exactly the inputs the next attempt
    reproduces byte-for-byte.

    **ADR-0012's extractor contract** goes next (#156), ahead of the one that is owed, because it
    is the one committed input whose *other* reader is Task 2: a contract that only appeared once
    the mask bank did would be gated on work the task inheriting it does not depend on. Unlike
    the bar it carries §3.4's guard, refusing to overwrite a version of itself whose content hash
    has moved.

    **§3.2's eval mask list** goes last of the four (#150), because it is a function of the test
    list two stages above it — 30,000 rows is 5,000 test tiles x two synthetic families x three
    buckets — and it reads that list back off disk rather than out of the draw, on the contract
    family's rule that the committed file is what later phases re-read. It carries the same §3.4
    guard the spec does, and the same generator config hash #151's bank will.
    """
    paths = resolution.config.paths
    bar = prereg.write_bar(Path(paths.artifacts_dir) / prereg.BAR_NAME, plan=paths.plan_path)
    print(f"\nwrote {bar}")
    print(
        "  ADR-0013's register: twelve rules, seven with a number, each anchored in "
        f"{prereg.SECTION} of {paths.plan_path}"
    )
    for line in tiles.format_written(tiles.freeze(resolution.config), paths.artifacts_dir):
        print(line)
    spec = extract.write_spec(resolution.config)
    print(f"wrote {spec}")
    print(
        f"  ADR-0012's contract, md5 {artifacts.file_md5(spec)} — raw int16 metres in, "
        f"{extract.LAYERS} block outputs plus a separate `output` out, re-checked at load"
    )
    for line in masks.format_written(masks.freeze(resolution.config), resolution.config.masks):
        print(line)
    _unbuilt(
        "freeze",
        f"wrote {prereg.BAR_NAME}, §3.1's tile lists, {extract.SPEC_FILENAME} and "
        f"{artifacts.EVAL_MASKS_NAME}, and stopped — the last of the freeze is not built",
        "the train mask bank, written beside them — K = 50 draws per head-train tile from the "
        "same generator and the same config hash, kept out of git and committed as an md5 "
        "sidecar `run` refuses a mismatch against (§3.4, #151)",
    )


def _run(resolution: Resolution) -> int:
    """`run` — execute exactly one (arm, size, seed) cell, having refused a box it cannot.

    §4.5's step 2, and the first verb that produces a number. The order below is the argument:

    1. **The box, before the clock.** A run cannot be scored against inputs that are not there,
       and discovering that after an encoder has loaded is discovering it on the rented clock.
       Both refusals are about *this box* rather than about what is built. What counts as a
       missing input is the **arm's** question (:attr:`~benchmark.artifacts.Resolution.missing`),
       which is what lets A1 run ahead of §3.4's train mask bank.
    2. **The two §8 gates, before a single instance.** ``fill(identity-mask) == input`` and the
       golden tile, both inside :func:`benchmark.evaluate.run_pass` so that no caller of the
       pass can route around them. A gate consulted after 30,000 instances is a gate that costs
       2.0–5.2 CPU-hours to ask.
    3. **The rows, then the record.** :func:`benchmark.results.write` lands both, in that order,
       for the reason stated there.

    A1's numbers gate everything else (§4.5, user story 29), so the arms that are not built name
    their ticket rather than failing as an unknown key — :data:`benchmark.evaluate.FILLERS`.
    """
    missing = resolution.missing
    if missing:
        raise FileNotFoundError(
            f"nothing to score against — {', '.join(missing)} "
            f"{'is' if len(missing) == 1 else 'are'} not on this box. Run "
            f"`python -m benchmark.launch freeze` first (§4.5 step 1; it is CPU-only and local)."
        )
    checkpoint = resolution.checkpoint
    if checkpoint is not None and not checkpoint.present:
        raise FileNotFoundError(
            f"arm {resolution.arm.name} reads the {resolution.arm.encoder} encoder and "
            f"{checkpoint.absent_note}"
        )

    cfg = resolution.config
    scored = evaluate.run_pass(cfg, resolution.arm)
    print(f"\n{'gates':<16} {evaluate.format_gates(scored.gates)}")

    rows = results.rows_from(
        scored.scored,
        arm=cfg.arm,
        init=resolution.arm.encoder,
        size=cfg.size,
        seed=cfg.seed,
        run=results.run_id(resolution),
        config_hash=cfg.config_hash,
    )
    path, total, record = results.write(
        rows,
        resolution,
        metrics=scored.bar.registered_metrics,
        seconds=scored.seconds,
        parameters=arms.parameter_counts(resolution.arm),
        gates=scored.gates,
    )
    for line in results.format_written(path, rows=len(rows), total=total, record=record):
        print(line)
    return 0


def _report(resolution: Resolution) -> NoReturn:
    """`report` — render the table from the results parquet, re-running no arm."""
    _unbuilt(
        "report",
        "renders nothing yet",
        "the 396-number printed table, the 66-cell C4 rank block and the three figures, read "
        "from the results parquet and printing nothing R5 does not permit it to print (#136)",
    )


#: The whole command surface. A fourth verb is an argparse refusal, and a second entry point
#: with its own ``main`` would be the ``pretrain.monitor`` shape all over again (ADR-0009 §2).
VERBS: Final = {"freeze": _freeze, "run": _run, "report": _report}


def build_parser() -> argparse.ArgumentParser:
    """The command. Split out from :func:`main` so ``--help`` is one readable thing to review."""
    parser = argparse.ArgumentParser(
        prog="python -m benchmark.launch",
        description="Freeze, run or report the Task 1 void-filling benchmark (#147).",
    )
    verbs = parser.add_subparsers(dest="verb", required=True, metavar="{freeze,run,report}")

    def add(name: str, help_text: str) -> argparse.ArgumentParser:
        sub = add_config_arguments(verbs.add_parser(name, help=help_text))
        sub.add_argument(
            "--dry-run",
            dest="dry_run",
            action="store_true",
            help="assemble the config, the artifacts, the extractor and the arm; write nothing",
        )
        return sub

    add("freeze", "write every committed input (CPU-only and local)")
    run = add("run", "execute exactly one (arm, size, seed) cell")
    # The cell, as flags — and they are config knobs rather than a second recipe: `--arm A5` is
    # `--set arm=A5`, applied in the same place, so an arm stays "this object with a knob moved".
    run.add_argument(
        "--arm", default=None, help=f"one of {sorted(arms.ARMS)} (default: A4-R5)"
    )
    run.add_argument(
        "--size", default=None, help=f"label-efficiency subset: {list(LABEL_EFFICIENCY_SIZES)}"
    )
    run.add_argument("--seed", type=int, default=None, help="§4.1's seed for this cell")
    add("report", "render the printed table, the C4 rank block and the three figures")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Resolve what this command would act on, say it, and then act on it."""
    args = build_parser().parse_args(argv)
    resolution = resolve(args)
    print(format_resolution(resolution, verb=args.verb))
    if args.dry_run:
        print("\n--dry-run: nothing was written")
        return 0
    return VERBS[args.verb](resolution)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
