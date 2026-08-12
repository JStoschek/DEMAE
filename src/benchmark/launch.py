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

## What this ticket shipped, and what it did not

The command surface, the frozen run config, the resolution line and ``--dry-run``: the
pre-flight a build session walks before renting anything. **Nothing is scored here.** Past
``--dry-run`` every verb raises and names the stage that has not been built yet — deliberately,
because a verb that quietly returned 0 having done nothing is precisely the shape of failure
this module exists to prevent, and it is the shape that survived four tickets in Phase 1.

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
from typing import Final, NoReturn

from . import arms, artifacts
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


def _unbuilt(verb: str, does: str, produces: str) -> NoReturn:
    """The far side of the seam, out loud.

    Named rather than absent, and raising rather than returning 0, because the alternative is a
    command that reports success having done nothing — which is the failure ADR-0009 exists to
    close and the one this repo has a name for.
    """
    raise NotImplementedError(
        f"`{verb}` {does} yet: #147 shipped the command surface, the frozen run config and the "
        f"resolution line, and nothing behind them. What belongs here is {produces} — see the "
        f"stage tickets under the Task 1 map (#145) and `docs/task1-build-spec.md`. "
        f"`--dry-run` is the whole of what this command does today."
    )


#: The three verbs take one resolution and return an exit code — the shape they will have once
#: they are built. Today every one of them raises instead, so ``NoReturn`` is the honest
#: annotation and the uniform parameter is what keeps :data:`VERBS` a table rather than a
#: cascade. `freeze` and `report` do not read the resolution yet; the command has already
#: printed it.


def _freeze(resolution: Resolution) -> NoReturn:
    """`freeze` — write every committed input. §4.5 step 1, CPU-only and local."""
    _unbuilt(
        "freeze",
        "writes nothing",
        "the four tile lists, the eval mask list, the train mask bank, `extractor_spec.json` "
        "and `task1-prereg-bar.json` — idempotent, and refusing to overwrite an artifact whose "
        "content hash has moved (§3.4)",
    )


def _run(resolution: Resolution) -> NoReturn:
    """`run` — execute exactly one (arm, size, seed) cell, having refused a box it cannot."""
    # Both refusals come before the stage refusal, and both are about *this box* rather than
    # about what is built: a run cannot be scored against inputs that are not there, and
    # discovering that after the encoder has loaded is discovering it on the clock.
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
    _unbuilt(
        "run",
        "scores nothing",
        "exactly one (arm, size, seed) cell, appending its rows to the results parquet, having "
        "asserted the extractor contract, the registered bar and the split containment on the "
        "way in (ADR-0012, ADR-0013)",
    )


def _report(resolution: Resolution) -> NoReturn:
    """`report` — render the table from the results parquet, re-running no arm."""
    _unbuilt(
        "report",
        "renders nothing",
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
