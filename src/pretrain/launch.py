"""``python -m pretrain.launch`` — the documented way to start a run (#95).

Everything before this file is a piece that works. This is the command a rented box actually
invokes, and what it adds to #93's ``python -m pretrain.train`` is the three things you only
want on a box you are paying for:

- **The full override surface.** A config file plus repeatable ``--set group.knob=value``, so
  an ablation is a command line rather than a file per arm, and so a run can be pinned from
  the shell when the file on the box is not the file you meant.
- **The levers, confirmed rather than configured.** Before the first step it prints what the
  built model will *do* — §7.2's attention path read off the timm blocks, §7.3's compile state,
  the device, the parameter split — and it refuses to start when that disagrees with the
  recipe. §7's levers all fail by being silently absent (see :mod:`pretrain.runtime`).
- **The counters, while it runs.** Every step line carries samples/s, step time and the loader
  wait fraction; every epoch and the run as a whole are summarized the same way. §7.5's gate —
  *measured samples/s at least 2x model demand, in the first rented hour* — is read off these
  lines, and R0 (#81) is the ticket that reads it.

``--dry-run`` does all of the assembling and none of the spending: the header, the fingerprint
and the step count, with nothing built and no corpus read past its manifest. That is the form
the §15 pre-flight checklist is meant to be walked in.

## What this does not own

Checkpoints and resume are #94's, and the per-epoch loss reporting, the alarms and the W&B
sink are #97's — both hang off :func:`~pretrain.train.train`'s ``on_step`` seam, which is why
the reporting here stays deliberately plain. What is printed is what a human watching a
terminal needs in order to decide whether to let the run continue.
"""

from __future__ import annotations

import argparse
import json

from .config import RunConfig
from .fingerprint import fingerprint_for
from .header import add_config_arguments, build_config, format_header
from .runtime import CompileWatch, attention_path, eager_module
from .throughput import ThroughputSummary, format_throughput, summarize
from .train import StepRecord, Trainer, build_loader, format_step, steps_per_epoch_for, train


def resolve_steps(cfg: RunConfig, loader, *, steps: int | None, epochs: int | None) -> int:
    """How many optimizer steps this invocation runs — from ``--steps``, ``--epochs``, or the plan.

    Asked for nothing, a launcher runs **the config's own schedule to its end**. The default
    matters: one that quietly ran a single epoch would turn §6's 400 into a documentation
    detail, and R1 would be launched by remembering to pass a number.

    ``--epochs`` is converted through :func:`~pretrain.train.steps_per_epoch_for`, so it counts
    the steps the split *actually yields* — micro-batches over the accumulation factor, with
    the remainder dropped — rather than a number derived from the corpus size on paper.
    """
    if steps is not None:
        return steps
    return (cfg.schedule.epochs if epochs is None else epochs) * steps_per_epoch_for(loader, cfg)


def runtime_line(model, cfg: RunConfig, *, device) -> str:
    """§7's levers as the built model will really run them, printed before step 1.

    Every number here is read off the *model*, not off the recipe that asked for it — which is
    the point of printing it at all. The attention path has already been **refused** if it
    disagreed, by :meth:`~pretrain.train.Trainer.from_config`; by the time this runs it is a
    fact to report rather than a check to make.

    The compile state is reported as configured because there is nothing to read yet — the
    graph is traced on the first forward. What happens *after* that is watched instead, by
    :class:`~pretrain.runtime.CompileWatch`, and reported the moment it happens.
    """
    counts = eager_module(model).parameter_counts()
    compile_state = (
        f"on (dynamic={cfg.runtime.compile_dynamic})" if cfg.runtime.compile else "off (eager)"
    )
    return (
        f"runtime          {cfg.runtime.precision}, "
        f"attention {attention_path(model)} (confirmed on the built blocks), "
        f"compile {compile_state}, device {device}\n"
        f"                 {counts['encoder'] / 1e6:.1f}M encoder + "
        f"{counts['decoder'] / 1e6:.1f}M decoder parameters"
    )


def format_step_line(record: StepRecord, total: int) -> str:
    """#93's step line, with §7.5's three counters after it."""
    line = format_step(record, total)
    if record.throughput is None:
        return line
    return f"{line}   {format_throughput(record.throughput)}"


def format_epoch_line(summary: ThroughputSummary) -> str:
    """One completed epoch's throughput — the row #97's dashboard plots."""
    return (
        f"epoch {summary.epoch:4d}  {summary.steps:5d} steps   "
        f"{format_throughput(summary)}"
    )


def format_recompile_warning(frames: int, step: int) -> str:
    """What to say when §7.3's precondition breaks, mid-run and again in the footer.

    A recompile loop is the one §7 failure that keeps costing money for as long as the run
    lasts, so it is said as soon as :class:`~pretrain.runtime.CompileWatch` sees it — a footer
    on a 400-epoch run arrives days after the point at which someone would have killed it.
    """
    return (
        f"WARNING: {frames} frame(s) recompiled after warmup (first seen by step {step + 1}). "
        "§7.3 expects none — the mask count is fixed (#90) and short batches are dropped "
        "(#93), so a shape or a guard is moving, and compiling is costing this run rather "
        "than paying for it. `--set runtime.compile=false` is the cheaper fallback until it "
        "is understood"
    )


def main(argv: list[str] | None = None) -> int:
    """Assemble the config, confirm the levers, run the loop, and report what it achieved."""
    p = add_config_arguments(
        argparse.ArgumentParser(
            prog="python -m pretrain.launch",
            description="Launch a DEMAE Phase-1 pretraining run (#95).",
        )
    )
    length = p.add_mutually_exclusive_group()
    length.add_argument("--steps", type=int, default=None, help="optimizer steps to run")
    length.add_argument(
        "--epochs", type=int, default=None, help="epochs to run (default: the config's schedule)"
    )
    p.add_argument("--device", default=None, help="cuda | cpu | … (default: cuda if present)")
    p.add_argument("--log-every", type=int, default=10, help="steps between printed lines")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print the header, the levers and the step count; train nothing",
    )
    args = p.parse_args(argv)

    cfg = build_config(
        config_path=args.config,
        vit_base=args.vit_base,
        corpus=args.corpus,
        assignment=args.assignment,
        blocks=args.blocks,
        overrides=args.overrides,
    )
    print(format_header(cfg, fingerprint_for(cfg)))
    print()

    # The loader is built before anything is spent on a model: it is what says how many steps
    # an epoch is, and #89's split containment check (plan §15.2) fires at its construction.
    loader = build_loader(cfg, cfg.data.train_split)
    steps = resolve_steps(cfg, loader, steps=args.steps, epochs=args.epochs)
    print(f"running          {steps} steps ({steps_per_epoch_for(loader, cfg)} per epoch)")
    if args.dry_run:
        print("\n--dry-run: nothing was trained")
        return 0

    watch = CompileWatch()
    # Accumulated as it goes, never accrued: R1 is ~195,000 optimizer steps, and a footer
    # computed by re-summing a list of all of them — or merely by keeping one — would be a
    # memory leak inside the thing measuring the run's health. #97's sink is what records the
    # per-step history off-instance; this prints and forgets.
    run = steady = summarize([])
    epoch = summarize([])
    first_loss = last_loss = None
    warned = False

    def announce(trainer: Trainer) -> None:
        print(runtime_line(trainer.model, cfg, device=trainer.device))
        print()

    def close_epoch() -> None:
        """Print the epoch that has just ended, if any steps landed in it."""
        if epoch.steps:
            print(format_epoch_line(epoch), flush=True)

    def report(record: StepRecord) -> None:
        nonlocal run, steady, epoch, first_loss, last_loss, warned
        counters = record.throughput  # always set: `train` times every step it takes
        if record.step == 0:
            # Step 0 *is* the compile (§7.3). Anything counted after this point is a recompile,
            # which is the one way the compile lever turns into a loss rather than a win.
            watch.reset()
            first_loss = record.loss
            epoch = summarize([], epoch=counters.epoch)
        elif counters.epoch != epoch.epoch:
            # An epoch's line is only correct once its last step has landed, which is observable
            # only when the next epoch's first step arrives. The final epoch is closed below.
            close_epoch()
            epoch = summarize([], epoch=counters.epoch)
        last_loss = record.loss
        epoch = epoch.plus(counters)
        run = run.plus(counters)
        if record.step > 0:
            steady = steady.plus(counters)
        if record.step % args.log_every == 0 or record.step + 1 == steps:
            print(format_step_line(record, steps), flush=True)
        if watch.compiled and not warned:
            # Said the moment it happens rather than in the footer. A recompile loop is the one
            # §7 failure that costs money continuously, and a 400-epoch run's footer arrives
            # days after the point at which someone would have killed it.
            warned = True
            print(format_recompile_warning(watch.compiled, record.step), flush=True)

    train(cfg, steps=steps, device=args.device, on_step=report, on_start=announce, loader=loader)

    close_epoch()
    print()
    if first_loss is not None:
        print(f"{run.steps} steps: loss {first_loss:.4f} -> {last_loss:.4f}")
    print(f"run              {format_throughput(run)}")
    if steady.steps:
        # The first step carries `torch.compile`'s whole compilation, the worker pool's startup
        # and the page cache's first fill. §7.5's gate is about the other 400 epochs.
        print(
            f"steady           {format_throughput(steady)}"
            "   (excluding step 1: compile + loader warmup)"
        )
    if watch.compiled:
        print(f"\n{format_recompile_warning(watch.compiled, run.steps - 1)}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
