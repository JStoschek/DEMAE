"""``python -m pretrain.launch`` — the one command a run is started with (#95, #106).

Everything before this file is a piece that works. This is the command a rented box actually
invokes, and what it adds to #93's ``python -m pretrain.train`` is the four things you only
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
- **Survival, and the instruments.** §11's checkpoints on both triggers, resume from them, and
  §10's whole panel — the val pass, the alarms, §4's per-patch distribution, §10.2's grids,
  §10.3's probe, and the W&B run they all go to. See below; this is #106.

``--dry-run`` does all of the assembling and none of the spending: the header, the fingerprint,
the step count, where the checkpoints will go and what the sink is, with nothing built and no
corpus read past its manifest. That is the form the §15 pre-flight checklist is walked in.

## Why this is the *only* command that trains (#106)

It was not, and the gap was the last thing standing between the codebase and R1. Each piece was
built and tested — #94's ``train_resumable``, #97's ``EpochMonitor``, #98's grids, #99's probe —
and nothing composed them, so this entrypoint called plain :func:`~pretrain.train.train` and a
400-epoch run launched by the documented command would have produced **no checkpoints** (with
``sync_checkpoints.sh`` faithfully reporting ``ok`` over an empty directory for forty hours),
**no W&B run**, and none of §10's instruments. There was a second training command,
``python -m pretrain.monitor``, which had the panel and no checkpoints — and which accepted
``--set`` and threw it away, so the two arms that *are* ``--set`` ablations (R2 #83, R3 #84)
would have run R1's recipe and read out as "no difference".

So: one command, composing all of it, and ``pretrain.monitor`` is a library rather than a run
path. The composition itself is :func:`~pretrain.monitor.train_epochs`; what lives here is the
command — the flags, the console, and the two decisions a run needs made before it starts.

**Where the checkpoints go.** ``runs/<run_id>``, unless ``--checkpoint-dir`` says otherwise.
Derived rather than defaulted to a fixed path because two arms writing into one directory is
how R2 resumes R1's weights, and the run id is the thing that already distinguishes them.
``--no-checkpoints`` is the opt-out and belongs to laptops and tests; on a rented box it is
§11's whole survival argument switched off.

**Resume is the default.** vast.ai instances vanish permanently, so a first launch and a
relaunch are the same command, typed on a fresh box — and a flag that had to be remembered at
exactly that moment is a flag that will be missing. An empty directory is a fresh start rather
than an error, and what was found is printed before anything is spent. ``--no-resume`` forces a
new run past an existing directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .alarms import Alarm
from .checkpoint import DEFAULT_INTERVAL_SECONDS, CheckpointPolicy, resolve_resume
from .config import RunConfig
from .fingerprint import fingerprint_for
from .header import add_config_arguments, config_from_args, format_header
from .monitor import EpochRecord, format_epoch, train_epochs
from .probe import format_probe
from .runtime import CompileWatch, attention_path, eager_module
from .sink import open_run_logger
from .throughput import format_throughput, summarize
from .train import StepRecord, Trainer, build_loader, format_step, steps_per_epoch_for


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


def checkpoint_directory(cfg: RunConfig, given: str | None) -> Path:
    """Where §11's checkpoints go: ``--checkpoint-dir``, else ``runs/<run_id>``.

    Derived from the run id rather than fixed, because the alternative is every arm writing
    into one directory — and with resume on by default (see the module docstring) that is R2
    silently continuing R1's weights. The run id is the thing that already distinguishes two
    arms, it is what §15.7 tags the W&B run with, and ``--set run_id=R2`` is how an ablation
    is launched, so the directory follows the ablation without anyone remembering a second flag.
    """
    return Path(given) if given is not None else Path("runs") / cfg.run_id


def describe_checkpoints(policy: CheckpointPolicy | None) -> str:
    """The §11 line: where they go, on what triggers, and what is protected — or that none go.

    Printed at the top of a run, and again in the footer when there is a policy — because the
    failure it exists to make visible is silent at both ends. R0 (#81) found the shipping
    command writing none at all while ``sync_checkpoints.sh`` reported ``ok`` over an empty
    directory every pass; naming the directory is what makes that noticeable in the first
    minute rather than at teardown. The footer repeats it so the last thing on screen after
    forty hours is where to look, and it is skipped under ``--no-checkpoints``, where the
    header line has already said so and a second "none" is noise.
    """
    if policy is None:
        return "none — --no-checkpoints (§11's survival argument is off)"
    minutes = (
        "never"
        if policy.interval_seconds is None
        else f"every {policy.interval_seconds / 60:.0f} min"
    )
    return (
        f"{policy.directory}  (every epoch and {minutes}; "
        f"keep last {policy.keep_last} + milestones {list(policy.milestones)})"
    )


def describe_sink(cfg: RunConfig) -> str:
    """Where §10's numbers go — the W&B mode and project, or that the sink is off."""
    m = cfg.monitoring
    if m.wandb_mode == "disabled":
        return "wandb disabled — nothing leaves this process (§11)"
    entity = "" if m.wandb_entity is None else f"{m.wandb_entity}/"
    return f"wandb {m.wandb_mode}, {entity}{m.wandb_project}, run {cfg.run_id}"


def format_recompile_warning(frames: int, step: int) -> str:
    """The **first** train-path recompile: said, located, and left alone (#109).

    A recompile loop is the one §7 failure that keeps costing money for as long as the run
    lasts, so it is said as soon as :class:`~pretrain.runtime.CompileWatch` sees it — a footer
    on a 400-epoch run arrives days after the point at which someone would have killed it.

    **It does not recommend turning compile off**, and that is the correction #109 bought. One
    frame is a one-time compilation, ~60 s; ``runtime.compile=false`` would trade that minute
    for R0's measured ~2.4x across every remaining epoch. The flag belongs to the *loop* case,
    which is :func:`format_recompile_escalation`, and the only honest discriminator between the
    two is whether the count grows again. So this one states the fact and waits.
    """
    return (
        f"NOTE: the training step compiled {frames} frame(s) after warmup (first seen at step "
        f"{step + 1}). §7.3 expects none — the mask count is fixed (#90) and short batches are "
        "dropped (#93), so a shape or a guard is moving. A single compile is a one-time cost "
        "and compile is worth ~2.4x, so this is worth understanding, not worth switching off. "
        "If the count grows again this line will say so"
    )


def format_recompile_escalation(frames: int, previous: int, step: int) -> str:
    """The second one — the count is still climbing, which is the case the flag is for.

    Printed once and then never again: a genuine per-step loop would otherwise put a warning
    beside every one of R1's 92,400 step lines, and the pre-#109 code avoided that only by
    never speaking a second time at all, which is also why it could not tell the two cases
    apart.
    """
    return (
        f"WARNING: the training step's compile count grew again, {previous} -> {frames}, by "
        f"step {step + 1}. It is still climbing, so this is a recompile loop rather than a "
        "one-time cost, and compiling is now costing this run rather than paying for it. "
        "`--set runtime.compile=false` is the cheaper fallback until it is understood. No "
        "further recompile lines will be printed"
    )


def format_compile_summary(
    train: int, monitoring: int, first_step: int | None, *, enabled: bool = True
) -> str:
    """What a finished run leaves on screen about §7.3 — an outcome, not a replayed warning.

    The footer knows the one thing the inline message could not: whether it recurred. #109's
    footer instead reprinted the mid-run text with the *current* step substituted for the
    first-seen one, so a run whose only compile happened at step 232 signed off saying step
    92,400 — the opposite end of the run from where it happened. ``first_step`` is therefore
    recorded when it is seen and carried here, never recomputed.

    The healthy line is printed too, rather than only the unhappy one. A run that says
    ``0 after warmup`` is a run nobody has to reopen this ticket to interpret, and the
    monitoring count is the number R1, R2 and R3 all reported as 18.

    ``enabled=False`` is §7.3's lever switched off, where the counts are zero because nothing
    was compiled. Reporting "§7.3 holds" there would be true and useless — the claim is about
    a compiled graph — and it is exactly the reading that would let an eager run pass for a
    fast one, which is the failure mode this whole module exists to close.
    """
    if not enabled:
        return "compile          off (runtime.compile=false) — §7.3's lever is not in play"
    monitored = (
        f"{monitoring} frame(s) at the first epoch boundary (§10's val pass and grids, expected)"
        if monitoring
        else "none"
    )
    if not train:
        return (
            f"compile          training step: 0 frames after warmup (§7.3 holds)\n"
            f"                 monitoring:    {monitored}"
        )
    where = "" if first_step is None else f", first at step {first_step + 1}"
    return (
        f"compile          training step: {train} frame(s) after warmup{where} — §7.3 does NOT "
        f"hold\n                 monitoring:    {monitored}"
    )


def build_parser() -> argparse.ArgumentParser:
    """The command. Split out from :func:`main` so `--help` is one readable thing to review."""
    p = add_config_arguments(
        argparse.ArgumentParser(
            prog="python -m pretrain.launch",
            description="Launch a DEMAE Phase-1 pretraining run (#95, #106).",
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
        help="print the header, the levers, the step count and the destinations; train nothing",
    )
    p.add_argument(
        "--checkpoint-dir",
        default=None,
        help="§11 checkpoints go here (default: runs/<run_id>)",
    )
    p.add_argument(
        "--checkpoint-minutes",
        type=float,
        default=DEFAULT_INTERVAL_SECONDS / 60,
        help="wall-clock checkpoint interval beside the per-epoch one; §11 asks for 30-60",
    )
    p.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="write none — a laptop or a test, never a rented box (§11)",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="start fresh even if --checkpoint-dir already holds a run",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Assemble the config, confirm the levers, run the loop, and report what it achieved."""
    args = build_parser().parse_args(argv)

    cfg = config_from_args(args)
    fingerprint = fingerprint_for(cfg)
    print(format_header(cfg, fingerprint))
    print()

    # The loader is built before anything is spent on a model: it is what says how many steps an
    # epoch is, and #89's split containment check (plan §15.2) fires at its construction. It is
    # then dropped — the run builds its own inside the loop, because the DataLoader generator a
    # checkpoint records has to be the one the loop is actually reading through (§11).
    loader = build_loader(cfg, cfg.data.train_split)
    steps = resolve_steps(cfg, loader, steps=args.steps, epochs=args.epochs)
    print(f"running          {steps} steps ({steps_per_epoch_for(loader, cfg)} per epoch)")
    del loader

    directory = checkpoint_directory(cfg, args.checkpoint_dir)
    policy = (
        None
        if args.no_checkpoints
        else CheckpointPolicy.from_config(
            cfg, directory=directory, interval_seconds=args.checkpoint_minutes * 60
        )
    )
    print(f"checkpoints      {describe_checkpoints(policy)}")
    print(f"monitoring       {describe_sink(cfg)}")
    # Said before the first step, not discovered in a curve afterwards: a relaunch that silently
    # started over, or one that silently continued the wrong run, both look like a normal run.
    # There is nothing to resume from without a policy, and `resume` is forced off rather than
    # passed as a flag the run would ignore.
    resume = policy is not None and not args.no_resume
    if policy is not None:
        found = resolve_resume(policy.directory, resume)
        print(
            "resuming         (nothing to resume from — this is a first launch)"
            if found is None
            else f"resuming         {found}"
        )
    if args.dry_run:
        print("\n--dry-run: nothing was trained")
        return 0

    watch = CompileWatch()
    # Accumulated as it goes, never accrued: R1 is ~195,000 optimizer steps, and a footer
    # computed by re-summing a list of all of them — or merely by keeping one — would be a
    # memory leak inside the thing measuring the run's health. #97's sink is what records the
    # per-step history off-instance; this prints and forgets.
    run = steady = summarize([])
    first_loss = last_loss = None
    started = False
    # #109: the count at which the last recompile line was printed, the absolute step the first
    # train-path frame was seen at, and whether the escalation has been spent. `warned_at` is
    # what the footer reads — recorded where it happens, so the footer cannot substitute the
    # current step for it the way the pre-#109 one did.
    warned_at: int | None = None
    warned_frames = 0
    escalated = False

    def announce(trainer: Trainer) -> None:
        print(runtime_line(trainer.model, cfg, device=trainer.device))
        print()

    def report(record: StepRecord) -> None:
        nonlocal run, steady, first_loss, last_loss, started
        nonlocal warned_at, warned_frames, escalated
        counters = record.throughput  # always set: the loop times every step it takes
        if not started:
            # The watch's own baseline is taken by `EpochMonitor.on_step`, which runs before
            # this callback: the warmup convention lives with the watch rather than being
            # re-implemented by every caller that holds one.
            started = True
            first_loss = record.loss
        else:
            steady = steady.plus(counters)
        last_loss = record.loss
        run = run.plus(counters)
        # The first line of *this* process, the cadence, and the run's last step — the middle
        # one is an absolute step index, so a resumed run's lines stay on the same grid the
        # killed one's were on rather than restarting the count.
        if run.steps == 1 or record.step % args.log_every == 0 or record.step + 1 == steps:
            print(format_step_line(record, steps), flush=True)
        # Said the moment it happens rather than in the footer. A recompile loop is the one §7
        # failure that costs money continuously, and a 400-epoch run's footer arrives days after
        # the point at which someone would have killed it. `watch.compiled` is the *training
        # step's* frames alone (#109) — §10's boundary compiles land in the other tally and are
        # reported by the footer without alarm, which is what R1, R2 and R3 all tripped.
        if watch.compiled > warned_frames and not escalated:
            frames = watch.compiled
            if warned_at is None:
                warned_at = record.step
                print(format_recompile_warning(frames, record.step), flush=True)
            else:
                # Second and last. The count is still climbing, so the flag is now the cheaper
                # trade — and printing it once more, not once per step, is what keeps a broken
                # run readable.
                escalated = True
                print(
                    format_recompile_escalation(frames, warned_frames, record.step), flush=True
                )
            warned_frames = frames

    def close_epoch(record: EpochRecord) -> None:
        """§10's row: the losses, the LR, the counters — and §10.3's probe on its cadence."""
        print(format_epoch(record), flush=True)
        if record.probe is not None:
            # Printed as well as logged, and with §10.3's caveat attached — the person most
            # likely to over-read a probe number is the one watching the run it came out of.
            print(format_probe(record.probe), flush=True)

    def announce_alarm(alarm: Alarm) -> None:
        # Printed as well as logged: an alarm that exists only in W&B is one nobody sees during
        # the run that fired it, which is the run where it is still worth acting on.
        print(f"  {alarm}", flush=True)

    with open_run_logger(cfg, fingerprint) as logger:
        train_epochs(
            cfg,
            steps=steps,
            device=args.device,
            logger=logger,
            policy=policy,
            resume=resume,
            on_start=announce,
            on_step=report,
            on_epoch=close_epoch,
            on_alarm=announce_alarm,
            watch=watch,
        )

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
    if policy is not None:
        print(f"checkpoints      {describe_checkpoints(policy)}")
    print(
        format_compile_summary(
            watch.compiled, watch.monitored, warned_at, enabled=cfg.runtime.compile
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
