# ADR 0009 — One shipping entry point, composing resume and monitoring (Phase 1)

- **Status:** Accepted
- **Date:** 2026-08-03
- **Context ticket:** "The shipping launcher composes neither resume nor monitoring" ([issue #106](https://github.com/JStoschek/DEMAE/issues/106)), part of the Phase-1 pre-R1 map ([#76](https://github.com/JStoschek/DEMAE/issues/76))
- **Builds on:** [ADR 0003 — Dataloader design](0003-dataloader-design.md) §1's per-epoch sampler, which is what makes a resumable loop possible at all

## Context

Phase 1 was built as a chain of tickets that each ended in a piece: #93 the training loop,
#94 checkpoints and exact resume, #95 the launch entrypoint and §7.5's counters, #97 the
per-epoch panel and the metrics sink, #98 the reconstruction grids, #99 the kNN probe. Each
piece was tested. Nothing composed them.

R0 ([#81](https://github.com/JStoschek/DEMAE/issues/81)) established what that meant for the
command the ops runbook tells a rented box to run:

```
uv run python -m pretrain.launch --vit-base --set 'monitoring.wandb_mode="online"'
```

It called `pretrain.train.train`, not `train_resumable`, and built no `RunLogger` and no
`EpochMonitor`. A 400-epoch R1 launched that way would have produced:

- **no checkpoints at all.** `sync_checkpoints.sh` would have found an empty directory and
  reported `ok` every pass, for forty hours. Plan §11's entire survival argument — vast.ai
  instances vanish permanently, so checkpoint every epoch and every 30–60 minutes — was
  unbacked on the path that ships, while the runbook called the sync loop "not optional".
- **no W&B run**, so §15.7's four tags never existed and `monitoring.wandb_mode="online"` was
  parsed, hashed as environment, and read by nobody.
- **no val pass, no per-patch histogram, no alarms, no recon grids, no probe** — every §10
  instrument.

`train_resumable` was called from exactly one place in the repository: a test.
`EpochMonitor` was driven by `pretrain.monitor.main` and tests. The module docstrings had
been promising the wrap for four tickets ("*#95's launcher … drives the same `EpochMonitor`*"
— it did not; "*this is what #94 and #95 wrap*" — they did not).

There was a **second** command that trained, `python -m pretrain.monitor`, which had §10's
panel and no checkpoints. Its `main` called `build_config(...)` without `overrides=`, while
the shared `add_config_arguments` defined `--set` with `dest="overrides"`. Measured on the box:

| command | config hash | masking |
|---|---|---|
| `launch --set masking.ratio=0.9` | `5d28a7384705e525` | 90% — 176 hidden |
| `monitor --set masking.ratio=0.9` | `c993d65cdbbde5ea` | 75% — 147 hidden |

The flag was accepted and discarded. R2 ([#83](https://github.com/JStoschek/DEMAE/issues/83))
and R3 ([#84](https://github.com/JStoschek/DEMAE/issues/84)) *are* `--set` ablations
(`masking.ratio=0.9`, `loss.norm_pix=false`), so on that path they would have run R1's recipe,
reported R1's config hash, and read out as "no difference". `pretrain.recon.main` had the same
hole.

## Decision

### 1. `python -m pretrain.launch` is the only command that trains

It composes #94's resumable loop, #97's `EpochMonitor` and sink, #98's grids and #99's probe,
plus the override surface and lever confirmation it already had. `--dry-run` still assembles
everything and spends nothing.

`python -m pretrain.train` stays as #93's tracer bullet and is documented as *not* a way to
start a run: steps rather than epochs, opt-in checkpoints, no monitoring. It exists to watch a
step go by.

### 2. `pretrain.monitor` stops being a run path

`monitor.main` is deleted rather than fixed. Two commands that both train are two recipes to
keep in step, and the one nobody watched is the one that drifted. What was reachable through
it — §10's reporting against a real corpus, before renting anything — is `pretrain.launch
--epochs 2`, which additionally checkpoints.

### 3. One loop, in one place

`pretrain.monitor.train_epochs` is rewritten as the composition: it calls
`pretrain.train.train_resumable` and supplies the epoch boundary's content. It is no longer a
parallel loop with its own grouping and its own meter. `train_resumable` gains what it needed
to carry that:

- **A `ThroughputMeter`**, so §7.5's counters are on the path that ships rather than only on
  the tracer bullet. `ThroughputMeter` delegates unknown attributes to the loader it wraps, so
  the `EpochCursor` can drive it as the loader.
- **An `on_epoch(epoch, throughput)` seam**, fired **after** that boundary's checkpoint is
  written. Two reasons, and both are about what a vanishing box costs: an epoch's val pass,
  grids and probe are minutes of work and should not take the epoch with them; and the RNG
  streams a checkpoint records are then the training loop's, captured before monitoring can
  draw from them, so §11's bit-identity does not depend on every instrument being
  RNG-neutral.
- **An `epochs=` argument**, so a run's length can be stated in the schedule's own unit.

### 4. `header.config_from_args(args)` is the only consumer of the shared flags

`add_config_arguments` defines them; `config_from_args` consumes them. Every entrypoint calls
the pair. A copy of an argument *list* drifts visibly; a copy of a six-keyword *call* loses an
argument silently, which is exactly what happened twice — in `pretrain.monitor.main` and again
in `pretrain.recon.main`.

Pinned against the **call**, not the flag: neither broken command's parser was wrong, so a test
that parsed `--set` and then called `config_from_args` itself would have stayed green through
both. What is asserted instead is the invariant — every module in the package that adds the
shared flags reaches its config through `config_from_args` and calls `build_config` nowhere —
read off the package's own source, so a new command is covered the moment it is written rather
than when someone remembers to add it to a list.

### 5. The checkpoint directory is `runs/<run_id>`, and resume is the default

Derived rather than fixed, because with resume on by default two arms sharing a directory is
R2 silently continuing R1's weights. The run id already distinguishes the arms, is what §15.7
tags the W&B run with, and `--set run_id=R2` is how an ablation is launched — so the directory
follows the ablation with no second flag to remember. `--checkpoint-dir` overrides it.

Resume defaults to on because vast.ai instances vanish permanently: a first launch and a
relaunch on a replacement box are the same command, typed at the one moment nobody wants to be
remembering a flag. An empty directory is a first launch, not an error. What was found is
**printed before the first step**, since a relaunch that silently restarted a 40-hour run and
one that silently continued the wrong run both look like a normal run afterwards.
`--no-resume` and `--no-checkpoints` are the opt-outs.

## Consequences

- The runbook's start command now produces what the runbook says it produces. Checkpoints land
  in `/workspace/DEMAE/runs/R1`, which is where `sync_checkpoints.sh` and `teardown_box.sh` are
  pointed (they said `runs/r1`; the case now matches the run id).
- `provision_box.sh`'s throughput probe passes `--no-checkpoints` and a disabled sink. It is a
  measurement of the box, not an experiment, and left alone it would now open a W&B run and
  write §11 checkpoints for it.
- A run's alarm trajectories are **not** checkpointed, so a resumed run's §10.1 alarms are
  evaluated over the epochs since the resume rather than over the whole run. Accepted for now:
  the trajectories live in W&B under a step axis that survives the resume, so the *record* is
  whole even where the in-process alarm state is not. Worth revisiting if a real R1 resume
  turns out to silence an alarm that should have fired.
- `--steps N` can end a run mid-epoch, in which case that partial epoch never reaches a
  boundary and is never scored. Deliberate: half an epoch's train loss is not a row of §10's
  table, and R0's smoke runs are the only thing that uses `--steps`.
- The three things whose absence nothing noticed are now asserted against the shipping entry
  point, not against the pieces: `test_the_shipping_command_writes_a_checkpoint`,
  `test_the_shipping_command_opens_a_run_and_logs_the_epoch_panel`, and
  `test_the_checkpoint_carries_the_recipe_the_run_was_launched_with`.

## Alternatives considered

**Keep both commands and fix `monitor`'s `--set`.** Rejected. It closes the measured symptom
and leaves the cause — two training paths with different capabilities, one of them the one
people reach for when they want to see the panel. The `--set` bug is what that costs; the
missing checkpoints were the other half of the same shape.

**Put the composition in `launch.py`.** It would make "the launcher owns the run" literal. But
`monitor.py` already owns the `EpochMonitor`, the val pass and the epoch record, and
`train.py` cannot import `monitor` (the dependency runs the other way), so the composition
placed in `launch.py` would be a third module reaching into the internals of the other two.
`train_epochs` keeps the loop next to the thing that reports it, and `launch.py` stays the
command: flags, console, and the two pre-run decisions.

**Refuse to resume across a config-hash change.** It would close the R2-resumes-R1 footgun
directly. Rejected because it blocks the one legitimate case the plan names — R5 extending
400 → 800 epochs by rebuilding the schedule and resuming ([#87](https://github.com/JStoschek/DEMAE/issues/87)),
which §6 calls a config change rather than a new run. The mismatch still warns
(`train_resumable`), and the per-`run_id` directory is what makes the accidental case unlikely
in the first place.
