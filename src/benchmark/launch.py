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
its six artifacts, the registered bar (:mod:`benchmark.prereg`), #149 the second, §3.1's tile
lists (:mod:`benchmark.tiles`), #156 the third, ADR-0012's extractor contract
(:mod:`benchmark.extract`), and #150 the fourth, §3.2's eval mask list
(:mod:`benchmark.masks`) — into which #153 transplanted §3.3's real-void family, the one stage
that reads anything off the network — and #151 the fifth, §3.4's train mask bank and the md5
sidecar git carries in its place. #154 wrote the sixth and last, §3.4's dataset card
(:mod:`benchmark.card`), which is why ``freeze`` **returns 0**: there is no stage left to name.
#158 made ``run`` real for **A1**: §2.4's classical fill over the committed eval mask list, both
of §8's norm-inversion gates, and the rows and run record that come out the other side
(:mod:`benchmark.evaluate`, :mod:`benchmark.results`).

#159 made ``run`` **train**. An arm whose trainable part is §4.1's head — A4, A5, A6 — is a
train-then-score cell: :mod:`benchmark.train` fits :mod:`benchmark.head` against the committed
train mask bank, early-stops on §3.1's val draw, and hands the pass the weights it kept. It also
gave ``run`` its one mode, ``--tune``: §4.1's single hyperparameter search, on A6, over val,
which writes ``head_tuning.json`` and deliberately scores nothing (:mod:`benchmark.tune`). Every
later cell's head recipe is refused if it has drifted from the line that search froze, which is
what "no thumb lands on the scale" is, as a mechanism.

#169 made ``report`` real: the 396-number printed table, the 66-cell C4 rank block and §9's
three figures, rendered from the results parquet and printing nothing R5 does not permit it to
print. It re-runs no arm, which is what makes it free to re-render.

With the dataset card, **every verb finishes what it starts**. Up to #154 each one raised past
the work it had and named the stage that was missing — deliberately, because a verb that quietly
returned 0 having done part of its job is precisely the shape of failure this module exists to
prevent, and it is the shape that survived four tickets in Phase 1. It mattered more once a verb
wrote real files than it did when none did: a ``freeze`` that lands three of six artifacts and
exits 0 is read as a frozen box. What outlives that defence is narrower and still load-bearing:
an arm whose input convention has no implementation yet names its own
ticket rather than returning quietly (:data:`benchmark.evaluate.FILLERS`).

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
from typing import Final

from . import (
    arms,
    artifacts,
    card,
    evaluate,
    extract,
    masks,
    prereg,
    report,
    results,
    tiles,
    train,
    tune,
    zeroshot,
)
from .artifacts import Resolution
from .config import (
    LABEL_EFFICIENCY_SIZES,
    BenchmarkConfig,
    add_config_arguments,
    config_from_args,
)


def resolve(args: argparse.Namespace) -> Resolution:
    """Assemble the config, the arm and every input this cell reads — in that order.

    The order is what makes *"a typo'd override stops before the corpus is touched"* true
    rather than nearly true: the config is built first, so an unknown knob raises before a
    single file is stat'd, let alone hashed.

    §4.1's two recipe refusals are here for the same reason, and specifically **here rather than
    in** :func:`_run` (#159): ``--dry-run`` returns before the verb runs, so a check inside `run`
    would let ``--dry-run --set head.lr=3e-4`` exit 0 while the resolution line printed the moved
    learning rate under the words *tuned once ... then frozen for every arm*. A pre-flight that
    prints a drifted recipe as the frozen one is worse than no pre-flight.

    Only for ``run``. `freeze` writes artifacts that are a function of the ``tiles`` and ``masks``
    groups alone (:attr:`~benchmark.config.BenchmarkConfig.artifact_hash`), so refusing a freeze
    over a head knob would be a refusal about a recipe that stage does not read.
    """
    cfg = config_from_args(args)
    arm = arms.get(cfg.arm)
    if args.verb == "run":
        if getattr(args, "tune", False):
            tune.assert_tuning_arm(cfg)
        if arm.trains_head or arm.trains_everything:
            # Both, and #162 is why the second half is here: §4.4 trains A7 under §4.1's loss,
            # schedule length and seeds, so a moved `head.lr` is exactly as much a drift for A7
            # as it is for A4 — and an arm exempted from this check could be trained at a recipe
            # the frozen arms never saw, which is the one thing C3's bar must not be.
            #
            # A tuning pass is exempted from the *record* half of the guard — it is the thing
            # that resolves a record/config disagreement, so refusing it would make that state
            # unrecoverable. See `tune.assert_frozen`.
            tune.assert_frozen(cfg, tuning=getattr(args, "tune", False))
        _checked_cell(cfg, arm)
    return artifacts.resolve(cfg, arm)


def _checked_cell(cfg: BenchmarkConfig, arm: arms.Arm) -> None:
    """§2.3's shape for the appendix cell — one seed, at 20k — refused beside the recipe ones.

    Two axes and one function (#164), because they are two readings of a single sentence: §2.3
    cut the ViT-S row to *one observation at the headline size*. §7's table prices the
    label-efficiency sweep at three cells by two seeds and annotates it *"R4 and ViT-S are both
    out of the sweep (§2.3)"*, and §4.5's order of operations says the same — the appendix cell
    runs with the 20k block, and the 1k/5k sweep that follows is three arms and not four. So a
    cell arriving at 1k is exactly as much a row the plan does not have as a second seed is.

    :attr:`benchmark.arms.Arm.single_observation` says *that* this cell is one observation and
    :attr:`benchmark.config.Head.appendix_seeds` says how many that is, so this reads both rather
    than comparing against a literal `1`. That is the split the registry keeps everywhere — the
    arm declares the fact, the recipe declares the number — and it is also what finally makes
    `appendix_seeds` load-bearing: it has been in `TUNED_FIELDS`, and therefore in the config
    hash, since §4.1 was written, with nothing reading it. `config.Head` makes that complaint
    about `early_stop_metric` in its own words — *a frozen knob nothing reads is one a reader is
    entitled to think does something* — and this field was the other instance of it.

    Here rather than in :func:`_run` for §4.1's two refusals' reason, stated in :func:`resolve`:
    ``--dry-run`` returns before the verb runs, so a check inside `run` would let
    ``--dry-run --arm A4-S --seed 1`` exit 0 with the resolution line printing a second
    observation as though the plan had asked for one.

    The refusal is what makes the missing ± structural instead of conventional. Nothing
    downstream computes a per-cell ± today — :func:`benchmark.report.seed_spread` walks
    :data:`benchmark.arms.PRINTED_ROWS` and this cell is not in it — but "no ± is stored" is a
    claim about the *parquet*, and the parquet is one row per instance per seed. A second seed
    landing there is a bracket any reader could take, whatever the report chose to print.
    """
    if not arm.single_observation:
        return
    headline = LABEL_EFFICIENCY_SIZES[-1]
    if cfg.size != headline:
        raise ValueError(
            f"arm {arm.name} is out of the label-efficiency sweep, so {cfg.size} is not a cell "
            f"this plan has — §2.3 demoted ViT-S to one appendix observation at {headline} and "
            f"§7 prices the 1k/5k sweep at three cells with ViT-S and R4 both named as outside "
            f"it. A fourth arm arriving there is a row nothing budgeted and no §0 claim reads."
        )
    allowed = cfg.head.appendix_seeds
    if cfg.seed < allowed:
        return
    raise ValueError(
        f"arm {arm.name} is a single observation — §2.3 cut this cell to "
        f"{allowed} seed and reports it without a ±, so seed {cfg.seed} is not a cell this "
        f"plan has. ViT-B is the primary capacity on both sides and R1's ViT-S rides along as "
        f"a cheap extra row; a second observation here would put a ± in the appendix that reads "
        f"as the capacity bracket §2.3's rewrite removed. If the plan really has changed, "
        f"`head.appendix_seeds` is the number and amending §2.3 is the decision."
    )


def format_resolution(resolution: Resolution, *, verb: str) -> str:
    """The resolution block as one string.

    The layout belongs to :meth:`~benchmark.artifacts.Resolution.lines`, beside the facts it is
    laying out; what is here is the console's join and nothing else.
    """
    return "\n".join(resolution.lines(verb=verb))


#: The three verbs take one resolution and the parsed arguments, and return an exit code. ``run``
#: was the first one that did (#158), then `report` (#169) and now `freeze` (#154). None of them
#: raises past its work any more, and the helper that named a missing stage went with the last of
#: them: there is no stage left for it to name. The uniform parameters are what keep
#: :data:`VERBS` a table rather than a cascade; ``args`` joined them at #159, which gave `run` a
#: mode (``--tune``) rather than a knob.


def _freeze(resolution: Resolution, args: argparse.Namespace) -> int:
    """`freeze` — write every committed input. §4.5 step 1, CPU-only.

    All six are built, and the order is each ticket's own argument rather than the order they
    happened to land in.

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

    **The eval mask list** goes next (#150, #153), because it is a function of the
    test list two stages above it — §3.2's 27,138 synthetic rows are the 4,523 clean test tiles
    x two families x three buckets, and §3.3's real family is transplanted onto hosts drawn from
    that same filtered list — and it reads the list back off disk rather than out of the draw,
    on the contract family's rule that the committed file is what later phases re-read. It
    carries the same §3.4 guard the spec does, and the same generator config hash the bank does.

    It is also the one stage that is not purely local: §3.3's real family needs #127's 24
    granules on disk and reads one GLO-30 COG per cell for the band the void is matched on. A
    box without the granules is **refused** here rather than committing a well-formed eval mask
    list of a benchmark with no C4 arm in it.

    **§3.4's train mask bank** goes next (#151), because it is a function of the *head-train*
    list and because it is the expensive one: 1,000,000 draws is twenty minutes, against every
    other stage's seconds. A stage that fails above it therefore costs nothing to retry, and the
    bank's own guard reads the committed sidecar before it draws rather than after — it is the
    one artifact here whose refusal would otherwise arrive twenty minutes late. It is also the
    one that is **not** committed as bytes: 104 MB cannot go in git, so what lands in
    ``artifacts_dir`` is the sidecar holding its md5.

    **§3.4's dataset card** goes last (#154), and it is last by definition: it is the document
    that says what the five stages above it produced, so it cannot be written before they have.
    It is also the only artifact here written for a *reader* — the two things a reader would
    otherwise have to infer are that the module commits tile lists and masks and never tiles,
    and that the bank is not in this repository (:mod:`benchmark.card`).

    With it, `freeze` **returns 0** for the first time. Every verb above raised past the work it
    had and named the stage that was missing, which is the defence against a command that writes
    five of six artifacts and reports success; there is no sixth stage left to name.
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
    evaluated = masks.freeze(resolution.config)
    for line in masks.format_written(evaluated, resolution.config.masks):
        print(line)
    banked = masks.freeze_bank(resolution.config)
    for line in masks.format_bank_written(banked):
        print(line)
    # Handed the two mask stages' own return values rather than re-reading their files: what the
    # card owes about them — the real family's attrition, the bank's draw hash — is in those
    # objects, and a second reader would be a second answer to what this freeze just did. The
    # tile lists it re-reads itself, on the contract family's rule (:mod:`benchmark.card`).
    written = card.freeze(resolution.config, eval_masks=evaluated, bank=banked)
    for line in card.format_written(written):
        print(line)
    return 0


def _run(resolution: Resolution, args: argparse.Namespace) -> int:
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

    Two refusals fire ahead of all of it, in :func:`resolve` rather than here, because they are
    about the **recipe** rather than about the box: a cell whose head hyperparameters have drifted
    from §4.1's frozen ones, and a ``--tune`` asked for an arm that is not the neutral one. They
    belong beside the config-typo refusal, and they have to be upstream of ``--dry-run`` — see
    :func:`resolve`.
    """
    # §4.1's "tuned once, then frozen for every arm", enforced before anything is stat'd, let
    # alone spent — it is a fact about the recipe, so it belongs beside the config-typo refusal
    # rather than after a 104 MB hash. Asked of every arm that trains under that recipe, which
    # since #162 means A7 as well as the three frozen arms: A1 is the only one with no §4.1
    # recipe to have drifted from.
    missing = resolution.missing
    if missing:
        raise FileNotFoundError(
            f"nothing to score against — {', '.join(missing)} "
            f"{'is' if len(missing) == 1 else 'are'} not on this box. Run "
            f"`python -m benchmark.launch freeze` first (§4.5 step 1; it is CPU-only, and needs "
            f"#127's granules and a GLO-30 read per cell for §3.3's real family)."
        )
    # §3.4's one input that is committed as a hash rather than as bytes, so presence is not
    # enough: a 104 MB parquet at the configured path is *a* bank, and the only thing that says
    # it is this run's is the md5 the sidecar commits. Asked of the arms that read it and no
    # others, on `missing`'s rule — A1 trains nothing, and refusing it for a bank it never opens
    # would gate the first number in this benchmark on the artifact only the learned arms use.
    if resolution.reads("mask_bank"):
        masks.assert_bank_matches(resolution.config)
    checkpoint = resolution.checkpoint
    if checkpoint is not None and checkpoint.required and not checkpoint.present:
        raise FileNotFoundError(
            f"arm {resolution.arm.name} reads the {resolution.arm.encoder} encoder and "
            f"{checkpoint.absent_note}"
        )
    # A2's second file, on the same rule and for a sharper reason: §14's release is encoder-only,
    # so an arm whose whole mechanism *is* the pretrained decoder cannot be driven by the file
    # every other arm initializes from (#167). Refused here rather than inside the pass so the
    # message names the path and the field instead of a missing tensor.
    mae = resolution.mae_checkpoint
    if mae is not None and not mae.present:
        raise FileNotFoundError(
            f"arm {resolution.arm.name} runs the pretrained MAE decoder and {mae.absent_note}"
        )

    cfg = resolution.config
    trained = None
    if resolution.arm.trains_head:
        if args.tune:
            return _tune(resolution)
        trained = train.train_head(
            cfg, resolution.arm, on_epoch=lambda line: print(f"{'':<16} {line.line()}")
        )
        for line in train.format_trained(trained):
            print(line)
    elif resolution.arm.trains_everything:
        # A7 (#162). A separate branch rather than a third case inside `train_head`, because the
        # two passes differ in what they build and share everything after it — see
        # `train.train_unet`. There is no `--tune` arm of this branch: §4.1's recipe is tuned on
        # A6 and frozen for every arm, and `tune.assert_tuning_arm` has already refused any other
        # arm inside `resolve`, before this function is reached.
        trained = train.train_unet(
            cfg, resolution.arm, on_epoch=lambda line: print(f"{'':<16} {line.line()}")
        )
        for line in train.format_trained(trained):
            print(line)

    # The trained head goes to the pass, which hands it to the convention's factory (#159 over
    # #167's shape). It carries the encoder it was trained over, so the cell is scored against
    # the object the val curve was measured on rather than a second load of the same file.
    scored = evaluate.run_pass(cfg, resolution.arm, trained=trained)
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
    register = _register_disposition(resolution, scored)
    path, total, record = results.write(
        rows,
        resolution,
        metrics=scored.bar.registered_metrics,
        seconds=scored.seconds,
        parameters=arms.parameter_counts(resolution.arm),
        gates=scored.gates,
        training=None if trained is None else trained.to_dict(),
        register=register,
        precision=results.precision_block(resolution, _measured_agreement(resolution, trained)),
    )
    for line in results.format_written(path, rows=len(rows), total=total, record=record):
        print(line)
    if register is not None:
        print(f"{'register':<16} {zeroshot.format_disposition(register)}")
    return 0


def _measured_agreement(resolution: Resolution, trained) -> dict | None:
    """This run's own fp32-versus-compute-dtype agreement, off the encoder that actually ran.

    Read from the *trained* stack rather than from a second :class:`benchmark.extract.Extractor`
    built here, on `_head_fill`'s rule: a figure measured on a fresh load of the same file would
    describe a forward pass this cell never made. It is the same object the val curve was measured
    on, which is the only one the rows are answerable to.

    ``None`` from every arm with no extractor in the loop — A1, A7's scratch UNet, and the
    inference-only arms that hand `run_pass` no trained stack. Which of those it was is
    :func:`benchmark.results.precision_block`'s question, not this one's; all this reports is
    whether there is a number.

    **The trained stack is the only seam this reaches through, and #168 settled why A9 does not
    need a second one.** The note here used to say that A9 would record *no measurement reached
    the record* for a figure it had in fact taken, and that the arm building
    `a1_fill_through_tokenizer` should carry the filler's extractor out of `run_pass` and hand it
    here. It did not, because there is no such figure to carry: A9 runs TerraMind's diffusion
    decoder, which ADR-0012 §6 does not govern and nothing in this plan has measured at a reduced
    precision, so :func:`benchmark.passthrough.load` **refuses** a cell declaring anything but the
    reference dtype. A9's cells therefore land on :func:`benchmark.results.precision_block`'s
    third branch — *the run declares the reference dtype* — which is the true reading and reaches
    it without a seam. What A9 does owe, and what that seam was standing in for, is proof that its
    encoder half is the contract's; it discharges that at load time and exactly, by comparing the
    two builds' parameters rather than two forward passes (:func:`benchmark.passthrough.
    assert_same_encoder`).

    ``None`` from every arm with no extractor in the loop — A1, A7's scratch UNet, and the
    inference-only arms that hand `run_pass` no trained stack. Which of those it was is
    :func:`benchmark.results.precision_block`'s question, not this one's; all this reports is
    whether there is a number.
    """
    extractor = getattr(trained, "stack", None)
    extractor = None if extractor is None else getattr(extractor, "extractor", None)
    if extractor is None:
        return None
    return extractor.precision_agreement(resolution.arm.name)


def _tune(resolution: Resolution) -> int:
    """`run --arm A6 --tune` — §4.1's one tuning pass, and the record it leaves (#159).

    Scores nothing, and says so. The pass selects a recipe on **val**; A6's own primary cells are
    three ordinary `run --arm A6 --seed {0,1,2}` invocations afterwards, at the full 50 epochs.
    Folding the two together would have made the tuning pass produce a printed row, which is the
    one thing a val-only selection must not do.
    """
    cfg = resolution.config
    tuning = tune.tune(cfg, on_point=lambda point: print(f"{'':<16} {point.line()}"))
    written = tune.write_tuning(cfg, tuning)
    print()
    for line in tuning.lines():
        print(line)
    print(f"{'wrote':<16} {written}")
    print(
        f"{'scored':<16} nothing — the tuning pass reads val only. A6's own primary cells are "
        f"`run --arm {tune.TUNING_ARM} --seed {{0,1,2}}` at §4.1's full "
        f"{cfg.head.epochs} epochs"
    )
    return 0


def _register_disposition(resolution: Resolution, scored: evaluate.Pass) -> dict | None:
    """How this cell's row is to be read, for the one registered rule whose trigger is a result.

    R12 governs A2 and nothing else, so this is ``None`` for every other arm and the run record
    carries no ``register`` block at all (:func:`benchmark.results.run_record`).

    The comparison is against **A1's rows in the same cell** — same size, same seed — read back
    out of the results parquet rather than recomputed, because §4.5 already runs A1 first and its
    numbers gate every learned arm. A2's own rows come from the pass in memory, before
    :func:`benchmark.results.write` lands them: the disposition is part of the record that
    describes those rows, so it cannot be computed from a file they are not in yet.

    Nothing here decides anything. R12 is discursive (ADR-0013) — the enforcement is the
    writeup's and the reviewer's — so a run that trips the contingency still writes its rows and
    still exits 0. What changes is that the record says so, which is the whole of what a
    discursive rule can be held to.
    """
    arm = resolution.arm
    if not zeroshot.governs(arm):
        return None
    cfg = resolution.config
    return zeroshot.disposition(
        arm,
        a1_rows=results.rows_for_cell(
            cfg.paths.results_path,
            arm="A1",
            size=cfg.size,
            seed=cfg.seed,
            columns=zeroshot.COMPARISON_COLUMNS,
        ),
        a2_rows=[
            {
                "band": item.band,
                "family": item.family,
                "bucket": item.bucket,
                "tile_id": item.tile_id,
                "instance": item.instance,
                zeroshot.COMPARISON_METRIC: item.metrics[zeroshot.COMPARISON_METRIC],
            }
            for item in scored.scored
        ],
    )


def _report(resolution: Resolution, args: argparse.Namespace) -> int:
    """`report` — render the table from the results parquet, re-running no arm (#169).

    The one verb that reads no committed input except the registered bar and writes nothing into
    ``artifacts_dir``. It is a **pure function of the results parquet** and the run records
    beside it (#136's "computed and printed are different questions"), which is what makes
    re-rendering free: a reviewer asking a question the table's shape did not anticipate is
    asking it of a file that already holds the answer, not of a rented box.

    Unlike `run` it takes no cell. The table is every printed arm at the headline size, so
    ``--arm``/``--size``/``--seed`` would name a cell the render does not have — what it renders
    is whatever the parquet holds, and a row nobody has scored says so in its own cells.
    """
    report.render(resolution.config)
    return 0


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
        # Set for every verb so `args.tune` is answerable without asking which one ran; only
        # `run` offers the flag that moves it.
        sub.set_defaults(tune=False)
        return sub

    add("freeze", "write every committed input (CPU-only; reads GLO-30 for §3.3)")
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
    # A mode rather than a knob: it does not change what a cell *is*, it changes whether this
    # command produces one. Only `run` has it, and only on `--arm A6` does it do anything but
    # refuse (§4.1, #159).
    run.add_argument(
        "--tune",
        action="store_true",
        help=f"§4.1's one tuning pass, on {tune.TUNING_ARM}, over val — writes "
        f"{tune.TUNING_NAME} and scores nothing",
    )
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
    return VERBS[args.verb](resolution, args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
