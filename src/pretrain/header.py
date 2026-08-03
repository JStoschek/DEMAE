"""The run header — the config + fingerprint block every run records (issue #89, plan §15.7).

``python -m pretrain --corpus data/corpus_500k`` prints exactly what a real run stamps into
its checkpoints and its W&B config, against a real corpus, with nothing else attached. That
makes the pre-flight checklist's item 7 (*"W&B run tagged with git SHA, config hash,
selection-list md5, split md5"*) a thing you can look at before renting a GPU, and it makes
this ticket's foundations runnable rather than merely importable.

:func:`run_record` is the machine-readable half — the dict #93's checkpoints and #97's logger
carry — and :func:`format_header` is the same content laid out for a human. They are one
function apart on purpose: a header that could disagree with the recorded config would be
worse than no header.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .config import RunConfig
from .fingerprint import DataFingerprint, fingerprint_for


def run_record(config: RunConfig, fingerprint: DataFingerprint | None = None) -> dict:
    """The full provenance payload: the config, its hash, and the data fingerprints.

    This is what goes into every checkpoint (plan §11) and every W&B run (§15.7), and it is
    plain JSON throughout, so a reader with neither this code nor this corpus can still tell
    which recipe produced a released encoder and what it was trained on.
    """
    return {
        "config": config.to_dict(),
        "config_hash": config.config_hash,
        "fingerprint": (fingerprint or fingerprint_for(config)).to_dict(),
    }


def format_header(config: RunConfig, fingerprint: DataFingerprint) -> str:
    """The header a run prints at step 0 — the recipe's headline numbers, then the data."""
    a, o, s = config.architecture, config.optimization, config.schedule
    lines = [
        f"DEMAE Phase 1 — run {config.run_id}   config {config.config_hash}",
        "",
        f"encoder          {a.encoder_dim}d x {a.encoder_depth} ({a.encoder_heads} heads), "
        f"patch {a.patch_px}, {a.n_patches} tokens, in_chans {a.in_chans}",
        # `pos_encoding` earns a line of its own because R4 and R5 differ by it and by almost
        # nothing else (#102): mmselfsup's sin-cos embedding is the transpose of MAE's, so the
        # continual arm holds a different spatial convention from every scratch run. A knob two
        # arms disagree on unseen is the thing this header exists to make visible.
        f"positional       {a.pos_encoding}",
        f"decoder          {a.decoder_dim}d x {a.decoder_depth} ({a.decoder_heads} heads) "
        f"-> {a.patch_dim} per patch",
        f"masking          {config.masking.ratio:.0%} {config.masking.strategy} — "
        f"{config.masked_tokens} hidden, {config.visible_tokens} visible",
        f"loss             MSE on masked patches, norm_pix "
        f"{'on' if config.loss.norm_pix_loss else 'off'} (var floor "
        f"{config.loss.norm_pix_var_floor:g}, {config.loss.norm_pix_stats_dtype} stats)",
        f"                 ocean-weighted {config.loss.ocean_weighted}, patches below "
        f"{config.loss.min_valid_patch_frac:.0%} valid excluded",
        f"optimizer        {o.optimizer} betas {o.betas} wd {o.weight_decay}"
        f"{' fused' if o.fused else ''}, clip {o.grad_clip}, "
        f"dropout {o.dropout:g}/droppath {o.drop_path:g}, ema {o.ema}",
        f"batch            {o.batch_size} x {o.accum_iter} accum "
        f"= {config.effective_batch} effective",
        f"lr               {config.lr:.2e}  (blr {o.blr:g} x "
        f"{config.effective_batch}/{o.lr_batch_base})",
        f"schedule         {s.epochs} epochs, {s.warmup_epochs} warmup, {s.decay} to 0, "
        f"milestones {list(s.milestones)}",
        f"runtime          {config.runtime.precision}, attention {config.runtime.attention}, "
        f"compile {config.runtime.compile} (dynamic {config.runtime.compile_dynamic}), "
        f"seed {config.runtime.seed:#x}",
        f"data             {config.data.normalization} normalization, "
        f"train on {config.data.train_split!r}, val on {config.data.val_split!r}, "
        f"{config.data.num_workers} workers",
        f"augmentation     dihedral {config.data.dihedral}, "
        f"random-resized-crop {config.data.random_resized_crop}",
        f"init             {config.init.mode}"
        + (f" from {config.init.checkpoint} (patch embed {config.init.patch_embed_adapt}med)"
           if config.init.mode != "scratch" else ""),
        "",
        *fingerprint.lines(),
    ]
    return "\n".join(lines)


def parse_override(text: str) -> dict:
    """``group.knob=value`` -> the one-key overlay dict it means (#95).

    The launcher's escape hatch, and the reason an ablation does not need a file: R2 is
    ``--set masking.ratio=0.9`` and nothing else. What comes back is shaped like a partial
    config, so it goes through :meth:`RunConfig.from_dict` with everything else and gets that
    method's refusal of unknown groups and unknown knobs for free — a typo'd override has to
    stop the run for exactly the reason a typo'd config file does.

    **The value is read as JSON, falling back to the string itself.** That is what makes
    ``runtime.compile=false`` the boolean and not the (truthy) string ``"false"``, keeps
    ``optimization.batch_size=512`` an int, lets ``schedule.milestones=[50, 100]`` through, and
    still leaves ``data.corpus_dir=data/corpus_500k`` — which is not JSON — a path.
    """
    key, sep, raw = text.partition("=")
    if not sep:
        raise ValueError(f"override {text!r} is not KNOB=VALUE (e.g. masking.ratio=0.9)")
    path = key.strip().split(".")
    if len(path) > 2:
        raise ValueError(
            f"override {text!r}: a config knob is `group.knob` or a top-level `run_id`, "
            f"and {key.strip()!r} is deeper than that"
        )
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw  # a bare string: a path, a run id, a split name
    return {path[0]: value} if len(path) == 1 else {path[0]: {path[1]: value}}


def _overlay(base: dict, overlay: dict) -> dict:
    """A full config dict with ``overlay``'s groups merged in, one level deep.

    An unknown group passes straight through rather than raising here, so
    :meth:`RunConfig.from_dict` is the single place that names what a config may contain and
    the single message a typo produces.
    """
    merged = dict(base)
    for key, value in overlay.items():
        merged[key] = {**base.get(key, {}), **value} if isinstance(value, dict) else value
    return merged


def build_config(
    *,
    config_path: Path | str | None = None,
    vit_base: bool = False,
    corpus: str | None = None,
    assignment: str | None = None,
    blocks: str | None = None,
    overrides: Sequence[str] | None = None,
) -> RunConfig:
    """Assemble the run config: a preset, a partial config file, the paths, then ``--set``.

    Order is the point, and it is documented because a run's recipe is the thing this whole
    package exists to keep unambiguous:

    1. **The preset** — the plan's defaults, or ``--vit-base``'s B/16 shape.
    2. **The config file**, naming only the knobs its run moves — merged onto the preset, so
       ``--vit-base --config r5.json`` keeps the B/16 shape.
    3. **The path flags**, which are environment rather than recipe and never touch the hash.
    4. **``--set`` overrides**, last, so the command line can always pin a knob whatever the
       file said. It is the general escape hatch, so it wins over the special-case flags: a
       ``--set data.corpus_dir=…`` beats a ``--corpus``.
    """
    base = RunConfig.vit_base() if vit_base else RunConfig()
    if config_path is not None:
        overlay = json.loads(Path(config_path).read_text())
        base = RunConfig.from_dict(_overlay(base.to_dict(), overlay))
    data = base.data
    if corpus is not None:
        data = data.replace(corpus_dir=corpus)
    if assignment is not None:
        data = data.replace(split_assignment_path=assignment)
    if blocks is not None:
        data = data.replace(split_blocks_path=blocks)
    base = base.replace(data=data)
    for override in overrides or ():
        base = RunConfig.from_dict(_overlay(base.to_dict(), parse_override(override)))
    return base


def config_from_args(args: argparse.Namespace) -> RunConfig:
    """:func:`build_config` off the namespace :func:`add_config_arguments` filled. Use this.

    The two halves of one flag set, written next to each other so they cannot come apart. Every
    entrypoint used to spell the consumer half out — six keyword arguments, retyped per module —
    and ``pretrain.monitor`` shipped one that named five of them: ``--set`` was *defined*, parsed
    into ``args.overrides``, and then never passed to :func:`build_config`. The flag was accepted
    and discarded, so ``monitor --set masking.ratio=0.9`` masked 75% and reported R1's config
    hash. R2 (#83) and R3 (#84) are precisely ``--set`` ablations, and on that path they would
    have run R1's recipe and read out as "no difference" — the one failure mode a config hash
    exists to make visible, arriving through the code that computes it.

    A copy of an argument list can drift visibly; a copy of a *call* loses an argument silently.
    So there is one call, here, and
    `test_every_command_offering_set_consumes_it_through_the_one_consumer` holds it — by reading
    the package's own source for modules that offer the shared flags and refusing any that
    assembles a config some other way. Written against the call rather than the flag because
    neither broken command's *parser* was ever wrong.
    """
    return build_config(
        config_path=args.config,
        vit_base=args.vit_base,
        corpus=args.corpus,
        assignment=args.assignment,
        blocks=args.blocks,
        overrides=args.overrides,
    )


def add_config_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The six flags that assemble a :class:`RunConfig` — :func:`build_config`'s arguments.

    Shared rather than retyped, because every entrypoint that runs anything takes exactly these
    (``python -m pretrain`` here, ``python -m pretrain.train`` in #93, ``python -m
    pretrain.launch`` in #95), and two copies of one option list drift into two slightly
    different ways to name the corpus.

    :func:`config_from_args` is the other half. Call the pair; do not hand-write the six
    keyword arguments at an entrypoint, for the reason that function's docstring gives.
    """
    parser.add_argument("--corpus", default=None, help="corpus directory (default: the 500k tier)")
    parser.add_argument("--config", default=None, help="JSON file of the knobs this run moves")
    parser.add_argument(
        "--vit-base", action="store_true", help="the ViT-B/16 release preset (§2.1)"
    )
    parser.add_argument("--assignment", default=None, help="committed split assignment CSV (#77)")
    parser.add_argument("--blocks", default=None, help="committed split block lists JSON (#77)")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="GROUP.KNOB=VALUE",
        help="override one config knob, JSON-valued; repeatable, applied last",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """``python -m pretrain`` — print the header a real run would record."""
    p = add_config_arguments(
        argparse.ArgumentParser(
            prog="python -m pretrain",
            description="Print the Phase-1 run config + data fingerprints (#89).",
        )
    )
    p.add_argument("--json", action="store_true", help="emit the record a checkpoint carries")
    args = p.parse_args(argv)

    config = config_from_args(args)
    fingerprint = fingerprint_for(config)
    if args.json:
        print(json.dumps(run_record(config, fingerprint), indent=2))
    else:
        print(format_header(config, fingerprint))
    return 0
