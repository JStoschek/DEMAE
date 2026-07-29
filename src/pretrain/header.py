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
) -> RunConfig:
    """Assemble the run config: a preset, a partial config file, then the path overrides.

    Order is the point. The plan's defaults come first, the config file names only the knobs
    its run moves — merged onto the *preset*, so ``--vit-base --config r5.json`` keeps the B/16
    shape — and the paths, which are environment rather than recipe, go on last and never touch
    the config hash.
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
    return base.replace(data=data)


def main(argv: list[str] | None = None) -> int:
    """``python -m pretrain`` — print the header a real run would record."""
    p = argparse.ArgumentParser(
        prog="python -m pretrain",
        description="Print the Phase-1 run config + data fingerprints (#89).",
    )
    p.add_argument("--corpus", default=None, help="corpus directory (default: the 500k tier)")
    p.add_argument("--config", default=None, help="JSON file of the knobs this run moves")
    p.add_argument("--vit-base", action="store_true", help="the ViT-B/16 release preset (§2.1)")
    p.add_argument("--assignment", default=None, help="committed split assignment CSV (#77)")
    p.add_argument("--blocks", default=None, help="committed split block lists JSON (#77)")
    p.add_argument("--json", action="store_true", help="emit the record a checkpoint carries")
    args = p.parse_args(argv)

    config = build_config(
        config_path=args.config,
        vit_base=args.vit_base,
        corpus=args.corpus,
        assignment=args.assignment,
        blocks=args.blocks,
    )
    fingerprint = fingerprint_for(config)
    if args.json:
        print(json.dumps(run_record(config, fingerprint), indent=2))
    else:
        print(format_header(config, fingerprint))
    return 0
