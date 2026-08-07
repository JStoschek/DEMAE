"""§14's release artifacts: the encoder-only export, and the manifest that identifies it.

Plan §14 releases, per run, *"encoder-only weights (timm-loadable) + full MAE checkpoint +
config + ``dataset_version``/split-assignment fingerprints + **init provenance** + the training
curves"*. The full checkpoint, the config and the fingerprints already exist inside every
checkpoint payload (:func:`~pretrain.checkpoint.save_checkpoint`); what does not exist until
here is the **encoder-only file a consumer actually loads**, the **curves**, the **milestone
set**, and the manifest that says what all of it is.

Two of those carry a caveat that is part of the artifact rather than commentary on it:

- **The curves are indexed by** :data:`EPOCH_AXIS`, spelled out in the manifest. W&B's ``_step``
  is the 0-based index of the epoch that just ended and
  :attr:`~pretrain.checkpoint.RunState.epoch` is epochs completed, so the two are off by one for
  a run's entire length (#132). Neither is wrong; an unlabelled curve is.
- **No milestone carries a probe number**, because none was ever measured at a milestone epoch —
  milestones fire at completed {50, 100, 200, 400} and §10.3's probe at completed
  {1, 21, ..., 381}, and the intersection is empty (#131). :data:`PROBE_NOTE` ships that fact
  next to the files, and :func:`milestone_set` records how far the nearest real probe was so a
  reader who carries one forward knows they are inferring.

## Why this is not ``{k: v for k, v in state.items() if not is_decoder_parameter(k)}``

Because that file loads into timm cleanly and is wrong.

:attr:`~pretrain.model.MaskedAutoencoder.pos_embed` is registered ``persistent=False``
(:mod:`pretrain.model`), so ``state_dict()`` does not carry it — the buffer is *derived* from
the config, which is exactly what a §11 resume on a fresh box needs. timm's
``VisionTransformer`` holds the same tensor as a **learned** ``nn.Parameter`` and requires it.
So the encoder half of our state dict is 149 tensors that match timm's names exactly, with
nothing unexpected and one thing missing, and the obvious consumer line —

.. code-block:: python

    model.load_state_dict(torch.load("encoder.pt"), strict=False)

— reports a single missing key against a perfect-looking export and leaves timm's **randomly
initialized** positional embedding in place. No loss, no shape, no key report differs. That is
this map's recurring shape — the seam between two correct halves, and #95's silent lever — on
the one artifact outside consumers touch.

It is worse than a generic export bug because **R4 and R5 do not share an axis order**. R5 ships
``sincos2d`` (column-then-row, MAE's) and R4 ships ``sincos2d_rowfirst`` (mmselfsup's transpose,
P102 §4), so a ``pos_embed`` regenerated from a default rather than from the run's own
:class:`~pretrain.config.Architecture` would ship R4 under R5's ordering — which P102 measured
as the difference between 6.7e-7 and 2.0 on that very tensor, and which no name or shape check
can see.

Hence three rules, all enforced rather than documented:

1. :func:`encoder_state_dict` **materializes** ``pos_embed`` from the run's own architecture.
2. :func:`assert_timm_loadable` loads the result into a real timm model **strictly** — zero
   missing, zero unexpected, shapes checked by ``load_state_dict`` itself.
3. The manifest records :attr:`~pretrain.config.Architecture.pos_encoding` by name, so a
   consumer rebuilding the buffer cannot pick the other one silently.

## Provenance is recognized, never assumed

:func:`~pretrain.init.recognize`'s rule (P101 §7) reaches the release here. A ``scratch`` run's
provenance is a property of the code path and needs no file: `init.mode == "scratch"` means
:func:`~pretrain.init.initialize_from_config` was a no-op and no external weight entered the
model. A warm-started run's provenance is a claim about **a file**, and this module will not
make it unless the file is presented and matches a :data:`~pretrain.init.KNOWN_CHECKPOINTS`
entry on counts, parameter total, recorded epoch, size and md5.

There is a gap worth naming, because it is the reason ``init_file`` is a parameter at all: a
checkpoint payload carries ``config`` — and so ``init.mode`` and ``init.checkpoint`` — but **not
the** :class:`~pretrain.init.InitReport` **the warm start produced**. The path it does carry is a
path on a rented box that no longer exists. So exporting R4 from a laptop can state the mode and
the configured path from measurement, and can state the URL and the licence only if someone puts
the 485 MB file in front of it. An export that cannot verify says so in the manifest and makes
**no licence claim** — which is the whole point of P101, and strictly better than a default that
would assert Apache-2.0 about a file nobody checked.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Iterable, Mapping, Sequence

import torch
from torch import Tensor

from .checkpoint import discover, load_checkpoint
from .config import Architecture, RunConfig
from .init import KNOWN_CHECKPOINTS, checkpoint_md5, read_state_dict, recognize
from .model import AXIS_ORDERS, is_decoder_parameter, sincos2d_pos_embed

#: The timm architecture each of our encoder shapes *is*, keyed by
#: ``(encoder_dim, encoder_depth, encoder_heads)``. A mapping rather than a string built from the
#: dim, because "timm-loadable" is a claim about a specific published model name and the release
#: has to name the one it was checked against — a consumer types this string, not our config.
TIMM_MODELS: Final[dict[tuple[int, int, int], str]] = {
    (384, 12, 6): "vit_small_patch16_224",
    (768, 12, 12): "vit_base_patch16_224",
}

#: The key that is in timm's state dict and not in ours, and the entire reason this module runs
#: an assertion instead of a comment.
POS_EMBED_KEY: Final = "pos_embed"

#: What a consumer must pass to :func:`timm.create_model` for the export to load. Single-channel
#: and headless: the released artifact is a backbone, and §14 hands the head to Phase 2.
TIMM_CREATE_KWARGS: Final = {"num_classes": 0, "global_pool": ""}


def timm_model_name(arch: Architecture) -> str:
    """The published timm model this encoder is, or a refusal naming the shape.

    Refuses rather than guessing, because the name goes into the manifest as the string a
    consumer will type. A ViT-L export (§2.1 rules it out of Phase 1) should stop here loudly
    rather than ship a manifest naming a model whose depth does not match the weights.
    """
    key = (arch.encoder_dim, arch.encoder_depth, arch.encoder_heads)
    if key not in TIMM_MODELS:
        known = ", ".join(sorted(TIMM_MODELS.values()))
        raise ValueError(
            f"no timm model registered for encoder dim/depth/heads {key} — §14 releases "
            f"timm-loadable weights, so an unregistered shape has no name to ship under "
            f"(known: {known})"
        )
    if arch.patch_px != 16 or arch.img_px != 224:
        raise ValueError(
            f"timm's {TIMM_MODELS[key]} is patch 16 at 224 px; this architecture is patch "
            f"{arch.patch_px} at {arch.img_px} px"
        )
    return TIMM_MODELS[key]


# --- The weights --------------------------------------------------------------


def encoder_state_dict(model_state: Mapping[str, Tensor], arch: Architecture) -> dict[str, Tensor]:
    """§14's encoder-only weights, with ``pos_embed`` materialized from ``arch``.

    ``model_state`` is a checkpoint payload's ``model`` entry — the full MAE. The decoder is
    dropped by :func:`~pretrain.model.is_decoder_parameter`, which is the definition of
    "encoder-only" and gets ``mask_token`` right where a ``decoder_`` prefix test would not.

    The positional embedding is then **rebuilt from this run's own architecture** and inserted.
    See the module docstring for why the export is wrong without it and silently wrong with the
    wrong axis order.
    """
    encoder = {k: v for k, v in model_state.items() if not is_decoder_parameter(k)}
    if POS_EMBED_KEY in encoder:
        raise ValueError(
            f"{POS_EMBED_KEY} is already in the state dict — this module materializes it from "
            f"the config, and two sources for one tensor is how they come to disagree"
        )
    encoder[POS_EMBED_KEY] = sincos2d_pos_embed(
        arch.encoder_dim,
        arch.img_px // arch.patch_px,
        axis_order=AXIS_ORDERS[arch.pos_encoding],
    )
    return encoder


def assert_timm_loadable(state: Mapping[str, Tensor], arch: Architecture) -> str:
    """Load ``state`` into a real timm model **strictly**, or raise. Returns the model name.

    Strict, and against a constructed model rather than a key list, because the failure this
    exists to catch is a *silent* one: ``strict=False`` reports a missing ``pos_embed`` and
    carries on with timm's random one, and a hand-written key comparison would not check shapes
    at all. ``load_state_dict(strict=True)`` checks names and shapes together and refuses.

    Imports timm inside the function: :mod:`pretrain.release` is importable for its manifest
    types on a box with no ``[pretrain]`` extra, and only this assertion needs the dependency.
    """
    import timm

    name = timm_model_name(arch)
    model = timm.create_model(name, in_chans=arch.in_chans, **TIMM_CREATE_KWARGS)
    model.load_state_dict(dict(state), strict=True)
    return name


# --- Provenance ---------------------------------------------------------------


@dataclass(frozen=True)
class InitProvenance:
    """§14's required per-run field: what the encoder started from, under what licence."""

    #: :attr:`~pretrain.config.Init.mode` as the run recorded it — ``scratch`` or ``imagenet_mae``.
    mode: str
    #: The path the run was configured with. Kept even when unverifiable: it is what the run
    #: actually read, and it names the file someone would have to produce to verify the claim.
    checkpoint: str | None
    #: Whether the claim below was *measured*. False never means "probably fine".
    verified: bool
    #: The publisher's identity, only ever set from :func:`~pretrain.init.recognize`.
    spec: dict[str, Any] | None
    #: Why ``verified`` is what it is, in a sentence a release note can quote.
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def licence(self) -> str | None:
        """The init's licence, or ``None`` when no verified claim exists."""
        return None if self.spec is None else self.spec["licence"]


def init_provenance(config: RunConfig, *, init_file: Path | str | None = None) -> InitProvenance:
    """Build §14's init-provenance field for a run, verifying against ``init_file`` if given.

    ``scratch`` is verified without a file — :attr:`~pretrain.config.Init.mode` *is* the fact,
    since :func:`~pretrain.init.initialize_from_config` is a no-op there and no external tensor
    can have entered the model. Every other mode is a claim about bytes, and gets one only when
    the bytes are presented and :func:`~pretrain.init.recognize` matches them.
    """
    mode = config.init.mode
    configured = config.init.checkpoint
    if mode == "scratch":
        return InitProvenance(
            mode=mode,
            checkpoint=None,
            verified=True,
            spec=None,
            note=(
                "trained from random initialization; no external weights entered this model, "
                "which is a property of the configured code path rather than of any file"
            ),
        )

    if init_file is None:
        return InitProvenance(
            mode=mode,
            checkpoint=configured,
            verified=False,
            spec=None,
            note=(
                f"warm-started from {configured!r}, which was not presented to the export — the "
                f"checkpoint payload records the path but not the init report, so no URL, md5 "
                f"or licence is claimed here. Re-run the export with the file to verify"
            ),
        )

    path = Path(init_file)
    state, meta = read_state_dict(path)
    spec = recognize(state, meta, path)
    if spec is None:
        known = ", ".join(s.name for s in KNOWN_CHECKPOINTS)
        return InitProvenance(
            mode=mode,
            checkpoint=configured,
            verified=False,
            spec=None,
            note=(
                f"warm-started from {path}, which matches none of the checkpoints DEMAE ships a "
                f"spec for ({known}) — a legitimate thing to have done, but it gets no "
                f"provenance claim, and in particular no licence"
            ),
        )
    return InitProvenance(
        mode=mode,
        checkpoint=configured,
        verified=True,
        spec={
            "name": spec.name,
            "url": spec.url,
            "md5": spec.md5,
            "size_bytes": spec.size_bytes,
            "epochs": spec.epochs,
            "licence": spec.licence,
            "licence_source": spec.licence_source,
        },
        note=(
            f"warm-started from {spec.name}, recognized against {path} on tensor counts, "
            f"parameter total, recorded epoch, size and md5"
        ),
    )


# --- The training curves ------------------------------------------------------

#: The column every shipped curve is indexed by, and the reason it needs a name rather than a
#: convention. W&B's ``_step`` is the **0-based index of the epoch that just ended**;
#: :attr:`~pretrain.checkpoint.RunState.epoch` is **epochs completed**; ``completed = _step + 1``
#: (P88 §1, found by #132). The two are off by one for a run's whole length and neither is
#: wrong — so a curve that does not say which one it carries is a curve that will eventually be
#: read against the other, and every point will look one epoch early.
EPOCH_AXIS: Final = "epochs_completed"

#: W&B's own bookkeeping. ``_step`` is not dropped so much as *renamed*: it becomes
#: :data:`EPOCH_AXIS`, plus one. The other two are wall-clock facts about a rented box that no
#: longer exists, and mean nothing to a consumer reading the curve.
WANDB_INTERNAL: Final = ("_step", "_runtime", "_timestamp")


def _is_scalar(value: Any) -> bool:
    """Whether ``value`` belongs in a CSV cell.

    ``recon/hillshade`` and ``recon/heatmap`` are W&B image references and ``val/per_patch_loss``
    is a histogram — dicts, all three. They are part of the run's record and no part of a *curve*,
    and flattening them into a cell would produce a column of unreadable JSON rather than a
    series. Strings stay: ``probe/set_md5`` is what makes two runs' probe points comparable at
    all (#132 leaned on it), and dropping it would cost the reader the ability to check that.
    """
    return isinstance(value, (int, float, str, bool))


def curve_rows(history: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """§14's training curves, moved onto the epochs-completed axis and stripped to scalars.

    ``history`` is one run's W&B history, newest last — a list of per-epoch rows. Every row is
    re-keyed from ``_step`` to :data:`EPOCH_AXIS`, which is the *only* transformation here that
    changes a number, and the one #132 found the release could not do implicitly.

    Sparse by design: the probe fires every 20 epochs and its columns are simply absent from the
    other rows. They are left empty rather than forward-filled, because a carried-forward probe
    value read as a measurement is exactly the confusion #131 is about.
    """
    rows = []
    for row in history:
        if EPOCH_AXIS in row:
            raise ValueError(
                f"history row already carries {EPOCH_AXIS!r} — this function derives it from "
                f"W&B's `_step`, and two sources for the epoch axis is how they come to disagree"
            )
        if "_step" not in row:
            raise ValueError("history row has no `_step`, so its epoch cannot be established")
        out: dict[str, Any] = {EPOCH_AXIS: int(row["_step"]) + 1}
        for key, value in row.items():
            if key in WANDB_INTERNAL or not _is_scalar(value):
                continue
            out[key] = value
        rows.append(out)
    return sorted(rows, key=lambda r: r[EPOCH_AXIS])


def curve_columns(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    """Every column present in ``rows``, with :data:`EPOCH_AXIS` first and the rest sorted."""
    seen: set[str] = set()
    for row in rows:
        seen.update(row)
    seen.discard(EPOCH_AXIS)
    return [EPOCH_AXIS, *sorted(seen)]


def write_curves(rows: Sequence[Mapping[str, Any]], path: Path | str) -> Path:
    """Write ``rows`` as CSV. Absent values are empty cells, not zeros and not carried forward."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = curve_columns(rows)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_history(path: Path | str) -> list[dict[str, Any]]:
    """One run's exported W&B history from JSON — a list of rows, or ``{"history": [...]}``."""
    payload = json.loads(Path(path).read_text())
    rows = payload["history"] if isinstance(payload, Mapping) else payload
    if not isinstance(rows, list):
        raise ValueError(f"{path} does not hold a list of history rows")
    return rows


def probe_epochs(rows: Sequence[Mapping[str, Any]], key: str = "probe/set_md5") -> list[int]:
    """The epochs at which a probe actually ran, on :data:`EPOCH_AXIS`.

    Keyed off ``probe/set_md5`` rather than a metric, because the md5's presence is what says
    *the probe ran against a committed set* — a metric column could in principle be present and
    null, and §10.3 treats a probe against no committed set as no probe at all.
    """
    return [int(r[EPOCH_AXIS]) for r in rows if r.get(key)]


# --- The milestone set --------------------------------------------------------


def milestone_set(
    checkpoint_dir: Path | str,
    *,
    probed_epochs: Sequence[int] = (),
    hash_files: bool = True,
) -> list[dict[str, Any]]:
    """§14's milestone checkpoints, which ride along for Phase 2's compute-scaling plot.

    Each entry carries **no probe number, ever** — see :data:`PROBE_NOTE`. ``probed_epochs`` is
    taken only to state, per milestone, how far the nearest real probe was, so that a reader who
    wants a number knows exactly what they would be inferring from.
    """
    entries = []
    for found in discover(checkpoint_dir):
        if not found.milestone:
            continue
        nearest = min(probed_epochs, key=lambda e: abs(e - found.epoch), default=None)
        entries.append(
            {
                "file": found.path.name,
                EPOCH_AXIS: found.epoch,
                "step": found.step,
                "md5": checkpoint_md5(found.path) if hash_files else None,
                # #131, structurally rather than by convention: the probe never ran at a
                # milestone epoch, so there is no measurement to put here and the key exists to
                # keep anyone from adding one.
                "probe": None,
                "nearest_probed_epoch": nearest,
            }
        )
    return entries


#: Why every milestone's ``probe`` is null, in a sentence the release note can quote (#131).
#: Milestones fire at completed {50, 100, 200, 400} and the probe at completed {1, 21, ..., 381};
#: the intersection is empty, so **no released checkpoint has ever been probed**. Carrying the
#: 381-completed numbers onto the 400 milestone is defensible — the tail is flat, val moves
#: −0.00012 over the final 19 epochs and ``lr`` has gone to zero — but it is an *inference*, and
#: the artifact says so rather than letting a number arrive unlabelled.
PROBE_NOTE: Final = (
    "No released checkpoint has ever been probed: milestones are written at completed epochs "
    "{50, 100, 200, 400} and §10.3's probe fires at completed {1, 21, ..., 381}, and those sets "
    "do not intersect (#131). Every milestone's `probe` is therefore null. The nearest probed "
    "epoch is given per milestone so that carrying a number forward is a stated inference rather "
    "than an unlabelled measurement; the run's own tail is flat enough to make that defensible, "
    "but the curve in curves.csv is the measurement and the milestone is not."
)


# --- The manifest -------------------------------------------------------------


@dataclass(frozen=True)
class ReleaseManifest:
    """What identifies one released encoder. §14's fields, in one JSON object."""

    #: The file this was exported from, and what it hashes to — so a bucket object and a
    #: manifest can be checked against each other without re-reading 1.2 GB of optimizer state.
    source_checkpoint: str
    source_checkpoint_md5: str
    #: Epochs completed and optimizer steps taken, from :class:`~pretrain.checkpoint.RunState`.
    #: **Epochs completed**, not W&B's 0-based epoch index — P88 §1 on why the two get confused.
    epochs_completed: int
    steps: int
    config_hash: str
    #: The timm model name a consumer passes to ``timm.create_model``, and the kwargs it needs.
    timm_model: str
    timm_create_kwargs: dict[str, Any]
    #: Which axis ordering the shipped ``pos_embed`` carries. R4 and R5 differ here (P102 §4),
    #: and the two are each other's transpose, so a consumer rebuilding it needs the name.
    pos_encoding: str
    encoder_tensors: int
    encoder_parameters: int
    encoder_file: str
    encoder_md5: str
    #: §14's fingerprints — ``dataset_version``, the selection and split md5s, the git sha.
    fingerprint: dict[str, Any]
    #: §14's required field, from :func:`init_provenance`.
    init_provenance: dict[str, Any]
    #: §14's training curves: the shipped file, its hash, and — the part that is not decoration —
    #: the name of the epoch axis it is indexed by. ``None`` when no history was presented, which
    #: is a §14-incomplete release and says so rather than shipping an unlabelled curve.
    curves: dict[str, Any] | None
    #: §14's milestone set, riding along for Phase 2's compute-scaling plot. Possibly empty.
    milestones: list[dict[str, Any]]
    #: Why every milestone's probe is null. :data:`PROBE_NOTE`, carried in the artifact so the
    #: caveat travels with the files rather than living in an issue nobody reads.
    probe_note: str
    #: The whole recipe, so an ablation is a diff against a released config rather than prose.
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        return path


def export_encoder(
    checkpoint: Path | str,
    out_dir: Path | str,
    *,
    init_file: Path | str | None = None,
    history: Path | str | None = None,
    checkpoint_dir: Path | str | None = None,
    verify_timm: bool = True,
    hash_milestones: bool = True,
) -> ReleaseManifest:
    """Write one run's §14 release into ``out_dir``, and return the manifest.

    Writes ``encoder.pt`` and ``manifest.json`` always, and ``curves.csv`` when ``history`` is
    given. ``checkpoint_dir`` defaults to the directory the exported checkpoint sits in, which is
    where its milestone siblings are; pass it explicitly when they live elsewhere.

    ``verify_timm`` exists for tests that have no timm-shaped architecture to check against, and
    defaults to on everywhere else: the strict load is the assertion this whole module is built
    around, so making it optional at the call site the release uses would be a way to lose it.
    """
    out_dir = Path(out_dir)
    ckpt = load_checkpoint(checkpoint)
    arch = ckpt.config.architecture
    state = encoder_state_dict(ckpt.model, arch)
    name = assert_timm_loadable(state, arch) if verify_timm else timm_model_name(arch)

    out_dir.mkdir(parents=True, exist_ok=True)
    encoder_path = out_dir / "encoder.pt"
    torch.save(state, encoder_path)

    curves: dict[str, Any] | None = None
    probed: list[int] = []
    if history is not None:
        rows = curve_rows(read_history(history))
        probed = probe_epochs(rows)
        curves_path = write_curves(rows, out_dir / "curves.csv")
        curves = {
            "file": curves_path.name,
            "md5": checkpoint_md5(curves_path),
            "epoch_axis": EPOCH_AXIS,
            "epoch_axis_note": (
                "epochs completed, 1-based. W&B's `_step` is the 0-based index of the epoch that "
                "just ended; this column is `_step + 1` and matches RunState.epoch (P88 §1)"
            ),
            "rows": len(rows),
            "columns": curve_columns(rows),
            "probed_epochs": probed,
        }

    milestones = milestone_set(
        Path(checkpoint).parent if checkpoint_dir is None else checkpoint_dir,
        probed_epochs=probed,
        hash_files=hash_milestones,
    )

    manifest = ReleaseManifest(
        source_checkpoint=str(checkpoint),
        source_checkpoint_md5=checkpoint_md5(checkpoint),
        epochs_completed=ckpt.state.epoch,
        steps=ckpt.state.step,
        config_hash=ckpt.config_hash,
        timm_model=name,
        timm_create_kwargs={"in_chans": arch.in_chans, **TIMM_CREATE_KWARGS},
        pos_encoding=arch.pos_encoding,
        encoder_tensors=len(state),
        encoder_parameters=sum(int(v.numel()) for v in state.values()),
        encoder_file=encoder_path.name,
        encoder_md5=checkpoint_md5(encoder_path),
        fingerprint=ckpt.fingerprint.to_dict(),
        init_provenance=init_provenance(ckpt.config, init_file=init_file).to_dict(),
        curves=curves,
        milestones=milestones,
        probe_note=PROBE_NOTE,
        config=ckpt.config.to_dict(),
    )
    manifest.write(out_dir / "manifest.json")
    return manifest


# --- CLI ----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m pretrain.release",
        description="Export §14's encoder-only release weights and manifest from a checkpoint.",
    )
    p.add_argument("checkpoint", help="the full MAE checkpoint to export from")
    p.add_argument("out_dir", help="directory to write encoder.pt and manifest.json into")
    p.add_argument(
        "--init-file",
        default=None,
        help=(
            "the checkpoint this run warm-started from, so its provenance can be verified. "
            "Without it a warm-started export makes no licence claim; scratch runs need nothing"
        ),
    )
    p.add_argument(
        "--history",
        default=None,
        help=(
            "this run's exported W&B history as JSON, shipped as §14's training curves. Without "
            "it the release carries no curve, which is a §14-incomplete release"
        ),
    )
    p.add_argument(
        "--checkpoint-dir",
        default=None,
        help="where the milestone siblings live (default: the exported checkpoint's directory)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = export_encoder(
        args.checkpoint,
        args.out_dir,
        init_file=args.init_file,
        history=args.history,
        checkpoint_dir=args.checkpoint_dir,
    )
    prov = manifest.init_provenance
    print(f"exported  {manifest.encoder_file}  ({manifest.encoder_md5})")
    print(f"  timm      {manifest.timm_model}  {manifest.timm_create_kwargs}")
    print(f"  pos_embed {manifest.pos_encoding}  (materialized, strict load verified)")
    print(f"  weights   {manifest.encoder_tensors} tensors, {manifest.encoder_parameters:,} params")
    print(f"  position  epoch {manifest.epochs_completed} completed, step {manifest.steps:,}")
    print(f"  init      {prov['mode']}  ({'verified' if prov['verified'] else 'UNVERIFIED'})")
    print(f"            {prov['note']}")
    if manifest.curves is None:
        print("  curves    NONE — §14 requires them; re-run with --history")
    else:
        c = manifest.curves
        print(f"  curves    {c['file']}  {c['rows']} rows on {c['epoch_axis']}, "
              f"{len(c['columns'])} columns, probe at {len(c['probed_epochs'])} epochs")
    if manifest.milestones:
        at = ", ".join(str(m[EPOCH_AXIS]) for m in manifest.milestones)
        print(f"  milestone {len(manifest.milestones)} riding along (completed {at}); "
              f"none carries a probe number (#131)")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
