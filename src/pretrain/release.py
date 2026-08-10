"""§14's release artifacts: the encoder-only export, and the manifest that identifies it.

Plan §14 releases, per run, *"encoder-only weights (timm-loadable) + full MAE checkpoint +
config + ``dataset_version``/split-assignment fingerprints + **init provenance** + the training
curves"*. The full checkpoint, the config and the fingerprints already exist inside every
checkpoint payload (:func:`~pretrain.checkpoint.save_checkpoint`); what does not exist until
here is the **encoder-only file a consumer actually loads**, the **curves**, the **milestone
set**, and the manifest that says what all of it is.

Three of those carry a caveat or a claim that is part of the artifact rather than commentary
on it:

- **The weights state their own licence** (:func:`release_licence`), which the first export did
  not: it named every *neighbour's* licence and the *init's* licence and said nothing about the
  files in the directory. #103 had already settled the chain — an Apache-2.0 init means one set
  of terms across every arm — and the conclusion never reached the artifact.

- **The curves are indexed by** :data:`EPOCH_AXIS`, spelled out in the manifest. W&B's ``_step``
  is the 0-based index of the epoch that just ended and
  :attr:`~pretrain.checkpoint.RunState.epoch` is epochs completed, so the two are off by one for
  a run's entire length (#132). Neither is wrong; an unlabelled curve is.
- **No milestone carries a probe number**, because none was ever measured at a milestone epoch —
  milestones fire at completed {50, 100, 200, 400} and §10.3's probe at completed
  {1, 21, ..., 381}, and the intersection is empty (#131). :data:`PROBE_NOTE` ships that fact
  next to the files, and :func:`milestone_set` records how far the nearest real probe was so a
  reader who carries one forward knows they are inferring.

And one that is a claim rather than a caveat: :func:`model_card` writes the release's own
write-up, because P100's pre-release scoop re-check found §13's headline sentence was **already
false** — an Apache-2.0 ViT-B/16 on ~9 M single-channel Copernicus DEM patches has been
downloadable since 2025-02-20. :data:`NOVELTY_CLAIM` is the narrower sentence that survives,
:data:`RETIRED_CLAIMS` holds the ones that do not so a test can assert they never come back, and
:data:`RELATED_WORK` names the neighbours strongest-first — which is the only ordering that
survives review, since a reviewer will find TerraMind whether or not the card does.

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


def read_history_document(path: Path | str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One run's exported W&B history, plus whatever it declares about where it came from.

    Two accepted shapes. A bare ``[...]`` is rows and nothing else — usable, and unable to show
    which run it belongs to. A ``{"config_hash": ..., "run_id": ..., "history": [...]}`` document
    carries its own provenance, and :func:`export_encoder` will refuse it against a checkpoint
    whose config hash differs.

    That refusal matters for the same reason the split-blocks one does. A curve from a *different*
    run of the same project plots perfectly: right length, right columns, a loss that descends.
    Nothing about the released figure would look wrong, and it would be describing other weights.
    """
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, Mapping):
        rows = payload.get("history")
        meta = {k: v for k, v in payload.items() if k != "history"}
    else:
        rows, meta = payload, {}
    if not isinstance(rows, list):
        raise ValueError(f"{path} does not hold a list of history rows")
    return rows, meta


def read_history(path: Path | str) -> list[dict[str, Any]]:
    """Just the rows of :func:`read_history_document`, for callers with no provenance to check."""
    return read_history_document(path)[0]


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


# --- The Phase-2 evaluation-regions contract ----------------------------------

#: What the ``test`` geocells *are*, as a rule a Phase-2 task builder can act on (plan §9, §14).
#: Not a description of the split — a constraint on what may be built where.
#:
#: The separation numbers are **measured** (P77 §3), and the sentence this replaced was not. Plan
#: §9 argued that blocking at 10° "puts hundreds of km between train and test, far beyond terrain
#: autocorrelation ranges"; #77 measured the median at 97 km and the *minimum* at 3.0 km, and the
#: correction landed in `p77-split.md` without amending §9 — so the first export copied §9's
#: uncorrected sentence into the one artifact outside consumers read, in the paragraph justifying
#: why a "generalizes to unseen regions" claim is honest. A claim a reader can check against a
#: cell list we ship beside it has to be the measured one.
REGIONS_CONTRACT: Final = (
    "These geocells are the downstream evaluation regions. Every tile inside them was held out "
    "of pretraining, and Phase-2 task datasets (void filling, geomorphon labels, and anything "
    "else evaluated as transfer) are built here and never from `train` geocells. Holding out "
    "whole 10-degree cells bounds the *typical* separation and not the smallest: measured "
    "tile-to-nearest-train-tile distance has a median of 97 km and a p95 of 348 km, but a "
    "minimum of 3.0 km, with 3.0% of test tiles inside 10 km of training data and 12.8% inside "
    "25 km (P77 §3). A whole-cell hold-out has a perimeter. So a 'generalizes to unseen regions' "
    "claim is honest for the interior and needs a buffer at the rim — and the buffer is priced: "
    "dropping the train tiles within 10 km of a test cell costs 2,964 tiles (0.62% of train), "
    "and within 25 km, 12,931 (2.72%). Phase 1 left it unspent; a Phase-2 task built here should "
    "either spend it or report the rim separately."
)


def evaluation_regions(blocks: Path | str, fingerprint: Mapping[str, Any]) -> dict[str, Any]:
    """The ``test`` geocell list, restated as §14's Phase-2 contract and **verified against the
    run's own fingerprint**.

    The release already carries ``split_blocks_md5``, which is enough to *check* a blocks file
    and not enough to *read* one — a consumer holding only the release cannot tell which regions
    were held out. This lifts the list into the release, and refuses to do so unless the
    presented file is byte-for-byte the one this run recorded.

    That check is the whole point. Shipping a plausible-looking cell list drawn from a *different*
    split is this map's recurring failure shape one more time: the regions would look entirely
    reasonable, Phase 2 would build its evaluation sets inside geocells the encoder had actually
    trained on, and every downstream number would be quietly optimistic with nothing to show it.
    """
    blocks = Path(blocks)
    recorded = fingerprint.get("split_blocks_md5")
    actual = checkpoint_md5(blocks)
    if not recorded:
        raise ValueError(
            f"this run recorded no split_blocks_md5, so {blocks} cannot be shown to be the split "
            f"it trained against — and an unverifiable region list is worse than none, because "
            f"Phase 2 would build evaluation sets from it"
        )
    if actual != recorded:
        raise ValueError(
            f"{blocks} hashes to {actual} but this run trained against split blocks {recorded} — "
            f"refusing to ship one split's evaluation regions with another split's weights"
        )
    doc = json.loads(blocks.read_text())
    test = doc["test"]
    return {
        "block": test["block"],
        "n_cells": test["n_cells"],
        "cells": list(test["cells"]),
        "n_tiles": doc["totals"]["test"],
        "share": doc["shares"]["test"],
        "per_stratum": doc["per_stratum"]["test"],
        "split_blocks_md5": actual,
        "assignment_md5": doc["assignment_md5"],
        "contract": REGIONS_CONTRACT,
    }


# --- The licence the weights ship under ---------------------------------------
#
# #103 settled the licence *chain*: R4 warm-starts from an Apache-2.0 reproduction rather than
# the official checkpoint whose terms P101 showed are unknowable, so no released artifact
# inherits a non-commercial term and every one of them can ship under the same licence. What
# that argument never reached was the artifact. The first export named every neighbour's licence
# in its related-work section and the *init's* licence under provenance, and said nothing at all
# about the weights sitting in the directory — the one fact a consumer needs before using them,
# and the one this project had already spent a research ticket establishing.

#: The terms the released weights are offered under. Apache-2.0 rather than the repo's MIT,
#: because R4 is a derivative of Apache-2.0 weights and Apache-2.0 is what carries that chain
#: forward intact; R5 owes nothing to anyone and rides along under the same terms, so the release
#: has one licence rather than one per arm — which is what #103 bought.
RELEASE_LICENCE: Final = "Apache-2.0"
RELEASE_LICENCE_URL: Final = "https://www.apache.org/licenses/LICENSE-2.0"

#: Init licences a release may inherit and still ship under :data:`RELEASE_LICENCE`. Deliberately
#: a set of one: this is the only provenance the claim has been checked against, so a future arm
#: warm-started from something else has to extend this on purpose. P101 is the record of what
#: assuming a licence costs.
COMPATIBLE_INIT_LICENCES: Final = frozenset({"Apache-2.0"})

#: The canonical Apache-2.0 text, shipped **beside the weights** rather than linked, because
#: §4(a) asks a redistributor to give recipients a copy of the License and a release directory
#: pulled out of a bucket carries no repo with it.
LICENCE_FILE: Final = "LICENSE"
LICENCE_TEXT_MD5: Final = "3b83ef96387f14655fc854ddc3c6bd57"


def licence_text() -> str:
    """The packaged Apache-2.0 text, refused unless it hashes to :data:`LICENCE_TEXT_MD5`.

    Hashed rather than trusted because this is the one shipped file whose *content* nothing else
    in the release checks: a truncated or edited licence would look like a licence in every
    listing, and the artifact would be making a legal claim against text nobody read.
    """
    path = Path(__file__).parent / "licences" / f"{RELEASE_LICENCE}.txt"
    actual = checkpoint_md5(path)
    if actual != LICENCE_TEXT_MD5:
        raise ValueError(
            f"{path} hashes to {actual}, not the canonical {RELEASE_LICENCE} text "
            f"({LICENCE_TEXT_MD5}) — refusing to ship weights under a licence whose text has "
            f"been altered"
        )
    return path.read_text()


def release_licence(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Under what terms this run's weights may be redistributed, given where they came from.

    A claim is made only when the chain supports it: ``scratch`` owes nothing to anyone, and a
    warm start owes whatever its init's licence says. An init that is unverified, or verified
    under terms outside :data:`COMPATIBLE_INIT_LICENCES`, gets **no licence claim at all** rather
    than a default — the same rule :func:`init_provenance` applies to provenance, for the same
    reason. A release that silently claims Apache-2.0 over weights derived from a CC-BY-NC
    checkpoint would look exactly like this one in every listing.
    """
    mode = provenance.get("mode")
    inherited = (provenance.get("spec") or {}).get("licence")
    # Written sentence-cased at the source. The card renders these verbatim: an earlier version
    # called `str.capitalize()` on them to fix the leading letter, which lowercases *everything
    # after* it — and shipped "the corpus is copernicus glo-30" and "this repo's source code is
    # mit" to consumers before anyone read the published file.
    if mode == "scratch":
        claim, why = True, (
            "Trained from random initialization, so these weights derive from no other "
            "licensed work and are offered under the release's own terms."
        )
    elif provenance.get("verified") and inherited in COMPATIBLE_INIT_LICENCES:
        claim, why = True, (
            f"Warm-started from {(provenance.get('spec') or {}).get('name')}, which is "
            f"{inherited} — a permissive licence this release carries forward rather than "
            f"relicenses, so the derived weights ship under the same terms as their init (#103)."
        )
    else:
        claim, why = False, (
            "This run was warm-started from weights whose licence this export could not verify, "
            "so **no licence is claimed for the derived weights** and they must not be "
            "redistributed on the strength of this manifest. Re-run the export with the init "
            "file, or establish its terms, before publishing — see `init_provenance.note`."
        )
    return {
        "spdx": RELEASE_LICENCE if claim else None,
        "url": RELEASE_LICENCE_URL if claim else None,
        "file": LICENCE_FILE if claim else None,
        "text_md5": LICENCE_TEXT_MD5 if claim else None,
        "applies_to": (
            "These terms cover `encoder.pt` and the milestone checkpoints it was exported from. "
            "The corpus is Copernicus GLO-30 under its own terms and is not redistributed here; "
            "this repo's source code is MIT and is a separate grant from this one."
        ),
        "inherited_from": None if not inherited else {
            "name": (provenance.get("spec") or {}).get("name"),
            "url": (provenance.get("spec") or {}).get("url"),
            "licence": inherited,
            "licence_source": (provenance.get("spec") or {}).get("licence_source"),
        },
        "note": why,
    }


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
    #: Under what terms these weights may be redistributed, from :func:`release_licence`. Its
    #: ``spdx`` is ``None`` when the chain does not support a claim, which is a publishable state
    #: for an internal artifact and not one for an open release.
    licence: dict[str, Any]
    #: §14's training curves: the shipped file, its hash, and — the part that is not decoration —
    #: the name of the epoch axis it is indexed by. ``None`` when no history was presented, which
    #: is a §14-incomplete release and says so rather than shipping an unlabelled curve.
    curves: dict[str, Any] | None
    #: §14's milestone set, riding along for Phase 2's compute-scaling plot. Possibly empty.
    milestones: list[dict[str, Any]]
    #: §14's other Phase-2 interface: the ``test`` geocells, restated as the contract that task
    #: datasets are built there and nowhere else. ``None`` when no verified blocks file was
    #: presented — never a guess, for the reason :func:`evaluation_regions` gives.
    evaluation_regions: dict[str, Any] | None
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


# --- The write-up -------------------------------------------------------------
#
# P100 re-checked §13's scoop assumption before release and found it already broken: an
# Apache-2.0 ViT-B/16 pretrained on ~9 M single-channel Copernicus DEM patches has been on the
# hub since 2025-02-20. §13's sentence is retired, and what replaces it is narrower and
# checkable. The constants below are the release's *claims*, kept next to the code that ships
# them so that the card cannot be written from memory — and so that
# :func:`model_card` can be tested for the sentences that must never reappear.


#: What this release is first at. Every qualifier is load-bearing (P100 §2): MIM is not FSQ-VAE
#: tokenization; IBM publishes no probe or transfer number for its DEM tokenizer as a feature
#: extractor; TerraMesh samples the 10 m Sentinel grid rather than stratifying on terrain.
NOVELTY_CLAIM: Final = (
    "the first **masked-reconstruction** encoder on **raw single-channel** elevation, released "
    "and **evaluated as a transferable backbone**, pretrained on a **terrain-stratified** corpus"
)

#: Sentences that are false and must not reappear in any release text (P100 §2, §4.3). Held as
#: data rather than as prose in a doc, because :func:`model_card`'s test asserts against them —
#: a claim this specific coming back is exactly the kind of regression review does not catch.
RETIRED_CLAIMS: Final = (
    "first open-weights encoder pretrained on elevation alone",
    "first open encoder on elevation",
    "first open model on Copernicus GLO-30 elevation",
    "first open-weights model trained on Copernicus GLO-30",
)


@dataclass(frozen=True)
class RelatedWork:
    """One neighbouring open artifact, and the one thing that differs. P100 §4."""

    name: str
    url: str
    licence: str
    what_it_is: str
    differs: str


#: The nearest open elevation-pretrained artifacts, **strongest competitor first** — the only
#: ordering that survives review (P100 §6.3). A release that says "first" beside a
#: better-resourced IBM/ESA artifact doing the same thing is a credibility problem; naming it
#: first and stating the difference is a stronger claim because it is checkable.
RELATED_WORK: Final[tuple[RelatedWork, ...]] = (
    RelatedWork(
        name="TerraMind 1.0 DEM Tokenizer (IBM + ESA + FZ Jülich)",
        url="https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-Tokenizer-DEM",
        licence="Apache-2.0",
        what_it_is=(
            "a ViT-B/16 encoder on single-channel DEM, 20 epochs on ~9 M Copernicus DEM patches "
            "from TerraMesh — the same architecture and modality as this release, at ~18x the "
            "corpus, on the hub since 2025-02-20"
        ),
        differs=(
            "its objective is FSQ-VAE tokenization, not masked reconstruction, and its "
            "documented job is to encode DEM into codes for TerraMind's multimodal backbone and "
            "decode them back. No probe, kNN, linear-eval or transfer number is published for it "
            "as a feature extractor. Its corpus is co-registered to the 10 m Sentinel grid rather "
            "than stratified on terrain"
        ),
    ),
    RelatedWork(
        name="DINO-CV (Budj Bim stone-wall detection)",
        url="https://github.com/MLinArcheaomatics/BudjBimStoneWall-DINO-CV",
        licence="CC BY-NC 4.0 (non-commercial)",
        what_it_is=(
            "DINO cross-view self-distillation on 188,006 unlabelled DEM-derivative tiles; "
            "ResNet-50, WideResNet-50-2 and ViT-S/16 backbones, weights downloadable"
        ),
        differs=(
            "it pretrains on DEM *derivatives* — multi-directional hillshade and VAT, both "
            "3-channel visualizations — where the point of this release is that the derivative "
            "step can be skipped. One 301 km² site at 10 cm, so regional and three orders of "
            "magnitude off this corpus's resolution; DINO rather than MAE; ViT-S is its only ViT"
        ),
    ),
    RelatedWork(
        name="terrain-diffusion-30m",
        url="https://huggingface.co/xandergos/terrain-diffusion-30m",
        licence="MIT",
        what_it_is=(
            "a hierarchical generative diffusion stack trained on the Copernicus GLO-30 "
            "dataset — this release's own corpus — on the hub since 2025-12-17"
        ),
        differs=(
            "it is a terrain *synthesis* model and the released artifacts are diffusion UNets; "
            "nothing in it is offered as a transferable backbone. It is named here because it "
            "retires any 'first open model on Copernicus GLO-30 elevation' framing outright"
        ),
    ),
    RelatedWork(
        name="COP-GEN-Beta VAEs",
        url="https://huggingface.co/mespinosami/copgen-vaes",
        licence="CC-BY-4.0",
        what_it_is=(
            "a DEM-only KL-regularised VAE at 64x64 with 8 latent channels, released as one "
            "per-modality component of a multimodal generative system"
        ),
        differs=(
            "a convolutional VAE at low resolution rather than a ViT, and not offered as a "
            "representation encoder. The card does not state the DEM source, scale or resolution"
        ),
    ),
    RelatedWork(
        name="divae-tokenizer-ahn-448",
        url="https://huggingface.co/hwh-datascience/divae-tokenizer-ahn-448",
        licence="Apache-2.0",
        what_it_is=(
            "a third-party DiVAE tokenizer following TerraMind's methodology, ViT-B/16 on "
            "2-channel fused AHN6 elevation over the Netherlands"
        ),
        differs=(
            "a single-checkpoint project release at epoch 2 with no paper and no evaluation, "
            "and 2-channel rather than raw single-channel. Recorded for completeness rather than "
            "because it is competitive"
        ),
    ),
)

#: The obvious reviewer question, answered by saying which phase owns it (P100 §6.3(3)). Left
#: silent it reads as an oversight; stated, it is a scope boundary.
PHASE2_COMPARISON: Final = (
    "**How does this encoder compare to TerraMind's DEM tokenizer encoder on the same "
    "modality?** No head-to-head has been run, and this release does not claim one. It is a "
    "Phase-2 benchmark rather than a Phase-1 deliverable: it needs a downstream task suite that "
    "does not exist yet, and running it against an encoder its publisher has never evaluated as "
    "a feature extractor is a piece of work in its own right. The gap is named here rather than "
    "left for a reader to notice."
)

#: Why a *narrow* R4-vs-R5 gap must not be over-read in either direction (#86). Shown only on a
#: warm-started release, where it is the relevant caveat.
CONTINUAL_CAVEAT: Final = (
    "This arm was warm-started from an independent Apache-2.0 reproduction of the ImageNet MAE "
    "checkpoint, which trails the official weights by 0.9 pt ImageNet linear probe at the "
    "checkpoint used (#86). A narrow gap between this run and the scratch run should not be "
    "over-read in either direction."
)


def _related_work_markdown() -> str:
    lines = []
    for i, work in enumerate(RELATED_WORK, start=1):
        lines.append(f"{i}. **[{work.name}]({work.url})** — {work.licence}")
        lines.append(f"   - What it is: {work.what_it_is}.")
        lines.append(f"   - What differs: {work.differs}.")
    return "\n".join(lines)


def model_card(manifest: ReleaseManifest) -> str:
    """The release's write-up, rendered from ``manifest`` so its numbers cannot drift.

    Everything quantitative here is read off the manifest rather than typed, which is the point:
    a card maintained by hand beside a generated manifest is a card that will eventually
    describe a different artifact than the one in the directory.
    """
    prov = manifest.init_provenance
    licence = (prov.get("spec") or {}).get("licence")
    if prov.get("mode") == "scratch":
        init_line = (
            "Trained from random initialization. No external weights entered this model, which "
            "is a property of the configured code path rather than a claim about a file."
        )
    elif prov.get("verified"):
        init_line = (
            f"Warm-started from {(prov.get('spec') or {}).get('name')}, verified against the "
            f"file on tensor counts, parameter total, recorded epoch, size and md5. "
            f"Init licence: **{licence}**.\n\n{CONTINUAL_CAVEAT}"
        )
    else:
        init_line = (
            f"Warm-started from `{prov.get('checkpoint')}`, which was **not presented to the "
            f"export**. No URL, md5 or licence is claimed for the initialization — see the "
            f"manifest's `init_provenance.note`. Provenance is recognized, never assumed."
        )

    lic = manifest.licence
    if lic.get("spdx"):
        inherited = lic.get("inherited_from")
        licence_badge = f"`{lic['spdx']}` — full text in `{lic['file']}` beside these weights"
        licence_line = (
            f"**These weights are released under [{lic['spdx']}]({lic['url']}).** The full text "
            f"ships as `{lic['file']}` in this directory (md5 `{lic['text_md5']}`).\n\n"
            f"{lic['applies_to']}\n\n{lic['note']}"
            + (
                ""
                if not inherited
                else f"\n\nInherited from **[{inherited['name']}]({inherited['url']})** "
                     f"({inherited['licence']}, stated at {inherited['licence_source']}). That "
                     f"licence travels with these weights; it is not replaced by this one."
            )
        )
    else:
        licence_badge = "**none claimed** — see below before redistributing"
        licence_line = (
            f"**No licence is claimed for these weights, and they should not be published as "
            f"open weights in this state.** {lic['note']}\n\n{lic['applies_to']}"
        )

    curves = manifest.curves
    curves_line = (
        "No training curves were shipped with this export, which makes it **§14-incomplete**."
        if curves is None
        else (
            f"`{curves['file']}` — {curves['rows']} rows, {len(curves['columns'])} columns, "
            f"indexed by `{curves['epoch_axis']}`.\n\n"
            + (
                ""
                if curves.get("config_hash")
                else "> **These curves are not shown to belong to these weights.** The history "
                     "file declared no config hash, so nothing here rules out its coming from "
                     "another run of the same project.\n\n"
            )
            + f"> **Read the epoch axis carefully.** {curves['epoch_axis_note']}. The two "
            f"conventions are off by one for the run's whole length and neither is wrong, so a "
            f"curve read against the other one puts every point an epoch early."
        )
    )

    milestones = (
        "None shipped."
        if not manifest.milestones
        else "\n".join(
            f"- `{m['file']}` — {m[EPOCH_AXIS]} epochs completed, step {m['step']:,}, "
            f"probe: none (nearest measured probe: {m['nearest_probed_epoch']} epochs completed)"
            for m in manifest.milestones
        )
    )

    regions = manifest.evaluation_regions
    if regions is None:
        regions_line = (
            "### Evaluation regions (Phase-2 contract)\n\n"
            "No verified evaluation-region contract shipped with this export. The `test` geocells "
            "are reserved as the downstream evaluation regions, but this release does not name "
            "them — a Phase-2 task builder must read the committed split blocks whose md5 is "
            "above, and check it against that hash before building anything."
        )
    else:
        cells = "\n".join(f"- `{c}`" for c in regions["cells"])
        regions_line = (
            f"### Evaluation regions (Phase-2 contract)\n\n"
            f"**{regions['n_cells']} held-out `{regions['block']}`s — "
            f"{regions['n_tiles']:,} tiles ({regions['share']:.2%} of the corpus).**\n\n"
            f"{regions['contract']}\n\n{cells}\n\n"
            f"Verified against this run's `split_blocks_md5` "
            f"(`{regions['split_blocks_md5']}`) at export time, so these are the regions this "
            f"encoder was actually held out of, not a plausible list from another split. Machine-"
            f"readable in `evaluation_regions.json`."
        )

    kwargs = ", ".join(f"{k}={v!r}" for k, v in manifest.timm_create_kwargs.items())
    fp = manifest.fingerprint
    return f"""# DEMAE — {manifest.timm_model} pretrained on single-channel elevation

A masked-autoencoder encoder pretrained on raw single-channel Copernicus GLO-30 elevation
tiles, released as a transferable backbone for downstream terrain tasks.

- **Architecture:** {manifest.timm_model}, {manifest.encoder_parameters:,} parameters across \
{manifest.encoder_tensors} tensors
- **Trained for:** {manifest.epochs_completed} epochs completed, {manifest.steps:,} optimizer steps
- **Positional embedding:** `{manifest.pos_encoding}`
- **Config hash:** `{manifest.config_hash}`
- **Licence:** {licence_badge}

## Licence

{licence_line}

## Loading it

```python
import timm, torch

model = timm.create_model("{manifest.timm_model}", {kwargs})
state = torch.load("{manifest.encoder_file}", map_location="cpu", weights_only=True)
model.load_state_dict(state, strict=True)  # strict on purpose — see below
```

Load it **strictly**. `pos_embed` is a non-persistent buffer in the training model and is
therefore absent from its `state_dict()`, while timm holds the same tensor as a required learned
parameter. This export materializes `pos_embed` from the run's own architecture so that the
strict load succeeds. A `strict=False` load of an export that had *not* done that reports one
missing key and silently keeps timm's **randomly initialized** positional embedding — no loss,
shape or key report would differ. The axis ordering shipped here is `{manifest.pos_encoding}`,
recorded because two orderings are in use across this project's runs and they are each other's
transpose.

## What this is first at — and what it is not

This release is {NOVELTY_CLAIM}.

Each qualifier does real work, and the broader sentence does not hold. The following claims are
**false** and are not made here:

{chr(10).join(f'- ~~"{c}"~~' for c in RETIRED_CLAIMS)}

An Apache-2.0 ViT-B/16 pretrained on ~9 M single-channel Copernicus DEM patches has been
publicly downloadable since 2025-02-20, and an MIT-licensed generative model trained on
Copernicus GLO-30 since 2025-12-17. Both are named below.

## Related work

Nearest open elevation-pretrained artifacts, strongest first.

{_related_work_markdown()}

## The comparison this release does not make

{PHASE2_COMPARISON}

## Initialization provenance

{init_line}

## Training curves

{curves_line}

## Milestone checkpoints

Kept so that downstream performance can be plotted against pretraining compute.

{milestones}

> {manifest.probe_note}

## Data

- **Dataset version:** `{fp.get('dataset_version')}`
- **Tiles:** {fp.get('n_tiles') if fp.get('n_tiles') is None else format(fp['n_tiles'], ',')}
- **Selection list md5:** `{fp.get('selection_list_md5')}`
- **Split md5:** `{fp.get('split_md5')}` (blocks `{fp.get('split_blocks_md5')}`, \
state `{fp.get('split_state')}`)
- **Git SHA:** `{fp.get('git_sha')}`

{regions_line}

## Source

Exported from `{manifest.source_checkpoint}` (md5 `{manifest.source_checkpoint_md5}`).
`manifest.json` in this directory carries the full config and every field above in machine-
readable form.
"""


def export_encoder(
    checkpoint: Path | str,
    out_dir: Path | str,
    *,
    init_file: Path | str | None = None,
    history: Path | str | None = None,
    checkpoint_dir: Path | str | None = None,
    split_blocks: Path | str | None = None,
    verify_timm: bool = True,
    hash_milestones: bool = True,
) -> ReleaseManifest:
    """Write one run's §14 release into ``out_dir``, and return the manifest.

    Writes ``encoder.pt``, ``manifest.json`` and ``MODEL_CARD.md`` always; ``LICENSE`` when the
    provenance chain supports a licence claim, and nothing when it does not (see
    :func:`release_licence`); ``curves.csv`` when
    ``history`` is given; ``evaluation_regions.json`` when ``split_blocks`` is, and verifies that
    file against the run's own fingerprint before believing it. ``checkpoint_dir`` defaults to the
    directory the exported checkpoint sits in, which is where its milestone siblings are; pass it
    explicitly when they live elsewhere.

    ``verify_timm`` exists for tests that have no timm-shaped architecture to check against, and
    defaults to on everywhere else: the strict load is the assertion this whole module is built
    around, so making it optional at the call site the release uses would be a way to lose it.
    """
    out_dir = Path(out_dir)
    ckpt = load_checkpoint(checkpoint)
    arch = ckpt.config.architecture
    state = encoder_state_dict(ckpt.model, arch)
    name = assert_timm_loadable(state, arch) if verify_timm else timm_model_name(arch)

    # Everything that can refuse, refuses *before* anything is written. A half-written release is
    # worse than no release: the first file out is `encoder.pt`, which loads into timm perfectly
    # and carries none of the provenance that would say which run it came from — a valid weights
    # file with nothing beside it is exactly what this module spends its length preventing.
    rows: list[dict[str, Any]] = []
    meta: dict[str, Any] = {}
    if history is not None:
        raw, meta = read_history_document(history)
        declared = meta.get("config_hash")
        if declared and declared != ckpt.config_hash:
            raise ValueError(
                f"{history} declares config_hash {declared} but this checkpoint's is "
                f"{ckpt.config_hash} — refusing to ship one run's curves with another's weights"
            )
        rows = curve_rows(raw)
    regions = (
        None if split_blocks is None
        else evaluation_regions(split_blocks, ckpt.fingerprint.to_dict())
    )
    provenance = init_provenance(ckpt.config, init_file=init_file)
    licence = release_licence(provenance.to_dict())
    # Read before the first write, with everything else that can refuse: a release directory
    # holding weights and a licence file that turned out to be unreadable is worse than one
    # holding neither, because the listing looks complete.
    licence_body = licence_text() if licence["spdx"] else None

    out_dir.mkdir(parents=True, exist_ok=True)
    if licence_body is not None:
        (out_dir / LICENCE_FILE).write_text(licence_body)
    encoder_path = out_dir / "encoder.pt"
    torch.save(state, encoder_path)

    curves: dict[str, Any] | None = None
    probed: list[int] = []
    if history is not None:
        probed = probe_epochs(rows)
        curves_path = write_curves(rows, out_dir / "curves.csv")
        curves = {
            "file": curves_path.name,
            "md5": checkpoint_md5(curves_path),
            "run_id": meta.get("run_id"),
            "config_hash": meta.get("config_hash"),
            "provenance": (
                "verified: the history file declares this checkpoint's config hash"
                if meta.get("config_hash")
                else "UNVERIFIED: the history file declares no config_hash, so nothing shows "
                     "these curves belong to these weights rather than to another run"
            ),
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

    if regions is not None:
        (out_dir / "evaluation_regions.json").write_text(
            json.dumps(regions, indent=2, sort_keys=True) + "\n"
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
        init_provenance=provenance.to_dict(),
        licence=licence,
        curves=curves,
        milestones=milestones,
        evaluation_regions=regions,
        probe_note=PROBE_NOTE,
        config=ckpt.config.to_dict(),
    )
    manifest.write(out_dir / "manifest.json")
    (out_dir / "MODEL_CARD.md").write_text(model_card(manifest))
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
    p.add_argument(
        "--split-blocks",
        default=None,
        help=(
            "the committed split-blocks JSON, shipped as §14's Phase-2 evaluation-regions "
            "contract. Verified against this run's split_blocks_md5 and refused on a mismatch"
        ),
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
        split_blocks=args.split_blocks,
    )
    prov = manifest.init_provenance
    print(f"exported  {manifest.encoder_file}  ({manifest.encoder_md5})")
    print(f"  timm      {manifest.timm_model}  {manifest.timm_create_kwargs}")
    print(f"  pos_embed {manifest.pos_encoding}  (materialized, strict load verified)")
    print(f"  weights   {manifest.encoder_tensors} tensors, {manifest.encoder_parameters:,} params")
    print(f"  position  epoch {manifest.epochs_completed} completed, step {manifest.steps:,}")
    print(f"  init      {prov['mode']}  ({'verified' if prov['verified'] else 'UNVERIFIED'})")
    print(f"            {prov['note']}")
    lic = manifest.licence
    if lic["spdx"]:
        print(f"  licence   {lic['spdx']}  ({lic['file']} written, text md5 {lic['text_md5']})")
    else:
        print("  licence   NONE CLAIMED — these weights must not be published as open weights")
    if manifest.curves is None:
        print("  curves    NONE — §14 requires them; re-run with --history")
    else:
        c = manifest.curves
        print(f"  curves    {c['file']}  {c['rows']} rows on {c['epoch_axis']}, "
              f"{len(c['columns'])} columns, probe at {len(c['probed_epochs'])} epochs")
        print(f"            {c['provenance']}")
    if manifest.milestones:
        at = ", ".join(str(m[EPOCH_AXIS]) for m in manifest.milestones)
        print(f"  milestone {len(manifest.milestones)} riding along (completed {at}); "
              f"none carries a probe number (#131)")
    if manifest.evaluation_regions is None:
        print("  regions   NONE — Phase 2 has no evaluation-region contract; pass --split-blocks")
    else:
        r = manifest.evaluation_regions
        print(f"  regions   {r['n_cells']} {r['block']}s held out, {r['n_tiles']:,} tiles "
              f"({r['share']:.2%}), verified against this run's split_blocks_md5")
    print(f"  card      MODEL_CARD.md  ({len(RELATED_WORK)} related-work entries, "
          f"{len(RETIRED_CLAIMS)} retired claims named as false)")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
