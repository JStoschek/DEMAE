"""Survival — the checkpoint a vanished instance is resumed from (#94, plan §11).

vast.ai instances *vanish permanently*, not just pause, so this is not a convenience module.
Plan §11 makes exact resume a gate and §15's pre-flight checklist makes it item 4, ahead of
every GPU dollar after R0. What that costs is spelled out here rather than assumed: a
checkpoint carries everything needed to continue, it is written so that a process killed
mid-write leaves the previous one intact, and loading one against the wrong corpus is refused
rather than absorbed.

## What "everything" means

Weights and optimizer state are the obvious two, and the ones a naive resume gets right. The
rest are the ones that fail quietly:

- **Position** — :class:`RunState`: epochs completed, optimizer steps completed, and how many
  micro-batches into the current epoch the run was. The third is what makes a mid-epoch resume
  continue the same shuffle rather than draw a fresh one (:class:`pretrain.train.EpochCursor`).
- **Every RNG stream** — :data:`RNG_STREAMS`. Five of them, and they are genuinely independent:
  the ambient torch generator the in-model mask draw reads, CUDA's per-device generators, NumPy's
  legacy global, Python's :mod:`random`, and the DataLoader's own :class:`torch.Generator`.
  Plus :data:`OPTIONAL_RNG_STREAMS`, which is the same list continued onto devices a checkpoint
  may predate — "the ambient generator" is per device, and CUDA is not the only one (#105).
- **The full config** — so a checkpoint says what recipe produced it without reference to a
  file that may not have travelled with it, and #97/#14 can read a released artifact's recipe
  off the artifact.
- **#89's fingerprint block** — which corpus, which selection, which split, which commit.

## The `num_workers` caveat, and how it was closed

This module shipped with a hole, and the hole is worth keeping on the page because the shape of
it recurs: **a stream cannot be checkpointed from a process that cannot see it.**
:class:`pretrain.augment.DihedralDataset` used to draw its element from a NumPy generator held
by whichever process was reading. At ``num_workers = 0`` that is this process and the state was
captured and restored like any other; at the shipped ``num_workers = 8`` it lived in worker
processes the parent cannot read, so the dihedral element each sample was shown was silently
redrawn on resume — which meant §11's bit-identity gate, and item 4 of §15's pre-flight
checklist, held only at a configuration no run uses.

#104 closed it by **deleting the stream** rather than by plumbing state out of the workers. The
element is now a pure function of ``(seed, epoch, sample index)``, so there is nothing
process-local left to capture: resume is exact at any worker count, and :data:`RNG_STREAMS` is
five entries rather than six. The general lesson, for anything added here later: state that has
to survive a resume belongs in a *pure function of position*, and only what genuinely cannot
be — the optimizer's moments, the mask draw's generator — belongs in this payload.

:data:`FORMAT_VERSION` is deliberately **not** bumped for that removal. A pre-#104 checkpoint
carries an extra ``rng["augmentation"]`` key, :func:`restore_rng` only checks for streams that
are *missing*, and every other field means what it always did — so a run in flight across the
change still resumes, which is the entire point of §11. What it does not get is bit-identity
with the run that was killed, since its dihedral elements from that point on come from the hash
rather than from the stream the checkpoint recorded. That is a code change, not a corrupt
checkpoint, and it is what the fingerprint's ``git_sha`` is for.

## The device the block arrives on, and why the fix is at the sink

The hole above had a twin, and it is on the page for the same reason.
:func:`pretrain.train.train_resumable` loads with ``map_location=trainer.device``, so on a GPU
box *every* tensor in the payload arrives on the GPU — the RNG block included, whose streams
are CPU ``ByteTensor``s by definition. :func:`restore_rng` then converted each one with
``.to(torch.uint8)``, which names a dtype and not a device: it returned a tensor that was
already ``uint8`` and still on the GPU, and ``torch.set_rng_state`` refused it. Exact resume —
§11's gate and item 4 of §15's checklist — therefore raised on every real run while the suite
stayed green, because the suite trains at ``device="cpu"`` where ``map_location`` moves nothing
(#105). That is #104's shape on a second axis, and #104's lesson did not generalise past the
axis it was learned on.

**The fix is at the sink**, in :func:`_cpu_byte`, at all three call sites. The alternative was
to stop device-mapping the block at :func:`load_checkpoint`, and it was not taken:
``map_location`` is one argument to one ``torch.load`` and cannot be applied per key, so
honouring it for the weights but not for the RNG block means either a second read of the file
or the same copy-back one layer further from the requirement. The requirement belongs to
``torch.set_rng_state``, not to the loader — and :func:`restore_rng` is public and also takes
dicts that never came from a file. The copy is free at this size: the block is ~10 KB against a
~300 MB payload at the shipped 24.7 M-parameter recipe, once per resume.

## And how a gate like that gets tested at all

The premise that this needs a rented box is wrong, in the useful direction. ``set_rng_state``
wants a CPU tensor from **every** accelerator, not from CUDA specially, so an Apple-silicon
laptop with no CUDA at all reproduces #105 line for line on ``mps`` — and the laptop is where
this suite runs. ``test_a_checkpoint_resumes_when_the_payload_arrived_on_a_device`` is therefore
an ordinary test, parametrized over whatever non-CPU devices the box has, asserting the same
bit-identity §15.4 asks for. It runs on ``mps`` here and on ``cuda`` on the rented box with
nothing changed, and it skips *visibly* on a box with neither.

It paid for itself on the first run: with the device-mapping fixed the resume stopped raising
and was still not bit-identical, because :func:`pretrain.model.random_masking` draws from the
ambient generator **of the tensor's own device** and only the CPU's and CUDA's were ever
captured. Hence :data:`OPTIONAL_RNG_STREAMS`. Two bugs, one axis, and neither reachable from a
test that trains on the CPU.

Beside it sit two tests that need no accelerator at all — one standing in for a device-mapped
stream, one for the CUDA surface — because a gate that only runs where the hardware happens to
be is how this file got here twice. Those hold the *class* anywhere; the device test holds the
*path* wherever there is a device to hold it on.

## Retention, and why the milestone set is read rather than written down

Last 3 plus the milestone set, and the set comes off :attr:`Schedule.milestones
<pretrain.config.Schedule>`. §14 keeps the milestones so Phase 2 can plot downstream
performance against pretraining compute — the cheapest interesting plot the field rarely
reports — and R5 (#87) may extend to 800 epochs. With the set hardcoded at the 50/100/200/400
default, the 800 checkpoint would be an ordinary one, and last-3 retention would delete the
endpoint of the very plot the milestones exist for. R2/R3 stop at 200 and their milestone sets
are shorter for the same reason, which :class:`~pretrain.config.Schedule` already handles by
dropping the unreachable ones.

Which checkpoints are protected is legible from the directory alone — a milestone's filename
says so — so retention is recomputable on a box that has only the synced files and no run.
"""

from __future__ import annotations

import contextlib
import os
import random
import re
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

from .config import RunConfig
from .fingerprint import DataFingerprint

#: Bumped when the payload's shape changes incompatibly. A checkpoint from a different version
#: is refused by name rather than half-read: the fields it is missing are exactly the ones whose
#: absence does not show up until the loss curve is already wrong.
FORMAT_VERSION: Final = 1

#: The RNG streams a checkpoint carries. Named here so the capture, the restore and the test
#: that they round-trip all read one list — a stream added to two of the three is a stream that
#: silently is not restored.
RNG_STREAMS: Final = ("torch", "cuda", "numpy", "python", "dataloader")

@dataclass(frozen=True)
class _DeviceStream:
    """An accelerator's ambient generator, as the three calls capturing and restoring it need.

    The functions are wrapped rather than referenced, so the ``torch`` attribute behind each is
    resolved when it is *called* and only after ``available`` has said yes. A rented Linux box
    imports this module too, and an accelerator's namespace is not somewhere to find out at
    import time whether it exists on the platform.
    """

    get: Callable[[], torch.Tensor]
    set: Callable[[torch.Tensor], None]
    available: Callable[[], bool]


#: The optional streams, keyed by the name their payload entry carries. One entry per device,
#: read by :func:`capture_rng` and :func:`restore_rng` *both* — which is the point, and the same
#: point :data:`RNG_STREAMS`' comment makes: a stream named in one of the two and not the other
#: is a stream that silently is not restored, and a single entry holding both halves is what
#: makes that unrepresentable rather than merely discouraged.
#:
#: ``mps`` is the only one because the ambient generator the in-model mask draw reads is *the
#: one belonging to the tensor's device* (:func:`pretrain.model.random_masking`), and "torch
#: plus CUDA" is not the whole set of devices — only the set this project rents. The laptop's is
#: the other one, and #105's gate runs there.
_DEVICE_STREAMS: Final = {
    "mps": _DeviceStream(
        lambda: torch.mps.get_rng_state(),
        lambda state: torch.mps.set_rng_state(state),
        lambda: torch.backends.mps.is_available(),
    ),
}

#: Streams whose *absence* from a payload is not a defect. The key is written on every box,
#: present device or not, so absence means one thing only: a checkpoint older than the stream.
#: They are kept out of :data:`RNG_STREAMS` — which :func:`restore_rng` refuses a checkpoint for
#: missing — for the same reason :data:`FORMAT_VERSION` was not bumped for #104's removal: every
#: checkpoint already sitting on a box predates this key, and refusing those would cost exactly
#: what §11 exists to protect.
OPTIONAL_RNG_STREAMS: Final = tuple(_DEVICE_STREAMS)

#: The fingerprint fields a resume refuses to disagree on: *which data*. ``corpus_dir`` and
#: ``git_sha`` are deliberately absent — §11 resumes on a fresh instance where the corpus is
#: mounted elsewhere and the checkout may have moved, and refusing on either would make every
#: legitimate resume look like a corrupted one. They are still recorded, and #97 reports them.
DATA_IDENTITY: Final = (
    "dataset_version",
    "n_tiles",
    "selection_list_md5",
    "split_md5",
    "split_blocks_md5",
)

#: §11: "every epoch *and* every 30-60 minutes of wall-clock". The midpoint, because the plan
#: gives a range rather than a number: at ~$0.31/hr the lost work either side of it is cents,
#: and the real cost of a shorter interval is checkpoint write time against the training loop.
DEFAULT_INTERVAL_SECONDS: Final = 45 * 60.0

#: §11: "keep last 3 + the milestone set".
DEFAULT_KEEP_LAST: Final = 3

_NAME = re.compile(r"^checkpoint_e(?P<epoch>\d+)_s(?P<step>\d+)(?P<milestone>_milestone)?\.pt$")


# --- Where a run is -----------------------------------------------------------


@dataclass(frozen=True)
class RunState:
    """The position a resume restarts from: the epoch, the step, and the offset inside both."""

    #: Epochs **completed**. 0 during the first one; 1 once epoch 0 has been walked whole.
    epoch: int = 0
    #: Optimizer steps completed. The unit the LR schedule is written in, so restoring it wrong
    #: restarts the warmup — which trains, and looks like nothing but a slightly odd curve.
    step: int = 0
    #: Micro-batches drawn from the loader in the current epoch. Always 0 at an epoch boundary.
    #: This is the number that makes a mid-epoch resume continue the interrupted shuffle.
    batches_into_epoch: int = 0

    def to_dict(self) -> dict:
        return {"epoch": self.epoch, "step": self.step, "batches_into_epoch": self.batches_into_epoch}

    @classmethod
    def from_dict(cls, d: dict) -> RunState:
        return cls(
            epoch=int(d["epoch"]),
            step=int(d["step"]),
            batches_into_epoch=int(d["batches_into_epoch"]),
        )


# --- Every RNG stream ---------------------------------------------------------


def _numpy_global_state() -> dict:
    """NumPy's legacy global state as plain types, so the payload needs no unpickler.

    ``np.random.get_state(legacy=False)`` hands back the 624-word Mersenne key as an ndarray;
    everything else in the dict is already an int or a float. Converting it is what lets a
    checkpoint be read with ``weights_only=True`` (see :func:`load_checkpoint`).
    """
    state = np.random.get_state(legacy=False)
    key = state["state"]["key"]
    return {
        "bit_generator": state["bit_generator"],
        "key": [int(v) for v in key],
        "pos": int(state["state"]["pos"]),
        "has_gauss": int(state["has_gauss"]),
        "gauss": float(state["gauss"]),
    }


def _restore_numpy_global(state: dict) -> None:
    np.random.set_state(
        {
            "bit_generator": state["bit_generator"],
            "state": {
                "key": np.array(state["key"], dtype=np.uint32),
                "pos": int(state["pos"]),
            },
            "has_gauss": int(state["has_gauss"]),
            "gauss": float(state["gauss"]),
        }
    )


def _cpu_byte(state: torch.Tensor) -> torch.Tensor:
    """The one form every ``set_rng_state`` on this page accepts: a CPU ``ByteTensor``.

    The **device** is named here and not only the dtype, and that is #105 in a line: ``.to()``
    given a dtype alone preserves the device, so on a box where the checkpoint was loaded with
    ``map_location="cuda"`` a conversion to ``torch.uint8`` returns a tensor that was already
    ``uint8`` and is still on the GPU. Every stream restored below goes through here, including
    the CUDA ones — CUDA's generators keep their state on the host like the rest.
    """
    return state.to("cpu", torch.uint8)


def capture_rng(*, generator: torch.Generator | None = None) -> dict:
    """Snapshot every stream — :data:`RNG_STREAMS` and :data:`OPTIONAL_RNG_STREAMS` both — from
    wherever each one lives.

    ``generator`` is the DataLoader's (:func:`pretrain.train.seeded_generator`). It is optional
    so a checkpoint written outside a loop still records the four ambient streams rather than
    silently recording none.

    An optional stream's key is written on every box whether or not the device is there, so the
    payload's *shape* does not depend on what wrote it — only its values do. A reader can then
    tell "this box had no MPS" (``None``) from "this checkpoint predates the stream" (no key),
    and only the second is something :func:`restore_rng` has to reason about.

    The read path contributes nothing here, and that is the #104 fix rather than an omission:
    :class:`pretrain.augment.DihedralDataset` holds no stream to snapshot.
    """
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    return {
        "torch": torch.get_rng_state(),
        "cuda": list(cuda),
        "numpy": _numpy_global_state(),
        "python": random.getstate(),
        "dataloader": None if generator is None else generator.get_state(),
        **{
            name: stream.get() if stream.available() else None
            for name, stream in _DEVICE_STREAMS.items()
        },
    }


def restore_rng(state: dict, *, generator: torch.Generator | None = None) -> None:
    """Put every captured stream back where it was.

    Every stream is installed as a CPU ``ByteTensor`` (:func:`_cpu_byte`) whatever device it
    arrived on, and the optional ones (:data:`OPTIONAL_RNG_STREAMS`) are put back when this box
    has the device and skipped when the payload predates them. ``state`` usually comes from
    :attr:`Checkpoint.rng`, and
    :func:`pretrain.train.train_resumable` loads with ``map_location=trainer.device`` — so on a
    GPU box the block arrives on the GPU along with the weights, and putting it back is this
    function's job rather than the loader's (#105; the module docstring argues the seam).

    CUDA state is restored only when the resumed box has as many devices as the killed one had.
    §11's resume lands on a *fresh* instance, and a single-GPU run moving to a single-GPU box is
    the only shape Phase 1 plans for — but a mismatch is skipped with the rest still restored,
    because a CPU-side resume of a GPU checkpoint is a thing people do to debug one. It **warns**
    rather than passing quietly: a stream that was not restored is exactly the kind of thing that
    turns a bit-identical resume into an almost-identical one, and "almost" is unfalsifiable
    after the fact.
    """
    missing = sorted(set(RNG_STREAMS) - set(state))
    if missing:
        raise ValueError(f"the checkpoint's RNG block is missing {missing}: it cannot resume exactly")
    torch.set_rng_state(_cpu_byte(state["torch"]))
    if state["cuda"] and torch.cuda.is_available():
        if len(state["cuda"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all([_cpu_byte(s) for s in state["cuda"]])
        else:
            warnings.warn(
                f"the checkpoint carries CUDA RNG state for {len(state['cuda'])} device(s) and "
                f"this box has {torch.cuda.device_count()}; that stream is not being restored, "
                f"so the resumed run will not be bit-identical to the killed one (plan §11)",
                stacklevel=2,
            )
    # Silent when absent, unlike the CUDA mismatch above, and the asymmetry is worth arguing
    # rather than assuming. A missing optional stream matters only to a run that trains *on*
    # that device, and no §11 run does: nothing this project rents is an MPS box, so the only
    # payloads with the key absent *and* the device in use would be ones a developer wrote
    # locally with `--device mps` before this key existed — which is not a shape that outlives
    # the change, because those runs are not the ones checkpoints are synced off-instance for.
    # Warning on absence alone would instead fire on the common case, a Mac reading a rented
    # checkpoint at `device="cpu"` to debug one, where the stream is never consulted at all.
    for name, stream in _DEVICE_STREAMS.items():
        if state.get(name) is not None and stream.available():
            stream.set(_cpu_byte(state[name]))
    _restore_numpy_global(state["numpy"])
    version, internal, gauss = state["python"]
    random.setstate((version, tuple(internal), gauss))
    if generator is not None and state["dataloader"] is not None:
        generator.set_state(_cpu_byte(state["dataloader"]))


# --- The file on disk ---------------------------------------------------------


@dataclass(frozen=True)
class CheckpointFile:
    """One checkpoint in a directory, as its name describes it. No payload is read."""

    path: Path
    epoch: int
    step: int
    #: Written at the end of a milestone epoch, so retention never deletes it. Carried in the
    #: filename rather than derived at prune time, so a directory that has been synced to a
    #: bucket and pulled onto a fresh box is still legible without the run's config.
    milestone: bool


def checkpoint_name(*, epoch: int, step: int, milestone: bool) -> str:
    """``checkpoint_e0050_s000012200[_milestone].pt`` — sorts by epoch, then by step."""
    return f"checkpoint_e{epoch:04d}_s{step:09d}{'_milestone' if milestone else ''}.pt"


def _temporary_name(name: str) -> str:
    """The name a half-written checkpoint carries: hidden, and outside :func:`discover`'s glob.

    Both properties matter. A hard kill (SIGKILL, or the instance simply going away) leaves the
    file behind with no chance to clean it up, and a partial checkpoint that a later resume can
    *find* is worse than no checkpoint at all — it loads weights that are partly the previous
    step's, and nothing about the resumed run looks wrong.
    """
    return f".{name}.tmp"


def discover(directory: Path | str) -> list[CheckpointFile]:
    """Every checkpoint in ``directory``, oldest first. An absent directory holds none."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    found = []
    for path in directory.glob("checkpoint_*.pt"):
        m = _NAME.match(path.name)
        if m is not None:
            found.append(
                CheckpointFile(
                    path=path,
                    epoch=int(m["epoch"]),
                    step=int(m["step"]),
                    milestone=bool(m["milestone"]),
                )
            )
    return sorted(found, key=lambda f: (f.step, f.epoch))


def latest(directory: Path | str) -> CheckpointFile | None:
    """The newest checkpoint by step, or ``None`` — which is a first launch, not an error."""
    found = discover(directory)
    return found[-1] if found else None


def atomic_save(payload: dict, path: Path) -> Path:
    """Write ``payload`` so that an interrupted write leaves the previous file untouched.

    tmp + ``os.replace``, which is atomic within a filesystem: at every instant the name either
    refers to the whole previous checkpoint or to the whole new one. The ``fsync`` before the
    rename is what makes that true across a *power* loss rather than only a process death —
    without it the rename can reach the disk before the bytes it renames.

    A failed write takes its temporary file with it. A killed *process* cannot, which is why
    :func:`_temporary_name` puts it outside :func:`discover`'s reach.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / _temporary_name(path.name)
    try:
        with open(tmp, "wb") as fh:
            torch.save(payload, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    # The rename itself is a directory entry, and it is durable only once the directory is.
    with contextlib.suppress(OSError):
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return path


def save_checkpoint(
    path: Path | str,
    *,
    config: RunConfig,
    fingerprint: DataFingerprint,
    state: RunState,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    rng: dict,
) -> Path:
    """Write one checkpoint atomically. :meth:`CheckpointPolicy.save` is the usual caller."""
    payload = {
        "format_version": FORMAT_VERSION,
        "config": config.to_dict(),
        "config_hash": config.config_hash,
        "fingerprint": fingerprint.to_dict(),
        "state": state.to_dict(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": rng,
    }
    return atomic_save(payload, Path(path))


@dataclass(frozen=True)
class Checkpoint:
    """A loaded checkpoint. Everything §11 asks one to carry, as objects rather than a dict."""

    path: Path
    config: RunConfig
    #: The hash **as recorded**, not recomputed: a checkpoint written by a different version of
    #: :mod:`pretrain.config` should report what it claimed, so the disagreement is visible.
    config_hash: str
    fingerprint: DataFingerprint
    state: RunState
    model: dict
    optimizer: dict
    rng: dict

    def verify_against(self, current: DataFingerprint) -> None:
        """Refuse, loudly, unless this checkpoint trained on the data in front of us now.

        Only :data:`DATA_IDENTITY` is compared — see that constant for what is deliberately
        excluded and why. A mismatch cannot be repaired downstream: the released encoder would
        record one ``dataset_version`` and have been trained across two.
        """
        differ = [
            f"  {name}: checkpoint {getattr(self.fingerprint, name)!r} "
            f"vs corpus {getattr(current, name)!r}"
            for name in DATA_IDENTITY
            if getattr(self.fingerprint, name) != getattr(current, name)
        ]
        if differ:
            raise ValueError(
                f"{self.path} was trained on different data than the corpus at "
                f"{current.corpus_dir}:\n" + "\n".join(differ) + "\n"
                "Resuming across that would produce a checkpoint whose recorded fingerprint "
                "names data it was only partly trained on (plan §11, §14). Point the run at "
                "the corpus this checkpoint names, or start a new run."
            )


def load_checkpoint(path: Path | str, *, map_location: Any = "cpu") -> Checkpoint:
    """Read a checkpoint. ``weights_only=True``, because these arrive from a bucket.

    Every field of the payload is a tensor or a plain Python type — :func:`_numpy_global_state`
    exists to keep it that way — so the restricted unpickler is enough. §11 syncs checkpoints
    off-instance and pulls them back onto a fresh box; making that path an arbitrary-code-
    execution one for the sake of storing an ndarray would be a poor trade.
    """
    path = Path(path)
    payload = torch.load(path, map_location=map_location, weights_only=True)
    version = payload.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"{path} is a format-{version} checkpoint and this is format {FORMAT_VERSION}. "
            "The fields a newer format adds are exactly the ones whose absence does not show "
            "up until the loss curve is already wrong, so it is refused rather than half-read"
        )
    return Checkpoint(
        path=path,
        config=RunConfig.from_dict(payload["config"]),
        config_hash=payload["config_hash"],
        fingerprint=DataFingerprint(**payload["fingerprint"]),
        state=RunState.from_dict(payload["state"]),
        model=payload["model"],
        optimizer=payload["optimizer"],
        rng=payload["rng"],
    )


# --- When to write one, and which to keep -------------------------------------


@dataclass
class CheckpointPolicy:
    """§11's two triggers and its retention rule, over one directory.

    Build it with :meth:`from_config` so the milestone set comes off the run's own
    :class:`~pretrain.config.Schedule` — see the module docstring for why that is not a detail.

    Mutable, unlike everything else in this package: it owns the wall-clock timer, and a timer
    is state by definition. Nothing about the *recipe* lives here.
    """

    directory: Path
    #: The epochs whose end-of-epoch checkpoint is protected from retention (§14).
    milestones: tuple[int, ...] = ()
    #: §11's wall-clock trigger. ``None`` disables it, leaving the per-epoch one.
    interval_seconds: float | None = DEFAULT_INTERVAL_SECONDS
    #: §11's per-epoch trigger.
    every_epoch: bool = True
    keep_last: int = DEFAULT_KEEP_LAST
    #: Monotonic by default — a wall-clock trigger driven by a clock that can step backwards
    #: over an NTP correction would sulk for the length of the correction.
    clock: Callable[[], float] = time.monotonic
    _last_save: float = field(init=False)

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)
        if self.keep_last < 1:
            raise ValueError(f"keep_last {self.keep_last} must be >= 1: a resume needs one")
        self._last_save = self.clock()

    @classmethod
    def from_config(cls, config: RunConfig, directory: Path | str, **changes) -> CheckpointPolicy:
        """The policy for a run: its own milestone set, and §11's defaults for the rest."""
        return cls(directory=Path(directory), milestones=config.schedule.milestones, **changes)

    def due(self, *, ends_epoch: bool) -> bool:
        """Is a checkpoint owed? Reads the clock; only :meth:`save` restarts it."""
        if ends_epoch and self.every_epoch:
            return True
        if self.interval_seconds is None:
            return False
        return self.clock() - self._last_save >= self.interval_seconds

    def is_milestone(self, state: RunState, *, ends_epoch: bool) -> bool:
        """§14's protection applies to *the end of* a milestone epoch, not to anything in it."""
        return ends_epoch and state.epoch in self.milestones

    def path_for(self, state: RunState, *, ends_epoch: bool) -> Path:
        return self.directory / checkpoint_name(
            epoch=state.epoch, step=state.step, milestone=self.is_milestone(state, ends_epoch=ends_epoch)
        )

    def save(
        self,
        *,
        config: RunConfig,
        fingerprint: DataFingerprint,
        state: RunState,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        rng: dict,
        ends_epoch: bool = True,
    ) -> Path:
        """Write one checkpoint, restart the wall-clock timer, and prune."""
        path = save_checkpoint(
            self.path_for(state, ends_epoch=ends_epoch),
            config=config,
            fingerprint=fingerprint,
            state=state,
            model=model,
            optimizer=optimizer,
            rng=rng,
        )
        self._last_save = self.clock()
        self.prune()
        return path

    def prune(self) -> list[Path]:
        """Delete everything that is neither one of the last :attr:`keep_last` nor a milestone.

        The two rules are a union, not a precedence: three recent checkpoints so a resume has
        somewhere to land even if the newest is the one that was interrupted, and every
        milestone regardless of age so §14's compute-vs-quality plot keeps its x-axis.
        """
        found = discover(self.directory)
        keep = {f.path for f in found[-self.keep_last :]} | {f.path for f in found if f.milestone}
        removed = []
        for f in found:
            if f.path not in keep:
                f.path.unlink(missing_ok=True)
                removed.append(f.path)
        return removed


def resolve_resume(directory: Path | str, resume: bool | str | Path | None) -> Path | None:
    """Which checkpoint a run should resume from, if any.

    ``False`` is a fresh start, ``True`` is "the newest in the directory, if there is one", and
    a path is that path. The ``True``-with-nothing-there case is deliberately *not* an error:
    on the rented box a first launch and a relaunch are the same command, and a flag that
    refused the first one would simply be removed from the command line and then forgotten.
    """
    if resume is False or resume is None:
        return None
    if resume is True:
        found = latest(directory)
        return None if found is None else found.path
    path = Path(resume)
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint at {path}")
    return path


__all__ = [
    "Checkpoint",
    "CheckpointFile",
    "CheckpointPolicy",
    "DATA_IDENTITY",
    "FORMAT_VERSION",
    "OPTIONAL_RNG_STREAMS",
    "RNG_STREAMS",
    "RunState",
    "atomic_save",
    "capture_rng",
    "checkpoint_name",
    "discover",
    "latest",
    "load_checkpoint",
    "resolve_resume",
    "restore_rng",
    "save_checkpoint",
]
