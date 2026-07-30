"""§6's augmentation: the full dihedral group over the read path, and nothing else (#91).

Plan §6 settles the whole augmentation policy in one paragraph — *"horizontal + vertical flips
and 90 degree rotations (the full dihedral group)"*, geometric-only — and this module is that
paragraph, plus the two deviations from the MAE default it records as deviations:

- **No RandomResizedCrop.** MAE's gain from RRC over fixed crops is small (84.9 vs 84.7), and
  rescaling changes the effective GSD of a modality where scale is physically meaningful.
  Scale augmentation belongs with GSD-aware positional encoding in Phase 4. A config that asks
  for it is *refused* here rather than ignored (:func:`make_augmented`), because a run that
  trains without an augmentation it reported having is worse than one that will not start.
- **No intensity jitter.** Harmful in MAE's own ablation (Table 1e) and physically meaningless
  on per-tile-standardized elevation, where the offset and scale have already been divided out.

## Why a wrapper, and not a Dataset argument

ADR-0003's read path is consumed unchanged: :class:`dataset.loader.TileDataset` still resolves
``manifest[i] -> memmap``, derives the land mask before normalization, and returns its sample
dict; this wraps that dict and permutes two of its six entries. So augmentation is a decision
the *training* package makes, the ETL core stays torch-free and augmentation-free, and #93 can
build a val loader without one by not wrapping.

## The coupling, which is the actual risk

``image`` and ``land_mask`` are transformed by **one draw**, used twice. If they ever drift
apart the model still trains, the loss still descends, and #96's ocean weighting quietly scores
a rotated copy of where the ocean used to be — no crash, no curve, nothing. One variable used
twice is the cheapest possible defence, and `test_pretrain_augment.py` decodes the element back
out of both tensors sample by sample rather than trusting that it was.

## The draw is a hash, not a stream (#104)

The element is a **pure function of** ``(run seed, epoch, sample index)`` —
:func:`dihedral_element` — and not a sequential draw from a generator this object owns. That is
the whole of the seeding design, and it is worth saying why, because the obvious implementation
is the sequential one and #91 shipped it.

A generator has *position*, and position is process-local. Under ``num_workers > 0`` the reads
happen in worker processes, so a per-process stream has to be seeded per worker (or every worker
draws the identical sequence and each sample gets its neighbour's transform), and once it is
per-worker the parent can no longer see it: :mod:`pretrain.checkpoint` cannot capture it, and a
run killed and resumed silently redraws every dihedral element from the kill point on. That was
a real hole in plan §11's exact-resume gate at the shipped ``num_workers = 8``, and #94
documented it rather than closing it.

A hash has no position. Sample ``i`` of epoch ``e`` gets the same element whichever process
reads it, whether it is read now or after a kill and a resume, and whether it is the first
sample that process has read or the ten-thousandth. So:

- there is **no augmentation RNG stream to checkpoint** — the gap closes by deletion, not by
  plumbing the state out of the workers;
- **exact resume holds at any worker count**, which is what §15.4's checklist item actually
  needs, since ``num_workers = 0`` is not a configuration anyone runs;
- two workers cannot collide by construction, because the key differs per *sample* rather than
  per *process*.

## Where the epoch comes from

A worker is handed sample indices and knows nothing else, so the epoch has to travel with the
index: :class:`EpochIndex` is what :class:`pretrain.train.EpochShuffleSampler` yields, an ``int``
that also says which epoch it was drawn for. It is an ``int`` subclass rather than a tuple so
that everything downstream of the sampler — an unaugmented shuffled loader, plain list indexing,
the wrapped dataset itself — keeps working on it unchanged; and it defines :meth:`__reduce__`
explicitly, because the default pickling of an ``int`` subclass would drop the attribute on the
way into a worker and leave every epoch augmenting identically.

Indexing a :class:`DihedralDataset` with a plain ``int`` is epoch 0. That is the right answer for
the one caller that does it — a test, or anything reading the dataset directly — and the
resumable loop cannot reach it: :class:`pretrain.train.EpochCursor` refuses a loader whose
sampler is not an :class:`~pretrain.train.EpochShuffleSampler`.

The cost is a BLAKE2b of a short string per sample, measured at ~0.5 us, plus ~0.3 us to build
the key. Against a memmap tile read and a normalization that is well into the tens of
microseconds, that is not a number the §10.4 loader-wait fraction can see.
"""

from __future__ import annotations

import hashlib
from typing import Final

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .config import Data

#: |D4| — the four rotations times {identity, mirror}. Eight elements, uniform.
DIHEDRAL_ORDER: Final = 8

#: The sample keys the transform acts on. The other four entries of the ADR-0003 dict are
#: per-tile scalars and an id, none of which a rotation can touch: `mean` and `std` are
#: rotation-invariant by construction, and they are what inverts a reconstruction back to
#: metres for the §10.2 grids.
GEOMETRIC_KEYS: Final = ("image", "land_mask")


class EpochIndex(int):
    """A sample index that also says which epoch it was drawn for (#104).

    :class:`pretrain.train.EpochShuffleSampler` yields these; :class:`DihedralDataset` is the
    only thing that reads the :attr:`epoch` off them. Everything in between — the DataLoader's
    batch sampler, the index queue into a worker, the wrapped dataset's own ``__getitem__`` —
    treats it as the ``int`` it is, which is the reason it is an ``int`` subclass and not a
    tuple: an unaugmented shuffled loader indexes :class:`pretrain.data.SplitDataset` with the
    sampler's output directly, and a tuple would break that for no gain.
    """

    epoch: int

    def __new__(cls, index: int, epoch: int) -> EpochIndex:
        self = super().__new__(cls, index)
        self.epoch = epoch
        return self

    def __reduce__(self):
        """Pickle the epoch along with the index — spelled out, because the default drops it.

        An ``int`` subclass reduces through ``int.__getnewargs__``, which knows about the value
        and nothing about ``__dict__``. Left to itself, every key would arrive in the worker as
        epoch 0, every epoch would apply the same transform to the same tile, and the result
        would be a fixed relabelling of the corpus that looks exactly like augmentation from
        every angle except a second epoch's. `test_the_epoch_survives_being_pickled` and
        `test_workers_apply_the_identical_elements_the_parent_would_have` are the assertions,
        and this is the reason they exist.
        """
        return (EpochIndex, (int(self), self.epoch))

    def __repr__(self) -> str:
        return f"EpochIndex({int(self)}, epoch={self.epoch})"

    #: **Not** the repr. ``int`` defines no ``__str__`` of its own (``int.__str__`` *is*
    #: ``object.__str__``, which defers to ``__repr__``), so overriding ``__repr__`` alone
    #: silently changes what ``f"{index}"`` produces — and a wrapped dataset that builds an id,
    #: a cache key or a filename out of its index would start emitting
    #: ``tile_EpochIndex(3, epoch=5)`` for the augmented read path and ``tile_3`` for every
    #: other caller. An index that does not print as its own value is not a drop-in for one.
    #: ``int.__repr__``, not ``int.__str__``, for exactly the reason above.
    __str__ = int.__repr__


def dihedral_element(*, seed: int, epoch: int, index: int) -> int:
    """Which of the eight elements sample ``index`` of epoch ``epoch`` is shown, for run ``seed``.

    A keyed hash, for the reason :func:`pretrain.train.epoch_seed` is one and
    :mod:`dataset.sampling`'s tie-breaks are: adjacent keys must not produce related answers.
    ``seed + epoch + index`` would hand the same element to (seed 7, epoch 2, tile 5) and
    (seed 7, epoch 3, tile 4), which is a diagonal correlation across the corpus that nothing
    downstream would ever surface.

    The modulo is over the whole 64-bit digest — which is to say its low three bits, since
    :data:`DIHEDRAL_ORDER` is 8 and divides 2**64 exactly. That makes the draw exactly uniform
    with no rejection step, rather than the near-uniform a modulo of a non-power-of-two range
    would give.
    """
    digest = hashlib.blake2b(f"{seed}:{epoch}:{index}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % DIHEDRAL_ORDER


def apply_dihedral(x: Tensor, k: int) -> Tensor:
    """Apply element ``k`` of the dihedral group to the trailing two (spatial) axes.

    ``k = 2 * rotations + mirror``: the quarter-turns first, then an optional horizontal flip.
    Indexing the trailing axes rather than fixed ones is what lets the *same* call serve the
    ``(C, H, W)`` image and the ``(H, W)`` land mask — which is the point, since they must not
    be able to receive different transforms through different code paths.

    Contiguous on the way out: :func:`torch.rot90` and :func:`torch.flip` leave a tensor whose
    stride order the default collate would otherwise have to fix up per batch.
    """
    if not 0 <= k < DIHEDRAL_ORDER:
        raise ValueError(f"{k} is not a dihedral element (0..{DIHEDRAL_ORDER - 1})")
    rotations, mirror = divmod(k, 2)
    y = torch.rot90(x, rotations, dims=(-2, -1))
    if mirror:
        y = torch.flip(y, dims=(-1,))
    return y.contiguous()


def _check_square(x: Tensor, *, key: str) -> None:
    """Refuse a non-square tile, where a quarter-turn would silently change the shape.

    The tile is 224x224 by construction (ADR-0001/0002), so this can only mean something
    upstream is wrong — and the alternative is a shape mismatch surfacing inside the patch
    embed, several frames away from the thing that caused it. Same reasoning, and the same
    shape as, :meth:`pretrain.model.MaskedAutoencoder._check_tiles`.
    """
    if x.shape[-2] != x.shape[-1]:
        raise ValueError(
            f"{key} is {tuple(x.shape)}: the dihedral group needs a square tile, since a "
            f"quarter-turn of a non-square one changes its shape"
        )


class DihedralDataset(Dataset):
    """A read path with plan §6's augmentation on it: one dihedral draw per sample, applied to
    the image and its land mask alike.

    Wraps any dataset yielding the ADR-0003 sample dict — :class:`dataset.loader.TileDataset`,
    or the :class:`pretrain.data.SplitDataset` subset of one — and yields the same six keys,
    with :data:`GEOMETRIC_KEYS` transformed. Nothing about the wrapped dataset changes, and the
    sample it returns is not mutated: in a worker that dict can be backed by shared memory, and
    transforming it in place would corrupt the next read of the same tile.

    Holds no RNG and no state of any kind beyond the run's seed, which is what makes it
    identical in the parent and in all eight workers, and what leaves #94's checkpoint with no
    augmentation stream to capture — see the module docstring.
    """

    def __init__(self, dataset, *, seed: int) -> None:
        self.dataset = dataset
        #: The run's seed (:attr:`pretrain.config.Runtime.seed`) — the *whole* entropy of the
        #: augmentation, and the same number at every worker count. Required rather than
        #: defaulted, because a silently-zero seed is a run that reports one recipe's
        #: augmentation and gets another's.
        self.seed = seed

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        # `index` carries an epoch when it came from an EpochShuffleSampler and is a bare int
        # otherwise; see the module docstring for why epoch 0 is the right reading of a bare one.
        # The wrapped dataset is indexed with it unchanged — an EpochIndex *is* an int.
        sample = dict(self.dataset[index])
        k = dihedral_element(
            seed=self.seed, epoch=getattr(index, "epoch", 0), index=int(index)
        )
        for key in GEOMETRIC_KEYS:
            _check_square(sample[key], key=key)
            sample[key] = apply_dihedral(sample[key], k)
        return sample

    def __repr__(self) -> str:
        return f"DihedralDataset({self.dataset!r}, seed={self.seed!r})"


def make_augmented(dataset, data: Data, *, seed: int) -> Dataset:
    """The §6 augmentation :class:`~pretrain.config.Data` asks for, wrapped around ``dataset``.

    Returns the dataset itself when augmentation is off, so a val loader — which #93 builds
    without augmentation, to keep §10.1's val loss comparable across epochs — costs nothing.

    ``seed`` is the run's (:attr:`pretrain.config.Runtime.seed`) and has no default: it is now
    the entire entropy of the augmentation rather than a test-only override of an ambient
    stream, so a caller that forgets it should not be able to get a plausible-looking run.
    """
    if data.random_resized_crop:
        raise ValueError(
            "data.random_resized_crop is set, and §6 deliberately deviates from MAE's "
            "RandomResizedCrop default: rescaling changes the effective GSD of a modality "
            "where scale is physically meaningful, so scale augmentation waits for GSDPE in "
            "Phase 4. Nothing here implements it, and a run must not report an augmentation "
            "it did not have"
        )
    if not data.dihedral:
        return dataset
    return DihedralDataset(dataset, seed=seed)
