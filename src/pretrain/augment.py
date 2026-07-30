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

## Seeding, and the per-worker stream

The draw comes from a NumPy generator built **lazily, inside whichever process is reading** —
the same fork-safety pattern as the loader's per-worker memmaps, and for a sharper reason. A
generator built in the parent is copied byte-for-byte into every worker (under fork, and under
spawn via pickle), so all of them would draw the identical sequence: the augmentation would
still look random, still cover all eight elements, and still deliver each sample the same
transform its neighbour got. Hence :attr:`DihedralDataset._rng_key` — the RNG is rebuilt
whenever the worker identity it was built for is not the one now reading.

Left unseeded (the default) the entropy is :func:`torch.initial_seed`, which inside a worker is
the DataLoader's ``base_seed + worker_id`` — drawn from the DataLoader's own generator. That is
ADR-0003's reproducibility note taken literally: the run's generator seeds the shuffle *and*
the augmentation, so one seed reproduces one epoch. An explicit ``seed`` overrides it for
tests and for anything that wants a pinned stream independent of the loader.

One asymmetry worth knowing rather than discovering: torch passes ``base_seed`` to *workers*
and leaves the main process's RNG alone, so at ``num_workers=0`` the loader's generator has no
say and the seed of record is the ambient one — :attr:`pretrain.config.Runtime.seed`, applied
with :func:`torch.manual_seed` the way :meth:`pretrain.model.MaskedAutoencoder.from_config`
already applies it. Either way a run reproduces from one number; it is simply a different
number in the two cases.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, get_worker_info

from .config import Data

if TYPE_CHECKING:  # torch does not re-export the type `get_worker_info` returns
    from torch.utils.data._utils.worker import WorkerInfo

#: |D4| — the four rotations times {identity, mirror}. Eight elements, uniform.
DIHEDRAL_ORDER: Final = 8

#: The sample keys the transform acts on. The other four entries of the ADR-0003 dict are
#: per-tile scalars and an id, none of which a rotation can touch: `mean` and `std` are
#: rotation-invariant by construction, and they are what inverts a reconstruction back to
#: metres for the §10.2 grids.
GEOMETRIC_KEYS: Final = ("image", "land_mask")

#: Mask for normalizing torch's 64-bit seed into the non-negative entropy NumPy wants.
_UINT64: Final = (1 << 64) - 1


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
    """

    def __init__(self, dataset, *, seed: int | None = None) -> None:
        self.dataset = dataset
        self.seed = seed
        # Built lazily, per reading process — see the module docstring. `None` here is what
        # makes the object safe to fork or pickle into a worker.
        self._rng: np.random.Generator | None = None
        self._rng_key: tuple[int | None, int | None] | None = None

    def __len__(self) -> int:
        return len(self.dataset)

    def _entropy(self, worker: WorkerInfo | None) -> int | list[int]:
        """This process's seed: the DataLoader generator's, or an explicit one split per worker."""
        if self.seed is None:
            # Inside a worker this is the DataLoader's `base_seed + worker_id`, drawn from the
            # loader's generator; in the main process it is the ambient torch seed.
            return torch.initial_seed() & _UINT64
        return [self.seed, 0 if worker is None else worker.id]

    def _rng_for_this_worker(self) -> np.random.Generator:
        worker = get_worker_info()
        key = (self.seed, None if worker is None else worker.id)
        if self._rng_key != key:
            self._rng = np.random.default_rng(self._entropy(worker))
            self._rng_key = key
        return self._rng

    def __getitem__(self, index: int) -> dict:
        sample = dict(self.dataset[index])
        k = int(self._rng_for_this_worker().integers(DIHEDRAL_ORDER))
        for key in GEOMETRIC_KEYS:
            _check_square(sample[key], key=key)
            sample[key] = apply_dihedral(sample[key], k)
        return sample

    def rng_state(self) -> dict | None:
        """This process's augmentation stream, as plain types — #94's checkpoint reads it.

        ``None`` when nothing has been read yet, which is an honest answer rather than a
        missing one: a checkpoint written before the first sample has no stream to record, and
        the process that resumes it seeds an identical one from the same ambient seed.

        **Only this process's.** At ``num_workers > 0`` the stream that matters lives in the
        workers, is unreachable from here, and is not what this returns — see
        :mod:`pretrain.checkpoint`'s module docstring for what that costs a resume.
        """
        return None if self._rng is None else self._rng.bit_generator.state

    def load_rng_state(self, state: dict | None) -> None:
        """Put a stream captured by :meth:`rng_state` back. ``None`` is a no-op, deliberately."""
        if state is None:
            return
        self._rng_for_this_worker().bit_generator.state = state

    def __repr__(self) -> str:
        return f"DihedralDataset({self.dataset!r}, seed={self.seed!r})"


def make_augmented(dataset, data: Data, *, seed: int | None = None) -> Dataset:
    """The §6 augmentation :class:`~pretrain.config.Data` asks for, wrapped around ``dataset``.

    Returns the dataset itself when augmentation is off, so a val loader — which #93 builds
    without augmentation, to keep §10.1's val loss comparable across epochs — costs nothing.
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
