"""§6's augmentation: the dihedral group, applied to the image *and* its land mask (#91).

The interesting risk here is not "does it rotate". It is the coupling. `land_mask` is what #96
weights the loss by, so an image and a mask that drift apart do not crash, do not show up in a
loss curve, and simply score the wrong pixels — ocean weighting applied to a rotated copy of
where the ocean used to be. So the coupling is *decoded from the output and asserted*, sample
by sample, rather than being taken on trust from the fact that one variable was used twice.

The transform each sample received is recovered by matching the output against all eight
candidates (:func:`_element_of`), which needs a pattern no two group elements agree on;
:func:`test_the_fixture_pattern_distinguishes_all_eight` is that precondition, checked rather
than assumed, because every decode-based test below is worthless without it.

The seeding tests are the other half, and since #104 they are a different set of claims. The
element is a pure function of ``(run seed, epoch, sample index)`` rather than a draw from a
per-process stream, so what has to hold is that **the same sample of the same epoch gets the
same element wherever and whenever it is read** — in the parent, in any of eight workers, and
after a kill and a resume. That is what makes plan §11's exact resume hold at the shipped
``num_workers = 8`` (`test_pretrain_checkpoint.py` runs the gate itself), and it replaces the
old per-worker-stream tests wholesale: workers cannot collide when the key is per *sample*.

The one thing that can still fail silently is the epoch's trip into a worker — it rides on an
:class:`~pretrain.augment.EpochIndex`, and a lost attribute would make every epoch apply the
identical transform to the identical tile. Hence a test for exactly that, through real worker
processes.
"""

from __future__ import annotations

import pickle

import pytest
import torch

from pretrain.augment import (
    DIHEDRAL_ORDER,
    DihedralDataset,
    EpochIndex,
    apply_dihedral,
    dihedral_element,
    make_augmented,
)
from pretrain.config import Data

#: Small enough that 2,400 samples through a DataLoader is a fast test, big enough that the
#: eight group elements act differently on it.
PX = 8

#: Enough draws that "all eight are reachable and roughly uniform" is a claim about the
#: distribution rather than about one lucky epoch.
DRAWS = 2400


def _image() -> torch.Tensor:
    """A (1, PX, PX) tile whose every pixel value is distinct — so any position permutation
    that is not the identity produces a different tensor."""
    return torch.arange(PX * PX, dtype=torch.float32).reshape(1, PX, PX)


def _land_mask() -> torch.Tensor:
    """A (PX, PX) bool mask marking two adjacent cells of the top-left corner.

    Two cells, not one: a single marked corner cannot tell a rotation from its mirror, and the
    mirror pairs are exactly the ones a "the mask follows the image" bug is likeliest to swap.
    """
    return (_image()[0] < 2).clone()


class _FakeTiles(torch.utils.data.Dataset):
    """A stand-in for `dataset.loader`'s Dataset: the ADR-0003 sample dict, same tile every time.

    Same tile deliberately — with the sample held fixed, everything that varies across draws is
    the transform, which is what these tests measure. The real read path is exercised against
    the corpus fixture further down.
    """

    def __init__(self, n: int = DRAWS) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> dict:
        return {
            "image": _image(),
            "land_mask": _land_mask(),
            "mean": torch.tensor(500.0),
            "std": torch.tensor(200.0),
            "tile_id": f"tile_{index}",
            "stratum_id": index % 20,
        }


def _element_of(before: torch.Tensor, after: torch.Tensor) -> int:
    """Which group element carried ``before`` to ``after``. Raises if it is not exactly one."""
    matches = [k for k in range(DIHEDRAL_ORDER) if torch.equal(apply_dihedral(before, k), after)]
    assert len(matches) == 1, f"{len(matches)} of the 8 elements match, expected exactly one"
    return matches[0]


def _keys(n: int, epoch: int = 0) -> list[EpochIndex]:
    """The keys epoch ``epoch`` would hand a dataset of ``n`` samples, in index order.

    The real ones come out of :class:`pretrain.train.EpochShuffleSampler` permuted; the order is
    irrelevant to a pure function of the key, and building them here keeps these tests off the
    training package.
    """
    return [EpochIndex(i, epoch) for i in range(n)]


def _elements(dataset, indices=None) -> list[int]:
    """The element applied to each sample's *image*, in order."""
    indices = _keys(len(dataset)) if indices is None else indices
    return [_element_of(_image(), dataset[i]["image"]) for i in indices]


def _loader_elements(dataset, *, keys, **kwargs) -> list[int]:
    """The elements a pass applies, read off batches from a DataLoader driven by ``keys``."""
    from dataset.loader import build_dataloader

    loader = build_dataloader(
        dataset, shuffle=False, sampler=keys, pin_memory=False, **kwargs
    )
    return [_element_of(_image(), image) for batch in loader for image in batch["image"]]


# --- The fixture's own precondition -------------------------------------------


def test_the_fixture_pattern_distinguishes_all_eight():
    # every decode below rests on this: if two elements agreed on this pattern, `_element_of`
    # could not tell them apart and a coupling bug could hide inside the ambiguity.
    for tensor in (_image(), _land_mask()):
        transformed = [apply_dihedral(tensor, k) for k in range(DIHEDRAL_ORDER)]
        assert len({t.numpy().tobytes() for t in transformed}) == DIHEDRAL_ORDER


# --- The group ----------------------------------------------------------------


def test_the_eight_elements_are_the_four_rotations_and_their_mirrors():
    x = _image()
    assert torch.equal(apply_dihedral(x, 0), x)
    assert torch.equal(apply_dihedral(x, 2), torch.rot90(x, 1, dims=(-2, -1)))
    assert torch.equal(apply_dihedral(x, 1), torch.flip(x, dims=(-1,)))


def test_an_element_outside_the_group_is_refused():
    for k in (-1, DIHEDRAL_ORDER):
        with pytest.raises(ValueError, match="dihedral"):
            apply_dihedral(_image(), k)


def test_every_element_is_reachable_and_roughly_uniform():
    counts = [0] * DIHEDRAL_ORDER
    for k in _elements(DihedralDataset(_FakeTiles(), seed=0)):
        counts[k] += 1
    expected = DRAWS / DIHEDRAL_ORDER
    assert min(counts) > 0, f"unreachable element(s): {counts}"
    assert all(abs(c - expected) < 0.2 * expected for c in counts), counts


# --- The coupling: the failure #96 would inherit silently ----------------------


def test_the_image_and_the_land_mask_receive_the_identical_element():
    # the whole ticket, in one assertion: if these ever disagree, the ocean weighting in #96
    # scores a rotated copy of where the ocean used to be, and nothing else notices.
    dataset = DihedralDataset(_FakeTiles(n=200), seed=1)
    for i in range(len(dataset)):
        sample = dataset[i]
        assert _element_of(_image(), sample["image"]) == _element_of(
            _land_mask(), sample["land_mask"]
        )


def test_the_coupling_holds_through_a_multi_worker_dataloader():
    from dataset.loader import build_dataloader

    dataset = DihedralDataset(_FakeTiles(n=64), seed=2)
    loader = build_dataloader(
        dataset, batch_size=8, num_workers=2, shuffle=False, pin_memory=False
    )
    for batch in loader:
        for image, mask in zip(batch["image"], batch["land_mask"]):
            assert _element_of(_image(), image) == _element_of(_land_mask(), mask)


# --- Positions move; values do not --------------------------------------------


def test_pixel_values_are_permuted_never_changed():
    x = _image()
    for k in range(DIHEDRAL_ORDER):
        y = apply_dihedral(x, k)
        assert y.shape == x.shape and y.dtype == x.dtype
        assert torch.equal(y.flatten().sort().values, x.flatten().sort().values)


def test_the_land_mask_stays_boolean_and_keeps_its_land_fraction():
    mask = _land_mask()
    for k in range(DIHEDRAL_ORDER):
        rotated = apply_dihedral(mask, k)
        assert rotated.dtype == torch.bool
        assert int(rotated.sum()) == int(mask.sum())


def test_the_sample_dict_is_otherwise_untouched():
    plain, augmented = _FakeTiles()[3], DihedralDataset(_FakeTiles(), seed=0)[3]
    assert set(augmented) == set(plain)  # the ADR-0003 schema, unchanged
    assert augmented["tile_id"] == plain["tile_id"]
    assert augmented["stratum_id"] == plain["stratum_id"]
    assert torch.equal(augmented["mean"], plain["mean"])
    assert torch.equal(augmented["std"], plain["std"])


def test_the_underlying_sample_is_not_mutated_in_place():
    # a memmap-backed sample dict is shared state in a worker; transforming it in place would
    # corrupt the next read of the same tile rather than augment this one.
    underlying = _FakeTiles()
    DihedralDataset(underlying, seed=0)[0]
    assert torch.equal(underlying[0]["image"], _image())


# --- The draw is a function of (seed, epoch, index) ---------------------------


def test_the_element_depends_on_all_three_of_the_seed_the_epoch_and_the_index():
    # the whole of #104's seeding design in one assertion. Drop any of the three from the key
    # and one of these equalities becomes an inequality: a seed-blind element makes two runs
    # of different seeds identical, an epoch-blind one turns augmentation into a fixed
    # relabelling of the corpus, an index-blind one gives a whole batch the same transform.
    base = dict(seed=7, epoch=2, index=5)
    here = dihedral_element(**base)
    assert dihedral_element(**base) == here  # and it is a *function*
    for name in ("seed", "epoch", "index"):
        moved = [dihedral_element(**{**base, name: v}) for v in range(9)]
        assert any(k != here for k in moved), f"the element ignores {name}"


def test_adjacent_keys_are_not_related_and_in_particular_not_diagonally():
    # `seed + epoch + index` would be a pure function too, and would hand the identical element
    # to (7, 2, 5) and (7, 3, 4) — every epoch a relabelled shift of the last, across the whole
    # corpus at once, and nothing downstream that could ever surface it. Hence a keyed hash.
    grid = [[dihedral_element(seed=7, epoch=e, index=i) for i in range(64)] for e in range(8)]
    for e in range(7):
        shifted = [grid[e][i + 1] for i in range(63)]
        assert grid[e + 1][:63] != shifted
        # and consecutive epochs differ on most tiles, not on a lucky few
        assert sum(a != b for a, b in zip(grid[e], grid[e + 1])) > 0.6 * 64


def test_the_same_seed_reproduces_the_transform_sequence():
    first = _elements(DihedralDataset(_FakeTiles(n=64), seed=7))
    second = _elements(DihedralDataset(_FakeTiles(n=64), seed=7))
    assert first == second


def test_different_seeds_diverge():
    first = _elements(DihedralDataset(_FakeTiles(n=64), seed=7))
    second = _elements(DihedralDataset(_FakeTiles(n=64), seed=8))
    assert first != second


def test_the_second_epoch_augments_differently_from_the_first():
    # a transform that repeated per epoch would be a fixed relabelling of the corpus rather
    # than augmentation — the model would see 500k tiles, each in one orientation, forever.
    dataset = DihedralDataset(_FakeTiles(n=32), seed=5)
    assert _elements(dataset, _keys(32, epoch=0)) != _elements(dataset, _keys(32, epoch=1))


def test_a_sample_gets_the_same_element_however_and_whenever_it_is_read():
    # the resume property, in miniature and without a checkpoint in sight: read the epoch
    # forwards, backwards, from a fresh wrapper, and interleaved with other epochs, and sample
    # i of epoch 3 is shown the same thing every time. A sequential stream has none of this —
    # its answer depends on how many samples this *process* has read, which is exactly what a
    # kill and a resume changes.
    keys = _keys(32, epoch=3)
    forwards = _elements(DihedralDataset(_FakeTiles(n=32), seed=5), keys)

    fresh = DihedralDataset(_FakeTiles(n=32), seed=5)
    backwards = _elements(fresh, list(reversed(keys)))
    _elements(fresh, _keys(32, epoch=9))  # another epoch, in between
    assert list(reversed(backwards)) == forwards
    assert _elements(fresh, keys) == forwards


def test_a_bare_index_reads_as_epoch_zero():
    # indexing the wrapper directly is what a test or a diagnostic does, and it has no epoch to
    # offer. Epoch 0 is the honest reading; the resumable loop cannot reach this path, because
    # `EpochCursor` refuses a loader whose sampler is not an EpochShuffleSampler.
    dataset = DihedralDataset(_FakeTiles(n=8), seed=5)
    assert _elements(dataset, list(range(8))) == _elements(dataset, _keys(8, epoch=0))


def test_the_epoch_survives_being_pickled():
    # an int subclass reduces through `int.__getnewargs__`, which knows nothing about the
    # attribute — so `EpochIndex.__reduce__` is written out by hand, and this is why.
    assert pickle.loads(pickle.dumps(EpochIndex(7, 3))).epoch == 3
    assert pickle.loads(pickle.dumps(EpochIndex(7, 3))) == 7


def test_an_epoch_index_is_usable_as_the_plain_index_it_is():
    # everything between the sampler and `DihedralDataset` — the batch sampler, the index
    # queue, the *wrapped* dataset's own __getitem__ — sees it as an int and must not care.
    key = EpochIndex(3, 5)
    assert [10, 11, 12, 13][key] == 13 and key + 1 == 4 and isinstance(key, int)
    # It must *print* as its index too: a wrapped dataset that builds an id or a cache key out
    # of the index would otherwise produce a different one for the augmented read path than for
    # every other caller. `int` has no `__str__` of its own, so overriding `__repr__` — which
    # `EpochIndex` does, because a debugger should show the epoch — takes `str()` with it
    # unless it is put back deliberately.
    assert f"{key}" == "3" and str(key) == "3" and f"{key:04d}" == "0003"
    assert _FakeTiles()[key]["tile_id"] == "tile_3"
    assert repr(key) == "EpochIndex(3, epoch=5)"


# --- Worker processes: the configuration that actually ships ------------------


def test_workers_apply_the_identical_elements_the_parent_would_have():
    """The gap #104 closed, at its root, and the strongest form the claim comes in.

    The shipped default is ``num_workers = 8``. Under the old per-worker NumPy stream the
    elements a worker applied were a function of *that process's* draw count and could not be
    predicted from the parent at all — which is what made plan §11's exact resume hold only at
    ``num_workers = 0``. As a pure function of the key they are the same list either way, and
    "the same list" is a much sharper assertion than "no two workers collided".

    Two epochs, because one would pass even if the epoch never reached the workers.
    """
    dataset = DihedralDataset(_FakeTiles(n=32), seed=5)
    for epoch in (0, 1):
        keys = _keys(32, epoch=epoch)
        single = _loader_elements(dataset, keys=keys, batch_size=8, num_workers=0)
        workered = _loader_elements(dataset, keys=keys, batch_size=8, num_workers=2)
        assert workered == single

    # ...and the two epochs are genuinely different, so the loop above compared two things.
    assert _loader_elements(
        dataset, keys=_keys(32, epoch=0), batch_size=8, num_workers=0
    ) != _loader_elements(dataset, keys=_keys(32, epoch=1), batch_size=8, num_workers=0)


# --- The config seam ----------------------------------------------------------


def test_make_augmented_wraps_when_the_config_asks_for_the_dihedral_group():
    dataset = _FakeTiles()
    assert isinstance(make_augmented(dataset, Data(), seed=0), DihedralDataset)


def test_make_augmented_returns_the_read_path_untouched_when_augmentation_is_off():
    dataset = _FakeTiles()
    assert make_augmented(dataset, Data(dihedral=False), seed=0) is dataset


def test_random_resized_crop_is_refused_rather_than_silently_dropped():
    # §6 deviates from MAE's default deliberately and nothing here implements RRC. A config
    # that asks for it must not train without it and report that it had it.
    with pytest.raises(ValueError, match="RandomResizedCrop"):
        make_augmented(_FakeTiles(), Data(random_resized_crop=True), seed=0)


# --- Guards -------------------------------------------------------------------


def test_a_non_square_tile_is_refused():
    # 90 degree rotation of a non-square array changes its shape; the tile is 224x224 by
    # construction, so this can only mean something upstream is wrong. Better here than as a
    # shape mismatch inside the patch embed.
    class _Rectangular(_FakeTiles):
        def __getitem__(self, index):
            sample = super().__getitem__(index)
            sample["image"] = sample["image"][:, : PX // 2, :]
            return sample

    with pytest.raises(ValueError, match="square"):
        DihedralDataset(_Rectangular(), seed=0)[0]


# --- Against the real read path -----------------------------------------------


@pytest.fixture
def augmented_train_split(split_corpus):
    """§6's augmentation over the real read path — the composition #93 will build."""
    from pretrain import data

    return make_augmented(
        data.SplitDataset(split_corpus, "train", blocks_path=None, assignment_path=None),
        Data(),
        seed=0,
    )


def test_it_wraps_the_split_dataset_and_yields_the_adr_0003_sample(augmented_train_split):
    sample = augmented_train_split[0]
    assert set(sample) == {"image", "land_mask", "mean", "std", "tile_id", "stratum_id"}
    assert sample["image"].shape == (1, 224, 224)
    assert sample["land_mask"].shape == (224, 224)


def test_it_batches_through_the_adr_0003_dataloader(augmented_train_split):
    from dataset.loader import build_dataloader

    loader = build_dataloader(
        augmented_train_split, batch_size=2, num_workers=0, pin_memory=False
    )
    batch = next(iter(loader))
    assert batch["image"].shape == (2, 1, 224, 224)
    assert batch["land_mask"].shape == (2, 224, 224)
