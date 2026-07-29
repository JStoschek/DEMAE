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

The seeding tests are the other half. A run reproduces off the DataLoader generator (ADR-0003),
and two workers must not draw the same stream — a failure that would halve the effective
augmentation and be invisible everywhere except here.
"""

from __future__ import annotations

import pytest
import torch

from pretrain.augment import DIHEDRAL_ORDER, DihedralDataset, apply_dihedral, make_augmented
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


def _elements(dataset, indices=None) -> list[int]:
    """The element applied to each sample's *image*, in order."""
    indices = range(len(dataset)) if indices is None else indices
    return [_element_of(_image(), dataset[i]["image"]) for i in indices]


def _loader_elements(dataset, *, seed: int | None, **kwargs) -> list[int]:
    """The elements an epoch applies, read off batches from a seeded DataLoader."""
    from dataset.loader import build_dataloader

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    loader = build_dataloader(
        dataset, shuffle=False, pin_memory=False, generator=generator, **kwargs
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


# --- Reproducibility ----------------------------------------------------------


def test_the_same_seed_reproduces_the_transform_sequence():
    first = _elements(DihedralDataset(_FakeTiles(n=64), seed=7))
    second = _elements(DihedralDataset(_FakeTiles(n=64), seed=7))
    assert first == second


def test_different_seeds_diverge():
    first = _elements(DihedralDataset(_FakeTiles(n=64), seed=7))
    second = _elements(DihedralDataset(_FakeTiles(n=64), seed=8))
    assert first != second


def test_the_dataloader_generator_seeds_the_workers():
    # ADR-0003's reproducibility note: the run's generator is the seed of record, so an
    # unseeded wrapper takes its stream from it rather than from an RNG of its own. Torch
    # hands that generator's `base_seed + worker_id` to each worker, and this is the epoch
    # that comes back out.
    kwargs = dict(batch_size=8, num_workers=2)
    first = _loader_elements(DihedralDataset(_FakeTiles(n=32)), seed=11, **kwargs)
    again = _loader_elements(DihedralDataset(_FakeTiles(n=32)), seed=11, **kwargs)
    other = _loader_elements(DihedralDataset(_FakeTiles(n=32)), seed=12, **kwargs)
    assert first == again
    assert first != other


def test_the_process_seed_is_the_stream_when_there_are_no_workers():
    # torch seeds *workers* from the loader's generator and does nothing to the main process's
    # RNG, so at num_workers=0 the seed of record is the ambient one — `Runtime.seed`, which
    # the run applies with `torch.manual_seed` exactly as `MaskedAutoencoder.from_config` does.
    #
    # Two epochs off *one* wrapper, because that is what a run is: the claim being pinned is
    # that a run reproduces, not that a freshly built object does. Testing it on a throwaway
    # wrapper per epoch would certify a property the production object never has to hold.
    def run(seed: int) -> list[list[int]]:
        torch.manual_seed(seed)
        dataset = DihedralDataset(_FakeTiles(n=32))
        return [_loader_elements(dataset, seed=None, num_workers=0) for _ in range(2)]

    assert run(11) == run(11)
    assert run(11) != run(12)


def test_the_second_epoch_augments_differently_from_the_first():
    # the flip side of reproducibility, and the reason the RNG is not reset per epoch: a stream
    # that restarted would hand every tile the same transform it got last time, which is a
    # fixed relabelling of the corpus rather than augmentation.
    torch.manual_seed(5)
    dataset = DihedralDataset(_FakeTiles(n=32))
    first = _loader_elements(dataset, seed=None, num_workers=0)
    assert first != _loader_elements(dataset, seed=None, num_workers=0)


@pytest.mark.parametrize("seed", [3, None])
def test_workers_do_not_share_a_transform_stream(seed):
    # two workers drawing the same stream would apply the same 16 transforms to consecutive
    # batches — half the augmentation, and visible nowhere but here.
    #
    # Both seeding paths, because they are different mechanisms and only one of them ships:
    # `seed=None` is production (torch's own `base_seed + worker_id`), an explicit seed splits
    # the entropy per worker here. A test of only the explicit path would leave the shipped one
    # unasserted.
    dataset = DihedralDataset(_FakeTiles(n=32), seed=seed)
    elements = _loader_elements(dataset, seed=13, batch_size=16, num_workers=2)
    assert elements[:16] != elements[16:]


@pytest.mark.parametrize("seed", [4, None])
def test_a_wrapper_iterated_single_process_first_still_forks_distinct_streams(seed):
    # the trap the lazy per-worker RNG exists for: an RNG built in the parent (by a
    # num_workers=0 pass, or a val loader) would otherwise be copied identically into every
    # worker, under fork and under spawn alike.
    dataset = DihedralDataset(_FakeTiles(n=32), seed=seed)
    _elements(dataset, indices=range(4))  # builds the parent's RNG
    elements = _loader_elements(dataset, seed=13, batch_size=16, num_workers=2)
    assert elements[:16] != elements[16:]


# --- The config seam ----------------------------------------------------------


def test_make_augmented_wraps_when_the_config_asks_for_the_dihedral_group():
    dataset = _FakeTiles()
    assert isinstance(make_augmented(dataset, Data()), DihedralDataset)


def test_make_augmented_returns_the_read_path_untouched_when_augmentation_is_off():
    dataset = _FakeTiles()
    assert make_augmented(dataset, Data(dihedral=False)) is dataset


def test_random_resized_crop_is_refused_rather_than_silently_dropped():
    # §6 deviates from MAE's default deliberately and nothing here implements RRC. A config
    # that asks for it must not train without it and report that it had it.
    with pytest.raises(ValueError, match="RandomResizedCrop"):
        make_augmented(_FakeTiles(), Data(random_resized_crop=True))


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
        data.SplitDataset(split_corpus, "train", blocks_path=None, assignment_path=None), Data()
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
