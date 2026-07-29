"""Split containment: the check that keeps the Phase-2 evaluation regions out of training (#89).

Plan §15.2 requires that **no test geocell is reachable from a train batch**, asserted in the
Dataset. These tests are that requirement: the honest split passes, each way a split can be
contaminated fails, and it fails at *construction* — a warning that scrolls past in a training
log would be worth nothing, and neither would a check a caller can skip.

The contaminations are staged by writing the manifest's ``split`` column directly, never
through `dataset.splits`: what is on trial is the reader's refusal, whoever wrote the split.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from dataset import config as dataset_config
from pretrain import data

CORPUS_500K = Path("data/corpus_500k")

# The fixture corpus (conftest.SPLIT_COGS), tile by tile: two tiles per COG, in COG order.
#   0,1 train  (+000_+040)   2,3 val   (+000_+040)
#   4,5 test   (+010_+050)   6,7 test  (+010_+050)
#   8,9 train  (+030_+000)
HONEST = ["train", "train", "val", "val", "test", "test", "test", "test", "train", "train"]
TEST_CELL = "+010_+050"
VAL_COG = "Copernicus_DSM_COG_10_N45_00_E007_00_DEM"


def _blocks_file(path: Path, *, cells=(TEST_CELL,), cogs=(VAL_COG,)) -> Path:
    """The #77 blocks artifact, in the shape `dataset.splits.write_blocks` emits."""
    path.write_text(
        json.dumps({"test": {"cells": list(cells)}, "val": {"cogs": list(cogs)}}, indent=2)
    )
    return path


# --- The honest split ---------------------------------------------------------


def test_the_train_split_holds_only_train_tiles(split_corpus):
    ds = data.SplitDataset(split_corpus, "train", blocks_path=None, assignment_path=None)
    labels = pq.read_table(split_corpus / "manifest.parquet")["split"].to_pylist()
    assert len(ds) == HONEST.count("train") == 4
    assert {labels[i] for i in ds.indices} == {"train"}


def test_the_train_split_reaches_no_test_geocell_and_no_val_cog(split_corpus):
    cols = pq.read_table(split_corpus / "manifest.parquet", columns=["geocell_id", "source_cog"])
    cells, cogs = cols["geocell_id"].to_pylist(), cols["source_cog"].to_pylist()
    ds = data.SplitDataset(split_corpus, "train", blocks_path=None, assignment_path=None)
    assert not {cells[i] for i in ds.indices} & {TEST_CELL}
    assert not {cogs[i] for i in ds.indices} & {VAL_COG}


def test_it_still_yields_the_adr_0003_sample(split_corpus):
    # a Subset over `dataset.loader`, consumed unchanged: the sample schema is ADR-0003's.
    sample = data.SplitDataset(split_corpus, "train", blocks_path=None, assignment_path=None)[0]
    assert set(sample) == {"image", "land_mask", "mean", "std", "tile_id", "stratum_id"}
    assert sample["image"].shape == (1, 224, 224)


def test_the_val_and_test_splits_build_too(split_corpus):
    val = data.SplitDataset(split_corpus, "val", blocks_path=None, assignment_path=None)
    test = data.SplitDataset(split_corpus, "test", blocks_path=None, assignment_path=None)
    assert (len(val), len(test)) == (2, 4)
    assert val.split == "val" and test.split == "test"


def test_it_batches_through_a_dataloader(split_corpus):
    from dataset.loader import build_dataloader

    ds = data.SplitDataset(split_corpus, "train", blocks_path=None, assignment_path=None)
    batch = next(iter(build_dataloader(ds, batch_size=2, num_workers=0)))
    assert batch["image"].shape == (2, 1, 224, 224)


# --- Contamination: each way a split can leak, refused at construction ---------


def test_a_train_tile_inside_a_test_geocell_is_refused(split_corpus, set_split):
    # the failure the whole check exists for: one tile of a held-out evaluation region
    # relabelled train. Invisible in every loss curve; fatal to Phase 2 after the fact.
    set_split(split_corpus, [*HONEST[:4], "train", *HONEST[5:]])
    with pytest.raises(ValueError, match=re.escape(TEST_CELL)):
        data.SplitDataset(split_corpus, "train", blocks_path=None, assignment_path=None)


def test_a_train_tile_inside_a_val_cog_is_refused(split_corpus, set_split):
    set_split(split_corpus, [*HONEST[:3], "train", *HONEST[4:]])
    with pytest.raises(ValueError, match=VAL_COG):
        data.SplitDataset(split_corpus, "train", blocks_path=None, assignment_path=None)


def test_a_val_tile_inside_a_test_geocell_is_refused(split_corpus, set_split):
    # val is monitored, not trained on, but plan §9 draws it from non-test cells: a val tile
    # in a test region means the region is no longer untouched.
    set_split(split_corpus, [*HONEST[:4], "val", *HONEST[5:]])
    with pytest.raises(ValueError, match=re.escape(TEST_CELL)):
        data.SplitDataset(split_corpus, "val", blocks_path=None, assignment_path=None)


def test_a_cell_the_artifact_holds_out_is_refused_even_when_no_tile_says_test(
    split_corpus, set_split, tmp_path
):
    # the contamination the manifest alone cannot see: every tile of a test geocell relabelled
    # train, so nothing in the column looks held out. The committed block list is what catches
    # it, which is why it is read rather than inferred.
    set_split(split_corpus, ["train"] * 10)
    with pytest.raises(ValueError, match=re.escape(TEST_CELL)):
        data.SplitDataset(
            split_corpus, "train",
            blocks_path=_blocks_file(tmp_path / "b.json"), assignment_path=None,
        )


def test_it_fails_loudly_rather_than_warning(split_corpus, set_split, recwarn):
    set_split(split_corpus, [*HONEST[:4], "train", *HONEST[5:]])
    with pytest.raises(ValueError):
        data.SplitDataset(split_corpus, "train", blocks_path=None, assignment_path=None)
    assert not recwarn.list  # a warning in a training log is worth nothing


def test_the_error_names_the_offending_tiles_and_counts_them(split_corpus, set_split):
    set_split(split_corpus, [*HONEST[:4], "train", "train", *HONEST[6:]])
    with pytest.raises(ValueError) as exc:
        data.SplitDataset(split_corpus, "train", blocks_path=None, assignment_path=None)
    assert "2 " in str(exc.value)  # the count, so a reader knows the scale of the leak
    assert "_0_0" in str(exc.value)  # and at least one tile_id to chase


def test_a_manifest_column_edited_away_from_the_committed_assignment_is_refused(
    split_corpus, set_split, write_assignment, tmp_path
):
    # the blocks defend the regions; this defends the claim. Every run records the assignment's
    # md5 as the split it trained under — a column edited out from under that file would make
    # the recorded fingerprint a lie no downstream reader could catch.
    assignment = write_assignment(tmp_path / "split.csv", split_corpus)
    set_split(split_corpus, [*HONEST[:2], "train", *HONEST[3:]])  # tile 2 was val when committed
    # it is a containment leak too, and that it reports the *disagreement* is the point: a
    # column the committed file does not vouch for is not evidence about anything downstream.
    with pytest.raises(ValueError, match="disagrees"):
        data.SplitDataset(split_corpus, "train", blocks_path=None, assignment_path=assignment)


def test_a_matching_assignment_is_accepted(split_corpus, write_assignment, tmp_path):
    assignment = write_assignment(tmp_path / "split.csv", split_corpus)
    ds = data.SplitDataset(split_corpus, "train", blocks_path=None, assignment_path=assignment)
    assert len(ds) == 4


def test_the_check_cannot_be_routed_around_by_constructing_the_dataset_directly(split_corpus):
    # plan §15.2 asks for the assertion *in the Dataset*, and an assertion a caller can skip is
    # not one: SplitDataset builds itself from the corpus, so there is no path to an instance
    # that has not been checked.
    from torch.utils.data import Subset

    assert issubclass(data.SplitDataset, Subset)
    with pytest.raises(TypeError):  # no (dataset, indices) back door
        data.SplitDataset(object(), [0, 1], split="train")


# --- Splits that are not there ------------------------------------------------


def test_an_unfilled_split_column_is_refused_with_a_pointer_to_77(unsplit_corpus):
    # not an empty dataset that trains on nothing and reports success
    with pytest.raises(ValueError, match="#77"):
        data.SplitDataset(unsplit_corpus, "train", blocks_path=None, assignment_path=None)


def test_a_split_label_no_tile_carries_is_refused(split_corpus):
    with pytest.raises(ValueError, match="holdout"):
        data.SplitDataset(split_corpus, "holdout", blocks_path=None, assignment_path=None)


def test_a_blocks_artifact_from_another_corpus_is_refused(split_corpus, tmp_path):
    # pointing the 500k block list at another tier would "hold out" cells this corpus does not
    # hold, and the check would pass while protecting nothing
    blocks = _blocks_file(tmp_path / "b.json", cells=("+170_-080",), cogs=())
    with pytest.raises(ValueError, match=re.escape("+170_-080")):
        data.SplitDataset(split_corpus, "train", blocks_path=blocks, assignment_path=None)


def test_a_missing_blocks_artifact_is_not_an_error(split_corpus, tmp_path):
    # before #77 commits its block lists, the manifest's own column is the whole check
    ds = data.SplitDataset(
        split_corpus, "train", blocks_path=tmp_path / "absent.json", assignment_path=None
    )
    assert len(ds) == 4


# --- The package boundary -----------------------------------------------------


def test_the_etl_core_still_imports_with_torch_made_unimportable():
    # #89 makes torch a real dependency of `pretrain` and of nothing else. ADR-0003's lazy
    # import is what keeps `dataset.loader` on the right side of that line, and this is the
    # test that notices when someone hoists it to the top of a module. Torch is *blocked*
    # rather than merely counted, so the assertion is the acceptance criterion itself —
    # "`dataset` still imports without torch present" — and not a proxy for it.
    blocked = (
        "import sys, importlib, pkgutil\n"
        "sys.modules['torch'] = None\n"  # any `import torch` below now raises
        "import dataset\n"
        "for m in pkgutil.iter_modules(dataset.__path__):\n"
        "    importlib.import_module(f'dataset.{m.name}')\n"
        "print('ok')\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", blocked], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "ok"


# --- Against the shipped corpus and the committed split -----------------------


@pytest.mark.skipif(
    not (CORPUS_500K.exists() and Path(dataset_config.SPLIT_BLOCKS_PATH).exists()),
    reason="the shipped corpus or the committed split (#77) is not present",
)
def test_the_real_train_split_reaches_no_held_out_block():
    blocks = data.read_blocks(dataset_config.SPLIT_BLOCKS_PATH)
    assert len(blocks.test_geocells) == 13 and len(blocks.val_cogs) == 518  # #77's draw
    ds = data.SplitDataset(CORPUS_500K, "train")
    assert len(ds) == 475_030
    cols = pq.read_table(CORPUS_500K / "manifest.parquet", columns=["geocell_id", "source_cog"])
    cells, cogs = cols["geocell_id"].to_numpy(), cols["source_cog"].to_numpy()
    assert not set(cells[ds.indices]) & blocks.test_geocells
    assert not set(cogs[ds.indices]) & blocks.val_cogs
