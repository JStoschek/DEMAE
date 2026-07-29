"""The data-fingerprint block: what a run records about the data it read (#89).

Nothing here imports :mod:`dataset.splits`. That is the point of the ticket's "it must work
before #77 lands": the fingerprint reads *committed artifacts*, so an assignment CSV written
by hand in a test is indistinguishable from one the split driver produced — and a corpus with
no assignment at all records the absence rather than failing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from dataset import config as dataset_config
from dataset.materialize import file_md5
from pretrain import fingerprint as fp
from pretrain.config import RunConfig

CORPUS_500K = Path("data/corpus_500k")


# --- Resolution ---------------------------------------------------------------


def test_it_resolves_the_corpus_fingerprints_from_the_shipped_meta(split_corpus):
    meta = json.loads((split_corpus / "corpus_meta.json").read_text())
    f = fp.collect_fingerprint(split_corpus, split_assignment=None, split_blocks=None)
    assert f.dataset_version == meta["dataset_version"]
    assert f.selection_list_md5 == meta["build_params"]["selection_list_md5"]
    assert f.n_tiles == meta["n_tiles"] == 10


def test_the_git_sha_is_recorded_and_never_raises(split_corpus):
    f = fp.collect_fingerprint(split_corpus, split_assignment=None, split_blocks=None)
    assert isinstance(f.git_sha, str) and f.git_sha  # "unknown" outside a checkout, never None


def test_a_corpus_without_a_meta_is_refused(tmp_path):
    # the fingerprint is the run's claim about its data; guessing it is worse than stopping
    with pytest.raises(FileNotFoundError):
        fp.collect_fingerprint(tmp_path, split_assignment=None, split_blocks=None)


# --- The split, present and absent --------------------------------------------


def test_a_committed_split_resolves_to_the_artifact_md5(split_corpus, tmp_path, write_assignment):
    assignment = write_assignment(tmp_path / "artifacts" / "split.csv", split_corpus)
    f = fp.collect_fingerprint(split_corpus, split_assignment=assignment, split_blocks=None)
    assert f.split_md5 == file_md5(assignment)
    assert f.split_state == fp.COMMITTED
    assert f.split_note is None


def test_the_blocks_artifact_is_fingerprinted_beside_it(split_corpus, tmp_path, write_assignment):
    assignment = write_assignment(tmp_path / "split.csv", split_corpus)
    blocks = tmp_path / "blocks.json"
    blocks.write_text(json.dumps({"test": {"cells": ["+010_+050"]}}))
    f = fp.collect_fingerprint(split_corpus, split_assignment=assignment, split_blocks=blocks)
    assert f.split_blocks_md5 == file_md5(blocks)


def test_a_missing_assignment_records_as_absent_not_as_an_error(unsplit_corpus):
    # this is the ticket's "works before #77 lands" clause, and the reason it is a *state*
    # rather than a bare null: an absent entry and a forgotten one otherwise look identical.
    f = fp.collect_fingerprint(
        unsplit_corpus, split_assignment=unsplit_corpus / "nope.csv", split_blocks=None
    )
    assert f.split_md5 is None
    assert f.split_state == fp.ABSENT
    assert "nope.csv" in f.split_note


def test_an_unfilled_manifest_column_records_as_absent_even_with_an_artifact(
    unsplit_corpus, tmp_path
):
    # the artifact exists but nothing filled the column: the corpus this run would read is
    # still unsplit, so the honest record is "absent", with the md5 kept for the reader.
    assignment = tmp_path / "split.csv"
    assignment.write_text("tile_id,split\n")
    f = fp.collect_fingerprint(unsplit_corpus, split_assignment=assignment, split_blocks=None)
    assert f.split_state == fp.ABSENT
    assert "not filled" in f.split_note
    assert f.split_md5 == file_md5(assignment)  # what exists is still recorded


def test_a_partly_filled_manifest_column_is_absent_too(split_corpus, set_split):
    # a partial fill is the dangerous shape: the nulls read as "not yet split" and would
    # silently rejoin held-out tiles to training, so it must not read as a split corpus.
    labels = pq.read_table(split_corpus / "manifest.parquet")["split"].to_pylist()
    set_split(split_corpus, [None, *labels[1:]])
    f = fp.collect_fingerprint(split_corpus, split_assignment=None, split_blocks=None)
    assert f.split_state == fp.ABSENT
    assert "1 of 10" in f.split_note


# --- What the run records -----------------------------------------------------


def test_the_block_serializes_to_plain_json_with_the_four_tagged_facts(
    split_corpus, tmp_path, write_assignment
):
    # plan §15.7: every W&B run is tagged with git SHA, config hash, selection-list md5 and
    # split md5. Three of the four live here; the config hash is the config's own.
    assignment = write_assignment(tmp_path / "split.csv", split_corpus)
    f = fp.collect_fingerprint(split_corpus, split_assignment=assignment, split_blocks=None)
    d = f.to_dict()
    assert json.loads(json.dumps(d)) == d
    assert {"dataset_version", "selection_list_md5", "split_md5", "git_sha"} <= set(d)


def test_it_can_be_collected_straight_from_a_run_config(split_corpus, tmp_path, write_assignment):
    assignment = write_assignment(tmp_path / "split.csv", split_corpus)
    cfg = RunConfig()
    cfg = cfg.replace(
        data=cfg.data.replace(
            corpus_dir=str(split_corpus),
            split_assignment_path=str(assignment),
            split_blocks_path=str(tmp_path / "absent.json"),
        )
    )
    f = fp.fingerprint_for(cfg)
    assert f.split_md5 == file_md5(assignment)
    assert f.corpus_dir == str(split_corpus)
    assert f.split_blocks_md5 is None


# --- Against the shipped corpus -----------------------------------------------


@pytest.mark.skipif(not CORPUS_500K.exists(), reason="the shipped 500k corpus is not present")
def test_it_resolves_against_the_shipped_500k_corpus():
    f = fp.collect_fingerprint(CORPUS_500K)
    assert f.dataset_version == "bcbd6484d8116199c5c1160e21ea0284"  # the tier the plan names
    assert f.selection_list_md5 == file_md5(dataset_config.SCALE_SELECTION_PATH)
    assert f.n_tiles == 499_934
    assert f.git_sha != "unknown"


@pytest.mark.skipif(
    not Path(dataset_config.SPLIT_ASSIGNMENT_PATH).exists(), reason="the split (#77) is not filled"
)
def test_the_committed_split_resolves_against_the_shipped_corpus():
    f = fp.collect_fingerprint(CORPUS_500K)
    assert f.split_state == fp.COMMITTED
    assert f.split_md5 == file_md5(dataset_config.SPLIT_ASSIGNMENT_PATH)
    assert f.split_blocks_md5 == file_md5(dataset_config.SPLIT_BLOCKS_PATH)
