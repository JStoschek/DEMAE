"""ADR-0012's extractor contract — the seam mandated rather than chosen (#156).

`docs/task1-build-spec.md` §Seam 2 names this file before it existed, and gives the reason it
cannot attach at `benchmark.launch.main(argv)` like nearly everything else in this build: **a
second consumer (Task 2) re-reads this contract**, and ADR-0012 Decision 4 explicitly requires
Task 1 to test the intermediates path *even though no Task 1 arm reads it*. No launch-level
behaviour exercises that path, so no launch-level test can cover it — and an unexercised path
shipped with a contract's authority is this repo's named failure with a document attached.

What is pinned here is Decision 4's list — shapes, ordering, count, and the layer-12-versus-
``output`` relation — plus the five refusals the build spec enumerates: a wrong checkpoint md5,
a declared layer count that does not match the model loaded, a spec that has drifted from the
code, recomputed statistics that no longer reproduce the manifest, and a pre-normalized tensor,
which is refused by *type* rather than caught by an assertion.

## Why a synthesized ViT-S carries most of it

The three checkpoints this contract covers are 340 MB each and two of them are not mounted on
any box. The registry's ``demae-vits`` entry (§2.3's appendix backbone, arm ``A4-S``) is a real
registered encoder with no released weights, so a randomly-initialized `vit_small_patch16_224`
saved under its path is a *legitimate* subject: every property Decision 4 names — twelve
intermediates in block order, the canonical ``(N, 196, D)`` slice, the final norm standing
between block twelve and ``output`` — is a property of the composition and not of the weights.
The refusals that never reach a forward pass use hand-built state dicts of a few tensors, which
is why this module is fast. The R5 release encoder, where a box mounts one, then checks the same
claims against the real thing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from benchmark import arms, artifacts
from benchmark.config import BenchmarkConfig
from benchmark.extract import (
    ADR,
    LAYERS,
    SPEC_FILENAME,
    STD_FLOOR,
    EncoderSpec,
    Extractor,
    ExtractorSpec,
    build_spec,
    load_spec,
    normalize,
    spec_path,
    tile_statistics,
    write_spec,
)
from dataset import config as dataset_config
from dataset import loader as dataset_loader
from dataset.normalize import land_mask, land_stats

REPO = Path(__file__).resolve().parents[1]

#: The §2.3 appendix backbone: a registered encoder with no released weights, so a synthesized
#: checkpoint under its path is the contract's own subject rather than a stand-in for one.
ENCODER = "demae-vits"
ARM = "A4-S"
WIDTH = arms.ENCODERS[ENCODER].width
GRID = dataset_config.TILE_PX // dataset_config.PATCH_PX

#: The R5 release encoder, where a box mounts it. Its md5 is *published* (#156 settled that both
#: the `-v2` and non-`-v2` bundles ship a byte-identical `encoder.pt`), so unlike the ViT-S it is
#: pinned by the registry rather than observed at freeze time.
R5 = REPO / ".release-work" / "release-R5-v2" / "encoder.pt"
requires_r5 = pytest.mark.skipif(not R5.is_file(), reason="the R5 release encoder is not mounted")

MANIFEST = REPO / "data" / "corpus_500k" / "manifest.parquet"
SHARDS = REPO / "data" / "corpus_500k" / "shards"
requires_corpus = pytest.mark.skipif(
    not (MANIFEST.is_file() and SHARDS.is_dir()), reason="data/corpus_500k not present"
)


# --- Fixtures -----------------------------------------------------------------


def _sincos(order: str, *, width: int = WIDTH):
    from pretrain.model import sincos2d_pos_embed

    return sincos2d_pos_embed(width, GRID, axis_order=order)


@pytest.fixture(scope="module")
def vits_checkpoint(tmp_path_factory):
    """A real `vit_small_patch16_224` at random weights, saved the way §14 exports one.

    Random weights and a *correct* composition: what Decision 4 asks about is the shape of the
    pass, and `pretrain.release` already guarantees the released state dict loads strictly into
    this timm model. Its `pos_embed` is MAE's own column-first sin-cos, which is what the
    registry declares for this encoder.
    """
    import timm
    import torch

    model = timm.create_model(
        "vit_small_patch16_224", pretrained=False, in_chans=1, num_classes=0, global_pool=""
    )
    state = model.state_dict()
    state["pos_embed"] = _sincos("col_row")
    path = tmp_path_factory.mktemp("vits") / "encoder.pt"
    torch.save(state, path)
    return path


def _config(tmp_path, checkpoint) -> BenchmarkConfig:
    """A config whose ViT-S encoder is ``checkpoint`` and whose artifacts live in ``tmp_path``.

    The two release encoders are **unmounted** here. Their md5s are published, so `freeze` pins
    them from the registry either way and only *compares* against a file that is present — which
    means unmounting them changes nothing about the spec these tests read, and saves hashing 680
    MB on every `build_spec` call. The tests that are actually about those files mount them.
    """
    base = BenchmarkConfig()
    return base.replace(
        arm=ARM,
        paths=base.paths.replace(
            artifacts_dir=str(tmp_path),
            demae_r5_checkpoint=None,
            demae_r4_checkpoint=None,
            demae_vits_checkpoint=None if checkpoint is None else str(checkpoint),
        ),
        # ADR-0012 §6 keeps fp32 as the reference and lets a run declare its compute dtype. The
        # relations Decision 4 pins are exact ones, so they are checked at the reference.
        extract=base.extract.replace(compute_dtype="fp32"),
    )


def _stub_checkpoint(path, *, depth=LAYERS, pos_embed="col_row", width=WIDTH):
    """A few named tensors, enough for the refusals that never reach a forward pass.

    The load order is what makes this legitimate: md5, then block count off the checkpoint's own
    keys, then `pos_embed`'s presence, shape and axis order — every one of them before a model is
    built. A refusal that needed 88 MB of weights to fire would be a refusal firing too late.
    """
    import torch

    state = {f"blocks.{i}.attn.qkv.bias": torch.zeros(width) for i in range(depth)}
    if pos_embed is not None:
        state["pos_embed"] = (
            pos_embed if torch.is_tensor(pos_embed) else _sincos(pos_embed, width=width)
        )
    torch.save(state, path)
    return path


def _extractor(tmp_path, checkpoint):
    cfg = _config(tmp_path, checkpoint)
    return Extractor(build_spec(cfg), cfg)


@pytest.fixture
def tile():
    """One synthetic 224x224 int16 tile in metres, with a strip of ocean and a void.

    Deliberately built rather than read: the contract is about what the interface accepts and
    what it computes from the mask, and both are true of a tile nobody had to mount 47 GB for.
    """
    rng = np.random.default_rng(0)
    raw = (rng.normal(1200.0, 180.0, size=(dataset_config.TILE_PX,) * 2)).astype(np.int16)
    raw[:16] = 0  # ocean: exact-zero fill, which `land_mask` excludes
    valid = land_mask(raw)
    valid[100:140, 100:140] = False  # the void: valid = land ∧ ¬void
    return raw, valid


# --- The committed file, and its md5 ------------------------------------------


def test_freeze_writes_the_contract_and_load_reads_it_back(tmp_path, vits_checkpoint):
    cfg = _config(tmp_path, vits_checkpoint)
    written = write_spec(cfg)
    assert written == spec_path(cfg) == tmp_path / SPEC_FILENAME
    assert load_spec(written) == build_spec(cfg)


def test_the_contract_is_a_file_task_two_re_reads_rather_than_a_description(tmp_path, vits_checkpoint):
    """ADR-0012 §7's whole argument, as a property of the bytes.

    A consumer that has never imported this repo's build code can read the file and find every
    value it would otherwise have to re-derive: the layout, the diet, the layer count, which
    encoder each arm reads, and — the one nobody would guess — that `output` is not
    `intermediates[11]`.
    """
    body = json.loads(write_spec(_config(tmp_path, vits_checkpoint)).read_text())
    assert body["adr"] == ADR
    assert body["layers"] == LAYERS and body["tokens"] == 196
    assert body["reference_dtype"] == "fp32"
    assert body["valid_mask"] == "land_and_not_void"
    assert "never read from the manifest" in body["statistics"]
    assert body["arms"]["A4-R5"] == "demae-r5" and body["arms"]["A1"] is None
    entry = body["encoders"]["demae-r5"]
    assert entry["output_is_normed"] is True
    assert "**not** `intermediates[11]`" in entry["note"]


def test_freezing_twice_writes_the_same_bytes(tmp_path, vits_checkpoint):
    cfg = _config(tmp_path, vits_checkpoint)
    first = write_spec(cfg).read_bytes()
    assert write_spec(cfg).read_bytes() == first


def test_a_spec_whose_content_hash_has_moved_is_not_overwritten(tmp_path, vits_checkpoint):
    """§3.4's rule: re-running `freeze` after a code change must not silently re-draw an input.

    The spec's md5 is in every run record, so refusing here is what keeps a table produced under
    the old contract attributable to it. Deleting the file is the deliberate act that says the
    contract really did change.
    """
    cfg = _config(tmp_path, vits_checkpoint)
    write_spec(cfg)
    (tmp_path / SPEC_FILENAME).write_text('{"layers": 12}\n')
    with pytest.raises(FileExistsError, match="content hash has moved"):
        write_spec(cfg)


def test_the_specs_md5_is_in_the_run_record(tmp_path, vits_checkpoint):
    cfg = _config(tmp_path, vits_checkpoint)
    written = write_spec(cfg)
    record = artifacts.resolve(cfg).to_dict()
    assert record["artifacts"]["extractor_spec"]["md5"] == hashlib.md5(
        written.read_bytes()
    ).hexdigest()
    assert load_spec(written).md5() == record["artifacts"]["extractor_spec"]["md5"]


def test_the_committed_contract_in_this_repository_loads_and_re_checks():
    """The one Task 2 will actually open, checked the way Task 2 will open it.

    Not asserted byte-for-byte against what `freeze` writes *here*, and deliberately: the two
    unreleased encoders are pinned by whatever a box mounts, so the committed bytes are a fact
    about the box that froze them. What has to hold on every box is that the file still describes
    this code — which is exactly what `load_spec` refuses on.
    """
    committed = load_spec(REPO / "artifacts" / "task1" / SPEC_FILENAME)
    assert committed.encoders["demae-r5"].checkpoint_md5 == arms.DEMAE_R5_MD5
    assert committed.encoders["demae-r4"].checkpoint_md5 == arms.DEMAE_R4_MD5
    # Neither unreleased encoder is mounted anywhere yet, so neither is pinned — and an unpinned
    # entry refuses at extraction rather than running against whatever sits at the path.
    assert committed.encoders["demae-vits"].checkpoint_md5 is None
    assert committed.encoders["imagenet-mae"].checkpoint_md5 is None


def test_the_specs_md5_is_not_in_phase_ones_frozen_four_field_fingerprint():
    """ADR-0012 §7's last paragraph, as a check rather than a sentence.

    A results table should say which contract produced it *without* moving a hash that four
    committed artifacts quote. The fingerprint is a frozen block; this asserts the extractor
    never became a fifth field in it.
    """
    from pretrain.fingerprint import DataFingerprint

    names = {f.name for f in DataFingerprint.__dataclass_fields__.values()}
    assert not {name for name in names if "extract" in name or "spec" in name}
    assert {"dataset_version", "selection_list_md5", "split_md5", "git_sha"} <= names


# --- The self-check: it refuses rather than warns ------------------------------


def _loaded(tmp_path, vits_checkpoint, **moved) -> ExtractorSpec:
    """Write a spec with ``moved`` applied to its top-level fields, and load it back."""
    cfg = _config(tmp_path, vits_checkpoint)
    body = build_spec(cfg).to_dict()
    body.update(moved)
    (tmp_path / SPEC_FILENAME).write_text(json.dumps(body))
    return load_spec(tmp_path / SPEC_FILENAME)


@pytest.mark.parametrize(
    "moved, message",
    [
        ({"layers": 11}, "declares 11 layers"),
        ({"tokens": 256}, "declares 256 tokens"),
        ({"reference_dtype": "bf16"}, "fp32 is the reference tensor"),
        ({"valid_mask": "land"}, "valid = land"),
        ({"statistics": "read from the manifest"}, "recomputes statistics from the passed mask"),
        ({"img_px": 256}, "declares a 256/16 tile"),
        ({"width_grid": 16}, "is not 196 tokens"),
    ],
)
def test_a_contract_that_no_longer_states_the_adrs_decision_is_refused(
    tmp_path, vits_checkpoint, moved, message
):
    with pytest.raises(ValueError, match=message):
        _loaded(tmp_path, vits_checkpoint, **moved)


def test_an_arm_map_that_has_drifted_from_the_registry_is_refused(tmp_path, vits_checkpoint):
    # The clause Task 2 leans on hardest: which encoder each arm reads. A spec that said A4-R5
    # reads R4 would have both tasks' C1 claims about different weights and nothing would raise.
    cfg = _config(tmp_path, vits_checkpoint)
    body = build_spec(cfg).to_dict()
    body["arms"]["A4-R5"] = "demae-r4"
    (tmp_path / SPEC_FILENAME).write_text(json.dumps(body))
    with pytest.raises(ValueError, match="arm A4-R5 reading 'demae-r4'"):
        load_spec(tmp_path / SPEC_FILENAME)


def test_an_arm_the_registry_does_not_have_is_refused(tmp_path, vits_checkpoint):
    cfg = _config(tmp_path, vits_checkpoint)
    body = build_spec(cfg).to_dict()
    body["arms"]["A8"] = "demae-r5"
    (tmp_path / SPEC_FILENAME).write_text(json.dumps(body))
    with pytest.raises(ValueError, match="arm map is not the registry's"):
        load_spec(tmp_path / SPEC_FILENAME)


@pytest.mark.parametrize(
    "moved, message",
    [
        ({"pos_encoding": "sincos2d_rowfirst"}, "drifted from the registry"),
        ({"width": 384}, "drifted from the registry"),
        ({"output_is_normed": False}, "drifted from the registry"),
        ({"normalization": "terramind_v1"}, "a 'terramind_v1' diet"),
        ({"checkpoint_md5_source": None}, "an md5 nobody can source"),
        ({"checkpoint_md5_source": "guessed"}, "neither published nor observed"),
    ],
)
def test_an_encoder_entry_that_has_drifted_from_the_registry_is_refused(
    tmp_path, vits_checkpoint, moved, message
):
    cfg = _config(tmp_path, vits_checkpoint)
    body = build_spec(cfg).to_dict()
    body["encoders"]["demae-r5"].update(moved)
    (tmp_path / SPEC_FILENAME).write_text(json.dumps(body))
    with pytest.raises(ValueError, match=message):
        load_spec(tmp_path / SPEC_FILENAME)


def test_an_unknown_or_missing_spec_field_is_refused_rather_than_ignored(tmp_path, vits_checkpoint):
    # A silently dropped key is a contract clause Task 2 thinks it re-read and did not; a missing
    # one is a clause it never read at all.
    cfg = _config(tmp_path, vits_checkpoint)
    body = build_spec(cfg).to_dict()
    (tmp_path / SPEC_FILENAME).write_text(json.dumps({**body, "dtype": "fp16"}))
    with pytest.raises(ValueError, match=r"unknown spec field\(s\) \['dtype'\]"):
        load_spec(tmp_path / SPEC_FILENAME)

    del body["valid_mask"]
    (tmp_path / SPEC_FILENAME).write_text(json.dumps(body))
    with pytest.raises(ValueError, match=r"missing field\(s\) \['valid_mask'\]"):
        load_spec(tmp_path / SPEC_FILENAME)


def test_an_encoder_entry_missing_the_note_is_refused(tmp_path, vits_checkpoint):
    """The field-level note is a clause, not a comment — ADR-0012 §3 asks for it by name."""
    cfg = _config(tmp_path, vits_checkpoint)
    body = build_spec(cfg).to_dict()
    del body["encoders"]["demae-r5"]["note"]
    (tmp_path / SPEC_FILENAME).write_text(json.dumps(body))
    with pytest.raises(ValueError, match=r"encoders.demae-r5 is missing field\(s\) \['note'\]"):
        load_spec(tmp_path / SPEC_FILENAME)


def test_a_spec_that_is_not_there_names_the_command_that_writes_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="benchmark.launch freeze"):
        load_spec(tmp_path / SPEC_FILENAME)


def test_the_contracts_layer_count_is_the_configs():
    # Two homes for twelve would be two places for it to move. `load_spec` cannot read a run
    # config — Task 2 has none — so the constant is here and pinned against the knob.
    assert LAYERS == BenchmarkConfig().extract.layers == 12


def test_the_std_floor_is_the_loaders_own():
    assert STD_FLOOR == dataset_loader._STD_FLOOR


# --- TerraMind is named as owed, not absent ------------------------------------


def test_terramind_is_named_as_owed_rather_than_missing_from_the_contract(tmp_path, vits_checkpoint):
    """#12's half, in the committed file. An absent contract that read as a resolved one is
    exactly what this family of artifacts exists to prevent."""
    spec = build_spec(_config(tmp_path, vits_checkpoint))
    assert "terramind" not in spec.encoders and "terramind-tokenizer" not in spec.encoders
    owed = spec.pending["terramind"]
    assert owed["ticket"] == 12
    assert set(owed["encoders"]) == {"terramind", "terramind-tokenizer"}
    assert any("670.665" in line for line in owed["owes"])
    for arm in ("A5", "A9"):
        with pytest.raises(NotImplementedError, match="ticket #12"):
            spec.encoder_for(arm)


def test_an_arm_with_no_encoder_has_no_frozen_feature_contract(tmp_path, vits_checkpoint):
    spec = build_spec(_config(tmp_path, vits_checkpoint))
    for arm in ("A1", "A7"):
        with pytest.raises(ValueError, match="runs no encoder"):
            spec.encoder_for(arm)


# --- ADR-0012 Decision 1: a pre-normalized tensor is not expressible ------------


def test_the_pretraining_loaders_own_output_is_not_something_the_interface_accepts(
    tmp_path, vits_checkpoint, tile
):
    """p118 §7's trap, closed by type rather than guarded by a test that must be remembered.

    The loader hands out per-tile *standardized* float32. An encoder accepts it silently and
    returns a well-shaped, plausible tensor from a distribution it has never seen — nothing
    raises, no loss moves, and the arm scores. So the door is the dtype.
    """
    raw, valid = tile
    extractor = _extractor(tmp_path, vits_checkpoint)
    mean, std = land_stats(raw.astype(np.float32), land_mask(raw))
    standardized = dataset_loader.PerTileNormalizer()(raw.astype(np.float32), mean=mean, std=std)
    assert standardized.dtype == np.float32  # this is the loader's own `image`, not a lookalike

    with pytest.raises(TypeError, match="raw int16 metres"):
        extractor.extract(standardized, valid, arm=ARM)


def test_a_torch_tensor_is_refused_and_the_message_names_the_loader(tmp_path, vits_checkpoint, tile):
    import torch

    raw, valid = tile
    extractor = _extractor(tmp_path, vits_checkpoint)
    with pytest.raises(TypeError, match="dataset.loader"):
        extractor.extract(torch.from_numpy(raw.astype(np.float32)), valid, arm=ARM)


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "uint8"])
def test_any_dtype_but_int16_metres_is_refused(tmp_path, vits_checkpoint, tile, dtype):
    raw, valid = tile
    extractor = _extractor(tmp_path, vits_checkpoint)
    with pytest.raises(TypeError, match="raw int16 metres"):
        extractor.extract(raw.astype(dtype), valid, arm=ARM)


def test_a_mask_that_is_not_boolean_is_refused(tmp_path, vits_checkpoint, tile):
    raw, valid = tile
    extractor = _extractor(tmp_path, vits_checkpoint)
    with pytest.raises(TypeError, match="valid = land"):
        extractor.extract(raw, valid.astype(np.uint8), arm=ARM)


def test_a_mask_that_does_not_match_the_tiles_is_refused(tmp_path, vits_checkpoint, tile):
    raw, valid = tile
    extractor = _extractor(tmp_path, vits_checkpoint)
    with pytest.raises(ValueError, match="does not match"):
        extractor.extract(raw, valid[:100, :100], arm=ARM)


# --- ADR-0012 Decision 2: the statistics, recomputed ---------------------------


def test_the_statistics_come_from_the_passed_mask_and_nothing_else(tile):
    """One code path: Task 1's void-aware mask and Task 2's land-only mask are the same call.

    The two answers differ — that is the *point*, and §2.1 calls the difference the thing that
    stops the leak — so a branch that read the manifest for one of them would be reading a
    number computed under the other's definition.
    """
    raw, valid = tile
    land_only = land_mask(raw)
    assert tile_statistics(raw, land_only) != tile_statistics(raw, valid)
    # And it really is the mask that decides, not the tile: masking the void back in reproduces
    # the land-only answer exactly.
    assert tile_statistics(raw, land_only) == tile_statistics(raw, land_only.copy())


def test_normalization_standardizes_by_the_recomputed_statistics(tile):
    raw, valid = tile
    mean, std = tile_statistics(raw, valid)
    out = normalize(raw[None], valid[None], normalization="per_tile")
    assert out.shape == (1, dataset_config.TILE_PX, dataset_config.TILE_PX)
    assert out.dtype == np.float32
    expected = (raw.astype(np.float32) - np.float32(mean)) / np.float32(max(std, STD_FLOOR))
    np.testing.assert_array_equal(out[0], expected)
    # The valid pixels are what carry the statistic, so they are what comes out standardized.
    assert out[0][valid].mean() == pytest.approx(0.0, abs=1e-4)
    assert out[0][valid].std() == pytest.approx(1.0, rel=1e-4)


def test_a_tile_with_no_valid_pixels_is_refused_rather_than_divided_by_the_floor(tile):
    raw, _ = tile
    with pytest.raises(ValueError, match="no valid pixels"):
        tile_statistics(raw, np.zeros_like(raw, dtype=bool))


def test_a_normalization_this_ticket_does_not_own_says_so(tile):
    raw, valid = tile
    with pytest.raises(NotImplementedError, match="pending"):
        normalize(raw[None], valid[None], normalization="terramind_v1")


@requires_corpus
def test_recomputed_statistics_reproduce_the_manifest_within_the_measured_bound():
    """ADR-0012 Decision 2's *checked claim*, over the sample p137 §1 measured it on.

    The extractor recomputes per-tile statistics where Phase 1's loader reads the manifest's
    stored columns, and that is only defensible if the two agree. p137 measured the gap over 500
    real tiles from three shards: max **3.637e-4 m** on the mean and max **1.059e-7 relative**
    on the std — float32 accumulation order, not a definitional gap, because recomputing from the
    land mask *is* the ETL's land definition.

    This is the check that makes a future drift in that definition surface in Task 1 rather than
    in Task 2, which is the whole reason ADR-0012 ships it as a test instead of a paragraph.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(MANIFEST, columns=["shard_id", "row_index", "mean", "std"])
    shard_id = table.column("shard_id").to_numpy()
    row_index = table.column("row_index").to_numpy()
    stored_mean = table.column("mean").to_numpy()
    stored_std = table.column("std").to_numpy()

    rng = np.random.default_rng(0)
    shards = sorted(set(shard_id[:20_000].tolist()))[:3]
    chosen = rng.choice(np.flatnonzero(np.isin(shard_id, shards)), size=500, replace=False)

    cache: dict[int, np.ndarray] = {}
    mean_gap, std_rel = [], []
    for i in chosen:
        sid = int(shard_id[i])
        if sid not in cache:
            cache[sid] = np.load(next(SHARDS.glob(f"*{sid:05d}*.npy")), mmap_mode="r")
        raw = np.asarray(cache[sid][int(row_index[i])])
        # Task 2's mask — land only, no void — which is the one the manifest's columns describe.
        mean, std = tile_statistics(raw, land_mask(raw))
        mean_gap.append(abs(mean - stored_mean[i]))
        std_rel.append(abs(std - stored_std[i]) / stored_std[i])

    assert max(mean_gap) <= 3.7e-4, "the recomputed mean has drifted from the manifest's"
    assert max(std_rel) <= 1.1e-7, "the recomputed std has drifted from the manifest's"
    assert np.median(mean_gap) <= 1e-5


# --- ADR-0012 Decision 4: the intermediates path, which no Task 1 arm reads -----


@pytest.fixture(scope="module")
def features(tmp_path_factory, vits_checkpoint):
    """One forward pass at the fp32 reference, shared by the claims Decision 4 enumerates."""
    rng = np.random.default_rng(1)
    raw = rng.normal(900.0, 150.0, size=(2, dataset_config.TILE_PX, dataset_config.TILE_PX))
    raw = raw.astype(np.int16)
    valid = land_mask(raw)
    extractor = _extractor(tmp_path_factory.mktemp("features"), vits_checkpoint)
    return extractor, raw, valid, extractor.extract(raw, valid, arm=ARM)


def test_the_extractor_emits_every_block_in_order(features):
    _, raw, _, out = features
    assert len(out.intermediates) == LAYERS == 12
    assert all(block.shape == (len(raw), 196, WIDTH) for block in out.intermediates)
    # Ordering, not just count: a residual stack's block outputs are strictly non-identical, and
    # each one is what the *next* block consumed. Reversing the tuple would pass a count check.
    assert all(
        not block.equal(nxt) for block, nxt in zip(out.intermediates, out.intermediates[1:])
    )


def test_the_intermediates_are_this_models_blocks_in_order(features):
    """``intermediates[i]`` is the output of block ``i + 1``, checked index by index.

    The oracle is the composition written out by hand here rather than the one the module runs,
    which is what makes it worth asserting: any permutation of the tuple fails it, an off-by-one
    fails it, and slicing the CLS column off the wrong end fails it. Ordering is the property
    Task 2's layer sweep actually depends on — its UperNet selects indices out of this tuple and
    has no way to notice that they mean something else.
    """
    import torch

    from benchmark.extract import normalize as apply_diet

    extractor, raw, valid, out = features
    model = extractor.encoder(ARM)
    x = torch.from_numpy(apply_diet(raw, valid, normalization="per_tile")).unsqueeze(1)
    with torch.no_grad():
        state = model.norm_pre(model.patch_drop(model._pos_embed(model.patch_embed(x))))
        for index, block in enumerate(model.blocks):
            state = block(state)
            assert torch.equal(out.intermediates[index], state[:, 1:]), f"block {index + 1}"


def test_layer_twelve_is_not_the_output_and_the_relation_is_the_declared_one(features):
    """The one nobody would guess by reading tensors, which is why ADR-0012 §3 names it.

    DEMAE runs twelve blocks and *then* applies its final norm, so ``output`` and
    ``intermediates[11]`` are two different tensors — and the spec carries a field-level note
    saying so. Under a naive shared index, "layer 12" would mean a normed tensor here and an
    un-normed one on the TerraMind side.
    """
    import torch

    extractor, _, _, out = features
    model = extractor.encoder(ARM)
    entry = extractor.spec.encoder_for(ARM)
    assert entry.output_is_normed is True

    last = out.intermediates[-1]
    assert not torch.allclose(out.output, last), "the final norm is doing nothing"
    with torch.no_grad():
        normed = model.norm(last)
    assert torch.allclose(out.output, normed, atol=1e-6)


def test_the_canonical_layout_is_patch_tokens_with_cls_as_a_side_field(features):
    _, raw, _, out = features
    assert out.output.shape == (len(raw), 196, WIDTH)  # ADR-0012 §5
    assert out.cls is not None and out.cls.shape == (len(raw), WIDTH)
    # The side field is the *final* CLS and is never one of the 196: a consumer that concatenated
    # it would be feeding the head a token TerraMind structurally does not have.
    assert out.output.shape[1] == 196


def test_the_reference_tensor_is_fp32_whatever_the_compute_dtype(tmp_path, vits_checkpoint, tile):
    """ADR-0012 §6: the compute dtype is a declared field; what comes back is the reference.

    A task moving its compute dtype re-checks it under #134's 0.99 materiality bar rather than
    changing what the contract hands its consumer.
    """
    import torch

    raw, valid = tile
    cfg = _config(tmp_path, vits_checkpoint)
    cfg = cfg.replace(extract=cfg.extract.replace(compute_dtype="bf16"))
    out = Extractor(build_spec(cfg), cfg).extract(raw, valid, arm=ARM)
    assert out.output.dtype == torch.float32
    assert all(block.dtype == torch.float32 for block in out.intermediates)


def test_a_single_tile_and_a_batch_are_the_same_call(tmp_path, vits_checkpoint, tile):
    import torch

    raw, valid = tile
    extractor = _extractor(tmp_path, vits_checkpoint)
    one = extractor.extract(raw, valid, arm=ARM)
    batched = extractor.extract(raw[None], valid[None], arm=ARM)
    assert one.output.shape == batched.output.shape == (1, 196, WIDTH)
    assert torch.allclose(one.output, batched.output)


def test_the_unrolled_pass_is_timms_own_forward_features(tmp_path, vits_checkpoint, tile):
    """The composition, pinned bit-identical to the library call it replaces.

    Unrolled rather than routed through ``get_intermediate_layers`` so that ``output`` is the
    exact tensor `MaskedAutoencoder.encode` computes — patch embed, CLS and position, every
    block, final norm — and not a helper's interpretation of it. That is only worth anything if
    it *is* the same composition, so this asserts equality rather than closeness.
    """
    import torch

    from benchmark.extract import _forward_blocks

    raw, valid = tile
    extractor = _extractor(tmp_path, vits_checkpoint)
    model = extractor.encoder(ARM)
    x = normalize(raw[None], valid[None], normalization="per_tile")
    tensor = torch.from_numpy(x).unsqueeze(1)
    with torch.no_grad():
        assert torch.equal(_forward_blocks(model, tensor)[0], model.forward_features(tensor))


# --- ADR-0012 §7.1: encoder identity, checked against the model actually loaded -


def test_a_wrong_checkpoint_md5_is_refused(tmp_path, vits_checkpoint):
    """An encoder handed the wrong checkpoint returns a plausible tensor and raises nothing.

    So the md5 the spec was frozen against is checked before the file is read at all — which is
    also why a 340 MB swap costs a hash rather than a forward pass to detect.
    """
    cfg = _config(tmp_path, vits_checkpoint)
    spec = build_spec(cfg)
    swapped = tmp_path / "swapped.pt"
    swapped.write_bytes(vits_checkpoint.read_bytes() + b"\n")
    moved = cfg.replace(paths=cfg.paths.replace(demae_vits_checkpoint=str(swapped)))
    with pytest.raises(ValueError, match="declares demae-vits as"):
        Extractor(spec, moved).encoder(ARM)


def test_freeze_refuses_to_pin_a_checkpoint_that_is_not_the_published_one(tmp_path, vits_checkpoint):
    """The other end of the same rule: the moment to catch a wrong file is while it is pinned.

    R5's md5 is *published* (#156), so a box mounting something else at `paths.demae_r5_checkpoint`
    would otherwise freeze a contract naming weights nobody ran and hand it to Task 2.
    """
    cfg = _config(tmp_path, vits_checkpoint)
    cfg = cfg.replace(paths=cfg.paths.replace(demae_r5_checkpoint=str(vits_checkpoint)))
    with pytest.raises(ValueError, match="the demae-r5 encoder this plan benchmarks is"):
        build_spec(cfg)


def test_a_declared_layer_count_that_does_not_match_the_model_loaded_is_refused(tmp_path):
    """Off the checkpoint's own keys, before a model is built.

    A strict load would refuse too, but it would name two dozen missing tensors rather than the
    one fact that is wrong — and ADR-0012 §3's contract is *about* the layer count, so it is the
    thing the refusal should say.
    """
    stub = _stub_checkpoint(tmp_path / "eleven.pt", depth=11)
    with pytest.raises(ValueError, match="has 11 transformer blocks"):
        _extractor(tmp_path, stub).encoder(ARM)


def test_a_pos_embed_under_the_wrong_axis_order_is_refused_and_the_message_says_which(tmp_path):
    """§2.3's R4-versus-R5 precondition, enforced rather than remembered.

    Every name and every shape matches under either order, so a `strict=True` load succeeds and
    the encoder reads every tile as its own transpose. The two differ by 2.0 — the full range of
    a sinusoid — so there is nothing borderline about the check.
    """
    stub = _stub_checkpoint(tmp_path / "transposed.pt", pos_embed="row_col")
    with pytest.raises(ValueError, match="not sin-cos under the declared 'sincos2d'"):
        _extractor(tmp_path, stub).encoder(ARM)


def test_a_pos_embed_that_is_not_sin_cos_at_all_is_refused(tmp_path):
    import torch

    stub = _stub_checkpoint(
        tmp_path / "learned.pt", pos_embed=torch.zeros(1, 1 + 196, WIDTH)
    )
    with pytest.raises(ValueError, match="not sin-cos under the declared"):
        _extractor(tmp_path, stub).encoder(ARM)


def test_a_pos_embed_of_the_wrong_shape_is_refused(tmp_path):
    stub = _stub_checkpoint(tmp_path / "wide.pt", pos_embed="col_row", width=768)
    with pytest.raises(ValueError, match=r"`pos_embed` is \(1, 197, 768\)"):
        _extractor(tmp_path, stub).encoder(ARM)


def test_a_checkpoint_with_no_pos_embed_is_refused(tmp_path):
    stub = _stub_checkpoint(tmp_path / "bare.pt", pos_embed=None)
    with pytest.raises(ValueError, match="no `pos_embed`"):
        _extractor(tmp_path, stub).encoder(ARM)


def test_an_unmounted_encoder_names_the_path_knob(tmp_path, vits_checkpoint):
    cfg = _config(tmp_path, vits_checkpoint)
    spec = build_spec(cfg)
    with pytest.raises(FileNotFoundError, match="paths.demae_vits_checkpoint"):
        Extractor(spec, _config(tmp_path, None)).encoder(ARM)


def test_an_unpinned_md5_refuses_rather_than_running_against_whatever_is_there(tmp_path, vits_checkpoint):
    """A spec frozen on a box that had no ViT-S pins no md5 for it, and that is a refusal.

    Nothing distinguishes the file at the path from any other, which is the exact condition
    §7.1 exists to make impossible — so the answer is "re-freeze here", not "trust it".
    """
    cfg = _config(tmp_path, vits_checkpoint)
    spec = build_spec(_config(tmp_path, None))  # frozen with the encoder unmounted
    assert spec.encoders[ENCODER].checkpoint_md5 is None
    with pytest.raises(ValueError, match="pins no md5 for demae-vits"):
        Extractor(spec, cfg).encoder(ARM)


def test_an_observed_md5_says_it_was_observed(tmp_path, vits_checkpoint):
    # "3ec5eb… because a box had it" and "3ec5eb… because the release says so" are different
    # claims, and only one of them survives someone mounting the wrong file.
    spec = build_spec(_config(tmp_path, vits_checkpoint))
    assert spec.encoders[ENCODER].checkpoint_md5_source == "observed"
    assert spec.encoders["demae-r5"].checkpoint_md5_source == "published"
    assert spec.encoders["demae-r5"].checkpoint_md5 == arms.DEMAE_R5_MD5


# --- The two published md5s this ticket settled --------------------------------


def test_the_release_bundles_ship_one_set_of_weights_under_two_directories():
    """#156's other question: which of `-v2` and non-`-v2` is authoritative.

    It has no weights axis. Both R5 bundles ship a byte-identical `encoder.pt` and so do both R4
    bundles — `-v2` re-exported the *bundle* (it adds `LICENSE` and the manifest's `licence`
    block) and not the weights. So these md5s identify the encoders under either directory, and
    `Paths` points at `-v2` because that is the bundle carrying its own terms.
    """
    assert arms.DEMAE_R5_MD5 == "3ec5ebc72fa28c43fe88f67692d9b41a"
    assert arms.DEMAE_R4_MD5 == "59e8346c4b08e1ab4fedd44f0802e12b"
    assert arms.ENCODERS["demae-r5"].checkpoint_md5 == arms.DEMAE_R5_MD5
    assert arms.ENCODERS["demae-r4"].checkpoint_md5 == arms.DEMAE_R4_MD5
    assert BenchmarkConfig().paths.demae_r5_checkpoint.endswith("release-R5-v2/encoder.pt")
    assert BenchmarkConfig().paths.demae_r4_checkpoint.endswith("release-R4-v2/encoder.pt")


@pytest.mark.parametrize("tag, md5", [("R5", arms.DEMAE_R5_MD5), ("R4", arms.DEMAE_R4_MD5)])
def test_the_published_md5s_are_the_files_on_this_box(tag, md5):
    """Where a box mounts them. The claim above is only worth having if it is checkable."""
    both = [REPO / ".release-work" / f"release-{tag}{suffix}" / "encoder.pt"
            for suffix in ("", "-v2")]
    mounted = [path for path in both if path.is_file()]
    if not mounted:
        pytest.skip(f"the {tag} release bundles are not mounted")
    for path in mounted:
        assert artifacts.file_md5(path) == md5, f"{path} is not the {tag} encoder this plan pins"


def test_the_registrys_timm_names_are_the_release_modules_own():
    """One home for "which published model is this shape". A second would be a second answer."""
    from pretrain.release import TIMM_MODELS

    published = set(TIMM_MODELS.values())
    for encoder in arms.ENCODERS.values():
        if encoder.checkpoint_format in ("timm", "mmselfsup_rgb"):
            assert encoder.timm_model in published
    assert TIMM_MODELS[(768, 12, 12)] == arms.ENCODERS["demae-r5"].timm_model
    assert TIMM_MODELS[(384, 12, 6)] == arms.ENCODERS["demae-vits"].timm_model


# --- The ImageNet-MAE family: §8's collapse, not a second key map ---------------


@pytest.fixture(scope="module")
def mmselfsup_checkpoint(tmp_path_factory):
    """A file shaped like the published mmselfsup MAE, at the shapes A6's ViT-B has.

    Built from `pretrain.init`'s own key map, so this fixture cannot drift from the translation
    it exercises: if P102's map gains a row, this checkpoint gains the tensor.
    """
    import timm
    import torch

    from pretrain.init import ENCODER_PREFIX, POS_EMBED_KEY, encoder_key_map

    reference = timm.create_model(
        "vit_base_patch16_224", pretrained=False, in_chans=1, num_classes=0, global_pool=""
    ).state_dict()
    patch_key = f"{ENCODER_PREFIX}patch_embed.projection.weight"
    state = {}
    for theirs, ours in encoder_key_map(LAYERS).items():
        tensor = reference[ours]
        if theirs == patch_key:  # theirs is RGB; ours was built single-channel
            tensor = tensor.repeat(1, 3, 1, 1) / 3.0
        state[theirs] = tensor
    state[POS_EMBED_KEY] = _sincos("row_col", width=768)
    path = tmp_path_factory.mktemp("mae") / "mae.pth"
    torch.save({"state_dict": state}, path)
    return path


@pytest.fixture(scope="module")
def imagenet_mae_entry():
    """The A6 clause, off a spec built without hashing the two mounted release encoders."""
    base = BenchmarkConfig()
    return build_spec(
        base.replace(
            paths=base.paths.replace(demae_r5_checkpoint=None, demae_r4_checkpoint=None)
        )
    ).encoders["imagenet-mae"]


def test_the_imagenet_mae_arm_reads_the_published_file_through_section_eights_collapse(
    tmp_path, mmselfsup_checkpoint
):
    """A6, end to end, against a file laid out the way the published ImageNet MAE is.

    The translation is :mod:`pretrain.init`'s — P102's key map and §8's summed patch-embed
    collapse — reused rather than restated, because every row of that map is a checked claim and
    a second copy is a second place for `backbone.ln1` to be mistaken for a block's.
    """
    base = BenchmarkConfig()
    cfg = base.replace(
        arm="A6",
        paths=base.paths.replace(
            artifacts_dir=str(tmp_path),
            demae_r5_checkpoint=None,
            demae_r4_checkpoint=None,
            imagenet_mae_checkpoint=str(mmselfsup_checkpoint),
        ),
        extract=base.extract.replace(compute_dtype="fp32"),
    )
    rng = np.random.default_rng(2)
    raw = rng.normal(700.0, 120.0, size=(1, 224, 224)).astype(np.int16)
    out = Extractor(build_spec(cfg), cfg).extract(raw, land_mask(raw), arm="A6")
    assert out.output.shape == (1, 196, 768)
    assert len(out.intermediates) == LAYERS
    assert out.cls is not None


def test_the_summed_collapse_is_the_one_section_eight_specifies(
    mmselfsup_checkpoint, imagenet_mae_entry
):
    """``sum``, not ``mean``: for an input identical in every channel — which is what a grayscale
    image replicated to RGB is — the summed kernel reproduces the original convolution exactly
    and the mean reproduces a third of it.
    """
    import torch

    from benchmark.extract import _timm_state_from_mmselfsup
    from pretrain.init import ENCODER_PREFIX, read_state_dict

    state, _ = read_state_dict(mmselfsup_checkpoint)
    translated = _timm_state_from_mmselfsup(state, imagenet_mae_entry)
    theirs = state[f"{ENCODER_PREFIX}patch_embed.projection.weight"]
    assert translated["patch_embed.proj.weight"].shape == (768, 1, 16, 16)
    assert torch.allclose(translated["patch_embed.proj.weight"], theirs.sum(dim=1, keepdim=True))


def test_a_checkpoint_that_is_not_the_mmselfsup_mae_is_refused(vits_checkpoint, imagenet_mae_entry):
    from benchmark.extract import _timm_state_from_mmselfsup
    from pretrain.init import read_state_dict

    state, _ = read_state_dict(vits_checkpoint)
    with pytest.raises(ValueError, match="not the mmselfsup MAE this contract declares"):
        _timm_state_from_mmselfsup(state, imagenet_mae_entry)


# --- The real encoder, where a box mounts one ----------------------------------


@requires_r5
def test_the_r5_release_encoder_satisfies_the_contract_it_is_declared_under(tmp_path):
    """Every claim above, against the shipped weights rather than a synthesized composition.

    This is what makes the contract a statement about the encoder Phase 1 released: its
    `pos_embed` really is MAE's column-first order, its `encoder.pt` really is the md5 the
    registry pins, and the tensor `output` names really is the one `MaskedAutoencoder.encode`
    computes.
    """
    import torch

    base = BenchmarkConfig()
    cfg = base.replace(
        arm="A4-R5",
        paths=base.paths.replace(artifacts_dir=str(tmp_path), demae_r5_checkpoint=str(R5)),
        extract=base.extract.replace(compute_dtype="fp32"),
    )
    extractor = Extractor(build_spec(cfg), cfg)
    rng = np.random.default_rng(3)
    raw = rng.normal(1500.0, 400.0, size=(1, 224, 224)).astype(np.int16)
    out = extractor.extract(raw, land_mask(raw), arm="A4-R5")

    assert out.output.shape == (1, 196, 768)
    assert len(out.intermediates) == 12
    assert out.cls.shape == (1, 768)
    # The relation, on the released weights: block twelve is not the output, and the final norm
    # is exactly what stands between them.
    assert not torch.allclose(out.output, out.intermediates[-1])
    with torch.no_grad():
        assert torch.allclose(
            out.output, extractor.encoder("A4-R5").norm(out.intermediates[-1]), atol=1e-6
        )


@requires_r5
def test_the_r5_encoder_under_the_r4_axis_order_is_refused(tmp_path):
    """The precondition that would otherwise be a sentence in §2.3.

    R5's weights learned MAE's column-first order and R4's learned mmselfsup's. The registry
    declares which is which; this asserts the spec is what enforces it, by handing R5's file to
    R4's declared entry and watching it refuse.
    """
    base = BenchmarkConfig()
    cfg = base.replace(
        arm="A4-R4",
        paths=base.paths.replace(artifacts_dir=str(tmp_path), demae_r4_checkpoint=str(R5)),
        extract=base.extract.replace(compute_dtype="fp32"),
    )
    # `freeze` refuses first, on the md5 — which is the earlier and cheaper of the two guards.
    with pytest.raises(ValueError, match="the demae-r4 encoder this plan benchmarks is"):
        build_spec(cfg)

    # Past that guard — a spec frozen honestly and the file swapped afterwards — the axis order
    # is what catches it, and it is the check that would survive an md5 collision.
    spec = build_spec(base.replace(paths=base.paths.replace(artifacts_dir=str(tmp_path))))
    entry = spec.encoders["demae-r4"]
    forged = EncoderSpec(**{**entry.to_dict(), "checkpoint_md5": artifacts.file_md5(R5)})
    forged_spec = replace(spec, encoders={**spec.encoders, "demae-r4": forged})
    with pytest.raises(ValueError, match="not sin-cos under the declared 'sincos2d_rowfirst'"):
        Extractor(forged_spec, cfg).encoder("A4-R4")
