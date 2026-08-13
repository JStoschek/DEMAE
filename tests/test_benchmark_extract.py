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
from benchmark import extract as extract_module
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

#: TerraMind's DEM tokenizer, which is not a file this repository mounts anywhere: it comes down
#: from the Hugging Face hub and lives in that cache. Resolved **offline** — `try_to_load_from_cache`
#: never reaches the network — so a box that has it runs these tests and a box that does not skips
#: them, and neither downloads 1.15 GB because a test module was imported.
def _cached_terramind() -> Path | None:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:  # terratorch is an extra; so is everything it brings
        return None
    hit = try_to_load_from_cache(
        "ibm-esa-geospatial/TerraMind-1.0-Tokenizer-DEM", "TerraMind_Tokenizer_DEM.pt"
    )
    return Path(hit) if isinstance(hit, str) else None


TERRAMIND = _cached_terramind()
requires_terramind = pytest.mark.skipif(
    TERRAMIND is None, reason="TerraMind's DEM tokenizer is not in this box's Hugging Face cache"
)

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


def _publish(monkeypatch, path, encoder=ENCODER):
    """Pin a checkpoint's md5 in the registry — the reviewed line a real box would commit.

    Since #156 an unpinned encoder **refuses**: a box does not get to certify its own weights by
    having mounted them, and the committed contract carries only md5s the registry publishes. So
    a test that wants to actually load something has to publish first, which is the workflow and
    not a way around it — `test_an_unpinned_encoder_refuses_rather_than_running_against_whatever_is_there`
    is the other half.
    """
    monkeypatch.setitem(
        arms.ENCODERS,
        encoder,
        replace(
            arms.ENCODERS[encoder],
            checkpoint_md5=hashlib.md5(Path(path).read_bytes()).hexdigest(),
        ),
    )
    return path


@pytest.fixture
def publish(monkeypatch):
    """:func:`_publish` bound to this test's monkeypatch. Returns the path, so it composes."""
    return lambda path, encoder=ENCODER: _publish(monkeypatch, path, encoder)


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


def test_the_committed_contract_in_this_repository_is_what_this_code_writes():
    """The contract family's defining property, on the file Task 2 will actually open.

    Byte-for-byte, which is only a fair thing to ask because `build_spec` is a function of the
    code and not of what the box has mounted — so this fails when the committed artifact goes
    stale against the registry, and never merely because of where it was frozen.
    """
    committed = REPO / "artifacts" / "task1" / SPEC_FILENAME
    assert committed.read_text() == build_spec().to_json()
    spec = load_spec(committed)
    assert spec.encoders["demae-r5"].checkpoint_md5 == arms.DEMAE_R5_MD5
    assert spec.encoders["demae-r4"].checkpoint_md5 == arms.DEMAE_R4_MD5
    # Neither unreleased encoder has a published md5, so neither is pinned — and an unpinned
    # entry refuses at extraction rather than running against whatever sits at the path.
    assert spec.encoders["demae-vits"].checkpoint_md5 is None
    assert spec.encoders["imagenet-mae"].checkpoint_md5 is None


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
        ({"grid_px": 16}, "is not 196 tokens"),
        # The clause Task 2 re-reads most directly: a `(D, 14, 14)` spec would load clean and
        # every consumer would unpack the tensors the other way round.
        ({"layout": "(N, D, 14, 14) channels-first"}, "canonical layout is"),
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
        ({"checkpoint_md5": "0" * 32}, "the registry publishes"),
        ({"checkpoint_md5": None}, "the registry publishes"),
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


# --- Nothing is owed, and the block that would say so is still there ------------


def test_every_registered_encoder_is_in_the_contract_and_nothing_is_pending(
    tmp_path, vits_checkpoint
):
    """#157 discharged the last owed half, so the committed file covers the whole registry.

    The `pending` block stays, empty. It is the shape this contract uses to say an encoder's
    half is *owed rather than absent*, and an empty block is the claim "nothing is owed" where a
    deleted field would be silence — which is the state #156 was careful not to ship.
    """
    spec = build_spec(_config(tmp_path, vits_checkpoint))
    assert set(spec.encoders) == set(arms.ENCODERS)
    assert spec.pending == {}
    for arm in ("A5", "A9"):
        assert spec.encoder_for(arm).family == "terramind"


def test_an_encoder_named_as_owed_refuses_by_naming_its_ticket(tmp_path, vits_checkpoint):
    """The mechanism itself, exercised against a spec that has something outstanding.

    Nothing in this repository is pending today, so the branch would otherwise ship untested —
    which is exactly the failure ADR-0012 Decision 4 names one paragraph over. A spec built by
    hand is the honest way to keep it a feature: the *next* encoder to be registered ahead of
    its build is the case, and this is what it will do.
    """
    spec = build_spec(_config(tmp_path, vits_checkpoint))
    owed = replace(
        spec,
        encoders={k: v for k, v in spec.encoders.items() if k != "terramind"},
        pending={
            "terramind": {
                "ticket": 12,
                "encoders": ["terramind"],
                "owes": ["the named tok_dem@224 lookup, checked against 670.665 / 951.272"],
            }
        },
    )
    with pytest.raises(NotImplementedError, match="ticket #12"):
        owed.encoder_for("A5")


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


def test_a_normalization_this_contract_does_not_implement_says_so(tile):
    raw, valid = tile
    with pytest.raises(NotImplementedError, match="pending"):
        normalize(raw[None], valid[None], normalization="none")


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
    """One forward pass at the fp32 reference, shared by the claims Decision 4 enumerates.

    Module-scoped, so it publishes the md5 through its own ``MonkeyPatch`` context rather than
    the function-scoped ``publish`` fixture, which pytest will not hand to it. The context is
    closed **before** the fixture returns: a module-scoped patch of a module-level registry
    would stay live for every test that ran afterwards, and two of them are precisely about the
    registry *not* publishing this md5. Closing it is safe because the extractor caches the
    model it loaded, so nothing re-reads the registry after this line.
    """
    rng = np.random.default_rng(1)
    raw = rng.normal(900.0, 150.0, size=(2, dataset_config.TILE_PX, dataset_config.TILE_PX))
    raw = raw.astype(np.int16)
    valid = land_mask(raw)
    with pytest.MonkeyPatch.context() as mp:
        _publish(mp, vits_checkpoint)
        extractor = _extractor(tmp_path_factory.mktemp("features"), vits_checkpoint)
        out = extractor.extract(raw, valid, arm=ARM)
    return extractor, raw, valid, out


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


def test_the_note_states_the_relation_on_both_sides_not_just_the_demae_one(tmp_path, vits_checkpoint):
    """ADR-0012 Decision 4 says "on both sides", and the two sides say different things."""
    spec = build_spec(_config(tmp_path, vits_checkpoint))

    normed = spec.encoders["demae-r5"].note
    assert "**not** `intermediates[11]`" in normed

    # TerraMind: p118 §5.3 measured twelve intermediates with no separate post-norm entry, the
    # last bit-identical to `encoder(x)`. Same two fields, and on this side they are one tensor.
    un_normed = spec.encoders["terramind"].note
    assert "bit-identical to `intermediates[11]`" in un_normed
    assert arms.ENCODERS["terramind"].output_is_normed is False
    # And the note says *why* they are one tensor, which is not "nothing happens after block 12":
    # `ViTEncoder` writes its residual `post_mlp` back over the last intermediate, so both fields
    # carry it. A note that got the mechanism wrong would be a plausible sentence in a file whose
    # whole job is being re-read rather than re-derived.
    assert "post_mlp" in un_normed


def test_the_canonical_layout_is_patch_tokens_with_cls_as_a_side_field(features):
    _, raw, _, out = features
    assert out.output.shape == (len(raw), 196, WIDTH)  # ADR-0012 §5
    assert out.cls is not None and out.cls.shape == (len(raw), WIDTH)
    # The side field is the *final* CLS and is never one of the 196: a consumer that concatenated
    # it would be feeding the head a token TerraMind structurally does not have.
    assert out.output.shape[1] == 196


def test_the_reference_tensor_is_fp32_whatever_the_compute_dtype(tmp_path, vits_checkpoint, tile, publish):
    """ADR-0012 §6: the compute dtype is a declared field; what comes back is the reference.

    A task moving its compute dtype re-checks it under #134's 0.99 materiality bar rather than
    changing what the contract hands its consumer.
    """
    import torch

    raw, valid = tile
    cfg = _config(tmp_path, publish(vits_checkpoint))
    cfg = cfg.replace(extract=cfg.extract.replace(compute_dtype="bf16"))
    out = Extractor(build_spec(cfg), cfg).extract(raw, valid, arm=ARM)
    assert out.output.dtype == torch.float32
    assert all(block.dtype == torch.float32 for block in out.intermediates)


def test_a_single_tile_and_a_batch_are_the_same_call(tmp_path, vits_checkpoint, tile, publish):
    import torch

    raw, valid = tile
    extractor = _extractor(tmp_path, publish(vits_checkpoint))
    one = extractor.extract(raw, valid, arm=ARM)
    batched = extractor.extract(raw[None], valid[None], arm=ARM)
    assert one.output.shape == batched.output.shape == (1, 196, WIDTH)
    assert torch.allclose(one.output, batched.output)


def test_the_unrolled_pass_is_timms_own_forward_features(tmp_path, vits_checkpoint, tile, publish):
    """The composition, pinned bit-identical to the library call it replaces.

    Unrolled rather than routed through ``get_intermediate_layers`` so that ``output`` is the
    exact tensor `MaskedAutoencoder.encode` computes — patch embed, CLS and position, every
    block, final norm — and not a helper's interpretation of it. That is only worth anything if
    it *is* the same composition, so this asserts equality rather than closeness.
    """
    import torch

    from benchmark.extract import _forward_blocks

    raw, valid = tile
    extractor = _extractor(tmp_path, publish(vits_checkpoint))
    model = extractor.encoder(ARM)
    x = normalize(raw[None], valid[None], normalization="per_tile")
    tensor = torch.from_numpy(x).unsqueeze(1)
    with torch.no_grad():
        assert torch.equal(_forward_blocks(model, tensor)[0], model.forward_features(tensor))


# --- ADR-0012 §7.1: encoder identity, checked against the model actually loaded -


def test_a_wrong_checkpoint_md5_is_refused(tmp_path, vits_checkpoint, publish):
    """An encoder handed the wrong checkpoint returns a plausible tensor and raises nothing.

    So the md5 the spec was frozen against is checked before the file is read at all — which is
    also why a 340 MB swap costs a hash rather than a forward pass to detect.
    """
    cfg = _config(tmp_path, publish(vits_checkpoint))
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


def test_a_declared_layer_count_that_does_not_match_the_model_loaded_is_refused(tmp_path, publish):
    """Off the checkpoint's own keys, before a model is built.

    A strict load would refuse too, but it would name two dozen missing tensors rather than the
    one fact that is wrong — and ADR-0012 §3's contract is *about* the layer count, so it is the
    thing the refusal should say.
    """
    stub = _stub_checkpoint(tmp_path / "eleven.pt", depth=11)
    with pytest.raises(ValueError, match="has 11 transformer blocks"):
        _extractor(tmp_path, publish(stub)).encoder(ARM)


def test_a_pos_embed_under_the_wrong_axis_order_is_refused_and_the_message_says_which(tmp_path, publish):
    """§2.3's R4-versus-R5 precondition, enforced rather than remembered.

    Every name and every shape matches under either order, so a `strict=True` load succeeds and
    the encoder reads every tile as its own transpose. The two differ by 2.0 — the full range of
    a sinusoid — so there is nothing borderline about the check.
    """
    stub = _stub_checkpoint(tmp_path / "transposed.pt", pos_embed="row_col")
    with pytest.raises(ValueError, match="not sin-cos under the declared 'sincos2d'"):
        _extractor(tmp_path, publish(stub)).encoder(ARM)


def test_a_pos_embed_that_is_not_sin_cos_at_all_is_refused(tmp_path, publish):
    import torch

    stub = _stub_checkpoint(
        tmp_path / "learned.pt", pos_embed=torch.zeros(1, 1 + 196, WIDTH)
    )
    with pytest.raises(ValueError, match="not sin-cos under the declared"):
        _extractor(tmp_path, publish(stub)).encoder(ARM)


def test_a_pos_embed_of_the_wrong_shape_is_refused(tmp_path, publish):
    stub = _stub_checkpoint(tmp_path / "wide.pt", pos_embed="col_row", width=768)
    with pytest.raises(ValueError, match=r"`pos_embed` is \(1, 197, 768\)"):
        _extractor(tmp_path, publish(stub)).encoder(ARM)


def test_a_checkpoint_with_no_pos_embed_is_refused(tmp_path, publish):
    stub = _stub_checkpoint(tmp_path / "bare.pt", pos_embed=None)
    with pytest.raises(ValueError, match="no `pos_embed`"):
        _extractor(tmp_path, publish(stub)).encoder(ARM)


def test_an_unmounted_encoder_names_the_path_knob(tmp_path, vits_checkpoint, publish):
    cfg = _config(tmp_path, publish(vits_checkpoint))
    spec = build_spec(cfg)
    with pytest.raises(FileNotFoundError, match="paths.demae_vits_checkpoint"):
        Extractor(spec, _config(tmp_path, None)).encoder(ARM)


def test_an_unpinned_encoder_refuses_rather_than_running_against_whatever_is_there(
    tmp_path, vits_checkpoint
):
    """This plan publishes no ViT-S md5, so nothing distinguishes the file at the path.

    That is the exact condition §7.1 exists to make impossible, and the answer is to *publish*
    one — a reviewed line in the registry — not to let a box certify its own weights by having
    mounted them.
    """
    with pytest.raises(ValueError, match="publishes no md5 for demae-vits"):
        _extractor(tmp_path, vits_checkpoint).encoder(ARM)


def test_the_committed_bytes_do_not_move_with_what_a_box_has_mounted(tmp_path, vits_checkpoint):
    """The property that lets the spec's md5 mean *which contract*, not *whose freeze*.

    An md5 recorded because a box happened to have the file would put environment inside the one
    committed artifact that must not carry any — and would hand the next box a `FileExistsError`
    against a contract that had not changed. So `build_spec` is a function of the code, and
    mounting an encoder moves nothing in the file.
    """
    mounted = build_spec(_config(tmp_path, vits_checkpoint))
    unmounted = build_spec(_config(tmp_path, None))
    assert mounted == unmounted
    assert mounted.encoders[ENCODER].checkpoint_md5 is None  # unpinned either way
    assert mounted.encoders["demae-r5"].checkpoint_md5 == arms.DEMAE_R5_MD5  # published either way


def test_a_spec_pinning_an_md5_the_registry_does_not_publish_is_refused(tmp_path, vits_checkpoint):
    cfg = _config(tmp_path, vits_checkpoint)
    body = build_spec(cfg).to_dict()
    body["encoders"][ENCODER]["checkpoint_md5"] = hashlib.md5(
        vits_checkpoint.read_bytes()
    ).hexdigest()
    (tmp_path / SPEC_FILENAME).write_text(json.dumps(body))
    with pytest.raises(ValueError, match="the registry publishes None"):
        load_spec(tmp_path / SPEC_FILENAME)


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
    tmp_path, mmselfsup_checkpoint, monkeypatch
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
    # A6's md5 is unpublished, and an unpinned encoder refuses. Publishing one is a registry
    # line, so that is what this does — the same act a box mounting the real file would commit.
    monkeypatch.setitem(
        arms.ENCODERS,
        "imagenet-mae",
        replace(
            arms.ENCODERS["imagenet-mae"],
            checkpoint_md5=hashlib.md5(mmselfsup_checkpoint.read_bytes()).hexdigest(),
        ),
    )
    rng = np.random.default_rng(2)
    raw = rng.normal(700.0, 120.0, size=(1, 224, 224)).astype(np.int16)
    out = Extractor(build_spec(cfg), cfg).extract(raw, land_mask(raw), arm="A6")
    assert out.output.shape == (1, 196, 768)
    assert len(out.intermediates) == LAYERS
    assert out.cls is not None


def test_the_imagenet_mae_arm_refuses_until_its_md5_is_published(tmp_path, mmselfsup_checkpoint):
    """The state this repository is actually in, asserted rather than left implicit."""
    base = BenchmarkConfig()
    cfg = base.replace(
        arm="A6",
        paths=base.paths.replace(
            artifacts_dir=str(tmp_path),
            demae_r5_checkpoint=None,
            demae_r4_checkpoint=None,
            imagenet_mae_checkpoint=str(mmselfsup_checkpoint),
        ),
    )
    with pytest.raises(ValueError, match="publishes no md5 for imagenet-mae"):
        Extractor(build_spec(cfg), cfg).encoder("A6")


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


# --- The TerraMind half (#157) -------------------------------------------------
#
# Three checks specific to this encoder, plus ADR-0012 §6's unmeasured half. Split the way the
# module is: everything that is a fact about *geometry* or about the *contract* is checked with
# no weights at all — ADR-0011's whole argument is closed-form and needs none — and the refusals
# that fire before a model is built use the same few-tensor stubs the timm side does. The real
# 1.15 GB tokenizer, where a box mounts one, then runs the forward pass none of that can fake.


def _terramind_state(*, depth=LAYERS, width=768, native_px=16, pos_emb="closed_form"):
    """The `encoder.`-prefixed few-tensor stub, for the refusals that never build a model."""
    import torch

    from terratorch.models.backbones.terramind.tokenizer.models.vit_models import (
        build_2d_sincos_posemb,
    )

    state = {f"encoder.blocks.{i}.norm1.weight": torch.zeros(width) for i in range(depth)}
    if pos_emb == "closed_form":
        state["encoder.pos_emb"] = build_2d_sincos_posemb(native_px, native_px, embed_dim=width)
    elif pos_emb is not None:
        state["encoder.pos_emb"] = pos_emb
    return state


def _terramind_checkpoint(path, **kwargs):
    import torch

    torch.save(_terramind_state(**kwargs), path)
    return path


def _terramind_config(tmp_path, checkpoint) -> BenchmarkConfig:
    base = BenchmarkConfig()
    return base.replace(
        arm="A5",
        paths=base.paths.replace(
            artifacts_dir=str(tmp_path),
            demae_r5_checkpoint=None,
            demae_r4_checkpoint=None,
            terramind_checkpoint=None if checkpoint is None else str(checkpoint),
        ),
        extract=base.extract.replace(compute_dtype="fp32"),
    )


def _terramind_extractor(tmp_path, checkpoint, monkeypatch):
    """An extractor whose TerraMind md5 is the stub's, so the load reaches the checks under test.

    Republishing rather than bypassing: §7.1's md5 refusal is the *first* guard and has its own
    test below, so a TerraMind test that wanted to reach the second guard would otherwise only
    ever be re-testing the first.

    **Both** entries, because A5 and A9 read the same file — `terramind` and
    `terramind-tokenizer` share a `checkpoint_field`, so republishing one leaves `freeze`
    comparing the other against the real 1.15 GB tokenizer's md5.
    """
    for name in ("terramind", "terramind-tokenizer"):
        _publish(monkeypatch, checkpoint, name)
    cfg = _terramind_config(tmp_path, checkpoint)
    return Extractor(build_spec(cfg), cfg)


# --- The pin, which is the first thing this ticket owes


def test_the_terramind_environment_pins_diffusers_at_the_verified_version():
    """AC 1, and the hazard p118 §4 actually found.

    terratorch declares a **bare** `diffusers` — the `==0.30.0` everyone quotes lives in the body
    of a warning that fires when the VQVAE import fails, not in `requires-dist` — so with a range
    here a future diffusers release breaks the tokenizer import with no lockfile signal at all.
    Both pins are therefore exact, and this asserts the exactness rather than the values alone.
    """
    import tomllib

    extras = tomllib.loads((REPO / "pyproject.toml").read_text())["project"][
        "optional-dependencies"
    ]
    assert extras["terramind"] == ["terratorch==1.2.10", "diffusers==0.39.0"]
    # And it is an *extra*: `pip install demae` still builds a corpus on a box with no CUDA
    # stack, which is the property p118's "must not become a project dependency" was protecting.
    assert not any("terratorch" in dep for dep in tomllib.loads(
        (REPO / "pyproject.toml").read_text()
    )["project"]["dependencies"])


def test_the_installed_versions_are_the_pinned_ones():
    """The pin is only a pin if the environment the suite runs in honours it."""
    import importlib.metadata as md

    assert md.version("terratorch") == "1.2.10"
    assert md.version("diffusers") == "0.39.0"


# --- ADR-0012 §7.2: the constants, selected by name and checked by value


def test_the_constants_are_read_out_of_the_installed_package(monkeypatch):
    from benchmark.extract import TERRAMIND_MEAN, TERRAMIND_STD, terramind_constants

    assert terramind_constants() == (TERRAMIND_MEAN, TERRAMIND_STD) == (670.665, 951.272)


def test_constants_that_resolve_by_name_to_another_value_are_refused(monkeypatch):
    """The *value* half. #138 found three DEM pairs across IBM's artifacts, and this is what
    catches a terratorch upgrade retuning the pair the name resolves to — a change that would
    otherwise silently rescale every A5 and A9 input and raise nothing (p118 §7)."""
    from terratorch.models.backbones.terramind.model import terramind_register

    from benchmark.extract import TERRAMIND_MODALITY, terramind_constants

    monkeypatch.setitem(terramind_register.v1_pretraining_mean, TERRAMIND_MODALITY, [651.663])
    with pytest.raises(ValueError, match="651.663"):
        terramind_constants()


def test_a_renamed_modality_is_refused_rather_than_falling_back_to_a_value(monkeypatch):
    """The *name* half. §2.1's "by name rather than by value" is about **selection**: the name is
    what picks the pair belonging to the v1 checkpoint this plan benchmarks, so a rename is a
    refusal and not something to paper over with the hardcoded number."""
    from terratorch.models.backbones.terramind.model import terramind_register

    from benchmark.extract import TERRAMIND_MODALITY, terramind_constants

    monkeypatch.delitem(terramind_register.v1_pretraining_mean, TERRAMIND_MODALITY)
    with pytest.raises(ValueError, match="publishes no v1 pretraining constants"):
        terramind_constants()


def test_multi_channel_constants_are_refused(monkeypatch):
    from terratorch.models.backbones.terramind.model import terramind_register

    from benchmark.extract import TERRAMIND_MODALITY, terramind_constants

    monkeypatch.setitem(
        terramind_register.v1_pretraining_mean, TERRAMIND_MODALITY, [670.665, 670.665]
    )
    with pytest.raises(ValueError, match="single-channel"):
        terramind_constants()


def test_the_terramind_diet_is_a_global_affine_over_metres(tile):
    """§2.1's other diet, and p118 §7's trap from the other side.

    A 1589 m offset must survive as 1589/951.272 — absolute elevation is *signal* to TerraMind,
    where per-tile standardization destroys it. The two normalizations produce identically shaped
    tensors the encoder accepts either of, so what makes them different has to be asserted.
    """
    raw, valid = tile
    out = normalize(raw[None], valid[None], normalization="terramind_v1")
    expected = (raw.astype(np.float32) - np.float32(670.665)) / np.float32(951.272)
    np.testing.assert_array_equal(out[0], expected)

    offset = (raw + 1589).astype(np.int16)
    shifted = normalize(offset[None], valid[None], normalization="terramind_v1")
    assert (shifted[0] - out[0]).mean() == pytest.approx(1589 / 951.272, rel=1e-4)


def test_the_terramind_diet_ignores_the_mask_and_says_so(tile):
    """ADR-0012 Decision 1: the affine is global, so there is no per-tile term a mask could enter.

    Asserted rather than left to the reader, because the *other* diet in this same function is
    defined entirely by the mask, and "which of the two reads it" is exactly the kind of thing a
    consumer would otherwise have to infer from the implementation.
    """
    raw, valid = tile
    everything = np.ones_like(valid)
    np.testing.assert_array_equal(
        normalize(raw[None], valid[None], normalization="terramind_v1"),
        normalize(raw[None], everything[None], normalization="terramind_v1"),
    )
    # Where the per-tile diet moves with it, which is the contrast that makes the claim mean
    # something rather than being true of both.
    assert not np.array_equal(
        normalize(raw[None], valid[None], normalization="per_tile"),
        normalize(raw[None], everything[None], normalization="per_tile"),
    )


# --- ADR-0011: the derivation, as a declared field


def test_the_contract_declares_the_centre_anchored_crop():
    """AC 3. The value is in the *committed* file, not only in the code that writes it."""
    from benchmark.extract import TERRAMIND_POS_EMB_ANCHOR

    committed = json.loads((REPO / "artifacts" / "task1" / SPEC_FILENAME).read_text())
    for name in ("terramind", "terramind-tokenizer"):
        block = committed["encoders"][name]["pos_emb"]
        assert block == {
            "derivation": "rederived_14x14_crop",
            "grid_px": 14,
            "native_grid_px": 16,
            "anchor": list(TERRAMIND_POS_EMB_ANCHOR),
            "adr": "ADR-0011",
        }
    # Every other encoder declares the identity rather than saying nothing, so a reader of the
    # file never has to work out whether a missing derivation means "none" or "not yet decided".
    for name in ("demae-r5", "demae-r4", "demae-vits", "imagenet-mae"):
        assert committed["encoders"][name]["pos_emb"]["derivation"] == "as_published"


def test_a_bicubic_derived_contract_is_distinguishable_from_a_re_derived_one(
    tmp_path, vits_checkpoint
):
    """AC 3's second half, and the reason the derivation is a *value* and not a code path.

    ADR-0011 Decision 4 pre-registers an A5-at-20k ablation against the bicubic resize (ADR-0013
    R11), so both have to be expressible — and the spec's md5 goes into every run record, so the
    two must not hash the same. If they did, a results table could not say which of them produced
    it, which is the one thing the committed file exists to make possible.
    """
    re_derived = build_spec(_config(tmp_path, vits_checkpoint))
    bicubic_registry = replace(arms.ENCODERS["terramind"], pos_emb_derivation="bicubic_resize")
    entry = re_derived.encoders["terramind"]
    bicubic = replace(
        re_derived,
        encoders={
            **re_derived.encoders,
            "terramind": EncoderSpec(
                **{
                    **entry.to_dict(),
                    "pos_emb": extract_module.pos_emb_block(bicubic_registry, grid_px=14),
                }
            ),
        },
    )
    assert bicubic.encoders["terramind"].pos_emb["derivation"] == "bicubic_resize"
    assert "anchor" not in bicubic.encoders["terramind"].pos_emb  # a resize has no anchor
    assert bicubic.md5() != re_derived.md5()
    # And it is refused at load while the registry says otherwise, so the ablation is a reviewed
    # registry change rather than an edit to a committed file nobody diffs.
    with pytest.raises(ValueError, match="derives its pos_emb as"):
        bicubic.check()


def test_the_crop_is_the_closed_form_at_this_grid_and_the_resize_is_not():
    """ADR-0011's Context section, re-measured with no weights and no GPU — as it was taken.

    Two claims, and the second is the one that bites. The crop is *bit-identical* to rebuilding
    the embedding at 14, because `build_2d_sincos_posemb` builds its grid from `torch.arange` and
    the index is absolute. The resize preserves *normalized* position instead, so it stretches
    the index spacing to 1.1429 where a crop's is exactly 1.0.
    """
    import torch

    from terratorch.models.backbones.terramind.tokenizer.models.vit_models import (
        build_2d_sincos_posemb,
    )

    native = build_2d_sincos_posemb(16, 16, embed_dim=768)
    at_grid = build_2d_sincos_posemb(14, 14, embed_dim=768)
    assert torch.equal(native[:, :, :14, :14], at_grid)  # max abs difference 0.0

    def rel_rms(a, b):
        return float((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt())

    centre = extract_module.derive_pos_emb(
        native, {"derivation": "rederived_14x14_crop", "grid_px": 14, "native_grid_px": 16,
                 "anchor": [1, 1]}
    )
    resized = extract_module.derive_pos_emb(
        native, {"derivation": "bicubic_resize", "grid_px": 14, "native_grid_px": 16}
    )
    # ADR-0011's anchor table, both rows: the resize sits 13.57% from the centre crop and 26.25%
    # from the top-left one — and the anchors are 23.35% apart, which is why the anchor is
    # declared rather than assumed.
    assert rel_rms(resized, centre) == pytest.approx(0.1357, abs=5e-4)
    assert rel_rms(resized, native[:, :, :14, :14]) == pytest.approx(0.2625, abs=5e-4)
    assert rel_rms(centre, native[:, :, :14, :14]) == pytest.approx(0.2335, abs=5e-4)


def test_substituting_the_crop_makes_terratorchs_own_resize_a_no_op():
    """Why `_build_terratorch` replaces a parameter instead of patching `ViTEncoder.forward`.

    `forward` bicubically interpolates `pos_emb` to whatever grid the input implies — including
    when it is *already* that grid, where `align_corners=False` maps every destination index to
    itself and the interpolation is exactly the identity. So the crop goes in as the parameter
    and terratorch's own line then adds it unchanged. Equality, not closeness: if this were
    approximate, "we corrected how terratorch feeds TerraMind" would be two changes, not one.
    """
    import torch

    from terratorch.models.backbones.terramind.tokenizer.models.vit_models import (
        build_2d_sincos_posemb,
    )

    crop = build_2d_sincos_posemb(16, 16, embed_dim=768)[:, :, 1:15, 1:15]
    same = torch.nn.functional.interpolate(
        crop, size=(14, 14), mode="bicubic", align_corners=False
    )
    assert torch.equal(same, crop)


def test_a_derivation_this_contract_does_not_implement_says_so():
    import torch

    with pytest.raises(NotImplementedError, match="pos_emb derivation"):
        extract_module.derive_pos_emb(
            torch.zeros(1, 8, 16, 16),
            {"derivation": "stretch_it", "grid_px": 14, "native_grid_px": 16},
        )


def test_a_crop_that_does_not_fit_the_native_grid_is_refused():
    import torch

    with pytest.raises(ValueError, match="does not fit"):
        extract_module.derive_pos_emb(
            torch.zeros(1, 8, 16, 16),
            {"derivation": "rederived_14x14_crop", "grid_px": 14, "native_grid_px": 16,
             "anchor": [4, 4]},
        )


# --- §7.1 on this side: identity, layer count and shape against the model loaded


def test_an_unmounted_terramind_encoder_names_the_path_knob(tmp_path):
    spec = build_spec(_terramind_config(tmp_path, None))
    with pytest.raises(FileNotFoundError, match="paths.terramind_checkpoint"):
        Extractor(spec, _terramind_config(tmp_path, None)).encoder("A5")


def test_a_wrong_terramind_checkpoint_md5_is_refused(tmp_path, monkeypatch):
    """The published md5 is #138's real guard here: the v0.1 DEM tokenizer is a different file
    with the same shape and a different diet, and it would load without complaint.

    Both ends of the rule, because both are reachable. `freeze` refuses to *pin* a file that is
    not the published one; and past that — a spec frozen honestly and the file swapped afterwards
    — the load refuses to read it.
    """
    checkpoint = _terramind_checkpoint(tmp_path / "not_the_published_one.pt")
    cfg = _terramind_config(tmp_path, checkpoint)
    with pytest.raises(ValueError, match="the terramind encoder this plan benchmarks is"):
        build_spec(cfg)

    extractor = _terramind_extractor(tmp_path, checkpoint, monkeypatch)
    swapped = tmp_path / "swapped.pt"
    swapped.write_bytes(checkpoint.read_bytes() + b"\n")
    moved = cfg.replace(paths=cfg.paths.replace(terramind_checkpoint=str(swapped)))
    with pytest.raises(ValueError, match="declares terramind as"):
        Extractor(extractor.spec, moved).encoder("A5")


def test_a_checkpoint_with_no_encoder_half_is_refused(tmp_path, monkeypatch):
    import torch

    path = tmp_path / "decoder_only.pt"
    torch.save({"decoder.blocks.0.norm1.weight": torch.zeros(8)}, path)
    with pytest.raises(ValueError, match="no `encoder.` tensors"):
        _terramind_extractor(tmp_path, path, monkeypatch).encoder("A5")


def test_a_declared_layer_count_that_does_not_match_the_terratorch_model_is_refused(
    tmp_path, monkeypatch
):
    """AC 4, off the checkpoint's own keys and before anything 361 MB is built.

    ADR-0012 §3 emits one intermediate per block, so a spec declaring twelve against an
    eleven-block checkpoint would hand Task 2 a contract whose `intermediates` index means
    something else — and a strict load would report it as two dozen unmatched tensor names
    rather than as the one fact that is wrong.
    """
    checkpoint = _terramind_checkpoint(tmp_path / "eleven.pt", depth=11)
    with pytest.raises(ValueError, match="has 11 transformer blocks"):
        _terramind_extractor(tmp_path, checkpoint, monkeypatch).encoder("A5")


def test_a_terramind_checkpoint_with_no_pos_emb_is_refused(tmp_path, monkeypatch):
    checkpoint = _terramind_checkpoint(tmp_path / "no_pos.pt", pos_emb=None)
    with pytest.raises(ValueError, match="no `encoder.pos_emb`"):
        _terramind_extractor(tmp_path, checkpoint, monkeypatch).encoder("A5")


def test_a_pos_emb_of_the_wrong_native_shape_is_refused(tmp_path, monkeypatch):
    """AC 4's shape half. A `[1, 768, 14, 14]` parameter would crop to nothing meaningful and a
    `[1, 384, 16, 16]` one is a different width than the head is built for (§4.1)."""
    import torch

    from terratorch.models.backbones.terramind.tokenizer.models.vit_models import (
        build_2d_sincos_posemb,
    )

    checkpoint = _terramind_checkpoint(
        tmp_path / "native_14.pt", pos_emb=build_2d_sincos_posemb(14, 14, embed_dim=768)
    )
    with pytest.raises(ValueError, match=r"is \(1, 768, 14, 14\)"):
        _terramind_extractor(tmp_path, checkpoint, monkeypatch).encoder("A5")

    narrow = _terramind_checkpoint(
        tmp_path / "narrow.pt", pos_emb=torch.zeros(1, 384, 16, 16), width=384
    )
    with pytest.raises(ValueError, match=r"is \(1, 384, 16, 16\)"):
        _terramind_extractor(tmp_path, narrow, monkeypatch).encoder("A5")


def test_a_pos_emb_that_is_not_the_closed_form_is_refused(tmp_path, monkeypatch):
    """ADR-0011's tamper-evidence consequence, discharged.

    The whole decision rests on the embedding being a closed form of the token's integer grid
    index, in the checkpoint only because it is wrapped in an `nn.Parameter`. A terratorch that
    shipped a learned or retuned one would keep every name and every shape — and cropping it
    would no longer be re-deriving it, which is the claim the ADR makes and this is the check
    the ADR asked for.
    """
    import torch

    checkpoint = _terramind_checkpoint(
        tmp_path / "learned.pt", pos_emb=torch.randn(1, 768, 16, 16) * 0.02
    )
    with pytest.raises(ValueError, match="not `build_2d_sincos_posemb"):
        _terramind_extractor(tmp_path, checkpoint, monkeypatch).encoder("A5")


def test_the_timm_path_refuses_a_derivation_it_cannot_apply(tmp_path, vits_checkpoint, publish):
    """The other side of the same clause. The timm path loads the checkpoint's own tensor
    unchanged, so a declared derivation it does not apply would run silently as the identity —
    right today, wrong the moment an encoder declares otherwise."""
    cfg = _config(tmp_path, publish(vits_checkpoint))
    spec = build_spec(cfg)
    entry = spec.encoders[ENCODER]
    moved = EncoderSpec(
        **{
            **entry.to_dict(),
            "pos_emb": {
                "derivation": "rederived_14x14_crop",
                "grid_px": 14,
                "native_grid_px": 16,
                "anchor": [1, 1],
                "adr": "ADR-0011",
            },
        }
    )
    forged = replace(spec, encoders={**spec.encoders, ENCODER: moved})
    with pytest.raises(NotImplementedError, match="the timm path does not apply"):
        Extractor(forged, cfg).encoder(ARM)


# --- The real tokenizer, where a box mounts one --------------------------------


@requires_terramind
def test_the_terramind_encoder_satisfies_the_contract_it_is_declared_under(tmp_path):
    """AC 6 and the shape of ADR-0012 Decision 4, on the shipped weights.

    Twelve intermediates in block order, the canonical `(N, 196, D)` slice, and `cls` correctly
    `None` — which is not "we did not compute one" but a claim about the architecture: TerraMind
    structurally has no CLS token, so a statistic pooled over it would compare one model's token
    to the other's absence (ADR-0012 §5).
    """
    import torch

    cfg = _terramind_config(tmp_path, TERRAMIND)
    extractor = Extractor(load_spec(write_spec(cfg)), cfg)
    rng = np.random.default_rng(5)
    raw = rng.normal(1500.0, 400.0, size=(2, 224, 224)).astype(np.int16)
    out = extractor.extract(raw, land_mask(raw), arm="A5")

    assert out.output.shape == (2, 196, 768)
    assert out.output.dtype == torch.float32
    assert len(out.intermediates) == LAYERS
    assert all(block.shape == (2, 196, 768) for block in out.intermediates)
    assert out.cls is None
    assert extractor.spec.encoders["terramind"].has_cls is False

    # The relation on the un-normed side: `output` *is* block twelve, bit-identically (p118
    # §5.3), where on the DEMAE side a final norm stands between them. Both fields still exist,
    # because a shared index would quietly compare one to the other.
    assert torch.equal(out.output, out.intermediates[-1])
    assert extractor.spec.encoders["terramind"].output_is_normed is False
    # And they really are twelve *different* tensors, not one repeated.
    assert not torch.allclose(out.intermediates[0], out.intermediates[-1])


@requires_terramind
def test_the_re_derivation_is_what_the_encoder_actually_runs(tmp_path):
    """ADR-0011 Decision 2 as behaviour, not as a field.

    Two things at once. The parameter the built encoder holds is the declared crop, and running
    the same tile under terratorch's default bicubic path gives *different features* — so the
    intervention is real rather than a relabelling. #134 measured what that difference is worth:
    mean per-token cosine 0.9708, below its own pre-registered 0.99 bar.
    """
    import torch

    cfg = _terramind_config(tmp_path, TERRAMIND)
    extractor = Extractor(load_spec(write_spec(cfg)), cfg)
    model = extractor.encoder("A5")
    assert tuple(model.pos_emb.shape) == (1, 768, 14, 14)
    assert model.pos_emb.requires_grad is False

    from pretrain.init import read_state_dict

    state, _ = read_state_dict(TERRAMIND)
    native = state["encoder.pos_emb"]
    assert torch.equal(model.pos_emb.detach().cpu(), native[:, :, 1:15, 1:15])

    rng = np.random.default_rng(6)
    raw = rng.normal(1500.0, 400.0, size=(1, 224, 224)).astype(np.int16)
    crop_features = extractor.extract(raw, land_mask(raw), arm="A5").output

    from torch import nn

    model.pos_emb = nn.Parameter(
        extract_module.derive_pos_emb(
            native, {"derivation": "bicubic_resize", "grid_px": 14, "native_grid_px": 16}
        ).clone(),
        requires_grad=False,
    )
    bicubic_features = extractor.extract(raw, land_mask(raw), arm="A5").output
    assert not torch.allclose(crop_features, bicubic_features)
    cosine = torch.nn.functional.cosine_similarity(
        crop_features.flatten(0, 1), bicubic_features.flatten(0, 1), dim=-1
    )
    assert float(cosine.mean()) < extract_module.PRECISION_THRESHOLD  # #134's bar, 0.9708


@requires_terramind
def test_the_precision_agreement_is_measured_on_first_run_rather_than_inherited(tmp_path):
    """AC 5 / ADR-0012 §6's *unmeasured half*.

    The 0.999968 §6 quotes is DEMAE-family weights on MPS, and §6 names the rest as unmeasured
    rather than generalizing it. So the figure this arm records has to come from this arm, and
    what is asserted here is that it *was measured* — the recorded token count is the batch that
    actually ran — rather than that it landed on any particular number.
    """
    base = _terramind_config(tmp_path, TERRAMIND)
    cfg = base.replace(extract=base.extract.replace(compute_dtype="bf16"))
    extractor = Extractor(load_spec(write_spec(cfg)), cfg)
    assert extractor.precision_agreement("A5") is None  # nothing has run yet

    rng = np.random.default_rng(7)
    raw = rng.normal(1500.0, 400.0, size=(2, 224, 224)).astype(np.int16)
    extractor.extract(raw, land_mask(raw), arm="A5")

    measured = extractor.precision_agreement("A5")
    assert measured["encoder"] == "terramind"
    assert (measured["reference_dtype"], measured["compute_dtype"]) == ("fp32", "bf16")
    assert measured["tiles"] == 2 and measured["tokens"] == 2 * 196
    assert 0.0 <= measured["mean_cosine"] <= 1.0
    assert measured["min_cosine"] <= measured["mean_cosine"]
    assert measured["threshold"] == extract_module.PRECISION_THRESHOLD
    assert measured["immaterial"] is (
        measured["mean_cosine"] >= extract_module.PRECISION_THRESHOLD
    )
    # And A9 reads the *other* encoder entry, so it does not inherit this one's figure either.
    assert extractor.precision_agreement("A9") is None


@requires_terramind
def test_no_agreement_is_recorded_when_the_run_declares_the_reference_dtype(tmp_path):
    """There is no second number to compare, so measuring would be measuring fp32 against itself
    — and recording a 1.0 would read as a result rather than as an absence."""
    cfg = _terramind_config(tmp_path, TERRAMIND)  # compute_dtype fp32
    extractor = Extractor(load_spec(write_spec(cfg)), cfg)
    rng = np.random.default_rng(8)
    raw = rng.normal(1500.0, 400.0, size=(1, 224, 224)).astype(np.int16)
    extractor.extract(raw, land_mask(raw), arm="A5")
    assert extractor.precision == {}


@requires_terramind
def test_the_published_terramind_md5_is_the_file_on_this_box():
    assert artifacts.file_md5(TERRAMIND) == arms.TERRAMIND_TOKENIZER_DEM_MD5


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
