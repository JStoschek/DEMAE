"""B9, `terramind-backbone`: registered with its pin, loaded strictly, forward hooked (#205).

Spec #197's user stories 33-35 and ADR-0012 Amendment 2, at the seams that already carry them:
the registry (:mod:`benchmark.arms`) and ADR-0012's contract (:mod:`benchmark.extract`) — the
committed file, the loader that re-checks it, and the extractor that builds an encoder.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmark import arms

B9 = "terramind-backbone"

#: Amendment 2's identity, quoted from the ADR rather than from the registry under test.
B9_MD5 = "690349bd1328e50b64979847d4f362e7"
B9_REPO = "ibm-esa-geospatial/TerraMind-1.0-base"
B9_FILE = "TerraMind_v1_base.pt"
B9_REVISION = "fb96c70d0a5f68dcc44030b89cbfd8ec3fb0c67a"
#: Plan §2's capacity table, B9's row, and #194's `parameters()`-as-built count.
B9_PARAMS = 85_151_232


# --- Story 33: registered, with its pin -------------------------------------------


def test_b9_is_registered_under_amendment_2s_pin_and_declarations():
    encoder = arms.ENCODERS[B9]
    assert encoder.checkpoint_md5 == B9_MD5
    assert (encoder.family, encoder.width, encoder.depth) == ("terramind", 768, 12)
    # D1/D7.2: the tokenizer's pair, selected by the backbone's own key.
    assert (encoder.normalization, encoder.modality) == ("terramind_v1", "untok_dem@224")
    # D3: `output` is `encoder_norm(raw block 12)`.
    assert encoder.output_is_normed is True
    assert encoder.output_tap == "final"
    # D5: no CLS. D7.4: the native 14x14 sin-cos table, no ADR-0011 derivation.
    assert encoder.has_cls is False
    assert encoder.pos_emb_derivation == "as_published"
    assert encoder.pos_emb_native_grid_px is None
    # Its own mount, not the tokenizer's: they are two different files.
    assert encoder.checkpoint_field == "terramind_backbone_checkpoint"
    assert encoder.checkpoint_field != arms.ENCODERS["terramind"].checkpoint_field
    # No Task 1 arm reads it: Task 1's table does not move.
    assert B9 not in {arm.encoder for arm in arms.ARMS.values()}


def test_b9s_count_is_plan_section_2s():
    assert arms.ENCODERS[B9].params == B9_PARAMS
    assert arms.CAPACITY_TABLE[B9] == B9_PARAMS
    assert arms.check_capacity(B9) == B9_PARAMS


def test_only_task_twos_paths_group_mounts_b9():
    """Task 1's config is out of scope (#197): B9's mount is Task 2's alone."""
    from benchmark.config import Paths
    from benchmark.landform_config import LandformPaths

    assert not hasattr(Paths(), "terramind_backbone_checkpoint")
    assert LandformPaths().terramind_backbone_checkpoint is None


def test_task_ones_freeze_builds_the_same_contract_without_b9s_mount(tmp_path):
    """The contract's bytes are a function of the code, whichever task's config builds it."""
    from benchmark.config import BenchmarkConfig
    from benchmark.extract import build_spec

    task1 = BenchmarkConfig()
    task1 = task1.replace(paths=task1.paths.replace(artifacts_dir=str(tmp_path)))
    assert build_spec(task1).to_json() == build_spec(_unmounted(tmp_path)).to_json()


# --- Story 39 for B9: the contract carries it --------------------------------------


def _unmounted(tmp_path):
    from benchmark.landform_config import LandformConfig

    base = LandformConfig()
    return base.replace(
        paths=base.paths.replace(
            artifacts_dir=str(tmp_path),
            demae_r5_checkpoint=None,
            demae_r4_checkpoint=None,
            demae_vits_checkpoint=None,
            imagenet_mae_checkpoint=None,
            terramind_checkpoint=None,
            terramind_backbone_checkpoint=None,
        )
    )


def test_the_contract_declares_b9_on_its_native_grid_and_its_own_key(tmp_path):
    from benchmark.extract import build_spec

    entry = build_spec(_unmounted(tmp_path)).encoder_named(B9)
    assert entry.checkpoint_md5 == B9_MD5
    assert (entry.normalization, entry.modality) == ("terramind_v1", "untok_dem@224")
    assert entry.output_is_normed is True
    assert (entry.output_tap, entry.output_width, entry.has_cls) == ("final", 768, False)
    # As published, at the run's own 14x14 grid: no ADR-0011 block, no anchor.
    assert entry.pos_emb == {"derivation": "as_published", "grid_px": 14, "native_grid_px": 14}
    assert "**not** `intermediates[11]`" in entry.note


@pytest.mark.parametrize(
    "moved",
    [
        {"modality": "tok_dem@224"},
        {"output_is_normed": False},
        {"checkpoint_md5": "66b7fbcb81e04684df31873623d7a3de"},
        {
            "pos_emb": {
                "derivation": "rederived_14x14_crop",
                "grid_px": 14,
                "native_grid_px": 16,
                "adr": "ADR-0011",
                "anchor": [1, 1],
            }
        },
    ],
)
def test_a_contract_whose_b9_clause_has_drifted_is_refused(tmp_path, moved):
    import json

    from benchmark.extract import build_spec, load_spec

    body = build_spec(_unmounted(tmp_path)).to_dict()
    body["encoders"][B9].update(moved)
    path = tmp_path / "extractor_spec.json"
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match=B9):
        load_spec(path)


def test_the_committed_contract_carries_b9():
    """The re-frozen file Task 2 opens (story 39), not a description of it."""
    from benchmark.extract import load_spec

    committed = Path(__file__).resolve().parents[1] / "artifacts" / "task1" / "extractor_spec.json"
    entry = load_spec(committed).encoder_named(B9)
    assert (entry.checkpoint_md5, entry.modality) == (B9_MD5, "untok_dem@224")


# --- Story 33: refused until the pin checks -----------------------------------------


def _mounted(tmp_path, checkpoint):
    return _unmounted(tmp_path).replace(
        paths=_unmounted(tmp_path).paths.replace(terramind_backbone_checkpoint=str(checkpoint)),
    )


def test_an_unmounted_b9_is_refused_by_its_field(tmp_path):
    from benchmark.extract import Extractor, build_spec

    cfg = _unmounted(tmp_path)
    with pytest.raises(FileNotFoundError, match="paths.terramind_backbone_checkpoint"):
        Extractor(build_spec(cfg), cfg).encoder(encoder_name=B9)


def test_a_b9_file_that_is_not_the_pinned_one_is_refused_at_load(tmp_path):
    from benchmark.extract import Extractor, build_spec

    impostor = tmp_path / "TerraMind_v1_base.pt"
    impostor.write_bytes(b"not IBM's file")
    spec = build_spec(_unmounted(tmp_path))
    with pytest.raises(ValueError, match=f"md5.*declares {B9} as {B9_MD5}"):
        Extractor(spec, _mounted(tmp_path, impostor)).encoder(encoder_name=B9)


def test_freeze_refuses_to_pin_b9_against_a_file_that_is_not_it(tmp_path):
    from benchmark.extract import build_spec

    impostor = tmp_path / "TerraMind_v1_base.pt"
    impostor.write_bytes(b"not IBM's file")
    with pytest.raises(ValueError, match="terramind_backbone_checkpoint.*weights nobody ran"):
        build_spec(_mounted(tmp_path, impostor))


# --- Stories 34-35: our own strict load, native grid, hooked forward -----------------


def _republished(monkeypatch, checkpoint):
    """Pin B9 at ``checkpoint``'s md5, so a synthetic file passes identity and reaches the load."""
    import hashlib
    from dataclasses import replace

    md5 = hashlib.md5(checkpoint.read_bytes()).hexdigest()
    monkeypatch.setitem(arms.ENCODERS, B9, replace(arms.ENCODERS[B9], checkpoint_md5=md5))


@pytest.fixture(scope="module")
def synthetic_backbone(tmp_path_factory):
    """A random-weight `terramind_v1_base` (DEM only), saved inside a larger any-to-any file.

    What is under test is the composition — which keys are loaded, which tensor each field is,
    which diet is applied — and none of it depends on what the weights are. The two extra keys
    stand in for the 289 tensors of other modalities and the decoder that the real file carries.
    """
    torch = pytest.importorskip("torch")
    register = pytest.importorskip("terratorch.models.backbones.terramind.model.terramind_register")
    torch.manual_seed(0)
    model = register.terramind_v1_base(pretrained=False, modalities=["DEM"], merge_method=None)
    state = dict(model.state_dict())
    state["encoder_embeddings.untok_s2l2a@224.proj.weight"] = torch.zeros(768, 16 * 16 * 12)
    state["decoder.0.mlp.fc1.weight"] = torch.zeros(4, 4)
    path = tmp_path_factory.mktemp("backbone") / "TerraMind_v1_base.pt"
    torch.save(state, path)
    return path


def _tiles(seed=21, n=2):
    import numpy as np

    from dataset.normalize import land_mask

    raw = np.random.default_rng(seed).normal(1500.0, 400.0, size=(n, 224, 224)).astype(np.int16)
    return raw, land_mask(raw)


def _untok_diet(raw):
    """ADR-0012 Amendment 2's D1, written from the ADR's numbers rather than the code's lookup."""
    import numpy as np

    return ((raw.astype(np.float32) - np.float32(670.665)) / np.float32(951.272)).astype(np.float32)


@pytest.fixture
def b9(tmp_path, synthetic_backbone, monkeypatch):
    """An extractor mounting the synthetic backbone under a republished pin."""
    from benchmark.extract import Extractor, build_spec

    _republished(monkeypatch, synthetic_backbone)
    cfg = _mounted(tmp_path, synthetic_backbone)
    cfg = cfg.replace(extract=cfg.extract.replace(compute_dtype="fp32"))
    return Extractor(build_spec(cfg), cfg)


def test_b9s_intermediates_11_is_the_raw_block_12_and_its_output_the_normed_one(b9):
    """The numeric check the spec allows below the seam: the hooked forward (story 35)."""
    import torch

    raw, valid = _tiles()
    feats = b9.extract(raw, valid, encoder_name=B9)

    assert feats.output.shape == (2, 196, 768)
    assert feats.cls is None
    assert len(feats.intermediates) == 12
    assert all(block.shape == (2, 196, 768) for block in feats.intermediates)
    assert not torch.allclose(feats.intermediates[11], feats.output)
    model = b9.encoder(encoder_name=B9)
    with torch.no_grad():
        assert torch.equal(feats.output, model.encoder_norm(feats.intermediates[11]))


def test_b9s_unrolled_forward_is_terratorchs_list_but_for_its_last_entry(b9):
    """Amendment 2's owed test: entries 1-11 bit-identical, and `output` is entry 12.

    terratorch's input is built here from the ADR's constants, so this is also the diet test:
    the extractor fed the backbone `(m − 670.665) / 951.272` and nothing else.
    """
    import torch

    raw, valid = _tiles()
    feats = b9.extract(raw, valid, encoder_name=B9)
    model = b9.encoder(encoder_name=B9)
    with torch.no_grad():
        returned = model(torch.from_numpy(_untok_diet(raw)).unsqueeze(1))

    assert len(returned) == 12
    for i in range(11):
        assert torch.equal(feats.intermediates[i], returned[i]), i
    assert torch.equal(feats.output, returned[11])
    # And the reason there is a forward of our own: the list's last entry is not block 12.
    assert not torch.allclose(feats.intermediates[11], returned[11])


def test_b9_runs_on_its_native_grid_with_the_files_own_position_table(b9, synthetic_backbone):
    """D7.4: `as_published` — the shipped 14x14 buffer, unchanged, and nothing derived."""
    import torch

    state = torch.load(synthetic_backbone, map_location="cpu", weights_only=True)
    model = b9.encoder(encoder_name=B9)
    assert list(model.encoder_embeddings) == ["untok_dem@224"]
    pos_emb = model.encoder_embeddings["untok_dem@224"].pos_emb
    assert tuple(pos_emb.shape) == (1, 196, 768)
    assert torch.equal(pos_emb, state["encoder_embeddings.untok_dem@224.pos_emb"])


def test_b9_loads_every_model_tensor_from_the_file(b9, synthetic_backbone):
    """Amendment 2's other owed test, and the parameter count against §2's table."""
    import torch

    state = torch.load(synthetic_backbone, map_location="cpu", weights_only=True)
    model = b9.encoder(encoder_name=B9)
    loaded = model.state_dict()
    assert all(torch.equal(tensor, state[key]) for key, tensor in loaded.items())
    assert sum(p.numel() for p in model.parameters()) == B9_PARAMS


def test_b9s_diet_is_looked_up_under_its_own_key(b9, monkeypatch):
    """Retuning the backbone's pair refuses B9 by name, at load, before a tile arrives."""
    register = pytest.importorskip("terratorch.models.backbones.terramind.model.terramind_register")
    monkeypatch.setitem(register.v1_pretraining_mean, "untok_dem@224", [651.663])
    with pytest.raises(ValueError, match="'untok_dem@224' to 651.663"):
        b9.encoder(encoder_name=B9)


def test_a_b9_file_missing_a_model_tensor_is_refused_by_name(tmp_path, monkeypatch):
    """Our own strict load, never terratorch's filter, which keeps the random init and warns."""
    torch = pytest.importorskip("torch")
    from benchmark.extract import Extractor, build_spec

    state = {f"encoder.{i}.norm1.weight": torch.ones(768) for i in range(12)}
    path = tmp_path / "TerraMind_v1_base.pt"
    torch.save(state, path)
    _republished(monkeypatch, path)
    cfg = _mounted(tmp_path, path)
    with pytest.raises(ValueError, match=r"missing \d+ of the \d+ tensors.*'encoder\.0\.attn\."):
        Extractor(build_spec(cfg), cfg).encoder(encoder_name=B9)


def test_a_b9_file_of_another_depth_is_refused_by_name(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from benchmark.extract import Extractor, build_spec

    state = {f"encoder.{i}.norm1.weight": torch.ones(768) for i in range(11)}
    path = tmp_path / "TerraMind_v1_base.pt"
    torch.save(state, path)
    _republished(monkeypatch, path)
    cfg = _mounted(tmp_path, path)
    with pytest.raises(ValueError, match="11 transformer blocks.*declares 12"):
        Extractor(build_spec(cfg), cfg).encoder(encoder_name=B9)


def test_a_built_b9_whose_count_is_not_the_registrys_is_refused(b9, monkeypatch):
    """A terratorch that grew the architecture would load strictly and miscount; this catches it."""
    from dataclasses import replace

    monkeypatch.setitem(arms.ENCODERS, B9, replace(arms.ENCODERS[B9], params=85_254_144))
    with pytest.raises(ValueError, match=r"85,151,232 parameters.*85,254,144"):
        b9.encoder(encoder_name=B9)


# --- The real backbone, where a box has it -----------------------------------------


def _cached_backbone():
    """The pinned revision's file in the HF cache, offline — a different snapshot skips."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    hit = try_to_load_from_cache(B9_REPO, B9_FILE, revision=B9_REVISION)
    return Path(hit) if isinstance(hit, str) else None


BACKBONE = _cached_backbone()


@pytest.mark.skipif(BACKBONE is None, reason="TerraMind_v1_base.pt is not in the HF cache")
def test_b9_loads_strictly_from_the_pinned_file_and_hooks_its_forward(tmp_path):
    """The shipped file under the shipped pin: every tensor its own, block 12 raw, `output` normed."""
    import torch

    from benchmark.extract import Extractor, load_spec, write_spec

    cfg = _mounted(tmp_path, BACKBONE)
    cfg = cfg.replace(extract=cfg.extract.replace(compute_dtype="fp32"))
    extractor = Extractor(load_spec(write_spec(cfg)), cfg)
    model = extractor.encoder(encoder_name=B9)

    state = torch.load(BACKBONE, map_location="cpu", weights_only=True, mmap=True)
    assert all(torch.equal(tensor, state[key]) for key, tensor in model.state_dict().items())
    assert sum(p.numel() for p in model.parameters()) == B9_PARAMS
    del state

    raw, valid = _tiles(seed=41, n=1)
    feats = extractor.extract(raw, valid, encoder_name=B9)
    with torch.no_grad():
        returned = model(torch.from_numpy(_untok_diet(raw)).unsqueeze(1))
    for i in range(11):
        assert torch.equal(feats.intermediates[i], returned[i]), i
    assert torch.equal(feats.output, returned[11])
    # #194 measured the two at mean cosine 0.958 over real tiles: different tensors by design.
    cosine = torch.nn.functional.cosine_similarity(feats.output, feats.intermediates[11], dim=-1)
    assert float(cosine.mean()) < 0.99
