"""Encoder declarations per entry, plus B8's latent and the B4-ALT/B5-ALT entries (#204).

Spec #197's user stories 36, 37, 39 and 40, at the two seams that already carry them: the
registry (:mod:`benchmark.arms`) and ADR-0012's contract (:mod:`benchmark.extract`) — the file it
writes, the loader that re-checks it, and the extractor that loads an encoder strictly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmark import arms

TERRAMIND_FILE_ENTRIES = ("terramind", "terramind-latent", "terramind-per-tile")


def _kwargs(**moved) -> dict:
    """The smallest registrable encoder, with ``moved`` applied."""
    base = dict(
        name="probe",
        family="demae",
        width=768,
        pos_encoding="sincos2d",
        checkpoint_field="demae_r5_checkpoint",
        normalization="per_tile",
        modality=None,
        output_is_normed=True,
    )
    base.update(moved)
    return {k: v for k, v in base.items() if v is not _ABSENT}


_ABSENT = object()


# --- Story 36: declared per encoder, never defaulted ---------------------------


@pytest.mark.parametrize("field", ["output_is_normed", "modality", "normalization"])
def test_an_encoder_that_does_not_declare_its_diet_or_its_output_is_not_registrable(field):
    with pytest.raises(TypeError, match=field):
        arms.Encoder(**_kwargs(**{field: _ABSENT}))


@pytest.mark.parametrize(
    "moved, message",
    [
        # A global affine with no key has nothing to select its pair by — #138's three pairs.
        ({"normalization": "terramind_v1", "modality": None}, "selects"),
        # A per-tile diet with a key would print a constant pair nothing applies.
        ({"normalization": "per_tile", "modality": "tok_dem@224"}, "uses no constants"),
        # `none` is an arm with no encoder, never an encoder's own diet.
        ({"normalization": "none"}, "unknown normalization"),
        ({"normalization": "z_score"}, "unknown normalization"),
    ],
)
def test_a_diet_and_its_key_that_do_not_agree_are_refused(moved, message):
    with pytest.raises(ValueError, match=message):
        arms.Encoder(**_kwargs(**moved))


def test_an_arm_fed_a_diet_its_encoder_does_not_declare_is_refused():
    """ALT is a *new encoder*, not A5 with its diet moved — the registry makes that the only way."""
    with pytest.raises(ValueError, match="declares 'terramind_v1'"):
        arms.Arm(
            name="A5-PER-TILE",
            encoder="terramind",
            trains="head",
            input_convention="a1_fill_plus_mask",
            normalization="per_tile",
            what="A5 on DEMAE's diet, as a flag",
            printed=False,
        )


def test_every_registered_encoder_declares_its_own_diet_and_output():
    """The values, against the ADR rather than the code (ADR-0012 §3, §7.2; Amendment 2)."""
    expected = {
        "demae-r5": ("per_tile", None, True),
        "demae-r4": ("per_tile", None, True),
        "demae-vits": ("per_tile", None, True),
        "imagenet-mae": ("per_tile", None, True),
        "terramind": ("terramind_v1", "tok_dem@224", False),
        "terramind-bicubic": ("terramind_v1", "tok_dem@224", False),
        "terramind-tokenizer": ("terramind_v1", "tok_dem@224", False),
        "terramind-latent": ("terramind_v1", "tok_dem@224", False),
        "terramind-per-tile": ("per_tile", None, False),
    }
    for name, (diet, key, normed) in expected.items():
        encoder = arms.ENCODERS[name]
        assert (encoder.normalization, encoder.modality, encoder.output_is_normed) == (
            diet,
            key,
            normed,
        ), name
        assert arms.normalization_for(name) == diet, name


def test_no_module_constant_decides_the_key_or_the_relation():
    """The constant that used to select every TerraMind encoder's pair is gone (story 36)."""
    from benchmark import extract

    assert not hasattr(extract, "TERRAMIND_MODALITY")


def test_the_constants_are_selected_by_the_key_the_caller_declares(monkeypatch):
    """Two keys, two lookups: retuning the tokenizer's pair cannot move the backbone's.

    This is the accident story 36 names. Under one module constant, B9 would have read
    ``tok_dem@224`` — the same values today, and the wrong *name*, so a terratorch that retuned
    one pair would have moved both encoders and the refusal would have blamed the wrong one.
    """
    terramind_register = pytest.importorskip(
        "terratorch.models.backbones.terramind.model.terramind_register"
    )
    from benchmark.extract import terramind_constants

    assert terramind_constants("tok_dem@224") == (670.665, 951.272)
    assert terramind_constants("untok_dem@224") == (670.665, 951.272)

    monkeypatch.setitem(terramind_register.v1_pretraining_mean, "tok_dem@224", [651.663])
    with pytest.raises(ValueError, match="'tok_dem@224' to 651.663"):
        terramind_constants("tok_dem@224")
    assert terramind_constants("untok_dem@224") == (670.665, 951.272)


def test_a_global_affine_asked_for_with_no_key_is_refused():
    import numpy as np

    from benchmark.extract import normalize

    raw = np.full((1, 224, 224), 900, dtype=np.int16)
    with pytest.raises(ValueError, match="modality key"):
        normalize(raw, raw > 0, normalization="terramind_v1")


# --- Story 37: B8 and ALT, as entries of their own, through the strict load ------


def test_b8_and_alt_read_the_same_weights_as_b4_under_the_same_pin():
    """Same file, same md5, same crop — the plan's "same weights and md5 as `terramind`"."""
    base = arms.ENCODERS["terramind"]
    for name in ("terramind-latent", "terramind-per-tile"):
        entry = arms.ENCODERS[name]
        for attr in (
            "family", "width", "depth", "pos_encoding", "pos_emb_derivation",
            "pos_emb_native_grid_px", "checkpoint_field", "checkpoint_format", "checkpoint_md5",
            "params", "has_cls", "output_is_normed",
        ):
            assert getattr(entry, attr) == getattr(base, attr), (name, attr)
    # And each differs from B4 in exactly the thing it is the control for.
    latent, per_tile = arms.ENCODERS["terramind-latent"], arms.ENCODERS["terramind-per-tile"]
    assert (latent.output_tap, latent.output_width, latent.normalization) == (
        "pre_fsq_latent", 5, "terramind_v1",
    )
    assert (per_tile.output_tap, per_tile.output_width, per_tile.normalization) == (
        "final", 768, "per_tile",
    )
    # No Task 1 arm reads either: they are Task 2's, and Task 1's table does not move.
    assert not {a.encoder for a in arms.ARMS.values()} & {latent.name, per_tile.name}


def _terramind_box(tmp_path, checkpoint, monkeypatch):
    """A config mounting ``checkpoint`` as the tokenizer, every entry reading it republished."""
    import hashlib
    from dataclasses import replace

    from benchmark.config import BenchmarkConfig

    md5 = hashlib.md5(checkpoint.read_bytes()).hexdigest()
    for name, encoder in arms.ENCODERS.items():
        if encoder.checkpoint_field == "terramind_checkpoint":
            monkeypatch.setitem(arms.ENCODERS, name, replace(encoder, checkpoint_md5=md5))
    base = BenchmarkConfig()
    return base.replace(
        arm="A5",
        paths=base.paths.replace(
            artifacts_dir=str(tmp_path),
            demae_r5_checkpoint=None,
            demae_r4_checkpoint=None,
            terramind_checkpoint=str(checkpoint),
        ),
        extract=base.extract.replace(compute_dtype="fp32"),
    )


@pytest.fixture(scope="module")
def synthetic_tokenizer(tmp_path_factory):
    """A random-weight `vit_b_enc` plus `quant_proj`, saved the way the DEM tokenizer ships them.

    The same argument `test_benchmark_extract.synthetic_terramind` makes: what is under test is
    the composition — which tensor `output` is, which diet is applied, that the load is strict —
    and none of that depends on what the weights are.
    """
    torch = pytest.importorskip("torch")
    vit_models = pytest.importorskip(
        "terratorch.models.backbones.terramind.tokenizer.models.vit_models"
    )
    torch.manual_seed(0)
    model = vit_models.vit_b_enc(
        in_channels=1, patch_size=16, resolution=256, patch_proj=True, post_mlp=True
    )
    quant_proj = torch.nn.Conv2d(768, 5, 1)
    state = {f"encoder.{k}": v for k, v in model.state_dict().items()}
    state.update({f"quant_proj.{k}": v for k, v in quant_proj.state_dict().items()})
    path = tmp_path_factory.mktemp("tokenizer") / "TerraMind_Tokenizer_DEM.pt"
    torch.save(state, path)
    return path


def _tiles(seed=21, n=2):
    import numpy as np

    from dataset.normalize import land_mask

    raw = np.random.default_rng(seed).normal(1500.0, 400.0, size=(n, 224, 224)).astype(np.int16)
    return raw, land_mask(raw)


def test_b8s_output_is_the_pre_fsq_latent_of_b4s_block_twelve(
    tmp_path, synthetic_tokenizer, monkeypatch
):
    """What terratorch's own `return_type="latent"` returns: `quant_proj(embedding[-1])`."""
    import torch

    from benchmark.extract import Extractor, build_spec
    from pretrain.init import read_state_dict

    cfg = _terramind_box(tmp_path, synthetic_tokenizer, monkeypatch)
    extractor = Extractor(build_spec(cfg), cfg)
    raw, valid = _tiles()
    latent = extractor.extract(raw, valid, encoder_name="terramind-latent")
    feats = extractor.extract(raw, valid, arm="A5")

    assert latent.output.shape == (2, 196, 5)
    assert latent.cls is None
    # The blocks are B4's blocks: one set of weights, one diet, one crop.
    assert len(latent.intermediates) == 12
    for mine, theirs in zip(latent.intermediates, feats.intermediates):
        assert torch.equal(mine, theirs)

    state, _ = read_state_dict(synthetic_tokenizer)
    weight, bias = state["quant_proj.weight"], state["quant_proj.bias"]
    expected = feats.intermediates[-1] @ weight.squeeze(-1).squeeze(-1).T + bias
    assert torch.allclose(latent.output, expected, atol=1e-5)


def test_alt_is_b4s_encoder_on_demaes_per_tile_diet(tmp_path, synthetic_tokenizer, monkeypatch):
    """ALT's one difference from B4 is the input distribution, and it is really applied."""
    import torch

    from benchmark.extract import Extractor, build_spec, normalize

    cfg = _terramind_box(tmp_path, synthetic_tokenizer, monkeypatch)
    extractor = Extractor(build_spec(cfg), cfg)
    raw, valid = _tiles()
    alt = extractor.extract(raw, valid, encoder_name="terramind-per-tile")

    model = extractor.encoder(arm="A5")
    x = torch.from_numpy(normalize(raw, valid, normalization="per_tile")).unsqueeze(1)
    with torch.no_grad():
        spatial = model(x)
    assert torch.equal(alt.output, spatial.flatten(2).transpose(1, 2))
    assert not torch.allclose(alt.output, extractor.extract(raw, valid, arm="A5").output)


def test_b8_refuses_a_tokenizer_with_no_quant_proj_by_name(tmp_path, monkeypatch):
    """The strict load: an encoder-only file loads B4 and cannot be B8."""
    torch = pytest.importorskip("torch")
    vit_models = pytest.importorskip(
        "terratorch.models.backbones.terramind.tokenizer.models.vit_models"
    )
    from benchmark.extract import Extractor, build_spec

    native = vit_models.build_2d_sincos_posemb(16, 16, embed_dim=768)
    state = {f"encoder.blocks.{i}.norm1.weight": torch.zeros(768) for i in range(12)}
    state["encoder.pos_emb"] = native
    path = tmp_path / "encoder_only.pt"
    torch.save(state, path)
    cfg = _terramind_box(tmp_path, path, monkeypatch)
    with pytest.raises(ValueError, match="no `quant_proj.` tensors"):
        Extractor(build_spec(cfg), cfg).encoder(encoder_name="terramind-latent")


def test_b8_refuses_a_latent_of_another_width_by_name(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    vit_models = pytest.importorskip(
        "terratorch.models.backbones.terramind.tokenizer.models.vit_models"
    )
    from benchmark.extract import Extractor, build_spec

    state = {f"encoder.blocks.{i}.norm1.weight": torch.zeros(768) for i in range(12)}
    state["encoder.pos_emb"] = vit_models.build_2d_sincos_posemb(16, 16, embed_dim=768)
    state["quant_proj.weight"] = torch.zeros(8, 768, 1, 1)
    state["quant_proj.bias"] = torch.zeros(8)
    path = tmp_path / "wide_latent.pt"
    torch.save(state, path)
    cfg = _terramind_box(tmp_path, path, monkeypatch)
    with pytest.raises(ValueError, match="8-d latent.*declares 5"):
        Extractor(build_spec(cfg), cfg).encoder(encoder_name="terramind-latent")


# --- Story 39: the contract is re-frozen, and still refuses a drift ---------------


def _unmounted(tmp_path):
    from benchmark.config import BenchmarkConfig

    base = BenchmarkConfig()
    return base.replace(
        paths=base.paths.replace(
            artifacts_dir=str(tmp_path),
            demae_r5_checkpoint=None,
            demae_r4_checkpoint=None,
            demae_vits_checkpoint=None,
            imagenet_mae_checkpoint=None,
            terramind_checkpoint=None,
        )
    )


def test_the_committed_contract_carries_b8_and_alt_with_their_declarations():
    """The file Task 2 opens, not a description of it."""
    import json

    from benchmark.extract import load_spec

    committed = Path(__file__).resolve().parents[1] / "artifacts" / "task1" / "extractor_spec.json"
    spec = load_spec(committed)
    for name in TERRAMIND_FILE_ENTRIES:
        assert spec.encoders[name].checkpoint_md5 == arms.TERRAMIND_TOKENIZER_DEM_MD5
    body = json.loads(committed.read_text())["encoders"]
    assert body["terramind"]["modality"] == "tok_dem@224"
    assert body["terramind-per-tile"]["modality"] is None
    assert body["terramind-per-tile"]["normalization"] == "per_tile"
    assert (body["terramind-latent"]["output_tap"], body["terramind-latent"]["output_width"]) == (
        "pre_fsq_latent",
        5,
    )
    assert body["demae-r5"]["output_width"] == 768
    assert "pre-FSQ latent" in body["terramind-latent"]["note"]


@pytest.mark.parametrize(
    "encoder, moved",
    [
        ("terramind", {"modality": "untok_dem@224"}),
        ("terramind-latent", {"output_tap": "final"}),
        ("terramind-latent", {"output_width": 768}),
        ("terramind-per-tile", {"normalization": "terramind_v1"}),
    ],
)
def test_a_contract_whose_new_declarations_have_drifted_is_refused(tmp_path, encoder, moved):
    import json

    from benchmark.extract import build_spec, load_spec

    body = build_spec(_unmounted(tmp_path)).to_dict()
    body["encoders"][encoder].update(moved)
    path = tmp_path / "extractor_spec.json"
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match=f"{encoder}"):
        load_spec(path)


def test_a_head_over_b8_absorbs_the_latents_width():
    """`Arm.width` is `output`'s width — 5 for B8, unchanged for every Task 1 arm."""
    latent_arm = arms.Arm(
        name="B8-PROBE",
        encoder="terramind-latent",
        trains="head",
        input_convention="a1_fill_plus_mask",
        normalization="terramind_v1",
        what="a probe over the pre-FSQ latent",
        printed=False,
    )
    assert latent_arm.width == 5
    assert {arms.ARMS[name].width for name in ("A4-R5", "A5", "A6")} == {768}
    assert arms.ARMS["A4-S"].width == 384


# --- The real tokenizer, where a box has it ---------------------------------------


def _cached_tokenizer():
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    hit = try_to_load_from_cache(
        "ibm-esa-geospatial/TerraMind-1.0-Tokenizer-DEM", "TerraMind_Tokenizer_DEM.pt"
    )
    return Path(hit) if isinstance(hit, str) else None


TOKENIZER = _cached_tokenizer()


@pytest.mark.skipif(TOKENIZER is None, reason="the DEM tokenizer is not in the HF cache")
def test_b8_and_alt_load_strictly_from_the_pinned_tokenizer(tmp_path):
    """The shipped file under the shipped pin: B8's latent is terratorch's own, ALT's diet moves."""
    import torch

    from benchmark.config import BenchmarkConfig
    from benchmark.extract import Extractor, load_spec, write_spec

    base = BenchmarkConfig()
    cfg = base.replace(
        paths=base.paths.replace(
            artifacts_dir=str(tmp_path),
            demae_r5_checkpoint=None,
            demae_r4_checkpoint=None,
            terramind_checkpoint=str(TOKENIZER),
        ),
        extract=base.extract.replace(compute_dtype="fp32"),
    )
    extractor = Extractor(load_spec(write_spec(cfg)), cfg)
    raw, valid = _tiles(seed=31, n=1)
    latent = extractor.extract(raw, valid, encoder_name="terramind-latent")
    feats = extractor.extract(raw, valid, arm="A5")
    alt = extractor.extract(raw, valid, encoder_name="terramind-per-tile")

    assert latent.output.shape == (1, 196, 5)
    model = extractor.encoder(encoder_name="terramind-latent")
    with torch.no_grad():
        spatial = feats.intermediates[-1].transpose(1, 2).reshape(1, 768, 14, 14)
        expected = model.quant_proj(spatial).flatten(2).transpose(1, 2)
    assert torch.allclose(latent.output, expected, atol=1e-5)
    assert alt.output.shape == (1, 196, 768)
    assert not torch.allclose(alt.output, feats.output)


# --- Story 40: every count against plan §2's capacity table ----------------------

PLAN = Path(__file__).resolve().parents[1] / "docs" / "research" / "task2-landform-plan.md"

#: Which plan §2 row each registered encoder is — by the Task 2 arms the row lists.
PLAN_ROW_ARM = {
    "demae-r5": "B1",
    "demae-r4": "B1-R4",
    "imagenet-mae": "B6",
    "terramind": "B4",
    "terramind-latent": "B8",
    "terramind-per-tile": "B4-ALT",
    "demae-vits": "B1-S",
}


def _plan_capacity_rows() -> dict[str, int]:
    """Plan §2's "Capacity, in full" table, as arm -> parameter count, read off the plan."""
    text = PLAN.read_text()
    section = text.split("### Capacity, in full", 1)[1].split("\n\n", 2)[2]
    rows = {}
    for line in section.splitlines():
        if not line.startswith("|") or "---" in line or "Params" in line:
            if rows:
                break
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        count = int(cells[2].replace(",", ""))
        for arm in cells[1].split(","):
            rows[arm.strip()] = count
    return rows


def test_the_capacity_table_is_the_plans_own():
    """Quoted, and pinned against the text: a plan edit or a registry edit fails here."""
    plan = _plan_capacity_rows()
    assert sorted(set(plan.values())) == sorted(
        {85_254_144, 85_405_440, 90_172_416, 85_151_232, 21_393_408}
    )
    for name, arm in PLAN_ROW_ARM.items():
        assert arms.CAPACITY_TABLE[name] == plan[arm], name
    assert set(arms.CAPACITY_TABLE) == set(PLAN_ROW_ARM)


def test_every_registered_encoder_is_in_the_table_or_exempt_by_name():
    """Fails closed: an encoder registered later is refused until it is placed (B9, #205)."""
    for name in arms.ENCODERS:
        assert (name in arms.CAPACITY_TABLE) != (name in arms.CAPACITY_EXEMPT), name
        assert arms.check_capacity(name) == arms.ENCODERS[name].params


def test_a_count_that_disagrees_with_the_table_is_refused_by_name(monkeypatch):
    from dataclasses import replace

    monkeypatch.setitem(
        arms.ENCODERS, "terramind", replace(arms.ENCODERS["terramind"], params=90_126_336)
    )
    with pytest.raises(ValueError, match=r"terramind.*90,126,336.*90,172,416"):
        arms.check_capacity("terramind")


def test_an_encoder_the_table_does_not_place_is_refused(monkeypatch):
    from dataclasses import replace

    unplaced = replace(arms.ENCODERS["terramind"], name="terramind-backbone")
    monkeypatch.setitem(arms.ENCODERS, "terramind-backbone", unplaced)
    with pytest.raises(ValueError, match="terramind-backbone.*capacity table"):
        arms.check_capacity("terramind-backbone")


def test_the_contract_refuses_a_registry_whose_counts_have_drifted(tmp_path, monkeypatch):
    """Checked on every run: `load_spec` is the call every encoder arm makes."""
    from dataclasses import replace

    from benchmark.config import BenchmarkConfig
    from benchmark.extract import load_spec, write_spec

    base = BenchmarkConfig()
    cfg = base.replace(
        paths=base.paths.replace(
            artifacts_dir=str(tmp_path),
            demae_r5_checkpoint=None,
            demae_r4_checkpoint=None,
            demae_vits_checkpoint=None,
            imagenet_mae_checkpoint=None,
            terramind_checkpoint=None,
        )
    )
    written = write_spec(cfg)
    monkeypatch.setitem(
        arms.ENCODERS, "demae-r5", replace(arms.ENCODERS["demae-r5"], params=85_405_440)
    )
    with pytest.raises(ValueError, match="demae-r5.*85,254,144"):
        load_spec(written)
