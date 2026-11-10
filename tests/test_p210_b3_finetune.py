"""B3 under `run --task landform`: R5 unfrozen with §4.2's UperNet, §4.3's recipe as given (#210).

**Seam 1** is :func:`benchmark.launch.main`, on #206's fixture box with #208's UperNet line. A B3
cell fine-tunes a small trainable stand-in for the encoder (named the way timm names a ViT, so
the decay ladder has real layers to climb) together with the UperNet, and appends rows to the
same parquet. B3 is refused at any rung but 100%, under any recipe but §4.3's, and as a tuning
arm.

**Below the seam**, as the spec allows: the per-layer learning rates, which the command cannot
show. The expected values are written out as literals from §4.3's three numbers (lr 1e-4, 0.75
layer-wise decay, head lr 1e-3), not recomputed the way the code computes them.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from benchmark import landform_train  # noqa: E402
from benchmark.landform_config import Finetune  # noqa: E402


# --- Below the seam: §4.3's per-layer learning rates ------------------------------


def _vit(depth: int = 2):
    """A real timm ViT, tiny: the parameter names are what the ladder reads."""
    import timm

    return timm.models.vision_transformer.VisionTransformer(
        img_size=224, patch_size=16, in_chans=1, embed_dim=48, depth=depth, num_heads=2,
        num_classes=0, global_pool="",
    )


def test_layer_wise_decay_gives_each_encoder_layer_its_own_lr_and_the_head_1e_3():
    """Two blocks: layer 0 is the patch embedding and the CLS token, blocks 0 and 1 are layers 1
    and 2, and the top of MAE's ladder (3) is the final norm's. Layer l trains at
    1e-4 x 0.75^(3 - l). The norm itself is left frozen: it makes `output`, which UperNet never
    reads, so it would get no gradient (#197 review)."""
    encoder, head = _vit(depth=2), torch.nn.Conv2d(48, 10, 1)
    landform_train.unfreeze(encoder)
    groups = landform_train.finetune_groups(encoder, head, Finetune(), weight_decay=0.05)
    lr_of = {id(p): g["lr"] for g in groups for p in g["params"]}
    by_name = {n: lr_of.get(id(p)) for n, p in encoder.named_parameters()}

    assert by_name["patch_embed.proj.weight"] == pytest.approx(4.21875e-05)
    assert by_name["cls_token"] == pytest.approx(4.21875e-05)
    assert by_name["blocks.0.attn.qkv.weight"] == pytest.approx(5.625e-05)
    assert by_name["blocks.0.mlp.fc2.bias"] == pytest.approx(5.625e-05)
    assert by_name["blocks.1.attn.proj.weight"] == pytest.approx(7.5e-05)
    assert by_name["norm.weight"] is None and by_name["norm.bias"] is None
    assert all(lr_of[id(p)] == pytest.approx(1e-03) for p in head.parameters())


def test_the_12_block_ladder_runs_from_7_5e_5_at_block_12_to_2_4e_6_at_the_patch_embedding():
    """MAE's ladder keeps its top (13, 1e-4) at the final norm, which B3 leaves frozen, so the
    highest layer that trains is block 12, one decay below."""
    encoder = _vit(depth=12)
    landform_train.unfreeze(encoder)
    groups = landform_train.finetune_groups(encoder, torch.nn.Identity(), Finetune(),
                                            weight_decay=0.05)
    by_layer = {g["layer"]: g["lr"] for g in groups if g["part"] == "encoder"}
    assert sorted(by_layer) == list(range(13))
    assert by_layer[12] == pytest.approx(7.5e-5)        # block 12 of 12
    assert by_layer[1] == pytest.approx(3.1676352e-06)  # block 1 of 12
    assert by_layer[0] == pytest.approx(2.3757264e-06)  # patch embedding, CLS


def test_the_sin_cos_position_embedding_stays_fixed_and_biases_and_norms_do_not_decay():
    """ADR-0011 verifies R5's `pos_embed` as sin-cos at load: it was never a learned weight, so
    the fine-tune does not learn it either. Weight decay skips 1-d tensors and the CLS token."""
    encoder, head = _vit(depth=2), torch.nn.Conv2d(48, 10, 1)
    groups = landform_train.finetune_groups(encoder, head, Finetune(), weight_decay=0.05)
    trained = {id(p) for g in groups for p in g["params"]}
    assert id(encoder.pos_embed) not in trained
    wd_of = {id(p): g["weight_decay"] for g in groups for p in g["params"]}
    assert wd_of[id(encoder.blocks[0].attn.qkv.weight)] == 0.05
    assert wd_of[id(head.weight)] == 0.05
    assert wd_of[id(encoder.blocks[0].norm1.weight)] == 0.0
    assert wd_of[id(encoder.cls_token)] == 0.0
    assert wd_of[id(head.bias)] == 0.0


# --- Seam 1: the refusals ---------------------------------------------------------

from benchmark import launch  # noqa: E402


@pytest.mark.parametrize("rung", ["1%", "10%"])
def test_b3_is_refused_at_any_rung_but_100_percent(rung, capsys):
    """§4.3: 100% labels only. Refused at resolution, so `--dry-run` cannot print it either."""
    for dry in (["--dry-run"], []):
        with pytest.raises(ValueError, match=rf"arm B3 runs at \['100%'\] only.*rung {rung}"):
            launch.main(["run", "--task", "landform", "--arm", "B3", "--rung", rung, *dry])
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("knob, value", [
    ("finetune.lr", "3e-4"),
    ("finetune.layer_decay", "0.65"),
    ("finetune.head_lr", "1e-4"),
    ("finetune.epochs", "30"),
    ("finetune.precision", '"fp32"'),
])
def test_a_b3_cell_under_any_recipe_but_section_4_3_s_is_refused(knob, value, capsys):
    """The literature recipe is used as given (#196): no knob of it moves, by `--set` or file."""
    for dry in (["--dry-run"], []):
        with pytest.raises(ValueError, match=rf"B3 trains under §4\.3's literature recipe.*{knob}"):
            launch.main(["run", "--task", "landform", "--arm", "B3", *dry,
                         "--set", f"{knob}={value}"])
    assert capsys.readouterr().out == ""


def test_b3_is_refused_as_a_tuning_arm(capsys):
    for argv in (["--arm", "B3"], ["--arm", "B3", "--head", "upernet"]):
        with pytest.raises(ValueError, match=r"B3 is not tuned.*§4\.3.*#196"):
            launch.main(["run", "--task", "landform", "--tune", "--dry-run", *argv])
    assert capsys.readouterr().out == ""


# --- Seam 1: a B3 cell end to end ---------------------------------------------------

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from benchmark import arms, landform_recipes, results, upernet  # noqa: E402
from benchmark.landform_config import build_config  # noqa: E402
from tests.test_p206_linear_probe_cell import (  # noqa: E402,F401  (fixtures)
    TILES_PER_COG,
    StubExtractor,
    _flags,
    box,
    corpus,
)

#: §4.2's UperNet at test scale, frozen as B6's line at 100%: batch 8 over the fixture's 8-tile
#: floor is one step an epoch, so §4.3's 50 epochs are 50 steps. 32 channels keep it cheap.
UPERNET_TEST = [
    "--set", "upernet.batch_size=8",
    "--set", "upernet.epochs=5",
    "--set", "upernet.channels=32",
]
B3_STEPS = 50
WIDTH = arms.ENCODERS["demae-r5"].width


class _Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(WIDTH)
        self.scale = torch.nn.Parameter(torch.ones(WIDTH))

    def forward(self, x):
        return x + self.norm1(x) * self.scale


class _TrainableEncoder(torch.nn.Module):
    """Named as timm names a ViT — patch embedding, CLS token, sin-cos position, twelve blocks,
    final norm — and cheap: each block is a LayerNorm and a per-channel scale."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.patch_embed = torch.nn.Conv2d(1, WIDTH, 16, 16)
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, WIDTH))
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, 197, WIDTH))
        self.blocks = torch.nn.ModuleList(_Block() for _ in range(12))
        self.norm = torch.nn.LayerNorm(WIDTH)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(len(x), -1, -1), x], 1) + self.pos_embed
        intermediates = []
        for block in self.blocks:
            x = block(x)
            intermediates.append(x)
        return self.norm(x), intermediates


class TrainableExtractor(StubExtractor):
    """ADR-0012's surface over :class:`_TrainableEncoder`, honouring ``grad`` as the real
    extractor does: no graph unless the caller asks for one."""

    instances: list = []
    grads: list = []

    def __init__(self, spec=None, cfg=None, *, device: str = "cpu") -> None:
        super().__init__(spec, cfg, device=device)
        self._model = _TrainableEncoder()
        TrainableExtractor.instances.append(self)

    def encoder(self, arm=None, *, encoder_name=None):
        return self._model

    def extract(self, raw_i16, valid_mask, *, arm=None, encoder_name=None, grad=False):
        from benchmark import extract as extract_module

        TrainableExtractor.grads.append(grad)
        raw, mask = extract_module._as_batch(raw_i16, valid_mask)
        x = torch.from_numpy(extract_module.normalize(raw, mask, normalization="per_tile"))
        with torch.set_grad_enabled(grad):
            tokens, intermediates = self._model(x.unsqueeze(1))
        return extract_module.Features(
            output=tokens[:, 1:], intermediates=tuple(b[:, 1:] for b in intermediates), cls=None
        )


@pytest.fixture
def trainable(monkeypatch):
    from benchmark import extract as extract_module

    TrainableExtractor.instances, TrainableExtractor.grads = [], []
    monkeypatch.setattr(extract_module, "Extractor", TrainableExtractor)
    return TrainableExtractor


@pytest.fixture
def flags(box, corpus):
    """#206's box with B6's UperNet line at 100%, the decoder B3 must share."""
    argv = [*_flags(box, corpus), *UPERNET_TEST]
    cfg = build_config(arm="B6", head="upernet", overrides=argv[2:][1::2])
    landform_recipes.write_line(cfg, lr=3e-4, scores={"0.0003": 0.5})
    return argv


def b3(*argv: str) -> list[str]:
    return ["run", "--arm", "B3", "--seed", "0", *argv]


def records_of(box) -> list[dict]:
    return results.read_run_records(results.run_record_path(box / "results.parquet"))


def test_a_b3_cell_fine_tunes_encoder_and_upernet_and_appends_rows(box, flags, trainable,
                                                                    capsys):
    import copy

    start = copy.deepcopy(_TrainableEncoder().state_dict())
    assert launch.main([*b3(), *flags]) == 0
    out = capsys.readouterr().out
    assert ("encoder 0.0001 with 0.75 layer-wise decay, head lr 0.001, 50 epochs, bf16 — "
            "§4.3's literature recipe, used as given and not tuned") in out
    assert f"{B3_STEPS} steps = 50 epochs" in out

    rows = pq.read_table(box / "results.parquet").to_pylist()
    assert len(rows) == 2 * TILES_PER_COG, "one row per test tile"
    assert {(r["arm"], r["head"], r["rung"]) for r in rows} == {("B3", "upernet", "100%")}
    assert all(np.asarray(r["confusion"]).shape == (100,) for r in rows)

    (record,) = records_of(box)
    assert record["cell"]["arm"] == "B3"
    assert record["training"]["steps"] == B3_STEPS
    assert record["training"]["kind"] == "finetune"
    assert record["finetune"] == {"lr": 1e-4, "layer_decay": 0.75, "head_lr": 1e-3,
                                  "epochs": 50, "precision": "bf16"}
    assert record["training"]["precision_declared"] == "bf16"
    assert record["training"]["precision_used"] == "fp32", "CPU drops bf16 and says so"

    model = trainable.instances[-1]._model
    every_but_sin_cos = sum(p.numel() for n, p in model.named_parameters() if n != "pos_embed")
    # The final norm makes `output`, which UperNet never reads (it reads raw intermediates[3, 5,
    # 7, 11]): it gets no gradient, so it is left frozen and not counted. LayerNorm(768): 1,536.
    encoder_trained = every_but_sin_cos - 1_536
    head = upernet.parameter_count(WIDTH, channels=32)
    assert record["parameters"] == {
        "encoder": 85_254_144, "trainable": encoder_trained + head, "total": 85_254_144 + head,
    }, "the encoder's weights train, all but its fixed sin-cos position embedding and final norm"
    assert torch.equal(model.state_dict()["norm.weight"], start["norm.weight"])

    # The encoder that scored is the fine-tuned one: its weights moved, its sin-cos did not.
    now = model.state_dict()
    assert not torch.equal(now["blocks.0.scale"], start["blocks.0.scale"])
    assert not torch.equal(now["patch_embed.weight"], start["patch_embed.weight"])
    assert torch.equal(now["pos_embed"], start["pos_embed"])
    assert True in trainable.grads and False in trainable.grads, \
        "a graph through the encoder in training, none in val and test"


def test_a_b3_cell_whose_decoder_drifts_from_the_upernet_line_is_refused(box, flags, trainable):
    """'Same decoder, same layers' (§4.3) is B6's frozen UperNet line at 100%, not a copy."""
    with pytest.raises(ValueError, match="recipe drifted from its frozen line.*upernet.channels"):
        launch.main([*b3("--dry-run"), *flags, "--set", "upernet.channels=64"])


def test_the_optimizer_steps_each_layer_at_its_own_lr_under_the_one_cosine(
    box, flags, trainable, monkeypatch
):
    """#197 review: the run loop's per-group lr, as the optimizer sees it on each step. Every
    group peaks at its own §4.3 lr (literals, as above) and all groups share one cosine factor,
    so the ratio between any two groups is the same on every step."""
    seen: list[dict] = []

    class SpyAdamW(torch.optim.AdamW):
        def step(self, closure=None):
            seen.append({(g["part"], g["layer"]): g["lr"] for g in self.param_groups})
            return super().step(closure)

    monkeypatch.setattr(torch.optim, "AdamW", SpyAdamW)
    assert launch.main([*b3(), *flags]) == 0

    assert len(seen) > 1
    peak = {key: max(step[key] for step in seen) for key in seen[0]}
    head = [key for key in peak if key[0] == "head"]
    assert len(head) == 1 and peak[head[0]] == pytest.approx(1e-3)
    assert peak[("encoder", 12)] == pytest.approx(7.5e-05)         # block 12 of 12
    assert peak[("encoder", 1)] == pytest.approx(3.1676352e-06)    # block 1 of 12
    assert peak[("encoder", 0)] == pytest.approx(2.3757264e-06)    # patch embedding, CLS
    assert ("encoder", 13) not in peak, "the unread final norm has no group"
    for step in seen:
        factor = step[head[0]] / 1e-3
        assert factor <= 1.0 + 1e-12
        for key, lr in step.items():
            assert lr == pytest.approx(peak[key] * factor, rel=1e-9, abs=1e-15), key
