"""Plan §7's kernel levers, as things that are checked rather than configured (#95).

Every claim here is one a loss curve cannot make. A model whose attention fell back to the
unfused path trains identically and slower; a `torch.compile` that silently recompiles every
step trains identically and slower still, which is the whole of what §7.3's 1.1-1.4x was
supposed to buy. So the tests ask what the *modules actually do*, not what the config asked
for.

Compiling on CPU is expensive — seconds for even the 2-block fixture model — so the compiled
tests here are few and deliberate: the wrapper, the recompiles, and the eager fallback.
Everything else runs eager. That compiling does not move the *loss* is asserted where the
losses are, in `test_pretrain_train.py`.
"""

from __future__ import annotations

import pytest
import torch

from pretrain import runtime as runtime_mod
from pretrain.config import Architecture, Data, Optimization, RunConfig, Runtime, Schedule
from pretrain.model import MaskedAutoencoder


def tiny_arch() -> Architecture:
    """A 2-block encoder and a 1-block decoder — the smallest thing with both halves."""
    return Architecture(
        encoder_dim=64, encoder_depth=2, encoder_heads=2,
        decoder_dim=64, decoder_depth=1, decoder_heads=2,
    )


def tiny_config(**changes) -> RunConfig:
    """The fixture recipe, compile off by default: turning it on is what a test opts into."""
    cfg = RunConfig(
        run_id="test",
        architecture=tiny_arch(),
        optimization=Optimization(batch_size=4, accum_iter=1, blr=3e-3, lr_batch_base=4),
        schedule=Schedule(epochs=150, warmup_epochs=6, milestones=(150,)),
        data=Data(num_workers=0),
        runtime=Runtime(compile=False),
    )
    return cfg.replace(**changes) if changes else cfg


def tiles(n: int = 2, *, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 1, 224, 224, generator=g)


# --- §7.2: SDPA is the attention path in use ----------------------------------


def test_the_attention_path_actually_calls_scaled_dot_product_attention(monkeypatch):
    # §7.2 chose `F.scaled_dot_product_attention` because that is what timm's Block routes
    # through. That is a claim about someone else's library, so it is checked against the
    # function itself rather than against timm's flag.
    calls = []
    real = torch.nn.functional.scaled_dot_product_attention

    def spy(*args, **kwargs):
        calls.append(kwargs.get("dropout_p", 0.0))
        return real(*args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", spy)
    model = MaskedAutoencoder.from_config(tiny_config())
    model(tiles())
    # Once per block, encoder and decoder: 2 + 1.
    assert len(calls) == 3


def test_the_math_escape_hatch_does_not_reach_sdpa_at_all(monkeypatch):
    # §7.2's debugging fallback has to be a real fallback: if it still went through SDPA there
    # would be nothing to compare a suspected kernel bug against.
    calls = []
    real = torch.nn.functional.scaled_dot_product_attention
    monkeypatch.setattr(
        torch.nn.functional,
        "scaled_dot_product_attention",
        lambda *a, **k: (calls.append(1), real(*a, **k))[1],
    )
    math_cfg = tiny_config(runtime=Runtime(compile=False, attention="math"))
    model = MaskedAutoencoder.from_config(math_cfg)
    model(tiles())
    assert calls == []


def test_the_attention_path_is_readable_off_a_built_model():
    # What a run reports in its header has to come from the modules, not from the config it was
    # asked for — those are exactly the two things that can disagree.
    assert runtime_mod.attention_path(MaskedAutoencoder.from_config(tiny_config())) == "sdpa"
    math_cfg = tiny_config(runtime=Runtime(compile=False, attention="math"))
    assert runtime_mod.attention_path(MaskedAutoencoder.from_config(math_cfg)) == "math"


def test_a_model_whose_blocks_disagree_is_reported_as_mixed_rather_than_averaged():
    # Half-fused attention is not a state §7.2 has an opinion about, and it is reachable by
    # hand. Naming it beats reporting whichever half was inspected first.
    model = MaskedAutoencoder.from_config(tiny_config())
    model.blocks[0].attn.fused_attn = False
    assert runtime_mod.attention_path(model) == "mixed"


def test_verify_attention_refuses_a_model_that_does_not_match_its_config():
    cfg = tiny_config()
    model = MaskedAutoencoder.from_config(cfg)
    assert runtime_mod.verify_attention(model, cfg) == "sdpa"
    model.blocks[0].attn.fused_attn = False
    with pytest.raises(ValueError, match="sdpa"):
        runtime_mod.verify_attention(model, cfg)


# --- §7.3: compile, and the recompiles that would eat its win -----------------


def test_compile_is_a_config_flag_and_eager_is_the_same_object_back():
    # "Compile is a config flag: eager stays a working fallback for debugging" — and the
    # fallback has to be the model itself, not a wrapper that merely behaves like one.
    model = MaskedAutoencoder.from_config(tiny_config())
    assert runtime_mod.compile_model(model, tiny_config()) is model


def test_compiling_wraps_the_model_without_replacing_its_parameters():
    # The optimizer is built over the eager model (#93), so a compile that copied parameters
    # would train a model nobody was optimizing — and the loss would fall anyway, because the
    # compiled copy would still be descending its own gradient.
    cfg = tiny_config(runtime=Runtime())
    model = MaskedAutoencoder.from_config(cfg)
    compiled = runtime_mod.compile_model(model, cfg)
    assert compiled is not model
    assert next(compiled.parameters()) is next(model.parameters())
    assert runtime_mod.eager_module(compiled) is model


def test_the_compiled_model_reaches_sdpa_and_the_original_blocks():
    # `attention_path` has to see through the wrapper: at launch what is on hand is the
    # compiled callable, and reporting "no attention blocks found" would be worse than useless.
    cfg = tiny_config(runtime=Runtime())
    compiled = runtime_mod.compile_model(MaskedAutoencoder.from_config(cfg), cfg)
    assert runtime_mod.attention_path(compiled) == "sdpa"


def test_a_warmed_up_compiled_model_does_not_recompile_on_later_batches():
    # §7.3's win is contingent on this and nothing else: a fixed mask count (#90) plus a
    # dropped short batch (#93) means every step sees one set of shapes. A silent recompile
    # loop trains correctly, reports nothing, and costs more than compiling ever saved.
    cfg = tiny_config(runtime=Runtime())
    compiled = runtime_mod.compile_model(MaskedAutoencoder.from_config(cfg), cfg)
    compiled(tiles(4, seed=1))  # warmup: this is the compile
    watch = runtime_mod.CompileWatch()
    for seed in range(2, 6):
        compiled(tiles(4, seed=seed))
    assert watch.compiled == 0


def test_the_watch_sees_a_recompile_when_the_shape_moves():
    # The guard on the test above: a watch that could never report anything would pass it.
    cfg = tiny_config(runtime=Runtime())
    compiled = runtime_mod.compile_model(MaskedAutoencoder.from_config(cfg), cfg)
    compiled(tiles(4, seed=1))
    watch = runtime_mod.CompileWatch()
    compiled(tiles(3, seed=2))  # a short final batch, which is why `drop_last` is on
    assert watch.compiled >= 1
    watch.reset()
    assert watch.compiled == 0


def test_frames_compiled_inside_a_monitoring_block_are_not_charged_to_the_training_step():
    # #109 in one assertion. The pre-split watch reported one number, and §7.3's sentence is
    # about the training step, so §10's boundary compiles read as a training-step recompile —
    # on every healthy run, at 18 frames, identically across R1, R2 and R3.
    cfg = tiny_config(runtime=Runtime())
    compiled = runtime_mod.compile_model(MaskedAutoencoder.from_config(cfg), cfg)
    compiled(tiles(4, seed=1))  # warmup: the training step's own compile
    watch = runtime_mod.CompileWatch()
    with watch.monitoring():
        # A batch size no other test in this module uses: Dynamo's cache is process-wide, so a
        # shape a neighbour already compiled would hit the cache and compile nothing here.
        compiled(tiles(7, seed=2))  # stands in for §10's boundary: a shape the step never sees
    assert watch.monitored >= 1
    assert watch.compiled == 0  # §7.3's claim, in §7.3's units, holds
    assert watch.total == watch.monitored  # and nothing was lost in the split


def test_a_recompile_outside_a_monitoring_block_still_lands_on_the_training_step():
    # The guard on the test above: attribution that swallowed everything would pass it, and
    # would have hidden the one failure the watch exists to catch.
    cfg = tiny_config(runtime=Runtime())
    compiled = runtime_mod.compile_model(MaskedAutoencoder.from_config(cfg), cfg)
    compiled(tiles(4, seed=1))
    watch = runtime_mod.CompileWatch()
    with watch.monitoring():
        compiled(tiles(9, seed=2))  # again, a shape unique to this test
    compiled(tiles(11, seed=3))  # the step itself moving a shape — this is the real fault
    assert watch.compiled >= 1
    assert watch.monitored >= 1


# --- Peak memory, printed beside the batch that caused it (#115) ---------------


def test_peak_memory_says_nothing_rather_than_zero_off_cuda():
    # A laptop --dry-run and an mps test both have no peak worth reporting. Zeros would be
    # quotable, and #113 is a whole ticket about a quotable number attached to the wrong run.
    cfg = tiny_config(runtime=Runtime())
    assert runtime_mod.format_peak_memory(cfg, "cpu") is None
    assert runtime_mod.format_peak_memory(cfg, "mps") is None


def test_peak_memory_reads_the_batch_split_off_the_config_it_was_given(monkeypatch):
    # The property #113 turned on: R0 recorded a batch size taken from the plan while the preset
    # ran another. Reading it off `cfg.optimization` is what makes the two numbers inseparable,
    # so the line has to move when the split does — and say the same effective batch either way.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: 18 * 1024**3)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *a, **k: 19 * 1024**3)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: type("P", (), {"total_memory": 32 * 1024**3})(),
    )

    split = tiny_config(optimization=Optimization(batch_size=512, accum_iter=4))
    other = tiny_config(optimization=Optimization(batch_size=256, accum_iter=8))

    assert "at batch 512 x 4 accum = 2048 effective" in runtime_mod.format_peak_memory(
        split, "cuda"
    )
    assert "at batch 256 x 8 accum = 2048 effective" in runtime_mod.format_peak_memory(
        other, "cuda"
    )
    # The memory half is reported against the card, so a peak is readable as a margin and not
    # just as a number — the thing an OOM never tells you.
    assert "18.00 GiB allocated / 19.00 GiB reserved of 32.00 GiB" in (
        runtime_mod.format_peak_memory(split, "cuda")
    )
