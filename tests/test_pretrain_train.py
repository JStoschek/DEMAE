"""The training step, end to end: loader -> augment -> mask -> encode -> decode -> loss ->
backward -> step, with the loss going down (#93, plan §6 and §7.1).

This is the tracer bullet, so the tests are of two kinds and it is worth keeping them apart.

**The loop runs and learns.** A few hundred steps over a small fixture corpus, on CPU, and the
loss falls. The corpus is deliberately *smooth* synthetic terrain rather than the noise the
other fixtures use: a masked autoencoder cannot reduce a reconstruction loss on white noise
below the variance it normalizes to, so on a noise corpus "the loss falls" would be untestable
and a broken loop would pass.

**The invisible properties.** Everything else here is a claim that no loss curve would
contradict: that the biases and norms are exempt from weight decay, that a step's gradient is
the same whether the effective batch arrives whole or in accumulated pieces, that the LR at
step 0 is 0 and not the peak, that autocast is bf16 and no GradScaler exists anywhere, that the
run reads `dataset.loader`'s sample dict unchanged, and that one seed and one config reproduce a
trajectory exactly. Every one of these trains perfectly well when it is wrong.

The fixture config is R1's recipe with three things shrunk to fit a laptop — a 64-wide 2-block
encoder, a batch of 4, and a 150-epoch schedule — plus one number moved for a reason worth
stating: ``lr_batch_base`` drops from 256 to the fixture's own batch, so §6's rule lands on a
learning rate that can move a model in 300 steps rather than on the 2.3e-6 the rule would give
a batch of 4. The rule itself, at the shipped batch of 2048, is asserted in
`test_pretrain_schedule.py`.
"""

from __future__ import annotations

import ast
import inspect
import json

import numpy as np
import pytest
import torch

from pretrain import optim
from pretrain import schedule as schedule_mod
from pretrain import train as train_mod
from pretrain.config import Architecture, Data, Optimization, RunConfig, Runtime, Schedule
from pretrain.train import StepRecord, Trainer

#: conftest's geography — one geocell held out as `test`, one COG as `val` — reused so the
#: split containment assertion (#89) is exercised by the training path too.
from tests.conftest import SPLIT_COGS, build_corpus

#: Five COGs x 4 tiles = 20 tiles, of which 8 are train. Small enough to memorize, which is
#: what makes "the loss falls" a sharp claim rather than a hopeful one.
TILES_PER_COG = 4


def smooth_tile(rng: np.random.Generator, base: float) -> np.ndarray:
    """A 224x224 window of *learnable* terrain: one broad ridge, a cross-ridge, and a tilt.

    Smooth on purpose, and at a wavelength several patches wide. A masked patch has to be
    predictable from its neighbours for a reconstruction loss to fall at all, and the default
    fixture window is white noise, whose per-patch-normalized target is unpredictable by
    construction — on that corpus a broken loop and a working one both plateau at 1.0.
    """
    y, x = np.mgrid[0:224, 0:224] / 224.0
    theta = rng.uniform(0, np.pi)
    u = x * np.cos(theta) + y * np.sin(theta)
    v = -x * np.sin(theta) + y * np.cos(theta)
    field = (
        60.0 * np.sin(np.pi * (u + rng.uniform(0, 1)))
        + 25.0 * np.sin(2 * np.pi * (v + rng.uniform(0, 1)))
        + 40.0 * u
    )
    return (base + field).astype(np.float32)


@pytest.fixture(scope="module")
def smooth_corpus(tmp_path_factory):
    """A tiny corpus of smooth terrain, with the #77 split shape."""
    root = tmp_path_factory.mktemp("smooth_corpus")
    return build_corpus(root, SPLIT_COGS, tiles_per_cog=TILES_PER_COG, tile_fn=smooth_tile)


def tiny_config(corpus, **changes) -> RunConfig:
    """R1's recipe at laptop scale — see the module docstring for the four shrunken knobs."""
    cfg = RunConfig(
        run_id="test",
        architecture=Architecture(
            encoder_dim=64, encoder_depth=2, encoder_heads=2,
            decoder_dim=64, decoder_depth=1, decoder_heads=2,
        ),
        optimization=Optimization(
            batch_size=4, accum_iter=1, blr=3e-3, lr_batch_base=4,
        ),
        schedule=Schedule(epochs=150, warmup_epochs=6, milestones=(150,)),
        data=Data(
            corpus_dir=str(corpus),
            # The committed artifacts belong to the real 500k corpus; pointing at absent files
            # is how the fixture keeps `SplitDataset`'s block check on the derived lists only.
            split_assignment_path=str(corpus / "no-assignment.csv"),
            split_blocks_path=str(corpus / "no-blocks.json"),
            num_workers=0,
        ),
    )
    return cfg.replace(**changes) if changes else cfg


@pytest.fixture
def cfg(smooth_corpus) -> RunConfig:
    return tiny_config(smooth_corpus)


def hand_batch(n: int, *, seed: int = 0) -> dict:
    """A batch shaped like the loader's sample dict, all land, built without any IO."""
    g = torch.Generator().manual_seed(seed)
    return {
        "image": torch.randn(n, 1, 224, 224, generator=g),
        "land_mask": torch.ones(n, 224, 224, dtype=torch.bool),
    }


# --- The read path: `dataset.loader`, consumed unchanged ----------------------


def test_the_train_loader_is_the_adr_0003_read_path_with_augmentation_wrapped_around_it(cfg):
    from dataset.loader import TileDataset
    from pretrain.augment import DihedralDataset
    from pretrain.data import SplitDataset

    loader = train_mod.build_loader(cfg, cfg.data.train_split)
    augmented = loader.dataset
    assert isinstance(augmented, DihedralDataset)  # §6's augmentation, as a wrapper
    split = augmented.dataset
    assert isinstance(split, SplitDataset) and split.split == "train"
    # And underneath, `dataset.loader`'s own Dataset — not a copy, not a subclass of one.
    assert type(split.dataset) is type(TileDataset(cfg.data.corpus_dir))


def test_the_batch_that_reaches_the_step_is_the_adr_0003_sample(cfg):
    batch = next(iter(train_mod.build_loader(cfg, "train")))
    assert set(batch) == {"image", "land_mask", "mean", "std", "tile_id", "stratum_id"}
    assert batch["image"].shape == (4, 1, 224, 224)
    assert batch["land_mask"].shape == (4, 224, 224)


def test_the_train_loader_drops_a_partial_batch_so_every_step_has_one_shape(cfg):
    # The #95 precondition: `torch.compile(dynamic=False)` recompiles on a short final batch.
    loader = train_mod.build_loader(cfg, "train")
    assert loader.drop_last is True
    assert all(b["image"].shape[0] == 4 for b in loader)


def test_the_val_loader_is_built_without_augmentation(cfg):
    # §10.1's val loss has to be comparable epoch to epoch; a random dihedral draw per read
    # would make it a moving measurement.
    from pretrain.augment import DihedralDataset

    val = train_mod.build_loader(cfg, cfg.data.val_split, augment=False, shuffle=False)
    assert not isinstance(val.dataset, DihedralDataset)
    assert val.dataset.split == "val"


def test_the_loader_generator_is_seeded_from_the_config(cfg):
    # §6: "fixed seed for model init and the DataLoader generator".
    order = [
        [t for b in train_mod.build_loader(cfg, "train") for t in b["tile_id"]] for _ in range(2)
    ]
    assert order[0] == order[1]
    other = cfg.replace(runtime=Runtime(seed=cfg.runtime.seed + 1))
    assert [t for b in train_mod.build_loader(other, "train") for t in b["tile_id"]] != order[0]


# --- Precision: bf16 autocast, and no GradScaler ------------------------------


def test_the_forward_pass_runs_under_bf16_autocast(cfg):
    trainer = Trainer.from_config(cfg, steps_per_epoch=2)
    seen = {}

    def probe(module, args):
        seen["enabled"] = torch.is_autocast_enabled("cpu")
        seen["dtype"] = torch.get_autocast_dtype("cpu")

    trainer.model.blocks[0].register_forward_pre_hook(probe)
    trainer.step([hand_batch(2)])
    assert seen == {"enabled": True, "dtype": torch.bfloat16}


def test_fp32_is_the_debugging_escape_hatch_and_turns_autocast_off(cfg):
    trainer = Trainer.from_config(cfg.replace(runtime=Runtime(precision="fp32")), steps_per_epoch=2)
    seen = {}
    trainer.model.blocks[0].register_forward_pre_hook(
        lambda m, a: seen.update(enabled=torch.is_autocast_enabled("cpu"))
    )
    trainer.step([hand_batch(2)])
    assert seen == {"enabled": False}


def test_no_grad_scaler_exists_anywhere_in_the_loop():
    # §7.1: bf16 needs none, and one would mask the loss scale. The config refuses the flag;
    # this refuses the object, in the three modules that make up the step. Over the parsed
    # source rather than the text, so the module docstrings can *discuss* the thing they must
    # not construct.
    for module in (train_mod, optim, schedule_mod):
        tree = ast.parse(inspect.getsource(module))
        referenced = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        referenced |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        assert "GradScaler" not in referenced, module.__name__
    assert RunConfig().runtime.grad_scaler is False
    with pytest.raises(ValueError, match="GradScaler"):
        Runtime(grad_scaler=True)


def test_the_loss_and_the_gradients_stay_fp32_under_bf16_autocast(cfg):
    trainer = Trainer.from_config(cfg, steps_per_epoch=2)
    record = trainer.step([hand_batch(2)])
    assert isinstance(record.loss, float) and np.isfinite(record.loss)
    grads = [p.grad for p in trainer.model.parameters() if p.grad is not None]
    assert grads and all(g.dtype is torch.float32 for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)


# --- The step -----------------------------------------------------------------


def test_a_step_moves_the_weights_once_the_warmup_has_lifted_the_lr(cfg):
    trainer = Trainer.from_config(cfg, steps_per_epoch=1)
    before = trainer.model.decoder_pred.weight.detach().clone()
    trainer.step([hand_batch(2)], step=10)  # ten epochs in, well past the 2-epoch warmup
    assert not torch.allclose(before, trainer.model.decoder_pred.weight)


def test_the_first_step_is_taken_at_a_learning_rate_of_zero(cfg):
    # §6's warmup starts at 0. The gradient is computed and the optimizer steps — but at lr 0
    # AdamW moves nothing, which is the observable difference between a schedule that is
    # applied and one that is merely constructed.
    trainer = Trainer.from_config(cfg, steps_per_epoch=1)
    before = trainer.model.decoder_pred.weight.detach().clone()
    record = trainer.step([hand_batch(2)])
    assert record.lr == 0.0
    assert torch.equal(before, trainer.model.decoder_pred.weight)


def test_the_step_records_what_the_schedule_said(cfg):
    trainer = Trainer.from_config(cfg, steps_per_epoch=4)
    record = trainer.step([hand_batch(2)], step=6)
    assert isinstance(record, StepRecord)
    assert record.step == 6 and record.epoch == 1.5
    assert record.lr == pytest.approx(trainer.schedule.lr_at_step(6))
    assert [g["lr"] for g in trainer.optimizer.param_groups] == [record.lr, record.lr]


def test_gradient_accumulation_matches_one_batch_of_twice_the_size(smooth_corpus):
    """§6's `accum_iter` is a memory trick, not a recipe change: the gradient must be the same.

    Run in fp32 deliberately. What is on trial is the accumulation arithmetic — the 1/N
    division and the shared mask draw — and under bf16 the same computation at two batch sizes
    differs by more than the property being measured.
    """
    base = tiny_config(smooth_corpus).replace(runtime=Runtime(precision="fp32"))
    batch = hand_batch(8, seed=3)
    halves = [
        {k: v[:4] for k, v in batch.items()},
        {k: v[4:] for k, v in batch.items()},
    ]

    def grads_after(cfg, micro_batches):
        trainer = Trainer.from_config(cfg, steps_per_epoch=1)
        torch.manual_seed(11)  # the mask draw: 8 rows, or 4 then 4 — the same stream either way
        trainer.step(micro_batches, step=10)
        return {n: p.grad.clone() for n, p in trainer.model.named_parameters() if p.grad is not None}

    whole = grads_after(base.replace(optimization=base.optimization.replace(accum_iter=1)), [batch])
    accumulated = grads_after(
        base.replace(optimization=base.optimization.replace(accum_iter=2)), halves
    )
    assert set(whole) == set(accumulated)
    for name, g in whole.items():
        assert torch.allclose(g, accumulated[name], rtol=1e-4, atol=1e-7), name


def test_the_loss_of_an_accumulated_step_is_the_mean_over_its_micro_batches(cfg):
    trainer = Trainer.from_config(
        cfg.replace(optimization=cfg.optimization.replace(accum_iter=2)), steps_per_epoch=1
    )
    torch.manual_seed(5)
    # Step 0, so the LR is 0 and the weights the recomputation below reads are the same ones.
    record = trainer.step([hand_batch(2, seed=1), hand_batch(2, seed=2)], step=0)
    assert record.micro_batches == 2
    torch.manual_seed(5)
    losses = [
        float(trainer.micro_loss(hand_batch(2, seed=1)).detach()),
        float(trainer.micro_loss(hand_batch(2, seed=2)).detach()),
    ]
    assert record.loss == pytest.approx(sum(losses) / 2, rel=1e-5)


def test_gradients_do_not_leak_from_one_step_into_the_next(cfg):
    # Accumulating across steps would grow the gradient without bound and is invisible in a
    # loss curve for a surprisingly long time — AdamW normalizes by the second moment, so an
    # accumulating gradient mostly changes the *direction* it points.
    trainer = Trainer.from_config(cfg, steps_per_epoch=1)
    weight = trainer.model.decoder_pred.weight
    weight.grad = torch.full_like(weight, 1e6)
    trainer.step([hand_batch(2, seed=1)], step=10)
    assert float(weight.grad.abs().max()) < 1.0


def test_clipping_is_off_by_default_and_reachable_from_the_config(cfg):
    # §6: MAE clips nothing. The knob exists for the bf16-instability contingency (§13).
    assert cfg.optimization.grad_clip is None
    assert Trainer.from_config(cfg, steps_per_epoch=1).step([hand_batch(2)]).grad_norm is None

    clipped = cfg.replace(optimization=cfg.optimization.replace(grad_clip=1e-6))
    trainer = Trainer.from_config(clipped, steps_per_epoch=1)
    record = trainer.step([hand_batch(2)], step=10)
    assert record.grad_norm is not None and record.grad_norm > 1e-6
    total = torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(p.grad) for p in trainer.model.parameters()])
    )
    assert float(total) == pytest.approx(1e-6, rel=1e-3)  # clipped to the ceiling


# --- The objective the step trains --------------------------------------------


def test_the_land_mask_reaches_the_loss_as_pixel_weights(cfg):
    # §4's ocean weighting is configured on by default, and the loss refuses to score without
    # weights — so a step completing is already evidence. This pins *which* weights: the
    # loader's own land mask, in the target's layout.
    trainer = Trainer.from_config(cfg, steps_per_epoch=1)
    batch = hand_batch(2)
    batch["land_mask"][0, :112] = False  # half of the first sample is ocean
    seen = {}
    real = trainer.loss_fn.forward

    def spy(pred, target, mask, *, pixel_weights=None):
        seen["weights"] = pixel_weights
        return real(pred, target, mask, pixel_weights=pixel_weights)

    trainer.loss_fn.forward = spy
    trainer.step([batch], step=10)
    weights = seen["weights"]
    assert weights is not None and weights.shape == (2, 196, 256)
    assert float(weights[0].mean()) == pytest.approx(0.5) and float(weights[1].mean()) == 1.0


def test_an_unweighted_run_passes_no_weights_at_all(cfg):
    # The mirror image: with §4's weighting ablated off, weights must not arrive — the loss
    # refuses them, precisely so a run cannot report one objective and train the other.
    unweighted = cfg.replace(loss=cfg.loss.replace(ocean_weighted=False))
    trainer = Trainer.from_config(unweighted, steps_per_epoch=1)
    assert np.isfinite(trainer.step([hand_batch(2)], step=10).loss)


def test_the_target_is_the_tile_the_model_was_shown(cfg):
    # patchify is the model's own, so the target and the prediction share one layout by
    # construction; this is the check that the *image* is what got patchified.
    trainer = Trainer.from_config(cfg, steps_per_epoch=1)
    batch = hand_batch(2)
    target = trainer.model.patchify(batch["image"])
    assert torch.equal(trainer.model.unpatchify(target), batch["image"])


# --- The loop -----------------------------------------------------------------


def test_a_few_hundred_steps_run_to_completion_and_the_loss_falls(cfg):
    # 300 steps is the fixture's whole 150-epoch schedule, so this also walks the LR curve from
    # 0 through the peak and back to 0 — the shape a real run has, at a size a laptop can hold.
    records = train_mod.train(cfg, steps=300)
    assert len(records) == 300
    assert [r.step for r in records] == list(range(300))
    losses = [r.loss for r in records]
    assert all(np.isfinite(losses))
    first, last = np.mean(losses[:20]), np.mean(losses[-20:])
    assert last < 0.8 * first, f"loss went {first:.3f} -> {last:.3f}"
    # The final step sits half an epoch short of the schedule's end, so it is taken at the
    # bottom of the cosine rather than at its zero — which is exactly where a 150-epoch run's
    # last step is. The zero itself is asserted in `test_pretrain_schedule.py`.
    assert records[-1].lr < 1e-3 * cfg.lr


def test_the_loop_walks_the_schedule_and_crosses_epoch_boundaries(cfg):
    # 8 train tiles / batch 4 / accum 1 = 2 optimizer steps per epoch, so six steps is three
    # epochs — and the fractional epoch is what the warmup is measured in.
    records = train_mod.train(cfg, steps=6)
    assert [r.epoch for r in records] == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    assert records[0].lr == 0.0 and records[-1].lr > 0.0


def test_the_shipped_accumulation_of_two_runs_through_the_whole_loop(cfg):
    # `accum_iter=2` is the shipped default (§6), and everything else here trains at 1 — so the
    # loop's own grouping, and the schedule's units, are otherwise never exercised at the shape
    # R1 actually runs.
    accumulated = cfg.replace(optimization=cfg.optimization.replace(accum_iter=2))
    records = train_mod.train(accumulated, steps=4)
    assert [r.micro_batches for r in records] == [2, 2, 2, 2]
    # 8 train tiles / batch 4 = 2 micro-batches = 1 optimizer step per epoch, so every step is
    # a whole epoch and the schedule's fractional epochs are whole numbers.
    assert [r.epoch for r in records] == [0.0, 1.0, 2.0, 3.0]


def test_a_short_group_at_an_epoch_boundary_is_dropped_rather_than_stepped_on(cfg):
    # 3 micro-batches per epoch against accum_iter 2 leaves one over. Stepping on it would take
    # a step on half an effective batch at the LR a whole one earns — and would put the loop
    # one step per epoch ahead of the schedule it is being driven by.
    odd = cfg.replace(optimization=cfg.optimization.replace(batch_size=2, accum_iter=2))
    loader = train_mod.build_loader(odd, "train")
    assert len(loader) == 4  # 8 tiles / batch 2
    groups = list(train_mod.micro_batch_groups(loader, 3, steps=2))
    assert [len(g) for g in groups] == [3, 3]  # one micro-batch dropped per epoch, not stepped on
    assert train_mod.steps_per_epoch_for(loader, odd.replace(
        optimization=odd.optimization.replace(accum_iter=3)
    )) == 1


def test_the_same_seed_and_config_reproduce_the_loss_trajectory(cfg):
    a = [r.loss for r in train_mod.train(cfg, steps=12)]
    b = [r.loss for r in train_mod.train(cfg, steps=12)]
    assert a == b  # exactly: same seed, same config, same box


def test_a_different_seed_gives_a_different_trajectory(cfg):
    # The guard on the test above: it must be reproducing a *run*, not a constant.
    other = cfg.replace(runtime=Runtime(seed=cfg.runtime.seed + 1))
    assert [r.loss for r in train_mod.train(cfg, steps=12)] != [
        r.loss for r in train_mod.train(other, steps=12)
    ]


def test_the_loop_reports_every_step_to_a_callback(cfg):
    # The seam #97's monitoring hangs off: one record per optimizer step, as it happens.
    seen: list[StepRecord] = []
    records = train_mod.train(cfg, steps=4, on_step=seen.append)
    assert seen == records


def test_training_longer_than_the_split_reshuffles_rather_than_repeating_the_order(cfg):
    # Epoch 2 must not be epoch 1 again: the loop re-iterates the loader (a fresh shuffle)
    # rather than cycling one materialized epoch, which would train fine and quietly turn a
    # 400-epoch run into 400 repetitions of one ordering.
    train_mod.seed_run(cfg)
    loader = train_mod.build_loader(cfg, "train")
    order = [
        tile
        for group in train_mod.micro_batch_groups(loader, 1, steps=4)
        for batch in group
        for tile in batch["tile_id"]
    ]
    assert len(order) == 16  # four steps of a 2-step epoch: two whole epochs
    assert order[:8] != order[8:]
    assert sorted(order[:8]) == sorted(order[8:])  # the same tiles, in a different order


# --- The entrypoint -----------------------------------------------------------


def test_the_entrypoint_trains_for_n_steps_from_a_config_file(cfg, tmp_path, capsys):
    config_file = tmp_path / "run.json"
    config_file.write_text(json.dumps(cfg.to_dict()))
    assert train_mod.main(["--config", str(config_file), "--steps", "4", "--log-every", "1"]) == 0
    out = capsys.readouterr().out
    assert "DEMAE Phase 1" in out  # #89's header, printed before the first step
    assert len([line for line in out.splitlines() if line.startswith("step ")]) == 4


def test_the_entrypoint_refuses_an_unknown_knob(cfg, tmp_path):
    bad = tmp_path / "typo.json"
    bad.write_text(json.dumps({"optimization": {"lr": 1e-3}}))
    with pytest.raises(ValueError, match="unknown optimization knob"):
        train_mod.main(["--config", str(bad), "--steps", "1"])


def test_a_split_smaller_than_one_batch_is_refused_rather_than_spun_on(cfg):
    # With `drop_last` on, such a loader yields nothing — and a loop that waits for a step it
    # will never be handed hangs silently, which on a rented box is an hour of paid nothing.
    big_batch = cfg.replace(optimization=cfg.optimization.replace(batch_size=64))
    loader = train_mod.build_loader(big_batch, "train")
    assert len(loader) == 0
    with pytest.raises(ValueError, match="no complete step"):
        next(train_mod.micro_batch_groups(loader, 1, steps=1))
    # Same refusal one rung up: batches exist, but never enough of them to accumulate one.
    small = cfg.replace(optimization=cfg.optimization.replace(batch_size=4, accum_iter=8))
    with pytest.raises(ValueError, match="no complete step"):
        next(train_mod.micro_batch_groups(train_mod.build_loader(small, "train"), 8, steps=1))
