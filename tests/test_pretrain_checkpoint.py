"""Survival: atomic checkpoints, retention, and kill-and-resume equivalence (#94, plan §11).

vast.ai instances vanish permanently rather than pausing, so exact resume is plan §11's gate
and item 4 of the §15 pre-flight checklist. The tests here are in five groups, and the split
matters because only two of them — **Equivalence** and **The device** — would fail visibly on a
rented box. The other three fail on a laptop, which is why they are cheap to keep honest and
why the two that do not have twice been the ones found wrong.

**The artifact.** What a checkpoint carries, and that a killed process cannot leave one that
half-loads. A truncated checkpoint that `torch.load` accepts is the worst outcome available:
the run resumes from weights that are partly the previous step's.

**Retention.** Last 3 plus the milestone set, with the set read off the config. Asserted with
a milestone set containing 800, not the 50/100/200/400 default — R5 (#87) may extend, and with
a hardcoded set the 800 checkpoint would not be protected, so last-3 retention would delete
the endpoint of the very §14 compute-vs-quality plot the milestones exist for.

**The order.** :class:`~pretrain.train.EpochShuffleSampler` makes an epoch's sample order a
pure function of ``(seed, epoch)`` and lets a resumed epoch start partway through *that* order.
Resuming mid-epoch and getting a fresh shuffle is the bug this whole ticket exists to prevent,
and it is invisible: the run trains, the loss descends, and some tiles are simply seen twice in
one epoch while others are skipped.

**Equivalence.** Kill at step k, resume, and the loss trajectory over the next 50+ steps is
*bit-identical* to an uninterrupted run's. That is the §15.4 gate, and it is the one assertion
that no amount of reading the code substitutes for. It runs **at two worker counts**, because
#94 first shipped it at ``num_workers = 0`` while the default was 8 and the dihedral draw did
not survive the difference (#104); a gate that holds only at a configuration nobody runs is not
a gate, and the extra process spawn is the cheapest thing in this file.

**The device.** The same gate again on the axis the fixtures above hardcode. Every equivalence
test in this file trains at ``device="cpu"``, and #105 was two separate resume bugs sitting
behind that one default — a payload device-mapped onto the accelerator with the RNG block
inside it, and a mask stream drawn from a device generator nothing captured. Neither is
CUDA-specific and neither needs a rented box: ``mps`` reproduces both, and the laptop this
suite runs on has one. What cannot be reached without hardware — the CUDA branch — is stood in
for rather than skipped, because a group that runs only where the hardware happens to be is
what let this file be green through both bugs.
"""

from __future__ import annotations

import dataclasses
import json
import random
import shutil
import time

import numpy as np
import pytest
import torch

from pretrain import checkpoint as ckpt_mod
from pretrain import train as train_mod
from pretrain.checkpoint import CheckpointPolicy, RunState
from pretrain.config import Data, Optimization, RunConfig, Runtime, Schedule
from pretrain.fingerprint import fingerprint_for
from tests.conftest import SPLIT_COGS
from tests.test_pretrain_train import TILES_PER_COG, smooth_tile, tiny_config

#: Eight train tiles at batch 2 is four micro-batches — four optimizer steps — per epoch. Big
#: enough that a kill at an odd step lands *inside* an epoch, which is the case that separates
#: a resumed order from a re-shuffled one.
BATCH = 2
STEPS_PER_EPOCH = 4


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """#93's smooth-terrain fixture corpus, module-scoped: every test here reads the same one."""
    from tests.conftest import build_corpus

    root = tmp_path_factory.mktemp("resume_corpus")
    return build_corpus(root, SPLIT_COGS, tiles_per_cog=TILES_PER_COG, tile_fn=smooth_tile)


@pytest.fixture
def cfg(corpus) -> RunConfig:
    """#93's laptop-scale R1, at a batch that gives four optimizer steps per epoch."""
    base = tiny_config(corpus)
    return base.replace(optimization=base.optimization.replace(batch_size=BATCH, accum_iter=1))


@pytest.fixture
def policy(cfg, tmp_path) -> CheckpointPolicy:
    """A policy that writes after **every** step, so a test picks its own kill point."""
    return CheckpointPolicy.from_config(
        cfg, directory=tmp_path / "checkpoints", interval_seconds=0.0
    )


def a_trainer(cfg):
    return train_mod.Trainer.from_config(cfg, steps_per_epoch=STEPS_PER_EPOCH)


def write_one(policy, cfg, *, epoch, step, ends_epoch=True, trainer=None):
    """One checkpoint at a stated position, with a real model and optimizer behind it."""
    trainer = trainer or a_trainer(cfg)
    return policy.save(
        config=cfg,
        fingerprint=fingerprint_for(cfg),
        state=RunState(epoch=epoch, step=step),
        model=trainer.model,
        optimizer=trainer.optimizer,
        rng=ckpt_mod.capture_rng(),
        ends_epoch=ends_epoch,
    )


# --- The artifact -------------------------------------------------------------


def test_a_checkpoint_carries_everything_needed_to_continue(cfg, policy):
    # §11's list, item by item: weights, optimizer state, position, every RNG stream, the full
    # config, and #89's fingerprint block. A checkpoint missing any one of them resumes into a
    # run that is *nearly* the one that was killed, which is the failure this ticket prevents.
    trainer = a_trainer(cfg)
    path = write_one(policy, cfg, epoch=3, step=12, trainer=trainer)

    loaded = ckpt_mod.load_checkpoint(path)
    assert loaded.state == RunState(epoch=3, step=12, batches_into_epoch=0)
    assert loaded.config == cfg and loaded.config_hash == cfg.config_hash
    assert set(loaded.model) == set(trainer.model.state_dict())
    assert loaded.optimizer["param_groups"][0]["name"] == "decay"
    # The optional streams are *written* on every box, present or not, so the payload's shape
    # does not depend on what wrote it — only their values do (#105).
    assert set(loaded.rng) == set(ckpt_mod.RNG_STREAMS) | set(ckpt_mod.OPTIONAL_RNG_STREAMS)
    assert loaded.fingerprint == fingerprint_for(cfg)
    assert loaded.fingerprint.dataset_version and loaded.fingerprint.selection_list_md5


def test_a_checkpoint_loads_without_unpickling_arbitrary_objects(cfg, policy):
    # Every field is a tensor or a plain type, so `weights_only=True` is enough to read one.
    # Checkpoints come back from an off-instance bucket (§11) — a payload that needed the
    # unrestricted unpickler would make restoring a run an arbitrary-code-execution path.
    path = write_one(policy, cfg, epoch=1, step=4)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["format_version"] == ckpt_mod.FORMAT_VERSION


def test_the_write_goes_through_a_temporary_name_and_a_rename(cfg, policy, monkeypatch):
    seen: list[str] = []
    real = torch.save
    monkeypatch.setattr(torch, "save", lambda obj, f, *a, **k: (seen.append(str(f.name)), real(obj, f, *a, **k))[1])
    path = write_one(policy, cfg, epoch=1, step=4)
    assert seen and seen[0] != str(path) and seen[0].endswith(".tmp")
    assert path.exists() and not list(policy.directory.glob("*.tmp"))


def test_an_interrupted_write_leaves_the_previous_checkpoint_loadable_and_nothing_partial(
    cfg, policy, monkeypatch
):
    # The whole point of tmp + rename. A half-written file the loader would happily accept is
    # worse than no checkpoint at all: the run comes back with weights that are partly the
    # previous step's, and nothing about it looks wrong.
    good = write_one(policy, cfg, epoch=1, step=4)

    def die(obj, f, *a, **k):
        f.write(b"\x80\x02}q")  # a plausible pickle prefix, and nothing after it
        raise KeyboardInterrupt("the instance vanished mid-write")

    monkeypatch.setattr(torch, "save", die)
    with pytest.raises(KeyboardInterrupt):
        write_one(policy, cfg, epoch=2, step=8)

    assert [f.path for f in ckpt_mod.discover(policy.directory)] == [good]
    assert ckpt_mod.load_checkpoint(good).state.step == 4
    assert list(policy.directory.iterdir()) == [good]


def test_a_temporary_file_left_by_a_hard_kill_is_not_discovered_as_a_checkpoint(cfg, policy):
    # A SIGKILL gives the writer no chance to clean up, so the tmp file survives. It must not
    # be reachable as a checkpoint — which is why the temporary name is outside the glob.
    good = write_one(policy, cfg, epoch=1, step=4)
    (policy.directory / f".{good.name}.tmp").write_bytes(b"\x80\x02}q")
    assert [f.path for f in ckpt_mod.discover(policy.directory)] == [good]
    assert ckpt_mod.latest(policy.directory).path == good


def test_a_checkpoint_written_by_another_format_is_refused_rather_than_half_read(cfg, policy):
    # The fields a format change adds are exactly the ones whose absence does not surface until
    # the loss curve is already wrong — an RNG stream, a position field. Refused by name.
    path = write_one(policy, cfg, epoch=1, step=4)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["format_version"] = ckpt_mod.FORMAT_VERSION + 1
    ckpt_mod.atomic_save(payload, path)
    with pytest.raises(ValueError, match="format"):
        ckpt_mod.load_checkpoint(path)


def test_a_directory_with_no_checkpoints_resumes_from_nothing_rather_than_raising(tmp_path):
    assert ckpt_mod.discover(tmp_path / "absent") == []
    assert ckpt_mod.latest(tmp_path / "absent") is None


# --- Every RNG stream ---------------------------------------------------------


def test_every_named_rng_stream_round_trips(cfg):
    # §11: "all RNG states". Each one is captured, moved on, restored, and asked to produce the
    # same next draw — the streams are independent, so a missing one is invisible in the others.
    generator = train_mod.build_loader(cfg, "train").generator
    torch.manual_seed(7)

    state = ckpt_mod.capture_rng(generator=generator)
    expected = (
        torch.rand(3),
        np.random.rand(3).tolist(),
        [random.random() for _ in range(3)],
        torch.rand(3, generator=generator),
    )

    torch.manual_seed(99)
    np.random.seed(99)
    random.seed(99)
    generator.manual_seed(99)

    ckpt_mod.restore_rng(state, generator=generator)
    assert torch.equal(torch.rand(3), expected[0])
    assert np.random.rand(3).tolist() == expected[1]
    assert [random.random() for _ in range(3)] == expected[2]
    assert torch.equal(torch.rand(3, generator=generator), expected[3])


def test_the_read_path_contributes_no_rng_stream_to_capture(cfg):
    """#104: the augmentation is a pure function of ``(seed, epoch, index)``, not a stream.

    This is the assertion that the ``num_workers`` gap was closed by *deletion*. Reading a
    sample must leave nothing behind for a checkpoint to have to capture — if it did, that
    state would live in whichever of the eight workers did the reading, and the parent could
    no more record it than before.
    """
    dataset = train_mod.build_loader(cfg, "train").dataset
    before = ckpt_mod.capture_rng()
    dataset[0], dataset[1], dataset[2]
    assert "augmentation" not in ckpt_mod.RNG_STREAMS
    assert not hasattr(dataset, "rng_state")
    # and the ambient streams are untouched by the read, too
    assert ckpt_mod.capture_rng()["numpy"] == before["numpy"]


def test_a_checkpoint_written_before_the_stream_removal_still_resumes(cfg, policy):
    """A pre-#104 payload carries an extra ``rng["augmentation"]``, and must not be refused.

    :data:`~pretrain.checkpoint.FORMAT_VERSION` is deliberately not bumped for the removal:
    §11 exists so that a run in flight survives, and killing every checkpoint already on a box
    to tidy up a key nothing reads would be the opposite of that.
    """
    path = write_one(policy, cfg, epoch=1, step=4)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["rng"]["augmentation"] = {"bit_generator": "PCG64", "state": {}}
    ckpt_mod.atomic_save(payload, path)

    loaded = ckpt_mod.load_checkpoint(path)
    ckpt_mod.restore_rng(loaded.rng, generator=train_mod.seeded_generator(cfg))
    assert trajectory(cfg, steps=6, policy=policy, resume=True)[0].step == 4


# --- The device the RNG block arrives on --------------------------------------

#: Every non-CPU device this box actually has.
#:
#: :func:`~pretrain.train.train_resumable` loads with ``map_location=trainer.device``, so on any
#: of them the whole payload arrives off the CPU — the RNG block included, whose streams are CPU
#: ``ByteTensor``s by definition. #105 was that the restore did not put them back.
#:
#: **This is not a CUDA-only gate, which is the whole reason it can live here.**
#: ``torch.set_rng_state`` wants a CPU tensor from *every* accelerator, so an Apple-silicon
#: laptop with no CUDA at all reproduces #105's traceback line for line on ``mps`` — and the
#: laptop is where this suite runs. On the rented box the same test runs on ``cuda`` with
#: nothing changed.
ACCELERATORS = [
    name
    for name, available in (
        ("cuda", torch.cuda.is_available()),
        ("mps", torch.backends.mps.is_available()),
    )
    if available
]


class ArrivedOnADevice:
    """An RNG stream as ``map_location`` leaves it: on an accelerator, and staying there.

    Not a tensor and not trying to be one — it implements ``.to`` and ``.device`` and nothing
    else, because those are what a restore touches. What it stands in for is the single property
    that made #105 possible and that a box with no accelerator cannot otherwise show:
    ``.to(dtype)`` names a dtype and not a device, so it preserves the device. That is the whole
    bug — three conversions that changed the dtype of a tensor already ``uint8``, and moved it
    nowhere. ``cuda:0`` because that is where the payload really lands; any non-CPU device
    would do, which is the point.

    The two overloads real ``Tensor.to`` offers here are honoured in the order it demands —
    ``to(dtype)`` and ``to(device, dtype)`` — so a call this accepts is one the production code
    could actually make.
    """

    def __init__(self, state: torch.Tensor):
        self._state = state
        self.device = torch.device("cuda:0")

    def to(self, *args, **kwargs):
        target, dtype = kwargs.get("device"), kwargs.get("dtype")
        if args and isinstance(args[0], torch.dtype):
            dtype = args[0]  # `.to(dtype)` — a device may not follow it
        elif args:
            target = args[0]
            dtype = args[1] if len(args) > 1 else dtype
        if target is None or torch.device(target).type != "cpu":
            return self  # converted, unmoved, and still unusable
        return self._state if dtype is None else self._state.to(dtype)


def test_restore_rng_installs_cpu_state_when_the_block_arrived_on_a_device(cfg):
    """#105's torch and DataLoader streams, on a box with no accelerator at all.

    The device test below is the one that measures the real path, and it is skipped on a
    CPU-only box. This one cannot be: it is what keeps the *class* pinned wherever the suite
    runs, which after #104 is the property worth paying for. It fails before the fix with the
    same ``TypeError`` a GPU raises.
    """
    generator = train_mod.build_loader(cfg, "train").generator
    torch.manual_seed(7)
    state = ckpt_mod.capture_rng(generator=generator)
    expected = (torch.rand(3), torch.rand(3, generator=generator))

    torch.manual_seed(99)
    generator.manual_seed(99)
    on_device = dict(
        state,
        torch=ArrivedOnADevice(state["torch"]),
        dataloader=ArrivedOnADevice(state["dataloader"]),
    )
    ckpt_mod.restore_rng(on_device, generator=generator)

    assert torch.equal(torch.rand(3), expected[0])
    assert torch.equal(torch.rand(3, generator=generator), expected[1])


def test_the_cuda_streams_are_put_back_on_the_cpu_too(cfg, monkeypatch):
    """#105's third site, which no accelerator available here reaches.

    ``torch.cuda.set_rng_state_all`` wants CPU ``ByteTensor``s exactly as
    ``torch.set_rng_state`` does — CUDA's generators keep their state on the host — so the
    stream device-mapped onto ``cuda:0`` is refused there too. Only a CUDA box runs this
    branch for real, so the CUDA surface is stood in for and what the call *receives* is what
    is asserted on. It is the cheapest way to stop the one site the fix could have missed from
    being the third axis.
    """
    state = dict(ckpt_mod.capture_rng(), cuda=[ArrivedOnADevice(torch.get_rng_state())])
    installed: list = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", installed.append)

    ckpt_mod.restore_rng(state)

    [streams] = installed
    assert [s.device.type for s in streams] == ["cpu"]
    assert [s.dtype for s in streams] == [torch.uint8]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="this box has no mps device")
def test_the_mps_stream_round_trips_where_that_device_exists():
    """The stream the gate below turned up, asserted on its own so a failure says which.

    :func:`pretrain.model.random_masking` draws from the ambient generator *of the tensor's
    device*, so on ``mps`` the mask comes from a stream that "torch plus CUDA" does not name.
    Uncaptured, the resumed run drew a different mask from the kill point on and the loss
    trajectory parted in the fourth decimal — a resume that looks like a rounding difference
    and is not one.
    """
    torch.mps.manual_seed(7)
    state = ckpt_mod.capture_rng()
    expected = torch.rand(3, device="mps")

    torch.mps.manual_seed(99)
    ckpt_mod.restore_rng(state)
    assert torch.equal(torch.rand(3, device="mps"), expected)


def test_a_checkpoint_written_without_the_optional_streams_still_resumes(cfg, policy):
    """The mirror of the pre-#104 test, for the key #105 added.

    :data:`~pretrain.checkpoint.OPTIONAL_RNG_STREAMS` is deliberately not in the set a resume
    refuses a checkpoint for missing, and this is the assertion that says so. Every checkpoint
    already sitting on a rented box was written before the key existed, and a run in flight
    surviving the change is the entire point of §11.
    """
    path = write_one(policy, cfg, epoch=1, step=4)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    for stream in ckpt_mod.OPTIONAL_RNG_STREAMS:
        payload["rng"].pop(stream)
    ckpt_mod.atomic_save(payload, path)

    loaded = ckpt_mod.load_checkpoint(path)
    ckpt_mod.restore_rng(loaded.rng, generator=train_mod.seeded_generator(cfg))
    assert trajectory(cfg, steps=6, policy=policy, resume=True)[0].step == 4


@pytest.mark.parametrize(
    "device",
    ACCELERATORS
    or [
        pytest.param(
            "cpu",
            marks=pytest.mark.skip(
                reason="this box has no non-CPU device, so #105's gate did not run here"
            ),
        )
    ],
)
def test_a_checkpoint_resumes_when_the_payload_arrived_on_a_device(cfg, policy, device):
    """§15.4's gate again, on the axis it was blind on: the device the run trains on.

    Every assertion above this line trains at ``device="cpu"``, where ``map_location`` moves
    nothing and the RNG block is never anywhere it should not be. R0 (#81) ran the real thing
    on a rented GPU and it raised before the first resumed step — the gate had been green
    throughout at a device no run uses, which is #104's shape on a second axis.

    So this runs wherever the box can put a tensor. It is deliberately the *whole* loop rather
    than a load-and-restore unit: what broke was ``train_resumable``'s
    ``map_location=trainer.device``, and a test that hands ``restore_rng`` a dict of its own
    making would not have seen it.

    It earned that scope immediately. With #105's device-mapping fixed the resume stopped
    raising and was still not bit-identical, because the mask draw reads the ambient generator
    of *its own device* and only the CPU's and CUDA's were ever captured — a second stream
    missing on a second axis, found by the first test to run the loop anywhere but the CPU. The
    ``lr`` assertion is the schedule's, and it holds regardless; the ``loss`` one is the one
    that both of those bugs broke.
    """
    uninterrupted = trajectory(cfg, steps=DEVICE_TOTAL, device=device)

    trajectory(cfg, steps=KILL_AT, policy=policy, device=device)
    resumed = trajectory(cfg, steps=DEVICE_TOTAL, policy=policy, resume=True, device=device)

    assert [r.step for r in resumed] == list(range(KILL_AT, DEVICE_TOTAL))
    assert [r.loss for r in resumed] == [r.loss for r in uninterrupted[KILL_AT:]]
    assert [r.lr for r in resumed] == [r.lr for r in uninterrupted[KILL_AT:]]


# --- Retention ----------------------------------------------------------------


def test_retention_keeps_the_last_three_plus_the_milestone_set(cfg, tmp_path):
    milestones = (2, 5)
    policy = CheckpointPolicy.from_config(
        cfg.replace(schedule=Schedule(epochs=10, warmup_epochs=1, milestones=milestones)),
        directory=tmp_path / "ckpt",
    )
    trainer = a_trainer(cfg)
    for epoch in range(1, 9):
        write_one(policy, cfg, epoch=epoch, step=epoch * STEPS_PER_EPOCH, trainer=trainer)

    kept = {(f.epoch, f.milestone) for f in ckpt_mod.discover(policy.directory)}
    assert kept == {(2, True), (5, True), (6, False), (7, False), (8, False)}


def test_the_milestone_set_is_read_from_the_config_and_not_hardcoded(cfg, tmp_path):
    """R5 (#87) may extend to 800, and 800 must then be a *protected* epoch.

    With a fixed 50/100/200/400 set the 800 checkpoint would be an ordinary one, so last-3
    retention would delete it — the endpoint of the compute-vs-quality plot §14 promises
    Phase 2, thrown away by the code that exists to keep it.
    """
    extended = cfg.replace(
        schedule=Schedule(epochs=800, warmup_epochs=40, milestones=(50, 100, 200, 400, 800))
    )
    policy = CheckpointPolicy.from_config(extended, directory=tmp_path / "ckpt")
    trainer = a_trainer(cfg)
    for epoch in (400, 800, 801, 802, 803, 804):
        write_one(policy, extended, epoch=epoch, step=epoch, trainer=trainer)

    kept = sorted(f.epoch for f in ckpt_mod.discover(policy.directory))
    assert kept == [400, 800, 802, 803, 804]  # 801 is neither a milestone nor one of the last 3
    assert {f.epoch for f in ckpt_mod.discover(policy.directory) if f.milestone} == {400, 800}


def test_a_milestone_is_kept_however_many_checkpoints_follow_it(cfg, tmp_path):
    marked = cfg.replace(schedule=Schedule(epochs=20, warmup_epochs=1, milestones=(1,)))
    policy = CheckpointPolicy.from_config(marked, directory=tmp_path / "ckpt")
    trainer = a_trainer(cfg)
    for epoch in range(1, 20):
        write_one(policy, marked, epoch=epoch, step=epoch, trainer=trainer)
    assert min(f.epoch for f in ckpt_mod.discover(policy.directory)) == 1


def test_a_mid_epoch_checkpoint_at_a_milestone_epoch_is_not_a_milestone(cfg, tmp_path):
    # The milestone is *the end of epoch 2*, not "anything that happened during it". A
    # wall-clock write two batches into epoch 2 is an ordinary checkpoint.
    marked = cfg.replace(schedule=Schedule(epochs=10, warmup_epochs=1, milestones=(2,)))
    policy = CheckpointPolicy.from_config(marked, directory=tmp_path / "ckpt")
    path = write_one(policy, marked, epoch=2, step=9, ends_epoch=False)
    assert ckpt_mod.discover(policy.directory)[0].path == path
    assert ckpt_mod.discover(policy.directory)[0].milestone is False


# --- Both triggers ------------------------------------------------------------


def test_a_checkpoint_is_due_at_every_epoch_boundary(cfg, tmp_path):
    policy = CheckpointPolicy.from_config(
        cfg, directory=tmp_path / "ckpt", interval_seconds=3600.0, clock=lambda: 0.0
    )
    assert policy.due(ends_epoch=True) is True
    assert policy.due(ends_epoch=False) is False


def test_a_checkpoint_is_due_once_the_wall_clock_interval_has_passed(cfg, tmp_path):
    now = [0.0]
    policy = CheckpointPolicy.from_config(
        cfg, directory=tmp_path / "ckpt", interval_seconds=1800.0, clock=lambda: now[0]
    )
    assert policy.due(ends_epoch=False) is False
    now[0] = 1799.0
    assert policy.due(ends_epoch=False) is False
    now[0] = 1800.0
    assert policy.due(ends_epoch=False) is True


def test_the_wall_clock_timer_restarts_at_every_write(cfg, tmp_path):
    now = [0.0]
    policy = CheckpointPolicy.from_config(
        cfg, directory=tmp_path / "ckpt", interval_seconds=100.0, clock=lambda: now[0]
    )
    now[0] = 150.0
    assert policy.due(ends_epoch=False) is True
    write_one(policy, cfg, epoch=1, step=4)
    assert policy.due(ends_epoch=False) is False  # the clock restarts from the write
    now[0] = 251.0
    assert policy.due(ends_epoch=False) is True


def test_the_shipped_interval_sits_inside_plan_11s_thirty_to_sixty_minutes(cfg, tmp_path):
    policy = CheckpointPolicy.from_config(cfg, directory=tmp_path / "ckpt")
    assert 30 * 60 <= policy.interval_seconds <= 60 * 60
    assert policy.keep_last == 3


# --- The fingerprint guard ----------------------------------------------------


def test_a_checkpoint_whose_corpus_fingerprint_disagrees_is_refused_loudly(cfg, policy):
    loaded = ckpt_mod.load_checkpoint(write_one(policy, cfg, epoch=1, step=4))
    other = dataclasses.replace(loaded.fingerprint, dataset_version="a-different-corpus")
    with pytest.raises(ValueError, match="dataset_version"):
        loaded.verify_against(other)


def test_a_disagreeing_split_is_refused_too(cfg, policy):
    # The split is the other half of "which data": same tiles, different hold-out, and a run
    # that resumed across the change would have trained on its own evaluation regions.
    loaded = ckpt_mod.load_checkpoint(write_one(policy, cfg, epoch=1, step=4))
    other = dataclasses.replace(loaded.fingerprint, split_md5="0" * 32)
    with pytest.raises(ValueError, match="split_md5"):
        loaded.verify_against(other)


def test_the_same_corpus_mounted_somewhere_else_still_resumes(cfg, policy, tmp_path):
    # §11 resumes on a *fresh* vast.ai instance, where the corpus sits at another path and the
    # commit may have moved. Neither is a claim about the data, and refusing on either would
    # make every legitimate resume look like a corrupted one.
    loaded = ckpt_mod.load_checkpoint(write_one(policy, cfg, epoch=1, step=4))
    moved = dataclasses.replace(
        loaded.fingerprint, corpus_dir=str(tmp_path / "mnt"), git_sha="deadbee-dirty"
    )
    loaded.verify_against(moved)


# --- The epoch order ----------------------------------------------------------


def order_of(sampler) -> list[int]:
    return list(iter(sampler))


def test_an_epochs_order_is_a_pure_function_of_the_seed_and_the_epoch():
    a = train_mod.EpochShuffleSampler(12, seed=7, epoch=5)
    b = train_mod.EpochShuffleSampler(12, seed=7, epoch=5)
    assert order_of(a) == order_of(b)
    assert sorted(order_of(a)) == list(range(12))
    assert order_of(train_mod.EpochShuffleSampler(12, seed=8, epoch=5)) != order_of(b)


def test_a_later_epoch_is_a_different_order_over_the_same_tiles():
    orders = [order_of(train_mod.EpochShuffleSampler(24, seed=7, epoch=e)) for e in range(3)]
    # Pairwise, not chained: `a != b != c` never compares a against c, so an implementation
    # that alternated between two orders would slip through it.
    assert len({tuple(o) for o in orders}) == len(orders)
    assert all(sorted(o) == list(range(24)) for o in orders)


def test_skipping_into_an_epoch_continues_that_epochs_order_rather_than_starting_a_new_one():
    # The bug this ticket exists to prevent, at its smallest: a resumed epoch must be the tail
    # of the order it was interrupted in, not a fresh draw over the same tiles.
    whole = order_of(train_mod.EpochShuffleSampler(24, seed=7, epoch=2))
    resumed = train_mod.EpochShuffleSampler(24, seed=7, epoch=2, skip=10)
    # Asked *before* it is drained: the length is what is left of this epoch, which is what
    # `drop_last` divides to decide how many batches the resumed epoch still owes.
    assert len(resumed) == 14
    assert order_of(resumed) == whole[10:]
    assert (resumed.epoch, resumed.skip) == (3, 0)  # and the next pass is a whole fresh epoch


def test_re_iterating_a_sampler_advances_the_epoch_so_a_plain_loop_still_reshuffles():
    # #93's loop re-iterates the loader and expects a fresh shuffle each pass. Owning the
    # permutation must not quietly turn a 400-epoch run into 400 repetitions of one ordering.
    sampler = train_mod.EpochShuffleSampler(24, seed=7)
    first, second = order_of(sampler), order_of(sampler)
    assert sampler.epoch == 2 and sampler.skip == 0
    assert first != second and sorted(first) == sorted(second)


def test_the_training_loader_reads_through_the_resumable_sampler(cfg):
    loader = train_mod.build_loader(cfg, "train")
    assert isinstance(loader.sampler, train_mod.EpochShuffleSampler)
    assert loader.drop_last is True and loader.batch_size == BATCH
    assert len(loader) == STEPS_PER_EPOCH


def test_a_val_loader_is_read_in_manifest_order_rather_than_shuffled(cfg):
    val = train_mod.build_loader(cfg, "val", augment=False, shuffle=False)
    assert not isinstance(val.sampler, train_mod.EpochShuffleSampler)


# --- Kill and resume ----------------------------------------------------------

#: §15.4 asks for ≥50 steps of agreement. 12 epochs of 4 steps, killed at 5, is 50 further
#: steps across a mid-epoch boundary and eleven whole-epoch ones.
KILL_AT = 5
TOTAL = KILL_AT + 50

#: The device gate's total. Shorter than :data:`TOTAL` on purpose: §15.4 asks for ≥50 steps of
#: agreement from the *equivalence* gate, and it gets them on the CPU. What the device axis adds
#: is whether the payload survives arriving on an accelerator at all (#105), which the first
#: resumed step settles — 15 further steps across three whole-epoch boundaries is margin, not
#: the measurement, and every one of them is paid for three times on hardware the laptop shares
#: with everything else it is doing.
DEVICE_TOTAL = KILL_AT + 15


def trajectory(cfg, *, steps, policy=None, resume=False, on_step=None, device="cpu"):
    return train_mod.train_resumable(
        cfg, steps=steps, policy=policy, resume=resume, device=device, on_step=on_step
    )


def tiles_read(cfg, *, steps, state=RunState()) -> list[str]:
    """The tile ids the cursor hands the loop, in order, starting from ``state``.

    The cursor and the sampler, driven directly and without a model — so what this measures is
    the *order*, which is the cause, rather than the loss, which is the consequence.
    """
    train_mod.seed_run(cfg)
    loader = train_mod.build_loader(cfg, "train")
    cursor = train_mod.EpochCursor(
        loader, accum_iter=cfg.optimization.accum_iter, steps_per_epoch=STEPS_PER_EPOCH,
        state=state,
    )
    seen: list[str] = []
    for i, group in enumerate(cursor.groups()):
        if i == steps:
            break
        seen += [t for batch in group.micro_batches for t in batch["tile_id"]]
    return seen


@pytest.mark.parametrize("num_workers", [0, 2])
def test_kill_at_step_k_and_resume_reproduces_the_loss_trajectory_bit_identically(
    cfg, policy, num_workers
):
    """Plan §15.4, and the gate every GPU dollar after R0 sits behind.

    Not `approx`: bit-identical. An approximate match would pass with the optimizer's second
    moments reset, with the LR curve restarted, or with a fresh shuffle — every one of which
    is a real bug that a run recovers from silently and never reports.

    **Parametrized over the worker count, because for a while it was only true at one of them.**
    #94 shipped this gate at ``num_workers = 0`` while the default was 8: the dihedral element
    came from a NumPy stream held by whichever process was reading, so at any worker count that
    matters it was unreachable from the parent, uncheckpointable, and silently redrawn from the
    kill point on. A gate that holds only at a configuration nobody runs is not a gate. #104 made
    the element a pure function of ``(seed, epoch, index)``, and ``num_workers = 2`` here is what
    says so — it costs a process spawn or two, which is the cheapest thing on this page.
    """
    cfg = cfg.replace(data=cfg.data.replace(num_workers=num_workers))
    uninterrupted = trajectory(cfg, steps=TOTAL)

    killed = trajectory(cfg, steps=KILL_AT, policy=policy)
    assert [r.step for r in killed] == list(range(KILL_AT))
    resumed = trajectory(cfg, steps=TOTAL, policy=policy, resume=True)

    assert [r.step for r in resumed] == list(range(KILL_AT, TOTAL))
    assert len(resumed) == 50
    assert [r.loss for r in resumed] == [r.loss for r in uninterrupted[KILL_AT:]]
    assert [r.lr for r in resumed] == [r.lr for r in uninterrupted[KILL_AT:]]


def test_the_kill_point_is_mid_epoch_so_the_shuffle_order_is_actually_on_trial(cfg, policy):
    # The guard on the test above. Killed at an epoch boundary, a reshuffling resume would
    # still match, and the assertion would be measuring nothing.
    trajectory(cfg, steps=KILL_AT, policy=policy)
    state = ckpt_mod.latest(policy.directory).path
    assert ckpt_mod.load_checkpoint(state).state == RunState(
        epoch=1, step=KILL_AT, batches_into_epoch=1
    )


def test_a_mid_epoch_resume_continues_the_same_sample_order(cfg):
    # Resuming mid-epoch and getting a fresh shuffle is the bug this ticket exists to prevent:
    # the run trains, the loss descends, and epoch 1 quietly shows some tiles twice and others
    # never. Read the tiles, not the losses.
    uninterrupted = tiles_read(cfg, steps=TOTAL)
    resumed = tiles_read(
        cfg, steps=TOTAL - KILL_AT, state=RunState(epoch=1, step=KILL_AT, batches_into_epoch=1)
    )
    assert resumed == uninterrupted[KILL_AT * BATCH :]

    # And the epoch it resumed into is still a whole epoch: every train tile once, no repeats.
    epoch_one = uninterrupted[STEPS_PER_EPOCH * BATCH : 2 * STEPS_PER_EPOCH * BATCH]
    assert sorted(epoch_one) == sorted(set(epoch_one))


def test_the_order_resumes_exactly_with_real_worker_processes_too(cfg):
    """The shipped shape is ``num_workers = 8``, and the order must resume there or nowhere.

    Worth its own test because the parent process is not where the reading happens: with
    workers the sampler still runs here, but batches are assembled over there, prefetched
    ahead of the loop, and — under ``persistent_workers`` — across the epoch boundary. An
    order derived from a *rolling* generator would come apart on exactly those seams, which is
    why :class:`~pretrain.train.EpochShuffleSampler` derives it from ``(seed, epoch)`` instead.

    The dihedral draw used to be excluded from this claim, because it lived in the workers and
    the parent could not checkpoint it. Since #104 it is a pure function of ``(seed, epoch,
    index)`` and resumes with everything else; the gate that measures *that* is
    :func:`test_kill_at_step_k_and_resume_reproduces_the_loss_trajectory_bit_identically`,
    which now runs with workers too.
    """
    workered = cfg.replace(data=cfg.data.replace(num_workers=2))
    uninterrupted = tiles_read(workered, steps=12)
    resumed = tiles_read(
        workered, steps=7, state=RunState(epoch=1, step=KILL_AT, batches_into_epoch=1)
    )
    assert resumed == uninterrupted[KILL_AT * BATCH :]


def test_resume_restores_the_model_the_optimizer_and_the_position(cfg, policy):
    killed = trajectory(cfg, steps=KILL_AT, policy=policy)
    loaded = ckpt_mod.load_checkpoint(ckpt_mod.latest(policy.directory).path)

    # The optimizer's AdamW moments are the piece a naive resume drops: a run that reloads the
    # weights and rebuilds the optimizer trains on, with the loss barely twitching.
    assert loaded.optimizer["state"], "no AdamW moments in the checkpoint"
    assert loaded.state.step == KILL_AT == len(killed)

    fresh = a_trainer(cfg)
    before = fresh.model.decoder_pred.weight.detach().clone()
    fresh.model.load_state_dict(loaded.model)
    assert not torch.equal(before, fresh.model.decoder_pred.weight)


def test_both_triggers_fire_in_one_run_the_epoch_and_the_wall_clock(cfg, tmp_path):
    # §11 asks for both — "every epoch *and* every 30-60 minutes of wall-clock" — because
    # either alone has a hole: an epoch can be hours long, and a wall-clock write alone would
    # leave the milestone epochs to luck.
    now = [0.0]
    policy = CheckpointPolicy.from_config(
        cfg,
        directory=tmp_path / "ckpt",
        interval_seconds=3.0,
        clock=lambda: now[0],
        keep_last=100,
    )
    # One simulated minute per step: the interval fires on steps 3 and 6, the epoch boundary
    # on 4 and 8, so the two triggers are distinguishable rather than coincident.
    trajectory(
        cfg,
        steps=STEPS_PER_EPOCH * 2,
        policy=policy,
        on_step=lambda record: now.__setitem__(0, record.step + 1.0),
    )
    written = ckpt_mod.discover(policy.directory)
    boundaries = {f.step for f in written if f.step % STEPS_PER_EPOCH == 0}
    assert boundaries == {4, 8}
    assert {f.epoch for f in written if f.milestone} == set()  # 1 and 2 are not milestones
    assert {f.step for f in written} - boundaries, "the wall-clock trigger never fired"


def test_a_round_trip_through_a_stand_in_for_the_off_instance_bucket(cfg, policy, tmp_path):
    """§15.8's sync check, minus the network. #80 owns the bucket; this owns what is synced.

    The instance vanishes, the directory goes with it, and the only surviving copy is the one
    that was pushed off-box. A checkpoint that only resumes beside the directory it was written
    in would pass every other test here and fail the one case §11 exists for.
    """
    uninterrupted = [r.loss for r in trajectory(cfg, steps=TOTAL)]
    trajectory(cfg, steps=KILL_AT, policy=policy)

    bucket = tmp_path / "bucket"
    shutil.copytree(policy.directory, bucket)
    shutil.rmtree(policy.directory)  # the instance is gone

    pulled = tmp_path / "new-instance"
    shutil.copytree(bucket, pulled)
    fresh_policy = CheckpointPolicy.from_config(cfg, directory=pulled, interval_seconds=0.0)
    resumed = trajectory(cfg, steps=TOTAL, policy=fresh_policy, resume=True)
    assert [r.loss for r in resumed] == uninterrupted[KILL_AT:]


def test_resuming_against_a_changed_corpus_is_refused_before_a_step_is_taken(
    cfg, policy, corpus, tmp_path
):
    # The mirror of the unit test above, through the loop: a resume that silently accepted a
    # different corpus would produce a released encoder whose recorded `dataset_version` names
    # data it was only partly trained on.
    trajectory(cfg, steps=KILL_AT, policy=policy)
    twin = tmp_path / "twin"
    shutil.copytree(corpus, twin)
    meta = twin / "corpus_meta.json"
    doc = json.loads(meta.read_text())
    doc["dataset_version"] = "not-the-corpus-this-run-trained-on"
    meta.write_text(json.dumps(doc))

    elsewhere = cfg.replace(data=cfg.data.replace(corpus_dir=str(twin)))
    with pytest.raises(ValueError, match="dataset_version"):
        trajectory(elsewhere, steps=TOTAL, policy=policy, resume=True)


def test_resume_against_an_empty_directory_starts_from_scratch_rather_than_raising(cfg, policy):
    # A first launch and a relaunch are the same command on the rented box (§11); asking for a
    # resume that has nothing to resume from is the normal first case, not an error.
    records = trajectory(cfg, steps=3, policy=policy, resume=True)
    assert [r.step for r in records] == [0, 1, 2]


def test_a_run_without_a_policy_trains_and_writes_nothing(cfg, policy):
    # Takes the *policy* fixture and then declines to pass it, so the directory it would have
    # written to is a real one that must stay empty. Asserting against a `tmp_path` the run was
    # never told about would pass no matter what the loop did.
    records = trajectory(cfg, steps=3, policy=None)
    assert [r.step for r in records] == [0, 1, 2]
    assert not policy.directory.exists()


def test_the_entrypoint_takes_a_checkpoint_directory_and_resumes_from_it(cfg, tmp_path, capsys):
    config_file = tmp_path / "run.json"
    config_file.write_text(json.dumps(cfg.to_dict()))
    args = ["--config", str(config_file), "--checkpoint-dir", str(tmp_path / "ckpt")]
    assert train_mod.main([*args, "--steps", "4", "--log-every", "1"]) == 0
    assert ckpt_mod.latest(tmp_path / "ckpt").step == 4

    capsys.readouterr()
    assert train_mod.main([*args, "--steps", "8", "--resume", "--log-every", "1"]) == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("step ")]
    assert len(lines) == 4  # steps 5..8: the run continued rather than restarting


def test_the_default_run_uses_wall_clock_time_rather_than_a_test_clock(cfg, tmp_path):
    policy = CheckpointPolicy.from_config(cfg, directory=tmp_path / "ckpt")
    assert policy.clock is time.monotonic
