"""The metrics sink: one interface, two payload kinds, and a W&B run with no account (#97).

Three claims are on trial here and only the first is obvious.

**The seam serves both consumers.** #98 emits reconstruction grids and #99 emits scalar probe
trends, and they are the reason this is a payload-typed interface rather than
``log(dict[str, float])``. A seam designed for scalars alone would send #98 around it, and then
the recon grids would reach W&B by a second path with its own step axis and its own tags.

**The tags are §15.7's four**, and they come off the config and the fingerprint rather than
being typed at a call site — including the case the shipped corpus was in for most of Phase 1,
where the split is not committed and the tag has to say so rather than go missing.

**Offline mode is a real run**, not a mock. The acceptance criterion is that the whole logging
path works with no account and no network, so one test drives `wandb.init(mode="offline")` for
real, logs all three payload kinds through it, and reads the transaction log back off disk.
"""

from __future__ import annotations

import numpy as np
import pytest

from pretrain import sink as sink_mod
from pretrain.config import RunConfig
from pretrain.fingerprint import ABSENT, COMMITTED, DataFingerprint
from pretrain.header import run_record
from pretrain.sink import Histogram, Image, MemoryLogger, WandbLogger, open_run_logger


def a_fingerprint(**changes) -> DataFingerprint:
    base = {
        "corpus_dir": "data/corpus_500k",
        "dataset_version": "bcbd6484d8116199c5c1160e21ea0284",
        "n_tiles": 499_934,
        "selection_list_md5": "1111111111111111111111111111aaaa",
        "split_md5": "2222222222222222222222222222bbbb",
        "split_blocks_md5": "3333333333333333333333333333cccc",
        "split_state": COMMITTED,
        "split_note": None,
        "git_sha": "0123456789abcdef0123456789abcdef01234567",
    }
    return DataFingerprint(**{**base, **changes})


# --- The payload types --------------------------------------------------------


def test_a_histogram_carries_counts_and_the_edges_they_fell_between():
    h = Histogram(counts=(1, 2, 3), edges=(0.0, 1.0, 2.0, 3.0))
    assert len(h.edges) == len(h.counts) + 1


def test_a_histogram_whose_edges_do_not_bound_its_counts_is_refused():
    # Off by one either way and W&B renders *something*, silently shifted by a bin.
    with pytest.raises(ValueError, match="edges"):
        Histogram(counts=(1, 2, 3), edges=(0.0, 1.0, 2.0))


def test_an_image_is_renderable_data_and_says_so_when_it_is_not():
    Image(np.zeros((8, 8), dtype=np.uint8))
    Image(np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8), caption="stratum 07")
    # #98 owns the metres -> colour mapping, so a NaN reaching here is a renderer bug, not a
    # model one. Left alone it renders as a black pixel and reads as a reconstruction failure.
    with pytest.raises(ValueError, match="finite"):
        Image(np.full((8, 8), np.nan, dtype=np.float32))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        Image(np.full((8, 8), 42.0, dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        Image(np.zeros((8, 8, 5), dtype=np.uint8))


# --- The interface, and its two consumers -------------------------------------


def test_an_image_caller_and_a_scalar_caller_share_one_logger_and_one_step_axis():
    # The seam the ticket asks for: #98's payload and #99's payload, through one interface, at
    # one epoch, with neither knowing about the other.
    logger = MemoryLogger()

    def probe_trends(log, epoch):  # #99-shaped
        log.log({"probe/stratum_knn_acc": 0.41, "probe/slope_mae": 2.7}, step=epoch)

    def recon_grids(log, epoch):  # #98-shaped
        log.log(
            {"recon/stratum_07": Image(np.zeros((8, 8), dtype=np.uint8), caption="masked")},
            step=epoch,
        )

    probe_trends(logger, 20)
    recon_grids(logger, 20)
    assert [step for step, _ in logger.history] == [20, 20]
    assert set(logger.at(20)) == {"probe/stratum_knn_acc", "probe/slope_mae", "recon/stratum_07"}


def test_the_step_axis_is_the_epoch_and_it_may_not_go_backwards():
    # W&B's step is one global counter, so #98's 10-epoch cadence and #99's 20-epoch cadence
    # have to share it. Going backwards drops the point silently on W&B's side; here it raises.
    logger = MemoryLogger()
    logger.log({"val/loss": 1.0}, step=3)
    logger.log({"val/loss": 0.9}, step=3)  # a second caller at the same epoch is fine
    with pytest.raises(ValueError, match="backwards"):
        logger.log({"val/loss": 0.8}, step=2)


def test_the_summary_holds_the_run_s_last_word_rather_than_a_trend():
    logger = MemoryLogger()
    logger.summary({"val/best_loss": 0.31, "alarms/fired": 2})
    assert logger.summaries == {"val/best_loss": 0.31, "alarms/fired": 2}


def test_the_logger_closes_itself_when_used_as_a_context_manager():
    with MemoryLogger() as logger:
        logger.log({"train/loss": 1.0}, step=0)
    assert logger.closed


# --- §15.7's tags -------------------------------------------------------------


def test_the_tags_are_the_four_the_preflight_checklist_names():
    cfg = RunConfig()
    tags = sink_mod.run_tags(cfg, a_fingerprint())
    assert tags == [
        "sha:0123456789abcdef0123456789abcdef01234567",
        f"config:{cfg.config_hash}",
        "selection:1111111111111111111111111111aaaa",
        "split:2222222222222222222222222222bbbb",
    ]


def test_an_uncommitted_split_tags_as_absent_rather_than_going_missing():
    # A run with no split tag and a run whose split is unfilled look identical in the W&B UI,
    # and the second one is the dangerous state (§9, #77). Absence is recorded, not skipped.
    tags = sink_mod.run_tags(
        RunConfig(), a_fingerprint(split_md5=None, split_state=ABSENT, split_note="not filled")
    )
    assert tags[-1] == "split:absent"


def test_a_tag_fits_in_what_wandb_will_accept():
    tags = sink_mod.run_tags(RunConfig(), a_fingerprint(git_sha="0" * 40 + "-dirty"))
    assert all(len(t) <= sink_mod.MAX_TAG_CHARS for t in tags)


def test_the_run_record_travels_with_the_run_so_the_box_holds_nothing_unique():
    # §11's requirement: the whole config and every fingerprint on the run page, not just tags.
    config = run_record(RunConfig(), a_fingerprint())
    assert config["config_hash"] == RunConfig().config_hash
    assert config["fingerprint"]["dataset_version"] == "bcbd6484d8116199c5c1160e21ea0284"
    assert config["config"]["monitoring"]["val_mask_seed"] == RunConfig().monitoring.val_mask_seed


# --- The factory --------------------------------------------------------------


def test_disabled_mode_needs_no_wandb_and_still_records_in_process():
    cfg = RunConfig().replace(monitoring=RunConfig().monitoring.replace(wandb_mode="disabled"))
    logger = open_run_logger(cfg, a_fingerprint())
    assert isinstance(logger, MemoryLogger)
    logger.log({"train/loss": 1.0}, step=0)
    assert logger.at(0)["train/loss"] == 1.0


def test_offline_is_the_default_mode_so_a_run_needs_no_account_to_report():
    assert RunConfig().monitoring.wandb_mode == "offline"


# --- A real offline run -------------------------------------------------------


@pytest.fixture
def offline_cfg(tmp_path):
    cfg = RunConfig()
    return cfg.replace(
        monitoring=cfg.monitoring.replace(
            wandb_mode="offline", wandb_project="demae-test", wandb_dir=str(tmp_path / "wb")
        )
    )


def test_the_whole_logging_path_runs_offline_with_no_account_and_no_network(offline_cfg):
    fingerprint = a_fingerprint()
    with open_run_logger(offline_cfg, fingerprint) as logger:
        assert isinstance(logger, WandbLogger)
        logger.log({"train/loss": 1.0, "val/loss": 1.02}, step=0)
        logger.log(
            {
                "val/per_patch_loss": Histogram(counts=(3, 5, 1), edges=(0.0, 0.5, 1.0, 1.5)),
                "recon/stratum_00": Image(np.zeros((16, 16), dtype=np.uint8), caption="epoch 1"),
                "throughput/samples_per_s": 3100.0,
            },
            step=1,
        )
        logger.summary({"val/best_loss": 1.02})
        run_dir = logger.run_dir

    # An offline run is a directory of transaction log + files, ready for `wandb sync`.
    assert run_dir.name.startswith("offline-run-")
    transactions = list(run_dir.glob("*.wandb"))
    assert len(transactions) == 1
    body = transactions[0].read_bytes()
    # The keys are written into the log verbatim, so this is a read-back of what was recorded
    # rather than a check that the call did not raise.
    for key in (b"train/loss", b"val/loss", b"val/per_patch_loss", b"throughput/samples_per_s"):
        assert key in body, key


def test_an_offline_run_carries_the_tags_and_the_config(offline_cfg):
    fingerprint = a_fingerprint()
    with open_run_logger(offline_cfg, fingerprint) as logger:
        assert logger.run.tags == tuple(sink_mod.run_tags(offline_cfg, fingerprint))
        assert logger.run.config["config_hash"] == offline_cfg.config_hash
        assert logger.run.name == offline_cfg.run_id


def test_the_offline_logger_refuses_a_step_that_goes_backwards(offline_cfg):
    with open_run_logger(offline_cfg, a_fingerprint()) as logger:
        logger.log({"train/loss": 1.0}, step=5)
        with pytest.raises(ValueError, match="backwards"):
            logger.log({"train/loss": 0.9}, step=4)
