"""The run header: what `python -m pretrain` prints, and what a checkpoint records (#89)."""

from __future__ import annotations

import json

import pytest

from pretrain import header
from pretrain.config import RunConfig
from pretrain.fingerprint import collect_fingerprint


@pytest.fixture
def cfg(split_corpus, tmp_path):
    base = RunConfig()
    return base.replace(
        data=base.data.replace(
            corpus_dir=str(split_corpus),
            split_assignment_path=str(tmp_path / "absent.csv"),
            split_blocks_path=str(tmp_path / "absent.json"),
        )
    )


# --- The record ---------------------------------------------------------------


def test_the_record_is_the_config_its_hash_and_the_fingerprint(cfg):
    record = header.run_record(cfg)
    assert record["config"] == cfg.to_dict()
    assert record["config_hash"] == cfg.config_hash
    assert record["fingerprint"]["dataset_version"]
    assert json.loads(json.dumps(record)) == record  # it has to survive a checkpoint


def test_the_record_round_trips_back_into_the_config_that_made_it(cfg):
    assert RunConfig.from_dict(header.run_record(cfg)["config"]) == cfg


def _fingerprint(cfg):
    """The fingerprint for a fixture corpus that has no committed split artifacts."""
    return collect_fingerprint(cfg.data.corpus_dir, split_assignment=None, split_blocks=None)


# --- The printed header -------------------------------------------------------


def test_the_header_states_the_four_tagged_facts(cfg):
    # plan §15.7: git SHA, config hash, selection-list md5, split md5 on every run
    text = header.format_header(cfg, _fingerprint(cfg))
    assert cfg.config_hash in text
    assert "test-selection-md5" in text
    assert "git sha" in text
    assert "split md5" in text


def test_the_header_states_the_derived_recipe_numbers(cfg):
    text = header.format_header(cfg, _fingerprint(cfg))
    assert "2048" in text  # effective batch
    assert "1.20e-03" in text  # the LR the scaling rule lands on
    assert "49" in text  # visible tokens at 75% of 196


# --- The command --------------------------------------------------------------


def test_it_runs_end_to_end_against_a_corpus(cfg, capsys):
    assert header.main(["--corpus", cfg.data.corpus_dir, "--blocks", cfg.data.split_blocks_path,
                        "--assignment", cfg.data.split_assignment_path]) == 0
    out = capsys.readouterr().out
    assert "R1" in out and "absent" in out  # this fixture has no committed split


def test_json_mode_emits_the_record_a_checkpoint_carries(cfg, capsys):
    header.main(["--corpus", cfg.data.corpus_dir, "--blocks", cfg.data.split_blocks_path,
                 "--assignment", cfg.data.split_assignment_path, "--json"])
    record = json.loads(capsys.readouterr().out)
    assert RunConfig.from_dict(record["config"]).config_hash == record["config_hash"]


def test_a_partial_config_file_moves_only_the_knob_it_names(cfg, tmp_path, capsys):
    path = tmp_path / "r2.json"
    path.write_text(json.dumps({"run_id": "R2", "masking": {"ratio": 0.9}}))
    header.main(["--corpus", cfg.data.corpus_dir, "--config", str(path), "--blocks",
                 cfg.data.split_blocks_path, "--assignment",
                 cfg.data.split_assignment_path, "--json"])
    loaded = RunConfig.from_dict(json.loads(capsys.readouterr().out)["config"])
    assert loaded.masking.ratio == 0.9 and loaded.run_id == "R2"
    assert loaded.schedule == RunConfig().schedule  # everything else is still the plan


def test_the_vit_base_preset_is_reachable_from_the_command(cfg, capsys):
    header.main(["--corpus", cfg.data.corpus_dir, "--vit-base", "--blocks",
                 cfg.data.split_blocks_path, "--assignment",
                 cfg.data.split_assignment_path, "--json"])
    loaded = RunConfig.from_dict(json.loads(capsys.readouterr().out)["config"])
    assert loaded == RunConfig.vit_base().replace(data=loaded.data)


def test_a_typo_in_a_config_file_stops_the_run(cfg, tmp_path):
    path = tmp_path / "typo.json"
    path.write_text(json.dumps({"masking": {"mask_ratio": 0.9}}))  # not the knob's name
    with pytest.raises(ValueError, match="mask_ratio"):
        header.main(["--corpus", cfg.data.corpus_dir, "--config", str(path)])


# --- `--set` overrides (#95) --------------------------------------------------


def test_an_override_moves_one_knob_and_leaves_the_rest_of_the_plan_alone():
    cfg = header.build_config(overrides=["masking.ratio=0.9"])
    assert cfg.masking.ratio == 0.9
    assert cfg == RunConfig().replace(masking=cfg.masking)


def test_an_override_reads_its_value_as_json_so_the_types_survive_the_shell():
    cfg = header.build_config(
        overrides=[
            "runtime.compile=false",
            "optimization.batch_size=512",
            "schedule.milestones=[50, 100]",
            "init.checkpoint=null",
            "data.corpus_dir=data/corpus_500k",
        ]
    )
    assert cfg.runtime.compile is False  # not the string "false", which is truthy
    assert cfg.optimization.batch_size == 512
    assert cfg.schedule.milestones == (50, 100)  # a list, restored to the field's tuple
    assert cfg.init.checkpoint is None
    assert cfg.data.corpus_dir == "data/corpus_500k"  # not valid JSON, so: a string


def test_a_top_level_knob_is_reachable_without_a_group():
    assert header.build_config(overrides=["run_id=R0-smoke"]).run_id == "R0-smoke"


def test_an_override_naming_a_knob_that_does_not_exist_stops_the_run():
    # The whole reason the config refuses unknown keys: a typo'd override would otherwise
    # report the recipe it was asked for and train the one it defaulted to.
    with pytest.raises(ValueError, match="batch"):
        header.build_config(overrides=["optimization.batch=512"])
    with pytest.raises(ValueError, match="optimisation"):
        header.build_config(overrides=["optimisation.batch_size=512"])


def test_an_override_that_is_not_a_key_value_pair_stops_the_run():
    with pytest.raises(ValueError, match="masking.ratio"):
        header.build_config(overrides=["masking.ratio"])


def test_an_override_path_deeper_than_a_group_and_a_knob_stops_the_run():
    with pytest.raises(ValueError, match="masking.ratio.value"):
        header.build_config(overrides=["masking.ratio.value=0.9"])


def test_overrides_are_applied_after_the_config_file_and_after_the_path_flags(tmp_path):
    # The documented precedence: preset, then the file, then the paths, then `--set` — the
    # general escape hatch wins, so a run can always be pinned from the command line alone.
    path = tmp_path / "r2.json"
    path.write_text(json.dumps({"masking": {"ratio": 0.9}, "run_id": "R2"}))
    cfg = header.build_config(
        config_path=path,
        corpus="from-the-flag",
        overrides=["masking.ratio=0.5", "data.corpus_dir=from-the-override"],
    )
    assert cfg.masking.ratio == 0.5 and cfg.run_id == "R2"
    assert cfg.data.corpus_dir == "from-the-override"


def test_an_override_on_the_vit_base_preset_keeps_the_preset(tmp_path):
    cfg = header.build_config(vit_base=True, overrides=["optimization.accum_iter=8"])
    assert cfg.architecture.encoder_dim == 768  # still B/16
    assert cfg.optimization.accum_iter == 8


def test_overrides_are_reachable_from_the_command_line(cfg, capsys):
    header.main(["--corpus", cfg.data.corpus_dir, "--blocks", cfg.data.split_blocks_path,
                 "--assignment", cfg.data.split_assignment_path,
                 "--set", "masking.ratio=0.9", "--json"])
    loaded = RunConfig.from_dict(json.loads(capsys.readouterr().out)["config"])
    assert loaded.masking.ratio == 0.9


def test_an_override_changes_the_config_hash_and_a_path_flag_does_not(cfg):
    plain = header.build_config(corpus=cfg.data.corpus_dir)
    assert header.build_config(corpus="somewhere/else").config_hash == plain.config_hash
    moved = header.build_config(corpus=cfg.data.corpus_dir, overrides=["masking.ratio=0.9"])
    assert moved.config_hash != plain.config_hash
