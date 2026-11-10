"""`run --task landform --tune`: §4.5's per-head, per-rung lr pass and the lines it freezes (#207).

**Seam 1** is :func:`benchmark.launch.main`. On #206's fixture box with its recipe lines removed,
``--tune --head linear --rung 1%`` trains five log-spaced lr points on B6, each for a fifth of
the rung's schedule at seed 0 on ladder 0, scores each as the better of its two val reads, and
freezes the winner as one recipe line through :func:`benchmark.landform_recipes.write_line`.
The resolver's refusals (tuning arm, seed, an already-frozen line) are asserted there too.

**Seam 2** is :func:`benchmark.tune.search`, the edge rule generalised out of Task 1's pass: a
winner on a grid edge widens the grid one point that way and the whole pass reruns. It is a pure
function of a score callable, so the widening is tested against scores chosen by hand; seam 1
then checks the record says so, with the trainer standing in for a hand-chosen val curve.
"""

from __future__ import annotations

import json

import pytest

from benchmark import landform_recipes, landform_train, launch, tune
from tests.test_p206_linear_probe_cell import (  # noqa: F401  (fixtures)
    STEPS,
    _flags,
    box,
    corpus,
    stub,
)

torch = pytest.importorskip("torch")

LINEAR_GRID = (3e-4, 1e-3, 3e-3, 1e-2, 3e-2)


@pytest.fixture
def bare(box):
    """#206's box before any tuning pass: no recipe lines."""
    (box / "task2" / landform_recipes.RECIPE_LINES_NAME).unlink()
    return box


@pytest.fixture
def flags(bare, corpus):
    return _flags(bare, corpus)


def tune_argv(*argv: str) -> list[str]:
    return ["run", "--tune", "--head", "linear", "--rung", "1%", *argv]


def lines_of(box) -> dict:
    return json.loads((box / "task2" / landform_recipes.RECIPE_LINES_NAME).read_text())["lines"]


def record_of(box, head="linear", rung="1%") -> dict:
    return json.loads(landform_recipes.tuning_record_path_for(box / "task2", head, rung)
                      .read_text())


# --- Seam 1: the command ------------------------------------------------------


def test_tune_writes_one_recipe_line_from_five_log_spaced_points_on_b6(bare, flags, stub, capsys):
    assert launch.main([*tune_argv(), *flags]) == 0
    line = lines_of(bare)["linear"]["1%"]
    assert line["tuned_on"] == {"arm": "B6", "seed": 0, "ladder": 0}
    assert tuple(line["grid"][:5]) == LINEAR_GRID or tuple(line["grid"][-5:]) == LINEAR_GRID
    assert line["lr"] in line["grid"]
    assert len(line["scores"]) == len(line["grid"])
    assert set(lines_of(bare)) == {"linear"} and set(lines_of(bare)["linear"]) == {"1%"}

    record = record_of(bare)
    first = record["passes"][0]
    assert tuple(first["grid"]) == LINEAR_GRID
    # A fifth of the rung's ten steps, two val reads, each point scored as its better read.
    assert record["steps_per_point"] == STEPS // 5 and record["val_reads"] == 2
    for point in first["points"]:
        assert len(point["reads"]) == 2
        assert point["score"] == max(read["miou"] for read in point["reads"])
    # Each point checkpoints into its own directory rather than overwriting the last.
    checkpoints = {point["checkpoint"] for p in record["passes"] for point in p["points"]}
    assert len(checkpoints) == sum(len(p["points"]) for p in record["passes"])
    assert record["selected"] == line["lr"]
    assert "scored" in capsys.readouterr().out


class _Fitted:
    """What the tuning pass reads off a trained point: its val reads, checkpoint and extractor."""

    def __init__(self, reads, checkpoint):
        self.reads = reads
        self.checkpoint = checkpoint
        self.extractor = None
        self.seconds = 0.0


def curve_trainer(peak: float, calls: list):
    """A trainer whose val mIoU peaks at ``peak`` on a log scale — a hand-chosen val curve."""
    import math

    def train(cfg, *, lr, tiles, extractor=None, steps=None, val_reads=None,
              checkpoint_dir=None, **_):
        calls.append((lr, steps, val_reads, str(checkpoint_dir)))
        score = 1.0 / (1.0 + abs(math.log10(lr) - math.log10(peak)))
        reads = [landform_train.ValRead(step=1, miou=score / 2, seconds=0.0),
                 landform_train.ValRead(step=steps, miou=score, seconds=0.0)]
        return _Fitted(reads, str(checkpoint_dir / "best.pt"))

    return train


def test_an_edge_winner_widens_the_grid_one_point_reruns_the_whole_pass_and_says_so(
    bare, flags, stub, monkeypatch, capsys
):
    calls: list = []
    # Peaks between 3e-2 and 1e-1, nearer 3e-2: 3e-2 wins on the top edge, then inside.
    monkeypatch.setattr(landform_train, "train_linear", curve_trainer(0.05, calls))
    assert launch.main([*tune_argv(), *flags]) == 0

    record = record_of(bare)
    assert [tuple(p["grid"]) for p in record["passes"]] == [LINEAR_GRID, (*LINEAR_GRID, 0.1)]
    assert record["passes"][0]["edge"] == "high" and record["passes"][0]["widened"] == "high"
    assert record["passes"][1]["edge"] is None and record["widenings"] == 1
    # The whole pass reran: every point of the widened grid trained again, in its own directory.
    assert [lr for lr, *_ in calls] == [*LINEAR_GRID, *LINEAR_GRID, 0.1]
    assert len({where for *_, where in calls}) == len(calls)
    assert {(steps, reads) for _, steps, reads, _ in calls} == {(STEPS // 5, 2)}

    line = lines_of(bare)["linear"]["1%"]
    assert line["lr"] == 3e-2 and line["grid"] == [*LINEAR_GRID, 0.1]
    assert set(line["scores"]) == {f"{lr:g}" for lr in (*LINEAR_GRID, 0.1)}
    assert "widened" in capsys.readouterr().out


def test_a_low_edge_winner_widens_downwards(bare, flags, stub, monkeypatch):
    monkeypatch.setattr(landform_train, "train_linear", curve_trainer(2e-4, []))
    assert launch.main([*tune_argv(), *flags]) == 0
    assert lines_of(bare)["linear"]["1%"]["grid"] == [1e-4, *LINEAR_GRID]
    assert record_of(bare)["passes"][0]["widened"] == "low"


def test_tuned_lines_are_per_rung_and_a_cell_then_runs_at_its_rung_s_line(
    bare, flags, stub, monkeypatch, capsys
):
    real = landform_train.train_linear
    monkeypatch.setattr(landform_train, "train_linear", curve_trainer(3e-3, []))
    assert launch.main([*tune_argv(), *flags]) == 0
    monkeypatch.setattr(landform_train, "train_linear", curve_trainer(1e-3, []))
    assert launch.main(["run", "--tune", "--head", "linear", "--rung", "10%", *flags]) == 0
    lines = lines_of(bare)["linear"]
    assert (lines["1%"]["lr"], lines["10%"]["lr"]) == (3e-3, 1e-3)
    monkeypatch.setattr(landform_train, "train_linear", real)  # the scored cell trains for real
    capsys.readouterr()

    assert launch.main(["run", "--arm", "B1", "--rung", "1%", "--seed", "0", *flags]) == 0
    assert "recipe line linear @ 1%: lr 0.003" in capsys.readouterr().out


def test_a_run_whose_recipe_differs_from_its_tuned_line_is_refused(
    bare, flags, stub, monkeypatch
):
    monkeypatch.setattr(landform_train, "train_linear", curve_trainer(3e-3, []))
    assert launch.main([*tune_argv(), *flags]) == 0
    with pytest.raises(ValueError, match="recipe drifted from its frozen line.*weight_decay"):
        launch.main(["run", "--dry-run", "--rung", "1%", *flags,
                     "--set", "linear.weight_decay=0.5"])


def test_a_second_pass_over_a_frozen_line_is_refused_before_it_trains(
    bare, flags, stub, monkeypatch
):
    calls: list = []
    monkeypatch.setattr(landform_train, "train_linear", curve_trainer(3e-3, calls))
    assert launch.main([*tune_argv(), *flags]) == 0
    calls.clear()
    with pytest.raises(ValueError, match="already freezes linear @ 1%.*amendment"):
        launch.main([*tune_argv(), *flags])
    assert calls == []


@pytest.mark.parametrize(
    "argv, refusal",
    [
        (["--arm", "B1", "--head", "linear"], "tunes the linear head on B6 alone.*arm B1"),
        (["--arm", "B1-R4", "--head", "linear"], "on B6 alone.*arm B1-R4"),
        (["--arm", "B1-S"], "on B6 alone.*arm B1-S"),
        (["--arm", "B1", "--head", "upernet"], "tunes the upernet head on B6 alone.*arm B1"),
        (["--arm", "B6", "--head", "unet"], "tunes the unet head on B7 alone.*arm B6"),
    ],
)
def test_a_tuning_pass_on_any_arm_but_its_tuning_arm_is_refused(argv, refusal, flags, capsys):
    with pytest.raises(ValueError, match=refusal):
        launch.main(["run", "--dry-run", "--tune", "--rung", "1%", *argv, *flags])
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("head, ticket", [("upernet", "#208"), ("unet", "#209")])
def test_a_head_with_no_trainer_yet_names_its_ticket_on_its_tuning_arm(head, ticket, flags):
    with pytest.raises(NotImplementedError, match=ticket):
        launch.main(["run", "--dry-run", "--tune", "--head", head, "--rung", "1%", *flags])


def test_tuning_needs_a_head_and_runs_at_the_tuning_seed_alone(flags):
    with pytest.raises(ValueError, match="--head"):
        launch.main(["run", "--dry-run", "--tune", "--rung", "1%", *flags])
    with pytest.raises(ValueError, match="one seed.*--seed 1"):
        launch.main([*tune_argv("--dry-run", "--seed", "1"), *flags])


def test_tune_writes_nothing_of_task_one_s_and_scores_nothing(bare, flags, stub, monkeypatch):
    monkeypatch.setattr(landform_train, "train_linear", curve_trainer(3e-3, []))
    assert launch.main([*tune_argv(), *flags]) == 0
    assert not list(bare.rglob(tune.TUNING_NAME))
    assert not (bare / "results.parquet").exists()


# --- Seam 2: the edge rule ----------------------------------------------------


def _scores(table: dict[float, float], seen: list):
    def score(lr, pass_index):
        seen.append((pass_index, lr))
        return table[lr], {"lr": lr}

    return score


def test_search_returns_one_pass_when_the_winner_is_interior():
    seen: list = []
    passes = tune.search((1.0, 2.0, 4.0), _scores({1.0: 0.1, 2.0: 0.9, 4.0: 0.2}, seen),
                         maximize=True)
    assert len(passes) == 1 and passes[0].winner == 2.0 and passes[0].edge is None
    assert seen == [(0, 1.0), (0, 2.0), (0, 4.0)]


def test_search_widens_on_an_edge_and_reruns_every_point():
    seen: list = []
    table = {1.0: 0.1, 2.0: 0.2, 4.0: 0.5, 8.0: 0.9, 16.0: 0.3}
    passes = tune.search((1.0, 2.0, 4.0, 8.0), _scores(table, seen), maximize=True)
    assert [p.grid for p in passes] == [(1.0, 2.0, 4.0, 8.0), (1.0, 2.0, 4.0, 8.0, 16.0)]
    assert [p.edge for p in passes] == ["high", None]
    assert seen == [(0, 1.0), (0, 2.0), (0, 4.0), (0, 8.0),
                    (1, 1.0), (1, 2.0), (1, 4.0), (1, 8.0), (1, 16.0)]


def test_search_minimizing_takes_the_lowest_score_and_a_tie_goes_to_the_first_point():
    passes = tune.search((1.0, 2.0, 4.0, 8.0),
                         _scores({1.0: 5.0, 2.0: 1.0, 4.0: 1.0, 8.0: 3.0}, []), maximize=False)
    assert passes[-1].winner == 2.0


def test_search_refuses_an_optimum_it_cannot_bracket():
    def climbing(lr, pass_index):
        return lr, None

    with pytest.raises(tune.TuningError, match="not bracketed"):
        tune.search((1.0, 2.0, 4.0), climbing, maximize=True, max_widenings=2)


@pytest.mark.parametrize(
    "edge, want",
    [
        ("high", (3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)),
        ("low", (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)),
    ],
)
def test_widening_continues_the_grid_s_half_decade_spacing(edge, want):
    assert tune.widen(LINEAR_GRID, edge) == want
