"""§4.5's per-head, per-rung lr pass — Task 1's tuning discipline, run at every rung (#207).

Plan §4.5: *one knob, lr; 5 log-spaced points per head; each point trains the first fifth of its
rung's schedule at one seed on ladder 0, and scores as the better of its two val reads. ... A
winner on a grid edge widens the grid by one point that way and reruns the whole pass. Winners
are copied into one frozen config line per head per rung.*

What each clause is here:

* **The grid** is the head group's ``lr_grid``. It and every ``schedule.tune_*`` knob must be
  §4.5's committed values (:func:`check_tuning_knobs`), refused by name before the config is
  resolved further. A widened grid is provenance in the line (``grid``), never drift.
* **A fifth of the schedule** is ``ceil(cfg.steps x schedule.tune_fraction)`` steps, handed to
  the trainer as a parameter so the frozen line's recipe is the cell's own (see
  :func:`benchmark.landform_train.train_linear`). Warmup and cosine run over that shorter pass,
  as Task 1's 10 of 50 epochs did.
* **The better of two val reads** is :data:`VAL_READS` reads and the best of them.
* **The edge rule** is :func:`benchmark.tune.search`, which this module shares with nothing
  else yet and Task 1 does not call: Task 1's ``head_tuning.json`` and its refusal are unchanged.
* **The line** is written by :func:`benchmark.landform_recipes.write_line`, which refuses any arm
  but the head's tuning arm; the pass's record goes beside it
  (:func:`benchmark.landform_recipes.tuning_record_path_for`).

Only the linear probe trains today. UperNet is #208 and B7's UNet is #209; asking to tune
either names its ticket, after the resolver's tuning-arm refusal has had its say.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable, Final

from . import landform_recipes, landform_train, tune
from .landform_config import LandformConfig

#: Each point is scored as the better of this many val reads (§4.5).
VAL_READS: Final = 2

#: Heads whose trainer is not built yet, and the ticket that builds it.
PENDING_HEADS: Final = {"upernet": "#208", "unet": "#209"}


def assert_buildable(head: str) -> None:
    """Refuse a head no ticket has built a trainer for yet, naming its ticket."""
    if head in PENDING_HEADS:
        raise NotImplementedError(
            f"the {head} head has no trainer yet ({PENDING_HEADS[head]}), so `run --task "
            f"landform --tune --head {head}` cannot train its points; the linear probe tunes today"
        )


def tune_steps(cfg: LandformConfig) -> int:
    """How many steps one point trains: the first fifth (``tune_fraction``) of the cell's run."""
    return max(1, math.ceil(cfg.steps * cfg.schedule.tune_fraction))


def plan_tuning_knobs(cfg: LandformConfig) -> dict[str, Any]:
    """§4.5's tuning knobs for this cell's head, as the plan states them: the head group's
    ``lr_grid`` and every ``schedule.tune_*``, at their committed defaults."""
    schedule = type(cfg.schedule)()
    return {
        f"{cfg.head}.lr_grid": tuple(float(x) for x in type(cfg.recipe)().lr_grid),
        **{
            f"schedule.{f.name}": getattr(schedule, f.name)
            for f in fields(schedule) if f.name.startswith("tune_")
        },
    }


def check_tuning_knobs(cfg: LandformConfig) -> tuple[float, ...]:
    """Refuse a pass whose grid or ``tune_*`` knobs are not §4.5's, naming every one that moved.

    The frozen line records the grid it searched as provenance, so a ``--set`` grid or a
    ``tune_fraction`` of 1.0 would otherwise freeze a line that looks like §4.5's and is not.
    Changing how tuning runs is an amendment to the plan, not a flag. Returns the grid.
    """
    now = {
        f"{cfg.head}.lr_grid": tuple(float(x) for x in cfg.recipe.lr_grid),
        **{
            f"schedule.{f.name}": getattr(cfg.schedule, f.name)
            for f in fields(cfg.schedule) if f.name.startswith("tune_")
        },
    }
    moved = [f"{knob} {now[knob]!r} (§4.5: {want!r})"
             for knob, want in plan_tuning_knobs(cfg).items() if now[knob] != want]
    if moved:
        raise ValueError(
            f"a tuning pass runs on §4.5's tuning knobs, and this one moved {'; '.join(moved)}. "
            f"How lr is tuned is an amendment to the plan, not a `--set`"
        )
    return now[f"{cfg.head}.lr_grid"]


@dataclass(frozen=True)
class Tuned:
    """The whole pass for one head at one rung: every pass over the grid, and the winner."""

    cfg: LandformConfig
    passes: tuple[tune.GridPass, ...]
    steps_per_point: int
    seconds: float

    @property
    def final(self) -> tune.GridPass:
        return self.passes[-1]

    @property
    def selected(self) -> float:
        return self.final.winner

    @property
    def scores(self) -> dict[str, float]:
        """The final pass's val mIoU per lr, keyed as the line keys them."""
        return {f"{lr:g}": round(score, 6) for lr, score in zip(self.final.grid,
                                                                  self.final.scores)}

    def to_dict(self, *, line_path: Path | None = None, git_sha: str | None = None) -> dict:
        cfg = self.cfg
        return {
            "task": "landform",
            "head": cfg.head,
            "rung": cfg.rung,
            "arm": cfg.arm,
            "seed": cfg.seed,
            "ladder": cfg.ladder,
            "config_hash": cfg.config_hash,
            "base_grid": list(cfg.recipe.lr_grid),
            "full_steps": cfg.steps,
            "tune_fraction": cfg.schedule.tune_fraction,
            "steps_per_point": self.steps_per_point,
            "val_reads": VAL_READS,
            "score": f"the better of {VAL_READS} val mIoU reads",
            "passes": [
                {
                    "grid": list(p.grid),
                    "points": [dict(d) for d in p.details],
                    "winner": p.winner,
                    "edge": p.edge,
                    "widened": None if p is self.final else p.edge,
                }
                for p in self.passes
            ],
            "widenings": len(self.passes) - 1,
            "selected": self.selected,
            "recipe_line": None if line_path is None else str(line_path),
            "git_sha": git_sha,
            "seconds": round(self.seconds, 3),
        }

    def lines(self) -> list[str]:
        block = [f"{'tuned':<16} {self.cfg.head} @ {self.cfg.rung} on {self.cfg.arm}: "
                 f"{len(self.passes)} pass(es), {self.steps_per_point:,} steps per point, "
                 f"val only"]
        for p in self.passes:
            if p is not self.final:
                block.append(f"{'widened':<16} winner {p.winner:g} on the {p.edge} edge of "
                             f"{[f'{x:g}' for x in p.grid]}; widened one point and reran the "
                             f"whole pass (§4.5)")
        block.append(f"{'selected':<16} lr {self.selected:g}, val mIoU "
                     f"{self.final.scores[self.final.winner_index]:.4f}")
        return block


def run_pass(
    cfg: LandformConfig,
    *,
    tiles: landform_train.CellTiles,
    extractor=None,
    trainer: Callable[..., landform_train.Fitted] | None = None,
    on_point: Callable[[dict], None] | None = None,
) -> Tuned:
    """Train every point of the head's grid on the tuning cell, widening on an edge winner."""
    landform_recipes.assert_tuning_arm(cfg.arm, cfg.head)
    assert_buildable(cfg.head)
    if cfg.seed != cfg.schedule.tune_seed or cfg.ladder != cfg.schedule.tune_ladder:
        raise ValueError(
            f"§4.5 tunes at seed {cfg.schedule.tune_seed} on ladder {cfg.schedule.tune_ladder}; "
            f"this pass is seed {cfg.seed} on ladder {cfg.ladder}"
        )
    trainer = landform_train.train_linear if trainer is None else trainer
    grid = check_tuning_knobs(cfg)
    steps = tune_steps(cfg)
    root = Path(cfg.runtime.checkpoint_dir) / f"tune-{cfg.arm}-{cfg.head}-{landform_recipes.rung_tag(cfg.rung)}"
    shared = {"extractor": extractor}
    started = time.perf_counter()

    def score(lr: float, pass_index: int) -> tuple[float, dict[str, Any]]:
        fitted = trainer(
            cfg, lr=lr, tiles=tiles, extractor=shared["extractor"], steps=steps,
            val_reads=VAL_READS, checkpoint_dir=root / f"pass{pass_index}-lr{lr:g}",
        )
        shared["extractor"] = fitted.extractor  # one encoder load for the whole pass
        best = max(read.miou for read in fitted.reads)
        detail = {
            "lr": lr,
            "score": round(best, 6),
            "reads": [{"step": r.step, "miou": round(r.miou, 6)} for r in fitted.reads],
            "checkpoint": fitted.checkpoint,
            "seconds": round(fitted.seconds, 3),
        }
        if on_point is not None:
            on_point(detail)
        return round(best, 6), detail

    passes = tune.search(grid, score, maximize=True)
    return Tuned(cfg=cfg, passes=passes, steps_per_point=steps,
                 seconds=time.perf_counter() - started)


def freeze(tuned: Tuned, *, git_sha: str | None = None) -> tuple[Path, Path]:
    """Write the winner as the head's frozen line at this rung, then the pass's record."""
    cfg = tuned.cfg
    line_path, _ = landform_recipes.write_line(
        cfg, lr=tuned.selected, scores=tuned.scores, grid=tuned.final.grid
    )
    record = landform_recipes.tuning_record_path_for(cfg.paths.artifacts_dir, cfg.head, cfg.rung)
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(tuned.to_dict(line_path=line_path, git_sha=git_sha),
                                 indent=2, sort_keys=True) + "\n")
    return line_path, record
