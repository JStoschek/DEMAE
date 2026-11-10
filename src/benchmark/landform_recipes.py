"""Task 2's frozen recipe lines: one per head per rung, and the refusal of a cell that drifts (#206).

Plan §4.5: *"Winners are copied into one frozen config line per head per rung; a run whose
recipe differs from it is refused."* This module owns that file. #207's ``run --tune`` writes it
through :func:`write_line`; every ``run`` reads its own line through :func:`line_for` before it
takes a step, and :func:`check` refuses a cell whose recipe has moved since the line was written.

## The file

``recipe_lines.json`` in Task 2's ``paths.artifacts_dir``, committed. One JSON object::

    {
      "rule": "...plan §4.5's sentence...",
      "lines": {
        "<head>": {
          "<rung>": {
            "head": "linear", "rung": "1%",
            "lr": 0.001,                                   # the winner: the one tuned knob
            "recipe": {"optimizer": ..., "weight_decay": ..., "batch_size": ..., "epochs": ...},
            "schedule": {"step_floor": ..., "warmup_frac": ..., "decay": ..., "val_reads": ...},
            "tuned_on": {"arm": "B6", "seed": 0, "ladder": 0},
            "grid": [3e-4, 1e-3, ...],                     # what the pass searched
            "scores": {"0.001": 0.41, ...}                 # val mIoU per lr, provenance only
          }
        }
      }
    }

``recipe`` is the head's config group without ``lr_grid``, and ``schedule`` is §4.5's schedule
group without its tuning knobs: together, everything the cell trains under except the lr, which
the line supplies. A widened grid (§4.5's edge rule) is provenance (``grid``), not drift. Sorted
keys and a trailing newline, so a second write of the same line is byte-identical.

## Frozen means frozen

:func:`write_line` adds a line, or confirms one already there is identical; it refuses to replace
a different one. Re-tuning a rung is an amendment, and its first act is deleting that line by
hand, so the git history says it happened.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping

from .config import group_to_dict
from .landform_config import LandformConfig

#: The committed file, in Task 2's artifacts directory.
RECIPE_LINES_NAME: Final = "recipe_lines.json"

#: Which arm tunes each head's shared recipe (§4.5): the neutral encoder B6 for the probes, and
#: B7 for itself.
TUNING_ARMS: Final = {"linear": "B6", "upernet": "B6", "unet": "B7"}

#: The head-group knob that is searched rather than frozen.
SEARCHED: Final = ("lr_grid",)

#: §4.5's schedule knobs a cell trains under. The others (``tune_*``) say how tuning ran.
SCHEDULE_KNOBS: Final = ("step_floor", "warmup_frac", "decay", "val_reads")

RULE: Final = (
    "plan §4.5: winners are copied into one frozen config line per head per rung; a run whose "
    "recipe differs from it is refused"
)


#: Where each tuning pass's record goes, one file per head per rung, beside the lines file.
TUNING_DIR_NAME: Final = "tuning"


def path_for(cfg: LandformConfig) -> Path:
    return Path(cfg.paths.artifacts_dir) / RECIPE_LINES_NAME


def tuning_record_path_for(artifacts_dir: Path | str, head: str, rung: str) -> Path:
    """The record of the pass that froze ``head`` at ``rung``: every pass, point and read.

    Provenance, like Task 1's ``head_tuning.json``: nothing loads an lr from it. The line is the
    source; this says how the line was chosen and whether the grid had to widen.
    """
    return Path(artifacts_dir) / TUNING_DIR_NAME / f"{head}-{rung.replace('%', 'pct')}.json"


def assert_tuning_arm(arm: str, head: str) -> None:
    """Refuse a tuning pass for ``head`` on any arm but its tuning arm (§4.5, story 54).

    B6, the neutral encoder, tunes the linear probe and UperNet; B7 tunes itself. Any other arm
    tuning a shared recipe is the thumb on the scale the rule exists to prevent.
    """
    if head not in TUNING_ARMS:
        raise ValueError(f"unknown head {head!r}; the tuned heads are {list(TUNING_ARMS)}")
    want = TUNING_ARMS[head]
    if arm != want:
        raise ValueError(
            f"§4.5 tunes the {head} head on {want} alone, and this pass was asked for arm {arm}: "
            f"the {head} recipe is shared by every arm at a rung, so only "
            + ("the neutral encoder B6" if want == "B6" else "B7, on itself,")
            + " may choose it"
        )


def frozen_recipe(cfg: LandformConfig) -> dict[str, dict]:
    """What a line freezes about this cell besides its lr: the head recipe and the schedule."""
    recipe = {k: v for k, v in group_to_dict(cfg.recipe).items() if k not in SEARCHED}
    schedule = {k: v for k, v in group_to_dict(cfg.schedule).items() if k in SCHEDULE_KNOBS}
    return {"recipe": recipe, "schedule": schedule}


@dataclass(frozen=True)
class RecipeLine:
    """One head's frozen recipe at one rung."""

    head: str
    rung: str
    lr: float
    recipe: Mapping[str, Any]
    schedule: Mapping[str, Any]
    tuned_on: Mapping[str, Any]
    grid: tuple[float, ...] = ()
    scores: Mapping[str, float] | None = None

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> RecipeLine:
        return cls(
            head=d["head"], rung=d["rung"], lr=float(d["lr"]), recipe=dict(d["recipe"]),
            schedule=dict(d["schedule"]), tuned_on=dict(d["tuned_on"]),
            grid=tuple(d.get("grid", ())), scores=d.get("scores"),
        )

    def to_dict(self) -> dict:
        return {
            "head": self.head,
            "rung": self.rung,
            "lr": self.lr,
            "recipe": dict(self.recipe),
            "schedule": dict(self.schedule),
            "tuned_on": dict(self.tuned_on),
            "grid": list(self.grid),
            **({"scores": dict(self.scores)} if self.scores is not None else {}),
        }

    def line(self) -> str:
        t = self.tuned_on
        return (
            f"{'recipe line':<16} {self.head} @ {self.rung}: lr {self.lr:g}, tuned on "
            f"{t['arm']} (seed {t['seed']}, ladder {t['ladder']})"
        )


def _read(path: Path) -> dict:
    body = json.loads(path.read_text())
    if not isinstance(body.get("lines"), dict):
        raise ValueError(f"{path} is not a recipe-lines file: it has no `lines` object")
    return body


def lookup(cfg: LandformConfig) -> RecipeLine | None:
    """This cell's line, or ``None`` if the file or the line is not there yet. Not checked."""
    path = path_for(cfg)
    if not path.is_file():
        return None
    raw = _read(path)["lines"].get(cfg.head, {}).get(cfg.rung)
    return None if raw is None else RecipeLine.from_dict(raw)


def check(cfg: LandformConfig, line: RecipeLine) -> RecipeLine:
    """Refuse a cell whose recipe differs from its frozen line, naming every knob that moved."""
    want = TUNING_ARMS[cfg.head]
    if line.tuned_on.get("arm") != want:
        raise ValueError(
            f"the frozen recipe line for {cfg.head} @ {cfg.rung} says it was tuned on "
            f"{line.tuned_on.get('arm')}; §4.5 tunes the {cfg.head} head on {want} alone"
        )
    now = frozen_recipe(cfg)
    moved = [
        f"{group}.{knob} {now[group].get(knob)!r} (line: {frozen.get(knob)!r})"
        for group, frozen in (("recipe", line.recipe), ("schedule", line.schedule))
        for knob in sorted(set(frozen) | set(now[group]))
        if now[group].get(knob) != frozen.get(knob)
    ]
    if moved:
        raise ValueError(
            f"recipe drifted from its frozen line for {cfg.head} @ {cfg.rung}: "
            f"{'; '.join(m.replace('recipe.', f'{cfg.head}.', 1) for m in moved)}. {RULE}. "
            f"Re-tuning is an amendment (`run --task landform --tune`, #207), not a `--set`"
        )
    return line


def line_for(cfg: LandformConfig) -> RecipeLine:
    """This cell's frozen line, checked — or a refusal naming #207 if it is not frozen yet."""
    line = lookup(cfg)
    if line is None:
        raise FileNotFoundError(
            f"no frozen recipe line for {cfg.head} @ {cfg.rung} in {path_for(cfg)}: §4.5's "
            f"per-rung lr is tuned on {TUNING_ARMS[cfg.head]} and frozen by "
            f"`run --task landform --tune` (#207) before any cell runs"
        )
    return check(cfg, line)


def write_line(
    cfg: LandformConfig,
    *,
    lr: float,
    scores: Mapping[str, float] | None = None,
    grid: tuple[float, ...] | None = None,
) -> tuple[Path, RecipeLine]:
    """Freeze ``cfg``'s head and rung at ``lr`` — #207's writer.

    ``cfg`` is the tuning cell: its arm must be the head's tuning arm, and its seed and ladder
    §4.5's (``schedule.tune_seed``/``tune_ladder``). Idempotent for an identical line; refuses
    to replace a different one.
    """
    want = TUNING_ARMS[cfg.head]
    if cfg.arm != want:
        raise ValueError(
            f"arm {cfg.arm} cannot freeze the {cfg.head} recipe: §4.5 tunes it on {want}"
        )
    line = RecipeLine(
        head=cfg.head,
        rung=cfg.rung,
        lr=float(lr),
        **frozen_recipe(cfg),
        tuned_on={
            "arm": cfg.arm, "seed": cfg.schedule.tune_seed, "ladder": cfg.schedule.tune_ladder,
        },
        grid=tuple(cfg.recipe.lr_grid if grid is None else grid),
        scores=None if scores is None else dict(scores),
    )
    path = path_for(cfg)
    body = _read(path) if path.is_file() else {"rule": RULE, "lines": {}}
    existing = body["lines"].get(cfg.head, {}).get(cfg.rung)
    if existing is not None and existing != line.to_dict():
        raise ValueError(
            f"{path} already freezes {cfg.head} @ {cfg.rung} at lr {existing['lr']:g}, and this "
            f"pass would write lr {line.lr:g} or a different recipe. A frozen line is not "
            f"replaced: re-tuning is an amendment, whose first act is deleting that line"
        )
    body["lines"].setdefault(cfg.head, {})[cfg.rung] = line.to_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    return path, line
