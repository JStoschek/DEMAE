"""The one tuning pass this benchmark is allowed, and the guard that keeps it to one (#159).

§4.1: *hyperparameters are tuned once, on A6, using val only, and then frozen for every arm*.
Three clauses, and each one is a mechanism here rather than a discipline:

**Once.** :func:`assert_frozen` refuses any `run` whose head recipe has drifted from
:data:`benchmark.config.TUNED_HEAD`. So a second tuning pass is not something a build session can
do by accident with a ``--set head.lr=`` on a cell that later reaches the parquet — it has to
move the frozen line and say so in a commit.

**On A6.** :data:`TUNING_ARM`. A6 is the *neutral* encoder: it is the row no C-claim rests on,
so a recipe that happens to suit it is not a recipe chosen to suit the arm we hope wins. Tuning
on A4 and then reporting C1 would be the classic thumb on the scale, and the refusal is worth
more than the comment.

**Using val only.** :class:`benchmark.train.TrainingTiles` is the only tile source the training
path opens and it cannot reach a test tile; :data:`benchmark.train.TRAINING_ARTIFACTS` is the
roster of committed files this path may read, with the test list and the eval mask list
conspicuously absent. Both are asserted by tests rather than argued for here.

## What is searched, and what is not

:data:`GRID`. Two knobs, six points — and the smallness is the argument rather than a
concession. §4.1 states a value for every knob in the optimizer block, and most of them are not
things a val number over 2,000 tiles can distinguish: a 1.08M-parameter head's weight decay, the
batch size and the epoch count are §7 budget decisions with their own arithmetic, and moving
them here would be re-deciding a priced line on the strength of a val RMSE. What is left is the
learning rate — the one knob whose wrong value produces a head that looks broken rather than
merely worse — and §4.1's own 0.1 slope weight, which is the one term in the loss the plan
introduces without a citation to a measurement.

``slope_weight = 0.0`` is in the grid deliberately. It is the null of §4.1's own claim that the
slope term is worth having, and a grid that could not select it would be evidence for that claim
constructed so as not to be able to disconfirm it.

## What this writes, and what it does not decide

``head_tuning.json``, beside the other committed artifacts: the grid, every point's val RMSE, and
the winner. It is a **record**, not a source: nothing loads a hyperparameter from it. The
selected point is folded into :data:`benchmark.config.TUNED_HEAD` by hand, in one committed line,
and :func:`assert_frozen` then refuses every later run that disagrees with that line — including
one whose committed tuning artifact says something else. That indirection is on purpose. A config
that read its recipe out of a file on the box would make the config hash a function of what
someone last ran, and §11's whole argument is that it must not be.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable, Final, Sequence

from . import arms, config, train
from .config import BenchmarkConfig, Head

#: :data:`benchmark.config.TUNED_HEAD` is reached **through the module** everywhere in this file
#: rather than imported by name, and that is not a style preference: the frozen recipe has one
#: home, and a second binding of it here is a second thing to move on the day the tuning pass
#: moves it. It is also what lets a test stand in a different frozen line, which is the only
#: honest way to exercise a guard whose real fix is a commit.
#:
#: The arm §4.1 tunes on, and the reason is in the module docstring.
TUNING_ARM: Final = "A6"

#: The committed record of the pass. Beside the bar, the tile lists, the spec, the eval mask list
#: and the bank sidecar — a seventh file in ``artifacts_dir``, and the only one `run` writes
#: rather than `freeze`.
TUNING_NAME: Final = "head_tuning.json"

#: How many epochs each grid point trains for. **Not** §4.1's 50: the pass is comparing six
#: recipes against each other rather than producing a head anything scores, and §7 prices the
#: whole GPU program at 2.42 hours with head training as thirteen 50-epoch cells. Ten epochs is
#: the first tenth of the same cosine — enough for a learning rate that is wrong by 3x to be
#: visibly wrong, and a twelfth of what a full grid would have cost. The winner is then trained
#: at the full 50 as A6's own primary cell, which is what ships with this ticket.
TUNING_EPOCHS: Final = 10

#: The two knobs the pass searches, and their values. See the module docstring on why these two.
GRID: Final[dict[str, tuple]] = {
    "lr": (3e-4, 1e-3, 3e-3),
    "slope_weight": (0.0, 0.1),
}


class TuningError(RuntimeError):
    """A tuning pass that would not be *the* tuning pass, or a run that has drifted from it."""


def points(grid: dict[str, tuple] | None = None) -> list[dict[str, Any]]:
    """The grid as a list of ``{knob: value}`` overlays, in a deterministic order.

    Ordered rather than a set, and the order is the declaration's, so two boxes that run this
    pass visit the six points in the same sequence and a tie is broken the same way on both.
    """
    grid = GRID if grid is None else grid
    out: list[dict[str, Any]] = [{}]
    for knob, values in grid.items():
        out = [{**point, knob: value} for point in out for value in values]
    return out


def recipe_for(point: dict[str, Any], base: Head | None = None) -> Head:
    """§4.1's frozen recipe with one grid point's knobs moved, and nothing else."""
    base = config.TUNED_HEAD if base is None else base
    unknown = sorted(set(point) - {f.name for f in fields(Head)})
    if unknown:
        raise TuningError(f"{unknown} are not knobs of §4.1's head recipe")
    return base.replace(**point)


# --- The guard ----------------------------------------------------------------


def assert_frozen(cfg: BenchmarkConfig) -> None:
    """Refuse a cell whose head recipe is not the one frozen for every arm.

    Called by `run` before it spends anything, for the arms that train a head. What it compares
    is :attr:`benchmark.config.Head.tuned` — the knobs the one pass may move — and not the whole
    object, so a smoke run at ``head.seeds=1`` passes and ``--set head.lr=3e-4`` does not. The
    difference is the point: seeds are how many observations §4.1 asks for, and the learning rate
    is the thing tuned once on the neutral arm.

    The refusal names the two ways out, and only one of them is fast. That asymmetry is
    deliberate — moving a hyperparameter for one arm is exactly the thing this benchmark
    pre-registered against, so the cheap path is to stop and the expensive path is to re-tune,
    re-freeze the line and say so.
    """
    moved = {
        name: (value, config.TUNED_HEAD.tuned[name])
        for name, value in cfg.head.tuned.items()
        if value != config.TUNED_HEAD.tuned[name]
    }
    if not moved:
        return
    detail = ", ".join(f"{n}={got!r} (frozen: {want!r})" for n, (got, want) in sorted(moved.items()))
    raise TuningError(
        f"this cell's head recipe has moved from the one frozen for every arm: {detail}. §4.1 "
        f"tunes once, on {TUNING_ARM}, using val only, and then freezes — so an arm with its own "
        f"learning rate is not a row this table can print beside the others. Either drop the "
        f"override, or re-run the tuning pass "
        f"(`python -m benchmark.launch run --arm {TUNING_ARM} --tune`) and move "
        f"`benchmark.config.TUNED_HEAD` in a commit that says what moved and why"
    )


def assert_tuning_arm(cfg: BenchmarkConfig) -> None:
    """Refuse a tuning pass on any arm but the neutral one.

    Beside :func:`assert_frozen` rather than inside :func:`tune`, because it is the same kind of
    fact — about the recipe, not about the box — and `run` asks both before it stats a file. A
    build session that typed `--arm A4-R5 --tune` should be told why that is not a thing, not
    told that a 340 MB checkpoint is missing.
    """
    if cfg.arm != TUNING_ARM:
        raise TuningError(
            f"§4.1 tunes on {TUNING_ARM} — the neutral encoder — and this pass was asked for "
            f"{cfg.arm}. Tuning on the arm a claim rests on is the thumb on the scale the rule "
            f"exists to prevent; the arm's own registered description says so "
            f"({arms.get(TUNING_ARM).what})"
        )


# --- The pass -----------------------------------------------------------------


@dataclass(frozen=True)
class Point:
    """One grid point and what val said about it."""

    knobs: dict[str, Any]
    val_rmse: float
    best_epoch: int
    seconds: float

    def to_dict(self) -> dict:
        return {
            "knobs": dict(self.knobs),
            "val_rmse": round(float(self.val_rmse), 6),
            "best_epoch": self.best_epoch,
            "seconds": round(float(self.seconds), 3),
        }

    def line(self) -> str:
        knobs = " ".join(f"{k}={v!r}" for k, v in sorted(self.knobs.items()))
        return f"{knobs:<34} val_rmse {self.val_rmse:9.4f} m  (epoch {self.best_epoch})"


@dataclass(frozen=True)
class Tuning:
    """The whole pass: every point, the winner, and what the frozen recipe currently says.

    ``agrees`` is the sentence a build session needs and the one a reader of the artifact needs:
    did the search select what :data:`benchmark.config.TUNED_HEAD` already holds? A `True` is the
    frozen line surviving its own search, which is a result. A `False` is a one-line edit owed
    to the config, and this class says so rather than making it.
    """

    arm: str
    epochs: int
    grid: dict[str, tuple]
    searched: tuple[Point, ...]
    selected: dict[str, Any]
    seconds: float

    @property
    def winner(self) -> Point:
        """The point :attr:`selected` names — the lowest val RMSE, ties to the earliest point."""
        return min(
            self.searched,
            key=lambda p: (p.val_rmse, self.searched.index(p)),
        )

    @property
    def agrees(self) -> bool:
        """Does the search select the recipe that is already frozen?"""
        return all(config.TUNED_HEAD.tuned[name] == value for name, value in self.selected.items())

    def to_dict(self) -> dict:
        return {
            "arm": self.arm,
            "epochs": self.epochs,
            "grid": {name: list(values) for name, values in self.grid.items()},
            "searched": [point.to_dict() for point in self.searched],
            "selected": dict(self.selected),
            "frozen": {name: config.TUNED_HEAD.tuned[name] for name in self.selected},
            "agrees_with_frozen": self.agrees,
            "seconds": round(float(self.seconds), 3),
            "reads": sorted(train.TRAINING_ARTIFACTS),
        }

    def render(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2) + "\n"

    def lines(self) -> list[str]:
        """What the console gets: every point, then the winner and whether it moves anything."""
        block = [f"{'tuned':<16} {self.arm} over {len(self.searched)} points, val only"]
        block += [f"{'':<16} {point.line()}" for point in self.searched]
        winner = self.winner
        block.append(f"{'selected':<16} {winner.line()}")
        if self.agrees:
            block.append(
                f"{'frozen':<16} unchanged — the search selects what `config.TUNED_HEAD` "
                f"already holds"
            )
        else:
            moved = ", ".join(
                f"{name}: {config.TUNED_HEAD.tuned[name]!r} -> {value!r}"
                for name, value in sorted(self.selected.items())
                if config.TUNED_HEAD.tuned[name] != value
            )
            block.append(
                f"{'frozen':<16} OWED — move `benchmark.config.TUNED_HEAD` ({moved}) in a "
                f"commit, and every later arm loads the moved one"
            )
        return block


def tuning_path(cfg: BenchmarkConfig) -> Path:
    return Path(cfg.paths.artifacts_dir) / TUNING_NAME


def write_tuning(cfg: BenchmarkConfig, tuning: Tuning) -> Path:
    """Write the record of the pass, replacing whatever was there.

    **Not through** :func:`benchmark.artifacts.write_committed`, which every input `freeze` owns
    goes through, and the reason is the same one :class:`benchmark.masks.BankSidecar` gives for
    the same exemption: that function's contract is *this file already holds what I would write,
    or it is a refusal*, and this file cannot hold that. Its content is a set of measured val
    RMSEs, and GPU training is not bit-reproducible — a second run of the identical pass writes
    the same winner and different sixth decimals, which `write_committed` would read as a moved
    artifact and refuse.

    So "tuned once" is not enforced here. It is enforced by :func:`assert_frozen`, against
    :data:`benchmark.config.TUNED_HEAD`, which is a committed line of Python rather than a file
    on a box — see the module docstring on why the recipe is never loaded from this artifact.
    """
    path = tuning_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tuning.render(), encoding="utf-8")
    return path


def read_tuning(path: Path | str) -> dict:
    """The committed record, as it was written. Nothing loads a hyperparameter from it."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} has not been written: §4.1's one tuning pass is "
            f"`python -m benchmark.launch run --arm {TUNING_ARM} --tune`"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def tune(
    cfg: BenchmarkConfig,
    *,
    grid: dict[str, tuple] | None = None,
    epochs: int = TUNING_EPOCHS,
    on_point: Callable[[Point], None] | None = None,
) -> Tuning:
    """Train §4.1's head once per grid point on A6, select on val RMSE, and say what won.

    Every point trains through :func:`benchmark.train.train_head`, which is the same function a
    scored cell uses — the same composition, the same bank, the same split guard. A tuning pass
    with its own training loop would be a second recipe to keep in step with the first, and the
    thing being selected here is precisely a recipe.
    """
    assert_tuning_arm(cfg)
    started = time.perf_counter()
    grid = GRID if grid is None else grid
    searched: list[Point] = []
    for knobs in points(grid):
        at = time.perf_counter()
        trained = train.train_head(
            cfg, arms.get(cfg.arm), recipe=recipe_for(knobs), epochs=epochs
        )
        point = Point(
            knobs=knobs,
            val_rmse=trained.best_val_rmse,
            best_epoch=trained.best_epoch,
            seconds=time.perf_counter() - at,
        )
        searched.append(point)
        if on_point is not None:
            on_point(point)

    # The winner is the lowest val RMSE, ties broken by the grid's own declared order — so two
    # boxes that measure the same tie select the same point rather than whichever finished first.
    won = min(searched, key=lambda p: (p.val_rmse, searched.index(p)))
    return Tuning(
        arm=cfg.arm,
        epochs=epochs,
        grid={name: tuple(values) for name, values in grid.items()},
        searched=tuple(searched),
        selected=dict(won.knobs),
        seconds=time.perf_counter() - started,
    )


def format_frozen(recipe: Head | None = None) -> Sequence[str]:
    """The frozen recipe as the resolution line prints it — every knob the pass may move."""
    recipe = config.TUNED_HEAD if recipe is None else recipe
    knobs = ", ".join(f"{name}={value!r}" for name, value in recipe.tuned.items())
    return [
        f"{'head recipe':<16} {knobs}",
        f"{'':<16} tuned once on {TUNING_ARM} using val only, then frozen for every arm (§4.1)",
    ]
