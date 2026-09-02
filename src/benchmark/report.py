"""`report` — the printed table, the C4 rank block and the three figures (#169).

A **pure function of the results parquet** (:mod:`benchmark.results`) and of the run records
beside it. Re-rendering the table never re-runs an arm, and nothing here opens an encoder, a
checkpoint or the train mask bank. That is #136's "computed and printed are different
questions" arriving at its second half: the parquet holds every metric on every axis, and this
module prints the deliberate subset R5 fixed **before any result existed**.

## What prints, and why the shape was fixed in advance

A table shape chosen after the numbers is a table shape chosen to flatter them — the same logic
that put C1-C4 in §0 before an arm ran. So the shape is read off the registered bar rather than
typed here (:class:`benchmark.prereg.Bar`), and the render asserts itself against R5's own
arithmetic on the way out: **66 (row, band) cells x 6 columns = 396 numbers**.

Two axes come off the page and the reason is measured, not aesthetic. p119 found the classical
baseline's error varying **~40x across the four ruggedness bands and only ~3x across the three
size buckets**, so the buckets pool and the bands stay whole. And the **rectangle family is not
printed**: it is a sensor-dropout proxy that no C-claim mentions, still drawn, still scored and
still in the parquet, but a family carrying no claim should not cost half the page width.

**The four appendix metrics print nowhere.** MAE, p95 and profile curvature were graded into the
appendix *because* they were redundant or could not separate the A1 variants, and eroded-mask
slope is published by arithmetic rather than by column (the seam gap is full-minus-eroded and
the full slope is headline). Printing any of them would re-import the exact noise the
24,000-instance grading in `p123-metric-discrimination.md` was run to remove.

## C4 is a rank block because a caption cannot stop a subtraction

The real family is a natural-distribution family and synthetic small is a bucket-balanced one,
so a difference between them in metres is not a quantity. §0 makes C4 an **ordering** claim for
exactly that reason, and three columns of metres side by side would invite the subtraction
whatever the caption said. So :class:`RankCell` carries a rank and an instance count and **no
metre value at all** — the metres are local to the ranking and are not reachable from the block.

## The bootstrap resamples tiles

~8 real voids share one host tile's terrain (#153 measured 8,824 real-small instances on 1,111
clean R4 test tiles), so a per-instance resample would treat eight readings of one landscape as
eight independent observations and overstate precision by up to sqrt(8). The cluster is the test
tile, and every printed interval carries both numbers — the tiles it drew from and the
instances that sat on them — so the claim is checkable off the page rather than asserted
here. A pre-registered pair's interval is drawn on *that pair's* shared tiles, never on an
intersection over every arm the box happens to have run.

**Exactly four pairs carry a CI mark**, named in R6 before any number existed, of the 55 that
C(11, 2) enumerates. The other 51 are computed and left unmarked — which comparisons matter is
fixed by the claims, not by which ones happen to clear zero.

*One registered name needs resolving.* R6 names **A3 vs A1**, and the registry has no bare `A3`
— §2.3 splits A3 by init. R3 settles it without a new decision (*R5 is the headline
unconditionally*), so an unsuffixed name resolves to its R5 row and :func:`resolve_pair` prints
the resolution rather than performing it silently. That is a *reading* of two registered rules
against each other, not an amendment to either.

## Why the resample count is a constant and not a knob

§5 registers "10k resamples". A box able to move it could narrow a printed CI by re-rendering,
which is the same failure the pre-registration exists to close — so it is :data:`BOOTSTRAP_
RESAMPLES` here, beside the seed that makes two renders of one parquet agree to the digit.

## The figures, and the one thing this module reads that is not the parquet

All three are marked **load-bearing** and it is not decoration. RMSE-vs-void-size is the *only*
place void size appears now that the buckets pool, so it is the whole of C2's "at every void
size". The label-efficiency curve is the whole of C3. The hillshade panel stands in for the
structural metrics §5 cut — no perceptual metric is scored, so the panel is the whole of the
visual-quality evidence.

The panel is the exception to purity, and it is bounded: a hillshade needs **pixels**, and a
metrics parquet holds none. So it reads the corpus for truth, the committed eval mask list for
the void, and a per-arm *fill exhibit* beside the results file for each arm column — and where a
box holds none of those, the cell renders as a labelled absence naming what is missing rather
than the figure failing or, worse, an arm being re-run to fill it in. Nothing here ever computes
a fill.
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final, Iterator, Mapping, Sequence

import numpy as np

from . import arms, masks, metrics, prereg, results, voids
from .config import LABEL_EFFICIENCY_SIZES, SIZE_TILES, BenchmarkConfig, Masks

#: The default mask geometry, read once for the family names and bucket labels below. A
#: *config* rather than a constant is what registers them (:class:`benchmark.config.Masks`),
#: and :func:`assert_r5_shape` refuses a run whose config disagrees with these.
_MASKS: Final = Masks()

#: The size the printed table is at — §3.1's largest nested subset, derived rather than typed.
#: The other two label-efficiency subsets are C3's sweep and print in a figure and nowhere in
#: the table (§9).
HEADLINE_SIZE: Final = LABEL_EFFICIENCY_SIZES[-1]

#: §5's headline family, and §3.3's. Both come off :class:`benchmark.config.Masks`, which
#: :mod:`benchmark.masks` already states is where a family's *name* lives — a second spelling
#: here is a box that could override the config and still render a table labelled `blob`.
#: :func:`assert_r5_shape` re-checks the headline one against R5's own registered value, so a
#: drift between the register and this module is a refusal rather than a quiet relabelling.
BLOB_FAMILY: Final = _MASKS.synthetic_families[0]
REAL_FAMILY: Final = _MASKS.real_family

#: ``R1``..``R4``, from :class:`benchmark.voids.Bands`' own labelling rather than typed again.
BANDS: Final[tuple[str, ...]] = tuple(
    f"R{i + 1}" for i in range(voids.N_BANDS)
)

#: §3.3's two published sub-rows, derived from §3.2's buckets exactly as `voids` derives them.
#: The count-only 128-160 row is not here: it carries a count and no metric by construction, so
#: it is a footnote rather than one of R5's 66 cells.
REAL_SUB_ROWS: Final[tuple[str, ...]] = tuple(
    row.name for row in voids.size_rows() if row.carries_metric
)

#: The band the real family is printed in. §3.3's realised histogram is 99.94% R4 (#135), so the
#: family has one ruggedness column and a footnote rather than four columns.
REAL_BAND: Final = "R4"

#: A **section** is §5's *row* — the blob family, or one of §3.3's two real sub-rows. It is
#: called `section` throughout and not `row` because this module's other row is a parquet row
#: (:func:`cell_rows` returns those), and one word for two things in a module that joins them is
#: the confusion worth a longer name.
#:
#: The blob family over four bands, then §3.3's two sub-rows in R4 alone: 11 arms x (4+1+1) = 66.
SECTIONS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    (BLOB_FAMILY, BANDS),
    *((sub, (REAL_BAND,)) for sub in REAL_SUB_ROWS),
)

#: C4's two metrics. RMSE and slope RMSE, as §5 names them — and the block prints their *ranks*.
C4_METRICS: Final = ("rmse", "slope_rmse")

#: The metrics a bootstrap CI is computed on, which are C4's two for the same reason: they are
#: the two §5's statistics paragraph names.
CI_METRICS: Final = C4_METRICS

#: Every arm pair the bootstrap enumerates — C(11, 2) = 55. Four of them are marked in print.
ALL_PAIRS: Final[tuple[tuple[str, str], ...]] = tuple(
    itertools.combinations(arms.PRINTED_ROWS, 2)
)

#: §5's registered resample count, and the seed that makes two renders of one parquet agree.
#: Constants rather than config knobs — see the module docstring.
BOOTSTRAP_RESAMPLES: Final = 10_000
BOOTSTRAP_SEED: Final = 20260901
CI_LEVEL: Final = 0.95

#: How many resamples are held in memory at once. The weight matrix is ``(chunk, tiles)``, so
#: 500 x 5,000 test tiles is 20 MB — the whole 10,000 at once would be 400 MB per cell.
_RESAMPLE_CHUNK: Final = 500

#: §9's panel: truth, the voided input, and four arms. ``A3`` is §9's word and resolves through
#: :func:`resolve_arm` for R6's reason.
PANEL_COLUMNS: Final = ("truth", "voided", "A1", "A3", "A5", "A7")

#: Two rugged tiles and one flat one (§9). Which ones is chosen from the parquet, deterministically.
PANEL_BANDS: Final = ("R4", "R4", "R1")

#: How many bins the RMSE-vs-void-size figure cuts the equivalent-diameter axis into. Finer than
#: §3.2's three buckets on purpose: the figure exists because *"a monotonicity claim reads better
#: off a curve than off three columns"*, and three points is not a curve.
SIZE_BINS: Final = 8


# --- Resolving a registered name against the registry -------------------------


def resolve_arm(name: str) -> str:
    """A registered arm name as the registry spells it, or a refusal saying why it cannot be.

    R6 names **A3 vs A1** and §9's panel names **A3**, and the registry has no bare ``A3``:
    §2.3 splits A2, A3 and A4 by init. R3 settles which one is meant without deciding anything
    new — *R5 is the headline unconditionally* — so an unsuffixed name resolves to its ``-R5``
    row. A name that is neither registered nor resolvable that way is refused rather than
    guessed at, because a silently wrong resolution here mismarks a pre-registered comparison.
    """
    if name in arms.PRINTED_ROWS:
        return name
    candidate = f"{name}-R5"
    if candidate in arms.PRINTED_ROWS:
        return candidate
    raise ValueError(
        f"{name!r} is not one of R5's printed rows {list(arms.PRINTED_ROWS)}, and R3's "
        f"resolution ({name}-R5) is not either. A registered comparison naming an arm this "
        f"table has no row for is a drift between the register and `benchmark.arms`"
    )


def resolve_pair(pair: Sequence[str]) -> tuple[str, str]:
    """R6's pair, both names resolved. Order is kept: the first name is what the claim is about."""
    first, second = pair
    return resolve_arm(first), resolve_arm(second)


def marked_pairs(bar: prereg.Bar) -> tuple[tuple[str, str], ...]:
    """R6's four, resolved and in the order they were registered."""
    return tuple(resolve_pair(pair) for pair in bar.bootstrap_pairs)


def mark_order(bar: prereg.Bar) -> dict[tuple[str, str], int]:
    """``pair -> 1..4``, R6's registration order. One map, so a mark and its legend agree."""
    return {pair: i + 1 for i, pair in enumerate(marked_pairs(bar))}


def resolutions(bar: prereg.Bar) -> tuple[str, ...]:
    """Every registered name this module had to resolve, as a line each.

    Printed rather than performed quietly: a reader diffing the marked pairs against R6 sees
    `A3-R5` where the register says `A3`, and is owed the rule that turned one into the other.
    """
    out = []
    for registered, resolved in zip(bar.bootstrap_pairs, marked_pairs(bar)):
        for was, now in zip(registered, resolved):
            if was != now:
                out.append(
                    f"R6 registers `{was}`, which §2.3 splits by init and the arm registry "
                    f"therefore has no row for. Resolved to `{now}` under R3 — *R5 is the "
                    f"headline unconditionally* — rather than by choosing after the numbers"
                )
    return tuple(dict.fromkeys(out))


# --- The parquet, as arrays ---------------------------------------------------


@dataclass(frozen=True)
class Frame:
    """The results parquet as columns of arrays — everything this module is a function of."""

    columns: dict[str, np.ndarray]
    path: Path
    #: ``tile_id`` as a dense integer code, so a per-tile sum is one :func:`numpy.bincount`
    #: rather than a Python loop over 45,000 rows per cell.
    tile_code: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.intp))
    tile_names: tuple[str, ...] = ()
    #: One boolean array per distinct ``(column, value)``, built on first use.
    #:
    #: The table asks for the same 27 or so predicates thousands of times over — eleven arms,
    #: four bands, six family/bucket labels, three sizes — and the columns are arrays of Python
    #: strings, where a single ``==`` walks the whole file at interpreter speed. Caching turns
    #: the render's cost from *predicates x rows* into *distinct values x rows*.
    _masks: dict = field(default_factory=dict, compare=False, repr=False)
    #: One index array per ``(arm, section, band, size)`` cell — see :func:`cell_rows`.
    _cells: dict = field(default_factory=dict, compare=False, repr=False)

    @property
    def n(self) -> int:
        return int(len(self.columns["arm"]))

    @property
    def n_tiles(self) -> int:
        return len(self.tile_names)

    def mask(self, column: str, value: Any) -> np.ndarray:
        cached = self._masks.get((column, value))
        if cached is None:
            cached = self.columns[column] == value
            self._masks[(column, value)] = cached
        return cached

    def select(self, **equals: Any) -> np.ndarray:
        """A boolean row mask. ``family``/``bucket`` are labels, so equality is the whole test."""
        keep = np.ones(self.n, dtype=bool)
        for name, value in equals.items():
            keep = keep & self.mask(name, value)
        return keep


def read(path: Path | str) -> Frame:
    """The results parquet, refusing a box that has none rather than printing an empty table."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"there are no results at {path}. `report` renders what `run` scored and never "
            f"scores anything itself, so a box with no results parquet has nothing to render — "
            f"run at least one cell first (`python -m benchmark.launch run --arm A1`)"
        )
    table = results.read_rows(path)
    columns = {
        name: table.column(name).to_numpy(zero_copy_only=False) for name in table.schema.names
    }
    names, codes = np.unique(columns["tile_id"], return_inverse=True)
    return Frame(
        columns=columns,
        path=path,
        tile_code=codes.astype(np.intp),
        tile_names=tuple(names.tolist()),
    )


def instances_and_tiles(frame: Frame, rows: np.ndarray) -> tuple[int, int]:
    """How many **instances** and how many **test tiles** a set of parquet rows covers.

    Neither is ``len(rows)``, and the difference is load-bearing twice over.

    *Instances, not rows.* A cell pools every seed that scored it, so an arm carrying three seeds
    has three rows per instance. R8's floor is stated in *instances* — §5 counts "1,250 per (row,
    band)" from 5,000 test tiles x 2 families x 3 buckets, which the part-ocean tiles
    `masks.scoreable_tiles` drops take to 1,111-1,158, and either way it is one pass — so
    counting rows would let a three-seed arm clear a floor on 34 instances a single pass fails
    on 99.
    An instance is identified within a cell by ``(bucket, tile_id, instance)``: the bucket is in
    the key because the blob section pools three of them and the same tile carries one void in
    each.

    *Tiles, not instances.* The bootstrap's cluster is the tile, and these two numbers side by
    side are what make that visible rather than asserted.
    """
    if rows.size == 0:
        return 0, 0
    tiles = frame.tile_code[rows]
    keys = set(
        zip(
            frame.columns["bucket"][rows].tolist(),
            tiles.tolist(),
            frame.columns["instance"][rows].tolist(),
        )
    )
    return len(keys), len(set(tiles.tolist()))


def printed_cells() -> Iterator[tuple[str, str, str]]:
    """Every ``(arm, section, band)`` R5 prints, in printed order. 66 of them.

    One walk, so the table, the seed spread and the bootstrap cannot disagree about which cells
    the printed table has — three copies of a nested loop is three places for R5's shape to
    drift, which is the defect ADR-0013 exists to prevent one document up.
    """
    for arm in arms.PRINTED_ROWS:
        for section, bands in SECTIONS:
            for band in bands:
                yield arm, section, band


def cell_rows(frame: Frame, arm: str, section: str, band: str, *, size: str) -> np.ndarray:
    """The **row indices** of one (row, band) cell — the one place the two vocabularies meet.

    ``family`` and ``bucket`` mean different things on the two halves of the eval mask list: a
    synthetic row's bucket is §3.2's ``small``/``medium``/``large`` and a real row's is one of
    §3.3's sub-rows. So the blob section pools **every** bucket within a band, and a real section
    *is* a bucket. Anything that grouped on ``bucket`` without splitting on ``family`` first
    would pool two axes that do not mean the same thing.

    Indices rather than a boolean mask, and memoized on the frame, because the same 726 cells are
    selected three times over — once to average, once to bootstrap, once for the seed spread —
    and re-deriving each one is the difference between a render that is instant and one that is
    not.
    """
    key = (arm, section, band, size)
    cached = frame._cells.get(key)
    if cached is None:
        if section == BLOB_FAMILY:
            keep = frame.select(arm=arm, size=size, family=BLOB_FAMILY, band=band)
        else:
            keep = frame.select(
                arm=arm, size=size, family=REAL_FAMILY, bucket=section, band=band
            )
        cached = np.flatnonzero(keep)
        frame._cells[key] = cached
    return cached


# --- The printed table --------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One (row, band) cell: six numbers, or a count and the rule that withheld them."""

    arm: str
    section: str
    band: str
    #: **Instances**, not parquet rows — see :func:`instances_and_tiles`. R8's floor is read
    #: against this, and it is the ``n=`` a withheld cell prints.
    count: int
    #: Distinct test tiles those instances sit on. The bootstrap's cluster population.
    tiles: int
    #: The six headline metrics, or ``None`` when R8's floor fired or the arm has not been run.
    values: dict[str, float] | None
    #: Why there is no metric. Empty when there is one.
    note: str = ""
    #: Which of R6's four marked pairs this cell is the claim-side row of, and on which metric.
    marks: tuple[str, ...] = ()


@dataclass(frozen=True)
class Table:
    """R5's printed table: 66 cells, six columns, and the footnotes the family owes."""

    columns: tuple[str, ...]
    cells: tuple[Cell, ...]
    footnotes: tuple[str, ...] = ()

    def cell(self, arm: str, section: str, band: str) -> Cell:
        for cell in self.cells:
            if (cell.arm, cell.section, cell.band) == (arm, section, band):
                return cell
        raise KeyError(f"no printed cell for {arm} / {section} / {band}")

    @property
    def numbers(self) -> int:
        """R5's arithmetic, as this render actually produced it."""
        return len(self.cells) * len(self.columns)


def build_table(
    frame: Frame,
    bar: prereg.Bar,
    *,
    intervals: Sequence[Interval] = (),
    size: str = HEADLINE_SIZE,
) -> Table:
    """The 66 cells, with R8's floor applied and R6's marks attached.

    Every cell value is the **mean over instances** of that instance's own metric. That is the
    unit p123 graded the metric set on ("the within-cell instance spread"), and it is what §5
    means when it defines mean per-instance |bias| as *averaged within an instance, then over
    instances* — so one aggregation rule serves all six columns rather than each column carrying
    its own.
    """
    floor = bar.cell_floor
    marks = _marks_by_cell(intervals, bar)
    cells = []
    for arm, section, band in printed_cells():
        rows = cell_rows(frame, arm, section, band, size=size)
        count, tiles = instances_and_tiles(frame, rows)
        if count == 0:
            values, note = None, "not scored on this box"
        elif count < floor:
            values, note = None, f"below R8's {floor}-instance floor"
        else:
            values = {
                name: float(np.mean(frame.columns[name][rows]))
                for name in bar.headline_metrics
            }
            note = ""
        cells.append(
            Cell(
                arm=arm,
                section=section,
                band=band,
                count=count,
                tiles=tiles,
                values=values,
                note=note,
                marks=marks.get((arm, section, band), ()),
            )
        )
    return Table(
        columns=tuple(bar.headline_metrics),
        cells=tuple(cells),
        footnotes=_real_family_footnotes(frame, size=size),
    )


#: §3.3's third published row. It carries a **count and no metric by construction** — it is not
#: one of R5's 66 cells and never was — so it is a footnote here rather than a row, which is the
#: only place a bare count belongs.
COUNT_ONLY_ROWS: Final[tuple[str, ...]] = tuple(
    row.name for row in voids.size_rows() if row.reporting == "count"
)


def _real_family_footnotes(frame: Frame, *, size: str) -> tuple[str, ...]:
    """What §3.3 publishes beside the two scored sub-rows: counts, and only counts.

    Two of them, and they are different things. The **other three bands** are R8's floor firing
    on a family that is 99.94% R4 (#135) — a scored row whose metric is withheld. The
    **128-160 px row** is §5's *"bare instance count for 128 to 160 px"*, which was never a
    scored row at all: it is above the sub-row boundaries that keep real-small size-comparable
    to synthetic small, so there is nothing for it to be compared against.
    """
    def instances(**equals) -> int:
        # Across every arm's rows, so it is deduplicated to the instances the *artifact* holds.
        # A raw row count here would multiply §3.3's histogram by however many arms have run,
        # which is a statement about this box rather than about the eval mask list.
        return instances_and_tiles(frame, np.flatnonzero(frame.select(**equals)))[0]

    out = []
    for sub in REAL_SUB_ROWS:
        counts = {
            band: instances(size=size, family=REAL_FAMILY, bucket=sub, band=band)
            for band in BANDS
            if band != REAL_BAND
        }
        if any(counts.values()):
            spelled = ", ".join(f"{band} {n:,}" for band, n in counts.items())
            out.append(
                f"{sub}: {spelled} instance(s) outside {REAL_BAND}, counted and not scored — "
                f"§3.3's histogram is 99.94% {REAL_BAND} (#135)"
            )
    for row in COUNT_ONLY_ROWS:
        total = instances(size=size, family=REAL_FAMILY, bucket=row)
        out.append(
            f"{row}: {total:,} instance(s), counted and never scored — §5 publishes this band "
            f"as a bare count because it sits above the sub-row boundaries that keep real-small "
            f"size-comparable to synthetic small"
        )
    return tuple(out)


def assert_r5_shape(table: Table, bar: prereg.Bar) -> None:
    """The render, checked against R5's own arithmetic before it reaches a page.

    Read off the bar rather than compared against 66 and 396 typed here: a second home for the
    printed shape is exactly what [#146](https://github.com/JStoschek/DEMAE/issues/146) cost, and
    the bar is the copy a run is answerable to.
    """
    cells, columns, numbers = bar.printed_shape
    if len(table.cells) != cells or len(table.columns) != columns:
        raise ValueError(
            f"this render is {len(table.cells)} cells x {len(table.columns)} columns and R5 "
            f"registers {cells} x {columns}. The printed table's shape was fixed before any "
            f"result existed, so a render that does not match it is the render that is wrong"
        )
    if table.numbers != numbers:
        raise ValueError(f"{table.numbers} printed numbers against R5's {numbers}")
    if bar.headline_family != BLOB_FAMILY:
        raise ValueError(
            f"R5 registers the {bar.headline_family!r} family as the headline and this module "
            f"prints {BLOB_FAMILY!r}. The family names come from `benchmark.config.Masks`, so "
            f"this is the register and the config having come apart — not a table to relabel"
        )
    appendix = set(bar.appendix_metrics) & set(table.columns)
    if appendix:
        raise ValueError(
            f"{sorted(appendix)} is an appendix metric and this render put it in a column. They "
            f"were graded into the appendix for being redundant or unable to separate the A1 "
            f"variants, so printing one re-imports the noise 24,000 instances were spent removing"
        )


def assert_rank_block_shape(block: RankBlock) -> None:
    """The block's own arithmetic — **not** R5's 66, which is a different 66.

    §5 is explicit that the two are unrelated: *"its 66 cells are the same 66 the C4 rank block
    has, and the two are unrelated — the coincidence is that both are 11 arms x 6, and nothing
    should be read across them."* Checking the block against R5's registered cell count would be
    exactly the reading across that sentence forbids, so the block is checked against what it is:
    three R4 instance sets, eleven arms, two metrics.
    """
    expected = len(block.sets) * len(arms.PRINTED_ROWS) * len(block.metrics)
    if len(block.cells) != expected:
        raise ValueError(
            f"the C4 block rendered {len(block.cells)} rank cells and {len(block.sets)} sets x "
            f"{len(arms.PRINTED_ROWS)} arms x {len(block.metrics)} metrics is {expected}"
        )


# --- C4's rank block ----------------------------------------------------------


@dataclass(frozen=True)
class RankCell:
    """One of C4's 66 rank cells. **No metre value**, by construction — see the module docstring."""

    arm: str
    set_name: str
    metric: str
    #: 1 is best. ``None`` when the arm has no scored instances in this set, or too few for R8.
    rank: int | None
    n: int


@dataclass(frozen=True)
class RankBlock:
    sets: tuple[str, ...]
    metrics: tuple[str, ...]
    cells: tuple[RankCell, ...]


def c4_sets(cfg: BenchmarkConfig) -> tuple[tuple[str, str, str], ...]:
    """C4's three R4 instance sets as ``(label, family, bucket)``.

    Synthetic blob **small** rather than the pooled blob row: §3.3 compares the real family
    against the synthetic small bucket *within band*, and pooling the buckets in the printed
    table is precisely what deleted that comparator from it. The block is what restores it.
    """
    small = cfg.masks.buckets[0][0]
    return (
        (f"{BLOB_FAMILY}/{small}", BLOB_FAMILY, small),
        *((sub, REAL_FAMILY, sub) for sub in REAL_SUB_ROWS),
    )


def rank_block(
    frame: Frame, bar: prereg.Bar, cfg: BenchmarkConfig, *, size: str = HEADLINE_SIZE
) -> RankBlock:
    """The 66-cell block: three R4 instance sets x 11 arms x RMSE and slope RMSE, as ranks."""
    sets = c4_sets(cfg)
    cells = []
    for label, family, bucket in sets:
        for metric in C4_METRICS:
            means: dict[str, float] = {}
            counts: dict[str, int] = {}
            for arm in arms.PRINTED_ROWS:
                keep = frame.select(
                    arm=arm, size=size, family=family, bucket=bucket, band=REAL_BAND
                )
                counts[arm] = int(keep.sum())
                if counts[arm] >= bar.cell_floor:
                    means[arm] = float(np.mean(frame.columns[metric][keep]))
            order = sorted(means, key=lambda arm: (means[arm], arm))
            rank_of = {arm: i + 1 for i, arm in enumerate(order)}
            for arm in arms.PRINTED_ROWS:
                cells.append(
                    RankCell(
                        arm=arm,
                        set_name=label,
                        metric=metric,
                        rank=rank_of.get(arm),
                        n=counts[arm],
                    )
                )
    return RankBlock(
        sets=tuple(label for label, _, _ in sets), metrics=C4_METRICS, cells=tuple(cells)
    )


# --- The bootstrap, over tiles ------------------------------------------------


@dataclass(frozen=True)
class Interval:
    """One pair's paired-bootstrap CI in one cell on one metric.

    ``lo``/``hi`` are ``first`` minus ``second`` in the metric's own units, and ``first`` is the
    arm the claim is about — R6 registers *A4-R5 vs A5*, and what C1 asserts is something about
    A4-R5.

    :attr:`clusters` is what the resample drew from, and it is a **tile** count. What it should
    be read against is the cell's instance count in the table above — on §3.3's real rows the two
    differ by ~8x, which is the entire reason the cluster is the tile, and the CI header prints
    that gap for the largest printed cell rather than repeating it on every line.
    """

    pair: tuple[str, str]
    section: str
    band: str
    metric: str
    lo: float
    hi: float
    #: Test tiles shared by both arms of the pair — the resampled population.
    clusters: int
    marked: bool

    @property
    def excludes_zero(self) -> bool:
        return self.lo > 0.0 or self.hi < 0.0


def _cell_stats(
    frame: Frame, arm: str, section: str, band: str, *, size: str
) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
    """Per-tile instance counts and per-tile metric sums for one arm in one cell.

    Sums and counts rather than means, because the resample re-weights *tiles*: a tile drawn
    twice contributes twice its sum and twice its count, and the cell mean is the ratio. Taking
    per-tile means first and averaging those would silently reweight a tile carrying 23 real
    voids to match one carrying a single synthetic void.
    """
    rows = cell_rows(frame, arm, section, band, size=size)
    if rows.size == 0:
        return None
    codes = frame.tile_code[rows]
    counts = np.bincount(codes, minlength=frame.n_tiles).astype(float)
    sums = {
        metric: np.bincount(
            codes, weights=frame.columns[metric][rows].astype(float), minlength=frame.n_tiles
        )
        for metric in CI_METRICS
    }
    return counts, sums


def _resampled_means(
    sums: np.ndarray, counts: np.ndarray, *, resamples: int, rng: np.random.Generator
) -> np.ndarray:
    """``(arms, resamples)`` cell means under one **tile** resample shared across every arm.

    One draw serves every arm on the tile set, which is what makes the comparison *paired*: two
    arms are differenced on the same resampled tiles, so the terrain that drives both cancels
    rather than entering as noise. It is also why the 55 pairs cost barely more than the four —
    the means are computed once per tile set and every pair is a subtraction.
    """
    n_tiles = sums.shape[1]
    probability = np.full(n_tiles, 1.0 / n_tiles)
    blocks = []
    drawn = 0
    while drawn < resamples:
        block = min(_RESAMPLE_CHUNK, resamples - drawn)
        weights = rng.multinomial(n_tiles, probability, size=block).astype(float)
        numerator = weights @ sums.T
        denominator = weights @ counts.T
        with np.errstate(invalid="ignore", divide="ignore"):
            blocks.append(np.where(denominator > 0, numerator / denominator, np.nan))
        drawn += block
    return np.concatenate(blocks, axis=0).T


def _draw_seed(section: str, band: str, metric: str, signature: bytes) -> int:
    """A deterministic seed for one cell, metric and tile set.

    Derived rather than threaded through a single generator, so a cell's interval does not
    depend on how many cells were computed before it. Two renders of one parquet agree to the
    digit, and so do a render of the whole table and a render that only reached this cell.
    """
    digest = hashlib.blake2b(
        f"{section}|{band}|{metric}".encode("utf-8") + signature,
        key=str(BOOTSTRAP_SEED).encode("utf-8"),
        digest_size=8,
    )
    return int.from_bytes(digest.digest(), "big")


def paired_intervals(
    frame: Frame,
    bar: prereg.Bar,
    *,
    size: str = HEADLINE_SIZE,
    resamples: int | None = None,
) -> tuple[Interval, ...]:
    """Every enumerable one of the 55 pairs, in every printed cell, on RMSE and slope RMSE.

    **A pair's interval is computed on that pair's own shared tiles.** That is the whole reason
    the tile set is a key below rather than one intersection taken over every arm on the box: an
    intersection over all of them would make A4-R5 vs A5 — a comparison registered before any
    number existed — move when A9 is run, which is a pre-registered CI depending on an arm that
    is not in it. Where every arm scores the same committed eval mask list, and they do, there is
    exactly one tile set and this costs nothing.

    A pair is skipped where either arm has no rows in the cell: a box mid-benchmark is the normal
    case, and an interval against an arm nobody has run is not a missing number but a comparison
    that does not exist yet.

    The four R6 marks ride on the intervals rather than being applied afterwards, so an unmarked
    pair cannot acquire a mark by clearing zero.
    """
    resamples = BOOTSTRAP_RESAMPLES if resamples is None else resamples
    # Keyed on the unordered pair, valued by R6's own orientation. `ALL_PAIRS` enumerates in
    # registry order and R6 does not — it registers *A3 vs A1*, claim-side first, and A1 is the
    # earlier row. Matching on the ordered tuple would silently leave C2's pair unmarked, so the
    # lookup is unordered and the interval is computed in the registered direction.
    registered = {frozenset(pair): pair for pair in marked_pairs(bar)}
    tail = (1.0 - CI_LEVEL) / 2.0
    out: list[Interval] = []
    for section, bands in SECTIONS:
        for band in bands:
            stats = {}
            for arm in arms.PRINTED_ROWS:
                cell = _cell_stats(frame, arm, section, band, size=size)
                if cell is not None:
                    stats[arm] = cell
            if len(stats) < 2:
                continue
            present = [arm for arm in arms.PRINTED_ROWS if arm in stats]
            for metric in CI_METRICS:
                drawn: dict[bytes, tuple[np.ndarray, np.ndarray] | None] = {}
                for enumerated in ALL_PAIRS:
                    key = frozenset(enumerated)
                    pair = registered.get(key, enumerated)
                    if pair[0] not in stats or pair[1] not in stats:
                        continue
                    shared = (stats[pair[0]][0] > 0) & (stats[pair[1]][0] > 0)
                    signature = shared.tobytes()
                    if signature not in drawn:
                        drawn[signature] = _draw_for(
                            stats,
                            present,
                            shared,
                            metric=metric,
                            resamples=resamples,
                            seed=_draw_seed(section, band, metric, signature),
                        )
                    computed = drawn[signature]
                    if computed is None:
                        continue
                    tiles, means = computed
                    difference = (
                        means[present.index(pair[0])] - means[present.index(pair[1])]
                    )
                    lo, hi = np.nanpercentile(difference, [100 * tail, 100 * (1.0 - tail)])
                    out.append(
                        Interval(
                            pair=pair,
                            section=section,
                            band=band,
                            metric=metric,
                            lo=float(lo),
                            hi=float(hi),
                            clusters=int(tiles.size),
                            marked=key in registered,
                        )
                    )
    return tuple(out)


def _draw_for(
    stats: Mapping[str, tuple[np.ndarray, dict[str, np.ndarray]]],
    present: Sequence[str],
    shared: np.ndarray,
    *,
    metric: str,
    resamples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """One tile set's resample: the tiles drawn from, and every present arm's means over them."""
    tiles = np.flatnonzero(shared)
    if tiles.size == 0:
        return None
    sums = np.array([stats[arm][1][metric][tiles] for arm in present], dtype=float)
    counts = np.array([stats[arm][0][tiles] for arm in present], dtype=float)
    means = _resampled_means(
        sums, counts, resamples=resamples, rng=np.random.default_rng(seed)
    )
    return tiles, means


#: How a marked metric is spelled inside a mark token — one letter, so a cell carrying six of
#: them stays narrower than the six numbers beside it.
_METRIC_LETTER: Final = {"rmse": "r", "slope_rmse": "s"}


def mark_token(interval: Interval, order: Mapping[tuple[str, str], int]) -> str:
    """``1r`` — R6's registration index for the pair, then the metric's letter.

    The index rather than the pair's names, because a cell can carry three marks at once and
    three spelled-out pairs would be wider than the six numbers they annotate. What the index
    means is printed once, under the table, in R6's own order.
    """
    return f"{order[interval.pair]}{_METRIC_LETTER[interval.metric]}"


def _marks_by_cell(
    intervals: Sequence[Interval], bar: prereg.Bar
) -> dict[tuple[str, str, str], tuple[str, ...]]:
    """Which cells carry a mark: the claim-side arm of a marked pair whose CI excludes zero.

    The mark rides on the pair's **first** name because that is the arm the claim is about —
    R6 registers *A4-R5 vs A5*, and what C1 asserts is something about A4-R5.
    """
    order = mark_order(bar)
    out: dict[tuple[str, str, str], list[str]] = {}
    for interval in intervals:
        if not (interval.marked and interval.excludes_zero):
            continue
        key = (interval.pair[0], interval.section, interval.band)
        out.setdefault(key, []).append(mark_token(interval, order))
    return {key: tuple(sorted(value)) for key, value in out.items()}


# --- Seed spread --------------------------------------------------------------


@dataclass(frozen=True)
class Spread:
    """§5's one printed line: the largest seed-to-seed spread anywhere in the headline table.

    Deliberately **not** a per-cell ±. Only 5 of the 11 arm rows carry three seeds at all, so a
    per-cell ± would imply a seed structure most rows do not have — and it reads as a CI besides.
    A spread large enough to threaten a claim earns a paragraph, not a decoration in 396 cells.
    """

    value: float | None
    arm: str = ""
    section: str = ""
    band: str = ""
    metric: str = ""
    seeds: int = 0


def seed_spread(
    frame: Frame, bar: prereg.Bar, *, size: str = HEADLINE_SIZE
) -> Spread:
    """The maximum over every (cell, headline metric) of ``max - min`` across that cell's seeds.

    Over the cells the table actually **printed a metric for**, and R8's floor is why. A cell
    below the floor has no printed number, so a spread quoted off it would be a statement about
    a quantity the table withheld — and the printed line says "across the headline table".
    """
    best = Spread(value=None)
    for arm in arms.PRINTED_ROWS:
        for section, bands in SECTIONS:
            for band in bands:
                rows = cell_rows(frame, arm, section, band, size=size)
                if instances_and_tiles(frame, rows)[0] < bar.cell_floor:
                    continue
                seeds, within = np.unique(
                    frame.columns["seed"][rows], return_inverse=True
                )
                if seeds.size < 2:
                    continue
                counts = np.bincount(within)
                for name in bar.headline_metrics:
                    per_seed = (
                        np.bincount(
                            within, weights=frame.columns[name][rows].astype(float)
                        )
                        / counts
                    )
                    spread = float(per_seed.max() - per_seed.min())
                    if best.value is None or spread > best.value:
                        best = Spread(
                            value=spread,
                            arm=arm,
                            section=section,
                            band=band,
                            metric=name,
                            seeds=int(seeds.size),
                        )
    return best


# --- §6 item 4's parameter counts ---------------------------------------------


def parameter_counts(frame: Frame) -> dict[str, dict[str, int | None]]:
    """``arm -> {"total", "trainable"}``, from the run records beside the parquet.

    The records rather than :func:`benchmark.arms.parameter_counts`, because what §6 item 4 asks
    for is what the run *counted* — a registry figure is what a table would print for an arm
    nobody has built. The registry answers for arms with no record, which is how A1's true zeros
    reach the page before A1 has been run on this box.
    """
    out = {
        arm: dict(arms.parameter_counts(arms.get(arm))) for arm in arms.PRINTED_ROWS
    }
    for record in results.read_run_records(results.run_record_path(frame.path)):
        arm = record.get("cell", {}).get("arm")
        if arm in out and record.get("parameters"):
            out[arm] = dict(record["parameters"])
    return out


# --- The three figures --------------------------------------------------------


@dataclass(frozen=True)
class Series:
    label: str
    x: tuple[float, ...]
    y: tuple[float, ...]


@dataclass(frozen=True)
class PanelCell:
    tile_id: str
    column: str
    #: The rendered hillshade, or ``None`` with :attr:`absent` saying what this box does not hold.
    image: np.ndarray | None = None
    absent: str = ""


@dataclass(frozen=True)
class Panel:
    columns: tuple[str, ...]
    tiles: tuple[str, ...]
    cells: tuple[PanelCell, ...]


@dataclass(frozen=True)
class Figure:
    """One of §9's three. All three are marked load-bearing and none may be cut for space."""

    key: str
    title: str
    #: Why it is load-bearing. Carried as a field so the marking is a fact about the figure
    #: rather than a sentence in a memo that can be edited away from it.
    because: str
    load_bearing: bool = True
    x_label: str = ""
    y_label: str = ""
    series: tuple[Series, ...] = ()
    panel: Panel | None = None


#: §9's three, declared before any data is attached — so "all three are load-bearing" is a
#: property of the module rather than of a particular render.
FIGURES: Final[tuple[Figure, ...]] = (
    Figure(
        key="rmse-vs-void-size",
        title="RMSE versus void size, per arm",
        because=(
            "#136 pooled the size buckets out of the printed table, so this is the only place "
            "void size appears and therefore the whole of C2's 'at every void size'"
        ),
        x_label="void equivalent diameter (px)",
        y_label="RMSE (m)",
    ),
    Figure(
        key="label-efficiency",
        title="The label-efficiency curve (C3)",
        because="the whole of C3 — the sweep's scored passes print here and nowhere else",
        x_label="head-train tiles",
        y_label="RMSE (m)",
    ),
    Figure(
        key="hillshade-panel",
        title="Hillshade panel — truth, voided, and four arms",
        because=(
            "§5 scores no perceptual metric, so this stands in for the structural metrics that "
            "were cut and is the whole of the visual-quality evidence"
        ),
    ),
)


def declared(key: str) -> Figure:
    """One of §9's three by name. Positional indexing into :data:`FIGURES` is one reordering
    away from a figure rendered under another's caption."""
    for figure in FIGURES:
        if figure.key == key:
            return figure
    raise KeyError(f"no declared figure {key!r}; §9's three are {[f.key for f in FIGURES]}")


def assert_load_bearing(figs: Sequence[Figure]) -> None:
    """§9 marks all three load-bearing and none may be cut for space, so an unmarked one is a
    defect rather than a category. Refused here rather than printed as "decorative"."""
    unmarked = [figure.key for figure in figs if not figure.load_bearing]
    if unmarked:
        raise ValueError(
            f"{unmarked} rendered without §9's load-bearing marking. All three carry it — the "
            f"size figure is the whole of C2's 'at every void size', the curve is the whole of "
            f"C3, and the panel is the whole of the visual-quality evidence"
        )


def _mean_by_bin(
    frame: Frame, arm: str, edges: np.ndarray, *, size: str, floor: int
) -> Series | None:
    """One arm's RMSE against void size over the blob family, pooled across bands."""
    keep = frame.select(arm=arm, size=size, family=BLOB_FAMILY)
    if not keep.any():
        return None
    diameter = frame.columns["void_eqdiam_px"][keep].astype(float)
    rmse = frame.columns["rmse"][keep].astype(float)
    xs, ys = [], []
    for lo, hi in zip(edges, edges[1:]):
        inside = (diameter >= lo) & (diameter < hi if hi != edges[-1] else diameter <= hi)
        if int(inside.sum()) < floor:
            continue
        xs.append(float(np.mean(diameter[inside])))
        ys.append(float(np.mean(rmse[inside])))
    return Series(label=arm, x=tuple(xs), y=tuple(ys)) if xs else None


def rmse_vs_void_size(frame: Frame, bar: prereg.Bar, *, size: str = HEADLINE_SIZE) -> Figure:
    """C2's evidence, at the grain the table gave up when the buckets pooled."""
    blob = frame.select(size=size, family=BLOB_FAMILY)
    if not blob.any():
        return replace(declared("rmse-vs-void-size"), series=())
    diameter = frame.columns["void_eqdiam_px"][blob].astype(float)
    edges = np.linspace(diameter.min(), diameter.max(), SIZE_BINS + 1)
    series = [
        s
        for s in (
            _mean_by_bin(frame, arm, edges, size=size, floor=bar.cell_floor)
            for arm in arms.PRINTED_ROWS
        )
        if s is not None
    ]
    return replace(declared("rmse-vs-void-size"), series=tuple(series))


def label_efficiency(frame: Frame) -> Figure:
    """C3's whole sweep. The x axis is tiles rather than the size's name, so the curve is one."""
    series = []
    for arm in arms.PRINTED_ROWS:
        xs, ys = [], []
        for size in LABEL_EFFICIENCY_SIZES:
            keep = frame.select(arm=arm, size=size, family=BLOB_FAMILY)
            if keep.any():
                xs.append(float(SIZE_TILES[size]))
                ys.append(float(np.mean(frame.columns["rmse"][keep])))
        if len(xs) > 1:
            series.append(Series(label=arm, x=tuple(xs), y=tuple(ys)))
    return replace(declared("label-efficiency"), series=tuple(series))


def panel_exhibit_dir(results_path: Path | str) -> Path:
    """Where an arm's saved fills live, if this box holds any — ``<results>/panel/``.

    Derived from the results file rather than configured, on :func:`benchmark.results.
    run_record_path`'s rule: the exhibits are part of the same artifact, and a box that could
    point them somewhere else is a box that can draw a panel from another run's pixels.

    **Nothing writes this directory yet, and that is stated rather than left to be discovered.**
    A fill is a 224x224 raster and the results parquet holds metrics, so the panel's four arm
    columns cannot come from the file `report` is a pure function of. The two honest options were
    a `run` that saves three tiles' fills per arm — which is the arm tickets' work, not this
    one's — or a figure that renders the absence and names it. This is the second: `truth` and
    `voided` are drawn from the corpus and the committed eval mask list, which `report` may read
    because reading them re-runs nothing, and each arm column says what the box does not hold.
    What `report` may never do is compute the fill itself.
    """
    return Path(results_path).parent / "panel"


def panel_tiles(frame: Frame, *, size: str = HEADLINE_SIZE) -> tuple[str, ...]:
    """§9's three host tiles — two rugged, one flat — chosen from the parquet, deterministically.

    Sorted and taken from the front rather than sampled: the panel is quoted in a memo, so which
    three tiles it draws must not move when the file is re-read.
    """
    chosen: list[str] = []
    for band in PANEL_BANDS:
        keep = frame.select(size=size, family=BLOB_FAMILY, band=band)
        available = sorted(set(frame.columns["tile_id"][keep].tolist()) - set(chosen))
        if available:
            chosen.append(available[0])
    return tuple(chosen)


def _shade(raw: np.ndarray, lat: float) -> np.ndarray:
    from dataset.gallery import shade_tile

    return shade_tile(np.asarray(raw), lat=lat)[0]


def hillshade_panel(frame: Frame, cfg: BenchmarkConfig) -> Figure:
    """The panel, composed here and drawn from whatever pixels this box actually holds.

    Never by computing a fill: an arm column this box has no exhibit for renders as a labelled
    absence, which is a true statement about the box. The alternative — re-running the arm to
    fill the cell in — is the one thing `report` may not do.
    """
    tiles = panel_tiles(frame)
    columns = tuple(
        column if column in ("truth", "voided") else resolve_arm(column)
        for column in PANEL_COLUMNS
    )
    store, store_absent = _tile_store(cfg)
    voids_by_tile = _panel_voids(cfg, tiles)
    exhibits = panel_exhibit_dir(cfg.paths.results_path)
    cells = []
    for tile_id in tiles:
        tile, missing = None, store_absent
        if store is not None:
            try:
                tile = store.read(tile_id)
            except KeyError:  # pragma: no cover - a corpus that lost a committed tile
                # Per tile, not carried forward: one tile the corpus has lost says nothing about
                # the next one, and a message that leaked across rows would misreport the box.
                missing = f"{tile_id} is not in this corpus"
        for column in columns:
            image, absent = None, ""
            if tile is None:
                absent = missing
            elif column == "truth":
                image = _shade(tile.raw_i16, tile.lat)
            elif column == "voided":
                void = voids_by_tile.get(tile_id)
                if void is None:
                    absent = "no committed eval mask for this tile on this box"
                else:
                    voided = np.asarray(tile.raw_i16, dtype=np.float64).copy()
                    voided[void] = float(np.median(voided[~void])) if (~void).any() else 0.0
                    image = _shade(voided, tile.lat)
            else:
                fill = exhibits / f"{column}--{tile_id}.npy"
                if fill.is_file():
                    image = _shade(np.load(fill), tile.lat)
                else:
                    # Accurate about *which* thing is missing. An arm can have been run — its
                    # rows can be in the parquet, in the table above — and still have no fill
                    # here, because a fill is a raster and no arm writes one yet.
                    absent = f"no fill exhibit for {column} in {exhibits.name}/"
            cells.append(
                PanelCell(tile_id=tile_id, column=column, image=image, absent=absent)
            )
    return replace(
        declared("hillshade-panel"),
        panel=Panel(columns=columns, tiles=tiles, cells=tuple(cells)),
    )


def _tile_store(cfg: BenchmarkConfig) -> tuple[Any | None, str]:
    """The corpus, or the reason this box has none. The panel is the one figure that needs it."""
    from .evaluate import TileStore

    try:
        return TileStore(cfg.paths.corpus_dir), ""
    except FileNotFoundError:
        return None, f"no corpus at {cfg.paths.corpus_dir}"


def _panel_voids(cfg: BenchmarkConfig, tiles: Sequence[str]) -> dict[str, np.ndarray]:
    """The committed blob mask for each panel tile, or nothing if the list is not on this box."""
    from .evaluate import read_instances

    try:
        instances = read_instances(cfg)
    except Exception:
        # Deliberately broad. Every way this box can lack a usable eval mask list — the file
        # absent, a test list it disagrees with, a parquet that will not open — has the same
        # consequence here: the `voided` column is a labelled absence rather than a picture.
        # A figure is not the place to re-raise a `freeze` problem `run` already refuses on.
        return {}
    wanted = set(tiles)
    out: dict[str, np.ndarray] = {}
    for instance in instances:
        if instance.tile_id in wanted and instance.family == BLOB_FAMILY:
            out.setdefault(instance.tile_id, masks.decode_rle(instance.mask_rle))
    return out


def figures(frame: Frame, bar: prereg.Bar, cfg: BenchmarkConfig) -> tuple[Figure, ...]:
    """§9's three, with their data attached. All three, always — none is cut for space."""
    out = (
        rmse_vs_void_size(frame, bar),
        label_efficiency(frame),
        hillshade_panel(frame, cfg),
    )
    assert_load_bearing(out)
    return out


def render_figures(figs: Sequence[Figure], directory: Path | str) -> tuple[Path, ...]:
    """Write one PNG per figure. matplotlib is imported here and nowhere at module scope.

    `src/dataset` keeps the same rule (`dataset.gallery`: *"matplotlib is a dev-only dependency
    and no module imports it"*), and the reason carries: the freeze and scoring paths run on a
    box that installs neither the plotting stack nor torch.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for figure in figs:
        path = directory / f"{figure.key}.png"
        if figure.panel is not None:
            _render_panel(plt, figure, path)
        else:
            _render_curve(plt, figure, path)
        written.append(path)
    return tuple(written)


def _caption(figure: Figure) -> str:
    return f"LOAD-BEARING — {figure.because}"


def _render_curve(plt: Any, figure: Figure, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for series in figure.series:
        ax.plot(series.x, series.y, marker="o", label=series.label)
    if not figure.series:
        ax.text(
            0.5, 0.5, "no scored arm on this box", ha="center", va="center",
            transform=ax.transAxes,
        )
    ax.set_title(figure.title)
    ax.set_xlabel(figure.x_label)
    ax.set_ylabel(figure.y_label)
    if figure.series:
        ax.legend(fontsize="small")
    fig.text(0.01, 0.01, _caption(figure), fontsize=7, wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _render_panel(plt: Any, figure: Figure, path: Path) -> None:
    panel = figure.panel
    rows = max(len(panel.tiles), 1)
    columns = max(len(panel.columns), 1)
    fig, axes = plt.subplots(
        rows, columns, figsize=(2.0 * columns, 2.2 * rows), squeeze=False
    )
    by_key = {(cell.tile_id, cell.column): cell for cell in panel.cells}
    for r, tile_id in enumerate(panel.tiles):
        for c, column in enumerate(panel.columns):
            ax = axes[r][c]
            ax.set_xticks([])
            ax.set_yticks([])
            cell = by_key.get((tile_id, column))
            if cell is not None and cell.image is not None:
                ax.imshow(cell.image)
            else:
                ax.text(
                    0.5, 0.5, (cell.absent if cell else "—"),
                    ha="center", va="center", fontsize=6, wrap=True,
                    transform=ax.transAxes,
                )
            if r == 0:
                ax.set_title(column, fontsize=8)
        axes[r][0].set_ylabel(tile_id, fontsize=6)
    fig.suptitle(figure.title, fontsize=10)
    fig.text(0.01, 0.01, _caption(figure), fontsize=7, wrap=True)
    # `subplots_adjust` rather than `tight_layout`: the grid is six narrow square panels and
    # tight_layout cannot fit their decorations, which it says in a warning on every render.
    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.08, wspace=0.08, hspace=0.16)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- The render ---------------------------------------------------------------


@dataclass(frozen=True)
class Report:
    """Everything `report` prints, computed. The lines are :meth:`lines`' business."""

    table: Table
    ranks: RankBlock
    intervals: tuple[Interval, ...]
    spread: Spread
    parameters: dict[str, dict[str, int | None]]
    figures: tuple[Figure, ...]
    bar: prereg.Bar
    #: What the bootstrap actually drew, so the printed line cannot quote a count it did not use.
    resamples: int = 0

    @property
    def marked(self) -> tuple[Interval, ...]:
        return tuple(i for i in self.intervals if i.marked)

    def lines(self) -> list[str]:
        return [
            *_table_lines(self.table, self.bar),
            "",
            *_rank_lines(self.ranks),
            "",
            *_ci_lines(self.intervals, self.table, self.bar, resamples=self.resamples),
            "",
            *_spread_lines(self.spread),
            "",
            *_parameter_lines(self.parameters),
            "",
            *_figure_lines(self.figures),
        ]


def build(cfg: BenchmarkConfig, *, resamples: int | None = None) -> Report:
    """Read the parquet and compute everything that prints. Opens no encoder and runs no arm.

    ``resamples`` defaults to :data:`BOOTSTRAP_RESAMPLES` and resolves it **at call time** rather
    than as a default argument, so the constant is the single place §5's registered count lives —
    a default bound at import would make the module attribute a decoration nothing reads.
    """
    resamples = BOOTSTRAP_RESAMPLES if resamples is None else resamples
    frame = read(cfg.paths.results_path)
    bar = metrics.bar_for(cfg)
    intervals = paired_intervals(frame, bar, resamples=resamples)
    table = build_table(frame, bar, intervals=intervals)
    assert_r5_shape(table, bar)
    ranks = rank_block(frame, bar, cfg)
    assert_rank_block_shape(ranks)
    return Report(
        table=table,
        ranks=ranks,
        intervals=intervals,
        spread=seed_spread(frame, bar),
        parameters=parameter_counts(frame),
        figures=figures(frame, bar, cfg),
        bar=bar,
        resamples=resamples,
    )


def figures_dir(results_path: Path | str) -> Path:
    """Where the three figures land — derived from the results file, like the run record."""
    return Path(results_path).parent / "figures"


def render(cfg: BenchmarkConfig, *, resamples: int | None = None) -> Report:
    """Build the report, write the figures, and print every line. What `launch report` is."""
    report = build(cfg, resamples=resamples)
    written = render_figures(report.figures, figures_dir(cfg.paths.results_path))
    for line in report.lines():
        print(line)
    print("")
    for path in written:
        print(f"{'wrote':<16} {path}")
    return report


# --- Formatting ---------------------------------------------------------------

_VALUE_WIDTH: Final = 14
_ROW_WIDTH: Final = 34


def _format_value(name: str, value: float, bar: prereg.Bar) -> str:
    """One printed number. The seam gap is signed to 3 dp with the sign as a visual channel.

    §5 asks for the sign to be *carried* rather than left to be read off the digits, because the
    scale is wildly asymmetric — the break side sits an order of magnitude above the intact side,
    and the intact side is a cluster straddling zero, so a raw signed column reads as noise. The
    ``!`` is R7's +0.15° bar, and it is the bar's number rather than one typed here.
    """
    if name == "seam_gap":
        flag = "!" if metrics.is_seam_break(value, bar=bar) else ""
        return f"{value:+.3f}{flag}"
    return f"{value:.3f}"


def _table_lines(table: Table, bar: prereg.Bar) -> list[str]:
    cells, columns, numbers = bar.printed_shape
    out = [
        "== The printed table ==",
        f"  R5: {cells} (row, band) cells x {columns} columns = {numbers} numbers; size buckets "
        f"pooled, {bar.headline_family} family headline, all four bands whole",
        "",
        f"{'arm':<8}{'section':<16}{'band':<10}"
        + "".join(f"{name:>{_VALUE_WIDTH}}" for name in table.columns),
    ]
    for cell in table.cells:
        label = f"{cell.arm:<8}{cell.section:<16}{cell.band:<4}"[:_ROW_WIDTH]
        if cell.values is None:
            body = f"n={cell.count:,} — {cell.note}"
        else:
            body = "".join(
                f"{_format_value(name, cell.values[name], bar):>{_VALUE_WIDTH}}"
                for name in table.columns
            )
            if cell.marks:
                body += "   " + " ".join(f"*{token}" for token in cell.marks)
        out.append(f"{label:<{_ROW_WIDTH}}{body}")
    out.append("")
    out.append(f"  `!` marks R7's seam break: a gap above +{bar.seam_break_deg:g}°")
    for note in table.footnotes:
        out.append(f"  footnote: {note}")
    return out


def _rank_lines(block: RankBlock) -> list[str]:
    """Six lines of eleven ranks. Transposed against the table: the *set* is what varies here."""
    out = [
        "== C4 rank block ==",
        f"  {len(block.cells)} rank cells — {len(block.sets)} R4 instance sets x "
        f"{len(arms.PRINTED_ROWS)} arms x {len(block.metrics)} metrics. Ranks, never metres: §0 "
        f"makes C4 an ordering claim precisely so that a natural-distribution family is never "
        f"differenced against a bucket-balanced one, and a caption cannot stop a subtraction the "
        f"columns invite",
        "",
        f"{'set / metric':<30}" + "".join(f"{arm:>8}" for arm in arms.PRINTED_ROWS),
    ]
    by_row: dict[tuple[str, str], dict[str, RankCell]] = {}
    for cell in block.cells:
        by_row.setdefault((cell.set_name, cell.metric), {})[cell.arm] = cell
    for set_name in block.sets:
        for metric in block.metrics:
            cells = by_row.get((set_name, metric), {})
            body = "".join(
                f"{_rank_of(cells.get(arm)):>8}" for arm in arms.PRINTED_ROWS
            )
            out.append(f"{f'{set_name} / {metric}':<30}{body}")
    out.append("")
    out.append(
        "  `—` is an arm with no scored instances in that set, or too few for R8's floor"
    )
    return out


def _rank_of(cell: RankCell | None) -> str:
    return "—" if cell is None or cell.rank is None else f"#{cell.rank}"


def _ci_lines(
    intervals: Sequence[Interval], table: Table, bar: prereg.Bar, *, resamples: int
) -> list[str]:
    registered = marked_pairs(bar)
    order = mark_order(bar)
    computed = len({i.pair for i in intervals})
    out = [
        "== Bootstrap CIs ==",
        f"  paired bootstrap over **test tiles**, {resamples:,} resamples, {CI_LEVEL:.0%} CI. "
        f"The cluster is the tile: ~8 real voids share one host tile's terrain, so a "
        f"per-instance resample would overstate precision by up to sqrt(8)",
        f"  {_cluster_gap(table)}",
        f"  {len(registered)} pairs carry a mark, named in advance in R6, of "
        f"{bar.computed_pairs} enumerated ({computed} computable on this box). Every other pair "
        f"is computed and left unmarked — which comparisons matter is fixed by the claims, not "
        f"by which ones clear zero",
    ]
    for note in resolutions(bar):
        out.append(f"  resolved: {note}")
    for pair, claim in zip(registered, bar.bootstrap_claims):
        index = order[pair]
        letters = ", ".join(
            f"{index}{letter} = {metric}" for metric, letter in _METRIC_LETTER.items()
        )
        out.append(f"  marked {index}: {pair[0]} vs {pair[1]} ({claim}) — {letters}")
        firing = [i for i in intervals if i.pair == pair and i.excludes_zero]
        if not firing:
            out.append("      no cell's CI excludes zero on this box")
        for interval in firing:
            out.append(
                f"      *{mark_token(interval, order):<3} {interval.section} "
                f"{interval.band}: [{interval.lo:+.3f}, {interval.hi:+.3f}] "
                f"({interval.pair[0]} minus {interval.pair[1]}, "
                f"over {interval.clusters:,} tiles)"
            )
    return out


def _cluster_gap(table: Table) -> str:
    """The tile-versus-instance gap, in the printed table's own numbers.

    Off the widest cell that actually printed, so it is a fact about this table rather than a
    figure quoted from §3.3. Where the two are equal the bootstrap is a plain instance
    bootstrap and saying so is the honest reading.
    """
    printed = [cell for cell in table.cells if cell.values is not None and cell.tiles]
    if not printed:
        return "no cell printed a metric, so there is no cluster population to describe"
    widest = max(printed, key=lambda cell: cell.count)
    ratio = widest.count / widest.tiles
    return (
        f"widest printed cell ({widest.arm} {widest.section} {widest.band}): "
        f"{widest.count:,} instances on {widest.tiles:,} tiles — {ratio:.1f} per tile"
    )


def _spread_lines(spread: Spread) -> list[str]:
    out = ["== Seed spread =="]
    if spread.value is None:
        out.append(
            "  no cell on this box carries more than one seed, so there is no spread to report"
        )
        return out
    out.append(
        f"  maximum seed spread across the headline table: {spread.value:.3f} "
        f"({spread.metric} at {spread.arm} {spread.section} {spread.band}, "
        f"{spread.seeds} seeds)"
    )
    out.append(
        "  reported once rather than per cell: only 5 of the 11 arm rows carry three seeds, so "
        "a per-cell spread would imply a seed structure most rows do not have"
    )
    return out


def _parameter_lines(counts: Mapping[str, Mapping[str, int | None]]) -> list[str]:
    def spell(value: int | None) -> str:
        return "not built yet" if value is None else f"{value:,}"

    out = [
        "== Parameters (§6 item 4) ==",
        "  both counts, always: A7's total reads as handicapped beside A4's until you know how "
        "much more of it trains",
        f"  {'arm':<10}{'total':>18}{'trainable':>18}",
    ]
    for arm in arms.PRINTED_ROWS:
        entry = counts.get(arm, {})
        out.append(
            f"  {arm:<10}{spell(entry.get('total')):>18}{spell(entry.get('trainable')):>18}"
        )
    # Named rather than absent, and named rather than approximated. §5's efficiency footnote asks
    # for a third quantity beside these two, and the run record holds nothing that is it: its
    # `seconds` is a whole pass — fill, score and all ten metrics — over ~40,000 instances, and
    # dividing it would print a scoring cost under an inference label. A9's diffusion decode
    # against A3's single forward pass is the contrast it exists to draw, and #163 is where that
    # is measured.
    out.append(
        "  §5's third efficiency number — single-tile inference time per arm — is not measured "
        "by any run on this box and is deliberately not derived from the pass timing, which is "
        "scoring as much as inference"
    )
    return out


def _figure_lines(figs: Sequence[Figure]) -> list[str]:
    out = ["== Figures =="]
    for figure in figs:
        out.append(f"  {figure.key:<20} load-bearing — {figure.because}")
        if figure.panel is not None:
            absent = sorted({cell.absent for cell in figure.panel.cells if cell.absent})
            for note in absent:
                out.append(f"{'':<22} · {note}")
    return out
