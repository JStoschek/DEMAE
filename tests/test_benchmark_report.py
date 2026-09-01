"""`report` — the printed table, the C4 rank block and the three figures (#169).

**Seam 1**, as the build spec asks: every acceptance assertion here drives
`launch.main(["report", ...])` and reads what it printed, because *the composition* is this
repo's named failure and a module that renders correctly inside a command that never calls it is
the shape ADR-0009 exists to close. The computed structures are asserted on
:class:`benchmark.report.Report`, which is what that command builds — one object, printed by one
formatter, rather than a second render only the tests can see.

The fixture is a **hand-written results parquet and no corpus at all**. That is not a shortcut:
`report` is a pure function of that file (#136's "computed and printed are different questions"),
so a fixture that ran an arm to produce it would be testing the arm. It is written through
:func:`benchmark.results.append_rows` — the writer `run` itself uses — so the columns are the
committed ones rather than a stand-in free to drift from them.

Nothing here asserts the magnitude of any number. The values are fabricated; what is under test
is the shape R5 fixed before any real one existed.
"""

from __future__ import annotations

import io
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
from pathlib import Path

import pytest

from benchmark import arms, launch, prereg, report, results
from benchmark.prereg import BAR_NAME
from benchmark.config import BenchmarkConfig

REPO = Path(__file__).resolve().parents[1]
PLAN = REPO / prereg.PLAN_PATH

#: Enough instances per (row, band) cell to clear R8's registered 100 floor, and few enough to
#: write quickly. A fixture under the floor would be testing R8 in every cell at once.
PER_CELL = 120

#: Four instances per tile, so the bootstrap's cluster population is four times smaller than its
#: instance count — which is what makes "it resamples tiles" an assertion rather than a claim.
PER_TILE = 4

#: Six arms scored and five not. A box mid-benchmark is the normal case, and R5's shape must not
#: depend on how much of the table has been run.
SCORED_ARMS = ("A1", "A3-R5", "A4-R5", "A5", "A6", "A7")

#: §5's five three-seed rows, intersected with what this fixture scores.
THREE_SEED_ARMS = ("A4-R5", "A5", "A6", "A7")


def _metrics(bar: prereg.Bar, arm: str, band: str, i: int) -> dict[str, float]:
    """A plausible metric row, separated by arm and by band so no rank block comes out all ties."""
    base = 1.0 + arms.PRINTED_ROWS.index(arm) + 3.0 * int(band[1])
    return {
        name: base + 0.01 * i + 0.5 * j - (1.0 if name == "seam_gap" else 0.0)
        for j, name in enumerate(bar.registered_metrics)
    }


#: How many instances §3.3's bare-count row gets in the fixture. Well under R8's floor, because
#: that row carries a count and no metric *by construction* rather than by failing a threshold.
COUNT_ONLY = 7


def _rows(bar, cfg, *, arm, seed, size, per_cell=PER_CELL):
    """One cell's rows: both synthetic families over four bands, plus §3.3's real sub-rows in R4."""
    buckets = [name for name, _ in cfg.masks.buckets]
    out = []
    for family, row_buckets, bands in (
        ("blob", buckets, report.BANDS),
        ("rectangle", buckets, report.BANDS),
        ("real", list(report.REAL_SUB_ROWS), (report.REAL_BAND,)),
        ("real", list(report.COUNT_ONLY_ROWS), (report.REAL_BAND,)),
    ):
        if family == "real" and row_buckets == list(report.COUNT_ONLY_ROWS):
            per_cell = COUNT_ONLY
        for bucket in row_buckets:
            for band in bands:
                for i in range(per_cell):
                    out.append(
                        results.ResultRow(
                            arm=arm,
                            init=arms.get(arm).encoder,
                            size=size,
                            seed=seed,
                            family=family,
                            bucket=bucket,
                            band=band,
                            tile_id=f"tile-{i // PER_TILE:04d}",
                            instance=i % PER_TILE,
                            void_area_px=200 + i,
                            void_eqdiam_px=16.0 + (i % 60) * 2.0,
                            mask_rle_md5="0" * 32,
                            run_id="fixture",
                            config_hash=cfg.config_hash,
                            metrics=_metrics(bar, arm, band, i),
                        )
                    )
    return out


@pytest.fixture(scope="module")
def bar(tmp_path_factory) -> prereg.Bar:
    """The registered bar, written by its own writer and checked against the plan's §6."""
    path = prereg.write_bar(tmp_path_factory.mktemp("bar") / BAR_NAME, plan=PLAN)
    return prereg.read_bar(path, plan=PLAN)


#: How many instances a deliberately thin cell keeps — under R8's registered floor of 100.
THIN = 20


def _write_results(box, bar, *, thin=(), half_tile_arms=(), extra_arms=()):
    """The parquet.

    ``thin`` cuts any ``(arm, family, band)`` to :data:`THIN` instances, which is how R8's floor
    is made to fire. ``half_tile_arms`` scores an arm on only the lower half of the tiles, which
    is how a box where two arms saw different ground is built — the state a pre-registered pair's
    interval must not notice.
    """
    cfg = BenchmarkConfig()
    thinned = set(thin)
    halved = set(half_tile_arms)

    def keep(row) -> bool:
        # `_rows` writes `void_area_px = 200 + i`, so the row carries the index it was built at.
        if (row.arm, row.family, row.band) in thinned and row.void_area_px - 200 >= THIN:
            return False
        if row.arm in halved and int(row.tile_id.rsplit("-", 1)[1]) >= PER_CELL // PER_TILE // 2:
            return False
        return True

    rows = []
    for arm in (*SCORED_ARMS, *extra_arms):
        seeds = (0, 1, 2) if arm in THREE_SEED_ARMS else (0,)
        for seed in seeds:
            rows.extend(_rows(bar, cfg, arm=arm, seed=seed, size="20k"))
        # C3's sweep: the two smaller subsets print in a figure and nowhere in the table.
        for size in ("1k", "5k"):
            rows.extend(_rows(bar, cfg, arm=arm, seed=0, size=size))
    results.append_rows(
        [row for row in rows if keep(row)],
        box / "results.parquet",
        metrics=bar.registered_metrics,
    )
    return box / "results.parquet"


def _config(flags) -> BenchmarkConfig:
    """The config the command would resolve — parsed through the command's own parser."""
    from benchmark.config import config_from_args

    return config_from_args(launch.build_parser().parse_args(["report", *flags]))


def _flags(box) -> list[str]:
    return [
        "--set", f'paths.artifacts_dir="{box / "artifacts"}"',
        "--set", f'paths.results_path="{box / "results.parquet"}"',
        "--set", f'paths.corpus_dir="{box / "corpus"}"',
        "--set", f'paths.plan_path="{PLAN}"',
    ]


def _make_box(root, bar, *, thin=(), half_tile_arms=(), extra_arms=()):
    """A box carrying the registered bar and a results parquet with six of eleven arms scored."""
    directory = root / "artifacts"
    directory.mkdir(exist_ok=True)
    prereg.write_bar(directory / BAR_NAME, plan=PLAN)
    _write_results(
        root, bar, thin=thin, half_tile_arms=half_tile_arms, extra_arms=extra_arms
    )
    return root


@contextmanager
def cheap_bootstrap():
    """200 resamples rather than §5's registered 10,000.

    Patched rather than configured, and that is as much the design under test as a convenience:
    §5 registers the resample count, so a box able to move it could narrow a printed CI by
    re-rendering. What these tests assert — which pairs are marked, and what the resampling unit
    is — is not a function of the count.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(report, "BOOTSTRAP_RESAMPLES", 200)
        yield


@dataclass(frozen=True)
class Rendered:
    """One render of one box: what the command printed, and the object it printed from."""

    box: Path
    flags: list[str]
    printed: str
    built: report.Report


def _render(box) -> Rendered:
    flags = _flags(box)
    stream = io.StringIO()
    with cheap_bootstrap(), redirect_stdout(stream):
        assert launch.main(["report", *flags]) == 0
        built = report.build(_config(flags))
    return Rendered(box=box, flags=flags, printed=stream.getvalue(), built=built)


@pytest.fixture(scope="module")
def rendered(tmp_path_factory, bar) -> Rendered:
    """The render every shape assertion reads.

    Module-scoped and rendered once: `report` is a pure function of the parquet, so twenty
    assertions about one file's table have no reason to build it twenty times — and the two
    tests that *do* need a second render are the two that are about re-rendering.
    """
    return _render(_make_box(tmp_path_factory.mktemp("box"), bar))


@pytest.fixture
def printed(rendered) -> str:
    return rendered.printed


@pytest.fixture
def built(rendered) -> report.Report:
    return rendered.built


@pytest.fixture
def box(rendered) -> Path:
    return rendered.box


@pytest.fixture
def flags(rendered) -> list[str]:
    return rendered.flags


# --- The four assertions the build spec names --------------------------------


def test_the_printed_tables_shape_is_exactly_r5s(built, bar):
    """66 (row, band) cells x 6 columns = 396 numbers, read off the bar rather than typed here."""
    r5 = bar.value("R5")
    assert len(built.table.cells) == r5["printed_cells"] == 66
    assert len(built.table.columns) == r5["printed_columns"] == 6
    assert built.table.columns == bar.headline_metrics
    assert built.table.numbers == r5["printed_numbers"] == 396


def test_the_blob_family_headlines_and_the_rectangle_family_is_not_printed(printed, built):
    """§5's second axis off the page: rectangles are drawn, scored, and print nowhere."""
    assert "rectangle" not in printed
    assert {cell.section for cell in built.table.cells} == {
        report.BLOB_FAMILY, *report.REAL_SUB_ROWS
    }


def test_the_size_buckets_pool_and_the_four_bands_stay_whole(printed, built):
    """p119 measured ~40x across the bands and ~3x across the buckets — the low-information axis
    is the one that comes off the page."""
    blob = [c for c in built.table.cells if c.section == report.BLOB_FAMILY]
    assert len(blob) == len(arms.PRINTED_ROWS) * len(report.BANDS) == 44
    assert {c.band for c in blob} == set(report.BANDS)
    for bucket, _ in BenchmarkConfig().masks.buckets:
        assert f" {bucket} " not in printed


def test_an_appendix_metric_never_reaches_print(printed, built, bar):
    """They were demoted for being redundant or unable to separate; printing re-imports that."""
    assert bar.appendix_metrics
    for name in bar.appendix_metrics:
        assert name not in built.table.columns
        assert name not in printed


def test_exactly_four_pairs_carry_a_ci_mark_of_the_fifty_five_enumerated(printed, built, bar):
    assert len({i.pair for i in built.marked}) == len(bar.bootstrap_pairs) == 4
    assert {i.pair for i in built.marked} == set(report.marked_pairs(bar))
    assert len(report.ALL_PAIRS) == bar.value("R6")["computed"] == 55
    assert "4 pairs carry a mark" in printed


def test_a_thin_cell_prints_a_count_and_no_metric(tmp_path, bar):
    """R8's floor, fired: a (row, band) cell under 100 instances is a count, not a number."""
    thin = _render(_make_box(tmp_path, bar, thin=[("A1", "blob", "R2")]))
    cell = thin.built.table.cell("A1", report.BLOB_FAMILY, "R2")
    assert cell.values is None
    assert cell.count == THIN * len(BenchmarkConfig().masks.buckets)
    assert cell.count < bar.cell_floor
    assert f"below R8's {bar.cell_floor}-instance floor" in thin.printed
    assert f"n={cell.count:,}" in thin.printed


def test_an_unscored_row_says_so_rather_than_printing_a_number(built):
    """Five of the eleven arms are not on this box, and the table's shape does not care."""
    cell = built.table.cell("A9", report.BLOB_FAMILY, "R1")
    assert cell.count == 0 and cell.values is None
    assert "not scored" in cell.note


# --- C4 ----------------------------------------------------------------------


def test_c4_prints_sixty_six_ranks_over_three_r4_instance_sets(built):
    block = built.ranks
    assert len(block.cells) == 66
    assert block.metrics == report.C4_METRICS
    assert block.sets == ("blob/small", *report.REAL_SUB_ROWS)
    ranked = [c for c in block.cells if c.rank is not None]
    assert ranked and all(1 <= c.rank <= len(arms.PRINTED_ROWS) for c in ranked)


def test_no_metres_difference_form_of_c4_is_reachable(built, printed):
    """§0 makes C4 an ordering claim, so the block holds no metre value to subtract."""
    for cell in built.ranks.cells:
        assert set(vars(cell)) == {"arm", "set_name", "metric", "rank", "n"}
    assert "Ranks, never metres" in printed


# --- The bootstrap -----------------------------------------------------------


def test_the_bootstrap_resamples_tiles_rather_than_instances(built):
    """~8 real voids sharing one host tile's terrain would overstate precision by up to sqrt(8).

    Asserted on the resampling unit rather than on an interval's width: the fixture puts four
    instances on every tile, so a per-instance resample would draw from a population four times
    larger, and every computed interval records the cluster count it actually drew from.
    """
    cell = built.table.cell("A1", report.BLOB_FAMILY, "R1")
    # The three pooled buckets land on the same tiles, so instances outrun clusters 12x.
    assert cell.count == PER_CELL * len(BenchmarkConfig().masks.buckets)
    assert cell.tiles == PER_CELL // PER_TILE
    assert cell.count == cell.tiles * PER_TILE * len(BenchmarkConfig().masks.buckets)
    blob = [
        i for i in built.intervals if i.section == report.BLOB_FAMILY and i.band == "R1"
    ]
    assert blob and {i.clusters for i in blob} == {cell.tiles}


def test_a_registered_pair_naming_an_unsplit_arm_resolves_through_r3(bar):
    """R6 names `A3 vs A1` and the registry splits A3 by init — R3 says which row is meant."""
    assert ("A3", "A1") in [tuple(p) for p in bar.bootstrap_pairs]
    assert report.marked_pairs(bar)[2] == ("A3-R5", "A1")
    with pytest.raises(ValueError, match="not one of R5's printed rows"):
        report.resolve_arm("A8")


def test_an_unmarked_pair_cannot_acquire_a_mark_by_clearing_zero(built, bar):
    """Which comparisons matter is fixed by the claims, not by which ones clear zero."""
    registered = set(report.marked_pairs(bar))
    clearing = [i for i in built.intervals if i.excludes_zero and not i.marked]
    assert clearing, "the fixture separates the arms, so unmarked pairs do clear zero"
    assert all(i.pair not in registered for i in clearing)


# --- Seeds, parameters, figures ----------------------------------------------


def test_exactly_one_seed_spread_maximum_is_printed_and_no_cell_carries_a_plus_minus(
    printed, built
):
    assert printed.count("maximum seed spread") == 1
    assert built.spread.seeds == 3
    assert "±" not in printed


def test_parameter_counts_print_total_and_trainable(printed, built):
    """A7's total reads as handicapped beside A4's until you know how much more of it trains."""
    assert set(built.parameters["A1"]) == {"total", "trainable"}
    assert built.parameters["A1"] == {"total": 0, "trainable": 0}
    assert "total" in printed and "trainable" in printed
    # An arm whose trainable half nothing has counted says so rather than printing a plausible 0.
    assert built.parameters["A7"]["total"] is None
    assert "not built yet" in printed


def test_all_three_figures_render_and_are_marked_load_bearing(printed, built, box):
    rendered = sorted(p.name for p in report.figures_dir(box / "results.parquet").glob("*.png"))
    assert rendered == [f"{figure.key}.png" for figure in sorted(report.FIGURES, key=lambda f: f.key)]
    assert len(report.FIGURES) == 3
    for figure in built.figures:
        assert figure.load_bearing
        assert f"{figure.key:<20} load-bearing" in printed


def test_the_size_axis_survives_in_the_figure_the_table_gave_it_up_for(built):
    """C2's 'at every void size' is the whole reason #136 could pool the buckets away."""
    size_figure = next(f for f in built.figures if f.key == "rmse-vs-void-size")
    assert {s.label for s in size_figure.series} == set(SCORED_ARMS)
    assert all(len(s.x) > len(BenchmarkConfig().masks.buckets) for s in size_figure.series)


def test_the_label_efficiency_curve_carries_the_sweep_and_the_table_does_not(built, printed):
    curve = next(f for f in built.figures if f.key == "label-efficiency")
    assert {s.label for s in curve.series} == set(SCORED_ARMS)
    assert all(len(s.x) == 3 for s in curve.series)
    # The table is the headline size alone: A1 is single-seed, so its pooled blob cell holds
    # three buckets of instances and not nine — the 1k and 5k rows print here and nowhere else.
    blob = built.table.cell("A1", report.BLOB_FAMILY, "R1")
    assert blob.count == PER_CELL * len(BenchmarkConfig().masks.buckets)


def test_the_panel_names_what_this_box_does_not_hold_rather_than_re_running_an_arm(built):
    """A hillshade needs pixels and a metrics parquet holds none — so the absence is labelled."""
    panel = next(f for f in built.figures if f.key == "hillshade-panel").panel
    assert panel.columns == ("truth", "voided", "A1", "A3-R5", "A5", "A7")
    assert len(panel.tiles) == len(report.PANEL_BANDS)
    assert all(cell.image is None and cell.absent for cell in panel.cells)
    assert any("no corpus at" in cell.absent for cell in panel.cells)


# --- Purity -------------------------------------------------------------------


def test_report_re_renders_without_re_running_an_arm(rendered):
    """The purity claim as a behaviour: a re-run would move the parquet, and it does not."""
    before = (rendered.box / "results.parquet").stat().st_mtime_ns
    again = _render(rendered.box)
    assert (rendered.box / "results.parquet").stat().st_mtime_ns == before
    # And the second render agrees to the digit: the bootstrap seed is a constant, so a CI is a
    # property of the file rather than of the run that printed it.
    assert again.printed == rendered.printed


def test_a_box_with_no_results_is_refused_rather_than_printing_an_empty_table(tmp_path, bar):
    """A frozen box that has scored nothing: refused, rather than an empty table with R5's shape."""
    (tmp_path / "artifacts").mkdir()
    prereg.write_bar(tmp_path / "artifacts" / BAR_NAME, plan=PLAN)
    with pytest.raises(FileNotFoundError, match="no results at"):
        launch.main(["report", *_flags(tmp_path)])


def test_a_render_that_did_not_match_r5s_arithmetic_would_be_refused(built, bar):
    """The guard, exercised: the render checks itself against the bar rather than against 66."""
    short = report.Table(columns=built.table.columns, cells=built.table.cells[:-1])
    with pytest.raises(ValueError, match="cells x"):
        report.assert_r5_shape(short, bar)


# --- What the two-axis review turned up ---------------------------------------


def test_a_cell_counts_instances_rather_than_parquet_rows(built):
    """R8's floor is stated in instances, and a three-seed arm has three rows per instance.

    Counting rows would let a three-seed arm clear a 100-instance floor on 34 instances while a
    single-pass arm fails it on 99 — the floor would mean something different per row of the
    table, which is the one thing a floor cannot do.
    """
    single, triple = "A1", "A4-R5"
    for arm in (single, triple):
        cell = built.table.cell(arm, report.BLOB_FAMILY, "R1")
        assert cell.count == PER_CELL * len(BenchmarkConfig().masks.buckets)
        assert cell.tiles == PER_CELL // PER_TILE


def test_the_bare_count_row_prints_as_a_count_and_is_not_one_of_the_sixty_six(printed, built):
    """§5: *"plus a bare instance count for 128 to 160 px"* — a footnote, never a row.

    It carries a count and no metric **by construction**: it sits above the sub-row boundaries
    that keep real-small size-comparable to synthetic small, so it has nothing to be compared
    against. That is a different thing from R8's floor, which withholds a metric from a row that
    would otherwise have earned one.
    """
    assert report.COUNT_ONLY_ROWS == ("count-only-128-160",)
    assert {c.section for c in built.table.cells}.isdisjoint(report.COUNT_ONLY_ROWS)
    assert f"count-only-128-160: {COUNT_ONLY:,} instance(s), counted and never scored" in printed


def test_the_registered_name_the_registry_lacks_is_resolved_in_print(printed, bar):
    """R6 says `A3`; the table has no such row. The reader is owed the rule that bridged it."""
    assert "R6 registers `A3`" in printed
    assert "Resolved to `A3-R5` under R3" in printed
    assert report.resolutions(bar)


def test_a_marked_pairs_interval_is_drawn_on_that_pairs_own_tiles(tmp_path, bar, rendered):
    """A pre-registered CI must not move because an arm that is not in it got run.

    Intersecting the tile set over *every* arm on the box would make C1's A4-R5 vs A5 narrow the
    day A9 lands, which is a comparison registered before any number existed depending on one
    that was not.
    """
    # The same box, plus A2-R5 — an arm in none of R6's four pairs — scored on half the tiles.
    lopsided = _render(
        _make_box(tmp_path, bar, extra_arms=("A2-R5",), half_tile_arms=("A2-R5",))
    )

    def clusters(rep, pair):
        return {
            i.clusters
            for i in rep.built.intervals
            if i.pair == pair and i.section == report.BLOB_FAMILY
        }

    c1 = report.marked_pairs(bar)[0]
    assert clusters(lopsided, c1) == clusters(rendered, c1) == {PER_CELL // PER_TILE}
    # A2-R5's own pairs do see the smaller set — the intersection is per pair, not absent.
    assert clusters(lopsided, ("A1", "A2-R5")) == {PER_CELL // PER_TILE // 2}


def test_the_rank_block_is_checked_against_its_own_arithmetic_and_not_r5s(built, bar):
    """§5: *"the two are unrelated ... nothing should be read across them."*

    Both counts are 66 and the coincidence is that both are 11 arms x 6. Checking the block
    against R5's registered cell count would be exactly the reading-across that sentence forbids,
    so the guard is stated in the block's own terms.
    """
    from dataclasses import replace as _replace

    report.assert_rank_block_shape(built.ranks)
    short = _replace(built.ranks, cells=built.ranks.cells[:-1])
    with pytest.raises(ValueError, match="rank cells"):
        report.assert_rank_block_shape(short)
    # And the guard is not pinned to 66: a block over two sets is 44 cells and passes, where a
    # check written against R5's registered count would reject it.
    two_sets = _replace(
        built.ranks,
        sets=built.ranks.sets[:2],
        cells=tuple(c for c in built.ranks.cells if c.set_name in built.ranks.sets[:2]),
    )
    assert len(two_sets.cells) == 44 != bar.printed_shape[0]
    report.assert_rank_block_shape(two_sets)


def test_a_figure_that_lost_its_load_bearing_marking_is_refused(built):
    """§9 marks all three and none may be cut for space, so an unmarked one is a defect."""
    from dataclasses import replace as _replace

    report.assert_load_bearing(built.figures)
    with pytest.raises(ValueError, match="load-bearing"):
        report.assert_load_bearing(
            [_replace(built.figures[0], load_bearing=False), *built.figures[1:]]
        )
