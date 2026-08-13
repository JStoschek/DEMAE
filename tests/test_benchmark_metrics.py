"""`benchmark.metrics.score(...)` — the numeric identities, and the golden tile (#155).

`docs/task1-build-spec.md` §Seam 3 names this file before it existed, and gives the reason it
does not attach at `benchmark.launch.main(argv)` like nearly everything else in this build: what
is asserted here are **equalities against another module or against a fixed constant**, and an
equality to 0.0006° reads badly through a CLI. Six of them:

1. ZT slope is numerically identical to `dataset.metrics.slope_deg` over a tile sample — the
   identity §5 chose the kernel *for*, since `slope_mean` off that same estimator is what the
   ruggedness bands are cut on.
2. NMAD is `1.4826 · median|Δh − m|` — Höhle & Höhle Eq. 2, against an independent expression
   rather than against itself.
3. The seam gap is full-mask minus eroded-mask slope RMSE **by construction**.
4. R7's +0.15° and R1's roster are read from the registered bar, never typed here.
5. Aspect error is circular.
6. Mean per-instance |bias| is reported and the pooled mean signed error is refused.

Plus the two the build spec attaches to this seam rather than to a claim: a metric outside the
**registered** set raises (#146 — registered, not graded, or the gate would refuse the seam gap
itself), and the **golden tile** — a committed tile with a known void and a hand-checked fill,
whose ten numbers #158 asserts again through the shipping path as A1.

Prior art is `tests/test_metrics.py`, and the style is its: every oracle is a closed form over a
synthetic surface, so a test that fails says which arithmetic broke rather than which number
moved.
"""

from __future__ import annotations

import ast
import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from benchmark import metrics, prereg
from benchmark.config import BenchmarkConfig
from dataset import surfaces
from dataset.geometry import band_pixel_spacing_m
from dataset.metrics import slope_deg

REPO = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO / "artifacts" / "task1"
PLAN = REPO / prereg.PLAN_PATH

#: Square 30 m pixels — Copernicus at the equator, and `tests/test_metrics.py`'s default. The
#: cases that turn on anisotropy spell their own out, because Copernicus's columns are up to
#: ~1.55x wider on the ground than its rows and a kernel that averaged the two would pass here.
PX = (30.0, 30.0)

MANIFEST = REPO / "data" / "corpus_500k" / "manifest.parquet"
SHARDS = REPO / "data" / "corpus_500k" / "shards"
requires_corpus = pytest.mark.skipif(
    not (MANIFEST.is_file() and SHARDS.is_dir()), reason="data/corpus_500k not present"
)


@pytest.fixture(scope="module")
def bar() -> prereg.Bar:
    """The committed bar, read from the artifacts directory `freeze` writes it to."""
    return prereg.read_bar(ARTIFACTS / prereg.BAR_NAME, plan=PLAN)


@pytest.fixture(scope="module")
def golden() -> metrics.GoldenTile:
    return metrics.read_golden(ARTIFACTS)


def _tile(shape=(64, 64), *, slope=8.0, aspect=35.0) -> np.ndarray:
    """A tilted plane with corrugations across it — terrain enough for a derivative."""
    plane = surfaces.inclined_plane(shape, pixel_size=PX, slope_deg=slope, aspect_deg=aspect)
    wave = surfaces.sinusoid(shape, pixel_size=PX, wavelength_m=480.0, amplitude=12.0, base=0.0)
    return plane + wave


def _void(shape=(64, 64), *, row0=20, row1=44, col0=18, col1=46) -> np.ndarray:
    void = np.zeros(shape, dtype=bool)
    void[row0:row1, col0:col1] = True
    return void


# --- 1. The kernel is the corpus's own ----------------------------------------


def test_zt_slope_is_the_corpus_slope_over_a_surface_sample():
    """p123 §5's identity: ZT's first derivatives *are* central differences.

    Asserted at 0.0006°, which is what the divergence measured at worst over 12,000 real
    instances — and observed here at machine zero, because the two expressions differ only in
    the order they subtract. That is the whole reason Horn was rejected: it sits ~7% lower in
    level, so adopting it would have made "benchmark slope" and "the `slope_mean` the ruggedness
    bands are cut on" two different quantities for no gain.
    """
    rng = np.random.default_rng(0)
    for surface in (
        _tile(),
        surfaces.ridge((64, 64), pixel_size=PX),
        surfaces.basin((64, 64), pixel_size=PX),
        surfaces.plateau((64, 64)),
        _tile() + rng.normal(0.0, 3.0, size=(64, 64)),
    ):
        zt = metrics.zt_surface(surface, PX).slope_deg()
        corpus = slope_deg(surface, PX)
        both = np.isfinite(zt) & np.isfinite(corpus)
        assert both.any()
        assert np.abs(zt[both] - corpus[both]).max() < 0.0006


def test_zt_slope_is_the_corpus_slope_under_anisotropic_pixels():
    """Rows and columns keep their own spacing — a kernel that shared one would pass at 30/30."""
    px = (30.9, 19.9)  # ~50 deg N: columns ~1.55x wider on the ground than rows
    z = surfaces.inclined_plane((48, 48), pixel_size=px, slope_deg=12.0, aspect_deg=70.0)
    zt = metrics.zt_surface(z, px).slope_deg()
    both = np.isfinite(zt) & np.isfinite(slope_deg(z, px))
    assert np.abs(zt[both] - slope_deg(z, px)[both]).max() < 0.0006
    assert zt[both].mean() == pytest.approx(12.0, abs=1e-9)


@requires_corpus
def test_zt_slope_is_the_corpus_slope_over_a_real_tile_sample():
    """The same identity on real terrain, at the pixel size the tile's own band gives it.

    Real tiles are what the 0.0006° figure was measured on, and they carry the case a synthetic
    surface does not: the ocean fill, which both estimators must NaN *before* differencing
    (ADR-0001 §3). A kernel that masked after would agree inland and diverge at every coast.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(MANIFEST, columns=["shard_id", "row_index", "lat"])
    shard_id = table.column("shard_id").to_numpy()
    row_index = table.column("row_index").to_numpy()
    lat = table.column("lat").to_numpy()

    rng = np.random.default_rng(0)
    shards = sorted(set(shard_id[:20_000].tolist()))[:2]
    chosen = rng.choice(np.flatnonzero(np.isin(shard_id, shards)), size=40, replace=False)

    cache: dict[int, np.ndarray] = {}
    worst = 0.0
    for i in chosen:
        sid = int(shard_id[i])
        if sid not in cache:
            cache[sid] = np.load(next(SHARDS.glob(f"*{sid:05d}*.npy")), mmap_mode="r")
        z = np.asarray(cache[sid][int(row_index[i])], dtype=np.float64)
        px = band_pixel_spacing_m(float(lat[i]))
        zt, corpus = metrics.zt_surface(z, px).slope_deg(), slope_deg(z, px)
        both = np.isfinite(zt) & np.isfinite(corpus)
        worst = max(worst, float(np.abs(zt[both] - corpus[both]).max()))
    assert worst < 0.0006


# --- 2. NMAD is Höhle & Höhle Eq. 2 -------------------------------------------


def test_nmad_is_the_scaled_median_deviation_from_the_median_error(bar):
    """`1.4826 · median|Δh − m|`, with *m* the median error — not the mean, and not zero.

    Written out independently here rather than reusing the module's expression, because the
    failure this guards is the plausible one: a median absolute *error* (deviation from zero)
    is a different statistic that agrees with this one on any symmetric fill, which is most
    synthetic ones.
    """
    rng = np.random.default_rng(7)
    truth = _tile()
    void = _void()
    err = rng.normal(3.0, 5.0, size=int(void.sum()))  # offset, so median error is not 0
    filled = truth.copy()
    filled[void] = truth[void] + err

    scored = metrics.score(truth, filled, void, PX, bar=bar)
    expected = 1.4826 * np.median(np.abs(err - np.median(err)))
    assert scored["nmad"] == pytest.approx(expected, rel=1e-12)
    assert scored["nmad"] != pytest.approx(1.4826 * np.median(np.abs(err)), rel=1e-3)
    assert metrics.NMAD_SCALE == 1.4826


# --- 3. The seam gap, by construction -----------------------------------------


def test_the_seam_gap_is_full_mask_minus_eroded_mask_slope_rmse(bar):
    """Derived, never measured: the gap is the subtraction, so the three columns cannot drift.

    This is also what makes the eroded-mask slope recoverable from the printed table by
    arithmetic rather than needing a column of its own (§5).
    """
    rng = np.random.default_rng(11)
    truth = _tile()
    void = _void()
    filled = truth.copy()
    filled[void] = truth[void] + rng.normal(0.0, 6.0, size=int(void.sum()))

    scored = metrics.score(truth, filled, void, PX, bar=bar)
    assert scored["seam_gap"] == scored["slope_rmse"] - scored["slope_rmse_eroded"]


def test_the_eroded_mask_is_the_void_without_its_boundary_ring():
    """Every surviving pixel's 3x3 kernel reads predicted values only — that is the control."""
    void = _void((32, 32), row0=10, row1=20, col0=10, col1=20)
    eroded = metrics.erode(void)
    assert void.sum() == 100
    assert eroded.sum() == 64
    assert not (eroded & ~void).any()
    edge = np.zeros((8, 8), dtype=bool)
    edge[0, 0] = True  # a void touching the array edge keeps nothing: border_value is unset
    assert not metrics.erode(edge).any()


def test_a_void_too_thin_to_erode_scores_no_contamination_control(bar):
    """NaN, not zero: "no control here" and "a perfect interior" are different readings."""
    truth = _tile()
    void = np.zeros((64, 64), dtype=bool)
    void[30, 20:40] = True  # one pixel thick, so erosion leaves nothing
    filled = truth.copy()
    filled[void] = truth[void] + 5.0

    scored = metrics.score(truth, filled, void, PX, bar=bar)
    assert math.isnan(scored["slope_rmse_eroded"])
    assert math.isnan(scored["seam_gap"])
    assert scored["bias_abs"] == pytest.approx(5.0)


# --- 4. The bar, not a literal ------------------------------------------------


def test_the_seam_threshold_is_read_from_the_bar(bar):
    """R7's +0.15°, strictly above — and the verdict follows the *bar*, not this module."""
    assert bar.seam_break_deg == 0.15
    assert metrics.is_seam_break(0.151, bar=bar)
    assert not metrics.is_seam_break(0.15, bar=bar)
    assert not metrics.is_seam_break(-1.33, bar=bar)  # a smooth fill is not a break

    moved = _bar_with(bar, "R7", 0.5)
    assert not metrics.is_seam_break(0.3, bar=moved)
    assert metrics.is_seam_break(0.3, bar=bar)


def test_the_seam_threshold_is_nowhere_typed_in_this_module():
    """Past `benchmark.prereg` no threshold in the benchmark is a literal, and this is the
    module that would otherwise have typed one — R7's 0.15 sits one function call away from
    `is_seam_break`. Parsed rather than grepped, so the prose that *quotes* +0.15° in the
    docstrings does not read as a constant and a constant cannot hide inside a comment.
    """
    tree = ast.parse(Path(metrics.__file__).read_text())
    assert 0.15 not in {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float))
    }
    # And `is_seam_break` compares against something it was handed, not a number of its own.
    verdict = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "is_seam_break"
    )
    assert not [
        node
        for node in ast.walk(verdict)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float))
    ]


def test_the_metric_set_is_the_bars_roster_and_nothing_else(bar):
    """R1's roster is the single home for the metric names — five graded, the derived seam gap,
    and four appendix ones (#146). `score` returns exactly that set, in that order."""
    truth, void = _tile(), _void()
    filled = truth.copy()
    filled[void] = truth[void] + 2.0

    scored = metrics.score(truth, filled, void, PX, bar=bar)
    assert tuple(scored) == bar.registered_metrics
    assert tuple(scored) == (
        "rmse", "nmad", "bias_abs", "slope_rmse", "aspect_error", "seam_gap",
        "mae", "p95", "profile_curvature", "slope_rmse_eroded",
    )
    # The seam gap is *in* the roster rather than carved out by name, which is the #146 fix: a
    # gate keyed on the ten p123 *graded* candidates would raise on the sixth printed column.
    assert "seam_gap" in bar.registered_metrics
    assert "seam_gap" not in bar.value("R1")["graded"]


def test_a_metric_outside_the_registered_set_raises(bar):
    """Refused rather than quietly computed: a number no pre-registration covers is a number
    the results parquet has no business carrying."""
    truth, void = _tile(), _void()
    filled = truth.copy()
    filled[void] = truth[void] + 2.0

    with pytest.raises(ValueError, match="outside the registered metric set"):
        metrics.score(truth, filled, void, PX, bar=bar, metrics=["rmse", "hillshade_ssim"])
    with pytest.raises(ValueError, match="outside the registered metric set"):
        metrics.score(truth, filled, void, PX, bar=bar, metrics=["slope_horn"])
    # And an empty selection is refused rather than returning an empty row — a scored instance
    # with no metrics in it is a row the parquet would carry and nothing would notice.
    with pytest.raises(ValueError, match="no metrics requested"):
        metrics.score(truth, filled, void, PX, bar=bar, metrics=[])


def test_the_membership_gate_follows_the_bar_it_is_given(bar):
    """Membership comes from the file, not from a set typed here: a bar whose roster has lost a
    name refuses that name, and one that never had it never computes it."""
    truth, void = _tile(), _void()
    filled = truth.copy()
    filled[void] = truth[void] + 2.0

    roster = copy.deepcopy(bar.value("R1"))
    roster["appendix"] = [name for name in roster["appendix"] if name != "p95"]
    thinner = _bar_with(bar, "R1", roster)

    assert "p95" not in thinner.registered_metrics
    assert "p95" not in metrics.score(truth, filled, void, PX, bar=thinner)
    with pytest.raises(ValueError, match="outside the registered metric set"):
        metrics.score(truth, filled, void, PX, bar=thinner, metrics=["p95"])
    assert "p95" in metrics.score(truth, filled, void, PX, bar=bar)


def _bar_with(bar: prereg.Bar, rule: str, value) -> prereg.Bar:
    """The committed bar with one registered value moved — a stand-in, never written to disk.

    `prereg.read_bar` would refuse this file, and that is the point: the anchor check is what
    stops a threshold moving in the artifact. What this fixture isolates is the *other* half —
    that this module reads whatever the bar it was handed says, rather than a constant that
    happens to agree with it.
    """
    doc = copy.deepcopy(bar.doc)
    for entry in doc["entries"]:
        if entry["rule"] == rule:
            entry["value"] = value
    return prereg.Bar(doc=doc, path=bar.path)


# --- 5. Aspect is circular ----------------------------------------------------


def test_circular_error_wraps_at_the_branch_cut():
    """359 deg and 1 deg are two degrees apart; a linear difference reads them as 358."""
    a = np.radians(np.array([1.0, 350.0, 90.0, -179.0]))
    b = np.radians(np.array([359.0, 10.0, 270.0, 179.0]))
    assert np.degrees(metrics.circular_error_rad(a, b)) == pytest.approx([2.0, 20.0, 180.0, 2.0])


def test_aspect_error_is_circular_across_the_branch_cut(bar):
    """A fill rotated one way and the other are equally wrong, even across ±180 deg.

    Built on planes so the oracle is exact: a plane's aspect is one number everywhere, so a fill
    that replaces the void with a plane rotated by delta has exactly that aspect error on every
    pixel whose kernel is inside the void. The truth here is chosen to sit **on the branch cut**
    — an east-falling plane reads atan2(+0, −) = 180 deg — so one of the two rotations lands at
    −172 deg and a linear difference would report it as 352 deg rather than 8.
    """
    shape, delta = (96, 96), 8.0
    truth = surfaces.inclined_plane(shape, pixel_size=PX, slope_deg=10.0, aspect_deg=90.0)
    void = _void(shape, row0=24, row1=72, col0=24, col1=72)
    interior = metrics.erode(void)

    def rotated(by: float) -> tuple[float, float]:
        other = surfaces.inclined_plane(shape, pixel_size=PX, slope_deg=10.0, aspect_deg=90.0 + by)
        filled = truth.copy()
        # Matched at the void's mean, so the rotation is what is being scored and not a step.
        filled[void] = other[void] - other[void].mean() + truth[void].mean()
        pixel = np.degrees(
            metrics.circular_error_rad(
                metrics.zt_surface(filled, PX).aspect_rad(),
                metrics.zt_surface(truth, PX).aspect_rad(),
            )
        )
        scored = metrics.score(truth, filled, void, PX, bar=bar, metrics=["aspect_error"])
        return float(np.abs(pixel[interior] - delta).max()), scored["aspect_error"]

    for by in (delta, -delta):
        worst, _ = rotated(by)
        assert worst < 1e-9  # every interior pixel is exactly `delta` out
    assert rotated(delta)[1] == pytest.approx(rotated(-delta)[1], rel=1e-9)


def test_aspect_error_ignores_a_pixel_with_no_aspect_to_get_wrong(bar):
    """Slope-weighted: flat truth contributes nothing, rather than contributing noise."""
    shape = (64, 64)
    truth = surfaces.plateau(shape, height=800.0)
    void = _void(shape, row0=20, row1=44, col0=20, col1=44)
    rng = np.random.default_rng(3)
    filled = truth.copy()
    filled[void] = truth[void] + rng.normal(0.0, 4.0, size=int(void.sum()))

    scored = metrics.score(truth, filled, void, PX, bar=bar, metrics=["aspect_error"])
    assert math.isnan(scored["aspect_error"])


# --- 6. Bias: the magnitude reported, the pooled sign refused -----------------


def test_bias_is_this_instances_mean_error_as_a_magnitude(bar):
    """A fill 4 m high and one 4 m low are equally wrong, and score identically."""
    truth, void = _tile(), _void()
    high, low = truth.copy(), truth.copy()
    high[void] = truth[void] + 4.0
    low[void] = truth[void] - 4.0

    assert metrics.score(truth, high, void, PX, bar=bar)["bias_abs"] == pytest.approx(4.0)
    assert metrics.score(truth, low, void, PX, bar=bar)["bias_abs"] == pytest.approx(4.0)


def test_the_pooled_mean_signed_error_is_refused_by_name(bar):
    """§5's refusal, with its reason: pooled it cancels to a null column for every arm.

    Refused *and* unavailable — the signed value is not among the keys `score` returns, so no
    aggregation downstream can average the sign away even if it wanted to.
    """
    truth, void = _tile(), _void()
    filled = truth.copy()
    filled[void] = truth[void] + 4.0

    scored = metrics.score(truth, filled, void, PX, bar=bar)
    assert "bias" not in scored and "mean_error" not in scored
    for name in ("bias", "mean_signed_error", "me"):
        with pytest.raises(ValueError, match="pooled mean signed error"):
            metrics.score(truth, filled, void, PX, bar=bar, metrics=[name])


def test_a_fill_at_the_wrong_level_is_visible_where_a_pooled_sign_would_cancel(bar):
    """The failure the guard exists for: two instances offset in opposite directions.

    Averaged as signed numbers they are a null; averaged as magnitudes, which is the only thing
    R1's column permits, they are the 4 m offset both of them carry.
    """
    truth, void = _tile(), _void()
    biases = []
    for offset in (+4.0, -4.0):
        filled = truth.copy()
        filled[void] = truth[void] + offset
        biases.append(metrics.score(truth, filled, void, PX, bar=bar)["bias_abs"])
    assert np.mean(biases) == pytest.approx(4.0)


# --- Everything in metres, and nothing outside the void -----------------------


def test_a_fill_left_in_its_own_normalized_space_is_refused(bar):
    """Plan §2.1, as a refusal rather than a convention.

    An arm writes inside the void and nowhere else, so the untouched pixels are a free assertion
    about units — a per-tile standardized array sits within a few sigma of zero where the terrain
    it came from is hundreds of metres. This is the bug §8 lists as a risk precisely because it
    does not raise on its own: a scale error flatters or sabotages an arm and prints normally.
    """
    truth, void = _tile(), _void()
    valid = np.ones(truth.shape, dtype=bool) & ~void
    mean, std = truth[valid].mean(), truth[valid].std()
    standardized = (truth - mean) / std

    with pytest.raises(ValueError, match="not the tile it was handed"):
        metrics.score(truth, standardized, void, PX, bar=bar)


def test_an_arm_that_wrote_outside_its_void_is_refused(bar):
    truth, void = _tile(), _void()
    filled = truth.copy()
    filled[void] = truth[void] + 2.0
    filled[0, 0] += 3.0

    with pytest.raises(ValueError, match="outside the void"):
        metrics.score(truth, filled, void, PX, bar=bar)


def test_to_metres_inverts_the_extractors_own_normalization():
    """The inversion reads its statistics from `extract.tile_statistics` — one code path.

    `valid = land ∧ ¬void`, so the statistics a fill is returned to metres by are the non-void
    ones it was handed: a voided tile's true statistics are unknowable at inference, and
    standardizing by them would leak the answer into the input (§2.1).
    """
    from benchmark import extract

    truth = _tile((224, 224)).astype(np.int16)
    void = _void((224, 224), row0=60, row1=100, col0=90, col1=130)
    valid = (truth != 0) & ~void

    standardized = extract.normalize(truth[None], valid[None], normalization="per_tile")[0]
    recovered = metrics.to_metres(standardized, truth, valid)
    assert np.abs(recovered - truth).max() < 1e-2

    # And the statistics are the non-void ones: a mask that includes the void gives a different
    # answer, which is exactly the leak §2.1 refuses.
    leaky = metrics.to_metres(standardized, truth, truth != 0)
    assert np.abs(leaky - truth).max() > 1e-2


# --- Scoring the whole instance -----------------------------------------------


def test_a_perfect_fill_scores_zero_on_every_column(bar):
    truth, void = _tile(), _void()
    scored = metrics.score(truth, truth.copy(), void, PX, bar=bar)
    for name, value in scored.items():
        assert value == pytest.approx(0.0, abs=1e-9), name


def test_scoring_happens_only_inside_the_void(bar):
    """Terrain outside the void cannot move a number — the mask is the whole population."""
    rng = np.random.default_rng(5)
    truth, void = _tile(), _void()
    filled = truth.copy()
    filled[void] = truth[void] + rng.normal(0.0, 5.0, size=int(void.sum()))

    elsewhere = truth.copy()
    elsewhere[:12, :12] += 400.0  # a mountain nowhere near the void
    moved = filled.copy()
    moved[:12, :12] += 400.0

    before = metrics.score(truth, filled, void, PX, bar=bar)
    after = metrics.score(elsewhere, moved, void, PX, bar=bar)
    assert before == pytest.approx(after)


def test_an_empty_void_is_not_an_instance(bar):
    truth = _tile()
    with pytest.raises(ValueError, match="empty void"):
        metrics.score(truth, truth.copy(), np.zeros(truth.shape, dtype=bool), PX, bar=bar)


def test_a_fill_of_a_different_shape_is_refused(bar):
    truth, void = _tile(), _void()
    with pytest.raises(ValueError, match="one tile"):
        metrics.score(truth, _tile((32, 32)), void, PX, bar=bar)


def test_the_boundary_ring_is_where_a_discontinuous_fill_is_caught(bar):
    """p123 §4's reversal, as a behaviour: a fill that steps at the void edge is caught by the
    full mask and invisible to the eroded one, so the gap is positive and large.

    A smooth fill of the same terrain scores a gap near zero — which is why the threshold is
    +0.15° rather than `> 0`, and why the eroded score is a control rather than the primary.
    """
    truth, void = _tile(), _void()
    step, smooth = truth.copy(), truth.copy()
    step[void] = truth[void] + 6.0  # right in shape, wrong in level: a cliff at the boundary
    smooth[void] = truth[void] + 0.02

    assert metrics.score(truth, step, void, PX, bar=bar)["seam_gap"] > bar.seam_break_deg
    assert abs(metrics.score(truth, smooth, void, PX, bar=bar)["seam_gap"]) < bar.seam_break_deg


# --- The golden tile ----------------------------------------------------------


def test_the_golden_tile_scores_exactly_what_it_is_committed_to_score(golden, bar):
    """§8's gate: a committed tile, a known void, a hand-checked fill, and ten pinned numbers.

    #158 drives this same tile through `python -m benchmark.launch` as A1, so the numbers are
    asserted at both altitudes — the seam and the shipping path — off one committed fixture.
    """
    scored = metrics.score(golden.truth, golden.filled, golden.void, golden.px, bar=bar)
    assert set(scored) == set(golden.expected)
    for name, expected in golden.expected.items():
        assert scored[name] == pytest.approx(expected, rel=1e-12, abs=1e-12), name
        assert golden.derivations[name]


def test_the_golden_tiles_elevation_columns_close_in_arithmetic(golden, bar):
    """The hand-check, spelled out: b = 4.0 m of offset and a = 1.5 m of ripple, 800 pixels
    each way, so five of the ten columns are arithmetic rather than observation."""
    b = golden.recipe["offset_m"]
    a = golden.recipe["ripple_m"]
    scored = metrics.score(golden.truth, golden.filled, golden.void, golden.px, bar=bar)

    assert scored["rmse"] == pytest.approx(math.sqrt(b**2 + a**2), rel=1e-15)
    assert scored["mae"] == pytest.approx(b, rel=1e-15)
    assert scored["p95"] == pytest.approx(b + a, rel=1e-15)
    assert scored["nmad"] == pytest.approx(metrics.NMAD_SCALE * a, rel=1e-15)
    assert scored["bias_abs"] == pytest.approx(b, rel=1e-15)


def test_the_golden_tiles_ripple_is_invisible_to_slope_and_caught_at_the_seam(golden, bar):
    """The fixture's whole point, and p123 §4's finding in one tile.

    A checkerboard flips sign between a pixel and *both* its neighbours along either axis, so it
    cancels exactly out of a central difference: the eroded-mask slope is **zero** on a fill
    sitting 4 m off with a broken seam. The full mask reads the step at the 156-pixel boundary
    ring, and the gap is a seam break. Eroding the ring does not remove a free ride; it deletes
    the evidence.
    """
    scored = metrics.score(golden.truth, golden.filled, golden.void, golden.px, bar=bar)
    assert scored["slope_rmse_eroded"] == 0.0
    assert scored["seam_gap"] == scored["slope_rmse"]
    assert metrics.is_seam_break(scored["seam_gap"], bar=bar)

    ring = golden.void & ~metrics.erode(golden.void)
    assert ring.sum() == 156 and golden.void.sum() == 1600
    slope_err = (
        metrics.zt_surface(golden.filled.astype(np.float64), golden.px).slope_deg()
        - metrics.zt_surface(golden.truth.astype(np.float64), golden.px).slope_deg()
    )
    assert np.abs(slope_err[metrics.erode(golden.void)]).max() == 0.0
    ring_rmse = math.sqrt(float(np.mean(slope_err[ring] ** 2)))
    assert scored["slope_rmse"] == pytest.approx(
        ring_rmse * math.sqrt(ring.sum() / golden.void.sum()), rel=1e-12
    )


def test_the_golden_tiles_aspect_error_is_the_ring_diluted_over_the_void(golden, bar):
    """The committed aspect number, re-derived rather than taken on trust.

    It is the one column of the ten whose value is neither closed form nor structurally pinned
    elsewhere, so without this it would be asserted only against itself. The derivation is the
    fixture's own geometry: the ripple cancels out of a central difference, so aspect is exact on
    every interior pixel; the truth is a plane, so the sin(slope) weight is one number and the
    weighted mean collapses to the ring's total circular error over the whole void.
    """
    interior = metrics.erode(golden.void)
    ring = golden.void & ~interior
    aspect_err = np.degrees(
        metrics.circular_error_rad(
            metrics.zt_surface(golden.filled.astype(np.float64), golden.px).aspect_rad(),
            metrics.zt_surface(golden.truth.astype(np.float64), golden.px).aspect_rad(),
        )
    )
    assert np.abs(aspect_err[interior]).max() == 0.0
    scored = metrics.score(golden.truth, golden.filled, golden.void, golden.px, bar=bar)
    assert scored["aspect_error"] == pytest.approx(
        aspect_err[ring].sum() / golden.void.sum(), rel=1e-12
    )


def test_the_golden_tiles_curvature_sees_what_slope_cannot(golden):
    """ZT's *second* differences do not cancel the ripple: +/- 4a/dx^2 on every interior pixel."""
    a, (dy, dx) = golden.recipe["ripple_m"], golden.px
    interior = metrics.erode(golden.void)
    k_err = (
        metrics.zt_surface(golden.filled.astype(np.float64), golden.px).profile_curvature()
        - metrics.zt_surface(golden.truth.astype(np.float64), golden.px).profile_curvature()
    )
    assert np.abs(k_err[interior]) == pytest.approx(4 * a / dx**2, rel=1e-12)
    assert dy == dx  # the closed form above is the square-pixel case, which the recipe fixes


def test_the_golden_tile_is_a_small_bucket_void_well_inside_the_tile(golden):
    """§3.2's constraints, so the fixture is an instance the eval mask list could have drawn.

    Read off `BenchmarkConfig().masks` rather than typed here: §3.2's margin, cap and bucket
    edges have one home, and a fixture checked against a second copy of them is a fixture that
    stays "interior" after the generator's definition of interior has moved.
    """
    geometry = BenchmarkConfig().masks
    rows, cols = np.nonzero(golden.void)
    height, width = golden.void.shape
    margin = min(rows.min(), cols.min(), height - 1 - rows.max(), width - 1 - cols.max())
    assert margin >= geometry.edge_margin_px
    assert golden.void.mean() <= geometry.max_area_frac
    low, high = geometry.small_px
    assert low <= math.sqrt(4 * golden.void.sum() / math.pi) <= high


def test_the_golden_tile_refuses_a_sidecar_that_has_drifted_from_its_arrays(tmp_path, golden):
    """Two files, one fixture: the scores and the pixels they are scores *of* are pinned by md5."""
    metrics.write_golden(tmp_path, golden.expected, golden.derivations)
    doc = json.loads((tmp_path / metrics.GOLDEN_SIDECAR).read_text())
    doc["arrays_md5"] = "0" * 32
    (tmp_path / metrics.GOLDEN_SIDECAR).write_text(json.dumps(doc))

    with pytest.raises(ValueError, match="drifted apart"):
        metrics.read_golden(tmp_path)


def test_the_golden_tile_refuses_arrays_that_are_no_longer_its_recipe(tmp_path, golden):
    """The recipe is what makes the fill hand-checked rather than merely committed."""
    metrics.write_golden(tmp_path, golden.expected, golden.derivations)
    doc = json.loads((tmp_path / metrics.GOLDEN_SIDECAR).read_text())
    doc["recipe"]["offset_m"] = 9.0
    (tmp_path / metrics.GOLDEN_SIDECAR).write_text(json.dumps(doc))

    with pytest.raises(ValueError, match="not what .* recipe builds"):
        metrics.read_golden(tmp_path)


def test_the_committed_golden_tile_is_the_recipe_this_module_carries(golden):
    """The committed file and `GOLDEN_RECIPE` are one recipe, so the derivations describe it."""
    assert golden.recipe == json.loads(json.dumps(metrics.GOLDEN_RECIPE))
    assert golden.truth.dtype == np.int16  # metres, as a shard stores a tile
    assert golden.valid_mask.sum() == golden.truth.size - golden.void.sum()
