"""§5's registered metric set, scored inside the void and in metres (#155).

``score(truth, filled, void, px) -> dict`` is the whole surface, and it returns **R1's roster
read from the registered bar** — the five graded headline columns, the derived seam gap and the
four appendix ones. Promoted from ``prototypes/p123_metric_discrimination_proto.py``, whose
Zevenbergen–Thorne kernel is the one the **0.0006°** divergence against
:func:`dataset.metrics.slope_deg` was measured on
([p123](../../docs/research/p123-metric-discrimination.md) §5), with the prototype's Horn
kernel, hillshade SSIM and pooled signed bias left behind — all three were *measured out* of the
set rather than dropped for convenience.

## Metres, and the one place an arm could have been scored in its own units

Plan §2.1: *loss and metrics are always computed in metres after inverting DEMAE's per-tile
transform with the non-void stats, so no arm is scored in its own normalized space.* That is one
sentence and two separate hazards.

The inversion is :func:`to_metres`, and it reads its statistics from
:func:`benchmark.extract.tile_statistics` rather than recomputing them — the same function, over
the same ``valid = land ∧ ¬void`` mask, that :func:`benchmark.extract.normalize` standardized
with. A second implementation of "the per-tile mean and std" is exactly the drift ADR-0012
Decision 2 exists to prevent, and here it would land as a *scale error on every printed number*
rather than as a crash.

The second hazard is a caller that simply never inverts, and it is not caught by having an
inverter available. So :func:`score` checks the one thing a normalized array cannot fake: an arm
writes **inside the void and nowhere else**, so ``filled`` must still equal ``truth`` outside it,
in metres, to float32's tolerance. An array left in per-tile units fails that on its first
untouched pixel, with a message that names the inversion. A wrong-units arm therefore cannot
score a plausible-looking table — which is the failure mode §8's risk row is about, since a
scale error flatters or sabotages an arm without ever raising.

## Everything inside the void, and the derivatives on the *full* mask

Elevation error is scored over ``void`` and nothing else. The derivatives — slope, aspect,
curvature — are scored over the full void mask too, with the 1-px-eroded mask kept beside them
as a **contamination control** rather than as the primary. That is p123 §4 reversing the plan's
own prior expectation, and it is the reason this module computes both:

    A 3×3 kernel one pixel inside the void does read six true pixels, and that was expected to
    hand an idle arm a free ride. Measured, the effect runs the other way — eroding the ring
    **halves** slope's separation (1.89 → 0.91), because the boundary ring is not where a lazy
    fill hides, it is where a **discontinuous** one is caught.

The free ride is real and small (−0.005 to −0.107° on the smooth interpolators) against 17% of
the scored pixels on a small void. The eroded score is published as an appendix column, and the
difference of the two is the **seam gap** — free, since both masks are computed anyway, and the
sixth printed column (R1, #146).

## What is read from the bar rather than typed

Two things, and both would otherwise be literals here:

1. **The metric set's membership.** :func:`score` gates on ``bar.registered_metrics`` — R1's
   roster — so a metric outside it raises rather than being quietly computed. *Registered, not
   graded* (#146): the seam gap is derived and was never one of the ten p123 candidates, so a
   gate keyed on the graded set would raise on the very column §5 requires this module to
   return. R1's roster is the single list; the seam gap is **not carved out by name** anywhere,
   because a second home for the metric roster is what produced #146 in the first place.
2. **R7's +0.15°**, via :func:`is_seam_break`. Past :mod:`benchmark.prereg` no threshold in this
   benchmark is typed as a literal, and this is the module that would have typed one.

## The pooled mean signed error, refused

§5 registers **mean per-instance |bias|** and refuses the pooled mean signed error. The refusal
is measured rather than stylistic: pooled over the corpus the signed statistic cancels to
0.003–0.9 m against an MAE of 0.87–38.6 m, so it prints a null column for every arm, while ~half
of a typical fill's error is a DC offset whose sign is arbitrary per tile (p123 §6). Roughly
42–50% of instances carry an offset exceeding half their own MAE.

So this module returns ``bias_abs`` — the magnitude of *this instance's* mean signed error — and
never the signed value. That is the refusal as a mechanism rather than as a note: the sign does
not reach the results parquet, so no aggregation downstream can average it away, and the only
pooling available is the mean of magnitudes R1 registers. Asking for it by name raises, and says
this (:data:`REFUSED`).

## The golden tile

A committed tile with a known void and a hand-checked fill, plus the ten scores it must produce
— :func:`read_golden`, and see :data:`GOLDEN_RECIPE` for what the fill is and why. It gates
every arm's eval at this seam, and #158 drives the same tile through
``benchmark.launch main(argv)`` as A1 so the same numbers are asserted at both altitudes.
"""

from __future__ import annotations

import functools
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterable, Mapping

import numpy as np

from dataset import config as dataset_config
from dataset.geometry import PixelSize
from dataset.metrics import land_mask

from . import extract, prereg
from .config import BenchmarkConfig

#: Höhle & Höhle 2009 Eq. 2's constant: NMAD is ``1.4826 · median|Δh − m|``, the factor that
#: makes the median absolute deviation a consistent estimator of σ under a normal distribution.
#: §5 takes NMAD (and p95) from that paper's Table 2, whose argument is that DEM error
#: distributions are heavy-tailed and the Gaussian measures are therefore the wrong ones.
NMAD_SCALE: Final = 1.4826

#: The tail quantile of ``|Δh|`` the same table proposes, in per cent. Appendix, not headline:
#: p123 §3 measured it 0.99 redundant with RMSE and the weakest of the three elevation metrics.
P95_QUANTILE: Final = 95.0

#: The structuring element the contamination control erodes by, in pixels. One, so that every
#: surviving pixel's 3×3 ZT kernel reads only predicted values — which is the whole definition
#: of the control, and why it is 1 and not a tunable.
EROSION_PX: Final = 1


# --- The Zevenbergen–Thorne surface -------------------------------------------


def zt_coefficients(z: np.ndarray, px: PixelSize) -> dict[str, np.ndarray]:
    """Zevenbergen & Thorne (1987) quadratic-surface coefficients ``D, E, F, G, H``, per pixel.

    The interior only — the one-pixel border is NaN, where the 3×3 window does not fit. Rows and
    columns carry their own spacing, because Copernicus's columns are up to ~1.55× wider on the
    ground than its rows near 50° (ADR-0001 §1) and a single ``px`` would quietly rotate every
    aspect poleward.

    The mask goes on **before** the differences, per ADR-0001 §3 and for
    :func:`dataset.metrics.slope_deg`'s reason: the ocean fill is painted, not measured, so the
    step down to it is a digitization artifact and a resolution-dependent one. That ordering is
    also what makes ``G``/``H`` *the same estimator* as the corpus's central-difference gradient
    on a coastal tile and not only on an inland one.
    """
    dy, dx = px
    a = np.where(land_mask(z), np.asarray(z, dtype=np.float64), np.nan)
    z1, z2, z3 = a[:-2, :-2], a[:-2, 1:-1], a[:-2, 2:]
    z4, z5, z6 = a[1:-1, :-2], a[1:-1, 1:-1], a[1:-1, 2:]
    z7, z8, z9 = a[2:, :-2], a[2:, 1:-1], a[2:, 2:]
    inner = {
        "D": ((z4 + z6) / 2 - z5) / dx**2,
        "E": ((z2 + z8) / 2 - z5) / dy**2,
        "F": (-z1 + z3 + z7 - z9) / (4 * dx * dy),
        "G": (-z4 + z6) / (2 * dx),
        "H": (z2 - z8) / (2 * dy),
    }
    out: dict[str, np.ndarray] = {}
    for key, value in inner.items():
        full = np.full(a.shape, np.nan)
        full[1:-1, 1:-1] = value
        out[key] = full
    return out


def zt_slope_deg(z: np.ndarray, px: PixelSize) -> np.ndarray:
    """Per-pixel slope in degrees off the ZT surface — ``arctan|∇z|``, NaN on the border.

    **Numerically the corpus's own** :func:`dataset.metrics.slope_deg`: ZT's first derivatives
    ``G`` and ``H`` are central differences by construction, so the two agree to 0.0006° at worst
    over 12,000 instances (p123 §5). That identity is the reason this kernel was chosen and Horn
    was not — Horn sits ~7% lower in level and ranks instances 0.980 the same, so it would have
    changed every printed number without changing a conclusion, at the cost of making *benchmark
    slope* and *the `slope_mean` the ruggedness bands are cut on* two different quantities.
    """
    c = zt_coefficients(z, px)
    return np.degrees(np.arctan(np.hypot(c["G"], c["H"])))


def zt_aspect_rad(z: np.ndarray, px: PixelSize) -> np.ndarray:
    """Per-pixel aspect in radians off the same surface, ``atan2(H, G)``, NaN on the border.

    The zero direction never leaves this module: aspect is only ever *differenced* between truth
    and fill (:func:`circular_error_rad`), so any consistent convention gives the same circular
    error. What does matter is that it comes off ZT rather than off a second kernel — §5 puts
    slope, aspect and curvature on one quadratic surface precisely so the three cannot disagree
    about what the local gradient is.
    """
    c = zt_coefficients(z, px)
    return np.arctan2(c["H"], c["G"])


def zt_profile_curvature(z: np.ndarray, px: PixelSize) -> np.ndarray:
    """ZT profile curvature in 1/m: curvature along the direction of steepest slope.

    **Undefined on flat ground**, where there is no steepest direction — NaN there rather than a
    divide-by-zero, which matters because the flattest ruggedness band is a quarter of the
    corpus and would otherwise fill an appendix column with garbage.
    """
    c = zt_coefficients(z, px)
    g, h = c["G"], c["H"]
    p = g**2 + h**2
    with np.errstate(invalid="ignore", divide="ignore"):
        kp = -2 * (c["D"] * g**2 + c["E"] * h**2 + c["F"] * g * h) / p
    return np.where(p > 1e-12, kp, np.nan)


def circular_error_rad(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Shortest angular distance between two aspect fields, radians, in ``[0, π]``.

    Circular because aspect is: 359° and 1° are two degrees apart, and a linear difference would
    read them as 358 and hand every north-facing pixel a spurious error. ``atan2(sin, cos)``
    rather than a modulo, so the wrap is exact at the branch cut instead of nearly exact.
    """
    d = a - b
    return np.abs(np.arctan2(np.sin(d), np.cos(d)))


# --- The masks ----------------------------------------------------------------


def erode(mask: np.ndarray) -> np.ndarray:
    """``mask`` eroded by a 3×3 structuring element, off the array edge as if unset.

    A pixel survives iff its whole 3×3 neighbourhood is inside the mask — so every surviving
    pixel's ZT kernel reads predicted values only, which is what makes the eroded score a
    contamination control rather than a second opinion.

    Nine ANDed slices rather than ``scipy.ndimage.binary_erosion``: :mod:`benchmark` is
    torch-optional so that `freeze` and the scoring path run on a box with no CUDA stack, and the
    same argument applies one layer down — this is the whole of the dependency, it is six lines,
    and scipy is in the environment only transitively through scikit-learn.
    """
    if mask.ndim != 2:
        raise ValueError(f"the void mask is a 2-D boolean array; got shape {mask.shape}")
    h, w = mask.shape
    padded = np.pad(mask.astype(bool), EROSION_PX, constant_values=False)
    out = np.ones_like(mask, dtype=bool)
    for dr in range(2 * EROSION_PX + 1):
        for dc in range(2 * EROSION_PX + 1):
            out &= padded[dr : dr + h, dc : dc + w]
    return out


# --- The registered set, and the two things read from the bar -----------------

#: Names a caller might reach for that this module refuses, and the reason each is refused.
#: Checked before the roster, so the refusal explains *why* rather than reporting an unregistered
#: name — the difference between a caller who mistyped and a caller who wants the statistic §5
#: measured out of the set.
REFUSED: Final[dict[str, str]] = {
    name: (
        "the pooled mean signed error is deliberately not reported (§5, p123 §6): pooled over "
        "the corpus it cancels to 0.003–0.9 m against an MAE of 0.87–38.6 m and prints a null "
        "column for every arm, while ~half of a typical fill's error is a DC offset whose sign "
        "is arbitrary per tile. The registered column is mean per-instance |bias| — `bias_abs` "
        "here, magnitudes only, so nothing downstream can average the sign away."
    )
    for name in ("bias", "bias_signed", "mean_error", "mean_signed_error", "me")
}


def bar_path(cfg: BenchmarkConfig) -> Path:
    """Where this config's registered bar sits — ``freeze`` writes it, `score` reads R1 and R7."""
    return Path(cfg.paths.artifacts_dir) / prereg.BAR_NAME


def bar_for(cfg: BenchmarkConfig) -> prereg.Bar:
    """The registered bar this config's artifacts directory carries, checked against its §6."""
    return prereg.read_bar(bar_path(cfg), plan=cfg.paths.plan_path)


@functools.cache
def default_bar() -> prereg.Bar:
    """The committed bar at :class:`~benchmark.config.Paths`' default location.

    Cached, and that is a scoring decision rather than a micro-optimisation: the refusal fires at
    **load**, and an eval pass calls :func:`score` ~40,000 times per arm. Re-reading the artifact
    and re-scanning plan §6 per instance would buy nothing — the file cannot change under a run
    that has already resolved it — and would put a regex over the plan inside the hot loop.

    A caller that has a :class:`~benchmark.config.BenchmarkConfig` should pass :func:`bar_for`'s
    bar instead; this is the bare-call default, resolved relative to the working directory
    exactly as :data:`benchmark.prereg.PLAN_PATH` and :func:`dataset.validate.read_bar` are.
    """
    return bar_for(BenchmarkConfig())


def is_seam_break(seam_gap: float, *, bar: prereg.Bar | None = None) -> bool:
    """Is this gap a seam break? **R7's +0.15°, read from the bar and never typed.**

    Strictly above, per R7's own wording — *a seam gap above +0.15° is a seam break; |gap| ≤
    0.15° is not*. The asymmetry is the measurement's: the break side sits an order of magnitude
    above the intact side (+1.33° on the flat-fill control) and the intact side is a cluster
    straddling zero (−0.107° to −0.005°), so a bare ``> 0`` rule would fire on that scatter.
    """
    bar = default_bar() if bar is None else bar
    return float(seam_gap) > bar.seam_break_deg


# --- Scoring ------------------------------------------------------------------


def to_metres(standardized: np.ndarray, raw_i16: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Invert the per-tile transform with the **non-void** statistics — plan §2.1's inversion.

    The exact inverse of :func:`benchmark.extract.normalize`, and it reads its ``(mean, std)``
    from :func:`benchmark.extract.tile_statistics` rather than recomputing them: one code path,
    one land definition, and no way for the transform and its inverse to drift apart. The mask is
    ``valid = land ∧ ¬void``, so the statistics an arm's output is returned to metres by are the
    ones it was handed — a voided tile's true statistics are unknowable at inference, and
    standardizing by them would leak the answer into the input.

    The float32 round trip is not lossless; :func:`score`'s outside-the-void check carries a
    tolerance for exactly that, and nothing else.
    """
    mean, std = extract.tile_statistics(raw_i16, valid_mask)
    return np.asarray(standardized, dtype=np.float64) * max(std, extract.STD_FLOOR) + mean


def _finite_rmse(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else float("nan")


def _check_scored_in_metres(truth: np.ndarray, filled: np.ndarray, void: np.ndarray) -> None:
    """Refuse a fill that was not returned to metres — plan §2.1's rule, made structural.

    An arm writes inside the void and nowhere else, so outside it the fill *is* the tile it was
    handed. That makes the untouched pixels a free assertion about units: per-tile standardized
    values sit within a few σ of zero while the terrain they came from is hundreds of metres, so
    an un-inverted array misses by orders of magnitude on its first untouched pixel.

    The tolerance is a float32 round trip and nothing more (``rtol`` 1e-6 is ~3 mm at 2,700 m),
    because the failure this catches is not a near miss. A fill that genuinely differs outside
    the void is also refused, and should be: it is scoring pixels no mask says it may write.
    """
    outside = ~void
    if not outside.any():
        return
    t, f = truth[outside], filled[outside]
    if np.allclose(t, f, rtol=1e-6, atol=1e-3, equal_nan=False):
        return
    worst = int(np.nanargmax(np.abs(f - t))) if np.isfinite(f).any() else 0
    raise ValueError(
        "the fill does not match the truth outside the void, so it is not the tile it was "
        f"handed: worst untouched pixel is {f.ravel()[worst]:.3f} against {t.ravel()[worst]:.3f} "
        "metres. Plan §2.1: metrics are always computed **in metres**, after inverting the "
        "per-tile transform with the non-void statistics (`benchmark.metrics.to_metres`), so "
        "that no arm is ever scored in its own normalized space. If this array is still "
        "standardized, invert it; if an arm wrote outside its void, that is the bug."
    )


def _requested(bar: prereg.Bar, metrics: Iterable[str] | None) -> tuple[str, ...]:
    """The metric keys to compute, refusing any name R1's roster does not carry.

    **Registered, not graded** (#146). The roster is R1's single home and the seam gap is not
    named here — a second list of metric names is precisely what put a count in one place and a
    set in another, and the register cannot gate a build on a count that contradicts itself.
    """
    registered = bar.registered_metrics
    if metrics is None:
        return registered
    wanted = tuple(dict.fromkeys(metrics))
    for name in wanted:
        if name in REFUSED:
            raise ValueError(f"{name!r} is not computed here: {REFUSED[name]}")
        if name not in registered:
            raise ValueError(
                f"{name!r} is outside the registered metric set. R1 registers {registered} — "
                f"five graded headline columns, the derived seam gap and four appendix ones — "
                f"and {bar.path} is where that roster lives. A metric outside it is one the "
                f"24,000-instance grading never scored, so adding it here would put a number in "
                f"the results parquet that no pre-registration covers: amend ADR-0013 and plan "
                f"§6 first, then re-freeze."
            )
    return wanted


def score(
    truth: np.ndarray,
    filled: np.ndarray,
    void: np.ndarray,
    px: PixelSize,
    *,
    bar: prereg.Bar | None = None,
    metrics: Iterable[str] | None = None,
) -> dict[str, float]:
    """Score one instance: every registered metric, inside the void, in metres.

    ``truth`` and ``filled`` are whole tiles in **metres** — the fill is the tile it was handed
    with the void written over, so everything outside the void is untouched and is checked to be
    (:func:`_check_scored_in_metres`). ``void`` is the boolean mask of the pixels an arm was
    asked to predict, and ``px`` the ``(dy, dx)`` ground metres per pixel of *this* tile, passed
    in rather than looked up for :mod:`dataset.metrics`' reason: the pixel size is a property of
    where the tile is, not of the metric.

    Returns exactly the requested keys — R1's roster by default — and nothing else. The instance
    geometry a results row also carries (void area, equivalent diameter, the mask's RLE hash) is
    the caller's, because it is a property of the *mask* and this function scores a *fill*.

    Units: metres for ``rmse``, ``nmad``, ``bias_abs``, ``mae`` and ``p95``; degrees for
    ``slope_rmse``, ``slope_rmse_eroded`` and ``seam_gap``; degrees for ``aspect_error``, which
    is circular and slope-weighted; and 1/m for ``profile_curvature``.
    """
    bar = default_bar() if bar is None else bar
    wanted = _requested(bar, metrics)

    truth = np.asarray(truth, dtype=np.float64)
    filled = np.asarray(filled, dtype=np.float64)
    void = np.asarray(void, dtype=bool)
    if truth.shape != filled.shape or truth.shape != void.shape:
        raise ValueError(
            f"truth {truth.shape}, fill {filled.shape} and void {void.shape} are one tile"
        )
    if truth.ndim != 2:
        raise ValueError(f"score takes one tile at a time; got shape {truth.shape}")
    if not void.any():
        raise ValueError(
            "an empty void is not an instance: every metric here is computed only inside the "
            "void (§5), so there is nothing to score. The identity-mask case — fill(no void) == "
            "input — is asserted against the *fill*, not against a table of NaNs."
        )
    _check_scored_in_metres(truth, filled, void)

    err = (filled - truth)[void]
    out: dict[str, float] = {}

    # Elevation error, over the void and nothing else.
    if "rmse" in wanted:
        out["rmse"] = float(np.sqrt(np.mean(np.square(err))))
    if "mae" in wanted:
        out["mae"] = float(np.mean(np.abs(err)))
    if "p95" in wanted:
        out["p95"] = float(np.percentile(np.abs(err), P95_QUANTILE))
    if "nmad" in wanted:
        # Höhle & Höhle Eq. 2, spelled out rather than delegated: the median of the *deviations
        # from the median error*, scaled. Robust to the blunder tail §5 chose it for.
        out["nmad"] = float(NMAD_SCALE * np.median(np.abs(err - np.median(err))))
    if "bias_abs" in wanted:
        # This instance's mean signed error, as a magnitude. The sign stops here — see REFUSED.
        out["bias_abs"] = float(abs(np.mean(err)))

    # The derivatives. All three come off one ZT surface (§5), and the slope error is computed
    # once for both masks — which is what makes the seam gap free rather than a third pass.
    wants = set(wanted)
    slope_metrics = wants & {"slope_rmse", "slope_rmse_eroded", "seam_gap"}
    if slope_metrics or "aspect_error" in wants:
        s_truth = zt_slope_deg(truth, px)
    if slope_metrics:
        slope_err = zt_slope_deg(filled, px) - s_truth
        slope_full = _finite_rmse(slope_err[void])
        eroded = erode(void)
        # An eroded mask can be empty — a void thinner than 3 px has no pixel whose kernel is
        # entirely inside it. NaN rather than 0.0, because "no contamination control here" and
        # "a perfect interior" are different readings and only one of them is true.
        slope_eroded = _finite_rmse(slope_err[eroded]) if eroded.any() else float("nan")
        if "slope_rmse" in wants:
            out["slope_rmse"] = slope_full
        if "slope_rmse_eroded" in wants:
            out["slope_rmse_eroded"] = slope_eroded
        if "seam_gap" in wants:
            # Derived, by construction, from two rows already computed — never measured
            # separately, which is what makes the eroded number recoverable by subtraction and
            # is why R1 registers it outside the ten graded candidates (#146).
            out["seam_gap"] = slope_full - slope_eroded
    if "aspect_error" in wants:
        aspect_err = circular_error_rad(zt_aspect_rad(filled, px), zt_aspect_rad(truth, px))
        # Slope-weighted: a flat pixel has no aspect to get wrong, so weighting by sin(truth
        # slope) keeps the noise of an undefined direction out of the column p123 §3 measured as
        # the only orthogonal signal in the set.
        w, e = np.sin(np.radians(s_truth))[void], aspect_err[void]
        ok = np.isfinite(w) & np.isfinite(e) & (w > 0)
        out["aspect_error"] = (
            float(np.degrees(np.sum(w[ok] * e[ok]) / np.sum(w[ok]))) if ok.any() else float("nan")
        )
    if "profile_curvature" in wants:
        k_err = zt_profile_curvature(filled, px) - zt_profile_curvature(truth, px)
        out["profile_curvature"] = _finite_rmse(k_err[void])

    return {name: out[name] for name in wanted}


# --- The golden tile ----------------------------------------------------------

#: What the committed fixture is called, in ``paths.artifacts_dir``. Two files: the arrays, and
#: the recipe plus the expected scores that make them readable without numpy.
GOLDEN_ARRAYS: Final = "golden-tile.npz"
GOLDEN_SIDECAR: Final = "golden-tile.json"

#: The fixture, as arithmetic. Every array in :data:`GOLDEN_ARRAYS` is a pure function of this,
#: and :func:`read_golden` refuses a committed file that is no longer what its own recipe builds
#: — :mod:`benchmark.prereg`'s first link, on a fixture instead of on a threshold.
#:
#: **The truth surface** is a plane rising 4 m per row and 3 m per column off a 1,200 m base, on
#: 30 m pixels: integer metres everywhere, so int16 storage is exact and no quantisation noise
#: sits under the numbers below. Its ZT slope is ``atan(5/30)`` — 9.4623°, the 3-4-5 triangle —
#: at every interior pixel.
#:
#: **The fill is right in shape and wrong in level.** Inside the void it is the truth plus a
#: 4.0 m offset plus a ±1.5 m checkerboard ripple. That one construction lights up every column
#: for a different reason, which is what a gate fixture is for:
#:
#: * ``bias_abs`` sees the offset (4.0 m) and ``nmad`` sees the ripple (1.4826 × 1.5 m) — the
#:   guard column and the spread column, doing their separate jobs on one fill.
#: * **The ripple is invisible to slope.** A checkerboard flips sign between a pixel and *both*
#:   its neighbours along either axis, so it cancels exactly out of a central difference — which
#:   is what ZT's ``G`` and ``H`` are. The eroded-mask slope is therefore **zero**, and a
#:   contamination control that reads zero on a fill this wrong is p123 §4's finding as a
#:   fixture: eroding the ring does not remove a free ride, it hides a seam.
#: * **The full mask catches it**, because a ring pixel differences a filled neighbour against a
#:   true one and reads the 4 m step. The seam gap is that ring, and it clears R7's +0.15°.
#: * ``profile_curvature`` sees what slope cannot: the ripple survives ZT's *second* differences
#:   (``D`` and ``E``), at ``4a/dx²`` = 1/150 m⁻¹ on every interior pixel.
#:
#: The void is a 40×40 rectangle — even in both axes, so the checkerboard splits the 1,600 void
#: pixels exactly 800/800 and the elevation columns close in arithmetic — placed 60 px from the
#: top and 90 from the left, well inside §3.2's 8 px interiority margin. Its equivalent diameter
#: is 45.1 px, which is the **small** bucket.
GOLDEN_RECIPE: Final[dict[str, Any]] = {
    "shape": [dataset_config.TILE_PX, dataset_config.TILE_PX],
    "px": [30.0, 30.0],
    "base_m": 1200,
    "rise_row_m": 4,
    "rise_col_m": 3,
    "void": {"row0": 60, "row1": 100, "col0": 90, "col1": 130},
    "offset_m": 4.0,
    "ripple_m": 1.5,
}


@dataclass(frozen=True)
class GoldenTile:
    """The committed fixture: a tile, a void, a hand-checked fill, and what it must score."""

    truth: np.ndarray  # (224, 224) int16 metres, as a shard stores one
    filled: np.ndarray  # (224, 224) float32 metres — truth, with the void written over
    void: np.ndarray  # (224, 224) bool
    px: PixelSize
    #: The ten registered metrics and the value each must take, as committed.
    expected: dict[str, float]
    #: How each expected number was arrived at — closed form where there is one, and named as a
    #: pinned observation where there is not.
    derivations: dict[str, str]
    recipe: dict[str, Any]

    @property
    def valid_mask(self) -> np.ndarray:
        """``valid = land ∧ ¬void`` — what an extractor is handed for this tile (ADR-0012 §1)."""
        return land_mask(self.truth) & ~self.void


def build_golden(recipe: Mapping[str, Any] = GOLDEN_RECIPE) -> tuple[np.ndarray, ...]:
    """``(truth, filled, void)`` from the recipe — the fixture's arrays, as arithmetic."""
    rows, cols = recipe["shape"]
    row_i, col_j = np.mgrid[0:rows, 0:cols]
    truth = (
        recipe["base_m"] + recipe["rise_row_m"] * row_i + recipe["rise_col_m"] * col_j
    ).astype(np.int16)

    span = recipe["void"]
    void = np.zeros((rows, cols), dtype=bool)
    void[span["row0"] : span["row1"], span["col0"] : span["col1"]] = True

    ripple = recipe["ripple_m"] * np.where((row_i + col_j) % 2 == 0, 1.0, -1.0)
    filled = truth.astype(np.float32)
    filled[void] = (truth + recipe["offset_m"] + ripple).astype(np.float32)[void]
    return truth, filled, void


def golden_document(expected: Mapping[str, float], derivations: Mapping[str, str]) -> dict:
    """The sidecar as committed — the recipe, the scores, and how each score was arrived at."""
    return {
        "artifact": GOLDEN_SIDECAR,
        "arrays": GOLDEN_ARRAYS,
        "purpose": (
            "§8's golden tile: a committed tile with a known void and a hand-checked fill, plus "
            "the ten registered metrics it must score. It gates every arm's eval against a "
            "norm-inversion bug, which is the failure that silently flatters or sabotages an arm "
            "rather than raising. Asserted at both altitudes — at the `benchmark.metrics.score` "
            "seam (#155) and through `python -m benchmark.launch` as A1 (#158)."
        ),
        "recipe": dict(GOLDEN_RECIPE),
        "expected": dict(expected),
        "derivations": dict(derivations),
    }


def write_golden(
    directory: Path | str, expected: Mapping[str, float], derivations: Mapping[str, str]
) -> tuple[Path, Path]:
    """Write the fixture's two files, the sidecar pinning the arrays by md5.

    Not a `freeze` artifact and deliberately so: `freeze` writes what a *run* reads, and this is
    a fixture a **test** reads. Its expected scores are hand-derived (see
    :func:`read_golden`), so a writer that computed them by calling :func:`score` would pin the
    code against itself and the gate would pass on any wrong answer it was consistent about.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    truth, filled, void = build_golden()
    arrays = directory / GOLDEN_ARRAYS
    with arrays.open("wb") as handle:
        np.savez_compressed(handle, truth=truth, filled=filled, void=void)
    doc = golden_document(expected, derivations)
    doc["arrays_md5"] = hashlib.md5(arrays.read_bytes()).hexdigest()
    sidecar = directory / GOLDEN_SIDECAR
    sidecar.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    return arrays, sidecar


def read_golden(directory: Path | str) -> GoldenTile:
    """Load the fixture, refusing one whose halves have drifted.

    Two refusals, both of the shape :mod:`benchmark.prereg` uses on the registered bar:

    1. **The sidecar pins the arrays by md5.** The scores and the pixels they are scores *of* are
       two files, so nothing but a hash keeps them one fixture.
    2. **The arrays must still be what the recipe builds.** The recipe is what makes the fill
       hand-checkable — every closed form in :data:`GOLDEN_RECIPE` is arithmetic over its
       numbers — so a committed array that has drifted from it is a fixture whose derivations
       are about a different tile.

    The expected *scores* are not recomputed here. They are the assertion, and a loader that
    re-derived them from the code under test would be checking nothing at all.
    """
    directory = Path(directory)
    sidecar = directory / GOLDEN_SIDECAR
    arrays = directory / GOLDEN_ARRAYS
    doc = json.loads(sidecar.read_text())

    digest = hashlib.md5(arrays.read_bytes()).hexdigest()
    if digest != doc["arrays_md5"]:
        raise ValueError(
            f"{arrays} has md5 {digest} and {sidecar} pins {doc['arrays_md5']}: the golden tile's "
            "pixels and its expected scores are two files and have drifted apart. Re-write both "
            "together, with the derivations re-checked — a fixture whose scores are about a "
            "different tile gates nothing."
        )

    with np.load(arrays) as loaded:
        truth, filled, void = loaded["truth"], loaded["filled"], loaded["void"]
    rebuilt = build_golden(doc["recipe"])
    for name, committed, from_recipe in zip(
        ("truth", "filled", "void"), (truth, filled, void), rebuilt
    ):
        if not np.array_equal(committed, from_recipe):
            raise ValueError(
                f"{arrays}: the committed `{name}` is not what {sidecar}'s recipe builds. The "
                "recipe is what makes the fill hand-checked rather than merely committed, so the "
                "two cannot be moved apart — rebuild the arrays, or restate the recipe and "
                "re-derive every expected score against it."
            )
    return GoldenTile(
        truth=truth,
        filled=filled,
        void=void,
        px=tuple(doc["recipe"]["px"]),
        expected=doc["expected"],
        derivations=doc["derivations"],
        recipe=doc["recipe"],
    )
