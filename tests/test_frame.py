"""The stratification frame — what k-means actually sees (Phase D1, issue #40).

Zero-network and zero-catalog: every case is a hand-built frame of a few rows. The
3.05 M-row measurements that decided the frame live in
[`docs/research/d1-stratification-frame.md`](../docs/research/d1-stratification-frame.md);
this suite pins the *contract* that lets a persisted scaler be trusted years later.

The load-bearing checks are the ones a wrong frame would silently pass:

1. **The transform set is ADR-0004's** — ``ln(x + 1)`` on ``slope_mean`` and
   ``local_relief``, nothing on the other four, and the column order is the ADR's.
2. **The scaler is fitted over every kept row**, tags included — the frame decision
   #40 settled, which no keyword may quietly re-open.
3. **A moved frame is refused, not absorbed.** Re-using a scaler across a re-gated or
   re-scored catalog produces a complete, plausible, wrong corpus; it has to raise.
4. **The tags ride alongside the matrix** rather than filtering it, because #42 needs
   them row-aligned to the fit.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from dataset import frame

# --- Frame construction -------------------------------------------------------


def _cols(**overrides) -> dict[str, np.ndarray]:
    """A column frame over the six metrics; unnamed metrics default to a flat ramp."""
    n = len(next(iter(overrides.values())))
    base = {m: np.linspace(1.0, 10.0, n) for m in frame.METRICS}
    base.update({k: np.asarray(v, dtype=np.float64) for k, v in overrides.items()})
    return base


def _fingerprint(**overrides):
    fp = {"n_kept": 3, "gate_thresholds": {"water_drop_max": 0.30}, "catalog_code_version": "abc123"}
    fp.update(overrides)
    return fp


# --- 1. The transform set is ADR-0004's ---------------------------------------


def test_column_order_is_the_adr_order():
    assert frame.METRICS == (
        "slope_mean",
        "local_relief",
        "texture",
        "elevation_relief_ratio",
        "mean_elevation",
        "geomorphon_entropy",
    )


def test_only_slope_and_relief_are_logged():
    assert frame.LOG_METRICS == ("slope_mean", "local_relief")


@pytest.mark.parametrize("name", frame.LOG_METRICS)
def test_logged_metrics_get_ln_x_plus_1(name):
    raw = np.array([0.0, 1.0, 99.0])
    X = frame.metric_matrix(_cols(**{name: raw}))
    got = X[:, frame.METRICS.index(name)]
    assert np.allclose(got, np.log(raw + 1.0))
    # the ``+1`` is what makes an exact-zero window finite rather than -inf
    assert got[0] == 0.0


@pytest.mark.parametrize("name", [m for m in frame.METRICS if m not in frame.LOG_METRICS])
def test_unlogged_metrics_pass_through(name):
    raw = np.array([-28.0, 0.0, 4.5])  # mean_elevation genuinely reaches -28 m
    X = frame.metric_matrix(_cols(**{name: raw}))
    assert np.allclose(X[:, frame.METRICS.index(name)], raw)


def test_matrix_is_float64_even_from_float32_columns():
    cols = {m: np.linspace(1, 5, 4, dtype=np.float32) for m in frame.METRICS}
    assert frame.metric_matrix(cols).dtype == np.float64


# --- 2. The scaler is fitted over every row -----------------------------------


def test_scaler_is_a_plain_z_score_over_all_rows():
    X = np.arange(24, dtype=np.float64).reshape(4, 6)
    scaler = frame.fit_scaler(X, fingerprint=_fingerprint())
    assert np.allclose(scaler.mean, X.mean(axis=0))
    assert np.allclose(scaler.std, X.std(axis=0))
    Z = scaler.standardize(X)
    assert np.allclose(Z.mean(axis=0), 0.0)
    assert np.allclose(Z.std(axis=0), 1.0)


def test_standardize_does_not_whiten():
    """ADR-0004: no PCA in the distance — correlation between dimensions survives."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    X = np.column_stack([a, a * 2 + rng.normal(scale=0.01, size=200), *rng.normal(size=(4, 200))])
    Z = frame.fit_scaler(X, fingerprint=_fingerprint()).standardize(X)
    assert abs(np.corrcoef(Z[:, 0], Z[:, 1])[0, 1]) > 0.99


def test_fit_refuses_a_frame_with_a_nan():
    X = np.ones((3, 6))
    X[1, 2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        frame.fit_scaler(X, fingerprint=_fingerprint())


def test_fit_refuses_an_empty_frame():
    with pytest.raises(ValueError, match="empty frame"):
        frame.fit_scaler(np.empty((0, 6)), fingerprint=_fingerprint())


def test_scaler_refuses_a_zero_variance_metric():
    X = np.ones((4, 6))
    with pytest.raises(ValueError, match="zero-variance"):
        frame.fit_scaler(X, fingerprint=_fingerprint())


def test_standardize_refuses_a_wrong_width_matrix():
    scaler = frame.fit_scaler(np.arange(24, dtype=float).reshape(4, 6), fingerprint=_fingerprint())
    with pytest.raises(ValueError, match=r"\(n, 6\)"):
        scaler.standardize(np.arange(12, dtype=float).reshape(4, 3))


# --- 3. A moved frame is refused ----------------------------------------------


def _written(tmp_path, **fp):
    scaler = frame.fit_scaler(
        np.arange(24, dtype=np.float64).reshape(4, 6), fingerprint=_fingerprint(**fp)
    )
    return frame.write_scaler(scaler, tmp_path / "scaler.json"), scaler


def test_artifact_round_trips(tmp_path):
    path, scaler = _written(tmp_path)
    back = frame.read_scaler(path)
    assert np.allclose(back.mean, scaler.mean)
    assert np.allclose(back.std, scaler.std)
    assert back.fingerprint == scaler.fingerprint


def test_artifact_is_keyed_by_metric_name_not_position(tmp_path):
    path, _ = _written(tmp_path)
    obj = json.loads(path.read_text())
    assert set(obj["params"]) == set(frame.METRICS)


@pytest.mark.parametrize(
    "moved",
    [
        {"n_kept": 3_052_734},
        {"catalog_code_version": "deadbee"},
        {"gate_thresholds": {"water_drop_max": 0.25}},
    ],
)
def test_read_refuses_a_frame_that_moved(tmp_path, moved):
    path, _ = _written(tmp_path)
    with pytest.raises(ValueError, match="different frame"):
        frame.read_scaler(path, fingerprint=_fingerprint(**moved))


def test_read_accepts_the_frame_it_was_fitted_over(tmp_path):
    path, _ = _written(tmp_path)
    assert frame.read_scaler(path, fingerprint=_fingerprint()) is not None


def test_read_without_a_fingerprint_does_not_check(tmp_path):
    """Loading for inspection is allowed; it is *selection* that must be gated."""
    path, _ = _written(tmp_path)
    assert frame.read_scaler(path) is not None


def test_missing_artifact_names_the_command_that_makes_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m dataset.frame"):
        frame.read_scaler(tmp_path / "nope.json")


def test_stale_schema_version_is_refused(tmp_path):
    path, _ = _written(tmp_path)
    obj = json.loads(path.read_text())
    obj["schema_version"] = frame.SCALER_SCHEMA_VERSION + 1
    path.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="schema_version"):
        frame.read_scaler(path)


def test_reordered_metrics_are_refused(tmp_path):
    path, _ = _written(tmp_path)
    obj = json.loads(path.read_text())
    obj["metrics"] = list(reversed(obj["metrics"]))
    path.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="metric order"):
        frame.read_scaler(path)


# --- 4. The tags ride alongside, they do not filter ---------------------------


def test_frame_carries_the_tags_row_aligned():
    X = np.arange(18, dtype=np.float64).reshape(3, 6)
    f = frame.Frame(
        X=X,
        window_id=np.array(["a", "b", "c"]),
        water_affected=np.array([False, True, False]),
        coastline=np.array([False, False, True]),
        fingerprint=_fingerprint(),
    )
    assert f.n == 3  # the tagged rows are in the fit frame, not filtered out of it
    assert f.water_affected.sum() == 1
    assert f.coastline.sum() == 1
