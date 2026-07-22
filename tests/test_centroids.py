"""The stratum centroids — the artifact contract and the deterministic assignment (#41).

Zero-network, zero-catalog, zero-sklearn: every case is a tiny hand-built set of
centroids in standardized space. The k-sweep that *chose* k=20 over 3.05 M rows lives in
[`docs/research/d1-choose-k.md`](../docs/research/d1-choose-k.md); this suite pins the
two things that have to hold for the committed centroids to be trustworthy years later.

1. **Assignment is a pure, deterministic ``argmin``** — the same window lands in the
   same stratum on every machine, ties broken by lowest stratum index. This is the
   contract the map's determinism bar is met by, since the *fit* is not bitwise-stable.
2. **A moved frame is refused, not absorbed** — a centroid set re-used across a re-gated
   or re-scored catalog would produce a plausible, wrong partition; it has to raise.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from dataset import centroids, frame

D = len(frame.METRICS)


def _centroids(centers, **fp) -> centroids.Centroids:
    fingerprint = {"n_kept": 3, "gate_thresholds": {"water_drop_max": 0.30},
                   "catalog_code_version": "abc123"}
    fingerprint.update(fp)
    return centroids.Centroids(
        centers=np.asarray(centers, dtype=np.float64),
        fingerprint=fingerprint,
        provenance={"implementation": "test"},
    )


# --- 1. Assignment is a deterministic argmin ----------------------------------


def test_assign_picks_the_nearest_centroid():
    # Two centroids far apart on the first axis; rows sit clearly nearer one or the other.
    c = _centroids([[-5.0] + [0.0] * (D - 1), [5.0] + [0.0] * (D - 1)])
    Z = np.array([[-4.0] + [0.0] * (D - 1), [4.0] + [0.0] * (D - 1)], dtype=np.float32)
    assert centroids.assign(Z, c).tolist() == [0, 1]


def test_assign_breaks_ties_to_the_lowest_index():
    # A row equidistant from two centroids must land in the lower-indexed stratum, on
    # every machine — argmin returns the first minimum, and the sampler relies on it.
    c = _centroids([[-1.0] + [0.0] * (D - 1), [1.0] + [0.0] * (D - 1)])
    Z = np.zeros((1, D), dtype=np.float32)
    assert centroids.assign(Z, c).tolist() == [0]


def test_assign_is_identical_across_chunk_boundaries():
    # The chunked loop must give the same labels as a single pass — a row's stratum
    # cannot depend on where the 200k-row chunk boundary happens to fall.
    rng = np.random.default_rng(0)
    c = _centroids(rng.standard_normal((8, D)))
    Z = rng.standard_normal((1000, D)).astype(np.float32)
    labels = centroids.assign(Z, c)
    # Re-assign one row at a time; must match the batched result exactly.
    one_by_one = np.array([centroids.assign(Z[i : i + 1], c)[0] for i in range(Z.shape[0])])
    assert np.array_equal(labels, one_by_one)


def test_assign_is_repeatable():
    rng = np.random.default_rng(1)
    c = _centroids(rng.standard_normal((12, D)))
    Z = rng.standard_normal((5000, D)).astype(np.float32)
    assert np.array_equal(centroids.assign(Z, c), centroids.assign(Z, c))


def test_assign_rejects_wrong_width():
    c = _centroids(np.zeros((3, D)))
    with pytest.raises(ValueError, match=f"n, {D}"):
        centroids.assign(np.zeros((4, D + 1), dtype=np.float32), c)


# --- 2. The artifact round-trips and refuses a moved frame --------------------


def test_round_trip_through_json(tmp_path):
    c = _centroids(np.arange(4 * D, dtype=np.float64).reshape(4, D))
    path = centroids.write_centroids(c, tmp_path / "centroids.json")
    back = centroids.read_centroids(path)
    assert back.k == 4
    assert np.allclose(back.centers, c.centers)


def test_read_refuses_a_moved_frame(tmp_path):
    c = _centroids(np.zeros((3, D)), n_kept=3_052_735)
    path = centroids.write_centroids(c, tmp_path / "centroids.json")
    with pytest.raises(ValueError, match="different frame"):
        centroids.read_centroids(path, fingerprint={"n_kept": 3_000_000})


def test_read_accepts_a_matching_frame(tmp_path):
    c = _centroids(np.zeros((3, D)), n_kept=42)
    path = centroids.write_centroids(c, tmp_path / "centroids.json")
    back = centroids.read_centroids(path, fingerprint={"n_kept": 42})
    assert back.k == 3


def test_from_dict_rejects_reordered_metrics(tmp_path):
    c = _centroids(np.zeros((3, D)))
    obj = c.as_dict()
    obj["metrics"] = list(reversed(obj["metrics"]))
    with pytest.raises(ValueError, match="metric order"):
        centroids.Centroids.from_dict(obj)


def test_from_dict_rejects_a_bumped_schema(tmp_path):
    obj = _centroids(np.zeros((3, D))).as_dict()
    obj["schema_version"] = 999
    with pytest.raises(ValueError, match="schema_version"):
        centroids.Centroids.from_dict(obj)


def test_rejects_non_finite_centroid():
    with pytest.raises(ValueError, match="non-finite"):
        _centroids([[np.inf] + [0.0] * (D - 1)])


def test_rejects_wrong_width_centroids():
    with pytest.raises(ValueError, match=f"k, {D}"):
        _centroids(np.zeros((3, D + 2)))


# --- 3. The committed artifact matches the locked choices ---------------------


def test_shipped_artifact_is_k20_lloyd_pinned():
    """The committed centroids are the k=20, Lloyd, pinned-sklearn fit #41 locked.

    Guards against a stray re-fit at a different k or with MiniBatch silently replacing
    the reviewed artifact.
    """
    obj = json.loads(centroids.CENTROIDS_PATH.read_text())
    assert obj["k"] == centroids.K == 20
    assert obj["provenance"]["algorithm"] == "lloyd"
    assert obj["provenance"]["n_init"] == centroids.N_INIT
    assert obj["provenance"]["sklearn_version"].startswith("1.9")
    assert len(obj["centers"]) == 20
