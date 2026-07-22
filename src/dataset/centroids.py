"""The stratum centroids — what k-means produces over the frame (Phase D1, issue #41).

:mod:`dataset.frame` defines *what k-means sees*; this module defines *what it
produces*: the ``k = 20`` cluster centroids that partition the standardized kept frame
into strata, persisted as the artifact Phase F re-uses. Assignment to the nearest of
these centroids is how every later window gets a ``stratum_id`` — the sampler (#44) and
Phase F both label rows by :func:`assign`, never by re-fitting.

## Why k = 20, and why it is not a silhouette pick

k was measured, not inherited (issue #41; evidence in
``docs/research/d1-choose-k.md``). Sweeping k over the full frame found **no interior
optimum**: silhouette, seed-stability, and the §D2 floor cost all improve monotonically
as k *shrinks*, right down to k=8 — the six-dimensional descriptor cloud is a continuum,
not clumps, so a clustering-quality curve cannot pick k. The one measured boundary is a
**seed-stability cliff**: two independent fits agree at ARI 0.85 at k=20 but only 0.64 at
k=30. k=20 is the largest k whose partition still reproduces across seeds — maximum
terrain resolution before the strata become a coin-flip — and it halves the plan's
incumbent k=40 floor cost (20% of budget vs 40%), handing Neyman allocation (#42) its
dynamic range back.

## Determinism is met by persisting, not by seeding

The fit is **not** bitwise-reproducible. Measured at k=20: two fits with the *same seed
and same thread count* land centroids 4e-5 apart, and a single-threaded fit 2e-4 away —
float32 Lloyd accumulates thread-reduction order into the result. So a re-fit is not a
contract. The **committed centroids are the contract**: :func:`assign` is a pure
``argmin`` of Euclidean distance with a deterministic tie-break, so the *same window
lands in the same stratum on every machine for the life of the project* — exactly the
bar ``sampling.py`` holds, met the same way :mod:`dataset.frame` meets it for the scaler.

## Lloyd, not MiniBatch

Priced at k=20: MiniBatch is ~2-4% worse on inertia, only ~2x faster (a full-frame Lloyd
fit at ``n_init=10`` is ~45 s — trivially affordable), and its seed-stability *collapses*
to ARI 0.51, barely better than full Lloyd at k=40. It buys speed this frame does not
need at the cost of the reproducibility the whole map is built to protect. The shipped
fit is full-batch Lloyd, ``n_init = 25`` (#40 priced restarts at ~4 min for k=40 and
showed they are the lever against the multi-modality — the shipped fit is not the stingy
one).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from . import catalog, frame

# --- The locked choices (issue #41) -------------------------------------------

#: Number of strata. Locked by measurement (see module docstring / d1-choose-k.md):
#: the largest k inside the seed-stable band, not a silhouette optimum (there is none).
K: Final = 20

#: Restarts for the shipped fit. Generous on purpose — restarts are the lever against
#: the k=40-era multi-modality #40 found, and a full-frame fit is seconds.
N_INIT: Final = 25

#: Lloyd (full batch), not MiniBatch — MiniBatch's seed-stability collapses at this k.
ALGORITHM: Final = "lloyd"

#: The seed handed to the fit. Recorded for provenance, **not** relied on for
#: reproducibility — the fit is not bitwise-stable across threads (see docstring). The
#: committed ``centers`` are the contract; :func:`assign` is what reproduces.
RANDOM_STATE: Final = 0

#: Where the fitted centroids live. Committed beside the scaler — ``data/`` is
#: gitignored and Phase F's comparability rests on re-loading this exact file.
CENTROIDS_PATH: Final = Path("artifacts/phase_d/centroids.json")
#: Bumped when the artifact's field set changes, so an old file fails to load loudly.
CENTROIDS_SCHEMA_VERSION: Final = 1


# --- The artifact -------------------------------------------------------------


@dataclass(frozen=True)
class Centroids:
    """The ``(k, 6)`` centroids in **standardized** space, plus their provenance.

    Standardized space — the same z-scored, log-transformed space the scaler defines —
    so a centroid coordinate is still readable as "1.4 sd above mean log slope", and
    :func:`assign` compares like with like. The fingerprint is the frame these were fit
    over; it is checked on load exactly as the scaler's is, because a set of centroids
    looks equally plausible whatever frame produced it.
    """

    centers: np.ndarray  # (k, 6), standardized
    fingerprint: dict[str, Any]
    provenance: dict[str, Any]

    def __post_init__(self) -> None:
        if self.centers.ndim != 2 or self.centers.shape[1] != len(frame.METRICS):
            raise ValueError(
                f"centroids must be (k, {len(frame.METRICS)}); got {self.centers.shape}"
            )
        if not np.isfinite(self.centers).all():
            raise ValueError("non-finite centroid coordinate")

    @property
    def k(self) -> int:
        return int(self.centers.shape[0])

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CENTROIDS_SCHEMA_VERSION,
            "k": self.k,
            "space": "standardized (z-scored, ADR-0004) — the scaler's space",
            "metrics": list(frame.METRICS),
            "provenance": dict(self.provenance),
            "frame": dict(self.fingerprint),
            # One row per stratum; column order is frame.METRICS, checked on load.
            "centers": [[float(v) for v in row] for row in self.centers],
        }

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "Centroids":
        if obj.get("schema_version") != CENTROIDS_SCHEMA_VERSION:
            raise ValueError(
                f"centroids schema_version {obj.get('schema_version')!r} != "
                f"{CENTROIDS_SCHEMA_VERSION} — refit rather than guess the field set"
            )
        if tuple(obj.get("metrics", ())) != frame.METRICS:
            raise ValueError(
                f"centroid metric order {tuple(obj.get('metrics', ()))} != "
                f"{frame.METRICS} — the persisted vector no longer lines up with this code"
            )
        return cls(
            centers=np.asarray(obj["centers"], dtype=np.float64),
            fingerprint=dict(obj.get("frame", {})),
            provenance=dict(obj.get("provenance", {})),
        )


# --- Fit ----------------------------------------------------------------------


def fit_centroids(
    Z: np.ndarray,
    *,
    fingerprint: dict[str, Any],
    k: int = K,
    n_init: int = N_INIT,
    random_state: int = RANDOM_STATE,
) -> Centroids:
    """Fit ``k`` centroids over **every row** of the standardized frame ``Z``.

    ``Z`` is expected already standardized by the committed scaler — this function does
    not re-standardize, so the centroids land in the same space :func:`assign` will
    compare against. scikit-learn is imported lazily: fitting needs it, but *loading and
    assigning* the committed artifact need only NumPy, so the reproducible read path
    carries no sklearn dependency.
    """
    from sklearn import __version__ as sklearn_version
    from sklearn.cluster import KMeans

    Z = np.ascontiguousarray(Z, dtype=np.float32)
    if Z.ndim != 2 or Z.shape[1] != len(frame.METRICS):
        raise ValueError(f"expected an (n, {len(frame.METRICS)}) matrix; got {Z.shape}")

    km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state, algorithm=ALGORITHM)
    km.fit(Z)
    return Centroids(
        centers=km.cluster_centers_.astype(np.float64),
        fingerprint=dict(fingerprint),
        provenance={
            "implementation": "scikit-learn KMeans",
            "sklearn_version": sklearn_version,
            "algorithm": ALGORITHM,
            "n_init": n_init,
            "random_state": random_state,
            "inertia": float(km.inertia_),
            "n_iter": int(km.n_iter_),
            "fit_rows": int(Z.shape[0]),
            "note": "the fit is not bitwise-reproducible across threads; these "
            "committed centers are the contract, reproduced by assign(), not by refit",
        },
    )


def assign(Z: np.ndarray, centroids: Centroids) -> np.ndarray:
    """Label each row of standardized ``Z`` with its nearest centroid — the stratum id.

    A pure ``argmin`` of squared Euclidean distance, ties broken by lowest stratum index
    (NumPy ``argmin`` returns the first minimum). No RNG, no sklearn: this is the
    deterministic step that makes selection reproduce from the committed artifact, so it
    is deliberately a few lines of NumPy rather than ``KMeans.predict``.

    Chunked over rows: the full ``(3.05M, 20)`` distance matrix is ~490 MB in float32,
    which is fine, but a 200-500k Phase-F frame against a larger future k need not be
    materialized whole.
    """
    Z = np.asarray(Z, dtype=np.float32)
    if Z.ndim != 2 or Z.shape[1] != len(frame.METRICS):
        raise ValueError(f"expected an (n, {len(frame.METRICS)}) matrix; got {Z.shape}")
    C = centroids.centers.astype(np.float32)
    csq = np.einsum("kj,kj->k", C, C)  # ‖c‖² per centroid; the ‖z‖² term is common to
    out = np.empty(Z.shape[0], dtype=np.int32)  # every centroid so it drops out of argmin
    step = 200_000
    for i in range(0, Z.shape[0], step):
        block = Z[i : i + step]
        # argmin over ‖z-c‖² = ‖z‖² - 2 z·c + ‖c‖²; ‖z‖² is constant across c, omit it.
        d = csq[None, :] - 2.0 * (block @ C.T)
        out[i : i + block.shape[0]] = np.argmin(d, axis=1)
    return out


# --- Persist / load -----------------------------------------------------------


def write_centroids(centroids: Centroids, path: Path | str = CENTROIDS_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(centroids.as_dict(), indent=2) + "\n")
    return path


def read_centroids(
    path: Path | str = CENTROIDS_PATH,
    *,
    fingerprint: dict[str, Any] | None = None,
) -> Centroids:
    """Load the centroids, refusing a frame that does not match the one they were fit over.

    Same guard as :func:`frame.read_scaler`: pass a ``fingerprint`` whenever a catalog is
    in hand, and a moved frame raises instead of silently producing a plausible, wrong
    partition. The scaler and the centroids share one fingerprint, so if both load
    against the same frame they are guaranteed to describe the same space.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"no centroids at {path} — fit them with `python -m dataset.centroids`"
        )
    centroids = Centroids.from_dict(json.loads(path.read_text()))
    if fingerprint is not None:
        drift = {
            key: (centroids.fingerprint.get(key), val)
            for key, val in fingerprint.items()
            if centroids.fingerprint.get(key) != val
        }
        if drift:
            detail = "; ".join(f"{k}: artifact {a!r} != frame {b!r}" for k, (a, b) in drift.items())
            raise ValueError(
                f"centroids at {path} were fit over a different frame — {detail}. "
                "Re-fit, and expect every stratum boundary to move."
            )
    return centroids


# --- Live driver --------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """``python -m dataset.centroids`` — fit the k=20 centroids and write the artifact.

    Reads the kept frame, standardizes it with the *committed* scaler (never a fresh
    one — that is what keeps Phase F comparable), fits, and persists. Minutes, not hours,
    and it touches no COG.
    """
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m dataset.centroids",
        description="Fit and persist the Phase-D stratum centroids over the kept frame "
        f"(k={K}, issue #41).",
    )
    p.add_argument("--catalog-dir", type=Path, default=Path("data/catalog_gated"))
    p.add_argument("--out", type=Path, default=CENTROIDS_PATH)
    p.add_argument("--k", type=int, default=K)
    p.add_argument("--n-init", type=int, default=N_INIT)
    args = p.parse_args(argv)

    fr = frame.kept_frame(args.catalog_dir)
    scaler = frame.read_scaler(fingerprint=fr.fingerprint)
    Z = fr.standardized(scaler)

    centroids = fit_centroids(
        Z, fingerprint=fr.fingerprint, k=args.k, n_init=args.n_init
    )
    path = write_centroids(centroids, args.out)

    counts = np.bincount(assign(Z, centroids), minlength=centroids.k)
    print(f"Fit k={centroids.k} over {fr.n:,} kept rows, n_init={args.n_init}.")
    print(f"  inertia {centroids.provenance['inertia']:,.0f}  "
          f"({centroids.provenance['n_iter']} iters, "
          f"sklearn {centroids.provenance['sklearn_version']})")
    print(f"  occupancy N_h: min {counts.min():,}  max {counts.max():,}  "
          f"median {int(np.median(counts)):,}  ({counts.max() / counts.min():.1f}x spread)")
    print(f"  centroids: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
