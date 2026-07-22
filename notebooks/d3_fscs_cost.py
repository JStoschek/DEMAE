"""Price FSCS at catalog scale, and its fallbacks — the measurement behind #43.

Throwaway measurement harness, not pipeline code. Plan §D3 selects *within* a stratum
by **sub-k-means with k = n_h**: cluster the stratum into as many sub-clusters as it
has quota, take the window nearest each sub-centroid, geo-cell round-robin as the
tie-breaker. The mechanics are verified; the ticket is the **cost**, which the plan
never prices.

The worry the ticket states is that a large stratum holds 10^5-10^6 rows with n_h in
the thousands, and that k-means with k in the thousands over ~10^6 points is a
different animal from the k~40 global fit. So this measures cost as a function of
``(N_h, n_h)`` over the plausible ranges and extrapolates — the allocation ticket
(#42) supplies the real points later.

Four things get measured, in the order they decide the answer:

- **cost**: wall clock and peak RSS as ``(N_h, n_h)`` grows, over *real* strata rather
  than random slices of the frame — a stratum is a compact cluster, and compactness
  drives Lloyd's iteration count, which is the term that actually varies;
- **determinism**: two runs at the same seed must select the same window ids. The
  map's contract binds here identically to everywhere else;
- **spread**: what the fallbacks cost in *accuracy*, measured as descriptor-space
  coverage of the stratum rather than asserted;
- **projection**: the per-stratum figures summed over a real k=40 occupancy
  distribution at both the 50k debug budget and the 500k scale tier.

Cover sheet: [`docs/research/d3-fscs-cost.md`](../docs/research/d3-fscs-cost.md).
Regenerate the committed JSON in `docs/research/` by re-running the subcommands below and
copying `notebooks/out/d3_*.json` across; this reads the Parquet the global pass already
wrote and touches no COG, so it needs only the gated catalog and the committed scaler.

Usage::

    uv run python notebooks/d3_fscs_cost.py grid          # cost -> JSON
    uv run python notebooks/d3_fscs_cost.py determinism
    uv run python notebooks/d3_fscs_cost.py spread
    uv run python notebooks/d3_fscs_cost.py project
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset import frame as frame_mod  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
CACHE = OUT / "d3_strata_cache.npz"

#: Stratum sizes swept. The top of the range is deliberately past what k=40 produces —
#: #41's sweep tops out at 291k at k=20 — so the extrapolation is anchored on measured
#: points rather than on the shape of the curve below them.
N_H_VALUES = (30_000, 100_000, 300_000, 1_000_000)

#: Quotas swept. 500 is §D2's floor at the 50k budget; 5,000 is that floor at the 500k
#: scale tier; 25,000 is a scale-tier stratum that also wins a large Neyman share.
N_VALUES = (500, 1_000, 2_500, 5_000, 25_000)

#: The global k this harness fits to get *real* strata to draw from. Not a claim about
#: k — that is #41. Only compactness matters here, and it does not turn on the count.
GLOBAL_K = 40

#: Skip any grid cell projected past this. A cell that costs 20 minutes has already
#: answered the ticket; paying for it adds a digit, not a decision.
CELL_BUDGET_S = 900.0

#: Largest quota-to-stratum ratio the grid bothers to time. Above this the sub-k-means
#: is asking for a cluster per handful of rows, which is a degenerate fit (k-means++
#: init alone is O(N_h * n_h) *sequential*) and — more to the point — cannot occur:
#: even the 500k tier's largest stratum takes roughly N_h/6. Timing n_h = 0.8 * N_h
#: measures a case the sampler will never be asked for, at the price of the cases it
#: will.
MAX_QUOTA_FRACTION = 0.25

SEED = 0


# --- Frame and strata ---------------------------------------------------------


def load_frame() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardized kept frame in float32, plus window ids and geo-cell ids.

    Standardized with the **committed** scaler, not a re-fitted one: that artifact is
    what makes Phase F comparable, and re-fitting here would silently price a
    different space. ``geocell`` is read straight off the catalog because
    :class:`dataset.frame.Frame` does not carry it — §D3's tie-breaker needs it, and
    re-deriving it from ``window_id`` would be a second definition of the same thing.
    """
    import pyarrow.parquet as pq

    from dataset import catalog

    fr = frame_mod.kept_frame()
    scaler = frame_mod.read_scaler(fingerprint=fr.fingerprint)
    Z = np.ascontiguousarray(fr.standardized(scaler).astype(np.float32))

    table = pq.read_table(
        Path("data/catalog_gated") / catalog.PARTS_DIR, columns=["keep", "geocell"]
    )
    keep = table.column("keep").to_numpy(zero_copy_only=False).astype(bool)
    geocell = table.column("geocell").to_numpy(zero_copy_only=False)[np.flatnonzero(keep)]
    return Z, fr.window_id, np.asarray(geocell)


def global_labels(Z: np.ndarray, k: int = GLOBAL_K) -> np.ndarray:
    """Labels of one cheap global fit, cached — the source of *real* strata.

    ``n_init=1``: this fit is not a claim about where the strata are, only a source of
    realistically compact groups to time against. #40 measured that reseeding moves
    a k=40 partition more than either tag does, so a restarted fit would be a
    different-but-equally-arbitrary partition at 10x the cost.
    """
    if CACHE.exists():
        cached = np.load(CACHE)
        if int(cached["k"]) == k and int(cached["n"]) == Z.shape[0]:
            return cached["labels"]

    from sklearn.cluster import KMeans

    t0 = time.perf_counter()
    labels = KMeans(n_clusters=k, n_init=1, random_state=SEED, algorithm="lloyd").fit_predict(Z)
    print(f"  global k={k} fit: {time.perf_counter() - t0:.1f}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, labels=labels, k=k, n=Z.shape[0])
    return labels


def synthetic_stratum(Z: np.ndarray, labels: np.ndarray, n_h: int, seed: int = SEED) -> np.ndarray:
    """Row indices of a stratum of exactly ``n_h`` rows, as compact as a real one.

    Built from the largest real stratum: sub-sampled down when ``n_h`` is smaller,
    and topped up with the next-nearest strata's rows when ``n_h`` exceeds anything
    k=40 produces. Timing against a random slice of the whole frame would flatter
    Lloyd — a frame-wide sample is more spread out than a stratum, and converges in
    fewer iterations than the case that actually has to be paid for.
    """
    rng = np.random.default_rng(seed)
    sizes = np.bincount(labels)
    order = np.argsort(sizes)[::-1]
    pool: list[np.ndarray] = []
    have = 0
    for lab in order:
        rows = np.flatnonzero(labels == lab)
        pool.append(rows)
        have += rows.size
        if have >= n_h:
            break
    idx = np.concatenate(pool)
    if idx.size > n_h:
        idx = rng.choice(idx, n_h, replace=False)
    return np.sort(idx)


# --- The four selection rules -------------------------------------------------


def _hash_order(window_id: np.ndarray, key: str) -> np.ndarray:
    """Row indices in keyed-BLAKE2b order of their window id — a stable shuffle.

    The house rule (``sampling.py``, and #40's scaler): NumPy guarantees bit-generator
    *streams* across releases but not the distributions layered on them, so anything
    that has to reproduce for the life of the project is ordered by hash rather than by
    ``Generator``. Keyed on ``window_id`` rather than on row position, so the order does
    not move when the frame is filtered differently upstream.
    """
    keys = np.array(
        [
            int.from_bytes(
                hashlib.blake2b(f"{key}:{w}".encode(), digest_size=8).digest(), "big"
            )
            for w in window_id
        ],
        dtype=np.uint64,
    )
    return np.argsort(keys, kind="stable")


def _nearest_to_centroids(S: np.ndarray, centers: np.ndarray, geocell: np.ndarray) -> np.ndarray:
    """One representative per sub-centroid: nearest window, geo-cell round-robin ties.

    §D3's rule. The tie-break is not decoration — without it, two sub-centroids in the
    same dense pocket can both resolve to the same over-represented geo-cell, and the
    corpus quietly re-concentrates inside a stratum that looks well spread in
    descriptor space.
    """
    from scipy.spatial import cKDTree

    # A single tree query for the C nearest rows to every sub-centroid at once. C > 1
    # because two sub-centroids in the same dense pocket resolve to the same row, and
    # a rule that just drops the duplicate silently returns fewer than n_h windows.
    C = 8
    tree = cKDTree(S)
    _, cand = tree.query(centers, k=min(C, S.shape[0]), workers=-1)
    cand = np.atleast_2d(cand)

    taken: set[int] = set()
    cell_use: dict[str, int] = {}
    out: list[int] = []
    for r in range(cand.shape[0]):
        # Nearest first, but a row whose geo-cell is already over-used yields to the
        # next-nearest from a less-used cell: §D3's round-robin tie-break. Without it
        # a stratum re-concentrates geographically while looking well spread in
        # descriptor space.
        best: int | None = None
        best_key: tuple[int, int] | None = None
        for rank, row in enumerate(cand[r]):
            row = int(row)
            if row in taken:
                continue
            key = (cell_use.get(str(geocell[row]), 0), rank)
            if best_key is None or key < best_key:
                best, best_key = row, key
        if best is None:
            continue
        taken.add(best)
        cell_use[str(geocell[best])] = cell_use.get(str(geocell[best]), 0) + 1
        out.append(best)
    return np.array(sorted(out), dtype=np.int64)


def select_lloyd(
    S: np.ndarray, n_h: int, geocell: np.ndarray, window_id: np.ndarray, seed: int = SEED
) -> np.ndarray:
    """§D3 as specified: full Lloyd sub-k-means at k = n_h."""
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=n_h, n_init=1, random_state=seed, algorithm="lloyd").fit(S)
    return _nearest_to_centroids(S, km.cluster_centers_, geocell)


def select_minibatch(
    S: np.ndarray, n_h: int, geocell: np.ndarray, window_id: np.ndarray, seed: int = SEED
) -> np.ndarray:
    """Fallback 1: MiniBatch sub-k-means — same rule, approximate centroids."""
    from sklearn.cluster import MiniBatchKMeans

    km = MiniBatchKMeans(
        n_clusters=n_h, n_init=1, random_state=seed, batch_size=4096, max_iter=100
    ).fit(S)
    return _nearest_to_centroids(S, km.cluster_centers_, geocell)


#: Rows a stratum is cut to before Lloyd, under the sub-sample fallback. 20x quota is
#: enough that every sub-centroid still has a dense neighbourhood to land in.
SUBSAMPLE_MULT = 20


def select_subsampled(
    S: np.ndarray, n_h: int, geocell: np.ndarray, window_id: np.ndarray, seed: int = SEED
) -> np.ndarray:
    """Fallback 2: cut the stratum to ``20 * n_h`` rows, then §D3 unchanged."""
    cap = min(S.shape[0], SUBSAMPLE_MULT * n_h)
    if cap < S.shape[0]:
        # A keyed BLAKE2b over the row's *window id*, not ``Generator.choice``. This
        # is the map's standing determinism rule and it bites hardest here: under this
        # fallback the subsample is not a convenience, it is the frame the selection
        # is made from, so which rows it holds is part of the corpus contract. NumPy
        # guarantees bit-generator streams across releases but not the distributions
        # layered on them, and ``choice`` is such a distribution.
        sub = np.sort(_hash_order(window_id, f"d3-subsample:{seed}")[:cap])
    else:
        sub = np.arange(S.shape[0])
    picks = select_lloyd(
        np.ascontiguousarray(S[sub]), n_h, geocell[sub], window_id[sub], seed=seed
    )
    return sub[picks]


def select_cheap(
    S: np.ndarray, n_h: int, geocell: np.ndarray, window_id: np.ndarray, seed: int = SEED
) -> np.ndarray:
    """Fallback 3: no clustering at all — geo-cell round-robin + a spacing constraint.

    Walks the stratum's rows in geo-cell round-robin order (the machinery
    ``sampling.py`` already ships) and accepts a row only when it is at least ``d_min``
    from every row already accepted, with ``d_min`` shrinking until the quota fills.
    This is the "replace FSCS entirely" option: it spreads by *rejection* rather than
    by partition, so it never pays the N*k term.
    """
    from scipy.spatial import cKDTree

    shuffled = _hash_order(window_id, f"d3-cheap:{seed}")
    # Round-robin order: rank each row within its own geo-cell by that stable shuffle,
    # then sort by (rank, shuffle position). Every cell contributes its 1st candidate
    # before any cell contributes its 2nd — the shape ``sampling.py`` uses across COGs.
    pos = np.empty(geocell.size, dtype=np.int64)
    pos[shuffled] = np.arange(geocell.size)
    _, cell_idx = np.unique(geocell, return_inverse=True)
    rank = np.empty(geocell.size, dtype=np.int64)
    for c in range(cell_idx.max() + 1):
        rows = np.flatnonzero(cell_idx == c)
        rank[rows[np.argsort(pos[rows], kind="stable")]] = np.arange(rows.size)
    order = np.lexsort((pos, rank))

    # Typical nearest-neighbour spacing if n_h points were spread evenly through the
    # stratum's bounding box: volume per point, to the 1/d power. Start there and
    # relax geometrically until the quota fills.
    d = S.shape[1]
    span = float(np.mean(S.max(axis=0) - S.min(axis=0)))
    d_min = span * (1.0 / n_h) ** (1.0 / d)
    tree = cKDTree(S)
    for _ in range(20):
        # Rejection by exclusion ball: accepting a row knocks out everything within
        # d_min of it, so the scan is one pass over ``order`` and the total tree work
        # is bounded by the number of rows excluded, not by n_h^2.
        alive = np.ones(S.shape[0], dtype=bool)
        chosen: list[int] = []
        for row in order:
            if not alive[row]:
                continue
            chosen.append(int(row))
            if len(chosen) >= n_h:
                break
            alive[tree.query_ball_point(S[row], d_min)] = False
        if len(chosen) >= n_h:
            return np.array(sorted(chosen), dtype=np.int64)
        d_min *= 0.6
    return np.array(sorted(chosen), dtype=np.int64)


#: Only the rules the grid times. ``cheap`` is timed separately — its naive form above
#: is O(n_h^2) in Python and would dominate the table for the wrong reason.
METHODS = {
    "lloyd": select_lloyd,
    "minibatch": select_minibatch,
    "subsampled": select_subsampled,
}


# --- Measurement --------------------------------------------------------------


def peak_rss_mb() -> float:
    """Peak RSS of this process, MB. macOS reports bytes; Linux reports KB."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1e6 if sys.platform == "darwin" else raw / 1e3


def cmd_grid(args) -> int:
    Z, window_id, geocell = load_frame()
    print(f"frame {Z.shape} {Z.dtype}, {Z.nbytes / 1e6:.0f} MB", flush=True)
    labels = global_labels(Z)

    rows = []
    for N_h in N_H_VALUES:
        idx = synthetic_stratum(Z, labels, N_h)
        S = np.ascontiguousarray(Z[idx])
        g = geocell[idx]
        w = window_id[idx]
        for n_h in N_VALUES:
            if n_h > MAX_QUOTA_FRACTION * N_h:
                continue
            for name, fn in METHODS.items():
                base = peak_rss_mb()
                t0 = time.perf_counter()
                picks = fn(S, n_h, g, w, seed=SEED)
                dt = time.perf_counter() - t0
                rec = {
                    "N_h": N_h,
                    "n_h": n_h,
                    "method": name,
                    "seconds": dt,
                    "peak_rss_mb": peak_rss_mb(),
                    "rss_delta_mb": peak_rss_mb() - base,
                    "n_selected": int(picks.size),
                }
                rows.append(rec)
                print(
                    f"  N_h={N_h:>9,} n_h={n_h:>6,} {name:<11} {dt:8.2f}s  "
                    f"got {picks.size:>6,}  rss {rec['peak_rss_mb']:.0f} MB",
                    flush=True,
                )
                if name == "lloyd" and dt > CELL_BUDGET_S:
                    print("    (over cell budget — larger n_h at this N_h skipped)", flush=True)
                    break
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "d3_fscs_cost.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"-> {OUT / 'd3_fscs_cost.json'}")
    return 0


def cmd_determinism(args) -> int:
    """Same seed twice must give the same window ids; different seeds show the drift."""
    Z, window_id, geocell = load_frame()
    labels = global_labels(Z)
    idx = synthetic_stratum(Z, labels, args.N_h)
    S = np.ascontiguousarray(Z[idx])
    g = geocell[idx]
    wid = window_id[idx]

    out = []
    for name, fn in METHODS.items():
        a = set(wid[fn(S, args.n_h, g, wid, seed=0)].tolist())
        b = set(wid[fn(S, args.n_h, g, wid, seed=0)].tolist())
        c = set(wid[fn(S, args.n_h, g, wid, seed=1)].tolist())
        rec = {
            "method": name,
            "same_seed_identical": a == b,
            "same_seed_jaccard": len(a & b) / len(a | b),
            "cross_seed_jaccard": len(a & c) / len(a | c),
        }
        out.append(rec)
        print(
            f"  {name:<11} same-seed identical={rec['same_seed_identical']}  "
            f"cross-seed jaccard={rec['cross_seed_jaccard']:.3f}",
            flush=True,
        )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "d3_determinism.json").write_text(json.dumps(out, indent=2) + "\n")
    return 0


def cmd_spread(args) -> int:
    """What each rule costs in *accuracy*: how well the picks cover the stratum.

    Two numbers, because "spread" can mean either and they trade off:

    - **coverage**: mean distance from a stratum row to its nearest selected row —
      lower is better, and this is the quantity FSCS exists to minimise;
    - **min separation**: the closest pair among the selected rows — higher means
      fewer near-duplicates, which is the other half of what FSCS is bought for.

    Both against a uniform-random draw as the do-nothing baseline.
    """
    from scipy.spatial import cKDTree

    Z, window_id, geocell = load_frame()
    labels = global_labels(Z)
    idx = synthetic_stratum(Z, labels, args.N_h)
    S = np.ascontiguousarray(Z[idx])
    g = geocell[idx]
    wid = window_id[idx]

    def score(picks: np.ndarray) -> dict:
        P = S[picks]
        cover = cKDTree(P).query(S, k=1)[0].mean()
        dd = cKDTree(P).query(P, k=2)[0][:, 1]
        return {
            "n": int(picks.size),
            "coverage_mean_dist": float(cover),
            "min_separation": float(dd.min()),
            "median_separation": float(np.median(dd)),
        }

    rng = np.random.default_rng(SEED)
    results = {"random": score(rng.choice(S.shape[0], args.n_h, replace=False))}
    for name, fn in METHODS.items():
        results[name] = score(fn(S, args.n_h, g, wid, seed=SEED))
    t0 = time.perf_counter()
    results["cheap"] = score(select_cheap(S, args.n_h, g, wid, seed=SEED))
    results["cheap"]["seconds"] = time.perf_counter() - t0

    for name, r in results.items():
        print(
            f"  {name:<11} coverage {r['coverage_mean_dist']:.4f}  "
            f"min-sep {r['min_separation']:.4f}  med-sep {r['median_separation']:.4f}",
            flush=True,
        )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "d3_spread.json").write_text(
        json.dumps({"N_h": args.N_h, "n_h": args.n_h, "results": results}, indent=2) + "\n"
    )
    return 0


def cmd_project(args) -> int:
    """Sum the measured per-stratum cost over a real k=40 occupancy distribution.

    Two budgets: the 50k debug tier this map ships, and the 500k scale tier the
    sampler has to stay affordable at, since it is budget-parameterized (map §Notes).
    Allocation is the crude §D2 shape — floor first, remainder proportional to N_h —
    because #42 has not settled the real one and the cost model only needs the n_h
    distribution to be the right *size*, not the right rule.
    """
    Z, _, _ = load_frame()
    labels = global_labels(Z)
    sizes = np.bincount(labels)
    rows = json.loads((OUT / "d3_fscs_cost.json").read_text())

    def cost(method: str, N_h: int, n_h: int) -> float:
        """Bilinear-ish interpolation in log space over the measured grid."""
        pts = [(r["N_h"], r["n_h"], r["seconds"]) for r in rows if r["method"] == method]
        if not pts:
            return float("nan")
        # Cost is ~linear in N_h * n_h * iterations; fit a power law in (N_h, n_h).
        A = np.array([[1.0, np.log(a), np.log(b)] for a, b, _ in pts])
        y = np.log(np.array([c for _, _, c in pts]))
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        return float(np.exp(coef @ np.array([1.0, np.log(N_h), np.log(n_h)])))

    out = {}
    for budget in (50_000, 500_000):
        floor = max(500, int(0.01 * budget))
        k = sizes.size
        rest = budget - k * floor
        quota = floor + np.maximum(0, (rest * sizes / sizes.sum())).astype(int)
        per = {
            m: [cost(m, int(N), int(q)) for N, q in zip(sizes, quota)]
            for m in METHODS
        }
        out[str(budget)] = {
            "floor": floor,
            "k": int(k),
            "n_h_min": int(quota.min()),
            "n_h_max": int(quota.max()),
            "total_seconds": {m: float(np.sum(v)) for m, v in per.items()},
        }
        print(f"  budget {budget:,}: floor {floor}, n_h {quota.min():,}-{quota.max():,}")
        for m, v in per.items():
            print(f"    {m:<11} {np.sum(v) / 60:8.1f} min")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "d3_projection.json").write_text(json.dumps(out, indent=2) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="d3_fscs_cost_proto")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("grid").set_defaults(fn=cmd_grid)
    d = sub.add_parser("determinism")
    d.add_argument("--N-h", type=int, default=300_000)
    d.add_argument("--n-h", type=int, default=1_000)
    d.set_defaults(fn=cmd_determinism)
    s = sub.add_parser("spread")
    s.add_argument("--N-h", type=int, default=300_000)
    s.add_argument("--n-h", type=int, default=1_000)
    s.set_defaults(fn=cmd_spread)
    sub.add_parser("project").set_defaults(fn=cmd_project)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
