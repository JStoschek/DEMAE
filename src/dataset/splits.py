"""The geographic split — fill the manifest's ``split`` column, once (Phase 1 §9, issue #77).

The corpus card is unambiguous (§4.6): 423 source COGs were drawn dry and 2,788 more past
half, so two tiles from one COG share geology, climate, the same radar pass and the same
void-filling history. A random split puts immediate neighbours on both sides of the line
and validation reads better than it should. This module fills the column the manifest has
carried declared-and-null since #48, geographically blocked, and the assignment ships as a
committed artifact — the same contract philosophy as the selection list.

## Three splits, two blocking units

- **``test`` — whole 10-degree geocells.** The Phase-2 downstream evaluation regions,
  untouched by pretraining. Whole cells, so no train tile lies inside a test region.
- **``val`` — whole source COGs**, drawn from geocells *not* in ``test``. Masked-recon
  validation loss and the kNN probe. COG-level blocking is the *minimum* honest unit the
  corpus card allows; anything coarser would spend geographic coverage the train set needs.
- **``train`` — everything else.**

## One objective at both granularities: relative stratum shortfall

Both draws answer the same question — *does this held-out pool look like the corpus?* —
against the same measure, :func:`relative_shortfall`: ``sum_h max(0, q_h - p_h) / q_h``
over the k=20 strata, where ``q`` is the corpus composition and ``p`` the pool's. It is
scale-free by construction: a stratum at half its share costs 0.5 whether it holds 7% of
the corpus or 1.5%, and a *missing* stratum costs a whole 1.0. That is what makes plan §9's
"stratified so all 20 strata appear in the held-out pool" a property of the objective
rather than a floor bolted onto it — and it is why total-variation distance was rejected, which prices a missing rare
stratum at its share (0.015) and cannot see the difference.

The two draws differ only in how they search, and the difference is forced by measurement,
not taste (see ``docs/research/p77-split.md``):

- **Test (13 cells of 425):** small enough to optimize. A greedy seed, then steepest-descent
  single-cell swaps to a local optimum.
- **Val (~530 COGs of 22,906):** the same greedy at COG granularity was **measured and
  rejected**. Free to pick any of 22,203 eligible COGs, it minimizes shortfall by taking
  the *shallowest* ones — 3,115 COGs of mean depth 3.2 — and a COG holding two kept tiles
  is a COG the gates nearly emptied. It matched stratum composition to TV 0.0013 and
  carried **9.4x the corpus's coastline rate** while doing it. Val therefore draws by
  per-stratum tile quota in keyed-hash order: size-blind, so composition and provenance
  character both land within a few percent.

## Determinism is met by persisting, and by hashing

The house rule (``sampling.py``): every tie-break is a keyed BLAKE2b hash of the block id,
never a NumPy ``Generator`` — NumPy guarantees bit-generator streams across releases but
not the distributions layered on them. Nothing here re-fits, re-clusters, or reads a COG;
the inputs are the shipped manifest's own columns. But as with the selection list, the
guarantee the project rests on is that **the emitted assignment is committed and re-read**
(:func:`write_assignment` / :func:`read_assignment`), not that this code re-runs the same.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from . import config
from .materialize import file_md5
from .sampling import _shuffle_key
from .select import largest_remainder

#: Split labels, in the order they are reported. The manifest column holds exactly these.
TRAIN: Final = "train"
VAL: Final = "val"
TEST: Final = "test"
SPLITS: Final = (TRAIN, VAL, TEST)

# --- The design's numbers (Phase-1 plan §9) -----------------------------------
#: Share of the corpus held out as whole 10-degree geocells, and as whole source COGs.
TEST_FRACTION: Final = 0.03
VAL_FRACTION: Final = 0.02
#: How many geocells carry the test share. The plan fixes the window at 12-15; 13 is its
#: midpoint, and the shortfall the search reaches is flat across it (0.57-0.63), so the
#: count is a stated choice rather than a tuned one.
TEST_CELLS: Final = 13
#: The size band a test geocell must fall in, as a share of the test budget. See
#: :func:`select_test_geocells` for what each end buys.
TEST_CELL_MIN_SHARE: Final = 0.025
TEST_CELL_MAX_SHARE: Final = 0.25
#: Minimum great-circle separation between two test cells' tile centroids. A diversity
#: rule, not a leakage buffer — see :func:`select_test_geocells`.
TEST_CELL_SEPARATION_KM: Final = 1000.0


#: Mean Earth radius, km — the units every distance in this module is stated in.
EARTH_RADIUS_KM: Final = 6371.0

#: Non-termination guard on the swap search, not a knob: every accepted swap strictly
#: lowers the shortfall, so the loop is finite by construction and this only bounds a
#: floating-point pathology. The shipped draw converges in 4.
_MAX_SWAP_ROUNDS: Final = 100


@dataclass(frozen=True)
class Blocks:
    """The per-block view the two draws share: who holds what terrain, and where.

    A *block* is whatever unit is being held out whole — a 10-degree geocell for ``test``,
    a source COG for ``val``. Both draws see the same three facts about one, which is why
    one dataclass serves both: its id, its per-stratum tile counts, and the centroid of
    its own tiles (not of its cell — a geocell that holds land in one corner should be
    placed at that corner, since the separation rule is about where the terrain is).
    """

    ids: np.ndarray
    counts: np.ndarray
    lon: np.ndarray
    lat: np.ndarray

    @property
    def n(self) -> np.ndarray:
        """Tiles per block."""
        return self.counts.sum(axis=1)

    @property
    def k(self) -> int:
        """Number of strata."""
        return int(self.counts.shape[1])

    def composition(self) -> np.ndarray:
        """Corpus-wide stratum shares over every block held — the reference ``q``."""
        total = self.counts.sum(axis=0)
        return total / total.sum()


def blocks_from(
    block_id: np.ndarray,
    stratum_id: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    *,
    k: int,
) -> Blocks:
    """Aggregate flat tile columns into :class:`Blocks`, ids sorted.

    Sorted ids, and sums accumulated by :func:`numpy.add.at`, so the result is a pure
    function of the *contents* of the manifest and not of the order rows arrive in — the
    same independence :mod:`dataset.sampling` buys by hashing stems.
    """
    ids, index = np.unique(np.asarray(block_id), return_inverse=True)
    counts = np.zeros((len(ids), k), dtype=np.int64)
    np.add.at(counts, (index, np.asarray(stratum_id, dtype=np.int64)), 1)
    n = counts.sum(axis=1)
    clon = np.zeros(len(ids), dtype=np.float64)
    clat = np.zeros(len(ids), dtype=np.float64)
    np.add.at(clon, index, np.asarray(lon, dtype=np.float64))
    np.add.at(clat, index, np.asarray(lat, dtype=np.float64))
    return Blocks(ids=ids, counts=counts, lon=clon / n, lat=clat / n)


def relative_shortfall(counts: np.ndarray, reference: np.ndarray) -> float:
    """Total relative stratum shortfall of a pool against a reference composition.

    ``sum_h max(0, q_h - p_h) / q_h`` where ``p`` is ``counts`` normalized. Zero iff the
    pool matches the reference exactly (shares sum to one, so no shortfall means no
    surplus either). A stratum the pool misses entirely contributes a whole 1.0 however
    rare it is — the scale-free property the split's "all 20 strata" requirement needs.
    """
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return float(len(reference))
    p = counts / total
    return float((np.maximum(reference - p, 0.0) / reference).sum())


# --- Geometry and tie-breaks --------------------------------------------------


def great_circle_km(lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    """Great-circle distance in km, broadcasting like any NumPy ufunc.

    Haversine rather than the law of cosines: the separation floor is checked between
    centroids that can be a few km apart at high latitude, where the cosine form loses
    its precision to catastrophic cancellation.
    """
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dlam = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def tie_rank(ids: np.ndarray, seed: int = config.RANDOM_SEED) -> np.ndarray:
    """Rank of every block in keyed-hash order — the size-blind draw order ``val`` walks.

    The key is :func:`dataset.sampling._shuffle_key`, imported rather than re-derived:
    NumPy guarantees its bit-generator streams across releases but not the distributions
    layered on them, so every tie-break in this project is a keyed BLAKE2b of the block id.
    A geocell id and a COG stem are both "the name of a block", which is exactly what that
    function hashes — :mod:`dataset.select` reaches for the same one.
    """
    keys = [_shuffle_key(str(b), seed) for b in ids]
    order = sorted(range(len(keys)), key=lambda i: keys[i])
    rank = np.empty(len(keys), dtype=np.int64)
    rank[order] = np.arange(len(keys))
    return rank


# --- test: whole geocells -----------------------------------------------------


@dataclass(frozen=True)
class TestSelection:
    """The chosen test geocells, and what the search cost to find them."""

    ids: list[str]
    counts: np.ndarray
    shortfall: float
    seed_shortfall: float
    swaps: int
    eligible: int

    @property
    def n_tiles(self) -> int:
        return int(self.counts.sum())


def select_test_geocells(
    blocks: Blocks,
    *,
    budget: int,
    reference: np.ndarray | None = None,
    n_cells: int = TEST_CELLS,
    min_share: float = TEST_CELL_MIN_SHARE,
    max_share: float = TEST_CELL_MAX_SHARE,
    separation_km: float = TEST_CELL_SEPARATION_KM,
    seed: int = config.RANDOM_SEED,
) -> TestSelection:
    """Choose ``n_cells`` whole geocells minimizing :func:`relative_shortfall`.

    Constraints, each of which is an argument the resolution has to make rather than a
    fitted knob:

    - **the size band** ``[min_share, max_share] * budget`` — one cell beyond a quarter of
      the pool would let a single province decide a downstream verdict; one below a fortieth
      spends a slot without buying a region;
    - **``separation_km``** between the centroids of any two chosen cells — the rule that
      makes "distinct geomorphic provinces" operational rather than asserted. It is *not* a
      leakage buffer: two test cells may sit as close as they like without leaking, since
      both are held out. Its cost is measured, not assumed (``p77-split.md`` §3);
    - **``budget``** — the total is never exceeded, and the greedy reserves room for the
      cells it still owes so the count is reachable.

    Search is a greedy seed followed by steepest-descent single-cell swaps: replace the one
    cell whose replacement lowers the shortfall most, repeat until no swap helps. The seed
    alone is myopic (it spends its budget before it can see what it needed), and the swap
    pass is what makes the result insensitive to ``n_cells`` — both numbers are reported so
    the report can show that, and ties everywhere fall to :func:`tie_rank`'s keyed hash.
    """
    q = blocks.composition() if reference is None else np.asarray(reference, dtype=np.float64)
    n = blocks.n
    pool = np.flatnonzero((n >= min_share * budget) & (n <= max_share * budget))
    if len(pool) < n_cells:
        raise ValueError(
            f"only {len(pool)} geocells fall inside the size band "
            f"[{min_share:.3f}, {max_share:.3f}] x {budget}; need {n_cells}"
        )
    keys = {int(i): _shuffle_key(str(blocks.ids[i]), seed) for i in pool}
    dist = great_circle_km(blocks.lon[pool][:, None], blocks.lat[pool][:, None], blocks.lon[pool], blocks.lat[pool])
    at = {int(b): j for j, b in enumerate(pool)}

    def separated(cand: int, chosen: list[int]) -> bool:
        return all(dist[at[cand], at[c]] >= separation_km for c in chosen)

    # --- greedy seed, reserving room for the cells still owed
    chosen: list[int] = []
    have = np.zeros(blocks.k, dtype=np.int64)
    while len(chosen) < n_cells:
        far = [int(i) for i in pool if int(i) not in chosen and separated(int(i), chosen)]
        owed = n_cells - len(chosen) - 1
        reserve = int(sum(sorted(int(n[i]) for i in far)[:owed])) if owed else 0
        fits = [i for i in far if have.sum() + n[i] + reserve <= budget]
        if not fits:
            break
        pick = min(fits, key=lambda i: (relative_shortfall(have + blocks.counts[i], q), keys[i]))
        chosen.append(pick)
        have = have + blocks.counts[pick]
    seed_shortfall = relative_shortfall(have, q)

    # --- steepest descent over single-cell swaps
    swaps = 0
    for _ in range(_MAX_SWAP_ROUNDS):
        current = relative_shortfall(have, q)
        best: tuple[tuple[float, bytes], int, int] | None = None
        for pos, out in enumerate(chosen):
            rest = chosen[:pos] + chosen[pos + 1 :]
            base = have - blocks.counts[out]
            for i in pool:
                i = int(i)
                if i in chosen or not separated(i, rest) or base.sum() + n[i] > budget:
                    continue
                score = relative_shortfall(base + blocks.counts[i], q)
                if score < current - 1e-12 and (best is None or (score, keys[i]) < best[0]):
                    best = ((score, keys[i]), pos, i)
        if best is None:
            break
        _, pos, incoming = best
        have = have - blocks.counts[chosen[pos]] + blocks.counts[incoming]
        chosen[pos] = incoming
        swaps += 1

    order = sorted(chosen, key=lambda i: str(blocks.ids[i]))
    return TestSelection(
        ids=[str(blocks.ids[i]) for i in order],
        counts=blocks.counts[order].sum(axis=0),
        shortfall=relative_shortfall(have, q),
        seed_shortfall=seed_shortfall,
        swaps=swaps,
        eligible=len(pool),
    )


# --- val: whole source COGs ---------------------------------------------------


@dataclass(frozen=True)
class ValSelection:
    """The drawn val COGs, the quota they were drawn against, and what they realized."""

    ids: list[str]
    counts_by_block: dict[str, np.ndarray]
    quota: np.ndarray
    spent: np.ndarray
    counts: np.ndarray
    eligible: int

    @property
    def n_tiles(self) -> int:
        return int(self.counts.sum())

    @property
    def mean_depth(self) -> float:
        """Tiles per drawn COG — the number that catches a shallow-COG draw."""
        return float(self.n_tiles / len(self.ids)) if self.ids else 0.0


def select_val_cogs(
    blocks: Blocks,
    *,
    budget: int,
    reference: np.ndarray | None = None,
    eligible: np.ndarray | None = None,
    seed: int = config.RANDOM_SEED,
) -> ValSelection:
    """Draw whole source COGs to a per-stratum tile quota, in keyed-hash order.

    The quota is :func:`dataset.select.largest_remainder` over ``reference`` — the same
    apportionment the Neyman allocation uses, so the arithmetic is one implementation and
    the total is exactly ``budget`` when the pool can supply it. Each COG is charged to its
    **dominant** stratum (the one holding most of its tiles; ties to the lowest id) and
    taken whole or not at all; a COG that would overshoot its stratum's quota is skipped,
    not truncated, because half a COG is not a blocking unit.

    Order is :func:`tie_rank`, which is **size-blind on purpose**. The alternative — greedy
    on :func:`relative_shortfall` — was implemented and measured: it matched composition
    better (TV 0.0013 against 0.045) by taking 3,115 COGs of mean depth 3.2, and drew
    coastline tiles at 9.4x the corpus rate, because the COGs that let a greedy take small
    steps are the ones the gates nearly emptied. Composition is not the only thing a val
    set has to look like.
    """
    q = blocks.composition() if reference is None else np.asarray(reference, dtype=np.float64)
    quota = largest_remainder(q, budget)
    mask = np.ones(len(blocks.ids), dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)
    n = blocks.n
    dominant = blocks.counts.argmax(axis=1)
    rank = tie_rank(blocks.ids, seed)

    spent = np.zeros(blocks.k, dtype=np.int64)
    drawn: list[int] = []
    for i in np.argsort(rank, kind="stable"):
        i = int(i)
        if not mask[i] or n[i] == 0:
            continue
        h = int(dominant[i])
        if spent[h] + n[i] > quota[h]:
            continue
        drawn.append(i)
        spent[h] += int(n[i])
    order = sorted(drawn, key=lambda i: str(blocks.ids[i]))
    ids = [str(blocks.ids[i]) for i in order]
    return ValSelection(
        ids=ids,
        counts_by_block={str(blocks.ids[i]): blocks.counts[i] for i in order},
        quota=quota,
        spent=spent,
        counts=blocks.counts[order].sum(axis=0) if order else np.zeros(blocks.k, dtype=np.int64),
        eligible=int(mask.sum()),
    )


# --- the assignment, and the artifact that carries it -------------------------


def assign_splits(
    geocell_id: np.ndarray,
    source_cog: np.ndarray,
    *,
    test_cells: list[str],
    val_cogs: list[str],
) -> np.ndarray:
    """Label every tile ``train`` / ``val`` / ``test`` from the two held-out block lists.

    ``test`` takes precedence over ``val``. The draw never puts a val COG inside a test
    geocell — :func:`select_val_cogs` is handed an eligibility mask that excludes them —
    but the *labelling* still has to be total and unambiguous, and the guarantee worth
    keeping when the two disagree is the one Phase 2 depends on: a test region is
    untouched.

    Both lists are checked against the corpus. A geocell or COG the manifest does not hold
    is a typo or a stale artifact, and silently assigning nothing would ship a split whose
    committed cell list and realized column disagree.
    """
    cells = np.asarray(geocell_id)
    cogs = np.asarray(source_cog)
    missing_cells = sorted(set(test_cells) - set(np.unique(cells).tolist()))
    missing_cogs = sorted(set(val_cogs) - set(np.unique(cogs).tolist()))
    if missing_cells or missing_cogs:
        raise ValueError(f"blocks not present in the corpus: cells {missing_cells}, COGs {missing_cogs}")
    out = np.full(len(cells), TRAIN, dtype=object)
    out[np.isin(cogs, list(val_cogs))] = VAL
    out[np.isin(cells, list(test_cells))] = TEST
    return out


@dataclass(frozen=True)
class SplitPlan:
    """The whole assignment: the two draws, the per-tile labels, and what they realized."""

    tile_id: np.ndarray
    labels: np.ndarray
    test: TestSelection
    val: ValSelection
    reference: np.ndarray
    counts: dict[str, np.ndarray]

    def totals(self) -> dict[str, int]:
        return {s: int(self.counts[s].sum()) for s in SPLITS}

    def shares(self) -> dict[str, float]:
        n = len(self.labels)
        return {s: self.counts[s].sum() / n for s in SPLITS}


def plan_splits(
    *,
    tile_id: np.ndarray,
    geocell_id: np.ndarray,
    source_cog: np.ndarray,
    stratum_id: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    k: int,
    test_fraction: float = TEST_FRACTION,
    val_fraction: float = VAL_FRACTION,
    n_cells: int = TEST_CELLS,
    min_share: float = TEST_CELL_MIN_SHARE,
    max_share: float = TEST_CELL_MAX_SHARE,
    separation_km: float = TEST_CELL_SEPARATION_KM,
    seed: int = config.RANDOM_SEED,
) -> SplitPlan:
    """Draw both hold-outs and label every tile — the whole of plan §9 in one call.

    Order matters and is the design: ``test`` is drawn first over whole geocells, then
    ``val`` over whole COGs **from the geocells test did not take**, then everything left
    is ``train``. Both draws score against the same reference — the corpus's own stratum
    composition — so "representative" means one thing in this module.
    """
    cells = blocks_from(geocell_id, stratum_id, lon, lat, k=k)
    cogs = blocks_from(source_cog, stratum_id, lon, lat, k=k)
    reference = cells.composition()
    n_tiles = len(tile_id)

    test = select_test_geocells(
        cells,
        budget=round(test_fraction * n_tiles),
        reference=reference,
        n_cells=n_cells,
        min_share=min_share,
        max_share=max_share,
        separation_km=separation_km,
        seed=seed,
    )
    # a COG sits in exactly one geocell (10-degree cells tile the 1-degree grid), so its
    # eligibility is a lookup, not a straddle test
    cog_cell = {}
    for cog, cell in zip(np.asarray(source_cog), np.asarray(geocell_id)):
        cog_cell.setdefault(str(cog), str(cell))
    eligible = np.array([cog_cell[str(c)] not in set(test.ids) for c in cogs.ids])
    val = select_val_cogs(
        cogs,
        budget=round(val_fraction * n_tiles),
        reference=reference,
        eligible=eligible,
        seed=seed,
    )

    labels = assign_splits(geocell_id, source_cog, test_cells=test.ids, val_cogs=val.ids)
    counts = {}
    for s in SPLITS:
        c = np.zeros(k, dtype=np.int64)
        np.add.at(c, np.asarray(stratum_id)[labels == s].astype(np.int64), 1)
        counts[s] = c

    # Plan §9 requires every stratum in the held-out pool, and the shortfall objective
    # prices a missing one at a whole unit — which makes that likely, not certain. Checking
    # it is what stops a re-draw at other parameters from quietly shipping a pool with a
    # terrain family nothing downstream can evaluate.
    for s in (VAL, TEST):
        empty = np.flatnonzero(counts[s] == 0)
        if empty.size:
            raise ValueError(
                f"the {s} pool misses stratum {', '.join(map(str, empty))} of {k}; "
                f"plan §9 requires all {k} in the held-out pool"
            )
    return SplitPlan(
        tile_id=np.asarray(tile_id), labels=labels, test=test, val=val,
        reference=reference, counts=counts,
    )


def write_assignment(tile_id: np.ndarray, split: np.ndarray, path: Path | str) -> Path:
    """Write the committed ``tile_id,split`` CSV, sorted by ``tile_id``.

    Sorted and LF-terminated, so two runs that agree on the assignment produce a
    byte-identical file whatever order the manifest handed the rows over in — the same
    property that makes the selection list's md5 a contract rather than a timestamp.
    """
    tile_id = np.asarray(tile_id)
    split = np.asarray(split)
    unknown = sorted(set(map(str, np.unique(split))) - set(SPLITS))
    if unknown:
        raise ValueError(f"unknown split label(s) {unknown}; expected one of {SPLITS}")
    order = np.argsort(tile_id.astype(str), kind="stable")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["tile_id,split"]
    lines += [f"{tile_id[i]},{split[i]}" for i in order]
    path.write_text("\n".join(lines) + "\n")
    return path


def read_assignment(path: Path | str) -> dict[str, str]:
    """Read the committed CSV back. Neither column ever contains a comma or a quote."""
    lines = Path(path).read_text().splitlines()
    if not lines or lines[0] != "tile_id,split":
        raise ValueError(f"{path} is not a split assignment (header {lines[:1]})")
    out = {}
    for line in lines[1:]:
        tid, label = line.rsplit(",", 1)
        if label not in SPLITS:
            raise ValueError(f"unknown split label {label!r} in {path}")
        out[tid] = label
    return out


def fill_manifest_split(path: Path | str, assignment: dict[str, str]) -> Path:
    """Fill the manifest's ``split`` column in place, from the committed assignment.

    Everything else is left untouched, and that is checked rather than trusted: rows keep
    their order and their values, and the file is re-written under ``MANIFEST_SCHEMA``, so
    a schema drift fails here instead of in the loader. ``dataset_version`` hashes sorted
    ``(tile_id, content_hash)`` and ``build_params`` alone (ADR-0002 §3), so filling this
    column does **not** move the shipped ``bcbd6484…`` fingerprint — which is the reason
    the column could be declared-and-null in the first place.

    The assignment must cover every tile. A partial fill would leave nulls that read as
    "not yet split" and silently rejoin held-out tiles to training.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from .manifest import MANIFEST_SCHEMA

    path = Path(path)
    table = pq.read_table(path)
    tile_ids = table["tile_id"].to_pylist()
    missing = [t for t in tile_ids if t not in assignment]
    if missing:
        raise ValueError(
            f"the assignment does not cover {len(missing)} of {len(tile_ids)} tiles "
            f"(first: {missing[:3]})"
        )
    labels = [assignment[t] for t in tile_ids]
    filled = table.set_column(
        table.schema.get_field_index("split"), "split", pa.array(labels, type=pa.string())
    )
    pq.write_table(filled.cast(MANIFEST_SCHEMA), path)
    return path


# --- Driving it from the shipped manifest -------------------------------------

#: The manifest columns the whole draw reads. Nothing else is touched, and no COG is read.
MANIFEST_COLUMNS: Final = ("tile_id", "geocell_id", "source_cog", "stratum_id", "lon", "lat")


def plan_from_manifest(path: Path | str, **kwargs) -> SplitPlan:
    """Draw the split for a shipped corpus, reading only :data:`MANIFEST_COLUMNS`."""
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=list(MANIFEST_COLUMNS))
    cols = {c: table[c].to_numpy(zero_copy_only=False) for c in MANIFEST_COLUMNS}
    k = int(cols["stratum_id"].max()) + 1
    return plan_splits(
        tile_id=cols["tile_id"], geocell_id=cols["geocell_id"], source_cog=cols["source_cog"],
        stratum_id=cols["stratum_id"], lon=cols["lon"], lat=cols["lat"], k=k, **kwargs
    )


def write_blocks(plan: SplitPlan, path: Path | str, *, assignment_md5: str) -> Path:
    """Write the geocell/COG lists beside the assignment, with its md5 (plan §9d).

    The CSV says which tile went where; this says *what was held out as a block*, which is
    the thing a reviewer checks and a Phase-2 task builder reads to find its evaluation
    regions. It carries the assignment's md5 so the two cannot drift apart unnoticed.
    """
    import json

    doc = {
        "schema_version": 1,
        "assignment_md5": assignment_md5,
        "n_tiles": int(len(plan.labels)),
        "totals": plan.totals(),
        "shares": {s: round(v, 6) for s, v in plan.shares().items()},
        "reference_composition": [round(float(x), 8) for x in plan.reference],
        "test": {
            "block": "geocell_id",
            "n_cells": len(plan.test.ids),
            "cells": plan.test.ids,
            "eligible_cells": plan.test.eligible,
            "shortfall": round(plan.test.shortfall, 6),
            "seed_shortfall": round(plan.test.seed_shortfall, 6),
            "swaps": plan.test.swaps,
        },
        "val": {
            "block": "source_cog",
            "n_cogs": len(plan.val.ids),
            "cogs": plan.val.ids,
            "eligible_cogs": plan.val.eligible,
            "mean_depth": round(plan.val.mean_depth, 3),
            "quota": [int(x) for x in plan.val.quota],
        },
        "per_stratum": {s: [int(x) for x in plan.counts[s]] for s in SPLITS},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    """``python -m dataset.splits`` — draw the split, commit it, and fill the column.

    One command, run once per corpus tier. ``--fill`` is opt-in: drawing and reporting the
    assignment is safe to repeat, writing the manifest is the irreversible half.
    """
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m dataset.splits",
        description="Draw the geographic train/val/test split and commit the assignment (#77).",
    )
    p.add_argument("--manifest", type=Path, default=Path("data/corpus_500k/manifest.parquet"))
    p.add_argument("--out", type=Path, default=Path(config.SPLIT_ASSIGNMENT_PATH))
    p.add_argument("--blocks", type=Path, default=Path(config.SPLIT_BLOCKS_PATH))
    p.add_argument("--n-cells", type=int, default=TEST_CELLS)
    p.add_argument("--test-fraction", type=float, default=TEST_FRACTION)
    p.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    p.add_argument("--fill", action="store_true", help="write the split column into the manifest")
    args = p.parse_args(argv)

    plan = plan_from_manifest(
        args.manifest, test_fraction=args.test_fraction, val_fraction=args.val_fraction,
        n_cells=args.n_cells, seed=args.seed,
    )
    out = write_assignment(plan.tile_id, plan.labels, args.out)
    digest = md5(out)
    blocks = write_blocks(plan, args.blocks, assignment_md5=digest)

    totals, shares = plan.totals(), plan.shares()
    print(f"Split {len(plan.labels):,} tiles of {args.manifest}:")
    for s in SPLITS:
        print(f"  {s:5s} {totals[s]:7,d}  {shares[s]:7.3%}")
    print(f"  test:  {len(plan.test.ids)} geocells of {plan.test.eligible} eligible, "
          f"shortfall {plan.test.seed_shortfall:.4f} -> {plan.test.shortfall:.4f} "
          f"in {plan.test.swaps} swaps")
    print(f"  val:   {len(plan.val.ids)} COGs of {plan.val.eligible} eligible, "
          f"mean depth {plan.val.mean_depth:.1f}")
    print(f"  assignment: {out}  md5 {digest}")
    print(f"  blocks:     {blocks}")
    if args.fill:
        filled = fill_manifest_split(args.manifest, read_assignment(out))
        print(f"  filled:     {filled}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
