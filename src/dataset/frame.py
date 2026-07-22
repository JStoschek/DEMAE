"""The stratification frame — what k-means actually sees (Phase D1, issue #40).

Phase C handed over 3,052,735 kept windows scored on ADR-0004's frozen six. This
module is the one definition of how those rows become a matrix: which rows, which
columns, which transforms, and the standardization that puts all six on a common
scale. Everything downstream in Phase D — the k sweep (#41), the sampler (#44) — goes
through here, so a frame can only be built one way.

**Standardize, don't whiten** (ADR-0004). The six are z-scored after the ADR's two
``ln(x + 1)`` transforms; no PCA and no whitening enter the k-means distance, so a
centroid coordinate is still readable as "1.4 sd above mean log slope".

**The scaler is a shipped artifact, not incidental state.** It is fitted once, written
to ``artifacts/phase_d/scaler.json``, and *loaded* by every later run. Phase F re-runs
Phase D at a 200-500k budget against the same strata; a scaler silently re-fitted on a
different frame moves every centroid without changing a line of code, and the two
tiers stop being comparable while still looking it. So the artifact carries a
**fingerprint** of the frame it was fitted over (row count, the C4 gate thresholds, the
source catalog's ``code_version``) and :func:`read_scaler` refuses a frame that does
not match it.

## The three frame decisions, and the measurements behind them

Recorded in full in `docs/research/d1-stratification-frame.md`; in brief:

1. **The scaler is fitted over all 3,052,735 kept rows in one full pass** — no
   subsample. Not for accuracy (a 100k subsample reproduces every mean and sd to
   within 0.5% relative) but because the full pass costs seconds, and a subsample
   would make a sampling seed part of a contract that has to hold for the life of the
   project.

2. **``water_affected`` rows are fitted like any other row.** Plan §C2 promised to
   exclude them from "texture-based stratification distances" — **amended here**. That
   sentence predates the metric vector, and the measurement does not support it: the
   159,011 tagged rows differ from the rest by **0.030 sd in ``texture``**, the
   smallest gap of the six and 17x smaller than the largest. They are lower, flatter
   and more landform-mixed (``mean_elevation`` -0.55 sd, ``elevation_relief_ratio``
   -0.53, ``geomorphon_entropy`` +0.51) — wet, not texturally anomalous. The flag
   travels into the Phase-E manifest, where a dry-only corpus is one filter away.

3. **``coastline`` rows are fitted like any other row too**, and the tag becomes a
   reporting axis across strata rather than a stratum. §C2's "their own reportable
   stratum" is **amended** on the same footing: the 40,815 coastal rows scatter across
   21 strata to reach 90% of their mass, and the most coastal stratum is 4.9%
   coastline. ``coastline`` is a fact about where a window *is* (``exact_zero_frac <
   0.75 AND wbm1_frac > 0``), and the six metrics cannot see the ocean — carving it out
   would build a stratum internally heterogeneous in the very descriptors the design
   rests on. The presence guarantee that carving would have bought is owed by the
   allocation rule (#42) as a minimum coastline count.

Both exclusions were also tested directly, by fitting k=40 with and without each group
and comparing where the shared rows land (adjusted Rand index). Excluding
``water_affected`` scores 0.698 and excluding ``coastline`` 0.656 against a **noise
floor of 0.607** — two fits of the *same* frame differing only in seed. Neither tag's
inclusion moves the strata as much as re-running the fit does.

**A note for whoever fits the centroids (#41).** That noise floor is not an artifact of
this measurement: at k=40 two full-frame fits land 0.06% apart in inertia while
agreeing on only ~61% of pairs, which is genuine multi-modality. ``n_init`` is the
lever (25 restarts on a 500k subsample lifts agreement to 0.889), and a full-frame fit
costs 7-9 s in float32, so restarts are cheap. Inertia differences between neighbouring
k must clear that floor before they mean anything.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np

if TYPE_CHECKING:  # pyarrow is imported lazily — only the catalog reader needs it
    import pyarrow as pa

from . import catalog, gates

# --- The frozen vector (ADR-0004 §1) ------------------------------------------

#: The six metrics, in ADR-0004 §1 table order. Column order is part of the artifact
#: contract: the scaler and the centroids are plain arrays, so a reordering here
#: silently mismatches every persisted vector rather than raising.
METRICS: Final = (
    "slope_mean",
    "local_relief",
    "texture",
    "elevation_relief_ratio",
    "mean_elevation",
    "geomorphon_entropy",
)

#: The two ``ln(x + 1)`` transforms — k-means only (ADR-0004 §1). They collapse long
#: right tails so no raw scale dominates the distance (``local_relief`` skew +3.58 ->
#: -0.29, ``slope_mean`` +2.48 -> +0.68); the ``+1`` also absorbs the ~0.8% exact-zero
#: windows. The other four are left raw: wrong skew direction, a bounded fraction, or
#: genuine negatives (``mean_elevation`` reaches -28 m) that break ``ln``.
LOG_METRICS: Final = ("slope_mean", "local_relief")

#: Where the fitted scaler lives. Committed — ``data/`` is gitignored and this is not
#: regenerable-on-demand state: Phase F's comparability rests on re-loading this file.
SCALER_PATH: Final = Path("artifacts/phase_d/scaler.json")
#: Bumped when the artifact's field set changes, so an old file fails to load loudly.
SCALER_SCHEMA_VERSION: Final = 1


# --- Transform ----------------------------------------------------------------


def metric_matrix(columns: "pa.Table | dict[str, Any]") -> np.ndarray:
    """The ``(n, 6)`` transformed metric matrix — ADR-0004's log set applied, unscaled.

    Accepts anything column-addressable by metric name (a ``pyarrow.Table`` or a plain
    mapping of arrays), so the tests can hand-build a frame without Parquet.

    Always ``float64``. The catalog stores float32, and a mean over 3 M float32 values
    accumulates visible error; the sampler casts back down to float32 for the k-means
    fit, where the precision does not matter and the memory does.
    """
    n = None
    out_cols: list[np.ndarray] = []
    for name in METRICS:
        col = columns[name]
        arr = col.to_numpy(zero_copy_only=False) if hasattr(col, "to_numpy") else np.asarray(col)
        arr = arr.astype(np.float64, copy=False)
        if n is None:
            n = arr.size
        out_cols.append(np.log(arr + 1.0) if name in LOG_METRICS else arr)
    return np.column_stack(out_cols) if out_cols else np.empty((0, 0))


# --- The scaler artifact ------------------------------------------------------


@dataclass(frozen=True)
class Scaler:
    """Per-dimension mean and sd over the fit frame, plus the fingerprint of that frame.

    The fingerprint is not decoration. The scaler is a set of six numbers that look
    equally plausible whatever frame produced them, so nothing about a re-used scaler
    is self-evidently wrong — the fingerprint is the only thing that can tell a Phase-F
    run that the catalog underneath it moved.
    """

    mean: np.ndarray
    std: np.ndarray
    fingerprint: dict[str, Any]

    def __post_init__(self) -> None:
        if self.mean.shape != (len(METRICS),) or self.std.shape != (len(METRICS),):
            raise ValueError(
                f"scaler must carry one mean and one sd per metric ({len(METRICS)}); "
                f"got mean {self.mean.shape}, std {self.std.shape}"
            )
        if not np.all(self.std > 0):
            zero = [m for m, s in zip(METRICS, self.std) if not s > 0]
            raise ValueError(f"zero-variance metric(s) in the fit frame: {zero}")

    def standardize(self, X: np.ndarray) -> np.ndarray:
        """z-score a transformed metric matrix. No whitening (ADR-0004)."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != len(METRICS):
            raise ValueError(f"expected an (n, {len(METRICS)}) matrix; got {X.shape}")
        return (X - self.mean) / self.std

    def as_dict(self) -> dict[str, Any]:
        """The artifact's JSON form — per-metric, not two bare arrays.

        Named keys rather than positional lists so a human reading the file can see
        which number belongs to which metric, and so a column reordering shows up as a
        key mismatch instead of six plausible-looking wrong numbers.
        """
        return {
            "schema_version": SCALER_SCHEMA_VERSION,
            "metrics": list(METRICS),
            "log_metrics": list(LOG_METRICS),
            "standardization": "z-score (standardize, not whiten — ADR-0004)",
            "params": {
                name: {"mean": float(self.mean[j]), "std": float(self.std[j])}
                for j, name in enumerate(METRICS)
            },
            "frame": dict(self.fingerprint),
        }

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "Scaler":
        if obj.get("schema_version") != SCALER_SCHEMA_VERSION:
            raise ValueError(
                f"scaler schema_version {obj.get('schema_version')!r} != "
                f"{SCALER_SCHEMA_VERSION} — refit rather than guess the field set"
            )
        if tuple(obj.get("metrics", ())) != METRICS:
            raise ValueError(
                f"scaler metric order {tuple(obj.get('metrics', ()))} != {METRICS} — "
                "the persisted vector no longer lines up with this code"
            )
        params = obj["params"]
        return cls(
            mean=np.array([params[m]["mean"] for m in METRICS], dtype=np.float64),
            std=np.array([params[m]["std"] for m in METRICS], dtype=np.float64),
            fingerprint=dict(obj.get("frame", {})),
        )


def fit_scaler(X: np.ndarray, *, fingerprint: dict[str, Any]) -> Scaler:
    """Fit mean/sd over **every row of** ``X`` — the caller has already chosen the frame.

    Deliberately unable to subsample: the frame decision belongs to :func:`kept_frame`
    and the artifact, not to a keyword argument that a later caller could pass
    differently without anyone noticing.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != len(METRICS):
        raise ValueError(f"expected an (n, {len(METRICS)}) matrix; got {X.shape}")
    if X.shape[0] == 0:
        raise ValueError("cannot fit a scaler over an empty frame")
    if not np.isfinite(X).all():
        raise ValueError(
            "non-finite value in the fit frame — the gated catalog has 0 NaN metrics "
            "among kept rows (#37), so this means the frame is not the kept frame"
        )
    return Scaler(mean=X.mean(axis=0), std=X.std(axis=0), fingerprint=dict(fingerprint))


def write_scaler(scaler: Scaler, path: Path | str = SCALER_PATH) -> Path:
    """Write the scaler artifact, creating its directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scaler.as_dict(), indent=2) + "\n")
    return path


def read_scaler(
    path: Path | str = SCALER_PATH,
    *,
    fingerprint: dict[str, Any] | None = None,
) -> Scaler:
    """Load the scaler, refusing a frame that does not match the one it was fitted over.

    Pass ``fingerprint`` — from :func:`frame_fingerprint` — whenever the caller has a
    catalog in hand. A mismatch names the fields that differ and raises: re-using a
    scaler across a moved frame is the one failure mode that produces a complete,
    plausible, wrong corpus rather than an error.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"no scaler at {path} — fit one with `python -m dataset.frame`"
        )
    scaler = Scaler.from_dict(json.loads(path.read_text()))
    if fingerprint is not None:
        drift = {
            k: (scaler.fingerprint.get(k), v)
            for k, v in fingerprint.items()
            if scaler.fingerprint.get(k) != v
        }
        if drift:
            detail = "; ".join(f"{k}: artifact {a!r} != frame {b!r}" for k, (a, b) in drift.items())
            raise ValueError(
                f"scaler at {path} was fitted over a different frame — {detail}. "
                "Re-fit, and expect every centroid to move."
            )
    return scaler


# --- Reading the frame off the gated catalog ----------------------------------


def frame_fingerprint(catalog_dir: Path | str, *, n_kept: int) -> dict[str, Any]:
    """What has to be identical for a persisted scaler to still apply.

    Three fields, each covering a different way the frame can move underneath the
    artifact: ``n_kept`` (rows added or dropped), ``gate_thresholds`` (the same rows
    re-gated differently), ``catalog_code_version`` (the same gates over re-scored
    metrics).
    """
    meta_path = Path(catalog_dir) / catalog.META_FILE
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return {
        "n_kept": int(n_kept),
        "gate_thresholds": meta.get("thresholds", gates.thresholds()),
        "catalog_code_version": meta.get("source_catalog", {}).get("code_version"),
    }


@dataclass(frozen=True)
class Frame:
    """The kept frame as k-means sees it, plus the tags carried alongside it.

    ``water_affected`` and ``coastline`` ride along *unused by the fit* — they are the
    reporting and boost axes the allocation rule (#42) needs, and keeping them beside
    the matrix is what stops a later caller re-reading the catalog to recover them and
    getting the row order subtly wrong.
    """

    X: np.ndarray  # (n, 6), transformed, unscaled
    window_id: np.ndarray
    water_affected: np.ndarray
    coastline: np.ndarray
    fingerprint: dict[str, Any]

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    def standardized(self, scaler: Scaler) -> np.ndarray:
        return scaler.standardize(self.X)


#: Columns the frame carries beside the six metrics — identity plus the two tags.
CARRIED: Final = ("window_id", "water_affected", "coastline")


def kept_frame(catalog_dir: Path | str = Path("data/catalog_gated")) -> Frame:
    """Read the gated catalog and keep the rows where ``keep`` is true — the whole frame.

    Reads only the nine columns it needs out of the gated catalog's thirty. The full
    gated frame is ~190 MB of zstd Parquet that expands well past its own size in
    memory, and the stratifier has no use for the mask fractions or the gate booleans
    it took to compute ``keep``.
    """
    import pyarrow.parquet as pq

    catalog_dir = Path(catalog_dir)
    table = pq.read_table(
        catalog_dir / catalog.PARTS_DIR,
        columns=[*METRICS, "keep", *CARRIED],
    )
    keep = table.column("keep").to_numpy(zero_copy_only=False).astype(bool)
    idx = np.flatnonzero(keep)

    def carried(name: str) -> np.ndarray:
        return table.column(name).to_numpy(zero_copy_only=False)[idx]

    X = metric_matrix({m: table.column(m) for m in METRICS})[idx]
    return Frame(
        X=X,
        window_id=carried("window_id"),
        water_affected=carried("water_affected").astype(bool),
        coastline=carried("coastline").astype(bool),
        fingerprint=frame_fingerprint(catalog_dir, n_kept=idx.size),
    )


# --- Live driver --------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """``python -m dataset.frame`` — fit the scaler over the kept frame and write it.

    Seconds, not hours: this reads Parquet the global pass already wrote and touches no
    COG. Re-run it whenever the gated catalog is rebuilt — and then expect to re-fit
    the centroids too, because every one of them moves.
    """
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m dataset.frame",
        description="Fit and persist the Phase-D standardization scaler over the kept "
        "frame of the gated catalog (D1, issue #40).",
    )
    p.add_argument("--catalog-dir", type=Path, default=Path("data/catalog_gated"))
    p.add_argument("--out", type=Path, default=SCALER_PATH)
    args = p.parse_args(argv)

    frame = kept_frame(args.catalog_dir)
    scaler = fit_scaler(frame.X, fingerprint=frame.fingerprint)
    path = write_scaler(scaler, args.out)

    print(f"Fitted over {frame.n:,} kept rows ({frame.X.shape[1]} metrics).")
    for j, name in enumerate(METRICS):
        tag = " (log)" if name in LOG_METRICS else ""
        print(f"  {name + tag:<28} mean {scaler.mean[j]:>10.4f}   sd {scaler.std[j]:>10.4f}")
    print(
        f"  tags carried, not fitted-around: water_affected "
        f"{int(frame.water_affected.sum()):,}, coastline {int(frame.coastline.sum()):,}"
    )
    print(f"  scaler: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
