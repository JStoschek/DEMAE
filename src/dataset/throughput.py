"""Throughput measurement + the data-driven cloud-burst decision (issue #7).

The pipeline runs **locally** at zero compute cost unless measured I/O is bad; the
lever, per recon §5, is that the bucket is in ``eu-central-1`` while the dev box
is in the US, so a same-region cloud burst slashes COG-open latency. This module
graduates the map's "cloud-burst trigger" fog from reasoning to a *number*: run a
small smoke sample, measure per-tile read+process wall-clock, extrapolate to the
full ADR-0001 workload (~26k overview reads + ~50k full-res reads), and compare a
parallel projection against a wall-clock budget.

The decision is transparent and reproducible: the report carries every measured
rate and every projection, so the verdict can be audited and the budget/worker
assumptions re-run without re-measuring.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import config
from .pipeline import run_pipeline
from .tiling import CogReader

# Full Phase-0 workload (ADR-0001 Consequences; recon §1).
FULL_N_COGS = 26_454
FULL_N_TILES = config.DEBUG_CORPUS_TILES  # 50k debug tier

# Defaults that *define* the graduated cloud-burst trigger. A build projected to
# exceed BUDGET_HOURS of wall-clock, assuming ASSUMED_WORKERS parallel COG opens,
# moves off-local. Both are explicit knobs, not magic — override per run.
BUDGET_HOURS = 6.0
ASSUMED_WORKERS = 8


@dataclass
class ThroughputReport:
    """Measured rates, projections to the full corpus, and the verdict."""

    n_cogs_scored: int
    n_tiles_measured: int
    # measured per-unit rates (seconds)
    per_cog_overview_s: float
    per_cog_score_s: float
    per_window_read_s: float
    per_window_process_s: float
    # projections to the full workload
    projected_scoring_s: float
    projected_materialize_s: float
    projected_total_single_thread_s: float
    assumed_workers: int
    projected_total_parallel_s: float
    effective_per_tile_s: float
    # decision
    budget_s: float
    recommendation: str  # "local" | "cloud-burst"
    rationale: str


def measure_throughput(
    reader: CogReader,
    out_dir: Path | str,
    *,
    sample_cogs: int = 8,
    per_stratum_quota: int = 8,
    workers: int = ASSUMED_WORKERS,
    budget_hours: float = BUDGET_HOURS,
    full_n_cogs: int = FULL_N_COGS,
    full_n_tiles: int = FULL_N_TILES,
) -> ThroughputReport:
    """Run a smoke build, measure, project, and decide local-vs-cloud-burst.

    ``reader`` is injected so this measures the real bucket in production yet runs
    deterministically against a :class:`~dataset.tiling.SyntheticCogReader` in
    tests. ``sample_cogs`` bounds the smoke tier; the projection scales the
    measured per-unit costs up to ``full_n_cogs`` / ``full_n_tiles``.
    """
    res = run_pipeline(
        reader,
        out_dir,
        per_stratum_quota=per_stratum_quota,
        limit_cogs=sample_cogs,
        snapshot_date="smoke",
        code_version="smoke",
    )
    t = res.timings
    if t.n_windows_read == 0:
        raise RuntimeError(
            "smoke sample materialized no tiles; increase sample_cogs / quota"
        )

    per_cog_score_s = t.score_s / t.n_cogs_scored if t.n_cogs_scored else 0.0

    projected_scoring_s = full_n_cogs * (t.per_cog_overview_s + per_cog_score_s)
    projected_materialize_s = full_n_tiles * (t.per_window_read_s + t.per_window_process_s)
    total_single = projected_scoring_s + projected_materialize_s
    workers = max(1, workers)
    total_parallel = total_single / workers
    effective_per_tile_s = total_parallel / full_n_tiles if full_n_tiles else 0.0

    budget_s = budget_hours * 3600.0
    if total_parallel > budget_s:
        recommendation = "cloud-burst"
        rationale = (
            f"projected local build ~{total_parallel / 3600:.1f} h "
            f"(@ {workers} workers) exceeds the {budget_hours:.1f} h budget; "
            f"burst to eu-central-1 to cut COG-open latency"
        )
    else:
        recommendation = "local"
        rationale = (
            f"projected local build ~{total_parallel / 3600:.1f} h "
            f"(@ {workers} workers) is within the {budget_hours:.1f} h budget; "
            f"run the ETL locally at zero compute cost"
        )

    return ThroughputReport(
        n_cogs_scored=t.n_cogs_scored,
        n_tiles_measured=t.n_windows_read,
        per_cog_overview_s=t.per_cog_overview_s,
        per_cog_score_s=per_cog_score_s,
        per_window_read_s=t.per_window_read_s,
        per_window_process_s=t.per_window_process_s,
        projected_scoring_s=projected_scoring_s,
        projected_materialize_s=projected_materialize_s,
        projected_total_single_thread_s=total_single,
        assumed_workers=workers,
        projected_total_parallel_s=total_parallel,
        effective_per_tile_s=effective_per_tile_s,
        budget_s=budget_s,
        recommendation=recommendation,
        rationale=rationale,
    )


def format_report(r: ThroughputReport) -> str:
    """Render a report as a human-readable block for logs / the ticket."""
    return "\n".join(
        [
            "=== DEMAE ETL throughput measurement ===",
            f"smoke sample:        {r.n_cogs_scored} COGs, {r.n_tiles_measured} tiles materialized",
            f"per-COG overview:    {r.per_cog_overview_s * 1e3:.1f} ms read + "
            f"{r.per_cog_score_s * 1e3:.1f} ms score",
            f"per-tile full-res:   {r.per_window_read_s * 1e3:.1f} ms read + "
            f"{r.per_window_process_s * 1e3:.1f} ms process",
            f"projected scoring:   {r.projected_scoring_s / 3600:.2f} h "
            f"({FULL_N_COGS} overview reads)",
            f"projected materialize:{r.projected_materialize_s / 3600:.2f} h "
            f"({FULL_N_TILES} full-res reads)",
            f"projected total:     {r.projected_total_single_thread_s / 3600:.2f} h "
            f"single-thread -> {r.projected_total_parallel_s / 3600:.2f} h "
            f"@ {r.assumed_workers} workers",
            f"effective per-tile:  {r.effective_per_tile_s * 1e3:.1f} ms",
            f"budget:              {r.budget_s / 3600:.1f} h",
            f">>> RECOMMENDATION:  {r.recommendation.upper()} — {r.rationale}",
        ]
    )


def main() -> None:  # pragma: no cover - live-bucket entry point
    """Measure against the live Copernicus bucket and print the verdict."""
    import tempfile

    from .tiling import RasterioCogReader

    with tempfile.TemporaryDirectory() as tmp:
        report = measure_throughput(RasterioCogReader(), tmp)
    print(format_report(report))


if __name__ == "__main__":  # pragma: no cover
    main()
