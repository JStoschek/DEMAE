"""Corpus build entry point — the reproducible "run the pipeline" artifact (#8).

Wraps :func:`dataset.pipeline.run_pipeline` with the two things a real build run
needs on top of the pure orchestrator:

1. **A smoke gate.** Before committing to the full ~50k debug build, a tiny tier
   (a couple of COGs, quota 1) is run end-to-end into its own directory. If it
   materializes zero tiles the build aborts with :class:`SmokeValidationError` —
   catching a broken reader/credentials/recipe in seconds instead of hours.
2. **A source switch.** ``--source live`` reads the real Copernicus bucket
   (:class:`~dataset.tiling.RasterioCogReader`); ``--source synthetic`` builds the
   deterministic zero-network sample (:mod:`dataset.synthetic`) so a fresh reader
   can reproduce a sample and the viz notebook has a corpus to render.

Run it with ``python -m dataset.build`` (see :mod:`dataset.__main__`). Provenance
(git SHA -> ``code_version``, an ISO snapshot date, a UTC build timestamp) is
stamped into the ADR-0002 §3 ``corpus_meta.json`` fingerprint so every build is
traceable; the recipe knobs forwarded to :func:`run_pipeline` are the ADR-0001
§4-6 quota / cap / clip parameters.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

from . import config
from .pipeline import PipelineResult, ProgressFactory, run_pipeline
from .tiling import CogReader

log = logging.getLogger(__name__)


def tqdm_progress() -> ProgressFactory | None:
    """A tqdm-backed progress factory for the long stages, or ``None`` if tqdm is
    absent (the pipeline then runs silently — the bar is cosmetic, never required)."""
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover - tqdm is a declared dependency
        return None

    def factory(total: int, desc: str) -> object:
        return tqdm(total=total, desc=desc, unit="it", dynamic_ncols=True, smoothing=0.02)

    return factory


class SmokeValidationError(RuntimeError):
    """Raised when the smoke tier materializes no tiles — the full build is aborted."""


@dataclass
class BuildOutcome:
    """The result of a (optionally smoke-gated) corpus build."""

    corpus: PipelineResult
    smoke: PipelineResult | None = None


def build_corpus(
    reader: CogReader,
    out_dir: Path | str,
    *,
    per_stratum_quota: int = config.PER_STRATUM_QUOTA,
    per_tile_cap: int = config.PER_TILE_WINDOW_CAP,
    limit_cogs: int | None = None,
    seed: int = config.RANDOM_SEED,
    snapshot_date: str = "unknown",
    code_version: str = "unknown",
    build_timestamp: str | None = None,
    run_smoke: bool = True,
    smoke_cogs: int = 4,
    smoke_quota: int = 2,
    progress: ProgressFactory | None = None,
    workers: int = 8,
) -> BuildOutcome:
    """Build a corpus into ``out_dir``, gating on a smoke tier first.

    The smoke tier is written to ``out_dir/smoke`` and validated (``n_written >
    0``) before the full build lands in ``out_dir``. Pass ``run_smoke=False`` to
    skip the gate. ``progress`` is a bar factory for the (long) full build only —
    the smoke tier runs in seconds and stays quiet. ``workers`` fans the full
    build's COG reads across a thread pool (the smoke tier stays sequential). All
    other recipe knobs mirror :func:`run_pipeline`.
    """
    out_dir = Path(out_dir)

    smoke: PipelineResult | None = None
    if run_smoke:
        log.info("smoke tier: %d COGs, quota %d ...", smoke_cogs, smoke_quota)
        smoke = run_pipeline(
            reader,
            out_dir / "smoke",
            per_stratum_quota=smoke_quota,
            per_tile_cap=per_tile_cap,
            limit_cogs=smoke_cogs,
            seed=seed,
            snapshot_date=snapshot_date,
            code_version=code_version,
            build_timestamp=build_timestamp,
        )
        if smoke.n_written == 0:
            raise SmokeValidationError(
                f"smoke tier wrote 0 tiles from {smoke.n_candidates} candidates "
                f"(drops={smoke.drop_reasons}); aborting before the full build"
            )
        log.info(
            "smoke OK: %d tiles from %d candidates -> full build",
            smoke.n_written,
            smoke.n_candidates,
        )

    corpus = run_pipeline(
        reader,
        out_dir,
        per_stratum_quota=per_stratum_quota,
        per_tile_cap=per_tile_cap,
        limit_cogs=limit_cogs,
        seed=seed,
        snapshot_date=snapshot_date,
        code_version=code_version,
        build_timestamp=build_timestamp,
        progress=progress,
        workers=workers,
    )
    log.info(
        "corpus built: %d tiles (%d candidates, %d selected) -> %s",
        corpus.n_written,
        corpus.n_candidates,
        corpus.n_selected,
        out_dir,
    )
    return BuildOutcome(corpus=corpus, smoke=smoke)


# --- provenance ---------------------------------------------------------------


def git_sha() -> str:
    """Short git SHA of the working tree (``-dirty`` if uncommitted), or unknown."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True
    ).stdout.strip()
    return f"{sha}-dirty" if dirty else sha


def _today() -> str:
    from datetime import date

    return date.today().isoformat()


# --- CLI ----------------------------------------------------------------------


def _make_reader(source: str) -> CogReader:
    if source == "synthetic":
        from .synthetic import synthetic_reader

        return synthetic_reader()
    from .tiling import RasterioCogReader

    return RasterioCogReader()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m dataset.build",
        description="Build the DEMAE Phase-0 elevation corpus (smoke-gated).",
    )
    p.add_argument(
        "--source",
        choices=["live", "synthetic"],
        default="live",
        help="live Copernicus bucket, or the zero-network synthetic sample",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="output corpus dir (default: data/corpus, or data/sample for synthetic)",
    )
    p.add_argument(
        "--per-stratum-quota",
        type=int,
        default=None,
        help=f"tiles per stratum (default: {config.PER_STRATUM_QUOTA} live -> "
        "~50k; 40 synthetic -> a few-hundred-tile sample)",
    )
    p.add_argument("--limit-cogs", type=int, default=None, help="cap COGs scanned")
    p.add_argument(
        "--workers",
        type=int,
        default=8,
        help="thread-pool size for the I/O-bound COG reads (1 = sequential); the "
        "corpus is identical regardless (default: 8, matching the throughput probe)",
    )
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    p.add_argument("--snapshot-date", default=None, help="corpus snapshot date")
    p.add_argument("--no-smoke", action="store_true", help="skip the smoke gate")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = build_parser().parse_args(argv)

    synthetic = args.source == "synthetic"
    out_dir = args.out_dir or Path("data/sample" if synthetic else "data/corpus")
    quota = args.per_stratum_quota
    if quota is None:
        quota = 40 if synthetic else config.PER_STRATUM_QUOTA

    outcome = build_corpus(
        _make_reader(args.source),
        out_dir,
        per_stratum_quota=quota,
        limit_cogs=args.limit_cogs,
        seed=args.seed,
        snapshot_date=args.snapshot_date or ("synthetic" if synthetic else _today()),
        code_version=git_sha(),
        run_smoke=not args.no_smoke,
        progress=tqdm_progress(),
        workers=args.workers,
    )

    r = outcome.corpus
    print(
        f"\nBuilt {r.n_written} tiles -> {out_dir}\n"
        f"  candidates={r.n_candidates} selected={r.n_selected} "
        f"drops={r.drop_reasons}\n"
        f"  dataset_version={r.dataset_version}\n"
        f"  manifest={r.manifest_path}\n"
        f"  meta={r.corpus_meta_path}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
