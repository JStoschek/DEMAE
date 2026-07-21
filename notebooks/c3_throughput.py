"""C3 (#38) — times the global scoring pass per executor and worker count.

Question this answers
---------------------
C1 (#33) projected the global pass at 4.19 h by measuring a **serial** per-COG cost
and dividing by 16 workers. The 300-COG dry run came in at 1.87 s/COG with 16
threads — ~13.7 h corpus-wide. So: which executor and how many workers actually make
the pass practical, and what is the *measured* rate?

What it measures
----------------
The same band-spread COG draw scored repeatedly, once per ``(executor, workers)``
config, each into a throwaway output directory (so no run resumes another's ledger).
Per config: wall-clock, rows, s/COG, and the corpus projection at 26,450 COGs. The
draw is a prefix of the production ``--sample-cogs`` draw at the same seed, and
:func:`dataset.sampling.sample_cogs` guarantees any prefix of it is itself a spread
sample — so the rate is proportional to the corpus, not to whatever is cheap.

Two things it deliberately does *not* do:

- **No warm-up discount.** Every config re-streams every COG from eu-central-1, so
  all of them carry the same US-latency penalty and the comparison is fair. The
  masks come from the shared local cache (``data/mask_cache``), as they will in the
  real pass.
- **No arithmetic parallelism claim.** Wall-clock is measured end to end per config.
  The whole point of this ticket is that dividing a serial number by a worker count
  is how the projection went wrong.

Needs the live bucket (it is a transport measurement), so it is not part of the test
suite. Re-run it on any box before trusting this repo's worker defaults there — the
knee moves with the read latency you have to hide.

Run:  uv run python notebooks/c3_throughput.py
      N_COGS=24 CONFIGS=process:4,process:6,thread:16 uv run python notebooks/...
Writes: docs/research/c3-throughput.json — the artifact behind the table in
        docs/research/c3-scoring-throughput.md.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from dataset import config
from dataset.catalog import scan
from dataset.geometry import parse_cog_key
from dataset.masks import RasterioMaskReader
from dataset.sampling import sample_cogs
from dataset.tiling import RasterioCogReader

CORPUS_COGS = 26_450  # the full frame (C1's projection base)
RESEARCH = Path("docs/research")
SCRATCH = Path("data/_c3_throughput_scratch")  # under data/, which is gitignored

# (executor, workers), all in one sweep so every number in the write-up comes from
# one draw on one machine state. Threads are included, and so is a single process:
# the ceiling and the serial baseline are both re-measured here rather than carried
# over from a microbenchmark.
DEFAULT_CONFIGS = (
    ("thread", 4),
    ("thread", 8),
    ("thread", 16),
    ("process", 1),
    ("process", 4),
    ("process", 6),
    ("process", 8),
    ("process", 12),
    ("process", 16),
    ("process", 20),
)


def _configs() -> list[tuple[str, int]]:
    raw = os.environ.get("CONFIGS")
    if not raw:
        return list(DEFAULT_CONFIGS)
    out = []
    for item in raw.split(","):
        kind, _, workers = item.partition(":")
        out.append((kind.strip(), int(workers)))
    return out


def _draw(n: int):
    frame = [parse_cog_key(s) for s in json.loads(Path("data/cog_frame.json").read_text())]
    return sample_cogs(frame, n, allocation="proportional", seed=config.RANDOM_SEED)


def main() -> int:
    n_cogs = int(os.environ.get("N_COGS", "24"))
    cogs = _draw(n_cogs)
    print(f"drawn {len(cogs)} COGs (spread, seed {config.RANDOM_SEED})")

    results = []
    for kind, workers in _configs():
        out_dir = SCRATCH / f"{kind}-{workers}"
        shutil.rmtree(out_dir, ignore_errors=True)
        t0 = time.perf_counter()
        res = scan(
            RasterioCogReader(),
            RasterioMaskReader(),
            cogs,
            out_dir,
            workers=workers,
            executor=kind,  # type: ignore[arg-type]
            rows_per_part=10_000_000,  # one flush; part-file I/O is not what we time
        )
        elapsed = time.perf_counter() - t0
        scored = res.n_cogs
        per_cog = elapsed / max(1, scored)
        rec = {
            "executor": kind,
            "workers": workers,
            "n_cogs": scored,
            "n_anomalies": res.n_anomalies,
            "n_rows": res.n_rows,
            "wall_s": round(elapsed, 2),
            "per_cog_s": round(per_cog, 3),
            "corpus_hours": round(per_cog * CORPUS_COGS / 3600, 2),
        }
        results.append(rec)
        print(
            f"  {kind:>7} x{workers:<3} {elapsed:7.1f}s  {per_cog:5.2f} s/COG  "
            f"-> {rec['corpus_hours']:5.2f} h  ({res.n_rows} rows, "
            f"{res.n_anomalies} anomalies)"
        )

    # Speedups are quoted against the *serial* run when it is in the sweep — that is
    # the only baseline a parallel claim means anything against, and measuring it
    # rather than assuming it is the whole point of #38.
    baseline = next(
        (r for r in results if (r["executor"], r["workers"]) == ("process", 1)), results[0]
    )
    for r in results:
        r["speedup_vs_baseline"] = round(baseline["per_cog_s"] / r["per_cog_s"], 2)

    RESEARCH.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_cogs_per_config": len(cogs),
        "seed": config.RANDOM_SEED,
        "cpu_count": os.cpu_count(),
        "corpus_cogs": CORPUS_COGS,
        "baseline": f"{baseline['executor']}x{baseline['workers']}",
        "results": results,
    }
    out = RESEARCH / "c3-throughput.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\n[wrote {out}]")
    shutil.rmtree(SCRATCH, ignore_errors=True)
    return 0


if __name__ == "__main__":  # spawn-safe: workers re-import this module
    raise SystemExit(main())
