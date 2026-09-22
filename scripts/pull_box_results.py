#!/usr/bin/env python3
"""Bring a rented box's scored rows home, and prove they arrived — run this on the **laptop**.

    ./scripts/pull_box_results.py --host root@1.2.3.4 --port 55644          # report only
    ./scripts/pull_box_results.py --host root@1.2.3.4 --port 55644 --apply  # fold them in

`p159_deploy.sh` pushes code to a box and nothing brings results back, so every rented run has
ended with someone remembering to scp a parquet. #159 was closed with its A6 rows still on a box
that was later destroyed, and #168's A9 rows sat on one for 12.9 hours after the pass finished.
`benchmark.report` reads `data/task1/results.parquet` and nothing else, so a row that is not in
that file does not exist as far as the table is concerned.

**Report-only by default**, on `teardown_box.sh`'s rule: the irreversible half needs the
deliberate flag. `--apply` is the one that writes.

What it refuses rather than guesses:

* a schema that differs from the local table's, in columns or order;
* a cell — one `(arm, size, seed)` — present on both sides under *different* `run_id`s, which
  means the box re-scored something the laptop already holds and "fold" has two meanings;
* a box row whose `config_hash` disagrees with the local rows of the same cell.

Cells already local under the same `run_id` are skipped, so running it twice folds nothing the
second time. That matters more than it sounds: `benchmark.results.append_rows` replaces by cell,
but this script appends, and an append run twice is a table with every row doubled.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOCAL_RESULTS = REPO / "data" / "task1" / "results.parquet"
LOCAL_RECORDS = REPO / "data" / "task1" / "results-runs.jsonl"
CELL = ["arm", "size", "seed"]


def scp(host: str, port: str, remote: str, dest: Path) -> bool:
    out = subprocess.run(
        ["scp", "-q", "-P", port, "-o", "BatchMode=yes", f"{host}:{remote}", str(dest)],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        print(f"   could not fetch {remote}: {out.stderr.strip()}", file=sys.stderr)
    return out.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", required=True, help="user@host of the box")
    ap.add_argument("--port", default="22")
    ap.add_argument("--remote-root", default="/root/DEMAE")
    ap.add_argument("--apply", action="store_true", help="write; without it, report only")
    args = ap.parse_args()

    import pandas as pd

    work = Path(tempfile.mkdtemp())
    try:
        box_results = work / "results.parquet"
        if not scp(args.host, args.port, f"{args.remote_root}/data/task1/results.parquet", box_results):
            return 2
        box_records = work / "results-runs.jsonl"
        have_records = scp(
            args.host, args.port, f"{args.remote_root}/data/task1/results-runs.jsonl", box_records
        )

        box = pd.read_parquet(box_results)
        if not LOCAL_RESULTS.is_file():
            print(f"no local table at {LOCAL_RESULTS}; the box's {len(box)} rows would be all of it")
            local = box.iloc[0:0]
        else:
            local = pd.read_parquet(LOCAL_RESULTS)

        if list(box.columns) != list(local.columns) and len(local):
            print("REFUSING: the box's schema is not the local table's", file=sys.stderr)
            print(f"  box   {list(box.columns)}", file=sys.stderr)
            print(f"  local {list(local.columns)}", file=sys.stderr)
            return 1

        box_cells = box.groupby(CELL, dropna=False)
        local_cells = {k: v for k, v in local.groupby(CELL, dropna=False)} if len(local) else {}

        new, skipped, conflicts = [], [], []
        for key, rows in box_cells:
            here = local_cells.get(key)
            if here is None:
                new.append((key, rows))
                continue
            box_runs, local_runs = set(rows["run_id"]), set(here["run_id"])
            if box_runs == local_runs:
                skipped.append((key, len(here)))
            else:
                conflicts.append((key, sorted(local_runs), sorted(box_runs)))

        print(f"box {len(box)} rows / {box_cells.ngroups} cells   local {len(local)} rows")
        for key, n in skipped:
            print(f"  already here   {key}  ({n} rows, same run_id)")
        for key, lr, br in conflicts:
            print(f"  CONFLICT       {key}  local {lr} vs box {br}", file=sys.stderr)
        for key, rows in new:
            print(f"  to fold        {key}  ({len(rows)} rows, run {sorted(set(rows['run_id']))})")

        if conflicts:
            print(
                "\nREFUSING: a cell is scored on both sides under different run_ids. The box "
                "re-scored something this table already holds, so 'fold' would mean either "
                "'append a duplicate' or 'replace' and this script will not pick. Decide which "
                "row is the one, and use `benchmark.results.append_rows` to replace it.",
                file=sys.stderr,
            )
            return 1
        if not new:
            print("\nnothing to fold — the box holds no cell this table is missing")
            return 0
        if not args.apply:
            total = sum(len(r) for _, r in new)
            print(f"\nreport only: {total} rows in {len(new)} cell(s) would be folded. Re-run with --apply")
            return 0

        backup = LOCAL_RESULTS.with_suffix(".parquet.before-pull")
        shutil.copy2(LOCAL_RESULTS, backup) if LOCAL_RESULTS.is_file() else None
        merged = pd.concat([local, *[r for _, r in new]], ignore_index=True)
        # Through Arrow under the results schema, not `to_parquet`: pandas types every string
        # column `large_string` and stamps its own metadata, and `benchmark.results.append_rows`
        # compares schemas — a file written that way refused every later laptop `run` (#173).
        import pyarrow as pa
        import pyarrow.parquet as pq

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from benchmark import results as results_module

        metrics = [c for c in merged.columns if c not in results_module.columns([])]
        table = pa.Table.from_pandas(merged, preserve_index=False).cast(
            results_module.schema(metrics)
        ).replace_schema_metadata(None)
        pq.write_table(table, LOCAL_RESULTS, compression="zstd")
        folded = sum(len(r) for _, r in new)
        print(f"\nwrote {len(merged)} rows to {LOCAL_RESULTS} (+{folded}); previous at {backup.name}")

        if have_records:
            wanted = {rid for _, rows in new for rid in set(rows["run_id"])}
            existing = LOCAL_RECORDS.read_text() if LOCAL_RECORDS.is_file() else ""
            added = 0
            with LOCAL_RECORDS.open("a") as fh:
                for line in box_records.read_text().splitlines():
                    if any(rid in line for rid in wanted) and line not in existing:
                        fh.write(line + "\n")
                        added += 1
            print(f"appended {added} run record(s) to {LOCAL_RECORDS.name}")
        else:
            print("no run records fetched — the rows are here, their provenance is not", file=sys.stderr)
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
