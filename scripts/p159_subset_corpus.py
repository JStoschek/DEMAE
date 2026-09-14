"""Carve the tiles one benchmark cell reads out of `corpus_500k`, for a rented box (#159).

`python -m benchmark.launch run` reads **27,000** of the corpus's 500,000 tiles — §3.1's 20,000
head-train, 2,000 val and 5,000 test draws — and nothing else. The shards holding those 27,000
tiles are 2.7 GB; the corpus is 47 GB. On a 2.8 MB/s uplink that is the difference between
**eighteen minutes and four and three-quarter hours** of a rented box's clock, which is the whole
reason this script exists.

## What it does not do

It does not make a smaller *corpus*. It makes the **same corpus with most of its pixels absent**,
which is a different and much safer thing:

* **The manifest is copied whole.** All 500,000 rows, every column. That matters more than it
  sounds: §5's ruggedness bands are quartiles of ``slope_mean`` over the eligible **test pool**
  (:meth:`benchmark.voids.Bands.from_manifest`), and that pool is read off the manifest rather
  than off the tiles. A manifest holding only the 5,000 drawn test tiles would cut its quartiles
  over a twentieth of the pool and hand every scored instance a band index that means something
  else. Copying the manifest makes the bands **identical by construction** rather than
  approximately right.
* **Only ``shard_id`` and ``row_index`` move**, and only for the tiles that were carried. Those
  two columns are the corpus's own address of a tile's pixels, so re-packing the pixels means
  re-addressing them and nothing else.
* **Every tile that was not carried is addressed to shard −1**, which does not exist. So a run
  that reaches for one gets a `FileNotFoundError` naming the shard, rather than the pixels of
  whichever tile happens to sit at that row of a shard that does exist. A subset that fails
  loudly on a tile it does not have is a subset; one that returns the wrong tile is a corrupted
  corpus wearing the right md5s.

## What it checks before it claims to have worked

Three things, because a wrong subset produces a full table of plausible numbers and nothing
raises:

1. **Byte-equality on a sample.** N carried tiles are read back through the subset's own
   addresses and compared to the source corpus's pixels for the same ``tile_id``.
2. **The bands are the same.** :meth:`~benchmark.voids.Bands.from_manifest` is run against both
   corpora and the edges must match exactly.
3. **Every tile the committed artifacts name is carried.** The tile lists and the eval mask
   list's hosts, checked against what was written.

Not a way to produce a result, and not part of any arm: this is an ops script, like
`scripts/p127_fetch_srtm_voids.py`. Nothing in `src/benchmark` imports it, and a cell scored
against its output is answerable to the same manifest and the same committed artifacts as one
scored against the full corpus.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from benchmark.artifacts import EVAL_MASKS_NAME, TILE_LIST_NAMES, TEST_TILES_NAME, VAL_TILES_NAME
from benchmark.voids import Bands
from dataset import config as dataset_config
from dataset.shards import ShardWriter, shard_path
from pretrain.artifact import read_tile_strata

#: The address a tile that was not carried is given. Not a real shard, so a read raises.
ABSENT_SHARD: int = -1

#: How many carried tiles are read back and compared byte-for-byte. 200 over 27,000 is a sample
#: rather than a proof, and it is aimed at the failure that is actually plausible here — an
#: off-by-one in the re-addressing, which would show up in the first few — rather than at random
#: corruption, which the md5s of everything else already cover.
VERIFY_SAMPLE: int = 200


def tiles_a_run_reads(artifacts_dir: Path) -> dict[str, list[str]]:
    """Every tile id `run` can reach, by which committed draw names it.

    All three label-efficiency rungs, because they are nested prefixes of one draw (§3.1) and a
    subset carrying only the 20k list would still be carrying the 1k and 5k ones. The eval mask
    list is read too and its tiles asserted to be a subset of the test list — §3.2 draws one mask
    per test tile and §3.3 transplants onto hosts drawn from that same list, so a host outside it
    is a `freeze` whose two halves disagree.
    """
    draws: dict[str, list[str]] = {}
    for size, name in TILE_LIST_NAMES.items():
        path = artifacts_dir / name
        if path.is_file():
            draws[f"head_train:{size}"] = [
                t for t, _ in read_tile_strata(path, what=f"head-train {size} list")
            ]
    for key, name in (("val", VAL_TILES_NAME), ("test", TEST_TILES_NAME)):
        draws[key] = [t for t, _ in read_tile_strata(artifacts_dir / name, what=f"{key} list")]

    masks = artifacts_dir / EVAL_MASKS_NAME
    if masks.is_file():
        hosts = set(pq.read_table(masks, columns=["tile_id"]).column("tile_id").to_pylist())
        stray = sorted(hosts - set(draws["test"]))
        if stray:
            raise SystemExit(
                f"{EVAL_MASKS_NAME} names {len(stray)} tile(s) the committed {TEST_TILES_NAME} "
                f"does not ({stray[:3]}). Re-freeze: the two artifacts are one draw."
            )
    if not draws:
        raise SystemExit(
            f"no committed tile lists in {artifacts_dir}. Run `python -m benchmark.launch "
            f"freeze` first — this script carries what that draw named."
        )
    return draws


def carve(source: Path, out: Path, keep: list[str], *, rows_per_shard: int) -> dict:
    """Write the subset. Returns what it wrote, for the caller to print and check."""
    started = time.perf_counter()
    table = pq.read_table(source / "manifest.parquet")
    ids = table.column("tile_id").to_pylist()
    row_of = {tile_id: i for i, tile_id in enumerate(ids)}

    missing = sorted(set(keep) - set(row_of))
    if missing:
        raise SystemExit(
            f"{len(missing)} tile(s) named by the committed artifacts are not in "
            f"{source}/manifest.parquet ({missing[:3]}). The artifacts and this corpus are not "
            f"the same draw."
        )

    shard_ids = np.asarray(table.column("shard_id").to_pylist(), dtype=np.int64)
    row_index = np.asarray(table.column("row_index").to_pylist(), dtype=np.int64)
    new_shard = np.full(len(ids), ABSENT_SHARD, dtype=np.int64)
    new_row = np.zeros(len(ids), dtype=np.int64)

    # Carried in the source's own (shard, row) order, so the reads are sequential over each
    # source shard's memmap rather than scattered across 34 of them 27,000 times.
    ordered = sorted(keep, key=lambda t: (shard_ids[row_of[t]], row_index[row_of[t]]))
    handles: dict[int, np.memmap] = {}
    out.mkdir(parents=True, exist_ok=True)
    with ShardWriter(out / "shards", rows_per_shard=rows_per_shard) as writer:
        for done, tile_id in enumerate(ordered, start=1):
            i = row_of[tile_id]
            shard = int(shard_ids[i])
            handle = handles.get(shard)
            if handle is None:
                handle = handles[shard] = np.load(
                    shard_path(source / "shards", shard), mmap_mode="r"
                )
            ref = writer.add(np.asarray(handle[int(row_index[i])]))
            new_shard[i] = ref.shard_id
            new_row[i] = ref.row_index
            if done % 2000 == 0:
                print(f"  carried {done:,}/{len(ordered):,}", flush=True)

    rewritten = table.set_column(
        table.schema.get_field_index("shard_id"), "shard_id", pa.array(new_shard, pa.int32())
    ).set_column(
        table.schema.get_field_index("row_index"), "row_index", pa.array(new_row, pa.int32())
    )
    pq.write_table(rewritten, out / "manifest.parquet", compression="zstd")
    for name in ("corpus_meta.json", "build_state.json"):
        if (source / name).is_file():
            shutil.copy(source / name, out / name)

    written = sorted((out / "shards").glob("shard_*.npy"))
    return {
        "tiles": len(ordered),
        "manifest_rows": len(ids),
        "shards": len(written),
        "bytes": sum(p.stat().st_size for p in written),
        "seconds": time.perf_counter() - started,
    }


def verify(source: Path, out: Path, keep: list[str], *, sample: int = VERIFY_SAMPLE) -> None:
    """The three checks in the module docstring. Raises rather than warns."""
    from benchmark.evaluate import TileStore

    before = Bands.from_manifest(source)
    after = Bands.from_manifest(out)
    if before.edges != after.edges:
        raise SystemExit(
            f"§5's band edges moved: {before.edges} -> {after.edges}. The manifest is supposed "
            f"to be carried whole precisely so they cannot, so this is a bug in this script and "
            f"not a thing to work around — every scored instance's band would mean something "
            f"different from a full-corpus run's."
        )
    print(
        f"  bands unchanged: {after.count} bands over {after.pool_n:,} eligible test tiles",
        flush=True,
    )

    rng = np.random.default_rng(0)
    checked = [keep[i] for i in rng.choice(len(keep), size=min(sample, len(keep)), replace=False)]
    original, subset = TileStore(source), TileStore(out)
    for tile_id in checked:
        a, b = original.read(tile_id), subset.read(tile_id)
        if not np.array_equal(a.raw_i16, b.raw_i16):
            raise SystemExit(f"{tile_id} does not read back byte-identically from the subset")
        if (a.lat, a.slope_mean) != (b.lat, b.slope_mean):
            raise SystemExit(f"{tile_id}'s manifest columns moved between the two corpora")
    print(f"  {len(checked)} tiles read back byte-identical", flush=True)

    absent = next((t for t in set(pq.read_table(
        out / "manifest.parquet", columns=["tile_id"]
    ).column("tile_id").to_pylist()) - set(keep)), None)
    if absent is not None:
        try:
            subset.read(absent)
        except (FileNotFoundError, OSError, ValueError):
            print("  a tile that was not carried raises rather than reading someone else's",
                  flush=True)
        else:
            raise SystemExit(
                f"{absent} was not carried and read anyway — the absent rows are supposed to "
                f"address shard {ABSENT_SHARD}, which does not exist"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python scripts/p159_subset_corpus.py",
        description=__doc__.split("\n\n")[0],
    )
    parser.add_argument("--corpus", required=True, help="the full corpus to carve from")
    parser.add_argument("--out", required=True, help="where the subset goes")
    parser.add_argument(
        "--artifacts-dir", default="artifacts/task1", help="the committed tile lists to read"
    )
    parser.add_argument(
        "--rows-per-shard",
        type=int,
        default=dataset_config.ROWS_PER_SHARD,
        help="tiles per shard; the default is the corpus's own",
    )
    args = parser.parse_args(argv)

    source, out = Path(args.corpus), Path(args.out)
    draws = tiles_a_run_reads(Path(args.artifacts_dir))
    keep = sorted({tile_id for ids in draws.values() for tile_id in ids})
    for name, ids in sorted(draws.items()):
        print(f"{name:<18} {len(ids):,}")
    print(f"{'carrying':<18} {len(keep):,} distinct tiles from {source}")

    wrote = carve(source, out, keep, rows_per_shard=args.rows_per_shard)
    print(
        f"{'wrote':<18} {wrote['tiles']:,} tiles in {wrote['shards']} shards "
        f"({wrote['bytes'] / 1e9:.2f} GB) in {wrote['seconds']:.0f}s"
    )
    print(f"{'manifest':<18} {wrote['manifest_rows']:,} rows, carried whole")
    verify(source, out, keep)
    print(f"{'ok':<18} {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
