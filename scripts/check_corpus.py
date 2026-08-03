#!/usr/bin/env python3
"""Is this a *complete* corpus? — one definition, checked against a local directory or a bucket.

`rclone copy --checksum` answers a different and narrower question: did the bytes arrive
faithfully. It compares content hashes and re-transfers anything that disagrees, which is
strictly better than the size comparison this script's ancestor did by hand. What it cannot
tell you is whether the *source* was whole. A bucket missing `shard_00007` copies cleanly,
reports success, and leaves a box training on a corpus that is not the one `dataset_version`
names — the ADR-0002 manifest still lists 499,934 tiles, and the loader still opens.

So the transport verifies the transfer and this verifies the corpus. Two questions, two tools.

Run it against the bucket **before renting a box**, when an incomplete upload costs nothing but
your patience:

    python3 scripts/check_corpus.py --bucket demae-corpus --prefix corpus_500k/ \
        --endpoint-url https://s3.us-west-004.backblazeb2.com

and against the local copy **after fetching**, which is what `provision_box.sh` does:

    python3 scripts/check_corpus.py --local data/corpus_500k

One implementation of "what a complete corpus looks like", pointed at either end. The bucket
mode needs boto3 (a core dependency); the local mode imports nothing beyond the stdlib, so it
runs on a freshly rented box before `uv sync` has built anything.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: The shipped 500k tier (#72) — plan §15 item 1's "dataset_version verified".
DEFAULT_EXPECT_VERSION = "bcbd6484d8116199c5c1160e21ea0284"

#: Only ever a *signing* region for an S3-compatible store; see :func:`signing_region`.
DEFAULT_REGION = "us-east-1"

#: Mirrors `data/corpus_500k/` (ADR-0002). `build_state.json` is the resumable-build ledger:
#: not read at training time, but its absence means the upload copied a subset of the tree.
CORPUS_META_FILE = "corpus_meta.json"
SIDECAR_FILES = (CORPUS_META_FILE, "manifest.parquet", "build_state.json")
SHARD_RE = re.compile(r"^shards/shard_(\d{5})\.npy$")

#: ADR-0002's storage vocabulary, as bytes per sample. Spelled out rather than imported from
#: numpy so that the local mode is stdlib-only and runs on a box before `uv sync` has built
#: anything — which is exactly when a bad corpus is cheapest to discover.
DTYPE_BYTES = {"int8": 1, "uint8": 1, "int16": 2, "uint16": 2, "float16": 2, "float32": 4}

#: Each shard is a .npy, so it carries a header ahead of its payload — 128 bytes for every
#: shard in the shipped corpus, but the format pads to a boundary and a future writer could
#: land elsewhere. The byte-total check below is one-sided with this much slack per shard: far
#: too little to hide a missing shard (~1.4 GB), far more than any header will ever need.
NPY_HEADER_SLACK = 4096


@dataclass
class Inventory:
    """What is actually there — the same shape whether it came off a disk or a bucket."""

    where: str
    sizes: dict[str, int] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @property
    def dataset_version(self) -> str | None:
        return self.meta.get("dataset_version")

    @property
    def shards(self) -> list[str]:
        return sorted(k for k in self.sizes if SHARD_RE.match(k))

    @property
    def shard_bytes(self) -> int:
        return sum(self.sizes[k] for k in self.shards)

    @property
    def total_bytes(self) -> int:
        return sum(self.sizes.values())

    def expected_shard_payload(self) -> int | None:
        """Bytes the shards must hold, from the corpus's own record of what it contains.

        `n_tiles x tile_px^2 x itemsize` — derived, never configured, so it cannot drift out of
        agreement with the corpus it describes. This is what catches the incomplete upload that
        index-gap detection structurally cannot see: a transfer interrupted partway leaves a
        *dense* prefix of shards, so the tail simply goes missing with no gap to find.
        """
        try:
            itemsize = DTYPE_BYTES[self.meta["storage_dtype"]]
            return int(self.meta["n_tiles"]) * int(self.meta["tile_px"]) ** 2 * itemsize
        except (KeyError, TypeError, ValueError):
            return None


def inspect_local(directory: Path) -> Inventory:
    inv = Inventory(where=str(directory))
    if not directory.is_dir():
        return inv
    for name in SIDECAR_FILES:
        path = directory / name
        if path.is_file():
            inv.sizes[name] = path.stat().st_size
    shard_dir = directory / "shards"
    if shard_dir.is_dir():
        for path in shard_dir.iterdir():
            rel = f"shards/{path.name}"
            if SHARD_RE.match(rel) and path.is_file():
                inv.sizes[rel] = path.stat().st_size
    meta = directory / CORPUS_META_FILE
    if meta.is_file():
        inv.meta = json.loads(meta.read_text())
    return inv


def signing_region(endpoint_url: str | None, region: str) -> str:
    """R2 has no AWS regions and wants the literal ``auto`` in the SigV4 credential scope.

    Detected rather than left to the caller because getting it wrong produces a
    ``SignatureDoesNotMatch``, which reads like a wrong secret key and sends you looking in
    entirely the wrong place. An explicitly-passed non-default region still wins.
    """
    if endpoint_url and "r2.cloudflarestorage.com" in endpoint_url and region == DEFAULT_REGION:
        return "auto"
    return region


def inspect_bucket(
    bucket: str,
    prefix: str,
    *,
    endpoint_url: str | None,
    region: str,
    profile: str | None,
) -> Inventory:
    import boto3

    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    s3 = session.client(
        "s3", endpoint_url=endpoint_url, region_name=signing_region(endpoint_url, region)
    )

    inv = Inventory(where=f"s3://{bucket}/{prefix}")
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            rel = item["Key"][len(prefix) :]
            if rel and (rel in SIDECAR_FILES or SHARD_RE.match(rel)):
                inv.sizes[rel] = item["Size"]

    # The identity check needs the file's *contents*, so this one small object is fetched
    # rather than listed. Absent is not an error here — `check` reports it as a missing
    # sidecar, which is a clearer message than a traceback from json.loads.
    if CORPUS_META_FILE in inv.sizes:
        body = s3.get_object(Bucket=bucket, Key=prefix + CORPUS_META_FILE)["Body"].read()
        inv.meta = json.loads(body)
    return inv


def check(inv: Inventory, *, expect_version: str | None, expect_shards: int | None) -> list[str]:
    """Every problem, not the first one — a caller should learn about all of them in one pass."""
    problems: list[str] = []

    for name in SIDECAR_FILES:
        if name not in inv.sizes:
            problems.append(f"missing sidecar: {name}")

    shards = inv.shards
    if not shards:
        problems.append("no shards/shard_NNNNN.npy objects at all")
    else:
        # The check that matters, and the one no transfer tool performs. A gap copies
        # faithfully and trains silently on less corpus than the manifest describes.
        indices = sorted(int(SHARD_RE.match(s).group(1)) for s in shards)
        gaps = sorted(set(range(indices[-1] + 1)) - set(indices))
        if gaps:
            shown = ", ".join(f"{g:05d}" for g in gaps[:10])
            problems.append(
                f"shard index gaps ({len(gaps)}): {shown}{' …' if len(gaps) > 10 else ''}"
            )
        if expect_shards is not None and len(shards) != expect_shards:
            problems.append(f"{len(shards)} shards, expected {expect_shards}")

        # The check gap detection cannot close. An upload that stopped partway leaves shards
        # 0..N dense with the tail simply absent, and `dataset_version` still reads correctly
        # because that hash is over the manifest rather than over what is on disk. Only the
        # byte total notices — one missing shard is ~1.4 GB short of what n_tiles demands.
        payload = inv.expected_shard_payload()
        if payload is None:
            problems.append(
                "cannot derive the expected shard payload — corpus_meta.json is missing "
                "n_tiles / tile_px / storage_dtype, or names a dtype outside ADR-0002"
            )
        else:
            slack = NPY_HEADER_SLACK * len(shards)
            if inv.shard_bytes < payload:
                short = payload - inv.shard_bytes
                # Scaled by the mean of the shards actually present rather than by a constant,
                # so the "how much is missing" reads correctly for the debug tier and any
                # future corpus, not only the 500k one this script ships defaults for.
                mean = inv.shard_bytes / len(shards)
                problems.append(
                    f"shards hold {inv.shard_bytes:,} bytes, {short:,} short of the "
                    f"{payload:,} that {inv.meta.get('n_tiles')} tiles require "
                    f"(~{short / mean:.1f} shards' worth missing or truncated)"
                )
            elif inv.shard_bytes > payload + slack:
                problems.append(
                    f"shards hold {inv.shard_bytes:,} bytes, {inv.shard_bytes - payload:,} "
                    f"more than {inv.meta.get('n_tiles')} tiles require — extra or stale data"
                )

    for rel, size in sorted(inv.sizes.items()):
        if size == 0:
            problems.append(f"zero-length object: {rel}")

    if expect_version:
        if inv.dataset_version is None:
            problems.append(f"no dataset_version readable (expected {expect_version})")
        elif inv.dataset_version != expect_version:
            problems.append(
                f"dataset_version {inv.dataset_version} != expected {expect_version} "
                "— this is a different corpus, not a damaged one"
            )
    return problems


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python3 scripts/check_corpus.py",
        description="Assert a corpus is complete, locally or in a bucket (#80).",
    )
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument("--local", type=Path, help="a corpus directory on this machine")
    target.add_argument("--bucket", help="a bucket holding the corpus")
    p.add_argument("--prefix", default="corpus_500k/", help="key prefix (default: %(default)s)")
    p.add_argument("--endpoint-url", default=os.environ.get("AWS_ENDPOINT_URL"))
    p.add_argument("--region", default=os.environ.get("AWS_REGION", DEFAULT_REGION))
    p.add_argument("--profile", default=None, help="boto3 profile, if not using env credentials")
    p.add_argument(
        "--expect-version",
        default=DEFAULT_EXPECT_VERSION,
        help="dataset_version the corpus must report (default: the shipped 500k tier)",
    )
    p.add_argument(
        "--expect-shards",
        type=int,
        default=None,
        help="exact shard count, if you know it (gap detection runs regardless)",
    )
    p.add_argument("--no-version-check", action="store_true", help="skip the identity check")
    args = p.parse_args(argv)

    if args.local:
        inv = inspect_local(args.local)
    else:
        prefix = args.prefix if args.prefix.endswith("/") else args.prefix + "/"
        try:
            inv = inspect_bucket(
                args.bucket,
                prefix,
                endpoint_url=args.endpoint_url,
                region=args.region,
                profile=args.profile,
            )
        except ImportError:
            print("bucket mode needs boto3 — `uv sync` first, or use --local", file=sys.stderr)
            return 2
        except Exception as exc:  # botocore raises a wide family; the message is what matters
            print(f"could not read {args.bucket}/{args.prefix}: {exc}", file=sys.stderr)
            return 2

    problems = check(
        inv,
        expect_version=None if args.no_version_check else args.expect_version,
        expect_shards=args.expect_shards,
    )

    payload = inv.expected_shard_payload()
    print(f"{inv.where}")
    print(f"  objects          {len(inv.sizes)} ({len(inv.shards)} shards)")
    print(f"  total            {inv.total_bytes / 1e9:.1f} GB")
    print(
        f"  shard payload    {inv.shard_bytes / 1e9:.2f} GB"
        + (f" against {payload / 1e9:.2f} GB required" if payload else " (requirement unknown)")
    )
    print(f"  dataset_version  {inv.dataset_version or '(unreadable)'}")

    if problems:
        print(f"\n{len(problems)} problem(s):", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print("\ncomplete.")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
