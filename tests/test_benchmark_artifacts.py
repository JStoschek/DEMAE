"""`benchmark.artifacts` — §3.4's one guard over every committed input (#157).

`benchmark.artifacts.resolve` is asserted where it is read, against a whole frozen box
(`tests/test_benchmark_launch.py`). What attaches here is the other half of the module: the
refusal :func:`benchmark.artifacts.write_committed` holds for every stage that writes a committed
input, tested on its own rather than only through the two stages that call it, because the third
caller — #151's train mask bank sidecar — is not written yet and the guard is what it will be
relying on.

The property that earns the parameter is the last test: **what counts as "the same artifact" is
the caller's to say.** `benchmark.masks` compares the committed draw rather than the committed
bytes, because a parquet footer carries the writing library's version string; a guard that could
only compare bytes would refuse a `freeze` that had re-derived an identical artifact the day
pyarrow was upgraded.
"""

from __future__ import annotations

import hashlib

import pytest

from benchmark.artifacts import write_committed


def test_it_writes_the_body_and_returns_the_committed_files_own_md5(tmp_path):
    target = tmp_path / "not" / "made" / "yet" / "spec.json"
    path, md5 = write_committed(target, lambda: b'{"layers": 12}\n', what="contract")

    assert path == target
    assert path.read_bytes() == b'{"layers": 12}\n'
    assert md5 == hashlib.md5(path.read_bytes()).hexdigest()


def test_re_freezing_the_same_artifact_is_a_no_op_rather_than_a_rewrite(tmp_path):
    """What makes `freeze` idempotent rather than merely repeatable.

    The committed bytes are what every run record quotes, so leaving them exactly where they were
    is the point — a rewrite that produced identical content would still be a rewrite, and the
    one artifact this matters most for is the one whose md5 a half-finished run is scoring under.
    """
    target = tmp_path / "spec.json"
    _, first = write_committed(target, lambda: b"one\n", what="contract")
    written_at = target.stat().st_mtime_ns

    assert write_committed(target, lambda: b"one\n", what="contract") == (target, first)
    assert target.stat().st_mtime_ns == written_at


def test_an_artifact_whose_content_hash_has_moved_is_refused_naming_both_md5s(tmp_path):
    """§3.4's rule: re-running `freeze` after a code change must not silently re-draw an input.

    Both md5s are in the message because the person reading it has a results table quoting one of
    them, and being told which is which is the difference between "delete this file" and "delete
    this file if you meant to".
    """
    target = tmp_path / "spec.json"
    write_committed(target, lambda: b"one\n", what="contract")

    with pytest.raises(FileExistsError) as refused:
        write_committed(target, lambda: b"two\n", what="contract")

    message = str(refused.value)
    assert "content hash has moved" in message
    assert "delete it deliberately" in message
    assert "quotes the old md5" in message
    assert hashlib.md5(b"one\n").hexdigest() in message  # what is committed
    assert hashlib.md5(b"two\n").hexdigest() in message  # what this config would write
    assert "different contract" in message  # the noun the caller gave it
    # A refusal is not a partial write: the committed artifact is untouched.
    assert target.read_bytes() == b"one\n"


def test_the_callers_predicate_is_what_decides_and_the_bytes_are_only_the_default(tmp_path):
    """`benchmark.masks`'s reason for the parameter, at the guard rather than at the stage.

    A file whose bytes differ but whose *content* is the same one this call would write is a
    no-op, and the md5 that comes back is the committed file's — not the one the body would have
    hashed to. That is the number a run record quotes, and after a pyarrow upgrade the two are
    genuinely different numbers.
    """
    target = tmp_path / "masks.parquet"
    target.write_bytes(b"the same draw, re-encoded")

    path, md5 = write_committed(
        target, lambda: b"the same draw", what="draw", holds=lambda committed: True
    )

    assert (path, md5) == (target, hashlib.md5(b"the same draw, re-encoded").hexdigest())
    assert target.read_bytes() == b"the same draw, re-encoded"


def test_a_holding_artifact_never_renders_the_body(tmp_path):
    """The no-op path costs nothing but the predicate — which is why the body is a callable.

    #151's bank sidecar is written beside a 740 MB bank and #150's mask list is 30,000 rows
    through pyarrow; serializing either only to discover the file already holds it would make the
    idempotent path the expensive one.
    """
    target = tmp_path / "masks.parquet"
    target.write_bytes(b"committed")
    rendered = []

    def body() -> bytes:
        rendered.append(1)
        return b"committed"

    write_committed(target, body, what="draw", holds=lambda committed: True)
    assert rendered == []
