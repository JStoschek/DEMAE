"""The corpus build orchestrator — smoke gate then full build."""

from __future__ import annotations

import json

import pyarrow.parquet as pq
import pytest

from dataset import build
from dataset.synthetic import synthetic_reader


def _build(tmp_path, **kw):
    return build.build_corpus(
        synthetic_reader(),
        tmp_path,
        per_stratum_quota=kw.pop("per_stratum_quota", 4),
        smoke_cogs=kw.pop("smoke_cogs", 2),
        smoke_quota=kw.pop("smoke_quota", 1),
        snapshot_date="2026-07",
        code_version="testsha",
        build_timestamp="2026-07-14T00:00:00Z",
        **kw,
    )


def test_build_runs_smoke_then_full(tmp_path):
    out = _build(tmp_path)
    assert out.smoke is not None
    assert out.smoke.n_written > 0
    assert out.corpus.n_written > 0
    # the smoke tier is validated in its own dir, not mixed into the corpus
    assert out.smoke.manifest_path != out.corpus.manifest_path
    # the shipped corpus is the bigger of the two
    assert out.corpus.n_written >= out.smoke.n_written


def test_build_writes_full_corpus_artifacts(tmp_path):
    out = _build(tmp_path)
    assert out.corpus.manifest_path.exists()
    assert out.corpus.corpus_meta_path.exists()
    assert out.corpus.shard_dir.exists()
    meta = json.loads(out.corpus.corpus_meta_path.read_text())
    assert meta["n_tiles"] == out.corpus.n_written
    assert meta["code_version"] == "testsha"
    assert pq.read_table(out.corpus.manifest_path).num_rows == out.corpus.n_written


def test_build_can_skip_smoke(tmp_path):
    out = _build(tmp_path, run_smoke=False)
    assert out.smoke is None
    assert out.corpus.n_written > 0


def test_build_is_deterministic(tmp_path):
    a = _build(tmp_path / "a")
    b = _build(tmp_path / "b")
    assert a.corpus.dataset_version == b.corpus.dataset_version
    assert a.corpus.n_written == b.corpus.n_written


def test_smoke_gate_raises_when_smoke_empty(tmp_path):
    # An empty bucket materializes nothing; the smoke gate must fail loudly
    # rather than proceed to a doomed full build.
    from dataset.tiling import SyntheticCogReader

    with pytest.raises(build.SmokeValidationError):
        build.build_corpus(
            SyntheticCogReader({}),
            tmp_path,
            per_stratum_quota=4,
            smoke_cogs=2,
            smoke_quota=1,
            snapshot_date="x",
            code_version="x",
        )
