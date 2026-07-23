"""Sharded int16 memmap storage + content hashing (ADR-0002 §1-2)."""

from __future__ import annotations

import numpy as np
import pytest

from dataset import config, shards


def _tile(fill):
    return np.full((config.TILE_PX, config.TILE_PX), fill, dtype=np.int16)


def test_content_hash_is_stable_and_endian_pinned():
    t = _tile(123)
    assert shards.content_hash(t) == shards.content_hash(t.copy())
    # hash is over little-endian int16 bytes, so byte-order of the input array
    # must not change the result
    assert shards.content_hash(t) == shards.content_hash(t.astype(">i2"))
    assert isinstance(shards.content_hash(t), int)
    assert 0 <= shards.content_hash(t) < 2**64


def test_content_hash_differs_for_different_tiles():
    assert shards.content_hash(_tile(1)) != shards.content_hash(_tile(2))


def test_writer_rolls_shards_at_capacity(tmp_path):
    w = shards.ShardWriter(tmp_path, rows_per_shard=4)
    refs = [w.add(_tile(i)) for i in range(10)]
    w.close()
    # 10 tiles / 4 per shard -> shards 0,1,2 with rows [4,4,2]
    shard_ids = [r.shard_id for r in refs]
    assert shard_ids == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2]
    row_idx = [r.row_index for r in refs]
    assert row_idx == [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]


def test_written_tiles_read_back_by_ref(tmp_path):
    w = shards.ShardWriter(tmp_path, rows_per_shard=4)
    refs = [w.add(_tile(i)) for i in range(10)]
    w.close()
    for i, r in enumerate(refs):
        mm = np.load(shards.shard_path(tmp_path, r.shard_id), mmap_mode="r")
        assert np.array_equal(mm[r.row_index], _tile(i))


def test_last_partial_shard_is_truncated_to_real_length(tmp_path):
    w = shards.ShardWriter(tmp_path, rows_per_shard=4)
    for i in range(6):
        w.add(_tile(i))
    w.close()
    last = np.load(shards.shard_path(tmp_path, 1))
    assert last.shape == (2, config.TILE_PX, config.TILE_PX)  # not padded to 4
    assert w.shard_row_counts == [4, 2]


def test_full_shard_keeps_capacity_shape(tmp_path):
    w = shards.ShardWriter(tmp_path, rows_per_shard=4)
    for i in range(4):
        w.add(_tile(i))
    w.close()
    only = np.load(shards.shard_path(tmp_path, 0))
    assert only.shape == (4, config.TILE_PX, config.TILE_PX)


def test_returned_hash_matches_content(tmp_path):
    w = shards.ShardWriter(tmp_path, rows_per_shard=8)
    ref = w.add(_tile(77))
    w.close()
    assert ref.content_hash == shards.content_hash(_tile(77))


def test_context_manager_closes(tmp_path):
    with shards.ShardWriter(tmp_path, rows_per_shard=2) as w:
        w.add(_tile(1))
        w.add(_tile(2))
        w.add(_tile(3))
    assert shards.shard_path(tmp_path, 1).exists()
    assert np.load(shards.shard_path(tmp_path, 1)).shape[0] == 1


def test_rejects_wrong_shape(tmp_path):
    w = shards.ShardWriter(tmp_path, rows_per_shard=4)
    with pytest.raises(ValueError):
        w.add(np.zeros((10, 10), dtype=np.int16))
    w.close()


# --- Resume (a half-written shard as a resume point, issue #51 item 5) --------


def _crash(out_dir, n, rows_per_shard=4):
    """Write ``n`` tiles and abandon the writer mid-shard, as a crash would.

    ``checkpoint`` (not ``close``) is what a long run calls periodically, so this
    leaves exactly what a killed process leaves: a full-capacity shard file whose
    first ``n`` rows are durable and whose tail is undefined.
    """
    w = shards.ShardWriter(out_dir, rows_per_shard=rows_per_shard)
    for i in range(n):
        w.add(_tile(i))
    state = w.checkpoint()
    del w  # no close(): the trailing shard is never truncated
    return state


def test_checkpoint_reports_the_durable_watermark(tmp_path):
    state = _crash(tmp_path, 6)
    assert state == shards.ShardState(shard_id=1, shard_row=2, shard_row_counts=[4])
    assert state.total_rows == 6


def test_resume_appends_after_the_watermark(tmp_path):
    state = _crash(tmp_path, 6)

    w = shards.ShardWriter(tmp_path, rows_per_shard=4, resume=state)
    refs = [w.add(_tile(i)) for i in range(6, 10)]
    w.close()

    assert [(r.shard_id, r.row_index) for r in refs] == [(1, 2), (1, 3), (2, 0), (2, 1)]
    assert w.shard_row_counts == [4, 4, 2]


def test_resumed_corpus_is_identical_to_an_uninterrupted_one(tmp_path):
    straight = tmp_path / "straight"
    with shards.ShardWriter(straight, rows_per_shard=4) as w:
        for i in range(10):
            w.add(_tile(i))

    broken = tmp_path / "broken"
    state = _crash(broken, 6)
    with shards.ShardWriter(broken, rows_per_shard=4, resume=state) as w:
        for i in range(6, 10):
            w.add(_tile(i))

    for shard_id in range(3):
        a = np.load(shards.shard_path(straight, shard_id))
        b = np.load(shards.shard_path(broken, shard_id))
        assert np.array_equal(a, b)


def test_resume_overwrites_tiles_written_past_the_watermark(tmp_path):
    # The crash case the checkpoint ordering allows: rows land in the shard after
    # the last checkpoint, so a resumed run re-does that work over the top of them.
    state = _crash(tmp_path, 6)
    stale = shards.ShardWriter(tmp_path, rows_per_shard=4, resume=state)
    stale.add(_tile(999))  # a row that a later checkpoint never claimed
    del stale

    with shards.ShardWriter(tmp_path, rows_per_shard=4, resume=state) as w:
        w.add(_tile(6))
    assert np.array_equal(np.load(shards.shard_path(tmp_path, 1))[2], _tile(6))


def test_resume_at_a_shard_boundary_starts_a_fresh_shard(tmp_path):
    state = _crash(tmp_path, 4)
    assert state == shards.ShardState(shard_id=0, shard_row=4, shard_row_counts=[])

    with shards.ShardWriter(tmp_path, rows_per_shard=4, resume=state) as w:
        ref = w.add(_tile(4))
    assert (ref.shard_id, ref.row_index) == (1, 0)
    assert w.shard_row_counts == [4, 1]


def test_resume_from_nothing_is_a_fresh_writer(tmp_path):
    with shards.ShardWriter(tmp_path, rows_per_shard=4, resume=shards.ShardState()) as w:
        ref = w.add(_tile(0))
    assert (ref.shard_id, ref.row_index) == (0, 0)


def test_checkpoint_on_a_fresh_writer_is_empty(tmp_path):
    w = shards.ShardWriter(tmp_path, rows_per_shard=4)
    assert w.checkpoint() == shards.ShardState()
    w.close()


def test_resume_refuses_a_shard_already_truncated_by_a_clean_close(tmp_path):
    # close() rewrites the trailing shard to its real length, so it can no longer
    # take an append. Resuming into it would silently lose rows; it raises instead.
    with shards.ShardWriter(tmp_path, rows_per_shard=4) as w:
        for i in range(6):
            w.add(_tile(i))
    stale = shards.ShardState(shard_id=1, shard_row=2, shard_row_counts=[4])
    with pytest.raises(ValueError, match="shard_00001"):
        shards.ShardWriter(tmp_path, rows_per_shard=4, resume=stale).add(_tile(6))


def test_resume_refuses_a_watermark_past_the_shard_capacity(tmp_path):
    with pytest.raises(ValueError, match="rows_per_shard"):
        shards.ShardWriter(tmp_path, rows_per_shard=4, resume=shards.ShardState(0, 5, []))


def test_checkpointed_rows_are_readable_before_close(tmp_path):
    _crash(tmp_path, 6)
    # The point of the watermark: rows 0..5 are on disk and correct even though
    # close() never ran and the trailing shard is still at full capacity.
    assert np.array_equal(np.load(shards.shard_path(tmp_path, 0)), np.stack([_tile(i) for i in range(4)]))
    partial = np.load(shards.shard_path(tmp_path, 1), mmap_mode="r")
    assert partial.shape == (4, config.TILE_PX, config.TILE_PX)
    assert np.array_equal(partial[0], _tile(4))
    assert np.array_equal(partial[1], _tile(5))
