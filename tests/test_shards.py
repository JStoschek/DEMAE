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
