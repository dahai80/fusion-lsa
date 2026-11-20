import logging

import mlx.core as mx
import numpy as np
import pytest

from fusion_lsa.core.reference import build_buckets
from fusion_lsa.index.incremental import IncrementalBucketIndex

logger = logging.getLogger(__name__)

BLOCK = 512


def _keys(total: int, seed: int = 5) -> mx.array:
    keys = mx.random.normal((total, 64), key=mx.random.key(seed))
    return keys / mx.linalg.norm(keys, axis=1, keepdims=True)


def test_incremental_matches_batched():
    # Both sides run on the CPU stream: the ward merge's argmin over fp32
    # gram similarities is not reproducible ACROSS devices (CPU vs GPU
    # reduction order flips argmin ties, cascading into different cluster
    # memberships), so byte-equal parity is only pinned per device.
    keys = _keys(BLOCK * 4)
    with mx.stream(mx.cpu):
        batched = build_buckets(keys, 4, 8, 16, BLOCK)
    index = IncrementalBucketIndex(pool_size=4, bucket_size=8, bucket_cap=16, cluster_block_size=BLOCK)
    with mx.stream(mx.cpu):
        for block in range(4):
            index.append(keys[block * BLOCK : (block + 1) * BLOCK])
    incremental = index.buckets()
    assert np.array_equal(np.array(incremental.bucket_centroids), np.array(batched.bucket_centroids))
    assert np.array_equal(np.array(incremental.bucket_members), np.array(batched.bucket_members))
    assert np.array_equal(np.array(incremental.bucket_sizes), np.array(batched.bucket_sizes))


def test_append_runs_without_metal_module(monkeypatch):
    # CI runs on ubuntu: mlx ships Linux CPU wheels with no mx.metal
    # submodule. The page-close path must degrade to a no-op cache trim
    # there instead of raising AttributeError.
    import fusion_lsa.index.incremental as inc

    monkeypatch.delattr(inc.mx, "metal", raising=False)
    index = IncrementalBucketIndex(pool_size=4, bucket_size=8, bucket_cap=16, cluster_block_size=BLOCK)
    index.append(_keys(BLOCK))
    assert index._key_count == BLOCK


def test_append_matches_batched_on_default_stream():
    # Issue #7 regression: b066be6 forced the page-close ward merge onto the
    # CPU stream, and the CPU vs GPU fp32 reduction order flipped argmin ties
    # into different cluster memberships - the consumer oracle (batch build
    # on the default stream) then selected different rows. The page close
    # must run on the caller's stream so the incremental index stays
    # byte-equal to a batch build on that same stream.
    keys = _keys(BLOCK * 4)
    batched = build_buckets(keys, 4, 8, 16, BLOCK)
    mx.eval(batched.bucket_centroids, batched.bucket_members, batched.bucket_sizes)
    index = IncrementalBucketIndex(pool_size=4, bucket_size=8, bucket_cap=16, cluster_block_size=BLOCK)
    for block in range(4):
        index.append(keys[block * BLOCK : (block + 1) * BLOCK])
    incremental = index.buckets()
    mx.eval(incremental.bucket_centroids, incremental.bucket_members, incremental.bucket_sizes)
    assert np.array_equal(np.array(incremental.bucket_centroids), np.array(batched.bucket_centroids))
    assert np.array_equal(np.array(incremental.bucket_members), np.array(batched.bucket_members))
    assert np.array_equal(np.array(incremental.bucket_sizes), np.array(batched.bucket_sizes))


def test_append_follows_caller_stream(monkeypatch):
    # The page close builds on whatever stream the caller is on. Callers on
    # the CPU stream (long-context jobs that pre-build their index on CPU)
    # keep CPU-side byte parity with their own batch reference; the metal
    # quota is only ever hit with large live GPU buffers, which a CPU-stream
    # caller does not have.
    import fusion_lsa.index.incremental as inc

    streams = []
    orig_stream = inc.mx.stream

    def spy(device=None):
        streams.append(device)
        return orig_stream(device)

    monkeypatch.setattr(inc.mx, "stream", spy)
    index = IncrementalBucketIndex(pool_size=4, bucket_size=8, bucket_cap=16, cluster_block_size=BLOCK)
    with mx.stream(mx.cpu):
        index.append(_keys(BLOCK))
    assert mx.cpu in streams


def test_append_falls_back_to_cpu_on_resource_error(monkeypatch, caplog):
    # The 1M runs (v5/v7/v8) hit the process-wide 499000 MTLResource cap at
    # prefill-time page closes with ~81GB of live GPU buffers: the GPU build
    # of the ward merge is denied a scalar allocation. The close must then
    # retry on the CPU stream - unified memory lets the resulting buckets
    # feed GPU decode kernels without a copy - instead of killing the run.
    import fusion_lsa.index.incremental as inc

    calls = []
    orig_build = inc.build_buckets

    def flaky(keys, *args):
        calls.append(len(calls))
        if len(calls) == 1:
            raise RuntimeError("metal::malloc is not allowed to allocate 4 bytes")
        return orig_build(keys, *args)

    monkeypatch.setattr(inc, "build_buckets", flaky)
    index = IncrementalBucketIndex(pool_size=4, bucket_size=8, bucket_cap=16, cluster_block_size=BLOCK)
    with caplog.at_level(logging.WARNING, logger="fusion_lsa.index.incremental"):
        index.append(_keys(BLOCK))
    assert len(calls) == 2, "append did not retry the page close on the CPU stream"
    assert index._key_count == BLOCK
    assert any("falling back" in r.getMessage().lower() for r in caplog.records)


def test_append_cpu_stream_matches_batched():
    keys = _keys(BLOCK * 2)
    with mx.stream(mx.cpu):
        batched = build_buckets(keys, 4, 8, 16, BLOCK)
    index = IncrementalBucketIndex(pool_size=4, bucket_size=8, bucket_cap=16, cluster_block_size=BLOCK)
    with mx.stream(mx.cpu):
        for block in range(2):
            index.append(keys[block * BLOCK : (block + 1) * BLOCK])
    incremental = index.buckets()
    assert np.array_equal(np.array(incremental.bucket_centroids), np.array(batched.bucket_centroids))
    assert np.array_equal(np.array(incremental.bucket_members), np.array(batched.bucket_members))
    assert np.array_equal(np.array(incremental.bucket_sizes), np.array(batched.bucket_sizes))


def test_incremental_refuses_partial_page():
    index = IncrementalBucketIndex(pool_size=4, bucket_size=8, bucket_cap=16, cluster_block_size=BLOCK)
    with pytest.raises(ValueError, match="page"):
        index.append(_keys(BLOCK + 100))


def test_partial_tail_never_bucketed():
    # A key count that is not a whole cluster block leaves the tail outside the buckets
    keys = _keys(BLOCK * 3 + 200)
    batched = build_buckets(keys, 4, 8, 16, BLOCK)
    assert batched.bucket_centroids.shape[0] == 3 * (BLOCK // 8)
    index = IncrementalBucketIndex(pool_size=4, bucket_size=8, bucket_cap=16, cluster_block_size=BLOCK)
    for block in range(3):
        index.append(keys[block * BLOCK : (block + 1) * BLOCK])
    with pytest.raises(ValueError, match="page"):
        index.append(keys[3 * BLOCK :])


def test_append_stores_blocks_on_host():
    # The 1M decode-entry rebuild (v8 2026-11-17, v9r2 2026-11-17) died at the
    # process-wide 499000 MTLResource cap: every closed page stored its lazy
    # Buckets arrays un-evaluated, pinning ~203 MTL intermediate buffers per
    # close (measured with an in-process MetalAllocator counter probe). The
    # intermediates are invisible to gc (python only holds the three output
    # arrays), untracked by mx.get_active_memory for the CPU-stream path, and
    # unreachable by mx.clear_cache (live graph nodes, not cache), so the
    # counter climbed ~200 per close for 2100+ closes until metal::malloc was
    # denied. Even with mx.eval on store, the three retained arrays cost 3
    # MTL buffers per close - 410k resources across a full 64-layer rebuild
    # (136,896 closes), 82% of the cap. Blocks are therefore stored as numpy
    # on the host: zero MTL resources per close, and buckets() materializes
    # one evaluated concatenation per field on demand.
    index = IncrementalBucketIndex(pool_size=4, bucket_size=8, bucket_cap=16, cluster_block_size=BLOCK)
    index.append(_keys(BLOCK))
    block = index._blocks[-1]
    for stored in (block.bucket_centroids, block.bucket_members, block.bucket_sizes):
        assert isinstance(stored, np.ndarray), "append stored an MTL-resident block array"
    assert block.bucket_members.dtype == np.int32
    assert block.bucket_centroids.dtype == np.float32
    assert int(block.bucket_members.max()) == BLOCK - 1
    assert int(block.bucket_sizes.sum()) == BLOCK


def test_append_block_storage_does_not_pin_mtl_buffers():
    # mlx's numpy conversion is zero-copy on unified memory: a numpy array
    # taken straight from an mx.array keeps the MTL buffer alive through the
    # view (base is the mx buffer). Host storage must detach - a pinned
    # buffer per field per close still walks a 64-layer rebuild (136,896
    # closes) into the 499000 MTLResource cap.
    index = IncrementalBucketIndex(pool_size=4, bucket_size=8, bucket_cap=16, cluster_block_size=BLOCK)
    index.append(_keys(BLOCK))
    block = index._blocks[-1]
    for stored in (block.bucket_centroids, block.bucket_members, block.bucket_sizes):
        assert stored.base is None, "append stored a numpy view pinning an MTL buffer"


def test_buckets_concat_from_host_blocks():
    # buckets() rebuilds its evaluated mx.array concatenation from the
    # host-resident blocks; values must match a straight build_buckets run.
    keys = _keys(BLOCK * 2)
    with mx.stream(mx.cpu):
        batched = build_buckets(keys, 4, 8, 16, BLOCK)
    index = IncrementalBucketIndex(pool_size=4, bucket_size=8, bucket_cap=16, cluster_block_size=BLOCK)
    with mx.stream(mx.cpu):
        for block in range(2):
            index.append(keys[block * BLOCK : (block + 1) * BLOCK])
    incremental = index.buckets()
    assert np.array_equal(np.array(incremental.bucket_centroids), np.array(batched.bucket_centroids))
    assert np.array_equal(np.array(incremental.bucket_members), np.array(batched.bucket_members))
    assert np.array_equal(np.array(incremental.bucket_sizes), np.array(batched.bucket_sizes))


def test_two_thousand_appends_hold_resources_flat():
    # Issue #4 regression: the ad-hoc v10 probe ("2,000 consecutive appends
    # hold the resource counter flat, delta 0, active 0.1MB vs +203/close
    # before") committed as a standing test. Guards the host-storage fix
    # (2dc3698) whose absence killed v5/v7/v8/v9r2 at close 2113 with the
    # 499000 MTLResource cap denial. Mutation check: reverting append to
    # store live (or merely evaluated) mx.array blocks fails the isinstance
    # sweep below; storing numpy views instead of detached copies fails the
    # base-None sweep. Active-memory growth is bounded because host numpy
    # blocks allocate zero MTL buffers per close.
    metal = getattr(mx, "metal", None)
    samples = []
    keys = _keys(BLOCK * 2000)
    mx.eval(keys)
    active_before = metal.get_active_memory() if metal is not None else None
    index = IncrementalBucketIndex(pool_size=4, bucket_size=8, bucket_cap=16, cluster_block_size=BLOCK)
    for block in range(2000):
        index.append(keys[block * BLOCK : (block + 1) * BLOCK])
        if block % 500 == 0 and metal is not None:
            samples.append(metal.get_active_memory() - active_before)
        # semantics guard: sampled closes must match the eager reference
        # partition on the same input and stream (host storage changed no
        # selection); the close now follows the caller's stream (issue #7)
        if block % 200 == 0:
            sampled = build_buckets(keys[block * BLOCK : (block + 1) * BLOCK], 4, 8, 16, BLOCK)
            mx.eval(sampled.bucket_members, sampled.bucket_sizes)
            stored = index._blocks[-1]
            expected = np.where(
                np.asarray(sampled.bucket_members) >= 0,
                np.asarray(sampled.bucket_members) + block * BLOCK,
                np.asarray(sampled.bucket_members),
            ).astype(np.int32)
            assert np.array_equal(stored.bucket_members, expected)
            assert np.array_equal(stored.bucket_sizes, np.asarray(sampled.bucket_sizes))
    assert len(index._blocks) == 2000
    for stored in index._blocks:
        assert isinstance(stored.bucket_centroids, np.ndarray)
        assert isinstance(stored.bucket_members, np.ndarray)
        assert isinstance(stored.bucket_sizes, np.ndarray)
        assert stored.bucket_centroids.base is None
        assert stored.bucket_members.base is None
        assert stored.bucket_sizes.base is None
    if metal is not None:
        growth = metal.get_active_memory() - active_before
        logger.info("active-memory growth over 2000 appends: %d bytes (samples %s)", growth, samples)
        assert growth < 16 * 1024 * 1024, f"append retained MTL buffers across closes (growth {growth} bytes)"


def test_closed_blocks_stay_unchanged():
    index = IncrementalBucketIndex(pool_size=4, bucket_size=8, bucket_cap=16, cluster_block_size=BLOCK)
    index.append(_keys(BLOCK, seed=7))
    first = index.buckets()
    index.append(_keys(BLOCK, seed=8))
    second = index.buckets()
    assert np.array_equal(np.array(first.bucket_centroids), np.array(second.bucket_centroids[: first.bucket_centroids.shape[0]]))
    assert np.array_equal(np.array(first.bucket_members), np.array(second.bucket_members[: first.bucket_members.shape[0]]))
    assert second.bucket_centroids.shape[0] == first.bucket_centroids.shape[0] * 2
