import mlx.core as mx
import numpy as np


def _ward_rounds_seq(pool_keys, pool_sizes, bucket_count, bucket_cap, max_rounds=256):
    # Reference: the original per-pair sequential merge loop.
    import math

    blocks, pools, _ = pool_keys.shape
    key_sums = pool_keys.astype(mx.float32)
    bucket_sizes = pool_sizes.astype(mx.float32)
    alive = mx.ones((blocks, pools), dtype=mx.bool_)
    parent = mx.broadcast_to(mx.arange(pools), (blocks, pools))
    index = mx.broadcast_to(mx.arange(pools), (blocks, pools))
    diagonal = mx.eye(pools, dtype=mx.bool_)
    rows = mx.arange(blocks)[:, None]
    cols = mx.arange(pools)[None, :]
    for _ in range(max_rounds):
        merges_left = alive.sum(axis=1) - bucket_count
        if not bool((merges_left > 0).any()):
            break
        gram = key_sums @ key_sums.transpose(0, 2, 1)
        squares = mx.take_along_axis(gram, mx.arange(pools)[None, :, None], axis=2)[:, :, 0]
        merged_alignments = mx.maximum(squares[:, :, None] + squares[:, None, :] + 2 * gram, 0.0).sqrt()
        alignments = mx.maximum(squares, 0.0).sqrt()
        merge_costs = alignments[:, :, None] + alignments[:, None, :] - merged_alignments
        allowed = alive[:, :, None] & alive[:, None, :] & ~diagonal
        allowed = allowed & (bucket_sizes[:, :, None] + bucket_sizes[:, None, :] <= bucket_cap)
        masked = mx.where(allowed, merge_costs, mx.inf)
        best_costs = masked.min(axis=2)
        best = masked.argmin(axis=2)
        mutual = (best_costs < mx.inf) & (mx.take_along_axis(best, best, axis=1) == index) & (index < best)
        rank = mx.argsort(mx.argsort(mx.where(mutual, best_costs, mx.inf), axis=1), axis=1)
        mutual = mutual & (rank < mx.maximum(merges_left, 0)[:, None])
        flat = mx.argsort((~mutual.reshape(-1)).astype(mx.int32))
        count = int(mutual.reshape(-1).sum().item())
        if count == 0:
            break
        flat_idx = flat[:count]
        block_ids = flat_idx // pools
        kept = flat_idx % pools
        absorbed = best[block_ids, kept]
        updates = key_sums[block_ids, absorbed]
        for i in range(count):
            row_hit = block_ids[i : i + 1] == rows
            kept_hit = row_hit & (cols == kept[i])
            abs_hit = row_hit & (cols == absorbed[i])
            key_sums = key_sums + kept_hit[:, :, None] * updates[i][None, None, :]
            bucket_sizes = bucket_sizes + mx.where(kept_hit, bucket_sizes[block_ids[i], absorbed[i]], 0.0)
            alive = alive & ~abs_hit
            parent = mx.where(abs_hit, mx.full(parent.shape, int(kept[i]), dtype=parent.dtype), parent)
    for _ in range(math.ceil(math.log2(max(pools, 2))) + 1):
        parent = mx.take_along_axis(parent, parent, axis=1)
    number = mx.cumsum(alive.astype(mx.int32), axis=1) - 1
    labels = mx.take_along_axis(number, parent, axis=1)
    reached = alive.sum(axis=1) <= bucket_count
    return labels, reached


def test_matches_sequential():
    from fusion_lsa.core.ward import ward_rounds

    rng = np.random.default_rng(3)
    for trial in range(5):
        blocks, pools, dim = 4, 16, 8
        keys = mx.array(rng.normal(size=(blocks, pools, dim)).astype(np.float32))
        sizes = mx.array(rng.integers(1, 3, size=(blocks, pools)).astype(np.float32))
        bucket_count = 4
        cap = int(sizes.sum(axis=1).max().item() / bucket_count) * 2
        labels_v, reached_v = ward_rounds(keys, sizes, bucket_count, cap)
        labels_s, reached_s = _ward_rounds_seq(keys, sizes, bucket_count, cap)
        assert np.array_equal(np.array(labels_v), np.array(labels_s)), f"labels differ trial {trial}"
        assert np.array_equal(np.array(reached_v), np.array(reached_s)), f"reached differ trial {trial}"
    print("ward_rounds vectorized == sequential on 5 random cases")


def test_partition_and_cap():
    from fusion_lsa.core.ward import ward_rounds

    rng = np.random.default_rng(11)
    blocks, pools, dim = 3, 32, 16
    keys = mx.array(rng.normal(size=(blocks, pools, dim)).astype(np.float32))
    sizes = mx.array(rng.integers(1, 4, size=(blocks, pools)).astype(np.float32))
    bucket_count = 8
    cap = int(sizes.sum(axis=1).max().item() / bucket_count) * 2 + 2
    labels, reached = ward_rounds(keys, sizes, bucket_count, cap)
    labels_np = np.array(labels)
    for b in range(blocks):
        counts = np.bincount(labels_np[b], minlength=bucket_count)
        assert counts.max() <= cap, f"block {b} bucket over cap: {counts.max()} > {cap}"
    print("labels form a valid partition under the cap")


if __name__ == "__main__":
    test_matches_sequential()
    test_partition_and_cap()
