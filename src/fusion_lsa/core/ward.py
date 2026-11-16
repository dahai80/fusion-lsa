import logging
import math

import mlx.core as mx

logger = logging.getLogger(__name__)


def _nonzero_flat(mask: mx.array) -> tuple[mx.array, int]:
    # Indices where mask is true, ascending, plus the count - nonzero emulation via argsort
    flat = mask.reshape(-1)
    order = mx.argsort((~flat).astype(mx.int32))
    count = int(flat.sum().item())
    return order[:count], count


def pool_keys(index_keys: mx.array, pool_size: int) -> mx.array:
    # index_keys [G, T, index_dim] -> [G, T / pool_size, index_dim]
    blocks, key_count, index_dim = index_keys.shape
    if key_count % pool_size:
        raise ValueError(f"key_count {key_count} must be a multiple of pool_size {pool_size}")
    return index_keys.reshape(blocks, key_count // pool_size, pool_size, index_dim).sum(axis=2)


def ward_rounds(
    pool_keys: mx.array,
    pool_sizes: mx.array,
    bucket_count: int,
    bucket_cap: int,
    max_rounds: int = 256,
) -> tuple[mx.array, mx.array]:
    # Merge each block's pools down to bucket_count buckets; labels [G, P] int32, reached [G] bool
    blocks, pools, _ = pool_keys.shape
    key_sums = pool_keys.astype(mx.float32)
    bucket_sizes = pool_sizes.astype(mx.float32)
    alive = mx.ones((blocks, pools), dtype=mx.bool_)
    parent = mx.broadcast_to(mx.arange(pools), (blocks, pools))
    index = mx.broadcast_to(mx.arange(pools), (blocks, pools))
    diagonal = mx.eye(pools, dtype=mx.bool_)
    rows = mx.arange(blocks)[:, None]
    cols = mx.arange(pools)[None, :]
    merges_done = 0
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
        flat_idx, count = _nonzero_flat(mutual)
        if count == 0:
            break
        block_ids = flat_idx // pools
        kept = flat_idx % pools
        absorbed = best[block_ids, kept]
        # Mutual pairs are disjoint by construction (best[j] is unique, so a
        # pool can be kept by at most one partner and absorbed at most once),
        # so the whole round applies as one vectorized pass instead of a
        # per-pair Python loop. The loop version queued thousands of lazy
        # intermediates and hit the Metal live-buffer resource limit
        # (499000) on the 1M decode-start ward merge.
        # kept-side grid: recv[b, kept] = absorbed (kept slots receive the
        # absorbed pool's keys); absorbed-side grid: gone[b, absorbed] = kept
        # (absorbed slots die and point at their keeper).
        recv = mx.full((blocks, pools), -1, dtype=mx.int32)
        recv = mx.reshape(
            mx.put_along_axis(
                mx.reshape(recv, (-1,)),
                block_ids * pools + kept,
                absorbed.astype(mx.int32),
                axis=0,
            ),
            (blocks, pools),
        )
        has_recv = recv >= 0
        safe_r = mx.maximum(recv, 0)
        key_sums = key_sums + mx.where(
            has_recv[:, :, None],
            mx.take_along_axis(key_sums, safe_r[:, :, None], axis=1),
            0.0,
        )
        bucket_sizes = bucket_sizes + mx.where(
            has_recv,
            mx.take_along_axis(bucket_sizes, safe_r, axis=1),
            0.0,
        )
        gone = mx.full((blocks, pools), -1, dtype=mx.int32)
        gone = mx.reshape(
            mx.put_along_axis(
                mx.reshape(gone, (-1,)),
                block_ids * pools + absorbed,
                kept.astype(mx.int32),
                axis=0,
            ),
            (blocks, pools),
        )
        is_gone = gone >= 0
        alive = alive & ~is_gone
        parent = mx.where(is_gone, gone, parent)
        merges_done += count
        # Bound the lazy chain: each round queues a bounded set of
        # intermediates; a per-round eval flush keeps the pending-buffer
        # count flat over the 256-round budget.
        mx.eval(key_sums, bucket_sizes, alive, parent)
    for _ in range(math.ceil(math.log2(max(pools, 2))) + 1):
        parent = mx.take_along_axis(parent, parent, axis=1)
    number = mx.cumsum(alive.astype(mx.int32), axis=1) - 1
    labels = mx.take_along_axis(number, parent, axis=1)
    reached = alive.sum(axis=1) <= bucket_count
    logger.debug("ward_rounds done: %d merges over %d blocks", merges_done, blocks)
    return labels, reached


def ward_merge(
    pool_keys: mx.array, pool_sizes: mx.array, bucket_count: int, bucket_cap: int
) -> mx.array:
    bucket_size = pool_sizes.sum(axis=1).max().item() / bucket_count
    if bucket_cap < 2 * bucket_size:
        raise ValueError(
            f"the bucket cap must be at least twice the bucket size, {2 * bucket_size:g} - got {bucket_cap}"
        )
    labels, reached = ward_rounds(pool_keys, pool_sizes, bucket_count, bucket_cap)
    if not bool(reached.all()):
        raise ValueError(
            f"{int((~reached).sum().item())} blocks stopped short of {bucket_count} buckets under the cap"
        )
    return labels


def members_from_labels(labels: mx.array, bucket_count: int, bucket_cap: int) -> mx.array:
    # labels [T] -> members [bucket_count, bucket_cap] int32, -1 padded, positions in order
    members = mx.full((bucket_count, bucket_cap), -1, dtype=mx.int32)
    for bucket in range(bucket_count):
        is_mem = labels == bucket
        count = int(is_mem.sum().item())
        if count == 0:
            continue
        if count > bucket_cap:
            raise ValueError(
                f"bucket {bucket} holds {count} keys against a cap of {bucket_cap} - labels are not a valid partition"
            )
        order = mx.argsort((~is_mem).astype(mx.int32))
        keys = order[:count]
        row = mx.full((1, bucket_cap), -1, dtype=mx.int32)
        row = mx.put_along_axis(row, mx.arange(count)[None, :], keys[None, :].astype(mx.int32), axis=1)
        members = mx.put_along_axis(members, mx.array([[bucket]]), row, axis=0)
    return members
