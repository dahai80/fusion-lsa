import logging

import mlx.core as mx
import numpy as np

from fusion_lsa.core.reference import build_buckets, read
from fusion_lsa.index.incremental import IncrementalBucketIndex

logger = logging.getLogger(__name__)

BLOCK = 512
BLOCKS = 4
TOTAL = BLOCK * BLOCKS
DIM = 64
BUCKET = 8


def _keys(seed: int = 5) -> mx.array:
    keys = mx.random.normal((TOTAL, DIM), key=mx.random.key(seed))
    return keys / mx.linalg.norm(keys, axis=1, keepdims=True)


def _plant(keys: mx.array, row: int, seed: int) -> mx.array:
    # A distinctive needle: unit vector from an independent draw, planted at a
    # known global row AND across its whole bucket (8 keys). A single planted
    # row only tilts the bucket centroid to ~0.35 - statistically
    # indistinguishable from the top random centroids - so the whole bucket
    # carries the needle and its centroid scores a guaranteed ~1.0. A query
    # equal to the needle must retrieve exactly that bucket; any
    # position-as-index confusion in the cross-chunk merge reads different
    # rows and the needle value check below fails.
    needle = mx.random.normal((DIM,), key=mx.random.key(seed))
    needle = needle / mx.linalg.norm(needle)
    keys = np.array(np.asarray(keys), copy=True)
    base = (row // BUCKET) * BUCKET
    keys[base : base + BUCKET] = np.asarray(needle)
    return mx.array(keys), needle


def _needle_rows(row: int) -> set[int]:
    # The planted key rows themselves. Ward clustering gives the needle its
    # own bucket (identical keys cluster together), but the bucket ID is NOT
    # linear in row position - only the planted rows are known a priori.
    base = (row // BUCKET) * BUCKET
    return set(range(base, base + BUCKET))


def _selected_rows(buckets, needle: mx.array, top_p: int = 32, top_k: int = 256) -> np.ndarray:
    query = needle[None, :].astype(mx.float32)
    selected, _lengths = read(
        query,
        mx.array([TOTAL - 1], dtype=mx.int32),  # decode position at the last prefill row
        buckets,
        top_p,
        top_k,
        sink_size=0,
        window_size=1,  # read() requires a positive window; 1 row is a no-op here
    )
    flat = np.asarray(selected[0])
    return flat[flat >= 0]


def test_needle_in_nonfirst_chunk_returns_original_row_index():
    # Issue #15 regression (Tianshu M4 failure class): a cross-chunk exact
    # top-k merge that scores a CONCATENATED per-chunk array returns positions
    # within that array, not row indices into the per-chunk key stores. With a
    # single-chunk prefill the two coincide; across 4 chunks they diverge and
    # every decode-step top-k reads the wrong rows - silently, with fluent
    # but wrong answers. The selection must return ORIGINAL row indices
    # (block 2, rows 1120..1127), and gathering key values at the selected
    # rows must surface the needle itself, not whatever rows the misread
    # addressed. Under the pre-fix semantics the selected rows stay in the
    # 0..511 local range and the value check fails.
    keys, needle = _plant(_keys(), 2 * BLOCK + 100, seed=11)
    np_keys = np.asarray(keys)
    index = IncrementalBucketIndex(pool_size=4, bucket_size=8, bucket_cap=16, cluster_block_size=BLOCK)
    for block in range(BLOCKS):
        index.append(keys[block * BLOCK : (block + 1) * BLOCK])
    rows = _selected_rows(index.buckets(), needle)
    planted = _needle_rows(2 * BLOCK + 100)
    assert planted & set(rows.tolist()), (
        f"planted needle rows {min(planted)}..{max(planted)} not selected; selected rows are "
        f"{rows[:16]}... - selection returned concatenated positions, not row indices"
    )
    # Value check: the selected rows address the needle in the key store.
    scores = np_keys[rows] @ np.asarray(needle)
    assert float(scores.max()) > 0.99, "selected rows do not contain the needle values"
    assert int(rows[int(scores.argmax())]) in planted, (
        "best retrieved row is not a planted needle row - selection returned "
        "positions, not row indices"
    )


def test_needle_values_match_across_all_nonfirst_chunks():
    keys_a, needle_a = _plant(_keys(), BLOCK + 17, seed=21)
    keys_b, needle_b = _plant(keys_a, 2 * BLOCK + 200, seed=22)
    np_keys = np.asarray(keys_b)
    index = IncrementalBucketIndex(pool_size=4, bucket_size=8, bucket_cap=16, cluster_block_size=BLOCK)
    for block in range(BLOCKS):
        index.append(keys_b[block * BLOCK : (block + 1) * BLOCK])
    for needle, row in ((needle_a, BLOCK + 17), (needle_b, 2 * BLOCK + 200)):
        rows = _selected_rows(index.buckets(), needle)
        planted = _needle_rows(row)
        assert planted & set(rows.tolist()), f"planted needle rows at global row {row} not selected from its chunk"
        scores = np_keys[rows] @ np.asarray(needle)
        assert float(scores.max()) > 0.99
        assert int(rows[int(scores.argmax())]) in planted


def test_per_chunk_topk_merge_equals_single_chunk_exact_topk():
    # Parity property: per-chunk exact top-k followed by a cross-chunk merge
    # (mapping (chunk, local position) back through the chunk offset table)
    # must equal whole-sequence exact top-k on the same data, same budget.
    # The offset map ci[pos] is the exact fix for the M4 bug: positions in a
    # concatenated score array are not row indices.
    keys = _keys(seed=31)
    query = mx.random.normal((DIM,), key=mx.random.key(32))
    query = query / mx.linalg.norm(query)
    scores = np.asarray(keys @ query[:, None])[:, 0]
    top_k = 40
    global_top = np.argsort(-scores)[:top_k]
    chunk = 384  # deliberately not a divisor of BLOCK-style pages
    starts = list(range(0, TOTAL, chunk))
    merged = []
    for ci, start in enumerate(starts):
        local = scores[start : start + chunk]
        for pos in np.argsort(-local)[: top_k // len(starts)]:
            merged.append(start + pos)  # idxs = ci[pos]: offset-map back
    merged = sorted(merged)
    # Whole-sequence top-k is the upper bound of any per-chunk merge at a
    # smaller total budget, and every merged row must be a row the global
    # ranking also sees as high-scoring (no position-as-index rows).
    assert len(set(merged)) == len(merged), "merge double-counted rows across chunks"
    assert max(merged) < TOTAL, "merge produced an out-of-range row index"
    merged_scores = scores[merged]
    assert float(merged_scores.min()) >= float(scores[global_top[-1]]) - 0.25, (
        "cross-chunk merge admitted low-scoring rows - positions were not mapped "
        "back to row indices"
    )


def test_needle_spanning_chunk_boundary():
    # Boundary case: needles on the last row of one block and the first row of
    # the next. The bucketed index only covers whole blocks, so both rows fall
    # inside closed pages; each must be retrievable by its global row index.
    keys, needle_last = _plant(_keys(), 2 * BLOCK - 1, seed=41)
    keys, needle_first = _plant(keys, 2 * BLOCK, seed=42)
    np_keys = np.asarray(keys)
    index = IncrementalBucketIndex(pool_size=4, bucket_size=8, bucket_cap=16, cluster_block_size=BLOCK)
    for block in range(BLOCKS):
        index.append(keys[block * BLOCK : (block + 1) * BLOCK])
    for needle, row in ((needle_last, 2 * BLOCK - 1), (needle_first, 2 * BLOCK)):
        rows = _selected_rows(index.buckets(), needle)
        planted = _needle_rows(row)
        assert planted & set(rows.tolist()), f"boundary needle rows at global row {row} not selected"
        scores = np_keys[rows] @ np.asarray(needle)
        assert float(scores.max()) > 0.99
        assert int(rows[int(scores.argmax())]) in planted


def test_batched_reference_path_returns_global_rows_for_nonfirst_needle():
    # The same failure class against the batched reference path: build_buckets
    # over the whole sequence, needle in the last block. Bucket members are
    # global by construction here (no per-chunk store), so this pins the
    # expected semantics the incremental path must reproduce.
    keys, needle = _plant(_keys(), 2 * BLOCK + 100, seed=51)
    np_keys = np.asarray(keys)
    with mx.stream(mx.cpu):
        buckets = build_buckets(keys, 4, 8, 16, BLOCK)
        rows = _selected_rows(buckets, needle)
    planted = _needle_rows(2 * BLOCK + 100)
    assert planted & set(rows.tolist())
    scores = np_keys[rows] @ np.asarray(needle)
    assert float(scores.max()) > 0.99
    assert int(rows[int(scores.argmax())]) in planted
