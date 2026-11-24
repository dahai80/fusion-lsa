import logging

import numpy as np
import pytest

from fusion_lsa.core.repair import repair_memory_delta, select_repair_pages

logger = logging.getLogger(__name__)


def test_zero_repair_is_bit_identical_to_static():
    # N=0 regression guard: the default path must be untouched - same
    # object, order and dtype, no copy, no resorting.
    static = np.array([3, 1, 7, 0], dtype=np.int32)
    scores = np.array([0.1, 0.9, 0.5, 0.2, 0.8])
    result = select_repair_pages(scores, static, 0)
    assert result is static, "N=0 must return the static selection unchanged"
    assert np.array_equal(result, np.array([3, 1, 7, 0], dtype=np.int32))


def test_union_adds_pages_without_double_count():
    static = np.array([0, 5, 9], dtype=np.int64)
    # Top-scoring non-static pages: 8 (0.95), 2 (0.9), then 5 (0.8, static).
    scores = np.array([0.0, 0.1, 0.9, 0.2, 0.3, 0.8, 0.1, 0.4, 0.95, 0.05])
    result = select_repair_pages(scores, static, 2)
    assert sorted(result.tolist()) == [0, 2, 5, 8, 9]
    assert len(set(result.tolist())) == result.size, "union double-counted a page"
    assert np.issubdtype(result.dtype, np.integer)


def test_repair_never_duplicates_static_pages():
    static = np.array([1, 2], dtype=np.int32)
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    for n in range(1, 5):
        result = select_repair_pages(scores, static, n)
        assert len(set(result.tolist())) == result.size
        assert {1, 2} <= set(result.tolist())
    # Asking for more repair pages than the index holds exhausts the index
    # without crashing or duplicating.
    everything = select_repair_pages(scores, static, 100)
    assert sorted(everything.tolist()) == [0, 1, 2, 3, 4]


def test_ties_break_by_lower_page_id():
    static = np.array([], dtype=np.int64)
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    result = select_repair_pages(scores, static, 2)
    assert result.tolist() == [0, 1], "tie-break must be deterministic (lower page id first)"


def test_static_ids_outside_index_rejected():
    with pytest.raises(ValueError, match="outside the page index"):
        select_repair_pages(np.array([0.1, 0.2]), np.array([5]), 1)
    with pytest.raises(ValueError, match="empty"):
        select_repair_pages(np.array([]), np.array([0]), 1)


def test_memory_delta_arithmetic():
    # 2 repair pages x 512 tokens x 24.8KB/token - the 950DT k8v4 figure.
    assert repair_memory_delta(2, 512, 25395) == 2 * 512 * 25395
    assert repair_memory_delta(0, 512, 25395) == 0
    with pytest.raises(ValueError):
        repair_memory_delta(-1, 512, 25395)
