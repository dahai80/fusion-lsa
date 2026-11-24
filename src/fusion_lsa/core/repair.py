import logging

logger = logging.getLogger(__name__)


def select_repair_pages(page_scores, static_pages, repair_count: int):
    # Minimal query-dependent repair on top of the static tier (issue #12):
    # pick `repair_count` extra pages by exact q.k score over the FULL page
    # index and union them with the static-tier pages for attention.
    #
    # page_scores: [P] float array (mlx or numpy) - exact q.k score per page
    #   of the full page index (same scoring as the select-probe).
    # static_pages: 1-D int array of page ids already selected by the static
    #   tier. May be empty.
    # repair_count: N extra pages. N=0 returns static_pages UNCHANGED (same
    #   object, same order, same dtype) - the regression guard that keeps
    #   the default path bit-identical to pure static-tier behavior.
    #
    # Returns: sorted int64 array of the union (static ∪ adaptive), no
    # double-counted ids. Ties in page_scores break by lower page id
    # (stable descending sort), so the selection is deterministic.
    import numpy as np

    static = np.asarray(static_pages)
    if repair_count <= 0:
        return static
    scores = np.asarray(page_scores).reshape(-1)
    if scores.size == 0:
        raise ValueError("page_scores is empty - no page index to repair from")
    static_ids = set(int(p) for p in static.tolist())
    if max(static_ids, default=0) >= scores.size or min(static_ids, default=0) < 0:
        raise ValueError("static_pages holds ids outside the page index")
    # Stable sort on negated scores: descending score, ascending id on ties.
    order = np.argsort(-scores, kind="stable")
    picked = []
    for pos in order:
        page = int(pos)
        if page in static_ids:
            continue
        picked.append(page)
        if len(picked) >= repair_count:
            break
    union = np.union1d(static, np.array(picked, dtype=static.dtype if static.size else np.int64))
    logger.debug(
        "repair selection: static %d pages + %d repair pages = %d union",
        static.size, len(picked), union.size,
    )
    return union


def repair_memory_delta(repair_count: int, page_tokens: int, kv_bytes_per_token: int) -> int:
    # Extra expanded-KV footprint of the repair quota vs the pure static
    # tier, in bytes: each additionally-selected page must be expanded to
    # per-token K/V for the attention read (the static tier already holds
    # its own pages, so only the quota pages are new).
    if min(repair_count, page_tokens, kv_bytes_per_token) < 0:
        raise ValueError("repair_count, page_tokens and kv_bytes_per_token must be non-negative")
    delta = repair_count * page_tokens * kv_bytes_per_token
    logger.debug(
        "repair memory delta: %d pages x %d tokens x %d B/token = %d bytes",
        repair_count, page_tokens, kv_bytes_per_token, delta,
    )
    return delta
