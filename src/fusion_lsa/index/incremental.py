import logging
from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np

from fusion_lsa.core.reference import Buckets, build_buckets

logger = logging.getLogger(__name__)


def _memory_snapshot() -> str:
    # One-line allocator + process state for the 1M decode-time page-close
    # postmortem (metal::malloc Resource limit 499000 at this exact site,
    # runs v5 2026-11-12 and v7 2026-11-15). Logged per page close; pages
    # are infrequent so the cost is noise.
    import os
    import subprocess

    try:
        rss = int(
            subprocess.run(
                ["ps", "-o", "rss=", "-p", str(os.getpid())],
                capture_output=True,
                text=True,
                timeout=2,
            ).stdout.strip()
        ) * 1024
    except Exception:
        rss = 0
    return (
        f"active={mx.get_active_memory() / 2**30:.2f}GB "
        f"cache={mx.get_cache_memory() / 2**30:.2f}GB "
        f"peak={mx.get_peak_memory() / 2**30:.2f}GB "
        f"rss={rss / 2**30:.2f}GB"
    )


@dataclass
class IncrementalBucketIndex:
    # Page-boundary contract: buckets are built only over whole cluster blocks, one page of
    # keys at a time. A prefill that stops mid page accumulates keys outside this index and
    # feeds them here once the page closes - mid-page ward merges are refused, not attempted.
    pool_size: int
    bucket_size: int
    bucket_cap: int
    cluster_block_size: int
    _blocks: list[Buckets] = field(default_factory=list, init=False, repr=False)
    _key_count: int = field(default=0, init=False, repr=False)
    _cached_buckets: Buckets | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.bucket_size > self.cluster_block_size or self.cluster_block_size % self.bucket_size:
            raise ValueError(
                f"bucket_size {self.bucket_size} must divide cluster_block_size {self.cluster_block_size}"
            )

    def append(self, keys: mx.array) -> None:
        # One closed page of keys [cluster_block_size, index_dim]; anything else is refused
        if keys.ndim != 2 or keys.shape[0] != self.cluster_block_size:
            raise ValueError(
                f"append takes exactly one page of {self.cluster_block_size} keys - got shape {keys.shape}"
            )
        # The Metal buffer quota is process-wide and the allocator cache counts
        # toward it. Long decode steps churn small buffers into the pool until
        # an unrelated allocation (the ward merge here) trips the 499000 limit.
        # Page closes are infrequent (one per cluster_block_size tokens), so
        # trimming the cache here is cheap insurance. 1M runs v5/v7 died at a
        # decode-time page close inside build_buckets. Linux mlx builds have
        # no mx.metal submodule (CI runs the CPU wheel) - degrade to a no-op.
        metal = getattr(mx, "metal", None)
        if metal is not None:
            metal.clear_cache()
        try:
            # v8 (2026-11-17) showed the quota denial reproduces even at
            # PREFILL-time page closes (98.7% through the 1M prefill) with
            # ~81GB live buffers and a denied scalar allocation - clear_cache
            # cannot fix it. The ward merge is small (one cluster block) and
            # infrequent, so it runs on the CPU stream; unified memory lets
            # the resulting buckets feed GPU decode kernels without a copy.
            with mx.stream(mx.cpu):
                block = build_buckets(keys, self.pool_size, self.bucket_size, self.bucket_cap, self.cluster_block_size)
        except RuntimeError as exc:
            # 1M decode-time page close hit metal::malloc Resource limit
            # (499000) in v5/v7. Log the allocator state, flush everything
            # pending, clear the cache once and retry before giving up - if
            # the denial is a transient pressure spike this rides over it,
            # otherwise the retry fails with the telemetry line right above.
            logger.error(
                "incremental index: page close failed, %s; snapshot %s",
                exc,
                _memory_snapshot(),
            )
            mx.synchronize(mx.default_stream(mx.default_device()))
            if metal is not None:
                metal.clear_cache()
            block = build_buckets(keys, self.pool_size, self.bucket_size, self.bucket_cap, self.cluster_block_size)
            logger.info("incremental index: page close retry succeeded, snapshot %s", _memory_snapshot())
        logger.info(
            "incremental index: block %d appended, %d blocks, %d keys, %s",
            len(self._blocks),
            len(self._blocks) + 1,
            self._key_count + self.cluster_block_size,
            _memory_snapshot(),
        )
        # The block is stored for the life of the request: a lazy array pins
        # its whole materialized upstream graph (build_buckets intermediates),
        # ~203 MTL buffers per page close - invisible to gc, untracked by
        # active-memory accounting on the CPU-stream path, and unreachable by
        # clear_cache (live graph nodes, not cache). The 1M decode-entry
        # rebuilds (v8/v9r2) died at the process-wide 499000 MTLResource cap
        # after 2100+ closes this way. Even evaluated mx.arrays would retain 3
        # buffers per close (410k over a full 64-layer rebuild), so blocks are
        # copied to host numpy instead: zero MTL resources per close, and
        # buckets() materializes one evaluated concatenation per field.
        mx.eval(block.bucket_centroids, block.bucket_members, block.bucket_sizes)
        # np.asarray on an mx.array is a zero-copy VIEW that pins the MTL
        # buffer through numpy's base reference - one buffer per field per
        # close still walks a 64-layer rebuild into the 499000 cap. The
        # second np.array(copy=True) produces an owned host copy and lets
        # the view (and its buffer) die at scope exit.
        centroids = np.array(np.asarray(block.bucket_centroids), copy=True)
        # build_buckets over one page numbers keys locally; shift them to positions in the whole sequence
        offset = len(self._blocks) * self.cluster_block_size
        members = np.array(np.asarray(block.bucket_members), copy=True)
        members = np.where(members >= 0, members + offset, members).astype(np.int32)
        sizes = np.array(np.asarray(block.bucket_sizes), copy=True)
        block = Buckets(
            centroids, members, sizes, block.block_bucket_count, block.cluster_block_size
        )
        self._blocks.append(block)
        self._key_count += self.cluster_block_size
        self._cached_buckets = None
        logger.debug(
            "incremental index: block %d appended, %d blocks, %d keys",
            len(self._blocks),
            len(self._blocks),
            self._key_count,
        )

    @property
    def block_count(self) -> int:
        return len(self._blocks)

    @property
    def key_count(self) -> int:
        return self._key_count

    def buckets(self) -> Buckets:
        # The concatenated buckets of every closed block; block i's rows are immutable once built.
        # Concat is cached: called every decode step, and blocks only grow on page close.
        if not self._blocks:
            raise ValueError("the index holds no closed blocks")
        if self._cached_buckets is None:
            # Host-side numpy concat, one mx.array materialization per field:
            # concatenating mx.arrays here would pin every block's buffer in
            # the lazy result graph (the same retention class that killed the
            # 1M rebuilds at the 499000 MTLResource cap).
            self._cached_buckets = Buckets(
                mx.array(np.concatenate([block.bucket_centroids for block in self._blocks])),
                mx.array(np.concatenate([block.bucket_members for block in self._blocks])),
                mx.array(np.concatenate([block.bucket_sizes for block in self._blocks])),
                self._blocks[0].block_bucket_count,
                self.cluster_block_size,
            )
        return self._cached_buckets
