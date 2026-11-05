# fusion-lsa

Latent Sparse Attention (LSA) ported to MLX/Metal for the [Fusion](../) monorepo — a local-first
Apple Silicon AI ecosystem.

LSA keeps the attention kernel unchanged and only decides **which keys attention reads**: index
keys are pooled and Ward-clustered into buckets, queries score bucket centroids, pick the top-p
buckets, and expand them to a top-k key set beside the sink tokens and the sliding window. This
repo ports the selection logic (not the CUDA kernels) from the
[latent-sparse-attention](https://github.com/Prefrontal-AI/latent-sparse-attention) plain-torch
reference, parity-tested against it as an oracle.

Status: P0 (algorithm port + parity + cost gate) complete. P1 (packaging, map converter,
checkpoint e2e) next. See `docs/p0-report.md`.

## Results (P0, M5 Max 128GB)

- Exact oracle parity on bucket partitions, members, sizes and selected key sets; centroid parity
  within 1.5e-4 (fp32 summation order).
- Decode selection ~0.7-1.0 ms/query; ward/bucket build 7.2s per 128k context (page-boundary only).
- 128k end-to-end: needle retrieved, selected keys hold > 0.9 of attention softmax mass.

## Layout

```
src/fusion_lsa/
  core/         ward.py, reference.py — MLX port of the oracle selection logic
  evaluation/   selection_bench.py — per-block selection cost bench
tests/          oracle parity + 128k end-to-end
docs/           P0 report, bench results and logs
```

## Setup

```bash
cd /Users/dahai/fusion
source .venv/bin/activate
cd fusion-lsa && pip install -e ".[test]"
python -m pytest tests/
```

Parity tests read the upstream oracle from `~/model/latent-sparse-attention/src`
(override with `LSA_UPSTREAM`); they skip when it is absent.

## Hard boundaries

- Selection-logic only: attention itself runs as standard SDPA over the selected keys.
- 1M+ context requires a latent-KV architecture plus KV quantization; full-KV models do not fit
  120GB at 1M.
- Only page-boundary bucket/hierarchy growth; decode selection is per token, never re-clusters.
