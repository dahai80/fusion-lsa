# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

fusion-lsa: Latent Sparse Attention (LSA) ported to MLX/Metal for the Fusion monorepo. The attention kernel is unchanged; only **which keys attention reads** is decided — index keys are pooled and Ward-clustered into buckets, queries score bucket centroids, top-p buckets expand to a top-k key set beside sink tokens and the sliding window. Ported from the [latent-sparse-attention](https://github.com/Prefrontal-AI/latent-sparse-attention) plain-torch reference, parity-tested against it as an oracle (see NOTICE.md).

Status: P0 (algorithm port + parity + cost gate) complete; P1 in progress (map converter for GLM-5.3-Flash / Qwen3.8-27B, index space port, page-boundary incremental index, real-checkpoint import).

## Commands

```bash
pip install -e ".[test]"     # install with test extras
python -m pytest tests/ -q   # full suite (asyncio_mode=auto)
ruff check .                 # lint gate (also enforced in CI)
ruff check --fix .           # auto-fix then hand-fix remainder
```

Key scripts (all under `scripts/`):

- `official_ruler/run_official_ruler.py` — official RULER harness; tokenizer dir defaults to `models/qwen38-27b` (override with `RULER_TOKENIZER_DIR`)
- `ruler32k_bench.py`, `ruler_vt_fwe_sweep.sh` — 32k RULER bench / vt+fwe sweeps
- `tq_probe.py`, `tq_kv_probe.py` — K/V quantization probes (outputs land in `docs/tq-*.md`)
- `kv_compress_report.py`, `system_memory_watchdog.py` — reporting and OOM watchdog

## Layout

- `src/fusion_lsa/` — `core/` (selection logic, parity), `index/`, `quant/`, `evaluation/`, `convert/`, `budget.py`
- `tests/` — pytest; some tests skip without MLX/Metal or model artifacts
- `docs/` — probe reports and design docs (P3 coexistence, HSA M0 spike plan)

## Conventions

- `models/` holds local model weights (git-ignored, ~23G). The Qwen3.8-27B snapshot is the locked model for the HSA M0 spike and probe vector sources — do not delete; scripts reference it by default.
- Parity rule: selection logic must match the upstream torch oracle bit-for-bit on identical inputs; new selection features need a parity test.
- Probe outputs are committed as markdown docs (`docs/tq-*.md`), not binary dumps.
- English-only for code comments, logs, and all GitHub artifacts; user-facing docs may be bilingual.
