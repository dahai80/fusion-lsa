# Latent Sparse Attention on MLX — Migration Experience (LSA × fusion-mlx)

Status: P0–P4 delivered, 1M-token prefill at 90% evidence (2026-11-07)
Scope: Qwen3.8-27B on Apple Silicon (M5 Max 128 GB), MLX backend
Audience: engineers porting LSA to other backends (e.g. Ascend NPU), and anyone integrating sparse attention into a serving engine

## 1. Problem and Approach

Full-attention KV cache in fp16 is the memory wall for long-context serving on
unified-memory machines. For Qwen3.8-27B (64 layers, 64 KV-cache layers of
head_dim 256 in the wrapped subset) fp16 KV costs ~64 kB/token; a 1M-token
context needs 64 GB of KV alone, on top of ~50 GB of weights.

LSA (Latent Sparse Attention) attacks the decode side: at each decode step,
instead of attending to all T history tokens, a small selection index picks a
window of relevant rows, and only those rows are gathered and dequantized.
Sparsity applies to DECODE ONLY — prefill is still full attention per chunk
(see Pitfall 6). The KV cache itself is stored packed with TurboQuant k8v4
(8-bit keys, 4-bit values): ~24.7 kB/token, a 2.6x reduction, and the packed
rows are what the gather reads — no full dequantize ever materializes.

## 2. Architecture

- **Wrapped layers**: LSA wraps 16 of the model's full-attention layers
  (Qwen3.8-27B: layers 3,7,...,63). Each wrapped layer carries a selection
  index built from the packed keys (8-bit keys chosen deliberately: probe
  showed top-64 agreement 0.99 at 8 bits vs 0.83–0.91 at 4 bits — 4-bit keys
  degrade selection and are refused for LSA serving).
- **Selection index**: per-layer, rebuilt from dequantized key rows. At 128k
  the rebuild + SSD block writes cost ~780 s — the dominant prefill-side term
  after early conversion removed the fp16 KV growth (see Open Items).
- **Decode read path**: per step, index picks top rows, `gather_dequantize`
  materializes only the window, attention runs over the window. Decode cost
  at 32k measured ~1.5 tok/s vs 6.8 tok/s full attention (same-session A/B)
  — the trade is deliberate: memory footprint and prefill feasibility at
  128k+ are the product, decode throughput at short context is not.
- **SSD prefix cache**: packed cache states are stored/restored across
  requests in block-aligned snapshots (16k repeats hit `cached_tokens` with
  full accuracy; 32k wall 132 s fresh vs 56 s restored).
- **Memory budget gate**: at boot, `arm_lsa_memory_budget` computes a
  TQ-accurate KV bytes/token and a unified-memory x 0.85 ceiling; the
  budget-derived max context feeds the API context preflight so long requests
  are not rejected by the tokenizer's native cap (262144) before the model
  ever sees them.

## 3. Integration Surface in fusion-mlx

All integration lives behind five patch points — none fork the model:

1. **Attention patch** (`patches/turboquant_attention.py`): replaces
   `scaled_dot_product_attention` so TQ caches route to quantized kernels
   (decode / prefill fast path / chunked quantized attention / dequant+SDPA
   fallback). Without it, `update_and_fetch` returns quantized state proxies
   that stock SDPA cannot subscript.
2. **Coexistence gate** (`_turboquant_should_enable`): LSA + TurboQuant only
   with explicitly requested k8v4; auto-enabled 4-bit TQ stands down because
   it would quantize the very rows the selection index reads.
3. **Early TQ conversion** (`sched_batch._maybe_early_tq_convert`): quantize
   the KV cache after the FIRST prefill chunk, not post-prefill. This single
   change took the 1M run from a hard-tier abort at 424k tokens to 986k:
   fp16 prefill KV was growing at 64 kB/token until the old epilogue ran.
   An outlier gate probes the first-chunk sample and keeps the request fp16
   if key outliers are detected (per-layer model property; negative decision
   is final — #1423).
4. **Memory budget gate** (§2) + **context preflight**
   (`helpers.get_model_max_context`): budget-derived context overrides the
   tokenizer cap; dict-shaped model configs need explicit probing.
5. **Prefix-cache restore**: block restore must rebuild the cache class that
   matches how the state was packed (k8v4 keys are 8-bit packed: head_dim 256
   -> packed index width 64; a plain-cache rebuild infers bits=4 and reads
   head_dim 512 — silent corruption, fixed in `rebuild_dequantized_kv`).

## 4. Migration Playbook (what a new backend, e.g. Ascend, must provide)

Ports are kernel-level, not algorithm-level. Checklist in dependency order:

1. **Packed-state container**: quantize/append for K (8-bit prod-codec with
   random Hadamard transform) and V (4-bit MSE codebook), plus
   `state`/`offset` semantics the scheduler already drives.
2. **Decode kernel**: single-token quantized attention reading packed rows
   directly (no dequantize). This is the hot path — every step, every layer.
3. **Gather-dequantize**: materialize an arbitrary row subset (the LSA
   window), not the full cache.
4. **Chunked prefill attention**: quantized keys/values with per-key-chunk
   online softmax (`max_score`/`normalizer` rescale, `mx.eval` per chunk to
   bound in-flight buffers). Chunk size is a memory/throughput knob: 16384
   -> 8192 halved the deep-context transient (~28 GB over a 75 GB live set)
   that tripped the memory enforcer at 986k tokens.
5. **Prefix restore**: class-aware rebuild from packed state (§3.5).
6. **Memory measurement discipline**: the enforcer acts on
   `max(active, phys_footprint)`; in-flight kernel buffers show up in phys
   before active. Per-chunk logging of active/peak/cache/phys
   (`[memcheck:chunk]`, gate must count restored prefix via `base_size`) is
   what made every diagnosis above possible — port it FIRST, not last.
7. **Enforcer interplay**: transient spikes are not exhaustion. The
   sustained-soft busy-eviction threshold is env-tunable
   (`FUSION_MLX_SOFT_BUSY_EVICT_POLLS`, default 10 x 1 s polls); deep-context
   prefill needs widened patience or a lower transient. Post-eviction
   footprint (65.8–77.1 GB) vs eviction reading (>102 GB) is the signature
   that distinguishes the two.

## 5. Pitfalls (each with evidence)

1. **Fused single-token quantize kernel corrupts power-of-two head dims**
   (upstream Neywa-Labs/mlx-vlm#2686): rel_err 1.54/1.46 at head_dim 128/256
   via single-token appends vs 0.076/0.096 multi-token; codec-level roundtrip
   is fine — the bug is in the fused Metal kernels. Workaround: dense-codec
   guard disabling RHT rotation for power-of-two head dims (#1230). Lesson:
   repro through the full cache path, not the codec in isolation; port
   must re-verify kernel numerics per head_dim.
2. **Post-prefill quantization epilogue is a trap**: any "convert when
   prefill finishes" design pays full fp16 KV for the entire prefill. Convert
   at the first chunk boundary (§3.3).
3. **Restore must be class-aware**: packed width encodes bits; a naive
   rebuild infers the wrong head_dim silently (§3.5).
4. **Selection index quality bounds everything**: 4-bit keys look fine until
   you probe top-64 agreement (0.83–0.91) — quantize keys at 8 bits whenever
   an index reads them.
5. **Enforcer vs transients**: a 1 s-poll memory enforcer evicting BUSY
   models on sustained soft pressure will kill long prefills that are
   healthy (chunk-side phys 89–103 GB vs 75.4 GB active). Instrument
   phys-vs-active per chunk before touching tiers.
6. **Sparsity is decode-only**: prefill stays O(T²) full attention per chunk
   (~10 h wall for 1M tokens on M5 Max; tail decays to 13–25 tok/s). A true
   1M product needs sparse prefill — separate design round.
7. **Measurement hygiene**: decode tok/s only comparable same-session on a
   cool machine (thermal throttling); first request carries index-build
   warmup (32k wall 111.6 s vs 60.8 s steady).

## 6. Measured Results

Qwen3.8-27B, M5 Max 128 GB, k8v4 + LSA:

| Item | Result |
|---|---|
| Packed KV rate | ~24.7 kB/token vs 64 kB/token fp16 (2.6x) |
| RULER single-needle (needle at end) | 4k/32k/128k HIT; completed-run accuracy 5/5 |
| Prefix reuse | 32k wall 132 s fresh vs 56 s restored, `cached_tokens` confirmed |
| Decode A/B at 32k (same session) | 1.5 tok/s LSA+k8v4 vs 6.8 tok/s full attention (~4.6x decode cost) |
| Prefill memory (1M run) | active flat 50.4 -> 75.4 GB at 986k tokens, exact packed rate, zero hard-tier touches |
| 1M prefill progress | 986k/1095k (90%) — blocked by enforcer eviction, not memory (mitigation landed, rerun in flight) |
| Early conversion prefill speed | ~4.6x faster than fp16 in linear phase (600 vs 131 tok/s) |

## 7. Open Items

- 1M final 4%: rerun with halved key chunk (8192) + widened enforcer
  patience (`FUSION_MLX_SOFT_BUSY_EVICT_POLLS=120`) — instrumentation
  (`[tq-prefill-route]`, `[memcheck:chunk]` with peak) landed first.
- Sparse prefill (LSA read path during prefill): required for true 1M
  serving; needs its own design round.
- LSA index rebuild at 128k (~780 s) now dominates prefill-side wall time.
- Deferred minors: partial_rotary_factor probe, `--enable-lsa` no-op audit
  on VLM load paths, `_state_cache` pinning, rebuild churn.

## 7. Open Items

TODO
