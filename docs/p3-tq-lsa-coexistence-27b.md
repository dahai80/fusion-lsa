# P3 step 1: TurboQuant k8v4 + LSA serving coexistence (Qwen3.8-27B)

Date: 2026-11-06. Model: Qwen3.8-27B (`~/.fusion-mlx/models/qwen38-27b`), map
`qwen38_27b__long-16k-1M-heavy-key-and-query-d2048`. fusion-mlx branch
`fix/mac-app-compile-pbxproj`, commits `35e1be6`, `f8e7157`, `3f4bc88`.

## What landed

LSA decode can now serve over a TurboQuant k8v4 KV cache. The scheduler
converts the prefilled KVCache to TurboQuant after prefill; the LSA wrapper
detects the swap, rebuilds the selection index from `cache.dequantize()` rows,
and the decode gather attends over dequantized keys/values (the quantized
state proxy only supports contiguous time slices, so per-row gather must
happen in fp16).

Mode gating (per the TQ KV probe on this checkpoint): coexistence is allowed
only in k8v4 mode. 8-bit keys keep top-64 agreement at 0.99; 4-bit keys
measured 0.83-0.91 and would silently degrade the selection index. Changes:

- `K8V4TurboQuantKVCache` in `fusion_mlx/turboquant_kv.py`: the pinned
  mlx_vlm TurboQuantKVCache has a single `bits` knob and quantized keys at
  4 bits while the boot log claimed K=8 — k8v4 never reached the codecs.
  The subclass builds the key codec at 8 bits; SSD codec rebuild honors
  the cache's `key_bits`.
- `serve_command`: `--enable-lsa` + `--kv-cache-turboquant` requires k8v4;
  the quantized-dtype guard no longer fires when TurboQuant owns the KV
  cache (its default `--kv-cache-dtype int4` is skipped downstream anyway).
- Engine gates: auto-enabled TQ still stands down under LSA (auto mode
  would be 4-bit keys); explicit k8v4 — via model settings or the CLI flag,
  which now wins over a settings-file false — enables coexistence.
- `model_settings`: lsa/turboquant mutual exclusion dropped.

## Real-server validation (single-needle RULER, streaming)

Server: `--enable-lsa --lsa-map <map> --kv-cache-turboquant k8v4 --text-only
--no-memory-aware-cache --spec-decode none`, port 11555, fresh boot,
LSA_SELECT_LOG on. "TurboQuant: converted 15/64 cache layers" logged per
request (last layer kept fp16 by `turboquant_skip_last`).

| length | repeats | exact-match | decode tps | selection at needle |
|---|---|---|---|---|
| 4096 | 2 | 2/2 HIT | 2.18 | key_count 3072, picked 3516 |
| 16384 | 2 | 2/2 HIT | 1.85 | key_count 13312, blocks 26, tail 400 |

Unit suite: LSA attention/CLI/patch/selection 30/30, TurboQuant 41/41,
including new tests: k8v4 mode reaches codecs (key_codec.bits == 8), LSA
decode runs on a real converted TurboQuant cache with correct key_count,
flag-over-settings precedence, settings coexistence allowed.

## Decode perf work (same day, commits `2d99afa` worktree / `bd2f985` fusion-lsa)

Profiling the 1.85 tps decode (control dense: 10.2 tps, same flags) found
three per-step costs, all fixed:

- The fp16 dequant row cache grew by `mx.concatenate` every step, re-copying
  the whole history each token (~2 GB/step at 16k over 16 layers). Replaced
  with a preallocated buffer (capacity doubling); writes touch only new rows.
- `select()` synced on a `kept_count` `.item()` per step per layer. `read()`'s
  -1 padding is now clamped to row 0 and masked out of the softmax
  (additive mask), keeping decode GPU-sync free.
- Tail dedupe (`_dedupe_sorted`) synced and was redundant: read picks sit
  inside closed pages and the tail is everything past them (disjoint), and
  duplicate rows would not change the softmax anyway. Removed.
- `IncrementalBucketIndex.buckets()` (fusion-lsa) re-concatenated every
  closed block's arrays on each call — once per decode step per layer. The
  concat is now cached and invalidated on page append.

Correctness held end-to-end after each fix (16k needle, 8/8 HIT across
configs; unit suites 55/55 fusion-mlx LSA/TQ, 47/47 fusion-lsa).

Measurement caveat, ledgered as a ruling: sustained decode on this machine
thermal-throttles hard — the same config measured 4.7 tps mean on a cool
machine and 1.8 tps hot, and hot fp16-LSA (no TQ) measured 1.3. Per-run
decode_tps is only comparable within one back-to-back session, so the e2e
delta of these fixes is unproven; they stand on the removed O(history)-per-
step work plus unit tests. P2's 8.3 tps @32k is not explained by spec-decode
(measured 2.1 with it on); most plausibly a cool machine plus the fp16 path.
Future perf claims must use same-session A/B runs.

## Known limits / follow-ups

- Decode tps dropped vs fp16 LSA serving (8.3 tok/s at 32k in P2): the LSA
  decode path dequantizes the full per-layer cache every step. Next step:
  dequantize only on page boundaries + cache dequantized pages, or gather
  from packed state via a take_along_axis-aware dequant kernel.
- TQ conversion skips the memory-aware cache stack: with the default
  memory-aware cache (`SizedArraysCache`/`ArraysCache` layers) the
  scheduler converts 0 layers and serving silently stays fp16. Pre-existing
  gap for all TQ serving under that config, not LSA-specific; the bench
  above ran with `--no-memory-aware-cache` to get plain KVCaches.
- Boot log prints the stale line "TurboQuant KV cache auto-enabled (4-bit)"
  right after the coexistence line; cosmetic.
- `test_turboquant_kvcache_state.py` (28 cases) fails on the pinned
  mlx-vlm commit (`from_kv_cache` missing) — pre-existing environment
  mismatch, unrelated to this change.
