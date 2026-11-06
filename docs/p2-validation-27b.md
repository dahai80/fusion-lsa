# P2 validation — LSA serving adapter on Qwen3.8-27B (MLX)

Date: 2026-11-06. Machine: M5 Max 128GB, bf16 weights (~54GB), bf16 KV cache.
Server: fusion-mlx `serve --enable-lsa --lsa-map <map> --kv-cache-dtype bf16 --text-only`
on port 11555; bench: `scripts/ruler32k_bench.py` (single-needle-in-a-haystack, needle at
60% depth, exact-match scoring, streaming decode tps).

## Accuracy

| length (words) | tokens | needle token | result |
|---|---|---|---|
| 4,000 | ~3,385 | ~1,700 | HIT (repeated, 100%) |
| 16,000 | ~13,453 | 8,007 | HIT (repeated, 100%) |
| 32,000 | ~26,773 | 16,021 | HIT (repeated, 100%) |
| 128,000 | 106,782 | 64,068 | HIT |

Selection probe (`LSA_SELECT_LOG`, `LSA_PROBE_LO/HI`): at 32k, ~49 of the 59-token
needle window selected on nearly every decode step; index `key_count` exactly equals
the sequence's closed pages on every run (no cross-request pollution).

Full-attention control (same machine, same prompt, no `--enable-lsa`): 16k 2/2 HIT —
confirms the bench measures retrieval, not harness luck.

## Throughput

| config | 32k decode tps |
|---|---|
| full attention (control) | 5.9 |
| LSA adapter, final | 8.3 |

Decode tps measured first-token to last-token from streaming chunks (prefill excluded).
The early implementation (prefill-time index appends + first-decode rebuild double work)
measured 4.9 tok/s; removing prefill appends (index rebuilt from cache once per request)
lifted it above the dense control.

Prefill: ~155 tok/s at 128k (chunked dense path unchanged). Memory at 128k (106,782
tokens): ~78GB total resident (weights 54 + KV ~27 + workspace), inside the 102GB
(120GB x 0.85) budget gate. fp16 KV tier validated to 128k.

## Bugs found and fixed during validation

1. **Cross-request index pollution** — the per-layer selector kept its bucket index
   across requests; a full prefill after an earlier request appended fresh keys behind
   stale rows, shifting every slot and pushing the needle's bucket outside the visible
   range (fabricated `"1234567890"` at 16k/32k on a warm server, hits on a fresh one).
   Fix: selector reset per request + rebuild from cache keys at first decode
   (fusion-mlx a5ea979d).
2. **Batched-cache crash** — `int(cache.offset)` crashed any B>=2 forward
   (BatchKVCache.offset is a [B] array) before the dense fallback could run; offset now
   passes to RoPE untouched, int coercion only in the single-stream branch (8615b926).
3. **In-place batch join/leave** — mlx_lm extend/filter mutate the cache object, so
   identity-only reset missed batch reshapes; width/row-regression tracking re-arms the
   rebuild (8615b926).

## Known limits

- Selection cost grows with bucket count: 32k 8.3 tok/s, 128k 3.6 tok/s (top_p=1024
  cuts through ~13.3k buckets at 128k). Batched decode is dense (fallback).
- KV cache must stay bf16/fp16: the index reads raw keys; TurboQuant and quantized
  KV dtypes are refused at boot (probe data in docs/tq-kv-probe-27b.md: 8-bit safe,
  4-bit borderline, 2-bit unusable on attention K/V).
