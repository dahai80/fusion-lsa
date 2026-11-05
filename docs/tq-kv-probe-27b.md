# TQ attention-KV probe on real checkpoint KV

Source: `models/qwen38-27b/vectors-kv.npz`, layers 3, 7, 11. K quantized per-channel with 8 fp16 escape channels, V per-token, top-64 key agreement and attention-output relative drift against fp32, causal-masked, GQA groups averaged.

## layer3 (decoder layer 3)

| bits | hadamard | top-64 key agreement | output relative drift |
|---|---|---|---|
| 8 | False | 0.9944 | 0.0033 |
| 8 | True | 0.9942 | 0.0028 |
| 4 | False | 0.9223 | 0.0608 |
| 4 | True | 0.9148 | 0.0470 |
| 2 | False | 0.6399 | 0.2707 |
| 2 | True | 0.6422 | 0.2514 |

## layer7 (decoder layer 7)

| bits | hadamard | top-64 key agreement | output relative drift |
|---|---|---|---|
| 8 | False | 0.9931 | 0.0035 |
| 8 | True | 0.9927 | 0.0029 |
| 4 | False | 0.9112 | 0.0563 |
| 4 | True | 0.9025 | 0.0481 |
| 2 | False | 0.6079 | 0.3119 |
| 2 | True | 0.5924 | 0.2428 |

## layer11 (decoder layer 11)

| bits | hadamard | top-64 key agreement | output relative drift |
|---|---|---|---|
| 8 | False | 0.9932 | 0.0026 |
| 8 | True | 0.9922 | 0.0023 |
| 4 | False | 0.9100 | 0.0417 |
| 4 | True | 0.8967 | 0.0352 |
| 2 | False | 0.6052 | 0.2486 |
| 2 | True | 0.5709 | 0.1927 |
## Verdict

- **8-bit**: invisible damage (output drift 0.3%, top-64 agreement 99%+). Safe everywhere, including the sink and window.
- **4-bit**: borderline raw - top-64 agreement 91-92%, output drift 4-6%. The Hadamard rotation cuts output drift 20-30% (e.g. layer3 0.061 -> 0.047) even though it does not move key agreement; on unit-index keys it bought ~2pp only, but on attention K/V it is worth keeping.
- **2-bit**: unusable for attention K/V (drift 19-31%, agreement ~60%). Confined to the index path at best, and only with LSA's own error tolerance re-checked end-to-end.

Damage budget for a precision-neutral 4-bit cache (per the KV compression plan): combine per-channel K
with 8 escape channels + per-token V + Hadamard, then (a) keep sink + window in fp16, (b) raise the
escape-channel count or add a MiKV-style second fp16 copy of the top-outlier channels until drift
drops under ~1%, and (c) gate on the end-task RULER score, not on this probe - the probe measures
one attention read, not the model.

Cross-reference: index-path results in docs/tq-probe-27b.md (index keys tolerate 4-bit at 0.93-0.95
selection overlap). Attention K/V is the harder target; index keys are unit rows and quantize better.
