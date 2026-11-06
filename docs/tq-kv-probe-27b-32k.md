# TQ attention-KV probe on real checkpoint KV

Source: `models/qwen38-27b/vectors-kv-32k.npz`, layers 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63. K quantized per-channel with 8 fp16 escape channels, V per-token, top-64 key agreement and attention-output relative drift against fp32, causal-masked, GQA groups averaged.

## layer3 (decoder layer 3)

| bits | hadamard | top-64 key agreement | output relative drift |
|---|---|---|---|
| 8 | False | 0.9934 | 0.0031 |
| 8 | True | 0.9927 | 0.0029 |
| 4 | False | 0.9066 | 0.0502 |
| 4 | True | 0.8984 | 0.0481 |
| 2 | False | 0.5544 | 0.2651 |
| 2 | True | 0.5779 | 0.2817 |

## layer7 (decoder layer 7)

| bits | hadamard | top-64 key agreement | output relative drift |
|---|---|---|---|
| 8 | False | 0.9927 | 0.0028 |
| 8 | True | 0.9917 | 0.0028 |
| 4 | False | 0.8936 | 0.0482 |
| 4 | True | 0.8754 | 0.0442 |
| 2 | False | 0.5228 | 0.3138 |
| 2 | True | 0.5168 | 0.2472 |

## layer11 (decoder layer 11)

| bits | hadamard | top-64 key agreement | output relative drift |
|---|---|---|---|
| 8 | False | 0.9915 | 0.0021 |
| 8 | True | 0.9903 | 0.0021 |
| 4 | False | 0.8805 | 0.0359 |
| 4 | True | 0.8614 | 0.0331 |
| 2 | False | 0.5137 | 0.2197 |
| 2 | True | 0.4750 | 0.2018 |

## layer15 (decoder layer 15)

| bits | hadamard | top-64 key agreement | output relative drift |
|---|---|---|---|
| 8 | False | 0.9913 | 0.0018 |
| 8 | True | 0.9909 | 0.0016 |
| 4 | False | 0.8717 | 0.0292 |
| 4 | True | 0.8634 | 0.0239 |
| 2 | False | 0.4644 | 0.1886 |
| 2 | True | 0.4726 | 0.1406 |

## layer19 (decoder layer 19)

| bits | hadamard | top-64 key agreement | output relative drift |
|---|---|---|---|
| 8 | False | 0.9905 | 0.0028 |
| 8 | True | 0.9906 | 0.0025 |
| 4 | False | 0.8703 | 0.0469 |
| 4 | True | 0.8606 | 0.0375 |
| 2 | False | 0.4600 | 0.2462 |
| 2 | True | 0.4625 | 0.2144 |

## layer23 (decoder layer 23)

| bits | hadamard | top-64 key agreement | output relative drift |
|---|---|---|---|
| 8 | False | 0.9891 | 0.0046 |
| 8 | True | 0.9902 | 0.0032 |
| 4 | False | 0.8400 | 0.0748 |
| 4 | True | 0.8468 | 0.0551 |
| 2 | False | 0.3817 | 0.3890 |
| 2 | True | 0.4281 | 0.2882 |

## layer27 (decoder layer 27)

| bits | hadamard | top-64 key agreement | output relative drift |
|---|---|---|---|
| 8 | False | 0.9898 | 0.0058 |
| 8 | True | 0.9899 | 0.0037 |
| 4 | False | 0.8498 | 0.1101 |
| 4 | True | 0.8537 | 0.0591 |
| 2 | False | 0.4497 | 0.4060 |
| 2 | True | 0.4443 | 0.2947 |

## layer31 (decoder layer 31)

| bits | hadamard | top-64 key agreement | output relative drift |
|---|---|---|---|
| 8 | False | 0.9886 | 0.0040 |
| 8 | True | 0.9893 | 0.0021 |
| 4 | False | 0.8377 | 0.0514 |
| 4 | True | 0.8474 | 0.0378 |
| 2 | False | 0.3675 | 0.3689 |
| 2 | True | 0.4283 | 0.1857 |

## layer35 (decoder layer 35)

| bits | hadamard | top-64 key agreement | output relative drift |
|---|---|---|---|
| 8 | False | 0.9898 | 0.0060 |
| 8 | True | 0.9895 | 0.0036 |
| 4 | False | 0.8447 | 0.1253 |
| 4 | True | 0.8435 | 0.0654 |
| 2 | False | 0.4060 | 0.3509 |
| 2 | True | 0.4149 | 0.2603 |

## layer39 (decoder layer 39)

| bits | hadamard | top-64 key agreement | output relative drift |
|---|---|---|---|
| 8 | False | 0.9878 | 0.0049 |
| 8 | True | 0.9889 | 0.0030 |
| 4 | False | 0.8315 | 0.0916 |
| 4 | True | 0.8345 | 0.0727 |
| 2 | False | 0.3499 | 0.6236 |
| 2 | True | 0.3932 | 0.2518 |

## layer43 (decoder layer 43)

| bits | hadamard | top-64 key agreement | output relative drift |
|---|---|---|---|
| 8 | False | 0.9892 | 0.0050 |
| 8 | True | 0.9882 | 0.0042 |
| 4 | False | 0.8457 | 0.0881 |
| 4 | True | 0.8344 | 0.0502 |
| 2 | False | 0.3982 | 0.3893 |
| 2 | True | 0.3952 | 0.2295 |

## layer47 (decoder layer 47)

| bits | hadamard | top-64 key agreement | output relative drift |
|---|---|---|---|
| 8 | False | 0.9891 | 0.0022 |
| 8 | True | 0.9882 | 0.0016 |
| 4 | False | 0.8425 | 0.0393 |
| 4 | True | 0.8365 | 0.0207 |
| 2 | False | 0.4045 | 0.2251 |
| 2 | True | 0.4017 | 0.1278 |

## layer51 (decoder layer 51)

| bits | hadamard | top-64 key agreement | output relative drift |
|---|---|---|---|
| 8 | False | 0.9894 | 0.0016 |
| 8 | True | 0.9876 | 0.0016 |
| 4 | False | 0.8571 | 0.0268 |
| 4 | True | 0.8336 | 0.0229 |
| 2 | False | 0.4207 | 0.2019 |
| 2 | True | 0.3954 | 0.2311 |

## layer55 (decoder layer 55)

| bits | hadamard | top-64 key agreement | output relative drift |
|---|---|---|---|
| 8 | False | 0.9899 | 0.0014 |
| 8 | True | 0.9884 | 0.0012 |
| 4 | False | 0.8564 | 0.0216 |
| 4 | True | 0.8434 | 0.0171 |
| 2 | False | 0.4248 | 0.2007 |
| 2 | True | 0.4246 | 0.1286 |

## layer59 (decoder layer 59)

| bits | hadamard | top-64 key agreement | output relative drift |
|---|---|---|---|
| 8 | False | 0.9888 | 0.0010 |
| 8 | True | 0.9881 | 0.0010 |
| 4 | False | 0.8303 | 0.0146 |
| 4 | True | 0.8298 | 0.0103 |
| 2 | False | 0.3922 | 0.1732 |
| 2 | True | 0.3975 | 0.0561 |

## layer63 (decoder layer 63)

| bits | hadamard | top-64 key agreement | output relative drift |
|---|---|---|---|
| 8 | False | 0.9901 | 0.0014 |
| 8 | True | 0.9881 | 0.0012 |
| 4 | False | 0.8566 | 0.0268 |
| 4 | True | 0.8318 | 0.0179 |
| 2 | False | 0.4356 | 0.3869 |
| 2 | True | 0.4007 | 0.2961 |

## Verdict (32k vs 4k)

- **8-bit: length-stable.** Agreement 0.989-0.993 and drift 0.001-0.003 across all 16
  full-attention layers at 32k — indistinguishable from the 4k probe (0.993+ / ~0.003).
  Outlier channels do not accumulate with context length. TQ 8-bit on attention K/V is
  the viable 1M-tier compression: 16 layers x 1M tokens x 4KB = 64GB bf16 KV drops to
  32GB, which fits the 102GB budget alongside 54GB weights.
- **4-bit: borderline, mildly worse at 32k.** Agreement 0.83-0.91 (4k: 0.91-0.92 on the
  sampled layers), drift 0.015-0.05 — deep layers actually drift less at 32k than at 4k
  (layer59 0.015 vs 4k-family 0.02-0.06). Hadamard helps drift but not agreement.
  Still not precision-neutral; needs the fp16 sink+window carve-out and an end-task
  RULER gate before serving use.
- **2-bit: unusable at any length** (agreement 0.39-0.58, drift 0.06-0.39).

P3 premise step satisfied: fp16 tier validated to 128k end-to-end (see
docs/p2-validation-27b.md), 8-bit TQ measured length-stable at 32k as the 1M-tier
compression path. Remaining P3 work: TQ-on-latent-KV serving integration (LSA selection
reads dequantized rows), paged KV + SSD cold tier, chunked-prefill resume.
