# TQ 4-bit probe on real checkpoint index vectors

Source: `models/qwen38-27b/vectors.npz`, layers 11, 3, 7, 4096 keys, index dim 768.

Run with top_p 32, top_k 512 (tight budget - the full-config run picks every bucket, since a
4096-key document holds 512 buckets against top_p 1024, and measures nothing).

Metric: mean/min per-query selection set overlap against the fp32 baseline (same buckets config, same queries). Keys dequantized then renormalized.

## layer11 (decoder layer 11)

| bits | hadamard | mean overlap | min overlap |
|---|---|---|---|
| 8 | False | 0.9945 | 0.9756 |
| 8 | True | 0.9880 | 0.9789 |
| 4 | False | 0.9298 | 0.8756 |
| 4 | True | 0.9487 | 0.9163 |
| 2 | False | 0.8708 | 0.8128 |
| 2 | True | 0.8833 | 0.8128 |

## layer3 (decoder layer 3)

| bits | hadamard | mean overlap | min overlap |
|---|---|---|---|
| 8 | False | 0.9909 | 0.9662 |
| 8 | True | 0.9997 | 0.9951 |
| 4 | False | 0.9109 | 0.8756 |
| 4 | True | 0.9333 | 0.8841 |
| 2 | False | 0.8261 | 0.7399 |
| 2 | True | 0.8791 | 0.8373 |

## layer7 (decoder layer 7)

| bits | hadamard | mean overlap | min overlap |
|---|---|---|---|
| 8 | False | 0.9942 | 0.9805 |
| 8 | True | 0.9942 | 0.9805 |
| 4 | False | 0.9558 | 0.9122 |
| 4 | True | 0.9697 | 0.9317 |
| 2 | False | 0.8526 | 0.7902 |
| 2 | True | 0.8815 | 0.8010 |

## Reading

- Index keys are unit rows with no wild outliers, so the Hadamard rotation buys ~2pp at 4/2 bit,
  not the dramatic gains seen on raw attention KV.
- 4-bit keeps ~93-95% of the tight-budget selection; 2-bit ~87-88%. Selection overlap is a proxy -
  the gate is end-task RULER accuracy with the needle retained and attention mass held.
- These numbers cover the *index* path only. The full TQ 4-bit latent-KV probe (review P0-4) also
  quantizes the attention KV itself and runs on the N-LSA 0.8B checkpoint once its download lands.
