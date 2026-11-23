# Issue #10 execution record — 950DT win-rate tasks for the HSA paper

Issue: https://github.com/dahai80/fusion-lsa/issues/10
Task list source: `fusion-insight/docs/950dt-winrate-tasks-2026-11-22.md`
Node: Ascend 950DT 91.75GB HBM (10.8.82.152), container `vllm-ascend:v0.30.0-a5`,
model `qwen38-27b-bf16`, `--max-new-tokens 512` everywhere.

Primary artifacts live in the hsa repo under `results/` (paths below). This
document is the fusion-lsa-side index and cross-check of what was executed,
in the issue's suggested order T3 → T4 → T2 → T1 → T5.

## T3 — dense floor for VT/FWE/QA/CWE (done 2026-11-22)

`{vt,fwe,qa,cwe} x {32768,131072} x seeds 41-43`, dense path, greedy, 512 new
tokens. Artifacts: `hsa/results/dt152_20261122/t3/dense_ruler_{task}_{len}.json`.

Result table (dense node vs LSA serving, same calibrated generators):

| task | dense 32K | dense 131K | serving 32K | serving 131K |
|---|---|---|---|---|
| vt  | 2/3 | 1/3 | 0/3 | 0/3 |
| fwe | 3/3 | 3/3 | 2/3 | 0/3 |
| qa  | 3/3 | 3/3 | 3/3 | 3/3 |
| cwe | 2/3 | 3/3 | 3/3 | 0/3 |

Follow-up VT decode-budget probe (`--max-tokens 2048`): 32K serving = strict
0/3 but semantic 3/3 (value re-formatted as arithmetic); 131K serving = 1/3
semantic, hop-1 numeric-anchor line never retrieved. Dense at 2048 = 2/3 both
lengths (`hsa/results/dt152_20261122/vt2048/`). Verdict recorded in
`hsa/results/dt152_20261122/summary.md`: report VT at 512 strict with the
budget probe as analysis.

## T4 — K8 denominator hardening (done 2026-11-22)

NIAH 131072, 3 depths x 3 repeats (seeds 41-49) = 9/9.
Artifact: `hsa/results/dt152_20261122/t4/dense_ruler_131072.json`.

## T2 — Quest baseline gap fill (this PR, 2026-11-23)

Protocol matches `hsa/results/quest_baseline_2026-11-21.md`: tier ckpt_8000,
window 512, block 2048, prefill-chunk 1024, seeds 41-43.

- T2a: 4K / depth 0.95, P in {1, 32, 64} — fills the "---" row in tab:ablation.
- T2b: 32K full-page budgets P = 512 (= 100% of 512 pages) and P = 1024
  (>= page count, selection inactive) — closes the "collapse up to half the
  pages" claim either way.

Results (`hsa/results/dt152_20261122/t2/`, logs `dt152_20261122/quest_ruler_*.log`):

| config | P | hits |
|---|---|---|
| 4096 / depth 0.95 | 1 | 0/3 |
| 4096 / depth 0.95 | 32 | 3/3 |
| 4096 / depth 0.95 | 64 | 3/3 |
| 32768 (full page budget, 512 pages) | 512 | 3/3 |
| 32768 (>= page count) | 1024 | 3/3 |

Reading: 4K depth 0.95 matches the depth 0.75 shape (needs >= ~50% page
budget); at 32K the full-page budget recovers 3/3, so the collapse is
budget-bound exactly as the "up to half the pages" claim states — report both
endpoints, no extrapolated curve needed. Docs updated on the hsa side:
`results/quest_baseline_2026-11-21.md` (2026-11-23 section) and
`results/dt152_20261122/summary.md` (T2 row + section).

## T1 — dense 256K/512K/1M (done 2026-11-22)

- 256K: 1/1 hit, 138s wall (`hsa/results/dt152_20261122/t1/dense_ruler_262144.json`).
- 512K: OOM at prefill-chunk 16384/8192/4096/2048 (77-81GB allocated,
  1.2-4GB request vs 0.6-3.9GB free — not chunk-bound).
- 1M: OOM at 16384/8192/4096, identical signature.
- Measured dense ceiling on 91.75GB = 256K; OOM telemetry replaces the paper's
  extrapolated 512K/1M dense rows.

## T5 — official Quest baseline (done 2026-11-22)

mit-han-lab/quest CUDA kernels not runnable on Ascend; algorithm extracted
line-by-line from kernels and ported to torch ops (hsa commit b24e8bc), parity
vs official semantics 1.8e-07, full-budget decode == dense 1.7e-07. Node
re-measurement (`hsa/results/dt152_20261122/t5q/`): quest full-budget 3/3 at
32K; sparse 25%-of-pages budgets 3/3 at 4K/32K/131K.

## Ops notes (node 10.8.82.152)

- Access: `ssh dt152` (ProxyJump via 139.9.159.231, key auth on both hops).
- `docker logs` on job containers is empty — job output is redirected into
  `/mnt/sfs_turbo/dahai/hsa-quest/logs_g/*.log` by the exec command.
- Containers need `-e PYTHONPATH=/mnt/sfs_turbo/dahai/hsa-quest/src` (stock
  image has no `hsa` installed); missing it surfaced as
  `ModuleNotFoundError: No module named 'hsa'` on the first T2 attempt.
- 2026-11-23: a foreign 8-card megatron job started mid-run and OOM-killed all
  five first-attempt T2 jobs during weight loading (~8GB allocated each,
  ~20-140MB free); it crashed and was removed minutes later. Rerun on verified
  idle cards. Per quest_baseline doc: always re-verify `npu-smi` immediately
  before launching.
