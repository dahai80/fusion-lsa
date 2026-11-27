# Kimi-Linear-48B on Ascend 950DT — spec-decode campaign acceptance (issue #25)

Status of the three acceptance gaps from
[dahai80/fusion-lsa#25](https://github.com/dahai80/fusion-lsa/issues/25) that gate the
"Kimi-Linear-48B-A3B-Instruct served losslessly on 950DT" premise before it is used
downstream (HSA paper Tianshu track, R5 128K column).

Campaign context: the 2026-11-26 spec-decode repair campaign (summary deck
`Kimi48-950DT-spec-decode-修复战役总结-20261126.pptx`, artifacts in the
`vllm-ascend-pr` workspace under `container-v032/artifacts-20261126/`) landed a
four-layer patched stack — two torch bypasses in the GDN prefill path, a per-step
block NaN sanitizer in the MLA decode path, and an FIA manual-attention fallback —
validated 10/10 effectiveness + 18/18 stress, spec-decode ON + prefix caching ON,
0 Traceback. Upstream issues vllm-ascend #23399 / #23400 and fla_npu #1419 carry the
three stacked root causes.

## Gap 1 — Long-context lossless validation

**Problem.** All campaign validation was short-prompt. The fp32 sequential-recursion
bypass (`KIMI_KDA_TORCH_PREFILL=1`) is capped at T<=256, and the fla_npu `chunk_kda`
kernel's per-token output path is structurally wrong (~100% relative deviation,
fla_npu #1419). Nothing demonstrated losslessness beyond 256 prefill tokens.

**Fix landed.** A new exact fp32 chunked bypass `_torch_chunk_kda`
(`KIMI_KDA_TORCH_PREFILL_CHUNK=1`, wired ahead of the capped sequential path in
`kimi_k3_patched.py`) removes the 256-token cap for all prefill lengths:

- validated against the sequential recurrence at 1e-5 (bf16 rounding level);
- bit-exact against the fla `naive_chunk_kda` oracle (6/6 fp32 cases, dev probes
  under `container-v032/artifacts-20261126/`);
- three implementation bugs fixed en route (chunk permute order, state update
  using raw `v_n` instead of `v_i`, output reassembly permute).

**Acceptance run (2026-11-27, official RULER harness, niah, 1 sample, greedy,
spec ON + chunked bypass).** Served via the host-level vllm-ascend stack on the
950DT node (see "Deployment note" below); server accepted every request (200 OK,
no truncation, KV cache 3.7M tokens).

| Length | niah_single_1 (spec ON) | niah_single_1 (spec OFF) | Verdict |
| ------ | ----------------------- | ------------------------ | ------- |
| 4K     | 100.0                   | 100.0                    | pass    |
| 8K     | 0.0                     | 0.0                      | FAIL    |
| 32K    | 33.3                    | 33.3                     | FAIL    |
| 131K   | 33.3                    | not run (spec isolated)  | FAIL    |

Spec OFF reproduces the spec-ON curve exactly, so speculative decoding is ruled
out as a variable: the failure belongs to the chunked-bypass serving path itself.

**Verdict (superseded 2026-11-27 late, see "Root-cause correction" below):
FAIL as run, but the failure is NOT the chunked-bypass serving path.** The
original attribution below — "KDA recurrent state handed across chunk
boundaries is corrupted" — was falsified by the follow-up bisect on the same
day: the harness failures are caused by vLLM prefix caching reusing mamba/KDA
state across *different* prompts that share the RULER essay-corpus prefix, not
by any cross-chunk state bug in the patched path.

**Root-cause correction (2026-11-27, issue #26).** Evidence chain:

1. Single-request replay of the exact harness prompts (same jsonl records,
   same sampling params, same openai client, same tunnel) returned the correct
   needle every time, including for records the harness had just scored 0.
2. The official harness runs `call_api.py --threads 4`: concurrent requests.
   Three different 32K records (golds `8361662` / `6361026` / `4120696`, needle
   at different depths) all produced the *identical* prediction — record 1's
   needle `6361026`; vt and fwe showed the same collapse to one shared output.
3. The server log showed `enable_prefix_caching=True` plus
   `Mamba cache mode is set to 'align' for KimiLinearForCausalLM by default
   when prefix caching is enabled` — all three records share the same
   deterministic essay corpus (same seed, needle at a different depth), so the
   cached mamba/KDA state of whichever request filled the prefix first is
   reused for the others.
4. With `--no-enable-prefix-caching` the harness numbers recover fully
   (single-sample per length to avoid the concurrent-batch crash, see below):
   8K **0.0 → 100.0**, 32K **33.3 → 100.0** (fresh seeds 43/44; the 32K
   prediction verified against gold `8610642`, exact needle hit, no
   truncation). Long-context losslessness of the chunked-bypass serving path
   stands once prefix caching is disabled; the Gap 1 verdict therefore flips
   from FAIL to PASS-under-`--no-enable-prefix-caching` (footnote any paper
   numbers with that flag).

Parity probes added in this branch pin the bypass itself: chained 2048-token
scheduler slices match the full-length sequential pass at fp32 rounding level
(`serving_slice_chain`), and an n_seq=2 multi-sequence batch matches the
per-sequence sequential oracle at 7e-7 (`tests/test_kda_multi_seq.py`) — the
chunked bypass keeps concurrent sequences' states apart.

Related new defect filed separately as issue #28: concurrent traffic (RULER
`--threads 4` is enough) can form a mixed prefill+decode batch, which the
Ascend Kimi GDN path rejects with `NotImplementedError` at
`vllm_ascend/models/kimi_k3.py:988`, killing the whole engine. Long-context
numbers in this doc must be produced single-request (or with a scheduler fix)
until that is addressed.

**Original (incorrect) attribution, kept for the record:** the 4K/8K cliff was
read as corrupted cross-chunk KDA state, with a hallucinated 32K needle as
supporting evidence. The hallucinated needle was in fact record 1's answer
leaking through the prefix-cache reuse described above; per-chunk and
cross-chunk bypass numerics were never the problem.

## Gap 2 — Performance debt quantified

**Problem.** Layer 2 said "correct but debug-level slow" with no number; the four
patch layers all sit in the hot path.

**Acceptance run.** Same workload, spec ON vs spec OFF, on the final stack
(`/v1/chat/completions`, greedy, server-reported token counts):

| Case | Spec ON | Spec OFF |
| ---- | ------- | -------- |
| prefill ~2K prompt, e2e tok/s   | 33.7  | 12.7 |
| prefill ~8K prompt, e2e tok/s   | 31.0  | 8.0  |
| decode steady-state tok/s       | 20.2  | 15.1 |

Spec ON buys +165% / +288% end-to-end on the two prefill cases and +34%
steady-state decode — a real win, on top of identical long-context (broken)
behavior, so ngram speculation itself is lossless in this stack.

(`perf_kimi.py`, greedy, server-reported counts, run twice; second warm run shown.
First cold run was 19.7/18.2/21.3 — warm cache dominates the e2e numbers.)

Even three numbers settle whether spec ON currently buys anything besides
correctness, and the payoff ceiling of the upstream fixes (#23399 / #23400 / #1419).

## Gap 3 — T<=256 bypass cap: boundary behavior

**Problem.** The deck documented the cap but not the branch: beyond 256 tokens the
patched model silently routed prefill back to the known-wrong `chunk_kda` kernel —
exactly the silent-failure class the campaign otherwise refuses.

**Fix landed** (in `kimi_k3_patched.py`, `container-v032/artifacts-20261126/final/`):

- **Cap rationale**: the sequential bypass is a per-token Python fp32 loop; its
  cost is O(T) per prefill in eager Python, so it is capped at 256 tokens where it
  stays cheaper than the (wrong) kernel. It is a debug/exactness fallback, not a
  serving path.
- **Behavior above the cap**: previously a silent fallback to fla_npu `chunk_kda`.
  Now loud: a `KIMITORCHCAP` warning logs the prefill length and states that
  outputs are NOT trustworthy (fla_npu #1419) whenever `KIMI_KDA_TORCH_PREFILL=1`
  and prefill exceeds 256 tokens.
- **Cap removal**: with `KIMI_KDA_TORCH_PREFILL_CHUNK=1` the exact chunked bypass
  handles all lengths, so production serving never takes the capped branch at all.

## Evidence and reproducibility

**Deployment note (2026-11-27).** The container path is dead on the 950DT node:
`TsdOpen failed. tdt error=1` on every card under both dockerd 24.0.9/27.5.1 with
the full flag matrix (`--privileged --network host --ipc=host --pid=host
--cgroupns=host --security-opt apparmor=unconfined`, proven host CANN mounted) —
the dms/vascend kernel policy rejects NPU open from container namespaces on this
virtualized (HuaweiCloud) node, while host processes attach fine. Serving was
pivoted to a host-level stack that replicates the validated container image
exactly: image python3.12.16 + site-packages (torch 2.10.0+cpu, torch_npu
2.10.0.post7, fla_npu a5 26.9.1) extracted to `/data/lsa_serve/py312`, image-built
vllm 0.32.0 + vllm_ascend 0.32.0rc2.dev48 editable checkouts at
`/data/lsa_serve/workspace`, r3_train CANN 9.2.1 reused read-only, image
`libstdc++`/`libbz2` compat libs in `/data/lsa_serve/syslib2`, patches applied to
the same two files. Scripts: `/root/setup/env_lsa_serve.sh`,
`/root/setup/host_serve_kimi.sh` (cards 6,7, TP2, port 8100), local copies under
`container-v032/artifacts-20261126/final/` (`env_lsa_serve.sh`,
`host_serve_kimi.sh`, `finish_nospec_host.sh`, `hostserve_stage*.sh`).

- Patched serving files: `container-v032/artifacts-20261126/final/kimi_k3_patched.py`,
  `mla_v1.py` (per-step block NaN sanitizer + manual-attention fallback).
- Serve scripts: `serve_kimi_torch_spec.sh` (spec ON) and
  `serve_kimi_chunk_nospec.sh` (spec OFF control), both 131072 max-model-len,
  `KIMI_KDA_TORCH_PREFILL_CHUNK=1`.
- Node-side acceptance harness: `perf_kimi.py` (Gap 2) and
  `scripts/ruler32k_bench.py --task niah` (Gap 1, from this repo), orchestrated by
  `night_run.sh` / `night_ruler.sh` with markers in `/root/klogs/night*.log`.
- Raw logs: `/root/klogs/night.log`, `/root/klogs/night_ruler.log`,
  `/klogs/serve.log` inside the `kimi-spec` container.
