# Design: Sub-Quadratic Long-Context Attention for Qwen3.8-27B on Apple Silicon (working name: project TBD)

Status: DRAFT for review — 2026-11-08
Owner: dahai80
Deliverable: new standalone project (sibling of fusion-lsa), paper-grade method + MLX serving integration

## 1. Intent and Why

The LSA-on-MLX migration (fusion-lsa) proved latent-sparse decode on
Qwen3.8-27B: 24.7 kB/token packed KV, 1M-context prefill reached 91.2%
before hardware limits. But prefill stayed O(T^2) full attention: 13 h
wall for 1M tokens, tail 13-25 tok/s, and three machine deaths in two
days (one swap-storm kernel panic at 105+ GB, two SoC thermal resets)
during full-load runs. Decode-only sparsity is half the product; the
other half is a prefill that does not burn the machine.

This project designs and trains a full-lifecycle sparse attention
(prefill + decode + KV) for one locked model — Qwen3.8-27B — served on
MLX / Apple Silicon unified memory, with the intent to publish.

## 2. Goals / Non-Goals / Anti-Goals

Goals:
- G1 512K-context prefill at >= 150 tok/s on M5 Max 128 GB, measured
  END-TO-END: wall clock / 512k including index/landmark build,
  quantization, and SSD prefix snapshot writes (no side-path exemptions;
  the ~780 s index rebuild at 128k is the cautionary precedent).
- G2 Task-level accuracy: per-category zones. HARD floor >= 0.99
  (ratio to the dense baseline, all 3 repeats) on retrieval/needle
  categories (NIAH / variable-track / FWE); QA-style categories
  (SQuAD/HotpotQA F1) are REPORTED but not floored — dense models do
  not reach 0.99 there either. Dense RULER baselines at all lengths are
  an M0 deliverable (ratio needs an anchor).
- G3 Model locked to Qwen3.8-27B; weights may be modified via training,
  INCLUDING new injected modules (latent projections, landmark scorer).
  Checkpoints are therefore not stock-dense-loadable; dense serving
  remains available only from the original weights file.
- G4 Method is novel enough for a paper (positioned against LSA,
  Quest, MInference, Gemma-style SWA+global hybrids). Single-model
  demonstration (Qwen3.8-27B only) is acceptable; generality beyond one
  model is a stretch goal, not a gate.
- G5 Training may run off-ecosystem (Ascend 950DT, PyTorch/torch_npu);
  serving stays MLX. Weights cross via safetensors; parity oracle gates
  every transfer. Probe-first: if the 950DT probe fails, MLX LoRA on
  the M5 Max becomes the PRIMARY path (accepted, thermal care required).
- G6 Standalone repo, independent of fusion-lsa (may reuse its ideas
  and its torch-parity test pattern). fusion-mlx integration follows the
  fusion-lsa precedent: in-tree patch points, owner-coordinated.
- G7 Thermal safety (quantified A3): the 512k prefill completes TWO
  CONSECUTIVE runs with zero SoC thermal events, phys footprint < 106 GB,
  and zero swap-file growth.
- G8 Decode must not regress below the current LSA level (1.5 tok/s at
  32k, documented A/B); improvement is welcome, regression is a defect.
  New-state structures must be snapshot-restorable via the SSD prefix
  cache (hard requirement, not best-effort).

Non-goals:
- N1 Serving models other than Qwen3.8-27B.
- N2 Contexts beyond 2M tokens (k8v4 packing caps ~2M on 128 GB).
- N3 Multi-GPU / cluster serving.
- N4 Bit-exact parity with dense attention (task-level >= 0.99 suffices).

Anti-goals (failure even if it works):
- A1 Accuracy regressions on needle/retrieval tasks (the 0.99 line is a
  hard floor, not an average — every RULER category must hold).
- A2 A method that only works with non-standard inference stacks; the
  serving path must remain MLX-native.
- A3 Thermal-unsafe serving (the method must reduce, not relocate, the
  sustained-load problem that killed the 1M runs).

## 3. Constraints

- Serving hardware: single M5 Max 128 GB (unified memory).
- Training hardware: single Ascend 950DT (PyTorch + torch_npu, CANN);
  fallback: MLX LoRA on the M5 Max itself.
- All code/comments/logs English; GitHub artifacts English; HF
  downloads via https://hf-mirror.com.
- Fusion monorepo rules: independent git repo, venv shared, port
  registry if it opens a service.
- Existing evidence to build on: k8v4 packed KV (24.7 kB/token),
  Hadamard-rotated 8-bit keys with top-64 agreement 0.99, incremental
  bucket index, SSD prefix cache, torch-parity test harness pattern.

## 4. Method Direction (decided so far)

Three-tier hybrid attention, trained into the model (not training-free):

1. Local sliding window (fine, recent context) — O(T * w).
2. Global landmark tokens (few, exact) — O(T * g).
3. Latent bottleneck projection (d2048 class, LSA-proven) carrying
   long-range gist — O(T * d_latent).

Prefill complexity O(T*(w + g + d_latent)) — at 512K this admits
>= 150 tok/s with margin on M-class GPU. Decode reuses the same
structures (LSA decode is the special case; it may be replaced).

Pass structure DECIDED: single-pass streaming. Landmark selection and
latent-gist accumulation happen online per chunk — compatible with the
existing chunked scheduler and early-TQ-conversion flow, keeps the
memory transient profile flat (the 1M-run lesson), and positions the
method as single-pass in the paper (a differentiator vs multi-pass
selection methods).

Safety net independent of training: rigorous block-max score upper
bounds from the 8-bit Hadamard keys skip provably-zero blocks during
prefill; useful during transition checkpoints before training converges,
and as the paper's exactness baseline.

## 5. Left to the Builder

- Kernel-level design of the three tiers on MLX (Metal-friendly ops),
  single-pass streaming per §4.
- Training recipe: data mix, sequence lengths, landmark count and
  w/g/d_latent sweep, LoRA-first vs deep continue-pretrain staging
  (staging choice driven by the 950DT probe outcome, not preference).
- torch_npu feasibility probe for Qwen3.8-27B + sparse-attention
  training loop (pure standard ops only on the NPU side).
- Project name, repo layout, benchmark harness details.
- Paper outline and target venue.

## 6. Milestones (indicative)

- M0 Spike: feasibility numbers — dense 512k prefill tok/s curve AND
  dense RULER baselines at all eval lengths (G2 ratio anchor),
  torch_npu training smoke on 950DT, LoRA smoke on M5 Max.
- M1 Kernels + parity: three-tier attention in MLX behind tests,
  torch-parity oracle green.
- M2 Training: injection run(s), RULER sweep at each checkpoint.
- M3 Integration: serving path on :11434-class stack, packed KV,
  SSD prefix cache interplay.
- M4 Paper: evals, baselines (dense, LSA, MInference), writeup.
