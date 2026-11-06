# SDD ledger — plan: /Users/dahai/fusion/architecture/fusion-lsa-mlx-migration-1105.md (P1 phase)
P0 closed: prior ledger finalized and workspace deleted after clean review (record in git: commits eecc0d7..665ae2f + docs/p0-report.md).
P1 session 2: preflight done — fusion-mlx has patches/glm_moe_dsa (sparse_mla handles kv_latent), cache/latent_cache.py, turboquant_kv; P2 integration point confirmed. GLM d-lsa map (2.8MB) downloaded, SHA verified vs release manifest; structure: 11 indexer layers (3..43 step 4), per-layer W_Q_idx/W_K_idx [512->64]. Converter src/fusion_lsa/convert/map_converter.py landed: .pt -> safetensors with model_arch/indexer_layers/latent_dim/index_dim metadata, fail-loud on arch/latent-dim/layer-range mismatch (review P0-6); project() = rownorm(rows @ W) matching upstream index_space.project. 5/5 tests incl. real map smoke. Commit 90bb391 pushed. N-LSA 0.8B checkpoint (2.3GB) downloading in background for P1 e2e.
P1: Qwen3.8-27B d-lsa map downloaded (6MB, SHA verified), converted to safetensors: 16 indexer layers (3..63 step 4), 768->64, arch qwen38_27b, project() smoke passed. User directive: 27B is the primary validation model.
P1: index_space.py ported (regroup/rownorm/LearnedProjection), 5 parity tests vs torch oracle, fp32 summation tolerance 1e-3. Suite 34/34. Commit pushed.
P1 e2e pending: N-LSA 0.8B checkpoint download in progress (2.3GB).

## P1 e2e prep (2026-11-05)
- scripts/extract_nlsa_vectors.py: torch-side extractor, upstream adapter path (Qwen35Adapter + capture_layer_inputs + torch_reference.build_buckets/read), CPU, fp32, 16k doc, 32 queries, npz out
- deps: hydra-core + omegaconf installed into fusion venv for upstream lsa imports
- tests/test_real_weights_parity.py: MLX build_buckets+read vs torch selected sets (exact), early-budget sanity; skips until vectors.npz exists
- blocked on: aria2c download of decay128k-lsa-1024.tar parts (in progress)
- minors cleared: members_from_labels cap guard (032ccea), bench warmup (c8a651c), early-budget parity test
- Ruling: MLX argsort documented stable (mx.argsort docstring), composite-key hardening unnecessary; matches torch nonzero row-major order

## Qwen3.8-27B e2e (2026-11-05) — MILESTONE
- 27B bf16 downloaded (hf-mirror, 52GB); extractor hardened for MPS (accelerate placeholder workaround, ids/positions/outputs device moves, torch oracle on CPU)
- vectors.npz: layers 3,7,11, 4096 keys, 16 queries, rotary prefix-64 confirms 768-dim index space
- test_real_weights_parity 3/3 GREEN on real 27B vectors: MLX selection == torch reference sets (exact), early budget ok, map 768->64 unit rows ok
- KV cache compression survey: 33 papers analyzed, Word report at ~/fusion/architecture/fusion-kv-cache-compression-plan-1105.docx

## TQ 4bit probe prep (2026-11-06)
- src/fusion_lsa/quant/tq.py: TqQuantizer (2/4/8-bit per-row asymmetric, fp16 escape channels), HadamardRotation (zero-pad to pow-2, 768->1024)
- tests/test_tq_quant.py 7/7 (RED->GREEN: int8 overflow bug caught at 8-bit, fixed to uint8)
- scripts/tq_probe.py: bit-width x hadamard sweep on real vectors; first run caught probe design flaw (512 buckets < top_p 1024 = trivial selection), rerun tight top_p 32/top_k 512
- docs/tq-probe-27b.md: 8bit 0.994 / 4bit 0.930 (+hadamard 0.949) / 2bit 0.871 (0.883) selection overlap
- pending: N-LSA 0.8B checkpoint download (~256M/2.3G) -> full TQ-4bit latent-KV probe per review P0-4
- P0-4 (pivoted): attention-KV TQ probe complete on real Qwen3.8-27B K/V (layers 3/7/11, 4096 tokens, vectors-kv.npz). 8-bit drift 0.3%, 4-bit 4-6% (Hadamard cuts ~25%), 2-bit unusable. Attention K/V needs sink+window fp16 + wider escape channels to go under 1%; gate stays RULER. Commits 0169466, report regenerated.

## P2 session (2026-11-06)
- Ruling: GLM-5.3-Flash-4bit = 190GiB (mlx-community), bf16 = 310GiB — exceeds 128GB unified memory and 203GB free disk. Hard physical boundary. P2 validation pivots to Qwen3.8-27B (local 52GB bf16, user-designated primary validation model): fusion-mlx --enable-lsa --lsa-map adapter + 32k RULER on 27B. GLM latent-KV (1M tier) deferred until loadable weights exist; report annotates honestly. Cost if wrong: 1M capability delayed; 32k LSA serving path still validated e2e.
- GLM map converted to safetensors (models/maps/glm53_flash__...-latent.safetensors): 11 indexer layers 3..43 step4, latent 512 -> index 64, arch glm53_flash (git-ignored artifact).
- P2 task breakdown (branch feat/lsa-qwen35-adapter in fusion-mlx, tests run with PYTHONPATH to repo, install untouched): T1 patch skeleton + registry + pre-load dispatch + armed log; T2 LSA selection glue (imports fusion_lsa: IncrementalBucketIndex + LsaMap, oracle parity test); T3 attention wrapper (prefill incremental build, decode L==1 selected-gather SDPA + sink + window, cache surface intact); T4 CLI flags --enable-lsa --lsa-map + start.sh whitelist + plumbing; T5 e2e serve smoke on 27B + RULER-32k + bench tps.
- Scout findings folded in: patches/__init__.py hard-coded tuple, maybe_apply_pre_load_patches dispatch branch, mlx_lm qwen3_5 Attention = Qwen3NextAttention (partial rope first 64 dims of 256, q/k RMSNorm, q+gate split, KVCache, SDPA), index vectors from pre-rope position-free dims.
- P2-T1 complete (fusion-mlx 7cc2b48): qwen35_lsa patch skeleton, registry + ensure_all_registered tuple, pre-load dispatch on model_type qwen3_5, ModelSettings lsa_enabled/lsa_map_path (+conflict guard vs turboquant), armed log, fail-loud map validation. Tests 7/7 new, regression 516 passed (model_settings/model_loading/registry). Ruff clean.
- P2-T2 complete (fusion-mlx f7c9762): LsaLayerSelector in patches/qwen35_lsa/selection.py — index_keys/index_queries (position-free + regroup + rownorm + learned map), page-boundary incremental append (page = window_size, tail held back), select() decode read via fusion_lsa reference read() with kept_count prefix slice (no boolean indexing in MLX) + tail union on mid-page positions. Tests 4/4: batch-oracle set parity (2048 keys, 19 positions), midpage tail always selected, small-config page semantics, wrong-head-count rejection. Ruling: select() is decode-only L==1 contract; test bugs fixed (test passed full q — row 0 != position row; k[..., :128] sliced dh axis no-op). Regression: tests/unit 16604 passed, 48 failed — all pre-existing at HEAD~1 (ltx2_5/minimax/dflash/batching, timing + config asserts), verified by rerunning at pre-T1 commit (same failures); manifest cascade (dependency_lock checksums on patches/) fixed by regenerating patches.checksums.json, tracked in commit. Ruff clean.
- P2-T3 complete (fusion-mlx a1d607e): LsaAttention wrapper in patches/qwen35_lsa/attention.py — prefill runs dense SDPA while feeding pre-RoPE keys to the incremental index; decode L==1 selects (top-p buckets -> top-k + sink + window), gathers keys/values from cache via take_along_axis, attends with mlx_lm scaled_dot_product_attention (positional signature: queries, keys, values, cache, scale, mask). B>1 decode falls back dense (one selection cannot serve a batch) — Ruling: B>1 per-sequence selection deferred; full KV cache still maintained (cache surface intact, pruning deferred). wrap_lsa_layers binds per-layer LsaMap selectors post-load. Tests 4/4 incl. decode == manual gather parity. Ruling: rope before index_keys is safe — partial rotary leaves free dims untouched. Full-suite regression 49 failed/16607 passed — same pre-existing families (ltx2_5/minimax/dflash/batching/diffusion/vision-cache/g2-freeze) verified at stash baseline.
- P2-T4 complete (fusion-mlx 0ce2976): --enable-lsa/--lsa-map serve flags (dest lsa_map_path — argparse would otherwise yield args.lsa_map), SchedulerConfig lsa fields (MTP channel pattern), BatchedEngine._overlay_scheduler_lsa onto ModelSettings copy pre-dispatch (serve flags win over settings.json; None passthrough when off), serve_command refuses --enable-lsa + --kv-cache-turboquant (exit 2, plain-language), start.sh whitelist both flags (--lsa-map value-taking). Tests 6/6. Targeted regression 818 passed, 5 failed — same failures reproduce with changes stashed (diffusion_engine, g2_freeze_1179, vision_feature_cache x3), pre-existing. Ruff clean.
- P2-T5a complete (fusion-mlx 25b0255): apply_post_load_transforms lsa_enabled branch — _apply_lsa_wrap derives head_dim/kv_heads/rotary/latent from text_config (latent = kv*(head_dim-rotary) = 768 for 27B), loads map via LsaMap(arch qwen38_27b) fail-loud, wraps named full-attention layers; BatchedEngine post-load now receives _overlay_scheduler_lsa() settings (serve flags reach wrap too). Tests 8/8 cli+wiring; targeted load-path regression 129 passed/0 failed. Map confirmed compatible: 16 indexer layers 3..63 step4, latent 768, index 64.
- P2-T5b e2e smoke (fusion-mlx 4b987f6): real Qwen3.8-27B bf16 serve on :11555 with --enable-lsa --lsa-map (32k map, 16 indexer layers) — armed log + "LSA wrapped 16 full-attention layers [3..63]" + first generation returns "OK" through the LSA decode path. Boot pitfalls fixed: (1) kv-cache-dtype defaults int4 — LSA needs raw keys, serve with --kv-cache-dtype bf16; (2) auto-enabled TurboQuant quantizes the rows the window gathers — _turboquant_should_enable extracted, stands down when scheduler LSA armed, mirrored in _turboquant_kv_will_enable; (3) scheduler cache offsets surface as 0-d arrays — coerced int in wrapper + select; (4) stale server on the port served old code for two rounds — verify lsof before blaming code. Smoke cmd: serve models/qwen38-27b --enable-lsa --lsa-map ... --text-only --kv-cache-dtype bf16 --spec-decode none.

## T5b e2e validation — RULER NIAH 4k/16k/32k (2026-11-06)

Task: Ruling: control arm first, then attribution — full-attention control hit 2/2 at 16k
while LSA missed, so the miss was LSA-attributable, not harness.
- Control 16k needed TurboQuant auto-enable off (no mlx_vlm installed); used the designed
  override channel `~/.fusion-mlx/model_settings.json` key
  `--Users--dahai--.fusion-mlx--models--qwen38-27b` with `turboquant_kv_enabled=false`
  (canonical id preserves dots; first attempt with a wrong key silently no-oped).
- Length sweep + selection probe (LSA_SELECT_LOG, LSA_PROBE_LO/HI) localized the failure:
  `LsaLayerSelector` kept its bucket index across requests. A full prefill after an earlier
  request appended fresh keys behind stale rows; every slot shifted by the stale count.
  Evidence: decode `key_count=37376` on a ~26.7k-token prompt (clean = 26624); 32k on a
  polluted index missed exactly as predicted, clean index hit.
- Fix (fusion-mlx a5ea979d): reset selector on cache-identity change; first decode rebuilds
  the index from cache keys sliced to offset (buffer is preallocated/zero-padded). Post-rope
  keys are valid index input because RoPE touches only the rotary prefix.
- Result: 4k/16k/32k sequential sweep 100% exact-match; probe shows ~49/59 needle-window
  tokens selected per step at 32k.
- Perf (plan risk #1): 32k decode 4.9 tok/s (LSA) vs 5.9 tok/s (full attention) on M5 Max
  bf16 — LSA 17% slower; per-step selection overhead (argsort over bucket scores, .item()
  syncs, gather) exceeds read savings at this scale. Deferred optimization, not P2 scope.
- Tests: fusion-mlx LSA unit 26/26 (2 new: reset, radix-rebuild); bench committed to
  fusion-lsa 56ffb68 with streaming decode tps (fixes the meaningless wall/tps readings).
- Final: Ruling: whole-branch reviewer dispatched on Sonnet, not the most capable model —
  session cost already critical (~$69, hook-flagged); Sonnet with the full diff + ledger +
  plan brief is adequate for a 1.2k-line adapter diff — cost if wrong: a subtle finding
  the cheaper model misses slips to production; mitigated by the e2e sweep evidence.

## Final review pass (2026-11-06)

Reviewer: fresh-context subagent (Sonnet) over merge-base 328e5890..HEAD, package at
fusion-mlx/.superpowers/sdd/progress/review-328e5890..c9ce2960.diff.

Findings and grading:
- C1 int(cache.offset) crashed every B>=2 forward (BatchKVCache.offset is a [B] array;
  dense fallback unreachable) — fixed, commit 8615b926: offset passed to RoPE untouched,
  int coercion only in the single-stream selection branch. Tests: batched-cache dense run.
- I1 in-place extend/filter bypass identity-only reset — fixed in the same commit:
  _observe_cache tracks batch width and row regressions, re-arms rebuild. Tests:
  extend/filter re-arm.
- I2 rebuild churn when prefill and decode interleave on different caches — Ruling:
  accepted for P2; correctness holds (rebuild is always safe), the cost is per-flip
  index rebuild under concurrent prefill+decode. Single-stream is the P2 target; the
  prefill append removal already removed the double work (32k decode 4.9 -> 8.3 tok/s,
  now above the 5.9 full-attention control). Cost if wrong: concurrent-mixed-workload
  decode tps degrades toward dense; revisit with per-request selector state in P3.
- I3 quantized --kv-cache-dtype + LSA crashed at first decode — fixed: boot guard
  refuses the combination (serve_command).
- Final: minor (deferred): M1 partial_rotary_factor should read text_config
  rope_parameters (today the 0.25 default coincides with the checkpoint value; a
  checkpoint with a different factor would mis-slice free dims).
- Final: minor (deferred): M2 --enable-lsa silently ignored on VLM/other load paths
  (fusion_mlx_vlm / fusion_mlx_lm call apply_post_load_transforms without the overlay).
- Final: minor (deferred): M3 last_picked lands in the nn.Module parameter tree;
  _state_cache pins the last request's caches until replaced.

Post-fix e2e sanity: 16k and 32k NIAH exact-match hit, key_count exactly 13312/26624,
decode 8.3-8.4 tok/s at 32k. LSA unit suite 29/29. Manifest refreshed, committed.

Note: fusion-mlx branch pointer moved mid-session (another session landed mac-app fixes
c9ce2960/46c9f709 on the same line; branch now fix/mac-app-compile-pbxproj). LSA commits
are linear on that branch: 7cc2b483..a5ea979d..8615b926. Branch naming left to the user.

## P3 premise step 1: fp16 KV to 128k (2026-11-06)

- 128k-word NIAH (106,782 tokens, needle at token 64068) on LSA server, bf16 KV:
  exact-match HIT, index key_count exactly 106,496 (no pollution), selection ran every
  step. Prefill ~10 min (~155 tok/s); decode 3.6 tok/s (top_p now cuts 13.3k buckets —
  selection cost grows with length, expected).
- Memory: system free 39% of 128GB with server resident (~78GB used total incl. weights
  ~54GB + KV ~27GB + workspace) — inside the 102GB (120GB x 0.85) hard gate. fp16 tier
  validated to 128k; TQ-on-latent-KV outlier study and SSD cold tier remain for the 1M
  tier.
- Ruling: 1M-tier work (paged KV + SSD offload + chunked-prefill resume) NOT started in
  this session — cost state critical (~$80) and P3 is a 1-2 week multi-component effort
  needing a fresh session. P4 docs partially done: fusion-mlx README LSA section added.
- Final: minor (deferred): CHANGELOG not updated for the LSA branch — repo convention
  ties entries to version bumps + issue numbers (0.10.50 style); the LSA work has no
  issue number yet. File the issue first, then the entry.

## P3 premise step 2: TQ outlier behavior at 32k (2026-11-06)

- Re-extracted real attention K/V at 32k from the 27B checkpoint (torch/MPS forward,
  ~55 min; fast path unavailable without flash-linear-attention), probed all 16
  full-attention layers (docs/tq-kv-probe-27b-32k.md; intermediate 3.3GB npz removed).
- 8-bit TQ is length-stable: agreement 0.989-0.993, drift <=0.003 at every layer —
  matches 4k. 4-bit borderline (agreement 0.83-0.91), 2-bit unusable. No outlier
  accumulation with length.
- P3 path fixed: fp16 KV to 128k validated end-to-end; 8-bit TQ as the 1M compression
  (64GB bf16 KV -> 32GB, fits 102GB budget). Remaining P3: TQ serving integration
  (LSA reads dequantized rows), paged KV + SSD cold tier, chunked-prefill resume.
- Fixed tq_kv_probe.py report-writer syntax bug (stray paren made a string call).

## P3 step 1: TQ k8v4 + LSA serving coexistence (2026-11-06)

Scope: fusion-mlx branch `fix/mac-app-compile-pbxproj` (LSA line now tip 3f4bc88; another session organized the LSA commits onto this branch and rebased to v0.10.50 main + checksum regen; `feat/lsa-qwen35-adapter` exists but stale at a5ea979d).

Shipped (commits 35e1be6, f8e7157, 3f4bc88):
- K8V4TurboQuantKVCache: k8v4 mode never reached the codecs on the pinned mlx_vlm (single bits knob, keys quantized at 4 bits while boot log claimed K=8). Subclass builds key codec at 8 bits; SSD codec rebuild honors key_bits; sched convert/empty sites use make_turboquant_kv_cache helper.
- LSA wrapper TQ-aware: rebuild reads cache.dequantize() rows (proxy only supports contiguous time slices, gather must be fp16); decode gather dequantize-then-attend; _observe_cache sentinel for shapeless quantized states.
- Guards: serve requires k8v4 for LSA+TQ; quantized-dtype guard skipped when TQ owns the cache; CLI flag counts as operator-explicit (wins over settings-file false); engine auto-TQ still stands down under LSA; model_settings mutual exclusion dropped.

E2E (Qwen3.8-27B, real server, fresh boot, TQ conversion live "converted 15/64 cache layers"):
- 4k 2/2 HIT (decode 2.18 tps), 16k 2/2 HIT (decode 1.85 tps). Selection index exact: key_count 3072/13312.
- Unit: LSA suites 30/30, TQ suite 41/41 on pinned mlx_vlm (git commit f96138e, installed via gh-proxy mirror — github.com unreachable directly; PyPI mlx-vlm 0.7.12 lacks the #1244 masked-decode fix AND regressed it).

Rulings:
- Ruling: coexistence gated to k8v4 only — 4-bit K probe agreement 0.83-0.91 breaks selection; v4 mode today quantizes keys at 4 bits despite "K stays FP16" boot log (pre-existing mislabel, separate issue).
- Ruling: full per-step dequantize accepted for this step — correctness first; tps 8.3→~1.9 at 16k. Next P3 perf step: page-granular dequant caching.

Findings (new, ledgered):
- TQ conversion skips memory-aware cache stacks (ArraysCache/SizedArraysCache): converts 0 layers, serving silently fp16. Pre-existing, affects all TQ serving under default settings; bench used --no-memory-aware-cache.
- Boot log contradiction: "auto-enabled (4-bit)" line after k8v4 coexistence line. Cosmetic.
- test_turboquant_kvcache_state.py 28 failures pre-exist on clean tree (pinned mlx_vlm lacks from_kv_cache) — env mismatch, not a regression.
- Repo env drift found: venv had stale non-editable fusion-mlx 0.10.40; fixed with pip install -e (now 0.10.50 editable). mlx stayed 0.31.2 per requirements.lock (mlx-vlm pip install attempted to bump to 0.32.4 — blocked, --no-deps used).

Artifacts: docs/p3-tq-lsa-coexistence-27b.md (this repo). Next P3 steps: page-granular dequant caching (perf), ArraysCache TQ conversion gap, then chunked-prefill/paged-KV scoping.

## P3 perf step (2026-11-06, commits 2d99afa fusion-mlx worktree / bd2f985 fusion-lsa)

Perf investigation of LSA decode (16k needle, control dense 10.2 tps same flags):
- Ruled out: LSA_SELECT_LOG (1.58 without vs 1.85 with), spec-decode (ON measured 2.1 — hurts, not explains P2 8.3), full per-step dequantize (+9% only once fixed).
- Root causes found and fixed (fusion-mlx worktree 2d99afa):
  1. Concat-growth of fp16 dequant rows re-copied whole history every step (~2 GB/step at 16k over 16 layers) — replaced with preallocated buffer, capacity doubling, buffer writes touch only new rows.
  2. kept_count `.item()` sync per step per layer — replaced with clamp + additive mask on read()'s -1 padding (no GPU sync).
  3. Tail `_dedupe_sorted` synced and was a no-op safeguard (read picks disjoint from tail; duplicates don't change softmax) — removed.
  4. IncrementalBucketIndex.buckets() re-concatenated all closed blocks every step (fusion-lsa bd2f985) — concat cached, invalidated on append.
- Unit: fusion-mlx LSA/TQ suites 55/55; fusion-lsa suite 47/47 (+3 skip). Full fusion-mlx unit run: 16658 pass, 3 fail (G5 vendor checksum + 2 fusion_gui — unrelated, pre-existing).

E2E rebench (16k needle, all HIT): baseline 1.85 → buffer-fix 2.47 → sync-free 4.68 mean (runs 2.0-8). BUT follow-up A/B showed thermal throttling dominates: hot-machine reruns gave TQ 3.2/1.3/0.9 (mean 1.8) and hot fp16 1.3 — same range as pre-fix baseline. Per-run decode_tps on this machine is not comparable across time; only same-back-to-back runs are.

Ruling: e2e tps deltas for these fixes are unproven (thermal noise > effect size); the fixes stand on the unit tests and the removed O(history)-per-step work (logically real, ~2 GB/step copies + 16 syncs/step). Any future perf claim must use same-session A/B, interleaved, or a non-thermally-bound metric.

Reconciling P2's 8.3 tps @32k: not spec-decode (measured 2.1 with it ON); most plausibly a cool machine + fp16 path (no TQ dequant buffers). Untestable today — ledger as open question.

Artifacts: docs/p3-tq-lsa-coexistence-27b.md updated with perf section.

## P3 ArraysCache gap re-check (2026-11-06, after 2d99afa)

Re-tested on the worktree build with default memory-aware cache (no --no-memory-aware-cache flag), Qwen3.8-27B, k8v4 + LSA:
- First request (fresh make_cache list, cache_type=ArraysCache): "TurboQuant: converted 15/64" — conversion live.
- Repeat requests via prefix-cache restore (cache_type=SizedArraysCache): still "converted 15/64", 3/3 needle HIT.

Ruling: the earlier finding "memory-aware cache stack silently converts 0 layers" does NOT reproduce on this build — conversion works on both fresh and restored caches. Previous session's observation was likely a misread of a different failure (that boot may have lacked the make_turboquant_kv_cache refactor or had a radix-hit signature mismatch). Gap entry withdrawn; --no-memory-aware-cache is not required for LSA+TQ serving. Cost if wrong: someone keeps passing an unnecessary flag — harmless.

Also noted: hot-machine repeats measured 0.8-0.9 tps at 4k (thermal), first request 5.3 — more thermal evidence, consistent with the thermal ruling.

## P3 1M-memory blocker removed (2026-11-06, worktree commit 4631939a)

Memory math: holding a full fp16 dequantized copy of the KV cache costs
K+V = 2 arrays x 4 kv heads x 256 dim x 2 bytes = 4 GB per full-attention
layer at 1M tokens; 16 LSA layers = 64 GB — the 1M path could not carry
the wrapper's dequant row cache. Fixed by gathering picked rows directly
from the packed TurboQuant MSE state:
- _gather_dequantize: take_along_axis on the time axis of norms [B,H,T]
  and indices [B,H,T,G], dequantize only the picked rows (exactly matches
  full dequantize, unit-tested); exposed on K8V4TurboQuantKVCache and
  BatchTurboQuantKVCache (server wraps caches in the Batch class — first
  e2e failed with AttributeError, caught by real-server validation).
- Rebuild: chunked dequantize (4096 rows/chunk) feeds the selection index;
  rows discarded — no whole-history fp16 copy at rebuild either.
- Decode: per-step dequantize of picked rows only (read width, not cache
  size); no row buffer state remains on the wrapper.

Tests: LSA+TQ suites 52/52 (new: gather matches full dequantize; decode
never calls full cache.dequantize after rebuild). E2E 16k needle 2/2 HIT,
1.8 tps hot steady-state (thermal caveat applies). Ruling: full cache
dequantize after rebuild is now a test-enforced regression.

Remaining P3: chunked-prefill checkpoint/resume, memory budget hard gate,
1M prefill acceptance run (compute-heavy, deferred to a cool-machine window).

## Env drift incident (2026-11-06 16:27)

A background `pip install mlx-vlm` task from earlier in the session completed late and silently replaced the pinned git install (f96138e, v0.5.0) with PyPI 0.7.12 — the version that regressed the #1244 masked-decode RHT fix. All e2e/unit results above ran BEFORE that install landed (task completed after the last bench at 16:16), so results stand. Reinstalled the pinned commit from the gh-proxy tarball (--no-deps); LSA/TQ suites 56/56 green after restore. Rule: no background pip installs against the shared venv; verify `pip show mlx-vlm` = 0.5.0 before TQ work.

## P3 #4: memory budget hard gate — complete (2026-11-06)

Worktree `fusion-mlx-lsa-wt`, commit `d4d71664`.

- `patches/qwen35_lsa/budget.py`: `arm_lsa_memory_budget` — TQ-accurate KV
  estimate (24832 B/token for Qwen3.8-27B: `(4*256*1.5 + 4*4) * 16`), doc
  ceiling unified-memory x 0.85 unless a memory enforcer already set one,
  guard `_prefill_memory_guard = True`. Reuses existing MemoryMonitor
  override + PrefillMemoryExceededError handler.
- Monitor model info missing at engine-setup time (scheduler init runs
  before config populated; `model.config` absent, `model.args` = dict with
  `text_config` dict). Fallback `_extract_dims` handles dict/namespace
  configs, nested text_config, make_cache KVCache layer counting.
  Ruling: arm extracts dims itself instead of patching scheduler init —
  surgical, no init-order changes; cost: duplicated probe logic (~40 lines).
- Engine tq_mode fix: `batched.py` now uses `_effective_turboquant_mode()`
  (scheduler_config CLI flag first). Prior e2e "k8v4" runs actually served
  4-bit v4 keys; HITs unaffected (4-bit agreement 0.83-0.91).
- Verified live on Qwen3.8-27B :11555: `TurboQuant K8V4 mode: keys
  quantized at 8 bits` + `LSA memory budget armed: ceiling 108.8 GB, KV
  ~24832 B/token, weights 50.1 GB, max context ~2538048 tokens`; conversion
  `mode=k8v4` (15/64 layers); 0 guard rejections; 4k needle 2/2 HIT,
  accuracy 1.0.
- Tests: `test_qwen35_lsa_budget.py` 11/11 (incl. dict-args fallback +
  hybrid make_cache counting), cli 12/12, budget math asserts 1M fits
  (2.5M headroom). Full LSA/TQ unit set 74/74.

## P3 #3: chunked prefill + prefix cache reuse — partial (2026-11-06)

Worktree `fusion-mlx-lsa-wt`, commit `e51bd0bd`.

Existing machinery verified, no new build needed:
- Chunked prefill exists (sched_batch `_step_prefill_chunk`); LSA page
  growth already closes whole pages only (selection.append_keys).
- Prefix cache reuse works end-to-end for plain LSA fp16: 16k bench
  repeat 2 hit 12288 cached tokens, wall 32.5s -> 17.4s, needle 2/2 HIT.
- Index rebuild on prefix restore is unit-covered (radix hit, extend/
  filter re-arm) and chunked at 4096 rows.

Bug found + fixed (`e51bd0bd`): BatchTurboQuantKVCache.merge lost the
K8V4 subclass and extract(idx) (mlx_lm BatchGenerator, request
completion) returned base TurboQuantKVCache, so k8v4 blocks were stored
with "TurboQuantKVCache" metadata and every later prefix lookup
mismatched the K8V4 expected signature. merge propagates key_bits;
extract rebuilds the concrete subclass. Unit test RED->GREEN
(test_batch_extract_preserves_k8v4_subclass); TQ+LSA suites 76/76.

OPEN (next session): under k8v4 the boundary-snapshot/SSD store path
drops the 15 TQ layers - blocks stored with num_layers=49 vs expected 64
("Cache layer count mismatch: block has 49 layers"), prefix hits
invalidated, cached_tokens stays 0. fp16 unaffected. Evidence:
/tmp/lsa-tq-prefix.log (18:13-18:16). The block store path handles TQ
packed state ("__turboquant_v2__", TurboQuantMSEState in file metadata)
but the boundary snapshot capture/serialize path skips TQ layers as
"sliceable". Fix = serialize TQ norms+indices in boundary snapshots and
restore them as K8V4 caches. Both signature and layer-count guards
correctly refuse the broken blocks - no corruption served.

## P3 #3 continued: k8v4 block store/restore (2026-11-06, commits `6e46c45b`)

- Instrumented store path: "Extracted 64 layer states from 64 live
  caches" — extraction fine. Block metadata was inconsistent:
  num_layers=49 but layer_cache_types had 64 entries (48 ArraysCache +
  15 K8V4 + 1 KVCache). Root cause: prefix_cache block store/restore
  branches match TQ by exact names; K8V4TurboQuantKVCache missed all
  three sites, so the 15 packed layers were dropped from block DATA
  while types stayed 64. Fixed (all 3 sites) + shape-consistency guard
  in TQ restore (mixed v4/k8v4 packed widths refuse instead of crash).
- After fix, clean-slate k8v4 run: block stores 64 layers; one repeat
  hit 12288 cached_tokens earlier (18:34 run) proving reuse plumbing
  works when blocks match.
- NEW BLOCKER (open): with TQ layers restored, continued prefill
  crashes in ArraysCache (linear-attention) layers — mlx_lm
  update_and_fetch concatenate (1,4,12288,512) vs (1,4,1536,256).
  Restored linear-layer state width 512 vs live chunk 256. Prefix
  reuse under k8v4 blocked on this; fp16 path unaffected (fp16 reuse
  verified: 12288 cached tokens, wall 32.5s -> 17.4s). Traceback:
  qwen35_model.py -> attention.py:123 update_and_fetch. Next session:
  compare restored ArraysCache state layout vs live append shape
  (suspect store/restore packs linear state differently than the live
  fused-recurrent op expects).
- Stale SSD blocks from pre-fix v4/fp16 runs poisoned lookups (same
  seed-42 prompt hashes); purged /Users/dahai/.fusion-mlx/ssd_cache
  (regenerable process data).
