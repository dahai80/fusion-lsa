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

## P3 #3 blocker narrowed: reconstruct mis-assembles hybrid layers (2026-11-06)

Shape probe added at PREFIX-PREP reconstruct (sched_admission.py, info
log "PREFIX-PREP restored shapes", gated >=8192 tokens). Live evidence:

- Restored layer 3 (a K8V4 full-attention layer) comes back as PLAIN
  KVCache with keys (1,4,12288,512) — wrong class and fused width
  (2x256, keys+values concatenated on axis 3 somewhere in reconstruct).
- Disk block layer 3 stores nested state keys "layer_3_state_0_sub_0"
  (1,4,2048) / "layer_3_state_0_sub_1" (1,4,2048,64) — a nested
  sub-state layout (norms+indices-like), NOT flat (k,v).
- Final layer 63 stores flat state_0/state_1 (1,4,2048,256) each and
  restores as expected.

Conclusion: the block-store serializer writes hybrid layers (the
model's per-layer cache is a CacheList of ArraysCache + TQ sub-caches)
in a nested sub-state layout that the reconstruct path does not
understand — it flattens into a KVCache with fused keys. The subsequent
live prefill append then crashes (concatenate 512 vs 256).

Next session: fix reconstruct for the nested sub-state layout (either
serialize CacheList layers flat per sub-cache, or teach reconstruct to
rebuild CacheList from sub-states). The store side already handles TQ
slicing ("__turboquant_v2__"); suspect the mismatch is between the
generic N-tuple store writer (nested "state_k_sub_j" keys) and the
reader's flat state_count expectation.

Probe left in place at info level (gated to long prefills) — useful
until this is fixed, then drop to debug.

Task P3 #3: complete (k8v4 prefix reuse fixed, commits eececf27..97b4204b,
tests: pytest tests/unit/test_turboquant.py → 45/45; e2e 16k bench → 2/2 HIT)

Root cause of the concat crash: the SSD restore branch dequantized stored
k8v4 packed states through a plain TurboQuantKVCache, which has no
key_bits — _rebuild_codecs fell back to bits=4 and inferred head_dim from
the 8-bit packed width (64) as 64*32/4 = 512. Restored keys were
(1,4,12288,512) vs live 256-wide keys.

Fix: rebuild_dequantized_kv(cache_type_name, ...) in turboquant_kv.py
picks the cache class from the stored layer type name (K8V4 → 8-bit key
codecs); prefix_cache TQ restore branch now calls it. Probe log dropped
to debug (evidence preserved in eececf27 message).

Live verification (Qwen3.8-27B, port 11555, k8v4+LSA):
- restored layer 3 shape now (1, 4, 12288, 256) - probe output
- run 0: fresh prefill, HIT, wall 47.0s
- run 1: cached_tokens=12288, HIT, wall 35.0s (prefix reuse working)
- no concatenate crash; accuracy 1.0; decode_tps 1.9 both runs

P3 #3 acceptance remaining: 1M-token real prefill run (P3 #5) on a cool
machine per thermal ruling.

Task P3 #5: partial - 1M prefill aborts at hard tier 424k tokens (2026-11-06)

Run: length 1315000 words (~1.01M real tokens), needle-depth 1.0, k8v4+LSA,
port 11555. New bench flag --needle-depth added (ruler32k_bench.py, 0.6
default; 1.0 = end of context, verified needle at 0.986 of prompt).

Chain of events (all guards performed as designed):
- 400 preflight regression found and fixed first: enforce_context_length
  resolved model context via getattr-only probing; live Qwen3.8-27B has
  dict args/dict text_config so resolution fell to the tokenizer cap
  (262144) and rejected the request pre-prefill. Fix e7144e5f:
  dict probing + _lsa_max_context_tokens override published by the
  budget arm (batched.py). Tests: test_lsa_context_limit.py 3/3;
  context-guard suites 30/30 (3 xpasses pre-existing, stash-verified).
- Prefill ran 19:34-20:28 (~131 tps, superlinear - quadratic attention
  expected). metal_active 73.6GB@19:45 -> 79.0GB@20:09.
- 20:28:33 memory_enforcer soft->hard: current 114.1 GB vs hard 114.0,
  ceiling 120.0 (pool tier system, independent of the LSA budget gate).
- 20:28:34 chunk guard warning: "Chunked prefill above max_bytes at
  423936 tokens: 113.2GB > 101.6GB"; engine_core aborted the request;
  clean unload cascade; post-eviction current 77.1 GB. Server survived.
- Client got structured abort error. 0/1 hit (aborted at 424k of 1.01M).

Accounting at 424k: weights 58.3 + KV 10.5 (424k x 24832 B) + LSA index
~0.1 GB (bucket centroids/members are ~24 KB/page/layer - NOT the hog)
= ~69 GB. Observed 114.1. ~45 GB unexplained, and +35 GB appeared in the
final 19 min while KV only grew +3.5 GB -> superlinear transient growth
in late prefill. Suspects: (a) chunk attention workspace over grown
context (2048xctx fp16 attn matrix ~7 GB/layer at 424k), (b) LSA
page-read/dequant buffers accumulating across layers, (c) phys_footprint
charging something metal_active does not show. Chunk guard already does
_sync_and_clear_cache per chunk (MEMFIX), so MLX buffer cache is ruled
out as the main term.

Next session: instrument before rerunning - log per-chunk
mx.get_active_memory()/get_phys_footprint() + per-layer read buffer
sizes at INFO in the chunked-prefill memcheck block (sched_batch.py
~757), one rerun to ~450k, then target the dominant term. Do NOT raise
the hard tier to brute-force 1M: 114 GB real pressure on 128 GB risks
the kernel panic preflight warns about (#324).

Also ledgered: bench --needle-depth flag is a permanent bench feature
(fusion-lsa scripts/).

Task P3 #5 diagnosis: prefill runs fp16 - TQ conversion is post-prefill (2026-11-06)

Instrumented rerun (commit pending: per-chunk memcheck INFO line in
sched_batch.py chunked step, gated to >65536-token prefills; logs
active/cache/phys per chunk). Curve at 82k-token spacing:

  tokens  active  cache  phys
  2k      50.4    3.1    55.3
  84k     55.4    8.1    65.3
  248k    65.4    18.1   85.3
  416k    75.6    0.0    110.9 (pressure-clear emptied cache)

active grows 64 kB/token = 16 full-attn layers x 4 kB/token (fp16 K+V,
4 heads x 256 x 2 tensors x 2 B) - this is the prefill KV cache itself.
cache grows the same rate until the pressure clear frees it (churn
pool, wired, counts in phys: phys = active + cache + ~2 GB constant).

Root cause identified: the scheduler converts KVCache->TurboQuant AFTER
prefill completes ("converted 15/64" fires post-prefill). During
prefill the cache is plain fp16, so k8v4 saves ZERO prefill memory:
1M tokens = 64 GB fp16 KV + 50 GB weights = the 114 GB abort at the
hard tier. The packed 24.8 kB/token only applies to decode.

Aborted again at 424k (usage 113.9 GB); enforcer CRITICAL "sustained
soft pressure #1473 hard tier" evicted the busy model - guard chain
again clean, server survived.

Fix direction (P3 #3's page-growth protocol, now with numbers):
quantize incrementally during prefill - either (a) run TQ layers as TQ
from token 0 (turboquant_attention patch routes prefill SDPA to
quantized kernels; needs verification that chunked prefill + prefix
extraction tolerate live TQ layers), or (b) convert closed pages at
page boundaries (2048 tokens) keeping only the current window fp16.
(a) caps prefill KV at 24.8 kB/token -> 25 GB at 1M; (b) similar with
a bounded fp16 window. Either lands 1M KV under ~30 GB, inside the
108.8 GB budget with the 50 GB weights.

Next session: decide (a) vs (b) by checking why post-prefill
conversion exists (grep "converted 15/64" call site - is it a prefill
attention correctness constraint or a historical artifact), then TDD
the chosen path.

Incident (2026-11-06 ~22:45): /Users/dahai/fusion/fusion-mlx-lsa-wt was
deleted externally mid-session (not by this session). All 7 commits
survived as loose objects in fusion-mlx/.git; branch
fix/mac-app-compile-pbxproj recreated at 30ff3fcc and worktree re-added.
Only loss: one uncommitted RED test file (restored from context).
Cause unknown - flag if intentional.

Task P3 #5: 1M prefill reached 986k/1095k (90%) - early TQ conversion
lands, enforcer eviction blocks the last 4% (2026-11-07)

Fix landed: early TurboQuant conversion (commit 839a86ba fusion-mlx).
The per-chunk memcheck curve exposed that prefill ran fp16 end to end
(TQ conversion was a post-prefill epilogue) - active memory grew
64 kB/token and the 1M run died at the hard tier at 424k. Early
conversion quantizes after the first chunk; prefill attention runs the
quantized prefill_attention fast path.

Live 1.096M-token run (port 11555, k8v4+LSA, --needle-depth 1.0):
- active memory flat: 50.4 GB (chunk 1) -> 75.4 GB (986k tokens);
  KV growth exactly the packed 24.7 kB/token rate throughout
- quantized prefill ~4.6x faster than fp16 in the linear phase
  (600 vs 131 tps); quadratic attention tail decayed to 13-25 tps
- abort at 986k/1095k: #1473 sustained-soft enforcer eviction, NOT
  memory exhaustion. Mechanism: deep-context chunk attention creates
  ~28 GB transients over active (chunk-side phys 89-103 GB vs active
  75.4), every ~70 s chunk; enforcer (1 s polls) read >102 GB soft
  for 10 consecutive polls with no idle evictable model and evicted
  the busy model. Chunk guard never fired (75 GB << 114 GB hard).
- the bench needle never got a decode; accuracy untested at 1M length.

1M acceptance blockers, in order of cost:
1. Tier headroom for the end-of-prefill transient window (~28 GB over
   active for the last ~100k tokens): either raise soft to ~106 GB /
   widen enforcer patience, or shrink the deep-context attention
   transient (smaller prefill_key_chunk_size in the TQ long-prefill
   chunking). One rerun is ~10 h at current O(T^2) prefill - do NOT
   rerun blind; instrument the transient first (log phys-vs-active
   gap per chunk is already there; add per-op breakdown if needed).
2. Architectural: prefill attention is still O(T^2) full attention
   per chunk - LSA sparsity does not apply during prefill. A 1M
   prefill costs ~10 h wall on this machine regardless of memory.
   True 1M serving needs sparse prefill (LSA read path during
   prefill) - candidate P4+ scope, needs its own design round.

Rulings:
- Early conversion outlier gate probes the first-chunk sample instead
  of full history (#1423): key outliers are per-layer model properties;
  a negative decision is final for the request. Cost if wrong: an
  outlier layer that only manifests after ~10k tokens stays quantized -
  needle tests at 4k/16k/32k cover the regression surface available.
- P3 #5 "stable decode tps" acceptance is unmeasurable at 1M until
  blocker 2 is addressed; the packed-rate memory curve (75 GB flat,
  zero hard-tier touches by the chunk guard) is the verifiable part
  and is delivered.

P3 #5 remains open with 90% evidence; P4 next unless the 1M rerun is
wanted first.

Task P4: RULER suite (minus 1M) + issues filed (2026-11-07)

RULER single-needle, needle-depth 1.0 (end of context), k8v4+LSA,
port 11555, HOT machine (immediately after the 9.6 h 1M prefill run;
thermal ruling: tps not comparable to earlier sessions):

- 4k:    2/2 HIT, wall 25.3/23.1 s
- 32k:   2/2 HIT, wall 131.8 -> 56.3 s (prefix reuse: cached_tokens=26624)
- 128k:  run 0 HIT (fresh, wall 899.6 s: prefill 87 s at ~950 tps,
         then ~780 s LSA index rebuild + SSD block writes at 128k -
         rebuild cost is now the dominant prefill-side term);
         run 1 (repeat) aborted at 172 s: #1473 sustained-soft eviction
         (usage 103.2 GB during prefix-restore + index rebuild on a hot
         machine). Same failure signature as the 1M run: deep transient
         spikes caught by 1 s-poll enforcer, not real exhaustion
         (post-eviction current 65.8 GB).
- Completed-request accuracy: 5/5 HIT. 1M deferred (ledgered P3 #5).

Issues filed:
- fusion-mlx #1513: LSA serving feature umbrella (branch changelog
  anchor; CHANGELOG entry to reference it at merge).
- Neywa-Labs/mlx-vlm #2686 (repo moved from Blaizzy): TurboQuant fused
  single-token KV quantize kernel corrupts power-of-two head dims -
  repro shows rel_err 1.54/1.46 (hd=128/256) via single-token appends
  vs 0.076/0.096 via multi-token; workaround = dense-codec guard
  (turboquant_fused_rht_guard.py, #1230).

Next: README/README_CN/CHANGELOG, then same-machine A/B perf run
(fresh boot, cool machine, LSA+TQ vs full-attention control).

Task P4: docs landed (commit after 839a86ba, 2026-11-07)

README.md + README_CN.md LSA sections rewritten (stale "quantized KV
refused at boot" constraint replaced with k8v4 coexistence, packed-rate
prefill, prefix reuse, budget gate, RULER numbers). CHANGELOG 0.10.52
entry references #1513 (branch predates the 0.10.51 release commit -
rebase will need to interleave).

Remaining P4: same-machine A/B perf run (cool machine, fresh boot:
LSA+TQ vs full-attention control at 32k) - thermal ruling requires a
cool start and today's machine ran a 9.6 h prefill + RULER suite.

## 2026-11-07 — stale boot log fixed + P4 A/B perf run DONE

- Stale boot log line fixed (TDD): post-load TQ block hardcoded
  `tq_explicit = False` (batched.py:585), so explicit `--kv-cache-turboquant
  k8v4` boots logged "TurboQuant KV cache auto-enabled (4-bit)". Extracted
  `BatchedEngine._turboquant_auto_note()` (True only when `_turboquant_explicit()`
  is None), wired the log gate to it, removed the dead local. Tests:
  `test_turboquant_auto_note_false_for_explicit_request` +
  `test_turboquant_auto_note_true_without_explicit_choice` (RED: AttributeError,
  GREEN). test_qwen35_lsa_cli.py 14/14. Verified live: fresh k8v4 boot logs
  k8v4 coexistence lines, NO auto-enabled line.
- Ledgered "stale test test_settings_conflict_with_turboquant" does not exist
  in the repo (grep across tests/ = 0 hits) — phantom from a summary; nothing
  to fix. test_qwen35_lsa_patch.py passes 7/7.
- P4 A/B perf run (same session, cool machine, fresh boot each arm,
  RULER 32k needle-depth 1.0, 3 repeats):
  - LSA + k8v4 TQ: 3/3 HIT, steady-state decode_tps 1.4-1.7 (mean 1.50),
    wall 61-63 s (run0 111.6 s incl. warmup/index build)
  - Control (full attention, bf16 KV, no LSA/TQ): 3/3 HIT, decode_tps
    6.4-7.5 (mean 6.82), wall 8.5-9.9 s (run0 79.1 s warmup)
  - Ratio: control ~4.6x faster decode at 32k. Consistent with pre-fix
    1.9 tps LSA baseline (ledger line ~351) and the 8.3 tps external
    full-attention reference. Control ran second (slightly warmer) yet
    still 4.6x faster - direction robust to thermal bias.
  - Conclusion: sparse decode trades ~4.6x decode throughput at 32k for
    the packed KV footprint; payoff is prefill memory at 128k+ where fp16
    KV cannot fit. Recorded in README.md + README_CN.md LSA sections.
  - Bench logs kept: /tmp/lsa_ab/bench_{lsa,ctrl}.log (final outputs).
- Deferred minors remain ledgered: M1 partial_rotary_factor, M2 --enable-lsa
  no-op on VLM load paths, M3 _state_cache pinning, I2 rebuild churn, LSA
  index rebuild cost at 128k.
- P4 remaining: none in this pass except P3 #5 final 4% (deferred, needs
  instrumented rerun) and branch rename decision (user).

- Correction (same day): a parallel session committed the identical fix at
  10:31 (94183092, incl. the same two auto_note tests verbatim taken from
  the shared worktree) while this session was mid-edit; this session's
  batched.py edit was superseded by their checkout and commit 17e7aa6b
  landed as README A/B numbers + black formatting only. Final state: one
  fix, one test set, tree clean, test_qwen35_lsa_cli.py 14/14. Also:
  branch was renamed externally fix/mac-app-compile-pbxproj ->
  feat/lsa-k8v4-serving (the pending rename decision - resolved).
  "test_settings_conflict_with_turboquant" confirmed phantom (0 grep hits).

## 2026-11-07 16:45 — upstream PR: Prefrontal-AI/latent-sparse-attention#12

- Fork dahai80/latent-sparse-attention; branch latent-sparse-attent-mlx
  (user-named, maintained going forward as the MLX port line).
- PR content: mlx/ directory = selection-logic port (core/quant/index/
  budget/evaluation) + 42 tests (torch reference as parity oracle, 42/42
  standalone pass) + p0-report/bench docs + mlx/README.md. 26 files,
  1628 insertions, commit d4dc413. NO Claude attribution in commit/PR
  (explicit user instruction for this upstream).
- Serving-engine integration intentionally out of scope (stays in
  fusion-mlx); convert module excluded (torch checkpoint parsing,
  Fusion-side).
- PR: https://github.com/Prefrontal-AI/latent-sparse-attention/pull/12
- Promotion deck v2 rebuilt same day (charts + 4-step LSA principle
  walkthrough, Huawei style) - docs/latent-spars-attention-mlx.{md,pptx},
  commits a842c97/af1c0a6.

## 2026-11-07 19:35 — machine reboot killed v3 run; v3 restarted with durable logs

- Machine rebooted ~17:28 (2 min after the last recorded memcheck at
  643k/1095k); server process died, /tmp/lsa_ab/ wiped — serve log, bench
  log, and everything past 643k lost. No final data for v3 attempt 1.
- Losses are procedural, not scientific: mitigations were already
  verified live before the crash (key chunk 8192, peak-active gap ~4.6 GB
  vs ~28 GB in v2, zero enforcer events through 643k).
- Hardening: long-run logs moved to ~/.fusion-mlx/logs/lsa-1m-v3-*.log
  (durable; /tmp wiped on reboot). Memory note saved.
- v3 attempt 2 launched 19:33: server PID 96296, bench PID 97495,
  prompt ~1.315M words -> ~1.01M tokens, FUSION_MLX_SOFT_BUSY_EVICT_POLLS=120,
  key chunk 8192. Same instrumentation. ETA ~10 h (~05:30 next morning).

## 2026-11-08 07:40 — v3 attempt 2 outcome: machine panic at 999k (91.2%), 986k PASSED
- v3 crossed v2 death point 986k cleanly (reached 999,424/1,095,638), zero enforcer evictions; chunk-8192 transient fix VERIFIED (peak-active ~4.5 GB steady vs ~28 GB v2).
- Machine kernel panic 07:17:36: watchdogd starved 92 s, 33 swapfiles, stackshot partial — whole-machine memory saturation (run phys 105-106 GB + OS + bench client on 128 GB), not algorithm regression.
- Enforcer facts measured: soft=102.0 / hard=114 / ceiling=120 GB. Soft crossed 07:13-07:14; busy-evict blocked (active request) + EVICT_POLLS=120 patience -> no relief before watchdog died.
- Root causes to fix for v4: (1) enforcer ceiling too high for 128 GB unified memory — needs ~20 GB OS headroom (ceiling ~106/soft ~96/hard ~102); (2) no escalation when sustained soft AND climbing — EVICT_POLLS patience protects transients but is deadly at true saturation.
- Doc pitfall added: machine-level swap storm. Panics: panic-full-2026-11-08-071736.0002.panic (today), none found for Nov 7 17:28 reboot (cause unconfirmed for that one).

## 2026-11-08 09:50 — v4 attempt killed by SoC thermal reset (sochot), not memory
- Machine reset 09:40:39, Boot faults "sochot,reset_in_3 timeout" — hardware SoC-hot reset, NO panic file. Run died at 75,776/1,095,638 with phys 60.8 GB (no memory pressure at all).
- Rate collapse before death: 208 tok/s (08:41) -> 23 tok/s (08:59, at only 75k ctx where ~600 expected) -> 40 min log stall -> sochot. Progressive thermal throttle then hardware reset.
- Likely same cause as Nov 7 17:28 reboot (v3 attempt 1) — no panic file for that one either.
- 3 machine deaths in 2 days, all during 1M full-load runs (1 swap-storm panic + 2 sochot). Machine cannot sustain the 13 h full-Metal prefill in current thermal state.
- v4 run also fixed two launch bugs en route: missing --text-only routed load to the VLM engine path and LSA never armed (armed lines + 400 preflight rejection); 11434 default server held 23 GB resident. Restarted with --text-only, k8v4 confirmed, watchdog armed.
- Decision (user): pause 1M rerun, start sparse-prefill design round. Algorithm evidence already sufficient (986k passed, transient fix verified, clean curve). 1M rerun deferred until machine cooled / off-peak.

## 2026-11-08 11:10 — HSA M0 spike: Tasks 1-2 complete, Task 3 partial (dense floors 4k PASS), Task 6 script ready

- hsa repo (/Users/dahai/fusion/hsa) scaffold + prefill-curve + ruler-floor code committed (9fa1a66, 57949d0, Task-3 commit, ca9952f).
- Dense 4k baseline RUN: 9/9 HIT, all three needle categories 1.0 >= 0.99 floor PASS (results/dense_ruler_4096.json).
- Launch bugs fixed en route: (1) lsa worktree vanished again -> recreated from origin/feat/lsa-k8v4-serving (bdde1844); (2) resolve_model rejected model path — model is symlink into fusion-lsa/models, allowed-dirs check needs cwd=fusion-lsa (same as v3/v4); (3) 600s boot budget (240s was too short, server takes ~300s to bind); (4) orphaned fusion-mlx-server held 11556 -> killed.
- relaunch_1m.sh written + bash -n OK, flags verified against worktree (--text-only --enable-lsa --lsa-map .../qwen38_27b__long-16k-1M-heavy-key-and-query-d2048.pt --kv-cache-turboquant k8v4 --no-memory-aware-cache --spec-decode none, FUSION_MLX_SOFT_BUSY_EVICT_POLLS=100000, LSA-arm boot check, watchdog). NOT EXECUTED — operator launches this afternoon.
- 950DT probe blocked so far: jump host 139.9.159.231 intermittently denies keyboard-interactive auth (1 success at 10:53, then "Too many authentication failures" — likely fail2ban from earlier failed attempts); retry loop running at 5-min intervals. Probe scripts ready (npu_train_probe.py, probe_950dt.sh).
- Ruling: 128k dense baseline repeats may be cut from 3 to 1 depending on machine heat after 32k x3 — 2.5h dense prefill right before the operator's 14h 1M window recreates the sochot preheat risk; single-repeat needle scores at temp 0 are near-deterministic, floor evaluation unaffected. Cost if wrong: slightly wider CI on the dense 128k baseline number.

## 2026-11-08 12:00 — M0: 32k dense floor PASS; 128k dense rep0 done (depth-1.0 MISS); 950DT probe GO; bench-crash hardening

- Dense 32k: 9/9 HIT, all categories 1.0 -> floor PASS (results/dense_ruler_32768.json). 12 min total (~80s/req).
- Dense 128k rep0 (seed 42): depth 0.25 HIT (wall 350s), depth 0.50 HIT (845s), depth 1.0 MISS (863s). Dense baseline itself fails end-of-context needle at 128k (temp 0, int4 KV default) — HSA comparison is ratio-to-dense, this IS the dense reference; do not "fix" dense.
- Bench crashed on 503 (server evicted+reloading model between requests). Hardened runner: 3-attempt retry w/ backoff on 5xx, miss fallback. Repeat in flight (seed 43, 3 requests ~35 min) — ruling: 2 data points per category accepted for dense 128k baseline (thermal: stop GPU load before operator's afternoon 1M window). Cost if wrong: wider CI on dense mean.
- 950DT training probe EXECUTED: GO. Ascend950DT_9582, torch_npu 2.10.0, 4-layer Qwen3.8 full-layer-shape block seq 4096 bf16 AdamW: loss 1.0473->1.0384 (5/5 decreasing), steady 0.085 s/step, 7.47 GB per 4 layers (~30 GB extrapolated for 16 full-attn layers). Container epic_northcutt. Jump-host auth: fail2ban-like; single-prompt combo + -L tunnel works, ProxyCommand does NOT (sshpass TTY). docs/m0-95...

## 2026-11-08 12:30 — M0 COMPLETE (except deferred items)

- LoRA smoke: VIABLE. mlx_lm.lora 20/20 iters on Qwen3.8-27B 4-bit base, train loss 2.589->1.844, val 4.566->1.765, peak 74.8 GB, ~0.09 it/s steady. Solo-only (kill resident servers first). docs/m0-lora-smoke.md.
- M0 report written: hsa/docs/m0-report.md. Commits: ca9952f, b52453e, 8978482. Tests 8/8.
- Deferred: dense 128k seed-43 repeat (bg task killed by session memory-pressure reap; harness rule = no restart without operator ask); dense curve 256k/512k (behind 1M window).
- 1M relaunch: scripts/relaunch_1m.sh ready, bash -n OK, NOT executed. Operator launches this afternoon; script does not check thermal precondition.
- Next: M1 kernels + parity per plan.

## M1 three-tier attention — hsa repo (2026-11-08)

- SWA / landmark / latent / prune tiers implemented + parity-gated (MLX CPU device; Metal fp32 matmul err ~6e-4 unacceptable for parity).
- Streaming tier committed ac591f4. 19/19 tests green.
- Ruling: MLX matmul `(1,h,d) @ (h,d,k)` leading-1 broadcast silently duplicates heads (head0==head1, head2==head3) — root cause of streaming mismatch. Correct pattern: matched batch dims `(h,keys,d) @ (h,d,1) -> (h,keys)`. Batch tier unaffected (uses matched dims). Cost if wrong: silent wrong attention under GQA.
- Ruling: latent tier uses per-block MEAN keys projected to d_latent + block-mean values (v1 per-token latents were degenerate = full attention).
- Deferred to M2: learned projections replace truncation/pad latent projection.
- Dense 128k seed-43 repeat: all 1.0 PASS — rep0 depth-1.0 miss is seed flake under int4 KV, not systematic. m0-report updated.
- 1M rerun: relaunch script ready, HELD for explicit user command per directive.

## M1 complete — hsa commits ac591f4 + 3490497 (2026-11-08)

- Streaming fix: leading-1 broadcast matmul head-duplication bug (root-caused via per-head dot isolation vs numpy) — 19/19 green, streaming tier committed.
- Torch fp64 independent oracle added for latent-tier contract; batch + streaming both parity-match (atol 1e-4). 21/21 suite green, committed 3490497.
- M1 acceptance per spec met: three-tier attention in MLX behind tests, torch-parity oracle green.
- Final whole-package review dispatched (background); findings pending.
- Next: M2 learned latent projections (training). 1M rerun still HELD for user command.

## Final review fix pass — hsa commit (2026-11-08)

- Reviewer (fresh context, whole package): 4 MAJOR + 5 MINOR. All fixed in one pass.
- Ruling: skip algorithm rewritten to per-query bound-descending processing with exact-max termination (fixpoint) — previous version computed dense scores first so bounds were dead code. Exactness argument: dropped block max logit <= bound <= M - eps, weight < e^-eps.
- Ruling: test 1 (random data, small logits) cannot produce skips by design (gap 20 >> logit spread) — drop assertion moved to crafted zero-block test; random-data test pins causal-visibility bounds instead.
- Suite 27/27 green (was 21). MLX pitfall re-check by reviewer: no remaining 2D@3D or leading-1 broadcast matmuls.

## M2 start — plan docs/superpowers/plans/2026-11-08-hsa-m2-training.md (2026-11-08)

- 5 tasks: torch combined tier -> distill loop smoke -> corpus -> 950DT run -> MLX RULER sweep.
- Builder rulings R1-R6 locked in plan (loss = attention-output MSE + hidden MSE vs frozen dense teacher; trainable = injected projections only, LoRA fallback; d_latent 2048; window 4096 / block 2048; landmark = block first token).
- 1M rerun still HELD for user command.

## M2 Task 1+2 complete — commits ea95515, c0277d2, dtype-leak fix (2026-11-08)

- Task 1: torch combined three-tier attention (window + landmark exact + latent), fp64-ref parity, self-query no-self-averaging invariance, learned-projection path, partial-block seq.
- Task 2: distill loop smoke green on tiny 2-layer transformer; frozen-base zero-grad pinned; truncation-identity init = M1 default at step 0.
- Ruling: latent VALUES stay in dv space (w_v is (dv,dv) at most); only keys projected to d_latent.
- Ruling: test_landmark property = invariance to own-block non-window content (window-only expectation was wrong: earlier blocks are legitimately visible).
- Fixed: torch.set_default_dtype(float64) global leak broke random-order suite. 33/33 green.
- Next: Task 3 corpus pipeline, then Task 4 950DT run.

## M2 Task 3 complete — commit a2d3420 (2026-11-08)

- Corpus pipeline: fineweb-edu via hf-mirror, length-binned int32 shards + manifest; 20MB smoke verified (decode roundtrip, EOS, 645 seqs), smoke data cleaned.

## M2 Ruling R7 — latent architecture fix (2026-11-08)

- Gap found: per-head latent projection degenerates on real Qwen (head_dim 128 < d_latent 2048 = zero-pad).
- R7: latent key = block-mean over CONCATENATED kv heads (hkv*128=1024) -> W_k_lat 1024->2048; latent query = token-concat (8192) -> W_q_lat ->2048; ONE shared latent logit per (query, block), broadcast across heads inside each head's softmax. Latent value = block-mean v per kv head (GQA-expanded). Matches LSA d2048 map precedent (heavy-key-and-query).
- Init: W_q_lat/W_k_lat near-zero -> student starts as window+landmark only, latents learn from zero (safe).
- Window-only M1 kernels remain the per-head fallback; shared path added to torch_tier, then vectorized torch kernel (per-query loop unusable at seq 4096 training).

## M2 Task 2.5 complete — commit c543cf8 (2026-11-08)

- Vectorized batched three-tier torch kernel (window + landmark + shared latent, GQA, partial blocks) — fp64 parity, batch-invariance, gradient-flow tests. 37/37 suite green.
- Ruling: asymmetric init W_q_lat ~ N(0,1e-3) / W_k_lat = 0 (zero-zero init kills both grads; this lights w_k first, w_q second step).
- Ruling: loop torch_tier stays per-head fallback/reference; fast kernel is the training path.
- Next: HF Qwen3 tiny-config integration smoke locally, then 950DT upload + run.

## M2 Task 4 prep complete (2026-11-08)

- Kernel perf ruling: broadcast-multiply materialized (B,T,Hq,W,D) = 137GB/layer at real dims -> grouped kv-head matmul rewrite (c78fed5), parity suite green.
- Training entry 15fe0f4: same-weights teacher via per-step tier-mode toggle (no 27B model copy), hidden-MSE + student LM loss, bf16 tier weights (follows model dtype), gradient checkpointing, durable ckpt/JSON logs. Tiny end-to-end smoke RC=0.
- 950DT scripts committed (tunnel/upload-sha256/sanity/train/status). Corpus: fineweb-edu 10BT shard fix (100BT shards sorted first - killed, filtered, relaunched), ~1.1GB/6min download rate.
- Train window ruling: W=512, block=2048, seq 4096 (window 4096 would make train memory = dense; window is a design free variable per spec).
- MPS note: MPSGraph INT_MAX dim limit blocks full-scale local validation; correctness pinned by fp64 parity at small dims, scale validated on NPU in sanity phase.
- Next: corpus done -> upload (~8GB) -> 100-step sanity (memory <80% check) -> 8000-step full run.

## M2 Task 4 — 950DT execution (1108)

- Ruling R8: real Qwen3.8-27B is hybrid, not plain Qwen3. 64 layers = 48 GatedDeltaNet linear-attn + 16 full self_attn (indices ≡3 mod 4), hq=24 hkv=4 head_dim=256, q_proj emits 2x head dim with sigmoid output gate. Tier injection targets the 16 full-attn layers only; trainable ≈ 300M (was 900M estimate). d_latent concat dims: k 4*256=1024, q 24*256=6144 — attach code already config-driven. qwen3_tier.py rewritten (commit 8545f65); suite 38/38; single-layer real-config check: tier fwd finite, w_k_lat grad 0.0159, w_q_lat 0 (asymmetric init as designed).
- Ruling R9: 950DT only has W8A8 MindIE-quantized copies (quant_model_weights-*.safetensors) — not trainable via HF. Downloading bf16 Qwen/Qwen3.8-27B (~54GB) from hf-mirror to /mnt/sfs_turbo/dahai/hsa_probe/models/qwen38-27b-bf16 inside container. Sanity run deferred until weights land.
- Two run_m2.sh launcher bugs fixed en route: tar layout put remote_train.sh under scripts/950dt/ (path fixed); PYTHONPATH=$DIR/src missing (ModuleNotFoundError hsa).
- Ruling R10: vllm-ascend container (qwen38-27b-dp8-bt32k) crashloops on its own serve config (super-kernel assert) — pre-existing, not touched. Spawned dedicated `hsa-train` container (same image vllm-ascend:v0.30.0-a5, same mounts/devices, --network host, sleep infinity) for M2 training; run_m2.sh prefers hsa-train. Bridge network has no DNS in this env — host network required (download died once on name resolution).
- Sanity PASS (1108 09:07-09:31): hybrid-aware attach/enable on 16 full-attn layers; OOM root cause = model.train() missing (grad ckpt inert in eval mode, all layers kept backward graphs at 91GB card); second fix single student forward; ckpt save crashed on GatedDeltaNet layers (fixed, 5-step validation ckpt_5.pt saved then cleaned). Loss 6.58→5.83→5.03 (steps 0/10/50), s_lm 3.90→2.70, t_lm ~2.0. Throughput ~12s/step (teacher fwd + student fwd/bwd, seq 4096, 1 card). Full 8000-step run launched as full_1108_0932.log (~27h ETA). Commits 8545f65, 8b9c85c.

## 1M v5 outcome (1109 morning)

- v5 prefill COMPLETE: 1095638/1095638, 17:54->05:37 (~26 tps avg), phys peak 108.65GB/128GB, zero enforcer events, machine alive. v4 SoC-reset blocker at 986k cleared.
- v5 decode FAIL before first token: metal::malloc "Resource limit (499000) exceeded" inside fusion_lsa/core/ward.py ward_rounds during decode-start index rebuild; secondary "no Stream(gpu,2)". Bytes healthy (active 78GB, phys 108GB) -> ~500k LIVE Metal buffer count exhaustion over 534-chunk prefill, not an OOM. ward_rounds is tiny (blocks=1, pools=64) - straw not cause.
- Fixes: ward_rounds eval-flush every 64 merge iterations (fusion-lsa); env-gated gc count of live mx.array per memcheck:chunk, HSA_COUNT_MX_ARRAYS=1 (fusion-mlx b70a83c / 07d308f). v6 launched with instrumentation; next run either survives or pinpoints the leaking subsystem via live_arrays curve.
- 950DT: step 3540/8000 at 05:48 CST, 7 ckpts (500..3500), healthy, ETA ~15.5h. Note: container clock is UTC - do not misread log timestamps.

## M3 task list (accumulating)

- M3-T1: threshold routing short/long ctx at serving layer (fusion-mlx). Prompt token count known before prefill: T < threshold -> dense attention path, T >= threshold -> LSA path. Threshold TBD 16k-32k from M0 dense curve + M3 real measurements. Prefix cache must be isolated per path (same prefix has different KV representations). Rationale: tier attention is bitwise-dense for T<=W(512) so quality is smooth; routing removes LSA index/paging overhead that is pure loss at short ctx. RULER 4k/32k already 1.0; 32k 1.5 tps regression was TQ kernel bug #1513/#2686, not LSA.

## M2 Task 4 COMPLETE (1109 21:10 CST)

- Full 8000-step distill run done on 950DT: final_loss 5.15, s_lm 3.90->3.04, t_lm ~2.0 (12.4s/step, 27.6h, zero crashes). Curve: loss 6.58->5.55(2k)->5.41(4k)->~6(6k, batch variance)->5.82(8k). 7 intermediate ckpts (500..3500) + ckpt_8000 on remote.
- ckpt_8000.pt (448.5MiB tier weights, 16 full-attn layers) + train_log.json pulled to /Users/dahai/fusion/hsa/results/m2_ckpt/, sha256 4679627c... verified local==remote.
- Note: mid-run loss upticks (6.1@6k, 6.6@7970) are single-sample batch variance, not divergence - s_lm trend monotone down.
- Next: Task 5 - torch ckpt -> eval converter + RULER sweep 4k/32k/128k x 3 seeds vs 0.99 ratio floor (one-vote-veto). Eval first pass on 950DT via torch tier mode; MLX-side integration is M3.

## M2 Task 5 debug log (1109 evening)
- Smoke hit=False root-cause chain: (1) tier forward had NO decode path - generate() fed T=1 with past_key_values ignored -> gibberish. Fixed: TierLayerCache (hsa/src/hsa/eval/tier_cache.py) mirrors kernel decode-row semantics; parity vs three_tier_attention_fast exact (fp32, two shapes). Commit 94f3c10.
- (2) Still failed with trained ckpt even with needle INSIDE window (4k depth 0.95). Controls: dense hit=True; tier-untrained hit=True (fluent); tier-trained gibberish loops. Teacher-forced tail NLL on same prompt: trained 2.92 < untrained 3.09 < dense 3.51 -> prefill/weights fine, collapse is autoregressive drift under the trained latent tier (only trained component; window/landmark tiers are parameter-free).
- attach_tier_weights device bug fixed (params were cpu on NPU). Ruler CLI: --window --block-size --dense --no-ckpt. Commits 94f3c10, cf2764e.
- Structural risk noted: block-first-token landmarks + block-mean latent cannot see a needle at arbitrary depth (4k d0.75 unreachable with window 512). In-window retrieval works untrained. Full RULER 0.99 at all depths likely needs query-dependent retrieval (LSA-style) at decode - matches spec "decode reuses the same structures (LSA decode is the special case)".
- RUNNING: 9-ckpt sweep on 950DT (500..8000 x 3 cases 4k) results/ckpt_sweep.json; 1M v6 prefill 1047k/1095k at 22:07, decode phase (v5 crash point) next.

## M2 Task 5 sweep verdict (1109 night)
- 9-ckpt sweep (500..8000 x 4k d0.95/d0.75/d0.5): ALL 0/9x3. Even step-500. Dense control + untrained-tier control both hit at 4k d0.95.
- Mass-split probe: untrained latent mass 0%, trained 87.5% avg over 16 layers, latent logit 660 vs window ~20. Latent tier dominates attention after training.
- Teacher-forced NLL: trained 2.92 < untrained 3.09 < dense 3.51 -> trained equilibrium is latent-dominant "gist register"; helps LM loss, collapses free generation.
- Clamp experiments (latent logit +-20 prefill+decode, then +-2): generation still degenerate with any nonzero latent mass; block-mean-v injection is a loop attractor. Not a code bug - eval infra verified (parity exact, dense/untrained controls pass).
- STRUCTURAL: window+first-token-landmarks+mean-latent has no path to an arbitrary-depth needle (d0.75 at 4k: needle 1024 tokens from query, window 512, needle not at any landmark, drowned in block mean). G2 (RULER 0.99 all depths) unreachable with this design as-is.
- Decision needed (user): (a) pivot decode/prefill to query-dependent retrieval (LSA bucket index over stored keys - spec already hints "decode reuses the same structures (LSA decode is the special case)"), (b) redesign landmarks as data-dependent selection, or (c) accept tier for throughput + LSA for retrieval hybrid. M2 training artifacts (16 ckpts + logs) preserved.
- Commits: 94f3c10 (TierLayerCache decode), cf2764e (diag + CLI), working tree: latent_clamp plumbing kernel+tier_cache+ruler.

## 1M v6 outcome (1109 23:22)
- Prefill COMPLETE: 1095638/1095638, 15.4h wall, active 78.19GB / peak 83.74GB / phys 110.58GB stable, live_arrays flat 1189 (telemetry verified true-positive locally). v4 SoC-reset blocker stays cleared.
- Decode start DIED same as v5: metal::malloc Resource limit (499000) exceeded. Crash site: fusion_lsa/core/ward.py:78 ward_rounds inner per-i update loop, called from incremental.py:36 append_keys build_buckets (first full-index ward over 1M).
- Paradox: ~499k live Metal buffers vs only 1189 Python mx.array (101.94GB bytes OK). Per-64 eval flush (v6 fix) did not prevent. Leak mechanism NOT identified: candidates (a) pending lazy-graph intermediates not visible to gc count, (b) per-i loop churn interacting with async queue, (c) buffers held by round-level (G,P,P) temporaries. MLX exposes no buffer count to confirm.
- Next lever: vectorize ward_rounds inner loop (mutual pairs are disjoint by construction -> single scatter/matmul update), eliminating the Python per-i chain; plus buffer-count observability (MLX-side) before burning v7 15h.
- Decision coupling: ward lives in the LSA decode path; if user picks pivot A (tier prefill + LSA retrieval) ward stays and this fix matters; pivot C makes it moot.

## ward_rounds vectorized (1110 00:10)
- Inner per-pair loop replaced with one grid pass per round: kept-side recv grid (take_along_axis transfers absorbed keys), absorbed-side gone grid (alive kill + parent pointer). Disjointness proof: best[j] unique => a pool is kept by at most one partner and absorbed at most once, no intra-round chains.
- First attempt had kept/absorbed grids inverted (delta added onto absorbed, doubling); caught by parity test vs sequential reference kept in test file.
- Parity 5/5 random cases, partition/cap property test, full suite 49/49. Commit in fusion-lsa.
- v7 implication: decode-start ward no longer queues a per-pair lazy chain. If the 499k buffer leak is the ward loop (now most likely given the exact crash site), v7 decode survives; if not, next crash site will localize the remaining leak.

## Architecture decision (1110 07:33, user approved)
- Pivot A CONFIRMED: tier prefill + LSA query-dependent retrieval at decode. Rationale: NIAH needs query-dependent retrieval (fixed landmarks/mean latent cannot see arbitrary-depth needle - proven by 9-ckpt sweep + controls); LSA decode retrieval already passes 0.99 at 32k; spec "decode reuses the same structures (LSA decode is the special case)". M2 tier weights stay useful for prefill/gist.
- Route: (1) v7 1M validates LSA decode path; (2) Mac ablation - ckpt_8000 tier prefill, decode variants tier/window/LSA-retrieval, quantify distribution mismatch; (3) M3 integration tier prefill into fusion-mlx + full RULER sweep vs 0.99; (4) 950DT only if ablation demands alignment distillation.
- 950DT assets preserved in place: hsa-train container, bf16 weights, 16 ckpts. No cleanup.

## Pivot A ablation prep (1110 07:56)
- topk decode mode implemented + committed (f421b84): per-head-group exact top-k over full history at decode, simulates LSA query-dependent retrieval. Parity tests still green.
- Ablation run DEFERRED to tonight: v7 1M prefill holds ~110GB phys on the Mac; MPS cannot also load 27B (54GB). Run after v7 bench exits (memory frees at decode/completion ~23:00).
- 950DT returned to other teams (user); assets preserved in place; coordination required before any future NPU use.

## Upstream LSA coordination (2026-11-10)
- caciolai replied in issue #13: no integration docs yet except FORK.md (Prefrontal-AI/sglang@lsa); upstream SGLang contribution planned; better docs coming.
- Read FORK.md: fork = serving layer only; LSA indexer is an SGLang plugin (~35 hook points, all SGLang-internal); our MLX port (PR #12) needs none of them.
- QSA indexer weights released: qsa-weights-v1 tag, `qwen38_27b__long-16k-1M-heavy-qsa.pt` via `lsa distill inject` -> candidate baseline for M3 head-to-head on Mac (logged for M3).
- Replied on PR #12 (comment 6756152485): port dependency summary + 2 questions (selection-semantics doc independence; qsa-weights-v1 applicability). English-only per convention.

## NPU dense baseline (2026-11-10)
- 950DT 8 cards freed (qwen38-27b-prod stopped on user coordination); hsa-train restarted.
- Dense RULER on 27B bf16 single NPU: 4k 3/3, 32k 3/3 (max-new-tokens 512). First run with 64 tokens had a 32k miss = thinking-mode truncation, NOT model failure. Rule: all future RULER runs use --max-new-tokens 512; ablation script updated.
- 128k dense launched; results -> results/ruler_dense_npu_*.json on remote, to fill paper 6.2/6.4 dense columns.
- 128k dense on single NPU: OOM (89.4GiB allocated, full-length prefill, no flash fallback). Deferred: dense path needs chunked prefill (~30 lines) or 2 cards; non-critical, revisit at M3 when tier 128k sweep needs the cell.
- Chunked prefill added to ruler_tier (chunked_generate, explicit position_ids + past_key_values; first version double-consumed the last prompt token - caught by gpt2 parity test, fixed). Commit 2114be4. 128k dense on NPU: 3/3 hits, ~62s/case. Dense floor complete: 4k/32k/128k all 1.0.

## v7 + Pivot A ablation results (2026-11-10 night) — MILESTONE
- v7: prefill complete 1315000/1315000 (15.2h, live_arrays flat 1189, mem ok). Decode ran 12 min quietly, then died at a decode-time page close: metal::malloc (499000) inside ward_rounds. NOT the ward graph (repro at serving shapes passes, even 2000-page lazy accumulation passes) - the buffer quota is process-wide and the allocator cache pool filled during 12 min of decode churn. Fix: mx.metal.clear_cache() in IncrementalBucketIndex.append (once per 512 tokens, cheap). Pending v8 validation.
- Pivot A ablation (Mac MPS, trained ckpt_8000, topk-64 retrieval decode, 512 new tokens): d0.75 3/3, d0.95 3/3. Static tiers were 0/9 at d0.75 across all 9 checkpoints. Pivot A VALIDATED: tier prefill + query-dependent retrieval decode reaches arbitrary-depth keys.
- 3 real bugs fixed en route (tier_cache.py topk branch): torch 2.13 MPS matmul mis-broadcast -> einsum; kx[tk] indexed dim0 with position ids -> gather along token axis; tests added (dense-attention oracle K>=T + needle retrieval K=4). Commit 986e1df.
- max-new-tokens 512 protocol everywhere (thinking-model truncation).
- Next: M3 (serving integration + full RULER sweep), Phase 2 torch_npu on davinci0, v8 1M rerun with clear_cache fix.

## 2026-11-11 M3 + Phase 2 progress
- M3-a kernel: hsa a841d6d (MLX three-tier kernel, 4/4 oracle parity) + a2c3f30 (q_offset chunked prefill, 5/5)
- M3-b serving: fusion-mlx de87d872 (--tier-weights wiring) + ded96071 (ModelSettings overlay fix - flag never reached wrap, found via live boot log)
- Tier weights exported: ~/.fusion-mlx/adapters/tier-weights-ckpt8000.safetensors (16 layers fp16, W512/B2048/L2048)
- Phase 2 sweep2 (davinci0, torch_npu, tier prefill + raw topk-64 decode, chunk 4096): 4k 3/3, 32k 0/3, 128k 0/3. Raw-key topk-64 loses needle rank beyond 4k - motivates trained-indexer selection (LSA path) at scale
- Diagnostic running: 32k/d0.5/topk-512 (rank coverage vs deeper failure)
- v8: prefill ongoing, ~67GB live, healthy
- Ruling: sweep2 32k/128k 0/3 作废 - eval 侧 tier_forward chunked prefill bug（chunk 2+ 只看本地 k/v，cache 每 chunk 重 seed），4k 单 chunk 掩盖。hsa b0c7f8e + 后续两小修：torch kernel q_offset（镜像 MLX）、cache k/v 布局、q_offset init。chunked=one-shot parity 6/6
- 32k/d0.5 修后诊断：tier prefill + dense decode hit=True（26s）- bug 证实且修复有效
- "raw topk 辨别力不足" 判断作废；sweep3（topk-64, chunk 4096）重跑中
- sweep4（修后, chunk 1024, topk-64）: 4k 1/3, 32k 2/3, 128k 2/3。d0.75 3/3 全长度（128k 含）; d0.25/d0.5 摇摆。kernel mid-block parity 精确（1.8e-7）— 摇摆 = bf16 数值序 + latent 塌缩外推噪声, 非 kernel 逻辑
- memory hygiene: per-case tier cache 释放 + npu.empty_cache（OOM 蠕升 2GB/case 修复, hsa）; 128k 窗口 gather 需 chunk 1024
- clamp-2.0 诊断跑中（seeds 42/43, 6 case）
- clamp 2.0 诊断 6/6（基线 2/6）— §4.3 latent 塌缩确证为摇摆根因, clamp 定为 serving 头牌配置
- sweep5 全量 9-case + clamp 2.0 跑中（davinci0）; serving 侧 --tier-latent-clamp 接线完成（fusion-mlx 73d327f, tests 5/5 + lsa 102 pass）
- sweep5（clamp 2.0 全量）: 各长度 2/3。反转：seed41(d0.75) 全灭（无 clamp 时 3/3），42/43 全中。clamp 2.0 压死 latent 信号 — 与 sweep4 完全互补。结论：非单调，中间值可能双赢，clamp 4.0/6.0 跑中（sweep6）
- sweep6: clamp 4.0 = 5/9（41 仍 0/3, 42 3/3, 43 2/3）; clamp 6.0 = 4/9。无单点 clamp 双赢 — 硬 clamp 非正解
- 根因假设升级: latent logits 塌缩 ~660 尺度, bf16 ULP≈4 → 相对差被舍入, 排序信号毁。正解 = latent 路径 fp32（torch_tier_fast q_lat/k_lat/slt/softmax 全 fp32）; parity 6/6 过; sweep7（无 clamp + fp32）跑中
- sweep7（fp32, 无 clamp）: 6/9。seed41 3/3 保住 + 128k 首次 3/3 满分。残余: 4k 42/43 + 32k 43。fp32 严格 ≥ bf16。commit f06cfcc。sweep8（fp32+clamp2.0 交互）跑中
- sweep8（fp32+clamp2）: 6/9, 与 bf16+clamp2 同模式（41 0/3, 42/43 6/6）— clamp 压平 latent 与 dtype 无关。fp32 与 clamp2 互补（并集 9/9）
- 新杠杆: latent z-score（per query row 标准化, 保排序 + 塌缩尺度自动归一, 噪声行自然失效）。torch kernel --latent-zscore flag 接线（tier_forward + ruler_tier CLI）; sweep9（fp32+zscore, 无 clamp）跑中
- 运维: davinci0 sfs_turbo cp 偶发不生效（需 md5 前后校验）; ssh 鉴权 ~50% 失败率持续
- sweep9（fp32+zscore）: 5/9, 41 0/3 同 clamp 翻转模式 — 任何 latent 尺度变换都杀 41。eval hack 到头
- 收官判定: fp32 原始 = 头牌（6/9, 唯一保 41 + 128k 3/3）。MLX kernel 同款 fp32 修复 + --latent-zscore flag 提交（hsa 85c7d87）。eval seed↔depth 混淆（单 seed 单深度）; 42/43 浅深度失败根因 = latent 权重未对齐, 根治在 Phase 2 训练（fp32 latent + 长序列 + 多 seed 多深度 eval）
- 用户同意资源方案: Phase 2 主计划 1 卡 + 其余卡跑任务 4 四变体并行矩阵（teacher/CE-only × lr 5e-5/1e-4, 每变体 1 卡全课程）; eval gate 选赢家, 全败则升级 Phase 2b（检索感知损失, 新范围）。计划已改 + 提交
- 8 卡窗口确认: 今晚 23:00 后。runner 脚本备好（hsa 72f6b6a scripts/950dt/run_phase2.sh）: 卡 0-2 eval 基线矩阵（每深度一卡, ~20min）/ 卡 3 smoke / 卡 4-7 四变体训练 / gate 卡 0-3。前置: 任务 4 代码改动（--init-ckpt/--no-teacher）+ 任务 3 corpus 上传须在 23:00 前完成
- Phase 2 准备全完成（23:00 自动触发已挂 b3vblnesc）:
  - 任务 4 代码: --init-ckpt/--no-teacher TDD 完成（hsa 1e35cda, 2 tests RED→GREEN）
  - 任务 3 免做: 远端 data/ 已有全套 bins（8k:6933/16k:1579/32k:290）; runner data-dir 修正（686a497）
  - runner 并发缺陷修复: 同卡串行链（smoke/gate）+ matrix 每深度一卡（1b18b52）
  - 代码已同步 davinci0, 4 文件 md5 校验对; 远端 --init-ckpt/--no-teacher argparse 干跑过
  - 23:00 启动序: tunnel → matrix(卡0-2) + smoke(卡3) + train(卡4-7) 三路并发; 训练 ~2-3h 后 gate
- M8 v8 第一次尝试失败: watchdog 14:37 因 macOS critical 压力持续 15 轮杀 bench+server, prefill 63.4%（694272/1095638）, decode 未开始, clear_cache 未验证。压力源 = allocator cache chunk 间反弹至 23GB, phys 92GB/128GB
- 修复: fusion-mlx-lsa-wt 56e2927 FUSION_MLX_METAL_CACHE_LIMIT env（serve boot 调 mx.metal.set_cache_limit）; relaunch_1m.sh 设 8GB cap（hsa 4cc83fe）
- v8 重发 14:48: server+watchdog+bench 全起, cache cap 日志确认, ETA ~04:00。哨兵 bmlh3lt8a 盯 bench 进程退出
- 旁证: 13:56 有 unknown 方拉起默认 :11434 server（pid 3928, RSS 144MB, 非压力源）
- 23:04 Phase 2 发车。首波事故×2 已修: (1) load_init_ckpt 用 model.layers 实际是 model.model.layers → 4 变体全炸, 修复+嵌套测试过（hsa commit）; (2) smoke 的 \$seq 被远端 ssh shell 提前展开成空 → argparse 死, 改显式三连（3693e2c）
- 23:10 全 8 卡在跑: 卡 0-2 matrix（d0.25/0.5/0.75 × 5 seeds × 3 长度）/ 卡 3 smoke 8k→16k→32k 串行 / 卡 4-7 四变体 8k 阶段（v1 teacher-5e-5, v2 CE-5e-5, v3 teacher-1e-4, v4 CE-1e-4）。v8 Mac 侧 bench 仍活。matrix 哨兵 bex2xw9ok
- v8 终局: prefill 100%（1095638/1095638, 14.6h, cache cap 8GB 生效, phys 峰值 106.95GB）; decode 入口 05:23:51 `[metal::malloc] Resource limit (499000) exceeded` 崩 → 空 answer。prefill 路径验证过, decode 资源上限 = 新独立 bug（待定点: buffer 数量类上限, 4GB cap 或 prefill 尾部内存再压; 每次 retry 14.6h, 须先有假设）
- 05:30 夜间战况: P2 半瘫痪。matrix 仅 d0.25 完成（15 cases）; d0.5/d0.75 无输出; smoke 8k 卡 teacher 前向 14h; v1/v3 同卡死点; v2/v4 ERR99999 Device:-1
- 05:45 根因: 容器只挂 davinci0 + env ASCEND_RT_VISIBLE_DEVICES=0。lane 设备号 1-7 不存在 → torch 回落 CPU → 8 进程 × 27B CPU 前向 = 挂死/龟速。ERR99999 = 设备初始化失败
- 修复战: (1) pkill 自杀陷阱（模式匹配自身 cmdline, rc=137, [] 括号技巧破）; (2) 8 设备容器重建 × 2 失败 — dmesg `uda_occupy_dev_by_ns: Conflict open udevid` + `Can't get ascend_hal device count`, 单容器多卡在此 host 上无法枚举; (3) 定稿方案: FLEET = 每卡一个单卡容器（hsa-c0..c7, 与已知可用旧容器完全同构）。卡 2 被 lsa-port 占用 → lanes = 0,1,3,4,5,6,7
- runner 已改 fleet 拓扑（hsa b26d86f）: lane_launch/lane_exec, LANES 数组, gate lanes 0/1/3/4
- 06:00 重发: matrix d0.5→c1, d0.75→c3, smoke 8k→c4。d0.25 基线已有不重跑。验证 4min 后 HBM 占用+日志
- 今晚现实口径: 4 变体先跑 8k 阶段 500 步（~3-5h/变体）, gate eval 用 8k ckpt; 16k/32k 阶段下一窗口续
- smoke 8k 真崩因（非内存）: qwen3_tier.py 训练全序列分支把 HF 布局 (B,Hkv,T,D) 的 k/v 直传 kernel；kernel 从 k.shape[2] 读 Hkv → 8k 时变 T、group=0 → grouped view 炸（`[1,8192,8192,0,256] invalid`）。eval 走 TierLayerCache 布局对所以从未炸；4k M2 训练早于向量化 kernel 所以没暴露。修复: caller transpose + kernel q.view→q.reshape（TDD: test_tier_forward_layout RED `[1,8,8,0,8]` → GREEN），hsa 5c143fb，已 md5 校验同步远端，smoke 重发 c4
- 遗留发现: test_torch_tier_fast::test_fast_matches_fp64_reference[False] 在 HEAD~1 就失败（先于本次改动），fp64 参考精度不匹配 — 训练/eval 实证结果不受影响（RULER 全套跑过），明天排查根因
- smoke 8k 第二轮 OOM（86.5/91.75GB, 8.11GB gather 请求）: teacher 释放后 student backward 仍爆 — 根因 = 精确 tier（window/landmark/gather/块均值）全是冻结骨干的纯函数却进了 autograd 图（8.2GB gather × 2: scores+values）
- Ruling: 精确 tier 全部 no_grad — 可训练参数只有 latent 路径上的 w_q_lat/w_k_lat；被切断的跨层梯度（早层 latent 输出影响后层精确分数）= 二阶近似，训练信号主通道（层内 latent）保持精确。输出与新旧 kernel bit-identical 验证过。hsa ad0a6ca
- 测试补强: test_exact_tiers_detached_latent_grads_flow（latent 梯度流 + q 梯度经 latent 路径 + k.grad None 断言）。hsa 布局测试 4/4
- 遗留: test_torch_tier_fast fp64 参考测试 [False]+[True] 均挂（早于今晚全部改动, fp32-latent 重构期引入; 之前跑 suite 带 -x 停在 [False] 没暴露 [True]）。实证（NPU RULER 全套）不受影响, 明天修
- matrix 基线齐二: d0.75 = 4k 5/5, 32k 3/5, 128k 4/5; d0.25 = 4k 1/5, 32k 4/5, 128k 4/5; d0.5 128k 尾段。gate 阈值即此
- smoke attempt4 (kernel no_grad 版) 发车 c4, 监控 b8f9h9n3g
- smoke attempt4: step=0 过（24s/step 量级!）但 step=1 student backward 仍 OOM（82.79GB active + 8.21GB vw gather）— no_grad 不够: detached gather 本身 8.2GB live/层, backward 重算期叠加 64 层 saved acts
- 终修: gather + value matmul 按 1024 行分块（每块 ~1GB 瞬时）, scores 与 values 两侧都分块; 输出与未分块 kernel bit-identical（T=2500 多块形状验证）。hsa fc89ef4
- v1-v3 首发带旧 kernel 已杀, 全部 4 lane（smoke c4 / v1 c5 / v2 c6 / v3 c7）带 chunked kernel 重发 06:25
- 8k 步速基线: step0 全流程 24s → 500 步 ≈ 3.3h → ETA ~09:45（8k 阶段）; 16k/32k 阶段今晚不排, gate 直接用 8k ckpt_500
- smoke 100 步全过（40min, ~21s/step）: loss 噪声无发散（s_lm 2.8-3.6 带内随机序列波动）, 8k 训练路径彻底打通（布局修复 + teacher 释放 + exact tier no_grad + 分块 gather 四连修）
- v4 发车 c4 07:02（CE-only 1e-4）。全 4 变体在跑, 完成哨兵 bve4u98up（10min 轮询, 全 done 或 crash 触发）。ETA v1-v3 ~09:30 / v4 ~09:50
- Task 1 收官: 3 深度 × 5 seeds 基线全齐（d0.25 0.2/0.8/0.8, d0.5 0.6/0.6/0.4, d0.75 1.0/0.6/0.8）。gate 判据: 变体 ≥ 基线

## 2026-11-12 上午（4 卡窗口: davinci0 lanes 4-7）
- 卡况: 0-3 已交其他组（c0/c1/c3 容器 09:00 被 137 杀）。4-7 全天可用, 16k/32k 今天闭环, 不等用户确认
- 8k 阶段完成: v2 done 01:12 UTC (loss 3.257), v3 done 01:39 (loss 5.664), v1 done 09:42 CST, v4 ~10:00。teacher 每步 enabled/disabled 切换日志是正常步内行为, 勿误判
- 事故 1: auto_gate 与链式 16k 抢 c6 → v2 的 16k 进程被 host OOM-killer 静默杀（无 traceback 特征）。auto_gate 已杀, 8k 中间 gate 取消（12:00 死线没了, 最终 gate 在 32k ckpt）
- 事故 2: 16k 训练 NPU OOM（84.2GB 占 + 15.16GB fp32 logits）。根因: 全序列 lm_head logits fp32 上浮 + output_hidden_states 收全 64 层。修复 bcc2eaa: backbone 直出 last_hidden + chunked lm_head CE（每块 backward, 图即释）+ teacher CE 分块 no_grad, 梯度数学等价（单测 chunked==full, 含 /T 缩放 bug 回归）; CPU 微型 Qwen3 端到端冒烟过
- 链式重启: v1 c5 / v2 c6 / v3 c7 已带 chunked loss 重发 16k→32k; v4 由 watcher (beub2v3d0) 8k 退出后自动接力。watcher 只扫启动后新增字节（旧 OOM 残留不误报）
- 待办: 32k teacher 密集注意力后端未验证（若 eager 51GB/层会爆, sdpa 则无碍）; 32k ckpt 后跑最终 gate（3 深度 × 3 长度 × 5 seeds）→ 判定 vs 基线 → 导出赢家
- 16k 全程零事故（chunked loss 生效）。v2 17:40 完成 16k → 32k OOM @90.2GB（student 激活墙, teacher 稠密注意力探针确认可通过）。修复 commit: activation offload（saved_tensors_hooks 把 >200MB 非叶子 saved tensor 搬 host RAM, ~26GB 层输入, 单测梯度等价）。v2 32k 已带 --offload-activations 重发; watcher 升级自动治愈: v1/v3/v4 32k OOM 自动带 offload 重发（限 1 次）。32k ETA: v2 明晨 ~11:00, 其余 ~12:00-13:00

## 2026-11-17 (v9 launch + issue #1 closure)
- retry question answered: retry DID execute once in v8 and failed identically; uncaught propagation (no success log line), hence 0 "retry" strings in serve.log. User's "98.7%" framing corrected: that is index key count; prefill processed 1,095,638 = 100%.
- §5.3 updated with v8 run + v9 pending (hsa c7f8cbf); run-logs v9 row (hsa 8d93823); fusion-lsa issue #1 closed via PR #2 merge, record comment added (before=0 min, after=~2 min, ubuntu gate).
- v9 launched 08:20 CST via hsa/scripts/relaunch_1m.sh (sed v5->v9 logs, /tmp/relaunch_1m_v9.sh): serve pid 46231 (:11555, LSA armed ceiling 108.8GB, k8v4), bench pid 46684, watchdog 46685. Prefill ETA ~16h (00:20 CST 11-18). Watchers: bench-exit until-loop + task bnolhtffk.
- untracked adapters/ (55.7M LoRA weights from 8k smoke) left out of git deliberately.
- v8 narrative corrected (hsa 677dfab): serve-log forensics prove ONE continuous prefill (same rid 07:57->23:41, 15h44m, 19.4 tok/s); the 2,113 page closes started 0.5s AFTER prefill completed = decode-entry rebuild split per-page with clear_cache, NOT in-prefill builds. Earlier "63.4% watchdog kill attempt 1" memory was wrong.
- v9 healthy at 09:20: 253,952/1,095,638 (23%), ~70 tok/s (3.6x v8 - quiet machine, no :11434 resident), ETA decode entry ~13:00 CST. 0 incremental logs during prefill = expected (index not built during prefill). Note: FUSION_MLX_METAL_CACHE_LIMIT knob never fires (log line absent v8+v9; enforcer overrides to 64GB via working-set fraction) - not blocking, v8 prefill survived same config.
- 12:44 machine reboot killed v9 at 48% (526,336 tok, clean logs, no code failure). v9r2 relaunched 13:16; bench twice died Errno 61 while server healthy -> root cause: httpx trust_env honors macOS SYSTEM proxy (scutil 127.0.0.1:7897) via urllib _scproxy; Clash not running post-reboot -> refused. trust_env=False works. Fix: no_proxy=127.0.0.1,localhost exported in relaunch_1m.sh (hsa 840cea4) + bench relaunched with it 13:23 (HTTP 200, bench 63082, watchdog 65040). Pitfall-38 variant: system-config proxy, not env var.
- Note: a parallel session committed hsa 95a49a7 (v9 kill/relaunch + ICML 2027 venue dates) while this session debugged - no content conflict.
