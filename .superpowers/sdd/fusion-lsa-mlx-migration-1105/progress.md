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
