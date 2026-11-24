# fusion-lsa

Latent Sparse Attention (LSA) ported to MLX/Metal for the [Fusion](../) monorepo — a local-first
Apple Silicon AI ecosystem.

LSA keeps the attention kernel unchanged and only decides **which keys attention reads**: index
keys are pooled and Ward-clustered into buckets, queries score bucket centroids, pick the top-p
buckets, and expand them to a top-k key set beside the sink tokens and the sliding window. This
repo ports the selection logic (not the CUDA kernels) from the
[latent-sparse-attention](https://github.com/Prefrontal-AI/latent-sparse-attention) plain-torch
reference, parity-tested against it as an oracle. Upstream attribution:
[NOTICE.md](NOTICE.md).

Status: P0 (algorithm port + parity + cost gate) complete. P1 in progress: released map converter
(GLM-5.3-Flash, Qwen3.8-27B), index space port, page-boundary incremental index, real-checkpoint
e2e. See `docs/p0-report.md`.

## Results (P0, M5 Max 128GB)

- Exact oracle parity on bucket partitions, members, sizes and selected key sets; centroid parity
  within 1.5e-4 (fp32 summation order).
- Decode selection ~0.7-1.0 ms/query; ward/bucket build 7.2s per 128k context (page-boundary only).
- 128k end-to-end: needle retrieved, selected keys hold > 0.9 of attention softmax mass.

## Layout

```
src/fusion_lsa/
  core/         ward.py, reference.py — MLX port of the oracle selection logic
                index_space.py — regroup, rownorm, learned projections
  convert/      map_converter.py — released .pt indexer maps to safetensors (fail-loud LsaMap)
  index/        incremental.py — page-boundary bucket index (batched-equivalence contract)
  evaluation/   selection_bench.py — per-block selection cost bench
scripts/        extract_nlsa_vectors.py — torch-side real-checkpoint vector extraction
tests/          oracle parity + 128k end-to-end + real-weights parity
docs/           P0 report, bench results and logs
```

## Setup

```bash
cd /Users/dahai/fusion
source .venv/bin/activate
cd fusion-lsa && pip install -e ".[test]"
python -m pytest tests/
```

Parity tests read the upstream oracle from `~/model/latent-sparse-attention/src`
(override with `LSA_UPSTREAM`); they skip when it is absent.

## LongBench QA task (real-text eval)

`scripts/ruler32k_bench.py --task longbench_qa` runs a real-text QA task
against the same server as the synthetic RULER tasks. The dataset is a local
JSONL file (`datasets/longbench_qa.jsonl`, gitignored) — no network access at
run time. Fetch it once via the HF mirror:

```bash
mkdir -p datasets
HF_MIRROR=https://hf-mirror.com python - <<'PY'
import json, zipfile
from huggingface_hub import hf_hub_download
# datasets>=3 refuses script-based repos; pull data.zip from the hub instead
path = hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset")
with zipfile.ZipFile(path) as z:
    z.extract("data/narrativeqa.jsonl", ".")
with open("data/narrativeqa.jsonl", encoding="utf-8") as f, \
     open("datasets/longbench_qa.jsonl", "w", encoding="utf-8") as w:
    for line in f:
        row = json.loads(line)
        w.write(json.dumps({
            "question": row["input"],
            "passages": [row["context"]],
            "answers": row["answers"],
        }, ensure_ascii=False) + "\n")
PY
rm -rf data
```

This yields 200 narrativeqa records.

Expected JSONL schema, one record per line (records are selected by `--seed`):

```json
{"question": "...", "passages": ["...", "..."], "answers": ["gold1", "gold2"]}
```

Scoring follows the LongBench convention: token-level F1 (normalized,
max over gold answers) plus normalized-substring hits; the summary JSON
reports `mean_f1`, `f1_scores`, and `substring_hits` instead of the
synthetic tasks' `hits/total`. Prompt length control matches the synthetic
tasks: passages are cycled/truncated to the token budget with the question
at the end, passage order fixed by the seed.

## Hard boundaries

- This repo is the semantics SSOT for the LSA index (ward merge, incremental
  index, index space). Change protocol and consumer obligations live in
  [INTERFACE.md](INTERFACE.md).
- Selection-logic only: attention itself runs as standard SDPA over the selected keys.
- 1M+ context requires a latent-KV architecture plus KV quantization; full-KV models do not fit
  120GB at 1M.
- Only page-boundary bucket/hierarchy growth; decode selection is per token, never re-clusters.
