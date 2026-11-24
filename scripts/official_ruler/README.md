# Official RULER adapter

Side-by-side runs of the **official NVIDIA RULER harness** against the
fusion-mlx tier server, so our in-repo RULER numbers
(`scripts/ruler32k_bench.py`) are directly comparable to published tables.
Parallel path only — the in-repo generators and their outputs are not
touched.

## Pinned version

- Repo: https://github.com/NVIDIA/RULER
- Pinned commit: `db43f92ff634` (2026-11-03)

```bash
git clone https://github.com/NVIDIA/RULER ~/work/RULER
git -C ~/work/RULER checkout db43f92ff634
```

Running on a different commit requires `--allow-any-commit` and a note in
the results PR documenting the delta.

## Config deltas vs our in-repo generators

| Aspect | Official harness | Our bench | Delta handling |
|---|---|---|---|
| Model access | `OPENAI_API_BASE` / `MODEL_NAME` env, expects a vLLM process | fusion-mlx tier server `:11555` (OpenAI-compatible) | adapter exports `OPENAI_API_BASE=http://localhost:11555/v1` |
| Seeds | per-example seeds derived from the sequence config | `--seed` per run | adapter passes `--seed` into the RULER data-gen invocation; a `(length, seed)` pair fully determines the dataset |
| Tokenizer | `TOKENIZER_PATH` config | `RULER_TOKENIZER_DIR` convention | adapter exports `TOKENIZER_PATH` from the same directory |
| Decoding | temperature 0 (greedy) | temperature 0, `--max-tokens` | identical; `--max-tokens 512` mandatory at 32K+ (truncated thinking preambles read as false misses) |
| Scoring | per-task evaluators, containment of the golden string | exact/substring containment | the mapper (`map_results.py`) reproduces containment on the official predictions |

## Reproduction

```bash
cd /Users/dahai/fusion
source .venv/bin/activate
cd fusion-lsa

# tier server must be up (it is NOT started by this adapter unless --serve-cmd is given)
python scripts/official_ruler/run_official_ruler.py \
    --ruler-repo ~/work/RULER \
    --base-url http://localhost:11555/v1 \
    --model qwen38-27b \
    --length 32000 --seed 42 \
    --tasks niah,vt,fwe

# dry-run: print the exact command plan without executing
... --dry-run
```

Mapped results land in `results/official-ruler/official_ruler_results.json`
(schema: `summary.hits/total` + per-task details + `mean_accuracy`), next to
the in-repo runs in `results/`.

## Offline rules

Any model/tokenizer assets are fetched via the mirror
(`HF_MIRROR=https://hf-mirror.com`, exported by the adapter). No cloud
calls; the serving path is the local tier server.
