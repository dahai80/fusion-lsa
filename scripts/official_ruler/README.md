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

## Pipeline (pinned layout)

The pinned harness is a three-step pipeline run with `cwd=<RULER>/scripts`
(the scripts import `data.synthetic.constants` and read `synthetic.yaml`
via relative paths, so the cwd is mandatory):

1. `python data/prepare.py --save_dir <DATA> --benchmark synthetic --task <task> --tokenizer_path <dir> --tokenizer_type hf --max_seq_length <len> --model_template_type base --num_samples <n> --random_seed <seed>` — task names come from `scripts/synthetic.yaml` (`niah` maps to `niah_single_1`, `qa` to `qa_1`)
2. `python pred/call_api.py --data_dir <DATA> --save_dir <PRED> --benchmark synthetic --task <task> --server_type openai --model_name_or_path <api-model> --temperature 0 --top_p 1.0` — reads `OPENAI_API_KEY`/`OPENAI_BASE_URL` env
3. `python eval/evaluate.py --data_dir <PRED> --benchmark synthetic` — official per-task scorers

`pred/call_api.py` and `eval/evaluate.py` import nemo's
`manifest_utils` (JSONL reader/writer). The adapter injects
`ruler_compat/` via `PYTHONPATH`, which provides a faithful shim so the
full `nemo_toolkit` install is not needed.

`call_api.py`'s OpenAI client also requires `--model_name_or_path` to be a
key of its hardcoded `model2length` table (KeyError otherwise). The default
`--api-model-name gpt-4-0125-preview` (128000) satisfies it; the tier server
ignores the `model` field and serves its loaded model.

## Config deltas vs our in-repo generators

| Aspect | Official harness | Our bench | Delta handling |
|---|---|---|---|
| Model access | `OPENAI_API_KEY`/`OPENAI_BASE_URL` env + `--server_type openai` | fusion-mlx tier server `:11555` (OpenAI-compatible) | adapter exports both env vars pointing at the tier server |
| Seeds | per-example seeds derived from the sequence config | `--seed` per run | adapter passes `--random_seed` into the data-gen invocation; a `(length, seed)` pair fully determines the dataset |
| Tokenizer | `--tokenizer_path`/`--tokenizer_type` on `data/prepare.py` | `RULER_TOKENIZER_DIR` convention | adapter passes the same directory with `--tokenizer_type hf` |
| Decoding | temperature 1.0 default on `call_api.py` | temperature 0 (greedy) | adapter passes `--temperature 0 --top_p 1.0`; token budgets come from `synthetic.yaml` per task |
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
