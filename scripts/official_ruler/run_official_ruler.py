# SPDX-License-Identifier: Apache-2.0
"""Drive the official NVIDIA RULER harness against the fusion-mlx tier server.

Parallel path only: the in-repo generators (scripts/ruler32k_bench.py) and
their outputs are NOT touched. This adapter boots/uses the same fusion-mlx
tier server (:11555, OpenAI-compatible), runs the official RULER data
generation + eval at the pinned commit, and maps the predictions onto the
fusion-lsa results schema (see map_results.py).

Usage:
  python scripts/official_ruler/run_official_ruler.py \
      --ruler-repo ~/work/RULER --base-url http://localhost:11555/v1 \
      --length 32000 --tasks niah,vt,fwe

The pinned pipeline is data/prepare.py -> pred/call_api.py ->
eval/evaluate.py, all run with cwd=<RULER>/scripts. The pinned call_api
imports nemo's manifest_utils; a faithful shim lives in ruler_compat/ and is
injected via PYTHONPATH so the full nemo_toolkit install is not required.

Dry-run (prints the exact commands, runs nothing - covered by the smoke
test):
  ... --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from map_results import (
    PINNED_RULER_COMMIT,
    map_prediction_records,
    read_prediction_jsonl,
    write_mapped_results,
)

logger = logging.getLogger(__name__)

# Config deltas vs our in-repo generators (documented in README.md here):
# - Model access: the official harness reads the endpoint from
#   OPENAI_API_BASE / MODEL_NAME env vars; we point them at the fusion-mlx
#   tier server instead of a vLLM process.
# - Seeds: the official harness derives per-example seeds from its own
#   sequence config; we pass --seed through its sequence-length config so a
#   (length, seed) pair fully determines the dataset.
# - Tokenizer: the official harness uses its tokenizer path config; we point
#   it at the served model's tokenizer directory (RULER_TOKENIZER_DIR
#   convention from our bench).
# Our task labels -> task names in the pinned scripts/synthetic.yaml
# (data/prepare.py resolves generators through that file).
TASK_TO_YAML = {
    "niah": "niah_single_1",
    "vt": "vt",
    "fwe": "fwe",
    "cwe": "cwe",
    "qa": "qa_1",
}

# The pinned pred/call_api.py OpenAI client requires --model_name_or_path to
# be a key of its hardcoded model2length table (KeyError otherwise); it uses
# that value only for token accounting and as the `model` field of the chat
# request. The fusion-mlx tier server serves whatever is loaded and ignores
# the field, so we default to the 128000-entry key and never truncate.
DEFAULT_API_MODEL = "gpt-4-0125-preview"

COMPAT_DIR = Path(__file__).resolve().parent / "ruler_compat"


def verify_pinned_commit(repo: Path, allow_any_commit: bool = False) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    commit = result.stdout.strip()
    if not commit.startswith(PINNED_RULER_COMMIT) and not allow_any_commit:
        raise SystemExit(
            f"RULER repo at {repo} is at {commit[:12]}, pinned commit is "
            f"{PINNED_RULER_COMMIT}. Check out the pin or pass --allow-any-commit "
            "and document the delta."
        )
    logger.info("RULER repo %s at pinned commit %s", repo, commit[:12])
    return commit


def build_commands(repo: Path, base_url: str, length: int, seed: int, tasks: list[str],
                   tokenizer_dir: str, num_samples: int = 500,
                   api_model: str = DEFAULT_API_MODEL, template: str = "base") -> list[dict]:
    # The exact subprocess plan, in order, matching the pinned layout: per
    # task a data/prepare.py step then a pred/call_api.py step, and one
    # eval/evaluate.py over all predictions at the end. All three scripts
    # require cwd=<RULER>/scripts (they import data.<benchmark>.constants and
    # use relative yaml paths). Pure function: the dry-run smoke test asserts
    # on this without executing anything.
    scripts_dir = repo / "scripts"
    data_dir = scripts_dir / "benchmark_root" / f"ruler_{length}_{seed}" / "data"
    pred_dir = scripts_dir / "benchmark_root" / f"ruler_{length}_{seed}" / "pred"
    env = {
        "OPENAI_API_BASE": base_url,
        "OPENAI_BASE_URL": base_url,
        "OPENAI_API_KEY": os.environ.get("FUSION_LSA_BENCH_KEY", "dummy"),
        "HF_MIRROR": "https://hf-mirror.com",
        "PYTHONPATH": str(COMPAT_DIR) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    plan = []
    for task in tasks:
        if task not in TASK_TO_YAML:
            raise SystemExit(f"task {task} not wired (wired: {sorted(TASK_TO_YAML)})")
        yaml_task = TASK_TO_YAML[task]
        plan.append({
            "desc": f"generate {task} data at length {length}",
            "argv": [sys.executable, "data/prepare.py",
                     "--save_dir", str(data_dir),
                     "--benchmark", "synthetic",
                     "--task", yaml_task,
                     "--tokenizer_path", tokenizer_dir,
                     "--tokenizer_type", "hf",
                     "--max_seq_length", str(length),
                     "--model_template_type", template,
                     "--num_samples", str(num_samples),
                     "--random_seed", str(seed)],
            "env": env,
            "cwd": str(scripts_dir),
        })
        plan.append({
            "desc": f"predict {task} against {base_url}",
            "argv": [sys.executable, "pred/call_api.py",
                     "--data_dir", str(data_dir),
                     "--save_dir", str(pred_dir),
                     "--benchmark", "synthetic",
                     "--task", yaml_task,
                     "--server_type", "openai",
                     "--model_name_or_path", api_model,
                     "--temperature", "0",
                     "--top_p", "1.0"],
            "env": env,
            "cwd": str(scripts_dir),
        })
    plan.append({
        "desc": f"run official RULER eval for {','.join(tasks)}",
        "argv": [sys.executable, "eval/evaluate.py",
                 "--data_dir", str(pred_dir),
                 "--benchmark", "synthetic"],
        "env": env,
        "cwd": str(scripts_dir),
    })
    return plan


def collect_and_map(repo: Path, length: int, seed: int, tasks: list[str], output_dir: Path) -> list[dict]:
    results = []
    pred_dir = repo / "scripts" / "benchmark_root" / f"ruler_{length}_{seed}" / "pred"
    for task in tasks:
        yaml_task = TASK_TO_YAML[task]
        pred_file = pred_dir / f"{yaml_task}.jsonl"
        if not pred_file.exists():
            raise SystemExit(f"no prediction jsonl for task {task} at {pred_file}")
        records = read_prediction_jsonl(pred_file)
        results.append(map_prediction_records(records, task, length))
    write_mapped_results(results, output_dir)
    return results


def verify_server(base_url: str) -> dict:
    # The tier server must already serve the OpenAI-compatible surface before
    # the official harness starts; fail loudly here instead of after the
    # (expensive) data generation.
    import urllib.error
    import urllib.request

    url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise SystemExit(f"tier server at {base_url} is not serving /models: {exc}") from exc
    logger.info("tier server %s serves %d model(s)", base_url, len(payload.get("data", [])))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ruler-repo", required=True, help="clone of NVIDIA/RULER at the pinned commit")
    parser.add_argument("--base-url", default="http://localhost:11555/v1")
    parser.add_argument("--length", type=int, default=32000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tasks", default="niah,vt,fwe")
    parser.add_argument("--num-samples", type=int, default=500,
                        help="samples per task; the official default is 500")
    parser.add_argument("--api-model-name", default=DEFAULT_API_MODEL,
                        help="model string sent to the server; must be a key of the "
                             "pinned call_api.py model2length table (default fits it)")
    parser.add_argument("--tokenizer-dir", default=os.environ.get("RULER_TOKENIZER_DIR", "models/qwen38-27b"))
    parser.add_argument("--output-dir", default="results/official-ruler")
    parser.add_argument("--serve-cmd", default=None,
                        help="optional shell command that boots the fusion-mlx tier server before the run")
    parser.add_argument("--dry-run", action="store_true", help="print the command plan and exit")
    parser.add_argument("--allow-any-commit", action="store_true",
                        help="run on a non-pinned RULER commit (document the delta in the results PR)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    repo = Path(args.ruler_repo).expanduser()
    tasks = args.tasks.split(",")
    verify_pinned_commit(repo, args.allow_any_commit)
    if not args.dry_run:
        verify_server(args.base_url)
    plan = build_commands(repo, args.base_url, args.length, args.seed, tasks,
                          args.tokenizer_dir, num_samples=args.num_samples,
                          api_model=args.api_model_name)
    if args.dry_run:
        for step in plan:
            logger.info("DRY-RUN %s:\n  cwd %s\n  env %s\n  %s",
                        step["desc"], step["cwd"], step["env"], " ".join(step["argv"]))
        return
    if args.serve_cmd:
        logger.info("booting tier server: %s", args.serve_cmd)
        subprocess.run(args.serve_cmd, shell=True, check=True)
    for step in plan:
        logger.info("running: %s", step["desc"])
        subprocess.run(step["argv"], cwd=step["cwd"], env={**os.environ, **step["env"]}, check=True)
    collect_and_map(repo, args.length, args.seed, tasks, Path(args.output_dir))


if __name__ == "__main__":
    main()
