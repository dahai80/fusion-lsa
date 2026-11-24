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
      --model qwen38-27b --length 32000 --tasks niah,vt,fwe

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
from map_results import (  # noqa: E402
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
DATA_GEN_ARGS = {
    "niah": ["--task", "niah_single_1"],
    "vt": ["--task", "vt"],
    "fwe": ["--task", "fwe"],
}


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


def build_commands(repo: Path, base_url: str, model: str, length: int, seed: int,
                   tasks: list[str], max_tokens: int, tokenizer_dir: str) -> list[dict]:
    # The exact subprocess plan, in order. Pure function: the dry-run smoke
    # test asserts on this without executing anything.
    data_dir = repo / "data" / f"ruler_{length}_{seed}"
    pred_dir = repo / "results" / f"ruler_{length}_{seed}"
    env = {
        "OPENAI_API_BASE": base_url,
        "OPENAI_API_KEY": os.environ.get("FUSION_LSA_BENCH_KEY", "dummy"),
        "MODEL_NAME": model,
        "TOKENIZER_PATH": tokenizer_dir,
        "HF_MIRROR": "https://hf-mirror.com",
    }
    plan = []
    for task in tasks:
        if task not in DATA_GEN_ARGS:
            raise SystemExit(f"task {task} not wired (wired: {sorted(DATA_GEN_ARGS)})")
        plan.append({
            "desc": f"generate {task} data at length {length}",
            "argv": [sys.executable, str(repo / "scripts" / "data" / "synthetic.py"),
                     "--length", str(length), "--seed", str(seed)] + DATA_GEN_ARGS[task],
            "env": env,
        })
    plan.append({
        "desc": f"run official RULER eval for {','.join(tasks)} against {base_url}",
        "argv": [sys.executable, str(repo / "scripts" / "eval.py"),
                 "--data-dir", str(data_dir), "--output-dir", str(pred_dir),
                 "--max-tokens", str(max_tokens)],
        "env": env,
    })
    return plan


def collect_and_map(repo: Path, length: int, seed: int, tasks: list[str], output_dir: Path) -> list[dict]:
    results = []
    pred_dir = repo / "results" / f"ruler_{length}_{seed}"
    for task in tasks:
        matches = sorted(pred_dir.glob(f"*{task}*.jsonl"))
        if not matches:
            raise SystemExit(f"no prediction jsonl for task {task} under {pred_dir}")
        records = read_prediction_jsonl(matches[-1])
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
    parser.add_argument("--model", default="qwen38-27b")
    parser.add_argument("--length", type=int, default=32000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tasks", default="niah,vt,fwe")
    parser.add_argument("--max-tokens", type=int, default=512)
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
    plan = build_commands(repo, args.base_url, args.model, args.length, args.seed,
                          tasks, args.max_tokens, args.tokenizer_dir)
    if args.dry_run:
        for step in plan:
            logger.info("DRY-RUN %s:\n  env %s\n  %s", step["desc"], step["env"], " ".join(step["argv"]))
        return
    if args.serve_cmd:
        logger.info("booting tier server: %s", args.serve_cmd)
        subprocess.run(args.serve_cmd, shell=True, check=True)
    for step in plan:
        logger.info("running: %s", step["desc"])
        subprocess.run(step["argv"], env={**os.environ, **step["env"]}, check=True)
    collect_and_map(repo, args.length, args.seed, tasks, Path(args.output_dir))


if __name__ == "__main__":
    main()
