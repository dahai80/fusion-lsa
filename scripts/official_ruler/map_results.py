# SPDX-License-Identifier: Apache-2.0
"""Map official NVIDIA RULER outputs to the fusion-lsa results schema.

The official RULER harness writes per-example prediction records (one JSON
object per line, with the model prediction and the golden answer). This
module maps those records onto the schema our in-repo generators emit
(``summary.hits/total`` + per-task details) so numbers from both harnesses
are directly comparable, and writes the mapped JSON under
``results/official-ruler/``.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Field names the official RULER harness has used for predictions and gold
# answers across versions. The mapper accepts any of them so a pinned-version
# bump that renames fields degrades loudly (ValueError) instead of silently
# scoring zero.
PRED_FIELDS = ("pred", "prediction", "output", "response")
GOLD_FIELDS = ("golden", "answer", "gold", "label")

# The pinned pipeline (db43f92ff634) stores the gold answer in the pred jsonl
# record under `outputs` (call_api.py copies data_point["outputs"] verbatim,
# and eval/evaluate.py reads references from that same field), so no merge
# with the data jsonl is needed. `outputs` is a list of gold strings
# (stringified lists tolerated); `others` is empty at this pin.
GOLD_LIST_FIELD = "outputs"

# Official scoring per task family, scripts/eval/synthetic/constants.py at the
# pin: string_match_all for niah/vt/cwe/fwe (EVERY ref must be contained in
# the prediction; per-record score is the contained fraction), string_match_part
# for qa (ANY ref). All containment is case-insensitive after postprocess_pred
# (strip + control chars folded to newline).
MATCH_MODE_BY_TASK = {"niah": "all", "vt": "all", "fwe": "all", "cwe": "all", "qa": "part"}


def _pick(record: dict, fields: tuple[str, ...], what: str, line_no: int) -> str:
    for field in fields:
        if field in record and record[field] is not None:
            value = record[field]
            return value if isinstance(value, str) else json.dumps(value)
    raise ValueError(f"line {line_no}: no {what} field (looked for {fields})")


def _normalize_refs(value) -> list[str]:
    # Accept a list of gold strings, a single string, or a stringified list
    # ("['9275605']") as written by some data generators. None elements are
    # dropped (official eval skips them the same way).
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                value = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                return [text]
        else:
            return [text]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _gold_refs(record: dict, line_no: int) -> list[str]:
    if GOLD_LIST_FIELD in record and record[GOLD_LIST_FIELD] is not None:
        refs = _normalize_refs(record[GOLD_LIST_FIELD])
        if refs:
            return refs
    for field in GOLD_FIELDS:
        if field in record and record[field] is not None:
            return _normalize_refs(record[field])
    raise ValueError(
        f"line {line_no}: no golden answer field (looked for {GOLD_FIELDS + (GOLD_LIST_FIELD,)})"
    )


def _postprocess_pred(predict_str: str) -> str:
    # Mirror of the pinned eval/evaluate.py postprocess_pred: strip, fold
    # non-printable characters to newline, strip again.
    predict_str = predict_str.strip()
    predict_str = re.compile(r"[\x00-\x1f]").sub("\n", predict_str).strip()
    return predict_str


def map_prediction_records(records: list[dict], task: str, length: int) -> dict:
    # One RULER per-example jsonl -> our summary schema. Scoring mirrors the
    # pinned official evaluators: case-insensitive containment, all-refs for
    # niah/vt/fwe/cwe, any-ref for qa. `hits` accumulates the per-record
    # score (fraction of contained refs for all-mode), so accuracy matches
    # the official summary.csv exactly for single-ref tasks.
    mode = MATCH_MODE_BY_TASK.get(task, "all")
    hits = 0.0
    details = []
    for line_no, record in enumerate(records, 1):
        pred = _postprocess_pred(_pick(record, PRED_FIELDS, "prediction", line_no))
        refs = _gold_refs(record, line_no)
        contained = [1.0 if ref.lower() in pred.lower() else 0.0 for ref in refs]
        if mode == "part":
            score = max(contained)
        else:
            score = sum(contained) / len(contained)
        hits += score
        details.append({
            "hit": score == 1.0,
            "score": score,
            "pred": pred[:500],
            "golden": refs,
        })
    if not records:
        raise ValueError("no prediction records to map")
    return {
        "task": task,
        "length": length,
        "hits": hits,
        "total": len(records),
        "accuracy": hits / len(records),
        "details": details,
    }


def read_prediction_jsonl(path: str | Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSON: {exc}") from exc
    return records


def write_mapped_results(results: list[dict], output_dir: str | Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "official_ruler_results.json"
    summary = {
        "harness": "NVIDIA/RULER (official)",
        "pinned_commit": PINNED_RULER_COMMIT,
        "tasks": results,
        "mean_accuracy": sum(r["accuracy"] for r in results) / len(results) if results else 0.0,
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("wrote %s (%d tasks, mean accuracy %.3f)", path, len(results), summary["mean_accuracy"])
    return path


PINNED_RULER_COMMIT = "db43f92ff634"
