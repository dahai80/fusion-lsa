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

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Field names the official RULER harness has used for predictions and gold
# answers across versions. The mapper accepts any of them so a pinned-version
# bump that renames fields degrades loudly (ValueError) instead of silently
# scoring zero.
PRED_FIELDS = ("pred", "prediction", "output", "response")
GOLD_FIELDS = ("golden", "answer", "gold", "label")


def _pick(record: dict, fields: tuple[str, ...], what: str, line_no: int) -> str:
    for field in fields:
        if field in record and record[field] is not None:
            value = record[field]
            return value if isinstance(value, str) else json.dumps(value)
    raise ValueError(f"line {line_no}: no {what} field (looked for {fields})")


def map_prediction_records(records: list[dict], task: str, length: int) -> dict:
    # One RULER per-example jsonl -> our summary schema. Substring scoring
    # mirrors the official RULER niah/vt/fwe/qa/cwe evaluators (they all
    # score containment of the golden string in the prediction).
    hits = 0
    details = []
    for line_no, record in enumerate(records, 1):
        pred = _pick(record, PRED_FIELDS, "prediction", line_no)
        gold = _pick(record, GOLD_FIELDS, "golden answer", line_no)
        hit = gold in pred
        hits += int(hit)
        details.append({"hit": hit, "pred": pred[:500], "golden": gold[:200]})
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
