import json
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from ruler32k_bench import (
    _normalize_answer,
    build_longbench_qa_prompt,
    load_longbench_records,
    qa_f1,
    qa_substring_hit,
)

logger = logging.getLogger(__name__)


def test_normalize_answer_strips_articles_punct_case():
    assert _normalize_answer("The Quick-Brown Fox!") == "quick brown fox"
    assert _normalize_answer("  a  AN the ") == ""
    assert _normalize_answer("New York City.") == "new york city"


def test_qa_f1_exact_partial_and_empty():
    assert qa_f1("Paris", ["Paris"]) == 1.0
    # Normalization must not cost F1 when the prediction is exactly the gold.
    assert qa_f1("paris.", ["Paris"]) == 1.0
    # Extra prediction tokens dilute token-level F1 (LongBench convention).
    assert qa_f1("The answer is: Paris.", ["paris"]) == 0.5
    # Partial overlap: token-level F1, max over gold answers.
    score = qa_f1("Paris France", ["Paris"])
    assert 0.0 < score < 1.0
    assert qa_f1("", ["Paris"]) == 0.0
    assert qa_f1("Berlin", ["London", "Paris"]) == 0.0
    assert qa_f1("nothing relevant at all", []) == 0.0


def test_qa_substring_hit_normalized():
    assert qa_substring_hit("The capital is Paris.", ["paris"])
    assert not qa_substring_hit("The capital is Lyon.", ["Paris"])
    assert qa_substring_hit("PARIS", ["paris"])
    assert not qa_substring_hit("anything", [])


def test_longbench_prompt_builder_deterministic_and_length_controlled():
    record = {
        "question": "Who signed the treaty?",
        "passages": [f"Passage {i} talks about topic {i} with some detail text." for i in range(4)],
        "answers": ["Alice"],
    }
    prompt_a, answers_a = build_longbench_qa_prompt(record, 2000, seed=41)
    prompt_b, answers_b = build_longbench_qa_prompt(record, 2000, seed=41)
    assert prompt_a == prompt_b, "same seed must give the same prompt"
    assert answers_a == ["Alice"]
    # Word budget follows the ~1.3 tokens-per-word convention of the other
    # tasks; the question sits after the context, outside the budget.
    context = prompt_a.split("\n\n")[0]
    assert len(context.split()) <= int(2000 // 1.3)
    short, _ = build_longbench_qa_prompt(record, 40, seed=41)
    assert len(short.split("\n\n")[0].split()) <= int(40 // 1.3), "prompt must truncate to the budget"
    assert short.endswith("Who signed the treaty?"), "question sits at the end"


def test_longbench_prompt_pads_by_cycling_passages():
    record = {
        "question": "Q?",
        "passages": ["alpha beta gamma delta epsilon"],
        "answers": ["x"],
    }
    prompt, _ = build_longbench_qa_prompt(record, 500, seed=41)
    assert prompt.split("\n\n")[0].count("alpha beta gamma delta epsilon") >= 5, "short records must cycle to fill the budget"


def test_load_longbench_records_validates_schema(tmp_path):
    good = tmp_path / "ok.jsonl"
    good.write_text(
        json.dumps({"question": "q", "passages": ["p"], "answers": ["a"]}) + "\n"
        + json.dumps({"question": "q2", "passages": ["p2"], "answers": ["a2"]}) + "\n"
    )
    records = load_longbench_records(str(good))
    assert len(records) == 2
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"question": "q", "passages": ["p"]}) + "\n")
    with pytest.raises(ValueError, match="missing keys"):
        load_longbench_records(str(bad))
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    with pytest.raises(ValueError, match="no records"):
        load_longbench_records(str(empty))
