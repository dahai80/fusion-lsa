import logging
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from ruler32k_bench import build_cwe_prompt, build_fwe_prompt, build_qa_prompt, build_vt_prompt

logger = logging.getLogger(__name__)


def test_vt_build_chain_and_question():
    prompt, expected = build_vt_prompt(2000, seed=41)
    assert len(expected.split("-")) == 2
    head = re.search(r"VAR_(grasveld\d+)1 = (\d+-\d+)\.", prompt)
    assert head is not None, "chain head with literal value missing"
    assert head.group(2) == expected
    tag = head.group(1)
    for i in range(2, 5):
        assert f"VAR_{tag}{i} = VAR_{tag}{i - 1}." in prompt
    assert f"What is the value of VAR_{tag}4" in prompt
    logger.info("vt build ok: tag=%s expected=%s", tag, expected)


def test_vt_hops_span_context_depths():
    prompt, expected = build_vt_prompt(2000, seed=42)
    head = re.search(r"VAR_(grasveld\d+)1 = ", prompt)
    assert head is not None
    tag = head.group(1)
    head_pos = prompt.index(f"VAR_{tag}1 = {expected}.")
    assert head_pos < len(prompt) / 3, "chain head must sit in the early context"
    positions = [prompt.index(f"= VAR_{tag}{i}.") for i in (1, 2, 3)]
    assert positions == sorted(positions), "hops must sit at increasing depths"
    assert positions[-1] > 2 * len(prompt) / 3, "last hop must sit deep in context"
    logger.info("vt hop positions: %s / len=%d", positions, len(prompt))


def test_fwe_coded_words_dominate():
    prompt, coded = build_fwe_prompt(2000, seed=43)
    assert len(coded) == 3
    words = prompt.split("\n\n")[0].split()
    counts = Counter(words)
    coded_counts = [counts[w] for w in coded]
    filler_counts = [c for w, c in counts.items() if w not in coded]
    assert all(c >= 5 for c in coded_counts), coded_counts
    assert min(coded_counts) > max(filler_counts), (
        f"coded {min(coded_counts)} vs filler max {max(filler_counts)}"
    )
    logger.info(
        "fwe ok: coded=%s filler_max=%d total_words=%d",
        coded_counts,
        max(filler_counts),
        len(words),
    )


@pytest.mark.parametrize("target,seed", [(2000, 41), (8000, 42), (32000, 43)])
def test_fwe_dominates_across_lengths(target, seed):
    prompt, coded = build_fwe_prompt(target, seed=seed)
    words = prompt.split("\n\n")[0].split()
    counts = Counter(words)
    assert min(counts[w] for w in coded) > max(
        c for w, c in counts.items() if w not in coded
    )
    logger.info("fwe dominance at %d tokens ok", target)


def test_qa_gold_fact_present_and_key_unique():
    prompt, city = build_qa_prompt(4000, seed=41)
    assert city in prompt, "gold city must appear in context"
    question = prompt.split("\n\n")[-1]
    key = re.search(r"grasveld-\d+", question).group(0)
    # the questioned key's fact names the gold city exactly once in context
    fact = re.search(rf"The {re.escape(key)} partnership agreement was signed in (\w+)", prompt)
    assert fact is not None, "gold fact missing for the questioned key"
    assert fact.group(1) == city
    logger.info("qa ok: key=%s city=%s", key, city)


def test_qa_distractor_cities_differ():
    prompt, city = build_qa_prompt(4000, seed=42)
    key = re.search(r"grasveld-\d+", prompt.split("\n\n")[-1]).group(0)
    # no distractor fact may use the gold city
    for m in re.finditer(r"The (kelvor-\d+) partnership agreement was signed in (\w+)", prompt):
        assert not (m.group(2) == city), f"distractor {m.group(1)} reuses gold city"
    logger.info("qa distractors ok, city=%s", city)


@pytest.mark.parametrize("target,seed", [(2000, 41), (32000, 42)])
def test_cwe_common_words_tie_and_dominate(target, seed):
    prompt, coded = build_cwe_prompt(target, seed=seed)
    assert len(coded) == 10
    words = prompt.split("\n\n")[0].split()
    counts = Counter(words)
    coded_counts = {counts[w] for w in coded}
    assert len(coded_counts) == 1, f"common words must tie: {coded_counts}"
    filler_max = max(c for w, c in counts.items() if w not in coded)
    assert min(coded_counts) > filler_max, (coded_counts, filler_max)
    logger.info("cwe ok at %d: count=%s filler_max=%d", target, coded_counts, filler_max)
