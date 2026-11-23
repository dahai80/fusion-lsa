# SPDX-License-Identifier: Apache-2.0
"""RULER-32k style NIAH smoke + decode-throughput bench against a fusion-mlx server.

Builds one long-needle-in-haystack prompt at a target token length, asks the
server to retrieve the magic key, and reports exact-match accuracy plus decode
tokens/second measured from streaming chunks (first token to last token, so
prefill wall time is excluded). Identical requests may be served from the
server response cache, so pass distinct seeds for independent samples.

Usage:
  python scripts/ruler32k_bench.py --base-url http://localhost:11555/v1 \
      --model qwen38-27b --length 32000 --repeats 3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import statistics
import time

import httpx

logger = logging.getLogger(__name__)

HAYSTACK_WORDS = (
    "The sun rises over the quiet valley and the farmers begin another day of work "
    "tending fields that have fed their families for generations. "
)
NEEDLE_TEMPLATE = (
    "One of the special magic numbers for {key_word} is: {value}. "
)
QUESTION_TEMPLATE = (
    "What is the special magic number for {key_word} mentioned in the provided text?"
)

# --- RULER VT (variable tracking): chained assignments, ask for the tail ---
VT_CHAIN_TEMPLATE = "VAR_{tag}{i} = {value}. "
VT_HOP_TEMPLATE = "VAR_{tag}{i} = VAR_{tag}{prev}. "
VT_QUESTION_TEMPLATE = (
    "What is the value of VAR_{tag}{n} mentioned in the provided text?"
)

# --- RULER FWE (frequent words extraction): coded words outnumber fillers ---
FWE_QUESTION_TEMPLATE = (
    "What are the 3 most frequent words in the provided text? "
    "Answer with the 3 words separated by commas."
)


def _filler_words(rng: random.Random, count: int) -> list[str]:
    # Unique coded filler tokens: each appears exactly once, so the repeated
    # coded words (counts >= 5) are guaranteed to be the most frequent words.
    # Natural sentences would let stop words ("the") outnumber the coded words
    # and make the ground truth wrong.
    words: list[str] = []
    seen: set[str] = set()
    while len(words) < count:
        token = f"zelvar{rng.randint(100000, 999999)}"
        if token in seen:
            continue
        seen.add(token)
        words.append(token)
    return words


def calibrate_prompt(build_fn, target_tokens: int, seed: int, tokenizer):
    # Synthetic coded fillers tokenize to ~8 tokens per word (vs ~1.3 for
    # natural text), so the word-level budget overshoots ~4x. Rebuild with
    # a rescaled target until the measured token count is within 5% of the
    # requested budget; coded words are drawn before filler sizing, so
    # ground truth is stable across rescales.
    target = target_tokens
    prompt, extra = build_fn(target, seed=seed)
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    for _ in range(2):
        if abs(len(ids) - target_tokens) <= 0.05 * target_tokens:
            break
        target = max(1024, int(target * target_tokens / len(ids)))
        prompt, extra = build_fn(target, seed=seed)
        ids = tokenizer(prompt, add_special_tokens=False).input_ids
    logger.info(
        "calibrated %s prompt to %d tokens (target %d)", build_fn.__name__, len(ids), target_tokens
    )
    return prompt, extra


def build_vt_prompt(target_tokens: int, seed: int, hops: int = 4):
    rng = random.Random(seed)
    tag = f"grasveld{rng.randint(1000, 9999)}"
    value = f"{rng.randint(100000, 999999)}-{rng.randint(100, 999)}"
    # chain head carries the literal value, hops 2..n copy the previous var;
    # each element sits at its own depth so the chain spans the context
    words = _filler_words(rng, target_tokens // 2)
    total = len(words)
    for i in range(1, hops + 1):
        piece = (
            VT_CHAIN_TEMPLATE.format(tag=tag, i=i, value=value)
            if i == 1
            else VT_HOP_TEMPLATE.format(tag=tag, i=i, prev=i - 1)
        )
        pos = int(total * i / (hops + 1))
        words[pos:pos] = piece.split()
    question = VT_QUESTION_TEMPLATE.format(tag=tag, n=hops)
    prompt = " ".join(words) + "\n\n" + question
    return prompt, value


def build_fwe_prompt(target_tokens: int, seed: int):
    rng = random.Random(seed)
    # 3 coded words with decreasing counts well above any filler word count;
    # drawn collision-free so the ground-truth set always has 3 distinct words
    coded: list[str] = []
    while len(coded) < 3:
        word = f"krovex{rng.randint(100, 999)}"
        if word not in coded:
            coded.append(word)
    counts = [max(8, target_tokens // 40), max(6, target_tokens // 55), max(5, target_tokens // 80)]
    filler_count = target_tokens // 2 - sum(counts)
    words = _filler_words(rng, max(1, filler_count))
    for word, count in zip(coded, counts):
        for _ in range(count):
            words.insert(rng.randrange(len(words) + 1), word)
    question = FWE_QUESTION_TEMPLATE
    prompt = " ".join(words) + "\n\n" + question
    return prompt, coded


# --- RULER QA: one fact-bearing passage among distractor essays ---
QA_FACT_TEMPLATE = (
    "The {key} partnership agreement was signed in {city}, and the "
    "ceremony lasted exactly {days} days. "
)
QA_QUESTION_TEMPLATE = (
    "In which city was the {key} partnership agreement signed?"
)
QA_CITIES = [
    "Ravensburg", "Thornbury", "Calderdale", "Mistelbach", "Ostrowiec",
    "Velenje", "Koprivnica", "Almelo", "Sonderborg", "Larvik",
]


def build_qa_prompt(target_tokens: int, seed: int, distractors: int = 7):
    rng = random.Random(seed)
    key = f"grasveld-{rng.randint(1000, 9999)}"
    gold_city = rng.choice(QA_CITIES)
    days = rng.randint(2, 9)
    other_cities = [c for c in QA_CITIES if c != gold_city]
    rng.shuffle(other_cities)
    gold_passage = QA_FACT_TEMPLATE.format(key=key, city=gold_city, days=days)
    # distractor essays mention other keys' agreements in other cities —
    # the question key must match only the gold passage
    passages = [
        QA_FACT_TEMPLATE.format(
            key=f"kelvor-{rng.randint(1000, 9999)}",
            city=other_cities[i % len(other_cities)],
            days=rng.randint(2, 9),
        )
        for i in range(distractors)
    ]
    insert_at = rng.randrange(len(passages) + 1)
    passages.insert(insert_at, gold_passage)
    filler_words = _filler_words(rng, max(1, target_tokens // 2 - 40 * len(passages)))
    third = max(1, len(filler_words) // (len(passages) + 1))
    chunks = []
    fill_pos = 0
    for passage in passages:
        chunks.append(" ".join(filler_words[fill_pos:fill_pos + third]))
        fill_pos += third
        chunks.append(passage)
    context = " ".join(chunks)
    prompt = context + "\n\n" + QA_QUESTION_TEMPLATE.format(key=key)
    return prompt, gold_city


# --- RULER CWE: 10 coded words with equal frequency, all must be named ---
CWE_QUESTION_TEMPLATE = (
    "What are the 10 most common words in the provided text? "
    "Answer with the 10 words separated by commas."
)


def build_cwe_prompt(target_tokens: int, seed: int, common: int = 10):
    rng = random.Random(seed)
    # every common word appears the same number of times (RULER CWE
    # semantics: a set of equally frequent coded words); filler words are
    # unique coded tokens appearing once, so the top-10 set is exact.
    # Drawn collision-free: random draws from 900 would collide ~5% of the
    # time at common=10 and silently shrink the ground truth to 9 words.
    coded: list[str] = []
    while len(coded) < common:
        word = f"myrvik{rng.randint(100, 999)}"
        if word not in coded:
            coded.append(word)
    repeats = max(3, target_tokens // 200)
    filler_count = target_tokens // 2 - common * repeats
    words = _filler_words(rng, max(1, filler_count))
    for word in coded:
        for _ in range(repeats):
            words.insert(rng.randrange(len(words) + 1), word)
    prompt = " ".join(words) + "\n\n" + CWE_QUESTION_TEMPLATE
    return prompt, coded


def build_prompt(
    target_tokens: int, seed: int, needle_depth: float = 0.6
) -> tuple[str, str]:
    rng = random.Random(seed)
    key_word = f"grasveld-{rng.randint(1000, 9999)}"
    value = f"{rng.randint(100000, 999999)}-{rng.randint(100, 999)}"
    needle = NEEDLE_TEMPLATE.format(key_word=key_word, value=value)
    filler_words = HAYSTACK_WORDS.split()
    words = []
    needle_words = needle.split()
    # ~1.3 tokens per English word; inject the needle at needle_depth
    # of the context (1.0 = end of context, right before the question)
    needle_at = min(int(target_tokens / 1.3 * needle_depth), target_tokens // 1.3 - 8)
    while len(words) < target_tokens // 1.3:
        if abs(len(words) - needle_at) < len(needle_words) // 2:
            words.extend(needle_words)
        else:
            words.append(rng.choice(filler_words))
    context = " ".join(words)
    prompt = context + "\n\n" + QUESTION_TEMPLATE.format(key_word=key_word)
    return prompt, value


def run_once(
    client: httpx.Client,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    api_key: str | None = None,
):
    # Streaming: decode tps = completion tokens between the first and last
    # chunk, excluding prefill wall time. A response served from the server
    # cache yields a near-zero decode window; vary the seed for fresh samples.
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    start = time.monotonic()
    text_parts = []
    first_at = None
    last_at = None
    with client.stream(
        "POST", f"{base_url}/chat/completions", json=payload, timeout=86400.0, headers=headers
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:].strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            delta = event.get("choices", [{}])[0].get("delta", {})
            piece = delta.get("content")
            if piece:
                now = time.monotonic()
                if first_at is None:
                    first_at = now
                last_at = now
                text_parts.append(piece)
    text = "".join(text_parts)
    elapsed = time.monotonic() - start
    decode_window = (last_at - first_at) if first_at and last_at and last_at > first_at else 0.0
    completion_tokens = max(1, len(text_parts))
    decode_tps = (completion_tokens - 1) / decode_window if decode_window > 0 else 0.0
    return text, elapsed, completion_tokens, decode_tps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:11555/v1")
    parser.add_argument("--model", default="qwen38-27b")
    parser.add_argument("--length", type=int, default=32000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--needle-depth",
        type=float,
        default=0.6,
        help="needle position as fraction of context depth (1.0 = end)",
    )
    parser.add_argument(
        "--task",
        choices=["niah", "vt", "fwe", "qa", "cwe"],
        default="niah",
        help="RULER subtask: niah (single needle), vt (variable tracking chain), fwe (frequent words), qa (fact-bearing passage among distractors), cwe (common words)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    hits = 0
    tps_readings = []
    coded = None
    tokenizer = None
    if args.task != "niah":
        from transformers import AutoTokenizer
        tok_dir = os.environ.get("RULER_TOKENIZER_DIR", "models/qwen38-27b")
        tokenizer = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)
    if args.task == "vt":
        prompt, expected = calibrate_prompt(build_vt_prompt, args.length, args.seed, tokenizer)
    elif args.task == "fwe":
        prompt, coded = calibrate_prompt(build_fwe_prompt, args.length, args.seed, tokenizer)
        expected = ", ".join(coded)
    elif args.task == "qa":
        prompt, expected = calibrate_prompt(build_qa_prompt, args.length, args.seed, tokenizer)
    elif args.task == "cwe":
        prompt, coded = calibrate_prompt(build_cwe_prompt, args.length, args.seed, tokenizer)
        expected = ", ".join(coded)
    else:
        prompt, expected = build_prompt(
            args.length, seed=args.seed, needle_depth=args.needle_depth
        )
    logger.info("task=%s prompt ~%d tokens, expected %s", args.task, args.length, expected)
    probe_lo = -1
    if os.environ.get("RULER_PROBE_TOKENS"):
        from transformers import AutoTokenizer
        tok_dir = os.environ.get("RULER_TOKENIZER_DIR", "models/qwen38-27b")
        tokenizer = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)
        needle_chars = prompt.find("One of the special magic numbers")
        head_ids = tokenizer(prompt[:needle_chars], add_special_tokens=True).input_ids
        needle_ids = tokenizer(prompt[needle_chars:], add_special_tokens=False).input_ids
        probe_lo = len(head_ids)
        logger.info("needle token span [%d, ...] (%d needle tokens)", probe_lo, len(needle_ids))
    with httpx.Client() as client:
        for index in range(args.repeats):
            text, elapsed, _completion_tokens, decode_tps = run_once(
                client, args.base_url, args.model, prompt, args.max_tokens, args.api_key
            )
            hit = all(word in text for word in coded) if coded else expected in text
            hits += int(hit)
            if decode_tps > 0:
                tps_readings.append(decode_tps)
            logger.info(
                "run %d: hit=%s answer=%r wall=%.1fs decode_tps=%.1f",
                index, hit, text.strip()[:500], elapsed, decode_tps,
            )
    summary = {
        "hits": hits,
        "repeats": args.repeats,
        "accuracy": hits / args.repeats,
        "mean_decode_tps": statistics.mean(tps_readings) if tps_readings else 0.0,
        "length": args.length,
        "seed": args.seed,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
