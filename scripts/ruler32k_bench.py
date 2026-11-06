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


def build_prompt(target_tokens: int, seed: int) -> tuple[str, str]:
    rng = random.Random(seed)
    key_word = f"grasveld-{rng.randint(1000, 9999)}"
    value = f"{rng.randint(100000, 999999)}-{rng.randint(100, 999)}"
    needle = NEEDLE_TEMPLATE.format(key_word=key_word, value=value)
    filler_words = HAYSTACK_WORDS.split()
    words = []
    needle_words = needle.split()
    # ~1.3 tokens per English word; inject the needle about 60% deep
    needle_at = int(target_tokens / 1.3 * 0.6)
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
        "POST", f"{base_url}/chat/completions", json=payload, timeout=1800.0, headers=headers
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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    hits = 0
    tps_readings = []
    prompt, expected = build_prompt(args.length, seed=args.seed)
    logger.info("prompt ~%d tokens, expected value %s", args.length, expected)
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
            hit = expected in text
            hits += int(hit)
            if decode_tps > 0:
                tps_readings.append(decode_tps)
            logger.info(
                "run %d: hit=%s answer=%r wall=%.1fs decode_tps=%.1f",
                index, hit, text.strip()[:80], elapsed, decode_tps,
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
