"""TQ 4-bit probe: quantize real index keys at several bit widths and measure selection damage.

Reads a vectors.npz produced by extract_nlsa_vectors.py (real checkpoint index vectors), and for
each layer compares the fp32 baseline selection against selections from dequantized-renormalized
keys at 2/4/8 bit, with and without the Hadamard rotation. Reports bucket partition agreement,
selection set overlap and the mean overlap per bit width. Pure MLX, no model needed.

    python scripts/tq_probe.py --vectors models/qwen38-27b/vectors.npz --out docs/tq-probe-27b.md
"""

import argparse
import logging
from pathlib import Path

import mlx.core as mx
import numpy as np

from fusion_lsa.core.reference import build_buckets, read
from fusion_lsa.quant.tq import HadamardRotation, TqQuantizer

logger = logging.getLogger(__name__)


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    want = set(int(x) for x in a if x >= 0)
    got = set(int(x) for x in b if x >= 0)
    return len(want & got) / max(len(want), 1)


def probe_layer(keys: mx.array, queries: mx.array, positions: mx.array, meta: dict) -> list[dict]:
    rows = []
    buckets_fp = build_buckets(
        keys, meta["pool_size"], meta["bucket_size"], meta["bucket_cap"], meta["window_size"]
    )
    selected_fp, _ = read(
        queries, positions, buckets_fp, meta["top_p"], meta["top_k"], meta["sink_size"], meta["window_size"]
    )
    selected_fp = np.array(selected_fp)
    rot = HadamardRotation(keys.shape[1])
    for bits in (8, 4, 2):
        for rotated in (False, True):
            prepared = rot.apply(keys) if rotated else keys
            q = TqQuantizer(bits=bits)
            deq = q.dequantize(q.quantize(prepared))
            if rotated:
                deq = rot.invert(deq)
            deq = deq / mx.linalg.norm(deq, axis=1, keepdims=True)
            buckets_q = build_buckets(
                deq, meta["pool_size"], meta["bucket_size"], meta["bucket_cap"], meta["window_size"]
            )
            selected_q, _ = read(
                queries, positions, buckets_q, meta["top_p"], meta["top_k"], meta["sink_size"], meta["window_size"]
            )
            selected_q = np.array(selected_q)
            overlaps = [overlap(selected_fp[r], selected_q[r]) for r in range(selected_fp.shape[0])]
            # bucket partition agreement: identical centroid-argmax partitions are not directly
            # comparable across clusterings, so compare through the selection they produce
            rows.append(
                {
                    "bits": bits,
                    "hadamard": rotated,
                    "mean_overlap": float(np.mean(overlaps)),
                    "min_overlap": float(np.min(overlaps)),
                }
            )
            logger.info("bits %d hadamard %s: overlap %.4f (min %.4f)", bits, rotated, rows[-1]["mean_overlap"], rows[-1]["min_overlap"])
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--top-p", type=int, default=None, help="override the npz top_p - use a tight value when the document is short")
    parser.add_argument("--top-k", type=int, default=None, help="override the npz top_k")
    args = parser.parse_args()

    with np.load(args.vectors) as data:
        meta = {name.removeprefix("meta."): int(data[name]) for name in data.files if name.startswith("meta.")}
        if args.top_p is not None:
            meta["top_p"] = args.top_p
        if args.top_k is not None:
            meta["top_k"] = args.top_k
        positions = mx.array(data["positions"])
        prefixes = sorted(k.split(".")[0] for k in data.files if k.startswith("layer") and k.endswith(".index_keys"))
        layers = {}
        for prefix in prefixes:
            layers[prefix] = (mx.array(data[f"{prefix}.index_keys"]), mx.array(data[f"{prefix}.index_queries"]))

    lines = ["# TQ 4-bit probe on real checkpoint index vectors", "",
             f"Source: `{args.vectors}`, layers {', '.join(p[5:] for p in prefixes)}, "
             f"{layers[prefixes[0]][0].shape[0]} keys, index dim {layers[prefixes[0]][0].shape[1]}.", "",
             "Metric: mean/min per-query selection set overlap against the fp32 baseline "
             "(same buckets config, same queries). Keys dequantized then renormalized.", ""]
    for prefix, (keys, queries) in layers.items():
        lines.append(f"## {prefix} (decoder layer {prefix[5:]})", )
        lines.append("")
        lines.append("| bits | hadamard | mean overlap | min overlap |")
        lines.append("|---|---|---|---|")
        for row in probe_layer(keys, queries, positions, meta):
            lines.append(f"| {row['bits']} | {row['hadamard']} | {row['mean_overlap']:.4f} | {row['min_overlap']:.4f} |")
        lines.append("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
