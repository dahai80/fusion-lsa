"""TQ attention-KV probe: quantize the real attention K/V cache and measure the damage.

Reads a vectors-kv.npz produced by extract_nlsa_vectors.py --save-kv. Per layer, compares the
fp32 attention over the full document against attention over dequantized K/V at 2/4/8 bit:
per-channel K (KIVI style) with fp16 escape channels for outlier channels, per-token V.
Reports top-k key agreement and attention-output relative drift. Pure MLX, no model needed.

    python scripts/tq_kv_probe.py --vectors models/qwen38-27b/vectors-kv.npz --out docs/tq-kv-probe-27b.md
"""

import argparse
import logging
from pathlib import Path

import mlx.core as mx
import numpy as np

from fusion_lsa.quant.tq import HadamardRotation, TqQuantizer

logger = logging.getLogger(__name__)

ESCAPE_CHANNEL_COUNT = 8
TOPK = 64
BITS = (8, 4, 2)


def mask_logits(logits: mx.array, positions: mx.array) -> mx.array:
    # logits [Hkv, group, Q, T]; query row t may attend to key slots <= positions[t]
    slots = mx.arange(logits.shape[-1])
    allow = slots[None, :] <= positions[:, None]
    return mx.where(allow[None, None, :, :], logits, mx.array(-mx.inf, dtype=logits.dtype))


def topk_of(logits: mx.array) -> mx.array:
    # [-HkV, group, Q, T] -> top indices over the allowed region (masked slots sort last)
    return mx.argsort(-logits, axis=-1)[..., :TOPK]


def agreement(fp: mx.array, q: mx.array) -> float:
    # mean set overlap of the top-k key indices, per (kv head, group member, query)
    a_np, b_np = np.array(fp), np.array(q)
    scores = [
        len(set(a_np[i, j, t].tolist()) & set(b_np[i, j, t].tolist())) / TOPK
        for i in range(a_np.shape[0])
        for j in range(a_np.shape[1])
        for t in range(a_np.shape[2])
    ]
    return float(np.mean(scores))


def attend(logits: mx.array, v: mx.array) -> mx.array:
    # [Hkv, group, Q, T] x [Hkv, T, dh] -> mean over (kv head, group) read [Q, dh]
    probs = mx.softmax(logits, axis=-1)
    return mx.mean(probs @ v[:, None], axis=(0, 1))


def drift(a: mx.array, b: mx.array) -> float:
    return float((mx.linalg.norm(b - a, axis=-1) / mx.linalg.norm(a, axis=-1)).mean().item())


def quantize_heads(
    x: mx.array, bits: int, per_channel: bool, hadamard: HadamardRotation | None
) -> mx.array:
    # x [H, T, dh]. The Hadamard rotates the dh axis before quantizing and its inverse runs after
    # dequantization. per_channel flips to [dh, T] so the quantizer groups over T per channel and
    # the escape channels live in dh; per-token keeps [T, dh] rows and no escape channels.
    heads = []
    for h in range(x.shape[0]):
        if per_channel:
            block = hadamard.apply(x[h]) if hadamard is not None else x[h]  # [T, dh]
            block = block.transpose(1, 0)  # [dh, T]
            q = TqQuantizer(bits=bits, fp16_channels=escape_channels(block, ESCAPE_CHANNEL_COUNT))
            deq = q.dequantize(q.quantize(block)).transpose(1, 0)  # [T, dh]
        else:
            block = hadamard.apply(x[h]) if hadamard is not None else x[h]
            deq = TqQuantizer(bits=bits).dequantize(TqQuantizer(bits=bits).quantize(block))
        if hadamard is not None:
            deq = hadamard.invert(deq)
        heads.append(deq)
    return mx.stack(heads, axis=0)


def escape_channels(rows: mx.array, count: int) -> list[int]:
    # row indices with the largest magnitudes - the massive-activation channels
    norms = mx.abs(rows).max(axis=1)
    order = mx.argsort(-norms)
    return [int(order[i].item()) for i in range(min(count, rows.shape[0]))]


def probe_layer(
    q: mx.array, k: mx.array, v: mx.array, scale: float, positions: mx.array
) -> list[dict]:
    # q [Hq, Q, dh], k/v [Hkv, T, dh], already rotated; GQA group = Hq // Hkv
    heads_kv, _, dh = k.shape
    group = q.shape[0] // heads_kv
    qg = q.reshape(heads_kv, group, q.shape[1], dh)

    def logits_of(keys: mx.array) -> mx.array:
        # [Hkv, group, Q, T] - each query head group scores its own kv head
        return mask_logits((qg @ keys.transpose(0, 2, 1)[:, None]) * scale, positions)

    logits_fp = logits_of(k)
    out_fp = attend(logits_fp, v)
    topk_fp = topk_of(logits_fp)
    rot = HadamardRotation(k.shape[-1])
    rows = []
    for bits in BITS:
        for rotated in (False, True):
            k_deq = quantize_heads(k, bits, per_channel=True, hadamard=rot if rotated else None)
            v_deq = quantize_heads(v, bits, per_channel=False, hadamard=rot if rotated else None)
            logits_q = logits_of(k_deq)
            topk_q = topk_of(logits_q)
            out_q = attend(logits_q, v_deq)
            rows.append(
                {
                    "bits": bits,
                    "hadamard": rotated,
                    "topk_agreement": agreement(topk_fp, topk_q),
                    "output_drift": drift(out_fp, out_q),
                }
            )
            logger.info(
                "bits %d hadamard %s: top-%d agreement %.4f, output drift %.4f",
                bits,
                rotated,
                TOPK,
                rows[-1]["topk_agreement"],
                rows[-1]["output_drift"],
            )
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with np.load(args.vectors) as data:
        scale = float(data["meta.attention_scale"])
        positions = mx.array(data["positions"])
        prefixes = sorted(
            (
                name.split(".")[0]
                for name in data.files
                if name.startswith("layer") and name.endswith(".attn_v")
            ),
            key=lambda name: int(name.removeprefix("layer")),
        )
        layers = {
            prefix: (
                mx.array(data[f"{prefix}.attn_q"]).astype(mx.float32),
                mx.array(data[f"{prefix}.attn_k"]).astype(mx.float32),
                mx.array(data[f"{prefix}.attn_v"]).astype(mx.float32),
            )
            for prefix in prefixes
        }
        logger.info(
            "layers %s, q %s k %s v %s, scale %s",
            prefixes,
            layers[prefixes[0]][0].shape,
            layers[prefixes[0]][1].shape,
            layers[prefixes[0]][2].shape,
            scale,
        )

    lines = [
        "# TQ attention-KV probe on real checkpoint KV",
        "",
        f"Source: `{args.vectors}`, layers {', '.join(p[5:] for p in prefixes)}. "
        f"K quantized per-channel with {ESCAPE_CHANNEL_COUNT} fp16 escape channels, V per-token, "
        f"top-{TOPK} key agreement and attention-output relative drift against fp32, "
        "causal-masked, GQA groups averaged.",
        "",
    ]
    for prefix, (q, k, v) in layers.items():
        lines += [
            f"## {prefix} (decoder layer {prefix[5:]})",
            "",
            "| bits | hadamard | top-64 key agreement | output relative drift |",
            "|---|---|---|---|",
        ]
        for row in probe_layer(q, k, v, scale, positions):
            lines.append(
                f"| {row['bits']} | {row['hadamard']} | {row['topk_agreement']:.4f} | {row['output_drift']:.4f} |"
            )
        lines.append("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
