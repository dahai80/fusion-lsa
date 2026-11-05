"""Extract real index vectors and torch-reference selections from the N-LSA 0.8B checkpoint.

Runs on CPU in the fusion venv (torch + transformers 5.x). Loads the checkpoint through the
upstream adapter (Qwen3_5ForConditionalGeneration), forwards one synthetic document, captures
the full-attention layers' inputs, then per layer builds the teacher view (index queries/keys)
and the torch reference selection. Everything lands in one npz the MLX parity test reads.

    python scripts/extract_nlsa_vectors.py \
        --checkpoint models/nlsa-checkpoint/decay128k-lsa-1024 \
        --out models/nlsa-checkpoint/vectors.npz
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger("extract_nlsa_vectors")

DEFAULT_TOKENS = 16384
DEFAULT_QUERY_COUNT = 32
SLICE = 2048


def upstream_path() -> str:
    path = os.environ.get("LSA_UPSTREAM", str(Path.home() / "model/latent-sparse-attention/src"))
    if not Path(path).is_dir():
        raise SystemExit(f"upstream src not found at {path} - set LSA_UPSTREAM")
    return path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="extracted HF checkpoint directory")
    parser.add_argument("--out", required=True, help="output npz path")
    parser.add_argument("--tokens", type=int, default=DEFAULT_TOKENS, help="document length")
    parser.add_argument("--queries", type=int, default=DEFAULT_QUERY_COUNT, help="sampled query count")
    parser.add_argument("--layers", default="", help="comma-separated decoder layers, default all full-attention")
    args = parser.parse_args()

    sys.path.insert(0, upstream_path())
    from lsa.evaluation import adapter
    from lsa.evaluation.config import ModelConfig
    from lsa.sglang.settings.design import IndexerConfig
    from lsa.training.distillation.data.teacher_capture import capture_layer_inputs
    from lsa.core import reference as torch_reference

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_dir():
        raise SystemExit(f"checkpoint directory not found: {checkpoint}")

    model_config = ModelConfig(checkpoint=str(checkpoint), dtype="float32")
    model_adapter = adapter.adapter_for(model_config)
    logger.info("adapter %s for %s", type(model_adapter).__name__, checkpoint)
    model, tokenizer = model_adapter.load(model_config, "cpu", attention="sdpa")
    text_model = model_adapter.text_model(model)
    text_config = model_adapter.text_config(model_config)
    every_layer = model_adapter.full_attention_layers(text_config, model_config)
    layers = [int(x) for x in args.layers.split(",") if x] if args.layers else every_layer
    rotary = model_adapter.rotary_layout(model, model_config, layers)
    logger.info("full-attention layers %s, using %s, rotary %s", every_layer, layers, rotary)

    indexer = IndexerConfig()
    vocab = text_config.vocab_size
    gen = torch.Generator().manual_seed(1105)
    ids = torch.randint(0, vocab, (1, args.tokens), generator=gen)
    logger.info("document: %d random tokens (vocab %d)", args.tokens, vocab)

    captured = capture_layer_inputs(text_model, layers, ids)
    logger.info("captured layer inputs for %d layers", len(captured))

    first = min(2 * indexer.window_size, args.tokens - 2)
    positions = torch.linspace(first, args.tokens - 2, args.queries).long().unique()
    logger.info("query positions %s", positions.tolist())

    width = indexer.top_k + indexer.window_size + indexer.sink_size
    fields: dict[str, np.ndarray] = {
        "positions": positions.numpy().astype(np.int64),
        "meta.window_size": np.array(indexer.window_size),
        "meta.sink_size": np.array(indexer.sink_size),
        "meta.top_k": np.array(indexer.top_k),
        "meta.top_p": np.array(indexer.top_p),
        "meta.pool_size": np.array(indexer.pool_size),
        "meta.bucket_size": np.array(indexer.bucket_size),
        "meta.bucket_cap": np.array(indexer.bucket_cap),
        "meta.index_dim": np.array(indexer.index_dim),
    }
    for index in layers:
        hidden, position_embeddings = captured[index]
        reader = model_adapter.attention_class(text_model.layers[index].self_attn, indexer, rotary)
        with torch.no_grad():
            view = reader.teacher_view(hidden, positions, position_embeddings)
            buckets = torch_reference.build_buckets(
                view.index_keys,
                indexer.pool_size,
                indexer.bucket_size,
                indexer.bucket_cap,
                indexer.window_size,
            )
            selected, lengths = torch_reference.read(
                view.index_queries,
                positions,
                buckets,
                indexer.top_p,
                indexer.top_k,
                indexer.sink_size,
                indexer.window_size,
            )
        selected_np = selected.numpy().astype(np.int32)
        lengths_np = lengths.numpy().astype(np.int64)
        fields[f"layer{index}.index_queries"] = view.index_queries.numpy().astype(np.float32)
        fields[f"layer{index}.index_keys"] = view.index_keys.numpy().astype(np.float32)
        fields[f"layer{index}.selected"] = selected_np
        fields[f"layer{index}.lengths"] = lengths_np
        used = int(sum(int(x) for x in lengths_np))
        logger.info(
            "layer %d: keys %d, queries %d, selected total %d (mean %.1f of width %d)",
            index,
            view.index_keys.shape[0],
            view.index_queries.shape[0],
            used,
            used / max(len(lengths_np), 1),
            width,
        )
        del captured[index]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **fields)
    size = out.stat().st_size / 2**20
    logger.info("wrote %s (%.1f MiB), %d arrays", out, size, len(fields))


if __name__ == "__main__":
    main()
