import logging
import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from fusion_lsa.budget import EarlyBudget
from fusion_lsa.core.reference import build_buckets, read

logger = logging.getLogger(__name__)

VECTORS = Path(os.environ.get("LSA_VECTORS", "models/nlsa-checkpoint/vectors.npz"))


def _meta(data: dict, name: str) -> int:
    return int(data[f"meta.{name}"])


@pytest.fixture(scope="module")
def vectors():
    if not VECTORS.exists():
        pytest.skip("real-weight vectors not extracted - run scripts/extract_nlsa_vectors.py")
    with np.load(VECTORS) as data:
        yield {key: data[key] for key in data.files}


def test_real_weights_selection_parity(vectors):
    data = vectors
    pool_size = _meta(data, "pool_size")
    bucket_size = _meta(data, "bucket_size")
    bucket_cap = _meta(data, "bucket_cap")
    window_size = _meta(data, "window_size")
    sink_size = _meta(data, "sink_size")
    top_k = _meta(data, "top_k")
    top_p = _meta(data, "top_p")
    positions = mx.array(data["positions"])
    layer_keys = sorted(key.split(".")[0] for key in data if key.startswith("layer") and key.endswith(".index_keys"))
    assert layer_keys, "the npz holds no layer fields"
    mismatched_rows = 0
    total_rows = 0
    for prefix in layer_keys:
        keys = mx.array(data[f"{prefix}.index_keys"])
        queries = mx.array(data[f"{prefix}.index_queries"])
        torch_selected = data[f"{prefix}.selected"]
        torch_lengths = data[f"{prefix}.lengths"]
        buckets = build_buckets(keys, pool_size, bucket_size, bucket_cap, window_size)
        selected, lengths = read(queries, positions, buckets, top_p, top_k, sink_size, window_size)
        got = np.array(selected)
        want = torch_selected
        for row in range(got.shape[0]):
            total_rows += 1
            got_set = set(int(x) for x in got[row] if x >= 0)
            want_set = set(int(x) for x in want[row][: torch_lengths[row]] if x >= 0)
            if got_set != want_set:
                mismatched_rows += 1
                overlap = len(got_set & want_set) / max(len(want_set), 1)
                logger.warning(
                    "%s row %d: %d/%d keys shared, %d vs %d selected",
                    prefix,
                    row,
                    len(got_set & want_set),
                    len(want_set),
                    len(got_set),
                    len(want_set),
                )
        logger.info("%s: %d rows checked, lengths mean %.1f", prefix, got.shape[0], float(np.mean(torch_lengths)))
    assert mismatched_rows == 0, f"{mismatched_rows}/{total_rows} rows differ from the torch reference"


def test_real_weights_budget_respected(vectors):
    data = vectors
    window_size = _meta(data, "window_size")
    sink_size = _meta(data, "sink_size")
    top_k = _meta(data, "top_k")
    budget = EarlyBudget(position=4 * window_size, top_k=top_k // 2, top_p=64)
    positions = mx.array(data["positions"])
    layer_keys = sorted(key.split(".")[0] for key in data if key.startswith("layer") and key.endswith(".index_keys"))
    keys = mx.array(data[f"{layer_keys[0]}.index_keys"])
    queries = mx.array(data[f"{layer_keys[0]}.index_queries"])
    buckets = build_buckets(
        keys,
        _meta(data, "pool_size"),
        _meta(data, "bucket_size"),
        _meta(data, "bucket_cap"),
        window_size,
    )
    selected, lengths = read(
        queries,
        positions,
        buckets,
        _meta(data, "top_p"),
        top_k,
        sink_size,
        window_size,
        early_budget=budget,
    )
    got_lengths = np.array(lengths)
    # Early rows spend the reduced budget, late rows spend the full one
    for row, position in enumerate(data["positions"].tolist()):
        if position < budget.position:
            assert got_lengths[row] <= budget.top_k + window_size + sink_size
        else:
            assert got_lengths[row] <= top_k + window_size + sink_size
    logger.info("early-budget lengths: %s", got_lengths.tolist())


QWEN_MAP = Path("models/maps/qwen38_27b__long-16k-1M-heavy-key-and-query-d2048.safetensors")


def test_qwen38_27b_map_projects_real_vectors(vectors):
    from fusion_lsa.convert.map_converter import LsaMap

    if not QWEN_MAP.exists():
        pytest.skip("converted Qwen3.8-27B map not present")
    data = vectors
    prefixes = sorted(key.split(".")[0] for key in data if key.startswith("layer") and key.endswith(".index_keys"))
    loaded = LsaMap(str(QWEN_MAP), model_arch="qwen38_27b", num_layers=64, latent_dim=768)
    keys = mx.array(data[f"{prefixes[0]}.index_keys"])
    queries = mx.array(data[f"{prefixes[0]}.index_queries"])
    assert keys.shape[-1] == 768, f"index space is {keys.shape[-1]} dims, the map reads 768"
    q_idx, k_idx = loaded.project(int(prefixes[0][5:]), queries, keys)
    assert q_idx.shape[-1] == loaded.index_dim and k_idx.shape[-1] == loaded.index_dim
    from fusion_lsa.core.reference import check_unit_rows

    check_unit_rows("q", q_idx.reshape(-1, loaded.index_dim))
    check_unit_rows("k", k_idx.reshape(-1, loaded.index_dim))
    logger.info("map projection ok: %s -> %s", keys.shape, k_idx.shape)
