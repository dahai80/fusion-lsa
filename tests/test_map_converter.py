import logging
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
import torch
from safetensors.numpy import save_file

from fusion_lsa.convert.map_converter import LsaMap, convert_map_pt, read_map_pt

logger = logging.getLogger(__name__)


@pytest.fixture()
def tiny_map_pt(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.pt"
    torch.save(
        {f"{l}.{s}": torch.randn(16, 8) for l in (2, 6) for s in ("W_Q_idx", "W_K_idx")},
        path,
    )
    return path


def test_read_map_pt_parses_layers(tiny_map_pt):
    layers, meta = read_map_pt(str(tiny_map_pt))
    assert sorted(layers) == [2, 6]
    assert meta["latent_dim"] == 16 and meta["index_dim"] == 8
    assert set(layers[2]) == {"query_projection", "key_projection"}


def test_read_map_pt_rejects_bad_shapes(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"2.W_Q_idx": torch.randn(16, 8), "2.W_K_idx": torch.randn(16, 4)}, path)
    with pytest.raises(ValueError, match="disagree"):
        read_map_pt(str(path))
    path2 = tmp_path / "stray.pt"
    torch.save({"2.W_Q_idx": torch.randn(16, 8), "2.W_K_idx": torch.randn(16, 8), "bias": torch.randn(8)}, path2)
    with pytest.raises(ValueError, match="unexpected"):
        read_map_pt(str(path2))


def test_convert_roundtrip_and_validation(tmp_path, tiny_map_pt):
    out = tmp_path / "tiny.safetensors"
    meta = convert_map_pt(str(tiny_map_pt), str(out), model_arch="tiny-test-1b")
    assert meta["latent_dim"] == "16"
    loaded = LsaMap(str(out), model_arch="tiny-test-1b", num_layers=8, latent_dim=16)
    assert loaded.layers == [2, 6]
    assert loaded.index_dim == 8
    with pytest.raises(ValueError, match="refusing"):
        LsaMap(str(out), model_arch="other-arch", num_layers=8, latent_dim=16)
    with pytest.raises(ValueError, match="refusing"):
        LsaMap(str(out), model_arch="tiny-test-1b", num_layers=8, latent_dim=32)
    with pytest.raises(ValueError, match="layers"):
        LsaMap(str(out), model_arch="tiny-test-1b", num_layers=4, latent_dim=16)


def test_project_matches_reference_semantics(tmp_path, tiny_map_pt):
    # project() = rownorm(rows @ W), matching upstream index_space.project
    from fusion_lsa.core.reference import check_unit_rows

    out = tmp_path / "tiny.safetensors"
    convert_map_pt(str(tiny_map_pt), str(out), model_arch="tiny-test-1b")
    loaded = LsaMap(str(out), model_arch="tiny-test-1b", num_layers=8, latent_dim=16)
    q = mx.random.normal((2, 5, 16), key=mx.random.key(1))
    k = mx.random.normal((2, 7, 16), key=mx.random.key(2))
    q_unit = q / mx.linalg.norm(q, axis=-1, keepdims=True)
    k_unit = k / mx.linalg.norm(k, axis=-1, keepdims=True)
    q_idx, k_idx = loaded.project(2, q_unit, k_unit)
    check_unit_rows("q", q_idx.reshape(-1, 8))
    check_unit_rows("k", k_idx.reshape(-1, 8))
    w = loaded.projections[2]["query_projection"]
    want = q_unit @ w
    want = want / mx.maximum(mx.linalg.norm(want, axis=-1, keepdims=True), 1e-30)
    assert float(mx.abs(q_idx - want).max().item()) < 1e-5


@pytest.mark.skipif(
    not Path("models/maps/glm53_flash__long-16k-1M-heavy-d2048-lr3e-4-latent.pt").exists(),
    reason="released GLM map not downloaded",
)
def test_released_glm_map_smoke(tmp_path):
    src = "models/maps/glm53_flash__long-16k-1M-heavy-d2048-lr3e-4-latent.pt"
    out = tmp_path / "glm_lsa.safetensors"
    meta = convert_map_pt(src, str(out), model_arch="glm53_flash")
    assert meta["indexer_layers"] == "3,7,11,15,19,23,27,31,35,39,43"
    loaded = LsaMap(str(out), model_arch="glm53_flash", num_layers=44, latent_dim=512)
    q = mx.random.normal((1, 4, 512), key=mx.random.key(3))
    k = mx.random.normal((1, 9, 512), key=mx.random.key(4))
    q_idx, k_idx = loaded.project(3, q, k)
    assert q_idx.shape == (1, 4, 64) and k_idx.shape == (1, 9, 64)
