import logging
import re

import mlx.core as mx
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

logger = logging.getLogger(__name__)

# Upstream map tensor name: "<layer>.W_Q_idx" / "<layer>.W_K_idx"
LAYER_PATTERN = re.compile(r"^(\d+)\.(W_Q_idx|W_K_idx)$")
SUFFIX_TO_ROLE = {"W_Q_idx": "query_projection", "W_K_idx": "key_projection"}
# Metadata written into every converted safetensors file; the loader fails loud on mismatch
METADATA_KEYS = ("model_arch", "indexer_layers", "latent_dim", "index_dim", "source")


def read_map_pt(path: str) -> tuple[dict[int, dict[str, np.ndarray]], dict]:
    import torch

    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{path} is not a non-empty dict of layer tensors")
    layers: dict[int, dict[str, np.ndarray]] = {}
    meta_source = raw.get("__metadata__", None)
    for key, value in raw.items():
        match = LAYER_PATTERN.match(key)
        if not match:
            raise ValueError(f"{path} holds unexpected tensor {key!r} - refusing a partial map")
        layer, suffix = int(match.group(1)), match.group(2)
        tensor = value.detach().cpu().float().numpy()
        if tensor.ndim != 2:
            raise ValueError(f"{key} must be a matrix - got shape {tensor.shape}")
        if not np.isfinite(tensor).all():
            raise ValueError(f"{key} holds a non-finite value")
        layers.setdefault(layer, {})[SUFFIX_TO_ROLE[suffix]] = tensor
    for layer, roles in layers.items():
        if set(roles) != {"query_projection", "key_projection"}:
            raise ValueError(f"layer {layer} lacks one of W_Q_idx / W_K_idx")
        shapes = {t.shape for t in roles.values()}
        if len(shapes) != 1:
            raise ValueError(f"layer {layer} projections disagree on shape: {shapes}")
    shape = next(iter(layers.values()))["query_projection"].shape
    meta = {
        "latent_dim": shape[0],
        "index_dim": shape[1],
        "indexer_layers": sorted(layers),
    }
    if meta_source:
        meta["source"] = str(meta_source)
    logger.info("read map %s: %d layers, latent %d -> index %d", path, len(layers), shape[0], shape[1])
    return layers, meta


def convert_map_pt(path: str, out_path: str, model_arch: str) -> dict:
    layers, meta = read_map_pt(path)
    tensors = {}
    for layer, roles in layers.items():
        tensors[f"layer{layer}.query_projection"] = roles["query_projection"]
        tensors[f"layer{layer}.key_projection"] = roles["key_projection"]
    metadata = {
        "model_arch": model_arch,
        "indexer_layers": ",".join(str(l) for l in meta["indexer_layers"]),
        "latent_dim": str(meta["latent_dim"]),
        "index_dim": str(meta["index_dim"]),
        "source": meta.get("source", path),
    }
    save_file(tensors, out_path, metadata=metadata)
    logger.info("wrote %s (%d layer pairs), arch %s", out_path, len(layers), model_arch)
    return metadata


class LsaMap:
    # One converted map: per-layer MLX projections plus validated metadata
    def __init__(self, path: str, model_arch: str, num_layers: int, latent_dim: int):
        self.projections: dict[int, dict[str, mx.array]] = {}
        with safe_open(path, framework="numpy") as f:
            metadata = f.metadata() or {}
            _check_field(metadata, "model_arch", model_arch, path)
            _check_field(metadata, "latent_dim", str(latent_dim), path)
            stored_layers = [int(x) for x in metadata.get("indexer_layers", "").split(",") if x]
            for layer in stored_layers:
                for role in ("query_projection", "key_projection"):
                    name = f"layer{layer}.{role}"
                    if name not in f.keys():
                        raise ValueError(f"{path} lacks tensor {name} named in its metadata")
                    arr = f.get_tensor(name).astype(np.float32)
                    self.projections.setdefault(layer, {})[role] = mx.array(arr)
        shape = self.projections[stored_layers[0]]["query_projection"].shape
        _check_field(metadata, "index_dim", str(shape[1]), path)
        for layer, roles in self.projections.items():
            for role, arr in roles.items():
                if arr.shape[0] != latent_dim:
                    raise ValueError(
                        f"{path} layer{layer}.{role} reads latent {arr.shape[0]}, model has {latent_dim}"
                    )
        self.model_arch = model_arch
        self.index_dim = shape[1]
        self.layers = sorted(self.projections)
        if num_layers > 0 and self.layers[-1] >= num_layers:
            raise ValueError(
                f"{path} names indexer layer {self.layers[-1]} but the model has {num_layers} layers"
            )
        logger.info(
            "loaded lsa map %s: arch %s, %d layers, latent %d -> index %d",
            path,
            model_arch,
            len(self.layers),
            latent_dim,
            self.index_dim,
        )

    def project(self, layer: int, queries: mx.array, keys: mx.array) -> tuple[mx.array, mx.array]:
        # queries [..., latent_dim] unit rows, keys [..., latent_dim] unit rows -> index space, renormalized
        roles = self.projections[layer]
        q = queries @ roles["query_projection"]
        k = keys @ roles["key_projection"]
        q = q / mx.maximum(mx.linalg.norm(q, axis=-1, keepdims=True), 1e-30)
        k = k / mx.maximum(mx.linalg.norm(k, axis=-1, keepdims=True), 1e-30)
        return q, k


def _check_field(metadata: dict, field: str, expected: str, path: str) -> None:
    got = metadata.get(field)
    if got is None:
        raise ValueError(f"{path} lacks required metadata {field} - it was not produced by convert_map_pt")
    if str(got) != str(expected):
        raise ValueError(f"{path} metadata {field} is {got!r}, the model needs {expected!r} - refusing")
