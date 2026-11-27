import ast
import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

DEFAULT_PATCHED_SRC = (
    "~/fusion/vllm-ascend-pr/container-v032/"
    "artifacts-20261126/final/kimi_k3_patched.py"
)
PATCHED_SRC_ENV = "KIMI_K3_PATCHED_SRC"
REQUIRED_FUNCS = ("_torch_seq_kda", "_torch_chunk_kda", "_kda_post_gate")


def patched_src_path():
    return Path(os.path.expanduser(os.environ.get(PATCHED_SRC_ENV, DEFAULT_PATCHED_SRC)))


def extract_patched_kda(src_path=None):
    # Pull the deployed fp32 KDA bypass implementations out of
    # kimi_k3_patched.py by AST source extraction and exec them into a
    # torch-only namespace, so validation runs the exact code that serves on
    # the 950DT node (not a re-implementation that could drift).
    path = Path(src_path) if src_path else patched_src_path()
    if not path.is_file():
        raise FileNotFoundError(f"patched model source not found: {path}")
    src = path.read_text()
    tree = ast.parse(src)
    ns = {"torch": torch, "__name__": "kimi_k3_patched.extracted"}
    found = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in REQUIRED_FUNCS:
            # Intentional exec: run the exact deployed source extracted by AST
            # (see docstring); a re-implementation could drift from serving.
            exec(ast.get_source_segment(src, node), ns)  # noqa: S102
            found.append(node.name)
    missing = [name for name in REQUIRED_FUNCS if name not in ns]
    if missing:
        raise RuntimeError(f"functions missing from patched source {path}: {missing}")
    logger.info("extracted KDA bypasses from %s: %s", path, ", ".join(found))
    return ns


def fla_naive_recurrent_kda(q, k, v, g, beta, scale=None, initial_state=None):
    # Vendored from fla (flash-linear-attention) naive_recurrent_kda, MIT
    # license, (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li — see
    # probes/kda_naive.py in the vllm-ascend-pr workspace. GVA group expansion
    # is dropped: the Kimi KDA bypass path always runs H == HV.
    dtype = v.dtype
    if scale is None:
        scale = q.shape[-1] ** -0.5
    q = q.to(torch.float) * scale
    k = k.to(torch.float)
    v = v.to(torch.float)
    g = g.to(torch.float)
    beta = beta.to(torch.float)
    B, T, H, K = q.shape
    V = v.shape[-1]
    s = q.new_zeros(B, H, K, V)
    if initial_state is not None:
        s = s + initial_state.to(torch.float)
    o = torch.zeros_like(v)
    for i in range(T):
        q_i, k_i, v_i = q[:, i], k[:, i], v[:, i]
        s = s * g[:, i][..., None].exp()
        s = s + torch.einsum(
            "b h k, b h v -> b h k v",
            beta[:, i][..., None] * k_i,
            v_i - (k_i[..., None] * s).sum(-2),
        )
        o[:, i] = torch.einsum("b h k, b h k v -> b h v", q_i, s)
    return o.to(dtype), s


def _chunk_rearrange(x, bt):
    # einops 'b (n c) h ... -> b h n c ...' in pure torch.
    b, t = x.shape[0], x.shape[1]
    tail = list(x.shape[2:])
    n = t // bt
    x = x.reshape(b, n, bt, *tail)
    perm = [0, 3, 1, 2] + list(range(4, 4 + len(tail) - 1))
    return x.permute(*perm).contiguous()


def fla_naive_chunk_kda(q, k, v, g, beta, scale=None, initial_state=None, chunk_size=64):
    # Vendored from fla naive_chunk_kda (same provenance as above); GVA
    # expansion dropped (H == HV), einops replaced with plain torch ops.
    # Requires T % chunk_size == 0, as upstream does.
    dtype = v.dtype
    bt = chunk_size
    t = q.shape[1]
    assert t % bt == 0, f"fla chunk oracle requires T % {bt} == 0, got T={t}"
    if scale is None:
        scale = q.shape[-1] ** -0.5
    q = _chunk_rearrange(q, bt).to(torch.float) * scale
    k = _chunk_rearrange(k, bt).to(torch.float)
    v = _chunk_rearrange(v, bt).to(torch.float)
    g = _chunk_rearrange(g, bt).to(torch.float).cumsum(-2)
    beta = _chunk_rearrange(beta, bt).to(torch.float)
    b, h, n = q.shape[0], q.shape[1], q.shape[2]
    kk, vv = q.shape[-1], v.shape[-1]

    mask_diag = torch.triu(torch.ones(bt, bt, dtype=torch.bool, device=q.device), diagonal=0)
    a = torch.zeros(b, h, n, bt, bt, dtype=torch.float, device=q.device)
    for i in range(bt):
        k_i = k[..., i, :]
        g_i = g[..., i : i + 1, :]
        a[..., i] = torch.einsum("... c d, ... d -> ... c", k * (g - g_i).exp(), k_i)
    a = a * beta[..., None]
    a = -a.masked_fill(mask_diag, 0)
    for i in range(1, bt):
        a[..., i, :i] = a[..., i, :i].clone() + (
            a[..., i, :, None].clone() * a[..., :, :i].clone()
        ).sum(-2)
    a = (a + torch.eye(bt, dtype=torch.float, device=q.device)) * beta[..., None, :]
    w = a @ (g.exp() * k)
    u = a @ v

    s = q.new_zeros(b, h, kk, vv)
    if initial_state is not None:
        s = s + initial_state.to(torch.float)
    o = torch.zeros_like(v)
    mask_off = torch.triu(torch.ones(bt, bt, dtype=torch.bool, device=q.device), diagonal=1)
    for i in range(n):
        q_i, k_i, u_i = q[:, :, i], k[:, :, i], u[:, :, i]
        g_i, w_i = g[:, :, i], w[:, :, i]
        aqk = torch.zeros(b, h, bt, bt, dtype=torch.float, device=q.device)
        for j in range(bt):
            k_j = k[:, :, i, j]
            g_j = g[:, :, i, j : j + 1, :]
            aqk[..., j] = torch.einsum("... c d, ... d -> ... c", q_i * (g_i - g_j).exp(), k_j)
        aqk = aqk.masked_fill(mask_off, 0)
        v_i = u_i - w_i @ s
        o[:, :, i] = (q_i * g_i.exp()) @ s + aqk @ v_i
        s = s * g_i[..., -1].exp().unsqueeze(-1)
        s = s + ((g_i[..., -1:] - g_i).exp() * k_i).transpose(-1, -2) @ v_i
    o = o.permute(0, 2, 3, 1, 4).reshape(b, n * bt, h, vv)
    return o.to(dtype), s


def make_inputs(
    t,
    h=8,
    kk=128,
    vv=128,
    seed=7,
    dtype=torch.float32,
    cu_seqlens=None,
    lower_bound=None,
    a_log=False,
    scale=0.5,
):
    # Synthetic KDA layer inputs in the patched-file calling convention:
    # q/k [1,T,H,K], v [1,T,H,V], raw gate [1,T,H,K], beta [1,T,H],
    # initial state [n_seq,H,V,K] (cache layout; the bypasses transpose it).
    gen = torch.Generator().manual_seed(seed)
    q = (torch.randn(1, t, h, kk, generator=gen) * scale).to(dtype)
    k = (torch.randn(1, t, h, kk, generator=gen) * scale).to(dtype)
    v = (torch.randn(1, t, h, vv, generator=gen) * scale).to(dtype)
    g = (torch.randn(1, t, h, kk, generator=gen) * 0.1).to(dtype)
    beta = (torch.rand(1, t, h, generator=gen) * 1.4 + 0.3).to(dtype)
    cu = cu_seqlens if cu_seqlens is not None else [0, t]
    n_seq = len(cu) - 1
    init = (torch.randn(n_seq, h, vv, kk, generator=gen) * 0.1).to(dtype)
    al = (torch.randn(h, generator=gen) * 0.1).to(dtype) if a_log else None
    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "init": init,
        "cu": cu,
        "a_log": al,
        "dt_bias": None,
        "lower_bound": lower_bound,
    }


def dev_stats(ref, got, eps=1e-9):
    # Elementwise deviation of `got` against `ref`, plus a per-token series
    # when the tensor carries a token axis ([1, T, H, D] output layout).
    ref = ref.float()
    got = got.float()
    diff = (ref - got).abs()
    stats = {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "max_rel": (diff / (ref.abs() + eps)).max().item(),
        "per_token_abs": None,
        "per_token_rel": None,
    }
    if ref.dim() == 4 and ref.shape[0] == 1:
        stats["per_token_abs"] = diff.amax(dim=(2, 3))[0]
        stats["per_token_rel"] = (diff / (ref.abs() + eps)).amax(dim=(2, 3))[0]
    return stats


def run_bypasses(ns, inp):
    # Run the deployed sequential reference and chunked bypass on one input
    # bundle; returns outputs normalized to oracle layout (state [n_seq,H,K,V]).
    o_seq, s_seq = ns["_torch_seq_kda"](
        inp["q"],
        inp["k"],
        inp["v"],
        inp["g"],
        inp["beta"],
        inp["init"],
        inp["cu"],
        inp["a_log"],
        inp["dt_bias"],
        inp["lower_bound"],
    )
    o_chk, s_chk = ns["_torch_chunk_kda"](
        inp["q"],
        inp["k"],
        inp["v"],
        inp["g"],
        inp["beta"],
        inp["init"],
        inp["cu"],
        inp["a_log"],
        inp["dt_bias"],
        inp["lower_bound"],
    )
    return (o_seq, s_seq.transpose(-1, -2)), (o_chk, s_chk.transpose(-1, -2))


def run_case(ns, inp):
    # Full comparison matrix for one input bundle: deployed chunk bypass vs
    # deployed sequential reference, and both against the vendored fla
    # naive oracles (chunk oracle only when every segment is chunk-aligned).
    (o_seq, s_seq), (o_chk, s_chk) = run_bypasses(ns, inp)
    gate = ns["_kda_post_gate"](inp["g"], inp["q"].shape[2], inp["a_log"], inp["dt_bias"], inp["lower_bound"])
    init_kvk = inp["init"].transpose(-1, -2).contiguous()
    g_exp = gate.expand(inp["q"].shape)
    o_seq_or, s_seq_or = fla_naive_recurrent_kda(
        inp["q"], inp["k"], inp["v"], g_exp, inp["beta"], initial_state=init_kvk
    )
    res = {
        "chunk_vs_seq": {"out": dev_stats(o_seq, o_chk), "state": dev_stats(s_seq, s_chk)},
        "seq_vs_oracle": {"out": dev_stats(o_seq_or, o_seq), "state": dev_stats(s_seq_or, s_seq)},
    }
    cu = inp["cu"]
    aligned = all((cu[i + 1] - cu[i]) % 64 == 0 for i in range(len(cu) - 1))
    if aligned:
        o_chk_or, s_chk_or = fla_naive_chunk_kda(
            inp["q"], inp["k"], inp["v"], g_exp, inp["beta"], initial_state=init_kvk
        )
        res["chunk_vs_oracle"] = {
            "out": dev_stats(o_chk_or, o_chk),
            "state": dev_stats(s_chk_or, s_chk),
        }
    return res


def t_sweep(t_list, ns, seed=7, lower_bound=None, dtype=torch.float32):
    # Chunk-vs-sequential deviation as prefill length grows: the premise to
    # validate is that the fp32 chunked bypass stays at rounding level (no
    # structural divergence) for all T, including far beyond the 256 cap.
    rows = []
    for t in t_list:
        inp = make_inputs(t, seed=seed, lower_bound=lower_bound, dtype=dtype)
        res = run_case(ns, inp)["chunk_vs_seq"]
        rows.append(
            {
                "T": t,
                "out_max_abs": res["out"]["max_abs"],
                "out_max_rel": res["out"]["max_rel"],
                "state_max_abs": res["state"]["max_abs"],
                "state_max_rel": res["state"]["max_rel"],
            }
        )
        logger.info(
            "T=%d out_abs=%.3e out_rel=%.3e state_abs=%.3e state_rel=%.3e",
            t,
            rows[-1]["out_max_abs"],
            rows[-1]["out_max_rel"],
            rows[-1]["state_max_abs"],
            rows[-1]["state_max_rel"],
        )
    return rows


def compare_candidate_output(ref_out, cand_out, tag="candidate"):
    # Quantify a captured NPU kernel output (e.g. fla_npu chunk_kda per-token
    # path) against the fp32 sequential reference, per token. This is the
    # half of the comparison that can only run with 950DT hardware.
    stats = dev_stats(ref_out, cand_out)
    logger.warning(
        "%s vs fp32 sequential reference: max_abs=%.3e max_rel=%.3e",
        tag,
        stats["max_abs"],
        stats["max_rel"],
    )
    if stats["per_token_rel"] is not None:
        worst = int(stats["per_token_rel"].argmax().item())
        logger.warning(
            "%s worst token t=%d rel=%.3e abs=%.3e",
            tag,
            worst,
            stats["per_token_rel"][worst].item(),
            stats["per_token_abs"][worst].item(),
        )
    return stats


def serving_slice_chain(ns, t, slice_len, seed=7, h=8, kk=128, vv=128):
    # Simulate the serving flow end-to-end at the KDA layer: the model runner
    # chunks a long prefill into scheduler slices (observed T=2048 in
    # KTORCHCHUNK logs), each slice runs _torch_chunk_kda with the previous
    # slice's final recurrent state (cache layout [n_seq,H,V,K]) and writes
    # the returned final state back. Reference: one _torch_seq_kda call over
    # the whole T. Any structural divergence here reproduces the RULER cliff
    # between 1 and 2 scheduler chunks without needing NPU hardware.
    inp = make_inputs(t, seed=seed, h=h, kk=kk, vv=vv)
    args = (inp["q"], inp["k"], inp["v"], inp["g"], inp["beta"])
    o_ref, s_ref = ns["_torch_seq_kda"](*args, inp["init"], [0, t], inp["a_log"], inp["dt_bias"], inp["lower_bound"])
    state = inp["init"].clone()
    outs = []
    for st in range(0, t, slice_len):
        en = min(st + slice_len, t)
        o_i, s_i = ns["_torch_chunk_kda"](
            args[0][:, st:en], args[1][:, st:en], args[2][:, st:en], args[3][:, st:en], args[4][:, st:en],
            state, [0, en - st], inp["a_log"], inp["dt_bias"], inp["lower_bound"],
        )
        outs.append(o_i)
        state = s_i
    o_sliced = torch.cat(outs, dim=1)
    return {
        "t": t,
        "slices": (t + slice_len - 1) // slice_len,
        "out": dev_stats(o_ref, o_sliced),
        # Both raw bypass states use the cache layout [n_seq, H, V, K]; the
        # chunked bypass transposes internally, so no transpose here.
        "state": dev_stats(s_ref, state),
    }
