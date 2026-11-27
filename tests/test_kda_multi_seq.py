import logging

import pytest

torch = pytest.importorskip("torch")

from fusion_lsa.kda_validation import (
    dev_stats,
    extract_patched_kda,
    make_inputs,
    patched_src_path,
)

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def ns():
    # The bypasses are extracted from the deployed serving source, which
    # only exists on machines with the vllm-ascend artifacts checkout.
    if not patched_src_path().is_file():
        pytest.skip("deployed kimi_k3_patched.py artifact not available")
    return extract_patched_kda()


def _per_seq_reference(ns, inp, cu, inits):
    outs, states = [], []
    for si in range(len(cu) - 1):
        st, en = cu[si], cu[si + 1]
        sl = (slice(None), slice(st, en))
        o, s = ns["_torch_seq_kda"](
            inp["q"][sl],
            inp["k"][sl],
            inp["v"][sl],
            inp["g"][sl],
            inp["beta"][sl],
            inits[si : si + 1],
            [0, en - st],
            inp["a_log"],
            inp["dt_bias"],
            inp["lower_bound"],
        )
        outs.append(o)
        states.append(s)
    return torch.cat(outs, dim=1), torch.cat(states, 0)


def test_chunk_bypass_multi_seq_matches_sequential(ns):
    # Serving batches prefill slices from concurrent requests: cu_seqlens
    # carries n_seq>1 and the bypass must keep per-sequence states apart
    # (issue #26: identical answers across concurrent RULER requests).
    t1, t2 = 2048, 1024
    inp = make_inputs(t1 + t2, seed=11)
    inits = torch.stack([inp["init"][0], inp["init"][0] * 0.3 + 0.05])
    o_multi, s_multi = ns["_torch_chunk_kda"](
        inp["q"],
        inp["k"],
        inp["v"],
        inp["g"],
        inp["beta"],
        inits,
        [0, t1, t1 + t2],
        inp["a_log"],
        inp["dt_bias"],
        inp["lower_bound"],
    )
    o_ref, s_ref = _per_seq_reference(ns, inp, [0, t1, t1 + t2], inits)
    out_dev = dev_stats(o_ref, o_multi)["max_abs"]
    st_dev = dev_stats(s_ref, s_multi)["max_abs"]
    logger.warning("multi-seq chunk bypass: out_abs=%.3e state_abs=%.3e", out_dev, st_dev)
    assert out_dev < 1e-5
    assert st_dev < 1e-5


def test_chunk_bypass_mixed_decode_like_len1_seqs(ns):
    # Mixed prefill+decode batches (issue #28 fix) route decode rows through
    # the chunk bypass as T=1 sequences, positioned first in serving token
    # order. A T=1 chunk must equal the sequential recurrence step, with
    # committed initial states read and written back per sequence.
    lengths = [1, 1, 1024]
    cu = [0]
    for t in lengths:
        cu.append(cu[-1] + t)
    total = cu[-1]
    inp = make_inputs(total, seed=23)
    inits = torch.stack(
        [
            inp["init"][0],
            inp["init"][0] * 0.3 + 0.05,
            inp["init"][0] * (-0.7) - 0.02,
        ]
    )
    o_multi, s_multi = ns["_torch_chunk_kda"](
        inp["q"],
        inp["k"],
        inp["v"],
        inp["g"],
        inp["beta"],
        inits,
        cu,
        inp["a_log"],
        inp["dt_bias"],
        inp["lower_bound"],
    )
    o_ref, s_ref = _per_seq_reference(ns, inp, cu, inits)
    out_dev = dev_stats(o_ref, o_multi)["max_abs"]
    st_dev = dev_stats(s_ref, s_multi)["max_abs"]
    logger.warning("mixed len-1 chunk bypass: out_abs=%.3e state_abs=%.3e", out_dev, st_dev)
    assert out_dev < 1e-5
    assert st_dev < 1e-5
    # A T=1 sequence's single output token must match its recurrence step
    # exactly at the same tolerance, not just in aggregate.
    for si in (0, 1):
        single_dev = dev_stats(o_ref[:, cu[si] : cu[si + 1]], o_multi[:, cu[si] : cu[si + 1]])[
            "max_abs"
        ]
        assert single_dev < 1e-5


def test_serving_slice_chain_rounding_level(ns):
    # Chained scheduler slices (T=2048 each) must reproduce the full-length
    # sequential pass at fp32 rounding level, including the final state that
    # seeds the next slice.
    from fusion_lsa.kda_validation import serving_slice_chain

    for t in (4096, 8192):
        r = serving_slice_chain(ns, t, 2048)
        logger.warning(
            "slice chain T=%d slices=%d out_abs=%.3e state_abs=%.3e",
            r["t"],
            r["slices"],
            r["out"]["max_abs"],
            r["state"]["max_abs"],
        )
        assert r["out"]["max_abs"] < 1e-5
        assert r["state"]["max_abs"] < 1e-5
