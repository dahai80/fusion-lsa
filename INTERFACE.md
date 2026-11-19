# fusion-lsa interface contract

fusion-lsa is the **single source of truth (SSOT)** for LSA index semantics:
the ward merge, the incremental bucket index, and the index space
(`src/fusion_lsa/core/reference.py`, `src/fusion_lsa/index/`). Downstream
implementations are derived ports and must not diverge.

## Parity chain

1. `Prefrontal-AI` reference implementation (upstream research code).
2. fusion-lsa parity tests (`tests/test_incremental_index.py` — incremental
   vs. batched reference, per-device pinning).
3. hsa cross-backend parity (`hsa/tests/test_index_ward.py` — the derived
   torch port in `hsa/src/hsa/index/` is cross-checked against these
   semantics).

## Change protocol

- **Semantic change** — anything that can alter selection results: retrieval
  selection, tie-break rules, float accumulation order, bucket boundaries.
  Requires, before merge:
  1. rerun of the 5 fusion-lsa parity tests (`tests/test_incremental_index.py`);
  2. rerun of the hsa cross-backend parity suite (`hsa/tests/test_index_ward.py`);
  3. commit message prefix `semantics:`.
- **Resource-only change** — storage placement, execution stream, caching,
  memory strategy. Exempt from parity reruns, but must carry its own
  regression test (precedent: the host-storage block fix 2dc3698 and its
  resource-flat test in `tests/test_incremental_index.py`).

If a change is both semantic and resource-only in part, split it or treat it
as semantic.

## Consumers

| consumer | integration | notes |
|---|---|---|
| fusion-mlx | serving via `fusion_mlx.patches.qwen35_lsa` | runs record the implementation version in the serve log banner |
| hsa | derived torch index (`hsa/src/hsa/index/`) + paper numbers | pins runs to fusion-lsa commits; see `hsa/release/run-logs.md` |

## Version stamp

`fusion_lsa.__version__` is the release version; `fusion_lsa.FUSION_LSA_COMMIT`
is the best-effort git commit resolved at import (env `FUSION_LSA_COMMIT`
overrides; `unknown-commit` outside a repo). Both are logged once at import so
every serve log identifies the implementation a run was produced with.
