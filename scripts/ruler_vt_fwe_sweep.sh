#!/usr/bin/env bash
# RULER VT/FWE pre-submission sweep (gate checklist S1) on the M5 Max serving path.
# Run when the shared fusion-mlx server is stopped: the bf16 27B tier instance
# needs ~80 GB and cannot co-exist with a busy main server (62.5 GB loaded).
set -euo pipefail

PORT=11555
BASE_URL="http://localhost:${PORT}/v1"
MODEL=qwen38-27b
MAP="${HOME}/fusion/fusion-lsa/models/maps/qwen38_27b__long-16k-1M-heavy-key-and-query-d2048.safetensors"
OUT_DIR="${HOME}/.fusion-mlx/logs"
STAMP=$(date +%Y%m%d-%H%M%S)

# 1. dedicated LSA tier server (config = p3 coexistence doc / run-of-record path)
~/fusion/fusion-mlx/start.sh start --text-only \
    --enable-lsa --lsa-map "${MAP}" \
    --kv-cache-turboquant k8v4 \
    --no-memory-aware-cache --spec-decode none --port "${PORT}"
until curl -sf "http://localhost:${PORT}/health" >/dev/null; do sleep 5; done
echo "tier server ready on :${PORT}"

# 2. sweeps: {niah control, vt, fwe} x {32k, 128k} x seeds 41-43, 512-token budget
# one seed per invocation: distinct seeds bypass the server response cache
for TASK in niah vt fwe; do
    for LEN in 32000 131072; do
        for SEED in 41 42 43; do
            python3 scripts/ruler32k_bench.py \
                --base-url "${BASE_URL}" --model "${MODEL}" \
                --task "${TASK}" --length "${LEN}" --repeats 1 \
                --seed "${SEED}" \
                --max-tokens 512 \
                --api-key fg-admin-key \
                2>&1 | tee -a "${OUT_DIR}/ruler-${TASK}-${LEN}-${STAMP}.log"
        done
    done
done

echo "sweep complete: ${OUT_DIR}/ruler-*-${STAMP}.log"
