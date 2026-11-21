#!/usr/bin/env bash
# RULER VT/FWE pre-submission sweep (gate checklist S1) on the M5 Max serving path.
# start.sh hardcodes port 11434 with no --port passthrough, so the dedicated
# LSA tier instance launches fusion-mlx serve directly on :11555 and the
# script cleans it up on exit.
set -euo pipefail

PORT=11555
BASE_URL="http://localhost:${PORT}/v1"
MODEL_PATH="${HOME}/fusion/fusion-lsa/models/qwen38-27b"   # bf16 dir (symlink target of the old ~/.fusion-mlx alias)
MODEL=qwen38-27b                                           # advertised name = dir basename; bench default matches
MAP="${HOME}/fusion/fusion-lsa/models/maps/qwen38_27b__long-16k-1M-heavy-key-and-query-d2048.safetensors"
OUT_DIR="${HOME}/.fusion-mlx/logs"
STAMP=$(date +%Y%m%d-%H%M%S)
SERVE_LOG="${OUT_DIR}/ruler-vtfwe-serve-${STAMP}.log"
FUSION_MLX_BIN="/Users/dahai/fusion/.venv/bin/fusion-mlx"

cleanup() {
    if [[ -n "${SERVE_PID:-}" ]] && kill -0 "${SERVE_PID}" 2>/dev/null; then
        kill "${SERVE_PID}" && echo "tier server (pid ${SERVE_PID}) stopped"
    fi
}
trap cleanup EXIT

# 1. dedicated LSA tier server (config = p3 coexistence doc / run-of-record path)
"${FUSION_MLX_BIN}" serve --host 127.0.0.1 --port "${PORT}" --text-only \
    --enable-lsa --lsa-map "${MAP}" \
    --kv-cache-turboquant k8v4 \
    --spec-decode none \
    "${MODEL_PATH}" > "${SERVE_LOG}" 2>&1 &
SERVE_PID=$!
echo "tier server launching on :${PORT} (pid ${SERVE_PID}), log ${SERVE_LOG}"
for i in $(seq 1 120); do
    if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then break; fi
    if ! kill -0 "${SERVE_PID}" 2>/dev/null; then
        echo "server died during boot; last log lines:"; tail -20 "${SERVE_LOG}"; exit 1
    fi
    sleep 5
done
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
