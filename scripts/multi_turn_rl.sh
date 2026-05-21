#!/usr/bin/env bash
# Multi-turn RL: start CARE reward server → preprocess → run verl PPO
# (the inner script launches the shared Patient+Supervisor vLLM server itself).
#
# Configure model + output dir here (or override via env when invoking):
#   MODEL_PATH=/path/to/merged/hf/dir  OUTPUT_PATH=./checkpoints/my_run  EXP_NAME=my_exp  bash multi_turn_rl.sh

set -euo pipefail

# ---- knobs (override via env) ----------------------------------------------
MODEL_PATH="${MODEL_PATH:-/home/asbahk/EMNLP_FINAL/Rasingan/sft_training/results/Qwen3-4B-sft-respair-new-3-merged}"
EXP_NAME="${EXP_NAME:-}"         # empty → run_multiturn.sh picks therapist_multiturn_<ts>
OUTPUT_PATH="${OUTPUT_PATH:-}"   # empty → run_multiturn.sh picks ./checkpoints/<EXP_NAME>
SERVER_URL="http://127.0.0.1:8000/health"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RASINGAN_ROOT="$(dirname "$SCRIPT_DIR")"
SERVER_DIR="$RASINGAN_ROOT/server"
VERL_DIR="$RASINGAN_ROOT/verl"
PREPROCESS_SCRIPT="$VERL_DIR/examples/faith/data_preprocess/preprocess_multiturn.py"
RUN_MULTI_TURN_SCRIPT="$VERL_DIR/examples/faith/scripts/run_multiturn.sh"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-verl}"
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1090
source "$CONDA_BASE/etc/profile.d/conda.sh"
# A pre-activated Python venv on PATH would shadow conda activate and leak
# through `bash -lc` to child scripts (server.sh). Drop it first.
if [ -n "${VIRTUAL_ENV:-}" ] && command -v deactivate &>/dev/null; then
    deactivate 2>/dev/null || true
fi
unset VIRTUAL_ENV
conda activate "$CONDA_ENV_NAME"
CONDA_PY="$CONDA_PREFIX/bin/python"

SERVER_PGID=""
cleanup() {
    if [[ -n "$SERVER_PGID" ]] && kill -0 "-$SERVER_PGID" >/dev/null 2>&1; then
        echo "Stopping CARE server (pgid=$SERVER_PGID)…"
        kill -TERM "-$SERVER_PGID" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT INT TERM

wait_for_server() {
    for _ in $(seq 1 60); do
        curl -sSf "$SERVER_URL" >/dev/null 2>&1 && { echo "CARE server healthy: $SERVER_URL"; return 0; }
        sleep 2
    done
    echo "CARE server did not become healthy at $SERVER_URL" >&2
    return 1
}

echo "[1/3] Starting CARE reward server…"
if curl -sSf "$SERVER_URL" >/dev/null 2>&1; then
    echo "CARE server already running; reusing existing instance."
else
    RUN_TS="$(date '+%Y%m%d_%H%M%S')"
    SERVER_LOG_FILE="$SERVER_DIR/server_multi_turn_rl_${RUN_TS}.log"
    setsid bash -lc "cd '$SERVER_DIR' && bash server.sh prod" >"$SERVER_LOG_FILE" 2>&1 &
    SERVER_PGID=$!
    echo "CARE server logs: $SERVER_LOG_FILE"
    wait_for_server
fi

echo "[2/3] Building multi-turn dataset parquets from reddit splits…"
( cd "$VERL_DIR" && "$CONDA_PY" "$PREPROCESS_SCRIPT" )

echo "[3/3] Starting multi-turn VERL training…"
export MODEL_PATH OUTPUT_PATH EXP_NAME
export CARE_SERVER_URL="${SERVER_URL%/health}"
echo "  MODEL_PATH=$MODEL_PATH"
echo "  EXP_NAME=${EXP_NAME:-(auto)}"
echo "  OUTPUT_PATH=${OUTPUT_PATH:-(auto)}"
echo "  CARE_SERVER_URL=$CARE_SERVER_URL"
( cd "$VERL_DIR" && bash "$RUN_MULTI_TURN_SCRIPT" )

echo "multi_turn_rl pipeline completed."