#!/usr/bin/env bash
# Single-turn RL: start CARE reward server → preprocess → run verl PPO.
#
# Configure model + output dir here (or override via env when invoking):
#   MODEL_PATH=/path/to/merged/hf/dir  OUTPUT_PATH=./checkpoints/my_run  bash single_turn_rl.sh

set -euo pipefail

# ---- knobs (override via env) ----------------------------------------------
MODEL_PATH="${MODEL_PATH:-/home/asbahk/EMNLP_FINAL/Rasingan/sft_training/results/Qwen3-4B-sft-respair-new-3-merged}"
EXP_NAME="qwen_3_v2"         # empty → run_single_turn.sh picks qwen_3_v1_<ts>
OUTPUT_PATH="${OUTPUT_PATH:-}"   # empty → run_single_turn.sh picks ./checkpoints/<EXP_NAME>
DATA_CONTEXT_WINDOW="${DATA_CONTEXT_WINDOW:-6}"
SERVER_URL="http://127.0.0.1:8000/health"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RASINGAN_ROOT="$(dirname "$SCRIPT_DIR")"
SERVER_DIR="$RASINGAN_ROOT/server"
PREPROCESS_SCRIPT="$RASINGAN_ROOT/verl/examples/faith/data_preprocess/preprocess_singleturn.py"
RUN_SINGLE_TURN_SCRIPT="$RASINGAN_ROOT/verl/examples/faith/scripts/run_single_turn.sh"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-verl}"
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1090
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

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
        curl -sSf "$SERVER_URL" >/dev/null 2>&1 && { echo "Server healthy: $SERVER_URL"; return 0; }
        sleep 2
    done
    echo "Server did not become healthy at $SERVER_URL" >&2
    return 1
}

echo "[1/3] Starting CARE server…"
if curl -sSf "$SERVER_URL" >/dev/null 2>&1; then
    echo "Server already running; reusing existing instance."
else
    RUN_TS="$(date '+%Y%m%d_%H%M%S')"
    SERVER_LOG_FILE="$SERVER_DIR/server_single_turn_rl_${RUN_TS}.log"
    setsid bash -lc "cd '$SERVER_DIR' && bash server.sh prod" >"$SERVER_LOG_FILE" 2>&1 &
    SERVER_PGID=$!
    echo "Server logs: $SERVER_LOG_FILE"
    wait_for_server
fi

echo "[2/3] Building single-turn dataset parquet files…"
( cd "$RASINGAN_ROOT/verl" && python "$PREPROCESS_SCRIPT" --context_window "$DATA_CONTEXT_WINDOW" )

echo "[3/3] Starting single-turn VERL training…"
export MODEL_PATH OUTPUT_PATH EXP_NAME
export CARE_SERVER_URL="${SERVER_URL%/health}"
echo "  MODEL_PATH=$MODEL_PATH"
echo "  EXP_NAME=${EXP_NAME:-(auto)}"
echo "  OUTPUT_PATH=${OUTPUT_PATH:-(auto)}"
echo "  CARE_SERVER_URL=$CARE_SERVER_URL"
( cd "$RASINGAN_ROOT/verl" && bash "$RUN_SINGLE_TURN_SCRIPT" )

echo "single_turn_rl pipeline completed."