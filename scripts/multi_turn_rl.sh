#!/usr/bin/env bash
# Multi-turn RL: start CARE reward server → preprocess → run verl PPO
# (the inner script launches the shared Patient+Supervisor vLLM server itself).
#
# Configure model + output dir here (or override via env when invoking):
#   MODEL_PATH=/path/to/merged/hf/dir  OUTPUT_PATH=./checkpoints/my_run  EXP_NAME=my_exp  bash multi_turn_rl.sh

set -euo pipefail

# ---- knobs (override via env) ----------------------------------------------
MODEL_PATH="${MODEL_PATH:-/home/asbahk/EMNLP_FINAL/Rasingan/verl/checkpoints/qwen_3_v3/global_step_514-merged}"
EXP_NAME="${EXP_NAME:-therapist_multiturn_final_v3}"               # empty → run_multiturn.sh picks therapist_multiturn_<ts>
OUTPUT_PATH="${OUTPUT_PATH:-}"         # empty → run_multiturn.sh picks ./checkpoints/<EXP_NAME>
RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"   # empty → auto-pick latest in SAVE_PATH
# Multi-turn reward is automatically IR + CTRS-P (computed from the full
# session transcript at end of rollout) — see _compute_multiturn_reward in
# verl/.../faith.py. EMB_REWARD_WEIGHT only applies to single-turn runs.

# Multi-turn rollout shape — bigger = more realistic sessions but slower steps.
MAX_THERAPY_TURNS="${MAX_THERAPY_TURNS:-20}"

# Limit number of conversations used (stratified per category). Default
# subsamples to keep training tractable: 240 train (120 anxiety + 120 dep),
# 25 val. Set to 0 or unset to disable.
TRAIN_LIMIT="${TRAIN_LIMIT:-240}"
VAL_LIMIT="${VAL_LIMIT:-25}"

# Qwen3 instruct (non-thinking) sampling for the shared patient+supervisor
# model. Default ON (recommended for Qwen3 instruct variants — disables
# thinking-mode prefix). Set INSTRUCT_MODE=0 to fall back to caller-supplied
# temperature/top_p.
INSTRUCT_MODE="${INSTRUCT_MODE:-1}"

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
PREPROCESS_ARGS=()
[ -n "${TRAIN_LIMIT:-}" ] && [ "$TRAIN_LIMIT" != "0" ] && PREPROCESS_ARGS+=(--train-limit "$TRAIN_LIMIT")
[ -n "${VAL_LIMIT:-}"   ] && [ "$VAL_LIMIT"   != "0" ] && PREPROCESS_ARGS+=(--val-limit   "$VAL_LIMIT")
echo "  train_limit=${TRAIN_LIMIT:-(all)}  val_limit=${VAL_LIMIT:-(all)}"
( cd "$VERL_DIR" && "$CONDA_PY" "$PREPROCESS_SCRIPT" "${PREPROCESS_ARGS[@]}" )

echo "[3/3] Starting multi-turn VERL training…"
export MODEL_PATH OUTPUT_PATH EXP_NAME MAX_THERAPY_TURNS INSTRUCT_MODE TOTAL_EPOCHS RESUME_FROM_PATH
export CARE_SERVER_URL="${SERVER_URL%/health}"
echo "  MODEL_PATH=$MODEL_PATH"
echo "  EXP_NAME=${EXP_NAME:-(auto)}"
echo "  OUTPUT_PATH=${OUTPUT_PATH:-(auto)}"
echo "  MAX_THERAPY_TURNS=$MAX_THERAPY_TURNS"
echo "  INSTRUCT_MODE=$INSTRUCT_MODE  (Qwen3 non-thinking sampling for patient+supervisor)"
echo "  REWARD=IR + CTRS-P  (session-level, from supervisor checklist + CARE on therapist turns)"
echo "  CARE_SERVER_URL=$CARE_SERVER_URL"
( cd "$VERL_DIR" && bash "$RUN_MULTI_TURN_SCRIPT" )

echo "multi_turn_rl pipeline completed."