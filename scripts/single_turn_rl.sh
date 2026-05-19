#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RASINGAN_ROOT="$(dirname "$SCRIPT_DIR")"
SERVER_DIR="$RASINGAN_ROOT/server"
VERL_DIR="$RASINGAN_ROOT/verl"
PREPROCESS_SCRIPT="$VERL_DIR/examples/faith/data_preprocess/preprocess_singleturn.py"
RUN_SINGLE_TURN_SCRIPT="$VERL_DIR/examples/faith/scripts/run_single_turn.sh"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-verl}"
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1090
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

SERVER_URL="${SERVER_URL:-http://127.0.0.1:8000/health}"
START_SERVER="${START_SERVER:-1}"
DATA_CONTEXT_WINDOW="${DATA_CONTEXT_WINDOW:-6}"
RUN_TS="$(date '+%Y%m%d_%H%M%S')"
SERVER_LOG_FILE="${SERVER_LOG_FILE:-$SERVER_DIR/server_single_turn_rl_${RUN_TS}.log}"

# Keep reward endpoint aligned with server health URL/port.
SERVER_BASE_URL="${SERVER_URL%/health}"
CARE_SERVER_URL="${CARE_SERVER_URL:-$SERVER_BASE_URL}"

SERVER_PID=""
SERVER_STARTED_BY_SCRIPT="0"
SERVER_PGID=""

cleanup() {
	if [[ "$SERVER_STARTED_BY_SCRIPT" != "1" ]]; then
		return
	fi

	if [[ -n "$SERVER_PGID" ]] && kill -0 "-$SERVER_PGID" >/dev/null 2>&1; then
		echo "Stopping server process group (pgid=$SERVER_PGID)..."
		kill -TERM "-$SERVER_PGID" >/dev/null 2>&1 || true
		return
	fi

	if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
		echo "Stopping server (pid=$SERVER_PID)..."
		kill -TERM "$SERVER_PID" >/dev/null 2>&1 || true
	fi
}

handle_interrupt() {
	echo "Interrupt received, shutting down..."
	cleanup
	exit 130
}

trap handle_interrupt INT TERM
trap cleanup EXIT

wait_for_server() {
	local retries=60
	local delay=2
	local i

	for ((i = 1; i <= retries; i++)); do
		if curl -sSf "$SERVER_URL" >/dev/null 2>&1; then
			echo "Server is healthy: $SERVER_URL"
			return 0
		fi
		sleep "$delay"
	done

	echo "Server did not become healthy at $SERVER_URL" >&2
	return 1
}

echo "[1/3] Starting CARE server..."
if [[ "$START_SERVER" == "1" ]]; then
	if curl -sSf "$SERVER_URL" >/dev/null 2>&1; then
		echo "Server already running; reusing existing instance."
	else
		# Start in a new process group so Ctrl+C cleanup can terminate all descendants.
		setsid bash -lc "cd '$SERVER_DIR' && bash server.sh prod" >"$SERVER_LOG_FILE" 2>&1 &
		SERVER_PID=$!
		SERVER_STARTED_BY_SCRIPT="1"
		SERVER_PGID="$SERVER_PID"
		echo "Server logs: $SERVER_LOG_FILE"
		wait_for_server
	fi
else
	echo "Skipping server start (START_SERVER=$START_SERVER)."
fi

echo "[2/3] Building single-turn dataset parquet files..."
(
	cd "$VERL_DIR"
	python "$PREPROCESS_SCRIPT" --context_window "$DATA_CONTEXT_WINDOW"
)

echo "[3/3] Starting single-turn VERL training..."
(
	cd "$VERL_DIR"
	export CARE_SERVER_URL
	echo "Using CARE_SERVER_URL=$CARE_SERVER_URL"
	bash "$RUN_SINGLE_TURN_SCRIPT"
)

echo "single_turn_rl pipeline completed."
