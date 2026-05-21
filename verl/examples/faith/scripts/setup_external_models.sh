#!/bin/bash
# Launch the external vLLM servers used by the multi-turn therapist training:
#   - Patient model server   (default port 8001)
#   - Supervisor model server (default port 8002)
#
# These are OpenAI-compatible servers spun up via vllm.entrypoints.openai.api_server.
# By default patient + supervisor share the SAME model (Qwen3-4B-Instruct-2507), so
# you can save a GPU + one model load by running one server in --shared mode.
#
# Usage:
#   ./setup_external_models.sh                    # start both (separate servers)
#   ./setup_external_models.sh --shared           # start ONE server for both roles
#   ./setup_external_models.sh --patient-only     # only patient
#   ./setup_external_models.sh --no-supervisor    # same as --patient-only
#   ./setup_external_models.sh --kill             # stop all servers we launched
#   ./setup_external_models.sh --status           # check what's running
#
# Env overrides:
#   PATIENT_MODEL / SUPERVISOR_MODEL                HF model id
#   PATIENT_PORT / SUPERVISOR_PORT                  TCP port
#   PATIENT_GPU / SUPERVISOR_GPU                    CUDA_VISIBLE_DEVICES value
#   GPU_MEM_UTIL                                    vllm --gpu-memory-utilization (default 0.40)
#   MAX_MODEL_LEN                                   vllm --max-model-len (default 8192)
#   SERVER_PYTHON                                   python binary to launch with
#                                                   (default: a venv with vLLM 0.19.1 if
#                                                   present, else the current python)
#   READY_TIMEOUT                                   seconds to wait for /v1/models (default 300)

set -e

DEFAULT_MODEL="Qwen/Qwen3-4B-Instruct-2507"

PATIENT_MODEL="${PATIENT_MODEL:-$DEFAULT_MODEL}"
PATIENT_PORT="${PATIENT_PORT:-8001}"
# PATIENT_GPU can be a single GPU ("1") or a CSV ("0,1") — count determines
# tensor-parallel-size automatically (unless TENSOR_PARALLEL_SIZE is set).
PATIENT_GPU="${PATIENT_GPU:-0}"

SUPERVISOR_MODEL="${SUPERVISOR_MODEL:-$DEFAULT_MODEL}"
SUPERVISOR_PORT="${SUPERVISOR_PORT:-8002}"
SUPERVISOR_GPU="${SUPERVISOR_GPU:-1}"

# Optional explicit override; if unset, derived from comma-count of *_GPU below.
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-}"

GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.40}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
READY_TIMEOUT="${READY_TIMEOUT:-300}"

# Pick a Python with a working vLLM. Prefer the venv that the rest of the
# pipeline already uses; fall back to whatever `python` resolves to.
if [ -z "${SERVER_PYTHON:-}" ]; then
    if [ -x /home/asbahk/hallucination/.venv/bin/python ]; then
        SERVER_PYTHON=/home/asbahk/hallucination/.venv/bin/python
    else
        SERVER_PYTHON="$(command -v python)"
    fi
fi

# CUDA 12.8 nvcc (needed only if we end up using verl env's vLLM 0.11.0 with
# flashinfer JIT compile). Cheap to export unconditionally — harmless if the
# chosen python doesn't need it.
if [ -x /usr/local/cuda-12.8/bin/nvcc ]; then
    export PATH=/usr/local/cuda-12.8/bin:$PATH
    export CUDA_HOME=/usr/local/cuda-12.8
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs/external_models"
PATIENT_LOG="$LOG_DIR/patient_vllm_${PATIENT_PORT}.log"
SUPERVISOR_LOG="$LOG_DIR/supervisor_vllm_${SUPERVISOR_PORT}.log"
SHARED_LOG="$LOG_DIR/shared_vllm_${PATIENT_PORT}.log"

# Flags
START_PATIENT=true
START_SUPERVISOR=true
SHARED=false
OPERATION="start"

while [[ $# -gt 0 ]]; do
    case $1 in
        --shared)              SHARED=true; shift ;;
        --patient-only)        START_SUPERVISOR=false; shift ;;
        --no-supervisor)       START_SUPERVISOR=false; shift ;;
        --kill)                OPERATION="kill"; shift ;;
        --status)              OPERATION="status"; shift ;;
        --patient-model)       PATIENT_MODEL="$2"; shift 2 ;;
        --supervisor-model)    SUPERVISOR_MODEL="$2"; shift 2 ;;
        --patient-port)        PATIENT_PORT="$2"; shift 2 ;;
        --supervisor-port)     SUPERVISOR_PORT="$2"; shift 2 ;;
        --patient-gpu)         PATIENT_GPU="$2"; shift 2 ;;
        --supervisor-gpu)      SUPERVISOR_GPU="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,/^set -e$/p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Run with --help for usage" >&2
            exit 1
            ;;
    esac
done

mkdir -p "$LOG_DIR"

# ----- helpers ---------------------------------------------------------------

wait_until_ready() {
    local url="$1"; local label="$2"; local pid="$3"
    local deadline=$(( $(date +%s) + READY_TIMEOUT ))
    while [ "$(date +%s)" -lt "$deadline" ]; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "❌ $label process exited before becoming ready. Tail of log:"
            tail -30 "$LOG_DIR"/*.log 2>/dev/null
            return 1
        fi
        if curl -sf --max-time 3 "$url/v1/models" -o /dev/null 2>/dev/null; then
            echo "✓ $label ready at $url"
            return 0
        fi
        sleep 5
    done
    echo "❌ $label did not become ready within ${READY_TIMEOUT}s"
    return 1
}

launch_server() {
    local model="$1"; local port="$2"; local gpu="$3"; local log="$4"; local pidfile="$5"; local label="$6"
    # Tensor parallelism: auto-derive from GPU count unless explicitly set.
    local tp
    if [ -n "$TENSOR_PARALLEL_SIZE" ]; then
        tp="$TENSOR_PARALLEL_SIZE"
    else
        tp=$(echo "$gpu" | awk -F',' '{print NF}')
    fi
    echo "🚀 Starting $label on port $port (GPU $gpu, tp=$tp): $model"
    echo "   python: $SERVER_PYTHON   log: $log"
    : > "$log"
    CUDA_VISIBLE_DEVICES="$gpu" nohup "$SERVER_PYTHON" -m vllm.entrypoints.openai.api_server \
        --model "$model" \
        --served-model-name "$model" \
        --host 127.0.0.1 \
        --port "$port" \
        --tensor-parallel-size "$tp" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --max-model-len "$MAX_MODEL_LEN" \
        >> "$log" 2>&1 &
    local pid=$!
    echo "$pid" > "$pidfile"
    echo "   PID: $pid"
    wait_until_ready "http://127.0.0.1:$port" "$label" "$pid"
}

stop_one() {
    local pidfile="$1"; local label="$2"
    if [ -f "$pidfile" ]; then
        local pid; pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            for _ in 1 2 3 4 5; do
                kill -0 "$pid" 2>/dev/null || break
                sleep 1
            done
            kill -9 "$pid" 2>/dev/null || true
            echo "✓ stopped $label (PID $pid)"
        fi
        rm -f "$pidfile"
    fi
}

# ----- ops -------------------------------------------------------------------

case $OPERATION in
    start)
        echo "🎯 Launching external model servers"
        echo "=================================================="

        if [ "$SHARED" = true ]; then
            # One server, both roles point at it. Patient port is reused as
            # the shared port; SUPERVISOR_PORT is unused in this mode.
            if [ "$PATIENT_MODEL" != "$SUPERVISOR_MODEL" ]; then
                echo "❌ --shared requires patient and supervisor models to be identical." >&2
                echo "   PATIENT_MODEL=$PATIENT_MODEL  SUPERVISOR_MODEL=$SUPERVISOR_MODEL" >&2
                exit 1
            fi
            launch_server "$PATIENT_MODEL" "$PATIENT_PORT" "$PATIENT_GPU" \
                          "$SHARED_LOG" "$LOG_DIR/shared.pid" "shared-patient+supervisor"
            echo ""
            echo "✅ Shared server ready. Point BOTH client URLs at http://127.0.0.1:$PATIENT_PORT"
        else
            if [ "$START_PATIENT" = true ]; then
                launch_server "$PATIENT_MODEL" "$PATIENT_PORT" "$PATIENT_GPU" \
                              "$PATIENT_LOG" "$LOG_DIR/patient.pid" "patient server"
            fi
            if [ "$START_SUPERVISOR" = true ]; then
                launch_server "$SUPERVISOR_MODEL" "$SUPERVISOR_PORT" "$SUPERVISOR_GPU" \
                              "$SUPERVISOR_LOG" "$LOG_DIR/supervisor.pid" "supervisor server"
            fi
        fi

        echo ""
        echo "Next: launch training. From the repo root:"
        echo "  cd $(cd "$SCRIPT_DIR/../../.." && pwd)"
        echo "  ./examples/faith/scripts/run_multiturn.sh"
        echo ""
        echo "To stop:  $0 --kill"
        echo "To check: $0 --status"
        ;;

    kill)
        echo "🛑 Stopping external model servers"
        stop_one "$LOG_DIR/patient.pid"    "patient server"
        stop_one "$LOG_DIR/supervisor.pid" "supervisor server"
        stop_one "$LOG_DIR/shared.pid"     "shared server"
        echo "✓ cleanup complete"
        ;;

    status)
        echo "📊 External Model Server Status"
        echo "================================"
        # Treat any pid we tracked as a candidate; also probe the configured ports.
        for label_port in "patient:$PATIENT_PORT" "supervisor:$SUPERVISOR_PORT"; do
            label="${label_port%%:*}"; port="${label_port##*:}"
            if curl -sf --max-time 3 "http://127.0.0.1:$port/v1/models" -o /dev/null 2>/dev/null; then
                model=$(curl -s --max-time 3 "http://127.0.0.1:$port/v1/models" | \
                        "$SERVER_PYTHON" -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "?")
                echo "✓ $label server: RUNNING (port $port)  model=$model"
            else
                echo "✗ $label server: NOT RUNNING (port $port)"
            fi
        done
        echo ""
        echo "Configuration for run_multiturn.sh:"
        echo "  data.patient_model.base_url=\"http://127.0.0.1:$PATIENT_PORT\""
        echo "  data.patient_model.model=\"$PATIENT_MODEL\""
        echo "  data.supervisor_model.base_url=\"http://127.0.0.1:$SUPERVISOR_PORT\""
        echo "  data.supervisor_model.model=\"$SUPERVISOR_MODEL\""
        ;;
esac
