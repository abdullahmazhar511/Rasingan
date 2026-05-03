#!/bin/bash
# Setup script to launch external vLLM servers for patient and supervisor models
# These run alongside the VERL training to provide patient responses and supervisor feedback
# 
# Usage:
#   ./setup_external_models.sh                    # Start both with defaults
#   ./setup_external_models.sh --patient-only    # Start only patient model
#   ./setup_external_models.sh --no-supervisor   # Start only patient (skip supervisor)
#   ./setup_external_models.sh --kill            # Kill all external servers
#   ./setup_external_models.sh --status          # Check if servers are running

set -e

# Configuration
PATIENT_MODEL="${PATIENT_MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
PATIENT_PORT="${PATIENT_PORT:-8001}"
PATIENT_GPU="${PATIENT_GPU:-0}"

SUPERVISOR_MODEL="${SUPERVISOR_MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
SUPERVISOR_PORT="${SUPERVISOR_PORT:-8002}"
SUPERVISOR_GPU="${SUPERVISOR_GPU:-1}"

LOG_DIR="./logs/external_models"
PATIENT_LOG="$LOG_DIR/patient_vllm_${PATIENT_PORT}.log"
SUPERVISOR_LOG="$LOG_DIR/supervisor_vllm_${SUPERVISOR_PORT}.log"

# Flags
START_PATIENT=true
START_SUPERVISOR=true
OPERATION="start"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --patient-only)
            START_SUPERVISOR=false
            shift
            ;;
        --no-supervisor)
            START_SUPERVISOR=false
            shift
            ;;
        --kill)
            OPERATION="kill"
            shift
            ;;
        --status)
            OPERATION="status"
            shift
            ;;
        --patient-model)
            PATIENT_MODEL="$2"
            shift 2
            ;;
        --supervisor-model)
            SUPERVISOR_MODEL="$2"
            shift 2
            ;;
        --patient-port)
            PATIENT_PORT="$2"
            shift 2
            ;;
        --supervisor-port)
            SUPERVISOR_PORT="$2"
            shift 2
            ;;
        --patient-gpu)
            PATIENT_GPU="$2"
            shift 2
            ;;
        --supervisor-gpu)
            SUPERVISOR_GPU="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--patient-only | --no-supervisor | --kill | --status]"
            echo "       [--patient-model MODEL] [--supervisor-model MODEL]"
            echo "       [--patient-port PORT] [--supervisor-port PORT]"
            echo "       [--patient-gpu GPU] [--supervisor-gpu GPU]"
            exit 1
            ;;
    esac
done

# Helper functions
ensure_log_dir() {
    mkdir -p "$LOG_DIR"
}

start_patient_server() {
    echo "🚀 Starting patient model on port $PATIENT_PORT (GPU $PATIENT_GPU)..."
    echo "   Model: $PATIENT_MODEL"
    ensure_log_dir
    
    CUDA_VISIBLE_DEVICES=$PATIENT_GPU python -m vllm.entrypoints.openai.api_server \
        --model "$PATIENT_MODEL" \
        --port $PATIENT_PORT \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.7 \
        > "$PATIENT_LOG" 2>&1 &
    
    PATIENT_PID=$!
    echo "$PATIENT_PID" > "$LOG_DIR/patient.pid"
    echo "✓ Patient server started (PID: $PATIENT_PID)"
    echo "  Log: $PATIENT_LOG"
    
    # Wait for server to be ready
    sleep 5
    if curl -s http://localhost:$PATIENT_PORT/v1/models > /dev/null 2>&1; then
        echo "✓ Patient server is ready!"
    else
        echo "⚠ Patient server may not be ready yet, waiting..."
        sleep 5
    fi
}

start_supervisor_server() {
    echo "🚀 Starting supervisor model on port $SUPERVISOR_PORT (GPU $SUPERVISOR_GPU)..."
    echo "   Model: $SUPERVISOR_MODEL"
    ensure_log_dir
    
    CUDA_VISIBLE_DEVICES=$SUPERVISOR_GPU python -m vllm.entrypoints.openai.api_server \
        --model "$SUPERVISOR_MODEL" \
        --port $SUPERVISOR_PORT \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.7 \
        > "$SUPERVISOR_LOG" 2>&1 &
    
    SUPERVISOR_PID=$!
    echo "$SUPERVISOR_PID" > "$LOG_DIR/supervisor.pid"
    echo "✓ Supervisor server started (PID: $SUPERVISOR_PID)"
    echo "  Log: $SUPERVISOR_LOG"
    
    # Wait for server to be ready
    sleep 5
    if curl -s http://localhost:$SUPERVISOR_PORT/v1/models > /dev/null 2>&1; then
        echo "✓ Supervisor server is ready!"
    else
        echo "⚠ Supervisor server may not be ready yet, waiting..."
        sleep 5
    fi
}

kill_servers() {
    echo "🛑 Killing external model servers..."
    
    if [ -f "$LOG_DIR/patient.pid" ]; then
        PID=$(cat "$LOG_DIR/patient.pid")
        if kill $PID 2>/dev/null; then
            echo "✓ Killed patient server (PID: $PID)"
        fi
        rm "$LOG_DIR/patient.pid"
    fi
    
    if [ -f "$LOG_DIR/supervisor.pid" ]; then
        PID=$(cat "$LOG_DIR/supervisor.pid")
        if kill $PID 2>/dev/null; then
            echo "✓ Killed supervisor server (PID: $PID)"
        fi
        rm "$LOG_DIR/supervisor.pid"
    fi
    
    # Also try to kill by port (in case PIDs don't exist)
    pkill -f "port $PATIENT_PORT" || true
    pkill -f "port $SUPERVISOR_PORT" || true
    
    echo "✓ Cleanup complete"
}

check_status() {
    echo "📊 External Model Server Status"
    echo "================================"
    
    # Check patient
    if curl -s http://localhost:$PATIENT_PORT/v1/models > /dev/null 2>&1; then
        echo "✓ Patient server: RUNNING (port $PATIENT_PORT, GPU $PATIENT_GPU)"
        echo "  Model: $PATIENT_MODEL"
    else
        echo "✗ Patient server: NOT RUNNING (port $PATIENT_PORT)"
    fi
    
    # Check supervisor
    if curl -s http://localhost:$SUPERVISOR_PORT/v1/models > /dev/null 2>&1; then
        echo "✓ Supervisor server: RUNNING (port $SUPERVISOR_PORT, GPU $SUPERVISOR_GPU)"
        echo "  Model: $SUPERVISOR_MODEL"
    else
        echo "✗ Supervisor server: NOT RUNNING (port $SUPERVISOR_PORT)"
    fi
    
    echo ""
    echo "Configuration for run_multiturn.sh:"
    echo "  data.patient_model.base_url=\"http://localhost:$PATIENT_PORT\""
    echo "  data.patient_model.model=\"$PATIENT_MODEL\""
    echo "  data.supervisor_model.base_url=\"http://localhost:$SUPERVISOR_PORT\""
    echo "  data.supervisor_model.model=\"$SUPERVISOR_MODEL\""
}

# Main execution
case $OPERATION in
    start)
        echo "🎯 Setting up external model servers for training"
        echo "=================================================="
        
        if [ "$START_PATIENT" = true ]; then
            start_patient_server
        fi
        
        if [ "$START_SUPERVISOR" = true ]; then
            start_supervisor_server
        fi
        
        echo ""
        echo "✅ External models ready!"
        echo ""
        echo "Next: Run training with:"
        echo "  cd /home/umairai/faithfulness_emnlp/Rasingan/verl"
        echo "  ./examples/faith/scripts/run_multiturn.sh"
        echo ""
        echo "To monitor:"
        echo "  tail -f $PATIENT_LOG"
        echo "  tail -f $SUPERVISOR_LOG"
        echo ""
        echo "To stop servers:"
        echo "  ./setup_external_models.sh --kill"
        ;;
    kill)
        kill_servers
        ;;
    status)
        check_status
        ;;
esac
