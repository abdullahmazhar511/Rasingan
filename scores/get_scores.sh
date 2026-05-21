#!/bin/bash

# Evaluation pipeline for a model — split into two phases:
#
#   PHASE A  (single-turn, benchmark-style)
#     A1  generate.py             → <model>/responses/*.csv
#     A2  score_care.py           → <model>/care_scores/*.csv
#     A3  score_care_loss.py      → <model>/care_loss.json
#     A4  score_nlp.py            → <model>/nlp_metrics.json
#
#   PHASE B  (multi-turn, final_pipeline therapy sessions)
#     B1  final_pipeline/run.py   → <model>/sessions/session_*.json
#     B2  score_ctrs.py           → <model>/ctrs_scores.json
#     B3  score_information_retrieval.py → <model>/ir_score.json
#
# Phases can be skipped independently with --skip-single-turn / --skip-multi-turn.
#
# --------------------------------------------------------------------------
# EXAMPLE INVOCATIONS
# --------------------------------------------------------------------------
#
# 1) Base-model evaluation (no merge):
#
#     bash ./scores/get_scores.sh \
#         --model-name qwen-base \
#         --base-model Qwen/Qwen3-4B-Instruct-2507 \
#         --max-turns 20 \
#         --categories "anxiety depression" \
#         --n-per-category 20 \
#         --concurrency 4
#
# 2) SFT (or RL) evaluation via PEFT merge — the adapter is folded into a
#    standalone HF dir BEFORE Phase A/B, then both phases load that dir.
#
#     bash ./scores/get_scores.sh \
#         --model-name qwen-sft \
#         --base-model Qwen/Qwen3-4B-Instruct-2507 \
#         --merge sft \
#         --merge-checkpoint /home/asbahk/EMNLP_FINAL/Rasingan/sft_training/results/Qwen3-4B-sft-respair-new-3/checkpoint-425 \
#         --max-turns 20 \
#         --categories "anxiety depression" \
#         --n-per-category 20 \
#         --concurrency 4
#
# 3) Same but for a verl FSDP RL checkpoint (model_merger handles desharding):
#
#     bash ./scores/get_scores.sh \
#         --model-name qwen-rl \
#         --merge verl \
#         --merge-checkpoint /home/asbahk/EMNLP_FINAL/Rasingan/verl/checkpoints/qwen_3_v1_XX-XX-XX/global_step_120 \
#         --max-turns 20 \
#         --categories "anxiety depression" \
#         --n-per-category 20 \
#         --concurrency 4
#
# Common skips:
#     --skip-single-turn         only do Phase B (CTRS, IR)
#     --skip-multi-turn          only do Phase A (NLP, CARE-loss)
#     -s / --skip-generation     keep Phase A but reuse existing responses/

set -e

CONDA_ENV_NAME="${CONDA_ENV_NAME:-verl}"
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1090
source "$CONDA_BASE/etc/profile.d/conda.sh"
# If a Python venv is active in the calling shell, it shadows conda activate
# (its bin/ is earlier on PATH). Deactivate it so we actually use $CONDA_ENV_NAME's python.
if [ -n "${VIRTUAL_ENV:-}" ] && command -v deactivate &>/dev/null; then
    deactivate 2>/dev/null || true
fi
unset VIRTUAL_ENV
conda activate "$CONDA_ENV_NAME"

# CUDA 12.8 for nvcc — the system nvcc on this box is 11.5 which can't emit
# compute_90a/sm_90a (H100/H200). flashinfer's JIT-compile of its sampler
# kernel needs sm_90a; without this, vLLM's engine core blows up at startup.
# This mirrors verl/examples/faith/scripts/run_single_turn.sh.
if [ -x /usr/local/cuda-12.8/bin/nvcc ]; then
    export PATH=/usr/local/cuda-12.8/bin:$PATH
    export CUDA_HOME=/usr/local/cuda-12.8
fi

# Pin python to the conda env binary by absolute path — `conda activate` alone
# isn't enough when a Python venv is on PATH first; we use $CONDA_PY everywhere.
CONDA_PY="$CONDA_PREFIX/bin/python"
if [ ! -x "$CONDA_PY" ]; then
    echo "ERROR: conda env python not found at $CONDA_PY" >&2
    exit 1
fi
echo "[env] CONDA_DEFAULT_ENV=$CONDA_DEFAULT_ENV  python=$CONDA_PY"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RASINGAN_DIR="$(dirname "$SCRIPT_DIR")"
EVAL_ROOT="${EVAL_ROOT:-$RASINGAN_DIR/evaluation_pipeline}"
DATASET_PATH="${DATASET_PATH:-$RASINGAN_DIR/respair_mhcopilot_format}"
CONTEXT_WINDOW="${CONTEXT_WINDOW:-6}"
MAX_TURNS="${MAX_TURNS:-20}"
N_SCENARIOS="${N_SCENARIOS:-}"  # cap: run only the first N from the selected set
CONCURRENCY="${CONCURRENCY:-4}"  # parallel sessions sharing the same vLLM clients

# CSV-driven scenarios — Phase B always walks the rows of this CSV.
# Default = the multiturn_reddit_data test split (20 anxiety + 20 depression
# when combined with --categories / --n-per-category). Override via env if you
# need a different file (e.g. SCENARIOS_FROM=.../splits/val.csv).
SCENARIOS_FROM="${SCENARIOS_FROM:-$RASINGAN_DIR/multiturn_reddit_data/splits/test.csv}"
SCENARIO_CATEGORIES="${SCENARIO_CATEGORIES:-anxiety depression}"   # space-separated
N_PER_CATEGORY="${N_PER_CATEGORY:-}"                                # e.g. 20 → 20 per category

# ---- Phase B (multi-turn) configuration ----------------------------------
# Therapist (Agent 1) runs in the current conda env (verl) on its own GPU.
# Patient + Supervisor share a vLLM OpenAI-compatible server launched in a
# separate Python env on a different GPU (so the verl env's older vLLM doesn't
# have to host them; also avoids flashinfer JIT issues for the supervisor).
THERAPIST_MODEL="${THERAPIST_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
SHARED_MODEL="${SHARED_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
SUPERVISOR_PYTHON="${SUPERVISOR_PYTHON:-/home/asbahk/hallucination/.venv/bin/python}"
SUPERVISOR_GPU="${SUPERVISOR_GPU:-1}"
THERAPIST_GPU="${THERAPIST_GPU:-0}"
SUPERVISOR_GPU_MEM_UTIL="${SUPERVISOR_GPU_MEM_UTIL:-0.55}"
SUPERVISOR_MAX_LEN="${SUPERVISOR_MAX_LEN:-8192}"
SUPERVISOR_READY_TIMEOUT="${SUPERVISOR_READY_TIMEOUT:-300}"  # seconds

# ---- Therapist (Agent 1) loads in-process via HuggingFace transformers ----
# The therapist is the model under evaluation. It runs IN PROCESS inside run.py
# (no vLLM, no HTTP server). For SFT/RL runs use --merge to fold the adapter
# into a standalone HF dir first; the therapist always loads a self-contained
# model directly. GPU pinning happens via CUDA_VISIBLE_DEVICES=$THERAPIST_GPU,
# so inside Python the therapist always sees "cuda:0" (no separate knob needed).

# NOTE: flashinfer JIT-compile works fine with the CUDA 12.8 nvcc set above.
# No need for VLLM_USE_FLASHINFER_SAMPLER=0; if it still fails set it manually.

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
print_status()  { echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"; }
print_success() { echo -e "${GREEN}✓ $1${NC}"; }
print_error()   { echo -e "${RED}✗ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠ $1${NC}"; }

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

OPTIONS:
    -m, --model-name NAME        Model name (folder under EVAL_ROOT)
    -b, --base-model PATH        Base model path from HuggingFace
    -d, --dataset PATH           Single-turn dataset (default: $DATASET_PATH)
    -c, --context-window N       Context window for generate.py (default: $CONTEXT_WINDOW)
    -e, --eval-root PATH         Evaluation root directory (default: $EVAL_ROOT)
    -s, --skip-generation        Phase A: skip generate.py (use existing responses/)
        --skip-single-turn       Skip Phase A entirely (no CARE / NLP)
        --skip-sessions          Phase B: skip running final_pipeline (use existing sessions/)
        --skip-multi-turn        Skip Phase B entirely (no CTRS / IR)
        --n-scenarios N          Cap: run only the first N scenarios from the selected set
        --concurrency N          Run N scenarios in parallel (default: $CONCURRENCY)
        --categories "a d"       Restrict CSV scenarios to these categories
                                 (default: "$SCENARIO_CATEGORIES")
        --n-per-category N       Take first N rows from each category (e.g. 20)
        --merge sft|verl         Merge a checkpoint into a standalone HF dir
                                 BEFORE Phase A/B. "sft" uses PEFT
                                 (sft_training/merge_peft_checkpoint.py).
                                 "verl" uses verl.model_merger (FSDP backend).
        --merge-checkpoint PATH  Path to the unmerged checkpoint to merge from.
                                 sft → PEFT adapter dir.
                                 verl → an FSDP actor dir (e.g. global_step_N/actor).
        --merge-output PATH      Where to write the merged dir. Default: alongside
                                 the checkpoint with "-merged" appended.
                                 (skips auto-launch).
        --max-turns N            Max turns per simulated session (default: $MAX_TURNS)
    -h, --help                   Show this help

ENV OVERRIDES:
    EVAL_ROOT, DATASET_PATH, CONTEXT_WINDOW, MAX_TURNS, SCENARIOS_FROM

EOF
    exit 1
}

# Defaults
MODEL_NAME="qwen_rl"
# BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}" 
BASE_MODEL="/home/asbahk/EMNLP_FINAL/Rasingan/sft_training/results/Qwen3-4B-sft-respair-new-3-merged" #checkpoint for sft model used for rl"
MERGE_MODE="verl"               # "" | "sft" | "verl"
MERGE_CHECKPOINT="/home/asbahk/EMNLP_FINAL/Rasingan/verl/checkpoints/qwen_3_v1_21-09-39/global_step_160"   # path to the unmerged checkpoint
MERGE_OUTPUT="${MERGE_OUTPUT:-}"           # where to write merged dir; auto if empty
SKIP_SINGLE_TURN=false
SKIP_GENERATION=false
SKIP_MULTI_TURN=true
SKIP_SESSIONS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model-name)      MODEL_NAME="$2";   shift 2 ;;
        -b|--base-model)      BASE_MODEL="$2";   shift 2 ;;
        -d|--dataset)         DATASET_PATH="$2"; shift 2 ;;
        -c|--context-window)  CONTEXT_WINDOW="$2"; shift 2 ;;
        -e|--eval-root)       EVAL_ROOT="$2";    shift 2 ;;
        -s|--skip-generation) SKIP_GENERATION=true; shift ;;
        --skip-single-turn)   SKIP_SINGLE_TURN=true; shift ;;
        --skip-sessions)      SKIP_SESSIONS=true; shift ;;
        --skip-multi-turn)    SKIP_MULTI_TURN=true; shift ;;
        --n-scenarios)        N_SCENARIOS="$2";  shift 2 ;;
        --concurrency)        CONCURRENCY="$2";  shift 2 ;;
        --categories)         SCENARIO_CATEGORIES="$2"; shift 2 ;;
        --n-per-category)     N_PER_CATEGORY="$2"; shift 2 ;;
        --merge)              MERGE_MODE="$2"; shift 2 ;;
        --merge-checkpoint)   MERGE_CHECKPOINT="$2"; shift 2 ;;
        --merge-output)       MERGE_OUTPUT="$2"; shift 2 ;;
        --max-turns)          MAX_TURNS="$2";    shift 2 ;;
        -h|--help)            usage ;;
        *) print_error "Unknown option: $1"; usage ;;
    esac
done

[ -z "$MODEL_NAME" ] && { print_error "Model name is required"; usage; }
if [ "$SKIP_SINGLE_TURN" = false ] && [ "$SKIP_GENERATION" = false ] && [ -z "$BASE_MODEL" ]; then
    print_error "Base model is required (or use --skip-generation / --skip-single-turn)"
    usage
fi

print_status "========================================="
print_status "Evaluation Pipeline"
print_status "========================================="
print_status "Model Name:        $MODEL_NAME"
print_status "Base Model:        $BASE_MODEL"
print_status "Eval Root:         $EVAL_ROOT"
print_status "Phase A (single):  skip-single-turn=$SKIP_SINGLE_TURN  skip-generation=$SKIP_GENERATION"
print_status "Phase B (multi):   skip-multi-turn=$SKIP_MULTI_TURN    skip-sessions=$SKIP_SESSIONS"

MODEL_DIR="$EVAL_ROOT/$MODEL_NAME"
RESPONSES_DIR="$MODEL_DIR/responses"
CARE_SCORES_DIR="$MODEL_DIR/care_scores"
SESSIONS_DIR="$MODEL_DIR/sessions"
mkdir -p "$RESPONSES_DIR" "$CARE_SCORES_DIR" "$SESSIONS_DIR"

export EVAL_ROOT DATASET_PATH

# ============================================================================
# OPTIONAL: merge a PEFT (SFT) or FSDP (VERL) checkpoint into a standalone dir.
# Runs BEFORE Phase A/B. On success, rewrites BASE_MODEL and THERAPIST_MODEL
# to the merged dir so both phases see the merged weights.
# ============================================================================
if [ -n "$MERGE_MODE" ]; then
    if [ -z "$MERGE_CHECKPOINT" ]; then
        print_error "--merge $MERGE_MODE requires --merge-checkpoint <path>"
        exit 1
    fi
    if [ ! -d "$MERGE_CHECKPOINT" ]; then
        print_error "merge checkpoint dir not found: $MERGE_CHECKPOINT"
        exit 1
    fi
    # Default output dir: alongside the checkpoint with "-merged" suffix.
    if [ -z "$MERGE_OUTPUT" ]; then
        MERGE_OUTPUT="${MERGE_CHECKPOINT%/}-merged"
    fi
    print_status "========================================="
    print_status "MERGE — preparing standalone checkpoint"
    print_status "========================================="
    print_status "  mode:       $MERGE_MODE"
    print_status "  checkpoint: $MERGE_CHECKPOINT"
    print_status "  output:     $MERGE_OUTPUT"

    # Idempotent: skip the merge if the output dir already looks like a merged model.
    if [ -f "$MERGE_OUTPUT/config.json" ]; then
        print_warning "  $MERGE_OUTPUT already contains a merged model — skipping merge."
        print_warning "  Delete the directory if you want to redo it."
    else
        case "$MERGE_MODE" in
            sft)
                SFT_MERGE_PY="$RASINGAN_DIR/sft_training/merge_peft_checkpoint.py"
                if [ ! -f "$SFT_MERGE_PY" ]; then
                    print_error "SFT merge script missing: $SFT_MERGE_PY"; exit 1
                fi
                print_status "  invoking: $CONDA_PY $SFT_MERGE_PY --base-model $BASE_MODEL --checkpoint-dir $MERGE_CHECKPOINT --output-dir $MERGE_OUTPUT"
                "$CONDA_PY" "$SFT_MERGE_PY" \
                    --base-model     "$BASE_MODEL" \
                    --checkpoint-dir "$MERGE_CHECKPOINT" \
                    --output-dir     "$MERGE_OUTPUT" \
                    || { print_error "SFT merge failed"; exit 1; }
                ;;
            verl)
                # verl.model_merger expects an FSDP actor dir directly.
                if [ -d "$MERGE_CHECKPOINT/actor" ]; then
                    VERL_LOCAL_DIR="$MERGE_CHECKPOINT/actor"
                else
                    VERL_LOCAL_DIR="$MERGE_CHECKPOINT"
                fi
                print_status "  invoking: $CONDA_PY -m verl.model_merger merge --backend fsdp"
                print_status "             --local_dir $VERL_LOCAL_DIR --target_dir $MERGE_OUTPUT"
                mkdir -p "$MERGE_OUTPUT"
                "$CONDA_PY" -m verl.model_merger merge \
                    --backend fsdp \
                    --local_dir "$VERL_LOCAL_DIR" \
                    --target_dir "$MERGE_OUTPUT" \
                    || { print_error "VERL merge failed"; exit 1; }
                ;;
            *)
                print_error "--merge must be 'sft' or 'verl', got: $MERGE_MODE"; exit 1
                ;;
        esac
    fi

    # Sanity check the result looks like a real HF model.
    if [ ! -f "$MERGE_OUTPUT/config.json" ]; then
        print_error "Merge produced no config.json at $MERGE_OUTPUT — something is wrong."
        exit 1
    fi
    print_success "Merged model ready at: $MERGE_OUTPUT"

    # Rewire downstream so Phase A + Phase B use the merged dir.
    BASE_MODEL="$MERGE_OUTPUT"
    THERAPIST_MODEL="$MERGE_OUTPUT"  # Phase B HF in-process therapist
    print_status "After merge: BASE_MODEL / THERAPIST_MODEL → $MERGE_OUTPUT"
fi

# ============================================================================
# PHASE A — Single-turn benchmark metrics
# ============================================================================
if [ "$SKIP_SINGLE_TURN" = false ]; then
    print_status "========================================="
    print_status "PHASE A — Single-turn benchmark metrics"
    print_status "========================================="

    # A1: Generate responses
    if [ "$SKIP_GENERATION" = false ]; then
        print_status "[A1] Generating responses…"
        cd "$RASINGAN_DIR"
        "$CONDA_PY" "$SCRIPT_DIR/generate.py" \
            --model-name "$MODEL_NAME" \
            --base-model "$BASE_MODEL" \
            --adapter-path null \
            --eval-root "$EVAL_ROOT" \
            --context-window "$CONTEXT_WINDOW" \
            --dataset-path "$DATASET_PATH" || { print_error "Generation failed"; exit 1; }
        print_success "[A1] Response generation completed"
    else
        print_warning "[A1] Skipping generation (using existing $RESPONSES_DIR)"
    fi

    if [ ! -d "$RESPONSES_DIR" ] || [ -z "$(ls -A "$RESPONSES_DIR" 2>/dev/null)" ]; then
        print_error "No responses in $RESPONSES_DIR — cannot run A2/A3/A4"
        exit 1
    fi

    cd "$SCRIPT_DIR"

    # Wipe stale care_scores so A2 always rescores. score_care.py now also
    # regenerates the gold-response CARE scores (overwriting dataset
    # annotations), so any pre-existing CSV in care_scores/ is incompatible
    # with the L1/L2 loss A3 will compute.
    if [ -d "$CARE_SCORES_DIR" ] && [ -n "$(ls -A "$CARE_SCORES_DIR" 2>/dev/null)" ]; then
        print_status "[A2] Clearing stale $CARE_SCORES_DIR (re-scoring model + gold)"
        rm -f "$CARE_SCORES_DIR"/*.csv
    fi

    print_status "[A2] Computing CARE scores (model + gold, same classifier)…"
    "$CONDA_PY" score_care.py --eval-root "$EVAL_ROOT" --model-name "$MODEL_NAME" \
        || print_warning "CARE scoring encountered an issue (non-fatal)"
    print_success "[A2] CARE scoring completed"

    print_status "[A3] Computing CARE Loss (L1/L2)…"
    "$CONDA_PY" score_care_loss.py --eval-root "$EVAL_ROOT" --model-name "$MODEL_NAME" \
        || print_warning "CARE loss computation encountered an issue (non-fatal)"
    print_success "[A3] CARE loss computation completed"

    print_status "[A4] Computing NLP metrics…"
    "$CONDA_PY" score_nlp.py --eval-root "$EVAL_ROOT" --model-name "$MODEL_NAME" \
        || print_warning "NLP metrics computation encountered an issue (non-fatal)"
    print_success "[A4] NLP metrics computation completed"
else
    print_warning "Skipping PHASE A (single-turn metrics)"
fi

# ============================================================================
# PHASE B — Multi-turn final_pipeline metrics
# ============================================================================
SUP_SERVER_PID=""
SUP_SERVER_LOG=""

_terminate_pid() {
    local pid="$1"; local label="$2"
    [ -z "$pid" ] && return
    if kill -0 "$pid" 2>/dev/null; then
        print_status "[B0] Stopping $label (PID $pid)…"
        kill "$pid" 2>/dev/null || true
        for _ in 1 2 3 4 5; do
            kill -0 "$pid" 2>/dev/null || break
            sleep 1
        done
        kill -9 "$pid" 2>/dev/null || true
    fi
}

cleanup_servers() {
    _terminate_pid "$SUP_SERVER_PID" "supervisor server"
}
trap cleanup_servers EXIT INT TERM

start_supervisor_server() {
    local model="$1"
    local port="$2"
    local gpu="$3"
    SUP_SERVER_LOG="$MODEL_DIR/supervisor_server.log"
    : > "$SUP_SERVER_LOG"

    if [ ! -x "$SUPERVISOR_PYTHON" ]; then
        print_error "SUPERVISOR_PYTHON not found or not executable: $SUPERVISOR_PYTHON"
        print_error "Set SUPERVISOR_PYTHON to the path of a python binary in an env with a working vLLM."
        return 1
    fi

    print_status "[B0] Launching supervisor model on GPU $gpu via $SUPERVISOR_PYTHON"
    print_status "[B0]   model=$model  port=$port  gpu_mem_util=$SUPERVISOR_GPU_MEM_UTIL  max_len=$SUPERVISOR_MAX_LEN"
    print_status "[B0]   server log: $SUP_SERVER_LOG"

    CUDA_VISIBLE_DEVICES="$gpu" nohup "$SUPERVISOR_PYTHON" -m vllm.entrypoints.openai.api_server \
        --model "$model" \
        --host 127.0.0.1 \
        --port "$port" \
        --gpu-memory-utilization "$SUPERVISOR_GPU_MEM_UTIL" \
        --max-model-len "$SUPERVISOR_MAX_LEN" \
        --served-model-name "$model" \
        >> "$SUP_SERVER_LOG" 2>&1 &
    SUP_SERVER_PID=$!
    print_status "[B0]   supervisor PID=$SUP_SERVER_PID — waiting up to ${SUPERVISOR_READY_TIMEOUT}s for readiness…"

    local deadline=$(( $(date +%s) + SUPERVISOR_READY_TIMEOUT ))
    while [ "$(date +%s)" -lt "$deadline" ]; do
        if ! kill -0 "$SUP_SERVER_PID" 2>/dev/null; then
            print_error "[B0] Supervisor server died before becoming ready. Last log lines:"
            tail -30 "$SUP_SERVER_LOG" >&2
            return 1
        fi
        if curl -sf "http://127.0.0.1:${port}/v1/models" -o /dev/null 2>/dev/null; then
            print_success "[B0] Supervisor server ready at http://127.0.0.1:${port}/v1"
            return 0
        fi
        sleep 5
    done

    print_error "[B0] Supervisor server did not become ready within ${SUPERVISOR_READY_TIMEOUT}s. Last log lines:"
    tail -30 "$SUP_SERVER_LOG" >&2
    return 1
}

if [ "$SKIP_MULTI_TURN" = false ]; then
    print_status "========================================="
    print_status "PHASE B — Multi-turn final_pipeline metrics"
    print_status "========================================="

    # B1: Run final_pipeline therapy sessions
    if [ "$SKIP_SESSIONS" = false ]; then
        # Launch the supervisor vLLM HTTP server (port 8001, GPU $SUPERVISOR_GPU).
        start_supervisor_server "$SHARED_MODEL" 8001 "$SUPERVISOR_GPU" || {
            print_error "[B0] Failed to start supervisor server — aborting Phase B"
            exit 1
        }
        SUPERVISOR_ENDPOINT="http://127.0.0.1:8001/v1"

        print_status "[B1] Running final_pipeline therapy sessions → $SESSIONS_DIR"
        print_status "[B1]   therapist_model=$THERAPIST_MODEL  (HF in-process on GPU $THERAPIST_GPU)"
        print_status "[B1]   shared_model=$SHARED_MODEL  endpoint=$SUPERVISOR_ENDPOINT"

        print_status "[B1] Loading scenarios from CSV: $SCENARIOS_FROM"
        print_status "[B1]   categories=\"$SCENARIO_CATEGORIES\"  n_per_category=${N_PER_CATEGORY:-(all)}"

        # One run.py invocation handles all scenarios — model stays loaded.
        cd "$RASINGAN_DIR/final_pipeline"
        RUN_ARGS=(
            --max-turns "$MAX_TURNS"
            --output-dir "$SESSIONS_DIR"
            --therapist-model "$THERAPIST_MODEL"
            --shared-model "$SHARED_MODEL"
            --supervisor-endpoint "$SUPERVISOR_ENDPOINT"
            --quiet
        )
        RUN_ARGS+=(--scenarios-from "$SCENARIOS_FROM")
        if [ -n "$SCENARIO_CATEGORIES" ]; then
            # shellcheck disable=SC2206
            CAT_ARR=($SCENARIO_CATEGORIES)
            RUN_ARGS+=(--categories "${CAT_ARR[@]}")
        fi
        if [ -n "$N_PER_CATEGORY" ]; then
            RUN_ARGS+=(--n-per-category "$N_PER_CATEGORY")
        fi
        if [ -n "$N_SCENARIOS" ]; then
            RUN_ARGS+=(--n-scenarios "$N_SCENARIOS")
            print_status "[B1] Capping to first $N_SCENARIOS scenario(s)"
        fi
        if [ "$CONCURRENCY" -gt 1 ] 2>/dev/null; then
            RUN_ARGS+=(--concurrency "$CONCURRENCY")
            print_status "[B1] Running scenarios in parallel (concurrency=$CONCURRENCY)"
        fi
        CUDA_VISIBLE_DEVICES="$THERAPIST_GPU" "$CONDA_PY" run.py "${RUN_ARGS[@]}" || {
            print_warning "[B1] final_pipeline run.py exited with errors (continuing to scoring)"
        }
        print_success "[B1] final_pipeline sessions completed"

        # Stop the supervisor server now that B1 is done; B2/B3 are pure post-processing.
        cleanup_servers
        SUP_SERVER_PID=""
    else
        print_warning "[B1] Skipping session generation (using existing $SESSIONS_DIR)"
    fi

    if [ ! -d "$SESSIONS_DIR" ] || [ -z "$(ls -A "$SESSIONS_DIR" 2>/dev/null | grep '^session_')" ]; then
        print_warning "No session_*.json in $SESSIONS_DIR — skipping B2/B3"
    else
        cd "$SCRIPT_DIR"

        print_status "[B2] Computing CTRS from sessions…"
        "$CONDA_PY" score_ctrs.py --eval-root "$EVAL_ROOT" --model-name "$MODEL_NAME" \
            || print_warning "CTRS scoring encountered an issue (non-fatal)"
        print_success "[B2] CTRS scoring completed"

        print_status "[B3] Aggregating Information Retrieval (checklist coverage)…"
        "$CONDA_PY" score_information_retrieval.py --eval-root "$EVAL_ROOT" --model-name "$MODEL_NAME" \
            || print_warning "IR scoring encountered an issue (non-fatal)"
        print_success "[B3] Information Retrieval scoring completed"
    fi
else
    print_warning "Skipping PHASE B (multi-turn metrics)"
fi

# ============================================================================
# Results summary
# ============================================================================
print_status "========================================="
print_status "Evaluation Results Summary"
print_status "========================================="

if [ -f "$MODEL_DIR/care_loss.json" ]; then
    print_success "CARE Loss (L1/L2 vs Ground Truth):"
    "$CONDA_PY" -m json.tool "$MODEL_DIR/care_loss.json" | sed 's/^/  /'; echo ""
fi

if [ -f "$MODEL_DIR/nlp_metrics.json" ]; then
    print_success "NLP Metrics:"
    "$CONDA_PY" -m json.tool "$MODEL_DIR/nlp_metrics.json" | sed 's/^/  /'; echo ""
fi

if [ -f "$MODEL_DIR/ctrs_scores.json" ]; then
    print_success "CTRS Scores (from final_pipeline sessions):"
    "$CONDA_PY" -c "
import json
with open('$MODEL_DIR/ctrs_scores.json') as f:
    d = json.load(f)
a = d['aggregate']
print(f\"  Sessions scored: {d['n_sessions']}\")
print(f\"  Mean CTRS-P:                   {a['avg_CTRS_P']:.4f} (±{a['std_CTRS_P']:.4f})\")
print(f\"  Understanding:                 {a['avg_Understanding_score']:.4f}\")
print(f\"  Interpersonal Effectiveness:   {a['avg_Interpersonal_Effectiveness_score']:.4f}\")
print(f\"  Collaboration:                 {a['avg_Collaboration_score']:.4f}\")
print(f\"  Technical Appropriateness:     {a['avg_Technical_Appropriateness_score']:.4f}\")
by_scn = d.get('by_scenario_category', {}) or {}
if by_scn and not (len(by_scn) == 1 and '(uncategorized)' in by_scn):
    print('  By scenario category:')
    for cat, s in by_scn.items():
        print(f\"    [{cat}] n={s['n_sessions']}  CTRS-P={s['avg_CTRS_P']:.4f} (±{s['std_CTRS_P']:.4f})  \"
              f\"U={s['avg_Understanding_score']:.3f}  IE={s['avg_Interpersonal_Effectiveness_score']:.3f}  \"
              f\"C={s['avg_Collaboration_score']:.3f}  TA={s['avg_Technical_Appropriateness_score']:.3f}\")
"
    echo ""
fi

if [ -f "$MODEL_DIR/ir_score.json" ]; then
    print_success "Information Retrieval (Checklist Coverage from final_pipeline):"
    "$CONDA_PY" -c "
import json
with open('$MODEL_DIR/ir_score.json') as f:
    d = json.load(f)
print(f\"  Sessions: {d['n_sessions']}  |  Items: {d['total_items']}\")
print(f\"  IR score: {d['ir_score']:.2f}% (±{d['ir_score_std']:.2f})\")
for cat in d['category_coverage'].values():
    print(f\"    • {cat['name']}: {cat['mean_coverage_pct']:.2f}%\")
by_scn = d.get('by_scenario_category', {}) or {}
if by_scn and not (len(by_scn) == 1 and '(uncategorized)' in by_scn):
    print('  By scenario category:')
    for cat, s in by_scn.items():
        print(f\"    [{cat}] n={s['n_sessions']}  IR={s['ir_score']:.2f}% (±{s['ir_score_std']:.2f})\")
"
    echo ""
fi

print_success "All generated score files:"
find "$MODEL_DIR" -type f \( -name "*.csv" -o -name "*.json" \) | sed 's/^/  /'

print_status "========================================="
print_status "Evaluation Pipeline Completed Successfully!"
print_status "Results saved to: $MODEL_DIR"
print_status "========================================="
