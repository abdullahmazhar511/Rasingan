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

set -e

CONDA_ENV_NAME="${CONDA_ENV_NAME:-verl}"
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1090
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RASINGAN_DIR="$(dirname "$SCRIPT_DIR")"
FINAL_PIPELINE_DIR="$RASINGAN_DIR/final_pipeline"
EVAL_ROOT="${EVAL_ROOT:-$RASINGAN_DIR/evaluation_pipeline}"
DATASET_PATH="${DATASET_PATH:-$RASINGAN_DIR/respair_mhcopilot_format}"
CONTEXT_WINDOW="${CONTEXT_WINDOW:-6}"
MAX_TURNS="${MAX_TURNS:-8}"
SCENARIOS="${SCENARIOS:-}"  # space-separated; empty = all from list_scenarios

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
    -a, --adapter PATH           Optional: path to PEFT adapter
    -d, --dataset PATH           Single-turn dataset (default: $DATASET_PATH)
    -c, --context-window N       Context window for generate.py (default: $CONTEXT_WINDOW)
    -e, --eval-root PATH         Evaluation root directory (default: $EVAL_ROOT)
    -s, --skip-generation        Phase A: skip generate.py (use existing responses/)
        --skip-single-turn       Skip Phase A entirely (no CARE / NLP)
        --skip-sessions          Phase B: skip running final_pipeline (use existing sessions/)
        --skip-multi-turn        Skip Phase B entirely (no CTRS / IR)
        --scenarios "A B C"      Space-separated scenarios for Phase B (default: all)
        --max-turns N            Max turns per simulated session (default: $MAX_TURNS)
    -h, --help                   Show this help

ENV OVERRIDES:
    EVAL_ROOT, DATASET_PATH, CONTEXT_WINDOW, MAX_TURNS, SCENARIOS

EOF
    exit 1
}

# Defaults
MODEL_NAME="qwen"
BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_PATH="/home/asbahk/EMNLP_FINAL/Rasingan/sft_training/results/Qwen3-4B-sft-respair-new-3/checkpoint-425"
SKIP_GENERATION=true
SKIP_SINGLE_TURN=false
SKIP_SESSIONS=false
SKIP_MULTI_TURN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model-name)      MODEL_NAME="$2";   shift 2 ;;
        -b|--base-model)      BASE_MODEL="$2";   shift 2 ;;
        -a|--adapter)         ADAPTER_PATH="$2"; shift 2 ;;
        -d|--dataset)         DATASET_PATH="$2"; shift 2 ;;
        -c|--context-window)  CONTEXT_WINDOW="$2"; shift 2 ;;
        -e|--eval-root)       EVAL_ROOT="$2";    shift 2 ;;
        -s|--skip-generation) SKIP_GENERATION=true; shift ;;
        --skip-single-turn)   SKIP_SINGLE_TURN=true; shift ;;
        --skip-sessions)      SKIP_SESSIONS=true; shift ;;
        --skip-multi-turn)    SKIP_MULTI_TURN=true; shift ;;
        --scenarios)          SCENARIOS="$2";    shift 2 ;;
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
[ -n "$ADAPTER_PATH" ] && print_status "Adapter:           $ADAPTER_PATH"
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
        python3 "$SCRIPT_DIR/generate.py" \
            --model-name "$MODEL_NAME" \
            --base-model "$BASE_MODEL" \
            --adapter-path "${ADAPTER_PATH:-null}" \
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

    print_status "[A2] Computing CARE scores…"
    python3 score_care.py --eval-root "$EVAL_ROOT" --model-name "$MODEL_NAME" \
        || print_warning "CARE scoring encountered an issue (non-fatal)"
    print_success "[A2] CARE scoring completed"

    print_status "[A3] Computing CARE Loss (L1/L2)…"
    python3 score_care_loss.py --eval-root "$EVAL_ROOT" --model-name "$MODEL_NAME" \
        || print_warning "CARE loss computation encountered an issue (non-fatal)"
    print_success "[A3] CARE loss computation completed"

    print_status "[A4] Computing NLP metrics…"
    python3 score_nlp.py --eval-root "$EVAL_ROOT" --model-name "$MODEL_NAME" \
        || print_warning "NLP metrics computation encountered an issue (non-fatal)"
    print_success "[A4] NLP metrics computation completed"
else
    print_warning "Skipping PHASE A (single-turn metrics)"
fi

# ============================================================================
# PHASE B — Multi-turn final_pipeline metrics
# ============================================================================
if [ "$SKIP_MULTI_TURN" = false ]; then
    print_status "========================================="
    print_status "PHASE B — Multi-turn final_pipeline metrics"
    print_status "========================================="

    # B1: Run final_pipeline therapy sessions
    if [ "$SKIP_SESSIONS" = false ]; then
        print_status "[B1] Running final_pipeline therapy sessions → $SESSIONS_DIR"

        if [ -z "$SCENARIOS" ]; then
            # Use every scenario exposed by reddit_posts.list_scenarios()
            SCENARIOS="$(cd "$FINAL_PIPELINE_DIR" && python3 -c 'from reddit_posts import list_scenarios; print(" ".join(list_scenarios()))')"
            print_status "[B1] Using all scenarios: $SCENARIOS"
        else
            print_status "[B1] Using scenarios: $SCENARIOS"
        fi

        cd "$FINAL_PIPELINE_DIR"
        for scenario in $SCENARIOS; do
            print_status "[B1]   scenario=$scenario  max_turns=$MAX_TURNS"
            python3 run.py \
                --scenario "$scenario" \
                --max-turns "$MAX_TURNS" \
                --output-dir "$SESSIONS_DIR" \
                --quiet || {
                print_warning "[B1]   scenario '$scenario' failed (continuing)"
                continue
            }
        done
        print_success "[B1] final_pipeline sessions completed"
    else
        print_warning "[B1] Skipping session generation (using existing $SESSIONS_DIR)"
    fi

    if [ ! -d "$SESSIONS_DIR" ] || [ -z "$(ls -A "$SESSIONS_DIR" 2>/dev/null | grep '^session_')" ]; then
        print_warning "No session_*.json in $SESSIONS_DIR — skipping B2/B3"
    else
        cd "$SCRIPT_DIR"

        print_status "[B2] Computing CTRS from sessions…"
        python3 score_ctrs.py --eval-root "$EVAL_ROOT" --model-name "$MODEL_NAME" \
            || print_warning "CTRS scoring encountered an issue (non-fatal)"
        print_success "[B2] CTRS scoring completed"

        print_status "[B3] Aggregating Information Retrieval (checklist coverage)…"
        python3 score_information_retrieval.py --eval-root "$EVAL_ROOT" --model-name "$MODEL_NAME" \
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
    python3 -m json.tool "$MODEL_DIR/care_loss.json" | sed 's/^/  /'; echo ""
fi

if [ -f "$MODEL_DIR/nlp_metrics.json" ]; then
    print_success "NLP Metrics:"
    python3 -m json.tool "$MODEL_DIR/nlp_metrics.json" | sed 's/^/  /'; echo ""
fi

if [ -f "$MODEL_DIR/ctrs_scores.json" ]; then
    print_success "CTRS Scores (from final_pipeline sessions):"
    python3 -c "
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
"
    echo ""
fi

if [ -f "$MODEL_DIR/ir_score.json" ]; then
    print_success "Information Retrieval (Checklist Coverage from final_pipeline):"
    python3 -c "
import json
with open('$MODEL_DIR/ir_score.json') as f:
    d = json.load(f)
print(f\"  Sessions: {d['n_sessions']}  |  Items: {d['total_items']}\")
print(f\"  IR score: {d['ir_score']:.2f}% (±{d['ir_score_std']:.2f})\")
for cat in d['category_coverage'].values():
    print(f\"    • {cat['name']}: {cat['mean_coverage_pct']:.2f}%\")
"
    echo ""
fi

print_success "All generated score files:"
find "$MODEL_DIR" -type f \( -name "*.csv" -o -name "*.json" \) | sed 's/^/  /'

print_status "========================================="
print_status "Evaluation Pipeline Completed Successfully!"
print_status "Results saved to: $MODEL_DIR"
print_status "========================================="
