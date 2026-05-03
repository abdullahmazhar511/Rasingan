#!/bin/bash

# Script to generate responses for a model and compute all evaluation scores

set -e  # Exit on error

# Activate verl conda environment
source /home/umairai/anaconda3/bin/activate verl

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RASINGAN_DIR="$(dirname "$SCRIPT_DIR")"
EVAL_ROOT="${EVAL_ROOT:-/home/umairai/faithfulness_emnlp/Rasingan/evaluation_pipeline}"
DATASET_PATH="${DATASET_PATH:-/home/umairai/faith_data/dataset/llm_test}"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Usage information
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

OPTIONS:
    -m, --model-name NAME        Model name (required - used as folder in EVAL_ROOT)
    -b, --base-model PATH        Base model path from HuggingFace (required)
    -a, --adapter PATH           Optional: Path to PEFT adapter
    -d, --dataset PATH           Dataset path (default: $DATASET_PATH)
    -e, --eval-root PATH         Evaluation root directory (default: $EVAL_ROOT)
    -s, --skip-generation        Skip generation, only run scoring
    -h, --help                   Show this help message

EXAMPLES:
    # Generate and score a new model
    $0 --model-name my-model --base-model meta-llama/Llama-3.2-1B

    # Generate and score with adapter
    $0 --model-name my-model-sft --base-model meta-llama/Llama-3.2-1B --adapter /path/to/adapter

    # Only score (skip generation)
    $0 --model-name my-model --skip-generation

EOF
    exit 1
}

# Default values
MODEL_NAME=""
BASE_MODEL=""
ADAPTER_PATH=""
SKIP_GENERATION=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        -b|--base-model)
            BASE_MODEL="$2"
            shift 2
            ;;
        -a|--adapter)
            ADAPTER_PATH="$2"
            shift 2
            ;;
        -d|--dataset)
            DATASET_PATH="$2"
            shift 2
            ;;
        -e|--eval-root)
            EVAL_ROOT="$2"
            shift 2
            ;;
        -s|--skip-generation)
            SKIP_GENERATION=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate required arguments
if [ -z "$MODEL_NAME" ]; then
    print_error "Model name is required"
    usage
fi

if [ "$SKIP_GENERATION" = false ] && [ -z "$BASE_MODEL" ]; then
    print_error "Base model is required (or use --skip-generation)"
    usage
fi

print_status "========================================="
print_status "Starting Evaluation Pipeline"
print_status "========================================="
print_status "Model Name: $MODEL_NAME"
print_status "Base Model: $BASE_MODEL"
[ ! -z "$ADAPTER_PATH" ] && print_status "Adapter: $ADAPTER_PATH"
print_status "Dataset Path: $DATASET_PATH"
print_status "Eval Root: $EVAL_ROOT"
print_status "Skip Generation: $SKIP_GENERATION"

# Create output directories
MODEL_DIR="$EVAL_ROOT/$MODEL_NAME"
RESPONSES_DIR="$MODEL_DIR/responses"
CARE_SCORES_DIR="$MODEL_DIR/care_scores"

mkdir -p "$RESPONSES_DIR" "$CARE_SCORES_DIR"
print_success "Created output directories"

# Export variables for generate.py and scoring scripts
export EVAL_ROOT DATASET_PATH

# Generate responses if not skipping
if [ "$SKIP_GENERATION" = false ]; then
    print_status "========================================="
    print_status "Step 1: Generating Responses"
    print_status "========================================="
    
    cd "$RASINGAN_DIR"
    
    python3 "$SCRIPT_DIR/generate.py" \
        --model-name "$MODEL_NAME" \
        --base-model "$BASE_MODEL" \
        --adapter-path "${ADAPTER_PATH:-null}" \
        --eval-root "$EVAL_ROOT" \
        --dataset-path "$DATASET_PATH" || {
        print_error "Generation failed"
        exit 1
    }
    
    print_success "Response generation completed"
else
    print_warning "Skipping generation (responses should already exist in $RESPONSES_DIR)"
fi

# Check if responses exist
if [ ! -d "$RESPONSES_DIR" ] || [ -z "$(ls -A "$RESPONSES_DIR" 2>/dev/null)" ]; then
    print_error "No responses found in $RESPONSES_DIR"
    exit 1
fi

# Run scoring scripts
print_status "========================================="
print_status "Step 2: Computing CARE Scores"
print_status "========================================="

cd "$SCRIPT_DIR"
python3 score_care.py --eval-root "$EVAL_ROOT" --model-name "$MODEL_NAME" || {
    print_warning "CARE scoring encountered an issue (non-fatal)"
}
print_success "CARE scoring completed"

print_status "========================================="
print_status "Step 3: Computing CARE Loss (L1/L2)"
print_status "========================================="

python3 score_care_loss.py --eval-root "$EVAL_ROOT" --model-name "$MODEL_NAME" || {
    print_warning "CARE loss computation encountered an issue (non-fatal)"
}
print_success "CARE loss computation completed"

print_status "========================================="
print_status "Step 4: Computing Conversation-Level CARE Metrics"
print_status "========================================="

python3 score_care_conv.py --eval-root "$EVAL_ROOT" --model-name "$MODEL_NAME" || {
    print_warning "Conversation CARE metrics encountered an issue (non-fatal)"
}
print_success "Conversation CARE metrics completed"

print_status "========================================="
print_status "Step 5: Computing CTRS (Clinical Therapeutic Response System) Scores"
print_status "========================================="

python3 score_ctrs.py --eval-root "$EVAL_ROOT" --model-name "$MODEL_NAME" || {
    print_warning "CTRS scoring encountered an issue (non-fatal)"
}
print_success "CTRS scoring completed"

# print_status "========================================="
# print_status "Step 6: Computing Coherence Scores"
# print_status "========================================="

# python3 score_coherence.py --eval-root "$EVAL_ROOT" --model-name "$MODEL_NAME" || {
#     print_warning "Coherence scoring encountered an issue (non-fatal)"
# }
# print_success "Coherence scoring completed"

print_status "========================================="
print_status "Step 7: Computing NLP Metrics"
print_status "========================================="

python3 score_nlp.py --eval-root "$EVAL_ROOT" --model-name "$MODEL_NAME" || {
    print_warning "NLP metrics computation encountered an issue (non-fatal)"
}
print_success "NLP metrics computation completed"

# Collect and display results
print_status "========================================="
print_status "Evaluation Results Summary"
print_status "========================================="

# Check for CARE loss metrics
if [ -f "$MODEL_DIR/care_loss.json" ]; then
    print_success "CARE Loss (L1/L2 vs Ground Truth):"
    python3 -m json.tool "$MODEL_DIR/care_loss.json" | sed 's/^/  /'
    echo ""
fi

# Check for conversation-level CARE metrics
if [ -f "$MODEL_DIR/care_conv_metrics.json" ]; then
    print_success "Conversation CARE Metrics:"
    python3 -c "
import json, sys
with open('$MODEL_DIR/care_conv_metrics.json') as f:
    d = json.load(f)
a = d['aggregate']
print(f\"  Mean CARE: {a['avg_mean_care']:.4f}\")
print(f\"  Mean UPR:  {a['avg_mean_upr']:.4f}\")
print(f\"  Mean REF:  {a['avg_mean_ref']:.4f}\")
print(f\"  Consistency (std): {a['avg_std_care']:.4f}\")
print(f\"  Min-turn CARE:  {a['avg_min_turn_care']:.4f}\")
print(f\"  Trajectory:  {a['avg_trajectory_slope']:+.4f}\")
"
    echo ""
fi

# Check for CTRS scores
if [ -f "$MODEL_DIR/ctrs_scores.json" ]; then
    print_success "CTRS Scores:"
    python3 -c "
import json
with open('$MODEL_DIR/ctrs_scores.json') as f:
    d = json.load(f)
a = d['aggregate']
print(f\"  Mean CTRS-P: {a['avg_CTRS_P']:.4f}\")
print(f\"  Understanding: {a['avg_Understanding_score']:.4f}\")
print(f\"  Interpersonal Effectiveness: {a['avg_Interpersonal_Effectiveness_score']:.4f}\")
print(f\"  Collaboration: {a['avg_Collaboration_score']:.4f}\")
print(f\"  Technical Appropriateness: {a['avg_Technical_Appropriateness_score']:.4f}\")
"
    echo ""
fi

# Check for coherence scores
if [ -f "$MODEL_DIR/coherence_score.csv" ]; then
    print_success "Coherence Scores:"
    head -20 "$MODEL_DIR/coherence_score.csv" | column -t -s',' | sed 's/^/  /'
    echo ""
fi

# Check for coherence metrics (mean UPR/REF)
if [ -f "$MODEL_DIR/coherence_metrics.json" ]; then
    print_success "Coherence Metrics (Mean):"
    python3 -m json.tool "$MODEL_DIR/coherence_metrics.json" | sed 's/^/  /'
    echo ""
fi

# Check for NLP metrics
if [ -f "$MODEL_DIR/nlp_metrics.json" ]; then
    print_success "NLP Metrics:"
    python3 -m json.tool "$MODEL_DIR/nlp_metrics.json" | sed 's/^/  /'
    echo ""
fi

# List all generated score files
print_success "All generated score files:"
find "$MODEL_DIR" -type f \( -name "*.csv" -o -name "*.json" \) | sed 's/^/  /'

print_status "========================================="
print_status "Evaluation Pipeline Completed Successfully!"
print_status "Results saved to: $MODEL_DIR"
print_status "========================================="
