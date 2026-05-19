#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RASINGAN_ROOT="$(dirname "$SCRIPT_DIR")"
SFT_DIR="$RASINGAN_ROOT/sft_training"

# Defaults
CONDA_ENV_NAME="${CONDA_ENV_NAME:-verl}"
RUN_MODE="infer-trained"
SINGLE_GPU="0"
MULTI_GPUS="2"

BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507"
OUTPUT_DIR="results/Qwen3-4B-sft-respair"
DATA_DIR="respair_mhcopilot_format"
MAX_LENGTH="768"

INFER_CONTEXT_WINDOW="6"
INFER_BATCH_SIZE="128"
INFER_MAX_NEW_TOKENS="128"
ADAPTER_PATH="/home/asbahk/EMNLP_FINAL/Rasingan/sft_training/results/Qwen3-4B-sft-respair-new-3/checkpoint-425"

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Run modes:
  --run MODE                 Which command(s) to run.
                             MODE: single-train | multi-train | infer-base | infer-trained | all
                             Default: all

Environment:
  --conda-env NAME           Conda env name to activate (default: verl)

Training options:
  --single-gpu IDX           CUDA_VISIBLE_DEVICES for single-train (default: 0)
  --multi-gpus N             nproc_per_node for multi-train (default: 2)
  --base-model NAME          Base model for train/inference
  --output-dir PATH          Output dir for training checkpoints
  --data-dir PATH            Dataset directory (relative to sft_training)
  --max-length N             Max sequence length for training (default: 768)

Inference options:
  --context-window N         Context window for inference_test.py (default: 6)
  --infer-batch-size N       Batch size for inference_test.py (default: 128)
  --max-new-tokens N         Max new tokens for inference_test.py (default: 128)
  --adapter-path PATH        Adapter path for infer-trained

Misc:
  -h, --help                 Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run)
            RUN_MODE="$2"
            shift 2
            ;;
        --conda-env)
            CONDA_ENV_NAME="$2"
            shift 2
            ;;
        --single-gpu)
            SINGLE_GPU="$2"
            shift 2
            ;;
        --multi-gpus)
            MULTI_GPUS="$2"
            shift 2
            ;;
        --base-model)
            BASE_MODEL="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --max-length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        --context-window)
            INFER_CONTEXT_WINDOW="$2"
            shift 2
            ;;
        --infer-batch-size)
            INFER_BATCH_SIZE="$2"
            shift 2
            ;;
        --max-new-tokens)
            INFER_MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --adapter-path)
            ADAPTER_PATH="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

case "$RUN_MODE" in
    single-train|multi-train|infer-base|infer-trained|all)
        ;;
    *)
        echo "Invalid --run value: $RUN_MODE" >&2
        usage
        exit 1
        ;;
esac

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1090
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

cd "$SFT_DIR"

run_single_train() {
    export CUDA_VISIBLE_DEVICES="$SINGLE_GPU"
    python train.py \
        --model_name_or_path "$BASE_MODEL" \
        --output_dir "$OUTPUT_DIR"
}

run_multi_train() {
    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node="$MULTI_GPUS" \
        train.py \
        --model_name_or_path "$BASE_MODEL" \
        --output_dir "$OUTPUT_DIR" \
        --data_dir "$DATA_DIR" \
        --max_length "$MAX_LENGTH"
}

run_infer_base() {
    python inference_test.py \
        --base_model "$BASE_MODEL" \
        --data_dir "$DATA_DIR" \
        --context_window "$INFER_CONTEXT_WINDOW" \
        --batch_size "$INFER_BATCH_SIZE" \
        --max_new_tokens "$INFER_MAX_NEW_TOKENS"
}

run_infer_trained() {
    python inference_test.py \
        --base_model "$BASE_MODEL" \
        --adapter_path "$ADAPTER_PATH" \
        --data_dir "$DATA_DIR" \
        --context_window "$INFER_CONTEXT_WINDOW" \
        --batch_size "$INFER_BATCH_SIZE" \
        --max_new_tokens "$INFER_MAX_NEW_TOKENS"
}

case "$RUN_MODE" in
    single-train)
        run_single_train
        ;;
    multi-train)
        run_multi_train
        ;;
    infer-base)
        run_infer_base
        ;;
    infer-trained)
        run_infer_trained
        ;;
    all)
        run_single_train
        run_multi_train
        run_infer_base
        run_infer_trained
        ;;
esac



