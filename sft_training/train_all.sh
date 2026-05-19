#!/usr/bin/env bash
# Sequentially train multiple models. If one fails, log it and move on.
#
# Usage (inside a screen session):
#   screen -S sft_all
#   export HF_TOKEN=hf_xxx
#   export WANDB_API_KEY=xxx
#   bash train_all.sh
#   # Detach: Ctrl-A then D    Reattach: screen -r sft_all
#
# Per-run logs are also written to training_logs/*.log so you can tail them
# from any shell without attaching to the screen.
set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [ -z "${HF_TOKEN:-}" ]; then
    echo "WARNING: HF_TOKEN not set. Gated models (Llama, Mistral) will fail to download."
fi
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "WARNING: WANDB_API_KEY not set. wandb.login() will hang or fail."
fi

LOG_DIR="$SCRIPT_DIR/training_logs"
mkdir -p "$LOG_DIR"
STAMP=$(date +"%Y%m%d_%H%M%S")
SUMMARY="$LOG_DIR/summary_${STAMP}.txt"

# ---- Models and per-model overrides (parallel arrays) ----
MODELS=(
    "meta-llama/Llama-3.2-1B-Instruct"
    "Qwen/Qwen3-4B-Instruct-2507"
    "mistralai/Ministral-3-8B-Instruct-2512"
)
BATCH_SIZES=(16 16 16)    # 140GB GPU — 8B Mistral safely fits bs=16 at max_length=768
MAX_LENGTHS=(768 768 768)
EPOCHS=(3 3 3)
# ----------------------------------------------------------

{
    echo "Training run started: $(date)"
    echo "Host: $(hostname)   CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "Models: ${#MODELS[@]}"
    echo ""
} | tee "$SUMMARY"

declare -a STATUSES

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    BS="${BATCH_SIZES[$i]}"
    ML="${MAX_LENGTHS[$i]}"
    EP="${EPOCHS[$i]}"
    SHORT="$(basename "$MODEL")"
    OUT="results/${SHORT}-sft-respair-${STAMP}"
    LOG="$LOG_DIR/${SHORT}_${STAMP}.log"

    {
        echo "================================================================"
        echo "[$(date +%H:%M:%S)] [$((i+1))/${#MODELS[@]}] Training: $MODEL"
        echo "  batch_size=$BS  max_length=$ML  epochs=$EP"
        echo "  output:    $OUT"
        echo "  log file:  $LOG"
        echo "================================================================"
    } | tee -a "$SUMMARY"

    START_TS=$(date +%s)

    set +e
    python train.py \
        --model_name_or_path "$MODEL" \
        --output_dir "$OUT" \
        --data_dir respair_mhcopilot_format \
        --batch_size "$BS" \
        --max_length "$ML" \
        --epochs "$EP" \
        > "$LOG" 2>&1
    STATUS=$?
    set -e

    ELAPSED=$(( $(date +%s) - START_TS ))
    HMS=$(printf '%02d:%02d:%02d' $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60)))

    if [ $STATUS -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] [OK]   $MODEL  (took $HMS)" | tee -a "$SUMMARY"
        STATUSES+=("OK    $MODEL  $HMS")
    else
        echo "[$(date +%H:%M:%S)] [FAIL] $MODEL  (exit=$STATUS, after $HMS) — continuing" | tee -a "$SUMMARY"
        echo "    --- last 25 lines of $LOG ---" | tee -a "$SUMMARY"
        tail -n 25 "$LOG" | sed 's/^/    /' | tee -a "$SUMMARY"
        STATUSES+=("FAIL  $MODEL  exit=$STATUS")
    fi
    echo "" | tee -a "$SUMMARY"
done

{
    echo "================================================================"
    echo "All runs finished: $(date)"
    echo "Final status:"
    for s in "${STATUSES[@]}"; do
        echo "  $s"
    done
    echo ""
    echo "Per-run logs: $LOG_DIR/"
    echo "Summary:      $SUMMARY"
    echo "================================================================"
} | tee -a "$SUMMARY"
