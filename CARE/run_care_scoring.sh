#!/usr/bin/env bash
# Score the 4 latest-checkpoint SFT prediction CSVs with the trained CARE classifier.
# Outputs go to sft_CARE_output/<short_model_name>.csv
#
# Usage: bash run_care_scoring.sh
set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # CARE/
REPO_ROOT="$(dirname "$SCRIPT_DIR")"                         # EMNLP_FINAL/
OUT_DIR="$SCRIPT_DIR/sft_CARE_output"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Each entry is "<short_name>|<predictions_csv_path>"
SFT_DIR="$REPO_ROOT/Rasingan/sft_training"
declare -a JOBS=(
    "Llama-3.2-1B-Instruct|$SFT_DIR/_home_asbahk_EMNLP_FINAL_Rasingan_sft_training_results_Llama-3.2-1B-Instruct-sft-respair-20260516_122639_checkpoint-1275_test_predictions.csv"
    "Qwen3-4B-Instruct-2507|$SFT_DIR/_home_asbahk_EMNLP_FINAL_Rasingan_sft_training_results_Qwen3-4B-Instruct-2507-sft-respair-20260516_122639_checkpoint-1275_test_predictions.csv"
    "gemma-3-4b-it|$SFT_DIR/_home_asbahk_EMNLP_FINAL_Rasingan_sft_training_results_gemma-3-4b-it_checkpoint-1275_test_predictions.csv"
    "Ministral-8B-Instruct-2410|$SFT_DIR/_home_asbahk_EMNLP_FINAL_Rasingan_sft_training_results_mistralai_Ministral-8B-Instruct-2410_checkpoint-1275_test_predictions.csv"
)

declare -a STATUSES

for entry in "${JOBS[@]}"; do
    NAME="${entry%%|*}"
    CSV="${entry##*|}"
    OUT_CSV="$OUT_DIR/${NAME}.csv"
    LOG="$LOG_DIR/${NAME}.log"

    echo "================================================================"
    echo "[$(date +%H:%M:%S)] CARE scoring: $NAME"
    echo "  predictions: $CSV"
    echo "  output csv:  $OUT_CSV"
    echo "  log:         $LOG"
    echo "================================================================"

    if [ ! -f "$CSV" ]; then
        echo "[SKIP] predictions CSV not found: $CSV"
        STATUSES+=("SKIP  $NAME  (missing CSV)")
        continue
    fi

    START_TS=$(date +%s)
    set +e
    python "$SCRIPT_DIR/score_with_care.py" \
        --predictions_csv "$CSV" \
        --output_csv "$OUT_CSV" \
        --batch_size 8 \
        > "$LOG" 2>&1
    STATUS=$?
    set -e
    ELAPSED=$(( $(date +%s) - START_TS ))
    HMS=$(printf '%02d:%02d:%02d' $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60)))

    if [ $STATUS -eq 0 ]; then
        echo "[OK]   $NAME  (took $HMS)"
        # Print the per-trait summary tail from the log
        grep -E "^\s+(NJ|WE|RA|AL|RF|SA|AVG):" "$LOG" | sed 's/^/    /'
        STATUSES+=("OK    $NAME  $HMS")
    else
        echo "[FAIL] $NAME  exit=$STATUS  (after $HMS)"
        echo "    --- last 15 lines of $LOG ---"
        tail -n 15 "$LOG" | sed 's/^/    /'
        STATUSES+=("FAIL  $NAME  exit=$STATUS")
    fi
    echo ""
done

echo "================================================================"
echo "All jobs finished: $(date)"
echo "Summary:"
for s in "${STATUSES[@]}"; do
    echo "  $s"
done
echo "Outputs at: $OUT_DIR/"
echo "================================================================"
