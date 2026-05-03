#!/bin/bash
set -e

# Complete pipeline for retraining CareModel on combined datasets
# Step 1: Data preparation (already done)
# Step 2: Train with curriculum learning
# Step 3: Evaluate on both domains
# Step 4: Compare with baseline

echo "============================================================================"
echo "CAREMODEL RETRAINING PIPELINE"
echo "Combining faith_data and PTSD with curriculum learning"
echo "============================================================================"

DATASETS_DIR="/home/umairai/faith_data/dataset"
OUTPUT_DIR="/home/umairai/faithfulness_emnlp/Rasingan/sft_training/results/caremodel_combined"
LOG_DIR="${OUTPUT_DIR}/logs"

mkdir -p "$LOG_DIR"

# Check if datasets exist
echo ""
echo "Checking data files..."
for file in combined_train.csv combined_val.csv combined_test.csv curriculum_phase1.csv curriculum_phase2.csv; do
    if [ -f "$DATASETS_DIR/$file" ]; then
        echo "  ✓ $file ($(wc -l < "$DATASETS_DIR/$file") lines)"
    else
        echo "  ✗ MISSING: $file"
        exit 1
    fi
done

echo ""
echo "============================================================================"
echo "Step 1: TRAINING with Curriculum Learning"
echo "============================================================================"
echo "  Phase 1 (3 epochs): High-quality examples"
echo "  Phase 2 (7 epochs): All data"
echo ""

python3 /home/umairai/faithfulness_emnlp/Rasingan/sft_training/train_caremodel_combined.py \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --data_dir "$DATASETS_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 32 \
    --phase1_epochs 3 \
    --phase2_epochs 7 \
    --learning_rate 1e-4 \
    --use_curriculum 2>&1 | tee "$LOG_DIR/training.log"

echo ""
echo "============================================================================"
echo "Step 2: EVALUATION on both domains"
echo "============================================================================"
echo ""

python3 /home/umairai/faithfulness_emnlp/Rasingan/sft_training/eval_caremodel_combined.py \
    --model_dir "$OUTPUT_DIR" \
    --base_model meta-llama/Llama-3.2-1B-Instruct \
    --eval_dir /home/umairai/faithfulness_emnlp/Rasingan/evaluation_pipeline \
    --output_dir "$OUTPUT_DIR" 2>&1 | tee "$LOG_DIR/evaluation.log"

echo ""
echo "============================================================================"
echo "TRAINING COMPLETE"
echo "============================================================================"
echo ""
echo "Outputs saved to: $OUTPUT_DIR"
echo "  - Logs: $LOG_DIR"
echo "  - Model weights: $OUTPUT_DIR/adapter_model.bin"
echo "  - Evaluation results: $OUTPUT_DIR/retrained_caremodel_eval.json"
echo ""
echo "Next steps:"
echo "  1. Review evaluation results to confirm improvement"
echo "  2. Use new model in evaluation pipeline:"
echo "     - Update get_scores.sh to point to: $OUTPUT_DIR"
echo "  3. Retrain SFT models with improved CARE evaluator (optional)"
echo ""
