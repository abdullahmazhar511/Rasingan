#!/bin/bash
set -e

# Train CareModel on combined faith_data + PTSD datasets with curriculum learning

echo "========================================"
echo "CareModel Training: Combined Datasets"
echo "Curriculum Learning Approach"
echo "========================================"

python3 /home/umairai/faithfulness_emnlp/Rasingan/sft_training/train_caremodel_combined.py \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --data_dir /home/umairai/faith_data/dataset \
    --output_dir /home/umairai/faithfulness_emnlp/Rasingan/sft_training/results/caremodel_combined \
    --batch_size 32 \
    --phase1_epochs 3 \
    --phase2_epochs 7 \
    --learning_rate 1e-4 \
    --use_curriculum

echo "========================================"
echo "Training Complete!"
echo "Model saved to: /home/umairai/faithfulness_emnlp/Rasingan/sft_training/results/caremodel_combined"
echo "========================================"
