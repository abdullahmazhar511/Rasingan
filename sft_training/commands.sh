#train

#inference base
python inference_test.py \
    --base_model Qwen/Qwen3-4B-Instruct-2507 \
    --data_dir respair_mhcopilot_format \
    --context_window 6 \
    --batch_size 128 \
    --max_new_tokens 128

#inference trained
python inference_test.py \
    --base_model Qwen/Qwen3-4B-Instruct-2507 \
    --adapter_path results/llama3.2-1b-sft-respair-new-1/checkpoint-426 \
    --data_dir respair_mhcopilot_format \
    --context_window 6 \
    --batch_size 128 \
    --max_new_tokens 128