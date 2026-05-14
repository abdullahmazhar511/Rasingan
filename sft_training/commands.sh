#train
#single gpu training
export CUDA_VISIBLE_DEVICES=0
python train.py \
    --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 \
    --output_dir results/Qwen3-4B-sft-respair \

#multi gpu training
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    train.py \
    --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 \
    --output_dir results/Qwen3-4B-sft-respair \
    --data_dir respair_mhcopilot_format \
    --max_length 768

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
    --adapter_path results/llama3.2-1b-sft-respair-new-3/checkpoint-1275 \
    --data_dir respair_mhcopilot_format \
    --context_window 6 \
    --batch_size 128 \
    --max_new_tokens 128