#train

#inference base
python inference_test.py \
    --base_model google/gemma-2-2b-it \
    --data_dir /home/umairai/faithfulness_emnlp/Rasingan/sft_training/respair_mhcopilot_format \
    --batch_size 128 \
    --max_new_tokens 128

#inference trained
python inference_test.py \
    --base_model meta-llama/Llama-3.2-1B-Instruct \
    --adapter_path results/llama3.2-1b-sft-respair-4/checkpoint-712 \
    --data_dir /home/umairai/faithfulness_emnlp/Rasingan/sft_training/respair_mhcopilot_format \
    --batch_size 64 \
    --max_new_tokens 256 