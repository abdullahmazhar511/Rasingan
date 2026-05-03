#train

#inference base
python inference_test.py \
    --base_model meta-llama/Llama-3.2-1B-Instruct \
    --data_dir /home/umairai/faith_data/dataset \
    --batch_size 16 \
    --max_new_tokens 1024

#inference trained
python inference_test.py \
    --base_model meta-llama/Llama-3.2-1B-Instruct \
    --adapter_path results/llama3.2-1b-sft/checkpoint-288 \
    --data_dir /home/umairai/faith_data/dataset \
    --batch_size 32 \
    --max_new_tokens 1024 