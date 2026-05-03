MODEL_DIR="/home/umairai/faithfulness_emnlp/Rasingan/verl/checkpoints/llama3.2_sft_14-21-00/global_step_186"
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir $MODEL_DIR/actor \
    --target_dir $MODEL_DIR/converted