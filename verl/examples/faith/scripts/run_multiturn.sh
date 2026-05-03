#!/bin/bash
# Multi-turn therapist agent training script
# run on 8xH100 or adjust for your hardware
# make sure your current working directory is the root of the project

# Activate verl conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate verl

set -x

# Use CUDA 12.8 nvcc (system nvcc is 11.5 and doesn't support sm_90a)
export PATH=/usr/local/cuda-12.8/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.8

export CUDA_LAUNCH_BLOCKING=1
# export CUDA_VISIBLE_DEVICES=1
ulimit -n 65535


function now() {
    date '+%d-%H-%M'
}

EXPERIMENT_NAME="therapist_multiturn_$(now)"
PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/faith/scripts/config"
source "$PROJECT_DIR/examples/faith/scripts/api_key"
SERVER_PORT=29500  # port for reward model

SAVE_PATH="./checkpoints/${EXPERIMENT_NAME}"

# Use final_pipeline agent_1 model
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
RESUME_FROM_PATH=""
TRAIN_BATCH_SIZE=8
VAL_BATCH_SIZE=16
PPO_MINI_BATCH_SIZE=4
MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=512

# Training data from final_pipeline therapy conversations
FINAL_PIPELINE_DIR="$HOME/faithfulness_emnlp/Rasingan/final_pipeline"
DATA_DIR="$FINAL_PIPELINE_DIR/data/therapy_conversations"
TRAIN_FILES="$DATA_DIR/train.parquet"
VAL_FILES="$DATA_DIR/val.parquet"

# Generate training data if it doesn't exist
if [ ! -f "$TRAIN_FILES" ] || [ ! -f "$VAL_FILES" ]; then
    echo "Generating training data from final_pipeline..."
    cd "$FINAL_PIPELINE_DIR"
    python generate_verl_training_data.py \
        --output_dir "$DATA_DIR" \
        --num_samples_per_scenario 3 \
        --model "$MODEL"
    cd "$PROJECT_DIR"
fi

VAL_ONLY=False
RESUME=False

if [ "$VAL_ONLY" = True ]; then
    EXTRA_FLAGS="trainer.val_only=True trainer.val_before_train=True"
else
    EXTRA_FLAGS="trainer.val_only=False trainer.val_before_train=False"
fi

if [ "$RESUME" = True ]; then
    RESUME_FLAGS="trainer.resume_from_path=$RESUME_FROM_PATH trainer.resume_mode='resume_path'"
else
    RESUME_FLAGS="trainer.resume_mode='auto'"
fi

#hf login
if [ -n "$HF_TOKEN" ]; then
    export HF_TOKEN="$HF_TOKEN"
fi

#wandb login
if [ -n "$WANDB_API_KEY" ]; then
    wandb login --relogin "$WANDB_API_KEY"
    export WANDB_DIR="$SAVE_PATH/wandb"
fi

if [ ! -d "$SAVE_PATH" ]; then
    mkdir -p "$SAVE_PATH"
fi

ROLLOUT_SAVE_PATH="$SAVE_PATH/rollout"
if [ ! -d "$ROLLOUT_SAVE_PATH" ]; then
    mkdir -p "$ROLLOUT_SAVE_PATH"
fi

python3 -m verl.trainer.main_ppo \
    +server.port=$SERVER_PORT \
    +reward_model.max_concurrent=16 \
    +reward_model.max_rpm=32 \
    +reward_model.estimated_tokens_per_request=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
    actor_rollout_ref.rollout.agent.default_agent_loop='therapist_multiturn_agent' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager="care" \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='therapeutic_agents' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_epochs=3 \
    ${EXTRA_FLAGS} \
    ${RESUME_FLAGS} \
    trainer.validation_data_dir=${SAVE_PATH}/rollout \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.rollout_data_dir=${ROLLOUT_SAVE_PATH} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((4 * (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((4 * (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((4 * (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))) \
    data.train_files=$TRAIN_FILES \
    data.val_files=$VAL_FILES \
    data.patient_model.base_url="http://localhost:8001" \
    data.patient_model.model="meta-llama/Meta-Llama-3.1-8B-Instruct" \
    data.patient_model.api_key="EMPTY" \
    data.patient_model.max_tokens=256 \
    data.patient_model.temperature=0.7 \
    data.supervisor_model.base_url="http://localhost:8002" \
    data.supervisor_model.model="meta-llama/Meta-Llama-3.1-8B-Instruct" \
    data.supervisor_model.api_key="EMPTY" \
    hydra.run.dir=$SAVE_PATH/outputs | tee $SAVE_PATH/run.log

