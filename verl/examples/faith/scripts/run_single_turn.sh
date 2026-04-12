# run on 8xH100
# make sure your current working directory is the root of the project

set -x
export CUDA_LAUNCH_BLOCKING=1
ulimit -n 65535


function now() {
    date '+%d-%H-%M'
}

EXPERIMENT_NAME="qwen2.5-3b_baseline_$(now)"
PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/faith/scripts/config"
WANDB_API_KEY="your_wandb_api_key_here"  # Replace with your actual WandB API key
HF_TOKEN="your_huggingface_token_here"  # Replace with your actual Hugging Face token
SERVER_PORT=29500  # port for reward model

SAVE_PATH="./checkpoints/${EXPERIMENT_NAME}"

MODEL="google/gemma-3-4b-it"
RESUME_FROM_PATH=""
TRAIN_BATCH_SIZE=256
VAL_BATCH_SIZE=256
PPO_MINI_BATCH_SIZE=256
MAX_PROMPT_LENGTH=1024
MAX_RESPONSE_LENGTH=1024

TRAIN_FILES="$HOME/data/gsm8k/train.parquet"
VAL_FILES="$HOME/data/gsm8k/test.parquet"

VAL_ONLY=False
RESUME=False

if [ "$VAL_ONLY" = True ]; then
    EXTRA_FLAGS="trainer.val_only=True trainer.val_before_train=True"
else
    EXTRA_FLAGS="trainer.val_only=False trainer.val_before_train=True"
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
    wandb login --key "$WANDB_API_KEY"
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
    +reward_model.max_rpm=1000 \
    +reward_model.estimated_tokens_per_request=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
    actor_rollout_ref.rollout.agent.default_agent_loop='naive' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.use_dynamic_bcz=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager="care_judge" \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='rasingan' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=20 \
    trainer.total_epochs=3 \
    ${EXTRA_FLAGS} \
    ${RESUME_FLAGS} \
    trainer.validation_data_dir=${SAVE_PATH}/rollout \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.rollout_data_dir=${ROLLOUT_SAVE_PATH} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4*((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4*((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4*((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) \
    data.train_files=$TRAIN_FILES \
    data.val_files=$VAL_FILES \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/gsm8k_tool_config.yaml" \
    trainer.total_epochs=15 \
    hydra.run.dir=$SAVE_PATH/outputs | tee $SAVE_PATH/run.log

