#!/bin/bash
# Multi-turn therapist agent PPO training.
#
# Wraps the whole flow:
#   1. Preprocess multiturn_reddit_data/splits/*.csv → parquet (skipped if up-to-date)
#   2. Auto-launch the shared Patient+Supervisor vLLM server via setup_external_models.sh
#   3. Run verl PPO with the therapist_multiturn_agent loop
#   4. Teardown the support server on exit (trap)
#
# Prereq: a CARE reward server must already be running at $CARE_SERVER_URL
#         (default http://127.0.0.1:8000). Start it with:
#             cd ../../../server && ./server.sh prod

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate verl
# Pin python so subshells use the conda env even if a venv earlier on PATH
# is re-activated by .bashrc (same fix as get_scores.sh / multi_turn_rl.sh).
if [ -n "${VIRTUAL_ENV:-}" ] && command -v deactivate &>/dev/null; then
    deactivate 2>/dev/null || true
fi
unset VIRTUAL_ENV
CONDA_PY="$CONDA_PREFIX/bin/python"

set -x
set -e

# CUDA 12.8 nvcc — required for flashinfer JIT-compile on H100/H200 (sm_90a).
if [ -x /usr/local/cuda-12.8/bin/nvcc ]; then
    export PATH=/usr/local/cuda-12.8/bin:$PATH
    export CUDA_HOME=/usr/local/cuda-12.8
fi

ulimit -n 65535

function now() { date '+%d-%H-%M'; }
EXPERIMENT_NAME="${EXP_NAME:-therapist_multiturn_$(now)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"            # verl/
RASINGAN_DIR="$(cd "$PROJECT_DIR/.." && pwd)"                # Rasingan/
FINAL_PIPELINE_DIR="$RASINGAN_DIR/final_pipeline"
REDDIT_DATA_DIR="$RASINGAN_DIR/multiturn_reddit_data"

# Make the agent loop's import of `shared_config` resolve.
export RASINGAN_FINAL_PIPELINE="$FINAL_PIPELINE_DIR"

if [ -f "$PROJECT_DIR/examples/faith/scripts/api_key" ]; then
    # shellcheck disable=SC1090
    source "$PROJECT_DIR/examples/faith/scripts/api_key"
fi

SERVER_PORT=29500
CARE_SERVER_URL="${CARE_SERVER_URL:-http://127.0.0.1:8000}"

# GPU partition: verl rollout AND the shared patient+supervisor server BOTH
# span both GPUs (verl via its 2-rank engine, shared via tensor-parallel-size=2).
# CARE reward server lives on GPU 0 too (~16 GB), so GPU 0 hosts THREE tenants;
# GPU 1 hosts two. Sizes tuned to fit all three on GPU 0:
#   - CARE server:                          ~16 GB on GPU 0
#   - shared TP=2  per GPU:  SUPPORT_GPU_MEM_UTIL=0.40   (~56 GB each)
#   - actor rollout per GPU: gpu_memory_utilization=0.40 (~56 GB each)
# GPU 0 total: 16 + 56 + 56 = 128 / 140 GB (~12 GB buffer)
# GPU 1 total:       56 + 56 = 112 / 140 GB (~28 GB buffer)
# Shared aggregate KV cache = 112 GB (still 2× the single-GPU layout).

# Shared Patient + Supervisor server config.
SHARED_MODEL="${SHARED_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
SHARED_PORT="${SHARED_PORT:-8001}"
# CSV of GPUs — comma count = tensor-parallel-size on the shared server.
SHARED_GPU="${SHARED_GPU:-0,1}"
SHARED_URL="http://127.0.0.1:${SHARED_PORT}"
SUPPORT_GPU_MEM_UTIL="${SUPPORT_GPU_MEM_UTIL:-0.40}"
# Bigger context so 20-turn transcripts don't overflow the supervisor.
SUPPORT_MAX_LEN="${SUPPORT_MAX_LEN:-16384}"

# Therapist (Agent 1) — the model being trained. MODEL_PATH / OUTPUT_PATH come
# from scripts/multi_turn_rl.sh; fall back to defaults.
MODEL="${MODEL_PATH:-$RASINGAN_DIR/sft_training/results/Qwen3-4B-sft-respair-new-3-merged}"
SAVE_PATH="${OUTPUT_PATH:-./checkpoints/${EXPERIMENT_NAME}}"

# Training knobs
TRAIN_BATCH_SIZE=8
VAL_BATCH_SIZE=16
PPO_MINI_BATCH_SIZE=4
MAX_PROMPT_LENGTH=2048
# Response budget needs to fit the *whole* multi-turn rollout (therapist + patient
# tokens both accumulate into this budget, just with different masks). With
# MAX_THERAPY_TURNS=20 we observed ~80-token therapist turns + ~250-token patient
# turns from the get_scores.sh sessions → ~6.6K tokens. 8192 leaves headroom.
MAX_RESPONSE_LENGTH=8192
MAX_THERAPY_TURNS=20

# ----- 1) Preprocess data ----------------------------------------------------
PARQUET_DIR="$PROJECT_DIR/examples/faith/data_multiturn"
TRAIN_PARQUET="$PARQUET_DIR/train.parquet"
VAL_PARQUET="$PARQUET_DIR/val.parquet"

if [ ! -f "$TRAIN_PARQUET" ] || [ ! -f "$VAL_PARQUET" ] \
   || [ "$REDDIT_DATA_DIR/splits/train.csv" -nt "$TRAIN_PARQUET" ]; then
    echo "[preprocess] (re)generating multiturn parquets…"
    "$CONDA_PY" "$PROJECT_DIR/examples/faith/data_preprocess/preprocess_multiturn.py"
fi

# ----- 2) Launch the shared external server ---------------------------------
# Idempotent: skip if already up.
SUPPORTING_STARTED=false
cleanup_support() {
    if [ "$SUPPORTING_STARTED" = true ]; then
        echo "[cleanup] stopping shared external server…"
        SUPERVISOR_PORT="$SHARED_PORT" PATIENT_PORT="$SHARED_PORT" \
            bash "$SCRIPT_DIR/setup_external_models.sh" --kill || true
    fi
}
trap cleanup_support EXIT INT TERM

if curl -sf --max-time 3 "$SHARED_URL/v1/models" -o /dev/null 2>/dev/null; then
    echo "[deps] shared server already up at $SHARED_URL — reusing"
else
    echo "[deps] launching shared Patient+Supervisor server via setup_external_models.sh…"
    SHARED=true \
    PATIENT_MODEL="$SHARED_MODEL" \
    SUPERVISOR_MODEL="$SHARED_MODEL" \
    PATIENT_PORT="$SHARED_PORT" \
    PATIENT_GPU="$SHARED_GPU" \
    GPU_MEM_UTIL="$SUPPORT_GPU_MEM_UTIL" \
    MAX_MODEL_LEN="$SUPPORT_MAX_LEN" \
        bash "$SCRIPT_DIR/setup_external_models.sh" --shared
    SUPPORTING_STARTED=true
fi

# ----- 3) Sanity-check CARE reward server is up -----------------------------
if curl -sf --max-time 3 "$CARE_SERVER_URL/health" -o /dev/null 2>/dev/null \
   || curl -sf --max-time 3 "$CARE_SERVER_URL/v1/models" -o /dev/null 2>/dev/null; then
    echo "[deps] CARE reward server at $CARE_SERVER_URL — ok"
else
    echo "[deps] WARNING: CARE reward server at $CARE_SERVER_URL is not responding."
    echo "       Start it with:  cd $RASINGAN_DIR/server && ./server.sh prod"
    echo "       Training will fail when it tries to compute rewards."
fi

# ----- 4) Training -----------------------------------------------------------

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

if [ -n "$HF_TOKEN" ]; then
    export HF_TOKEN="$HF_TOKEN"
fi
if [ -n "$WANDB_API_KEY" ]; then
    wandb login --relogin "$WANDB_API_KEY"
    export WANDB_DIR="$SAVE_PATH/wandb"
fi

mkdir -p "$SAVE_PATH"
ROLLOUT_SAVE_PATH="$SAVE_PATH/rollout"
mkdir -p "$ROLLOUT_SAVE_PATH"

"$CONDA_PY" -m verl.trainer.main_ppo \
    +server.port=$SERVER_PORT \
    +reward_model.max_concurrent=16 \
    +reward_model.max_rpm=32 \
    +reward_model.estimated_tokens_per_request=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
    +reward_model.care_server_url="$CARE_SERVER_URL" \
    actor_rollout_ref.rollout.agent.default_agent_loop='therapist_multiturn_agent' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    +data.max_therapy_turns=$MAX_THERAPY_TURNS \
    +data.session_type=counseling \
    +data.enable_supervisor_feedback=True \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.40 \
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
    data.train_files="$TRAIN_PARQUET" \
    data.val_files="$VAL_PARQUET" \
    +data.patient_model.base_url="$SHARED_URL" \
    +data.patient_model.model="$SHARED_MODEL" \
    +data.patient_model.api_key="EMPTY" \
    +data.patient_model.max_tokens=256 \
    +data.patient_model.temperature=0.7 \
    +data.supervisor_model.base_url="$SHARED_URL" \
    +data.supervisor_model.model="$SHARED_MODEL" \
    +data.supervisor_model.api_key="EMPTY" \
    +data.supervisor_model.max_tokens=400 \
    +data.supervisor_model.temperature=0.3 \
    hydra.run.dir=$SAVE_PATH/outputs | tee $SAVE_PATH/run.log
