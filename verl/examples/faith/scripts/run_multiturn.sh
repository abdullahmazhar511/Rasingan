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

# Note: do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here —
# vLLM's memory pool (used by verl's internal rollout vLLM) asserts against
# it, see pytorch/pytorch#147851. Rely on the conservative
# gpu_memory_utilization values below instead.

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

source "$PROJECT_DIR/examples/faith/scripts/api_key"

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
# Qwen3.5-4B = GDN/linear-attention 4B; same architecture as 3.5-9B but
# ~half the memory and ~2x faster decode. Use 3.5-9B for stronger judgments
# if quality matters more than speed.
SHARED_MODEL="${SHARED_MODEL:-Qwen/Qwen3.5-4B}"
SHARED_PORT="${SHARED_PORT:-8001}"
# CSV of GPUs — comma count = tensor-parallel-size on the shared server.
SHARED_GPU="${SHARED_GPU:-0,1}"
SHARED_URL="http://127.0.0.1:${SHARED_PORT}"
# With 4B at TP=2, each rank holds ~4 GB of weights. 0.30 = ~42 GB per rank
# → ~38 GB KV cache after weights — more concurrent decoding capacity than
# the 9B setup allowed. (For 9B use 0.25 → ~26 GB KV, or you'll OOM verl
# backward.)
# 0.22 → 0.32: SUPPORT_MAX_LEN=12288 makes each KV slot 1.5× larger than at
# 8192. 0.32 of 140 GiB ≈ 45 GiB total → ~21 concurrent KV slots after
# model weights + activations, modestly above the ~19 of 0.30 while
# keeping trainer-peak (~60 GiB) + supervisor combined safely under the
# watchdog's 120 GiB kill threshold.
SUPPORT_GPU_MEM_UTIL="${SUPPORT_GPU_MEM_UTIL:-0.32}"
# Match the rollout vLLM's max_model_len (= MAX_PROMPT_LENGTH +
# MAX_RESPONSE_LENGTH = 24576). The supervisor's prompt also includes the
# running transcript, so any budget less than the rollout's would 400
# silently when the transcript grows past it (mid-session degradation).
# 24576 → 14336: training logs from the previous run showed the THERAPIST
# (separate vLLM) hitting prompt_ids≈14.5K at long-session turn ~13. The
# PATIENT prompt on this shared server is smaller (no supervisor feedback
# injection) but still reaches ~10–11K tokens in worst-case long sessions
# with long Reddit posts. 14336 sits comfortably above that worst case with
# ~2.5K headroom for chat-template variance, while still freeing ~1.7× KV
# cache per slot vs the original 24576. Empirically safe: the previous
# trainer-side rollout vLLM ran at this exact max_model_len without
# context-overflow rejections on the same prompt distribution.
# Bump toward 16384–24576 if 400 context-overflow errors appear in
# train_multiturn.log.
SUPPORT_MAX_LEN="${SUPPORT_MAX_LEN:-14336}"

# Therapist (Agent 1) — the model being trained. MODEL_PATH / OUTPUT_PATH come
# from scripts/multi_turn_rl.sh; fall back to defaults.
MODEL="${MODEL_PATH:-$RASINGAN_DIR/sft_training/results/Qwen3-4B-sft-respair-new-3-merged}"
SAVE_PATH="${OUTPUT_PATH:-./checkpoints/${EXPERIMENT_NAME}}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"   # empty → auto-pick latest in SAVE_PATH
# Training knobs — tuned for speed after smoketest4 showed 451s/step @ 84%
# generation. We keep rollout.n=4 and train_batch_size=8 per spec; speed wins
# come from inside the agent loop:
#   * Parallelize the two supervisor calls per turn (asyncio.gather)
#   * Skip supervisor on the last turn (feedback would be unused)
#   * Trim per-call decode budgets: supervisor feedback 400→200,
#     extraction 500→256, patient 256→192
#   * SUPPORT_GPU_MEM_UTIL 0.30 → 0.40 — more KV cache for the shared
#     patient+supervisor server, allowing higher continuous-batch parallelism
#     across the 32 concurrent rollout sessions.
#   * MAX_RESPONSE_LENGTH 8192 → 12288 — smoketest4 hit clip_ratio=1.0 at 8192;
#     12288 gives clean fit for 20-turn rollouts with shorter patient turns.
#   * kl_loss_coef=0.01, lr=1e-6, total_epochs=2, save/test_freq=40 — unchanged.
TRAIN_BATCH_SIZE=8
VAL_BATCH_SIZE=16
PPO_MINI_BATCH_SIZE=4
# MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH = rollout vLLM's max_model_len.
# Per-turn prompt = entire accumulated conversation history; if it exceeds
# max_model_len, vLLM crashes with `ValueError: max_tokens must be at least
# 1, got -N`. The therapist agent loop guards against this by *early-stopping*
# the session when the next turn's prompt would overflow (see
# `therapist_agent_loop.py:_generate_therapist_response`). It does NOT
# truncate accumulated history — that would corrupt the PPO trajectory
# (training-time forward would not match the rollout context).
#
# 2048 + 12288 = 14336 — matches the working 360s/step baseline. Sessions
# typically early-stop around turn 12-14 when the accumulated history hits
# the budget. Increase MAX_PROMPT_LENGTH if you want most sessions to
# complete the full MAX_THERAPY_TURNS=20 (cost: ~3× longer wall-clock per
# step due to vLLM throughput collapse at long context).
MAX_PROMPT_LENGTH=2048
# Paired with THERAPIST_PER_TURN_MAX_TOKENS=256 in the agent loop, this gives
# 20 × 256 = 5120 worst-case + headroom for any over-shoot. 8192 is plenty.
MAX_RESPONSE_LENGTH=8192
MAX_THERAPY_TURNS="${MAX_THERAPY_TURNS:-20}"

# ----- 1) Preprocess data ----------------------------------------------------
PARQUET_DIR="$PROJECT_DIR/examples/faith/data_multiturn"
TRAIN_PARQUET="$PARQUET_DIR/train.parquet"
VAL_PARQUET="$PARQUET_DIR/val.parquet"

if [ ! -f "$TRAIN_PARQUET" ] || [ ! -f "$VAL_PARQUET" ] \
   || [ "$REDDIT_DATA_DIR/splits/train.csv" -nt "$TRAIN_PARQUET" ]; then
    echo "[preprocess] (re)generating multiturn parquets…"
    PREP_ARGS=()
    [ -n "${TRAIN_LIMIT:-}" ] && [ "$TRAIN_LIMIT" != "0" ] && PREP_ARGS+=(--train-limit "$TRAIN_LIMIT")
    [ -n "${VAL_LIMIT:-}"   ] && [ "$VAL_LIMIT"   != "0" ] && PREP_ARGS+=(--val-limit   "$VAL_LIMIT")
    "$CONDA_PY" "$PROJECT_DIR/examples/faith/data_preprocess/preprocess_multiturn.py" "${PREP_ARGS[@]}"
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

# Reusable launcher so the supervisor health watchdog can re-invoke the exact
# same env when restarting after a crash. All knobs come from outer-scope vars.
launch_shared_server() {
    SHARED=true \
    PATIENT_MODEL="$SHARED_MODEL" \
    SUPERVISOR_MODEL="$SHARED_MODEL" \
    PATIENT_PORT="$SHARED_PORT" \
    PATIENT_GPU="$SHARED_GPU" \
    GPU_MEM_UTIL="$SUPPORT_GPU_MEM_UTIL" \
    MAX_MODEL_LEN="$SUPPORT_MAX_LEN" \
    MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-98304}" \
    SERVER_PYTHON="${SERVER_PYTHON:-/home/asbahk/vllmenv/bin/python}" \
        bash "$SCRIPT_DIR/setup_external_models.sh" --shared
}

if curl -sf --max-time 3 "$SHARED_URL/v1/models" -o /dev/null 2>/dev/null; then
    echo "[deps] shared server already up at $SHARED_URL — reusing"
else
    echo "[deps] launching shared Patient+Supervisor server via setup_external_models.sh…"
    # Bump MAX_NUM_BATCHED_TOKENS 32768 → 65536 so the supervisor can pack ~2×
    # more concurrent requests per scheduler step. The supervisor was the gen-
    # phase bottleneck (70% SM util while trainer sat idle at 0-12%); doubling
    # the batched-token budget directly increases supervisor throughput.
    #
    # SERVER_PYTHON forced to vllmenv (vLLM 0.20 + flashinfer-jit-cache). We do
    # NOT rely on setup_external_models.sh's auto-pick because we want this
    # decision visible in the run config.
    launch_shared_server
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

if [ "$VAL_ONLY" = True ]; then
    EXTRA_FLAGS="trainer.val_only=True trainer.val_before_train=True"
else
    EXTRA_FLAGS="trainer.val_only=False trainer.val_before_train=False"
fi

# Resume contract:
#   RESUME_FROM_PATH empty  → resume_mode=auto  (verl picks latest checkpoint in
#                                                trainer.default_local_dir, or
#                                                starts fresh if none exist)
#   RESUME_FROM_PATH set    → resume_mode=resume_path with that exact path
if [ -n "${RESUME_FROM_PATH:-}" ]; then
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

mkdir -p "$SAVE_PATH"
ROLLOUT_SAVE_PATH="$SAVE_PATH/rollout"
mkdir -p "$ROLLOUT_SAVE_PATH"

# === GPU memory watchdog =====================================================
# Poll nvidia-smi for GPU memory used by our training process tree. If any GPU
# exceeds GPU_MEM_LIMIT_GB, SIGTERM (then SIGKILL) the trainer. Protects against
# OOM crashes from over-aggressive rollout/activation memory and from neighbor
# processes (e.g., another user's job) growing into our reservation.
GPU_MEM_LIMIT_GB="${GPU_MEM_LIMIT_GB:-120}"
GPU_WATCHDOG_INTERVAL_S="${GPU_WATCHDOG_INTERVAL_S:-10}"
GPU_WATCHDOG_GRACE_S="${GPU_WATCHDOG_GRACE_S:-180}"   # ignore startup ramp-up

gpu_watchdog() {
    # Silence bash xtrace inside the watchdog — the parent script has `set -x`
    # so the poll loop would otherwise emit hundreds of `+ sleep 10` lines per
    # minute into the main training log. Real warnings (limit exceeded) still
    # print via the explicit `echo` below.
    set +x
    local trainer_pid="$1"
    local limit_gb="$2"
    local interval_s="$3"
    local grace_s="$4"

    sleep "$grace_s"

    while kill -0 "$trainer_pid" 2>/dev/null; do
        # Collect all PIDs in our trainer's process tree.
        local our_pids
        our_pids=$( { pgrep -g "$trainer_pid" 2>/dev/null; \
                       pstree -p "$trainer_pid" 2>/dev/null \
                         | grep -oE '\([0-9]+\)' | tr -d '()'; \
                       echo "$trainer_pid"; } | sort -u | tr '\n' ' ')

        if [ -z "$our_pids" ]; then
            sleep "$interval_s"; continue
        fi

        # Per-GPU MiB used by our PIDs. Awk indexes used_memory by gpu_uuid,
        # filtered to PIDs in our tree.
        local max_used_mib
        max_used_mib=$(nvidia-smi \
            --query-compute-apps=pid,gpu_uuid,used_memory \
            --format=csv,noheader,nounits 2>/dev/null \
            | awk -F', *' -v pids="$our_pids" '
                BEGIN {
                    n=split(pids, arr, " ");
                    for (i=1; i<=n; i++) ok[arr[i]] = 1
                }
                ($1 in ok) { sum[$2] += $3 }
                END {
                    max = 0
                    for (g in sum) if (sum[g] > max) max = sum[g]
                    print max + 0
                }
            ')

        local max_used_gb=$(( max_used_mib / 1024 ))

        if [ "$max_used_gb" -gt "$limit_gb" ]; then
            echo "[gpu-watchdog] !! Trainer GPU memory exceeded ${limit_gb} GiB (peak ${max_used_gb} GiB on the busiest GPU). Killing PID $trainer_pid." >&2
            kill -TERM "$trainer_pid" 2>/dev/null
            sleep 10
            kill -KILL "$trainer_pid" 2>/dev/null
            return
        fi
        sleep "$interval_s"
    done
}

# === Supervisor health watchdog ============================================
# Polls the shared (patient+supervisor) vLLM server. When it fails N consecutive
# health checks, kills any stale supervisor processes and re-launches the
# server via setup_external_models.sh. The agent loop's ExternalModelClient
# has exponential-backoff retry built in, so it will keep waiting + retrying
# while we restart the server — no rollout corruption from a transient
# supervisor crash.
SUP_WATCHDOG_INTERVAL_S="${SUP_WATCHDOG_INTERVAL_S:-15}"
SUP_WATCHDOG_GRACE_S="${SUP_WATCHDOG_GRACE_S:-180}"     # ignore startup ramp-up
SUP_WATCHDOG_FAILURES_BEFORE_RESTART="${SUP_WATCHDOG_FAILURES_BEFORE_RESTART:-3}"
SUP_WATCHDOG_RESTART_GRACE_S="${SUP_WATCHDOG_RESTART_GRACE_S:-300}"   # wait for re-ready

supervisor_watchdog() {
    set +x   # silence bash xtrace
    local trainer_pid="$1"
    local url="$2"
    local interval_s="$3"
    local grace_s="$4"
    local fail_thresh="$5"

    # Start a low-frequency telemetry recorder so we have a timeline of GPU /
    # supervisor state RIGHT BEFORE the next crash. Writes 1 sample/30s to a
    # CSV ring file. On crash, the most recent ~10 samples capture the freeze.
    local telemetry_csv="$SAVE_PATH/supervisor_telemetry.csv"
    echo "ts,gpu0_mem_mib,gpu1_mem_mib,gpu0_util,gpu1_util,sup_pid_count,health_ok" > "$telemetry_csv"
    (
        while kill -0 "$trainer_pid" 2>/dev/null; do
            local now t row
            now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
            t=$(nvidia-smi --query-gpu=memory.used,utilization.gpu \
                --format=csv,noheader,nounits 2>/dev/null \
                | tr '\n' ',' | tr -d ' ')
            local sup_pids
            sup_pids=$(pgrep -cf "vllm.entrypoints.openai.api_server.*--port $SHARED_PORT" 2>/dev/null)
            sup_pids="${sup_pids:-0}"
            local health="0"
            if curl -sf --max-time 2 "$url/v1/models" -o /dev/null 2>/dev/null; then
                health="1"
            fi
            row="$now,$t$sup_pids,$health"
            echo "$row" >> "$telemetry_csv"
            sleep 30
        done
    ) &
    local TELEMETRY_PID=$!
    # Make sure the inner telemetry process dies when the watchdog exits.
    trap 'kill "$TELEMETRY_PID" 2>/dev/null' EXIT

    sleep "$grace_s"

    local consec_fail=0
    while kill -0 "$trainer_pid" 2>/dev/null; do
        if curl -sf --max-time 3 "$url/v1/models" -o /dev/null 2>/dev/null; then
            consec_fail=0
        else
            consec_fail=$(( consec_fail + 1 ))
            echo "[sup-watchdog] health check failed (${consec_fail}/${fail_thresh}) for $url" >&2
        fi

        if [ "$consec_fail" -ge "$fail_thresh" ]; then
            local crash_ts
            crash_ts=$(date -u +%Y%m%dT%H%M%SZ)
            local diag_dir="$SAVE_PATH/crash_diagnostics/$crash_ts"
            mkdir -p "$diag_dir"
            echo "[sup-watchdog] !! Supervisor down for $((consec_fail * interval_s))s. Capturing diagnostics to $diag_dir then restarting…" >&2

            # === Pre-restart diagnostic capture =============================
            # Capture as much as possible BEFORE the kill, so we can post-mortem
            # the exact state that triggered the hang.

            # 1. Current GPU state (memory per process, utilization)
            nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total,utilization.gpu,utilization.memory,temperature.gpu,power.draw \
                --format=csv > "$diag_dir/nvidia-smi-gpus.csv" 2>&1 || true
            nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid,name \
                --format=csv > "$diag_dir/nvidia-smi-procs.csv" 2>&1 || true

            # 2. Per-process SM / memory-bw utilization (one sample)
            nvidia-smi pmon -c 1 -s u > "$diag_dir/nvidia-smi-pmon.txt" 2>&1 || true

            # 3. Supervisor process states + kernel stacks
            local sup_pids
            sup_pids=$(pgrep -f "vllm.entrypoints.openai.api_server.*--port $SHARED_PORT" 2>/dev/null)
            sup_pids+=" $(pgrep -f 'VLLM::Worker_TP' 2>/dev/null)"
            for pid in $sup_pids; do
                [ -z "$pid" ] && continue
                ps -p "$pid" -o pid,ppid,state,etime,pcpu,pmem,rss,vsz,cmd --no-headers \
                    >> "$diag_dir/supervisor-ps.txt" 2>/dev/null || true
                # Kernel stack of the (potentially hung) worker — shows whether
                # it's spinning in NCCL all-reduce, GPU op, mutex, etc.
                cat "/proc/$pid/stack" 2>/dev/null \
                    > "$diag_dir/kernel-stack-pid-$pid.txt" || true
                # Wait-channel (current syscall the process is blocked on)
                cat "/proc/$pid/wchan" 2>/dev/null \
                    >> "$diag_dir/supervisor-wchan.txt" || true
                echo " ^ wchan for pid $pid" >> "$diag_dir/supervisor-wchan.txt" 2>/dev/null
            done

            # 4. Python stack traces of TP workers + API server. py-spy is a
            #    standalone Rust binary (NOT a python module — `python -m py_spy`
            #    will not work). We locate the binary next to SERVER_PYTHON, or
            #    fall back to PATH. This is THE useful trace — tells us exactly
            #    which Python frame the worker is stuck in.
            local sup_python="${SERVER_PYTHON:-/home/asbahk/vllmenv/bin/python}"
            local pyspy_bin="$(dirname "$sup_python")/py-spy"
            if [ ! -x "$pyspy_bin" ]; then
                pyspy_bin="$(command -v py-spy 2>/dev/null || true)"
            fi
            if [ ! -x "$pyspy_bin" ]; then
                "$sup_python" -m pip install --quiet py-spy 2>/dev/null || true
                pyspy_bin="$(dirname "$sup_python")/py-spy"
            fi
            for pid in $sup_pids; do
                [ -z "$pid" ] && continue
                if [ -d "/proc/$pid" ] && [ -x "$pyspy_bin" ]; then
                    "$pyspy_bin" dump --pid "$pid" --nonblocking \
                        > "$diag_dir/py-spy-dump-pid-$pid.txt" 2>&1 || true
                fi
            done

            # 5. Last 200 lines of the supervisor's own server log.
            tail -n 200 "$LOG_DIR/shared_vllm_${SHARED_PORT}.log" 2>/dev/null \
                > "$diag_dir/supervisor-server-log-tail.txt" || true

            # 6. Recent agent-loop side warnings (last 60 lines worth of API errors)
            grep -E "WARNING.*supervisor|WARNING.*Patient|Connection|disconnected" \
                "$SAVE_PATH/run.log" 2>/dev/null | tail -60 \
                > "$diag_dir/agent-loop-recent-warnings.txt" || true

            # 7. Summary marker
            {
                echo "Crash detected at: $crash_ts"
                echo "Trainer PID:       $trainer_pid"
                echo "Supervisor URL:    $url"
                echo "Consec failures:   $consec_fail (× ${interval_s}s polling)"
                echo "Supervisor PIDs:   $sup_pids"
            } > "$diag_dir/SUMMARY.txt"

            echo "[sup-watchdog]    diagnostics captured to $diag_dir" >&2

            # === Restart ===================================================
            # Kill any stale supervisor processes (api_server + TP workers).
            bash "$SCRIPT_DIR/setup_external_models.sh" --kill >/dev/null 2>&1 || true
            # Also catch stragglers vllm leaks via process group.
            pkill -9 -f "vllm.entrypoints.openai.api_server.*--port $SHARED_PORT" 2>/dev/null || true
            sleep 5
            # Re-launch with the same env knobs as the initial boot.
            launch_shared_server || {
                echo "[sup-watchdog] !! launch_shared_server failed; will retry on next check" >&2
            }
            # Wait for re-readiness (bounded). If never comes back, give up
            # this restart round; next cycle will retry.
            local deadline=$(( $(date +%s) + SUP_WATCHDOG_RESTART_GRACE_S ))
            while [ "$(date +%s)" -lt "$deadline" ]; do
                if curl -sf --max-time 3 "$url/v1/models" -o /dev/null 2>/dev/null; then
                    echo "[sup-watchdog] Supervisor recovered." >&2
                    break
                fi
                sleep 10
            done
            consec_fail=0
        fi
        sleep "$interval_s"
    done
}

# Background the trainer so we can run the watchdog alongside it. We still
# tee to run.log via process substitution (bash-only feature; relies on the
# script's #!/bin/bash shebang).
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
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR:-2e-6} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${LR_WARMUP_STEPS:-10} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF:-0.005} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF:-0.001} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE:-0.7} \
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_N:-1} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=${PARAM_OFFLOAD:-True} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OPTIMIZER_OFFLOAD:-True} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL:-0.30} \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${REF_PARAM_OFFLOAD:-True} \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager="care" \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='rasingan' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=${SAVE_FREQ:-20} \
    trainer.test_freq=${TEST_FREQ:-20} \
    trainer.total_epochs=${TOTAL_EPOCHS:-4} \
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS:-null} \
    ${EXTRA_FLAGS} \
    ${RESUME_FLAGS} \
    trainer.validation_data_dir=${SAVE_PATH}/rollout \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.rollout_data_dir=${ROLLOUT_SAVE_PATH} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((2 * (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((2 * (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((2 * (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))) \
    data.train_files="$TRAIN_PARQUET" \
    data.val_files="$VAL_PARQUET" \
    +data.patient_model.base_url="$SHARED_URL" \
    +data.patient_model.model="$SHARED_MODEL" \
    +data.patient_model.api_key="EMPTY" \
    +data.patient_model.max_tokens=192 \
    +data.patient_model.temperature=0.7 \
    +data.supervisor_model.base_url="$SHARED_URL" \
    +data.supervisor_model.model="$SHARED_MODEL" \
    +data.supervisor_model.api_key="EMPTY" \
    +data.supervisor_model.max_tokens=256 \
    +data.supervisor_model.temperature=0.2 \
    hydra.run.dir=$SAVE_PATH/outputs \
    > >(tee "$SAVE_PATH/run.log") 2>&1 &
TRAINER_PID=$!
echo "[run_multiturn] Trainer PID=$TRAINER_PID; gpu watchdog limit=${GPU_MEM_LIMIT_GB} GiB (poll ${GPU_WATCHDOG_INTERVAL_S}s, grace ${GPU_WATCHDOG_GRACE_S}s)"

GPU_WATCHDOG_LOG="${GPU_WATCHDOG_LOG:-$SAVE_PATH/gpu_watchdog.log}"
echo "[run_multiturn] GPU watchdog log: $GPU_WATCHDOG_LOG"
gpu_watchdog "$TRAINER_PID" "$GPU_MEM_LIMIT_GB" "$GPU_WATCHDOG_INTERVAL_S" "$GPU_WATCHDOG_GRACE_S" \
    > "$GPU_WATCHDOG_LOG" 2>&1 &
WATCHDOG_PID=$!

SUP_WATCHDOG_LOG="${SUP_WATCHDOG_LOG:-$SAVE_PATH/supervisor_watchdog.log}"
echo "[run_multiturn] Supervisor watchdog log: $SUP_WATCHDOG_LOG  (restart after ${SUP_WATCHDOG_FAILURES_BEFORE_RESTART}× ${SUP_WATCHDOG_INTERVAL_S}s failures)"
supervisor_watchdog "$TRAINER_PID" "$SHARED_URL" "$SUP_WATCHDOG_INTERVAL_S" "$SUP_WATCHDOG_GRACE_S" "$SUP_WATCHDOG_FAILURES_BEFORE_RESTART" \
    > "$SUP_WATCHDOG_LOG" 2>&1 &
SUP_WATCHDOG_PID=$!

# Make sure watchdogs + trainer + supervisor die with the script (even on
# Ctrl-C / failures). cleanup_support kills the supervisor server we started.
# Order matters: kill watchdogs first so they don't try to "rescue" a dying
# supervisor; then trainer (SIGTERM, then SIGKILL); then the supervisor.
_full_cleanup() {
    kill -TERM "$WATCHDOG_PID" "$SUP_WATCHDOG_PID" 2>/dev/null || true
    kill -TERM "$TRAINER_PID" 2>/dev/null || true
    # Wait briefly for graceful shutdown; then SIGKILL anything still alive.
    for _ in 1 2 3 4 5; do
        kill -0 "$TRAINER_PID" 2>/dev/null || break
        sleep 1
    done
    kill -KILL "$TRAINER_PID" 2>/dev/null || true
    cleanup_support
    # Ray leaks workers sometimes — sweep them by name as a backstop.
    pkill -9 -f "ray::WorkerDict" 2>/dev/null || true
    pkill -9 -f "ray::AgentLoopWorker" 2>/dev/null || true
}
trap _full_cleanup EXIT INT TERM

wait "$TRAINER_PID"
TRAINER_EXIT=$?
kill "$WATCHDOG_PID" "$SUP_WATCHDOG_PID" 2>/dev/null
exit "$TRAINER_EXIT"
