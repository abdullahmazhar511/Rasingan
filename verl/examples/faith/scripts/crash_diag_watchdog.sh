#!/usr/bin/env bash
# Standalone crash-diagnostic watchdog.
# Polls the supervisor health endpoint; when it goes unresponsive,
# immediately captures py-spy dumps + kernel stacks + nvidia-smi BEFORE
# the main run_multiturn.sh watchdog kills and relaunches the server.
#
# This runs INDEPENDENTLY of run_multiturn.sh so a fix to its
# diagnostic-capture code can take effect mid-run without restarting
# training.
#
# Safe to run multiple times — diagnostics go in timestamped subdirs.
#
# Usage:
#   nohup ./crash_diag_watchdog.sh > crash_diag_watchdog.log 2>&1 &

set -u
set +e

SUPERVISOR_URL="${SUPERVISOR_URL:-http://127.0.0.1:8001}"
PYSPY_BIN="${PYSPY_BIN:-/home/asbahk/vllmenv/bin/py-spy}"
DIAG_ROOT="${DIAG_ROOT:-/home/asbahk/EMNLP_FINAL/Rasingan/verl/checkpoints/therapist_multiturn_final_v3/crash_diagnostics_external}"
SUP_LOG="${SUP_LOG:-/home/asbahk/EMNLP_FINAL/Rasingan/verl/examples/faith/scripts/logs/external_models/shared_vllm_8001.log}"
POLL_INTERVAL_S="${POLL_INTERVAL_S:-10}"
FAIL_THRESHOLD="${FAIL_THRESHOLD:-2}"   # capture earlier than main watchdog (3)
COOLDOWN_S="${COOLDOWN_S:-120}"          # don't re-capture within 2 min

mkdir -p "$DIAG_ROOT"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

find_supervisor_pids() {
    # API server, EngineCore, and TP workers — all live under the same
    # `vllm.entrypoints.openai.api_server ... --port 8001` cmdline tree.
    pgrep -af "vllm.entrypoints.openai.api_server" 2>/dev/null \
        | grep -E -- "--port[= ]8001" \
        | awk '{print $1}'
    # Also include EngineCore + Worker_TP children (different cmdline, same PGID)
    local parent
    parent="$(pgrep -af "vllm.entrypoints.openai.api_server" 2>/dev/null \
        | grep -E -- "--port[= ]8001" | awk '{print $1}' | head -1)"
    if [ -n "$parent" ]; then
        pgrep -P "$parent" 2>/dev/null || true
        # And grandchildren (workers spawn from EngineCore)
        for c in $(pgrep -P "$parent" 2>/dev/null); do
            pgrep -P "$c" 2>/dev/null || true
        done
    fi
}

capture_diagnostics() {
    local ts="$1"
    local diag="$DIAG_ROOT/$ts"
    mkdir -p "$diag"
    log "!! Capturing crash diagnostics to $diag"

    # 1. nvidia-smi snapshot
    nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total,utilization.gpu,utilization.memory,temperature.gpu,power.draw \
        --format=csv > "$diag/nvidia-smi-gpus.csv" 2>&1 || true
    nvidia-smi pmon -c 1 > "$diag/nvidia-smi-pmon.txt" 2>&1 || true
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv \
        > "$diag/nvidia-smi-procs.csv" 2>&1 || true

    # 2. Process list
    local pids
    pids="$(find_supervisor_pids | sort -u)"
    {
        echo "Discovered supervisor PIDs:"
        echo "$pids"
        echo ""
        echo "ps -F for those PIDs:"
        for p in $pids; do
            ps -F -p "$p" 2>/dev/null || true
        done
    } > "$diag/supervisor-ps.txt"

    # 3. Kernel stacks + wait-channel for each PID
    for p in $pids; do
        [ -z "$p" ] && continue
        cat "/proc/$p/stack" 2>/dev/null > "$diag/kernel-stack-pid-$p.txt" || true
        {
            cat "/proc/$p/wchan" 2>/dev/null
            echo " ^ wchan for pid $p"
        } >> "$diag/wchan.txt" || true
    done

    # 4. py-spy dumps — THE useful trace. Uses --nonblocking so we don't
    #    stall a process that might still be in a recoverable state.
    if [ -x "$PYSPY_BIN" ]; then
        for p in $pids; do
            [ -z "$p" ] && continue
            [ -d "/proc/$p" ] || continue
            timeout 30 "$PYSPY_BIN" dump --pid "$p" --nonblocking \
                > "$diag/py-spy-dump-pid-$p.txt" 2>&1 || true
        done
    else
        echo "py-spy binary not found at $PYSPY_BIN" > "$diag/py-spy-MISSING.txt"
    fi

    # 5. Tail of supervisor log
    tail -n 400 "$SUP_LOG" 2>/dev/null > "$diag/supervisor-log-tail.txt" || true

    # 6. SUMMARY
    {
        echo "Crash detected at: $ts"
        echo "Supervisor URL:    $SUPERVISOR_URL"
        echo "Fail threshold:    $FAIL_THRESHOLD (× ${POLL_INTERVAL_S}s polling)"
        echo "PIDs captured:"
        echo "$pids"
    } > "$diag/SUMMARY.txt"

    log "   diagnostics captured (pids: $(echo "$pids" | tr '\n' ' '))"
}

log "[crash-diag-watchdog] starting; url=$SUPERVISOR_URL diag_root=$DIAG_ROOT"
log "[crash-diag-watchdog] py-spy: $PYSPY_BIN ($(${PYSPY_BIN} --version 2>/dev/null || echo MISSING))"

fails=0
last_capture_ts=0

while true; do
    if curl -sf -m 5 "${SUPERVISOR_URL}/v1/models" >/dev/null 2>&1; then
        if [ "$fails" -gt 0 ]; then
            log "[crash-diag-watchdog] supervisor recovered after $fails failed polls"
        fi
        fails=0
    else
        fails=$((fails + 1))
        log "[crash-diag-watchdog] health poll failed ($fails/$FAIL_THRESHOLD)"
        if [ "$fails" -ge "$FAIL_THRESHOLD" ]; then
            now=$(date +%s)
            if [ $((now - last_capture_ts)) -ge "$COOLDOWN_S" ]; then
                capture_diagnostics "$(date -u +%Y%m%dT%H%M%SZ)"
                last_capture_ts=$now
            else
                log "[crash-diag-watchdog]   cooldown active, skipping capture"
            fi
            # Don't reset fails — let main watchdog handle the restart;
            # we'll just stop re-capturing until COOLDOWN_S elapses or
            # supervisor recovers (which resets fails to 0).
        fi
    fi
    sleep "$POLL_INTERVAL_S"
done
