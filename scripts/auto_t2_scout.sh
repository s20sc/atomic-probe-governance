#!/bin/bash
# Auto-retry wrapper for T2_Place scout training (~10h).
# Crash-safe: per-iter saves + intermediate snapshots persist across restarts.

set -o pipefail
cd "$(cd "$(dirname "$0")"/.. && pwd)"
PYTHON="${PYTHON:-python3}"
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

attempt=0
consecutive_fast=0

while true; do
    attempt=$((attempt + 1))
    log "=== T2 scout: attempt $attempt ==="
    start_time=$(date +%s)

    $PYTHON -u src/train_t2_scout.py 2>&1 | tee "results/training/t2_scout_attempt${attempt}.log"
    exit_code=${PIPESTATUS[0]}
    elapsed=$(( $(date +%s) - start_time ))

    if [ $exit_code -eq 0 ]; then
        log "T2 scout: completed (attempt $attempt, ${elapsed}s)"
        break
    fi

    log "T2 scout: crashed after ${elapsed}s (exit $exit_code)"

    if [ $elapsed -lt 60 ]; then
        consecutive_fast=$((consecutive_fast + 1))
        if [ $consecutive_fast -ge 5 ]; then
            log "!!! 5 consecutive fast crashes — code bug, abort !!!"
            exit 1
        fi
    else
        consecutive_fast=0
    fi

    log "Waiting 30s before retry..."
    sleep 30
done

log "=== Phase 1 complete. Running Phase 2 (Goldilocks eval) ==="
$PYTHON -u src/check_atomic_sr_for_ckpt.py --task T2_Place 2>&1 | tee results/composition/t2_goldilocks_eval.log
