#!/bin/bash
# Auto-retry wrapper for T1_Pick multiseed training
# Goal: train seeds 7, 123, 2024 for 15 iterations each.
# Wall time estimate: ~9-12 hours total (T1 saturates fast).

set -o pipefail
cd "$(cd "$(dirname "$0")"/.. && pwd)"
PYTHON="${PYTHON:-python3}"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

attempt=0; consecutive_fast=0
while true; do
    attempt=$((attempt + 1))
    log "=== T1 multiseed: attempt $attempt ==="
    start_time=$(date +%s)

    $PYTHON -u src/train_t3_t4_multiseed.py \
        --tasks T1_Pick \
        --seeds 7 123 2024 \
        --iterations 15 \
        2>&1 | tee "results/training/t1_multiseed_attempt${attempt}.log"
    exit_code=${PIPESTATUS[0]}
    elapsed=$(( $(date +%s) - start_time ))

    if [ $exit_code -eq 0 ]; then
        log "T1 multiseed: completed (attempt $attempt, ${elapsed}s)"
        break
    fi
    log "T1 multiseed: crashed after ${elapsed}s (exit $exit_code)"

    if [ $elapsed -lt 60 ]; then
        consecutive_fast=$((consecutive_fast + 1))
        if [ $consecutive_fast -ge 5 ]; then
            log "!!! 5 consecutive fast crashes — abort"; exit 1
        fi
    else
        consecutive_fast=0
    fi
    log "Waiting 30s before retry..."
    sleep 30
done

log "=== Phase 1 (training) complete. Phase 2 (atomic eval) ==="
$PYTHON -u src/compute_atomic_baseline.py \
    --task T1_Pick \
    --out atomic_T1_Pick.json \
    --num-episodes 30 \
    2>&1 | tee results/composition/atomic_T1_Pick.log

log "=== ALL DONE ==="
