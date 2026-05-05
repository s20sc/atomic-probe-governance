#!/bin/bash
# Auto-retry wrapper for cross-seed composition eval (this repo).
# Unlimited retries — each cell saves to disk immediately, so resumption is safe.

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
    log "=== Compose eval: attempt $attempt ==="

    start_time=$(date +%s)
    $PYTHON -u src/compose_cross_seed_eval.py "$@" \
        2>&1 | tee "results/composition/run_attempt${attempt}.log"
    exit_code=${PIPESTATUS[0]}
    elapsed=$(( $(date +%s) - start_time ))

    if [ $exit_code -eq 0 ]; then
        log "Compose eval: completed successfully (attempt $attempt, ${elapsed}s)"
        break
    fi

    log "Compose eval: crashed after ${elapsed}s (exit $exit_code, attempt $attempt)"

    if [ $elapsed -lt 30 ]; then
        consecutive_fast=$((consecutive_fast + 1))
        log "  → fast crash #$consecutive_fast (<30s)"
        if [ $consecutive_fast -ge 5 ]; then
            log "!!! 5 consecutive fast crashes — code bug likely, giving up !!!"
            exit 1
        fi
    else
        consecutive_fast=0
    fi

    log "Waiting 15s before retry..."
    sleep 15
done

log "=== COMPOSE EVAL COMPLETE ==="
