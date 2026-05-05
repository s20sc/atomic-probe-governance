#!/bin/bash
# Auto-retry wrapper for T3+T4 multi-seed training (~40 hours).
# Per-iteration checkpoints + per-result-JSON saves give crash-resilient resume.

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
    log "=== train_t3_t4: attempt $attempt ==="
    start_time=$(date +%s)

    $PYTHON -u src/train_t3_t4_multiseed.py \
        2>&1 | tee "results/training/run_attempt${attempt}.log"
    exit_code=${PIPESTATUS[0]}
    elapsed=$(( $(date +%s) - start_time ))

    if [ $exit_code -eq 0 ]; then
        log "train_t3_t4: completed successfully (attempt $attempt, ${elapsed}s)"
        break
    fi
    log "train_t3_t4: crashed after ${elapsed}s (exit $exit_code)"

    if [ $elapsed -lt 60 ]; then
        consecutive_fast=$((consecutive_fast + 1))
        if [ $consecutive_fast -ge 5 ]; then
            log "!!! 5 consecutive fast crashes — code bug, giving up !!!"
            exit 1
        fi
    else
        consecutive_fast=0
    fi

    log "Waiting 30s before retry..."
    sleep 30
done

log "=== TRAINING COMPLETE ==="

# Quick summary
$PYTHON - <<'PY'
import json, os
out = 'results/training'
print('\n=== TRAINING SUMMARY ===')
for f in sorted(os.listdir(out)):
    if not f.endswith('.json'): continue
    p = os.path.join(out, f)
    try:
        d = json.load(open(p))
        n = len(d.get('iterations', []))
        meta = d.get('_meta', {})
        wall = meta.get('wall_seconds', 0)
        print(f'  {f}: {n}/20 iters, {wall/3600:.1f}h')
    except Exception as e:
        print(f'  {f}: error {e}')
PY
