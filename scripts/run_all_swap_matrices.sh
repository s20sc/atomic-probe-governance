#!/bin/bash
# Sequentially run all 4 phase-swap matrices (reach, grasp already done, lift, place)
# Each matrix: 4x4 cells, 30 episodes/cell, ~25 min total.

set -o pipefail
cd "$(cd "$(dirname "$0")"/.. && pwd)"
PYTHON="${PYTHON:-python3}"
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

for phase in reach lift place; do
    log "===== Starting swap=$phase matrix ====="
    attempt=0
    while true; do
        attempt=$((attempt + 1))
        $PYTHON -u src/compose_cross_seed_eval.py \
            --num-episodes 30 --swap-phase "$phase" \
            2>&1 | tee "results/composition/run_${phase}_attempt${attempt}.log"
        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            log "swap=$phase done"
            break
        fi
        log "swap=$phase crashed (attempt $attempt), retrying in 15s..."
        sleep 15
        if [ $attempt -ge 10 ]; then
            log "!!! swap=$phase: giving up after 10 attempts !!!"
            break
        fi
    done
done

log "=== ALL SWAP MATRICES COMPLETE ==="

# Summary across all 4 matrices
$PYTHON - <<'PY'
import json, os, numpy as np
out = 'results/composition'
print('\n=== SUMMARY: 4 swap-phase matrices ===\n')
print(f'{"swap_phase":>12s} {"diag_mean":>11s} {"off_mean":>10s} {"delta":>8s} {"diag_std":>10s} {"off_std":>10s}')
for phase in ['reach', 'grasp', 'lift', 'place']:
    p = os.path.join(out, f'cross_seed_{phase}_swap.json')
    if not os.path.exists(p):
        print(f'  {phase}: missing')
        continue
    d = json.load(open(p))
    diag, off = [], []
    for v in d.get('cells', {}).values():
        (diag if v.get('is_diagonal') else off).append(v['success_rate'])
    if diag and off:
        print(f'{phase:>12s} {np.mean(diag):>10.1f}% {np.mean(off):>9.1f}% '
              f'{np.mean(off)-np.mean(diag):>+7.1f} {np.std(diag):>9.1f} {np.std(off):>9.1f}')
PY
