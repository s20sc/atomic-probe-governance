#!/bin/bash
# Sequential runner for §6.1 follow-up experiments:
#   1. Atomic-skill baseline (16 ECMs × 30 episodes) — ~25 min
#   2. Behavioral distance (no rollouts beyond probe) — ~5 min
#   3. Paired cross-seed for all 4 phases — ~95 min
# Total: ~2 hours. All resumable, all save per-cell.

set -o pipefail
cd "$(cd "$(dirname "$0")"/.. && pwd)"
PYTHON="${PYTHON:-python3}"
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

retry_until_done() {
    local label="$1"; shift
    local attempt=0
    while true; do
        attempt=$((attempt + 1))
        log "=== $label: attempt $attempt ==="
        local t0=$(date +%s)
        $PYTHON -u "$@" 2>&1 | tee "results/composition/followup_${label}_attempt${attempt}.log"
        local exit_code=${PIPESTATUS[0]}
        local elapsed=$(( $(date +%s) - t0 ))
        if [ $exit_code -eq 0 ]; then
            log "$label: OK (attempt $attempt, ${elapsed}s)"
            return 0
        fi
        log "$label: crashed after ${elapsed}s (exit $exit_code), retrying in 15s..."
        sleep 15
        if [ $attempt -ge 8 ]; then
            log "!!! $label: giving up after 8 attempts !!!"
            return 1
        fi
    done
}

log "===== STEP 1: Atomic baseline ====="
retry_until_done atomic src/compute_atomic_baseline.py --num-episodes 30

log "===== STEP 2: Behavioral distance ====="
retry_until_done bdist src/compute_behavioral_distance.py

log "===== STEP 3: Paired cross-seed × 4 phases ====="
for phase in reach grasp lift place; do
    retry_until_done "paired_${phase}" src/compose_paired_eval.py \
        --num-episodes 30 --swap-phase "$phase"
done

log "=== ALL FOLLOW-UP EXPERIMENTS COMPLETE ==="

# Final aggregated summary
$PYTHON - <<'PY'
import json, os, numpy as np
out_dir = 'results/composition'

print('\n=== ATOMIC SKILL QUALITY (single ECM controls full episode) ===\n')
p = os.path.join(out_dir, 'atomic_baselines.json')
if os.path.exists(p):
    d = json.load(open(p))
    SEEDS = d['seeds']; PHASES = d['phases']
    print(f'{"phase\\\\seed":>10s}', end='')
    for s in SEEDS: print(f' {f"seed={s}":>10s}', end='')
    print()
    for phase in PHASES:
        print(f'{phase:>10s}', end='')
        for s in SEEDS:
            cell = d['cells'].get(f'seed={s},phase={phase}', {})
            sr = cell.get('success_rate')
            print(f' {sr:>9.1f}%' if sr is not None else f' {"—":>10s}', end='')
        print()

print('\n=== PAIRED CROSS-SEED COMPOSITION (per swap_phase, with paired t-test) ===\n')
print(f'{"swap_phase":>12s} {"diag":>8s} {"off":>8s} {"Δ":>6s} {"t":>7s} {"p":>9s}')
for phase in ['reach', 'grasp', 'lift', 'place']:
    p = os.path.join(out_dir, f'paired_cross_seed_{phase}_swap.json')
    if not os.path.exists(p):
        print(f'  {phase}: missing')
        continue
    d = json.load(open(p))
    diag = [v['success_rate'] for v in d['cells'].values() if v.get('is_diagonal')]
    off = [v['success_rate'] for v in d['cells'].values() if not v.get('is_diagonal')]
    tt = d.get('paired_ttest', {})
    print(f'{phase:>12s} {np.mean(diag):>7.1f}% {np.mean(off):>7.1f}% '
          f'{np.mean(diag)-np.mean(off):>+5.1f} '
          f'{tt.get("t_stat", float("nan")):>7.3f} {tt.get("p_value", float("nan")):>9.4f}')

print('\n=== BEHAVIORAL DISTANCE SUMMARY (mean L2 of action) ===\n')
p = os.path.join(out_dir, 'behavioral_distance.json')
if os.path.exists(p):
    d = json.load(open(p))
    print(f'{"phase":>10s} {"mean_l2":>10s} {"max_l2":>10s} {"min_l2":>10s}')
    for phase in d['phases']:
        m = d['distance_per_phase'][phase]
        offs = [m[a][b]['l2'] for a in d['seeds'] for b in d['seeds'] if a != b]
        print(f'{phase:>10s} {np.mean(offs):>10.4f} {np.max(offs):>10.4f} {np.min(offs):>10.4f}')
PY
