#!/bin/bash
# Run evaluations on T3 and T4 to test generalization of the dominant-skill effect.
#
# For each task ∈ {T3_Stack, T4_NutAssembly}, run:
#   1. Atomic baseline (16 ECMs single-controller)        ~25 min
#   2. Behavioral distance                                ~5 min
#   3. Paired cross-seed × 4 phases                       ~95 min
#   4. Subset swap (4 pairs × 16 subsets)                 ~95 min
#
# Total per task: ~3.5 hours. Both tasks: ~7 hours.

set -o pipefail
cd "$(cd "$(dirname "$0")"/.. && pwd)"
PYTHON="${PYTHON:-python3}"
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

retry() {
    local label="$1"; shift
    local attempt=0
    while true; do
        attempt=$((attempt + 1))
        log "=== $label: attempt $attempt ==="
        local t0=$(date +%s)
        $PYTHON -u "$@" 2>&1 | tee "results/composition/eval_${label}_attempt${attempt}.log"
        local exit_code=${PIPESTATUS[0]}
        local elapsed=$(( $(date +%s) - t0 ))
        if [ $exit_code -eq 0 ]; then
            log "$label: OK (${elapsed}s)"
            return 0
        fi
        log "$label: crashed after ${elapsed}s (exit $exit_code), retrying in 15s..."
        sleep 15
        if [ $attempt -ge 8 ]; then
            log "!!! $label giving up after 8 attempts !!!"
            return 1
        fi
    done
}

for TASK in T3_Stack T4_NutAssembly; do
    log "###############################################"
    log "##### TASK: $TASK"
    log "###############################################"

    retry "atomic_${TASK}" compute_atomic_baseline.py \
        --num-episodes 30 --task "$TASK" --out "atomic_${TASK}.json"

    retry "bdist_${TASK}" compute_behavioral_distance.py --task "$TASK" \
        --out "behavioral_distance_${TASK}.json"

    for PHASE in reach grasp lift place; do
        retry "paired_${TASK}_${PHASE}" compose_paired_eval.py \
            --num-episodes 30 --swap-phase "$PHASE" --task "$TASK" \
            --out "paired_${TASK}_${PHASE}_swap.json"
    done

    retry "subset_${TASK}" compose_subset_swap.py \
        --num-episodes 30 --task "$TASK" \
        --pairs 42:2024 42:7 123:2024 7:123 \
        --out "subset_swap_${TASK}.json"
done

log "=== ALL T3/T4 EVALUATIONS COMPLETE ==="

# Cross-task summary
$PYTHON - <<'PY'
import json, os
import numpy as np
out = 'results/composition'

print('\n\n=== CROSS-TASK SUMMARY: Dominant Skill Effect ===\n')
for task in ['T6_TwoArmPegInHole', 'T3_Stack', 'T4_NutAssembly']:
    if task == 'T6_TwoArmPegInHole':
        atomic_p = os.path.join(out, 'atomic_baselines.json')
        subset_p = os.path.join(out, 'subset_swap.json')
    else:
        atomic_p = os.path.join(out, f'atomic_{task}.json')
        subset_p = os.path.join(out, f'subset_swap_{task}.json')

    print(f'--- {task} ---')
    if os.path.exists(atomic_p):
        d = json.load(open(atomic_p))
        # Find peak atomic per phase
        max_per_phase = {}
        for k, v in d['cells'].items():
            phase = dict(p.split('=') for p in k.split(','))['phase']
            sr = v['success_rate']
            if phase not in max_per_phase or sr > max_per_phase[phase][1]:
                max_per_phase[phase] = (k, sr)
        print(f'  Atomic peak per phase:')
        for phase, (k, sr) in max_per_phase.items():
            print(f'    {phase}: {sr:.1f}% ({k})')
    if os.path.exists(subset_p):
        d = json.load(open(subset_p))
        # Compute "reach in swap" vs "reach not in swap" mean per pair
        for pair in d.get('pairs', []):
            with_r, without_r = [], []
            p, s = pair.split(':')
            for cell_key, v in d['cells'].items():
                if not cell_key.startswith(f'p={p},s={s}'):
                    continue
                if v.get('has_reach_swapped'):
                    with_r.append(v['success_rate'])
                else:
                    without_r.append(v['success_rate'])
            if with_r and without_r:
                print(f'  Pair {pair}: reach∈swap mean={np.mean(with_r):.1f}%, '
                      f'reach∉swap mean={np.mean(without_r):.1f}%, '
                      f'Δ={np.mean(with_r)-np.mean(without_r):+.1f}pp')
PY
