"""
Bootstrap CIs for Hybrid(margin=20) and Hybrid(margin=30) rows of Table 6.

Existing Wilson CIs in v7 are statistically defensible but methodology-
inconsistent with the bootstrap-CI rows for Hybrid(m=10) etc. This script
re-computes per-task and aggregate CIs using bootstrap-with-replacement
(B=5000) on the same event pool that algo_compare_v2 uses.

Usage:
    python3 bootstrap_cis_table6.py --threshold 5 --bootstrap 5000

Output:
    results/composition/bootstrap_cis_table6.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from algo_compare_v2 import (
    PHASES, SEEDS, SELECTORS,
    load_atomic, load_paired_per_phase,
)

TASKS = ['T6_TwoArmPegInHole', 'T3_Stack', 'T4_NutAssembly']
TARGET_SELECTORS = ['Hybrid(margin=10)', 'Hybrid(margin=20)', 'Hybrid(margin=30)']


def build_events(task, metric):
    atomic = load_atomic(task=task, metric=metric)
    paired = {phase: load_paired_per_phase(phase, task=task, metric=metric)
              for phase in PHASES}
    events = []
    for phase in PHASES:
        for baseline_seed in SEEDS:
            for new_seed in SEEDS:
                if new_seed == baseline_seed:
                    continue
                events.append({
                    'phase': phase,
                    'q_old': atomic[(baseline_seed, phase)],
                    'q_new': atomic[(new_seed, phase)],
                    'comp_old': paired[phase][(baseline_seed, baseline_seed)],
                    'comp_new': paired[phase][(baseline_seed, new_seed)],
                    'oracle_accept': (paired[phase][(baseline_seed, new_seed)] >
                                       paired[phase][(baseline_seed, baseline_seed)]),
                })
    return events


def per_event_oracle_match(events, selector_fn, threshold):
    """Binary outcome per event: 1 if selector decision == oracle_accept, else 0.

    This matches the 'OracleMatch %' column reported in Table 6 of v7.
    """
    out = np.empty(len(events), dtype=float)
    for i, ev in enumerate(events):
        decision, _ = selector_fn(
            ev['q_old'], ev['q_new'], ev['comp_old'], ev['comp_new'],
            threshold=threshold,
        )
        out[i] = 1.0 if decision == ev['oracle_accept'] else 0.0
    return out


def bootstrap_ci(values, B, alpha=0.05, rng=None):
    if rng is None:
        rng = np.random.default_rng(20260503)
    n = len(values)
    means = np.empty(B, dtype=float)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        means[b] = values[idx].mean()
    lo = float(np.percentile(means, 100 * (alpha / 2)))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--threshold', type=float, default=5.0,
                   help='τ in pp; v7 table uses 5')
    p.add_argument('--bootstrap', type=int, default=5000)
    p.add_argument('--out', type=str, default=None)
    args = p.parse_args()

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'results', 'composition', 'bootstrap_cis_table6.json',
    )

    rng = np.random.default_rng(20260503)
    results = {
        'threshold': args.threshold,
        'bootstrap_B': args.bootstrap,
        'metric_per_task': {
            'T6_TwoArmPegInHole': 'success_rate',
            'T3_Stack': 'avg_reward',
            'T4_NutAssembly': 'avg_reward',
        },
        'per_task': {},
    }

    per_event_pool = {}
    for task in TASKS:
        metric = results['metric_per_task'][task]
        events = build_events(task, metric)
        per_event_pool[task] = {}
        results['per_task'][task] = {'n_events': len(events), 'selectors': {}}
        for sel_name in TARGET_SELECTORS:
            sel_fn = SELECTORS[sel_name]
            outcomes = per_event_oracle_match(events, sel_fn, args.threshold)
            per_event_pool[task][sel_name] = outcomes
            mean_obs = float(outcomes.mean()) * 100  # report as percent
            lo, hi = bootstrap_ci(outcomes, args.bootstrap, rng=rng)
            results['per_task'][task]['selectors'][sel_name] = {
                'oracle_match_pct': mean_obs,
                'ci_lo_pct': lo * 100,
                'ci_hi_pct': hi * 100,
            }

    # Aggregate: mean across (T6, T3, T4) at the per-event level if comparable;
    # but T6 metric is success_rate (0-100), T3/T4 use avg_reward (different scale).
    # The v7 "Avg" column is the simple arithmetic mean of the per-task values.
    # Compute its CI via paired bootstrap over the union event index (same n per task).
    results['avg_three_tasks'] = {'selectors': {}}
    for sel_name in TARGET_SELECTORS:
        per_task_means = []
        per_task_outcomes = []
        for task in TASKS:
            per_task_outcomes.append(per_event_pool[task][sel_name])
            per_task_means.append(float(per_event_pool[task][sel_name].mean()))
        avg_mean = float(np.mean(per_task_means)) * 100
        ns = [len(o) for o in per_task_outcomes]
        boot = np.empty(args.bootstrap, dtype=float)
        for b in range(args.bootstrap):
            mt = []
            for o, n in zip(per_task_outcomes, ns):
                idx = rng.integers(0, n, size=n)
                mt.append(o[idx].mean())
            boot[b] = np.mean(mt)
        lo = float(np.percentile(boot, 2.5)) * 100
        hi = float(np.percentile(boot, 97.5)) * 100
        results['avg_three_tasks']['selectors'][sel_name] = {
            'oracle_match_pct': avg_mean,
            'ci_lo_pct': lo,
            'ci_hi_pct': hi,
        }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Wrote: {out_path}\n")
    print(f"Threshold τ = {args.threshold} pp,  bootstrap B = {args.bootstrap}")
    print('=' * 78)
    print(f"{'Task':<22s} {'Selector':<22s} {'Oracle %':>9s} {'CI lo':>8s} {'CI hi':>8s}")
    print('-' * 78)
    for task in TASKS + ['avg_three_tasks']:
        block = (results['per_task'][task] if task in TASKS
                 else results['avg_three_tasks'])
        for sel_name in TARGET_SELECTORS:
            v = block['selectors'][sel_name]
            print(f"{task:<22s} {sel_name:<22s} "
                  f"{v['oracle_match_pct']:>8.2f}  {v['ci_lo_pct']:>7.2f}  {v['ci_hi_pct']:>7.2f}")


if __name__ == '__main__':
    main()
