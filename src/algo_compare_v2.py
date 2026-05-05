"""
Algorithm Comparison v2 — Hybrid Selector + Threshold Sweep
============================================================

Adds a Hybrid selector:
  - If |q_new - q_old| >= confidence_margin → use AtomicOnly's decision
  - If |q_new - q_old| < confidence_margin  → fall back to FullReval

Also sweeps threshold τ ∈ {0, 5, 10, 15, 20} pp to characterize the tradeoff.

Key claim being tested:
  "Hybrid saves >70% of FullReval probes while matching its decision quality."
"""
import json
import os
import sys
from itertools import product

import numpy as np

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(REPO_DIR, 'results', 'composition')

PHASES = ['reach', 'grasp', 'lift', 'place']
SEEDS = [42, 7, 123, 2024]


def _atomic_path(task):
    if task == 'T6_TwoArmPegInHole':
        return os.path.join(DATA_DIR, 'atomic_baselines.json')
    return os.path.join(DATA_DIR, f'atomic_{task}.json')


def _paired_path(task, phase):
    if task == 'T6_TwoArmPegInHole':
        return os.path.join(DATA_DIR, f'paired_cross_seed_{phase}_swap.json')
    return os.path.join(DATA_DIR, f'paired_{task}_{phase}_swap.json')


def load_atomic(task='T6_TwoArmPegInHole', metric='success_rate'):
    d = json.load(open(_atomic_path(task)))
    out = {}
    for cell_key, v in d['cells'].items():
        parts = dict(p.split('=') for p in cell_key.split(','))
        out[(int(parts['seed']), parts['phase'])] = v.get(metric, v['avg_reward'])
    return out


def load_paired_per_phase(phase, task='T6_TwoArmPegInHole', metric='success_rate'):
    d = json.load(open(_paired_path(task, phase)))
    out = {}
    for cell_key, v in d['cells'].items():
        parts = dict(p.split('=') for p in cell_key.split(','))
        out[(int(parts['primary']), int(parts['swap']))] = v.get(metric, v['avg_reward'])
    return out


# Selector functions: each returns (accept: bool, used_full_reval: bool)
def naive(q_old, q_new, comp_old, comp_new, threshold=5.0, margin=10.0):
    return True, False

def freeze(q_old, q_new, comp_old, comp_new, threshold=5.0, margin=10.0):
    return False, False

def atomic_only(q_old, q_new, comp_old, comp_new, threshold=5.0, margin=10.0):
    return q_new >= q_old - threshold, False

def full_reval(q_old, q_new, comp_old, comp_new, threshold=5.0, margin=10.0):
    return comp_new >= comp_old - threshold, True

def hybrid(q_old, q_new, comp_old, comp_new, threshold=5.0, margin=10.0):
    """Use AtomicOnly when atomic difference is large; otherwise FullReval."""
    if abs(q_new - q_old) >= margin:
        return q_new >= q_old - threshold, False  # confident atomic decision
    else:
        return comp_new >= comp_old - threshold, True  # uncertain → fall back


SELECTORS = {
    'Naive': naive,
    'Freeze': freeze,
    'AtomicOnly': atomic_only,
    'FullReval': full_reval,
    'Hybrid(margin=10)': lambda *a, **k: hybrid(*a, **{**k, 'margin': 10.0}),
    'Hybrid(margin=20)': lambda *a, **k: hybrid(*a, **{**k, 'margin': 20.0}),
    'Hybrid(margin=30)': lambda *a, **k: hybrid(*a, **{**k, 'margin': 30.0}),
}


def evaluate_at_threshold(events, threshold):
    """Return per-selector summary at the given threshold."""
    summary = {sel: {'accept': 0, 'reject': 0, 'sum_outcome': 0.0,
                     'oracle_match': 0, 'fullreval_calls': 0}
               for sel in SELECTORS}
    for ev in events:
        for sel_name, sel_fn in SELECTORS.items():
            decision, used_full = sel_fn(ev['q_old'], ev['q_new'],
                                          ev['comp_old'], ev['comp_new'],
                                          threshold=threshold)
            outcome = ev['comp_new'] if decision else ev['comp_old']
            s = summary[sel_name]
            if decision:
                s['accept'] += 1
            else:
                s['reject'] += 1
            s['sum_outcome'] += outcome
            if decision == ev['oracle_accept']:
                s['oracle_match'] += 1
            if used_full:
                s['fullreval_calls'] += 1
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='T6_TwoArmPegInHole')
    parser.add_argument('--metric', type=str, default='auto',
                        choices=['auto', 'success_rate', 'avg_reward'],
                        help="auto = success_rate for T6, avg_reward for T3/T4 (where success ~0)")
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    metric = args.metric
    if metric == 'auto':
        metric = 'success_rate' if args.task == 'T6_TwoArmPegInHole' else 'avg_reward'
    out_name = args.out or f'algo_compare_v2_{args.task}.json'

    print(f"Task: {args.task}, metric: {metric}")
    atomic = load_atomic(task=args.task, metric=metric)
    paired = {phase: load_paired_per_phase(phase, task=args.task, metric=metric)
              for phase in PHASES}

    events = []
    for phase in PHASES:
        for baseline_seed in SEEDS:
            for new_seed in SEEDS:
                if new_seed == baseline_seed:
                    continue
                events.append({
                    'phase': phase,
                    'baseline_seed': baseline_seed,
                    'new_seed': new_seed,
                    'q_old': atomic[(baseline_seed, phase)],
                    'q_new': atomic[(new_seed, phase)],
                    'comp_old': paired[phase][(baseline_seed, baseline_seed)],
                    'comp_new': paired[phase][(baseline_seed, new_seed)],
                    'oracle_accept': paired[phase][(baseline_seed, new_seed)] >
                                      paired[phase][(baseline_seed, baseline_seed)],
                })

    n = len(events)
    print(f"\n{'='*80}")
    print(f"ALGORITHM COMPARISON v2 — Threshold sweep + Hybrid selector")
    print(f"n={n} events, FullReval probe cost = 30 episodes per call")
    print(f"{'='*80}\n")

    thresholds = [0.0, 5.0, 10.0, 15.0, 20.0]
    print(f"{'Threshold':>10s}", end='')
    for sel in SELECTORS:
        print(f" {sel:>20s}", end='')
    print()

    print(f"{'(MeanOutcome %)':>10s}")
    for tau in thresholds:
        s = evaluate_at_threshold(events, tau)
        print(f"{tau:>9.0f}pp", end='')
        for sel in SELECTORS:
            mean_out = s[sel]['sum_outcome'] / n
            print(f" {mean_out:>19.2f}%", end='')
        print()

    print(f"\n{'(OracleMatch %)':>10s}")
    for tau in thresholds:
        s = evaluate_at_threshold(events, tau)
        print(f"{tau:>9.0f}pp", end='')
        for sel in SELECTORS:
            match = s[sel]['oracle_match'] / n * 100
            print(f" {match:>19.1f}%", end='')
        print()

    print(f"\n{'(FullReval call rate, lower=cheaper)':>10s}")
    for tau in thresholds:
        s = evaluate_at_threshold(events, tau)
        print(f"{tau:>9.0f}pp", end='')
        for sel in SELECTORS:
            calls = s[sel]['fullreval_calls']
            print(f" {calls/n*100:>18.1f}%", end='')
        print()

    # Headline at threshold = 5
    print(f"\n{'='*80}")
    print("HEADLINE @ threshold = 5pp")
    print(f"{'='*80}\n")
    s_default = evaluate_at_threshold(events, 5.0)
    print(f"{'Algorithm':>22s} {'Mean':>10s} {'Oracle':>9s} {'FullReval calls':>17s} {'Notes':>30s}")
    notes = {
        'Naive': 'always accept',
        'Freeze': 'always reject',
        'AtomicOnly': '0 FullReval, atomic only',
        'FullReval': 'gold standard (expensive)',
        'Hybrid(margin=10)': 'cheap fallback to FullReval',
        'Hybrid(margin=20)': 'medium margin',
        'Hybrid(margin=30)': 'wide margin (most calls to FullReval)',
    }
    for sel, info in s_default.items():
        mean = info['sum_outcome'] / n
        match = info['oracle_match'] / n * 100
        calls = info['fullreval_calls'] / n * 100
        print(f"{sel:>22s} {mean:>9.2f}% {match:>8.1f}% {calls:>15.1f}% {notes.get(sel, ''):>30s}")

    # Save full data
    out = {
        'n_events': n,
        'thresholds': thresholds,
        'selectors': list(SELECTORS),
        'sweep': {tau: {sel: {k: v for k, v in s.items()}
                        for sel, s in evaluate_at_threshold(events, tau).items()}
                  for tau in thresholds},
        'events': events,
    }
    out_path = os.path.join(DATA_DIR, out_name)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
