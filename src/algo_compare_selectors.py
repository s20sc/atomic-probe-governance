"""
Algorithm Comparison: Skill Update Selectors
=============================================

Compare 4 decision algorithms for "should we accept this skill update?"
using the existing pre-experiment data — no new training or simulation.

Algorithms:
  1. Naive       — always accept
  2. Freeze      — always reject (BLADE/SymSkill default)
  3. AtomicOnly  — accept iff atomic_quality(v_new) >= atomic_quality(v_old) - τ
  4. FullReval   — accept iff comp_success(use v_new) >= comp_success(use v_old) - τ

Setup: 4 phases × 4 baseline_seeds × 3 candidate_new_seeds = 48 update events.
For each event, each algorithm makes accept/reject; we measure:
  - Resulting composition success after the decision
  - Decision cost in probe-episodes
  - Decision quality vs the oracle (best a priori choice for that event)

Data sources:
  results/composition/atomic_baselines.json
  results/composition/paired_cross_seed_<phase>_swap.json (×4)

Output:
  results/composition/algo_compare.json + console summary table.
"""
import json
import os
import sys
from collections import defaultdict
from itertools import product

import numpy as np

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(REPO_DIR, 'results', 'composition')

PHASES = ['reach', 'grasp', 'lift', 'place']
SEEDS = [42, 7, 123, 2024]


def load_atomic():
    """Returns {(seed, phase): success_rate}."""
    d = json.load(open(os.path.join(DATA_DIR, 'atomic_baselines.json')))
    out = {}
    for cell_key, v in d['cells'].items():
        # cell_key = "seed=42,phase=reach"
        parts = dict(p.split('=') for p in cell_key.split(','))
        out[(int(parts['seed']), parts['phase'])] = v['success_rate']
    return out


def load_paired_per_phase(phase):
    """For a given swap_phase, returns dict {(primary, swap): success_rate}.
    Diagonal cells (primary == swap) = baseline composition success of that seed.
    Off-diagonal = composition success after replacing this phase's ECM with swap_seed's.
    """
    d = json.load(open(os.path.join(
        DATA_DIR, f'paired_cross_seed_{phase}_swap.json')))
    out = {}
    for cell_key, v in d['cells'].items():
        # cell_key = "primary=42,swap=7"
        parts = dict(p.split('=') for p in cell_key.split(','))
        out[(int(parts['primary']), int(parts['swap']))] = v['success_rate']
    return out


# -------------------------------------------------------------
# 4 selector algorithms — each returns ACCEPT (True) or REJECT
# -------------------------------------------------------------

def naive_selector(q_old, q_new, comp_old, comp_new, threshold=5.0):
    return True


def freeze_selector(q_old, q_new, comp_old, comp_new, threshold=5.0):
    return False


def atomic_selector(q_old, q_new, comp_old, comp_new, threshold=5.0):
    return q_new >= q_old - threshold


def fullreval_selector(q_old, q_new, comp_old, comp_new, threshold=5.0):
    return comp_new >= comp_old - threshold


SELECTORS = {
    'Naive': naive_selector,
    'Freeze': freeze_selector,
    'AtomicOnly': atomic_selector,
    'FullReval': fullreval_selector,
}


def main():
    threshold = 5.0  # pp tolerance for "no worse than"
    probe_episodes = 30

    atomic = load_atomic()
    paired = {phase: load_paired_per_phase(phase) for phase in PHASES}

    # Iterate over all 48 update events
    # Event: in baseline composition (all 4 ECMs from baseline_seed),
    #        propose to replace `phase` ECM with new_seed's ECM.
    events = []
    for phase in PHASES:
        for baseline_seed in SEEDS:
            for new_seed in SEEDS:
                if new_seed == baseline_seed:
                    continue
                q_old = atomic[(baseline_seed, phase)]
                q_new = atomic[(new_seed, phase)]
                comp_old = paired[phase][(baseline_seed, baseline_seed)]
                comp_new = paired[phase][(baseline_seed, new_seed)]
                events.append({
                    'phase': phase,
                    'baseline_seed': baseline_seed,
                    'new_seed': new_seed,
                    'q_old': q_old, 'q_new': q_new,
                    'comp_old': comp_old, 'comp_new': comp_new,
                    # Oracle (best a priori): pick the higher composition success
                    'oracle_accept': comp_new > comp_old,
                })

    # For each algorithm, compute:
    #  - decision per event
    #  - resulting comp success (comp_new if accept, comp_old if reject)
    #  - probe cost per decision
    #  - agreement with oracle
    summary = {sel: {'accept_count': 0, 'reject_count': 0,
                      'sum_outcome': 0.0, 'oracle_match': 0,
                      'wrong_accepts': [], 'wrong_rejects': [],
                      'per_event': []} for sel in SELECTORS}
    cost_per_decision = {
        'Naive': 0,                 # no probe
        'Freeze': 0,                # no probe
        'AtomicOnly': probe_episodes,  # 1 atomic eval (reusable across compositions!)
        'FullReval': probe_episodes,   # 1 full composition eval (per composition)
    }

    for ev in events:
        for sel_name, sel_fn in SELECTORS.items():
            decision = sel_fn(ev['q_old'], ev['q_new'], ev['comp_old'], ev['comp_new'],
                               threshold)
            outcome = ev['comp_new'] if decision else ev['comp_old']
            s = summary[sel_name]
            if decision:
                s['accept_count'] += 1
            else:
                s['reject_count'] += 1
            s['sum_outcome'] += outcome
            if decision == ev['oracle_accept']:
                s['oracle_match'] += 1
            else:
                if decision:
                    s['wrong_accepts'].append(ev)
                else:
                    s['wrong_rejects'].append(ev)
            s['per_event'].append({**ev, 'decision': decision, 'outcome': outcome})

    n = len(events)
    print(f"\n{'='*78}")
    print(f"ALGORITHM COMPARISON: 4 skill-update selectors on T6 (n = {n} events)")
    print(f"Threshold: {threshold} pp ; Probe size: {probe_episodes} episodes")
    print(f"{'='*78}\n")

    print(f"{'Algorithm':>14s} {'Accepts':>9s} {'Rejects':>9s} "
          f"{'MeanOutcome':>13s} {'OracleMatch':>13s} {'Cost(probes)':>14s}")
    for sel_name, s in summary.items():
        mean_out = s['sum_outcome'] / n
        match_pct = s['oracle_match'] / n * 100
        cost_label = '0 (none)' if cost_per_decision[sel_name] == 0 else (
            f'{probe_episodes} reusable' if sel_name == 'AtomicOnly' else
            f'{probe_episodes} per comp'
        )
        print(f"{sel_name:>14s} {s['accept_count']:>9d} {s['reject_count']:>9d} "
              f"{mean_out:>12.2f}% {match_pct:>12.1f}% {cost_label:>14s}")

    # Cost analysis: amortized over K compositions per skill
    print(f"\n{'='*78}")
    print("AMORTIZED PROBE COST per (skill update × 1 query)")
    print(f"  K = number of compositions that reuse this skill")
    print(f"{'='*78}\n")
    print(f"{'Algorithm':>14s} {'K=1':>10s} {'K=4':>10s} {'K=16':>10s} {'K=64':>10s}")
    for sel_name in SELECTORS:
        if sel_name in ('Naive', 'Freeze'):
            costs = ['0'] * 4
        elif sel_name == 'AtomicOnly':
            # one atomic eval, reused across all K compositions
            costs = [f'{probe_episodes/K:.1f}' for K in (1, 4, 16, 64)]
        else:  # FullReval
            # one full eval per composition
            costs = [f'{probe_episodes}'] * 4
        print(f"{sel_name:>14s}", end='')
        for c in costs:
            print(f" {c:>9s}", end='')
        print()

    # Detailed wrong decisions per algorithm
    for sel_name, s in summary.items():
        if s['wrong_accepts'] or s['wrong_rejects']:
            print(f"\n--- {sel_name}: wrong decisions ---")
            for ev in s['wrong_accepts'][:5]:
                print(f"  WRONG-ACCEPT: phase={ev['phase']}, "
                      f"baseline=seed{ev['baseline_seed']}, new=seed{ev['new_seed']}, "
                      f"q_old={ev['q_old']:.0f}%, q_new={ev['q_new']:.0f}%, "
                      f"comp_old={ev['comp_old']:.0f}%, comp_new={ev['comp_new']:.0f}%")
            for ev in s['wrong_rejects'][:5]:
                print(f"  WRONG-REJECT: phase={ev['phase']}, "
                      f"baseline=seed{ev['baseline_seed']}, new=seed{ev['new_seed']}, "
                      f"q_old={ev['q_old']:.0f}%, q_new={ev['q_new']:.0f}%, "
                      f"comp_old={ev['comp_old']:.0f}%, comp_new={ev['comp_new']:.0f}%")
            n_wrong_a = len(s['wrong_accepts'])
            n_wrong_r = len(s['wrong_rejects'])
            print(f"  total wrong: {n_wrong_a} false-accepts + {n_wrong_r} false-rejects = "
                  f"{n_wrong_a + n_wrong_r}/{n}")

    # Save
    out = {
        'n_events': n,
        'threshold_pp': threshold,
        'probe_episodes': probe_episodes,
        'summary': {sel: {k: v for k, v in s.items() if k != 'per_event'}
                    for sel, s in summary.items()},
        'events': events,
        'per_event_decisions': {
            sel: [{**ev_d, 'wrong_accepts': None, 'wrong_rejects': None}
                  for ev_d in summary[sel]['per_event']]
            for sel in SELECTORS
        },
    }
    out_path = os.path.join(DATA_DIR, 'algo_compare.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
