#!/usr/bin/env python3
"""Compute mechanism (b) action smoothness and mechanism (c) trajectory
length from the B' rollout npz files.

Run AFTER scp'ing `states_t6_reach_full/` from Linux into this folder:

    cd <repo>/data/mechanism_probes/
    python3 compute_b_prime_mechanisms.py

Output: prints a table per mechanism and writes a JSON summary +
human-readable .txt to this same folder, ready for direct lift into
the v12 §app:extdisc patch.

Mechanism definitions (per peer-review I4 and §app:extdisc):

  (b) action smoothness:
        For each ECM s, average over 30 episodes of
            mean_{t=1..61} ||a_t - a_{t-1}||_2
        Lower => smoother.

  (c) trajectory length (state-space path length):
        For each ECM s, average over 30 episodes of
            sum_{t=1..61} ||obs_t - obs_{t-1}||_2
        Lower => more efficient / shorter trajectory.

Both metrics are computed FOR THE REACH ECM (= swap-seed), aggregated
over all 4 primaries (each primary contributes 30 eps) → 120 episodes
per ECM.
"""

import json
import os
import sys
import glob
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'states_t6_reach_full')
OUT_TXT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'b_prime_mechanism_results.txt')
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'b_prime_mechanism_results.json')

SEEDS = [42, 7, 123, 2024]
DOM_SEED = 2024


def main():
    if not os.path.isdir(DATA_DIR):
        print(f"ERROR: {DATA_DIR} not found.")
        print("Did you scp from Linux yet?")
        print()
        print("  rsync -avz user@linux-host:"
              "<framework>/results/states_t6_reach_full/ \\")
        print(f"      {DATA_DIR}/")
        sys.exit(1)

    files = sorted(glob.glob(os.path.join(DATA_DIR, 'reach_primary*_full.npz')))
    print(f"Found {len(files)} npz files in {DATA_DIR}")
    if len(files) != 16:
        print(f"WARNING: expected 16 files, got {len(files)}.")

    # Aggregate per (swap-ECM seed): we want ECM behavior, indexed by
    # the seed of the reach ECM regardless of which primary the rest
    # of the composition came from.
    agg_action_smooth = {s: [] for s in SEEDS}
    agg_traj_length = {s: [] for s in SEEDS}

    for f in files:
        data = np.load(f)
        # parse "reach_primary{P}_swap{S}_full.npz"
        name = os.path.basename(f)
        try:
            primary = int(name.split('primary')[1].split('_')[0])
            swap = int(name.split('swap')[1].split('_')[0])
        except (IndexError, ValueError):
            print(f"WARNING: cannot parse {name}; skipping")
            continue

        actions = data['actions_seq']  # (30, 62, act_dim)
        states = data['states_seq']    # (30, 62, obs_dim)

        # (b) action smoothness: per-step delta L2 norm, averaged over time
        action_deltas = np.diff(actions, axis=1)  # (30, 61, act_dim)
        action_delta_norms = np.linalg.norm(action_deltas, axis=2)  # (30, 61)
        per_episode_smoothness = action_delta_norms.mean(axis=1)  # (30,)
        agg_action_smooth[swap].extend(per_episode_smoothness.tolist())

        # (c) trajectory length: per-step state delta L2 norm, summed over time
        state_deltas = np.diff(states, axis=1)  # (30, 61, obs_dim)
        state_delta_norms = np.linalg.norm(state_deltas, axis=2)  # (30, 61)
        per_episode_path_length = state_delta_norms.sum(axis=1)  # (30,)
        agg_traj_length[swap].extend(per_episode_path_length.tolist())

    # Summary stats per ECM seed
    print()
    print("=" * 78)
    print(" Mechanism (b) action smoothness  (mean of ||a_t - a_{t-1}||_2)")
    print(" Lower = smoother. 120 episodes per ECM (4 primaries × 30 eps).")
    print("=" * 78)
    print()
    print(f"  {'ECM seed':<10} {'mean':>8} {'std':>8} {'95% CI':>22} "
          f"{'note'}")
    out_b = {}
    for s in SEEDS:
        vals = np.asarray(agg_action_smooth[s])
        m = vals.mean()
        sd = vals.std(ddof=1)
        ci_lo, ci_hi = bootstrap_ci(vals)
        marker = "  ← DOMINANT" if s == DOM_SEED else ""
        print(f"  seed={s:<6} {m:>7.4f}  {sd:>7.4f}  "
              f"[{ci_lo:>+6.3f}, {ci_hi:>+6.3f}]   {marker}")
        out_b[str(s)] = {'mean': m, 'std': sd, 'ci95': [ci_lo, ci_hi],
                         'n_episodes': len(vals)}

    dom = out_b[str(DOM_SEED)]
    sib_means = [out_b[str(s)]['mean'] for s in SEEDS if s != DOM_SEED]
    sib_avg = sum(sib_means) / 3
    print()
    print(f"  Dominant: {dom['mean']:.4f}    Sibling avg: {sib_avg:.4f}    "
          f"Δ = {(dom['mean'] - sib_avg):+.4f} ({(dom['mean'] - sib_avg)/sib_avg*100:+.1f}%)")
    print(f"  Verdict: dominant ECM is "
          f"{'SMOOTHER' if dom['mean'] < sib_avg else 'NOT smoother'} "
          f"than the sibling mean.")

    print()
    print("=" * 78)
    print(" Mechanism (c) trajectory length  (sum of ||s_t - s_{t-1}||_2)")
    print(" Lower = shorter / more efficient. 120 episodes per ECM.")
    print("=" * 78)
    print()
    print(f"  {'ECM seed':<10} {'mean':>10} {'std':>10} {'95% CI':>26} "
          f"{'note'}")
    out_c = {}
    for s in SEEDS:
        vals = np.asarray(agg_traj_length[s])
        m = vals.mean()
        sd = vals.std(ddof=1)
        ci_lo, ci_hi = bootstrap_ci(vals)
        marker = "  ← DOMINANT" if s == DOM_SEED else ""
        print(f"  seed={s:<6} {m:>9.3f}  {sd:>9.3f}  "
              f"[{ci_lo:>+8.2f}, {ci_hi:>+8.2f}]   {marker}")
        out_c[str(s)] = {'mean': m, 'std': sd, 'ci95': [ci_lo, ci_hi],
                         'n_episodes': len(vals)}

    dom = out_c[str(DOM_SEED)]
    sib_means = [out_c[str(s)]['mean'] for s in SEEDS if s != DOM_SEED]
    sib_avg = sum(sib_means) / 3
    print()
    print(f"  Dominant: {dom['mean']:.3f}    Sibling avg: {sib_avg:.3f}    "
          f"Δ = {(dom['mean'] - sib_avg):+.3f} ({(dom['mean'] - sib_avg)/sib_avg*100:+.1f}%)")
    print(f"  Verdict: dominant ECM has a "
          f"{'SHORTER' if dom['mean'] < sib_avg else 'LONGER'} "
          f"trajectory than the sibling mean.")

    # Save JSON
    summary = {
        'task': 'T6_TwoArmPegInHole',
        'phase': 'reach',
        'n_episodes_per_ecm': 120,
        'mechanism_b_action_smoothness': out_b,
        'mechanism_c_trajectory_length': out_c,
    }
    with open(OUT_JSON, 'w') as f:
        json.dump(summary, f, indent=2)
    print()
    print(f"JSON summary saved to: {OUT_JSON}")


def bootstrap_ci(arr, B=5000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    n = len(arr)
    means = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, size=n)
        means[i] = arr[idx].mean()
    means.sort()
    return float(means[int(alpha / 2 * B)]), float(means[int((1 - alpha / 2) * B)])


if __name__ == '__main__':
    main()
