"""
B — Hand-off state-distribution analysis.

Loads the 16 phase-end-state npz files produced by
compose_paired_eval.py --save-states, groups them by *swap seed*
(the seed whose ECM is plugged into the reach phase), and computes
pairwise distribution distances between the four swap-groups.

Question: is the dominant ECM (seed=2024) phase-end distribution
WIDER (higher per-dim entropy) or SHIFTED (further from pooled
centroid) than the three siblings (42, 7, 123)?

Distances reported:
- Wasserstein-2 between Gaussian fits (closed form for diagonal cov)
- Per-dim variance ratio: dominant/sibling
- Centroid L2 distance to pooled mean
"""
import json
import os
from itertools import combinations

import numpy as np

STATES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'results', 'states_t6_reach',
)
SEEDS = [42, 7, 123, 2024]
DOMINANT_SEED = 2024  # paper claim
OUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'results', 'composition', 'b_handoff_state_analysis.json',
)


def load_swap_group(swap_seed):
    """Concat phase_end_obs across all 4 primaries for given swap_seed."""
    arrs = []
    for primary in SEEDS:
        path = os.path.join(STATES_DIR, f'reach_primary{primary}_swap{swap_seed}.npz')
        d = np.load(path)
        arrs.append(d['phase_end_obs'])  # shape (30, 218)
    return np.concatenate(arrs, axis=0)  # shape (120, 218)


def w2_diag_gaussian(mu1, sigma1_diag, mu2, sigma2_diag):
    """Wasserstein-2 distance between two diagonal Gaussians.

    Closed form: W2^2 = ||mu1-mu2||^2 + sum_i (sigma1_i + sigma2_i - 2 sqrt(sigma1_i sigma2_i))
    where sigma_i are variances.
    """
    mean_term = float(np.sum((mu1 - mu2) ** 2))
    cov_term = float(np.sum(sigma1_diag + sigma2_diag
                             - 2 * np.sqrt(np.maximum(sigma1_diag * sigma2_diag, 0))))
    return float(np.sqrt(max(mean_term + cov_term, 0)))


def diag_kl_gaussian(mu1, sigma1, mu2, sigma2, eps=1e-10):
    """KL(N(mu1,sigma1) || N(mu2,sigma2)) for diagonal Gaussians."""
    sigma1 = np.maximum(sigma1, eps)
    sigma2 = np.maximum(sigma2, eps)
    d = mu1.shape[0]
    log_term = float(np.sum(np.log(sigma2) - np.log(sigma1)))
    trace_term = float(np.sum(sigma1 / sigma2))
    quad_term = float(np.sum((mu2 - mu1) ** 2 / sigma2))
    return 0.5 * (log_term + trace_term + quad_term - d)


def main():
    groups = {s: load_swap_group(s) for s in SEEDS}
    print('Loaded swap groups:')
    for s, arr in groups.items():
        print(f'  swap={s}: shape={arr.shape}, mean L2 to origin={float(np.linalg.norm(arr.mean(axis=0))):.2f}')

    # Per-group statistics
    stats = {}
    for s in SEEDS:
        arr = groups[s]
        stats[s] = {
            'mean': arr.mean(axis=0),
            'var': arr.var(axis=0, ddof=1),
            'mean_diag_var': float(arr.var(axis=0, ddof=1).mean()),
            'sum_diag_var': float(arr.var(axis=0, ddof=1).sum()),
            'frobenius_cov': float(np.linalg.norm(np.cov(arr, rowvar=False), 'fro')),
            'n': arr.shape[0],
        }

    # Pooled centroid for shift analysis
    pooled = np.concatenate([groups[s] for s in SEEDS], axis=0)
    pooled_mean = pooled.mean(axis=0)

    print('\n=== Per-swap-group: width (sum diag var) and shift to pooled centroid ===')
    print(f"{'swap_seed':<10} {'sum_diag_var':>14} {'mean_diag_var':>14} {'shift_L2_to_pool':>18}")
    for s in SEEDS:
        shift = float(np.linalg.norm(stats[s]['mean'] - pooled_mean))
        marker = ' ← dominant' if s == DOMINANT_SEED else ''
        print(f"  {s:<8} {stats[s]['sum_diag_var']:>14.3f} {stats[s]['mean_diag_var']:>14.4f} {shift:>18.3f}{marker}")

    # Pairwise W2 between groups (diagonal Gaussian fit)
    print('\n=== Pairwise W2 (diag Gaussian) between swap-groups ===')
    print(f"{'pair':<22} {'W2':>10}")
    pairwise = {}
    for s1, s2 in combinations(SEEDS, 2):
        w2 = w2_diag_gaussian(stats[s1]['mean'], stats[s1]['var'],
                               stats[s2]['mean'], stats[s2]['var'])
        pairwise[f'{s1}_vs_{s2}'] = w2
        marker = '  ← involves dominant' if DOMINANT_SEED in (s1, s2) else ''
        print(f"  swap={s1} vs swap={s2:<8} {w2:>10.3f}{marker}")

    # Dominant-vs-sibling vs sibling-vs-sibling summary
    dom_vs_sibling = []
    sibling_vs_sibling = []
    for k, v in pairwise.items():
        a, b = [int(x) for x in k.split('_vs_')]
        if DOMINANT_SEED in (a, b):
            dom_vs_sibling.append(v)
        else:
            sibling_vs_sibling.append(v)
    print(f'\nDominant-vs-sibling W2: mean={np.mean(dom_vs_sibling):.3f}, '
          f'min={np.min(dom_vs_sibling):.3f}, max={np.max(dom_vs_sibling):.3f}')
    print(f'Sibling-vs-sibling W2:  mean={np.mean(sibling_vs_sibling):.3f}, '
          f'min={np.min(sibling_vs_sibling):.3f}, max={np.max(sibling_vs_sibling):.3f}')

    # Per-dim variance ratio: dominant / mean(sibling)
    dom_var = stats[DOMINANT_SEED]['var']
    sib_var_mean = np.mean([stats[s]['var'] for s in SEEDS if s != DOMINANT_SEED], axis=0)
    var_ratio = dom_var / np.maximum(sib_var_mean, 1e-10)
    print(f'\n=== Variance ratio (dominant {DOMINANT_SEED} / mean-of-siblings) per-dim ===')
    print(f'  median ratio: {np.median(var_ratio):.3f}')
    print(f'  mean ratio:   {np.mean(var_ratio):.3f}')
    print(f'  pct dims dominant > sibling: {(var_ratio > 1).mean()*100:.1f}%')
    print(f'  ratio sum: {dom_var.sum() / sib_var_mean.sum():.3f}')

    # Headline interpretation
    print('\n' + '='*70)
    print('HEADLINE FOR PAPER §9')
    print('='*70)
    dom_shift = float(np.linalg.norm(stats[DOMINANT_SEED]['mean'] - pooled_mean))
    sib_shifts = [float(np.linalg.norm(stats[s]['mean'] - pooled_mean))
                  for s in SEEDS if s != DOMINANT_SEED]
    sib_shift_mean = np.mean(sib_shifts)
    print(f'Phase-end shift to pooled centroid:')
    print(f'  dominant (seed=2024): {dom_shift:.3f}')
    print(f'  siblings (mean):      {sib_shift_mean:.3f}')
    rel = (dom_shift - sib_shift_mean) / max(sib_shift_mean, 1e-10) * 100
    print(f'  dominant is {abs(rel):.1f}% '
          f"{'further from' if rel > 0 else 'closer to'} pooled centroid than siblings")
    width_rel = (stats[DOMINANT_SEED]['sum_diag_var'] -
                  np.mean([stats[s]['sum_diag_var'] for s in SEEDS if s != DOMINANT_SEED])) / \
                 max(np.mean([stats[s]['sum_diag_var'] for s in SEEDS if s != DOMINANT_SEED]), 1e-10) * 100
    print(f'Phase-end distribution width (sum diag var):')
    print(f'  dominant {"wider" if width_rel > 0 else "narrower"} by {abs(width_rel):.1f}% '
          f'than mean-of-siblings')
    print(f'Pairwise W2:')
    print(f'  dominant-vs-sibling mean: {np.mean(dom_vs_sibling):.3f}')
    print(f'  sibling-vs-sibling mean:  {np.mean(sibling_vs_sibling):.3f}')
    print(f'  ratio: {np.mean(dom_vs_sibling)/max(np.mean(sibling_vs_sibling),1e-10):.3f}x')

    # Save results
    out = {
        'analysis': 'B handoff state distribution',
        'task': 'T6_TwoArmPegInHole',
        'phase': 'reach',
        'dominant_seed': DOMINANT_SEED,
        'seeds': SEEDS,
        'n_per_group': stats[SEEDS[0]]['n'],
        'obs_dim': int(stats[SEEDS[0]]['mean'].shape[0]),
        'per_group': {
            str(s): {
                'sum_diag_var': stats[s]['sum_diag_var'],
                'mean_diag_var': stats[s]['mean_diag_var'],
                'frobenius_cov': stats[s]['frobenius_cov'],
                'shift_L2_to_pooled_centroid': float(
                    np.linalg.norm(stats[s]['mean'] - pooled_mean)),
            } for s in SEEDS
        },
        'pairwise_w2_diag_gaussian': pairwise,
        'summary': {
            'dom_vs_sibling_w2_mean': float(np.mean(dom_vs_sibling)),
            'sibling_vs_sibling_w2_mean': float(np.mean(sibling_vs_sibling)),
            'dominant_shift_pct_vs_siblings': float(rel),
            'dominant_width_pct_vs_siblings': float(width_rel),
            'var_ratio_pct_dims_dom_wider': float((var_ratio > 1).mean() * 100),
        },
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved: {OUT_PATH}')


if __name__ == '__main__':
    main()
