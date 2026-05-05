"""
Generate all the paper figures from the collected data.

Figures:
  Fig 1: Atomic skill quality heat-maps (3 tasks × 4×4 matrices)
  Fig 2: Dominant-skill cross-task comparison (bar chart of Δreward)
  Fig 3: Subset swap dominance — primary∈swap_set vs not, per task per pair
  Fig 4: Behavioral distance matrices (3 tasks × 4 phases) — visually uninformative
  Fig 5: Algorithm Pareto frontier (cost vs oracle match)
  Fig 6: Negative control — T4 grasp/place show no effect
  Fig 7: Per-task subset swap details (T3 example)

Output: paper/figs/*.{png,pdf} (override with --out-dir).
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.patches as mpatches

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.dpi': 100,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(REPO_DIR, 'results', 'composition')
FIG_DIR = os.path.join(REPO_DIR, 'results', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

SEEDS = [42, 7, 123, 2024]
PHASES = ['reach', 'grasp', 'lift', 'place']
TASKS = ['T6_TwoArmPegInHole', 'T3_Stack', 'T4_NutAssembly']
TASK_LABELS = {
    'T6_TwoArmPegInHole': 'T6: TwoArmPegInHole',
    'T3_Stack': 'T3: Stack',
    'T4_NutAssembly': 'T4: NutAssembly',
}
DOMINANT = {  # (task, dominant_phase, dominant_seed)
    'T6_TwoArmPegInHole': ('reach', 2024),
    'T3_Stack': ('grasp', 42),
    'T4_NutAssembly': ('reach', 42),
}


def load_atomic(task):
    suffix = '' if task == 'T6_TwoArmPegInHole' else f'_{task}'
    base = 'atomic_baselines' if task == 'T6_TwoArmPegInHole' else f'atomic_{task}'
    p = os.path.join(DATA_DIR, f'{base}.json')
    d = json.load(open(p))
    M = np.zeros((len(PHASES), len(SEEDS)))
    for k, v in d['cells'].items():
        parts = dict(p.split('=') for p in k.split(','))
        s = int(parts['seed']); ph = parts['phase']
        M[PHASES.index(ph), SEEDS.index(s)] = v['avg_reward']
    return M


def load_paired(task, phase):
    base = f'paired_cross_seed_{phase}_swap' if task == 'T6_TwoArmPegInHole' else f'paired_{task}_{phase}_swap'
    p = os.path.join(DATA_DIR, f'{base}.json')
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    M = np.zeros((len(SEEDS), len(SEEDS)))
    for k, v in d['cells'].items():
        parts = dict(p.split('=') for p in k.split(','))
        i = SEEDS.index(int(parts['primary']))
        j = SEEDS.index(int(parts['swap']))
        M[i, j] = v['avg_reward']
    return M


def load_subset(task):
    base = 'subset_swap' if task == 'T6_TwoArmPegInHole' else f'subset_swap_{task}'
    p = os.path.join(DATA_DIR, f'{base}.json')
    return json.load(open(p))


def load_bdist(task):
    base = 'behavioral_distance' if task == 'T6_TwoArmPegInHole' else f'behavioral_distance_{task}'
    p = os.path.join(DATA_DIR, f'{base}.json')
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    out = {}
    for ph, mat in d['distance_per_phase'].items():
        M = np.zeros((len(SEEDS), len(SEEDS)))
        for s_i, row in mat.items():
            i = SEEDS.index(int(s_i))
            for s_j, v in row.items():
                j = SEEDS.index(int(s_j))
                M[i, j] = v['l2']
        out[ph] = M
    return out


# -----------------------------------------------------------------
# Figure 1: Atomic skill quality heat-maps
# -----------------------------------------------------------------
def fig1_atomic_heatmaps():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, task in zip(axes, TASKS):
        M = load_atomic(task)
        # Normalize each task to its own scale (different absolute reward ranges)
        vmax = M.max()
        im = ax.imshow(M, cmap='YlOrRd', aspect='auto', vmin=0, vmax=vmax)
        ax.set_xticks(range(4)); ax.set_xticklabels([f's={s}' for s in SEEDS])
        ax.set_yticks(range(4)); ax.set_yticklabels(PHASES)
        ax.set_title(TASK_LABELS[task])
        # Annotate cells
        dom_phase, dom_seed = DOMINANT[task]
        dom_i = PHASES.index(dom_phase); dom_j = SEEDS.index(dom_seed)
        for i in range(4):
            for j in range(4):
                txt = f'{M[i,j]:.1f}'
                color = 'white' if M[i, j] > vmax * 0.5 else 'black'
                ax.text(j, i, txt, ha='center', va='center', color=color, fontsize=9)
                # Mark dominant cell
                if i == dom_i and j == dom_j:
                    ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                                fill=False, edgecolor='lime',
                                                linewidth=3))
        ax.set_xlabel('seed')
        if ax is axes[0]:
            ax.set_ylabel('phase')
        plt.colorbar(im, ax=ax, fraction=0.046, label='Atomic reward')
    fig.suptitle('Atomic skill quality (single ECM controls full episode)\n'
                 'Green box = task-specific dominant skill', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/fig1_atomic_heatmaps.png')
    plt.savefig(f'{FIG_DIR}/fig1_atomic_heatmaps.pdf')
    plt.close()
    print('fig1_atomic_heatmaps saved')


# -----------------------------------------------------------------
# Figure 2: Cross-task dominance Δ
# -----------------------------------------------------------------
def fig2_cross_task_dominance():
    rows = []
    for task in TASKS:
        dom_phase, dom_seed = DOMINANT[task]
        d = load_subset(task)
        for pair in d['pairs']:
            with_d, without_d = [], []
            for k, v in d['cells'].items():
                if not k.startswith(f'p={pair.split(":")[0]},s={pair.split(":")[1]}'):
                    continue
                sw = v.get('swap_set', [])
                (with_d if dom_phase in sw else without_d).append(v['avg_reward'])
            if with_d and without_d:
                rows.append({
                    'task': task, 'pair': pair, 'dom_phase': dom_phase,
                    'with_d_mean': np.mean(with_d),
                    'without_d_mean': np.mean(without_d),
                    'delta': np.mean(with_d) - np.mean(without_d),
                })

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, task in zip(axes, TASKS):
        task_rows = [r for r in rows if r['task'] == task]
        labels = [r['pair'] for r in task_rows]
        with_means = [r['with_d_mean'] for r in task_rows]
        without_means = [r['without_d_mean'] for r in task_rows]
        x = np.arange(len(labels))
        width = 0.35
        b1 = ax.bar(x - width/2, with_means, width,
                    label=f'{DOMINANT[task][0]} ∈ swap', color='#d62728')
        b2 = ax.bar(x + width/2, without_means, width,
                    label=f'{DOMINANT[task][0]} ∉ swap', color='#1f77b4')
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha='right')
        ax.set_ylabel('Mean composition reward')
        ax.set_title(f'{TASK_LABELS[task]}\nDominant: seed={DOMINANT[task][1]}, {DOMINANT[task][0]}')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        # Add Δ annotation on top
        for i, r in enumerate(task_rows):
            ymax = max(r['with_d_mean'], r['without_d_mean'])
            ax.text(i, ymax + (max(with_means) * 0.05), f'Δ={r["delta"]:+.1f}',
                    ha='center', fontsize=9,
                    color='red' if r['delta'] < 0 else 'green',
                    fontweight='bold')
    fig.suptitle('Dominant-Skill Effect: composition reward changes when '
                 'dominant skill is/is not in swap set', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/fig2_cross_task_dominance.png')
    plt.savefig(f'{FIG_DIR}/fig2_cross_task_dominance.pdf')
    plt.close()
    print('fig2_cross_task_dominance saved')


# -----------------------------------------------------------------
# Figure 3: Subset swap detail (T3 example with k=swap_size)
# -----------------------------------------------------------------
def fig3_subset_detail():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, task in zip(axes, TASKS):
        d = load_subset(task)
        dom_phase, _ = DOMINANT[task]
        # Plot reward by k (swap-set size), colored by whether dominant is swapped
        x_with, y_with, x_without, y_without = [], [], [], []
        for k, v in d['cells'].items():
            sw = v.get('swap_set', [])
            k_size = len(sw)
            if dom_phase in sw:
                x_with.append(k_size); y_with.append(v['avg_reward'])
            else:
                x_without.append(k_size); y_without.append(v['avg_reward'])
        # Jitter x for visibility
        x_with = np.array(x_with) + np.random.uniform(-0.15, 0.15, len(x_with))
        x_without = np.array(x_without) + np.random.uniform(-0.15, 0.15, len(x_without))
        ax.scatter(x_with, y_with, s=80, alpha=0.7, label=f'{dom_phase} ∈ swap',
                   color='#d62728', edgecolor='darkred')
        ax.scatter(x_without, y_without, s=80, alpha=0.7, label=f'{dom_phase} ∉ swap',
                   color='#1f77b4', edgecolor='darkblue')
        ax.set_xlabel('# phases swapped (k)')
        ax.set_ylabel('Composition reward')
        ax.set_title(TASK_LABELS[task])
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle('Composition reward vs swap-set size, partitioned by dominant-skill inclusion',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/fig3_subset_detail.png')
    plt.savefig(f'{FIG_DIR}/fig3_subset_detail.pdf')
    plt.close()
    print('fig3_subset_detail saved')


# -----------------------------------------------------------------
# Figure 4: Behavioral distance matrices (uninformative)
# -----------------------------------------------------------------
def fig4_behavioral_distance():
    fig, axes = plt.subplots(3, 4, figsize=(13, 9))
    for row, task in enumerate(TASKS):
        bd = load_bdist(task)
        if bd is None:
            continue
        for col, phase in enumerate(PHASES):
            ax = axes[row, col]
            M = bd[phase]
            vmax = M.max()
            im = ax.imshow(M, cmap='Blues', vmin=0, vmax=vmax)
            ax.set_xticks(range(4)); ax.set_xticklabels([f's={s}' for s in SEEDS], fontsize=8)
            ax.set_yticks(range(4)); ax.set_yticklabels([f's={s}' for s in SEEDS], fontsize=8)
            for i in range(4):
                for j in range(4):
                    ax.text(j, i, f'{M[i,j]:.1f}', ha='center', va='center',
                            fontsize=7, color='black' if M[i,j] < vmax*0.5 else 'white')
            if row == 0:
                ax.set_title(f'phase: {phase}')
            if col == 0:
                ax.set_ylabel(TASK_LABELS[task].split(':')[0])
    fig.suptitle('Action-L2 distance between (seed_i, seed_j) ECMs — '
                 'uniform across all phases, all tasks → uninformative',
                 fontsize=13, y=0.99)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/fig4_behavioral_distance.png')
    plt.savefig(f'{FIG_DIR}/fig4_behavioral_distance.pdf')
    plt.close()
    print('fig4_behavioral_distance saved')


# -----------------------------------------------------------------
# Figure 5: Algorithm Pareto frontier
# -----------------------------------------------------------------
def fig5_algo_pareto():
    """Pareto frontier for all 3 tasks + average."""
    files = {
        'T6': 'algo_compare_v2.json',
        'T3': 'algo_compare_v2_T3_Stack.json',
        'T4': 'algo_compare_v2_T4_NutAssembly.json',
    }
    all_data = {}
    for label, fname in files.items():
        p = os.path.join(DATA_DIR, fname)
        if os.path.exists(p):
            all_data[label] = json.load(open(p))

    if not all_data:
        return

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    colors = {
        'Naive': '#bbbbbb', 'Freeze': '#7f7f7f',
        'AtomicOnly': '#1f77b4', 'FullReval': '#d62728',
        'Hybrid(margin=10)': '#2ca02c', 'Hybrid(margin=20)': '#9467bd',
        'Hybrid(margin=30)': '#ff7f0e',
    }
    # 3 task panels + 1 average panel
    panels = list(all_data.items()) + [('Average', None)]
    avg_oracle = {}
    avg_cost = {}
    for ax, (label, d) in zip(axes, panels):
        if label == 'Average':
            # Compute average across tasks for each selector
            for sel in colors:
                oracles = []; costs = []
                for task_d in all_data.values():
                    n = task_d['n_events']
                    info = task_d['sweep']['5.0'].get(sel)
                    if info:
                        oracles.append(info['oracle_match'] / n * 100)
                        costs.append(info['fullreval_calls'] / n * 100)
                if oracles:
                    avg_oracle[sel] = np.mean(oracles)
                    avg_cost[sel] = np.mean(costs)
            for sel in avg_oracle:
                marker = 250 if sel.startswith('Hybrid') else 180
                edge = 'black' if 'Hybrid(margin=10)' in sel else 'none'
                edgew = 2.5 if 'Hybrid(margin=10)' in sel else 0
                ax.scatter(avg_cost[sel], avg_oracle[sel], s=marker,
                           c=colors[sel], edgecolor=edge, linewidth=edgew, alpha=0.85)
                ax.annotate(sel.replace('margin=', 'm='), (avg_cost[sel] + 2, avg_oracle[sel]),
                            fontsize=8, va='center')
            ax.set_title('Average across 3 tasks')
        else:
            tau5 = d['sweep']['5.0']
            n = d['n_events']
            for sel, info in tau5.items():
                oracle = info['oracle_match'] / n * 100
                cost = info['fullreval_calls'] / n * 100
                marker = 250 if sel.startswith('Hybrid') else 180
                ax.scatter(cost, oracle, s=marker, c=colors.get(sel, 'gray'),
                           alpha=0.85)
                ax.annotate(sel.replace('margin=', 'm='), (cost + 2, oracle),
                            fontsize=8, va='center')
            ax.set_title(f'{label} (n={n})')
        ax.set_xlabel('FullReval call rate (%)')
        if ax is axes[0]:
            ax.set_ylabel('Oracle match (%)')
        ax.set_xlim(-5, 115); ax.set_ylim(35, 95)
        ax.grid(alpha=0.3)
    fig.suptitle('Algorithm Pareto frontier per task (τ=5pp threshold)',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/fig5_algo_pareto.png')
    plt.savefig(f'{FIG_DIR}/fig5_algo_pareto.pdf')
    plt.close()
    print('fig5_algo_pareto saved (3 tasks + average)')


# -----------------------------------------------------------------
# Figure 6: Negative control — T4 grasp/place show no effect
# -----------------------------------------------------------------
def fig6_negative_control():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    # Plot per-column means for paired matrix on T4 grasp and T4 place
    cases = [('T4_NutAssembly', 'grasp', 'No dominant skill at grasp\n(expect: no effect)'),
             ('T4_NutAssembly', 'place', 'No dominant skill at place\n(expect: no effect)')]
    for ax, (task, phase, title) in zip(axes, cases):
        M = load_paired(task, phase)
        if M is None:
            continue
        col_means = M.mean(axis=0)
        diag = np.diag(M).mean()
        ax.bar(range(4), col_means, color='#1f77b4', alpha=0.7)
        ax.axhline(diag, color='red', linestyle='--',
                   label=f'diagonal mean = {diag:.2f}')
        ax.set_xticks(range(4)); ax.set_xticklabels([f's={s}' for s in SEEDS])
        ax.set_xlabel('swap_seed (column)')
        ax.set_ylabel(f'Mean reward (T4 {phase} swap)')
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        for i, v in enumerate(col_means):
            ax.text(i, v + 0.1, f'{v:.2f}', ha='center', fontsize=10)
    fig.suptitle('Negative control: no swap-seed dominates when there is no dominant skill at this phase',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/fig6_negative_control.png')
    plt.savefig(f'{FIG_DIR}/fig6_negative_control.pdf')
    plt.close()
    print('fig6_negative_control saved')


# -----------------------------------------------------------------
# Figure 7: Per-cell paired matrix heatmap for the dominant phase
# -----------------------------------------------------------------
def fig7_paired_heatmap():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, task in zip(axes, TASKS):
        dom_phase, _ = DOMINANT[task]
        M = load_paired(task, dom_phase)
        if M is None:
            continue
        vmax = M.max()
        im = ax.imshow(M, cmap='RdYlGn', vmin=0, vmax=vmax)
        ax.set_xticks(range(4)); ax.set_xticklabels([f's={s}' for s in SEEDS])
        ax.set_yticks(range(4)); ax.set_yticklabels([f's={s}' for s in SEEDS])
        for i in range(4):
            for j in range(4):
                marker = '★' if i == j else ''
                txt = f'{M[i,j]:.1f}{marker}'
                ax.text(j, i, txt, ha='center', va='center',
                        fontsize=10, color='black')
        ax.set_xlabel(f'swap-in {dom_phase}')
        ax.set_ylabel('primary seed')
        ax.set_title(f'{TASK_LABELS[task]}\n({dom_phase} swap matrix; ★ = matched diagonal)')
        plt.colorbar(im, ax=ax, fraction=0.046, label='Composition reward')
    fig.suptitle('Paired composition reward — dominant phase swap matrix per task',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/fig7_paired_dominant_phase.png')
    plt.savefig(f'{FIG_DIR}/fig7_paired_dominant_phase.pdf')
    plt.close()
    print('fig7_paired_dominant_phase saved')


# -----------------------------------------------------------------
# Figure 8: Atomic quality vs composition gain (linear scaling)
# -----------------------------------------------------------------
def fig8_atomic_vs_gain():
    """For each task, scatter (swap_seed atomic at dom_phase) vs (column mean of dom_phase paired matrix)."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, task in zip(axes, TASKS):
        dom_phase, dom_seed = DOMINANT[task]
        atomic_M = load_atomic(task)
        paired_M = load_paired(task, dom_phase)
        if paired_M is None:
            continue
        # x: atomic reward of swap_seed at dom_phase
        # y: column mean of paired matrix (composition mean when this seed is swap-in)
        x = atomic_M[PHASES.index(dom_phase), :]
        y = paired_M.mean(axis=0)  # column means
        ax.scatter(x, y, s=200, c=['#d62728' if s == dom_seed else '#1f77b4' for s in SEEDS],
                   edgecolor='black', linewidth=1.5)
        for i, s in enumerate(SEEDS):
            ax.annotate(f'seed={s}', (x[i] + (x.max() - x.min()) * 0.02, y[i]), fontsize=9)
        # Linear fit
        if x.std() > 0:
            coef = np.polyfit(x, y, 1)
            xx = np.linspace(x.min(), x.max(), 100)
            ax.plot(xx, np.polyval(coef, xx), 'k--', alpha=0.5,
                    label=f'fit: slope={coef[0]:.2f}')
            ax.legend(fontsize=9)
        ax.set_xlabel(f'Atomic quality of swap-seed at {dom_phase}')
        ax.set_ylabel('Composition column mean')
        ax.set_title(TASK_LABELS[task])
        ax.grid(alpha=0.3)
    fig.suptitle('Composition gain scales with swap-in atomic quality (Finding 2)',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/fig8_atomic_vs_gain.png')
    plt.savefig(f'{FIG_DIR}/fig8_atomic_vs_gain.pdf')
    plt.close()
    print('fig8_atomic_vs_gain saved')


def main():
    print(f'Generating figures in {FIG_DIR}\n')
    fig1_atomic_heatmaps()
    fig2_cross_task_dominance()
    fig3_subset_detail()
    fig4_behavioral_distance()
    fig5_algo_pareto()
    fig6_negative_control()
    fig7_paired_heatmap()
    fig8_atomic_vs_gain()
    print(f'\nAll figures saved to {FIG_DIR}')


if __name__ == '__main__':
    main()
