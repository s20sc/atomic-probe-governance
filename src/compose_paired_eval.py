"""
Paired Cross-Seed Composition Evaluation
========================================

Same as compose_cross_seed_eval.py but uses **paired sampling**: all 16
cells in a swap_phase matrix share the same 30 reset states. This makes
per-cell differences purely the effect of the swapped ECM (not reset-state
noise) and enables paired t-tests.

Implementation:
- Pre-generate 30 episode seeds [0, 30) once
- Before each episode within a cell, seed numpy/torch with the episode seed
  so robosuite's object placement randomization is identical across cells

Output:
    results/composition/paired_cross_seed_<phase>_swap.json
    └── per-cell list of (episode_seed, success_bool, reward) — enables
        paired t-test post-hoc.
"""
import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMEWORK_DIR = os.environ.get(
    'FRAMEWORK_DIR',
    os.path.normpath(os.path.join(REPO_DIR, '..', 'capability-evolution'))
)
sys.path.insert(0, FRAMEWORK_DIR)

import yaml
from envs import TaskSuite
from agent.ecm import ECM, ECMRegistry, ECMDescriptor

CKPT_DIR = os.path.join(FRAMEWORK_DIR, 'results', 'checkpoints')
OUT_DIR = os.path.join(REPO_DIR, 'results', 'composition')
os.makedirs(OUT_DIR, exist_ok=True)

TASK_NAME = 'T6_TwoArmPegInHole'
SEEDS = [42, 7, 123, 2024]
ECM_PHASES = ['reach', 'grasp', 'lift', 'place']

sys.stdout.reconfigure(line_buffering=True)


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_config():
    with open(os.path.join(FRAMEWORK_DIR, 'configs', 'default.yaml')) as f:
        return yaml.safe_load(f)


TASK_SHORT = {'T3_Stack': 't3', 'T4_NutAssembly': 't4', 'T6_TwoArmPegInHole': 't6'}

def get_ckpt_dir(seed, task=None):
    task = task or TASK_NAME
    if seed == 42:
        if task == 'T6_TwoArmPegInHole':
            return os.path.join(CKPT_DIR, 'exp2_T6_TwoArmPegInHole_ours')
        return os.path.join(CKPT_DIR, f'exp1_{task}')
    short = TASK_SHORT.get(task, task.lower())
    return os.path.join(CKPT_DIR, f'{short}_multiseed_seed_{seed}_ours')


def load_ecm_state(seed, task=None):
    d = get_ckpt_dir(seed, task)
    weights_path = os.path.join(d, 'ecm_weights.pt')
    return torch.load(weights_path, weights_only=False)


def build_registry_from_states(state_per_phase, obs_dim, act_dim, cfg):
    registry = ECMRegistry()
    for phase in ECM_PHASES:
        desc = ECMDescriptor(
            name=f'ecm_{phase}', description=f'ECM for {phase}',
            phase=phase, input_dim=obs_dim, output_dim=act_dim,
        )
        ecm = ECM(desc, hidden_dims=cfg['ecm']['hidden_dims'],
                  rollback_threshold=cfg['ecm']['rollback_threshold'])
        registry.register(ecm)
    for ecm in registry.active_ecms():
        phase = ecm.descriptor.phase
        ecm_name = f'ecm_{phase}'
        src_states = state_per_phase[phase]
        if ecm_name in src_states:
            ecm.network.load_state_dict(src_states[ecm_name]['state_dict'])
            ecm.version = src_states[ecm_name].get('version', 0)
    return registry


def evaluate_paired(registry, env, episode_seeds, task_ecm_steps=8,
                     phase_end_capture_step=None):
    """Run one episode per seed in episode_seeds. Before each reset,
    seed the global RNGs so robosuite's randomization is reproducible.

    If `phase_end_capture_step` is not None, capture the observation at
    that step index of every episode and return it as `phase_end_obs`
    (shape n_episodes × obs_dim). Used by experiment B (hand-off
    state-distribution probe).
    """
    horizon = env._env.horizon if hasattr(env, '_env') else 500
    n_phases = task_ecm_steps
    steps_per_phase = horizon // n_phases
    active_ecms = registry.active_ecms()
    schedule = []
    for i in range(n_phases):
        ecm = active_ecms[i % len(active_ecms)]
        start = i * steps_per_phase
        end = (i + 1) * steps_per_phase if i < n_phases - 1 else horizon
        schedule.append((start, end, ecm))

    per_episode = []
    phase_end_obs_per_ep = [] if phase_end_capture_step is not None else None
    for ep_seed in episode_seeds:
        # Seed everything so robosuite's reset is reproducible
        np.random.seed(ep_seed)
        random.seed(ep_seed)
        torch.manual_seed(ep_seed)

        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        step = 0
        env_info = {}
        captured_phase_end = None
        while not done:
            current_ecm = schedule[-1][2]
            for start, end, ecm in schedule:
                if start <= step < end:
                    current_ecm = ecm
                    break
            action, _ = current_ecm.get_action(obs, deterministic=True)
            obs, reward, terminated, truncated, env_info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            step += 1
            if (phase_end_capture_step is not None
                    and step == phase_end_capture_step
                    and captured_phase_end is None):
                captured_phase_end = np.asarray(obs, dtype=np.float32).copy()
        if phase_end_obs_per_ep is not None:
            if captured_phase_end is None:
                captured_phase_end = np.asarray(obs, dtype=np.float32).copy()
            phase_end_obs_per_ep.append(captured_phase_end)
        per_episode.append({
            'ep_seed': int(ep_seed),
            'success': bool(env_info.get('success', False)),
            'reward': float(ep_reward),
            'steps': int(step),
        })
    successes = sum(1 for ep in per_episode if ep['success'])
    rewards = [ep['reward'] for ep in per_episode]
    out = {
        'success_rate': successes / len(per_episode) * 100,
        'avg_reward': float(np.mean(rewards)),
        'reward_variance': float(np.var(rewards)),
        'n_episodes': len(per_episode),
        'per_episode': per_episode,
    }
    if phase_end_obs_per_ep is not None:
        out['phase_end_obs'] = np.stack(phase_end_obs_per_ep, axis=0)
    return out


def main():
    global TASK_NAME
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-episodes', type=int, default=30)
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    parser.add_argument('--swap-phase', type=str, default='reach',
                        choices=ECM_PHASES)
    parser.add_argument('--episode-seed-base', type=int, default=10000,
                        help='First episode seed; episode_seeds = base..base+N')
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--task', type=str, default=TASK_NAME)
    parser.add_argument('--save-states', type=str, default=None,
                        help='If set, save phase-end-step obs as <dir>/<swap_phase>_primary{p}_swap{s}.npz')
    args = parser.parse_args()
    TASK_NAME = args.task

    out_name = args.out or f'paired_cross_seed_{args.swap_phase}_swap.json'
    seeds = args.seeds
    cfg = load_config()
    episode_seeds = list(range(args.episode_seed_base,
                                args.episode_seed_base + args.num_episodes))

    log(f"Paired cross-seed eval: seeds={seeds}, swap={args.swap_phase}, "
        f"n_ep={args.num_episodes}, ep_seeds={episode_seeds[0]}..{episode_seeds[-1]}")

    # Pre-load all ECM states
    states = {s: load_ecm_state(s, task=TASK_NAME) for s in seeds}
    log(f"  loaded ECM states for seeds {list(states.keys())}")

    suite = TaskSuite(
        robot=cfg['env']['robot'], controller=cfg['env']['controller'],
        horizon=cfg['env']['horizon'], reward_shaping=cfg['env']['reward_shaping'],
        object_pos_noise=cfg['env']['object_pos_noise'],
        obs_noise_std=cfg['env']['obs_noise_std'],
        actuation_failure_prob=cfg['env']['actuation_failure_prob'],
        force_limit=cfg['runtime']['force_limit'],
        workspace_bounds=cfg['runtime']['workspace_bounds'],
        forbidden_zones=cfg['runtime']['forbidden_zones'],
        seed=cfg['seed'],
    )
    task = suite.get_task(TASK_NAME)
    env = suite.make_env(task)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    results = {
        'task': TASK_NAME, 'swap_phase': args.swap_phase,
        'seeds': seeds, 'episode_seeds': episode_seeds,
        'cells': {},
        'started_at': datetime.now().isoformat(timespec='seconds'),
    }
    out_path = os.path.join(OUT_DIR, out_name)
    if os.path.exists(out_path):
        try:
            results = json.load(open(out_path))
            log(f"  resumed: {len(results.get('cells', {}))} cells already done")
        except Exception:
            pass

    total = len(seeds) * len(seeds)
    idx = 0
    for primary in seeds:
        for swap in seeds:
            idx += 1
            cell_key = f'primary={primary},swap={swap}'
            if cell_key in results.get('cells', {}):
                log(f"  [{idx}/{total}] {cell_key}: skipping (done)")
                continue
            log(f"  [{idx}/{total}] {cell_key}: evaluating...")
            t0 = time.time()
            state_per_phase = {}
            for phase in ECM_PHASES:
                state_per_phase[phase] = states[swap if phase == args.swap_phase else primary]
            registry = build_registry_from_states(state_per_phase, obs_dim, act_dim, cfg)
            phase_end_step = None
            if args.save_states:
                horizon = cfg['env']['horizon']
                steps_per_phase = horizon // task.ecm_steps
                phase_idx = ECM_PHASES.index(args.swap_phase)
                # End-of-phase = first step where the next phase ECM takes over
                phase_end_step = (phase_idx + 1) * steps_per_phase
            metrics = evaluate_paired(registry, env, episode_seeds,
                                       task_ecm_steps=task.ecm_steps,
                                       phase_end_capture_step=phase_end_step)
            phase_end_obs = metrics.pop('phase_end_obs', None)
            elapsed = time.time() - t0
            results.setdefault('cells', {})[cell_key] = {
                **metrics,
                'primary_seed': primary, 'swap_seed': swap,
                'is_diagonal': primary == swap,
                'wall_seconds': elapsed,
            }
            if phase_end_obs is not None:
                os.makedirs(args.save_states, exist_ok=True)
                npz_name = f'{args.swap_phase}_primary{primary}_swap{swap}.npz'
                np.savez(os.path.join(args.save_states, npz_name),
                         phase_end_obs=phase_end_obs,
                         episode_seeds=np.asarray(episode_seeds, dtype=np.int64),
                         phase_end_step=phase_end_step)
            log(f"    success={metrics['success_rate']:.1f}%  "
                f"reward={metrics['avg_reward']:.1f}  ({elapsed:.0f}s)")
            results['updated_at'] = datetime.now().isoformat(timespec='seconds')
            with open(out_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)

    env.close()
    log(f"\nSaved: {out_path}")

    # Print matrix and stats
    print(f"\n{'='*70}\nPAIRED MATRIX — {TASK_NAME}, swap={args.swap_phase}\n{'='*70}")
    print(f"{'primary\\swap':>15s}", end='')
    for s in seeds:
        print(f" {f'seed={s}':>10s}", end='')
    print()
    for primary in seeds:
        print(f"{f'seed={primary}':>15s}", end='')
        for swap in seeds:
            cell = results['cells'].get(f'primary={primary},swap={swap}', {})
            sr = cell.get('success_rate')
            marker = '*' if primary == swap else ' '
            print(f" {sr:>8.1f}%{marker}" if sr is not None else f" {'—':>10s}", end='')
        print()

    diag = [v['success_rate'] for v in results['cells'].values() if v.get('is_diagonal')]
    off = [v['success_rate'] for v in results['cells'].values() if not v.get('is_diagonal')]
    if diag and off:
        print(f"\nDiagonal:   {np.mean(diag):.1f}% ± {np.std(diag):.1f}")
        print(f"Off-diag:   {np.mean(off):.1f}% ± {np.std(off):.1f}")
        print(f"Δ:          {np.mean(diag) - np.mean(off):+.1f} pp")

    # Paired t-test: per-episode success across diagonal vs off-diagonal cells
    # For each (episode_seed, primary), compare diagonal vs mean(off-diagonal)
    try:
        from scipy import stats as _st
        diag_per_ep = []  # success rates aggregated per episode
        off_per_ep = []
        for ep_idx in range(len(episode_seeds)):
            d_succ = []
            o_succ = []
            for primary in seeds:
                for swap in seeds:
                    cell = results['cells'].get(f'primary={primary},swap={swap}')
                    if cell and 'per_episode' in cell:
                        s = int(cell['per_episode'][ep_idx]['success'])
                        if primary == swap:
                            d_succ.append(s)
                        else:
                            o_succ.append(s)
            if d_succ and o_succ:
                diag_per_ep.append(np.mean(d_succ) * 100)
                off_per_ep.append(np.mean(o_succ) * 100)
        if len(diag_per_ep) >= 2:
            t, p = _st.ttest_rel(diag_per_ep, off_per_ep)
            print(f"\nPaired t-test (per-episode diag vs off-diag): t={t:.3f}, p={p:.4f}")
            results['paired_ttest'] = {'t_stat': float(t), 'p_value': float(p),
                                        'n_pairs': len(diag_per_ep)}
            with open(out_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
    except Exception as e:
        print(f"  (paired t-test failed: {e})")


if __name__ == '__main__':
    main()
