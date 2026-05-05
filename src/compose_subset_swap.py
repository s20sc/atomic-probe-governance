"""
Subset Swap Matrix
==================

Tests: how does composition success change as we vary WHICH and HOW MANY
phases are swapped between two seeds?

Design:
- For each (primary_seed, swap_seed) pair, evaluate ALL 16 possible swap-sets
  (subsets of {reach, grasp, lift, place}, including empty set = diagonal).
- 30 paired episodes per cell, same reset seeds across cells (and across pairs).

Hypothesis being tested:
  "If swap_set includes the dominant skill (e.g., seed=2024's reach),
  composition success is high regardless of swap-set size or which other
  phases are included."

Output:
    results/composition/subset_swap_<pair_label>.json
"""
import argparse
import itertools
import json
import os
import random
import sys
import time
from datetime import datetime

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
    weights_path = os.path.join(get_ckpt_dir(seed, task), 'ecm_weights.pt')
    return torch.load(weights_path, weights_only=False)


def build_registry_with_subset(primary_states, swap_states, swap_set,
                                obs_dim, act_dim, cfg):
    """Build registry where phases in `swap_set` use swap_states,
    other phases use primary_states."""
    registry = ECMRegistry()
    for phase in ECM_PHASES:
        desc = ECMDescriptor(name=f'ecm_{phase}', description=f'ECM for {phase}',
                             phase=phase, input_dim=obs_dim, output_dim=act_dim)
        ecm = ECM(desc, hidden_dims=cfg['ecm']['hidden_dims'],
                  rollback_threshold=cfg['ecm']['rollback_threshold'])
        registry.register(ecm)
    for ecm in registry.active_ecms():
        phase = ecm.descriptor.phase
        ecm_name = f'ecm_{phase}'
        src = swap_states if phase in swap_set else primary_states
        ecm.network.load_state_dict(src[ecm_name]['state_dict'])
        ecm.version = src[ecm_name].get('version', 0)
    return registry


def evaluate_paired(registry, env, episode_seeds, task_ecm_steps=8):
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
    for ep_seed in episode_seeds:
        np.random.seed(ep_seed)
        random.seed(ep_seed)
        torch.manual_seed(ep_seed)
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        step = 0
        env_info = {}
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
        per_episode.append({
            'ep_seed': int(ep_seed),
            'success': bool(env_info.get('success', False)),
            'reward': float(ep_reward),
        })
    rewards = [ep['reward'] for ep in per_episode]
    successes = sum(1 for ep in per_episode if ep['success'])
    return {
        'success_rate': successes / len(per_episode) * 100,
        'avg_reward': float(np.mean(rewards)),
        'reward_variance': float(np.var(rewards)),
        'per_episode': per_episode,
    }


def all_subsets():
    """Return all 16 subsets of ECM_PHASES, each as a frozenset."""
    out = []
    for r in range(len(ECM_PHASES) + 1):
        for combo in itertools.combinations(ECM_PHASES, r):
            out.append(frozenset(combo))
    return out


def subset_label(s):
    if not s:
        return '∅'
    return '+'.join(p for p in ECM_PHASES if p in s)


def main():
    global TASK_NAME
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-episodes', type=int, default=30)
    parser.add_argument('--episode-seed-base', type=int, default=10000)
    parser.add_argument('--pairs', type=str, nargs='+',
                        default=['42:2024', '42:7', '123:2024', '7:123'],
                        help='Pairs to evaluate, format primary:swap.')
    parser.add_argument('--task', type=str, default=TASK_NAME)
    parser.add_argument('--out', type=str, default='subset_swap.json')
    args = parser.parse_args()
    TASK_NAME = args.task

    cfg = load_config()
    episode_seeds = list(range(args.episode_seed_base,
                                args.episode_seed_base + args.num_episodes))
    pairs = []
    for p in args.pairs:
        a, b = p.split(':')
        pairs.append((int(a), int(b)))

    log(f"Subset swap eval: pairs={pairs}, n_ep={args.num_episodes}")

    # Pre-load all needed seeds
    needed_seeds = set()
    for p, s in pairs:
        needed_seeds.update([p, s])
    states = {s: load_ecm_state(s, task=TASK_NAME) for s in needed_seeds}
    log(f"  loaded ECM states for seeds {sorted(needed_seeds)}")

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

    subsets = all_subsets()
    log(f"  testing {len(subsets)} swap subsets per pair")

    results = {
        'task': TASK_NAME,
        'pairs': [f'{p}:{s}' for p, s in pairs],
        'subsets': [list(s) for s in subsets],
        'episode_seeds': episode_seeds,
        'cells': {},
        'started_at': datetime.now().isoformat(timespec='seconds'),
    }
    out_path = os.path.join(OUT_DIR, args.out)
    if os.path.exists(out_path):
        try:
            results = json.load(open(out_path))
            log(f"  resumed: {len(results.get('cells', {}))} cells already done")
        except Exception:
            pass

    total = len(pairs) * len(subsets)
    idx = 0
    t_start = time.time()
    for primary, swap in pairs:
        for swap_set in subsets:
            idx += 1
            cell_key = f'p={primary},s={swap},set={subset_label(swap_set)}'
            if cell_key in results.get('cells', {}):
                log(f"  [{idx}/{total}] {cell_key}: skipping (done)")
                continue
            log(f"  [{idx}/{total}] {cell_key}: evaluating...")
            t0 = time.time()
            registry = build_registry_with_subset(
                states[primary], states[swap], swap_set,
                obs_dim, act_dim, cfg,
            )
            metrics = evaluate_paired(registry, env, episode_seeds,
                                       task_ecm_steps=task.ecm_steps)
            elapsed = time.time() - t0
            results.setdefault('cells', {})[cell_key] = {
                **metrics,
                'primary_seed': primary, 'swap_seed': swap,
                'swap_set': sorted(swap_set),
                'k_swapped': len(swap_set),
                'has_reach_swapped': 'reach' in swap_set,
                'wall_seconds': elapsed,
            }
            log(f"    success={metrics['success_rate']:.1f}%  "
                f"reward={metrics['avg_reward']:.1f}  ({elapsed:.0f}s)")
            results['updated_at'] = datetime.now().isoformat(timespec='seconds')
            with open(out_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)

    env.close()
    log(f"\nFinished {total} cells in {(time.time()-t_start)/60:.1f} min")
    log(f"Saved: {out_path}")

    # Print per-pair table grouped by k_swapped (number of phases swapped)
    print(f"\n{'='*70}\nSUBSET SWAP — grouped by # phases swapped\n{'='*70}")
    for primary, swap in pairs:
        print(f"\n--- pair (primary={primary}, swap={swap}) ---")
        print(f"{'k':>3s} {'subset':>30s} {'success':>10s} {'reach∈swap':>12s}")
        rows = []
        for swap_set in subsets:
            label = subset_label(swap_set)
            cell = results['cells'].get(f'p={primary},s={swap},set={label}')
            if cell:
                rows.append((len(swap_set), label, cell['success_rate'],
                             'YES' if 'reach' in swap_set else 'no'))
        rows.sort(key=lambda r: (r[0], r[1]))
        for k, lbl, sr, has_r in rows:
            print(f"{k:>3d} {lbl:>30s} {sr:>9.1f}% {has_r:>12s}")

    # Per-pair: split mean by has_reach_swapped
    print(f"\n{'='*70}\nDOMINANCE TEST: success when reach is swapped vs not\n{'='*70}")
    for primary, swap in pairs:
        print(f"\n--- pair (primary={primary}, swap={swap}) ---")
        with_reach = []
        without_reach = []
        for swap_set in subsets:
            cell = results['cells'].get(f'p={primary},s={swap},set={subset_label(swap_set)}')
            if cell:
                if cell['has_reach_swapped']:
                    with_reach.append(cell['success_rate'])
                else:
                    without_reach.append(cell['success_rate'])
        if with_reach and without_reach:
            print(f"  reach IS swapped (n={len(with_reach):>2d}): "
                  f"{np.mean(with_reach):>5.1f}% ± {np.std(with_reach):>4.1f}")
            print(f"  reach NOT swapped (n={len(without_reach):>2d}): "
                  f"{np.mean(without_reach):>5.1f}% ± {np.std(without_reach):>4.1f}")
            print(f"  Δ = {np.mean(with_reach) - np.mean(without_reach):+.1f} pp")


if __name__ == '__main__':
    main()
