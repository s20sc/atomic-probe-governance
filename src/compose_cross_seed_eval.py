"""
Cross-Seed Composition Evaluation
====================================================================

Tests "composition integrity under skill version mismatch" using existing
T6 multi-seed ECM checkpoints from the simulation framework. No new training needed.

Setup:
- 4 seeds × 4 ECMs = 16 ECM versions of T6_TwoArmPegInHole
- For each (primary_seed, swap_seed) cell:
    reach ECM   from primary_seed
    grasp ECM   from swap_seed       ← only this varies (configurable)
    lift  ECM   from primary_seed
    place ECM   from primary_seed
- Diagonal (primary == swap)         = same-seed baseline
- Off-diagonal                       = "one skill from a different version"

Total: 4×4 = 16 cells, 30 episodes/cell, ~3-4 hours wall time.

Reuses the simulation framework (envs / agent / runtime) but stores all
outputs under this repo's tree:

    atomic-probe-governance/
      src/compose_cross_seed_eval.py    <- this script
      compose_cross_seed_eval.py        ← this script
      results/
        composition/
          cross_seed_grasp_swap.json    ← raw data per cell
          cross_seed_grasp_swap.log     ← training/eval log
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict

import numpy as np
import torch

# This file lives in <repo>/src/; the simulation framework is expected
# at $FRAMEWORK_DIR (defaults to a sibling capability-evolution/).
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMEWORK_DIR = os.environ.get(
    'FRAMEWORK_DIR',
    os.path.normpath(os.path.join(REPO_DIR, '..', 'capability-evolution'))
)

# Make the framework's library code importable
sys.path.insert(0, FRAMEWORK_DIR)

import yaml
from envs import TaskSuite
from agent.ecm import ECM, ECMRegistry, ECMDescriptor

# Framework checkpoint storage (read-only from this repo's perspective)
CKPT_DIR = os.path.join(FRAMEWORK_DIR, 'results', 'checkpoints')

# This repo's output directory
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


def get_ckpt_dir(seed):
    """Return the framework's checkpoint dir for T6 ours of given seed.
    seed=42 lives in exp2_T6_ours; seeds 7/123/2024 in t6_multiseed_seed_X_ours.
    """
    if seed == 42:
        return os.path.join(CKPT_DIR, 'exp2_T6_TwoArmPegInHole_ours')
    return os.path.join(CKPT_DIR, f't6_multiseed_seed_{seed}_ours')


def load_ecm_state(seed):
    """Load the saved {ecm_name: state_dict, version} dict for a seed."""
    d = get_ckpt_dir(seed)
    weights_path = os.path.join(d, 'ecm_weights.pt')
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Missing {weights_path}")
    return torch.load(weights_path, weights_only=False)


def build_registry_from_states(state_per_phase: Dict[str, Dict],
                                obs_dim: int, act_dim: int, cfg: Dict):
    """Build an ECMRegistry where each ECM's weights come from a (possibly
    different) source's state_dict.

    state_per_phase: {'reach': state_dict_for_reach, 'grasp': ..., ...}
    """
    registry = ECMRegistry()
    for phase in ECM_PHASES:
        desc = ECMDescriptor(
            name=f'ecm_{phase}',
            description=f'ECM for {phase}',
            phase=phase,
            input_dim=obs_dim,
            output_dim=act_dim,
        )
        ecm = ECM(desc,
                  hidden_dims=cfg['ecm']['hidden_dims'],
                  rollback_threshold=cfg['ecm']['rollback_threshold'])
        registry.register(ecm)

    for ecm in registry.active_ecms():
        phase = ecm.descriptor.phase
        ecm_name = f'ecm_{phase}'
        src_states = state_per_phase[phase]
        if ecm_name in src_states:
            sd = src_states[ecm_name]['state_dict']
            ecm.network.load_state_dict(sd)
            ecm.version = src_states[ecm_name].get('version', 0)
    return registry


def evaluate_composition(registry, env, num_episodes, task_ecm_steps=8):
    """Phase-based eval matching the framework's evaluator: divide horizon into
    ecm_steps phases, cycle through 4 ECMs."""
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

    successes = 0
    total_reward = 0.0
    ep_rewards = []
    for ep_idx in range(num_episodes):
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
        ep_rewards.append(ep_reward)
        if env_info.get('success', False):
            successes += 1
        total_reward += ep_reward

    return {
        'success_rate': successes / num_episodes * 100,
        'avg_reward': total_reward / num_episodes,
        'reward_variance': float(np.var(ep_rewards)),
        'n_episodes': num_episodes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-episodes', type=int, default=30)
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    parser.add_argument('--swap-phase', type=str, default='grasp',
                        choices=ECM_PHASES,
                        help='Which ECM phase varies independently in the matrix.')
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    out_name = args.out or f'cross_seed_{args.swap_phase}_swap.json'
    seeds = args.seeds
    cfg = load_config()
    log(f"Cross-seed composition eval: seeds={seeds}, swap_phase={args.swap_phase}, "
        f"n_ep={args.num_episodes}")
    log(f"Framework checkpoint dir:  {CKPT_DIR}")
    log(f"Repo output dir:            {OUT_DIR}")

    # Pre-load all ECM state dicts
    states = {}
    for s in seeds:
        try:
            states[s] = load_ecm_state(s)
            log(f"  loaded ECM states for seed={s}")
        except Exception as e:
            log(f"  ERROR loading seed={s}: {e}")
            return

    # Build T6 env once
    suite = TaskSuite(
        robot=cfg['env']['robot'], controller=cfg['env']['controller'],
        horizon=cfg['env']['horizon'],
        reward_shaping=cfg['env']['reward_shaping'],
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
        'task': TASK_NAME,
        'swap_phase': args.swap_phase,
        'seeds': seeds,
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

    total_cells = len(seeds) * len(seeds)
    cell_idx = 0
    t_start = time.time()

    for primary in seeds:
        for swap in seeds:
            cell_idx += 1
            cell_key = f'primary={primary},swap={swap}'
            if cell_key in results.get('cells', {}):
                log(f"  [{cell_idx}/{total_cells}] {cell_key}: already done, skipping")
                continue

            log(f"  [{cell_idx}/{total_cells}] {cell_key}: evaluating...")
            t0 = time.time()

            state_per_phase = {}
            for phase in ECM_PHASES:
                if phase == args.swap_phase:
                    state_per_phase[phase] = states[swap]
                else:
                    state_per_phase[phase] = states[primary]

            registry = build_registry_from_states(state_per_phase, obs_dim, act_dim, cfg)
            metrics = evaluate_composition(registry, env, args.num_episodes,
                                            task_ecm_steps=task.ecm_steps)
            elapsed = time.time() - t0
            results.setdefault('cells', {})[cell_key] = {
                **metrics,
                'primary_seed': primary,
                'swap_seed': swap,
                'is_diagonal': primary == swap,
                'wall_seconds': elapsed,
            }
            log(f"    success={metrics['success_rate']:.1f}%  "
                f"reward={metrics['avg_reward']:.1f}  "
                f"({elapsed:.0f}s)")

            results['updated_at'] = datetime.now().isoformat(timespec='seconds')
            with open(out_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)

    env.close()
    total_time = time.time() - t_start
    log(f"\nFinished {total_cells} cells in {total_time/60:.1f} min")
    log(f"Saved: {out_path}")

    # Print headline matrix
    print("\n" + "=" * 70)
    print(f"COMPOSITION MATRIX — {TASK_NAME}, swap={args.swap_phase}")
    print("=" * 70)
    print(f"{'primary\\swap':>15s}", end='')
    for s in seeds:
        print(f" {f'seed={s}':>10s}", end='')
    print()
    for primary in seeds:
        print(f"{f'seed={primary}':>15s}", end='')
        for swap in seeds:
            cell = results['cells'].get(f'primary={primary},swap={swap}', {})
            sr = cell.get('success_rate', None)
            if sr is None:
                print(f" {'—':>10s}", end='')
            else:
                marker = '*' if primary == swap else ' '
                print(f" {sr:>8.1f}%{marker}", end='')
        print()

    diag = []
    off_diag = []
    for k, v in results['cells'].items():
        if v.get('is_diagonal'):
            diag.append(v['success_rate'])
        else:
            off_diag.append(v['success_rate'])
    if diag and off_diag:
        print(f"\nDiagonal mean (matched):    {np.mean(diag):.1f}% ± {np.std(diag):.1f}")
        print(f"Off-diagonal mean (swapped): {np.mean(off_diag):.1f}% ± {np.std(off_diag):.1f}")
        print(f"Δ (degradation):             {np.mean(diag) - np.mean(off_diag):+.1f} pp")


if __name__ == '__main__':
    main()
