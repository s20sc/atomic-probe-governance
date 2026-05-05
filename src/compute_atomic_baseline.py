"""
Atomic Skill Baseline Evaluation
================================

For each (seed, phase) ∈ 4×4 = 16 ECMs, evaluate that single ECM controlling
the FULL T6 episode (no composition, no phase scheduling). This gives the
"atomic skill quality" axis to compare against composition results.

Output:
    results/composition/atomic_baselines.json
    └── per (seed, phase) ECM: success_rate, avg_reward over 30 episodes
        with paired episode seeds.

Total: 16 ECMs × 30 episodes ≈ 25-30 minutes.
"""
import argparse
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
        # seed=42 lives in exp1/exp2 paths, depending on the task
        if task == 'T6_TwoArmPegInHole':
            return os.path.join(CKPT_DIR, 'exp2_T6_TwoArmPegInHole_ours')
        return os.path.join(CKPT_DIR, f'exp1_{task}')
    short = TASK_SHORT.get(task, task.lower())
    return os.path.join(CKPT_DIR, f'{short}_multiseed_seed_{seed}_ours')


def load_one_ecm(seed, phase, obs_dim, act_dim, cfg, task=None):
    weights_path = os.path.join(get_ckpt_dir(seed, task), 'ecm_weights.pt')
    states = torch.load(weights_path, weights_only=False)
    desc = ECMDescriptor(name=f'ecm_{phase}', description=f'ECM for {phase}',
                         phase=phase, input_dim=obs_dim, output_dim=act_dim)
    ecm = ECM(desc, hidden_dims=cfg['ecm']['hidden_dims'],
              rollback_threshold=cfg['ecm']['rollback_threshold'])
    ecm.network.load_state_dict(states[f'ecm_{phase}']['state_dict'])
    return ecm


def evaluate_single_ecm(ecm, env, episode_seeds):
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
            action, _ = ecm.get_action(obs, deterministic=True)
            obs, reward, terminated, truncated, env_info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            step += 1
        per_episode.append({
            'ep_seed': int(ep_seed),
            'success': bool(env_info.get('success', False)),
            'reward': float(ep_reward),
            'steps': int(step),
        })
    rewards = [ep['reward'] for ep in per_episode]
    successes = sum(1 for ep in per_episode if ep['success'])
    return {
        'success_rate': successes / len(per_episode) * 100,
        'avg_reward': float(np.mean(rewards)),
        'reward_variance': float(np.var(rewards)),
        'per_episode': per_episode,
    }


def main():
    global TASK_NAME
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-episodes', type=int, default=30)
    parser.add_argument('--episode-seed-base', type=int, default=10000)
    parser.add_argument('--out', type=str, default='atomic_baselines.json')
    parser.add_argument('--task', type=str, default=TASK_NAME)
    args = parser.parse_args()
    TASK_NAME = args.task

    cfg = load_config()
    episode_seeds = list(range(args.episode_seed_base,
                                args.episode_seed_base + args.num_episodes))

    log(f"Atomic baseline: 16 ECMs × {args.num_episodes} episodes")
    log(f"Episode seeds: {episode_seeds[0]}..{episode_seeds[-1]}")

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

    out_path = os.path.join(OUT_DIR, args.out)
    results = {
        'task': TASK_NAME, 'seeds': SEEDS, 'phases': ECM_PHASES,
        'episode_seeds': episode_seeds,
        'cells': {},
        'started_at': datetime.now().isoformat(timespec='seconds'),
    }
    if os.path.exists(out_path):
        try:
            results = json.load(open(out_path))
            log(f"  resumed: {len(results.get('cells', {}))} cells already done")
        except Exception:
            pass

    total = len(SEEDS) * len(ECM_PHASES)
    idx = 0
    for seed in SEEDS:
        for phase in ECM_PHASES:
            idx += 1
            cell_key = f'seed={seed},phase={phase}'
            if cell_key in results.get('cells', {}):
                log(f"  [{idx}/{total}] {cell_key}: skipping (done)")
                continue
            log(f"  [{idx}/{total}] {cell_key}: evaluating single-ECM control...")
            t0 = time.time()
            ecm = load_one_ecm(seed, phase, obs_dim, act_dim, cfg, task=TASK_NAME)
            metrics = evaluate_single_ecm(ecm, env, episode_seeds)
            elapsed = time.time() - t0
            results.setdefault('cells', {})[cell_key] = {
                **metrics,
                'seed': seed, 'phase': phase,
                'wall_seconds': elapsed,
            }
            log(f"    success={metrics['success_rate']:.1f}%  "
                f"reward={metrics['avg_reward']:.1f}  ({elapsed:.0f}s)")
            results['updated_at'] = datetime.now().isoformat(timespec='seconds')
            with open(out_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)

    env.close()
    log(f"\nSaved: {out_path}")

    # Print 4×4 matrix
    print(f"\n{'='*70}\nATOMIC SKILL QUALITY MATRIX — {TASK_NAME}\n{'='*70}")
    print(f"{'phase\\seed':>15s}", end='')
    for s in SEEDS:
        print(f" {f'seed={s}':>10s}", end='')
    print()
    for phase in ECM_PHASES:
        print(f"{phase:>15s}", end='')
        for s in SEEDS:
            cell = results['cells'].get(f'seed={s},phase={phase}', {})
            sr = cell.get('success_rate')
            print(f" {sr:>9.1f}%" if sr is not None else f" {'—':>10s}", end='')
        print()


if __name__ == '__main__':
    main()
