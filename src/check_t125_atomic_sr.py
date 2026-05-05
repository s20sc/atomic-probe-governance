"""
Diagnostic: Check T1/T2/T5 atomic SR on existing seed=42 checkpoints
====================================================================

Goal: decide which tasks (besides T6) are worth investing in multiseed
training for the paper. T3/T4 turned out to have 0% atomic SR on all
(seed, phase) cells, so their "Dominant-Skill Effect" results are
reward-shaping artifacts. We need at least 1-2 more tasks where the
atomic policies actually succeed at > 0%.

This script does NOT need new training. It just runs each existing
seed=42 ECM (from exp1_T*/ folders) solo for N episodes on its task
and reports atomic SR per (task, phase) cell.

Decision rule:
  - Task has ≥ 1 phase with atomic SR ≥ 30%  →  worth multiseed training
  - Task has all phases atomic SR < 10%      →  same problem as T3/T4, skip
  - In between                                →  borderline, manual judgment

Total time budget: ~3 tasks × 4 phases × 30 episodes ≈ 30-45 minutes on RTX 5090.
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

# Mirror compute_atomic_baseline.py imports
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMEWORK_DIR = os.environ.get(
    'FRAMEWORK_DIR',
    os.path.normpath(os.path.join(REPO_DIR, '..', 'capability-evolution'))
)
sys.path.insert(0, FRAMEWORK_DIR)

import yaml
from envs import TaskSuite
from agent.ecm import ECM, ECMDescriptor

CKPT_DIR = os.path.join(FRAMEWORK_DIR, 'results', 'checkpoints')
OUT_DIR = os.path.join(REPO_DIR, 'results', 'composition')
os.makedirs(OUT_DIR, exist_ok=True)

# Candidate tasks. Tries both exp1 and exp2 paths.
# T6 already validated (works), so it's not in this list.
# T3/T4 already known broken (0% SR). Skip.
CANDIDATE_TASKS = [
    'T1_Pick',
    'T2_Place',
    'T5_PickPlaceMulti',
]

ECM_PHASES = ['reach', 'grasp', 'lift', 'place']

sys.stdout.reconfigure(line_buffering=True)


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_config():
    with open(os.path.join(FRAMEWORK_DIR, 'configs', 'default.yaml')) as f:
        return yaml.safe_load(f)


def find_seed42_ckpt(task):
    """Try exp1 first, then exp2 (analogous to compute_atomic_baseline.py logic)."""
    for prefix in ['exp1_', 'exp2_']:
        for suffix in ['', '_ours']:
            d = os.path.join(CKPT_DIR, f'{prefix}{task}{suffix}')
            wp = os.path.join(d, 'ecm_weights.pt')
            if os.path.exists(wp):
                return wp, d
    return None, None


def load_ecm_for_phase(weights_path, phase, obs_dim, act_dim, cfg):
    states = torch.load(weights_path, weights_only=False)
    if f'ecm_{phase}' not in states:
        return None  # phase not in this task's checkpoint
    desc = ECMDescriptor(name=f'ecm_{phase}', description=f'ECM for {phase}',
                         phase=phase, input_dim=obs_dim, output_dim=act_dim)
    ecm = ECM(desc, hidden_dims=cfg['ecm']['hidden_dims'],
              rollback_threshold=cfg['ecm']['rollback_threshold'])
    ecm.network.load_state_dict(states[f'ecm_{phase}']['state_dict'])
    return ecm


def evaluate_single_ecm(ecm, env, episode_seeds):
    per_episode = []
    for ep_seed in episode_seeds:
        np.random.seed(ep_seed); random.seed(ep_seed); torch.manual_seed(ep_seed)
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
        'per_episode': per_episode,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-episodes', type=int, default=30)
    parser.add_argument('--episode-seed-base', type=int, default=20000)
    parser.add_argument('--out', type=str, default='diagnostic_t125_atomic_sr.json')
    parser.add_argument('--tasks', nargs='+', default=CANDIDATE_TASKS,
                        help='Tasks to scan')
    args = parser.parse_args()

    cfg = load_config()
    episode_seeds = list(range(args.episode_seed_base,
                               args.episode_seed_base + args.num_episodes))

    log(f"Diagnostic: {len(args.tasks)} tasks × ≤4 phases × {args.num_episodes} episodes")
    log(f"Episode seeds: {episode_seeds[0]}..{episode_seeds[-1]}")

    out_path = os.path.join(OUT_DIR, args.out)
    results = {
        'tasks': args.tasks,
        'phases': ECM_PHASES,
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

    for task_name in args.tasks:
        weights_path, ckpt_dir = find_seed42_ckpt(task_name)
        if weights_path is None:
            log(f"\n!! {task_name}: no checkpoint found in exp1/exp2 — skipping")
            continue
        log(f"\n>> {task_name}: found ckpt at {ckpt_dir}")

        try:
            task = suite.get_task(task_name)
            env = suite.make_env(task)
        except Exception as e:
            log(f"   !! cannot make env for {task_name}: {e}")
            continue
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

        for phase in ECM_PHASES:
            cell_key = f'task={task_name},phase={phase}'
            if cell_key in results['cells']:
                log(f"  [{cell_key}]: skipping (done, SR={results['cells'][cell_key]['success_rate']:.1f}%)")
                continue
            ecm = load_ecm_for_phase(weights_path, phase, obs_dim, act_dim, cfg)
            if ecm is None:
                log(f"  [{cell_key}]: phase not in checkpoint, skip")
                continue
            log(f"  [{cell_key}]: evaluating {args.num_episodes} episodes...")
            t0 = time.time()
            metrics = evaluate_single_ecm(ecm, env, episode_seeds)
            elapsed = time.time() - t0
            results['cells'][cell_key] = {
                **metrics,
                'task': task_name,
                'phase': phase,
                'wall_seconds': elapsed,
            }
            log(f"    SR={metrics['success_rate']:.1f}%  reward={metrics['avg_reward']:.1f}  ({elapsed:.0f}s)")
            results['updated_at'] = datetime.now().isoformat(timespec='seconds')
            with open(out_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)

        env.close()

    log(f"\nSaved: {out_path}")

    # ===== Summary table + verdict =====
    print(f"\n{'='*70}")
    print("ATOMIC SR DIAGNOSTIC — single seed=42 ECM, full episode")
    print(f"{'='*70}")
    print(f"{'task':>22} | " + " | ".join(f"{p:>7}" for p in ECM_PHASES) + " | max")
    print("-" * 70)
    verdicts = {}
    for task_name in args.tasks:
        cells = [results['cells'].get(f'task={task_name},phase={p}', {}) for p in ECM_PHASES]
        srs = [c.get('success_rate') for c in cells]
        srs_filled = [s if s is not None else float('nan') for s in srs]
        max_sr = max((s for s in srs if s is not None), default=None)
        sr_str = " | ".join(f"{s:>6.1f}%" if s is not None else f"{'—':>7}" for s in srs)
        verdict_emoji = "✅" if max_sr and max_sr >= 30 else ("⚠️" if max_sr and max_sr >= 10 else "❌")
        print(f"{task_name:>22} | {sr_str} | max={max_sr:>5.1f}% {verdict_emoji}" if max_sr is not None
              else f"{task_name:>22} | {sr_str} | (no data)")
        verdicts[task_name] = max_sr

    print("\nVERDICT")
    print("-" * 70)
    print("✅ max SR ≥ 30%  →  worth investing in multiseed training")
    print("⚠️  10-30%       →  borderline, manual judgment")
    print("❌ max SR < 10%  →  same training problem as T3/T4, skip\n")

    for t, sr in verdicts.items():
        if sr is None:
            print(f"  {t}: no data (likely env or ckpt issue)")
        elif sr >= 30:
            print(f"  {t}: ✅ GO — train multiseed (4 seeds), then run the swap matrix")
        elif sr >= 10:
            print(f"  {t}: ⚠️  borderline — judge by phase pattern; might be worth fine-tune extension")
        else:
            print(f"  {t}: ❌ SKIP — try a different task or extend training duration significantly")


if __name__ == '__main__':
    main()
