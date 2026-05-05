"""
Atomic SR per-checkpoint Evaluator
===================================

Evaluates atomic SR (single-ECM, full-episode) for each intermediate snapshot
saved by train_t2_scout.py, so we can identify the "Goldilocks zone" where
T2_Place atomic SR is in [40%, 80%] for at least one phase.

Usage (default = scan all iter snapshots in t2_multiseed_seed_42_ours/):
    python check_atomic_sr_for_ckpt.py

Or specify:
    python check_atomic_sr_for_ckpt.py \
        --task T2_Place \
        --ckpt-dir <path/to/snapshots>

Output: results/composition/goldilocks_T2_Place.json + verdict table.
"""
import argparse
import glob
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
from agent.ecm import ECM, ECMDescriptor

CKPT_DIR = os.path.join(FRAMEWORK_DIR, 'results', 'checkpoints')
OUT_DIR = os.path.join(REPO_DIR, 'results', 'composition')
os.makedirs(OUT_DIR, exist_ok=True)

ECM_PHASES = ['reach', 'grasp', 'lift', 'place']

sys.stdout.reconfigure(line_buffering=True)


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_config():
    with open(os.path.join(FRAMEWORK_DIR, 'configs', 'default.yaml')) as f:
        return yaml.safe_load(f)


def load_ecm_for_phase(weights_path, phase, obs_dim, act_dim, cfg):
    states = torch.load(weights_path, weights_only=False)
    if f'ecm_{phase}' not in states:
        return None
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
        done = False; ep_reward = 0.0; step = 0; env_info = {}
        while not done:
            action, _ = ecm.get_action(obs, deterministic=True)
            obs, reward, terminated, truncated, env_info = env.step(action)
            done = terminated or truncated
            ep_reward += reward; step += 1
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
    parser.add_argument('--task', type=str, default='T2_Place')
    parser.add_argument('--ckpt-dir', type=str, default=None,
                        help='Dir containing ecm_weights_iter*.pt snapshots. '
                             'If None, infer from task (assumes seed=42 multiseed dir).')
    parser.add_argument('--num-episodes', type=int, default=30)
    parser.add_argument('--episode-seed-base', type=int, default=20000)
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    short = {'T2_Place': 't2', 'T3_Stack': 't3', 'T4_NutAssembly': 't4',
             'T6_TwoArmPegInHole': 't6', 'T1_Pick': 't1',
             'T5_PickPlaceMulti': 't5'}.get(args.task, args.task.lower())
    if args.ckpt_dir is None:
        args.ckpt_dir = os.path.join(CKPT_DIR, f'{short}_multiseed_seed_42_ours')
    out_name = args.out or f'goldilocks_{args.task}.json'

    snapshots = sorted(glob.glob(os.path.join(args.ckpt_dir, 'ecm_weights_iter*.pt')))
    if not snapshots:
        log(f"!!! No snapshots found in {args.ckpt_dir}")
        return
    log(f"Task: {args.task}")
    log(f"Found {len(snapshots)} snapshots: {[os.path.basename(s) for s in snapshots]}")

    cfg = load_config()
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
    task = suite.get_task(args.task)
    env = suite.make_env(task)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    episode_seeds = list(range(args.episode_seed_base,
                               args.episode_seed_base + args.num_episodes))
    out_path = os.path.join(OUT_DIR, out_name)
    results = {
        'task': args.task,
        'snapshots': [os.path.basename(s) for s in snapshots],
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

    for snap_path in snapshots:
        snap_name = os.path.basename(snap_path)
        # extract iteration count, e.g. ecm_weights_iter05.pt -> 5
        iter_str = snap_name.replace('ecm_weights_iter', '').replace('.pt', '')
        try:
            iter_num = int(iter_str)
        except ValueError:
            iter_num = -1

        for phase in ECM_PHASES:
            cell_key = f'iter={iter_num},phase={phase}'
            if cell_key in results['cells']:
                log(f"  [{cell_key}] skipping (done, SR={results['cells'][cell_key]['success_rate']:.1f}%)")
                continue
            ecm = load_ecm_for_phase(snap_path, phase, obs_dim, act_dim, cfg)
            if ecm is None:
                log(f"  [{cell_key}] phase not in checkpoint, skip")
                continue
            log(f"  [{cell_key}] evaluating {args.num_episodes} eps...")
            t0 = time.time()
            metrics = evaluate_single_ecm(ecm, env, episode_seeds)
            elapsed = time.time() - t0
            results['cells'][cell_key] = {
                **metrics,
                'iter': iter_num,
                'phase': phase,
                'snapshot': snap_name,
                'wall_seconds': elapsed,
            }
            log(f"    SR={metrics['success_rate']:.1f}%  reward={metrics['avg_reward']:.2f}  ({elapsed:.0f}s)")
            results['updated_at'] = datetime.now().isoformat(timespec='seconds')
            with open(out_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
    env.close()
    log(f"\nSaved: {out_path}")

    # Summary table + Goldilocks verdict
    print(f"\n{'='*70}")
    print(f"GOLDILOCKS DIAGNOSTIC — {args.task}")
    print(f"{'='*70}")
    print(f"{'iter':>6} | " + " | ".join(f"{p:>7}" for p in ECM_PHASES) + " | max | zone")
    print('-' * 75)
    iters = sorted(set(c['iter'] for c in results['cells'].values()))
    best_iter = None; best_max = -1
    for it in iters:
        srs = []
        for ph in ECM_PHASES:
            c = results['cells'].get(f'iter={it},phase={ph}', {})
            srs.append(c.get('success_rate'))
        srs_str = ' | '.join(f'{s:>6.1f}%' if s is not None else f'{"—":>7}' for s in srs)
        max_sr = max((s for s in srs if s is not None), default=None)
        if max_sr is None:
            zone = '?'
        elif 40 <= max_sr <= 80:
            zone = '✅ goldilocks'
        elif max_sr < 10:
            zone = '❌ too low'
        elif max_sr > 90:
            zone = '⚠️ saturated'
        elif max_sr < 40:
            zone = '⚠️ early'
        else:
            zone = '⚠️ late'
        print(f"{it:>6} | {srs_str} | {max_sr:>5.1f}% | {zone}")
        # Pick the smallest iter with at least one phase in [40, 80]
        if max_sr is not None and 40 <= max_sr <= 80 and best_iter is None:
            best_iter = it; best_max = max_sr

    print(f"\n{'='*70}")
    if best_iter is not None:
        print(f"✅ Goldilocks I* = {best_iter} (one phase has SR={best_max:.1f}%)")
        print(f"   Recommended: train other 3 seeds to {best_iter} iterations.")
    else:
        all_max = [c.get('success_rate', 0) for c in results['cells'].values()]
        if all_max and max(all_max) < 10:
            print('❌ ABORT: all snapshots have max SR < 10%, atomic broken.')
        elif all_max and min(all_max) > 90:
            print('❌ ABORT: all snapshots saturate, no variation possible.')
        else:
            best_iter_loose = max(iters, key=lambda i: max(
                (results['cells'].get(f'iter={i},phase={ph}', {}).get('success_rate', 0)
                 for ph in ECM_PHASES), default=0))
            print(f'⚠️ Borderline: no clean Goldilocks. Closest: iter={best_iter_loose}')
            print('   Manual review recommended.')


if __name__ == '__main__':
    main()
