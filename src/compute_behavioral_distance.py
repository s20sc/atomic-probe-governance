"""
Behavioral Distance Between ECM Versions
========================================

For each phase, compute pairwise behavioral distance between all 4 seeds'
ECMs of that phase. Distance metrics:

1. Action L2 distance:  ||π_a(s) - π_b(s)||_2 averaged over a held-out
   state batch.
2. Action correlation:  Pearson correlation between action sequences.

Output:
    results/composition/behavioral_distance.json
    └── For each phase: 4×4 distance matrices.

We use a small "probe" rollout to gather ~1000 states (T6 with random ECM
to get diverse states). Each pair of ECMs is then evaluated on these
fixed states to compute action-space divergence.

Total: ~5-10 minutes (no environment rollouts beyond probe).
"""
import json
import os
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
PROBE_EPISODES = 5  # number of probe rollouts to collect states

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


def load_one_ecm(seed, phase, obs_dim, act_dim, cfg, task=None):
    weights_path = os.path.join(get_ckpt_dir(seed, task), 'ecm_weights.pt')
    states = torch.load(weights_path, weights_only=False)
    desc = ECMDescriptor(name=f'ecm_{phase}', description=f'ECM for {phase}',
                         phase=phase, input_dim=obs_dim, output_dim=act_dim)
    ecm = ECM(desc, hidden_dims=cfg['ecm']['hidden_dims'],
              rollback_threshold=cfg['ecm']['rollback_threshold'])
    ecm.network.load_state_dict(states[f'ecm_{phase}']['state_dict'])
    return ecm


def collect_probe_states(env, ecm, n_episodes=5, seed_base=99000):
    """Run a few episodes with the given ECM and collect all observations."""
    states = []
    for ep in range(n_episodes):
        np.random.seed(seed_base + ep)
        torch.manual_seed(seed_base + ep)
        obs, _ = env.reset()
        done = False
        while not done:
            states.append(obs.copy())
            action, _ = ecm.get_action(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    return np.asarray(states, dtype=np.float32)


def compute_distances(ecms_per_seed, probe_states):
    """For each pair (seed_i, seed_j), compute:
       - Mean L2 distance ||π_i(s) - π_j(s)||
       - Mean cosine similarity 1 - cos(π_i(s), π_j(s))
       Returns dict[seed_i][seed_j] -> {l2, cos_dist}
    """
    actions_per_seed = {}
    obs_t = torch.from_numpy(probe_states)
    with torch.no_grad():
        for s, ecm in ecms_per_seed.items():
            mean, _ = ecm.network.forward(obs_t)
            actions_per_seed[s] = mean.detach().cpu().numpy()

    matrix = {}
    for s_i in SEEDS:
        matrix[s_i] = {}
        for s_j in SEEDS:
            a_i = actions_per_seed[s_i]
            a_j = actions_per_seed[s_j]
            l2 = float(np.mean(np.linalg.norm(a_i - a_j, axis=-1)))
            # cosine similarity per state, then 1 - mean_cos
            num = np.sum(a_i * a_j, axis=-1)
            den = np.linalg.norm(a_i, axis=-1) * np.linalg.norm(a_j, axis=-1) + 1e-8
            cos = num / den
            cos_dist = float(1.0 - np.mean(cos))
            matrix[s_i][s_j] = {'l2': l2, 'cos_dist': cos_dist}
    return matrix


def main():
    global TASK_NAME
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default=TASK_NAME)
    parser.add_argument('--out', type=str, default='behavioral_distance.json')
    args = parser.parse_args()
    TASK_NAME = args.task
    cfg = load_config()
    log(f"Behavioral distance for {TASK_NAME}: probing then computing 4×4 distances per phase.")

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

    # Probe with seed=42's reach ECM (any reasonable policy works for state collection)
    log("  collecting probe states...")
    probe_ecm = load_one_ecm(42, 'reach', obs_dim, act_dim, cfg, task=TASK_NAME)
    probe_states = collect_probe_states(env, probe_ecm, n_episodes=PROBE_EPISODES)
    log(f"  collected {len(probe_states)} states")

    # Compute distance matrix per phase
    distance_per_phase = {}
    for phase in ECM_PHASES:
        log(f"  phase={phase}: loading ECMs from all 4 seeds...")
        ecms = {s: load_one_ecm(s, phase, obs_dim, act_dim, cfg, task=TASK_NAME) for s in SEEDS}
        log(f"  phase={phase}: computing 4×4 distance matrix...")
        distance_per_phase[phase] = compute_distances(ecms, probe_states)

    env.close()

    out = {
        'task': TASK_NAME, 'seeds': SEEDS, 'phases': ECM_PHASES,
        'n_probe_states': len(probe_states),
        'distance_per_phase': distance_per_phase,
        'generated_at': datetime.now().isoformat(timespec='seconds'),
    }
    out_path = os.path.join(OUT_DIR, args.out)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    log(f"\nSaved: {out_path}")

    # Print L2 matrix per phase
    for phase in ECM_PHASES:
        print(f"\n{'='*60}\nL2 distance — phase={phase}\n{'='*60}")
        print(f"{'':>10s}", end='')
        for s in SEEDS:
            print(f" {f'seed={s}':>10s}", end='')
        print()
        for s_i in SEEDS:
            print(f"{f'seed={s_i}':>10s}", end='')
            for s_j in SEEDS:
                v = distance_per_phase[phase][s_i][s_j]['l2']
                print(f" {v:>10.4f}", end='')
            print()


if __name__ == '__main__':
    main()
