"""
B' — re-run T6 reach paired cross-seed eval with per-step action AND
per-step state logging, so the v11 paper can compute mechanism (b)
action smoothness and mechanism (c) trajectory length.

This is a focused subset of compose_paired_eval.py: only the T6
\\textsc{reach} swap matrix (4 primaries x 4 swaps = 16 configs),
N=30 episodes per config, framework phase boundary at step 62.

Output: 16 .npz files in exp/results/states_t6_reach_full/, each
with per-step actions and per-step states for the reach phase.

Run:

    cd <repo>
    .venv/bin/python -u eval_b_prime_with_actions.py

Wall time: ~30 min on RTX 5090 (16 configs x 30 eps x 62 steps).
"""

import os, sys, json, random
import numpy as np
import torch
import yaml

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMEWORK_DIR = os.environ.get(
    'FRAMEWORK_DIR',
    os.path.normpath(os.path.join(REPO_DIR, '..', 'capability-evolution'))
)
sys.path.insert(0, FRAMEWORK_DIR)

from envs import TaskSuite
from agent.ecm import ECM, ECMRegistry, ECMDescriptor

CKPT_DIR = os.path.join(FRAMEWORK_DIR, 'results', 'checkpoints')
OUT_DIR = os.path.join(REPO_DIR, 'results', 'states_t6_reach_full')
os.makedirs(OUT_DIR, exist_ok=True)

TASK_NAME = 'T6_TwoArmPegInHole'
SEEDS = [42, 7, 123, 2024]
PHASES = ['reach', 'grasp', 'lift', 'place']
N_EPISODES = 30
EPISODE_SEEDS = list(range(10000, 10030))
PHASE_END_STEP = 62  # framework's reach-phase boundary


def log(msg):
    from datetime import datetime
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_config():
    with open(os.path.join(FRAMEWORK_DIR, 'configs', 'default.yaml')) as f:
        return yaml.safe_load(f)


def get_ckpt_dir(seed):
    if seed == 42:
        return os.path.join(CKPT_DIR, 'exp2_T6_TwoArmPegInHole_ours')
    return os.path.join(CKPT_DIR, f't6_multiseed_seed_{seed}_ours')


def load_ecm_state(seed):
    weights_path = os.path.join(get_ckpt_dir(seed), 'ecm_weights.pt')
    return torch.load(weights_path, weights_only=False)


def build_registry(state_per_phase, obs_dim, act_dim, cfg):
    reg = ECMRegistry()
    for phase in PHASES:
        desc = ECMDescriptor(
            name=f'ecm_{phase}', description=f'ECM for {phase}',
            phase=phase, input_dim=obs_dim, output_dim=act_dim,
        )
        ecm = ECM(desc,
                  hidden_dims=cfg['ecm']['hidden_dims'],
                  rollback_threshold=cfg['ecm']['rollback_threshold'])
        reg.register(ecm)
    for ecm in reg.active_ecms():
        ecm_name = f'ecm_{ecm.descriptor.phase}'
        src = state_per_phase[ecm.descriptor.phase]
        if ecm_name in src:
            ecm.network.load_state_dict(src[ecm_name]['state_dict'])
            ecm.version = src[ecm_name].get('version', 0)
    return reg


def evaluate_one_config(registry, env, primary, swap):
    """Run 30 paired episodes; save per-step action + per-step state for
    the reach phase only (steps 0..PHASE_END_STEP-1)."""
    horizon = env._env.horizon if hasattr(env, '_env') else 500
    n_phases = 8
    steps_per_phase = horizon // n_phases
    active_ecms = registry.active_ecms()
    schedule = []
    for i in range(n_phases):
        ecm = active_ecms[i % len(active_ecms)]
        start = i * steps_per_phase
        end = (i + 1) * steps_per_phase if i < n_phases - 1 else horizon
        schedule.append((start, end, ecm))

    actions_seq = []   # (N, T_reach, act_dim)
    states_seq = []    # (N, T_reach, obs_dim)
    phase_end_obs = [] # (N, obs_dim)
    successes = 0

    for ep_seed in EPISODE_SEEDS:
        np.random.seed(ep_seed); random.seed(ep_seed); torch.manual_seed(ep_seed)
        obs, _ = env.reset()
        done = False; step = 0; env_info = {}
        actions_this_ep = []
        states_this_ep = []
        while not done:
            current_ecm = schedule[-1][2]
            for start, end, ecm in schedule:
                if start <= step < end:
                    current_ecm = ecm
                    break
            action, _ = current_ecm.get_action(obs, deterministic=True)
            if step < PHASE_END_STEP:
                actions_this_ep.append(np.asarray(action, dtype=np.float32))
                states_this_ep.append(np.asarray(obs, dtype=np.float32))
            obs, reward, terminated, truncated, env_info = env.step(action)
            done = terminated or truncated
            step += 1
        actions_seq.append(np.stack(actions_this_ep))   # (T_reach, act_dim)
        states_seq.append(np.stack(states_this_ep))     # (T_reach, obs_dim)
        phase_end_obs.append(states_this_ep[-1])
        if env_info.get('success', False):
            successes += 1

    return {
        'actions_seq': np.stack(actions_seq),       # (N, T_reach, act_dim)
        'states_seq': np.stack(states_seq),         # (N, T_reach, obs_dim)
        'phase_end_obs': np.stack(phase_end_obs),   # (N, obs_dim)
        'episode_seeds': np.array(EPISODE_SEEDS, dtype=np.int64),
        'phase_end_step': PHASE_END_STEP,
        'success_rate': successes / N_EPISODES * 100.0,
    }


def main():
    cfg = load_config()
    log("Loading 4 ECM checkpoints (T6, seeds 42/7/123/2024)...")
    states = {s: load_ecm_state(s) for s in SEEDS}

    suite = TaskSuite(
        robot=cfg['env']['robot'], controller=cfg['env']['controller'],
        horizon=cfg['env']['horizon'],
        reward_shaping=cfg['env']['reward_shaping'],
        object_pos_noise=cfg['env']['object_pos_noise'],
        obs_noise_std=cfg['env']['obs_noise_std'],
        actuation_failure_prob=cfg['env']['actuation_failure_prob'],
        force_limit=cfg['runtime']['force_limit'],
    )
    task = suite.get_task(TASK_NAME)
    env = suite.make_env(task)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    log(f"obs_dim={obs_dim}, act_dim={act_dim}")

    try:
        for primary in SEEDS:
            for swap in SEEDS:
                # build per-phase state map: reach=swap, others=primary
                state_per_phase = {
                    'reach': states[swap],
                    'grasp': states[primary],
                    'lift':  states[primary],
                    'place': states[primary],
                }
                registry = build_registry(state_per_phase, obs_dim, act_dim, cfg)
                log(f"Eval primary={primary}, swap={swap} ...")
                result = evaluate_one_config(registry, env, primary, swap)
                out_path = os.path.join(OUT_DIR, f'reach_primary{primary}_swap{swap}_full.npz')
                np.savez_compressed(out_path, **{
                    k: v for k, v in result.items() if k != 'success_rate'
                })
                log(f"  saved {out_path} | SR = {result['success_rate']:.1f}%")
    finally:
        try: env.close()
        except Exception: pass

    log("All 16 configs done. Run scp on results/states_t6_reach_full/ → mac.")


if __name__ == '__main__':
    main()
