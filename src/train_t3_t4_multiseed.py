"""
Train T3_Stack and T4_NutAssembly with seeds {7, 123, 2024}
============================================================

We already have:
  - T3 / T4 with seed = 42 from the framework's exp1
    (checkpoints in $FRAMEWORK_DIR/results/checkpoints/exp1_T3_Stack
     and exp1_T4_NutAssembly)

We need 3 more seeds for each task to enable 4×4 cross-seed composition matrices.

Plan:
  - 6 training runs: T3 × {7, 123, 2024} + T4 × {7, 123, 2024}
  - Each run = 20 iterations × 50K steps × 4 ECMs = ~6.5h at 12 envs
  - Total: ~40 hours wall-time.

Storage:
  - Checkpoints: $FRAMEWORK_DIR/results/checkpoints/<task_short>_multiseed_seed_<S>_ours/
    (matches existing T6 multiseed naming pattern, so all this repo's eval scripts
     can use the same loader.)
  - Metrics: results/training/<task_short>_seed_<S>.json
"""
import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMEWORK_DIR = os.environ.get(
    'FRAMEWORK_DIR',
    os.path.normpath(os.path.join(REPO_DIR, '..', 'capability-evolution'))
)
sys.path.insert(0, FRAMEWORK_DIR)

# Framework runner (we reuse, do not modify)
from run_experiment import ExperimentRunner, load_config, log  # noqa: E402

OUT_DIR = os.path.join(REPO_DIR, 'results', 'training')
os.makedirs(OUT_DIR, exist_ok=True)

TASK_SHORT = {
    'T3_Stack': 't3',
    'T4_NutAssembly': 't4',
    'T6_TwoArmPegInHole': 't6',  # already done; included for completeness
}

DEFAULT_SEEDS = [7, 123, 2024]
DEFAULT_TASKS = ['T3_Stack', 'T4_NutAssembly']

sys.stdout.reconfigure(line_buffering=True)


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def result_path(task: str, seed: int) -> str:
    short = TASK_SHORT.get(task, task.lower())
    return os.path.join(OUT_DIR, f'{short}_seed_{seed}.json')


def is_complete(task: str, seed: int, iterations: int) -> bool:
    p = result_path(task, seed)
    if not os.path.exists(p):
        return False
    try:
        d = json.load(open(p))
        return len(d.get('iterations', [])) >= iterations
    except Exception:
        return False


def load_partial(task: str, seed: int) -> Optional[Dict]:
    p = result_path(task, seed)
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None


def save_metrics(task: str, seed: int, data: Dict):
    p = result_path(task, seed)
    tmp = p + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, p)


def train_one(task_name: str, seed: int, iterations: int, eval_episodes: int):
    log(f"========== TRAINING {task_name}, seed={seed} ==========")
    seed_everything(seed)

    config = load_config()
    config['seed'] = seed
    args = argparse.Namespace(
        exp='exp1', task=task_name,
        iterations=iterations, eval_episodes=eval_episodes,
        seed=seed, config=None,
    )
    runner = ExperimentRunner(config, args)

    task = runner.task_suite.get_task(task_name)
    env = runner.task_suite.make_env(task)

    # Use a checkpoint label that matches the existing T6 multiseed pattern
    # so this repo's eval scripts can load it the same way.
    short = TASK_SHORT.get(task_name, task_name.lower())
    ckpt_label = f'{short}_multiseed/seed_{seed}/ours'

    resume = load_partial(task_name, seed)
    if resume and len(resume.get('iterations', [])) > 0:
        log(f"  resuming from iter {len(resume['iterations'])}/{iterations}")

    def _on_iter(partial):
        save_metrics(task_name, seed, partial)

    t0 = time.time()
    try:
        result = runner._run_evolution(
            env, task,
            label=ckpt_label,
            on_iter_done=_on_iter,
            resume_from=resume,
        )
    finally:
        try:
            env.close()
        except Exception:
            pass

    result['_meta'] = {
        'seed': seed, 'task': task_name,
        'iterations': iterations, 'eval_episodes': eval_episodes,
        'wall_seconds': time.time() - t0,
        'finished_at': datetime.now().isoformat(timespec='seconds'),
        'ckpt_label': ckpt_label,
    }
    save_metrics(task_name, seed, result)
    log(f"  {task_name} seed={seed} DONE in {result['_meta']['wall_seconds']:.0f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', type=str, nargs='+', default=DEFAULT_TASKS)
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--eval-episodes', type=int, default=30)
    args = parser.parse_args()

    log(f"Multi-task multi-seed training")
    log(f"  tasks: {args.tasks}")
    log(f"  seeds: {args.seeds}")
    log(f"  iterations: {args.iterations}, eval: {args.eval_episodes}")
    log(f"  output: {OUT_DIR}")

    total = len(args.tasks) * len(args.seeds)
    idx = 0
    t_start = time.time()
    for task in args.tasks:
        for seed in args.seeds:
            idx += 1
            log(f"\n[{idx}/{total}] === {task}, seed={seed} ===")
            if is_complete(task, seed, args.iterations):
                log(f"  already complete, skipping")
                continue
            train_one(task, seed, args.iterations, args.eval_episodes)

    log(f"\n=== ALL TRAINING DONE in {(time.time()-t_start)/3600:.1f} hours ===")


if __name__ == '__main__':
    main()
