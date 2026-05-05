"""
Scout Training: T2_Place seed=42, 30 iterations
================================================

Trains T2_Place with
seed=42 for 30 iterations and snapshots an intermediate ECM-weights checkpoint
every 5 iterations so we can later identify the "Goldilocks zone" for T2.

Snapshots are saved alongside the live checkpoint:
  $FRAMEWORK_DIR/results/checkpoints/t2_multiseed_seed_42_ours/
    ecm_weights.pt              (live, overwritten each iteration)
    ecm_weights_iter05.pt       (snapshot)
    ecm_weights_iter10.pt
    ...
    ecm_weights_iter30.pt

Per-iteration metrics also saved to results/training/t2_seed_42.json

After this completes, run check_atomic_sr_for_ckpt.py to evaluate each
intermediate checkpoint's atomic SR per phase.

Wall time: ~10 hours on RTX 5090 (single-arm task, ~17 min/iter).
"""
import argparse
import json
import os
import random
import shutil
import sys
import time
from datetime import datetime

import numpy as np

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMEWORK_DIR = os.environ.get(
    'FRAMEWORK_DIR',
    os.path.normpath(os.path.join(REPO_DIR, '..', 'capability-evolution'))
)
sys.path.insert(0, FRAMEWORK_DIR)

from run_experiment import ExperimentRunner, load_config, log  # noqa: E402

OUT_DIR = os.path.join(REPO_DIR, 'results', 'training')
os.makedirs(OUT_DIR, exist_ok=True)

CKPT_BASE = os.path.join(FRAMEWORK_DIR, 'results', 'checkpoints')

TASK_NAME = 'T2_Place'
SHORT = 't2'
SEED = 42
TOTAL_ITERS = 30
SNAPSHOT_EVERY = 5  # iters

sys.stdout.reconfigure(line_buffering=True)


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def result_path():
    return os.path.join(OUT_DIR, f'{SHORT}_seed_{SEED}.json')


def load_partial():
    p = result_path()
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None


def save_metrics(data):
    p = result_path()
    tmp = p + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, p)


def main():
    log(f"Scout training: {TASK_NAME} seed={SEED} for {TOTAL_ITERS} iterations")
    log(f"Snapshots every {SNAPSHOT_EVERY} iters at iter {list(range(SNAPSHOT_EVERY, TOTAL_ITERS + 1, SNAPSHOT_EVERY))}")
    log(f"Output dir: {OUT_DIR}")

    seed_everything(SEED)
    config = load_config()
    config['seed'] = SEED
    args = argparse.Namespace(
        exp='exp1', task=TASK_NAME,
        iterations=TOTAL_ITERS, eval_episodes=30,
        seed=SEED, config=None,
    )
    runner = ExperimentRunner(config, args)
    task = runner.task_suite.get_task(TASK_NAME)
    env = runner.task_suite.make_env(task)

    # Use a checkpoint label matching the multi-seed naming convention
    ckpt_label = f'{SHORT}_multiseed/seed_{SEED}/ours'
    # The actual checkpoint dir created by ExperimentRunner._save_checkpoint
    ckpt_dir = os.path.join(CKPT_BASE, ckpt_label.replace('/', '_'))
    weights_file = os.path.join(ckpt_dir, 'ecm_weights.pt')

    snapshots_taken = []

    def _on_iter(partial):
        # Save metrics
        save_metrics(partial)
        # Snapshot ECM weights every SNAPSHOT_EVERY iterations
        n = len(partial.get('iterations', []))
        if n > 0 and n % SNAPSHOT_EVERY == 0 and n not in snapshots_taken:
            snapshot_path = os.path.join(ckpt_dir, f'ecm_weights_iter{n:02d}.pt')
            if os.path.exists(weights_file):
                shutil.copy2(weights_file, snapshot_path)
                snapshots_taken.append(n)
                log(f"  📸 snapshot @ iter {n}: {snapshot_path}")

    resume = load_partial()
    if resume and len(resume.get('iterations', [])) > 0:
        # Pre-fill snapshots_taken so we don't overwrite if resuming
        for n in range(SNAPSHOT_EVERY, len(resume['iterations']) + 1, SNAPSHOT_EVERY):
            if os.path.exists(os.path.join(ckpt_dir, f'ecm_weights_iter{n:02d}.pt')):
                snapshots_taken.append(n)
        log(f"  resuming from iter {len(resume['iterations'])}/{TOTAL_ITERS}, "
            f"existing snapshots: {snapshots_taken}")

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
        'seed': SEED, 'task': TASK_NAME,
        'iterations': TOTAL_ITERS, 'eval_episodes': 30,
        'wall_seconds': time.time() - t0,
        'finished_at': datetime.now().isoformat(timespec='seconds'),
        'ckpt_label': ckpt_label,
        'snapshots_taken': snapshots_taken,
    }
    save_metrics(result)
    log(f"\n✅ Scout complete in {result['_meta']['wall_seconds']:.0f}s "
        f"({result['_meta']['wall_seconds']/3600:.1f}h)")
    log(f"Snapshots saved: {snapshots_taken}")
    log(f"\nNext step: python check_atomic_sr_for_ckpt.py")


if __name__ == '__main__':
    main()
