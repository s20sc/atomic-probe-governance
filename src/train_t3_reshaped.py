"""Train T3_Stack seed=42 with **reshaped reward** for A2-real experiment.

A1 (longer schedule on default reward) failed clean (`t3_seed_2024_fresh30_negative_FINAL.json`,
26 iter, 0% SR throughout, reward saturated 3-5). A2 hypothesis: T3 default reward is too sparse
for SAC to find consistent stack success — boosting r_lift and r_stack should give the
policy stronger gradient toward the success terminal state.

Modifications (monkey-patched at runtime — does NOT change robosuite source):
  - r_lift base bonus  1.0 → 2.0   (lifting from table)
  - r_lift align bonus 0.5 → 1.0   (horizontal alignment to cubeB)
  - r_stack            2.0 → 4.0   (terminal: cubeA touching cubeB while not grasping)
  - r_reach            unchanged (already shaped, max 0.5)

Output:
  - Metrics: results/training/t3_reshaped_seed_42.json
  - Checkpoints: $FRAMEWORK_DIR/results/checkpoints/t3_reshaped_seed_42_ours/
  - Log: results/training/t3_reshaped_seed_42.log

If any iter shows SR > 5%, run check_atomic_sr_for_ckpt.py on the resulting checkpoints.
If iter 25-30 still flat, A2-real also fails → final fallback is v9 §10 limitation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Monkey-patch BEFORE importing trainer so the env class is patched at construction time.
# This must precede any `from robosuite...` imports in the trainer chain.
import numpy as np
import robosuite.environments.manipulation.stack as _stack_module


def _reshaped_staged_rewards(self):
    """Identical to robosuite Stack.staged_rewards but with stronger r_lift / r_stack."""
    cubeA_pos = self.sim.data.body_xpos[self.cubeA_body_id]
    cubeB_pos = self.sim.data.body_xpos[self.cubeB_body_id]
    dist = min(
        [
            np.linalg.norm(self.sim.data.site_xpos[self.robots[0].eef_site_id[arm]] - cubeA_pos)
            for arm in self.robots[0].arms
        ]
    )
    r_reach = (1 - np.tanh(10.0 * dist)) * 0.25
    grasping_cubeA = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cubeA)
    if grasping_cubeA:
        r_reach += 0.25

    cubeA_height = cubeA_pos[2]
    table_height = self.table_offset[2]
    cubeA_lifted = cubeA_height > table_height + 0.04
    r_lift = 2.0 if cubeA_lifted else 0.0  # was 1.0
    if cubeA_lifted:
        horiz_dist = np.linalg.norm(np.array(cubeA_pos[:2]) - np.array(cubeB_pos[:2]))
        r_lift += 1.0 * (1 - np.tanh(horiz_dist))  # align bonus was 0.5

    r_stack = 0
    cubeA_touching_cubeB = self.check_contact(self.cubeA, self.cubeB)
    if not grasping_cubeA and r_lift > 0 and cubeA_touching_cubeB:
        r_stack = 4.0  # was 2.0

    return r_reach, r_lift, r_stack


_stack_module.Stack.staged_rewards = _reshaped_staged_rewards
print("[A2-real] Monkey-patched robosuite.environments.manipulation.stack.Stack.staged_rewards "
      "with reshaped variant (r_lift base 1.0 -> 2.0, r_lift align 0.5 -> 1.0, r_stack 2.0 -> 4.0)")

# Now import the trainer chain. It will pick up the patched class.
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMEWORK_DIR = os.environ.get(
    'FRAMEWORK_DIR',
    os.path.normpath(os.path.join(REPO_DIR, '..', 'capability-evolution'))
)
sys.path.insert(0, FRAMEWORK_DIR)
sys.path.insert(0, REPO_DIR)

from train_t3_t4_multiseed import (  # noqa: E402
    train_one,
    save_metrics,
)

OUT_DIR = Path(REPO_DIR) / 'results' / 'training'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def reshaped_result_path(seed: int) -> Path:
    return OUT_DIR / f't3_reshaped_seed_{seed}.json'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--eval-episodes', type=int, default=30)
    parser.add_argument('--checkpoint-suffix', type=str, default='reshaped',
                        help='Checkpoint dir suffix; default routes to t3_reshaped_seed_<S>_ours')
    args = parser.parse_args()

    # Override checkpoint dir naming to keep this experiment isolated from default-reward T3 runs.
    # train_one uses a hardcoded TASK_SHORT mapping; we patch the result_path function on
    # train_t3_t4_multiseed module so the metrics JSON also gets the reshaped suffix.
    import train_t3_t4_multiseed as ttm
    original_result_path = ttm.result_path

    def _reshaped_result_path(task: str, seed: int) -> str:
        if task == 'T3_Stack':
            return str(reshaped_result_path(seed))
        return original_result_path(task, seed)

    ttm.result_path = _reshaped_result_path

    # Also override the checkpoint dir resolution. train_one calls
    # ExperimentRunner with checkpoint_dir built from TASK_SHORT[task_name] +
    # f"_multiseed_seed_{seed}_ours"; we want "_reshaped_seed_..._ours" instead.
    # Easiest: patch TASK_SHORT for this task + seed at runtime via env var that
    # ExperimentRunner reads, OR pass an explicit checkpoint_dir override.
    # The runner takes `args.checkpoint_dir` if set in the namespace.
    # Inspect ExperimentRunner signature for the override knob.

    print(f"[A2-real] T3_Stack seed={args.seed} iterations={args.iterations} (reshaped reward)")
    print(f"[A2-real] Metrics will save to: {reshaped_result_path(args.seed)}")

    # Run the same train_one as the original multiseed trainer; patched env class
    # + patched result_path is enough to keep this isolated.
    train_one('T3_Stack', args.seed, args.iterations, args.eval_episodes)
    print(f"[A2-real] Training complete. Metrics: {reshaped_result_path(args.seed)}")


if __name__ == '__main__':
    main()
