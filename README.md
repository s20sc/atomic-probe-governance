# Atomic-Probe Governance for Skill Updates in Compositional Robot Policies

[![arXiv](https://img.shields.io/badge/arXiv-2604.26689-b31b1b.svg)](https://arxiv.org/abs/2604.26689)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Release](https://img.shields.io/badge/release-v1.0--arxiv-blue)](https://github.com/s20sc/atomic-probe-governance/releases/tag/v1.0-arxiv)

Code and data for reproducing the experiments of *"Atomic-Probe
Governance for Skill Updates in Compositional Robot Policies"*
(Qin et al., 2026).

> **Paper**: <https://arxiv.org/abs/2604.26689>

## Abstract (TL;DR)

We study **composition stability** under skill-update events in
compositional robot policies. Four findings:

1. On the dual-arm peg-in-hole task **T6**, swapping a single
   high-quality phase ECM (the *dominant-skill effect*) shifts
   composition success rate by up to **50 percentage points** —
   a 22.9pp gap with full revalidation that is statistically
   significant under both McNemar exact-binomial ($p=0.013$) and
   cluster-permutation ($p=0.018$, respecting ECM-level dependence).
2. On the saturated single-arm pick task **T1**, every (seed, phase)
   ECM achieves 100% atomic success rate, so the effect is by
   construction undefined — a clean boundary case.
3. Off-policy behavioral-distance metrics fail to identify the
   dominant ECM: on T6 it ranks 7/16 by mean pairwise L² distance,
   $P(\mathrm{rank}\le 7) = 0.44$ under a uniform-rank null.
4. We propose the **atomic-quality probe** (per-skill, zero
   per-decision cost) and a **Hybrid Selector** that combines
   probes with selective composition revalidation; on 48 T6 update
   events the Hybrid$(m{=}10)$ achieves 75% oracle match at 45.8%
   FullReval cost.

We additionally measure three candidate sub-mechanisms (hand-off
state coverage, action smoothness, trajectory length) and find that
the dominant ECM is not the absolute outlier on any single channel,
suggesting the robustness asymmetry operates through interaction
rather than any single behavioral signal.

## Repository layout

```
atomic-probe-governance/
├── src/                          17 Python scripts
│   ├── train_t3_t4_multiseed.py        multi-seed RL training (T1/T3/T4)
│   ├── train_t2_scout.py               T2 scout-with-snapshots training
│   ├── train_t3_reshaped.py            A2 reward-reshape training (§10 Limitations)
│   ├── compute_atomic_baseline.py      single-ECM atomic SR per (seed, phase)
│   ├── compute_behavioral_distance.py  pairwise off-policy distance (§7)
│   ├── compose_paired_eval.py          paired cross-seed swap (Table 3)
│   ├── compose_subset_swap.py          subset swap matrix (Table 2)
│   ├── compose_cross_seed_eval.py      bare cross-seed composition
│   ├── algo_compare_v2.py              Hybrid Selector benchmark (§8)
│   ├── algo_compare_selectors.py       baseline selectors (§8)
│   ├── bootstrap_cis_table6.py         Wilson CI helper for Table 6
│   ├── b_handoff_state_analysis.py     mechanism (a) phase-end state probe
│   ├── eval_b_prime_with_actions.py    per-step actions+states rollout
│   ├── compute_b_prime_mechanisms.py   mechanism (b) smoothness + (c) trajectory length
│   ├── generate_figures.py             reproduce all paper figures
│   ├── check_atomic_sr_for_ckpt.py     per-checkpoint atomic-SR diagnostic
│   └── check_t125_atomic_sr.py         T1/T2/T5 diagnostic on seed=42
├── scripts/                      7 bash runners (auto-retry, sequential)
└── data/                         18 JSON + txt evaluation outputs
    ├── atomic/                   per-(seed,phase) atomic SR (T1, T6)
    ├── paired/                   T6 paired cross-seed swap (4 phases)
    ├── subset/                   T6 subset swap matrix
    ├── behavioral_distance/      T6 pairwise L² distance
    ├── algo_compare/             cross-event selector benchmark
    ├── mechanism_probes/         (a) hand-off + (b)(c) smoothness/trajectory
    ├── replication/              B' independent re-run (robustness check)
    ├── scaling_attempts/         T3 longer-schedule + reward-reshape negatives
    └── statistical_analysis/     McNemar / Spearman / cluster-perm / Holm-B
```

## Dependencies

This repository depends on a sibling **simulation framework** that
provides the `agent.ecm`, `envs`, and `run_experiment` modules
(robosuite/MuJoCo Panda environment + SAC training loop):

> **<https://github.com/s20sc/capability-evolution>**

Clone both repos as siblings:

```bash
git clone https://github.com/s20sc/capability-evolution.git
git clone https://github.com/s20sc/atomic-probe-governance.git
```

Resulting layout:

```
your-workspace/
├── capability-evolution/      <- simulation framework
│   ├── agent/
│   ├── envs/
│   ├── run_experiment.py
│   └── configs/default.yaml
└── atomic-probe-governance/   <- this repo
```

If your local clone of the framework is named differently, override:

```bash
export FRAMEWORK_DIR=/path/to/capability-evolution
```

(every Python script reads `FRAMEWORK_DIR` from the env var; default
is the sibling `../capability-evolution`).

### Python environment

Tested on Python 3.10 / Ubuntu 22.04 / NVIDIA RTX 5090. Core packages:

```
torch>=2.0
numpy
scipy
matplotlib
pyyaml
robosuite>=1.4
mujoco>=2.3
```

The framework repo provides a complete `requirements.txt`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r ../capability-evolution/requirements.txt
```

## Reproducing the paper

### Option A — re-render figures from shipped data (≈ 1 minute)

The `data/` folder contains all evaluation outputs the paper depends
on. With the Python env active:

```bash
mkdir -p figures
python3 src/generate_figures.py --data-root data/ --out-dir figures/
```

Produces `fig1_t6_atomic_sr.{pdf,png}`, `fig4_behavioral_distance.{pdf,png}`,
`fig5_algo_pareto.{pdf,png}` under `figures/`.

### Option B — full reproduction from scratch (multi-day GPU)

| Step | Script | Wall time | Output |
|------|--------|-----------|--------|
| Train T1 multi-seed (4 seeds × 15 iter) | `bash scripts/auto_t1_multiseed.sh` | ~12 h | `data/atomic/atomic_T1_Pick.json` |
| Train T6 multi-seed (4 seeds × 20 iter) | (use framework's training entry-point) | ~24 h | T6 ECM checkpoints |
| T6 atomic baseline | `python src/compute_atomic_baseline.py --num-episodes 30` | ~25 min | `data/atomic/atomic_T6_TwoArmPegInHole.json` |
| T6 behavioral distance | `python src/compute_behavioral_distance.py` | ~5 min | `data/behavioral_distance/behavioral_distance_T6.json` |
| T6 paired cross-seed (×4 phases) | `bash scripts/run_followup_experiments.sh` | ~95 min | `data/paired/paired_T6_*_swap.json` |
| T6 subset swap | `python src/compose_subset_swap.py --num-episodes 30` | ~25 min | `data/subset/subset_T6_TwoArmPegInHole.json` |
| Selector benchmark | `python src/algo_compare_v2.py` | <1 min | `data/algo_compare/algo_compare_v2.json` |
| Mechanism (a) hand-off probe | `python src/b_handoff_state_analysis.py` | ~30 min | `data/mechanism_probes/b_handoff_state_analysis.json` |
| Per-step rollout (for mechanisms b, c) | `python src/eval_b_prime_with_actions.py` | ~30 min | `states_t6_reach_full/*.npz` (16 files) |
| Mechanisms (b) + (c) analysis | `python src/compute_b_prime_mechanisms.py` | <1 min | `data/mechanism_probes/b_prime_mechanism_results.json` |
| **Limitations side-experiments**: |||
| A1 — T3 longer schedule (default reward, negative) | `python src/train_t3_t4_multiseed.py --tasks T3_Stack --seeds 2024 --iterations 30` | ~6 h | `data/scaling_attempts/t3_seed_2024_fresh30_negative_FINAL.json` |
| A2 — T3 reward reshape (negative) | `python src/train_t3_reshaped.py --seed 42 --iterations 30` | ~6 h | `data/scaling_attempts/t3_reshaped_seed_42_FINAL.json` |

Every script is **resumable** — JSON outputs are written per-cell, so
re-running a partially-completed run skips finished cells.

## Data → paper-claim map

| Paper claim / table / figure | Data file |
|---|---|
| Table 1 (T6 atomic per-cell SR) | `data/atomic/atomic_T6_TwoArmPegInHole.json` |
| Table 2 (T6 subset-swap dominance) | `data/subset/subset_T6_TwoArmPegInHole.json` |
| Table 3 (T6 reach paired matrix) | `data/paired/paired_T6_reach_swap.json` |
| Table 4 (T1 saturation matrix) | `data/atomic/atomic_T1_Pick.json` |
| Figure 4 (12-panel L² heatmap) | `data/behavioral_distance/behavioral_distance_T6.json` |
| Table 5 (cross-task L² summary) | `data/behavioral_distance/behavioral_distance_T6.json` |
| Table 6 (selector benchmark) | `data/algo_compare/algo_compare_v2.json` |
| §6 permutation rank test | derived from `data/atomic/` + `data/behavioral_distance/` |
| §7.3 McNemar + cluster-perm | derived from `data/algo_compare/algo_compare_v2.json` |
| §app:extdisc 3-mechanism table | `data/mechanism_probes/b_handoff_state_analysis.json` + `b_prime_mechanism_results.json` |
| §app:fig1 robustness re-run | `data/replication/t6_reach_paired_sr_b_prime.json` |
| §10 Limitations T3 negatives | `data/scaling_attempts/t3_*_FINAL.json` |
| §4 Holm-Bonferroni + statistical reporting | `data/statistical_analysis/*.txt` |

## Citation

```bibtex
@article{qin2026atomicprobe,
  title={Atomic-Probe Governance for Skill Updates in Compositional Robot Policies},
  author={Qin, Xue and Luan, Simin and See, John and Boukhers, Zeyd and Yang, Cong and Li, Zhijun},
  journal={arXiv preprint arXiv:2604.26689},
  year={2026}
}
```

## License

Code and data in this repository are released under the **Apache
License 2.0** — see [LICENSE](LICENSE).
