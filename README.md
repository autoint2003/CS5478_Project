# CS5478 grasp recovery

A MuJoCo Panda recovers a cylinder after a disturbance, then either hands the object to the nominal controller or runs the Precision Oracle before that handoff. Control is a legal 7D action at 20 ms. Runtime observations are 39D. Task success for a new expert is **SUCCESS_NEW**. **SUCCESS_12S** is recorded separately and does not replace it. On `GROUND_SCENE`, a latched **GROUND_DROP** is terminal.

The authoritative Oracle is `data_generation/oracles/precision/state_machine.py`. Plan A, B, and C stage-1 search stays in `data_generation/oracles/plan_{a,b,c}/`. Expert generation has six branches, selected by id:

| Branch | Meaning |
|---|---|
| `A_STABLE_DIRECT_HANDOFF` | Plan A stable grasp, then direct HANDOFF |
| `A_STABLE_PRECISION` | Plan A stable grasp, then Precision |
| `B_STABLE_DIRECT_HANDOFF` | Plan B stable grasp, then direct HANDOFF |
| `B_STABLE_PRECISION` | Plan B stable grasp, then Precision |
| `C_RECATCH_DIRECT_HANDOFF` | Plan C recatch, then direct HANDOFF |
| `C_RECATCH_PRECISION` | Plan C recatch, then Precision |
| `D_PRECISION` | Precision only |

Previous Precision Oracle trajectories are not training data for this Oracle. They remain in `data/unified_experts/` and under `results/`. Frozen first run: `data/experts_v2/` (122). Fallback-enabled recovery set `data/experts_v2_recovered/` (169) is diagnostic only. Official production set: `data/experts_v2_official/` (support-transfer only, no verified_lateral fallback).

## Documents

| Doc | Contents |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Package boundaries, six-branch flow, dataset versions, success definitions |
| [docs/ORACLE_STATE_MACHINE_SPEC.md](docs/ORACLE_STATE_MACHINE_SPEC.md) | Precision state machine and the module that implements each phase |
| [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) | Checkpoints, regression commands, historical experiment locations |

## Setup

Run from the repository root with the project Python environment and MuJoCo. Robot models in `mujoco_menagerie/` are vendored; do not modify them. `assets/` holds the project scenes.

## Commands

```powershell
python -u scripts/generate_experts.py --list-branches
python -u scripts/generate_experts.py --list-ics
python -u scripts/generate_experts.py --all-branches
python -u scripts/validate_dataset.py --manifest data/experts_v2_official/manifest.jsonl
python -u -m pytest -q tests/test_precision_oracle_v23_scheduler.py tests/test_expert_branch_dispatch.py
python -u scripts/replay_experts.py
python -u scripts/train_policy.py --supervised --config config/training/sft.yaml
python -u scripts/evaluate_policy.py --eval --config config/training/unified_transformer.yaml --checkpoint <ckpt>
python -u scripts/render_trajectory.py
```

Generation config: `config/generation/experts_v2.yaml` (seven branches; verified_lateral fallback forced off). Output: `data/experts_v2_official/`. Do not start SFT/GRPO until generation and validation finish.

## Layout

| Path | Role |
|---|---|
| `controllers/`, `envs/`, `sensors/` | Control laws, simulation, sensing |
| `data_generation/` | Oracles, expert pipeline, recording, dataset construction code |
| `evaluation/`, `training/`, `models/` | Success criteria, trainers, policy architectures |
| `scripts/` | The commands above |
| `experiments/` | Experiment code archive (including code relocated from `results/`) |
| `results/` | Experimental outputs only (reports, caches, checkpoints, videos) |
| `data/` | Local manifests, gitignored |
| `coursework/` | Course PDFs |

Historical K49 mining scripts live under `experiments/` and are not the supported workflow.
