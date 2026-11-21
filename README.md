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

Run every command below from the **repository root** (`cs5478_project/`) with Python **≥ 3.10** (`pyproject.toml`).

```powershell
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install torch   # required by P64 / experiment code; not listed in requirements.txt
```

Robot meshes: clone [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) to `mujoco_menagerie/` at the repo root (gitignored; `assets/panda_torque.xml` loads `../mujoco_menagerie/franka_emika_panda/assets`). Do not edit the vendored tree. Scene XMLs live in `assets/`.

## Run P64 (frozen actor eval / inference)

P64 is the multi-round PPO continuation under `experiments/precision/p64_ppo_multiround/`. The frozen production actor is **P64 step-4**.

| Role | Path (relative to repo root) | Notes |
|---|---|---|
| Actor (required for eval) | `results/experiments/p64_ppo_multiround/checkpoints/P64_step4_actor.pt` | MD5 `9f4c50413535e4c7db3d3a868bc1b6e2` (370227 bytes) |
| Critic (optional for actor-only eval) | `results/experiments/p64_ppo_multiround/checkpoints/P64_step4_critic.pt` | MD5 `95de94e5a5d31ba8acddbfb49a6981cb` (332359 bytes) |

`*.pt` files are **gitignored**. Place the checkpoints at the paths above before running. Paths are defined in `experiments/precision/p64_ppo_multiround/paths.py` (`CKPT_DIR`) and cited in `results/experiments/final_consolidation/FINAL_SYSTEM_MANIFEST.md`.

### 1) Load / smoke-check the actor (CPU OK)

Confirms the checkpoint loads as M111 (`book` shape `(111, 7)`). Expected ~87927 actor parameters (see final system manifest).

```powershell
python -u -c "from pathlib import Path; from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor; ckpt=Path('results/experiments/p64_ppo_multiround/checkpoints/P64_step4_actor.pt'); actor, book, rho, saved = load_actor(ckpt, 'cpu'); print({'book': tuple(book.shape), 'n_params': sum(p.numel() for p in actor.parameters()), 'keys': sorted(saved)[:12]})"
```

### 2) Development eval (inference rollouts → MSP)

Uses `eval_tag` → `eval_checkpoint_dev` (P61 heldout seeds: 4 ICs × 4 skills × 64 rollouts). Default device in the training scripts is `cuda`; pass `cpu` if needed (slower).

**Prerequisite:** local heldout IC sources under `results/` (same trees the P53/P61 loaders read, e.g. P21 / P35 / P44 artifacts). A fresh clone without those `results/` trees cannot complete this eval.

```powershell
python -u -c "import json; from pathlib import Path; from experiments.precision.p64_ppo_multiround.eval_p64 import eval_tag; from experiments.precision.p64_ppo_multiround.paths import CKPT_DIR; rep = eval_tag('P64_step4', CKPT_DIR / 'P64_step4_actor.pt', device='cuda'); print(json.dumps({k: v for k, v in rep.items() if k != 'per_ic'}, indent=2))"
```

**Expected stdout (summary keys):** `tag`, `ckpt`, `MSP_RELAXED`, `MSP_12S`, `MSP_NEW`, `zero_success_ics`, `by_class` / `process_by_class` (A/B/C/D), plus per-IC progress lines `p_rel=...`. Historical P64 step-4 MSP on this development panel is about **0.27** (see consolidation docs; not a sealed final-test claim).

Optional resume helper (needs existing `BASELINES.json` / `P64_step{1,2,3}.json` under the P64 results dir):

```powershell
python -u -m experiments.precision.p64_ppo_multiround.finish_step4_eval --device cuda
```

Writes/updates `results/experiments/p64_ppo_multiround/P64_step4.json`, `BEST_CHECKPOINT.json`, and reports via `write_reports.py`.

### 3) Full P64 training campaign (optional)

Continues from P63-A checkpoints at:

- `results/experiments/p63_transformer_ppo_c_grasp/checkpoints/P63b_pilot_A_actor.pt`
- `results/experiments/p63_transformer_ppo_c_grasp/checkpoints/P63b_pilot_A_critic.pt`

```powershell
python -u -m experiments.precision.p64_ppo_multiround.run_p64 --device cuda
```

Verified flags only: `--device` (default `cuda`), `--max-steps` (default `4`), `--start-step` (default `1`), `--skip-integrity`, `--skip-train`. Outputs land in `results/experiments/p64_ppo_multiround/` (`checkpoints/P64_step{N}_{actor,critic,optim}.pt`, `P64_step{N}.json`, `BEST_CHECKPOINT.json`, `FINAL_REPORT.md`, …).

Frozen-stack detail: `results/experiments/final_consolidation/FINAL_SYSTEM_MANIFEST.md` and `docs/REPRODUCIBILITY.md`.

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
