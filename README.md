# CS5478 grasp recovery (Plans A / B / C)

A MuJoCo Panda recovers a cylinder after disturbance. Legal recovery uses a **7D** action interface at **20 ms**, **39D** runtime observations, and the frozen **K49** vocabulary (IDs 0–48; **HANDOFF = 47**; token **48** = slow reverse wrist). After HANDOFF the unchanged nominal controller continues; task success is **SUCCESS_NEW** (with **SUCCESS_12S** tracked). **GROUND_SCENE** adds irreversible **GROUND_DROP**.

Plans A/B/C are offline expert families for a **single** Transformer policy (not three runtime controllers). Plan C has two verified continuous-action SUCCESS_NEW experts (`gbi_010`, `gbi_022`) on GROUND_SCENE.

**Do not start SFT/GRPO until the K49 Plan C representability gap is addressed** (see `results/consolidation/K49_REPRESENTABILITY.md` / `docs/FINAL_CONSOLIDATION_REPORT.md`).

| Doc | Purpose |
|---|---|
| `docs/ARCHITECTURE.md` | Minimal dependency map and entry points |
| `docs/ARCHIVE_MANIFEST.md` | Historical experiment archive index |
| `docs/REPRODUCIBILITY.md` | Golden replay and checkpoint recovery |
| `docs/DATASET_SCHEMA.md` | Unified expert schema `unified_experts_v1` |
| `docs/FINAL_CONSOLIDATION_REPORT.md` | Consolidation summary |
| `config/training/unified_plan_abc.yaml` | Minimal training config (not launched) |

### Consolidation commands

```powershell
python -u scripts/build_unified_dataset.py
python -u scripts/replay_golden_experts.py
python -u scripts/audit_k49.py
```

Tracked dataset manifests: `datasets/unified_experts/`. Heavy tapes: `data/unified_experts/` (gitignored). Archive checkpoint: tag `archive/pre-consolidation-20261021-src`.

## Directory layout

Run commands from the repository root.

| Path | Role |
|---|---|
| `assets/` | MuJoCo XML, including `scene_ballistic_impact.xml` |
| `config/` | `simulation/`, `controllers/`, `recovery/`, `training/` (incl. `unified_plan_abc.yaml`) |
| `controllers/` | `nominal.py`, `cartesian/`, `gripper/`, `recovery/action_map.py` |
| `envs/` | `step_legal.py`, `ballistic_impact.py`, `simulation/` (incl. `ground_sim.py`), `observations/`, `contacts/`, `outcomes/`, `initialization/` |
| `sensors/` | Finger-pad tactile array and the hand-x offset estimate `lateral_offset.py` |
| `models/` | Policy modules. `airborne_v3.py` is the 39-D policy and `step_v3`. `historical/` is K33/K36 |
| `data_generation/` | Generators and `strategies/` (incl. Plan C `ground_*.py`) |
| `training/` | `train_sft.py`, `train_grpo.py`, `action_vocab_k49.py`, `unified_k48.py`, … |
| `evaluation/` | Closed-loop eval, audits, SUCCESS_NEW helpers |
| `visualization/` | Camera, captions, MP4 encoding, tape replay |
| `experiments/` | Historical / diagnostic experiments (not formal runtime deps) |
| `archive/` | Checkpoint notes, golden copies, experiment index |
| `datasets/` | Tracked unified expert manifests |
| `scripts/` | Entry points (`build_unified_dataset`, `replay_golden_experts`, `audit_k49`, `train`, `evaluate`, Plan C runners) |
| `tests/` | Regression tests (incl. Plan C golden replay) |
| `data/` | Local manifests, demonstrations, snapshots (gitignored) |
| `results/` | Local SFT/GRPO/diagnostics (gitignored) |
| `mujoco_menagerie/` | Vendored robot models |
| `docs/` | Architecture, archive manifest, reproducibility, dataset schema, consolidation report, package map, coursework PDFs |

`data/` and `results/` are gitignored. File moves: `docs/MIGRATION.md`. Package map: `docs/MODULE_ORGANIZATION.md`.

## Dataset

| File | Rows |
|---|---:|
| `data/supervised_targets.jsonl` | 303 training demonstrations (296 recovery, 7 nominal-solvable controls with weight 0) |
| `data/supervised_targets_with_horizontal.jsonl` | 325 rows: the 303 above plus 22 Horizontal Recatch train tapes (current formal SFT entry) |
| `data/supervised_targets_sliding_equal_weight.jsonl` | Same 325 tapes; Sliding `sample_weight` raised to 1.0 (ablation only; does not replace the main manifest) |
| `data/valid_demonstrations.jsonl` | 22 validation demonstrations |
| `data/valid_skill_quotas.jsonl` | 17 unique validation ICs for closed-loop skill eval (11 sliding / 6 horizontal / 5 vertical labels; some ICs carry both sliding and vertical) |
| `data/grpo_initial_conditions.jsonl` | 87 confirmed-recoverable training initial conditions |
| `data/nominal_solvable_control.jsonl` | 7 nominal-solvable controls, excluded from the GRPO recovery pool |
| `data/valid_recovery.jsonl` | 8 validation recovery initial conditions (older held-out list) |

Formal SFT with Horizontal Recatch reads `data/supervised_targets_with_horizontal.jsonl`. Older commands that still point at `data/supervised_targets.jsonl` omit those 22 tapes. Validation demonstrations stay out of the supervised loss. Train and validation snapshots do not overlap. Every recovery row has `verified_success: true` and `nominal_class: RECOVERY_CANDIDATE`. The 7 control rows are `NOMINAL_SOLVABLE` with `sample_weight` 0. A recovery tape that ends before 4 seconds supervises token 47 on the observation after its last action. A tape that lands exactly on the 4 second limit does not add an extra HANDOFF target; the environment forces handoff before the next observation.

Loss uses per-trajectory mean CE (+ residual MSE), then the batch mean of `sample_weight * traj_loss` (`training/train_sft.py`). There is no class sampler: each epoch shuffles every row once. In the horizontal manifest, Sliding tapes have `sample_weight ≈ 0.02` while Horizontal and Vertical Recatch tapes are `1.0`, so Sliding is only ~4% of effective weight despite 77 Sliding ICs.

Supervised strategy counts: `chase_lost` 101, `inward_reseat` 65, `controlled_slide` 47, `release_long` 47, `follow_early` 26, `immediate_handoff` 7, `release_short` 6, `later_close` 4. Validation: `inward_reseat` 10, `chase_lost` 4, `controlled_slide` 4, `follow_early` 4. `controlled_slide` and `inward_reseat` are one Gravity-Assisted In-Hand Sliding family. Task success is `SUCCESS_12S` after a legal recovery and the unchanged nominal handoff. Finger slip and post-impact tilt are diagnostic fields, not extra success gates.

Independent nominal-fail snapshots in the current set, with at least one legal successful tape: sliding 77 train / 11 valid, horizontal recatch 22 / 6, vertical recatch 41 / 5. One snapshot can hold several tapes, so these counts are lower than the jsonl row counts. The quota targets 100/20, 50/10, and 50/10 are not filled. The census is `results/diagnostics/dataset_expansion/CORRECTION_REPORT.md`.

Category manifests: `data/gravity_assisted_in_hand_sliding/`, `data/controlled_sliding/`, `data/inward_sliding/`, `data/recatch/` (horizontal cylinder), `data/vertical_recatch/`, `data/regrasp/`, and `data/immediate_handoff/`. Initial states are in `data/snapshots/`. Two vertical rows, `ball_train_v2_0168` and `ball_train_v2_0261`, remain `SUCCESS_12S` but have `counts_as_recatch: false` because the stored-action replay shows only one both-off contact sample. They are not in the supervised jsonl as vertical recatch.

## Setup

From the repository root, with the audited environment (MuJoCo 3.12, PyTorch, NumPy):

```powershell
python -m pip install -r requirements.txt
```

The training entry points import PyTorch in addition to the packages in `requirements.txt`. Use `python -u` so logs flush.

## Dataset commands

Print the rebuilt category counts. This does not generate data:

```powershell
python -u scripts/generate_data.py --counts
```

Generation entry point, when new ballistic initial conditions are actually required:

```powershell
python -u scripts/generate_data.py --recatch --target-train 200 --target-valid 50 --max-search-trials 400
python -u scripts/generate_data.py --controlled-slide --target-train 200 --target-valid 50 --max-search-trials 400
python -u scripts/generate_data.py --inward-reseat --target-train 200 --target-valid 50 --max-search-trials 400
```

`--max-search-trials` is a cap, not a request to synthesize that many initial conditions. Static dataset checks:

```powershell
python -u evaluation/audit_pretrain_data.py static
```

`python -u evaluation/verify_handoff_targets.py` replays every recovery demonstration in physics and then checks training batches. It is the full physical check, not a quick import test.

## Training and evaluation

Run these from the repository root. Evaluation is deterministic argmax on CPU. Training uses the `device` in the config.

Check the codebook, the HANDOFF head, one supervised loss, and the reward order. This does not update parameters or write a checkpoint:

```powershell
python -u scripts/train.py --check
```

### Supervised training (Horizontal Recatch included)

40 epochs, batch 8, seed 1, Adam learning rate 3e-4, no scheduler. `init: scratch` builds a new K48 network; residual head starts at zero. No prior K=47/K=48 weights are loaded. The 47 motion codes stay fixed.

`config/training/k48/unified_k48_recovery_v3_horizontal.yaml` trains on `data/supervised_targets_with_horizontal.jsonl` and writes `results/sft/checkpoint_epoch_0001.pt` through `checkpoint_epoch_0040.pt`. Do not pass `--checkpoint`. Do not use `config/training/k48/unified_k48_recovery_v3.yaml` for a scratch run; that older config expands the historical K=47 checkpoint.

```powershell
python -u scripts/train.py --supervised --config config/training/k48/unified_k48_recovery_v3_horizontal.yaml
```

Closed-loop skill eval uses the 17 unique ICs in `data/valid_skill_quotas.jsonl` (deterministic argmax). Mechanism labels (Contact-Preserving / Release–Recatch / Hybrid / Other) come from contact timelines, not data tags:

```powershell
python -u evaluation/classify_validation_skills.py --device cuda
```

On that 17-IC set, the horizontal SFT Epoch 40 is **8/17** `SUCCESS_12S` and **0/17** Contact-Preserving. Tree-augmented GRPO Epoch 3 from that checkpoint is **15/17** success and still **0/17** Contact-Preserving. Reports: `results/diagnostics/validation_skill_classification/`, `results/diagnostics/sft_sliding_root_cause/`.

### Sliding equal-weight SFT (controlled ablation)

Hypothesis: Sliding was lost mainly because its `sample_weight` was ~0.02. This run keeps the same 325 tapes, seed, epochs, and hparams, but sets every Sliding tape weight to **1.0** in an independent manifest. Train from **scratch**; do not continue from Epoch 40.

```powershell
python -u scripts/train.py --supervised --config config/training/k48/sft_sliding_equal_weight.yaml
python -u experiments/sliding/eval_sft_sliding_equal_weight.py --device cuda
```

Writes `results/sft/sliding_equal_weight/checkpoint_epoch_*.pt` and does **not** overwrite `results/sft/checkpoint_epoch_0040.pt`. Equal-weight Epoch 40 on the same 17 ICs: **17/17** `SUCCESS_12S`, **9/17** Contact-Preserving; Horizontal Recatch stays 6/6. Full write-up: `results/diagnostics/sft_sliding_equal_weight/REPORT.md`.

Optional older path that writes under `results/sft/current/` from the 303-row manifest without Horizontal tapes:

```powershell
python -u scripts/train.py --supervised --config config/training/k48/unified_k48_recovery_v3_current.yaml
```

### Tree-augmented GRPO

`config/training/k48/unified_k48_recovery_v3_tree80.yaml` writes `results/grpo/tree80/`. One round visits each unique training snapshot once. `--checkpoint` selects the file to load. `--ic-epochs` is the number of new rounds; omit it to use `ic_epochs` in the config. A finished round is saved as `checkpoint_epoch_XXXX.pt` and includes the Adam first and second moments.

The finished tree80 run started from the **downweighted** horizontal SFT (`results/sft/checkpoint_epoch_0040.pt`) and used that file as the KL reference. Epoch 3 is `results/grpo/tree80/checkpoint_epoch_0003.pt` (15/17 success, 0/17 Contact-Preserving). For a new GRPO that should keep Sliding, start from the equal-weight SFT instead and point a dedicated `grpo_output` / KL reference at that tree; do not overwrite `results/grpo/tree80/`.

Start from supervised epoch 40 into an empty GRPO output directory:

```powershell
python -u scripts/train.py --adaptive-grpo --config config/training/k48/unified_k48_recovery_v3_tree80.yaml --checkpoint results/sft/checkpoint_epoch_0040.pt --ic-epochs 1
```

Continue from a saved epoch that contains Adam moments. Replace the checkpoint path and the round count:

```powershell
python -u scripts/train.py --adaptive-grpo --config config/training/k48/unified_k48_recovery_v3_tree80.yaml --checkpoint results/grpo/tree80/checkpoint_epoch_0001.pt --ic-epochs 1
```

`results/grpo/tree80/history/checkpoint_epoch_0001.pt` is the earlier weights-only pass. It has no Adam moments, and training refuses to resume it.

### GRPO reward v1

Three passes over the 87 training recovery initial conditions, 261 updates. Writes `results/grpo/v1/`. The last file is `results/grpo/v1/checkpoint_u0261.pt`. The success terminal is only `SUCCESS_12S`.

```powershell
python -u scripts/train.py --grpo --config config/training/k48/unified_k48_recovery_v3.yaml --checkpoint results/sft/checkpoint_epoch_0040.pt
```

### GRPO reward v2

`config/training/k48/unified_k48_recovery_v3_reward_v2.yaml` runs nine passes, 783 updates, and writes `results/grpo/v2/`. A checkpoint saved after update 261 is the end of the first three passes. This command continues that file from update 262 through 783 and appends `training_log.jsonl` and `rollouts.jsonl`. It does not overwrite `checkpoint_u0261.pt`. The frozen KL reference stays `results/sft/checkpoint_epoch_0040.pt`. The last new file is `results/grpo/v2/checkpoint_u0783.pt`.

```powershell
python -u scripts/train.py --grpo --config config/training/k48/unified_k48_recovery_v3_reward_v2.yaml --checkpoint results/grpo/v2/checkpoint_u0261.pt
```

### Evaluation

The default IC list is `data/valid_recovery.jsonl` (8 validation recovery initial conditions). The summary is `results/evaluations/heldout_train.json`.

SFT epoch 40:

```powershell
python -u scripts/evaluate.py --eval --config config/training/k48/unified_k48_recovery_v3.yaml --checkpoint results/sft/checkpoint_epoch_0040.pt
```

GRPO v1, same 8 validation initial conditions:

```powershell
python -u scripts/evaluate.py --eval --config config/training/k48/unified_k48_recovery_v3.yaml --checkpoint results/grpo/v1/checkpoint_u0261.pt
```

GRPO v2, same 8 validation initial conditions:

```powershell
python -u scripts/evaluate.py --eval --config config/training/k48/unified_k48_recovery_v3_reward_v2.yaml --checkpoint results/grpo/v2/checkpoint_u0261.pt
```

`--eval-data` selects another recovery list. The 87 training recovery initial conditions are `data/grpo_initial_conditions.jsonl`. Pass `--eval-output` so this does not overwrite `results/evaluations/heldout_train.json`. The rows are written to `closed_loop.json` in that directory.

```powershell
python -u scripts/evaluate.py --eval --config config/training/k48/unified_k48_recovery_v3.yaml --checkpoint results/grpo/v1/checkpoint_u0261.pt --eval-data data/grpo_initial_conditions.jsonl --eval-output results/evaluations/raw/grpo_v1_train_eval
```

Replace the checkpoint path to evaluate another run on the same 87 initial conditions. Reward v1 and reward v2 use the same rollout. The config selects the codebook and the 4 s / 12 s limits.

## Video

Replay a saved tape from its snapshot. The default is `ball_train_v2_0070` / `follow_early`, with recovery frames repeated 6 times and the official 12 s nominal hold at 25 fps. `--extra-hold 5` continues the nominal controller for another 5 seconds. This resimulates the initial condition; it does not interpolate poses.

```powershell
python -u scripts/render_video.py
python -u scripts/render_video.py --episode ball_train_v2_0070 --strategy follow_early --extra-hold 5
```

Side-by-side **policy vs immediate nominal handoff** on the 17 validation ICs (`data/valid_skill_quotas.jsonl`). Left panel is deterministic K48; right panel hands off to nominal immediately. Labels match `horizon_outcome`.

```powershell
python -u visualization/compare_policy_nominal.py --checkpoint results/sft/sliding_equal_weight/checkpoint_epoch_0040.pt --manifest data/valid_skill_quotas.jsonl --out results/videos/policy_vs_nominal/sliding_equal_weight_epoch40 --device cuda
python -u visualization/compare_policy_nominal.py --checkpoint results/grpo/tree80/checkpoint_epoch_0003.pt --manifest data/valid_skill_quotas.jsonl --out results/videos/policy_vs_nominal/checkpoint_epoch_0003 --device cuda
```

## Model

The policy is `models/k48_policy.py`: width 64, 2 layers, 4 heads, feed-forward 128, 160 position embeddings. Each step reads the 39-D observation and the previous 7-D action. The motion table is `(47, 7)`. The residual is `0.25 * tanh(residual_head)`. Token 47 is outside that table. Checkpoint loading is `models/k48_checkpoint.py`. `training/unified_k48.py` owns the supervised loss and GRPO loop. `training/unified_k47.py` remains the shared codebook and array loader.

## Historical experiments

One-off searches live under `experiments/`. The old `demo/` scripts are in `experiments/legacy/demo/`. Their previous outputs, reports, and raw traces are in `results/archive/demo_results/`. A stored path that still begins with `results/` is resolved to `results/archive/demo_results/` when that historical file exists, so old manifest provenance is left unchanged.

Older K=47 commands in this file's previous revision used `config/training/k47/unified_k47_horizon.yaml` and wrote under what is now `results/archive/demo_results/diagnostics/raw/unified_k47_horizon/`. Those configs still point at that historical tree. Package roles are in `docs/MODULE_ORGANIZATION.md`.
