# CS5478 unified recovery

A MuJoCo Panda recovers a cylinder after a genuine ball impact. The nominal grasp controller runs until the cylinder leaves the grasp basin. A causal Transformer then commands a legal 7-D recovery action until the cylinder is back inside that basin, at which point control returns to the nominal controller. Success is a real handoff followed by the cylinder still being held through the nominal continuation. An intermediate contact is not success.

The earlier K=32 write-up is in `docs/HISTORICAL_K32_README.md`.

## Architecture

The policy is the causal Transformer in `training/action_representation_expansion.py` (`build_policy`): width 64, 2 layers, 4 heads, feed-forward 128, 160 position embeddings. Each step reads the 39-D `observe_airborne` vector and the previous executed 7-D action. History is not cleared at release, recontact, a repeated recatch, or the handoff.

The action is `recovery7d_airborne_v3`: hand-frame velocity, wrist rate, and gripper command. It is a discrete prototype plus a continuous residual. The selected table has 47 prototypes, shape `(47, 7)`. Tokens 0–35 are the original K=36 table. Tokens 36–46 are eleven appended recovery centers. The residual is `0.25 * tanh(residual_head)`, so it stays inside ±0.25, and the executed command is clipped to ±1. The same decode is used for supervised targets, GRPO rollouts, evaluation, and the one-step physics check.

`training/formal_grpo.py` is the historical K=36 launcher. It still loads the K=36 `FuturePolicy` checkpoint and rejects a 36-wide token head on a 47-token model. K=47 training goes through `training/unified_k47.py` and `step_legal`, which does not park the ball.

## Dataset

| Split | What it contains | Used for gradients |
| --- | --- | --- |
| TRAIN | 312 ball-impact initial conditions: 28 v1 and 284 v2 | yes |
| VALIDATION | sealed v1 validation episodes | no |
| FINAL_TEST | sealed v1 final-test episodes | no |

A TRAIN initial condition is a restored MuJoCo snapshot at recovery entry (hand-frame position error above 6 mm for 40 ms). It is not itself a demonstration. Privileged search produced 359 successful continuous recoveries. The K=47 decoder represents 296 of them, and 288 of those still succeed when the decoded action is replayed. Those 288 trajectories, on 151 initial conditions, are the supervised demonstrations. Five mechanism groups are balanced by `sample_weight`, not by copying rows. `ball_train_v2_0021 / release_long` is one 46-step episode with three complete-loss intervals.

The other 161 TRAIN snapshots, including the 114 v2 cases with no successful search, are GRPO starting states. Eight decoded trajectories that stay inside the residual cap but fail physically are listed in `fragile_trajectories.jsonl` and are not supervised targets.

## Training pipeline

Training has not been run.

1. Supervised initialization on the 288 verified demonstrations.
2. Closed-loop evaluation of that checkpoint on held-out TRAIN snapshots (`ball_train_v2` ids whose number ends in 0 or 5). The policy acts by itself. This is not a teacher replay and it does not touch FINAL_TEST.
3. A 20-update GRPO pilot from the supervised checkpoint, sampling all 312 TRAIN snapshots.
4. A 120-update GRPO run only after that pilot has been inspected.

GRPO advantages are group-relative. A group whose non-truncated returns are identical gets zero advantage and skips the policy surrogate. A rollout stopped by the step cap, without a drop and without a handoff, is omitted from the advantage group; it is not scored as a drop. The KL term compares the two 47-way categorical distributions. The K=36 launcher's one-logit pad is unchanged and is not used here.

## Setup

```powershell
python -m pip install -r requirements.txt
```

```bash
python -m pip install -r requirements.txt
```

The validation environment used MuJoCo 3.12.0. Commands below are run from the repository root. Use `python -u` so the logs flush.

## Commands

Check data, the codebook, one forward pass, one supervised loss, and the GRPO log-probability. This does not update any parameter:

```powershell
python -u training/unified_k47.py --check
```

```bash
python -u training/unified_k47.py --check
```

Supervised initialization (40 epochs, batch 8, seed 1, learning rate 3e-4, checkpoint every epoch):

```powershell
python -u training/unified_k47.py --supervised --config config/unified_k47.yaml
```

```bash
python -u training/unified_k47.py --supervised --config config/unified_k47.yaml
```

Closed-loop evaluation of the last supervised checkpoint:

```powershell
python -u training/unified_k47.py --eval --config config/unified_k47.yaml --checkpoint results/diagnostics/raw/unified_k47/supervised/checkpoint_epoch_0040.pt
```

```bash
python -u training/unified_k47.py --eval --config config/unified_k47.yaml --checkpoint results/diagnostics/raw/unified_k47/supervised/checkpoint_epoch_0040.pt
```

GRPO pilot (20 updates, group 4, learning rate 3e-5, KL coefficient 0.05, clip 0.2, seed 1):

```powershell
python -u training/unified_k47.py --grpo --config config/unified_k47.yaml --checkpoint results/diagnostics/raw/unified_k47/supervised/checkpoint_epoch_0040.pt
```

```bash
python -u training/unified_k47.py --grpo --config config/unified_k47.yaml --checkpoint results/diagnostics/raw/unified_k47/supervised/checkpoint_epoch_0040.pt
```

Longer GRPO run, only after the pilot is inspected. Pass the supervised checkpoint again so the KL reference stays at that policy:

```powershell
python -u training/unified_k47.py --grpo --config config/unified_k47_long.yaml --checkpoint results/diagnostics/raw/unified_k47/supervised/checkpoint_epoch_0040.pt
```

```bash
python -u training/unified_k47.py --grpo --config config/unified_k47_long.yaml --checkpoint results/diagnostics/raw/unified_k47/supervised/checkpoint_epoch_0040.pt
```

## Paths

- Codebook: `results/diagnostics/raw/legal_recovery_v2/codebook_selected_k47.npy`
- Supervised rows: `results/diagnostics/raw/legal_recovery_v2/supervised_targets.jsonl`
- GRPO starts: `results/diagnostics/raw/legal_recovery_v2/grpo_initial_conditions.jsonl`
- Pilot config: `config/unified_k47.yaml`
- Longer-run config: `config/unified_k47_long.yaml`
- Supervised checkpoints: `results/diagnostics/raw/unified_k47/supervised/`
- Pilot checkpoints: `results/diagnostics/raw/unified_k47/grpo_pilot/`
- Longer-run checkpoints: `results/diagnostics/raw/unified_k47/grpo_long/`

## Current status and limitations

The 312 TRAIN snapshots, the 288 executable demonstrations, and the K=47 table are ready. The loader accepts that table, rejects a mismatched token head, and the non-training checks passed. No K=47 policy has been trained.

Sixty-three continuous successes are outside the ±0.25 residual and are not demonstrations. Eight in-range decoded replays fail because the contact solver reacts to a float32 reconstruction of a few billionths of a command. Recovery claims have to come from a nominal handoff and the physical continuation, not from a token being legal or from contact during the recovery phase.
