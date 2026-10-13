# Final RL training readiness

Formal GRPO is not ready to start. The observation, the K=35 codebook, and the nominal-basin handoff rule are frozen, and a new episode split exists. The train set is still mostly states for which a bounded legal search found no recovery. Impact and airborne have no verified recoverable training episode.

The historical eight-case diagnostic set is recorded as `LEGACY_DIAGNOSTIC_SET`. It is not this split, and the final-test episodes were not rolled out.

## Frozen observation

The policy input stays the 39-D airborne observation plus the previous 7-D command. History is the causal Transformer context, truncated to the latest 160 steps. No skill id and no privileged state enter the network.

## Frozen action representation

Retained demonstrations were reconstructed with K=35 and residual cap ±0.25.

| family | file | steps | max L2 | median L2 | clipped components |
|---|---|---:|---:|---:|---:|
| wrist reseating | `demo_wrist_reseat.npz` | 18 | 0 | 0 | 0 |
| inward sliding | `demo_gravity_inward.npz` | 52 | 0 | 0 | 0 |
| open-finger recatch | `demo_recatch_open20.npz` | 32 | 0 | 0 | 0 |
| natural-loss recatch | three demonstrations | 8 each | 0 | 0 | 0 |
| lateral chase | `demo_impact_zp6.npz` | 300 | 0.0365 | 0 | 102 |
| nominal-grasp restoration | `success_trajectory.npz` | 29 | 0 | 0 | 0 |

The lateral-chase maximum is 0.036 on a 300-step trace whose median error is zero. That is inside the frozen cap. No further token was added.

New-mode tokens, copied from the learned checkpoint and not refit:

- token 32, lateral chase: `[1, 0.002, -0.050, -0.072, 0.003, 0.003, -1]`
- token 33, closed lift: `[0, 0, -0.4, 0, 0, 0, +1]`
- token 34, second release: `[-0.065, 0.021, 0.526, 0, 0, 0, -1]`

Wrist uses tokens 0 and 4. Sliding uses 0, 10, and 12. Open-finger recatch uses 0, 6, and 30. Natural-loss and restoration use 33 and 34. K=35 is frozen.

## Success and handoff

Canonical config: `config/recovery_success.yaml`. Evaluator: `training/recovery_success.py`.

A rollout is `RECOVERY_SUCCESS` only when recovery entry has occurred, the basin below has persisted for 0.100 s, handoff has run, recovery actions have stopped, and nominal control then holds for 10 s. The 12 s window is the extended diagnostic. Both timers start at handoff. Absolute t = 12 does not stop the test. `RETAINED_TO_12` is stored only when the clock actually passes t = 12.

Frozen basin, from the undisturbed-hold measurement:

- orientation error ≤ 0.117 rad
- object-in-hand pinch error ≤ 3.03 mm
- relative speed ≤ 0.020 m/s
- persistence 0.100 s (5 control steps)

Geometry entry is still `config/geometry_switch.yaml`: position error above 6 mm for 40 ms. The old exit in that file (14.7 mm, 0.35 rad, 0.08 m/s) is legacy and is not the success label.

Handoff writes the saved nominal `r_des`, not the recovered hand orientation. On the first verified handoff in this run (t = 4.876 s, phase `lift`):

- `qpos` and `qvel` changed by 0
- simulation time did not jump
- `r_des` matched the saved nominal orientation (0 rad)
- `r_des` differed from the current hand by 0.021 rad
- `v_cmd` and `w_cmd` were 0
- grip command was −18 N·m
- `p_des` was set to the current hand position so the arm does not yank
- the following 12 s called only `nominal_step`

No snapshot restore and no object teleport occur at handoff.

## Dataset

Twenty-four new full episodes were generated from a seated nominal start through live geometry entry. Each id is one physical episode. Suffixes are not separate cases, and the same impact is not stored under several skill names. Parameters were assigned to train, validation, and final test before any recovery rollout.

| split | impact | slip | mis-seat | airborne | invalid (no entry) | trivial |
|---|---:|---:|---:|---:|---:|---:|
| train | 4 | 4 | 2 | 2 | 0 | 0 |
| validation | 2 | 2 | 1 | 1 | 1 (`imp_f`) | 0 |
| final test | 2 | 2 | 1 | 1 | 1 (`imp_h`) | 0 |

Final-test candidates, not rolled out: `imp_g`, `slip_g`, `slip_h`, `seat_d`, `air_d`.

No entry state was already inside the basin. The nearest cases are the airborne entries, at 6.01 mm pinch, 0.001 rad, and relative speed under 0.001 m/s, with bilateral contact. They sit just outside the 3.03 mm basin, which is why geometry entry fired and why they were kept.

Entry ranges on the non-trivial candidates:

- impact: pinch 7.9–37.6 mm, orientation about 0.01 rad
- slip: pinch 12.3–15.6 mm, orientation 0.0036 rad, bilateral
- mis-seat: pinch 9.4–11.0 mm, orientation about 0.004 rad, bilateral
- airborne: pinch 6.01 mm, orientation under 0.001 rad, bilateral

## Privileged recoverability

From each train and validation capture, the search tried the legal restoration feedback law, then the open-finger, wrist, and inward-slide schedules. Every command was a v3 action. The capture was not edited after the attempt started. Search failure is not a proof of impossibility.

| split | verified recoverable | search failed | invalid |
|---|---:|---:|---:|
| train | 5 / 12 | 7 / 12 | 0 |
| validation | 2 / 6 entered | 3 / 6 entered | 1 |

Train coverage by family: slip 4/4 verified, mis-seat 1/2, impact 0/4, airborne 0/2. Validation slip was 2/2 verified. Validation impact, mis-seat, and airborne were not verified by this search. The seven unverified train episodes are a majority of the train set, and they include the whole impact family.

## Pretrained baseline

One deterministic rollout of `demo_results/diagnostics/raw/learned_nominal_restoration/ckpt.pt` on each train and validation candidate. Final test was not run.

| case | split | family | result | 10 s | 12 s |
|---|---|---|---|---|---|
| imp_a | train | impact | drop during recovery | no | no |
| imp_b | train | impact | basin not reached | no | no |
| imp_c | train | impact | basin not reached | no | no |
| imp_d | train | impact | basin not reached | no | no |
| imp_e | validation | impact | basin not reached | no | no |
| slip_a | train | slip | success | yes | yes |
| slip_b | train | slip | success | yes | yes |
| slip_c | train | slip | drop during recovery | no | no |
| slip_d | train | slip | success | yes | yes |
| slip_e | validation | slip | success | yes | yes |
| slip_f | validation | slip | drop during recovery | no | no |
| seat_a | train | mis-seat | drop during recovery | no | no |
| seat_b | train | mis-seat | drop during recovery | no | no |
| seat_c | validation | mis-seat | drop during recovery | no | no |
| air_a | train | airborne | drop during recovery | no | no |
| air_b | train | airborne | drop during recovery | no | no |
| air_c | validation | airborne | success | yes | yes |

Train success is 3/12, all slip. Validation success is 2/5 entered episodes (`slip_e`, `air_c`). `air_c` succeeded under the learned policy even though the privileged menu had failed, which is why that search failure is not treated as impossibility. End-of-window pinch on the slip successes was 3.3–5.5 mm: the handoff itself was inside the basin, and the nominal hold did not drop, while the pinch drifted outward. `air_c` ended at 0.84 mm.

## GRPO signal

Groups of 4 stochastic rollouts were drawn on the 12 train candidates from the frozen policy. The return is the existing per-step shaping plus 1.0 when the 10 s success label is met. A group has a useful advantage when outcomes are mixed or the return span is at least 0.5. No GRPO update was run.

Useful groups: 4/12 (`slip_a`, `slip_b`, `slip_d`, `seat_a`), each with 1 success and 3 failures and a return span near 1.0. The other 8 groups have zero successes. Their return spans are 0.03–0.16, below the 0.5 rule, so shaping alone does not create an advantage. By family, slip has signal on 3/4 groups, mis-seat on 1/2, and impact and airborne on 0/6.

Mean action distance inside every group is between 2.9 and 5.4, so the members are not the same action sequence. A separate seeded check of the numpy sampler now used by the script repeats with action L2 0, and a different seed differs by L2 1.13 over 15 steps. The simulator is not the source of the mixed outcomes. The groups stored in `readiness.json` were sampled before that sampler was pinned to NumPy; their token draws used `torch.multinomial`. The action distances above are from those stored groups.

## Why SAC is not the default

The critic study on `ckpt_39d.pt` fitted trajectory-level returns: retained and dropped Q values sat near 0 (about 0.07 versus 0.03, later 0.11 versus 0.05), while the return gap was about 0.5 or larger. Same-entry action ranking was 3/26 on the small pool and 72/110 on the medium pool. A history-aware critic did not fix that ranking. Online SAC then drove wrist-demonstration RMSE to 0.201 when the critic was allowed to keep stepping, and the selected SAC checkpoints did not add a validation success. SAC stays a later comparison, not the optimizer that seeds the final run.

## Why DPO is not the default

DPO needs supplied preference pairs and does not generate the recovery attempts itself. On the shared pool the selected update had only a handful of retained-versus-drop pairs and did not change closed-loop validation. The larger pair set dropped a previously retained validation impact. It remains an offline ablation for a later time when successful-versus-failed pairs are already in hand.

## Cleanup

Three superseded weight files were moved to `demo_results/archive/pre_final_training/checkpoints/`: the rejected ±1 residual-cap checkpoint, and the K=32 fine-tune and from-scratch checkpoints from the earlier recatch retrain. Cited weights stay where the reports point. The manifest is `demo_results/diagnostics/CLEANUP_MANIFEST.md`.

## Seed for a later formal run

When the train set has verified impact and airborne coverage, the run should start from:

- checkpoint `demo_results/diagnostics/raw/learned_nominal_restoration/ckpt.pt`
- K=35, residual cap 0.25, 39-D causal Transformer
- entry `config/geometry_switch.yaml`
- success `config/recovery_success.yaml` through `training/recovery_success.py`
- episodes `demo_results/diagnostics/raw/final_rl_readiness/ics/` with the split in `readiness.json`
- final-test ids held out of every update and every model-selection rollout

TRAINING DATASET STILL LACKS VERIFIED RECOVERABLE COVERAGE
