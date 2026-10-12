# Final pre-GRPO sanity check

Forty critic-free GRPO updates on the train split. The action interface, K=36 codebook, 39-D observation, geometry entry, and success rule were not changed. No new demonstrations. `FINAL_TEST` was not rolled out.

## Frozen policies

| role | file | sha256 |
|---|---|---|
| A. old-skill reference | `results/diagnostics/raw/pre_grpo_sanity/policy_A_reference.pt` | `f0beb2d857a88c034002f8e71b14458014f50340131ee729b49ae0955a5181c7` |
| B. supervised initialization | `results/diagnostics/raw/pre_grpo_sanity/policy_B_init.pt` | `904d183b4d24527654967b44b6563036901a942c2a3a967c302e57c3f7523bd9` |
| C. trainable copy of B | `results/diagnostics/raw/pre_grpo_sanity/policy_C_weak_u{0,10,20,40}.pt` | updated in place only for C |

A is the K=35 checkpoint `learned_nominal_restoration/ckpt.pt`. B is the saved K=36 checkpoint `future_state_ablation/ckpt_l0.1_s2.pt` (epoch 240, lambda 0.1, seed 2). Among saved checkpoints its closed-loop union of 11 is tied with lambda 0 seed 1, and it is the one that still holds slip, natural-loss, and airborne together. The higher union of 13 in that study was an unsaved epoch-40 snapshot. A and B were copied before training and were not overwritten.

## Objective

Return is 1 after the nominal basin, an explicit handoff, and a nominal-only hold of at least 10 s (`SUCCESS_10S` or `SUCCESS_12S`). Otherwise the return is 0. Shaping is not added. There is no critic.

On a mixed group the update is a clipped GRPO surrogate, clip 0.2, group size 4, residual noise 0.05, Adam learning rate 3e-5. Advantages are `(return - group mean) / group std`. A group whose returns are identical, including every all-failure group and every all-success group, gets advantage 0 and no GRPO step.

Retention is a categorical KL from the trainable policy to reference A, coefficient 0.05, and only on non-airborne groups. Airborne groups are not pulled toward the policy that never solved airborne. There is no demonstration behavior-cloning term, so the 2-step release in the natural-loss demonstrations is not a supervised target. The weak coefficient was enough; a medium coefficient was not needed.

## Closed-loop counts

Deterministic evaluation. Slip is out of 3, natural-loss out of 3, impact out of 2 (`impact_zp6`, `imp_c`), airborne out of 8.

| update | slip | natural-loss | impact | airborne | union |
|---:|---:|---:|---:|---:|---:|
| 0 | 2 | 2 | 0 | 7 | 11 |
| 10 | 3 | 1 | 1 | 7 | 12 |
| 20 | 2 | 2 | 0 | 7 | 11 |
| 40 | 2 | 2 | 1 | 7 | 12 |

Airborne stays at 7. `air_a` and `air_b` remain `SUCCESS_12S` with token 35 selected once and a 2-step chase. The same holds for the nearby train-side pulses except `near_b_short`, which was already not a success at initialization. `air_lat_pos` and `air_fast_pos` stay successful.

Slip does not collapse. `slip_a` stays successful with a 2-step token-34 release. `slip_d` is gained by update 40 and `slip_b` is lost; the slip count stays 2. Natural-loss returns to 2 (`train_dz13.5`, `nl_dvz_p`). `nl_dz14.6` was already a drop at initialization and stays a drop. Impact goes from 0 to 1 because `impact_zp6` becomes `SUCCESS_12S`. `imp_c` is successful at update 10 and not at update 40.

`impact_zm6` is the shared wrist / sliding / open-finger scene. It is `DROP_DURING_RECOVERY` at every checkpoint, which is the same failure it had at initialization.

## Slip release duration

Deterministic `slip_a` and `slip_b` stay at two steps of token 34, matching the demonstrations rather than the reference policy's three steps.

Stochastic groups are allowed to leave that timing. At update 13 a `slip_a` member held token 34 for 3 steps and dropped; the group was all failures, so the advantage was 0 and that timing was not reinforced. At update 33 on `slip_c` the successful member is `[34, 34, 34, 0, ...]` and receives advantage `+1.73`. The three failures use a 2-step release and receive `-0.58`. The reward did support the longer release on that group, and the demonstration timing was not enforced by a cloning term.

## GRPO signal

Forty updates, 13 mixed groups, 26 all-failure groups, 1 all-success group. Mixed groups are slip (6), airborne (5), impact (1), and mis-seat (1).

Examples, return then advantage:

| update | case | returns | advantages |
|---:|---|---|---|
| 2 | air_a | 1, 1, 0, 1 | +0.58, +0.58, -1.73, +0.58 |
| 3 | imp_c | 0, 1, 0, 0 | -0.58, +1.73, -0.58, -0.58 |
| 25 | slip_a | 0, 1, 0, 0 | -0.58, +1.73, -0.58, -0.58 |
| 33 | slip_c | 0, 1, 0, 0 | -0.58, +1.73, -0.58, -0.58 |
| 40 | slip_b | 1, 0, 1, 0 | +1.00, -1.00, +1.00, -1.00 |

Success and failure get opposite signs. All-failure groups contribute no GRPO gradient. The scalar surrogate is about `1e-5` because the samples are on-policy, so the ratio sits near 1.

## Success-label audit

Six successful training rollouts were checked from the hold trace, not from a proxy flag. Each is `SUCCESS_12S`, `held_10` and `held_12` are true, handoff does not edit `qpos` (`qpos_max_abs` 0), `v_cmd` and `w_cmd` are 0, and `recovery_called_during_hold` is false. Handoff times are 6.46–6.68 s on `air_a`, 3.68 s on `imp_c`, and 4.78–5.16 s on `seat_a`. The GRPO return on these rollouts is 1. Failed rollouts in the same groups have return 0. The training loop calls `nominal_validate`, which steps the nominal controller only. `RETAINED_TO_12` is recorded and is not the return.

## Action drift

On the first 8 steps of saved histories, tokens versus reference A:

| history | switches at update 0 | switches at update 40 | token 35 at update 40 | action L2 at update 40 |
|---|---:|---:|---:|---:|
| wrist reseat | 1 | 1 | 0 | 0.23 |
| nl_dz14.6 | 1 | 1 | 0 | 2.09 |
| impact_zp6 | 0 | 0 | 0 | 0.18 |
| air_a demonstration | 7 | 7 | 1 | 2.24 |

The natural-loss difference is the known one: reference `[34, 34, 34, 9, ...]`, trainable `[34, 34, 0, 9, ...]`. It does not grow over 40 updates. Token 35 does not appear on those old-skill prefixes. The airborne history stays far from reference A (KL about 26.5), which is the intended gap. KL on old-skill groups averages about 3.4 and does not run away on slip; the large KL values are on mis-seat states that already select airborne-like tokens.

## Gradient scale

On mixed old-skill updates the GRPO gradient norm is about 470–690 and the KL gradient norm is about 0.4–4. The surrogate sums token and residual log-probabilities over the whole rollout, so a near-one ratio still produces a large parameter gradient. At learning rate 3e-5 that step did not move old-skill tokens or destroy airborne in this budget. Before a long run the sequence term should be averaged over steps, so the GRPO gradient sits on the same scale as the KL term. The KL coefficient does not need to be raised. Two mixed updates recorded a GRPO gradient of 0 because the summed surrogate cancelled exactly; the step still included the KL term.

## Decision

The learning signal matches physical success and failure. Airborne, slip, and natural-loss are still present after 40 updates, and `impact_zp6` was gained. The actor is not frozen and it is not diverging on old-skill tokens. The one change to make before the formal schedule is to length-normalize the GRPO log-probability so its gradient does not dwarf the retention term.

GRPO IS STABLE BUT RETENTION REGULARIZATION NEEDS MINOR ADJUSTMENT
