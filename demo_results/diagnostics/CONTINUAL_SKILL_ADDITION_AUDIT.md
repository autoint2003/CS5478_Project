# Continual skill addition

Formal GRPO was not started. The K=35 policy is the base. Token 35 is the same airborne open-chase mean as the previous study. No new tokens, no new impact demonstrations, and no final-test rollout.

## Base checkpoint

`demo_results/diagnostics/raw/learned_nominal_restoration/ckpt.pt`

K=35, residual cap ±0.25. Deterministic closed loop under the final success rule, remeasured in this study:

| case | result |
|---|---|
| slip_a, slip_b, slip_d | SUCCESS_12S |
| slip_c | drop during recovery |
| train_dz13.5, nl_dz14.6, nl_dvz_p | SUCCESS_12S |
| impact_zp6 | SUCCESS_12S |
| imp_a, imp_c | drop during recovery |
| imp_b, imp_d | basin not reached |
| seat_a, seat_b | drop during recovery |
| air_a, air_b, four nearby pulses, air_lat_pos, air_fast_pos | drop or basin not reached |
| impact_zm6 (wrist / slide / open-finger scene) | basin not reached |

Train deterministic success is 3/12, all slip. The group-of-4 audit already stored for this checkpoint has useful groups on `slip_a` (2/4), `slip_b` (1/4), `slip_d` (1/4), and `imp_c` (1/4). Impact other than that one `imp_c` sample, and both airborne groups, are all failures. `imp_c` does not succeed on the deterministic rollout.

## Extending K=35 to K=36

Token 35 is `[-0.0910, 0.5576, 0.4496, 0, 0, 0, -1]`. Tokens 0–34 are copied bit for bit. The backbone, the old 35 logit rows, and the shared residual head are copied from the base. The new logit is initialized so its score is −100, and its residual head is initialized at zero. It is used only when token 35 is the argmax.

Before any update, deployed actions on the retained demonstrations match the base with max absolute error 0. Selected tokens match. Appending the unused token does not change old behavior.

## Stage A: new-token parameters only

Trainable parameters: `logit35_w`, `logit35_b`, `res35.weight`, `res35.bias`. The Transformer, the old logits, and the shared residual stay frozen. Sixty epochs, airborne demonstrations on half the batches and base-policy targets on the old demonstrations for the other half.

Old-demo drift after Stage A: token match 1, action error 0, residual error 0. A small KL (0.02–0.04 on the natural-loss demos) is probability mass on the unused token. It never wins the argmax.

Closed loop, every base success is still a success: the three natural-loss cases, `slip_a`, `slip_b`, `slip_d`, and `impact_zp6`. Wrist, sliding, and open-finger demo actions stay identical. `impact_zm6` stays outside the basin, as it did for the base.

Airborne does not succeed. Token 35 is selected once on `air_a` and `air_b`, but the hand stays open for 9 steps and the object drops. The same drop occurs on the four nearby pulses and on `air_lat_pos` / `air_fast_pos`. The new head can be chosen. The frozen old logits do not produce the two-step open-and-close timing.

## Stage B: shared output layers

Stage A did not recover airborne, so the old logit rows and the shared residual were unfrozen together with the token-35 head. The Transformer stayed frozen. The same airborne loss and base-policy consistency loss were used for another 60 epochs. This is the first stage with closed-loop airborne success, so Stage C was not run.

| case | base | Stage A | Stage B | from-scratch K=36 |
|---|---|---|---|---|
| air_a, air_b | drop | drop, token 35 once | SUCCESS_12S, chase 2, token 35 once | SUCCESS_12S |
| three of four nearby pulses | fail | drop | SUCCESS_12S | SUCCESS_12S |
| near_b_short | drop | drop | drop | SUCCESS_12S |
| air_lat_pos, air_fast_pos | drop | drop | SUCCESS_12S | SUCCESS_12S |
| nl_dz14.6, nl_dvz_p, train_dz13.5 | SUCCESS_12S | SUCCESS_12S | drop | dz13.5 held; the other two dropped |
| slip_a, slip_b, slip_d | SUCCESS_12S | SUCCESS_12S | drop | only slip_a held |
| impact_zp6 | SUCCESS_12S | SUCCESS_12S | drop | SUCCESS_12S |
| imp_c deterministic | drop | drop | drop | basin not reached |
| impact_zm6 | basin not reached | basin not reached | basin not reached | drop |

Train deterministic success: base 3/12, Stage A 3/12, Stage B 2/12 (`air_a`, `air_b` only).

Stage B old-demo drift is an action L2 of about 2.07 on the natural-loss traces and on the tail of `demo_impact_zp6`. Token match there falls to 0.63–0.87. Mean residual L2 stays about 0.04–0.05, so the action error is the wrong prototype, not a small residual shift. Wrist, sliding, open-finger, and `success_trajectory` keep the same tokens; their residual shift is 0.04–0.06.

## Where the old successes break

On `slip_a` and `slip_b` the base opens with token 34 for three steps: `[34, 34, 34, 0, 34, 33, ...]`. The from-scratch model and Stage B both leave that token at step 2 and select token 0, then token 9. `slip_d` leaves token 34 at step 2 in the from-scratch model and at step 4 in Stage B. The shared Transformer was frozen in Stage B, and the same token is still abandoned. The interference is in the shared logit head.

On `imp_c`, the from-scratch model keeps token 32 at step 0 but the action already differs by L2 0.063, which is residual drift on the same token. Stage B keeps token 32 for two steps, then selects token 34 where the base selects token 0.

This is not a capacity limit. Stage A has the new action available and does not move old successes. Stage B learns the airborne two-step chase only after the shared logits are allowed to move, and those logits are what drop token 34 on the slip release.

## GRPO

No group-of-4 audit was run on Stage B. Retention of `nl_dz14.6`, `nl_dvz_p`, the slip successes, and `impact_zp6` failed, so a learning-signal audit would not be a reason to start GRPO.

Stage A checkpoint: `demo_results/diagnostics/raw/continual_skill_addition/ckpt_stage_A.pt`. It preserves the base and does not recover airborne.

Stage B checkpoint: `demo_results/diagnostics/raw/continual_skill_addition/ckpt_stage_B.pt`. It recovers airborne and removes the retained natural-loss, slip, and `impact_zp6` successes. It is not a GRPO initialization.

UNIFIED POLICY STILL SUFFERS MULTI-SKILL INTERFERENCE
