# Targeted action-representation expansion

The privileged `impact_zp6` chase is the only demonstrated action that the frozen K=32 codebook cannot express under a ±0.25 residual. One appended lateral-chase prototype, with the residual cap left at ±0.25, represents that chase and, after the same 120-epoch supervised training, retains `impact_zp6` through t=12 without dropping the natural-loss recatches or the established skills. Widening the residual to ±1.0 also retains it. Widening it only to ±0.5 does not reconstruct the lateral command. The adopted representation is the single new token.

The 39-D observation, Transformer width, depth, and heads, geometry entry and exit, `recovery7d_airborne_v3` physical bounds, controller gains, natural-loss demonstrations, and t=12 rule were not changed. `ckpt_39d.pt` was not overwritten. Actions are still clipped to `[-1, 1]` before `step_v3`.

## Where the current codebook fails

Every step of the 300-step privileged `impact_zp6` trajectory was scored against the frozen K=32 prototypes. The nearest token, the residual `action − prototype`, the residual clipped to ±0.25, and the remaining error are stored for all 300 steps in `results/diagnostics/raw/action_representation/representation.json`.

Two chase steps sit on token 28, whose prototype is `[0.124, −0.388, 0.252, 0, 0, 0, −1]`. The legal actions saturate the hand-frame x command at +1 while the grip is fully open.

| Step | Phase | Token | Legal action (ax, ay, az, wx, wy, wz, grip) | Residual | Clipped residual | Error | Error L2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | chase | 28 | 1.000, −0.036, −0.336, −0.073, 0.002, 0.004, −1 | 0.876, 0.351, −0.588, −0.073, 0.002, 0.004, 0 | 0.250, 0.250, −0.250, −0.073, 0.002, 0.004, 0 | 0.626, 0.101, −0.338, 0, 0, 0, 0 | 0.719 |
| 1 | chase | 28 | 1.000, 0.041, 0.237, −0.072, 0.003, 0.002, −1 | 0.876, 0.428, −0.015, −0.072, 0.003, 0.002, 0 | 0.250, 0.250, −0.015, −0.072, 0.003, 0.002, 0 | 0.626, 0.178, 0, 0, 0, 0, 0 | 0.651 |

The hand-frame x residual is +0.876 on both steps. The ±0.25 cap leaves 0.626 of that command unrepresented. That channel is ±1 m/s in `recovery7d_airborne_v3`, so the missing piece is about 0.63 m/s of the legal lateral chase.

The other 298 steps are inside or just barely outside the same cap. The close step and the early hold reconstruct with error 0 (198 steps are exact). From about step 200 the hold's hand-frame y command drifts from 0.250 to 0.280 against token 0, which is the zero action with a closed grip. The cap clips that channel at 0.25, and the largest error on any non-chase step is 0.030 at step 299. No established skill has any step outside the cap.

| Family | Steps | Mean error L2 | Max error L2 | Steps with L2 > 0.05 |
| --- | --- | --- | --- | --- |
| wrist | 18 | 0 | 0 | 0 |
| sliding | 52 | 0 | 0 | 0 |
| open-finger recatch | 32 | 0 | 0 | 0 |
| `train_dz13.5` | 8 | 0.042 | 0.181 | 2 |
| `nl_dz14.6` | 8 | 0.044 | 0.219 | 2 |
| `nl_dvz_p` | 8 | 0.042 | 0.182 | 2 |
| `impact_zp6` | 300 | 0.011 | 0.719 | 2 |

The natural-loss residuals that exceed 0.25 are the downward chase channel, by about 0.18–0.22, and those policies already retain in closed loop. `impact_zp6` is the disproportionate gap: its mean error looks small only because 298 of 300 steps are a hold, while the two chase steps are several times farther outside the residual box than any skill or natural-loss step.

## Representation options

A step is treated as accurately represented when its reconstruction L2 is at most 0.05. An option is rejected if that bar fails on `impact_zp6`, or if the mean reconstruction error of wrist, sliding, open-finger recatch, or any natural-loss demonstration rises by more than 0.01. The existing 32 prototypes were not refit. The optional token is the mean of the two chase actions, appended as prototype 32: `[1.000, 0.002, −0.050, −0.072, 0.003, 0.003, −1]`.

| Option | Codebook | Residual cap | `impact_zp6` max L2 | Natural-loss mean L2 | Skill mean L2 | Viable |
| --- | --- | --- | --- | --- | --- | --- |
| Current | K=32 | ±0.25 | 0.719 | 0.042–0.044 | 0 | no |
| Wider residual | K=32 | ±0.5 | 0.386 | 0 | 0 | no |
| Full residual | K=32 | ±1.0 | 0 | 0 | 0 | yes |
| One lateral token | K=33 | ±0.25 | 0.036 | 0.042–0.044 | 0 | yes |

At ±0.5 the chase is still token 28, and the hand-frame x error remains 0.376. That is not an accurate copy of the legal action, so this option was not trained. At ±1.0 the same token plus the unclipped residual matches both chase steps exactly. The appended token matches the saturated lateral command exactly and leaves a 0.036 error on the hand-frame z channel, because the two chase steps command opposite vertical signs around the prototype. Skill reconstruction is unchanged by the extra token. Neither viable option worsens established-skill reconstruction. The ±0.5 and ±1.0 caps make the natural-loss demos exact as well; that is extra range, not a requirement, since those demos already train successfully at ±0.25.

## Supervised retraining

Both viable representations were trained from scratch for 120 epochs, Adam learning rate 1e-3, batch 16, 80 steps per epoch. The pool is the previous balanced set plus `demo_impact_zp6.npz`. Draws are equal across wrist, sliding, open-finger recatch, natural-loss recatch, and impact: 256 suffixes each per epoch. Half of the impact draws start at step 0, so the two chase commands fall inside the 96-step training window; the other half are uniform suffixes of the long hold. Impact does not receive more draws than any other family.

| Run | Final loss | Loss at 40 / 80 | Seconds |
| --- | --- | --- | --- |
| K=32, residual ±1.0 | 0.0130 | 0.0754 / 0.0207 | 224.5 |
| K=33, residual ±0.25 | 0.0093 | 0.0220 / 0.0389 | 218.1 |

Teacher-forced action RMSE, with the same 160-step context the controller uses:

| Demonstration | Prior scratch, K=32, ±0.25, no zp6 | Residual ±1.0 | Lateral token |
| --- | --- | --- | --- |
| wrist | 0.012 | 0.023 | 0.013 |
| sliding | 0.023 | 0.033 | 0.025 |
| open-finger recatch | 0.016 | 0.026 | 0.019 |
| three natural-loss recatches | 0.043–0.046 | 0.021–0.034 | 0.042 |
| `impact_zp6` chase steps | 0.714 | 0.022 | 0.029 |

The prior scratch emits a chase near `[0, 0.04, −0.19, …, −1]` and then closes. The legal chase is `[1, …, −1]`. The lateral-token model emits `[0.992, −0.046, −0.262, …, −1]` and `[1.000, 0.015, 0.164, …, −1]`. The ±1.0 model also matches the saturated lateral command. The token model keeps the skill RMSE next to the prior scratch. The ±1.0 model loosens it.

## Closed loop through t=12

Each row is the autonomous pipeline from that case's geometry entry. Stable exit is the frozen geometry handoff. Retention is the physical outcome at absolute t=12. The prior column is the selected from-scratch model from the previous supervised study: K=32, residual ±0.25, `impact_zp6` not in the training pool.

| Case | Prior scratch | Residual ±1.0 | Lateral token |
| --- | --- | --- | --- |
| `impact_zm6` | retained, stable exit, 20 steps, fingers closed | retained, stable exit, 18 steps, fingers closed | retained, stable exit, 18 steps, fingers closed |
| `impact_z3` | retained, stable exit, 18 steps, fingers closed | retained, stable exit, 18 steps, fingers closed | retained, stable exit, 26 steps, open-chase-reclose |
| `air_lat_pos` | retained, no stable exit, 236 recovery steps | retained, stable exit, 8 steps, open-chase-reclose | retained, stable exit, 5 steps, fingers closed |
| `air_fast_pos` | drop, fall at 8 steps | retained, stable exit, 8 steps, open-chase-reclose | retained, stable exit, 5 steps, fingers closed |
| `train_dz13.5`, `nl_dvz_p`, `dz13.2`, `dz14.2`, `dvz_m` | retained, stable exit, open-chase-reclose | retained, stable exit, open-chase-reclose | retained, stable exit, open-chase-reclose |
| `nl_dz14.6` | retained, stable exit, 17 steps | retained, stable exit, 8 steps | retained, stable exit, 10 steps |
| `h_dz14.00`, `h_dz14.25`, `h_dy`, `h_dx` | retained, stable exit | retained, stable exit | retained, stable exit |
| `impact_zp6` | drop, fall at a few steps, no lateral command | retained, no stable exit, 300-step cap | retained, no stable exit, 300-step cap |

On `impact_zp6` the lateral-token policy opens, commands hand-frame x at +1.0, loses both pads, recontacts, and recloses. The first two actions are `[0.992, −0.045, −0.261, −0.080, 0.011, 0.000, −1]` and `[1.000, 0.013, 0.164, −0.068, −0.002, 0.002, −1]`. The stable-grasp exit does not fire. Recovery runs the 300-step cap and the nominal continuation is `RETAINED_TO_12`, so the capped continuation is benign. That is the same split as the privileged demonstration: physical retention without the frozen exit. The ±1.0 policy does the same, with peak hand-frame x 0.960.

The lateral-token run also puts `air_lat_pos` and `air_fast_pos` back on the short stable exit the pretrained policy used, and it keeps every natural-loss gain, including the four states that were excluded from the earlier training pool. `impact_z3` opens before its stable exit and still retains. Wrist, sliding, and open-finger behavior remain on `impact_zm6` as a closed-finger geometry exit, with teacher-forced errors 0.013, 0.025, and 0.019.

## Decision

±0.5 cannot carry the legal lateral command, so it is not a sufficient expansion. ±1.0 can, and its closed loop retains `impact_zp6` together with the skills, but it widens the residual on every token and loosens the skill fit. One prototype at the original ±0.25 cap is enough to express the chase, and the retrained policy reproduces it in closed loop while keeping the natural-loss retentions and the wrist, sliding, open-finger, impact, and airborne cases. That is the smaller change.

A NEW LATERAL-CHASE TOKEN IS REQUIRED
