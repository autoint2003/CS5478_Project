# New recatch demonstration supervised retraining

Supervised retraining on the new privileged recatch demonstrations expands closed-loop recovery. The model selected before any held-out roll is the from-scratch run. From the live geometry entries it reproduces open, loss of both pads, recontact, reclose, stable-grasp exit, and `RETAINED_TO_12` on all three training recatches, on the two clean validation offsets the pretrained policy drops, and on all four excluded natural-loss states. Wrist reseating, inward sliding, open-finger recatch, `impact_zm6`, and `impact_z3` still retain. `air_fast_pos` drops. `impact_zp6` still drops, by a fall during the attempted open, so the frozen exit rule is not what stops the learned policy there.

The 39-D observation, previous 7-D command, Transformer, K=32 codebook, `recovery7d_airborne_v3`, geometry entry, stable-grasp exit, controller gains, and absolute t=12 rule were left unchanged. `ckpt_39d.pt` was not overwritten. No SAC, DPO, or GRPO update was run.

## Updated training pool

Four families, drawn equally. Each epoch draws 320 suffixes from each family (120 epochs, 80 Adam steps, batch 16, learning rate 1e-3). The three new recatches are one family, so they receive the same draw count as wrist, sliding, or the existing open-finger recatch, and they do not outnumber those skills inside the batch. There is no training-side impact or lateral-chase demonstration, so that family was not added.

| Family | Files | Role |
| --- | --- | --- |
| wrist | `demo_wrist_reseat.npz` (18 steps) | existing skill |
| sliding | `demo_gravity_inward.npz` (52 steps) | existing skill |
| recatch | `demo_recatch_open20.npz` (32 steps) | existing open-finger recatch |
| natural_recatch | `demo_train_dz13.5.npz`, `demo_nl_dz14.6.npz`, `demo_nl_dvz_p.npz` (8 steps each) | new privileged legal recatches |

Excluded from training and from selection: `demo_h_dz14.00.npz`, `demo_h_dz14.25.npz`, `demo_h_dy.npz`, `demo_h_dx.npz`, `demo_impact_zp6.npz`. Failed rollouts were not used as positive demonstrations.

Clean validation, unused as demonstration sources and unused as the held-out diagnostic entries: `dz13.2` (dz −13.2 mm), `dz14.2` (dz −14.2 mm), `dvz_m` (dz −13.5 mm, object vz −0.01 m/s), plus the established successes `impact_z3`, `air_lat_pos` (0.30 m/s, +y), and `air_fast_pos` (0.45 m/s, +y).

## Tokenizer check

K stayed 32. Mean action error after the ±0.25 residual cap is 0.042–0.044 on the three new training trajectories. That is inside the representation the policy can emit. The check that would have stopped training (mean capped residual above 0.15 on a training demonstration) did not fire.

Nearest-token sequences:

| Trajectory | Tokens | Prototype L2 | Capped residual L2 | Max \|residual\| |
| --- | --- | --- | --- | --- |
| `train_dz13.5` | 15, 28, 0, 9, 0, 0, 0, 0 | 0.268 | 0.042 | 0.431 |
| `nl_dz14.6` | 15, 15, 0, 9, 0, 0, 0, 0 | 0.258 | 0.044 | 0.469 |
| `nl_dvz_p` | 15, 28, 0, 9, 0, 0, 0, 0 | 0.268 | 0.042 | 0.432 |
| `impact_zp6` (analysis only) | 28, 28, 0, 29, then 0 through step 300 | 0.263 | 0.011 | 0.876 |

On the training recatches the open/chase phase and the reclose phase use disjoint tokens. Chase is grip −1 on tokens 15 and 28. Reclose and hold are tokens 0 and 9, with grip positive. Token 0 is shared by the first close step and the start of the hold, so the settled hold is not a wholly separate code from the reclose. The existing open-finger demonstration already uses this split: token 30 for the open step, tokens 6 and 0 for the close. Wrist (tokens 4 then 0) and sliding (tokens 10, 12, then 0) contain no open step, and their reconstruction error is numerically zero.

The residual that remains after the nearest codeword is mostly the downward channel during the two chase steps (about 0.38–0.47, against a 0.25 cap). The open, the reclose, and the hold are still distinct in the token sequence, and the mean action left after the cap is about 0.04.

`impact_zp6` was scored with the same frozen codebook and was not trained. Its two chase steps sit on token 28 with the hand-frame x command saturated at +1. The residual on that channel is +0.876, which the ±0.25 residual head cannot cover. Open token 28 is disjoint from close tokens 0 and 29. Across the 300 steps, 102 steps have some channel outside the residual cap, nearly all of them the small hand-frame y command during the long hold. The saturated lateral chase is an action region the current codebook does not represent. K was not refit, because that file is not a training demonstration and the training recatches are representable.

## Fine-tune and from-scratch

Both runs use the same architecture, the same family-balanced sampler, and 120 epochs.

| Run | Init | Loss at 40 / 80 / 120 | Seconds | Final loss |
| --- | --- | --- | --- | --- |
| Fine-tune | `ckpt_39d.pt` | 0.0155 / 0.0237 / 0.0073 | 221.6 | 0.0073 |
| From scratch | fresh weights, seed 1 | 0.0580 / 0.0254 / 0.0138 | 221.1 | 0.0138 |

Teacher-forced action RMSE on the three skill demonstrations, after training:

| Skill | Pretrained | Fine-tune | From scratch |
| --- | --- | --- | --- |
| wrist | 0.0017 | 0.0059 | 0.0121 |
| sliding | 0.0218 | 0.0225 | 0.0234 |
| open-finger recatch | 0.0138 | 0.0144 | 0.0164 |

The skill sequences are still fit. Wrist error rises from 0.0017 to 0.012 on the from-scratch model and stays far below the level that previously destroyed the wrist action.

## Training closed loop

Each row is the autonomous pipeline from that case's own geometry entry: policy steps only, stable exit if the frozen rule fires, then nominal continuation to t=12. No teacher forcing on the physical roll. The three scripted skills share the `impact_zm6` entry, so that one roll cannot replay three different action sequences; their action match is the RMSE table above, and the physical skill check is `impact_zm6` itself.

| Case | Pretrained | Fine-tune | From scratch |
| --- | --- | --- | --- |
| `impact_zm6` | retained, geometry, 18 steps, no open | retained, geometry, 18 steps, no open | retained, geometry, 18 steps, no open |
| `train_dz13.5` | drop, geometry, 5 steps, no open | retained, geometry, 10 steps, open-chase-reclose | retained, geometry, 8 steps, open-chase-reclose |
| `nl_dz14.6` | drop, geometry, 5 steps, no open | retained, geometry, 10 steps, open-chase-reclose | retained, geometry, 17 steps, open-chase-reclose |
| `nl_dvz_p` | drop, geometry, 5 steps, no open | retained, geometry, 10 steps, open-chase-reclose | retained, geometry, 8 steps, open-chase-reclose |

Training recatch success is 0/3, 3/3, and 3/3. Both retrained policies execute the new mechanism on the states the demonstrations came from: the grip opens, both pads lose contact, contact returns, the grip recloses, the stable-grasp exit fires, and the object is retained at t=12.

## Validation

Selection used only these rows, before the held-out roll.

| Case | Split | Pretrained | Fine-tune | From scratch |
| --- | --- | --- | --- | --- |
| `dz13.2` | new | retained, geometry, 5 steps, no open | retained, geometry, 10 steps, open-chase-reclose | retained, geometry, 8 steps, open-chase-reclose |
| `dz14.2` | new | drop, geometry, 5 steps, no open | retained, geometry, 10 steps, open-chase-reclose | retained, geometry, 8 steps, open-chase-reclose |
| `dvz_m` | new | drop, geometry, 5 steps, no open | retained, geometry, 10 steps, open-chase-reclose | retained, geometry, 8 steps, open-chase-reclose |
| `impact_z3` | old | retained, geometry, 18 steps, no open | retained, geometry, 18 steps, no open | retained, geometry, 18 steps, no open |
| `air_lat_pos` | old | retained, geometry, 5 steps, no open | drop, fall at 18 steps, open-chase-reclose | retained at t=12, no stable exit, 236 recovery steps, open-chase-reclose |
| `air_fast_pos` | old | retained, geometry, 5 steps, no open | drop, fall at 20 steps, open-chase-reclose | drop, fall at 8 steps, opened, no recontact |

New-offset validation is 1/3, 3/3, and 3/3. The pretrained retention on `dz13.2` is the short geometry handoff with the fingers staying closed. Both retrained policies replace that with the open-chase-reclose exit and still retain. `dz14.2` and `dvz_m` were drops for the pretrained policy and are retentions for both retrained policies.

Old validation is 3/3, 1/3, and 2/3. Fine-tune loses both airborne carries by opening and falling. From-scratch keeps `air_lat_pos` physically through t=12, but the stable-grasp exit never fires: recovery continues until the task clock (236 steps, short of the 300-step cap) and the object is still held. `air_fast_pos` falls after the from-scratch policy opens.

Ranking before held-out, in order: validation successes, `impact_zm6` retained, training-recatch successes, lower skill RMSE. From-scratch has 5 validation successes against 4 for fine-tune. Both keep `impact_zm6` and all three training recatches. From-scratch is the selected model. Its higher skill RMSE is the last key and does not change the choice. The logged selection scalar −0.05185 is that skill-error term for the winner, not a second training run.

## Old-skill retention

Through t=12 on the selected from-scratch model: `impact_zm6` retained, `impact_z3` retained, teacher-forced wrist / sliding / open-finger errors 0.012 / 0.023 / 0.016. The closed-loop impact exits stay the 18-step geometry handoff with the fingers closed, matching the pretrained impact behavior.

The airborne regression is real and is why fine-tune lost the selection. Fine-tune drops `air_lat_pos` and `air_fast_pos`. From-scratch drops `air_fast_pos` and keeps `air_lat_pos` only by staying in recovery until the task clock. The three scripted skills and the successful impact cases are still retained.

## Excluded privileged-success states

Rolled once each, after selection. These states were not put back into training.

| State | Pretrained | Fine-tune | From scratch (selected) |
| --- | --- | --- | --- |
| `h_dz14.00` | drop, geometry, 5 steps, no open | retained, geometry, 10 steps, open-chase-reclose | retained, geometry, 8 steps, open-chase-reclose |
| `h_dz14.25` | drop, geometry, 5 steps, no open | retained, geometry, 10 steps, open-chase-reclose | retained, geometry, 8 steps, open-chase-reclose |
| `h_dy` | drop, geometry, 5 steps, no open | retained, geometry, 10 steps, open-chase-reclose | retained, geometry, 18 steps, open-chase-reclose |
| `h_dx` | drop, fall at 129 steps, opened, no recontact | retained, geometry, 10 steps, open-chase-reclose | retained, geometry, 13 steps, open-chase-reclose |
| `impact_zp6` | drop, geometry, 5 steps, no open | drop, fall at 3 steps, opened, no reclose | drop, fall at 6 steps, grip reclosed, no pad recontact |

The four held-out natural-loss states now follow the same mechanism as the training recatches: open, both pads lost, recontact, reclose, stable exit, `RETAINED_TO_12`. The pretrained policy drops all four. That transfer was not used to pick the checkpoint.

## impact_zp6 exit behavior

The privileged `impact_zp6` demonstration still retains through t=12 without a stable-grasp exit before the 300-step cap. This run did not change that exit rule, and it did not train on that demonstration.

| | Physical retention | Stable exit | Recovery until the cap |
| --- | --- | --- | --- |
| Pretrained | drop | fires at 5 steps, then nominal drops | no |
| Fine-tune | drop | does not fire | no; fall abort at 3 steps |
| From scratch | drop | does not fire | no; fall abort at 6 steps |

None of the three policies produce a long recovery that is physically retained and merely blocked by the exit rule. Both retrained policies open immediately and the object falls before recontact. Continued recovery until the cap is not reached, so it is not a benign cap-hold. On this state the gap is the learned recovery behavior. Separately, the privileged lateral command itself is outside the current residual cap, as reported in the tokenizer section.

## Comparison at t=12

Counts are physical `RETAINED_TO_12` outcomes, not token accuracy.

| | Pretrained | Fine-tune | From scratch |
| --- | --- | --- | --- |
| Training recatches | 0/3 | 3/3 | 3/3 |
| `impact_zm6` | retained | retained | retained |
| New validation offsets | 1/3 | 3/3 | 3/3 |
| Old validation | 3/3 | 1/3 | 2/3 |
| Skill RMSE sum | 0.037 | 0.043 | 0.052 |
| Excluded natural-loss states | 0/4 | 4/4 | 4/4 |
| `impact_zp6` | drop | drop | drop |

The from-scratch policy solves the training states that now have legal recatch demonstrations, and it solves nearby and held-out natural-loss entries that were not in the training pool. The supervised fit is consistent with that closed loop: final loss 0.014, and the physical rolls show the open-chase-reclose contact sequence rather than a short closed-finger exit. The remaining misses are `air_fast_pos`, where the new open is harmful, and `impact_zp6`, which was withheld from training and whose lateral command the codebook does not cover. Those misses do not put the new natural-loss recoveries back into a representation or compounding failure. The coverage that was missing from the pretrained policy is what this training added.

NEW SUCCESSFUL DEMONSTRATIONS EXPAND AUTONOMOUS RECOVERY
