# Recovery-policy pretraining and handoff staging

This is the preparation for a recovery actor, plus a Stage A behavior-cloning pilot. No viability classifier is trained or used. Physical labels are unchanged nominal continuation to t = 12. Learned handoff is not trained. Joint SAC is not started.

Raw files: `results/diagnostics/raw/recovery_policy_pretraining_preparation/`. The runner is `training/recovery_policy_pretraining_preparation.py`.

## 1. Action space

Stage A actions are 7D, called `recovery7`. The map is the same one already checked in the Cartesian feasibility audit:

| index | command | scale |
| ---: | --- | --- |
| 0–2 | hand-frame linear velocity | 0.08 m/s |
| 3–5 | angular velocity in the `r_des` frame | 4 rad/s |
| 6 | grip | −1 maps to +2 N·m, +1 maps to −18 N·m |

`v = R_hand @ a[0:3] * 0.08` and `w = R_des @ a[3:6] * 4`.

`recovery4d` is `[wy, v_hx, v_z_world, grip]`. Its angular command is only about `r_des` y, so `wx` is identically 0. On an identity frame, the wrist reseat `[0.8, 0.6, 0, 1]` and its 7D embedding `[0.6, 0, 0, 0, 0.8, 0, 1]` produce the same world velocity, world rate, and tendon torque. A nonzero 4D world-z channel is not embedded, because hand-frame `vz` is a different direction. None of the three established skills use that channel.

The gravity-assisted slide commands `wx = 0.28` and `wy = 0.50` together (1.12 rad/s and 2.0 rad/s) for 0.40 s, then a grip of −0.48 with zero rate. That `wx` term has no 4D counterpart. Stage A rollouts use the 7D map. `RecoveryEnv.step` still exposes 4D; switching that env is a prerequisite for later online fine-tuning, and this pilot does not route actions through it.

## 2. Observation and history

The policy input is the legal 27-D `observe_observable` vector, then 4 past observations, 4 past actions, and 4 history-valid bits. Feature size 167.

The 27 names are `e_hat_x`, `estimate_valid`, the two contact flags, the two validity flags, `u_L`, `v_L`, `u_R`, `v_R`, `fn_L`, `fn_R`, aperture, aperture rate, tendon torque, hand-frame gravity, hand linear and angular velocity, hand height above the table, z-track error, and the tactile-estimate rate.

Excluded from the input: mass, friction, object pose, object velocity, oracle outputs, outcome labels, and mechanism names. Mechanism names are stored beside the rows for balancing and for the tables below. The imitation arrays `X` and `Y` do not contain them. The handoff head, when it is trained later, reads this same 167-D history.

## 3. Dataset

One success recipe per family, plus a nearby failure. Initial conditions and the held-out cells were named before the outcomes were read.

| family | success recipe | nearby failure | where the success recipe retains and is imitated |
| --- | --- | --- | --- |
| wrist micro-reseating | `[0.8, 0.6, 0, 1]` for 12 steps, then a secure hold for 6 | opposite sign, which drops on `impact_zm6` | `impact_zm6` at nominal mass and at 0.16 kg |
| wrist airborne recatch | 19 secure wrist steps, one 20 ms open, 12-step reclose | three open steps (60 ms), which escape | `impact_zm6`, `impact_zp6`, `drift_0`, and 0.16 kg on `zm6` and `drift_0` |
| gravity-assisted slide | `wx=0.28`, `wy=+0.50` for 20 steps, grip −0.48 for 24, retighten 8 | outward `wy=−0.50`, which escapes; secure grip retains without the slide | `impact_zm6`, `impact_zm6` at 0.16 kg, `impact_zp6` |

The slide counts as executed only when the relax window moves hand x by at least 2 mm inward and both pads stay in contact. That is −3.3 mm on `zm6`, −2.4 mm at 0.16 kg, and −3.7 mm on `zp6`. On `drift_0` the same command retains with only −1.2 mm, so it is kept as a row and is not an imitation target. The secure-grip control on `zm6` retains to t = 12 and is not imitated.

Imitation steps, after that filter: wrist 36, recatch 160, gravity 156. The recatch count is one recipe on three initial conditions and two masses, not a grid of 17/19/21-step copies. The loss divides each family by its own step count, so the three families have equal total weight. Raw step share before that weight is about 10% wrist, 45% recatch, 44% gravity.

Wrist reseating drops on `impact_zp6` and on `drift_0`. Those rows stay in the set as failures. The slide at 0.18 kg / 0.55 on `zm6` runs away (hand-x change −86 mm, escape). Wrist reseating and the 20 ms recatch still retain on that cell.

## 4. Held-out split

Splits are by initial condition, dynamics cell, and family. They are not a random cut of frames.

| split | contents | used in the loss |
| --- | --- | --- |
| train | `impact_zm6`, `impact_zp6`, `drift_0`, nominal dynamics and mass 0.16 kg | success rows that also executed the recipe |
| held-out IC | all of `impact_zp14` | no |
| held-out dynamics | 0.18 kg / friction 0.55 on `impact_zm6` and on the `drift_0` recatch | no |
| leave-one-family-out | three extra actors, each missing one family's imitation steps | that actor only |

`impact_zp14` scripted outcomes, unread at design time: wrist reseat retains, 20 ms recatch retains, inward slide retains but the relax-window change is −1.8 mm, so it misses the 2 mm execution cut.

## 5. Pretraining method

Family-balanced behavior cloning. A 167→128→128→7 network with tanh outputs is fit by weighted mean squared error for 120 Adam steps. Teacher-forced mean absolute error on the imitation steps is 0.041.

This loss has a real limit on `impact_zm6`. Wrist reseat, recatch, and the slide are three different action sequences from the same initial observation. A deterministic squared error fit averages those modes. Equal family weights stop the recatch from owning the loss. They do not give one state three actions. Mass and friction are not in the observation, so a different dynamics cell at the same pose does not create a new mode.

## 6. Stage A pilot

Learned handoff stayed off. The actor ran closed loop for 52 steps (the slide recipe length). Nominal continuation to t = 12 was queried at steps 0, 16, 32, and 52, and at a contact break when one occurred more than four steps from those times. Step 0 is before any recovery action. A further rollout on `impact_zm6` was queried at step 60, eight steps past that horizon.

| rollout | action signature over the first 0.40 s | oracle at 16 / 32 / 52 |
| --- | --- | --- |
| `impact_zm6` nominal | wrist-reseat-like | success / success / success |
| `impact_zp6` nominal | unstructured | failure / success / failure |
| `impact_zp14` (held-out IC) | unstructured | success / success / success |
| `impact_zm6` at 0.18 kg / 0.55 | wrist-reseat-like | success / success / success |
| `drift_0` nominal | unstructured | success / success / failure |
| `drift_0` at 0.18 kg / 0.55 | unstructured | success / success / failure |
| `impact_zm6` through step 60 | wrist-reseat-like | success at step 60 |

Every one of these starts as `HANDOFF_FAILURE` at step 0. The actor does not drop the primary impact on the first queries. On `drift_0` and `impact_zp6` it reaches an oracle-positive state and then loses it by step 52, because nothing is allowed to stop the actor early.

Leave-one-family-out, same horizon, endpoint oracle:

| actor | state | signature | endpoint |
| --- | --- | --- | --- |
| without wrist demos | `impact_zm6` | unstructured | success |
| without recatch demos | `drift_0` | gravity-orientation | success |
| without slide demos | `impact_zm6` | wrist-reseat-like | success |

The full actor's closed-loop signature is wrist-like on the primary impact and unstructured elsewhere. The gravity-orientation signature appears only in the actor that never saw the recatch demos. One deterministic network fit to all three recipes does not execute more than one mechanism.

Stage B gate from the pilot, which is stricter than the preparation gate:

| requirement | pilot |
| --- | --- |
| more than one mechanism in closed loop | no |
| no immediate destruction on the key initial conditions | yes on the impact cells; the drift cell is still held at 0.32 s and lost by 1.04 s |
| an oracle-positive state on nontrivial cases | yes, including held-out `impact_zp14` and the 0.18 kg / 0.55 impact cell |
| some held-out robustness | yes for that impact cell; the drift endpoint is not |

Do not start Stage B on this actor.

## 7. Handoff dataset and head

The handoff file is the 28 oracle queries on the full-actor rollouts above. Each query is `HANDOFF_SUCCESS` or `HANDOFF_FAILURE`. The sample times are before recovery, during recovery, the endpoint, and one over-control step. Adjacent frames are not labeled. Scripted demonstration endpoints were used to decide which open-loop rows to imitate. They are not a learned stop.

`HandoffHead` is a 167→64→1 logit on the same history. Its `fit` method raises if called. Stage A did not call it.

Planned sequence:

| stage | what runs | what stays fixed |
| --- | --- | --- |
| A | recovery-actor pretraining from the legal demonstrations | learned handoff disabled; external oracle only at the evaluation times above |
| B | handoff logit on the actor's own states, labeled by nominal continuation to t = 12 | actor frozen, or almost frozen |
| C | joint off-policy fine-tuning of recovery and handoff | only after A produces more than one mechanism and B has a handoff head that does not fire on the pre-recovery state |

Stage C is not started. Unrestricted handoff is not in the Stage A loss.

## 8. What this does not claim

The pilot is one network on these initial conditions. Teacher-forced error on the imitation steps is not held-out generalization. Success of a scripted recipe on a held-out cell is not success of the actor. The actor's held-out results are the closed-loop rows in section 6.

READY FOR RECOVERY POLICY PRETRAINING
