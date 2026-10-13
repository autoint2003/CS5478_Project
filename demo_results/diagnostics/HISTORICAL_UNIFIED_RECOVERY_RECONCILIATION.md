# Historical unified recovery reconciliation

The balanced comparison was read from git commit `8bba335^` (`demo_results/diagnostics/BALANCED_UNIFIED_RECOVERY_EVALUATION.md`). That file is not in the working tree. It was deleted in `8bba335` and was not restored. No checkpoint was retrained, and no new dataset was generated.

## 1. Inventory

| artifact | role | status |
| --- | --- | --- |
| `training/balanced_unified_recovery.py` | 24-case definitions, equal-family trainer, `mechanism_of` | present. `main` raises immediately so it cannot rewrite the archived report |
| `training/recovery_policy_pretraining_preparation.py` `schedules()` | open-loop wrist, slide, and recatch commands | present |
| `training/natural_loss_recatch_transformer_v3.py` `record_local` | converts those commands with `v1_normalized_to_v3`, steps `recovery7d_airborne_v3`, keeps a demo only if `continue_long` retains to absolute t = 12 | present |
| `demo_results/diagnostics/raw/balanced_unified_recovery/ckpt.pt` | seed-1 120-epoch balanced weights, K=32 | missing |
| `demo_results/diagnostics/raw/natural_loss_controlled_dagger/ckpt_round0.pt` | baseline the report compares against | missing |
| `demo_results/diagnostics/raw/balanced_unified_recovery/ckpt_39d.pt` | later 39-D retraining | missing |
| `demo_results/diagnostics/raw/balanced_unified_recovery/summary.json` | per-case log of the 45-D run | missing |
| `demo_results/diagnostics/raw/balanced_39d/summary.json` | saved 39-D run, not a replay from this audit | present |
| `demo_results/diagnostics/raw/autonomous_training/demo_{wrist_reseat,gravity_inward,recatch_open20}.npz` | stored 39-D demonstration sequences | present |

The published score is baseline 13/24 and balanced 15/24. Family counts match the table in the request: impact 2/4 to 3/4, sliding 4/4 to 4/4, wrist 3/4 to 4/4, open-finger recatch 4/4 to 4/4, airborne 0/4, natural-loss 0/4. Success is bilateral contact above the lift height at absolute t = 12 after the nominal controller continues. It is not the current 3.03 mm / 0.117 rad basin.

The report text says the observation is 45-D. The code that produced it already called `observe_airborne`. The later 39-D vector is indices 0–38 of that vector. Indices 39–44 were a second copy of the hand twist. The 39-D prefix still contains `g_hx`, `g_hy`, and `g_hz`.

Training did not feed family labels to the network. `train_equal` draws equal batch share from whichever pools have demonstrations. The published run had one wrist trajectory, one sliding trajectory, one recatch trajectory, 18 natural-loss windows, and zero airborne-carry demonstrations. Batch fractions were 0.25 on those four pools. The tokenizer is K=32. The action map is `recovery7d_airborne_v3`.

## 2. What the three schedules actually command

All three are open-loop lists in `schedules()`, recorded on a 6.5 m/s impact with axial offset −6 mm (`zm6`). They are not the CENTER-6 large-angle sequence, and they are not the later ball-impact A/B search.

**Wrist reseating** (`wrist_reseat`, 18 steps). For 0.24 s the normalized command is hand-x speed 0.6 on the old 0.08 m/s map (0.048 m/s after the v3 rewrite), wrist-y 0.8 (3.2 rad/s), grip +1 (`τ = -18 N·m`). Then six secure hold steps. Grip stays closed. This is a short secure rotation plus a small translation. It is not the CENTER-6 sequence, which freezes the Cartesian pose, rotates about +120° at a secure grip, and only then unloads to `τ = -5`.

**Inward sliding** (`gravity_inward`, 52 steps). For 0.40 s, `wx = 0.28` and `wy = 0.50` (1.12 and 2.0 rad/s) with grip +1. Then 0.48 s at grip −0.48 (`τ = -3.2 N·m`) with the wrist stopped. Then eight steps at grip +1. Translation is zero. The opposite sign, `gravity_outward`, flips only `wy`.

The code's own execution flag, `mechanism_flags`, asks for a hand-frame x change of at least −2 mm between log rows 19 and 43, with both fingers still in contact. It does not ask for a unilateral support phase.

**Open-finger recatch** (`recatch_open20`, 32 steps). Nineteen steps of wrist-y = 1.0 at grip +1, one step at grip −1 (`τ = +2 N·m`), then twelve steps at grip +1. Translation is zero. The historical flag is one interval with both contacts gone and a later return of both contacts. Current release–chase–regrasp is a different controller: it opens, translates to follow the cylinder, and closes near the pinch. The historical recatch does not chase.

## 3. Open-loop replay on a fresh zm6

`training/audit_historical_skill_physics.py` rebuilt the impact and replayed the schedules through `v1_normalized_to_v3` and `step_v3`, then `continue_long` to t = 12. Raw log: `demo_results/diagnostics/raw/historical_unified_reconciliation/schedule_physics.json`. This is the demonstration generator. It is not the Transformer.

| schedule | t = 12 | bilateral steps | unilateral steps | no-contact steps | pinch x | hand-x angle to world up |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `wrist_reseat` | retained | 18/18 | 0 | 0 | 9.35 → 8.98 mm | 89° → 45° |
| `gravity_inward` | retained | 52/52 | 0 | 0 | 9.35 → 7.40 mm | 89° → 46° |
| `gravity_outward` | physical failure | 28 | 1 | 23 | 9.35 → 102 mm | 89° → 134° |
| `recatch_open20` | retained | 31 | 0 | 1 | 9.35 → 3.84 mm | 89° → 1.8° |

On `gravity_inward` the indexed unload window, rows 19 to 43, moves pinch x by −0.83 mm. That misses the code's own −2 mm flag. The larger part of the 1.9 mm pinch change happens while the grip is still `τ = -18`. During the unload the aperture goes from 17.8 mm to 18.0 mm, both pads stay in contact (6/6, then 10/10), and `g_h.x` is about −6.8 m/s². Hand z stays about 128° from world up. The finger plane is tilted, not a near-horizontal shelf, and the cylinder never rides on one finger.

`gravity_outward` points `g_h.x` the other way (+6.8 m/s²), spins the cylinder (relative rate about 10 rad/s), and loses the grasp. Sign matters. The successful sign still does not implement single-sided support.

`recatch_open20` spends one 20 ms step with both contacts gone and `τ = +2`, then regrasps. At the end of the command, pinch x is 3.84 mm and `g_h.x` is −9.81 m/s² because the pinch axis has been turned nearly vertical. The later nominal hold is what the t = 12 label scores. The earlier pose audit measured these three scripts at hand errors 0.78, 0.93, and 1.60 rad with pinch still about 10–13 mm, and classified them `RECATCh_SUCCEEDED_BUT_POSE_NOT_RESTORED` under the current basin.

## 4. The four sliding evaluation cases were not re-run closed-loop

The four sliding rows are built in `balanced_unified_recovery.py`, not loaded from a pickle:

- `slide_suffix_20` and `slide_suffix_36`: `zm6` after 20 and 36 steps of the open-loop `gravity_inward` list
- `slide_dy1` and `slide_dz1`: the same `zm6` with the object joint shifted by 1 mm in y or z

The published balanced labels on those four, from the deleted report, are `recontact`, `contact-preserving`, `wrist reseat`, and `wrist reseat`. All four were `RETAINED_TO_12` for both the baseline and the balanced policy. `mechanism_of` assigns `wrist reseat` when contact is never lost and some `|wy|` exceeds 0.4. It assigns `contact-preserving` when contact is kept without that wrist command. It never reads gravity, a support plane, or unilateral contact.

Those labels cannot be regenerated. `ckpt.pt` and `ckpt_round0.pt` are gone, and `main` refuses to train a replacement. The 39-D summary that is still on disk, from a later training run, labels the same four ids `wrist reseat`, `contact-preserving`, `wrist reseat`, and `wrist reseat`, and also records retention. That file is not a replay performed in this audit, and `ckpt_39d.pt` is gone as well.

## 5. Dominance

The only historical use of "dominant" is the column written by `mode()` over `mechanism_of` on the retained cases of each family. For the sliding family that mode is `wrist reseat` (two of the four retained rollouts). No report, trajectory field, or commit message states that gravity-assisted unilateral sliding was the dominant physical mechanism, the most common token, or the fastest recovery. The 4/4 figure is retention of a disturbance family. It is not a count of B executions.

That claim is contradicted by the published mechanism column.

## 6. Historical skills versus the current experiments

| | historical unified policy | current ball-impact line |
| --- | --- | --- |
| observation | 45-D, later cut to 39-D by dropping a duplicate hand twist. Gravity components are in the prefix | 39-D `observe_airborne` |
| actions | K=32, `recovery7d_airborne_v3` | K=36, residual ±0.25, same v3 map |
| A in that line | secure 0.24 s wrist-y plus 0.048 m/s hand-x | CENTER-6 is a different experiment: secure +120°, then `τ = -5`, about 1 mm of bilateral slip, residual near 8 mm |
| B in that line | secure combined `wx`/`wy` for 0.40 s, then `τ = -3.2` with both pads still touching | the failed search tilted and translated until the cylinder tumbled. The owner's single-sided shelf is in neither implementation |
| C | one 20 ms open after a large secure wrist-y, no chase translation | release, chase, regrasp, with many zero-contact steps |
| score | bilateral grasp above the lift height at absolute t = 12 | nominal basin, explicit handoff, 10 s nominal-only hold |

The three historical open-loop action lists sit on current K=36 prototypes with residual 0, measured against `policy_B_init.pt`. Representing the script is not the same as closed-loop transfer. The weights that executed the 15/24 policy are missing, and the scripts themselves finish outside the 3.03 mm basin.

The current 39-D vector does contain the hand-frame gravity components and the relative pinch, so the missing piece for the historical tilt is not an absent gravity channel. The missing piece for the owner's single-sided B is that no verified trajectory of that mechanism exists to represent.

## 7. Reusable demonstrations

Preserved, with the original t = 12 criterion, not relabeled as basin successes:

- `demo_wrist_reseat.npz`, `demo_gravity_inward.npz`, `demo_recatch_open20.npz` (39-D observations, 7-D actions)
- the open-loop replay log cited above

They can be replayed as legal v3 actions. They are exactly representable by the current codebook. Replaying them does not produce a 3.03 mm handoff. The closed-loop token traces from the 15/24 evaluation were not stored, and the weights that would regenerate them are missing.

## 8. Blockers

1. `ckpt.pt` and `ckpt_round0.pt` are absent, so the four sliding Transformer rollouts cannot be reproduced.
2. Historical B, as implemented, never enters single-sided support. Verifying the owner's B requires a different controller, which this audit did not add.
3. t = 12 retention and the current basin are different outcomes. The three scripts meet the old one and not the new one.
4. `slide_dy1` and `slide_dz1` are 1 mm edits of the object joint after impact. They are historical evaluation cases, not ball-impact training states.

## 9. Next experiment

Reload is impossible until the seed-1 K=32 checkpoint is found outside this tree. Do not retrain it and call the result a reproduction. The measurement that is still available is already done: the open-loop `gravity_inward` list retains the object and does not perform unilateral support. A later experiment, if the goal is the owner's B, has to be labeled a new controller and judged on support-plane angle, one-finger contact, and inward motion along that finger.

HISTORICAL UNIFIED RECOVERY REPRODUCTION: PARTIAL

HISTORICAL B MECHANISM VERIFIED: NO

HISTORICAL B DOMINANCE CLAIM: CONTRADICTED

CURRENT K36 TRANSFER READINESS: PARTIAL

POLICY TRAINING HAS NOT BEEN STARTED
