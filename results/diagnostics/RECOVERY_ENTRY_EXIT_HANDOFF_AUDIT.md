# Recovery entry, exit, and handoff audit

The active 24-case evaluation never decides to enter recovery. It restores a prepared snapshot and runs the 39-D policy for a fixed number of 20 ms steps. If the object stays above 0.44 m, `continue_long` then runs the nominal controller until absolute t = 12. No controller code was changed for this audit.

## Control flow

```text
prepared snapshot
    restore qpos, qvel, ctrl, time, FSM
    policy history = empty
    previous action = 0
        |
        v
for k in 0 .. n_steps-1                         eval_case / eval_policy
    observe_airborne (39) + previous action (7)
    Transformer
    step_v3: rebase p_des to the hand, hold v/w/grip for 10 sim steps
    if obj_z < 0.44: label DROP and stop          no nominal continuation
        |
        v
continue_long(sim, gains, t_end=12)              recovery_runtime.py
    while sim.data.time < 12
        if phase == lift and p_des.z >= z_tgt:
            tick_vw(v=0, w=0, tau=-18)
        else:
            GraspFSM.step(in_recovery=False)      resumes the lift integrator
    stop early only after a drop or escape has lasted 0.15 s
        |
        v
physical_of: DROP, ESCAPE, or RETAINED_TO_12
```

`DeteriorationMeter.update_mode` is a different function. It sets `in_recovery` when the deterioration score `D` exceeds 0.85, and clears it after 10 updates below 0.40 (`envs/deterioration.py`, thresholds in `config/nominal.yaml`). The only caller is `run_to_lift_or_deterioration` in `envs/grasp_sim.py`. `eval_case`, `eval_policy`, and `continue_long` do not call it. The 24-case benchmark does not use that flag.

`HandoffHead` in `training/recovery_policy_pretraining_preparation.py` is not called by the evaluation. Its `fit` method raises.

## How each benchmark family starts

| family | what the eval restores | how recovery begins |
|---|---|---|
| impact | `run_impact` snapshot, 40 ms after the ball is parked | C. Scripted impact, then the policy starts on that snapshot |
| sliding suffixes | `gravity_inward` played for 20 or 36 steps on the canonical impact, then stamped | B. Saved suffix of a scripted recovery |
| sliding shifts | canonical impact snapshot with the object qpos shifted 1 mm | C. Edited initial condition |
| wrist | `wrist_reseat` played for 3, 6, 9, or 12 steps, then stamped | B. Saved suffix |
| open-finger recatch | `recatch_open20` played for 8, 16, 24, or 28 steps, then stamped | B. Saved suffix |
| airborne carry | `carry_then_release`: sideways motion, then an open, stamped after both pads are clear | C. Scripted release |
| natural-loss | longest pre-loss window from `slip_until_loss` under the hold action | C. Offset grasp, snapshot taken while contact is still bilateral |

None of the six families is A. The policy loop does not read contact, force, pose, phase, or `D` to decide whether to start. `n_steps` is stored on the case: 56 for a full impact or shift, `max(12, 56 - suffix index)` for a suffix, and 120 for airborne carry and natural-loss.

## What is initialized at the start of that loop

`load_snapshot` (`envs/grasp_sim.py`) copies `qpos`, `qvel`, `ctrl`, `data.time`, FSM phase, `t_phase`, `p_des`, `r_des`, `v_cmd`, `w_cmd`, and the meter flag. On `impact_zm6` that restored state is absolute t = 2.948 s, phase `lift`, `p_des` 17.1 mm from the hand, and `p_des.z` 104 mm below `z_tgt` (0.713 m). `lift_hold` is already false.

The evaluation then creates a new `ObservableObsState` and sets the previous action to the zero vector. The Transformer context is the growing list of recovery observations. It is not padded and it does not contain the nominal approach. There is no separate recovery clock.

The restored `p_des` is visible to the first observation through `track_z`. The first `step_v3` replaces it. `step_v3` sets `p_des` to the current hand position, writes `v_cmd` / `w_cmd` from the v3 action, and integrates `r_des` for ten simulation steps of 2 ms. The snapshot setpoint does not keep commanding the arm after that step.

## Policy execution

| quantity | value | source |
|---|---|---|
| policy period | 20 ms | `DT_POLICY` |
| simulation step | 2 ms | `model.opt.timestep` |
| action hold | 10 simulation steps | `N_SUB` in `step_v3` |
| horizon | `n_steps` × 20 ms | the case record |
| previous action at step 0 | zeros | `eval_case` |
| context | empty, then appended | same loop |
| early stop | `obj_z < 0.44` | same loop, labeled DROP with no `continue_long` |

There is no success flag that ends the policy early. A 1 s regrasp is not read by this loop.

## Exit and handoff

Recovery stops for one of two reasons:

1. The for-loop reaches `n_steps`. `continue_long` is called with `t_end = 12`.
2. `obj_z` falls below 0.44 m. The case is DROP and nominal continuation does not run.

`continue_long` does not test contact, a policy flag, or `fsm.success`. While `data.time < 12` it does this:

- if `phase == "lift"` and `p_des.z >= z_tgt`, hold with `tick_vw(0, 0, tau=-18)`;
- otherwise call `physics_step(in_recovery=False)`, which is `GraspFSM.step`.

In the lift phase that FSM step overwrites the command with `v_cmd = [0, 0, lift_speed]` (`lift_speed` is 0.08 m/s) and integrates `p_des`. Grip goes to the nominal secure torque. The recovery `v_cmd` is replaced on that first nominal step. `p_des` and `r_des` are the integration base, so they are the values left by the last `step_v3`, not the pre-recovery lift target.

## Absolute time

`data.time` is restored from the snapshot and is not set back to zero when the policy starts or when `continue_long` starts. The impact policy occupies 2.948 s to 4.068 s, which is 56 × 20 ms. `continue_long` then advances the same clock until `t >= 12`. The four retained probes below all end at absolute t = 12.0. The step budget `n = (12 - t0) / dt + 50` cannot extend the logged success time, because the loop breaks at `t_end`.

## Matched handoff on retained cases

The state at the end of the policy was saved. Branch A is the current `continue_long` path. Branch B copies that state, sets `p_des` and `r_des` to the current hand, zeroes `v_cmd` and `w_cmd`, and then runs the same continuation. Both branches reach t = 12 with the same label.

| case | time at handoff | p_des − hand | orientation error | p_des.z − z_tgt | first nominal mode | A t=12 | B t=12 |
|---|---:|---:|---:|---:|---|---|---|
| impact_zm6 | 4.068 s | 0.36 mm | 0.12 deg | −128 mm | nominal lift | RETAINED_TO_12 | RETAINED_TO_12 |
| slide_suffix_20 | 4.068 s | 0.37 mm | 0.11 deg | −148 mm | nominal lift | RETAINED_TO_12 | RETAINED_TO_12 |
| wrist_suffix_3 | 4.068 s | 0.36 mm | 0.12 deg | −144 mm | nominal lift | RETAINED_TO_12 | RETAINED_TO_12 |
| recatch_suffix_8 | 4.068 s | 0.37 mm | 0.13 deg | −136 mm | nominal lift | RETAINED_TO_12 | RETAINED_TO_12 |

At the handoff instant the arm command is already the rebased hand pose. Grip is about −18 N·m and both pads are in contact. The object is at 0.51–0.54 m, above the 0.51 m retention height. `z_tgt` is 0.713 m, so `lift_hold` is false.

The first nominal step commands `v = [0, 0, 0.08]` m/s. By 200 ms, `p_des` is about 14 mm above the hand on every retained probe. Branch B starts 0.2 mm closer and is still about 14 mm behind the rising setpoint at 200 ms. Contact counts stay bilateral. Object height stays near 0.52 m. The official `continue_long` label matches branch A on all four cases.

The pre-recovery setpoint does not reappear. On `impact_zm6` it was 17 mm away at restore and 0.36 mm away after the policy. The discontinuity at handoff is the lift law turning on, because the hand is still well below `z_tgt`. Putting the setpoint exactly on the hand does not change the t = 12 result.

## Where the two failures happen

`air_lat_pos` runs all 120 policy steps, from t = 4.606 s to t = 7.006 s. Bilateral contact is present on 115 of those steps. Object height reaches 0.436 m, the loop labels DROP, and `continue_long` is not called.

`nl_h_dz14.00` runs 103 of 120 steps, from t = 7.176 s to t = 9.236 s. Bilateral contact is present on 99 steps. Object height reaches 0.435 m, and nominal continuation is not called.

Both failures are inside the policy window. They are not a later nominal drop and not a handoff applied to a grasp that had already finished the policy above 0.44 m.

## Minimal semantics that match this evidence

Entry, on an online episode, should be a physical event during nominal lift: bilateral contact lost and still lost after a short persistence, the same kind of persistence `slip_until_loss` already uses to mark a loss. The benchmark's snapshots can remain the way those episodes are constructed. They are not a substitute for that trigger. `D` is not on this path and is not required to explain the retained recoveries. A learned entry classifier is not required by these traces.

Stay in recovery for the policy steps, and leave immediately if the object falls below 0.44 m, which is what the loop does now. A fixed step count is a budget, not a measurement that the grasp has recovered.

Hand back with `continue_long` on the same absolute clock. On the four retained probes that handoff retains the object through t = 12 whether or not `p_des` is first snapped onto the hand. The behavior to keep explicit is that nominal lift resumes whenever `p_des.z < z_tgt`, even after the object is already above 0.51 m.

MULTIPLE CONTROL-FLOW ISSUES REMAIN
