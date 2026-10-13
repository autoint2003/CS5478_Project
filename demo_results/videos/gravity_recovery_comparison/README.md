# ZERO vs HEURISTIC — complete pick-and-lift task

Presentation of a continuous nominal grasping task with an in-lift disturbance.
Physics, disturbance family, RULE, SAC, reward, and observation were not changed.
Heuristic parameters were not retuned.

**HEURISTIC** means the *manually constructed* gravitational-reposition sequence.
It is **not** `rule_based_recovery.yaml`, and not SAC.

## USER VISUAL VALIDATION: PENDING

Until the user watches the videos, do not treat them as visually confirmed.

Desired visual sequence to inspect:

    hand starts above object → descends → grasps → starts lifting
        → disturbance suddenly tilts/displaces cylinder DURING lift
        → ZERO vs HEURISTIC diverge → final lift/hold outcome.

## Experiment identity

**B. IN-LIFT DISTURBANCE CREATES A NEW POST-DISTURBANCE STATE `S_FAIL_LIFT`;
old S_FAIL recovery evidence is not being reused.**

canonical parent is a frozen airborne hold (v_cmd≈0 after freeze), not a live in-lift state; do not splice it into a continuing LIFT

The previous videos started from an airborne parent / `sfail.pkl` (or a
presentation of that construction). They hid APPROACH/DESCEND/CLOSE and
applied the disturbance before a separate lift continuation.

This revision runs the actual nominal Cartesian GraspFSM from reset.
Disturbance is applied while LIFT is live. Controller targets are not reset.
`sfail.pkl` is left untouched.

## Disturbance trigger

```json
{
  "definition": "fsm.phase==lift AND captured AND obj_z>=Z_AIR (0.48 m) AND (hand_z - hand_z_at_lift_start) >= 0.05 m",
  "Z_AIR": 0.48,
  "HAND_RISE_M": 0.05,
  "obj_z_pre": 0.48087303782406465,
  "hand_rise": 0.050064140635850185
}
```

## Parent embed audit

```json
{
  "verdict": "B",
  "reason": "canonical parent is a frozen airborne hold (v_cmd\u22480 after freeze), not a live in-lift state; do not splice it into a continuing LIFT",
  "parent_v_cmd": [
    0.0,
    0.0,
    0.0
  ],
  "parent_phase": "lift",
  "parent_time": 4.111999999999769,
  "parent_obj_z": 0.4961196938563631,
  "frozen_parent": true,
  "best_replay_qpos_diff": 0.0,
  "first_airborne_live": {
    "t": 3.9139999999997905,
    "obj_z": 0.4800382065574465,
    "v_cmd": [
      0.0,
      0.0,
      0.08
    ],
    "qpos_vs_parent": 0.040709116003618595,
    "p_des_vs_parent": 0.004569404615981654,
    "v_cmd_vs_parent": 0.08
  },
  "old_apply_sfail_vs_sfail_pkl": {
    "diffs": {
      "qpos": 0.0,
      "qvel": 0.0,
      "p_des": 0.0,
      "r_des": 0.0,
      "v_cmd": 0.0,
      "ctrl": 0.0,
      "time": 0.0,
      "phase_same": true
    },
    "match": true
  },
  "sfail_pkl_untouched": true
}
```

## Controller at disturbance

```json
{
  "untouched": true,
  "diffs": {
    "p_des": 0.0,
    "r_des": 0.0,
    "v_cmd": 0.0,
    "w_cmd": 0.0,
    "phase_same": true,
    "t_phase": 0.0,
    "lift_started_same": true,
    "t_held": 0.0
  },
  "v_cmd_pre": [
    0.0,
    0.0,
    0.08
  ]
}
```

## S_FAIL_LIFT vs old sfail.pkl

- old parent: `D:/NUS/CS5478 Intelligent Robots/CS5478_Project/demo_results/diagnostics/grav_reposition_recovery_construction/raw/parent.pkl`
- old S_FAIL: `D:/NUS/CS5478 Intelligent Robots/CS5478_Project/demo_results/diagnostics/grav_reposition_recovery_construction/raw/sfail.pkl` (untouched)
- new pre-disturbance: `D:/NUS/CS5478 Intelligent Robots/CS5478_Project/demo_results/videos/gravity_recovery_comparison/raw/pre_disturbance_lift.pkl`
- new S_FAIL_LIFT: `D:/NUS/CS5478 Intelligent Robots/CS5478_Project/demo_results/videos/gravity_recovery_comparison/raw/sfail_lift.pkl`
- S_FAIL_LIFT sha256: `9830db40cfa9ee69926bc86d848ecd42dd7c4526947ed05d97dcedb1f5af5f59`
- ZERO / HEURISTIC post-disturbance hashes equal: **True**
- disturbance: `coupled_pose_from_ref(ref, s=2.0, kind='coupled') + apply_coupled_pose`
- vs old sfail.pkl match: **False**

```json
{
  "diffs": {
    "qpos": 0.038696928580087775,
    "qvel": 0.018261642456313905,
    "p_des": 0.004577857476512126,
    "r_des": 0.021430216156933662,
    "v_cmd": 0.08,
    "ctrl": 0.12623523136419834,
    "time": 0.18799999999997974,
    "phase_same": true
  },
  "match": false
}
```

## Physics

```json
{
  "mujoco": "3.12.0",
  "noslip_iterations": 1,
  "noslip_tolerance": 1e-06,
  "timestep": 0.002,
  "gravity": [
    0.0,
    0.0,
    -9.81
  ]
}
```

m=0.20 kg, pair μ=1.0, noslip_iterations=1, coupled teleport s=2.0, pose-only.

## ZERO

Nominal FSM through disturbance, then remaining LIFT (same p_des integration,
no snap) → HIGH HOLD 12 s or drop + 2 s.

```json
{
  "t0": 3.9239999999997894,
  "t_lift_done": 5.417999999999625,
  "t_drop": 11.91600000000217,
  "t_both_loss": 11.91600000000217,
  "lift_completed": true,
  "drop": true,
  "hold_s": 12.0,
  "hold_survived": false,
  "end_nL": 0,
  "end_nR": 0,
  "end_obj_z": 0.01791868449380277,
  "end_p_des_z": 0.7130316394873875,
  "z_tgt": 0.7129664062868253,
  "dt": 0.002,
  "v_lift": 0.08,
  "tau": -18.0
}
```

## HEURISTIC

Same prefix. After disturbance: ROTATE +30° τ=−18 → WEAK REPOSITION τ=−4
until ~1 mm useful x → SECURE 0.5 s τ=−18 → RETURN. Then **resume** remaining
lift to the original `grasp_z + lift_offset_z` (no second lift from a snapped p_des).

```json
{
  "t0": 6.563999999999499,
  "t_lift_done": 8.381999999999469,
  "t_drop": null,
  "t_both_loss": null,
  "lift_completed": true,
  "drop": false,
  "hold_s": 12.0,
  "hold_survived": true,
  "end_nL": 6,
  "end_nR": 6,
  "end_obj_z": 0.5992018247093056,
  "end_p_des_z": 0.7130219881109461,
  "z_tgt": 0.7129664062868253,
  "dt": 0.002,
  "v_lift": 0.08,
  "tau": -18.0
}
```

## Actual outcomes (not forced)

- ZERO dropped: **True**
- HEURISTIC hold survived 12 s: **True**

## Playback / render

- task 0.7×, lift 0.6×, disturbance 0.35×,
  recovery 0.4×, high hold 1.5×, near drop 0.5×
- playback is sleep/frame skip only; dt=0.002 and step count are physical
- ~0.40 s presentation freeze after the teleport (no mj_step)
- video 25 fps, 640×480
- camera distance 1.65 m, azimuth 142.0, elevation -22.0, one-shot
- superseded airborne-opening mp4s: `superseded_airborne_opening/`

## Commands

```text
python training/demo_grav_reposition_comparison.py
python training/demo_grav_reposition_comparison.py --mode zero
python training/demo_grav_reposition_comparison.py --mode heuristic
python training/demo_grav_reposition_comparison.py --render-video side_by_side
```

## Paths

- videos: `D:/NUS/CS5478 Intelligent Robots/CS5478_Project/demo_results/videos/gravity_recovery_comparison`
- raw: `D:/NUS/CS5478 Intelligent Robots/CS5478_Project/demo_results/videos/gravity_recovery_comparison/raw`