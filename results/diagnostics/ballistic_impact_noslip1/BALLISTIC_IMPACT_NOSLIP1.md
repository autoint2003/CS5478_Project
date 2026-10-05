# Ballistic impact family under noslip=1

ZERO only. No recovery, RULE, SAC, recatch, or heuristic retune.

## USER VISUAL OBSERVATION: pending

Until the representative I0–I3 animations are watched, do not treat
the severity labels as visually confirmed.

## 1. Diagnostic scene construction

New diagnostic scene `assets/scene_ballistic_impact.xml` (not the official
nominal scene). Cylinder `contype/conaffinity=3`, ball `=2` so the ball
collides with the cylinder only (not fingers, arm, table, floor).
Historical `scene_impact_diag.xml` / 0.5 kg claims are quarantined in
`OLD_IMPACT_QUARANTINE.md`.

## 2. Actual ball mass / inertia audit

Masses are declared in XML before `MjModel` construction. No `set_ball_mass`.

```json
{
  "xml": {
    "scene": "D:\\NUS\\CS5478 Intelligent Robots\\CS5478_Project\\assets\\scene_ballistic_impact.xml",
    "xml_contains_ball_0p05": true,
    "xml_contains_cyl_0p20": true,
    "declared_ball_m": 0.05,
    "declared_ball_I": 4.5e-06,
    "declared_cyl_m": 0.2,
    "declared_cyl_Ixx": 7.62e-05,
    "declared_cyl_Izz": 3.2399999999999995e-05,
    "ball_r": 0.015
  },
  "noslip_iterations": 1,
  "ball": {
    "body_id": 13,
    "body_mass": 0.05,
    "body_inertia": [
      4.5e-06,
      4.5e-06,
      4.5e-06
    ],
    "qM_lin0": 0.05
  },
  "cylinder": {
    "body_id": 12,
    "body_mass": 0.2,
    "body_inertia": [
      7.62e-05,
      7.62e-05,
      3.24e-05
    ],
    "qM_lin0": 0.2
  },
  "runtime_set_ball_mass_used": false,
  "mj_setConst_used_for_ball": false,
  "ball_mass_ok": true,
  "cyl_mass_ok": true,
  "ball_qM_ok": true
}
```

Free-flight drop:

```json
{
  "t": 0.4,
  "z0": 1.1,
  "z_end": 0.3112759999999991,
  "z_pred_end": 0.3151999999999999,
  "max_abs_z_error": 0.003924000000000816,
  "g_model": -9.81,
  "eq_active": 0,
  "ok": true
}
```

## 3. Free-flight / release audit

Weld holds the ball at a visible pose until the lift trigger. At trigger:
weld off (`eq_active=0`), one write of free-flight `(p,v)`, then no ball
qpos/qvel overwrite through collision.

## 4. Nominal-task impact timing

Trigger (not tuned for recovery): `phase==lift` AND captured AND
`obj_z>=Z_AIR (0.48)` AND hand rise since lift start `>= 0.05 m`.
Impact occurs after a free-flight interval while LIFT is still integrating.

## 5. Impact frame / direction

Launch along **+hand-x** (`R_h[:,0]`), gravity-compensated so along-track
speed at the intended hit ≈ `v_hit`. Report `v_ball,h = R_h^T v_ball,world`.

## 6. Ball-cylinder contact audit

Runs are rejected if the first ball contact is not the cylinder.

## 7. Impulse and momentum cross-check

`J = Σ F_ball dt` over ball-cylinder contacts using `mj_contactForce`
(force-on-ball after geom1/geom2 sign). Cross-check `Δp_ball = m(v_after-v_before)`.

## 8. Coarse severity sweep

One mass `m=0.05 kg`. Sweep `v_hit` only.

```json
[
  {
    "v_hit": 1.2,
    "m_v": 0.06,
    "class": "STABLE",
    "t_release": 2.747999999999919,
    "t_impact": 2.873999999999905,
    "free_flight_s": 0.12599999999998612,
    "v_ball_h_pre": [
      1.2000516225052493,
      -0.012958807869837798,
      0.6155633972394057
    ],
    "J_mag": 0.06690057279478646,
    "impulse_consistency": 1.186550877381772e-15,
    "max_ex_mm": 0.2928885417803809,
    "unilateral": false,
    "t_drop": null,
    "hold_survived": true,
    "reject": null
  },
  {
    "v_hit": 1.8,
    "m_v": 0.09000000000000001,
    "class": "STABLE",
    "t_release": 2.747999999999919,
    "t_impact": 2.8319999999999097,
    "free_flight_s": 0.08399999999999075,
    "v_ball_h_pre": [
      1.8000447364823169,
      -0.008397044559238528,
      0.4038926977317213
    ],
    "J_mag": 0.09556193381711427,
    "impulse_consistency": 1.2959147466469352e-15,
    "max_ex_mm": 0.4255888609575603,
    "unilateral": false,
    "t_drop": null,
    "hold_survived": true,
    "reject": null
  },
  {
    "v_hit": 2.4,
    "m_v": 0.12,
    "class": "STABLE",
    "t_release": 2.747999999999919,
    "t_impact": 2.811999999999912,
    "free_flight_s": 0.06399999999999295,
    "v_ball_h_pre": [
      2.4000387805303256,
      -0.0063577616520812084,
      0.30786654631089044
    ],
    "J_mag": 0.12882974265352104,
    "impulse_consistency": 1.3013387050422135e-15,
    "max_ex_mm": 0.7006289506601093,
    "unilateral": false,
    "t_drop": null,
    "hold_survived": true,
    "reject": null
  },
  {
    "v_hit": 2.55,
    "m_v": 0.1275,
    "class": "STABLE",
    "t_release": 2.747999999999919,
    "t_impact": 2.8079999999999123,
    "free_flight_s": 0.05999999999999339,
    "v_ball_h_pre": [
      2.5500369899179063,
      -0.005904019315994483,
      0.28630445142050626
    ],
    "J_mag": 0.1348943247932239,
    "impulse_consistency": 1.2975734499242483e-15,
    "max_ex_mm": 0.7378047242722348,
    "unilateral": false,
    "t_drop": null,
    "hold_survived": true,
    "reject": null
  },
  {
    "v_hit": 2.7,
    "m_v": 0.135,
    "class": "DISTURBED_CAPTURED",
    "t_release": 2.747999999999919,
    "t_impact": 2.8059999999999126,
    "free_flight_s": 0.05799999999999361,
    "v_ball_h_pre": [
      2.700036948612737,
      -0.005819207303606058,
      0.2823982326980527
    ],
    "J_mag": 0.14850726078961748,
    "impulse_consistency": 1.2822611531947469e-15,
    "max_ex_mm": 1.1132928455541284,
    "unilateral": true,
    "t_drop": null,
    "hold_survived": true,
    "reject": null
  },
  {
    "v_hit": 2.85,
    "m_v": 0.14250000000000002,
    "class": "DISTURBED_CAPTURED",
    "t_release": 2.747999999999919,
    "t_impact": 2.801999999999913,
    "free_flight_s": 0.05399999999999405,
    "v_ball_h_pre": [
      2.8500345105969638,
      -0.005292437335033986,
      0.2572177237082963
    ],
    "J_mag": 0.15023781948639642,
    "impulse_consistency": 1.2993084308549064e-15,
    "max_ex_mm": 1.0740399860168293,
    "unilateral": true,
    "t_drop": null,
    "hold_survived": true,
    "reject": null
  },
  {
    "v_hit": 2.95,
    "m_v": 0.14750000000000002,
    "class": "DISTURBED_CAPTURED",
    "t_release": 2.747999999999919,
    "t_impact": 2.7999999999999132,
    "free_flight_s": 0.05199999999999427,
    "v_ball_h_pre": [
      2.9500334450447094,
      -0.0050613920328989835,
      0.24617639127581217
    ],
    "J_mag": 0.15415877742808237,
    "impulse_consistency": 1.4085959673747843e-15,
    "max_ex_mm": 1.210672704427622,
    "unilateral": true,
    "t_drop": null,
    "hold_survived": true,
    "reject": null
  },
  {
    "v_hit": 2.967,
    "m_v": 0.14835,
    "class": "IMMEDIATE_LOSS",
    "t_release": 2.747999999999919,
    "t_impact": 2.7999999999999132,
    "free_flight_s": 0.05199999999999427,
    "v_ball_h_pre": [
      2.9670336353794964,
      -0.005090195608302727,
      0.2475773865514603
    ],
    "J_mag": 0.1560152863369837,
    "impulse_consistency": 1.301042769612921e-15,
    "max_ex_mm": 1.3602858213416602,
    "unilateral": true,
    "t_drop": 0.005999999999999339,
    "hold_survived": false,
    "reject": null
  },
  {
    "v_hit": 2.983,
    "m_v": 0.14915,
    "class": "IMMEDIATE_LOSS",
    "t_release": 2.747999999999919,
    "t_impact": 2.7999999999999132,
    "free_flight_s": 0.05199999999999427,
    "v_ball_h_pre": [
      2.9830338125361426,
      -0.005117004368125686,
      0.24888139209638796
    ],
    "J_mag": 0.15822695570877776,
    "impulse_consistency": 1.3013419252893017e-15,
    "max_ex_mm": 1.4214253970009052,
    "unilateral": true,
    "t_drop": 0.005999999999999339,
    "hold_survived": false,
    "reject": null
  },
  {
    "v_hit": 3.0,
    "m_v": 0.15000000000000002,
    "class": "DISTURBED_CAPTURED",
    "t_release": 2.747999999999919,
    "t_impact": 2.7999999999999132,
    "free_flight_s": 0.05199999999999427,
    "v_ball_h_pre": [
      3.0000339986943274,
      -0.005145174728480592,
      0.25025166676011373
    ],
    "J_mag": 0.16026838346647052,
    "impulse_consistency": 1.2978699952728162e-15,
    "max_ex_mm": 1.6881691858954313,
    "unilateral": true,
    "t_drop": null,
    "hold_survived": true,
    "reject": null
  },
  {
    "v_hit": 3.017,
    "m_v": 0.15085,
    "class": "DISTURBED_CAPTURED",
    "t_release": 2.747999999999919,
    "t_impact": 2.7999999999999132,
    "free_flight_s": 0.05199999999999427,
    "v_ball_h_pre": [
      3.017034182754368,
      -0.0051730269889658,
      0.25160650871631385
    ],
    "J_mag": 0.16266596086520096,
    "impulse_consistency": 1.3013392831516745e-15,
    "max_ex_mm": 1.7605097932319167,
    "unilateral": true,
    "t_drop": null,
    "hold_survived": true,
    "reject": null
  },
  {
    "v_hit": 3.033,
    "m_v": 0.15165,
    "class": "IMMEDIATE_LOSS",
    "t_release": 2.747999999999919,
    "t_impact": 2.7999999999999132,
    "free_flight_s": 0.05199999999999427,
    "v_ball_h_pre": [
      3.0330343541022993,
      -0.005198955088956399,
      0.25286778879381117
    ],
    "J_mag": 0.16477085378142015,
    "impulse_consistency": 1.3056928961620209e-15,
    "max_ex_mm": 1.896364506733404,
    "unilateral": true,
    "t_drop": 0.0039999999999995595,
    "hold_survived": false,
    "reject": null
  },
  {
    "v_hit": 3.05,
    "m_v": 0.1525,
    "class": "IMMEDIATE_LOSS",
    "t_release": 2.747999999999919,
    "t_impact": 2.7999999999999132,
    "free_flight_s": 0.05199999999999427,
    "v_ball_h_pre": [
      3.0500345341894475,
      -0.005226205018889657,
      0.25419340850842154
    ],
    "J_mag": 0.1673284376356079,
    "impulse_consistency": 1.2978706292699406e-15,
    "max_ex_mm": 1.9633675428047443,
    "unilateral": true,
    "t_drop": 0.0039999999999995595,
    "hold_survived": false,
    "reject": null
  },
  {
    "v_hit": 3.2,
    "m_v": 0.16000000000000003,
    "class": "IMMEDIATE_LOSS",
    "t_release": 2.747999999999919,
    "t_impact": 2.7959999999999137,
    "free_flight_s": 0.047999999999994714,
    "v_ball_h_pre": [
      3.2000314833807386,
      -0.004640066494008559,
      0.22603747122875753
    ],
    "J_mag": 0.16633353305681556,
    "impulse_consistency": 1.4085969687923323e-15,
    "max_ex_mm": 1.5273297008591593,
    "unilateral": true,
    "t_drop": 0.005999999999999339,
    "hold_survived": false,
    "reject": null
  },
  {
    "v_hit": 4.0,
    "m_v": 0.2,
    "class": "IMMEDIATE_LOSS",
    "t_release": 2.747999999999919,
    "t_impact": 2.7879999999999145,
    "free_flight_s": 0.039999999999995595,
    "v_ball_h_pre": [
      4.000028187705174,
      -0.003941288006623681,
      0.19263832498947978
    ],
    "J_mag": 0.21163403372985629,
    "impulse_consistency": 1.4016565853621897e-15,
    "max_ex_mm": 3.763185146556186,
    "unilateral": true,
    "t_drop": 0.0039999999999995595,
    "hold_survived": false,
    "reject": null
  }
]
```

## 9. Regime boundaries

```json
{
  "STABLE": {
    "first_v": 1.2,
    "m_v": 0.06
  },
  "DISTURBED_CAPTURED": {
    "first_v": 3.0,
    "m_v": 0.15000000000000002
  },
  "IMMEDIATE_LOSS": {
    "first_v": 3.033,
    "m_v": 0.15165
  }
}
```

## 10. Representative I0–I3

```json
{
  "I0": {
    "v_hit": 1.2,
    "class": "STABLE",
    "t_impact": 2.873999999999905,
    "J_mag": 0.06690057279478646,
    "max_ex_mm": 0.2928885417803809,
    "t_drop": null,
    "hold_survived": true,
    "v_ball_h_pre": [
      1.2000516225052493,
      -0.012958807869837798,
      0.6155633972394057
    ]
  },
  "I1": {
    "v_hit": 2.7,
    "class": "DISTURBED_CAPTURED",
    "t_impact": 2.8059999999999126,
    "J_mag": 0.14850726078961748,
    "max_ex_mm": 1.1132928455541284,
    "t_drop": null,
    "hold_survived": true,
    "v_ball_h_pre": [
      2.700036948612737,
      -0.005819207303606058,
      0.2823982326980527
    ]
  },
  "I2": null,
  "I3": {
    "v_hit": 2.967,
    "class": "IMMEDIATE_LOSS",
    "t_impact": 2.7999999999999132,
    "J_mag": 0.1560152863369837,
    "max_ex_mm": 1.3602858213416602,
    "t_drop": 2.8059999999999126,
    "hold_survived": false,
    "v_ball_h_pre": [
      2.9670336353794964,
      -0.005090195608302727,
      0.2475773865514603
    ]
  }
}
```

## 11. Delayed-failure trajectories

No DELAYED_FAILURE trajectory exists in this family. Boundary refinement
between last DISTURBED_CAPTURED (`v=2.95` and locally `3.000`/`3.017`) and
IMMEDIATE_LOSS (`v=2.967`, `2.983`, `3.033`, `3.05`, …) shows an **abrupt**
capture→loss cliff (~0.1 m/s) that is **locally non-monotonic**. That is a
physical finding of this mass/direction/lift timing, not a missing plot.

## 12. Repeatability

```json
{
  "v_hit": 2.7,
  "class_a": "DISTURBED_CAPTURED",
  "class_b": "DISTURBED_CAPTURED",
  "max_obj_pos_diff": 0.0,
  "note": "identical IC, deterministic replay; not a statistical N-trial claim"
}
```

Deterministic identical-IC replays are not statistical robustness.

## 13. Viewer / video commands

```text
python training/demo_ballistic_impact.py
python training/demo_ballistic_impact.py --severity mild
python training/demo_ballistic_impact.py --severity disturbed
python training/demo_ballistic_impact.py --severity delayed_failure
python training/demo_ballistic_impact.py --severity loss
python training/demo_ballistic_impact.py --render-video
```

## 14. USER VISUAL OBSERVATION: pending

## 15. Unresolved issues

- No I2/DELAYED_FAILURE representative under m_ball=0.05 kg, +hand-x free-flight,
  and the in-lift trigger used here. The captured-to-loss transition is sharp
  and locally non-monotonic around v≈2.97–3.03 m/s. This was not “fixed” by
  changing s, τ, or the gravitational heuristic (those are out of scope).
- I3 video uses the first IMMEDIATE_LOSS on the refined grid (`v=2.967`).
  Unambiguous knockout also exists at `v=3.20` and `v=4.00` (`raw/ep_v4.00.json`).
- Gravity-compensated launch still adds a vertical `v_h,z` at lower speeds
  (I0: v_h ≈ [1.20, -0.01, 0.62]); lateral `v_h,x` remains the largest component
  for I0–I3 in this sweep.
- Old 0.5 kg / 8–10 m/s impact claims remain quarantined; do not mix those
  numbers with this family.