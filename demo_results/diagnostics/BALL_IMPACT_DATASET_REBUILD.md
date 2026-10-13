# Ball-impact dataset rebuild

Generation and validation only. No GRPO, BC, or SAC update was run.

The completed 120-update GRPO run is sealed at `demo_results/diagnostics/raw/formal_grpo_pilot`.
New initial conditions are only under `demo_results/diagnostics/raw/ball_impact_dataset`.

## Physics

```
{
  "scene": "assets/scene_ballistic_impact.xml",
  "timestep_s": 0.002,
  "gravity": [
    0.0,
    0.0,
    -9.81
  ],
  "noslip_iterations": 1,
  "noslip_tolerance": 1e-06,
  "solver_iterations": 100,
  "integrator": 3,
  "cylinder_mass_kg": 0.2,
  "cylinder_radius_m": 0.018,
  "cylinder_half_height_m": 0.03,
  "cylinder_friction": [
    1.0,
    0.005,
    0.0001
  ],
  "cylinder_condim": 4,
  "cylinder_contype": 3,
  "cylinder_conaffinity": 3,
  "ball_mass_kg": 0.05,
  "ball_radius_m": 0.015,
  "ball_friction": [
    0.4,
    0.005,
    0.0001
  ],
  "ball_condim": 3,
  "ball_contype": 2,
  "ball_conaffinity": 2,
  "finger_object_mu": 1.0,
  "xml_cylinder_mass_kg": 0.2,
  "xml_ball_mass_kg": 0.05,
  "xml_cylinder_radius_m": 0.018,
  "xml_ball_radius_m": 0.015,
  "source_constants_match_model": true,
  "mass_friction_randomized": false,
  "reason": "nominal.yaml randomize is false and BallisticSim.reset keeps the XML cylinder mass and PAIR_MU"
}
```

Cylinder and ball masses, radii, friction, condim, and the noslip solver are the current ballistic scene.
Nominal mass and friction were not randomized: `nominal.yaml` has `randomize: false`, and `BallisticSim.reset` keeps the XML mass and finger-object mu.
The ball free-joint state is written only at launch, before contact. The cylinder pose, hand pose, and post-impact velocity are not edited.

## Rejection funnel

Attempt log: `demo_results/diagnostics/raw/ball_impact_dataset/attempts.jsonl` (449 rows).

```
{
  "impact_without_recovery_entry": 351,
  "valid_impact_recovery_ic": 98
}
```

## Accepted counts

Targets are train 300, validation 60, final test 120.

```
{
  "accepted": {
    "train": 28,
    "validation": 26,
    "final_test": 44
  },
  "by_direction": {
    "train": {
      "+X": 9,
      "oblique_+X_+Y": 9,
      "+Y": 10
    },
    "validation": {
      "oblique_-X_+Y": 22,
      "-X": 4
    },
    "final_test": {
      "oblique_-X_-Y": 28,
      "-Y": 8,
      "oblique_+X_-Y": 8
    }
  }
}
```

Splits were assigned by launch azimuth before any episode:

```
{
  "train": [
    0.0,
    45.0,
    90.0
  ],
  "validation": [
    135.0,
    180.0
  ],
  "final_test": [
    225.0,
    270.0,
    315.0
  ]
}
```

Azimuth overlap across splits: False.

## Impact parameters of accepted episodes

```
{
  "speed_mps": {
    "6.5": 25,
    "6.7": 13,
    "6.3": 16,
    "8.5": 6,
    "9.5": 20,
    "7.5": 9,
    "7.3": 6,
    "7.7": 2,
    "8.7": 1
  },
  "z_off_m": {
    "-0.015": 44,
    "-0.013": 11,
    "-0.017": 14,
    "0.0": 5,
    "0.002": 5,
    "0.015": 11,
    "-0.002": 3,
    "0.013": 5
  },
  "phi_deg": {
    "-40.0": 20,
    "-35.0": 7,
    "0.0": 24,
    "-5.0": 15,
    "5.0": 9,
    "40.0": 16,
    "35.0": 7
  },
  "delay_s": {
    "0.0": 27,
    "0.2": 22,
    "0.4": 49
  }
}
```

## Contact verification

An accepted episode has a `ball-cylinder` contact with normal force, a contact timestamp after launch, and geometry entry after that contact.
Ball-finger contact is recorded separately and is not accepted as a substitute for cylinder contact.
Accepted episodes with the ball still touching the cylinder at entry: 0.

## Post-impact physical states

Labels overlap. A sample can carry several. Rolling is labeled only when the object angular rate is at least 1 rad/s, the finger contact point migrates by at least 3 mm in the object frame, and the tangential speed is consistent with radius times angular rate. Contact migration with a small angular rate is labeled separately.

```
{
  "accepted_episodes": 98,
  "state_counts": {
    "tangential_slip": 13,
    "relative_rolling": 50,
    "asymmetric_or_unilateral": 91,
    "complete_grasp_loss": 6,
    "airborne_object": 1,
    "intermittent_contact": 0,
    "recaught_displaced": 0,
    "delayed_grasp_failure": 0,
    "stable_bilateral": 0,
    "contact_migration": 61
  },
  "by_split": {
    "train": {
      "asymmetric_or_unilateral": 25,
      "relative_rolling": 9,
      "contact_migration": 15,
      "complete_grasp_loss": 2,
      "tangential_slip": 3,
      "airborne_object": 1
    },
    "validation": {
      "asymmetric_or_unilateral": 26,
      "contact_migration": 16,
      "relative_rolling": 19,
      "tangential_slip": 3
    },
    "final_test": {
      "asymmetric_or_unilateral": 40,
      "contact_migration": 30,
      "relative_rolling": 22,
      "tangential_slip": 7,
      "complete_grasp_loss": 4
    }
  }
}
```

Acceptance is not conditioned on these labels. Counts above are how often each state occurred on accepted impact entries.
Under-represented states used only to retarget pre-impact launch parameters: ['complete_grasp_loss', 'airborne_object', 'intermittent_contact', 'recaught_displaced', 'delayed_grasp_failure'].

Accepted-entry coverage, as a fraction of accepted episodes:

- unilateral or asymmetric finger contact: 91 / 98
- rolling signature (angular rate, contact migration, and rolling-speed match): 50 / 98
- contact migration while angular rate stayed below 1 rad/s: 61 / 98
- tangential slip (slip speed, and not explained by rolling): 13 / 98
- complete grasp loss before entry: 6 / 98
- airborne separation before entry: 1 / 98
- intermittent contact, recaught-displaced, delayed failure, stable bilateral before entry: 0, 0, 0, 0

Stable bilateral contact before entry is rare because an already-nominal grasp is rejected as trivial.
Delayed failure and recatch usually need longer than the short interval from impact to the 6 mm entry, so they show up in the later rollout trace rather than in the entry label.
Low strikes (`z_off` -15 mm) are the largest height bin so far. Later sampling prefers heights and tangential offsets that are still thin.

## Geometry entry and snapshot fidelity

Entry is the existing position threshold, 6 mm for 40 ms, and only after the ball has been launched.
Restore-and-continue qpos error over 10 nominal steps, max 0.000869947403755276, tolerance 0.001.
The capture stores MuJoCo state, time, controller desired pose and commands, geometry-switch reference, and the in-flight ball. Controller targets are not rewritten.

## Recoverability audit

Train and validation only. Final test was not rolled.
Zero means the nominal hold continues from the entry snapshot. The frozen K=36 initialization is rolled with the nominal-basin, explicit handoff, and 10 s nominal-only definition.
A finite-search failure is not treated as proof that the state is unreachable.
Full rollout records: `demo_results/diagnostics/raw/ball_impact_dataset/recoverability_audit.json`.

## Recovery mechanisms

The legal interface is unchanged: 7-D `recovery7d_airborne_v3`, K=36 tokens, residual cap ±0.25.
Previously verified behaviors, on the older pilot initial conditions rather than this dataset, are token 34 open-finger release for slip, token 35 airborne chase, and token 32 lateral chase on `impact_zp6`.
Object rotation alone is not counted as controlled rolling. Slip velocity alone is not counted as controlled sliding. Recatch is not counted as successful unless the rollout reaches the nominal basin, hands off, and holds for 10 s under the nominal controller.

On this audit, a mechanism is reported only as concurrent evidence:

- closed-grip lateral command together with measured tangential slip
- wrist command together with the rolling signature above
- a grip-close command while both finger contacts are present
- an open command followed by bilateral contact
- loss of both contacts followed by bilateral contact

Those coincidences do not by themselves identify which command produced the motion.

Frozen K=36 successes on this subset: 1 / 10. Nominal continuation already inside the basin: 0 / 10.
Successes in this subset used a brief open and then closed-lift tokens. That is open-then-recontact followed by a nominal handoff, not evidence that sliding or rolling was controlled.
Wrist motion during a rolling signature, and open-then-recontact after grasp loss, occurred on unsuccessful rollouts as well. Those episodes show the physical events without a 10 s nominal survival.

- `ball_train_0001` (+X): entry ['asymmetric_or_unilateral', 'relative_rolling']; zero basin False; K=36 DROP_DURING_RECOVERY; concurrent ['open then bilateral recontact', 'loss then bilateral recontact', 'unilateral entry later showed bilateral contact'].
- `ball_train_0011` (+Y): entry ['contact_migration']; zero basin False; K=36 BASIN_NOT_REACHED; concurrent ['open then bilateral recontact', 'loss then bilateral recontact'].
- `ball_train_0010` (oblique_+X_+Y): entry ['asymmetric_or_unilateral', 'contact_migration', 'relative_rolling']; zero basin False; K=36 BASIN_NOT_REACHED; concurrent ['wrist command during rolling signature', 'open then bilateral recontact', 'loss then bilateral recontact'].
- `ball_train_0015` (oblique_+X_+Y): entry ['asymmetric_or_unilateral', 'contact_migration', 'tangential_slip']; zero basin False; K=36 BASIN_NOT_REACHED; concurrent ['none isolated'].
- `ball_train_0003` (+X): entry ['asymmetric_or_unilateral', 'relative_rolling']; zero basin False; K=36 SUCCESS_12S; concurrent ['open then bilateral recontact', 'loss then bilateral recontact', 'unilateral entry later showed bilateral contact'].
- `ball_train_0002` (+X): entry ['asymmetric_or_unilateral']; zero basin False; K=36 BASIN_NOT_REACHED; concurrent ['open then bilateral recontact', 'loss then bilateral recontact'].
- `ball_validation_0023` (-X): entry ['asymmetric_or_unilateral', 'relative_rolling']; zero basin False; K=36 DROP_DURING_RECOVERY; concurrent ['open then bilateral recontact', 'loss then bilateral recontact'].
- `ball_validation_0001` (oblique_-X_+Y): entry ['asymmetric_or_unilateral', 'contact_migration', 'relative_rolling']; zero basin False; K=36 DROP_DURING_RECOVERY; concurrent ['none isolated'].
- `ball_validation_0020` (oblique_-X_+Y): entry ['asymmetric_or_unilateral', 'contact_migration', 'tangential_slip']; zero basin False; K=36 DROP_DURING_RECOVERY; concurrent ['none isolated'].
- `ball_validation_0002` (oblique_-X_+Y): entry ['asymmetric_or_unilateral', 'contact_migration', 'relative_rolling']; zero basin False; K=36 DROP_DURING_RECOVERY; concurrent ['none isolated'].

## Loader

Formal GRPO reads this manifest when `dataset: ball_impact`.
Train ids and validation ids come from the manifest splits. Final-test ids are not sampled.
The observation, action tokens, residual cap, geometry entry, reward, success rule, actor, and KL coefficient are unchanged.
Impact episodes keep the existing impact-family KL. The pilot catalog and `demo_results/diagnostics/raw/formal_grpo_pilot` are not overwritten.

Future training command, not executed here:

```
python -u training/formal_grpo.py --config config/formal_grpo_ballistic.yaml
```

Resume generation:

```
python -u training/ball_impact_dataset.py --attempts 200
```

Accepted so far 98 of 449 logged launches. Remaining to target 382.
Latest generation batch: 2.82 s per launch.
Rough remaining time at the current acceptance rate: 82.3 minutes.

BALL-IMPACT DATASET INCOMPLETE

TRAINING HAS NOT BEEN STARTED.
