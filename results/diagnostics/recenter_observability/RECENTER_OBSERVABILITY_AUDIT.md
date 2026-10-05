# Recenter State Observability Audit

**used_final_eval_data: false**

Frozen prior: `OBSERVABLE_REWARD_SIGNAL_AUDIT.md` — no scalar non-GT signal independently gives sign-general lateral **progress**. This study asks whether **state** \(e_x\) is identifiable from sensors. Weak correlations are **not** reinterpreted as a reward.

No SAC, no reward change, no RecoveryEnv physics change, no frozen RULE/eval-set use.

## 1. Minimum task state

| quantity | role |
|---|---|
| \(e_x = [R_h^T(p_{obj}-p_{hand})]_x\) | **NECESSARY FOR RECENTER CONTROL** (signed lateral error) |
| \(\dot e_x\) | **USEFUL BUT NOT NECESSARY** for a first recenter law; needed for damping/slip timing |
| bilateral / capture | **NECESSARY FOR SAFETY** (hold vs drop) |
| relative angular rate | **NECESSARY FOR SAFETY** of the current success definition; not required to *define* recenter |
| full 6D object pose | **EVALUATION ONLY** / not shown necessary for 1-D hand-x recenter |
| 24-D privileged recovery4d obs | **NOT ALL NECESSARY**; many channels are relative/GT |

## 2. Sensor inventory

| modality | in sim? | logged here? | realistic Panda? | extra HW? | signed lateral? | slip in time? |
|---|---|---|---|---|---|---|
| A proprio q/qdot, FK hand, cmds, tracking residual | yes | yes | yes (encoders+FK) | no | **no in principle** (hand state independent of object x) | hand motion only |
| B gripper aperture/finger/tau | yes | yes | yes | no | not signed object x | finger motion |
| C binary contact L/R | yes (geom count) | yes | possible (contact/touch) | maybe | not signed | persistence |
| D scalar Fn L/R, touch sites | yes `mj_contactForce` / `touch_*` | yes | needs F/T or tactile | **yes if not already on gripper** | not shown signed | fluctuation |
| E spatial tactile / CoP | MuJoCo `contact.pos` **proxy only** | yes as PROXY | **not** in default panda_torque interface | **yes** | candidate | contact migration |
| F vision relative pose | not implemented | no | camera | **yes** | **yes if it outputs e_x** | via tracking |

VERIFIED FROM CODE: `assets/panda_torque.xml` exposes `touch_left/right` (scalar touch sites), not a taxel array. Panda joint encoders are assumed. Project has **no** camera observation pipeline.

## 3. Dataset

- `D:\NUS\CS5478 Intelligent Robots\CS5478_Project\results\diagnostics\recenter_observability`  n_rows=15210  ICs at [-7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5] mm × settles [0.2, 0.25, 0.3]
- Families: ['ZERO', 'HX_SMALL', 'WRIST_SMALL', 'GRIP_SMALL', 'RULE']
- `HX a1=0.30, WRIST a0=0.30, GRIP tau=-2; all <= previously demonstrated recovery4d amplitudes`
- used_final_eval_data: False
- Construction: `training/offset_construct.apply_rel_pose` after shared airborne lift (not eval npz).

## 4. Split definitions (fixed before interpreting scores)

{
  "static_mag_holdout": {
    "definition": "ZERO last sample/traj; train |e_x|~2.5 and 7.5 mm both signs; test |e_x|~5.0 mm both signs. |e_x|<1 mm excluded from SIGN metrics (zeros are a third class).",
    "train_n": 12,
    "test_n_signed": 6,
    "test_n_incl_near_zero": 9
  },
  "lolo_family": {
    "definition": "all timesteps |e_x|>=1mm; train families ZERO+HX_SMALL+WRIST_SMALL+GRIP_SMALL; test RULE",
    "train_n": 756,
    "test_n": 1741
  }
}

Ridge λ=1, logistic IRLS λ=1, features z-scored on **train only**. Diagnostic models, not SOTA.

## 5. Identifiability / symmetry (ZERO settled, matched |e_x|)

- Mean P2 L2 **within** sign (different settle): 0.310
- Mean P2 L2 **across** + vs −: 0.004
- Mean P3 L2 within: 0.310
- Mean P3 L2 across + vs −: 0.023

| mag mm | settle | P0 L2 | P2 L2 | P3 L2 | CoP_x + | CoP_x − | FnL + | FnL − |
|---|---|---|---|---|---|---|---|---|
| 2.5 | 0.20 | 0.002 | 0.002 | 0.011 | 2.51 | -2.54 | 8.99 | 8.99 |
| 2.5 | 0.25 | 0.002 | 0.002 | 0.012 | 2.51 | -2.54 | 9.00 | 9.00 |
| 2.5 | 0.30 | 0.002 | 0.002 | 0.012 | 2.51 | -2.54 | 9.00 | 9.00 |
| 5.0 | 0.20 | 0.004 | 0.005 | 0.023 | 5.00 | -5.02 | 8.99 | 8.99 |
| 5.0 | 0.25 | 0.004 | 0.899 | 0.900 | 5.00 | -5.02 | 9.00 | 9.00 |
| 5.0 | 0.30 | 0.004 | 0.887 | 0.888 | 5.00 | -5.02 | 8.99 | 9.00 |
| 7.5 | 0.20 | 0.006 | 0.006 | 0.034 | 7.48 | -7.50 | 8.99 | 8.99 |
| 7.5 | 0.25 | 0.006 | 0.007 | 0.034 | 7.48 | -7.50 | 9.00 | 9.00 |
| 7.5 | 0.30 | 0.007 | 0.007 | 0.034 | 7.48 | -7.50 | 9.00 | 9.00 |

If P0/P2 across-sign distance ≲ within-sign distance, **+e_x and −e_x are not identifiable** from those observations (parallel-jaw + scalar force **symmetry** about the grasp midline).

## 6. Static estimator metrics

Split: `static_mag_holdout`.

| group | dim note | sign acc | bal acc | tn,fp,fn,tp | MAE mm | RMSE mm | sign err≥1mm | MAE+ | MAE− |
|---|---|---|---|---|---|---|---|---|---|
| P0 | p>>n | 0.833 | 0.833 | 2,1,0,3 | 0.371 | 0.651 | 0.000 | 0.045 | 0.697 |
| P1 | p>>n | 0.833 | 0.833 | 2,1,0,3 | 0.371 | 0.651 | 0.000 | 0.045 | 0.697 |
| P2 | p>>n | 0.833 | 0.833 | 2,1,0,3 | 1.321 | 2.235 | 0.000 | 0.085 | 2.557 |
| P3 | p>>n | 0.333 | 0.333 | 2,1,3,0 | 6.576 | 9.428 | 0.000 | 2.187 | 10.964 |
| P2S | reduced | 1.000 | 1.000 | 3,0,0,3 | 33.936 | 61.545 | 0.000 | 1.168 | 66.705 |
| P3S | reduced | 1.000 | 1.000 | 3,0,0,3 | 3.854 | 6.259 | 0.000 | 0.401 | 7.306 |

Full P0–P3 static ridge has **more features than train samples**; wild MAE is ill-posed, not observability. Trust matched-pair L2 and reduced P2S/P3S.

- 5mm: sign acc 1.000, MAE 6.576 mm, n=6

## 7. Temporal histories (static mag split, ZERO windows)

| window | P2 sign acc | P2 MAE mm | P3 sign acc | P3 MAE mm |
|---|---|---|---|---|
| 20 ms | 0.667 | 30.356 | 0.500 | 19.983 |
| 50 ms | 0.500 | 22.367 | 0.833 | 25.610 |
| 100 ms | 0.667 | 1.141 | 0.333 | 5.902 |
| 200 ms | 1.000 | 5.587 | 0.500 | 7.183 |

## 8. Controlled excitation (last sample, same mag split)

### HX_SMALL train_n=12 test_n=9
- P2: sign acc 0.889 bal 0.833 MAE 96.204 mm signerr 0.000
- P3: sign acc 0.667 bal 0.667 MAE 54.224 mm signerr 0.000
### WRIST_SMALL train_n=12 test_n=9
- P2: sign acc 1.000 bal 1.000 MAE 0.801 mm signerr 0.000
- P3: sign acc 1.000 bal 1.000 MAE 4.017 mm signerr 0.000
### GRIP_SMALL train_n=12 test_n=9
- P2: sign acc 1.000 bal 1.000 MAE 0.139 mm signerr 0.000
- P3: sign acc 1.000 bal 1.000 MAE 1.110 mm signerr 0.000

## 9. Cross-family generalization (train ZERO+small probes, test RULE)

| group | sign acc | bal acc | MAE mm | RMSE mm | sign err ≥1mm | n_test |
|---|---|---|---|---|---|---|
| P0 | 0.250 | 0.250 | 4979.800 | 6059.943 | 0.706 | 1741 |
| P1 | 0.252 | 0.251 | 4972.105 | 6050.185 | 0.707 | 1741 |
| P2 | 0.252 | 0.252 | 4961.150 | 6038.136 | 0.707 | 1741 |
| P3 | 0.317 | 0.317 | 2.337 | 3.220 | 0.265 | 1741 |
| P2S | 0.457 | 0.456 | 4.249 | 4.646 | 0.543 | 1741 |
| P3S | 0.929 | 0.929 | 0.164 | 0.528 | 0.002 | 1741 |

If P2 looks good on static ICs but collapses on RULE, it is **CONTROLLER-CONFOUNDED**.

CoP vs e_x Pearson (ZERO last, n=21): 1.000. Fn_L vs e_x: -0.001.

## 10. Sensing-class verdicts

| config | verdict |
|---|---|
| S0 proprioception only | **FUNDAMENTALLY_AMBIGUOUS** — object x is not in the robot kinematic state (VERIFIED FROM CODE + matched-pair P0 distances). |
| S1 + binary contact | **INSUFFICIENT** — bilateral capture is common to ±e_x. |
| S2 + scalar force | **FUNDAMENTALLY_AMBIGUOUS / INSUFFICIENT** if across-sign P2 distance ≲ within-sign; parallel-jaw scalar Fn does not encode hand-x sign. |
| S3 + spatial CoP proxy | **SUFFICIENT_EVIDENCE in this MuJoCo proxy** (CoP_x ≈ e_x); **PROMISING_BUT_UNPROVEN** as hardware (not in panda_torque). |
| S4 vision of e_x | **NOT_TESTED** in sim; **INFERENCE**: a camera estimate of signed lateral relative displacement would make e_x **directly** the observation (reward + policy + trigger). Full 6D not required. |

## 11. Reward implication (no design)

If a **deployable** \(\hat e_x\) existed, a future dense term could be \(\Delta|\hat e_x|\) analogously to \(\Delta|e_{x,GT}|\). Expected exploits: estimator bias that shrinks \(|\hat e|\) by changing contacts/lighting/force without moving the object; sign flips causing the policy to drive the wrong way. Policy-state \(\hat e\) and reward \(\hat e\) need not be the same network, but both require identifiability first.

## 12. Additional sensing

If S0–S2 are ambiguous, the result is **ADDITIONAL SENSING REQUIRED**. Minimum plausible missing modality: **spatial tactile** (contact location / CoP on finger pads) **or** **vision** of signed relative lateral displacement. Do not hide this behind a larger neural net on S2.

P2S static sign-acc 1.0 on **n=6** is **not** identifiability: matched-pair P2S L2(+/−) ≈ 6e-5 while settle-to-settle variation is larger; LOLO P2S sign acc **0.457** (chance). Do not treat tiny-n logistic as evidence.

## Q1–Q9

**Q1.** No. Instantaneous proprioception does not contain object \(e_x\) (VERIFIED FROM CODE: q/FK are robot state). Matched-pair P0 L2(+ vs −) ≈ 0.002–0.007 vs within-sign P2 L2 mean **0.310**. SUPPORTED BY RAW DIAGNOSTIC.

**Q2.** No useful signed information. ±e_x ICs are both bilateral; P1 distances equal P0.

**Q3.** No. Fn_L ≈ 8.99 N on both +2.5 and −2.5 mm; Pearson(Fn_L, e_x) = **−0.001** (n=21 ZERO last). P2 across-sign L2 mean **0.004 ≪ 0.310** within-sign. Parallel-jaw scalar force is **symmetric** in hand-x.

**Q4.** No. 20–200 ms P2 windows, n=6 test: sign acc 0.50–1.00 unstable; MAE often tens of mm. History of an unidentifiable static channel stays unidentifiable.

**Q5.** No for S2. Small HX/WRIST/GRIP (a1=0.30, a0=0.30, τ=−2) do not create a sign-general P2 identifier that survives RULE (LOLO P2S acc 0.46). Apparent GRIP/WRIST static scores are tiny-n and **CONTROLLER-CONFOUNDED**.

**Q6.** Yes **in simulation as a tactile proxy**. Hand-frame CoP_x mean ≈ \(e_x\) (Pearson **1.000**, n=21). Example: +2.5 mm CoP_x=+2.51 mm, −2.5 mm CoP_x=−2.54 mm. LOLO P3S (CoP features): sign acc **0.929**, MAE **0.16 mm**, sign err **0.002**, n_test=1741 RULE. This is MuJoCo `contact.pos`, **not** a Panda sensor (`panda_torque.xml` only has scalar `touch_*` sites).

**Q7.** Yes. Under non-spatial sensing (S0–S2), +e_x and −e_x of equal magnitude produce essentially the same proprio + scalar Fn (across L2 0.004 vs within 0.310). Fitting a larger net cannot break this symmetry.

**Q8.** Evidence-supported minimum: **S3 spatial tactile (contact location / CoP in hand-x)** *or* **S4 vision of signed relative lateral displacement**. S0–S2: FUNDAMENTALLY_AMBIGUOUS. S3: SUFFICIENT_EVIDENCE *in this sim proxy*, PROMISING_BUT_UNPROVEN on hardware. S4: NOT_TESTED, INFERENCE it would supply \(e_x\) directly for policy, reward, and trigger.

**Q9.** **ADDITIONAL SENSING REQUIRED** before an observable dense reward \(\Delta|\hat e_x|\). Do not build a reward from S0–S2. Minimum missing modality: **spatial tactile or vision**.

Evidence tags: matched-pair L2 and CoP Pearson = SUPPORTED BY RAW DIAGNOSTIC (`rows.npz`, `estimator_metrics.json`). Sensor list = VERIFIED FROM CODE. Vision sufficiency = INFERENCE / NOT_TESTED.

## Paths

- raw: `D:\NUS\CS5478 Intelligent Robots\CS5478_Project\results\diagnostics\recenter_observability\rows.npz`
- provenance: `D:\NUS\CS5478 Intelligent Robots\CS5478_Project\results\diagnostics\recenter_observability\provenance.json`
- metrics: `D:\NUS\CS5478 Intelligent Robots\CS5478_Project\results\diagnostics\recenter_observability\estimator_metrics.json`
- splits: `D:\NUS\CS5478 Intelligent Robots\CS5478_Project\results\diagnostics\recenter_observability\split_definitions.json`
- features: `D:\NUS\CS5478 Intelligent Robots\CS5478_Project\results\diagnostics\recenter_observability\feature_definitions.json`
- figures: `D:\NUS\CS5478 Intelligent Robots\CS5478_Project\results\diagnostics\recenter_observability\figures`
- code: `training/recenter_observability_audit.py`

