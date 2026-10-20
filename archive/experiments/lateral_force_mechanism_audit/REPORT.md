# Lateral-Force Mechanism Audit

**Scope:** raw replay of 5 stabilization SUCCESS_NEW + 5 pose-correction SUCCESS_NEW (+ detach/hard failure diagnostics). No new expert search.

**Thresholds:** sustain bilateral/uni >=0.04s; full-detach none >=0.02s and separation evidence; slide >=2.0 mm hand-frame; pivot >=3.0 deg axis change; rigid if mean |v_rel|<0.05 and omega alignment high while bilateral.

## Summary table (SUCCESS_NEW)

| IC | Family | Primary | slide mm | d_axis_deg | d_pos | d_th | detach | recatch |
|----|--------|---------|---------:|-----------:|------:|-----:|-------:|--------:|
| lforce_0025__SLIDING_BILATERAL__0 | stabilization_only | PASSIVE_SETTLING | 0.019 | 0.046 | -0.016 | 0.003 | 0 | 0 |
| lforce_0029__SLIDING_BILATERAL__0 | stabilization_only | PASSIVE_SETTLING | 0.017 | 0.03 | -0.013 | 0.002 | 0 | 0 |
| lforce_0032__SLIDING_BILATERAL__0 | stabilization_only | PASSIVE_SETTLING | 0.061 | 0.114 | -0.055 | 0.004 | 0 | 0 |
| lforce_0035__SLIDING_BILATERAL__0 | stabilization_only | PASSIVE_SETTLING | 0.06 | 0.181 | -0.058 | 0.007 | 0 | 0 |
| lforce_0064__SLIDING_BILATERAL__0 | stabilization_only | PASSIVE_SETTLING | 0.007 | 0.017 | -0.004 | 0.001 | 0 | 0 |
| lforce_0000__SLIDING_BILATERAL__0 | pose_correction | CONTACT_PRESERVING_PIVOTING | 10.15 | 10.804 | 3.413 | 10.803 | 0 | 0 |
| lforce_0001__SLIDING_BILATERAL__0 | pose_correction | CONTACT_PRESERVING_SLIDING | 7.277 | 0.94 | 2.572 | 0.937 | 0 | 0 |
| lforce_0009__SLIDING_BILATERAL__0 | pose_correction | CONTACT_PRESERVING_SLIDING | 7.26 | 0.619 | 2.543 | 0.61 | 0 | 0 |
| lforce_0026__SLIDING_BILATERAL__0 | pose_correction | CONTACT_PRESERVING_SLIDING | 7.32 | 2.235 | 2.94 | 2.235 | 0 | 0 |
| lforce_0036__SLIDING_BILATERAL__0 | pose_correction | CONTACT_PRESERVING_PIVOTING | 7.479 | 10.356 | 3.439 | 10.354 | 0 | 0 |

## Required conclusions (Q1-Q10)

1. **Did any successful lateral-force trajectory fully detach?** **No** (0/10 SUCCESS).
2. **Did any successful trajectory genuinely recatch?** **No** (0/10). A valid recatch requires verified full-detachment then bilateral reacquire. **No recatch video was fabricated.**
3. **Were the original five merely passive settling?** **Yes**. Token `[0]` only; hand-relative errors already inside SUCCESS_NEW tolerance; primary = PASSIVE_SETTLING for all five.
4. **New five: genuine hand-relative position correction?** **5/5**: lforce_0000__SLIDING_BILATERAL__0, lforce_0001__SLIDING_BILATERAL__0, lforce_0009__SLIDING_BILATERAL__0, lforce_0026__SLIDING_BILATERAL__0, lforce_0036__SLIDING_BILATERAL__0.
5. **Genuine hand-relative orientation correction?** **3/5**: lforce_0000__SLIDING_BILATERAL__0, lforce_0026__SLIDING_BILATERAL__0, lforce_0036__SLIDING_BILATERAL__0.
6. **Was observed robot-cylinder rotation mostly rigid co-motion?** **Often yes during Return/wrist segments** (high omega alignment, low |v_rel| while bilateral). World-frame co-rotation is not in-grasp pivoting. Pose-correction experts still show measurable hand-frame slide/pivot on top of co-motion.
7. **Which demonstrate sliding or pivoting?** Sliding (5): lforce_0000__SLIDING_BILATERAL__0, lforce_0001__SLIDING_BILATERAL__0, lforce_0009__SLIDING_BILATERAL__0, lforce_0026__SLIDING_BILATERAL__0, lforce_0036__SLIDING_BILATERAL__0. Pivoting (2): lforce_0000__SLIDING_BILATERAL__0, lforce_0036__SLIDING_BILATERAL__0.
8. **Suitable contact-preserving recovery demos?** 10 ICs: lforce_0025__SLIDING_BILATERAL__0, lforce_0029__SLIDING_BILATERAL__0, lforce_0032__SLIDING_BILATERAL__0, lforce_0035__SLIDING_BILATERAL__0, lforce_0064__SLIDING_BILATERAL__0, lforce_0000__SLIDING_BILATERAL__0, lforce_0001__SLIDING_BILATERAL__0, lforce_0009__SLIDING_BILATERAL__0, lforce_0026__SLIDING_BILATERAL__0, lforce_0036__SLIDING_BILATERAL__0.
9. **Detached-object recatch demos?** **0** among SUCCESS. Failure detach-without-recatch: lforce_0047__FULL_DETACHMENT__0, lforce_0040__SLIDING_BILATERAL__1. See DETACHMENT_EVENTS.csv.
10. **Missing from dataset?** Genuine FULL_DETACHMENT_RECATCH SUCCESS_NEW; long airborne chase with active pursuit; verified unilateral-escape then re-grasp cohort.

## Failure diagnostics

Recovery-onset for `lforce_0047__FULL_DETACHMENT__0` is already hand-free (sep_mid ~82 mm), confirming full detachment occurred in the force phase before the saved recovery snapshot; Recovery does not reacquire. `lforce_0040__SLIDING_BILATERAL__1` shows contact-preserving relative motion then sustained no-hand contact ending on the table (FULL_DETACHMENT_NO_RECATCH).

## User observation

Cylinder and robot appearing to rotate together is consistent with **RIGID_CO_MOTION / wrist Return** under sustained bilateral contact. That is **not** evidence of escape+recatch. Hand-frame traces separate true relative sliding/pivoting from shared world motion.

## Artifacts

CONTACT_TIMELINES.csv, RELATIVE_POSE_TRACES.csv, DETACHMENT_EVENTS.csv, RECAPTURE_EVENTS.csv, POSE_CORRECTION_MECHANISMS.csv, EXPERT_RECLASSIFICATION.csv, plots/, videos/, traces/, raw/THRESHOLDS.json.

Script: `scripts/run_lateral_force_mechanism_audit.py`
