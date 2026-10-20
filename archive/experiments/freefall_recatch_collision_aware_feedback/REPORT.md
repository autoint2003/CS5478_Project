# Collision-Aware Freefall Recatch Feedback

**Script:** `scripts/run_freefall_recatch_collision_aware_feedback.py`  
**Prior diagnostic:** `results/diagnostics/freefall_recatch_pose_feedback/`  
**Output:** `results/diagnostics/freefall_recatch_collision_aware_feedback/`  

No SFT/GRPO. No robot/cylinder geometry, contact-parameter, nominal-control, or SUCCESS_NEW criterion changes. Legal continuous 7D Recovery only.

## Diagnoses (Stages 1–4)

### Controller O horizontal failure

| Check | Result |
|-------|--------|
| 39D `p_rel` frame | hand frame (`Rh.T @ (po − ph)`) |
| Action `a[0:3]` frame | hand frame (`v_world = Rh @ v_hand`) |
| Unit-axis tracking | `+a[0] → +Δhand_x` (ACTION_RESPONSE_AUDIT.csv) |
| Original chase law | `a ∝ P_TGT − p_rel` → **sign-reversed** for freefall chase |
| Sign probe | `+p_rel_x` → orig `a0=−1`, corrected `a0=+1` |

**Verdict:** O generated the wrong horizontal command; the robot executed it. Not a clipping or tracking failure.

### Controller P collision geometry

| Field | Value |
|-------|-------|
| First contact body | `hand` |
| First contact geom | `geom_66` (hand-body mesh) |
| Class | **hand housing** (not fingertip) |
| Mechanism | descend while XY still open → housing sweeps cylinder |

See `HAND_COLLISION_AUDIT.csv`, `FIRST_CONTACT_GEOMETRY.csv`.

## Corrected shared controller

Single frozen law (`raw/shared_gains.json`) for diagnostic P (privileged) and O (39D + history + fixed `MID_OFFSET_H`):

1. Chase **finger mid** (not hand origin / COM-only).
2. Brief retract while aligning XY (avoid housing).
3. Scoop descend; soften near capture.
4. Capture-time wrist from relative orientation (+ small wy/wz prior from distal SN evidence).
5. XY-only free-fall lead for chase; phase uses current height.

## First-contact comparison (`perp_off01_r15`)

| controller | href8 | housing | jaw | bil | SN |
|------------|------:|--------:|----:|----:|---:|
| orig_P | +0.003 | **1** | 0 | — | 0 |
| orig_O | **−0.068** | 0 | 0 | — | 0 |
| corr_O (no pred) | +0.028 | 1 | 0 | — | 0 |
| corr_O | +0.026 | 0 | 0 | — | 0 |
| corr_P (no pred) | +0.039 | 0 | **1** | **6** | 0 |
| corr_P | +0.039 | 0 | **1** | **6** | 0 |

Prediction vs no-pred is nearly identical on this IC (XY lead helps little once retract/scoop phasing is fixed).

## Held-out (12 constructed freefall ICs, frozen gains)

| controller | SN | first_bil | jaw_entry | housing |
|------------|---:|----------:|----------:|--------:|
| open_loop (`perp_off01_r15` replay) | 3/12 | 8/12 | 8/12 | 0/12 |
| orig_P | 0/12 | 0/12 | 0/12 | **12/12** |
| orig_O | 0/12 | 0/12 | 0/12 | 0/12 |
| **corr_P** | 0/12 | **11/12** | **11/12** | 5/12 |
| corr_O | 0/12 | 1/12 | 1/12 | 7/12 |

Full 12 s SUCCESS_NEW after bil: **0** verified (FULL_RECOVERY_VERIFICATION.csv empty of SN). Open-loop SN only on near-memorized ICs.

## Required conclusions (Q1–Q14)

1. **Why O failed horizontal?** Sign-reversed chase: `a ∝ P_TGT − p_rel` instead of toward jaw-mid error `p_rel − MID_OFFSET_H`.
2. **Wrong action vs bad tracking?** Wrong action; tracking verified by unit-axis tests.
3. **P colliding geometry?** Hand-body mesh `geom_66` (housing), not fingertip.
4. **Cause?** Lateral chase with continuous descend while XY open (height/lateral/timing); housing sweeps cylinder.
5. **Jaw interception avoid housing?** Yes on the calibration IC for corr_P (0 vs 1). Held-out housing 5/12 vs orig_P 12/12.
6. **Corrected O horizontal vs P?** Sign-fixed O reduces horizontal error (`href8≈+0.026` vs orig −0.068) but still lags corr_P (`≈+0.039`) and rarely reaches bil (1/12 held-out).
7. **Future prediction?** Little benefit here once phasing is fixed (corr_P ≈ corr_P_nopred).
8. **Shared jaw entry multi-IC?** corr_P **11/12** jaw_entry / first_bil under one frozen gain set.
9. **Stable bilateral / full recovery?** Brief bil yes; **stable through lift/HANDOFF/SN no** (0/12 SN).
10. **Capture-time wrist?** Shared law applies approach wrist from relative orientation; distal SN evidence used only as prior scales, not root scripts.
11. **Held-out SUCCESS_NEW?** Neither corrected controller; open-loop 3/12 memorization only.
12. **Sensitivity?** Height variants and distal offsets raise housing / miss rates (see HELDOUT rows `*_hz`, `distal_*`).
13. **Remaining failures?** After corr_P: **post-contact capture dynamics / unstable bil → no HANDOFF/SN**. Orig O: action sign. Orig P: housing interception. Corr O: still mostly interception (weak XY close).
14. **Shared closed-loop vs memorization?** One frozen shared structure/gains; corr_P jaw/bil generalizes across nearby ICs where open-loop SN does not — evidence of shared interception behavior, but **not** yet deployable SUCCESS_NEW experts.

## Artifacts

- `OBSERVATION_ACTION_MAPPING.md`
- `ACTION_RESPONSE_AUDIT.csv`, `FIRST_CONTACT_GEOMETRY.csv`, `HAND_COLLISION_AUDIT.csv`
- `JAW_ENTRY_GEOMETRY.csv`, `INTERCEPTION_PREDICTIONS.csv`
- `CONTROLLER_COMPARISON.csv`, `HELDOUT_GENERALIZATION.csv`, `FULL_RECOVERY_VERIFICATION.csv`
- `FAILURE_ANALYSIS.md`, `raw/`, `trajectories/`, `videos/`
- Repro: `python -u scripts/run_freefall_recatch_collision_aware_feedback.py --device cpu`
