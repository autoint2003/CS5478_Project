# Freefall Recatch: Post-Capture Pose Correction + Closed-Loop Feedback

**Source experts:** `results/expert_generation/horizontal_offset_freefall_recatch/`  
**Script:** `scripts/run_freefall_recatch_pose_feedback.py`  
**Output:** `results/diagnostics/freefall_recatch_pose_feedback/`  
**ICs:** constructed free-fall (explicitly labeled). Legal continuous 7D only.  
**Not done:** SFT/GRPO, K49 changes, 39D/physics/controller/reward edits, large-offset expansion.

## Stage A — Reference replay (3/3 each)

| IC | Replay | SN | CH | theta@bil | theta@HO | pos@HO |
|----|--------:|---:|---:|----------:|---------:|-------:|
| `perp_off01_r15` | 3/3 | 1 | 1 | 5.02 | 4.65 | 2.19 |
| `perp_off01_r100` | 3/3 | 1 | 1 | 5.02 | 4.91 | 3.02 |
| `distal_off01_r8` | 3/3 | 0 | 1 | **21.65** | **20.47** | 8.95 |

Distal timeline (`DISTAL_POSE_ERROR_TIMELINE.csv`): theta falls from ~89° during freefall, then locks ~21.7° at first bilateral (step 7) and stays ~20.2–20.5° through lift/HANDOFF/nominal.

## Stages B–C — Distal pose correction

**Post-capture** (snapshots after bil / mid-lift / pre-HO; modes none/wrist/pivot/slide/combined):  
max orientation improvement ≈ **1.5°**; **0/64** trials reach theta≤10°.

**During-capture early wrist** (inject `wx,wy,wz` into catch prefix through cut+1, then legal lift/seat/axis_align):  
**27/87** trials reach theta≤10° and pos≤12 mm locally. Best bil theta ≈ **0.11°**.

**Full SUCCESS_NEW:** **Yes**, method `early_wrist_best` (`wy=0.14`, `wz=0.20`), cut=10.  
SN mean theta≈0.79°, pos≈2.08 mm, held 12 s, **3/3 exact replay**.  
Trajectory: `trajectories/distal_off01_r8__SUCCESS_NEW.npz`  
Provenance: `open_loop_traj_opt_during_capture_early_wrist` (distinct from privileged/obs feedback experts).

Videos: `videos/distal_off01_r8__CATCH_HANDOFF_fail.mp4`, `...__best_correction.mp4`, `...__SUCCESS_NEW.mp4`.

## Stage D — 39D observation audit

See `OBSERVATION_39D_AUDIT.md`. Summary:

| Quantity | In 39D? | Notes |
|----------|:-------:|-------|
| Cylinder relative position | yes | airborne `p_rel` (scaled) |
| Relative orientation | yes | `rot_rel` rotvec (scaled) |
| Relative linear / angular velocity | yes | `v_rel`, `w_rel` |
| Gripper opening | yes | aperture (+ ap_dot) |
| Hand velocity | yes | 27D prefix |
| L/R contact presence / normals | yes | contact channels |
| Remaining task time | **no** | not observed |
| Explicit axis-vs-`-z_hand` angle | infer | from `rot_rel` / history |
| Relative acceleration / contact onset | history | finite differences / contact sequence |

Matched freefall timing changes produce distinguishable 39D vectors (`raw/obs_39d_audit.json`).

## Stages E–G — Feedback controllers

Shared Controllers **P** (privileged MuJoCo state → legal 7D) and **O** (39D + causal history → legal 7D) calibrated on perp×2 + distal, then **frozen**. Gains: `raw/feedback_gains.json`.

### Held-out (8 independent constructed freefall ICs)

| Controller | SN | CH | first_contact | first_bil |
|------------|---:|---:|:-------------:|:---------:|
| open_loop (`perp_off01_r15` replay) | **1/8** | 1/8 | 6/8 | 6/8 |
| Controller P | **0/8** | 0/8 | **0/8** | 0/8 |
| Controller O | **0/8** | 0/8 | **0/8** | 0/8 |

Only held-out SN is near-identical `perp_off01_h0` under open-loop expert replay. P/O fail **before contact**.

### Timing robustness

OL / P / O: **0/6** SN under small height/offset perturbations of `perp_off01_r15`.

### New expert candidates

| IC | SN | Provenance |
|----|---:|------------|
| `distal_off01_r8` | 1 | open-loop during-capture early wrist (verified 3/3) |
| observation-only held-out | 0 | none accepted |
| privileged held-out | 0 SN | diagnostic only; not deployable experts |

## Required conclusions (Q1–Q14)

1. **Why ~20.5° on distal?** Capture geometry: locked at first bilateral (~21.7° at step 7); persists through lift/HANDOFF/nominal. Not created by nominal continuation.
2. **Legal correction below 10°?** Post-capture: **no** (0/64). During-capture early wrist: **yes** (27/87 local; full SN verified).
3. **Preserve pos ≤12 mm?** Yes jointly with (2) for early-wrist successes (incl. verified SN pos≈2.1 mm).
4. **Complete SUCCESS_NEW?** **Yes** for distal with early-wrist open-loop correction (3/3).
5. **Correction during capture?** **Required.** Post-capture pivoting lacks orientation authority (<2°).
6. **39D available?** Relative pose, twist, aperture, contacts, hand velocity — yes.
7. **Need history?** Acceleration, contact onset/offset, freefall progress timing — yes. Absolute remaining time — unavailable.
8. **Privileged feedback generalize?** **No** on this held-out set (0/8; no first contact).
9. **39D feedback generalize?** **No** (0/8; no first contact).
10. **Improve on 0/12 open-loop robustness?** **Not with these controllers.** Held-out OL 1/8 (near-exact IC only); P/O 0/8.
11. **New independent SUCCESS_NEW roots?** **0** new independent ICs. Distal same-root corrected to SN (1 verified candidate). Obs-feedback: 0.
12. **Failure sources?** Distal baseline: **capture geometry / pose**. Post-capture search: **contact authority**. Feedback held-out: **interception**. Distal corrected CH gate: **HANDOFF relative-motion** (SN still passes). Not 39D observability for relative pose.
13. **Unified Transformer suitability?** Distal early-wrist SN is a valid continuous Recovery expert candidate for a shared pool, but provenance must stay open-loop (not privileged runtime). Observation-feedback does **not** yet provide multi-root coverage; memorized open-loop still dominates.
14. **Missing before SFT?** (a) Controllers that achieve first-contact / jaw-entry on nearby ICs under frozen shared params; (b) more independent SUCCESS_NEW roots beyond perp×2 + corrected distal; (c) held-out SN rates clearly above open-loop memorization; (d) shared discrete K49 mapping later — **not** yet.

## Artifacts

- `REFERENCE_REPLAY.csv`, `DISTAL_POSE_ERROR_TIMELINE.csv`
- `POST_CAPTURE_CORRECTION.csv`, `HANDOFF_VERIFICATION.csv`
- `OBSERVATION_39D_AUDIT.md`
- `FEEDBACK_CONTROLLER_COMPARISON.csv`, `HELDOUT_GENERALIZATION.csv`, `TIMING_ROBUSTNESS.csv`
- `EXPERT_CANDIDATES.csv`, `FAILURE_ANALYSIS.md`
- `raw/`, `trajectories/`, `videos/`, `snapshots/`
- Repro: `python -u scripts/run_freefall_recatch_pose_feedback.py --device cpu`
