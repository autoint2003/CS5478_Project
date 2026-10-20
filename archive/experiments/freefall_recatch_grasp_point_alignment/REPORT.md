# Grasp-Point Alignment & Closure Timing

**Script:** `scripts/run_freefall_recatch_grasp_point_alignment.py`  
**Out:** `results/diagnostics/freefall_recatch_grasp_point_alignment/`  
No SFT/GRPO. Legal 7D only. Cylinder half-length = 0.03 m. Physics / 39D / SUCCESS_NEW unchanged.

## Matched audit (`perp_off01_r15`)

Same IC / MuJoCo state / controller clock for expert vs corrected-P.

| Quantity at first bilateral | Expert | corr_P |
|----------------------------|-------:|-------:|
| SUCCESS_NEW | 1 | 0 |
| first bil step | 8 | 6 |
| mean `s_axial` (m) | −0.0051 | −0.0239 |
| near-tail finger contacts | 1 | 25 |
| near-center finger contacts | 26 | 0 |
| theta (deg) | 5.0 | 83.3 |
| close command step | 7 | 5 |
| `p_rel` at close | ≈(−0.003, 0.011, 0.104) | ≈(−0.031, 0.005, 0.106) |

Also verified: `perp_off01_r100` and corrected `distal_off01_r8` SUCCESS_NEW experts (axial-center contacts).

**Mechanism:** Expert gets a brief hand graze (~step 2) that spins the cylinder (~15 rad/s) from horizontal to upright before finger close. corr_P wrist bias prevents that graze and closes early on a still-orthogonal cylinder → tip/tail clamp.

## Ablations (calib `perp_off01_r15`)

| Var | Meaning | bil | sustained | lift | CH | SN | fail |
|-----|---------|----:|----------:|-----:|---:|---:|------|
| A | corrected P baseline | 6 | 1 | 0 | 0 | 0 | relative motion |
| B | zero wrist bias + longer align | 5 | 2 | 0 | 0 | 0 | relative motion |
| C | delayed soft-capture 0.022 | 9 | 11 | 1 | 0 | 0 | position |
| D | velocity match only | 6 | 2 | 0 | 0 | 0 | relative motion |
| E | B+C+D + settle (joint) | 7 | 46 | 1 | **1** | 0 | ori ~18° |
| F | E + capture-time wrist | 7 | 46 | 1 | **1** | 0 | ori ~18° |

On calib, E/F reach CATCH_HANDOFF with θ≈18°, pos≈13 mm (SN needs ≤10° / ≤12 mm).

## Held-out (12 ICs, frozen gains)

| variant | SN | CH | bil | correct_axial | stable_lift | jaw_entry | housing |
|---------|---:|---:|----:|--------------:|------------:|----------:|--------:|
| A | 0/12 | 0/12 | 11/12 | 0/12 | 0/12 | 0/12 | 5/12 |
| B | 0/12 | 0/12 | 11/12 | 0/12 | 0/12 | 1/12 | 7/12 |
| C | 0/12 | 1/12 | 11/12 | 2/12 | 7/12 | 11/12 | 12/12 |
| D | 0/12 | 0/12 | 12/12 | 0/12 | 0/12 | 0/12 | 5/12 |
| **E** | **1/12** | **5/12** | 12/12 | **6/12** | **11/12** | 12/12 | 8/12 |
| **F** | **1/12** | **5/12** | 12/12 | **6/12** | **11/12** | 12/12 | 8/12 |
| corr_O_E | 0/12 | 0/12 | 11/12 | 2/12 | 4/12 | 11/12 | 8/12 |
| corr_O_F | 0/12 | 0/12 | 11/12 | 2/12 | 4/12 | 11/12 | 8/12 |

### Full SUCCESS_NEW verification (12 s, 3/3 replay)

| ic_id | variant | SN_12s | replay_ok | θ (°) | pos (mm) | provenance |
|-------|---------|-------:|----------:|------:|---------:|------------|
| distal_off01_h8 | E | 1 | 3 | 9.33 | 11.44 | privileged_state_feedback |
| distal_off01_h8 | F | 1 | 3 | 9.36 | 11.83 | privileged_state_feedback |

**Independent held-out SUCCESS_NEW roots: 1** (`distal_off01_h8`), achieved by both E and F.

## Required conclusions (Q1–Q15)

1. **Tail grasp?** Yes — corr_P on matched IC clamps near a cylinder end.
2. **Coordinates?** mean `s_axial ≈ −0.024` m (half-length 0.03); 25 near-tail / 0 near-center vs expert −0.005 / 26 near-center.
3. **When?** At close (step 5) / first bil (step 6), with θ≈83° and `p_rel_xy≈3` cm still open.
4. **Cause?** Early closure + incomplete XY/orientation alignment under loose `xy_soft_capture=0.042`, compounded by wrist bias that blocks the expert’s early graze-spin. Actuator delay is secondary.
5. **Expert avoids how?** Brief hand graze → cylinder tumbles upright; closes at step 7 with `p_rel≈P_TGT`, θ≲8°, axial contacts near COM; then lifts with matched relative velocity.
6. **Axial alone (B)?** Insufficient — enables graze path but without delayed close still fails capture stability (0 CH / 0 SN).
7. **Timing alone (C)?** Improves sustained bil / lift (7/12 lift held-out) and 1 CH, but housing-heavy and weak SN.
8. **Vel match (D)?** Alone does not fix tail grasps (0 axial, 0 CH).
9. **Capture-time wrist (F)?** Not necessary for the one SN root (E equals F); does not unlock the ~18° calib lock. Distal-style early wrist remains useful on some distal ICs historically.
10. **Most stable bilateral grasps?** **E** (and F): 46 sustained steps on calib, 11/12 stable lift, 5/12 CH held-out.
11. **corr_P / privileged E held-out SN?** **1/12** triple-verified (`distal_off01_h8`).
12. **corr_O (39D)?** Does **not** yet reproduce the improvement: 0/12 SN, 0/12 CH; grasp often lost after brief bil (relative-motion fail). Obs-only delay/vel-match needs better causal θ / vz estimates.
13. **Independent SN roots?** **1** held-out root (E and F both succeed on it).
14. **Generalize without per-root opt?** Partially — single frozen gain set; CH 5/12 and axial 6/12 without root ID, but SN still rare.
15. **Before larger Transformer expert set?** Need (a) more independent SUCCESS_NEW from the **shared** controller (esp. obs-only O), (b) a fix for the ~15–18° post-capture orientation lock without sacrificing grasp retention, (c) reduce housing graze rate while keeping the beneficial spin, (d) then expand mining — not SFT/GRPO yet.

## Controller (frozen E)

- Zero wrist bias (`kw=wy0=wz0=0`) during approach → preserves graze-spin.
- `n_align_retract=6`, `xy_early_scoop=0.035`.
- `xy_soft_capture=0.022`, `preclose=-1`.
- Velocity match: `a[2] += 0.8 * v_rel_z` near scoop/soft.
- Optional 1-step zero-grip settle; then lift + `P_TGT` hold.

## Artifacts

- `MATCHED_TRAJECTORY_AUDIT.csv`, `AXIAL_CONTACT_GEOMETRY.csv`, `CLOSURE_TIMING.csv`
- `CONTACT_FORCE_TRACES.csv`, `CAPTURE_STABILITY.csv`, `ABLATION_RESULTS.csv`
- `CONTROLLER_COMPARISON.csv`, `HELDOUT_GENERALIZATION.csv`, `FULL_SUCCESS_VERIFICATION.csv`
- `FAILURE_ANALYSIS.md`, `raw/`, `trajectories/`, `videos/`
- Videos: `expert_perp_r15.mp4`, `corr_P_fail_perp_r15.mp4`, `distal_off01_h8__E__SUCCESS_NEW.mp4`, `distal_off01_h8__F__SUCCESS_NEW.mp4`
