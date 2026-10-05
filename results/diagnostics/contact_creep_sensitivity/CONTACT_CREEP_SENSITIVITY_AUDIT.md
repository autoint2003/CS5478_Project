# Contact-creep sensitivity: disturbed vs nominal vertical hold

Diagnostic only. Official scene XML, recovery strategy, tau, wrist rates, disturbance, reward, observation, and training were not changed. Live sliding `mu = 1.0` and `condim = 4` throughout.

Script: `training/contact_creep_sensitivity_audit.py`  
Raw: `results/diagnostics/contact_creep_sensitivity/raw/`  
Figures: `results/diagnostics/contact_creep_sensitivity/figures/`

This report separates **VERIFIED CODE FACT**, **RAW TRAJECTORY EVIDENCE**, **USER VISUAL OBSERVATION**, and **INTERPRETATION**. Cursor interpretation is not established fact.

---

## Evidence classes

- **VERIFIED CODE FACT:** MuJoCo / this repo’s restore and hold code.
- **RAW TRAJECTORY EVIDENCE:** logged 10 s holds (npz + json).
- **USER VISUAL OBSERVATION:** not filled in here; viewer commands are below.
- **INTERPRETATION:** case label under the requested rules, not a hardware claim.

---

## 1. Scientific question

Is persistent `+hand-z` relative creep specific to the disturbed post-RETURN grasp, or does the same MuJoCo contact formulation produce comparable creep on a healthy nominal centered grasp? Then: is that creep sensitive to contact-constraint softness vs solver accuracy?

---

## 2. Frozen states

### STATE D (disturbed vertical)

**RAW:** captured from the existing 1.0 s inward-slide → RETURN trajectory at first `vertical_hold` sample with `t >= 7.80 s` and `|v_rel| < 0.01`. Capture `t = 7.802 s` (near the previous audit B at `t ≈ 8.006 s`).

**VERIFIED CODE FACT:** store is a full `GraspSim.snapshot()`: `qpos`, `qvel`, `act`, `ctrl`, `time`, `qacc_warmstart`, `p_des`, `r_des`, `v_cmd`, `w_cmd`, FSM clocks, meter refs. Not reconstructed from object pose only.

Files: `raw/state_D.json`.

### STATE N (nominal vertical)

**RAW:** real nominal FSM (`advance_to_parent`, `mode=ZERO`): centered cylinder, `m = 0.20 kg`, pair `mu = 1.0`, normal grasp, normal airborne lift. No privileged pose injection. Freeze at parent airborne bilateral hold (`t = 4.132 s`).

Files: `raw/state_N.json`.

---

## 3. Hold semantics (VERIFIED CODE FACT)

For every cell, once at diagnostic start:

```
p_des := current p_hand
r_des := current R_hand
v_cmd := 0
w_cmd := 0
```

then `tick_vw(v=0, w=0, tau=-18)` for 10.0 s. Targets stay fixed. This is not “v_cmd=0 with a stale p_des”.

D baseline freeze `p_des == ph` at `[0.4949, 0.0029, 0.5783]`.

No task lift. No OPEN / RECLOSE. No recovery overlay beyond STATE / variant / hold time.

---

## Restore pitfall (VERIFIED CODE FACT) — why the first attempt was invalid

`set_finger_object_sliding_mu(..., data=sim.data)` calls `mj_setConst`. After a restored grasp, that **reset `qpos` to the home keyframe** while leaving stale `ncon=20`. Freeze then locked `p_des` at home `[0.088, 0, 0.926]` and the object dropped. That run is **not** used below.

The matrix uses restore **without** post-restore `mj_setConst`. Pads/object already have `mu=1.0` from `make_parent_sim`. Official XML was not written.

---

## 4. Live contact / solver defaults (CODE + RAW at D restore)

| quantity | value |
|---|---|
| MuJoCo | 3.12.0 |
| timestep | 0.002 s |
| solver | Newton (`opt.solver=2`) |
| iterations / ls_iterations | 100 / 50 |
| tolerance | 1e-8 |
| cone | pyramidal (`0`) |
| impratio | 1.0 |
| object XML solref | `[0.01, 1]` |
| pad geom_solref | `[0.02, 1]` |
| **live contact solref** | **`[0.015, 1.0]`** |
| live solimp | `[0.9, 0.95, 0.001, 0.5, 2.0]` |
| live mu / condim | 1.0 / 4 |

Harder-contact family (one variable): runtime `geom_solref[0] = 0.005` on object + finger geoms only (stiffer time-constant vs live 0.015; damping ratio unchanged). No solimp / cone / condim / mu / solver change. Diagnostic copy of the in-memory model only.

Solver-accuracy family (one variable): iterations 100→200, ls 50→100, tolerance 1e-8→1e-9. Physical contact parameters unchanged.

---

## 6. STATE D baseline replay vs previous audit

Previous vertical-hold creep ≈ **1.24 mm/s** with `v_rel_h.z > 0`, `rho ≈ 0.11`, `nL/nR ≈ 10/10`, `Fn ≈ 9 N`.

**RAW, this D baseline (freeze-here, 10 s):**

- robust slope `r_h.z` over 1–10 s = **1.239 mm/s**
- median `v_rel_h.z` 1–10 s = **+0.00124 m/s**
- Δ`r_h.z` 0–2 / 0–5 / 0–10 s = **2.48 / 6.21 / 12.38 mm**
- `rho` mean/max ≈ 0.109 / 0.128
- `nL` 9–10, `nR` 9–10; `Fn` ≈ 9 N
- `|m a − (Fc+Fg)|` median **0.0003 N**
- not dropped; no unilateral / both-contact loss

**Replay gate: PASS.** Sensitivity matrix is valid to interpret.

Freeze-here did **not** remove the creep (not CASE 4).

---

## 11–12. Matrix (RAW)

Duration 10 s, log 10 ms. Creep rate = Theil–Sen median pairwise slope of `r_h.z` on 1–10 s (not two-point). Excess = rate_D − rate_N.

| cell | Δez 2/5/10 s (mm) | slope 1–10 (mm/s) | med vz 1–10 (m/s) | rho mean / max | nL / nR | Fn L / R (N) | drop / uni / both | \|res\| (N) |
|---|---|---|---|---|---|---|---|---|
| D baseline | 2.48 / 6.21 / 12.38 | **1.239** | +0.00124 | 0.109 / 0.128 | 9–10 / 9–10 | 8.99–9.02 / 8.98–9.01 | F/F/F | 0.0003 |
| N baseline | 2.39 / 6.11 / 12.08 | **1.220** | +0.00120 | 0.113 / 0.224 | 8–14 / 8–15 | 9.04–9.10 / 9.08–9.15 | F/F/F | 0.0052 |
| D harder | 0.84 / 2.11 / 4.30 | **0.430** | +0.00042 | 0.109 / 0.130 | 8–10 / 9–10 | 8.96–10.70 / 8.94–9.49 | F/F/F | 0.0007 |
| N harder | 0.74 / 1.89 / 3.81 | **0.385** | +0.00038 | 0.109 / 1.000* | 7–16 / 8–14 | 8.95–10.95 / 9.06–11.20 | F/F/F | 0.0050 |
| D solver | 2.48 / 6.21 / 12.38 | **1.239** | +0.00124 | 0.109 / 0.128 | 9–10 / 9–10 | same as D baseline | F/F/F | 0.0003 |
| N solver | 2.39 / 6.11 / 12.08 | **1.220** | +0.00120 | 0.113 / 0.224 | same as N baseline | same as N baseline | F/F/F | 0.0052 |

\*N harder `rho_max=1.0` is a peak on some contact; median mean-rho stays 0.109. Force residual remains small; cell kept VALID.

All cells: live `mu=1.0`, `condim=4`. Harder live solref timeconst = 0.005. Solver trajectories **bit-match** baseline for D and for N (Newton already converged at 100 / 1e-8).

### Excess creep (mm/s)

| condition | rate D | rate N | excess D−N |
|---|---|---|---|
| BASELINE | 1.239 | 1.220 | **+0.018** |
| HARDER CONTACT | 0.430 | 0.385 | **+0.045** |
| HIGHER SOLVER ACC | 1.239 | 1.220 | **+0.018** |

Absolute creep is ~1.2 mm/s for **both** D and N. Excess is ~1–4% of that floor.

Centroid `z` drift: D baseline +0.40 / +0.07 mm (L/R); N baseline −1.55 / −0.63 mm. Aperture drift is ~1e-7 to 1e-4 m. Hand `|p_des−p_hand|` ends near **10 mm in world z** on every cell (arm tracking sag under the freeze target, similar D vs N). Orientation error ~0.2 deg.

`r_h.z(0)`: D 107.3 mm, N 98.6 mm (N is more proximal, as expected for a centered grasp). Creep **rates** still match.

---

## 13. Force balance

**RAW:** median `|m a − (Fc+Fg)|` is 3e-4 N (D) and 5e-3 N (N) on all three contact-model families. No cell marked INVALID.

---

## 17. Figures

- `figures/fig1_rh_z.png` — D vs N baseline Δ`r_h.z`
- `figures/fig2_vrel_z.png` — `v_rel_h.z`
- `figures/fig3_rho.png` — rho mean/max
- `figures/fig4_centroid_z.png`
- `figures/fig5_Fn.png`
- `figures/fig6_track.png` — hand `|p_des−p_hand|`
- `figures/fig_all_rh_z.png` — all six trajectories (not hidden behind bars)
- `figures/fig_creep_rates.png` — slope D, N, excess by condition

Reproduce rates from `raw/{D,N}_{baseline,harder,solver}.npz` keys `t_hold`, `e_z`.

---

## 18. Viewer (full 10 s hold)

```
python training/contact_creep_sensitivity_audit.py --view D --variant baseline --playback-speed 0.35
python training/contact_creep_sensitivity_audit.py --view N --variant baseline --playback-speed 0.35
python training/contact_creep_sensitivity_audit.py --view D --variant harder --playback-speed 0.35
python training/contact_creep_sensitivity_audit.py --view N --variant harder --playback-speed 0.35
```

Requires the saved `raw/state_D.json` / `state_N.json`. Overlay: DISTURBED/NOMINAL, BASELINE/HARDER/SOLVER, elapsed hold time. No early “success” exit. No recovery actions.

**USER VISUAL OBSERVATION:** not recorded in this file.

---

## 15. INTERPRETATION (not a hardware result)

Requested labels, applied only as supported:

- Not CASE 4: freeze-here D still creeps at the previously audited ~1.24 mm/s.
- Not CASE 1: nominal is **not** stationary.
- Not CASE 5: force checks valid; hardness response is monotonic (stiffer → slower).
- Solver doubling is a null result (already tight), not non-monotonic physics.

**CASE 2 — CONTACT-MODEL / NUMERICAL CREEP DOMINATES**

Nominal and disturbed both show ~1.22–1.24 mm/s distal creep under the unmodified soft contacts (`live solref` timeconst 0.015 s, `condim=4`, pyramidal cone). Both shrink by about **3×** when only that time-constant is shortened to 0.005 s. Disturbed minus nominal excess stays ≤ 0.05 mm/s, far below the shared floor.

Under the stated rules: **do not treat this vertical creep as evidence that aggressive regrasp is physically required.** It is not a disturbed-state-specific Coulomb failure (`rho` remains ~0.11).

**Real-world claim limit:** this does **not** prove a real robot would be stable or would slip. Strongest allowed statement:

> Within the tested MuJoCo 3.12 formulations, the apparent post-RETURN vertical creep is a shared soft-contact / constraint-creep floor, not a disturbed-grasp-only instability. Harder contact constraints reduce it; extra Newton iterations at the default tightness do not.

---

## STOP

No OPEN/RECLOSE, no tau/theta change, no official model overwrite, no SAC, no training-distribution change. Next recovery experiment should treat vertical creep as **simulator-contact-dominated** unless a later, separate study (e.g. condim family) shows otherwise.
