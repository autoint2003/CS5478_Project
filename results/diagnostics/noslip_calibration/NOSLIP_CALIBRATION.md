# NoSlip calibration (MuJoCo 3.12.0)

**Later freeze:** the project now sets `noslip_iterations = 1` and `noslip_tolerance = 1e-6` in `config/sim.yaml`, `envs/xml_build.py` / `assets/panda_torque.xml`, and `envs/grasp_sim.py` `apply_solver_from_cfg`. solref / solimp / mu / condim / dt / solver / gains were not changed with that freeze.

This calibration sweep itself did **not** rewrite official XML at the time it was run. Live contact softness stayed at **solref = [0.015, 1.0]**. `noslip_tolerance` was **not** swept (live **1e-6**). Main Newton iterations/tolerance, dt, integrator, mu, condim, cone, impratio unchanged during the sweep.

Script: `training/noslip_calibration.py`  
Raw: `results/diagnostics/noslip_calibration/raw/`  
Figures: `results/diagnostics/noslip_calibration/figures/`

Predefined **before** looking at the sweep (simulation numerical floors, not robot specs):

- Preferred: **|slope| ≤ 0.05 mm/s** and **|Δ r_h.z| ≤ 0.5 mm / 10 s** on **both** cylinder and box.
- Loose: 0.10 mm/s and 1.0 mm / 10 s.
- Choose the **smallest** `noslip_iterations` that meets the preferred floor **and** preserves capacity-limited slip and true release. Do **not** choose the creep minimum.

---

## VERIFIED API / IMPLEMENTATION FACT

Installed **MuJoCo 3.12.0**. Live `mjOption` at parent:

```json
{"noslip_iterations": 0, "noslip_tolerance": 1e-06, "iterations": 100, "tolerance": 1e-08,
 "solver": 2, "cone": 0, "impratio": 1.0, "timestep": 0.002, "integrator": 3}
```

From `engine_forward.c` (3.12.0) and `engine_solver.c` `solNoSlip`:

- NoSlip runs **after** the main solver (here Newton). If `noslip_iterations > 0`, `mj_solNoSlip` then `mj_dualFinish` (dual force mapping is required).
- It is a **modified PGS post-process**. It does **not** re-solve normals/equalities as a full Newton step.
- Friction rows: dry friction `ne … ne+nf`, then contact friction. For **pyramidal** cones it loops **pairs of opposing pyramid edges**. With **condim=4**, `dim=4` → `2*(dim-1)=6` edge rows per contact. The pair **midpoint** `(f0+f1)/2` (mixed normal content) is **held**; only the difference `y` (friction) is optimized and clamped to `[-mid, mid]`.
- Regularization is **removed in this step**: `ARdiaginv(..., flg_subR=1)` and `extractBlock(..., flg_subR=1)` use **AR − R** on the diagonal (docs: “ignores constraint regularization”).
- Stop: scaled `improvement < noslip_tolerance` (live 1e-6).
- Docs: this cascade is **ad hoc**, not a well-defined optimization problem; possible instability with dense multi-contact. Cost: extra friction PGS sweeps ∝ `noslip_iterations` × frictional rows.

---

## Test C grip sweep at noslip=0 (RAW) — chosen before NoSlip replay

2.5 s freeze-here from the same centered airborne cylinder, vary **only tau**:

| tau | class | Δez mm | med vz | rho mean | notes |
|---|---|---|---|---|---|
| −18 | marginal* | 2.98 | 0.0012 | ~0.11 | *this is the known 1.22 mm/s creep, not Coulomb |
| −12 | same floor | 3.05 | 0.0012 | ~0.11 | |
| −8 | same floor | 2.91 | 0.0012 | ~0.11 | |
| −4 | faster creep | 4.33 | 0.0017 | | still low-to-mid utilization |
| **−2** | **clearly slipping** | **15.2** | 0.0037 | **0.97** | Fn~1 N, ρ→1, unilateral |
| 0 / +4 / +8 | released | 66.8 | ~0 | n=0 | contacts gone, object falls |

**CLEAR-SLIP state for Test C:** `tau = −2` (ρ ≈ 0.97, under-supported). **Not** reused blindly from recovery lore: the sweep showed −2 is the first **capacity-limited** case.

**TRUE RELEASE:** `tau = +8` (`config/nominal.yaml` `open_tau`).

---

## Tests A/B — secure 10 s hold, tau=−18 (RAW)

| noslip | cyl slope / Δ10s | box slope / Δ10s | preferred both? | 10 s cyl wall s |
|---|---|---|---|---|
| 0 | 1.220 / 12.08 | 0.774 / 7.73 | no | 3.60 |
| **1** | **−0.000 / −0.031** | **−0.007 / −0.100** | **yes** | 2.88 |
| 2 | −0.000 / 0.005 | −0.000 / 0.000 | yes | 3.91 |
| 3 | −0.000 / 0.006 | −0.000 / 0.000 | yes | 3.57 |
| 5 | −0.000 / 0.006 | −0.000 / 0.000 | yes | 3.64 |

Plateau at **1**. 2–5 add nothing material. Did **not** extend to 10/20.

Secure residual remains ~10⁻⁴–10⁻³ N. Live solref still [0.015, 1.0]. Aperture drift ≪ 0.1 mm. Hand world sag ~10 mm remains (arm PD, not relative creep): at ns=1, Δ object world z ≈ −10 mm **with** the hand, Δ `r_h.z` ≈ 0.

Trajectories: `raw/cyl_ns*.npz`, `raw/box_ns*.npz`; `figures/fig_traj_ns0.png`, `fig_traj_ns1.png`, `fig_creep_vs_noslip.png`.

---

## Tests C/D — under-supported slip and release (RAW)

| noslip | tau=−2 Δez mm / ρ / drop | tau=+8 Δobj z mm / drop |
|---|---|---|
| 0 | 15.2 / 0.97 / no (unilateral, object −25 mm) | −66 / **yes** |
| 1 | 18.0 / 0.97 / unilateral, object −28 mm | −66 / **yes** |
| 2 | 16.6 / 0.97 / same class | −66 / yes |
| 3–5 | still capacity-limited slip | still falls |

NoSlip **does not** weld a τ=−2 grasp. Open τ=+8 still loses contacts and the object falls (~1.1 m/s peak |v_rel| at release).

---

## Test E — tilt / “controlled slip” (RAW)

Tilt IC: from the same parent, rotate **+hand-y to ~60° at τ=−18, noslip=0**, freeze, then **replay the 2 s hold** at each noslip (same qpos/qvel/controls).

**E1 — 60° hold, τ=−18 (low utilization):**

| noslip | Δez mm / 2 s | Δex mm | med vz | ρ |
|---|---|---|---|---|
| 0 | 1.83 | −0.30 | 0.00099 | **0.10** |
| 1 | 0.032 | ~0 | ~0 | 0.11 |

This motion is **class A** (ρ ≪ 1), the same regularization creep as the vertical pinch, only with gravity partly in −hand-x. NoSlip **removes it**. That is expected, not a Coulomb-pivot.

**E2 — same 60° IC, τ=−2 (capacity-limited):**

| noslip | Δez mm | Δex mm | ρ | drop |
|---|---|---|---|---|
| 0 | 70.6 | −97.5 | 0.94 | **yes** |
| 1 | 73.1 | −96.2 | 0.94 | **yes** |
| 2 | 73.5 | −95.8 | 0.93 | **yes** |

Intended **under-supported** relative motion at 60° is **not** suppressed.

If the project’s “gravity-assisted inward slide at τ=−18” was counted as physical slip: it was not. Enabling NoSlip will remove that particular motion. Repositioning that needs real sliding must run **near/over friction capacity**, not at τ=−18.

---

## Test F — canonical ball–cylinder impact (RAW)

`measure_canonical_impact` with only `opt.noslip_iterations` changed:

| | ns=0 | ns=1 |
|---|---|---|
| t_first_contact | 4.392 s | 4.392 s |
| Δp_rel norm | 4.44 mm | 5.37 mm |
| rot angle | 85.70° | 85.87° |
| wall | 1.80 s | 1.79 s |

Same contact time; impulse rotation essentially unchanged; translation within the same millimetre-scale. **Not pathological.** Not used to pick the iteration count.

---

## Cost

10 s cylinder hold wall-clock ≈ 2.9–3.9 s across ns=0–5. **No useful slowdown trend.** Extra iterations after 1 are not buying physics.

---

## INTERPRETATION

**CASE A — small NoSlip value works**, with a scoped caveat.

- Preferred secure-creep floor is met at **noslip_iterations = 1** for **both** cylinder and box.
- Larger values plateau; **do not pick 5 because creep is slightly flatter**.
- Capacity-limited gravity slip (τ=−2, ρ≈0.97) and true release (τ=+8) remain.
- Low-ρ “slides” at τ=−18 (vertical **and** 60°) disappear — they were the artifact under test.
- Impact transient not obviously corrupted.
- Force residuals remain valid.

**RECOMMENDATION (candidate only, not an official model write):**

> **`noslip_iterations = 1`**  
> keep **`noslip_tolerance = 1e-6`** (untested to change).

Do **not** combine this with solref=0.005 in the same change. Do **not** treat this as proof a real Panda sticks. Review this evidence before touching XML.

If a later recovery strategy **requires** millimetre-scale relative motion at **τ=−18** (ρ~0.11), NoSlip is the wrong tool for that motion (**CASE B for that specific behavior**). Use a weaker, capacity-limited grip for physical sliding.

---

## USER VISUAL OBSERVATION

Not filled. Viewer (full grasp + 10 s, or tilt+hold):

```
python training/noslip_calibration.py --view secure --noslip 0 --playback-speed 0.35
python training/noslip_calibration.py --view secure --noslip 1 --playback-speed 0.35
python training/noslip_calibration.py --view pivot --noslip 0 --playback-speed 0.35
python training/noslip_calibration.py --view pivot --noslip 1 --playback-speed 0.35
```

Pivot viewer is the 60° τ=−18 hold (E1). Weak-grip slip is the headless τ=−2 logs (`raw/slip_ns0.npz`, `raw/slip_ns1.npz`).

---

Stopped. No official XML change, no recovery rerun, no SAC.
