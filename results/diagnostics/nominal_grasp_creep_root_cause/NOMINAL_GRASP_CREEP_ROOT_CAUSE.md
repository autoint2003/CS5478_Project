# Why an undisturbed nominal pinch creeps (+hand-z)

Root-cause diagnostic of the **ordinary centered airborne grasp**. No recovery controller, no teleport, no impact, no RULE, no SAC. Official XML was not overwritten. Contact parameters were changed only on diagnostic in-memory copies.

Script: `training/nominal_grasp_creep_root_cause.py`  
Raw: `results/diagnostics/nominal_grasp_creep_root_cause/raw/`  
Figures: `results/diagnostics/nominal_grasp_creep_root_cause/figures/`

This file separates **VERIFIED CODE FACT**, **RAW TRAJECTORY EVIDENCE**, **USER VISUAL OBSERVATION**, and **INTERPRETATION / HYPOTHESIS**.

This is **not** a recovery-failure result. Disturbed vs nominal creep rates were already shown to match; the object of this audit is the **undisturbed** pinch.

---

## 1. Minimal test construction

**VERIFIED CODE FACT**

- Pipeline: `make_parent_sim()` + `advance_to_parent(..., mode="ZERO")`.
- Cylinder centered, grasp offset 0, `m = 0.20 kg`, pair sliding `mu = 1.0`.
- No `s=2` teleport, no impact ball, no wrist recovery, no OPEN/RECLOSE.
- After a healthy airborne bilateral pinch (`e_x < 3 mm`, both contacts, `|v_rel| < 0.05` for 0.20 s): freeze-here once:

```
p_des := p_hand
r_des := R_hand
v_cmd := 0
w_cmd := 0
tau := -18
```

then `tick_vw(0, 0, tau=-18)` for **10 s** with no further lift.

Object is airborne (table collision irrelevant).

Snapshot: `raw/state_natural.json`.

---

## 2. Viewer (USER VISUAL OBSERVATION not filled here)

```
python training/nominal_grasp_creep_root_cause.py --view natural --playback-speed 0.35
python training/nominal_grasp_creep_root_cause.py --view ideal --playback-speed 0.35
python training/nominal_grasp_creep_root_cause.py --view g0 --playback-speed 0.35
```

Natural viewer: approach → close → lift → freeze-here → full 10 s hold. Overlay: `NOMINAL STATIC PINCH`, phase, hold time, aperture, Δ`r_h.z`. No recovery overlay. No early exit.

Ideal / zero-g viewers start from the saved airborne snapshot (ideal is a privileged re-center; zero-g is a diagnostic gravity override).

---

## 3. Baseline 10 s (RAW)

| quantity | value |
|---|---|
| Δ `r_h.z` 1 / 2 / 5 / 10 s | **1.21 / 2.39 / 6.11 / 12.08 mm** |
| robust slope 1–10 s | **1.220 mm/s** |
| median `v_rel_h.z` 1–10 s | **+0.00120 m/s** |
| `r_h.z(0)` | 98.59 mm |
| rho mean / max | 0.113 / 0.224 |
| Fn L / R median | 9.07 / 9.10 N |
| `nL` / `nR` | 8–14 / 8–15 |
| mean `contact.dist` | **−0.117 mm** (penetration, nearly constant) |
| `|m a − (Fc+Fg)|` median | 0.0046 N |
| ctrl[7] / actuator_force[7] | −18 / −18 N |
| finger `qfrc_actuator` | −9.0 N, −9.0 N |

**VERIFIED CODE FACT (distal sign):** `grasp_orientation()` in `controllers/nominal.py` sets hand Z = world down (`R[:,2] = [0,0,-1]`). `r_h = R_h^T (p_obj − p_hand)`. **+`r_h.z` is distal / toward the fingertips.** Persistent **positive** `v_rel_h.z` is object creeping down the fingers.

Figures: `figA_rh_vrel.png`.

---

## 4. Is the hand moving or the object? (RAW)

Over 10 s at 1 g:

| | Δ world z | notes |
|---|---|---|
| hand | **−10.03 mm** | sinks vs frozen `p_des` (PD sag) |
| object | **−22.11 mm** | |
| object − hand (world z) | **−12.08 mm** | |
| `r_h.z` | **+12.08 mm** | matches world-down relative motion |

`p_des − p_hand` ends at about `[-1.08, 0.00, +10.03]` mm. Orientation error ~0.20 deg.

**INTERPRETATION:** Cartesian tracking **does** drop the hand ~10 mm in world z (gravity load on the arm). That does **not** explain `r_h.z`. Relative object–hand distal motion is a further **12 mm**. Rule 4 “if the hand itself drifts enough to explain `r_h.z`, STOP” is **not** triggered for the relative creep.

Figure: `figB_world_vs_rel.png`.

Zero-g control (below): hand sag and relative creep both collapse. Arm sag is gravity on the arm; relative creep is gravity on the object through the contacts.

**A (controller / hand drift) is ruled out as the cause of relative creep.** It is a parallel, separate 10 mm world-z tracking bias.

---

## 5. Grip actuation semantics (VERIFIED CODE FACT)

Live `mjModel` at parent (`raw/actuator_at_parent.json`):

| | |
|---|---|
| actuator | `actuator8` (index 7) |
| type | `motor` |
| transmission | **TENDON** (`trntype=3`) tendon `split` |
| dyntype / gaintype / biastype | **none / FIXED / NONE** |
| gear[0] | 1 |
| ctrlrange / forcerange | [−50, 50] N |
| `ctrl[7] = −18` | command |
| `actuator_force[7] = −18` | realized tendon force (N) |

XML: `assets/panda_torque.xml`

```xml
<tendon>
  <fixed name="split">
    <joint joint="finger_joint1" coef="0.5"/>
    <joint joint="finger_joint2" coef="0.5"/>
  </fixed>
</tendon>
<actuator>
  <motor name="actuator8" tendon="split" gear="1" ctrlrange="-50 50" forcerange="-50 50"/>
</actuator>
```

**What `ctrl[7] = −18` means:** this is a **force-like tendon motor**, not a position or velocity servo.  
`F_tendon = gear * ctrl = −18 N`.  
Each slide finger joint receives `coef * F_tendon = 0.5 * (−18) = **−9 N**`.  
Measured `qfrc_actuator` on `finger_joint1/2` is exactly **−9, −9 N**.

That reconciles with ~9 N **per-finger summed |Fn|**: the closing load is ~9 N per finger; pad normals carry that plus a small extra from kinematics/contact multiplicity. It is **not** “18 N normal force per pad.”

The `<equality><joint finger_joint1 finger_joint2 …></equality>` block is a **symmetry constraint** (`solref="0.005 1"`), not the grip actuator.

`fg_cmd` in the FSM is only a naming convention: `fg_to_tau(fg) = −fg`. Do not read “fg” as Newtons of pad normal.

---

## 6. Do the fingers creep? (RAW)

| | 10 s change |
|---|---|
| aperture | **+0.011 mm** |
| finger qpos | ~0.01771 m, essentially flat |
| tendon length | tracks aperture; velocity ~1e-4 m/s at parent, not a 12 mm mechanism |

**0.01 mm of finger opening cannot produce 12 mm of distal object travel.** The object is sliding relative to essentially **fixed pads**, not riding yielding joints.

Figure: `figC_fingers.png`.

**B (grip actuator compliance) is ruled out as the relative-creep mechanism.**

---

## 7. Force path (RAW; same extraction as the contact-mechanics audit)

`mj_contactForce` → contact frame → world → force-on-cylinder → hand frame.

At hold start (1 g), hand frame:

- `F_contact_h ≈ [0.001, 0.046, −1.959] N`
- `F_gravity_h ≈ [0.000, −0.042, +1.962] N`  (hand-z is world-down, so weight is **+hand-z**)
- `a ≈ 0`
- `|m a − (Fc+Fg)| ≈ 0.005 N`

Same check at 1 / 5 / 10 s remains ~10⁻³ N. **VALID.** Quasi-static: contacts support weight; they do not produce a Coulomb-limit sliding force (`rho ≈ 0.11`).

---

## 8–10. Contact formulation and origin of live `solref = [0.015, 1.0]`

**VERIFIED CODE FACT (installed MuJoCo 3.12.0 + this model)**

| | |
|---|---|
| integrator | `implicitfast` (`opt.integrator=3`), from `assets/panda_torque.xml` |
| timestep | **0.002 s** (same file `<option>`) |
| solver | Newton, 100 iterations, ls 50, tolerance 1e-8 |
| cone | pyramidal (`0`) |
| impratio | 1.0 |
| noslip_iterations | **0** |
| cylinder | `assets/scene.xml`: `condim="4" solref="0.01 1"`; solimp **unspecified** → default `(0.9, 0.95, 0.001, 0.5, 2)` |
| pads | `assets/panda_torque.xml` `fingertip_pad_collision_*` boxes: **no solref/solimp** → default **`solref=(0.02, 1)`**, default solimp, `condim=3` on pads |
| mix | `geom_solmix=1`, `priority=0` both sides → **average** of solref timeconst: `0.5*(0.01+0.02)=0.015` |
| live contact | `solref=[0.015, 1.0]`, `condim=4` (max of pair), `mu=1.0`, margin=0, gap=0 |

The **0.015 s time-constant was not a chosen pair parameter**. It is the accidental mix of an object-only XML `0.01` and Menagerie/default pad `0.02`.

**Low rho vs sticking (docs, not a guess):** condim=4 pyramidal contacts are **soft constraints**. `solref`/`solimp` regularize **all** constraint rows of that contact, including friction. With `noslip_iterations=0` there is no extra projection that drives tangential velocity to zero. A friction force well inside `μ Fn` (`rho~0.11`) can coexist with a small persistent tangential velocity set by constraint compliance and the tangential load (here, object weight along +hand-z). That is **regularized constraint creep**, not “Coulomb slip at the cone.”

---

## 11. Harder-contact causal copy (RAW)

In-memory only: object + pad `geom_solref[0] → 0.005`. Nothing else.

| | slope mm/s | Δez 10 s mm | mean dist mm | aperture Δ mm |
|---|---|---|---|---|
| baseline 0.015 | 1.220 | 12.08 | −0.117 | +0.011 |
| harder 0.005 | **0.385** | **3.81** | **−0.012** | +0.104 |

Creep drops ~**3.2×**; penetration drops ~10×. Hand world sag stays **−10.03 mm** (arm PD, independent of pad stiffness). Fn still ~9 N; rho mean still ~0.11.

Figure: `figE_harder.png`.

This confirms **causal dependence on contact time-constant**, not a retune recommendation.

---

## 12. Timestep sensitivity (RAW)

Same physics parameters; only `opt.timestep` on a copy. solref **not** compensated.

| dt | slope mm/s | Δez 10 s mm |
|---|---|---|
| 0.002 (official) | 1.220 | 12.08 |
| 0.001 | 1.173 | 11.80 |
| 0.0005 | 1.172 | 11.69 |

**Interpretation A:** creep is **nearly unchanged** as `dt → dt/4` (~4% then flat). Discretization is not the primary source.

Figure: `figF_timestep.png`.

Solver doubling was already a null result in the previous audit (100→200, 1e-8→1e-9, bit-identical). Not re-swept.

---

## 14–15. Gravity controls (RAW)

Same airborne snapshot; only `opt.gravity` on a copy. `tau=-18` unchanged.

| g | slope mm/s | Δez 10 s | Δ hand world z | slope / (1 g) |
|---|---|---|---|---|
| 1.0 g | 1.220 | 12.08 mm | −10.03 mm | 1.00 |
| 0.5 g | **0.611** | 6.08 mm | −4.59 mm | **0.50** |
| 0.0 g | **−0.002** | 0.00 mm | +0.85 mm | **~0** |

**Distal creep disappears at 0 g** and **scales linearly with g**. Fn stays ~9 N (grip), so the **tangential** load is what scales.

Figure: `figG_gravity.png`.

Mass sweep was **not** run: gravity already isolates load dependence.

---

## 17–18. Penetration vs churn (RAW)

Mean `contact.dist` is a **steady ~0.12 mm overlap**, not a growing penetration that would look like sinking into the pad along the normal.

Position-hashed contact keys (0.1 mm grid) show high birth/death rates (~355 / s) and mean key lifetime **0.083 s**. That lifetime is what you get if contact **points translate along the cylinder at ~1.2 mm/s** (`0.1 mm / 1.2 mm/s ≈ 0.08 s`). It is the **kinematics of the creep**, not an independent regeneration engine.

Pad–cylinder contact **counts stay bilateral** (`nL` 8–14, `nR` 8–15) for the full 10 s. Zero-g still has churn from residual micro-motion but **zero distal slope**.

**E (contact-point churn as cause) is not supported.** Points migrate because the object is already creeping.

---

## 19. Ideal static pinch (privileged debug only)

From the same airborne hand/finger pose: zero all `qvel`, place cylinder COM on the hand axis (`e_x=e_y=0`), align object frame to `R_hand`, freeze-here, `tau=-18`, 10 s.

**RAW:** slope **1.165 mm/s**, Δez 10 s **11.40 mm** vs natural **1.220 / 12.08**.

**Grasp history is not required.** A constructed zero-velocity centered pinch creeps at the same order.

Figure: `figH_ideal.png`.

**F is ruled out.**

---

## 20. Causal decision tree

| hypothesis | verdict |
|---|---|
| A. Controller / hand drift | **Ruled out for relative creep.** 10 mm arm sag is real but separate; relative Δ`r_h.z` is another 12 mm and vanishes at 0 g with the object load. |
| B. Grip actuator compliance | **Ruled out.** Tendon motor delivers a constant −18 N; fingers move 0.01 mm. |
| C. Contact compliance / regularization | **Supported.** Creep with stationary fingers; linear in g; strongly reduced by shorter solref; force-balanced with rho≪1; live timeconst is a mixed 0.015 s soft constraint; `noslip_iterations=0`. |
| D. Timestep / integration | **Ruled out as primary.** dt/2 and dt/4 leave ~1.17 mm/s. |
| E. Contact-point churn as cause | **Not supported.** Hash churn matches sliding of persistent pairs. |
| F. Initial-state / history | **Ruled out.** Ideal pinch matches. |
| G. Mixed | Only in the weak sense that the **arm also sags 10 mm in world z**; that is not the distal relative motion. |
| H. Unresolved | No. |

**INTERPRETATION / HYPOTHESIS (not a hardware claim):**

The current nominal simulation creeps because **MuJoCo’s regularized condim-4 frictional contacts**, with an **unintentionally mixed solref time-constant of 0.015 s**, permit a quasi-static tangential velocity under the cylinder’s weight even though `|Ft|/(μ Fn) ≈ 0.11`. The grip is a constant tendon force (~9 N per finger), the pads barely move, and ordinary Newton convergence / timestep refinement do not remove the motion.

Do **not** write: “MuJoCo friction is broken,” “a real Panda would not slip,” “solref=0.005 is correct,” or “the problem is fixed.”

Official model unchanged. No recovery work.

---

## Required checklist

1. Minimal test — §1  
2. Actuator semantics — §5  
3. Freeze semantics — §1  
4. Live contact parameters — §8–10  
5. Origin of solref/solimp — §8–10  
6. Baseline 10 s — §3  
7. Force balance — §7  
8. Baseline vs harder — §11  
9. Timestep — §12  
10. Zero-g — §14–15  
11. Ideal pinch — §19  
12. Churn — §18  
13. Decision tree — §20  

Machine-readable rates: `raw/natural_*.npz`, `raw/ideal_baseline.npz`, `raw/metrics.json`.
