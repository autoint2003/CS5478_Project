# Omega_y authority audit (`|ω_y| ≤ 3 rad/s`)

**Question:** why is `omega_y` limited to 3 rad/s, and is that limit actually justified?

**Answer:** the bound is a **software action-normalization** taken from the RULE wrist-align *time* budget, not from the MuJoCo Panda joint-velocity or torque envelope. In this clone it is **uncommitted working-tree code**. At the legal command `ω_y = 3 rad/s` the Cartesian rate tracker already delivers **~2.87–2.99 rad/s** of actual hand spin. Constrained throw speed `ω r_eff ≈ 0.30 m/s` therefore matches the **action bound × measured lever**, not a tracking failure. The XML **does not encode joint velocity limits**, so a hardware-safe increase **cannot** be claimed from this model. A modest **simulation-only** next command of **4.0 rad/s** is defensible as unused *simulated* torque/rate headroom; it is **not** a real-Panda clearance.

No bound was changed. No `ω > 3` command was run. No release-angle sweep, recatch, SAC, reward, observation, detector, or MP4 work.

Raw logs: `results/diagnostics/dynamic_airborne_recatch/raw/omega_authority_audit.json`, `omega_authority_centered.npz`, `omega_authority_early.npz`. Legal-`ω=3` tracking also in `rotation_audit_s1.json`.

---

## 1. Provenance of `omega_y = 3 rad/s`

| Item | Finding |
| --- | --- |
| File | `controllers/residual.py` |
| Lines | 104–106 |
| Configuration | `RECOVERY4D_W_HY_MAX = 3.0  # rad/s, body y of r_des` |
| Consumer | `map_recovery4d`: `w_hy = a[0] * RECOVERY4D_W_HY_MAX`; `tick_4d` clips to this constant |
| Original comment | `recovery4d: independent of 7d dw_max=0.6. RULE 90 deg about hand-y in t_align_max=0.6 s needs ~2.62 rad/s; 3.0 rad/s reaches 90 deg in 0.52 s.` |
| Git / history | `git blame` on those lines: **Not Committed Yet**. `git log -S RECOVERY4D_W_HY_MAX` is empty in this clone. Commit hash: **unknown**. |
| YAML duplicate | `config/nominal.yaml` (also `no_contact.yaml`, `randomized.yaml`) `recovery.w_hy_max: 3.0`. **Not read** by `map_recovery4d`. |
| Related, distinct bound | 7-D residual `dw_max: [0.6, 0.6, 0.6]` in `config/nominal.yaml` (`ResidualLimiter`). Comment states recovery4d is **independent** of this 0.6 rad/s cap. |
| RULE time budget | `config/rule_based_recovery.yaml` `t_align_max: 0.6`; `controllers/rule_based_recovery.py` default `t_align_max: 0.60`. RULE itself commands an **orientation setpoint** (`r_des_align`), not this 3 rad/s scalar. |

Arithmetic in the comment (not invented here):

- `(π/2) / 0.6 s ≈ 2.618 rad/s`
- `90° / 3.0 rad/s ≈ 0.524 s`

**Provenance class: D — chosen heuristically by us.**

Not A (Panda hardware specification): no datasheet citation, no FR3/Panda velocity table in the XML or this comment.

Not B (joint velocity / torque limits): the comment never mentions `q̇` or `τ`. XML torque numbers are 87 / 12 N·m (below); they do not appear in the 3.0 rationale.

Not C (Cartesian-controller stability testing): no documented sweep that selected 3.0 as the largest stable `ω_y`.

Not E: the comment is explicit even though it is heuristic.

---

## 2. MuJoCo joint / actuator limits (this model)

Source of truth: `assets/panda_torque.xml` compiled with `autolimits="true"`, plus runtime `MjModel` dump in `omega_authority_audit.json`.

This project’s arm is **torque-motored**. Menagerie `mujoco_menagerie/franka_emika_panda/panda.xml` is a **position** actuator stack and is **not** what `tick_4d` drives.

### Joint position ranges (runtime = XML)

| Joint | `jnt_range` (rad) | `jnt_limited` |
| --- | --- | --- |
| joint1 | ±2.8973 | 1 |
| joint2 | ±1.7628 | 1 |
| joint3 | ±2.8973 | 1 |
| joint4 | −3.0718 … −0.0698 | 1 |
| joint5 | ±2.8973 | 1 |
| joint6 | −0.0175 … 3.7525 | 1 |
| joint7 | ±2.8973 | 1 |

XML: default class `range="-2.8973 2.8973"`; joint2 / joint4 / joint6 override as above (`panda_torque.xml` lines 9, 144, 159, 178).

### Joint velocity limits

XML contains **no** `velocity=` attributes on arm joints.

Runtime: `MjModel` in this MuJoCo build **does not expose** `jnt_velocity` (`hasattr` false → logged `null`). There is therefore **no encoded `q̇_limit`** to form `q̇ / q̇_limit`.

MuJoCo convention (when the field exists): `jnt_velocity == 0` means unlimited. Either way, **this model does not cap arm `q̇`.**

### Actuator control and force ranges (runtime = XML)

| Actuator | joint / tendon | `ctrlrange` | `forcerange` |
| --- | --- | --- | --- |
| actuator1–4 | joint1–4 | ±87 | ±87 N·m |
| actuator5–7 | joint5–7 | ±12 | ±12 N·m |
| actuator8 | tendon `split` | ±50 | ±50 |

`cartesian_torque` clips arm commands to `ids.ctrl_low/high[:7]`, i.e. the same ±87 / ±12 N·m.

Joints that dominate **hand-y** rotation at the throw pose (Jacobian map at ~60°, centered): **joint5 coefficient ≈ 0.985**. Then joint3 ≈ 0.16, joint4 ≈ 0.15, joint6 ≈ 0.15. Hand `ω_y ≈ q̇_5` at this configuration.

---

## 3. `q̇` utilization at `ω_y_cmd = 3` (clean airborne / high-clearance)

Legal constrained swing, `v_x = v_z = 0`, `τ = −18`, table geom mask off. Same pose family as the centered throw parent.

Peak `|q̇|` over the 0→~125° swing:

| Joint | peak `|q̇|` (rad/s) | `q̇ / q̇_limit` |
| --- | --- | --- |
| 1 | 0.349 | **N/A** (no XML velocity limit) |
| 2 | 0.038 | N/A |
| 3 | 0.574 | N/A |
| 4 | 0.918 | N/A |
| **5** | **2.807** | N/A |
| 6 | 0.890 | N/A |
| 7 | 0.448 | N/A |

At ~60° (throw-relevant):

`q̇ ≈ [0.056, 0.000, −0.566, −0.491, 2.791, −0.519, −0.426]`

`joint_omega_y_share ≈ [0.002, 0, 0.089, 0.076, 2.749, 0.076, −0.006]` (rad/s about commanded `y_des`).

**Are we near a robot/controller velocity limit at `ω = 3`?** Not in this XML: there is no `q̇` cap, so joint5 at 2.81 rad/s is **not saturating a model velocity limit**. It *is* the joint that must speed up if `ω_y` is raised (≈ 1:1 with hand-y).

---

## 4. Torque utilization at `ω_y_cmd = 3`

Unclipped Cartesian+null torque vs `forcerange`. **Zero steps** with `|τ_unclipped| ≥ |τ_limit|`. Applied `ctrl[:7]` matches unclipped to ~0.1 N·m.

Peak `|τ| / τ_limit` over the swing (often at **t0 hold**, not peak spin):

| Joint | limit (N·m) | peak util | at ~60° util | at ~60° τ (N·m) |
| --- | --- | --- | --- | --- |
| 1 | 87 | 0.028 | 0.002 | 0.16 |
| 2 | 87 | 0.268 | 0.260 | −22.7 |
| 3 | 87 | 0.041 | 0.002 | 0.18 |
| 4 | 87 | 0.257 | 0.244 | 21.3 |
| **5** | **12** | **0.641** | **0.398** | **4.78** |
| 6 | 12 | 0.219 | 0.024 | 0.29 |
| 7 | 12 | 0.044 | 0.034 | −0.40 |

Peak wrist util 0.64 is **~7.7 N·m on joint5 at the static parent**, i.e. gravity / bias / hold, not the 3 rad/s cruise. During rotation, joint5 sits near **40%** of 12 N·m.

**Saturation: none.** At `ω_y = 3` the simulated torque stack is **not** at a hard actuator limit.

---

## 5. Tracking quality at legal `ω_y_cmd = 3`

Command: `w_world = r_des @ [0, 3, 0]`. Actual axis rate: `w_hand · y_des`.

### This audit (high-clearance centered, 2 ms samples)

| Angle (°) | `ω_y` cmd | `ω` about `y_des` | `‖w_hand‖` | SO(3) err (°) | `‖p_des − p_hand‖` (mm) |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0.004 | 0.056 | 0.00 | 0.0 |
| 15 | 3 | 2.655 | 2.657 | 9.57 | 11.3 |
| 30 | 3 | 2.877 | 2.879 | 10.72 | 15.9 |
| 45 | 3 | 2.951 | 2.952 | 11.17 | 15.5 |
| 60 | 3 | 2.987 | 2.988 | 11.33 | 12.8 |
| 90 | 3 | 2.990 | 2.991 | 11.36 | 12.8 |
| 120 | 3 | 2.922 | 2.924 | 11.71 | 22.9 |

Settled (cmd on): median axis rate **2.947 rad/s**, max **2.999 rad/s**.

### Prior throw log (`rotation_audit_s1.json`, world `w_hand`)

| mark | `w_hand` world | `‖w_hand‖` |
| --- | --- | --- |
| 30° | `[2.874, −0.019, 0.015]` | 2.874 |
| 60° | `[2.986, −0.054, 0.054]` | 2.987 |
| 90° | `[2.989, −0.077, 0.085]` | 2.992 |
| 120° | `[2.923, −0.093, 0.073]` | 2.926 |

**Verified:** actual hand rate is **2.87–2.99 rad/s** after the first ~30°. The previous report is consistent with raw logs.

SO(3) error ~11° is a **phase lag** of `r_cur` behind the integrated `r_des` (PD `kp_ori=16`, `kd_ori=2.4`). It does **not** reduce cruise `|ω|` below ~3. Position error grows to ~1.4 cm median / 2.5 cm max with `v_cmd = 0` (frozen `p_des`) during a large wrist motion — Cartesian coupling, not a missing 3 rad/s of spin.

**Statement:** the Cartesian controller is **not** the primary reason the throw velocity is capped near **0.3 m/s** at the current bound. The hand already rotates at the bound; `v ~ ω r_eff` with `r_eff ≈ 0.099 m` is kinematics of that bound.

---

## 6. CENTER-6 EARLY large-angle utilization

Same legal `ω_y = 3`, restore `snap_early.pkl`, table geom off, swing to ~125°.

| Quantity | CENTER-6 EARLY | Centered high-clearance |
| --- | --- | --- |
| axis `ω` median / max | 2.940 / 2.999 | 2.947 / 2.999 |
| peak `|q̇_5|` (rad/s) | 2.791 | 2.807 |
| peak `|q̇_6|` | 1.081 | 0.890 |
| peak τ util (joint5) | 0.644 | 0.641 |
| τ sat steps | 0 | 0 |
| SO(3) max (°) | 12.06 | 11.84 |
| `‖p_err‖` max (mm) | 23.6 | 24.7 |
| `cond(J)` max | 10.52 | 11.05 |
| `σ_min(J_ω)` at 60° | 1.018 | 1.019 |
| joint-limit proximity max (0 = center, 1 = stop) | 0.700 at ~125° | 0.706 at ~125° |

At 60° EARLY: `q̇_5 = 2.771`, τ util j5 = 0.39, prox = 0.35 (joint5). At 120° prox ≈ 0.67–0.68, still **inside** range, not on the stop.

Would a higher `ω_y` hit, in **this simulated envelope**:

| Mechanism | At `ω = 3` | Implication for a small increase |
| --- | --- | --- |
| XML joint velocity limit | none | cannot be the reason to keep 3 |
| Torque saturation | none; cruise ~40% on j5 | linear scale 3→4 still likely &lt; 1 unless unmodeled accel spikes |
| Singularity / conditioning | `cond(J) ~ 9–11`, `σ_min(J_ω) ~ 1.02` | not a wrist singularity |
| Cartesian **rate** tracking | cruise 2.99 / 3.00 | not the throw cap |
| Cartesian **pose** lag | ~11° ori, ~1–2 cm pos | may grow with `ω`; not a hard limit in this audit |
| Joint-limit proximity | 0.70 at 125° | large-angle geometry, not `ω` magnitude per se |

---

## 7. Idealized `ω r` throw-scale table

Measured effective lever from the throw program: **`r_eff ≈ 0.099 m`**.

These are **IDEALIZED UPPER-SCALE ESTIMATES**, not simulation results. They assume the entire useful vertical component equals `ω r_eff` and is preserved at release. `Δz = v_z² / (2g)`, `g = 9.81 m/s²`.

| `ω` (rad/s) | `ω r_eff` (m/s) | ideal `Δz` (m) | ideal `Δz` (mm) |
| --- | --- | --- | --- |
| 3 | 0.297 | 0.00450 | 4.5 |
| 4 | 0.396 | 0.00799 | 8.0 |
| 5 | 0.495 | 0.01249 | 12.5 |
| 6 | 0.594 | 0.01798 | 18.0 |

Measured constrained peaks at legal `ω = 3` (centered 0.300 m/s, offset 0.309 m/s) already sit on the 3 rad/s row. That is expected if the object is carried near `ω × r`.

---

## 8. Simulation vs hardware

**Simulation controller authority (this repo):** torque motors ±87 / ±12 N·m, **no** joint-velocity inequality, Cartesian PD as in `controllers/jacobian_controller.py`. At `ω_y = 3` the rate loop is essentially at command, actuators are not clipped, Jacobian is well-conditioned.

**Real Panda safe dynamic authority:** **not encoded.** The XML copies **torque** magnitudes typical of Panda wrist/shoulder motors and **position** ranges; it does **not** copy a manufacturer joint-velocity table, collision model of a real cell, or Franka Cartesian-velocity safety. `MuJoCo feasibility ≠ real Panda hardware safety`.

**Therefore:** a larger `ω_y` **may be valid for simulation authority exploration**, but **cannot yet be claimed hardware-realistic**.

---

## 9. Conclusion

**Why 3 rad/s?** Because recovery4d `a ∈ [−1, 1]` was scaled so that `|a0| = 1` can rotate ~90° inside RULE `t_align_max = 0.6 s` (`~2.62` rounded to `3.0`). That is action design for **align-time coverage**, independent of 7-D `dw_max = 0.6`, independent of throw lever `r_eff`, and independent of XML `q̇` (which does not exist).

**Is that physically/controller justified as a throw-speed cap?**

- **Not** as a MuJoCo joint-velocity limit (none).
- **Not** as torque saturation (no clip; cruise ~40% on the wrist motor).
- **Not** as Cartesian rate tracking failure (actual `ω ≈ 3`).
- **Yes** as a **documented heuristic software bound** (class **D**).
- **Hardware:** class **C** — the model does not encode enough information to justify a **hardware-safe** increase.

Relative to the **simulated** envelope, 3 rad/s is **substantially below torque saturation** and is an **arbitrary conservative action normalization** (class **B** for simulation only). It is **not** “already at a physical XML/controller limit” (class A is rejected for this platform as modeled).

---

## 10. One recommended next experimental `ω` (not run)

Justified **only** as unused **simulation** torque/rate envelope plus the kinematic identity `v ~ ω r`, **not** because “3 failed to throw high enough.”

**Next command to test later: `ω_y = 4.0 rad/s` (one value).**

Reasons to keep the step small:

- joint5 is ~1:1 with `ω_y`; 3→4 scales cruise `q̇_5` from ~2.81 toward ~3.7 rad/s, still unlimited in XML but no longer a tiny perturbation;
- cruise τ util ~0.40 on the 12 N·m wrist motor still has room **if** transients stay similar;
- 4.0 is +33%, not a jump to 6–10;
- idealized `ω r` goes 0.30 → 0.40 m/s (still centimetre-scale `Δz` even in the upper-bound ballistic formula).

Do **not** treat 4.0 as hardware-cleared. The next experiment (separate task) should measure whether `v_obj,z` at `FIRST_BOTH_OFF` and ballistic excursion actually scale, with the same legal open path, **without** retuning recatch.
