# Ballistic cliff audit (ZERO only)

**APPARENT CAPTURE-LOSS CLIFF IN CURRENT SWEEP; PHYSICAL INTERPRETATION UNVERIFIED.**

Physics, noslip, friction, ball mass, and the nominal controller were not changed.
No recovery, recatch, SAC, or heuristic work was run.

## 14. USER VISUAL OBSERVATION: PENDING

Videos exist. They have not been treated as watched. Do not treat this file as a visual confirmation.

---

## 1. Current sweep implementation audit

Source: `training/demo_ballistic_impact.py` (`BallisticSim`, `run_episode`, `launch_state`, `classify`).

Frozen physics for this audit:

- `noslip_iterations = 1`
- XML ball mass `0.05 kg`, cylinder `0.20 kg`
- pair sliding μ = `1.0`
- timestep `dt = 0.002 s`
- diagnostic scene `assets/scene_ballistic_impact.xml` (ball `contype=2`, cylinder `3`; ball does not collide with fingers/table)

Task: nominal GraspFSM `approach → descend → close → lift`. Disturbance is injected only after lift has started (`captured`, `obj_z ≥ Z_AIR`, hand rise ≥ `0.05 m`). After `p_des.z` reaches `grasp_z + lift_offset_z`, the command is `v=0` hold for ≥12 s.

The original episode loop **stops 2 s after `dropped(...)` first returns True**. That is why the prior 2.967 JSON could show `t_drop` at +6 ms and still report `end_nL=6`, `end_nR=5`, `end_obj_z≈0.60`: the run was cut while the object was still in the air with contacts already able to return.

This audit re-simulates the same launch law **without** stopping on a 1-step both-off. Termination is table contact + 2 s, or lift-done + 12 s hold.

---

## 2. Ball aiming / release implementation

Read from `launch_state` (not comments):

```396:422:training/demo_ballistic_impact.py
def launch_state(sim, v_hit: float) -> dict:
    u = incoming_hand_x(sim)
    po = np.array(sim.data.xpos[sim.ids.object_body], float)
    ...
    p_hit = po - u * r_surf
    p_launch = po - u * float(D_LAUNCH)
    dist_hit = float(np.linalg.norm(p_hit - p_launch))
    T = dist_hit / max(float(v_hit), 1e-9)
    v0 = (p_hit - p_launch) / T - 0.5 * g * T
```

Classification of the original aimer:

| Option | Used? |
|---|---|
| A. fixed world-space point | no |
| **B. cylinder COM at release** | **yes** (`p_hit = p_obj(release) − u (r_cyl+r_ball)`) |
| C. predicted future cylinder pose | no |
| D. hand-frame target other than current +hand-x | no; direction is `R_hand[:,0]` at release |
| E. other | gravity-compensated intercept of that **stale** COM |

Launch distance is fixed: `D_LAUNCH = 0.18 m`. Time of flight is therefore `T ≈ 0.18 / v_hit` (minus the surface offset). Faster balls fly for less time. The cylinder is still lifting, so impact pose is **not** independent of speed.

Guide: weld is turned off at release; `apply_ball_free_state` places the free ball at `p_launch` with `v0`. Collision is physical. No teleport at impact.

---

## 3. Impact timing vs speed

### Original aimer (this audit, same `launch_state`)

All cases share `t_release = 2.748 s`. `dt = 2 ms` quantizes TOF.

| v (m/s) | t_impact (s) | TOF (s) | t_lift0 (s) |
|---:|---:|---:|---:|
| 2.700 | 2.806 | 0.058 | 1.922 |
| 2.850 | 2.802 | 0.054 | 1.922 |
| 2.950 | 2.800 | 0.052 | 1.922 |
| 2.967 | 2.800 | 0.052 | 1.922 |
| 2.983 | 2.800 | 0.052 | 1.922 |
| 3.000 | 2.800 | 0.052 | 1.922 |
| 3.017 | 2.800 | 0.052 | 1.922 |
| 3.033 | 2.800 | 0.052 | 1.922 |
| 3.050 | 2.800 | 0.052 | 1.922 |

From 2.95 through 3.05, **impact time is identical** (same 2 ms bin). From 2.70 to 2.95, TOF still drops by 6 ms.

Existing stored sweep (`cliff_audit/existing_sweep_events.json`) is more extreme at low speed: e.g. v=0.80 TOF=0.312 s, v=1.20 TOF=0.126 s. That stored family is **not** a one-dimensional impact-severity sweep.

### Matched aimer (`T_flight = 0.10 s` fixed)

| v (m/s) | t_impact (s) | TOF (s) |
|---:|---:|---:|
| 2.6–3.1 (all six) | 2.852 | 0.104 |

---

## 4. Impact point vs speed

Cylinder-relative first ball–cylinder contact. Axial z is along the cylinder axis. `near_end` threshold was `|z| > 22 mm`. None of these hits are end-clipping.

### Original aimer

| v | pos_w (m) | pos_cyl (mm) | axial z (mm) | radial (mm) | obj_z at impact (m) | hand_z (m) |
|---:|---|---|---:|---:|---:|---:|
| 2.70 | (0.499, −0.0155, 0.480) | (0.02, −15.91, −5.64) | −5.64 | 15.91 | 0.48594 | 0.58226 |
| 2.85 | (0.499, −0.0170, 0.481) | (0.00, −17.41, −4.94) | −4.94 | 17.41 | 0.48561 | 0.58194 |
| 2.95 | (0.499, −0.0174, 0.481) | (0.00, −17.76, −4.69) | −4.69 | 17.76 | 0.48544 | 0.58177 |
| 2.967 | (0.499, −0.0170, 0.481) | (0.00, −17.34, −4.76) | −4.76 | 17.34 | 0.48544 | 0.58177 |
| 2.983 | (0.499, −0.0166, 0.481) | (0.00, −16.94, −4.83) | −4.83 | 16.94 | 0.48544 | 0.58177 |
| 3.00 | (0.499, −0.0161, 0.480) | (0.00, −16.51, −4.90) | −4.90 | 16.51 | 0.48544 | 0.58177 |
| 3.05 | (0.499, −0.0149, 0.480) | (0.01, −15.26, −5.11) | −5.11 | 15.26 | 0.48544 | 0.58177 |

Hits are on the cylinder **side**, slightly **below center** (~5 mm), not an end. Incoming direction is +hand-x (`v_ball_h.x ≈ v_hit`).

Even when `t_impact` and hand/object **COM pose are identical** (2.95–3.05), the contact **radial coordinate still moves ~2.5 mm**. A speed change is not a pure impulse change.

### Matched aimer

All six speeds: `p_obj_imp ≈ (0.49830, 0.00034, 0.48968)`, `p_hand_imp ≈ (0.49628, 0.00010, 0.58601)`, axial z ≈ **−2.07 mm**, radial 15.40 → 14.90 mm. Geometry is much closer to constant; residual radial drift is ~0.5 mm.

Plots: `cliff_audit/figures/geom_vs_speed_original.png`, `geom_vs_speed_matched.png`.

---

## 5. Pre-impact hand/object state vs speed

Release trigger is the same for every speed (`t_release = 2.748 s`).

Original aimer, at impact:

- FSM = `lift` in every case
- `v_cmd = [0, 0, 0.08]` (still climbing)
- `ctrl[7] = −18` (close command unchanged)
- `p_des` continues integrating; Δp_des from release to impact equals `0.08 × TOF` (4.64 mm at 2.70, 4.16 mm at 2.95+)
- `r_des` not reset
- no freeze, no snapshot reload

Because TOF shrinks with v, **pre-impact lift height is speed-dependent** until TOF falls into the same 2 ms bin. From 2.70 to 2.95, `obj_z_imp` drops 0.49 mm (hand has risen less when the faster ball arrives — wait: faster ball arrives **earlier**, so the hand is **lower**. 2.70 impact is 6 ms later, so hand_z is 0.49 mm **higher**. That matches `v_z=0.08`).

Matched aimer: identical impact COM pose for 2.6–3.1; `p_des` delta release→impact = 8.32 mm = `0.08 × 0.104`.

**Do not call the original family a speed-only severity sweep.**

---

## 6. Exact contact-loss classifier

`training/demo_grav_reposition_recovery.py`:

```python
def dropped(sim, m) -> bool:
    return bool(sim.dropped() or float(m["obj_z"]) < TABLE_DROP or (m["nL"] == 0 and m["nR"] == 0))
```

`sim.dropped()` is `object_dropped`: lift phase and (xy from hand > 0.18 **or** (`t_phase > 0.4` and z below table+half+1 cm)).

`classify` then does:

```python
if td is not None and (td - ti) < 0.10:
    return "IMMEDIATE_LOSS"
```

So **IMMEDIATE_LOSS** means: the first time `dropped(...)` is True occurs within 100 ms of first ball–cylinder contact.

A **single 2 ms step with nL=nR=0** is sufficient. That is not:

- first one-finger loss
- sustained both-off
- table contact
- xy escape
- physical object leaving the hand for the rest of the episode

The original 2.967 run also **stopped 2 s after that blip**, so the stored JSON cannot be used as a drop outcome.

---

## 7. 2.967 m/s step-by-step event trace

Files:

- `cliff_audit/hires_v2p967.csv` (every 2 ms from t_impact−20 ms to +200 ms)
- `cliff_audit/hires_v2p967.npz`

Raw times for this audit episode (not I0–I3 labels):

| event | t (s) | Δt from impact |
|---|---:|---:|
| t_release | 2.748 | −0.052 |
| t_impact | 2.800 | 0 |
| t_first_unilateral | 2.806 | +0.006 |
| t_first_both_off | 2.806 | +0.006 |
| t_bilateral_restored | 2.808 | +0.008 |
| duration both-off | 0.002 s (one step) | |
| t_table / t_drop physical | none | |
| t_xy_escape | none | |
| t_lift_done | 4.244 | +1.444 |
| t_end (12 s hold) | 16.244 | |
| legacy `dropped` fire | 2.806 | +0.006 |

Finger contacts around impact (`hires_v2p967.csv`):

| t | nL | nR | Fn L | Fn R | rho_max | aperture | v_cmd_z | fsm |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2.780–2.798 | 7 | 6 | 9.07 | 9.08 | ~0.25 | 0.01765 | 0.08 | lift |
| **2.800 impact** | 7 | 6 | **13.95** | **14.49** | **1.00** | 0.01777 | 0.08 | lift |
| 2.802 | 4 | 4 | 9.78 | 9.83 | 0.92 | 0.01788 | 0.08 | lift |
| 2.804 | 4 | 4 | 8.58 | 8.70 | 0.57 | 0.01795 | 0.08 | lift |
| **2.806** | **0** | **0** | 0 | 0 | nan | 0.01772 | 0.08 | lift |
| **2.808 restored** | 2 | 3 | 10.65 | 10.64 | 0.53 | 0.01755 | 0.08 | lift |
| 2.810 | 4 | 4 | 10.26 | 10.38 | 0.54 | 0.01742 | 0.08 | lift |

What “loss in 6 ms” **physically is**, in this implementation:

1. Ball–cylinder contact at 2.800 while both fingers still report contacts.
2. Fn spike, ρ hits 1, aperture opens ~0.3 mm.
3. At +6 ms, finger–object contact counts go to zero for **one solver step**.
4. At +8 ms, contacts are already back (2/3, then 4/4).
5. Gripper command stays `ctrl[7]=−18`. Lift command stays `v_z=0.08`. FSM never leaves `lift`.
6. Object remains at z≈0.48–0.60 throughout; 12 s high hold completes with end_nL=6, end_nR=6, obj_z≈0.605.

That is **not** equivalent to irrecoverable grasp loss.

---

## 8. Raw post-impact continuum (original aimer)

Samples at +20 / +50 / +100 / +200 ms. `rh.x` in mm, `|v_rel_h|` in m/s.

| v | +20 ms rh.x | +200 ms rh.x | +20 \|v_rel\| | +200 \|v_rel\| | +20 nL/nR | +200 nL/nR | table |
|---:|---:|---:|---:|---:|---|---|---|
| 2.70 | 1.10 | 1.08 | 0.013 | 0.002 | 6/7 | 6/4 | no |
| 2.85 | 1.06 | 0.98 | 0.014 | 0.002 | 5/5 | 6/4 | no |
| 2.95 | 1.19 | 1.13 | 0.015 | 0.002 | 9/9 | 7/5 | no |
| 2.967 | 1.33 | 1.31 | 0.016 | 0.002 | 5/4 | 4/4 | no |
| 2.983 | 1.39 | 1.37 | 0.017 | 0.002 | 4/4 | 4/4 | no |
| 3.00 | 1.64 | 1.64 | 0.019 | 0.002 | 5/4 | 4/4 | no |
| 3.05 | (see json) | ~1.8 | ~0.021 | 0.002 | contacts present | contacts present | no |

Offset grows slowly with speed; relative velocity damps by 50–100 ms. Every original-boundary case in this audit is **B_stable_disturbed** under the descriptive (non-policy) tagging, and all complete the 12 s hold.

The previous DISTURBED vs IMMEDIATE split at 2.95 vs 2.967 is the 2 ms both-off **blip**, not a change from captured to ballistic ejection.

Time series: `cliff_audit/figures/post_impact_original.png`.

---

## 9. Matched-impact-geometry construction

Implemented in `launch_state_matched` (`training/audit_ballistic_cliff.py`).

Same release trigger as the original task. At release:

1. `u = R_hand[:,0]`
2. `T = 0.10 s` **fixed** (not `dist/v`)
3. `p_obj_pred = p_obj + v_obj T` (constant-velocity prediction of the lifting cylinder)
4. `p_hit = p_obj_pred − u (r_cyl + r_ball)` — lateral +hand-x face, mid-body
5. `p_launch = p_hit − u (v_hit T)` so the along-track fly-in distance scales with speed
6. `v0 = (p_hit − p_launch)/T − 0.5 g T`
7. weld off; free flight; physical collision; **no teleport**

This is still a free-flight ballistic hit. It only holds **impact time / lift phase / COM pose / contact recipe** approximately constant so impulse can be separated from TOF geometry.

Chosen geometry: +hand-x incoming, contact ~2 mm below cylinder center, radial ~15 mm, not an end hit. Chosen for a clean lateral family, **not** for recovery success.

---

## 10. Matched local sweep

ZERO only. Speeds 2.6, 2.7, 2.8, 2.9, 3.0, 3.1 m/s. All: `t_release=2.748`, `t_impact=2.852`, TOF=0.104 s, 12 s hold, no table.

| v | t_ul | t_both | t_restore | both-off | end nL/nR | obj_z end | legacy fire |
|---:|---|---|---|---:|---|---:|---|
| 2.6 | 2.858 | 2.858 | 2.860 | 0.002 | 6/6 | 0.605 | yes (+6 ms) |
| 2.7 | 2.856 | — | — | 0 | 6/6 | 0.605 | — |
| 2.8 | 2.856 | — | — | 0 | 6/6 | 0.605 | — |
| 2.9 | 2.856 | 2.856 | 2.858 | 0.002 | 6/6 | 0.605 | yes (+4 ms) |
| 3.0 | 2.856 | 2.856 | 2.858 | 0.002 | 6/6 | 0.605 | yes (+4 ms) |
| 3.1 | 2.856 | 2.856 | 2.858 | 0.002 | 6/6 | 0.605 | yes (+4 ms) |

If the **legacy** classifier were applied here, several captured holds would still be labelled IMMEDIATE_LOSS.

Post-impact continuum (matched):

| v | +20 rh.x mm | +200 rh.x mm | +20 \|v_rel\| | +200 \|v_rel\| | +20 nL/nR |
|---:|---:|---:|---:|---:|---|
| 2.6 | 1.33 | 1.29 | 0.016 | 0.002 | 6/5 |
| 2.7 | 1.62 | 1.61 | 0.018 | 0.002 | 8/6 |
| 2.8 | 1.70 | 1.69 | 0.019 | 0.002 | 6/4 |
| 2.9 | 1.92 | 1.94 | 0.021 | 0.002 | 4/4 |
| 3.0 | 2.01 | 2.06 | 0.023 | 0.002 | 2/4 |
| 3.1 | 2.15 | 2.21 | 0.024 | 0.002 | 2/3 |

This is a **small offset continuum**, not a capture/ejection cliff.

Videos: `videos/matched/v2p600.mp4` … `v3p100.mp4`.

---

## 11. Progressive-instability search

Descriptive modes (not policy labels):

- **A** impact transient that damps
- **B** residual motion → ~0, remains captured
- **C** remains captured initially, displacement/rotation keeps growing
- **D** ballistic ejection immediately after impact

Original boundary 2.70–3.05: all **B** (relative velocity ~0 by 100 ms; rh.x plateaus; 12 s hold; no table).

Matched 2.6–3.1: all **B**. Residual `|v_rel_h|` falls from ~0.02 m/s at +20 ms to ~0.002 m/s at +200 ms and stays small for the rest of the hold. `rh.x` does not run away. Cylinder tilt (via `cyl_h` / `w_rel`) damps. Contacts remain bilateral after the possible 2 ms blip.

**C was not observed in this local window.** That is a result, not a requirement. This window also does not contain a physical D (ejection). A 1-step both-off still occurs at several speeds and would still trip the **legacy** label.

Whether a progressive-instability regime exists at higher matched impulse, different impact latitude, or different grasp preload is **outside this audit**.

---

## 12. Video paths and commands

```text
python training/audit_ballistic_cliff.py
```

Fixed oblique camera (not per-frame tracking): lookat `(0.50, −0.02, 0.51)`, distance `0.88`, azimuth `118°`, elevation `−16°`. Intended to show both fingers, cylinder axis, ball, impact, tilt, and whether the cylinder stays between the fingers.

Playback: task speed 0.70× until impact−100 ms; **0.08× from −100 ms to +500 ms** around first ball–cylinder contact; then 1.10× through ≥12 s hold (or 2 s after table). Overlay phase is informational; it does **not** stop the sim.

### Original aiming (boundary)

`results/diagnostics/ballistic_impact_noslip1/videos/boundary/`

| file | v (m/s) |
|---|---:|
| `v2p700.mp4` / `v2p70.mp4` | 2.70 |
| `v2p850.mp4` / `v2p85.mp4` | 2.85 (nearest available to 2.90) |
| `v2p950.mp4` / `v2p95.mp4` | 2.95 |
| `v2p967.mp4` | 2.967 |
| `v2p983.mp4` / `v2p98.mp4` | 2.983 |
| `v3p000.mp4` / `v3p00.mp4` | 3.00 |
| `v3p017.mp4` / `v3p02.mp4` | 3.017 |
| `v3p033.mp4` / `v3p03.mp4` | 3.033 |
| `v3p050.mp4` / `v3p05.mp4` | 3.05 |
| `captured_vs_reported_loss.mp4` | 2.95 (left) vs 2.967 (right), aligned on first contact |

There is no stored original case at exactly 2.90; 2.85 was used.

### Matched aiming

`results/diagnostics/ballistic_impact_noslip1/videos/matched/`

`v2p600.mp4`, `v2p700.mp4`, `v2p800.mp4`, `v2p900.mp4`, `v3p000.mp4`, `v3p100.mp4`.

JSON/NPZ: `results/diagnostics/ballistic_impact_noslip1/cliff_audit/`.

---

## 13. Unresolved discrepancies

- Original stored sweep **classifies** a cliff at ~2.967 because `nL=nR=0` for one step. When the same physics is allowed to continue, contacts return and the 12 s hold succeeds. The classifier and the physical trajectory disagree.
- Original aiming is **B (stale COM)** plus `T=dist/v`. TOF and therefore lift pose change with speed; even inside one 2 ms TOF bin the contact radial point still moves millimetres. Non-monotonic legacy labels in the stored sweep can be geometry+classifier, not a 1-D impulse cliff.
- Matched geometry removes most of that confound in 2.6–3.1 m/s and still shows **no ejection and no progressive C**. That does not prove C cannot exist somewhere else.
- `nL`/`nR` are contact **counts**, not a geometric “finger fully off” predicate. A 0/0 step during an impulse is a solver/contact-report event; it needs the videos.
- Camera and slow-mo are provided specifically because numerical summaries were insufficient. **Visual inspection is still required.**
- This document does **not** claim a physical capture-loss cliff.

---

## 14. USER VISUAL OBSERVATION: PENDING

No recatch, recovery, or heuristic tuning follows from this audit.
