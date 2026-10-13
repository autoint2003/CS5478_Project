# Ballistic recovery transfer (CENTER-6)

**Correction (do not stop at this file):** Section 9 small-angle (±13°) wrist pulses are **insufficient**. They do **not** show that gravitational repositioning fails on CENTER-6. Large-angle transfer is in `LARGE_ANGLE_GRAVITY_TRANSFER.md`.

Frozen diagnostic. **Not** a learned policy, **not** an optimized heuristic.

**CENTER-6: ballistic translational progressive failure.**

## USER VISUAL OBSERVATION (ZERO)

Confirmed by the user:

- ball hits the cylinder normally
- cylinder stays between the fingers after impact
- slow lateral drift through the grasp
- eventual escape and fall
- **no meaningful impact-induced rotation**

Raw tilt after the impact transient is **~0.6°** (axis vs hand), then **~0.2°** at EARLY. That matches the visual: do not call this translation+tilt.

---

## 1. Frozen CENTER-6 benchmark

| item | value |
|---|---|
| geometry | matched +hand-x CENTER |
| v | 6.0 m/s |
| m_ball | 0.05 kg |
| noslip | 1 |
| pair μ | 1.0 |

Impact speed, mass, point, and timing are **not** changed. UPPER/LOWER were **not** run.

---

## 2. User visual validation

See above. Staging below uses privileged `r_h`, `|v_rel|`, nL/nR, Fn, aperture, z — not `D_t` or `success_obs`.

---

## 3. ZERO stage timeline

| stage | t (s) | Δt from impact | notes |
|---|---:|---:|---|
| T_IMPACT | 2.852 | 0 | first ball–cylinder contact; n=7/6 |
| bilateral restore | 2.860 | +8 ms | 2 ms both-off blip only |
| T_TRANSIENT_END | 2.914 | +62 ms | \|v_rel\|~6 mm/s; n=2/2; rh.x~10.5 mm; tilt~0.6° |
| T_EARLY_DRIFT | 3.322 | +0.47 s | still captured n=7/4; rh.x~10.9 mm; \|v_rel\|<1 mm/s |
| T_lift_done | 4.244 | +1.39 s | nominal lift finishes **while captured** |
| T_MID_DRIFT | 4.872 | +2.02 s | n=2/4; rh.x~12.5 mm; still between fingers |
| T_LATE | 5.472 | +2.62 s | still n>0 both sides; offset growing |
| T_RUNAWAY | 5.822 | +2.97 s | rh.x≳16 mm and \|v_rel\| jumps |
| T_DROP | 6.132 | +3.28 s | table contact |

Natural ZERO is already `tau=−18` through this entire story. “Secure only” is the baseline trajectory, not a recovery.

---

## 4. EARLY / MID / LATE snapshots

| snap | t | file |
|---|---:|---|
| EARLY | 3.322 | `raw/snap_early.pkl` |
| MID | 4.872 | `raw/snap_mid.pkl` |
| LATE | 5.472 | `raw/snap_late.pkl` |

Each pickle is full `GraspSim.snapshot` plus `eq_active` (weld off), mocap, `z_tgt`, FSM clocks, `p_des` / `r_des` / `v_cmd` / `w_cmd`, `qacc_warmstart`.

---

## 5. Restore audit

ZERO continuation from the snapshot reproduces drift → runaway → table.

| snap | qpos | p_des | time | FSM | v_cmd | weld | table | max rh.x error |
|---|---|---|---|---|---|---|---:|---:|
| EARLY | exact | exact | exact | lift | +z 0.08 | off | 6.132 | **0.013 mm** (to +2 s) |
| MID | exact | exact | exact | lift | 0 (already at hold z) | off | 6.132 | 0.015 mm to +0.5 s; **0.20 mm at +1 s** (runaway onset) |

EARLY restore is numerically tight. MID +1 s is already in the chaotic escape; table time still matches 6.132 s.

---

## 6. Single-action authority (legal 4D)

Action: `(omega_y, v_x, v_z, tau_g)` via `map_recovery4d` (hand-x, world-z, r_des y). Bounds: `|ω_y|≤3`, `|v_x|≤0.08`, `|v_z|≤0.08`, `tau∈[−1,−18]`.

Pulses from EARLY and MID. `drhx` = change in `r_h.x` over the pulse (positive = more of the failure offset).

### Baseline: secure only (`tau=−18`, v=0)

| stage | dur | drhx | contacts | table |
|---|---:|---:|---|---|
| EARLY | 0.40 s | +0.28 mm | kept | no |
| MID | 0.40 s | +0.64 mm | kept | no |

Same sign as ZERO creep. **Maintaining secure grip does not arrest progressive failure** (ZERO already was `tau=−18`).

---

## 7. Grip-state dependence

| stage | tau | drhx | contacts | table |
|---|---:|---:|---|---|
| EARLY | −8 | +0.17 mm | kept | no |
| EARLY | −3 | +14.8 mm | lost | **3.776 s** |
| EARLY | −1 | +10.4 mm | lost | **3.502 s** |
| MID | −8 | +0.41 mm | kept | no |
| MID | −3 | +35.8 mm | lost | **5.174 s** |
| MID | −1 | +18.6 mm | lost | **5.088 s** |

`tau=−8` behaves like secure (no useful relative slide). `tau=−3` and `−1` dump the cylinder within ~0.2–0.5 s.

**No intermediate grip regime was found that allows controlled relative repositioning without immediate loss** on CENTER-6 at EARLY/MID.

The teleport construction used weakened grip on a different IC (offset/tilt with a still-workable contact). That **does not** transfer to this already-creeping 2/4-contact grasp.

---

## 8. Hand-x authority

| stage | v_x | dur | drhx | contacts | hand moved |
|---|---:|---:|---:|---|---|
| EARLY | +0.06 | 0.30 s | +0.21 mm | kept | ~18 mm along hand-x |
| EARLY | −0.06 | 0.30 s | +0.22 mm | kept | opposite |
| MID | +0.06 | 0.30 s | +0.46 mm | kept | ~18 mm |
| MID | −0.06 | 0.30 s | +0.47 mm | kept | opposite |

`r_h.x` is **insensitive** to legal hand-x at `tau=−18`: the object **co-moves** with the hand. Translating the pair does not recenter the cylinder in the fingers.

That is the opposite of a useful “follow the object to kill relative error” effect **while the grip is secure**.

---

## 9. Wrist authority

| stage | ω_y | dur | Δwrist | drhx | g_h change | contacts |
|---|---:|---:|---:|---:|---|---|
| EARLY | +1.2 | 0.25 s | +12.9° | +0.14 mm | small | kept |
| EARLY | −1.2 | 0.25 s | −12.8° | +0.23 mm | small | kept |
| MID | +1.2 | 0.25 s | +12.7° | +0.38 mm | small | kept |
| MID | −1.2 | 0.25 s | −12.7° | +0.42 mm | small | kept |

Wrist **does** rotate the hand (~13°). It does **not** reverse lateral creep. `+ω_y` slightly slows `r_h.x` growth vs secure (0.14 vs 0.28 mm) but does not recenter. No angle search was performed.

CENTER-6 has almost no object tilt, so wrist-as-gravity-reprojection is weakly motivated here; the data agree.

---

## 10. v_z and small combinations

`±v_z` (0.04 m/s, 0.30 s): `drhx` indistinguishable from secure (~0.21 mm EARLY, ~0.46 mm MID). Contacts kept. Not used as hidden lift assist.

Combinations `follow-x + secure` copy the vx pulses: still no `r_h.x` reduction. Weak-grip + follow-x was **not** run as a “best tau” search; weak grip already loses the object alone.

No existence plan was selected (`raw/recovery_plan.json`: `none`).

---

## 11. Comparison with the teleport mechanism

Teleport (idealized IC) established:

> secure → wrist reorientation → weakened grip → gravitational relative motion → secure again.

CENTER-6 probes:

| mechanism | transfers? |
|---|---|
| **A. Same (wrist + weak grip + arrest)** | **No.** Weak/open grip causes immediate drop. Wrist does not create a usable slip direction at secure/τ=−8. |
| **B. Translation follow / recenter with v_x** | **No at secure grip.** Object co-moves; `r_h.x` unchanged. Follow would need slip, which currently equals loss. |
| **C. Hybrid** | Not demonstrated. |
| **D. Contact-preserving authority insufficient (this probe set)** | **Yes, for this frozen case and these short legal pulses.** |

This is an **authority** result, not a proof that no sequence of any length exists. It is enough to **stop optimizing** a CENTER-6 heuristic from these pulses.

---

## 12. Task-level existence proof

**Not run.** No interpretable correcting sequence was found, so no recovery-from-drift viewer claim.

`--mode recovery` currently replays ZERO and labels that there is no plan.

---

## 13. Interactive viewer commands

No MP4. Camera once; mouse rotate/pan/zoom. SPACE pause, `[` `]` speed, R restart.

```text
python training/demo_ballistic_recovery.py --mode zero
python training/demo_ballistic_recovery.py --mode recovery
```

ZERO shows the full task: approach → descend → grasp → lift → free-flight ball → impact → lateral drift → drop.

Recovery mode has **no** constructed intervention on this frozen case.

---

## 14. Unresolved limitations

- Pulses are 0.25–0.40 s. A longer wrist or a different τ between −18 and −8 was not searched (by instruction).
- `r_h.x` co-motion under secure grip may still allow **task-space** recentering that looks better in world frame while the **relative** error stays; that is not contact-preserving recovery of the grasp.
- MID restore error grows near runaway; use EARLY for kinematics-sensitive claims.
- LATE snapshot exists but was not used for a recatch test (out of scope).
- Next scientific step (not this task): off-center impact for angular impulse, then loss/recatch — only after accepting this translational result.

**No SAC, recatch, reward/obs change, impact retune, or MP4.**
