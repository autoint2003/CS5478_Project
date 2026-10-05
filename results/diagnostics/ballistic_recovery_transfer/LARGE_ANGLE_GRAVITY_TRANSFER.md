# CENTER-6 large-angle gravity transfer

Contact-preserving **authority** test on the frozen CENTER-6 ballistic translational progressive-failure case.

Not a learned policy. Not an optimized heuristic. Not recatch.

**BALLISTIC CONTACT-PRESERVING RECOVERY AUTHORITY PROOF** (numerical hold). User visual of the large-angle sequence is still **PENDING**.

---

## Interpretation of the previous small-angle result

The previous ±13° wrist pulses are **small-angle probes that did not produce correction**.

They do **not** support: “wrist/gravity mechanism does not transfer.”

That question had not been tested at the orientations where `g_h.x` is large.

Likewise: `τ=-3` dropping at the **original** wrist pose does **not** imply `τ=-3` drops after a >90° reorientation. Grip regime is contact-geometry dependent and was re-audited below.

---

## 1. EARLY state geometry

Frozen snapshot `raw/snap_early.pkl` (CENTER 6 m/s, matched +hand-x, `m_ball=0.05`, noslip=1). Impact was **not** retuned.

| quantity | value |
|---|---|
| t | 3.322 s (T_EARLY_DRIFT; impact 2.852 s) |
| r_h | **[+10.87, −0.02, 98.33] mm** |
| g_h | [+0.013, −0.230, **+9.807**] m/s² |
| nL / nR | 7 / 4 |
| axis tilt | 0.19° |
| FSM | lift; `v_cmd = +z 0.08`; weld off |

Gravity is still almost entirely along hand-z. There is **no** useful tangential gravity at EARLY. That is why ±13° could not drive correcting slip.

---

## 2. Required correction sign

`r_h.x > 0`, so desired `Δr_h.x` has sign **−1**.

If the object slides under gravity in the hand frame, `Δr_h.x` should have the same sign as `g_h.x`.

After reorientation we therefore want **`g_h.x < 0`**, which is **positive** `ω_y` / **positive** wrist angle in the `signed_rot_about_y` convention used here.

This sign is taken from EARLY geometry, not from the old teleport heuristic.

---

## 3. `g_h` versus wrist angle

Secure rotation only (`τ=-18`). Legal `ω_y = 3 rad/s`. Cartesian pose frozen (no extra `v_x`/`v_z`). Actual angle overshoots the command by ~4–5° (tracking lag, then freeze to the true hand).

| θ cmd | θ act | g_h.x pred | g_h.x meas | g_h.z meas | nL/nR | r_h.x mm | Fn L/R | hold |
|---:|---:|---:|---:|---:|---|---:|---|---|
| 0 | +0.1 | 0.01 | 0.00 | +9.81 | 4/5 | 10.97 | 8.9/8.9 | OK |
| +30 | +34.2 | −4.89 | **−5.51** | +8.12 | 4/6 | 11.01 | 9.0/9.0 | OK |
| −30 | −34.0 | +4.91 | **+5.50** | +8.12 | 5/4 | 11.50 | 8.8/8.8 | OK |
| +60 | +64.4 | −8.49 | −8.84 | +4.25 | 5/4 | 10.79 | 9.1/9.0 | OK |
| −60 | −64.5 | +8.50 | +8.86 | +4.21 | 4/4 | 12.33 | 8.6/8.6 | OK |
| +90 | +94.8 | −9.81 | **−9.78** | −0.80 | 4/4 | 10.34 | 9.1/9.0 | OK |
| −90 | −94.4 | +9.81 | **+9.78** | −0.77 | 4/2 | **17.69** | 7.1/7.1 | OK (stressed) |
| +105 | +110.0 | −9.48 | −9.22 | −3.34 | 4/4 | 10.13 | 9.0/9.0 | OK |
| −105 | −109.7 | +9.47 | +9.23 | −3.32 | **0/0** | 48.07 | — | **FAIL** |
| +120 | +125.1 | −8.50 | **−8.04** | −5.63 | 4/2 | 9.96 | 9.0/9.0 | OK |
| −120 | −125.2 | +8.49 | +8.00 | −5.67 | **0/0** | 157.8 | — | **FAIL** |

Measured `g_h.x` matches the `R_y(θ)` prediction. Positive wrist reverses hand-x gravity into the **correcting** half-plane. Negative wrist **amplifies** the existing `+r_h.x` error (already +17.7 mm at −90° while still nominally held).

Relative pose during **+** rotation stays ~10–11 mm: the cylinder **follows the gripper**. That follow is **not** recovery.

---

## 4. Maximum secure rotation

- **Positive** `ω_y`: the grasp remains bilateral through the compact set, including **~125°** actual (`θ_cmd=+120`).
- **Negative** `ω_y`: first loss at **`θ_cmd=−105°`** (actual −110°): both contacts off, `r_h.x` ~48 mm. −120° is a full dump.

Prerequisite #1 holds **only in the correcting-sign direction**. The wrong-sign large angle is already a gravity dump at `τ=-18`.

Panda/workspace did **not** prevent >90° about this hand-y axis from the EARLY lift pose.

---

## 5. Grip regimes at large angle

Selected held orientations with substantial correcting `g_h.x`: **+120, +105, +90** (actual ~125, 110, 95°). Compact `τ` set only. 0.70 s dwell, wrist held.

| θ_cmd | τ | kind | Δr_h.x mm | keep | agrees g_h.x | \|v_rel\|_max |
|---:|---:|---|---:|---|---|---:|
| +120 | −18 | SECURE_LIKE | −0.08 | yes | — | 0.026 |
| +120 | −8 | SECURE_LIKE | −0.36 | yes | — | 0.026 |
| +120 | **−5** | **CONTROLLED_CORRECTING_SLIP** | **−1.62** | yes | yes | 0.028 |
| +120 | −4 | CONTROLLED_CORRECTING_SLIP | −1.89 | yes | yes | 0.029 |
| +120 | −3 | CONTROLLED_CORRECTING_SLIP | −2.10 | yes | yes | 0.029 |
| +120 | −2 | DROP | −211 | no | yes | 1.99 |
| +105 | −18/−8/−5 | SECURE_LIKE | −0.07…−0.30 | yes | — | 0.026 |
| +105 | −4 / −3 | CONTROLLED_CORRECTING_SLIP | −2.11 / −2.40 | yes | yes | 0.03 |
| +105 | −2 | LOSS | −33 | no | yes | 0.42 |
| +90 | −18/−8/−5 | SECURE_LIKE | ~0 | yes | — | 0.026 |
| +90 | −4 / −3 | CONTROLLED_CORRECTING_SLIP | −2.25 / −2.58 | yes | yes | 0.03 |
| +90 | −2 | CONTROLLED_CORRECTING_SLIP | −5.57 | yes | yes | 0.047 |

At the original EARLY orientation, `τ=-3` dumped the cylinder. After large **+** rotation, **`τ=-3` is an intermediate correcting-slip regime** (bilateral contact retained, ~2 mm toward zero in 0.7 s). `τ=-8` is still secure-like. `τ=-2` is near-loss / drop except at +90°.

This is state dependence, not a retuned grip schedule.

---

## 6. Controlled-slip evidence

Existence candidate (first qualifying row, **not** optimized): **`θ_cmd=+120°` (actual ~125°), `τ=-5`**.

| | start of weak dwell | end (0.70 s) |
|---|---|---|
| r_h.x | 9.96 mm | **8.34 mm** (Δ **−1.62 mm**) |
| nL/nR | 4/2 | 10/10 |
| Fn L/R | 9.0 / 9.0 N | 2.5 / 2.5 N |
| ρ_max | 0.17 | 0.42 |
| aperture | 17.6 mm | 18.0 mm |
| g_h.x | −8.04 | −8.59 |
| \|v_rel\| | 0.020 | 0.003 (max in window 0.028) |
| tilt | 0.14° | 0.10° |

- bilateral contact retained
- directed relative motion (`r_h.x` toward 0)
- sign agrees with `g_h.x < 0`
- not a ballistic escape (`\|v_rel\|` stays ~cm/s)

Additional (not used in the construction): +90° / `τ=-2` moved **−5.6 mm** still in contact. That is evidence of authority, not a “best τ” claim.

---

## 7. Gravity causality

| condition | result |
|---|---|
| +120°, `τ=-5` | correcting slip, Δr_h.x = **−1.62 mm** |
| −120°, `τ=-18` rotate | **cannot hold**; contacts lost |
| −90° (largest secure opposite), `τ=-5` | **DROP**, Δr_h.x = **+164 mm** |
| +120°, `τ=-5`, **0g** | **SECURE_LIKE**, Δr_h.x = **+0.09 mm** |

The 0g dwell from the same large-angle state **removes** the correcting slide. Wrong-sign orientation **reverses** the outcome from slide-toward-zero into loss. This is gravity-driven, not a kinematic artifact of opening the fingers.

0g is diagnostic only.

---

## 8. Secure-brake evidence

Same +120° pose, after the `τ=-5` window: `τ → -18`, wrist **held**.

| | after slip | after 0.40 s brake |
|---|---|---|
| r_h.x | 8.34 mm | **8.25 mm** (correction retained) |
| nL/nR | 10/10 | 10/10 |
| Fn | 2.5 N | **9.0 N** |
| \|v_rel\| | 2.6 mm/s | **~0.5 mm/s** |
| table | — | none |

Restoring secure grip **arrests** the relative motion without undoing the ~1.6 mm correction. No extra dwell was added beyond this measurement window.

---

## 9. Return-to-nominal evidence

Return about hand-y at `τ=-18` (`ω_y=3`).

| | after return |
|---|---|
| residual wrist | **+0.65°** |
| r_h.x | **7.51 mm** |
| nL/nR | 10 / 8 |
| g_h | ~[0, 0, +9.81] (nominal gravity again) |
| table | none |

The reduced offset **survives** the return. Grip stays secure during return (not weakened).

A last-sample `v_rel` spike (~0.24 m/s) appears on the return endpoint log while contacts remain; it is not a drop. After the subsequent lift/hold, `v_rel` is again ~mm/s.

---

## 10. Task-level existence proof

**One** construction, not a search:

nominal grasp → lift → frozen CENTER-6 impact → **visible EARLY progressive drift** (t=3.322 s, not at ball contact) → secure **+120°** rotation → `τ=-5` slip 0.70 s → `τ=-18` 0.40 s → return wrist → resume lift/hold.

| | ZERO CENTER-6 | this construction |
|---|---|---|
| intervene | never | EARLY, after visible drift |
| table | **6.132 s** | **none** through **t=22.08 s** |
| end nL/nR | lost | **3 / 4** |
| end obj_z | table | **0.589 m** (elevated) |
| end r_h.x | escaped | **11.5 mm** |

`held_like=True` on the numerical gates (≥12 s after intervention, bilateral contact, object still in air).

**Honest limits of this proof**

- `r_h.x` is **not** recentered to 0. Slip removed ~1.6–3 mm; long hold at nominal orientation **creeps back** toward ~11 mm.
- Arrest of **runaway/drop** is the existence claim. ZERO’s escape is around `r_h.x ≳ 16 mm` near t=5.8–6.1 s. Spending that window in a gravity-safe orientation, plus a few millimetres of correcting slip, keeps the state **below the escape cliff**.
- This is **not** the teleport sequence tuned for millimetre recentering, and **not** a policy.

Mechanism vs teleport:

| | teleport (ideal IC) | CENTER-6 large-angle |
|---|---|---|
| A. wrist → weak grip → gravity slip → secure | established | **transfers at large +θ** |
| B. hand-x follow at secure grip | — | previously: object co-moves; not retested here |
| D. no contact-preserving authority | **withdrawn** as the gravity conclusion | false for large-angle +θ |

Outcome: **A transfers**, but it needs **large** wrist reorientation because CENTER-6 is translational and `g_h.x≈0` at EARLY. Small-angle wrist was the wrong test.

---

## 11. Interactive viewer commands

No MP4. Camera initialized once; mouse remains active. SPACE pause/resume, `[` `]` speed, R restart.

```text
python training/demo_ballistic_recovery.py --mode zero
python training/demo_ballistic_recovery.py --mode gravity_large_angle
```

Headless audit (already run):

```text
python training/demo_ballistic_recovery.py --mode large_angle
```

ZERO: full approach → descend → grasp → lift → free-flight ball → impact → lateral drift → drop.

`gravity_large_angle`: the same through EARLY drift, then the large-angle construction above.

## USER VISUAL OBSERVATION: PENDING for `gravity_large_angle`

---

## 12. Unresolved limitations

- Compact angle/τ sets only. No optimizer for “best” angle, τ, or timing.
- Existence sequence uses the **first** qualifying pair `(+120°, τ=-5)`, not the largest Δr_h.x.
- Residual ~11 mm offset after 12 s is still a deteriorated grasp; it is not a “good as new” hold.
- Return/lift later shows modest wrist drift (end wrist ~−10° in the long snapshot-from-EARLY continuation vs 0.65° immediately after return). Not cleaned up.
- Legal 4D only: no privileged absolute repositioning, no extra action dim, no object-GT feedback in the commands.
- Off-center ballistic impact (angular impulse) and recatch are **out of scope**.
- No SAC, no reward/obs change, no impact retune, no MP4.

Raw: `raw/large_angle/` (`secure_sweep.json`, `grip.json`, `causality.json`, `brake_return.json`, `task_level.json`, `summary.json`) and `raw/large_angle_plan.json`.

## Return-to-Nominal Orientation Audit

This section is appended after the validated large-angle mechanism. Earlier sections are not rewritten.

### Previous residual orientation error

From the construction that the user visually validated (`raw/large_angle/brake_return.json`):

- RETURN stop metric was `signed_rot_about_y_deg(R_hand_start, R_hand)`; it reported **+0.65°** at return-end.
- That metric is a **single-axis atan2(R[0,2], R[0,0])**, not the SO(3) geodesic.
- `r_des` vs `R_nominal` was **not** logged.
- After the subsequent lift/hold, `wrist_deg` was **−9.64°** and `g_h.x = +1.65 m/s²` (atan2(1.65, 9.67) ≈ **9.7°**). That matches the visible leftover wrist tilt.

### Cause in implementation

RETURN did **not** use a fixed duration, but it was equivalent in effect to an open-loop unwind:

- direction: `sign = −1` if `signed_rot_about_y(R_hand_start, R_hand) > 0` else `+1`
- rate: legal bound `ω_y = ±3 rad/s` (constant, not reduced near the target)
- timeout: `|θ_cmd| / 3 + 1.5 s` with `θ_cmd = +120°`
- stop: `|signed_rot_about_y(R_hand_start, R_hand)| < 0.8°`

`tick_4d` / `map_recovery4d` integrates **`r_des`** at 3 rad/s about current `r_des` y. The hand lags. Stopping on the **hand** angle therefore leaves **`r_des` past the nominal pose**. `omega` then goes to 0, so `r_des` stays overshot. `continue_zero` / lift holds that `r_des` (`w_cmd = 0`), and the hand tracks the leftover tilt.

`freeze()` at RETURN start parked `r_des := R_hand` (good for aligning the target with the arm) but never drove `r_des` back to the **saved EARLY `r_des`**.

### Corrected return logic

Frozen mechanism unchanged: CENTER-6, EARLY, +120° secure, `τ=-5` for 0.70 s, `τ=-18` brake 0.40 s.

Before recovery: `R_nominal = r_des` at EARLY (FSM lift / grasp orientation); also log `R_hand_start`.
RETURN: `τ=-18`; legal `ω_y` P-control on the body-y component of `log(R_des^T R_nominal)`, saturated at 3 rad/s; stop `ω_y` when `|y-error| ≤ 0.50°`; latch `r_des := R_nominal` (controller target only, not `qpos`); 0.30 s catch-up with `ω_y=0` so the hand can track. Off-axis leftovers cannot be cancelled by 1-DOF `ω_y`; the latch restores the exact saved desired matrix. No object GT. No `v_x`.

### After the fix (exact same construction, new RETURN only)

- weak-slip Δr_h.x: **-1.62 mm** (previous construction **−1.62 mm**)
- contacts at slip end nL/nR: [10, 10]
- contacts at return end nL/nR: [10, 10]
- `angle(R_nominal, R_des)` at RETURN end: **0.000°**
- `angle(R_nominal, R_hand)` at RETURN end: **0.133°**
- signed-y(R_hand_start, R_hand) leftover (old metric): 0.03°
- final r_h.x after hold: **7.51 mm**
- table: None; held_like=True
- end nL/nR: [10, 10]; obj_z=0.591 m; t_end=23.49 s
- after long hold: `angle(R_nominal, R_des)`=0.000°, `angle(R_nominal, R_hand)`=0.070°

Weak-slip before RETURN is the same recipe (duration/τ/angle). Millimetre Δr_h.x is compared above; it is not re-optimized.

Viewer: `python training/demo_ballistic_recovery.py --mode gravity_large_angle`

