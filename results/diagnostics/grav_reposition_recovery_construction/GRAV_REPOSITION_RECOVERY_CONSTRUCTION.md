# Gravitational reposition recovery construction (noslip=1)

Privileged / manual physics construction on one matched delayed-failure state.
RULE was not called. `controllers/rule_based_recovery.py` and `rule_based_recovery.yaml` were not modified.
SAC was not trained. Reward, observation, detector, and disturbance direction were not redesigned.

Frozen solver on every restored snapshot in this folder:

- MuJoCo 3.12.0
- `noslip_iterations = 1`
- `noslip_tolerance = 1e-6`
- `timestep = 0.002`
- gravity `[0, 0, -9.81]`
- object mass 0.20 kg, pair μ = 1.0
- pose-only coupled teleport, canonical continuation
  `p_des := p_hand` once, world-z `+0.08 m/s`, Δz = 0.18 m, `τ = -18`, then hold 10 s

**USER VISUAL OBSERVATION**

- Simple CONTROLLED SLIP primitive: **CONFIRMED** (prior viewer).
- S_FAIL FULL recovery: **PENDING** until the user watches `--mode full`.
  Cursor does not mark this visually confirmed.

Stop condition **A** is met in the headless logs (same `S_FAIL`: ZERO drops, FULL survives lift + 10 s hold). That is physical log evidence, not a visual confirmation.

---

## 1. noslip=1 S_FAIL revalidation

s = 2.0 was reconstructed from the successful centered airborne parent and re-run under noslip=1. It still belongs to the required class. No local s-search beyond 2.0 was needed.

| s | t0 valid | ZERO class | lift completed | drop | t_drop after continuation start | 10 s hold survived |
|---|----------|------------|----------------|------|---------------------------------|--------------------|
| 2.0 | yes | DELAYED_FAIL | yes | yes | 11.808 s | no |

`S_FAIL` **s = 2.0**. Snapshot: `raw/sfail.pkl`. Parent: `raw/parent.pkl`.

ZERO is not immediate loss: the object stays captured through the entire 0.18 m lift (lift done at continuation t ≈ 2.25 s) and through most of the 10 s hold, then both-contact loss / drop at continuation t = 11.808 s (object height collapses from ≈ 0.659 m to the table). That is the delayed-failure class requested.

---

## 2. Exact disturbed initial state

Restored `raw/sfail.pkl` (not interpolated). Live noslip=1. Time t = 4.114 s.

**qpos (first 16):**
`[8.92e-5, -0.212, -2.37e-6, -1.895, -5.23e-5, 1.705, -0.785, 0.01765, 0.01765, 0.49844, 0.00885, 0.49440, 0.07313, 0.99695, -0.00959, 0.02542]`

Arm/finger qpos matches the parent; object pose is the coupled teleport. qvel is the parent qvel (pose-only perturbation).

**ctrl:** gripper `ctrl[7] = -18`. **p_des** = hand `[0.49542, 3.80e-5, 0.59309]`. **v_cmd = w_cmd = 0**.

| quantity | value |
|----------|--------|
| r_h (m) | `[0.008816, 0.000903, 0.098730]` |
| e_x | **+8.82 mm** (object +x in the hand vs centered 0) |
| e_z | −0.27 mm vs r_h.z nom 0.099 m |
| nL / nR | **11 / 0** (right pad open at the instant of teleport) |
| Fn_L / Fn_R | 8.36 N / 0 |
| Ft_L / Ft_R | 2.83 N / 0 |
| ρ_max | 0.46 |
| g_h | `[0.00048, -0.210, 9.808]` (wrist still vertical) |
| cylinder axis in hand | `[-0.146, 0.028, 0.989]` (~8.4° from hand +z) |
| v_rel_h | ~ `[1.7e-5, 3.5e-4, -1.4e-5]` m/s |
| ω_rel_h | ~ `[0.010, 5e-5, -1e-4]` rad/s |
| object / hand world p | obj `[0.49844, 0.00885, 0.49440]`, hand `[0.49542, 3.8e-5, 0.59309]` |
| ncon | 11 |

Parent (same t, before teleport): nL/nR = 7/6, r_h.x ≈ 0.22 mm, cyl_h.z ≈ −1 (upright cylinder convention), τ = −18, noslip=1.

The t=0 unilateral right-pad gap is the captured disturbed topology, not an immediate drop: ZERO at 0.5 s already has nL/nR = 6/4, and FULL/ROTATE recover bilateral during the 100 ms observe + rotate.

Full dump: `raw/sfail_snapshot.json`, `raw/parent_snapshot.json`, `raw/sfail_t0.json`.

---

## 3. Wrist-direction derivation

100 ms freeze-here, τ = −18, no wrist command (`raw/geometry_100ms.json`):

| quantity | after 100 ms |
|----------|----------------|
| r_h | `[0.008926, -0.000136, 0.098744]` |
| e_x | **+8.93 mm** (still +x of the desired centered grasp) |
| nL / nR | **8 / 4** (bilateral restored) |
| ρ_max | 0.377 |
| Fn_L / Fn_R | 8.93 / 9.01 N |
| g_h | `[-0.0037, -0.178, 9.808]` |
| cyl_h | `[-0.147, 0.017, 0.989]` |
| \|v_rel_h\| | ~ 3.4 mm/s |

Desired grasp region: drive **r_h.x → 0** (and keep cylinder axis near hand z). The error is **positive e_x**.

Primitive calibration: wrist **+30° about hand-y** produces **g_h.x < 0**, and relative motion **Δ r_h.x < 0**.

Therefore the commanded wrist sign is **+30°**, not −30°, so tangential gravity in the pad plane pulls the object toward −hand-x (smaller e_x). This is geometry on this snapshot, not the old RULE table.

ω_rotate = 1.2 rad/s, τ = −18 during ROTATE (no intentional slip).

---

## 4. Local controlled-slip regime

After rotate, grip was stepped locally in `[-5, -2.5]`. 0.40 s probes from the post-rotate snapshot (`raw/slip_probe.json`):

| τ | class | useful Δr_h·û (mm) | nL/nR | ρ_max |
|---|--------|---------------------|-------|--------|
| **−3.0** | **LOSS** | 0.017 | 0 / 9 | 1.0 |
| **−4.0** | **CONTROLLED_SLIP** | 0.455 | 10 / 10 | 0.51 |

τ = −3 is LOSS on this disturbed geometry (primitive LOSS boundary). Grip was strengthened once to **τ = −4**. No further sweep. û = (−1, 0, 0) in the hand (useful motion = decrease of r_h.x).

Slip on S_FAIL at τ = −4 is **slower** than the centered primitive (~4.5 mm / 1 s at τ = −3, 30°). Here ~0.45 mm in 0.40 s with bilateral contact retained.

---

## 5. Brake-target comparison

From the same S_FAIL: observe 100 ms → +30° rotate at τ = −18 → slip at τ = −4 until useful progress hits the target or 1.2 s timeout → τ → −18, hold 0.5 s, **do not return vertical**.

| target | hit in 1.2 s? | lost? | progress | post-brake nL/nR | post-brake e_x | post-brake ρ | score |
|--------|----------------|-------|----------|------------------|----------------|--------------|-------|
| 1 mm | yes (1.000 mm) | no | 1.000 mm | 10 / 10 | 7.92 mm | 0.20 | +15.15 |
| 2 mm | no | no | 1.206 mm plateau | — | — | — | discarded |
| 3 mm | no | no | 1.206 mm | — | — | — | discarded |
| 4 mm | no | no | 1.206 mm | — | — | — | discarded |

The 2–4 mm targets were **not loss**; useful x-progress saturated near **1.2 mm** in the 1.2 s window at τ = −4. Only the 1 mm target is a defined brake endpoint.

At the 1 mm brake instant: r_h ≈ `[0.00797, 0, 0.10120]`. After 0.5 s secure: r_h ≈ `[0.00792, 0, 0.10119]`, extra motion ~0.05 mm, |v_rel| ≈ 0.44 mm/s, ω_rel small, g_h ≈ `[-4.74, -0.04, 8.59]` (wrist still ~29°).

---

## 6. Selected S_BRAKE

**S_BRAKE = 1 mm useful progress**, then 0.5 s τ = −18 at the tilted pose.

Why (lift success was **not** used): it is the only bilateral 1–4 mm endpoint; after the secure hold it is still 10/10, low ρ (stick), residual twist small, e_x reduced 8.93 → 7.92 mm. Cylinder tilt is **not** materially corrected (cyl_h.x still ≈ −0.147).

Snapshot: `raw/s_brake.pkl`, `raw/s_brake_snapshot.json`. Restored live noslip=1, `ctrl[7] = -18`.

---

## 7. Return-vertical trajectory

From S_BRAKE: τ = −18, ω_return = 1.0 rad/s, no snap of r_des, no open.

FULL log (`raw/full.npz`), times relative to S_FAIL t0 = 4.116 s:

| rel t (s) | phase (inferred) | r_h.x mm | nL/nR | ρ | \|v_rel\| | notes |
|-----------|------------------|----------|-------|---|-----------|--------|
| 0.00 | observe | 8.82 | 11/0 | 0.46 | 0.0003 | S_FAIL |
| 0.50 | rotate | 8.99 | 8/6 | 0.39 | 0.084 | wrist motion |
| 1.00 | slip τ=−4 | 8.51 | 10/10 | 0.51 | 0.006 | toward −x |
| 1.50 | slip / brake | 8.03 | 9/10 | 0.51 | 0.002 | ~1 mm useful |
| 2.00 | secure tilted | 7.93 | 10/10 | 0.20 | 0.0004 | τ=−18 |
| 2.50 | return | 7.86 | 10/10 | 0.21 | 0.073 | ω_return |
| 3.00 | near vertical | 7.91 | 10/10 | 0.23 | 0.004 | |

Wrist at FULL continuation start: 0.40° (from construction `wrist_end`). Cylinder axis stays ~8.4° in the hand; return does not unwind the object tilt.

---

## 8. Vertical secure-hold result

2.0 s freeze-here, τ = −18, **no lift**.

FULL at end of VERT_HOLD (rel t ≈ 4.64 s, abs t = 8.76 s):

- nL/nR = **10 / 10**
- r_h = `[0.00798, 0, 0.10173]` (e_x 7.98 mm, e_z +2.73 mm)
- |v_rel_h| ≈ 0.05 mm/s (decayed)
- ρ_max ≈ 0.23 (not the old noslip=0 low-ρ creep band)
- cyl_h ≈ `[-0.147, 0.001, 0.989]`
- object z ≈ 0.455 m

No both-contact loss. Relative pose bounded. A slow residual e_x drift remains (~0.03 mm/s later in the long hold) and is **orders of magnitude below** the superseded noslip=0 secure-grasp creep (~1.2 mm/s).

ZERO never gets a 2 s vertical recovery hold; under the same τ = −18 vertical continuation it re-bilateralizes then slowly loses the left pad during the post-lift hold (nL 6→2) before the drop.

---

## 9. Nominal lift + 10 s hold

Common continuation after VERT_HOLD (FULL) / immediately from S_FAIL (ZERO): `p_des := p_hand` once, +0.08 m/s world z, Δz = 0.18 m, τ = −18, then 10 s hold. No RULE, no extra recovery.

**ZERO** (`raw/zero.npz`, `raw/zero.json`)

- lift completed (abs t_lift = 6.364 s)
- first logged topology is unilateral (nR=0 at t0); bilateral by ~0.5 s
- both-contact loss at continuation t = 11.808 s
- drop yes; 10 s hold **not** survived
- end nL/nR = 0/0, obj_z = 0.35 m, e_x = 162 mm (fallen)

During the hold, e_x creeps 9.2 → 12.9 mm with nL thinning (6→2), then collapse after 11.5 s.

**FULL** (`raw/full.npz`)

- lift completed (abs t_lift = 11.010 s, continuation start 8.760 s)
- t_drop = none, t_both_loss = none
- 10 s hold **survived**
- end nL/nR = **7 / 9**, obj_z = 0.623 m (still airborne)
- end r_h = `[0.00855, 2.2e-5, 0.10176]`, |v_rel| ≈ 0.07 mm/s
- ρ_max = 0.26, Fn_L/Fn_R ≈ 8.98 / 9.02 N
- cyl_h still `[-0.147, 0.001, 0.989]`

Physical task outcome: **captured lift + 10 s hold**, not an E_TOL / `recovered()` / D_t flag.

---

## 10. ZERO vs ROTATE_ONLY vs NO_BRAKE vs FULL

Same `S_FAIL` snapshot. ROTATE_ONLY: +30° at τ = −18, no grip weaken, return, vertical hold, same continuation. NO_BRAKE: rotate, τ = −4 for up to 1.5 s, **return while still at τ = −4**, then secure vertical and continuation.

| mode | broke in construction? | lift | drop | t_drop | hold 10 s | end nL/nR | end e_x | end obj_z |
|------|------------------------|------|------|--------|-----------|-----------|---------|-----------|
| ZERO | — | yes | **yes** | 11.81 s | no | 0/0 | 162 mm | 0.35 m |
| ROTATE_ONLY | no | yes | **yes** | 10.17 s | no | 0/0 | 24 mm | 0.42 m |
| NO_BRAKE | no | yes | **no** | — | **yes** | 10/10 | 7.75 mm | 0.63 m |
| FULL | no | yes | **no** | — | **yes** | 7/9 | 8.55 mm | 0.62 m |

Causal reading from this one matched IC:

- Wrist rotation **alone** does not save S_FAIL (ROTATE_ONLY drops like ZERO, slightly earlier).
- Grip-weakened slip **does** (FULL and NO_BRAKE both survive).
- The **1 mm brake is not uniquely necessary** on this IC: longer τ = −4 slip without a pre-return brake also survives and finishes at a slightly smaller |e_x| (7.75 vs 8.55 mm).
- The useful mechanism on this state is **controlled relative x-reposition under a weakened but bilateral grip**, not “tilt the wrist and hold τ = −18”.

Figures: `figures/fig_ablation.png`, `figures/fig_early.png`, `figures/zero_vs_full.png`.

---

## 11. Raw trajectory evidence

Inspected independently (npz + restored pickles), not from summary tables alone.

| file | content |
|------|---------|
| `raw/sfail.pkl` + `raw/sfail_snapshot.json` | qpos/qvel/ctrl/p_des/r_des/v_cmd/w_cmd, live noslip=1, contacts |
| `raw/parent.pkl` + `raw/parent_snapshot.json` | centered airborne parent |
| `raw/s_brake.pkl` + `raw/s_brake_snapshot.json` | post 1 mm + 0.5 s secure, tilted |
| `raw/zero.npz` | ZERO r_h, nL/nR, ρ, obj_z, v_rel, g_h, cyl_h |
| `raw/full.npz` | FULL including hold to t = 21.01 s |
| `raw/rotate_only.npz` | ROTATE_ONLY drop |
| `raw/no_brake.npz` | NO_BRAKE survive |
| `raw/geometry_100ms.json` | 100 ms audit |
| `raw/slip_probe.json` | τ = −3 LOSS / τ = −4 slip |
| `raw/brake_grid.json` | 1–4 mm |
| `figures/fig_zero.png` | ZERO pose / contacts / height / \|v_rel\| |
| `figures/fig_full.png` | FULL same |

ZERO both-loss is a real sample: rel t = 11.81 s, nL=nR=0, |v_rel| = 2.12 m/s, obj_z falling. FULL last sample: nL=7 nR=9, |v_rel| = 7e-5 m/s, obj_z = 0.623 m.

---

## 12. Viewer commands

Replay uses the saved S_FAIL (no re-search). Camera is initialized **once** (distance 0.55 m, azimuth 148°, elevation −25°). Mouse owns the camera after that. **R** resets the preset. 2.5 s pause before the run. Overlay: phase, wrist angle, τ, relative displacement, ρ, nL/nR. Hand-fixed slip-start reference, trail, brake marker (FULL). The run includes the 10 s hold; it does not exit at lift completion.

```text
python training/demo_grav_reposition_recovery.py --mode zero
python training/demo_grav_reposition_recovery.py --mode full
python training/demo_grav_reposition_recovery.py --mode rotate_only
python training/demo_grav_reposition_recovery.py --mode no_brake
```

Headless construction (rebuilds S_FAIL from parent):

```text
python training/demo_grav_reposition_recovery.py
```

---

## 13. USER VISUAL OBSERVATION: pending

Do not treat the headless FULL survival as user-confirmed.

---

## 14. Unresolved issues

1. **NO_BRAKE also survives** this S_FAIL. The 1 mm brake is a valid secure waypoint, not the unique cause of task success. ROTATE_ONLY still fails, so slip/reposition is doing the work.
2. **Useful x-progress saturates near 1.2 mm** at τ = −4 in 1.2 s. The 2–4 mm grid never hit. Cylinder tilt (~8.4°) is essentially unchanged. Recovery here is a small pad-plane translation, not a full re-centering / re-alignment.
3. **τ = −3 is LOSS** on S_FAIL (it was CONTROLLED_SLIP on the centered primitive). Local τ = −4 was required. Frozen RULE `tau_slip = -2` remains invalid for this physics.
4. S_FAIL **t0 is unilateral** (nR = 0). It is still a valid captured delayed-failure IC under the stated t0 check; bilateral contact returns within ~0.1–0.5 s at τ = −18.
5. Residual e_x drift during the FULL 10 s hold (~8.00 → 8.55 mm) is small vs noslip=0 creep but not zero.
6. This is **not** a policy. Brake used privileged r_h. RULE / SAC / detector / reward were not updated.

---

## Construction used

- Rotate +30° about hand-y at 1.2 rad/s, τ = −18
- Slip τ = −4 until ~1.00 mm useful (−r_h.x)
- Brake τ = −18 for 0.5 s
- Return at 1.0 rad/s, τ = −18
- Vertical hold 2.0 s, τ = −18
- Nominal lift 0.18 m at 0.08 m/s + 10 s hold
