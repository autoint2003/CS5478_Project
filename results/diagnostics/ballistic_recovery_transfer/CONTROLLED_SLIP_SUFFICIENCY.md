# Controlled-slip sufficiency (CENTER-6)

**THIS IS NOT HEURISTIC OPTIMIZATION.**

One control variable: weak-slip **duration** at frozen `τ=-5`.
The physical quantity is `Δr_h.x`. Duration is not a recovery policy.

---

## 1. Frozen mechanism

- CENTER-6 ballistic translational progressive failure; EARLY intervention
- secure **+120°**, `τ=-18`
- weak grip **`τ=-5` only** (not swept)
- brake `τ=-18` for 0.40 s
- **corrected RETURN** to saved `R_nominal = r_des` (not modified here)
- noslip=1; no `v_x` / `v_z`; impact, angle, and ω limit frozen

Baseline (validated): 0.70 s → `Δr_h.x = −1.62 mm`, long hold survives.

---

## 2. Small duration experiment

Coarse set only: **0.70, 1.00, 1.30, 1.60 s**. No denser sweep. No τ/angle search.

| slip_s | Δr_h.x mm | r_h.x start mm | r_h.x end mm | lost during slip | held after RETURN |
|---:|---:|---:|---:|---|---|
| 0.70 | **−1.62** | 9.96 | 8.34 | no | yes |
| 1.00 | **−2.03** | 9.96 | 7.93 | no | yes |
| 1.30 | **−2.44** | 9.96 | 7.52 | no | yes |
| 1.60 | **−2.85** | 9.96 | 7.10 | no | yes |

`Δr_h.x` grows **approximately linearly** (~0.41 mm per extra 0.30 s). Bilateral contact is retained in all four. This is continuous sliding, not a contact-topology jump.

---

## 3. Actual Δr_h.x and slip kinematics

| slip_s | Δr_h.x mm | \|v_rel\| mean | max | p90 |
|---:|---:|---:|---:|---:|
| 0.70 | −1.62 | 14.3 mm/s | 27.8 mm/s | 26.3 mm/s |
| 1.00 | −2.03 | 10.7 mm/s | 27.8 mm/s | 25.3 mm/s |
| 1.30 | −2.44 | 8.6 mm/s | 27.8 mm/s | 24.1 mm/s |
| 1.60 | −2.85 | 7.3 mm/s | 27.8 mm/s | 22.6 mm/s |

Mean |v_rel| falls as the window includes more of the slow tail; the **max** is the same early transient (~28 mm/s). No ballistic escape in this set.

---

## 4. Contact geometry after return to nominal

Pad-1 half-width = **8.5 mm** (XML). Cylinder radius = **18 mm**.

| slip_s | r_h.x mm | COM to +x pad AABB rim mm | tactile CoP u_L / u_R mm | \|u\| / 8.5 mm | nL/nR | Fn L/R |
|---:|---:|---:|---|---:|---|---|
| 0.70 | 7.49 | **1.01** | +7.47 / −7.51 | **0.88** | 10/10 | 9.0/9.0 |
| 1.00 | 7.08 | 1.42 | +7.05 / −7.02 | 0.83 | 10/10 | 9.0/9.0 |
| 1.30 | 6.66 | 1.84 | +6.62 / −6.62 | 0.78 | 10/10 | 9.0/9.0 |
| 1.60 | 6.23 | **2.27** | +6.21 / −6.22 | **0.73** | 9/10 | 9.0/9.0 |

CoP `u` tracks `r_h.x` almost 1:1. Contacts stay on the cylinder wall (`r ≈ 17.9 mm`) with axial spread along the pad height (`v` ~ 1–2 mm from pad center Z). Fn left/right remain ~9 N after brake/return.

---

## 5. Edge-catch audit

**0.70 s is an edge-supported grasp, not a large-margin interior grasp.**

Evidence (not from contact count):

- cylinder COM is **1.0 mm** inside the +x pad AABB rim (pad `x_hi ≈ +8.5 mm`, `r_h.x ≈ 7.5 mm`)
- tactile CoP `|u| ≈ 7.5 mm` = **88% of pad half-width**
- per-contact `pos_h.x ≈ 7.4–7.5 mm` (`pad_x_frac ≈ 0.94`)
- the user’s visual “finger/pad edge catching near the cylinder diameter” matches these locations

It is still a **real support**: bilateral pads, Fn ~9 N each after secure, cylinder wall (not a corner-only 1-point), and the long hold does not creep. So: **stable but geometrically marginal**, not “unstable from appearance.”

Longer slip **shifts the same contact family inward** along pad-x (frac 0.94 → 0.86). It does **not** jump to a different contact set. At 1.60 s the COM is still only **2.3 mm** from the +x AABB rim and CoP is still at **73%** of half-width. That is better margin, not a centered basin.

---

## 6. Long-hold comparison

Same horizon after RETURN for every survivor.

| slip_s | held | hold Δr_h.x mm | hold \|v_rel\| mean | end r_h.x mm | end nL/nR |
|---:|---|---:|---:|---:|---|
| 0.70 | yes | **+0.02** | 0.32 mm/s | 7.51 | 10/10 |
| 1.00 | yes | −0.03 | 0.29 mm/s | 7.05 | 10/10 |
| 1.30 | yes | −0.02 | 0.28 mm/s | 6.64 | 10/10 |
| 1.60 | yes | −0.01 | 0.29 mm/s | 6.22 | 9/9 |

Additional slip does **not** buy a quieter hold: **all four already have ~0 post-return creep**. The difference is **where** the object sits on the pad, not whether it starts sliding again on this horizon.

Tilt stays ~0.05°. Relative rotation does not grow.

---

## 7. Observable differences (observation vector **not** changed)

What actually distinguishes 0.70 s vs 1.60 s:

| signal | distinguishes? |
|---|---|
| tactile CoP **u** (pad width) | **yes** — 7.5 mm → 6.2 mm; `|u|/8.5` is the edge metric |
| bilateral n / Fn after brake | **no** — ~10/10 and 9 N / 9 N in all survivors |
| aperture | **no** — ~17.9 mm, unchanged |
| relative velocity during **hold** | **no** — all ~0.3 mm/s |
| privileged `r_h.x` | tracks CoP `u`, but must not be the policy criterion |

A learned policy that should brake vs keep slipping can use **pad-frame CoP u** (already in the tactile abstraction), not a privileged recenter-to-zero rule.

---

## 8. Mechanism-level conclusion

**Stop condition: B, limited — then stop. Do not refine.**

- **A is false:** 0.70 s is **not** a large-margin interior grasp. The edge-catch appearance is real on pad-x / CoP `u`.
- **B, in a limited sense:** the validated gravity-slip mechanism **does** have extra authority. Over this coarse set, more exposure continuously moves the object **inward along the same pad-x edge family** (`Δr_h.x` −1.62 → −2.85 mm; CoP 88% → 73% of half-width). All remain captured.
- **Not a new basin:** 1.60 s is still a **near-edge** support (2.3 mm to AABB rim). This set does **not** produce a clearly deep, centered grasp.
- **D not reached:** 1.60 s did not hit a loss boundary.
- **C partial:** extra slip does not change long-hold creep (already arrested at 0.70 s).

**Do not conclude “the heuristic should slip for 1.30 s or 1.60 s.”**  
The mechanism can produce a **range** of correcting displacements. How much / how long / when to brake is for a later policy from observables (especially CoP `u`).

Raw: `raw/large_angle/slip_sufficiency.json`.

---

## 9. Representative viewer commands

No MP4. Baseline remains 0.70 s. The second command is the **coarse-set endpoint** (1.60 s) so the inward CoP shift is visible. It is **not** a recommended duration.

```text
python training/demo_ballistic_recovery.py --mode gravity_large_angle
python training/demo_ballistic_recovery.py --mode gravity_more_slip
```

RETURN was not modified. No SAC, recatch, impact change, τ sweep, or `r_h.x → 0` targeting.
