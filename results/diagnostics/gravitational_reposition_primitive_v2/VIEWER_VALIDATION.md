# Viewer validation + signed relative motion (v2)

Physics of the primitive was **not** changed (noslip, τ, wrist, μ, m, solref, dt, brake threshold).

## USER VISUAL OBSERVATION

- **CONTROLLED SLIP: CONFIRMED** (user, `--viewer slip`).
- **SLIP → BRAKE: PENDING** until the user watches `--viewer brake_visual`.
  Cursor logs are not visual confirmation of brake.

Canonical `--viewer brake` (1.2 mm) remains the quantitative case; it is too small for reliable naked-eye discrimination.

## 1. Viewer / headless trajectory equivalence

Viewer and headless `--verify` use the same `rotate_wrist` / `hold_window` / `tick_vw` path,
the same parent snapshot (`raw/parent_snapshot.pkl`), noslip=1, 30°, 1.2 rad/s, τ=−3,
dt=0.002, and brake `|progress_s|>=1.2 mm` then τ=−18.
Viewer `--viewer slip|brake` writes `raw/viewer_slip.npz` / `raw/viewer_brake.npz` of the trajectory actually shown. Compare those to `verify_slip_a.npz` / `verify_brake.npz` (same functions, no GUI).

```json
{
  "live_opt": {
    "mujoco": "3.12.0",
    "noslip_iterations": 1,
    "noslip_tolerance": 1e-06,
    "iterations": 100,
    "tolerance": 1e-08,
    "solver": 2,
    "cone": 0,
    "impratio": 1.0,
    "timestep": 0.002,
    "integrator": 3,
    "gravity": [
      0.0,
      0.0,
      -9.81
    ],
    "solref_object": [
      0.01,
      1.0
    ]
  },
  "parent": "D:/NUS/CS5478 Intelligent Robots/CS5478_Project/results/diagnostics/gravitational_reposition_primitive_v2/raw/parent_snapshot.pkl",
  "repeatability_slip_progress_mm": [
    4.490351550008682,
    4.490351550008682
  ],
  "repeatability_abs_diff_mm": 0.0,
  "vs_published_slip": {
    "kind": "slip",
    "replay_progress_s_end_mm": 4.490351550008682,
    "canonical_progress_s_end_mm": 4.490351550008682,
    "abs_diff_mm": 0.0,
    "replay_drh_end_mm": [
      -2.2537905895920627,
      0.0331078312665761,
      3.8838520609063427
    ],
    "canonical_drh_end_mm": [
      -2.252830260225995,
      0.03328673428933397,
      3.8806575742489633
    ],
    "replay_noslip": 1,
    "replay_t_brake": null,
    "dt": null
  },
  "vs_published_brake": {
    "kind": "brake",
    "replay_progress_s_end_mm": 1.2232274253238395,
    "canonical_progress_s_end_mm": 1.2232274253238395,
    "abs_diff_mm": 0.0,
    "replay_drh_end_mm": [
      -0.6809188702217116,
      0.016809839570275334,
      1.0196491078603849
    ],
    "canonical_drh_end_mm": [
      -0.6799585408556437,
      0.016988742593033204,
      1.0164546212030057
    ],
    "replay_noslip": 1,
    "replay_t_brake": 0.31999999999996476,
    "dt": null,
    "replay_s_at_brake_mm": 1.1663856382696254,
    "replay_extra_after_brake_mm": 0.056841787054214166
  },
  "verify_slip_progress_end_mm": 4.490351550008682,
  "verify_slip_drh_end_mm": [
    -2.2537905895920627,
    0.0331078312665761,
    3.8838520609063427
  ],
  "verify_brake_t_brake": 0.31999999999996476,
  "verify_brake_s_end_mm": 1.2232274253238395,
  "dt": 0.002,
  "omega": 1.2,
  "angle": 30.0,
  "tau_slip": -3.0,
  "tau_secure": -18.0,
  "brake_after_mm": 1.2,
  "hold_s_slip": 1.0,
  "hold_s_brake": 2.0,
  "playback_is_wallclock_only": true,
  "playback_speed": 0.28,
  "verify_s_at_brake_mm": 1.1663856382696254,
  "verify_extra_after_brake_mm": 0.056841787054214166
}
```

## 2. Exact definition of s / progress_s

At the start of the hold (after tilt, before grip reduction):

- `r_h` = `R_h^T (p_object - p_hand)`  (live hand frame)
- `g_h` = `R_h^T g`
- `u_hat` = `(g_h.x, 0, g_h.z) / ||...||`  **from that trial's measured g_h**
- `progress_s` [mm] = `dot(r_h(t)-r_h(0), u_hat) * 1000`

`progress_s` is a **progress / magnitude coordinate along trial gravity in the pad x–z plane**.
It is constructed so gravity-aligned motion is positive. It is **not** a fixed signed axis.
Do not use `progress_s` to claim directional reversal.

Legacy key `s_mm` in npz files is this same `progress_s`.

## 3. Fixed-frame Δr_h for +30 / −30

Same hand-frame convention both trials (live `R_h` of that trial):

```json
{
  "plus30": {
    "file": "direction_plus.npz",
    "g_h": [
      -4.896465081120489,
      -0.11961781655432005,
      8.499789484883161
    ],
    "Delta_r_h_mm": [
      -2.252830260225995,
      0.03328673428933397,
      3.8806575742489633
    ],
    "progress_s_mm": 4.48715083039046,
    "dot_Delta_rh_mm_g_xz": 44.0156771469217,
    "cos_Delta_rh_g_xz": 0.9999669034918092,
    "nL0": 8,
    "nL1": 4,
    "nR0": 8,
    "nR1": 4
  },
  "minus30": {
    "file": "direction_minus.npz",
    "g_h": [
      4.890925589700592,
      -0.09980206926103795,
      8.50323388029419
    ],
    "Delta_r_h_mm": [
      2.2505258876180334,
      0.019918466457398986,
      3.911699152571488
    ],
    "progress_s_mm": 4.51289893302124,
    "dot_Delta_rh_mm_g_xz": 44.26924741769862,
    "cos_Delta_rh_g_xz": 0.9999902534879279,
    "nL0": 8,
    "nL1": 10,
    "nR0": 8,
    "nR1": 10
  },
  "g_hx_flips": true,
  "Delta_rh_x_flips": true
}
```

## 4. Gravity-direction consistency

VERIFIED CODE FACT + RAW TRAJECTORY: `g_h.x` flips with wrist sign, and **`Δr_h.x` flips with it** (Δx_+ = −2.25 mm, Δx_− = +2.25 mm). `g_h.z` stays ≈ +8.50 m/s² on both trials and `Δr_h.z` stays ≈ +3.9 mm (shared distal component). `dot(Δr_h, g_h_xz)` is positive on both trials (cos ≈ 1). `progress_s ≈ +4.5 mm` on both trials by construction of trial-wise `u_hat` and is **not** the directional coordinate.

## 5. Viewer visualization changes (display only)

- Default camera: one-shot safe oblique (lookat=cylinder, distance=0.55 m, azimuth=148, elevation=-25). `--camera robot` wider (0.85 m). No per-frame tracking. Mouse owns the camera. R resets the initial preset.
- Hand-fixed x (red) and z (blue) axes (thin).
- Yellow ruler ticks at 0,1,2,3,4,5 mm along `u_hat`, offset 12 mm in hand-y from the slip-start center.
- Sparse white wire ghost: start axis + radial ticks (outside cylinder radius).
- Green trail: last ~1 s of cylinder center in the **current** hand frame.
- 2.5 s wall-clock PAUSED after the viewer opens (no mj_step). Then playback ~0.28×; 0.8 s pre-slip pause; 1 s end freeze (slip) / 2 s freeze (brake).
- Brake overlay: SLIP → BRAKE (~0.15 s sim after trigger, physics not paused) → SECURE.

## 6. Slip viewer command

```text
python training/gravitational_reposition_primitive_v2.py --viewer slip
```

## 7. Brake viewer command (canonical 1.2 mm)

```text
python training/gravitational_reposition_primitive_v2.py --viewer brake
```

Quantitative only. Pre-brake motion is ~1.2 mm — not a reliable naked-eye check.

## 8. Visual brake demo (3.0 mm) — NOT CANONICAL

```text
python training/gravitational_reposition_primitive_v2.py --viewer brake_visual
```

**VISUAL DIAGNOSTIC — NOT CANONICAL BRAKE THRESHOLD.**

Only change vs canonical: brake when `progress_s >= 3.0 mm`. Then τ=−18, 2 s physics hold. Magenta **BRAKE POINT** marker is hand-fixed at the trigger pose. Overlay **BRAKE / tau: -3 -> -18 / trigger: 3.00 mm** for ~0.5 wall-clock seconds without pausing `mj_step`.

Headless raw: `raw/brake_visual.json`, `raw/brake_visual.npz`. Viewer run writes `raw/viewer_brake_visual.npz` and does **not** overwrite `brake.npz` / `brake.json` / `viewer_brake.npz`.

| | canonical | visual |
|---|---|---|
| brake threshold | 1.2 mm | 3.0 mm |
| tau slip | −3 | −3 |
| tau brake | −18 | −18 |
| wrist | 30 deg | 30 deg |
| noslip | 1 | 1 |
| t_brake | 0.32 s | 0.708 s |
| progress_s at brake | ~1.22 mm | 3.008 mm |
| post-brake extra | 0.057 mm | +2 s: 0.017 mm |
| bilateral after | yes (8/8) | yes (4/8) |
| arrested | yes | yes |

3 mm is **inside the braking basin** (raw). USER VISUAL of this demo is still **PENDING**.
