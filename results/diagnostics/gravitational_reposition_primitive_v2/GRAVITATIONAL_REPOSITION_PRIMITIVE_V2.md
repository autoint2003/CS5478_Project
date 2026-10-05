# Gravitational reposition primitive v2

**noslip_iterations = 1** (frozen contact-model correction).

Old low-ρ secure-grip inward slide (60° + τ=-18, ρ≈0.10, ~1.8 mm / 2 s) is **SUPERSEDED BY NOSLIP CALIBRATION**. It is not physical recovery authority.

This experiment does **not** apply the primitive to coupled teleport s=2.0. No SAC. No RULE retune.

## VERIFIED CODE FACT

```json
{
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
}
```

- Canonical freeze: `config/sim.yaml` `noslip_iterations: 1`, `noslip_tolerance: 1e-06`.
- Applied in `envs/grasp_sim.py` `apply_solver_from_cfg` after XML load; XML option also set in `envs/xml_build.py` / `assets/panda_torque.xml`.
- Live `model.opt.noslip_iterations` at parent = **1**.
- Wrist rotation about live hand-y at 1.2 rad/s; `p_des := p_hand`; no table, impact, or teleport.
- Intended **progress** axis û is unit `(g_h.x, 0, g_h.z)` from **measured** `g_h = R_h^T g` at hold start.

**Definition of `progress_s` (legacy npz key `s_mm`):**

`progress_s [mm] = 1000 * dot( r_h(t) - r_h(0), u_hat )`

with `u_hat` from **that trial's** `g_h` x–z. This is a progress coordinate (gravity-aligned motion is positive). It is **not** a fixed signed axis. Directional claims use `Delta r_h` in the live hand frame. See `VIEWER_VALIDATION.md`.

## RAW TRAJECTORY EVIDENCE

### Baseline (vertical, τ=-18)

```json
{
  "slope_rh_z_mm_s": -0.0017516511546337176,
  "s_end_mm": -9.268177907768664e-05,
  "rho_med": 0.2336180558075664,
  "nL_end": 7,
  "nR_end": 6,
  "regime": "STICK",
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
  }
}
```

### Phase table (angle × τ)

| angle_cmd | wrist_act | g_hx | |g_xz| | τ | ρ_med | ρ_max | s_end mm | t_bil | t_loss | regime |
|---|---|---|---|---|---|---|---|---|---|---|
| 30 | 29.8 | -4.876 | 9.809 | -18.00 | 0.193 | 0.238 | 0.14 | 1.000 | None | STICK |
| 30 | 29.8 | -4.876 | 9.809 | -8.00 | 0.354 | 0.403 | 0.14 | 1.000 | None | STICK |
| 30 | 29.8 | -4.876 | 9.809 | -5.00 | 0.743 | 1.000 | 0.40 | 1.000 | None | STICK |
| 30 | 29.8 | -4.876 | 9.809 | -3.00 | 0.639 | 1.000 | 4.49 | 1.000 | None | CONTROLLED_SLIP |
| 30 | 29.8 | -4.876 | 9.809 | -2.00 | 1.000 | 1.000 | 11.66 | 0.272 | 0.037999999999995815 | LOSS |
| 30 | 29.8 | -4.876 | 9.809 | -1.00 | 1.000 | 1.000 | 75.14 | 0.012 | 0.013999999999998458 | LOSS |
| 30 | 29.8 | -4.876 | 9.809 | -4.33 | 0.449 | 1.000 | 2.66 | 1.000 | None | CONTROLLED_SLIP |
| 30 | 29.8 | -4.876 | 9.809 | -3.67 | 0.512 | 1.000 | 3.70 | 1.000 | None | CONTROLLED_SLIP |
| 30 | 29.8 | -4.876 | 9.809 | -2.50 | 0.884 | 1.000 | 6.06 | 0.242 | 0.25199999999997225 | LOSS |
| 45 | 44.7 | -6.907 | 9.809 | -18.00 | 0.178 | 0.235 | 0.19 | 1.000 | None | STICK |
| 45 | 44.7 | -6.907 | 9.809 | -8.00 | 0.416 | 0.684 | 0.22 | 1.000 | None | STICK |
| 45 | 44.7 | -6.907 | 9.809 | -5.00 | 0.659 | 0.997 | 0.27 | 1.000 | None | STICK |
| 45 | 44.7 | -6.907 | 9.809 | -3.00 | 0.647 | 1.000 | 3.98 | 1.000 | None | CONTROLLED_SLIP |
| 45 | 44.7 | -6.907 | 9.809 | -2.00 | 1.000 | 1.000 | 99.77 | 0.322 | 0.12999999999998568 | LOSS |
| 45 | 44.7 | -6.907 | 9.809 | -1.00 | 1.000 | 1.000 | 100.58 | 0.012 | 0.013999999999998458 | LOSS |
| 45 | 44.7 | -6.907 | 9.809 | -4.33 | 0.423 | 1.000 | 2.59 | 1.000 | None | CONTROLLED_SLIP |
| 45 | 44.7 | -6.907 | 9.809 | -3.67 | 0.517 | 1.000 | 3.71 | 1.000 | None | CONTROLLED_SLIP |
| 45 | 44.7 | -6.907 | 9.809 | -2.50 | 0.811 | 1.000 | 3.84 | 1.000 | None | CONTROLLED_SLIP |
| 60 | 59.7 | -8.471 | 9.808 | -18.00 | 0.225 | 0.294 | 0.24 | 1.000 | None | STICK |
| 60 | 59.7 | -8.471 | 9.808 | -8.00 | 0.658 | 1.000 | 0.31 | 1.000 | None | STICK |
| 60 | 59.7 | -8.471 | 9.808 | -5.00 | 0.363 | 0.957 | 0.40 | 1.000 | None | STICK |
| 60 | 59.7 | -8.471 | 9.808 | -3.00 | 0.679 | 1.000 | 3.52 | 1.000 | None | CONTROLLED_SLIP |
| 60 | 59.7 | -8.471 | 9.808 | -2.00 | 1.000 | 1.000 | 120.71 | 0.162 | 0.17199999999998106 | LOSS |
| 60 | 59.7 | -8.471 | 9.808 | -1.00 | 1.000 | 1.000 | 120.85 | 0.012 | 0.017999999999998018 | LOSS |
| 60 | 59.7 | -8.471 | 9.808 | -4.33 | 0.426 | 1.000 | 1.96 | 1.000 | None | CONTROLLED_SLIP |
| 60 | 59.7 | -8.471 | 9.808 | -3.67 | 0.565 | 1.000 | 3.28 | 1.000 | None | CONTROLLED_SLIP |
| 60 | 59.7 | -8.471 | 9.808 | -2.50 | 0.861 | 1.000 | 5.60 | 0.302 | 0.235999999999974 | CONTROLLED_SLIP |

Useful controlled-slip diagnostic (not task success): bilateral ≥ 0.30 s and |Δ| ≥ 1.0 mm, no drop.

Controlled-slip cells: **11**. Stick: **9**. Loss: **7**.
Authority: **REGION PRESENT**.

### Directional control (±tilt)

`progress_s` on both trials is ~+4.5 mm by construction of `u_hat`. **Fixed-frame** (same `r_h` convention):

| trial | g_h.x | g_h.z | Δr_h.x [mm] | Δr_h.y [mm] | Δr_h.z [mm] |
|---|---|---|---|---|---|
| +30° | −4.90 | +8.50 | **−2.25** | +0.03 | +3.88 |
| −30° | +4.89 | +8.50 | **+2.25** | +0.02 | +3.91 |

`g_h.x` flips; `Δr_h.x` flips with it. `g_h.z` stays positive and `Δr_h.z` stays ~+3.9 mm (distal). `dot(Δr_h, g_h_xz) > 0` on both trials (cos ≈ 1). Full JSON: `VIEWER_VALIDATION.md`.

### Gravity causality (1g vs 0g)

```json
{
  "angle": 30.0,
  "tau": -3.0,
  "g1": {
    "s_end_mm": 4.490351550008682,
    "v_rel_med": 0.006778603213435494,
    "rho_med": 0.6393676203028444,
    "regime": "CONTROLLED_SLIP"
  },
  "g0": {
    "s_end_mm": 0.044121826667492536,
    "v_rel_med": 0.002882767511639409,
    "rho_med": 0.21257317501199707,
    "regime": "STICK"
  },
  "ok": true
}
```

### noslip=1 vs noslip=0 on selected candidate

```json
{
  "angle": 30.0,
  "tau": -3.0,
  "ns1": {
    "s_end_mm": 4.490351550008682,
    "rho_med": 0.6393676203028444,
    "regime": "CONTROLLED_SLIP",
    "v_rel_med": 0.006778603213435494
  },
  "ns0": {
    "s_end_mm": 5.415432299509796,
    "rho_med": 0.7813856232332609,
    "regime": "CONTROLLED_SLIP",
    "v_rel_med": 0.008729228770394783
  },
  "ok": true
}
```

### SLIP → SECURE brake

```json
{
  "angle": 30.0,
  "tau_slip": -3.0,
  "t_brake": 0.31999999999996476,
  "s_end_mm": 1.2232274253238395,
  "additional_mm_after_brake": 0.056841787054214166,
  "v_rel_med_late": 5.7944012540440325e-05,
  "rho_med": 0.5256268253754621,
  "nL_end": 8,
  "nR_end": 8,
  "dropped": false,
  "t_both_loss": null,
  "ok": true
}
```

### Small robustness (fixed τ, no per-condition retune)

```json
[
  {
    "mass": 0.18,
    "mu": 1.0,
    "regime": "CONTROLLED_SLIP",
    "s_end_mm": 3.980193576835248,
    "rho_med": 0.557724308420315,
    "t_bil_from_start": 0.9999999999998899
  },
  {
    "mass": 0.2,
    "mu": 1.0,
    "regime": "CONTROLLED_SLIP",
    "s_end_mm": 4.490351550008682,
    "rho_med": 0.6393676203028444,
    "t_bil_from_start": 0.9999999999998899
  },
  {
    "mass": 0.22,
    "mu": 1.0,
    "regime": "CONTROLLED_SLIP",
    "s_end_mm": 4.814443664330348,
    "rho_med": 0.7182569555235871,
    "t_bil_from_start": 0.9999999999998899
  },
  {
    "mass": 0.2,
    "mu": 0.9,
    "regime": "CONTROLLED_SLIP",
    "s_end_mm": 4.088016884691398,
    "rho_med": 0.7263877985856145,
    "t_bil_from_start": 0.9999999999998899
  },
  {
    "mass": 0.2,
    "mu": 1.1,
    "regime": "CONTROLLED_SLIP",
    "s_end_mm": 4.63420680954515,
    "rho_med": 0.5841371561104655,
    "t_bil_from_start": 0.9999999999998899
  }
]
```

## USER VISUAL OBSERVATION

- **CONTROLLED SLIP: CONFIRMED** (user, `--viewer slip`).
- **SLIP → BRAKE: PENDING** (watch `--viewer brake_visual`; do not treat `--viewer brake` 1.2 mm as a visual fail).

Default camera: one-shot safe oblique; mouse rotate/pan/zoom. Overlay starts with `PAUSED — adjust camera` for 2.5 s (no physics).

```text
python training/gravitational_reposition_primitive_v2.py --viewer slip
python training/gravitational_reposition_primitive_v2.py --viewer brake
python training/gravitational_reposition_primitive_v2.py --viewer brake_visual
```

Optional whole-robot framing: `--camera robot`.

Details: [`VIEWER_VALIDATION.md`](VIEWER_VALIDATION.md).

## INTERPRETATION

CONTROLLED_SLIP occupies a region: angles [30.0, 45.0, 60.0], τ [-4.33, -3.67, -3.0] (and −2.5 at 45°; 60° −2.5 is an edge). `progress_s` is not directional evidence. Fixed-frame Δr_h.x reverses with g_h.x at ±30°. 0g removes the 1g reposition. High-ρ slip remains at noslip=0 and 1. SLIP→SECURE: relative motion stopped without ejection under the measured brake window.

Not a recovery-policy verdict. Disturbed s=2.0 was not tested.

## Unresolved

- `t_bil_from_start` is computed on 10 ms logs; `t_loss` is every physics step. **60° τ=-2.50** is CONTROLLED_SLIP in the table (`t_bil=0.302`) but `t_loss=0.236`. Treat that cell as a **LOSS-adjacent edge**, not as interior of the region.
- Frozen RULE still uses τ_slip=-2, which is **LOSS** in this simple airborne tilt (30/45/60°). This report does not retune RULE.
- USER VISUAL: slip **CONFIRMED**; slip→brake **PENDING** (`--viewer brake_visual`).
- No return-to-vertical, lift, or s=2.0 application.
