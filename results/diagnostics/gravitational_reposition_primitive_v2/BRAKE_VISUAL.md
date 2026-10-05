# Visual brake demo (NOT canonical)

**VISUAL DIAGNOSTIC — NOT CANONICAL BRAKE THRESHOLD.**

Canonical scientific brake remains 30°, τ=−3 → −18 at `|progress_s| ≥ 1.2 mm`.
Those files were not overwritten (`raw/brake.npz`, `raw/brake.json`, `raw/viewer_brake.npz`).

This demo delays the trigger to **3.0 mm** so the SLIP → BRAKE transition can be seen.
Command:

```text
python training/gravitational_reposition_primitive_v2.py --viewer brake_visual
```

## USER VISUAL OBSERVATION

SLIP → BRAKE: **PENDING** (user must watch `brake_visual`).

CONTROLLED SLIP (`--viewer slip`): **CONFIRMED**.

## RAW (headless dump, same physics path)

- t_brake = 0.708 s
- progress_s at brake = 3.008 mm
- Δr_h at brake [mm] = (−1.53, +0.03, +2.59)
- post-brake extra progress_s: +0.1 s → 0.016 mm; +0.5 s → −0.011 mm; +1.0 s → −0.001 mm; +2.0 s → 0.017 mm
- |v_rel| after brake: 3.58 → 0.44 → 0.09 → 0.03 mm/s
- nL/nR end = 4 / 8 (bilateral)
- obj_z end = 0.480 m
- dropped = false; t_loss = none; t_both_loss = none
- **arrested = true** → 3 mm is inside the braking basin

See `raw/brake_visual.json`.
