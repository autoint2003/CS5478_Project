# Cylinder vs box: nominal gravity-loaded pinch creep

Diagnostic only. Official `assets/scene.xml` was not modified. Recovery, disturbance, reward, and training were not touched. The box exists only as an in-memory `geom_type` swap on the same object body after the official model is loaded.

Script: `training/box_vs_cylinder_creep.py`  
Raw: `results/diagnostics/box_vs_cylinder_creep/raw/`  
Figures: `results/diagnostics/box_vs_cylinder_creep/figures/`

This report separates **VERIFIED CODE FACT**, **RAW TRAJECTORY EVIDENCE**, **USER VISUAL OBSERVATION**, and **INTERPRETATION**.

---

## Construction (VERIFIED CODE FACT)

Same Panda, pads, tendon motor `actuator8`, freeze-here hold (`p_des:=p_hand`, `r_des:=R_hand`, `v_cmd:=0`, `tau=-18`, 10 s), `m=0.20 kg`, pair sliding `mu=1.0`.

Official cylinder geom: `type=cylinder`, size `(radius=0.018, half-height=0.03)`, `condim=4`, `solref=(0.01,1)`.

Diagnostic box (runtime only): `type=BOX`, half-extents **`(0.018, 0.018, 0.03)` m**. Grasp faces are **36 mm** apart (same as the cylinder diameter), parallel to the finger pads (pinch along world X at identity / nominal grasp). Height matches the cylinder. **Same** object-geom `solref`, `solimp`, `condim`, `friction`. Body mass/inertia unchanged except the usual 0.20 kg reset.

Live mixed contact `solref` remains `[0.015, 1.0]`, `condim=4`, `mu=1.0` on both shapes.

Sequence: centered object → normal close → airborne lift → freeze-here 10 s. No recovery.

---

## Main table (RAW)

Slope = robust Theil–Sen of `r_h.z` over **1–10 s**. `+hand-z` is distal (fingertips).

| condition | cylinder slope (mm/s) | cylinder Δ 10 s (mm) | box slope (mm/s) | box Δ 10 s (mm) | box/cyl |
|---|---|---|---|---|---|
| **1 g baseline** | **1.220** | 12.08 | **0.774** | 7.73 | **0.63** |
| **0.5 g baseline** | **0.611** | 6.08 | **0.387** | 3.86 | 0.63 |
| **0 g** | **−0.002** | 0.00 | **0.000** | 0.00 | — |
| **harder 1 g** (`solref[0]=0.005`) | **0.385** | 3.81 | **0.260** | 2.59 | 0.68 |

Both shapes: 0.5 g / 1 g = **0.50**. Harder / baseline ≈ **0.32** (cylinder) and **0.34** (box). Zero g removes distal creep for **both**.

Figures: `fig_rh_z.png`, `fig_harder.png`, `fig_ncon.png`.

---

## Per-cell details (RAW)

| cell | aperture Δ mm | Fn L/R (N) | Ft sum (N) | rho mean/max | nL / nR | centroid z drift L/R mm | geom-pair births/s | \|ma−(Fc+Fg)\| |
|---|---|---|---|---|---|---|---|---|
| cylinder 1g | +0.011 | 9.07 / 9.10 | 1.96 | 0.113 / 0.224 | 8–14 / 8–15 | −1.55 / −0.63 | **0** | 0.0046 |
| box 1g | ~0 | 8.98 / 9.02 | 1.96 | 0.109 / 0.113 | **16 / 16** | ~0 / ~0 | **0** | 1e-6 |
| cylinder 0.5g | ~0 | ~9.08 | ~0.98* | ~0.11 | 8–15 | small | 0 | ~0.002 |
| box 0.5g | ~0 | ~9.00 | ~0.98* | ~0.11 | 16 / 16 | ~0 | 0 | small |
| cylinder 0g | ~0 | ~9.08 | ~0 | — | bilateral | — | 0 | ~0.005 |
| box 0g | ~0 | ~9.00 | ~0 | — | 16 / 16 | — | 0 | small |
| cylinder harder | +0.10 | ~9.07 | ~1.96 | ~0.11 | 7–16 | small | 0 | 0.007 |
| box harder | small | ~9.00 | ~1.96 | ~0.11 | 16 / 16 | ~0 | 0 | small |

\*Ft sum tracks weight (`mg` ≈ 1.96 N at 1 g, ~0.98 N at 0.5 g). Exact per-cell numbers: `raw/*.json`.

**Contact-set:** geom-pair identity is **stable for 10 s** on both shapes (8 unique pairs, zero births/deaths). Position-hashed (0.1 mm) keys still turn over; that is sliding of the same pairs, not pair regeneration.

Box contacts are **more numerous and frozen in count** (16/16) with **no centroid-z walk**. Cylinder counts flicker 8–15 and the centroid walks ~1 mm. The box **still creeps 0.77 mm/s**.

Force balance is valid on all reported cells.

---

## USER VISUAL OBSERVATION

Not recorded here. Commands (full grasp + 10 s hold):

```
python training/box_vs_cylinder_creep.py --view cylinder --g 1 --playback-speed 0.35
python training/box_vs_cylinder_creep.py --view box --g 1 --playback-speed 0.35
python training/box_vs_cylinder_creep.py --view box --g 0 --playback-speed 0.35
```

---

## INTERPRETATION

**D. Both scale with gravity but at different coefficients.**

- Not **C**: the box is not ~0 at 1 g.
- Not **A** (identical rates): the cylinder is ~**1.6×** faster than the box.
- Not a “box kills creep” geometry bug: flat parallel faces still show clear gravity-loaded distal creep, the same 0.5× scaling at 0.5 g, the same collapse at 0 g, and the same ~3× drop when only `solref[0]` is shortened.

**Hypothesis:** generic **gravity-loaded frictional-contact regularization** is sufficient to produce the phenomenon. **Cylinder curvature is not required**, but it **amplifies** the rate (~60%) and adds contact-count/centroid flicker. Geom-pair churn is not the driver on either shape.

Not a hardware claim. Not a recommended `solref`. Official model unchanged. Stop.
