# Quarantine: historical ballistic-impact claims

These files are **not** authoritative under the frozen `noslip_iterations=1`
contact model with a correctly realized ball mass.

Known issue: `assets/scene_impact_diag.xml` defined ball XML mass **0.02 kg**,
while some runtime paths called `set_ball_mass(0.50)` without a consistently
verified inertia/`qM` recomputation. Claims involving 0.5 kg, 8.5–10 m/s,
impact severity, and delayed failure from those runs must not be reused.

Do not delete the files. Do not treat them as calibrated evidence.

Historical (superseded for this diagnostic):

- `assets/scene_impact_diag.xml` (XML mass 0.02 kg)
- `training/write_impact_scene.py`
- `config/impact_demo.yaml` (`ball_mass: 0.35`)
- `training/impact_demo_core.py`, `training/visualize_impact_recovery.py`
- `results/eval_sets/impact_severity_four.npz`
- `results/diagnostics/noslip_calibration/raw/impact.json`

The replacement diagnostic is:

`results/diagnostics/ballistic_impact_noslip1/`
