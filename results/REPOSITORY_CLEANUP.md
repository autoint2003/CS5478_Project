# Repository cleanup (contact-model correction freeze)

Inventory-first. Nothing was deleted merely because a result was negative.
No SAC artifacts were removed as “failed training.” Official XML, env, controllers,
spatial tactile, held-out offset set, and NoSlip raw evidence were protected.

Date of this cleanup: 2026-10-03.

---

## Deleted

| Path | Reason |
|---|---|
| `.pytest_cache/` | Generated pytest cache. Recreated by tests. Not scientific evidence. |
| `__pycache__/` / `*.pyc` | Bytecode. Already gitignored; removed if present on disk. |

No diagnostic `results/diagnostics/*` families were deleted.
No raw NoSlip / creep npz/json were deleted.
No evaluation npz files were deleted.

---

## Archived / superseded (kept on disk, not valid recovery authority)

These remain as **historical evidence of the noslip=0 low-ρ creep artifact**
or as incomplete/negative experiments. They must not be cited as physical
gravitational reposition or as recovery success.

Marker: `results/diagnostics/_SUPERSEDED_BY_NOSLIP.md`

| Path | Classification | Reason |
|---|---|---|
| `results/diagnostics/inward_slide_return/` | ARCHIVE / SUPERSEDED | 60° + τ=-18 inward slide at ρ≈0.10 was regularization creep; removed by noslip=1. Contact-mechanics raw logs still useful as artifact documentation. |
| `results/diagnostics/reposition_regrasp/` | ARCHIVE / SUPERSEDED | Secure-grip / unload sequences interpreted under noslip=0. |
| `results/diagnostics/wrist_sweep_recapture/` | ARCHIVE / SUPERSEDED | Wrist-sweep recapture claims used the old low-ρ tilt motion. |
| `results/diagnostics/wrist_recovery_cycle/` | ARCHIVE / SUPERSEDED | Full cycle mixed the superseded gravity-slide with later stages. |
| `results/diagnostics/recovery_false_positive/` | KEEP (history) | Negative result: “success” can be a false positive. Not junk. Not a noslip claim. |
| `results/diagnostics/active_regrasp/` | KEEP (incomplete) | Aborted active open-regrasp checkpoint. Negative/incomplete ≠ junk. |
| `training/inward_slide_return_audit.py`, `wrist_sweep_recapture.py`, `wrist_recovery_cycle_audit.py`, `reposition_regrasp_audit.py` | KEEP scripts | Needed to reproduce the **artifact**, not current authority. |
| Root `*_AUDIT.md` / `*_DIAGNOSTIC.md` listed in older git status | not present on disk at cleanup | If reintroduced, treat recovery-success claims as superseded unless re-run under noslip=1. |

---

## Quarantined (do not silently reuse)

| Path | Reason |
|---|---|
| `results/eval_sets/impact_severity_four.npz` | Historical impact held-out. Mass / `mj_setConst` semantics were later found questionable. **Deprecated as a scientific IC source.** Keep the file; do not train or retune against it. Offset set is still the primary held-out. |

---

## Retained

### Contact-model correction (KEEP — raw evidence for the freeze)

| Path | Reason |
|---|---|
| `results/diagnostics/nominal_grasp_creep_root_cause/` + `training/nominal_grasp_creep_root_cause.py` | Cylinder secure creep ~1.220 mm/s at noslip=0; 0g / dt / Newton / ideal pinch. |
| `results/diagnostics/box_vs_cylinder_creep/` + `training/box_vs_cylinder_creep.py` | Box ~0.774 mm/s vs cylinder; gravity scaling. |
| `results/diagnostics/noslip_calibration/` + `training/noslip_calibration.py` | noslip=1 selected; slip/release/impact preserved. |
| `results/diagnostics/contact_creep_sensitivity/` | D vs N: disturbed residual ≈ undisturbed floor. |
| `results/diagnostics/gravitational_reposition_primitive_v2/` | New high-ρ primitive under noslip=1. |

### Project implementation (KEEP)

`assets/scene.xml`, `assets/panda_torque.xml`, `envs/*`, `controllers/*`, `sensors/*`, `config/*` (except the documented noslip freeze in `config/sim.yaml`), `training/replay_core.py`, `training/demo_teleport_recovery_state.py`.

### Held-out / RL interface (KEEP)

`results/eval_sets/airborne_offset_eval.npz`  
`results/diagnostics/observable_*`, `tactile_ex_estimator`, `recenter_observability`  
Frozen RULE yaml/py.

### Historical logs/figures (KEEP, not canonical physics)

`results/logs/*` CSV/npz from earlier oracles, impact autopsies, lift-entry, SSR, etc. Negative or superseded **interpretations** stay; they are not duplicate junk. `results/figures/` recovery_diagnostic_*.png are old viewer stills, not tmp files.

`training/` diagnostic scripts that produced the KEEP reports stay. `training/active_regrasp_authority_audit.py` stays (incomplete).

Course PDFs in the repo root stay.

---

## Controllers / envs / sensors / config / training (inventory)

- `controllers/` — KEEP. Frozen RULE hashes unchanged.
- `envs/` — KEEP. `apply_solver_from_cfg` added for the noslip freeze only.
- `sensors/` — KEEP. Spatial tactile not deleted.
- `config/` — KEEP. `sim.yaml` now records noslip=1. Other physics YAML unchanged.
- `training/` — KEEP all `.py`. No abandoned one-off without a report was deleted in this pass.
- `results/logs/` and `results/figures/` — KEEP contents; gitignore already excludes many generated dirs.

Cleanup correctness is **classification + retention of evidence**, not “tests still pass.”
