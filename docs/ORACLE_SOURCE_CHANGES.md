# Precision Oracle — Exact Source-Code Changes

Restore of the unified flowchart state machine (see `ORACLE_STATE_MACHINE_SPEC.md`).
No Stage-1 B/C experts, GRPO checkpoints, M111 vocabulary, or training datasets modified.

## Follow-up: mandatory DOWN before HANDOFF

| File | Change |
|---|---|
| `docs/ORACLE_STATE_MACHINE_SPEC.md` | `FINAL_DOWN_GATE` Mermaid; five canonical sequences; intermediate vs final eligibility |
| `precision_types.py` | `FAIL_POSTURE_DOWN_RESTORE`, `FAIL_POSTURE_DOWN_VERIFY`, `FAIL_STABLE_GRASP_PRE_HANDOFF` |
| `precision_handoff.py` | `posture_is_fingertips_down`, `run_final_down_handoff_gate` |
| `unified_precision_oracle.py` | No early HANDOFF while not DOWN; restore before dz outward; gate before all final HANDOFF |
| `experiments/precision/run_final_down_handoff_sequences.py` | Five-sequence MuJoCo replay |

---

## Files changed (initial restore)

### 1. `docs/ORACLE_STATE_MACHINE_SPEC.md` (new)

Authoritative Mermaid flowchart, coordinate conventions, DX1–DX6 / dz / HANDOFF contracts,
intended vs verified vs proposed layers.

### 2. `docs/ORACLE_IMPLEMENTATION_GAP_AUDIT.md` (new)

Code-to-spec mapping with MATCH / PARTIAL / MISSING / INCORRECT per state.

### 3. `data_generation/oracles/precision/precision_types.py`

- `use_verified_lateral_fallback` default **`False`** (was `True`).
- Comment clarifies: never silent substitute after ST failure; enable only for explicit baselines.

### 4. `data_generation/oracles/precision/endpoint_only_axial_roll.py`

- `run_tilt_phase` gains `support_guard`, `support_side`, `min_support_fn`, `max_vrel_support`.
- Under `support_guard=True` (DX2): abort on support-side force collapse / uncontrolled slip / latch — **not** on transient non-bilateral contact alone.
- Series logs include support force / mode when guarded.

### 5. `data_generation/oracles/precision/precision_dx_support_transfer.py`

- **DX1:** Designate `support_side` (live mode or roll_sign mapping); fail if undesignated / bilateral lost. Document that `Fn_z≈0` at pure DOWN is expected.
- **DX2:** Call tilt with `support_guard=True` and designated side; exit requires **force-backed S1** (`one_sided_mechanical` or Fn_z dominance), not “any bilateral”.
- Explicit `force_backed_s1_not_verified_at_horizontal` / `support_lost_during_rotation` reasons.

### 6. `data_generation/oracles/precision/unified_precision_oracle.py`

- On ST failure with fallback disabled: immediate `PRECISION_PRIMITIVE_FAILED` with note that silent `correct_dx_verified` is forbidden.
- When fallback explicitly enabled: phase `dx_verified_lateral_explicit`, log `legacy_fallback_explicit=True`, mechanism `verified_lateral_explicit`.

### 7. `experiments/precision/run_oracle_state_machine_path_tests.py` (new)

Physical replay harness for the 12 contract paths (MuJoCo snapshots, no fabricated forces).

### 8. `docs/ORACLE_SOURCE_CHANGES.md` (this file)

## Unchanged (by design)

- SUCCESS_NEW / SUCCESS_12S definitions
- Stage-1 experts, GRPO, M111, datasets
- DualLoop axial slip core, relative pivot tags
- `correct_dx_verified` / horizontal retained as **explicit** baselines only
