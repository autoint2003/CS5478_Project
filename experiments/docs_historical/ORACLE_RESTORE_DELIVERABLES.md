# Precision Oracle Restore — Final Deliverables

## 1. Specification

- [`docs/ORACLE_STATE_MACHINE_SPEC.md`](ORACLE_STATE_MACHINE_SPEC.md) — Mermaid flowchart, coordinates, DX1–DX6 / dz / HANDOFF contracts

## 2. Gap audit

- [`docs/ORACLE_IMPLEMENTATION_GAP_AUDIT.md`](ORACLE_IMPLEMENTATION_GAP_AUDIT.md) — per-state MATCH / PARTIAL / MISSING / INCORRECT

## 3. Exact source changes

- [`docs/ORACLE_SOURCE_CHANGES.md`](ORACLE_SOURCE_CHANGES.md)

Key code edits:

| File | Change |
|---|---|
| `precision_types.py` | `use_verified_lateral_fallback=False` by default |
| `endpoint_only_axial_roll.py` | `run_tilt_phase(..., support_guard=)` for S1-protected DX2 |
| `precision_dx_support_transfer.py` | DX1 side designation; DX2 force-backed S1 exit |
| `unified_precision_oracle.py` | No silent `correct_dx_verified` after ST failure |

## 4. State-transition / physical replay

- Harness: `experiments/precision/run_oracle_state_machine_path_tests.py`
- Results: `results/diagnostics/oracle_state_machine_paths/{PATH_RESULTS.json,REPORT.md}`

Highlights (genuine MuJoCo):

| Path | Result |
|---|---|
| 02 dx ST + restore DOWN | ST ok; DX1 side=R; DX2 `s1_under_gravity=True`; ex≈11→−0.8 mm; NEW=False, 12S=True |
| 07 dθ→dx→dz S1 reuse | `dz_s1_reused=True`, `support_transfer_dx`, **SUCCESS_NEW=True** |
| Contract | Silent verified_lateral **disabled** and not used |
| hard_910026 (extra) | After repair: `support_transfer_dx` → **SUCCESS_NEW=True** (no silent fallback) |

## 5. Remaining unsupported

- [`docs/ORACLE_REMAINING_UNSUPPORTED.md`](ORACLE_REMAINING_UNSUPPORTED.md)

## 6. Out of scope (unchanged)

Stage-1 B/C experts, GRPO checkpoints, M111 vocabulary, training datasets, SUCCESS_NEW / SUCCESS_12S definitions.
