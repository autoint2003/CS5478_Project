# Precision Oracle — Implementation Gap Audit

**Spec authority:** [`ORACLE_STATE_MACHINE_SPEC.md`](ORACLE_STATE_MACHINE_SPEC.md)  
**Primary sources inspected:**

- `data_generation/oracles/precision/unified_precision_oracle.py`
- `data_generation/oracles/precision/precision_dx_support_transfer.py`
- `data_generation/oracles/precision/endpoint_only_axial_roll.py`
- `data_generation/oracles/precision/precision_dz.py`
- `data_generation/oracles/precision/contact_modes.py`
- `data_generation/oracles/precision/support_aware_rotation.py`
- `data_generation/oracles/precision/precision_types.py`
- `data_generation/oracles/precision/precision_goals.py`
- Historical: `results/diagnostics/precision_oracle_unified_flow/{STATE_MACHINE,IMPLEMENTATION_AUDIT,REPORT}.md`
- Failure context: `hard_910026` DX1/DX2 (Fn_z≈0 at DOWN; DX2 `contact_loss` on bilateral abort → silent verified_lateral)

Compliance: **MATCH** / **PARTIAL** / **MISSING** / **INCORRECT**

---

## 1. Global scheduler (`run_precision`)

| Item | Detail |
|---|---|
| **Intended** | Measure residuals → gate → optional direct HANDOFF → dθ→dx→dz priority with skip + reassess; posture restore; frozen HANDOFF; explicit fail reasons |
| **Current** | `unified_precision_oracle.run_precision` (~L400–868) |
| **Entry guard** | `entry_gate` / `hold_and_confirm_stable`; `assess_need` / `decide_precision` |
| **Completion** | Loop until AXIS needs clear or budget; then `prepare_stable_handoff` + `evaluate_handoff_with_axis` |
| **Contact/force** | Gate uses bilateral + v_rel; modes logged partially via ST/dz |
| **Failure** | `PRECISION_PRIMITIVE_FAILED`, `PRECISION_EXECUTED_HANDOFF_FAIL`, skip success |
| **Compliance** | **PARTIAL** |
| **Required change** | (1) Default-disable silent `correct_dx_verified` after ST fail. (2) Ensure ST fail returns explicit phase reason without legacy substitute. (3) Keep HANDOFF fail distinct from `NO_REMAINING_CORRECTION`. |

### Transition table (scheduler)

| Transition | Compliance | Notes |
|---|---|---|
| Direct HANDOFF skip | MATCH | `probe_direct_handoff` |
| Skip unnecessary dθ/dx/dz | MATCH | `assess_need` AXIS flags |
| Reassess after correction | MATCH | `continue` after each cycle |
| Geometry ≠ HANDOFF success | MATCH | status `HANDOFF_FAIL` + `handoff_NEW_fail` |
| Silent verified_lateral after ST fail | **INCORRECT** | `use_verified_lateral_fallback=True` default; L597–629 |
| Legacy horizontal when ST enabled | MATCH (guarded) | Only if `use_support_transfer_dx` False |
| dz legacy DualLoop fallback | PARTIAL | `use_goal_driven_slip=False` retry — document, do not expand |
| restore_DOWN when no dz | MATCH | `restore_DOWN_exit` L796–807 |
| S1 pass dx→dz | MATCH | `s1_state` + `correct_dz_*(..., s1_state=)` |
| Clear S1 after dθ/dz | MATCH | |

---

## 2. dtheta (`correct_dtheta`)

| Item | Detail |
|---|---|
| **Intended** | Valid contact Pivot; topology; brake; DOWN restore; θ alone ≠ success |
| **Current** | `unified_precision_oracle.correct_dtheta` L144–245; pivot tags / `primitive_contact_pivot`; `restore_fingertips_down` |
| **Entry** | `cfg.pivot_tag` or book search |
| **Completion** | `pivot_ok` ∧ (DOWN or near-DOWN α≥160° + bilateral) |
| **Contact/force** | Bilateral after settle required for `pivot_ok` |
| **Failure** | `pivot_not_improved`, `dtheta_restore_DOWN_failed`, search fail |
| **Compliance** | **PARTIAL** |
| **Required change** | Preserve contract. Unresolved: some ICs lose bilateral during settle (Plan A) — diagnose, do not skip restore. |

---

## 3. DX1 — Establish S1 bias

| Item | Detail |
|---|---|
| **Intended** | Mechanically load-bearing single-sided S1 **designation** + topology before DX2 |
| **Current** | `precision_dx_support_transfer.correct_dx_support_transfer` DX1: optional tilt to DOWN + `run_roll_phase` ~12°; checks `p1.ok and bil1` only |
| **Entry** | Fingertips DOWN (prep tilt if needed) |
| **Completion** | Bilateral after modest roll — **does not** require `one_sided_mechanical` |
| **Contact/force** | Logs mode; note admits full one-sided appears under DX2 gravity |
| **Failure** | `mechanical_support_not_established` if roll/bilateral fail |
| **Compliance** | **PARTIAL** (physically honest at DOWN; under-specified vs “verified S1”) |
| **Required change** | After DX1: require designated `support_side` (from roll_sign / live forces). Optionally micro-tilt until `one_sided_mechanical` **or** explicitly document DX1 as “S1-designated” and force-verify in early DX2 before continuing. Fail closed if side cannot be designated. |

**hard_910026 note:** `Fn_z≈0` at DOWN is expected by geometry; treating that alone as DX1 failure without designation logic is incomplete. Real bug was proceeding into DX2 with bilateral-abort tilt that contradicts S1, then silent fallback.

---

## 4. DX2 — Protected horizontal rotation

| Item | Detail |
|---|---|
| **Intended** | S1-protected tilt to horizontal; monitor support-side contact; verify force-backed S1 |
| **Current** | `run_tilt_phase(..., contact_aware=True)` then require `bil2` and `s1_at_horiz = one_sided OR bilateral` |
| **Entry** | After DX1 bilateral roll — **no** force S1 gate |
| **Completion** | Near-horizontal + (one_sided **or any bilateral**) |
| **Contact/force** | **INCORRECT guard:** `run_tilt_phase` aborts on `not st2["bilateral"]` (L308–310), which conflicts with intentional one-sided load transfer |
| **Failure** | `support_lost_during_rotation` if `not bil2`; else `horizontal_unreachable` |
| **Compliance** | **INCORRECT** |
| **Required change** | Add `support_guard` + `support_side` to `run_tilt_phase`: abort on support-side force collapse / latch / free-fall, **not** on transient one-sided grasp. Require `one_sided_mechanical` (or verified support Fn_z) at horizontal exit — not “any bilateral”. Do not start full DX2 without DX1 side designation. |

---

## 5. DX3 — S2 bilateral friction

| Item | Detail |
|---|---|
| **Intended** | Establish and verify S2 before sliding |
| **Current** | `select_horizontal_sliding_roll` + `run_roll_phase` + `_score_horizontal_slide_config` |
| **Entry** | Post-DX2 horizontal |
| **Completion** | `p3.ok and sc.ok` |
| **Contact/force** | Score + mode log |
| **Failure** | `bilateral_friction_support_*` |
| **Compliance** | **MATCH** (retain; ensure not bypassed) |

---

## 6. DX4 — Controlled lateral sliding

| Item | Detail |
|---|---|
| **Intended** | Gravity-driven controlled slip; guard free slip; wrong-direction fail |
| **Current** | `run_goal_aware_axial_slip` with `SlipGoal(axis=dx)` |
| **Entry** | After S2 |
| **Completion** | Goal / travel; reject near-zero motion or wrong-way |
| **Contact/force** | DualLoop contact safety inside slip helper |
| **Failure** | `sliding_did_not_occur`, `sliding_wrong_direction`, slip codes |
| **Compliance** | **MATCH** / watch for uncontrolled slip thresholds |

---

## 7. DX5 — Brake / regrip

| Item | Detail |
|---|---|
| **Intended** | Early brake + regrip |
| **Current** | `settle_hold`; check `v_rel` and bilateral |
| **Compliance** | **MATCH** |

---

## 8. DX6 — Re-establish S1

| Item | Detail |
|---|---|
| **Intended** | Force-backed S1 for dz reuse; no DOWN restore |
| **Current** | Reverse S2 roll + reinforce roll; `make_s1_state`; `keep_s1_for_dz=True` |
| **Completion** | `ok` allows bilateral progress without requiring `s1.valid` |
| **Compliance** | **PARTIAL** |
| **Required change** | Prefer `s1.valid` / `one_sided_mechanical` when claiming reuse; if only bilateral, mark `s1_state.valid=False` and let dz re-establish S1 (already partially handled). |

---

## 9. dz outward (`correct_dz_outward`)

| Item | Detail |
|---|---|
| **Intended** | Case A: already DOWN → direct DualLoop; else safe path to DOWN then slide; no unnecessary ST |
| **Current** | `_near_down` → skip rotation; else S1 reuse / establish + tilt DOWN + slip |
| **Compliance** | **MATCH** |
| **Required change** | None for contract; keep Case A free of S1 |

---

## 10. dz inward (`correct_dz_inward`)

| Item | Detail |
|---|---|
| **Intended** | Reuse S1 or establish → UP → DualLoop → protected DOWN |
| **Current** | Case B/C/D with `_s1_valid` / `verify_mechanical_s1`; support-aware tilt |
| **Compliance** | **PARTIAL** |
| **Required change** | Unresolved: cold upright UP contact_loss on some ICs (gbi_063). Diagnose; do not invent substitute mechanism. Preserve S1 reuse path. |

---

## 11. HANDOFF / nominal continuation

| Item | Detail |
|---|---|
| **Intended** | Posture + contact + pose/vel; frozen nominal; SUCCESS_NEW/12S; fail explicit |
| **Current** | `prepare_stable_handoff` + `evaluate_handoff_with_axis`; status `PRECISION_EXECUTED_HANDOFF_FAIL` |
| **Compliance** | **MATCH** for status separation |
| **Required change** | Ensure tests assert geometric in-band + HANDOFF fail ≠ success. Do not weaken criteria. |

---

## 12. Legacy fallback branches

| Branch | Location | Compliance | Action |
|---|---|---|---|
| `correct_dx_verified` after ST fail | `run_precision` L597+ | **INCORRECT** (silent substitute) | Default **off**; if enabled, log `legacy_fallback_explicit` |
| `correct_dx_horizontal_attempt` | L564+ | OK only when ST disabled | Keep gated |
| dz `use_goal_driven_slip=False` retry | L663–669 | PARTIAL | Retain as last axial retry; log clearly |
| Pre-flowchart horizontal P7/P8 always DOWN | `precision_dx.py` | Contradicts ST exit | Do not call from flowchart path |

---

## 13. Contact mode classification

| Item | Detail |
|---|---|
| **Intended** | Force-backed modes drive transitions |
| **Current** | `contact_modes.classify_contact_mode` / `make_s1_state` (force_valid cannot upgrade bilateral-only) |
| **Compliance** | **MATCH** at classifier; **PARTIAL** at DX1/DX2 consumers |
| **Required change** | Wire DX2 guards to classifier / support_side |

---

## 14. Priority repair list (implementation order)

1. Explicit ST failure → `PRECISION_PRIMITIVE_FAILED` (no silent verified_lateral). Config default `use_verified_lateral_fallback=False`.
2. DX1: designate `support_side`; fail if undesignated.
3. `run_tilt_phase` support-side guard for DX2.
4. DX2 exit: require force-backed S1 under gravity (tighten `s1_at_horiz`).
5. DX6 / dz reuse: honor `s1.valid`.
6. Path replay tests for 12 contract paths (genuine MuJoCo snapshots).
7. Mark unresolved primitives (cold inward UP, some dtheta settle losses) without claiming full repair.

---

## 15. Unresolved / unsupported (do not claim fixed)

| Primitive / path | Diagnosis |
|---|---|
| Plan A dtheta settle contact_loss | Pivot tag retains θ improvement but bilateral lost in settle — restore never reached |
| Cold upright inward dz (no prior dx S1) | UP tilt contact_loss more often than post-dx Case C |
| Plan C gbi_063 inward | Historical `dz:contact_loss` — axial needed; S1→UP still drops |
| HANDOFF SUCCESS_NEW after long ST+dz | Mechanisms may succeed while `e_pos` slightly over working band — HANDOFF fail is correct, not ST failure |
| DX1 force `one_sided_mechanical` at pure DOWN | Often physically impossible (`Fn_z≈0`); designation + early-DX2 verification is the coherent contract |

---

## 16. Mapping summary

| Flowchart state | Status |
|---|---|
| ENTRY / GATE / ASSESS | MATCH |
| DIRECT HANDOFF | MATCH |
| dtheta + DOWN | PARTIAL |
| DX1 | PARTIAL |
| DX2 | INCORRECT |
| DX3–DX5 | MATCH |
| DX6 | PARTIAL |
| dz outward A | MATCH |
| dz inward B/C | PARTIAL |
| restore_DOWN exit | MATCH |
| HANDOFF eval | MATCH |
| Silent legacy dx fallback | INCORRECT |
