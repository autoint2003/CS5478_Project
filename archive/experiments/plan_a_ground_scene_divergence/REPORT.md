# REPORT — Plan A elevated vs GROUND_SCENE divergence

Expert focus: `hrv1_920031` (+ return-aware variant). No new experts, no controller retune, no vocabulary/threshold changes.

## 1. What GROUND_SCENE=FAIL meant (prior audit)

Prior audit raw traces (`results/diagnostics/plan_a_sliding_pivoting_legality/raw/`):

| Expert | Scene | Replay OK | Grasped / SUCCESS_NEW | Axial mm | Axis ° | Bilateral gap s | Contact gate | Sliding | Pivoting |
|---|---|---|---|---|---|---|---|---|---|
| hrv1_920031 | elevated | yes | **SUCCESS_NEW True** | −7.48 | ~8.0 | **0.020** PASS (`< 0.02`) | PASS | PASS | PASS |
| hrv1_920031 | ground (old transfer) | yes | **SUCCESS_NEW True** | −7.38 | ~5.4 | **0.040** FAIL | FAIL | FAIL* | FAIL* |
| hrv1_920031_return_aware | elevated | yes | **SUCCESS_NEW True** | −6.66 | ~8.6 | **0.020** PASS | PASS | PASS | PASS |
| hrv1_920031_return_aware | ground (old transfer) | yes | **SUCCESS_NEW True** | −6.58 | ~6.1 | **0.040** FAIL | FAIL | FAIL* | FAIL* |

\*Sliding/pivoting FAIL were **gated by contact_preservation**, not absence of axial/pivot motion. Ungated axial displacement remained ~7 mm on ground.

**SUCCESS_NEW and mechanism gates are different outcomes.** Prior “GROUND_SCENE=FAIL” meant mechanism-gap FAIL while full success still passed.

## 2. Call graph / code reuse

See `CALL_GRAPH_COMPARISON.md`.

- Shared: `step_legal`, `map_action`, TRACK Cartesian, K49 decode, `analyze_mechanism`.
- Separate sim classes: `BallisticSim` vs `GroundBallisticSim` (same Recovery executor).
- Prior divergence source: **incomplete `transfer_state_to_ground`** vs elevated `restore_capture` (missing `qacc_warmstart`, `v_des`/`w_des`, mocap/weld, conditional ball park).

## 3. Compiled models

See `COMPILED_MODEL_DIFF.json`.

- `opt.timestep`, gravity, solver: match.
- Support geoms only: elevated `table`+`floor` vs ground `ground` (expected).
- Sole “unexpected” named-body field: `body.object.body_pos` keyframe rest (0.43 vs 0.03) — **overridden by transferred qpos**; not a live dynamics parameter once state is restored.
- No unexpected robot/cylinder/actuator/finger friction or solref diffs among shared named geoms.

## 4. Initial-state equivalence

See `INITIAL_STATE_DIFF.json`.

| Transfer | qpos/qvel/ctrl | qacc_warmstart | FSM p_des / cmds | Bilateral after `mj_forward` |
|---|---|---|---|---|
| Old / incomplete | ball park overwrites qpos | **mismatch (~9.81)** | partial (no v_des/w_des) | yes |
| **Complete (fixed)** | **identical** | **0 diff** | full | yes |

## 5. Paired replay

See `PAIRED_STEP_TRACE.csv`, `FIRST_DIVERGENCE.md`.

- Identical decoded 7D actions via shared `step_legal`.
- **Complete transfer: no measured divergence** across the full tape (`first_comp = null`); mechanism metrics byte-match elevated.
- Elevated table never contacts object/robot during the Plan A tape (`CONTACT_COMPARISON.csv`; `table_ever = false`).
- Isolation B (table collisions disabled on elevated): mechanism still PASS — table not required for Plan A contact manip.

## 6. Physics vs audit-gate

Offline shared `analyze_mechanism` on paired traces:

- Prior FAIL: audit **gap gate** (`bilateral_gap_s < 0.02`) with kinematics still sliding-positive.
- After complete transfer: gaps match (0.020) and all mechanism gates PASS on both scenes.

## 7. Fix

`evaluation/plan_a_mechanism_audit.transfer_state_to_ground` now copies warmstart, full FSM desires, mocap/weld, and parks the ball only when elevated is parked — aligning with `restore_capture` physics fields. Thresholds and expert tapes untouched.

## 8. Direct answers

1. **Same legal actions executed?** Yes — shared K49 decode + `step_legal`.
2. **Identical initial physical/controller state?** Not under the old transfer; **yes after the fix** (warmstart + FSM + qpos/qvel/ctrl).
3. **Models identical except support?** Yes for dynamics-relevant shared named bodies/geoms/actuators; support geoms differ; object keyframe rest z differs but is overridden by qpos.
4. **Did elevated table contribute to Plan A sliding/pivoting?** **No** — zero object–table/floor contacts; disabling table collisions preserves PASS.
5. **First measured divergence?** Old transfer: step-0 ball/warmstart mismatch. Complete transfer: **none**.
6. **Cause?** **Initialization / incomplete snapshot transfer** (not a separate controller, not support contact, not weakened success). Gap gate amplified a thin contact-continuity margin.
7. **Code change?** Complete `transfer_state_to_ground` (warmstart + FSM desires + mocap/weld + conditional park).
8. **Does `hrv1_920031` now PASS sliding, pivoting, contact continuity, and full SUCCESS_NEW on GROUND_SCENE?** **Yes** under fixed transfer (return-aware likewise). Exact elevated replay remains PASS; ground is matched-state transfer replay of the same tape.

Regression: `tests/test_plan_a_paired_scene_regression.py`.
