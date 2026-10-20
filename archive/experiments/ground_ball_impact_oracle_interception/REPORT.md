# REPORT — GROUND_SCENE future-trajectory oracle (Plan C pilot)

Offline privileged oracle for ball-impact airborne recatch. No SFT/GRPO. No physics / controller / 39D / SUCCESS_NEW / K49 changes.

- Scene: `GROUND_SCENE`
- Library: `data_generation/strategies/ground_future_oracle.py`
- Driver: `scripts/run_ground_ball_impact_oracle_interception.py`
- Budget used: **323 / 500** counterfactual rollouts; **30** impact configs; **4** searchable roots

## Direct answers

1. **Real ball-impact roots with genuine grasp loss:** **5 / 30** configs (4 selected as recoverable-looking; 1 discarded for GROUND_DROP during impact).
2. **Airborne time after grasp loss:** ≈ **0.24–0.26 s** to GROUND_DROP under the legal non-chasing hold baseline (see `NO_INTERVENTION_FUTURES.csv`).
3. **Dynamically reachable interception candidates:** **103** Stage-A passes (authority-gap columns in `REACHABILITY.csv`).
4. **Genuine active airborne recatch?** **Partial yes** — useful finger contacts on all searched roots that passed A; **2** logged stable-bilateral events (≥60 ms). Not yet a maintained post-recatch grasp through pose/HANDOFF.
5. **Independent roots with full SUCCESS_NEW:** **0**.
6. **First unresolved bottleneck:** **Stage C/D bilateral capture stability** (`failed_capture_stabilization` on the best root `gbi_002`; majority roots stop at `no_bilateral_recatch`). Not impact generation and not Stage-A reachability.
7. **Continuous legal vs frozen K49:** Continuous 7D reached Stage B–D; **no** SUCCESS_NEW tape existed to fair-test K49. Continuous is currently necessary for feasibility search; vocabulary coverage is untested on winners.
8. **Missing capability before scaling:** A closing / relative-velocity / jaw-timing controller that converts first finger contact into **persistent** bilateral capture inside a ~0.25 s airborne budget, then legal pose return + HANDOFF. Secondary: longer airborne roots (higher / more lateral ejections) to relax timing.

## Success hierarchy (pilot)

| Gate | Result |
|---|---|
| Real ball–cylinder impact | 30/30 |
| Pre-impact stable bilateral | yes (after hand-raise + hand-frame install) |
| Post-impact genuine grasp loss | 5 |
| Active hand displacement toward escape | yes (Stage A/B) |
| Useful first finger contact | yes (87) |
| Genuine bilateral after loss | rare (1 Stage-C pass) |
| Stable post-recatch grasp | no (Stage D failed to hold) |
| Pose correction / HANDOFF / SUCCESS_NEW | 0 |

## Method notes

- Pre-impact: legal lift to airborne height, near-end cylinder install in the finger frame, settle with closed grip (no anticipatory Recovery chase).
- Impact: real free ball; Recovery actions = closed-grip hold only until loss.
- Loss labels distinguish transient unilateral vs sustained bilateral loss vs complete separation (`SUSTAIN_S = 20 ms`).
- Futures: every MuJoCo step under legal hold; oracle proposals use `p_COM + s·u` and `v_COM + ω×(s·u)`.
- Search: continuous legal 7D via `step_legal` / `_action_from`; SUCCESS_NEW via unchanged `apply_handoff` + nominal horizon.

## Deliverables

`IMPACT_ROOTS.csv`, `GRASP_LOSS_EVENTS.csv`, `NO_INTERVENTION_FUTURES.csv`, `INTERCEPTION_CANDIDATES.csv`, `REACHABILITY.csv`, `CONTACT_AND_CAPTURE.csv`, `SEARCH_FUNNEL.csv`, `FULL_EXPERTS.csv`, `FAILURE_ANALYSIS.md`, `raw/`, `videos/` (empty — no SUCCESS_NEW video).
