# REPORT — Future-informed dynamic airborne catch

Prior experiments preserved under `results/diagnostics/ground_ball_impact_*`.  
This run: **400/400** counterfactual rollouts (~138 s).

Joint sequence searched: **OPEN → INTERCEPT → VELOCITY MATCH → CLOSE WHILE FOLLOWING → SYNCHRONIZED BRAKE**.

No physics / friction / SUCCESS_NEW / GROUND_DROP / K49 / 39D / 20 ms interface changes. No SFT/GRPO.

## 1. Actuator measurements (measured, not datasheet assumptions)

| Quantity | Measured |
|---|---|
| Commanded TAU_OPEN / TAU_SECURE | +2.0 / −18.0 N·m |
| Commanded hand-frame v_z down (AIRBORNE_V3) | 3.8 m/s |
| Median finger open rate (max legal open, hold) | ≈ **0.29 m/s** |
| Median time open 0→0.03 m | ≈ **60 ms** |
| Median time open 0→0.04 m | ≈ **80–100 ms** |
| Max achieved hand speed (max-down command) | ≈ **2.48–2.53 m/s** (lags 3.8 m/s command) |
| Max lateral hand speed (a_y=+1) | ≈ **0.51 m/s** |

Source: `ACTUATOR_RESPONSE.csv`. Opening uses finger joint **qposadr**. Contact during open slows but does not block reaching ~0.04 m when the object is already escaping.

**Distinction:** 60–100 ms to useful opening is an **actuator/dynamics** measurement. It is not a hardcoded 80 ms heuristic phase.

## 2–7. Per-root catch outcomes

| root | remain→ground | jaw enter | open@t_c | v_rel@1st contact | max consec bil | mechanism | SUCCESS_NEW | limit |
|---|---|---|---|---|---|---|---|---|
| gbi_002 | 0.26 s | yes | 0.017* | 0.13 m/s | **0.76 s** | FULL_DETACH_ACTIVE_RECATCH | no | braking_or_handoff_slip |
| gbi_010 | 0.30 s | yes | 0.017* | 0.51 m/s | **0.74 s** | FULL_DETACH_ACTIVE_RECATCH | no | braking_or_handoff_slip |
| gbi_014 | 0.26 s | yes | 0.030 | 0.35 m/s | **0.74 s** | FULL_DETACH_ACTIVE_RECATCH | no | braking_or_handoff_slip |
| gbi_022 | 0.30 s | yes | 0.017* | 0.32 m/s | **0.74 s** | FULL_DETACH_ACTIVE_RECATCH | no | braking_or_handoff_slip |
| gradual lateral | 0.38 s | yes | 0.041 | — | 0.00 s | FAILED_OR_TRANSIENT | no | closing_ejection_or_miss |

\* `open_at_intercept` is sampled at predicted `t_c` after close-onset; peak opening during OPEN was higher (~0.028 m verified on gbi_002).

### Verified detach gap (gbi_002 expert tape)

Replay of capture actions from branch-A snap:

- Start: bilateral, opening ≈ 0.017 m.
- After open: **complete contact-free gap ≈ 40 ms** (no pad / no finger contact).
- Reacquisition: bilateral by t ≈ 80 ms; consecutive bilateral held ≈ **0.76 s**.
- Capture-tape exact replay: bilateral retained (3/3); SUCCESS_NEW handoff still `DROP_AFTER_HANDOFF`.

This is counted as **mechanism B** (`FULL_DETACH_ACTIVE_RECATCH`), not grasp preservation.

## Direct answers

1. **Is actual finger opening speed sufficient?**  
   **Yes** for these air times: open-to-0.03 m in ≈60 ms vs ≈0.26–0.30 s to ground. Limiting factor after open is geometry / relative velocity / close–brake, not opening rate.

2. **Can the hand reach the predicted interception with matching velocity?**  
   **Partially.** Reachability proxy (dist/`t_c` < 3.5 m/s) is true for many candidates. Achieved hand speed under max-down is ≈2.5 m/s (command 3.8). Best experts still place the cylinder in the jaw with v_rel as low as **0.13 m/s** (gbi_002).

3. **Does the cylinder enter the open jaw?**  
   **Yes** on all five roots (`jaw_entered=1` in `FULL_EXPERTS.csv`).

4. **Does velocity matching reduce collision-induced ejection?**  
   Best violent-root experts use match ∈ [0.7, 1.2] and achieve first-contact v_rel down to 0.13–0.51 m/s with persistent capture. Aggregate high-vs-low match over *all* jaw-entered trials is noisy (many failed closes). Evidence from successful experts: **matching enables low-impulse reacquisition**.

5. **Can closing-while-following produce persistent bilateral capture?**  
   **Yes** — four violent roots reach ≥60 ms and ≥150 ms consecutive bilateral (0.74–0.76 s).

6. **Can synchronized braking retain the cylinder?**  
   **Through the Recovery segment: yes** (0.74–0.76 s continuous with brake profiles). **Through HANDOFF/nominal lift: no** (`DROP_AFTER_HANDOFF`).

7. **Genuine full-detach recatch?**  
   **Yes on the four original violent roots** (`FULL_DETACH_ACTIVE_RECATCH`), with measured contact-free gaps then bilateral reacquisition.  
   **No** `POST_IMPACT_GRASP_PRESERVATION` experts in this search (open-first policy deliberately separates).  
   Gradual-slip lateral root: jaw entry without stable close (`closing_ejection_or_miss`).

8. **Full SUCCESS_NEW?**  
   **0 / 5**.

9. **Limiting constraint?**  
   For violent roots with stable Recovery capture: **braking_or_handoff_slip** — grasp survives synchronized Recovery braking but fails unchanged SUCCESS_NEW handoff/nominal lift.  
   For gradual lateral under open-first catch: **closing_ejection_or_miss**.  
   Not an opening-time impossibility (opening is fast enough).

## Mechanisms (kept distinct)

- **A. POST_IMPACT_GRASP_PRESERVATION** — not produced by this open-first dynamic catch search.
- **B. FULL_DETACH_ACTIVE_RECATCH** — four violent roots; complete separation then active reacquisition; ≥150 ms continuous bilateral; not SUCCESS_NEW.

## Replay

- Capture-tape bilateral: **True** (all replay trials for saved experts).
- SUCCESS_NEW replay: **False**.

## Artifacts

`ACTUATOR_RESPONSE.csv`, `INTERCEPTION_FEASIBILITY.csv`, `JAW_ENTRY.csv`,  
`CONTACT_RELATIVE_VELOCITY.csv`, `CLOSING_SEARCH.csv`, `BRAKING_SEARCH.csv`,  
`CAPTURE_STABILITY.csv`, `EXACT_REPLAY.csv`, `FULL_EXPERTS.csv`, `raw/`, `videos/`.
