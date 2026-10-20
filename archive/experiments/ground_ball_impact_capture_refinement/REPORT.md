# REPORT — Ball-impact capture refinement

Prior pilot left unchanged under `results/diagnostics/ground_ball_impact_oracle_interception/`.

This run corrected metric validity, reanalyzed the four existing roots (priority `gbi_002`), and spent **305** counterfactual rollouts on distinct closing/braking and first-contact branches.

## Direct answers

1. **How many previously reported bilateral events were genuinely consecutive and load-bearing?**  
   **Zero** on reanalysis. Corrected metrics show `historical_ge_60ms = 0` and `continuous_ge_60ms_at_end = 0` for the default chase on all four roots. Across the closing search, maximum consecutive load-bearing bilateral contact was **0.02 s** (one policy step). The pilot’s “≥60 ms bilateral” reports were **accumulated / non-consecutive** (and/or non–load-bearing) artifacts — not maintained grasps.

2. **What physically caused `gbi_002` to lose capture?**  
   At the saved post-loss snapshot the fingers are essentially **closed** (`finger_opening ≈ 6.8 mm`), the cylinder COM is already **~41 mm distal past the pinch** (`p_rel_z ≈ 0.147` vs `PINCH_Z = 0.105`), and relative speed is **~0.42 m/s** downward with only **~0.24 s** until GROUND_DROP. Corrected label: **`jaws_closed_cylinder_already_distal_past_pinch`**.  
   The same mechanism applies to **`gbi_010` / `gbi_014` / `gbi_022`** (all jaws ≈ closed, distal +35–46 mm, airborne ≈0.24–0.26 s). Transient chase contacts are not held regrasps.

3. **Which closing/braking parameters improved bilateral capture?**  
   No parameter set produced continuous ≥60 ms bilateral at termination. Best observed consecutive bilateral in search: **0.02 s**. Open/close timing and post-contact braking therefore did **not** unlock persistent capture on these roots.

4. **Persistent bilateral recatch (`STABILIZED_REGRASP`)?** **No** (0 roots).

5. **Legal HANDOFF and full SUCCESS_NEW?** **No** (0 roots). Exact replay N/A (no SUCCESS_NEW tape).

6. **If capture still fails, cause class?**  
   **Contact geometry + insufficient time** (primary), not Stage-A rediscovery or vocabulary. All four roots share closed jaws + distal escape at the loss snap; legal open/chase cannot re-admit the cylinder before ground in ~0.24 s. Search strategy was corrected but cannot invent jaw clearance that the snapshot does not have.

7. **Best result independently replayable from the original post-impact snapshot?**  
   **No SUCCESS_NEW expert exists to replay.** Exact-replay plumbing (`exact_replay_actions`) recomputes SUCCESS_NEW independently when a tape exists (`EXACT_REPLAY.csv` empty of successes).

## Metric corrections (summary)

See `PILOT_METRIC_CORRECTIONS.md`.

- Consecutive vs historical bilateral; load-bearing `|Fn|≥0.05 N`.
- Stage-A proximity ≠ verified reachability.
- Explicit `ChaseParams` / contact-branch profiles (not longer identical chases).
- Pose-return sign corrected: `v_hand = R @ (-(P_TGT - p_rel))` (`POSE_RETURN_VALIDATION.md`).
- Stratified impact grid for **future** root gen; existing IDs preserved.

## Implication for scaling

These four post-loss snapshots are **poor recatch ICs**: loss is recorded after the cylinder has already left the jaw with the gripper still commanded closed.

Before more blind rollouts, new real-impact roots should snapshot at **first sustained bilateral loss while the object is still between the pads** (or arm Recovery open-at-loss without pre-impact anticipation), preserving ≥~0.15 s airborne **and** jaw-entry clearance.

**Priority result:** one physically genuine persistent airborne recatch was **not** obtained from the current roots; the blocker is now diagnosed, not mistaking a 60 ms contact accumulator for success.
