# Lateral Recatch → Handoff Stabilization

**Source catches:** `D:\NUS\CS5478 Intelligent Robots\CS5478_Project\results\diagnostics\lateral_offset_falling_recatch` (40 trajectories, 9 roots) — **unchanged** as prefixes  
**No larger offsets. No SFT/GRPO. No privileged Recovery / teleport / p_des reset.**

## Frozen criterion: `CATCH_HANDOFF_SUCCESS`

| Gate | Value |
|------|------:|
| Nominal hold | ≥ 2.0 s |
| Pose window | last 0.5 s |
| e_theta / e_pos | ≤ 25.0° / ≤ 20.0 mm |
| obj_z | ≥ 0.5 m |
| wrist vs R0 | ≤ 45.0° |
| vrel at handoff | ≤ 0.08 m/s |
| Sticky latch during catch | must be absent |

`SUCCESS_NEW` (≤10.0°/≤12.0 mm, hold 0.5s) evaluated **separately**, unchanged.

## Budget

| Phase | Alloc | 
|-------|------:|
| Audit | 120 |
| Stabilization | 330 |
| Full verify | 90 |
| Robustness | 60 |
| **Used / Total** | **468 / 600** |
| Wall (s) | 765.6 |

## Results

| Metric | Value |
|--------|------:|
| Independent roots audited | 9 |
| Raw catch → HANDOFF successes | 0 |
| Stabilization search CH hits | 2 |
| Triple-verified CATCH_HANDOFF roots | **2** |
| Found (pre-verify) | 2 |
| Timing-robust survivors | 6/12 |
| fine_down among CH hits | 2 (of 106 fine_down builds) |

### Raw HANDOFF failure classes (after probe-latch clear)

{
  "H_nominal_continuation_drop_or_escape": 7,
  "D_cylinder_position_error": 2
}

### Root causes (audit)

1. **Sticky table latch during scoop hold:** original catches continue `FINE_DOWN` near the table, briefly lose bilateral contact, and latch — `apply_handoff` then fails immediately under original SUCCESS_NEW/latched monitors.
2. **Catch ends ~table height** (obj_z≈0.43 m) — not a handoff-ready pose even when bilateral.
3. **IC snapshots** included probe-artifact latch from `make_ic` table-time measurement; cleared at episode start only (documented).

### What worked

Early **branch** of the unchanged catch prefix (before latch) + legal **lift / XY return / seat / fine_down** + `apply_handoff` + nominal continuation.

## Q1–Q12

1. **Why do the 40 catches fail nominal HANDOFF?**  
   Primarily **H** (sticky table latch + near-table endpoint) after a brief contact loss during late scoop hold; secondary **E** (orientation ~20° vs SUCCESS_NEW 10°) on early-branch+lift states that otherwise hold.

2. **Stable grasps or transient captures?**  
   **Both phases:** ≥0.5 s bilateral at catch bottom is real but **near-table / latch-contaminated**; early-branch+lift yields **stable elevated grasps** that survive multi-second nominal hold.

3. **Can legal continuous actions stabilize them?**  
   **Yes** — lift-before-latch stabilization (2 CH-rated search hits).

4. **Reposition while retaining bilateral contact?**  
   **Yes** — XY return toward IC hand pose during closed-grip lift without mandatory contact loss (when branched early).

5. **Return to nominal-compatible configuration?**  
   **Partial** — wrist can stay ≤45° (often ~11° with wy bias); hand remains ~translated vs original mid; full SUCCESS_NEW basin not generally recovered.

6. **Complete catch-to-handoff trajectory?**  
   **Yes — 2 triple-verified roots** under `CATCH_HANDOFF_SUCCESS`: `distal_off01_r8`, `perp_off01_r15`.  
   Unchanged **SUCCESS_NEW** also holds on **`perp_off01_r15`** (3/3 verify, e_theta≈4.6°, e_pos≈2.2 mm).  
   `distal_off01_r8` holds CH but not SUCCESS_NEW (e_theta≈20.5° > 10°).

7. **Independent release roots succeeding?**  
   **2** triple-verified CH roots (1 of them also SUCCESS_NEW).

8. **Does fine_down help?**  
   Used in 2/106 successful CH builds among fine_down trials — **candidate, not mandatory**.

9. **Failure dominance?**  
   **Grasp-near-table / latch (stability geometry)** first; then **pose/orientation vs SUCCESS_NEW**; controller target mismatch secondary (r_des≈hand after handoff by design).

10. **Essential continuous behaviors?**  
    Early cut before latch; saturated upward lift with grip=+1; optional XY return; optional fine_down; legal handoff.

11. **Minimal discrete extension?**  
    **Justifiable to study** (not applied here): lift-while-gripping + optional XY return. See `MINIMAL_ACTION_ANALYSIS.md`. Must preserve original K49 IDs; require quantized full-traj CH replay before any vocab change.

12. **Evidence for unified Transformer dataset?**  
    **Yes, as a candidate mechanism** for 2 roots (1 also SUCCESS_NEW): intercept → early branch before table-latch → lift/stabilize → legal handoff. Post-catch timing partly robust (6/12); full-trajectory timing fragile. Do **not** auto-merge as ball-impact SUCCESS_NEW experts — explicit dataset decision required.

## SUCCESS_NEW meaningfulness

Partially. Absolute U_TGT/P_TGT (≤10°/≤12 mm) remain the ball-impact grasp geometry. Early-branch catches often hold with e_pos≤12 mm but e_theta≈20°, so SUCCESS_NEW is physically strict for this constructed IC. CATCH_HANDOFF_SUCCESS uses ≤25°/≤20 mm + 2 s nominal hold.

## Scripts

`scripts/run_lateral_recatch_handoff.py`
