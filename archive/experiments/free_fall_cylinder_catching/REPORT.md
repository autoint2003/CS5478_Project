# Free-Fall Cylinder Catching Feasibility

**Robot/controller:** unchanged Panda + AIRBORNE_V3 · K49 checksum `33d4820d1ec09c8b`  
**Criterion:** `FREE_FALL_CATCH_SUCCESS` (HOLD_S=**0.50 s**) — frozen in `BENCHMARK_DEFINITION.md`  
**No SFT/GRPO. Trajectories not mixed into K49 recovery experts.**

## Budget

| Item | Value |
|------|------:|
| Stage1 / Stage2 | 900 / 900 |
| Used (search) | 1006 |
| Wall (s) | ~194 |

## Headline result

**Yes — under a distal jaw-entry release + catch-ready wrist**, the existing stack achieved verified free-fall catches on **2** Level A roots (`A_d1`, `A_d6`), both with **K49** action sequences (exact replay verified). Level B/C and overhead-style releases did **not** yield `FREE_FALL_CATCH_SUCCESS` in this budget.

| IC | height | offset | ω | condition | bil hold | vz@contact | NOMINAL_HANDOFF |
|----|-------:|-------:|--:|-----------|--------:|-----------:|----------------:|
| A_d1 | 0.08 m | 0.00 m | 0 | B_k49 | 0.60 s | −0.27 | 0 |
| A_d6 | 0.07 m | 0.01 m | 0 | B_k49 | 0.60 s | −0.74 | 0 |

## Morphology constraint (critical)

With the default downward gripper, the **palm occludes overhead entry** into the jaw gap. Pure top-down drops land on the hand/finger tops (outer contacts). Successful ICs use **catch-ready wrist + distal tip entry** so the cylinder can enter the jaw volume while still descending.

## Baselines (aggregate)

| Baseline | catch successes (rows) |
|----------|----------------------:|
| A no-intercept | 0 |
| B K49 | 2 |
| C K49+fine_down | 0 |
| D continuous / scoop | 0 |

Continuous search produced bilateral contact up to ~1.1 s in some trials but did **not** meet the full frozen criterion more often than K49 under this budget (K49 was the only SUCCESS condition).

## Robustness (post-verify)

On the two successful roots with the same K49 sequence:

- **Release-state** nearby (Δh±1 cm, Δoffset±5 mm): **5/10** still SUCCESS (A_d1 more robust than A_d6).
- **Action timing** ±1–3 steps (20 ms): **0/10** still SUCCESS (timing-sensitive).
- Level B offsets 4–6 cm and Level C ω=2–3 rad/s with K49: **0** SUCCESS.

## Q1–Q12

1. Can the existing Panda catch a genuinely free-falling cylinder? **Yes**, for carefully constructed distal Level A releases.
2. Simplest verified IC: **`A_d1`** — height 0.08 m, offset 0.0 m, ω=0, distal entry, K49.
3. Max lateral offset among independent SUCCESS roots: **0.01 m** (`A_d6`). Robustness on `A_d1` survived ±0.005 m.
4. Rotating free-fall (Level C)? **No** verified SUCCESS in this budget.
5. Enter intended opening before closure? Successful catches require distal/jaw-volume entry; many failed trials still show **outer_or_edge** first contacts (`CONTACT_GEOMETRY.csv`). COM-between ≠ pad-inside.
6. Controller track required intercept? Reachability CSV shows limited downward hand speed; successes rely more on **cylinder falling into a prepared open jaw** than long chase tracking.
7. Geometry vs time: **Primary failure = contact geometry** (palm occlusion / outer contact). Table window is secondary (~0.1–0.3 s). Wrong entry → brief bil ≤0.1 s then drop.
8. Continuous outperform K49? **Not in this budget** — only K49 produced SUCCESS rows.
9. Compact discrete vocab? Existing K49 already represents the successful open-chase-close-hold pattern here; **do not expand** until continuous shows distinct useful behaviors.
10. Independent release roots with sustained capture: **2**.
11. Robust to small variations? **Partially** to release state; **not** to 20 ms timing shifts.
12. Implication for grasp-recovery: dynamic catching is feasible only when the object can **enter the jaw volume before high impulse**. Ball-impact ICs that begin on outer/tip contacts are a different problem — catching skill does not transfer without earlier onset / geometry repair. Next recovery work should prioritize **pre-contact jaw entry**, not larger chase alone.

## Scripts

`scripts/run_free_fall_cylinder_catching.py`, `scripts/_tmp_continue_freefall_catch.py`
