# Lateral-Offset Falling Recatch (Nominal Grasp Pose)

**Baseline:** K49 (`K49_tok48_wy0p25`, checksum `33d4820d1ec09c8b`) · AIRBORNE_V3 / 20 ms / 39D obs unchanged  
**Init:** genuine nominal grasp orientation (open jaws) — **not** inverted catch-ready  
**Criterion:** `LATERAL_RECATCH_SUCCESS` (bilateral hold ≥ 0.50 s, descending first contact, nontrivial chase, no pre-capture table hit)  
**ICs:** constructed free-fall diagnostics (not ball-impact roots; **not** mixed into K49 experts)  
**No SFT/GRPO.**

## Budget (frozen before run)

| Item | Value |
|------|------:|
| Stage A / B / C / D | 200 / 700 / 500 / 400 |
| Total | 1800 |
| Used | 1402 |
| Wall limit | 40 min |
| Wall used | 174.7 s |

## IC construction

- Robot: `grasp_settled` → open-gripper legal actions; store `R0` and finger mid.
- Cylinder: COM height = grasp mid; zero \(v,\omega\); no robot/table contact at snapshot.
- `offset_m` is **additional** separation beyond an empirically clear baseline (hand mesh free at \(z=\mathrm{mid}\)):
  - `open`: clear 0.12 m along jaw-opening (+Rh[:,1] / world +X)
  - `distal` / `perp`: clear 0.06 m along ±pad axis (±Rh[:,0] / ±world Y)
- For this nominal top grasp, approach ∥ −Z, so only two independent horizontal axes exist; `distal` and `perp` are **opposite sides** of the pad axis (not a third orthogonal).
- Time to table without intervention ≈ **0.204 s** from grasp height (not the ball-impact 0.23 s window).

See `INITIAL_STATE_AUDIT.md`, `OFFSET_SWEEP.csv`.

## Results summary

| Metric | Value |
|--------|------:|
| Verified SUCCESS ICs | **9** |
| Directions | **distal**, **perp** |
| Open-direction SUCCESS | **0** / 384 trials |
| Largest successful `offset_m` | **0.04 m** (actual along-axis sep ≈ 0.10 m) |
| Success conditions | `D_scoop` / `D_refine_scoop` only |
| K49 token overchase SUCCESS | **0** / 24 |
| Template replay SUCCESS | **0** / 3 |
| Continuous adapt (err-chase) SUCCESS | **0** / 177 |
| Mean hand displacement (successes) | **0.210 m** |
| Wrist: nominal / moderate / large / reversal | **32 / 8 / 0 / 0** |
| Robustness survivors | **43 / 240** |
| Legal HANDOFF_OK | **0 / 40** |
| SUCCESS_NEW (unchanged definition) | **0 / 40** |

SUCCESS ICs: `distal_off00..04`, `perp_off01..04` (see `CATCH_SUCCESS_VERIFICATION.csv`).

## Mechanism that worked

**Align-then-descend scoop** (legal continuous actions):

1. Open jaws; slight lift while closing planar error to the falling COM (~4×20 ms).
2. Saturated downward chase (`a[2]=+1`, hand +z → world −z) until COM is in the open-axis jaw range.
3. Close grip; `FINE_DOWN` hold ≥ 0.50 s.

Distal/perp ICs already project into the jaw open-axis (`com_between_fingers=1` at init) while being clear of the pads along ±Y — so the hand mainly translates along the pad axis, then descends around the cylinder. First bilateral contact ~0.14–0.16 s while still descending; jaw entry confirmed (`CONTACT_GEOMETRY.csv`).

Open-direction ICs require ≥12 cm planar chase past the hand mesh; approach from the opening side **knocks** the cylinder with the palm mesh and never forms bilateral capture under this budget.

## Q1–Q12

1. **Can Panda actively recatch a laterally displaced falling cylinder from nominal grasp pose?**  
   **Yes**, for pad-axis (distal/perp) offsets with continuous legal scoop control. Not demonstrated for opening-axis offsets.

2. **Which offset directions are feasible?**  
   **`distal` and `perp`** (opposite sides of the finger-pad horizontal axis). **`open` (jaw opening):** no verified SUCCESS.

3. **Largest tested offset with verified success?**  
   **`offset_m = 0.04 m`** beyond the 0.06 m clear baseline → ~**10 cm** mid-to-COM along the pad axis (`distal_off04`, `perp_off04`). Larger 6–8 cm additional offsets failed under Stage C budget.

4. **Does the proven over-chase mechanism transfer?**  
   **Not as a token/template replay.** Stage A `template_raw` and `K49_overchase` scored 0. The physical idea (open → chase → close → brake) transfers only after **re-timing** into a short-horizon align+descend scoop matched to the ~0.20 s table window. Long vertical-recatch sequences do not transfer unchanged.

5. **How much actual hand displacement is required?**  
   Mean **~0.21 m** among successes (path length of the chase+descend, not just the initial planar offset).

6. **Does the cylinder enter the intended jaw region before closure?**  
   **Yes** on successes: init already in open-axis jaw range for distal/perp; contact geometry shows `com_between=1` and bilateral pads (`CONTACT_GEOMETRY.csv`).

7. **How sensitive is success to closure timing?**  
   Moderately sensitive: robustness timing shifts (±20/40 ms) and ±5 mm offset variants → **43/240** still `LATERAL_RECATCH_SUCCESS`. Many successes are **nominal** rather than timing-robust.

8. **Does K49 suffice, or does continuous legal control help?**  
   **K49 alone insufficient** (0/24). Continuous legal scoop is required for all 40 verified catch trajectories. Discrete vocab lacks a saturated planar-chase + timed descend primitive for this IC family.

9. **Can the robot stabilize without inverting?**  
   **Yes.** 32/40 successes are `nominal_orientation_lateral` (wrist deviation &lt;15°); 8 are `moderate_wrist_adjustment` (&lt;45°). **No** orientation-reversal successes counted as evidence of nominal-pose recatch.

10. **Legal return to nominal-compatible grasp and HANDOFF?**  
    **Not under this budget.** Catch succeeds, but `HANDOFF_OK=0` and `SUCCESS_NEW=0` for all 40 catch trajectories (`HANDOFF_RESULTS.csv`). Pose after scoop is displaced ~20 cm from nominal; return/HANDOFF remains an open Stage-D gap.

11. **Failure modes?**  
    - **Open direction:** interception geometry — palm/hand mesh collision ejects the cylinder; 0 bilateral captures.  
    - **Large distal/perp (6–8 cm add):** interception timing — cannot close XY error before table (~0.20 s).  
    - **Near-miss distal/perp:** brief bilateral then loss before 0.50 s hold (stabilization), or unilateral tip scrape.  
    - **K49/template:** wrong timing/axis relative to free-fall window.

12. **Useful evidence for dynamic grasp recovery?**  
    **Yes, as a candidate mechanism**, with caveats: demonstrates **active lateral interception from nominal orientation** (not inverted free-fall catching), continuous-action only, pad-axis offsets up to ~10 cm mid-COM, no automatic dataset merge. Does **not** yet show HANDOFF/SUCCESS_NEW. Prefer these trajectories as mechanism candidates after explicit benchmark/dataset decisions — do not treat as ball-impact roots.

## Deliverables

| Path | Role |
|------|------|
| `OFFSET_SWEEP.csv` | IC geometry / clearance / \(t_\mathrm{table}\) |
| `OVERCHASE_TRANSFER.csv` | All search trials |
| `INTERCEPTION_TIMELINE.csv` | Timing summaries |
| `CONTACT_GEOMETRY.csv` | First-contact sites |
| `CATCH_SUCCESS_VERIFICATION.csv` | Verified catches |
| `HANDOFF_RESULTS.csv` | Post-catch HANDOFF / SUCCESS_NEW |
| `ROBUSTNESS_RESULTS.csv` | Timing / offset variants |
| `FAILURE_ANALYSIS.md` | Failure taxonomy |
| `trajectories/` | Action npz for successes |
| `videos/` | Demo + success renders |
| `raw/BUDGET.json`, `USAGE.json` | Frozen budget + usage |

## Scripts

`scripts/run_lateral_offset_falling_recatch.py`
