# Return-to-Nominal Pose Before HANDOFF — Paired Experiment

**Stop after paired experiment.** No SFT/GRPO. No physics / K48 / HANDOFF / SUCCESS_12S changes.

```text
python -u scripts/benchmark_return_to_nominal_pose_before_handoff.py --device cuda --seed 20261018
```

IC: `hard_910263` (`SNAPSHOTS/hard_910263.pkl`).  
Base: CPP + best sliding-pilot suffix (same for all branches).  
Objective: locked 10° / 12 mm + SUCCESS_12S.

---

## Hypothesis under test

COM regression after sliding is caused by entering nominal with a **non-nominal wrist/arm pose**, so that HANDOFF’s `r_des ← ref.r_des` forces a large orientation correction during nominal that drags the object COM.

## What the code does (inspected)

1. Recovery `step_legal` **integrates** `fsm.r_des` with commanded wrist rates → after CPP, both hand and live `r_des` sit ~73° from the IC nominal `ref.r_des`, but **match each other**.
2. `apply_handoff` snaps `r_des` back to `ref.r_des` and sets `p_des` to the **current** hand — creating an immediate ~73° hand↔`r_des` mismatch **without** moving joints yet.
3. Unchanged nominal then tracks the snapped `r_des`, rotating the arm back over ~0.5 s.

## Paired results (`PAIRED_RESULTS.csv`)

| Branch | S12 | NEW | Window θ / pos | Pos at HANDOFF | Hand↔ref.r_des at HANDOFF | Pos growth onset after HANDOFF | Return |
|---|---|---|---|---:|---:|---:|---|
| **A** slide→HANDOFF | 1 | 0 | 6.60° / **15.23 mm** | 11.51 mm | **73.2°** | **0.14 s** | n/a |
| **B** + secure return | 1 | 0 | 6.41° / **13.52 mm** | 11.00 mm | **72.9°** | 0.18 s | failed (`MAX_STEPS`, tok33×60) |
| **C** + any-grip return | 1 | 0 | 6.41° / **13.52 mm** | 11.00 mm | **72.9°** | 0.18 s | same as B |

Neither B nor C reached SUCCESS_NEW. B/C’s slight pos improvement came from **translation** (tok33), not from restoring the downward pose.

## Answers

| # | Answer |
|---|---|
| **Q1** | Regression begins **during nominal continuation** (~0.14 s after HANDOFF), concurrent with arm rotation from ~73°→~0° toward snapped `r_des`. Not during a successful pre-HANDOFF return (which never converged). |
| **Q2** | **Not via true pose restore.** Attempted return did not restore the arm; final window pos improved modestly (15.2→13.5 mm) from side-effect translation, still above 12 mm. |
| **Q3** | The attempted “return” (tok33×60) caused **negligible** relative θ/pos change (−0.08° / −0.5 mm) with bilateral contact. It did **not** reverse wrist orientation. |
| **Q4** | HANDOFF already defines orientation via **`ref.r_des`**. Sufficiency requires the **hand** to already be near that pose; otherwise nominal must reorient the arm while grasping. Extra controller state alone (without matching hand pose) does not prevent the snap mismatch. **K48 limitation:** no secure −wy token to undo CPP’s +wy tilt. |
| **Q5** | **Not in this paired experiment.** All branches S12 but not NEW. |

## Conclusion

The hypothesis is **mechanistically supported**: HANDOFF creates a large `r_des` discontinuity; COM error grows while nominal rotates the hand back. Fixing it by “return then HANDOFF” is **not achievable with current legal K48 wrist coverage** after a +wy CPP. Do not claim the regression is solved.

See also: `CONTACT_AUDIT.md`, `STAGE_METRICS.csv`, `RAW/*_stages.json`, `VIDEOS/`.

## Stop

Paired experiment complete. No training.
