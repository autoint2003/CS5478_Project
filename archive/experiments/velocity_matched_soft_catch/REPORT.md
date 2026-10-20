# Velocity-Matched Soft Catch Diagnostics

**Baseline K49:** `K49_tok48_wy0p25` checksum `33d4820d1ec09c8b`  
**Legal interface:** continuous AIRBORNE_V3 `a ∈ [-1,1]^7` / 20 ms. No SFT/GRPO.

## Budget (frozen before run)

| Item | Value |
|------|------:|
| Total rollouts | 2400 |
| Wall (s) | 2700 |
| Used | 1875 |
| Wall used (s) | 2701.7 |
| Stop reason | wall-clock limit (rollouts left unused) |
| Phase A / B | 30% audit / 70% search (40/30/20/10 within B) |

See `CATCH_IT_METHOD_COMPARISON.md` (Catch It uses two-stage RL; **no documented velocity-matching loss**).

## Phase A — Contact kinematics

Previously quoted ~0.41–0.45 m/s values are **COM-relative** speeds (`v_rel_h` / COM−hand), **not** contact-point relative velocities.

| Metric | Value |
|--------|------:|
| Contact events audited | 60 |
| Mean COM-relative at contact (m/s) | 0.2351 |
| Mean contact-point relative (m/s) | 0.2167 |
| Mean pt/com ratio | 0.837 |
| High-impact subset (v_com≥0.35): mean COM / pt | 0.451 / 0.479 |
| Controller VM tracking ratio | 0.539 |

**Important:** most Recovery ICs already have finger–cylinder contact at t≈0, so classical *pre-contact* M2 rarely applies. Early-contact relative-velocity reduction (`M2_early`) is the relevant soft-catch diagnostic.

## Phase B — Matched control

| Condition | n | bil rate | mean persist (s) | M2_pre | M2_early | SUCCESS_NEW |
|-----------|--:|---------:|-----------------:|-------:|---------:|------------:|
| A K49 chase | 120 | 0.73 | 0.037 | 0.00 | 0.00 | 0 |
| B velocity match | 660 | 0.72 | 0.019 | 0.00 | ~0.01 | 0 |
| C VM + grip timing | 396 | 0.72 | 0.018 | 0.00 | ~0.02 | 0 |

Co-motion vs stop (ablation mean persist): comotion=0.011 stop=0.018  
VM-family early-contact M2 rate: **0.01** (n=540). Pose/`fine_down` family M2_early is high (~0.96) but reflects already-captured pose correction, not soft intercept.

Stable bilateral (M6-like, persist≥0.30s) ICs: **1** (`hard_910012`)  
Complete SUCCESS_NEW ICs: **1** (`hard_910012`) — **only** via continuous `fine_down` pose recipe (reconfirming prior action-authority ablation), **not** via velocity-matched soft catch.

Unique verified FULL conditions: see `FULL_SUCCESS_VERIFICATION_UNIQUE.csv` (`stab2_recipe`, `pose_verified_fd`, CEM local variants). Diagnostic trajectories kept separate from K49 expert dataset.

## Q1–Q13

1. Contact-point relative velocities (audited): mean **0.217 m/s** (n=60). High-impact contacts ~**0.479 m/s**.
2. Different from COM-relative? Mean COM **0.235 m/s**; ratio pt/com **0.84**. Prior 0.41–0.45 m/s quotes were **COM-relative**, not contact-point. Rotation can make pt **higher** than COM (e.g. `ubimp_0040`, `hdimp_0039`).
3. Can legal continuous actions reduce relative velocity before contact? Pre-contact M2 ≈ **0** (contact already present). Early-contact M2_early under VM search ≈ **0.01**.
4. Velocity matching improve uni/bilateral capture? bil B≈0.72 vs A≈0.73 — **no clear gain**; **0** SUCCESS_NEW from VM.
5. Coordinated grip timing help? bil C≈0.72 vs B≈0.72 — **no clear gain** under budget.
6. Brief post-contact co-motion help? persist comotion=0.011 vs stop=0.018 (**no gain**).
7. Dominant failure factors: **high contact-point relative velocity** (ball-impact), **parallel-jaw morphology**, **controller tracking lag** (~0.54). Geometry often has COM between fingers; outer-contact risk low in audited seeds.
8. Controller achieve required velocity? Mean tracking ratio **0.539** under VM commands.
9. HIGH_MOTION ICs reaching stable bilateral: **1** (`hard_910012` only among the 15).
10. Complete SUCCESS_NEW: **1** IC — continuous `fine_down` pose path, not soft-catch VM.
11. Useful behaviors absent from K49: **`fine_down` (+0.15 vz secure)** for pose. Soft-catch VM/co-motion **not justified** by SUCCESS_NEW.
12. Small discrete vocabulary? **Yes for pose** (`fine_down` / tok63). Soft-catch VM: **do not quantize** — continuous VM did not unlock HIGH_MOTION FULL experts.
13. Remaining limitation: HIGH_MOTION capture → **gripper morphology + impact relative velocity / controller dynamics**. Pose-near ICs → **action representation** (gentle COM correction). Soft-catch search within frozen budget did not unlock dynamic recovery.

## Hypothesis verdict

**Velocity-matched soft catching under the existing legal Panda interface did not unlock physically valid HIGH_MOTION SUCCESS_NEW recoveries** in this budget. The one SUCCESS_NEW is pose correction via continuous `fine_down` on an already-capturable IC.

## Scripts
`scripts/run_velocity_matched_soft_catch.py`, `scripts/_tmp_continue_vm_soft_catch.py`
