# E Stable-Lift → Contact Manipulation → SUCCESS_NEW

**Script:** `scripts/run_freefall_e_stable_lift_contact_manip.py`  
**Out:** `results/diagnostics/freefall_e_stable_lift_contact_manip/`  
**Prerequisite audit:** `MECHANISM_AUDIT.md`

No SFT/GRPO. Physics / SUCCESS_NEW / 12 s verification unchanged.

---

## Step 1 — Mechanism audit (summary)

Full catalog: `MECHANISM_AUDIT.md` (merged from codebase + [mechanism search](77312f6d-8153-4662-9054-75dd0e692686)).

| Requested item | Status |
|----------------|--------|
| Full 6-stage slide→reclamp→90° wrist→pivot→slide→handoff as one verified unit | **MISSING** |
| Contact-preserving pivot (tilt→mid-unload→brake) with SUCCESS evidence | **FOUND** on hard-recovery ICs — `wrist_tilt_gravity_pivoting`, `grasp_recovery_pose_v1` |
| Contact-preserving sliding with SUCCESS_NEW alone | **Partial** — CPS verified; often fails final pos gate |
| Freefall `build_stabilization` (lift/return/seat) | Present; not the slide/pivot recipe |
| Lateral-force pose recovery (5 SN roots) | Found in other domain; **MISSING** freefall-E transfer test |
| Obs-only 39D closed-loop CPP | **MISSING** |
| CPP transfer hard-recovery → freefall E stable-lift | **FAILED** this round (0/11) |

This round attaches **only documented CPP schedules** (privileged open-loop K48→7D), not a newly invented 6-stage controller. No reusable `controllers/*` pose-recovery library exists — pilots emit legal 7D via `step_legal`.

Verified recipes used:

- `cpp_hrv1_920031`: tilt6×20 → unload12×15 → brake0×8  
- `cpp_hard_910101`: tilt10×8 → unload12×15 → brake0×5  
- `cpp_920031_refined`: tilt10×20 → unload12×10 → brake0×8  

---

## Step 2–4 — Matched snapshot experiment (12 held-out ICs)

Frozen controller **E** (same gains as grasp-point alignment). Snapshot at first stable-lift (sustained bilateral ≥8 after post-bilateral lift).

| Metric | Count |
|--------|------:|
| ICs evaluated | 12 |
| E stable-lift snapshots | **11/12** |
| Baseline E SUCCESS_NEW (search) | **1/12** |
| Baseline E triple-verified 12 s SUCCESS_NEW | **1/12** (`distal_off01_h8`) |
| Direct attempt (snap ≤10°/≤12 mm) | 1 |
| Direct SUCCESS_NEW (search) | 1* |
| Direct triple-verified 12 s | **0** (12 s pos≈12.5 mm fails gate) |
| Recovery attempted | **11** |
| Recovery SUCCESS_NEW (search or 12 s) | **0/11** |
| **Combined triple-verified SUCCESS_NEW roots** | **1** (baseline only; recovery added **0**) |

\*Direct search SN on `distal_off01_h8` did **not** survive 12 s / 3× replay (`FULL_SUCCESS_VERIFICATION.csv`).

### Per-IC (headline)

See `PER_IC_COMPARISON.csv`.

- **Baseline preserves E’s existing success:** `distal_off01_h8` → SN 3/3 (θ≈9.3°, pos≈11.4 mm).  
- **CPP recovery:** on all 11 stable-lift ICs, best recipe ends in **`LATCH_DURING_RECOVERY`** / `B_relative_motion_too_high` — mid-unload drops the cylinder on freefall-E states.  
- No IC gained SUCCESS_NEW from recovery that baseline lacked.

### Failure reasons

| Branch | Dominant fail |
|--------|----------------|
| Baseline (non-SN) | `E_cylinder_orientation_error` (~18–89°) |
| Recovery (CPP) | `B_relative_motion_too_high` after latch during unload |

Phase logs: `phase_logs/*__cpp_*.csv`, aggregate `PHASE_CONTACT_TRACES.csv`.

---

## Step 5 — Legality / transferability

| Layer | Result |
|-------|--------|
| Privileged E + open-loop CPP | Ran; CPP **does not transfer** to freefall E stable-lift |
| 39D observation-only CPP | **Not implemented / not claimed** |
| Invented 6-stage FSM | **Not built** (audit said MISSING) |

---

## Conclusions

1. **E direct SUCCESS_NEW path retained** (1/12 held-out, triple-verified).  
2. **Stable lift is common** (11/12) but mostly orientation-locked short of SUCCESS_NEW.  
3. **Documented CPP (tilt→mid-unload→brake) fails on these freefall-E snapshots** — unload loses contact / table latch; **0 recovery SUCCESS_NEW**.  
4. Combined total SUCCESS_NEW roots = **1** (same as E alone); recovery did not expand the set.  
5. Before claiming sliding/pivoting will fix freefall E, either (a) find a freefall-domain verified recipe, or (b) retune unload/tilt under matched E snapshots with contact logging — without inventing an unaudited controller or starting SFT/GRPO.

---

## Artifacts

- `MECHANISM_AUDIT.md`, `REPORT.md`, `FAILURE_ANALYSIS.md`  
- `PER_IC_COMPARISON.csv`, `PHASE_CONTACT_TRACES.csv`, `FULL_SUCCESS_VERIFICATION.csv`  
- `snapshots/`, `trajectories/`, `phase_logs/`, `videos/`, `raw/`  
- Script: `scripts/run_freefall_e_stable_lift_contact_manip.py`

## Files created/modified

| Path | Action |
|------|--------|
| `scripts/run_freefall_e_stable_lift_contact_manip.py` | created |
| `results/diagnostics/freefall_e_stable_lift_contact_manip/**` | created |
