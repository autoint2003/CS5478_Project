# Plan C — Horizontal Downward-Impact Dynamic End-Recatch (Feasibility)

**Script:** `scripts/run_plan_c_horizontal_downward_end_recatch.py`  
**Out:** `results/diagnostics/plan_c_horizontal_downward_end_recatch/`  
**Prior evidence reused:** `top_impact_open_overchase_recatch`, `k49_horizontal_endcatch_recatch`, `abc_recovery`

Privileged diagnostic. No SFT/GRPO. Physics / locked SUCCESS_NEW (≤10° / ≤12 mm / ≥0.5 s) unchanged.  
**Only `DIRECT_SUCCESS_NEW` with end-recatch mechanism evidence counts as Plan C success.**

---

## Step 1 — Legal IC construction

Real MuJoCo **top-down ball impacts** via `data_generation/strategies/recatch_oracle._impact` (no cylinder teleport to invent fall).

| Metric | N |
|--------|--:|
| Impact ICs surveyed | 18 |
| Plan C focus (near-horizontal + downward + nominal fail) | 6 |
| Focus ICs | center progressive slips at 6–7 m/s (see `IC_MANIFEST.csv`) |

Impact-before/after kinematics logged in IC rows: `theta_vert0`, `vz0`, `d_prel_z`, `peak_n`, `nominal_why`.  
Matched post-impact snapshots: `snapshots/*.pkl` (identical forks for A/B/C).

**Note:** End-axial cells (`|axial|≥0.022`) in this grid mostly remain contact-held with small slip (same as prior top-impact survey); Plan C focus this round is the **center progressive downward** cohort that fails nominal handoff — the intended release→fall→recatch basin.

---

## Plans (matched)

| Plan | Mechanism | Interface |
|------|-----------|-----------|
| A | Documented CPP tilt→mid-unload→brake | K48 tokens, zero-residual |
| B | Short release-close + mild slide + wrist | K48 tokens |
| C | Active release + end-chase + close/brake | Continuous legal 7D (`step_legal`) |

---

## Step 3 — Plan C outcome taxonomy (6 focus ICs)

| Outcome | Count | Plan C success? |
|---------|------:|:---------------:|
| **DIRECT_SUCCESS_NEW** (NEW gates + end/rotate mech) | **0** | YES |
| DYNAMIC_ROTATION_STABLE_RECATCH | 0 | NO |
| END_CONTACT_ONLY | 0 | NO |
| SUCCESS_12S_BUT_NOT_SUCCESS_NEW | **3** | NO |
| RECATCH_NO_END_CONTACT | 1 | NO |
| FAIL_NO_CAPTURE | 2 | NO |
| End-contact steps observed | **0** | — |

### What the 3× SUCCESS_12S cases actually did

On `s6.0_p10_*` and `s6.0_p25_a+0.000_e0.010`:

1. Grip opens (~6 zero-contact steps).
2. Recloses **near cylinder center** (`|s_axial|~2–8 mm`, **no end contact**).
3. Orientation stays **~86–90° from world vertical** (Δθ toward vertical ≤2°).
4. Legacy `SUCCESS_12S` can fire after lift, but locked **SUCCESS_NEW geometry fails** (θ≫10°).

→ Feasible as **center reclose under legal 7D**, **not** as Plan C end-recatch uprighting.

---

## Step 4 — Matched A / B / C

| Plan | SUCCESS_NEW | Dominant failure |
|------|------------:|------------------|
| A (CPP) | **0/6** | Mid-unload → latch/drop on falling IC |
| B (short + slide/pivot) | **0/6** | Latch/drop during short open |
| C (end-recatch search) | **0/6** | No end contact; no NEW; 3× S12-only center reclose |

Per-IC table: `MATCHED_ABC_COMPARISON.csv`.  
Traces (forces, s_axial, θ, ω): `traces/{ic}__{A,B,C_best}.csv`.

### Applicability

- **A:** Unsuitable once impact has already produced downward slip / open grasp — same unload failure mode as freefall-E CPP isolation.
- **B:** Short release on these snaps does not leave enough time to slide/pivot before table.
- **C:** Legal release+reclose is possible at 6 m/s center impacts, but **end intercept + inertial uprighting to SUCCESS_NEW is not yet demonstrated**. Harder 7 m/s cases lose capture entirely.

### Action cost (mean, diagnostic)

See `A_cost` / `B_cost` / `C_cost` columns (Σ ‖trans/wrist‖+|grip| over steps). C schedules that reach lift are longer (~99) than failed early aborts (~11–12).

---

## Mechanism verdict

| Question | Answer |
|----------|--------|
| Real horizontal + top impact + downward IC? | **Yes** (surveyed; 6 focus) |
| Legal active release? | **Yes** (zero-contact intervals observed) |
| End-contact intercept? | **Not achieved** in this budget (0 end-contact steps) |
| Dynamic uprighting during recatch? | **Not achieved** (Δθ≲4°) |
| Direct SUCCESS_NEW without pivot/slide? | **0 / 6** |
| Plan A/B better on same snaps? | **No** (also 0 NEW) |

Consistent with prior `k49_horizontal_endcatch_recatch` (0 Type3 NEW) and `top_impact_open_overchase` (0 SUCCESS_12S under older open-chase; here S12 appears on a milder center-reclose path but still not NEW).

---

## Minimal next steps (not run)

1. Bias chase target to **true cylinder tip** with privileged end pose; raise `|s_axial|` gate before close.
2. Add wrist/angular matching so tip contact produces measured `torque_proxy` and Δθ≥15–30° before brake.
3. Keep SUCCESS_NEW hold verification separate from SUCCESS_12S.
4. Still no obs-only / SFT / GRPO until privileged end-recatch hits DIRECT_SUCCESS_NEW.

---

## Legality

- Privileged diagnostic only.
- Not 39D observation-only.
- No SAC / SFT / GRPO.
- No physics / SUCCESS_NEW criterion edits.

## Artifacts

| Path | Role |
|------|------|
| `IC_MANIFEST.csv` | all surveyed impacts |
| `MATCHED_ABC_COMPARISON.csv` | A/B/C per IC |
| `PLAN_C_OUTCOMES.csv` | Plan C labels |
| `FAILURE_ANALYSIS.md` | failure modes |
| `traces/` | per-step contact / force / θ / ω |
| `snapshots/` | matched post-impact states |
| `videos/` | optional Plan C rollouts |
| `scripts/run_plan_c_horizontal_downward_end_recatch.py` | runner |
