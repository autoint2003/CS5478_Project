# K49 Return Axial Slip Mechanism — Diagnosis

**Legal baseline:** unchanged K49 token 48 via instrumented `step_legal`.  
**Counterfactuals** (ω-scale, holds, gravity, friction, pure−wy) are labeled; production physics and K49 vocabulary are unchanged.

```text
python -u scripts/diagnose_k49_return_axial_slip.py --device cuda
python -u scripts/_tmp_axial_slip_extra_cf.py
```

Outputs: `BASELINE_REPRODUCTION.md`, `FORCE_BALANCE.md`, `COUNTERFACTUAL_COMPARISON.csv`, `RAW/`, `plots/`.

---

## Q1 — Does −4.84 mm reproduce?

**Yes.** Measured Δe_ax = **−4.837 mm** (−7.107 → −11.944 mm). Lateral +0.19 mm. Hand 73.2° → 5.7°.

## Q2 — When does sliding begin?

First clear axial drift within the first few tok48 substeps: onset t≈4.502 s, hand↔ref≈**69.1°** (only ~4° into the return), e_ax≈−7.3 mm. Slip then continues roughly monotonically through the rotation (`plots/A_baseline_panels.png`).

## Q3 — Gravity, inertia, or combination?

**Not gravity.** Gravity-off CF: Δe_ax = **−4.846 mm** (indistinguishable from baseline). Half-g likewise −4.842 mm.

**Rate-dependent / rotation-driven.** Matched-angle slower returns (CF ω-scale):

| Condition | mean \|w_hand\| | Δe_ax (mm) | ≈ mm/deg |
|---|---:|---:|---:|
| A legal tok48 | 3.09 | −4.84 | 0.072 |
| B matched ×0.5 | 1.57 | −2.41 | 0.036 |
| B matched ×0.25 | 0.79 | −1.17 | 0.017 |

Same angular path, less slip at lower rate ⇒ not a fixed geometric mm/deg transport alone.

## Q4 — Does slowing reduce axial slip?

**Yes (counterfactual).** K49 has no legal lower-\|wy\| secure reverse; tests scaled `w_cmd` only. Roughly halves slip when ω halves at matched angle.

## Q5 — Slip during quasi-static holds?

**No sustained creep.** At holds along the path (0–100% return), sustained Δe_ax over the last 40 ms of a 0.1 s hold is **≲0.01 mm**. Transient settling ≲0.2 mm. Gravity-driven static sliding is not supported.

## Q6 — Does increasing friction suppress slip?

**Inconclusive — CF invalid.** Raising finger–object sliding μ (even ×1.1) collapses contact normals during Return (Fn_mean ≈0.09 N vs ≈18 N legal). Slip then balloons (−27 mm). Do **not** interpret as “more friction worsens slip”; the contact state is broken after the μ write. Production μ untouched.

## Q7 — Does finger normal force decrease during Return?

**No meaningful drop** on the legal baseline: Fn_total 18.00 → min 17.92 N (mean 18.11). Secure grip command and pad normals stay high.

## Q8 — Supported mechanism

**Rate-dependent axial slip under secure bilateral contact during wrist reorientation** — object lags along the cylinder axis while the hand rotates. Supported by:

- gravity-off invariance;
- no quasi-static hold creep;
- strong ω dependence at matched Δθ;
- stable Fn (not unloading);
- pure −wy (vx=0) CF still −4.85 mm → tok48’s tiny +vx is not the driver.

Uncertainties: world-axis force residual is noisy (finite-diff accel); rotating-hand inertial decomposition not claimed closed; friction CF unusable.

## Q9 — Smallest justified next **legal** K49 test

**Interleave token 48 with secure zero-motion `[0]`** to cover the same wrist angle at lower mean \|ω\| (legal approximation of the helpful slow CF). Re-measure Δe_ax and full-objective margin on `hard_910263`.

Only if that fails, consider a documented lower-\|wy\| append (K50) — not justified until the legal interleave test is done.

---

## Stop

Diagnosis complete. No expert dataset, no SFT/GRPO, no production physics/vocab edits.
