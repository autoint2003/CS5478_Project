# Freefall-E Post-Capture Stabilize + Axial Slide

**Script:** `scripts/run_freefall_e_postcapture_stabilize_slide.py`  
**Out:** `results/diagnostics/freefall_e_postcapture_stabilize_slide/`  
**Wall:** 113.8s

Privileged diagnostic. No SFT/GRPO. Physics / SUCCESS_NEW / frozen E / 39D unchanged.

---

## Decode audit (Stage 1.1)

- Formula: `action = clip(proto[tok] + clip(pre, -0.25, 0.25), -1, 1)`.
- **Documented CPP:** `pre = zeros(7)` every step -> tok12 always a6~-0.48.
- **Prior transfer:** `tokens_to_actions` sets `pre = action`, so after tilt6 (a4=1,a6=1) residual clips to 0.25 and unload becomes a6~-0.23 then ~-0.71/-0.73 with leftover wrist.
- Full step map: `raw/DECODE_AUDIT.json`.

## Stage 1 — Decode and settle

Pass rule: after zero-residual settle, verify hold 30 steps with bilateral frac>=0.95, no lose-bil, no latch, mean v_rel (last half)<=0.12.

**Stage1 pass ICs:** ['distal_off01_h8', 'perp_off01_h0', 'distal_off01_h4', 'perp_off01_h1', 'perp_off01_h7_hz'] (5/6)

See `stage1/STAGE1_COMPARISON.csv` and `traces/*__stage1__*.csv`.

## Stage 2 — Axial sliding and reclamp

Only from Stage1-pass matched settled captures. Variants reuse CPS families from `pilot_sliding_position_correction_v1` (mid+secure+reclamp / secure-only / opposite / interleaved).
Lever **target** ~8.0 mm is aspirational; Stage3 gate requires ecc or |s_axial| >= 6.0 mm **and** post-reclamp stable contact. Gravity align recorded (helps vs distance-only).

**Stage2 pass ICs:** ['distal_off01_h8', 'distal_off01_h4'] (2/6)

See `stage2/STAGE2_SLIDE.csv`.

## Stage 3 — Conditional pivoting

Ran on: ['distal_off01_h8', 'distal_off01_h4']. Branches: lock / mild / documented mid-unload.

See `stage3/STAGE3_PIVOT.csv`. Not claimed as SUCCESS_NEW.

## Baseline E (preserved)

See `BASELINE_E.csv`. Direct SUCCESS_NEW path unchanged.

## Legality

- Privileged diagnostic (forces, contacts, open-loop 7D).
- **Not** observation-only 39D.
- Stable-lift / bilateral / single-phase pivot != SUCCESS_NEW.
- 12s + 3/3 only if a full SUCCESS_NEW candidate appears (none claimed unless listed).

## Files

- `scripts/run_freefall_e_postcapture_stabilize_slide.py`

- `results/diagnostics/freefall_e_postcapture_stabilize_slide/**`
