# Active Recatch Generalization & Discretization

**Seeds:** `distal_off01_r8`, `perp_off01_r15` (triple-verified CATCH_HANDOFF from lateral_recatch_handoff)  
**Mechanism:** intercept → jaw entry → bilateral capture → **early lift** → pose correction → HANDOFF → nominal  
**No SFT/GRPO. Original K49 expert dataset unchanged. Constructed free-fall ICs ≠ ball-impact roots.**

## Frozen budgets

| Phase | Alloc |
|-------|------:|
| Audit | 120 |
| Generalization | 240 |
| Lift / orientation | 160 |
| Discrete | 200 |
| Robustness | 80 |
| **Used / Total** | **~509+final disc / 800** (see `raw/USAGE.json`) |

## Q1–Q12

1. **Full IC replay without privileged transitions?**  
   **Yes.** Both seeds replay as uninterrupted legal 7D sequences from the original free-fall IC (`seed_full_replay`: distal CH=1 SN=0; perp CH=1 SN=1). Probe-latch clear at t0 only removes IC-construction contamination. Sticky-latch avoidance is from **early lift timing**, not hidden branch state.

2. **Why early lift prevents sticky latch?**  
   Continuing scoop / `FINE_DOWN` near the table briefly loses bilateral contact → table bounce sets a **sticky latch** → HANDOFF fails under original monitors (`A_continue_scoop_hold` latch=1). Lifting while grip stays closed raises the object before that loss window. Matched `no_lift` ablation fails.

3. **Can distal_off01_r8 orientation reach SUCCESS_NEW (≤10°)?**  
   **Not under this budget.** Orientation searches keep e_theta≈20–27°. CATCH_HANDOFF remains achievable; SUCCESS_NEW does not for this distal root.

4. **Additional independent roots with SUCCESS_NEW?**  
   **0** new constructed ICs reached SUCCESS_NEW in search (budget burned on distal CH-only variants with θ≈19–20°). **CATCH_HANDOFF** generalized to new distal ICs (`distal_off01_base_g4`, `distal_off01_yaw15_g5`, 20 CH hits). Perp continue did not add SN roots under remaining budget.

5. **Timing sensitivity?**  
   **14/24** robustness CH survivors. Post-catch ±20/40 ms often OK; full-trajectory timing shifts frequently fail at **interception**.

6. **Essential continuous controls — interception?**  
   Open-jaw planar chase with evolving `(a0,a1,a2)` (scoop align+descend). Zeroing planar chase (`no_intercept_motion`) → no capture. **K49 NN quantization of the prefix destroys interception** (first_bil=None). Prefix is **not** replaceable by a single chase token.

7. **Essential — stabilization / HANDOFF?**  
   Early lift with `a[2]≈−0.9…−1.0`, grip=+1; optional short wy then pure up; fine_down seating. `no_lift` / `grip_open_post` fail. XY return optional (seed-dependent).

8. **Missing from K49?**  
   (i) Hard lift (`a2≲−0.9`, grip closed) — best K49 is tok33 at −0.4.  
   (ii) **Intercept scoop sequence** — no short token covers the multi-step open-chase profile; NN mapping loses bil.  
   (iii) fine_down ≈ tok16 (adequate).

9. **Smallest useful discrete extension?**  
   **`K49_AR_v1` (+ optional prefix library):**  
   | ID | Role |  
   |---:|------|  
   | 49 | LIFT_UP `[0,0,-0.95,0,0,0,1]` |  
   | 50 | LIFT_WY `[0,0,-0.9,0,0.25,0,1]` |  
   | 51 | CHASE_PAD (insufficient alone) |  
   | 52 | FINE_AR |  
   | 53+ | **Prefix library** — store verified scoop steps as tokens (identity quantize) |  
   Preserves K49 IDs 0–48 / HANDOFF 47. See `CANDIDATE_TOKEN_MAPPING.csv`.

10. **Can that vocabulary reproduce complete successes?**  
    **Yes, with identity-prefix + scheduled lift:**  
    - `perp_off01_r15`: **CH=1 and SUCCESS_NEW=1** with prefix library + `LIFT_WY×20 → LIFT_UP×20 → FINE`.  
    - `distal_off01_r8`: **CH=1**, SN=0 (orientation).  
    Naive NN quantize / single CHASE_PAD prefix / templates A–E: **fail**. Phase-aware AR without prefix library: **fail** on intercept.

11. **Suitable as a separate expert family for one unified Transformer?**  
    **Conditionally yes** as a **constructed free-fall / active-recatch family**, separate from ball-impact K49 experts, once: (a) prefix tokens are treated as a small library or learned sequence, (b) SUCCESS_NEW-complete experts preferred (perp-class), (c) distal orientation fixed or labeled CH-only.

12. **Before real ball-impact transfer?**  
    Replay on genuine ball-impact release roots; confirm 39D obs without latch privilege; timing-robust discrete policies; distal SUCCESS_NEW orientation; explicit dataset gate — do not auto-merge.

## Ablation highlights

| Ablation | Effect |
|----------|--------|
| `no_lift` | CH fails (latch path) |
| `no_intercept_motion` | No capture |
| `grip_open_post` | Grasp loss |
| `zero_wy` | Often still OK |
| `A_continue_scoop_hold` | latch=1 |
| Identity prefix + WY→UP lift | **perp SN+CH** |

## Discrete mapping (core)

| Token | Role | Action (7D) |
|------:|------|-------------|
| 49 | LIFT_UP | [0, 0, -0.95, 0, 0, 0, 1] |
| 50 | LIFT_WY | [0, 0, -0.9, 0, 0.25, 0, 1] |
| 51 | CHASE_PAD | [0, 1, 0.45, 0, 0, 0, -1] |
| 52 | FINE_AR | [0, 0, 0.15, 0, 0, 0, 1] |
| 53+ | prefix_library | per-step scoop actions from verified seeds |

## Scripts

- `scripts/run_active_recatch_generalization_discretization.py`
- `scripts/_continue_ar_gen_disc.py` (remaining-budget continue)
