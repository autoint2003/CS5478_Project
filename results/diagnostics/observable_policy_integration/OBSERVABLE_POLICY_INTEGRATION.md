# Observable Policy Integration

**used_final_eval_data: false**

Eval hashes unchanged (not loaded): offset `f041a2c5…ad63`, impact `7ef30ec8…8b5e`.

No SAC. Observable **reward** formula unchanged. This step adds a GT-free **policy observation** for `reward_mode=observable_tactile` only.

## Modes

| mode | observation | reward | terminate |
|---|---|---|---|
| `oracle` | 24-D privileged `observe_recovery4d` | GT `e_x` potential | GT `recovered` / `fail_kind` |
| `observable_tactile` | **27-D** `observe_observable` | tactile `ObservableTactileReward` | `success_obs` / `failure_obs` |

Do not mix oracle obs with observable reward in this main mode.

## Q1–Q3 Observation vector (27-D)

`envs/observable_obs.py`. Invalid \(\hat e\): **placeholder 0** with `estimate_valid=0` (0 is not “centered”).

| i | name | source class | physical source | scale / clip |
|---|---|---|---|---|
| 0 | e_hat_x | ESTIMATED | geometric CoP `u_L`, `-u_R` | / 0.0075, clip ±1 |
| 1 | estimate_valid | ESTIMATED | mask | {0,1} |
| 2–3 | contact_present_L/R | SENSOR | finger–object geom | {0,1} |
| 4–5 | valid_L/R | SENSOR | CoP defined | {0,1} |
| 6,8 | u_L, u_R | SENSOR | pad X; 0 if invalid | / 0.0075 |
| 7,9 | v_L, v_R | SENSOR | pad Z; 0 if invalid | / 0.0085 |
| 10–11 | fn_L/R | SENSOR | `|mj_contactForce[0]|` sum | / 20 N |
| 12 | aperture | DIRECT | mean finger joint | / 0.04 m |
| 13 | ap_dot | DIRECT | mean finger qvel | / 0.08 m/s **PROVISIONAL** |
| 14 | tau_from_secure | DIRECT | `ctrl[7]` | (τ+18)/17 |
| 15–17 | g_h | DIRECT | `R_hand^T g_world` (FK, not object) | / 9.81 |
| 18–19 | v_hx, v_hy | DIRECT | hand linear vel in hand frame | / 0.08 |
| 20 | v_z_world | DIRECT | hand world-z vel (for a2) | / 0.08 |
| 21–23 | w_h | DIRECT | hand angular vel, hand frame | / 2.0 rad/s |
| 24 | hand_z_off_table | DIRECT | `hand_z - TABLE_TOP` (0.40 m **scene**, not object height) | / 0.25 |
| 25 | track_z | DIRECT | `p_des_z - hand_z` | / 0.08 |
| 26 | e_hat_dot | ESTIMATED | Δê / dt_policy if both steps valid else 0 | / 0.08 |

**No PRIVILEGED_GT channels.** No object qpos/qvel/pose/twist, no `r_h`, `v_rel`, `R_rel`, `ω_rel`, object height, clearance, `D_t`.

Wrist: `g_h` from hand orientation. Vertical: hand z vs table + `v_z_world` + tracking residual. Short history: one-step ê-dot only (not 100 ms stack, not RNN).

## Q4 Leak tests

`training/test_observable_obs_no_gt_leak.py` + reward leak tests: **12 passed**.

Poison: cache tactile, move object qpos + `mj_forward`, recompute obs with **same** tactile → identical vector.

Oracle env obs dim still **24**. Observable env **27**.

## Q5 Normalization (diagnostic ICs, ZERO 1 s + RULE)

Binary flags sit at 0/1 so “clip_frac≈1” is **not** analog saturation.

Analog: `e_hat_x` clip **0**; `u,v` clip 0; `g_hx` uses full ±1 (90° wrist) without clip. `v_hx` / `w_hy` clip **~0.18** because RULE commands sit on `v_hx_max` / `w_hy_max` — expected, not ê saturation.

## Sanity set

N=**48** constructed ICs, seed **5478**, `|e_x|∈[3.5,7.5]` mm, 24+/24−, `offset_construct` protocol, **not** eval npz. `results/diagnostics/observable_policy_integration/sanity_ics.npz`.

ZERO mean return **0.50** (1 s). RULE mean return **8.79**.

## Q6–Q8 Termination confusion (RULE, same episode)

`success_obs` vs GT hold-complete on the **observable** episode (which **stops** at `success_obs`):

| | GT yes | GT no |
|---|---|---|
| obs yes | 8 | **40** |
| obs no | 0 | 0 |

Precision 0.17, recall 1.0. Failure_obs vs failure_GT: **tp=0 fp=0 tn=48** (no contact-loss fails on RULE).

**Cause of the 40 “FP” (not a table mix-up):** at `success_obs`, **every** case has `|e_x_GT| ∈ [1.65, 2.63] mm` (inside 3 mm) and bilateral contact, but `recovered()` is false because **`v_rel` is 0.023–0.087 m/s** (often 0.05–0.08). `success_obs` uses `|ê|≤3 mm` for 0.20 s and `|Δê|/dt≤0.02`; pad CoP can look still while the cylinder still has residual object-hand speed. That **is** an unstable-crossing / residual-slip terminal **+8**.

Full 40 listed in `gate.json` `dangerous_fp`. None have `|e_x|>3.5 mm` or wrong sign.

On the **8** episodes where GT hold also completed before obs success: `t_obs - t_GT_hold` min/median/max = **+0.04 / +0.06 / +0.24 s** (obs later). The other 40 never reach GT hold because the episode already ended.

## Q10 Open/regrasp

0.08 s open then reclose: `failure_obs=false`, obs finite, valid_frac=0.94, no GT fallback.

## Q11–Q12 Regressions

`python training/test_recovery4d_interface.py` → **INTERFACE: PASS** (oracle 24-D). recovery4d mapping / no hidden lift / no a3 penalty unchanged.

Observable reward still option-A 100 ms progress, no `D_t` / force / aperture / tracking / a3. Leak test `test_reward_formula_untouched` passed.

## Q13 Gate

**BLOCKER_BEFORE_SAC:** frequent `success_obs` while GT `recovered()` is false (residual `v_rel`). Observation leak and action mapping are **not** blockers.

**PROVISIONAL (not blockers):** 100 ms horizon, ê-dot threshold *as reward/hold feature*, analog scales, simulated tactile, `v_hx`/`w_hy` clip under RULE.

**Not frozen.** Do not start SAC smoke until terminal `success_obs` is addressed **without** secretly putting `v_rel_GT` in the policy. Possible later directions (not implemented): longer ê-dot window, require low `e_hat_dot` **and** low hand speed **and** force stability, or delay +8 until a longer hold — all still observable. **Do not tune from held-out eval.**

## Q14

Freeze hashes recorded in `gate.json` for the current **unfrozen** code, not a smoke-ready freeze.

Paths: `envs/observable_obs.py`, `envs/recovery_env.py` `_policy_obs`, `training/validate_observable_policy_integration.py`, `results/diagnostics/observable_policy_integration/`.
