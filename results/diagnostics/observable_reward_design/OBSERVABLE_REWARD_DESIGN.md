# Observable Recovery Reward Design

**used_final_eval_data: false**

Eval SHA256 unchanged (hashed, files not loaded):

- offset `f041a2c586ed98be47560fb863aa72be2a7cad30b6618cebe9dbdda2bce2ad63`
- impact `7ef30ec8c81402cc7446b007aab7c070290231e704767845e47aaa9081cb8b5e`

No SAC training. Oracle reward formula in `RecoveryEnv._reward` is unchanged. New mode: `observable_tactile`.

## Timing audit — VERIFIED FROM CODE

| quantity | value |
|---|---|
| `physics_dt` | 0.002 s (`config/sim.yaml`) |
| `n_substeps` | 10 |
| `policy_dt` = `dt_policy` | **0.020 s** (`GraspSim`: timestep × n_sub) |
| `RecoveryEnv.step` | one policy tick (`recovery4d_tick` loops `n_sub` mj_steps) |
| reward evaluation | **once per policy step**, not per physics step |

A `-0.01` time penalty is therefore **−0.50 / s**, not −5 / s. Same as the pre-existing oracle (`lambda_t` in `_reward`). Do not "preserve" 0.01 at 500 Hz.

`T_progress = 0.100 s` = **5 policy steps**. Provenance: tactile estimator audit, 100 ms false-progress = 0 on that diagnostic set (`envs/observable_reward.py` `T_PROGRESS_PROVENANCE`). **PROVISIONAL**, not a physical constant.

## Modes

Constructor / `config/*.yaml` `reward.mode`:

- `oracle` (default): GT \(e_x\) potential every policy step; GT `recovered()` / `fail_kind` terminate.
- `observable_tactile`: geometric \(\hat e_x\) from `sensors/spatial_tactile.py`; **no** object pose/twist/GT success in dense or training terminal reward.

Policy observation vector is still the 24-D privileged `observe_recovery4d` (unchanged this step). \(\hat e_x\) is in `info` for logging. Replacing obs is a later step.

## Q1 — Exact observable reward (policy step \(k\))

\[
r_k = r^{\mathrm{prog}}_k + r^{\mathrm{cap}}_k + r^{t}_k + r^{a}_k + r^{\mathrm{term}}_k
\]

**Progress (option A — emit once per interval):**

Let \(\phi(\hat e) = -|\hat e|/0.0075\). Window length \(N=5\) (`T_progress/dt_policy`).

- If `estimate_valid` is false: \(r^{\mathrm{prog}}=0\), **discard** the progress reference (no zero fill, no GT).
- If valid after a gap: set reference to current \(\hat e\), \(r^{\mathrm{prog}}=0\) (do not compare across the gap).
- If valid and reference valid: increment window counter. When counter hits \(N\):

\[
r^{\mathrm{prog}} = \phi(\hat e_{\mathrm{now}}) - \phi(\hat e_{\mathrm{ref}})
\]

then ref ← now, counter ← 0. Otherwise \(r^{\mathrm{prog}}=0\).

So a 100 ms displacement is credited **once**, not 5× at 20 ms and not 50× at 2 ms.

**Contact (tactile bilateral_valid only):**

\[
r^{\mathrm{cap}} = 0.02 \times (1 \text{ if bilateral else } -0.5)
\]

i.e. +0.02 / −0.01. Same numerical scale as oracle `lambda_cap`.

**Time:** \(r^{t} = -0.01\) per policy step.

**Action:** \(r^{a} = -0.01(a_0^2+a_1^2+a_2^2)\). No \(a_3\). VERIFIED FROM CODE (`ObservableTactileReward._action`, oracle `_reward` unchanged).

**Terminal (training only):**

- `success_obs` → \(r^{\mathrm{term}}=+8\), terminate
- `failure_obs` → \(r^{\mathrm{term}}=-8\), terminate
- timeout → truncate, \(r^{\mathrm{term}}=0\)

No \(D_t\), no force max, no aperture target, no tracking error.

## success_obs (TRAINING ONLY) — PROVISIONAL

For 0.20 s continuously (`success_hold`, policy clock):

1. `estimate_valid`
2. `bilateral_valid` (both pads have a force-weighted CoP)
3. \(|\hat e_x| \le 3\,\mathrm{mm}\) (`E_TOL`, analysis/task tolerance unchanged)
4. if previous \(\hat e\) valid: \(|\Delta\hat e|/dt_{\mathrm{policy}} \le 0.02\,\mathrm{m/s}\)

Clause 4 is **PROVISIONAL**: same number as GT `V_REL_TOL`, but it is **contact-location migration**, not \(v_{\mathrm{rel}}\). No GT \(\omega_{\mathrm{rel}}\), height, or clearance.

`success_GT` remains `recovered()` (physical) for evaluation.

## failure_obs (TRAINING ONLY)

Both `contact_present_L` and `contact_present_R` false for 0.15 s (`contact_loss_hold`).

**Invalid \(\hat e\) is not failure.** Unilateral or low-force CoP can zero progress without terminating.

GT `fail_kind` (scene / drop / clearance) is **logging only** in observable mode. Observable training does not terminate on GT height.

Timeout is truncation, not `failure_obs`.

## Scripted sanity (constructed 6 mm IC, 2 s cap)

`training/validate_observable_reward.py` — not eval npz.

| policy | obs return | oracle return | prog | \|e_x\| mm | success_obs | success_GT (oracle env) | failure_obs | valid frac | term obs |
|---|---|---|---|---|---|---|---|---|---|
| ZERO | 0.999 | 1.000 | −0.001 | 6.00 | n | n | n | 1.00 | — |
| RULE | **8.888** | **8.981** | 0.428 | 2.52 | **y** | **y** | n | 1.00 | success_obs |
| WRIST | 1.175 | 1.189 | 0.265 | 3.91 | n | n | n | 1.00 | — |
| SLIP | 0.603 | 0.605 | −0.007 | 6.04 | n | n | n | 1.00 | — |
| SLIP_BRAKE | 0.969 | 0.970 | −0.001 | 6.00 | n | n | n | 1.00 | — |
| OPEN_SHORT (0.08 s open) | 0.875 | 0.875 | −0.005 | 6.04 | n | n | **n** | 0.99 | — |
| CONTACT_LOSS | **−8.26** | −8.14 | 0 | 6.08 | n | n | **y** | 0.31 | failure_obs |
| HX_OSC | 0.749 | 0.750 | −0.001 | 6.00 | n | n | n | 1.00 | — |

Episode confusion vs oracle-env GT labels (n=8 policies): success **tp=1 fp=0 fn=0**; failure **tp=1 fp=0 fn=0**.

On the RULE observable episode, `success_obs` fired at 0.98 s (`n=49`) while GT hold completed at 1.06 s in the oracle env (`n=53`). Final \(|e_x|=2.52\,\mathrm{mm}\) already inside 3 mm. Not counted as obs-yes / GT-no against the oracle env.

## Exploit checklist

| id | result |
|---|---|
| A ZERO high vs recovery | **No.** 0.999 vs RULE 8.89 |
| B squeeze without recenter | ZERO/secure hold ≈ time+contact only (~1.0 / 2 s), no +8 |
| C opening farms | OPEN_SHORT 0.875 < ZERO |
| D contact loss avoids neg | CONTACT_LOSS **−8.26** |
| E wrist/HX osc farms progress | HX_OSC progress **−0.001** |
| F/G invalid→valid fake progress | reference discarded on invalid; OPEN_SHORT progress ≈ 0 |
| H obs success, GT unrecovered | **No** on this set (oracle env recovered) |
| open/regrasp auto-fail | OPEN_SHORT `failure_obs=false` |

Unit test: 7→2→7→2 mm \(\hat e\) windows net **exactly 0** progress (`test_oscillation_net_near_zero`).

## Leakage

`training/test_observable_reward_no_gt_leak.py`: **6 passed**. `observable_reward.py` has no `physical_pack`, `data.qpos/qvel`, or `o["e_x"]`. Poisoning extra GT keys in the tactile dict does not change `terms()`.

GT channels in `RecoveryEnv.step` (`physical_pack`, `recovered`) still run for **info/oracle mode** only. Observable `r` comes from `ObservableTactileReward.terms(tactile, a)`.

Existing `training/test_recovery4d_interface.py` **INTERFACE: PASS** (oracle default).

## Remaining PROVISIONAL before SAC smoke

1. `T_progress = 0.10 s`
2. `e_hat_dot_tol = 0.02 m/s` on \(\hat e\)
3. Unchanged lambdas (0.02 / 0.01 / 0.01 / ±8) copied from oracle scale
4. Policy **observation** still privileged
5. Hardware tactile unvalidated
6. success_obs can lead GT hold by a few policy steps
7. Diagnostic n=8 scripted policies on one 6 mm IC — not a coverage proof

## Paths

- `envs/observable_reward.py`, `envs/recovery_env.py` (`reward_mode`)
- `training/validate_observable_reward.py`, `training/test_observable_reward_no_gt_leak.py`
- `results/diagnostics/observable_reward_design/`
