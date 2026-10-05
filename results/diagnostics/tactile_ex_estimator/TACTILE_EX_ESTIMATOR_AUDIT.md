# Tactile e_x Estimator Audit

**used_final_eval_data: false**

Held-out eval SHA256 unchanged:

- offset `f041a2c586ed98be47560fb863aa72be2a7cad30b6618cebe9dbdda2bce2ad63`
- impact `7ef30ec8c81402cc7446b007aab7c070290231e704767845e47aaa9081cb8b5e`

Frozen prior (`RECENTER_OBSERVABILITY_AUDIT.md`): S0–S2 are **fundamentally ambiguous** for signed \(e_x\). This step does **not** retry S0–S2. It asks whether a **finger-local spatial tactile abstraction** still carries signed \(e_x\) after realistic (parameterized, synthetic) sensing limits.

No SAC. No RecoveryEnv reward/observation change. No frozen RULE edit. No hardware claim.

## 1. Sensor abstraction — VERIFIED FROM CODE

Files: `sensors/spatial_tactile.py`, `sensors/ex_estimator.py`.

Public reading (finger pad frame only):

| field | meaning |
|---|---|
| `contact_present_L/R` | at least one finger–object contact geom |
| `valid_L/R` | CoP defined (`sum Fn ≥ 0.01 N`) |
| `u, v` | pad coordinates; **NaN if invalid** (zero is a legal CoP) |
| `fn_L/R` | summed `|mj_contactForce[0]|` |

**Not in the API:** object qpos/qvel/pose, \(e_x\) GT, world-frame `contact.pos`.

Pad from `assets/panda_torque.xml` `fingertip_pad_collision_1`: pos `[0, 0.0055, 0.0445]` m, half-size `[0.0085, 0.004, 0.0085]` m. Inner-face y = 0.0015 m.

**Frames (must not assume u == e_x on both fingers):**

- Left finger is identity child of `hand` → finger X = hand X. `R_hand^T R_left` diag ≈ `(1,1,1)`.
- Right finger XML `quat="0 0 0 1"` (wxyz) = 180° about Z → finger X = **−hand X**. `R_hand^T R_right` diag ≈ `(-1,-1,1)`.

So \(u_L \approx e_x\) and \(-u_R \approx e_x\) for this vertical cylinder grasp.

**CoP:** \(\sum_i F_{n,i}\,p_i / \sum_i F_{n,i}\) in the finger frame. Empty contact → `valid=False`, u/v = NaN. Dust force `< 0.01 N` → invalid CoP even if a geom contact exists.

World → finger-local → world reconstruction: max error **1.1e-16 m** on 14 ICs (`geometry_check.json`). SUPPORTED BY RAW DIAGNOSTIC.

Leak test: `training/test_tactile_no_gt_leak.py` — **4 passed**. Sensor/estimator modules do not read `data.qpos/qvel` or object `xpos`.

## 2. Dataset — constructed ICs, not eval npz

`results/diagnostics/tactile_ex_estimator/`

- **n_rows = 16440**, dt = 2 ms
- \(e_x\) targets: −7.5 … +7.5 mm × settle 0.20 / 0.30 s
- Families: ZERO (0.12 s), SLIP τ=−2 (0.40 s), BRAKE slip then resecure (0.50 s), WRIST a0=0.30 (0.15 s), OPEN τ=−1 (0.15 s), RULE (1.20 s)
- Sensor: `sensor_outputs.npz`. Labels: `gt_labels.npz` (`EVAL_ONLY_GT_*` only).
- Split (fixed before scoring): train ZERO bilateral / test RULE bilateral; also leave-one-family-out on bilateral rows.

## 3. Estimator

**A. Geometric (selected):** mean of valid \(\{u_L\} \cup \{-u_R\}\). Unilateral allowed. No contact → `estimate_valid=False` (no fill with 0).

**B/C. Linear / ridge** on \([u_L, u_R]\), train ZERO n=840, test RULE n=7098:

- linear weights `[7.2e-6, 0.598, -0.401]`
- ridge similar
- RULE MAE 0.195 mm vs geometric **0.189 mm**

Linear is **not better**. No neural net.

## 4. Clean accuracy — SUPPORTED BY RAW DIAGNOSTIC

ZERO last sample, n=14 ICs:

| pair | Pearson | Spearman |
|---|---|---|
| \(u_L\) vs \(e_x\) | 0.9996 | — |
| \(-u_R\) vs \(e_x\) | 0.9992 | — |
| \(\hat e_x\) vs \(e_x\) | **0.9999** | 0.987 |

Example IC −7.50 mm: \(u_L=-7.497\) mm, \(u_R=+7.497\) mm, \(\hat e=-7.497\) mm.

All families, **any-contact valid** n=15770:

| | all | + offsets | − offsets |
|---|---|---|---|
| MAE mm | **0.109** | 0.070 | 0.064 |
| RMSE mm | 0.280 | | |
| sign error (\|e\|≥1 mm) | **0.000** | | |
| P(\|err\|<1 mm) | 0.969 | | |
| P(\|err\|<2 mm) | **0.999** | | |
| L/C/R zone acc (3 mm bins, analysis only) | 0.999 | | |

Bilateral-only n=15179: MAE 0.108 mm, sign error 0, P(\|err\|<2 mm)=1.0.

Train ZERO / test RULE geometric n=7098: MAE **0.189 mm**, sign error **0**, P(\|err\|<2 mm)=**1.0**.

Leave-one-family-out geometric MAE: ZERO 0.027, WRIST 0.030, BRAKE 0.032, SLIP ~0.05, RULE 0.189, OPEN 0.048 mm (OPEN bilateral n=112 only). No family collapse like S2.

Per-offset MAE (mm): 7.5→0.041, 5.0→0.030, 2.5→0.247, 0→0.136, −2.5→0.262, −5.0→0.037, −7.5→0.026. Larger MAE near ±2.5 mm is RULE recentering through the 3 mm band, not sign confusion.

## 5. Coverage (as important as MAE)

Overall: any-contact valid **0.959**; bilateral **0.923** (n=16440).

| family | coverage any | bilateral | MAE mm (valid) | n steps |
|---|---|---|---|---|
| ZERO | 1.000 | 1.000 | 0.027 | 840 |
| WRIST | 1.000 | 1.000 | 0.030 | 1050 |
| SLIP | 0.998 | 0.939 | 0.051 | 2800 |
| BRAKE | 1.000 | 0.985 | 0.034 | 3500 |
| RULE | 0.997 | 0.986 | 0.188 | 7200 |
| OPEN | **0.386** | **0.107** | 0.157 | 1050 |

RULE OPEN_REGRASP coverage 0.53 (n=34). No-contact windows are **undefined**, not guessed.

## 6. Synthetic degradations — SYNTHETIC SENSOR SENSITIVITY RESULT (not hardware)

Geometric estimator, all valid rows unless noted.

| degradation | MAE mm | P(\|err\|<2 mm) | sign err | 50 ms false-progress |
|---|---|---|---|---|
| ideal | 0.109 | 0.999 | 0 | 0.013 |
| res 0.25 mm | 0.108 | 0.999 | 0 | ~0 |
| res 0.5 mm | 0.120 | 0.999 | 0 | 0 |
| res 1.0 mm | 0.349 | 0.999 | 0 | 0 |
| res 2.0 mm | 0.539 | 0.999 | 0 | 0.007 |
| σ 0.25 mm | 0.210 | 0.999 | 0 | 0.005 |
| σ 0.5 mm | 0.338 | 0.998 | 0 | 0.005 |
| σ 1.0 mm | 0.606 | 0.988 | 0.0003 | 0.006 |
| σ 2.0 mm | 1.163 | 0.834 | 0.014 | (MAE exceeds 1 mm) |
| latency 50 ms | 0.226 | 0.980 | 0.0076 | — |
| dropout 10% | 0.111 | 0.999 | 0 | 0.013 |

Task scale is 3 mm. Ideal / 0.5 mm-class errors stay well inside that. **σ=2 mm** is not task-sufficient. Quantization to 1–2 mm still keeps sign and P(\|err\|<2 mm) high because bias is a fraction of a millimetre, not a sign flip.

## 7. Dynamic / failure slices

| slice | n valid | MAE mm | sign err | P(\|err\|<2 mm) |
|---|---|---|---|---|
| SLIP family | 2794 | 0.051 | 0 | 0.999 |
| WRIST family | 1050 | 0.030 | 0 | 1.0 |
| BRAKE family | 3500 | 0.034 | 0 | 0.999 |
| OPEN family | 405 | 0.157 | 0 | 0.963 |
| RULE CONTROLLED_SLIP | 929 | 0.024 | 0 | 1.0 |
| RULE SLIP_ALIGN | 1691 | 0.113 | 0 | 1.0 |
| RULE BRAKE mode | 720 | 0.436 | 0 | 1.0 |
| RULE SECURE | 1683 | 0.301 | 0 | 1.0 |
| unilateral only | 591 | 0.147 | 0 | 0.968 |
| Fn_L+Fn_R < 1 N | 55 | 0.215 | 0 | 0.927 |
| v_rel > 0.02 | 6194 | 0.206 | 0 | 0.998 |

RULE traj on this grid: **2 success / 10 fail** by `recovered()` at last step (1.2 s, 3 mm + hold — **success definition unchanged**). Fail steps n=6000 MAE 0.213 mm, sign err 0, P(\|err\|<2 mm)=1.0. Success steps n=1181 MAE 0.061 mm.

The mapping is **not** static-only. Worst *defined* estimates: RULE BRAKE/SECURE (still <0.5 mm MAE, no sign errors) and OPEN/low-force/unilateral (coverage hole + slightly higher MAE).

## 8. Reward feasibility — analysis only, SAC reward not modified

\(\Delta|\hat e_x|\) vs \(\Delta|e_{x,\mathrm{GT}}|\). Deadzone 0.1 mm. Consecutive 2 ms steps are smaller than that deadzone (~0.01 mm if 7.5 mm closes in 1.2 s), so 2 ms sign-agree is **not** a reward test.

| horizon | set | n_pairs | sign agree | false progress | false regression | Pearson of Δ\|e\| |
|---|---|---|---|---|---|---|
| 2 ms | all | 3486 | 0.22 | 0.012 | 0.011 | 0.13 |
| 20 ms | RULE | 2246 | 0.43 | 0.019 | 0.001 | 0.72 |
| **50 ms** | all | 4370 | **0.62** | **0.013** | 0.000 | **0.76** |
| **50 ms** | RULE | 3970 | **0.68** | **0.014** | 0.000 | **0.82** |
| **100 ms** | all | 5916 | **0.79** | **0.000** | 0.0007 | **0.85** |
| **100 ms** | RULE | 5490 | **0.85** | **0.000** | 0.0005 | **0.87** |

False progress = estimated improvement while GT |e_x| worsens.

Remaining disagreement at 50 ms is mostly **mixed deadzone** (one side <0.1 mm): 0.36 all / 0.30 RULE — not opposite-signed progress.

**INFERENCE:** a future dense term \(\Delta|\hat e_x|\) is **plausible if** it is (i) gated on `estimate_valid`, (ii) applied on ≥50–100 ms, not 2 ms chatter. It is **not** yet a coefficient choice. Noise σ≥0.5 mm wrecks 50 ms sign-agree even when MAE stays <0.4 mm.

## 9. Policy-observation candidates — not implemented

Legitimate later obs: `u_L,v_L,u_R,v_R`, `valid_L/R`, `fn_L/R`, \(\hat e_x\), `estimate_valid`, plus existing proprioception/aperture. **Not** object pose or GT \(e_x\). Policy \(\hat e\) and reward \(\hat e\) need not be the same, but both need this identifiability.

## 10. Trigger implication — D_t not redesigned

Observable candidates: \(|\hat e_x|\) vs 3 mm, \(\Delta u\) migration, `valid` falling, force drop. Keep trigger, reward, and GT success separate.

## Q1–Q10

**Q1.** Yes. Finger-local \(u_L\) and \(-u_R\) retain signed \(e_x\) (Pearson 0.9996 / 0.9992 on ZERO last, n=14). Reconstruction error 1e-16 m. SUPPORTED BY RAW DIAGNOSTIC. HARDWARE-UNVALIDATED.

**Q2.** Geometric mean of valid \(u_L\) and \(-u_R\). Linear/ridge do not beat it on RULE.

**Q3.** Clean any-contact MAE **0.109 mm**, sign error **0**, n=15770. ZERO last MAE ~0.027 mm. RULE (train ZERO) MAE **0.189 mm**, sign error 0.

**Q4.** Resolution 1–2 mm: MAE 0.35–0.54 mm, sign still 0. Noise σ=1 mm: MAE 0.61 mm; σ=2 mm: MAE 1.16 mm and P(\|err\|<2 mm)=0.83. Latency 50 ms: MAE 0.23 mm. Dropout 10%: MAE unchanged if remaining pads valid. SYNTHETIC.

**Q5.** Yes **in this sim, ideal and ≤0.5 mm-class noise/resolution**: P(\|err\|<2 mm)≥0.998 vs 3 mm task scale. σ=2 mm is **not** sufficient. HARDWARE-UNVALIDATED.

**Q6.** Valid on **0.959** of diagnostic steps (any contact); **0.923** bilateral. OPEN: 0.386 any / 0.107 bilateral. RULE: 0.997 any.

**Q7.** Fails to *exist* under open/regrasp and contact loss. When defined, slightly worse under RULE BRAKE/SECURE, unilateral, low force, high v_rel — still no sign errors on this set. Wrist and slip families are easy.

**Q8.** State \(\hat e_x\) is accurate enough to *consider* an observable dense reward next, **only** with validity gating and a ≥50–100 ms difference. 2 ms \(\Delta\) is not justified. Do not set reward weights in this step.

**Q9.** 50 ms false-progress **0.013** (all) / **0.014** (RULE). 100 ms false-progress **0**.

**Q10.** Yes: spatial tactile is a **viable SIMULATED sensing assumption** for signed \(e_x\) here. It remains **HARDWARE-UNVALIDATED**. `panda_torque.xml` still has only scalar `touch_*` sites.

## Paths

- `sensors/spatial_tactile.py`, `sensors/ex_estimator.py`
- `training/validate_tactile_ex_estimator.py`, `training/test_tactile_no_gt_leak.py`
- `results/diagnostics/tactile_ex_estimator/` (`sensor_outputs.npz`, `gt_labels.npz`, `estimator_metrics.json`, `provenance.json`, `geometry_check.json`, `figures/`)
