# Observable Reward Signal Audit

**used_final_eval_data: false**

This audit is signal discovery only. It does not modify RecoveryEnv, reward, success, recovery4d mapping, frozen RULE, or held-out eval sets. It does not train SAC.

## 1. Goal and information boundary

Oracle progress is \(P_{GT}=-|e_{x,GT}|\) and \(\Delta P_{GT}=|e_{x,GT}(t)|-|e_{x,GT}(t+1)|\), from simulator object pose in the hand frame (`obs_from_sim` / `physical_pack`). That quantity is **EVAL_ONLY_GT**. A future real-world learning signal must be computable from physically obtainable sensing.

Simulation GT is used here only as an **offline analysis label**.

## 2. Available observable signals

| signal | exact code source | unit | frame | class | realistic source | currently logged in recovery4d obs? |
|---|---|---|---|---|---|---|
| arm q, qdot | `data.qpos[ids.arm_jnt]`, `qvel[ids.arm_dof]` `envs/ids.py` | rad, rad/s | joint | DIRECT | joint encoders | no (recovery4d obs omits) |
| hand pose / twist | `data.xpos/xmat[hand_body]`, `mj_objectVelocity` `body_twist` | m, rad/s | world | ESTIMATED | FK from encoders | no (g_h only) |
| p_des − p_hand | `jacobian_controller.cartesian_torque` e_p; FSM `p_des` | m | world | DIRECT (controller) | Cartesian servo residual | no |
| ori tracking error | `rotation_error` / `ori_error_deg` | rad / deg | world | DIRECT (controller) | Cartesian servo residual | no |
| v_cmd, w_cmd | `sim.fsm.v_cmd/w_cmd` | m/s, rad/s | world | DIRECT | commanded twist | no |
| tendon ctrl[7] / tau | `data.ctrl[7]` | N (tendon) | actuator | DIRECT | gripper command | yes, scaled |
| aperture, finger q | `finger_opening` = mean finger joint pos | m | joint | DIRECT | finger encoders | aperture yes |
| finger qdot | `qvel[ids.finger_dof]` | m/s | joint | DIRECT | finger encoders | no |
| nL, nR | `obs_from_sim` contact geom counts | count | — | SENSOR | tactile / contact | no (only Fn) |
| Fn_L, Fn_R | `mj_contactForce` via `finger_object_normals` / `obs_from_sim` | N | contact normal | SENSOR | tactile/FT; **not** automatically DIRECT | yes |
| touch_left/right | `data.sensordata` sensors `touch_left/right` `panda_torque.xml` | touch | fingertip site | SENSOR | tactile | no |
| force imbalance/ratio | derived from Fn | N / 1 | — | SENSOR | same as Fn | no |
| D_t, Ddot | `DeteriorationMeter.compute` `envs/deterioration.py` | 1, 1/s | mixed | **PRIVILEGED_GT mix** | needs object pose/twist vs reference + forces | no |
| D_p, D_v, D_θ, D_ω | ‖Δp_rel‖, ‖v_rel‖, θ_rel, ‖ω_rel‖ vs captured ref | m, m/s, rad, rad/s | world/hand | PRIVILEGED_GT | object state | no |
| D_C, lost_contact | \|fL−fR\|/(sum+ε) + lost | 1 | — | SENSOR | tactile | no |
| e_x, object pose/twist | `Rh.T@(po-ph)`, object xpos/vel | m, m/s | hand/world | PRIVILEGED_GT | vision/tactile estimator **missing** | e_x in GT pack only |

MuJoCo contact force is classified **SENSOR**, not DIRECT. Object-relative quantities are **PRIVILEGED_GT** even though MuJoCo exposes them.

## 3. Diagnostic dataset and provenance

- Path: `D:\NUS\CS5478 Intelligent Robots\CS5478_Project\results\diagnostics\observable_reward_signal_audit`
- Rows: 12489 at physics dt (nominal 0.002 s)
- ICs (mm): [-7.50014547271084, -4.999514869987231, -2.496359324479269, 2.491286731124927, 4.9999659657016595, 7.500607952035806]
- Families: ['A_zero_hold', 'B_rule', 'D_slip', 'E_slip_brake', 'F_open_regrasp', 'H_wrist_only']
- Construction: one airborne lift (`offset_construct.lift_to_airborne`), table disabled, `apply_rel_pose` at training-range e_x ∈ {±2.5, ±5.0, ±7.5} mm, settle 0.25 s.
- `used_final_eval_data`: False
- Held-out hashes unchanged: offset `f041a2c586ed98be…` impact `7ef30ec8c81402cc…`
- Analysis code: `training/observable_signal_audit.py`

Trajectory counts by family:

- `A_zero_hold`: 6 trajs
- `B_rule`: 6 trajs
- `D_slip`: 6 trajs
- `E_slip_brake`: 6 trajs
- `F_open_regrasp`: 6 trajs
- `H_wrist_only`: 6 trajs

RULE `final_recovery` (0.20 s physical recovered() hold):
- recovery=True: 2 (`B_rule_ex+0.0075`, `B_rule_ex-0.0075`)
- recovery=False: 4 — several still **reduced** `|e_x|` (e.g. `B_rule_ex±0.0050` min `|e_x|` ~0) without meeting the 0.20 s v_rel/ω hold. Progress ≠ final physical recovery.

## 4. Ground-truth labels used only for offline analysis

Prefixed `EVAL_ONLY_GT_*` in `rows.csv` / `rows.npz`: e_x/y/z, |e_x|, P, v_rel, ω_rel, object pose, recovered_now, drop, scene, airborne, stable.

Progress: `EVAL_ONLY_GT_dP = |e|(t) − |e|(t+1)` within each trajectory. Positive = true lateral recenter.

These are **not** proposed as observable reward inputs.

## 5. Signal-by-signal analysis

Correlations are **Pearson/Spearman of Δx vs ΔP_GT** unless noted. Evidence is correlational on interventional tapes (ZERO / RULE / recovery4d open-loop), not a trained policy.

| candidate | n (all Δ) | Pearson Δx↔ΔP | Spearman | sign agree | +e_x Pearson | −e_x Pearson | RULE Pearson | ZERO Pearson |
|---|---|---|---|---|---|---|---|---|
| aperture | 12489 | -0.170 | -0.073 | +0.490 | -0.382 | +0.037 | -0.220 | -0.181 |
| aperture_vel | 12489 | -0.039 | +0.011 | +0.504 | -0.095 | +0.011 | -0.051 | +0.001 |
| tau | 12489 | -0.004 | +0.025 | +0.458 | -0.141 | +0.117 | -0.009 | NA |
| fn_l | 12489 | +0.017 | -0.046 | +0.507 | +0.128 | -0.070 | +0.022 | +0.012 |
| fn_r | 12489 | +0.032 | -0.031 | +0.468 | +0.146 | -0.076 | +0.046 | -0.004 |
| fn_sum | 12489 | +0.027 | -0.044 | +0.458 | +0.161 | -0.083 | +0.036 | +0.002 |
| fn_imbalance | 12489 | +0.007 | -0.056 | +0.493 | -0.006 | +0.018 | +0.009 | +0.010 |
| fn_ratio | 12489 | +0.006 | -0.055 | +0.494 | -0.028 | +0.037 | +0.012 | +0.010 |
| touch_l | 12489 | +0.017 | -0.043 | +0.501 | +0.080 | -0.036 | +0.019 | +0.002 |
| touch_r | 12489 | +0.014 | -0.034 | +0.474 | +0.078 | -0.055 | +0.015 | -0.000 |
| touch_sum | 12489 | +0.019 | -0.036 | +0.461 | +0.101 | -0.054 | +0.022 | +0.001 |
| nL | 12489 | +0.025 | +0.000 | +0.484 | +0.044 | +0.009 | +0.041 | +0.004 |
| nR | 12489 | +0.022 | -0.006 | +0.486 | +0.042 | +0.004 | +0.049 | -0.004 |
| bilat | 12489 | +0.026 | +0.027 | +0.484 | +0.064 | -0.009 | +0.042 | NA |
| track_pos_err | 12489 | -0.013 | +0.192 | +0.614 | -0.012 | -0.014 | -0.010 | -0.188 |
| track_ori_deg | 12489 | -0.007 | -0.304 | +0.440 | -0.007 | -0.007 | -0.007 | -0.147 |
| track_vel_err | 12489 | -0.012 | -0.215 | +0.406 | -0.132 | +0.100 | -0.015 | +0.029 |
| hand_speed | 12489 | -0.054 | -0.209 | +0.412 | -0.040 | -0.067 | -0.059 | +0.029 |
| v_cmd_norm | 12489 | +0.027 | +0.015 | +0.533 | -0.115 | +0.162 | +0.034 | NA |
| w_cmd_norm | 12489 | NA | -0.010 | NA | NA | NA | NA | NA |
| finger_qd_mean | 12489 | -0.039 | +0.011 | +0.504 | -0.095 | +0.011 | -0.051 | +0.001 |
| D | 12489 | +0.031 | -0.026 | +0.478 | -0.027 | +0.086 | +0.054 | +0.074 |
| Ddot | 12489 | -0.020 | +0.051 | +0.526 | -0.042 | +0.000 | -0.032 | +0.017 |
| D_C | 12489 | -0.010 | -0.055 | +0.494 | -0.048 | +0.023 | -0.017 | +0.010 |
| D_lost | 12489 | -0.026 | +0.025 | +0.516 | -0.064 | +0.009 | -0.042 | NA |

Instantaneous x vs |e_x_GT| (level, not progress):

| candidate | Pearson x↔|e_x| | Spearman |
|---|---|---|
| D | -0.385 | -0.259 |
| fn_imbalance | +0.006 | -0.296 |
| fn_sum | -0.196 | -0.358 |
| aperture | +0.510 | +0.524 |
| track_pos_err | -0.237 | -0.167 |
| bilat | -0.047 | -0.042 |
| D_C | +0.055 | -0.208 |

## 6. Temporal-history analysis

Windows 20/50/100/200 ms on physics samples. Compare instantaneous Δx vs slope of x.

| signal | inst Δ Pearson | slope 20 ms | 50 ms | 100 ms | 200 ms |
|---|---|---|---|---|---|
| D | +0.031 | +0.180 | +0.092 | +0.036 | +0.032 |
| fn_imbalance | +0.007 | +0.095 | +0.041 | +0.007 | -0.012 |
| aperture | -0.170 | +0.036 | +0.064 | +0.032 | +0.006 |
| track_pos_err | -0.013 | -0.071 | -0.173 | -0.125 | -0.067 |
| fn_sum | +0.027 | -0.098 | -0.249 | -0.147 | -0.077 |
| D_C | -0.010 | +0.131 | +0.121 | +0.037 | +0.007 |

If slope correlations are not materially stronger than instantaneous Δ, a short history is still likely needed for **events** (contact persistence), even if not for a dense signed progress proxy.

## 7. D_t analysis

`D` is a weighted sum of privileged object-relative terms plus contact imbalance (`envs/deterioration.py` lines 120–126). It is **not** a real-world-only signal.

- ΔD vs ΔP_GT all: Pearson +0.031, Spearman -0.026, n=12489
- +offset Pearson -0.027; −offset Pearson +0.086
- RULE Pearson +0.054; ZERO Pearson +0.074

Counterexample **counts** (physics steps):
- D_decreases_but_abs_e_worsens: n=383
- D_decreases_without_bilateral: n=129
- abs_e_improves_but_D_worsens: n=327
- D_improves_while_still_displaced: n=2802

Raw examples (first listed in `analysis.json`):

| tag | traj | t | mode | D | ΔD | |e_x| | ΔP | bilat |
|---|---|---|---|---|---|---|---|---|
| D_down_|e|_up | `B_rule_ex-0.0075` | 5.6080 | OPEN_REGRASP | 12.7930 | -2.19842 | 0.00002 | -0.000415 | 0 |
| D_down_|e|_up | `B_rule_ex-0.0075` | 5.6100 | RECLOSE | 10.5946 | -0.88742 | 0.00043 | -0.000338 | 1 |
| D_down_|e|_up | `B_rule_ex-0.0075` | 5.6120 | RECLOSE | 9.7071 | -0.62342 | 0.00077 | -0.000252 | 1 |
| D_down_|e|_up | `B_rule_ex-0.0075` | 5.6140 | RECLOSE | 9.0837 | -0.46486 | 0.00102 | -0.000187 | 1 |
| D_down_|e|_up | `B_rule_ex-0.0075` | 5.6160 | RECLOSE | 8.6189 | -0.34544 | 0.00121 | -0.000140 | 1 |
| |e|_down_D_up | `B_rule_ex-0.0075` | 4.8260 | CONTROLLED_SLIP | 7.8017 | 1.97002 | 0.00678 | 0.000012 | 1 |
| |e|_down_D_up | `B_rule_ex-0.0075` | 4.8460 | CONTROLLED_SLIP | 7.9283 | 0.00157 | 0.00659 | 0.000013 | 1 |
| |e|_down_D_up | `B_rule_ex-0.0075` | 4.8480 | CONTROLLED_SLIP | 7.9298 | 0.00214 | 0.00658 | 0.000013 | 1 |
| |e|_down_D_up | `B_rule_ex-0.0075` | 4.8500 | CONTROLLED_SLIP | 7.9320 | 0.00221 | 0.00656 | 0.000013 | 1 |
| |e|_down_D_up | `B_rule_ex-0.0075` | 4.8520 | CONTROLLED_SLIP | 7.9342 | 1.89276 | 0.00655 | 0.000012 | 1 |
| D_down_no_bilat | `B_rule_ex-0.0075` | 4.6240 | SLIP_ALIGN | 7.0647 | -1.64395 | 0.00746 | 0.000010 | 0 |
| D_down_no_bilat | `B_rule_ex-0.0075` | 4.7320 | SLIP_ALIGN | 9.1320 | -1.61049 | 0.00718 | 0.000018 | 0 |

## 8. Policy-information vs reward-information

A signal can inform the **policy** (what state am I in?) without having a safe **desired direction** for a reward.

| signal | POLICY_INFO | REWARD_INFO |
|---|---|---|
| Fn_L/R, nL/nR, touch | yes — grasp loaded? | no — larger force ≠ recenter |
| aperture, tau | yes — gripper regime | no — open/close are modes, not progress |
| tracking error | yes — servo lag | no — can be reduced by freezing a bad grasp |
| D_t | mixed (mostly GT) | no as dense reward — see counterexamples |
| ΔP_GT / e_x | oracle only | oracle only |
| force imbalance | maybe slip/asymmetry | not signed recenter; may shrink by dropping a finger |

## 9. Proxy-exploit analysis

- ZERO (do nothing, displaced grasp): mean ΔP=+0.000, mean ΔD=+0.000, std |e|=+0.002.
- WRIST-only (a0=+1, tau=−18): mean ΔP=-0.000; per-traj Δ|e_x| end−start=[-0.000448029050539932, -0.00028001517089944715, -0.00024425432074627346, 0.00035649153611257145, 0.00039418611724193354, 0.0004163910853026507].

| candidate | proxy risk | evidence |
|---|---|---|
| maximize −D or −ΔD | HIGH | D can fall while |e_x| stays large or worsens (counts in §7); ZERO still has D dynamics from privileged v_rel/C |
| maximize Fn_sum / squeeze | HIGH | secure hold already ~9 N; extra squeeze does not reduce |e_x| (ZERO/secure) |
| minimize force imbalance | HIGH | zero-contact sets imbalance→0 without capture |
| minimize tracking error | HIGH | ZERO freeze can keep small Cartesian error while object remains offset |
| aperture slope | HIGH | opening changes aperture without recenter; drop also changes it |
| wrist / ori error | HIGH | WRIST tape commands rotation; |e_x| not a signed recovery (figure E) |
| nL/nR / bilat | MEDIUM | necessary for capture events, not a recenter measure |

## 10. Observable event signals

| event | from assumed sensing | robust? | ≠ true recenter |
|---|---|---|---|
| bilateral acquired/maintained | tactile or Fn>eps both fingers | moderate (threshold) | yes |
| contact lost | both Fn/touch ~0 | moderate | yes |
| stable force regime | Fn variance low + bilateral | weak without tuning | yes |
| slip detected | touch/Fn fluctuation + finger motion | weak; no slip estimator in code | yes |
| resecure | tau→−18 and Fn recover | descriptive | yes |
| excessive motion | hand-speed / qdot | yes (proprio) | yes |
| drop precursor | aperture open + Fn collapse + downward? | proprio+tactile, z of **object** is GT | partial |

## 11. Candidate summary table

| candidate | class | tracks progress? | sign-general? | temporal needed? | policy-info | reward-info | proxy risk | evidence |
|---|---|---|---|---|---|---|---|---|
| D | PRIVILEGED_GT mix | weak/mixed | see ± Pearson | maybe ΔD | mixed | no dense | HIGH | NONE |
| D_C / fn_imbalance | SENSOR | no signed recenter | must check ± | trend maybe | yes | no | HIGH | NONE |
| fn_sum | SENSOR | no | n/a | no | yes | no | HIGH | NONE |
| aperture | DIRECT | no (confounded by open/drop; **not sign-general**: Δ Pearson +e_x −0.382 vs −e_x +0.037) | no | slope=mode | yes | no | HIGH | WEAK |
| tau | DIRECT | no | n/a | no | yes | no | HIGH | NONE |
| track_pos_err | DIRECT | no | n/a | no | yes | no | HIGH | NONE |
| bilat / nL nR | SENSOR | event only | yes (binary) | persistence | yes | sparse maybe | MEDIUM | NONE |
| touch_* | SENSOR | unknown/weak | check ± | maybe | yes | no dense | HIGH | NONE |
| e_x_GT | PRIVILEGED_GT | yes (oracle) | yes | Δ | oracle | oracle only | N/A (not observable) | STRONG as label only |

**No STRONG reward-progress claim is made from correlation.**

## 12. Counterexamples

See `figures/E_proxy_counterexamples.png`, `E_D_disagreement_scatter.png`, and examples in §7.
- ZERO: no recovery command; |e_x| stays in the offset band while D and forces still vary.
- WRIST-only: large orientation command / tracking change without a signed |e_x| recovery.
- D↓ while |e_x|↑ and |e_x|↓ while D↑ are both present in step counts above.

## 13. Implications for Observable SAC

Allowed conclusions: **no single observable dense proxy is sufficient** for signed lateral recenter.

Future directions (not chosen here):
A. sparse observable event reward (bilateral persist / drop)
B. multi-signal observable reward
C. learned estimator from sensor history
D. vision/tactile estimate of relative object state
E. offline pretraining with GT + real-world observable fine-tuning

## 14. Remaining sensing requirements

To know actual recenter progress without simulator GT, the system needs an estimate of **object pose in the hand/gripper frame** (or equivalent contact geometry), e.g. vision, proximity, or spatially resolved tactile. Finger encoders + Cartesian residuals + scalar fingertip force do not currently yield a sign-general dense \(e_x\).

## Q1–Q7

**Q1.** No. No currently logged non-GT signal independently provides a reliable **signed** measure of true lateral recovery progress on both +e_x and −e_x.

**Q2.** No. D_t is mostly privileged object-relative GT plus a contact term, and ΔD disagrees with ΔP_GT in substantial step counts. It is not a suitable direct dense real-world reward on current evidence.

**Q3.** Fn magnitude, squeeze/tau, aperture, Cartesian tracking error, D_t, force imbalance — useful as **state**, not as quantities to maximize/minimize as progress.

**Q4.** Plausible **event** ingredients: bilateral contact acquired/maintained, contact-loss/drop, (if sensing allows) slip/resecure flags. Not a dense signed recenter term.

**Q5.** Yes for events (persistence windows). Not shown to create a dense signed progress proxy from current scalars.

**Q6.** Missing: object-relative lateral pose (or a calibrated tactile/vision estimator of e_x).

**Q7.** An Observable-SAC **dense** reward cannot be designed from current evidence without an additional sensing/estimation step. Sparse/hybrid event rewards could be sketched later; this audit does not design them.

## Evidence pointers

- Raw log: `D:\NUS\CS5478 Intelligent Robots\CS5478_Project\results\diagnostics\observable_reward_signal_audit\rows.csv` and `D:\NUS\CS5478 Intelligent Robots\CS5478_Project\results\diagnostics\observable_reward_signal_audit\rows.npz`
- Provenance: `D:\NUS\CS5478 Intelligent Robots\CS5478_Project\results\diagnostics\observable_reward_signal_audit\provenance.json`
- Numeric analysis: `D:\NUS\CS5478 Intelligent Robots\CS5478_Project\results\diagnostics\observable_reward_signal_audit\analysis.json`
- Code: `training/observable_signal_audit.py` (`collect`, `analyze`, `figures`)
- Sample count: n_rows=12489, n_ics=6 (both offset signs)
- Figures: `results/diagnostics/observable_reward_signal_audit/figures/`

