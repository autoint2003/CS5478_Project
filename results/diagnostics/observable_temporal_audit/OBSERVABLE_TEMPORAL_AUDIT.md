# Observable temporal audit

No physics/action/obs/reward/terminal/SAC change. No detector. No MP4.
GT_* is evaluation-only. OBS_* is the policy-available namespace.

## 1–2. Current 27D observation (from `envs/observable_obs.py`, executed)

dim = 27. Policy rate: dt_policy = n_substeps×physics_dt = **20 ms**. Sim rate used for this audit: **2 ms**. `e_hat_dot` in the 27D vector uses **dt_policy**, not 2 ms.

| i | name | class | units/scale | instantaneous vs derived | hardware |
|---:|---|---|---|---|---|
| 0 | `e_hat_x` | **ESTIMATED** | m (normalized by SCALE_EX=7.5 mm) | instantaneous sample (clipped ±1) | not hardware; geometric map of SIM_PROXY CoP |
| 1 | `estimate_valid` | **ESTIMATED** | bool | mask | not hardware |
| 2 | `contact_present_L` | **SIM_PROXY** | bool | instantaneous sample (clipped ±1) | MuJoCo finger-object geom contact; not a Panda taxel array |
| 3 | `contact_present_R` | **SIM_PROXY** | bool | instantaneous sample (clipped ±1) | same |
| 4 | `valid_L` | **SIM_PROXY** | bool | instantaneous sample (clipped ±1) | CoP defined iff Fn sum >= 0.01 N |
| 5 | `valid_R` | **SIM_PROXY** | bool | instantaneous sample (clipped ±1) | same |
| 6 | `u_L` | **SIM_PROXY** | m (normalized by SCALE_EX) | instantaneous sample (clipped ±1) | MuJoCo contact.pos mapped to pad frame; SIMULATION PROXY |
| 7 | `v_L` | **SIM_PROXY** | m (normalized by SCALE_PAD_V=8.5 mm) | instantaneous sample (clipped ±1) | same |
| 8 | `u_R` | **SIM_PROXY** | m (normalized by SCALE_EX) | instantaneous sample (clipped ±1) | same; right-finger X = -hand X |
| 9 | `v_R` | **SIM_PROXY** | m (normalized by SCALE_PAD_V) | instantaneous sample (clipped ±1) | same |
| 10 | `fn_L` | **SIM_PROXY** | N (normalized by SCALE_F=20 N) | instantaneous sample (clipped ±1) | mj_contactForce wrench[0] abs sum |
| 11 | `fn_R` | **SIM_PROXY** | N (normalized by SCALE_F) | instantaneous sample (clipped ±1) | same |
| 12 | `aperture` | **ROBOT_STATE** | m (normalized by SCALE_AP=40 mm) | instantaneous sample (clipped ±1) | finger joint mean; Panda encoder-class |
| 13 | `ap_dot` | **ROBOT_STATE** | m/s (normalized by SCALE_APDOT=0.08) | instantaneous sample (clipped ±1) | finger qvel mean |
| 14 | `tau_from_secure` | **ROBOT_STATE** | 1; (ctrl[7]+18)/SCALE_TAU, SCALE_TAU uses TAU_OPEN=-1 | command | commanded ctrl[7], not a force sensor |
| 15 | `g_hx` | **ROBOT_STATE** | m/s^2 / 9.81 | instantaneous sample (clipped ±1) | FK orientation + known gravity |
| 16 | `g_hy` | **ROBOT_STATE** | m/s^2 / 9.81 | instantaneous sample (clipped ±1) | same |
| 17 | `g_hz` | **ROBOT_STATE** | m/s^2 / 9.81 | instantaneous sample (clipped ±1) | same |
| 18 | `v_hx` | **ROBOT_STATE** | m/s / 0.08 | instantaneous sample (clipped ±1) | hand body twist from MuJoCo (robot-state class) |
| 19 | `v_hy` | **ROBOT_STATE** | m/s / 0.08 | instantaneous sample (clipped ±1) | same |
| 20 | `v_z_world` | **ROBOT_STATE** | m/s / 0.08 | instantaneous sample (clipped ±1) | same |
| 21 | `w_hx` | **ROBOT_STATE** | rad/s / 2.0 | instantaneous sample (clipped ±1) | same |
| 22 | `w_hy` | **ROBOT_STATE** | rad/s / 2.0 | instantaneous sample (clipped ±1) | same |
| 23 | `w_hz` | **ROBOT_STATE** | rad/s / 2.0 | instantaneous sample (clipped ±1) | same |
| 24 | `hand_z_off_table` | **ROBOT_STATE** | m / 0.25 (hand_z - TABLE_TOP) | instantaneous sample (clipped ±1) | FK z minus scene TABLE_TOP=0.40 |
| 25 | `track_z` | **ROBOT_STATE** | m / 0.08 (p_des_z - hand_z) | instantaneous sample (clipped ±1) | controller p_des_z minus hand_z |
| 26 | `e_hat_dot` | **ESTIMATED** | m/s / 0.08; (Δe_hat)/dt_policy if consecutive valid else 0 | derived (causal Δ / dt_policy) | finite difference of e_hat at dt_policy=20 ms |

Code `OBS_SOURCES` labels several tactile fields SENSOR. This audit reclassifies them as **SIM_PROXY**: `measure_spatial_tactile` uses MuJoCo `contact.pos` and `mj_contactForce`. That is not Panda hardware tactile.

`SCALE_TAU` still uses `TAU_OPEN=-1.0` even though 4D `RECOVERY4D_TAU_OPEN=+2`. Therefore `ctrl[7]=+2` and `ctrl[7]=-1` both saturate `tau_from_secure` at **+1**. The 27D grip channel **cannot represent active opening vs old open-end**.

No dimension is object-pose GT. `contact_present_*` is still a simulator contact flag (binary), not nL/nR counts.

## 3. Trajectory dataset

| family | construction | notes |
|---|---|---|
| CENTER6_ZERO | frozen CENTER 6 m/s ballistic ZERO | table ~6.132 s |
| CENTER6_RECOVERY_070 | EARLY +120° τ=-5 0.70 s + brake + return + hold | stable edge-catch |
| CENTER6_RECOVERY_160 | same, 1.60 s weak slip | larger inward correction |
| CLEAN_RECAPTURE | frozen extra_after_both=0.04, τ_open=+2, v_z=0 | privileged close timing |

Control sequences were not modified. Missing OBS channels were collected by replaying those sequences.

## 4. GT labels (evaluation only; not in observation)

**Stability:** `GT_PROGRESSIVE` bilateral post-impact creep; `GT_RUNAWAY` |r_h.x|≥16 mm and |v_rel|>15 mm/s; `GT_STABLE_EDGE` / `GT_STABLE_MARGIN` bilateral, |v_rel|<0.02, |g_hx|<2.5, τ=-18, persisted 0.20 s.

**Recapture:** `GT_SECURE` / `GT_OPENING` / `GT_FREE` / `GT_RECONTACT` / `GT_CAPTURE_DYNAMIC` / `GT_CAPTURE_STABLE` aligned to privileged events OPEN_START, FIRST_BOTH_OFF, FIRST_RECONTACT, FIRST_BILATERAL, SECURE_CAPTURE, HOLD_START.

## 5. Progressive vs stable

Fn (~9 N) and aperture (~17–18 mm) overlap across ZERO creep, 0.70, and 1.60. They do **not** encode arrest.

CoP **position** on this dataset: ZERO progressive is already at the pad rim (`|u|/8.5 mm` median **0.99**), 0.70 sits at **0.88**, 1.60 at **0.73**. So |u| separates *how far the contact has walked*, not *whether it is still walking*. The 0.70 hold is the counterexample that **|u| large ≠ unstable**. A rule `|u| > 0.95 → fail` would only appear to work because ZERO has already saturated the pad; an arrested grasp at the same rim would be mislabeled.

Once CoP saturates at the rim, **Δu dies even if GT still creeps**. That is the main CoP-motion failure mode.

| feature | ZERO progressive | stable 0.70 | overlap (p10–p90) |
|---|---|---|---|
| |u|/8.5 mm | med=0.9944  [p10=0.9916, p90=0.9979]  n=1405 | med=0.879  [p10=0.8784, p90=0.88]  n=8413 | 0.0 |
| Δ\|u\| 100 ms mm | med=-0.001611  [p10=-0.006643, p90=0.001354]  n=1405 | med=0.0001172  [p10=4.589e-05, p90=0.0001876]  n=8413 | 0.11915540951746961 |
| Δ\|u\| 500 ms mm | med=-0.009253  [p10=-0.02086, p90=8.212]  n=1402 | med=0.0005871  [p10=-0.0002135, p90=0.0006575]  n=8413 | 0.10903544989367271 |
| OBS e_hat_dot (policy) | med=-8.917e-06  [p10=-4.737e-05, p90=3.388e-05]  n=1405 | med=1.174e-06  [p10=-8.713e-05, p90=8.945e-05]  n=8413 | 0.46074269179948146 |
| GT \|v_rel\| | med=0.0017  [p10=0.00095, p90=0.004222]  n=1405 | med=2.171e-05  [p10=2.546e-06, p90=0.001571]  n=8413 | 0.24102676443725654 |

## 6. CoP-u position vs motion

| W | ZERO prog Δ\|u\| mm | 0.70 edge | 1.60 margin |
|---:|---|---|---|
| 0.02 | med=-0.0001968  [p10=-0.00176, p90=0.0009052]  n=1405 | med=2.305e-05  [p10=-4.712e-05, p90=9.388e-05]  n=8413 | med=-1.784e-05  [p10=-0.001433, p90=0.001434]  n=8413 |
| 0.05 | med=-0.0006724  [p10=-0.003468, p90=0.001073]  n=1405 | med=5.864e-05  [p10=-1.242e-05, p90=0.000128]  n=8413 | med=-1.611e-05  [p10=-0.001459, p90=0.00144]  n=8413 |
| 0.1 | med=-0.001611  [p10=-0.006643, p90=0.001354]  n=1405 | med=0.0001172  [p10=4.589e-05, p90=0.0001876]  n=8413 | med=-2.767e-05  [p10=-0.001445, p90=0.001404]  n=8413 |
| 0.2 | med=-0.003477  [p10=-0.01159, p90=0.00671]  n=1402 | med=0.0002349  [p10=0.0001561, p90=0.0003051]  n=8413 | med=-5.094e-05  [p10=-0.001496, p90=0.0014]  n=8413 |
| 0.5 | med=-0.009253  [p10=-0.02086, p90=8.212]  n=1402 | med=0.0005871  [p10=-0.0002135, p90=0.0006575]  n=8413 | med=-0.0001161  [p10=-0.002798, p90=0.001312]  n=8413 |

Longer causal windows (200–500 ms) of |u| change are more useful than a single sample of |u|. Short 20 ms deltas are noisy relative to millimetre creep.

## 7. dehat audit

{
  "progressive_sign_agree_ehatdot_vs_GTvrelx": 0.5051020408163265,
  "frac_quiet_dehat_while_GTvrel_gt_15mm_s": 0.020640569395017794,
  "note": "e_hat_dot is CoP-migration / dt_policy, not object twist"
}

`OBS e_hat_dot` is **not** GT relative velocity. Sign agreement vs GT `v_rel.x` on progressive ZERO is **~0.51 (chance)**. Median |dehat| is ~10⁻⁵ m/s while GT |v_rel| is ~1.7 mm/s. dehat is quiet because CoP is pinned at the pad edge, not because the object is arrested. **Do not treat dehat≈0 as arrest.** Latency is one policy step (20 ms) when valid; validity gaps zero the channel.

## 8. Causal windows

All Δ features are x(t)−x(t−W) with W∈{20,50,100,200,500} ms. No centered/future windows.

## 9. Recapture phase observability

Events: {'OPEN_START': 4.113999999999769, 'FIRST_BOTH_OFF': 4.123999999999768, 'CLOSE_START': 4.161999999999764, 'FIRST_RECONTACT': 4.183999999999761, 'FIRST_BILATERAL': 4.183999999999761, 'SECURE_CAPTURE': 4.2359999999997555, 'HOLD_START': 4.2359999999997555}

Binary `OBS contact_present_L/R` tracks GT both-off on this construction (see `cp_equals_GTbothoff_FREE`). That is **not** nL/nR, but it **is** in the 27D vector. Privileged close timing is therefore *partially* replaceable by these flags, plus elapsed time since opening (not currently a dedicated obs dim — would need memory/stack).

- FREE: {'ap_mm': {'n': 30, 'median': 22.27913548791386, 'p10': 18.59197216623528, 'p90': 26.83652473296736, 'min': 16.95558541788661, 'max': 27.234243035290955}, 'apdot': {'n': 30, 'median': 0.168816556258871, 'p10': -0.7347715103381345, 'p90': 0.30766944029538124, 'min': -1.1379858905612408, 'max': 0.3414443859426445}, 'cpL': {'n': 30, 'median': 0.0, 'p10': 0.0, 'p90': 0.0, 'min': 0.0, 'max': 0.0}, 'fnL': {'n': 30, 'median': 0.0, 'p10': 0.0, 'p90': 0.0, 'min': 0.0, 'max': 0.0}, 'validL': {'n': 30, 'median': 0.0, 'p10': 0.0, 'p90': 0.0, 'min': 0.0, 'max': 0.0}, 'GT_vrel': {'n': 30, 'median': 0.3397534226751246, 'p10': 0.12362452294557631, 'p90': 0.5565898367135697, 'min': 0.06968483777423119, 'max': 0.610923441506664}}
- immediately before CLOSE (~12 ms): {'ap_mm': {'n': 7, 'median': 24.84155401100171, 'p10': 23.414019029276144, 'p90': 26.411369845424097, 'min': 23.075436766342648, 'max': 26.821103108555267}, 'apdot': {'n': 7, 'median': 0.30648394274674384, 'p10': 0.277158159737852, 'p90': 0.3345725011376694, 'min': 0.2696675706777578, 'max': 0.3414443859426445}, 'cpL': {'n': 7, 'median': 0.0, 'p10': 0.0, 'p90': 0.0, 'min': 0.0, 'max': 0.0}, 'cpR': {'n': 7, 'median': 0.0, 'p10': 0.0, 'p90': 0.0, 'min': 0.0, 'max': 0.0}, 'fnL': {'n': 7, 'median': 0.0, 'p10': 0.0, 'p90': 0.0, 'min': 0.0, 'max': 0.0}, 'validL': {'n': 7, 'median': 0.0, 'p10': 0.0, 'p90': 0.0, 'min': 0.0, 'max': 0.0}, 'GT_nL': {'n': 7, 'median': 0.0, 'p10': 0.0, 'p90': 0.0, 'min': 0.0, 'max': 0.0}, 'GT_vrel': {'n': 7, 'median': 0.3677492315096543, 'p10': 0.32296124984382063, 'p90': 0.4125697997231329, 'min': 0.31176822106339286, 'max': 0.42377911646376704}, 'elapsed_since_open_s': 0.047999999999994714}

`valid_L/R=0` means CoP undefined (no contact or Fn<0.01 N), **not** a measured CoP at the origin.

## 10. Close-timing counterfactual

| extra_after_both | Δt vs 40 ms | captured | ok | table | recontact | bilateral |
|---:|---:|---|---|---|---|---|
| 0.02 | -20 | True | False | None | 4.155999999999764 | 4.155999999999764 |
| 0.03 | -10 | True | True | None | 4.169999999999763 | 4.169999999999763 |
| 0.04 | 0 | True | True | None | 4.183999999999761 | 4.183999999999761 |
| 0.05 | 10 | True | True | None | 4.19799999999976 | 4.19799999999976 |
| 0.06 | 20 | False | False | 4.229999999999756 | 4.211999999999758 | 4.211999999999758 |

Physical recapture exists for extra 0.02–0.05 s (`captured=True`; 0.02 fails the old 20 ms both-off *gate* at 18 ms but still recaptures). extra=0.06 recontacts then **table**. Catch window is **about 30 ms wide**, not a long basin. Instantaneous tactile at CLOSE is invalid in every offset; the observable difference is **elapsed time since both-off** (and falling aperture/hand-z), which is not a 27D field.

## 11. Recontact vs stable capture

dynamic: {'fnL': {'n': 26, 'median': 10.702998408383557, 'p10': 9.70676902907189, 'p90': 21.416647707620783, 'min': 9.597622671221831, 'max': 26.767115383914547}, 'ap_mm': {'n': 26, 'median': 14.66902401949988, 'p10': 12.823794214501323, 'p90': 16.452954541602278, 'min': 12.676890696111185, 'max': 16.738480191040377}, 'GT_vrel': {'n': 26, 'median': 0.031948093005670246, 'p10': 0.005470090664448549, 'p90': 0.28312073954006484, 'min': 0.004788655518728669, 'max': 0.6296713831994086}, 'valid_both': 1.0}

stable: {'fnL': {'n': 6001, 'median': 9.026997416331511, 'p10': 9.019010216780414, 'p90': 9.050809229177053, 'min': 9.013020113000238, 'max': 11.193604167234854}, 'ap_mm': {'n': 6001, 'median': 17.6478511652753, 'p10': 17.64614643595037, 'p90': 17.67490134306274, 'min': 16.833982177366362, 'max': 17.686407575359574}, 'GT_vrel': {'n': 6001, 'median': 0.0001038811138477202, 'p10': 1.9491541044509457e-05, 'p90': 0.0002703473582935658, 'min': 1.2441251327774538e-06, 'max': 0.005405227589484839}, 'u_edge': {'n': 6001, 'median': 0.2055352356418226, 'p10': 0.11805426594794081, 'p90': 0.21099419259299923, 'min': 0.0010520452369874812, 'max': 0.21434096435348388}, 'du100': {'n': 5977, 'median': -0.004656064598429488, 'p10': -0.027469467147018763, 'p90': 0.002245686237628726, 'min': -1.2465566557014949, 'max': 0.44163994600608975}}

First tactile return is **not** recovered: Fn rebuilds, aperture returns toward ~17.6 mm, |v_rel| falls over tens of ms. An observable settling signature exists (validity both-on + low causal Δu + aperture/Fn plateau) but needs persistence, not a one-step classifier.

## 12. Sensor-imperfection (existing 0.5 mm quant/noise)

{
  "ideal_overlap_du100_A": 0.11915540951746961,
  "res_0.5mm_overlap_du100_A": 0.0,
  "sig_0.5mm_overlap_du100_A": 0.9985294199089295,
  "res_0.5mm_overlap_u_B": 0.0
}

Qualitative CoP-u **position** gap between 0.70 (0.88) and 1.60 (0.73) survives 0.5 mm quantization (separation remains). 0.5 mm **coordinate noise** makes 100 ms Δu overlap ~1: millimetre-scale CoP slope is **not** robust. Quantization to 0.5 mm bins also kills sub-millimetre Δu (both classes collapse toward 0; a 0.0 “overlap” score here is not better separability).

## 13. Separability summary

- A (ZERO progressive vs 0.70 arrest): |u| differs because ZERO is **rim-saturated**, not because |u| measures dynamics. dehat **fails** (overlap 0.46, chance sign). Causal Δu is tiny and saturates. GT |v_rel| still overlapping at the 1 mm/s scale. **Arrested-near-edge vs creeping-near-edge is not cleanly observable from current CoP/dehat.**
- B (0.70 vs 1.60): **|CoP u|** encodes geometric margin (0.88 vs 0.73, overlap 0). Motion features similar once both are arrested.
- C (FREE vs pre-close): both have invalid tactile, open aperture; not a one-frame class. Need time-since-both-off (memory).
- D (recontact vs stable): Fn/aperture settle over tens of ms; first contact ≠ recovered.

## 14. Likely need for temporal state

**Answer: B, with E on the hard stability pair.** Instantaneous 27D (A) cannot (i) tell arrested 0.70 from “looks offset”, (ii) time recapture CLOSE, or (iii) replace GT v_rel via dehat. Causal stacking (B) is the minimum next experiment: persist `contact_present`, causal Δu/Δaperture/ΔFn, and time-since-both-off. A recurrent net (C) is not justified before that. An explicit object-velocity estimator (D) is the missing *physical* quantity for creep-vs-arrest, but it must be observable, not GT. If stacked tactile still saturates at the pad rim, evidence is **insufficient (E)** and a new observable (not GT twist pasted into obs) is required.

## 15. Future stability criterion (hypothesis only; not implemented)

Against these four families, a later terminal would need **persistence**, not |ehat| small and not |CoP u| small:

- bilateral `contact_present` / `valid_*` held
- low causal CoP motion over ≥100–500 ms
- aperture and Fn settled
- **not** rejecting large |u| by itself (0.70 counterexample)
- **not** trusting dehat≈0 alone (ZERO aliasing)

No thresholds tuned here.

## 16. Unresolved gaps

- 27D `tau_from_secure` saturates for active open.
- No explicit time-since-contact-loss channel.
- Spatial tactile remains a MuJoCo proxy.
- Binary contact_present ≠ contact counts; flicker vs true both-off still needs persistence.
- GT relative velocity is **not** an allowed OBS add-on from this audit.

## Viewers

```text
python training/demo_observable_audit.py --case center6_zero
python training/demo_observable_audit.py --case stable_edge
python training/demo_observable_audit.py --case recapture
```

Existing: `python training/demo_ballistic_recovery.py --mode zero` · `python training/demo_airborne_recapture.py --mode recapture`

STOP.