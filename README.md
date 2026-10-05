# CS5478 — Model-Free Adaptive Control for Contact-Rich Robotic Grasping under Dynamics Uncertainty

MuJoCo simulation of a **Franka Panda** with a parallel gripper and a **cylinder**. A nominal Cartesian grasp-and-lift controller runs at 500 Hz. After a disturbance, recovery is a **mode switch** (not residual torque mixed into lift).

This repository currently demonstrates **recovery authority** on a matched ballistic-impact benchmark. A unified **observable** recovery policy has **not** been trained yet.

---

## Current task

1. Nominal grasp and lift of the cylinder.  
2. A genuine **ballistic ball impact** during lift (CENTER-6 family).  
3. Object deterioration (relative offset / loss of a secure centered pinch).  
4. Compare three responses from the **same** pre-recovery state:

| Branch | Meaning |
|---|---|
| **ZERO** | No recovery primitive; continue nominal lift/hold |
| **CONTROLLED_SLIP** | Privileged gravity-aligned weak-grip slip + brake + hold |
| **AIRBORNE_RECAPTURE** | Privileged open + free flight + recatch + hold |

---

## Recovery action space (current formulation)

The learned/interface action is **4D**, \(a\in[-1,1]^4\):

\[
a = (\omega_y,\; v_x,\; v_z,\; a_g)
\]

mapped by `controllers/residual.py` `map_recovery4d`:

| Channel | Physical command | **Main bound** |
|---|---|---|
| \(a_0\) | wrist rate \(\omega_y\) about `r_des` body-y | **`RECOVERY4D_W_HY_MAX = 3.0` rad/s** |
| \(a_1\) | hand-frame \(v_x\) | `RECOVERY4D_V_HX_MAX = 0.08` m/s |
| \(a_2\) | world \(v_z\) | `RECOVERY4D_V_Z_MAX = 0.08` m/s |
| \(a_3\) | grip tendon \(\tau\) | \(a_g=-1\) → `RECOVERY4D_TAU_OPEN = +2`; \(a_g=+1\) → `RECOVERY4D_TAU_SECURE = -18` |

World twist: \(v = R_\mathrm{hand}[v_x,0,0] + [0,0,v_z]\), \(\omega = R_\mathrm{des}[0,\omega_y,0]\).

**Diagnostic \(\omega_y=4\) rad/s** is used only in a simulation-authority construction for airborne recapture (`tick_omega_override`). It is **not** the training bound and is **not** a hardware-safe Panda velocity.

Do not treat the older 2D recovery action as current.

---

## Observation architecture

**Observable / policy inputs** (no object mass, no friction, no privileged \(D_t\) as an authority signal):

- spatial tactile (per-pad contact / force features)  
- gripper aperture and commanded tendon \(\tau\)  
- hand-centric relative pose estimates used in `observe_recovery4d` / 27D `envs/observable_obs.py`  
- binary contact-present flags in the 27D vector  

**Privileged simulator quantities** (logging, constructions, CLOSE timing, hashes — **not** claimed as policy inputs):

- true object pose/velocity, ball state  
- geom contact counts `nL`/`nR`, `mj_contactForce`  
- `FIRST_BOTH_OFF` from GT contacts  
- scheduled OPEN/CLOSE times  

---

## Contact-model correction (`noslip=1`)

With `noslip_iterations = 0`, a **secure** centered pinch showed gravity-loaded tangential **constraint creep** (cylinder ~1.22 mm/s). That is not physical grasp deterioration.

Canonical freeze: **`noslip_iterations = 1`**, `noslip_tolerance = 1e-6` (`config/sim.yaml`, `envs/xml_build.py`, `apply_solver_from_cfg`). Smallest tested setting that removes that creep while keeping capacity-limited slip and true release.

Calibration: [`results/diagnostics/noslip_calibration/NOSLIP_CALIBRATION.md`](results/diagnostics/noslip_calibration/NOSLIP_CALIBRATION.md).

---

## Recovery authority established so far

These are **constructions + raw logs + (where noted) visual checks**. They are not a trained SAC policy.

| Mode | What was shown | Status |
|---|---|---|
| Contact-preserving / secure manipulation | Wrist rotation at \(\tau=-18\) while bilateral contact holds | raw-log verified; legal \(\lvert\omega_y\rvert\le 3\) |
| Controlled gravitational slip + braking | Secure \(+120^\circ\) → \(\tau=-5\) (0.70 s) → \(\tau=-18\) brake → return/hold on CENTER-6 EARLY | raw-log verified; **privileged** timing/angle; visually used in the comparison video |
| Temporary full release + airborne recapture | Diagnostic \(\omega_y=+4\), OPEN ~70°, genuine `FIRST_BOTH_OFF`, privileged CLOSE, \(\tau=-18\) hold | raw-log verified; **user-visually-verified** candidate; **privileged + diagnostic ω** |

**Not yet learned.** **Not hardware validated.**

---

## Evidence status (how to read claims)

| Tag | Meaning |
|---|---|
| raw-log verified | Trajectory fields (`qpos`, contacts, relative pose/velocity, \(\tau\)) support the claim |
| visually verified | A human watched the MuJoCo viewer / video |
| privileged construction | Trigger, OPEN/CLOSE time, or \(\omega=4\) used simulator/scripted knowledge |
| not yet learned | No SAC/policy has reproduced the behavior from observations |
| not hardware validated | Simulation only |

Cursor chat summaries are **not** evidence.

---

## Repository structure (important paths)

```text
config/           nominal.yaml, sim.yaml (noslip=1), randomized.yaml, rule_based_recovery.yaml
assets/           scene.xml, scene_ballistic_impact.xml, panda_torque.xml
controllers/      nominal.py, jacobian_controller.py, residual.py (recovery4d bounds)
envs/             grasp_sim.py, recovery_env.py, physical_recovery.py, observable_obs.py
sensors/          spatial_tactile.py
training/
  demo_ballistic_impact.py
  map_ballistic_disturbance.py
  demo_ballistic_recovery.py          # ZERO continue_zero, EARLY snaps
  ballistic_large_angle.py            # +120° gravity transfer
  ballistic_slip_sufficiency.py       # τ=-5 duration
  demo_dynamic_recatch.py             # throw/recatch viewers
  omega4_throw_test.py                # diagnostic ω=4 override
  release_state_map_omega4.py         # EARLY × 70° recatch
  demo_ballistic_recovery_comparison.py   # ZERO | SLIP | AIRBORNE
  train_recovery.py / eval_heldout.py     # future SAC (do not run yet)
results/
  diagnostics/    contact, slip, ballistic, airborne, obs audits
  comparison/ballistic_recovery/      final comparison report + raw npz
  videos/         ballistic_*.mp4
CLEANUP_MANIFEST.md
```

Inventory of KEEP / UNCERTAIN / REMOVE: [`CLEANUP_MANIFEST.md`](CLEANUP_MANIFEST.md).

---

## Setup

Python 3.10+. From this directory:

```text
pip install -r requirements.txt
git clone https://github.com/google-deepmind/mujoco_menagerie.git
python -c "from envs.xml_build import write_panda_torque; write_panda_torque()"
```

`mujoco_menagerie/` is not in git. Physics **500 Hz** (`dt=0.002`). Policy rate 50 Hz (`n_substeps: 10`) for future SAC.

---

## Reproduction

**Nominal stack**

```text
python training/validate_hold.py
python training/validate_reach.py
python training/validate_grasp.py
```

**Ballistic impact (ZERO family, no recovery)**

```text
python training/demo_ballistic_impact.py --mode representative
python training/map_ballistic_disturbance.py
```

**Final recovery comparison** (same CENTER-6 prefix; videos optional)

```text
python training/demo_ballistic_recovery_comparison.py --headless --write-videos
```

**Interactive (same camera init; mouse preserved)**

```text
python training/demo_ballistic_recovery_comparison.py --mode zero
python training/demo_ballistic_recovery_comparison.py --mode controlled_slip
python training/demo_ballistic_recovery_comparison.py --mode airborne_recatch
python training/demo_ballistic_recovery_comparison.py --mode all
```

Related viewers: `python training/demo_dynamic_recatch.py --help`.

**Future training / evaluation (do not start SAC in this freeze)**

```text
python training/test_recovery4d_interface.py
python training/validate_recovery_env.py
# python training/train_recovery.py --action recovery4d
# python training/eval_heldout.py --which offset
```

Held-out offset set: `results/eval_sets/airborne_offset_eval.npz`.  
`results/eval_sets/impact_severity_four.npz` is **quarantined** (old ball-mass semantics). Do not train on it.

---

## ZERO vs stale STOP

On CENTER-6, ZERO is `continue_zero`: keep the **nominal** lift until `p_des.z` reaches the lift target, then hold with `v_cmd=0`, `w_cmd=0`, `tau=-18`. It does **not** freeze a stale pre-impact `p_des`.

---

## Current limitations / next step

Recovery **authority** is demonstrated (privileged constructions). The unified **observable** 4D policy has **not** been trained or validated. Next step is SAC on `RecoveryEnv` with the **3 rad/s** \(\omega_y\) bound — not further physics retuning.

Comparison report: [`results/comparison/ballistic_recovery/BALLISTIC_RECOVERY_COMPARISON.md`](results/comparison/ballistic_recovery/BALLISTIC_RECOVERY_COMPARISON.md).
