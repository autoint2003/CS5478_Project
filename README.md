# Adaptive Grasp Recovery (Mode-Switched Residual SAC)

CS5478 Intelligent Robots course project. A Franka Panda in MuJoCo follows a **nominal** Cartesian grasp-and-lift controller. When a ground-truth deterioration score \(D_t\) exceeds a threshold, Soft Actor-Critic adds a **bounded residual** to end-effector velocity and gripping force. Object mass \(m\) and friction \(\mu\) never enter the observation.

Research question: how can a recovery-specific SAC policy adapt its corrective actions to different grasp-deterioration states under uncertainty in mass, friction, and grasp placement?

Primary evidence is **recovery rate** among grasps that trigger recovery, compared with a tuned heuristic on the same detector and action bounds.

## Setup

Python 3.10+. From this directory:

```text
pip install -r requirements.txt
git clone https://github.com/google-deepmind/mujoco_menagerie.git
```

`mujoco_menagerie/` is **not** in this git repo. Only `franka_emika_panda` is required. Do not edit `mujoco_menagerie/franka_emika_panda/panda.xml`. Torque actuators are generated:

```text
python -c "from envs.xml_build import write_panda_torque; write_panda_torque()"
```

`assets/panda_torque.xml` is generated. `assets/scene.xml` includes it (floor, table, cylinder, lighting).

Physics: **500 Hz** (`physics_dt: 0.002`). SAC / recovery decisions run at **50 Hz** (`n_substeps: 10`) with residuals held and \(\tau_{\mathrm{nominal}}\) recomputed every physics step.

Optional viewers:

```text
python training/validate_grasp.py --viewer
```

## Checkpoints

Gated: finish CP\(n\) before CP\(n+1\). CP1–4 are the torque/Jacobian/contact foundation (historically run on a cube; CP5 re-validates the FSM on the cylinder).

| CP | Status | Command | Gate |
|---|---|---|---|
| 1 Torque hold | done | `python training/validate_hold.py` | Home pose \(\ge 5\,\mathrm{s}\), arm drift \(< 0.05\,\mathrm{rad}\) |
| 2 6D reaching | done | `python training/validate_reach.py` | Pos \(< 2\,\mathrm{cm}\), ori \(< 10^\circ\) |
| 3 Grasp FSM | done | `python training/validate_grasp.py` | Lift and hold \(1\,\mathrm{s}\), no RL |
| 4 Contact log | done | `python training/log_contact.py` | Touch \(\approx 0\) in free space; rises on pinch; stays during lift |
| 5 Cylinder lift | `python training/validate_grasp.py` | Lift/hold on the cylinder, centered grasp |
| 6 Detector | `python training/calibrate_detector.py` | \(D_t\) below enter on stable lifts; exceeds enter before drop |
| 7 Recovery buffer | `python training/collect_recovery_states.py` | Mild/moderate/severe bins; frozen eval split |
| 8 Recovery env | `python training/validate_recovery_env.py` | `check_env`; zero residual = nominal; heuristic rollout |
| 9 2D SAC | `python training/train_recovery.py --action 2d` | Pipeline stable; do **not** require beating the heuristic |
| 10 Eval | `python evaluation/evaluate_recovery.py` | Recovery rate vs nominal and heuristic on held-out in-range conditions |
| 11 Optional | `--action 7d` / `--no-contact-obs` | Cartesian residual and contact-obs ablation if 2D is insufficient |

Out of scope: perception, mass/friction estimation, regrasp after a full drop, cross-object geometry.

## Control

- **CP1.** \(\tau = b(q,\dot{q})\) from `data.qfrc_bias`.
- **CP2+.** \(\tau_{\mathrm{arm}} = J^\top\big(K_p e + K_d(v_{\mathrm{des}}-\dot{x})\big) + b(q,\dot{q})\). Approach/descend/close use \(v_{\mathrm{des}}=0\). Lift/recovery track a Cartesian velocity command.
- **CP3/5.** FSM `approach → descend → close → lift` plus contact-aware gripper squeeze.
- **CP6+.** Same ground-truth \(D_t\) for every method. Recovery when \(D_t > D_{\mathrm{enter}}\); return to nominal lift when \(D_t < D_{\mathrm{exit}}\) for \(N_{\mathrm{stable}}\) steps.
- **CP8+.** Stage-3 action \(a\in[-1,1]^2\) maps to \((\Delta v_z,\Delta F_g)\). Optional 7-D residual \((\Delta v,\Delta\omega,\Delta F_g)\). Never observe \(m\) or \(\mu\).

## Layout

```text
CS5478_Project/
├── assets/           scene.xml, generated panda_torque.xml
├── controllers/      bias, Cartesian PD + velocity, gripper, grasp FSM, heuristic
├── envs/             XML builder, ids, contact, deterioration, recovery Gym env
├── training/         validate_*, calibrate_detector, collect_recovery_states, train_recovery
├── evaluation/       evaluate_recovery
├── config/           sim.yaml, nominal.yaml, randomized.yaml, no_contact.yaml
└── results/          logs, figures, buffers, checkpoints
```

Clone [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) into `mujoco_menagerie/` beside these folders. Cursor metadata (`.cursor/`) is gitignored.
