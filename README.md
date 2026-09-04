# Residual Torque SAC for Contact-Rich Grasping

CS5478 Intelligent Robots course project. A Franka Panda in MuJoCo follows a **nominal** torque controller for reach-and-grasp. Soft Actor-Critic later adds a **bounded residual** $\Delta\tau$. Object mass $m$ and friction $\mu$ never enter the observation.

$$

\tau_{\mathrm{applied}}
=
\mathrm{sat}\big(
\tau_{\mathrm{nominal}}(s_t)
+
\Delta\tau_{\mathrm{requested}}
\big)

$$

**Question.** Can a model-free residual policy use interaction feedback to adapt grasping to unseen mass and friction?

Primary evidence is grasp success and degradation under dynamics shift. $\|\Delta\tau\|$ is supporting interpretability only.

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

`assets/panda_torque.xml` is generated. `assets/scene.xml` includes it (floor, pedestal, cube, lighting).

Physics: **500 Hz** (`physics_dt: 0.002`). From CP5, SAC runs at **50 Hz** (`n_substeps: 10`) with $\Delta\tau$ held and $\tau_{\mathrm{nominal}}$ recomputed every physics step.

Optional viewers need `import mujoco.viewer` (already in the validate scripts):

```text
python training/validate_grasp.py --viewer
```

## Checkpoints

Gated: finish CP$n$ before CP$n+1$.

| CP | Status | Command | Gate |
|---|---|---|---|
| 1 Torque hold | done | `python training/validate_hold.py` | Home pose $\ge 5\,\mathrm{s}$, arm drift $< 0.05\,\mathrm{rad}$ |
| 2 6D reaching | done | `python training/validate_reach.py` | Pos $< 2\,\mathrm{cm}$, ori $< 10^\circ$ |
| 3 Grasp FSM | done | `python training/validate_grasp.py` | Lift and hold $1\,\mathrm{s}$, no RL |
| 4 Contact log | done | `python training/log_contact.py` | Touch $\approx 0$ in free space; rises on pinch; stays during lift |
| 5 Residual Gym env | not started | — | Zero residual reproduces CP3; `check_env`; seeded reset |
| 6 Residual SAC (fixed $m,\mu$) | not started | `python training/train_residual.py --config config/nominal.yaml` | Pipeline stable; do **not** require beating nominal |
| 7 Hidden-dynamics rand | not started | `python evaluation/generate_eval_episodes.py` then train with `config/randomized.yaml` | Frozen nominal / ID / OOD episodes (OOD is $m,\mu$ only) |
| 8 Eval and ablations | not started | `python evaluation/evaluate_*.py` | Success and degradation first; retrained no-contact |

Contact CSV/plot after CP4: `results/logs/contact_grasp.csv`, `results/figures/contact_grasp.png`.

## Control

- **CP1.** $\tau = b(q,\dot{q})$ from `data.qfrc_bias` (static hold $\approx g(q)$).
- **CP2+.** $\tau_{\mathrm{arm}} = J^\top(K_p e - K_d \dot{x}) + b(q,\dot{q})$.
- **CP3.** FSM `approach → descend → close → lift` plus contact-aware gripper squeeze (no slip estimator).
- **CP5+.** SAC action $a\in[-1,1]^8$ maps to bounded $\Delta\tau$. Observation: $q,\dot{q}$, EE, object, contact, FSM one-hot. Never $m$ or $\mu$.

Gains and FSM dwells: `config/nominal.yaml`. Rates and cube/table: `config/sim.yaml`.

## Layout

```text
CS5478_Project/
├── assets/           scene.xml, generated panda_torque.xml
├── controllers/      bias, Cartesian PD, gripper, grasp FSM
├── envs/             XML builder, ids, contact, (Gym env from CP5)
├── training/         validate_*.py, log_contact.py, (train_residual from CP6)
├── evaluation/       (from CP7/CP8)
├── config/           sim.yaml, nominal.yaml, randomized.yaml, no_contact.yaml
└── results/          logs, figures, checkpoints, frozen eval episodes
```

Clone [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) into `mujoco_menagerie/` beside these folders. Cursor metadata (`.cursor/`) is gitignored.
