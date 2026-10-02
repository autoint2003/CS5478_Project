# CS5478 Grasp Recovery

Franka Panda in MuJoCo. A **nominal** Cartesian grasp-and-lift controller runs at 500 Hz. Recovery is a **mode switch**: freeze-hold plus a separate recovery policy, not residual torque mixed into the lift controller.

The scientific comparison is **ZERO vs frozen RULE vs SAC** on the same held-out initial conditions. Object mass and friction are never observed.

The recovery problem of record is **captured, airborne, hand-frame \(x\) grasp offset**. Impact ICs are a transfer check only.

## Setup

Python 3.10+. From this directory:

```text
pip install -r requirements.txt
git clone https://github.com/google-deepmind/mujoco_menagerie.git
```

`mujoco_menagerie/` is not in git. Only `franka_emika_panda` is required. Do not edit the Menagerie XML. Generate torque actuators with:

```text
python -c "from envs.xml_build import write_panda_torque; write_panda_torque()"
```

Physics: **500 Hz** (`physics_dt: 0.002`). Policy / recovery decisions: **50 Hz** (`n_substeps: 10`).

## Frozen RULE

Do not retune these files:

- `config/rule_based_recovery.yaml`
- `controllers/rule_based_recovery.py`

FSM: `STABILIZE → SLIP_ALIGN → CONTROLLED_SLIP → BRAKE → CHECK → [OPEN_REGRASP / RECLOSE] → SECURE → RESUME`.

Grip tendon commands of record: secure \(\tau=-18\), controlled slip \(\tau=-2\), open \(\tau=-1\). Offset success is \(|e_x|\le 3\,\mathrm{mm}\), bilateral contact, low relative motion, scene-free.

Replay helpers live in `training/replay_core.py`. Hashes are checked by `training/eval_heldout.py` and `training/test_mainline_replay.py`.

## Held-out evaluation (do not train on these)

| Set | Path | Use |
|---|---|---|
| Offset N=80 | `results/eval_sets/airborne_offset_eval.npz` | Primary ZERO vs RULE vs (later) SAC |
| Impact four speeds | `results/eval_sets/impact_severity_four.npz` | Impact transfer only |

```text
python training/eval_heldout.py --which both
python training/test_mainline_replay.py
```

## Nominal stack

| Piece | Path |
|---|---|
| Grasp FSM | `controllers/nominal.py` |
| Cartesian PD | `controllers/jacobian_controller.py` |
| Gripper | `controllers/gripper_controller.py` |
| Sim | `envs/grasp_sim.py` |
| Scene | `assets/scene.xml` |
| Config | `config/nominal.yaml`, `config/randomized.yaml` |

Smoke:

```text
python training/validate_hold.py
python training/validate_reach.py
python training/validate_grasp.py
```

## SAC pipeline (not redesigned yet)

Entry: `python training/train_recovery.py --action 2d`.

Current training ICs still come from `training/collect_recovery_states.py` (nominal lifts gated by \(D_t > D_{\mathrm{enter}}\)). `RecoveryEnv` still uses a \(D_t\) potential reward and \(D_{\mathrm{exit}}\) success. That is **not** the frozen RULE trigger and is **not** the intended final training distribution.

SAC must not read the held-out npz files above. Intended redesign: \(D_t\) may shape **reward only**; recovery ICs must be constructed physical recovery-required states.

Optional CP checks: `training/calibrate_detector.py`, `training/validate_recovery_env.py`, `evaluation/evaluate_recovery.py`.

## Layout

```text
assets/          scene.xml, generated panda_torque.xml, impact diagnostic scene
config/          sim + grasp YAML; frozen rule_based_recovery.yaml
controllers/     nominal FSM, Jacobian PD, gripper, residual limiter, frozen RULE
envs/            GraspSim, RecoveryEnv, deterioration score
training/        validate_*, collect, train_recovery, replay_core, eval_heldout
evaluation/      SAC vs heuristic vs nominal on the D_enter collect split
results/eval_sets/   frozen offset + impact ICs
```

Course PDFs in the repo root are the proposal/preproposal slides, not runtime code.

## Out of scope

Perception, mass/friction estimation, regrasp after a full drop, retuning physics or Cartesian gains to chase recovery metrics.
