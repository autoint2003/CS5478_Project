# Layout

Commands run from the repository root. `data/`, `results/`, and `results/archive/demo_results/` stay on disk and out of Git.

| Path | Responsibility |
|---|---|
| `assets/` | MuJoCo scenes. `scene_ballistic_impact.xml` is the impact scene. |
| `config/simulation/` | Physics and nominal grasp YAML. |
| `config/controllers/` | Geometry switch and the frozen rule controller. |
| `config/recovery/` | Success-label thresholds. |
| `config/training/k48/` | K48 SFT/GRPO configs. The live recovery set is `unified_k48_recovery_v3.yaml` and `unified_k48_recovery_v3_reward_v2.yaml`. |
| `config/training/k47/` | Historical K47 configs. |
| `config/training/legacy/` | K36 GRPO pilots and the codebook record. |
| `controllers/nominal.py` | Nominal grasp FSM. |
| `controllers/cartesian/` | Jacobian tracking, `qfrc_bias` torque, and the `TRACK` gains. |
| `controllers/gripper/` | Gripper torque. |
| `controllers/recovery/` | 7D action map, residual limits, geometry switch, rule baseline, skill tapes. |
| `envs/step_legal.py` | The only 20 ms legal recovery step. |
| `envs/ballistic_impact.py` | Ballistic scene and `make_sim`. |
| `envs/simulation/` | Grasp simulator, MuJoCo ids, object dynamics, XML, capture, scene state, RULE replay. |
| `envs/observations/` | 27D observable obs, 39D airborne obs, tactile reward, privileged pack. |
| `envs/contacts/` | Summed touch (`touch.py`), per-contact frames (`frames.py`), and scene contact rows. |
| `envs/outcomes/` | `SUCCESS_12S`, handoff, table-drop latch, pose error, physical pack. |
| `envs/initialization/grasp_seat.py` | Seated grasp initial conditions. |
| `sensors/` | Spatial tactile array and the hand-x offset `e_x` in `lateral_offset.py`. |
| `models/k48_policy.py` | K48 network, including the residual head. |
| `models/k48_checkpoint.py` | K48 load and K47 expansion. State-dict keys are unchanged. |
| `models/airborne_v3.py` | 39D airborne policy and `step_v3`. Demo generation is `experiments/legacy/airborne_v3_demos.py`. |
| `models/historical/` | K33 and K36 checkpoints and networks. |
| `data_generation/` | Official generators, strategy laws, oracles, recording, validation, and maintenance. |
| `training/` | SFT, GRPO, batches, `grpo_progress.py`, `k47_codebook.py`, the selected `reward_v2`, and the closed-loop episode runtime. |
| `evaluation/` | K48 closed-loop eval and reusable data audits. |
| `visualization/` | Camera, captions, MP4 encoding, tape replay, policy-vs-nominal videos. |
| `evaluation/` | Closed-loop eval, legality audits, validation skill/mechanism classification. |
| `experiments/` | Searches, skill diagnostics, hard-IC mining, and historical trainers. They call formal modules. |
| `scripts/` | `generate_data.py`, `train.py`, `evaluate.py`, `render_video.py`. |
| `results/` | `sft/`, `grpo/`, `evaluations/`, `diagnostics/`, `videos/`, and `archive/demo_results/`. |
| `mujoco_menagerie/` | External robot descriptions. |

`envs/step_legal.py` is the only 20 ms recovery transition. Experiment scripts call that function. K48 weights load through `models/k48_checkpoint.py`. `training/unified_k48.py` is the command-line entry; the supervised loop is `training/train_sft.py` and the GRPO loop is `training/train_grpo.py`.

`controllers/`, `envs/`, `sensors/`, `models/`, `data_generation/`, `training/`, `evaluation/`, and `visualization/` do not import `experiments/`. Historical products are `results/archive/demo_results/`. The second-round dependency notes are in `docs/REFACTOR_AUDIT.md`. Subpackage roles and the round-3 path map are in `docs/MODULE_ORGANIZATION.md`.
