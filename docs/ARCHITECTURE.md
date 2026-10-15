# Layout

Commands run from the repository root. `data/`, `results/`, and `results/archive/demo_results/` stay on disk and out of Git.

| Path | Responsibility |
|---|---|
| `assets/` | MuJoCo scenes. `scene_ballistic_impact.xml` is the impact scene. |
| `config/` | YAML for the simulator, reward, and K48 training. |
| `controllers/` | Nominal controller, Jacobian tracking, gripper torque, `recovery_actions.py`, and the fixed 7D skill tapes in `skill_schedules.py`. |
| `envs/` | `step_legal`, contact extraction, grasp seating, ballistic `make_sim` / `world`, success and handoff checks, table-drop latch. |
| `sensors/` | Tactile array and the pose estimator. |
| `models/` | K48 Transformer, K48 checkpoint expand/load, the K=36 continual head, and action k-means. State-dict keys are unchanged. |
| `data_generation/` | IC construction, oracle search, K48 encoding, and the distinct sliding and recatch laws. |
| `training/` | `train_sft.py`, `train_grpo.py`, `dataset.py`, `trainer.py`, the K47 codebook loader, and the training runtime. |
| `evaluation/` | Closed-loop eval, handoff verification, and static data audit. |
| `visualization/` | Camera, captions, MP4 encoding, and tape replay. |
| `experiments/` | Searches, ablations, and historical trainers. They call formal modules. |
| `scripts/` | `generate_data.py`, `train.py`, `evaluate.py`, `render_video.py`. |
| `results/` | `sft/`, `grpo/`, `evaluations/`, `diagnostics/`, `videos/`, and `archive/demo_results/`. |
| `mujoco_menagerie/` | External robot descriptions. |

`envs/step_legal.py` is the only 20 ms recovery transition. Experiment scripts call that function. K48 weights load through `models/k48_checkpoint.py`. `training/unified_k48.py` is the command-line entry; the supervised loop is `training/train_sft.py` and the GRPO loop is `training/train_grpo.py`.

`controllers/`, `envs/`, `sensors/`, `models/`, `data_generation/`, `training/`, `evaluation/`, and `visualization/` do not import `experiments/`. Historical products are `results/archive/demo_results/`. The second-round dependency notes are in `docs/REFACTOR_AUDIT.md`.
