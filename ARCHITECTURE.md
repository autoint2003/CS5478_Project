# Architecture (post Plan A/B/C consolidation)

## Dependency map (minimal active path)

| Capability | Modules |
|---|---|
| MuJoCo simulation | `envs/simulation/grasp_sim.py`, `envs/simulation/ground_sim.py`, `assets/` |
| Legacy elevated table | `envs/simulation/grasp_sim.py` + ballistic scene XML |
| Legal 7D actions / 20 ms | `controllers/recovery/action_map.py`, `envs/step_legal.py` |
| Nominal controller | `controllers/nominal.py` |
| 39D observation | `envs/observations/airborne_obs.py` (`OBS_DIM_AIR=39`) |
| SUCCESS_NEW / SUCCESS_12S | `evaluation/full_objective.py`, `envs/outcomes/recovery_success.py` |
| GROUND_DROP | `envs/outcomes/ground_drop.py` |
| Offline expert search (Plan C) | `data_generation/strategies/ground_*.py`, `scripts/run_ground_*.py` |
| Expert replay | `scripts/replay_golden_experts.py`, `training/recovery_execution.py` |
| Unified dataset | `scripts/build_unified_dataset.py` → `datasets/unified_experts/` |
| Frozen K49 | `training/action_vocab_k49.py` (IDs 0–48; HANDOFF=47; tok48 reverse wrist) |
| Transformer policy | `models/airborne_v3.py`, `training/train_sft.py`, `training/train_grpo.py` |
| Evaluation / viz | `evaluation/`, `visualization/` |

## Plan families

- **Plan A** — continuous-contact sliding / pivoting recovery (legacy elevated + mechanism audits).
- **Plan B** — legal release and recatch (legacy elevated).
- **Plan C** — GROUND_SCENE ball impact → full detach → active recatch → brake → slow lift → legal HANDOFF → SUCCESS_NEW.

Runtime inference uses **one** Transformer policy. Plan labels are offline metadata only. Offline search may use privileged future simulation; runtime observations must remain 39D without privileged state.

## Active entry points

```text
python -u scripts/build_unified_dataset.py
python -u scripts/replay_golden_experts.py
python -u scripts/audit_k49.py
python -u scripts/run_ground_plan_c_final_success.py   # regenerate Plan C search (expensive)
python -u scripts/train.py                             # SFT/GRPO — do not launch until ready
python -u scripts/evaluate.py
```

Historical one-off probes live under `experiments/` and `archive/experiments/`.
