# Unified Expert Dataset Schema (`unified_experts_v1`)

## Locations

| Path | Tracked? | Role |
|---|---|---|
| `data/unified_experts/index.json` | no | Expert census |
| `data/unified_experts/splits.json` | no | Root-level splits |
| `data/unified_experts/manifest.jsonl` | no | One metadata row per expert |
| `data/unified_experts/records/*.pkl` | no | Plan C action/obs tensors + snaps |
| `data/supervised_targets_with_horizontal.jsonl` | no* | Legacy Plan A/B source (*under `data/`) |
| `data_generation/datasets/` | yes | Dataset construction **code** (not manifests) |

## Record fields

| Field | Required | Notes |
|---|---|---|
| `schema_version` | yes | `unified_experts_v1` |
| `expert_id` | yes | Unique, e.g. `plan_c__gbi_010` |
| `plan_family` | yes | `A`, `B`, or `C` (metadata only) |
| `physical_root_id` | yes | Split key; independent IC / impact root |
| `scene` | yes | `GROUND_SCENE` or `legacy_elevated` |
| `environment_version` | yes | |
| `initial_physical_state_ref` | yes | Snapshot / pkl pointer |
| `actions_7d` | Plan C | `(T, 7)` legal continuous actions |
| `observations_39d` | Plan C | `(T, 39)` runtime obs aligned pre-action |
| `tokens_k49_ref` | Plan A/B | Path to npz token tape |
| `action_timestamps_s` | Plan C | 20 ms grid |
| `control_period_s` | yes | `0.02` |
| `handoff` | yes | Token 47 semantics |
| `SUCCESS_12S` / `SUCCESS_NEW` | yes | |
| `exact_replay` | yes | Pass counts when available |
| `contact_grasp_mechanism` | yes | |
| `source_experiment` | yes | Provenance |
| `data_integrity_checksum` | yes | SHA-256 |
| `split` | yes | `train` / `val` / `test` / `regression_fixture` |

## Splits

- Split by **physical root**, never by random timesteps within a trajectory.
- Plan C roots `gbi_010` and `gbi_022` are **regression fixtures only** (too few for generalization holdout).
- Privileged oracle diagnostics are **not** model inputs.

## Timing convention

Observation at step `t` is recorded **before** applying action `t`. Control period is 20 ms. HANDOFF is terminal and does not decode a motion residual.
