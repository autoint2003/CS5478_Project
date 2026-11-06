# Experiment Result Index

Canonical artifact roots under `results/experiments/`. Only entries with verified on-disk reports are listed. Panels are **not** interchangeable.

| ID | Title | Case / decision | Primary artifacts | Panel note |
|---|---|---|---|---|
| P64 | PPO multi-round Actor/Critic | Production checkpoints | `p64_ppo_multiround/checkpoints/P64_step4_{actor,critic}.pt` | Training era; not the P83 panel |
| P70 | Geometry-to-R3F / adaptive V2 | `do_not_train`; best=V2 | `p70_geometry_to_r3f/FINAL_REPORT.md`, `FULL_EPISODE_VALIDATION.md`, `P71_DECISION.json` | Held full episodes n=96/variant |
| P71 | Contact-safe alignment variants | Retained V2 reference | `p71_contact_safe_alignment/controllers.py` | Spec library; V3–V5 not frozen |
| P73 | First-contact bilateral overlay | Case C → `retain_v2` | `p73_first_contact_bilateral/P74_DECISION.json` | Held n=96; ΔSUCCESS=−0.09375 vs V2 |
| P75 | Critic calibration | Case C (no Actor PPO) | `p75_critic_calibration/P76_DECISION.json` | FC-window Critic diagnostics |
| P76 | Value observability audit | Case C | `p76_value_observability_audit/FINAL_REPORT.md` | Post-hoc affine; aliasing |
| P77 | Contact observability / sensing | Case A (obs study only) | `p77_contact_observability/FINAL_REPORT.md` | Sidewall AUROC study |
| P78 | Action-relevant representation | Case C | `p78_action_relevant_representation/FINAL_REPORT.md` | Matched CF ranking underpowered |
| P79 | Matched CF coverage | Case B | `p79_matched_cf_coverage/FINAL_REPORT.json` | 66 ICs; eng ≯ history |
| P80 | History-conditioned policy feasibility | Case B | `p80_policy_learning_feasibility/FINAL_REPORT.json` | Static tok_56 explains gains |
| P81 | Static contact-stage bias | Case B | `p81_static_contact_bias/FINAL_REPORT.json`, `FULL_EPISODE_TRANSFER.md` | Matched +; full-ep MSP −0.25 |
| P82 | Global failure budget | Freeze baseline; stop fixed-token | `p82_global_failure_budget/BASELINE_FREEZE.json`, evidence/decide | 16 ICs/class stage (reused into P83) |
| P83 | Cross-class recoverable failure mass | Case C → non-C diagnostic | `p83_recoverable_failure_mass/*` | **Canonical report panel: 32/class** |
| P84 | Non-C (B1) mechanism discovery | Case D → stop | `p84_nonC_mechanism_discovery/FINAL_REPORT.json` | Concentration gate fail |

## Frozen baseline pointer

`results/experiments/p82_global_failure_budget/BASELINE_FREEZE.json`

## Canonical evaluation panel pointer

`results/experiments/p83_recoverable_failure_mass/BASELINE_PANEL_VALIDATION.md`  
Episode cache: `results/experiments/p83_recoverable_failure_mass/cache/panel_32.pkl`

## Stop decision pointer

`results/experiments/p84_nonC_mechanism_discovery/FINAL_REPORT.json` → `p85: stop_consolidate_baseline`
