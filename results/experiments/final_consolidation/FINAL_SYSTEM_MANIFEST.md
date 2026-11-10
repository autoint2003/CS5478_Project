# Final System Manifest — CS5478 Adaptive Grasp Recovery

**Status:** Frozen after P83 Case D post-audit revision + P84 Case D (`stop_consolidate_baseline`). No P85. Model behavior unchanged by this consolidation. See `POST_AUDIT_DECISION_REVISION.md`.

## Frozen stack

| Component | Identity | Path / definition |
|---|---|---|
| Actor | P64 step-4 | `results/experiments/p64_ppo_multiround/checkpoints/P64_step4_actor.pt` |
| Critic | P64 step-4 | `results/experiments/p64_ppo_multiround/checkpoints/P64_step4_critic.pt` |
| Pre-contact controller | P70 V2 (`V2_p70_ref`) | Spec in `experiments/precision/p71_contact_safe_alignment/controllers.py` → `make_specs()["V2"]` |
| Action vocabulary | Original M111 | Loaded from Actor checkpoint (`book_n=111`) |
| Environment | `StabilizeGrpoEnv` | `experiments/precision/p57_post_recatch_stabilization/env_stabilize.py` |
| Reward (training-era) | P57-A PoseRewardWeights, `use_stabilization=False` | Documented in P82 `BASELINE_FREEZE.json` (inference does not retrain) |
| R3F | Original `evaluate_predicates` | `experiments/precision/p67_r3f_gate_diagnosis/gate_predicates.py` |
| Terminal success | `SUCCESS_RELAXED_12S` | `experiments/precision/p53_independent_grpo64/success_relaxed.py` |

## Checkpoint hashes (verified 2026-11-06)

| File | MD5 | Bytes | Load check |
|---|---|---:|---|
| `P64_step4_actor.pt` | `9f4c50413535e4c7db3d3a868bc1b6e2` | 370227 | OK (CPU); Actor params 87927; book 111 |
| `P64_step4_critic.pt` | `95de94e5a5d31ba8acddbfb49a6981cb` | 332359 | OK (CPU); Critic params 80257 |

Hashes match `results/experiments/p82_global_failure_budget/BASELINE_FREEZE.json` and P83 `BASELINE_PANEL_VALIDATION.md` (`freeze_hashes_match_p82: true`).

## V2 controller configuration (exact)

From P82 freeze / `make_specs()["V2"]`:

- `name`: `V2_p70_ref`
- `kind`: `v2`
- `max_activate_step`: 7
- `require_no_contact`: true
- `max_approach`: 6
- `min_approach`: 2
- `grip_steps`: 1
- `post_approach`: 2
- `ready_score_thresh`: −0.05
- `approach_pool`: `[56, 50, 71]`
- Ready score uses deployable proprioceptive proxies (`hand_speed` / aperture / `track_z`) — **not** privileged `e_pos`.

## Canonical inference entry point

**Primary (evaluation-grade, matches P82/P83 baseline panel):**

`experiments.precision.p82_global_failure_budget.collect.rollout_episode`

Call pattern (frozen path):

1. Load Actor/Critic via `experiments.precision.p63_transformer_critic_ppo.checkpointing.load_actor` / `load_critic`.
2. Build `StabilizeGrpoEnv` with P57-A reward weights as in P82 collect.
3. At each step: `act_with_value(actor, critic, ...)` then  
   `select_token(..., v2=v2_spec(), rule=None)` from  
   `experiments.precision.p73_first_contact_bilateral.controller`  
   (**`rule=None` is mandatory** — disables P73 post-contact overlays while retaining unchanged V2 pre-contact logic).
4. Terminal metrics from env `info["SUCCESS_RELAXED_12S"]` and original R3-ready / handoff predicates.

**Controller implementation source of truth for V2:**  
`experiments.precision.p71_contact_safe_alignment.controllers.select_token` (imported by the P73 wrapper).

**Do not use as frozen inference:**

- P73 `EventRule` overlays (`soft_approach`, etc.)
- P81 fixed-token inject (`tok_56` force)
- P71 V3/V4/V5 specs
- P69 fixed sequence-only interventions as production control
- Any post-P64 Actor/Critic training branches (P66 curriculum, etc.)

## Success definitions (original)

### SUCCESS_RELAXED_12S

From `success_relaxed.py` (`SUCCESS_RELAXED_VERSION = SUCCESS_RELAXED_12S_v1`):

- Built on SUCCESS_12S path with looser terminal pose: `RELAXED_THETA_DEG = 15.0`, `RELAXED_POS_MM = 15.0`
- Requires hold dwell `HOLD_S` in the final window (same hold machinery as success criteria)
- Explicitly **does not** redefine SUCCESS_NEW / SUCCESS_12S / R3F HANDOFF geometry

### R3-ready / R3F

Original predicate stack via `evaluate_predicates` (frame_ok + dwell / handoff). P82/P83 record `r3_ready` from terminal `_r3_terminal` and ever-flags; do not substitute alternate R3 definitions.

## Seeds and IC splits

| Item | Value | Source |
|---|---|---|
| P83 panel construction seed | `5483` | `experiments/precision/p83_recoverable_failure_mass/paths.py` |
| P54 dataset seed (pool) | `5478` | `panel.py` → `build_p54_dataset(seed=5478)` |
| Seeds per IC (P83) | `N_SEEDS = 1` | paths.py |
| Rollout seed function | `rollout_seed(split="train", traj_id=..., rollout_i=84000 + 30*ii + k)` | `panel.py` |
| Blocked eval / final-test ICs | Excluded via `blocked_eval_ids()` | `p64_ppo_multiround/schedule.py` |
| P83 panel composition | 32 ICs × 4 classes = 128; 64 reused from P82 + 64 newly collected | `BASELINE_PANEL_VALIDATION.md` |

## Final-test set status

A separate blocked / fresh final-test IC set exists via `_fresh_final_starts` inside `blocked_eval_ids()` and is **excluded** from P82/P83 panel construction (`final_test_excluded: true`).

**Status:** untouched for selection and for this consolidation. **Not evaluated automatically.**

## Environment / runtime notes

- Physics: unchanged MuJoCo (per freeze)
- Device used in campaigns: typically CUDA when available; CPU load of checkpoints verified for consolidation
- Max episode steps in P82/P83 collect: `MAX_EP_STEPS = 80` (see experiment `paths.py`)

## Why this freeze (selection rationale)

1. P70 V2 improved full-episode R3/SUCCESS vs Actor-only on **its own** held panel (C-focused P69 split; activation rate 1.0) without training (`do_not_train` for PPO). That evidence is **panel-specific**.
2. Critic recalibration / sensing / obs expansion / history-conditioned policy / fixed-token bias / post-contact overlays all failed promotion gates or closed-loop transfer (P75–P81, P73).
3. P83 post-audit Case D (fair mean non-C ≈ C_cf) + P84 B1 dispersion (concentration gate fail) → `stop_consolidate_baseline`; consolidate.

### Critical evaluation caveat (P83/P82)

On `panel_32.pkl`, V2 forced tokens in **2/128** episodes only (both B1, never contacted). Starts are typically already in pad contact, so V2’s `require_no_contact` gate never opens. **P83 MSP therefore measures ≈ P64 Actor-only behavior**, not active V2 closed-loop control. See `WORKFLOW_AUDIT.md`.

This is **not** a claim of global optimality for M111 or V2.
