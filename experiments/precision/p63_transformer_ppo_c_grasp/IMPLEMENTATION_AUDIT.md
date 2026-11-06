# IMPLEMENTATION_AUDIT — P63 Transformer PPO + C Grasp

## Phase 0 finding

**Existing P63 Critic PPO already implemented** under:

- Code: `experiments/precision/p63_transformer_critic_ppo/`
- Results: `results/experiments/p63_transformer_critic_ppo/`

This package **reuses** that core (Actor/Critic/GAE/PPO/sampling/checkpointing/eval) and adds C grasp event reward + Branch A/B/C pilots. No parallel PPO rewrite.

## Confirmed components

| Item | Status | Location |
|------|--------|----------|
| Full-history Transformer Actor | Reused | P62 / `load_actor(P62_epoch3)` |
| M111 action distribution | Reused | `policy_sample` + mask HANDOFF |
| Deterministic residual decoder | Reused | `decode_m111_action` |
| P62_epoch3 checkpoint | Present | `results/experiments/p62_grpo_optimization/checkpoints/P62_epoch3.pt` |
| Transformer Critic (same backbone, value head) | Reused | `critic.py` + `P63_critic_warm.pt` |
| P57-A original reward | Reused | `StabilizeGrpoEnv(use_stabilization=False)` |
| R3F HANDOFF | Unchanged | `R3FOnline` via env |
| SUCCESS_RELAXED_12S | Unchanged | success path untouched by C reward |
| P61 SHA-256 eval | Reused | `eval_dev.py` |
| On-policy rollout + old logp | Reused | `rollout.py` / `ppo.py` |
| New C ICs (32) | Present | `c_experts/` + `C_NEW_IC_MANIFEST.jsonl` |
| C grasp event reward | **New** | `contact_events.py`, `env_c_grasp.py` |

## Immutable constraints (not modified)

- Physics parameters
- M111 action vocabulary
- R3F gate
- SUCCESS / SUCCESS_RELAXED_12S criteria
- P53–P62 training/eval code retained

## New experiment tree

- Code: `experiments/precision/p63_transformer_ppo_c_grasp/`
- Results: `results/experiments/p63_transformer_ppo_c_grasp/`

## C grasp reward (first round)

C episodes only; once-only events; privilege geometry for reward/audit only:

1. Correct opposing sidewall recatch: **+2**
2. Sustained bilateral 0.5 s continuous: **+2**
3. R3F-ready (`R3_EARLY`): **+1**
4. Bad contact pop / lost-after-recatch: **−1** each
5. Positive event cap **+5**

## Pilot plan

Matched simulator budget across branches (default 64 IC × 8 traj × 4 updates ≈ 2048 traj/branch).

1. Branch A — original 321-pool stratified subset, P57-A reward
2. Branch B — same budget, swap in new C ICs, P57-A reward
3. Branch C — same as B + C grasp events

**Do not start B/C if A is unstable. No long training in this phase.**
