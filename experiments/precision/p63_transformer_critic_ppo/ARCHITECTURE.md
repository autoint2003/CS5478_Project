# ARCHITECTURE — P63 Full-History Transformer Critic PPO

## Actor

- Reuses M111 causal Transformer policy (`models/k48_policy.py` via `load_m111`).
- Init: `results/experiments/p62_grpo_optimization/checkpoints/P62_epoch3.pt`.
- Obs: 39-D airborne + 7-D prev command; history window 160; pos embedding + 2-layer TransformerEncoder (d=64, 4 heads).
- Action head: 112 logits (111 motion + HANDOFF); residual 7-D; HANDOFF masked at sample time (R3F decides transfer).
- Deterministic residual decoding and observation preprocessing unchanged.

## Critic

- **Separate** `TransformerCritic` (`critic.py`): same `in_proj` / `pos` / `enc` as Actor.
- Backbone initialized by cloning Actor weights; **new** scalar `Linear(64→1)` value head.
- Independent parameters and Adam optimizer (`critic_lr=1e-4` vs actor `1e-5`).
- Causal attention only; full observation history; no future tokens; no privileged sim state.
- Variable-length prefixes padded with `src_key_padding_mask` (True = pad).
- \(V(h_t)\): expected discounted return given history through \(t\) (γ=1.0 initially).

## PPO

- On-policy buffer of **complete causal history prefixes** (one prefix per decision step).
- Frozen old log-probabilities; clipped surrogate (ε=0.2); entropy bonus; value MSE.
- GAE(λ=0.95); γ=1.0 to match prior undiscounted return objective.
- True terminals (DROP / HANDOFF / finite-horizon TIMEOUT): no value bootstrap.
- Artificial mid-buffer cuts (if used): bootstrap \(V(s_T)\).
- Minibatch shuffle over prefixes (never strip history from a timestep).
- Target approx KL 0.02 for early-stop diagnostics within an update.

## Environment / reward

- `StabilizeGrpoEnv` with `use_stabilization=False` + `PoseRewardWeights` (P57-A).
- SUCCESS_RELAXED_12S / R3F / M111 physics unchanged.

## Branches

- **A:** P54 schedule subset (or full 321) × 8 traj/IC.
- **B:** matched IC×traj budget with expanded C ICs from `C_DATA_MANIFEST.jsonl`.
