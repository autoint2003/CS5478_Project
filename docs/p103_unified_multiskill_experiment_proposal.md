# P103 — Retention-Constrained Unified Residual Learning (Locked Spec)

**Status:** EXECUTED — outcome `C_RETENTION_STILL_FAILS` (see `results/experiments/p103_retention_constrained_residual/`). Stop residual-only tuning; do not launch P104 without approval.  
**Grounding:** P102 causal audit. **Baseline:** P64 `ec08f59816aa2fdd` (immutable).

## Hypothesis (two necessary conditions)

Strong behavioral preservation on P64-successful trajectories can prevent residual-induced forgetting **while** allowing useful corrections on B1/C failure states.

1. Retain P64 previously successful physical behaviors.  
2. Improve recovery on ≥1 previously unsuccessful **root-disjoint** case.

Recovering baseline alone ≠ successful multi-skill consolidation.

## Architecture

P101 Arm B only:

\[
r(h)=\rho\max\tanh(W_{\mathrm{P64}}h+b_{\mathrm{P64}}+\Delta_\theta(h))
\]

Frozen: backbone, token head, proto, \(W_{\mathrm{P64}},b_{\mathrm{P64}}\). \(\Delta\) zero-init. No token-conditioned residual, router, or multi-head.

## Objective

\[
L_{\mathrm{total}}=L_{\mathrm{correction}}+\lambda_{\mathrm{ret}} L_{\mathrm{retention}}
\]

- \(L_{\mathrm{correction}}\): P101 feasibility-gated frozen-\(z_hat\) B1/C correction (mask C/D; early weight).  
- \(L_{\mathrm{retention}}=\mathbb{E}_{H\sim\mathcal{D}_{\mathrm{P64\_success}}}\|r_{\mathrm{new}}(H)-r_{\mathrm{P64}}(H)\|^2\) on **full** A1/B1/D success trajectories (not only t≤4).  
- Monitor \(\|a_{\mathrm{new}}-a_{\mathrm{P64}}\|\); do not rely on token KL.  
- Balanced sampling by physical root and protected class.  
- \(\lambda_{\mathrm{ret}}\) fixed by **training-only** loss-scale calibration before training (no post-hoc tuning).

## Comparisons

A = immutable P64 · B = reused P101 Arm B checkpoints · C = P103 retention-constrained.

Seeds {0,1,2}, 2000 updates, same LR/budget as P101. No sweep. No PPO/GRPO. No M111/R3F change. No sealed independent panel.

## Gates (summary)

- **Retention:** no losses on protected A1 and B1-success development panels across seeds (exploratory); report D. Must beat P101-B retention.  
- **New-skill:** ≥1 root-disjoint recovery vs P64 with no compensating protected losses.  
- Failures: offline ret↓ but CL still fails; Δ→0 with no new gains; correction↑ without root-disjoint CL; provenance/parity fail → stop residual-only tuning; reassess joint token–residual.

## Preservation-deviation tolerance (from P102)

P101-B mean residual L2 vs P64 on preservation histories ≈ 0.23–0.30 caused CL losses; B-tok+P64-res rescued 5/5.  
**Exploratory tolerance:** mean residual L2 on \(\mathcal{D}_{\mathrm{pres}}\) **< 0.05**; p95 residual L2 **< 0.10**; max per-dim **< 0.15**. Offline pass ≠ CL proof.
