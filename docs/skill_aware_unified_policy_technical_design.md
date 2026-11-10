# Skill-Aware Unified Policy (SAUP) — Unified Technical Design

**Status:** Documentation only. No Actor implementation / three-arm training authorized until explicit approval.  
**Phase:** P100 Phase 0 complete (frozen-token feasibility audit + preregistration).  
**Baseline Actor:** `results/experiments/p64_ppo_multiround/checkpoints/P64_step4_actor.pt`  
**SHA-256 prefix:** `ec08f59816aa2fdd`  
**Audit artifacts:** `results/experiments/p100_frozen_token_feasibility_audit/`  
**Audit runner:** `experiments/precision/p100_frozen_token_feasibility_audit/run_audit.py`

This is the **single** SAUP design document (architecture + theory + P100 audit + experiment preregistration).

Evidence tags: **CODE** | **ARTIFACT** | **HYPOTHESIS** | **UNKNOWN**.

Outcome vocabulary for hypotheses: **Supports** | **Against** | **No improvement** | **Inconclusive**  
(“No improvement” ≠ “Against”. Exploratory claims ≠ P64 replacement. Seeds ≠ independent physical roots.)

---

## 1. Executive Summary

Deployed Actor = one causal Transformer (`models/k48_policy.Policy`):

- `concat(obs_39, prev_7) → ℝ⁴⁶` → `ctx ∈ ℝ⁶⁴`
- one token head `logits ∈ ℝ¹¹²` (111 + HANDOFF)
- one residual `r = 0.35·tanh(Linear(ctx))` — **not** token-conditioned (**CODE**)
- `a = clip(proto[z] + r, ±1)` + official R3F / `SUCCESS_RELAXED_12S`

**SAUP residual candidate (not implemented):** one shared residual-delta

\[
r=\rho_{\max}\tanh(W_{\mathrm{P64}}h+b_{\mathrm{P64}}+\Delta_\theta),\qquad
\Delta\equiv 0\ \text{at init}.
\]

Arm B: \(\Delta(h)\). Arm C: \(\Delta(h,z_hat)\). Token head frozen in the first experiment.

| Hypothesis | Claim |
|---|---|
| **H1** | Nonlinear \(\Delta(h)\) beats linear P64 residual |
| **H2** | Token-conditioned \(\Delta(h,z_hat)\) beats capacity-matched \(\Delta(h)\) |
| **H3** | Gains without unacceptable A1/D/B1 loss or train-root memorization |

**P100 audit (expert B1/C train):** **88%** of actions exactly feasible under frozen \(z_hat\) (A+B); early \(t=0..4\) only **40%** (C early **22.5%**, mostly token-limited).

**Recommendation:** **Exploratory GO** (after approval) for three-arm frozen-token residual-delta. **NO-GO** for P64 replacement, token unfreeze in this contrast, PPO/GRPO.

---

## 2. Objective and Non-Negotiable Constraints

Maximize class-balanced physical closed-loop success on A1/B1/C/D under official R3F, with **one** deployed policy.

**Allowed at inference:** one Transformer, one M111 token head, one residual head (optionally internally token-conditioned), existing decoder + R3F, history-only inputs.

**Disallowed:** routers, multi residual heads, skill-label inputs, separate skill Actors, class-specific inference branches.

---

## 3. Existing System Architecture (CODE)

### 3.1 Load path

P64 → `p63/.../checkpointing.load_actor` → `policy_sample.load_policy_actor` → `m111_checkpoint.load_m111` → `k48_policy.Policy`.  
Vocab: M111, `res_cap=0.35`, `residual_rho=[0.35]×7`.  
`FullHistoryPolicy` is **not** P64 (P64 has `pos.Embedding(160,64)`).

### 3.2 Modules (`k48_policy.py`)

| Module | Shape | Role |
|---|---|---|
| `in_proj` | 46→64 | obs+prev |
| `pos` | Emb(160,64) | absolute PE |
| `enc` | Transformer×2, d=64, 4 heads, ff=128 | causal |
| `logits` | 64→112 | tokens |
| `res` | 64→7 (zero-init) | residual preact |
| `proto` | (111,7) buffer | frozen prototypes |

### 3.3 Forward / decode

\[
r=\rho_{\max}\tanh(W_r\,\mathrm{ctx}+b_r),\quad
\hat z=\arg\max\mathrm{mask\_HANDOFF}(\ell),\quad
a=\Pi(\mathrm{proto}[\hat z]+r).
\]

Env: `decode_m111_action` clips residual per-dim ρ then action to ±1. HANDOFF masked at sample; R3F drives handoff (`r3f_online.py`).

### 3.4 Diagram

```
obs,prev → in_proj+pos → Transformer×2 → ctx
                              ├─ logits → z_hat (HANDOFF masked)
                              └─ res Linear → r = 0.35·tanh  (no z input today)
proto[z] + r → clip → MuJoCo → R3F → SUCCESS_RELAXED_12S
```

### 3.5 Current training / RL note

BC: CE + residual/action MSE (± KL to P64). PPO/GRPO today samples **tokens only**; residual density undefined ⇒ joint PPO not yet well-posed.

---

## 4. Historical Evidence (P85–P99) — ARTIFACT

| Exp | Core | Strength |
|---|---|---|
| P89/P95 | Early residual causality; residual swaps rescue | **Causal** |
| P90–P94 | B1 residual gains trade off others; routing not deployable | Mixed |
| P97 | Sealing; **0** C preservation; **10** verified C correction (6 expert-M111 + 4 new-IC); 4 M111 replay fails | Provenance |
| P98 | Offline↑, CL C **0/2**; P98-B harms A1/D | Offline≠CL |
| P99 | C train 1/8→4/8; holdout 0/2; A1/D lost 2/3; agg +0 | Train-root only; **not** full DAgger |

P98 split of the 10 C roots: **8 train / 2 holdout**.

---

## 5. Root Provenance

Canonical `phys_root`: strip `plan_*__|A|B|C|D__` and trailing `__`+hex≥8.  
Ban: frozen independent, P93 panels, P94 ind, `gbi_351`, `gbi_022`.  
Exposed eval roots: diagnostics only — never train or model selection.

---

## 6. Current Policy Math

\[
\pi(z,r\mid H)=\pi(z\mid H)\,\delta(r-f(H)),\quad f(H)=\rho_{\max}\tanh(W_{\mathrm{P64}}h+b_{\mathrm{P64}}).
\]

Expert residual labels are **token-relative**: \(r^*(a^*,z)=\mathrm{clip}(a^*-\mathrm{proto}[z],\pm\rho)\).

---

## 7. Why Multi-Skill Learning Fails (ranked)

1. Shared residual field / early residual drift (**causal** P89/P95)  
2. Thin correction-only C data → train-root fit (P97–P99)  
3. Backbone drift harming A1/D (P98-B, P99)  
4. Teacher vs student state mismatch (P99)  
5. Gradient conflict (weak)  
6. M111 unrepresentable segments (P97)

---

## 8. Theory — H1 / H2 / H3

### 8.1 Hypotheses and outcomes

| ID | Supports | Against | No improvement | Inconclusive |
|---|---|---|---|---|
| H1 | B ≫ A on primary CL | B ≪ A | B ≈ A | underpowered / protocol break |
| H2 | C ≫ B on primary CL | C ≪ B | C ≈ B | underpowered / unmatched capacity |
| H3 | retention + root-disjoint gains | clear retention/gen failure | flat | \(n_{\mathrm{C}}\) too small |

Offline-only gains ⇒ **not** Supports.

### 8.2 Risk math (tempered)

Bayes gap \(R_{\mathrm{unc}}-R_{\mathrm{cond}}\ge 0\), strict iff \(R^*\not\!\perp Z^*\mid H\).

- If \(Z^*=g(H)\) a.s. ⇒ gap **0** (no new information).  
- Benefit of conditioning is then **inductive bias / finite capacity**, not Shannon info.  
- Deploy uses \(\hat Z\); mismatch can **hurt**.  
- \(\Delta R_{\mathrm{Bayes}}>0 \not\Rightarrow \Delta\mathbb{P}(\mathrm{SUCCESS})>0\).

### 8.3 Preactivation residual delta (required)

**Reject** \(r_{\mathrm{P64}}+\rho\tanh(\mathrm{MLP})\).

**Require** \(r=\rho_{\max}\tanh(W_{\mathrm{P64}}h+b_{\mathrm{P64}}+\Delta)\) with \(\Delta\equiv 0\) at init ⇒ exact P64 residual/token/action parity; bounds preserved; decoder/R3F unchanged.

### 8.4 Primitives ≠ skills

M111 tokens are short-horizon primitives. Skills are temporally extended. Token-conditioned residual alone is **not** theoretically sufficient; history representation and objectives matter equally. No router / multi-head.

---

## 9. P100 Frozen-Token Feasibility Audit

### 9.1 Method

Training-eligible B1/C expert demos (P98) + verified student-state continuations (P99).  
Per step: rebuild causal history → frozen P64 \(z_hat\) → expert \(z^*,a^*\) → feasible set

\[
\mathcal{Z}(a^*)=\{k:\exists r\in[-\rho,\rho]^7,\ \Pi(\mathrm{proto}[k]+r)=a^*\}
\]

(tol \(10^{-5}\)). Mutually exclusive:

| Cat | Definition |
|---|---|
| **A** | \(z_hat\in\mathcal{Z}(a^*)\) and \(z_hat=z^*\) |
| **B** | \(z_hat\in\mathcal{Z}(a^*)\) and \(z_hat\neq z^*\) |
| **C** | \(z_hat\notin\mathcal{Z}\), \(\mathcal{Z}\neq\emptyset\) (token-limited) |
| **D** | \(\mathcal{Z}=\emptyset\) (exactly unrepresentable) |

Exact representability ≠ physical SUCCESS. Class D ≠ “no recovery exists.”

### 9.2 Results

| Slice | Feasible @ \(z_hat\) (A+B) | Notes |
|---|---:|---|
| Expert B1+C train | **88.0%** | A 68.1 / B 20.0 / C 8.3 / D 3.6 |
| Expert B1 | 95.2% | D=0 |
| Expert C | 83.3% | |
| Expert early \(t=0..4\) | **40.0%** | C early **22.5%** (C 45% + D 32.5%) |
| All \(t>4\) | **96.0%** | |

**Implication:** residual-delta Arms B/C can test H1/H2 on A+B mass; they **cannot** be expected to fix early-C token errors under frozen tokens.

---

## 10. Proposed Architecture (not implemented)

| Piece | Spec |
|---|---|
| Backbone / logits / proto | Frozen P64 |
| Arm A | Frozen linear residual |
| Arm B | \(\Delta^{\mathrm{unc}}(h)\), zero-init |
| Arm C | \(\Delta^{\mathrm{cond}}(h,z_hat)\) on `concat(ctx, emb(z_hat), proto[z_hat])`, zero-init |
| Params | B vs C matched within 10% |
| Init check | identical \(\hat z,r,a\) within \(10^{-6}\) before train |

---

## 11. Primary Supervision Rule (locked for B and C)

**Feasibility-gated frozen-predicted-token residual targets:**

1. History from expert / verified student-state demos.  
2. \(z_hat=\arg\max\mathrm{mask\_HANDOFF}(\mathrm{frozen\ P64}(H))\).  
3. If A/B: \(r^*=\mathrm{clip}(a^*-\mathrm{proto}[z_hat],\pm\rho)\);  
   \(L=\|r-r^*\|_2^2+\|\Pi(\mathrm{proto}[z_hat]+r)-a^*\|_2^2\) (early weight \(t\le 4\), `EARLY_W=3`).  
   Arm C conditions on **`z_hat`**.  
4. If C/D: **mask** residual/action loss.  
5. No token CE optimization; no teacher-\(z^*\) primary residual targets; no post-hoc mixture.

---

## 12. Preregistered Three-Arm Experiment

### 12.1 Data

**Train:** P98 B1×6 + C train×8; optional listed P99 student-state demos.  
**Eval C (two distributions):**

1. **C-Feasible** — expert+M111-verified, root-disjoint (current holdouts `gbi_217`,`gbi_152`, \(n=2\): reportable, **not decisive**).  
2. **C-Unfiltered** — independently sampled legal C ICs from original task prior, no success filter; sealed vs train/C-Feasible/bans.

Expand to \(n_{\mathrm{C-Feasible}}\ge 6\), \(n_{\mathrm{C-Unfiltered}}\ge 8\) before decision-grade C / replacement claims.  
Also: B1 paired panels; A1/D/B1 retention. C-train success = diagnostic only, never selection.

Seeds `{0,1,2}` = optimization variance only — **not** new roots.

### 12.2 Metrics

Offline (secondary): residual/action L2 on unmasked A/B steps; ABCD rates.  
CL (primary): `SUCCESS_RELAXED_12S` on C-Feasible, C-Unfiltered (when present), B1 net, A1/D/B1 retention; root-level paired tables + seed mean±std / Wilson-bootstrap.

Primary score for H1/H2: \(S=\tfrac12(\mathrm{B1\_eval}+\mathrm{C\text{-}Feasible})\).

### 12.3 Gates

**G0:** init parity, sealing, sha, loss rule, param match.

**Exploratory (H1/H2 claims only)**  
- E-Retain floor: A1 & D & B1-success lost/base ≤ 1/3.  
- Supports / Against / No improvement / Inconclusive per §8.1 on \(S\).  
- **Removed:** B1 improvement substituting for C root-disjoint evidence.  
- If \(n_{\mathrm{C-Feasible}}<6\) and B1 signal unclear ⇒ default **Inconclusive** for C-heavy claims.

**Strict P64 replacement (separate, harder)** — all required:  
stricter retention (≤1/6), C-Feasible \(n\ge 6\) with clear gain, C-Unfiltered \(n\ge 8\) non-degradation, B1 net ≥ +1, sealed independent panel, human approval. No auto-replace.

### 12.4 Budget

A eval + B×3 + C×3; 2000 steps/seed; balanced B1:C on unmasked steps.  
Forbidden: backbone/token unfreeze, large sweeps, RL, post-hoc loss-rule changes.

### 12.5 Approval stops

1. Design/audit (this doc) — awaiting review.  
2. Implementation approval (residual-delta code).  
3. Execution approval (Arms A/B/C).  
4. C-panel expansion (for replacement-grade).  
5. Replacement approval only if strict gates pass.

---

## 13. Failure Modes (selected)

| Mode | Diagnostic |
|---|---|
| Histories not identifiable | irreducible expert-action confusion given \(H\) |
| M111 can’t represent \(a^*\) | class D rate / P97-style replay |
| \(\hat z\) errors | class C rate; teacher-force vs predicted ablation |
| Overfit train roots | train↑ holdout flat (P99) |
| A1/D forgetting | paired retention |
| Offline↑ CL flat | dual reporting mandatory |
| Eval leakage | phys_root audit |

---

## 14. Future Closed-Loop Training (post-validation only)

Only after exploratory H1/H2 resolution: multi-skill BC, student-state corrections, retention constraints.  
PPO/GRPO **blocked** until residual density + joint \(\log\pi(z,r\mid H)\) + clipping Jacobian are specified. Single deployed Actor remains mandatory.

---

## 15. Implementation Impact (if approved later)

| Path | Change |
|---|---|
| `models/k48_policy.py` or `saup_policy.py` | residual-delta head |
| `m111_checkpoint.py` | save/load new keys |
| BC trainers / `policy_sample` | pass \(z_hat\) into \(\Delta\) if API splits |
| Env / R3F / M111 book | **no change** |

---

## 16. Open Questions

1. Approve exploratory implementation + run?  
2. Expand C-Feasible / C-Unfiltered before execution?  
3. Include P99 student-state demos in train manifest?  
4. Keep \(W_{\mathrm{P64}}\) frozen inside preactivation (default yes)?

---

## 17. Evidence Traceability

| Claim | Tag | Source |
|---|---|---|
| P64 = `k48_policy.Policy`, residual not token-cond. | CODE | `k48_policy.py` |
| SHA `ec08f59816aa2fdd` | CODE | checkpoint hash |
| Residual causality | ARTIFACT | P89/P95 |
| 0 C preservation; 10 verified C corr; 8/2 split | ARTIFACT | P97/P98 |
| Offline≠CL; P99 train-only C | ARTIFACT | P98/P99 |
| Expert feasible@\(z_hat\)=88%; early C token-limited | ARTIFACT | P100 audit |
| Token cond. = inductive bias if \(Z=g(H)\) | THEORY | §8 |
| Preactivation \(\Delta\) for init parity | THEORY | §8.3 |

---

## 18. Final Recommendation

1. **Exploratory GO** (pending approval) for frozen-token Arms A/B/C with feasibility-gated \(z_hat\) targets.  
2. **NO-GO** for P64 replacement, token unfreeze in this contrast, PPO/GRPO.  
3. Treat \(n_{\mathrm{C}}=2\) as provisional; seeds do not fix root scarcity.  
4. Use Supports/Against/No-improvement/Inconclusive; never substitute B1 for C generalization.

**STOP.** Await explicit approval before modifying Actor code or launching training.

---

## Appendix A — Key paths

| Topic | Path |
|---|---|
| Actor | `models/k48_policy.py` |
| M111 decode | `models/m111_checkpoint.py` |
| Quantize / \(\mathcal{Z}\) helpers | `training/skill_aware_action_vocab.py` |
| Sample / HANDOFF mask | `experiments/precision/p53_independent_grpo64/policy_sample.py` |
| R3F | `experiments/precision/p44_r3f_unified_sft/r3f_online.py` |
| P64 ckpt | `results/experiments/p64_ppo_multiround/checkpoints/P64_step4_actor.pt` |
| P100 audit | `experiments/precision/p100_frozen_token_feasibility_audit/run_audit.py` |
| P100 results | `results/experiments/p100_frozen_token_feasibility_audit/` |

## Appendix B — Residual-delta pseudocode (not implemented)

```python
# preactivation delta; Delta final layer zero-init ⇒ P64 parity at init
preact_p64 = res_lin(ctx)                  # frozen P64 Linear
delta = mlp_delta(ctx)                     # Arm B
# Arm C: mlp_delta(cat([ctx, emb(z_hat), proto[z_hat]]))
r = res_cap * torch.tanh(preact_p64 + delta)
a = torch.clamp(proto[z_hat] + r, -1, 1)
```
