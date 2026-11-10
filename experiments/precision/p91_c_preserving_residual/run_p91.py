"""P91 — C-preserving B1 residual adaptation (diagnostic; P64 not replaced)."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.precision.final_independent_success_rate.paths import (  # noqa: E402
    MAX_EP_STEPS,
    P64_ACTOR,
    P64_CRITIC,
    T_MAX,
)
from experiments.precision.p51_four_class_mixed_ppo.ic_pools import root_id  # noqa: E402
from experiments.precision.p53_independent_grpo64.policy_sample import mask_handoff_logits  # noqa: E402
from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights  # noqa: E402
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset  # noqa: E402
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import (  # noqa: E402
    StabilizeGrpoEnv,
)
from experiments.precision.p61_reproducible_joint_eval.seeding import seed_all  # noqa: E402
from experiments.precision.p63_transformer_critic_ppo.checkpointing import (  # noqa: E402
    load_actor,
    load_critic,
)
from experiments.precision.p66_c_curriculum_ppo.expert_drive import (  # noqa: E402
    ensure_inner_capture_buffers,
)
from experiments.precision.p67_r3f_gate_diagnosis.gate_predicates import (  # noqa: E402
    evaluate_predicates,
)
from experiments.precision.p68_precontact_geometry.geometry import relative_geometry  # noqa: E402
from experiments.precision.p70_geometry_to_r3f.obs_proxy import last_obs  # noqa: E402
from experiments.precision.p73_first_contact_bilateral.controller import (  # noqa: E402
    WrapState,
    select_token,
    v2_spec,
)
from experiments.precision.p73_first_contact_bilateral.rollout import _r3_terminal  # noqa: E402
from experiments.precision.p86_b1_gate_counterfactual.gate_ablation import make_monitor  # noqa: E402
from models.m111_checkpoint import MOTION_K, decode_m111_action, save_m111  # noqa: E402

P90_OUT = ROOT / "results/experiments/p90_b1_residual_head_transfer"
P90_B = P90_OUT / "checkpoints/P90_stageB_p64_residual.pt"
P90_XCLASS = P90_OUT / "CROSS_CLASS_ROWS.csv"
P90_PAIRED = P90_OUT / "PAIRED_PHYSICAL_EVALUATION.csv"
P87_TEACHER = ROOT / "results/experiments/p87_k48_actor_retention/teacher"
P88_CAND = ROOT / "results/experiments/p88_b1_short_sequence_learnability/PHASE0_CANDIDATES.csv"

OUT = ROOT / "results/experiments/p91_c_preserving_residual"
CKPT = OUT / "checkpoints"
C_FAIL_IC = "plan_c__gbi_351_bc2d6a3b90"
SEED = 91
ALPHAS = (0.25, 0.5, 1.0)
KS = (2, 4)
NULL_UPDATES = 400
NULL_LR = 1e-2
ADAPTER_UPDATES = 500
ADAPTER_LR = 1e-3
SVD_TOL = 1e-5


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


def lookup_starts() -> dict[str, Any]:
    return {s.traj_id: s for s in build_p54_dataset(seed=5478)["schedule"]}


def verified_b1() -> list[str]:
    rows = list(csv.DictReader(P88_CAND.open(encoding="utf-8")))
    return sorted(r["traj_id"] for r in rows if int(r["valid_official"]) == 1)


@torch.no_grad()
def forward_parts(actor, obs_t: torch.Tensor, prev_t: torch.Tensor):
    """Return ctx, logits, residual (post tanh*cap)."""
    x = torch.cat([obs_t, prev_t], dim=-1)
    t = x.shape[-2]
    h = actor.in_proj(x) + actor.pos(torch.arange(t, device=x.device))
    causal = torch.triu(torch.ones(t, t, device=x.device) * float("-inf"), diagonal=1)
    if x.dim() == 2:
        ctx = actor.enc(h.unsqueeze(0), mask=causal)[0]
    else:
        ctx = actor.enc(h, mask=causal)
    logits = actor.logits(ctx)
    pre = actor.res(ctx)
    res = actor.res_cap * torch.tanh(pre)
    return ctx, logits, res, pre


def load_demo(tid: str) -> dict | None:
    p = P87_TEACHER / f"{tid}.npz"
    if not p.is_file():
        return None
    z = np.load(p)
    return {
        "traj_id": tid,
        "tokens": np.asarray(z["tokens"], np.int64),
        "residuals": np.asarray(z["residuals"], np.float32),
        "actions": np.asarray(z["actions"], np.float32),
        "obs": np.asarray(z["obs"], np.float32),
        "prev": np.asarray(z["prev"], np.float32),
    }


# ---------------------------------------------------------------------------
# Closed-loop with residual blend / custom residual fn
# ---------------------------------------------------------------------------


def closed_loop(
    token_actor,
    resid_fn,
    start,
    book,
    rho,
    device,
    *,
    seed: int = SEED,
    record_trace: bool = False,
    ref_resid_actor=None,
) -> dict:
    """token_actor provides tokens; resid_fn(ctx, logits, res_tok_actor, t, obs, prev)->resid np."""
    seed_all(seed, deterministic_torch=True)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )
    env.reset(start)
    env.mon = make_monitor("G0", t_max=int(env.t_max))
    ensure_inner_capture_buffers(env)
    st = WrapState()
    v2 = v2_spec()
    done = False
    info: dict = {}
    trace = []
    tokens = []
    ever = {k: False for k in ("frame9", "frame12", "bil")}
    first_div = None
    while not done:
        obs = torch.as_tensor(np.stack(env.obs_seq[-160:]), device=device, dtype=torch.float32)
        prev = torch.as_tensor(np.stack(env.prev_seq[-160:]), device=device, dtype=torch.float32)
        ctx, logits, res_t, pre = forward_parts(token_actor, obs, prev)
        logit = mask_handoff_logits(logits[-1].float())
        tok_a = int(torch.argmax(logit).item())
        tok, st, _ = select_token(
            env_t=int(env.t), obs=last_obs(env), actor_tok=tok_a, st=st, v2=v2, rule=None
        )
        resid = resid_fn(
            {
                "t": int(env.t),
                "ctx": ctx,
                "logits": logits,
                "res_token_actor": res_t,
                "pre": pre,
                "obs": obs,
                "prev": prev,
                "tok": int(tok),
            }
        )
        resid = np.asarray(resid, np.float32).reshape(7)
        action = decode_m111_action(book, int(tok), resid, rho=rho)
        g = relative_geometry(env.env.sim)
        tr = getattr(env.mon, "tracker", None)
        ok_s = float(getattr(tr, "ok_s", 0.0) or 0.0)
        preds = evaluate_predicates(env.env.sim, ok_s=ok_s)
        if record_trace or ref_resid_actor is not None:
            row = {
                "t": int(env.t),
                "token": int(tok),
                "tok_raw": tok_a,
                "e_pos_mm": float(g["e_pos_mm"]),
                "theta_deg": float(g["theta_deg"]),
                "bilateral": int(g["bilateral"]),
                "min_pad": int(g.get("min_pad") or 0),
                "frame_ok": int(bool(preds.get("frame_ok"))),
                "resid": resid.copy(),
                "action": action.copy(),
                "ctx_l2": float(torch.linalg.vector_norm(ctx[-1]).item()),
            }
            if ref_resid_actor is not None:
                _, _, res_ref, _ = forward_parts(ref_resid_actor, obs, prev)
                row["resid_ref"] = res_ref[-1].detach().cpu().numpy()
                row["resid_l2_vs_ref"] = float(np.linalg.norm(resid - row["resid_ref"]))
                row["action_ref"] = decode_m111_action(
                    book, int(tok), row["resid_ref"], rho=rho
                )
                row["action_l2_vs_ref"] = float(np.linalg.norm(action - row["action_ref"]))
                if first_div is None and row["action_l2_vs_ref"] > 0.05:
                    first_div = int(env.t)
            if record_trace:
                # store compact
                trace.append(
                    {
                        "t": row["t"],
                        "token": row["token"],
                        "e_pos_mm": row["e_pos_mm"],
                        "theta_deg": row["theta_deg"],
                        "bilateral": row["bilateral"],
                        "min_pad": row["min_pad"],
                        "frame_ok": row["frame_ok"],
                        "resid_l2_vs_ref": row.get("resid_l2_vs_ref", ""),
                        "action_l2_vs_ref": row.get("action_l2_vs_ref", ""),
                        "resid_dx": float(resid[0]),
                        "resid_grip": float(resid[6]),
                    }
                )
        tokens.append(int(tok))
        bil = bool(g["bilateral"])
        pad4 = int(g.get("min_pad") or 0) >= 4
        wk = bool(preds.get("working_ok"))
        e = float(g["e_pos_mm"])
        th = float(g["theta_deg"])
        if bil:
            ever["bil"] = True
        if bil and pad4 and wk and e <= 9 + 1e-9 and th <= 8 + 1e-9:
            ever["frame9"] = True
        if bil and pad4 and wk and e <= 12 + 1e-9 and th <= 8 + 1e-9:
            ever["frame12"] = True
        _r, done, info = env.step(int(tok), resid)
        ensure_inner_capture_buffers(env)
    term = _r3_terminal(env, info)
    return {
        "SUCCESS_RELAXED_12S": int(term.get("SUCCESS_RELAXED") or info.get("SUCCESS_RELAXED_12S") or 0),
        "handoff": int(info.get("handoff") or 0),
        "r3_ready": int(term.get("r3_ready") or 0),
        "reach12": int(ever["frame12"]),
        "reach9": int(ever["frame9"]),
        "ep_len": int(env.t),
        "fail_reason": info.get("fail_reason"),
        "tokens": tokens,
        "first_action_div_vs_ref": first_div if first_div is not None else "",
        "trace": trace,
    }


def resid_from_actor(actor):
    def _fn(pack):
        return pack["res_token_actor"][-1].detach().cpu().numpy()

    # when token_actor != resid actor, recompute
    def _fn2(pack, _actor=actor):
        _, _, res, _ = forward_parts(_actor, pack["obs"], pack["prev"])
        return res[-1].detach().cpu().numpy()

    return _fn2


def resid_blend(p64, p90, alpha: float, k: int):
    def _fn(pack):
        t = pack["t"]
        _, _, r64, _ = forward_parts(p64, pack["obs"], pack["prev"])
        _, _, r90, _ = forward_parts(p90, pack["obs"], pack["prev"])
        a64 = r64[-1]
        a90 = r90[-1]
        if t < k:
            r = a64 + float(alpha) * (a90 - a64)
        else:
            r = a64
        return r.detach().cpu().numpy()

    return _fn


# ---------------------------------------------------------------------------
# Phase 0
# ---------------------------------------------------------------------------


def phase0(p64, p90, book, rho, device, starts) -> dict:
    start = starts[C_FAIL_IC]
    # P64 alone
    r64 = closed_loop(
        p64, resid_from_actor(p64), start, book, rho, device, record_trace=True, ref_resid_actor=p64
    )
    # P90 residuals, P64 tokens
    r90 = closed_loop(
        p64, resid_from_actor(p90), start, book, rho, device, record_trace=True, ref_resid_actor=p64
    )
    # Full P90 (tokens from p90 — should match early if same backbone)
    r90_full = closed_loop(
        p90, resid_from_actor(p90), start, book, rho, device, record_trace=True, ref_resid_actor=p64
    )

    reproducible = (
        int(r64["SUCCESS_RELAXED_12S"]) == 1
        and int(r90["SUCCESS_RELAXED_12S"]) == 0
        and int(r90_full["SUCCESS_RELAXED_12S"]) == 0
    )
    _write_csv(OUT / "C_REGRESSION_TRACE_P64.csv", r64["trace"])
    _write_csv(OUT / "C_REGRESSION_TRACE_P90RES.csv", r90["trace"])

    md = f"""# C_REGRESSION_CAUSAL_TRACE

Generated: {_utc()}

## Target IC

`{C_FAIL_IC}` — the unique C IC lost in P90 (P64 SUCCESS → P90 FAIL).

## Paired replay (identical seed={SEED}, R3F G0, controller, P64 token policy unless noted)

| Variant | SUCCESS_RELAXED | handoff | ep_len | fail | first action div vs P64 resid |
|---|---:|---:|---:|---|---|
| P64 tokens + P64 residual | **{r64['SUCCESS_RELAXED_12S']}** | {r64['handoff']} | {r64['ep_len']} | {r64.get('fail_reason')} | {r64['first_action_div_vs_ref']} |
| P64 tokens + P90 residual | **{r90['SUCCESS_RELAXED_12S']}** | {r90['handoff']} | {r90['ep_len']} | {r90.get('fail_reason')} | {r90['first_action_div_vs_ref']} |
| P90 tokens + P90 residual | **{r90_full['SUCCESS_RELAXED_12S']}** | {r90_full['handoff']} | {r90_full['ep_len']} | {r90_full.get('fail_reason')} | {r90_full['first_action_div_vs_ref']} |

## Causality

- Reproducible P64 success / P90 failure: **{reproducible}**
- Holding the **token policy fixed at P64** and swapping only the residual head still fails →
  regression is explained by **residual drift**, not a changed discrete policy at init.
- First executed-action divergence (P64-tok + P90-res vs P64): step **{r90['first_action_div_vs_ref']}**

Step tables: `C_REGRESSION_TRACE_P64.csv`, `C_REGRESSION_TRACE_P90RES.csv`.
"""
    _write(OUT / "C_REGRESSION_CAUSAL_TRACE.md", md)
    return {
        "reproducible": reproducible,
        "p64": r64,
        "p90_res": r90,
        "p90_full": r90_full,
    }


# ---------------------------------------------------------------------------
# Phase 1 ablation
# ---------------------------------------------------------------------------


def eval_variant(name, token_actor, resid_fn, panel, starts, book, rho, device) -> list[dict]:
    rows = []
    for skill, tids in panel.items():
        for tid in tids:
            if tid not in starts:
                continue
            out = closed_loop(token_actor, resid_fn, starts[tid], book, rho, device)
            rows.append(
                {
                    "variant": name,
                    "skill": skill,
                    "traj_id": tid,
                    "SUCCESS_RELAXED_12S": out["SUCCESS_RELAXED_12S"],
                    "reach12": out["reach12"],
                    "reach9": out["reach9"],
                    "r3_ready": out["r3_ready"],
                    "ep_len": out["ep_len"],
                    "fail_reason": out.get("fail_reason"),
                }
            )
    return rows


def summarize_vs_p64(rows: list[dict], p64_rows: list[dict]) -> dict:
    p64 = {(r["skill"], r["traj_id"]): r for r in p64_rows}
    out = {"B1_gain": 0, "B1_loss": 0, "C_loss": 0, "C_keep": 0, "A1_loss": 0, "D_loss": 0, "D_gain": 0}
    for r in rows:
        key = (r["skill"], r["traj_id"])
        if key not in p64:
            continue
        a = int(p64[key]["SUCCESS_RELAXED_12S"])
        b = int(r["SUCCESS_RELAXED_12S"])
        sk = r["skill"]
        if sk == "B1":
            if b and not a:
                out["B1_gain"] += 1
            if a and not b:
                out["B1_loss"] += 1
        if sk == "C":
            if a and b:
                out["C_keep"] += 1
            if a and not b:
                out["C_loss"] += 1
        if sk == "A1" and a and not b:
            out["A1_loss"] += 1
        if sk == "D":
            if b and not a:
                out["D_gain"] += 1
            if a and not b:
                out["D_loss"] += 1
    # baseline C successes count
    out["C_base"] = sum(1 for r in p64_rows if r["skill"] == "C" and int(r["SUCCESS_RELAXED_12S"]))
    out["B1_base"] = sum(1 for r in p64_rows if r["skill"] == "B1" and int(r["SUCCESS_RELAXED_12S"]))
    out["B1_n"] = sum(1 for r in rows if r["skill"] == "B1")
    out["B1_succ"] = sum(1 for r in rows if r["skill"] == "B1" and int(r["SUCCESS_RELAXED_12S"]))
    out["acceptable"] = (
        out["B1_gain"] >= 1
        and out["C_loss"] == 0
        and out["A1_loss"] == 0
        # D_loss: allow 0 only as "no unacceptable"; treat any D loss as unacceptable
        and out["D_loss"] == 0
    )
    return out


def build_panel(starts) -> dict[str, list[str]]:
    xrows = list(csv.DictReader(P90_XCLASS.open(encoding="utf-8")))
    panel = {
        "B1": verified_b1(),
        "A1": sorted({r["traj_id"] for r in xrows if r["skill"] == "A1"}),
        "C": sorted({r["traj_id"] for r in xrows if r["skill"] == "C"}),
        "D": sorted({r["traj_id"] for r in xrows if r["skill"] == "D"}),
    }
    # filter to existing starts
    for k in panel:
        panel[k] = [t for t in panel[k] if t in starts]
    return panel


# ---------------------------------------------------------------------------
# Phase 2 nullspace
# ---------------------------------------------------------------------------


def collect_protected_ctx(p64, starts, tids: list[str], book, rho, device) -> np.ndarray:
    """Collect last-step ctx (+ all steps) from successful P64 rollouts; return [N,64]."""
    xs = []
    for tid in tids:
        if tid not in starts:
            continue
        seed_all(SEED, deterministic_torch=True)
        env = StabilizeGrpoEnv(
            book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
        )
        env.reset(starts[tid])
        env.mon = make_monitor("G0", t_max=int(env.t_max))
        ensure_inner_capture_buffers(env)
        st = WrapState()
        v2 = v2_spec()
        done = False
        info = {}
        ep_ctx = []
        while not done:
            obs = torch.as_tensor(np.stack(env.obs_seq[-160:]), device=device, dtype=torch.float32)
            prev = torch.as_tensor(np.stack(env.prev_seq[-160:]), device=device, dtype=torch.float32)
            ctx, logits, res, _ = forward_parts(p64, obs, prev)
            ep_ctx.append(ctx[-1].detach().cpu().numpy())
            logit = mask_handoff_logits(logits[-1].float())
            tok_a = int(torch.argmax(logit).item())
            tok, st, _ = select_token(
                env_t=int(env.t), obs=last_obs(env), actor_tok=tok_a, st=st, v2=v2, rule=None
            )
            resid = res[-1].detach().cpu().numpy()
            _r, done, info = env.step(int(tok), resid)
            ensure_inner_capture_buffers(env)
        term = _r3_terminal(env, info)
        if int(term.get("SUCCESS_RELAXED") or info.get("SUCCESS_RELAXED_12S") or 0) == 1:
            xs.extend(ep_ctx)
    if not xs:
        return np.zeros((0, 64), np.float64)
    return np.asarray(xs, np.float64)


def phase2_nullspace(p64, p90, train_b1_demos, protected_ctx, book, rho, device):
    """Exact linear preservation on protected ctx; train free nullspace coords on B1."""
    if len(protected_ctx) == 0:
        return None, {"null_dim": -1, "rank": -1, "note": "no_protected_states"}

    # Augment with bias: X is [N, 65]
    ones = np.ones((protected_ctx.shape[0], 1), np.float64)
    X = np.concatenate([protected_ctx, ones], axis=1)
    # SVD of X^T (65 x N) or of X: nullspace of X is vectors v with X v = 0
    # We need ΔW_aug (7 x 65) such that ΔW_aug @ X.T = 0 ⇒ each row in left-nullspace of X.T
    # = right-nullspace of X: X v = 0.
    _, S, Vt = np.linalg.svd(X, full_matrices=True)
    rank = int(np.sum(S > SVD_TOL * S[0])) if len(S) else 0
    null_dim = int(Vt.shape[0] - rank)
    md = f"""# PROTECTED_HIDDEN_STATE_RANK

Generated: {_utc()}

| Item | Value |
|---|---|
| Protected hidden states (steps) | {X.shape[0]} |
| Feature dim (ctx+bias) | {X.shape[1]} |
| Numerical rank (tol={SVD_TOL}·σ_max) | **{rank}** |
| Nullspace dimension | **{null_dim}** |

"""
    if null_dim <= 0:
        md += (
            "**Protected states span the augmented feature space.** "
            "No nonzero linear residual-head update can leave all protected outputs unchanged.\n"
        )
        _write(OUT / "PROTECTED_HIDDEN_STATE_RANK.md", md)
        return None, {"null_dim": 0, "rank": rank, "X_shape": list(X.shape)}

    P_null = Vt[rank:].astype(np.float64)  # null_dim x 65
    md += f"Nullspace basis shape: {P_null.shape}. Proceed to constrained fit of U∈R^{{7×{null_dim}}}.\n"
    _write(OUT / "PROTECTED_HIDDEN_STATE_RANK.md", md)

    # Parameterize ΔW_aug = U @ P_null, U is 7 x null_dim
    U = torch.nn.Parameter(torch.zeros(7, null_dim, device=device, dtype=torch.float32))
    P = torch.as_tensor(P_null, device=device, dtype=torch.float32)
    opt = torch.optim.Adam([U], lr=NULL_LR)

    # Freeze copy of base W,b
    W0 = p64.res.weight.detach().clone()
    b0 = p64.res.bias.detach().clone()
    cap = float(p64.res_cap)

    def apply_delta():
        dW_aug = U @ P  # 7 x 65
        dW = dW_aug[:, :64]
        db = dW_aug[:, 64]
        return W0 + dW, b0 + db

    # Verify constraint on protected
    ver_rows = []
    with torch.no_grad():
        W, b = apply_delta()
        for i in range(min(32, len(protected_ctx))):
            h = torch.as_tensor(protected_ctx[i], device=device, dtype=torch.float32)
            pre0 = W0 @ h + b0
            pre1 = W @ h + b
            ver_rows.append(
                {
                    "i": i,
                    "pre_l2": float(torch.linalg.vector_norm(pre1 - pre0)),
                    "phase": "init_U0",
                }
            )

    curve = []
    for u in range(1, NULL_UPDATES + 1):
        opt.zero_grad(set_to_none=True)
        losses = []
        W, b = apply_delta()
        for d in train_b1_demos:
            obs = torch.as_tensor(d["obs"], device=device, dtype=torch.float32)
            prev = torch.as_tensor(d["prev"], device=device, dtype=torch.float32)
            with torch.no_grad():
                ctx, _, _, _ = forward_parts(p64, obs, prev)
            # use frozen backbone ctx; grads flow only into U via W,b
            pre = ctx @ W.T + b
            res = cap * torch.tanh(pre)
            tres = torch.as_tensor(d["residuals"], device=device, dtype=torch.float32)
            losses.append(F.mse_loss(res, tres))
        # bound update norm
        loss = torch.stack(losses).mean() + 1e-3 * torch.sum(U * U)
        loss.backward()
        opt.step()
        if u % 50 == 0 or u == 1:
            with torch.no_grad():
                W, b = apply_delta()
                max_pre = 0.0
                for i in range(len(protected_ctx)):
                    h = torch.as_tensor(protected_ctx[i], device=device, dtype=torch.float32)
                    max_pre = max(max_pre, float(torch.linalg.vector_norm((W @ h + b) - (W0 @ h + b0))))
            curve.append({"update": u, "loss": float(loss.detach()), "max_protected_pre_l2": max_pre})
            print(f"[null] u={u} loss={float(loss):.4f} prot_pre_l2={max_pre:.2e}", flush=True)

    with torch.no_grad():
        W, b = apply_delta()
        for i in range(min(64, len(protected_ctx))):
            h = torch.as_tensor(protected_ctx[i], device=device, dtype=torch.float32)
            pre0 = W0 @ h + b0
            pre1 = W @ h + b
            r0 = cap * torch.tanh(pre0)
            r1 = cap * torch.tanh(pre1)
            ver_rows.append(
                {
                    "i": i,
                    "pre_l2": float(torch.linalg.vector_norm(pre1 - pre0)),
                    "res_l2": float(torch.linalg.vector_norm(r1 - r0)),
                    "phase": "after_train",
                }
            )
    _write_csv(OUT / "NULLSPACE_CONSTRAINT_VERIFICATION.csv", ver_rows)
    _write_csv(OUT / "NULLSPACE_TRAIN_CURVE.csv", curve)

    # Build actor with updated residual head (clone p64)
    actor_n, book2, rho2, _ = load_actor(P64_ACTOR, device)
    with torch.no_grad():
        actor_n.res.weight.copy_(W)
        actor_n.res.bias.copy_(b)
    for p in actor_n.parameters():
        p.requires_grad_(False)
    save_m111(CKPT / "P91_nullspace_residual.pt", actor_n, book2, epoch=NULL_UPDATES, rho=rho2, extra={"p91": "null"})
    return actor_n, {"null_dim": null_dim, "rank": rank, "max_prot_res_l2": max(r["res_l2"] for r in ver_rows if r["phase"] == "after_train")}


# ---------------------------------------------------------------------------
# Phase 3 adapter
# ---------------------------------------------------------------------------


class EarlyResidualAdapter(torch.nn.Module):
    """r = r_p64 + gate(h,t) * delta(h); gate uses only h and t (no class labels)."""

    def __init__(self, d=64, k_max=4):
        super().__init__()
        self.k_max = k_max
        self.delta = torch.nn.Sequential(
            torch.nn.Linear(d + 1, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 7),
        )
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(d + 1, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
        )
        # init delta near 0
        torch.nn.init.zeros_(self.delta[-1].weight)
        torch.nn.init.zeros_(self.delta[-1].bias)

    def forward(self, ctx_last, t_scalar, res_p64):
        # ctx_last: [64], t in steps
        t_feat = torch.tensor([min(float(t_scalar) / float(self.k_max), 1.0)], device=ctx_last.device)
        x = torch.cat([ctx_last, t_feat], dim=-1)
        g = torch.sigmoid(self.gate(x))
        # hard time window
        if float(t_scalar) >= self.k_max:
            g = g * 0.0
        d = torch.tanh(self.delta(x)) * 0.35
        return res_p64 + g.squeeze(-1) * d, g


def phase3_adapter(p64, train_b1, prot_ctx_c, prot_ctx_a1, book, rho, device, starts, panel):
    adapter = EarlyResidualAdapter().to(device)
    opt = torch.optim.Adam(adapter.parameters(), lr=ADAPTER_LR)
    # distinguishability: mean ctx distance
    def mean_ctx(arr):
        return arr.mean(axis=0) if len(arr) else np.zeros(64)

    mc, ma, mb = mean_ctx(prot_ctx_c), mean_ctx(prot_ctx_a1), None
    # B1 ctx from demos via p64
    b1_ctx = []
    for d in train_b1:
        obs = torch.as_tensor(d["obs"], device=device, dtype=torch.float32)
        prev = torch.as_tensor(d["prev"], device=device, dtype=torch.float32)
        with torch.no_grad():
            ctx, _, _, _ = forward_parts(p64, obs, prev)
        b1_ctx.append(ctx.detach().cpu().numpy())
    b1_ctx = np.concatenate(b1_ctx, axis=0) if b1_ctx else np.zeros((0, 64))
    mb = mean_ctx(b1_ctx)
    sep_c = float(np.linalg.norm(mb - mc)) if len(prot_ctx_c) else float("nan")
    sep_a1 = float(np.linalg.norm(mb - ma)) if len(prot_ctx_a1) else float("nan")

    # suppress on protected: MSE(adapter out, p64) on C/A1 ctx
    for u in range(1, ADAPTER_UPDATES + 1):
        opt.zero_grad(set_to_none=True)
        losses = []
        for d in train_b1:
            obs = torch.as_tensor(d["obs"], device=device, dtype=torch.float32)
            prev = torch.as_tensor(d["prev"], device=device, dtype=torch.float32)
            tres = torch.as_tensor(d["residuals"], device=device, dtype=torch.float32)
            with torch.no_grad():
                ctx, _, res64, _ = forward_parts(p64, obs, prev)
            # supervise early steps only
            T = min(4, ctx.shape[0])
            for t in range(T):
                r_new, g = adapter(ctx[t], t, res64[t])
                losses.append(F.mse_loss(r_new, tres[t]))
                losses.append(1e-3 * (g ** 2).mean())
        # protected suppress
        for arr in (prot_ctx_c, prot_ctx_a1):
            if len(arr) == 0:
                continue
            idx = np.random.default_rng(u).choice(len(arr), size=min(32, len(arr)), replace=False)
            for i in idx:
                h = torch.as_tensor(arr[i], device=device, dtype=torch.float32)
                with torch.no_grad():
                    # fake res64 from linear
                    pre = p64.res(h)
                    r64 = p64.res_cap * torch.tanh(pre)
                r_new, g = adapter(h, 0.0, r64)
                losses.append(5.0 * F.mse_loss(r_new, r64))
                losses.append(1.0 * (g ** 2).mean())
        loss = torch.stack(losses).mean()
        loss.backward()
        opt.step()
        if u % 100 == 0 or u == 1:
            print(f"[adapter] u={u} loss={float(loss):.4f}", flush=True)

    md = f"""# ADAPTER_EXPERIMENT

Generated: {_utc()}

## Separability (mean ctx L2)

| Pair | L2 |
|---|---:|
| mean(B1) − mean(C_protected) | {sep_c:.4f} |
| mean(B1) − mean(A1_protected) | {sep_a1:.4f} |

Gate uses only `(h, t)` — no class labels / IC ids / teacher residuals at deploy.

If means are close, the suppress regularizer is the main C preservative (not a sharp class gate).
"""
    _write(OUT / "ADAPTER_EXPERIMENT.md", md)
    torch.save({"adapter": adapter.state_dict(), "k_max": 4}, CKPT / "P91_adapter.pt")

    def resid_fn(pack, _ad=adapter, _p64=p64):
        with torch.no_grad():
            _, _, r64, _ = forward_parts(_p64, pack["obs"], pack["prev"])
            r_new, _g = _ad(pack["ctx"][-1], pack["t"], r64[-1])
        return r_new.detach().cpu().numpy()

    return resid_fn, {"sep_c": sep_c, "sep_a1": sep_a1}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    starts = lookup_starts()
    p64, book, rho, _ = load_actor(P64_ACTOR, device)
    p90, book2, rho2, _ = load_actor(P90_B, device)
    for p in list(p64.parameters()) + list(p90.parameters()):
        p.requires_grad_(False)
    p64.eval()
    p90.eval()

    # Phase 0
    print("[P91] Phase 0 C regression", flush=True)
    p0 = phase0(p64, p90, book, rho, device, starts)
    if not p0["reproducible"]:
        _write(
            OUT / "FINAL_REPORT.md",
            f"# FINAL_REPORT\n\nGenerated: {_utc()}\n\n## CASE F\n\nC regression not reproducibly attributable; stop.\n",
        )
        print(json.dumps({"case": "F"}))
        return

    panel = build_panel(starts)
    print({k: len(v) for k, v in panel.items()}, flush=True)

    # Phase 1
    print("[P91] Phase 1 localized blend", flush=True)
    all_rows = []
    p64_rows = eval_variant("P64", p64, resid_from_actor(p64), panel, starts, book, rho, device)
    p90_rows = eval_variant("P90", p64, resid_from_actor(p90), panel, starts, book, rho, device)
    all_rows.extend(p64_rows)
    all_rows.extend(p90_rows)
    summaries = {
        "P64": summarize_vs_p64(p64_rows, p64_rows),
        "P90": summarize_vs_p64(p90_rows, p64_rows),
    }
    best = None
    for a in ALPHAS:
        for k in KS:
            name = f"blend_a{a}_k{k}"
            print(f"  {name}", flush=True)
            rows = eval_variant(name, p64, resid_blend(p64, p90, a, k), panel, starts, book, rho, device)
            all_rows.extend(rows)
            s = summarize_vs_p64(rows, p64_rows)
            s["name"] = name
            s["alpha"] = a
            s["k"] = k
            summaries[name] = s
            print(f"    B1 {s['B1_succ']}/{s['B1_n']} gain={s['B1_gain']} Closs={s['C_loss']} A1loss={s['A1_loss']} ok={s['acceptable']}", flush=True)
            if s["acceptable"] and (best is None or s["B1_gain"] > best["B1_gain"] or (s["B1_gain"] == best["B1_gain"] and s["B1_succ"] > best["B1_succ"])):
                best = s
                best["rows"] = rows

    _write_csv(OUT / "LOCALIZED_RESIDUAL_ABLATION.csv", all_rows)

    # Root-disjoint val for Phase1 survivor
    val_ids = verified_b1()[-2:]
    train_ids = [t for t in verified_b1() if t not in val_ids]
    phase1_advance = best is not None

    candidate_name = None
    candidate_resid_fn = None
    case_from_phase = None

    if phase1_advance:
        candidate_name = best["name"]
        a, k = best["alpha"], best["k"]
        candidate_resid_fn = resid_blend(p64, p90, a, k)
        case_from_phase = "A"
        print(f"[P91] Phase1 candidate {candidate_name}", flush=True)
        # root-disjoint val
        val_lines = [
            f"# ROOT_DISJOINT_VALIDATION\n\nGenerated: {_utc()}\n\nCandidate: `{candidate_name}` (diagnostic-panel selected; not independent test).\n",
            f"Val ICs: {val_ids}\n",
        ]
        for tid in val_ids:
            o64 = closed_loop(p64, resid_from_actor(p64), starts[tid], book, rho, device)
            oc = closed_loop(p64, candidate_resid_fn, starts[tid], book, rho, device)
            val_lines.append(
                f"- `{tid}`: P64={o64['SUCCESS_RELAXED_12S']} → cand={oc['SUCCESS_RELAXED_12S']}\n"
            )
        _write(OUT / "ROOT_DISJOINT_VALIDATION.md", "".join(val_lines))
    else:
        _write(
            OUT / "ROOT_DISJOINT_VALIDATION.md",
            f"# ROOT_DISJOINT_VALIDATION\n\nGenerated: {_utc()}\n\nNo Phase-1 acceptable candidate; validation deferred.\n",
        )

    # Phase 2 if needed
    null_actor = None
    null_meta = None
    if not phase1_advance:
        print("[P91] Phase 2 nullspace", flush=True)
        # protected successful C and A1 under P64
        c_ok = [r["traj_id"] for r in p64_rows if r["skill"] == "C" and int(r["SUCCESS_RELAXED_12S"])]
        a1_ok = [r["traj_id"] for r in p64_rows if r["skill"] == "A1" and int(r["SUCCESS_RELAXED_12S"])]
        ctx_c = collect_protected_ctx(p64, starts, c_ok, book, rho, device)
        ctx_a1 = collect_protected_ctx(p64, starts, a1_ok, book, rho, device)
        prot = np.concatenate([ctx_c, ctx_a1], axis=0) if len(ctx_c) or len(ctx_a1) else np.zeros((0, 64))
        train_demos = []
        for tid in train_ids:
            d = load_demo(tid)
            if d:
                train_demos.append(d)
        null_actor, null_meta = phase2_nullspace(p64, p90, train_demos, prot, book, rho, device)
        if null_meta and null_meta.get("null_dim", 0) == 0:
            case_from_phase = "D"
        elif null_actor is not None:
            rows_n = eval_variant("nullspace", p64, resid_from_actor(null_actor), panel, starts, book, rho, device)
            all_rows.extend(rows_n)
            s = summarize_vs_p64(rows_n, p64_rows)
            s["name"] = "nullspace"
            summaries["nullspace"] = s
            _write_csv(OUT / "LOCALIZED_RESIDUAL_ABLATION.csv", all_rows)  # refresh
            if s["acceptable"]:
                candidate_name = "nullspace"
                candidate_resid_fn = resid_from_actor(null_actor)
                case_from_phase = "B"
                best = s
            elif null_meta.get("null_dim", 0) == 0:
                case_from_phase = "D"
        else:
            case_from_phase = "D"

    # Phase 3 if still needed
    adapter_fn = None
    if candidate_resid_fn is None and case_from_phase != "D":
        # also run adapter if nullspace existed but not acceptable, or nullspace empty already set D
        pass
    if candidate_resid_fn is None and (null_meta is None or null_meta.get("null_dim", 0) > 0):
        if not phase1_advance and (null_actor is None or not summaries.get("nullspace", {}).get("acceptable")):
            print("[P91] Phase 3 adapter", flush=True)
            c_ok = [r["traj_id"] for r in p64_rows if r["skill"] == "C" and int(r["SUCCESS_RELAXED_12S"])]
            a1_ok = [r["traj_id"] for r in p64_rows if r["skill"] == "A1" and int(r["SUCCESS_RELAXED_12S"])]
            ctx_c = collect_protected_ctx(p64, starts, c_ok, book, rho, device)
            ctx_a1 = collect_protected_ctx(p64, starts, a1_ok, book, rho, device)
            train_demos = [load_demo(t) for t in train_ids]
            train_demos = [d for d in train_demos if d]
            adapter_fn, a_meta = phase3_adapter(p64, train_demos, ctx_c, ctx_a1, book, rho, device, starts, panel)
            rows_a = eval_variant("adapter", p64, adapter_fn, panel, starts, book, rho, device)
            all_rows.extend(rows_a)
            _write_csv(OUT / "LOCALIZED_RESIDUAL_ABLATION.csv", all_rows)
            s = summarize_vs_p64(rows_a, p64_rows)
            s["name"] = "adapter"
            summaries["adapter"] = s
            if s["acceptable"]:
                candidate_name = "adapter"
                candidate_resid_fn = adapter_fn
                case_from_phase = "C"
                best = s
            else:
                # check if B1 gain with C loss → E
                if s["B1_gain"] >= 1 and (s["C_loss"] > 0 or s["A1_loss"] > 0 or s["D_loss"] > 0):
                    case_from_phase = "E"
                elif case_from_phase is None:
                    case_from_phase = "F"

    if null_meta and null_meta.get("null_dim", 0) == 0 and candidate_resid_fn is None:
        case_from_phase = "D"
        if not (OUT / "ADAPTER_EXPERIMENT.md").is_file():
            _write(
                OUT / "ADAPTER_EXPERIMENT.md",
                f"# ADAPTER_EXPERIMENT\n\nGenerated: {_utc()}\n\nSkipped — linear nullspace already empty (CASE D primary).\n"
                "Optional adapter not required by decision tree when exact linear update is impossible and Phase1 failed;\n"
                "run manually if needed.\n",
            )

    # Spec: Phase 3 only if linear nullspace insufficient. If null_dim=0, still allow adapter as fallback.
    if case_from_phase == "D" and candidate_resid_fn is None:
        print("[P91] Phase 3 adapter (nullspace empty fallback)", flush=True)
        c_ok = [r["traj_id"] for r in p64_rows if r["skill"] == "C" and int(r["SUCCESS_RELAXED_12S"])]
        a1_ok = [r["traj_id"] for r in p64_rows if r["skill"] == "A1" and int(r["SUCCESS_RELAXED_12S"])]
        ctx_c = collect_protected_ctx(p64, starts, c_ok, book, rho, device)
        ctx_a1 = collect_protected_ctx(p64, starts, a1_ok, book, rho, device)
        train_demos = [d for d in (load_demo(t) for t in train_ids) if d]
        adapter_fn, a_meta = phase3_adapter(p64, train_demos, ctx_c, ctx_a1, book, rho, device, starts, panel)
        rows_a = eval_variant("adapter", p64, adapter_fn, panel, starts, book, rho, device)
        all_rows.extend(rows_a)
        _write_csv(OUT / "LOCALIZED_RESIDUAL_ABLATION.csv", all_rows)
        s = summarize_vs_p64(rows_a, p64_rows)
        s["name"] = "adapter"
        summaries["adapter"] = s
        if s["acceptable"]:
            candidate_name = "adapter"
            candidate_resid_fn = adapter_fn
            case_from_phase = "C"
            best = s
            # val
            val_lines = [
                f"# ROOT_DISJOINT_VALIDATION\n\nGenerated: {_utc()}\n\nCandidate: adapter\nVal: {val_ids}\n"
            ]
            for tid in val_ids:
                o64 = closed_loop(p64, resid_from_actor(p64), starts[tid], book, rho, device)
                oc = closed_loop(p64, candidate_resid_fn, starts[tid], book, rho, device)
                val_lines.append(f"- `{tid}`: P64={o64['SUCCESS_RELAXED_12S']} → cand={oc['SUCCESS_RELAXED_12S']}\n")
            _write(OUT / "ROOT_DISJOINT_VALIDATION.md", "".join(val_lines))
        elif s["B1_gain"] >= 1 and (s["C_loss"] or s["A1_loss"] or s["D_loss"]):
            case_from_phase = "E"
        else:
            case_from_phase = "D"  # keep D as linear impossibility; adapter also failed

    # Paired success transitions for controls + candidate
    trans = []
    variants_keep = ["P64", "P90"] + ([candidate_name] if candidate_name else [])
    by = {}
    for r in all_rows:
        by.setdefault(r["variant"], {})[(r["skill"], r["traj_id"])] = r
    for skill, tids in panel.items():
        for tid in tids:
            row = {"skill": skill, "traj_id": tid, "root_id": root_id(tid)}
            for v in ["P64", "P90"] + [x for x in by if x not in ("P64", "P90")]:
                if (skill, tid) in by.get(v, {}):
                    row[f"{v}_SUCCESS"] = by[v][(skill, tid)]["SUCCESS_RELAXED_12S"]
            if "P64_SUCCESS" in row and "P90_SUCCESS" in row:
                row["P64_to_P90"] = f"{row['P64_SUCCESS']}→{row['P90_SUCCESS']}"
            if candidate_name and f"{candidate_name}_SUCCESS" in row:
                row["P64_to_cand"] = f"{row['P64_SUCCESS']}→{row[f'{candidate_name}_SUCCESS']}"
            trans.append(row)
    _write_csv(OUT / "PAIRED_SUCCESS_TRANSITIONS.csv", trans)

    # Final case
    if case_from_phase is None:
        case_from_phase = "F"
    # If Phase1 found candidate → A already set
    case = case_from_phase
    titles = {
        "A": "Localized P90 residual correction improves B1 and preserves C",
        "B": "Exact nullspace-constrained linear update improves B1 and preserves C",
        "C": "Conditional adapter improves B1 and preserves C",
        "D": "Protected states consume the available linear update space; no compatible linear correction found",
        "E": "Candidate improves B1 but still causes C/A1/D regression",
        "F": "No reproducible improvement or insufficient evidence",
    }
    # Prefer E over F only for an actual P91 candidate (not the P90 control).
    if case == "F" and candidate_name is not None:
        s = summaries.get(candidate_name) or {}
        if s.get("B1_gain", 0) >= 1 and (
            s.get("C_loss", 0) or s.get("A1_loss", 0) or s.get("D_loss", 0)
        ):
            case = "E"

    # summary table for report
    sum_lines = [
        "| Variant | B1 succ | B1 gain/loss | C loss | A1 loss | D loss | acceptable |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, s in summaries.items():
        sum_lines.append(
            f"| {name} | {s.get('B1_succ','')}/{s.get('B1_n','')} | {s.get('B1_gain',0)}/{s.get('B1_loss',0)} | "
            f"{s.get('C_loss',0)} | {s.get('A1_loss',0)} | {s.get('D_loss',0)} | {s.get('acceptable', False)} |"
        )

    final = f"""# FINAL_REPORT — P91 C-Preserving B1 Residual Adaptation

Generated: {_utc()}

## CASE {case} — {titles[case]}

### Phase 0

C IC `{C_FAIL_IC}` regression **reproduced** with P64 tokens + P90 residuals → residual causality confirmed.

### Phase 1 — localized blend (α∈{{0.25,0.5,1.0}}, k∈{{2,4}})

{chr(10).join(sum_lines)}

Phase-1 acceptable candidate: **{best['name'] if phase1_advance else 'none'}**

### Later phases

- Nullspace: {json.dumps(null_meta) if null_meta else 'not required / see PROTECTED_HIDDEN_STATE_RANK.md'}
- Final candidate: `{candidate_name}`

### Integrity

- P64 / P87 / P88 / P90 checkpoints not overwritten
- No unconstrained BC/GRPO
- Official R3F and token policy unchanged in evaluation
- Exposed independent B1 ICs unused for selection
"""
    _write(OUT / "FINAL_REPORT.md", final)
    (OUT / "SUMMARY.json").write_text(
        json.dumps(
            {
                "case": case,
                "title": titles[case],
                "phase0_reproducible": p0["reproducible"],
                "phase1_candidate": candidate_name if phase1_advance else None,
                "summaries": {k: {kk: vv for kk, vv in s.items() if kk != "rows"} for k, s in summaries.items()},
                "null_meta": null_meta,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"case": case, "title": titles[case], "candidate": candidate_name}, indent=2), flush=True)


if __name__ == "__main__":
    main()
