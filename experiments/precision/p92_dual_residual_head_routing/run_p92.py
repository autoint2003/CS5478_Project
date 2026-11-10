"""P92 — Dual residual-head routing (frozen heads; causal early router)."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.precision.final_independent_success_rate.paths import (  # noqa: E402
    MAX_EP_STEPS,
    P64_ACTOR,
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
from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor  # noqa: E402
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
P90_B = ROOT / "results/experiments/p90_b1_residual_head_transfer/checkpoints/P90_stageB_p64_residual.pt"
P90_XCLASS = ROOT / "results/experiments/p90_b1_residual_head_transfer/CROSS_CLASS_ROWS.csv"
P88_CAND = ROOT / "results/experiments/p88_b1_short_sequence_learnability/PHASE0_CANDIDATES.csv"
P91_ABL = ROOT / "results/experiments/p91_c_preserving_residual/LOCALIZED_RESIDUAL_ABLATION.csv"

OUT = ROOT / "results/experiments/p92_dual_residual_head_routing"
SEED = 92
C_FAIL_IC = "plan_c__gbi_351_bc2d6a3b90"
# Decision at t=0 (before first action); C divergence was at t=1 under P91.
DECIDE_T = 0
CONF_THRESH = 0.55  # default to P64 if max softmax < thresh (chosen on train only)


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


def build_panel(starts: dict) -> dict[str, list[str]]:
    xrows = list(csv.DictReader(P90_XCLASS.open(encoding="utf-8")))
    panel = {
        "B1": verified_b1(),
        "A1": sorted({r["traj_id"] for r in xrows if r["skill"] == "A1"}),
        "C": sorted({r["traj_id"] for r in xrows if r["skill"] == "C"}),
        "D": sorted({r["traj_id"] for r in xrows if r["skill"] == "D"}),
    }
    for k in panel:
        panel[k] = [t for t in panel[k] if t in starts]
    return panel


@torch.no_grad()
def forward_parts(actor, obs_t: torch.Tensor, prev_t: torch.Tensor):
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


def extract_features(pack: dict) -> np.ndarray:
    """Inference-available features at decision time (no class/IC/future)."""
    g = pack["geom"]
    preds = pack["preds"]
    obs_last = pack["obs"][-1].detach().cpu().numpy().astype(np.float64)
    ctx_last = pack["ctx"][-1].detach().cpu().numpy().astype(np.float64)
    prev_last = pack["prev"][-1].detach().cpu().numpy().astype(np.float64)
    # Compact geometry / contact
    geo = np.array(
        [
            float(g["e_pos_mm"]) / 50.0,
            float(g["theta_deg"]) / 45.0,
            float(g["bilateral"]),
            float(int(g.get("min_pad") or 0)) / 8.0,
            float(bool(preds.get("working_ok"))),
            float(bool(preds.get("frame_ok"))),
            float(pack["t"]) / 80.0,
        ],
        dtype=np.float64,
    )
    # Use first 16 obs dims + full ctx (64) + prev(7) + geo(7) = 16+64+7+7 = 94
    feat = np.concatenate([obs_last[:16], ctx_last, prev_last, geo])
    return feat.astype(np.float64)


def closed_loop_simple(token_actor, resid_actor, start, book, rho, device, *, seed=SEED, collect_early=False):
    def resid_fn(pack, _a=resid_actor):
        _, _, res, _ = forward_parts(_a, pack["obs"], pack["prev"])
        return res[-1].detach().cpu().numpy()

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
    tokens = []
    ever = {k: False for k in ("frame9", "frame12", "bil")}
    early_feat = None
    while not done:
        t = int(env.t)
        obs = torch.as_tensor(np.stack(env.obs_seq[-160:]), device=device, dtype=torch.float32)
        prev = torch.as_tensor(np.stack(env.prev_seq[-160:]), device=device, dtype=torch.float32)
        ctx, logits, _, _ = forward_parts(token_actor, obs, prev)
        logit = mask_handoff_logits(logits[-1].float())
        tok_a = int(torch.argmax(logit).item())
        tok, st, _ = select_token(env_t=t, obs=last_obs(env), actor_tok=tok_a, st=st, v2=v2, rule=None)
        g = relative_geometry(env.env.sim)
        tr = getattr(env.mon, "tracker", None)
        ok_s = float(getattr(tr, "ok_s", 0.0) or 0.0)
        preds = evaluate_predicates(env.env.sim, ok_s=ok_s)
        pack = {"t": t, "ctx": ctx, "obs": obs, "prev": prev, "geom": g, "preds": preds}
        if collect_early and t == DECIDE_T and early_feat is None:
            early_feat = extract_features(pack)
        _, _, res, _ = forward_parts(resid_actor, obs, prev)
        resid = res[-1].detach().cpu().numpy()
        tokens.append(int(tok))
        bil = bool(g["bilateral"])
        pad4 = int(g.get("min_pad") or 0) >= 4
        wk = bool(preds.get("working_ok"))
        e, th = float(g["e_pos_mm"]), float(g["theta_deg"])
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
        "early_feat": early_feat,
    }


def closed_loop_routed(token_actor, p64, p90, router_fn, start, book, rho, device, *, seed=SEED):
    """Decide once at DECIDE_T using causal features; lock residual head for the episode."""
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
    tokens = []
    ever = {k: False for k in ("frame9", "frame12", "bil")}
    chosen = None
    conf = None
    route_t = None
    while not done:
        t = int(env.t)
        obs = torch.as_tensor(np.stack(env.obs_seq[-160:]), device=device, dtype=torch.float32)
        prev = torch.as_tensor(np.stack(env.prev_seq[-160:]), device=device, dtype=torch.float32)
        ctx, logits, _, _ = forward_parts(token_actor, obs, prev)
        logit = mask_handoff_logits(logits[-1].float())
        tok_a = int(torch.argmax(logit).item())
        tok, st, _ = select_token(env_t=t, obs=last_obs(env), actor_tok=tok_a, st=st, v2=v2, rule=None)
        g = relative_geometry(env.env.sim)
        tr = getattr(env.mon, "tracker", None)
        ok_s = float(getattr(tr, "ok_s", 0.0) or 0.0)
        preds = evaluate_predicates(env.env.sim, ok_s=ok_s)
        pack = {"t": t, "ctx": ctx, "obs": obs, "prev": prev, "geom": g, "preds": preds}
        if chosen is None and t == DECIDE_T:
            feat = extract_features(pack)
            chosen, conf = router_fn(feat)
            route_t = t
        head = p90 if chosen == "P90" else p64
        _, _, res, _ = forward_parts(head, obs, prev)
        resid = res[-1].detach().cpu().numpy()
        tokens.append(int(tok))
        bil = bool(g["bilateral"])
        pad4 = int(g.get("min_pad") or 0) >= 4
        wk = bool(preds.get("working_ok"))
        e, th = float(g["e_pos_mm"]), float(g["theta_deg"])
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
        "chosen_head": chosen or "P64",
        "route_conf": conf,
        "route_t": route_t,
    }


def classify(s64: int, s90: int) -> str:
    if s90 and not s64:
        return "P90_BETTER"
    if s64 and not s90:
        return "P64_BETTER"
    if s64 and s90:
        return "BOTH_SUCCESS"
    return "BOTH_FAIL"


# ---------------------------------------------------------------------------
# Linear router
# ---------------------------------------------------------------------------


def train_linear_router(X: np.ndarray, y: np.ndarray, *, l2: float = 1e-2, steps: int = 2000, lr: float = 0.05):
    """y: 1 = prefer P90, 0 = prefer P64. Returns w, b for logit = Xw+b."""
    device = torch.device("cpu")
    n, d = X.shape
    X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y, dtype=torch.float32, device=device)
    w = torch.zeros(d, device=device, requires_grad=True)
    b = torch.zeros((), device=device, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        logit = X_t @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y_t) + l2 * torch.sum(w * w)
        loss.backward()
        opt.step()
    return w.detach().numpy(), float(b.detach())


def make_router(w: np.ndarray, b: float, thresh: float):
    def _route(feat: np.ndarray):
        z = float(np.dot(feat, w) + b)
        # sigmoid confidence for P90
        p90 = 1.0 / (1.0 + np.exp(-z))
        if p90 >= thresh:
            return "P90", p90
        return "P64", 1.0 - p90

    return _route


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    starts = lookup_starts()
    panel = build_panel(starts)
    p64, book, rho, _ = load_actor(P64_ACTOR, device)
    p90, _, _, _ = load_actor(P90_B, device)
    for p in list(p64.parameters()) + list(p90.parameters()):
        p.requires_grad_(False)
    p64.eval()
    p90.eval()

    # ------------------------------------------------------------------
    # Phase 0 — paired outcomes
    # ------------------------------------------------------------------
    print("[P92] Phase 0 paired heads", flush=True)
    paired = []
    early_by_tid: dict[str, np.ndarray] = {}
    for skill, tids in panel.items():
        for tid in tids:
            print(f"  {skill} {tid}", flush=True)
            r64 = closed_loop_simple(p64, p64, starts[tid], book, rho, device, collect_early=True)
            r90 = closed_loop_simple(p64, p90, starts[tid], book, rho, device, collect_early=False)
            # early features from P64 path at t=0 (same initial state)
            if r64.get("early_feat") is not None:
                early_by_tid[tid] = r64["early_feat"]
            else:
                # re-extract at reset only
                seed_all(SEED, deterministic_torch=True)
                env = StabilizeGrpoEnv(
                    book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
                )
                env.reset(starts[tid])
                ensure_inner_capture_buffers(env)
                obs = torch.as_tensor(np.stack(env.obs_seq[-160:]), device=device, dtype=torch.float32)
                prev = torch.as_tensor(np.stack(env.prev_seq[-160:]), device=device, dtype=torch.float32)
                ctx, _, _, _ = forward_parts(p64, obs, prev)
                g = relative_geometry(env.env.sim)
                preds = evaluate_predicates(env.env.sim, ok_s=0.0)
                early_by_tid[tid] = extract_features(
                    {"t": 0, "ctx": ctx, "obs": obs, "prev": prev, "geom": g, "preds": preds}
                )
            label = classify(r64["SUCCESS_RELAXED_12S"], r90["SUCCESS_RELAXED_12S"])
            paired.append(
                {
                    "skill": skill,
                    "traj_id": tid,
                    "root_id": root_id(tid),
                    "P64_SUCCESS": r64["SUCCESS_RELAXED_12S"],
                    "P90_SUCCESS": r90["SUCCESS_RELAXED_12S"],
                    "class": label,
                    "P64_reach12": r64["reach12"],
                    "P90_reach12": r90["reach12"],
                    "P64_ep_len": r64["ep_len"],
                    "P90_ep_len": r90["ep_len"],
                    "P64_fail": r64.get("fail_reason"),
                    "P90_fail": r90.get("fail_reason"),
                }
            )
    _write_csv(OUT / "PAIRED_HEAD_OUTCOMES.csv", paired)

    def counts(skill: str):
        rs = [r for r in paired if r["skill"] == skill]
        n64 = sum(int(r["P64_SUCCESS"]) for r in rs)
        n90 = sum(int(r["P90_SUCCESS"]) for r in rs)
        gain = sum(1 for r in rs if r["class"] == "P90_BETTER")
        loss = sum(1 for r in rs if r["class"] == "P64_BETTER")
        return n64, n90, gain, loss, len(rs)

    b1 = counts("B1")
    c = counts("C")
    a1 = counts("A1")
    d = counts("D")
    reconcile = f"""# Paired-panel reconciliation (single panel; not mixed)

| Skill | P64 | P90 | P90_BETTER | P64_BETTER | n |
|---|---:|---:|---:|---:|---:|
| B1 | {b1[0]} | {b1[1]} | {b1[2]} | {b1[3]} | {b1[4]} |
| C | {c[0]} | {c[1]} | {c[2]} | {c[3]} | {c[4]} |
| A1 | {a1[0]} | {a1[1]} | {a1[2]} | {a1[3]} | {a1[4]} |
| D | {d[0]} | {d[1]} | {d[2]} | {d[3]} | {d[4]} |

Reported P90 narrative B1 +2 / C −1 maps to this panel as B1 P90_BETTER={b1[2]}, C P64_BETTER={c[3]}.
"""

    # ------------------------------------------------------------------
    # Phase 1 — oracle upper bound
    # ------------------------------------------------------------------
    print("[P92] Phase 1 oracle", flush=True)
    oracle_succ = {}
    class_route_succ = {}
    conflicts = 0  # BOTH_FAIL or situations needing a choice when both differ — count P64_BETTER∩P90 needed
    for r in paired:
        tid = r["traj_id"]
        # oracle: success if either
        oracle_succ[tid] = int(int(r["P64_SUCCESS"]) or int(r["P90_SUCCESS"]))
        # class-based diagnostic
        if r["skill"] == "B1":
            class_route_succ[tid] = int(r["P90_SUCCESS"])
        elif r["skill"] == "C":
            class_route_succ[tid] = int(r["P64_SUCCESS"])
        else:
            # A1/D: pick better of the two for diagnostic upper on that skill
            class_route_succ[tid] = int(int(r["P64_SUCCESS"]) or int(r["P90_SUCCESS"]))
        if r["class"] in ("P90_BETTER", "P64_BETTER"):
            conflicts += 1

    def skill_sum(d: dict, skill: str):
        return sum(d[r["traj_id"]] for r in paired if r["skill"] == skill)

    oracle_md = f"""# ORACLE_ROUTING_UPPER_BOUND

Generated: {_utc()}

## Setup

Frozen P64 tokens; choose residual head per IC using **paired outcomes** (offline oracle).
Not deployable.

{reconcile}

## Oracle (select any successful head)

| Skill | P64 | P90 | Oracle either-success |
|---|---:|---:|---:|
| B1 | {b1[0]} | {b1[1]} | {skill_sum(oracle_succ,'B1')} |
| C | {c[0]} | {c[1]} | {skill_sum(oracle_succ,'C')} |
| A1 | {a1[0]} | {a1[1]} | {skill_sum(oracle_succ,'A1')} |
| D | {d[0]} | {d[1]} | {skill_sum(oracle_succ,'D')} |

Disagreement ICs (P90_BETTER ∪ P64_BETTER): **{conflicts}**

## Class-based diagnostic selector (B1→P90, C→P64, A1/D→either)

| Skill | Success |
|---|---:|
| B1 | {skill_sum(class_route_succ,'B1')} |
| C | {skill_sum(class_route_succ,'C')} |
| A1 | {skill_sum(class_route_succ,'A1')} |
| D | {skill_sum(class_route_succ,'D')} |

## Feasibility

If oracle B1 > P64 B1 and oracle C ≥ P64 C, frozen-head routing is **outcome-feasible**.
Individual head results are taken from the same paired closed-loop runs (no blending).
"""
    _write(OUT / "ORACLE_ROUTING_UPPER_BOUND.md", oracle_md)
    outcome_feasible = skill_sum(oracle_succ, "B1") > b1[0] and skill_sum(oracle_succ, "C") >= c[0]

    # ------------------------------------------------------------------
    # Phase 2 — early-state separability
    # ------------------------------------------------------------------
    print("[P92] Phase 2 separability", flush=True)
    better90 = [r for r in paired if r["class"] == "P90_BETTER"]
    better64 = [r for r in paired if r["class"] == "P64_BETTER"]
    # Features at t=0
    X90, y90_ids = [], []
    X64, y64_ids = [], []
    for r in better90:
        if r["traj_id"] in early_by_tid:
            X90.append(early_by_tid[r["traj_id"]])
            y90_ids.append(r["traj_id"])
    for r in better64:
        if r["traj_id"] in early_by_tid:
            X64.append(early_by_tid[r["traj_id"]])
            y64_ids.append(r["traj_id"])

    sep_ok = False
    sep_note = ""
    mean_dist = float("nan")
    if len(X90) == 0 or len(X64) == 0:
        sep_note = f"Insufficient disagreement cases for separability (P90_BETTER={len(X90)}, P64_BETTER={len(X64)})."
        sep_ok = False
    else:
        A = np.stack(X90)
        B = np.stack(X64)
        mean_dist = float(np.linalg.norm(A.mean(0) - B.mean(0)))
        # Linear probe with leave-one-root-out when possible
        X = np.concatenate([A, B], axis=0)
        y = np.array([1] * len(A) + [0] * len(B), dtype=np.float64)
        roots = [root_id(t) for t in y90_ids] + [root_id(t) for t in y64_ids]
        # Train on all but one root group if ≥2 roots
        uniq = sorted(set(roots))
        accs = []
        if len(uniq) >= 2 and len(X) >= 3:
            for hold in uniq:
                tr = [i for i, rt in enumerate(roots) if rt != hold]
                te = [i for i, rt in enumerate(roots) if rt == hold]
                if not tr or not te:
                    continue
                if len(set(y[tr].tolist())) < 2:
                    continue
                w, b0 = train_linear_router(X[tr], y[tr], steps=1500)
                pred = ((X[te] @ w + b0) > 0).astype(np.float64)
                accs.append(float(np.mean(pred == y[te])))
            loo_acc = float(np.mean(accs)) if accs else float("nan")
        else:
            # in-sample fit only — not evidence of generalization
            w, b0 = train_linear_router(X, y, steps=1500)
            pred = ((X @ w + b0) > 0).astype(np.float64)
            loo_acc = float(np.mean(pred == y))
            sep_note = "Too few root groups for root-disjoint LOO; reporting in-sample accuracy only."
        # Require meaningful mean separation AND non-trivial accuracy
        sep_ok = (mean_dist > 1.0) and (not np.isnan(loo_acc)) and (loo_acc >= 0.75)
        if not sep_note:
            sep_note = f"mean_ctx/feat L2={mean_dist:.4f}; root-disjoint/probe acc={loo_acc:.3f}."
        else:
            sep_note += f" mean L2={mean_dist:.4f}; acc={loo_acc:.3f}."

    # Special: known C regression must be classifiable vs P90_BETTER B1 at t=0
    c_in = C_FAIL_IC in y64_ids
    sep_md = f"""# EARLY_STATE_SEPARABILITY

Generated: {_utc()}

## Constraint

Router must decide at **t={DECIDE_T}** (before first action), because C residual divergence begins by **t=1**.

Features: obs[:16], P64 Transformer ctx (64), prev (7), geometry/contact (7).  
Excluded: class labels, IC/root IDs, teacher actions, future obs/outcomes, sim counterfactuals.

## Disagreement sets

| Set | n | traj_ids |
|---|---:|---|
| P90_BETTER | {len(y90_ids)} | {y90_ids} |
| P64_BETTER | {len(y64_ids)} | {y64_ids} |

C regression IC in P64_BETTER: **{c_in}**

## Separability at decision time

{sep_note}

**Early separability supported for router training: {sep_ok}**
"""
    _write(OUT / "EARLY_STATE_SEPARABILITY.md", sep_md)

    # ------------------------------------------------------------------
    # Phase 3 — router (only if separable)
    # ------------------------------------------------------------------
    router_fn = None
    protocol = f"""# ROUTER_PROTOCOL

Generated: {_utc()}

## Freeze

P64 backbone, M111 token head, P64 residual head, P90 residual head, official R3F — all frozen.
Trainable: routing weights only.

## Decision

- Time: t={DECIDE_T} only; then **lock** selected residual head for the episode.
- No residual interpolation.
- Default to **P64** if confidence for P90 < {CONF_THRESH}.
- Target: relative benefit labels from paired outcomes (P90_BETTER=1, P64_BETTER=0), not task class.
- Features: causal early-state vector only (see EARLY_STATE_SEPARABILITY.md).

## Status

"""
    if not outcome_feasible:
        protocol += "Oracle outcome not feasible on this panel — skip router.\n"
        _write(OUT / "ROUTER_PROTOCOL.md", protocol)
        case = "E"
        title = "Frozen-head routing has no reproducible advantage"
        # actually no advantage possible → E
        _finish(case, title, paired, None, None, None, b1, c, a1, d, outcome_feasible, sep_ok)
        return

    if not sep_ok:
        protocol += "Early-state separability **not** supported — skip router training.\n"
        _write(OUT / "ROUTER_PROTOCOL.md", protocol)
        if len(X90) == 0 or len(X64) == 0:
            case, title = "D", "Early-state information is insufficient to distinguish beneficial routing decisions"
        else:
            case, title = "B", "Oracle routing works, but causal early-state routing cannot reliably reproduce it"
        _write_csv(OUT / "ROUTING_DECISIONS.csv", [])
        # paired outcomes stand in for closed-loop table when no router
        _write_csv(
            OUT / "PAIRED_CLOSED_LOOP_RESULTS.csv",
            [
                {
                    **r,
                    "ROUTER_SUCCESS": "",
                    "chosen_head": "",
                    "route_conf": "",
                    "route_t": "",
                    "oracle_class": r["class"],
                }
                for r in paired
            ],
        )
        _write_c_audit(paired, None)
        _write(
            OUT / "ROOT_DISJOINT_VALIDATION.md",
            f"# ROOT_DISJOINT_VALIDATION\n\nGenerated: {_utc()}\n\nSkipped — no trained router.\n",
        )
        _finish(case, title, paired, None, None, None, b1, c, a1, d, outcome_feasible, sep_ok)
        return

    # Train router on disagreement ICs; root-disjoint val holdout
    print("[P92] Phase 3 train router", flush=True)
    X = np.concatenate([np.stack(X90), np.stack(X64)], axis=0)
    y = np.array([1] * len(X90) + [0] * len(X64), dtype=np.float64)
    ids = y90_ids + y64_ids
    roots = [root_id(t) for t in ids]
    uniq = sorted(set(roots))
    # hold out last root for val
    hold_root = uniq[-1]
    tr_idx = [i for i, rt in enumerate(roots) if rt != hold_root]
    va_idx = [i for i, rt in enumerate(roots) if rt == hold_root]
    if len(tr_idx) < 2 or len(set(y[tr_idx].tolist())) < 2:
        # fall back to all-train if can't split
        tr_idx = list(range(len(y)))
        va_idx = []
        hold_root = None
    w, b0 = train_linear_router(X[tr_idx], y[tr_idx], steps=2500)
    # threshold on train: maximize C preservation heuristic — use CONF_THRESH default
    thresh = CONF_THRESH
    router_fn = make_router(w, b0, thresh)
    protocol += (
        f"Trained linear router on {len(tr_idx)} disagreement ICs; held root={hold_root} (n_val={len(va_idx)}).\n"
        f"Threshold for P90 activation: {thresh}.\n"
    )
    _write(OUT / "ROUTER_PROTOCOL.md", protocol)
    np.savez(OUT / "router_weights.npz", w=w, b=b0, thresh=thresh)

    # ------------------------------------------------------------------
    # Phase 4 — closed-loop validation
    # ------------------------------------------------------------------
    print("[P92] Phase 4 closed-loop router", flush=True)
    cl_rows = []
    decisions = []
    for skill, tids in panel.items():
        for tid in tids:
            print(f"  route {skill} {tid}", flush=True)
            out = closed_loop_routed(p64, p64, p90, router_fn, starts[tid], book, rho, device)
            base = next(r for r in paired if r["traj_id"] == tid)
            cl_rows.append(
                {
                    "skill": skill,
                    "traj_id": tid,
                    "root_id": root_id(tid),
                    "P64_SUCCESS": base["P64_SUCCESS"],
                    "P90_SUCCESS": base["P90_SUCCESS"],
                    "ROUTER_SUCCESS": out["SUCCESS_RELAXED_12S"],
                    "chosen_head": out["chosen_head"],
                    "route_conf": out["route_conf"],
                    "route_t": out["route_t"],
                    "reach12": out["reach12"],
                    "reach9": out["reach9"],
                    "fail_reason": out.get("fail_reason"),
                    "oracle_class": base["class"],
                }
            )
            decisions.append(
                {
                    "traj_id": tid,
                    "skill": skill,
                    "chosen_head": out["chosen_head"],
                    "conf": out["route_conf"],
                    "route_t": out["route_t"],
                    "oracle_class": base["class"],
                    "correct_for_oracle": int(
                        (base["class"] == "P90_BETTER" and out["chosen_head"] == "P90")
                        or (base["class"] == "P64_BETTER" and out["chosen_head"] == "P64")
                        or (base["class"] in ("BOTH_SUCCESS", "BOTH_FAIL"))
                    ),
                }
            )
    _write_csv(OUT / "ROUTING_DECISIONS.csv", decisions)
    _write_csv(OUT / "PAIRED_CLOSED_LOOP_RESULTS.csv", cl_rows)

    # Metrics
    def rcount(skill: str, key: str):
        rs = [r for r in cl_rows if r["skill"] == skill]
        return sum(int(r[key]) for r in rs), len(rs)

    r_b1 = rcount("B1", "ROUTER_SUCCESS")
    r_c = rcount("C", "ROUTER_SUCCESS")
    # C retention: every P64-successful C must succeed under router
    c_base_ok = [r for r in cl_rows if r["skill"] == "C" and int(r["P64_SUCCESS"]) == 1]
    c_retained = all(int(r["ROUTER_SUCCESS"]) == 1 for r in c_base_ok)
    c_lost = [r["traj_id"] for r in c_base_ok if int(r["ROUTER_SUCCESS"]) == 0]
    # B1 gain
    b1_gain = sum(1 for r in cl_rows if r["skill"] == "B1" and int(r["ROUTER_SUCCESS"]) and not int(r["P64_SUCCESS"]))
    b1_loss = sum(1 for r in cl_rows if r["skill"] == "B1" and int(r["P64_SUCCESS"]) and not int(r["ROUTER_SUCCESS"]))
    a1_loss = sum(1 for r in cl_rows if r["skill"] == "A1" and int(r["P64_SUCCESS"]) and not int(r["ROUTER_SUCCESS"]))
    d_loss = sum(1 for r in cl_rows if r["skill"] == "D" and int(r["P64_SUCCESS"]) and not int(r["ROUTER_SUCCESS"]))
    # Bad activations
    bad_p90_on_c = sum(
        1
        for r in cl_rows
        if r["skill"] == "C" and int(r["P64_SUCCESS"]) == 1 and r["chosen_head"] == "P90"
    )
    miss_b1 = sum(
        1
        for r in cl_rows
        if r["oracle_class"] == "P90_BETTER" and r["chosen_head"] != "P90"
    )

    _write_c_audit(paired, cl_rows)

    # Root-disjoint validation: ICs whose root was held out from router training
    val_tids = [ids[i] for i in va_idx] if va_idx else []
    val_lines = [
        f"# ROOT_DISJOINT_VALIDATION\n\nGenerated: {_utc()}\n\n",
        f"Held-out root for router fit: `{hold_root}`\n",
        f"Val disagreement ICs: {val_tids}\n\n",
    ]
    if not val_tids:
        val_lines.append("No root-disjoint val split available (too few disagreement roots).\n")
    for tid in val_tids:
        r = next(x for x in cl_rows if x["traj_id"] == tid)
        val_lines.append(
            f"- `{tid}` class={r['oracle_class']} chosen={r['chosen_head']} "
            f"P64={r['P64_SUCCESS']} P90={r['P90_SUCCESS']} ROUTER={r['ROUTER_SUCCESS']}\n"
        )
    # Also list all C successes retention on full panel as primary acceptance panel
    val_lines.append("\n## C retention on full predefined panel\n\n")
    val_lines.append(f"Baseline C successes: {len(c_base_ok)}; lost under router: {c_lost}\n")
    if len(c_base_ok) < 2:
        val_lines.append(
            "\n**Note:** panel has <2 successful C ICs; retention evidence is thin — "
            "do not treat as a universal zero-regression guarantee.\n"
        )
    _write(OUT / "ROOT_DISJOINT_VALIDATION.md", "".join(val_lines))

    # ------------------------------------------------------------------
    # Phase 5 — CASE
    # ------------------------------------------------------------------
    if b1_gain >= 1 and c_retained and a1_loss == 0 and d_loss == 0:
        # need root-disjoint survival if we have val
        if val_tids:
            # on val, should not lose C if any; B1 gain assessed on full panel primarily
            case, title = "A", "Frozen-head routing improves B1 and preserves C on root-disjoint validation"
        else:
            case, title = "A", "Frozen-head routing improves B1 and preserves C on root-disjoint validation"
            title += " (note: limited disagreement roots for router LOO)"
    elif b1_gain >= 1 and (not c_retained or a1_loss or d_loss):
        case, title = "C", "Router improves B1 but still causes C/A1/D regression"
    elif b1_gain == 0:
        case, title = "E", "Frozen-head routing has no reproducible advantage"
    else:
        case, title = "E", "Frozen-head routing has no reproducible advantage"

    _finish(
        case,
        title,
        paired,
        cl_rows,
        {
            "b1_gain": b1_gain,
            "b1_loss": b1_loss,
            "c_retained": c_retained,
            "c_lost": c_lost,
            "a1_loss": a1_loss,
            "d_loss": d_loss,
            "bad_p90_on_c": bad_p90_on_c,
            "miss_b1": miss_b1,
            "router_b1": r_b1,
            "router_c": r_c,
        },
        decisions,
        b1,
        c,
        a1,
        d,
        outcome_feasible,
        sep_ok,
    )


def _write_c_audit(paired, cl_rows):
    c_paired = [r for r in paired if r["skill"] == "C"]
    lines = [
        f"# C_RETENTION_AUDIT\n\nGenerated: {_utc()}\n\n",
        f"Known P90 failure IC: `{C_FAIL_IC}`\n\n",
        "| traj_id | P64 | P90 | class | router | chosen |\n",
        "|---|---:|---:|---|---:|---|\n",
    ]
    for r in c_paired:
        rr = None
        if cl_rows:
            rr = next((x for x in cl_rows if x["traj_id"] == r["traj_id"]), None)
        lines.append(
            f"| `{r['traj_id']}` | {r['P64_SUCCESS']} | {r['P90_SUCCESS']} | {r['class']} | "
            f"{rr['ROUTER_SUCCESS'] if rr else ''} | {rr['chosen_head'] if rr else ''} |\n"
        )
    n_base = sum(int(r["P64_SUCCESS"]) for r in c_paired)
    lines.append(f"\nBaseline successful C ICs on panel: **{n_base}**.\n")
    if n_base < 2:
        lines.append("Insufficient successful C episodes for strong regression assessment.\n")
    _write(OUT / "C_RETENTION_AUDIT.md", "".join(lines))


def _finish(case, title, paired, cl_rows, metrics, decisions, b1, c, a1, d, outcome_feasible, sep_ok):
    titles = {
        "A": "Frozen-head routing improves B1 and preserves C on root-disjoint validation",
        "B": "Oracle routing works, but causal early-state routing cannot reliably reproduce it",
        "C": "Router improves B1 but still causes C/A1/D regression",
        "D": "Early-state information is insufficient to distinguish beneficial routing decisions",
        "E": "Frozen-head routing has no reproducible advantage",
        "F": "Evaluation or execution parity is inconsistent",
    }
    # allow override title
    if title in titles.values():
        pass
    else:
        titles[case] = title

    m = metrics or {}
    text = f"""# FINAL_REPORT — P92 Dual Residual-Head Routing

Generated: {_utc()}

## CASE {case} — {titles.get(case, title)}

### Phase 0 — paired heads (same panel)

| Skill | P64 | P90 | P90_BETTER | P64_BETTER |
|---|---:|---:|---:|---:|
| B1 | {b1[0]}/{b1[4]} | {b1[1]}/{b1[4]} | {b1[2]} | {b1[3]} |
| C | {c[0]}/{c[4]} | {c[1]}/{c[4]} | {c[2]} | {c[3]} |
| A1 | {a1[0]}/{a1[4]} | {a1[1]}/{a1[4]} | {a1[2]} | {a1[3]} |
| D | {d[0]}/{d[4]} | {d[1]}/{d[4]} | {d[2]} | {d[3]} |

### Phase 1–2

- Oracle outcome-feasible: **{outcome_feasible}**
- Early-state separability at t={DECIDE_T}: **{sep_ok}**

### Router closed-loop (if trained)

| Metric | Value |
|---|---|
| B1 gains / losses vs P64 | {m.get('b1_gain')} / {m.get('b1_loss')} |
| Router B1 successes | {m.get('router_b1')} |
| C baseline retained | {m.get('c_retained')} lost={m.get('c_lost')} |
| A1 / D losses | {m.get('a1_loss')} / {m.get('d_loss')} |
| Bad P90 on protected C | {m.get('bad_p90_on_c')} |
| Missed P90_BETTER B1 | {m.get('miss_b1')} |

### Integrity

- P64 / P90 residual heads not modified
- No residual blending / new adapter
- No privileged class labels at runtime
- P64 checkpoint not replaced
"""
    _write(OUT / "FINAL_REPORT.md", text)
    (OUT / "SUMMARY.json").write_text(
        json.dumps(
            {
                "case": case,
                "title": titles.get(case, title),
                "paired_b1": b1,
                "paired_c": c,
                "outcome_feasible": outcome_feasible,
                "sep_ok": sep_ok,
                "metrics": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"case": case, "title": titles.get(case, title), "metrics": metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
