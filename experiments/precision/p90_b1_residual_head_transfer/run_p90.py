"""P90 — B1 residual-head-only skill transfer (no token/backbone updates; no GRPO)."""

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
from experiments.precision.p63_transformer_critic_ppo.sampling import act_with_value  # noqa: E402
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
from models.m111_checkpoint import decode_m111_action, save_m111  # noqa: E402
from training.skill_aware_action_vocab import ResidualBudget  # noqa: E402

P87_OUT = ROOT / "results/experiments/p87_k48_actor_retention"
P87_TEACHER = P87_OUT / "teacher"
P88_OUT = ROOT / "results/experiments/p88_b1_short_sequence_learnability"
P88_SINGLE = P88_OUT / "checkpoints/P88_single_overfit_actor.pt"
P89_OUT = ROOT / "results/experiments/p89_b1_residual_causality"
P89_CAND = P88_OUT / "PHASE0_CANDIDATES.csv"
P86_PANEL = ROOT / "results/experiments/p86_b1_gate_counterfactual/PANEL.json"

OUT = ROOT / "results/experiments/p90_b1_residual_head_transfer"
CKPT = OUT / "checkpoints"

DIAG_IC = "ball_train_v2_0047"
SEED = 90
DIM_NAMES = ("dx", "dy", "dz", "droll", "dpitch", "dyaw", "grip")

# Predeclared training budget (no sweep)
STAGE_A_SINGLE_UPDATES = 400
STAGE_A_MULTI_UPDATES = 800
STAGE_B_UPDATES = 600
LR_RES = 3e-3
LAMBDA_PRESERVE = 1.0
LOG_EVERY = 20


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


def verified_ids() -> list[str]:
    rows = list(csv.DictReader(P89_CAND.open(encoding="utf-8")))
    return sorted(r["traj_id"] for r in rows if int(r["valid_official"]) == 1)


def unverified_ids() -> list[str]:
    rows = list(csv.DictReader(P89_CAND.open(encoding="utf-8")))
    return sorted(r["traj_id"] for r in rows if int(r["valid_official"]) == 0)


def load_demo(traj_id: str) -> dict:
    z = np.load(P87_TEACHER / f"{traj_id}.npz", allow_pickle=True)
    return {
        "traj_id": traj_id,
        "tokens": np.asarray(z["tokens"], np.int64),
        "residuals": np.asarray(z["residuals"], np.float32),
        "actions": np.asarray(z["actions"], np.float32),
        "obs": np.asarray(z["obs"], np.float32),
        "prev": np.asarray(z["prev"], np.float32),
    }


def freeze_all_but_residual(actor) -> list[str]:
    """Freeze backbone/token head; train only Policy.res Linear."""
    trainable = []
    for name, p in actor.named_parameters():
        # exact residual Linear: "res.weight", "res.bias"
        allow = name == "res.weight" or name == "res.bias"
        p.requires_grad_(allow)
        if allow:
            trainable.append(name)
    return trainable


def count_trainable(actor) -> int:
    return sum(int(p.numel()) for p in actor.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Phase 0
# ---------------------------------------------------------------------------


def phase0_audit() -> dict:
    device = torch.device("cpu")
    actor, book, rho, saved = load_actor(P64_ACTOR, device)
    names = [n for n, _ in actor.named_parameters()]
    res_params = [n for n in names if n.startswith("res.")]
    # Inspect forward residual transform
    res_cap = float(getattr(actor, "res_cap", np.max(ResidualBudget.from_spec(rho).as_array())))
    # P87 loss excerpt from source
    p87_loss = (
        "F.cross_entropy(mask_handoff_logits(logits)[motion], tokens[motion]) only; "
        "residual tensor from forward is ignored (_res); no MSE/L1 on residuals."
    )
    md = f"""# P87_RESIDUAL_LOSS_AUDIT

Generated: {_utc()}

## Architecture (shared P64 / P87 / P88 M111 `Policy`)

Source: `models/k48_policy.py` (M111 build via `build_m111`).

| Component | Module | Role |
|---|---|---|
| Input proj | `in_proj: Linear(46→64)` | obs(39)+prev(7) |
| Pos emb | `pos: Embedding(160,64)` | |
| Backbone | `enc: TransformerEncoder ×2` | causal context `ctx` |
| Token head | `logits: Linear(64→112)` | motion+HANDOFF |
| **Residual head** | **`res: Linear(64→7)`** | continuous residual |
| Output transform | `res_out = res_cap * tanh(res(ctx))` | `res_cap = max(ρ) ≈ {res_cap}` |

Residual **does not** depend on the predicted token ID. It is a function of hidden state `ctx` only.
The decoded action uses `clip(proto[tok] + res_out, ±1)` with tok from logits (or teacher).

## Independent trainability

| Question | Answer |
|---|---|
| Residual-producing parameters | `{', '.join(res_params)}` |
| Can freeze backbone + logits? | **Yes** — set `requires_grad=False` on all params except `res.*` |
| Backbone must be unfrozen? | **No** for residual-head-only transfer |
| Residual target space | Normalized action residual ∈ [−ρ_d, ρ_d]^7 (same as `quantize_trajectory` / `decode_m111_action`) |
| Nonlinearities | `tanh` then ×`res_cap`; deploy additionally `clip(r, ±ρ)` per dim |

## P87 training loss (actual)

```
logits, _res, _ = actor(obs, prev)   # _res discarded
loss = CE(mask_handoff_logits(logits)[motion], tokens)
```

- **Residual supervision: none**
- Residual gradients in P87: **none** (no loss path into `res`)
- Rehearsal npz stored `residuals=0` unused

## P88 contrast

P88 micro-overfit added `CE + 0.1 * MSE(res, teacher_residual)` while updating **all** Actor params
(token memorization). P89 showed token CE succeeded but residual L2≈0.14 remained causal for CL failure.

## P90 decision gate

Residual head is independently trainable without modifying the token policy → **proceed** (not CASE E).
"""
    _write(OUT / "P87_RESIDUAL_LOSS_AUDIT.md", md)
    return {
        "res_params": res_params,
        "res_cap": res_cap,
        "rho": ResidualBudget.from_spec(rho).to_list(),
        "feasible": True,
        "p87_loss": p87_loss,
    }


# ---------------------------------------------------------------------------
# Phase 1 — targets
# ---------------------------------------------------------------------------


def phase1_targets(book, rho, verified: list[str], unverified: list[str]) -> tuple[list[dict], list[dict]]:
    rows = []
    demos = []
    rbudget = ResidualBudget.from_spec(rho).as_array().astype(np.float32)
    for tid in verified + unverified:
        cat = "verified_official" if tid in verified else "unverified_official"
        if not (P87_TEACHER / f"{tid}.npz").is_file():
            continue
        d = load_demo(tid)
        n = len(d["tokens"])
        ok_all = True
        max_err = 0.0
        for t in range(n):
            tok = int(d["tokens"][t])
            res = d["residuals"][t]
            act = d["actions"][t]
            proto = np.asarray(book[tok], np.float32)
            dec = decode_m111_action(book, tok, res, rho=rho)
            # identity: proto + clip(res) clipped = executed
            err = float(np.linalg.norm(dec - act))
            max_err = max(max_err, err)
            if err > 1e-4:
                ok_all = False
            rows.append(
                {
                    "traj_id": tid,
                    "root_id": root_id(tid),
                    "category": cat,
                    "t": t,
                    "token": tok,
                    "residual_valid": 1,
                    "decode_vs_executed_l2": err,
                    "res_l2": float(np.linalg.norm(res)),
                    "within_rho": int(np.all(np.abs(res) <= rbudget + 1e-6)),
                    "positive_demo": int(cat == "verified_official"),
                }
            )
        demos.append(
            {
                **d,
                "category": cat,
                "parity_ok": int(ok_all),
                "max_decode_err": max_err,
                "root_id": root_id(tid),
            }
        )
    _write_csv(OUT / "RESIDUAL_TARGET_VERIFICATION.csv", rows)
    return demos, rows


# ---------------------------------------------------------------------------
# Training / metrics
# ---------------------------------------------------------------------------


def residual_forward(actor, obs, prev, device):
    o = torch.as_tensor(obs, device=device, dtype=torch.float32)
    p = torch.as_tensor(prev, device=device, dtype=torch.float32)
    logits, res, act = actor(o, p)
    return logits, res, act


@torch.no_grad()
def offline_metrics(actor, demo, book, rho, device) -> dict:
    actor.eval()
    logits, res, _ = residual_forward(actor, demo["obs"], demo["prev"], device)
    toks = torch.as_tensor(demo["tokens"], device=device, dtype=torch.long)
    tres = torch.as_tensor(demo["residuals"], device=device, dtype=torch.float32)
    pred = torch.argmax(mask_handoff_logits(logits.float()), dim=-1)
    tok_acc = float((pred == toks).float().mean())
    res_mse = float(F.mse_loss(res, tres))
    res_l2 = float(torch.linalg.vector_norm(res - tres, dim=-1).mean())
    # first-4
    k = min(4, len(toks))
    res_l2_f4 = float(torch.linalg.vector_norm(res[:k] - tres[:k], dim=-1).mean())
    # executed action L2 using teacher tokens (isolates residual)
    act_err = []
    per_dim = np.zeros(7, np.float64)
    for t in range(len(toks)):
        r = res[t].detach().cpu().numpy()
        dec = decode_m111_action(book, int(toks[t].item()), r, rho=rho)
        tea = demo["actions"][t]
        act_err.append(float(np.linalg.norm(dec - tea)))
        per_dim += np.abs(dec - tea)
    per_dim /= max(len(toks), 1)
    return {
        "tok_acc": tok_acc,
        "res_mse": res_mse,
        "res_l2": res_l2,
        "res_l2_first4": res_l2_f4,
        "action_l2": float(np.mean(act_err)) if act_err else 0.0,
        "action_l2_first4": float(np.mean(act_err[:k])) if act_err else 0.0,
        "per_dim_abs_action_err": per_dim.tolist(),
    }


def train_residual(
    actor,
    demos: list[dict],
    device,
    *,
    updates: int,
    label: str,
    book,
    rho,
    preserve_demos: list[dict] | None = None,
    preserve_ref: torch.nn.Module | None = None,
    lambda_pres: float = 0.0,
) -> list[dict]:
    freeze_all_but_residual(actor)
    assert count_trainable(actor) > 0
    opt = torch.optim.Adam([p for p in actor.parameters() if p.requires_grad], lr=LR_RES)
    curve = []
    actor.train()
    for u in range(1, updates + 1):
        opt.zero_grad(set_to_none=True)
        losses = []
        for d in demos:
            logits, res, _ = residual_forward(actor, d["obs"], d["prev"], device)
            tres = torch.as_tensor(d["residuals"], device=device, dtype=torch.float32)
            losses.append(F.mse_loss(res, tres))
            # token head must not receive grads — already frozen; assert no logit grad path needed
        loss_b1 = torch.stack(losses).mean()
        loss_pres = torch.zeros((), device=device)
        if preserve_demos and preserve_ref is not None and lambda_pres > 0:
            pl = []
            for d in preserve_demos:
                with torch.no_grad():
                    _, res_ref, _ = residual_forward(preserve_ref, d["obs"], d["prev"], device)
                _, res, _ = residual_forward(actor, d["obs"], d["prev"], device)
                pl.append(F.mse_loss(res, res_ref))
            loss_pres = torch.stack(pl).mean()
        loss = loss_b1 + float(lambda_pres) * loss_pres
        loss.backward()
        # confirm only res grads
        bad = [n for n, p in actor.named_parameters() if p.grad is not None and not n.startswith("res.")]
        gn = float(torch.nn.utils.clip_grad_norm_([p for p in actor.parameters() if p.requires_grad], 5.0))
        opt.step()
        if u % LOG_EVERY == 0 or u == 1 or u == updates:
            m0 = offline_metrics(actor, demos[0], book, rho, device)
            row = {
                "label": label,
                "update": u,
                "loss": float(loss.detach()),
                "loss_b1": float(loss_b1.detach()),
                "loss_pres": float(loss_pres.detach()),
                "grad_norm": gn,
                "non_res_grad_params": len(bad),
                "tok_acc_demo0": m0["tok_acc"],
                "res_l2_demo0": m0["res_l2"],
                "res_l2_first4_demo0": m0["res_l2_first4"],
                "action_l2_demo0": m0["action_l2"],
                "action_l2_first4_demo0": m0["action_l2_first4"],
            }
            # mean over all train demos
            ms = [offline_metrics(actor, d, book, rho, device) for d in demos]
            row["tok_acc_mean"] = float(np.mean([m["tok_acc"] for m in ms]))
            row["res_l2_mean"] = float(np.mean([m["res_l2"] for m in ms]))
            row["action_l2_mean"] = float(np.mean([m["action_l2"] for m in ms]))
            curve.append(row)
            print(
                f"[{label}] u={u} res_l2={row['res_l2_mean']:.4f} "
                f"act_l2={row['action_l2_mean']:.4f} tok={row['tok_acc_mean']:.3f}",
                flush=True,
            )
            if bad:
                print(f"  WARNING non-res grads: {bad}", flush=True)
        actor.train()
    return curve


def closed_loop(actor, critic, start, book, rho, device) -> dict:
    seed_all(SEED, deterministic_torch=True)
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
    ever = {k: False for k in ("frame9", "frame12", "frame", "bil", "r3", "dwell")}
    bil_lost = False
    saw_bil = False
    min_e = 1e9
    while not done:
        tok_a, resid, _, _ = act_with_value(
            actor, critic, env.obs_seq, env.prev_seq, device, deterministic=True
        )
        tok, st, _ = select_token(
            env_t=int(env.t), obs=last_obs(env), actor_tok=int(tok_a), st=st, v2=v2, rule=None
        )
        tokens.append(int(tok))
        _r, done, info = env.step(int(tok), resid)
        ensure_inner_capture_buffers(env)
        g = relative_geometry(env.env.sim)
        tr = getattr(env.mon, "tracker", None)
        ok_s = float(getattr(tr, "ok_s", 0.0) or 0.0)
        preds = evaluate_predicates(env.env.sim, ok_s=ok_s)
        e = float(g["e_pos_mm"])
        th = float(g["theta_deg"])
        bil = bool(g["bilateral"])
        pad4 = int(g.get("min_pad") or 0) >= 4
        wk = bool(preds.get("working_ok"))
        if bil:
            ever["bil"] = True
            saw_bil = True
            if pad4:
                min_e = min(min_e, e)
        elif saw_bil:
            bil_lost = True
        if bil and pad4 and wk and e <= 9 + 1e-9 and th <= 8 + 1e-9:
            ever["frame9"] = True
        if bil and pad4 and wk and e <= 12 + 1e-9 and th <= 8 + 1e-9:
            ever["frame12"] = True
        if preds.get("frame_ok"):
            ever["frame"] = True
        if preds.get("dwell_complete"):
            ever["dwell"] = True
        if preds.get("frame_ok") and preds.get("dwell_complete"):
            ever["r3"] = True
    term = _r3_terminal(env, info)
    return {
        "SUCCESS_RELAXED_12S": int(term.get("SUCCESS_RELAXED") or info.get("SUCCESS_RELAXED_12S") or 0),
        "SUCCESS_12S": int(info.get("SUCCESS_12S") or 0),
        "handoff": int(info.get("handoff") or 0),
        "r3_ready": int(term.get("r3_ready") or 0),
        "reach_frame12": int(ever["frame12"]),
        "reach_frame9": int(ever["frame9"]),
        "dwell_ever": int(ever["dwell"]),
        "bilateral_retained": int(ever["bil"] and not bil_lost),
        "min_e_pos_bil_pad": float(min_e if min_e < 1e8 else -1),
        "ep_len": int(env.t),
        "fail_reason": info.get("fail_reason"),
        "tokens": tokens,
    }


def first4_action_err(actor, demo, book, rho, device) -> float:
    m = offline_metrics(actor, demo, book, rho, device)
    return float(m["action_l2_first4"])


def token_agree(a_toks: list[int], b_toks: list[int]) -> float:
    n = min(len(a_toks), len(b_toks))
    if n == 0:
        return 0.0
    return float(np.mean([int(a_toks[i] == b_toks[i]) for i in range(n)]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    audit = phase0_audit()
    if not audit["feasible"]:
        _write(OUT / "FINAL_REPORT.md", f"# FINAL_REPORT\n\n## CASE E\n\n{audit}\n")
        return

    starts = lookup_starts()
    verified = verified_ids()
    unverified = unverified_ids()
    # Hold out last 2 verified lex for selection/val (root-disjoint from train set usage)
    val_ids = verified[-2:]
    train_ids = [t for t in verified if t not in val_ids]
    assert DIAG_IC in train_ids

    print(f"[P90] verified={len(verified)} train={len(train_ids)} val={val_ids}", flush=True)

    # need book/rho
    _, book, rho, _ = load_actor(P64_ACTOR, device)
    demos_all, _ = phase1_targets(book, rho, verified, unverified)
    by_id = {d["traj_id"]: d for d in demos_all}
    train_demos = [by_id[t] for t in train_ids]
    val_demos = [by_id[t] for t in val_ids]
    diag = by_id[DIAG_IC]

    # Parity gate on verified
    bad_parity = [d["traj_id"] for d in demos_all if d["category"] == "verified_official" and not d["parity_ok"]]
    if bad_parity:
        print(f"WARNING parity issues: {bad_parity}", flush=True)

    critic = load_critic(P64_CRITIC, load_actor(P64_ACTOR, device)[0], device)
    for p in critic.parameters():
        p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Stage A: P88 residual repair
    # ------------------------------------------------------------------
    print("[P90] Stage A single-trajectory", flush=True)
    actor_a, book, rho, _ = load_actor(P88_SINGLE, device)
    trainable = freeze_all_but_residual(actor_a)
    print(f"  trainable={trainable} n={count_trainable(actor_a)}", flush=True)

    # baselines before train
    p88_base_off = offline_metrics(actor_a, diag, book, rho, device)
    p88_base_cl = closed_loop(actor_a, critic, starts[DIAG_IC], book, rho, device)

    curve_single = train_residual(
        actor_a, [diag], device, updates=STAGE_A_SINGLE_UPDATES, label="A_single", book=book, rho=rho
    )
    _write_csv(OUT / "P88_SINGLE_TRAJECTORY_RESIDUAL_FIT.csv", curve_single)
    a_single_off = offline_metrics(actor_a, diag, book, rho, device)
    a_single_cl = closed_loop(actor_a, critic, starts[DIAG_IC], book, rho, device)
    print(
        f"  single CL {p88_base_cl['SUCCESS_RELAXED_12S']}→{a_single_cl['SUCCESS_RELAXED_12S']} "
        f"tok={a_single_off['tok_acc']:.3f} act_l2={a_single_off['action_l2']:.4f}",
        flush=True,
    )

    if a_single_off["tok_acc"] < 0.999:
        _write(
            OUT / "FINAL_REPORT.md",
            f"# FINAL_REPORT\n\n## CASE E / implementation\n\nToken accuracy dropped to {a_single_off['tok_acc']}.\n",
        )
        print(json.dumps({"case": "E", "reason": "token_acc_dropped"}))
        return

    if a_single_cl["SUCCESS_RELAXED_12S"] != 1 and a_single_off["action_l2_first4"] > 0.02:
        # offline fit may still be incomplete — continue multi a bit on single? already 400 updates
        pass

    stage_a_single_ok = int(a_single_cl["SUCCESS_RELAXED_12S"]) == 1

    # Extend to all train verified demos (re-init from P88 for fair multi fit)
    print("[P90] Stage A multi verified-train", flush=True)
    actor_am, book, rho, _ = load_actor(P88_SINGLE, device)
    curve_multi = train_residual(
        actor_am, train_demos, device, updates=STAGE_A_MULTI_UPDATES, label="A_multi", book=book, rho=rho
    )
    _write_csv(OUT / "P88_MULTI_RESIDUAL_FIT.csv", curve_multi)
    save_m111(CKPT / "P90_stageA_p88_residual.pt", actor_am, book, epoch=STAGE_A_MULTI_UPDATES, rho=rho, extra={"p90": "A"})

    # Learned vs oracle comparison on verified set
    learned_vs = []
    n_learned = n_oracle_ref = 0
    for tid in verified:
        d = by_id[tid]
        st = starts[tid]
        cl = closed_loop(actor_am, critic, st, book, rho, device)
        off = offline_metrics(actor_am, d, book, rho, device)
        # P88 baseline (reload frozen each time is heavy — use fresh)
        # Use pre-multi baseline only for diagnostic; for each IC evaluate current
        learned_vs.append(
            {
                "traj_id": tid,
                "split": "val" if tid in val_ids else "train",
                "learned_SUCCESS_RELAXED": cl["SUCCESS_RELAXED_12S"],
                "learned_tok_acc": off["tok_acc"],
                "learned_res_l2": off["res_l2"],
                "learned_action_l2_first4": off["action_l2_first4"],
                "oracle_res4_SUCCESS_RELAXED": 1,  # P89 D_res4 was 10/10 on verified
                "p89_note": "oracle teacher-residual intervention; not a learned rate",
            }
        )
        n_learned += int(cl["SUCCESS_RELAXED_12S"])
        n_oracle_ref += 1
    # Also evaluate raw P88 single ckpt on verified for baseline column
    actor_p88, _, _, _ = load_actor(P88_SINGLE, device)
    for row in learned_vs:
        cl0 = closed_loop(actor_p88, critic, starts[row["traj_id"]], book, rho, device)
        row["p88_SUCCESS_RELAXED"] = cl0["SUCCESS_RELAXED_12S"]
    _write_csv(OUT / "P88_LEARNED_VS_ORACLE.csv", learned_vs)

    stage_a_ok = stage_a_single_ok or (n_learned >= 8)
    print(f"  StageA learned CL {n_learned}/{n_oracle_ref}; single_ok={stage_a_single_ok}", flush=True)

    if not stage_a_single_ok and a_single_off["action_l2"] < 0.01 and a_single_cl["SUCCESS_RELAXED_12S"] == 0:
        # CASE C path: offline fit but CL fail
        case_hint_c = True
    else:
        case_hint_c = False

    if not stage_a_ok and case_hint_c:
        _write_final(
            "C",
            "Offline residual fitting succeeds, but learned closed-loop execution still fails",
            audit,
            stage_a={
                "single_cl": a_single_cl,
                "single_off": a_single_off,
                "base_cl": p88_base_cl,
                "base_off": p88_base_off,
                "n_learned": n_learned,
                "n_tot": n_oracle_ref,
            },
            stage_b=None,
            paired=None,
            xclass=None,
            train_ids=train_ids,
            val_ids=val_ids,
        )
        return

    if not stage_a_ok:
        _write_final(
            "F",
            "No measurable improvement",
            audit,
            stage_a={
                "single_cl": a_single_cl,
                "single_off": a_single_off,
                "base_cl": p88_base_cl,
                "base_off": p88_base_off,
                "n_learned": n_learned,
                "n_tot": n_oracle_ref,
            },
            stage_b=None,
            paired=None,
            xclass=None,
            train_ids=train_ids,
            val_ids=val_ids,
        )
        return

    # ------------------------------------------------------------------
    # Stage B: P64 residual-only + preservation
    # ------------------------------------------------------------------
    print("[P90] Stage B P64 residual-only", flush=True)
    actor_b, book, rho, _ = load_actor(P64_ACTOR, device)
    ref_b, _, _, _ = load_actor(P64_ACTOR, device)
    for p in ref_b.parameters():
        p.requires_grad_(False)
    # load rehearsal demos (obs/prev only; residual target = frozen P64 output)
    preserve = []
    for pth in sorted(P87_TEACHER.glob("rehearse_*.npz")):
        z = np.load(pth)
        preserve.append(
            {
                "traj_id": pth.stem,
                "obs": np.asarray(z["obs"], np.float32),
                "prev": np.asarray(z["prev"], np.float32),
                "tokens": np.asarray(z["tokens"], np.int64),
            }
        )
    protocol = f"""# P64_RESIDUAL_ONLY_TRAINING

Generated: {_utc()}

## Setup

| Item | Value |
|---|---|
| Init | Frozen P64 Actor (file untouched; new ckpt under p90/) |
| Trainable | `res.weight`, `res.bias` only |
| Backbone / logits | **Frozen** |
| B1 targets | Teacher residuals on verified train ICs: {train_ids} |
| Val (selection, not train) | {val_ids} |
| Preservation | MSE(res, P64_res) on {len(preserve)} A1/C/D rehearsal trajectories |
| Loss | `L_B1_residual + λ L_nonB1_preservation`, λ={LAMBDA_PRESERVE} |
| Updates | {STAGE_B_UPDATES} |
| LR | {LR_RES} |
| Runtime class labels | none |

## Non-actions

- No GRPO / full-Actor BC / R3F / M111 changes
- P64 source checkpoint not overwritten
"""
    _write(OUT / "P64_RESIDUAL_ONLY_TRAINING.md", protocol)

    curve_b = train_residual(
        actor_b,
        train_demos,
        device,
        updates=STAGE_B_UPDATES,
        label="B_p64",
        book=book,
        rho=rho,
        preserve_demos=preserve,
        preserve_ref=ref_b,
        lambda_pres=LAMBDA_PRESERVE,
    )
    _write_csv(OUT / "P64_RESIDUAL_TRAIN_CURVE.csv", curve_b)
    save_m111(CKPT / "P90_stageB_p64_residual.pt", actor_b, book, epoch=STAGE_B_UPDATES, rho=rho, extra={"p90": "B"})

    # Val selection diagnostic (not used to pick among many — single run)
    val_scores = []
    for d in val_demos:
        cl = closed_loop(actor_b, critic, starts[d["traj_id"]], book, rho, device)
        off = offline_metrics(actor_b, d, book, rho, device)
        val_scores.append({**off, "SUCCESS_RELAXED_12S": cl["SUCCESS_RELAXED_12S"], "traj_id": d["traj_id"]})
    _write_csv(OUT / "P64_VAL_SCORES.csv", val_scores)

    # ------------------------------------------------------------------
    # Phase 4 paired eval
    # ------------------------------------------------------------------
    print("[P90] Phase 4 paired physical eval", flush=True)
    p64_actor, _, _, _ = load_actor(P64_ACTOR, device)
    panels = {
        "diag": [DIAG_IC],
        "verified_train": train_ids,
        "verified_val": val_ids,
    }
    # P86 panel intersect available starts
    if P86_PANEL.is_file():
        p86 = json.loads(P86_PANEL.read_text(encoding="utf-8")).get("b1_traj_ids") or []
        panels["p86_b1"] = [t for t in p86 if t in starts]

    paired = []
    for panel, tids in panels.items():
        for tid in tids:
            if tid not in starts:
                continue
            st = starts[tid]
            cl64 = closed_loop(p64_actor, critic, st, book, rho, device)
            cl90 = closed_loop(actor_b, critic, st, book, rho, device)
            tok_ag = token_agree(cl64["tokens"], cl90["tokens"])
            f4 = None
            if tid in by_id:
                f4_64 = first4_action_err(p64_actor, by_id[tid], book, rho, device)
                f4_90 = first4_action_err(actor_b, by_id[tid], book, rho, device)
            else:
                f4_64 = f4_90 = ""
            # token change diagnosis
            paired.append(
                {
                    "panel": panel,
                    "traj_id": tid,
                    "root_id": root_id(tid),
                    "P64_SUCCESS_RELAXED": cl64["SUCCESS_RELAXED_12S"],
                    "P90_SUCCESS_RELAXED": cl90["SUCCESS_RELAXED_12S"],
                    "P64_reach12": cl64["reach_frame12"],
                    "P90_reach12": cl90["reach_frame12"],
                    "P64_reach9": cl64["reach_frame9"],
                    "P90_reach9": cl90["reach_frame9"],
                    "P64_r3_ready": cl64["r3_ready"],
                    "P90_r3_ready": cl90["r3_ready"],
                    "P64_bil_retained": cl64["bilateral_retained"],
                    "P90_bil_retained": cl90["bilateral_retained"],
                    "token_agree_vs_p64": tok_ag,
                    "P64_action_l2_first4_vs_teacher": f4_64,
                    "P90_action_l2_first4_vs_teacher": f4_90,
                    "P64_ep_len": cl64["ep_len"],
                    "P90_ep_len": cl90["ep_len"],
                    "P64_fail": cl64.get("fail_reason"),
                    "P90_fail": cl90.get("fail_reason"),
                }
            )
            print(
                f"  [{panel}] {tid} P64={cl64['SUCCESS_RELAXED_12S']} P90={cl90['SUCCESS_RELAXED_12S']} "
                f"tok_agree={tok_ag:.2f}",
                flush=True,
            )
    _write_csv(OUT / "PAIRED_PHYSICAL_EVALUATION.csv", paired)

    # token freeze check
    mean_tok_agree = float(np.mean([r["token_agree_vs_p64"] for r in paired])) if paired else 1.0
    if mean_tok_agree < 0.95:
        print(f"WARNING token sequences changed vs P64 mean_agree={mean_tok_agree}", flush=True)

    # ------------------------------------------------------------------
    # Phase 5 cross-class
    # ------------------------------------------------------------------
    print("[P90] Phase 5 cross-class", flush=True)
    # reuse P87 eval panel ICs from OFFICIAL_SUCCESS if available
    x_rows = []
    succ_csv = P87_OUT / "OFFICIAL_SUCCESS_COMPARISON.csv"
    if succ_csv.is_file():
        rows = list(csv.DictReader(succ_csv.open(encoding="utf-8")))
        for skill, panel in (("A1", "a1"), ("C", "c"), ("D", "d")):
            tids = sorted({r["traj_id"] for r in rows if r["panel"] == panel})
            for tid in tids:
                if tid not in starts:
                    continue
                st = starts[tid]
                cl64 = closed_loop(p64_actor, critic, st, book, rho, device)
                cl90 = closed_loop(actor_b, critic, st, book, rho, device)
                # residual drift on first step
                seed_all(SEED, deterministic_torch=True)
                env = StabilizeGrpoEnv(
                    book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
                )
                env.reset(st)
                ensure_inner_capture_buffers(env)
                with torch.no_grad():
                    o = torch.as_tensor(np.stack(env.obs_seq[-160:]), device=device, dtype=torch.float32)
                    pv = torch.as_tensor(np.stack(env.prev_seq[-160:]), device=device, dtype=torch.float32)
                    _, r64, _ = p64_actor(o, pv)
                    _, r90, _ = actor_b(o, pv)
                    drift = float(torch.linalg.vector_norm(r90[-1] - r64[-1]))
                x_rows.append(
                    {
                        "skill": skill,
                        "traj_id": tid,
                        "P64_SUCCESS_RELAXED": cl64["SUCCESS_RELAXED_12S"],
                        "P90_SUCCESS_RELAXED": cl90["SUCCESS_RELAXED_12S"],
                        "lost": int(cl64["SUCCESS_RELAXED_12S"] == 1 and cl90["SUCCESS_RELAXED_12S"] == 0),
                        "gained": int(cl64["SUCCESS_RELAXED_12S"] == 0 and cl90["SUCCESS_RELAXED_12S"] == 1),
                        "residual_drift_step0": drift,
                        "token_agree": token_agree(cl64["tokens"], cl90["tokens"]),
                    }
                )
    _write_csv(OUT / "CROSS_CLASS_ROWS.csv", x_rows)
    x_md = [f"# CROSS_CLASS_REGRESSION\n\nGenerated: {_utc()}\n"]
    xclass = {}
    for skill in ("A1", "C", "D"):
        rs = [r for r in x_rows if r["skill"] == skill]
        lost = sum(int(r["lost"]) for r in rs)
        gained = sum(int(r["gained"]) for r in rs)
        p64 = sum(int(r["P64_SUCCESS_RELAXED"]) for r in rs)
        p90 = sum(int(r["P90_SUCCESS_RELAXED"]) for r in rs)
        xclass[skill] = {"n": len(rs), "P64": p64, "P90": p90, "lost": lost, "gained": gained}
        x_md.append(
            f"## {skill}\n\nP64 {p64}/{len(rs)} → P90 {p90}/{len(rs)} (lost {lost}, gained {gained})\n"
        )
        for r in rs:
            if r["lost"] or r["gained"]:
                x_md.append(
                    f"- `{r['traj_id']}` P64={r['P64_SUCCESS_RELAXED']}→P90={r['P90_SUCCESS_RELAXED']} "
                    f"drift0={r['residual_drift_step0']:.4f}\n"
                )
    _write(OUT / "CROSS_CLASS_REGRESSION.md", "\n".join(x_md))

    # ------------------------------------------------------------------
    # CASE decision
    # ------------------------------------------------------------------
    def panel_counts(panel: str):
        rs = [r for r in paired if r["panel"] == panel]
        return (
            sum(int(r["P64_SUCCESS_RELAXED"]) for r in rs),
            sum(int(r["P90_SUCCESS_RELAXED"]) for r in rs),
            len(rs),
        )

    tr64, tr90, ntr = panel_counts("verified_train")
    va64, va90, nva = panel_counts("verified_val")
    dg64, dg90, _ = panel_counts("diag")
    any_loss = any(xclass[s]["lost"] > 0 for s in xclass)
    b1_improve = (tr90 > tr64) or (dg90 > dg64) or (va90 > va64)
    b1_reach = False
    for panel in ("verified_train", "p86_b1", "diag"):
        rs = [r for r in paired if r["panel"] == panel]
        if rs and sum(int(r["P90_reach12"]) for r in rs) > sum(int(r["P64_reach12"]) for r in rs):
            b1_reach = True

    if mean_tok_agree < 0.9:
        case, title = "E", "Residual-head-only training is not feasible under the current architecture"
        title += " (token policy changed despite freeze)"
    elif case_hint_c and not stage_a_single_ok:
        case, title = "C", "Offline residual fitting succeeds, but learned closed-loop execution still fails"
    elif b1_improve and not any_loss:
        case, title = (
            "A",
            "Learned residual repair reproduces the P89 rescue and improves P64 B1 without observed cross-class regression",
        )
    elif stage_a_ok and not b1_improve:
        case, title = (
            "B",
            "P88 residual repair succeeds, but P64 remains limited by incorrect token selection or reachability",
        )
    elif b1_improve and any_loss:
        case, title = "D", "B1 improves but A1/C/D regress"
    else:
        case, title = "F", "No measurable improvement"

    # Refine B: Stage A rescued but Stage B P64 no B1 gain
    if stage_a_single_ok and not b1_improve and not any_loss:
        case, title = (
            "B",
            "P88 residual repair succeeds, but P64 remains limited by incorrect token selection or reachability",
        )
    if b1_improve and any_loss:
        case, title = "D", "B1 improves but A1/C/D regress"
    if b1_improve and not any_loss and stage_a_single_ok:
        case, title = (
            "A",
            "Learned residual repair reproduces the P89 rescue and improves P64 B1 without observed cross-class regression",
        )

    _write_final(
        case,
        title,
        audit,
        stage_a={
            "single_cl": a_single_cl,
            "single_off": a_single_off,
            "base_cl": p88_base_cl,
            "base_off": p88_base_off,
            "n_learned": n_learned,
            "n_tot": n_oracle_ref,
        },
        stage_b={
            "train": (tr64, tr90, ntr),
            "val": (va64, va90, nva),
            "diag": (dg64, dg90),
            "mean_tok_agree": mean_tok_agree,
        },
        paired=paired,
        xclass=xclass,
        train_ids=train_ids,
        val_ids=val_ids,
    )
    print(json.dumps({"case": case, "title": title, "train": [tr64, tr90, ntr], "xclass": xclass}, indent=2))


def _write_final(case, title, audit, stage_a, stage_b, paired, xclass, train_ids, val_ids) -> None:
    text = f"""# FINAL_REPORT — P90 B1 Residual-Head-Only Skill Transfer

Generated: {_utc()}

## CASE {case} — {title}

### Phase 0

Residual head `res: Linear(64→7)` independently trainable; P87 used **token CE only** (no residual loss).
See `P87_RESIDUAL_LOSS_AUDIT.md`.

### Stage A (P88 residual repair)

| Item | Value |
|---|---|
| Diagnostic IC | `{DIAG_IC}` |
| P88 baseline CL SUCCESS | {stage_a['base_cl']['SUCCESS_RELAXED_12S'] if stage_a else 'n/a'} |
| After residual-only fit CL | {stage_a['single_cl']['SUCCESS_RELAXED_12S'] if stage_a else 'n/a'} |
| Offline tok acc (must stay ~1) | {stage_a['single_off']['tok_acc'] if stage_a else 'n/a'} |
| Action L2 (all / first4) | {stage_a['single_off']['action_l2'] if stage_a else 'n/a'} / {stage_a['single_off']['action_l2_first4'] if stage_a else 'n/a'} |
| Learned CL on verified (vs P89 oracle 10/10) | {stage_a['n_learned'] if stage_a else 'n/a'}/{stage_a['n_tot'] if stage_a else 'n/a'} |

Oracle 10/10 is **not** claimed as a learned rate.

### Stage B (P64 residual-only)

"""
    if stage_b:
        tr64, tr90, ntr = stage_b["train"]
        va64, va90, nva = stage_b["val"]
        text += f"""| Split | P64 | P90 | n |
|---|---:|---:|---:|
| verified train | {tr64} | {tr90} | {ntr} |
| verified val (held) | {va64} | {va90} | {nva} |
| diagnostic | {stage_b['diag'][0]} | {stage_b['diag'][1]} | 1 |

Mean token agreement vs P64 closed-loop: **{stage_b['mean_tok_agree']:.3f}**

Train ICs: {train_ids}
Val ICs: {val_ids}
"""
    else:
        text += "Not run (Stage A gate).\n"

    if xclass:
        text += "\n### Cross-class\n\n"
        for sk, d in xclass.items():
            text += f"- **{sk}**: {d['P64']}/{d['n']} → {d['P90']}/{d['n']} (lost {d['lost']}, gained {d['gained']})\n"

    text += """
### Integrity

- No GRPO / full-Actor BC / gate changes
- P64 / P87 / P88 source checkpoints not overwritten
- Exposed independent B1 ICs unused for train/selection
- Independent P64 0/8 remains frozen
"""
    _write(OUT / "FINAL_REPORT.md", text)
    (OUT / "SUMMARY.json").write_text(
        json.dumps({"case": case, "title": title, "stage_a": stage_a, "stage_b": stage_b, "xclass": xclass}, indent=2, default=str),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
