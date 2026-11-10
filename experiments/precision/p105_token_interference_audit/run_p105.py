"""P105 — SAUP token interference and action-compatibility audit (diagnostics only).

Loads immutable P64 + P104 v2 Arms B/C. No training, no backbone unfreeze, no residual
modification, no PPO/GRPO, no sealed panel. Produces CF interventions that are
non-degenerate (forced tokens at matched physical states).
"""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.precision.final_independent_success_rate.paths import (  # noqa: E402
    MAX_EP_STEPS,
    P64_ACTOR,
    T_MAX,
)
from experiments.precision.p51_four_class_mixed_ppo.env_mixed import MixedStart  # noqa: E402
from experiments.precision.p51_four_class_mixed_ppo.ic_pools import root_id as rid0  # noqa: E402
from experiments.precision.p53_independent_grpo64.policy_sample import mask_handoff_logits  # noqa: E402
from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights  # noqa: E402
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset  # noqa: E402
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv  # noqa: E402
from experiments.precision.p61_reproducible_joint_eval.seeding import seed_all  # noqa: E402
from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor  # noqa: E402
from experiments.precision.p66_c_curriculum_ppo.expert_drive import ensure_inner_capture_buffers  # noqa: E402
from experiments.precision.p67_r3f_gate_diagnosis.gate_predicates import evaluate_predicates  # noqa: E402
from experiments.precision.p68_precontact_geometry.geometry import relative_geometry  # noqa: E402
from experiments.precision.p70_geometry_to_r3f.obs_proxy import last_obs  # noqa: E402
from experiments.precision.p73_first_contact_bilateral.controller import WrapState, select_token, v2_spec  # noqa: E402
from experiments.precision.p73_first_contact_bilateral.rollout import _r3_terminal  # noqa: E402
from experiments.precision.p86_b1_gate_counterfactual.gate_ablation import make_monitor  # noqa: E402
from models.m111_checkpoint import HANDOFF, decode_m111_action  # noqa: E402
from models.p104_policy import build_p104_from_p64  # noqa: E402
from training.skill_aware_action_vocab import ResidualBudget  # noqa: E402

OUT = ROOT / "results/experiments/p105_token_interference_audit"
P104 = ROOT / "results/experiments/p104_joint_token_residual_pilot"
P98 = ROOT / "results/experiments/p98_unified_b1_c_consolidation"
P87_TEACHER = ROOT / "results/experiments/p87_k48_actor_retention/teacher"
P97_NEW = ROOT / "results/experiments/p97_c_expert_recoverability/new_ics/accepted"
FROZEN_MANIFEST = ROOT / "results/experiments/final_independent_success_rate/FROZEN_TEST_MANIFEST.json"
P93_MAN = ROOT / "results/experiments/p93_frozen_router_independent_validation/EVALUATION_SPLIT_MANIFEST.csv"
P94_SPLITS = ROOT / "results/experiments/p94_unified_multiskill_actor/SPLITS.json"

EXPECTED_SHA = "ec08f59816aa2fdd"
CL_SEED = 101
ACT_EPS = 1e-5
STATE_EPS_MM = 0.5
MARGIN_SMALL = 1.0  # logit units; exploratory threshold for "small-margin" flips


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def phys_root(tid: str) -> str:
    t = re.sub(r"^(plan_[abc]__|A__|B__|C__|D__)", "", str(tid), flags=re.I)
    t = re.sub(r"__([0-9a-f]{8,})$", "", t)
    t2 = rid0(tid)
    t2 = re.sub(r"^(plan_[abc]__|A__|B__|C__|D__)", "", t2, flags=re.I)
    t2 = re.sub(r"__([0-9a-f]{8,})$", "", t2)
    return t2 if len(t2) <= len(t) else t


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
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
        w.writerows(rows)


def rho_arr(rho) -> np.ndarray:
    return ResidualBudget.from_spec(rho).as_array().astype(np.float64).reshape(7)


def token_can_represent(book, k, a, rho, tol=ACT_EPS) -> bool:
    proto = book[int(k)]
    a = np.asarray(a, np.float64).reshape(7)
    lo = np.maximum(-1.0, proto - rho)
    hi = np.minimum(1.0, proto + rho)
    return bool(np.all(a >= lo - tol) and np.all(a <= hi + tol))


def feasible_tokens(book, a, rho) -> list[int]:
    return [k for k in range(len(book)) if token_can_represent(book, k, a, rho)]


def lookup_starts() -> dict[str, MixedStart]:
    return {s.traj_id: s for s in build_p54_dataset(seed=5478)["schedule"]}


def resolve_c_start(tid: str, starts: dict[str, MixedStart]) -> MixedStart | None:
    if tid in starts:
        return starts[tid]
    pkl = P97_NEW / f"{tid}.pkl"
    if pkl.is_file():
        ep = pickle.loads(pkl.read_bytes())["ep"]
        st = MixedStart(skill="C", traj_id=tid, family="C", kind="ep", ep=ep)
        starts[tid] = st
        return st
    return None


def load_p104(path: Path, p64, mode: str, device):
    actor = build_p104_from_p64(p64, mode, device)
    blob = torch.load(path, map_location=device, weights_only=False)
    actor.load_state_dict(blob["actor"], strict=True)
    actor.eval()
    return actor, blob


def predict_step(actor, env, device):
    obs = torch.as_tensor(np.stack(env.obs_seq[-160:]), device=device, dtype=torch.float32)
    prev = torch.as_tensor(np.stack(env.prev_seq[-160:]), device=device, dtype=torch.float32)
    logits, res, act = actor(obs, prev)
    masked = mask_handoff_logits(logits[-1].float())
    probs = F.softmax(masked, dim=-1)
    top2 = torch.topk(probs, k=2)
    tok_a = int(torch.argmax(masked).item())
    z0 = tok_a
    # margin vs runner-up among motion tokens
    vals, idx = torch.topk(masked, k=2)
    margin = float(vals[0] - vals[1])
    return {
        "logits": logits[-1].detach(),
        "masked": masked.detach(),
        "probs": probs.detach(),
        "res": res[-1].detach().cpu().numpy().astype(np.float32),
        "act": act[-1].detach().cpu().numpy().astype(np.float32),
        "tok_a": tok_a,
        "top2_tok": [int(x) for x in top2.indices.cpu().numpy()],
        "top2_p": [float(x) for x in top2.values.cpu().numpy()],
        "margin": margin,
        "obs": obs,
        "prev": prev,
        "ctx_logits": logits,
    }


@torch.no_grad()
def rollout_trace(
    token_actor,
    resid_actor,
    start,
    book,
    rho,
    device,
    *,
    seed: int = CL_SEED,
    force_tokens: list[int] | None = None,
    force_k: int = 0,
    label: str = "",
) -> dict:
    """Roll out with optional forced raw actor tokens for first force_k steps.

    force_tokens[t] overrides token_actor argmax before select_token wrapper.
    Residual always from resid_actor on current history (matched physical state).
    """
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
    steps: list[dict] = []
    while not done:
        t = int(env.t)
        g0 = relative_geometry(env.env.sim)
        pt = predict_step(token_actor, env, device)
        pr = predict_step(resid_actor, env, device)
        tok_a = pt["tok_a"]
        forced = 0
        if force_tokens is not None and t < force_k and t < len(force_tokens):
            tok_a = int(force_tokens[t])
            forced = 1
        tok, st, _ = select_token(env_t=t, obs=last_obs(env), actor_tok=tok_a, st=st, v2=v2, rule=None)
        resid = pr["res"]
        if int(tok) == HANDOFF:
            action = np.zeros(7, np.float32)
            proto = np.zeros(7, np.float64)
        else:
            proto = np.asarray(book[int(tok)], np.float64)
            action = decode_m111_action(book, int(tok), resid, rho=rho)
        tr = getattr(env.mon, "tracker", None)
        ok_s = float(getattr(tr, "ok_s", 0.0) or 0.0)
        preds = evaluate_predicates(env.env.sim, ok_s=ok_s)
        steps.append(
            {
                "t": t,
                "tok": int(tok),
                "tok_a_raw": int(tok_a),
                "forced": forced,
                "resid": resid.copy(),
                "action": np.asarray(action, np.float32).copy(),
                "proto": proto.astype(np.float32),
                "margin": pt["margin"],
                "top2_tok": pt["top2_tok"],
                "top2_p": pt["top2_p"],
                "e_pos_mm": float(g0["e_pos_mm"]),
                "theta_deg": float(g0["theta_deg"]),
                "bilateral": int(bool(g0["bilateral"])),
                "min_pad": int(g0.get("min_pad") or 0),
                "working_ok": int(bool(preds.get("working_ok"))),
                "handoff_flag": 0,
                "obs": env.obs_seq[-1].astype(np.float32).copy(),
                "prev": env.prev_seq[-1].astype(np.float32).copy(),
            }
        )
        _r, done, info = env.step(int(tok), resid)
        ensure_inner_capture_buffers(env)
        steps[-1]["handoff_flag"] = int(info.get("handoff") or 0)
    term = _r3_terminal(env, info)
    return {
        "label": label,
        "SUCCESS_RELAXED_12S": int(term.get("SUCCESS_RELAXED") or info.get("SUCCESS_RELAXED_12S") or 0),
        "ep_len": int(env.t),
        "fail_reason": info.get("fail_reason"),
        "r3_ready": int(term.get("r3_ready") or 0),
        "steps": steps,
    }


def first_div(a: dict, b: dict) -> dict:
    sa, sb = a["steps"], b["steps"]
    n = min(len(sa), len(sb))
    ft = fr = fa = fp = None
    for t in range(n):
        xa, xb = sa[t], sb[t]
        if ft is None and int(xa["tok"]) != int(xb["tok"]):
            ft = t
        if fr is None and float(np.linalg.norm(xa["resid"] - xb["resid"])) > ACT_EPS:
            fr = t
        if fa is None and float(np.linalg.norm(xa["action"] - xb["action"])) > ACT_EPS:
            fa = t
        phys = (
            abs(xa["e_pos_mm"] - xb["e_pos_mm"]) > STATE_EPS_MM
            or abs(xa["theta_deg"] - xb["theta_deg"]) > 1.0
            or int(xa["bilateral"]) != int(xb["bilateral"])
        )
        if fp is None and phys and t > 0:
            fp = t
    irrev = None
    for t in range(1, n):
        if int(sa[t]["bilateral"]) == 1 and int(sb[t]["bilateral"]) == 0:
            irrev = t
            break
        if sa[t]["e_pos_mm"] < 20 and sb[t]["e_pos_mm"] > sa[t]["e_pos_mm"] + 15:
            irrev = t
            break
    return {
        "first_token_div": "" if ft is None else ft,
        "first_residual_div": "" if fr is None else fr,
        "first_action_div": "" if fa is None else fa,
        "first_phys_div": "" if fp is None else fp,
        "first_irrev": "" if irrev is None else irrev,
        "tok_agree": float(np.mean([int(sa[t]["tok"] == sb[t]["tok"]) for t in range(n)])) if n else float("nan"),
    }


def load_pres_a1() -> list[dict]:
    demos = []
    with (P98 / "B1_A1_D_PRESERVATION_TARGETS.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["skill"] != "A1":
                continue
            pr = r["phys_root"]
            path = P98 / "cache" / f"pres_A1_{pr}.npz"
            z = np.load(path, allow_pickle=True)
            demos.append(
                {
                    "traj_id": r["traj_id"],
                    "phys_root": pr,
                    "skill": "A1",
                    "obs": np.asarray(z["obs"], np.float32),
                    "prev": np.asarray(z["prev"], np.float32),
                    "actions": np.asarray(z["actions"], np.float32),
                    "tokens": np.asarray(z["tokens"], np.int64) if "tokens" in z.files else None,
                }
            )
    return demos


def load_corr_c() -> list[dict]:
    demos = []
    with (P98 / "B1_C_CORRECTION_TARGETS.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["skill"] != "C":
                continue
            pr = r["phys_root"]
            path = P98 / "cache" / f"c_corr_{pr}.npz"
            z = np.load(path, allow_pickle=True)
            demos.append(
                {
                    "traj_id": r["traj_id"],
                    "phys_root": pr,
                    "skill": "C",
                    "obs": np.asarray(z["obs"], np.float32),
                    "prev": np.asarray(z["prev"], np.float32),
                    "actions": np.asarray(z["actions"], np.float32),
                    "tokens": np.asarray(z["tokens"], np.int64),
                }
            )
    return demos


def encode_ctx(actor, obs, prev, device):
    """Return last-step ctx (64,) and logits for each timestep via causal forward."""
    o = torch.as_tensor(obs, device=device, dtype=torch.float32)
    p = torch.as_tensor(prev, device=device, dtype=torch.float32)
    if hasattr(actor, "_encode"):
        ctx, logits = actor._encode(o, p)
    else:
        # raw P64 Policy
        x = torch.cat([o, p], dim=-1)
        t = x.shape[-2]
        h = actor.in_proj(x) + actor.pos(torch.arange(t, device=x.device))
        causal = torch.triu(torch.ones(t, t, device=x.device) * float("-inf"), diagonal=1)
        ctx = actor.enc(h.unsqueeze(0), mask=causal)[0]
        logits = actor.logits(ctx)
    return ctx, logits


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("[P105] Phase 0 — load P104 v2 + reproduce", flush=True)
    p64_sha = sha16(P64_ACTOR)
    if p64_sha != EXPECTED_SHA:
        raise SystemExit(f"P64 sha mismatch {p64_sha}")

    summ104 = json.loads((P104 / "SUMMARY.json").read_text(encoding="utf-8"))
    if summ104.get("rerun", {}).get("version") != "v2" and "lambda_tok_ret" in summ104:
        # accept v2 if λ is sane
        if float(summ104["lambda_tok_ret"]) > 1e3:
            raise SystemExit("Refusing P104 v1-like λ_tok; use v2 checkpoints")
    man104 = json.loads((P104 / "TRAIN_MANIFEST.json").read_text(encoding="utf-8"))
    if float(man104.get("lambda_tok_ret", 0)) > 1e3:
        raise SystemExit("TRAIN_MANIFEST looks like invalid v1 calibration")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p64, book, rho, _ = load_actor(P64_ACTOR, device)
    p64.eval()
    for p in p64.parameters():
        p.requires_grad_(False)
    book = np.asarray(book, np.float64)
    rho = rho_arr(rho)

    path_b = Path(summ104["ckpt_paths"]["B"])
    path_c = Path(summ104["ckpt_paths"]["C"])
    actor_b, blob_b = load_p104(path_b, p64, "token_only", device)
    actor_c, blob_c = load_p104(path_c, p64, "joint", device)
    assert actor_b.mode == "token_only" and actor_c.mode == "joint"

    # Reproduce from PAIRED_CLOSED_LOOP
    paired = list(csv.DictReader((P104 / "PAIRED_CLOSED_LOOP.csv").open(encoding="utf-8")))

    def panel_stats(arm, skill):
        rows = [r for r in paired if r["arm"] == arm and r["skill"] == skill]
        if skill == "A1":
            base = [r for r in rows if int(r["P64_SUCCESS"]) == 1]
            retained = sum(1 for r in base if int(r["CAND_SUCCESS"]) == 1)
            return {"n_base": len(base), "retained": retained, "lost": len(base) - retained, "rows": base}
        succ = sum(int(r["CAND_SUCCESS"]) for r in rows)
        gains = [r for r in rows if int(r["gain"]) == 1]
        return {"n": len(rows), "succ": succ, "gains": gains, "rows": rows}

    a_a1, b_a1, c_a1 = panel_stats("A", "A1"), panel_stats("B", "A1"), panel_stats("C", "A1")
    b_cf, c_cf = panel_stats("B", "C_Feasible"), panel_stats("C", "C_Feasible")
    feas_rows = list(csv.DictReader((P104 / "TOKEN_FEASIBILITY_TRANSITIONS.csv").open(encoding="utf-8")))
    early_b = [r for r in feas_rows if r["arm"] == "B" and r["skill"] == "C" and int(r["early"]) == 1]
    c2f_b = [r for r in early_b if r["transition"] == "C_to_feasible"]
    feas_rate_b = sum(int(r["feas_new"]) for r in early_b) / max(len(early_b), 1)

    lost_b = [r for r in b_a1["rows"] if int(r["CAND_SUCCESS"]) == 0]
    lost_c = [r for r in c_a1["rows"] if int(r["CAND_SUCCESS"]) == 0]
    same_lost = sorted({r["traj_id"] for r in lost_b} & {r["traj_id"] for r in lost_c})

    repro = {
        "p64_sha": p64_sha,
        "P64_A1": f"{a_a1['retained']}/{a_a1['n_base']}",
        "P104B_A1": f"{b_a1['retained']}/{b_a1['n_base']}",
        "P104C_A1": f"{c_a1['retained']}/{c_a1['n_base']}",
        "P104B_C_Feasible": f"{b_cf['succ']}/{b_cf['n']}",
        "P104C_C_Feasible": f"{c_cf['succ']}/{c_cf['n']}",
        "P104B_early_C_feas": feas_rate_b,
        "P104B_C_to_feasible_n": len(c2f_b),
        "lost_A1_B": [r["traj_id"] for r in lost_b],
        "lost_A1_C": [r["traj_id"] for r in lost_c],
        "same_lost_A1_root": same_lost,
        "confirm": {
            "P64_A1_3_3": a_a1["retained"] == 3 and a_a1["n_base"] == 3,
            "P104B_A1_2_3": b_a1["retained"] == 2 and b_a1["n_base"] == 3,
            "P104C_A1_2_3": c_a1["retained"] == 2 and c_a1["n_base"] == 3,
            "P104B_CF_1_2": b_cf["succ"] == 1 and b_cf["n"] == 2,
            "P104C_CF_0_2": c_cf["succ"] == 0 and c_cf["n"] == 2,
            "early_feas_0_72": abs(feas_rate_b - 0.72) < 0.01,
            "c2f_15": len(c2f_b) == 15,
        },
        "ckpt_B": str(path_b),
        "ckpt_C": str(path_c),
        "lambda_tok_ret": summ104.get("lambda_tok_ret"),
        "not_v1": float(summ104.get("lambda_tok_ret") or 0) < 1e3,
    }
    print(f"[P105] reproduce {repro['confirm']}", flush=True)
    print(f"[P105] same lost A1: {same_lost}", flush=True)
    if not all(repro["confirm"].values()):
        print("[P105] WARN: some reproduce checks failed — continuing with recorded artifacts", flush=True)

    starts = lookup_starts()
    splits = json.loads((P98 / "SPLITS.json").read_text(encoding="utf-8"))
    a1_panel = list(splits["dev"]["A1"])
    lost_tid = same_lost[0] if same_lost else (lost_b[0]["traj_id"] if lost_b else None)

    # ------------------------------------------------------------------
    # Phase 1 — A1 first divergence (CL traces)
    # ------------------------------------------------------------------
    print("[P105] Phase 1 — A1 first-divergence rollouts", flush=True)
    div_rows = []
    traces = {}
    for tid in a1_panel:
        st = starts.get(tid)
        if st is None:
            print(f"[warn] missing start {tid}", flush=True)
            continue
        print(f"  rollout A1 {tid}", flush=True)
        t64 = rollout_trace(p64, p64, st, book, rho, device, label="P64")
        tb = rollout_trace(actor_b, p64, st, book, rho, device, label="B_tok_P64_res")  # Arm B residual ≡ P64
        # Also full B for completeness (same resid for token-only)
        tb_full = rollout_trace(actor_b, actor_b, st, book, rho, device, label="B_full")
        tc = rollout_trace(actor_c, actor_c, st, book, rho, device, label="C_full")
        traces[tid] = {"P64": t64, "B": tb_full, "B_tokP64res": tb, "C": tc}
        for arm, tr in (("B", tb_full), ("C", tc)):
            info = first_div(t64, tr)
            # residual identity for B on P64 history (state-matched)
            sm_res = []
            with torch.no_grad():
                for step in t64["steps"]:
                    obs = torch.as_tensor(np.stack([step["obs"]]), device=device)  # wrong — need prefix
            div_rows.append(
                {
                    "traj_id": tid,
                    "phys_root": phys_root(tid),
                    "arm": arm,
                    "P64_SUCCESS": t64["SUCCESS_RELAXED_12S"],
                    "CAND_SUCCESS": tr["SUCCESS_RELAXED_12S"],
                    "loss": int(t64["SUCCESS_RELAXED_12S"] == 1 and tr["SUCCESS_RELAXED_12S"] == 0),
                    **info,
                    "P64_ep_len": t64["ep_len"],
                    "CAND_ep_len": tr["ep_len"],
                    "P64_fail": t64.get("fail_reason"),
                    "CAND_fail": tr.get("fail_reason"),
                }
            )

    # State-matched residual identity: Arm B residual vs P64 on identical P64 histories
    print("[P105] Phase 1b — Arm B residual identity on P64 histories", flush=True)
    resid_id_rows = []
    for tid, trs in traces.items():
        t64 = trs["P64"]
        max_dr = 0.0
        mean_dr = []
        for step in t64["steps"]:
            # rebuild prefix history tensors
            # Use single-step window from stored obs/prev sequences — approximate with stepwise re-encode
            pass
        # Better: re-run predict on cumulative prefixes from t64 steps
        obs_seq = []
        prev_seq = []
        for step in t64["steps"]:
            obs_seq.append(step["obs"])
            prev_seq.append(step["prev"])
            o = torch.as_tensor(np.stack(obs_seq[-160:]), device=device, dtype=torch.float32)
            p = torch.as_tensor(np.stack(prev_seq[-160:]), device=device, dtype=torch.float32)
            with torch.no_grad():
                _, r64, _ = p64(o, p)
                _, rb, _ = actor_b(o, p)
            dr = float((rb[-1] - r64[-1]).abs().max().item())
            max_dr = max(max_dr, dr)
            mean_dr.append(float((rb[-1] - r64[-1]).norm().item()))
        resid_id_rows.append(
            {
                "traj_id": tid,
                "max_abs_dres_B_vs_P64_on_P64_hist": max_dr,
                "mean_l2_dres": float(np.mean(mean_dr)) if mean_dr else 0.0,
                "pass_identity": int(max_dr < 1e-5),
            }
        )
    write_csv(OUT / "A1_FIRST_DIVERGENCE.csv", div_rows)
    write_csv(OUT / "ARM_B_RESIDUAL_IDENTITY.csv", resid_id_rows)

    # ------------------------------------------------------------------
    # Phase 2 — Token margin on recorded P64-success A1 histories
    # ------------------------------------------------------------------
    print("[P105] Phase 2 — token margin audit", flush=True)
    margin_rows = []
    a1_pres = load_pres_a1()
    # Also include lost eval root by rolling P64 history from traces
    for d in a1_pres:
        obs = torch.as_tensor(d["obs"], device=device, dtype=torch.float32)
        prev = torch.as_tensor(d["prev"], device=device, dtype=torch.float32)
        with torch.no_grad():
            l64, r64, a64 = p64(obs, prev)
            lb, rb, ab = actor_b(obs, prev)
            lc, rc, ac = actor_c(obs, prev)
            m64 = mask_handoff_logits(l64.float())
            mb = mask_handoff_logits(lb.float())
            mc = mask_handoff_logits(lc.float())
            z64 = torch.argmax(m64, dim=-1)
            zb = torch.argmax(mb, dim=-1)
            zc = torch.argmax(mc, dim=-1)
            p64_prob = F.softmax(m64, dim=-1)
            log_b = F.log_softmax(mb, dim=-1)
            log_c = F.log_softmax(mc, dim=-1)
            kl_b = (p64_prob * (p64_prob.clamp_min(1e-8).log() - log_b)).sum(-1)
            kl_c = (p64_prob * (p64_prob.clamp_min(1e-8).log() - log_c)).sum(-1)
            # margins
            top2_64 = torch.topk(m64, k=2, dim=-1).values
            margin64 = top2_64[..., 0] - top2_64[..., 1]
            # executed action with frozen P64 residual under each token
            for t in range(len(d["obs"])):
                z0 = int(z64[t].item())
                zb_t, zc_t = int(zb[t].item()), int(zc[t].item())
                proto0 = book[z0]
                # action under new argmax + P64 residual
                r_p64 = r64[t].cpu().numpy()
                a_same = np.clip(proto0 + r_p64, -1, 1)
                a_b = np.clip(book[zb_t] + r_p64, -1, 1) if zb_t != HANDOFF else np.zeros(7)
                a_c_tok = np.clip(book[zc_t] + r_p64, -1, 1) if zc_t != HANDOFF else np.zeros(7)
                phase = "early" if t < 5 else ("late" if t > len(d["obs"]) * 0.7 else "middle")
                flip_b = int(zb_t != z0)
                flip_c = int(zc_t != z0)
                m = float(margin64[t].item())
                kind_b = "same_token" if not flip_b else ("small_margin_flip" if m < MARGIN_SMALL else "large_margin_flip")
                kind_c = "same_token" if not flip_c else ("small_margin_flip" if m < MARGIN_SMALL else "large_margin_flip")
                margin_rows.append(
                    {
                        "source": "A1_preservation_train",
                        "traj_id": d["traj_id"],
                        "phys_root": d["phys_root"],
                        "t": t,
                        "phase": phase,
                        "z_P64": z0,
                        "z_B": zb_t,
                        "z_C": zc_t,
                        "margin_P64": m,
                        "KL_B": float(kl_b[t].item()),
                        "KL_C": float(kl_c[t].item()),
                        "argmax_changed_B": flip_b,
                        "argmax_changed_C": flip_c,
                        "proto_l2_B": float(np.linalg.norm(book[zb_t] - proto0)) if flip_b and zb_t != HANDOFF else 0.0,
                        "act_l2_frozenRes_B": float(np.linalg.norm(a_b - a_same)),
                        "act_l2_frozenRes_C": float(np.linalg.norm(a_c_tok - a_same)),
                        "act_l2_B_ownRes": float(np.linalg.norm(ab[t].cpu().numpy() - a64[t].cpu().numpy())),
                        "act_l2_C_ownRes": float(np.linalg.norm(ac[t].cpu().numpy() - a64[t].cpu().numpy())),
                        "res_l2_B": float(np.linalg.norm(rb[t].cpu().numpy() - r_p64)),
                        "res_l2_C": float(np.linalg.norm(rc[t].cpu().numpy() - r_p64)),
                        "flip_kind_B": kind_b,
                        "flip_kind_C": kind_c,
                    }
                )

    # Lost eval root: use P64 closed-loop history from traces
    if lost_tid and lost_tid in traces:
        t64 = traces[lost_tid]["P64"]
        obs_seq, prev_seq = [], []
        for step in t64["steps"]:
            obs_seq.append(step["obs"])
            prev_seq.append(step["prev"])
            o = torch.as_tensor(np.stack(obs_seq[-160:]), device=device, dtype=torch.float32)
            p = torch.as_tensor(np.stack(prev_seq[-160:]), device=device, dtype=torch.float32)
            t = step["t"]
            with torch.no_grad():
                l64, r64, a64 = p64(o, p)
                lb, rb, ab = actor_b(o, p)
                lc, rc, ac = actor_c(o, p)
                m64 = mask_handoff_logits(l64[-1].float())
                mb = mask_handoff_logits(lb[-1].float())
                mc = mask_handoff_logits(lc[-1].float())
                z0 = int(torch.argmax(m64).item())
                zb_t = int(torch.argmax(mb).item())
                zc_t = int(torch.argmax(mc).item())
                p64_prob = F.softmax(m64, dim=-1)
                kl_b = float((p64_prob * (p64_prob.clamp_min(1e-8).log() - F.log_softmax(mb, dim=-1))).sum())
                kl_c = float((p64_prob * (p64_prob.clamp_min(1e-8).log() - F.log_softmax(mc, dim=-1))).sum())
                top2 = torch.topk(m64, k=2).values
                m = float(top2[0] - top2[1])
                r_p64 = r64[-1].cpu().numpy()
                a_same = np.clip(book[z0] + r_p64, -1, 1) if z0 != HANDOFF else np.zeros(7)
                a_b = np.clip(book[zb_t] + r_p64, -1, 1) if zb_t != HANDOFF else np.zeros(7)
                flip_b = int(zb_t != z0)
                flip_c = int(zc_t != z0)
                margin_rows.append(
                    {
                        "source": "A1_lost_eval_P64_CL_hist",
                        "traj_id": lost_tid,
                        "phys_root": phys_root(lost_tid),
                        "t": t,
                        "phase": "early" if t < 5 else ("late" if t > 0.7 * len(t64["steps"]) else "middle"),
                        "z_P64": z0,
                        "z_B": zb_t,
                        "z_C": zc_t,
                        "margin_P64": m,
                        "KL_B": kl_b,
                        "KL_C": kl_c,
                        "argmax_changed_B": flip_b,
                        "argmax_changed_C": flip_c,
                        "proto_l2_B": float(np.linalg.norm(book[zb_t] - book[z0])) if flip_b and zb_t != HANDOFF and z0 != HANDOFF else 0.0,
                        "act_l2_frozenRes_B": float(np.linalg.norm(a_b - a_same)),
                        "act_l2_frozenRes_C": float(
                            np.linalg.norm((np.clip(book[zc_t] + r_p64, -1, 1) if zc_t != HANDOFF else np.zeros(7)) - a_same)
                        ),
                        "act_l2_B_ownRes": float(np.linalg.norm(ab[-1].cpu().numpy() - a64[-1].cpu().numpy())),
                        "act_l2_C_ownRes": float(np.linalg.norm(ac[-1].cpu().numpy() - a64[-1].cpu().numpy())),
                        "res_l2_B": float(np.linalg.norm(rb[-1].cpu().numpy() - r_p64)),
                        "res_l2_C": float(np.linalg.norm(rc[-1].cpu().numpy() - r_p64)),
                        "flip_kind_B": "same_token" if not flip_b else ("small_margin_flip" if m < MARGIN_SMALL else "large_margin_flip"),
                        "flip_kind_C": "same_token" if not flip_c else ("small_margin_flip" if m < MARGIN_SMALL else "large_margin_flip"),
                    }
                )
    write_csv(OUT / "TOKEN_MARGIN_AUDIT.csv", margin_rows)

    # ------------------------------------------------------------------
    # Phase 3 — Non-degenerate CF on lost A1
    # ------------------------------------------------------------------
    print("[P105] Phase 3 — non-degenerate counterfactuals", flush=True)
    cf_rows = []
    if lost_tid and lost_tid in starts and lost_tid in traces:
        st = starts[lost_tid]
        t64, tb = traces[lost_tid]["P64"], traces[lost_tid]["B"]
        info = first_div(t64, tb)
        t_star = info["first_token_div"]
        t_star = int(t_star) if t_star != "" else 0
        # Matched-state force: at each step of a joint rollout, when histories are still identical,
        # forcing tokens from the other policy. For first disagreement at t*, prefixes 0..t*-1 match.
        # Intervention: start from reset; for steps < t*+k use forced P64 tokens while running B tok+P64 res
        # Actually cleaner: force P64 raw tokens for first (t_star+k) steps on B policy residual=P64, then free B.
        p64_toks = [s["tok_a_raw"] for s in t64["steps"]]
        b_toks = [s["tok_a_raw"] for s in tb["steps"]]

        modes = [
            ("P64_full", p64, p64, None, 0),
            ("B_full_tokB_resP64", actor_b, p64, None, 0),
            ("B_full_tokB_resB", actor_b, actor_b, None, 0),
        ]
        for k in (1, 2, 4):
            # Force P64 tokens for first (t_star + k) steps, then resume B — identifying if differs from both
            fk = min(t_star + k, len(p64_toks))
            modes.append((f"force_P64tok_then_B_k{k}_from_tstar{t_star}", actor_b, p64, p64_toks, fk))
            # Reverse: force B tokens for first (t_star + k) on P64 residual path then resume P64
            modes.append((f"force_Btok_then_P64_k{k}_from_tstar{t_star}", p64, p64, b_toks, fk))

        for name, ta, ra, forced, fk in modes:
            tr = rollout_trace(ta, ra, st, book, rho, device, force_tokens=forced, force_k=fk, label=name)
            # Check non-degeneracy vs baselines
            same_p64 = tr["SUCCESS_RELAXED_12S"] == t64["SUCCESS_RELAXED_12S"] and tr["ep_len"] == t64["ep_len"]
            same_b = tr["SUCCESS_RELAXED_12S"] == tb["SUCCESS_RELAXED_12S"] and tr["ep_len"] == tb["ep_len"]
            # Token sequence equality quick check
            if forced is None:
                identifying = 1
                note = "baseline"
            else:
                # Compare token sequences to baselines
                toks = [s["tok"] for s in tr["steps"]]
                toks64 = [s["tok"] for s in t64["steps"]]
                toksb = [s["tok"] for s in tb["steps"]]
                n = min(len(toks), len(toks64), len(toksb))
                eq64 = toks[:n] == toks64[:n]
                eqb = toks[:n] == toksb[:n]
                identifying = int(not (eq64 or eqb))
                note = "nondeg_if_token_seq_differs_from_both_baselines"
            cf_rows.append(
                {
                    "traj_id": lost_tid,
                    "phys_root": phys_root(lost_tid),
                    "mode": name,
                    "t_star_first_tok_div": t_star,
                    "force_k": fk,
                    "SUCCESS_RELAXED_12S": tr["SUCCESS_RELAXED_12S"],
                    "ep_len": tr["ep_len"],
                    "fail_reason": tr.get("fail_reason"),
                    "identifying": identifying,
                    "rescues_vs_B": int(tr["SUCCESS_RELAXED_12S"] == 1 and tb["SUCCESS_RELAXED_12S"] == 0),
                    "destroys_vs_P64": int(tr["SUCCESS_RELAXED_12S"] == 0 and t64["SUCCESS_RELAXED_12S"] == 1),
                    "note": note,
                }
            )
            print(f"  CF {name}: succ={tr['SUCCESS_RELAXED_12S']} id={identifying}", flush=True)

        # Arm C: separate token vs residual — force P64 tokens with C residual, and C tokens with P64 residual
        tc = traces[lost_tid]["C"]
        for name, ta, ra in (
            ("C_tok_P64_res", actor_c, p64),
            ("P64_tok_C_res", p64, actor_c),
            ("C_full", actor_c, actor_c),
        ):
            tr = rollout_trace(ta, ra, st, book, rho, device, label=name)
            # For C, tok/res swap is identifying if residual delta non-zero (token heads differ from P64)
            cf_rows.append(
                {
                    "traj_id": lost_tid,
                    "phys_root": phys_root(lost_tid),
                    "mode": name,
                    "t_star_first_tok_div": first_div(t64, tc)["first_token_div"],
                    "force_k": 0,
                    "SUCCESS_RELAXED_12S": tr["SUCCESS_RELAXED_12S"],
                    "ep_len": tr["ep_len"],
                    "fail_reason": tr.get("fail_reason"),
                    "identifying": 1,
                    "rescues_vs_B": int(tr["SUCCESS_RELAXED_12S"] == 1 and tb["SUCCESS_RELAXED_12S"] == 0),
                    "destroys_vs_P64": int(tr["SUCCESS_RELAXED_12S"] == 0 and t64["SUCCESS_RELAXED_12S"] == 1),
                    "note": "Arm C token/residual separation (heads may differ; not P102-degenerate)",
                }
            )
    write_csv(OUT / "COUNTERFACTUAL_INTERVENTIONS.csv", cf_rows)

    # ------------------------------------------------------------------
    # Phase 4 — Early C action compatibility
    # ------------------------------------------------------------------
    print("[P105] Phase 4 — C action compatibility", flush=True)
    c_demos = load_corr_c()
    compat_rows = []
    for d in c_demos:
        obs = torch.as_tensor(d["obs"], device=device, dtype=torch.float32)
        prev = torch.as_tensor(d["prev"], device=device, dtype=torch.float32)
        with torch.no_grad():
            l64, r64, _ = p64(obs, prev)
            lb, rb, _ = actor_b(obs, prev)
            z64 = torch.argmax(mask_handoff_logits(l64.float()), dim=-1).cpu().numpy()
            zb = torch.argmax(mask_handoff_logits(lb.float()), dim=-1).cpu().numpy()
        for t in range(len(d["actions"])):
            a_star = d["actions"][t]
            zh0, zh1 = int(z64[t]), int(zb[t])
            Z = feasible_tokens(book, a_star, rho)
            cat = "D" if not Z else ("C" if zh0 not in Z else ("A" if zh0 == int(d["tokens"][t]) else "B"))
            if cat != "C" and t >= 5:
                # still record early focus; skip late non-C for brevity? Keep all C and early
                if cat != "C":
                    continue
            r_p = r64[t].cpu().numpy()
            # E_frozen: min over z of ||clip(proto[z]+r_P64)-a||
            e_frozen = 1e9
            best_z = -1
            for k in range(len(book)):
                if k == HANDOFF:
                    continue
                a_k = np.clip(book[k] + r_p, -1, 1)
                e = float(np.linalg.norm(a_k - a_star))
                if e < e_frozen:
                    e_frozen, best_z = e, k
            a_old = np.clip(book[zh0] + r_p, -1, 1) if zh0 != HANDOFF else np.zeros(7)
            a_new = np.clip(book[zh1] + r_p, -1, 1) if zh1 != HANDOFF else np.zeros(7)
            err_old = float(np.linalg.norm(a_old - a_star))
            err_new = float(np.linalg.norm(a_new - a_star))
            admits = int(zh1 in Z) if Z else 0
            compat_rows.append(
                {
                    "traj_id": d["traj_id"],
                    "phys_root": d["phys_root"],
                    "t": t,
                    "early": int(t < 5),
                    "p64_category": cat,
                    "z_P64": zh0,
                    "z_B": zh1,
                    "z_expert": int(d["tokens"][t]),
                    "new_token_admits_legal_residual": admits,
                    "err_frozenRes_P64tok": err_old,
                    "err_frozenRes_Btok": err_new,
                    "err_reduced": int(err_new + 1e-9 < err_old),
                    "E_frozen_min": e_frozen,
                    "E_frozen_best_z": best_z,
                    "transition_C_to_feas": int(cat == "C" and admits == 1),
                }
            )
    write_csv(OUT / "C_ACTION_COMPATIBILITY.csv", compat_rows)
    c2f_compat = [r for r in compat_rows if int(r["transition_C_to_feas"]) == 1 and int(r["early"]) == 1]
    n_err_reduced = sum(int(r["err_reduced"]) for r in c2f_compat)

    # ------------------------------------------------------------------
    # Phase 5 — Skill conflict (ctx NN + gradient cosine)
    # ------------------------------------------------------------------
    print("[P105] Phase 5 — skill conflict analysis", flush=True)
    # Sample ctx from A1 preservation and C correction at early steps using frozen backbone (P64 encode)
    a1_ctx, c_ctx = [], []
    a1_meta, c_meta = [], []
    for d in a1_pres:
        ctx, logits = encode_ctx(p64, d["obs"], d["prev"], device)
        for t in range(min(8, len(d["obs"]))):
            a1_ctx.append(ctx[t].detach().cpu().numpy())
            m = mask_handoff_logits(logits[t].float())
            a1_meta.append({"traj": d["traj_id"], "t": t, "z": int(torch.argmax(m).item())})
    for d in c_demos:
        ctx, logits = encode_ctx(p64, d["obs"], d["prev"], device)
        for t in range(min(8, len(d["obs"]))):
            c_ctx.append(ctx[t].detach().cpu().numpy())
            m = mask_handoff_logits(logits[t].float())
            # desired token = expert if feasible else expert anyway for conflict probe
            c_meta.append({"traj": d["traj_id"], "t": t, "z_expert": int(d["tokens"][t]), "z_p64": int(torch.argmax(m).item())})
    a1_ctx_a = np.stack(a1_ctx)
    c_ctx_a = np.stack(c_ctx)
    # cosine NN: for each C early state, nearest A1
    a1_n = a1_ctx_a / (np.linalg.norm(a1_ctx_a, axis=1, keepdims=True) + 1e-8)
    c_n = c_ctx_a / (np.linalg.norm(c_ctx_a, axis=1, keepdims=True) + 1e-8)
    sims = c_n @ a1_n.T
    nn_idx = sims.argmax(axis=1)
    nn_sim = sims.max(axis=1)
    conflict_pairs = []
    for i, j in enumerate(nn_idx):
        if nn_sim[i] < 0.9:
            continue
        # conflicting desired tokens?
        z_a1 = a1_meta[j]["z"]
        z_c = c_meta[i]["z_expert"]
        if z_a1 != z_c:
            conflict_pairs.append(
                {
                    "cos_sim": float(nn_sim[i]),
                    "c_traj": c_meta[i]["traj"],
                    "c_t": c_meta[i]["t"],
                    "a1_traj": a1_meta[j]["traj"],
                    "a1_t": a1_meta[j]["t"],
                    "z_a1_p64": z_a1,
                    "z_c_expert": z_c,
                }
            )
    conflict_pairs = sorted(conflict_pairs, key=lambda x: -x["cos_sim"])[:20]

    # Gradient cosine on token head (logits.weight) under A1 retention CE vs C correction CE
    # Use actor_b (token-only) for grads on logits
    def token_ce_grad(actor, demos, target_fn):
        actor.zero_grad(set_to_none=True)
        losses = []
        for d in demos:
            obs = torch.as_tensor(d["obs"][:8], device=device, dtype=torch.float32)
            prev = torch.as_tensor(d["prev"][:8], device=device, dtype=torch.float32)
            logits, _, _ = actor(obs, prev)
            logp = F.log_softmax(mask_handoff_logits(logits.float()), dim=-1)
            tgt = torch.as_tensor([target_fn(d, t) for t in range(len(obs))], device=device, dtype=torch.long)
            losses.append(F.nll_loss(logp, tgt))
        loss = torch.stack(losses).mean()
        loss.backward()
        g = actor.base.logits.weight.grad
        if g is None:
            return None
        return g.detach().flatten().cpu().clone()

    def a1_tgt(d, t):
        # retention target = P64 argmax on that history
        with torch.no_grad():
            o = torch.as_tensor(d["obs"][: t + 1], device=device, dtype=torch.float32)
            p = torch.as_tensor(d["prev"][: t + 1], device=device, dtype=torch.float32)
            logits, _, _ = p64(o, p)
            return int(torch.argmax(mask_handoff_logits(logits[-1].float())).item())

    def c_tgt(d, t):
        return int(d["tokens"][t])

    # fresh copy for grad probe to avoid polluting actor_b
    probe = build_p104_from_p64(p64, "token_only", device)
    probe.load_state_dict(actor_b.state_dict(), strict=True)
    g_a1 = token_ce_grad(probe, a1_pres[:4], a1_tgt)
    probe2 = build_p104_from_p64(p64, "token_only", device)
    probe2.load_state_dict(actor_b.state_dict(), strict=True)
    g_c = token_ce_grad(probe2, c_demos[:4], c_tgt)
    if g_a1 is not None and g_c is not None:
        cos_g = float(F.cosine_similarity(g_a1.unsqueeze(0), g_c.unsqueeze(0)).item())
    else:
        cos_g = float("nan")

    # Tokens strongly updated by C: compare P64 vs B logits on C states
    strong_c_updates = []
    for d in c_demos:
        obs = torch.as_tensor(d["obs"][:5], device=device, dtype=torch.float32)
        prev = torch.as_tensor(d["prev"][:5], device=device, dtype=torch.float32)
        with torch.no_grad():
            l64, _, _ = p64(obs, prev)
            lb, _, _ = actor_b(obs, prev)
            dlogit = (lb - l64).abs()
            for t in range(len(obs)):
                z_exp = int(d["tokens"][t])
                strong_c_updates.append(float(dlogit[t, z_exp].item()))
    # On lost A1 root history, see if flipped tokens had large C-driven logit deltas
    lost_flip_c_overlap = []
    if lost_tid and lost_tid in traces:
        obs_seq, prev_seq = [], []
        for step in traces[lost_tid]["P64"]["steps"][:10]:
            obs_seq.append(step["obs"])
            prev_seq.append(step["prev"])
            o = torch.as_tensor(np.stack(obs_seq[-160:]), device=device, dtype=torch.float32)
            p = torch.as_tensor(np.stack(prev_seq[-160:]), device=device, dtype=torch.float32)
            with torch.no_grad():
                l64, _, _ = p64(o, p)
                lb, _, _ = actor_b(o, p)
                z0 = int(torch.argmax(mask_handoff_logits(l64[-1].float())).item())
                zb = int(torch.argmax(mask_handoff_logits(lb[-1].float())).item())
                if zb != z0:
                    lost_flip_c_overlap.append(
                        {
                            "t": step["t"],
                            "z_P64": z0,
                            "z_B": zb,
                            "dlogit_at_zB": float((lb[-1, zb] - l64[-1, zb]).item()),
                            "dlogit_at_zP64": float((lb[-1, z0] - l64[-1, z0]).item()),
                        }
                    )

    skill_md = f"""# SKILL_CONFLICT_ANALYSIS

Generated: {utc()}

## Representation nearest neighbors (frozen P64 ctx, early t≤7)

- A1 preservation early states: n={len(a1_ctx)}
- C correction early states: n={len(c_ctx)}
- High-similarity (cos≥0.9) pairs with **conflicting** desired tokens (P64-A1 argmax vs C expert): **{len(conflict_pairs)}**
- Top pairs (up to 10):

| cos | C state | A1 state | z_A1 | z_C_expert |
|---:|---|---|---:|---:|
"""
    for p in conflict_pairs[:10]:
        skill_md += f"| {p['cos_sim']:.3f} | {p['c_traj']} t={p['c_t']} | {p['a1_traj']} t={p['a1_t']} | {p['z_a1_p64']} | {p['z_c_expert']} |\n"

    skill_md += f"""

## Token-head gradient cosine (A1 retention CE vs C expert CE)

cosine(g_A1, g_C) on `logits.weight` = **{cos_g:.4f}**

Interpretation guide:
- Near **+1**: updates aligned (not evidence of conflict).
- Near **0**: weakly related.
- Near **-1**: opposing updates on shared parameters — *suggestive* of competition, not proof of irreducible conflict.

## Lost-A1 flip tokens vs C-driven logit shifts

Flips on lost root `{lost_tid}` (first 10 P64-hist steps): {json.dumps(lost_flip_c_overlap, indent=2)}

Mean |Δlogit| at expert token on early C demos: {float(np.mean(strong_c_updates)) if strong_c_updates else float('nan'):.4f}

## Conclusion draft

Feature-space neighbors with conflicting targets exist ({len(conflict_pairs)} pairs at cos≥0.9). Gradient cosine = {cos_g:.4f}.
This is **not** sufficient alone to claim an irreducible A1/C skill conflict; combine with CF rescue evidence and coverage analysis.
"""
    (OUT / "SKILL_CONFLICT_ANALYSIS.md").write_text(skill_md, encoding="utf-8")

    # ------------------------------------------------------------------
    # Synthesize answers + P106 proposal direction
    # ------------------------------------------------------------------
    lost_div = [r for r in div_rows if r["traj_id"] == lost_tid]
    flip_on_lost = [r for r in margin_rows if r["traj_id"] == lost_tid and int(r.get("argmax_changed_B", 0)) == 1]
    small_kl_large_act = [
        r
        for r in margin_rows
        if r["source"] == "A1_lost_eval_P64_CL_hist"
        and float(r["KL_B"]) < 0.05
        and float(r["act_l2_frozenRes_B"]) > 0.1
    ]
    cf_rescue = [r for r in cf_rows if int(r.get("rescues_vs_B", 0)) == 1 and int(r.get("identifying", 0)) == 1]
    resid_id_ok = all(int(r["pass_identity"]) == 1 for r in resid_id_rows) if resid_id_rows else False

    # Decide P106 primary mechanism
    n_small = sum(1 for r in flip_on_lost if r.get("flip_kind_B") == "small_margin_flip")
    n_large = sum(1 for r in flip_on_lost if r.get("flip_kind_B") == "large_margin_flip")
    mean_act_on_flip = float(np.mean([float(r["act_l2_frozenRes_B"]) for r in flip_on_lost])) if flip_on_lost else 0.0
    frac_c2f_err_reduced = n_err_reduced / max(len(c2f_compat), 1)

    if flip_on_lost and mean_act_on_flip > 0.15 and (n_small + n_large) > 0:
        primary = "B_ACTION_AWARE_RETENTION"
        hyp = (
            "A1 loss is driven by early argmax token flips that induce large executed-action jumps "
            "under the frozen P64 residual despite modest KL; action-aware / margin-sensitive token "
            "retention can preserve A1 without erasing C corrections."
        )
    elif n_small >= n_large and flip_on_lost:
        primary = "A_DECISION_MARGIN_RETENTION"
        hyp = "Critical small-margin argmax flips on A1 histories cause the loss; margin retention stabilizes P64 tokens."
    elif frac_c2f_err_reduced < 0.5:
        primary = "D_ACTION_COMPATIBLE_TOKEN_SUPERVISION"
        hyp = "Many feasible token flips do not reduce frozen-residual action error; supervise action-compatible tokens."
    elif cos_g < -0.2 and len(conflict_pairs) >= 5:
        primary = "C_ROOT_DIVERSE_PRESERVATION"
        # Actually if conflict, coverage might still help; prefer conflict-aware retention
        primary = "A_DECISION_MARGIN_RETENTION"
        hyp = "A1/C gradient opposition plus NN conflicts; strengthen decision-margin retention on A1."
    else:
        primary = "C_ROOT_DIVERSE_PRESERVATION"
        hyp = "Failure states underrepresented in preservation; expand root-diverse A1 retention coverage."

    # If CF rescues with forced P64 tokens, that strongly supports token-flip causality → A or B
    if cf_rescue:
        if mean_act_on_flip > 0.15:
            primary = "B_ACTION_AWARE_RETENTION"
        else:
            primary = "A_DECISION_MARGIN_RETENTION"
        hyp = (
            "Identifying forced-P64-token interventions rescue A1 from P104-B failure, confirming "
            "token-selection divergence as causal; primary P106 = targeted token retention "
            f"({primary})."
        )

    answers = {
        "1_cause_A1_loss": None,
        "2_argmax_flip_timestep": None,
        # Q3 must reflect measured small-KL∧large-act coexistence — do NOT OR with generic large act_l2.
        "3_small_KL_large_action": bool(small_kl_large_act),
        "4_targeted_token_intervention_rescue": bool(cf_rescue)
        or any(int(r.get("rescues_vs_B", 0)) == 1 and "force_P64tok" in str(r.get("mode", "")) for r in cf_rows),
        "5_early_C_corrections_reducing_action_error": {"n_c2f_early": len(c2f_compat), "n_err_reduced": n_err_reduced},
        "6_frozen_residual_compatibility_limiting": frac_c2f_err_reduced < 0.5,
        "7_genuine_A1_C_conflict": {
            "grad_cosine": cos_g,
            "nn_conflict_pairs": len(conflict_pairs),
            "claim": "suggestive" if (cos_g < -0.1 and len(conflict_pairs) >= 3) else "insufficient_alone",
        },
        "8_best_P106": primary,
    }
    if lost_div:
        ld = [r for r in lost_div if r["arm"] == "B"][0]
        answers["2_argmax_flip_timestep"] = ld.get("first_token_div")
        answers["1_cause_A1_loss"] = (
            f"First token divergence at t={ld.get('first_token_div')}, action div at t={ld.get('first_action_div')}, "
            f"phys div at t={ld.get('first_phys_div')}; Arm B residual identity on P64 hist pass={resid_id_ok}."
        )

    ban = set()
    if FROZEN_MANIFEST.is_file():
        man = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
        ban |= {phys_root(e["traj_id"]) for e in man["episodes"]}
    provenance = {
        "created": utc(),
        "p64_sha": p64_sha,
        "p104_v2_ckpts": {"B": str(path_b), "C": str(path_c)},
        "lambda_tok_ret": summ104.get("lambda_tok_ret"),
        "refused_v1": True,
        "lost_A1_root": lost_tid,
        "same_lost_B_and_C": same_lost,
        "reproduce": repro,
        "sealed_panel_opened": False,
        "training_performed": False,
    }
    write_json(OUT / "PROVENANCE_AUDIT.json", provenance)

    summary = {
        "case": "P105_COMPLETE",
        "created": utc(),
        "p64_sha": p64_sha,
        "reproduce": repro,
        "lost_A1": lost_tid,
        "armB_residual_identity_pass": resid_id_ok,
        "n_margin_rows": len(margin_rows),
        "n_cf_rows": len(cf_rows),
        "n_cf_identifying_rescues": len(cf_rescue),
        "c2f_early_action_err_reduced": {"n": len(c2f_compat), "reduced": n_err_reduced},
        "grad_cosine_A1_vs_C": cos_g,
        "nn_conflict_pairs": len(conflict_pairs),
        "answers": answers,
        "p106_primary": primary,
        "p106_hypothesis": hyp,
    }
    write_json(OUT / "SUMMARY.json", summary)

    report = f"""# FINAL_REPORT — P105 Token Interference and Action-Compatibility Audit

Generated: {utc()}

## 0. Reproduce P104 v2

| Check | Value | OK |
|---|---|---|
| P64 A1 | {repro['P64_A1']} | {repro['confirm']['P64_A1_3_3']} |
| P104-B A1 | {repro['P104B_A1']} | {repro['confirm']['P104B_A1_2_3']} |
| P104-C A1 | {repro['P104C_A1']} | {repro['confirm']['P104C_A1_2_3']} |
| P104-B C-Feasible | {repro['P104B_C_Feasible']} | {repro['confirm']['P104B_CF_1_2']} |
| P104-C C-Feasible | {repro['P104C_C_Feasible']} | {repro['confirm']['P104C_CF_0_2']} |
| Early C feas (B) | {repro['P104B_early_C_feas']:.3f} | {repro['confirm']['early_feas_0_72']} |
| C→feasible count | {repro['P104B_C_to_feasible_n']} | {repro['confirm']['c2f_15']} |
| Same lost A1 root B∩C | {same_lost} | — |
| Not using v1 λ | λ_tok={repro['lambda_tok_ret']} | {repro['not_v1']} |

**Lost A1 root (both arms):** `{lost_tid}`

## 1. A1 first divergence

See `A1_FIRST_DIVERGENCE.csv`. Arm B residual identity on P64 histories: **pass={resid_id_ok}** (`ARM_B_RESIDUAL_IDENTITY.csv`).

| Arm | first_token | first_action | first_phys | first_irrev | P64→Cand |
|---|---|---|---|---|---|
"""
    for r in div_rows:
        if r["traj_id"] != lost_tid:
            continue
        report += (
            f"| {r['arm']} | {r['first_token_div']} | {r['first_action_div']} | {r['first_phys_div']} | "
            f"{r['first_irrev']} | {r['P64_SUCCESS']}→{r['CAND_SUCCESS']} |\n"
        )

    report += f"""

## 2. Token margin (identical P64 histories)

Flip kinds on lost-root P64-CL history: small={n_small}, large={n_large}; mean act L2 under frozen residual on flips={mean_act_on_flip:.4f}.
Small-KL & large-action rows on lost hist: n={len(small_kl_large_act)}.

## 3. Non-degenerate CF

Identifying rescues of P104-B failure: **{len(cf_rescue)}** (see `COUNTERFACTUAL_INTERVENTIONS.csv`).
Modes force P64/B tokens at matched states; residual for Token-only comparisons uses frozen P64 residual.

## 4. Early C action compatibility

Among early Category-C→feasible transitions with frozen P64 residual: **{n_err_reduced}/{len(c2f_compat)}** reduced executed-action error vs P64 token.
`E_frozen` reported per step in `C_ACTION_COMPATIBILITY.csv`.

## 5. Skill conflict

Gradient cosine(A1 retention, C correction) = **{cos_g:.4f}**; NN conflict pairs (cos≥0.9) = **{len(conflict_pairs)}**.
Details: `SKILL_CONFLICT_ANALYSIS.md`.

## 6. Answers

1. **Cause of A1 loss:** {answers['1_cause_A1_loss']}
2. **Argmax flip timestep:** {answers['2_argmax_flip_timestep']}
3. **Small KL hide large action change?** {answers['3_small_KL_large_action']}
4. **Targeted token intervention rescue A1?** {answers['4_targeted_token_intervention_rescue']} (n_identifying_rescues={len(cf_rescue)})
5. **Early C corrections improving executed actions:** {n_err_reduced}/{len(c2f_compat)}
6. **Frozen-residual compatibility limiting?** {answers['6_frozen_residual_compatibility_limiting']}
7. **Genuine A1/C conflict?** {answers['7_genuine_A1_C_conflict']}
8. **Best P106 intervention:** **{primary}**

Hypothesis: {hyp}

## 7. Stop

Diagnostics only. P106 proposal: `docs/p106_token_retention_pilot_proposal.md`. **Do not train P106 without approval.**
"""
    (OUT / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    print(report, flush=True)

    # Write P106 proposal
    write_p106_proposal(primary, hyp, lost_tid, cos_g, len(conflict_pairs), n_err_reduced, len(c2f_compat), len(cf_rescue))
    print(f"[P105] DONE primary_P106={primary}", flush=True)


def write_p106_proposal(primary, hyp, lost_tid, cos_g, n_pairs, n_err_red, n_c2f, n_rescue) -> None:
    path = ROOT / "docs/p106_token_retention_pilot_proposal.md"
    # Map primary to concrete loss
    if primary.startswith("A_"):
        mech = "decision-margin retention"
        loss_extra = (
            "L_margin = E_{H~D_A1}[ softplus( m_min - (logits_new[z0]-max_{z≠z0} logits_new[z]) ) ] "
            "with z0=argmax(logits_P64), m_min predefined (e.g. 1.0)."
        )
        change = "Add margin retention on A1/B1/D P64-success histories; keep P104-B CE correction + KL retention as reference terms with fixed coeffs from train-only calib."
    elif primary.startswith("B_"):
        mech = "action-aware token retention"
        loss_extra = (
            "L_act = E_{H~D_A1}[ || proto[z_new] - proto[z_P64] ||_2^2 ] "
            "(equivalently executed-action distance under frozen r_P64). "
            "Optionally gate to steps where ||Δproto|| exceeds τ."
        )
        change = "Replace/augment KL-only retention with action-prototype distance retention; one primary change vs P104-B."
    elif primary.startswith("D_"):
        mech = "action-compatible token supervision"
        loss_extra = (
            "On C Category-C steps, choose z_tgt = argmin_{z in Z(a*)} ||clip(proto[z]+r_P64(H))-a*|| "
            "instead of expert token when expert is not action-compatible under frozen residual."
        )
        change = "Change C token target construction only; keep retention recipe identical to P104-B."
    else:
        mech = "root-diverse preservation coverage"
        loss_extra = "Expand D_A1 preservation to include more phys-roots / phases (esp. states near lost root family), balanced sampling unchanged otherwise."
        change = "Data coverage only; loss identical to P104-B."

    text = f"""# P106 — Token Retention Pilot Proposal (from P105)

**Status:** Proposal only. **Do not train without explicit approval.**  
**Baseline:** Immutable P64 `ec08f59816aa2fdd`.  
**Reference control:** P104-B v2 recipe (fixed; not retuned).

## Exact hypothesis

{hyp}

Primary mechanism: **{mech}** (`{primary}`).

P105 anchors: lost A1 `{lost_tid}`; identifying CF rescues={n_rescue}; early C→feas action-err reduced={n_err_red}/{n_c2f}; grad cos(A1,C)={cos_g:.4f}; NN conflict pairs={n_pairs}.

## Architecture and trainable parameters

- One unified causal Transformer Actor (P64 init).
- **Freeze** Transformer backbone and **entire residual** (including any Δ; Token-only).
- **Train only** existing M111 token head (`logits.weight`, `logits.bias`).
- No router, no skill labels at inference, no M111 vocab change, no PPO/GRPO.

## Training data and root provenance

- **Correction:** same P98 B1/C correction targets as P104 (feasibility-gated; Category D masked).
- **Preservation:** P98 A1/B1/D P64-success trajectories (full length; root-balanced).
- Exclude sealed / exposed / ind roots (same ban set as P104).
- No C preservation fabrication.
- Record frozen `TRAIN_MANIFEST.json` before training.

## One primary intervention

{change}

Keep P104-B as Arm-Ref (reload v2 ckpt; do not retrain reference). New Arm-P106 changes **only** the primary mechanism above.

## Loss equations

Reference P104-B:

`L_P104B = L_tok_corr + λ_tok_KL * KL(π_P64 || π_new)` on preservation.

P106:

`L = L_tok_corr + λ_tok_KL * L_KL + λ_prim * L_primary`

where `L_primary` is:

{loss_extra}

`L_tok_corr` unchanged from P104 (expert/verified targets; early C Category-C upweight).

## Coefficient calibration

- Training-only, before any CL eval.
- Calibrate `λ_tok_KL` via temperature KL reference (T≠1), as in P104 v2 — **never** uniform logit shift.
- Calibrate `λ_prim` so that at a unit primary deviation (e.g. mean margin deficit or mean proto L2 on P104-B vs P64), `λ_prim * L_primary ≈ strength * L_tok_corr` with strength∈{{2,3}} locked a priori.
- Cap λ at predefined maxima; abort if references invalid.
- **No** post-hoc tuning from eval/CL.

## Evaluation protocol

- Seed: **one** predefined seed (e.g. 0).
- Budget: 2000 updates, LR=1e-3, same early weights as P104.
- Panels: same P98/P104 A1, B1-success, D, B1, C-Feasible, C-train (diagnostic).
- Metric: `SUCCESS_RELAXED_12S` / official R3F.
- Report offline token KL, margin, action-L2 under frozen residual, early C feas, CF on any new A1 loss.

## Success / failure gates

**Success (exploratory):**
1. A1 retained = P64 A1 base (0 losses on protected A1 panel).
2. B1-success and D: 0 losses.
3. Keep ≥1 root-disjoint C-Feasible (or B1) gain vs P64, or at least match P104-B’s C-Feasible gain without A1 loss.
4. Early C feas@z_hat not worse than P104-B by >10 pp.

**Failure:**
- A1 still lost, or C gains disappear with no retention benefit.
- Provenance/parity fail.
- Do **not** automatically stack multiple new losses or unfreeze backbone.

## Compute budget

- 1 seed × 2000 updates (Token-only) + CL eval ≈ same order as P104 Arm B.
- No sweeps.

## Stop conditions

- Stop after one pilot.
- If success: consider limited multi-seed confirm — only with approval.
- If failure: reassess data coverage / temporal skill / whether residual must join — **no** automatic PPO.

## Explicit non-goals

- Not a P64 replacement candidate.
- No sealed independent panel.
- No combining A+B+C+D mechanisms in one run.
"""
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
