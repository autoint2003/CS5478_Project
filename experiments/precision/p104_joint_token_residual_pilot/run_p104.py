"""P104 — Unified token–residual controlled pilot (mechanism validation).

Arm A: immutable P64.
Arm B: token-head SFT only (residual frozen) + token KL retention.
Arm C: token-head + unconditional Δ(h) residual (P101-B style) + token KL + residual MSE retention.

One seed. Backbone frozen. No PPO/GRPO. No sealed ind panel. Does NOT replace P64.
"""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
import re
import sys
import time
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

OUT = ROOT / "results/experiments/p104_joint_token_residual_pilot"
CKPT = OUT / "checkpoints"
LOGS = OUT / "logs"
P98 = ROOT / "results/experiments/p98_unified_b1_c_consolidation"
P87_TEACHER = ROOT / "results/experiments/p87_k48_actor_retention/teacher"
P97 = ROOT / "results/experiments/p97_c_expert_recoverability"
P97_CORR = P97 / "C_CORRECTION_MANIFEST.csv"
P97_NEW = P97 / "new_ics/accepted"
FROZEN_MANIFEST = ROOT / "results/experiments/final_independent_success_rate/FROZEN_TEST_MANIFEST.json"
P93_MAN = ROOT / "results/experiments/p93_frozen_router_independent_validation/EVALUATION_SPLIT_MANIFEST.csv"
P94_SPLITS = ROOT / "results/experiments/p94_unified_multiskill_actor/SPLITS.json"

EXPECTED_SHA = "ec08f59816aa2fdd"
SEED = 0
UPDATES = 2000
LR = 1e-3
EARLY_K = 4
EARLY_W = 3.0
EARLY_C_CAT_C_W = 5.0  # prioritize early Category-C token corrections
GRAD_CLIP = 5.0
ACT_TOL = 1e-5
PARITY_TOL = 1e-6
CL_SEED = 101
CKPT_SELECT_RULE = "final_update_2000_no_cl_selection"
CALIB_STRENGTH_TOK = 3.0
CALIB_STRENGTH_RES = 3.0
CALIB_TEMP_T = 2.0  # non-uniform softmax reference: KL(p || softmax(logits/T)), T!=1
LAMBDA_TOK_MAX = 50.0
LAMBDA_RES_MAX = 50.0
KL_REF_MIN = 1e-4
# Locked schedules (no post-hoc change after CL):
TOKEN_TARGET_RULE = (
    "If expert z_star is M111-feasible for a*: use z_star. "
    "Else if Z nonempty: z_tgt = argmin_k∈Z ||proto[k]-a*||_2 (tie-break by token id; flagged alt_not_expert). "
    "Category D: mask token CE. "
    "No labels from a different physical state."
)
RESIDUAL_TARGET_SCHEDULE = (
    "Arm C: entire training uses teacher-token z_tgt for residual targets "
    "r*=clip(a*-proto[z_tgt],±ρ); deploy uses predicted z_hat. "
    "Mismatch z_hat vs z_tgt measured each update. No post-hoc retargeting."
)
# Offline retention / feasibility notes
TOL_MEAN_RES_L2 = 0.05


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
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def rho_arr(rho) -> np.ndarray:
    return ResidualBudget.from_spec(rho).as_array().astype(np.float64).reshape(7)


def token_can_represent(book, k, a, rho, tol=ACT_TOL) -> bool:
    proto = book[int(k)]
    a = np.asarray(a, np.float64).reshape(7)
    lo = np.maximum(-1.0, proto - rho)
    hi = np.minimum(1.0, proto + rho)
    return bool(np.all(a >= lo - tol) and np.all(a <= hi + tol))


def feasible_tokens(book, a, rho) -> list[int]:
    return [k for k in range(len(book)) if token_can_represent(book, k, a, rho)]


def best_under_token(book, k, a, rho) -> tuple[np.ndarray, float]:
    proto = book[int(k)]
    a = np.asarray(a, np.float64).reshape(7)
    r = np.clip(a - proto, -rho, rho)
    recon = np.clip(proto + r, -1.0, 1.0)
    return r.astype(np.float32), float(np.linalg.norm(recon - a))


def proto_action_l2(book, k, a) -> float:
    """Tie-break among feasible tokens: distance of prototype to action (pre-clip)."""
    proto = np.asarray(book[int(k)], np.float64).reshape(7)
    a = np.asarray(a, np.float64).reshape(7)
    return float(np.linalg.norm(proto - a))


def select_token_target(z_star: int, z_hat: int, a_star: np.ndarray, book, rho) -> dict:
    Z = feasible_tokens(book, a_star, rho)
    if len(Z) == 0:
        return {
            "category": "D",
            "z_tgt": -1,
            "z_hat": int(z_hat),
            "z_star": int(z_star),
            "supervise_token": 0,
            "target_source": "masked_D",
            "n_feasible": 0,
            "alt_not_expert": 0,
        }
    if int(z_hat) in Z:
        cat = "A" if int(z_hat) == int(z_star) else "B"
    else:
        cat = "C"
    if int(z_star) in Z:
        z_tgt = int(z_star)
        src = "expert_verified"
        alt = 0
    else:
        # Among feasible tokens recon error≈0; break ties by ||proto-a||_2 then token id.
        best_k, best_e = None, 1e9
        for k in Z:
            e = proto_action_l2(book, k, a_star)
            if e < best_e - 1e-12 or (abs(e - best_e) <= 1e-12 and (best_k is None or k < best_k)):
                best_e, best_k = e, k
        z_tgt = int(best_k)
        src = "nearest_proto_feasible_alt"
        alt = 1
    return {
        "category": cat,
        "z_tgt": z_tgt,
        "z_hat": int(z_hat),
        "z_star": int(z_star),
        "supervise_token": 1,
        "target_source": src,
        "n_feasible": len(Z),
        "alt_not_expert": alt,
    }


def lookup_starts() -> dict[str, MixedStart]:
    return {s.traj_id: s for s in build_p54_dataset(seed=5478)["schedule"]}


def resolve_c_start(tid: str, pr: str, starts: dict[str, MixedStart]) -> MixedStart | None:
    if tid in starts:
        return starts[tid]
    pkl = P97_NEW / f"{tid}.pkl"
    if pkl.is_file():
        ep = pickle.loads(pkl.read_bytes())["ep"]
        st = MixedStart(skill="C", traj_id=tid, family="C", kind="ep", ep=ep)
        starts[tid] = st
        return st
    base = pr.split("__")[0]
    for t, s in list(starts.items()):
        if s.skill == "C" and phys_root(t) == base:
            starts[tid] = s
            return s
    if P97_CORR.is_file():
        for r in csv.DictReader(P97_CORR.open(encoding="utf-8")):
            if r.get("phys_root") == pr or phys_root(r.get("traj_id") or "") == phys_root(tid):
                tt = r["traj_id"]
                if tt in starts:
                    starts[tid] = starts[tt]
                    return starts[tt]
                p2 = P97_NEW / f"{tt}.pkl"
                if p2.is_file():
                    ep = pickle.loads(p2.read_bytes())["ep"]
                    st = MixedStart(skill="C", traj_id=tt, family="C", kind="ep", ep=ep)
                    starts[tid] = st
                    starts[tt] = st
                    return st
    return None


def eval_banned_phys() -> set[str]:
    tids: set[str] = set()
    man = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
    tids |= {e["traj_id"] for e in man["episodes"]}
    if P93_MAN.is_file():
        for r in csv.DictReader(P93_MAN.open(encoding="utf-8")):
            if r["panel"] in ("B1", "C1", "C2", "A1", "D", "DIAG_EXPOSED_B1"):
                tids.add(r["traj_id"])
    if P94_SPLITS.is_file():
        sp = json.loads(P94_SPLITS.read_text(encoding="utf-8"))
        for block in (sp.get("ind") or {}).values():
            if isinstance(block, list):
                tids |= set(block)
    ban = {phys_root(t) for t in tids}
    ban.add(phys_root("plan_c__gbi_351_bc2d6a3b90"))
    ban.add(phys_root("plan_c__gbi_022_7c876a650b"))
    return ban


def load_correction_demos() -> list[dict]:
    demos = []
    with (P98 / "B1_C_CORRECTION_TARGETS.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            skill = r["skill"]
            tid = r["traj_id"]
            pr = r.get("phys_root") or phys_root(tid)
            path = (P87_TEACHER / f"{tid}.npz") if skill == "B1" else (P98 / "cache" / f"c_corr_{pr}.npz")
            if not path.is_file():
                raise FileNotFoundError(path)
            z = np.load(path, allow_pickle=True)
            demos.append(
                {
                    "traj_id": tid,
                    "phys_root": pr,
                    "skill": skill,
                    "role": "correction",
                    "tokens": np.asarray(z["tokens"], np.int64),
                    "actions": np.asarray(z["actions"], np.float32),
                    "obs": np.asarray(z["obs"], np.float32),
                    "prev": np.asarray(z["prev"], np.float32),
                }
            )
    return demos


def load_preservation_demos() -> list[dict]:
    demos = []
    with (P98 / "B1_A1_D_PRESERVATION_TARGETS.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            skill = r["skill"]
            tid = r["traj_id"]
            pr = r.get("phys_root") or phys_root(tid)
            path = P98 / "cache" / f"pres_{skill}_{pr}.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            z = np.load(path, allow_pickle=True)
            demos.append(
                {
                    "traj_id": tid,
                    "phys_root": pr,
                    "skill": skill,
                    "role": "preservation",
                    "tokens": np.asarray(z["tokens"], np.int64) if "tokens" in z.files else np.zeros(len(z["obs"]), np.int64),
                    "actions": np.asarray(z["actions"], np.float32),
                    "obs": np.asarray(z["obs"], np.float32),
                    "prev": np.asarray(z["prev"], np.float32),
                }
            )
    return demos


def annotate_correction_targets(demos, p64, book, rho, device) -> list[dict]:
    """Precompute per-step ABCD + token targets using P64 z_hat on recorded histories."""
    out = []
    p64.eval()
    for d in demos:
        obs = torch.as_tensor(d["obs"], device=device, dtype=torch.float32)
        prev = torch.as_tensor(d["prev"], device=device, dtype=torch.float32)
        with torch.no_grad():
            logits, _, _ = p64(obs, prev)
            z_hat = torch.argmax(mask_handoff_logits(logits.float()), dim=-1).cpu().numpy()
        step_meta = []
        for t in range(len(d["actions"])):
            meta = select_token_target(int(d["tokens"][t]), int(z_hat[t]), d["actions"][t], book, rho)
            meta["t"] = t
            meta["early"] = int(t < EARLY_K)
            meta["skill"] = d["skill"]
            step_meta.append(meta)
        nd = dict(d)
        nd["step_meta"] = step_meta
        out.append(nd)
    return out


@torch.no_grad()
def closed_loop(actor, start, book, rho, device, *, seed: int = CL_SEED, collect: bool = False) -> dict:
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
    ever12 = False
    toks: list[int] = []
    resids: list[np.ndarray] = []
    while not done:
        obs = torch.as_tensor(np.stack(env.obs_seq[-160:]), device=device, dtype=torch.float32)
        prev = torch.as_tensor(np.stack(env.prev_seq[-160:]), device=device, dtype=torch.float32)
        logits, res, _act = actor(obs, prev)
        tok_a = int(torch.argmax(mask_handoff_logits(logits[-1].float())).item())
        tok, st, _ = select_token(env_t=int(env.t), obs=last_obs(env), actor_tok=tok_a, st=st, v2=v2, rule=None)
        resid = res[-1].detach().cpu().numpy().astype(np.float32)
        if collect:
            toks.append(int(tok))
            resids.append(resid.copy())
        if int(tok) != HANDOFF:
            _ = decode_m111_action(book, int(tok), resid, rho)
        _r, done, info = env.step(int(tok), resid)
        ensure_inner_capture_buffers(env)
        g = relative_geometry(env.env.sim)
        tr = getattr(env.mon, "tracker", None)
        ok_s = float(getattr(tr, "ok_s", 0.0) or 0.0)
        preds = evaluate_predicates(env.env.sim, ok_s=ok_s)
        bil = bool(g["bilateral"])
        pad4 = int(g.get("min_pad") or 0) >= 4
        wk = bool(preds.get("working_ok"))
        e, th = float(g["e_pos_mm"]), float(g["theta_deg"])
        if bil and pad4 and wk and e <= 12 + 1e-9 and th <= 8 + 1e-9:
            ever12 = True
    term = _r3_terminal(env, info)
    out = {
        "SUCCESS_RELAXED_12S": int(term.get("SUCCESS_RELAXED") or info.get("SUCCESS_RELAXED_12S") or 0),
        "r3_ready": int(term.get("r3_ready") or 0),
        "reach12": int(ever12),
        "ep_len": int(env.t),
        "fail_reason": info.get("fail_reason"),
    }
    if collect:
        out["tokens"] = toks
        out["resids"] = resids
    return out


@torch.no_grad()
def closed_loop_swap(token_actor, resid_actor, start, book, rho, device, *, seed: int = CL_SEED) -> dict:
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
    while not done:
        obs = torch.as_tensor(np.stack(env.obs_seq[-160:]), device=device, dtype=torch.float32)
        prev = torch.as_tensor(np.stack(env.prev_seq[-160:]), device=device, dtype=torch.float32)
        logits, _, _ = token_actor(obs, prev)
        _, res, _ = resid_actor(obs, prev)
        tok_a = int(torch.argmax(mask_handoff_logits(logits[-1].float())).item())
        tok, st, _ = select_token(env_t=int(env.t), obs=last_obs(env), actor_tok=tok_a, st=st, v2=v2, rule=None)
        resid = res[-1].detach().cpu().numpy().astype(np.float32)
        _r, done, info = env.step(int(tok), resid)
        ensure_inner_capture_buffers(env)
    term = _r3_terminal(env, info)
    return {
        "SUCCESS_RELAXED_12S": int(term.get("SUCCESS_RELAXED") or info.get("SUCCESS_RELAXED_12S") or 0),
        "ep_len": int(env.t),
        "fail_reason": info.get("fail_reason"),
    }


@torch.no_grad()
def init_parity(p64, actor, demos, device) -> dict:
    d = demos[0]
    obs = torch.as_tensor(d["obs"], device=device, dtype=torch.float32)
    prev = torch.as_tensor(d["prev"], device=device, dtype=torch.float32)
    p64.eval()
    actor.eval()
    la, ra, aa = p64(obs, prev)
    lb, rb, ab = actor(obs, prev)
    za = torch.argmax(mask_handoff_logits(la.float()), dim=-1)
    zb = torch.argmax(mask_handoff_logits(lb.float()), dim=-1)
    out = {
        "mode": actor.mode,
        "max_abs_dz": float((za - zb).abs().max().item()),
        "max_abs_dr": float((ra - rb).abs().max().item()),
        "max_abs_da": float((aa - ab).abs().max().item()),
        "tol": PARITY_TOL,
        "pass": True,
    }
    for k in ("max_abs_dz", "max_abs_dr", "max_abs_da"):
        if out[k] > PARITY_TOL:
            out["pass"] = False
    return out


def token_kl_retention(actor, p64, demo, device) -> tuple[torch.Tensor, dict]:
    """L_token_ret = mean_t KL(pi_P64(·|H_t) || pi_new(·|H_t))."""
    obs = torch.as_tensor(demo["obs"], device=device, dtype=torch.float32)
    prev = torch.as_tensor(demo["prev"], device=device, dtype=torch.float32)
    with torch.no_grad():
        l_ref, r_ref, a_ref = p64(obs, prev)
        p_ref = F.softmax(mask_handoff_logits(l_ref.float()), dim=-1)
    l_new, r_new, a_new = actor(obs, prev)
    log_q = F.log_softmax(mask_handoff_logits(l_new.float()), dim=-1)
    kl = (p_ref * (p_ref.clamp_min(1e-8).log() - log_q)).sum(-1).mean()
    with torch.no_grad():
        act_l2 = ((a_new - a_ref) ** 2).sum(-1).sqrt().mean()
        res_l2 = ((r_new - r_ref) ** 2).sum(-1).sqrt().mean()
    return kl, {"kl": float(kl.detach()), "act_l2": float(act_l2), "res_l2": float(res_l2), "n": int(len(demo["obs"]))}


def residual_mse_retention(actor, p64, demo, device) -> tuple[torch.Tensor, dict]:
    obs = torch.as_tensor(demo["obs"], device=device, dtype=torch.float32)
    prev = torch.as_tensor(demo["prev"], device=device, dtype=torch.float32)
    with torch.no_grad():
        _, r_ref, a_ref = p64(obs, prev)
    _, r_new, a_new = actor(obs, prev)
    loss = ((r_new - r_ref) ** 2).sum(-1).mean()
    with torch.no_grad():
        act_l2 = ((a_new - a_ref) ** 2).sum(-1).sqrt().mean()
        res_l2 = ((r_new - r_ref) ** 2).sum(-1).sqrt().mean()
    return loss, {"mse": float(loss.detach()), "act_l2": float(act_l2), "res_l2": float(res_l2)}


def correction_losses(actor, demo, book, rho, device, *, joint: bool) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Token CE (A/B/C) + optional residual under teacher z_tgt. Mask D."""
    obs = torch.as_tensor(demo["obs"], device=device, dtype=torch.float32)
    prev = torch.as_tensor(demo["prev"], device=device, dtype=torch.float32)
    acts = torch.as_tensor(demo["actions"], device=device, dtype=torch.float32)
    meta = demo["step_meta"]
    logits, res, _ = actor(obs, prev)
    z_hat = torch.argmax(mask_handoff_logits(logits.float()), dim=-1)

    n = len(meta)
    w = torch.zeros(n, device=device, dtype=torch.float32)
    z_tgt = torch.zeros(n, device=device, dtype=torch.long)
    supervise = torch.zeros(n, device=device, dtype=torch.float32)
    n_A = n_B = n_C = n_D = 0
    n_early_C = 0
    for i, m in enumerate(meta):
        cat = m["category"]
        if cat == "A":
            n_A += 1
        elif cat == "B":
            n_B += 1
        elif cat == "C":
            n_C += 1
        else:
            n_D += 1
        if m["supervise_token"]:
            supervise[i] = 1.0
            z_tgt[i] = int(m["z_tgt"])
            wi = 1.0
            if i < EARLY_K:
                wi *= EARLY_W
            if cat == "C" and i < EARLY_K and demo["skill"] == "C":
                wi *= EARLY_C_CAT_C_W
                n_early_C += 1
            w[i] = wi

    logp = F.log_softmax(mask_handoff_logits(logits.float()), dim=-1)
    # gather CE only where supervised
    ce_all = F.nll_loss(logp, z_tgt.clamp(min=0), reduction="none")
    denom_t = (w * supervise).sum().clamp_min(1e-6)
    L_tok = ((ce_all * w * supervise).sum() / denom_t) if float(supervise.sum()) > 0 else torch.zeros((), device=device)

    L_res = torch.zeros((), device=device)
    mismatch = float((z_hat != z_tgt).float().mul(supervise).sum().item())
    n_sup = float(supervise.sum().item())
    if joint and n_sup > 0:
        # residual target consistent with teacher token z_tgt
        motion = torch.clamp(z_tgt, 0, actor.n_motion - 1)
        proto_z = actor.proto[motion]
        rho_t = torch.as_tensor(rho, device=device, dtype=torch.float32)
        r_star = torch.clamp(acts - proto_z, -rho_t, rho_t)
        a_star_dec = torch.clamp(proto_z + r_star, -1.0, 1.0)
        a_pred = torch.clamp(proto_z + res, -1.0, 1.0)
        # mask D already via supervise; also skip if z_tgt invalid
        wm = w * supervise
        denom = wm.sum().clamp_min(1e-6)
        loss_r = (((res - r_star) ** 2).mean(-1) * wm).sum() / denom
        loss_a = (((a_pred - a_star_dec) ** 2).mean(-1) * wm).sum() / denom
        L_res = loss_r + loss_a

    stats = {
        "L_tok": float(L_tok.detach()),
        "L_res": float(L_res.detach()),
        "n_sup": int(n_sup),
        "n_early_C_weighted": n_early_C,
        "A": n_A,
        "B": n_B,
        "C": n_C,
        "D": n_D,
        "teacher_pred_mismatch_frac": mismatch / max(n_sup, 1.0),
    }
    return L_tok, L_res, stats


def sample_balanced_pres(by_skill: dict[str, list], rng: np.random.Generator) -> list[dict]:
    out = []
    for sk in ("A1", "B1", "D"):
        pool = by_skill.get(sk) or []
        if pool:
            out.append(pool[int(rng.integers(0, len(pool)))])
    return out


def calibrate_lambdas(p64, corr_demos, pres_demos, book, rho, device) -> dict:
    """Training-only loss-scale calibration. No eval outcomes.

    Token KL reference uses temperature T!=1 (softmax-changing). Uniform logit
    shifts are forbidden — they leave softmax invariant and explode λ.
    """
    actor_b = build_p104_from_p64(p64, "token_only", device)
    actor_c = build_p104_from_p64(p64, "joint", device)
    b1 = [d for d in corr_demos if d["skill"] == "B1"]
    c = [d for d in corr_demos if d["skill"] == "C"]
    if not b1 or not c:
        raise RuntimeError("calibration requires nonempty B1 and C correction pools")
    # Match training scale: L_corr ≈ L_tok_B + L_tok_C (+ residuals for joint)
    tok_pair_vals = []
    res_pair_vals = []
    for db, dc in zip(b1, c + c[: max(0, len(b1) - len(c))]):
        lt_b, lr_b, sb = correction_losses(actor_b, db, book, rho, device, joint=False)
        lt_c, lr_c, sc = correction_losses(actor_b, dc, book, rho, device, joint=False)
        if sb["n_sup"] + sc["n_sup"] > 0:
            tok_pair_vals.append(float(lt_b.detach()) + float(lt_c.detach()))
        _, lr_bj, _ = correction_losses(actor_c, db, book, rho, device, joint=True)
        _, lr_cj, _ = correction_losses(actor_c, dc, book, rho, device, joint=True)
        res_pair_vals.append(float(lr_bj.detach()) + float(lr_cj.detach()))
    l_tok = float(np.mean(tok_pair_vals)) if tok_pair_vals else 1e-3
    l_res_corr = float(np.mean(res_pair_vals)) if res_pair_vals else l_tok

    by_skill: dict[str, list] = {"A1": [], "B1": [], "D": []}
    for d in demos_group(pres_demos):
        by_skill[d["skill"]].append(d)
    for sk in ("A1", "B1", "D"):
        if not by_skill[sk]:
            raise RuntimeError(f"calibration requires nonempty preservation class {sk}")

    # KL(p || softmax(logits/T)), T=CALIB_TEMP_T — temperature changes the distribution
    T = float(CALIB_TEMP_T)
    if abs(T - 1.0) < 1e-9:
        raise RuntimeError("CALIB_TEMP_T must differ from 1.0")
    kl_refs = []
    for sk, pool in by_skill.items():
        vals = []
        for d in pool:
            obs = torch.as_tensor(d["obs"], device=device, dtype=torch.float32)
            prev = torch.as_tensor(d["prev"], device=device, dtype=torch.float32)
            with torch.no_grad():
                l_ref, _, _ = p64(obs, prev)
                masked = mask_handoff_logits(l_ref.float())
                p_ref = F.softmax(masked, dim=-1)
                log_q = F.log_softmax(masked / T, dim=-1)
                kl = (p_ref * (p_ref.clamp_min(1e-8).log() - log_q)).sum(-1).mean()
                vals.append(float(kl))
        kl_refs.append(float(np.mean(vals)))
    kl_ref = float(np.mean(kl_refs))
    if not np.isfinite(kl_ref) or kl_ref < KL_REF_MIN:
        raise RuntimeError(f"token KL reference too small/invalid: {kl_ref}")
    lam_tok = float(CALIB_STRENGTH_TOK * l_tok / kl_ref)
    if lam_tok > LAMBDA_TOK_MAX:
        lam_tok = float(LAMBDA_TOK_MAX)

    p101b = ROOT / "results/experiments/p101_saup_three_arm/checkpoints/armB_seed0_final.pt"
    if not p101b.is_file():
        raise FileNotFoundError(f"P101-B required for residual λ calibration: {p101b}")
    from models.saup_policy import build_saup_from_p64

    harm = build_saup_from_p64(p64, "B", device)
    blob = torch.load(p101b, map_location=device, weights_only=False)
    harm.load_state_dict(blob["actor"], strict=True)
    harm.eval()
    class_means = []
    for sk, pool in by_skill.items():
        vals = [float(residual_mse_retention(harm, p64, d, device)[0].detach()) for d in pool]
        class_means.append(float(np.mean(vals)))
    res_ref = float(np.mean(class_means))
    if not np.isfinite(res_ref) or res_ref < 1e-8:
        raise RuntimeError(f"residual retention reference invalid: {res_ref}")
    # Training joint correction scale ≈ token pair + residual pair; balance vs harmful L_ret
    scale_corr = max(l_tok + l_res_corr, l_tok)
    lam_res = float(CALIB_STRENGTH_RES * scale_corr / res_ref)
    if lam_res > LAMBDA_RES_MAX:
        lam_res = float(LAMBDA_RES_MAX)

    return {
        "created": utc(),
        "method": "train_only_loss_scale_v2_temperature_kl",
        "mean_L_tok_corr_pair_init": l_tok,
        "kl_ref_temperature": kl_ref,
        "CALIB_TEMP_T": T,
        "lambda_tok_ret": lam_tok,
        "lambda_tok_capped": lam_tok >= LAMBDA_TOK_MAX - 1e-12,
        "mean_L_res_corr_pair_init": l_res_corr,
        "L_res_ret_ref_p101B": res_ref,
        "lambda_res_ret": lam_res,
        "lambda_res_capped": lam_res >= LAMBDA_RES_MAX - 1e-12,
        "CALIB_STRENGTH_TOK": CALIB_STRENGTH_TOK,
        "CALIB_STRENGTH_RES": CALIB_STRENGTH_RES,
        "LAMBDA_TOK_MAX": LAMBDA_TOK_MAX,
        "LAMBDA_RES_MAX": LAMBDA_RES_MAX,
        "TOKEN_TARGET_RULE": TOKEN_TARGET_RULE,
        "RESIDUAL_TARGET_SCHEDULE": RESIDUAL_TARGET_SCHEDULE,
        "note": "No post-hoc coefficient tuning from development/evaluation outcomes. v2 fixes softmax-invariant +1 logit bug.",
    }


def demos_group(demos):
    return demos


def train_arm(
    mode: str,
    seed: int,
    p64,
    book,
    rho,
    corr_demos,
    pres_demos,
    lam_tok: float,
    lam_res: float,
    device,
):
    seed_all(seed, deterministic_torch=True)
    actor = build_p104_from_p64(p64, "token_only" if mode == "B" else "joint", device)
    n_train = actor.trainable_param_count()
    opt = torch.optim.Adam([p for p in actor.parameters() if p.requires_grad], lr=LR)
    b1 = [d for d in corr_demos if d["skill"] == "B1"]
    c = [d for d in corr_demos if d["skill"] == "C"]
    by_skill: dict[str, list] = {"A1": [], "B1": [], "D": []}
    for d in demos_group(pres_demos):
        by_skill[d["skill"]].append(d)
    rng = np.random.default_rng(seed)
    log_rows = []
    actor.train()
    actor.base.eval()  # dropout-less
    if actor.base.training:
        actor.base.eval()
    t0 = time.time()
    joint = mode == "C"
    for u in range(1, UPDATES + 1):
        db = b1[int(rng.integers(0, len(b1)))]
        dc = c[int(rng.integers(0, len(c)))]
        lt_b, lr_b, sb = correction_losses(actor, db, book, rho, device, joint=joint)
        lt_c, lr_c, sc = correction_losses(actor, dc, book, rho, device, joint=joint)
        L_corr = lt_b + lt_c + (lr_b + lr_c if joint else 0.0)

        pres_batch = sample_balanced_pres(by_skill, rng)
        L_tok_ret = torch.zeros((), device=device)
        L_res_ret = torch.zeros((), device=device)
        for dp in pres_batch:
            kli, _ = token_kl_retention(actor, p64, dp, device)
            L_tok_ret = L_tok_ret + kli
            if joint:
                ri, _ = residual_mse_retention(actor, p64, dp, device)
                L_res_ret = L_res_ret + ri
        if pres_batch:
            L_tok_ret = L_tok_ret / len(pres_batch)
            if joint:
                L_res_ret = L_res_ret / len(pres_batch)

        loss = L_corr + float(lam_tok) * L_tok_ret + (float(lam_res) * L_res_ret if joint else 0.0)
        opt.zero_grad(set_to_none=True)
        if float(loss.detach()) > 0:
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in actor.parameters() if p.requires_grad], GRAD_CLIP)
            opt.step()
        if u % 50 == 0 or u == 1 or u == UPDATES:
            row = {
                "arm": mode,
                "seed": seed,
                "update": u,
                "loss": float(loss.detach()),
                "L_corr": float(L_corr.detach()) if torch.is_tensor(L_corr) else float(L_corr),
                "L_tok_B": sb["L_tok"],
                "L_tok_C": sc["L_tok"],
                "L_res_B": sb["L_res"],
                "L_res_C": sc["L_res"],
                "L_tok_ret": float(L_tok_ret.detach()),
                "L_res_ret": float(L_res_ret.detach()),
                "mismatch_C": sc["teacher_pred_mismatch_frac"],
                "elapsed_s": time.time() - t0,
            }
            log_rows.append(row)
            print(
                f"  [P104-{mode} s={seed}] u={u}/{UPDATES} loss={row['loss']:.5f} "
                f"corr~{row['L_corr']:.5f} tokret={row['L_tok_ret']:.5f}",
                flush=True,
            )
    path = CKPT / f"arm{mode}_seed{seed}_final.pt"
    torch.save(
        {
            "arm": mode,
            "mode": actor.mode,
            "seed": seed,
            "update": UPDATES,
            "p64_sha": EXPECTED_SHA,
            "lambda_tok_ret": float(lam_tok),
            "lambda_res_ret": float(lam_res) if joint else None,
            "actor": actor.state_dict(),
            "trainable_params": n_train,
            "ckpt_select_rule": CKPT_SELECT_RULE,
        },
        path,
    )
    actor.eval()
    return actor, log_rows, path, n_train


def load_p104_ckpt(path: Path, p64, mode: str, device):
    actor = build_p104_from_p64(p64, "token_only" if mode == "B" else "joint", device)
    blob = torch.load(path, map_location=device, weights_only=False)
    actor.load_state_dict(blob["actor"], strict=True)
    actor.eval()
    return actor


@torch.no_grad()
def feasibility_transitions(p64, actor, corr_demos, book, rho, device, arm: str) -> list[dict]:
    """Compare P64 vs learned predicted-token feasibility on identical recorded histories."""
    rows = []
    p64.eval()
    actor.eval()
    for d in corr_demos:
        obs = torch.as_tensor(d["obs"], device=device, dtype=torch.float32)
        prev = torch.as_tensor(d["prev"], device=device, dtype=torch.float32)
        l64, r64, a64 = p64(obs, prev)
        ln, rn, an = actor(obs, prev)
        z64 = torch.argmax(mask_handoff_logits(l64.float()), dim=-1).cpu().numpy()
        zn = torch.argmax(mask_handoff_logits(ln.float()), dim=-1).cpu().numpy()
        for t, m in enumerate(d["step_meta"]):
            a_star = d["actions"][t]
            z_star = int(d["tokens"][t])
            cat0 = m["category"]
            feas64 = token_can_represent(book, int(z64[t]), a_star, rho)
            feas_n = token_can_represent(book, int(zn[t]), a_star, rho)
            # transition relative to P64 category on this history
            if cat0 == "C" and feas_n:
                trans = "C_to_feasible"
            elif cat0 in ("A", "B") and not feas_n:
                trans = "AB_to_infeasible"
            elif cat0 == "C" and not feas_n:
                trans = "C_still_infeasible"
            elif cat0 in ("A", "B") and feas_n:
                trans = "AB_still_feasible"
            elif cat0 == "D":
                trans = "D_masked"
            else:
                trans = "other"
            _, e64 = best_under_token(book, int(z64[t]), a_star, rho)
            _, en = best_under_token(book, int(zn[t]), a_star, rho)
            a_err_n = float(np.linalg.norm(an[t].cpu().numpy() - a_star))
            rows.append(
                {
                    "arm": arm,
                    "skill": d["skill"],
                    "traj_id": d["traj_id"],
                    "phys_root": d["phys_root"],
                    "t": t,
                    "early": int(t < EARLY_K),
                    "p64_category": cat0,
                    "z_star": z_star,
                    "z_hat_p64": int(z64[t]),
                    "z_hat_new": int(zn[t]),
                    "z_tgt": m["z_tgt"],
                    "target_source": m["target_source"],
                    "feas_p64": int(feas64),
                    "feas_new": int(feas_n),
                    "transition": trans,
                    "tok_match_expert_new": int(int(zn[t]) == z_star),
                    "best_l2_under_z_hat_p64": e64,
                    "best_l2_under_z_hat_new": en,
                    "a_err_actor_l2": a_err_n,
                    "tok_changed": int(int(z64[t]) != int(zn[t])),
                }
            )
    return rows


@torch.no_grad()
def retention_analysis(p64, actor, demos, device, arm: str) -> list[dict]:
    rows = []
    for d in demos:
        kl, sk = token_kl_retention(actor, p64, d, device)
        if actor.mode == "joint":
            _, sr = residual_mse_retention(actor, p64, d, device)
        else:
            sr = {"mse": 0.0, "res_l2": sk["res_l2"], "act_l2": sk["act_l2"]}
        rows.append(
            {
                "arm": arm,
                "skill": d["skill"],
                "traj_id": d["traj_id"],
                "phys_root": d["phys_root"],
                "n": sk["n"],
                "token_kl": sk["kl"],
                "res_l2": sr["res_l2"],
                "act_l2": sr["act_l2"],
                "res_mse": sr.get("mse", 0.0),
            }
        )
    return rows


def eval_panels(p64, cand_actors: dict, starts, panels, book, rho, device, train_seed: int) -> list[dict]:
    rows = []
    arms = cand_actors if cand_actors else {"A": p64}
    p64_cache: dict[str, dict] = {}
    for arm, actor in arms.items():
        for skill, tids in panels.items():
            for tid in tids:
                st = starts.get(tid)
                if st is None and skill.startswith("C"):
                    st = resolve_c_start(tid, phys_root(tid), starts)
                if st is None:
                    print(f"[warn] missing start {tid}", flush=True)
                    continue
                if tid not in p64_cache:
                    p64_cache[tid] = closed_loop(p64, st, book, rho, device, collect=True)
                r64 = p64_cache[tid]
                if arm == "A":
                    ru = r64
                    tok_div = 0.0
                    first_tok = ""
                    first_res = ""
                else:
                    ru = closed_loop(actor, st, book, rho, device, collect=True)
                    n = min(len(r64.get("tokens", [])), len(ru.get("tokens", [])))
                    tok_div = float("nan")
                    first_tok = ""
                    first_res = ""
                    if n:
                        tok_div = 1.0 - sum(int(r64["tokens"][t] == ru["tokens"][t]) for t in range(n)) / n
                        for t in range(n):
                            if first_tok == "" and r64["tokens"][t] != ru["tokens"][t]:
                                first_tok = t
                            if first_res == "" and float(np.linalg.norm(r64["resids"][t] - ru["resids"][t])) > 1e-5:
                                first_res = t
                rows.append(
                    {
                        "arm": arm,
                        "train_seed": train_seed,
                        "skill": skill,
                        "traj_id": tid,
                        "phys_root": phys_root(tid),
                        "P64_SUCCESS": r64["SUCCESS_RELAXED_12S"],
                        "CAND_SUCCESS": ru["SUCCESS_RELAXED_12S"],
                        "gain": int(ru["SUCCESS_RELAXED_12S"] == 1 and r64["SUCCESS_RELAXED_12S"] == 0),
                        "loss": int(ru["SUCCESS_RELAXED_12S"] == 0 and r64["SUCCESS_RELAXED_12S"] == 1),
                        "P64_ep_len": r64["ep_len"],
                        "CAND_ep_len": ru["ep_len"],
                        "P64_fail": r64.get("fail_reason"),
                        "CAND_fail": ru.get("fail_reason"),
                        "tok_seq_div": tok_div if arm != "A" else 0.0,
                        "first_token_div": first_tok if arm != "A" else "",
                        "first_residual_div": first_res if arm != "A" else "",
                    }
                )
    return rows


def summarize(rows: list[dict], arm: str) -> dict:
    sub = [r for r in rows if r["arm"] == arm]

    def sk(name):
        return [r for r in sub if r["skill"] == name]

    a1 = [r for r in sk("A1") if int(r["P64_SUCCESS"]) == 1]
    d = [r for r in sk("D") if int(r["P64_SUCCESS"]) == 1]
    b1s = [r for r in sk("B1_success") if int(r["P64_SUCCESS"]) == 1]
    b1 = sk("B1")
    b1_fail = [r for r in b1 if int(r["P64_SUCCESS"]) == 0]
    c = sk("C_Feasible")
    ct = sk("C_train")
    return {
        "arm": arm,
        "a1_base": len(a1),
        "a1_lost": sum(1 for r in a1 if int(r["CAND_SUCCESS"]) == 0),
        "a1_retained": sum(1 for r in a1 if int(r["CAND_SUCCESS"]) == 1),
        "d_base": len(d),
        "d_lost": sum(1 for r in d if int(r["CAND_SUCCESS"]) == 0),
        "d_retained": sum(1 for r in d if int(r["CAND_SUCCESS"]) == 1),
        "b1_success_base": len(b1s),
        "b1_success_lost": sum(1 for r in b1s if int(r["CAND_SUCCESS"]) == 0),
        "b1_success_retained": sum(1 for r in b1s if int(r["CAND_SUCCESS"]) == 1),
        "b1_n": len(b1),
        "b1_succ": sum(int(r["CAND_SUCCESS"]) for r in b1),
        "b1_new_recover": sum(1 for r in b1_fail if int(r["CAND_SUCCESS"]) == 1),
        "c_feasible_n": len(c),
        "c_feasible_succ": sum(int(r["CAND_SUCCESS"]) for r in c),
        "c_feasible_gain": sum(int(r["gain"]) for r in c),
        "c_train_n": len(ct),
        "c_train_succ": sum(int(r["CAND_SUCCESS"]) for r in ct),
    }


def decide(feas_rows, sums, paired, *, b1_corr_roots: set[str], c_corr_roots: set[str]) -> dict:
    """Pilot decision labels (one seed; not statistical)."""
    def protected_ok(arm):
        s = sums[arm]
        return s["a1_lost"] == 0 and s["b1_success_lost"] == 0 and s["d_lost"] == 0

    def any_root_disjoint(arm):
        # Prefer paired rows when available: exclude correction training roots
        if paired:
            gains = [
                r
                for r in paired
                if r["arm"] == arm
                and int(r.get("gain", 0)) == 1
                and r["skill"] in ("B1", "C_Feasible")
                and int(r["P64_SUCCESS"]) == 0
            ]
            for r in gains:
                pr = r["phys_root"]
                if r["skill"] == "B1" and pr not in b1_corr_roots:
                    return True
                if r["skill"] == "C_Feasible" and pr not in c_corr_roots:
                    return True
            return False
        s = sums[arm]
        return s["b1_new_recover"] > 0 or s["c_feasible_gain"] > 0

    labels = {}
    for arm in ("B", "C"):
        arm_feas = [
            r
            for r in feas_rows
            if r["arm"] == arm and r["skill"] == "C" and int(r["early"]) == 1 and r["p64_category"] == "C"
        ]
        arm_c2f = sum(1 for r in arm_feas if r["transition"] == "C_to_feasible")
        ab_reg = sum(
            1
            for r in feas_rows
            if r["arm"] == arm and r["skill"] == "C" and int(r["early"]) == 1 and r["transition"] == "AB_to_infeasible"
        )
        if not protected_ok(arm):
            labels[arm] = "RETENTION_FAILURE"
        elif arm_c2f >= 1 and any_root_disjoint(arm) and protected_ok(arm):
            labels[arm] = "MECHANISTIC_AND_CLOSED_LOOP_SUPPORT"
        elif arm_c2f >= 1 and not any_root_disjoint(arm):
            labels[arm] = "TOKEN_LEARNING_WITHOUT_CONTROL_GENERALIZATION"
        elif any_root_disjoint(arm) and protected_ok(arm):
            labels[arm] = "CLOSED_LOOP_GAIN_WITHOUT_CLEAR_TOKEN_MECHANISM"
        else:
            labels[arm] = "NO_MECHANISTIC_TOKEN_GAIN"
        labels[arm + "_early_C_to_feas"] = arm_c2f
        labels[arm + "_early_C_n"] = len(arm_feas)
        labels[arm + "_early_AB_regress"] = ab_reg

    joint_needed = None
    if labels.get("B") in ("NO_MECHANISTIC_TOKEN_GAIN", "TOKEN_LEARNING_WITHOUT_CONTROL_GENERALIZATION") and labels.get(
        "C"
    ) in ("MECHANISTIC_AND_CLOSED_LOOP_SUPPORT", "CLOSED_LOOP_GAIN_WITHOUT_CLEAR_TOKEN_MECHANISM"):
        joint_needed = True
    elif labels.get("B") == labels.get("C"):
        joint_needed = False

    both_fail = all(
        labels[a] in ("NO_MECHANISTIC_TOKEN_GAIN", "RETENTION_FAILURE", "TOKEN_LEARNING_WITHOUT_CONTROL_GENERALIZATION")
        and sums[a]["b1_new_recover"] == 0
        and sums[a]["c_feasible_gain"] == 0
        for a in ("B", "C")
    )

    return {
        "labels": labels,
        "joint_residual_necessary_hint": joint_needed,
        "both_trainable_fail_cl": both_fail,
        "note": "One seed: mechanism validation only; not population inference. C-Feasible n=2 not decisive.",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    print("[P104] Phase 0 — provenance, parity, calibration", flush=True)
    p64_sha = sha16(P64_ACTOR)
    if p64_sha != EXPECTED_SHA:
        write_json(OUT / "SUMMARY.json", {"case": "ABORT", "reason": f"sha {p64_sha}"})
        raise SystemExit(f"P64 sha mismatch: {p64_sha}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p64, book, rho, saved = load_actor(P64_ACTOR, device)
    p64.eval()
    for p in p64.parameters():
        p.requires_grad_(False)
    book = np.asarray(book, np.float64)
    rho = rho_arr(rho)

    corr_raw = load_correction_demos()
    pres_demos = load_preservation_demos()
    ban = eval_banned_phys()
    corr_roots = {phys_root(d["phys_root"]) for d in corr_raw}
    for d in corr_raw:
        d["phys_root"] = phys_root(d["phys_root"])
    for d in demos_group(pres_demos):
        d["phys_root"] = phys_root(d["phys_root"])
    pres_roots = {d["phys_root"] for d in demos_group(pres_demos)}
    leak = sorted((corr_roots | pres_roots) & ban)
    if leak:
        write_json(OUT / "SUMMARY.json", {"case": "ABORT", "reason": "sealed_leak", "leak": leak})
        raise SystemExit(f"sealed leak: {leak}")
    if any(d["skill"] == "C" for d in demos_group(pres_demos)):
        raise SystemExit("C preservation must not exist")
    corr_keys = {(d["skill"], d["phys_root"]) for d in corr_raw}
    preserv_keys = {(d["skill"], d["phys_root"]) for d in demos_group(pres_demos)}
    role_overlap = sorted(corr_keys & preserv_keys)
    if role_overlap:
        write_json(OUT / "SUMMARY.json", {"case": "ABORT", "reason": "role_overlap", "overlap": role_overlap})
        raise SystemExit(f"correction∩preservation skill-root overlap: {role_overlap}")
    b1_corr_roots = {d["phys_root"] for d in corr_raw if d["skill"] == "B1"}
    c_corr_roots = {d["phys_root"] for d in corr_raw if d["skill"] == "C"}
    if not b1_corr_roots or not c_corr_roots:
        raise SystemExit("empty B1 or C correction pool")
    if not all(any(d["skill"] == sk for d in demos_group(pres_demos)) for sk in ("A1", "B1", "D")):
        raise SystemExit("missing A1/B1/D preservation class")

    corr_demos = annotate_correction_targets(corr_raw, p64, book, rho, device)
    # provenance of targets
    tgt_src = {}
    cat_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    early_c_cat_c = 0
    for d in corr_demos:
        for m in d["step_meta"]:
            cat_counts[m["category"]] = cat_counts.get(m["category"], 0) + 1
            tgt_src[m["target_source"]] = tgt_src.get(m["target_source"], 0) + 1
            if d["skill"] == "C" and m["early"] and m["category"] == "C":
                early_c_cat_c += 1

    arm_b0 = build_p104_from_p64(p64, "token_only", device)
    arm_c0 = build_p104_from_p64(p64, "joint", device)
    parity = {
        "B": init_parity(p64, arm_b0, corr_demos, device),
        "C": init_parity(p64, arm_c0, corr_demos, device),
    }
    write_json(OUT / "INIT_PARITY.json", parity)
    if not parity["B"]["pass"] or not parity["C"]["pass"]:
        write_json(OUT / "SUMMARY.json", {"case": "ABORT", "reason": "parity", "parity": parity})
        raise SystemExit("init parity failed")

    calib = calibrate_lambdas(p64, corr_demos, demos_group(pres_demos), book, rho, device)
    lam_tok = float(calib["lambda_tok_ret"])
    lam_res = float(calib["lambda_res_ret"])
    print(f"[P104] λ_tok_ret={lam_tok:.6g} λ_res_ret={lam_res:.6g}", flush=True)

    splits = json.loads((P98 / "SPLITS.json").read_text(encoding="utf-8"))
    provenance = {
        "created": utc(),
        "p64_sha": p64_sha,
        "correction_roots": sorted(corr_roots),
        "preservation_roots": sorted(pres_roots),
        "banned_overlap": leak,
        "role_skill_root_overlap": role_overlap,
        "c_preservation_count": 0,
        "abcd_counts_p64_on_corr": cat_counts,
        "token_target_sources": tgt_src,
        "early_C_category_C_steps": early_c_cat_c,
        "TOKEN_TARGET_RULE": TOKEN_TARGET_RULE,
        "RESIDUAL_TARGET_SCHEDULE": RESIDUAL_TARGET_SCHEDULE,
        "note": "P98 train-eligible B1/C correction + A1/B1/D P64-success preservation. No C preservation. Sealed/ind excluded.",
    }
    write_json(OUT / "PROVENANCE_AUDIT.json", provenance)

    manifest = {
        "created": utc(),
        "p64_sha": p64_sha,
        "seed": SEED,
        "updates": UPDATES,
        "lr": LR,
        "early_k": EARLY_K,
        "early_w": EARLY_W,
        "early_c_cat_c_w": EARLY_C_CAT_C_W,
        "grad_clip": GRAD_CLIP,
        "ckpt_select_rule": CKPT_SELECT_RULE,
        "backbone_frozen": True,
        "arms": {
            "A": "immutable_P64",
            "B": "token_head_only + token_KL_retention",
            "C": "token_head + unconditional_Delta_h + token_KL + residual_MSE_retention",
        },
        "lambda_tok_ret": lam_tok,
        "lambda_res_ret": lam_res,
        "lambda_calibration": calib,
        "TOKEN_TARGET_RULE": TOKEN_TARGET_RULE,
        "RESIDUAL_TARGET_SCHEDULE": RESIDUAL_TARGET_SCHEDULE,
        "correction_demos": [
            {"skill": d["skill"], "traj_id": d["traj_id"], "phys_root": d["phys_root"], "n": int(len(d["obs"]))}
            for d in corr_demos
        ],
        "preservation_demos": [
            {"skill": d["skill"], "traj_id": d["traj_id"], "phys_root": d["phys_root"], "n": int(len(d["obs"]))}
            for d in demos_group(pres_demos)
        ],
        "no_ppo_grpo": True,
        "p64_immutable": True,
        "mechanism_validation_only": True,
    }
    write_json(OUT / "TRAIN_MANIFEST.json", manifest)

    starts = lookup_starts()
    panel_dev = {
        "B1": list(splits["dev"]["B1"]),
        "C_Feasible": list(splits["dev"]["C"]),
        "A1": list(splits["dev"]["A1"]),
        "D": list(splits["dev"]["D"]),
        "B1_success": [d["traj_id"] for d in demos_group(pres_demos) if d["skill"] == "B1"],
        "C_train": [d["traj_id"] for d in corr_demos if d["skill"] == "C"],
    }
    for tid in panel_dev["C_Feasible"] + panel_dev["C_train"]:
        resolve_c_start(tid, phys_root(tid), starts)

    print("[P104] Phase 1 — train Arms B and C (seed=0)", flush=True)
    all_logs = []
    ckpts = {}
    nparams = {}
    for mode in ("B", "C"):
        print(f"[P104] train Arm {mode}", flush=True)
        actor, logs, path, ntr = train_arm(
            mode, SEED, p64, book, rho, corr_demos, demos_group(pres_demos), lam_tok, lam_res, device
        )
        all_logs.extend(logs)
        ckpts[mode] = str(path)
        nparams[mode] = ntr
        del actor
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_csv(LOGS / "TRAIN_CURVES.csv", all_logs)

    print("[P104] Phase 2 — diagnostics + closed-loop", flush=True)
    feas_rows = []
    ret_rows = []
    paired = []

    print("[P104] CL Arm A", flush=True)
    rows_a = eval_panels(p64, {}, starts, panel_dev, book, rho, device, train_seed=-1)
    paired.extend(rows_a)
    sums = {"A": summarize(rows_a, "A")}

    actors = {}
    for mode in ("B", "C"):
        print(f"[P104] diagnostics+CL Arm {mode}", flush=True)
        actor = load_p104_ckpt(Path(ckpts[mode]), p64, mode, device)
        actors[mode] = actor
        feas_rows.extend(feasibility_transitions(p64, actor, corr_demos, book, rho, device, mode))
        ret_rows.extend(retention_analysis(p64, actor, demos_group(pres_demos), device, mode))
        rows = eval_panels(p64, {mode: actor}, starts, panel_dev, book, rho, device, train_seed=SEED)
        rows = [r for r in rows if r["arm"] == mode]
        paired.extend(rows)
        sums[mode] = summarize(rows, mode)

        # counterfactual rescue on newly lost protected
        lost = [r for r in rows if r["skill"] in ("A1", "B1_success", "D") and int(r["loss"]) == 1]
        for r in lost:
            st = starts.get(r["traj_id"])
            if st is None:
                continue
            # token from candidate, residual from P64
            swap = closed_loop_swap(actor, p64, st, book, rho, device)
            r["swap_candTok_P64res_SUCCESS"] = swap["SUCCESS_RELAXED_12S"]
            # token from P64, residual from candidate (joint residual role)
            swap2 = closed_loop_swap(p64, actor, st, book, rho, device)
            r["swap_P64tok_candRes_SUCCESS"] = swap2["SUCCESS_RELAXED_12S"]

        print(
            f"[P104] Arm {mode}: a1_lost={sums[mode]['a1_lost']} b1s_lost={sums[mode]['b1_success_lost']} "
            f"d_lost={sums[mode]['d_lost']} b1_new={sums[mode]['b1_new_recover']} "
            f"c_feas={sums[mode]['c_feasible_succ']}",
            flush=True,
        )

    write_csv(OUT / "TOKEN_FEASIBILITY_TRANSITIONS.csv", feas_rows)
    write_csv(OUT / "RETENTION_ANALYSIS.csv", ret_rows)
    write_csv(OUT / "PAIRED_CLOSED_LOOP.csv", paired)

    decision = decide(feas_rows, sums, paired, b1_corr_roots=b1_corr_roots, c_corr_roots=c_corr_roots)

    # Aggregate early-C feasibility rates
    def early_c_stats(arm):
        sub = [
            r
            for r in feas_rows
            if r["arm"] == arm and r["skill"] == "C" and int(r["early"]) == 1
        ]
        n = len(sub) or 1
        return {
            "n": len(sub),
            "feas_p64_rate": sum(int(r["feas_p64"]) for r in sub) / n,
            "feas_new_rate": sum(int(r["feas_new"]) for r in sub) / n,
            "C_to_feasible": sum(1 for r in sub if r["transition"] == "C_to_feasible"),
            "AB_to_infeasible": sum(1 for r in sub if r["transition"] == "AB_to_infeasible"),
            "tok_acc_expert": sum(int(r["tok_match_expert_new"]) for r in sub) / n,
            "mean_a_err": float(np.mean([r["a_err_actor_l2"] for r in sub])) if sub else float("nan"),
        }

    ec_b, ec_c = early_c_stats("B"), early_c_stats("C")

    def mean_ret(arm):
        sub = [r for r in ret_rows if r["arm"] == arm]
        if not sub:
            return {}
        return {
            "token_kl": float(np.mean([r["token_kl"] for r in sub])),
            "res_l2": float(np.mean([r["res_l2"] for r in sub])),
            "act_l2": float(np.mean([r["act_l2"] for r in sub])),
        }

    ret_b, ret_c = mean_ret("B"), mean_ret("C")

    # Answer 6 questions
    q1 = (ec_b["C_to_feasible"] + ec_c["C_to_feasible"]) > 0
    q2 = sums["B"]["c_feasible_gain"] > 0 or sums["C"]["c_feasible_gain"] > 0 or sums["B"]["b1_new_recover"] > 0 or sums["C"]["b1_new_recover"] > 0
    q3 = decision["joint_residual_necessary_hint"]
    q4 = sums["B"]["a1_lost"] == 0 and sums["B"]["b1_success_lost"] == 0 and sums["B"]["d_lost"] == 0 and sums["C"]["a1_lost"] == 0 and sums["C"]["b1_success_lost"] == 0 and sums["C"]["d_lost"] == 0
    # limitation taxonomy
    d_frac = cat_counts.get("D", 0) / max(sum(cat_counts.values()), 1)
    if decision["both_trainable_fail_cl"] and ec_c["C_to_feasible"] == 0 and ec_b["C_to_feasible"] == 0:
        if d_frac > 0.3:
            limitation = "M111_representability_and_or_insufficient_token_learning"
        else:
            limitation = "token_learning_or_insufficient_data_coverage"
    elif q1 and not q2:
        limitation = "token_learning_without_control_generalization_residual_compat_or_data"
    elif not q4:
        limitation = "retention_failure"
    else:
        limitation = "mixed_or_inconclusive_one_seed"
    q6 = (
        "If both fail: reassess training coverage and temporal skill representation before unfreezing backbone; "
        "consider more verified C Category-C states, not automatic PPO. "
        "If token feas↑ without CL: expand matched-state coverage / on-policy correction. "
        "If joint>>token-only: residual adaptation is necessary for action fidelity under new tokens."
    )

    summary = {
        "case": "P104_COMPLETE",
        "created": utc(),
        "p64_sha": p64_sha,
        "p64_replaced": False,
        "seed": SEED,
        "lambda_tok_ret": lam_tok,
        "lambda_res_ret": lam_res,
        "trainable_params": nparams,
        "ckpt_paths": ckpts,
        "init_parity": parity,
        "arms": sums,
        "early_c_feasibility": {"B": ec_b, "C": ec_c},
        "retention_offline": {"B": ret_b, "C": ret_c},
        "decision": decision,
        "c_unfiltered": "N/A",
        "questions": {
            "1_token_corrects_early_C_infeasible": q1,
            "2_correction_survives_closed_loop": q2,
            "3_joint_residual_necessary": q3,
            "4_old_skills_preserved": q4,
            "5_limitation": limitation,
            "6_next_phase_guidance": q6,
        },
        "abcd_train_counts": cat_counts,
        "mechanism_validation_only": True,
    }
    write_json(OUT / "SUMMARY.json", summary)

    report = f"""# FINAL_REPORT — P104 Unified Token–Residual Controlled Pilot

Generated: {utc()}

## Setup

- P64 sha `{p64_sha}` (**immutable**; not replaced). Mechanism validation only — **not** a replacement candidate.
- Backbone frozen. One seed (`{SEED}`). Updates={UPDATES}. LR={LR}.
- Arm A: frozen P64. Arm B: token-head SFT + KL retention. Arm C: token + unconditional Δ(h) + KL + residual MSE retention.
- `λ_tok_ret={lam_tok:.6g}`, `λ_res_ret={lam_res:.6g}` (training-only calibration).
- Token targets: {TOKEN_TARGET_RULE}
- Residual schedule: {RESIDUAL_TARGET_SCHEDULE}
- C-Unfiltered: **N/A** (sealed independent panel not opened). C-Feasible n=2 **not decisive**.

## Pre-checks

| Check | Result |
|---|---|
| SHA / parity B / parity C | pass / {parity['B']['pass']} / {parity['C']['pass']} |
| Sealed leak | {leak or 'none'} |
| ABCD on correction (P64 z_hat) | {cat_counts} |
| Early C Category-C steps | {early_c_cat_c} |
| Target sources | {tgt_src} |

## Early C token feasibility (recorded histories; skill=C, t<4)

| Arm | n | feas@P64 | feas@new | C→feasible | AB→infeasible | tok-acc vs expert | mean a_err |
|---|---:|---:|---:|---:|---:|---:|---:|
| B | {ec_b['n']} | {ec_b['feas_p64_rate']:.2f} | {ec_b['feas_new_rate']:.2f} | {ec_b['C_to_feasible']} | {ec_b['AB_to_infeasible']} | {ec_b['tok_acc_expert']:.2f} | {ec_b['mean_a_err']:.4f} |
| C | {ec_c['n']} | {ec_c['feas_p64_rate']:.2f} | {ec_c['feas_new_rate']:.2f} | {ec_c['C_to_feasible']} | {ec_c['AB_to_infeasible']} | {ec_c['tok_acc_expert']:.2f} | {ec_c['mean_a_err']:.4f} |

Improved feasibility ≠ closed-loop recovery.

## Offline retention (P64-success A1/B1/D histories)

| Arm | token KL | res L2 | act L2 |
|---|---:|---:|---:|
| B | {ret_b.get('token_kl', float('nan')):.4f} | {ret_b.get('res_l2', float('nan')):.4f} | {ret_b.get('act_l2', float('nan')):.4f} |
| C | {ret_c.get('token_kl', float('nan')):.4f} | {ret_c.get('res_l2', float('nan')):.4f} | {ret_c.get('act_l2', float('nan')):.4f} |

## Closed-loop SUCCESS_RELAXED_12S (seed={SEED})

| Arm | A1 ret/base (lost) | B1-succ ret/base (lost) | D ret/base (lost) | B1 new | C-Feasible | C-train |
|---|---:|---:|---:|---:|---:|---:|
| A | {sums['A']['a1_retained']}/{sums['A']['a1_base']} ({sums['A']['a1_lost']}) | {sums['A']['b1_success_retained']}/{sums['A']['b1_success_base']} ({sums['A']['b1_success_lost']}) | {sums['A']['d_retained']}/{sums['A']['d_base']} ({sums['A']['d_lost']}) | {sums['A']['b1_new_recover']} | {sums['A']['c_feasible_succ']}/{sums['A']['c_feasible_n']} | {sums['A']['c_train_succ']}/{sums['A']['c_train_n']} |
| B | {sums['B']['a1_retained']}/{sums['B']['a1_base']} ({sums['B']['a1_lost']}) | {sums['B']['b1_success_retained']}/{sums['B']['b1_success_base']} ({sums['B']['b1_success_lost']}) | {sums['B']['d_retained']}/{sums['B']['d_base']} ({sums['B']['d_lost']}) | {sums['B']['b1_new_recover']} | {sums['B']['c_feasible_succ']}/{sums['B']['c_feasible_n']} | {sums['B']['c_train_succ']}/{sums['B']['c_train_n']} |
| C | {sums['C']['a1_retained']}/{sums['C']['a1_base']} ({sums['C']['a1_lost']}) | {sums['C']['b1_success_retained']}/{sums['C']['b1_success_base']} ({sums['C']['b1_success_lost']}) | {sums['C']['d_retained']}/{sums['C']['d_base']} ({sums['C']['d_lost']}) | {sums['C']['b1_new_recover']} | {sums['C']['c_feasible_succ']}/{sums['C']['c_feasible_n']} | {sums['C']['c_train_succ']}/{sums['C']['c_train_n']} |

## Decision labels (exploratory, one seed)

| Arm | Label |
|---|---|
| B | **{decision['labels'].get('B')}** (early C→feas={decision['labels'].get('B_early_C_to_feas')}/{decision['labels'].get('B_early_C_n')}) |
| C | **{decision['labels'].get('C')}** (early C→feas={decision['labels'].get('C_early_C_to_feas')}/{decision['labels'].get('C_early_C_n')}) |

Joint residual necessary (hint): {decision['joint_residual_necessary_hint']}. Both fail CL gains: {decision['both_trainable_fail_cl']}.

## Answers

1. **Can token-head learning correct early C infeasible-token choices?** {q1}
2. **Does the correction survive physical closed-loop execution?** {q2}
3. **Is joint residual adaptation necessary?** {q3}
4. **Are old successful skills preserved?** {q4}
5. **Limitation?** {limitation}
6. **Next phase?** {q6}

## Artifacts

`results/experiments/p104_joint_token_residual_pilot/`: SUMMARY.json, FINAL_REPORT.md, TRAIN_MANIFEST.json, INIT_PARITY.json, PROVENANCE_AUDIT.json, TOKEN_FEASIBILITY_TRANSITIONS.csv, PAIRED_CLOSED_LOOP.csv, RETENTION_ANALYSIS.csv, checkpoints/, logs/.

**STOP after P104.** Do not launch P105 or replace P64 without approval.
"""
    (OUT / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    print(report, flush=True)
    print(f"[P104] DONE B={decision['labels'].get('B')} C={decision['labels'].get('C')}", flush=True)


if __name__ == "__main__":
    main()
