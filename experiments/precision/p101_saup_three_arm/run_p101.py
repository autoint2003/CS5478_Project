"""P101 — SAUP three-arm controlled exploratory experiment.

Arm A: immutable P64.
Arm B: preactivation + unconditional Δ(h).
Arm C: preactivation + token-conditioned Δ(h, z_hat).

Frozen backbone/token/proto/res-preact. Feasibility-gated frozen-z_hat supervision.
Does NOT replace P64, unfreeze tokens/backbone, change M111/R3F, or run PPO/GRPO.
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
from models.saup_policy import build_saup_from_p64, count_delta_params  # noqa: E402
from training.skill_aware_action_vocab import ResidualBudget  # noqa: E402

OUT = ROOT / "results/experiments/p101_saup_three_arm"
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
SEEDS = [0, 1, 2]
UPDATES = 2000
LR = 1e-3
EARLY_K = 4
EARLY_W = 3.0
GRAD_CLIP = 5.0
ACT_TOL = 1e-5
PARITY_TOL = 1e-6
CL_SEED = 101
# Checkpoint selection (locked before training): final step only — no CL-based selection.
CKPT_SELECT_RULE = "final_update_2000_no_cl_selection"


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
    a = ResidualBudget.from_spec(rho).as_array().astype(np.float64).reshape(7)
    return a


def token_can_represent(book: np.ndarray, k: int, a: np.ndarray, rho: np.ndarray, tol: float = ACT_TOL) -> bool:
    proto = book[int(k)]
    a = np.asarray(a, np.float64).reshape(7)
    lo = np.maximum(-1.0, proto - rho)
    hi = np.minimum(1.0, proto + rho)
    return bool(np.all(a >= lo - tol) and np.all(a <= hi + tol))


def feasible_mask_np(book, z_hat, actions, rho) -> np.ndarray:
    out = np.zeros(len(actions), dtype=bool)
    for i in range(len(actions)):
        out[i] = token_can_represent(book, int(z_hat[i]), actions[i], rho)
    return out


def any_token_feasible(book, a, rho) -> bool:
    return any(token_can_represent(book, k, a, rho) for k in range(len(book)))


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


def load_train_demos() -> list[dict]:
    demos = []
    with (P98 / "B1_C_CORRECTION_TARGETS.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            skill = r["skill"]
            tid = r["traj_id"]
            pr = r.get("phys_root") or phys_root(tid)
            if skill == "B1":
                path = P87_TEACHER / f"{tid}.npz"
            else:
                path = P98 / "cache" / f"c_corr_{pr}.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            z = np.load(path, allow_pickle=True)
            demos.append(
                {
                    "traj_id": tid,
                    "phys_root": pr,
                    "skill": skill,
                    "tokens": np.asarray(z["tokens"], np.int64),
                    "actions": np.asarray(z["actions"], np.float32),
                    "obs": np.asarray(z["obs"], np.float32),
                    "prev": np.asarray(z["prev"], np.float32),
                }
            )
    return demos


@torch.no_grad()
def closed_loop(actor, start, book, rho, device, *, seed: int = CL_SEED) -> dict:
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
    while not done:
        obs = torch.as_tensor(np.stack(env.obs_seq[-160:]), device=device, dtype=torch.float32)
        prev = torch.as_tensor(np.stack(env.prev_seq[-160:]), device=device, dtype=torch.float32)
        logits, res, _act = actor(obs, prev)
        tok_a = int(torch.argmax(mask_handoff_logits(logits[-1].float())).item())
        tok, st, _ = select_token(env_t=int(env.t), obs=last_obs(env), actor_tok=tok_a, st=st, v2=v2, rule=None)
        resid = res[-1].detach().cpu().numpy().astype(np.float32)
        # verify decode contract used by env
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
    return {
        "SUCCESS_RELAXED_12S": int(term.get("SUCCESS_RELAXED") or info.get("SUCCESS_RELAXED_12S") or 0),
        "r3_ready": int(term.get("r3_ready") or 0),
        "reach12": int(ever12),
        "ep_len": int(env.t),
        "fail_reason": info.get("fail_reason"),
    }


@torch.no_grad()
def init_parity(p64, arm_b, arm_c, demos, device) -> dict:
    # use first demo full trajectory
    d = demos[0]
    obs = torch.as_tensor(d["obs"], device=device, dtype=torch.float32)
    prev = torch.as_tensor(d["prev"], device=device, dtype=torch.float32)
    p64.eval()
    arm_b.eval()
    arm_c.eval()
    la, ra, aa = p64(obs, prev)
    lb, rb, ab = arm_b(obs, prev)
    lc, rc, ac = arm_c(obs, prev)
    za = torch.argmax(mask_handoff_logits(la.float()), dim=-1)
    zb = torch.argmax(mask_handoff_logits(lb.float()), dim=-1)
    zc = torch.argmax(mask_handoff_logits(lc.float()), dim=-1)
    out = {
        "max_abs_dz_B": float((za - zb).abs().max().item()),
        "max_abs_dz_C": float((za - zc).abs().max().item()),
        "max_abs_dr_B": float((ra - rb).abs().max().item()),
        "max_abs_dr_C": float((ra - rc).abs().max().item()),
        "max_abs_da_B": float((aa - ab).abs().max().item()),
        "max_abs_da_C": float((aa - ac).abs().max().item()),
        "tol": PARITY_TOL,
        "pass": True,
    }
    for k in ("max_abs_dz_B", "max_abs_dz_C", "max_abs_dr_B", "max_abs_dr_C", "max_abs_da_B", "max_abs_da_C"):
        if out[k] > PARITY_TOL:
            out["pass"] = False
    return out


def feasibility_gated_loss(actor, demo, book, rho, device) -> tuple[torch.Tensor, dict]:
    obs = torch.as_tensor(demo["obs"], device=device, dtype=torch.float32)
    prev = torch.as_tensor(demo["prev"], device=device, dtype=torch.float32)
    acts = torch.as_tensor(demo["actions"], device=device, dtype=torch.float32)
    with torch.no_grad():
        logits, _, _ = actor(obs, prev)
        z_hat = torch.argmax(mask_handoff_logits(logits.float()), dim=-1)
        z_np = z_hat.detach().cpu().numpy()
        a_np = demo["actions"]
        feas = feasible_mask_np(book, z_np, a_np, rho)
        # classify ABCD for logging
        n_A = n_B = n_C = n_D = 0
        for i in range(len(a_np)):
            z_star = int(demo["tokens"][i])
            zh = int(z_np[i])
            in_z = bool(feas[i])
            glob = any_token_feasible(book, a_np[i], rho)
            if not glob:
                n_D += 1
            elif not in_z:
                n_C += 1
            elif zh == z_star:
                n_A += 1
            else:
                n_B += 1
    if not feas.any():
        zero = torch.zeros((), device=device)
        return zero, {"loss": 0.0, "n_feas": 0, "n": len(a_np), "A": n_A, "B": n_B, "C": n_C, "D": n_D, "res": 0.0, "act": 0.0}

    # targets relative to z_hat
    motion = torch.clamp(z_hat, 0, actor.n_motion - 1)
    proto_z = actor.proto[motion]
    rho_t = torch.as_tensor(rho, device=device, dtype=torch.float32)
    r_star = torch.clamp(acts - proto_z, -rho_t, rho_t)
    a_star_dec = torch.clamp(proto_z + r_star, -1.0, 1.0)

    logits, res, _ = actor(obs, prev, z_cond=z_hat)
    # action under z_hat + predicted residual
    a_pred = torch.clamp(proto_z + res, -1.0, 1.0)

    w = torch.ones(len(acts), device=device, dtype=torch.float32)
    w[: min(EARLY_K, len(acts))] = EARLY_W
    m = torch.as_tensor(feas, device=device, dtype=torch.float32)
    wm = w * m
    denom = wm.sum().clamp_min(1e-6)
    loss_res = (((res - r_star) ** 2).mean(-1) * wm).sum() / denom
    loss_act = (((a_pred - a_star_dec) ** 2).mean(-1) * wm).sum() / denom
    loss = loss_res + loss_act
    return loss, {
        "loss": float(loss.detach()),
        "res": float(loss_res.detach()),
        "act": float(loss_act.detach()),
        "n_feas": int(feas.sum()),
        "n": int(len(acts)),
        "A": n_A,
        "B": n_B,
        "C": n_C,
        "D": n_D,
    }


def train_arm(arm_name: str, seed: int, p64, book, rho, demos, device) -> tuple:
    seed_all(seed, deterministic_torch=True)
    actor = build_saup_from_p64(p64, "B" if arm_name == "B" else "C", device)
    n_train = actor.trainable_param_count()
    opt = torch.optim.Adam([p for p in actor.parameters() if p.requires_grad], lr=LR)
    b1 = [d for d in demos if d["skill"] == "B1"]
    c = [d for d in demos if d["skill"] == "C"]
    rng = np.random.default_rng(seed)
    log_rows = []
    actor.train()
    # keep base in eval for dropout-less determinism (dropout=0 anyway)
    actor.base.eval()
    t0 = time.time()
    for u in range(1, UPDATES + 1):
        db = b1[int(rng.integers(0, len(b1)))]
        dc = c[int(rng.integers(0, len(c)))]
        lb, sb = feasibility_gated_loss(actor, db, book, rho, device)
        lc, sc = feasibility_gated_loss(actor, dc, book, rho, device)
        loss = lb + lc
        opt.zero_grad(set_to_none=True)
        if float(loss.detach()) > 0:
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in actor.parameters() if p.requires_grad], GRAD_CLIP)
            opt.step()
        if u % 50 == 0 or u == 1 or u == UPDATES:
            row = {
                "arm": arm_name,
                "seed": seed,
                "update": u,
                "loss": float(loss.detach()),
                "L_B": sb["loss"],
                "L_C": sc["loss"],
                "feas_B": sb["n_feas"],
                "feas_C": sc["n_feas"],
                "elapsed_s": time.time() - t0,
            }
            log_rows.append(row)
            print(
                f"  [{arm_name} s={seed}] u={u}/{UPDATES} loss={row['loss']:.4f} "
                f"B={row['L_B']:.4f} C={row['L_C']:.4f}",
                flush=True,
            )
    path = CKPT / f"arm{arm_name}_seed{seed}_final.pt"
    torch.save(
        {
            "arm": arm_name,
            "seed": seed,
            "update": UPDATES,
            "p64_sha": EXPECTED_SHA,
            "actor": actor.state_dict(),
            "trainable_params": n_train,
            "ckpt_select_rule": CKPT_SELECT_RULE,
        },
        path,
    )
    actor.eval()
    return actor, log_rows, path, n_train


def load_saup_ckpt(path: Path, p64, arm: str, device):
    actor = build_saup_from_p64(p64, "B" if arm == "B" else "C", device)
    blob = torch.load(path, map_location=device, weights_only=False)
    actor.load_state_dict(blob["actor"], strict=True)
    actor.eval()
    return actor


def early_token_feasibility(actor, demos, book, rho, device) -> list[dict]:
    rows = []
    actor.eval()
    with torch.no_grad():
        for d in demos:
            obs = torch.as_tensor(d["obs"], device=device, dtype=torch.float32)
            prev = torch.as_tensor(d["prev"], device=device, dtype=torch.float32)
            logits, res, _ = actor(obs, prev)
            z_hat = torch.argmax(mask_handoff_logits(logits.float()), dim=-1).cpu().numpy()
            for t in range(min(5, len(d["actions"]))):
                a = d["actions"][t]
                zh = int(z_hat[t])
                z_star = int(d["tokens"][t])
                glob = any_token_feasible(book, a, rho)
                under = token_can_represent(book, zh, a, rho)
                if not glob:
                    cat = "D"
                elif not under:
                    cat = "C"
                elif zh == z_star:
                    cat = "A"
                else:
                    cat = "B"
                proto = book[zh]
                rbest = np.clip(a - proto, -rho, rho)
                recon = np.clip(proto + rbest, -1, 1)
                a_p64 = np.clip(proto + np.clip(res[t].cpu().numpy(), -rho, rho), -1, 1)
                rows.append(
                    {
                        "traj_id": d["traj_id"],
                        "phys_root": d["phys_root"],
                        "skill": d["skill"],
                        "t": t,
                        "category": cat,
                        "z_hat": zh,
                        "z_star": z_star,
                        "feasible_under_z_hat": int(under),
                        "feasible_any_m111": int(glob),
                        "best_l2_under_z_hat": float(np.linalg.norm(recon - a)),
                        "a_err_actor_l2": float(np.linalg.norm(a_p64 - a)),
                    }
                )
    return rows


def eval_panels(actors: dict, starts, panels: dict, book, rho, device) -> list[dict]:
    rows = []
    p64 = actors["A"]
    for arm, actor in actors.items():
        for skill, tids in panels.items():
            for tid in tids:
                st = starts.get(tid)
                if st is None and skill == "C":
                    st = resolve_c_start(tid, phys_root(tid), starts)
                if st is None:
                    print(f"[warn] missing start {tid}", flush=True)
                    continue
                r64 = closed_loop(p64, st, book, rho, device)
                ru = closed_loop(actor, st, book, rho, device) if arm != "A" else r64
                rows.append(
                    {
                        "arm": arm,
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
                    }
                )
    return rows


def summarize_arm(rows: list[dict], arm: str) -> dict:
    sub = [r for r in rows if r["arm"] == arm]
    def skill_rows(sk):
        return [r for r in sub if r["skill"] == sk]

    b1 = skill_rows("B1")
    c = skill_rows("C_Feasible")
    a1 = skill_rows("A1")
    d = skill_rows("D")
    b1s = skill_rows("B1_success")
    c_train = skill_rows("C_train")

    b1_succ = sum(int(r["CAND_SUCCESS"]) for r in b1)
    b1_p64 = sum(int(r["P64_SUCCESS"]) for r in b1)
    b1_net = sum(int(r["gain"]) - int(r["loss"]) for r in b1)
    c_succ = sum(int(r["CAND_SUCCESS"]) for r in c)
    c_n = len(c)
    a1_base = [r for r in a1 if int(r["P64_SUCCESS"]) == 1]
    d_base = [r for r in d if int(r["P64_SUCCESS"]) == 1]
    b1s_base = [r for r in b1s if int(r["P64_SUCCESS"]) == 1]
    a1_lost = sum(1 for r in a1_base if int(r["CAND_SUCCESS"]) == 0)
    d_lost = sum(1 for r in d_base if int(r["CAND_SUCCESS"]) == 0)
    b1s_lost = sum(1 for r in b1s_base if int(r["CAND_SUCCESS"]) == 0)
    S = 0.5 * ((b1_succ / max(len(b1), 1)) + (c_succ / max(c_n, 1)))
    return {
        "arm": arm,
        "b1_n": len(b1),
        "b1_succ": b1_succ,
        "b1_p64": b1_p64,
        "b1_net": b1_net,
        "c_feasible_n": c_n,
        "c_feasible_succ": c_succ,
        "c_train_succ": sum(int(r["CAND_SUCCESS"]) for r in c_train),
        "c_train_n": len(c_train),
        "a1_base": len(a1_base),
        "a1_lost": a1_lost,
        "d_base": len(d_base),
        "d_lost": d_lost,
        "b1_success_base": len(b1s_base),
        "b1_success_lost": b1s_lost,
        "S": S,
    }


def decide(summaries: dict, c_n: int) -> dict:
    """H1/H2/H3 labels. Uses mean over seeds already aggregated into summaries if single; here one summary per arm after pooling seeds' final ckpts evaluated once — we evaluate each seed's final and average."""
    # summaries: arm -> list of per-seed summary dicts
    def mean_key(arm, key):
        xs = [s[key] for s in summaries[arm]]
        return float(np.mean(xs)), float(np.std(xs))

    Sb, sb_std = mean_key("B", "S")
    Sa, sa_std = mean_key("A", "S")
    Sc, sc_std = mean_key("C", "S")
    b1b, _ = mean_key("B", "b1_net")
    b1a, _ = mean_key("A", "b1_net")
    b1c, _ = mean_key("C", "b1_net")
    cb, _ = mean_key("B", "c_feasible_succ")
    ca, _ = mean_key("A", "c_feasible_succ")
    cc, _ = mean_key("C", "c_feasible_succ")

    def retain_ok(arm) -> bool:
        # exploratory floor: lost/base <= 1/3 on A1, D, B1_success
        ok = True
        for s in summaries[arm]:
            for base_k, lost_k in (
                ("a1_base", "a1_lost"),
                ("d_base", "d_lost"),
                ("b1_success_base", "b1_success_lost"),
            ):
                base = max(int(s[base_k]), 1)
                if s[lost_k] / base > 1 / 3 + 1e-9:
                    ok = False
        return ok

    def h_label(delta, pooled_std, underpowered: bool) -> str:
        if underpowered:
            # still allow Against if clearly worse; Supports blocked if C underpowered alone handled by caller
            pass
        if abs(delta) <= max(pooled_std, 1e-6):
            return "NO_IMPROVEMENT"
        if delta > max(pooled_std, 1e-6):
            return "SUPPORTED"
        return "AGAINST"

    # H3
    h3 = "PASSED"
    for arm in ("B", "C"):
        if not retain_ok(arm):
            h3 = "FAILED"
            break
    # if C panel tiny, retention can still fail/pass; generalization inconclusive separately

    under_c = c_n < 6
    # H1: B vs A on S, also require separate B1/C effects noted
    d_ba = Sb - Sa
    h1 = h_label(d_ba, max(sb_std, sa_std), under_c)
    if h1 == "SUPPORTED" and not retain_ok("B"):
        h1 = "NO_IMPROVEMENT"  # ineligible for Supports without retention
    if h1 == "SUPPORTED" and under_c and abs(b1b - b1a) < 0.5 and (cb - ca) <= 0:
        h1 = "INCONCLUSIVE"

    d_cb = Sc - Sb
    h2 = h_label(d_cb, max(sc_std, sb_std), under_c)
    if h2 == "SUPPORTED" and not retain_ok("C"):
        h2 = "NO_IMPROVEMENT"
    if h2 == "SUPPORTED" and under_c and (cc - cb) <= 0 and abs(b1c - b1b) < 0.5:
        h2 = "INCONCLUSIVE"
    if under_c and h2 == "SUPPORTED":
        # downgrade: n=2 cannot be decisive Supports for token conditioning on C
        if (cc - cb) <= 0:
            h2 = "INCONCLUSIVE"

    if h3 == "PASSED" and under_c:
        # retention may pass but multi-skill gen still weak
        pass

    return {
        "H1": h1,
        "H2": h2,
        "H3": h3 if not (h3 == "PASSED" and under_c and max(cb, cc, ca) == 0 and b1b <= b1a and b1c <= b1a) else ("INCONCLUSIVE" if h3 == "PASSED" else h3),
        "S_mean": {"A": Sa, "B": Sb, "C": Sc},
        "S_std": {"A": sa_std, "B": sb_std, "C": sc_std},
        "b1_net_mean": {"A": b1a, "B": b1b, "C": b1c},
        "c_feasible_succ_mean": {"A": ca, "B": cb, "C": cc},
        "retain_B": retain_ok("B"),
        "retain_C": retain_ok("C"),
        "c_feasible_n": c_n,
        "note": "C-Feasible n=2 is not statistically decisive; C-Unfiltered N/A (sealed ind not opened).",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    print("[P101] Phase 0 — pre-training checks", flush=True)
    p64_sha = sha16(P64_ACTOR)
    if p64_sha != EXPECTED_SHA:
        write_json(OUT / "SUMMARY.json", {"case": "ABORT", "reason": f"sha {p64_sha} != {EXPECTED_SHA}"})
        raise SystemExit(f"P64 sha mismatch: {p64_sha}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p64, book, rho, saved = load_actor(P64_ACTOR, device)
    p64.eval()
    for p in p64.parameters():
        p.requires_grad_(False)
    book = np.asarray(book, np.float64)
    rho = rho_arr(rho)

    # param match
    pb = count_delta_params("B")
    pc = count_delta_params("C")
    param_gap = abs(pb - pc) / max(pc, 1)
    print(f"[P101] delta params B={pb} C={pc} gap={param_gap:.3f}", flush=True)
    if param_gap > 0.10:
        raise SystemExit(f"param mismatch {param_gap}")

    demos = load_train_demos()
    ban = eval_banned_phys()
    train_roots = {d["phys_root"] for d in demos}
    leak = sorted(train_roots & ban)
    if leak:
        raise SystemExit(f"train roots sealed: {leak}")

    # build B/C for parity
    arm_b0 = build_saup_from_p64(p64, "B", device)
    arm_c0 = build_saup_from_p64(p64, "C", device)
    parity = init_parity(p64, arm_b0, arm_c0, demos, device)
    write_json(OUT / "INIT_PARITY.json", parity)
    print(f"[P101] init parity pass={parity['pass']} {parity}", flush=True)
    if not parity["pass"]:
        write_json(OUT / "SUMMARY.json", {"case": "ABORT", "reason": "init_parity_failed", "parity": parity})
        raise SystemExit("init parity failed")

    # decode / HANDOFF sanity
    decode_ok = True
    try:
        a0 = decode_m111_action(book.astype(np.float32), 0, np.zeros(7, np.float32), rho)
        assert a0.shape == (7,)
        logits, res, act = p64(
            torch.as_tensor(demos[0]["obs"][:2], device=device),
            torch.as_tensor(demos[0]["prev"][:2], device=device),
        )
        assert float(res.abs().max()) <= float(np.max(rho)) + 1e-5
        m = mask_handoff_logits(logits.float())
        assert float(m[..., HANDOFF].max()) < -1e8
    except Exception as e:
        decode_ok = False
        write_json(OUT / "SUMMARY.json", {"case": "ABORT", "reason": f"decode/handoff: {e}"})
        raise

    splits = json.loads((P98 / "SPLITS.json").read_text(encoding="utf-8"))
    manifest = {
        "created": utc(),
        "p64_sha": p64_sha,
        "p64_path": str(P64_ACTOR),
        "seeds": SEEDS,
        "updates": UPDATES,
        "lr": LR,
        "early_k": EARLY_K,
        "early_w": EARLY_W,
        "ckpt_select_rule": CKPT_SELECT_RULE,
        "supervision": "feasibility_gated_frozen_z_hat",
        "p99_demos": False,
        "train_demos": [
            {"skill": d["skill"], "traj_id": d["traj_id"], "phys_root": d["phys_root"], "n": int(len(d["tokens"]))}
            for d in demos
        ],
        "delta_params": {"B": pb, "C": pc, "rel_gap": param_gap},
        "hidden": {"B": 96, "C": 72, "d_z": 8},
        "c_unfiltered": None,
        "c_unfiltered_note": "No independently sampled C-Unfiltered panel locked; sealed ind panel not opened.",
    }
    write_json(OUT / "TRAIN_MANIFEST.json", manifest)
    write_json(
        OUT / "PROVENANCE_AUDIT.json",
        {
            "created": utc(),
            "train_phys_roots": sorted(train_roots),
            "banned_overlap": leak,
            "c_train_phys": splits.get("c_train_phys"),
            "c_dev_phys": splits.get("c_dev_phys"),
            "decode_handoff_ok": decode_ok,
            "banned_examples": sorted(list(ban))[:20],
            "n_banned": len(ban),
        },
    )

    starts = lookup_starts()
    # panels
    panel_dev = {
        "B1": list(splits["dev"]["B1"]),
        "C_Feasible": list(splits["dev"]["C"]),
        "A1": list(splits["dev"]["A1"]),
        "D": list(splits["dev"]["D"]),
    }
    # B1 P64-success retention: use preservation B1 roots from P98 targets (eval-only)
    b1_pres = []
    with (P98 / "B1_A1_D_PRESERVATION_TARGETS.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["skill"] == "B1":
                b1_pres.append(r["traj_id"])
    panel_dev["B1_success"] = b1_pres
    panel_dev["C_train"] = []
    # map c train traj ids from correction targets
    with (P98 / "B1_C_CORRECTION_TARGETS.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["skill"] == "C":
                panel_dev["C_train"].append(r["traj_id"])

    # resolve C starts
    for tid in panel_dev["C_Feasible"] + panel_dev["C_train"]:
        resolve_c_start(tid, phys_root(tid), starts)

    print("[P101] Phase 1 — train Arms B/C", flush=True)
    all_logs = []
    ckpt_paths = {"B": {}, "C": {}}
    nparams = {}
    for arm in ("B", "C"):
        for seed in SEEDS:
            print(f"[P101] train Arm {arm} seed={seed}", flush=True)
            actor, logs, path, ntr = train_arm(arm, seed, p64, book, rho, demos, device)
            all_logs.extend(logs)
            ckpt_paths[arm][seed] = str(path)
            nparams[arm] = ntr
            del actor
            if device.type == "cuda":
                torch.cuda.empty_cache()
    write_csv(LOGS / "TRAIN_CURVES.csv", all_logs)

    print("[P101] Phase 2 — closed-loop eval", flush=True)
    paired_all = []
    early_all = []
    per_seed_sum = {"A": [], "B": [], "C": []}

    # Arm A once (seed-invariant under CL_SEED)
    actors_a = {"A": p64}
    rows_a = eval_panels(actors_a, starts, panel_dev, book, rho, device)
    for r in rows_a:
        r["train_seed"] = -1
    paired_all.extend(rows_a)
    sum_a = summarize_arm(rows_a, "A")
    # replicate A summary for 3 seeds for mean machinery
    for _ in SEEDS:
        per_seed_sum["A"].append(sum_a)
    early_all.extend([{**r, "arm": "A", "train_seed": -1} for r in early_token_feasibility(p64, demos, book, rho, device)])

    for arm in ("B", "C"):
        for seed in SEEDS:
            actor = load_saup_ckpt(Path(ckpt_paths[arm][seed]), p64, arm, device)
            rows = eval_panels({"A": p64, arm: actor}, starts, panel_dev, book, rho, device)
            rows = [r for r in rows if r["arm"] == arm]
            for r in rows:
                r["train_seed"] = seed
            paired_all.extend(rows)
            per_seed_sum[arm].append(summarize_arm(rows, arm))
            early_all.extend(
                [{**r, "arm": arm, "train_seed": seed} for r in early_token_feasibility(actor, demos, book, rho, device)]
            )
            del actor
            if device.type == "cuda":
                torch.cuda.empty_cache()

    write_csv(OUT / "PAIRED_CLOSED_LOOP.csv", paired_all)
    write_csv(OUT / "EARLY_TOKEN_FEASIBILITY.csv", early_all)

    decision = decide(per_seed_sum, c_n=len(panel_dev["C_Feasible"]))
    # aggregate means for report
    def agg(arm):
        keys = per_seed_sum[arm][0].keys()
        out = {}
        for k in keys:
            if k == "arm":
                out[k] = arm
                continue
            xs = [s[k] for s in per_seed_sum[arm]]
            if isinstance(xs[0], (int, float)):
                out[k] = float(np.mean(xs))
                out[k + "_std"] = float(np.std(xs))
            else:
                out[k] = xs[0]
        return out

    summary = {
        "case": "P101_COMPLETE",
        "created": utc(),
        "p64_sha": p64_sha,
        "p64_replaced": False,
        "param_counts": nparams,
        "param_gap": param_gap,
        "ckpt_paths": ckpt_paths,
        "ckpt_select_rule": CKPT_SELECT_RULE,
        "init_parity_pass": parity["pass"],
        "c_unfiltered": "N/A",
        "arms": {"A": agg("A"), "B": agg("B"), "C": agg("C")},
        "per_seed": {a: per_seed_sum[a] for a in ("A", "B", "C")},
        "decision": decision,
        "H1": decision["H1"],
        "H2": decision["H2"],
        "H3": decision["H3"],
    }
    write_json(OUT / "SUMMARY.json", summary)

    # FINAL REPORT
    sa, sb, sc = agg("A"), agg("B"), agg("C")
    # early feas rates
    def early_rate(arm):
        sub = [r for r in early_all if r["arm"] == arm and int(r["t"]) <= 4]
        # for B/C multiple seeds — pool
        n = len(sub) or 1
        feas = sum(int(r["feasible_under_z_hat"]) for r in sub)
        return 100.0 * feas / n, n

    er_a, en_a = early_rate("A")
    er_b, en_b = early_rate("B")
    er_c, en_c = early_rate("C")

    report = f"""# FINAL_REPORT — P101 SAUP Three-Arm Exploratory Experiment

Generated: {utc()}

## Setup

- P64 sha `{p64_sha}` (immutable; **not** replaced).
- Arms: A=frozen P64; B=Δ(h); C=Δ(h,z_hat); backbone/token/res-preact frozen.
- Supervision: feasibility-gated frozen `z_hat` residual targets; C/D masked.
- Seeds {SEEDS}, updates={UPDATES}, LR={LR}, EARLY_W={EARLY_W}.
- Checkpoint rule: **{CKPT_SELECT_RULE}**.
- Trainable params: B={nparams.get('B')} C={nparams.get('C')} (gap={param_gap:.3%}).
- P99 demos: **not used**.
- C-Unfiltered panel: **N/A** (sealed independent panel not opened).

## Pre-checks

| Check | Result |
|---|---|
| SHA | pass (`{p64_sha}`) |
| Init parity | pass={parity['pass']} |
| HANDOFF/decode | pass={decode_ok} |
| Phys-root train∩ban | {leak or 'none'} |

## Closed-loop (mean ± std over seeds; A is deterministic)

| Arm | B1 succ | B1 net | C-Feasible | C-train (diag) | A1 lost/base | D lost/base | B1-succ lost/base | S |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | {sa['b1_succ']:.2f}/{sa['b1_n']:.0f} | {sa['b1_net']:.2f} | {sa['c_feasible_succ']:.2f}/{sa['c_feasible_n']:.0f} | {sa['c_train_succ']:.2f}/{sa['c_train_n']:.0f} | {sa['a1_lost']:.2f}/{sa['a1_base']:.0f} | {sa['d_lost']:.2f}/{sa['d_base']:.0f} | {sa['b1_success_lost']:.2f}/{sa['b1_success_base']:.0f} | {sa['S']:.3f} |
| B | {sb['b1_succ']:.2f}/{sb['b1_n']:.0f} | {sb['b1_net']:.2f} | {sb['c_feasible_succ']:.2f}/{sb['c_feasible_n']:.0f} | {sb['c_train_succ']:.2f}/{sb['c_train_n']:.0f} | {sb['a1_lost']:.2f}/{sb['a1_base']:.0f} | {sb['d_lost']:.2f}/{sb['d_base']:.0f} | {sb['b1_success_lost']:.2f}/{sb['b1_success_base']:.0f} | {sb['S']:.3f}±{sb.get('S_std',0):.3f} |
| C | {sc['b1_succ']:.2f}/{sc['b1_n']:.0f} | {sc['b1_net']:.2f} | {sc['c_feasible_succ']:.2f}/{sc['c_feasible_n']:.0f} | {sc['c_train_succ']:.2f}/{sc['c_train_n']:.0f} | {sc['a1_lost']:.2f}/{sc['a1_base']:.0f} | {sc['d_lost']:.2f}/{sc['d_base']:.0f} | {sc['b1_success_lost']:.2f}/{sc['b1_success_base']:.0f} | {sc['S']:.3f}±{sc.get('S_std',0):.3f} |

C-Feasible n={len(panel_dev['C_Feasible'])} — **not statistically decisive**.

## Early t=0–4 feasibility under frozen/actor z_hat (train demos)

| Arm | feasible@z_hat % | n rows |
|---|---:|---:|
| A | {er_a:.1f} | {en_a} |
| B | {er_b:.1f} | {en_b} |
| C | {er_c:.1f} | {en_c} |

Distinguish: **feasible under predicted token** (table) vs **any-M111 global** (see EARLY_TOKEN_FEASIBILITY.csv `feasible_any_m111`).

## Decision

| Hypothesis | Label |
|---|---|
| **H1** | **{decision['H1']}** |
| **H2** | **{decision['H2']}** |
| **H3** | **{decision['H3']}** |

Notes: {decision.get('note')}
Retain floors: B={decision['retain_B']} C={decision['retain_C']}.

## Artifacts

Under `results/experiments/p101_saup_three_arm/`: SUMMARY.json, FINAL_REPORT.md, INIT_PARITY.json, PROVENANCE_AUDIT.json, TRAIN_MANIFEST.json, PAIRED_CLOSED_LOOP.csv, EARLY_TOKEN_FEASIBILITY.csv, logs/, checkpoints/.
"""
    (OUT / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    print(report, flush=True)
    print(f"[P101] DONE H1={decision['H1']} H2={decision['H2']} H3={decision['H3']}", flush=True)


if __name__ == "__main__":
    main()
