"""P103 — Retention-constrained unified residual learning (Arm B only).

L_total = L_correction + lambda_ret * L_retention
L_retention = E_H~D_P64_success [ ||r_new(H) - r_P64(H)||^2 ] on full trajectories.

Reuses P101 Arm B checkpoints as unconstrained control. Does NOT replace P64,
unfreeze tokens, change M111/R3F, run PPO/GRPO, or open sealed ind panels.
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
from experiments.precision.p101_saup_three_arm.run_p101 import (  # noqa: E402
    feasibility_gated_loss,
)
from models.m111_checkpoint import HANDOFF, decode_m111_action  # noqa: E402
from models.saup_policy import build_saup_from_p64  # noqa: E402
from training.skill_aware_action_vocab import ResidualBudget  # noqa: E402

OUT = ROOT / "results/experiments/p103_retention_constrained_residual"
CKPT = OUT / "checkpoints"
LOGS = OUT / "logs"
P98 = ROOT / "results/experiments/p98_unified_b1_c_consolidation"
P87_TEACHER = ROOT / "results/experiments/p87_k48_actor_retention/teacher"
P97 = ROOT / "results/experiments/p97_c_expert_recoverability"
P97_CORR = P97 / "C_CORRECTION_MANIFEST.csv"
P97_NEW = P97 / "new_ics/accepted"
P101 = ROOT / "results/experiments/p101_saup_three_arm"
P101_CKPT = P101 / "checkpoints"
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
CKPT_SELECT_RULE = "final_update_2000_no_cl_selection"
# Calibration strength: at P101-B residual deviation scale, retention term ≈ STRENGTH × L_corr.
CALIB_STRENGTH = 3.0
# Offline preservation tolerances (P102: P101-B mean residual L2 ≈ 0.23–0.30 caused CL losses).
TOL_MEAN_RES_L2 = 0.05
TOL_P95_RES_L2 = 0.10
TOL_MAX_PER_DIM = 0.15
SAT_FRAC_RHO = 0.97  # residual saturation if |r_i| > 0.97 * rho_i


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


def load_saup_ckpt(path: Path, p64, device):
    actor = build_saup_from_p64(p64, "B", device)
    blob = torch.load(path, map_location=device, weights_only=False)
    actor.load_state_dict(blob["actor"], strict=True)
    actor.eval()
    return actor


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
    """Execute tokens from token_actor, residuals from resid_actor (P102-style residual swap)."""
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
def init_parity(p64, arm_b, demos, device) -> dict:
    d = demos[0]
    obs = torch.as_tensor(d["obs"], device=device, dtype=torch.float32)
    prev = torch.as_tensor(d["prev"], device=device, dtype=torch.float32)
    p64.eval()
    arm_b.eval()
    la, ra, aa = p64(obs, prev)
    lb, rb, ab = arm_b(obs, prev)
    za = torch.argmax(mask_handoff_logits(la.float()), dim=-1)
    zb = torch.argmax(mask_handoff_logits(lb.float()), dim=-1)
    out = {
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


def retention_loss(actor, p64, demo, device) -> tuple[torch.Tensor, dict]:
    """L_ret = mean_t ||r_new(t) - r_P64(t)||^2 on full trajectory."""
    obs = torch.as_tensor(demo["obs"], device=device, dtype=torch.float32)
    prev = torch.as_tensor(demo["prev"], device=device, dtype=torch.float32)
    with torch.no_grad():
        _, r_ref, a_ref = p64(obs, prev)
    _, r_new, a_new = actor(obs, prev)
    per_t = ((r_new - r_ref) ** 2).sum(-1)
    loss = per_t.mean()
    with torch.no_grad():
        act_dev = ((a_new - a_ref) ** 2).sum(-1).sqrt().mean()
        res_l2 = per_t.sqrt().mean()
    return loss, {
        "loss": float(loss.detach()),
        "res_l2": float(res_l2),
        "act_l2": float(act_dev),
        "n": int(len(demo["obs"])),
        "skill": demo["skill"],
        "phys_root": demo["phys_root"],
    }


def sample_balanced_pres(by_skill: dict[str, list], rng: np.random.Generator) -> list[dict]:
    """One demo per protected class; within class uniform over roots."""
    out = []
    for sk in ("A1", "B1", "D"):
        pool = by_skill.get(sk) or []
        if not pool:
            continue
        out.append(pool[int(rng.integers(0, len(pool)))])
    return out


@torch.no_grad()
def measure_retention_deviation(actor, p64, demos, rho, device, *, tag: str, seed: int = -1) -> list[dict]:
    rows = []
    actor.eval()
    p64.eval()
    rho_t = np.asarray(rho, np.float64)
    sat_thr = SAT_FRAC_RHO * rho_t
    for d in demos:
        obs = torch.as_tensor(d["obs"], device=device, dtype=torch.float32)
        prev = torch.as_tensor(d["prev"], device=device, dtype=torch.float32)
        _, r64, a64 = p64(obs, prev)
        logits, r, a = actor(obs, prev)
        l64, _, _ = p64(obs, prev)
        z64 = torch.argmax(mask_handoff_logits(l64.float()), dim=-1)
        z = torch.argmax(mask_handoff_logits(logits.float()), dim=-1)
        dr = (r - r64).detach().cpu().numpy()
        da = (a - a64).detach().cpu().numpy()
        res_l2 = np.linalg.norm(dr, axis=-1)
        act_l2 = np.linalg.norm(da, axis=-1)
        n = len(res_l2)
        early = res_l2[: min(5, n)]
        late = res_l2[5:] if n > 5 else np.array([0.0])
        per_dim_max = float(np.abs(dr).max()) if n else 0.0
        sat = float(np.mean(np.any(np.abs(r.detach().cpu().numpy()) > sat_thr, axis=-1))) if n else 0.0
        rows.append(
            {
                "tag": tag,
                "train_seed": seed,
                "skill": d["skill"],
                "traj_id": d["traj_id"],
                "phys_root": d["phys_root"],
                "n": n,
                "mean_res_l2": float(res_l2.mean()) if n else 0.0,
                "max_per_dim_res": per_dim_max,
                "p95_res_l2": float(np.percentile(res_l2, 95)) if n else 0.0,
                "mean_act_l2": float(act_l2.mean()) if n else 0.0,
                "mean_res_l2_t0_4": float(early.mean()) if len(early) else 0.0,
                "mean_res_l2_t_gt4": float(late.mean()) if n > 5 else float(res_l2.mean() if n else 0.0),
                "sat_frac": sat,
                "tok_agree": float((z == z64).float().mean().item()) if n else 1.0,
                "mean_mse_res": float(((r - r64) ** 2).sum(-1).mean().item()) if n else 0.0,
                "tol_mean_pass": int((float(res_l2.mean()) if n else 0.0) < TOL_MEAN_RES_L2),
                "tol_p95_pass": int((float(np.percentile(res_l2, 95)) if n else 0.0) < TOL_P95_RES_L2),
                "tol_maxdim_pass": int(per_dim_max < TOL_MAX_PER_DIM),
            }
        )
    return rows


def calibrate_lambda(p64, corr_demos, pres_demos, book, rho, device) -> dict:
    """Training-only loss-scale calibration. No eval/dev outcomes used."""
    actor0 = build_saup_from_p64(p64, "B", device)
    actor0.eval()
    # L_corr at Δ=0 (P64-equivalent residual capacity)
    corr_vals = []
    for d in corr_demos:
        loss, stats = feasibility_gated_loss(actor0, d, book, rho, device)
        if stats["n_feas"] > 0:
            corr_vals.append(float(loss.detach()))
    l_corr = float(np.mean(corr_vals)) if corr_vals else 1e-3

    # L_ret under unconstrained P101-B (seed0) on preservation train demos — scale of harmful drift
    p101_path = P101_CKPT / "armB_seed0_final.pt"
    if not p101_path.is_file():
        raise FileNotFoundError(f"missing P101-B control checkpoint: {p101_path}")
    actor_b = load_saup_ckpt(p101_path, p64, device)
    by_skill: dict[str, list] = {"A1": [], "B1": [], "D": []}
    for d in demos_group(pres_demos):
        by_skill[d["skill"]].append(d)
    # balanced: mean of per-class means of L_ret
    class_means = []
    per_class = {}
    for sk, pool in by_skill.items():
        vals = []
        for d in pool:
            loss, _ = retention_loss(actor_b, p64, d, device)
            vals.append(float(loss.detach()))
        m = float(np.mean(vals)) if vals else 0.0
        per_class[sk] = m
        class_means.append(m)
    l_ret = float(np.mean(class_means)) if class_means else 1e-3
    lam = float(CALIB_STRENGTH * l_corr / (l_ret + 1e-12))
    return {
        "created": utc(),
        "method": "train_only_loss_scale",
        "formula": "lambda_ret = CALIB_STRENGTH * mean_L_corr(Δ=0) / mean_L_ret(P101-B on D_pres balanced)",
        "CALIB_STRENGTH": CALIB_STRENGTH,
        "mean_L_corr_init": l_corr,
        "mean_L_ret_p101B_balanced": l_ret,
        "L_ret_per_class_p101B": per_class,
        "lambda_ret": lam,
        "p101_ref_ckpt": str(p101_path),
        "tol": {
            "mean_res_l2": TOL_MEAN_RES_L2,
            "p95_res_l2": TOL_P95_RES_L2,
            "max_per_dim": TOL_MAX_PER_DIM,
            "rationale": "P102: P101-B mean residual L2≈0.23–0.30 on preservation histories caused CL losses; B-tok+P64-res rescued 5/5. Offline exploratory tolerance set well below that scale.",
        },
        "note": "No post-hoc coefficient tuning from development/evaluation outcomes.",
    }


def demos_group(demos: list[dict]) -> list[dict]:
    return demos


def train_p103(seed: int, p64, book, rho, corr_demos, pres_demos, lambda_ret: float, device):
    seed_all(seed, deterministic_torch=True)
    actor = build_saup_from_p64(p64, "B", device)
    n_train = actor.trainable_param_count()
    opt = torch.optim.Adam([p for p in actor.parameters() if p.requires_grad], lr=LR)
    b1 = [d for d in corr_demos if d["skill"] == "B1"]
    c = [d for d in corr_demos if d["skill"] == "C"]
    by_skill: dict[str, list] = {"A1": [], "B1": [], "D": []}
    for d in pres_demos:
        by_skill[d["skill"]].append(d)
    rng = np.random.default_rng(seed)
    log_rows = []
    actor.train()
    actor.base.eval()
    t0 = time.time()
    for u in range(1, UPDATES + 1):
        db = b1[int(rng.integers(0, len(b1)))]
        dc = c[int(rng.integers(0, len(c)))]
        lb, sb = feasibility_gated_loss(actor, db, book, rho, device)
        lc, sc = feasibility_gated_loss(actor, dc, book, rho, device)
        l_corr = lb + lc
        pres_batch = sample_balanced_pres(by_skill, rng)
        l_ret = torch.zeros((), device=device)
        ret_stats = []
        for dp in pres_batch:
            lr_i, st_i = retention_loss(actor, p64, dp, device)
            l_ret = l_ret + lr_i
            ret_stats.append(st_i)
        if pres_batch:
            l_ret = l_ret / len(pres_batch)
        loss = l_corr + float(lambda_ret) * l_ret
        opt.zero_grad(set_to_none=True)
        if float(loss.detach()) > 0:
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in actor.parameters() if p.requires_grad], GRAD_CLIP)
            opt.step()
        if u % 50 == 0 or u == 1 or u == UPDATES:
            row = {
                "seed": seed,
                "update": u,
                "loss": float(loss.detach()),
                "L_corr": float(l_corr.detach()),
                "L_B": sb["loss"],
                "L_C": sc["loss"],
                "L_ret": float(l_ret.detach()),
                "lambda_ret": float(lambda_ret),
                "lambda_L_ret": float(lambda_ret) * float(l_ret.detach()),
                "ret_res_l2_mean": float(np.mean([s["res_l2"] for s in ret_stats])) if ret_stats else 0.0,
                "elapsed_s": time.time() - t0,
            }
            log_rows.append(row)
            print(
                f"  [P103 s={seed}] u={u}/{UPDATES} loss={row['loss']:.5f} "
                f"corr={row['L_corr']:.5f} ret={row['L_ret']:.5f} λLret={row['lambda_L_ret']:.5f}",
                flush=True,
            )
    path = CKPT / f"armB_ret_seed{seed}_final.pt"
    torch.save(
        {
            "arm": "B_ret",
            "seed": seed,
            "update": UPDATES,
            "p64_sha": EXPECTED_SHA,
            "lambda_ret": float(lambda_ret),
            "actor": actor.state_dict(),
            "trainable_params": n_train,
            "ckpt_select_rule": CKPT_SELECT_RULE,
        },
        path,
    )
    actor.eval()
    return actor, log_rows, path, n_train


def eval_panels(p64, cand_actors: dict, starts, panels: dict, book, rho, device, train_seed: int) -> list[dict]:
    """Evaluate candidate arms vs immutable P64. If cand_actors empty, evaluate A only."""
    rows = []
    arms = cand_actors if cand_actors else {"A": p64}
    # Cache P64 rollouts per traj within this call
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
                    first_tok_div: int | str = ""
                    first_res_div: int | str = ""
                else:
                    ru = closed_loop(actor, st, book, rho, device, collect=True)
                    n = min(len(r64.get("tokens", [])), len(ru.get("tokens", [])))
                    if n:
                        agree = sum(int(r64["tokens"][t] == ru["tokens"][t]) for t in range(n)) / n
                        tok_div = 1.0 - agree
                        first_tok_div = ""
                        first_res_div = ""
                        for t in range(n):
                            if first_tok_div == "" and r64["tokens"][t] != ru["tokens"][t]:
                                first_tok_div = t
                            if first_res_div == "" and float(np.linalg.norm(r64["resids"][t] - ru["resids"][t])) > 1e-5:
                                first_res_div = t
                    else:
                        tok_div = float("nan")
                        first_tok_div = ""
                        first_res_div = ""
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
                        "first_token_div": first_tok_div if arm != "A" else "",
                        "first_residual_div": first_res_div if arm != "A" else "",
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
    a1_base = [r for r in a1 if int(r["P64_SUCCESS"]) == 1]
    d_base = [r for r in d if int(r["P64_SUCCESS"]) == 1]
    b1s_base = [r for r in b1s if int(r["P64_SUCCESS"]) == 1]
    b1_fail_base = [r for r in b1 if int(r["P64_SUCCESS"]) == 0]
    return {
        "arm": arm,
        "b1_n": len(b1),
        "b1_succ": sum(int(r["CAND_SUCCESS"]) for r in b1),
        "b1_p64": sum(int(r["P64_SUCCESS"]) for r in b1),
        "b1_net": sum(int(r["gain"]) - int(r["loss"]) for r in b1),
        "b1_new_recover": sum(1 for r in b1_fail_base if int(r["CAND_SUCCESS"]) == 1),
        "c_feasible_n": len(c),
        "c_feasible_succ": sum(int(r["CAND_SUCCESS"]) for r in c),
        "c_feasible_gain": sum(int(r["gain"]) for r in c),
        "c_train_succ": sum(int(r["CAND_SUCCESS"]) for r in c_train),
        "c_train_n": len(c_train),
        "a1_base": len(a1_base),
        "a1_lost": sum(1 for r in a1_base if int(r["CAND_SUCCESS"]) == 0),
        "a1_retained": sum(1 for r in a1_base if int(r["CAND_SUCCESS"]) == 1),
        "d_base": len(d_base),
        "d_lost": sum(1 for r in d_base if int(r["CAND_SUCCESS"]) == 0),
        "d_retained": sum(1 for r in d_base if int(r["CAND_SUCCESS"]) == 1),
        "b1_success_base": len(b1s_base),
        "b1_success_lost": sum(1 for r in b1s_base if int(r["CAND_SUCCESS"]) == 0),
        "b1_success_retained": sum(1 for r in b1s_base if int(r["CAND_SUCCESS"]) == 1),
    }


def decide_gates(per_seed_p103: list[dict], per_seed_p101b: list[dict], new_success_rows: list[dict]) -> dict:
    """Retention + new-skill gates. Exploratory (small panels)."""
    # Retention gate: no losses on protected A1 and B1-success across all P103 seeds
    a1_losses = [s["a1_lost"] for s in per_seed_p103]
    b1s_losses = [s["b1_success_lost"] for s in per_seed_p103]
    d_losses = [s["d_lost"] for s in per_seed_p103]
    retention_zero_a1_b1s = all(x == 0 for x in a1_losses) and all(x == 0 for x in b1s_losses)
    # Material improvement vs P101-B
    p101_a1 = float(np.mean([s["a1_lost"] for s in per_seed_p101b]))
    p101_b1s = float(np.mean([s["b1_success_lost"] for s in per_seed_p101b]))
    p103_a1 = float(np.mean(a1_losses))
    p103_b1s = float(np.mean(b1s_losses))
    retention_beats_p101 = (p103_a1 + p103_b1s) < (p101_a1 + p101_b1s) - 1e-9
    retention_gate = bool(retention_zero_a1_b1s and retention_beats_p101)

    # New-skill: ≥1 root-disjoint recovery vs P64 with no compensating protected losses
    # Root-disjoint = B1 previously unsuccessful (dev B1 with P64 fail) or C-Feasible gain
    genuine_gains = [r for r in new_success_rows if int(r.get("genuine_root_disjoint", 0)) == 1]
    any_genuine = len(genuine_gains) > 0
    new_skill_gate = bool(any_genuine and retention_zero_a1_b1s)

    # Delta collapse heuristic: if mean retention MSE near 0 and no gains
    outcome = "UNKNOWN"
    if retention_gate and new_skill_gate:
        outcome = "A_RETENTION_AND_NEW_SUCCESS"
    elif retention_gate and not any_genuine:
        outcome = "B_RETENTION_NO_NEW_SUCCESS"
    elif not retention_zero_a1_b1s:
        outcome = "C_RETENTION_STILL_FAILS"
    else:
        outcome = "D_CORRECTION_RETENTION_CONFLICT"

    stop_residual_only = outcome in (
        "B_RETENTION_NO_NEW_SUCCESS",
        "C_RETENTION_STILL_FAILS",
        "D_CORRECTION_RETENTION_CONFLICT",
    )
    return {
        "retention_gate": retention_gate,
        "retention_zero_a1_b1s": retention_zero_a1_b1s,
        "retention_beats_p101B": retention_beats_p101,
        "a1_lost_per_seed": a1_losses,
        "b1_success_lost_per_seed": b1s_losses,
        "d_lost_per_seed": d_losses,
        "p101B_mean_a1_lost": p101_a1,
        "p101B_mean_b1s_lost": p101_b1s,
        "p103_mean_a1_lost": p103_a1,
        "p103_mean_b1s_lost": p103_b1s,
        "new_skill_gate": new_skill_gate,
        "n_genuine_root_disjoint_gains": len(genuine_gains),
        "genuine_gains": genuine_gains,
        "outcome_class": outcome,
        "stop_residual_only": stop_residual_only,
        "note": "Small development panels: exploratory evidence only, not population-level.",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    print("[P103] Phase 0 — provenance + parity + lambda calibration", flush=True)
    p64_sha = sha16(P64_ACTOR)
    if p64_sha != EXPECTED_SHA:
        write_json(OUT / "SUMMARY.json", {"case": "ABORT", "reason": f"sha {p64_sha} != {EXPECTED_SHA}"})
        raise SystemExit(f"P64 sha mismatch: {p64_sha}")

    for seed in SEEDS:
        p = P101_CKPT / f"armB_seed{seed}_final.pt"
        if not p.is_file():
            raise SystemExit(f"missing P101-B control: {p}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p64, book, rho, saved = load_actor(P64_ACTOR, device)
    p64.eval()
    for p in p64.parameters():
        p.requires_grad_(False)
    book = np.asarray(book, np.float64)
    rho = rho_arr(rho)

    corr_demos = load_correction_demos()
    pres_demos = load_preservation_demos()
    ban = eval_banned_phys()
    corr_roots = {d["phys_root"] for d in corr_demos}
    pres_roots = {d["phys_root"] for d in demos_group(pres_demos)}
    # Role separation: same phys_root may appear across skills (A1 preserv vs B1 corr); flag skill-role pairs
    corr_keys = {(d["skill"], d["phys_root"]) for d in corr_demos}
    preserv_keys = {(d["skill"], d["phys_root"]) for d in demos_group(pres_demos)}
    role_overlap = sorted(corr_keys & preserv_keys)
    leak_corr = sorted(corr_roots & ban)
    leak_pres = sorted(pres_roots & ban)
    if leak_corr or leak_pres:
        write_json(
            OUT / "SUMMARY.json",
            {"case": "ABORT", "reason": "sealed_root_leak", "corr": leak_corr, "pres": leak_pres},
        )
        raise SystemExit(f"sealed leak corr={leak_corr} pres={leak_pres}")
    if role_overlap:
        # Same skill+root used for both roles would be a provenance failure
        write_json(OUT / "SUMMARY.json", {"case": "ABORT", "reason": "role_overlap", "overlap": role_overlap})
        raise SystemExit(f"correction∩preservation skill-root overlap: {role_overlap}")

    # No C preservation fabricated
    c_pres = [d for d in demos_group(pres_demos) if d["skill"] == "C"]
    if c_pres:
        raise SystemExit("C preservation demos must not exist")

    arm0 = build_saup_from_p64(p64, "B", device)
    parity = init_parity(p64, arm0, corr_demos, device)
    write_json(OUT / "INIT_PARITY.json", parity)
    print(f"[P103] init parity pass={parity['pass']}", flush=True)
    if not parity["pass"]:
        write_json(OUT / "SUMMARY.json", {"case": "ABORT", "reason": "init_parity_failed", "parity": parity})
        raise SystemExit("init parity failed")

    eval_only = "--eval-only" in sys.argv
    if eval_only and (OUT / "TRAIN_MANIFEST.json").is_file():
        prior = json.loads((OUT / "TRAIN_MANIFEST.json").read_text(encoding="utf-8"))
        calib = prior.get("lambda_calibration") or {}
        lambda_ret = float(prior["lambda_ret"])
        print(f"[P103] --eval-only: reusing locked lambda_ret={lambda_ret:.6g} from TRAIN_MANIFEST", flush=True)
    else:
        calib = calibrate_lambda(p64, corr_demos, demos_group(pres_demos), book, rho, device)
        lambda_ret = float(calib["lambda_ret"])
        print(
            f"[P103] calibrated lambda_ret={lambda_ret:.6g} "
            f"(L_corr={calib['mean_L_corr_init']:.6g}, L_ret_p101={calib['mean_L_ret_p101B_balanced']:.6g})",
            flush=True,
        )

    splits = json.loads((P98 / "SPLITS.json").read_text(encoding="utf-8"))
    by_skill_counts = {
        sk: len([d for d in demos_group(pres_demos) if d["skill"] == sk]) for sk in ("A1", "B1", "D")
    }

    provenance = {
        "created": utc(),
        "p64_sha": p64_sha,
        "correction_roots": sorted(corr_roots),
        "preservation_roots": sorted(pres_roots),
        "preservation_by_skill": {
            sk: sorted({d["phys_root"] for d in demos_group(pres_demos) if d["skill"] == sk}) for sk in ("A1", "B1", "D")
        },
        "preservation_counts": by_skill_counts,
        "c_preservation_count": 0,
        "role_skill_root_overlap": role_overlap,
        "banned_overlap_corr": leak_corr,
        "banned_overlap_pres": leak_pres,
        "p102_cf_roots_not_added": True,
        "note": "Preservation = P98 P64-success train-eligible A1/B1/D only. No C preservation. Sealed/ind/exposed excluded via ban set. P102 CF roots not used as train eligibility.",
        "tol_rationale": calib["tol"],
    }
    write_json(OUT / "PROVENANCE_AUDIT.json", provenance)

    manifest = {
        "created": utc(),
        "p64_sha": p64_sha,
        "p64_path": str(P64_ACTOR),
        "architecture": "P101_Arm_B_Delta_h",
        "seeds": SEEDS,
        "updates": UPDATES,
        "lr": LR,
        "early_k": EARLY_K,
        "early_w": EARLY_W,
        "grad_clip": GRAD_CLIP,
        "ckpt_select_rule": CKPT_SELECT_RULE,
        "supervision_correction": "feasibility_gated_frozen_z_hat",
        "supervision_retention": "full_traj_residual_mse_vs_p64_balanced_A1_B1_D",
        "lambda_ret": lambda_ret,
        "lambda_calibration": calib,
        "p101_control_ckpts": {str(s): str(P101_CKPT / f"armB_seed{s}_final.pt") for s in SEEDS},
        "correction_demos": [
            {"skill": d["skill"], "traj_id": d["traj_id"], "phys_root": d["phys_root"], "n": int(len(d["obs"]))}
            for d in corr_demos
        ],
        "preservation_demos": [
            {"skill": d["skill"], "traj_id": d["traj_id"], "phys_root": d["phys_root"], "n": int(len(d["obs"]))}
            for d in demos_group(pres_demos)
        ],
        "no_token_conditioned_residual": True,
        "no_ppo_grpo": True,
        "p64_immutable": True,
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

    all_logs = []
    ckpt_paths = {}
    nparams = None
    if eval_only:
        print("[P103] Phase 1 — SKIPPED (--eval-only); loading existing checkpoints", flush=True)
        for seed in SEEDS:
            path = CKPT / f"armB_ret_seed{seed}_final.pt"
            if not path.is_file():
                raise SystemExit(f"missing checkpoint for --eval-only: {path}")
            blob = torch.load(path, map_location="cpu", weights_only=False)
            ckpt_paths[seed] = str(path)
            nparams = int(blob.get("trainable_params") or 6919)
            # Prefer locked lambda from checkpoint / prior manifest
            if "lambda_ret" in blob:
                lambda_ret = float(blob["lambda_ret"])
        if (LOGS / "TRAIN_CURVES.csv").is_file():
            with (LOGS / "TRAIN_CURVES.csv").open(encoding="utf-8") as f:
                all_logs = list(csv.DictReader(f))
    else:
        print("[P103] Phase 1 — train retention-constrained Arm B", flush=True)
        for seed in SEEDS:
            print(f"[P103] train seed={seed}", flush=True)
            actor, logs, path, ntr = train_p103(
                seed, p64, book, rho, corr_demos, demos_group(pres_demos), lambda_ret, device
            )
            all_logs.extend(logs)
            ckpt_paths[seed] = str(path)
            nparams = ntr
            del actor
            if device.type == "cuda":
                torch.cuda.empty_cache()
        write_csv(LOGS / "TRAIN_CURVES.csv", all_logs)

    print("[P103] Phase 2 — offline retention deviation + closed-loop", flush=True)
    ret_rows: list[dict] = []
    # P64 baseline offline (should be ~0)
    print("[P103] offline retention: A_P64", flush=True)
    ret_rows.extend(measure_retention_deviation(p64, p64, demos_group(pres_demos), rho, device, tag="A_P64", seed=-1))

    paired_all: list[dict] = []
    per_seed_sum = {"A": [], "B_P101": [], "C_P103": []}
    new_success_rows: list[dict] = []

    # Arm A once
    print("[P103] closed-loop: Arm A (P64)", flush=True)
    rows_a = eval_panels(p64, {}, starts, panel_dev, book, rho, device, train_seed=-1)
    paired_all.extend(rows_a)
    sum_a = summarize_arm(rows_a, "A")
    for _ in SEEDS:
        per_seed_sum["A"].append(sum_a)
    print(f"[P103] Arm A summary: a1_lost={sum_a['a1_lost']} b1s_lost={sum_a['b1_success_lost']}", flush=True)

    for seed in SEEDS:
        # P101-B control (reused)
        print(f"[P103] closed-loop: B_P101 seed={seed}", flush=True)
        actor_p101 = load_saup_ckpt(P101_CKPT / f"armB_seed{seed}_final.pt", p64, device)
        ret_rows.extend(
            measure_retention_deviation(
                actor_p101, p64, demos_group(pres_demos), rho, device, tag="B_P101", seed=seed
            )
        )
        rows_b = eval_panels(p64, {"B_P101": actor_p101}, starts, panel_dev, book, rho, device, train_seed=seed)
        rows_b = [r for r in rows_b if r["arm"] == "B_P101"]
        paired_all.extend(rows_b)
        per_seed_sum["B_P101"].append(summarize_arm(rows_b, "B_P101"))

        # P103
        print(f"[P103] closed-loop: C_P103 seed={seed}", flush=True)
        actor_p103 = load_saup_ckpt(Path(ckpt_paths[seed]), p64, device)
        ret_rows.extend(
            measure_retention_deviation(
                actor_p103, p64, demos_group(pres_demos), rho, device, tag="C_P103", seed=seed
            )
        )
        rows_c = eval_panels(p64, {"C_P103": actor_p103}, starts, panel_dev, book, rho, device, train_seed=seed)
        rows_c = [r for r in rows_c if r["arm"] == "C_P103"]
        paired_all.extend(rows_c)
        per_seed_sum["C_P103"].append(summarize_arm(rows_c, "C_P103"))
        print(
            f"[P103] seed={seed} P103 a1_lost={per_seed_sum['C_P103'][-1]['a1_lost']} "
            f"b1s_lost={per_seed_sum['C_P103'][-1]['b1_success_lost']} "
            f"b1_new={per_seed_sum['C_P103'][-1]['b1_new_recover']}",
            flush=True,
        )

        # New success analysis + residual-swap for newly lost protected
        b1_corr_roots = {d["phys_root"] for d in corr_demos if d["skill"] == "B1"}
        c_corr_roots = {d["phys_root"] for d in corr_demos if d["skill"] == "C"}
        for r in rows_c:
            if int(r["gain"]) == 1:
                pr = phys_root(r["phys_root"])
                # Root-disjoint = previously unsuccessful panel case whose phys_root was not
                # used as that skill's correction training root. C_train is diagnostic only.
                if r["skill"] == "B1" and int(r["P64_SUCCESS"]) == 0:
                    train_root = pr in b1_corr_roots
                    genuine = int(not train_root)
                elif r["skill"] == "C_Feasible" and int(r["P64_SUCCESS"]) == 0:
                    train_root = pr in c_corr_roots
                    genuine = int(not train_root)
                else:
                    train_root = pr in corr_roots
                    genuine = 0
                new_success_rows.append(
                    {
                        "train_seed": seed,
                        "skill": r["skill"],
                        "traj_id": r["traj_id"],
                        "phys_root": pr,
                        "P64_SUCCESS": r["P64_SUCCESS"],
                        "CAND_SUCCESS": r["CAND_SUCCESS"],
                        "genuine_root_disjoint": genuine,
                        "is_train_corr_root": int(train_root),
                        "is_c_train_diag": int(r["skill"] == "C_train"),
                        "note": "genuine only if B1/C_Feasible P64-fail→succ AND phys_root not in that skill's correction train set",
                    }
                )

        # First-divergence + residual-swap for newly lost protected A1 / B1_success
        lost = [
            r
            for r in rows_c
            if r["skill"] in ("A1", "B1_success") and int(r["loss"]) == 1
        ]
        for r in lost:
            st = starts.get(r["traj_id"])
            if st is None:
                continue
            swap = closed_loop_swap(actor_p103, p64, st, book, rho, device)
            new_success_rows.append(
                {
                    "train_seed": seed,
                    "skill": r["skill"],
                    "traj_id": r["traj_id"],
                    "phys_root": r["phys_root"],
                    "P64_SUCCESS": 1,
                    "CAND_SUCCESS": 0,
                    "genuine_root_disjoint": 0,
                    "is_c_train_diag": 0,
                    "diag": "protected_loss",
                    "first_token_div": r.get("first_token_div"),
                    "first_residual_div": r.get("first_residual_div"),
                    "tok_seq_div": r.get("tok_seq_div"),
                    "swap_Btok_P64res_SUCCESS": swap["SUCCESS_RELAXED_12S"],
                    "note": "residual-swap rescue diagnostic for newly lost protected root",
                }
            )

        del actor_p101, actor_p103
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_csv(OUT / "RETENTION_DEVIATION.csv", ret_rows)
    write_csv(OUT / "PAIRED_CLOSED_LOOP.csv", paired_all)
    write_csv(OUT / "NEW_SUCCESS_ANALYSIS.csv", new_success_rows)

    gates = decide_gates(per_seed_sum["C_P103"], per_seed_sum["B_P101"], new_success_rows)

    def agg(arm_key):
        keys = per_seed_sum[arm_key][0].keys()
        out = {}
        for k in keys:
            if k == "arm":
                out[k] = arm_key
                continue
            xs = [s[k] for s in per_seed_sum[arm_key]]
            if isinstance(xs[0], (int, float)):
                out[k] = float(np.mean(xs))
                out[k + "_std"] = float(np.std(xs))
            else:
                out[k] = xs[0]
        return out

    # Offline retention summary for P103 vs P101
    def ret_summary(tag):
        sub = [r for r in ret_rows if r["tag"] == tag]
        if not sub:
            return {}
        return {
            "mean_res_l2": float(np.mean([r["mean_res_l2"] for r in sub])),
            "p95_res_l2": float(np.mean([r["p95_res_l2"] for r in sub])),
            "max_per_dim": float(np.mean([r["max_per_dim_res"] for r in sub])),
            "mean_act_l2": float(np.mean([r["mean_act_l2"] for r in sub])),
            "early_res_l2": float(np.mean([r["mean_res_l2_t0_4"] for r in sub])),
            "late_res_l2": float(np.mean([r["mean_res_l2_t_gt4"] for r in sub])),
            "sat_frac": float(np.mean([r["sat_frac"] for r in sub])),
            "tol_mean_pass_frac": float(np.mean([r["tol_mean_pass"] for r in sub])),
        }

    ret_p101 = ret_summary("B_P101")
    ret_p103 = ret_summary("C_P103")
    # Delta collapse: if P103 residual deviation ≈ 0 and no new gains
    delta_collapsed = bool(
        ret_p103.get("mean_res_l2", 1.0) < 0.01
        and gates["n_genuine_root_disjoint_gains"] == 0
    )

    # Failure taxonomy
    failures = []
    if ret_p103.get("mean_res_l2", 1) < TOL_MEAN_RES_L2 and not gates["retention_zero_a1_b1s"]:
        failures.append("offline_ret_small_but_cl_retention_fails")
    if not gates["retention_zero_a1_b1s"]:
        # check if residual still first on losses
        loss_diags = [r for r in new_success_rows if r.get("diag") == "protected_loss"]
        if loss_diags:
            res_first = sum(
                1
                for r in loss_diags
                if r.get("first_residual_div") != ""
                and (r.get("first_token_div") == "" or int(r["first_residual_div"]) <= int(r["first_token_div"] or 9999))
            )
            if res_first >= max(1, len(loss_diags) // 2):
                failures.append("persistent_residual_deviations_still_cause_losses")
    if delta_collapsed:
        failures.append("strong_retention_collapsed_delta_no_new_gains")
    # offline corr may improve without CL — check train L_corr vs gains
    if gates["n_genuine_root_disjoint_gains"] == 0:
        failures.append("no_root_disjoint_physical_recovery")

    sa, sb, sc = agg("A"), agg("B_P101"), agg("C_P103")

    # Answer the 7 report questions
    q1 = gates["retention_zero_a1_b1s"] and gates["retention_beats_p101B"]
    q2 = ret_p103.get("mean_res_l2", 99) < ret_p101.get("mean_res_l2", 0) * 0.5
    q3 = gates["n_genuine_root_disjoint_gains"] > 0
    q4_c = sc["c_feasible_succ"]
    q5 = (ret_p103.get("mean_res_l2", 1) < TOL_MEAN_RES_L2) and (
        sc["b1_new_recover"] > 0 or sc["c_feasible_gain"] > 0 or gates["n_genuine_root_disjoint_gains"] > 0
    )
    q6 = gates["stop_residual_only"]
    q7 = (
        "Smallest justified next step: joint frozen-backbone SFT with (i) light token-head updates under "
        "feasibility-gated expert tokens on B1/C only, (ii) residual Δ(h) with the same retention MSE on "
        "full P64-success A1/B1/D trajectories, (iii) one seed pilot, no PPO — only if P103 outcome is B/C/D."
        if q6
        else "P103 supports continued residual-only exploration only if outcome A; otherwise stop."
    )

    summary = {
        "case": "P103_COMPLETE",
        "created": utc(),
        "p64_sha": p64_sha,
        "p64_replaced": False,
        "lambda_ret": lambda_ret,
        "lambda_calibration": calib,
        "trainable_params": nparams,
        "ckpt_paths": ckpt_paths,
        "ckpt_select_rule": CKPT_SELECT_RULE,
        "init_parity_pass": parity["pass"],
        "arms": {"A_P64": sa, "B_P101": sb, "C_P103": sc},
        "per_seed": per_seed_sum,
        "retention_offline": {"B_P101": ret_p101, "C_P103": ret_p103},
        "gates": gates,
        "failures": failures,
        "delta_collapsed": delta_collapsed,
        "c_unfiltered": "N/A",
        "questions": {
            "1_retention_prevented_p101_failures": q1,
            "2_persistent_residual_deviations_reduced": q2,
            "3_new_root_disjoint_successes": q3,
            "4_c_feasible_succ_mean": q4_c,
            "5_useful_correction_retention_tradeoff": q5,
            "6_stop_residual_only": q6,
            "7_smallest_joint_experiment": q7,
        },
        "outcome_class": gates["outcome_class"],
    }
    write_json(OUT / "SUMMARY.json", summary)

    report = f"""# FINAL_REPORT — P103 Retention-Constrained Unified Residual Learning

Generated: {utc()}

## Setup

- P64 sha `{p64_sha}` (**immutable**; not replaced).
- Architecture: P101 Arm B only — `r = ρ·tanh(W_P64 h + b + Δ(h))`; backbone/token/res-preact frozen; Δ zero-init.
- Objective: `L_corr + λ_ret · L_ret` with full-trajectory residual MSE vs P64 on balanced A1/B1/D success demos.
- `λ_ret = {lambda_ret:.6g}` locked by training-only calibration (strength={CALIB_STRENGTH}; L_corr_init={calib['mean_L_corr_init']:.6g}; L_ret@P101-B={calib['mean_L_ret_p101B_balanced']:.6g}).
- Seeds {SEEDS}, updates={UPDATES}, LR={LR}. No hyperparameter sweep. No PPO/GRPO. No token-conditioned residual.
- Control **B**: reused P101 Arm B checkpoints (not retrained).
- C-Unfiltered: **N/A** (sealed independent panel not opened).
- Offline preservation tolerances (P102-grounded): mean res L2 < {TOL_MEAN_RES_L2}; p95 < {TOL_P95_RES_L2}; max per-dim < {TOL_MAX_PER_DIM}.

## Pre-checks

| Check | Result |
|---|---|
| SHA | pass |
| Init parity | pass={parity['pass']} |
| Sealed leak | corr={leak_corr or 'none'}; pres={leak_pres or 'none'} |
| Role overlap (skill,root) | {role_overlap or 'none'} |
| C preservation | 0 |

## Offline retention deviation (preservation train histories)

| Tag | mean res L2 | p95 | max/dim | act L2 | t≤4 | t>4 | sat | tol_mean pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P101-B | {ret_p101.get('mean_res_l2', float('nan')):.4f} | {ret_p101.get('p95_res_l2', float('nan')):.4f} | {ret_p101.get('max_per_dim', float('nan')):.4f} | {ret_p101.get('mean_act_l2', float('nan')):.4f} | {ret_p101.get('early_res_l2', float('nan')):.4f} | {ret_p101.get('late_res_l2', float('nan')):.4f} | {ret_p101.get('sat_frac', float('nan')):.3f} | {ret_p101.get('tol_mean_pass_frac', float('nan')):.2f} |
| P103 | {ret_p103.get('mean_res_l2', float('nan')):.4f} | {ret_p103.get('p95_res_l2', float('nan')):.4f} | {ret_p103.get('max_per_dim', float('nan')):.4f} | {ret_p103.get('mean_act_l2', float('nan')):.4f} | {ret_p103.get('early_res_l2', float('nan')):.4f} | {ret_p103.get('late_res_l2', float('nan')):.4f} | {ret_p103.get('sat_frac', float('nan')):.3f} | {ret_p103.get('tol_mean_pass_frac', float('nan')):.2f} |

Offline pass ≠ closed-loop proof.

## Closed-loop SUCCESS_RELAXED_12S (mean ± std over seeds)

| Arm | A1 retained/base (lost) | D retained/base (lost) | B1-succ retained/base (lost) | B1 new recover | C-Feasible | C-train (diag) |
|---|---:|---:|---:|---:|---:|---:|
| A P64 | {sa['a1_retained']:.2f}/{sa['a1_base']:.0f} ({sa['a1_lost']:.2f}) | {sa['d_retained']:.2f}/{sa['d_base']:.0f} ({sa['d_lost']:.2f}) | {sa['b1_success_retained']:.2f}/{sa['b1_success_base']:.0f} ({sa['b1_success_lost']:.2f}) | {sa.get('b1_new_recover',0):.2f} | {sa['c_feasible_succ']:.2f}/{sa['c_feasible_n']:.0f} | {sa['c_train_succ']:.2f}/{sa['c_train_n']:.0f} |
| B P101 | {sb['a1_retained']:.2f}/{sb['a1_base']:.0f} ({sb['a1_lost']:.2f}±{sb.get('a1_lost_std',0):.2f}) | {sb['d_retained']:.2f}/{sb['d_base']:.0f} ({sb['d_lost']:.2f}) | {sb['b1_success_retained']:.2f}/{sb['b1_success_base']:.0f} ({sb['b1_success_lost']:.2f}±{sb.get('b1_success_lost_std',0):.2f}) | {sb.get('b1_new_recover',0):.2f} | {sb['c_feasible_succ']:.2f}/{sb['c_feasible_n']:.0f} | {sb['c_train_succ']:.2f}/{sb['c_train_n']:.0f} |
| C P103 | {sc['a1_retained']:.2f}/{sc['a1_base']:.0f} ({sc['a1_lost']:.2f}±{sc.get('a1_lost_std',0):.2f}) | {sc['d_retained']:.2f}/{sc['d_base']:.0f} ({sc['d_lost']:.2f}) | {sc['b1_success_retained']:.2f}/{sc['b1_success_base']:.0f} ({sc['b1_success_lost']:.2f}±{sc.get('b1_success_lost_std',0):.2f}) | {sc.get('b1_new_recover',0):.2f} | {sc['c_feasible_succ']:.2f}/{sc['c_feasible_n']:.0f} | {sc['c_train_succ']:.2f}/{sc['c_train_n']:.0f} |

Seed-level A1 losses P103: {gates['a1_lost_per_seed']}; B1-succ losses: {gates['b1_success_lost_per_seed']}; D losses: {gates['d_lost_per_seed']}.

C-Feasible n={len(panel_dev['C_Feasible'])} — **not decisive**. Do not treat two C development roots as population evidence.

## Decision gates

| Gate | Result |
|---|---|
| Retention (0 A1/B1-succ losses across seeds + beats P101-B) | **{gates['retention_gate']}** |
| New-skill (≥1 root-disjoint gain, no protected losses) | **{gates['new_skill_gate']}** |
| Outcome class | **{gates['outcome_class']}** |
| Stop residual-only tuning | **{gates['stop_residual_only']}** |
| Failures flagged | {failures or 'none'} |
| Δ collapse heuristic | {delta_collapsed} |

## Answers to required questions

1. **Did retention constraints prevent P101-style residual-induced failures?** {q1}
2. **Were persistent residual deviations actually reduced?** {q2} (P101 mean L2 {ret_p101.get('mean_res_l2', float('nan')):.4f} → P103 {ret_p103.get('mean_res_l2', float('nan')):.4f})
3. **Did the policy learn any new root-disjoint physical successes?** {q3} (n={gates['n_genuine_root_disjoint_gains']})
4. **Did C improve, or remain limited by frozen-token feasibility?** C-Feasible mean succ={q4_c:.2f}/{sc['c_feasible_n']:.0f}; training-root C remains diagnostic only.
5. **Evidence of useful correction–retention tradeoff?** {q5}
6. **Should residual-only training stop?** {q6}
7. **Smallest justified joint token–residual experiment?** {q7}

## Interpretation

Outcome **{gates['outcome_class']}**:
- A: retention + new root-disjoint success → residual-only consolidation supported (exploratory).
- B: retention works, but residual-only multi-skill learning unsupported → stop residual-only loss tuning; reassess joint token–residual + data coverage.
- C: preservation objective insufficient for closed-loop stability → stop residual-only tuning.
- D: correction and retention strongly conflict under frozen-token residual-only → reassess joint formulation.

## Artifacts

`results/experiments/p103_retention_constrained_residual/`: SUMMARY.json, FINAL_REPORT.md, TRAIN_MANIFEST.json, INIT_PARITY.json, PROVENANCE_AUDIT.json, RETENTION_DEVIATION.csv, PAIRED_CLOSED_LOOP.csv, NEW_SUCCESS_ANALYSIS.csv, checkpoints/, logs/.

**STOP after P103.** Do not launch P104 or replace P64 without approval.
"""
    (OUT / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    print(report, flush=True)
    print(f"[P103] DONE outcome={gates['outcome_class']} stop_residual={gates['stop_residual_only']}", flush=True)


if __name__ == "__main__":
    main()
