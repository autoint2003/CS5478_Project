"""P96 — Unified Actor with on-policy residual preservation.

P96-A: expanded-data control (P94-style joint loss).
P96-B: same data + early residual weighting + stronger protected preservation + update constraint.

No routers / multi-head / class labels. P64 immutable unless independent CASE A/B warrants note (never auto-replace).
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
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
    T_MAX,
)
from experiments.precision.p51_four_class_mixed_ppo.ic_pools import root_id  # noqa: E402
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
from models.m111_checkpoint import decode_m111_action, save_m111  # noqa: E402
from training.skill_aware_action_vocab import ResidualBudget  # noqa: E402

OUT = ROOT / "results/experiments/p96_unified_onpolicy_residual_preservation"
CKPT = OUT / "checkpoints"
CACHE = OUT / "cache"
P87_TEACHER = ROOT / "results/experiments/p87_k48_actor_retention/teacher"
P88_CAND = ROOT / "results/experiments/p88_b1_short_sequence_learnability/PHASE0_CANDIDATES.csv"
P94_SPLITS = ROOT / "results/experiments/p94_unified_multiskill_actor/SPLITS.json"
P93_MAN = ROOT / "results/experiments/p93_frozen_router_independent_validation/EVALUATION_SPLIT_MANIFEST.csv"
FROZEN_MANIFEST = ROOT / "results/experiments/final_independent_success_rate/FROZEN_TEST_MANIFEST.json"
P95_IV = ROOT / "results/experiments/p95_unified_actor_closed_loop_audit/ACTION_INTERVENTION_RESULTS.csv"

SEED = 96
SPLIT_SEED = 96
C_ROOT_TARGET_MIN = 8
C_ROOT_TARGET_IDEAL = 12
C_SCREEN_CAP = 55
UPDATES = 600
LR = 1e-3
LOG_EVERY = 25
CL_EVERY = 150
EARLY_K = 4
GRAD_CLIP = 5.0
PRES_TOL_A = 0.08  # looser for A
PRES_TOL_B = 0.03  # tighter for B
EARLY_W_A = 1.5
EARLY_W_B = 4.0
LAMBDA_B = 1.0
LAMBDA_PRES_A = 1.5
LAMBDA_PRES_B = 3.0
LAMBDA_KL_A = 0.25
LAMBDA_KL_B = 0.75
LAMBDA_REG = 1e-3

N_DEV_B1 = 3
N_DEV_A1 = 6
N_DEV_D = 6
N_DEV_C = 4
N_IND_B1 = 10
N_IND_A1 = 6
N_IND_C = 8
N_IND_D = 6


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(path: Path, n: int = 16) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:n]


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


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = (z / den) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, center - half), min(1.0, center + half)


def lookup_starts() -> dict[str, Any]:
    return {s.traj_id: s for s in build_p54_dataset(seed=5478)["schedule"]}


def verified_b1() -> list[str]:
    rows = list(csv.DictReader(P88_CAND.open(encoding="utf-8")))
    return sorted(r["traj_id"] for r in rows if int(r["valid_official"]) == 1)


def eval_banned_roots() -> set[str]:
    """Exposed evaluation roots: frozen independent, P93 eval panels, P94 sealed independent."""
    tids: set[str] = set()
    man = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
    tids |= {e["traj_id"] for e in man["episodes"]}
    for r in csv.DictReader(P93_MAN.open(encoding="utf-8")):
        if r["panel"] in ("B1", "C1", "C2", "A1", "D", "DIAG_EXPOSED_B1"):
            tids.add(r["traj_id"])
    if P94_SPLITS.is_file():
        sp = json.loads(P94_SPLITS.read_text(encoding="utf-8"))
        for tids_ in sp.get("ind", {}).values():
            tids |= set(tids_)
    roots = {root_id(t) for t in tids}
    roots |= {root_id(t.replace("plan_b__", "", 1)) for t in tids}
    return roots


def is_banned(tid: str, ban: set[str]) -> bool:
    rid = root_id(tid)
    if rid in ban:
        return True
    return any(rid in b or b in rid for b in ban)


def load_teacher(tid: str) -> dict | None:
    p = P87_TEACHER / f"{tid}.npz"
    if not p.is_file():
        return None
    z = np.load(p, allow_pickle=True)
    return {
        "traj_id": tid,
        "root_id": root_id(tid),
        "skill": "B1",
        "category": "verified_expert",
        "tokens": np.asarray(z["tokens"], np.int64),
        "residuals": np.asarray(z["residuals"], np.float32),
        "actions": np.asarray(z["actions"], np.float32),
        "obs": np.asarray(z["obs"], np.float32),
        "prev": np.asarray(z["prev"], np.float32),
    }


def freeze_last_block_and_res(actor) -> list[str]:
    names = []
    for name, p in actor.named_parameters():
        allow = name.startswith("enc.layers.1.") or name.startswith("res.")
        p.requires_grad_(allow)
        if allow:
            names.append(name)
    return names


@torch.no_grad()
def closed_loop(actor, start, book, rho, device, *, seed: int = SEED, record: bool = False) -> dict:
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
    tokens: list[int] = []
    residuals: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    obs_l: list[np.ndarray] = []
    prev_l: list[np.ndarray] = []
    ever = {k: False for k in ("frame9", "frame12", "bil")}
    while not done:
        obs = torch.as_tensor(np.stack(env.obs_seq[-160:]), device=device, dtype=torch.float32)
        prev = torch.as_tensor(np.stack(env.prev_seq[-160:]), device=device, dtype=torch.float32)
        logits, res, act = actor(obs, prev)
        tok_a = int(torch.argmax(mask_handoff_logits(logits[-1].float())).item())
        tok, st, _ = select_token(env_t=int(env.t), obs=last_obs(env), actor_tok=tok_a, st=st, v2=v2, rule=None)
        resid = res[-1].detach().cpu().numpy().astype(np.float32)
        if record:
            obs_l.append(env.obs_seq[-1].astype(np.float32))
            prev_l.append(env.prev_seq[-1].astype(np.float32))
            tokens.append(int(tok))
            residuals.append(resid)
            actions.append(act[-1].detach().cpu().numpy().astype(np.float32))
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
        if bil:
            ever["bil"] = True
        if bil and pad4 and wk and e <= 9 + 1e-9 and th <= 8 + 1e-9:
            ever["frame9"] = True
        if bil and pad4 and wk and e <= 12 + 1e-9 and th <= 8 + 1e-9:
            ever["frame12"] = True
    term = _r3_terminal(env, info)
    out = {
        "SUCCESS_RELAXED_12S": int(term.get("SUCCESS_RELAXED") or info.get("SUCCESS_RELAXED_12S") or 0),
        "handoff": int(info.get("handoff") or 0),
        "r3_ready": int(term.get("r3_ready") or 0),
        "reach12": int(ever["frame12"]),
        "reach9": int(ever["frame9"]),
        "ep_len": int(env.t),
        "fail_reason": info.get("fail_reason"),
    }
    if record:
        out.update(
            {
                "tokens": np.asarray(tokens, np.int64),
                "residuals": np.stack(residuals) if residuals else np.zeros((0, 7), np.float32),
                "actions": np.stack(actions) if actions else np.zeros((0, 7), np.float32),
                "obs": np.stack(obs_l) if obs_l else np.zeros((0, 39), np.float32),
                "prev": np.stack(prev_l) if prev_l else np.zeros((0, 7), np.float32),
            }
        )
    return out


def traj_demo(skill: str, tid: str, roll: dict, category: str) -> dict:
    return {
        "traj_id": tid,
        "root_id": root_id(tid),
        "skill": skill,
        "category": category,
        "tokens": np.asarray(roll["tokens"], np.int64),
        "residuals": np.asarray(roll["residuals"], np.float32),
        "actions": np.asarray(roll["actions"], np.float32),
        "obs": np.asarray(roll["obs"], np.float32),
        "prev": np.asarray(roll["prev"], np.float32),
        "verified": 1,
    }


def early_w(n: int, device, w0: float) -> torch.Tensor:
    w = torch.ones(n, device=device, dtype=torch.float32)
    w[: min(EARLY_K, n)] = w0
    return w


def b1_loss(actor, demo, device, early_weight: float) -> tuple[torch.Tensor, dict]:
    obs = torch.as_tensor(demo["obs"], device=device, dtype=torch.float32)
    prev = torch.as_tensor(demo["prev"], device=device, dtype=torch.float32)
    toks = torch.as_tensor(demo["tokens"], device=device, dtype=torch.long)
    tres = torch.as_tensor(demo["residuals"], device=device, dtype=torch.float32)
    tacts = torch.as_tensor(demo["actions"], device=device, dtype=torch.float32)
    logits, res, _ = actor(obs, prev)
    w = early_w(len(toks), device, early_weight)
    loss_res = (((res - tres) ** 2).mean(-1) * w).sum() / w.sum()
    loss_tok = (F.cross_entropy(mask_handoff_logits(logits.float()), toks, reduction="none") * w).sum() / w.sum()
    proto = actor.proto
    motion = torch.clamp(toks, 0, int(actor.n_motion) - 1)
    dec = torch.clamp(proto[motion] + res, -1.0, 1.0)
    if actor.handoff_index is not None:
        dec = torch.where((toks == actor.handoff_index).unsqueeze(-1), torch.zeros_like(dec), dec)
    loss_act = (((dec - tacts) ** 2).mean(-1) * w).sum() / w.sum()
    # first-4 action fidelity (explicit)
    k = min(EARLY_K, len(toks))
    loss_early = ((dec[:k] - tacts[:k]) ** 2).mean()
    loss = loss_res + 0.5 * loss_tok + 1.0 * loss_act + 0.5 * loss_early
    return loss, {"res": float(loss_res.detach()), "tok": float(loss_tok.detach()), "act": float(loss_act.detach())}


def pres_loss(actor, demo, ref, device, early_weight: float) -> tuple[torch.Tensor, dict]:
    obs = torch.as_tensor(demo["obs"], device=device, dtype=torch.float32)
    prev = torch.as_tensor(demo["prev"], device=device, dtype=torch.float32)
    with torch.no_grad():
        logits_r, res_r, act_r = ref(obs, prev)
    logits, res, act = actor(obs, prev)
    n = res.shape[0]
    w = early_w(n, device, early_weight)
    loss_res = (((res - res_r) ** 2).mean(-1) * w).sum() / w.sum()
    loss_act = (((act - act_r) ** 2).mean(-1) * w).sum() / w.sum()
    log_p = F.log_softmax(mask_handoff_logits(logits.float()), dim=-1)
    q = F.softmax(mask_handoff_logits(logits_r.float()), dim=-1)
    # token KL per-step weighted approx via batchmean then scale early by averaging first-k heavier
    loss_kl = F.kl_div(log_p, q, reduction="batchmean")
    loss = loss_res + loss_act + 0.5 * loss_kl
    return loss, {"skill": demo["skill"], "pres_res": float(loss_res.detach()), "pres_act": float(loss_act.detach())}


def param_l2(actor, ref) -> torch.Tensor:
    s = None
    for (n, p), (_, q) in zip(actor.named_parameters(), ref.named_parameters()):
        if not p.requires_grad:
            continue
        d = (p - q.detach()).pow(2).sum()
        s = d if s is None else s + d
    return s if s is not None else torch.zeros(())


def sample_one(pool: list, rng: np.random.Generator):
    return pool[int(rng.integers(0, len(pool)))]


def eval_panel(actor, p64, starts, panel: dict[str, list[str]], book, rho, device) -> list[dict]:
    rows = []
    for skill, tids in panel.items():
        for tid in tids:
            if tid not in starts:
                continue
            r64 = closed_loop(p64, starts[tid], book, rho, device)
            ru = closed_loop(actor, starts[tid], book, rho, device)
            # early residual drift vs p64 on shared initial steps: re-roll short compare at t0-3 via fresh
            # approximate using token agree on full ep
            rows.append(
                {
                    "skill": skill,
                    "traj_id": tid,
                    "root_id": root_id(tid),
                    "P64_SUCCESS": r64["SUCCESS_RELAXED_12S"],
                    "CAND_SUCCESS": ru["SUCCESS_RELAXED_12S"],
                    "gain": int(ru["SUCCESS_RELAXED_12S"] == 1 and r64["SUCCESS_RELAXED_12S"] == 0),
                    "loss": int(ru["SUCCESS_RELAXED_12S"] == 0 and r64["SUCCESS_RELAXED_12S"] == 1),
                    "P64_reach12": r64["reach12"],
                    "CAND_reach12": ru["reach12"],
                    "P64_r3": r64["r3_ready"],
                    "CAND_r3": ru["r3_ready"],
                    "P64_fail": r64.get("fail_reason"),
                    "CAND_fail": ru.get("fail_reason"),
                }
            )
    return rows


def retention(rows: list[dict], skill: str) -> dict:
    base = [r for r in rows if r["skill"] == skill and int(r["P64_SUCCESS"]) == 1]
    n = len(base)
    kept = sum(int(r["CAND_SUCCESS"]) for r in base)
    return {"n_base": n, "kept": kept, "lost": n - kept}


def train_variant(
    label: str,
    *,
    early_w: float,
    lambda_pres: float,
    lambda_kl: float,
    pres_tol: float,
    b1_demos: list[dict],
    prot: dict[str, list[dict]],
    ref,
    book,
    rho,
    device,
    starts,
    dev_panel: dict[str, list[str]],
    p64,
) -> tuple[Any, list[dict], dict]:
    actor, _, _, _ = load_actor(P64_ACTOR, device)
    freeze_last_block_and_res(actor)
    opt = torch.optim.Adam([p for p in actor.parameters() if p.requires_grad], lr=LR)
    rng = np.random.default_rng(SEED + (0 if label == "P96A" else 1))
    curve: list[dict] = []
    best = {"eligible": False, "score": (-1, -1, 0, 0), "path": None, "update": 0, "rows": None}
    print(f"[{label}] training updates={UPDATES} early_w={early_w} lam_pres={lambda_pres} tol={pres_tol}", flush=True)

    for u in range(1, UPDATES + 1):
        actor.train()
        snap = {n: p.detach().clone() for n, p in actor.named_parameters() if p.requires_grad}
        b1 = sample_one(b1_demos, rng)
        pres_batch = [sample_one(prot[sk], rng) for sk in ("A1", "C", "D") if prot[sk]]
        opt.zero_grad(set_to_none=True)
        lb, sb = b1_loss(actor, b1, device, early_w)
        pl_parts = []
        per_sk = {"A1": [], "C": [], "D": []}
        for d in pres_batch:
            lp, sp = pres_loss(actor, d, ref, device, early_w)
            pl_parts.append(lp)
            per_sk[d["skill"]].append(sp["pres_act"])
        loss_pres = torch.stack(pl_parts).mean() if pl_parts else torch.zeros((), device=device)
        # KL on B1 + one protected
        obs = torch.as_tensor(b1["obs"], device=device, dtype=torch.float32)
        prev = torch.as_tensor(b1["prev"], device=device, dtype=torch.float32)
        with torch.no_grad():
            logits_r, _, _ = ref(obs, prev)
        logits, _, _ = actor(obs, prev)
        loss_kl = F.kl_div(
            F.log_softmax(mask_handoff_logits(logits.float()), dim=-1),
            F.softmax(mask_handoff_logits(logits_r.float()), dim=-1),
            reduction="batchmean",
        )
        loss_reg = param_l2(actor, ref)
        with torch.no_grad():
            pre = float(loss_pres.detach())
        loss = LAMBDA_B * lb + lambda_pres * loss_pres + lambda_kl * loss_kl + LAMBDA_REG * loss_reg
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_([p for p in actor.parameters() if p.requires_grad], GRAD_CLIP))
        opt.step()
        # protected constraint: also refuse if C-only preservation worsened beyond tol
        actor.eval()
        reverted = 0
        with torch.no_grad():
            post_parts = []
            post_c = []
            for d in pres_batch:
                lp, sp = pres_loss(actor, d, ref, device, early_w)
                post_parts.append(float(lp.detach()))
                if d["skill"] == "C":
                    post_c.append(sp["pres_act"])
            post = float(np.mean(post_parts)) if post_parts else pre
            # C-specific: compare using a fixed C demo if available
            c_violate = False
            if prot["C"]:
                lp_c, sp_c = pres_loss(actor, prot["C"][0], ref, device, early_w)
                # store running — use absolute early act error threshold
                if sp_c["pres_act"] > 0.05:  # absolute soft guard on C
                    c_violate = sp_c["pres_act"] > 0.05 and post > pre
        if post > pre + pres_tol or (label == "P96B" and c_violate and post > pre):
            with torch.no_grad():
                for n, p in actor.named_parameters():
                    if n in snap:
                        p.copy_(snap[n])
            reverted = 1

        if u % LOG_EVERY == 0 or u == 1 or u == UPDATES:
            row = {
                "variant": label,
                "update": u,
                "loss": float(loss.detach()),
                "loss_b1": float(lb.detach()),
                "loss_pres": float(loss_pres.detach()),
                "loss_kl": float(loss_kl.detach()),
                "grad_norm": gn,
                "reverted": reverted,
                "pres_act_A1": float(np.mean(per_sk["A1"])) if per_sk["A1"] else "",
                "pres_act_C": float(np.mean(per_sk["C"])) if per_sk["C"] else "",
                "pres_act_D": float(np.mean(per_sk["D"])) if per_sk["D"] else "",
            }
            curve.append(row)
            print(
                f"  [{label}] u={u} L={row['loss']:.4f} b1={row['loss_b1']:.4f} "
                f"pres={row['loss_pres']:.4f} rev={reverted}",
                flush=True,
            )

        if u % CL_EVERY == 0 or u == UPDATES:
            print(f"  [{label}] closed-loop @{u}", flush=True)
            rows = eval_panel(actor, p64, starts, dev_panel, book, rho, device)
            b1g = sum(int(r["gain"]) for r in rows if r["skill"] == "B1")
            b1l = sum(int(r["loss"]) for r in rows if r["skill"] == "B1")
            cs = retention(rows, "C")
            a1s = retention(rows, "A1")
            ds = retention(rows, "D")
            eligible = (
                b1g >= 1
                and b1l == 0
                and cs["n_base"] >= 1
                and cs["lost"] == 0
                and a1s["lost"] == 0
                and ds["lost"] == 0
            )
            net = b1g - b1l
            score = (int(eligible), net, cs["kept"], -a1s["lost"] - ds["lost"])
            print(
                f"    DEV b1 +{b1g}/-{b1l} C {cs['kept']}/{cs['n_base']} "
                f"A1 lost={a1s['lost']} D lost={ds['lost']} elig={eligible}",
                flush=True,
            )
            if score >= best["score"]:
                path = CKPT / f"{label}_best.pt"
                save_m111(path, actor, book, epoch=u, rho=rho, extra={"p96": label, "update": u})
                best.update(
                    {"eligible": eligible, "score": score, "path": str(path), "update": u, "rows": rows, "net": net}
                )

    save_m111(CKPT / f"{label}_final.pt", actor, book, epoch=UPDATES, rho=rho, extra={"p96": label})
    if best["path"] and Path(best["path"]).is_file():
        actor, _, _, _ = load_actor(Path(best["path"]), device)
    for p in actor.parameters():
        p.requires_grad_(False)
    actor.eval()
    return actor, curve, best


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[P96] device={device} P64={_sha(P64_ACTOR)}", flush=True)

    starts = lookup_starts()
    ban = eval_banned_roots()
    verified = [t for t in verified_b1() if t in starts and (P87_TEACHER / f"{t}.npz").is_file()]
    # verified B1 are training demos (not exposed final eval) — always allowed for train/dev
    rng = np.random.default_rng(SPLIT_SEED)

    pools = {"A1": [], "B1": [], "C": [], "D": []}
    for tid, s in starts.items():
        if is_banned(tid, ban):
            continue
        pools[s.skill].append(tid)
    for sk in pools:
        pools[sk] = sorted(set(pools[sk]))
        rng.shuffle(pools[sk])

    # ------------------------------------------------------------------
    # Phase 0 — inventory + C screen
    # ------------------------------------------------------------------
    print("[P96] Phase 0 C screen", flush=True)
    p64, book, rho, _ = load_actor(P64_ACTOR, device)
    for p in p64.parameters():
        p.requires_grad_(False)
    p64.eval()
    ref = copy.deepcopy(p64).to(device)
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()

    cache_c = CACHE / "c_screen.json"
    if cache_c.is_file():
        c_screen = json.loads(cache_c.read_text(encoding="utf-8"))
        print(f"  loaded C screen cache n={len(c_screen)}", flush=True)
    else:
        c_screen = []
        for tid in pools["C"][:C_SCREEN_CAP]:
            print(f"  screen C {tid}", flush=True)
            roll = closed_loop(p64, starts[tid], book, rho, device, record=True)
            c_screen.append(
                {
                    "traj_id": tid,
                    "root_id": root_id(tid),
                    "P64_SUCCESS": roll["SUCCESS_RELAXED_12S"],
                    "reach12": roll["reach12"],
                    "ep_len": roll["ep_len"],
                    "fail_reason": roll.get("fail_reason"),
                }
            )
            if roll["SUCCESS_RELAXED_12S"] == 1:
                # cache trajectory arrays separately
                np.savez_compressed(
                    CACHE / f"c_succ_{tid}.npz",
                    tokens=roll["tokens"],
                    residuals=roll["residuals"],
                    actions=roll["actions"],
                    obs=roll["obs"],
                    prev=roll["prev"],
                )
        cache_c.write_text(json.dumps(c_screen, indent=2), encoding="utf-8")

    c_succ_ids = [r["traj_id"] for r in c_screen if int(r["P64_SUCCESS"]) == 1]
    c_succ_roots = sorted({root_id(t) for t in c_succ_ids})
    # known historical successes still in schedule / not eval-banned
    known_extra = []
    for tid in (
        "plan_c__gbi_013_f9f3616bab",
        "plan_c__gbi_351_bc2d6a3b90",
        "C__gbi_066_15297dcc8c__ca9d6bec10e357f4",
        "plan_c__gbi_022_7c876a650b",
        "plan_c__gbi_195_4f90edd916",
    ):
        if tid in starts and not is_banned(tid, ban) and root_id(tid) not in c_succ_roots:
            known_extra.append(tid)

    # verify known extras with P64
    for tid in known_extra:
        print(f"  verify known C {tid}", flush=True)
        roll = closed_loop(p64, starts[tid], book, rho, device, record=True)
        c_screen.append(
            {
                "traj_id": tid,
                "root_id": root_id(tid),
                "P64_SUCCESS": roll["SUCCESS_RELAXED_12S"],
                "reach12": roll["reach12"],
                "ep_len": roll["ep_len"],
                "fail_reason": roll.get("fail_reason"),
                "source": "known_reverify",
            }
        )
        if roll["SUCCESS_RELAXED_12S"] == 1:
            c_succ_ids.append(tid)
            np.savez_compressed(
                CACHE / f"c_succ_{tid}.npz",
                tokens=roll["tokens"],
                residuals=roll["residuals"],
                actions=roll["actions"],
                obs=roll["obs"],
                prev=roll["prev"],
            )
    c_succ_ids = sorted(set(c_succ_ids))
    c_succ_roots = sorted({root_id(t) for t in c_succ_ids})

    inv = f"""# TRAINING_ROOT_INVENTORY

Generated: {_utc()}

## Ban

Exposed evaluation roots excluded (frozen independent, P93 eval panels, P94 independent).
P64 sha `{_sha(P64_ACTOR)}`.

## Independent root counts (not trajectory/augmentation counts)

| Skill | Train-eligible pool ICs | Notes |
|---|---:|---|
| B1 verified expert demos | {len(verified)} | Official P88 replay; roots reused as train demos (not exposed final-eval) |
| A1 pool | {len(pools['A1'])} | to be P64-screened |
| C pool screened | {min(C_SCREEN_CAP, len(pools['C']))} + known re-verify | |
| D pool | {len(pools['D'])} | |

## C coverage gate

| Metric | Value |
|---|---|
| Independent P64-successful C roots found | **{len(c_succ_roots)}** |
| Target min / ideal | {C_ROOT_TARGET_MIN} / {C_ROOT_TARGET_IDEAL} |
| Roots | {c_succ_roots} |

"""
    if len(c_succ_roots) < C_ROOT_TARGET_MIN:
        inv += (
            f"\n**STOP — CASE E.** Cannot meaningfully expand protected C coverage "
            f"({len(c_succ_roots)} < {C_ROOT_TARGET_MIN} independent successful roots). "
            f"Do not silently duplicate one root.\n"
        )
        _write(OUT / "TRAINING_ROOT_INVENTORY.md", inv)
        _write(
            OUT / "FINAL_REPORT.md",
            f"# FINAL_REPORT — P96\n\nGenerated: {_utc()}\n\n"
            f"## CASE E — Insufficient independent successful or recoverable training states\n\n"
            f"Independent P64-successful C roots: **{len(c_succ_roots)}** (need ≥{C_ROOT_TARGET_MIN}).\n"
            f"Roots found: {c_succ_roots}\n\nP64 not replaced. No training run.\n",
        )
        (OUT / "SUMMARY.json").write_text(
            json.dumps(
                {"case": "E", "c_roots": c_succ_roots, "n_c_roots": len(c_succ_roots), "p64_replaced": False},
                indent=2,
            ),
            encoding="utf-8",
        )
        # still write empty stubs for required artifacts
        for name in (
            "MULTISKILL_DATASET_MANIFEST.csv",
            "PHYSICAL_TARGET_VERIFICATION.csv",
            "P96_A_TRAINING_RESULTS.csv",
            "P96_B_TRAINING_RESULTS.csv",
            "EARLY_RESIDUAL_DRIFT.csv",
            "PROTECTED_SKILL_RETENTION.csv",
            "PAIRED_DEVELOPMENT_RESULTS.csv",
        ):
            _write_csv(OUT / name, [])
        _write(OUT / "INDEPENDENT_EVALUATION.md", f"# INDEPENDENT_EVALUATION\n\nGenerated: {_utc()}\n\nNot opened (CASE E).\n")
        print(f"[P96] CASE E — only {len(c_succ_roots)} C roots", flush=True)
        return

    inv += "\n**C gate PASSED** — proceed to dataset expansion and dual training.\n"
    _write(OUT / "TRAINING_ROOT_INVENTORY.md", inv)

    # ------------------------------------------------------------------
    # Phase 1–2 dataset
    # ------------------------------------------------------------------
    print("[P96] Phase 1–2 dataset", flush=True)
    rng2 = np.random.default_rng(SPLIT_SEED + 7)
    ver = list(verified)
    rng2.shuffle(ver)
    b1_dev = ver[:N_DEV_B1]
    b1_train_ids = ver[N_DEV_B1:]

    # Hold out.Dev/ind C from succ list
    c_ids = list(c_succ_ids)
    rng2.shuffle(c_ids)
    # Prefer distinct roots for.Dev
    c_dev, c_train_ids, seen_r = [], [], set()
    for tid in c_ids:
        r = root_id(tid)
        if len(c_dev) < N_DEV_C and r not in seen_r:
            # first N_DEV_C roots ->.Dev (also allowed in train if only few — but prefer disjoint)
            c_dev.append(tid)
            seen_r.add(r)
        else:
            if r not in {root_id(x) for x in c_dev}:
                c_train_ids.append(tid)
            elif tid not in c_dev:
                c_train_ids.append(tid)  # same root only if needed — avoid for independence
    # Ensure train has the non-dev roots
    c_train_ids = [t for t in c_ids if t not in set(c_dev)]
    if not c_train_ids and c_dev:
        # split: keep 1.Dev, rest train
        c_train_ids = c_dev[1:]
        c_dev = c_dev[:1]

    # A1/D screen for preservation
    def screen_skill(skill: str, cand: list[str], target: int, split: str) -> list[dict]:
        out = []
        rows_meta = []
        for tid in cand:
            if len(out) >= target:
                break
            print(f"  screen {skill} {tid}", flush=True)
            roll = closed_loop(p64, starts[tid], book, rho, device, record=True)
            rows_meta.append(
                {
                    "skill": skill,
                    "traj_id": tid,
                    "root_id": root_id(tid),
                    "split": split,
                    "P64_SUCCESS": roll["SUCCESS_RELAXED_12S"],
                    "category": "verified_p64" if roll["SUCCESS_RELAXED_12S"] else "unverified",
                }
            )
            if roll["SUCCESS_RELAXED_12S"] == 1:
                out.append(traj_demo(skill, tid, roll, "verified_p64"))
        return out, rows_meta

    # Reserve ind first from non-C-succ pools
    ind = {
        "B1": [t for t in pools["B1"] if t not in set(verified)][:N_IND_B1],
        "A1": pools["A1"][:N_IND_A1],
        "C": [t for t in pools["C"] if t not in set(c_succ_ids)][:N_IND_C],
        "D": pools["D"][:N_IND_D],
    }
    rem_a1 = [t for t in pools["A1"] if t not in set(ind["A1"])]
    rem_d = [t for t in pools["D"] if t not in set(ind["D"])]
    a1_dev = rem_a1[:N_DEV_A1]
    d_dev = rem_d[:N_DEV_D]
    a1_train, a1_meta = screen_skill("A1", [t for t in rem_a1 if t not in set(a1_dev)], 10, "train")
    d_train, d_meta = screen_skill("D", [t for t in rem_d if t not in set(d_dev)], 10, "train")

    # Load C train demos from cache
    c_train = []
    for tid in c_train_ids:
        zp = CACHE / f"c_succ_{tid}.npz"
        if not zp.is_file():
            roll = closed_loop(p64, starts[tid], book, rho, device, record=True)
            if not roll["SUCCESS_RELAXED_12S"]:
                continue
            np.savez_compressed(
                zp,
                tokens=roll["tokens"],
                residuals=roll["residuals"],
                actions=roll["actions"],
                obs=roll["obs"],
                prev=roll["prev"],
            )
        z = np.load(zp)
        c_train.append(
            {
                "traj_id": tid,
                "root_id": root_id(tid),
                "skill": "C",
                "category": "verified_p64",
                "tokens": np.asarray(z["tokens"], np.int64),
                "residuals": np.asarray(z["residuals"], np.float32),
                "actions": np.asarray(z["actions"], np.float32),
                "obs": np.asarray(z["obs"], np.float32),
                "prev": np.asarray(z["prev"], np.float32),
                "verified": 1,
            }
        )

    # B1 demos + optional on-policy corrective from P95 (expert-rescued B1 loss IC)
    b1_train = []
    man_rows = []
    phys_rows = []
    rbudget = ResidualBudget.from_spec(rho).as_array().astype(np.float32)
    for tid in b1_train_ids + b1_dev:
        d = load_teacher(tid)
        if not d:
            continue
        ok = True
        max_err = 0.0
        for t in range(len(d["tokens"])):
            dec = decode_m111_action(book, int(d["tokens"][t]), d["residuals"][t], rho=rho)
            err = float(np.linalg.norm(dec - d["actions"][t]))
            max_err = max(max_err, err)
            if err > 1e-4:
                ok = False
            phys_rows.append(
                {
                    "traj_id": tid,
                    "skill": "B1",
                    "t": t,
                    "category": "verified_expert",
                    "decode_vs_executed_l2": err,
                    "within_rho": int(np.all(np.abs(d["residuals"][t]) <= rbudget + 1e-6)),
                    "split": "train" if tid in b1_train_ids else "dev",
                }
            )
        man_rows.append(
            {
                "traj_id": tid,
                "root_id": root_id(tid),
                "skill": "B1",
                "category": "verified_expert",
                "split": "train" if tid in b1_train_ids else "dev",
                "n": len(d["tokens"]),
                "parity_ok": int(ok),
                "max_decode_err": max_err,
            }
        )
        if tid in b1_train_ids and ok:
            b1_train.append(d)

    # Student-state corrective: if P95 showed expert prefix rescues k48r_train_00032, add expert demo (already in verified)
    # Mark as validated recoverable student-adjacent only diagnostically — expert traj already included.

    for d in a1_train + c_train + d_train:
        man_rows.append(
            {
                "traj_id": d["traj_id"],
                "root_id": d["root_id"],
                "skill": d["skill"],
                "category": d["category"],
                "split": "train",
                "n": len(d["tokens"]),
                "parity_ok": 1,
                "max_decode_err": 0.0,
            }
        )
        for t in range(len(d["tokens"])):
            dec = decode_m111_action(book, int(d["tokens"][t]), d["residuals"][t], rho=rho)
            err = float(np.linalg.norm(dec - d["actions"][t]))
            phys_rows.append(
                {
                    "traj_id": d["traj_id"],
                    "skill": d["skill"],
                    "t": t,
                    "category": d["category"],
                    "decode_vs_executed_l2": err,
                    "within_rho": int(np.all(np.abs(d["residuals"][t]) <= rbudget + 1e-6)),
                    "split": "train",
                }
            )

    _write_csv(OUT / "MULTISKILL_DATASET_MANIFEST.csv", man_rows)
    _write_csv(OUT / "PHYSICAL_TARGET_VERIFICATION.csv", phys_rows)

    prot = {"A1": a1_train, "C": c_train, "D": d_train}
    print(
        f"[P96] demos B1={len(b1_train)} A1={len(a1_train)} C={len(c_train)} D={len(d_train)} "
        f"devC={c_dev}",
        flush=True,
    )
    if not b1_train or not c_train or not a1_train or not d_train:
        _write(
            OUT / "FINAL_REPORT.md",
            f"# FINAL_REPORT\n\n## CASE E\n\nMissing class demos after screen "
            f"(B1={len(b1_train)},A1={len(a1_train)},C={len(c_train)},D={len(d_train)}).\n",
        )
        print("[P96] CASE E missing demos", flush=True)
        return

    # Also ensure.Dev C list has starts; re-verify
    c_dev = [t for t in c_dev if t in starts]
    if not c_dev:
        c_dev = c_train_ids[:1]

    # Screen.Dev A1/D for baseline labels during eval (not only successes)
    # Dev panel:
    # B1: verified holdouts; C: successful holdouts; A1/D: mixed reserved
    # Expand.Dev A1/D with some known successes for retention tracking
    for tid in list(a1_dev):
        pass
    # Add a few P64-success A1 into.Dev by screening a1_dev quickly
    a1_dev_succ = []
    for tid in a1_dev:
        roll = closed_loop(p64, starts[tid], book, rho, device)
        if roll["SUCCESS_RELAXED_12S"]:
            a1_dev_succ.append(tid)
    d_dev_succ = []
    for tid in d_dev:
        roll = closed_loop(p64, starts[tid], book, rho, device)
        if roll["SUCCESS_RELAXED_12S"]:
            d_dev_succ.append(tid)

    # Dev panel uses full reserved lists (success+fail) for paired metrics; retention uses P64-success subset
    # Ensure at least some A1/D successes: borrow from train demos' ICs into.Dev if needed (root-disjoint preferred)
    if len(a1_dev_succ) < 2 and a1_train:
        for d in a1_train:
            if d["traj_id"] not in a1_dev:
                a1_dev.append(d["traj_id"])
                a1_dev_succ.append(d["traj_id"])
            if len(a1_dev_succ) >= 2:
                break
    if len(d_dev_succ) < 2 and d_train:
        for d in d_train:
            if d["traj_id"] not in d_dev:
                d_dev.append(d["traj_id"])
                d_dev_succ.append(d["traj_id"])
            if len(d_dev_succ) >= 2:
                break

    # Remove borrowed from train pools to avoid identical traj train+.Dev when possible
    a1_train = [d for d in a1_train if d["traj_id"] not in set(a1_dev)]
    d_train = [d for d in d_train if d["traj_id"] not in set(d_dev)]
    # If emptied, restore
    if not a1_train:
        a1_train, _ = screen_skill("A1", [t for t in rem_a1 if t not in set(a1_dev)][:20], 4, "train")
    if not d_train:
        d_train, _ = screen_skill("D", [t for t in rem_d if t not in set(d_dev)][:20], 4, "train")
    prot = {"A1": a1_train, "C": c_train, "D": d_train}

    # Dev C must be P64-success — use c_dev verified
    for tid in list(c_dev):
        zp = CACHE / f"c_succ_{tid}.npz"
        if not zp.is_file():
            roll = closed_loop(p64, starts[tid], book, rho, device, record=True)
            if roll["SUCCESS_RELAXED_12S"]:
                np.savez_compressed(
                    zp,
                    tokens=roll["tokens"],
                    residuals=roll["residuals"],
                    actions=roll["actions"],
                    obs=roll["obs"],
                    prev=roll["prev"],
                )
            else:
                c_dev.remove(tid)

    # Update inventory with final counts
    inv += f"""
## Final training set (independent roots)

| Skill | Train demos | Dev ICs |
|---|---:|---:|
| B1 | {len(b1_train)} | {len(b1_dev)} |
| A1 | {len(a1_train)} | {len(a1_dev)} |
| C | {len(c_train)} roots={sorted({d['root_id'] for d in c_train})} | {c_dev} |
| D | {len(d_train)} | {len(d_dev)} |
"""
    _write(OUT / "TRAINING_ROOT_INVENTORY.md", inv)

    # ------------------------------------------------------------------
    # Phase 3 — P96-A and P96-B
    # ------------------------------------------------------------------
    # Exclude.Dev C from train if overlapping
    c_train = [d for d in c_train if d["traj_id"] not in set(c_dev)]
    if not c_train:
        # allow train_and_dev single-root only if necessary
        c_train = [
            {
                "traj_id": c_dev[0],
                "root_id": root_id(c_dev[0]),
                "skill": "C",
                "category": "verified_p64",
                **{k: np.asarray(np.load(CACHE / f"c_succ_{c_dev[0]}.npz")[k]) for k in ("tokens", "residuals", "actions", "obs", "prev")},
                "verified": 1,
            }
        ]
        c_train[0]["tokens"] = np.asarray(c_train[0]["tokens"], np.int64)
    prot["C"] = c_train

    # Fix dict unpack above carefully if used — rewrite safer:
    if c_train and isinstance(c_train[0].get("tokens"), np.ndarray) is False:
        pass
    # Rebuild if broken
    rebuilt = []
    for d in c_train:
        if "obs" in d and isinstance(d["obs"], np.ndarray):
            rebuilt.append(d)
        else:
            z = np.load(CACHE / f"c_succ_{d['traj_id']}.npz")
            rebuilt.append(
                {
                    "traj_id": d["traj_id"],
                    "root_id": root_id(d["traj_id"]),
                    "skill": "C",
                    "category": "verified_p64",
                    "tokens": np.asarray(z["tokens"], np.int64),
                    "residuals": np.asarray(z["residuals"], np.float32),
                    "actions": np.asarray(z["actions"], np.float32),
                    "obs": np.asarray(z["obs"], np.float32),
                    "prev": np.asarray(z["prev"], np.float32),
                    "verified": 1,
                }
            )
    prot["C"] = rebuilt
    c_train = rebuilt

    # Dev panel for training feedback — use fixed roots
    # Include.Dev C successes + B1 verified.Dev + A1/D reserved
    # Also add P64-successful A1/D into panel for retention
    # Use union of a1_dev and a1_dev_succ etc already in a1_dev
    # Actually.Dev A1 may have few successes — add train A1 ICs only for eval retention? Prefer not.
    # Screen more A1 into.Dev from rem
    while len(a1_dev_succ) < 3:
        for tid in rem_a1:
            if tid in a1_dev or tid in {d["traj_id"] for d in a1_train}:
                continue
            roll = closed_loop(p64, starts[tid], book, rho, device)
            if roll["SUCCESS_RELAXED_12S"]:
                a1_dev.append(tid)
                a1_dev_succ.append(tid)
            if len(a1_dev_succ) >= 3:
                break
        break

    # Dev panel
    # B1.Dev includes both success and fail for paired gains
    # For C use only verified successful.Dev ICs for retention primary
    # Add fail C for rates? Phase 5: track every original P64-successful.Dev IC
    # Dev panel
    # Ensure.Dev panel dict
    # Fix:.Dev_panel variable
    # Define:
    # B1: b1_dev
    # C: c_dev
    # A1: a1_dev
    # D: d_dev
    # Dev_panel =
    # {
    # "B1": b1_dev,
    # "A1": a1_dev,
    # "C": c_dev,
    # "D": d_dev,
    # }

    # Dev_panel (keep name without leading dots)
    # Dev_panel = {...}
    # Use identifier.Dev_panel carefully as.Dev_panel was a bug before
    # Correct:
    # Dev_panel = {"B1": b1_dev, "A1": a1_dev, "C": c_dev, "D": d_dev}
    # Wait use:
    # Dev_panel_map = ...
    # Dev_panel_map:
    # Dev_panel_map = {"B1": b1_dev, "A1": a1_dev, "C": c_dev, "D": d_dev}
    # I'll call it.Dev_panel -> panel_dev
    panel_dev = {"B1": b1_dev, "A1": a1_dev, "C": c_dev, "D": d_dev}
    _write(
        OUT / "SPLITS.json",
        json.dumps({"dev": panel_dev, "ind": ind, "c_train": [d["traj_id"] for d in c_train]}, indent=2),
    )

    actor_a, curve_a, best_a = train_variant(
        "P96A",
        early_w=EARLY_W_A,
        lambda_pres=LAMBDA_PRES_A,
        lambda_kl=LAMBDA_KL_A,
        pres_tol=PRES_TOL_A,
        b1_demos=b1_train,
        prot=prot,
        ref=ref,
        book=book,
        rho=rho,
        device=device,
        starts=starts,
        dev_panel=panel_dev,
        p64=p64,
    )
    _write_csv(OUT / "P96_A_TRAINING_RESULTS.csv", curve_a)

    actor_b, curve_b, best_b = train_variant(
        "P96B",
        early_w=EARLY_W_B,
        lambda_pres=LAMBDA_PRES_B,
        lambda_kl=LAMBDA_KL_B,
        pres_tol=PRES_TOL_B,
        b1_demos=b1_train,
        prot=prot,
        ref=ref,
        book=book,
        rho=rho,
        device=device,
        starts=starts,
        dev_panel=panel_dev,
        p64=p64,
    )
    _write_csv(OUT / "P96_B_TRAINING_RESULTS.csv", curve_b)

    # ------------------------------------------------------------------
    # Phase 5 — development selection
    # ------------------------------------------------------------------
    print("[P96] Phase 5 development selection", flush=True)
    rows_p64 = eval_panel(p64, p64, starts, panel_dev, book, rho, device)
    for r in rows_p64:
        r["candidate"] = "P64"
        r["CAND_SUCCESS"] = r["P64_SUCCESS"]
        r["gain"] = 0
        r["loss"] = 0
    rows_a = eval_panel(actor_a, p64, starts, panel_dev, book, rho, device)
    for r in rows_a:
        r["candidate"] = "P96A"
    rows_b = eval_panel(actor_b, p64, starts, panel_dev, book, rho, device)
    for r in rows_b:
        r["candidate"] = "P96B"
    paired = rows_a + rows_b
    _write_csv(OUT / "PAIRED_DEVELOPMENT_RESULTS.csv", paired)

    def summarize(rows: list[dict], name: str) -> dict:
        b1g = sum(int(r["gain"]) for r in rows if r["skill"] == "B1")
        b1l = sum(int(r["loss"]) for r in rows if r["skill"] == "B1")
        cs = retention(rows, "C")
        a1s = retention(rows, "A1")
        ds = retention(rows, "D")
        eligible = b1g >= 1 and b1l == 0 and cs["n_base"] >= 1 and cs["lost"] == 0 and a1s["lost"] == 0 and ds["lost"] == 0
        return {
            "name": name,
            "b1_gain": b1g,
            "b1_loss": b1l,
            "b1_net": b1g - b1l,
            "c": cs,
            "a1": a1s,
            "d": ds,
            "eligible": eligible,
        }

    sa, sb = summarize(rows_a, "P96A"), summarize(rows_b, "P96B")
    ret_rows = []
    for s in (sa, sb):
        ret_rows.append(
            {
                "candidate": s["name"],
                "b1_gain": s["b1_gain"],
                "b1_loss": s["b1_loss"],
                "C_kept": s["c"]["kept"],
                "C_base": s["c"]["n_base"],
                "A1_kept": s["a1"]["kept"],
                "A1_base": s["a1"]["n_base"],
                "D_kept": s["d"]["kept"],
                "D_base": s["d"]["n_base"],
                "eligible": int(s["eligible"]),
            }
        )
    _write_csv(OUT / "PROTECTED_SKILL_RETENTION.csv", ret_rows)

    # Early residual drift on dev C/A1 between P64 and candidates (t=0 action L2)
    drift_rows = []
    for name, act in (("P96A", actor_a), ("P96B", actor_b)):
        for skill in ("B1", "C", "A1"):
            for tid in panel_dev.get(skill, []):
                seed_all(SEED, deterministic_torch=True)
                env = StabilizeGrpoEnv(
                    book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
                )
                env.reset(starts[tid])
                ensure_inner_capture_buffers(env)
                obs = torch.as_tensor(np.stack(env.obs_seq[-160:]), device=device, dtype=torch.float32)
                prev = torch.as_tensor(np.stack(env.prev_seq[-160:]), device=device, dtype=torch.float32)
                with torch.no_grad():
                    _, r64, a64 = p64(obs, prev)
                    _, rc, ac = act(obs, prev)
                drift_rows.append(
                    {
                        "candidate": name,
                        "skill": skill,
                        "traj_id": tid,
                        "t0_resid_l2": float(torch.linalg.vector_norm(rc[-1] - r64[-1]).item()),
                        "t0_act_l2": float(torch.linalg.vector_norm(ac[-1] - a64[-1]).item()),
                    }
                )
    _write_csv(OUT / "EARLY_RESIDUAL_DRIFT.csv", drift_rows)

    chosen = None
    if sb["eligible"] and (not sa["eligible"] or sb["b1_net"] >= sa["b1_net"]):
        chosen = ("P96B", actor_b, sb, best_b)
    elif sa["eligible"]:
        chosen = ("P96A", actor_a, sa, best_a)
    elif sb["eligible"]:
        chosen = ("P96B", actor_b, sb, best_b)

    case = None
    title = None
    if chosen is None:
        # diagnose failure mode
        if sa["b1_net"] > 0 or sb["b1_net"] > 0:
            case, title = "C", "B1 improves, but protected skills still regress"
        elif sa["c"]["lost"] == 0 and sb["c"]["lost"] == 0 and sa["b1_net"] <= 0 and sb["b1_net"] <= 0:
            case, title = "D", "Protected skills remain stable, but B1 does not improve"
        else:
            case, title = "F", "No improvement despite adequate data coverage; investigate model expressivity or conflicting supervision"
        # If A fails but B would have been needed
        if (not sa["eligible"]) and (sb["c"]["lost"] < sa["c"]["lost"] or sb["b1_net"] > sa["b1_net"]):
            # still not eligible
            pass
        _write(
            OUT / "INDEPENDENT_EVALUATION.md",
            f"# INDEPENDENT_EVALUATION\n\nGenerated: {_utc()}\n\nNot opened — no dev-eligible candidate.\n\n"
            f"P96A: {sa}\n\nP96B: {sb}\n",
        )
    else:
        cname, cactor, cs, cbest = chosen
        # CASE A vs B
        if cname == "P96A":
            case, title = "A", "Expanded data alone produces a unified policy with B1 improvement and protected-skill retention"
        else:
            case, title = "B", "Early-action constraints and on-policy supervision are necessary for joint improvement"
            if sa["eligible"]:
                # both eligible — B still if we preferred B for constraints; if A also eligible, A takes precedence per protocol
                case, title = "A", "Expanded data alone produces a unified policy with B1 improvement and protected-skill retention"
                cname, cactor, cs, cbest = "P96A", actor_a, sa, best_a

        print(f"[P96] Phase 6 independent with {cname}", flush=True)
        freeze_path = CKPT / f"{cname}_frozen_ind.pt"
        save_m111(freeze_path, cactor, book, epoch=cbest.get("update", UPDATES), rho=rho, extra={"p96_frozen": cname})
        actor_i, _, _, _ = load_actor(freeze_path, device)
        for p in actor_i.parameters():
            p.requires_grad_(False)
        actor_i.eval()
        ind_rows = eval_panel(actor_i, p64, starts, ind, book, rho, device)
        _write_csv(OUT / "INDEPENDENT_PAIRED_RESULTS.csv", ind_rows)

        def cls(skill: str):
            rs = [r for r in ind_rows if r["skill"] == skill]
            n = len(rs)
            p64s = sum(int(r["P64_SUCCESS"]) for r in rs)
            us = sum(int(r["CAND_SUCCESS"]) for r in rs)
            g = sum(int(r["gain"]) for r in rs)
            lo = sum(int(r["loss"]) for r in rs)
            rate, lo_ci, hi_ci = wilson(us, n)
            return dict(n=n, p64=p64s, u=us, gain=g, loss=lo, rate=rate, lo=lo_ci, hi=hi_ci)

        b1i, a1i, ci, di = cls("B1"), cls("A1"), cls("C"), cls("D")
        c_ret = retention(ind_rows, "C")
        a_ret = retention(ind_rows, "A1")
        d_ret = retention(ind_rows, "D")
        rates = [x["rate"] for x in (b1i, a1i, ci, di) if x["n"] > 0]
        overall = float(np.mean(rates)) if rates else float("nan")
        ind_ok = (
            b1i["gain"] >= 1
            and b1i["loss"] == 0
            and b1i["u"] > b1i["p64"]
            and (c_ret["n_base"] == 0 or c_ret["lost"] == 0)
            and a_ret["lost"] == 0
            and d_ret["lost"] == 0
            and c_ret["n_base"] >= 1
        )
        if not ind_ok:
            if b1i["u"] > b1i["p64"] and (c_ret["lost"] > 0 or a_ret["lost"] > 0 or d_ret["lost"] > 0):
                case, title = "C", "B1 improves, but protected skills still regress"
            elif b1i["u"] <= b1i["p64"]:
                case, title = "D", "Protected skills remain stable, but B1 does not improve"
        _write(
            OUT / "INDEPENDENT_EVALUATION.md",
            f"""# INDEPENDENT_EVALUATION

Generated: {_utc()}

Frozen candidate: `{cname}` `{freeze_path}` sha `{_sha(freeze_path)}`

## Paired SUCCESS_RELAXED_12S

| Skill | P64 | Candidate | gains | losses | Wilson |
|---|---:|---:|---:|---:|---|
| B1 | {b1i['p64']}/{b1i['n']} | {b1i['u']}/{b1i['n']} | {b1i['gain']} | {b1i['loss']} | [{b1i['lo']:.3f},{b1i['hi']:.3f}] |
| A1 | {a1i['p64']}/{a1i['n']} | {a1i['u']}/{a1i['n']} | {a1i['gain']} | {a1i['loss']} | [{a1i['lo']:.3f},{a1i['hi']:.3f}] |
| C | {ci['p64']}/{ci['n']} | {ci['u']}/{ci['n']} | {ci['gain']} | {ci['loss']} | [{ci['lo']:.3f},{ci['hi']:.3f}] |
| D | {di['p64']}/{di['n']} | {di['u']}/{di['n']} | {di['gain']} | {di['loss']} | [{di['lo']:.3f},{di['hi']:.3f}] |

Class-balanced overall: **{overall:.3f}**

## Conditional retention

| Skill | retained |
|---|---|
| C | {c_ret['kept']}/{c_ret['n_base']} |
| A1 | {a_ret['kept']}/{a_ret['n_base']} |
| D | {d_ret['kept']}/{d_ret['n_base']} |

Independent supports CASE A/B only if B1 net gain and zero protected losses with ≥1 C baseline success.
""",
        )

    final = f"""# FINAL_REPORT — P96 Unified On-Policy Residual Preservation

Generated: {_utc()}

## CASE {case} — {title}

### Integrity

- One unified Actor (enc.layers.1 + res trainable)
- No routers / multi-head / class labels
- Official R3F unchanged
- P64 `{_sha(P64_ACTOR)}` **not automatically replaced**
- C independent successful roots used in training: {len({d['root_id'] for d in c_train})}

### Development

| Candidate | B1 net | C retained | A1 lost | D lost | Eligible |
|---|---:|---|---:|---:|---|
| P96-A | {sa['b1_net']} | {sa['c']['kept']}/{sa['c']['n_base']} | {sa['a1']['lost']} | {sa['d']['lost']} | {sa['eligible']} |
| P96-B | {sb['b1_net']} | {sb['c']['kept']}/{sb['c']['n_base']} | {sb['a1']['lost']} | {sb['d']['lost']} | {sb['eligible']} |

Chosen: **{chosen[0] if chosen else 'none'}**

### Independent

See `INDEPENDENT_EVALUATION.md`.

### Note

P96-A tests expanded data alone; P96-B adds early residual weighting, stronger preservation, and update constraints addressing P95 residual drift.
"""
    _write(OUT / "FINAL_REPORT.md", final)
    (OUT / "SUMMARY.json").write_text(
        json.dumps(
            {
                "case": case,
                "title": title,
                "p96a": sa,
                "p96b": sb,
                "chosen": chosen[0] if chosen else None,
                "n_c_train_roots": len({d["root_id"] for d in c_train}),
                "p64_replaced": False,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"[P96] DONE CASE {case} — {title}", flush=True)


if __name__ == "__main__":
    main()
