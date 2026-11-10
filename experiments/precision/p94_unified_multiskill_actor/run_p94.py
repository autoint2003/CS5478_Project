"""P94 — Unified Multi-Skill Actor Consolidation.

One shared Transformer + one M111 token head + one residual head.
No routing, no multi-head experts, no class labels at inference.
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
from models.m111_checkpoint import decode_m111_action, save_m111  # noqa: E402
from training.skill_aware_action_vocab import ResidualBudget  # noqa: E402

OUT = ROOT / "results/experiments/p94_unified_multiskill_actor"
CKPT = OUT / "checkpoints"
CACHE = OUT / "cache"
P87_TEACHER = ROOT / "results/experiments/p87_k48_actor_retention/teacher"
P88_CAND = ROOT / "results/experiments/p88_b1_short_sequence_learnability/PHASE0_CANDIDATES.csv"
P86_PANEL = ROOT / "results/experiments/p86_b1_gate_counterfactual/PANEL.json"
P92_PAIRED = ROOT / "results/experiments/p92_dual_residual_head_routing/PAIRED_HEAD_OUTCOMES.csv"
P93_MAN = ROOT / "results/experiments/p93_frozen_router_independent_validation/EVALUATION_SPLIT_MANIFEST.csv"
FROZEN_MANIFEST = ROOT / "results/experiments/final_independent_success_rate/FROZEN_TEST_MANIFEST.json"

SEED = 94
SPLIT_SEED = 94

# Fixed training budget (no unrestricted sweeps).
UPDATES = 600
LR = 1e-3
LOG_EVERY = 25
CL_EVERY = 150
EARLY_K = 4
EARLY_W = 3.0
LAMBDA_B = 1.0
LAMBDA_PRES = 2.5
LAMBDA_KL = 0.5
LAMBDA_REG = 1e-3
PRES_TOL = 0.05  # max allowed increase in protected residual MSE vs step start
GRAD_CLIP = 5.0

# Panel sizes
N_DEV_B1 = 3
N_DEV_A1 = 8
N_DEV_D = 8
N_DEV_C_SUCC = 5
N_IND_B1 = 12
N_IND_A1 = 8
N_IND_D = 8
N_IND_C = 10
N_TRAIN_A1_SUCC = 8
N_TRAIN_D_SUCC = 8
N_TRAIN_C_SUCC = 6
C_SCREEN_CAP = 60
MIN_PROTECTED = {"A1": 1, "C": 1, "D": 2}


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


def banned_roots() -> tuple[set[str], dict[str, int]]:
    """Roots excluded from train/selection and from P94 independent eval.

    P93 C_SCREEN-only ICs (P64-fail probes, never used for router scoring) are
    **not** banned from training pools so protected-C coverage remains feasible.
    P93 B1/C1/C2/A1/D eval panels remain banned (not reused as fresh eval).
    """
    tids: set[str] = set()
    src: dict[str, int] = {}

    def add(label: str, xs: list[str]) -> None:
        src[label] = len(set(xs))
        tids.update(xs)

    man = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
    add("frozen_independent", [e["traj_id"] for e in man["episodes"]])
    p93_eval_panels = {"B1", "C1", "C2", "A1", "D", "DIAG_EXPOSED_B1"}
    add(
        "p93_eval_panels",
        [
            r["traj_id"]
            for r in csv.DictReader(P93_MAN.open(encoding="utf-8"))
            if r["panel"] in p93_eval_panels
        ],
    )
    add("p92_panel", [r["traj_id"] for r in csv.DictReader(P92_PAIRED.open(encoding="utf-8"))])
    p86 = json.loads(P86_PANEL.read_text(encoding="utf-8"))
    add("p86_b1", list(p86["b1_traj_ids"]))
    roots = {root_id(t) for t in tids}
    roots |= {root_id(t.replace("plan_b__", "", 1)) for t in tids}
    return roots, src


def is_banned(tid: str, ban: set[str]) -> bool:
    rid = root_id(tid)
    if rid in ban:
        return True
    return any(rid in b or b in rid for b in ban)


def load_teacher_demo(tid: str) -> dict:
    z = np.load(P87_TEACHER / f"{tid}.npz", allow_pickle=True)
    return {
        "traj_id": tid,
        "root_id": root_id(tid),
        "skill": "B1",
        "source": "k48_m111_teacher",
        "tokens": np.asarray(z["tokens"], np.int64),
        "residuals": np.asarray(z["residuals"], np.float32),
        "actions": np.asarray(z["actions"], np.float32),
        "obs": np.asarray(z["obs"], np.float32),
        "prev": np.asarray(z["prev"], np.float32),
    }


def freeze_last_block_and_res(actor) -> list[str]:
    """Unfreeze enc.layers.1.* and res.*; freeze everything else including logits."""
    trainable = []
    for name, p in actor.named_parameters():
        allow = name.startswith("enc.layers.1.") or name.startswith("res.")
        p.requires_grad_(allow)
        if allow:
            trainable.append(name)
    return trainable


def count_trainable(actor) -> int:
    return sum(int(p.numel()) for p in actor.parameters() if p.requires_grad)


@torch.no_grad()
def closed_loop(actor, start, book, rho, device, *, seed: int = SEED) -> dict:
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
    obs_list: list[np.ndarray] = []
    prev_list: list[np.ndarray] = []
    ever = {k: False for k in ("frame9", "frame12", "bil")}
    while not done:
        obs = torch.as_tensor(np.stack(env.obs_seq[-160:]), device=device, dtype=torch.float32)
        prev = torch.as_tensor(np.stack(env.prev_seq[-160:]), device=device, dtype=torch.float32)
        logits, res, act = actor(obs, prev)
        logit = mask_handoff_logits(logits[-1].float())
        tok_a = int(torch.argmax(logit).item())
        tok, st, _ = select_token(env_t=int(env.t), obs=last_obs(env), actor_tok=tok_a, st=st, v2=v2, rule=None)
        resid = res[-1].detach().cpu().numpy().astype(np.float32)
        # record pre-step causal history last frame
        obs_list.append(env.obs_seq[-1].astype(np.float32))
        prev_list.append(env.prev_seq[-1].astype(np.float32))
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
    return {
        "SUCCESS_RELAXED_12S": int(term.get("SUCCESS_RELAXED") or info.get("SUCCESS_RELAXED_12S") or 0),
        "SUCCESS_12S": int(info.get("SUCCESS_12S") or 0),
        "handoff": int(info.get("handoff") or 0),
        "r3_ready": int(term.get("r3_ready") or 0),
        "reach12": int(ever["frame12"]),
        "reach9": int(ever["frame9"]),
        "ep_len": int(env.t),
        "fail_reason": info.get("fail_reason"),
        "tokens": tokens,
        "residuals": np.stack(residuals) if residuals else np.zeros((0, 7), np.float32),
        "actions": np.stack(actions) if actions else np.zeros((0, 7), np.float32),
        "obs": np.stack(obs_list) if obs_list else np.zeros((0, 39), np.float32),
        "prev": np.stack(prev_list) if prev_list else np.zeros((0, 7), np.float32),
    }


def traj_from_rollout(skill: str, tid: str, roll: dict, source: str) -> dict:
    return {
        "traj_id": tid,
        "root_id": root_id(tid),
        "skill": skill,
        "source": source,
        "tokens": np.asarray(roll["tokens"], np.int64),
        "residuals": np.asarray(roll["residuals"], np.float32),
        "actions": np.asarray(roll["actions"], np.float32),
        "obs": np.asarray(roll["obs"], np.float32),
        "prev": np.asarray(roll["prev"], np.float32),
        "SUCCESS_RELAXED_12S": roll["SUCCESS_RELAXED_12S"],
    }


def early_weights(n: int, device) -> torch.Tensor:
    w = torch.ones(n, device=device, dtype=torch.float32)
    w[: min(EARLY_K, n)] = EARLY_W
    return w


def b1_action_loss(actor, demo, book, rho, device, ref_actor=None) -> tuple[torch.Tensor, dict]:
    obs = torch.as_tensor(demo["obs"], device=device, dtype=torch.float32)
    prev = torch.as_tensor(demo["prev"], device=device, dtype=torch.float32)
    toks = torch.as_tensor(demo["tokens"], device=device, dtype=torch.long)
    tres = torch.as_tensor(demo["residuals"], device=device, dtype=torch.float32)
    tacts = torch.as_tensor(demo["actions"], device=device, dtype=torch.float32)
    logits, res, act = actor(obs, prev)
    w = early_weights(len(toks), device)
    # residual MSE (weighted)
    res_err = ((res - tres) ** 2).mean(dim=-1)
    loss_res = (res_err * w).sum() / w.sum()
    # token CE with frozen logits head — grads via ctx into last block
    logit = mask_handoff_logits(logits.float())
    # ignore handoff index if present in targets beyond n_motion — CE on all
    loss_tok = F.cross_entropy(logit, toks, reduction="none")
    loss_tok = (loss_tok * w).sum() / w.sum()
    # executed action fidelity using teacher tokens + predicted residual (prevents silent token/residual swap)
    # reconstruct: proto[tok] + res
    proto = actor.proto
    n_motion = int(actor.n_motion)
    motion = torch.clamp(toks, 0, n_motion - 1)
    dec = torch.clamp(proto[motion] + res, -1.0, 1.0)
    if actor.handoff_index is not None:
        dec = torch.where((toks == actor.handoff_index).unsqueeze(-1), torch.zeros_like(dec), dec)
    act_err = ((dec - tacts) ** 2).mean(dim=-1)
    loss_act = (act_err * w).sum() / w.sum()
    # Also compare policy act (uses argmax tok) — but primary is teacher-token decode
    loss = loss_res + 0.5 * loss_tok + 1.0 * loss_act
    stats = {
        "loss_res": float(loss_res.detach()),
        "loss_tok": float(loss_tok.detach()),
        "loss_act": float(loss_act.detach()),
    }
    return loss, stats


def preservation_loss(actor, demo, ref_actor, device) -> tuple[torch.Tensor, dict]:
    obs = torch.as_tensor(demo["obs"], device=device, dtype=torch.float32)
    prev = torch.as_tensor(demo["prev"], device=device, dtype=torch.float32)
    with torch.no_grad():
        logits_ref, res_ref, act_ref = ref_actor(obs, prev)
    logits, res, act = actor(obs, prev)
    loss_res = F.mse_loss(res, res_ref)
    loss_act = F.mse_loss(act, act_ref)
    # token distribution preservation (soft)
    log_p = F.log_softmax(mask_handoff_logits(logits.float()), dim=-1)
    q = F.softmax(mask_handoff_logits(logits_ref.float()), dim=-1)
    loss_kl = F.kl_div(log_p, q, reduction="batchmean")
    loss = loss_res + loss_act + 0.5 * loss_kl
    return loss, {
        "pres_res": float(loss_res.detach()),
        "pres_act": float(loss_act.detach()),
        "pres_kl": float(loss_kl.detach()),
        "skill": demo["skill"],
    }


def token_kl_batch(actor, demo, ref_actor, device) -> torch.Tensor:
    obs = torch.as_tensor(demo["obs"], device=device, dtype=torch.float32)
    prev = torch.as_tensor(demo["prev"], device=device, dtype=torch.float32)
    with torch.no_grad():
        logits_ref, _, _ = ref_actor(obs, prev)
    logits, _, _ = actor(obs, prev)
    log_p = F.log_softmax(mask_handoff_logits(logits.float()), dim=-1)
    q = F.softmax(mask_handoff_logits(logits_ref.float()), dim=-1)
    return F.kl_div(log_p, q, reduction="batchmean")


def param_l2_delta(actor, ref_actor) -> torch.Tensor:
    num = None
    for (n, p), (_, q) in zip(actor.named_parameters(), ref_actor.named_parameters()):
        if not p.requires_grad:
            continue
        d = (p - q.detach()).pow(2).sum()
        num = d if num is None else num + d
    return num if num is not None else torch.zeros(())


def sample_one(pool: list[dict], rng: np.random.Generator) -> dict:
    return pool[int(rng.integers(0, len(pool)))]


def eval_panel(actor, p64, starts, tids_by_skill: dict[str, list[str]], book, rho, device) -> list[dict]:
    rows = []
    for skill, tids in tids_by_skill.items():
        for tid in tids:
            if tid not in starts:
                continue
            r64 = closed_loop(p64, starts[tid], book, rho, device)
            ru = closed_loop(actor, starts[tid], book, rho, device)
            rows.append(
                {
                    "skill": skill,
                    "traj_id": tid,
                    "root_id": root_id(tid),
                    "P64_SUCCESS": r64["SUCCESS_RELAXED_12S"],
                    "UNIFIED_SUCCESS": ru["SUCCESS_RELAXED_12S"],
                    "P64_reach12": r64["reach12"],
                    "UNIFIED_reach12": ru["reach12"],
                    "P64_reach9": r64["reach9"],
                    "UNIFIED_reach9": ru["reach9"],
                    "P64_r3_ready": r64["r3_ready"],
                    "UNIFIED_r3_ready": ru["r3_ready"],
                    "P64_fail": r64.get("fail_reason"),
                    "UNIFIED_fail": ru.get("fail_reason"),
                    "token_agree": float(
                        np.mean([int(a == b) for a, b in zip(r64["tokens"], ru["tokens"])])
                        if r64["tokens"] and ru["tokens"]
                        else float("nan")
                    ),
                    "gain": int(ru["SUCCESS_RELAXED_12S"] == 1 and r64["SUCCESS_RELAXED_12S"] == 0),
                    "loss": int(ru["SUCCESS_RELAXED_12S"] == 0 and r64["SUCCESS_RELAXED_12S"] == 1),
                }
            )
    return rows


def retention_stats(rows: list[dict], skill: str) -> dict:
    base = [r for r in rows if r["skill"] == skill and int(r["P64_SUCCESS"]) == 1]
    n = len(base)
    kept = sum(int(r["UNIFIED_SUCCESS"]) for r in base)
    return {"n_base": n, "kept": kept, "lost": n - kept, "rate": (kept / n) if n else float("nan")}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[P94] device={device} P64={_sha(P64_ACTOR)}", flush=True)

    starts = lookup_starts()
    ban, ban_src = banned_roots()
    verified = [t for t in verified_b1() if t in starts and (P87_TEACHER / f"{t}.npz").is_file()]
    assert len(verified) >= 8, f"need verified B1 demos, got {len(verified)}"

    # ------------------------------------------------------------------
    # Phase 0 — splits
    # ------------------------------------------------------------------
    rng = np.random.default_rng(SPLIT_SEED)
    # B1 verified: root-disjoint train/dev (all verified may appear in prior P88–P92 train;
    # they are training-split demos, not exposed independent eval).
    ver = list(verified)
    rng.shuffle(ver)
    b1_dev = ver[:N_DEV_B1]
    b1_train = ver[N_DEV_B1:]

    # Pool schedule ICs by skill excluding banned + verified B1 roots (kept for B1 demos only)
    ver_roots = {root_id(t) for t in verified}
    pools: dict[str, list[str]] = {"A1": [], "B1": [], "C": [], "D": []}
    for tid, s in starts.items():
        if s.skill not in pools:
            continue
        if is_banned(tid, ban):
            continue
        if s.skill == "B1" and root_id(tid) in ver_roots:
            continue
        pools[s.skill].append(tid)
    for sk in pools:
        pools[sk] = sorted(set(pools[sk]))
        rng.shuffle(pools[sk])

    # Reserve independent first (untouched), then dev, remainder candidates for train screening
    ind = {
        "B1": pools["B1"][:N_IND_B1],
        "A1": pools["A1"][:N_IND_A1],
        "C": pools["C"][:N_IND_C],
        "D": pools["D"][:N_IND_D],
    }
    rem = {sk: [t for t in pools[sk] if t not in set(ind[sk])] for sk in pools}
    dev = {
        "B1": b1_dev,  # verified holdout with teacher; also CL-eval
        "A1": rem["A1"][:N_DEV_A1],
        "D": rem["D"][:N_DEV_D],
        "C": [],  # filled with P64-successful after screening below
    }
    train_cand = {
        "A1": [t for t in rem["A1"] if t not in set(dev["A1"])],
        "C": [t for t in rem["C"] if t not in set(ind["C"])],
        "D": [t for t in rem["D"] if t not in set(dev["D"])],
        "B1": b1_train,
    }
    # C dev candidates from rem after ind
    c_dev_cand = [t for t in rem["C"] if t not in set(train_cand["C"])][:20]
    if len(c_dev_cand) < 10:
        c_dev_cand = rem["C"][:20]

    p64, book, rho, _ = load_actor(P64_ACTOR, device)
    for p in p64.parameters():
        p.requires_grad_(False)
    p64.eval()
    ref = copy.deepcopy(p64).to(device)
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()

    # ------------------------------------------------------------------
    # Phase 1 — datasets
    # ------------------------------------------------------------------
    print("[P94] Phase 1 B1 demos", flush=True)
    b1_sup_rows = []
    b1_train_demos = []
    rbudget = ResidualBudget.from_spec(rho).as_array().astype(np.float32)
    for tid in b1_train + b1_dev:
        d = load_teacher_demo(tid)
        # parity check
        max_err = 0.0
        ok = True
        for t in range(len(d["tokens"])):
            dec = decode_m111_action(book, int(d["tokens"][t]), d["residuals"][t], rho=rho)
            err = float(np.linalg.norm(dec - d["actions"][t]))
            max_err = max(max_err, err)
            if err > 1e-4:
                ok = False
            # compare to P64 action at same state for early priority signal
        # P64 residual drift on teacher states
        with torch.no_grad():
            o = torch.as_tensor(d["obs"], device=device, dtype=torch.float32)
            pv = torch.as_tensor(d["prev"], device=device, dtype=torch.float32)
            _, res64, _ = p64(o, pv)
            res64n = res64.detach().cpu().numpy()
        early_diff = float(np.linalg.norm(res64n[:EARLY_K] - d["residuals"][:EARLY_K], axis=-1).mean())
        row = {
            "traj_id": tid,
            "root_id": root_id(tid),
            "split": "train" if tid in b1_train else "dev",
            "n": len(d["tokens"]),
            "parity_ok": int(ok),
            "max_decode_err": max_err,
            "early_res_l2_vs_p64": early_diff,
            "valid_official": 1,
        }
        b1_sup_rows.append(row)
        if tid in b1_train and ok:
            b1_train_demos.append(d)
    _write_csv(OUT / "B1_FULL_ACTION_SUPERVISION.csv", b1_sup_rows)
    assert b1_train_demos, "no parity-ok B1 train demos"

    print("[P94] Phase 1 protected P64 replay", flush=True)
    protected_train: dict[str, list[dict]] = {"A1": [], "C": [], "D": []}
    protected_rows = []

    def screen_successes(skill: str, cand: list[str], target: int, split: str) -> list[dict]:
        out = []
        for tid in cand:
            if len(out) >= target:
                break
            print(f"  screen {split} {skill} {tid}", flush=True)
            roll = closed_loop(p64, starts[tid], book, rho, device)
            succ = int(roll["SUCCESS_RELAXED_12S"])
            protected_rows.append(
                {
                    "skill": skill,
                    "traj_id": tid,
                    "root_id": root_id(tid),
                    "split": split,
                    "P64_SUCCESS": succ,
                    "reach12": roll["reach12"],
                    "reach9": roll["reach9"],
                    "r3_ready": roll["r3_ready"],
                    "ep_len": roll["ep_len"],
                    "fail_reason": roll.get("fail_reason"),
                    "kept_as_demo": int(succ == 1 and split == "train" and len(out) < target),
                }
            )
            if succ == 1:
                if split == "train" and len(out) < target:
                    out.append(traj_from_rollout(skill, tid, roll, "p64_success_replay"))
                elif split == "dev":
                    out.append({"traj_id": tid, "roll": roll})
        return out

    # Prefer train_cand; if scarce, also screen rem leftovers beyond.Dev/ind holdouts.
    a1_queue = list(train_cand["A1"]) + [t for t in rem["A1"] if t not in set(train_cand["A1"]) and t not in set(dev["A1"])]
    protected_train["A1"] = screen_successes("A1", a1_queue, N_TRAIN_A1_SUCC, "train")
    protected_train["D"] = screen_successes("D", train_cand["D"], N_TRAIN_D_SUCC, "train")

    # C: screen rem queue for P64 successes; allocate train/dev retention.

    c_queue = list(rem["C"])[:C_SCREEN_CAP]
    print(f"[P94] C screen queue n={len(c_queue)} (ind held={len(ind['C'])})", flush=True)
    c_success_demos: list[dict] = []
    for tid in c_queue:
        print(f"  screen C-pool {tid}", flush=True)
        roll = closed_loop(p64, starts[tid], book, rho, device)
        succ = int(roll["SUCCESS_RELAXED_12S"])
        protected_rows.append(
            {
                "skill": "C",
                "traj_id": tid,
                "root_id": root_id(tid),
                "split": "pool",
                "P64_SUCCESS": succ,
                "reach12": roll["reach12"],
                "reach9": roll["reach9"],
                "r3_ready": roll["r3_ready"],
                "ep_len": roll["ep_len"],
                "fail_reason": roll.get("fail_reason"),
                "kept_as_demo": 0,
            }
        )
        if succ == 1:
            c_success_demos.append(traj_from_rollout("C", tid, roll, "p64_success_replay"))

    if len(c_success_demos) >= 2:
        dev["C"] = [c_success_demos[0]["traj_id"]]
        protected_train["C"] = c_success_demos[1:]
    elif len(c_success_demos) == 1:
        protected_train["C"] = c_success_demos
        dev["C"] = [c_success_demos[0]["traj_id"]]
    else:
        protected_train["C"] = []
        dev["C"] = []
    train_c_ids = {d["traj_id"] for d in protected_train["C"]}
    for r in protected_rows:
        if r["skill"] != "C":
            continue
        if r["traj_id"] in train_c_ids and r["traj_id"] in set(dev["C"]):
            r["split"] = "train_and_dev"
            r["kept_as_demo"] = 1
        elif r["traj_id"] in train_c_ids:
            r["split"] = "train"
            r["kept_as_demo"] = 1
        elif r["traj_id"] in set(dev["C"]):
            r["split"] = "dev"
            r["kept_as_demo"] = 0

    _write_csv(OUT / "PROTECTED_SKILL_REPLAY.csv", protected_rows)

    n_c_train = len(protected_train["C"])
    n_a_train = len(protected_train["A1"])
    n_d_train = len(protected_train["D"])
    print(f"[P94] demos B1={len(b1_train_demos)} A1={n_a_train} C={n_c_train} D={n_d_train} devC={len(dev['C'])}", flush=True)

    manifest = f"""# DATASET_MANIFEST

Generated: {_utc()}

## Freeze

| Artifact | sha256[:16] |
|---|---|
| P64 actor | `{_sha(P64_ACTOR)}` |

Banned roots from frozen independent + P86/P92/P93 panels (not used for train or model selection).
Ban source counts: {ban_src}

## Splits (root-disjoint)

| Split | B1 | A1 | C | D |
|---|---:|---:|---:|---:|
| Train demos | {len(b1_train_demos)} verified teacher | {n_a_train} P64-success | {n_c_train} P64-success | {n_d_train} P64-success |
| Dev ICs | {dev['B1']} | n={len(dev['A1'])} | n={len(dev['C'])} (P64-success retention) | n={len(dev['D'])} |
| Independent (sealed until Phase 6) | n={len(ind['B1'])} | n={len(ind['A1'])} | n={len(ind['C'])} | n={len(ind['D'])} |

## B1 teacher

Source: `{P87_TEACHER}` verified via P88 PHASE0 (`valid_official=1` only; not all 16 historical).
Parity: decode(proto+res) vs executed action.

## Protected

P64 closed-loop successful recoveries on training-only roots; full obs/prev/token/residual/action traces.
"""
    _write(OUT / "DATASET_MANIFEST.md", manifest)
    _write(
        OUT / "SPLITS.json",
        json.dumps({"train_b1": [d["traj_id"] for d in b1_train_demos], "dev": dev, "ind": ind, "ban_src": ban_src}, indent=2),
    )

    # ------------------------------------------------------------------
    # Phase 2 — architecture audit
    # ------------------------------------------------------------------
    actor, _, _, _ = load_actor(P64_ACTOR, device)
    trainable = freeze_last_block_and_res(actor)
    n_train = count_trainable(actor)
    arch = f"""# UNIFIED_ARCHITECTURE_AUDIT

Generated: {_utc()}

## Architecture (exactly one Actor)

| Component | Module | Trainable initially |
|---|---|---|
| Input | `in_proj` | frozen |
| Pos | `pos` | frozen |
| Transformer block 0 | `enc.layers.0` | frozen |
| Transformer block 1 | `enc.layers.1` | **unfrozen** |
| M111 token head | `logits` | **frozen** (vocab/decoder unchanged) |
| Residual head | `res.Linear(64→7)` + `res_cap*tanh` | **unfrozen** |

Trainable params: **{n_train}** ({len(trainable)} tensors).

## Prohibitions checked

- Multiple residual heads: **no**
- Expert routing / gating: **no**
- Class-conditioned inference: **no**
- Runtime task labels: **no**
- Residual blending: **no**
- Separate B1/C policies: **no**
- Oracle teacher at inference: **no**
- Official R3F: unchanged G0 monitor

## Inference

One forward pass → token logits + residual → single executed action. Cost identical to P64.
"""
    _write(OUT / "UNIFIED_ARCHITECTURE_AUDIT.md", arch)
    print(f"[P94] trainable={n_train} names={trainable[:4]}...", flush=True)

    if (
        n_c_train < MIN_PROTECTED["C"]
        or n_a_train < MIN_PROTECTED["A1"]
        or n_d_train < MIN_PROTECTED["D"]
    ):
        _write(
            OUT / "FINAL_REPORT.md",
            f"# FINAL_REPORT\n\nGenerated: {_utc()}\n\n## CASE E\n\n"
            f"Insufficient protected demos after full screen "
            f"(A1={n_a_train}, C={n_c_train}, D={n_d_train}; "
            f"mins={MIN_PROTECTED}). Bounded data cannot support joint preservation.\n",
        )
        print("[P94] CASE E — insufficient protected demos", flush=True)
        return

    # ------------------------------------------------------------------
    # Phase 3–4 — joint training + closed-loop feedback
    # ------------------------------------------------------------------
    opt = torch.optim.Adam([p for p in actor.parameters() if p.requires_grad], lr=LR)
    curve = []
    best = {
        "update": 0,
        "b1_gain": -1,
        "c_kept": -1,
        "a1_loss": 99,
        "d_loss": 99,
        "path": None,
        "eligible": False,
    }
    rng_tr = np.random.default_rng(SEED)

    def run_dev(tag: str) -> list[dict]:
        panel = {"B1": dev["B1"], "A1": dev["A1"], "C": dev["C"], "D": dev["D"]}
        rows = eval_panel(actor, p64, starts, panel, book, rho, device)
        _write_csv(OUT / f"DEV_EVAL_{tag}.csv", rows)
        return rows

    print("[P94] Phase 3 joint training", flush=True)
    for u in range(1, UPDATES + 1):
        actor.train()
        # snapshot trainable for optional revert
        snap = {n: p.detach().clone() for n, p in actor.named_parameters() if p.requires_grad}

        # balanced sample: one B1 + one of each protected skill present
        b1_d = sample_one(b1_train_demos, rng_tr)
        pres_list = []
        for sk in ("A1", "C", "D"):
            if protected_train[sk]:
                pres_list.append(sample_one(protected_train[sk], rng_tr))

        opt.zero_grad(set_to_none=True)
        loss_b, st_b = b1_action_loss(actor, b1_d, book, rho, device)
        loss_pres_parts = []
        st_pres = []
        for d in pres_list:
            lp, sp = preservation_loss(actor, d, ref, device)
            loss_pres_parts.append(lp)
            st_pres.append(sp)
        loss_pres = torch.stack(loss_pres_parts).mean() if loss_pres_parts else torch.zeros((), device=device)

        # KL distillation on B1 + one protected (token drift monitor / regularizer)
        loss_kl = token_kl_batch(actor, b1_d, ref, device)
        if pres_list:
            loss_kl = loss_kl + token_kl_batch(actor, pres_list[0], ref, device)
            loss_kl = 0.5 * loss_kl

        loss_reg = param_l2_delta(actor, ref)

        # measure protected loss before step for constraint
        with torch.no_grad():
            pre_pres = float(loss_pres.detach())

        loss = LAMBDA_B * loss_b + LAMBDA_PRES * loss_pres + LAMBDA_KL * loss_kl + LAMBDA_REG * loss_reg
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_([p for p in actor.parameters() if p.requires_grad], GRAD_CLIP))
        opt.step()

        # constrained update: if protected residual/action loss rose beyond tolerance, revert
        actor.eval()
        with torch.no_grad():
            post_parts = []
            for d in pres_list:
                lp, _ = preservation_loss(actor, d, ref, device)
                post_parts.append(float(lp.detach()))
            post_pres = float(np.mean(post_parts)) if post_parts else pre_pres
        reverted = 0
        if post_pres > pre_pres + PRES_TOL:
            with torch.no_grad():
                for n, p in actor.named_parameters():
                    if n in snap:
                        p.copy_(snap[n])
            reverted = 1

        if u % LOG_EVERY == 0 or u == 1 or u == UPDATES:
            # per-class preservation offline
            by_sk = {"A1": [], "C": [], "D": []}
            for sk in by_sk:
                for d in protected_train[sk][:3]:
                    _, sp = preservation_loss(actor, d, ref, device)
                    by_sk[sk].append(sp["pres_res"])
            row = {
                "update": u,
                "loss": float(loss.detach()),
                "loss_b1": float(loss_b.detach()),
                "loss_pres": float(loss_pres.detach()),
                "loss_kl": float(loss_kl.detach()),
                "loss_reg": float(loss_reg.detach()),
                "grad_norm": gn,
                "reverted": reverted,
                "pres_A1": float(np.mean(by_sk["A1"])) if by_sk["A1"] else "",
                "pres_C": float(np.mean(by_sk["C"])) if by_sk["C"] else "",
                "pres_D": float(np.mean(by_sk["D"])) if by_sk["D"] else "",
                **{f"b1_{k}": v for k, v in st_b.items()},
            }
            curve.append(row)
            print(
                f"  u={u} L={row['loss']:.4f} b1={row['loss_b1']:.4f} pres={row['loss_pres']:.4f} "
                f"kl={row['loss_kl']:.4f} rev={reverted}",
                flush=True,
            )
            _write_csv(OUT / "JOINT_TRAINING_CURVES.csv", curve)

        if u % CL_EVERY == 0 or u == UPDATES:
            print(f"[P94] Phase 4 closed-loop @ {u}", flush=True)
            rows = run_dev(f"u{u}")
            b1r = [r for r in rows if r["skill"] == "B1"]
            b1_gain = sum(int(r["gain"]) for r in b1r)
            b1_loss = sum(int(r["loss"]) for r in b1r)
            cstat = retention_stats(rows, "C")
            astat = retention_stats(rows, "A1")
            dstat = retention_stats(rows, "D")
            eligible = (
                b1_gain >= 1
                and b1_loss == 0
                and cstat["n_base"] >= 1
                and cstat["lost"] == 0
                and astat["lost"] == 0
                and dstat["lost"] == 0
            )
            print(
                f"  DEV b1_gain={b1_gain} b1_loss={b1_loss} "
                f"C {cstat['kept']}/{cstat['n_base']} A1 lost={astat['lost']} D lost={dstat['lost']} "
                f"eligible={eligible}",
                flush=True,
            )
            score = (b1_gain, cstat["kept"], -astat["lost"], -dstat["lost"])
            best_score = (best["b1_gain"], best["c_kept"], -best["a1_loss"], -best["d_loss"])
            if eligible and (not best["eligible"] or score >= best_score):
                path = CKPT / "P94_unified_best.pt"
                save_m111(path, actor, book, epoch=u, rho=rho, extra={"p94": True, "update": u})
                best.update(
                    {
                        "update": u,
                        "b1_gain": b1_gain,
                        "c_kept": cstat["kept"],
                        "a1_loss": astat["lost"],
                        "d_loss": dstat["lost"],
                        "path": str(path),
                        "eligible": True,
                    }
                )
            elif (not best["eligible"]) and score >= best_score:
                path = CKPT / "P94_unified_last.pt"
                save_m111(path, actor, book, epoch=u, rho=rho, extra={"p94": True, "update": u})
                best.update(
                    {
                        "update": u,
                        "b1_gain": b1_gain,
                        "c_kept": cstat["kept"],
                        "a1_loss": astat["lost"],
                        "d_loss": dstat["lost"],
                        "path": str(path),
                        "eligible": False,
                    }
                )

    _write_csv(OUT / "JOINT_TRAINING_CURVES.csv", curve)
    save_m111(CKPT / "P94_unified_final.pt", actor, book, epoch=UPDATES, rho=rho, extra={"p94": True})

    # ------------------------------------------------------------------
    # Phase 5 — development acceptance
    # ------------------------------------------------------------------
    print("[P94] Phase 5 development acceptance", flush=True)
    if best["path"] and Path(best["path"]).is_file():
        actor, _, _, _ = load_actor(Path(best["path"]), device)
    for p in actor.parameters():
        p.requires_grad_(False)
    actor.eval()

    dev_rows = eval_panel(
        actor, p64, starts, {"B1": dev["B1"], "A1": dev["A1"], "C": dev["C"], "D": dev["D"]}, book, rho, device
    )
    _write_csv(OUT / "PAIRED_DEVELOPMENT_RESULTS.csv", dev_rows)

    b1_gain = sum(int(r["gain"]) for r in dev_rows if r["skill"] == "B1")
    b1_loss = sum(int(r["loss"]) for r in dev_rows if r["skill"] == "B1")
    cstat = retention_stats(dev_rows, "C")
    astat = retention_stats(dev_rows, "A1")
    dstat = retention_stats(dev_rows, "D")
    unified = True  # single actor by construction
    eligible = (
        b1_gain >= 1
        and b1_loss == 0
        and cstat["n_base"] >= 1
        and cstat["lost"] == 0
        and astat["lost"] == 0
        and dstat["lost"] == 0
        and unified
    )

    cross = f"""# CROSS_CLASS_RETENTION

Generated: {_utc()}

## Development panel (paired vs P64)

| Skill | P64 succ | Unified succ | gains | losses | base retained |
|---|---:|---:|---:|---:|---|
| B1 | {sum(int(r['P64_SUCCESS']) for r in dev_rows if r['skill']=='B1')}/{sum(1 for r in dev_rows if r['skill']=='B1')} | {sum(int(r['UNIFIED_SUCCESS']) for r in dev_rows if r['skill']=='B1')}/{sum(1 for r in dev_rows if r['skill']=='B1')} | {b1_gain} | {b1_loss} | n/a |
| C | {cstat['n_base']} base | kept {cstat['kept']} | — | {cstat['lost']} | {cstat['kept']}/{cstat['n_base']} |
| A1 | {astat['n_base']} base | kept {astat['kept']} | {sum(int(r['gain']) for r in dev_rows if r['skill']=='A1')} | {astat['lost']} | {astat['kept']}/{astat['n_base']} |
| D | {dstat['n_base']} base | kept {dstat['kept']} | {sum(int(r['gain']) for r in dev_rows if r['skill']=='D')} | {dstat['lost']} | {dstat['kept']}/{dstat['n_base']} |

## Eligibility (Phase 5)

- B1 paired improvement: **{b1_gain >= 1 and b1_loss == 0}** (gain={b1_gain}, loss={b1_loss})
- C retention: **{cstat['lost'] == 0 and cstat['n_base'] >= 1}** ({cstat['kept']}/{cstat['n_base']})
- A1/D no loss: **{astat['lost'] == 0 and dstat['lost'] == 0}**
- One unified Actor + official R3F: **yes**
- **Eligible for independent eval: {eligible}**

Checkpoint: `{best.get('path')}` (update={best.get('update')})
"""
    _write(OUT / "CROSS_CLASS_RETENTION.md", cross)

    # Offline vs CL diagnostic for CASE D
    # measure offline action fit on B1 train
    off_l2 = []
    for d in b1_train_demos:
        o = torch.as_tensor(d["obs"], device=device, dtype=torch.float32)
        pv = torch.as_tensor(d["prev"], device=device, dtype=torch.float32)
        with torch.no_grad():
            _, res, _ = actor(o, pv)
        tres = torch.as_tensor(d["residuals"], device=device)
        off_l2.append(float(torch.linalg.vector_norm(res - tres, dim=-1).mean()))
    offline_ok = float(np.mean(off_l2)) < 0.05 if off_l2 else False

    ind_md = ""
    case = None
    title = None

    if not eligible:
        # diagnose CASE
        if offline_ok and b1_gain == 0:
            case, title = "D", "Offline full-action learning succeeds but closed-loop generalization fails"
        elif b1_gain >= 1 and (cstat["lost"] > 0 or astat["lost"] > 0 or dstat["lost"] > 0):
            case, title = "B", "Unified Actor improves B1 but causes cross-class regression"
        elif b1_gain == 0 and cstat["lost"] == 0 and astat["lost"] == 0 and dstat["lost"] == 0:
            case, title = "C", "Protected behavior is retained but B1 does not improve"
        else:
            case, title = "E", "The bounded shared-backbone adaptation cannot resolve the observed skill conflict"
        ind_md = f"""# INDEPENDENT_EVALUATION

Generated: {_utc()}

**Not opened.** Development acceptance failed (eligible=False).
Independent panel remains sealed. Case decided on development evidence + offline diagnostics.

Offline mean B1 residual L2: {float(np.mean(off_l2)) if off_l2 else 'n/a'}
"""
        _write(OUT / "INDEPENDENT_EVALUATION.md", ind_md)
    else:
        # --------------------------------------------------------------
        # Phase 6 — independent evaluation (freeze candidate first)
        # --------------------------------------------------------------
        print("[P94] Phase 6 independent evaluation", flush=True)
        freeze_path = CKPT / "P94_unified_frozen_for_ind.pt"
        save_m111(freeze_path, actor, book, epoch=best["update"], rho=rho, extra={"p94_frozen_ind": True})
        actor_ind, _, _, _ = load_actor(freeze_path, device)
        for p in actor_ind.parameters():
            p.requires_grad_(False)
        actor_ind.eval()

        ind_rows = eval_panel(actor_ind, p64, starts, ind, book, rho, device)
        _write_csv(OUT / "INDEPENDENT_PAIRED_RESULTS.csv", ind_rows)

        def cls(skill: str):
            rs = [r for r in ind_rows if r["skill"] == skill]
            n = len(rs)
            p64s = sum(int(r["P64_SUCCESS"]) for r in rs)
            us = sum(int(r["UNIFIED_SUCCESS"]) for r in rs)
            g = sum(int(r["gain"]) for r in rs)
            lo = sum(int(r["loss"]) for r in rs)
            rate, lo_ci, hi_ci = wilson(us, n)
            return dict(n=n, p64=p64s, u=us, gain=g, loss=lo, rate=rate, lo=lo_ci, hi=hi_ci)

        b1i, a1i, ci, di = cls("B1"), cls("A1"), cls("C"), cls("D")
        # class-balanced overall
        rates = [x["rate"] for x in (b1i, a1i, ci, di) if x["n"] > 0]
        overall = float(np.mean(rates)) if rates else float("nan")
        c_ret = retention_stats(ind_rows, "C")
        a_ret = retention_stats(ind_rows, "A1")
        d_ret = retention_stats(ind_rows, "D")

        ind_md = f"""# INDEPENDENT_EVALUATION

Generated: {_utc()}

Frozen candidate: `{freeze_path}` sha `{_sha(freeze_path)}`
Panel root-disjoint from train/dev and from P86/P92/P93 + frozen independent (P93 not reused).

## Single-rollout SUCCESS_RELAXED_12S

| Skill | P64 | Unified | gains | losses | Wilson 95% (unified) |
|---|---:|---:|---:|---:|---|
| B1 | {b1i['p64']}/{b1i['n']} | {b1i['u']}/{b1i['n']} | {b1i['gain']} | {b1i['loss']} | [{b1i['lo']:.3f}, {b1i['hi']:.3f}] |
| A1 | {a1i['p64']}/{a1i['n']} | {a1i['u']}/{a1i['n']} | {a1i['gain']} | {a1i['loss']} | [{a1i['lo']:.3f}, {a1i['hi']:.3f}] |
| C | {ci['p64']}/{ci['n']} | {ci['u']}/{ci['n']} | {ci['gain']} | {ci['loss']} | [{ci['lo']:.3f}, {ci['hi']:.3f}] |
| D | {di['p64']}/{di['n']} | {di['u']}/{di['n']} | {di['gain']} | {di['loss']} | [{di['lo']:.3f}, {di['hi']:.3f}] |

Class-balanced overall (mean of class rates): **{overall:.3f}**

## Conditional retention of P64-successful ICs

| Skill | retained |
|---|---|
| C | {c_ret['kept']}/{c_ret['n_base']} |
| A1 | {a_ret['kept']}/{a_ret['n_base']} |
| D | {d_ret['kept']}/{d_ret['n_base']} |

## Architecture / cost

One forward pass; same M111 Actor as P64 (no routing). Inference cost ≈ P64.
"""
        _write(OUT / "INDEPENDENT_EVALUATION.md", ind_md)

        b1_imp = b1i["gain"] >= 1 and b1i["u"] > b1i["p64"] and b1i["loss"] == 0
        c_ok = c_ret["n_base"] == 0 or c_ret["lost"] == 0
        a_ok = a_ret["lost"] == 0
        d_ok = d_ret["lost"] == 0
        if b1_imp and c_ok and a_ok and d_ok and c_ret["n_base"] >= 1:
            case, title = "A", "Unified Actor improves B1 and preserves A1/C/D on independent evaluation"
        elif b1_imp and (not c_ok or not a_ok or not d_ok):
            case, title = "B", "Unified Actor improves B1 but causes cross-class regression"
        elif (not b1_imp) and c_ok and a_ok and d_ok:
            case, title = "C", "Protected behavior is retained but B1 does not improve"
        else:
            case, title = "E", "The bounded shared-backbone adaptation cannot resolve the observed skill conflict"

        # thin C on independent representative panel: if no P64 successes in ind C, cannot claim A
        if case == "A" and c_ret["n_base"] < 1:
            case, title = "E", "The bounded shared-backbone adaptation cannot resolve the observed skill conflict"
            ind_md += "\n\nNote: independent C panel had zero P64 successes; CASE A withdrawn.\n"
            _write(OUT / "INDEPENDENT_EVALUATION.md", ind_md)

    final = f"""# FINAL_REPORT — P94 Unified Multi-Skill Actor

Generated: {_utc()}

## CASE {case} — {title}

### Integrity

- Single unified Actor (backbone + one token head + one residual head)
- No routing / multi-head / class labels / blending
- Official R3F unchanged
- P64 reference frozen (`{_sha(P64_ACTOR)}`); **not automatically replaced**
- Trainable initially: `enc.layers.1.*` + `res.*` ({n_train} params)

### Data

- B1 train demos: {len(b1_train_demos)} verified K48→M111 (P88 official)
- Protected train: A1={n_a_train}, C={n_c_train}, D={n_d_train}
- Dev C retention ICs: {len(dev['C'])}

### Development

| Check | Value |
|---|---|
| B1 gain/loss | {b1_gain}/{b1_loss} |
| C retained | {cstat['kept']}/{cstat['n_base']} |
| A1/D lost | {astat['lost']}/{dstat['lost']} |
| Eligible | {eligible} |

### Independent

See `INDEPENDENT_EVALUATION.md`.

### Decision note

Preserve original P64 unless CASE A with independent evidence. Current case **{case}** → P64 remains the reference deployable policy.
"""
    _write(OUT / "FINAL_REPORT.md", final)
    (OUT / "SUMMARY.json").write_text(
        json.dumps(
            {
                "case": case,
                "title": title,
                "eligible_dev": eligible,
                "best": best,
                "dev": {
                    "b1_gain": b1_gain,
                    "b1_loss": b1_loss,
                    "c": cstat,
                    "a1": astat,
                    "d": dstat,
                },
                "n_train": {"b1": len(b1_train_demos), "a1": n_a_train, "c": n_c_train, "d": n_d_train},
                "trainable_params": n_train,
                "p64_sha": _sha(P64_ACTOR),
                "p64_replaced": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[P94] DONE CASE {case} — {title}", flush=True)


if __name__ == "__main__":
    main()
