"""P98 — Leakage-Free Unified B1+C Skill Consolidation.

Single Transformer + one token head + one residual head.
Init from immutable original P64 only (never P94).
Canonical physical-root sealing. No Actor auto-replace of P64.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import pickle
import re
import sys
import time
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
from experiments.precision.p51_four_class_mixed_ppo.env_mixed import MixedStart  # noqa: E402
from experiments.precision.p51_four_class_mixed_ppo.ic_pools import root_id as rid0  # noqa: E402
from experiments.precision.p53_independent_grpo64.policy_sample import mask_handoff_logits  # noqa: E402
from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights  # noqa: E402
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset  # noqa: E402
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv  # noqa: E402
from experiments.precision.p61_reproducible_joint_eval.seeding import seed_all  # noqa: E402
from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor  # noqa: E402
from experiments.precision.p66_c_curriculum_ppo.expert_drive import (  # noqa: E402
    ensure_inner_capture_buffers,
    expert_actions,
    step_raw,
)
from experiments.precision.p67_r3f_gate_diagnosis.gate_predicates import evaluate_predicates  # noqa: E402
from experiments.precision.p68_precontact_geometry.geometry import relative_geometry  # noqa: E402
from experiments.precision.p70_geometry_to_r3f.obs_proxy import last_obs  # noqa: E402
from experiments.precision.p73_first_contact_bilateral.controller import WrapState, select_token, v2_spec  # noqa: E402
from experiments.precision.p73_first_contact_bilateral.rollout import _r3_terminal  # noqa: E402
from experiments.precision.p86_b1_gate_counterfactual.gate_ablation import make_monitor  # noqa: E402
from models.m111_checkpoint import decode_m111_action  # noqa: E402
from training.skill_aware_action_vocab import ResidualBudget  # noqa: E402

OUT = ROOT / "results/experiments/p98_unified_b1_c_consolidation"
CACHE = OUT / "cache"
CKPT = OUT / "checkpoints"
P87_TEACHER = ROOT / "results/experiments/p87_k48_actor_retention/teacher"
P88_CAND = ROOT / "results/experiments/p88_b1_short_sequence_learnability/PHASE0_CANDIDATES.csv"
P93_MAN = ROOT / "results/experiments/p93_frozen_router_independent_validation/EVALUATION_SPLIT_MANIFEST.csv"
FROZEN_MANIFEST = ROOT / "results/experiments/final_independent_success_rate/FROZEN_TEST_MANIFEST.json"
P94_SPLITS = ROOT / "results/experiments/p94_unified_multiskill_actor/SPLITS.json"
P97 = ROOT / "results/experiments/p97_c_expert_recoverability"
P97_CORR = P97 / "C_CORRECTION_MANIFEST.csv"
P97_CACHE = P97 / "cache"
P97_NEW = P97 / "new_ics/accepted"

SEED = 98
SPLIT_SEED = 98
UPDATES = 500
LR = 1e-3
LOG_EVERY = 25
CL_EVERY = 100
EARLY_K = 4
GRAD_CLIP = 5.0
LAMBDA_B = 1.0
LAMBDA_C = 1.25
LAMBDA_P = 1.0
LAMBDA_KL = 0.5
LAMBDA_REG = 1e-3
EARLY_W = 3.0

N_DEV_B1 = 3
N_DEV_A1 = 6
N_DEV_D = 6
N_DEV_C = 2
N_IND_B1 = 8
N_IND_A1 = 6
N_IND_C = 6
N_IND_D = 6
N_TRAIN_A1 = 8
N_TRAIN_D = 8
N_PRES_B1 = 4  # P64-successful B1 preservation (distinct from expert correction)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(path: Path, n: int = 16) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:n]


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_csv(path: Path, rows: list[dict], *, empty_headers: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        hdrs = empty_headers or []
        if hdrs:
            with path.open("w", encoding="utf-8", newline="") as f:
                csv.DictWriter(f, fieldnames=hdrs).writeheader()
        else:
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


def phys_root(tid: str) -> str:
    """Canonical physical root: strip plan_*/A|B|C|D__ prefixes and trailing hex suffixes."""
    t = re.sub(r"^(plan_[abc]__|A__|B__|C__|D__)", "", str(tid), flags=re.I)
    t = re.sub(r"__([0-9a-f]{8,})$", "", t)
    # also strip rid0-style plan_b leftover then re-apply
    t2 = rid0(tid)
    t2 = re.sub(r"^(plan_[abc]__|A__|B__|C__|D__)", "", t2, flags=re.I)
    t2 = re.sub(r"__([0-9a-f]{8,})$", "", t2)
    # prefer shorter/more canonical of the two when one still has prefix junk
    return t2 if len(t2) <= len(t) else t


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
        # also seal P94 C train/dev demos that were contamination
        for block in (sp.get("dev") or {}).values():
            if isinstance(block, list):
                for t in block:
                    if phys_root(t).startswith("gbi_") or "gbi_" in t:
                        # only auto-seal C-like from P94.dev if also frozen/P93 — handled below via phys
                        pass
    return {phys_root(t) for t in tids}


def is_banned(tid: str, ban: set[str]) -> bool:
    """Exact physical-root equality only (no substring). Avoids gbi_058 vs gbi_058_<hash> collisions."""
    return phys_root(tid) in ban


def seal_sources(tid: str) -> str:
    pr = phys_root(tid)
    bits: list[str] = []
    man = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
    for e in man["episodes"]:
        if phys_root(e["traj_id"]) == pr:
            bits.append(f"FROZEN:{e['traj_id']}")
    if P93_MAN.is_file():
        for r in csv.DictReader(P93_MAN.open(encoding="utf-8")):
            if phys_root(r["traj_id"]) == pr and r["panel"] in (
                "B1",
                "C1",
                "C2",
                "A1",
                "D",
                "DIAG_EXPOSED_B1",
            ):
                bits.append(f"P93_{r['panel']}:{r['traj_id']}")
    if P94_SPLITS.is_file():
        sp = json.loads(P94_SPLITS.read_text(encoding="utf-8"))
        for k, tids in (sp.get("ind") or {}).items():
            if isinstance(tids, list):
                for t in tids:
                    if phys_root(t) == pr:
                        bits.append(f"P94_ind_{k}:{t}")
        for t in (sp.get("dev") or {}).get("C") or []:
            if phys_root(t) == pr:
                bits.append(f"P94_dev_C:{t}")
    return "|".join(bits)


def lookup_starts() -> dict[str, MixedStart]:
    return {s.traj_id: s for s in build_p54_dataset(seed=5478)["schedule"]}


def verified_b1() -> list[str]:
    rows = list(csv.DictReader(P88_CAND.open(encoding="utf-8")))
    return sorted(r["traj_id"] for r in rows if int(r.get("valid_official") or 0) == 1)


def load_teacher(tid: str) -> dict | None:
    p = P87_TEACHER / f"{tid}.npz"
    if not p.is_file():
        return None
    z = np.load(p, allow_pickle=True)
    return {
        "traj_id": tid,
        "phys_root": phys_root(tid),
        "skill": "B1",
        "category": "b1_correction_expert",
        "tokens": np.asarray(z["tokens"], np.int64),
        "residuals": np.asarray(z["residuals"], np.float32),
        "actions": np.asarray(z["actions"], np.float32),
        "obs": np.asarray(z["obs"], np.float32),
        "prev": np.asarray(z["prev"], np.float32),
        "verified": 1,
    }


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
    ever12 = False
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
        if bil and pad4 and wk and e <= 12 + 1e-9 and th <= 8 + 1e-9:
            ever12 = True
    term = _r3_terminal(env, info)
    out = {
        "SUCCESS_RELAXED_12S": int(term.get("SUCCESS_RELAXED") or info.get("SUCCESS_RELAXED_12S") or 0),
        "handoff": int(info.get("handoff") or 0),
        "r3_ready": int(term.get("r3_ready") or 0),
        "reach12": int(ever12),
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


def expert_record(start: MixedStart, book, rho, *, seed: int = SEED) -> dict:
    acts = expert_actions(start.ep or {})
    demo_steps = max(int(MAX_EP_STEPS), int(len(acts)) + 8)
    seed_all(seed, deterministic_torch=True)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=demo_steps, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )
    env.reset(start)
    env.mon = make_monitor("G0", t_max=int(env.t_max))
    ensure_inner_capture_buffers(env)
    done = False
    info: dict = {}
    actions: list[np.ndarray] = []
    obs_l: list[np.ndarray] = []
    prev_l: list[np.ndarray] = []
    for a in acts:
        if done or env.t >= env.max_steps:
            break
        obs_l.append(env.obs_seq[-1].astype(np.float32))
        prev_l.append(env.prev_seq[-1].astype(np.float32))
        actions.append(np.asarray(a, np.float32).reshape(7))
        _r, done, info = step_raw(env, a)
        ensure_inner_capture_buffers(env)
    return {
        "SUCCESS_RELAXED_12S": int(info.get("SUCCESS_RELAXED_12S") or 0),
        "actions": np.stack(actions) if actions else np.zeros((0, 7), np.float32),
        "obs": np.stack(obs_l) if obs_l else np.zeros((0, 39), np.float32),
        "prev": np.stack(prev_l) if prev_l else np.zeros((0, 7), np.float32),
    }


def early_w(n: int, device, w0: float) -> torch.Tensor:
    w = torch.ones(n, device=device, dtype=torch.float32)
    w[: min(EARLY_K, n)] = w0
    return w


def correction_loss(actor, demo, device, early_weight: float) -> tuple[torch.Tensor, dict]:
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
    k = min(EARLY_K, len(toks))
    loss_early = ((dec[:k] - tacts[:k]) ** 2).mean()
    loss = loss_res + 0.5 * loss_tok + 1.0 * loss_act + 0.5 * loss_early
    return loss, {
        "res": float(loss_res.detach()),
        "tok": float(loss_tok.detach()),
        "act": float(loss_act.detach()),
        "early": float(loss_early.detach()),
    }


def preservation_loss(actor, demo, ref, device, early_weight: float) -> tuple[torch.Tensor, dict]:
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
    loss_kl = F.kl_div(log_p, q, reduction="batchmean")
    loss = loss_res + loss_act + 0.5 * loss_kl
    return loss, {
        "skill": demo["skill"],
        "pres_res": float(loss_res.detach()),
        "pres_act": float(loss_act.detach()),
        "pres_kl": float(loss_kl.detach()),
    }


def param_l2(actor, ref) -> torch.Tensor:
    s = None
    for (n, p), (_, q) in zip(actor.named_parameters(), ref.named_parameters()):
        if not p.requires_grad:
            continue
        d = (p - q.detach()).pow(2).sum()
        s = d if s is None else s + d
    return s if s is not None else torch.tensor(0.0, device=next(actor.parameters()).device)


def freeze_residual_only(actor) -> list[str]:
    names = []
    for name, p in actor.named_parameters():
        allow = name.startswith("res.")
        p.requires_grad_(allow)
        if allow:
            names.append(name)
    return names


def freeze_last_block_and_res(actor) -> list[str]:
    names = []
    for name, p in actor.named_parameters():
        allow = name.startswith("enc.layers.1.") or name.startswith("res.")
        p.requires_grad_(allow)
        if allow:
            names.append(name)
    return names


def load_p97_c_demos(book, rho, device, ban: set[str]) -> list[dict]:
    """Load physically verified C correction demos; rebuild obs if missing."""
    rows = list(csv.DictReader(P97_CORR.open(encoding="utf-8")))
    demos: list[dict] = []
    seen: set[str] = set()
    starts_extra: dict[str, MixedStart] = {}

    for r in rows:
        if r.get("set_kind") != "correction":
            continue
        tid = r["traj_id"]
        pr = r.get("phys_root") or phys_root(tid)
        if pr in seen or is_banned(tid, ban) or is_banned(pr, ban):
            continue
        # seal gbi_351 explicitly
        if "351_bc2d6a3b90" in pr or "351_bc2d6a3b90" in tid:
            continue
        zp = P97_CACHE / r["cache_npz"]
        tokens = residuals = actions = obs = prev = None
        if zp.is_file():
            z = np.load(zp)
            tokens = np.asarray(z["tokens"], np.int64)
            residuals = np.asarray(z["residuals"], np.float32)
            actions = np.asarray(z["actions"], np.float32)
            if "obs" in z.files:
                obs = np.asarray(z["obs"], np.float32)
                prev = np.asarray(z["prev"], np.float32)
        if obs is None or prev is None or len(obs) == 0:
            # rebuild from accepted new-IC pkl or skip
            pkl = P97_NEW / f"{tid}.pkl"
            if not pkl.is_file():
                # try stem match
                cands = list(P97_NEW.glob(f"*{tid.split('__')[-1][:10]}*.pkl")) if "__" in tid else []
                pkl = cands[0] if cands else pkl
            if not pkl.is_file():
                print(f"  skip C demo missing obs: {tid}", flush=True)
                continue
            blob = pickle.loads(pkl.read_bytes())
            ep = blob["ep"]
            start = MixedStart(skill="C", traj_id=tid, family="C", kind="ep", ep=ep)
            starts_extra[tid] = start
            # Prefer stored m111 + re-record obs via continuous actions
            m111 = blob.get("m111") or {}
            if m111.get("actions") is not None:
                actions = np.asarray(m111["actions"], np.float32)
                tokens = np.asarray(m111["tokens"], np.int64)
                residuals = np.asarray(m111["residuals"], np.float32)
            rec = expert_record(start, book, rho)
            if int(rec.get("SUCCESS_RELAXED_12S") or 0) != 1:
                print(f"  skip C rebuild fail R3F: {tid}", flush=True)
                continue
            obs = rec["obs"]
            prev = rec["prev"]
            actions = rec["actions"]
            # re-quantize if token length mismatch
            if tokens is None or len(tokens) != len(actions):
                from training.skill_aware_action_vocab import DEFAULT_WEIGHTS, quantize_trajectory

                q = quantize_trajectory(book, actions, rho, w=DEFAULT_WEIGHTS)
                tokens = np.asarray(q["tokens"], np.int64)
                residuals = np.asarray(q["residuals"], np.float32)
            # truncate to min length
            n = min(len(tokens), len(obs), len(actions))
            tokens, residuals, actions, obs, prev = tokens[:n], residuals[:n], actions[:n], obs[:n], prev[:n]
            np.savez_compressed(
                CACHE / f"c_corr_{pr}.npz",
                tokens=tokens,
                residuals=residuals,
                actions=actions,
                obs=obs,
                prev=prev,
            )
        else:
            # copy into P98 cache
            np.savez_compressed(
                CACHE / f"c_corr_{pr}.npz",
                tokens=tokens,
                residuals=residuals,
                actions=actions,
                obs=obs,
                prev=prev,
            )
        demos.append(
            {
                "traj_id": tid,
                "phys_root": pr,
                "skill": "C",
                "category": "c_correction_verified",
                "tokens": tokens,
                "residuals": residuals,
                "actions": actions,
                "obs": obs,
                "prev": prev,
                "verified": 1,
                "source": r.get("source"),
            }
        )
        seen.add(pr)
        print(f"  C demo ok {pr} n={len(tokens)}", flush=True)
    return demos, starts_extra


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p64_sha = _sha(P64_ACTOR)
    print(f"[P98] device={device} P64={p64_sha} (immutable init; P94 unused)", flush=True)

    ban = eval_banned_phys()
    # force-seal known contaminated
    ban.add(phys_root("plan_c__gbi_351_bc2d6a3b90"))
    ban.add(phys_root("C__gbi_351_bc2d6a3b90__ba998ceb7d01bdf0"))
    ban.add(phys_root("plan_c__gbi_022_7c876a650b"))
    ban.add("gbi_351_bc2d6a3b90")
    ban.add("gbi_022_7c876a650b")

    starts = lookup_starts()
    p64, book, rho, _ = load_actor(P64_ACTOR, device)
    for p in p64.parameters():
        p.requires_grad_(False)
    p64.eval()
    ref = copy.deepcopy(p64).to(device)
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()

    # ------------------------------------------------------------------
    # Phase 0 — provenance audit
    # ------------------------------------------------------------------
    print("[P98] Phase 0 provenance audit", flush=True)
    audit: list[dict] = []
    # P94 C demo contamination check
    p94_c_dev = []
    if P94_SPLITS.is_file():
        sp = json.loads(P94_SPLITS.read_text(encoding="utf-8"))
        p94_c_dev = list((sp.get("dev") or {}).get("C") or [])
    for tid in p94_c_dev:
        pr = phys_root(tid)
        audit.append(
            {
                "traj_id": tid,
                "phys_root": pr,
                "role": "P94_dev_C_demo",
                "sealed": int(is_banned(tid, ban)),
                "seal_sources": seal_sources(tid),
                "eligible_train": 0,
                "notes": "P94 C retention IC; sealed if physically equivalent to frozen/P93/P94-ind",
            }
        )
    # all schedule C + P97 corr + verified B1
    for tid, s in starts.items():
        pr = phys_root(tid)
        sealed = int(is_banned(tid, ban))
        audit.append(
            {
                "traj_id": tid,
                "phys_root": pr,
                "role": f"schedule_{s.skill}",
                "skill": s.skill,
                "sealed": sealed,
                "seal_sources": seal_sources(tid) if sealed else "",
                "eligible_train": int(not sealed),
                "notes": "",
            }
        )
    if P97_CORR.is_file():
        for r in csv.DictReader(P97_CORR.open(encoding="utf-8")):
            if r.get("set_kind") != "correction":
                continue
            tid = r["traj_id"]
            pr = r.get("phys_root") or phys_root(tid)
            sealed = int(is_banned(tid, ban) or is_banned(pr, ban))
            audit.append(
                {
                    "traj_id": tid,
                    "phys_root": pr,
                    "role": "P97_c_correction",
                    "skill": "C",
                    "sealed": sealed,
                    "seal_sources": seal_sources(tid),
                    "eligible_train": int(not sealed),
                    "notes": r.get("source"),
                }
            )
    for tid in verified_b1():
        audit.append(
            {
                "traj_id": tid,
                "phys_root": phys_root(tid),
                "role": "P88_verified_B1",
                "skill": "B1",
                "sealed": int(is_banned(tid, ban)),
                "seal_sources": seal_sources(tid),
                "eligible_train": int(not is_banned(tid, ban)),
                "notes": "expert B1 correction candidate",
            }
        )
    _write_csv(OUT / "PHYSICAL_ROOT_PROVENANCE_AUDIT.csv", audit)

    # P64 lineage note
    _write(
        OUT / "P64_LINEAGE.md",
        f"# P64 lineage\n\nsha `{p64_sha}`\npath `{P64_ACTOR}`\n\n"
        f"P98 initializes **only** from this checkpoint. P94 unified checkpoints are unused.\n"
        f"P64 was trained under P64 multiround PPO; independent eval uses FROZEN_TEST_MANIFEST "
        f"which is sealed from training via physical-root ban.\n",
    )

    # ------------------------------------------------------------------
    # Phase 1 — clean sets
    # ------------------------------------------------------------------
    print("[P98] Phase 1 clean datasets", flush=True)
    c_demos, c_starts_extra = load_p97_c_demos(book, rho, device, ban)
    # merge starts
    for tid, st in c_starts_extra.items():
        starts[tid] = st

    rng = np.random.default_rng(SPLIT_SEED)
    c_roots = [d["phys_root"] for d in c_demos]
    order = list(range(len(c_demos)))
    rng.shuffle(order)
    # Prefer: hold N_DEV_C for development; rest train
    c_dev_idx = order[:N_DEV_C]
    c_train_idx = order[N_DEV_C:]
    if len(c_train_idx) < 4 and len(c_demos) >= N_DEV_C + 1:
        # ensure at least some train
        c_dev_idx = order[:1]
        c_train_idx = order[1:]
    c_train = [c_demos[i] for i in c_train_idx]
    c_dev_demos = [c_demos[i] for i in c_dev_idx]
    c_dev_ids = [d["traj_id"] for d in c_dev_demos]
    print(
        f"  C verified={len(c_demos)} train_roots={len(c_train)} dev_roots={len(c_dev_demos)} "
        f"(no C preservation; held-out from the verified set)",
        flush=True,
    )

    # B1 verified expert correction
    ver = [t for t in verified_b1() if not is_banned(t, ban) and (P87_TEACHER / f"{t}.npz").is_file()]
    rng.shuffle(ver)
    b1_dev_ids = ver[:N_DEV_B1]
    b1_train_ids = ver[N_DEV_B1:]
    b1_corr = []
    for tid in b1_train_ids:
        d = load_teacher(tid)
        if d:
            b1_corr.append(d)

    # Eligible pools for A1/D/B1 preservation + ind
    pools = {"A1": [], "B1": [], "C": [], "D": []}
    for tid, s in starts.items():
        if is_banned(tid, ban):
            continue
        if s.skill in pools:
            pools[s.skill].append(tid)
    for sk in pools:
        pools[sk] = sorted(set(pools[sk]))
        rng.shuffle(pools[sk])

    # Independent first (root-disjoint from train/dev C and B1 verified)
    used_phys = {phys_root(t) for t in b1_dev_ids + b1_train_ids + c_dev_ids}
    used_phys |= {d["phys_root"] for d in c_train}

    def take_ind(skill: str, n: int) -> list[str]:
        out = []
        for tid in pools[skill]:
            pr = phys_root(tid)
            if pr in used_phys:
                continue
            # exclude C verified train/dev phys
            if skill == "C" and any(pr == d["phys_root"] or pr in d["phys_root"] or d["phys_root"] in pr for d in c_demos):
                continue
            out.append(tid)
            used_phys.add(pr)
            if len(out) >= n:
                break
        return out

    ind = {
        "B1": take_ind("B1", N_IND_B1),
        "A1": take_ind("A1", N_IND_A1),
        "C": take_ind("C", N_IND_C),
        "D": take_ind("D", N_IND_D),
    }

    # Dev A1/D from remaining
    def take_dev(skill: str, n: int, exclude: set[str]) -> list[str]:
        out = []
        for tid in pools[skill]:
            if tid in exclude or phys_root(tid) in used_phys:
                continue
            out.append(tid)
            used_phys.add(phys_root(tid))
            if len(out) >= n:
                break
        return out

    a1_dev = take_dev("A1", N_DEV_A1, set(ind["A1"]))
    d_dev = take_dev("D", N_DEV_D, set(ind["D"]))

    # Screen A1/D/B1 P64-success preservation for train
    def screen_pres(skill: str, cand: list[str], target: int) -> list[dict]:
        out = []
        for tid in cand:
            if len(out) >= target:
                break
            if tid not in starts or is_banned(tid, ban):
                continue
            print(f"  screen {skill} pres {tid}", flush=True)
            roll = closed_loop(p64, starts[tid], book, rho, device, record=True)
            if int(roll["SUCCESS_RELAXED_12S"]) != 1:
                continue
            demo = {
                "traj_id": tid,
                "phys_root": phys_root(tid),
                "skill": skill,
                "category": "p64_preservation",
                "tokens": roll["tokens"],
                "residuals": roll["residuals"],
                "actions": roll["actions"],
                "obs": roll["obs"],
                "prev": roll["prev"],
                "verified": 1,
            }
            np.savez_compressed(
                CACHE / f"pres_{skill}_{phys_root(tid)}.npz",
                tokens=demo["tokens"],
                residuals=demo["residuals"],
                actions=demo["actions"],
                obs=demo["obs"],
                prev=demo["prev"],
            )
            out.append(demo)
        return out

    rem_a1 = [t for t in pools["A1"] if t not in set(ind["A1"]) | set(a1_dev)]
    rem_d = [t for t in pools["D"] if t not in set(ind["D"]) | set(d_dev)]
    rem_b1 = [t for t in pools["B1"] if phys_root(t) not in {phys_root(x) for x in ver} and t not in set(ind["B1"])]
    a1_pres = screen_pres("A1", rem_a1, N_TRAIN_A1)
    d_pres = screen_pres("D", rem_d, N_TRAIN_D)
    b1_pres = screen_pres("B1", rem_b1, N_PRES_B1)

    # Manifests
    clean_rows = []
    corr_rows = []
    pres_rows = []
    for d in b1_corr:
        clean_rows.append(
            {
                "split": "train",
                "skill": "B1",
                "role": "correction",
                "traj_id": d["traj_id"],
                "phys_root": d["phys_root"],
                "n": len(d["tokens"]),
                "category": d["category"],
            }
        )
        corr_rows.append(
            {
                "skill": "B1",
                "traj_id": d["traj_id"],
                "phys_root": d["phys_root"],
                "n": len(d["tokens"]),
                "early_priority": 1,
                "source": "P87_teacher",
            }
        )
    for d in c_train:
        clean_rows.append(
            {
                "split": "train",
                "skill": "C",
                "role": "correction",
                "traj_id": d["traj_id"],
                "phys_root": d["phys_root"],
                "n": len(d["tokens"]),
                "category": d["category"],
            }
        )
        corr_rows.append(
            {
                "skill": "C",
                "traj_id": d["traj_id"],
                "phys_root": d["phys_root"],
                "n": len(d["tokens"]),
                "early_priority": 1,
                "source": d.get("source"),
            }
        )
    for d in a1_pres + d_pres + b1_pres:
        clean_rows.append(
            {
                "split": "train",
                "skill": d["skill"],
                "role": "preservation",
                "traj_id": d["traj_id"],
                "phys_root": d["phys_root"],
                "n": len(d["tokens"]),
                "category": d["category"],
            }
        )
        pres_rows.append(
            {
                "skill": d["skill"],
                "traj_id": d["traj_id"],
                "phys_root": d["phys_root"],
                "n": len(d["tokens"]),
                "source": "P64_success_replay",
            }
        )
    for tid in b1_dev_ids:
        clean_rows.append(
            {"split": "dev", "skill": "B1", "role": "eval", "traj_id": tid, "phys_root": phys_root(tid), "n": "", "category": "verified_holdout"}
        )
    for d in c_dev_demos:
        clean_rows.append(
            {
                "split": "dev",
                "skill": "C",
                "role": "eval_expert_success_holdout",
                "traj_id": d["traj_id"],
                "phys_root": d["phys_root"],
                "n": len(d["tokens"]),
                "category": "c_verified_holdout_not_p64_success",
            }
        )
        starts[d["traj_id"]] = starts.get(d["traj_id"]) or c_starts_extra.get(d["traj_id"])
        # ensure start exists for closed-loop: rebuild from pkl if needed
        if d["traj_id"] not in starts:
            pkl = P97_NEW / f"{d['traj_id']}.pkl"
            if pkl.is_file():
                ep = pickle.loads(pkl.read_bytes())["ep"]
                starts[d["traj_id"]] = MixedStart(skill="C", traj_id=d["traj_id"], family="C", kind="ep", ep=ep)
        # For historical C demos use schedule start by matching phys
        if d["traj_id"] not in starts:
            for tid, s in list(starts.items()):
                if s.skill == "C" and phys_root(tid) == d["phys_root"].split("__")[0] or phys_root(tid) == d["phys_root"]:
                    # alias
                    starts[d["traj_id"]] = s
                    break
            # try matching without fingerprint suffix
            base = d["phys_root"].split("__")[0]
            for tid, s in list(starts.items()):
                if s.skill == "C" and phys_root(tid) == base:
                    starts[d["traj_id"]] = s
                    break

    for tid in a1_dev:
        clean_rows.append({"split": "dev", "skill": "A1", "role": "eval", "traj_id": tid, "phys_root": phys_root(tid), "n": "", "category": "reserved"})
    for tid in d_dev:
        clean_rows.append({"split": "dev", "skill": "D", "role": "eval", "traj_id": tid, "phys_root": phys_root(tid), "n": "", "category": "reserved"})
    for sk, tids in ind.items():
        for tid in tids:
            clean_rows.append(
                {"split": "ind", "skill": sk, "role": "eval", "traj_id": tid, "phys_root": phys_root(tid), "n": "", "category": "independent"}
            )

    _write_csv(OUT / "CLEAN_DATASET_MANIFEST.csv", clean_rows)
    _write_csv(OUT / "B1_C_CORRECTION_TARGETS.csv", corr_rows)
    _write_csv(OUT / "B1_A1_D_PRESERVATION_TARGETS.csv", pres_rows)
    _write(
        OUT / "SPLITS.json",
        json.dumps(
            {
                "dev": {"B1": b1_dev_ids, "A1": a1_dev, "C": c_dev_ids, "D": d_dev},
                "ind": ind,
                "c_train_phys": [d["phys_root"] for d in c_train],
                "c_dev_phys": [d["phys_root"] for d in c_dev_demos],
                "p64_sha": p64_sha,
                "init": "P64_only",
            },
            indent=2,
        ),
    )

    if not b1_corr or not c_train or not a1_pres or not d_pres:
        _write(
            OUT / "FINAL_REPORT.md",
            f"# FINAL_REPORT — P98\n\n## CASE F\n\nInsufficient clean demos "
            f"(B1_corr={len(b1_corr)}, C_train={len(c_train)}, A1={len(a1_pres)}, D={len(d_pres)}).\n",
        )
        print("[P98] CASE F insufficient demos", flush=True)
        return

    # Resolve C starts for closed-loop on train/dev C ICs
    for d in c_train + c_dev_demos:
        tid = d["traj_id"]
        if tid in starts:
            continue
        pkl = P97_NEW / f"{tid}.pkl"
        if pkl.is_file():
            ep = pickle.loads(pkl.read_bytes())["ep"]
            starts[tid] = MixedStart(skill="C", traj_id=tid, family="C", kind="ep", ep=ep)
            continue
        base = d["phys_root"].split("__")[0]
        hit = next((t for t, s in starts.items() if s.skill == "C" and phys_root(t) == base), None)
        if hit:
            starts[tid] = starts[hit]

    panel_dev = {"B1": b1_dev_ids, "A1": a1_dev, "C": [t for t in c_dev_ids if t in starts], "D": d_dev}
    print(
        f"[P98] demos B1_corr={len(b1_corr)} C_corr={len(c_train)} "
        f"pres A1={len(a1_pres)} D={len(d_pres)} B1={len(b1_pres)} "
        f"devC={panel_dev['C']}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Phase 3–4 training
    # ------------------------------------------------------------------
    def train_variant(label: str, mode: str) -> tuple[Any, list[dict], dict]:
        actor, _, _, _ = load_actor(P64_ACTOR, device)
        if mode == "residual":
            trained = freeze_residual_only(actor)
        else:
            trained = freeze_last_block_and_res(actor)
        # token head remains frozen in both
        for name, p in actor.named_parameters():
            if "tok" in name or "head_tok" in name or name.startswith("logits") or "token" in name.lower():
                # keep frozen unless already allowed; P98-B says token-head frozen initially
                if mode == "backbone":
                    if not (name.startswith("enc.layers.1.") or name.startswith("res.")):
                        p.requires_grad_(False)
        opt = torch.optim.Adam([p for p in actor.parameters() if p.requires_grad], lr=LR)
        rng_t = np.random.default_rng(SEED + (0 if mode == "residual" else 7))
        curve: list[dict] = []
        best = {"score": (-1e9,), "path": None, "update": 0, "rows": None}
        print(f"[{label}] mode={mode} trainable≈{len(trained)} updates={UPDATES}", flush=True)

        grad_cos_hist: list[float] = []

        for u in range(1, UPDATES + 1):
            actor.train()
            # balanced sample: one B1 corr, one C corr, one preservation (rotate skills)
            db = b1_corr[int(rng_t.integers(0, len(b1_corr)))]
            dc = c_train[int(rng_t.integers(0, len(c_train)))]
            pres_pool = a1_pres + d_pres + b1_pres
            # skill-balance: pick skill then demo
            sk = ["A1", "D", "B1"][u % 3]
            pool_sk = [d for d in pres_pool if d["skill"] == sk] or pres_pool
            dp = pool_sk[int(rng_t.integers(0, len(pool_sk)))]

            opt.zero_grad(set_to_none=True)
            lb, sb = correction_loss(actor, db, device, EARLY_W)
            lc, sc = correction_loss(actor, dc, device, EARLY_W)
            # gradient conflict probe (before summing): retain grads separately via autograd
            # efficient approx: compute cos between flattened grads of B and C terms
            gb = torch.autograd.grad(lb, [p for p in actor.parameters() if p.requires_grad], retain_graph=True, allow_unused=True)
            gc = torch.autograd.grad(lc, [p for p in actor.parameters() if p.requires_grad], retain_graph=True, allow_unused=True)
            flat_b = torch.cat([g.reshape(-1) for g in gb if g is not None]) if any(g is not None for g in gb) else None
            flat_c = torch.cat([g.reshape(-1) for g in gc if g is not None]) if any(g is not None for g in gc) else None
            cos_bc = float("nan")
            if flat_b is not None and flat_c is not None and flat_b.numel() and flat_c.numel():
                cos_bc = float(F.cosine_similarity(flat_b.unsqueeze(0), flat_c.unsqueeze(0)).item())
                grad_cos_hist.append(cos_bc)

            lp, sp = preservation_loss(actor, dp, ref, device, EARLY_W)
            # token stability KL already inside preservation; add explicit KL on B1 corr states vs ref
            obs_b = torch.as_tensor(db["obs"], device=device, dtype=torch.float32)
            prev_b = torch.as_tensor(db["prev"], device=device, dtype=torch.float32)
            with torch.no_grad():
                logits_r, _, _ = ref(obs_b, prev_b)
            logits_a, _, _ = actor(obs_b, prev_b)
            lkl = F.kl_div(
                F.log_softmax(mask_handoff_logits(logits_a.float()), dim=-1),
                F.softmax(mask_handoff_logits(logits_r.float()), dim=-1),
                reduction="batchmean",
            )
            lreg = param_l2(actor, ref)
            loss = LAMBDA_B * lb + LAMBDA_C * lc + LAMBDA_P * lp + LAMBDA_KL * lkl + LAMBDA_REG * lreg
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in actor.parameters() if p.requires_grad], GRAD_CLIP)
            opt.step()

            if u % LOG_EVERY == 0 or u == 1:
                row = {
                    "update": u,
                    "loss": float(loss.detach()),
                    "L_B": float(lb.detach()),
                    "L_C": float(lc.detach()),
                    "L_P": float(lp.detach()),
                    "L_KL": float(lkl.detach()),
                    "L_reg": float(lreg.detach()),
                    "grad_cos_B_C": cos_bc,
                    "B_res": sb["res"],
                    "C_res": sc["res"],
                    "P_skill": sp["skill"],
                }
                curve.append(row)
                print(
                    f"  [{label}] u={u} loss={row['loss']:.4f} B={row['L_B']:.3f} C={row['L_C']:.3f} "
                    f"P={row['L_P']:.3f} cosBC={cos_bc:.3f}",
                    flush=True,
                )

            if u % CL_EVERY == 0 or u == UPDATES:
                actor.eval()
                rows = eval_panel(actor, p64, starts, panel_dev, book, rho, device, label)
                b1_net = sum(int(r["gain"]) - int(r["loss"]) for r in rows if r["skill"] == "B1")
                c_succ = sum(int(r["CAND_SUCCESS"]) for r in rows if r["skill"] == "C")
                a1_loss = sum(int(r["loss"]) for r in rows if r["skill"] == "A1")
                d_loss = sum(int(r["loss"]) for r in rows if r["skill"] == "D")
                score = (b1_net, c_succ, -a1_loss, -d_loss)
                path = CKPT / f"{label}_u{u}.pt"
                torch.save({"actor": actor.state_dict(), "update": u, "p64_sha": p64_sha}, path)
                if score > best["score"]:
                    best = {"score": score, "path": str(path), "update": u, "rows": rows}
                    torch.save({"actor": actor.state_dict(), "update": u, "p64_sha": p64_sha}, CKPT / f"{label}_best.pt")
                print(f"  [{label}] cl u={u} b1_net={b1_net} c_succ={c_succ} a1_loss={a1_loss} d_loss={d_loss}", flush=True)

        # write grad audit snippet
        cos_arr = np.asarray(grad_cos_hist, np.float32) if grad_cos_hist else np.zeros(0)
        _write(
            CACHE / f"grad_cos_{label}.json",
            json.dumps(
                {
                    "n": int(len(cos_arr)),
                    "mean": float(cos_arr.mean()) if len(cos_arr) else None,
                    "frac_neg": float((cos_arr < 0).mean()) if len(cos_arr) else None,
                },
                indent=2,
            ),
        )
        # reload best
        if best["path"]:
            blob = torch.load(best["path"], map_location=device)
            actor.load_state_dict(blob["actor"])
        actor.eval()
        return actor, curve, best

    def eval_panel(actor, p64_ref, starts_map, panel, book_, rho_, device_, cand_name: str) -> list[dict]:
        rows = []
        for skill, tids in panel.items():
            for tid in tids:
                if tid not in starts_map:
                    continue
                r64 = closed_loop(p64_ref, starts_map[tid], book_, rho_, device_)
                ru = closed_loop(actor, starts_map[tid], book_, rho_, device_)
                # early residual compare at shared reset: one-step teacher-force not available; use full-ep proxies
                rows.append(
                    {
                        "candidate": cand_name,
                        "skill": skill,
                        "traj_id": tid,
                        "phys_root": phys_root(tid),
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
                        "P64_ep_len": r64["ep_len"],
                        "CAND_ep_len": ru["ep_len"],
                    }
                )
        return rows

    print("[P98] Phase 4 train P98-A residual-only", flush=True)
    actor_a, curve_a, best_a = train_variant("P98A", "residual")
    _write_csv(OUT / "P98_A_RESULTS.csv", curve_a)

    print("[P98] Phase 4 train P98-B backbone+res", flush=True)
    actor_b, curve_b, best_b = train_variant("P98B", "backbone")
    _write_csv(OUT / "P98_B_RESULTS.csv", curve_b)

    # Gradient audit md
    ga = json.loads((CACHE / "grad_cos_P98A.json").read_text(encoding="utf-8")) if (CACHE / "grad_cos_P98A.json").is_file() else {}
    gb = json.loads((CACHE / "grad_cos_P98B.json").read_text(encoding="utf-8")) if (CACHE / "grad_cos_P98B.json").is_file() else {}
    _write(
        OUT / "PER_SKILL_GRADIENT_AUDIT.md",
        f"""# PER_SKILL_GRADIENT_AUDIT — P98

Generated: {_utc()}

Cosine similarity between B1-correction and C-correction parameter gradients
(same shared trainable set; negative values indicate conflict but do **not** trigger algorithm changes).

| Candidate | n probes | mean cos | fraction cos&lt;0 |
|---|---:|---:|---:|
| P98-A | {ga.get('n')} | {ga.get('mean')} | {ga.get('frac_neg')} |
| P98-B | {gb.get('n')} | {gb.get('mean')} | {gb.get('frac_neg')} |

Loss weights: λ_B={LAMBDA_B}, λ_C={LAMBDA_C}, λ_P={LAMBDA_P}, λ_KL={LAMBDA_KL}, λ_reg={LAMBDA_REG}.
Sampling balances B1 corr : C corr : rotating A1/D/B1 preservation each update.
""",
    )

    # ------------------------------------------------------------------
    # Phase 5 — paired development
    # ------------------------------------------------------------------
    print("[P98] Phase 5 development eval", flush=True)
    rows_p64 = eval_panel(p64, p64, starts, panel_dev, book, rho, device, "P64")
    for r in rows_p64:
        r["gain"] = 0
        r["loss"] = 0
    rows_a = eval_panel(actor_a, p64, starts, panel_dev, book, rho, device, "P98A")
    rows_b = eval_panel(actor_b, p64, starts, panel_dev, book, rho, device, "P98B")
    paired = rows_a + rows_b
    _write_csv(OUT / "PAIRED_DEVELOPMENT_RESULTS.csv", paired)

    def summarize(rows: list[dict], name: str) -> dict:
        b1g = sum(int(r["gain"]) for r in rows if r["skill"] == "B1")
        b1l = sum(int(r["loss"]) for r in rows if r["skill"] == "B1")
        c_n = sum(1 for r in rows if r["skill"] == "C")
        c_succ = sum(int(r["CAND_SUCCESS"]) for r in rows if r["skill"] == "C")
        c_p64 = sum(int(r["P64_SUCCESS"]) for r in rows if r["skill"] == "C")
        a1_base = [r for r in rows if r["skill"] == "A1" and int(r["P64_SUCCESS"]) == 1]
        d_base = [r for r in rows if r["skill"] == "D" and int(r["P64_SUCCESS"]) == 1]
        a1_lost = sum(1 for r in a1_base if int(r["CAND_SUCCESS"]) == 0)
        d_lost = sum(1 for r in d_base if int(r["CAND_SUCCESS"]) == 0)
        # eligibility: B1 net>0, C shows some success on expert-holdout, no heavy A1/D loss
        eligible = (
            (b1g - b1l) >= 1
            and b1l == 0
            and c_succ >= 1
            and a1_lost == 0
            and d_lost == 0
        )
        return {
            "name": name,
            "b1_gain": b1g,
            "b1_loss": b1l,
            "b1_net": b1g - b1l,
            "c_n": c_n,
            "c_succ": c_succ,
            "c_p64_succ": c_p64,
            "a1_base": len(a1_base),
            "a1_lost": a1_lost,
            "d_base": len(d_base),
            "d_lost": d_lost,
            "eligible": eligible,
        }

    sa, sb = summarize(rows_a, "P98A"), summarize(rows_b, "P98B")
    # select
    selected = None
    if sa["eligible"] and (not sb["eligible"] or sa["b1_net"] + sa["c_succ"] >= sb["b1_net"] + sb["c_succ"]):
        selected = ("P98A", actor_a, sa, best_a)
    elif sb["eligible"]:
        selected = ("P98B", actor_b, sb, best_b)
    elif sa["b1_net"] > 0 or sa["c_succ"] > sb["c_succ"]:
        selected = ("P98A", actor_a, sa, best_a)  # provisional; may fail ind
    elif sb["b1_net"] > 0 or sb["c_succ"] > 0:
        selected = ("P98B", actor_b, sb, best_b)

    c_ret_note = (
        "Dev C panel uses expert-verified holdouts, **not** P64-successful C roots "
        f"(P64 succ on dev C = {sa['c_p64_succ']}/{sa['c_n']}). "
        "C retention vs P64-success cannot be assessed on this development set."
    )

    # ------------------------------------------------------------------
    # Phase 6 — independent (only if selected)
    # ------------------------------------------------------------------
    ind_md = f"# INDEPENDENT_VALIDATION — P98\n\nGenerated: {_utc()}\n\n"
    case = "F"
    case_txt = "Clean development data or selection criteria insufficient."
    p64_replaced = False

    if selected is None:
        # Distinguish data scarcity (F) from offline-ok / closed-loop-fail (D)
        offline_ok = False
        try:
            for lab in ("P98A", "P98B"):
                p = OUT / f"P98_{lab[-1]}_RESULTS.csv"
                # files are P98_A_RESULTS / P98_B_RESULTS
            ra = list(csv.DictReader((OUT / "P98_A_RESULTS.csv").open(encoding="utf-8")))
            rb = list(csv.DictReader((OUT / "P98_B_RESULTS.csv").open(encoding="utf-8")))
            if ra and rb:
                offline_ok = float(ra[-1]["loss"]) < float(ra[0]["loss"]) or float(rb[-1]["loss"]) < float(rb[0]["loss"])
        except Exception:
            offline_ok = False
        if offline_ok and (sa["c_succ"] == 0 and sb["c_succ"] == 0 and sa["b1_net"] <= 0 and sb["b1_net"] <= 0):
            case = "D"
            case_txt = (
                "Offline B1/C correction improves, but physical closed-loop B1/C performance does not. "
                "Independent panel not opened."
            )
        elif len(c_train) < 4 or len(b1_corr) < 3:
            case = "F"
            case_txt = "Clean development data or physical-root independence is insufficient for a defensible conclusion."
        else:
            case = "F"
            case_txt = "No eligible development candidate; independent panel not opened."
        ind_md += f"No candidate opened for independent validation.\n\n**CASE {case}** — {case_txt}\n"
    else:
        name, actor_s, summ, best = selected
        print(f"[P98] Phase 6 independent with {name}", flush=True)
        # freeze selection
        torch.save(
            {"actor": actor_s.state_dict(), "candidate": name, "p64_sha": p64_sha, "dev": summ},
            CKPT / "SELECTED.pt",
        )
        ind_panel = {k: [t for t in v if t in starts] for k, v in ind.items()}
        rows_i = eval_panel(actor_s, p64, starts, ind_panel, book, rho, device, name)
        _write_csv(OUT / "INDEPENDENT_RESULTS.csv", rows_i)

        def ind_sum(skill: str) -> dict:
            rs = [r for r in rows_i if r["skill"] == skill]
            return {
                "n": len(rs),
                "p64": sum(int(r["P64_SUCCESS"]) for r in rs),
                "cand": sum(int(r["CAND_SUCCESS"]) for r in rs),
                "gain": sum(int(r["gain"]) for r in rs),
                "loss": sum(int(r["loss"]) for r in rs),
            }

        ib1, ia1, ic, id_ = ind_sum("B1"), ind_sum("A1"), ind_sum("C"), ind_sum("D")
        # class-balanced overall
        rates = []
        for s in (ib1, ia1, ic, id_):
            if s["n"]:
                rates.append(s["cand"] / s["n"])
        bal = float(np.mean(rates)) if rates else 0.0
        ind_md += f"Selected **{name}** (dev eligible={summ['eligible']}).\n\n"
        ind_md += f"{c_ret_note}\n\n"
        ind_md += "| Skill | n | P64 | Cand | gain | loss |\n|---|---:|---:|---:|---:|---:|\n"
        for sk, s in (("B1", ib1), ("A1", ia1), ("C", ic), ("D", id_)):
            ind_md += f"| {sk} | {s['n']} | {s['p64']} | {s['cand']} | {s['gain']} | {s['loss']} |\n"
        ind_md += f"\nClass-balanced cand success: **{bal:.3f}**\n"

        # CASE logic
        b1_ok = ib1["gain"] - ib1["loss"] >= 0 and ib1["loss"] <= 1
        c_ok = ic["cand"] >= ic["p64"] and ic["cand"] >= 1
        reg_ok = ia1["loss"] <= 1 and id_["loss"] <= 1
        offline_ok = True  # curves exist

        if not reg_ok or ib1["loss"] > 1:
            case = "E"
            case_txt = "B1/C changes caused unacceptable B1/A1/D forgetting on independent evaluation."
        elif summ["b1_net"] > 0 and summ["c_succ"] > 0 and (ib1["cand"] <= ib1["p64"] and ic["cand"] <= ic["p64"]):
            case = "D"
            case_txt = "Offline/dev B1/C signals improved, but independent closed-loop did not."
        elif name == "P98A" and sa["eligible"] and b1_ok and c_ok and reg_ok:
            case = "B"
            case_txt = "Residual-only joint training works; backbone adaptation is unnecessary."
        elif name == "P98B" and sb["eligible"] and b1_ok and c_ok and reg_ok and not sa["eligible"]:
            case = "C"
            case_txt = "Shared-backbone adaptation is needed for joint improvement."
        elif b1_ok and c_ok and reg_ok and (ib1["gain"] > 0 or ic["cand"] > ic["p64"]):
            case = "A"
            case_txt = "One unified Actor improves B1 and/or C without unacceptable regression on independent evaluation."
        elif b1_ok and reg_ok and ic["cand"] >= 1:
            case = "A"
            case_txt = "Unified Actor shows non-regressive B1/C evidence on independent evaluation."
        else:
            case = "E"
            case_txt = "Independent evaluation did not support joint B1+C improvement without regression."

        ind_md += f"\n## Decision signal\n\n**CASE {case}** — {case_txt}\n"
        ind_md += "\nP64 **not** replaced automatically.\n"

    _write(OUT / "INDEPENDENT_VALIDATION.md", ind_md)

    # Final report
    final = f"""# FINAL_REPORT — P98

Generated: {_utc()}

## Setup

- Init: original P64 sha `{p64_sha}` only (P94 unused).
- Architecture: one Transformer, one token head, one residual head.
- Physical-root sealing applied; `gbi_351` and P94 C demo `gbi_022` excluded.
- C preservation targets: **none** (0 eligible P64-successful C roots).

## Phase 1 data

| Set | Count |
|---|---:|
| B1 correction train roots | {len(b1_corr)} |
| C correction train roots | {len(c_train)} |
| C verified holdout (dev) | {len(c_dev_demos)} |
| A1 / D / B1 preservation | {len(a1_pres)} / {len(d_pres)} / {len(b1_pres)} |

{c_ret_note}

## Development

| Cand | B1 net | C succ (holdout) | A1 lost | D lost | eligible |
|---|---:|---:|---:|---:|---|
| P98-A | {sa['b1_net']} | {sa['c_succ']}/{sa['c_n']} | {sa['a1_lost']} | {sa['d_lost']} | {sa['eligible']} |
| P98-B | {sb['b1_net']} | {sb['c_succ']}/{sb['c_n']} | {sb['a1_lost']} | {sb['d_lost']} | {sb['eligible']} |

Selected for independent: **{selected[0] if selected else 'none'}**

## Decision

**CASE {case}** — {case_txt}

P64 replaced: **{p64_replaced}**

## Artifacts

Under `results/experiments/p98_unified_b1_c_consolidation/`.
"""
    _write(OUT / "FINAL_REPORT.md", final)
    (OUT / "SUMMARY.json").write_text(
        json.dumps(
            {
                "case": case,
                "selected": selected[0] if selected else None,
                "p64_sha": p64_sha,
                "p64_replaced": p64_replaced,
                "n_c_train": len(c_train),
                "n_c_dev": len(c_dev_demos),
                "n_b1_corr": len(b1_corr),
                "dev_A": sa,
                "dev_B": sb,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"[P98] CASE {case} selected={selected[0] if selected else None}", flush=True)


if __name__ == "__main__":
    main()
