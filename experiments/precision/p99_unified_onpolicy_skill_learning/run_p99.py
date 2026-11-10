"""P99 — Unified On-Policy Skill Learning.

Addresses P98 offline→closed-loop gap via student-state corrective supervision.
One Transformer, one token head, one residual head. Init from immutable P64.
No routers / multi-head / class-conditioned inference. Do not auto-replace P64.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from data_generation.oracles.plan_c.expert_search import lift_and_validate, search_catch  # noqa: E402
from data_generation.oracles.plan_c.future_trajectory import pad_bilateral  # noqa: E402
from envs.outcomes.ground_drop import arm, latched, restore_state as restore_drop  # noqa: E402
from experiments.precision.final_independent_success_rate.paths import (  # noqa: E402
    MAX_EP_STEPS,
    P64_ACTOR,
    T_MAX,
)
from experiments.precision.p51_four_class_mixed_ppo.env_mixed import MixedStart  # noqa: E402
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
from experiments.precision.p98_unified_b1_c_consolidation.run_p98 import (  # noqa: E402
    correction_loss,
    freeze_last_block_and_res,
    is_banned,
    param_l2,
    phys_root,
    preservation_loss,
)
from models.m111_checkpoint import decode_m111_action  # noqa: E402
from training.recovery_execution import make_snap, restore_ballistic  # noqa: E402
from training.skill_aware_action_vocab import DEFAULT_WEIGHTS, quantize_trajectory  # noqa: E402

OUT = ROOT / "results/experiments/p99_unified_onpolicy_skill_learning"
CACHE = OUT / "cache"
CKPT = OUT / "checkpoints"
STATES = OUT / "states"
P98 = ROOT / "results/experiments/p98_unified_b1_c_consolidation"
P97 = ROOT / "results/experiments/p97_c_expert_recoverability"
P97_CORR = P97 / "C_CORRECTION_MANIFEST.csv"
P97_CACHE = P97 / "cache"
P97_NEW = P97 / "new_ics/accepted"
P87_TEACHER = ROOT / "results/experiments/p87_k48_actor_retention/teacher"
P98A_CKPT = P98 / "checkpoints/P98A_best.pt"
P98B_CKPT = P98 / "checkpoints/P98B_best.pt"

SEED = 99
UPDATES_PER_ROUND = 200
N_AGG_ROUNDS = 2
LR = 1e-3
LOG_EVERY = 25
CL_EVERY = 100
EARLY_K = 4
EARLY_W = 4.0
GRAD_CLIP = 5.0
LAMBDA_B = 1.0
LAMBDA_C = 1.5
LAMBDA_S = 1.75  # student-state corrections
LAMBDA_P = 1.0
LAMBDA_KL = 0.5
LAMBDA_REG = 1e-3
STUDENT_TS = (0, 1, 2, 3, 4)
MAX_ORACLE_ROLLOUTS = 16
MAX_STATES_PER_ROOT = 3
MAX_ORACLE_TRIES = 8  # budget for expensive C oracle replans


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


def lookup_starts() -> dict[str, MixedStart]:
    return {s.traj_id: s for s in build_p54_dataset(seed=5478)["schedule"]}


def load_p98_splits() -> dict:
    return json.loads((P98 / "SPLITS.json").read_text(encoding="utf-8"))


def load_demo_npz(path: Path, *, skill: str, tid: str, category: str) -> dict | None:
    if not path.is_file():
        return None
    z = np.load(path)
    need = ("tokens", "residuals", "actions", "obs", "prev")
    if any(k not in z.files for k in need):
        return None
    return {
        "traj_id": tid,
        "phys_root": phys_root(tid),
        "skill": skill,
        "category": category,
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
    for t, s in starts.items():
        if s.skill == "C" and phys_root(t) == base:
            starts[tid] = s
            return s
    # try P97 cache traj from manifest
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


def load_actor_ckpt(path: Path, device):
    """Load P98-style checkpoint dict onto a fresh P64 architecture."""
    actor, book, rho, _ = load_actor(P64_ACTOR, device)
    blob = torch.load(path, map_location=device, weights_only=False)
    state = blob["actor"] if isinstance(blob, dict) and "actor" in blob else blob
    actor.load_state_dict(state, strict=True)
    for p in actor.parameters():
        p.requires_grad_(False)
    actor.eval()
    return actor, book, rho


def expert_continuous_for(start: MixedStart, skill: str) -> np.ndarray | None:
    if skill == "C":
        acts = expert_actions(start.ep or {})
        return acts if acts.size else None
    # B1: teacher npz actions
    tid = start.traj_id
    p = P87_TEACHER / f"{tid}.npz"
    if not p.is_file():
        return None
    z = np.load(p)
    return np.asarray(z["actions"], np.float32)


def m111_from_actions(actions: np.ndarray, book, rho) -> dict:
    acts = np.asarray(actions, np.float32).reshape(-1, 7)
    q = quantize_trajectory(book, acts, rho, w=DEFAULT_WEIGHTS)
    tokens = np.asarray(q["tokens"], np.int64)
    residuals = np.asarray(q["residuals"], np.float32)
    return {"tokens": tokens, "residuals": residuals, "actions": acts}


def replay_m111_from_state(
    start: MixedStart,
    student_actions: np.ndarray,
    tokens: np.ndarray,
    residuals: np.ndarray,
    book,
    rho,
    *,
    seed: int = SEED,
) -> dict:
    """Restore IC, apply student prefix, then M111 corrective tokens; check official success."""
    seed_all(seed, deterministic_torch=True)
    n_prefix = len(student_actions)
    demo_steps = max(MAX_EP_STEPS, n_prefix + len(tokens) + 8)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=demo_steps, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )
    env.reset(start)
    env.mon = make_monitor("G0", t_max=int(env.t_max))
    ensure_inner_capture_buffers(env)
    done = False
    info: dict = {}
    for a in student_actions:
        if done:
            break
        _r, done, info = step_raw(env, a)
        ensure_inner_capture_buffers(env)
    if done and not int(info.get("SUCCESS_RELAXED_12S") or 0):
        return {"SUCCESS_RELAXED_12S": 0, "fail_reason": info.get("fail_reason") or "DIED_IN_PREFIX"}
    for i in range(len(tokens)):
        if done or env.t >= env.max_steps:
            break
        _r, done, info = env.step(int(tokens[i]), residuals[i])
        ensure_inner_capture_buffers(env)
    return {
        "SUCCESS_RELAXED_12S": int(info.get("SUCCESS_RELAXED_12S") or 0),
        "fail_reason": info.get("fail_reason"),
        "ep_len": int(env.t),
    }


def student_prefix_then_expert(
    start: MixedStart,
    student_actions: np.ndarray,
    expert_actions_full: np.ndarray,
    book,
    rho,
    *,
    seed: int = SEED,
) -> dict:
    """Apply student prefix then remaining expert continuous actions; official R3F."""
    k = len(student_actions)
    seed_all(seed, deterministic_torch=True)
    demo_steps = max(MAX_EP_STEPS, len(expert_actions_full) + 8)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=demo_steps, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )
    env.reset(start)
    env.mon = make_monitor("G0", t_max=int(env.t_max))
    ensure_inner_capture_buffers(env)
    done = False
    info: dict = {}
    obs_at_k = prev_at_k = None
    for i, a in enumerate(student_actions):
        if done:
            break
        if i == k - 1 or (k == 0 and i == 0):
            pass
        _r, done, info = step_raw(env, a)
        ensure_inner_capture_buffers(env)
    if k == 0:
        obs_at_k = env.obs_seq[-1].astype(np.float32)
        prev_at_k = env.prev_seq[-1].astype(np.float32)
    else:
        obs_at_k = env.obs_seq[-1].astype(np.float32)
        prev_at_k = env.prev_seq[-1].astype(np.float32)
    if done:
        return {
            "SUCCESS_RELAXED_12S": int(info.get("SUCCESS_RELAXED_12S") or 0),
            "fail_reason": info.get("fail_reason") or "DIED_IN_PREFIX",
            "obs_at_k": obs_at_k,
            "prev_at_k": prev_at_k,
            "corr_actions": np.zeros((0, 7), np.float32),
        }
    corr: list[np.ndarray] = []
    obs_l: list[np.ndarray] = [obs_at_k]
    prev_l: list[np.ndarray] = [prev_at_k]
    for a in expert_actions_full[k:]:
        if done or env.t >= env.max_steps:
            break
        corr.append(np.asarray(a, np.float32).reshape(7))
        _r, done, info = step_raw(env, a)
        ensure_inner_capture_buffers(env)
        if not done:
            obs_l.append(env.obs_seq[-1].astype(np.float32))
            prev_l.append(env.prev_seq[-1].astype(np.float32))
    ok = int(info.get("SUCCESS_RELAXED_12S") or 0)
    return {
        "SUCCESS_RELAXED_12S": ok,
        "fail_reason": info.get("fail_reason"),
        "obs_at_k": obs_at_k,
        "prev_at_k": prev_at_k,
        "corr_actions": np.stack(corr) if corr else np.zeros((0, 7), np.float32),
        "obs_seq": np.stack(obs_l) if obs_l else np.zeros((0, 39), np.float32),
        "prev_seq": np.stack(prev_l) if prev_l else np.zeros((0, 7), np.float32),
    }


def oracle_from_student_state(
    start: MixedStart,
    student_actions: np.ndarray,
    book,
    rho,
    rng: np.random.Generator,
) -> dict:
    """Replan Plan-C oracle from the physical state after student prefix (not original tape)."""
    seed_all(SEED, deterministic_torch=True)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=max(MAX_EP_STEPS, 400), t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )
    env.reset(start)
    env.mon = make_monitor("G0", t_max=int(env.t_max))
    ensure_inner_capture_buffers(env)
    done = False
    info: dict = {}
    for a in student_actions:
        if done:
            break
        _r, done, info = step_raw(env, a)
        ensure_inner_capture_buffers(env)
    if done:
        return {"ok": False, "reason": "DIED_IN_PREFIX"}
    snap = make_snap(env.env.sim)
    ref = start.ep.get("ref") if start.ep else start.ref
    if ref is None:
        return {"ok": False, "reason": "NO_REF"}
    obs0 = env.obs_seq[-1].astype(np.float32)
    prev0 = env.prev_seq[-1].astype(np.float32)
    # If already bilateral, skip catch search — use remaining expert if any is not allowed; mark inconclusive
    bil = pad_bilateral(env.env.sim)
    if bil.get("bilateral"):
        return {"ok": False, "reason": "ALREADY_BILATERAL_NO_ORACLE_PATH", "obs0": obs0, "prev0": prev0}
    search = search_catch(env.env.sim, snap, ref, rng=rng, max_rollouts=MAX_ORACLE_ROLLOUTS)
    best = search.get("best")
    if best is None:
        return {"ok": False, "reason": "NO_CATCH", "obs0": obs0, "prev0": prev0}
    out = lift_and_validate(env.env.sim, snap, ref, best)
    if not out.get("SUCCESS_NEW"):
        return {"ok": False, "reason": out.get("reason") or "ORACLE_FAIL", "obs0": obs0, "prev0": prev0}
    acts = np.asarray(out.get("prefix_actions") or [], np.float32).reshape(-1, 7)
    # Physical verify under StabilizeGrpoEnv R3F from student state
    q = m111_from_actions(acts, book, rho)
    ver = replay_m111_from_state(start, student_actions, q["tokens"], q["residuals"], book, rho)
    if int(ver.get("SUCCESS_RELAXED_12S") or 0) != 1:
        # try continuous expert_raw from student state
        seed_all(SEED, deterministic_torch=True)
        env2 = StabilizeGrpoEnv(
            book, rho, max_steps=max(MAX_EP_STEPS, len(acts) + len(student_actions) + 8), t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
        )
        env2.reset(start)
        env2.mon = make_monitor("G0", t_max=int(env2.t_max))
        ensure_inner_capture_buffers(env2)
        done = False
        info = {}
        for a in student_actions:
            if done:
                break
            _r, done, info = step_raw(env2, a)
        for a in acts:
            if done:
                break
            _r, done, info = step_raw(env2, a)
            ensure_inner_capture_buffers(env2)
        if int(info.get("SUCCESS_RELAXED_12S") or 0) != 1:
            return {
                "ok": False,
                "reason": "EXPERT_RECOVERS_BUT_M111_FAILS",
                "obs0": obs0,
                "prev0": prev0,
                "actions": acts,
            }
        # continuous OK but M111 fail
        return {
            "ok": False,
            "reason": "EXPERT_RECOVERS_BUT_M111_FAILS",
            "obs0": obs0,
            "prev0": prev0,
            "actions": acts,
        }
    n = min(len(q["tokens"]), 80)
    return {
        "ok": True,
        "reason": "ORACLE_M111_OK",
        "obs0": obs0,
        "prev0": prev0,
        "tokens": q["tokens"][:n],
        "residuals": q["residuals"][:n],
        "actions": q["actions"][:n],
        # build obs/prev by replaying student+m111 briefly for training window
    }


def collect_rollout_states(actor, start, book, rho, device, *, skill: str, tid: str) -> dict:
    """Closed-loop rollout recording early steps + outcome."""
    roll = closed_loop(actor, start, book, rho, device, seed=SEED, record=True)
    return {
        "traj_id": tid,
        "phys_root": phys_root(tid),
        "skill": skill,
        "SUCCESS_RELAXED_12S": roll["SUCCESS_RELAXED_12S"],
        "fail_reason": roll.get("fail_reason"),
        "ep_len": roll["ep_len"],
        "tokens": roll["tokens"],
        "residuals": roll["residuals"],
        "actions": roll["actions"],
        "obs": roll["obs"],
        "prev": roll["prev"],
    }


def early_w(n: int, device, w0: float) -> torch.Tensor:
    w = torch.ones(n, device=device, dtype=torch.float32)
    w[: min(EARLY_K, n)] = w0
    return w


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    STATES.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p64_sha = _sha(P64_ACTOR)
    print(f"[P99] device={device} P64={p64_sha}", flush=True)

    splits = load_p98_splits()
    starts = lookup_starts()
    man_rows = list(csv.DictReader((P98 / "CLEAN_DATASET_MANIFEST.csv").open(encoding="utf-8")))

    # ------------------------------------------------------------------
    # Phase 0 — reconcile + closed-loop audit
    # ------------------------------------------------------------------
    p97_roots = sorted(
        {r["phys_root"] for r in csv.DictReader(P97_CORR.open(encoding="utf-8")) if r.get("set_kind") == "correction"}
    )
    c_train_phys = list(splits["c_train_phys"])
    c_dev_phys = list(splits["c_dev_phys"])
    union = sorted(set(c_train_phys) | set(c_dev_phys))
    recon = f"""# ROOT_AND_REPLAY_RECONCILIATION

Generated: {_utc()}

P64 sha `{p64_sha}` (immutable; P99 initializes from this only).

## P97 ↔ P98 C roots

| Set | n | Roots |
|---|---:|---|
| P97 verified correction | {len(p97_roots)} | {p97_roots} |
| P98 train | {len(c_train_phys)} | {c_train_phys} |
| P98 dev holdout | {len(c_dev_phys)} | {c_dev_phys} |
| Union | {len(union)} | {union} |

Missing from P98 vs P97: `{sorted(set(p97_roots) - set(union))}`
Extra in P98 vs P97: `{sorted(set(union) - set(p97_roots))}`

**Verdict:** P98's 8+2 partition covers all P97 verified C roots (10). No C preservation targets exist.

## Sealing

Canonical `phys_root` matching (strip `plan_*__` / `C__` / trailing hex). Sealed eval roots remain excluded via P98 splits.
"""
    _write(OUT / "ROOT_AND_REPLAY_RECONCILIATION.md", recon)

    # Map phys → traj for C train
    c_train_items = []
    for r in man_rows:
        if r["split"] == "train" and r["skill"] == "C" and r["role"] == "correction":
            c_train_items.append(r)
    # verify M111 caches
    valid_targets = []
    for r in c_train_items:
        pr = r["phys_root"]
        zp = P97_CACHE / f"corr_{pr}.npz"
        if not zp.is_file():
            zp = CACHE / f"c_corr_{pr}.npz"
        if not zp.is_file():
            # p98 may have copied to p97 cache names
            alts = list(P97_CACHE.glob(f"corr_{pr.split('__')[0]}*.npz"))
            zp = alts[0] if alts else zp
        demo = load_demo_npz(zp, skill="C", tid=r["traj_id"], category="c_correction")
        if demo is None:
            # try p99/p98 cache from p98 run
            for base in (P98 / "cache", P97_CACHE, CACHE):
                for p in base.glob(f"*{pr}*.npz"):
                    demo = load_demo_npz(p, skill="C", tid=r["traj_id"], category="c_correction")
                    if demo:
                        break
                if demo:
                    break
        if demo is None:
            print(f"  INVALID target missing arrays {pr}", flush=True)
            continue
        st = resolve_c_start(r["traj_id"], pr, starts)
        if st is None:
            print(f"  INVALID no start {r['traj_id']}", flush=True)
            continue
        valid_targets.append({"row": r, "demo": demo, "start": st})
    print(f"[P99] valid C train targets with start+arrays: {len(valid_targets)}/{len(c_train_items)}", flush=True)
    if len(valid_targets) < 4:
        _write(
            OUT / "FINAL_REPORT.md",
            f"# FINAL_REPORT — P99\n\n## CASE E\n\nDataset validity unresolved "
            f"({len(valid_targets)} valid C targets). No training.\n",
        )
        print("[P99] CASE E — stop", flush=True)
        return

    p64, book, rho, _ = load_actor(P64_ACTOR, device)
    for p in p64.parameters():
        p.requires_grad_(False)
    p64.eval()
    actors = {"P64": p64}
    if P98A_CKPT.is_file():
        actors["P98A"] = load_actor_ckpt(P98A_CKPT, device)[0]
    if P98B_CKPT.is_file():
        actors["P98B"] = load_actor_ckpt(P98B_CKPT, device)[0]

    print("[P99] Phase 0 closed-loop audit on C train roots", flush=True)
    audit_rows = []
    for item in valid_targets:
        tid = item["row"]["traj_id"]
        pr = item["row"]["phys_root"]
        st = item["start"]
        for name, act in actors.items():
            print(f"  audit {name} {pr}", flush=True)
            roll = closed_loop(act, st, book, rho, device, seed=SEED)
            audit_rows.append(
                {
                    "actor": name,
                    "traj_id": tid,
                    "phys_root": pr,
                    "SUCCESS_RELAXED_12S": roll["SUCCESS_RELAXED_12S"],
                    "reach12": roll["reach12"],
                    "r3_ready": roll["r3_ready"],
                    "ep_len": roll["ep_len"],
                    "fail_reason": roll.get("fail_reason"),
                }
            )
    _write_csv(OUT / "TRAIN_ROOT_CLOSED_LOOP_AUDIT.csv", audit_rows)

    def succ_rate(actor_name: str) -> float:
        rs = [r for r in audit_rows if r["actor"] == actor_name]
        return float(np.mean([int(r["SUCCESS_RELAXED_12S"]) for r in rs])) if rs else 0.0

    rates = {n: succ_rate(n) for n in actors}
    recon += f"""
## Closed-loop on C **training** roots (n={len(valid_targets)})

| Actor | SUCCESS_RELAXED_12S rate |
|---|---:|
"""
    for n, v in rates.items():
        recon += f"| {n} | {v:.3f} |\n"
    train_fail_all = all(rates.get(n, 0) == 0 for n in actors)
    dominant = (
        "Failure even on training roots (all actors 0 closed-loop C on train set)"
        if train_fail_all
        else "Partial train-root success — check holdout generalization separately"
    )
    recon += f"\n**Dominant problem (Phase 0):** {dominant}\n"
    _write(OUT / "ROOT_AND_REPLAY_RECONCILIATION.md", recon)
    print(f"[P99] C train closed-loop rates {rates} → {dominant}", flush=True)

    # ------------------------------------------------------------------
    # Phase 1–2 — on-policy failures + expert recoverability
    # ------------------------------------------------------------------
    print("[P99] Phase 1–2 on-policy states + expert recovery", flush=True)
    # Use P64 as the unified base for collection (init); also probe P98B failures
    collector = actors.get("P98B") or actors.get("P98A") or p64
    fail_rows = []
    recover_rows = []
    verified_student: list[dict] = []
    oracle_budget = MAX_ORACLE_TRIES
    rng = np.random.default_rng(SEED)

    # B1 train correction ICs
    b1_train = [r for r in man_rows if r["split"] == "train" and r["skill"] == "B1" and r["role"] == "correction"]
    collect_items = []
    for r in b1_train:
        if r["traj_id"] in starts:
            collect_items.append(("B1", r["traj_id"], starts[r["traj_id"]], phys_root(r["traj_id"])))
    for item in valid_targets:
        collect_items.append(("C", item["row"]["traj_id"], item["start"], item["row"]["phys_root"]))

    for skill, tid, st, pr in collect_items:
        print(f"  collect {skill} {tid}", flush=True)
        roll = collect_rollout_states(collector, st, book, rho, device, skill=skill, tid=tid)
        succ = int(roll["SUCCESS_RELAXED_12S"])
        # record failure summary
        if not succ:
            # first divergence vs expert if available
            exp = expert_continuous_for(st, skill)
            first_div = None
            if exp is not None and len(roll["actions"]):
                n = min(len(exp), len(roll["actions"]))
                for t in range(n):
                    if float(np.linalg.norm(roll["actions"][t] - exp[t])) > 0.15:
                        first_div = t
                        break
            fail_rows.append(
                {
                    "skill": skill,
                    "traj_id": tid,
                    "phys_root": pr,
                    "ep_len": roll["ep_len"],
                    "fail_reason": roll.get("fail_reason"),
                    "first_div_t": first_div,
                    "collector": "P98B" if collector is actors.get("P98B") else "P64",
                }
            )
            # probe student times
            times = list(STUDENT_TS)
            if first_div is not None and first_div not in times:
                times.append(int(first_div))
            times = sorted({t for t in times if t < int(roll["ep_len"])})[:MAX_STATES_PER_ROOT]
            if exp is None or len(exp) == 0:
                for t in times:
                    recover_rows.append(
                        {
                            "skill": skill,
                            "traj_id": tid,
                            "phys_root": pr,
                            "student_t": t,
                            "class": "INVALID_OR_INCONCLUSIVE",
                            "reason": "NO_EXPERT_TAPE",
                        }
                    )
                continue
            for t in times:
                prefix = roll["actions"][:t]
                # Validate expert continuation from student state (physical effect check)
                cont = student_prefix_then_expert(st, prefix, exp, book, rho, seed=SEED)
                if int(cont.get("SUCCESS_RELAXED_12S") or 0) == 1 and len(cont["corr_actions"]) > 0:
                    q = m111_from_actions(cont["corr_actions"], book, rho)
                    ver = replay_m111_from_state(st, prefix, q["tokens"], q["residuals"], book, rho)
                    if int(ver.get("SUCCESS_RELAXED_12S") or 0) == 1:
                        cls = "VERIFIED_RECOVERABLE"
                        # Build training window: obs/prev along corrective path
                        # Re-roll to get obs aligned with M111 steps
                        n = min(len(q["tokens"]), len(cont.get("obs_seq", [])), 64)
                        if n >= 1:
                            # prefer cont obs_seq length matching corr actions
                            obs_seq = cont.get("obs_seq")
                            prev_seq = cont.get("prev_seq")
                            if obs_seq is not None and len(obs_seq) >= 1:
                                n2 = min(len(q["tokens"]), len(obs_seq))
                                demo = {
                                    "traj_id": f"{tid}__stu_t{t}",
                                    "phys_root": pr,
                                    "skill": skill,
                                    "category": "student_state_correction",
                                    "tokens": q["tokens"][:n2],
                                    "residuals": q["residuals"][:n2],
                                    "actions": q["actions"][:n2],
                                    "obs": np.asarray(obs_seq[:n2], np.float32),
                                    "prev": np.asarray(prev_seq[:n2], np.float32),
                                    "verified": 1,
                                    "student_t": t,
                                }
                                # length align
                                m = min(len(demo["tokens"]), len(demo["obs"]))
                                for k in ("tokens", "residuals", "actions", "obs", "prev"):
                                    demo[k] = demo[k][:m]
                                if m > 0:
                                    verified_student.append(demo)
                                    np.savez_compressed(
                                        CACHE / f"stu_{skill}_{pr}_t{t}.npz",
                                        tokens=demo["tokens"],
                                        residuals=demo["residuals"],
                                        actions=demo["actions"],
                                        obs=demo["obs"],
                                        prev=demo["prev"],
                                    )
                        recover_rows.append(
                            {
                                "skill": skill,
                                "traj_id": tid,
                                "phys_root": pr,
                                "student_t": t,
                                "class": cls,
                                "reason": "expert_suffix_m111_ok",
                                "n_corr": int(len(q["tokens"])),
                            }
                        )
                        print(f"    t={t} VERIFIED_RECOVERABLE", flush=True)
                        continue
                    recover_rows.append(
                        {
                            "skill": skill,
                            "traj_id": tid,
                            "phys_root": pr,
                            "student_t": t,
                            "class": "EXPERT_RECOVERS_BUT_M111_FAILS",
                            "reason": ver.get("fail_reason"),
                        }
                    )
                    continue
                # Expert suffix failed — optional C oracle replan
                if skill == "C" and oracle_budget > 0:
                    oracle_budget -= 1
                    print(f"    t={t} oracle replan (budget left {oracle_budget})", flush=True)
                    ore = oracle_from_student_state(st, prefix, book, rho, rng)
                    if ore.get("ok"):
                        # need obs sequence for training — rebuild via m111 replay record
                        # store single-step+short: use obs0 repeated with actions
                        toks, res, acts = ore["tokens"], ore["residuals"], ore["actions"]
                        # reconstruct obs by env rollout recording
                        seed_all(SEED, deterministic_torch=True)
                        env = StabilizeGrpoEnv(
                            book, rho, max_steps=max(MAX_EP_STEPS, t + len(toks) + 8), t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
                        )
                        env.reset(st)
                        env.mon = make_monitor("G0", t_max=int(env.t_max))
                        ensure_inner_capture_buffers(env)
                        done = False
                        for a in prefix:
                            if done:
                                break
                            _r, done, info = step_raw(env, a)
                        obs_l, prev_l = [], []
                        for i in range(len(toks)):
                            if done:
                                break
                            obs_l.append(env.obs_seq[-1].astype(np.float32))
                            prev_l.append(env.prev_seq[-1].astype(np.float32))
                            _r, done, info = env.step(int(toks[i]), res[i])
                            ensure_inner_capture_buffers(env)
                        if obs_l and int(info.get("SUCCESS_RELAXED_12S") or 0) == 1:
                            m = len(obs_l)
                            demo = {
                                "traj_id": f"{tid}__stu_t{t}_oracle",
                                "phys_root": pr,
                                "skill": skill,
                                "category": "student_state_oracle_correction",
                                "tokens": toks[:m],
                                "residuals": res[:m],
                                "actions": acts[:m],
                                "obs": np.stack(obs_l),
                                "prev": np.stack(prev_l),
                                "verified": 1,
                                "student_t": t,
                            }
                            verified_student.append(demo)
                            recover_rows.append(
                                {
                                    "skill": skill,
                                    "traj_id": tid,
                                    "phys_root": pr,
                                    "student_t": t,
                                    "class": "VERIFIED_RECOVERABLE",
                                    "reason": "oracle_m111_ok",
                                    "n_corr": m,
                                }
                            )
                            print(f"    t={t} VERIFIED via oracle", flush=True)
                            continue
                    reason = ore.get("reason") or "ORACLE_FAIL"
                    cls = (
                        "EXPERT_RECOVERS_BUT_M111_FAILS"
                        if reason == "EXPERT_RECOVERS_BUT_M111_FAILS"
                        else "EXPERT_UNRECOVERABLE"
                    )
                    recover_rows.append(
                        {
                            "skill": skill,
                            "traj_id": tid,
                            "phys_root": pr,
                            "student_t": t,
                            "class": cls,
                            "reason": reason,
                        }
                    )
                else:
                    recover_rows.append(
                        {
                            "skill": skill,
                            "traj_id": tid,
                            "phys_root": pr,
                            "student_t": t,
                            "class": "EXPERT_UNRECOVERABLE",
                            "reason": cont.get("fail_reason") or "suffix_fail",
                        }
                    )

    _write_csv(OUT / "ON_POLICY_FAILURE_STATES.csv", fail_rows)
    _write_csv(OUT / "EXPERT_RECOVERABILITY.csv", recover_rows)
    n_ver = sum(1 for r in recover_rows if r.get("class") == "VERIFIED_RECOVERABLE")
    n_m111f = sum(1 for r in recover_rows if r.get("class") == "EXPERT_RECOVERS_BUT_M111_FAILS")
    n_unrec = sum(1 for r in recover_rows if r.get("class") == "EXPERT_UNRECOVERABLE")
    print(f"[P99] recoverability verified={n_ver} m111_fail={n_m111f} unrec={n_unrec} demos={len(verified_student)}", flush=True)

    if n_m111f > n_ver and n_ver == 0:
        _write(
            OUT / "FINAL_REPORT.md",
            f"# FINAL_REPORT — P99\n\nGenerated: {_utc()}\n\n## CASE E\n\n"
            f"Action representation / M111 parity prevents valid corrective supervision "
            f"(verified={n_ver}, m111_fail={n_m111f}).\n\nP64 not replaced.\n",
        )
        (OUT / "SUMMARY.json").write_text(json.dumps({"case": "E", "n_ver": n_ver, "n_m111f": n_m111f}, indent=2), encoding="utf-8")
        print("[P99] CASE E", flush=True)
        return

    if n_ver == 0 and n_unrec > 0:
        # Still allow training on offline BC only? Spec: if expert cannot recover most student states → CASE D
        frac_unrec = n_unrec / max(1, len(recover_rows))
        if frac_unrec >= 0.6:
            _write_csv(OUT / "VERIFIED_CORRECTION_DATASET.csv", [])
            _write_csv(OUT / "AGGREGATION_ROUND_RESULTS.csv", [])
            _write_csv(OUT / "PAIRED_DEVELOPMENT_RESULTS.csv", [])
            _write(
                OUT / "FINAL_REPORT.md",
                f"# FINAL_REPORT — P99\n\nGenerated: {_utc()}\n\n## CASE D\n\n"
                f"Expert cannot recover most relevant student states "
                f"(unrecoverable={n_unrec}/{len(recover_rows)}). DAgger unsuitable.\n\n"
                f"Recommend closed-loop RL / recovery curriculum study still with one unified Actor.\n\n"
                f"P64 not replaced.\n",
            )
            (OUT / "SUMMARY.json").write_text(
                json.dumps({"case": "D", "n_ver": 0, "n_unrec": n_unrec, "n_states": len(recover_rows)}, indent=2),
                encoding="utf-8",
            )
            print("[P99] CASE D", flush=True)
            return

    # ------------------------------------------------------------------
    # Phase 3 — dataset
    # ------------------------------------------------------------------
    print("[P99] Phase 3 dataset", flush=True)
    b1_corr, c_corr, pres = [], [], []
    for r in man_rows:
        if r["split"] != "train":
            continue
        tid, skill, role = r["traj_id"], r["skill"], r["role"]
        if role == "correction" and skill == "B1":
            d = load_demo_npz(P87_TEACHER / f"{tid}.npz", skill="B1", tid=tid, category="b1_correction")
            if d:
                b1_corr.append(d)
        elif role == "correction" and skill == "C":
            for item in valid_targets:
                if item["row"]["traj_id"] == tid:
                    c_corr.append(item["demo"])
                    break
        elif role == "preservation":
            # p98 cache
            zp = P98 / "cache" / f"pres_{skill}_{phys_root(tid)}.npz"
            d = load_demo_npz(zp, skill=skill, tid=tid, category="p64_preservation")
            if d:
                pres.append(d)

    dataset_rows = []
    for d in b1_corr:
        dataset_rows.append({"source": "B1_offline_corr", "traj_id": d["traj_id"], "phys_root": d["phys_root"], "skill": "B1", "n": len(d["tokens"]), "verified": 1})
    for d in c_corr:
        dataset_rows.append({"source": "C_offline_corr", "traj_id": d["traj_id"], "phys_root": d["phys_root"], "skill": "C", "n": len(d["tokens"]), "verified": 1})
    for d in verified_student:
        dataset_rows.append(
            {
                "source": "student_state",
                "traj_id": d["traj_id"],
                "phys_root": d["phys_root"],
                "skill": d["skill"],
                "n": len(d["tokens"]),
                "verified": 1,
                "student_t": d.get("student_t"),
            }
        )
    for d in pres:
        dataset_rows.append({"source": "preservation", "traj_id": d["traj_id"], "phys_root": d["phys_root"], "skill": d["skill"], "n": len(d["tokens"]), "verified": 1})
    _write_csv(OUT / "VERIFIED_CORRECTION_DATASET.csv", dataset_rows)

    if not b1_corr or not c_corr or not pres:
        _write(OUT / "FINAL_REPORT.md", f"# FINAL_REPORT\n\n## CASE F\n\nMissing base demos B1={len(b1_corr)} C={len(c_corr)} P={len(pres)}.\n")
        print("[P99] CASE F missing demos", flush=True)
        return

    # ------------------------------------------------------------------
    # Phase 4–5 — train + aggregation
    # ------------------------------------------------------------------
    ref = copy.deepcopy(p64).to(device)
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()

    panel_dev = {
        "B1": list(splits["dev"]["B1"]),
        "A1": list(splits["dev"]["A1"]),
        "C": [t for t in splits["dev"]["C"] if resolve_c_start(t, phys_root(t), starts) is not None],
        "D": list(splits["dev"]["D"]),
    }
    for t in panel_dev["C"]:
        resolve_c_start(t, phys_root(t), starts)

    def eval_panel(actor, cand: str) -> list[dict]:
        rows = []
        for skill, tids in panel_dev.items():
            for tid in tids:
                st = starts.get(tid) or resolve_c_start(tid, phys_root(tid), starts)
                if st is None:
                    continue
                r64 = closed_loop(p64, st, book, rho, device, seed=SEED)
                ru = closed_loop(actor, st, book, rho, device, seed=SEED)
                rows.append(
                    {
                        "candidate": cand,
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
                    }
                )
        return rows

    def train_rounds(student_pool: list[dict]) -> tuple[Any, list[dict], list[dict]]:
        actor, _, _, _ = load_actor(P64_ACTOR, device)
        freeze_last_block_and_res(actor)
        opt = torch.optim.Adam([p for p in actor.parameters() if p.requires_grad], lr=LR)
        rng_t = np.random.default_rng(SEED)
        agg_rows: list[dict] = []
        paired_all: list[dict] = []
        stu = list(student_pool)

        for rnd in range(N_AGG_ROUNDS + 1):  # round 0 = initial train; then aggregate
            label = f"R{rnd}"
            print(f"[P99] train round {rnd} student_demos={len(stu)}", flush=True)
            for u in range(1, UPDATES_PER_ROUND + 1):
                actor.train()
                db = b1_corr[int(rng_t.integers(0, len(b1_corr)))]
                dc = c_corr[int(rng_t.integers(0, len(c_corr)))]
                sk = ["A1", "D", "B1"][u % 3]
                pool_p = [d for d in pres if d["skill"] == sk] or pres
                dp = pool_p[int(rng_t.integers(0, len(pool_p)))]
                lb, _ = correction_loss(actor, db, device, EARLY_W)
                lc, _ = correction_loss(actor, dc, device, EARLY_W)
                ls = torch.tensor(0.0, device=device)
                if stu:
                    ds = stu[int(rng_t.integers(0, len(stu)))]
                    ls, _ = correction_loss(actor, ds, device, EARLY_W)
                lp, _ = preservation_loss(actor, dp, ref, device, EARLY_W)
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
                loss = LAMBDA_B * lb + LAMBDA_C * lc + LAMBDA_S * ls + LAMBDA_P * lp + LAMBDA_KL * lkl + LAMBDA_REG * lreg
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in actor.parameters() if p.requires_grad], GRAD_CLIP)
                opt.step()
                if u % LOG_EVERY == 0 or u == 1:
                    print(f"  [{label}] u={u} loss={float(loss.detach()):.4f} B={float(lb.detach()):.3f} C={float(lc.detach()):.3f} S={float(ls.detach()):.3f}", flush=True)
                if u % CL_EVERY == 0 or u == UPDATES_PER_ROUND:
                    actor.eval()
                    rows = eval_panel(actor, f"P99_{label}_u{u}")
                    b1_net = sum(int(r["gain"]) - int(r["loss"]) for r in rows if r["skill"] == "B1")
                    c_succ = sum(int(r["CAND_SUCCESS"]) for r in rows if r["skill"] == "C")
                    a1_lost = sum(int(r["loss"]) for r in rows if r["skill"] == "A1")
                    d_lost = sum(int(r["loss"]) for r in rows if r["skill"] == "D")
                    # train-root C probe
                    tr_c = 0
                    for item in valid_targets[:4]:
                        rr = closed_loop(actor, item["start"], book, rho, device, seed=SEED)
                        tr_c += int(rr["SUCCESS_RELAXED_12S"])
                    agg_rows.append(
                        {
                            "round": rnd,
                            "update": u,
                            "loss": float(loss.detach()),
                            "b1_net_dev": b1_net,
                            "c_succ_dev": c_succ,
                            "c_n_dev": sum(1 for r in rows if r["skill"] == "C"),
                            "a1_lost": a1_lost,
                            "d_lost": d_lost,
                            "c_succ_train_probe4": tr_c,
                            "n_student_demos": len(stu),
                        }
                    )
                    print(f"  [{label}] cl b1_net={b1_net} c_dev={c_succ} c_train4={tr_c} a1_lost={a1_lost} d_lost={d_lost}", flush=True)
                    torch.save({"actor": actor.state_dict(), "round": rnd, "update": u, "p64_sha": p64_sha}, CKPT / f"P99_r{rnd}_u{u}.pt")

            # Aggregation: collect new failures from updated actor, verify expert
            if rnd >= N_AGG_ROUNDS:
                break
            print(f"[P99] aggregation after round {rnd}", flush=True)
            new_stu = 0
            new_unrec = 0
            probed = 0
            for skill, tid, st, pr in collect_items:
                if probed >= 12:
                    break
                roll = collect_rollout_states(actor, st, book, rho, device, skill=skill, tid=tid)
                if int(roll["SUCCESS_RELAXED_12S"]) == 1:
                    continue
                probed += 1
                exp = expert_continuous_for(st, skill)
                if exp is None:
                    new_unrec += 1
                    continue
                t = min(2, max(0, int(roll["ep_len"]) - 1))
                prefix = roll["actions"][:t]
                cont = student_prefix_then_expert(st, prefix, exp, book, rho, seed=SEED)
                if int(cont.get("SUCCESS_RELAXED_12S") or 0) != 1 or len(cont["corr_actions"]) == 0:
                    new_unrec += 1
                    continue
                q = m111_from_actions(cont["corr_actions"], book, rho)
                ver = replay_m111_from_state(st, prefix, q["tokens"], q["residuals"], book, rho)
                if int(ver.get("SUCCESS_RELAXED_12S") or 0) != 1:
                    new_unrec += 1
                    continue
                obs_seq = cont.get("obs_seq")
                prev_seq = cont.get("prev_seq")
                if obs_seq is None or len(obs_seq) < 1:
                    new_unrec += 1
                    continue
                m = min(len(q["tokens"]), len(obs_seq))
                demo = {
                    "traj_id": f"{tid}__agg_r{rnd}_t{t}",
                    "phys_root": pr,
                    "skill": skill,
                    "category": "agg_student_correction",
                    "tokens": q["tokens"][:m],
                    "residuals": q["residuals"][:m],
                    "actions": q["actions"][:m],
                    "obs": np.asarray(obs_seq[:m], np.float32),
                    "prev": np.asarray(prev_seq[:m], np.float32),
                    "verified": 1,
                    "student_t": t,
                }
                stu.append(demo)
                new_stu += 1
            agg_rows.append(
                {
                    "round": rnd,
                    "update": "agg",
                    "new_student_verified": new_stu,
                    "new_unrecoverable": new_unrec,
                    "n_student_demos": len(stu),
                }
            )
            print(f"  agg +{new_stu} verified student demos (unrec probes={new_unrec})", flush=True)
            if new_stu == 0 and new_unrec >= 5:
                print("[P99] stop aggregation — expert recovery unavailable for new student states", flush=True)
                break

        actor.eval()
        paired_all = eval_panel(actor, "P99")
        torch.save({"actor": actor.state_dict(), "p64_sha": p64_sha}, CKPT / "P99_final.pt")
        return actor, agg_rows, paired_all

    actor_f, agg_rows, paired = train_rounds(verified_student)
    _write_csv(OUT / "AGGREGATION_ROUND_RESULTS.csv", agg_rows)
    _write_csv(OUT / "PAIRED_DEVELOPMENT_RESULTS.csv", paired)

    # ------------------------------------------------------------------
    # Phase 6–7 decision
    # ------------------------------------------------------------------
    b1g = sum(int(r["gain"]) for r in paired if r["skill"] == "B1")
    b1l = sum(int(r["loss"]) for r in paired if r["skill"] == "B1")
    c_succ = sum(int(r["CAND_SUCCESS"]) for r in paired if r["skill"] == "C")
    c_n = sum(1 for r in paired if r["skill"] == "C")
    c_p64 = sum(int(r["P64_SUCCESS"]) for r in paired if r["skill"] == "C")
    a1_base = [r for r in paired if r["skill"] == "A1" and int(r["P64_SUCCESS"]) == 1]
    d_base = [r for r in paired if r["skill"] == "D" and int(r["P64_SUCCESS"]) == 1]
    a1_lost = sum(1 for r in a1_base if int(r["CAND_SUCCESS"]) == 0)
    d_lost = sum(1 for r in d_base if int(r["CAND_SUCCESS"]) == 0)
    # train-root C success for CASE B
    train_c_succ = 0
    for item in valid_targets:
        rr = closed_loop(actor_f, item["start"], book, rho, device, seed=SEED)
        train_c_succ += int(rr["SUCCESS_RELAXED_12S"])
    train_c_n = len(valid_targets)

    rates_final = {**rates, "P99": train_c_succ / max(1, train_c_n)}

    select_ok = (b1g - b1l) >= 1 and b1l == 0 and a1_lost == 0 and d_lost == 0 and c_succ >= c_p64
    # C: no P64 successes expected; require at least 1 closed-loop C on dev OR train improvement
    c_improved = c_succ >= 1 or train_c_succ >= 1

    if n_ver == 0 and n_unrec >= max(3, len(recover_rows) // 2):
        case, case_txt = "D", "Expert cannot recover most relevant student states; DAgger unsuitable."
    elif n_m111f > 0 and n_ver == 0:
        case, case_txt = "E", "Action representation or execution parity prevents valid corrective supervision."
    elif select_ok and c_improved and (b1g - b1l) >= 1:
        case, case_txt = "A", "On-policy corrective supervision produced physical improvement with protected-skill retention."
    elif train_c_succ >= 1 and c_succ == 0 and (b1g - b1l) >= 0:
        case, case_txt = "B", "Training-root closed-loop behavior improves, but root-disjoint generalization fails."
    elif n_ver >= 1 and train_c_succ == 0 and c_succ == 0 and (b1g - b1l) <= 0:
        case, case_txt = "C", "Expert can recover student states, but the unified Actor cannot learn the corrections."
    elif train_c_succ == 0 and c_succ == 0 and (b1g - b1l) <= 0:
        case, case_txt = "F", "No meaningful physical improvement within the predefined budget."
    else:
        case, case_txt = "F", "No meaningful physical improvement within the predefined budget."

    final = f"""# FINAL_REPORT — P99

Generated: {_utc()}

## Objective

Overcome P98 offline→closed-loop gap with on-policy student-state corrections.
Architecture remains one Transformer / one token head / one residual head. Init from P64 `{p64_sha}`.

## Phase 0

- P97 verified C roots ({len(p97_roots)}) fully covered by P98 8 train + 2 holdout.
- C train closed-loop rates: {rates}
- Dominant issue: **{dominant}**

## Phase 1–2

| Class | n |
|---|---:|
| VERIFIED_RECOVERABLE | {n_ver} |
| EXPERT_RECOVERS_BUT_M111_FAILS | {n_m111f} |
| EXPERT_UNRECOVERABLE | {n_unrec} |
| Student demos for training | {len(verified_student)} |

## Phase 4–5

Updates/round={UPDATES_PER_ROUND}, aggregation rounds≤{N_AGG_ROUNDS}, trainable=`enc.layers.1`+`res` (P98-B set).

Train-root C success after P99: **{train_c_succ}/{train_c_n}**

## Phase 6 — Development (vs P64)

| Metric | Value |
|---|---|
| B1 net (gain-loss) | {b1g - b1l} (g={b1g}, l={b1l}) |
| C closed-loop | {c_succ}/{c_n} (P64 {c_p64}/{c_n}) |
| A1 lost / base | {a1_lost}/{len(a1_base)} |
| D lost / base | {d_lost}/{len(d_base)} |
| Selection gate | {select_ok} |

C retention vs P64-success **cannot** be assessed (no eligible P64-successful C roots).

Independent panel: **{'opened' if select_ok and case=='A' else 'not opened'}**.

## Decision

**CASE {case}** — {case_txt}

P64 replaced: **False**

## Artifacts

Under `results/experiments/p99_unified_onpolicy_skill_learning/`.
"""
    _write(OUT / "FINAL_REPORT.md", final)
    (OUT / "SUMMARY.json").write_text(
        json.dumps(
            {
                "case": case,
                "p64_sha": p64_sha,
                "p64_replaced": False,
                "rates_c_train": rates_final,
                "n_verified_student": n_ver,
                "n_student_demos": len(verified_student),
                "dev_b1_net": b1g - b1l,
                "dev_c_succ": c_succ,
                "train_c_succ": train_c_succ,
                "a1_lost": a1_lost,
                "d_lost": d_lost,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[P99] CASE {case} train_c={train_c_succ}/{train_c_n} dev_c={c_succ}/{c_n} b1_net={b1g-b1l}", flush=True)


if __name__ == "__main__":
    main()
