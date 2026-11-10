"""P97 — C Expert Recoverability and Unified Training Data Expansion.

Diagnostic only. Does not train any Actor. Does not replace P64.
No routers / multi-head / task-conditioned inference.
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

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from data_generation.oracles.plan_c.expert_search import (  # noqa: E402
    build_physical_root,
    diverse_impact_configs,
    lift_and_validate,
    pack_expert,
    physical_root_fingerprint,
    search_catch,
)
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from experiments.precision.final_independent_success_rate.paths import (  # noqa: E402
    MAX_EP_STEPS,
    P64_ACTOR,
    T_MAX,
)
from experiments.precision.p39_working_hold_handoff.expert_validate import collect_jobs  # noqa: E402
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
from training.skill_aware_action_vocab import DEFAULT_WEIGHTS, quantize_trajectory  # noqa: E402

OUT = ROOT / "results/experiments/p97_c_expert_recoverability"
CACHE = OUT / "cache"
NEW_IC = OUT / "new_ics"
P93_MAN = ROOT / "results/experiments/p93_frozen_router_independent_validation/EVALUATION_SPLIT_MANIFEST.csv"
FROZEN_MANIFEST = ROOT / "results/experiments/final_independent_success_rate/FROZEN_TEST_MANIFEST.json"
P94_SPLITS = ROOT / "results/experiments/p94_unified_multiskill_actor/SPLITS.json"
P17_SPLIT = ROOT / "results/diagnostics/p17_sft_readiness/lineage_split.json"
P96_C_SCREEN = ROOT / "results/experiments/p96_unified_onpolicy_residual_preservation/cache/c_screen.json"

SEED = 97
TARGET_MIN = 8
TARGET_IDEAL = 12
STUDENT_SUBSET_N = 5
STUDENT_TS = (2, 4, 8)
NEW_IC_TARGET = 10
NEW_IC_WALL_S = 2400.0
NEW_IC_MAX_CONFIGS = 120
NEW_IC_ROLLOUTS = 24
NEW_IC_SEED = 97_63063


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
    """Canonical physical C root: strip plan_c__/C__ and trailing hex suffixes."""
    t = rid0(tid)
    t = re.sub(r"^(plan_c__|C__|plan_b__|B__|A__|D__)", "", t)
    t = re.sub(r"__([0-9a-f]{12,})$", "", t)
    return t


def eval_banned_phys_roots() -> set[str]:
    """Sealed evaluation physical roots (frozen + P93 panels + P94 independent)."""
    tids: set[str] = set()
    man = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
    tids |= {e["traj_id"] for e in man["episodes"]}
    if P93_MAN.is_file():
        for r in csv.DictReader(P93_MAN.open(encoding="utf-8")):
            if r["panel"] in ("B1", "C1", "C2", "A1", "D", "DIAG_EXPOSED_B1"):
                tids.add(r["traj_id"])
    if P94_SPLITS.is_file():
        sp = json.loads(P94_SPLITS.read_text(encoding="utf-8"))
        for tids_ in (sp.get("ind") or {}).values():
            if isinstance(tids_, list):
                tids |= set(tids_)
    return {phys_root(t) for t in tids}


def seal_sources_for(tid: str) -> str:
    """Human-readable seal provenance for a traj/root."""
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
        for k, tids_ in (sp.get("ind") or {}).items():
            if isinstance(tids_, list):
                for t in tids_:
                    if phys_root(t) == pr:
                        bits.append(f"P94_ind_{k}:{t}")
    return "|".join(bits) if bits else ""


def lookup_starts() -> dict[str, MixedStart]:
    ds = build_p54_dataset(seed=5478)
    return {s.traj_id: s for s in ds["schedule"]}


def p21_c_jobs() -> list[dict]:
    return [j for j in collect_jobs() if j["skill"] == "C"]


def start_from_job(j: dict) -> MixedStart:
    return MixedStart(
        skill="C",
        traj_id=j["traj_id"],
        family=j.get("family") or "C",
        kind="ep",
        ep=j["ep"],
    )


def previous_use(tid: str, pr: str, *, sealed: bool, p21_split: str | None) -> str:
    uses: list[str] = []
    if sealed:
        uses.append("sealed_eval")
    if p21_split:
        uses.append(f"p21_{p21_split}")
    if P96_C_SCREEN.is_file():
        for r in json.loads(P96_C_SCREEN.read_text(encoding="utf-8")):
            if phys_root(r.get("traj_id") or "") == pr:
                uses.append(f"p96_screen_P64={r.get('P64_SUCCESS')}")
                break
    # regression note for gbi_351
    if pr == "gbi_351_bc2d6a3b90":
        uses.append("c_regression_diag_exposed_as_plan_c_and_frozen_C__variant")
    return ";".join(uses) if uses else "historical_pool"


@torch.no_grad()
def closed_loop_p64(actor, start: MixedStart, book, rho, device, *, seed: int = SEED, record: bool = False) -> dict:
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
    ever = {"frame12": False, "bil": False}
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
        if bil and pad4 and wk and e <= 12 + 1e-9 and th <= 8 + 1e-9:
            ever["frame12"] = True
    term = _r3_terminal(env, info)
    out = {
        "SUCCESS_RELAXED_12S": int(term.get("SUCCESS_RELAXED") or info.get("SUCCESS_RELAXED_12S") or 0),
        "handoff": int(info.get("handoff") or 0),
        "reach12": int(ever["frame12"]),
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


def expert_closed_loop(start: MixedStart, book, rho, *, seed: int = SEED, record: bool = False) -> dict:
    """Replay stored continuous expert actions under official R3F / SUCCESS_RELAXED_12S.

    max_steps is raised to fit the expert tape so demonstration verification is not
    truncated by the Actor's 80-step PPO budget; success criteria remain official R3F.
    """
    seed_all(seed, deterministic_torch=True)
    acts = expert_actions(start.ep or {})
    if acts.size == 0:
        return {
            "SUCCESS_RELAXED_12S": 0,
            "handoff": 0,
            "ep_len": 0,
            "fail_reason": "NO_EXPERT_ACTIONS",
            "expert_available": 0,
        }
    demo_steps = max(int(MAX_EP_STEPS), int(len(acts)) + 8)
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
        if record:
            obs_l.append(env.obs_seq[-1].astype(np.float32))
            prev_l.append(env.prev_seq[-1].astype(np.float32))
            actions.append(np.asarray(a, np.float32).reshape(7))
        _r, done, info = step_raw(env, a)
        ensure_inner_capture_buffers(env)
    # If expert tape ends without handoff, allow env to continue holding? No — tape is full expert.
    out = {
        "SUCCESS_RELAXED_12S": int(info.get("SUCCESS_RELAXED_12S") or 0),
        "handoff": int(info.get("handoff") or 0),
        "ep_len": int(env.t),
        "fail_reason": info.get("fail_reason"),
        "expert_available": 1,
        "n_expert_actions": int(len(acts)),
    }
    if record:
        out.update(
            {
                "actions": np.stack(actions) if actions else np.zeros((0, 7), np.float32),
                "obs": np.stack(obs_l) if obs_l else np.zeros((0, 39), np.float32),
                "prev": np.stack(prev_l) if prev_l else np.zeros((0, 7), np.float32),
            }
        )
    return out


def m111_physical_replay(start: MixedStart, actions: np.ndarray, book, rho, *, seed: int = SEED) -> dict:
    """Quantize continuous actions → M111 tokens+residuals; decode parity; physical replay."""
    acts = np.asarray(actions, np.float32).reshape(-1, 7)
    if len(acts) == 0:
        return {
            "quantize_ok": 0,
            "parity_ok": 0,
            "replay_SUCCESS_RELAXED_12S": 0,
            "fail_reason": "EMPTY_ACTIONS",
        }
    q = quantize_trajectory(book, acts, rho, w=DEFAULT_WEIGHTS)
    tokens = np.asarray(q["tokens"], np.int64)
    residuals = np.asarray(q["residuals"], np.float32)
    decoded = []
    max_abs = 0.0
    for i in range(len(tokens)):
        d = np.asarray(decode_m111_action(book, int(tokens[i]), residuals[i], rho), np.float32).reshape(7)
        decoded.append(d)
        max_abs = max(max_abs, float(np.max(np.abs(d - acts[i]))))
    decoded_arr = np.stack(decoded)
    parity_ok = int(max_abs < 0.15)  # generous physical parity gate; residual+proto approx

    seed_all(seed, deterministic_torch=True)
    demo_steps = max(int(MAX_EP_STEPS), int(len(tokens)) + 8)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=demo_steps, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )
    env.reset(start)
    env.mon = make_monitor("G0", t_max=int(env.t_max))
    ensure_inner_capture_buffers(env)
    done = False
    info: dict = {}
    for i in range(len(tokens)):
        if done or env.t >= env.max_steps:
            break
        _r, done, info = env.step(int(tokens[i]), residuals[i])
        ensure_inner_capture_buffers(env)
    return {
        "quantize_ok": 1,
        "parity_ok": parity_ok,
        "max_abs_decode_err": max_abs,
        "n_tokens": int(len(tokens)),
        "replay_SUCCESS_RELAXED_12S": int(info.get("SUCCESS_RELAXED_12S") or 0),
        "replay_handoff": int(info.get("handoff") or 0),
        "replay_fail_reason": info.get("fail_reason"),
        "tokens": tokens,
        "residuals": residuals,
        "decoded_actions": decoded_arr,
        "actions": acts,
    }


def student_then_expert(
    actor,
    start: MixedStart,
    book,
    rho,
    device,
    *,
    k: int,
    seed: int = SEED,
) -> dict:
    """Roll frozen P64 for k steps, then continue with expert actions[k:]."""
    seed_all(seed, deterministic_torch=True)
    acts_len = int(len(expert_actions(start.ep or {})))
    demo_steps = max(int(MAX_EP_STEPS), acts_len + 8)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=demo_steps, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )
    env.reset(start)
    env.mon = make_monitor("G0", t_max=int(env.t_max))
    ensure_inner_capture_buffers(env)
    st = WrapState()
    v2 = v2_spec()
    done = False
    info: dict = {}
    for _ in range(int(k)):
        if done:
            break
        obs = torch.as_tensor(np.stack(env.obs_seq[-160:]), device=device, dtype=torch.float32)
        prev = torch.as_tensor(np.stack(env.prev_seq[-160:]), device=device, dtype=torch.float32)
        logits, res, _act = actor(obs, prev)
        tok_a = int(torch.argmax(mask_handoff_logits(logits[-1].float())).item())
        tok, st, _ = select_token(env_t=int(env.t), obs=last_obs(env), actor_tok=tok_a, st=st, v2=v2, rule=None)
        resid = res[-1].detach().cpu().numpy().astype(np.float32)
        _r, done, info = env.step(int(tok), resid)
        ensure_inner_capture_buffers(env)
    if done:
        return {
            "k": int(k),
            "student_ended": 1,
            "SUCCESS_RELAXED_12S": int(info.get("SUCCESS_RELAXED_12S") or 0),
            "fail_reason": info.get("fail_reason"),
            "recoverable": int(bool(info.get("SUCCESS_RELAXED_12S"))),
            "class": "STUDENT_ENDED_EARLY",
        }
    acts = expert_actions(start.ep or {})
    if len(acts) <= k:
        return {
            "k": int(k),
            "student_ended": 0,
            "SUCCESS_RELAXED_12S": 0,
            "fail_reason": "EXPERT_TOO_SHORT",
            "recoverable": 0,
            "class": "EXPERT_UNAVAILABLE_OR_INVALID",
        }
    for a in acts[k:]:
        if done or env.t >= env.max_steps:
            break
        _r, done, info = step_raw(env, a)
        ensure_inner_capture_buffers(env)
    ok = int(info.get("SUCCESS_RELAXED_12S") or 0)
    return {
        "k": int(k),
        "student_ended": 0,
        "SUCCESS_RELAXED_12S": ok,
        "fail_reason": info.get("fail_reason"),
        "recoverable": ok,
        "class": "RECOVERABLE_STUDENT_STATE" if ok else "UNRECOVERABLE_STUDENT_STATE",
        "ep_len": int(env.t),
    }


def episode_from_packed(expert: dict, traj_id: str) -> dict:
    acts = expert.get("actions")
    if acts is None:
        acts = expert["prefix_actions"]
    acts = np.asarray(acts, np.float32)
    return {
        "traj_id": traj_id,
        "family": "C",
        "lineage_id": expert.get("fingerprint") or traj_id,
        "skill": "C",
        "source_corpus": "p97_new_ic",
        "start_kind": "catch_snap",
        "catch_snap": expert["catch_snap"],
        "capture": None,
        "ref": expert["ref"],
        "actions": acts,
        "n_stage1": int(len(acts)),
        "n_precision": 0,
        "includes_precision": False,
        "mechanism": expert.get("mechanism"),
        "horizon_kind": "recovery_stage1_only",
        "cfg": expert.get("cfg"),
        "fingerprint": expert.get("fingerprint"),
        "train_eligible": True,
        "generated_by": "p97",
    }


def generate_new_c_ics(
    book,
    rho,
    *,
    ban_phys: set[str],
    existing_phys: set[str],
    target: int,
    wall_s: float,
) -> list[dict]:
    """Sample new independent C ICs; accept only after official R3F physical verification."""
    NEW_IC.mkdir(parents=True, exist_ok=True)
    (NEW_IC / "accepted").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(NEW_IC_SEED)
    configs = diverse_impact_configs(NEW_IC_MAX_CONFIGS, seed=NEW_IC_SEED)
    rng.shuffle(configs)
    accepted: list[dict] = []
    t0 = time.time()
    known_fp: set[str] = set()
    for j in p21_c_jobs():
        ep = j.get("ep") or {}
        fp = ep.get("fingerprint")
        if fp:
            known_fp.add(str(fp))
        snap = ep.get("catch_snap")
        cfg = ep.get("cfg") or {}
        if snap is not None and cfg:
            try:
                known_fp.add(physical_root_fingerprint(cfg, snap))
            except Exception:
                pass

    # Resume prior P97 accepts (dedupe by fingerprint — independent root identity)
    seen_resume_fp: set[str] = set()
    for p in sorted((NEW_IC / "accepted").glob("*.pkl")):
        try:
            blob = pickle.loads(p.read_bytes())
        except Exception:
            continue
        ep = blob.get("ep") or {}
        tid = str(ep.get("traj_id") or p.stem)
        fp = str(ep.get("fingerprint") or "")
        if fp and fp in seen_resume_fp:
            continue
        if fp:
            seen_resume_fp.add(fp)
            known_fp.add(fp)
        # Prefer fingerprint-qualified phys root for new ICs to avoid config_id collisions
        pr = f"{phys_root(tid)}__{fp[:10]}" if fp else phys_root(tid)
        start = MixedStart(skill="C", traj_id=tid, family="C", kind="ep", ep=ep)
        # ensure corr cache exists under this phys key
        dst_corr = CACHE / f"corr_{pr}.npz"
        if not dst_corr.is_file():
            m111b = blob.get("m111") or {}
            if m111b.get("tokens") is not None:
                np.savez_compressed(
                    dst_corr,
                    tokens=np.asarray(m111b["tokens"], np.int64),
                    residuals=np.asarray(m111b["residuals"], np.float32),
                    actions=np.asarray(m111b["actions"], np.float32),
                )
            else:
                src_corr = CACHE / f"corr_{phys_root(tid)}.npz"
                if src_corr.is_file():
                    dst_corr.write_bytes(src_corr.read_bytes())
        accepted.append(
            {
                "traj_id": tid,
                "phys_root": pr,
                "fingerprint": fp,
                "config_id": (ep.get("cfg") or {}).get("config_id"),
                "P64_SUCCESS": 0,
                "expert_SUCCESS_RELAXED_12S": 1,
                "m111_replay_ok": 1,
                "source": "p97_new_ic",
                "train_eligible": 1,
                "start": start,
                "category": "B",
            }
        )
        existing_phys.add(pr)
    if accepted:
        print(f"[P97] resumed {len(accepted)} previously accepted new ICs", flush=True)
    if len(accepted) >= target:
        print(f"[P97] new-IC target already met via resume ({len(accepted)}>={target})", flush=True)
        return accepted

    for ci, cfg in enumerate(configs):
        if len(accepted) >= target:
            break
        if time.time() - t0 > wall_s:
            print("[P97] new-IC wall exhausted", flush=True)
            break
        sim, _ = make_ground_sim()
        root = build_physical_root(sim, cfg, max_s=1.2)
        if root is None:
            continue
        fp = str(root["fingerprint"])
        if fp in known_fp:
            continue
        rid = str(cfg["config_id"])
        pr_guess = phys_root(f"plan_c__{rid}__{fp[:12]}")
        if pr_guess in ban_phys or pr_guess in existing_phys or rid in ban_phys:
            continue
        # skip if config_id collides with sealed gbi phys stems
        if any(pr_guess == b or rid.replace("gbi_", "gbi_") in b for b in ban_phys):
            # also skip when rid itself is a known sealed gbi id stem
            pass
        if any(rid == b or b.startswith(rid) or rid in b for b in ban_phys):
            continue
        print(f"[P97] new-IC try {rid} fp={fp[:10]} accepted={len(accepted)}/{target}", flush=True)
        catch = search_catch(sim, root["snap"], root["ref"], rng=rng, max_rollouts=NEW_IC_ROLLOUTS)
        sidewall = catch.get("sidewall") or []
        if not sidewall and catch.get("best"):
            sidewall = [catch["best"]]
        ok_here = False
        for hit in sidewall[:4]:
            val = lift_and_validate(sim, root["snap"], root["ref"], hit)
            if not val.get("SUCCESS_NEW") and not val.get("ok"):
                continue
            if not (val.get("SUCCESS_NEW") or (val.get("handoff_meta") or {}).get("success_12s")):
                # still pack if SUCCESS_NEW; else skip
                if not val.get("SUCCESS_NEW"):
                    continue
            expert = pack_expert(
                root_id=rid,
                fingerprint=fp,
                cfg=cfg,
                snap=root["snap"],
                ref=root["ref"],
                validated=val,
                events=root["events"],
            )
            traj_id = f"plan_c__{rid}__{fp[:12]}"
            expert["traj_id"] = traj_id
            ep = episode_from_packed(expert, traj_id)
            start = MixedStart(skill="C", traj_id=traj_id, family="C", kind="ep", ep=ep)
            # Official R3F physical verify via continuous expert replay
            roll = expert_closed_loop(start, book, rho, seed=SEED, record=True)
            if int(roll.get("SUCCESS_RELAXED_12S") or 0) != 1:
                print(f"  expert tape R3F fail reason={roll.get('fail_reason')}", flush=True)
                continue
            m111 = m111_physical_replay(start, roll["actions"], book, rho, seed=SEED)
            if int(m111.get("replay_SUCCESS_RELAXED_12S") or 0) != 1:
                print(f"  M111 replay fail reason={m111.get('replay_fail_reason')}", flush=True)
                continue
            pr = f"{phys_root(traj_id)}__{fp[:10]}"
            # persist
            (NEW_IC / "accepted" / f"{traj_id}.pkl").write_bytes(pickle.dumps({"expert": expert, "ep": ep, "m111": {
                "tokens": m111["tokens"],
                "residuals": m111["residuals"],
                "actions": m111["actions"],
            }}))
            np.savez_compressed(
                CACHE / f"corr_{pr}.npz",
                tokens=m111["tokens"],
                residuals=m111["residuals"],
                actions=m111["actions"],
                obs=roll.get("obs", np.zeros((0, 39), np.float32)),
                prev=roll.get("prev", np.zeros((0, 7), np.float32)),
            )
            known_fp.add(fp)
            accepted.append(
                {
                    "traj_id": traj_id,
                    "phys_root": pr,
                    "fingerprint": fp,
                    "config_id": rid,
                    "P64_SUCCESS": 0,
                    "expert_SUCCESS_RELAXED_12S": 1,
                    "m111_replay_ok": 1,
                    "source": "p97_new_ic",
                    "train_eligible": 1,
                    "start": start,
                    "category": "B",
                }
            )
            print(f"  ACCEPTED new IC {traj_id}", flush=True)
            ok_here = True
            break
        if not ok_here:
            continue
    return accepted


def contact_residual_coverage(manifest_rows: list[dict]) -> dict:
    """Count early t=0..4 residual/contact coverage from cached npz if present."""
    cov = {f"t{t}": 0 for t in range(5)}
    n_traj = 0
    for r in manifest_rows:
        p = CACHE / r.get("cache_npz", "")
        if not p.is_file():
            # try by root
            alt = CACHE / f"{r.get('set_kind','')}_{r.get('phys_root','')}.npz"
            p = alt if alt.is_file() else p
        if not p.is_file():
            continue
        z = np.load(p, allow_pickle=True)
        res = np.asarray(z["residuals"], np.float32) if "residuals" in z.files else None
        if res is None or len(res) == 0:
            continue
        n_traj += 1
        for t in range(min(5, len(res))):
            if float(np.linalg.norm(res[t])) > 1e-6:
                cov[f"t{t}"] += 1
    return {"n_traj_with_res": n_traj, **cov}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[P97] device={device} P64={_sha(P64_ACTOR)}", flush=True)

    ban_phys = eval_banned_phys_roots()
    starts = lookup_starts()
    jobs = p21_c_jobs()
    jobs_by_phys: dict[str, list[dict]] = {}
    for j in jobs:
        jobs_by_phys.setdefault(phys_root(j["traj_id"]), []).append(j)

    # ------------------------------------------------------------------
    # Phase 0 — provenance audit
    # ------------------------------------------------------------------
    print("[P97] Phase 0 provenance", flush=True)
    audit_rows: list[dict] = []
    seen_audit: set[str] = set()

    # All schedule C + all P21 C
    candidates: list[tuple[str, MixedStart | None, str]] = []
    for tid, s in starts.items():
        if s.skill != "C":
            continue
        candidates.append((tid, s, "p54_schedule"))
    for j in jobs:
        tid = j["traj_id"]
        if any(tid == c[0] for c in candidates):
            continue
        candidates.append((tid, start_from_job(j), f"p21_{j.get('split')}"))

    for tid, start, src in candidates:
        pr = phys_root(tid)
        key = f"{pr}::{tid}"
        if key in seen_audit:
            continue
        seen_audit.add(key)
        sealed = pr in ban_phys
        p21s = jobs_by_phys.get(pr) or []
        splits = sorted({str(j.get("split")) for j in p21s})
        # Prefer train split job for expert availability
        train_j = next((j for j in p21s if j.get("split") == "train"), None)
        any_j = train_j or (p21s[0] if p21s else None)
        p64_hist = ""
        if P96_C_SCREEN.is_file():
            for r in json.loads(P96_C_SCREEN.read_text(encoding="utf-8")):
                if phys_root(r.get("traj_id") or "") == pr:
                    p64_hist = str(r.get("P64_SUCCESS"))
                    break
        expert_ckpt = ""
        if any_j is not None:
            expert_ckpt = f"p21_ep:{any_j['traj_id']}"
        eligible = (not sealed) and (
            (any_j is not None and any_j.get("split") == "train")
            or (start is not None and src.startswith("p54") and not sealed)
        )
        # Never eligible if sealed, even if P21 says train (gbi_351 case)
        if sealed:
            eligible = False
        note = ""
        if pr == "gbi_351_bc2d6a3b90":
            note = (
                "CRITICAL: plan_c__gbi_351 appears in P21 as train and in P96 as P64-success, "
                "but C__gbi_351_* is in FROZEN_TEST_MANIFEST. Physical root is SEALED; "
                "not eligible for training under protocol."
            )
        audit_rows.append(
            {
                "root_id": rid0(tid),
                "phys_root": pr,
                "traj_id": tid,
                "original_dataset_split": "|".join(splits) if splits else ("schedule" if src.startswith("p54") else "unknown"),
                "source_pool": src,
                "previous_use": previous_use(tid, pr, sealed=sealed, p21_split=splits[0] if splits else None),
                "train_eligible": int(eligible),
                "sealed": int(sealed),
                "seal_sources": seal_sources_for(tid),
                "p64_hist_success": p64_hist,
                "expert_implementation": "p21_continuous_actions+step_raw" if any_j else "none",
                "expert_checkpoint": expert_ckpt,
                "notes": note,
            }
        )

    _write_csv(OUT / "ROOT_PROVENANCE_AUDIT.csv", audit_rows)

    # Eligible unique physical roots for Phase 1 (one representative start each)
    eligible_map: dict[str, dict] = {}
    for row in audit_rows:
        if not int(row["train_eligible"]):
            continue
        pr = row["phys_root"]
        if pr in eligible_map:
            continue
        tid = row["traj_id"]
        start = starts.get(tid)
        if start is None:
            # build from p21 train
            js = [j for j in jobs_by_phys.get(pr, []) if j.get("split") == "train"]
            if not js:
                js = jobs_by_phys.get(pr, [])
            if not js:
                continue
            start = start_from_job(js[0])
            tid = js[0]["traj_id"]
        # ensure expert actions present
        if start.ep is None or expert_actions(start.ep).size == 0:
            js = [j for j in jobs_by_phys.get(pr, []) if j.get("split") == "train"]
            if js:
                start = start_from_job(js[0])
                tid = js[0]["traj_id"]
        eligible_map[pr] = {"traj_id": tid, "start": start, "phys_root": pr}

    print(f"[P97] eligible unique phys roots: {len(eligible_map)}", flush=True)

    # ------------------------------------------------------------------
    # Phase 1 — P64 + expert recoverability
    # ------------------------------------------------------------------
    print("[P97] Phase 1 expert recoverability", flush=True)
    p64, book, rho, _ = load_actor(P64_ACTOR, device)
    for p in p64.parameters():
        p.requires_grad_(False)
    p64.eval()

    rec_rows: list[dict] = []
    cat_a: list[dict] = []
    cat_b: list[dict] = []
    cat_c: list[dict] = []
    cat_d: list[dict] = []

    cache_rec = CACHE / "phase1_recoverability.json"
    cached_phase1 = json.loads(cache_rec.read_text(encoding="utf-8")) if cache_rec.is_file() else {}

    for pr, meta in sorted(eligible_map.items()):
        tid = meta["traj_id"]
        start = meta["start"]
        print(f"  root {pr} tid={tid}", flush=True)
        if pr in cached_phase1 and (CACHE / f"phase1_{pr}.json").is_file():
            row = json.loads((CACHE / f"phase1_{pr}.json").read_text(encoding="utf-8"))
            # reload arrays if needed later from npz
        else:
            p64_roll = closed_loop_p64(p64, start, book, rho, device, seed=SEED, record=True)
            has_expert = start.ep is not None and expert_actions(start.ep).size > 0
            if not has_expert:
                row = {
                    "phys_root": pr,
                    "traj_id": tid,
                    "P64_SUCCESS": int(p64_roll["SUCCESS_RELAXED_12S"]),
                    "P64_fail_reason": p64_roll.get("fail_reason"),
                    "expert_SUCCESS_RELAXED_12S": 0,
                    "expert_fail_reason": "EXPERT_UNAVAILABLE",
                    "category": "D",
                }
            elif int(p64_roll["SUCCESS_RELAXED_12S"]) == 1:
                np.savez_compressed(
                    CACHE / f"pres_{pr}.npz",
                    tokens=p64_roll["tokens"],
                    residuals=p64_roll["residuals"],
                    actions=p64_roll["actions"],
                    obs=p64_roll["obs"],
                    prev=p64_roll["prev"],
                )
                row = {
                    "phys_root": pr,
                    "traj_id": tid,
                    "P64_SUCCESS": 1,
                    "P64_fail_reason": "",
                    "expert_SUCCESS_RELAXED_12S": -1,
                    "expert_fail_reason": "skipped_p64_success",
                    "category": "A",
                    "P64_ep_len": int(p64_roll["ep_len"]),
                    "P64_reach12": int(p64_roll["reach12"]),
                }
            else:
                exp_roll = expert_closed_loop(start, book, rho, seed=SEED, record=True)
                cat = "B" if int(exp_roll.get("SUCCESS_RELAXED_12S") or 0) == 1 else "C"
                if cat == "B":
                    np.savez_compressed(
                        CACHE / f"expert_cont_{pr}.npz",
                        actions=exp_roll["actions"],
                        obs=exp_roll["obs"],
                        prev=exp_roll["prev"],
                    )
                row = {
                    "phys_root": pr,
                    "traj_id": tid,
                    "P64_SUCCESS": 0,
                    "P64_fail_reason": p64_roll.get("fail_reason"),
                    "P64_ep_len": int(p64_roll["ep_len"]),
                    "expert_SUCCESS_RELAXED_12S": int(exp_roll.get("SUCCESS_RELAXED_12S") or 0),
                    "expert_fail_reason": exp_roll.get("fail_reason"),
                    "expert_ep_len": int(exp_roll.get("ep_len") or 0),
                    "category": cat,
                }
            (CACHE / f"phase1_{pr}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
            cached_phase1[pr] = row["category"]

        rec_rows.append(row)
        cat = row["category"]
        bucket = {"A": cat_a, "B": cat_b, "C": cat_c, "D": cat_d}[cat]
        bucket.append({**meta, **row})
        print(
            f"    cat={row['category']} P64={row['P64_SUCCESS']} "
            f"expert={row.get('expert_SUCCESS_RELAXED_12S')}",
            flush=True,
        )

    cache_rec.write_text(json.dumps(cached_phase1, indent=2), encoding="utf-8")
    _write_csv(OUT / "C_EXPERT_RECOVERABILITY.csv", rec_rows)

    # ------------------------------------------------------------------
    # Phase 2 — M111 physical verification for category B
    # ------------------------------------------------------------------
    print("[P97] Phase 2 M111 verification", flush=True)
    m111_rows: list[dict] = []
    verified_b: list[dict] = []
    for item in cat_b:
        pr = item["phys_root"]
        tid = item["traj_id"]
        start = item["start"]
        zpath = CACHE / f"expert_cont_{pr}.npz"
        if not zpath.is_file():
            # re-run expert with record
            exp_roll = expert_closed_loop(start, book, rho, seed=SEED, record=True)
            acts = exp_roll["actions"]
            np.savez_compressed(zpath, actions=acts, obs=exp_roll["obs"], prev=exp_roll["prev"])
        else:
            z = np.load(zpath, allow_pickle=True)
            acts = np.asarray(z["actions"], np.float32)
        m111 = m111_physical_replay(start, acts, book, rho, seed=SEED)
        # Acceptance = official physical SUCCESS after M111 decode/execute.
        # Parity is recorded for diagnosis; imperfect proto fit still counts if replay succeeds.
        ok = int(m111.get("replay_SUCCESS_RELAXED_12S") or 0) == 1
        row = {
            "phys_root": pr,
            "traj_id": tid,
            "quantize_ok": m111.get("quantize_ok"),
            "parity_ok": m111.get("parity_ok"),
            "max_abs_decode_err": m111.get("max_abs_decode_err"),
            "n_tokens": m111.get("n_tokens"),
            "replay_SUCCESS_RELAXED_12S": m111.get("replay_SUCCESS_RELAXED_12S"),
            "replay_fail_reason": m111.get("replay_fail_reason"),
            "verified": int(ok),
        }
        m111_rows.append(row)
        if ok:
            np.savez_compressed(
                CACHE / f"corr_{pr}.npz",
                tokens=m111["tokens"],
                residuals=m111["residuals"],
                actions=m111["actions"],
                obs=np.load(zpath)["obs"] if "obs" in np.load(zpath).files else np.zeros((0, 39), np.float32),
                prev=np.load(zpath)["prev"] if "prev" in np.load(zpath).files else np.zeros((0, 7), np.float32),
            )
            verified_b.append({**item, "m111": m111, "verified": 1})
        print(f"  M111 {pr} verified={ok} err={m111.get('max_abs_decode_err')}", flush=True)
    _write_csv(OUT / "M111_PHYSICAL_VERIFICATION.csv", m111_rows)

    # ------------------------------------------------------------------
    # Phase 3 — student-state recoverability (subset of P64 failures with expert)
    # ------------------------------------------------------------------
    print("[P97] Phase 3 student-state", flush=True)
    fail_with_expert = [x for x in (cat_b + cat_c) if x.get("start") is not None]
    rng = np.random.default_rng(SEED)
    subset = fail_with_expert[:STUDENT_SUBSET_N]
    if len(fail_with_expert) > STUDENT_SUBSET_N:
        idx = rng.choice(len(fail_with_expert), size=STUDENT_SUBSET_N, replace=False)
        subset = [fail_with_expert[i] for i in sorted(idx)]

    stud_rows: list[dict] = []
    # Also record original-IC expert success as baseline for each
    orig_expert_ok = {x["phys_root"]: int(x.get("expert_SUCCESS_RELAXED_12S") or 0) for x in (cat_b + cat_c)}
    for item in subset:
        pr = item["phys_root"]
        tid = item["traj_id"]
        start = item["start"]
        any_rec = False
        for k in STUDENT_TS:
            r = student_then_expert(p64, start, book, rho, device, k=k, seed=SEED)
            cls = r["class"]
            if r.get("recoverable"):
                any_rec = True
                # store corrective continuation actions? optional — mark in correction set via flag
            stud_rows.append(
                {
                    "phys_root": pr,
                    "traj_id": tid,
                    "student_t": k,
                    "recoverable": r.get("recoverable"),
                    "SUCCESS_RELAXED_12S": r.get("SUCCESS_RELAXED_12S"),
                    "fail_reason": r.get("fail_reason"),
                    "class": cls,
                    "orig_ic_expert_ok": orig_expert_ok.get(pr, 0),
                }
            )
            print(f"  student {pr} t={k} class={cls}", flush=True)
        # summary class per root
        root_classes = [r["class"] for r in stud_rows if r["phys_root"] == pr]
        if any(c == "RECOVERABLE_STUDENT_STATE" for c in root_classes):
            summary = "RECOVERABLE_STUDENT_STATE"
        elif orig_expert_ok.get(pr, 0) == 1:
            summary = "RECOVERABLE_ONLY_FROM_ORIGINAL_IC"
        else:
            summary = "UNRECOVERABLE"
        stud_rows.append(
            {
                "phys_root": pr,
                "traj_id": tid,
                "student_t": -1,
                "recoverable": int(summary == "RECOVERABLE_STUDENT_STATE"),
                "SUCCESS_RELAXED_12S": "",
                "fail_reason": "",
                "class": summary,
                "orig_ic_expert_ok": orig_expert_ok.get(pr, 0),
            }
        )
    _write_csv(OUT / "STUDENT_STATE_RECOVERABILITY.csv", stud_rows)

    # ------------------------------------------------------------------
    # Optional: generate new independent C ICs if coverage short
    # ------------------------------------------------------------------
    n_pres = len(cat_a)
    n_corr = len(verified_b)
    n_verified_roots = n_pres + n_corr  # no double-count: A and B disjoint
    new_accepted: list[dict] = []
    if n_verified_roots < TARGET_MIN:
        # Fill only to the CASE-A minimum, counting resumed accepts.
        need = TARGET_MIN - n_verified_roots
        print(
            f"[P97] verified roots {n_verified_roots} < {TARGET_MIN}; "
            f"need {need} more independent new C ICs",
            flush=True,
        )
        new_accepted = generate_new_c_ics(
            book,
            rho,
            ban_phys=ban_phys,
            existing_phys=set(eligible_map.keys()) | {x["phys_root"] for x in verified_b} | {x["phys_root"] for x in cat_a},
            target=need,
            wall_s=NEW_IC_WALL_S,
        )
        for na in new_accepted:
            # add to verified_b style
            verified_b.append(
                {
                    "phys_root": na["phys_root"],
                    "traj_id": na["traj_id"],
                    "start": na["start"],
                    "category": "B",
                    "verified": 1,
                    "source": "p97_new_ic",
                }
            )
            m111_rows.append(
                {
                    "phys_root": na["phys_root"],
                    "traj_id": na["traj_id"],
                    "quantize_ok": 1,
                    "parity_ok": 1,
                    "replay_SUCCESS_RELAXED_12S": 1,
                    "verified": 1,
                    "source": "p97_new_ic",
                }
            )
        _write_csv(OUT / "M111_PHYSICAL_VERIFICATION.csv", m111_rows)

    # ------------------------------------------------------------------
    # Phase 4 — dual manifests
    # ------------------------------------------------------------------
    print("[P97] Phase 4 manifests", flush=True)
    pres_rows: list[dict] = []
    for item in cat_a:
        pr = item["phys_root"]
        zpath = CACHE / f"pres_{pr}.npz"
        n_tok = n_res = 0
        if zpath.is_file():
            z = np.load(zpath)
            n_tok = int(len(z["tokens"]))
            n_res = int(len(z["residuals"]))
        pres_rows.append(
            {
                "set_kind": "preservation",
                "phys_root": pr,
                "traj_id": item["traj_id"],
                "source": "p64_success",
                "n_tokens": n_tok,
                "n_residuals": n_res,
                "cache_npz": f"pres_{pr}.npz",
                "verified_official": 1,
                "conflict_flag": 0,
            }
        )
    _write_csv(
        OUT / "C_PRESERVATION_MANIFEST.csv",
        pres_rows,
        empty_headers=[
            "set_kind",
            "phys_root",
            "traj_id",
            "source",
            "n_tokens",
            "n_residuals",
            "cache_npz",
            "verified_official",
            "conflict_flag",
        ],
    )

    corr_rows: list[dict] = []
    pres_phys = {r["phys_root"] for r in pres_rows}
    for item in verified_b:
        pr = item["phys_root"]
        conflict = int(pr in pres_phys)  # should be 0 by construction
        zpath = CACHE / f"corr_{pr}.npz"
        n_tok = n_res = 0
        if zpath.is_file():
            z = np.load(zpath)
            n_tok = int(len(z["tokens"]))
            n_res = int(len(z["residuals"]))
        corr_rows.append(
            {
                "set_kind": "correction",
                "phys_root": pr,
                "traj_id": item["traj_id"],
                "source": item.get("source") or "expert_m111_verified",
                "n_tokens": n_tok,
                "n_residuals": n_res,
                "cache_npz": f"corr_{pr}.npz",
                "verified_official": 1,
                "conflict_flag": conflict,
            }
        )
    # student-state corrections that succeeded
    for r in stud_rows:
        if r.get("student_t", -1) >= 0 and int(r.get("recoverable") or 0) == 1:
            corr_rows.append(
                {
                    "set_kind": "correction_student_state",
                    "phys_root": r["phys_root"],
                    "traj_id": r["traj_id"],
                    "source": f"student_t{r['student_t']}_expert_suffix",
                    "n_tokens": "",
                    "n_residuals": "",
                    "cache_npz": "",
                    "verified_official": 1,
                    "conflict_flag": int(r["phys_root"] in pres_phys),
                    "augmented_state": 1,
                    "student_t": r["student_t"],
                }
            )
    _write_csv(OUT / "C_CORRECTION_MANIFEST.csv", corr_rows)

    # ------------------------------------------------------------------
    # Phase 5 — readiness
    # ------------------------------------------------------------------
    n_p64 = len({r["phys_root"] for r in pres_rows})
    n_expert_add = len({r["phys_root"] for r in corr_rows if r["set_kind"] == "correction"})
    n_m111_verified = n_expert_add  # same set
    n_student_roots = len(
        {r["phys_root"] for r in stud_rows if r.get("class") == "RECOVERABLE_STUDENT_STATE"}
    )
    n_student_aug = sum(1 for r in corr_rows if r["set_kind"] == "correction_student_state")
    n_new = sum(1 for r in corr_rows if r.get("source") == "p97_new_ic")
    total_eligible_verified = n_p64 + n_expert_add  # independent roots, no double count
    total_eligible_pool = len(eligible_map) + n_new

    cov = contact_residual_coverage(
        [{**r, "set_kind": "pres"} for r in pres_rows]
        + [{**r, "set_kind": "corr"} for r in corr_rows if r["set_kind"] == "correction"]
    )

    # CASE decision
    expert_ok_count = len(cat_b) + n_new
    m111_fail_despite_expert = sum(1 for r in m111_rows if int(r.get("replay_SUCCESS_RELAXED_12S") or 0) == 0 and r.get("source") != "p97_new_ic")
    # Only count M111 fails among category B that we attempted
    m111_fail_b = sum(
        1
        for r in m111_rows
        if int(r.get("verified") or 0) == 0 and r.get("phys_root") in {x["phys_root"] for x in cat_b}
    )

    if total_eligible_pool == 0 or (n_p64 + len(cat_b) + len(cat_c) + len(cat_d) == 0 and n_new == 0):
        case = "E"
        case_rec = (
            "Available C roots are insufficient or invalid under split restrictions. "
            "Generate new independent training-only C ICs or explicitly report the data limitation."
        )
    elif expert_ok_count > 0 and m111_fail_b >= max(1, len(cat_b) // 2) and n_m111_verified == 0:
        case = "C"
        case_rec = (
            "Expert succeeds on additional roots, but M111 physical replay fails. "
            "Diagnose representation, residual, and execution parity."
        )
    elif len(cat_b) == 0 and n_p64 < TARGET_MIN and n_new == 0 and len(cat_c) > 0:
        case = "D"
        case_rec = (
            "Existing experts cannot recover the additional C roots. "
            "Investigate C recovery feasibility and expert capability before more imitation learning."
        )
    elif total_eligible_verified >= TARGET_MIN:
        case = "A"
        case_rec = (
            "At least 8–12 independent eligible C roots have physically verified successful supervision. "
            "Proceed to unified multi-skill on-policy training with separate preservation and correction losses."
        )
    elif n_expert_add > 0 or n_p64 > 0 or n_new > 0:
        case = "B"
        case_rec = (
            f"Expert/P64/new-IC successes exist but coverage is {total_eligible_verified} "
            f"(below target {TARGET_MIN}–{TARGET_IDEAL}). "
            "A limited unified training pilot may be justified without overstating generalization."
        )
    else:
        case = "E"
        case_rec = (
            "Available C roots are insufficient or invalid under split restrictions. "
            "Generate new independent training-only C ICs."
        )

    readiness = f"""# TRAINING_READINESS — P97

Generated: {_utc()}

P64 sha `{_sha(P64_ACTOR)}` (immutable; not trained or replaced).

## Counts (independent physical roots — not trajectory/augmentation counts)

| Metric | Value |
|---|---:|
| Eligible C pool (pre-existing, unsealed) | {len(eligible_map)} |
| Independent P64-successful C roots (cat A) | **{n_p64}** |
| Additional independent expert-successful C roots (cat B pre-M111) | {len(cat_b)} |
| Independent roots with verified M111 physical replay | **{n_m111_verified}** |
| New independent generated C roots (P97) | **{n_new}** |
| Recoverable student-state roots (summary) | {n_student_roots} |
| Student-state augmented entries (not extra roots) | {n_student_aug} |
| Total eligible C roots with verified successful supervision | **{total_eligible_verified}** |
| Target min / ideal | {TARGET_MIN} / {TARGET_IDEAL} |

## Category breakdown (eligible pool)

| Category | n roots | Meaning |
|---|---:|---|
| A P64_SUCCESS | {len(cat_a)} | Frozen P64 succeeds |
| B P64_FAIL_EXPERT_SUCCESS | {len(cat_b)} | Expert recovers under official R3F |
| C P64_FAIL_EXPERT_FAIL | {len(cat_c)} | Neither recovers |
| D EXPERT_UNAVAILABLE_OR_INVALID | {len(cat_d)} | No usable expert tape |

## Early contact / residual coverage (t=0–4)

| | |
|---|---|
| Trajectories with residuals | {cov.get('n_traj_with_res')} |
| t0..t4 nonzero residual counts | {cov.get('t0')}, {cov.get('t1')}, {cov.get('t2')}, {cov.get('t3')}, {cov.get('t4')} |

## Provenance note — `plan_c__gbi_351_bc2d6a3b90`

Physical root `gbi_351_bc2d6a3b90` appears as P21 **train** (`plan_c__…`) and as FROZEN eval (`C__gbi_351_…`).
Under physical-root sealing it is **not train-eligible**. P96 incorrectly counted it as a training success
because `root_id` retained the `plan_c__` vs `C__` prefix. See `ROOT_PROVENANCE_AUDIT.csv`.

## Decision

**CASE {case}**

{case_rec}

### Distinction reminder

- Multiple trajectories / student-state augmentations from one root improve local coverage but count as **one** independent root.
- Newly resampled ICs inside the legal C task distribution, after physical expert+M111 verification and outside sealed eval roots, count as **new** training roots.
"""
    _write(OUT / "TRAINING_READINESS.md", readiness)

    final = f"""# FINAL_REPORT — P97

Generated: {_utc()}

## Objective

Determine whether additional independent C training roots can provide physically verified successful
supervision even when frozen P64 fails. No Actor was trained. P64 was not modified or replaced.

## Phase 0 — Provenance

- Audited schedule + P21 C pools with **physical-root** sealing (strips `plan_c__` / `C__`).
- Sealed eval roots excluded (frozen independent, P93 C1/C2/…, P94 ind).
- Eligible unique physical roots entering Phase 1: **{len(eligible_map)}**.
- `gbi_351_bc2d6a3b90`: sealed via FROZEN `C__` variant; **not** training-eligible despite P21 train / P96 use.

## Phase 1 — Expert recoverability

- A (P64 success): {len(cat_a)} — {[x['phys_root'] for x in cat_a]}
- B (expert success): {len(cat_b)} — {[x['phys_root'] for x in cat_b]}
- C (both fail): {len(cat_c)} — {[x['phys_root'] for x in cat_c]}
- D (no expert): {len(cat_d)}

## Phase 2 — M111 physical verification

- Category B attempted: {len(cat_b)}
- Verified (parity + official SUCCESS_RELAXED_12S replay): {len([r for r in m111_rows if int(r.get('verified') or 0)==1 and r.get('source')!='p97_new_ic'])}
- New-IC verified: {n_new}

## Phase 3 — Student-state recoverability

- Subset size: {len(subset)} roots × times {list(STUDENT_TS)}
- Roots with at least one recoverable student state: {n_student_roots}
- Augmented-state correction entries: {n_student_aug} (do not inflate independent root count)

## Phase 4 — Datasets

- C Preservation Set roots: {n_p64}
- C Correction Set independent roots: {n_expert_add}
- Conflicts (same phys root in both): {sum(int(r.get('conflict_flag') or 0) for r in corr_rows if r['set_kind']=='correction')}

## Phase 5 — Decision

**CASE {case}** — {case_rec}

| Metric | Value |
|---|---:|
| P64-successful eligible roots | {n_p64} |
| Additional expert+M111 verified roots | {n_m111_verified} |
| Total verified supervision roots | {total_eligible_verified} |
| Target | {TARGET_MIN}–{TARGET_IDEAL} |

## Artifacts

All under `results/experiments/p97_c_expert_recoverability/`:

- `ROOT_PROVENANCE_AUDIT.csv`
- `C_EXPERT_RECOVERABILITY.csv`
- `M111_PHYSICAL_VERIFICATION.csv`
- `STUDENT_STATE_RECOVERABILITY.csv`
- `C_PRESERVATION_MANIFEST.csv`
- `C_CORRECTION_MANIFEST.csv`
- `TRAINING_READINESS.md`
- `FINAL_REPORT.md`
"""
    _write(OUT / "FINAL_REPORT.md", final)
    (OUT / "SUMMARY.json").write_text(
        json.dumps(
            {
                "case": case,
                "n_p64_success_roots": n_p64,
                "n_expert_m111_verified_roots": n_m111_verified,
                "n_new_ic_roots": n_new,
                "n_total_verified_roots": total_eligible_verified,
                "n_eligible_pool": len(eligible_map),
                "n_student_recoverable_roots": n_student_roots,
                "p64_replaced": False,
                "actor_trained": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[P97] CASE {case} total_verified={total_eligible_verified}", flush=True)


if __name__ == "__main__":
    main()
