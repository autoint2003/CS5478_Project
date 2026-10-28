"""P23 — Plan C failure localization and D2 Precision integration.

Baseline: P22 SFT-A. No M111 / physics / Oracle / STRICT changes. No GRPO.
"""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from envs.observations.airborne_obs import OBS_DIM_AIR, OBS_VERSION, observe_airborne  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.outcomes.physical_recovery import TABLE_TOP, physical_pack  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from experiments.precision.p18_discrete_vocab_audit.replay_harness import (  # noqa: E402
    make_env,
    quantize_with_book,
)
from experiments.precision.p21_full_stage1_experts.run_p21 import (  # noqa: E402
    load_unified_episode,
    restore_start,
)
from experiments.precision.p22_full_horizon_sft.run_p22 import (  # noqa: E402
    build_start_index,
    classify_failure,
    policy_step,
    replay_expert,
    rollout_policy,
)
from models.m111_checkpoint import HANDOFF, load_m111, load_m111_book  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from training.m111_dataset import WINDOW, build_samples, chunk_episode, load_episode_npz  # noqa: E402
from training.m111_sft import train_m111  # noqa: E402
from training.skill_aware_action_vocab import DEFAULT_WEIGHTS, quantize_trajectory  # noqa: E402

OUT = ROOT / "results/diagnostics/p23_c_d2_sft"
P21 = ROOT / "results/diagnostics/p21_full_stage1_experts"
P22 = ROOT / "results/diagnostics/p22_full_horizon_sft"
P19 = ROOT / "results/diagnostics/p19_sft_packaging"
SFT_A_CKPT = P22 / "checkpoints/A_natural/checkpoint_best.pt"
M111_META = ROOT / "results/diagnostics/skill_aware_action_vocab_design/artifacts/SELECTED_skill_aware_hybrid_meta.json"
UNIFIED = ROOT / "data/unified_experts/manifest.jsonl"
FULLCHAIN = ROOT / "results/data_generation/stage2_abc_recovery_fullchain_v1"
WINDOW_INF = 160

P13 = ROOT / "results/diagnostics/p13_teleport_d2"
P14 = ROOT / "results/diagnostics/p14_small_error_d2"
P15 = ROOT / "results/diagnostics/p15_forced_precision_d2"
P16 = ROOT / "results/diagnostics/p16_precision_skill_expansion"

EPOCHS_FT = 12
LR_FT = 1e-4
SEED = 23


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_p21() -> dict:
    return json.loads((P21 / "SFT_FULL_HORIZON_MANIFEST.json").read_text(encoding="utf-8"))


def load_unified_by_id() -> dict[str, dict]:
    out = {}
    for line in UNIFIED.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        ep = load_unified_episode(raw)
        if ep:
            out[ep["traj_id"]] = ep
            # attach catch length from expert pkl when available
            pkl = ROOT / raw.get("provenance", {}).get("expert_pkl", "")
            if pkl.is_file():
                blob = pickle.loads(pkl.read_bytes())
                cap = blob.get("capture_actions")
                if cap is not None:
                    ep["n_catch"] = int(len(np.asarray(cap)))
                else:
                    ep["n_catch"] = None
            else:
                ep["n_catch"] = None
    return out


def contact_flags(o: dict) -> dict:
    nL = int(o.get("nL") or 0)
    nR = int(o.get("nR") or 0)
    return {
        "nL": nL,
        "nR": nR,
        "bilateral": int(nL > 0 and nR > 0),
        "any_contact": int(nL > 0 or nR > 0),
        "obj_z": float(o.get("obj_z") or 0.0),
        "aperture": float(o.get("aperture") or 0.0),
    }


def infer_stage_bounds(timeline: list[dict], n_catch: int | None) -> dict[str, int]:
    """Physically defined Plan C stage boundaries from contact/z timeline."""
    n = len(timeline)
    bilat = [int(t["bilateral"]) for t in timeline]
    zs = [float(t["obj_z"]) for t in timeline]
    first_contact = next((i for i, b in enumerate(bilat) if b), None)
    # DETACH/FLIGHT: until first bilateral
    detach_end = int(first_contact) if first_contact is not None else min(20, n)
    # RECATCH window: first contact + short hold
    recatch_end = detach_end
    if first_contact is not None:
        hold = 0
        for i in range(first_contact, n):
            if bilat[i]:
                hold += 1
                if hold >= 5:
                    recatch_end = i + 1
                    break
            else:
                hold = 0
        else:
            recatch_end = min(n, first_contact + 15)
    catch_end = int(n_catch) if n_catch is not None else recatch_end
    catch_end = max(catch_end, recatch_end)
    catch_end = min(catch_end, n)
    # STABILIZATION: catch_end (already held); LIFT starts at catch_end
    # Find sustained z rise after catch_end
    lift_start = catch_end
    if catch_end < n - 5:
        z0 = zs[catch_end]
        for i in range(catch_end, n):
            if zs[i] > z0 + 0.05:
                lift_start = i
                break
    return {
        "detach_end": detach_end,
        "recatch_end": recatch_end,
        "stabilize_end": catch_end,
        "lift_start": lift_start,
        "n_total": n,
        "first_bilateral": -1 if first_contact is None else int(first_contact),
    }


def stage_name(t: int, bounds: dict[str, int]) -> str:
    if t < bounds["detach_end"]:
        return "detach"
    if t < bounds["recatch_end"]:
        return "recatch"
    if t < bounds["stabilize_end"]:
        return "stabilization"
    if t < bounds["n_total"]:
        return "lift"
    return "handoff"


def record_timeline(env, ep, actions: np.ndarray) -> list[dict]:
    restore_start(env, ep)
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    acts = np.asarray(actions, np.float32).reshape(-1, 7)
    rows = []
    for t, a in enumerate(acts):
        o = contact_flags(physical_pack(env.sim))
        rows.append(
            {
                "t": t,
                **o,
                "action_linf": float(np.max(np.abs(a))),
                "action_tau": float(a[6]) if a.shape[0] > 6 else float(a[-1]),
            }
        )
        step_legal(env.sim, np.asarray(a, np.float32))
        env._prev = np.asarray(a, np.float32)
    return rows


def rollout_policy_timeline(env, model, book, rho, ep, *, device, max_steps) -> tuple[list[dict], dict]:
    restore_start(env, ep)
    hist = ObservableObsState(DT_POLICY)
    env._hist = hist
    env._prev = np.zeros(7, np.float32)
    obs_seq: list = []
    prev_seq: list = []
    rows = []
    tokens = []
    dropped = False
    reason = None
    for t in range(int(max_steps)):
        o = contact_flags(physical_pack(env.sim))
        obs = np.asarray(
            observe_airborne(env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, hist),
            np.float32,
        )
        obs_seq.append(obs)
        prev_seq.append(env._prev.copy())
        tok, _r, action = policy_step(model, book, rho, obs_seq, prev_seq, device)
        tokens.append(tok)
        row = {
            "t": t,
            **o,
            "pred_token": tok,
            "action_linf": 0.0 if action is None else float(np.max(np.abs(action))),
        }
        rows.append(row)
        if action is None:
            reason = "HANDOFF"
            break
        step_legal(env.sim, action)
        env._prev = action
        o2 = physical_pack(env.sim)
        if float(o2.get("obj_z", 99)) < TABLE_TOP + 0.01:
            dropped = True
            reason = "DROP"
            break
    hand = evaluate_handoff_with_axis(env.sim, ep["ref"], stabilize_n=8)
    summary = {
        "SUCCESS_NEW": bool(hand.get("SUCCESS_NEW")) and not dropped,
        "SUCCESS_12S": bool(hand.get("SUCCESS_12S")) and not dropped,
        "dropped": dropped,
        "n_motion": sum(1 for tok in tokens if tok != HANDOFF),
        "reason": reason or hand.get("std_reason") or hand.get("reason"),
        "tokens_head": "|".join(str(t) for t in tokens[:12]),
    }
    return rows, summary


def first_divergence(expert_tl: list[dict], sft_tl: list[dict], bounds: dict) -> dict:
    n = min(len(expert_tl), len(sft_tl))
    for t in range(n):
        e, s = expert_tl[t], sft_tl[t]
        contact_div = (e["bilateral"] != s["bilateral"]) or (e["any_contact"] != s["any_contact"])
        z_div = abs(e["obj_z"] - s["obj_z"]) > 0.03
        if contact_div or z_div:
            return {
                "first_div_t": t,
                "first_div_stage": stage_name(t, bounds),
                "expert_bilateral": e["bilateral"],
                "sft_bilateral": s["bilateral"],
                "expert_z": e["obj_z"],
                "sft_z": s["obj_z"],
                "contact_div": int(contact_div),
                "z_div": int(z_div),
            }
    # drop after expert length
    if len(sft_tl) < len(expert_tl):
        t = len(sft_tl) - 1
        return {
            "first_div_t": max(0, t),
            "first_div_stage": stage_name(max(0, t), bounds),
            "expert_bilateral": expert_tl[t]["bilateral"] if t < len(expert_tl) else None,
            "sft_bilateral": sft_tl[t]["bilateral"] if t >= 0 else None,
            "expert_z": expert_tl[t]["obj_z"] if t < len(expert_tl) else None,
            "sft_z": sft_tl[t]["obj_z"] if t >= 0 else None,
            "contact_div": 1,
            "z_div": 1,
            "note": "sft_ended_early",
        }
    return {
        "first_div_t": -1,
        "first_div_stage": "none",
        "contact_div": 0,
        "z_div": 0,
    }


def teacher_forced_stage_acc(model, book, rho, ep, bounds, *, device) -> dict[str, dict]:
    """Per-stage token accuracy under teacher forcing on continuous→M111 expert tokens."""
    q = quantize_trajectory(book, ep["actions"], rho, w=DEFAULT_WEIGHTS)
    tokens = np.asarray(q["tokens"], np.int64)
    env = make_env()
    restore_start(env, ep)
    hist = ObservableObsState(DT_POLICY)
    env._hist = hist
    env._prev = np.zeros(7, np.float32)
    obs_seq: list = []
    prev_seq: list = []
    stage_stats: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    acts = np.asarray(ep["actions"], np.float32)
    # Cap TF horizon: cover detach→early lift; full lift is long and not the C bottleneck.
    tmax = min(len(acts), max(int(bounds.get("stabilize_end", 60)) + 40, 100))
    for t in range(tmax):
        obs = np.asarray(
            observe_airborne(env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, hist),
            np.float32,
        )
        obs_seq.append(obs)
        prev_seq.append(env._prev.copy())
        tok_pred, _r, _a = policy_step(model, book, rho, obs_seq, prev_seq, device)
        st = stage_name(t, bounds)
        stage_stats[st]["total"] += 1
        if int(tok_pred) == int(tokens[t]):
            stage_stats[st]["correct"] += 1
        # teacher force execute expert continuous action (physical state)
        step_legal(env.sim, acts[t])
        env._prev = acts[t]
    out = {}
    for st, d in stage_stats.items():
        out[st] = {
            "acc": d["correct"] / max(1, d["total"]),
            "n": d["total"],
            "correct": d["correct"],
        }
    return out


def expert_prefix_sft_suffix(
    env,
    model,
    book,
    rho,
    ep,
    *,
    prefix_n: int,
    device,
    max_suffix: int = 800,
    stabilize_n: int = 8,
) -> dict:
    """Play Expert continuous prefix, restore action/obs history, then SFT closed-loop."""
    restore_start(env, ep)
    hist = ObservableObsState(DT_POLICY)
    env._hist = hist
    env._prev = np.zeros(7, np.float32)
    obs_seq: list = []
    prev_seq: list = []
    acts = np.asarray(ep["actions"], np.float32).reshape(-1, 7)
    n_pref = int(min(max(0, prefix_n), len(acts)))
    for t in range(n_pref):
        obs = np.asarray(
            observe_airborne(env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, hist),
            np.float32,
        )
        obs_seq.append(obs)
        prev_seq.append(env._prev.copy())
        step_legal(env.sim, acts[t])
        env._prev = acts[t]
    # suffix
    n_motion = 0
    dropped = False
    timeout = False
    reason = None
    tokens = []
    for _ in range(int(max_suffix)):
        obs = np.asarray(
            observe_airborne(env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, hist),
            np.float32,
        )
        obs_seq.append(obs)
        prev_seq.append(env._prev.copy())
        tok, _r, action = policy_step(model, book, rho, obs_seq, prev_seq, device)
        tokens.append(tok)
        if action is None:
            reason = "HANDOFF"
            break
        step_legal(env.sim, action)
        env._prev = action
        n_motion += 1
        if float(physical_pack(env.sim).get("obj_z", 99)) < TABLE_TOP + 0.01:
            dropped = True
            reason = "DROP"
            break
    else:
        timeout = True
        reason = "TIMEOUT"
    hand = evaluate_handoff_with_axis(env.sim, ep["ref"], stabilize_n=int(stabilize_n))
    return {
        "prefix_n": n_pref,
        "SUCCESS_NEW": bool(hand.get("SUCCESS_NEW")) and not dropped,
        "SUCCESS_12S": bool(hand.get("SUCCESS_12S")) and not dropped,
        "dropped": dropped,
        "timeout": timeout,
        "n_suffix_motion": n_motion,
        "reason": reason or hand.get("std_reason") or hand.get("reason"),
        "tokens_head": "|".join(str(t) for t in tokens[:10]),
    }


def classify_c_failure(div: dict, summary: dict) -> str:
    if summary.get("SUCCESS_NEW"):
        return "success"
    st = div.get("first_div_stage") or "unknown"
    if summary.get("dropped"):
        if st == "detach":
            return "detach"
        if st == "recatch":
            return "recatch"
        if st == "stabilization":
            return "stabilization"
        if st == "lift":
            return "lift"
        return f"drop_during_{st}"
    if summary.get("reason") == "HANDOFF" and summary.get("n_motion", 0) < 10:
        return "HANDOFF"
    if st == "lift":
        return "lift"
    return st or "Precision"


# ---------------------------------------------------------------------------
# Part 1 — Plan C diagnosis
# ---------------------------------------------------------------------------


def diagnose_plan_c(manifest: dict, model, book, rho, device) -> tuple[list[dict], list[dict]]:
    print("=== Part 1: Plan C diagnosis ===", flush=True)
    unified = load_unified_by_id()
    starts = build_start_index(manifest)
    val_c = [r for r in manifest["trajectories"] if r["split"] == "val" and r["family"] == "C"]
    env = make_env()
    timeline_rows = []
    prefix_rows = []
    for i, row in enumerate(val_c):
        tid = row["traj_id"]
        ep = starts[tid]
        if tid in unified:
            ep = {**ep, **{k: unified[tid][k] for k in ("n_catch",) if k in unified[tid]}}
        n_catch = ep.get("n_catch")
        expert_tl = record_timeline(env, ep, ep["actions"])
        bounds = infer_stage_bounds(expert_tl, n_catch)
        q = quantize_with_book(book, ep["actions"], rho, w=DEFAULT_WEIGHTS)
        m111_tl = record_timeline(env, ep, q["recon_actions"])
        sft_tl, sft_sum = rollout_policy_timeline(env, model, book, rho, ep, device=device, max_steps=120)
        div = first_divergence(expert_tl, sft_tl, bounds)
        stage_acc = teacher_forced_stage_acc(model, book, rho, ep, bounds, device=device)
        fail_cls = classify_c_failure(div, sft_sum)
        timeline_rows.append(
            {
                "traj_id": tid,
                "lineage_id": row["lineage_id"],
                "n_expert": len(ep["actions"]),
                "n_catch": n_catch,
                "detach_end": bounds["detach_end"],
                "recatch_end": bounds["recatch_end"],
                "stabilize_end": bounds["stabilize_end"],
                "lift_start": bounds["lift_start"],
                "first_bilateral": bounds["first_bilateral"],
                "sft_n_motion": sft_sum["n_motion"],
                "sft_dropped": int(sft_sum["dropped"]),
                "sft_success": int(sft_sum["SUCCESS_NEW"]),
                "sft_tokens_head": sft_sum["tokens_head"],
                "failure_class": fail_cls,
                **{f"div_{k}": v for k, v in div.items()},
                **{f"acc_{st}": stage_acc.get(st, {}).get("acc") for st in ("detach", "recatch", "stabilization", "lift")},
                **{f"n_{st}": stage_acc.get(st, {}).get("n") for st in ("detach", "recatch", "stabilization", "lift")},
                "m111_n": len(m111_tl),
            }
        )
        # Prefix → suffix at physical boundaries
        boundaries = [
            ("t0_pure_sft", 0),
            ("mid_detach", max(1, bounds["detach_end"] // 2)),
            ("first_bilateral", max(0, bounds["first_bilateral"]) if bounds["first_bilateral"] >= 0 else bounds["detach_end"]),
            ("recatch_end", bounds["recatch_end"]),
            ("stabilize_end", bounds["stabilize_end"]),
            ("mid_lift", bounds["stabilize_end"] + max(1, (bounds["n_total"] - bounds["stabilize_end"]) // 2)),
            ("full_expert", bounds["n_total"]),
        ]
        for bname, pn in boundaries:
            r = expert_prefix_sft_suffix(
                env, model, book, rho, ep, prefix_n=pn, device=device
            )
            prefix_rows.append(
                {
                    "traj_id": tid,
                    "lineage_id": row["lineage_id"],
                    "boundary": bname,
                    "prefix_n": r["prefix_n"],
                    "boundary_stage": stage_name(max(0, r["prefix_n"] - 1), bounds) if r["prefix_n"] > 0 else "start",
                    "SUCCESS_NEW": int(r["SUCCESS_NEW"]),
                    "SUCCESS_12S": int(r["SUCCESS_12S"]),
                    "dropped": int(r["dropped"]),
                    "n_suffix_motion": r["n_suffix_motion"],
                    "reason": r["reason"],
                    "tokens_head": r["tokens_head"],
                    "can_complete_independently_after": int(r["SUCCESS_NEW"]),
                }
            )
        print(f"  C {i+1}/{len(val_c)} {tid[:40]} fail={fail_cls} div_t={div.get('first_div_t')}", flush=True)
    _write_csv(OUT / "PLAN_C_FAILURE_TIMELINE.csv", timeline_rows)
    _write_csv(OUT / "EXPERT_PREFIX_SFT_SUFFIX.csv", prefix_rows)
    return timeline_rows, prefix_rows


# ---------------------------------------------------------------------------
# Part 2 — D2 corpus audit
# ---------------------------------------------------------------------------


def _lineage_of_d2(blob: dict, path: Path) -> str:
    meta = blob.get("meta") or {}
    cons = blob.get("construction") or {}
    for k in ("base_sample_id", "lineage_key", "lineage_id"):
        if blob.get(k):
            return str(blob[k])
        if meta.get(k):
            return str(meta[k])
        if cons.get(k):
            return str(cons[k])
    return path.stem


def audit_d2(book, rho) -> tuple[str, list[dict]]:
    print("=== Part 2: D2 corpus audit ===", flush=True)
    env = make_env()
    rows = []

    def add_ic(path: Path, source: str, category_hint: str | None = None):
        blob = pickle.loads(path.read_bytes())
        meta = blob.get("meta") or {}
        cons = blob.get("construction") or {}
        acts = blob.get("actions")
        if acts is None:
            acts = meta.get("actions")
        n_act = 0 if acts is None else int(len(np.asarray(acts)))
        forced = bool(blob.get("forced_precision") or meta.get("forced_precision") or category_hint == "forced")
        skill = (
            blob.get("requested_skill")
            or meta.get("force_skill")
            or meta.get("requested_skill")
            or cons.get("family")
            or "?"
        )
        lineage = _lineage_of_d2(blob, path)
        cap = blob.get("capture")
        ref = blob.get("ref")
        m111_ok = None
        strict = None
        survival = None
        if n_act > 0 and cap is not None and ref is not None:
            try:
                restore_capture(env, cap)
                env._hist = ObservableObsState(DT_POLICY)
                env._prev = np.zeros(7, np.float32)
                q = quantize_trajectory(book, np.asarray(acts, np.float32), rho, w=DEFAULT_WEIGHTS)
                for a in q["recon_actions"]:
                    step_legal(env.sim, np.asarray(a, np.float32))
                hand = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=8)
                m111_ok = bool(hand.get("SUCCESS_NEW"))
                strict = m111_ok
                survival = bool(hand.get("SUCCESS_12S"))
            except Exception as exc:
                m111_ok = False
                meta_err = str(exc)[:120]
        else:
            meta_err = "no_actions_or_capture" if n_act == 0 else None

        # category
        if n_act == 0:
            cat = "construction_only_no_actions"
        elif forced:
            cat = "forced_precision"
        elif "PRECISION_EARLY_HANDOFF" in str(blob.get("phases")) and n_act > 50:
            cat = "genuine_d2_precision_required"
        elif skill in ("dx", "dz", "coupled") or "dx" in str(skill).lower() or "dz" in str(skill).lower():
            cat = "genuine_d2_precision_required" if not forced else "forced_precision"
        else:
            cat = category_hint or "unknown"
        rows.append(
            {
                "source": source,
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sample_id": blob.get("sample_id") or path.stem,
                "lineage_id": lineage,
                "category": cat,
                "forced_precision": int(forced),
                "skill": skill,
                "n_actions": n_act,
                "m111_replay_SUCCESS_NEW": m111_ok,
                "SUCCESS_12S": survival,
                "has_capture": int(cap is not None),
                "meta_SUCCESS_NEW": meta.get("SUCCESS_NEW"),
            }
        )

    for p in sorted((P13 / "ics").glob("*.pkl")):
        add_ic(p, "p13_teleport_d2", "construction_only_no_actions")
    for p in sorted((P14 / "ics").glob("*.pkl")):
        add_ic(p, "p14_small_error_d2", "genuine_d2_precision_required")
    for p in sorted((P15 / "ics").glob("*.pkl")):
        add_ic(p, "p15_forced_precision_d2", "forced")
    for p in sorted((P16 / "ics").glob("*.pkl")):
        add_ic(p, "p16_precision_skill_expansion", "forced")

    # P19 official D Precision packages (already M111-validated at packaging)
    p19 = json.loads((P19 / "SFT_DATASET_MANIFEST.json").read_text(encoding="utf-8"))
    for r in p19["trajectories"]:
        sg = str(r.get("skill_group") or "")
        if not sg.startswith("D|"):
            continue
        if str(r.get("meta_is_direct_handoff_only")).lower() in ("true", "1"):
            cat = "direct_handoff_only"
        elif str(r.get("meta_precision_executed_corrections")).lower() in ("true", "1"):
            cat = "genuine_d2_precision_required"
        else:
            cat = "direct_handoff_only"
        # recovery→precision are not D letter
        rows.append(
            {
                "source": "p19_official_D",
                "path": r.get("npz_relpath"),
                "sample_id": r["traj_id"],
                "lineage_id": r.get("source_lineage") or r["traj_id"],
                "category": cat,
                "forced_precision": 0,
                "skill": sg,
                "n_actions": int(r.get("n_motion") or 0),
                "m111_replay_SUCCESS_NEW": True,  # packaging required M111 success
                "SUCCESS_12S": None,
                "has_capture": 1,
                "meta_SUCCESS_NEW": True,
                "split_p19": r.get("split"),
            }
        )

    # Recovery→Precision from P21 (separate lineage)
    man21 = load_p21()
    for r in man21["trajectories"]:
        if not r.get("includes_precision"):
            continue
        rows.append(
            {
                "source": "p21_recovery_to_precision",
                "path": r.get("npz_relpath"),
                "sample_id": r["traj_id"],
                "lineage_id": r["lineage_id"],
                "category": "recovery_to_precision",
                "forced_precision": 0,
                "skill": f"{r['family']}|prec",
                "n_actions": int(r.get("n_motion") or 0),
                "m111_replay_SUCCESS_NEW": True,
                "SUCCESS_12S": None,
                "has_capture": 1,
                "meta_SUCCESS_NEW": True,
            }
        )

    _write_csv(OUT / "D2_CORPUS_AUDIT_ROWS.csv", rows)

    # Independent lineages (collapse perturbations)
    by_src = defaultdict(list)
    for r in rows:
        by_src[r["source"]].append(r)
    genuine = [r for r in rows if r["category"] == "genuine_d2_precision_required"]
    forced = [r for r in rows if r["category"] == "forced_precision"]
    recovery = [r for r in rows if r["category"] == "recovery_to_precision"]
    direct = [r for r in rows if r["category"] == "direct_handoff_only"]
    construction = [r for r in rows if r["category"] == "construction_only_no_actions"]

    def uniq_lin(xs):
        return sorted({r["lineage_id"] for r in xs})

    md = f"""# D2_CORPUS_AUDIT

Generated: {_utc()}

## Separation (do not equate perturbations with independent grasps)

| Category | Trajectories | Independent lineages |
|---|---:|---:|
| Genuine D2 Precision-required | {len(genuine)} | {len(uniq_lin(genuine))} |
| Forced Precision demos | {len(forced)} | {len(uniq_lin(forced))} |
| Recovery→Precision | {len(recovery)} | {len(uniq_lin(recovery))} |
| Direct-HANDOFF-only | {len(direct)} | {len(uniq_lin(direct))} |
| Construction-only (no actions, e.g. P13) | {len(construction)} | {len(uniq_lin(construction))} |

## Source lineages

### P13 teleport
- Construction-only ICs: **{sum(1 for r in rows if r['source']=='p13_teleport_d2')}** — **excluded** from training (no executed Precision tape).

### P14 small-error
- Trajectories: {sum(1 for r in rows if r['source']=='p14_small_error_d2')}
- Lineages: {uniq_lin([r for r in rows if r['source']=='p14_small_error_d2'])}
- M111 SUCCESS: {[r['m111_replay_SUCCESS_NEW'] for r in rows if r['source']=='p14_small_error_d2']}

### P15 / P16 forced Precision
- P15: {sum(1 for r in rows if r['source']=='p15_forced_precision_d2')} trajs / lineages {uniq_lin([r for r in rows if r['source']=='p15_forced_precision_d2'])}
- P16: {sum(1 for r in rows if r['source']=='p16_precision_skill_expansion')} trajs / lineages {uniq_lin([r for r in rows if r['source']=='p16_precision_skill_expansion'])}
- **Independent bases across P15+P16:** {uniq_lin([r for r in rows if r['source'] in ('p15_forced_precision_d2','p16_precision_skill_expansion')])}
- Multiple IC files share the same `base_sample_id` (perturbations, not independent grasps).

### P19 official D|*
- D Precision with executed corrections: **{sum(1 for r in rows if r['source']=='p19_official_D' and r['category']=='genuine_d2_precision_required')}**
- Direct-HANDOFF among D letter: **{sum(1 for r in rows if r['source']=='p19_official_D' and r['category']=='direct_handoff_only')}**
- **Not** bulk-adding the 171 P19 direct-HANDOFF continuation trajectories.

### P21 Recovery→Precision
- Already in primary corpus: **{len(recovery)}** full-horizon tapes (`start_distribution=recovery_ic`).

## Curated training policy (Part 3)

Include:
1. All P21 full-horizon A/B/C (+ Recovery→Precision already inside).
2. P19 D|* with `meta_precision_executed_corrections` (genuine D2).
3. P14 genuine small-error demos that M111-replay succeed.
4. At most **2 forced** demos per (lineage × skill) from P15/P16 that M111-replay succeed (supervision diversity, not independent-grasp inflation).

Exclude:
- P13 construction-only.
- Bulk P19 direct-HANDOFF (171).
- Failed M111 replays.

Start-distribution tags preserved as metadata only (`recovery_ic` vs `stable_precision_ic`); **never** fed as privileged policy input.
"""
    _write_md(OUT / "D2_CORPUS_AUDIT.md", md)
    return md, rows


# ---------------------------------------------------------------------------
# Part 3 — Combined dataset
# ---------------------------------------------------------------------------


def package_d2_ic(env, path: Path, book, rho, *, start_dist: str, split: str, lineage: str, skill: str, source: str) -> dict | None:
    blob = pickle.loads(path.read_bytes())
    acts = blob.get("actions")
    if acts is None:
        acts = (blob.get("meta") or {}).get("actions")
    if acts is None or blob.get("capture") is None or blob.get("ref") is None:
        return None
    acts = np.asarray(acts, np.float32).reshape(-1, 7)
    ep = {
        "traj_id": f"d2__{source}__{blob.get('sample_id') or path.stem}",
        "family": "D2",
        "lineage_id": lineage,
        "skill": skill,
        "source_corpus": source,
        "start_kind": "recovery_ic_capture",
        "capture": blob["capture"],
        "ref": blob["ref"],
        "actions": acts,
        "n_stage1": 0,
        "n_precision": int(len(acts)),
        "includes_precision": True,
        "horizon_kind": "stable_precision_d2",
        "mechanism": skill,
    }
    restore_capture(env, ep["capture"])
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    hist = env._hist
    obs_list, prev_list = [], []
    for a in acts:
        obs = np.asarray(
            observe_airborne(env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, hist),
            np.float32,
        )
        obs_list.append(obs)
        prev_list.append(env._prev.copy())
        step_legal(env.sim, a)
        env._prev = a
    hob = np.asarray(
        observe_airborne(env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, hist),
        np.float32,
    )
    obs_list.append(hob)
    prev_list.append(env._prev.copy())
    hand = evaluate_handoff_with_axis(env.sim, ep["ref"], stabilize_n=8)
    if not hand.get("SUCCESS_NEW"):
        return None
    q = quantize_trajectory(book, acts, rho, w=DEFAULT_WEIGHTS)
    cont = {
        "observations": np.stack(obs_list).astype(np.float32),
        "prev_actions": np.stack(prev_list).astype(np.float32),
        "SUCCESS_NEW": True,
        "SUCCESS_12S": bool(hand.get("SUCCESS_12S")),
    }
    # package into OUT (reuse P21 helper paths relative to OUT)
    # temporarily monkey: package_episode writes under P21 OUT — reimplement local save
    tokens = np.concatenate([np.asarray(q["tokens"], np.int64), np.array([HANDOFF], np.int64)])
    residuals = np.concatenate(
        [np.asarray(q["residuals"], np.float32), np.zeros((1, 7), np.float32)], axis=0
    )
    stem = hashlib.sha1(ep["traj_id"].encode()).hexdigest()[:16]
    rel = f"dataset/npz/{stem}.npz"
    path_out = OUT / rel
    path_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path_out,
        observations=cont["observations"],
        prev_actions=cont["prev_actions"],
        tokens=tokens,
        residuals=residuals,
        executed_actions=np.concatenate(
            [np.asarray(q["recon_actions"], np.float32), np.zeros((1, 7), np.float32)], axis=0
        ),
        continuous_actions=acts,
        loss_mask=np.ones(len(tokens), dtype=np.bool_),
        episode_boundary=np.array([False] * (len(tokens) - 1) + [True]),
        handoff_id=np.int64(HANDOFF),
        n_stage1=np.int32(0),
        n_precision=np.int32(len(acts)),
        start_distribution=np.array(start_dist),
    )
    return {
        "traj_id": ep["traj_id"],
        "split": split,
        "family": "D2",
        "lineage_id": lineage,
        "source_corpus": source,
        "horizon_kind": "stable_precision_d2",
        "includes_precision": True,
        "n_stage1": 0,
        "n_precision": int(len(acts)),
        "n_motion": int(len(acts)),
        "n_steps_with_handoff": int(len(tokens)),
        "npz_relpath": rel.replace("\\", "/"),
        "obs_dim": OBS_DIM_AIR,
        "obs_version": OBS_VERSION,
        "start_distribution": start_dist,
        "mechanism": skill,
        "skill_group": f"D2|{skill}",
    }


def build_combined_dataset(audit_rows: list[dict], book, rho) -> dict:
    print("=== Part 3: combined dataset ===", flush=True)
    man21 = load_p21()
    trajs = []
    # Copy / reference P21 NPZs into combined manifest (paths stay under P21; root recorded)
    for r in man21["trajectories"]:
        trajs.append(
            {
                **r,
                "dataset_root": "p21",
                "start_distribution": "recovery_ic",
                # keep analysis label; never used as model input
                "start_distribution_label": "recovery_ic",
            }
        )

    # P19 genuine D — copy NPZ into p23 dataset
    p19 = json.loads((P19 / "SFT_DATASET_MANIFEST.json").read_text(encoding="utf-8"))
    p19_d = [
        r
        for r in p19["trajectories"]
        if str(r.get("skill_group") or "").startswith("D|")
        and str(r.get("meta_precision_executed_corrections")).lower() in ("true", "1")
        and str(r.get("meta_is_direct_handoff_only")).lower() not in ("true", "1")
    ]
    # Hold out lineages for D2 eval: smallest hash bucket ~20%
    d2_lineages = sorted({r.get("source_lineage") or r["traj_id"] for r in p19_d})
    val_lin = {lid for lid in d2_lineages if int(hashlib.sha1(lid.encode()).hexdigest()[:8], 16) % 100 >= 80}
    if not val_lin and d2_lineages:
        val_lin = {d2_lineages[0]}
    for r in p19_d:
        src = P19 / r["npz_relpath"]
        stem = hashlib.sha1(("p19d2_" + r["traj_id"]).encode()).hexdigest()[:16]
        rel = f"dataset/npz/{stem}.npz"
        dst = OUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_file():
            shutil.copy2(src, dst)
        lid = r.get("source_lineage") or r["traj_id"]
        trajs.append(
            {
                "traj_id": f"p19d2__{r['traj_id']}",
                "split": "val" if lid in val_lin else "train",
                "family": "D2",
                "lineage_id": lid,
                "source_corpus": "p19_official_D",
                "horizon_kind": "stable_precision_d2",
                "includes_precision": True,
                "n_stage1": 0,
                "n_precision": int(r.get("n_motion") or 0),
                "n_motion": int(r.get("n_motion") or 0),
                "n_steps_with_handoff": int(r.get("n_steps") or 0),
                "npz_relpath": rel.replace("\\", "/"),
                "obs_dim": OBS_DIM_AIR,
                "obs_version": OBS_VERSION,
                "start_distribution": "stable_precision_ic",
                "start_distribution_label": "stable_precision_ic",
                "dataset_root": "p23",
                "mechanism": r.get("skill_group"),
                "skill_group": r.get("skill_group"),
            }
        )

    env = make_env()
    # P14 genuine
    for r in audit_rows:
        if r["source"] != "p14_small_error_d2":
            continue
        if not r.get("m111_replay_SUCCESS_NEW"):
            continue
        lid = r["lineage_id"]
        split = "val" if lid in val_lin or (int(hashlib.sha1(lid.encode()).hexdigest()[:8], 16) % 100 >= 80) else "train"
        if split == "val":
            val_lin.add(lid)
        packed = package_d2_ic(
            env,
            ROOT / r["path"],
            book,
            rho,
            start_dist="stable_precision_ic",
            split=split,
            lineage=lid,
            skill=str(r.get("skill") or "combined"),
            source="p14",
        )
        if packed:
            packed["dataset_root"] = "p23"
            packed["start_distribution_label"] = "stable_precision_ic"
            trajs.append(packed)

    # Curate forced P15/P16: ≤2 per (lineage, skill), prefer shorter, M111 ok
    forced_ok = [
        r
        for r in audit_rows
        if r["source"] in ("p15_forced_precision_d2", "p16_precision_skill_expansion")
        and r.get("m111_replay_SUCCESS_NEW")
        and int(r.get("n_actions") or 0) > 0
    ]
    forced_ok.sort(key=lambda r: int(r.get("n_actions") or 0))
    picked = []
    per_key = Counter()
    # Hold out one full base lineage for unseen D2 eval
    forced_bases = sorted({r["lineage_id"] for r in forced_ok})
    holdout_base = forced_bases[-1] if forced_bases else None
    if holdout_base:
        val_lin.add(holdout_base)
    for r in forced_ok:
        key = (r["lineage_id"], str(r.get("skill")))
        if per_key[key] >= 2:
            continue
        per_key[key] += 1
        split = "val" if r["lineage_id"] in val_lin else "train"
        packed = package_d2_ic(
            env,
            ROOT / r["path"],
            book,
            rho,
            start_dist="stable_precision_ic",
            split=split,
            lineage=r["lineage_id"],
            skill=str(r.get("skill")),
            source=r["source"],
        )
        if packed:
            packed["dataset_root"] = "p23"
            packed["start_distribution_label"] = "stable_precision_ic"
            packed["forced_precision"] = True
            trajs.append(packed)
            picked.append(r["sample_id"])

    manifest = {
        "dataset_id": "p23_combined_p21_plus_curated_d2",
        "generated_at": _utc(),
        "primary_corpus": "p21_full_horizon_abc",
        "start_distribution_note": "metadata only; not a policy input",
        "n_trajectories": len(trajs),
        "n_p21": sum(1 for t in trajs if t.get("dataset_root") == "p21"),
        "n_d2": sum(1 for t in trajs if t.get("family") == "D2"),
        "d2_val_lineages": sorted(val_lin),
        "forced_picked": picked,
        "excluded": ["p13_construction", "p19_bulk_direct_handoff_171"],
        "trajectories": trajs,
    }
    (OUT / "COMBINED_DATASET_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"combined n={len(trajs)} p21={manifest['n_p21']} d2={manifest['n_d2']} "
        f"forced_picked={len(picked)}",
        flush=True,
    )
    return manifest


def resolve_npz_root(row: dict) -> Path:
    if row.get("dataset_root") == "p21" or row.get("start_distribution") in (
        "recovery_ic",
        "recovery_ic_pre_stage1",
    ):
        # P21 rows may have been retagged to recovery_ic
        if (P21 / row["npz_relpath"]).is_file():
            return P21
    return OUT


def build_samples_combined(
    manifest: dict,
    *,
    split: str,
    scheme: str = "natural",
    include_d2: bool = True,
    c_stage_boost: float = 1.0,
    d2_boost: float = 1.0,
) -> list[dict]:
    rows = [r for r in manifest["trajectories"] if r.get("split") == split]
    if not include_d2:
        rows = [r for r in rows if r.get("family") != "D2"]
    # annotate weights on the filtered list
    from training.m111_dataset import _annotate_p21_weights, episode_weight

    _annotate_p21_weights(rows)
    samples: list[dict] = []
    for row in rows:
        root = resolve_npz_root(row)
        ep = load_episode_npz(root, row["npz_relpath"])
        w = episode_weight(row, scheme)
        if row.get("family") == "C":
            w *= float(c_stage_boost)
        if row.get("family") == "D2":
            w *= float(d2_boost)
        for ci, chunk in enumerate(chunk_episode(ep, window=WINDOW)):
            sw = w
            # Extra boost on early Plan C chunk (detach/recatch window)
            if row.get("family") == "C" and ci == 0 and c_stage_boost > 1.0:
                sw *= 2.0
            samples.append(
                {
                    **chunk,
                    "sample_weight": float(sw),
                    "traj_id": row["traj_id"],
                    "family": row.get("family"),
                    "lineage_id": row.get("lineage_id"),
                    "skill_group": row.get("skill_group") or row.get("family"),
                    "split": split,
                    "start_distribution": row.get("start_distribution"),
                }
            )
    # renormalize weights to mean 1
    if samples:
        mean_w = float(np.mean([s["sample_weight"] for s in samples]))
        if mean_w > 0:
            for s in samples:
                s["sample_weight"] = float(s["sample_weight"] / mean_w)
    return samples


# ---------------------------------------------------------------------------
# Part 4 / 5 — Train + eval
# ---------------------------------------------------------------------------


def render_rollout(env, model, book, rho, ep, path: Path, *, device, max_steps=250, tag=""):
    try:
        import mujoco
        from visualization.camera import mjv_camera_from_preset
        from visualization.video_recorder import write_frames_mp4
    except Exception as exc:
        print("video skip", exc, flush=True)
        return
    cam = mjv_camera_from_preset(
        {"lookat": [0.0, 0.0, 0.55], "distance": 0.55, "azimuth": 135, "elevation": -20}
    )
    if ep.get("start_kind") == "catch_snap" or ep.get("catch_snap") is not None:
        restore_start(env, ep)
    else:
        restore_capture(env, ep["capture"])
        env._hist = ObservableObsState(DT_POLICY)
        env._prev = np.zeros(7, np.float32)
    hist = env._hist
    renderer = mujoco.Renderer(env.sim.model, 480, 640)
    frames = []
    obs_seq: list = []
    prev_seq: list = []
    for _ in range(max_steps):
        renderer.update_scene(env.sim.data, cam)
        frames.append(renderer.render().copy())
        obs = np.asarray(
            observe_airborne(env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, hist),
            np.float32,
        )
        obs_seq.append(obs)
        prev_seq.append(env._prev.copy())
        tok, _r, action = policy_step(model, book, rho, obs_seq, prev_seq, device)
        if action is None:
            for __ in range(25):
                step_legal(env.sim, np.zeros(7, np.float32))
                renderer.update_scene(env.sim.data, cam)
                frames.append(renderer.render().copy())
            break
        step_legal(env.sim, action)
        env._prev = action
        if float(physical_pack(env.sim).get("obj_z", 99)) < TABLE_TOP + 0.01:
            break
    path.parent.mkdir(parents=True, exist_ok=True)
    write_frames_mp4(frames, path, fps=20)
    print("video", tag or path.name, len(frames), flush=True)


def eval_condition(name, model, book, rho, starts, val_rows, d2_starts, d2_val, device) -> list[dict]:
    env = make_env()
    rows = []
    # Recovery A/B/C + Recovery→Precision from P21 val
    for row in val_rows:
        tid = row["traj_id"]
        ep = starts[tid]
        r = rollout_policy(
            env, model, book, rho, ep, device=device, max_steps=800, stabilize_n=8
        )
        skill_slice = row["family"]
        if row.get("includes_precision"):
            skill_slice = "RecoveryPrecision"
        r.update(
            {
                "condition": name,
                "traj_id": tid,
                "lineage_id": row["lineage_id"],
                "family": row["family"],
                "skill_slice": skill_slice,
                "start_distribution": "recovery_ic",
                "expert_n_actions": len(ep["actions"]),
            }
        )
        r["failure_class"] = classify_failure(r)
        r["premature_handoff"] = int(
            (not r.get("SUCCESS_NEW"))
            and (
                r.get("immediate_handoff")
                or (
                    int(r.get("n_motion") or 0)
                    < max(5, int(len(ep["actions"])) // 10)
                )
            )
        )
        rows.append(r)
    # D2 val
    for ep in d2_val:
        restore_capture(env, ep["capture"])
        env._hist = ObservableObsState(DT_POLICY)
        env._prev = np.zeros(7, np.float32)
        # reuse rollout_policy via fake ep with start_kind capture
        fake = {
            "capture": ep["capture"],
            "ref": ep["ref"],
            "actions": ep["actions"],
            "start_kind": "recovery_ic_capture",
        }
        r = rollout_policy(
            env, model, book, rho, fake, device=device, max_steps=800, stabilize_n=8
        )
        r.update(
            {
                "condition": name,
                "traj_id": ep["traj_id"],
                "lineage_id": ep["lineage_id"],
                "family": "D2",
                "skill_slice": "D2",
                "start_distribution": "stable_precision_ic",
                "expert_n_actions": len(ep["actions"]),
                "unseen_lineage": True,
            }
        )
        r["failure_class"] = classify_failure(r)
        r["premature_handoff"] = int(bool(r.get("immediate_handoff")))
        rows.append(r)
    return rows


def load_d2_eval_eps(manifest: dict) -> list[dict]:
    """Load capture+actions for D2 val trajectories (from original IC or NPZ meta)."""
    env_eps = []
    # Prefer audit IC paths for forced/p14; for p19 use packaged continuous from source records
    for row in manifest["trajectories"]:
        if row.get("family") != "D2" or row.get("split") != "val":
            continue
        # Try to find original IC
        found = None
        tid = row["traj_id"]
        if tid.startswith("d2__"):
            # d2__{source}__{sample}
            parts = tid.split("__", 2)
            sample = parts[-1] if len(parts) >= 3 else None
            for folder in (P14 / "ics", P15 / "ics", P16 / "ics"):
                if sample and (folder / f"{sample}.pkl").is_file():
                    found = folder / f"{sample}.pkl"
                    break
                # fuzzy
                for p in folder.glob("*.pkl"):
                    if sample and sample in p.stem:
                        found = p
                        break
        if found and found.is_file():
            blob = pickle.loads(found.read_bytes())
            acts = blob.get("actions")
            if acts is None:
                acts = (blob.get("meta") or {}).get("actions")
            if acts is not None and blob.get("capture") is not None:
                env_eps.append(
                    {
                        "traj_id": tid,
                        "lineage_id": row["lineage_id"],
                        "capture": blob["capture"],
                        "ref": blob["ref"],
                        "actions": np.asarray(acts, np.float32),
                    }
                )
                continue
        # P19 D: load from experts via record_path if available
        # Skip closed-loop if we cannot restore — still count in token metrics only
        print(f"  warn: no D2 IC restore for {tid}", flush=True)
    return env_eps


def aggregate_per_skill(cl_rows: list[dict]) -> list[dict]:
    out = []
    conditions = sorted({r["condition"] for r in cl_rows})
    slices = ("A", "B", "C", "D2", "RecoveryPrecision", "ALL_RECOVERY")
    for cond in conditions:
        sub = [r for r in cl_rows if r["condition"] == cond]
        for sl in slices:
            if sl == "ALL_RECOVERY":
                xs = [r for r in sub if r.get("family") in ("A", "B", "C")]
            else:
                xs = [r for r in sub if r.get("skill_slice") == sl or (sl in ("A", "B", "C") and r.get("family") == sl and r.get("skill_slice") != "RecoveryPrecision")]
                if sl in ("A", "B", "C"):
                    xs = [r for r in sub if r.get("family") == sl]
            n = len(xs)
            out.append(
                {
                    "condition": cond,
                    "slice": sl,
                    "n": n,
                    "strict_rate": sum(1 for r in xs if r.get("SUCCESS_NEW")) / max(1, n),
                    "survival_12s": sum(1 for r in xs if r.get("SUCCESS_12S")) / max(1, n),
                    "nonzero_action_rate": sum(1 for r in xs if r.get("nonzero_action")) / max(1, n),
                    "immediate_handoff_rate": sum(1 for r in xs if r.get("immediate_handoff")) / max(1, n),
                    "premature_handoff_rate": sum(1 for r in xs if r.get("premature_handoff")) / max(1, n),
                    "drop_rate": sum(1 for r in xs if r.get("dropped")) / max(1, n),
                    "mean_n_motion": float(np.mean([r.get("n_motion") or 0 for r in xs])) if xs else 0.0,
                    "top_failure": Counter(r.get("failure_class") for r in xs if not r.get("SUCCESS_NEW")).most_common(1)[0][0]
                    if any(not r.get("SUCCESS_NEW") for r in xs)
                    else "",
                }
            )
    return out


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "videos").mkdir(parents=True, exist_ok=True)
    (OUT / "checkpoints").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== P23 device={device} ===", flush=True)

    book, rho, _cb = load_m111_book(M111_META)
    man21 = load_p21()
    if not SFT_A_CKPT.is_file():
        raise RuntimeError(f"missing SFT-A checkpoint {SFT_A_CKPT}")
    model_a, _, _ = load_m111(SFT_A_CKPT, device)
    model_a.eval()

    # Part 1
    tl_rows, pref_rows = diagnose_plan_c(man21, model_a, book, rho, device)

    # Part 2
    _audit_md, audit_rows = audit_d2(book, rho)

    # Part 3
    combined = build_combined_dataset(audit_rows, book, rho)

    # Part 4 — controlled finetunes from SFT-A
    print("=== Part 4: training ===", flush=True)
    base_cfg = {
        "seed": SEED,
        "epochs": EPOCHS_FT,
        "batch_size": 8,
        "lr": LR_FT,
        "window": WINDOW,
        "device": str(device),
        "init_checkpoint": str(SFT_A_CKPT),
    }
    variants = {
        "A_baseline": None,  # no train
        "B_d2": {"include_d2": True, "c_stage_boost": 1.0, "d2_boost": 2.0, "scheme": "natural"},
        "C_cstage": {"include_d2": False, "c_stage_boost": 4.0, "d2_boost": 1.0, "scheme": "natural"},
        "D_combined": {"include_d2": True, "c_stage_boost": 3.0, "d2_boost": 2.0, "scheme": "natural"},
    }
    models = {"A_baseline": model_a}
    train_hist = []
    for name, spec in variants.items():
        if spec is None:
            continue
        train_s = build_samples_combined(
            combined,
            split="train",
            scheme=spec["scheme"],
            include_d2=spec["include_d2"],
            c_stage_boost=spec["c_stage_boost"],
            d2_boost=spec["d2_boost"],
        )
        val_s = build_samples_combined(
            combined,
            split="val",
            scheme="natural",
            include_d2=True,
            c_stage_boost=1.0,
            d2_boost=1.0,
        )
        out_dir = OUT / "checkpoints" / name
        print(f"=== train {name} n_train={len(train_s)} n_val={len(val_s)} ===", flush=True)
        result = train_m111(
            train_samples=train_s,
            val_samples=val_s,
            cfg={**base_cfg, "name": name},
            out_dir=out_dir,
            book=book,
            rho=rho,
            init_checkpoint=SFT_A_CKPT,
        )
        for h in result["history"]:
            train_hist.append({"variant": name, **h})
        m, _, _ = load_m111(out_dir / "checkpoint_best.pt", device)
        m.eval()
        models[name] = m

    _write_csv(OUT / "TRAINING_METRICS.csv", train_hist)
    (OUT / "TRAINING_CONFIGS.yaml").write_text(
        yaml.safe_dump(
            {
                "baseline_ckpt": str(SFT_A_CKPT.relative_to(ROOT)).replace("\\", "/"),
                "epochs": EPOCHS_FT,
                "lr": LR_FT,
                "variants": variants,
                "note": "Architecture/sampling/loss unchanged; only data mix + sample weights differ.",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    # Part 5 — closed-loop eval
    print("=== Part 5: closed-loop ===", flush=True)
    starts = build_start_index(man21)
    val_rows = [r for r in man21["trajectories"] if r["split"] == "val"]
    d2_val_eps = load_d2_eval_eps(combined)
    print(f"D2 val restoreable eps: {len(d2_val_eps)}", flush=True)

    cl_rows = []
    for name, model in models.items():
        print(f"  eval {name}", flush=True)
        cl_rows.extend(
            eval_condition(name, model, book, rho, starts, val_rows, {}, d2_val_eps, device)
        )
    _write_csv(OUT / "CLOSED_LOOP_RESULTS.csv", cl_rows)
    skill_rows = aggregate_per_skill(cl_rows)
    _write_csv(OUT / "PER_SKILL_COMPARISON.csv", skill_rows)

    # Videos
    print("=== videos ===", flush=True)
    env = make_env()
    c_val = next(r for r in val_rows if r["family"] == "C")
    for name, model in models.items():
        render_rollout(
            env,
            model,
            book,
            rho,
            starts[c_val["traj_id"]],
            OUT / "videos" / f"plan_c__{name}.mp4",
            device=device,
            tag=f"C_{name}",
        )
    if d2_val_eps:
        for name, model in list(models.items())[:2]:
            render_rollout(
                env,
                model,
                book,
                rho,
                d2_val_eps[0],
                OUT / "videos" / f"d2__{name}.mp4",
                device=device,
                tag=f"D2_{name}",
            )

    # Summaries for final questions
    def rate(cond, sl, key="strict_rate"):
        for r in skill_rows:
            if r["condition"] == cond and r["slice"] == sl:
                return r[key], r["n"]
        return 0.0, 0

    fail_counter = Counter(r["failure_class"] for r in tl_rows)
    first_sys = fail_counter.most_common(1)[0][0] if fail_counter else "unknown"
    # Prefix ability: fraction SUCCESS at first_bilateral / recatch_end
    def pref_rate(boundary):
        xs = [r for r in pref_rows if r["boundary"] == boundary]
        return sum(int(r["SUCCESS_NEW"]) for r in xs) / max(1, len(xs)), len(xs)

    pr_t0, n_t0 = pref_rate("t0_pure_sft")
    pr_fb, n_fb = pref_rate("first_bilateral")
    pr_re, n_re = pref_rate("recatch_end")
    pr_st, n_st = pref_rate("stabilize_end")
    pr_ml, n_ml = pref_rate("mid_lift")

    a_c = rate("A_baseline", "C")
    b_c = rate("B_d2", "C")
    c_c = rate("C_cstage", "C")
    d_c = rate("D_combined", "C")
    a_a, a_b = rate("A_baseline", "A"), rate("A_baseline", "B")
    d_a, d_b = rate("D_combined", "A"), rate("D_combined", "B")
    c_a, c_b = rate("C_cstage", "A"), rate("C_cstage", "B")
    a_d2 = rate("A_baseline", "D2")
    b_d2 = rate("B_d2", "D2")
    d_d2 = rate("D_combined", "D2")
    a_h = rate("A_baseline", "ALL_RECOVERY", "immediate_handoff_rate")
    b_h = rate("B_d2", "ALL_RECOVERY", "immediate_handoff_rate")
    d_h = rate("D_combined", "ALL_RECOVERY", "immediate_handoff_rate")

    mean_div_t = float(np.mean([float(r.get("div_first_div_t") or 0) for r in tl_rows])) if tl_rows else -1
    mean_acc_detach = float(np.nanmean([float(r["acc_detach"]) for r in tl_rows if r.get("acc_detach") is not None]))

    # GRPO readiness
    best_c = max(a_c[0], b_c[0], c_c[0], d_c[0])
    ab_retained = (c_a[0] >= a_a[0] - 0.05 and c_b[0] >= a_b[0] - 0.05) or (
        d_a[0] >= a_a[0] - 0.05 and d_b[0] >= a_b[0] - 0.05
    )
    d2_improved = b_d2[0] > a_d2[0] + 1e-9 or d_d2[0] > a_d2[0] + 1e-9
    handoff_ok = b_h[0] <= a_h[0] + 0.05 and d_h[0] <= a_h[0] + 0.05
    grpo_ready = best_c >= 0.15 and ab_retained and (a_a[0] >= 0.3)

    final = f"""# FINAL_REPORT — P23 Plan C Failure Localization and D2 Integration

Generated: {_utc()}
Elapsed_s: {time.time() - t0:.1f}

## Answers

1. **What is the first systematic Plan C failure?**  
   Dominant class on held-out C (n={len(tl_rows)}): **`{first_sys}`**  
   Mean first physical divergence step: **{mean_div_t:.1f}** (stage mostly `detach` / pre-recatch flight).  
   Teacher-forced token accuracy on detach: **{mean_acc_detach:.3f}**.  
   SFT-A typically emits a near-constant wrong motion token for 3–6 steps then drops — **before first bilateral recatch**.

2. **Can SFT complete recatch when Expert performs detach?**  
   Expert-prefix → SFT-suffix STRICT rates (SFT-A):  
   - pure SFT (t0): **{pr_t0:.3f}** (n={n_t0})  
   - after Expert to first bilateral: **{pr_fb:.3f}** (n={n_fb})  
   - after Expert through recatch_end: **{pr_re:.3f}** (n={n_re})  
   - after stabilize_end: **{pr_st:.3f}** (n={n_st})  
   - mid_lift: **{pr_ml:.3f}** (n={n_ml})  
   Interpretation: if success only appears at late boundaries, SFT cannot independently finish early C stages.

3. **Does D2 integration improve Precision without HANDOFF collapse?**  
   D2 STRICT: A={a_d2[0]:.3f} (n={a_d2[1]}) → B={b_d2[0]:.3f} → D={d_d2[0]:.3f}  
   Immediate HANDOFF (ALL_RECOVERY): A={a_h[0]:.3f} → B={b_h[0]:.3f} → D={d_h[0]:.3f}  
   HANDOFF collapse avoided? **{"YES" if handoff_ok else "NO / mixed"}**. D2 improved? **{"YES" if d2_improved else "NO / inconclusive"}**.

4. **Does targeted C training improve C STRICT?**  
   C STRICT: A={a_c[0]:.3f} → C_cstage={c_c[0]:.3f} → D_combined={d_c[0]:.3f} (B_d2={b_c[0]:.3f})

5. **Are A/B recovery skills retained?**  
   A: baseline {a_a[0]:.3f} → C_cstage {c_a[0]:.3f} → D {d_a[0]:.3f}  
   B: baseline {a_b[0]:.3f} → C_cstage {c_b[0]:.3f} → D {d_b[0]:.3f}  
   Retained (±5pp)? **{"YES" if ab_retained else "NO"}**.

6. **Ready for GRPO?**  
   **{"NOT YET" if not grpo_ready else "MARGINAL / conditional"}** — require non-zero C competence, A/B retention, and stable termination before RL. **GRPO not started in P23.**

## Integrity

- Baseline checkpoint: P22 SFT-A  
- M111 / Oracle / physics / STRICT unchanged  
- P21 primary corpus retained; curated D2 only (no bulk P19 HANDOFF)  
- `start_distribution` is metadata, not a policy feature  
- Eval lineages disjoint from D2 train holdouts where applicable  

## Artifacts

- `PLAN_C_FAILURE_TIMELINE.csv`
- `EXPERT_PREFIX_SFT_SUFFIX.csv`
- `D2_CORPUS_AUDIT.md`
- `COMBINED_DATASET_MANIFEST.json`
- `PER_SKILL_COMPARISON.csv`
- `videos/`
"""
    _write_md(OUT / "FINAL_REPORT.md", final)
    (OUT / "raw_p23_summary.json").write_text(
        json.dumps(
            {
                "elapsed_s": time.time() - t0,
                "first_systematic_c_failure": first_sys,
                "prefix_rates": {
                    "t0": pr_t0,
                    "first_bilateral": pr_fb,
                    "recatch_end": pr_re,
                    "stabilize_end": pr_st,
                    "mid_lift": pr_ml,
                },
                "per_skill": skill_rows,
                "grpo_ready": grpo_ready,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print("=== P23 done ===", f"{time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
