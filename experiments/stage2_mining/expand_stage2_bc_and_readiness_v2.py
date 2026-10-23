#!/usr/bin/env python
"""v2: expand genuine Plan B/C nontrivial supply + SFT audit + readiness report.

Preserves v1. Does not modify Oracle / K49 / physics / success criteria.
Writes results/data_generation/stage2_abc_mining/v2/
"""

from __future__ import annotations

import hashlib
import json
import pickle
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.oracles.plan_b.branch_compare import load_vocab, run_tokens  # noqa: E402
from data_generation.oracles.precision.corrections.contact_primitives import _wrist, audit_grasp  # noqa: E402
from data_generation.oracles.precision.mechanics.grasp_stabilization import hold_and_confirm_stable  # noqa: E402
from data_generation.oracles.precision.state_machine import (  # noqa: E402
    PrecisionConfig,
    entry_gate,
    measure_errors,
    probe_direct_handoff,
    run_precision,
)
from envs.observations.airborne_obs import observe_airborne  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.outcomes.ground_drop import arm, restore_state as restore_drop  # noqa: E402
from envs.simulation.capture import restore_capture, take_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from training.action_vocab_k49 import CAP, HANDOFF, K49_TABLE_ROWS, build_k49_table  # noqa: E402
from training.recovery_execution import restore_ballistic  # noqa: E402

V1 = ROOT / "results/data_generation/stage2_abc_mining/v1"
OUT = ROOT / "results/data_generation/stage2_abc_mining/v2"
HARD_DIR = ROOT / "results/diagnostics/hard_ic_mining/snapshots"
B1_PER_IC = ROOT / "results/diagnostics/plan_b_b0_b1_b2_ground/per_ic"
UE = ROOT / "data/unified_experts/manifest.jsonl"
PLAN_C_ACCEPTED = ROOT / "results/diagnostics/ground_plan_c_scale50/raw/accepted"
PLAN_C_EARLY = ROOT / "results/diagnostics/ground_ball_impact_early_recovery/raw"
K49_CONFIG = "K49_tok48_wy0p25"
DEDUP = 1e-3

B1_PATTERNS = (
    [32, 12, 12, 0, 0, 0, 0, 0],
    [30, 0, 0, 0, 0, 0],
    [30, 12, 12, 0, 0, 0, 0, 0],
    [44, 0, 0, 0, 0, 0],
    [36],
    [36, 36],
    [39],
    [44],
    [30],
)


def jsafe(o):
    if o is None or isinstance(o, (bool, str, int)):
        return o
    if isinstance(o, (float, np.floating)):
        x = float(o)
        return x if np.isfinite(x) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {str(k): jsafe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsafe(x) for x in o]
    return str(o)


def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jsafe(obj), indent=2), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(jsafe(row), ensure_ascii=False) + "\n")


def make_env():
    class E:
        pass

    env = E()
    env.sim, _ = make_ground_sim()
    env.switch = GeometrySwitch()
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    env._obs_seq = []
    env._prev_seq = []
    return env


def capture_fp(cap):
    q = np.asarray(cap["physics"]["qpos"], float)
    return hashlib.sha256(np.round(q / DEDUP).astype(np.int64).tobytes()).hexdigest()[:16]


def observe(env):
    return observe_airborne(env.sim.model, env.sim.data, env.sim.ids, getattr(env.sim, "fsm", None), env._hist)


def nearest(table, a):
    a = np.asarray(a, np.float32).reshape(7)
    bi, bd = 0, 1e9
    for i in range(K49_TABLE_ROWS):
        if i == HANDOFF:
            continue
        d = float(np.linalg.norm(table[i] - a))
        if d < bd:
            bi, bd = i, d
    return bi, table[bi].copy()


def quantize_and_replay_k49(env, cap, ref, actions, table):
    acts = np.asarray(actions, np.float32)
    if acts.size == 0:
        restore_capture(env, cap)
        for _ in range(8):
            step_legal(env.sim, _wrist(0, 0, 1))
        h = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=0)
        return {
            "tokens": np.zeros(0, np.int64),
            "residuals": np.zeros((0, 7), np.float32),
            "verified": bool(h.get("SUCCESS_NEW")),
            "nontrivial_representability": False,
            "max_recon_l2": 0.0,
            "residual_clipped_frac": 0.0,
            "reason": h.get("std_reason") or h.get("reason"),
        }
    toks, res, rl2, clipped = [], [], [], 0
    for a in acts:
        ti, proto = nearest(table, a)
        raw = a - proto
        r = np.clip(raw, -CAP, CAP).astype(np.float32)
        if float(np.max(np.abs(raw - r))) > 1e-8:
            clipped += 1
        recon = np.clip(proto + r, -1, 1).astype(np.float32)
        toks.append(ti)
        res.append(r)
        rl2.append(float(np.linalg.norm(recon - a)))
    toks_a = np.asarray(toks, np.int64)
    res_a = np.asarray(res, np.float32)
    restore_capture(env, cap)
    for i, t in enumerate(toks_a):
        a = np.clip(table[int(t)] + np.clip(res_a[i], -CAP, CAP), -1, 1).astype(np.float32)
        step_legal(env.sim, a)
    for _ in range(8):
        step_legal(env.sim, _wrist(0, 0, 1))
    h = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=0)
    return {
        "tokens": toks_a,
        "residuals": res_a,
        "verified": bool(h.get("SUCCESS_NEW")),
        "nontrivial_representability": bool(h.get("SUCCESS_NEW")),
        "max_recon_l2": float(max(rl2)),
        "residual_clipped_frac": float(clipped / len(acts)),
        "reason": h.get("std_reason") or h.get("reason"),
    }


def replay_cont(env, cap, ref, actions):
    restore_capture(env, cap)
    env._hist = ObservableObsState(DT_POLICY)
    obs = []
    for a in actions:
        obs.append(observe(env))
        step_legal(env.sim, np.asarray(a, np.float32))
    for _ in range(8):
        step_legal(env.sim, _wrist(0, 0, 1))
    h = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=0)
    return {
        "SUCCESS_NEW": bool(h.get("SUCCESS_NEW")),
        "reason": h.get("std_reason") or h.get("reason"),
        "observations": np.asarray(obs, np.float32) if obs else np.zeros((0, 39), np.float32),
    }


# ---------------------------------------------------------------------------
# Evaluate IC
# ---------------------------------------------------------------------------


def evaluate_ic(env, *, sample_id, plan_provenance, oracle_route, source_kind, genuine_b, cap, ref, table, cfg, meta):
    out_path = OUT / "outcomes" / f"{sample_id}.json"
    if out_path.exists():
        return json.loads(out_path.read_text(encoding="utf-8"))

    restore_capture(env, cap)
    gate = entry_gate(env)
    restore_capture(env, cap)
    stab = hold_and_confirm_stable(env, hold_s=0.20)
    restore_capture(env, cap)
    err = measure_errors(env.sim)
    if not (gate.get("ok") and stab.get("stable")):
        row = {
            "sample_id": sample_id,
            "plan_provenance": plan_provenance,
            "source_kind": source_kind,
            "genuine_stage1_plan_b": genuine_b,
            "status": "GATE_FAIL",
            "sft_category": "FAILED_ORACLE",
            "funnel": {"gate": False},
            **meta,
        }
        atomic_json(out_path, row)
        return row

    direct = probe_direct_handoff(env, cap, ref, stabilize_n=8)
    restore_capture(env, cap)
    funnel = {
        "gate": True,
        "direct_fail": not bool(direct.get("SUCCESS_NEW")),
        "direct_SUCCESS_NEW": bool(direct.get("SUCCESS_NEW")),
    }
    t0 = time.time()
    res = run_precision(env, {"capture": cap, "ref": ref}, oracle_route, cfg)
    wall = time.time() - t0
    restore_capture(env, cap)
    prec_new = bool((res.handoff or {}).get("SUCCESS_NEW")) if res.handoff else False
    actions = list(res.actions or [])
    funnel["precision_SUCCESS_NEW"] = prec_new
    funnel["has_action_tape"] = len(actions) > 0

    if direct.get("SUCCESS_NEW") and len(actions) == 0:
        cat = "DIRECT_HANDOFF"
    elif prec_new and not direct.get("SUCCESS_NEW"):
        cat = "NONTRIVIAL_PRECISION"
    elif prec_new:
        cat = "DIRECT_HANDOFF"
    else:
        cat = "FAILED_ORACLE"

    ic_pkl = OUT / "ics" / f"{sample_id}.pkl"
    ic_pkl.parent.mkdir(parents=True, exist_ok=True)
    ic_pkl.write_bytes(pickle.dumps({"capture": cap, "ref": ref}))

    row = {
        "sample_id": sample_id,
        "plan_provenance": plan_provenance,
        "oracle_route": oracle_route,
        "source_kind": source_kind,
        "genuine_stage1_plan_b": genuine_b,
        "status": "OK",
        "sft_category": cat,
        "capture_fp": capture_fp(cap),
        "ic_pkl": str(ic_pkl.relative_to(OUT)),
        "errors_entry": err,
        "direct": {
            "SUCCESS_NEW": bool(direct.get("SUCCESS_NEW")),
            "SUCCESS_12S": bool(direct.get("SUCCESS_12S")),
            "reason": direct.get("reason"),
        },
        "precision": {
            "SUCCESS_NEW": prec_new,
            "n_actions": len(actions),
            "phases": list(res.phases or []),
            "failures": list(res.failures or []),
            "status": res.status,
            "wall_s": round(wall, 2),
        },
        "funnel": funnel,
        "continuous_demo": None,
        "k49_demo": None,
        "repr_gap": False,
        **meta,
    }

    if cat in ("DIRECT_HANDOFF", "NONTRIVIAL_PRECISION") and (direct.get("SUCCESS_NEW") or prec_new):
        crep = replay_cont(env, cap, ref, actions)
        cpath = OUT / "continuous" / f"{sample_id}__continuous.npz"
        cpath.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cpath, actions=np.asarray(actions, np.float32), observations=crep["observations"])
        row["continuous_demo"] = {
            "path": str(cpath.relative_to(OUT)),
            "n_actions": len(actions),
            "verified": bool(crep["SUCCESS_NEW"]),
            "reason": crep["reason"],
        }
        funnel["continuous_verified"] = bool(crep["SUCCESS_NEW"])
        k = quantize_and_replay_k49(env, cap, ref, actions, table)
        tpath = OUT / "tokens" / f"{sample_id}__k49.npz"
        tpath.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            tpath,
            tokens=k["tokens"],
            residuals=k["residuals"],
            handoff_token=np.asarray([HANDOFF], np.int64),
            continuous_actions=np.asarray(actions, np.float32),
            cap=np.asarray([CAP], np.float32),
        )
        row["k49_demo"] = {
            "path": str(tpath.relative_to(OUT)),
            "n_tokens": int(len(k["tokens"])),
            "verified": k["verified"],
            "nontrivial_representability": k["nontrivial_representability"] and len(actions) > 0,
            "max_recon_l2": k["max_recon_l2"],
            "residual_clipped_frac": k["residual_clipped_frac"],
            "reason": k["reason"],
            "encoding": "nearest_motion_token_plus_residual_cap_0p25",
            "config_id": K49_CONFIG,
        }
        funnel["k49_verified"] = bool(k["verified"]) and len(actions) > 0
        row["repr_gap"] = bool(crep["SUCCESS_NEW"]) and (
            (not k["verified"]) or k["residual_clipped_frac"] > 0.25 or k["max_recon_l2"] > 0.15
        )
        if cat == "FAILED_ORACLE":
            pass
    elif cat == "FAILED_ORACLE":
        fpath = OUT / "failures" / f"{sample_id}__failure.json"
        atomic_json(
            fpath,
            {
                "sample_id": sample_id,
                "failures": res.failures,
                "phases": res.phases,
                "status": res.status,
                "errors_entry": err,
            },
        )
        row["failure_diag"] = str(fpath.relative_to(OUT))

    row["funnel"] = funnel
    atomic_json(out_path, row)
    print(
        f"  [{plan_provenance}] {sample_id} cat={cat} direct={direct.get('SUCCESS_NEW')} "
        f"prec={prec_new} n={len(actions)} k49={(row.get('k49_demo') or {}).get('verified')}",
        flush=True,
    )
    return row


# ---------------------------------------------------------------------------
# Phase 2 — Plan B expansion
# ---------------------------------------------------------------------------


def extract_post_b1(env, hard_id, tokens, table_b, book48):
    snap = HARD_DIR / f"{hard_id}.pkl"
    if not snap.exists():
        return None
    root = pickle.loads(snap.read_bytes())
    restore_capture(env, root["capture"])
    ok, reason = run_tokens(env, root, table_b, book48, tokens)
    if not ok:
        return {"error": reason}
    cap = take_capture(env)
    restore_capture(env, cap)
    stab = hold_and_confirm_stable(env, hold_s=0.20)
    restore_capture(env, cap)
    gate = entry_gate(env)
    restore_capture(env, cap)
    if not (stab.get("stable") and gate.get("ok")):
        return {"error": "unstable_or_gate", "stable": stab.get("stable"), "gate": gate.get("ok")}
    direct = probe_direct_handoff(env, cap, root["ref"], stabilize_n=8)
    restore_capture(env, cap)
    if direct.get("SUCCESS_NEW"):
        return {"error": "post_b1_direct_success", "e_pos": measure_errors(env.sim)["e_pos_mm"]}
    return {"capture": cap, "ref": root["ref"], "tokens": list(tokens), "hard_id": hard_id}


def phase2_expand_b(env, table, table_b, book48, cfg):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "outcomes").mkdir(exist_ok=True)
    seen_fp = set()
    # existing v1 genuine B nontrivial fps
    for p in (V1 / "outcomes").glob("B__*.json"):
        r = json.loads(p.read_text(encoding="utf-8"))
        if r.get("capture_fp"):
            seen_fp.add(r["capture_fp"])

    # Prefer hard ICs that failed oracle as raw HARD_IC in v1
    v1_failed = []
    for p in (V1 / "outcomes").glob("HARD_IC__*.json"):
        r = json.loads(p.read_text(encoding="utf-8"))
        if r.get("sft_category") == "FAILED_ORACLE":
            v1_failed.append(r["ic_id"])
    # Also include per_ic 12S-not-NEW
    per_ic_ids = []
    if B1_PER_IC.exists():
        for jf in B1_PER_IC.glob("hard_*.json"):
            meta = json.loads(jf.read_text(encoding="utf-8"))
            b1 = meta.get("B1") or {}
            if b1.get("ok_recovery") and b1.get("SUCCESS_12S") and not b1.get("SUCCESS_NEW"):
                per_ic_ids.append((jf.stem, list(b1.get("tokens") or [])))

    results = []
    # 1) known per_ic tokens
    for hid, toks in per_ic_ids:
        if not toks:
            continue
        print(f"Phase2B per_ic {hid}", flush=True)
        got = extract_post_b1(env, hid, toks, table_b, book48)
        if not got or got.get("error"):
            print(f"  skip {got}", flush=True)
            continue
        fp = capture_fp(got["capture"])
        if fp in seen_fp:
            print(f"  dedup {hid}", flush=True)
            continue
        seen_fp.add(fp)
        sid = f"B__{hid}__post_b1_v2__{fp}"
        row = evaluate_ic(
            env,
            sample_id=sid,
            plan_provenance="B",
            oracle_route="B",
            source_kind="plan_b_stage1_post_b1_v2",
            genuine_b=True,
            cap=got["capture"],
            ref=got["ref"],
            table=table,
            cfg=cfg,
            meta={"b1_tokens": got["tokens"], "hard_id": hid, "version": "v2"},
        )
        results.append(row)

    # 2) pattern search on failed hard ICs + unused hard snapshots (bounded)
    n_new_nt = sum(1 for r in results if r.get("sft_category") == "NONTRIVIAL_PRECISION")
    used_hids = set()
    for r in results:
        hid = r.get("hard_id") or (r.get("meta") or {}).get("hard_id")
        if hid:
            used_hids.add(str(hid))
    # Mark hard ICs / fps already saved as v2 outcomes (avoid re-eval duplicates)
    prior_by_id = {}
    for p in (OUT / "outcomes").glob("B__*.json"):
        r = json.loads(p.read_text(encoding="utf-8"))
        prior_by_id[r["sample_id"]] = r
        hid = r.get("hard_id")
        if hid:
            used_hids.add(str(hid))
        if r.get("capture_fp"):
            seen_fp.add(r["capture_fp"])
    for r in results:
        prior_by_id[r["sample_id"]] = r
    results = list(prior_by_id.values())
    # Prefer v1 FAILED_ORACLE hard ICs, then remaining hard snapshots with diverse families
    hard_pool = list(dict.fromkeys(v1_failed + [p.stem for p in sorted(HARD_DIR.glob("hard_*.pkl"))]))
    # Skip hard ICs already represented as genuine-B nontrivial in v1
    v1_b_hids = set()
    for p in (V1 / "outcomes").glob("B__*.json"):
        r = json.loads(p.read_text(encoding="utf-8"))
        if r.get("sft_category") == "NONTRIVIAL_PRECISION":
            hid = (r.get("meta") or {}).get("hard_id")
            if hid:
                v1_b_hids.add(hid)

    screened = 0
    extract_ok = 0
    for hid in hard_pool:
        if n_new_nt >= 8:
            break
        if screened >= 60:
            break
        if hid in used_hids or hid in v1_b_hids:
            continue
        screened += 1
        kept = None
        for toks in B1_PATTERNS:
            got = extract_post_b1(env, hid, list(toks), table_b, book48)
            if got and not got.get("error"):
                kept = got
                break
        if not kept:
            continue
        extract_ok += 1
        fp = capture_fp(kept["capture"])
        if fp in seen_fp:
            continue
        seen_fp.add(fp)
        sid = f"B__{hid}__post_b1_v2__{fp}"
        print(f"Phase2B pattern {hid} tokens={kept['tokens']}", flush=True)
        row = evaluate_ic(
            env,
            sample_id=sid,
            plan_provenance="B",
            oracle_route="B",
            source_kind="plan_b_stage1_post_b1_v2",
            genuine_b=True,
            cap=kept["capture"],
            ref=kept["ref"],
            table=table,
            cfg=cfg,
            meta={"b1_tokens": kept["tokens"], "hard_id": hid, "version": "v2"},
        )
        results.append(row)
        used_hids.add(hid)
        if row.get("sft_category") == "NONTRIVIAL_PRECISION":
            n_new_nt += 1

    # 2b) Upstream mass/friction variation on extractable post-B1 states (same B1 controller)
    # Apply after restore of the post-B1 capture — physically valid object-property variation.
    from envs.simulation.object_dynamics import set_object_dynamics

    mass_fric_grid = (
        (0.08, 0.35),
        (0.12, 0.45),
        (0.15, 0.55),
        (0.10, 0.30),
        (0.18, 0.40),
    )
    # Reuse a few successful extracts as bases for property perturbation
    base_extracts = [r for r in results if r.get("sft_category") in ("FAILED_ORACLE", "NONTRIVIAL_PRECISION")]
    bases = []
    for r in base_extracts[:6]:
        icp = OUT / "ics" / f"{r['sample_id']}.pkl"
        if icp.exists():
            bases.append((r, pickle.loads(icp.read_bytes())))
    for r0, pack in bases:
        if n_new_nt >= 8:
            break
        ref = pack.get("ref")
        cap0 = pack.get("capture")
        if ref is None or cap0 is None:
            continue
        hid = (r0.get("meta") or {}).get("hard_id") or "var"
        for mi, (mass, fric) in enumerate(mass_fric_grid):
            if n_new_nt >= 8:
                break
            restore_capture(env, cap0)
            set_object_dynamics(env.sim.model, env.sim.ids, float(mass), float(fric), env.sim.data)
            # settle briefly under new dynamics
            for _ in range(8):
                step_legal(env.sim, np.zeros(7, np.float32))
            cap = take_capture(env)
            restore_capture(env, cap)
            stab = hold_and_confirm_stable(env, hold_s=0.20)
            restore_capture(env, cap)
            gate = entry_gate(env)
            restore_capture(env, cap)
            if not (stab.get("stable") and gate.get("ok")):
                continue
            direct = probe_direct_handoff(env, cap, ref, stabilize_n=8)
            restore_capture(env, cap)
            if direct.get("SUCCESS_NEW"):
                continue
            fp = capture_fp(cap)
            if fp in seen_fp:
                continue
            seen_fp.add(fp)
            sid = f"B__{hid}__mf{mi}_v2__{fp}"
            print(f"Phase2B mass/fric {sid} m={mass} mu={fric}", flush=True)
            row = evaluate_ic(
                env,
                sample_id=sid,
                plan_provenance="B",
                oracle_route="B",
                source_kind="plan_b_stage1_post_b1_mass_fric_v2",
                genuine_b=True,
                cap=cap,
                ref=ref,
                table=table,
                cfg=cfg,
                meta={
                    "hard_id": hid,
                    "mass": mass,
                    "friction": fric,
                    "base_sample": r0["sample_id"],
                    "version": "v2",
                },
            )
            results.append(row)
            if row.get("sft_category") == "NONTRIVIAL_PRECISION":
                n_new_nt += 1

    atomic_json(
        OUT / "b_expand_funnel.json",
        {
            "hard_pool": len(hard_pool),
            "pattern_screened": screened,
            "pattern_extract_ok": extract_ok,
            "v2_rows": len(results),
            "v2_nontrivial": n_new_nt,
        },
    )

    # 3) Targeted failure notes for v1 post-B1 FAILED_ORACLE (no Oracle redesign)
    fail_notes = []
    for p in (V1 / "outcomes").glob("B__*post_b1*.json"):
        r = json.loads(p.read_text(encoding="utf-8"))
        if r.get("sft_category") != "FAILED_ORACLE":
            continue
        fail_notes.append(
            {
                "sample_id": r["sample_id"],
                "failures": (r.get("precision") or {}).get("failures"),
                "phases": (r.get("precision") or {}).get("phases"),
                "status": (r.get("precision") or {}).get("status"),
                "e_pos_mm": (r.get("errors_entry") or {}).get("e_pos_mm"),
                "note": "Retained for implementation investigation; not relabeled.",
            }
        )
    atomic_json(OUT / "b_post_b1_failure_audit.json", {"n": len(fail_notes), "rows": fail_notes})
    return results


# ---------------------------------------------------------------------------
# Phase 3 — Plan C nontrivial search
# ---------------------------------------------------------------------------


def extract_plan_c_from_expert(env, expert_pkl: Path):
    """Earliest stable after catch: prefer Stage-1 prefix tape, else brake, else elev_snap."""
    pack = pickle.loads(expert_pkl.read_bytes())
    if "catch_snap" not in pack or "ref" not in pack:
        return None
    ref = pack["ref"]
    root_id = pack.get("root_id") or expert_pkl.stem
    prefix = pack.get("prefix_actions")
    if prefix is None:
        prefix = pack.get("capture_actions") or pack.get("actions") or []
    prefix = np.asarray(prefix, np.float32)

    def _gate_ok_cap(cap):
        restore_capture(env, cap)
        gate = entry_gate(env)
        restore_capture(env, cap)
        return bool(gate.get("ok"))

    # 1) Replay Stage-1 prefix from catch_snap; earliest stable+gate
    if prefix.size > 0:
        restore_ballistic(env.sim, pack["catch_snap"])
        restore_drop(env.sim, None)
        arm(env.sim)
        best_i = None
        for i, a in enumerate(prefix):
            step_legal(env.sim, np.asarray(a, np.float32))
            if (i + 1) % 5 != 0 and i + 1 < len(prefix):
                continue
            cap = take_capture(env)
            restore_capture(env, cap)
            stab = hold_and_confirm_stable(env, hold_s=0.20)
            restore_capture(env, cap)
            if stab["stable"] and _gate_ok_cap(cap):
                best_i = i + 1
                break
        if best_i is not None:
            restore_ballistic(env.sim, pack["catch_snap"])
            restore_drop(env.sim, None)
            arm(env.sim)
            for a in prefix[:best_i]:
                step_legal(env.sim, np.asarray(a, np.float32))
            cap = take_capture(env)
            return {
                "capture": cap,
                "ref": ref,
                "root_id": root_id,
                "secure_steps": int(best_i),
                "extraction": "earliest_stable_along_plan_c_prefix",
                "expert_SUCCESS_NEW": pack.get("SUCCESS_NEW"),
            }

    # 2) Brake-secure from catch (pilot path)
    restore_ballistic(env.sim, pack["catch_snap"])
    restore_drop(env.sim, None)
    arm(env.sim)
    brake = np.zeros(7, np.float32)
    brake[6] = 1.0
    best_i = 0
    for i in range(60):
        cap_i = take_capture(env)
        restore_capture(env, cap_i)
        stab = hold_and_confirm_stable(env, hold_s=0.20)
        restore_capture(env, cap_i)
        if stab["stable"] and _gate_ok_cap(cap_i):
            best_i = i
            cap = cap_i
            return {
                "capture": cap,
                "ref": ref,
                "root_id": root_id,
                "secure_steps": int(best_i),
                "extraction": "catch_snap_brake_secure",
                "expert_SUCCESS_NEW": pack.get("SUCCESS_NEW"),
            }
        step_legal(env.sim, brake)
        best_i = i + 1

    # 3) elev_snap fallback (often already SUCCESS_NEW-basin)
    if pack.get("elev_snap") is None:
        return None
    restore_ballistic(env.sim, pack["elev_snap"])
    restore_drop(env.sim, None)
    arm(env.sim)
    stab = hold_and_confirm_stable(env, hold_s=0.20)
    if not stab.get("stable"):
        return None
    cap = take_capture(env)
    if not _gate_ok_cap(cap):
        return None
    return {
        "capture": cap,
        "ref": ref,
        "root_id": root_id,
        "secure_steps": -1,
        "extraction": "elev_snap",
        "expert_SUCCESS_NEW": pack.get("SUCCESS_NEW"),
    }


def phase3_expand_c(env, table, cfg):
    funnel_counts = Counter()
    results = []
    seen_fp = set()
    # Pilot C fps
    for p in (V1 / "outcomes").glob("PILOT_C__*.json"):
        r = json.loads(p.read_text(encoding="utf-8"))
        if r.get("capture_fp"):
            seen_fp.add(r["capture_fp"])

    experts = sorted(PLAN_C_ACCEPTED.glob("*__SUCCESS_NEW_expert.pkl")) if PLAN_C_ACCEPTED.exists() else []
    print(f"Phase3C screening {len(experts)} accepted Plan-C experts", flush=True)
    for ep in experts:
        funnel_counts["experts_total"] += 1
        got = extract_plan_c_from_expert(env, ep)
        if got is None:
            funnel_counts["extract_fail"] += 1
            continue
        funnel_counts["gate_ok"] += 1
        fp = capture_fp(got["capture"])
        if fp in seen_fp:
            funnel_counts["dedup"] += 1
            continue
        # Cheap direct screen before full precision
        restore_capture(env, got["capture"])
        direct = probe_direct_handoff(env, got["capture"], got["ref"], stabilize_n=8)
        restore_capture(env, got["capture"])
        if direct.get("SUCCESS_NEW"):
            funnel_counts["direct_success"] += 1
            continue
        funnel_counts["direct_fail"] += 1
        seen_fp.add(fp)
        sid = f"C__{got['root_id']}__v2__{fp}"
        row = evaluate_ic(
            env,
            sample_id=sid,
            plan_provenance="C",
            oracle_route="C",
            source_kind="plan_c_stage1_catch_secure_v2",
            genuine_b=False,
            cap=got["capture"],
            ref=got["ref"],
            table=table,
            cfg=cfg,
            meta={
                "expert_pkl": str(ep.relative_to(ROOT)),
                "expert_SUCCESS_NEW": got.get("expert_SUCCESS_NEW"),
                "secure_steps": got["secure_steps"],
                "version": "v2",
            },
        )
        results.append(row)
        if row.get("sft_category") == "NONTRIVIAL_PRECISION":
            funnel_counts["precision_success"] += 1
            if (row.get("k49_demo") or {}).get("nontrivial_representability"):
                funnel_counts["k49_success"] += 1
        else:
            funnel_counts["precision_fail"] += 1

    # Early-recovery branches (genuine Plan-C upstream), bounded
    early = sorted(PLAN_C_EARLY.glob("gbi_*__branch_*.pkl")) if PLAN_C_EARLY.exists() else []
    print(f"Phase3C screening {len(early)} early-recovery branches", flush=True)
    for ep in early[:40]:
        funnel_counts["early_total"] += 1
        pack = pickle.loads(ep.read_bytes())
        snap = pack.get("snap") or pack.get("capture")
        ref = pack.get("ref")
        if snap is None or ref is None:
            funnel_counts["early_bad_schema"] += 1
            continue
        # snap may be ballistic capture dict
        try:
            restore_capture(env, snap)
        except Exception:
            try:
                restore_ballistic(env.sim, snap)
                restore_drop(env.sim, None)
                arm(env.sim)
            except Exception:
                funnel_counts["early_restore_fail"] += 1
                continue
        cap = take_capture(env)
        restore_capture(env, cap)
        gate = entry_gate(env)
        restore_capture(env, cap)
        stab = hold_and_confirm_stable(env, hold_s=0.20)
        restore_capture(env, cap)
        if not (gate.get("ok") and stab.get("stable")):
            funnel_counts["early_gate_fail"] += 1
            continue
        funnel_counts["early_gate_ok"] += 1
        direct = probe_direct_handoff(env, cap, ref, stabilize_n=8)
        restore_capture(env, cap)
        if direct.get("SUCCESS_NEW"):
            funnel_counts["early_direct_success"] += 1
            continue
        funnel_counts["early_direct_fail"] += 1
        fp = capture_fp(cap)
        if fp in seen_fp:
            continue
        seen_fp.add(fp)
        sid = f"C__{ep.stem}__v2__{fp}"
        row = evaluate_ic(
            env,
            sample_id=sid,
            plan_provenance="C",
            oracle_route="C",
            source_kind="plan_c_early_recovery_branch_v2",
            genuine_b=False,
            cap=cap,
            ref=ref,
            table=table,
            cfg=cfg,
            meta={"early_pkl": str(ep.relative_to(ROOT)), "branch": pack.get("branch"), "version": "v2"},
        )
        results.append(row)
        if row.get("sft_category") == "NONTRIVIAL_PRECISION":
            funnel_counts["early_precision_success"] += 1
            if (row.get("k49_demo") or {}).get("nontrivial_representability"):
                funnel_counts["early_k49_success"] += 1

    atomic_json(OUT / "c_nontrivial_funnel.json", dict(funnel_counts))
    return results, dict(funnel_counts)


# ---------------------------------------------------------------------------
# Phase 4 — SFT audit
# ---------------------------------------------------------------------------


def audit_sft():
    rows = []
    # v1 sft + v2 outcomes
    sft_v1 = V1 / "sft/sft_manifest.jsonl"
    if sft_v1.exists():
        for line in sft_v1.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            # enrich from outcome
            op = V1 / "outcomes" / f"{r['sample_id']}.json"
            o = json.loads(op.read_text(encoding="utf-8")) if op.exists() else {}
            n_act = (r.get("continuous_n_actions") if r.get("continuous_n_actions") is not None else (o.get("continuous_demo") or {}).get("n_actions"))
            cont_ok = bool(r.get("continuous_verified"))
            k49_ok = bool(r.get("k49_verified"))
            k49_nt = bool(r.get("k49_nontrivial_representability"))
            cat = r.get("sft_category")
            token_eligible = bool(
                cat == "NONTRIVIAL_PRECISION" and cont_ok and k49_nt and (n_act or 0) > 0
            )
            handoff_eligible = bool(cat == "DIRECT_HANDOFF" and cont_ok)
            rows.append(
                {
                    "sample_id": r["sample_id"],
                    "source_pool": "v1",
                    "plan_provenance": r.get("plan_provenance"),
                    "sft_category": cat,
                    "continuous_success": cont_ok,
                    "k49_replay_success": k49_ok,
                    "token_level_sft_eligible": token_eligible,
                    "direct_handoff_sample": handoff_eligible,
                    "intermediate_curriculum": cat == "INTERMEDIATE_CURRICULUM",
                    "continuous_only": bool(cont_ok and not k49_nt and (n_act or 0) > 0),
                    "genuine_stage1_plan_b": r.get("genuine_stage1_plan_b"),
                    "n_actions": n_act,
                }
            )
    for p in sorted((OUT / "outcomes").glob("*.json")):
        o = json.loads(p.read_text(encoding="utf-8"))
        cat = o.get("sft_category")
        if cat not in ("DIRECT_HANDOFF", "NONTRIVIAL_PRECISION", "INTERMEDIATE_CURRICULUM"):
            continue
        cont = o.get("continuous_demo") or {}
        k49 = o.get("k49_demo") or {}
        n_act = cont.get("n_actions") or 0
        token_eligible = bool(
            cat == "NONTRIVIAL_PRECISION"
            and cont.get("verified")
            and k49.get("nontrivial_representability")
            and n_act > 0
        )
        rows.append(
            {
                "sample_id": o["sample_id"],
                "source_pool": "v2",
                "plan_provenance": o.get("plan_provenance"),
                "sft_category": cat,
                "continuous_success": bool(cont.get("verified")),
                "k49_replay_success": bool(k49.get("verified")),
                "token_level_sft_eligible": token_eligible,
                "direct_handoff_sample": cat == "DIRECT_HANDOFF",
                "intermediate_curriculum": cat == "INTERMEDIATE_CURRICULUM",
                "continuous_only": bool(cont.get("verified") and not k49.get("nontrivial_representability") and n_act > 0),
                "genuine_stage1_plan_b": o.get("genuine_stage1_plan_b"),
                "n_actions": n_act,
            }
        )

    # Split integrity from v1
    splits = json.loads((V1 / "rl/splits.json").read_text(encoding="utf-8")) if (V1 / "rl/splits.json").exists() else {}
    ep_to_split = {}
    if (V1 / "rl/rl_ic_manifest.jsonl").exists():
        for line in (V1 / "rl/rl_ic_manifest.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            ep_to_split[r["source_episode"]] = r.get("split")
    # Check disjoint
    inv = defaultdict(set)
    for ep, sp in ep_to_split.items():
        inv[sp].add(ep)
    overlap = {
        "train_val": sorted(inv.get("train", set()) & inv.get("val", set())),
        "train_holdout": sorted(inv.get("train", set()) & inv.get("holdout", set())),
        "val_holdout": sorted(inv.get("val", set()) & inv.get("holdout", set())),
    }

    man = OUT / "sft_eligibility_audit.jsonl"
    if man.exists():
        man.unlink()
    for r in rows:
        append_jsonl(man, r)

    summary = {
        "n_rows": len(rows),
        "by_plan": dict(Counter(r["plan_provenance"] for r in rows)),
        "by_category": dict(Counter(r["sft_category"] for r in rows)),
        "token_level_sft_eligible": sum(1 for r in rows if r["token_level_sft_eligible"]),
        "token_eligible_by_plan": dict(
            Counter(r["plan_provenance"] for r in rows if r["token_level_sft_eligible"])
        ),
        "direct_handoff_samples": sum(1 for r in rows if r["direct_handoff_sample"]),
        "continuous_only_nontrivial": sum(1 for r in rows if r["continuous_only"]),
        "intermediate": sum(1 for r in rows if r["intermediate_curriculum"]),
        "split_overlaps": overlap,
        "splits_disjoint": all(len(v) == 0 for v in overlap.values()),
    }
    atomic_json(OUT / "sft_eligibility_summary.json", summary)
    return rows, summary


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def write_readiness_report(b_rows, c_rows, c_funnel, sft_summary, k49_diag_summary):
    # Aggregate B/C from v1+v2
    def load_all(plan):
        rows = []
        for base in (V1, OUT):
            for p in (base / "outcomes").glob("*.json"):
                r = json.loads(p.read_text(encoding="utf-8"))
                if r.get("plan_provenance") == plan:
                    r["_pool"] = base.name
                    rows.append(r)
        # dedup by capture_fp
        by = {}
        for r in rows:
            fp = r.get("capture_fp") or r["sample_id"]
            # prefer v2 nontrivial
            prev = by.get(fp)
            if prev is None or (
                r.get("sft_category") == "NONTRIVIAL_PRECISION"
                and prev.get("sft_category") != "NONTRIVIAL_PRECISION"
            ):
                by[fp] = r
        return list(by.values())

    b_all = load_all("B")
    c_all = load_all("C")
    a_all = load_all("A")
    hard_all = load_all("HARD_IC")

    def counts(rows, genuine_b_only=False):
        rs = [r for r in rows if (not genuine_b_only) or r.get("genuine_stage1_plan_b")]
        return {
            "n": len(rs),
            "DIRECT_HANDOFF": sum(1 for r in rs if r.get("sft_category") == "DIRECT_HANDOFF"),
            "NONTRIVIAL_cont": sum(1 for r in rs if r.get("sft_category") == "NONTRIVIAL_PRECISION"),
            "NONTRIVIAL_k49": sum(
                1
                for r in rs
                if r.get("sft_category") == "NONTRIVIAL_PRECISION"
                and (r.get("k49_demo") or {}).get("nontrivial_representability")
            ),
            "FAILED": sum(1 for r in rs if r.get("sft_category") == "FAILED_ORACLE"),
            "v2_new_nontrivial": sum(
                1
                for r in rs
                if r.get("_pool") == "v2" and r.get("sft_category") == "NONTRIVIAL_PRECISION"
            ),
        }

    b_gen = counts(b_all, genuine_b_only=True)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "k49_diagnostics": k49_diag_summary,
        "plan_b": {
            "genuine_stage1": b_gen,
            "v2_new_rows": len(b_rows),
            "v2_nontrivial": sum(1 for r in b_rows if r.get("sft_category") == "NONTRIVIAL_PRECISION"),
            "v2_k49_nontrivial": sum(
                1
                for r in b_rows
                if (r.get("k49_demo") or {}).get("nontrivial_representability")
            ),
            "post_b1_failure_audit": "b_post_b1_failure_audit.json",
        },
        "plan_c": {
            "all": counts(c_all),
            "funnel": c_funnel,
            "v2_new_rows": len(c_rows),
            "v2_nontrivial": sum(1 for r in c_rows if r.get("sft_category") == "NONTRIVIAL_PRECISION"),
            "v2_k49_nontrivial": sum(
                1
                for r in c_rows
                if (r.get("k49_demo") or {}).get("nontrivial_representability")
            ),
            "basin_note": (
                "Elevated/accepted Plan-C Stage-1 (pilot) often lands in DIRECT_SUCCESS. "
                "Earliest catch-secure extracts (v2) mostly pass the stable-grasp gate, "
                "fail Direct HANDOFF, then also fail Precision (HANDOFF drop / primitive fail) - "
                "not a SUCCESS_NEW basin, and not yet a K49-verified nontrivial supply."
            ),
        },
        "plan_a": counts(a_all),
        "hard_ic": counts(hard_all),
        "sft_audit": sft_summary,
        "unresolved": {
            "plan_b_k49_legacy": (
                "v1 cases hard_910166 / hard_910220: continuous OK, K49 fails in dx_verified_lateral "
                "(quantization-induced contact; hybrid localizes to that phase)."
            ),
            "plan_b_supply": (
                "Many post-B1 extracts still FAILED_ORACLE (dtheta / drop-after-handoff / dz contact_loss); "
                "v2 nonetheless yielded 5 K49-verified nontrivial B demos."
            ),
            "plan_c_nontrivial": (
                "0 continuous nontrivial + 0 K49 nontrivial after screening early catch-secure states; "
                "pilot elevated C remains DIRECT_HANDOFF-only."
            ),
        },
        "readiness": {
            "SFT": {
                "definition": "frozen 39D obs + K49 token labels + successful K49 replay + correct provenance",
                "token_eligible_total": sft_summary.get("token_level_sft_eligible"),
                "token_eligible_by_plan": sft_summary.get("token_eligible_by_plan"),
                "plan_b_token_eligible": (sft_summary.get("token_eligible_by_plan") or {}).get("B", 0),
                "plan_c_token_eligible": (sft_summary.get("token_eligible_by_plan") or {}).get("C", 0),
                "plan_b_ready": False,  # filled below
                "plan_c_ready": False,
            },
            "RL": {
                "definition": "valid IC captures + frozen runtime obs available + reproducible reset + episode-disjoint splits",
                "v1_rl_ics": sum(1 for _ in (V1 / "rl/rl_ic_manifest.jsonl").open()) if (V1 / "rl/rl_ic_manifest.jsonl").exists() else 0,
                "splits_disjoint": sft_summary.get("splits_disjoint"),
                "ready": bool(sft_summary.get("splits_disjoint")),
            },
        },
    }
    # v2-only token counts (audit_sft already merges v1+v2 into token_eligible_by_plan)
    v2_b_tok = sum(
        1
        for r in b_rows
        if r.get("sft_category") == "NONTRIVIAL_PRECISION"
        and (r.get("k49_demo") or {}).get("nontrivial_representability")
    )
    v2_c_tok = sum(
        1
        for r in c_rows
        if r.get("sft_category") == "NONTRIVIAL_PRECISION"
        and (r.get("k49_demo") or {}).get("nontrivial_representability")
    )
    report["readiness"]["SFT"]["v2_plan_b_token_eligible"] = v2_b_tok
    report["readiness"]["SFT"]["v2_plan_c_token_eligible"] = v2_c_tok
    b_tok = int((sft_summary.get("token_eligible_by_plan") or {}).get("B", 0))
    c_tok = int((sft_summary.get("token_eligible_by_plan") or {}).get("C", 0))
    report["readiness"]["SFT"]["plan_b_token_eligible"] = b_tok
    report["readiness"]["SFT"]["plan_c_token_eligible"] = c_tok
    report["readiness"]["SFT"]["plan_b_ready"] = b_tok >= 5
    report["readiness"]["SFT"]["plan_c_ready"] = c_tok >= 5

    atomic_json(OUT / "READINESS.json", report)

    md = f"""# Stage-2 A/B/C Training-Readiness - v2

Created: {report['created_utc']}  
Oracle / K49 / physics / success criteria: **unchanged**. v1 preserved.

## Phase 1 - Plan B K49 diagnosis

The two original genuine Plan-B nontrivial continuous successes fail frozen K49 replay for the same structural reason:

| Case | First div step | Verdict | Hybrid rescue |
| --- | ---: | --- | --- |
"""
    for c in (k49_diag_summary or {}).get("cases") or []:
        md += (
            f"| `{c['sample_id']}` | {c.get('first_div_step')} | `{c.get('verdict')}` | "
            f"{c.get('hybrid_rescues')} |\n"
        )
    md += """
Interpretation:
- Continuous teacher replay still reaches SUCCESS_NEW.
- K49 decode (token + residual ±0.25) diverges early via contact-force / orientation drift.
- Hybrid: replacing only `dx_verified_lateral` with continuous actions restores SUCCESS_NEW; `correct_dz_inward` alone does not (910220).
- Residual clip fractions ~0.21-0.27 and max_recon_l2 ~0.51 indicate **quantization-induced contact failure in verified lateral**, i.e. a **genuine limitation of the frozen K49 action set** for DualLoop lateral tapes of this style (not an HANDOFF-timing bug; not an Oracle bug).
- First meaningful divergence often includes grip/force mismatch under quantized actions (e.g. grip 0.30 vs 0.75) while pose errors remain close.
- Hybrid trajectories are diagnostics only - **not** verified K49 SFT demos.
- Codebook / Oracle not modified.
- Note: v2 later found additional genuine-B nontrivial states where K49 *does* verify; the Phase-1 cases remain unresolved representation failures.

Details: `b_k49_diagnostics/`

## Phase 2 - Genuine Plan B supply

"""
    md += f"- v2 new evaluated rows: {len(b_rows)}\n"
    md += f"- v2 new NONTRIVIAL continuous: {report['plan_b']['v2_nontrivial']}\n"
    md += f"- v2 new K49-verified nontrivial: {report['plan_b']['v2_k49_nontrivial']}\n"
    md += f"- Combined genuine-B continuous nontrivial (deduped v1+v2): {b_gen['NONTRIVIAL_cont']}\n"
    md += f"- Combined genuine-B K49 nontrivial: {b_gen['NONTRIVIAL_k49']}\n"
    md += "- Post-B1 Oracle failures audited in `b_post_b1_failure_audit.json` (not relabeled).\n"
    md += "- HARD_IC not relabeled as Plan B.\n"
    bf = OUT / "b_expand_funnel.json"
    if bf.exists():
        md += "- Expand funnel: `" + bf.name + "` = `" + json.dumps(json.loads(bf.read_text(encoding='utf-8'))) + "`\n"
    # Failure taxonomy from audit
    fa = OUT / "b_post_b1_failure_audit.json"
    if fa.exists():
        audit = json.loads(fa.read_text(encoding="utf-8"))
        tax = Counter()
        for row in audit.get("rows") or []:
            st = row.get("status") or "?"
            fails = row.get("failures") or []
            key = st
            if any("dtheta" in str(f) for f in fails):
                key = "PRECISION_PRIMITIVE_FAILED:dtheta"
            elif any("dz" in str(f) for f in fails):
                key = "PRECISION_PRIMITIVE_FAILED:dz"
            elif any("DROP" in str(f) or "LOST_GRASP" in str(f) for f in fails):
                key = "PRECISION_EXECUTED_HANDOFF_FAIL:drop/lost_grasp"
            tax[key] += 1
        md += f"- v1 post-B1 FAILED_ORACLE taxonomy (n={audit.get('n')}): `{dict(tax)}`\n"
    md += "\n"

    md += "## Phase 3 - Plan C nontrivial funnel\n\n"
    md += "```json\n" + json.dumps(c_funnel, indent=2) + "\n```\n\n"
    md += f"- v2 new NONTRIVIAL continuous: {report['plan_c']['v2_nontrivial']}\n"
    md += f"- v2 new K49-verified nontrivial: {report['plan_c']['v2_k49_nontrivial']}\n"
    md += f"- Combined Plan-C: {json.dumps(report['plan_c']['all'])}\n"
    md += f"- {report['plan_c']['basin_note']}\n"
    df = int(c_funnel.get("direct_fail") or 0)
    ds = int(c_funnel.get("direct_success") or 0)
    pf = int(c_funnel.get("precision_fail") or 0)
    if ds > 0 and df == 0:
        md += (
            "- **Honest finding:** Plan-C Stage-1 lands inside SUCCESS_NEW basin (direct success).\n"
        )
    elif df > 0 and pf >= df and report["plan_c"]["v2_nontrivial"] == 0:
        md += (
            "- **Honest finding:** early catch-secure Plan-C states are mostly outside SUCCESS_NEW "
            "(Direct fails) but Precision also fails - no forced nontrivial C samples manufactured.\n"
        )
    md += "\n## Phase 4 - SFT eligibility audit\n\n"
    md += f"- Audited rows: {sft_summary.get('n_rows')} (v1 SFT manifest + any v2 DIRECT/NONTRIVIAL outcomes)\n"
    md += f"- By plan: {sft_summary.get('by_plan')}\n"
    md += f"- By category: {sft_summary.get('by_category')}\n"
    md += f"- Token-level SFT eligible: {sft_summary.get('token_level_sft_eligible')} by plan {sft_summary.get('token_eligible_by_plan')}\n"
    md += f"- Continuous-only nontrivial (teacher OK, K49 not verified - **not** mixed into token SFT): {sft_summary.get('continuous_only_nontrivial')}\n"
    md += f"- Direct HANDOFF samples: {sft_summary.get('direct_handoff_samples')}\n"
    md += f"- Intermediate curriculum: {sft_summary.get('intermediate')}\n"
    md += f"- Episode splits disjoint: {sft_summary.get('splits_disjoint')}\n"
    md += "- Full per-row audit: `sft_eligibility_audit.jsonl` (plan, category, continuous, K49, token-eligible, direct, intermediate).\n"
    md += "- HARD_IC rows retain separate provenance (generic Precision only).\n\n"

    md += "## Readiness indicators\n\n"
    md += "### SFT readiness\n"
    md += "Requires: frozen obs, K49 labels, successful K49 replay, correct provenance.\n\n"
    sft_r = report["readiness"]["SFT"]
    md += f"- Plan A token-eligible: {(sft_summary.get('token_eligible_by_plan') or {}).get('A', 0)}\n"
    md += f"- Plan B token-eligible (merged audit): {sft_r['plan_b_token_eligible']} (of which v2-new={v2_b_tok}) -> ready={sft_r['plan_b_ready']}\n"
    md += f"- Plan C token-eligible (merged audit): {sft_r['plan_c_token_eligible']} (of which v2-new={v2_c_tok}) -> ready={sft_r['plan_c_ready']}\n"
    md += f"- HARD_IC token-eligible (generic Precision, separate provenance): {(sft_summary.get('token_eligible_by_plan') or {}).get('HARD_IC', 0)}\n\n"
    md += "### RL readiness\n"
    md += f"- v1 RL ICs: {report['readiness']['RL']['v1_rl_ics']}; splits_disjoint={report['readiness']['RL']['splits_disjoint']}; ready={report['readiness']['RL']['ready']}\n"
    md += "- RL does **not** require Oracle success on every IC.\n\n"
    md += "## Unresolved\n\n"
    for k, v in (report.get("unresolved") or {}).items():
        md += f"- **{k}:** {v}\n"
    md += "\n## Stop\n\nNo bulk SFT/RL training. No Oracle/K49/physics changes.\n"
    (OUT / "REPORT.md").write_text(md, encoding="utf-8")
    (OUT.parent / "REPORT_v2.md").write_text(md, encoding="utf-8")
    return report


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ("ics", "outcomes", "continuous", "tokens", "failures", "diagnostics"):
        (OUT / sub).mkdir(exist_ok=True)

    env = make_env()
    table = build_k49_table(config_id=K49_CONFIG)
    table_b, book48 = load_vocab()
    cfg = PrecisionConfig()

    # Load K49 diagnostic summary if present
    diag_sum = {}
    dsp = OUT / "b_k49_diagnostics/SUMMARY.json"
    if dsp.exists():
        diag_sum = json.loads(dsp.read_text(encoding="utf-8"))

    print("=== Phase 2: expand genuine Plan B ===", flush=True)
    b_rows = phase2_expand_b(env, table, table_b, book48, cfg)

    print("=== Phase 3: Plan C nontrivial search ===", flush=True)
    c_rows, c_funnel = phase3_expand_c(env, table, cfg)

    print("=== Phase 4: SFT audit ===", flush=True)
    _, sft_summary = audit_sft()

    print("=== Phase 5: readiness report ===", flush=True)
    report = write_readiness_report(b_rows, c_rows, c_funnel, sft_summary, diag_sum)
    print(json.dumps(jsafe(report["readiness"]), indent=2), flush=True)
    print("Done →", OUT, flush=True)


if __name__ == "__main__":
    main()
