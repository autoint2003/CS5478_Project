#!/usr/bin/env python
"""Focused Plan B post-recatch correction: reuse actual Plan A oracle.

Compares, from the exact post-B1-recatch capture:
  A) immediate HANDOFF
  B) current B2 heuristic pose tokens
  C) Plan A controlled_sliding + inward_reseat (seed schedules + feedback)
  D) multi-step Plan-A-inspired combinations of promising short prefixes

Does not modify physics / SUCCESS_NEW / K49. Writes a new diagnostics folder.
No SFT/GRPO. No full 22-IC expansion.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.datasets.recovery_rebuild import (  # noqa: E402
    CAP_STEPS,
    LIMIT_S,
    controlled_slide_action,
    inward_reseat_action,
    legalize,
)
from data_generation.oracles.plan_a import controlled_sliding as cs_mod  # noqa: E402
from data_generation.oracles.plan_a import inward_reseat as ir_mod  # noqa: E402
from data_generation.oracles.plan_a.inward_reseat import _priority_cuts  # noqa: E402
from data_generation.oracles.plan_b.branch_compare import (  # noqa: E402
    decode_tok,
    eval_branch,
    exact_replay_3x,
    load_vocab,
    pose_candidates,
    pose_family,
    run_tokens,
)
from data_generation.oracles.plan_c.contact_surface import classify_contacts  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.outcomes.horizon_outcome import recovery_failed  # noqa: E402
from envs.outcomes.physical_recovery import physical_pack  # noqa: E402
from envs.outcomes.support_drop import latched  # noqa: E402
from envs.simulation.capture import restore_capture, take_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.grasp_geometry import measure  # noqa: E402
from evaluation.success_criteria import P_TGT, U_TGT, contact_summary, window_stats  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from evaluation.handoff_from_state import evaluate_handoff_from_state  # noqa: E402
from training.k47_codebook import load_codebook  # noqa: E402

OUT = ROOT / "results" / "diagnostics" / "plan_b_posta_plan_a_correction"
DT = 0.02

# Best known B1 active-recatch tapes from the paused 22-IC run (Recovery tokens only).
B1_TAPES = {
    "hard_910220": [32, 12, 12, 0, 0, 0, 0, 0],
    "hard_910101": [30, 0, 0, 0, 0, 0],  # short open + close body from search; override if per_ic has better
    "hard_910305": [30, 12, 12, 0, 0, 0, 0, 0],
}


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


def snapshot_path(ic_id: str) -> Path:
    for p in (
        ROOT / f"results/diagnostics/hard_ic_mining/snapshots/{ic_id}.pkl",
        ROOT / f"results/benchmarks/hard_recovery_v1/snapshots/{ic_id}.pkl",
    ):
        if p.exists():
            return p
    raise FileNotFoundError(ic_id)


def load_b1_tokens(ic_id: str) -> list[int]:
    per = ROOT / "results/diagnostics/plan_b_b0_b1_b2_ground/per_ic" / f"{ic_id}.json"
    if per.exists():
        row = json.loads(per.read_text(encoding="utf-8"))
        toks = row.get("B1", {}).get("tokens")
        if toks:
            return [int(t) for t in toks]
    return list(B1_TAPES[ic_id])


def geom_diag(sim) -> dict:
    m = measure(sim, U_TGT, P_TGT)
    cs = contact_summary(sim)
    pack = physical_pack(sim)
    surf = classify_contacts(sim)
    p_c = np.asarray(m["p_c_h"], float)
    p_t = np.asarray(P_TGT, float)
    err = p_c - p_t
    rh = np.asarray(pack["rh"], float)
    return {
        "e_theta_deg": float(m["e_theta_deg"]),
        "e_pos_mm": float(m["e_pos_mm"]),
        "p_c_h_mm": (p_c * 1e3).tolist(),
        "p_tgt_h_mm": (p_t * 1e3).tolist(),
        "e_signed_hand_mm": (err * 1e3).tolist(),
        "e_axial_mm": float(err[2] * 1e3),  # hand-z often along finger/pinch depth; report all axes
        "e_lateral_xy_mm": [float(err[0] * 1e3), float(err[1] * 1e3)],
        "rh_hand_mm": (rh * 1e3).tolist(),
        "v_rel_h": m["v_rel_h"],
        "w_rel_h": m["w_rel_h"],
        "nL": int(cs["nL"]),
        "nR": int(cs["nR"]),
        "bilateral": bool(cs["bilateral"]),
        "Fn_L": float(pack["Fn_L"]),
        "Fn_R": float(pack["Fn_R"]),
        "tau": float(pack["tau"]),
        "g_h": np.asarray(pack["g_h"], float).tolist(),
        "contact_s_mean": surf.get("contact_s_mean"),
        "contact_surface": {
            "grasp_kind": surf.get("grasp_kind"),
            "opposing_sidewall": surf.get("opposing_sidewall"),
            "n_sidewall": surf.get("n_sidewall"),
            "n_end_face": surf.get("n_end_face"),
            "maj_L": surf.get("maj_L"),
            "maj_R": surf.get("maj_R"),
            "contact_s_mean": surf.get("contact_s_mean"),
            "normal_dot_LR": surf.get("normal_dot_LR"),
        },
        "obj_z": float(m["obj_z"]),
        "ground_drop": bool(latched(sim)),
    }


def short_horizon_after_actions(env, post_blob, actions_7d: list, *, horizon_s: float = 0.6) -> dict:
    """Cheap screen: run correction only; no 12 s nominal."""
    restore_capture(env, post_blob["capture"])
    g0 = geom_diag(env.sim)
    lost = 0
    drop = False
    for a in actions_7d:
        step_legal(env.sim, np.asarray(a, np.float32))
        failed, lost = recovery_failed(env.sim, post_blob["ref"], lost)
        if failed:
            drop = True
            break
    # Optional short hold
    n_hold = int(round(horizon_s / DT))
    for _ in range(n_hold):
        step_legal(env.sim, np.zeros(7, np.float32))
        failed, lost = recovery_failed(env.sim, post_blob["ref"], lost)
        if failed:
            drop = True
            break
    g1 = geom_diag(env.sim)
    e0 = np.asarray(g0["e_signed_hand_mm"], float)
    e1 = np.asarray(g1["e_signed_hand_mm"], float)
    return {
        "drop": drop,
        "g0": g0,
        "g1": g1,
        "de_signed_mm": (e1 - e0).tolist(),
        "d_e_pos_mm": float(g1["e_pos_mm"] - g0["e_pos_mm"]),
        "d_theta_deg": float(g1["e_theta_deg"] - g0["e_theta_deg"]),
        "improved_pos": bool(g1["e_pos_mm"] + 1e-6 < g0["e_pos_mm"]),
        "retained_bilateral": bool(g1["bilateral"]) and not drop,
        "score": (
            (0 if drop else 10)
            + (5 if g1["bilateral"] else -20)
            + max(0.0, float(g0["e_pos_mm"] - g1["e_pos_mm"]))
            + 0.2 * max(0.0, float(g0["e_theta_deg"] - g1["e_theta_deg"]))
        ),
    }


def full_success_new(env, post_blob, actions_7d: list) -> dict:
    restore_capture(env, post_blob["capture"])
    lost = 0
    for a in actions_7d:
        step_legal(env.sim, np.asarray(a, np.float32))
        failed, lost = recovery_failed(env.sim, post_blob["ref"], lost)
        if failed:
            g = geom_diag(env.sim)
            return {
                "SUCCESS_12S": False,
                "SUCCESS_NEW": False,
                "reason": "DROP_DURING_CORRECTION",
                "pre": g,
                "window": None,
                "samples_tail": [],
            }
    pre = geom_diag(env.sim)
    meta = evaluate_handoff_from_state(env.sim, post_blob["ref"], stabilize_n=0)
    # Pull window from meta if present
    win = meta.get("window")
    return {
        "SUCCESS_12S": bool(meta.get("reason") == "SUCCESS_12S"),
        "SUCCESS_NEW": bool(meta.get("SUCCESS_NEW")),
        "reason": meta.get("reason"),
        "pre": {k: pre[k] for k in ("e_theta_deg", "e_pos_mm", "e_signed_hand_mm", "bilateral", "contact_s_mean")},
        "pre_theta": meta.get("pre_theta"),
        "pre_pos_mm": meta.get("pre_pos_mm"),
        "longest_both_ok_s": None if not win else win.get("longest_both_ok_s"),
        "mean_theta_final2": None if not win else win.get("mean_theta"),
        "mean_pos_final2": None if not win else win.get("mean_pos"),
        "max_theta_final2": None if not win else win.get("max_theta"),
        "max_pos_final2": None if not win else win.get("max_pos"),
        "working_SUCCESS_NEW_hold": None if not win else win.get("working_SUCCESS_NEW_hold"),
    }


def run_feedback(env, post_blob, kind: str) -> list:
    restore_capture(env, post_blob["capture"])
    spec = {"name": kind, "ref": post_blob["ref"]}
    memory: dict = {}
    actions = []
    lost = 0
    t0 = float(env.sim.data.time)
    fn = controlled_slide_action if kind == "controlled_slide" else inward_reseat_action
    for _ in range(CAP_STEPS):
        if float(env.sim.data.time) - t0 >= LIMIT_S - 1e-9:
            break
        a, phase = fn(env.sim, spec, memory)
        if phase == "done":
            break
        step_legal(env.sim, a)
        actions.append(np.asarray(a, np.float32).copy())
        failed, lost = recovery_failed(env.sim, post_blob["ref"], lost)
        if failed:
            break
    return actions


def tokens_to_7d(table, book48, tokens: list[int]) -> list:
    return [decode_tok(table, book48, t) for t in tokens]


def build_post_recatch(env, root_blob, table, book48, b1_tokens: list[int], ic_id: str, out_dir: Path):
    restore_capture(env, root_blob["capture"])
    ok, reason = run_tokens(env, root_blob, table, book48, b1_tokens, trace=None)
    if not ok:
        raise RuntimeError(f"{ic_id}: B1 replay failed: {reason}")
    # Confirm active recatch state
    g = geom_diag(env.sim)
    if not g["bilateral"]:
        print(f"  WARN {ic_id}: post-B1 not bilateral", flush=True)
    post_cap = take_capture(env)
    post_blob = {
        "capture": post_cap,
        "ref": root_blob["ref"],
        "ic_id": ic_id,
        "b1_tokens": list(b1_tokens),
        "root_identity": {
            "ic_id": ic_id,
            "snapshot": str(snapshot_path(ic_id)),
            "root_time": float(root_blob["capture"]["physics"]["time"]),
            "post_time": float(post_cap["physics"]["time"]),
        },
        "geom": g,
    }
    path = out_dir / f"{ic_id}__post_recatch.pkl"
    path.write_bytes(pickle.dumps(post_blob))
    (out_dir / f"{ic_id}__post_recatch_geom.json").write_text(
        json.dumps(jsafe({"geom": g, "root": post_blob["root_identity"]}), indent=2),
        encoding="utf-8",
    )
    print(
        f"  saved post-recatch {path.name} θ={g['e_theta_deg']:.2f} "
        f"e_pos={g['e_pos_mm']:.2f} e_signed_mm={g['e_signed_hand_mm']} "
        f"contact_s={g['contact_s_mean']} Fn=({g['Fn_L']:.2f},{g['Fn_R']:.2f})",
        flush=True,
    )
    return post_blob


def plan_a_menu(book) -> list[dict]:
    menu = []
    for seed in cs_mod.SEEDS:
        actions, cuts, illegal = cs_mod.build_schedule(seed, book)
        menu.append(
            {
                "arm": "C_plan_a_slide_seed",
                "name": f"cs:{seed['name']}",
                "actions": actions,
                "cuts": cuts,
                "illegal": illegal,
            }
        )
    for seed in ir_mod.SEEDS:
        for sign in (+1.0, -1.0):
            actions, cuts, illegal = ir_mod.build_schedule(seed, sign, book)
            menu.append(
                {
                    "arm": "C_plan_a_inward_seed",
                    "name": f"ir:{seed['name']}:{'pos' if sign > 0 else 'neg'}",
                    "actions": actions,
                    "cuts": cuts,
                    "illegal": illegal,
                }
            )
    return menu


def screen_score_row(screen: dict) -> float:
    return float(screen.get("score") or -1e9)


def run_ic(env, ic_id: str, table, book48, book, rng, *, max_full: int = 10) -> dict:
    t0 = time.time()
    ic_dir = OUT / ic_id
    ic_dir.mkdir(parents=True, exist_ok=True)
    root_blob = pickle.loads(snapshot_path(ic_id).read_bytes())
    b1_tokens = load_b1_tokens(ic_id)
    print(f"\n=== {ic_id} B1_tokens={b1_tokens} ===", flush=True)
    post = build_post_recatch(env, root_blob, table, book48, b1_tokens, ic_id, ic_dir)

    candidates: list[dict] = []

    # --- A: immediate HANDOFF ---
    print("  Arm A: immediate HANDOFF", flush=True)
    screen_a = short_horizon_after_actions(env, post, [], horizon_s=0.0)
    candidates.append(
        {
            "arm": "A_immediate_handoff",
            "name": "immediate",
            "actions": [],
            "screen": screen_a,
            "force_full": True,
        }
    )

    # --- B: current B2 heuristic pose tokens ---
    print("  Arm B: B2 heuristic pose tokens", flush=True)
    poses = [p for p in pose_candidates(rng, n=8) if p]
    for pose in poses:
        acts = tokens_to_7d(table, book48, pose)
        screen = short_horizon_after_actions(env, post, acts)
        candidates.append(
            {
                "arm": "B_b2_heuristic",
                "name": f"pose_{pose_family(pose)}_{pose[:4]}",
                "pose_tokens": list(pose),
                "actions": acts,
                "screen": screen,
                "force_full": False,
            }
        )
        print(
            f"    B {pose_family(pose)} drop={screen['drop']} "
            f"d_pos={screen['d_e_pos_mm']:.2f} d_th={screen['d_theta_deg']:.2f} "
            f"score={screen['score']:.2f}",
            flush=True,
        )

    # --- C: Plan A oracle ---
    print("  Arm C: Plan A schedules + feedback", flush=True)
    for item in plan_a_menu(book):
        if item["illegal"] or item["actions"] is None or len(item["actions"]) == 0:
            continue
        # Prefer short prefixes via cuts for screening
        cuts = item["cuts"] or [{"tag": "end", "end": len(item["actions"])}]
        trial = {"cuts": cuts, "summary": {"drop_step": None}}
        cut_list = list(_priority_cuts(trial))
        if not cut_list:
            cut_list = [{"tag": "end", "end": len(item["actions"])}]
        for cut in cut_list[:3]:
            end = int(cut["end"])
            acts = [np.asarray(a, np.float32) for a in item["actions"][:end]]
            screen = short_horizon_after_actions(env, post, acts)
            candidates.append(
                {
                    "arm": item["arm"],
                    "name": f"{item['name']}|{cut['tag']}|{end}",
                    "actions": acts,
                    "screen": screen,
                    "force_full": False,
                }
            )
            print(
                f"    C {item['name']}|{cut['tag']} n={end} drop={screen['drop']} "
                f"d_pos={screen['d_e_pos_mm']:.2f} bil={screen['retained_bilateral']} "
                f"score={screen['score']:.2f}",
                flush=True,
            )

    for kind in ("controlled_slide", "inward_reseat"):
        acts = run_feedback(env, post, kind)
        screen = short_horizon_after_actions(env, post, acts)
        candidates.append(
            {
                "arm": "C_plan_a_feedback",
                "name": f"fb:{kind}|n={len(acts)}",
                "actions": acts,
                "screen": screen,
                "force_full": True,
            }
        )
        print(
            f"    C fb:{kind} n={len(acts)} drop={screen['drop']} "
            f"d_pos={screen['d_e_pos_mm']:.2f} score={screen['score']:.2f}",
            flush=True,
        )

    # --- D: multi-step from top short improvers ---
    print("  Arm D: multi-step combinations", flush=True)
    improvers = sorted(
        [
            c
            for c in candidates
            if c["arm"].startswith("C_")
            and c["screen"].get("improved_pos")
            and c["screen"].get("retained_bilateral")
            and len(c["actions"]) > 0
        ],
        key=lambda c: float(c["screen"].get("d_e_pos_mm") or 0),
    )[:5]
    # Repeat best improver 2–3× (same Plan A schedule, continuous physics)
    for a in improvers[:2]:
        for reps in (2, 3):
            combo = list(a["actions"]) * reps
            if len(combo) > 160:
                combo = combo[:160]
            screen = short_horizon_after_actions(env, post, combo)
            candidates.append(
                {
                    "arm": "D_multistep",
                    "name": f"{a['name']}x{reps}",
                    "actions": combo,
                    "screen": screen,
                    "force_full": False,
                }
            )
            print(
                f"    D {a['name']}x{reps} drop={screen['drop']} "
                f"d_pos={screen['d_e_pos_mm']:.2f} score={screen['score']:.2f}",
                flush=True,
            )
    for i, a in enumerate(improvers):
        for j, b in enumerate(improvers):
            if i == j:
                continue
            combo = list(a["actions"]) + list(b["actions"])
            if len(combo) > 160:
                combo = combo[:160]
            screen = short_horizon_after_actions(env, post, combo)
            candidates.append(
                {
                    "arm": "D_multistep",
                    "name": f"{a['name']}+{b['name']}",
                    "actions": combo,
                    "screen": screen,
                    "force_full": False,
                }
            )

    # Rank and full-eval: reserve slots for (1) forced arms, (2) best e_pos improvers,
    # (3) best θ improvers, (4) remaining by screen score. Never mix θ/pos across candidates.
    improvers_pos = sorted(
        [
            c
            for c in candidates
            if c["screen"].get("retained_bilateral")
            and float(c["screen"].get("d_e_pos_mm") or 0) < -0.3
        ],
        key=lambda c: float(c["screen"]["d_e_pos_mm"]),
    )
    improvers_th = sorted(
        [
            c
            for c in candidates
            if c["screen"].get("retained_bilateral")
            and float(c["screen"].get("d_theta_deg") or 0) < -0.5
        ],
        key=lambda c: float(c["screen"]["d_theta_deg"]),
    )
    ranked = sorted(candidates, key=screen_score_row, reverse=True)
    forced = [c for c in candidates if c.get("force_full")]
    to_full = []
    seen = set()
    for c in forced + improvers_pos[:6] + improvers_th[:4] + ranked:
        key = c["name"]
        if key in seen:
            continue
        seen.add(key)
        to_full.append(c)
        if len(to_full) >= max_full:
            break
    print(
        f"  improvers_pos={[c['name'] for c in improvers_pos[:6]]} "
        f"improvers_th={[c['name'] for c in improvers_th[:4]]}",
        flush=True,
    )

    print(f"  Full SUCCESS_NEW evals: {len(to_full)} (cap={max_full})", flush=True)
    full_rows = []
    best_new = None
    for c in to_full:
        t1 = time.time()
        out = full_success_new(env, post, c["actions"])
        out["arm"] = c["arm"]
        out["name"] = c["name"]
        out["n_actions"] = len(c["actions"])
        out["screen_score"] = screen_score_row(c["screen"])
        out["screen_d_pos"] = c["screen"].get("d_e_pos_mm")
        out["wall_s"] = round(time.time() - t1, 2)
        # Legalize for potential expert packaging
        if c["actions"]:
            try:
                tokens, residuals, decoded, _ = legalize(np.stack(c["actions"], axis=0), book)
                out["legalized_n"] = int(len(tokens))
                out["legal_tokens_head"] = [int(x) for x in tokens[:12]]
                c["legal_tokens"] = [int(x) for x in tokens]
                c["legal_actions"] = [np.asarray(a, np.float32) for a in decoded]
            except Exception as exc:  # noqa: BLE001
                out["legalize_error"] = str(exc)
        full_rows.append(out)
        print(
            f"    FULL {c['arm']}|{c['name']}: NEW={out['SUCCESS_NEW']} S12={out['SUCCESS_12S']} "
            f"reason={out['reason']} pre_θ={out.get('pre_theta')} pre_pos={out.get('pre_pos_mm')} "
            f"hold={out.get('longest_both_ok_s')} meanθ={out.get('mean_theta_final2')} "
            f"meanpos={out.get('mean_pos_final2')} wall={out['wall_s']}s",
            flush=True,
        )
        if out["SUCCESS_NEW"] and best_new is None:
            best_new = {**c, "full": out}

    replay = None
    if best_new is not None:
        # Full airborne sequence: B1 tokens + legalized correction tokens
        corr_toks = best_new.get("legal_tokens") or []
        full_toks = list(b1_tokens) + list(corr_toks)
        print(f"  Stage D replay 3x full_toks_n={len(full_toks)}", flush=True)
        replay = exact_replay_3x(
            env,
            root_blob,
            table,
            book48,
            full_toks,
            branch="B2",
            prefix_len=1,
        )
        print(f"  replay {replay['n_pass']}/3 ok={replay['ok']}", flush=True)
        if replay["ok"]:
            expert = {
                "ic_id": ic_id,
                "family": "B2_plan_a",
                "b1_tokens": b1_tokens,
                "correction_name": best_new["name"],
                "correction_arm": best_new["arm"],
                "tokens": full_toks,
                "SUCCESS_NEW": True,
                "exact_replay": replay,
                "geom_post_recatch": post["geom"],
                "full": best_new["full"],
            }
            (ic_dir / f"{ic_id}__B2_plan_a_expert.json").write_text(
                json.dumps(jsafe(expert), indent=2), encoding="utf-8"
            )

    # Divergence summary A vs best C
    arm_a = next(r for r in full_rows if r["arm"] == "A_immediate_handoff")
    best_c = None
    for r in full_rows:
        if r["arm"].startswith("C_") or r["arm"] == "D_multistep":
            if best_c is None or (
                int(r["SUCCESS_NEW"]),
                int(r["SUCCESS_12S"]),
                -(r.get("mean_pos_final2") or 1e9),
            ) > (
                int(best_c["SUCCESS_NEW"]),
                int(best_c["SUCCESS_12S"]),
                -(best_c.get("mean_pos_final2") or 1e9),
            ):
                best_c = r
    best_b = None
    for r in full_rows:
        if r["arm"] == "B_b2_heuristic":
            if best_b is None or (int(r["SUCCESS_NEW"]), -(r.get("mean_pos_final2") or 1e9)) > (
                int(best_b["SUCCESS_NEW"]),
                -(best_b.get("mean_pos_final2") or 1e9),
            ):
                best_b = r

    result = {
        "ic_id": ic_id,
        "b1_tokens": b1_tokens,
        "post_geom": post["geom"],
        "n_screened": len(candidates),
        "n_full": len(full_rows),
        "full_rows": full_rows,
        "best_SUCCESS_NEW": None
        if best_new is None
        else {"arm": best_new["arm"], "name": best_new["name"], "full": best_new["full"]},
        "exact_replay": replay,
        "compare": {
            "A": {k: arm_a.get(k) for k in ("SUCCESS_NEW", "SUCCESS_12S", "reason", "pre_pos_mm", "pre_theta", "longest_both_ok_s", "mean_pos_final2", "mean_theta_final2")},
            "B": None
            if best_b is None
            else {k: best_b.get(k) for k in ("SUCCESS_NEW", "SUCCESS_12S", "reason", "name", "pre_pos_mm", "pre_theta", "longest_both_ok_s", "mean_pos_final2")},
            "C_or_D": None
            if best_c is None
            else {k: best_c.get(k) for k in ("SUCCESS_NEW", "SUCCESS_12S", "reason", "arm", "name", "pre_pos_mm", "pre_theta", "longest_both_ok_s", "mean_pos_final2", "mean_theta_final2")},
        },
        "wall_s": round(time.time() - t0, 1),
    }
    # Physical limitation note if no NEW
    if best_new is None:
        g = post["geom"]
        result["physical_limitation_hypothesis"] = {
            "e_signed_hand_mm": g["e_signed_hand_mm"],
            "dominant_axis": int(np.argmax(np.abs(g["e_signed_hand_mm"]))),
            "contact_s_mean": g["contact_s_mean"],
            "note": (
                "If all Plan A schedules retain bilateral grasp but leave |e_pos|>12 mm "
                "with θ already ≤10°, the residual is likely a grasp axial/lateral offset "
                "that Plan A gravity-slip cannot reduce without losing contact, or the "
                "error is not aligned with Plan A's rhx/finger-axis degree of freedom."
            ),
            "best_full_pre_pos": None
            if not full_rows
            else min((r.get("pre_pos_mm") or 1e9) for r in full_rows),
            "best_full_mean_pos_final2": None
            if not full_rows
            else min((r.get("mean_pos_final2") or 1e9) for r in full_rows if r.get("mean_pos_final2") is not None),
        }
    (ic_dir / "result.json").write_text(json.dumps(jsafe(result), indent=2), encoding="utf-8")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ics", default="hard_910220,hard_910101")
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--max-full", type=int, default=12)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    table, book48 = load_vocab()
    book = load_codebook()
    rng = np.random.default_rng(args.seed)
    env = make_env()
    ics = [x.strip() for x in args.ics.split(",") if x.strip()]
    print(
        f"Plan B post-recatch × Plan A correction | ics={ics} max_full={args.max_full}",
        flush=True,
    )
    t0 = time.time()
    rows = []
    for ic in ics:
        rows.append(run_ic(env, ic, table, book48, book, rng, max_full=args.max_full))

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "ics": ics,
        "wall_s": round(time.time() - t0, 1),
        "SUCCESS_NEW": [r["ic_id"] for r in rows if r.get("best_SUCCESS_NEW")],
        "replay_3x": {
            r["ic_id"]: None if r.get("exact_replay") is None else r["exact_replay"].get("n_pass")
            for r in rows
        },
        "compare": {r["ic_id"]: r.get("compare") for r in rows},
        "post_geom": {r["ic_id"]: r.get("post_geom") for r in rows},
    }
    (OUT / "summary.json").write_text(json.dumps(jsafe(summary), indent=2), encoding="utf-8")

    lines = [
        "# Plan B Post-Recatch × Plan A Oracle Correction",
        "",
        f"Created UTC: {summary['created_utc']}",
        f"Wall: {summary['wall_s']} s",
        "",
        "## ICs",
        "",
    ]
    for r in rows:
        g = r["post_geom"]
        lines += [
            f"### {r['ic_id']}",
            "",
            f"- B1 tokens: `{r['b1_tokens']}`",
            f"- Post-recatch: θ={g['e_theta_deg']:.2f}°, e_pos={g['e_pos_mm']:.2f} mm, "
            f"e_signed_hand_mm={g['e_signed_hand_mm']}, contact_s={g['contact_s_mean']}",
            f"- Screened={r['n_screened']} full_evals={r['n_full']}",
            f"- SUCCESS_NEW: {bool(r.get('best_SUCCESS_NEW'))} "
            f"({None if not r.get('best_SUCCESS_NEW') else r['best_SUCCESS_NEW']['name']})",
            f"- Replay: {None if r.get('exact_replay') is None else r['exact_replay']}",
            f"- Compare A/B/C: `{json.dumps(r.get('compare'), default=str)[:500]}`",
            "",
        ]
        if r.get("physical_limitation_hypothesis"):
            lines += [
                f"- Limitation hypothesis: `{json.dumps(r['physical_limitation_hypothesis'], default=str)[:600]}`",
                "",
            ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str), flush=True)
    print(f"Wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
