#!/usr/bin/env python
"""Audit Plan A gravity-slip: does it actively reorient gravity?

Traces controlled_slide_action / inward_reseat_action / seed schedules from the
exact hard_910220 post-B1 capture. Reports phase transitions, g_h during slip,
and SUCCESS_NEW. Does not change physics / SUCCESS_NEW / K49.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.datasets.recovery_rebuild import (  # noqa: E402
    CAP_STEPS,
    EX_ENTER,
    GX_SLIP,
    LIMIT_S,
    SLIDE_OMEGA,
    SLIDE_SLOW,
    SLIDE_TILT_MAX,
    _begin_slide,
    _finger_x,
    _gravity_target,
    _pinch_x,
    controlled_slide_action,
    inward_reseat_action,
    legalize,
)
from data_generation.oracles.plan_a import controlled_sliding as cs_mod  # noqa: E402
from data_generation.oracles.plan_a.inward_reseat import _priority_cuts  # noqa: E402
from data_generation.oracles.plan_b.branch_compare import (  # noqa: E402
    exact_replay_3x,
    load_vocab,
    run_tokens,
)
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.outcomes.horizon_outcome import recovery_failed  # noqa: E402
from envs.outcomes.physical_recovery import physical_pack  # noqa: E402
from envs.simulation.capture import restore_capture, take_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.grasp_geometry import measure  # noqa: E402
from evaluation.success_criteria import P_TGT, U_TGT  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from scripts.generate_plan_c_experts import evaluate_handoff_from_state  # noqa: E402
from training.k47_codebook import load_codebook  # noqa: E402

OUT = ROOT / "results" / "diagnostics" / "plan_a_gravity_slip_audit_910220"
IC = "hard_910220"
POST_PKL = ROOT / "results/diagnostics/plan_b_posta_plan_a_correction/hard_910220/hard_910220__post_recatch.pkl"
ROOT_PKL = ROOT / "results/diagnostics/hard_ic_mining/snapshots/hard_910220.pkl"
B1_TOKENS = [32, 12, 12, 0, 0, 0, 0, 0]
DT = 0.02


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


def sample(sim, ref) -> dict:
    m = measure(sim, U_TGT, P_TGT)
    pack = physical_pack(sim)
    rh = np.asarray(pack["rh"], float)
    g = np.asarray(pack["g_h"], float)
    p_c = np.asarray(m["p_c_h"], float)
    err = p_c - np.asarray(P_TGT, float)
    return {
        "t": float(sim.data.time),
        "phase": None,
        "pinch_x_mm": 1e3 * _pinch_x(sim, ref),
        "finger_x_mm": 1e3 * _finger_x(sim),
        "g_h": g.tolist(),
        "gx": float(g[0]),
        "gy": float(g[1]),
        "gz": float(g[2]),
        "rh_mm": (rh * 1e3).tolist(),
        "e_signed_mm": (err * 1e3).tolist(),
        "e_pos_mm": float(m["e_pos_mm"]),
        "e_theta_deg": float(m["e_theta_deg"]),
        "nL": int(pack["nL"]),
        "nR": int(pack["nR"]),
        "bilateral": bool(int(pack["nL"]) > 0 and int(pack["nR"]) > 0),
        "Fn_L": float(pack["Fn_L"]),
        "Fn_R": float(pack["Fn_R"]),
        "tau": float(pack["tau"]),
        "vx_rel": float(pack["v_rel_h"][0]),
        "turned_deg": None,
        "wy": None,
        "grip": None,
    }


def ensure_post_blob(env, table, book48):
    if POST_PKL.exists():
        return pickle.loads(POST_PKL.read_bytes())
    root = pickle.loads(ROOT_PKL.read_bytes())
    restore_capture(env, root["capture"])
    ok, reason = run_tokens(env, root, table, book48, B1_TOKENS)
    if not ok:
        raise RuntimeError(f"B1 replay failed: {reason}")
    blob = {
        "capture": take_capture(env),
        "ref": root["ref"],
        "ic_id": IC,
        "b1_tokens": list(B1_TOKENS),
    }
    POST_PKL.parent.mkdir(parents=True, exist_ok=True)
    POST_PKL.write_bytes(pickle.dumps(blob))
    return blob


def trace_feedback(env, post, kind: str) -> dict:
    """Instrument rotate/pitch → slip/relax → brake/retighten → return/done."""
    restore_capture(env, post["capture"])
    ref = post["ref"]
    spec = {"name": kind, "ref": ref}
    memory: dict = {}
    rows = []
    actions = []
    phase_log = []
    prev_phase = None
    lost = 0
    t0 = float(env.sim.data.time)
    fn = controlled_slide_action if kind == "controlled_slide" else inward_reseat_action

    # Probe initial decision
    probe = {}
    _begin_slide(env.sim, ref, probe, "slide" if kind == "controlled_slide" else "inward")
    init = sample(env.sim, ref)
    init.update(
        {
            "kind": kind,
            "init_phase": probe.get("phase"),
            "sign": probe.get("sign"),
            "gx_target": probe.get("gx_target"),
            "ex0_pinch_mm": 1e3 * float(probe.get("ex0") or 0.0),
            "EX_ENTER_mm": 1e3 * EX_ENTER,
            "GX_SLIP": GX_SLIP,
            "SLIDE_TILT_MAX_deg": float(np.degrees(SLIDE_TILT_MAX)),
            "slip_axis": "hand_x / g_h[0] (finger-axis gravity projection)",
            "error_coord": "pinch_x (world)" if kind == "controlled_slide" else "finger_x rh[0]",
            "actively_reorients": True,
            "mechanism": (
                "rotate wy until |gx| reaches gx_target (or tilt cap), then unload grip for slip, "
                "brake, then return/untilt toward R0"
                if kind == "controlled_slide"
                else "pitch (wx/wy) until |gx|>=GX_SLIP opposing finger offset, relax grip, retighten (no untilt)"
            ),
        }
    )

    for step in range(CAP_STEPS):
        if float(env.sim.data.time) - t0 >= LIMIT_S - 1e-9:
            break
        a, phase = fn(env.sim, spec, memory)
        row = sample(env.sim, ref)
        row["phase"] = phase
        row["step"] = step
        row["turned_deg"] = float(np.degrees(memory.get("turned") or 0.0))
        row["wy"] = float(a[4]) if len(a) > 4 else None  # may be normalized; also log grip
        row["grip"] = float(a[6])
        row["wx_cmd"] = float(a[3]) if len(a) > 3 else None
        row["wy_cmd"] = float(a[4]) if len(a) > 4 else None
        row["gx_target"] = float(memory.get("gx_target") or 0.0)
        row["sign"] = float(memory.get("sign") or 0.0)
        row["gates"] = list(memory.get("gates") or [])
        rows.append(row)
        actions.append(np.asarray(a, np.float32).copy())
        if phase != prev_phase:
            phase_log.append(
                {
                    "step": step,
                    "t_rel": float(env.sim.data.time) - t0,
                    "phase": phase,
                    "gx": row["gx"],
                    "pinch_x_mm": row["pinch_x_mm"],
                    "finger_x_mm": row["finger_x_mm"],
                    "e_pos_mm": row["e_pos_mm"],
                    "e_theta_deg": row["e_theta_deg"],
                    "turned_deg": row["turned_deg"],
                    "bilateral": row["bilateral"],
                    "gates": row["gates"],
                }
            )
            print(
                f"    phase→{phase} step={step} t={phase_log[-1]['t_rel']:.3f} "
                f"gx={row['gx']:.2f} turned={row['turned_deg']:.1f}° "
                f"ex_pin={row['pinch_x_mm']:.2f} rhx={row['finger_x_mm']:.2f} "
                f"e_pos={row['e_pos_mm']:.2f} θ={row['e_theta_deg']:.2f}",
                flush=True,
            )
            prev_phase = phase
        step_legal(env.sim, a)
        failed, lost = recovery_failed(env.sim, ref, lost)
        if failed or phase == "done":
            # one more sample after last step if done
            if phase == "done":
                end = sample(env.sim, ref)
                end["phase"] = "done"
                end["step"] = step
                end["turned_deg"] = float(np.degrees(memory.get("turned") or 0.0))
                rows.append(end)
            break

    # Phase occupancy
    phases_seen = [p["phase"] for p in phase_log]
    complete_slide = all(
        p in phases_seen for p in ("rotate", "slip", "brake", "return")
    ) or all(p in phases_seen for p in ("rotate", "slip", "brake", "done"))
    # return may end as done without logging return if already aligned — check memory path
    had_rotate = "rotate" in phases_seen or "pitch" in phases_seen
    had_slip = "slip" in phases_seen or "relax" in phases_seen
    had_brake = "brake" in phases_seen or "retighten" in phases_seen
    had_return = "return" in phases_seen
    # Max |gx| during slip/relax
    slip_rows = [r for r in rows if r["phase"] in ("slip", "relax")]
    rot_rows = [r for r in rows if r["phase"] in ("rotate", "pitch")]
    gx_slip = [abs(r["gx"]) for r in slip_rows] or [0.0]
    gx_rot_end = rot_rows[-1]["gx"] if rot_rows else None

    # Signed displacement along slip axis (finger_x / hand-x)
    if rows:
        d_finger = float(rows[-1]["finger_x_mm"] - rows[0]["finger_x_mm"])
        d_pinch = float(rows[-1]["pinch_x_mm"] - rows[0]["pinch_x_mm"])
        d_epos = float(rows[-1]["e_pos_mm"] - rows[0]["e_pos_mm"])
    else:
        d_finger = d_pinch = d_epos = 0.0

    # Full SUCCESS_NEW from end state
    restore_capture(env, post["capture"])
    for a in actions:
        step_legal(env.sim, a)
    meta = evaluate_handoff_from_state(env.sim, ref, stabilize_n=0)
    win = meta.get("window") or {}

    return {
        "kind": kind,
        "init": init,
        "phase_log": phase_log,
        "phases_seen": phases_seen,
        "coverage": {
            "had_rotate_or_pitch": had_rotate,
            "had_slip_or_relax": had_slip,
            "had_brake_or_retighten": had_brake,
            "had_return_untilt": had_return,
            "complete_rotate_slip_brake_return": bool(
                had_rotate and had_slip and had_brake and (had_return or kind == "inward_reseat")
            ),
            "note_inward": "inward_reseat has no untilt/return phase by design",
        },
        "n_actions": len(actions),
        "max_abs_gx_during_slip": float(max(gx_slip)),
        "gx_at_end_of_rotate": gx_rot_end,
        "max_turned_deg": float(max((r["turned_deg"] or 0) for r in rows)) if rows else 0.0,
        "d_finger_x_mm": d_finger,
        "d_pinch_x_mm": d_pinch,
        "d_e_pos_mm": d_epos,
        "start": {k: rows[0][k] for k in ("e_pos_mm", "e_theta_deg", "e_signed_mm", "gx", "finger_x_mm", "pinch_x_mm", "bilateral")}
        if rows
        else None,
        "end": {k: rows[-1][k] for k in ("e_pos_mm", "e_theta_deg", "e_signed_mm", "gx", "finger_x_mm", "pinch_x_mm", "bilateral", "phase")}
        if rows
        else None,
        "SUCCESS_12S": bool(meta.get("reason") == "SUCCESS_12S"),
        "SUCCESS_NEW": bool(meta.get("SUCCESS_NEW")),
        "reason": meta.get("reason"),
        "pre_theta": meta.get("pre_theta"),
        "pre_pos_mm": meta.get("pre_pos_mm"),
        "longest_both_ok_s": win.get("longest_both_ok_s"),
        "mean_pos_final2": win.get("mean_pos"),
        "mean_theta_final2": win.get("mean_theta"),
        "trace": rows[:: max(1, len(rows) // 40)],  # downsample for JSON
        "actions_7d": [a.tolist() for a in actions],
        "gates": list(memory.get("gates") or []),
    }


def trace_seed(env, post, book, seed_name: str) -> dict:
    seed = next(s for s in cs_mod.SEEDS if s["name"] == seed_name)
    actions, cuts, illegal = cs_mod.build_schedule(seed, book)
    if illegal or actions is None:
        return {"name": seed_name, "illegal": illegal}
    restore_capture(env, post["capture"])
    ref = post["ref"]
    rows = [sample(env.sim, ref)]
    lost = 0
    drop = False
    for i, a in enumerate(np.asarray(actions, np.float32)):
        step_legal(env.sim, a)
        r = sample(env.sim, ref)
        r["step"] = i
        # annotate cut tag
        tag = "run"
        for c in cuts:
            if i + 1 <= int(c["end"]):
                tag = c["tag"]
                break
        r["phase"] = tag
        rows.append(r)
        failed, lost = recovery_failed(env.sim, ref, lost)
        if failed:
            drop = True
            break
    # Prefer handoff after return if present
    trial = {"cuts": cuts, "summary": {"drop_step": None if not drop else len(rows)}}
    cuts_try = list(_priority_cuts(trial)) or [{"tag": "end", "end": len(actions)}]
    best = None
    for cut in cuts_try:
        end = int(cut["end"])
        restore_capture(env, post["capture"])
        acts = [np.asarray(a, np.float32) for a in actions[:end]]
        lost = 0
        ok = True
        for a in acts:
            step_legal(env.sim, a)
            failed, lost = recovery_failed(env.sim, ref, lost)
            if failed:
                ok = False
                break
        if not ok:
            continue
        meta = evaluate_handoff_from_state(env.sim, ref, stabilize_n=0)
        win = meta.get("window") or {}
        cand = {
            "cut": cut,
            "n": end,
            "SUCCESS_12S": bool(meta.get("reason") == "SUCCESS_12S"),
            "SUCCESS_NEW": bool(meta.get("SUCCESS_NEW")),
            "reason": meta.get("reason"),
            "pre_theta": meta.get("pre_theta"),
            "pre_pos_mm": meta.get("pre_pos_mm"),
            "longest_both_ok_s": win.get("longest_both_ok_s"),
            "mean_pos_final2": win.get("mean_pos"),
            "mean_theta_final2": win.get("mean_theta"),
        }
        if best is None or (cand["SUCCESS_NEW"], cand["SUCCESS_12S"], -(cand["pre_pos_mm"] or 1e9)) > (
            best["SUCCESS_NEW"],
            best["SUCCESS_12S"],
            -(best["pre_pos_mm"] or 1e9),
        ):
            best = cand
            best["actions"] = acts
    # gx during "relax" (slip) phase of full schedule
    relax = [r for r in rows if r.get("phase") == "relax"]
    reorient = [r for r in rows if r.get("phase") == "reorient"]
    return_rows = [r for r in rows if r.get("phase") == "return"]
    return {
        "name": seed_name,
        "parts": seed["parts"],
        "cuts": cuts,
        "drop_during_schedule": drop,
        "had_reorient": bool(reorient),
        "had_relax_slip": bool(relax),
        "had_return": bool(return_rows),
        "complete_open_loop_sequence": bool(reorient and relax and (return_rows or seed_name.endswith("neg"))),
        "max_abs_gx_relax": float(max(abs(r["gx"]) for r in relax)) if relax else None,
        "gx_end_reorient": reorient[-1]["gx"] if reorient else None,
        "d_finger_x_mm": float(rows[-1]["finger_x_mm"] - rows[0]["finger_x_mm"]) if len(rows) > 1 else None,
        "start": {k: rows[0][k] for k in ("e_pos_mm", "e_theta_deg", "e_signed_mm", "gx", "finger_x_mm")},
        "end_schedule": {k: rows[-1][k] for k in ("e_pos_mm", "e_theta_deg", "e_signed_mm", "gx", "finger_x_mm", "bilateral")},
        "best_handoff": {k: v for k, v in (best or {}).items() if k != "actions"},
        "actions_best": None if best is None else [a.tolist() for a in best["actions"]],
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    table, book48 = load_vocab()
    book = load_codebook()
    env = make_env()
    post = ensure_post_blob(env, table, book48)
    root = pickle.loads(ROOT_PKL.read_bytes())

    print("=== Plan A gravity-slip audit hard_910220 ===", flush=True)
    restore_capture(env, post["capture"])
    s0 = sample(env.sim, post["ref"])
    print(
        f"post-B1: e_pos={s0['e_pos_mm']:.2f} θ={s0['e_theta_deg']:.2f} "
        f"g_h={s0['g_h']} pinch_x={s0['pinch_x_mm']:.2f}mm finger_x={s0['finger_x_mm']:.2f}mm",
        flush=True,
    )

    mechanism_doc = {
        "slip_axis": "hand-frame x: object COM finger-axis offset rh[0]; gravity projection g_h[0]",
        "actively_reorients_gravity": True,
        "how": (
            "controlled_slide_action: phase rotate applies wrist pitch wy (±SLIDE_OMEGA=3 or ±SLIDE_SLOW=1 rad/s) "
            "until g_h[0] reaches gx_target = -sign(pinch_x)*min(6, |pinch_x|/0.012*6), or tilt accumulates 120°. "
            "Then slip unloads grip (grip_action(-5)), brake, return untilts toward stored R0."
        ),
        "inward_reseat": (
            "pitch with wx=1.12, wy=±2 until |g_h[0]|>=GX_SLIP=6 opposing finger offset; "
            "relax grip -0.48; retighten. No untilt phase."
        ),
        "seed_schedules": "open-loop center6_*: reorient(wy) → relax → brake → return (untilt)",
        "rates": {
            "SLIDE_OMEGA": SLIDE_OMEGA,
            "SLIDE_SLOW": SLIDE_SLOW,
            "SLIDE_TILT_MAX_deg": float(np.degrees(SLIDE_TILT_MAX)),
            "GX_SLIP": GX_SLIP,
            "EX_ENTER_m": EX_ENTER,
        },
        "prior_diagnostic": {
            "invoked_fb_controlled_slide": True,
            "invoked_fb_inward_reseat": True,
            "invoked_seed_schedules": True,
            "phase_instrumentation": False,
            "classification_if_no_phase_log": (
                "Prior negative SUCCESS_NEW is valid for outcome of invoked controllers, "
                "but without phase logs we must re-trace to confirm rotate→slip→brake→return completed."
            ),
        },
    }

    print("\n--- Trace controlled_slide_action ---", flush=True)
    t_cs = trace_feedback(env, post, "controlled_slide")
    print(
        f"  coverage={t_cs['coverage']} max|gx|_slip={t_cs['max_abs_gx_during_slip']:.2f} "
        f"turned={t_cs['max_turned_deg']:.1f}° d_rhx={t_cs['d_finger_x_mm']:.2f} "
        f"NEW={t_cs['SUCCESS_NEW']} S12={t_cs['SUCCESS_12S']} reason={t_cs['reason']}",
        flush=True,
    )

    print("\n--- Trace inward_reseat_action ---", flush=True)
    t_ir = trace_feedback(env, post, "inward_reseat")
    print(
        f"  coverage={t_ir['coverage']} max|gx|_slip={t_ir['max_abs_gx_during_slip']:.2f} "
        f"turned={t_ir['max_turned_deg']:.1f}° d_rhx={t_ir['d_finger_x_mm']:.2f} "
        f"NEW={t_ir['SUCCESS_NEW']} S12={t_ir['SUCCESS_12S']} reason={t_ir['reason']} gates={t_ir['gates']}",
        flush=True,
    )

    print("\n--- Trace seed schedules (open-loop full sequence) ---", flush=True)
    seed_rows = []
    for name in ("center6_pos", "center6_neg", "short_pos", "short_neg", "no_return_pos"):
        print(f"  seed {name} ...", flush=True)
        seed_rows.append(trace_seed(env, post, book, name))
        r = seed_rows[-1]
        bh = r.get("best_handoff") or {}
        print(
            f"    reorient={r.get('had_reorient')} relax={r.get('had_relax_slip')} return={r.get('had_return')} "
            f"gx_relax={r.get('max_abs_gx_relax')} NEW={bh.get('SUCCESS_NEW')} S12={bh.get('SUCCESS_12S')} "
            f"reason={bh.get('reason')} pre_pos={bh.get('pre_pos_mm')}",
            flush=True,
        )

    # Accept best SUCCESS_NEW if any
    best_new = None
    candidates = [
        ("fb:controlled_slide", t_cs.get("actions_7d"), t_cs),
        ("fb:inward_reseat", t_ir.get("actions_7d"), t_ir),
    ]
    for s in seed_rows:
        if s.get("actions_best") and (s.get("best_handoff") or {}).get("SUCCESS_NEW"):
            candidates.append((f"seed:{s['name']}", s["actions_best"], s["best_handoff"]))

    replay = None
    for name, acts, meta in candidates:
        if not acts:
            continue
        if not (meta.get("SUCCESS_NEW") if isinstance(meta, dict) else False):
            # re-check from actions
            restore_capture(env, post["capture"])
            for a in acts:
                step_legal(env.sim, np.asarray(a, np.float32))
            m = evaluate_handoff_from_state(env.sim, post["ref"], stabilize_n=0)
            if not m.get("SUCCESS_NEW"):
                continue
            meta = {**meta, "SUCCESS_NEW": True, "SUCCESS_12S": True, "reason": m["reason"]}
        # legalize + full IC replay
        arr = np.stack([np.asarray(a, np.float32) for a in acts], axis=0)
        tokens, _res, _dec, _ = legalize(arr, book)
        full_toks = list(B1_TOKENS) + [int(t) for t in tokens]
        print(f"  SUCCESS_NEW candidate {name}; exact_replay 3x n_tok={len(full_toks)}", flush=True)
        replay = exact_replay_3x(env, root, table, book48, full_toks, branch="B2", prefix_len=1)
        best_new = {"name": name, "tokens": full_toks, "meta": meta, "exact_replay": replay}
        if replay["ok"]:
            break

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "ic_id": IC,
        "post_state": s0,
        "mechanism": mechanism_doc,
        "controlled_slide_trace": {k: v for k, v in t_cs.items() if k not in ("actions_7d", "trace")},
        "controlled_slide_phase_log": t_cs["phase_log"],
        "inward_reseat_trace": {k: v for k, v in t_ir.items() if k not in ("actions_7d", "trace")},
        "inward_reseat_phase_log": t_ir["phase_log"],
        "seeds": [{k: v for k, v in s.items() if k != "actions_best"} for s in seed_rows],
        "best_SUCCESS_NEW": None
        if best_new is None
        else {
            "name": best_new["name"],
            "exact_replay": best_new["exact_replay"],
            "meta": {k: best_new["meta"].get(k) for k in ("SUCCESS_NEW", "SUCCESS_12S", "reason", "pre_pos_mm", "pre_theta", "longest_both_ok_s") if isinstance(best_new["meta"], dict)},
        },
        "verdict": None,  # filled below
    }

    cs_cov = t_cs["coverage"]["complete_rotate_slip_brake_return"]
    # Classify prior result
    if cs_cov:
        prior_class = (
            "complete_action_family_tested — prior/current feedback controlled_slide "
            "executed rotate→slip→brake→return; negative SUCCESS_NEW is an outcome of that family, "
            "not incomplete coverage"
        )
    else:
        prior_class = (
            "incomplete_action_family_coverage — feedback controller did not complete "
            "rotate→slip→brake→return on this IC; prior negative result must not be read as "
            "demonstrated physical infeasibility of the full gravity-slip family"
        )

    summary["verdict"] = {
        "plan_a_actively_reorients_gravity": True,
        "slip_axis": "hand_x (g_h[0] / rh[0])",
        "complete_sequence_tested_now": cs_cov,
        "prior_diagnostic_classification": prior_class,
        "SUCCESS_NEW": bool(best_new and (best_new.get("exact_replay") or {}).get("ok")),
        "failure_mechanism_if_any": None,
    }

    # Failure mechanism summary from traces
    if not summary["verdict"]["SUCCESS_NEW"]:
        summary["verdict"]["failure_mechanism_if_any"] = {
            "controlled_slide": {
                "phases": t_cs["phases_seen"],
                "gx_end_rotate": t_cs["gx_at_end_of_rotate"],
                "max_gx_slip": t_cs["max_abs_gx_during_slip"],
                "d_finger_x_mm": t_cs["d_finger_x_mm"],
                "end_e_signed_mm": (t_cs.get("end") or {}).get("e_signed_mm"),
                "end_theta": (t_cs.get("end") or {}).get("e_theta_deg"),
                "end_e_pos": (t_cs.get("end") or {}).get("e_pos_mm"),
                "handoff": t_cs["reason"],
                "interpretation": (
                    "If rotate reached gx_target and slip ran: hand-x may improve while hand-z/θ worsen; "
                    "SUCCESS_NEW needs joint hold. If rotate capped without |gx| target: tilt alone "
                    "insufficient or contact lost. If early done: |pinch_x|<EX_ENTER so controller skipped."
                ),
            },
            "inward_reseat": {
                "phases": t_ir["phases_seen"],
                "gates": t_ir["gates"],
                "max_gx_slip": t_ir["max_abs_gx_during_slip"],
                "handoff": t_ir["reason"],
            },
        }

    # Persist traces
    (OUT / "controlled_slide_trace.json").write_text(
        json.dumps(jsafe({k: v for k, v in t_cs.items() if k != "actions_7d"}), indent=2),
        encoding="utf-8",
    )
    (OUT / "inward_reseat_trace.json").write_text(
        json.dumps(jsafe({k: v for k, v in t_ir.items() if k != "actions_7d"}), indent=2),
        encoding="utf-8",
    )
    (OUT / "summary.json").write_text(json.dumps(jsafe(summary), indent=2), encoding="utf-8")

    lines = [
        "# Plan A Gravity-Slip Audit — hard_910220",
        "",
        f"Created UTC: {summary['created_utc']}",
        "",
        "## Mechanism (from code)",
        "",
        f"- Slip axis: **{mechanism_doc['slip_axis']}**",
        f"- Actively reorients gravity: **YES**",
        f"- controlled_slide: {mechanism_doc['how']}",
        f"- inward_reseat: {mechanism_doc['inward_reseat']}",
        "",
        "## Post-B1 state",
        "",
        f"- e_pos={s0['e_pos_mm']:.2f} mm, θ={s0['e_theta_deg']:.2f}°",
        f"- g_h={s0['g_h']}",
        f"- pinch_x={s0['pinch_x_mm']:.2f} mm, finger_x={s0['finger_x_mm']:.2f} mm",
        f"- e_signed_mm={s0['e_signed_mm']}",
        "",
        "## Feedback controlled_slide phase coverage",
        "",
        f"- phases: `{t_cs['phases_seen']}`",
        f"- complete rotate→slip→brake→return: **{cs_cov}**",
        f"- max |gx| during slip: {t_cs['max_abs_gx_during_slip']:.3f}",
        f"- gx at end of rotate: {t_cs['gx_at_end_of_rotate']}",
        f"- max turned: {t_cs['max_turned_deg']:.1f}°",
        f"- Δfinger_x: {t_cs['d_finger_x_mm']:.2f} mm, Δe_pos: {t_cs['d_e_pos_mm']:.2f} mm",
        f"- end: {t_cs.get('end')}",
        f"- HANDOFF: NEW={t_cs['SUCCESS_NEW']} S12={t_cs['SUCCESS_12S']} reason={t_cs['reason']} "
        f"pre_θ={t_cs['pre_theta']} pre_pos={t_cs['pre_pos_mm']} hold={t_cs['longest_both_ok_s']}",
        "",
        "## inward_reseat",
        "",
        f"- phases: `{t_ir['phases_seen']}` gates={t_ir['gates']}",
        f"- max |gx| slip: {t_ir['max_abs_gx_during_slip']:.3f}, Δfinger_x={t_ir['d_finger_x_mm']:.2f}",
        f"- HANDOFF: NEW={t_ir['SUCCESS_NEW']} S12={t_ir['SUCCESS_12S']} reason={t_ir['reason']}",
        "",
        "## Seed schedules",
        "",
    ]
    for s in seed_rows:
        bh = s.get("best_handoff") or {}
        lines.append(
            f"- `{s['name']}`: reorient={s.get('had_reorient')} relax={s.get('had_relax_slip')} "
            f"return={s.get('had_return')} |gx|_relax={s.get('max_abs_gx_relax')} "
            f"NEW={bh.get('SUCCESS_NEW')} S12={bh.get('SUCCESS_12S')} reason={bh.get('reason')} "
            f"pre_pos={bh.get('pre_pos_mm')}"
        )
    lines += [
        "",
        "## Verdict",
        "",
        f"- {summary['verdict']['prior_diagnostic_classification']}",
        f"- SUCCESS_NEW+3/3: {summary['verdict']['SUCCESS_NEW']}",
        "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(jsafe(summary["verdict"]), indent=2), flush=True)
    print(f"Wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
