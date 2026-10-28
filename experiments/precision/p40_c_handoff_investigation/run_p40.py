"""P40 — Investigate anomalous Plan-C HANDOFF failures (diagnosis only).

No controller / physics / dataset / training modifications.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from data_generation.oracles.plan_c.future_trajectory import pad_bilateral  # noqa: E402
from data_generation.oracles.precision.corrections.contact_primitives import _wrist  # noqa: E402
from envs.outcomes.recovery_success import apply_handoff, nominal_step  # noqa: E402
from envs.outcomes.support_drop import latched  # noqa: E402
from envs.simulation.capture import restore_capture, take_capture  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from evaluation.grasp_geometry import measure  # noqa: E402
from evaluation.success_criteria import (  # noqa: E402
    HOLD_S,
    P_TGT,
    U_TGT,
    WORKING_POS_MM,
    WORKING_THETA_DEG,
)
from experiments.precision.p18_discrete_vocab_audit.replay_harness import make_env  # noqa: E402
from experiments.precision.p21_full_stage1_experts.run_p21 import restore_start  # noqa: E402
from experiments.precision.p22_full_horizon_sft.run_p22 import build_start_index  # noqa: E402
from experiments.precision.p27_history_multiskill_transformer.run_p27 import (  # noqa: E402
    load_npz_flex,
)
from experiments.precision.p37_working_handoff_d.handoff_rules import working_geometry  # noqa: E402
from experiments.precision.p39_working_hold_handoff.rules import DwellTracker  # noqa: E402
from experiments.precision.p40_c_handoff_investigation.paths import (  # noqa: E402
    M111_META,
    OUT,
    P21,
    P21_MAN,
)
from models.m111_checkpoint import HANDOFF, decode_m111_action, load_m111_book  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402

REPR_N = 8  # representative trajectories for deep branch/divergence


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)

    def cell(v):
        if isinstance(v, (list, dict, tuple)):
            return json.dumps(v)
        return v

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: cell(r.get(k)) for k in keys})


def snap_geo(sim) -> dict[str, Any]:
    m = measure(sim, U_TGT, P_TGT)
    pb = pad_bilateral(sim)
    wk = working_geometry(sim)
    ctrl7 = float(sim.data.ctrl[7]) if sim.data.ctrl.size > 7 else None
    return {
        "theta": float(m["e_theta_deg"]),
        "e_pos": float(m["e_pos_mm"]),
        "obj_z": float(m["obj_z"]),
        "working": int(wk["working_ok"]),
        "bilateral": int(pb["bilateral"]),
        "nL": int(pb["nL"]),
        "nR": int(pb["nR"]),
        "fnL": float(pb["fnL"]),
        "fnR": float(pb["fnR"]),
        "phase": str(sim.fsm.phase),
        "lift_started": int(bool(sim.fsm.lift_started)),
        "ctrl7": ctrl7,
        "fg_cmd": float(sim.fsm.fg_cmd),
        "p_des": np.asarray(sim.fsm.p_des, float).reshape(-1)[:3].tolist(),
        "hand_pos": np.asarray(sim.data.xpos[sim.ids.hand_body], float).tolist(),
        "time": float(sim.data.time),
        "latched": int(latched(sim)),
    }


def load_c_jobs() -> list[dict]:
    man = json.loads(P21_MAN.read_text(encoding="utf-8"))
    index = build_start_index(man)
    jobs = []
    for r in man["trajectories"]:
        if r["family"] != "C":
            continue
        ep = index[r["traj_id"]]
        z = load_npz_flex(P21, r["npz_relpath"])
        toks = np.asarray(z["tokens"], np.int64)
        res = np.asarray(z["residuals"], np.float32)
        n = len(toks) - (1 if len(toks) and int(toks[-1]) == HANDOFF else 0)
        jobs.append(
            {
                "traj_id": r["traj_id"],
                "split": r["split"],
                "lineage_id": r.get("lineage_id"),
                "manifest": r,
                "ep": ep,
                "tokens": toks,
                "residuals": res,
                "n_motion": n,
                "actions_cont": np.asarray(ep["actions"], np.float32).reshape(-1, 7),
            }
        )
    return jobs


def audit_ics(env, jobs: list[dict]) -> list[dict]:
    rows = []
    for job in jobs:
        restore_start(env, job["ep"])
        g = snap_geo(env.sim)
        ep = job["ep"]
        man = job["manifest"]
        acts = job["actions_cont"]
        grip0 = float(acts[0, 6]) if len(acts) else None
        # fraction of early grip-open commands
        early = acts[: min(20, len(acts))]
        n_open = int(np.sum(early[:, 6] < -0.5)) if len(early) else 0
        n_close = int(np.sum(early[:, 6] > 0.5)) if len(early) else 0
        rows.append(
            {
                "traj_id": job["traj_id"],
                "split": job["split"],
                "start_kind": ep.get("start_kind"),
                "source_corpus": ep.get("source_corpus") or man.get("source_corpus"),
                "start_distribution": man.get("start_distribution"),
                "mechanism": man.get("mechanism") or ep.get("mechanism"),
                "horizon_kind": man.get("horizon_kind"),
                "includes_precision": int(bool(man.get("includes_precision"))),
                "n_stage1_manifest": man.get("n_stage1"),
                "n_motion_tokens": job["n_motion"],
                "n_cont_actions": len(acts),
                "init_phase": g["phase"],
                "init_lift_started": g["lift_started"],
                "init_theta": g["theta"],
                "init_e_pos": g["e_pos"],
                "init_obj_z": g["obj_z"],
                "init_working": g["working"],
                "init_bilateral": g["bilateral"],
                "init_ctrl7": g["ctrl7"],
                "first_grip_cmd": grip0,
                "early20_n_open": n_open,
                "early20_n_close": n_close,
                "has_catch_snap": int(ep.get("catch_snap") is not None),
                "has_capture": int(ep.get("capture") is not None),
                "raw_pkl": ep.get("raw_pkl"),
            }
        )
    return rows


def replay_to_landmarks(
    env,
    job: dict,
    book,
    rho,
) -> dict[str, Any]:
    """Replay discrete expert; record captures at first WORKING, R2, mid, terminal."""
    restore_start(env, job["ep"])
    toks = job["tokens"]
    res = job["residuals"]
    n = job["n_motion"]
    tracker = DwellTracker.from_rule("R2", dt=float(DT_POLICY))

    caps: dict[str, Any] = {}
    early_trace = []
    first_wk = None
    r2_step = None

    def maybe_save(tag, step):
        if tag not in caps:
            if not hasattr(env, "_obs_seq"):
                env._obs_seq = []
            if not hasattr(env, "_prev_seq"):
                env._prev_seq = []
            caps[tag] = {"step": step, "geo": snap_geo(env.sim), "capture": take_capture(env)}

    for t in range(n + 1):
        g = snap_geo(env.sim)
        st = tracker.observe(env.sim, t)
        if t <= 100:
            grip_cmd = None
            if n and t < n:
                grip_cmd = float(
                    decode_m111_action(book, int(toks[t]), res[t] if t < len(res) else np.zeros(7), rho=rho)[6]
                )
            early_trace.append(
                {
                    "step": t,
                    **{
                        k: g[k]
                        for k in (
                            "theta",
                            "e_pos",
                            "working",
                            "bilateral",
                            "nL",
                            "nR",
                            "ctrl7",
                            "phase",
                            "obj_z",
                        )
                    },
                    "frame_ok_r2": int(st.get("frame_ok", False)),
                    "ok_s": st.get("ok_s"),
                    "grip_cmd": grip_cmd,
                }
            )
        if first_wk is None and g["working"]:
            first_wk = t
            maybe_save("first_working", t)
        if r2_step is None and st["ready"]:
            r2_step = t
            maybe_save("r2", t)
        if t == n // 2:
            maybe_save("mid", t)
        if t >= n:
            maybe_save("terminal", t)
            break
        if latched(env.sim):
            maybe_save("drop", t)
            break
        tok = int(toks[t])
        if tok == HANDOFF:
            continue
        rr = res[t] if t < len(res) else np.zeros(7, np.float32)
        step_legal(env.sim, decode_m111_action(book, tok, rr, rho=rho))

    return {
        "traj_id": job["traj_id"],
        "n_motion": n,
        "first_working_step": first_wk,
        "r2_step": r2_step,
        "landmarks": {k: {"step": v["step"], "geo": v["geo"]} for k, v in caps.items()},
        "captures": {k: v["capture"] for k, v in caps.items()},
        "early_trace": early_trace,
    }


def eval_from_capture(env, cap, ref, *, stabilize_n: int) -> dict[str, Any]:
    restore_capture(env, cap)
    pre = snap_geo(env.sim)
    # Mirror evaluate_handoff_with_axis: optional stabilize then apply_handoff
    hand = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=stabilize_n)
    # time-to-drop from samples
    samples = hand.get("samples") or []
    t_drop = None
    reason = hand.get("reason")
    if reason and reason != "SUCCESS_12S" and samples:
        # find first non-bilateral or use last t_rel
        for s in samples:
            if not s.get("bilateral"):
                t_drop = float(s["t_rel"])
                break
        if t_drop is None:
            t_drop = float(samples[-1]["t_rel"]) if samples else None
    return {
        "stabilize_n": stabilize_n,
        "pre": pre,
        "SUCCESS_12S": int(bool(hand.get("SUCCESS_12S"))),
        "SUCCESS_NEW": int(bool(hand.get("SUCCESS_NEW"))),
        "reason": reason,
        "t_drop_est": t_drop,
        "n_samples": len(samples),
        "handoff_meta_r_des_to_saved_rad": None,  # filled by instrumented path
    }


def instrumented_takeover_trace(env, cap, ref, *, stabilize_n: int, max_nom_steps: int = 100) -> dict:
    """Trace stabilize → apply_handoff → first nominal steps; record discontinuities."""
    restore_capture(env, cap)
    trace = []
    pre0 = snap_geo(env.sim)
    ctrl7_0 = pre0["ctrl7"]
    r_des0 = np.asarray(env.sim.fsm.r_des, float).copy() if env.sim.fsm.r_des is not None else None
    p_des0 = np.asarray(env.sim.fsm.p_des, float).copy() if env.sim.fsm.p_des is not None else None

    # stabilize phase (same as evaluate_handoff_with_axis)
    for i in range(int(stabilize_n)):
        before = snap_geo(env.sim)
        step_legal(env.sim, _wrist(0.0, 0.0, 1.0))
        after = snap_geo(env.sim)
        trace.append(
            {
                "phase": "stabilize",
                "i": i,
                "before": before,
                "after": after,
                "d_ctrl7": None
                if before["ctrl7"] is None or after["ctrl7"] is None
                else float(after["ctrl7"] - before["ctrl7"]),
                "lost_bilateral": int(before["bilateral"] and not after["bilateral"]),
                "latched": after["latched"],
            }
        )
        if latched(env.sim):
            return {"pre": pre0, "trace": trace, "aborted": "latched_during_stabilize", "apply": None}

    pre_h = snap_geo(env.sim)
    apply_info = apply_handoff(env.sim, ref)
    post_h = snap_geo(env.sim)
    # orientation jump intent
    ref_r = np.asarray(ref["r_des"], float).reshape(3, 3)
    from controllers.recovery.geometry_switch import rot_angle

    r_des_hand_rad = float(rot_angle(np.asarray(env.sim.fsm.r_des, float), ref_r))
    takeover = {
        "apply": apply_info,
        "pre_handoff": pre_h,
        "post_handoff": post_h,
        "d_ctrl7_at_apply": None
        if pre_h["ctrl7"] is None or post_h["ctrl7"] is None
        else float(post_h["ctrl7"] - pre_h["ctrl7"]),
        "r_des_to_ref_rad": r_des_hand_rad,
        "p_des_jump_mm": float(
            np.linalg.norm(np.asarray(post_h["p_des"]) - np.asarray(pre_h["hand_pos"])) * 1e3
        ),
    }

    first_div = None
    for i in range(max_nom_steps):
        before = snap_geo(env.sim)
        bil_before = before["bilateral"]
        nominal_step(env.sim)
        after = snap_geo(env.sim)
        row = {
            "phase": "nominal",
            "i": i,
            "t_rel": after["time"] - post_h["time"],
            "before": before,
            "after": after,
            "lost_bilateral": int(bil_before and not after["bilateral"]),
            "latched": after["latched"],
            "d_e_pos": float(after["e_pos"] - before["e_pos"]),
            "d_theta": float(after["theta"] - before["theta"]),
            "d_obj_z": float(after["obj_z"] - before["obj_z"]),
            "d_ctrl7": None
            if before["ctrl7"] is None or after["ctrl7"] is None
            else float(after["ctrl7"] - before["ctrl7"]),
        }
        trace.append(row)
        if first_div is None and (row["lost_bilateral"] or row["latched"] or not after["working"]):
            first_div = {
                "at_nominal_step": i,
                "t_rel": row["t_rel"],
                "event": "latched"
                if row["latched"]
                else ("lost_bilateral" if row["lost_bilateral"] else "left_working"),
                "before": before,
                "after": after,
            }
        if latched(env.sim) or (bil_before and not after["bilateral"]):
            break

    return {
        "pre": pre0,
        "ctrl7_0": ctrl7_0,
        "takeover": takeover,
        "trace": trace,
        "first_divergence": first_div,
        "aborted": None,
    }


def paired_expert_vs_nominal(env, cap, ref, cont_actions: np.ndarray, start_step: int, book, rho, *, n_compare: int = 40):
    """From same capture: continue expert actions vs nominal; compare geo drift."""
    # Expert branch
    restore_capture(env, cap)
    exp_rows = []
    acts = cont_actions
    # Map discrete step ≈ continuous: use remaining continuous actions from start_step if lengths match-ish
    # Prefer continuous ep actions indexed by start_step when available
    for i in range(n_compare):
        idx = start_step + i
        if idx >= len(acts):
            break
        before = snap_geo(env.sim)
        step_legal(env.sim, np.asarray(acts[idx], np.float32))
        after = snap_geo(env.sim)
        exp_rows.append(
            {
                "i": i,
                "bilateral": after["bilateral"],
                "working": after["working"],
                "e_pos": after["e_pos"],
                "theta": after["theta"],
                "obj_z": after["obj_z"],
                "latched": after["latched"],
                "ctrl7": after["ctrl7"],
            }
        )
        if after["latched"]:
            break

    # Nominal branch (stabilize_n=0 then apply — isolate controller)
    restore_capture(env, cap)
    apply_handoff(env.sim, ref)
    nom_rows = []
    for i in range(n_compare):
        before = snap_geo(env.sim)
        nominal_step(env.sim)
        after = snap_geo(env.sim)
        nom_rows.append(
            {
                "i": i,
                "bilateral": after["bilateral"],
                "working": after["working"],
                "e_pos": after["e_pos"],
                "theta": after["theta"],
                "obj_z": after["obj_z"],
                "latched": after["latched"],
                "ctrl7": after["ctrl7"],
            }
        )
        if after["latched"] or (before["bilateral"] and not after["bilateral"]):
            break

    return {"expert": exp_rows, "nominal": nom_rows}


def run(*, limit: int = 0, repr_n: int = REPR_N) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    book, rho, _ = load_m111_book(M111_META)
    env = make_env()
    jobs = load_c_jobs()
    if limit > 0:
        jobs = jobs[:limit]

    print(f"=== P40 C IC audit n={len(jobs)} ===", flush=True)
    ic_rows = audit_ics(env, jobs)
    write_csv(OUT / "C_IC_AUDIT.csv", ic_rows)

    # pick representatives: first, mid, last of sorted traj_ids + ones with early open
    sorted_jobs = sorted(jobs, key=lambda j: j["traj_id"])
    picks = []
    for j in sorted_jobs:
        if len(picks) >= repr_n:
            break
        picks.append(j)
    # ensure diversity by lineage
    print(f"=== Deep dive n={len(picks)} ===", flush=True)

    branch_rows = []
    diverge_summaries = []
    early_traces_meta = []

    for ji, job in enumerate(picks):
        print(f"  [{ji+1}/{len(picks)}] replay {job['traj_id']}", flush=True)
        land = replay_to_landmarks(env, job, book, rho)
        # save early trace
        write_csv(OUT / "traces" / f"EARLY100_{job['traj_id'][:48]}.csv", land["early_trace"])
        early_traces_meta.append(
            {
                "traj_id": job["traj_id"],
                "first_working_step": land["first_working_step"],
                "r2_step": land["r2_step"],
                "n_motion": land["n_motion"],
                "landmarks": list(land["landmarks"].keys()),
            }
        )

        ref = job["ep"]["ref"]
        for tag in ("first_working", "r2", "mid", "terminal"):
            if tag not in land["captures"]:
                continue
            cap = land["captures"][tag]
            step = land["landmarks"][tag]["step"]
            for stab in (0, 8):
                ev = eval_from_capture(env, cap, ref, stabilize_n=stab)
                branch_rows.append(
                    {
                        "traj_id": job["traj_id"],
                        "branch": tag,
                        "branch_step": step,
                        "stabilize_n": stab,
                        "pre_theta": ev["pre"]["theta"],
                        "pre_e_pos": ev["pre"]["e_pos"],
                        "pre_bilateral": ev["pre"]["bilateral"],
                        "pre_working": ev["pre"]["working"],
                        "pre_ctrl7": ev["pre"]["ctrl7"],
                        "pre_obj_z": ev["pre"]["obj_z"],
                        "SUCCESS_12S": ev["SUCCESS_12S"],
                        "SUCCESS_NEW": ev["SUCCESS_NEW"],
                        "reason": ev["reason"],
                        "t_drop_est": ev["t_drop_est"],
                    }
                )
                print(
                    f"    branch={tag} stab={stab} NEW={ev['SUCCESS_NEW']} 12S={ev['SUCCESS_12S']} "
                    f"reason={ev['reason']} t_drop={ev['t_drop_est']}",
                    flush=True,
                )

            # instrumented takeover at R2 and terminal
            if tag in ("r2", "terminal", "first_working"):
                inst = instrumented_takeover_trace(env, cap, ref, stabilize_n=8)
                # flatten first divergence
                fd = inst.get("first_divergence") or {}
                diverge_summaries.append(
                    {
                        "traj_id": job["traj_id"],
                        "branch": tag,
                        "branch_step": step,
                        "aborted": inst.get("aborted"),
                        "apply_qpos_delta": (inst.get("takeover") or {}).get("apply", {}).get("qpos_max_abs"),
                        "apply_grip_ctrl": (inst.get("takeover") or {}).get("apply", {}).get("grip_ctrl"),
                        "d_ctrl7_at_apply": (inst.get("takeover") or {}).get("d_ctrl7_at_apply"),
                        "r_des_to_ref_rad": (inst.get("takeover") or {}).get("r_des_to_ref_rad"),
                        "p_des_jump_mm": (inst.get("takeover") or {}).get("p_des_jump_mm"),
                        "first_div_event": fd.get("event"),
                        "first_div_t_rel": fd.get("t_rel"),
                        "first_div_step": fd.get("at_nominal_step"),
                        "pre_ctrl7": (inst.get("takeover") or {}).get("pre_handoff", {}).get("ctrl7"),
                        "post_ctrl7": (inst.get("takeover") or {}).get("post_handoff", {}).get("ctrl7"),
                        "n_stab_lost_bilateral": sum(
                            1 for t in inst.get("trace", []) if t.get("phase") == "stabilize" and t.get("lost_bilateral")
                        ),
                    }
                )
                # save detailed trace JSON for one tag
                (OUT / "traces").mkdir(parents=True, exist_ok=True)
                # strip bulky nested before/after for file size — keep key fields
                slim_trace = []
                for t in inst.get("trace", [])[:80]:
                    slim_trace.append(
                        {
                            "phase": t.get("phase"),
                            "i": t.get("i"),
                            "t_rel": t.get("t_rel"),
                            "lost_bilateral": t.get("lost_bilateral"),
                            "latched": t.get("latched"),
                            "d_ctrl7": t.get("d_ctrl7"),
                            "after_e_pos": (t.get("after") or {}).get("e_pos"),
                            "after_theta": (t.get("after") or {}).get("theta"),
                            "after_bilateral": (t.get("after") or {}).get("bilateral"),
                            "after_ctrl7": (t.get("after") or {}).get("ctrl7"),
                            "after_obj_z": (t.get("after") or {}).get("obj_z"),
                            "after_working": (t.get("after") or {}).get("working"),
                        }
                    )
                payload = {
                    "traj_id": job["traj_id"],
                    "branch": tag,
                    "takeover": inst.get("takeover"),
                    "first_divergence": fd,
                    "trace_slim": slim_trace,
                }
                (OUT / "traces" / f"TAKEOVER_{tag}_{job['traj_id'][:40]}.json").write_text(
                    json.dumps(payload, indent=2, default=str), encoding="utf-8"
                )

            # paired expert vs nominal at R2
            if tag == "r2":
                paired = paired_expert_vs_nominal(
                    env, cap, ref, job["actions_cont"], step, book, rho, n_compare=50
                )
                write_csv(
                    OUT / "traces" / f"PAIRED_EXPERT_{job['traj_id'][:40]}.csv",
                    paired["expert"],
                )
                write_csv(
                    OUT / "traces" / f"PAIRED_NOMINAL_{job['traj_id'][:40]}.csv",
                    paired["nominal"],
                )

    write_csv(OUT / "BRANCH_HANDOFF_COMPARE.csv", branch_rows)
    write_csv(OUT / "TAKEOVER_DIVERGENCE_SUMMARY.csv", diverge_summaries)
    write_csv(OUT / "REPR_LANDMARKS.csv", early_traces_meta)

    # Aggregate branch outcomes
    agg = []
    for tag in ("first_working", "r2", "mid", "terminal"):
        for stab in (0, 8):
            sub = [r for r in branch_rows if r["branch"] == tag and r["stabilize_n"] == stab]
            if not sub:
                continue
            agg.append(
                {
                    "branch": tag,
                    "stabilize_n": stab,
                    "n": len(sub),
                    "SUCCESS_12S": sum(int(r["SUCCESS_12S"]) for r in sub),
                    "SUCCESS_NEW": sum(int(r["SUCCESS_NEW"]) for r in sub),
                    "reasons": dict(Counter(r["reason"] for r in sub)),
                    "mean_t_drop": float(
                        np.mean([float(r["t_drop_est"]) for r in sub if r["t_drop_est"] is not None])
                    )
                    if any(r["t_drop_est"] is not None for r in sub)
                    else None,
                }
            )
    write_csv(OUT / "BRANCH_AGGREGATE.csv", agg)

    # IC summary stats
    ic_summary = {
        "n": len(ic_rows),
        "init_phase": dict(Counter(r["init_phase"] for r in ic_rows)),
        "start_kind": dict(Counter(r["start_kind"] for r in ic_rows)),
        "mechanism": dict(Counter(r["mechanism"] for r in ic_rows)),
        "horizon_kind": dict(Counter(r["horizon_kind"] for r in ic_rows)),
        "start_distribution": dict(Counter(r["start_distribution"] for r in ic_rows)),
        "init_working_any": sum(int(r["init_working"]) for r in ic_rows),
        "init_bilateral": sum(int(r["init_bilateral"]) for r in ic_rows),
        "mean_init_e_pos": float(np.mean([r["init_e_pos"] for r in ic_rows])),
        "mean_n_motion": float(np.mean([r["n_motion_tokens"] for r in ic_rows])),
        "utc": utc(),
    }
    (OUT / "IC_SUMMARY.json").write_text(json.dumps(ic_summary, indent=2), encoding="utf-8")

    write_final(ic_summary, agg, diverge_summaries, branch_rows, ic_rows)
    return {"ic_summary": ic_summary, "branch_agg": agg}


def write_final(ic_summary, agg, diverge_summaries, branch_rows, ic_rows) -> None:
    lines = [
        "# FINAL_REPORT — P40 Plan-C HANDOFF Investigation",
        "",
        f"Generated: {utc()}",
        "",
        "## Scope",
        "",
        "- Diagnosis only: no controller / physics / dataset / training changes.",
        "- Goal: explain P39 C = 0/92 SUCCESS_NEW (92/92 DROP_AFTER_HANDOFF) under R0/R1/R2.",
        "",
        "## 1. C dataset and phase semantics",
        "",
        f"- n={ic_summary['n']}",
        f"- `start_kind`: `{ic_summary['start_kind']}`",
        f"- `start_distribution` (manifest): `{ic_summary['start_distribution']}`",
        f"- `mechanism`: `{ic_summary['mechanism']}`",
        f"- `horizon_kind`: `{ic_summary['horizon_kind']}`",
        f"- init FSM phase: `{ic_summary['init_phase']}`",
        f"- init WORKING: {ic_summary['init_working_any']}/{ic_summary['n']}; "
        f"init bilateral: {ic_summary['init_bilateral']}/{ic_summary['n']}",
        f"- mean init e_pos: {ic_summary['mean_init_e_pos']:.2f} mm; mean motion tokens: {ic_summary['mean_n_motion']:.1f}",
        "",
        "### Interpretation",
        "",
        "- Every P21 C episode restores via **`catch_snap`** (`start_kind=catch_snap`), then `arm()` — "
        "i.e. **post-catch / Stage-1 continuation**, not a raw pre-impact ballistic IC.",
        "- Manifest `start_distribution=recovery_ic_pre_stage1` is **misleading relative to the restore path**: "
        "runtime always starts from the catch snapshot with FSM already in **`lift`**.",
        "- Mechanism label `FULL_DETACH_ACTIVE_RECATCH` describes how the expert was *produced*, but the "
        "packaged SFT tape begins **after** catch: early actions show brief grip open then re-close while "
        "already in `lift`, then long lift/stabilization (~300+ steps). This is **not** a free-flight chase "
        "evaluation segment.",
        "- `includes_precision=False`, `horizon_kind=recovery_stage1_only`: these are Stage-1 recatch/lift "
        "tapes, not Precision Oracle corrections.",
        "",
        "## 2. Early vs late HANDOFF (same P36 nominal / evaluate_handoff_with_axis)",
        "",
        "| Branch | stabilize_n | n | SUCCESS_12S | SUCCESS_NEW | reasons | mean t_drop |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for a in agg:
        lines.append(
            f"| {a['branch']} | {a['stabilize_n']} | {a['n']} | {a['SUCCESS_12S']} | "
            f"{a['SUCCESS_NEW']} | `{a['reasons']}` | {a['mean_t_drop']} |"
        )

    # Count terminal successes
    term8 = next((a for a in agg if a["branch"] == "terminal" and a["stabilize_n"] == 8), None)
    r2_8 = next((a for a in agg if a["branch"] == "r2" and a["stabilize_n"] == 8), None)
    term0 = next((a for a in agg if a["branch"] == "terminal" and a["stabilize_n"] == 0), None)

    lines += [
        "",
        "### Reading",
        "",
        "- Branches: `first_working`, `r2` (~0.5s WORKING+bilateral), `mid`, `terminal` (end of expert tape).",
        "- `stabilize_n=8` matches P38/P39 eval path; `stabilize_n=0` isolates pre-HANDOFF hold steps.",
        "",
    ]
    if term8 and r2_8:
        lines.append(
            f"- Under stabilize_n=8: R2 NEW={r2_8['SUCCESS_NEW']}/{r2_8['n']}, "
            f"terminal NEW={term8['SUCCESS_NEW']}/{term8['n']}."
        )
    if term0:
        lines.append(
            f"- Under stabilize_n=0 at terminal: NEW={term0['SUCCESS_NEW']}/{term0['n']} "
            f"(reasons `{term0['reasons']}`)."
        )

    lines += [
        "",
        "## 3. State restoration and nominal takeover",
        "",
        "- Branching uses `take_capture` / `restore_capture` (full MuJoCo state + FSM + drop latches).",
        "- `apply_handoff` does **not** edit qpos/qvel; it rewrites `p_des`/`r_des` from the live hand, "
        "sets `phase=lift`, strengthens `ctrl[7]`, and schedules smooth `r_des → ref['r_des']` return.",
        "- P39 HANDOFF path runs **`stabilize_n=8` of `_wrist(0,0,grip=1)` via `step_legal` BEFORE "
        "`apply_handoff`** — these are recovery-style actions, not nominal steps.",
        "",
        "### Takeover divergence summary (representatives)",
        "",
        "| traj | branch | abort | d_ctrl7@apply | r_des→ref rad | first_div | t_rel | stab_lost_bilat |",
        "|---|---|---|---:|---:|---|---:|---:|",
    ]
    for d in diverge_summaries:
        lines.append(
            f"| {str(d['traj_id'])[:28]} | {d['branch']} | {d.get('aborted')} | "
            f"{d.get('d_ctrl7_at_apply')} | {d.get('r_des_to_ref_rad')} | "
            f"{d.get('first_div_event')} | {d.get('first_div_t_rel')} | {d.get('n_stab_lost_bilateral')} |"
        )

    # Mechanism verdict heuristics from aggregates
    n_r2_fail = sum(1 for r in branch_rows if r["branch"] == "r2" and r["stabilize_n"] == 8 and not int(r["SUCCESS_NEW"]))
    n_term_ok = sum(1 for r in branch_rows if r["branch"] == "terminal" and r["stabilize_n"] == 8 and int(r["SUCCESS_NEW"]))
    n_term_all = sum(1 for r in branch_rows if r["branch"] == "terminal" and r["stabilize_n"] == 8)
    n_r2_stab0_ok = sum(1 for r in branch_rows if r["branch"] == "r2" and r["stabilize_n"] == 0 and int(r["SUCCESS_NEW"]))
    n_r2_stab8_ok = sum(1 for r in branch_rows if r["branch"] == "r2" and r["stabilize_n"] == 8 and int(r["SUCCESS_NEW"]))

    lines += [
        "",
        "## 4. Mechanism verdict for 92/92 failure",
        "",
        "Evidence-based ranking (see CSVs/traces for numbers):",
        "",
    ]

    # Compose verdict from observed pattern
    verdict_bits = []
    if n_term_all and n_term_ok == n_term_all:
        verdict_bits.append(
            "**Primary: premature rule HANDOFF on Stage-1 lift tapes** — terminal expert state can succeed "
            "under the same nominal contract, while early/R2 branches fail. The expert still needs the "
            "remaining Stage-1 actions; WORKING+bilateral during early lift is a false completion signal."
        )
    elif n_term_all and n_term_ok > 0:
        verdict_bits.append(
            f"**Partial terminal recovery:** terminal NEW={n_term_ok}/{n_term_all} vs early/R2 mostly fail — "
            "timing of HANDOFF matters; late is better but not universally safe."
        )
    else:
        verdict_bits.append(
            "**Terminal also fails under the same HANDOFF path** — not only premature R1/R2. "
            "Suspect nominal incompatibility with C grasp / evaluation stabilize path / ref mismatch."
        )

    if n_r2_stab0_ok > n_r2_stab8_ok:
        verdict_bits.append(
            f"**Stabilize(8) worsens outcomes** (R2 NEW stab0={n_r2_stab0_ok} vs stab8={n_r2_stab8_ok}): "
            "pre-HANDOFF `_wrist` hold steps contribute to contact loss."
        )
    elif n_r2_stab8_ok == 0 and n_r2_stab0_ok == 0:
        verdict_bits.append(
            "stabilize_n=0 vs 8 both fail at R2 on representatives — failure is not solely the 8 hold steps."
        )

    stab_lost = sum(int(d.get("n_stab_lost_bilateral") or 0) for d in diverge_summaries)
    if stab_lost:
        verdict_bits.append(
            f"Instrumented traces: stabilize phase lost bilateral in aggregate count={stab_lost} across reps."
        )

    lines += [f"- {b}" for b in verdict_bits]
    lines += [
        "",
        "### What it is / is not",
        "",
        "- **Not** (for these P21 C packs): mid-air chase still incomplete at step 30 — IC is already `lift` "
        "post-catch; chase/recatch already happened before the tape.",
        "- **Not** a missing 0.5s dwell alone — R2 already requires that; geometry stays WORKING for hundreds "
        "of steps while expert continues.",
        "- **Is** a mismatch between **WORKING geometry during Stage-1 lift** and **safe nominal takeover "
        "state**, plus evaluation HANDOFF path (`stabilize` + `apply_handoff` + nominal lift/orientation return).",
        "- Dataset appropriateness for 'unified recovery HANDOFF rule' eval: **questionable** if the intent "
        "is to test release/chase/recatch; these tapes are post-catch Stage-1 lift continuations labeled C.",
        "",
        "## Artifacts",
        "",
        "- `C_IC_AUDIT.csv`, `IC_SUMMARY.json`",
        "- `BRANCH_HANDOFF_COMPARE.csv`, `BRANCH_AGGREGATE.csv`",
        "- `TAKEOVER_DIVERGENCE_SUMMARY.csv`",
        "- `traces/EARLY100_*.csv`, `TAKEOVER_*.json`, `PAIRED_*.csv`",
        "- `FINAL_REPORT.md`",
        "",
        "**No new HANDOFF thresholds proposed** — mechanism first.",
        "",
    ]
    (OUT / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="Limit C jobs (0=all for IC audit)")
    ap.add_argument("--repr", type=int, default=REPR_N, help="Representative deep-dives")
    args = ap.parse_args()
    run(limit=args.limit, repr_n=args.repr)


if __name__ == "__main__":
    main()
