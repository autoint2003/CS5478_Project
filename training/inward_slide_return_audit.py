"""Gravity-assisted inward slide timing: return after 0.25..2.0 s at 60 deg, tau=-18."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.demo_teleport_recovery_state import (
    S_COUPLED,
    load_or_measure_ref,
    run_teleport_case,
    slim,
)

OUT = ROOT / "results" / "diagnostics" / "inward_slide_return"
SLIDES = (0.25, 0.50, 1.00, 1.50, 2.00)


def _rows(log, phase):
    return [r for r in log if r.get("phase") == phase]


def _state(r):
    if r is None:
        return None
    return {
        "t": r["t"],
        "r_h_mm": [1e3 * r["e_x"], 1e3 * r["e_y"], 1e3 * r["e_z"]],
        "d_in_mm": r.get("d_in_mm"),
        "from_plus_x_rim_mm": r.get("from_plus_x_rim_mm"),
        "from_distal_tip_mm": r.get("from_distal_tip_mm"),
        "u_in_h": r.get("u_in_h"),
        "cyl_axis_h": r.get("cyl_axis_h"),
        "v_rel": r["v_rel"],
        "w_rel": r["w_rel"],
        "v_rel_hand": r.get("v_rel_hand"),
        "omega_rel_hand": r.get("omega_rel_hand"),
        "nL": r["nL"],
        "nR": r["nR"],
        "Fn_L": r["Fn_L"],
        "Fn_R": r["Fn_R"],
        "aperture": r["aperture"],
        "wrist_act_deg": r.get("wrist_act_deg"),
        "tau": r["tau"],
        "contacts_h": r.get("contacts_h"),
        "g_h": r.get("g_h"),
        "obj_z": r.get("obj_z"),
    }


def _slope_mm(rows, key, t_span=1.0):
    if len(rows) < 2:
        return None
    t1 = float(rows[-1]["t"])
    win = [r for r in rows if r["t"] >= t1 - t_span]
    if len(win) < 2:
        win = rows
    dt = float(win[-1]["t"] - win[0]["t"])
    if dt < 1e-6:
        return None
    y0 = float(win[0][key])
    y1 = float(win[-1][key])
    return 1e3 * (y1 - y0) / dt if key in ("e_x", "e_z", "d_in") else (y1 - y0) / dt


def _return_label(slide, ret):
    if not slide or not ret:
        return "unknown"
    d0, d1 = float(slide[-1].get("d_in") or 0), float(ret[-1].get("d_in") or 0)
    dd = d1 - d0
    lost = any(int(r["nL"]) == 0 or int(r["nR"]) == 0 for r in ret)
    vmean = float(np.mean([r["v_rel"] for r in ret]))
    if lost:
        return "loses_contact"
    if dd > 5e-4:
        return "continues_inward"
    if dd < -5e-4:
        return "reverses_outward"
    if vmean < 0.02:
        return "follows_gripper"
    return "relative_motion_without_clear_d_in_sign"


def vertical_verdict(vert):
    if not vert:
        return "NO_VERTICAL", {}
    lost = min(int(r["nL"]) for r in vert) == 0 or min(int(r["nR"]) for r in vert) == 0
    drop = any(r.get("obj_z", 1) < 0.35 for r in vert)
    v0, v1 = float(vert[0]["v_rel"]), float(vert[-1]["v_rel"])
    w0, w1 = float(vert[0]["w_rel"]), float(vert[-1]["w_rel"])
    din0, din1 = float(vert[0].get("d_in") or 0), float(vert[-1].get("d_in") or 0)
    ex_s = _slope_mm(vert, "e_x", 1.0)
    ez_s = _slope_mm(vert, "e_z", 1.0)
    din_s = _slope_mm(vert, "d_in", 1.0)
    stats = {
        "lost_contact": lost,
        "drop": drop,
        "v_rel_start": v0,
        "v_rel_end": v1,
        "w_rel_start": w0,
        "w_rel_end": w1,
        "d_in_mm_start": 1e3 * din0,
        "d_in_mm_end": 1e3 * din1,
        "e_x_slope_mm_s_last1s": ex_s,
        "e_z_slope_mm_s_last1s": ez_s,
        "d_in_slope_mm_s_last1s": din_s,
        "delta_d_in_mm": 1e3 * (din1 - din0),
        "delta_e_x_mm": 1e3 * (vert[-1]["e_x"] - vert[0]["e_x"]),
        "delta_e_z_mm": 1e3 * (vert[-1]["e_z"] - vert[0]["e_z"]),
    }
    if lost or drop:
        return "VERTICAL_FAILURE", stats
    migrating_out = din_s is not None and din_s < -0.3
    still_fast = v1 > 0.02 and v1 > 0.7 * v0
    w_growing = w1 > max(0.05, w0 * 1.2)
    if migrating_out or still_fast or w_growing or abs(stats["delta_e_x_mm"]) > 1.5 or abs(stats["delta_e_z_mm"]) > 2.0:
        return "MARGINAL / CREEPING", stats
    if v1 < 0.015 and abs(stats["delta_d_in_mm"]) < 1.0:
        return "STABLE_VERTICAL", stats
    return "MARGINAL / CREEPING", stats


def run_one(ref, slide_s, with_lift=False):
    print(f"slide_s={slide_s:.2f} lift={with_lift} ...", flush=True)
    r = run_teleport_case(
        S_COUPLED,
        ref,
        "coupled",
        interactive=False,
        playback=0.5,
        mode="REPOSITION_REGRASP",
        recovery_delay=0.10,
        diagnostic_hold=10.0,
        wrist_reposition={
            "omega_out": 1.2,
            "omega_return": 0.8,
            "theta_capture_deg": 60.0,
            "tau_secure": -18.0,
            "tau_reposition": -18.0,
            "reposition_s": 0.5,
            "capture_verify_s": 1.0,
            "vertical_s": 2.0,
            "fail_continue_s": 3.0,
            "do_return": True,
            "do_lift": bool(with_lift),
            "skip_unload": True,
            "secure_hold_s": float(slide_s),
            "inward_slide": True,
        },
    )
    log = r.get("log") or []
    slide = _rows(log, "inward_slide")
    ret = _rows(log, "return_vertical")
    vert = _rows(log, "vertical_hold")
    cy = r.get("wrist_reposition") or {}
    vlab, vstat = vertical_verdict(vert)
    pre = _state(slide[-1] if slide else None)
    out = {
        "slide_s": float(slide_s),
        "with_lift": bool(with_lift),
        "classification": r.get("classification"),
        "fail_phase": cy.get("fail_phase"),
        "t_drop": r.get("t_drop") or cy.get("t_drop"),
        "reached_capture": cy.get("reached_capture"),
        "before_return": pre,
        "after_return": _state(ret[-1] if ret else None),
        "after_vertical": _state(vert[-1] if vert else None),
        "at_capture": cy.get("at_capture"),
        "return_behavior": _return_label(slide, ret),
        "vertical_verdict": vlab,
        "vertical_stats": vstat,
        "d_in_at_60_start_mm": _state(slide[0])["d_in_mm"] if slide else None,
        "d_in_before_return_mm": pre["d_in_mm"] if pre else None,
        "delta_d_in_slide_mm": (
            None
            if not slide
            else 1e3 * (float(slide[-1].get("d_in") or 0) - float(slide[0].get("d_in") or 0))
        ),
        "viewer_end_t": cy.get("viewer_end_t"),
        "continuation": cy.get("continuation"),
        "lift": cy.get("lift"),
        "task_success_2s": r.get("task_success_2s"),
        "task_success": r.get("task_success"),
    }
    tag = f"slide_{slide_s:.2f}s" + ("_lift" if with_lift else "")
    (OUT / f"{tag}.json").write_text(
        json.dumps({"analysis": out, "summary": slim(r), "log": log}, indent=2, default=str),
        encoding="utf-8",
    )
    return out


def pick_earliest_stable(rows):
    stables = [r for r in rows if r["vertical_verdict"] == "STABLE_VERTICAL"]
    if not stables:
        return None
    stables.sort(key=lambda x: float(x["slide_s"]))
    return stables[0]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ref = load_or_measure_ref()
    rows = [run_one(ref, s, with_lift=False) for s in SLIDES]
    pick = pick_earliest_stable(rows)
    lift_row = None
    if pick is not None:
        lift_row = run_one(ref, pick["slide_s"], with_lift=True)
        late = None
        if lift_row.get("lift"):
            late = lift_row["lift"]
        hold_log_path = OUT / f"slide_{pick['slide_s']:.2f}s_lift.json"
        hold_cls = "not_run"
        if hold_log_path.exists():
            blob = json.loads(hold_log_path.read_text(encoding="utf-8"))
            hold = [r for r in blob.get("log", []) if r.get("phase") == "diagnostic_hold"]
            if hold:
                din_s = _slope_mm(hold, "d_in", 5.0)
                ex_s = _slope_mm(hold, "e_x", 5.0)
                ez_s = _slope_mm(hold, "e_z", 5.0)
                migrating = (din_s is not None and din_s < -0.2) or (
                    ex_s is not None and abs(ex_s) > 0.3
                ) or (ez_s is not None and abs(ez_s) > 0.4)
                drop = blob["summary"].get("t_drop") is not None
                if drop:
                    hold_cls = "DELAYED_FAILURE"
                elif migrating:
                    hold_cls = "MARGINAL / CREEPING"
                else:
                    hold_cls = "STABLE_RECOVERY_CANDIDATE"
                lift_row["diagnostic_hold_verdict"] = hold_cls
                lift_row["diagnostic_hold_slopes"] = {
                    "d_in_mm_s": din_s,
                    "e_x_mm_s": ex_s,
                    "e_z_mm_s": ez_s,
                }
    report = {
        "mechanism": (
            "wrist rotation creates a contact/gravity configuration; gravity causes "
            "useful relative sliding while bilateral support is maintained; return "
            "locks a new relative state. Rotation does not itself clamp the cylinder."
        ),
        "tau": -18.0,
        "omega_out": 1.2,
        "omega_return": -0.8,
        "theta_deg": 60.0,
        "branches": rows,
        "picked_slide_s": None if pick is None else pick["slide_s"],
        "pick_reason": (
            "earliest STABLE_VERTICAL from raw 2 s hold"
            if pick is not None
            else "NONE stable; STOP before another control variable"
        ),
        "lift_run": lift_row,
        "viewer_commands": {
            s: (
                "python training/demo_teleport_recovery_state.py "
                "--mode reposition_regrasp --case coupled_boundary --s 2.0 "
                "--recovery-delay 0.10 --wrist-omega 1.2 --theta-capture 60 "
                f"--inward-slide --slide-s {s} --omega-return 0.8 "
                "--fail-continue 3 --playback-speed 0.5"
            )
            for s in (0.5, 1.0, 2.0)
        },
        "viewer_lift_command": (
            None
            if pick is None
            else (
                "python training/demo_teleport_recovery_state.py "
                "--mode reposition_regrasp --case coupled_boundary --s 2.0 "
                "--recovery-delay 0.10 --wrist-omega 1.2 --theta-capture 60 "
                f"--inward-slide --slide-s {pick['slide_s']} --omega-return 0.8 "
                "--with-lift --fail-continue 3 --diagnostic-hold 10 --playback-speed 0.5"
            )
        ),
    }
    (OUT / "analysis.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("picked_slide_s", "pick_reason", "viewer_commands", "viewer_lift_command")}, indent=2))
    for row in rows:
        print(
            f"  t={row['slide_s']:.2f} d_in {row['d_in_at_60_start_mm']} -> {row['d_in_before_return_mm']} "
            f"return={row['return_behavior']} vert={row['vertical_verdict']} fail={row['fail_phase']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
