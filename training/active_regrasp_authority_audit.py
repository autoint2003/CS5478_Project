"""Active open-recapture after 1.0 s gravity slide. One modest aperture increase."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.demo_teleport_recovery_state import S_COUPLED, load_or_measure_ref, run_teleport_case, slim

OUT = ROOT / "results" / "diagnostics" / "active_regrasp"


def _ph(log, name):
    return [r for r in log if r.get("phase") == name]


def st(r):
    if r is None:
        return None
    return {
        "t": r.get("t"),
        "r_h_mm": [1e3 * r["e_x"], 1e3 * r["e_y"], 1e3 * r["e_z"]],
        "d_in_mm": r.get("d_in_mm"),
        "cyl_axis_h": r.get("cyl_axis_h"),
        "aperture": r.get("aperture"),
        "nL": r.get("nL"),
        "nR": r.get("nR"),
        "Fn_L": r.get("Fn_L"),
        "Fn_R": r.get("Fn_R"),
        "v_rel": r.get("v_rel"),
        "w_rel": r.get("w_rel"),
        "tau": r.get("tau"),
        "grip_ctrl": r.get("grip_ctrl"),
        "finger_q": r.get("finger_q"),
        "finger_dq": r.get("finger_dq"),
        "wrist_act_deg": r.get("wrist_act_deg"),
        "contacts_h": r.get("contacts_h"),
        "from_plus_x_rim_mm": r.get("from_plus_x_rim_mm"),
        "from_distal_tip_mm": r.get("from_distal_tip_mm"),
        "obj_z": r.get("obj_z"),
    }


def _slope(rows, key, span=1.0):
    if not rows or len(rows) < 2:
        return None
    t1 = float(rows[-1]["t"])
    win = [r for r in rows if r["t"] >= t1 - span] or rows
    dt = float(win[-1]["t"] - win[0]["t"])
    if dt < 1e-6:
        return None
    return (float(win[-1][key]) - float(win[0][key])) / dt


def run(ref, with_lift: bool):
    return run_teleport_case(
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
            "do_return": bool(with_lift),
            "do_lift": bool(with_lift),
            "skip_unload": True,
            "secure_hold_s": 1.0,
            "inward_slide": True,
            "active_regrasp": True,
            "aperture_delta": 0.003,
            "tau_open_geom": 8.0,
            "regrasp_verify_s": 2.0,
        },
    )


def analyze(r):
    log = r.get("log") or []
    cy = r.get("wrist_reposition") or {}
    opens = _ph(log, "open")
    closes = _ph(log, "reclose")
    ver = _ph(log, "regrasp_verify")
    slide = _ph(log, "gravity_slide")
    ret = _ph(log, "return_vertical")
    vert = _ph(log, "vertical_hold")
    hold10 = _ph(log, "diagnostic_hold")
    a = slide[0] if slide else None
    b = slide[-1] if slide else None
    before_open = opens[0] if opens else b
    end_open = opens[-1] if opens else None
    after_close = closes[-1] if closes else None
    after_ver = ver[-1] if ver else None
    delta_ap = None
    if opens:
        delta_ap = 1e3 * (float(opens[-1]["aperture"]) - float(opens[0]["aperture"]))
    din0 = before_open.get("d_in") if before_open else None
    din1 = after_ver.get("d_in") if after_ver else None
    rh0 = None if before_open is None else np.array([before_open["e_x"], before_open["e_y"], before_open["e_z"]])
    rh1 = None if after_ver is None else np.array([after_ver["e_x"], after_ver["e_y"], after_ver["e_z"]])
    geom_changed = False
    if rh0 is not None and rh1 is not None:
        geom_changed = float(np.linalg.norm(rh1 - rh0)) > 0.001 or abs(float(din1) - float(din0)) > 5e-4
    creep = None
    if ver:
        creep = {
            "delta_e_x_mm": 1e3 * (ver[-1]["e_x"] - ver[0]["e_x"]),
            "delta_e_z_mm": 1e3 * (ver[-1]["e_z"] - ver[0]["e_z"]),
            "delta_d_in_mm": 1e3 * (ver[-1].get("d_in", 0) - ver[0].get("d_in", 0)),
            "v_end": ver[-1]["v_rel"],
            "w_end": ver[-1]["w_rel"],
            "n_end": [ver[-1]["nL"], ver[-1]["nR"]],
        }
    outcome = cy.get("fail_phase")
    if outcome is None and ver:
        if abs(creep["delta_e_z_mm"]) > 2.0 or creep["w_end"] > 0.05:
            outcome = "REGRASP_CREEPS"
        elif not geom_changed:
            outcome = "ACTIVE_REGRASP_NO_STATE_CHANGE"
        else:
            outcome = "CHECKPOINT_NO_DROP"
    q1_q7_pass = (
        outcome not in (
            "OPEN_TOO_SMALL",
            "OBJECT_FALLS_DURING_OPEN",
            "RECLOSE_MISSES",
            "ACTIVE_REGRASP_NO_STATE_CHANGE",
            "REGRASP_CREEPS",
            "CAPTURE_FAILURE",
            "OUTBOUND_FAILURE",
        )
        and delta_ap is not None
        and 2.0 <= delta_ap <= 6.0
        and geom_changed
        and ver
        and ver[-1]["nL"] > 0
        and ver[-1]["nR"] > 0
        and abs(creep["delta_e_z_mm"]) < 2.0
        and creep["v_end"] < 0.02
    )
    return {
        "fail_phase": cy.get("fail_phase"),
        "classification": r.get("classification"),
        "outcome": outcome,
        "open_meta": cy.get("open_meta"),
        "delta_ap_mm": delta_ap,
        "before_open": st(before_open),
        "end_open": st(end_open),
        "after_reclose": st(after_close),
        "after_regrasp_verify": st(after_ver),
        "before_open_vs_after_verify": {
            "delta_r_h_mm": None if rh0 is None or rh1 is None else (1e3 * (rh1 - rh0)).tolist(),
            "delta_d_in_mm": None if din0 is None or din1 is None else 1e3 * (float(din1) - float(din0)),
        },
        "open_aperture_tau_fingers": [
            {
                "t": x["t"],
                "ap": x["aperture"],
                "tau": x["tau"],
                "finger_q": x.get("finger_q"),
                "finger_dq": x.get("finger_dq"),
                "nL": x["nL"],
                "nR": x["nR"],
                "v_rel": x["v_rel"],
            }
            for x in opens[:: max(len(opens) // 12, 1)]
        ]
        if opens
        else [],
        "verify_creep": creep,
        "q1_q7_raw_pass": bool(q1_q7_pass),
        "return_end": st(ret[-1]) if ret else None,
        "vertical_end": st(vert[-1]) if vert else None,
        "diag_hold_end": st(hold10[-1]) if hold10 else None,
        "t_drop": r.get("t_drop"),
        "lift": cy.get("lift"),
        "viewer_end_t": cy.get("viewer_end_t"),
        "continuation": cy.get("continuation"),
        "n_open": len(opens),
        "n_reclose": len(closes),
        "n_verify": len(ver),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ref = load_or_measure_ref()
    print("checkpoint (stop after REGRASP VERIFY) ...", flush=True)
    r0 = run(ref, with_lift=False)
    a0 = analyze(r0)
    (OUT / "checkpoint.json").write_text(
        json.dumps({"analysis": a0, "summary": slim(r0), "log": r0.get("log", [])}, indent=2, default=str),
        encoding="utf-8",
    )
    full = None
    if a0["q1_q7_raw_pass"]:
        print("raw Q1-Q7 look passable; running full task (still needs USER VISUAL) ...", flush=True)
        r1 = run(ref, with_lift=True)
        full = analyze(r1)
        (OUT / "full_task.json").write_text(
            json.dumps({"analysis": full, "summary": slim(r1), "log": r1.get("log", [])}, indent=2, default=str),
            encoding="utf-8",
        )
    else:
        print("raw Q1-Q7 did not pass; STOP before return/lift. outcome", a0["outcome"], flush=True)
    (OUT / "analysis.json").write_text(
        json.dumps({"checkpoint": a0, "full": full, "q1_q7_raw_pass": a0["q1_q7_raw_pass"]}, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps({k: a0[k] for k in ("outcome", "delta_ap_mm", "fail_phase", "q1_q7_raw_pass", "before_open_vs_after_verify", "verify_creep", "open_meta")}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
