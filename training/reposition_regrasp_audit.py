"""One complete reposition/regrasp recovery attempt. No sweep.

    python training/reposition_regrasp_audit.py
"""

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

OUT = ROOT / "results" / "diagnostics" / "reposition_regrasp"


def _arr(log, key):
    return np.array([r[key] for r in log], float)


def hold_snaps(log, t_hold0):
    out = {}
    if t_hold0 is None:
        return out
    for name, dt in (("lift_complete", 0.0), ("plus_2s", 2.0), ("plus_5s", 5.0), ("plus_10s", 10.0)):
        tgt = float(t_hold0) + dt
        i = int(np.argmin(np.abs(_arr(log, "t") - tgt)))
        rr = log[i]
        out[name] = {
            "t": rr["t"],
            "e_mm": [1e3 * rr["e_x"], 1e3 * rr["e_y"], 1e3 * rr["e_z"]],
            "cyl_axis_h": rr["cyl_axis_h"],
            "v_rel": rr["v_rel"],
            "w_rel": rr["w_rel"],
            "nL": rr["nL"],
            "nR": rr["nR"],
            "Fn_L": rr["Fn_L"],
            "Fn_R": rr["Fn_R"],
            "aperture": rr["aperture"],
            "tau": rr["tau"],
            "obj_z": rr["obj_z"],
        }
    return out


def classify(cy, snaps):
    if cy.get("fail_phase") in ("OUTBOUND_FAILURE", "REPOSITION_FAILURE", "CAPTURE_FAILURE", "RETURN_FAILURE", "VERTICAL_FAILURE"):
        return "FAILURE", cy.get("fail_phase")
    if cy.get("fail_phase") == "LIFT_FAILURE":
        return "FAILURE", "LIFT_FAILURE"
    if cy.get("fail_phase") == "DELAYED_FAILURE":
        return "DELAYED FAILURE", "DELAYED_FAILURE"
    h0 = snaps.get("lift_complete")
    h10 = snaps.get("plus_10s")
    if h0 and h10:
        dez = abs(h10["e_mm"][2] - h0["e_mm"][2])
        dex = abs(h10["e_mm"][0] - h0["e_mm"][0])
        if dez > 8.0 or dex > 4.0:
            return "MARGINAL / CREEPING", None
        if h10["v_rel"] > 0.02:
            return "MARGINAL / CREEPING", None
    if cy.get("lift", {}).get("task_success"):
        return "STABLE RECOVERY", None
    return "MARGINAL / CREEPING", cy.get("fail_phase")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ref = load_or_measure_ref()
    print("reposition/regrasp cycle (includes lift)...", flush=True)
    r = run_teleport_case(
        S_COUPLED,
        ref,
        "coupled",
        interactive=False,
        mode="REPOSITION_REGRASP",
        recovery_delay=0.10,
        diagnostic_hold=10.0,
        wrist_reposition={
            "omega_out": 1.2,
            "omega_return": 0.8,
            "theta_capture_deg": 60.0,
            "tau_secure": -18.0,
            "tau_reposition": -2.0,
            "reposition_s": 0.5,
            "capture_verify_s": 1.0,
            "vertical_s": 2.0,
        },
    )
    cy = r.get("wrist_reposition") or {}
    hs = hold_snaps(r["log"], (cy.get("lift") or {}).get("t_hold0"))
    label, phase = classify(cy, hs)
    pre = cy.get("pre_recovery") or {}
    ver = cy.get("after_capture_verify") or {}
    delta = None
    if pre and ver:
        delta = {
            "dex_mm": ver["e_mm"][0] - pre["e_mm"][0],
            "dey_mm": ver["e_mm"][1] - pre["e_mm"][1],
            "dez_mm": ver["e_mm"][2] - pre["e_mm"][2],
            "d_axis": (np.array(ver["cyl_axis_h"]) - np.array(pre["cyl_axis_h"])).tolist(),
            "d_aperture": ver["aperture"] - pre["aperture"],
            "d_v_rel": ver["v_rel"] - pre["v_rel"],
        }
    payload = {
        "label": label,
        "fail_phase": cy.get("fail_phase") or phase,
        "tendon": cy.get("tendon"),
        "e_z_convention": cy.get("e_z_convention"),
        "pre_vs_capture_verify": delta,
        "events": {
            "post_observe_100ms": cy.get("pre_recovery"),
            "theta_capture": cy.get("at_capture"),
            "end_reposition": cy.get("after_reposition"),
            "end_capture_verify": cy.get("after_capture_verify"),
            "returned_vertical": cy.get("after_return"),
            "end_vertical_verify": cy.get("after_vertical"),
            **hs,
        },
        "lift": cy.get("lift"),
        "reached_capture": cy.get("reached_capture"),
        "omega_out": cy.get("omega_out"),
        "omega_return": cy.get("omega_return"),
        "slim": slim(r),
        "viewer": (
            "python training/demo_teleport_recovery_state.py "
            "--mode reposition_regrasp --case coupled_boundary "
            "--playback-speed 0.5"
        ),
    }
    (OUT / "analysis.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print("label", label, "fail", cy.get("fail_phase"), "2s", r.get("task_success_2s"), "10s", r.get("task_success"), flush=True)


if __name__ == "__main__":
    main()
