"""Headless full +hand-y wrist recovery cycle. No SAC.

    python training/wrist_recovery_cycle_audit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from training.demo_teleport_recovery_state import (
    S_COUPLED,
    load_or_measure_ref,
    run_teleport_case,
    slim,
)

OUT = ROOT / "results" / "diagnostics" / "wrist_recovery_cycle"
FIG = OUT / "figures"
THETA = 60.0
OMEGA = 0.40


def _arr(log, key):
    return np.array([r[key] for r in log], float)


def plot_cycle(log, path, marks):
    t = _arr(log, "t")
    fig, axes = plt.subplots(9, 1, figsize=(10, 16), sharex=True)
    series = [
        ("wrist act deg", _arr(log, "wrist_act_deg")),
        ("e_x mm", 1e3 * _arr(log, "e_x")),
        ("e_y mm", 1e3 * _arr(log, "e_y")),
        ("e_z mm", 1e3 * _arr(log, "e_z")),
        ("nL", _arr(log, "nL")),
        ("nR", _arr(log, "nR")),
        ("Fn", np.column_stack([_arr(log, "Fn_L"), _arr(log, "Fn_R")])),
        ("|v_rel|", _arr(log, "v_rel")),
        ("|w_rel|", _arr(log, "w_rel")),
    ]
    for ax, (lab, y) in zip(axes, series):
        ax.plot(t, y, lw=1.0)
        ax.set_ylabel(lab, fontsize=8)
        for name, tt in marks.items():
            if tt is None:
                continue
            ax.axvline(float(tt), ls="--", lw=0.8, alpha=0.7)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("t (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def phase_times(log):
    out = {}
    for r in log:
        p = r["phase"]
        if p not in out:
            out[p] = r["t"]
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    ref = load_or_measure_ref()
    print("wrist recovery cycle with lift ...", flush=True)
    r = run_teleport_case(
        S_COUPLED,
        ref,
        "coupled",
        interactive=False,
        mode="WRIST_RECOVERY_CYCLE",
        recovery_delay=0.10,
        diagnostic_hold=10.0,
        wrist_cycle={
            "omega": OMEGA,
            "theta_capture_deg": THETA,
            "tau": -18.0,
            "secure_s": 0.5,
            "vertical_s": 0.5,
            "stop_before_lift": False,
        },
    )
    cy = r.get("wrist_cycle") or {}
    log = r["log"]
    p0 = phase_times(log)
    marks = {
        "tele": r.get("t_teleport"),
        "obs": p0.get("observe"),
        "rot": p0.get("rotate_to_capture"),
        "sec": p0.get("secure"),
        "ret": p0.get("return_vertical"),
        "vert": p0.get("vertical_stabilize"),
        "lift": p0.get("continue_lift"),
        "hold": p0.get("hold"),
        "drop": r.get("t_drop"),
    }
    plot_cycle(log, FIG / "cycle.png", marks)
    t_hold0 = cy.get("lift", {}).get("t_hold0")
    hold_snaps = {}
    if t_hold0 is not None:
        for name, dt in (("hold0", 0.0), ("plus_2s", 2.0), ("plus_5s", 5.0), ("plus_10s", 10.0)):
            tgt = float(t_hold0) + dt
            i = int(np.argmin(np.abs(_arr(log, "t") - tgt)))
            rr = log[i]
            hold_snaps[name] = {
                "t": rr["t"],
                "e_mm": [1e3 * rr["e_x"], 1e3 * rr["e_y"], 1e3 * rr["e_z"]],
                "nL": rr["nL"],
                "nR": rr["nR"],
                "Fn_L": rr["Fn_L"],
                "Fn_R": rr["Fn_R"],
                "v_rel": rr["v_rel"],
                "w_rel": rr["w_rel"],
                "v_rel_hand": rr["v_rel_hand"],
                "obj_z": rr["obj_z"],
            }
    payload = {
        "theta_capture_deg": THETA,
        "omega": OMEGA,
        "fail_phase": cy.get("fail_phase"),
        "classification": r.get("classification"),
        "cycle": cy,
        "hold_snaps": hold_snaps,
        "slim": slim(r),
        "phase_start": p0,
        "task_success_2s": r.get("task_success_2s"),
        "task_success": r.get("task_success"),
        "lift_completed": r.get("lift_completed"),
        "dt_drop": r.get("dt_drop"),
        "t_drop": r.get("t_drop"),
    }
    (OUT / "analysis.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(
        "fail_phase",
        cy.get("fail_phase"),
        "2s",
        r.get("task_success_2s"),
        "10s",
        r.get("task_success"),
        "drop",
        r.get("dt_drop"),
        flush=True,
    )


if __name__ == "__main__":
    main()
