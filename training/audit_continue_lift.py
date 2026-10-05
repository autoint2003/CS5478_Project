"""Canonical impact then common nominal-LIFT continuation. Diagnostic only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.impact_demo_core import LOG_DIR, frozen_hashes, run_episode  # noqa: E402

OUT = LOG_DIR / "continue_lift"
FIG = OUT / "figures"


def _metrics(r: dict) -> dict:
    cl = r.get("continue_lift") or {}
    log = cl.get("log") or []
    start = cl.get("start") or {}
    vis = r.get("visualization") or {}
    if not log:
        return {"empty": True}
    ex = np.abs(np.array([row["e_x"] for row in log]))
    tilt = np.array([row["tilt_deg"] for row in log])
    vrel = np.array([np.linalg.norm(row["v_rel_hand"]) for row in log])
    fnl = np.array([row["Fn_L"] for row in log])
    fnr = np.array([row["Fn_R"] for row in log])
    hz = np.array([row["hand_z"] for row in log])
    oz = np.array([row["obj_z"] for row in log])
    last = log[-1]
    captured = int(last["nL"]) > 0 and int(last["nR"]) > 0
    drop = vis.get("t_drop") is not None and (
        cl.get("t_both_lost_continuation") is not None
        or vis["t_drop"] >= float(start.get("t", 0.0))
    )
    cmd_done = cl.get("t_lift_cmd_done") is not None
    lift_ach = float(hz[-1] - start["z_hand"])
    obj_dz = float(oz[-1] - start["z_obj"])
    uni_t = cl.get("t_unilateral_continuation")
    uni_s = (
        None
        if uni_t is None
        else float(last["t"] - uni_t)
        if True
        else None
    )
    if uni_t is not None:
        uni_s = float(sum(0.01 for row in log if (row["nL"] == 0) ^ (row["nR"] == 0)))
    return {
        "t_continue": start.get("t"),
        "t_eval_end": r.get("t_eval_end"),
        "t_impact": r.get("t_first_contact"),
        "z_hand0": start.get("z_hand"),
        "z_target_hand": float(start["z_hand"]) + float(cl["lift_dz"]),
        "lift_requested_m": cl.get("lift_dz"),
        "v_lift": cl.get("v_lift"),
        "expected_duration_s": start.get("expected_duration_s"),
        "lift_achieved_hand_m": lift_ach,
        "object_dz_m": obj_dz,
        "max_abs_e_x_mm": float(np.max(ex) * 1e3),
        "final_abs_e_x_mm": float(ex[-1] * 1e3),
        "max_tilt_deg": float(np.max(tilt)),
        "final_tilt_deg": float(tilt[-1]),
        "max_v_rel": float(np.max(vrel)),
        "min_Fn_L": float(np.min(fnl)),
        "min_Fn_R": float(np.min(fnr)),
        "t_unilateral": uni_t,
        "time_unilateral_s": uni_s,
        "both_contact_loss": cl.get("t_both_lost_continuation") is not None,
        "drop": vis.get("t_drop") is not None,
        "t_drop": vis.get("t_drop"),
        "lift_cmd_done": cmd_done,
        "lift_completed_captured": bool(cmd_done and captured and vis.get("t_drop") is None),
        "recovered_at_end": bool(last["recovered"]),
        "final_nL": last["nL"],
        "final_nR": last["nR"],
        "final_obj_z": last["obj_z"],
        "slip_hx_cum": cl.get("slip_hx_cum"),
        "eval_success_GT": r["final"]["success_GT"],
        "eval_fail_kind": r["final"]["fail_kind_GT"],
        "used_rule_resume": cl.get("used_rule_resume"),
    }


def _plot(z: dict, h: dict) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    zl, hl = z["continue_lift"]["log"], h["continue_lift"]["log"]
    tz = np.array([row["t"] for row in zl])
    th = np.array([row["t"] for row in hl])
    marks_z = [
        ("impact", z["t_first_contact"], "k"),
        ("continue", z["t_continue"], "C1"),
        ("drop", z["visualization"].get("t_drop"), "r"),
        ("lift_done", z["continue_lift"].get("t_lift_cmd_done"), "g"),
    ]
    marks_h = [
        ("impact", h["t_first_contact"], "k"),
        ("recovered", h["t_eval_end"], "C2"),
        ("continue", h["t_continue"], "C1"),
        ("drop", h["visualization"].get("t_drop"), "r"),
        ("lift_done", h["continue_lift"].get("t_lift_cmd_done"), "g"),
    ]

    def panel(ax, yz, yh, ylab):
        ax.plot(tz, yz, label="ZERO", color="C3")
        ax.plot(th, yh, label="HEURISTIC", color="C0")
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)

    fig, axes = plt.subplots(6, 1, figsize=(9, 12), sharex=True)
    panel(axes[0], [row["obj_z"] for row in zl], [row["obj_z"] for row in hl], "obj z [m]")
    panel(axes[1], [1e3 * row["e_x"] for row in zl], [1e3 * row["e_x"] for row in hl], "e_x [mm]")
    panel(axes[2], [row["tilt_deg"] for row in zl], [row["tilt_deg"] for row in hl], "tilt [deg]")
    panel(
        axes[3],
        [np.linalg.norm(row["v_rel_hand"]) for row in zl],
        [np.linalg.norm(row["v_rel_hand"]) for row in hl],
        "|v_rel| [m/s]",
    )
    axes[4].plot(tz, [row["Fn_L"] for row in zl], "C3-", label="Z Fn_L")
    axes[4].plot(tz, [row["Fn_R"] for row in zl], "C3--", label="Z Fn_R")
    axes[4].plot(th, [row["Fn_L"] for row in hl], "C0-", label="H Fn_L")
    axes[4].plot(th, [row["Fn_R"] for row in hl], "C0--", label="H Fn_R")
    axes[4].set_ylabel("Fn [N]")
    axes[4].grid(True, alpha=0.3)
    axes[5].plot(tz, [row["nL"] for row in zl], "C3-", label="Z nL")
    axes[5].plot(tz, [row["nR"] for row in zl], "C3--", label="Z nR")
    axes[5].plot(th, [row["nL"] for row in hl], "C0-", label="H nL")
    axes[5].plot(th, [row["nR"] for row in hl], "C0--", label="H nR")
    axes[5].set_ylabel("n contacts")
    axes[5].set_xlabel("t [s]")
    axes[5].grid(True, alpha=0.3)
    for ax in axes:
        for name, t, c in marks_z + marks_h:
            if t is not None:
                ax.axvline(float(t), color=c, lw=0.8, alpha=0.5)
    axes[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "continue_lift_compare.png", dpi=140)
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    hashes = frozen_hashes()
    z = run_episode("zero", screenshots=False, seed=0)
    h = run_episode("heuristic", screenshots=False, seed=0)
    mz, mh = _metrics(z), _metrics(h)
    _plot(z, h)
    payload = {
        "hashes": hashes,
        "used_final_eval_data": False,
        "canonical_disturbance_unchanged": True,
        "zero": mz,
        "heuristic": mh,
        "zero_start": z["continue_lift"]["start"],
        "heur_start": h["continue_lift"]["start"],
        "resume_audit": z["continue_lift"]["rule_resume_audit"],
    }
    (OUT / "metrics.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (OUT / "zero_continue.json").write_text(
        json.dumps(
            {
                "final": z["final"],
                "continue_lift": {k: v for k, v in z["continue_lift"].items() if k != "log"},
                "metrics": mz,
                "log": z["continue_lift"]["log"],
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    (OUT / "heuristic_continue.json").write_text(
        json.dumps(
            {
                "final": h["final"],
                "continue_lift": {k: v for k, v in h["continue_lift"].items() if k != "log"},
                "metrics": mh,
                "log": h["continue_lift"]["log"],
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"zero": mz, "heuristic": mh, "hashes": hashes}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
