"""Audit false-positive recovery on the frozen coupled s=2.0 IC.

No new IC, no SAC, no RULE, no new recovery primitives.

    python training/recovery_false_positive_audit.py
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
    SEQ_JSON,
    load_or_measure_ref,
    run_teleport_case,
    slim,
)

OUT = ROOT / "results" / "diagnostics" / "recovery_false_positive"
FIG = OUT / "figures"
CANDIDATE = [
    {"name": "wrist", "w_hy": 1.5, "tau": -18.0, "duration": 0.10},
    {"name": "stop", "duration": 0.0, "freeze_targets": True},
]
DELAYS = (0.0, 0.05, 0.10, 0.20)
HOLD = 10.0
KEYS = (
    "hand",
    "hand_linvel",
    "hand_angvel",
    "object",
    "obj_linvel",
    "obj_angvel",
    "nL",
    "nR",
    "Fn_L",
    "Fn_R",
    "p_des",
    "r_des",
    "ctrl",
    "arm_qpos",
    "arm_qvel",
    "v_cmd",
    "w_cmd",
)


def _arr(log, key):
    return np.array([r[key] for r in log], float)


def nearest(log, t, *, after=False, phase=None):
    cand = log
    if after:
        cand = [r for r in log if r["t"] >= t - 1e-12]
        if phase:
            pref = [r for r in cand if r.get("phase") == phase]
            if pref:
                cand = pref
        if not cand:
            cand = log
    ts = np.array([r["t"] for r in cand], float)
    i = int(np.argmin(np.abs(ts - t)))
    return cand[i]


def window(log, t0, t1, *, exclusive_start=False):
    rows = []
    for r in log:
        t = r["t"]
        if exclusive_start:
            if t <= t0 + 1e-12 or t > t1 + 1e-12:
                continue
        elif t < t0 - 1e-12 or t > t1 + 1e-12:
            continue
        rows.append(r)
    return rows


def fd_accel(log):
    t = _arr(log, "t")
    v = _arr(log, "hand_linvel")
    w = _arr(log, "hand_angvel")
    a = np.zeros_like(v)
    al = np.zeros_like(w)
    if len(t) >= 2:
        dt = np.diff(t)
        dt[dt == 0] = np.nan
        a[1:] = np.diff(v, axis=0) / dt[:, None]
        al[1:] = np.diff(w, axis=0) / dt[:, None]
        a[0] = a[1]
        al[0] = al[1]
    return t, a, al


def compare_until(za, ra, t_end):
    z = [r for r in za if r["t"] <= t_end + 1e-12]
    r = [r for r in ra if r["t"] <= t_end + 1e-12]
    n = min(len(z), len(r))
    out = {"n": n, "max": {}, "ok": True}
    for k in KEYS:
        dz = []
        for i in range(n):
            a = np.asarray(z[i][k], float).reshape(-1)
            b = np.asarray(r[i][k], float).reshape(-1)
            if z[i]["t"] != r[i]["t"]:
                out["ok"] = False
                out["time_mismatch"] = [z[i]["t"], r[i]["t"], i]
                return out
            dz.append(float(np.max(np.abs(a - b))))
        m = float(max(dz) if dz else 0.0)
        out["max"][k] = m
        if m > 1e-8:
            out["ok"] = False
    return out


def contact_persist(log, t0, horizons):
    rows = {}
    for h in horizons:
        sl = window(log, t0, t0 + h, exclusive_start=True)
        if not sl:
            rows[str(h)] = {}
            continue
        nR = np.array([r["nR"] for r in sl], int)
        nL = np.array([r["nL"] for r in sl], int)
        dt = np.diff(np.array([r["t"] for r in sl], float), prepend=sl[0]["t"])
        dt[0] = 0.0
        right_s = float(np.sum(dt[nR > 0]))
        bilat_s = float(np.sum(dt[(nR > 0) & (nL > 0)]))
        breaks = int(np.sum((nR[1:] == 0) & (nR[:-1] > 0))) if len(nR) > 1 else 0
        fnR = np.array([r["Fn_R"] for r in sl], float)
        fnL = np.array([r["Fn_L"] for r in sl], float)
        ncon_r = [len([c for c in r["contacts"] if c.get("side") == "R"]) for r in sl]
        rows[str(h)] = {
            "frac_right": float(np.mean(nR > 0)),
            "frac_bilateral": float(np.mean((nL > 0) & (nR > 0))),
            "right_duration_s": right_s,
            "bilateral_duration_s": bilat_s,
            "right_break_count": breaks,
            "mean_nR": float(np.mean(nR)),
            "mean_nL": float(np.mean(nL)),
            "mean_ncon_R": float(np.mean(ncon_r) if ncon_r else 0),
            "mean_Fn_R": float(np.mean(fnR)),
            "mean_Fn_L": float(np.mean(fnL)),
            "min_Fn_R": float(np.min(fnR)),
            "min_Fn_L": float(np.min(fnL)),
        }
    return rows


def ez_at(log, t_abs, phase=None):
    r = nearest(log, t_abs, after=True, phase=phase)
    vrel = np.asarray(r["v_rel_hand"], float)
    return {
        "t": r["t"],
        "e_z_mm": 1e3 * float(r["e_z"]),
        "e_x_mm": 1e3 * float(r["e_x"]),
        "v_rel_z": float(vrel[2]) if vrel.size >= 3 else float("nan"),
        "Fn_L": float(r["Fn_L"]),
        "Fn_R": float(r["Fn_R"]),
        "obj_z": float(r["obj_z"]),
        "hand_z": float(np.asarray(r["hand"])[2]),
        "obj_minus_hand_z_mm": 1e3 * (float(r["obj_z"]) - float(np.asarray(r["hand"])[2])),
        "nL": int(r["nL"]),
        "nR": int(r["nR"]),
        "v_rel": float(r["v_rel"]),
        "w_rel": float(r["w_rel"]),
    }


def late_stats(log, t_hold0, hold_s=10.0):
    late0 = t_hold0 + 0.5 * hold_s
    late = [r for r in log if r["t"] >= late0]
    if len(late) < 4:
        late = [r for r in log if r["t"] >= t_hold0]
    if not late:
        return {"n": 0}
    t = np.array([r["t"] for r in late], float)
    ex = np.array([r["e_x"] for r in late], float)
    ey = np.array([r["e_y"] for r in late], float)
    ez = np.array([r["e_z"] for r in late], float)
    vr = np.array([r["v_rel"] for r in late], float)
    wr = np.array([r["w_rel"] for r in late], float)
    tilt = np.array([r["tilt_deg"] for r in late], float)
    fnL = np.array([r["Fn_L"] for r in late], float)
    fnR = np.array([r["Fn_R"] for r in late], float)

    def slope(y):
        if len(t) < 2:
            return 0.0
        p = np.polyfit(t - t[0], y, 1)
        return float(p[0])

    rng = lambda y: float(np.max(y) - np.min(y))
    return {
        "n": len(late),
        "t0": float(t[0]),
        "t1": float(t[-1]),
        "ex_range_mm": 1e3 * rng(ex),
        "ey_range_mm": 1e3 * rng(ey),
        "ez_range_mm": 1e3 * rng(ez),
        "ez_slope_mm_s": 1e3 * slope(ez),
        "tilt_range_deg": rng(tilt),
        "v_rel_mean": float(np.mean(vr)),
        "v_rel_end": float(vr[-1]),
        "w_rel_mean": float(np.mean(wr)),
        "w_rel_end": float(wr[-1]),
        "fnL_slope": slope(fnL),
        "fnR_slope": slope(fnR),
        "fnL_end": float(fnL[-1]),
        "fnR_end": float(fnR[-1]),
        "fn_collapse": bool(fnL[-1] < 1.0 and fnR[-1] < 1.0),
    }


def classify_long(run):
    drop = bool(run.get("drop"))
    t2 = bool(run.get("task_success_2s"))
    t10 = bool(run.get("hold_completed")) and not drop and bool(run.get("lift_completed"))
    late = run.get("late") or {}
    if drop and not t2:
        return "FAILURE"
    if drop and t2:
        return "DELAYED_FAILURE"
    if t10:
        drifting = (
            abs(float(late.get("ez_slope_mm_s") or 0)) > 0.4
            or float(late.get("ez_range_mm") or 0) > 8.0
            or float(late.get("v_rel_mean") or 0) > 0.015
            or float(late.get("w_rel_mean") or 0) > 0.15
            or bool(late.get("fn_collapse"))
        )
        if drifting:
            return "MARGINAL_SURVIVAL"
        return "STABLE_RECOVERY"
    if drop:
        return "FAILURE"
    return "MARGINAL_SURVIVAL"


def mark(ax, marks):
    colors = {
        "teleport": "k",
        "recovery start": "C1",
        "recovery end": "C3",
        "bilateral": "C2",
        "lift done": "C0",
        "2s": "0.4",
        "drop": "r",
        "10s": "purple",
    }
    for name, t in marks.items():
        if t is None:
            continue
        ax.axvline(float(t), color=colors.get(name, "0.5"), ls="--", lw=0.9, alpha=0.8)


def plot_late(name, run, marks, path):
    log = run["log"]
    t = _arr(log, "t")
    fig, axes = plt.subplots(11, 1, figsize=(10, 18), sharex=True)
    series = [
        ("e_x (m)", _arr(log, "e_x")),
        ("e_y (m)", _arr(log, "e_y")),
        ("e_z (m)", _arr(log, "e_z")),
        ("tilt (deg)", _arr(log, "tilt_deg")),
        ("|v_rel|", _arr(log, "v_rel")),
        ("|omega_rel|", _arr(log, "w_rel")),
        ("nL", _arr(log, "nL")),
        ("nR", _arr(log, "nR")),
        ("Fn_L", _arr(log, "Fn_L")),
        ("Fn_R", _arr(log, "Fn_R")),
        ("obj-hand z (m)", _arr(log, "obj_z") - _arr(log, "hand")[:, 2]),
    ]
    for ax, (lab, y) in zip(axes, series):
        ax.plot(t, y, lw=1.0)
        ax.set_ylabel(lab, fontsize=8)
        mark(ax, marks)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("t (s)")
    axes[0].set_title(name)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_jerk(name, log, t_tele, t_rec, path):
    sl = window(log, t_tele - 0.050, t_tele + 0.200)
    if len(sl) < 3:
        return
    t, a, al = fd_accel(sl)
    fig, axes = plt.subplots(8, 1, figsize=(10, 14), sharex=True)
    series = [
        ("hand pos (m)", _arr(sl, "hand")),
        ("hand linvel", _arr(sl, "hand_linvel")),
        ("hand angvel", _arr(sl, "hand_angvel")),
        ("hand lin acc (fd)", a),
        ("hand ang acc (fd)", al),
        ("p_des", _arr(sl, "p_des")),
        ("w_cmd", _arr(sl, "w_cmd")),
        ("ctrl[:7]", _arr(sl, "ctrl")[:, :7] if _arr(sl, "ctrl").ndim == 2 else _arr(sl, "ctrl")),
    ]
    for ax, (lab, y) in zip(axes, series):
        ax.plot(t, y, lw=1.0)
        ax.axvline(t_tele, color="k", ls="--")
        if t_rec is not None:
            ax.axvline(float(t_rec), color="C1", ls="--")
        ax.set_ylabel(lab, fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("t (s)")
    axes[0].set_title(name + "  black=teleport  orange=recovery start")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_overlay(z, rec, t_tele, t_rec, path):
    t0, t1 = t_tele - 0.02, t_tele + 0.25
    zs = window(z, t0, t1)
    rs = window(rec, t0, t1)
    fig, axes = plt.subplots(6, 1, figsize=(10, 11), sharex=True)
    pairs = [
        ("hand", "hand"),
        ("hand_linvel", "hand_linvel"),
        ("object", "object"),
        ("obj_linvel", "obj_linvel"),
        ("p_des", "p_des"),
        ("ctrl0", None),
    ]
    for ax, (lab, key) in zip(axes, pairs):
        if key:
            ax.plot(_arr(zs, "t"), _arr(zs, key), label="ZERO", lw=1.2)
            ax.plot(_arr(rs, "t"), _arr(rs, key), label="REC", lw=1.0, ls="--")
        else:
            ax.plot(_arr(zs, "t"), _arr(zs, "ctrl")[:, 0], label="ZERO ctrl0")
            ax.plot(_arr(rs, "t"), _arr(rs, "ctrl")[:, 0], label="REC ctrl0", ls="--")
        ax.axvline(t_tele, color="k", ls="--")
        if t_rec is not None:
            ax.axvline(float(t_rec), color="C1", ls="--")
        ax.set_ylabel(lab, fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("t (s)")
    fig.suptitle("ZERO vs candidate immediately after teleport")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def marks_for(run):
    tl = run.get("timeline") or {}
    ev = run.get("recovery_events") or {}
    t_hold0 = run.get("t_hold0")
    return {
        "teleport": run.get("t_teleport"),
        "recovery start": tl.get("t_recovery_command_start"),
        "recovery end": ev.get("t_intervention_end"),
        "bilateral": tl.get("t_first_bilateral"),
        "lift done": run.get("t_lift_done"),
        "2s": None if t_hold0 is None else float(t_hold0) + 2.0,
        "drop": run.get("t_drop"),
        "10s": None if t_hold0 is None else float(t_hold0) + 10.0,
    }


def summarize(run):
    log = run["log"]
    t_tele = float(run["t_teleport"])
    t_hold0 = run.get("t_hold0")
    t_lift = run.get("t_lift_done")
    rec_end = (run.get("recovery_events") or {}).get("t_intervention_end")
    snaps = {
        "after_teleport": ez_at(log, t_tele, phase="teleport"),
        "after_recovery": ez_at(log, rec_end) if rec_end else None,
        "lift_done": ez_at(log, t_lift) if t_lift else None,
        "plus_2s": ez_at(log, float(t_hold0) + 2.0) if t_hold0 else None,
        "plus_5s": ez_at(log, float(t_hold0) + 5.0) if t_hold0 else None,
        "plus_10s": ez_at(log, float(t_hold0) + 10.0) if t_hold0 else None,
    }
    late = late_stats(log, float(t_hold0 or t_tele), HOLD)
    run["late"] = late
    run["long_class"] = classify_long(run)
    run["ez_snaps"] = snaps
    run["contact_persist"] = contact_persist(log, t_tele, (0.10, 0.50, 2.0, 12.0))
    return run


def jerk_cause(zero, rec):
    t_tele = float(zero["t_teleport"])
    zlog, rlog = zero["log"], rec["log"]
    z0 = nearest(zlog, t_tele, after=True, phase="teleport")
    r0 = nearest(rlog, t_tele, after=True, phase="teleport")
    z1 = next(r for r in zlog if r["t"] > t_tele + 1e-12)
    r1 = next(r for r in rlog if r["t"] > t_tele + 1e-12)
    tl = rec.get("timeline") or {}
    d_z = float(np.linalg.norm(np.asarray(z1["hand_angvel"]) - np.asarray(z0["hand_angvel"])))
    d_r = float(np.linalg.norm(np.asarray(r1["hand_angvel"]) - np.asarray(r0["hand_angvel"])))
    d_lin_z = float(np.linalg.norm(np.asarray(z1["hand_linvel"]) - np.asarray(z0["hand_linvel"])))
    d_lin_r = float(np.linalg.norm(np.asarray(r1["hand_linvel"]) - np.asarray(r0["hand_linvel"])))
    same_tick_rec = abs(float(tl.get("t_recovery_command_start") or 1e9) - float(t_tele)) < 1e-9
    zwin = window(zlog, t_tele, t_tele + 0.10, exclusive_start=True)
    rwin = window(rlog, t_tele, t_tele + 0.10, exclusive_start=True)
    max_wh_z = float(max(np.linalg.norm(r["hand_angvel"]) for r in zwin) if zwin else 0)
    max_wh_r = float(max(np.linalg.norm(r["hand_angvel"]) for r in rwin) if rwin else 0)
    max_qvel_z = float(max(np.linalg.norm(r["arm_qvel"]) for r in zwin) if zwin else 0)
    max_qvel_r = float(max(np.linalg.norm(r["arm_qvel"]) for r in rwin) if rwin else 0)
    p_des_jump_z = float(np.linalg.norm(np.asarray(z1["p_des"]) - np.asarray(z0["p_des"])))
    r_des_jump_r = float(np.linalg.norm(np.asarray(r1["r_des"]) - np.asarray(r0["r_des"])))
    w_r = float(np.linalg.norm(r1["w_cmd"]))
    w_z = float(np.linalg.norm(z1["w_cmd"]))
    return {
        "teleport_hand_angvel": z0["hand_angvel"],
        "first_step_t": z1["t"],
        "d_hand_angvel_ZERO": d_z,
        "d_hand_angvel_REC": d_r,
        "d_hand_linvel_ZERO": d_lin_z,
        "d_hand_linvel_REC": d_lin_r,
        "recovery_command_armed_at_teleport_clock": same_tick_rec,
        "first_mj_step_applies_wrist": bool(w_r > 0.1),
        "p_des_delta_first_ZERO": p_des_jump_z,
        "r_des_delta_first_REC": r_des_jump_r,
        "w_cmd_first_ZERO": w_z,
        "w_cmd_first_REC": w_r,
        "ctrl_delta_ZERO": float(np.linalg.norm(np.asarray(z1["ctrl"]) - np.asarray(z0["ctrl"]))),
        "ctrl_delta_REC": float(np.linalg.norm(np.asarray(r1["ctrl"]) - np.asarray(r0["ctrl"]))),
        "qvel_delta_ZERO": float(np.linalg.norm(np.asarray(z1["arm_qvel"]) - np.asarray(z0["arm_qvel"]))),
        "qvel_delta_REC": float(np.linalg.norm(np.asarray(r1["arm_qvel"]) - np.asarray(r0["arm_qvel"]))),
        "max_|hand_angvel|_100ms_ZERO": max_wh_z,
        "max_|hand_angvel|_100ms_REC": max_wh_r,
        "max_|arm_qvel|_100ms_ZERO": max_qvel_z,
        "max_|arm_qvel|_100ms_REC": max_qvel_r,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    if SEQ_JSON.exists():
        schedule = json.loads(SEQ_JSON.read_text(encoding="utf-8"))["schedule"]
    else:
        schedule = CANDIDATE
    ref = load_or_measure_ref()
    print("ZERO diagnostic_hold=10 ...", flush=True)
    zero = run_teleport_case(
        S_COUPLED, ref, "coupled", interactive=False, mode="ZERO",
        recovery_delay=0.0, diagnostic_hold=HOLD,
    )
    summarize(zero)
    print("  class", zero["long_class"], "drop", zero.get("dt_drop"), "2s", zero.get("task_success_2s"), flush=True)

    recs = {}
    identity = {}
    stop_identity = False
    for d in DELAYS:
        print(f"RECOVERY delay={d:.2f} ...", flush=True)
        rec = run_teleport_case(
            S_COUPLED, ref, "coupled", interactive=False, mode="RECOVERY",
            schedule=schedule, recovery_delay=d, diagnostic_hold=HOLD,
        )
        summarize(rec)
        recs[str(d)] = rec
        t_int = rec["timeline"].get("t_recovery_command_start")
        t_cut = float(t_int if t_int is not None else rec["t_teleport"])
        # identical through last common sample strictly before intervention, including teleport
        cmp = compare_until(zero["log"], rec["log"], t_cut - 1e-9 if d > 0 else rec["t_teleport"])
        identity[str(d)] = cmp
        print("  identity_ok", cmp["ok"], "max", {k: v for k, v in cmp["max"].items() if v > 1e-10}, flush=True)
        print("  class", rec["long_class"], "2s", rec.get("task_success_2s"), "drop", rec.get("dt_drop"), flush=True)
        if d > 0 and not cmp["ok"]:
            stop_identity = True
            print("STOP: ZERO and RECOVERY diverged before declared intervention.", flush=True)
            break

    rec0 = recs["0.0"]
    cause = jerk_cause(zero, rec0)
    plot_jerk("ZERO", zero["log"], zero["t_teleport"], None, FIG / "jerk_ZERO.png")
    plot_jerk("REC delay0", rec0["log"], rec0["t_teleport"], rec0["timeline"].get("t_recovery_command_start"), FIG / "jerk_REC_d0.png")
    if "0.1" in recs:
        plot_jerk(
            "REC delay100ms",
            recs["0.1"]["log"],
            recs["0.1"]["t_teleport"],
            recs["0.1"]["timeline"].get("t_recovery_command_start"),
            FIG / "jerk_REC_d100.png",
        )
    plot_overlay(zero["log"], rec0["log"], zero["t_teleport"], rec0["timeline"].get("t_recovery_command_start"), FIG / "overlay_d0.png")
    plot_late("ZERO", zero, marks_for(zero), FIG / "late_ZERO.png")
    for d, rec in recs.items():
        plot_late(f"REC delay={d}", rec, marks_for(rec), FIG / f"late_REC_d{d}.png")

    payload = {
        "candidate_label": "CANDIDATE / UNVERIFIED RECOVERY",
        "not": "RECOVERY SUCCESS",
        "schedule": schedule,
        "identity": identity,
        "stop_identity": stop_identity,
        "jerk": cause,
        "zero": {**slim(zero), "long_class": zero["long_class"], "ez_snaps": zero["ez_snaps"], "late": zero["late"], "contact_persist": zero["contact_persist"]},
        "recovery": {
            d: {
                **slim(r),
                "long_class": r["long_class"],
                "ez_snaps": r["ez_snaps"],
                "late": r["late"],
                "contact_persist": r["contact_persist"],
            }
            for d, r in recs.items()
        },
    }
    (OUT / "analysis.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    np.savez_compressed(
        OUT / "logs.npz",
        **{f"zero_t": _arr(zero["log"], "t")},
    )
    print("wrote", OUT / "analysis.json", flush=True)
    return payload, zero, recs, cause, identity, stop_identity


if __name__ == "__main__":
    main()
