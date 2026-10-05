"""Headless ±hand-y continuous wrist recapture diagnostic. No lift, no SAC.

    python training/wrist_sweep_recapture.py
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

from controllers.residual import RECOVERY4D_W_HY_MAX
from training.demo_teleport_recovery_state import (
    S_COUPLED,
    load_or_measure_ref,
    run_teleport_case,
    slim,
)

OUT = ROOT / "results" / "diagnostics" / "wrist_sweep_recapture"
FIG = OUT / "figures"
OMEGA = 0.40  # rad/s, legal max 3.0
MAX_DEG = 120.0
TAU = -18.0
DELAY = 0.10
PERSIST = 0.050


def _arr(log, key):
    return np.array([r[key] for r in log], float)


def sweep_rows(log):
    return [r for r in log if r.get("phase") in ("wrist_sweep", "contact", "capture")]


def classify_motion(rows):
    if len(rows) < 4:
        return {"label": "INSUFFICIENT", "dex_mm": 0, "dey_mm": 0, "dez_mm": 0}
    e0 = np.array([rows[0]["e_x"], rows[0]["e_y"], rows[0]["e_z"]], float)
    e1 = np.array([rows[-1]["e_x"], rows[-1]["e_y"], rows[-1]["e_z"]], float)
    de = e1 - e0
    path = 0.0
    prev = e0
    for r in rows[1:]:
        e = np.array([r["e_x"], r["e_y"], r["e_z"]], float)
        path += float(np.linalg.norm(e - prev))
        prev = e
    vr = np.array([r["v_rel"] for r in rows], float)
    wr = np.array([r["w_rel"] for r in rows], float)
    nL = np.array([r["nL"] for r in rows], int)
    nR = np.array([r["nR"] for r in rows], int)
    lost = bool(np.any((nL == 0) & (nR == 0)))
    drop = any(r["obj_z"] < 0.47 for r in rows)
    if drop or lost and path > 0.04:
        lab = "LOSS"
    elif path < 0.004 and float(np.mean(vr)) < 0.015:
        lab = "STICK"
    elif float(np.max(vr)) > 0.25 and (nL[-1] == 0 or nR[-1] == 0):
        lab = "LOSS"
    else:
        lab = "SLIDE_OR_ROLL"
    return {
        "label": lab,
        "dex_mm": 1e3 * float(de[0]),
        "dey_mm": 1e3 * float(de[1]),
        "dez_mm": 1e3 * float(de[2]),
        "path_mm": 1e3 * path,
        "v_rel_mean": float(np.mean(vr)),
        "v_rel_max": float(np.max(vr)),
        "w_rel_mean": float(np.mean(wr)),
        "w_rel_max": float(np.max(wr)),
        "any_both_lost": lost,
        "end_nL": int(nL[-1]),
        "end_nR": int(nR[-1]),
    }


def opposing_contacts(r):
    L = [c for c in r.get("contacts_h") or [] if c.get("side") == "L"]
    R = [c for c in r.get("contacts_h") or [] if c.get("side") == "R"]
    if not L or not R:
        return False
    yL = float(np.mean([c["pos_h"][1] for c in L]))
    yR = float(np.mean([c["pos_h"][1] for c in R]))
    return (yL * yR) < 0


def persistent_windows(rows, dt_need=PERSIST):
    wins = []
    t0 = None
    for r in rows:
        ok = int(r["nL"]) > 0 and int(r["nR"]) > 0
        if ok and t0 is None:
            t0 = r
        if (not ok) and t0 is not None:
            dur = float(r["t"]) - float(t0["t"])
            if dur >= dt_need:
                wins.append(
                    {
                        "t0": t0["t"],
                        "t1": r["t"],
                        "dt": dur,
                        "act0": t0["wrist_act_deg"],
                        "act1": r["wrist_act_deg"],
                        "opposing_at_start": opposing_contacts(t0),
                    }
                )
            t0 = None
    if t0 is not None and rows:
        dur = float(rows[-1]["t"]) - float(t0["t"])
        if dur >= dt_need:
            wins.append(
                {
                    "t0": t0["t"],
                    "t1": rows[-1]["t"],
                    "dt": dur,
                    "act0": t0["wrist_act_deg"],
                    "act1": rows[-1]["wrist_act_deg"],
                    "opposing_at_start": opposing_contacts(t0),
                    "still_at_end": True,
                }
            )
    return wins


def start_geom(log, t_start):
    rows = [r for r in log if r["t"] >= t_start - 1e-9]
    r = rows[0] if rows else log[-1]
    return {
        "t": r["t"],
        "e_mm": [1e3 * r["e_x"], 1e3 * r["e_y"], 1e3 * r["e_z"]],
        "cyl_axis_h": r["cyl_axis_h"],
        "g_h": r.get("g_h"),
        "nL": r["nL"],
        "nR": r["nR"],
        "Fn_L": r["Fn_L"],
        "Fn_R": r["Fn_R"],
        "v_rel_hand": r["v_rel_hand"],
        "omega_rel_hand": r["omega_rel_hand"],
        "aperture": r["aperture"],
        "contacts_h": r.get("contacts_h"),
        "wrist_act_deg": r.get("wrist_act_deg"),
    }


def plot_dir(name, rows, path):
    if not rows:
        return
    t = _arr(rows, "t")
    act = _arr(rows, "wrist_act_deg")
    cmd = _arr(rows, "wrist_cmd_deg")
    fig, axes = plt.subplots(8, 1, figsize=(10, 14), sharex=True)
    axes[0].plot(t, cmd, label="cmd")
    axes[0].plot(t, act, label="act", ls="--")
    axes[0].set_ylabel("wrist deg")
    axes[0].legend(fontsize=8)
    axes[1].plot(t, _arr(rows, "e_x") * 1e3, label="ex")
    axes[1].plot(t, _arr(rows, "e_y") * 1e3, label="ey")
    axes[1].plot(t, _arr(rows, "e_z") * 1e3, label="ez")
    axes[1].set_ylabel("r_h mm")
    axes[1].legend(fontsize=8)
    axh = np.array([r["cyl_axis_h"] for r in rows], float)
    axes[2].plot(t, axh)
    axes[2].set_ylabel("cyl axis h")
    gh = np.array([r.get("g_h") or [0, 0, 0] for r in rows], float)
    axes[3].plot(t, gh)
    axes[3].set_ylabel("g_h")
    axes[4].plot(t, _arr(rows, "nL"), label="nL")
    axes[4].plot(t, _arr(rows, "nR"), label="nR")
    axes[4].legend(fontsize=8)
    axes[5].plot(t, _arr(rows, "Fn_L"), label="FnL")
    axes[5].plot(t, _arr(rows, "Fn_R"), label="FnR")
    axes[5].legend(fontsize=8)
    axes[6].plot(t, _arr(rows, "v_rel"))
    axes[6].set_ylabel("|v_rel|")
    axes[7].plot(t, _arr(rows, "w_rel"))
    axes[7].set_ylabel("|w_rel|")
    axes[7].set_xlabel("t (s)")
    axes[0].set_title(name)
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_one(ref, sign):
    tag = "plus" if sign > 0 else "minus"
    print(f"sweep {tag} w_hy={sign * OMEGA:.3f} ...", flush=True)
    r = run_teleport_case(
        S_COUPLED,
        ref,
        "coupled",
        interactive=False,
        mode="WRIST_SWEEP",
        recovery_delay=DELAY,
        diagnostic_hold=0.0,
        wrist_sweep={"w_hy": sign * OMEGA, "max_deg": MAX_DEG, "tau": TAU},
    )
    rows = sweep_rows(r["log"])
    sw = r.get("wrist_sweep") or {}
    mot = classify_motion(rows)
    wins = persistent_windows(rows)
    t0 = float(sw.get("t_start") or r["t_teleport"])
    geom = start_geom(r["log"], t0)
    track = []
    if rows:
        cmd = _arr(rows, "wrist_cmd_deg")
        act = _arr(rows, "wrist_act_deg")
        des = _arr(rows, "wrist_des_deg")
        track = {
            "cmd_end": float(cmd[-1]),
            "act_end": float(act[-1]),
            "des_end": float(des[-1]),
            "max_|act-cmd|": float(np.max(np.abs(act - cmd))),
            "max_|act-des|": float(np.max(np.abs(act - des))),
        }
    capturable = [
        w
        for w in wins
        if w.get("opposing_at_start") and w["dt"] >= 0.10
    ]
    plot_dir(tag, rows, FIG / f"sweep_{tag}.png")
    summary = {
        "sign": sign,
        "omega": sign * OMEGA,
        "legal_w_hy_max": RECOVERY4D_W_HY_MAX,
        "sweep": sw,
        "start_geom": geom,
        "motion": mot,
        "persist_windows": wins,
        "capturable_windows": capturable,
        "tracking": track,
        "timeline": r.get("timeline"),
        "t_drop": r.get("t_drop"),
        "dt_drop": r.get("dt_drop"),
    }
    (OUT / f"{tag}.json").write_text(
        json.dumps({"summary": summary, "slim": slim(r)}, indent=2, default=str),
        encoding="utf-8",
    )
    np.savez_compressed(
        OUT / f"{tag}_sweep.npz",
        t=_arr(rows, "t") if rows else np.zeros(0),
        cmd=_arr(rows, "wrist_cmd_deg") if rows else np.zeros(0),
        act=_arr(rows, "wrist_act_deg") if rows else np.zeros(0),
        ex=_arr(rows, "e_x") if rows else np.zeros(0),
        ey=_arr(rows, "e_y") if rows else np.zeros(0),
        ez=_arr(rows, "e_z") if rows else np.zeros(0),
        nL=_arr(rows, "nL") if rows else np.zeros(0),
        nR=_arr(rows, "nR") if rows else np.zeros(0),
    )
    print(" ", mot["label"], "act_end", (track or {}).get("act_end"), "persist", len(wins), flush=True)
    return summary, r


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    ref = load_or_measure_ref()
    plus, _ = run_one(ref, +1.0)
    minus, _ = run_one(ref, -1.0)
    payload = {
        "omega_rad_s": OMEGA,
        "max_deg": MAX_DEG,
        "tau": TAU,
        "delay": DELAY,
        "plus": plus,
        "minus": minus,
        "viewer": {
            "plus": "python training/demo_teleport_recovery_state.py --mode wrist_sweep --case coupled_boundary --recovery-delay 0.10 --wrist-sign 1 --wrist-omega 0.40 --wrist-max-deg 120 --playback-speed 0.25",
            "minus": "python training/demo_teleport_recovery_state.py --mode wrist_sweep --case coupled_boundary --recovery-delay 0.10 --wrist-sign -1 --wrist-omega 0.40 --wrist-max-deg 120 --playback-speed 0.25",
        },
    }
    (OUT / "analysis.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print("wrote", OUT / "analysis.json", flush=True)


if __name__ == "__main__":
    main()
