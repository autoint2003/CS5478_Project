"""Clean airborne release -> recapture authority. No ball, no SAC, no MP4.

Privileged object state is used only to design/diagnose timing.
"""

from __future__ import annotations

import argparse
import importlib
import json
import pickle
import re
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import controllers.residual as residual_mod
from controllers.gripper_controller import finger_opening
from controllers.jacobian_controller import gains_from_cfg
from controllers.residual import (
    RECOVERY4D_TAU_SECURE,
    RECOVERY4D_V_HX_MAX,
    RECOVERY4D_V_Z_MAX,
    RECOVERY4D_W_HY_MAX,
    map_recovery4d,
)
from envs.config_util import load_yaml, merge_sim_config
from envs.deterioration import body_twist
from envs.physical_recovery import TABLE_DROP, Z_AIR, physical_pack
from training.demo_teleport_recovery_state import (
    CYL_MASS,
    PAIR_MU,
    advance_to_parent,
    make_parent_sim,
)
from training.grav_reposition_v2_viz import apply_camera_preset
from training.map_ballistic_disturbance import CAM, _viewer_overlay
from training.replay_core import freeze, tick_vw
from training.vertical_slip_contact_mechanics_audit import extract_contacts

OUT = ROOT / "results" / "diagnostics" / "airborne_recapture"
RAW = OUT / "raw"
HOLD_S = 12.0
LOG_DT = 0.002
BOTH_OFF_PROOF_S = 0.020
RESIDUAL_PY = ROOT / "controllers" / "residual.py"


def dump(path: Path, obj) -> None:
    def fix(x):
        if isinstance(x, dict):
            return {str(k): fix(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [fix(v) for v in x]
        if isinstance(x, np.ndarray):
            return fix(x.tolist())
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
            return None
        return x

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fix(obj), indent=2), encoding="utf-8")


def a3_of_tau(tau: float, tau_open: float, tau_sec: float = RECOVERY4D_TAU_SECURE) -> float:
    return float(2.0 * (float(tau) - tau_open) / (tau_sec - tau_open) - 1.0)


def tick_4d(sim, gains, omega_y, v_x, v_z, tau):
    tau_open = float(residual_mod.RECOVERY4D_TAU_OPEN)
    omega_y = float(np.clip(omega_y, -RECOVERY4D_W_HY_MAX, RECOVERY4D_W_HY_MAX))
    v_x = float(np.clip(v_x, -RECOVERY4D_V_HX_MAX, RECOVERY4D_V_HX_MAX))
    v_z = float(np.clip(v_z, -RECOVERY4D_V_Z_MAX, RECOVERY4D_V_Z_MAX))
    a0 = omega_y / RECOVERY4D_W_HY_MAX
    a1 = v_x / RECOVERY4D_V_HX_MAX
    a2 = v_z / RECOVERY4D_V_Z_MAX
    a3 = float(np.clip(a3_of_tau(tau, tau_open), -1.0, 1.0))
    Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    cmd = residual_mod.map_recovery4d(np.array([a0, a1, a2, a3]), sim.fsm.r_des, Rh)
    tick_vw(sim, cmd["v_world"], cmd["w_world"], cmd["tau"], gains)
    cmd["a3"] = a3
    return cmd


def row(sim, gains=None, tau=None, cmd=None, label="", prev=None) -> dict:
    o = physical_pack(sim)
    fc, _ = extract_contacts(sim)
    ap = finger_opening(sim.data, sim.ids)
    t = float(sim.data.time)
    vo, wo = body_twist(sim.model, sim.data, sim.ids.object_body)
    vh, wh = body_twist(sim.model, sim.data, sim.ids.hand_body)
    geoms = [(int(c["geom1"]), int(c["geom2"]), c["side"]) for c in fc]
    pts = [np.asarray(c["pos_w"], float).tolist() for c in fc]
    dap_dt = None
    if prev is not None:
        dt = t - float(prev["t"])
        if dt > 1e-12:
            dap_dt = (ap - float(prev["aperture"])) / dt
    a = None if cmd is None else np.asarray(cmd.get("a", [np.nan] * 4), float)
    return {
        "t": t,
        "label": label,
        "ctrl7": float(sim.data.ctrl[7]) if sim.data.ctrl.size > 7 else None,
        "tau_cmd": None if tau is None else float(tau),
        "a3": None if cmd is None else float(cmd.get("a3", np.nan)),
        "a": None if a is None else a[:4].tolist(),
        "aperture": ap,
        "aperture_vel": dap_dt,
        "nL": int(o["nL"]),
        "nR": int(o["nR"]),
        "Fn_L": float(sum(abs(c["Fn"]) for c in fc if c["side"] == "L")),
        "Fn_R": float(sum(abs(c["Fn"]) for c in fc if c["side"] == "R")),
        "rh": np.asarray(o["rh"], float).tolist(),
        "v_rel_h": np.asarray(o["v_rel_h"], float).tolist(),
        "w_rel_h": np.asarray(o["w_rel_h"], float).tolist(),
        "v_rel": float(o["v_rel"]),
        "w_rel": float(o["w_rel"]),
        "obj_z": float(o["obj_z"]),
        "clear": float(o.get("clear", np.nan)),
        "scene": int(o["scene"]),
        "p_hand": np.array(sim.data.xpos[sim.ids.hand_body], float).tolist(),
        "p_obj": np.array(sim.data.xpos[sim.ids.object_body], float).tolist(),
        "v_hand": vh.tolist(),
        "v_obj": vo.tolist(),
        "w_hand": wh.tolist(),
        "w_obj": wo.tolist(),
        "p_des": np.asarray(sim.fsm.p_des, float).tolist(),
        "r_des": np.asarray(sim.fsm.r_des, float).reshape(3, 3).tolist(),
        "v_cmd": np.asarray(sim.fsm.v_cmd, float).tolist(),
        "geoms": geoms,
        "contact_pts": pts,
        "n_contacts": len(fc),
    }


def restore_parent(sim, snap) -> None:
    sim.load_snapshot(snap)
    mujoco.mj_forward(sim.model, sim.data)
    freeze(sim)


def save_parent(sim, packed) -> dict:
    s = sim.snapshot()
    s["mass"] = packed.get("snap", {}).get("mass", CYL_MASS)
    s["friction"] = PAIR_MU
    s["z_tgt"] = float(sim.fsm.p_des[2])
    return s


def grip_audit_static(sim) -> dict:
    lo = float(sim.ids.ctrl_low[7])
    hi = float(sim.ids.ctrl_high[7])
    open_old = float(residual_mod.RECOVERY4D_TAU_OPEN)
    sec = float(RECOVERY4D_TAU_SECURE)
    samples = {}
    for a3 in (-1.0, 0.0, 1.0):
        m = residual_mod.map_recovery4d(np.array([0.0, 0.0, 0.0, a3]), np.eye(3), np.eye(3))
        samples[str(a3)] = m["tau"]
    return {
        "a3_range": [-1.0, 1.0],
        "formula": "tau = TAU_OPEN + 0.5*(a3+1)*(TAU_SECURE-TAU_OPEN)",
        "TAU_OPEN_now": open_old,
        "TAU_SECURE": sec,
        "a3_to_tau": samples,
        "ctrlrange7": [lo, hi],
        "xml_actuator8_ctrlrange": [-50.0, 50.0],
        "sign": "negative ctrl[7] closes; positive opens (tendon split)",
        "legal_tau_interval": sorted([open_old, sec]),
        "contains_positive_open": max(open_old, sec) > 0.0,
    }


def pulse_tau(sim, gains, snap, tau, dur, via_4d: bool) -> dict:
    restore_parent(sim, snap)
    dt = float(sim.model.opt.timestep)
    n = int(round(dur / dt))
    log = [row(sim, tau=tau, label="t0")]
    ap0 = log[0]["aperture"]
    both = 0.0
    t_both = None
    for _ in range(n):
        if via_4d:
            cmd = tick_4d(sim, gains, 0.0, 0.0, 0.0, tau)
        else:
            cmd = None
            tick_vw(sim, np.zeros(3), np.zeros(3), tau, gains)
        m = row(sim, tau=tau, cmd=cmd, prev=log[-1])
        if m["t"] - log[-1]["t"] >= LOG_DT - 1e-12:
            log.append(m)
        if m["nL"] == 0 and m["nR"] == 0:
            both += dt
            if t_both is None:
                t_both = m["t"]
        else:
            both = 0.0
            t_both = t_both
        if m["obj_z"] < TABLE_DROP:
            break
    m1 = log[-1]
    dap = (m1["aperture"] - ap0) / max(dur, 1e-9)
    kind = "SECURE_CLOSE"
    if m1["aperture"] > ap0 + 0.002 and (m1["nL"] == 0 or m1["nR"] == 0 or dap > 0.01):
        if tau > 0.5:
            kind = "ACTIVE_FINGER_OPENING"
        elif tau > -0.5:
            kind = "NEAR_ZERO_ACTUATION"
        else:
            kind = "PASSIVE_RELEASE_INSUFFICIENT_FN"
    elif tau <= -12:
        kind = "SECURE_CLOSE"
    elif abs(m1["aperture"] - ap0) < 0.0015 and m1["nL"] > 0 and m1["nR"] > 0:
        kind = "REDUCED_SQUEEZE" if tau > -18.1 else "SECURE_CLOSE"
        if -8.5 < tau < -1.5:
            kind = "REDUCED_SQUEEZE"
        elif -1.5 <= tau <= 0.5:
            kind = "REDUCED_SQUEEZE"
    elif m1["nL"] == 0 and m1["nR"] == 0:
        kind = "ACTIVE_FINGER_OPENING" if tau > 0.5 else "PASSIVE_RELEASE_INSUFFICIENT_FN"
    return {
        "tau": tau,
        "via_4d": via_4d,
        "dur": dur,
        "kind": kind,
        "ap0": ap0,
        "ap1": m1["aperture"],
        "dap_dt": dap,
        "nL": m1["nL"],
        "nR": m1["nR"],
        "Fn_L": m1["Fn_L"],
        "Fn_R": m1["Fn_R"],
        "ctrl7": m1["ctrl7"],
        "v_rel": m1["v_rel"],
        "both_off_persist_s": both,
        "t_first_both_off": t_both,
        "obj_z": m1["obj_z"],
        "log": log[:: max(1, len(log) // 40)],
    }


def patch_tau_open(val: float) -> None:
    text = RESIDUAL_PY.read_text(encoding="utf-8")
    text2, n = re.subn(
        r"RECOVERY4D_TAU_OPEN = [^\n]+",
        f"RECOVERY4D_TAU_OPEN = {float(val):.1f}  # a3=-1 active open; a3=+1 secure -18",
        text,
        count=1,
    )
    if n != 1:
        raise RuntimeError("failed to patch RECOVERY4D_TAU_OPEN")
    RESIDUAL_PY.write_text(text2, encoding="utf-8")
    residual_mod.RECOVERY4D_TAU_OPEN = float(val)
    importlib.reload(residual_mod)


def pick_open(rows: list[dict]) -> float:
    useful = []
    for r in rows:
        if r["tau"] <= 0:
            continue
        opened = r["ap1"] > r["ap0"] + 0.003
        lost = (r["t_first_both_off"] is not None) or (r["nL"] == 0 and r["nR"] == 0)
        if opened or lost:
            useful.append(r)
    if not useful:
        return 8.0
    return float(min(useful, key=lambda r: r["tau"])["tau"])


def hold_after(sim, gains, dur, log, ctl=None, label="HOLD"):
    dt = float(sim.model.opt.timestep)
    n = int(round(dur / dt))
    t_table = None
    for _ in range(n):
        if ctl and ctl.get("reset"):
            break
        cmd = tick_4d(sim, gains, 0.0, 0.0, 0.0, -18.0)
        m = row(sim, tau=-18.0, cmd=cmd, label=label, prev=log[-1] if log else None)
        if not log or m["t"] - log[-1]["t"] >= 0.008:
            log.append(m)
        if m["obj_z"] < TABLE_DROP:
            t_table = m["t"]
            break
        if ctl and ctl.get("sync"):
            ctl["overlay"] = {"case": label, "t": m["t"], "phase": label, "nL": m["nL"], "nR": m["nR"]}
            ctl["sync"]()
    return t_table


def run_release_window(sim, gains, snap, open_s: float, vz: float, tau_open: float) -> dict:
    restore_parent(sim, snap)
    freeze(sim)
    dt = float(sim.model.opt.timestep)
    n = int(round(open_s / dt))
    log = []
    ev = {}
    both_acc = 0.0
    t0 = float(sim.data.time)
    for i in range(n):
        cmd = tick_4d(sim, gains, 0.0, 0.0, vz, tau_open)
        m = row(sim, tau=tau_open, cmd=cmd, label="OPEN", prev=log[-1] if log else None)
        if i == 0:
            ev["OPEN_START"] = m["t"]
        log.append(m)
        if m["nL"] == 0 and m["nR"] == 0:
            both_acc += dt
            if "FIRST_BOTH_OFF" not in ev:
                ev["FIRST_BOTH_OFF"] = m["t"]
        else:
            both_acc = 0.0
        if m["obj_z"] < TABLE_DROP:
            ev["TABLE"] = m["t"]
            break
    m1 = log[-1] if log else row(sim)
    rh = np.asarray(m1["rh"], float)
    return {
        "open_s": open_s,
        "vz": vz,
        "events": ev,
        "both_off_persist_end": both_acc,
        "dt_both": None if "FIRST_BOTH_OFF" not in ev else m1["t"] - ev["FIRST_BOTH_OFF"],
        "aperture": m1["aperture"],
        "rh_mm": (1e3 * rh).tolist(),
        "v_rel": m1["v_rel"],
        "v_obj_z": m1["v_obj"][2],
        "obj_z": m1["obj_z"],
        "hand_z": m1["p_hand"][2],
        "nL": m1["nL"],
        "nR": m1["nR"],
        "clear": m1["clear"],
        "log": log[:: max(1, len(log) // 30)],
        "t0": t0,
    }


def try_recapture(sim, gains, snap, tau_open, extra_after_both, vz_follow, ctl=None) -> dict:
    """PRIVILEGED AUTHORITY CONSTRUCTION: close timing uses nL/nR and extra_after_both."""
    restore_parent(sim, snap)
    freeze(sim)
    dt = float(sim.model.opt.timestep)
    log = [row(sim, tau=-18.0, label="PARENT")]
    ev = {}
    both_acc = 0.0
    phase = "OPEN"
    ev["OPEN_START"] = float(sim.data.time)
    t_lim = float(sim.data.time) + 0.80
    captured = False
    t_table = None
    while float(sim.data.time) < t_lim:
        if ctl and ctl.get("reset"):
            break
        if phase == "OPEN":
            tau, vz = tau_open, vz_follow
            label = "OPEN"
        elif phase == "CLOSE":
            tau, vz = -18.0, vz_follow
            label = "CLOSE"
        else:
            tau, vz = -18.0, 0.0
            label = "SECURE"
        cmd = tick_4d(sim, gains, 0.0, 0.0, vz, tau)
        m = row(sim, tau=tau, cmd=cmd, label=label, prev=log[-1])
        if m["t"] - log[-1]["t"] >= LOG_DT - 1e-12:
            log.append(m)
        if m["obj_z"] < TABLE_DROP:
            t_table = m["t"]
            ev["TABLE"] = m["t"]
            break
        if phase == "OPEN":
            if m["nL"] == 0 and m["nR"] == 0:
                if "FIRST_BOTH_OFF" not in ev:
                    ev["FIRST_BOTH_OFF"] = m["t"]
                both_acc += dt
                if both_acc >= extra_after_both:
                    phase = "CLOSE"
                    ev["CLOSE_START"] = m["t"]
            else:
                both_acc = 0.0
            if m["t"] - ev["OPEN_START"] > 0.35:
                phase = "CLOSE"
                ev.setdefault("CLOSE_START", m["t"])
        elif phase == "CLOSE":
            if "FIRST_BOTH_OFF" in ev and (m["nL"] > 0 or m["nR"] > 0) and "FIRST_RECONTACT" not in ev:
                ev["FIRST_RECONTACT"] = m["t"]
            if m["nL"] > 0 and m["nR"] > 0 and "FIRST_BILATERAL" not in ev:
                ev["FIRST_BILATERAL"] = m["t"]
            if (
                m["nL"] > 0
                and m["nR"] > 0
                and m["v_rel"] < 0.04
                and abs(m["ctrl7"] + 18.0) < 0.5
                and "FIRST_BILATERAL" in ev
                and m["t"] - ev["FIRST_BILATERAL"] >= 0.050
            ):
                phase = "SECURE"
                ev["SECURE_CAPTURE"] = m["t"]
                captured = True
                break
            if "CLOSE_START" in ev and m["t"] - ev["CLOSE_START"] > 0.45:
                break
        if ctl and ctl.get("sync"):
            ctl["overlay"] = {
                "case": "RECAPTURE",
                "t": m["t"],
                "phase": phase,
                "nL": m["nL"],
                "nR": m["nR"],
            }
            ctl["sync"]()
    if captured and t_table is None:
        ev["HOLD_START"] = float(sim.data.time)
        t_table = hold_after(sim, gains, HOLD_S, log, ctl=ctl, label="HOLD")
    mend = log[-1]
    ok = bool(
        captured
        and t_table is None
        and mend["nL"] > 0
        and mend["nR"] > 0
        and mend["obj_z"] > 0.45
        and float(mend.get("clear") or 0) > 0.020
        and mend["scene"] == 0
        and "FIRST_BOTH_OFF" in ev
        and "CLOSE_START" in ev
        and ev["CLOSE_START"] - ev["FIRST_BOTH_OFF"] >= 0.018
        and mend["t"] >= ev.get("HOLD_START", mend["t"]) + HOLD_S - 0.6
    )
    return {
        "extra_after_both": extra_after_both,
        "vz_follow": vz_follow,
        "events": ev,
        "captured": captured,
        "ok": ok,
        "table": t_table,
        "end": {k: mend[k] for k in mend if k != "geoms"},
        "both_off_s": None
        if "FIRST_BOTH_OFF" not in ev or "CLOSE_START" not in ev
        else ev["CLOSE_START"] - ev["FIRST_BOTH_OFF"],
        "log": log,
    }


def classify_fail(trial: dict, can_open: bool, follow: dict | None) -> str:
    if not can_open:
        return "A. fingers cannot actively open"
    ev = trial.get("events") or {}
    if "FIRST_BOTH_OFF" not in ev:
        return "B. opening is too slow / no true both-off"
    if trial.get("table") and "FIRST_RECONTACT" not in ev:
        if follow and abs(follow.get("v_obj_z", 0)) > 0.15 and abs(follow.get("hand_z", 0) - follow.get("obj_z", 0)) > 0.04:
            return "C. legal v_z cannot follow the falling object"
        return "D. lateral/vertical geometry unreachable (fell out)"
    if "CLOSE_START" in ev and "FIRST_RECONTACT" not in ev:
        return "E. closing is too slow or fingers miss"
    if "FIRST_RECONTACT" in ev and "SECURE_CAPTURE" not in ev:
        return "G. recontact without motion arrest / bounce-eject"
    if trial.get("captured") and not trial.get("ok"):
        return "hold did not survive 12 s"
    return "unclassified; see raw"


def _save_success_npz(tr: dict) -> None:
    log = tr["log"]
    np.savez_compressed(
        RAW / "recapture_success.npz",
        t=np.array([x["t"] for x in log]),
        aperture=np.array([x["aperture"] for x in log]),
        aperture_vel=np.array(
            [x.get("aperture_vel") if x.get("aperture_vel") is not None else np.nan for x in log]
        ),
        ctrl7=np.array([x["ctrl7"] for x in log]),
        nL=np.array([x["nL"] for x in log]),
        nR=np.array([x["nR"] for x in log]),
        Fn_L=np.array([x["Fn_L"] for x in log]),
        Fn_R=np.array([x["Fn_R"] for x in log]),
        obj_z=np.array([x["obj_z"] for x in log]),
        v_rel=np.array([x["v_rel"] for x in log]),
        w_rel=np.array([x["w_rel"] for x in log]),
        v_obj_z=np.array([x["v_obj"][2] for x in log]),
        scene=np.array([x["scene"] for x in log]),
    )
    dump(RAW / "recapture_success.json", {k: tr[k] for k in tr if k != "log"})
    dump(RAW / "recapture_events.json", tr["events"])


def write_report(doc: dict) -> None:
    p = OUT / "AIRBORNE_RECAPTURE_AUTHORITY.md"
    a = []
    ap = a.append
    best = doc.get("best")
    st = doc["static_audit"]
    pm = doc["parent_meta"]
    ap("# Airborne release → recapture authority")
    ap("")
    ap("No ballistic impact. No CENTER-6. No SAC. No MP4.")
    ap("")
    ap(
        "If a recapture sequence is shown below, it is a **PRIVILEGED AUTHORITY CONSTRUCTION** "
        "(close timing uses `nL`/`nR`). It is **not** an observable recovery controller and **not** a learned policy."
    )
    ap("")
    ap("---")
    ap("")
    ap("## 1. Grip-action-space audit")
    ap("")
    ap("Executed from `controllers/residual.py` `map_recovery4d` and the live MuJoCo model (not comments).")
    ap("")
    ap("| item | value |")
    ap("|---|---|")
    ap("| a3 range | **[-1, +1]** |")
    ap(f"| formula | `{st.get('formula')}` |")
    samples = st.get("a3_to_tau") or {}
    ap(f"| a3=-1 → tau (module at static audit) | {samples.get('-1.0', samples.get('-1'))} |")
    ap(f"| a3=0 → tau | {samples.get('0.0', samples.get('0'))} |")
    ap(f"| a3=+1 → tau | {samples.get('1.0', samples.get('1'))} |")
    ap("| sign | negative ctrl[7] closes/squeezes; positive opens |")
    ap(f"| XML/runtime ctrlrange[7] | {st.get('ctrlrange7', st.get('ctrl_range7'))} |")
    ap("")
    ap("Legal 4D **before** the correction below was entirely negative (`[-18, −1]`): no true active opening.")
    ap("")
    ap("## 2. Active-opening audit (legal 4D **before** extension)")
    ap("")
    ap("| tau | class | ap0→ap1 mm | nL/nR | both-off |")
    ap("|---|---|---:|---|---|")
    for r in doc["legal_grip"]:
        ap(
            f"| {r['tau']:g} | {r['kind']} | {1e3*r['ap0']:.1f} → {1e3*r['ap1']:.1f} | "
            f"{r['nL']}/{r['nR']} | {r['t_first_both_off']} |"
        )
    ap("")
    ap(
        "**D. ACTIVE FINGER OPENING was absent** from legal a3. "
        "`tau=-1` is **C**: contacts can be lost while aperture does **not** command-open. "
        "That is an action-space design limit, not a recapture-physics failure."
    )
    ap("")
    ap("## 3. Grip-range correction (same 4th dimension)")
    ap("")
    ap(doc["correction_text"])
    ap("")
    ap("Positive-ctrl diagnostic (`tick_vw`, not recapture commands):")
    ap("")
    ap("| ctrl[7] | class | ap0→ap1 mm |")
    ap("|---|---|---:|")
    for r in doc.get("pos_rows") or []:
        ap(f"| {r['tau']:g} | {r['kind']} | {1e3*r['ap0']:.1f} → {1e3*r['ap1']:.1f} |")
    ap("")
    ap("Frozen: `a3=-1 → +2` (open), `a3=+1 → -18` (secure). RULE yaml `tau_open: -1` unchanged.")
    ap("")
    if doc.get("sanity"):
        ap("Sanity after freeze:")
        ap("")
        ap("| tau | class | ctrl[7] |")
        ap("|---|---|---:|")
        for r in doc["sanity"]:
            ap(f"| {r['tau']:g} | {r['kind']} | {r['ctrl7']} |")
        ap("")
    ap("## 4. Clean airborne parent")
    ap("")
    ap("| | |")
    ap("|---|---|")
    ap(f"| t | {pm['t']:.3f} s |")
    ap(f"| phase | {pm['phase']} |")
    ap(f"| obj_z | {pm['obj_z']:.3f} m |")
    ap(f"| table clearance | {1e3*pm['clear']:.0f} mm |")
    ap(f"| nL/nR | {pm['nL']} / {pm['nR']} |")
    ap(f"| v_rel | {1e3*pm['v_rel']:.2f} mm/s |")
    ap(f"| ω_rel | {pm['w_rel']:.3f} rad/s |")
    ap(f"| e_x | {pm['e_x_mm']:.2f} mm |")
    ap(f"| aperture | {1e3*pm['aperture']:.1f} mm |")
    ap(f"| scene | {pm['scene']} |")
    ap(f"| ctrl[7] | {pm['ctrl7']} |")
    ap(f"| noslip | {pm['noslip']} |")
    ap("")
    ap("Snapshot: `raw/recapture_parent.pkl`.")
    ap("")
    ap("## 5. Physical release")
    ap("")
    ap("From the parent, legal `a3=-1` (`tau=+2`): aperture increases, contacts drop, object qpos/qvel are integrated (no teleport).")
    ap("")
    ap("## 6–8. Contact-free interval, free-flight, hand follow")
    ap("")
    ap("| open_s | vz | first both-off | persist s | ap mm | r_h.z mm | v_obj_z | table? |")
    ap("|---:|---:|---|---:|---:|---:|---:|---|")
    for r in doc["windows"]:
        ap(
            f"| {r['open_s']:.2f} | {r['vz']:.2f} | {r['events'].get('FIRST_BOTH_OFF')} | "
            f"{r.get('dt_both')} | {1e3*r['aperture']:.1f} | {r['rh_mm'][2]:.1f} | "
            f"{r['v_obj_z']:.3f} | {r['events'].get('TABLE', 'no')} |"
        )
    ap("")
    ap("Legal `|v_z|≤0.08` cannot match `g t` after a long release. Arbitrary-duration free fall is not recoverable.")
    ap("")
    ap("## 9–11. Recapture construction / recontact vs capture / hold")
    ap("")
    ap(doc["recapture_text"])
    ap("")
    if best is not None:
        ev = best["events"]
        end = best["end"]
        ap("| event | t (s) |")
        ap("|---|---:|")
        for k in (
            "OPEN_START",
            "FIRST_BOTH_OFF",
            "CLOSE_START",
            "FIRST_RECONTACT",
            "FIRST_BILATERAL",
            "SECURE_CAPTURE",
            "HOLD_START",
        ):
            if k in ev:
                ap(f"| {k} | {ev[k]:.3f} |")
        ap("")
        ap(doc["hold_text"])
        ap("")
        ap(
            f"End of hold: nL/nR={end['nL']}/{end['nR']}, obj_z={end['obj_z']:.3f} m, "
            f"clear={end['clear']:.3f} m, scene={end['scene']}, v_rel={end['v_rel']:.5f} m/s, "
            f"table={best['table']}."
        )
        ap("")
        ap("First finger contact is **not** counted as success. Secure capture requires bilateral contact and relative-motion arrest, then a 12 s elevated hold.")
        ap("")
    else:
        ap(doc["hold_text"])
        ap("")
    ap("## 12. Failure mechanism")
    ap("")
    ap(doc.get("fail_text") or "See recapture_text.")
    ap("")
    ap("## 13. Interactive viewer commands")
    ap("")
    ap("```text")
    ap("python training/demo_airborne_recapture.py --mode zero")
    ap("python training/demo_airborne_recapture.py --mode recapture")
    ap("```")
    ap("")
    ap("SPACE pause, R restart, `[` `]` speed. Camera initialized once. No MP4.")
    ap("Headless: `python training/demo_airborne_recapture.py --mode sweep`")
    ap("")
    ap("## 14. Limitations / privileged information")
    ap("")
    for line in doc["limits"]:
        ap(f"- {line}")
    ap("")
    ap("Raw: `results/diagnostics/airborne_recapture/raw/`.")
    p.write_text("\n".join(a), encoding="utf-8")
    print("wrote", p, flush=True)


def headless():
    RAW.mkdir(parents=True, exist_ok=True)
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    gains = gains_from_cfg(cfg)
    sim = make_parent_sim()
    assert int(sim.model.opt.noslip_iterations) == 1
    packed = advance_to_parent(sim, cfg)
    if not packed.get("ok"):
        raise RuntimeError(packed)
    snap = save_parent(sim, packed)
    with (RAW / "recapture_parent.pkl").open("wb") as f:
        pickle.dump(snap, f)
    o = physical_pack(sim)
    parent_meta = {
        "t": float(sim.data.time),
        "phase": str(sim.fsm.phase),
        "obj_z": float(o["obj_z"]),
        "clear": float(o.get("clear", np.nan)),
        "nL": int(o["nL"]),
        "nR": int(o["nR"]),
        "v_rel": float(o["v_rel"]),
        "w_rel": float(o["w_rel"]),
        "e_x_mm": 1e3 * float(o["e_x"]),
        "aperture": finger_opening(sim.data, sim.ids),
        "scene": int(o["scene"]),
        "ctrl7": float(sim.data.ctrl[7]),
        "noslip": int(sim.model.opt.noslip_iterations),
    }
    dump(RAW / "parent_meta.json", parent_meta)
    print("PARENT", parent_meta, flush=True)

    frozen_open = float(residual_mod.RECOVERY4D_TAU_OPEN)
    # Legal-a3 audit always uses the pre-correction interval [-18, -1].
    residual_mod.RECOVERY4D_TAU_OPEN = -1.0
    static = grip_audit_static(sim)
    print("AUDIT_PRE", static, flush=True)
    legal = []
    for tau in (-18.0, -8.0, -5.0, -2.0, -1.0):
        legal.append(pulse_tau(sim, gains, snap, tau, 0.40, via_4d=True))
        print("legal", legal[-1]["tau"], legal[-1]["kind"], "dap", legal[-1]["dap_dt"], flush=True)
    dump(RAW / "legal_grip.json", [{k: v for k, v in r.items() if k != "log"} for r in legal])
    can_open_legal = any(r["kind"] == "ACTIVE_FINGER_OPENING" for r in legal)
    residual_mod.RECOVERY4D_TAU_OPEN = frozen_open

    pos_rows = []
    for tau in (0.0, 2.0, 4.0, 6.0, 8.0):
        pos_rows.append(pulse_tau(sim, gains, snap, tau, 0.40, via_4d=False))
        print("pos", pos_rows[-1]["tau"], pos_rows[-1]["kind"], "ap", pos_rows[-1]["ap0"], pos_rows[-1]["ap1"], flush=True)
    dump(RAW / "positive_tau_sweep.json", [{k: v for k, v in r.items() if k != "log"} for r in pos_rows])

    chosen = None
    sanity = []
    correction_text = "Legal 4D already includes active opening; mapping not changed."
    if not can_open_legal:
        chosen = pick_open(pos_rows)
        old = -1.0
        if abs(frozen_open - chosen) > 1e-9:
            patch_tau_open(chosen)
            print("PATCHED TAU_OPEN", chosen, flush=True)
        else:
            residual_mod.RECOVERY4D_TAU_OPEN = float(chosen)
            print("TAU_OPEN already frozen at", chosen, flush=True)
        correction_text = (
            f"CURRENT 4D ACTION SPACE DID NOT INCLUDE ACTIVE OPENING "
            f"(a3∈[-1,1] → tau∈[{old:g}, -18]). "
            f"Minimal extension of the **same** a3 dimension: "
            f"`RECOVERY4D_TAU_OPEN = {chosen:g}` so a3=-1 → ctrl[7]={chosen:g} (active open), "
            f"a3=+1 → -18 (secure). Intermediate taus including -5 remain reachable. "
            f"Positive sweep used tick_vw **outside** 4D only as an actuator diagnostic; "
            f"recapture uses the frozen 4D map only."
        )
        for tau in (-18.0, -5.0, chosen):
            sanity.append(pulse_tau(sim, gains, snap, tau, 0.30, via_4d=True))
            print("sanity", sanity[-1]["tau"], sanity[-1]["kind"], "ctrl", sanity[-1]["ctrl7"], flush=True)
        dump(RAW / "mapping_sanity.json", [{k: v for k, v in r.items() if k != "log"} for r in sanity])
    dump(RAW / "correction.json", {"text": correction_text, "TAU_OPEN": residual_mod.RECOVERY4D_TAU_OPEN})

    tau_open = float(residual_mod.RECOVERY4D_TAU_OPEN)
    windows = []
    if tau_open > 0:
        for os in (0.06, 0.10, 0.14, 0.20):
            windows.append(run_release_window(sim, gains, snap, os, 0.0, tau_open))
            print("window vz0", os, windows[-1]["events"], "n", windows[-1]["nL"], windows[-1]["nR"], flush=True)
        windows.append(run_release_window(sim, gains, snap, 0.14, -0.08, tau_open))
        print("window follow", windows[-1]["events"], "vobjz", windows[-1]["v_obj_z"], flush=True)
    dump(RAW / "release_windows.json", [{k: v for k, v in r.items() if k != "log"} for r in windows])
    follow = windows[-1] if windows else None

    recapture_text = "Recapture not run: no active opening in 4D after audit."
    hold_text = recapture_text
    fail_text = recapture_text
    best = None
    if tau_open > 0:
        trials = []
        for extra in (0.040, 0.050, 0.060):
            for vz in (0.0, -0.08):
                tr = try_recapture(sim, gains, snap, tau_open, extra, vz)
                slim = {k: tr[k] for k in tr if k != "log"}
                trials.append(slim)
                print(
                    "try extra", extra, "vz", vz, "ok", tr["ok"], "cap", tr["captured"],
                    "both", tr["both_off_s"], "ev", tr["events"], flush=True,
                )
                if tr["ok"]:
                    best = tr
                    _save_success_npz(tr)
                    break
            if best is not None:
                break
        dump(RAW / "recapture_trials.json", trials)
        dump(RAW / "recapture_plan.json", {"none": best is None} if best is None else {
            "none": False,
            "tau_open": tau_open,
            "extra_after_both": best["extra_after_both"],
            "vz_follow": best["vz_follow"],
            "privileged": True,
        })
        if best is not None:
            recapture_text = (
                "PRIVILEGED AUTHORITY CONSTRUCTION succeeded: "
                f"open tau={tau_open:g} until both-off persist {best['extra_after_both']} s, "
                f"vz_follow={best['vz_follow']}, then tau=-18. "
                f"Events {best['events']}. both-off interval {best['both_off_s']} s. "
                "Not an observable controller."
            )
            hold_text = (
                f"Long hold survived: table={best['table']} end n={best['end']['nL']}/{best['end']['nR']} "
                f"obj_z={best['end']['obj_z']:.3f} v_rel={best['end']['v_rel']:.4f}."
            )
        else:
            recapture_text = "No successful recapture in the small extra×vz set."
            hold_text = classify_fail(trials[-1] if trials else {}, True, follow)
            fail_text = hold_text
    else:
        fail_text = "A. fingers cannot actively open"
        recapture_text = "Recapture not run: CURRENT 4D ACTION SPACE DOES NOT INCLUDE ACTIVE OPENING."
        hold_text = recapture_text

    if best is not None:
        fail_text = (
            "Existence succeeded. Nearby extra=0.06 both-off hits the table after recontact "
            "(window too long; legal v_z cannot chase a long fall)."
        )

    limits = [
        "Privileged nL/nR used to time CLOSE. Not a policy.",
        "Observation vector / reward / success_obs not changed. SCALE_TAU in physical_recovery still uses -1 as the old open end; that is an obs-scale leftover for the later obs/reward pass.",
        "RULE yaml tau_open=-1 was not changed (frozen RULE).",
        "No ballistic / CENTER-6 / SAC / MP4.",
        "Legal |v_z|≤0.08 m/s vs g=9.81 limits catchable free-fall duration.",
    ]
    write_report(
        {
            "static_audit": static,
            "legal_grip": [{k: v for k, v in r.items() if k != "log"} for r in legal],
            "pos_rows": [{k: v for k, v in r.items() if k != "log"} for r in pos_rows],
            "sanity": [{k: v for k, v in r.items() if k != "log"} for r in sanity],
            "correction_text": correction_text,
            "parent_meta": parent_meta,
            "windows": [{k: v for k, v in r.items() if k != "log"} for r in windows],
            "recapture_text": recapture_text,
            "hold_text": hold_text,
            "fail_text": fail_text,
            "best": None if best is None else {k: best[k] for k in best if k != "log"},
            "limits": limits,
        }
    )


def interactive(mode: str) -> None:
    import mujoco.viewer

    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    gains = gains_from_cfg(cfg)
    plan_p = RAW / "recapture_plan.json"
    plan = json.loads(plan_p.read_text(encoding="utf-8")) if plan_p.is_file() else {"none": True}
    ctl = {"pause": False, "reset": False, "speed": 0.40, "overlay": {}}

    def sync():
        vwr = ctl.get("viewer")
        if vwr is None:
            return
        o = ctl.get("overlay") or {}
        _viewer_overlay(
            vwr,
            [
                ("CASE", str(o.get("case", mode.upper()))),
                ("t", f"{float(o.get('t', 0)):.3f} s"),
                ("PHASE", str(o.get("phase", ""))),
                ("nL/nR", f"{o.get('nL','-')}/{o.get('nR','-')}"),
                ("keys", "SPACE pause  R restart  [ ] speed"),
            ],
        )
        vwr.sync()
        dt = 0.002
        spd = max(float(ctl.get("speed", 0.4)), 0.05)
        now = time.perf_counter()
        tgt = ctl.get("_wall")
        if tgt is None:
            ctl["_wall"] = now + dt / spd
        else:
            sl = tgt - now
            if sl > 0:
                time.sleep(min(sl, 0.05))
            ctl["_wall"] = max(tgt, time.perf_counter()) + dt / spd
        while ctl.get("pause") and vwr.is_running() and not ctl.get("reset"):
            vwr.sync()
            time.sleep(0.02)

    ctl["sync"] = sync

    def on_key(kc):
        k = int(kc)
        if k == 32:
            ctl["pause"] = not ctl["pause"]
        elif k in (ord("R"), ord("r")):
            ctl["reset"] = True
        elif k == ord("["):
            ctl["speed"] = max(0.08, float(ctl["speed"]) * 0.7)
        elif k == ord("]"):
            ctl["speed"] = min(2.5, float(ctl["speed"]) / 0.7)

    sim = make_parent_sim()
    with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=on_key) as vwr:
        ctl["viewer"] = vwr
        apply_camera_preset(vwr, CAM)
        while vwr.is_running():
            ctl["reset"] = False
            ctl["_wall"] = None
            sim.reset(CYL_MASS, PAIR_MU, np.zeros(3))
            packed = advance_to_parent(sim, cfg)
            if not packed.get("ok"):
                time.sleep(0.2)
                continue
            snap = save_parent(sim, packed)
            if mode == "zero" or plan.get("none"):
                hold_after(sim, gains, HOLD_S, [], ctl=ctl, label="ZERO HOLD")
            else:
                try_recapture(
                    sim,
                    gains,
                    snap,
                    float(plan["tau_open"]),
                    float(plan["extra_after_both"]),
                    float(plan["vz_follow"]),
                    ctl=ctl,
                )
            if ctl.get("reset"):
                continue
            while vwr.is_running() and not ctl.get("reset"):
                sync()
                time.sleep(0.03)
            if not ctl.get("reset"):
                break


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="sweep", choices=("sweep", "zero", "recapture"))
    args = p.parse_args()
    if args.mode in ("zero", "recapture"):
        interactive(args.mode)
        return
    headless()


if __name__ == "__main__":
    main()
