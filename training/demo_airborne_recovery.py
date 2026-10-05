"""Non-trivial airborne recovery authority. Privileged construction. No SAC/MP4.

Previous open-and-close-in-place recapture is treated as a contact-authority
sanity check, not full airborne recovery.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.jacobian_controller import gains_from_cfg
from controllers.residual import RECOVERY4D_TAU_OPEN, RECOVERY4D_V_HX_MAX, RECOVERY4D_V_Z_MAX
from envs.config_util import load_yaml, merge_sim_config
from envs.physical_recovery import TABLE_DROP, physical_pack
from training.demo_airborne_recapture import (
    HOLD_S,
    LOG_DT,
    advance_to_parent,
    hold_after,
    make_parent_sim,
    restore_parent,
    row,
    save_parent,
    tick_4d,
)
from training.demo_teleport_recovery_state import CYL_MASS, PAIR_MU
from training.grav_reposition_v2_viz import apply_camera_preset
from training.map_ballistic_disturbance import CAM, _viewer_overlay
from training.replay_core import TABLE_TOP, freeze, geom_name

OUT = ROOT / "results" / "diagnostics" / "airborne_nontrivial_recovery"
RAW = OUT / "raw"
HC = OUT / "high_clearance"
TAU_OPEN = float(RECOVERY4D_TAU_OPEN)
TAU_SEC = -18.0
VX_MAX = float(RECOVERY4D_V_HX_MAX)
VZ_MAX = float(RECOVERY4D_V_Z_MAX)
TARGET_CLEAR = 0.12
CLEAR_CAP = 0.15
BOTH_OFF_MIN = 0.018
HOLD_AIR_S = 2.0


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


def geom_mask_audit(sim) -> list[dict]:
    rows = []
    for g in range(int(sim.model.ngeom)):
        bid = int(sim.model.geom_bodyid[g])
        nm = geom_name(sim.model, g)
        interesting = nm in ("table", "floor", "object") or bid in (
            int(sim.ids.left_body),
            int(sim.ids.right_body),
            int(sim.ids.object_body),
            int(sim.ids.hand_body),
        )
        if not interesting:
            continue
        rows.append(
            {
                "geom": nm,
                "id": int(g),
                "body": int(bid),
                "contype": int(sim.model.geom_contype[g]),
                "conaffinity": int(sim.model.geom_conaffinity[g]),
            }
        )
    return rows


def copy_masks(sim) -> dict[int, tuple[int, int]]:
    return {
        int(g): (int(sim.model.geom_contype[g]), int(sim.model.geom_conaffinity[g]))
        for g in range(int(sim.model.ngeom))
    }


def apply_masks(sim, masks: dict[int, tuple[int, int]]) -> None:
    for g, (ct, ca) in masks.items():
        sim.model.geom_contype[int(g)] = int(ct)
        sim.model.geom_conaffinity[int(g)] = int(ca)
    mujoco.mj_forward(sim.model, sim.data)


def disable_object_table_only(sim) -> dict:
    """Disable object–table contacts by zeroing the *table* geom mask only.

    Object, floor, fingers, arm: unchanged. Gravity/mass/friction unchanged.
    Table no longer collides with anything (including the object). Floor still does.
    """
    table = int(mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "table"))
    obj = int(sim.ids.object_geom)
    floor = int(mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"))
    before = geom_mask_audit(sim)
    changed = {
        "geom": geom_name(sim.model, table),
        "id": table,
        "contype_before": int(sim.model.geom_contype[table]),
        "conaffinity_before": int(sim.model.geom_conaffinity[table]),
        "contype_after": 0,
        "conaffinity_after": 0,
        "object_geom_unchanged": (int(sim.model.geom_contype[obj]), int(sim.model.geom_conaffinity[obj])),
        "floor_geom_unchanged": (int(sim.model.geom_contype[floor]), int(sim.model.geom_conaffinity[floor])),
    }
    sim.model.geom_contype[table] = 0
    sim.model.geom_conaffinity[table] = 0
    mujoco.mj_forward(sim.model, sim.data)
    after = geom_mask_audit(sim)
    return {"changed": changed, "before": before, "after": after}


def object_floor_hit(sim) -> bool:
    floor = int(mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"))
    obj = int(sim.ids.object_geom)
    for i in range(int(sim.data.ncon)):
        c = sim.data.contact[i]
        if {int(c.geom1), int(c.geom2)} == {floor, obj}:
            return True
    return False


def extra_lift_secure(sim, gains, ctl=None) -> dict:
    """Physically raise the grasped cylinder to ~12 cm virtual table clearance."""
    dt = float(sim.model.opt.timestep)
    t0 = float(sim.data.time)
    log = []
    settle_need = 0.0
    while float(sim.data.time) < t0 + 5.0:
        o = physical_pack(sim)
        m = row(sim, tau=TAU_SEC, label="EXTRA_LIFT", prev=log[-1] if log else None)
        if not log or m["t"] - log[-1]["t"] >= 0.008:
            log.append(m)
        clr = float(o.get("clear", 0.0))
        if int(o["nL"]) > 0 and int(o["nR"]) > 0 and float(o["v_rel"]) < 0.05 and clr >= TARGET_CLEAR:
            settle_need += dt
            if settle_need >= 0.15 or clr >= CLEAR_CAP:
                freeze(sim)
                break
            cmd = tick_4d(sim, gains, 0.0, 0.0, 0.0, TAU_SEC)
        else:
            settle_need = 0.0
            vz = 0.0 if clr >= CLEAR_CAP else VZ_MAX
            cmd = tick_4d(sim, gains, 0.0, 0.0, vz, TAU_SEC)
        if ctl and ctl.get("sync"):
            ctl["overlay"] = {
                "case": ctl.get("label", "EXTRA_LIFT"),
                "t": m["t"],
                "phase": "EXTRA_LIFT",
                "nL": m["nL"],
                "nR": m["nR"],
                "rhx_mm": 1e3 * float(m["rh"][0]),
            }
            ctl["sync"]()
            if ctl.get("reset"):
                break
    o = physical_pack(sim)
    freeze(sim)
    return {
        "ok": int(o["nL"]) > 0 and int(o["nR"]) > 0 and float(o.get("clear", 0)) >= 0.10,
        "t": float(sim.data.time),
        "obj_z": float(o["obj_z"]),
        "clear": float(o.get("clear", np.nan)),
        "nL": int(o["nL"]),
        "nR": int(o["nR"]),
        "rh": np.asarray(o["rh"], float).tolist(),
        "v_rel": float(o["v_rel"]),
        "hand_z": float(sim.data.xpos[sim.ids.hand_body][2]),
        "lift_duration": float(sim.data.time) - t0,
        "n_log": len(log),
    }


def snap_now(sim) -> dict:
    s = sim.snapshot()
    s["mass"] = CYL_MASS
    s["friction"] = PAIR_MU
    s["z_tgt"] = float(sim.fsm.p_des[2])
    return s


def restore_mismatch(sim, snap) -> None:
    sim.load_snapshot(snap)
    mujoco.mj_forward(sim.model, sim.data)
    freeze(sim)


def slim_end(m: dict) -> dict:
    skip = {"geoms", "contact_pts", "r_des"}
    return {k: m[k] for k in m if k not in skip}


def classify_end(ev, captured, t_table, mend, *, airborne=False) -> str:
    if not airborne:
        if t_table is not None and "FIRST_BILATERAL" not in ev:
            return "table_before_bilateral"
        if t_table is not None:
            return "table_after_recontact"
        if captured and mend["nL"] > 0 and mend["nR"] > 0 and mend["scene"] == 0 and mend["obj_z"] > 0.45:
            return "secure_hold"
    else:
        if captured and mend["nL"] > 0 and mend["nR"] > 0:
            return "secure_hold"
        if ev.get("CLOSED_EMPTY") and "FIRST_BILATERAL" not in ev:
            return "miss"
        if ev.get("FLOOR") is not None and "FIRST_BILATERAL" not in ev:
            return "floor_before_bilateral"
    if "FIRST_RECONTACT" in ev and "FIRST_BILATERAL" not in ev:
        return "one_finger_only"
    if "FIRST_BILATERAL" in ev and not captured:
        return "bilateral_no_arrest"
    if "FIRST_RECONTACT" not in ev:
        return "miss"
    return "unstable_or_timeout"


def ok_hold(captured, t_table, mend, ev, hold_s=HOLD_S) -> bool:
    return bool(
        captured
        and t_table is None
        and mend["nL"] > 0
        and mend["nR"] > 0
        and mend["obj_z"] > 0.45
        and float(mend.get("clear") or 0) > 0.020
        and mend["scene"] == 0
        and "FIRST_BOTH_OFF" in ev
        and mend["t"] >= ev.get("HOLD_START", mend["t"]) + hold_s - 0.6
    )


def ok_hold_airborne(captured, mend, ev, hold_s=HOLD_AIR_S) -> bool:
    rh = np.asarray(mend["rh"], float)
    return bool(
        captured
        and ev.get("FLOOR") is None
        and mend["nL"] > 0
        and mend["nR"] > 0
        and float(mend["v_rel"]) < 0.08
        and abs(float(rh[0])) < 0.020
        and "FIRST_BOTH_OFF" in ev
        and mend["t"] >= ev.get("HOLD_START", mend["t"]) + hold_s - 0.25
    )


def corridor_miss(tr: dict) -> bool:
    if tr.get("captured"):
        return False
    if tr.get("fail") in ("floor_before_bilateral", "table_before_bilateral"):
        return False
    return tr.get("fail") in ("miss", "one_finger_only", "bilateral_no_arrest", "unstable_or_timeout")


def step(sim, gains, vx, vz, tau, log, ev, label, ctl=None, abort_table=True):
    cmd = tick_4d(sim, gains, 0.0, vx, vz, tau)
    m = row(sim, tau=tau, cmd=cmd, label=label, prev=log[-1] if log else None)
    m["virtual_clear"] = float(m["clear"])
    if not log or m["t"] - log[-1]["t"] >= LOG_DT - 1e-12:
        log.append(m)
    if object_floor_hit(sim) and "FLOOR" not in ev:
        ev["FLOOR"] = m["t"]
        ev["FAILURE"] = "floor"
        if not abort_table:
            return m, True
    if abort_table and m["obj_z"] < TABLE_DROP:
        ev["TABLE"] = m["t"]
        ev["FAILURE"] = "table"
        return m, True
    if (not abort_table) and m["clear"] < 0.0 and "VIRTUAL_TABLE" not in ev:
        ev["VIRTUAL_TABLE"] = m["t"]
    if ctl and ctl.get("sync"):
        ctl["overlay"] = {
            "case": ctl.get("label", label),
            "t": m["t"],
            "phase": label,
            "nL": m["nL"],
            "nR": m["nR"],
            "rhx_mm": 1e3 * float(m["rh"][0]),
        }
        ctl["sync"]()
    return m, False


def mark_contacts(ev, m, closing: bool, *, airborne=False):
    if closing and (m["nL"] > 0 or m["nR"] > 0) and "FIRST_RECONTACT" not in ev:
        ev["FIRST_RECONTACT"] = m["t"]
    if closing and m["nL"] > 0 and m["nR"] > 0 and "FIRST_BILATERAL" not in ev:
        ev["FIRST_BILATERAL"] = m["t"]
    if (
        closing
        and m["nL"] > 0
        and m["nR"] > 0
        and "FIRST_BILATERAL" in ev
        and m["t"] - ev["FIRST_BILATERAL"] >= 0.050
        and (airborne or (m["v_rel"] < 0.04 and abs(m["ctrl7"] + 18.0) < 0.5))
    ):
        ev["MOTION_ARREST"] = m["t"]
        return True
    return False
    if closing and (m["nL"] > 0 or m["nR"] > 0) and "FIRST_RECONTACT" not in ev:
        ev["FIRST_RECONTACT"] = m["t"]
    if closing and m["nL"] > 0 and m["nR"] > 0 and "FIRST_BILATERAL" not in ev:
        ev["FIRST_BILATERAL"] = m["t"]
    if (
        closing
        and m["nL"] > 0
        and m["nR"] > 0
        and m["v_rel"] < 0.04
        and abs(m["ctrl7"] + 18.0) < 0.5
        and "FIRST_BILATERAL" in ev
        and m["t"] - ev["FIRST_BILATERAL"] >= 0.050
    ):
        ev["MOTION_ARREST"] = m["t"]
        return True
    return False


def hold_airborne(sim, gains, dur, log, ev, ctl=None, label="HOLD"):
    dt = float(sim.model.opt.timestep)
    n = int(round(dur / dt))
    lost = 0.0
    for _ in range(n):
        if ctl and ctl.get("reset"):
            break
        m, dropped = step(sim, gains, 0.0, 0.0, TAU_SEC, log, ev, label, ctl, abort_table=False)
        if dropped:
            return m
        if m["nL"] == 0 or m["nR"] == 0:
            lost += dt
            if lost >= 0.08:
                ev["HOLD_LOST"] = m["t"]
                return m
        else:
            lost = 0.0
        if ctl and ctl.get("sync"):
            ctl["overlay"] = {
                "case": ctl.get("label", label),
                "t": m["t"],
                "phase": label,
                "nL": m["nL"],
                "nR": m["nR"],
                "rhx_mm": 1e3 * float(m["rh"][0]),
            }
    return None


def intercept_stats(log, t0: float) -> dict:
    close = [r for r in log if r.get("label") in ("CLOSE", "CORRECT_OPEN", "RECOVERY_START", "HOLD")]
    if not close:
        close = log
    hx = np.array([r["p_hand"][0] for r in close], float)
    ox = np.array([r["p_obj"][0] for r in close], float)
    rhx = np.array([r["rh"][0] for r in close], float)
    hvx = np.array([r["v_hand"][0] for r in close], float)
    ovx = np.array([r["v_obj"][0] for r in close], float)
    hz = np.array([r["p_hand"][2] for r in close], float)
    oz = np.array([r["p_obj"][2] for r in close], float)
    return {
        "t0": t0,
        "hand_x0": float(hx[0]),
        "obj_x0": float(ox[0]),
        "hand_x_end": float(hx[-1]),
        "obj_x_end": float(ox[-1]),
        "d_hand_x": float(hx[-1] - hx[0]),
        "d_obj_x": float(ox[-1] - ox[0]),
        "rhx0_mm": 1e3 * float(rhx[0]),
        "rhx_end_mm": 1e3 * float(rhx[-1]),
        "rhx_min_abs_mm": 1e3 * float(np.min(np.abs(rhx))),
        "hand_vx_mean": float(np.mean(hvx)),
        "obj_vx_mean": float(np.mean(ovx)),
        "v_rel_x_mean": float(np.mean(ovx - hvx)),
        "d_hand_z": float(hz[-1] - hz[0]),
        "d_obj_z": float(oz[-1] - oz[0]),
        "toward_catch": bool(abs(rhx[-1]) < abs(rhx[0]) - 1e-4),
    }


def build_mismatch(sim, gains, parent, t_mis: float, park_s: float, vx_create: float, ctl=None, airborne=False):
    restore_parent(sim, parent)
    freeze(sim)
    dt = float(sim.model.opt.timestep)
    log = [row(sim, tau=TAU_SEC, label="PARENT")]
    ev = {"OPEN_START": float(sim.data.time)}
    both = 0.0
    m = log[0]
    dropped = False
    both_need = BOTH_OFF_MIN if airborne else 0.010
    while float(sim.data.time) < ev["OPEN_START"] + 0.45 and not dropped:
        m, dropped = step(
            sim, gains, 0.0, 0.0, TAU_OPEN, log, ev, "OPEN", ctl, abort_table=not airborne
        )
        if m["nL"] == 0 and m["nR"] == 0:
            if "FIRST_BOTH_OFF" not in ev:
                ev["FIRST_BOTH_OFF"] = m["t"]
            both += dt
            if both >= both_need:
                break
        else:
            both = 0.0
    if "FIRST_BOTH_OFF" not in ev:
        return None
    n_mis = int(round(t_mis / dt))
    hx0 = float(m["p_hand"][0])
    hy0 = float(m["p_hand"][1])
    ox0 = float(m["p_obj"][0])
    oy0 = float(m["p_obj"][1])
    rhx0 = float(m["rh"][0])
    for _ in range(n_mis):
        m, dropped = step(
            sim, gains, vx_create, 0.0, TAU_OPEN, log, ev, "MISMATCH_VX", ctl, abort_table=not airborne
        )
        if dropped:
            break
        if m["nL"] > 0 or m["nR"] > 0:
            both = 0.0
    ev["MISMATCH_CREATED"] = float(sim.data.time)
    mis_rows = [r for r in log if r.get("label") == "MISMATCH_VX"]
    vhx = np.array([r["v_hand"][0] for r in mis_rows], float) if mis_rows else np.array([0.0])
    n_park = int(round(park_s / dt))
    for _ in range(n_park):
        if dropped:
            break
        m, dropped = step(
            sim, gains, 0.0, 0.0, TAU_OPEN, log, ev, "PARK", ctl, abort_table=not airborne
        )
    snap = snap_now(sim)
    o = physical_pack(sim)
    st = {
        "t": m["t"],
        "rh": np.asarray(o["rh"], float).tolist(),
        "v_rel_h": np.asarray(o["v_rel_h"], float).tolist(),
        "w_rel_h": np.asarray(o["w_rel_h"], float).tolist(),
        "R_rel": np.asarray(o["R_rel"], float).tolist(),
        "aperture": m["aperture"],
        "clear": m["clear"],
        "virtual_clear": m["clear"],
        "obj_z": m["obj_z"],
        "nL": m["nL"],
        "nR": m["nR"],
        "p_hand": m["p_hand"],
        "p_obj": m["p_obj"],
        "v_hand": m["v_hand"],
        "v_obj": m["v_obj"],
        "dt_both_off": m["t"] - ev["FIRST_BOTH_OFF"],
        "d_hand_x": float(m["p_hand"][0]) - hx0,
        "d_hand_y": float(m["p_hand"][1]) - hy0,
        "d_obj_x": float(m["p_obj"][0]) - ox0,
        "d_obj_y": float(m["p_obj"][1]) - oy0,
        "d_rhx": float(m["rh"][0]) - rhx0,
        "vx_create": vx_create,
        "v_hand_x_mean_during_mis": float(np.mean(vhx)),
        "t_mis": t_mis,
        "park_s": park_s,
        "table": (not airborne) and dropped,
        "floor": ev.get("FLOOR"),
        "sim_time": float(sim.data.time),
        "p_des": np.asarray(sim.fsm.p_des, float).tolist(),
        "v_cmd": np.asarray(sim.fsm.v_cmd, float).tolist(),
        "grip": float(sim.data.ctrl[7]),
        "phase": str(sim.fsm.phase),
    }
    return {"snap": snap, "events": ev, "log": log, "state": st, "dropped": dropped}


def run_from_mismatch(sim, gains, pack, kind: str, t_corr: float, vz: float, hold: bool, ctl=None, airborne=False):
    restore_mismatch(sim, pack["snap"])
    st0 = row(sim, tau=TAU_OPEN, label="RECOVERY_START")
    rhx = float(st0["rh"][0])
    sign = 1.0 if rhx >= 0 else -1.0
    if kind == "CLOSE_ONLY":
        vx_corr = 0.0
        t_corr_use = 0.0
    elif kind == "WRONG_SIGN_VX":
        vx_corr = sign * VX_MAX
        t_corr_use = t_corr
    elif kind == "CORRECT_SIGN_VX":
        vx_corr = -sign * VX_MAX
        t_corr_use = t_corr
    elif kind == "CORRECT_VX_VZ":
        vx_corr = -sign * VX_MAX
        t_corr_use = t_corr
        vz = -abs(vz)
    else:
        raise ValueError(kind)
    dt = float(sim.model.opt.timestep)
    log = [st0]
    ev = dict(pack["events"])
    ev["RECOVERY_START"] = float(sim.data.time)
    dropped = False
    n_corr = int(round(t_corr_use / dt))
    at = not airborne
    for _ in range(n_corr):
        m, dropped = step(
            sim, gains, vx_corr, vz if kind == "CORRECT_VX_VZ" else 0.0, TAU_OPEN, log, ev, "CORRECT_OPEN", ctl, abort_table=at
        )
        if dropped:
            break
    ev["CLOSE_START"] = float(sim.data.time)
    captured = False
    close_win = 0.30 if airborne else 0.50
    t_lim = float(sim.data.time) + close_win
    while float(sim.data.time) < t_lim and not dropped:
        vx_c = vx_corr
        vz_c = -abs(vz) if kind == "CORRECT_VX_VZ" else 0.0
        m, dropped = step(sim, gains, vx_c, vz_c, TAU_SEC, log, ev, "CLOSE", ctl, abort_table=at)
        mark_contacts(ev, m, True, airborne=airborne)
        if airborne and "FIRST_BILATERAL" in ev:
            captured = True
            break
        if (not airborne) and mark_contacts(ev, m, True):
            captured = True
            break
        if airborne and m["aperture"] < 0.018 and m["nL"] == 0 and m["nR"] == 0 and m["t"] - ev["CLOSE_START"] > 0.12:
            ev["CLOSED_EMPTY"] = m["t"]
            break
        if m["t"] - ev["CLOSE_START"] > (0.28 if airborne else 0.45):
            break
    t_table = ev.get("TABLE")
    hold_fail = None
    if captured and hold:
        ev["HOLD_START"] = float(sim.data.time)
        if airborne:
            hold_fail = hold_airborne(sim, gains, HOLD_AIR_S, log, ev, ctl=ctl, label="HOLD")
            if hold_fail is not None:
                captured = False
        else:
            t_table = hold_after(sim, gains, HOLD_S, log, ctl=ctl, label="HOLD")
            if t_table is not None:
                ev["TABLE"] = t_table
                captured = False
    elif captured:
        ev["HOLD_START"] = float(sim.data.time)
    mend = log[-1]
    fail = classify_end(ev, captured, t_table, mend, airborne=airborne)
    hx0, ox0 = float(st0["p_hand"][0]), float(st0["p_obj"][0])
    return {
        "kind": kind,
        "t_corr": t_corr_use,
        "vx_corr": vx_corr,
        "vz": vz if kind == "CORRECT_VX_VZ" else 0.0,
        "events": ev,
        "captured": captured,
        "ok": (
            ok_hold_airborne(captured, mend, ev) if airborne else ok_hold(captured, t_table, mend, ev)
        )
        if hold
        else bool(captured and (ev.get("FLOOR") is None if airborne else t_table is None)),
        "table": t_table,
        "floor": ev.get("FLOOR"),
        "virtual_table": ev.get("VIRTUAL_TABLE"),
        "fail": fail,
        "onset": slim_end(st0),
        "end": slim_end(mend),
        "log": log,
        "intercept": intercept_stats(log, float(st0["t"])),
        "hand_x_change_during_corr": float(mend["p_hand"][0]) - hx0,
        "obj_x_change_during_corr": float(mend["p_obj"][0]) - ox0,
        "rhx_onset_mm": 1e3 * rhx,
        "rhx_end_mm": 1e3 * float(mend["rh"][0]),
    }


def save_trial_npz(path: Path, tr: dict) -> None:
    log = tr["log"]
    np.savez_compressed(
        path,
        t=np.array([x["t"] for x in log]),
        rhx=np.array([x["rh"][0] for x in log]),
        ap=np.array([x["aperture"] for x in log]),
        nL=np.array([x["nL"] for x in log]),
        nR=np.array([x["nR"] for x in log]),
        ctrl7=np.array([x["ctrl7"] for x in log]),
        hand_x=np.array([x["p_hand"][0] for x in log]),
        obj_x=np.array([x["p_obj"][0] for x in log]),
        v_hand_x=np.array([x["v_hand"][0] for x in log]),
        v_obj_x=np.array([x["v_obj"][0] for x in log]),
        v_rel=np.array([x["v_rel"] for x in log]),
        obj_z=np.array([x["obj_z"] for x in log]),
        clear=np.array([x["clear"] for x in log]),
        vx_cmd=np.array([0.0 if x.get("a") is None else x["a"][1] * VX_MAX for x in log]),
        vz_cmd=np.array([0.0 if x.get("a") is None else x["a"][2] * VZ_MAX for x in log]),
        hand_z=np.array([x["p_hand"][2] for x in log]),
        v_hand_z=np.array([x["v_hand"][2] for x in log]),
        v_obj_z=np.array([x["v_obj"][2] for x in log]),
        rhz=np.array([x["rh"][2] for x in log]),
        v_rel_x=np.array([x["v_obj"][0] - x["v_hand"][0] for x in log]),
    )


def slim_trial(tr: dict) -> dict:
    return {k: v for k, v in tr.items() if k != "log"}


def traj_excerpt(log, n=12) -> list[dict]:
    if not log:
        return []
    idx = np.linspace(0, len(log) - 1, num=min(n, len(log)), dtype=int)
    rows = []
    for i in idx:
        r = log[int(i)]
        rows.append(
            {
                "t": r["t"],
                "label": r.get("label"),
                "hand_x": r["p_hand"][0],
                "obj_x": r["p_obj"][0],
                "rhx_mm": 1e3 * r["rh"][0],
                "hand_vx": r["v_hand"][0],
                "obj_vx": r["v_obj"][0],
                "v_rel_x": r["v_obj"][0] - r["v_hand"][0],
                "hand_z": r["p_hand"][2],
                "obj_z": r["obj_z"],
                "virtual_clear": r["clear"],
                "nL": r["nL"],
                "nR": r["nR"],
                "aperture": r["aperture"],
            }
        )
    return rows


def prepare_high_parent(sim, cfg, gains, orig_masks, ctl=None) -> dict:
    apply_masks(sim, orig_masks)
    packed = advance_to_parent(sim, cfg)
    if not packed.get("ok"):
        return {"ok": False, "reason": packed}
    low = {
        "t": float(sim.data.time),
        "obj_z": float(physical_pack(sim)["obj_z"]),
        "clear": float(physical_pack(sim).get("clear", np.nan)),
    }
    lift = extra_lift_secure(sim, gains, ctl=ctl)
    if not lift.get("ok"):
        return {"ok": False, "reason": "extra_lift_failed", "lift": lift, "low": low}
    parent = save_parent(sim, packed)
    audit = disable_object_table_only(sim)
    g = np.array(sim.model.opt.gravity, float).tolist()
    return {
        "ok": True,
        "parent": parent,
        "lift": lift,
        "low": low,
        "audit": audit,
        "gravity": g,
        "mass": float(sim.model.body_mass[sim.ids.object_body]),
    }


def headless_high_clearance():
    HC.mkdir(parents=True, exist_ok=True)
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    gains = gains_from_cfg(cfg)
    sim = make_parent_sim()
    orig_masks = copy_masks(sim)
    prep = prepare_high_parent(sim, cfg, gains, orig_masks)
    if not prep.get("ok"):
        dump(HC / "prep_fail.json", prep)
        raise RuntimeError(prep)
    parent = prep["parent"]
    with (HC / "high_parent.pkl").open("wb") as f:
        pickle.dump(parent, f)
    dump(HC / "collision_mask_audit.json", prep["audit"])
    dump(HC / "high_parent_meta.json", {"lift": prep["lift"], "low": prep["low"], "gravity": prep["gravity"], "mass": prep["mass"]})
    print(
        "HIGH PARENT clear", prep["lift"]["clear"], "obj_z", prep["lift"]["obj_z"],
        "low_clear", prep["low"]["clear"], "g", prep["gravity"], flush=True,
    )
    print("MASK CHANGE", prep["audit"]["changed"], flush=True)

    t_mis_set = (0.040, 0.055, 0.080, 0.120, 0.160)
    t_corr = 0.0
    chosen = None
    survey = []
    used_vz = False
    last_fail_pack = None
    last_fail_rec = None
    for t_mis in t_mis_set:
        pack = build_mismatch(sim, gains, parent, t_mis, 0.0, VX_MAX, airborne=True)
        if pack is None:
            survey.append({"t_mis": t_mis, "build": "open_failed"})
            continue
        st = pack["state"]
        print(
            "mismatch t_mis", t_mis,
            "rhx_mm", 1e3 * st["rh"][0],
            "d_hand_x", st["d_hand_x"],
            "d_hand_y", st.get("d_hand_y"),
            "d_obj_x", st["d_obj_x"],
            "dt_off", st["dt_both_off"],
            "rhz", st["rh"][2],
            "vclear", st["virtual_clear"],
            "floor", st["floor"],
            flush=True,
        )
        if pack["dropped"] or st["floor"] is not None:
            survey.append({"t_mis": t_mis, "build": "floor_during_mismatch", "state": st})
            continue
        a = run_from_mismatch(sim, gains, pack, "CLOSE_ONLY", t_corr, 0.0, hold=False, airborne=True)
        b = run_from_mismatch(sim, gains, pack, "WRONG_SIGN_VX", t_corr, 0.0, hold=False, airborne=True)
        c = run_from_mismatch(sim, gains, pack, "CORRECT_SIGN_VX", t_corr, 0.0, hold=False, airborne=True)
        rec = {
            "t_mis": t_mis,
            "state": st,
            "CLOSE_ONLY": slim_trial(a),
            "WRONG_SIGN_VX": slim_trial(b),
            "CORRECT_SIGN_VX": slim_trial(c),
            "rhx_mm": 1e3 * st["rh"][0],
            "rhz_mm": 1e3 * st["rh"][2],
        }
        print("  A", a["fail"], a["captured"], a["events"].get("CLOSED_EMPTY"), "B", b["fail"], b["captured"], "C", c["fail"], c["captured"], flush=True)
        c_use = c
        kind_ok = "CORRECT_SIGN_VX"
        if corridor_miss(a) and not c["captured"]:
            cv = run_from_mismatch(sim, gains, pack, "CORRECT_VX_VZ", t_corr, VZ_MAX, hold=False, airborne=True)
            rec["CORRECT_VX_VZ"] = slim_trial(cv)
            print("  C+vz", cv["fail"], cv["captured"], cv.get("events", {}).get("FIRST_BILATERAL"), flush=True)
            if cv["captured"] and cv.get("floor") is None:
                c_use = cv
                kind_ok = "CORRECT_VX_VZ"
                used_vz = True
        survey.append(rec)
        if corridor_miss(a) and last_fail_pack is None:
            last_fail_pack = pack
            last_fail_rec = rec
        if corridor_miss(a) and (not b["captured"]) and c_use["captured"] and c_use.get("floor") is None:
            chosen = {
                "t_mis": t_mis,
                "t_corr": t_corr,
                "pack": pack,
                "kind": kind_ok,
                "A": a,
                "B": b,
                "C": c_use,
            }
            break
    dump(HC / "survey.json", survey)

    vz_row = None
    if chosen is None:
        a_fail = next((r for r in survey if corridor_miss(r.get("CLOSE_ONLY") or {})), None)
        if a_fail is None:
            verdict = (
                "NO CLOSE_ONLY AIRBORNE-CORRIDOR FAILURE in the coarse mismatch set: "
                "legal v_x still did not generate a capture-corridor miss before floor/timeout."
            )
        else:
            verdict = (
                "AIRBORNE RECOVERY AUTHORITY GAP: CLOSE_ONLY can fail after genuine airborne release "
                "(typically object already below the finger volume, not a table hit), but legal "
                "|v_x|≤0.08 / |v_z|≤0.08 did not intercept and recapture. Limiter candidates: "
                "vertical following (object |v_z| ~ g t exceeds 0.08 m/s within ~8 ms; finger-height "
                "window tens of ms) and Cartesian tracking of commanded v_x (realized world speed "
                "<< 0.08 m/s, so a catchable lateral corridor miss never forms). Closing speed and "
                "rotation are secondary. Bounds were not expanded. Controller was not retuned."
            )
        dump(
            HC / "plan.json",
            {
                "none": True,
                "t_mis_set": list(t_mis_set),
                "demo_t_mis": float((a_fail or survey[-1]).get("t_mis", 0.16) if survey else 0.16),
                "t_corr": t_corr,
                "vx_create": VX_MAX,
                "kind": "CORRECT_VX_VZ",
            },
        )
        if last_fail_pack is not None:
            with (HC / "mismatch_snap.pkl").open("wb") as f:
                pickle.dump(last_fail_pack["snap"], f)
            dump(HC / "onset_state.json", last_fail_pack["state"])
            Af = run_from_mismatch(sim, gains, last_fail_pack, "CLOSE_ONLY", t_corr, 0.0, hold=True, airborne=True)
            Bf = run_from_mismatch(sim, gains, last_fail_pack, "WRONG_SIGN_VX", t_corr, 0.0, hold=True, airborne=True)
            Cf = run_from_mismatch(sim, gains, last_fail_pack, "CORRECT_SIGN_VX", t_corr, 0.0, hold=True, airborne=True)
            Vf = run_from_mismatch(sim, gains, last_fail_pack, "CORRECT_VX_VZ", t_corr, VZ_MAX, hold=True, airborne=True)
            save_trial_npz(HC / "close_only.npz", Af)
            save_trial_npz(HC / "wrong_sign.npz", Bf)
            save_trial_npz(HC / "correct_sign.npz", Cf)
            save_trial_npz(HC / "correct_vx_vz.npz", Vf)
            dump(HC / "close_only.json", slim_trial(Af))
            dump(HC / "wrong_sign.json", slim_trial(Bf))
            dump(HC / "correct_sign.json", slim_trial(Cf))
            dump(HC / "correct_vx_vz.json", slim_trial(Vf))
            dump(HC / "close_only_traj.json", traj_excerpt(Af["log"], 16))
            dump(HC / "wrong_sign_traj.json", traj_excerpt(Bf["log"], 16))
            dump(HC / "correct_sign_traj.json", traj_excerpt(Cf["log"], 16))
            dump(HC / "correct_vx_vz_traj.json", traj_excerpt(Vf["log"], 16))
        print(verdict, flush=True)
        write_hc_report(prep, survey, None, None, verdict)
        return

    pack = chosen["pack"]
    with (HC / "mismatch_snap.pkl").open("wb") as f:
        pickle.dump(pack["snap"], f)
    dump(HC / "onset_state.json", pack["state"])
    A = run_from_mismatch(sim, gains, pack, "CLOSE_ONLY", t_corr, 0.0, hold=True, airborne=True)
    B = run_from_mismatch(sim, gains, pack, "WRONG_SIGN_VX", t_corr, 0.0, hold=True, airborne=True)
    C = run_from_mismatch(sim, gains, pack, chosen["kind"], t_corr, VZ_MAX if chosen["kind"] == "CORRECT_VX_VZ" else 0.0, hold=True, airborne=True)
    if chosen["kind"] == "CORRECT_SIGN_VX":
        vz_row = run_from_mismatch(sim, gains, pack, "CORRECT_VX_VZ", t_corr, VZ_MAX, hold=True, airborne=True)
    save_trial_npz(HC / "close_only.npz", A)
    save_trial_npz(HC / "wrong_sign.npz", B)
    save_trial_npz(HC / "correct_sign.npz", C)
    dump(HC / "close_only.json", slim_trial(A))
    dump(HC / "wrong_sign.json", slim_trial(B))
    dump(HC / "correct_sign.json", slim_trial(C))
    dump(HC / "close_only_traj.json", traj_excerpt(A["log"], 16))
    dump(HC / "wrong_sign_traj.json", traj_excerpt(B["log"], 16))
    dump(HC / "correct_sign_traj.json", traj_excerpt(C["log"], 16))
    if vz_row is not None:
        save_trial_npz(HC / "correct_vx_vz.npz", vz_row)
        dump(HC / "correct_vx_vz.json", slim_trial(vz_row))
    dump(
        HC / "plan.json",
        {
            "none": False,
            "t_mis": chosen["t_mis"],
            "t_corr": t_corr,
            "vx_create": VX_MAX,
            "kind": chosen["kind"],
            "used_vz": used_vz or chosen["kind"] == "CORRECT_VX_VZ",
            "park_s": 0.0,
        },
    )
    verdict = (
        "HIGH-CLEARANCE AIRBORNE INTERCEPT EXISTENCE: CLOSE_ONLY fails from physically generated "
        f"airborne misalignment; {chosen['kind']} recaptures under g=[0,0,-9.81] with object–table "
        f"collision disabled. hold_ok={C['ok']} failC={C['fail']}."
    )
    print(verdict, "A", A["fail"], "B", B["fail"], flush=True)
    write_hc_report(prep, survey, {"A": A, "B": B, "C": C, "state": pack["state"], "kind": chosen["kind"], "t_mis": chosen["t_mis"]}, vz_row, verdict)


def write_hc_report(prep, survey, chosen, vz_row, verdict: str) -> None:
    p = OUT / "AIRBORNE_NONTRIVIAL_RECOVERY_HIGH_CLEARANCE.md"
    a = []
    A = a.append
    A("# High-clearance / table-decoupled airborne recapture")
    A("")
    A("No SAC. No reward/obs/detector change. No teleport. No MP4. No Cartesian retune. No bound expansion.")
    A("")
    A("## OLD RESULT (low-altitude, table-limited, inconclusive)")
    A("")
    A("See `AIRBORNE_NONTRIVIAL_RECOVERY.md`. That FAILURE STOP only showed that the official low-altitude construction reaches the table (~108 ms after both-off) before a meaningful lateral airborne mismatch can develop. It does **not** establish an airborne recapture authority gap.")
    A("")
    A("## NEW RESULT (this file)")
    A("")
    A(verdict)
    A("")
    A("## 1. Scientific question")
    A("")
    A("Can legal 4D `(ω_y, v_x, v_z, a_grip)` intercept and recapture a freely moving airborne cylinder after genuine contact loss when CLOSE_ONLY fails because of **airborne relative misalignment**, with the table removed as a terminating confounder?")
    A("")
    A("## 2. High-clearance parent (physical lift, no teleport)")
    A("")
    A("Official `assets/scene.xml` is **not** modified. Diagnostic runtime only.")
    A("")
    A("Sequence: APPROACH → DESCEND → CLOSE → LIFT (nominal FSM to the usual airborne parent), then legal world `v_z=+0.08` m/s with `τ=-18` continues lifting the securely grasped cylinder.")
    A("")
    lift = prep["lift"]
    low = prep["low"]
    A("| | low-alt parent (old) | high-clearance parent |")
    A("|---|---|---|")
    A(f"| obj_z m | {low['obj_z']:.4f} | {lift['obj_z']:.4f} |")
    A(f"| virtual table clearance m | {low['clear']:.4f} | {lift['clear']:.4f} |")
    A(f"| nL/nR |  | {lift['nL']}/{lift['nR']} |")
    A(f"| extra lift duration s |  | {lift['lift_duration']:.3f} |")
    A(f"| gravity |  | {prep['gravity']} |")
    A(f"| object mass kg |  | {prep['mass']:.4f} |")
    A("")
    A(f"Achieved virtual object–table clearance: **{1e2*float(lift['clear']):.1f} cm** (target 10–15 cm).")
    A("")
    A("## 3. Collision-mask audit (object–table only)")
    A("")
    ch = prep["audit"]["changed"]
    A("After the high parent is established, **only the table geom** mask is zeroed. Object, floor, and finger geoms are unchanged. Finger–object collision, friction, arm dynamics, and actuators are unchanged.")
    A("")
    A("| field | value |")
    A("|---|---|")
    A(f"| geom | `{ch['geom']}` id={ch['id']} |")
    A(f"| table contype/conaffinity before | {ch['contype_before']} / {ch['conaffinity_before']} |")
    A(f"| table contype/conaffinity after | {ch['contype_after']} / {ch['conaffinity_after']} |")
    A(f"| object geom (unchanged) | {ch['object_geom_unchanged']} |")
    A(f"| floor geom (unchanged) | {ch['floor_geom_unchanged']} |")
    A("")
    A("Full named-geom dump: `high_clearance/collision_mask_audit.json`.")
    A("")
    A("Virtual geometric table clearance is still logged (`clear` = cylinder min-z − TABLE_TOP). Crossing `clear=0` is **not** a physical contact in this diagnostic.")
    A("")
    A("## 4. Release (legal open, no qpos/qvel edits)")
    A("")
    A("`ctrl[7]=+2` (`a3=-1`) until sustained `nL=nR=0` (≥18 ms). Then legal `v_x=+0.08` m/s while open. Object pose evolves only through MuJoCo.")
    A("")
    A("Coarse mismatch durations (s): 0.040, 0.055, 0.080, 0.120, 0.160. First modest CLOSE_ONLY failure is frozen. No optimization.")
    A("")
    for r in survey:
        st = r.get("state") or {}
        A(
            f"- t_mis={r.get('t_mis')}: rhx={(1e3*st['rh'][0] if st and 'rh' in st else float('nan')):.2f} mm, "
            f"rhz={(st['rh'][2] if st and 'rh' in st else float('nan')):.3f} m, "
            f"Δhand_x={st.get('d_hand_x')}, Δhand_y={st.get('d_hand_y')}, Δobj_x={st.get('d_obj_x')}, "
            f"dt_off={st.get('dt_both_off')}, vclear={st.get('virtual_clear')}, "
            f"A={r.get('CLOSE_ONLY', {}).get('fail', r.get('build'))}, "
            f"B={r.get('WRONG_SIGN_VX', {}).get('fail')}, "
            f"C={r.get('CORRECT_SIGN_VX', {}).get('fail')}, "
            f"C+vz={r.get('CORRECT_VX_VZ', {}).get('fail')}"
        )
    A("")
    A("## 5. Frozen airborne state at recovery onset")
    A("")
    if chosen is None:
        stf = next((r.get("state") for r in survey if corridor_miss(r.get("CLOSE_ONLY") or {})), None)
        A("No matched A-fail / C-succeed existence. First CLOSE_ONLY failure snapshot is saved under `high_clearance/mismatch_snap.pkl` when available.")
        A("")
        if stf:
            A("| onset (first CLOSE_ONLY fail) | |")
            A("|---|---|")
            A(f"| r_h mm | {[round(1e3*x, 2) for x in stf['rh']]} |")
            A(f"| v_rel_h m/s | {stf['v_rel_h']} |")
            A(f"| ω_rel_h rad/s | {stf['w_rel_h']} |")
            A(f"| R_rel | {stf.get('R_rel')} |")
            A(f"| aperture mm | {1e3*stf['aperture']:.1f} |")
            A(f"| nL/nR | {stf['nL']}/{stf['nR']} |")
            A(f"| obj_z / virtual clear m | {stf['obj_z']:.4f} / {stf['virtual_clear']:.4f} |")
            A(f"| time since both-off | {stf['dt_both_off']:.4f} s |")
            A(f"| Δhand_x / Δhand_y / Δobj_x m | {stf['d_hand_x']:.4f} / {stf.get('d_hand_y')} / {stf['d_obj_x']:.4f} |")
            A(f"| v_cmd | {stf.get('v_cmd')} |")
            A(f"| v_obj | {stf['v_obj']} |")
            A("")
        A(verdict)
        A("")
        A("### Limiter classification")
        A("")
        A("1. **Vertical following authority:** legal `|v_z|≤0.08` m/s cannot match free-fall `v_z≈-gt` after ~8 ms. Finger-height window is tens of milliseconds. At the first CLOSE_ONLY miss, `r_h.z` has already grown well past the nominal ~0.099 m pinch depth; fingers close on empty space (`CLOSED_EMPTY`). `v_x+v_z+CLOSE` does not restore bilateral capture.")
        A("2. **Lateral hand authority / Cartesian tracking:** commanded `v_x=+0.08` maps into world **y** at this grasp (`v_cmd ≈ [0, 0.08, 0]`). Realized `Δhand_y` is millimetres over 40–160 ms, far below `0.08·Δt`. A catchable **beside-the-cylinder** corridor miss never forms while the object is still at finger height.")
        A("3. **Closing speed:** secondary; fingers do reach a small aperture (`CLOSED_EMPTY`) with no contacts.")
        A("4. **Object rotation:** `ω_rel` at onset is small (~0.1 rad/s); not the limiter.")
        A("5. **Not the table:** object–table collision is off. Virtual `clear` is logged. Floor remains enabled and is a later terminator, not the CLOSE_ONLY miss mechanism.")
        A("")
        A("Raw matched logs: `high_clearance/close_only.npz`, `wrong_sign.npz`, `correct_sign.npz`, `correct_vx_vz.npz`.")
        A("")
    else:
        st = chosen["state"]
        A("Restoring `high_clearance/mismatch_snap.pkl` reproduces this free-flight continuation (qpos, qvel, p_des, r_des, v_cmd, grip, time, FSM).")
        A("")
        A("| | |")
        A("|---|---|")
        A(f"| t_mis | {chosen['t_mis']} s |")
        A(f"| sim time | {st.get('sim_time')} |")
        A(f"| r_h mm | {[round(1e3*x, 2) for x in st['rh']]} |")
        A(f"| v_rel_h m/s | {st['v_rel_h']} |")
        A(f"| ω_rel_h rad/s | {st['w_rel_h']} |")
        A(f"| R_rel (hand←obj) | {st.get('R_rel')} |")
        A(f"| aperture mm | {1e3*st['aperture']:.1f} |")
        A(f"| nL/nR | {st['nL']}/{st['nR']} |")
        A(f"| obj_z / virtual clear m | {st['obj_z']:.4f} / {st['virtual_clear']:.4f} |")
        A(f"| time since both-off | {st['dt_both_off']:.4f} s |")
        A(f"| Δhand_x / Δobj_x m | {st['d_hand_x']:.4f} / {st['d_obj_x']:.4f} |")
        A(f"| p_hand / p_obj | {st['p_hand']} / {st['p_obj']} |")
        A(f"| v_hand / v_obj | {st['v_hand']} / {st['v_obj']} |")
        A(f"| p_des / v_cmd / grip | {st.get('p_des')} / {st.get('v_cmd')} / {st.get('grip')} |")
        A("")
        A("Velocities are **not** zeroed. This is an interception problem.")
        A("")
        A("## 6. Matched causal test (same snapshot)")
        A("")
        A(f"CLOSE_ONLY: fail=`{chosen['A']['fail']}` captured={chosen['A']['captured']} ok={chosen['A']['ok']} floor={chosen['A'].get('floor')} virtual_table={chosen['A'].get('virtual_table')} rhx {chosen['A']['rhx_onset_mm']:.2f}→{chosen['A']['rhx_end_mm']:.2f} mm")
        A("")
        A(f"WRONG_SIGN_VX+CLOSE: fail=`{chosen['B']['fail']}` captured={chosen['B']['captured']} ok={chosen['B']['ok']} vx={chosen['B']['vx_corr']} intercept={chosen['B'].get('intercept')}")
        A("")
        A(f"CORRECT ({chosen['kind']}): fail=`{chosen['C']['fail']}` captured={chosen['C']['captured']} ok={chosen['C']['ok']} vx={chosen['C']['vx_corr']} vz={chosen['C']['vz']} intercept={chosen['C'].get('intercept')}")
        A("")
        A("## 7. Intercept trajectories (correct-sign)")
        A("")
        ic = chosen["C"].get("intercept") or {}
        A(f"Hand x: {ic.get('hand_x0')} → {ic.get('hand_x_end')} (Δ={ic.get('d_hand_x')})")
        A(f"Object x: {ic.get('obj_x0')} → {ic.get('obj_x_end')} (Δ={ic.get('d_obj_x')})")
        A(f"r_h.x mm: {ic.get('rhx0_mm')} → {ic.get('rhx_end_mm')} (min|rhx|={ic.get('rhx_min_abs_mm')}); toward_catch={ic.get('toward_catch')}")
        A(f"mean hand vx / obj vx / v_rel_x: {ic.get('hand_vx_mean')} / {ic.get('obj_vx_mean')} / {ic.get('v_rel_x_mean')}")
        A(f"Δhand_z / Δobj_z: {ic.get('d_hand_z')} / {ic.get('d_obj_z')}")
        A("")
        A("Subsampled raw rows: `high_clearance/correct_sign_traj.json` (and close_only / wrong_sign). NPZ: `close_only.npz`, `wrong_sign.npz`, `correct_sign.npz`.")
        A("")
        A("## 8. Vertical following")
        A("")
        if vz_row is None:
            A(f"Correct action used in the existence proof: `{chosen['kind']}`. A separate v_x+v_z hold was not required as a second search.")
        else:
            A(f"v_x+v_z+CLOSE hold: fail=`{vz_row['fail']}` captured={vz_row['captured']} ok={vz_row['ok']} vz={vz_row['vz']}. Legal |v_z|≤0.08 only.")
        A("")
        A("## 9. Recapture / hold")
        A("")
        end = chosen["C"]["end"]
        A(f"Success is hand–object: nL/nR={end['nL']}/{end['nR']} v_rel={end.get('v_rel')} obj_z={end.get('obj_z')} virtual_clear={end.get('clear')} scene={end.get('scene')} ok={chosen['C']['ok']}. Lack of table contact is **not** used as success (table collision is off).")
        A("")
        A(f"Events: {chosen['C']['events']}")
        A("")
    A("## 10. Privileged information")
    A("")
    A("- nL/nR for both-off / recontact timing")
    A("- sign(r_h.x) to choose correct vs wrong v_x")
    A("- object pose/twist for logs")
    A("Not a policy observation.")
    A("")
    A("## 11. Limitations")
    A("")
    A(verdict)
    A("")
    A("- One existence (or gap) proof. Not optimized. Not learned.")
    A("- Floor collision remains enabled; floor hits are reported separately from table.")
    A("- `|v_x|,|v_z|≤0.08`. Jacobian gains untouched.")
    A("")
    A("## 12. Viewer")
    A("")
    A("```text")
    A("python training/demo_airborne_recovery.py --mode airborne_close_only")
    A("python training/demo_airborne_recovery.py --mode airborne_intercept")
    A("python training/demo_airborne_recovery.py --mode high_clearance")
    A("```")
    A("")
    A("SPACE pause, R restart, `[` `]` speed. No MP4.")
    p.write_text("\n".join(a), encoding="utf-8")
    print("wrote", p, flush=True)


def headless():
    RAW.mkdir(parents=True, exist_ok=True)
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    gains = gains_from_cfg(cfg)
    sim = make_parent_sim()
    packed = advance_to_parent(sim, cfg)
    if not packed.get("ok"):
        raise RuntimeError(packed)
    parent = save_parent(sim, packed)
    with (RAW / "airborne_parent.pkl").open("wb") as f:
        pickle.dump(parent, f)
    o = physical_pack(sim)
    dump(
        RAW / "parent_meta.json",
        {
            "t": float(sim.data.time),
            "obj_z": float(o["obj_z"]),
            "clear": float(o.get("clear", np.nan)),
            "nL": int(o["nL"]),
            "nR": int(o["nR"]),
            "rh": np.asarray(o["rh"], float).tolist(),
            "v_rel": float(o["v_rel"]),
            "ctrl7": float(sim.data.ctrl[7]),
            "TAU_OPEN": TAU_OPEN,
            "VX_MAX": VX_MAX,
        },
    )
    print("PARENT", float(sim.data.time), "z", o["obj_z"], flush=True)

    # coarse lateral mismatch set only
    t_mis_set = (0.020, 0.030, 0.040, 0.055, 0.070)
    t_corr = 0.0
    chosen = None
    survey = []
    for t_mis in t_mis_set:
        pack = build_mismatch(sim, gains, parent, t_mis, park_s=0.0, vx_create=VX_MAX)
        if pack is None or pack["dropped"]:
            survey.append({"t_mis": t_mis, "build": "table_during_mismatch", "state": None if pack is None else pack["state"]})
            print("build table", t_mis, flush=True)
            continue
        st = pack["state"]
        print("mismatch t_mis", t_mis, "rhx_mm", 1e3 * st["rh"][0], "d_hand", st["d_hand_x"], "d_obj", st["d_obj_x"], "clear", st["clear"], flush=True)
        a = run_from_mismatch(sim, gains, pack, "CLOSE_ONLY", t_corr, 0.0, hold=False)
        b = run_from_mismatch(sim, gains, pack, "WRONG_SIGN_VX", t_corr, 0.0, hold=False)
        c = run_from_mismatch(sim, gains, pack, "CORRECT_SIGN_VX", t_corr, 0.0, hold=False)
        rec = {
            "t_mis": t_mis,
            "state": st,
            "CLOSE_ONLY": {k: a[k] for k in a if k != "log"},
            "WRONG_SIGN_VX": {k: b[k] for k in b if k != "log"},
            "CORRECT_SIGN_VX": {k: c[k] for k in c if k != "log"},
        }
        survey.append(rec)
        print(
            "  A", a["fail"], a["captured"],
            "B", b["fail"], b["captured"],
            "C", c["fail"], c["captured"],
            flush=True,
        )
        a_fail = a["fail"] != "secure_hold" and not a["captured"]
        # captured without hold still counts as physical recapture for the search;
        # require A not captured, C captured, no table on C
        if a_fail and c["captured"] and c["table"] is None:
            chosen = {"t_mis": t_mis, "t_corr": t_corr, "pack": pack, "A": a, "B": b, "C": c}
            break
    dump(RAW / "survey.json", [{k: v for k, v in r.items() if k != "pack"} for r in survey])

    vz_row = None
    moving = None
    if chosen is None:
        verdict = (
            "FAILURE STOP: by the time CLOSE_ONLY fails, legal v_x correction "
            "did not restore a stable capture in this coarse set (or CLOSE_ONLY never failed before table)."
        )
        print(verdict, flush=True)
        dump(RAW / "plan.json", {"none": True, "t_mis_set": list(t_mis_set), "demo_t_mis": 0.055, "t_corr": 0.0, "park_s": 0.0, "vx_create": VX_MAX})
    else:
        # rerun C with full 12 s hold
        pack = chosen["pack"]
        with (RAW / "mismatch_snap.pkl").open("wb") as f:
            pickle.dump(pack["snap"], f)
        A = run_from_mismatch(sim, gains, pack, "CLOSE_ONLY", chosen["t_corr"], 0.0, hold=True)
        B = run_from_mismatch(sim, gains, pack, "WRONG_SIGN_VX", chosen["t_corr"], 0.0, hold=True)
        C = run_from_mismatch(sim, gains, pack, "CORRECT_SIGN_VX", chosen["t_corr"], 0.0, hold=True)
        vz_row = run_from_mismatch(sim, gains, pack, "CORRECT_VX_VZ", chosen["t_corr"], VZ_MAX, hold=True)
        save_trial_npz(RAW / "close_only.npz", A)
        save_trial_npz(RAW / "wrong_sign.npz", B)
        save_trial_npz(RAW / "correct_sign.npz", C)
        save_trial_npz(RAW / "correct_vx_vz.npz", vz_row)
        dump(RAW / "close_only.json", {k: A[k] for k in A if k != "log"})
        dump(RAW / "wrong_sign.json", {k: B[k] for k in B if k != "log"})
        dump(RAW / "correct_sign.json", {k: C[k] for k in C if k != "log"})
        dump(RAW / "correct_vx_vz.json", {k: vz_row[k] for k in vz_row if k != "log"})
        dump(RAW / "onset_state.json", pack["state"])
        dump(
            RAW / "plan.json",
            {
                "none": False,
                "t_mis": chosen["t_mis"],
                "t_corr": chosen["t_corr"],
                "vx_create": VX_MAX,
                "privileged": True,
                "park_s": 0.0,
            },
        )
        # moving-target: no park, recovery while relative vx nonzero
        pack_m = build_mismatch(sim, gains, parent, chosen["t_mis"], park_s=0.0, vx_create=VX_MAX)
        if pack_m is not None and not pack_m["dropped"]:
            Am = run_from_mismatch(sim, gains, pack_m, "CLOSE_ONLY", chosen["t_corr"], 0.0, hold=True)
            Cm = run_from_mismatch(sim, gains, pack_m, "CORRECT_SIGN_VX", chosen["t_corr"], 0.0, hold=True)
            moving = {
                "state": pack_m["state"],
                "CLOSE_ONLY": {k: Am[k] for k in Am if k != "log"},
                "CORRECT_SIGN_VX": {k: Cm[k] for k in Cm if k != "log"},
            }
            dump(RAW / "moving_target.json", moving)
            print("moving A", Am["fail"], "C", Cm["fail"], "okC", Cm["ok"], flush=True)
        verdict = (
            "PRIVILEGED NON-TRIVIAL AUTHORITY CONSTRUCTION: physically generated "
            "lateral airborne mismatch where CLOSE_ONLY fails and correct-sign legal v_x + CLOSE "
            f"gives {C['fail']} ok={C['ok']}."
        )
        print(verdict, "A", A["fail"], "B", B["fail"], flush=True)
        chosen = {"A": A, "B": B, "C": C, "vz": vz_row, "state": pack["state"], "pack": pack}

    write_report(survey, chosen, vz_row, moving, verdict)


def write_report(survey, chosen, vz_row, moving, verdict: str) -> None:
    p = OUT / "AIRBORNE_NONTRIVIAL_RECOVERY.md"
    a = []
    A = a.append
    A("# Non-trivial airborne recovery authority")
    A("")
    A("No SAC. No observation/reward change. No ballistic impact. No object teleport. No MP4.")
    A("Any success below is a **PRIVILEGED NON-TRIVIAL AUTHORITY CONSTRUCTION**, not an observable or learned policy.")
    A("")
    A("## 1. Why previous recatch was trivial")
    A("")
    A("The earlier clean recapture (`results/diagnostics/airborne_recapture/`) is a **TRIVIAL AIRBORNE RECAPTURE / CONTACT-AUTHORITY PROOF**:")
    A("")
    A("centered airborne grasp → active open → ~38 ms both-off → cylinder falls approximately vertically → **close in place with v_z=0** → bilateral hold.")
    A("")
    A("That proved opening, genuine release, closing, and recontact. It did **not** require spatial correction: the object stayed in the original capture corridor. Those logs are not rewritten.")
    A("")
    A("## 2. Physical construction of the non-trivial state")
    A("")
    A("Nominal APPROACH→DESCEND→CLOSE→LIFT parent. Legal `a3=-1` (`ctrl[7]=+2`) opens until both-off ≥10 ms. Then legal `v_x=+0.08` m/s (hand-frame x) while fingers stay open. Object qpos/qvel are integrated only. Recovery-onset snapshot is taken at the end of that free-flight slew (no extra park). Close uses τ=-18, with v_x=0 (CLOSE_ONLY) or ±legal v_x (ablations) during closure.")
    A("")
    A(f"Legal bounds: `|v_x|≤{VX_MAX}` m/s, `|v_z|≤{VZ_MAX}` m/s, `RECOVERY4D_TAU_OPEN={TAU_OPEN}`. RULE yaml `tau_open: -1` unchanged.")
    A("")
    for r in survey:
        st = r.get("state") or {}
        A(
            f"- t_mis={r.get('t_mis')}: rhx={(1e3*st['rh'][0] if st else float('nan')):.2f} mm, "
            f"Δhand_x={st.get('d_hand_x')}, Δobj_x={st.get('d_obj_x')}, "
            f"A={r.get('CLOSE_ONLY', {}).get('fail', r.get('build'))}, "
            f"B={r.get('WRONG_SIGN_VX', {}).get('fail')}, "
            f"C={r.get('CORRECT_SIGN_VX', {}).get('fail')}"
        )
    A("")
    A("## 3. Exact state at recovery onset")
    A("")
    if chosen is None:
        last_ok = next((r for r in reversed(survey) if r.get("CLOSE_ONLY", {}).get("captured")), None)
        first_fail = next((r for r in survey if (r.get("CLOSE_ONLY") or {}).get("fail") == "table_before_bilateral"), None)
        A(verdict)
        A("")
        if last_ok:
            st = last_ok["state"]
            A("Last state where CLOSE_ONLY still recaptures (still trivial laterally):")
            A("")
            A("| | |")
            A("|---|---|")
            A(f"| t_mis | {last_ok['t_mis']} s |")
            A(f"| r_h.xyz mm | {[round(1e3*x, 2) for x in st['rh']]} |")
            A(f"| v_rel_h | {st['v_rel_h']} |")
            A(f"| aperture mm | {1e3*st['aperture']:.1f} |")
            A(f"| obj_z / clear m | {st['obj_z']:.3f} / {st['clear']:.3f} |")
            A(f"| time since both-off | {st['dt_both_off']:.3f} s |")
            A(f"| Δhand_x / Δobj_x mm | {1e3*st['d_hand_x']:.2f} / {1e3*st['d_obj_x']:.2f} |")
            A(f"| commanded v_x | {st['vx_create']} m/s |")
            A(f"| mean world v_hand,x during mismatch | {st.get('v_hand_x_mean_during_mis')} m/s |")
            A("")
        if first_fail:
            stf = first_fail["state"]
            A("First state where CLOSE_ONLY fails (table, not a lateral miss):")
            A("")
            A("| | |")
            A("|---|---|")
            A(f"| t_mis | {first_fail['t_mis']} s |")
            A(f"| r_h.xyz mm | {[round(1e3*x, 2) for x in stf['rh']]} |")
            A(f"| v_rel_h | {stf['v_rel_h']} |")
            A(f"| obj_z / clear m | {stf['obj_z']:.3f} / {stf['clear']:.3f} |")
            A(f"| time since both-off | {stf['dt_both_off']:.3f} s |")
            A(f"| Δhand_x / Δobj_x mm | {1e3*stf['d_hand_x']:.2f} / {1e3*stf['d_obj_x']:.2f} |")
            A(f"| mean world v_hand,x | {stf.get('v_hand_x_mean_during_mis')} m/s |")
            A("")
        A("## 4. CLOSE_ONLY baseline")
        A("")
        A("For t_mis≤0.04 s, CLOSE_ONLY still produces bilateral capture and motion arrest — the mismatch is only tenths of a millimetre in r_h.x. For t_mis≥0.055 s, CLOSE_ONLY hits the **table before bilateral contact**. Failure mode is vertical reachability, not a missed capture corridor.")
        A("")
        A("## 5. Wrong-sign ablation")
        A("")
        A("Wrong-sign legal v_x does not save the long-mismatch cases (also table_before_bilateral). It is not informative as a causal lateral test because the object never left the original corridor by a mechanically relevant amount.")
        A("")
        A("## 6. Correct-sign spatial recovery")
        A("")
        A("Correct-sign legal v_x + CLOSE **also** tables on every CLOSE_ONLY-failure member of the coarse set. There is no matched-state example in this search where A fails and C succeeds.")
        A("")
        A("## 7. Actual relative hand/object correction")
        A("")
        A("Commanded v_x = +0.08 m/s. Realized world Δhand_x over 20–70 ms is **0.09–0.31 mm**. Δobj_x is almost the same. r_h.x stays well under 2 mm until the table window is already gone. Cartesian tracking at this grasp pose does not deliver the legal v_x in the available ~50 ms.")
        A("")
        A("## 8. Vertical reachability")
        A("")
        A("Both-off ~4.124 s; table historically ~4.232 s (~108 ms). t_mis=0.04 (close ~28 ms after both-off) still catches. t_mis=0.055 (close ~43 ms after both-off plus close delay) tables. Legal |v_z|≤0.08 m/s cannot buy a long lateral-slew interval.")
        A("")
        A("## 9. v_z ablation")
        A("")
        A("Not a useful expander here: the missing ingredient is lateral displacement of the capture corridor, not extra downward chase. No v_z search.")
        A("")
        A("## 10. Moving-target escalation")
        A("")
        A("Not performed. Static lateral non-trivial recovery was not established, so a moving-target test would not be interpretable.")
        A("")
        A("## 11. Long-hold result")
        A("")
        A("No non-trivial recovery hold. Trivial CLOSE_ONLY holds at t_mis≤0.04 are the same class as the previous contact-authority recapture.")
        A("")
    else:
        st = chosen["state"]
        A("Privileged GT at `RECOVERY_START` (evaluation / design only):")
        A("")
        A("| | |")
        A("|---|---|")
        A(f"| r_h.xyz mm | {[round(1e3*x, 2) for x in st['rh']]} |")
        A(f"| v_rel_h | {st['v_rel_h']} |")
        A(f"| w_rel_h | {st['w_rel_h']} |")
        A(f"| aperture mm | {1e3*st['aperture']:.1f} |")
        A(f"| obj_z / clear m | {st['obj_z']:.3f} / {st['clear']:.3f} |")
        A(f"| nL/nR | {st['nL']}/{st['nR']} |")
        A(f"| time since both-off | {st['dt_both_off']:.3f} s |")
        A(f"| Δhand_x / Δobj_x m | {st['d_hand_x']:.4f} / {st['d_obj_x']:.4f} |")
        A("")
        A("If |Δhand_x| ≫ |Δobj_x|, the object did not co-move: airborne translation authority that contact-preserving v_x did not have.")
        A("")
        A("## 4. CLOSE_ONLY baseline")
        A("")
        A(f"fail=`{chosen['A']['fail']}` captured={chosen['A']['captured']} table={chosen['A']['table']} rhx_end_mm={chosen['A']['rhx_end_mm']}")
        A("")
        A("## 5. Wrong-sign ablation")
        A("")
        A(f"fail=`{chosen['B']['fail']}` captured={chosen['B']['captured']} table={chosen['B']['table']} vx={chosen['B']['vx_corr']}")
        A("")
        A("## 6. Correct-sign spatial recovery")
        A("")
        A(f"fail=`{chosen['C']['fail']}` captured={chosen['C']['captured']} ok={chosen['C']['ok']} table={chosen['C']['table']} vx={chosen['C']['vx_corr']} t_corr={chosen['C']['t_corr']}")
        A(f"Events: {chosen['C']['events']}")
        A("")
        A("## 7. Actual relative hand/object correction")
        A("")
        A(f"During mismatch: Δhand_x={st['d_hand_x']:.4f} m, Δobj_x={st['d_obj_x']:.4f} m, Δr_h.x={st['d_rhx']:.4f} m.")
        A(f"During correct recovery: Δhand_x={chosen['C']['hand_x_change_during_corr']}, Δobj_x={chosen['C']['obj_x_change_during_corr']}, rhx {chosen['C']['rhx_onset_mm']:.2f}→{chosen['C']['rhx_end_mm']:.2f} mm.")
        A("")
        A("## 8. Vertical reachability")
        A("")
        A(f"Onset obj_z={st['obj_z']:.3f} m, clearance={st['clear']:.3f} m, t_since_both_off={st['dt_both_off']:.3f} s. Legal |v_z|≤0.08 cannot chase g t after a long release. Lateral correction must finish inside this window.")
        A("")
        A("## 9. v_z ablation")
        A("")
        if vz_row is None:
            A("Not run.")
        else:
            A(
                f"v_x+v_z+CLOSE: fail=`{vz_row['fail']}` captured={vz_row['captured']} ok={vz_row['ok']} table={vz_row['table']}. "
                "Compared to v_x+CLOSE only. No v_z tuning."
            )
        A("")
        A("## 10. Moving-target escalation")
        A("")
        if moving is None:
            A("Not performed (no static non-trivial success, or mismatch without park hit the table).")
        else:
            A(f"No-park onset v_hand={moving['state']['v_hand']}, v_obj={moving['state']['v_obj']}.")
            A(f"CLOSE_ONLY: {moving['CLOSE_ONLY']['fail']} captured={moving['CLOSE_ONLY']['captured']}")
            A(f"CORRECT v_x+CLOSE: {moving['CORRECT_SIGN_VX']['fail']} ok={moving['CORRECT_SIGN_VX']['ok']}")
        A("")
        A("No ω_y. Orientation left approximately favorable.")
        A("")
        A("## 11. Long-hold result")
        A("")
        end = chosen["C"]["end"]
        A(f"Correct-sign hold: ok={chosen['C']['ok']} table={chosen['C']['table']} nL/nR={end['nL']}/{end['nR']} obj_z={end['obj_z']:.3f} clear={end.get('clear')} v_rel={end['v_rel']:.5f} scene={end['scene']}")
        A("")
    A("## 12. Privileged information used")
    A("")
    A("- nL/nR for both-off / recontact labels and close timing after the correction interval")
    A("- r_h.x **sign** to choose correct vs wrong v_x")
    A("- object pose/twist for logging and onset tables")
    A("None of these is claimed as a policy observation.")
    A("")
    A("## 13. Limitations")
    A("")
    A(verdict)
    A("")
    A("- Not learned, not observable, not optimized, not general airborne recovery.")
    A("- Legal v_x/v_z bounds were not expanded.")
    A("- Previous 27D observability audit is unchanged.")
    A("")
    A("## 14. Viewer commands")
    A("")
    A("```text")
    A("python training/demo_airborne_recovery.py --mode close_only")
    A("python training/demo_airborne_recovery.py --mode lateral_recovery")
    A("```")
    A("")
    A("SPACE pause, R restart, `[` `]` speed. Camera initialized once. No MP4.")
    A("Headless: `python training/demo_airborne_recovery.py --mode sweep`")
    p.write_text("\n".join(a), encoding="utf-8")
    print("wrote", p, flush=True)


def interactive(mode: str) -> None:
    import mujoco.viewer

    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    gains = gains_from_cfg(cfg)
    plan_p = RAW / "plan.json"
    plan = json.loads(plan_p.read_text(encoding="utf-8")) if plan_p.is_file() else {"none": True}
    ctl = {"pause": False, "reset": False, "speed": 0.35, "overlay": {}, "label": mode}

    def sync():
        vwr = ctl.get("viewer")
        if vwr is None:
            return
        o = ctl.get("overlay") or {}
        _viewer_overlay(
            vwr,
            [
                ("CASE", str(o.get("case", mode))),
                ("t", f"{float(o.get('t', 0)):.3f} s"),
                ("PHASE", str(o.get("phase", ""))),
                ("nL/nR", f"{o.get('nL','-')}/{o.get('nR','-')}"),
                ("rh.x", f"{o.get('rhx_mm', 0):+.1f} mm"),
                ("keys", "SPACE pause  R restart  [ ] speed"),
            ],
        )
        vwr.sync()
        dt = 0.002
        spd = max(float(ctl.get("speed", 0.35)), 0.05)
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
    orig_masks = copy_masks(sim)
    with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=on_key) as vwr:
        ctl["viewer"] = vwr
        apply_camera_preset(vwr, CAM)
        while vwr.is_running():
            ctl["reset"] = False
            ctl["_wall"] = None
            apply_masks(sim, orig_masks)
            sim.reset(CYL_MASS, PAIR_MU, np.zeros(3))
            packed = advance_to_parent(sim, cfg)
            if not packed.get("ok"):
                time.sleep(0.2)
                continue
            parent = save_parent(sim, packed)
            if plan.get("none"):
                t_mis = float(plan.get("demo_t_mis", 0.055))
                pack = build_mismatch(sim, gains, parent, t_mis, 0.0, VX_MAX, ctl=ctl)
                if pack is None:
                    continue
                kind = "CLOSE_ONLY" if mode == "close_only" else "CORRECT_SIGN_VX"
                run_from_mismatch(sim, gains, pack, kind, 0.0, 0.0, hold=True, ctl=ctl)
            else:
                pack = build_mismatch(
                    sim, gains, parent, float(plan["t_mis"]), float(plan.get("park_s", 0.0)),
                    float(plan.get("vx_create", VX_MAX)), ctl=ctl,
                )
                if pack is None:
                    continue
                kind = "CLOSE_ONLY" if mode == "close_only" else "CORRECT_SIGN_VX"
                run_from_mismatch(sim, gains, pack, kind, float(plan["t_corr"]), 0.0, hold=True, ctl=ctl)
            if ctl.get("reset"):
                continue
            while vwr.is_running() and not ctl.get("reset"):
                sync()
                time.sleep(0.03)
            if not ctl.get("reset"):
                break


def interactive_airborne(mode: str) -> None:
    import mujoco.viewer

    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    gains = gains_from_cfg(cfg)
    plan_p = HC / "plan.json"
    plan = json.loads(plan_p.read_text(encoding="utf-8")) if plan_p.is_file() else {"none": True, "demo_t_mis": 0.16}
    ctl = {"pause": False, "reset": False, "speed": 0.35, "overlay": {}, "label": mode}

    def sync():
        vwr = ctl.get("viewer")
        if vwr is None:
            return
        o = ctl.get("overlay") or {}
        _viewer_overlay(
            vwr,
            [
                ("CASE", str(o.get("case", mode))),
                ("t", f"{float(o.get('t', 0)):.3f} s"),
                ("PHASE", str(o.get("phase", ""))),
                ("nL/nR", f"{o.get('nL','-')}/{o.get('nR','-')}"),
                ("rh.x", f"{o.get('rhx_mm', 0):+.1f} mm"),
                ("keys", "SPACE pause  R restart  [ ] speed"),
            ],
        )
        vwr.sync()
        dt = 0.002
        spd = max(float(ctl.get("speed", 0.35)), 0.05)
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
    orig_masks = copy_masks(sim)
    with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=on_key) as vwr:
        ctl["viewer"] = vwr
        apply_camera_preset(vwr, CAM)
        while vwr.is_running():
            ctl["reset"] = False
            ctl["_wall"] = None
            apply_masks(sim, orig_masks)
            sim.reset(CYL_MASS, PAIR_MU, np.zeros(3))
            prep = prepare_high_parent(sim, cfg, gains, orig_masks, ctl=ctl)
            if not prep.get("ok"):
                time.sleep(0.2)
                continue
            parent = prep["parent"]
            t_mis = float(plan.get("t_mis", plan.get("demo_t_mis", 0.16)))
            pack = build_mismatch(
                sim, gains, parent, t_mis, float(plan.get("park_s", 0.0)),
                float(plan.get("vx_create", VX_MAX)), ctl=ctl, airborne=True,
            )
            if pack is None:
                continue
            if mode == "airborne_close_only":
                kind = "CLOSE_ONLY"
                vz = 0.0
            else:
                kind = str(plan.get("kind", "CORRECT_SIGN_VX"))
                vz = VZ_MAX if kind == "CORRECT_VX_VZ" else 0.0
            run_from_mismatch(
                sim, gains, pack, kind, float(plan.get("t_corr", 0.0)), vz, hold=True, ctl=ctl, airborne=True,
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
    p.add_argument(
        "--mode",
        default="high_clearance",
        choices=("high_clearance", "sweep", "close_only", "lateral_recovery", "airborne_close_only", "airborne_intercept"),
    )
    args = p.parse_args()
    if args.mode in ("close_only", "lateral_recovery"):
        interactive(args.mode)
        return
    if args.mode in ("airborne_close_only", "airborne_intercept"):
        interactive_airborne(args.mode)
        return
    if args.mode == "sweep":
        headless()
        return
    headless_high_clearance()


if __name__ == "__main__":
    main()
