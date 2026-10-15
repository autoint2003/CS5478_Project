"""Fix over-chase depth and close geometry on one 6 m/s top-impact IC.

Uses official step_legal. Does not change the robot, nominal controller,
TABLE_DROP, or the K48 codebook.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.recovery_actions import TAU_OPEN, TAU_SECURE
from training.ball_impact_recoverability import _action_from, _nominal_fail, _setup
from envs.airborne_obs import observe_airborne
from envs.observable_obs import ObservableObsState
from training.capture_geometry import capture_geometry
from training.horizon_outcome import NOMINAL_S
from training.privileged_ball_impact_search import representability, step_legal
from training.table_drop import latched, restore_state
from training.top_impact_open_overchase_recatch import (
    _cells,
    _finish,
    _impact,
    _pose,
    _video,
)
from training.unified_k47 import load_codebook
from training.legal_recatch_transfer import _bottom
from training.replay_core import TABLE_TOP
from training.recovery_runtime import restore_ballistic

OUT = ROOT / "results" / "diagnostics" / "top_impact_recatch_geometry_fix"
IC_ID = "s6.0_p10_a+0.000_e0.000"
MAIN_HALF = np.array([0.0085, 0.004, 0.0085])
MAIN_CENTER = np.array([0.0, 0.0055, 0.0445])


def _geom_name(sim, geom_id):
    import mujoco

    name = mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
    return name or f"geom{int(geom_id)}"


def _geom_kind(sim, geom_id):
    """Which finger collision piece this geom is.

    The large pad box is recessed. The stable grasp touches the inner face
    of the smaller fingertip boxes. Finger +Y points outward, so that face
    is the low-Y side.
    """
    import mujoco

    kind = int(sim.model.geom_type[geom_id])
    if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
        return "MESH"
    size = np.array(sim.model.geom_size[geom_id], float)
    if np.allclose(size, MAIN_HALF, atol=1e-4):
        return "MAIN_PAD"
    pos = np.array(sim.model.geom_pos[geom_id], float)
    if float(pos[2]) > 0.046:
        return "CORNER_TIP"
    return "CORNER_PALM"


def _class_of(sim, geom_id, local):
    kind = _geom_kind(sim, geom_id)
    z = float(local[2])
    y = float(local[1])
    if kind == "MESH":
        if z < 0.036:
            return "UPPER_SLOPED_PART"
        return "OTHER"
    if kind in ("MAIN_PAD", "CORNER_PALM", "CORNER_TIP") and 0.034 <= z <= 0.055 and y < 0.005:
        return "LOWER_VERTICAL_PAD"
    return "EDGE_CONTACT"


def contacts_now(sim):
    """Every finger–cylinder contact, in the finger body frame."""
    import mujoco

    obj = int(sim.ids.object_geom)
    left = int(sim.ids.left_body)
    right = int(sim.ids.right_body)
    rows = []
    for index in range(int(sim.data.ncon)):
        contact = sim.data.contact[index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if obj not in (geom1, geom2):
            continue
        other_geom = geom2 if geom1 == obj else geom1
        body = int(sim.model.geom_bodyid[other_geom])
        if body not in (left, right):
            continue
        wrench = np.zeros(6)
        mujoco.mj_contactForce(sim.model, sim.data, index, wrench)
        normal = np.array(contact.frame[0:3], float)
        tan1 = np.array(contact.frame[3:6], float)
        tan2 = np.array(contact.frame[6:9], float)
        on_first = -float(wrench[0]) * normal + float(wrench[1]) * tan1 + float(wrench[2]) * tan2
        on_cylinder = -(on_first if geom1 == obj else -on_first)
        normal_on = -((-float(wrench[0]) * normal) if geom1 == obj else (float(wrench[0]) * normal))
        friction_on = on_cylinder - normal_on
        point = np.array(contact.pos, float)
        origin = np.array(sim.data.xpos[body], float)
        rotation = np.array(sim.data.xmat[body].reshape(3, 3), float)
        local = rotation.T @ (point - origin)
        center = np.array(sim.data.xpos[sim.ids.object_body], float)
        rows.append({
            "side": "L" if body == left else "R",
            "geom": _geom_name(sim, other_geom),
            "kind": _geom_kind(sim, other_geom),
            "class": _class_of(sim, other_geom, local),
            "local": local,
            "offset": point - center,
            "normal": normal_on,
            "fn": abs(float(wrench[0])),
            "ft": float(np.linalg.norm(wrench[1:3])),
            "fz_n": float(normal_on[2]),
            "fz_f": float(friction_on[2]),
            "dist": float(contact.dist),
            "dim": int(contact.dim),
            "efc": int(contact.efc_address),
        })
    return rows


def _pad_band(sim):
    """Hand-frame z span of the main vertical pad, not the corner boxes."""
    import mujoco

    hand = np.array(sim.data.xpos[sim.ids.hand_body], float)
    finger_axis = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)[:, 2]
    lows, highs = [], []
    for body in (int(sim.ids.left_body), int(sim.ids.right_body)):
        for geom_id in range(sim.model.ngeom):
            if int(sim.model.geom_bodyid[geom_id]) != body:
                continue
            if int(sim.model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_BOX):
                continue
            size = np.array(sim.model.geom_size[geom_id], float)
            if not np.allclose(size, MAIN_HALF, atol=1e-4):
                continue
            origin = np.array(sim.data.xpos[body], float)
            rotation = np.array(sim.data.xmat[body].reshape(3, 3), float)
            center = origin + rotation @ MAIN_CENTER
            half_z = abs(float(rotation[:, 2] @ finger_axis)) * MAIN_HALF[2]
            cz = float(np.dot(center - hand, finger_axis))
            lows.append(cz - half_z)
            highs.append(cz + half_z)
    return float(np.min(lows)), float(np.max(highs))


def _class_force(rows):
    out = {name: 0.0 for name in ("LOWER_VERTICAL_PAD", "UPPER_SLOPED_PART", "EDGE_CONTACT", "OTHER")}
    sides = {name: set() for name in out}
    for row in rows:
        out[row["class"]] += row["fn"]
        if row["fn"] > 0.2:
            sides[row["class"]].add(row["side"])
    return out, {name: "".join(sorted(value)) for name, value in sides.items()}


def _wrist_sat(sim):
    force = np.array(sim.data.actuator_force, float)
    limits = np.array(sim.model.actuator_forcerange, float)
    peak = 0.0
    saturated = False
    for index in range(min(7, force.size)):
        limit = max(abs(float(limits[index, 0])), abs(float(limits[index, 1])))
        peak = max(peak, abs(float(force[index])))
        if limit > 0.0 and abs(float(force[index])) >= 0.95 * limit:
            saturated = True
    return peak, saturated


def _metrics(rows):
    def first(pred):
        for row in rows:
            if pred(row):
                return row
        return None

    opened = first(lambda row: row["gap"] > rows[0]["gap"] + 0.003)
    off = first(lambda row: row["nL"] == 0 and row["nR"] == 0)
    seen = False
    recontact = None
    pad_hit = None
    for row in rows:
        if row["nL"] == 0 and row["nR"] == 0:
            seen = True
        elif seen and recontact is None and row["nL"] > 0 and row["nR"] > 0:
            recontact = row
        if seen and pad_hit is None and row["pad_sides"] == "LR":
            pad_hit = row
    pad_run = 0
    sustained = None
    for row in rows:
        pad_run = pad_run + 1 if row["pad_sides"] == "LR" and row["fn_pad"] >= row["fn_slope"] else 0
        if pad_run >= 3 and sustained is None:
            sustained = row
    vz0 = first(lambda row: row["v_obj"][2] > -0.05 and row["t"] > 0.05)
    return {
        "t_open": None if opened is None else round(opened["t"], 3),
        "t_off": None if off is None else round(off["t"], 3),
        "t_re": None if recontact is None else round(recontact["t"], 3),
        "t_pad": None if pad_hit is None else round(pad_hit["t"], 3),
        "t_pad_hold": None if sustained is None else round(sustained["t"], 3),
        "t_vz0": None if vz0 is None else round(vz0["t"], 3),
        "re_class": "" if recontact is None else recontact["dominant"],
        "re_cz": None if recontact is None else round(recontact["cz"], 4),
        "re_gap_mm": None if recontact is None else round(1000.0 * recontact["gap"], 1),
        "re_fn_pad": None if recontact is None else round(recontact["fn_pad"], 2),
        "re_fn_slope": None if recontact is None else round(recontact["fn_slope"], 2),
        "min_cz": round(min(row["cz"] for row in rows), 4),
        "max_hand_drop": round(max(rows[0]["p_hand"][2] - row["p_hand"][2] for row in rows), 4),
        "max_down_speed": round(min(row["v_hand"][2] for row in rows), 3),
        "gap_max_mm": round(1000.0 * max(row["gap"] for row in rows), 1),
        "gap_min_mm": round(1000.0 * min(row["gap"] for row in rows), 1),
        "min_bottom": round(min(row["bottom"] for row in rows), 4),
        "min_fz_n": round(min(row["fz_n"] for row in rows), 2),
        "peak_fn": round(max(row["fn"] for row in rows), 2),
        "peak_class": max(rows, key=lambda row: row["fn"])["dominant"],
        "table": any(row["table"] for row in rows),
        "end_mode": rows[-1]["mode"],
        "end_bottom": round(rows[-1]["bottom"], 4),
        "end_cz": round(rows[-1]["cz"], 4),
        "end_gap_mm": round(1000.0 * rows[-1]["gap"], 1),
        "end_vz": round(rows[-1]["v_obj"][2], 3),
        "lifted": rows[-1]["bottom"] > min(row["bottom"] for row in rows) + 0.03 and rows[-1]["v_obj"][2] > -0.02,
        "sat": any(row["sat"] for row in rows),
    }


def rollout(sim, snap, choose, horizon, after_restore=None):
    restore_ballistic(sim, snap)
    if after_restore is not None:
        after_restore()
    restore_state(sim, None)
    t0 = float(sim.data.time)
    rows = []
    actions = []
    contacts = []
    observations = []
    hist = ObservableObsState(0.02)
    handoff_obs = None
    state = {"mode": "open"}
    while float(sim.data.time) - t0 < horizon - 1e-9:
        pose = _pose(sim)
        raw = contacts_now(sim)
        forces, sides = _class_force(raw)
        band = _pad_band(sim)
        pose["t"] = float(sim.data.time) - t0
        pose["cz"] = float(capture_geometry(sim)["center_hand_m"][2])
        pose["pad_lo"], pose["pad_hi"] = band
        pose["fn_pad"] = forces["LOWER_VERTICAL_PAD"]
        pose["fn_slope"] = forces["UPPER_SLOPED_PART"]
        pose["fn_edge"] = forces["EDGE_CONTACT"]
        pose["pad_sides"] = sides["LOWER_VERTICAL_PAD"]
        pose["slope_sides"] = sides["UPPER_SLOPED_PART"]
        pose["dominant"] = max(forces, key=forces.get) if sum(forces.values()) > 1e-6 else "NONE"
        pose["fn"] = float(sum(item["fn"] for item in raw))
        pose["fz_n"] = float(sum(item["fz_n"] for item in raw))
        peak, sat = _wrist_sat(sim)
        pose["tau_peak"] = peak
        pose["sat"] = sat
        pose["table"] = bool(latched(sim) or pose["bottom"] < TABLE_TOP)
        decision = choose(pose, raw, state)
        pose["mode"] = state["mode"]
        pose["cmd_z"] = float(decision["v"][2])
        pose["cmd_tau"] = float(decision["tau"])
        rows.append(pose)
        for item in raw:
            item = dict(item)
            item["t"] = pose["t"]
            item["cz"] = pose["cz"]
            item["gap"] = pose["gap"]
            item["mode"] = state["mode"]
            contacts.append(item)
        if pose["table"] or decision.get("stop"):
            handoff_obs = observe_airborne(sim.model, sim.data, sim.ids, sim.fsm, hist)
            break
        observations.append(observe_airborne(sim.model, sim.data, sim.ids, sim.fsm, hist))
        action = _action_from(sim, decision["v"], np.zeros(3), decision["tau"])
        step_legal(sim, action)
        actions.append(np.asarray(action, np.float32))
    return {
        "rows": rows,
        "actions": actions,
        "contacts": contacts,
        "observations": observations,
        "handoff_obs": handoff_obs,
        "metrics": _metrics(rows) if rows else {},
    }


def _xy(pose):
    return np.array([
        float(np.clip(3.0 * pose["p_rel"][0], -0.2, 0.2)),
        float(np.clip(3.0 * pose["p_rel"][1], -0.2, 0.2)),
    ])


def original_choose(pose, _raw, state):
    """The over-deep chase that recontacted around 0.14 s."""
    rh_z = -float(2.2)
    centered = abs(float(pose["p_rel"][0])) < 0.025 and abs(float(pose["p_rel"][1])) < 0.02
    below = float(pose["pinch"][2]) <= float(pose["p_obj"][2]) - 0.004
    both = pose["nL"] > 0 and pose["nR"] > 0
    mode = state["mode"]
    if mode in ("open", "chase"):
        tau = TAU_OPEN if pose["gap"] < 0.044 else 0.0
        mode = "chase" if pose["gap"] >= 0.044 else "open"
        if below and centered and pose["gap"] >= 0.042 and pose["bottom"] > 0.50:
            mode = "close"
        vz = rh_z
    elif mode == "close":
        tau = TAU_SECURE if both else -8.0
        vz = float(min(pose["v_obj"][2] - 0.30, -0.20))
        state["both"] = state.get("both", 0) + 1 if both else 0
        if state["both"] >= 2:
            mode = "brake"
    elif mode == "brake":
        tau = TAU_SECURE
        speed = float(pose["v_obj"][2])
        vz = 0.0 if speed >= -0.02 else min(0.0, speed + 6.0 * 0.02)
        if both and pose["v_obj"][2] > -0.05 and pose["v_hand"][2] > -0.08:
            mode = "lift"
        if not both:
            mode = "close"
    else:
        tau = TAU_SECURE
        vz = 0.08
    state["mode"] = mode
    hand_xy = _xy(pose)
    # _xy is in hand frame; world XY correction stays small and is applied in world
    # by the same mapping the previous oracle used after rotating. Here the
    # stored p_rel is hand frame, and world X/Y are nearly hand Y/X. Keep the
    # correction in world by using the object velocity for X/Y during close.
    if mode in ("open", "chase"):
        v = np.array([hand_xy[1], hand_xy[0], vz])
    else:
        v = np.array([
            float(np.clip(pose["v_obj"][0], -0.3, 0.3)),
            float(np.clip(pose["v_obj"][1], -0.3, 0.3)),
            vz,
        ])
    v = np.clip(v, [-1.0, -1.0, -3.8], [1.0, 1.0, 1.0])
    return {"v": v, "tau": tau}


def staged_choose(spec):
    def choose(pose, _raw, state):
        both = pose["nL"] > 0 and pose["nR"] > 0
        pad = pose["pad_sides"] == "LR" and pose["fn_pad"] >= pose["fn_slope"]
        mode = state["mode"]
        xy = np.array([
            float(np.clip(pose["v_obj"][0], -0.25, 0.25)),
            float(np.clip(pose["v_obj"][1], -0.25, 0.25)),
        ])
        tau = spec["arrest"]
        vz = 0.0
        if mode == "open":
            tau = TAU_OPEN if pose["gap"] < spec["open_gap"] else spec["arrest"]
            vz = -spec["down"]
            if pose["gap"] >= spec["open_gap"]:
                mode = "chase"
        if mode == "chase":
            tau = spec["arrest"] if pose["gap"] >= spec["open_gap"] else TAU_OPEN
            vz = -spec["down"]
            deep_enough = pose["cz"] <= spec["stop_cz"]
            if deep_enough and pose["gap"] >= spec["close_gap"]:
                mode = "close"
            elif deep_enough:
                mode = "hold"
            elif pose["t"] > spec["chase_limit"]:
                mode = "fail"
        if mode == "hold":
            tau = TAU_OPEN if pose["gap"] < spec["open_gap"] else spec["arrest"]
            vz = _depth_vz(pose, spec)
            if pose["gap"] >= spec["close_gap"]:
                mode = "close"
        if mode == "close":
            tau = spec["hold_tau"] if pad else spec["close_tau"]
            vz = _depth_vz(pose, spec) if not spec["close_dive"] else float(min(pose["v_obj"][2] - 0.30, -0.20))
            state["pad"] = state.get("pad", 0) + 1 if pad else 0
            if state["pad"] >= spec["pad_steps"]:
                mode = "brake"
        elif mode == "brake":
            tau = spec["hold_tau"] if pad else spec["close_tau"]
            speed = float(pose["v_obj"][2])
            vz = 0.0 if speed >= -0.02 else min(0.0, speed + spec["brake"] * 0.02)
            vz = _blend_depth(vz, pose, spec)
            if not pad:
                state["pad"] = 0
                mode = "close"
            elif speed > -0.05 and pose["v_hand"][2] > -0.10 and state.get("pad", 0) >= spec["pad_steps"]:
                mode = "lift"
        elif mode == "lift":
            tau = spec["hold_tau"]
            risen = pose["bottom"] > float(spec.get("settle_bottom", 0.56)) and state.get("pad", 0) >= spec["pad_steps"]
            vz = 0.0 if risen else spec["lift"]
            if risen:
                mode = "settle"
        elif mode == "settle":
            tau = spec["hold_tau"]
            vz = 0.0
            quiet = pose["pad_sides"] == "LR" and abs(float(pose["v_obj"][2])) < 0.06 and pose["bottom"] > 0.52
            state["quiet"] = state.get("quiet", 0) + 1 if quiet else 0
            if state["quiet"] >= 8:
                mode = "handoff"
        elif mode == "fail":
            tau = spec["arrest"]
            vz = 0.0
        state["mode"] = mode
        v = np.clip(np.array([xy[0], xy[1], vz]), [-1.0, -1.0, -3.8], [1.0, 1.0, 1.0])
        return {"v": v, "tau": float(tau), "stop": mode in ("fail", "handoff")}

    return choose


def _depth_vz(pose, spec):
    """Match the cylinder, and stop driving the hand once the pad band is reached."""
    vz = float(np.clip(pose["v_obj"][2], -1.0, 0.4))
    if pose["cz"] < spec["stop_cz"] - 0.003:
        vz = max(vz, 0.05)
    elif pose["cz"] > spec["stop_cz"] + 0.004:
        vz = min(vz, -spec["down"])
    return vz


def _blend_depth(vz, pose, spec):
    if pose["cz"] < spec["stop_cz"] - 0.006:
        return max(vz, 0.0)
    return vz


def _base_spec():
    return {
        "open_gap": 0.044,
        "arrest": 0.0,
        "close_gap": 0.042,
        "down": 2.2,
        "stop_cz": 0.100,
        "close_dive": False,
        "close_tau": -8.0,
        "hold_tau": -18.0,
        "pad_steps": 2,
        "brake": 6.0,
        "lift": 0.08,
        "chase_limit": 0.35,
    }


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _public(stage, spec, result):
    row = {"stage": stage, **{key: spec[key] for key in spec}}
    row.update(result["metrics"])
    return row


def _save_contacts(path, contacts):
    if not contacts:
        return
    np.savez_compressed(
        path,
        t=np.array([item["t"] for item in contacts]),
        side=np.array([item["side"] for item in contacts]),
        klass=np.array([item["class"] for item in contacts]),
        local=np.array([item["local"] for item in contacts]),
        fn=np.array([item["fn"] for item in contacts]),
        ft=np.array([item["ft"] for item in contacts]),
        fz_n=np.array([item["fz_n"] for item in contacts]),
        dist=np.array([item["dist"] for item in contacts]),
        cz=np.array([item["cz"] for item in contacts]),
        gap=np.array([item["gap"] for item in contacts]),
    )


def _plot(path, contacts, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "LOWER_VERTICAL_PAD": "#1b7f4e",
        "UPPER_SLOPED_PART": "#c0392b",
        "EDGE_CONTACT": "#d68910",
        "OTHER": "#5d6d7e",
    }
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 4.2), sharey=True)
    for axis, side in zip(axes, ("L", "R")):
        chosen = [item for item in contacts if item["side"] == side]
        for name, color in colors.items():
            pts = [item for item in chosen if item["class"] == name]
            if not pts:
                continue
            axis.scatter(
                [item["local"][1] * 1000.0 for item in pts],
                [item["local"][2] * 1000.0 for item in pts],
                s=[8.0 + 2.0 * item["fn"] for item in pts],
                c=color, label=name, alpha=0.85,
            )
        axis.add_patch(plt.Rectangle(
            ((MAIN_CENTER[1] - MAIN_HALF[1]) * 1000.0, (MAIN_CENTER[2] - MAIN_HALF[2]) * 1000.0),
            2 * MAIN_HALF[1] * 1000.0, 2 * MAIN_HALF[2] * 1000.0,
            fill=False, linestyle="--", color="#1b7f4e",
        ))
        axis.set_title(f"{side} finger local y–z")
        axis.set_xlabel("local y (mm)")
        axis.grid(True, alpha=0.3)
    axes[0].set_ylabel("local z (mm), tip is up")
    axes[1].legend(fontsize=7, loc="best")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _better(candidate, incumbent):
    if incumbent is None:
        return True
    left = _rank(candidate["metrics"])
    right = _rank(incumbent["metrics"])
    return left > right


def _rank(metrics):
    return (
        int(metrics["t_pad_hold"] is not None),
        int(not metrics["table"]),
        int(metrics["lifted"]),
        int(metrics["t_pad"] is not None),
        metrics["min_bottom"],
        -abs((metrics["re_cz"] if metrics["re_cz"] is not None else 0.0) - 0.102),
        metrics["min_fz_n"],
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "VIDEOS").mkdir(exist_ok=True)
    sim, _base, ref, _target, _book = _setup()
    book = load_codebook()
    cell = next(item for item in _cells() if item["id"] == IC_ID)
    info, err = _impact(sim, _base, cell)
    if err:
        raise RuntimeError(err)
    band = _pad_band(sim)
    print("PAD", [round(v, 4) for v in band], "cz0", round(info["pose"]["cz"], 4), flush=True)
    nominal = _nominal_fail(sim, info["snap"], ref)
    print("NOMINAL", nominal["why"], nominal["t"], flush=True)

    original = rollout(sim, info["snap"], original_choose, 1.6)
    _save_contacts(OUT / "original_contacts.npz", original["contacts"])
    _plot(OUT / "original_contact_map.png", original["contacts"], "original over-chase contacts")
    window = [item for item in original["contacts"] if 0.02 - 1e-6 <= item["t"] <= 0.30 + 1e-6]
    at_14 = [item for item in original["contacts"] if abs(item["t"] - 0.14) < 0.011]
    audit_contacts = []
    for item in at_14:
        audit_contacts.append({
            "t": round(item["t"], 3),
            "side": item["side"],
            "geom": item["geom"],
            "class": item["class"],
            "kind": item["kind"],
            "local_mm": [round(1000.0 * float(v), 2) for v in item["local"]],
            "offset_mm": [round(1000.0 * float(v), 2) for v in item["offset"]],
            "normal": [round(float(v), 3) for v in item["normal"]],
            "fn": round(item["fn"], 2),
            "ft": round(item["ft"], 2),
            "fz_n": round(item["fz_n"], 2),
            "dist_mm": round(1000.0 * item["dist"], 3),
            "dim": item["dim"],
            "efc": item["efc"],
            "cz": round(item["cz"], 4),
            "gap_mm": round(1000.0 * item["gap"], 1),
        })
    by_t = {}
    for item in original["contacts"]:
        by_t.setdefault(round(item["t"], 3), []).append(item)
    spike_t = min(by_t, key=lambda key: sum(item["fz_n"] for item in by_t[key])) if by_t else None
    spike_rows = [] if spike_t is None else sorted(by_t[spike_t], key=lambda item: item["fz_n"])[:8]
    def _jsonable(value):
        if isinstance(value, dict):
            return {key: _jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_jsonable(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    (OUT / "contact_audit_0p14.json").write_text(json.dumps(_jsonable({
        "pad_band_hand_z": [round(band[0], 4), round(band[1], 4)],
        "n_window": len(window),
        "n_at_0p14": len(at_14),
        "class_counts": {
            name: sum(1 for item in at_14 if item["class"] == name)
            for name in ("LOWER_VERTICAL_PAD", "UPPER_SLOPED_PART", "EDGE_CONTACT", "OTHER")
        },
        "contacts": audit_contacts,
        "worst_normal": [
            {
                "t": spike_t,
                "class": item["class"],
                "kind": item["kind"],
                "fn": round(item["fn"], 2),
                "ft": round(item["ft"], 2),
                "fz_n": round(item["fz_n"], 2),
                "local_mm": [round(1000.0 * float(v), 2) for v in item["local"]],
            }
            for item in spike_rows
        ],
        "metrics": original["metrics"],
    }), indent=2), encoding="utf-8")
    print("AUDIT 0.14", json.loads((OUT / "contact_audit_0p14.json").read_text(encoding="utf-8"))["class_counts"], "spike", spike_t, flush=True)

    depth_rows = []
    best = None
    best_spec = None
    for stop_cz in (0.110, 0.106, 0.102, 0.098, 0.094):
        for close_dive in (True, False):
            spec = _base_spec()
            spec["stop_cz"] = stop_cz
            spec["close_dive"] = close_dive
            result = rollout(sim, info["snap"], staged_choose(spec), 1.4)
            depth_rows.append(_public("A", spec, result))
            print("A", stop_cz, "dive", close_dive, result["metrics"]["re_class"], result["metrics"]["re_cz"], "pad", result["metrics"]["t_pad"], "bot", result["metrics"]["min_bottom"], result["metrics"]["end_mode"], flush=True)
            if _better(result, best):
                best, best_spec = result, dict(spec)
    _write(OUT / "OVERCHASE_DEPTH_SWEEP.csv", depth_rows)

    close_rows = []
    spec_b = dict(best_spec)
    spec_b["close_dive"] = False
    for open_gap, close_gap, arrest in (
        (0.044, 0.042, 0.0),
        (0.050, 0.046, -1.0),
        (0.058, 0.048, -1.0),
        (0.070, 0.050, -1.0),
    ):
        for early in (0.0, 0.004):
            spec = dict(spec_b)
            spec["open_gap"] = open_gap
            spec["close_gap"] = close_gap
            spec["arrest"] = arrest
            spec["stop_cz"] = spec_b["stop_cz"] + early
            result = rollout(sim, info["snap"], staged_choose(spec), 1.4)
            close_rows.append(_public("B", spec, result))
            print("B", open_gap, early, result["metrics"]["re_class"], result["metrics"]["t_re"], result["metrics"]["gap_max_mm"], result["metrics"]["min_bottom"], flush=True)
            if _better(result, best):
                best, best_spec = result, dict(spec)

    spec_c = dict(best_spec)
    spec_c["close_dive"] = False
    for close_tau in (-4.0, -6.0, -8.0, -12.0):
        for hold_tau in (-6.0, -10.0, -18.0):
            spec = dict(spec_c)
            spec["close_tau"] = close_tau
            spec["hold_tau"] = hold_tau
            result = rollout(sim, info["snap"], staged_choose(spec), 1.4)
            close_rows.append(_public("C", spec, result))
            print("C", close_tau, hold_tau, result["metrics"]["re_class"], result["metrics"]["min_fz_n"], result["metrics"]["gap_min_mm"], result["metrics"]["t_pad_hold"], flush=True)
            if _better(result, best):
                best, best_spec = result, dict(spec)
    _write(OUT / "CLOSE_TIMING_FORCE_SWEEP.csv", close_rows)

    brake_rows = []
    spec_d = dict(best_spec)
    for brake in (3.0, 6.0, 10.0):
        for pad_steps in (1, 3):
            spec = dict(spec_d)
            spec["brake"] = brake
            spec["pad_steps"] = pad_steps
            result = rollout(sim, info["snap"], staged_choose(spec), 1.6)
            brake_rows.append(_public("D", spec, result))
            print("D", brake, pad_steps, result["metrics"]["min_bottom"], result["metrics"]["t_vz0"], result["metrics"]["end_bottom"], result["metrics"]["table"], flush=True)
            if _better(result, best):
                best, best_spec = result, dict(spec)
    for lift in (0.08, 0.15):
        spec = dict(best_spec)
        spec["lift"] = lift
        result = rollout(sim, info["snap"], staged_choose(spec), 2.2)
        brake_rows.append(_public("E", spec, result))
        print("E", lift, result["metrics"]["end_mode"], result["metrics"]["end_bottom"], result["metrics"]["lifted"], result["metrics"]["end_cz"], flush=True)
        if _better(result, best):
            best, best_spec = result, dict(spec)
    _write(OUT / "BRAKE_LIFT_RESULTS.csv", brake_rows)

    label = "NO_VERTICAL_PAD_HOLD"
    formal = None
    extension = None
    k48 = None
    held = best["metrics"]["t_pad_hold"] is not None and not best["metrics"]["table"]
    if held and best["metrics"]["end_mode"] == "handoff":
        actions = np.stack(best["actions"])
        packed = representability(actions, book)
        k48 = None if packed["max_linf"] is None else float(packed["max_linf"])
        if k48 is not None and k48 <= 0.25 + 1e-8:
            restore_ballistic(sim, info["snap"])
            restore_state(sim, None)
            for action in packed["quantized"]:
                step_legal(sim, action)
            out, extension = _finish(sim, ref, extra=5.0)
            formal = out.get("reason")
            label = formal if out.get("success") and extension == "HELD_EXTENSION" else (
                "FORMAL_SUCCESS_ONLY" if out.get("success") else formal
            )
            np.savez_compressed(
                OUT / "best_actions.npz",
                actions=actions,
                tokens=packed["tokens"],
                residual=(actions - book[packed["tokens"]]).astype(np.float32),
            )
        else:
            label = "K48_OUT_OF_RANGE"
    elif held:
        label = "PARTIAL_RECATCH"
    elif best["metrics"]["table"]:
        label = "TABLE"
    print("BEST", label, best_spec["stop_cz"], best["metrics"]["end_mode"], formal, extension, k48, flush=True)

    _save_contacts(OUT / "best_contacts.npz", best["contacts"])
    _plot(OUT / "best_contact_map.png", best["contacts"], f"best {label}")
    np.savez_compressed(
        OUT / "best_trace.npz",
        t=np.array([row["t"] for row in best["rows"]]),
        cz=np.array([row["cz"] for row in best["rows"]]),
        gap=np.array([row["gap"] for row in best["rows"]]),
        bottom=np.array([row["bottom"] for row in best["rows"]]),
        v_obj=np.array([row["v_obj"] for row in best["rows"]]),
        v_hand=np.array([row["v_hand"] for row in best["rows"]]),
        p_hand=np.array([row["p_hand"] for row in best["rows"]]),
        mode=np.array([row["mode"] for row in best["rows"]]),
        qpos=np.array([row["qpos"] for row in best["rows"]]),
        nL=np.array([row["nL"] for row in best["rows"]]),
        nR=np.array([row["nR"] for row in best["rows"]]),
    )
    t_off = info["rows"][-1]["t"] + 0.02
    video_rows = list(info["rows"])
    for row in best["rows"]:
        copied = dict(row)
        copied["t"] = row["t"] + t_off
        video_rows.append(copied)
    _video(video_rows, OUT / "VIDEOS" / "best.mp4", IC_ID)
    timeline = {
        "label": label,
        "formal": formal,
        "extension": extension,
        "k48_linf": k48,
        "nominal": nominal["why"],
        "nominal_t": nominal["t"],
        "spec": best_spec,
        "metrics": best["metrics"],
        "pad_band": [round(band[0], 4), round(band[1], 4)],
    }
    (OUT / "best.json").write_text(json.dumps(_jsonable(timeline), indent=2), encoding="utf-8")
    with (OUT / "EVENT_TIMELINE.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "t_open", "t_off", "t_re", "t_pad", "t_pad_hold", "t_vz0", "end_mode", "formal", "extension"])
        writer.writeheader()
        writer.writerow({"label": label, "formal": formal, "extension": extension, **{key: best["metrics"][key] for key in ("t_open", "t_off", "t_re", "t_pad", "t_pad_hold", "t_vz0", "end_mode")}})
    print("WROTE", OUT, flush=True)


if __name__ == "__main__":
    main()
