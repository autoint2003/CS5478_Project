"""Legal Open, right-and-down chase, Close, Brake for a vertical cylinder.

The initial conditions are the original sliding snapshots. Privileged pose is
used only to choose a recovery7d_airborne_v3 action. The action is projected
onto the K48 codebook plus a residual of at most 0.25 before it is applied.
Nothing here edits step_legal, the nominal controller, or the scene.
"""

from __future__ import annotations


import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.simulation.capture import restore_capture, take_capture
from controllers.recovery.geometry_switch import GeometrySwitch
from controllers.recovery.action_map import AIRBORNE_V3, TAU_OPEN, TAU_SECURE, normalize_action
from envs.observations.airborne_obs import observe_airborne
from envs.simulation.grasp_quality import body_twist
from envs.observations.observable_obs import ObservableObsState
from envs.outcomes.physical_recovery import physical_pack
from data_generation.strategies.capture_geometry import capture_geometry

from envs.outcomes.horizon_outcome import NOMINAL_S, nominal_verify
from envs.ballistic_impact import world
from data_generation.oracles.representability import representability
from envs.step_legal import step_legal
from envs.outcomes.recovery_success import nominal_step
from envs.simulation.scene_state import TABLE_TOP
from envs.outcomes.table_drop import latched
from data_generation.strategies.recatch_oracle import _class_force, contacts_now
from training.k47_codebook import load_codebook

OUT = ROOT / "results" / "diagnostics" / "vertical_recatch"
DT = 0.02
LIMIT = 4.0
CYL_R = 0.018
CYL_HALF = 0.03


def make_env():
    sim, gains, nominal, *_rest = world()

    class Env:
        pass

    env = Env()
    env.sim = sim
    env.switch = GeometrySwitch()
    env.gains = gains
    env.nominal = nominal
    return env


def load_blob(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def project(raw, book):
    packed = representability(np.asarray(raw, np.float32).reshape(1, 7), book)
    return np.asarray(packed["quantized"][0], np.float32), float(packed["max_linf"])


def _bottom(sim):
    axis = np.array(sim.data.xmat[sim.ids.object_body].reshape(3, 3), float)[:, 2]
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    vertical = abs(float(axis[2]))
    radial = float(np.sqrt(max(0.0, 1.0 - vertical * vertical)))
    extent = CYL_HALF * vertical + CYL_R * radial
    return float(sim.data.xpos[sim.ids.object_body][2]) - extent


def _sample(sim, t0):
    pack = physical_pack(sim)
    geo = capture_geometry(sim)
    raw = contacts_now(sim)
    forces, sides = _class_force(raw)
    vo, wo = body_twist(sim.model, sim.data, sim.ids.object_body)
    axis = np.array(sim.data.xmat[sim.ids.object_body].reshape(3, 3), float)[:, 2]
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    return {
        "t": float(sim.data.time) - t0,
        "rh": np.asarray(pack["rh"], float).copy(),
        "v_rel": np.asarray(pack["v_rel_h"], float).copy(),
        "v_obj": np.asarray(vo, float).copy(),
        "w_obj": np.asarray(wo, float).copy(),
        "axis": axis,
        "tilt": float(np.degrees(np.arccos(np.clip(abs(axis[2]), -1.0, 1.0)))),
        "nL": int(pack["nL"]),
        "nR": int(pack["nR"]),
        "z": float(pack["obj_z"]),
        "bottom": _bottom(sim),
        "gap": float(geo["inner_gap_m"]),
        "ready": bool(geo["ready_to_close"]),
        "between": bool(geo["between"]),
        "cx": float(geo["center_hand_m"][0]),
        "cy": float(geo["center_hand_m"][1]),
        "cz": float(geo["center_hand_m"][2]),
        "axial": float(geo["axial_overlap_m"]),
        "fn_pad": float(forces["LOWER_VERTICAL_PAD"]),
        "fn_slope": float(forces["UPPER_SLOPED_PART"]),
        "fn_edge": float(forces["EDGE_CONTACT"]),
        "pad_sides": sides["LOWER_VERTICAL_PAD"],
        "dominant": max(forces, key=forces.get) if sum(forces.values()) > 1e-6 else "NONE",
        "table": bool(latched(sim)),
    }


def _chase(sample, state, spec):
    """Match the cylinder, then correct only while it is still near the pads.

    Hand +z is world down. A large positive vz while the cylinder is still
    moving upward dives the hand out from under it. The correction is capped
    so the close can happen inside the pad span.
    """
    rotation = np.array(state["sim"].data.xmat[state["sim"].ids.hand_body].reshape(3, 3), float)
    v_obj_h = rotation.T @ sample["v_obj"]
    chase = v_obj_h.copy()
    chase[0] += float(spec["kp"]) * sample["cx"]
    chase[1] += float(spec["kp"]) * sample["cy"]
    chase[2] += float(spec["kz"]) * (sample["cz"] - float(spec["z"]))
    chase[0] = float(np.clip(chase[0], -1.0, 1.0))
    chase[1] = float(np.clip(chase[1], -1.0, 1.0))
    chase[2] = float(np.clip(chase[2], -0.35, 3.8 if state["mode"] == "close" else float(spec["down"])))
    return chase, v_obj_h


def _in_jaw(sample, spec):
    return (
        sample["between"]
        and sample["axial"] >= float(spec["axial"])
        and abs(sample["cx"]) <= float(spec["x_tol"])
        and abs(sample["cy"]) <= float(spec["y_tol"])
        and abs(sample["cz"] - float(spec["z"])) <= float(spec["z_tol"])
        and sample["gap"] >= float(spec["gap"])
    )


def _command(sample, state, spec):
    chase, v_obj_h = _chase(sample, state, spec)
    mode = state["mode"]
    if sample["nL"] == 0 and sample["nR"] == 0:
        state["released"] = True
    if mode == "open":
        if state["released"] and sample["t"] >= float(spec["open_min"]):
            if _in_jaw(sample, spec):
                state["mode"] = "close"
                state["pad_run"] = 0
                state["t_close"] = sample["t"]
            else:
                state["mode"] = "chase"
        chase, v_obj_h = _chase(sample, state, spec)
        state["v"] = chase
        if state["mode"] == "close":
            return chase, TAU_SECURE
        return chase, TAU_OPEN
    if mode == "chase":
        if state["released"] and _in_jaw(sample, spec):
            state["mode"] = "close"
            state["pad_run"] = 0
            state["t_close"] = sample["t"]
        elif sample["t"] > float(spec["chase_max"]):
            state["mode"] = "fail"
        chase, v_obj_h = _chase(sample, state, spec)
        state["v"] = chase
        if state["mode"] == "close":
            return chase, TAU_SECURE
        return chase, TAU_OPEN
    if mode == "close":
        chase, v_obj_h = _chase(sample, state, spec)
        hold = chase
        state["v"] = hold
        if sample["pad_sides"] == "LR" and sample["fn_pad"] >= sample["fn_slope"] and sample["fn_pad"] > 0.4:
            state["pad_run"] += 1
        else:
            state["pad_run"] = 0
        if state["pad_run"] >= int(spec["pad_steps"]):
            state["mode"] = "brake"
            state["v_brake"] = hold.copy()
        if sample["t"] - state.get("t_close", sample["t"]) > 0.40 and state["pad_run"] == 0:
            state["mode"] = "fail"
        state["t_close"] = state.get("t_close", sample["t"])
        return hold, TAU_SECURE
    if mode == "brake":
        bleed = float(spec["brake"]) * DT
        command = state["v_brake"]
        command = command - np.sign(command) * np.minimum(np.abs(command), bleed)
        state["v_brake"] = command
        state["v"] = command
        slow = float(np.linalg.norm(sample["v_obj"][:2])) < 0.10 and sample["bottom"] > 0.50
        rising = sample["bottom"] < 0.50
        if (slow or rising) and sample["pad_sides"] == "LR":
            state["mode"] = "lift"
            state["z_lift"] = sample["z"]
        return command, TAU_SECURE
    if mode == "lift":
        up = np.array([0.0, 0.0, -float(spec["lift"])], float)
        state["v"] = up
        if sample["z"] >= state["z_lift"] + float(spec["rise"]) and sample["pad_sides"] == "LR":
            state["mode"] = "settle"
            state["quiet"] = 0
        return up, TAU_SECURE
    if mode == "settle":
        state["v"] = np.zeros(3)
        calm = (
            sample["pad_sides"] == "LR"
            and abs(float(sample["v_rel"][0])) < 0.06
            and abs(float(sample["v_rel"][1])) < 0.06
            and abs(float(sample["v_obj"][2])) < 0.06
        )
        state["quiet"] = state.get("quiet", 0) + 1 if calm else 0
        if state["quiet"] >= 8:
            state["mode"] = "handoff"
        return np.zeros(3), TAU_SECURE
    return np.zeros(3), TAU_SECURE


def rollout(env, blob, book, spec):
    restore_capture(env, blob["capture"])
    sim = env.sim
    t0 = float(sim.data.time)
    hist = ObservableObsState(DT)
    state = {"mode": "open", "released": False, "pad_run": 0, "sim": sim, "v": np.zeros(3)}
    rows = []
    actions = []
    observations = []
    gaps = []
    while float(sim.data.time) - t0 < LIMIT - 1e-9 and state["mode"] not in ("handoff", "fail"):
        sample = _sample(sim, t0)
        if sample["table"] or sample["bottom"] < TABLE_TOP + 0.005:
            state["mode"] = "table"
            rows.append(sample)
            break
        velocity, tau = _command(sample, state, spec)
        sample["mode"] = state["mode"]
        sample["cmd"] = np.asarray(velocity, float).copy()
        raw = normalize_action(velocity, np.zeros(3), tau, AIRBORNE_V3)
        action, linf = project(raw, book)
        gaps.append(linf)
        observations.append(np.asarray(observe_airborne(sim.model, sim.data, sim.ids, sim.fsm, hist), np.float32))
        step_legal(sim, action)
        actions.append(action)
        rows.append(sample)
    end = _sample(sim, t0)
    end["mode"] = state["mode"]
    handoff_obs = np.asarray(observe_airborne(sim.model, sim.data, sim.ids, sim.fsm, hist), np.float32)
    return {
        "rows": rows,
        "end": end,
        "actions": np.stack(actions, axis=0) if actions else np.zeros((0, 7), np.float32),
        "observations": np.stack(observations, axis=0) if observations else np.zeros((0, 39), np.float32),
        "handoff_obs": handoff_obs,
        "max_linf": float(max(gaps) if gaps else 0.0),
        "mode": state["mode"],
    }


def _finish(sim, ref):
    class Env:
        pass

    env = Env()
    env.sim = sim
    out = nominal_verify(env, ref, seconds=NOMINAL_S)
    extra = None
    if out.get("success"):
        t0 = float(sim.data.time)
        extra = "HELD_EXTENSION"
        while float(sim.data.time) < t0 + 5.0:
            nominal_step(sim)
            if latched(sim) or _bottom(sim) < TABLE_TOP:
                extra = "FELL_DURING_EXTENSION"
                break
            geo = capture_geometry(sim)
            if int(geo["nL"]) == 0 or int(geo["nR"]) == 0:
                extra = "LOST_DURING_EXTENSION"
                break
    return out, extra


def _summary(result):
    rows = result["rows"]

    def first(pred):
        for row in rows:
            if pred(row):
                return row
        return None

    off = first(lambda row: row["nL"] == 0 and row["nR"] == 0)
    seen = False
    recontact = None
    pad = None
    for row in rows:
        if row["nL"] == 0 and row["nR"] == 0:
            seen = True
        elif seen and recontact is None and row["nL"] > 0 and row["nR"] > 0:
            recontact = row
        if seen and pad is None and row["pad_sides"] == "LR":
            pad = row
    cmds = np.stack([row["cmd"] for row in rows if "cmd" in row], axis=0) if any("cmd" in row for row in rows) else np.zeros((1, 3))
    return {
        "mode": result["mode"],
        "t_off": None if off is None else round(off["t"], 3),
        "t_re": None if recontact is None else round(recontact["t"], 3),
        "t_pad": None if pad is None else round(pad["t"], 3),
        "re_class": None if recontact is None else recontact["dominant"],
        "pad_class": None if pad is None else "LOWER_VERTICAL_PAD",
        "min_bottom": round(min(row["bottom"] for row in rows), 4),
        "end_z": round(result["end"]["z"], 4),
        "end_bottom": round(result["end"]["bottom"], 4),
        "end_rh": [round(float(x), 4) for x in result["end"]["rh"]],
        "max_right": round(float(np.max(np.abs(cmds[:, 0]))), 3),
        "max_open": round(float(np.max(np.abs(cmds[:, 1]))), 3),
        "max_down": round(float(np.max(cmds[:, 2])), 3),
        "max_linf": round(result["max_linf"], 4),
        "steps": int(len(result["actions"])),
        "tilt0": round(rows[0]["tilt"], 2) if rows else None,
        "vh0": [round(float(x), 4) for x in rows[0]["v_rel"]] if rows else None,
    }


def specs():
    """Laws measured on the vertical sliding grasp. Downward speed stays capped until close."""
    base = {
        "z": 0.098,
        "gap": 0.036,
        "chase_max": 0.45,
        "x_tol": 0.020,
        "y_tol": 0.015,
        "z_tol": 0.018,
        "axial": 0.008,
        "pad_steps": 2,
        "lift": 0.25,
        "rise": 0.04,
    }
    grid = []
    for kp, kz, down, open_min, brake, name in (
        (10.0, 4.0, 0.8, 0.02, 10.0, "follow_early"),
        (16.0, 6.0, 1.2, 0.04, 8.0, "later_close"),
        (8.0, 3.0, 0.6, 0.02, 12.0, "gentle_brake"),
    ):
        spec = dict(base)
        spec.update({"kp": kp, "kz": kz, "down": down, "open_min": open_min, "brake": brake, "name": name})
        grid.append(spec)
    return grid


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    env = make_env()
    book = load_codebook()
    targets = [
        "ball_train_v2_0168",
        "ball_train_0017",
        "ball_train_v2_0070",
        "ball_train_v2_0098",
    ]
    manifest = {}
    for folder in ("controlled_sliding", "inward_sliding"):
        path = ROOT / "data" / folder / "manifest.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                manifest.setdefault(row["id"], row)
    log = OUT / "search.jsonl"
    if log.exists():
        log.unlink()
    wins = []
    for episode in targets:
        row = manifest[episode]
        blob = load_blob(ROOT / row["snapshot"])
        restore_capture(env, blob["capture"])
        sim = env.sim
        rotation = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
        sample = _sample(sim, float(sim.data.time))
        print(
            "IC", episode,
            "az", round(float(blob["spec"]["azimuth_deg"]), 1),
            "spd", round(float(blob["spec"]["speed_mps"]), 2),
            "hand_x", np.round(rotation[:, 0], 3).tolist(),
            "hand_y", np.round(rotation[:, 1], 3).tolist(),
            "hand_z", np.round(rotation[:, 2], 3).tolist(),
            "rh", np.round(sample["rh"], 4).tolist(),
            "v_rel", np.round(sample["v_rel"], 4).tolist(),
            "v_obj", np.round(sample["v_obj"], 4).tolist(),
            "axis", np.round(sample["axis"], 3).tolist(),
            "strategy", row["strategy"],
            flush=True,
        )
        for index, spec in enumerate(specs()):
            result = rollout(env, blob, book, spec)
            summary = _summary(result)
            summary.update({"id": episode, "trial": index, "spec": spec})
            if result["mode"] == "handoff" and summary["t_off"] is not None and summary["t_pad"] is not None:
                out, extra = _finish(sim, blob["ref"])
                summary["formal"] = out.get("reason")
                summary["extension"] = extra
                summary["success"] = bool(out.get("success") and extra == "HELD_EXTENSION")
                if summary["success"]:
                    wins.append((episode, index, result, summary))
                    np.savez_compressed(
                        OUT / f"{episode}__t{index}.npz",
                        actions=result["actions"],
                        observations=result["observations"],
                    )
            else:
                summary["formal"] = None
                summary["extension"] = None
                summary["success"] = False
            with log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(summary) + "\n")
            if index % 6 == 0 or summary["success"] or result["mode"] in ("handoff", "table", "fail"):
                print(
                    " ", episode, index, result["mode"],
                    "off", summary["t_off"], "re", summary["t_re"], "pad", summary["t_pad"],
                    "class", summary["re_class"], "minb", summary["min_bottom"],
                    "down", summary["max_down"], "openax", summary["max_open"],
                    "formal", summary["formal"], "ext", summary["extension"],
                    flush=True,
                )
            if summary["success"]:
                break
    print("wins", [(item[0], item[1]) for item in wins], flush=True)


if __name__ == "__main__":
    main()
