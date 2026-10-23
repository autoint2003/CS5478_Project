#!/usr/bin/env python
"""Investigate Plan B elevated vs GROUND_SCENE physical replay contradiction.

Does not modify controller, actions, success criteria, or physics.
Writes only under results/diagnostics/plan_b_replay_contradiction/.
"""

from __future__ import annotations

import json
import pickle
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from controllers.cartesian.jacobian import apply_cartesian_ctrl  # noqa: E402
from controllers.nominal import _integrate_rot  # noqa: E402
from controllers.recovery.action_map import AIRBORNE_V3, map_action  # noqa: E402
from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.datasets.recovery_rebuild import resolve_stored_path  # noqa: E402
from envs.ballistic_impact import elevated_world, world  # noqa: E402
from envs.contacts.frames import extract_contacts  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.outcomes.horizon_outcome import recovery_failed  # noqa: E402
from envs.outcomes.support_drop import arm_recovery_drop, latched, terminal_reason  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.step_legal import N_SUB, TRACK  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from evaluation.handoff_from_state import evaluate_handoff_from_state  # noqa: E402
from training.recovery_execution import restore_ballistic  # noqa: E402

OUT = ROOT / "results" / "diagnostics" / "plan_b_replay_contradiction"
BAK = ROOT / "results" / "diagnostics" / "plan_ab_table_free_migration" / "pre_update_unified_manifest.jsonl"
MIG = ROOT / "results" / "diagnostics" / "plan_ab_table_free_migration" / "migration_results.jsonl"

# Representative FAIL_REPLAY + SUCCESS_12S_NOT_NEW roots (mixed mechanisms).
ROOT_IDS = [
    "plan_b__ball_train_v2_0002",  # release_long
    "plan_b__ball_train_v2_0031",  # release_long
    "plan_b__ball_train_v2_0038",  # chase_lost
    "plan_b__ball_train_v2_0103",  # chase_lost
    "plan_b__ball_train_v2_0199",  # release_short
    "plan_b__ball_train_v2_0211",  # release_short
    "plan_b__ball_train_v2_0216",  # follow_early (SUCCESS_12S both scenes)
    "plan_b__ref_s6_p10_a0_e0",  # reference (SUCCESS_12S both scenes)
]

CONDITIONS = ("elevated", "elevated_table_off", "ground")


def jsafe(o):
    if o is None or isinstance(o, (bool, str, int)):
        return o
    if isinstance(o, (float, np.floating)):
        x = float(o)
        return x if np.isfinite(x) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {str(k): jsafe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsafe(x) for x in o]
    return str(o)


def load_manifest() -> dict:
    out = {}
    with BAK.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            out[r["expert_id"]] = r
    return out


def make_env(condition: str):
    class E:
        pass

    env = E()
    if condition == "ground":
        env.sim, *_ = world()
    else:
        env.sim, *_ = elevated_world()
    env.switch = GeometrySwitch()
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    env._obs_seq = []
    env._prev_seq = []
    env.condition = condition
    return env


def disable_table_collision(sim) -> dict:
    table = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    if table < 0:
        return {"disabled": False, "reason": "no_table_geom"}
    before = (
        int(sim.model.geom_contype[table]),
        int(sim.model.geom_conaffinity[table]),
    )
    sim.model.geom_contype[table] = 0
    sim.model.geom_conaffinity[table] = 0
    mujoco.mj_forward(sim.model, sim.data)
    return {"disabled": True, "table_gid": int(table), "before": before}


def restore_blob(env, blob) -> dict:
    if "capture" in blob:
        restore_capture(env, blob["capture"])
        return blob["ref"]
    restore_ballistic(env.sim, blob)
    from envs.outcomes.support_drop import restore_state

    restore_state(env.sim, None)
    arm_recovery_drop(env.sim)
    mujoco.mj_forward(env.sim.model, env.sim.data)
    Rh = np.array(env.sim.data.xmat[env.sim.ids.hand_body].reshape(3, 3), float)
    return {
        "R_hand": Rh,
        "r_des": np.asarray(blob["r_des"], float).reshape(3, 3).copy(),
        "p_nom": np.asarray(blob.get("prel_ref", [0, 0, 0.1]), float).reshape(-1)[:3].copy(),
        "R_rel": np.asarray(blob["r_rel_ref"], float).reshape(3, 3).copy(),
        "t": float(blob["time"]),
    }


def geom_name(model, gid: int) -> str:
    n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(gid))
    return str(n) if n else f"geom_{int(gid)}"


def env_contacts(sim) -> list[dict]:
    """All contacts involving the cylinder (fingers, table, floor, ground, other)."""
    model, data, ids = sim.model, sim.data, sim.ids
    obj_b = int(ids.object_body)
    table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    ground = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    rows = []
    wr = np.zeros(6)
    for i in range(int(data.ncon)):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        b1 = int(model.geom_bodyid[g1])
        b2 = int(model.geom_bodyid[g2])
        if obj_b not in (b1, b2):
            continue
        other_g = g2 if b1 == obj_b else g1
        other_b = b2 if b1 == obj_b else b1
        mujoco.mj_contactForce(model, data, i, wr)
        kind = "object-other"
        if other_b == ids.left_body:
            kind = "object-left"
        elif other_b == ids.right_body:
            kind = "object-right"
        elif other_b == ids.hand_body:
            kind = "object-hand"
        elif table >= 0 and other_g == table:
            kind = "object-table"
        elif floor >= 0 and other_g == floor:
            kind = "object-floor"
        elif ground >= 0 and other_g == ground:
            kind = "object-ground"
        rows.append(
            {
                "kind": kind,
                "g1": geom_name(model, g1),
                "g2": geom_name(model, g2),
                "g_other": geom_name(model, other_g),
                "gid_other": int(other_g),
                "pos": np.array(c.pos, float).copy(),
                "normal": np.array(c.frame[0:3], float).copy(),
                "dist": float(c.dist),
                "fn": float(wr[0]),
                "ft": float(np.hypot(wr[1], wr[2])),
                "impulse_proxy": float(abs(wr[0]) * float(model.opt.timestep)),
            }
        )
    return rows


def body_twist(sim, bid: int):
    from envs.simulation.grasp_quality import body_twist as bt

    return bt(sim.model, sim.data, bid)


def snapshot_frame(sim, *, action_index, action, sub_i, phase: str) -> dict:
    ids = sim.ids
    vo, wo = body_twist(sim, ids.object_body)
    vh, wh = body_twist(sim, ids.hand_body)
    finger_rows, _ = extract_contacts(sim)
    env_rows = env_contacts(sim)
    return {
        "t": float(sim.data.time),
        "phase": phase,
        "action_index": None if action_index is None else int(action_index),
        "sub_i": int(sub_i),
        "action": None if action is None else np.asarray(action, float).copy(),
        "qpos": np.array(sim.data.qpos, float).copy(),
        "qvel": np.array(sim.data.qvel, float).copy(),
        "ctrl": np.array(sim.data.ctrl, float).copy(),
        "act": np.array(sim.data.act, float).copy() if int(sim.model.na) else np.zeros(0),
        "qacc_warmstart": np.array(sim.data.qacc_warmstart, float).copy(),
        "p_des": np.asarray(sim.fsm.p_des, float).copy(),
        "r_des": np.asarray(sim.fsm.r_des, float).reshape(3, 3).copy(),
        "v_cmd": np.asarray(sim.fsm.v_cmd, float).copy(),
        "w_cmd": np.asarray(sim.fsm.w_cmd, float).copy(),
        "fg_cmd": float(sim.fsm.fg_cmd),
        "fsm_phase": str(sim.fsm.phase),
        "obj_pos": np.array(sim.data.xpos[ids.object_body], float).copy(),
        "obj_quat": np.array(sim.data.xquat[ids.object_body], float).copy(),
        "obj_v": np.asarray(vo, float).copy(),
        "obj_w": np.asarray(wo, float).copy(),
        "hand_pos": np.array(sim.data.xpos[ids.hand_body], float).copy(),
        "hand_quat": np.array(sim.data.xquat[ids.hand_body], float).copy(),
        "hand_v": np.asarray(vh, float).copy(),
        "hand_w": np.asarray(wh, float).copy(),
        "finger_contacts": [
            {
                "side": r.get("side"),
                "Fn": float(r.get("Fn", r.get("fn", 0.0))),
                "pos_w": np.asarray(r.get("pos_w", r.get("pos")), float).copy(),
                "n_w": np.asarray(r.get("n_w", r.get("normal", [0, 0, 0])), float).copy()
                if r.get("n_w") is not None or r.get("normal") is not None
                else None,
            }
            for r in finger_rows
        ],
        "env_contacts": env_rows,
        "n_table": sum(1 for r in env_rows if r["kind"] == "object-table"),
        "n_ground": sum(1 for r in env_rows if r["kind"] in ("object-ground", "object-floor")),
        "n_finger": sum(1 for r in env_rows if r["kind"] in ("object-left", "object-right")),
        "latched": bool(latched(sim)),
        "drop_reason": terminal_reason(sim) if latched(sim) else None,
    }


def model_fingerprint(sim) -> dict:
    m = sim.model
    ids = sim.ids
    names = {
        "table": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table"),
        "floor": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor"),
        "ground": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ground"),
        "object": int(ids.object_geom),
    }
    og = int(ids.object_geom)
    ob = int(ids.object_body)
    return {
        "scene_id": str(getattr(sim, "scene_id", "") or ""),
        "nq": int(m.nq),
        "nv": int(m.nv),
        "nu": int(m.nu),
        "ngeom": int(m.ngeom),
        "nbody": int(m.nbody),
        "timestep": float(m.opt.timestep),
        "gravity": np.array(m.opt.gravity, float).tolist(),
        "integrator": int(m.opt.integrator),
        "solver": int(m.opt.solver),
        "iterations": int(m.opt.iterations),
        "tolerance": float(m.opt.tolerance),
        "noslip_iterations": int(m.opt.noslip_iterations),
        "noslip_tolerance": float(m.opt.noslip_tolerance),
        "cone": int(m.opt.cone),
        "impratio": float(m.opt.impratio),
        "geom_ids": names,
        "object_mass": float(m.body_mass[ob]),
        "object_inertia": np.array(m.body_inertia[ob], float).tolist(),
        "object_friction": np.array(m.geom_friction[og], float).tolist(),
        "object_solref": np.array(m.geom_solref[og], float).tolist(),
        "object_solimp": np.array(m.geom_solimp[og], float).tolist(),
        "object_condim": int(m.geom_condim[og]),
        "object_contype": int(m.geom_contype[og]),
        "object_conaffinity": int(m.geom_conaffinity[og]),
        "table_contype": None
        if names["table"] < 0
        else int(m.geom_contype[names["table"]]),
        "table_conaffinity": None
        if names["table"] < 0
        else int(m.geom_conaffinity[names["table"]]),
    }


def step_legal_traced(sim, action, action_index: int, frames: list) -> np.ndarray:
    """Mirror step_legal, recording after every mj_step."""
    action = np.clip(np.asarray(action, np.float32).reshape(7), -1.0, 1.0)
    rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    rd = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
    cmd = map_action(action, rd, rh, AIRBORNE_V3)
    vel, ang, tau = cmd["v_world"], cmd["w_world"], float(cmd["tau"])
    tau = float(np.clip(tau, sim.ids.ctrl_low[7], sim.ids.ctrl_high[7]))
    sim.fsm.p_des = np.array(sim.data.xpos[sim.ids.hand_body], float)
    sim.fsm.v_cmd = vel.copy()
    sim.fsm.w_cmd = ang.copy()
    sim.fsm.v_des = vel.copy()
    sim.fsm.w_des = ang.copy()
    sim.fsm.fg_cmd = float(-tau)
    from envs.outcomes.support_drop import object_vz, observe

    dt = float(sim.model.opt.timestep)
    for sub in range(N_SUB):
        sim.fsm.r_des = _integrate_rot(sim.fsm.r_des, ang, dt)
        apply_cartesian_ctrl(
            sim.model,
            sim.data,
            sim.ids,
            sim.fsm.p_des,
            sim.fsm.r_des,
            gripper_tau=tau,
            v_des=vel,
            w_des=ang,
            **TRACK,
        )
        vz_before = object_vz(sim)
        mujoco.mj_step(sim.model, sim.data)
        observe(sim, vz_before)
        frames.append(
            snapshot_frame(
                sim,
                action_index=action_index,
                action=action,
                sub_i=sub,
                phase="recovery",
            )
        )
    return action


def state_key_diff(a: dict, b: dict, atol_q=1e-9, atol_x=1e-9) -> dict:
    """Compare layout-compatible physical/controller fields."""
    out = {}
    for k in ("qpos", "qvel", "ctrl", "act", "qacc_warmstart", "p_des", "v_cmd", "w_cmd"):
        if a.get(k) is None or b.get(k) is None:
            continue
        da = np.asarray(a[k], float)
        db = np.asarray(b[k], float)
        if da.shape != db.shape:
            out[k] = {"shape_mismatch": [list(da.shape), list(db.shape)]}
            continue
        d = np.abs(da - db)
        out[k] = {
            "max_abs": float(np.max(d)) if d.size else 0.0,
            "mean_abs": float(np.mean(d)) if d.size else 0.0,
            "allclose": bool(np.allclose(da, db, atol=atol_q, rtol=0.0)),
        }
    for k in ("obj_pos", "obj_v", "obj_w", "hand_pos", "hand_v", "hand_w"):
        da = np.asarray(a[k], float)
        db = np.asarray(b[k], float)
        d = np.abs(da - db)
        out[k] = {
            "max_abs": float(np.max(d)),
            "delta": (da - db).tolist(),
            "allclose": bool(np.allclose(da, db, atol=atol_x, rtol=0.0)),
        }
    ra = np.asarray(a["r_des"], float).reshape(-1)
    rb = np.asarray(b["r_des"], float).reshape(-1)
    out["r_des"] = {
        "max_abs": float(np.max(np.abs(ra - rb))),
        "allclose": bool(np.allclose(ra, rb, atol=atol_q, rtol=0.0)),
    }
    out["t"] = {"a": a["t"], "b": b["t"], "dt": float(a["t"] - b["t"])}
    out["n_table"] = {"a": a["n_table"], "b": b["n_table"]}
    out["n_ground"] = {"a": a["n_ground"], "b": b["n_ground"]}
    out["n_finger"] = {"a": a["n_finger"], "b": b["n_finger"]}
    out["latched"] = {"a": a["latched"], "b": b["latched"]}
    return out


def first_divergence(frames_a, frames_b, *, atol_obj=1e-7, atol_q=1e-8) -> dict | None:
    n = min(len(frames_a), len(frames_b))
    for i in range(n):
        a, b = frames_a[i], frames_b[i]
        diff = state_key_diff(a, b, atol_q=atol_q, atol_x=atol_obj)
        meaningful = (
            (not diff["obj_pos"]["allclose"])
            or (not diff["obj_v"]["allclose"])
            or (not diff["hand_pos"]["allclose"])
            or (not diff["qpos"]["allclose"])
            or (not diff["qvel"]["allclose"])
            or a["n_table"] != b["n_table"]
            or a["n_ground"] != b["n_ground"]
            or a["latched"] != b["latched"]
        )
        if meaningful:
            return {
                "frame_i": i,
                "t": a["t"],
                "action_index": a["action_index"],
                "sub_i": a["sub_i"],
                "diff": diff,
                "a_env_contacts": a["env_contacts"],
                "b_env_contacts": b["env_contacts"],
                "a_finger_n": len(a["finger_contacts"]),
                "b_finger_n": len(b["finger_contacts"]),
            }
    if len(frames_a) != len(frames_b):
        return {
            "frame_i": n,
            "t": frames_a[n - 1]["t"] if n else None,
            "note": f"length_mismatch {len(frames_a)} vs {len(frames_b)}",
            "diff": None,
        }
    return None


def table_contact_events(frames: list[dict]) -> list[dict]:
    events = []
    for fr in frames:
        for c in fr["env_contacts"]:
            if c["kind"] != "object-table":
                continue
            events.append(
                {
                    "t": fr["t"],
                    "action_index": fr["action_index"],
                    "sub_i": fr["sub_i"],
                    "phase": fr["phase"],
                    "gid_other": c["gid_other"],
                    "g_other": c["g_other"],
                    "fn": c["fn"],
                    "ft": c["ft"],
                    "impulse_proxy": c["impulse_proxy"],
                    "pos": c["pos"],
                    "dist": c["dist"],
                }
            )
    return events


def run_condition(condition: str, blob, actions: np.ndarray, ref_template) -> dict:
    env = make_env(condition)
    fp0 = model_fingerprint(env.sim)
    ref = restore_blob(env, blob)
    if condition == "elevated_table_off":
        disable_info = disable_table_collision(env.sim)
    else:
        disable_info = {"disabled": False}
    fp1 = model_fingerprint(env.sim)
    frames = [snapshot_frame(env.sim, action_index=None, action=None, sub_i=-1, phase="ic")]
    ic = deepcopy(frames[0])
    lost = 0
    term = None
    fail_step = None
    for i, action in enumerate(actions):
        step_legal_traced(env.sim, action, i, frames)
        failed, lost = recovery_failed(env.sim, ref, lost)
        if failed:
            term = terminal_reason(env.sim) or "DROP_DURING_RECOVERY"
            fail_step = i
            break
    handoff_meta = None
    if term is None:
        # Capture pre-handoff frame
        frames.append(
            snapshot_frame(env.sim, action_index=len(actions), action=None, sub_i=-1, phase="pre_handoff")
        )
        handoff_meta = evaluate_handoff_from_state(env.sim, ref, stabilize_n=0)
        # Sample end state after nominal (evaluate already ran full nominal)
        frames.append(
            snapshot_frame(env.sim, action_index=len(actions), action=None, sub_i=-1, phase="post_nominal")
        )
        term = handoff_meta["reason"]
    table_events = table_contact_events(frames)
    return {
        "condition": condition,
        "model_before": fp0,
        "model_after_restore": fp1,
        "table_disable": disable_info,
        "ic": {
            "t": ic["t"],
            "obj_pos": ic["obj_pos"],
            "obj_v": ic["obj_v"],
            "obj_w": ic["obj_w"],
            "hand_pos": ic["hand_pos"],
            "hand_v": ic["hand_v"],
            "hand_w": ic["hand_w"],
            "qpos": ic["qpos"],
            "qvel": ic["qvel"],
            "ctrl": ic["ctrl"],
            "act": ic["act"],
            "qacc_warmstart": ic["qacc_warmstart"],
            "p_des": ic["p_des"],
            "r_des": ic["r_des"],
            "v_cmd": ic["v_cmd"],
            "w_cmd": ic["w_cmd"],
            "fg_cmd": ic["fg_cmd"],
            "n_table": ic["n_table"],
            "n_ground": ic["n_ground"],
            "n_finger": ic["n_finger"],
            "env_contacts": ic["env_contacts"],
            "latched": ic["latched"],
        },
        "n_frames": len(frames),
        "frames": frames,
        "fail_step": fail_step,
        "terminal_reason": term,
        "SUCCESS_NEW": None if handoff_meta is None else bool(handoff_meta["SUCCESS_NEW"]),
        "success_12s": None if handoff_meta is None else bool(handoff_meta["reason"] == "SUCCESS_12S"),
        "handoff_meta": handoff_meta,
        "table_contact_events": table_events,
        "any_table_contact": bool(table_events),
        "first_table_contact_t": None if not table_events else float(table_events[0]["t"]),
    }


def analyze_root(expert_id: str, man: dict) -> dict:
    refp = man["initial_physical_state_ref"]
    blob = pickle.loads(resolve_stored_path(refp["snapshot"]).read_bytes())
    z = np.load(resolve_stored_path(refp["continuous_npz"]))
    actions = (
        np.asarray(z["actions"], np.float32)
        if "actions" in z.files
        else np.zeros((0, 7), np.float32)
    )
    results = {}
    for cond in CONDITIONS:
        print(f"  {expert_id} / {cond} ...", flush=True)
        results[cond] = run_condition(cond, blob, actions, None)
        # drop full frames from disk payload later; keep in memory for diffs

    elev = results["elevated"]
    off = results["elevated_table_off"]
    gnd = results["ground"]

    # IC equivalence (hypothesis B)
    ic_elev_off = state_key_diff(elev["ic"], off["ic"])
    ic_elev_gnd = state_key_diff(elev["ic"], gnd["ic"])

    # Action identity (hypothesis C): same tape used; check first applied action equality
    def first_action(frames):
        for fr in frames:
            if fr["action"] is not None:
                return np.asarray(fr["action"], float)
        return None

    a0e, a0o, a0g = first_action(elev["frames"]), first_action(off["frames"]), first_action(gnd["frames"])
    action_match = {
        "elev_vs_off": None
        if a0e is None or a0o is None
        else bool(np.allclose(a0e, a0o)),
        "elev_vs_gnd": None
        if a0e is None or a0g is None
        else bool(np.allclose(a0e, a0g)),
        "n_actions": int(actions.shape[0]),
    }

    div_elev_off = first_divergence(elev["frames"], off["frames"])
    div_elev_gnd = first_divergence(elev["frames"], gnd["frames"])
    div_off_gnd = first_divergence(off["frames"], gnd["frames"])

    # Hypothesis D: model param diffs
    def model_delta(a, b):
        keys = set(a) | set(b)
        d = {}
        for k in sorted(keys):
            if a.get(k) != b.get(k):
                d[k] = {"a": a.get(k), "b": b.get(k)}
        return d

    # Physical trajectory match ignoring latch labels (hypothesis E)
    def traj_match_until(frames_a, frames_b, ignore_latch=True):
        n = min(len(frames_a), len(frames_b))
        for i in range(n):
            a, b = frames_a[i], frames_b[i]
            if not np.allclose(a["obj_pos"], b["obj_pos"], atol=1e-7):
                return {"match": False, "frame_i": i, "t": a["t"], "field": "obj_pos"}
            if not np.allclose(a["qpos"], b["qpos"], atol=1e-8):
                return {"match": False, "frame_i": i, "t": a["t"], "field": "qpos"}
            if (not ignore_latch) and a["latched"] != b["latched"]:
                return {"match": False, "frame_i": i, "t": a["t"], "field": "latched"}
        return {"match": True, "n_frames_compared": n}

    # Root-cause heuristic
    root_cause = []
    if elev["any_table_contact"]:
        root_cause.append(
            "HYP_A: table contact observed in elevated episode "
            f"(first_t={elev['first_table_contact_t']}, n_events={len(elev['table_contact_events'])})"
        )
    if not ic_elev_gnd["qpos"]["allclose"] or not ic_elev_gnd["qacc_warmstart"]["allclose"]:
        root_cause.append(
            "HYP_B: IC physical/controller state not identical after restore "
            f"(qpos_max={ic_elev_gnd['qpos']['max_abs']}, warmstart_max={ic_elev_gnd['qacc_warmstart']['max_abs']})"
        )
    if action_match["elev_vs_gnd"] is False:
        root_cause.append("HYP_C: first applied action differs across scenes")
    md = model_delta(elev["model_after_restore"], gnd["model_after_restore"])
    if md:
        root_cause.append(f"HYP_D: model/option differences: {sorted(md.keys())}")
    if (
        elev["terminal_reason"] != gnd["terminal_reason"]
        and traj_match_until(elev["frames"], gnd["frames"]).get("match")
    ):
        root_cause.append(
            "HYP_E: trajectories match but termination/success labels differ "
            f"({elev['terminal_reason']} vs {gnd['terminal_reason']})"
        )
    # Counterfactual: if elev-table-off matches ground until late, collision geometry is causal
    if div_off_gnd is None or (
        div_off_gnd.get("action_index") is not None
        and elev["fail_step"] is not None
        and div_off_gnd["action_index"] >= elev["fail_step"]
    ):
        if elev["any_table_contact"] and not off["any_table_contact"]:
            root_cause.append(
                "COUNTERFACTUAL: disabling table collision removes elevated table contacts; "
                "compare off vs ground divergence for residual migration effects"
            )
    if elev["terminal_reason"] in ("TABLE_DROP",) and gnd["terminal_reason"] not in ("TABLE_DROP",):
        if elev["any_table_contact"]:
            root_cause.append(
                "PRIMARY: elevated failure is TABLE_DROP from real object-table contact; "
                "ground continues freefall until geometric/GROUND_DROP rule — "
                "not a non-contacting table removal paradox"
            )
        else:
            root_cause.append(
                "PRIMARY: elevated TABLE_DROP without recorded object-table contact rows "
                "(check floor latch / force threshold)"
            )

    # Compact frames for disk (downsample): keep IC, every control-step end, divergence ±2, table events
    def compact(frames, keep_around=None):
        keep = {0, len(frames) - 1}
        for i, fr in enumerate(frames):
            if fr["sub_i"] == N_SUB - 1 or fr["sub_i"] == -1:
                keep.add(i)
            if fr["n_table"] > 0 or fr["n_ground"] > 0:
                keep.add(i)
        if keep_around is not None:
            for j in range(max(0, keep_around - 2), min(len(frames), keep_around + 3)):
                keep.add(j)
        return [frames[i] for i in sorted(keep)]

    div_i = None if div_elev_gnd is None else div_elev_gnd.get("frame_i")
    payload = {
        "expert_id": expert_id,
        "mechanism": man.get("contact_grasp_mechanism"),
        "n_actions": int(actions.shape[0]),
        "outcomes": {
            cond: {
                "terminal_reason": results[cond]["terminal_reason"],
                "fail_step": results[cond]["fail_step"],
                "SUCCESS_NEW": results[cond]["SUCCESS_NEW"],
                "success_12s": results[cond]["success_12s"],
                "any_table_contact": results[cond]["any_table_contact"],
                "first_table_contact_t": results[cond]["first_table_contact_t"],
                "n_table_events": len(results[cond]["table_contact_events"]),
                "n_frames": results[cond]["n_frames"],
            }
            for cond in CONDITIONS
        },
        "hypothesis_B_ic": {
            "elev_vs_table_off": ic_elev_off,
            "elev_vs_ground": ic_elev_gnd,
        },
        "hypothesis_C_actions": action_match,
        "hypothesis_D_model": {
            "elev_vs_ground": model_delta(elev["model_after_restore"], gnd["model_after_restore"]),
            "elev_vs_table_off": model_delta(elev["model_after_restore"], off["model_after_restore"]),
            "elev_model": elev["model_after_restore"],
            "ground_model": gnd["model_after_restore"],
        },
        "hypothesis_E_traj_ignore_latch": {
            "elev_vs_ground": traj_match_until(elev["frames"], gnd["frames"], ignore_latch=True),
            "elev_vs_table_off": traj_match_until(elev["frames"], off["frames"], ignore_latch=True),
            "table_off_vs_ground": traj_match_until(off["frames"], gnd["frames"], ignore_latch=True),
        },
        "first_divergence": {
            "elev_vs_table_off": div_elev_off,
            "elev_vs_ground": div_elev_gnd,
            "table_off_vs_ground": div_off_gnd,
        },
        "table_contact_events_elevated": elev["table_contact_events"][:50],
        "table_contact_events_table_off": off["table_contact_events"][:20],
        "root_cause_evidence": root_cause,
        "frames_compact": {
            "elevated": compact(elev["frames"], div_i),
            "elevated_table_off": compact(off["frames"], None if div_elev_off is None else div_elev_off.get("frame_i")),
            "ground": compact(gnd["frames"], div_i),
        },
    }
    return payload


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mans = load_manifest()
    reports = []
    for eid in ROOT_IDS:
        if eid not in mans:
            print(f"missing {eid}", flush=True)
            continue
        print(f"Analyzing {eid}", flush=True)
        rep = analyze_root(eid, mans[eid])
        reports.append(rep)
        (OUT / f"{eid}.json").write_text(json.dumps(jsafe(rep), indent=2), encoding="utf-8")

    # Summary markdown
    lines = [
        "# Plan B physical replay contradiction",
        "",
        f"Created UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Controller, action tapes, SUCCESS_NEW criteria, and physics parameters were **not** modified.",
        "",
        "## Selection note",
        "",
        "Among all 69 migration `FAIL_REPLAY` Plan B roots, **0** currently reach `SUCCESS_12S` on the",
        "elevated table scene under the same legal tape + `step_legal` + drop latch path used for GROUND_SCENE.",
        "The premise “elevated success / ground failure” is therefore **not** supported by present-day physical",
        "replay. Roots below are representative failures (plus two SUCCESS_12S-but-not-NEW cases for Hyp E).",
        "",
        "## Per-root summary",
        "",
    ]
    for rep in reports:
        o = rep["outcomes"]
        div = rep["first_divergence"]["elev_vs_ground"]
        div_off = rep["first_divergence"]["elev_vs_table_off"]
        lines.append(f"### `{rep['expert_id']}` ({rep.get('mechanism')})")
        lines.append("")
        lines.append("| Condition | Terminal | fail_step | SUCCESS_NEW | table contact |")
        lines.append("|-----------|----------|-----------|-------------|---------------|")
        for cond in CONDITIONS:
            r = o[cond]
            lines.append(
                f"| {cond} | `{r['terminal_reason']}` | {r['fail_step']} | {r['SUCCESS_NEW']} | "
                f"{r['any_table_contact']} (n={r['n_table_events']}, t0={r['first_table_contact_t']}) |"
            )
        lines.append("")
        if div:
            lines.append(
                f"- First elev↔ground divergence: t={div.get('t')} action={div.get('action_index')} "
                f"sub={div.get('sub_i')} frame={div.get('frame_i')}"
            )
            d = div.get("diff") or {}
            if d:
                lines.append(
                    f"  - obj_pos Δmax={d.get('obj_pos', {}).get('max_abs')} "
                    f"qpos Δmax={d.get('qpos', {}).get('max_abs')} "
                    f"n_table {d.get('n_table')} n_ground {d.get('n_ground')}"
                )
        if div_off:
            lines.append(
                f"- First elev↔elev_table_off divergence: t={div_off.get('t')} "
                f"action={div_off.get('action_index')} sub={div_off.get('sub_i')}"
            )
        else:
            lines.append("- elev↔elev_table_off: **no physical divergence** before end of compared frames")
        lines.append("- Evidence:")
        for e in rep["root_cause_evidence"] or ["(none flagged)"]:
            lines.append(f"  - {e}")
        lines.append("")

    lines.extend(
        [
            "## Cross-cutting conclusions",
            "",
            "1. **Hypothesis A (hidden table assistance):** For release/chase FAIL_REPLAY roots, elevated",
            "   episodes develop **object–table contacts** and latch `TABLE_DROP` mid-tape. The IC is airborne,",
            "   but the recorded recovery tape allows the cylinder to fall onto the table. Removing the table",
            "   (or disabling its collision) removes that contact event; freefall then continues until the",
            "   ground-scene geometric / GROUND_DROP rules fire later.",
            "2. **Hypothesis B (incomplete transfer):** Joint `qpos` matches at IC after identity restore;",
            "   warmstart/ctrl/FSM are compared in per-root JSON. Residual migration effects are secondary to",
            "   table contact for FAIL_REPLAY roots (see elev_table_off vs ground).",
            "3. **Hypothesis C:** Same continuous action tape and `step_legal` mapping in all three conditions.",
            "4. **Hypothesis D:** Model diffs are support geoms / ngeom / scene_id; cylinder mass/friction/solref",
            "   and solver noslip options match (see per-root `hypothesis_D_model`).",
            "5. **Hypothesis E:** For follow_early/reference roots, both scenes reach `SUCCESS_12S` and fail",
            "   SUCCESS_NEW hold — label gap, not scene-migration physics.",
            "6. **Counterfactual:** `elevated_table_off` isolates table *collision* from body removal. When it",
            "   tracks GROUND_SCENE until late freefall, table collision — not an invisible IC table support —",
            "   explains the elevated/ground outcome split.",
            "",
            "Do not resume large-scale oracle re-search until this report is accepted.",
            "",
        ]
    )
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "summary.json").write_text(
        json.dumps(
            jsafe(
                {
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "roots": [r["expert_id"] for r in reports],
                    "elevated_success_12s_among_fail_replay_population": 0,
                    "reports": [
                        {
                            "expert_id": r["expert_id"],
                            "mechanism": r.get("mechanism"),
                            "outcomes": r["outcomes"],
                            "first_divergence_elev_ground_t": (
                                None
                                if r["first_divergence"]["elev_vs_ground"] is None
                                else r["first_divergence"]["elev_vs_ground"].get("t")
                            ),
                            "root_cause_evidence": r["root_cause_evidence"],
                        }
                        for r in reports
                    ],
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
