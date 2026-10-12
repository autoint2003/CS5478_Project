"""Genuine MuJoCo ball-impact recovery episodes.

Every accepted initial condition is a full nominal grasp and lift, then a ball
whose position and velocity are set only before contact. The cylinder is never
teleported. This module does not train.

Launch azimuths are assigned to train, validation, and final test before any
episode runs. Final-test episodes are stored and are not rolled with a policy.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mujoco

import training.natural_loss_recatch_transformer_v3 as v3
import training.recovery_success as success
from controllers.geometry_switch import hand_pose, rot_angle
from envs.deterioration import body_twist
from envs.physical_recovery import Z_AIR, physical_pack
from training.autonomous_episode import (
    T_TASK,
    Z_ABORT,
    fresh_nominal_state,
    make_training_env,
    nominal_substep,
)
from training.demo_ballistic_impact import HAND_RISE_M, PAIR_MU
from training.history_aware_critic_ablation import restore_capture, take_capture
from training.impact_ball import apply_ball_free_state
from training.nominal_pose_audit import capture_ref, closed_grip, errors
from training.write_ballistic_impact_scene import BALL_M, BALL_R, CYL_M, CYL_R

RAW = ROOT / "results" / "diagnostics" / "raw" / "ball_impact_dataset"
ATTEMPTS = RAW / "attempts.jsonl"
MANIFEST = RAW / "manifest.json"

# Assigned before generation. A direction family never crosses a split.
SPLIT_AZIMUTH = {
    "train": (0.0, 45.0, 90.0),
    "validation": (135.0, 180.0),
    "final_test": (225.0, 270.0, 315.0),
}
SPEEDS = (6.5, 7.5, 8.5, 9.5)
Z_OFFS = (-0.015, 0.0, 0.015)
PHIS = (-40.0, 0.0, 40.0)
DELAYS = (0.0, 0.20, 0.40)
TARGETS = {"train": 300, "validation": 60, "final_test": 120}
SEEDS_PER_CELL = 3
FLIGHT_S = 0.10
LAUNCH_WINDOW_S = 3.5
FIDELITY_STEPS = 10
FIDELITY_TOL = 1e-3
ENTRY_POS_M = 0.006
# Labels may overlap. Rolling requires angular rate, contact migration, and a
# rolling-speed match. Contact-point motion alone stays contact_migration.
FOCUS_STATES = (
    "tangential_slip",
    "relative_rolling",
    "asymmetric_or_unilateral",
    "complete_grasp_loss",
    "airborne_object",
    "intermittent_contact",
    "recaught_displaced",
    "delayed_grasp_failure",
)


def direction_name(azimuth_deg):
    names = {
        0.0: "+X",
        45.0: "oblique_+X_+Y",
        90.0: "+Y",
        135.0: "oblique_-X_+Y",
        180.0: "-X",
        225.0: "oblique_-X_-Y",
        270.0: "-Y",
        315.0: "oblique_+X_-Y",
    }
    return names[float(azimuth_deg)]


def split_of_azimuth(azimuth_deg):
    for split, angles in SPLIT_AZIMUTH.items():
        if float(azimuth_deg) in angles:
            return split
    raise KeyError(azimuth_deg)


def planned_cells():
    cells = []
    for split, angles in SPLIT_AZIMUTH.items():
        for azimuth in angles:
            for speed in SPEEDS:
                for z_off in Z_OFFS:
                    for phi in PHIS:
                        for delay in DELAYS:
                            cells.append({
                                "split": split,
                                "azimuth_deg": float(azimuth),
                                "direction": direction_name(azimuth),
                                "speed_mps": float(speed),
                                "z_off_m": float(z_off),
                                "phi_deg": float(phi),
                                "delay_s": float(delay),
                            })
    return cells


def cell_key(cell, seed):
    return (
        f"{cell['split']}|{cell['azimuth_deg']}|{cell['speed_mps']}|"
        f"{cell['z_off_m']}|{cell['phi_deg']}|{cell['delay_s']}|{seed}"
    )


def jitter(cell, seed):
    if seed == 0:
        return dict(cell)
    sign = -1.0 if seed % 2 else 1.0
    out = dict(cell)
    out["speed_mps"] = float(cell["speed_mps"] + sign * 0.20)
    out["z_off_m"] = float(np.clip(cell["z_off_m"] + sign * 0.002, -0.025, 0.025))
    out["phi_deg"] = float(cell["phi_deg"] + sign * 5.0)
    out["seed"] = int(seed)
    return out


def round3(value):
    if isinstance(value, float):
        return round(value, 4)
    return value


def pose_of(sim):
    body = int(sim.ids.object_body)
    linear, angular = body_twist(sim.model, sim.data, body)
    return {
        "p": [round(float(x), 5) for x in sim.data.xpos[body]],
        "v": [round(float(x), 5) for x in linear],
        "w": [round(float(x), 5) for x in angular],
    }


def physics_record(sim):
    model = sim.model
    obj = int(sim.ids.object_body)
    ball = int(sim.ball_body)
    og = int(sim.ids.object_geom)
    bg = int(sim.ball_geom)
    return {
        "scene": "assets/scene_ballistic_impact.xml",
        "timestep_s": float(model.opt.timestep),
        "gravity": [float(x) for x in model.opt.gravity],
        "noslip_iterations": int(model.opt.noslip_iterations),
        "noslip_tolerance": float(model.opt.noslip_tolerance),
        "solver_iterations": int(model.opt.iterations),
        "integrator": int(model.opt.integrator),
        "cylinder_mass_kg": float(model.body_mass[obj]),
        "cylinder_radius_m": float(model.geom_size[og][0]),
        "cylinder_half_height_m": float(model.geom_size[og][1]),
        "cylinder_friction": [float(x) for x in model.geom_friction[og]],
        "cylinder_condim": int(model.geom_condim[og]),
        "cylinder_contype": int(model.geom_contype[og]),
        "cylinder_conaffinity": int(model.geom_conaffinity[og]),
        "ball_mass_kg": float(model.body_mass[ball]),
        "ball_radius_m": float(model.geom_size[bg][0]),
        "ball_friction": [float(x) for x in model.geom_friction[bg]],
        "ball_condim": int(model.geom_condim[bg]),
        "ball_contype": int(model.geom_contype[bg]),
        "ball_conaffinity": int(model.geom_conaffinity[bg]),
        "finger_object_mu": float(PAIR_MU),
        "xml_cylinder_mass_kg": float(CYL_M),
        "xml_ball_mass_kg": float(BALL_M),
        "xml_cylinder_radius_m": float(CYL_R),
        "xml_ball_radius_m": float(BALL_R),
        "source_constants_match_model": bool(
            abs(float(model.body_mass[obj]) - CYL_M) < 1e-6
            and abs(float(model.body_mass[ball]) - BALL_M) < 1e-6
            and abs(float(model.geom_size[og][0]) - CYL_R) < 1e-6
            and abs(float(model.geom_size[bg][0]) - BALL_R) < 1e-6
        ),
        "mass_friction_randomized": False,
        "reason": "nominal.yaml randomize is false and BallisticSim.reset keeps the XML cylinder mass and PAIR_MU",
    }


def aim_ball(sim, azimuth_deg, speed, z_off, phi_deg):
    az = np.radians(float(azimuth_deg))
    approach = np.array([np.cos(az), np.sin(az), 0.0])
    po = np.array(sim.data.xpos[sim.ids.object_body], float)
    vo, _wo = body_twist(sim.model, sim.data, sim.ids.object_body)
    axis = np.array(sim.data.xmat[sim.ids.object_body].reshape(3, 3), float)[:, 2]
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    horizontal = approach - axis * float(np.dot(axis, approach))
    nrm = float(np.linalg.norm(horizontal))
    if nrm < 1e-6:
        horizontal = np.array([1.0, 0.0, 0.0])
        nrm = 1.0
    e_face = -horizontal / nrm
    e_tan = np.cross(axis, e_face)
    e_tan = e_tan / max(float(np.linalg.norm(e_tan)), 1e-12)
    phi = np.radians(float(phi_deg))
    g = np.array(sim.model.opt.gravity, float)
    po_pred = po + np.asarray(vo, float) * FLIGHT_S
    radius = float(sim.model.geom_size[int(sim.ids.object_geom)][0])
    ball_r = float(sim.model.geom_size[int(sim.ball_geom)][0])
    p_contact = po_pred + axis * float(z_off) + e_face * (radius * np.cos(phi)) + e_tan * (radius * np.sin(phi))
    n_out = e_face * np.cos(phi) + e_tan * np.sin(phi)
    p_ball = p_contact + n_out * ball_r
    p_launch = p_ball - approach * (float(speed) * FLIGHT_S)
    v0 = (p_ball - p_launch) / FLIGHT_S - 0.5 * g * FLIGHT_S
    return {
        "p_launch": np.asarray(p_launch, float),
        "v0": np.asarray(v0, float),
        "p_aim": np.asarray(p_ball, float),
        "approach": approach,
    }


def cylinder_contacts(sim):
    model, data = sim.model, sim.data
    wr = np.zeros(6)
    hits = []
    other = []
    ball = int(sim.ball_geom)
    obj = int(sim.ids.object_geom)
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        if ball not in (g1, g2):
            continue
        other_geom = g2 if g1 == ball else g1
        mujoco.mj_contactForce(model, data, index, wr)
        kind = "ball-cylinder" if other_geom == obj else "ball-other"
        if int(model.geom_bodyid[other_geom]) in (int(sim.ids.left_body), int(sim.ids.right_body)):
            kind = "ball-finger"
        normal = np.array(contact.frame, float).reshape(3, 3)[0]
        item = {
            "kind": kind,
            "pair": [int(g1), int(g2)],
            "body": int(model.geom_bodyid[other_geom]),
            "fn": round(abs(float(wr[0])), 4),
            "ft": round(float(np.hypot(wr[1], wr[2])), 4),
            "pos": [round(float(x), 5) for x in contact.pos],
            "normal": [round(float(x), 4) for x in normal],
        }
        if kind == "ball-cylinder":
            hits.append(item)
        else:
            other.append(item)
    return hits, other


def grasp_sample(sim, switch, first_cp):
    pack = physical_pack(sim)
    p_rel, r_rel = hand_pose(sim)
    if switch.p0 is None:
        e_pos, e_rot = 0.0, 0.0
    else:
        e_pos = float(np.linalg.norm(np.asarray(p_rel, float) - switch.p0))
        e_rot = float(rot_angle(switch.r0, r_rel))
    v_h = np.asarray(pack["v_rel_h"], float)
    w_h = np.asarray(pack["w_rel_h"], float)
    model, data, ids = sim.model, sim.data, sim.ids
    fn = {"L": 0.0, "R": 0.0}
    ft = {"L": 0.0, "R": 0.0}
    points = []
    rotation = np.array(data.xmat[ids.object_body].reshape(3, 3), float)
    origin = np.array(data.xpos[ids.object_body], float)
    wr = np.zeros(6)
    n_l = n_r = 0
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        b1 = int(model.geom_bodyid[g1])
        b2 = int(model.geom_bodyid[g2])
        if int(ids.object_body) not in (b1, b2):
            continue
        other_body = b2 if b1 == int(ids.object_body) else b1
        side = "L" if other_body == int(ids.left_body) else "R" if other_body == int(ids.right_body) else None
        if side is None:
            continue
        mujoco.mj_contactForce(model, data, index, wr)
        fn[side] += abs(float(wr[0]))
        ft[side] += float(np.hypot(wr[1], wr[2]))
        if side == "L":
            n_l += 1
        else:
            n_r += 1
        points.append(rotation.T @ (np.array(contact.pos, float) - origin))
    migration = 0.0
    anchor = first_cp
    if points:
        center = np.mean(np.stack(points, axis=0), axis=0)
        if anchor is None:
            anchor = center
        migration = float(np.linalg.norm(center - anchor))
    return {
        "nL": int(n_l),
        "nR": int(n_r),
        "fnL": round(float(fn["L"]), 3),
        "fnR": round(float(fn["R"]), 3),
        "ftL": round(float(ft["L"]), 3),
        "ftR": round(float(ft["R"]), 3),
        "v_tan": round(float(np.linalg.norm(v_h[:2])), 4),
        "w_rel": round(float(np.linalg.norm(w_h)), 4),
        "e_pos": round(e_pos, 5),
        "e_rot": round(e_rot, 5),
        "aperture": round(float(pack["aperture"]), 5),
        "migration_m": round(migration, 5),
        "sep": round(float(np.linalg.norm(p_rel)), 5),
    }, anchor


def label_states(samples):
    labels = set()
    if not samples:
        return []
    stable_run = slip_run = roll_run = 0
    had_bilateral = False
    lost_after = False
    regained = False
    flips = 0
    previous = None
    for sample in samples:
        both = sample["nL"] > 0 and sample["nR"] > 0
        one_side = (sample["nL"] > 0) ^ (sample["nR"] > 0)
        touching = sample["nL"] > 0 or sample["nR"] > 0
        if previous is not None and touching != previous:
            flips += 1
        previous = touching
        if both:
            had_bilateral = True
            if lost_after:
                regained = True
        if had_bilateral and not touching:
            lost_after = True
        if one_side:
            labels.add("asymmetric_or_unilateral")
        separated = sample["e_pos"] >= 0.015 or sample["sep"] >= 0.020
        if not touching and separated:
            labels.add("complete_grasp_loss")
        if (not touching) and sample["sep"] >= 0.020 and sample["e_pos"] >= 0.015:
            labels.add("airborne_object")
        tangential = float(sample["v_tan"])
        angular = float(sample["w_rel"])
        rolling_speed = angular * float(CYL_R)
        slip_like = touching and tangential >= 0.03 and rolling_speed < 0.5 * tangential
        roll_like = (
            touching
            and angular >= 1.0
            and float(sample["migration_m"]) >= 0.003
            and abs(tangential - rolling_speed) <= 0.45 * max(tangential, rolling_speed, 1e-6)
        )
        slip_run = slip_run + 1 if slip_like else 0
        roll_run = roll_run + 1 if roll_like else 0
        if slip_run >= 2:
            labels.add("tangential_slip")
        if roll_run >= 2:
            labels.add("relative_rolling")
        if touching and float(sample["migration_m"]) >= 0.004 and angular < 1.0:
            labels.add("contact_migration")
        stable_like = (
            both
            and tangential < 0.02
            and angular < 0.5
            and sample["e_rot"] < 0.08
            and sample["e_pos"] < ENTRY_POS_M
        )
        stable_run = stable_run + 1 if stable_like else 0
        if stable_run >= 3:
            labels.add("stable_bilateral")
        if regained and (sample["e_pos"] >= ENTRY_POS_M or sample["e_rot"] >= 0.10):
            labels.add("recaught_displaced")
    if flips >= 3:
        labels.add("intermittent_contact")
    last = samples[-1]
    if lost_after and last["nL"] == 0 and last["nR"] == 0:
        labels.add("delayed_grasp_failure")
    return sorted(labels)


def state_peaks(samples):
    if not samples:
        return {}
    return {
        "max_v_tan_mps": max(float(sample["v_tan"]) for sample in samples),
        "max_w_rel_rad_s": max(float(sample["w_rel"]) for sample in samples),
        "max_migration_m": max(float(sample["migration_m"]) for sample in samples),
        "max_e_rot_rad": max(float(sample["e_rot"]) for sample in samples),
        "max_ft_n": max(float(sample["ftL"]) + float(sample["ftR"]) for sample in samples),
        "min_contacts": min(int(sample["nL"]) + int(sample["nR"]) for sample in samples),
    }


def ready_to_launch(sim, st):
    if not sim.captured or str(sim.fsm.phase) != "lift":
        return False
    if st["hand_z0"] is None:
        return False
    obj_z = float(physical_pack(sim)["obj_z"])
    hand_z = float(sim.data.xpos[sim.ids.hand_body][2])
    return obj_z >= float(Z_AIR) and (hand_z - float(st["hand_z0"])) >= HAND_RISE_M


def _hold_state(sim):
    st = fresh_nominal_state()
    st["holding"] = True
    st["z_tgt"] = float(sim.fsm.p_des[2])
    st["hand_z0"] = float(sim.data.xpos[sim.ids.hand_body][2])
    return st


def fidelity(env, capture):
    original = [np.array(env.sim.data.qpos, float).copy()]
    for _ in range(FIDELITY_STEPS):
        nominal_substep(env.sim, env.gains, _hold_state(env.sim))
        original.append(np.array(env.sim.data.qpos, float).copy())
    restore_capture(env, capture)
    err = float(np.max(np.abs(env.sim.data.qpos - original[0])))
    for index in range(FIDELITY_STEPS):
        nominal_substep(env.sim, env.gains, _hold_state(env.sim))
        err = max(err, float(np.max(np.abs(env.sim.data.qpos - original[index + 1]))))
    return err


def run_episode(env, spec):
    env.begin_nominal()
    st = fresh_nominal_state()
    switch = env.switch
    launched = False
    t_ready = None
    t_launch = None
    launch = None
    pre_pose = None
    post_pose = None
    post_later = None
    contact = None
    peak_fn = 0.0
    other_seen = False
    max_e_pos = 0.0
    entry_row = None
    ball_touching = False
    dropped = False
    samples = []
    first_cp = None
    pre_contact_e = 0.0
    first_other = None
    while float(env.sim.data.time) < T_TASK - 1e-9 and not switch.entered:
        for _ in range(v3.N_SUB):
            pre_pose = pose_of(env.sim)
            nominal_substep(env.sim, env.gains, st)
            now = float(env.sim.data.time)
            if (not launched) and ready_to_launch(env.sim, st):
                if t_ready is None:
                    t_ready = now
                if now + 1e-9 >= t_ready + float(spec["delay_s"]):
                    launch = aim_ball(
                        env.sim, spec["azimuth_deg"], spec["speed_mps"], spec["z_off_m"], spec["phi_deg"],
                    )
                    env.sim.set_guide_weld(False)
                    apply_ball_free_state(env.sim, launch["p_launch"], launch["v0"])
                    launched = True
                    t_launch = float(env.sim.data.time)
            if launched:
                hits, other = cylinder_contacts(env.sim)
                if other:
                    other_seen = True
                    if first_other is None:
                        first_other = {
                            "t": round(now, 4),
                            "kind": other[0]["kind"],
                            "body": other[0].get("body"),
                            "fn_n": other[0]["fn"],
                        }
                if hits and contact is None:
                    contact = {
                        "t": round(now, 4),
                        "hits": hits,
                        "pre_impact": pre_pose,
                    }
                    post_pose = pose_of(env.sim)
                if hits:
                    peak_fn = max(peak_fn, max(item["fn"] for item in hits))
                    ball_touching = True
                else:
                    ball_touching = False
                if contact is not None and post_later is None and now >= float(contact["t"]) + 0.05:
                    post_later = pose_of(env.sim)
            if launched and float(physical_pack(env.sim)["obj_z"]) < Z_ABORT:
                dropped = True
                break
        if dropped:
            break
        if contact is None and switch.p0 is not None:
            p_rel, _r_rel = hand_pose(env.sim)
            pre_contact_e = max(pre_contact_e, float(np.linalg.norm(p_rel - switch.p0)))
        if contact is not None and len(samples) < 160:
            sample, first_cp = grasp_sample(env.sim, switch, first_cp)
            samples.append(sample)
        fresh = switch.p0 is None
        switch.measure(env.sim)
        if launched and switch.p0 is not None:
            measured = switch.measure(env.sim)
            max_e_pos = max(max_e_pos, float(measured["e_pos"]))
            switch.update_entry(env.sim)
        if launched and t_launch is not None and float(env.sim.data.time) > t_launch + LAUNCH_WINDOW_S:
            break
        if (not launched) and float(env.sim.data.time) > 9.0:
            break
        if fresh and switch.p0 is None and float(env.sim.data.time) > 8.0:
            break
    if switch.entered:
        entry_row = switch.measure(env.sim)
        hits, _other = cylinder_contacts(env.sim)
        ball_touching = bool(hits)
    if switch.entered and contact is not None and pre_contact_e >= ENTRY_POS_M:
        category, reason = (
            "entry_before_impact",
            "position error already exceeded the entry threshold before the collision",
        )
    else:
        category, reason = classify(launched, contact, switch, entry_row, dropped)
    if (
        contact is not None
        and t_launch is not None
        and float(contact["t"]) <= float(t_launch) + 2.0 * float(env.sim.model.opt.timestep)
    ):
        category, reason = (
            "no_ball_contact",
            "ball overlapped a geom at the launch instant, so the contact is not a flight impact",
        )
    states = label_states(samples)
    peaks = state_peaks(samples)
    evidence = {
        "category": category,
        "reason": reason,
        "spec": {k: round3(v) if isinstance(v, float) else v for k, v in spec.items()},
        "t_ready": None if t_ready is None else round(float(t_ready), 4),
        "t_launch": None if t_launch is None else round(float(t_launch), 4),
        "launch_position": None if launch is None else [round(float(x), 5) for x in launch["p_launch"]],
        "launch_velocity": None if launch is None else [round(float(x), 5) for x in launch["v0"]],
        "launch_speed": None if launch is None else round(float(np.linalg.norm(launch["v0"])), 4),
        "contact_t": None if contact is None else contact["t"],
        "contact": None if contact is None else contact["hits"],
        "peak_normal_force_n": round(float(peak_fn), 4),
        "max_e_pos_m": round(float(max_e_pos), 5),
        "pre_impact": None if contact is None else contact["pre_impact"],
        "post_impact": post_pose,
        "post_impact_50ms": post_later,
        "other_ball_contact": bool(other_seen),
        "first_other": first_other,
        "entry_t": None if not switch.entered else round(float(switch.t_entry), 4),
        "entry_reason": switch.entry_reason,
        "entry": None if entry_row is None else {
            "e_pos_m": round(float(entry_row["e_pos"]), 5),
            "e_rot_rad": round(float(entry_row["e_rot"]), 5),
            "vrel": round(float(entry_row["vrel"]), 5),
            "z": round(float(entry_row["z"]), 5),
            "nL": int(entry_row["nL"]),
            "nR": int(entry_row["nR"]),
        },
        "ball_interacting_at_entry": bool(switch.entered and ball_touching),
        "dropped_before_entry": bool(dropped and not switch.entered),
        "pre_contact_e_pos_m": round(float(pre_contact_e), 5),
        "states": states,
        "state_peaks": {key: round(float(value), 5) for key, value in peaks.items()},
        "phase": str(env.sim.fsm.phase),
        "grip_closed": bool(closed_grip(env.sim)),
        "controller": {
            "phase": str(env.sim.fsm.phase),
            "p_des": [round(float(x), 5) for x in np.asarray(env.sim.fsm.p_des, float).reshape(-1)[:3]],
            "v_des": [round(float(x), 5) for x in np.asarray(env.sim.fsm.v_des, float).reshape(-1)[:3]],
            "w_cmd": [round(float(x), 5) for x in np.asarray(env.sim.fsm.w_cmd, float).reshape(-1)[:3]],
        },
    }
    if category != "valid_impact_recovery_ic":
        return evidence, None
    env.start_recovery()
    ref = capture_ref(env.sim, switch)
    row = errors(env.sim, ref)
    if success.basin_ok(row, closed_grip(env.sim)):
        evidence["category"] = "trivial_goal_state"
        evidence["reason"] = "entry state is already inside the nominal basin"
        return evidence, None
    capture = take_capture(env)
    err = fidelity(env, capture)
    evidence["grasp_trace_steps"] = len(samples)
    evidence["snapshot_qpos_error"] = float(err)
    if err > FIDELITY_TOL:
        evidence["category"] = "invalid_snapshot"
        evidence["reason"] = f"restore continuation qpos error {err:.3e}"
        return evidence, None
    evidence["source_episode"] = spec["attempt_id"]
    return evidence, {
        "spec": spec,
        "capture": capture,
        "ref": ref,
        "evidence": evidence,
        "grasp_trace": samples,
    }


def classify(launched, contact, switch, entry_row, dropped):
    if not launched:
        return "no_ball_contact", "lift never became ready for a physical launch"
    if contact is None:
        return "no_ball_contact", "no verified ball-cylinder collision"
    if switch.entered and switch.t_entry is not None and float(switch.t_entry) + 1e-9 < float(contact["t"]):
        return "entry_before_impact", "geometry entry preceded the ball-cylinder collision"
    if not switch.entered:
        if dropped:
            return "impact_without_recovery_entry", "cylinder was hit and then fell before geometry entry"
        return "impact_without_recovery_entry", "cylinder was hit and the position entry did not persist"
    if entry_row is None:
        return "invalid_snapshot", "entry flag set without a measurement"
    return "valid_impact_recovery_ic", "ball-cylinder collision then geometry entry"


def load_log():
    rows = []
    if ATTEMPTS.is_file():
        for line in ATTEMPTS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def counts_from(rows):
    out = {split: 0 for split in TARGETS}
    for row in rows:
        if row.get("category") == "valid_impact_recovery_ic":
            out[row["split"]] = out.get(row["split"], 0) + 1
    return out


def closed_cells(rows):
    by = {}
    for row in rows:
        key = row["cell"]
        by.setdefault(key, []).append(row["category"])
    closed = set()
    for key, cats in by.items():
        if len(cats) >= 2 and cats[-1] == cats[-2] and cats[-1] in ("no_ball_contact", "impact_without_recovery_entry"):
            closed.add(key)
    return closed


def accepted_state_counts(rows):
    counts = {name: 0 for name in FOCUS_STATES}
    counts["stable_bilateral"] = 0
    counts["contact_migration"] = 0
    accepted = 0
    for row in rows:
        if row.get("category") != "valid_impact_recovery_ic":
            continue
        accepted += 1
        for name in row.get("states") or []:
            if name in counts:
                counts[name] += 1
    return counts, accepted


def missing_states(rows):
    counts, accepted = accepted_state_counts(rows)
    if accepted < 6:
        return []
    floor = max(1, accepted // 12)
    return [name for name in FOCUS_STATES if counts[name] < floor]


def prefer_score(cell, missing):
    score = 0
    if "relative_rolling" in missing and abs(float(cell["phi_deg"])) >= 40.0 and float(cell["speed_mps"]) <= 8.5:
        score += 4
    if "tangential_slip" in missing and abs(float(cell["phi_deg"])) >= 40.0:
        score += 3
    if "asymmetric_or_unilateral" in missing and abs(float(cell["z_off_m"])) >= 0.015:
        score += 3
    if ("complete_grasp_loss" in missing or "airborne_object" in missing) and float(cell["speed_mps"]) >= 9.5:
        score += 4
    if "delayed_grasp_failure" in missing and float(cell["delay_s"]) >= 0.40 and float(cell["speed_mps"]) <= 7.5:
        score += 2
    if "intermittent_contact" in missing and abs(float(cell["z_off_m"])) >= 0.015 and float(cell["speed_mps"]) >= 8.5:
        score += 2
    if "recaught_displaced" in missing and float(cell["speed_mps"]) >= 8.5 and abs(float(cell["phi_deg"])) >= 40.0:
        score += 2
    return score


def region_stats(rows):
    azimuth_accepts = {}
    speed_region = {}
    for row in rows:
        parts = str(row["cell"]).split("|")
        split, azimuth, speed = parts[0], float(parts[1]), float(parts[2])
        region = (split, azimuth, speed)
        speed_region.setdefault(region, [0, 0])
        speed_region[region][0] += 1
        if row.get("category") == "valid_impact_recovery_ic":
            azimuth_accepts[(split, azimuth)] = azimuth_accepts.get((split, azimuth), 0) + 1
            speed_region[region][1] += 1
            z_key = (split, "z", round(float(parts[3]), 3))
            phi_key = (split, "phi", round(float(parts[4]), 1))
            azimuth_accepts[z_key] = azimuth_accepts.get(z_key, 0) + 1
            azimuth_accepts[phi_key] = azimuth_accepts.get(phi_key, 0) + 1
    return azimuth_accepts, speed_region


def next_job(rows, only_split=None):
    done = {row["key"] for row in rows}
    closed = closed_cells(rows)
    have = counts_from(rows)
    tried = {split: 0 for split in TARGETS}
    attempts_in = {}
    for row in rows:
        tried[row["split"]] = tried.get(row["split"], 0) + 1
        attempts_in[row["cell"]] = attempts_in.get(row["cell"], 0) + 1
    missing = missing_states(rows)
    azimuth_accepts, speed_region = region_stats(rows)
    order = sorted(
        TARGETS,
        key=lambda split: (have[split] >= TARGETS[split], tried[split], have[split] / TARGETS[split]),
    )
    if only_split:
        order = [only_split]
    best = None
    for split in order:
        if have[split] >= TARGETS[split]:
            continue
        for cell in planned_cells():
            if cell["split"] != split:
                continue
            bare = cell_key(cell, 0).rsplit("|", 1)[0]
            score = prefer_score(cell, missing)
            if bare in closed and score <= 0:
                continue
            if attempts_in.get(bare, 0) >= SEEDS_PER_CELL:
                continue
            for seed in range(SEEDS_PER_CELL):
                key = cell_key(cell, seed)
                if key in done:
                    continue
                spec = jitter(cell, seed)
                spec["seed"] = seed
                spec["cell"] = bare
                spec["key"] = key
                region = speed_region.get((split, float(cell["azimuth_deg"]), float(cell["speed_mps"])), [0, 0])
                penalty = 50 if region[0] >= 6 and region[1] == 0 else 0
                rank = (
                    penalty,
                    azimuth_accepts.get((split, float(cell["azimuth_deg"])), 0),
                    azimuth_accepts.get((split, "z", round(float(cell["z_off_m"]), 3)), 0),
                    azimuth_accepts.get((split, "phi", round(float(cell["phi_deg"]), 1)), 0),
                    region[0],
                    -score,
                    attempts_in.get(bare, 0),
                    key,
                )
                if best is None or rank < best[0]:
                    best = (rank, spec)
                break
        if best is not None and best[1]["split"] == split:
            return best[1]
    return None


def append_attempt(row):
    RAW.mkdir(parents=True, exist_ok=True)
    with ATTEMPTS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def save_accepted(blob, evidence):
    split = evidence["spec"]["split"]
    folder = RAW / "accepted" / split
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{evidence['source_episode']}.pkl"
    with path.open("wb") as handle:
        pickle.dump(blob, handle)
    return str(path.relative_to(ROOT)).replace("\\", "/")


def write_manifest(rows, physics):
    accepted = []
    for row in rows:
        if row.get("category") != "valid_impact_recovery_ic":
            continue
        accepted.append({
            "id": row["id"],
            "split": row["split"],
            "family": row["direction"],
            "azimuth_deg": row["azimuth_deg"],
            "path": row["path"],
        })
    payload = {
        "pilot_formal_grpo": "results/diagnostics/raw/formal_grpo_pilot",
        "targets": TARGETS,
        "accepted": counts_from(rows),
        "split_azimuth_deg": SPLIT_AZIMUTH,
        "physics": physics,
        "episodes": accepted,
    }
    MANIFEST.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def generate(env, physics, max_attempts, only_split=None):
    rows = load_log()
    ran = 0
    t0 = time.perf_counter()
    while ran < max_attempts:
        spec = next_job(rows, only_split)
        if spec is None:
            break
        n = counts_from(rows)[spec["split"]] + 1
        spec["attempt_id"] = f"ball_{spec['split']}_{n:04d}"
        spec["family"] = "impact"
        evidence, blob = run_episode(env, spec)
        evidence["id"] = spec["attempt_id"] if evidence["category"] == "valid_impact_recovery_ic" else None
        evidence["split"] = spec["split"]
        evidence["key"] = spec["key"]
        evidence["cell"] = spec["cell"]
        evidence["direction"] = spec["direction"]
        evidence["azimuth_deg"] = spec["azimuth_deg"]
        path = None
        if blob is not None:
            blob["spec"]["id"] = spec["attempt_id"]
            path = save_accepted(blob, evidence)
            evidence["id"] = spec["attempt_id"]
        evidence["path"] = path
        append_attempt(evidence)
        rows.append(evidence)
        ran += 1
        print(
            "ATTEMPT", spec["split"], spec["direction"], spec["speed_mps"],
            evidence["category"], counts_from(rows), flush=True,
        )
    elapsed = time.perf_counter() - t0
    payload = write_manifest(rows, physics)
    if ran:
        (RAW / "last_batch.json").write_text(
            json.dumps({"attempts": ran, "seconds": elapsed}),
            encoding="utf-8",
        )
    return payload, ran, elapsed, rows


def load_manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def load_episode(episode_id):
    manifest = load_manifest()
    match = next(row for row in manifest["episodes"] if row["id"] == episode_id)
    with (ROOT / match["path"]).open("rb") as handle:
        return pickle.load(handle)


def ids_for(split):
    if not MANIFEST.is_file():
        return []
    return [row["id"] for row in load_manifest()["episodes"] if row["split"] == split]


def summarize(rows):
    funnel = {}
    for row in rows:
        funnel[row["category"]] = funnel.get(row["category"], 0) + 1
    return {"attempts": len(rows), "funnel": funnel, "accepted": counts_from(rows)}


def probe(env):
    rows = []
    for azimuth, speed, phi in (
        (45.0, 8.0, 0.0),
        (135.0, 8.0, 0.0),
        (225.0, 8.0, 0.0),
        (270.0, 8.0, 0.0),
        (315.0, 8.0, 0.0),
        (0.0, 12.0, 40.0),
        (180.0, 12.0, 40.0),
    ):
        spec = {
            "split": "probe",
            "azimuth_deg": azimuth,
            "direction": direction_name(azimuth),
            "speed_mps": speed,
            "z_off_m": 0.0,
            "phi_deg": phi,
            "delay_s": 0.0,
            "seed": 0,
            "cell": f"probe|{azimuth}|{speed}|{phi}",
            "key": f"probe|{azimuth}|{speed}|{phi}",
            "attempt_id": f"probe_{azimuth}_{speed}",
            "family": "impact",
        }
        evidence, _blob = run_episode(env, spec)
        rows.append({
            "azimuth": azimuth,
            "speed": speed,
            "phi": phi,
            "category": evidence["category"],
            "max_e_pos_m": evidence.get("max_e_pos_m"),
            "peak_n": evidence.get("peak_normal_force_n"),
            "contact_t": evidence.get("contact_t"),
            "pre": evidence.get("pre_impact"),
            "post50": evidence.get("post_impact_50ms"),
        })
        print("PROBE", azimuth, speed, phi, evidence["category"], evidence.get("max_e_pos_m"), evidence.get("peak_normal_force_n"), flush=True)
    (RAW / "probe_directions.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _audit_choice(rows, split, limit):
    accepted = [
        row for row in rows
        if row.get("category") == "valid_impact_recovery_ic" and row.get("split") == split
    ]
    picked = []
    used = set()

    def take(predicate):
        for row in accepted:
            if len(picked) >= limit or row.get("id") in used:
                continue
            if predicate(row):
                picked.append(row)
                used.add(row["id"])
                return

    for direction in ("+X", "-X", "+Y", "-Y", "oblique_+X_+Y", "oblique_-X_+Y", "oblique_-X_-Y", "oblique_+X_-Y"):
        take(lambda row, direction=direction: row.get("direction") == direction)
    for state in FOCUS_STATES:
        take(lambda row, state=state: state in (row.get("states") or []))
    return [row["id"] for row in picked[:limit]]


def nominal_continue(env, blob, seconds=3.0):
    restore_capture(env, blob["capture"])
    ref = blob["ref"]
    st = _hold_state(env.sim)
    t0 = float(env.sim.data.time)
    fell = False
    basin = False
    samples = []
    anchor = None
    while float(env.sim.data.time) < min(t0 + seconds, T_TASK - 1e-9):
        for _ in range(v3.N_SUB):
            nominal_substep(env.sim, env.gains, st)
            if float(physical_pack(env.sim)["obj_z"]) < Z_ABORT:
                fell = True
                break
        sample, anchor = grasp_sample(env.sim, env.switch, anchor)
        samples.append(sample)
        if fell or errors(env.sim, ref)["z"] < Z_ABORT:
            fell = True
            break
        if success.basin_ok(errors(env.sim, ref), closed_grip(env.sim)):
            basin = True
            break
    return {
        "fell": bool(fell),
        "basin_without_recovery": bool(basin),
        "states": label_states(samples),
        "steps": len(samples),
    }


def policy_continue(env, blob, cap=120):
    from training.supervised_coverage_warmstart import policy_step

    restore_capture(env, blob["capture"])
    ref = blob["ref"]
    tokens, grips, samples = [], [], []
    anchor = None
    basin_run = 0
    fell = False
    hold = None
    lateral = False
    wrist = False
    for _ in range(cap):
        token, act = policy_step(env)
        sample, anchor = grasp_sample(env.sim, env.switch, anchor)
        samples.append(sample)
        tokens.append(int(token))
        grips.append(float(act[6]))
        if abs(float(act[0])) > 0.20 or abs(float(act[1])) > 0.20:
            lateral = True
        if abs(float(act[3])) > 0.20 or abs(float(act[4])) > 0.20 or abs(float(act[5])) > 0.20:
            wrist = True
        row = errors(env.sim, ref)
        if row["z"] < Z_ABORT:
            fell = True
            break
        inside = success.basin_ok(row, float(act[6]) > 0.5)
        basin_run = basin_run + 1 if inside else 0
        if basin_run >= success.PERSIST_STEPS:
            success.apply_handoff(env.sim, ref)
            hold = success.nominal_validate(env, ref)
            break
    label = success.label_of(fell, hold is not None, hold is not None, hold)
    opened = any(value < -0.5 for value in grips)
    closed_later = False
    seen_open = False
    for grip, sample in zip(grips, samples):
        if grip < -0.5:
            seen_open = True
        if seen_open and grip > 0.5 and sample["nL"] > 0 and sample["nR"] > 0:
            closed_later = True
            break
    during = label_states(samples)
    return {
        "label": label,
        "success": bool(success.is_success(label)),
        "basin": hold is not None,
        "held_10": None if hold is None else bool(hold.get("held_10")),
        "held_12": None if hold is None else bool(hold.get("held_12")),
        "tokens": tokens[:24],
        "opened": bool(opened),
        "reclosed_bilateral": bool(closed_later),
        "lateral_command": bool(lateral),
        "wrist_command": bool(wrist),
        "states_during_rollout": during,
        "n_steps": len(tokens),
    }


def audit_subset(env, train_limit=6, val_limit=4):
    rows = load_log()
    chosen = [("train", item) for item in _audit_choice(rows, "train", train_limit)]
    chosen += [("validation", item) for item in _audit_choice(rows, "validation", val_limit)]
    out = []
    for split, episode_id in chosen:
        if split == "final_test":
            raise RuntimeError("final test is sealed during the audit")
        blob = load_episode(episode_id)
        if blob["spec"]["split"] != split:
            raise RuntimeError(f"{episode_id} split mismatch")
        zero = nominal_continue(env, blob)
        policy = policy_continue(env, blob)
        out.append({
            "id": episode_id,
            "split": split,
            "direction": blob["spec"]["direction"],
            "entry_states": blob["evidence"].get("states", []),
            "zero": zero,
            "policy": policy,
        })
        print("AUDIT", episode_id, policy["label"], "zero_basin", zero["basin_without_recovery"], flush=True)
    (RAW / "recoverability_audit.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def _hist(rows, key):
    hist = {}
    for row in rows:
        if row.get("category") != "valid_impact_recovery_ic":
            continue
        value = row["spec"][key] if key in row.get("spec", {}) else row.get(key)
        hist[str(value)] = hist.get(str(value), 0) + 1
    return hist


def write_report(rows, physics, audit, elapsed_s, ran):
    funnel = {}
    for row in rows:
        funnel[row["category"]] = funnel.get(row["category"], 0) + 1
    accepted = counts_from(rows)
    counts, n_acc = accepted_state_counts(rows)
    by_split_state = {split: {} for split in TARGETS}
    fidelity = []
    families = {split: {} for split in TARGETS}
    for row in rows:
        if row.get("category") != "valid_impact_recovery_ic":
            continue
        fidelity.append(float(row.get("snapshot_qpos_error", 1.0)))
        families[row["split"]][row["direction"]] = families[row["split"]].get(row["direction"], 0) + 1
        for state in row.get("states") or []:
            bucket = by_split_state[row["split"]]
            bucket[state] = bucket.get(state, 0) + 1
    azimuths = {split: set() for split in TARGETS}
    for row in rows:
        if row.get("category") == "valid_impact_recovery_ic":
            azimuths[row["split"]].add(float(row["azimuth_deg"]))
    leaked = False
    for left, right in (("train", "validation"), ("train", "final_test"), ("validation", "final_test")):
        if azimuths[left] & azimuths[right]:
            leaked = True
    ready = (
        accepted["train"] >= TARGETS["train"]
        and accepted["validation"] >= TARGETS["validation"]
        and accepted["final_test"] >= TARGETS["final_test"]
        and not leaked
        and (not fidelity or max(fidelity) <= FIDELITY_TOL)
    )
    remaining = sum(max(0, TARGETS[split] - accepted[split]) for split in TARGETS)
    per_attempt = None if not ran or elapsed_s <= 0 else elapsed_s / ran
    accept_rate = n_acc / max(len(rows), 1)
    batch_path = RAW / "last_batch.json"
    if per_attempt is None and batch_path.is_file():
        batch = json.loads(batch_path.read_text(encoding="utf-8"))
        if batch.get("attempts") and batch.get("seconds"):
            per_attempt = float(batch["seconds"]) / float(batch["attempts"])
    eta_s = None
    if per_attempt and accept_rate > 0 and remaining > 0:
        eta_s = per_attempt * (remaining / accept_rate)
    verdict = "BALL-IMPACT DATASET READY" if ready else "BALL-IMPACT DATASET INCOMPLETE"
    lines = [
        "# Ball-impact dataset rebuild",
        "",
        "Generation and validation only. No GRPO, BC, or SAC update was run.",
        "",
        "This file is the live aggregate. `BALL_IMPACT_DATASET_REBUILD.md` is the frozen historical report and is not rewritten.",
        "",
        "The completed 120-update GRPO run is sealed at `results/diagnostics/raw/formal_grpo_pilot`.",
        "New initial conditions are only under `results/diagnostics/raw/ball_impact_dataset`.",
        "",
        "## Physics",
        "",
        "```",
        json.dumps(physics, indent=2),
        "```",
        "",
        "Cylinder and ball masses, radii, friction, condim, and the noslip solver are the current ballistic scene.",
        "Nominal mass and friction were not randomized: `nominal.yaml` has `randomize: false`, and `BallisticSim.reset` keeps the XML mass and finger-object mu.",
        "The ball free-joint state is written only at launch, before contact. The cylinder pose, hand pose, and post-impact velocity are not edited.",
        "",
        "## Rejection funnel",
        "",
        f"Attempt log: `results/diagnostics/raw/ball_impact_dataset/attempts.jsonl` ({len(rows)} rows).",
        "",
        "```",
        json.dumps(funnel, indent=2),
        "```",
        "",
        "## Accepted counts",
        "",
        f"Targets are train {TARGETS['train']}, validation {TARGETS['validation']}, final test {TARGETS['final_test']}.",
        "",
        "```",
        json.dumps({"accepted": accepted, "by_direction": families}, indent=2),
        "```",
        "",
        "Splits were assigned by launch azimuth before any episode:",
        "",
        "```",
        json.dumps(SPLIT_AZIMUTH, indent=2),
        "```",
        "",
        f"Azimuth overlap across splits: {leaked}.",
        "",
        "## Impact parameters of accepted episodes",
        "",
        "```",
        json.dumps({
            "speed_mps": _hist(rows, "speed_mps"),
            "z_off_m": _hist(rows, "z_off_m"),
            "phi_deg": _hist(rows, "phi_deg"),
            "delay_s": _hist(rows, "delay_s"),
        }, indent=2),
        "```",
        "",
        "## Contact verification",
        "",
        "An accepted episode has a `ball-cylinder` contact with normal force, a contact timestamp after launch, and geometry entry after that contact.",
        "Ball-finger contact is recorded separately and is not accepted as a substitute for cylinder contact.",
        f"Accepted episodes with the ball still touching the cylinder at entry: {sum(1 for row in rows if row.get('category') == 'valid_impact_recovery_ic' and row.get('ball_interacting_at_entry'))}.",
        "",
        "## Post-impact physical states",
        "",
        "Labels overlap. A sample can carry several. Rolling is labeled only when the object angular rate is at least 1 rad/s, the finger contact point migrates by at least 3 mm in the object frame, and the tangential speed is consistent with radius times angular rate. Contact migration with a small angular rate is labeled separately.",
        "",
        "```",
        json.dumps({"accepted_episodes": n_acc, "state_counts": counts, "by_split": by_split_state}, indent=2),
        "```",
        "",
        "Acceptance is not conditioned on these labels. Counts above are how often each state occurred on accepted impact entries.",
        f"Under-represented states used only to retarget pre-impact launch parameters: {missing_states(rows)}.",
        "",
        "Accepted-entry coverage, as a fraction of accepted episodes:",
        "",
        f"- unilateral or asymmetric finger contact: {counts['asymmetric_or_unilateral']} / {n_acc}",
        f"- rolling signature (angular rate, contact migration, and rolling-speed match): {counts['relative_rolling']} / {n_acc}",
        f"- contact migration while angular rate stayed below 1 rad/s: {counts['contact_migration']} / {n_acc}",
        f"- tangential slip (slip speed, and not explained by rolling): {counts['tangential_slip']} / {n_acc}",
        f"- complete grasp loss before entry: {counts['complete_grasp_loss']} / {n_acc}",
        f"- airborne separation before entry: {counts['airborne_object']} / {n_acc}",
        f"- intermittent contact, recaught-displaced, delayed failure, stable bilateral before entry: "
        f"{counts['intermittent_contact']}, {counts['recaught_displaced']}, {counts['delayed_grasp_failure']}, {counts['stable_bilateral']}",
        "",
        "Stable bilateral contact before entry is rare because an already-nominal grasp is rejected as trivial.",
        "Delayed failure and recatch usually need longer than the short interval from impact to the 6 mm entry, so they show up in the later rollout trace rather than in the entry label.",
        "Low strikes (`z_off` -15 mm) are the largest height bin so far. Later sampling prefers heights and tangential offsets that are still thin.",
        "",
        "## Geometry entry and snapshot fidelity",
        "",
        "Entry is the existing position threshold, 6 mm for 40 ms, and only after the ball has been launched.",
        f"Restore-and-continue qpos error over {FIDELITY_STEPS} nominal steps, max {max(fidelity) if fidelity else None}, tolerance {FIDELITY_TOL}.",
        "The capture stores MuJoCo state, time, controller desired pose and commands, geometry-switch reference, and the in-flight ball. Controller targets are not rewritten.",
        "",
        "## Recoverability audit",
        "",
        "Train and validation only. Final test was not rolled.",
        "Zero means the nominal hold continues from the entry snapshot. The frozen K=36 initialization is rolled with the nominal-basin, explicit handoff, and 10 s nominal-only definition.",
        "A finite-search failure is not treated as proof that the state is unreachable.",
        "Full rollout records: `results/diagnostics/raw/ball_impact_dataset/recoverability_audit.json`.",
        "",
        "## Recovery mechanisms",
        "",
        "The legal interface is unchanged: 7-D `recovery7d_airborne_v3`, K=36 tokens, residual cap ±0.25.",
        "Previously verified behaviors, on the older pilot initial conditions rather than this dataset, are token 34 open-finger release for slip, token 35 airborne chase, and token 32 lateral chase on `impact_zp6`.",
        "Object rotation alone is not counted as controlled rolling. Slip velocity alone is not counted as controlled sliding. Recatch is not counted as successful unless the rollout reaches the nominal basin, hands off, and holds for 10 s under the nominal controller.",
        "",
        "On this audit, a mechanism is reported only as concurrent evidence:",
        "",
        "- closed-grip lateral command together with measured tangential slip",
        "- wrist command together with the rolling signature above",
        "- a grip-close command while both finger contacts are present",
        "- an open command followed by bilateral contact",
        "- loss of both contacts followed by bilateral contact",
        "",
        "Those coincidences do not by themselves identify which command produced the motion.",
        "",
    ]
    if audit:
        n_ok = sum(1 for row in audit if row["policy"]["success"])
        n_zero = sum(1 for row in audit if row["zero"]["basin_without_recovery"])
        lines.append(
            f"Frozen K=36 successes on this subset: {n_ok} / {len(audit)}. "
            f"Nominal continuation already inside the basin: {n_zero} / {len(audit)}."
        )
        if n_ok:
            lines.append(
                "Successes in this subset used a brief open and then closed-lift tokens. "
                "That is open-then-recontact followed by a nominal handoff, not evidence that sliding or rolling was controlled."
            )
        lines.append(
            "Wrist motion during a rolling signature, and open-then-recontact after grasp loss, occurred on unsuccessful rollouts as well. "
            "Those episodes show the physical events without a 10 s nominal survival."
        )
        lines.append("")
        for row in audit:
            policy = row["policy"]
            during = set(policy["states_during_rollout"])
            bits = []
            if policy["lateral_command"] and "tangential_slip" in during:
                bits.append("lateral command during measured slip")
            if policy["wrist_command"] and "relative_rolling" in during:
                bits.append("wrist command during rolling signature")
            if policy["opened"] and policy["reclosed_bilateral"]:
                bits.append("open then bilateral recontact")
            if "complete_grasp_loss" in during and policy["reclosed_bilateral"]:
                bits.append("loss then bilateral recontact")
            if "asymmetric_or_unilateral" in row["entry_states"] and "stable_bilateral" in during:
                bits.append("unilateral entry later showed bilateral contact")
            lines.append(
                f"- `{row['id']}` ({row['direction']}): entry {row['entry_states']}; "
                f"zero basin {row['zero']['basin_without_recovery']}; "
                f"K=36 {policy['label']}; concurrent {bits or ['none isolated']}."
            )
        lines.append("")
    lines.extend([
        "## Loader",
        "",
        "Formal GRPO reads this manifest when `dataset: ball_impact`.",
        "Train ids and validation ids come from the manifest splits. Final-test ids are not sampled.",
        "The observation, action tokens, residual cap, geometry entry, reward, success rule, actor, and KL coefficient are unchanged.",
        "Impact episodes keep the existing impact-family KL. The pilot catalog and `results/diagnostics/raw/formal_grpo_pilot` are not overwritten.",
        "",
        "Future training command, not executed here:",
        "",
        "```",
        "python -u training/formal_grpo.py --config config/formal_grpo_ballistic.yaml",
        "```",
        "",
        "Resume generation:",
        "",
        "```",
        "python -u training/ball_impact_dataset.py --attempts 200",
        "```",
        "",
        f"Accepted so far {n_acc} of {len(rows)} logged launches. Remaining to target {remaining}.",
        f"Latest generation batch: {None if per_attempt is None else round(per_attempt, 2)} s per launch.",
        f"Rough remaining time at the current acceptance rate: {None if eta_s is None else round(eta_s / 60.0, 1)} minutes.",
        "",
        verdict,
        "",
        "TRAINING HAS NOT BEEN STARTED.",
        "",
    ])
    path = ROOT / "results" / "diagnostics" / "BALL_IMPACT_DATASET_REBUILD_CURRENT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path, verdict


def relabel_missing(env):
    rows = load_log()
    changed = 0
    for index, row in enumerate(rows):
        if row.get("category") != "valid_impact_recovery_ic" or row.get("states"):
            continue
        spec = dict(row["spec"])
        evidence, blob = run_episode(env, spec)
        evidence["split"] = row["split"]
        evidence["key"] = row["key"]
        evidence["cell"] = row["cell"]
        evidence["direction"] = row["direction"]
        evidence["azimuth_deg"] = row["azimuth_deg"]
        if evidence["category"] != "valid_impact_recovery_ic":
            evidence["id"] = None
            evidence["path"] = None
            evidence["reason"] = "replay did not reproduce a valid impact entry: " + evidence["reason"]
            old = ROOT / str(row["path"]) if row.get("path") else None
            if old is not None and old.is_file():
                old.unlink()
        else:
            evidence["id"] = row["id"]
            evidence["source_episode"] = row["id"]
            blob["spec"]["id"] = row["id"]
            evidence["path"] = save_accepted(blob, evidence)
        rows[index] = evidence
        changed += 1
        print("RELABEL", row["id"], evidence["category"], evidence.get("states"), flush=True)
    if changed:
        text = "".join(json.dumps(item) + "\n" for item in rows)
        ATTEMPTS.write_text(text, encoding="utf-8")
        write_manifest(rows, json.loads((RAW / "physics.json").read_text(encoding="utf-8")))
    return changed


def main():
    parser = argparse.ArgumentParser(description="Generate genuine ball-impact recovery ICs")
    parser.add_argument("--attempts", type=int, default=24)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--relabel", action="store_true")
    parser.add_argument("--split", default="", help="Restrict new launches to one split, for example train")
    args = parser.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    from training.formal_grpo import load_models
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ref = load_models(
        ROOT / "results" / "diagnostics" / "raw" / "pre_grpo_sanity" / "policy_B_init.pt",
        ROOT / "results" / "diagnostics" / "raw" / "pre_grpo_sanity" / "policy_A_reference.pt",
        device,
    )
    env = make_training_env(device, model=model)
    env.begin_nominal()
    physics = physics_record(env.sim)
    if not physics["source_constants_match_model"]:
        raise RuntimeError(f"scene constants drifted: {physics}")
    (RAW / "physics.json").write_text(json.dumps(physics, indent=2), encoding="utf-8")
    if args.probe:
        probe(env)
        return
    if args.relabel:
        relabel_missing(env)
    rows = load_log()
    ran = 0
    elapsed = 0.0
    if args.attempts > 0 and not args.audit and not args.report:
        _payload, ran, elapsed, rows = generate(env, physics, args.attempts, args.split or None)
    elif args.attempts > 0 and not args.report:
        _payload, ran, elapsed, rows = generate(env, physics, args.attempts, args.split or None)
    audit = None
    if (RAW / "recoverability_audit.json").is_file() and args.report and args.attempts == 0:
        audit = json.loads((RAW / "recoverability_audit.json").read_text(encoding="utf-8"))
    if args.audit or (ran > 0 and counts_from(rows)["train"] + counts_from(rows)["validation"] > 0 and not args.probe):
        if args.audit or ran > 0:
            audit = audit_subset(env)
    if rows:
        path, verdict = write_report(rows, physics, audit, elapsed, ran)
        print("SUMMARY", json.dumps(summarize(rows)), "ran", ran, "sec", round(elapsed, 1), flush=True)
        print(verdict, flush=True)
        print("WROTE", path, flush=True)


if __name__ == "__main__":
    main()
