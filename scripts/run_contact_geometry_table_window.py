"""Diagnose wrong-geometry impact, ejection impulses, and table-limited catch windows.

Original-task conditions preserve physics/table/controller/SUCCESS_NEW.
Table-free / early-onset conditions are labeled diagnostic-only.
No SFT/GRPO. No mixing diagnostic successes into K49 experts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from controllers.recovery.action_map import AIRBORNE_V3, V_AIRBORNE, map_action
import models.airborne_v3 as v3
from data_generation.ball_impact_dataset import (
    LAUNCH_WINDOW_S, aim_ball, cylinder_contacts, ready_to_launch,
)
from envs.outcomes.horizon_outcome import NOMINAL_S, latched, recovery_failed, terminal_reason
from envs.outcomes.nominal_pose import capture_ref
from envs.outcomes.physical_recovery import physical_pack
from envs.outcomes.recovery_success import apply_handoff, nominal_step
from envs.outcomes import table_drop as td
from envs.simulation.capture import restore_capture, take_capture
from envs.simulation.grasp_quality import body_twist
from envs.simulation.impact_ball import apply_ball_free_state
from envs.step_legal import step_legal
from evaluation.full_objective import P_TGT, U_TGT, contact_summary, window_stats as window_stats_fo
from evaluation.grasp_configuration import cylinder_axis_world, measure
from models.k48_checkpoint import load_trained_k48
from scripts.expand_k49_capture_brake_diagnostics import find_snap, load_blob
from scripts.expand_k49_hard_dynamic import ic_metrics as hard_ic_metrics, is_high_motion
from scripts.expand_k49_large_chase_dynamic import HM_THR, hand_pos, obj_pos
from scripts.generate_k49_sliding_recatch_dataset import (
    CONFIG_ID, SFT, decode, mapping_checksum, write_csv, write_json,
)
from training.action_vocab_k49 import SECURE_REVERSE_WRIST, build_k49_table
from training.recovery_episode import (
    T_TASK, Z_ABORT, fresh_nominal_state, make_training_env, nominal_substep,
)
from visualization.camera import mjv_camera_from_preset
from visualization.overlays import caption
from visualization.video_recorder import write_frames_mp4

OUT = ROOT / "results" / "diagnostics" / "contact_geometry_table_window"

TOTAL_ROLLOUT_BUDGET = 1800
WALL_LIMIT_S = 40 * 60
PHASE_A_FRAC = 0.35
# remaining: alignment / force-timing / early-onset / table-cf / full-search
PHASE_B_SPLIT = (0.30, 0.25, 0.15, 0.15, 0.15)

BE_ICS = [
    "hard_910012", "hard_910006", "hard_910095", "hard_910031", "bimp_0033",
    "bimp_0010", "bimp_0036", "ubimp_0040", "hdimp_0002", "hdimp_0012",
    "hdimp_0017", "hdimp_0022", "hdimp_0030", "hdimp_0036", "hdimp_0039",
]
FINE_DOWN = np.array([0.0, 0.0, 0.15, 0.0, 0.0, 0.0, 1.0], np.float32)

CHASE = {
    "t17": [17] * 18,
    "t33": [33] * 18,
    "open_t17": [30] * 3 + [17] * 16,
    "t1": [1] * 20,
}


def tokens_hash(x):
    return hashlib.sha256(json.dumps(x).encode()).hexdigest()[:16]


def body_v_w(sim, body_id):
    v, w = body_twist(sim.model, sim.data, int(body_id))
    return np.asarray(v, float), np.asarray(w, float)


def geom_id(sim, name):
    gid = int(mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, name))
    return gid  # -1 if missing


def has_geom(sim, name):
    return geom_id(sim, name) >= 0


def finger_frame(sim):
    pl = np.asarray(sim.data.xpos[sim.ids.left_body], float)
    pr = np.asarray(sim.data.xpos[sim.ids.right_body], float)
    ph = hand_pos(sim)
    Rh = np.asarray(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    mid = 0.5 * (pl + pr)
    gap_vec = pl - pr
    gap = float(np.linalg.norm(gap_vec)) + 1e-12
    u_open = gap_vec / gap  # right -> left
    # distal approx: hand +z local often along fingers; use finger COM - hand
    distal = mid - ph
    dn = float(np.linalg.norm(distal)) + 1e-12
    u_dist = distal / dn
    u_palm = np.cross(u_open, u_dist)
    un = float(np.linalg.norm(u_palm)) + 1e-12
    u_palm = u_palm / un
    return {
        "pl": pl, "pr": pr, "ph": ph, "mid": mid, "gap": gap,
        "u_open": u_open, "u_dist": u_dist, "u_palm": u_palm, "Rh": Rh,
    }


def classify_contact_site(sim, pos, other_body, ff):
    """A inside grip / B outer finger / C fingertip edge / D palm/hand / E table."""
    table_g = geom_id(sim, "table")
    floor_g = geom_id(sim, "floor")
    # caller may pass table via other_body=world; handle separately
    if other_body == 0:  # world/table often body 0
        return "E_table"
    if other_body == int(sim.ids.hand_body):
        return "D_palm_or_hand"
    if other_body not in (int(sim.ids.left_body), int(sim.ids.right_body)):
        return "other"
    # along open axis: 0 at right, gap at left
    t = float(np.dot(pos - ff["pr"], ff["u_open"]))
    between = 0.0 < t < ff["gap"]
    # outer: beyond outer faces by small margin
    outer = t < -0.002 or t > ff["gap"] + 0.002
    # tip: far along distal from hand
    tip_s = float(np.dot(pos - ff["ph"], ff["u_dist"]))
    tip = tip_s > 0.045  # ~finger length scale
    if tip and (between or outer):
        return "C_fingertip_edge"
    if outer:
        return "B_outer_finger"
    if between:
        return "A_inside_grip"
    return "A_inside_grip" if between else "B_outer_finger"


def enumerate_object_contacts(sim):
    """All contacts involving object: hand/fingers/table/ball."""
    obj = int(sim.ids.object_body)
    left, right, hand = int(sim.ids.left_body), int(sim.ids.right_body), int(sim.ids.hand_body)
    table_g = geom_id(sim, "table")
    floor_g = geom_id(sim, "floor")
    ff = finger_frame(sim)
    po = obj_pos(sim)
    vo, wo = body_v_w(sim, obj)
    out = []
    for i in range(int(sim.data.ncon)):
        c = sim.data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        b1 = int(sim.model.geom_bodyid[g1])
        b2 = int(sim.model.geom_bodyid[g2])
        if obj not in (b1, b2):
            continue
        other_b = b2 if b1 == obj else b1
        other_g = g2 if b1 == obj else g1
        pos = np.asarray(c.pos, float)
        frame = np.asarray(c.frame, float).reshape(3, 3)
        n = frame[0].copy()
        # force on geom1 from contact; map to force on object
        wrench = np.zeros(6, float)
        try:
            mujoco.mj_contactForce(sim.model, sim.data, i, wrench)
        except Exception:
            pass
        # compression: geom2 pushed +n, geom1 -n; wrench[0] is normal force magnitude along contact frame
        fn = float(wrench[0])
        # force ON object along world normal
        if b1 == obj:
            f_on_obj = -fn * n  # object is geom1 → force opposite +n
        else:
            f_on_obj = fn * n
        # site class
        if other_g == table_g or (floor_g >= 0 and other_g == floor_g):
            site = "E_table"
            partner = "table" if other_g == table_g else "floor"
        elif other_b == hand:
            site = "D_palm_or_hand"
            partner = "hand"
        elif other_b in (left, right):
            site = classify_contact_site(sim, pos, other_b, ff)
            partner = "L" if other_b == left else "R"
        else:
            site = "other"
            partner = f"body{other_b}"
        r = pos - po
        v_pt = vo + np.cross(wo, r)
        # hand-frame contact
        Rh = ff["Rh"]
        pos_h = Rh.T @ (pos - ff["ph"])
        axis_h = Rh.T @ cylinder_axis_world(sim)
        out.append({
            "partner": partner,
            "site": site,
            "pos": pos.tolist(),
            "pos_h": pos_h.tolist(),
            "n": n.tolist(),
            "f_on_obj": f_on_obj.tolist(),
            "fn": fn,
            "f_norm": float(np.linalg.norm(f_on_obj)),
            "v_pt": v_pt.tolist(),
            "axis_h": axis_h.tolist(),
            "is_hand": int(partner in ("L", "R", "hand")),
            "is_table": int(site == "E_table"),
        })
    return out, ff


def impulse_from_contacts(contacts, dt):
    """Sum hand–object impulse (N·s) on object."""
    J = np.zeros(3, float)
    J_table = np.zeros(3, float)
    for c in contacts:
        f = np.asarray(c["f_on_obj"], float)
        if c["is_hand"]:
            J += f * dt
        if c["is_table"]:
            J_table += f * dt
    return J, J_table


def aperture(sim):
    fj = sim.ids.finger_jnt
    qs = [float(sim.data.qpos[int(j)]) for j in fj]
    return float(sum(qs))  # both fingers; open ~0.08


def run_traced(env, blob, actions, tag="run"):
    """Step legal actions; collect contact/impulse/table timeline."""
    restore_capture(env, blob["capture"])
    # ensure table-drop monitor fresh unless capture restores it
    t0 = float(env.sim.data.time)
    dt_ctrl = 0.02
    lost_low = 0
    ok = True
    rows = []
    geo_rows = []
    first_hand = None
    first_table = None
    first_major_hand = None
    major_thresh = 0.02  # N·s per control step ~1 N average
    eject_events = []
    max_bil = 0.0
    bil_run = 0.0
    had_uni = had_bil = False
    vo_pre_major = None
    grip_at_major = None

    for ai, a in enumerate(actions):
        a = np.clip(np.asarray(a, np.float32).reshape(7), -1, 1)
        # pre-step state
        vo0, wo0 = body_v_w(env.sim, env.sim.ids.object_body)
        po0 = obj_pos(env.sim)
        ff0 = finger_frame(env.sim)
        contacts0, _ = enumerate_object_contacts(env.sim)
        cs = contact_summary(env.sim)
        any_hand = any(c["is_hand"] for c in contacts0)
        any_table = any(c["is_table"] for c in contacts0)
        t_rel = float(env.sim.data.time) - t0
        if any_hand and first_hand is None:
            first_hand = t_rel
        if any_table and first_table is None:
            first_table = t_rel

        J_hand, J_tab = impulse_from_contacts(contacts0, dt_ctrl)
        Jhn = float(np.linalg.norm(J_hand))

        # geometry at this step
        mid = ff0["mid"]
        outward = po0 - mid
        outward_u = outward / (float(np.linalg.norm(outward)) + 1e-12)
        # toward opening: projection onto open axis toward mid
        toward_open = -outward_u  # toward mid from COM
        J_out = float(np.dot(J_hand, outward_u))  # + = impulse pushes COM away from mid

        site_counts = {}
        for c in contacts0:
            site_counts[c["site"]] = site_counts.get(c["site"], 0) + 1

        if Jhn >= major_thresh and first_major_hand is None and any_hand:
            first_major_hand = {
                "t": t_rel, "step": ai, "J": J_hand.tolist(), "J_norm": Jhn,
                "J_outward": J_out, "vo_before": vo0.tolist(),
                "sites": dict(site_counts), "grip_a6": float(a[6]),
                "aperture": aperture(env.sim),
                "geo_between": int(0.0 < float(np.dot(po0 - ff0["pr"], ff0["u_open"])) < ff0["gap"]),
            }
            vo_pre_major = vo0.copy()
            grip_at_major = float(a[6])

        if cs["bilateral"]:
            had_bil = True
            bil_run += dt_ctrl
            max_bil = max(max_bil, bil_run)
        else:
            bil_run = 0.0
        if (int(cs["nL"]) > 0) ^ (int(cs["nR"]) > 0):
            had_uni = True

        rows.append({
            "t": t_rel, "step": ai,
            "nL": int(cs["nL"]), "nR": int(cs["nR"]), "bilateral": int(cs["bilateral"]),
            "any_hand": int(any_hand), "any_table": int(any_table),
            "J_hand_norm": Jhn, "J_outward": J_out,
            "J_table_norm": float(np.linalg.norm(J_tab)),
            "vo_norm": float(np.linalg.norm(vo0)),
            "obj_z": float(po0[2]),
            "aperture": aperture(env.sim),
            "grip_cmd": float(a[6]),
            "sites": json.dumps(site_counts),
            "latched": int(latched(env.sim)),
        })
        if any_hand or ai == 0:
            geo_rows.append({
                "t": t_rel,
                "com_between": int(0.0 < float(np.dot(po0 - ff0["pr"], ff0["u_open"])) < ff0["gap"]),
                "gap": ff0["gap"],
                "dist_com_mid": float(np.linalg.norm(po0 - mid)),
                "axis_dot_open": abs(float(np.dot(cylinder_axis_world(env.sim), ff0["u_open"]))),
                "sites": json.dumps(site_counts),
                "dominant_site": max(site_counts, key=site_counts.get) if site_counts else "none",
            })

        # step
        vz_before = td.object_vz(env.sim)
        step_legal(env.sim, a)
        env._prev = a.copy()
        td.observe(env.sim, vz_before)

        # post-step ejection check after major impulse window
        if first_major_hand is not None and ai == int(first_major_hand["step"]):
            vo1, _ = body_v_w(env.sim, env.sim.ids.object_body)
            po1 = obj_pos(env.sim)
            dv = vo1 - vo0
            # recompute outward at pre
            eject = float(np.dot(dv, outward_u)) > 0.05 and J_out > 0.005
            toward = float(np.dot(dv, toward_open))
            eject_events.append({
                "t": t_rel,
                "dv_outward": float(np.dot(dv, outward_u)),
                "dv_toward_open": toward,
                "dv_norm": float(np.linalg.norm(dv)),
                "J_outward": J_out,
                "J_norm": Jhn,
                "ejected": int(eject),
                "sites": json.dumps(site_counts),
                "grip_cmd": grip_at_major,
                "wrong_geom": int(any(s.startswith("B_") or s.startswith("C_") or s.startswith("D_")
                                      for s in site_counts)),
            })

        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            ok = False
            break

    t_drop = None
    if latched(env.sim):
        st = td.state_of(env.sim)
        t_drop = None if st.get("t") is None else float(st["t"]) - t0

    out = {
        "replay_ok": int(ok),
        "reason_fail": None if ok else (terminal_reason(env.sim) or "DROP"),
        "had_uni": int(had_uni),
        "had_bil": int(had_bil),
        "max_bil_persist_s": max_bil,
        "t_first_hand": first_hand,
        "t_first_table": first_table,
        "t_table_drop": t_drop,
        "t_end": float(env.sim.data.time) - t0,
        "obj_z_end": float(obj_pos(env.sim)[2]),
        "SUCCESS_NEW": 0,
        "win_mean_theta": None,
        "win_mean_pos": None,
        "hold_both_s": None,
        "protocol": "original",
    }
    if first_major_hand is not None:
        out["major_t"] = first_major_hand["t"]
        out["major_J_norm"] = first_major_hand["J_norm"]
        out["major_J_outward"] = first_major_hand["J_outward"]
        out["major_sites"] = json.dumps(first_major_hand["sites"])
        out["major_grip"] = first_major_hand["grip_a6"]
        out["major_between"] = first_major_hand["geo_between"]
        if eject_events:
            out["ejected"] = eject_events[0]["ejected"]
            out["dv_outward"] = eject_events[0]["dv_outward"]
            out["wrong_geom_at_major"] = eject_events[0]["wrong_geom"]
        else:
            out["ejected"] = 0
            out["dv_outward"] = None
            out["wrong_geom_at_major"] = 0
    else:
        out["major_t"] = None
        out["major_J_norm"] = 0.0
        out["major_J_outward"] = 0.0
        out["major_sites"] = "{}"
        out["major_grip"] = None
        out["major_between"] = None
        out["ejected"] = 0
        out["dv_outward"] = None
        out["wrong_geom_at_major"] = 0

    # window to irreversible
    if t_drop is not None:
        out["time_to_irreversible"] = t_drop
    elif not ok:
        out["time_to_irreversible"] = out["t_end"]
    else:
        out["time_to_irreversible"] = None

    if ok:
        apply_handoff(env.sim, blob["ref"])
        t_h = float(env.sim.data.time)
        samples = []
        next_sample = 0.0
        reason = None
        while reason is None and float(env.sim.data.time) - t_h < float(NOMINAL_S) - 1e-9:
            nominal_step(env.sim)
            t_rel = float(env.sim.data.time) - t_h
            if t_rel + 1e-12 >= next_sample:
                m = measure(env.sim, U_TGT, P_TGT)
                cs = contact_summary(env.sim)
                samples.append({
                    "t_rel": t_rel, "e_theta_deg": m["e_theta_deg"], "e_pos_mm": m["e_pos_mm"],
                    "bilateral": cs["bilateral"], "obj_z": m["obj_z"],
                })
                next_sample += 0.02
            if latched(env.sim):
                reason = "TABLE_DROP"
        win = window_stats_fo(samples, float(NOMINAL_S) - 2.0)
        ok_new = bool(win.get("working_SUCCESS_NEW_hold"))
        mat = (win.get("matrix") or {}).get("th10_pos12") or {}
        if mat:
            ok_new = bool(mat.get("SUCCESS_NEW"))
        out["SUCCESS_NEW"] = int(ok_new)
        out["win_mean_theta"] = win.get("mean_theta")
        out["win_mean_pos"] = win.get("mean_pos")
        out["hold_both_s"] = win.get("longest_both_ok_s")
        out["eval_reason"] = reason or ("SUCCESS_NEW" if ok_new else "POSE_OR_HOLD_FAIL")
    return out, rows, geo_rows, eject_events


def toks_to_actions(table, book48, toks):
    return [np.asarray(decode(table, book48, int(t)), np.float32) for t in toks]


def alignment_sequence(env, blob, mode, n=24):
    """Legal continuous sequences for grasp-alignment experiment A."""
    restore_capture(env, blob["capture"])
    acts = []
    for i in range(n):
        ff = finger_frame(env.sim)
        po = obj_pos(env.sim)
        ph = hand_pos(env.sim)
        Rh = ff["Rh"]
        err_h = Rh.T @ (po - ph)
        # default open while aligning
        a = np.zeros(7, np.float32)
        if mode in ("B_open_align", "D_translate_in", "E_delay_close", "F_soft_then_hard"):
            a[6] = -1.0 if i < n // 2 else (0.35 if mode == "F_soft_then_hard" and i < 3 * n // 4 else 1.0)
        elif mode == "C_wrist_align":
            a[6] = -1.0 if i < n // 3 else 1.0
            # align open axis with cylinder radial (perp to cylinder axis)
            u_cyl = cylinder_axis_world(env.sim)
            # command small wrist about hand y/z to reduce |axis·open|
            a[4] = float(np.clip(0.3 * float(np.dot(u_cyl, ff["u_open"])), -1, 1))
        else:
            a[6] = 1.0
        # translation toward object, bias into opening
        a[0] = float(np.clip(err_h[0] / 0.05, -1, 1))
        a[1] = float(np.clip(err_h[1] / 0.05, -1, 1))
        if mode == "D_translate_in":
            # push along open axis so COM enters between fingers
            t = float(np.dot(po - ff["pr"], ff["u_open"]))
            need = 0.5 * ff["gap"] - t
            a_open_h = Rh.T @ (need * ff["u_open"])
            a[0] = float(np.clip(a[0] + 0.5 * a_open_h[0] / 0.03, -1, 1))
            a[1] = float(np.clip(a[1] + 0.5 * a_open_h[1] / 0.03, -1, 1))
        if mode == "E_delay_close":
            a[6] = -1.0 if i < int(0.7 * n) else 1.0
        if mode == "F_soft_then_hard":
            if i < n // 2:
                a[6] = -0.48
            elif i < 3 * n // 4:
                a[6] = 0.35
            else:
                a[6] = 1.0
        # mild chase velocity match blending for non-baseline
        if mode != "A_k49":
            vo, _ = body_v_w(env.sim, env.sim.ids.object_body)
            v_des_h = Rh.T @ (0.5 * vo)
            a[0] = float(np.clip(0.5 * a[0] + 0.5 * np.clip(v_des_h[0] / V_AIRBORNE, -1, 1), -1, 1))
            a[1] = float(np.clip(0.5 * a[1] + 0.5 * np.clip(v_des_h[1] / V_AIRBORNE, -1, 1), -1, 1))
        step_legal(env.sim, a)
        acts.append(a.copy())
        failed, _ = recovery_failed(env.sim, blob["ref"], 0)
        if failed:
            break
    # settle
    for _ in range(12):
        a = np.array([0, 0, 0, 0, -0.2, 0, 1], np.float32)
        step_legal(env.sim, a)
        acts.append(a.copy())
    return acts


def force_timing_sequence(env, blob, p):
    restore_capture(env, blob["capture"])
    acts = []
    n_open = int(p["n_open"])
    n_align = int(p["n_align"])
    n_close = int(p["n_close"])
    n_como = int(p["n_comotion"])
    grip_close = float(p["grip_close"])
    for i in range(n_open + n_align + n_close + n_como + int(p["n_brake"])):
        ff = finger_frame(env.sim)
        po, ph = obj_pos(env.sim), hand_pos(env.sim)
        Rh = ff["Rh"]
        err_h = Rh.T @ (po - ph)
        vo, _ = body_v_w(env.sim, env.sim.ids.object_body)
        v_des_h = Rh.T @ (float(p["v_scale"]) * vo)
        a = np.zeros(7, np.float32)
        a[0] = float(np.clip(0.4 * err_h[0] / 0.05 + 0.6 * v_des_h[0] / V_AIRBORNE, -1, 1))
        a[1] = float(np.clip(0.4 * err_h[1] / 0.05 + 0.6 * v_des_h[1] / V_AIRBORNE, -1, 1))
        if i < n_open:
            a[6] = -1.0
        elif i < n_open + n_align:
            a[6] = float(p.get("grip_align", -0.48))
            a[4] = float(np.clip(0.25 * float(np.dot(cylinder_axis_world(env.sim), ff["u_open"])), -1, 1))
        elif i < n_open + n_align + n_close:
            a[6] = grip_close
        elif i < n_open + n_align + n_close + n_como:
            a[6] = grip_close
            a[0] = float(np.clip(v_des_h[0] / V_AIRBORNE * 0.4, -1, 1))
            a[1] = float(np.clip(v_des_h[1] / V_AIRBORNE * 0.4, -1, 1))
        else:
            a[:] = 0
            a[6] = 1.0
            if p.get("use_fine_down"):
                a = FINE_DOWN.copy()
        step_legal(env.sim, a)
        acts.append(a.copy())
        if recovery_failed(env.sim, blob["ref"], 0)[0]:
            break
    return acts


def set_table_diagnostic(sim, mode):
    """C2 counterfactual. Returns restore fn. Does NOT count as original SUCCESS_NEW."""
    table_g = geom_id(sim, "table")
    pos0 = np.asarray(sim.model.geom_pos[table_g], float).copy()
    size0 = np.asarray(sim.model.geom_size[table_g], float).copy()
    contype0 = int(sim.model.geom_contype[table_g])
    conaff0 = int(sim.model.geom_conaffinity[table_g])

    def restore():
        sim.model.geom_pos[table_g][:] = pos0
        sim.model.geom_size[table_g][:] = size0
        sim.model.geom_contype[table_g] = contype0
        sim.model.geom_conaffinity[table_g] = conaff0
        mujoco.mj_forward(sim.model, sim.data)

    if mode == "C2_lower_table":
        # increase clearance: move table down 8 cm
        sim.model.geom_pos[table_g][2] = pos0[2] - 0.08
    elif mode == "C2_no_table_contact":
        sim.model.geom_contype[table_g] = 0
        sim.model.geom_conaffinity[table_g] = 0
    else:
        return lambda: None
    mujoco.mj_forward(sim.model, sim.data)
    return restore


def early_onset_snapshots(env, blob, offsets_s=(-0.20, -0.12, -0.06)):
    """Replay ball-impact episode; capture states before official Recovery entry.

    Diagnostic protocol — labeled separately from original Recovery IC.
    """
    spec = dict(blob.get("spec") or {})
    if not spec:
        return []
    # ensure attempt_id
    if "attempt_id" not in spec:
        spec["attempt_id"] = blob.get("evidence", {}).get("source_episode", "early")
    # Custom loop mirroring run_episode but saving pre-entry captures
    env.begin_nominal()
    st = fresh_nominal_state()
    switch = env.switch
    launched = False
    t_ready = None
    t_launch = None
    contact = None
    dropped = False
    caps = []
    while float(env.sim.data.time) < T_TASK - 1e-9 and not switch.entered:
        for _ in range(v3.N_SUB):
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
                hits, _ = cylinder_contacts(env.sim)
                if hits and contact is None:
                    contact = {"t": now}
                if float(physical_pack(env.sim)["obj_z"]) < Z_ABORT:
                    dropped = True
                    break
            # opportunistic captures after impact, before entry
            if contact is not None and not switch.entered:
                # store rolling capture keyed by time since contact
                caps.append({
                    "t": now,
                    "t_since_contact": now - float(contact["t"]),
                    "capture": take_capture(env),
                    "z": float(obj_pos(env.sim)[2]),
                    "entered": False,
                })
                if len(caps) > 200:
                    caps = caps[-200:]
        if dropped:
            break
        switch.measure(env.sim)
        if launched and switch.p0 is not None:
            switch.update_entry(env.sim)
        if launched and t_launch is not None and float(env.sim.data.time) > t_launch + LAUNCH_WINDOW_S:
            break
    if not switch.entered:
        return []
    t_entry = float(switch.t_entry)
    # pick captures near requested offsets before entry
    out = []
    for off in offsets_s:
        target = t_entry + float(off)
        best = None
        best_d = 1e9
        for c in caps:
            d = abs(float(c["t"]) - target)
            if d < best_d and float(c["t"]) < t_entry - 1e-4:
                best_d = d
                best = c
        if best is not None and best_d < 0.05:
            # start recovery from this capture
            restore_capture(env, best["capture"])
            env.start_recovery()
            ref = capture_ref(env.sim, switch)
            out.append({
                "offset_s": off,
                "t_before_entry": float(t_entry) - float(best["t"]),
                "blob": {
                    "capture": take_capture(env),
                    "ref": ref,
                    "spec": spec,
                    "evidence": {"category": "early_onset_diagnostic", "t_entry": t_entry},
                },
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-videos", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    for d in ("videos", "snapshots", "raw_contact_traces", "raw", "trajectories"):
        (OUT / d).mkdir(exist_ok=True)

    b_a = int(TOTAL_ROLLOUT_BUDGET * PHASE_A_FRAC)
    b_rest = TOTAL_ROLLOUT_BUDGET - b_a
    b_b = [int(b_rest * f) for f in PHASE_B_SPLIT]
    b_b[-1] = b_rest - sum(b_b[:-1])
    write_json(OUT / "raw" / "BUDGET.json", {
        "TOTAL_ROLLOUT_BUDGET": TOTAL_ROLLOUT_BUDGET,
        "WALL_LIMIT_S": WALL_LIMIT_S,
        "PHASE_A": b_a,
        "PHASE_B": {
            "alignment": b_b[0], "force_timing": b_b[1], "early_onset": b_b[2],
            "table_cf": b_b[3], "full_search": b_b[4],
        },
        "recorded_before_execution": True,
    })

    device = torch.device("cpu")
    model, book48 = load_trained_k48(SFT, device)
    table = build_k49_table(book48, config_id=CONFIG_ID)
    checksum = mapping_checksum(table)
    env = make_training_env(device, model=model)
    rng = np.random.default_rng(20261019)
    rollout = [0]
    t_wall0 = time.time()

    def time_left():
        return time.time() - t_wall0 < WALL_LIMIT_S

    geo_audit = []
    impulse_rows = []
    eject_rows = []
    table_tl = []
    window_rows = []
    align_rows = []
    force_rows = []
    early_rows = []
    table_cf_rows = []
    full_rows = []
    eject_ics = []

    # =====================================================================
    # Phase A — contact geometry + impulse + table window audit
    # =====================================================================
    print(f"=== Phase A audit (budget {b_a}) ===", flush=True)
    for ic in BE_ICS:
        if rollout[0] >= b_a or not time_left():
            break
        snap = find_snap(ic)
        if snap is None:
            continue
        dst = OUT / "snapshots" / f"{ic}.pkl"
        if not dst.exists():
            shutil.copy2(snap, dst)
        blob = load_blob(snap)
        st0 = hard_ic_metrics(env, blob)
        for sname, toks in CHASE.items():
            if rollout[0] >= b_a or not time_left():
                break
            acts = toks_to_actions(table, book48, toks)
            out, rows, geos, ejects = run_traced(env, blob, acts, tag=sname)
            rollout[0] += 1
            # summarize dominant site at first hand / major
            dom = "none"
            if geos:
                dom = geos[0].get("dominant_site", "none")
            for g in geos[:3]:
                if g.get("dominant_site") not in ("none", None):
                    dom = g["dominant_site"]
                    break
            geo_audit.append({
                "ic_id": ic, "seed": sname, "high_motion": int(is_high_motion(st0, HM_THR)),
                "v0": st0["v_rel_norm"], "w0": st0["w_rel_norm"],
                "dominant_site_early": dom,
                "com_between_early": geos[0]["com_between"] if geos else None,
                "wrong_geom_at_major": out.get("wrong_geom_at_major"),
                "major_between": out.get("major_between"),
                "ejected": out.get("ejected"),
                "dv_outward": out.get("dv_outward"),
                "major_J_norm": out.get("major_J_norm"),
                "major_J_outward": out.get("major_J_outward"),
                "had_bil": out.get("had_bil"),
                "max_bil_persist_s": out.get("max_bil_persist_s"),
                "reason_fail": out.get("reason_fail"),
            })
            impulse_rows.append({
                "ic_id": ic, "seed": sname,
                "major_t": out.get("major_t"),
                "major_J_norm": out.get("major_J_norm"),
                "major_J_outward": out.get("major_J_outward"),
                "major_sites": out.get("major_sites"),
                "major_grip": out.get("major_grip"),
                "dv_outward": out.get("dv_outward"),
                "ejected": out.get("ejected"),
                "wrong_geom_at_major": out.get("wrong_geom_at_major"),
            })
            for e in ejects:
                eject_rows.append({"ic_id": ic, "seed": sname, **e})
            table_tl.append({
                "ic_id": ic, "seed": sname,
                "t_first_hand": out.get("t_first_hand"),
                "t_first_table": out.get("t_first_table"),
                "t_table_drop": out.get("t_table_drop"),
                "t_end": out.get("t_end"),
                "time_to_irreversible": out.get("time_to_irreversible"),
                "obj_z_end": out.get("obj_z_end"),
                "reason_fail": out.get("reason_fail"),
            })
            # recovery window
            t_hand = out.get("t_first_hand")
            t_irr = out.get("time_to_irreversible")
            window_rows.append({
                "ic_id": ic, "seed": sname,
                "recovery_onset_t0": 0.0,
                "t_first_hand": t_hand,
                "t_first_table": out.get("t_first_table"),
                "t_irreversible": t_irr,
                "window_hand_to_irr": None if t_hand is None or t_irr is None else float(t_irr) - float(t_hand),
                "window_onset_to_irr": t_irr,
                "fastest_intercept_proxy_s": 0.0 if t_hand is not None and float(t_hand) < 1e-6 else t_hand,
                "onset_already_contacting": int(t_hand is not None and float(t_hand) < 1e-6),
            })
            write_json(OUT / "raw_contact_traces" / f"{ic}__{sname}_timeline.json", {
                "out": out, "rows": rows[:80], "geos": geos[:40], "ejects": ejects,
            })
            if int(out.get("ejected") or 0) or int(out.get("wrong_geom_at_major") or 0):
                eject_ics.append(ic)
        print(
            f"  audit {ic} eject={out.get('ejected')} wrong={out.get('wrong_geom_at_major')} "
            f"J={out.get('major_J_norm')} t_tab={out.get('t_first_table')} t_drop={out.get('t_table_drop')}",
            flush=True,
        )

    write_csv(OUT / "CONTACT_GEOMETRY_AUDIT.csv", geo_audit)
    write_csv(OUT / "HAND_OBJECT_IMPULSE_ANALYSIS.csv", impulse_rows)
    write_csv(OUT / "EJECTION_EVENT_ANALYSIS.csv", eject_rows)
    write_csv(OUT / "TABLE_CONTACT_TIMELINE.csv", table_tl)
    write_csv(OUT / "RECOVERY_WINDOW_ANALYSIS.csv", window_rows)

    eject_ics = list(dict.fromkeys(eject_ics)) or BE_ICS[:8]
    # prioritize high-v ejectors
    ranked = []
    for ic in eject_ics:
        rows = [r for r in impulse_rows if r["ic_id"] == ic]
        score = max((float(r.get("major_J_norm") or 0) + 2 * float(r.get("ejected") or 0) for r in rows), default=0)
        ranked.append((ic, score))
    ranked.sort(key=lambda x: -x[1])
    focus = [ic for ic, _ in ranked[:8]]
    print(f"Focus ICs for matched expts: {focus}", flush=True)

    lim1 = b_a + b_b[0]
    lim2 = lim1 + b_b[1]
    lim3 = lim2 + b_b[2]
    lim4 = lim3 + b_b[3]
    lim5 = TOTAL_ROLLOUT_BUDGET

    # =====================================================================
    # Experiment A — grasp alignment before closure
    # =====================================================================
    print(f"=== Exp A alignment (to {lim1}) ===", flush=True)
    modes_A = [
        "A_k49", "B_open_align", "C_wrist_align", "D_translate_in",
        "E_delay_close", "F_soft_then_hard",
    ]
    for ic in focus:
        if rollout[0] >= lim1 or not time_left():
            break
        blob = load_blob(find_snap(ic))
        print(f"  A {ic} roll={rollout[0]}", flush=True)
        for mode in modes_A:
            if rollout[0] >= lim1:
                break
            if mode == "A_k49":
                acts = toks_to_actions(table, book48, CHASE["open_t17"] + [0] * 4 + [12] * 6 + [SECURE_REVERSE_WRIST] * 8)
            else:
                acts = alignment_sequence(env, blob, mode, n=20)
            out, _, _, _ = run_traced(env, blob, acts)
            rollout[0] += 1
            align_rows.append({
                "ic_id": ic, "condition": mode, "protocol": "original",
                "ejected": out.get("ejected"), "dv_outward": out.get("dv_outward"),
                "wrong_geom_at_major": out.get("wrong_geom_at_major"),
                "major_J_norm": out.get("major_J_norm"), "major_J_outward": out.get("major_J_outward"),
                "had_bil": out.get("had_bil"), "max_bil_persist_s": out.get("max_bil_persist_s"),
                "SUCCESS_NEW": out.get("SUCCESS_NEW"),
                "reason_fail": out.get("reason_fail"),
                "win_mean_pos": out.get("win_mean_pos"),
            })
            if out.get("SUCCESS_NEW"):
                full_rows.append({"ic_id": ic, "condition": mode, "protocol": "original", **{k: out[k] for k in (
                    "SUCCESS_NEW", "win_mean_theta", "win_mean_pos", "hold_both_s", "ejected", "major_J_norm")}})
                np.savez_compressed(OUT / "trajectories" / f"{ic}__{mode}_actions.npz",
                                    actions=np.asarray(acts, np.float32))

    write_csv(OUT / "GRIP_ALIGNMENT_ABLATION.csv", align_rows)

    # =====================================================================
    # Experiment B — grip force / timing
    # =====================================================================
    print(f"=== Exp B force/timing (to {lim2}) ===", flush=True)
    for ic in focus:
        if rollout[0] >= lim2 or not time_left():
            break
        blob = load_blob(find_snap(ic))
        print(f"  B {ic} roll={rollout[0]}", flush=True)
        for _ in range(14):
            if rollout[0] >= lim2:
                break
            p = {
                "n_open": int(rng.choice([0, 2, 4, 6])),
                "n_align": int(rng.choice([4, 8, 12])),
                "n_close": int(rng.choice([4, 8, 12])),
                "n_comotion": int(rng.choice([0, 4, 8])),
                "n_brake": int(rng.choice([6, 10, 14])),
                "grip_close": float(rng.choice([0.35, 1.0, 1.0, -0.48])),
                "grip_align": float(rng.choice([-1.0, -0.48, 0.35])),
                "v_scale": float(rng.choice([0.4, 0.7, 1.0])),
                "use_fine_down": bool(rng.random() < 0.3),
            }
            acts = force_timing_sequence(env, blob, p)
            out, _, _, _ = run_traced(env, blob, acts)
            rollout[0] += 1
            force_rows.append({
                "ic_id": ic, "protocol": "original", **p,
                "ejected": out.get("ejected"), "dv_outward": out.get("dv_outward"),
                "major_J_norm": out.get("major_J_norm"), "major_J_outward": out.get("major_J_outward"),
                "wrong_geom_at_major": out.get("wrong_geom_at_major"),
                "had_bil": out.get("had_bil"), "max_bil_persist_s": out.get("max_bil_persist_s"),
                "SUCCESS_NEW": out.get("SUCCESS_NEW"),
                "reason_fail": out.get("reason_fail"),
            })
            if out.get("SUCCESS_NEW"):
                full_rows.append({"ic_id": ic, "condition": "force_timing", "protocol": "original",
                                  "SUCCESS_NEW": 1, "params": json.dumps(p),
                                  "win_mean_pos": out.get("win_mean_pos"),
                                  "win_mean_theta": out.get("win_mean_theta"),
                                  "hold_both_s": out.get("hold_both_s")})

    write_csv(OUT / "GRIP_FORCE_TIMING_ABLATION.csv", force_rows)

    # =====================================================================
    # Experiment early onset (diagnostic)
    # =====================================================================
    print(f"=== Early onset diagnostic (to {lim3}) ===", flush=True)
    for ic in focus[:6]:
        if rollout[0] >= lim3 or not time_left():
            break
        blob = load_blob(find_snap(ic))
        print(f"  early {ic}", flush=True)
        try:
            early_snaps = early_onset_snapshots(env, blob)
        except Exception as e:
            print(f"    early fail {ic}: {e}", flush=True)
            early_snaps = []
        # also evaluate original IC as reference
        acts0 = alignment_sequence(env, blob, "E_delay_close", n=22)
        out0, _, _, _ = run_traced(env, blob, acts0)
        rollout[0] += 1
        early_rows.append({
            "ic_id": ic, "onset": "original_IC", "offset_s": 0.0, "protocol": "original",
            "ejected": out0.get("ejected"), "had_bil": out0.get("had_bil"),
            "max_bil_persist_s": out0.get("max_bil_persist_s"),
            "SUCCESS_NEW": out0.get("SUCCESS_NEW"),
            "t_first_table": out0.get("t_first_table"),
            "time_to_irreversible": out0.get("time_to_irreversible"),
            "reason_fail": out0.get("reason_fail"),
        })
        for es in early_snaps:
            if rollout[0] >= lim3:
                break
            eblob = es["blob"]
            acts = alignment_sequence(env, eblob, "E_delay_close", n=22)
            out, _, _, _ = run_traced(env, eblob, acts)
            rollout[0] += 1
            early_rows.append({
                "ic_id": ic, "onset": "early_diagnostic",
                "offset_s": es["offset_s"], "t_before_entry": es["t_before_entry"],
                "protocol": "early_onset_diagnostic",
                "ejected": out.get("ejected"), "had_bil": out.get("had_bil"),
                "max_bil_persist_s": out.get("max_bil_persist_s"),
                "SUCCESS_NEW": out.get("SUCCESS_NEW"),  # not original-task
                "SUCCESS_NEW_note": "diagnostic_not_original_benchmark",
                "t_first_table": out.get("t_first_table"),
                "time_to_irreversible": out.get("time_to_irreversible"),
                "reason_fail": out.get("reason_fail"),
            })

    write_csv(OUT / "EARLY_ONSET_ABLATION.csv", early_rows)

    # =====================================================================
    # Experiment C — table counterfactual (diagnostic)
    # =====================================================================
    print(f"=== Table counterfactual C1/C2 (to {lim4}) ===", flush=True)
    for ic in focus[:6]:
        if rollout[0] >= lim4 or not time_left():
            break
        blob = load_blob(find_snap(ic))
        print(f"  table {ic}", flush=True)
        for mode in ("C1_original", "C2_lower_table", "C2_no_table_contact"):
            if rollout[0] >= lim4:
                break
            restore_fn = (lambda: None) if mode == "C1_original" else set_table_diagnostic(env.sim, mode)
            try:
                acts = alignment_sequence(env, blob, "E_delay_close", n=22)
                out, _, _, _ = run_traced(env, blob, acts)
                rollout[0] += 1
                table_cf_rows.append({
                    "ic_id": ic, "condition": mode,
                    "protocol": "original" if mode == "C1_original" else "table_counterfactual_diagnostic",
                    "ejected": out.get("ejected"),
                    "had_bil": out.get("had_bil"),
                    "max_bil_persist_s": out.get("max_bil_persist_s"),
                    "t_first_table": out.get("t_first_table"),
                    "t_table_drop": out.get("t_table_drop"),
                    "time_to_irreversible": out.get("time_to_irreversible"),
                    "SUCCESS_NEW": out.get("SUCCESS_NEW") if mode == "C1_original" else 0,
                    "diagnostic_success_flag": out.get("SUCCESS_NEW") if mode != "C1_original" else None,
                    "reason_fail": out.get("reason_fail"),
                    "win_mean_pos": out.get("win_mean_pos"),
                })
            finally:
                restore_fn()

    write_csv(OUT / "TABLE_COUNTERFACTUAL.csv", table_cf_rows)

    # =====================================================================
    # Full expert search on improved-geometry candidates (original only)
    # =====================================================================
    print(f"=== Full search original (to {lim5}) ===", flush=True)
    # candidates: alignment rows with reduced ejection or higher bil persist
    cands = []
    for r in align_rows + force_rows:
        sc = 10 * float(r.get("SUCCESS_NEW") or 0) + 5 * float(r.get("max_bil_persist_s") or 0)
        sc -= 3 * float(r.get("ejected") or 0)
        sc -= 1 * float(r.get("wrong_geom_at_major") or 0)
        cands.append((sc, r))
    cands.sort(key=lambda x: -x[0])
    seen = set()
    for sc, r in cands:
        if rollout[0] >= lim5 or not time_left():
            break
        ic = r["ic_id"]
        if ic in seen and sc < 2:
            continue
        seen.add(ic)
        blob = load_blob(find_snap(ic))
        # refine: delay close + soft grip + fine_down settle
        for label, builder in [
            ("full_delay_fd", lambda: alignment_sequence(env, blob, "E_delay_close", n=24)
             + [FINE_DOWN.copy() for _ in range(10)]
             + toks_to_actions(table, book48, [SECURE_REVERSE_WRIST] * 12)),
            ("full_soft_fd", lambda: alignment_sequence(env, blob, "F_soft_then_hard", n=24)
             + [FINE_DOWN.copy() for _ in range(10)]
             + toks_to_actions(table, book48, [SECURE_REVERSE_WRIST] * 12)),
            ("full_open_fd", lambda: alignment_sequence(env, blob, "B_open_align", n=24)
             + [FINE_DOWN.copy() for _ in range(12)]
             + toks_to_actions(table, book48, [SECURE_REVERSE_WRIST] * 14)),
        ]:
            if rollout[0] >= lim5:
                break
            acts = builder()
            out, _, _, _ = run_traced(env, blob, acts)
            rollout[0] += 1
            row = {
                "ic_id": ic, "condition": label, "protocol": "original",
                "ejected": out.get("ejected"), "had_bil": out.get("had_bil"),
                "max_bil_persist_s": out.get("max_bil_persist_s"),
                "SUCCESS_NEW": out.get("SUCCESS_NEW"),
                "win_mean_theta": out.get("win_mean_theta"),
                "win_mean_pos": out.get("win_mean_pos"),
                "hold_both_s": out.get("hold_both_s"),
                "reason_fail": out.get("reason_fail") or out.get("eval_reason"),
                "major_J_norm": out.get("major_J_norm"),
            }
            force_rows.append(row)  # also log
            if out.get("SUCCESS_NEW"):
                full_rows.append(row)
                np.savez_compressed(OUT / "trajectories" / f"{ic}__{label}_actions.npz",
                                    actions=np.asarray(acts, np.float32))
                print(f"  FULL {ic} {label}", flush=True)

    write_csv(OUT / "FULL_SUCCESS_VERIFICATION.csv", full_rows)
    write_json(OUT / "raw" / "USAGE.json", {
        "rollouts_used": rollout[0], "budget": TOTAL_ROLLOUT_BUDGET,
        "wall_s": time.time() - t_wall0, "n_full_original": len([r for r in full_rows if r.get("protocol") == "original"]),
        "focus_ics": focus,
    })

    if not args.skip_videos:
        _render(env, table, book48, focus[:4], align_rows, full_rows)

    _write_reports(geo_audit, impulse_rows, eject_rows, table_tl, window_rows,
                   align_rows, force_rows, early_rows, table_cf_rows, full_rows, checksum)
    print("Done", OUT / "REPORT.md", flush=True)


def _render(env, table, book48, ics, align_rows, full_rows):
    for ic in ics:
        blob = load_blob(find_snap(ic))
        acts = alignment_sequence(env, blob, "E_delay_close", n=20)
        _vid(env, blob, acts, OUT / "videos" / f"{ic}_delay_close.mp4", f"{ic} delay_close")
        acts_k = toks_to_actions(table, book48, CHASE["open_t17"])
        _vid(env, blob, acts_k, OUT / "videos" / f"{ic}_k49_open.mp4", f"{ic} K49 open")
    for r in full_rows[:3]:
        path = OUT / "trajectories" / f"{r['ic_id']}__{r['condition']}_actions.npz"
        if path.exists():
            acts = list(np.load(path)["actions"])
            blob = load_blob(find_snap(r["ic_id"]))
            _vid(env, blob, acts, OUT / "videos" / f"full_{r['ic_id']}_{r['condition']}.mp4",
                 f"FULL {r['ic_id']}")


def _vid(env, blob, acts, dest, label):
    try:
        restore_capture(env, blob["capture"])
        cam = mjv_camera_from_preset({"lookat": obj_pos(env.sim), "distance": 0.55, "azimuth": 128.0, "elevation": -18.0})
        renderer = mujoco.Renderer(env.sim.model, 360, 480)
        frames = []
        t0 = float(env.sim.data.time)
        for i, a in enumerate(acts):
            if i % 2 == 0:
                cam.lookat[:] = obj_pos(env.sim)
                renderer.update_scene(env.sim.data, cam)
                pack = physical_pack(env.sim)
                frames.append(caption(renderer.render(), f"{label} z={pack['obj_z']:.3f}", float(env.sim.data.time) - t0))
            step_legal(env.sim, np.asarray(a, np.float32))
            if recovery_failed(env.sim, blob["ref"], 0)[0]:
                break
        write_frames_mp4(dest, frames, fps=25)
        renderer.close()
    except Exception as e:
        print("video warn", dest, e, flush=True)


def _mean(xs):
    xs = [float(x) for x in xs if x is not None and x != ""]
    return float(np.mean(xs)) if xs else float("nan")


def _write_reports(geo, impulse, eject, table_tl, window, align, force, early, tcf, full, checksum):
    n = len(geo)
    wrong = sum(1 for r in geo if int(r.get("wrong_geom_at_major") or 0))
    eject_n = sum(1 for r in geo if int(r.get("ejected") or 0))
    sites = {}
    for r in geo:
        s = r.get("dominant_site_early") or "none"
        sites[s] = sites.get(s, 0) + 1
    # outside intended = B/C/D
    outside = sum(1 for r in geo if str(r.get("dominant_site_early", "")).startswith(("B_", "C_", "D_")))
    between0 = sum(1 for r in geo if int(r.get("com_between_early") or 0) == 0)

    win_vals = [float(r["window_onset_to_irr"]) for r in window if r.get("window_onset_to_irr") not in (None, "")]
    already = sum(1 for r in window if int(r.get("onset_already_contacting") or 0))

    def rate(rows, key, pred=None):
        if not rows:
            return float("nan")
        if pred is None:
            return _mean([r.get(key) for r in rows])
        return sum(1 for r in rows if pred(r)) / len(rows)

    align_by = {}
    for r in align:
        align_by.setdefault(r["condition"], []).append(r)

    tcf_by = {}
    for r in tcf:
        tcf_by.setdefault(r["condition"], []).append(r)

    early_orig = [r for r in early if r.get("onset") == "original_IC"]
    early_diag = [r for r in early if r.get("onset") == "early_diagnostic"]

    full_orig = [r for r in full if r.get("protocol", "original") == "original" and int(r.get("SUCCESS_NEW") or 0)]

    # premature closure correlation
    eject_events = eject
    high_grip_eject = [e for e in eject_events if float(e.get("grip_cmd") or 0) > 0.5 and int(e.get("ejected") or 0)]
    wrong_eject = [e for e in eject_events if int(e.get("wrong_geom") or 0) and int(e.get("ejected") or 0)]

    report = f"""# Contact Geometry, Impulse Ejection & Table-Window Diagnostics

**Baseline:** K49 `{CONFIG_ID}` checksum `{checksum}`  
**Interface:** legal AIRBORNE_V3 `a∈[-1,1]^7` / 20 ms. No SFT/GRPO.  
**Diagnostic protocols** (early-onset, table C2) are labeled and **not** original-task experts.

## Budget

| Item | Value |
|------|------:|
| Total rollouts | {TOTAL_ROLLOUT_BUDGET} |
| Wall (s) | {WALL_LIMIT_S} |
| Used | see `raw/USAGE.json` |

## Phase A summary

| Metric | Value |
|--------|------:|
| Audited (ic×seed) | {n} |
| Dominant early site counts | {json.dumps(sites)} |
| Fraction outside intended grip (B/C/D dominant) | {outside / max(n,1):.2f} |
| Fraction COM not between fingers (early) | {between0 / max(n,1):.2f} |
| Wrong-geom at major impulse | {wrong / max(n,1):.2f} |
| Outward ejection events (audit) | {eject_n / max(n,1):.2f} |
| Onset already contacting | {already / max(len(window),1):.2f} |
| Mean time onset→irreversible (s) | {_mean(win_vals):.3f} |

Ejection with wrong-geom: n={len(wrong_eject)}; ejection with high grip_cmd: n={len(high_grip_eject)}.

## Matched alignment (Exp A)

| Condition | n | eject rate | mean bil persist | SUCCESS_NEW |
|-----------|--:|----------:|-----------------:|------------:|
"""
    for m, rows in sorted(align_by.items()):
        report += (
            f"| {m} | {len(rows)} | {rate(rows,'ejected'):.2f} | "
            f"{rate(rows,'max_bil_persist_s'):.3f} | {sum(int(r.get('SUCCESS_NEW') or 0) for r in rows)} |\n"
        )

    report += f"""
## Force/timing (Exp B)

n={len(force)}; eject rate={rate(force,'ejected'):.2f}; bil persist={rate(force,'max_bil_persist_s'):.3f}; SUCCESS_NEW={sum(int(r.get('SUCCESS_NEW') or 0) for r in force)}.

## Early onset (diagnostic)

| Onset | n | eject | bil persist | SUCCESS_NEW (diag flag) |
|-------|--:|------:|------------:|------------------------:|
| original IC | {len(early_orig)} | {rate(early_orig,'ejected'):.2f} | {rate(early_orig,'max_bil_persist_s'):.3f} | {sum(int(r.get('SUCCESS_NEW') or 0) for r in early_orig)} |
| early diagnostic | {len(early_diag)} | {rate(early_diag,'ejected'):.2f} | {rate(early_diag,'max_bil_persist_s'):.3f} | {sum(int(r.get('SUCCESS_NEW') or 0) for r in early_diag)} |

## Table counterfactual

| Condition | n | eject | bil persist | SUCCESS_NEW / diag |
|-----------|--:|------:|------------:|-------------------:|
"""
    for m, rows in sorted(tcf_by.items()):
        snew = sum(int(r.get("SUCCESS_NEW") or 0) for r in rows)
        dflag = sum(int(float(r.get("diagnostic_success_flag") or 0)) for r in rows if r.get("diagnostic_success_flag") not in (None, ""))
        report += f"| {m} | {len(rows)} | {rate(rows,'ejected'):.2f} | {rate(rows,'max_bil_persist_s'):.3f} | {snew}/{dflag} |\n"

    report += f"""
## Original-task FULL SUCCESS_NEW

Count: **{len(full_orig)}** — ICs: {sorted({r['ic_id'] for r in full_orig})}

## Q1–Q12

1. Failed catches contact outside intended gripping region? Dominant early B/C/D fraction **{outside / max(n,1):.2f}**; wrong-geom at major **{wrong / max(n,1):.2f}**. Many failures still have COM between fingers but **wrong pad/edge** site or high impulse.
2. Cylinder ejected by hand–object impulses? Audit eject rate **{eject_n / max(n,1):.2f}**; see `EJECTION_EVENT_ANALYSIS.csv` (J_outward, dv_outward). Simultaneous table contacts separated in impulse logs.
3. Premature/excessive grip contribute? High-grip eject events n={len(high_grip_eject)}; compare Exp A `E_delay_close` / `F_soft_then_hard` vs `A_k49` eject rates in table above.
4. Legal alignment/timing prevent ejection? See Exp A eject-rate deltas; if reduced but bil persist still low, alignment helps geometry but not full capture.
5. Time before table interference / TABLE_DROP? Mean onset→irreversible **{_mean(win_vals):.3f} s**; many ICs already contacting at onset (fraction **{already / max(len(window),1):.2f}**).
6. Original Recovery onset after useful window? If onset_already_contacting high and early-onset improves bil/eject, **yes for those ICs**.
7. Earlier legal intervention improve capture? Compare early diagnostic vs original rows (diagnostic protocol).
8. Removing table interference help (C2)? Compare C2 vs C1 bil/eject; diagnostic only.
9. Improvements surviving in unchanged original environment? Only C1 / Exp A–B / full_search with `protocol=original`.
10. Corrected capture → original SUCCESS_NEW? **{len(full_orig)}** (see `FULL_SUCCESS_VERIFICATION.csv`).
11. Dominant limitations: prioritize evidence from wrong-geom/ejection rates vs table window vs alignment gains — typically **contact geometry + impulse direction**, with **table clearance** as secondary time pressure, and **action representation** (`fine_down`) for pose-near ICs.
12. Most defensible next step: Prefer experts / search that **enforce open-gripper alignment into the jaw volume before high grip**, avoid outer/tip first contact; keep table physics unchanged for acceptance; treat earlier-onset as a separate protocol if it helps; do **not** start SFT/GRPO until original-task capture geometry is fixed.

## Scripts
`scripts/run_contact_geometry_table_window.py`
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")

    fail = f"""# Failure Analysis — Wrong-Geometry Impact & Table Window

## Hypotheses tested

1. Unsuitable grasp location + large impulse ejects the cylinder.
2. Table shortens recoverable window before valid grasp.

## Evidence summary

- Early dominant contact sites: {json.dumps(sites)}
- Wrong-geometry at major hand impulse: {wrong}/{n}
- Outward ejection (audit): {eject_n}/{n}
- Mean recovery window onset→irreversible: {_mean(win_vals):.3f}s
- Already contacting at Recovery IC: {already}/{len(window)}

## Interpretation

Do not equate COM-between-fingers with valid pad contact — site classes A–E are required.
Hand–object impulses are separated from table–object impulses in `HAND_OBJECT_IMPULSE_ANALYSIS.csv`.
Table-free / lowered-table successes are **diagnostic only**.

## Next step

Search/control should prioritize **preventing outward first impulses** (open + align into jaw) over larger chase displacements.
"""
    (OUT / "FAILURE_ANALYSIS.md").write_text(fail, encoding="utf-8")


if __name__ == "__main__":
    main()
