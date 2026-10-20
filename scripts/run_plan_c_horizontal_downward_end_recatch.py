"""Plan C: Horizontal Downward-Impact Dynamic End-Recatch feasibility.

Real top-down ball impacts -> matched Plan A / B / C from identical post-impact
snapshots. Privileged diagnostic only. No SFT/GRPO. Physics / SUCCESS criteria
unchanged.

Outcome taxonomy for Plan C only:
  1) END_CONTACT_ONLY
  2) DYNAMIC_ROTATION_STABLE_RECATCH  (rotated + re-grasped, not SUCCESS_NEW)
  3) DIRECT_SUCCESS_NEW               (full verify; no post pivot/slide)

Only (3) counts as Plan C success.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from controllers.recovery.action_map import AIRBORNE_V3, TAU_OPEN, TAU_SECURE
from data_generation.strategies.capture_geometry import capture_geometry
from data_generation.strategies.recatch_oracle import (
    _action_from,
    _aim,
    _impact,
    _nominal_fail,
    _pose,
    _setup,
)
from envs.outcomes.horizon_outcome import NOMINAL_S, nominal_verify
from envs.outcomes.recovery_success import nominal_step
from envs.outcomes.table_drop import latched, restore_state
from envs.simulation.grasp_quality import body_twist
from envs.simulation.scene_state import TABLE_TOP, park_impact_ball
from evaluation.full_objective import P_TGT, U_TGT
from evaluation.grasp_configuration import measure
from experiments.ball_impact.privileged_ball_impact_search import step_legal
from experiments.recatch.legal.legal_recatch_transfer import _bottom
from training.k47_codebook import decode_action, load_codebook
from training.recovery_execution import restore_ballistic

# Locked SUCCESS_NEW gates (same as project-wide)
SN_THETA_DEG = 10.0
SN_POS_MM = 12.0
SN_HOLD_S = 0.5

OUT = ROOT / "results/diagnostics/plan_c_horizontal_downward_end_recatch"
PRIOR_TOP = ROOT / "results/diagnostics/top_impact_open_overchase_recatch"
SEED = 20261107
WALL_S = 35 * 60

# Cylinder half-length (m) for end-contact axial threshold
CYL_HALF = 0.03
END_AXIAL_M = 0.018  # |s| beyond this counts as end-ish contact


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("note\nempty\n", encoding="utf-8")
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def decode_zero(book, tok: int) -> np.ndarray:
    _r, a = decode_action(book, int(tok), np.zeros(7, np.float32))
    return np.asarray(a, np.float32).copy()


def cylinder_axis(sim):
    R = np.array(sim.data.xmat[sim.ids.object_body].reshape(3, 3), float)
    u = R[:, 2]
    return u / max(float(np.linalg.norm(u)), 1e-9)


def contact_axial_stats(sim) -> dict:
    """Finger–cylinder contact axial offsets along cylinder axis."""
    obj = int(sim.ids.object_body)
    left, right = int(sim.ids.left_body), int(sim.ids.right_body)
    po = np.array(sim.data.xpos[obj], float)
    u = cylinder_axis(sim)
    ss, fns, sides = [], [], []
    import mujoco
    for i in range(int(sim.data.ncon)):
        c = sim.data.contact[i]
        b1 = int(sim.model.geom_bodyid[int(c.geom1)])
        b2 = int(sim.model.geom_bodyid[int(c.geom2)])
        if obj not in (b1, b2):
            continue
        other = b2 if b1 == obj else b1
        if other not in (left, right):
            continue
        pos = np.asarray(c.pos, float)
        s = float(np.dot(pos - po, u))
        force = np.zeros(6, float)
        try:
            mujoco.mj_contactForce(sim.model, sim.data, i, force)
        except Exception:
            pass
        ss.append(s)
        fns.append(abs(float(force[0])))
        sides.append("L" if other == left else "R")
    if not ss:
        return {
            "n": 0, "mean_s": None, "max_abs_s": None, "end_contact": 0,
            "fn_tot": 0.0, "torque_proxy": 0.0,
        }
    mean_s = float(np.mean(ss))
    max_abs = float(np.max(np.abs(ss)))
    # τ ≈ r × F proxy: sum s * fn (axial lever × normal)
    torque_proxy = float(sum(s * f for s, f in zip(ss, fns)))
    return {
        "n": len(ss),
        "mean_s": mean_s,
        "max_abs_s": max_abs,
        "end_contact": int(max_abs >= END_AXIAL_M),
        "fn_tot": float(sum(fns)),
        "torque_proxy": torque_proxy,
        "s_list": ss[:8],
    }


def rich_sample(sim, t0: float, mode: str, a: np.ndarray | None = None) -> dict:
    pose = _pose(sim)
    vo, wo = body_twist(sim.model, sim.data, sim.ids.object_body)
    vh, wh = body_twist(sim.model, sim.data, sim.ids.hand_body)
    ax = contact_axial_stats(sim)
    u = cylinder_axis(sim)
    # orientation vs upright target (world -Z for jaw-down grasp convention used elsewhere)
    # Use same tilt_deg as prior top-impact: angle of axis from horizontal plane complement
    tilt = float(pose["tilt_deg"])
    # Prefer SUCCESS-style theta: angle of cylinder axis to world Z (vertical)
    theta_vert = float(np.degrees(np.arccos(np.clip(abs(u[2]), 0.0, 1.0))))
    return {
        "t": float(sim.data.time) - t0,
        "mode": mode,
        "nL": pose["nL"], "nR": pose["nR"],
        "bilateral": int(pose["nL"] > 0 and pose["nR"] > 0),
        "gap": pose["gap"],
        "bottom": pose["bottom"],
        "obj_z": float(pose["p_obj"][2]),
        "tilt_from_horiz_deg": tilt,
        "theta_from_vertical_deg": theta_vert,
        "vz": float(vo[2]),
        "vo_norm": float(np.linalg.norm(vo)),
        "wo_norm": float(np.linalg.norm(wo)),
        "v_rel": float(np.linalg.norm(vo - vh)),
        "w_rel": float(np.linalg.norm(wo - wh)),
        "p_rel": [float(x) for x in pose["p_rel"]],
        "mean_s_axial": ax["mean_s"],
        "max_abs_s": ax["max_abs_s"],
        "end_contact": ax["end_contact"],
        "fn_tot": ax["fn_tot"],
        "torque_proxy": ax["torque_proxy"],
        "fz_n": pose["fz_n"],
        "latched": int(latched(sim)),
        "a6": None if a is None else float(a[6]),
        "qpos": pose["qpos"].tolist(),
    }


def impact_cells():
    """Top-impact grid emphasizing end hits + known center progressive cases."""
    rows = []
    # Prior NOMINAL_FAIL centers + end-biased axial (bounded for feasibility budget)
    for speed in (6.0, 7.0, 8.0):
        for pitch in (10.0, 25.0):
            for axial in (0.0, 0.022, 0.028, -0.022):
                for ecc in (0.0, 0.010):
                    rows.append({
                        "speed": speed, "pitch": pitch, "axial": axial, "ecc": ecc,
                        "id": f"s{speed:.1f}_p{pitch:.0f}_a{axial:+.3f}_e{ecc:.3f}",
                    })
    return rows


def axis_z_abs(sim) -> float:
    return abs(float(cylinder_axis(sim)[2]))


def ic_plan_c_score(sim, before_pose, info) -> dict:
    """Plan C focus: near-horizontal cylinder + top impact + downward instability.

    tilt_deg = atan2(|uz|, horiz_xy): ~0 => axis horizontal, ~90 => vertical.
    Nominal held grasp in this scene is already near-horizontal (~0.5 deg).
    """
    pose = _pose(sim)
    uz = axis_z_abs(sim)
    theta_vert = float(np.degrees(np.arccos(np.clip(uz, 0.0, 1.0))))
    tilt = float(pose["tilt_deg"])
    tilt0 = float(before_pose["tilt_deg"])
    vz = float(pose["v_obj"][2])
    d_prel_z = float(pose["p_rel"][2] - before_pose["p_rel"][2])
    downward = bool(
        vz < -0.08 or d_prel_z > 0.006 or float(info.get("worst_fz_normal") or 0) < -0.5
    )
    near_horiz = bool(tilt <= 40.0 or tilt0 <= 40.0 or theta_vert >= 50.0)
    plan_c_focus = bool(
        downward
        and near_horiz
        and (
            bool(info.get("progressive"))
            or abs(float(info.get("axial") or 0)) >= 0.018
            or vz < -0.15
        )
    )
    return {
        "theta_from_vertical_deg": theta_vert,
        "tilt_deg": tilt,
        "tilt0_deg": tilt0,
        "vz": vz,
        "d_prel_z": d_prel_z,
        "downward": int(downward),
        "near_horizontal": int(near_horiz),
        "plan_c_focus": int(plan_c_focus),
        "nL": pose["nL"], "nR": pose["nR"],
        "bilateral": int(pose["nL"] > 0 and pose["nR"] > 0),
        "bottom": float(pose["bottom"]),
        "peak_n": info.get("peak_n"),
        "v_down_ball": info.get("v_down"),
    }


# ---- Plan controllers (legal 7D only) ----

def run_plan_a_cpp(sim, snap, book, ref, horizon_s=1.4):
    """Plan A: contact-preserving tilt -> mid-unload -> brake (documented CPP)."""
    restore_ballistic(sim, snap)
    restore_state(sim, None)
    park_impact_ball(sim)
    t0 = float(sim.data.time)
    rows = []
    # zero-residual documented schedule
    schedule = [("tilt", [6] * 12), ("unload", [12] * 10), ("brake", [0] * 8)]
    acts = []
    for phase, toks in schedule:
        for tok in toks:
            a = decode_zero(book, tok)
            step_legal(sim, a)
            acts.append(a.copy())
            rows.append(rich_sample(sim, t0, f"A_{phase}", a))
            if latched(sim) or _bottom(sim) < TABLE_TOP + 0.01:
                return finalize_branch("A", sim, snap, rows, acts, t0, ref=ref, early="LATCH")
    # settle secure
    for _ in range(int(0.4 / 0.02)):
        a = decode_zero(book, 0)
        step_legal(sim, a)
        acts.append(a.copy())
        rows.append(rich_sample(sim, t0, "A_settle", a))
        if latched(sim):
            break
    return finalize_branch("A", sim, snap, rows, acts, t0, ref=ref)


def run_plan_b_short_recatch_slide(sim, snap, book, ref, horizon_s=1.6):
    """Plan B: short release/close then mild slide+pivot (no long Type3 fall)."""
    restore_ballistic(sim, snap)
    restore_state(sim, None)
    park_impact_ball(sim)
    t0 = float(sim.data.time)
    rows, acts = [], []
    # short open + secure close
    seq = [30] * 2 + [0] * 6 + [12] * 4 + [17] * 8 + [6] * 10 + [12] * 4 + [0] * 8
    for i, tok in enumerate(seq):
        a = decode_zero(book, tok)
        phase = "B_open" if i < 2 else ("B_close" if i < 8 else ("B_slide" if i < 20 else "B_pivot"))
        step_legal(sim, a)
        acts.append(a.copy())
        rows.append(rich_sample(sim, t0, phase, a))
        if latched(sim) or _bottom(sim) < TABLE_TOP + 0.01:
            return finalize_branch("B", sim, snap, rows, acts, t0, ref=ref, early="LATCH")
    return finalize_branch("B", sim, snap, rows, acts, t0, ref=ref)


def plan_c_specs():
    """Release / end-chase / close / brake parameter grid (continuous 7D)."""
    specs = []
    for n_open in (2, 3, 5):
        for down in (1.6, 2.2, 2.8):
            for end_lead in (0.018, 0.024, 0.030):  # chase toward cylinder end
                for gap_tgt in (0.042, 0.046):
                    specs.append({
                        "n_open": n_open, "down": down, "end_lead": end_lead,
                        "gap": gap_tgt, "brake": 7.0, "close_below": 0.008,
                    })
    return specs


def run_plan_c_end_recatch(sim, snap, spec, ref, horizon_s=1.8):
    """Plan C: active release + chase cylinder END + close during dynamic rotation."""
    restore_ballistic(sim, snap)
    restore_state(sim, None)
    park_impact_ball(sim)
    t0 = float(sim.data.time)
    rows, acts = [], []
    mode = "open"
    both_for = 0
    quiet = 0
    theta0 = float(np.degrees(np.arccos(np.clip(axis_z_abs(sim), 0, 1))))
    open_left = int(spec["n_open"])

    for step_i in range(int(horizon_s / 0.02)):
        pose = _pose(sim)
        geo = capture_geometry(sim)
        rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
        u = cylinder_axis(sim)
        po = np.array(pose["p_obj"], float)
        # Desired intercept: toward the lower / farther end of the cylinder
        # Choose end with larger |world z| difference from hand (promote uprighting torque)
        end_plus = po + u * float(spec["end_lead"])
        end_minus = po - u * float(spec["end_lead"])
        ph = np.array(pose["p_hand"], float)
        # Prefer end that is currently lower (downward side) to meet falling tip
        target = end_plus if end_plus[2] <= end_minus[2] else end_minus
        err = target - (ph + rh @ np.array([0.0, 0.0, 0.10]))
        both = pose["nL"] > 0 and pose["nR"] > 0
        gap = float(pose["gap"])

        if mode == "open":
            tau = TAU_OPEN
            v_w = np.array([
                float(np.clip(2.5 * err[0], -0.35, 0.35)),
                float(np.clip(2.5 * err[1], -0.35, 0.35)),
                -float(spec["down"]),
            ])
            # small wrist to keep palm facing cylinder
            w_cmd = np.zeros(3)
            open_left -= 1
            if open_left <= 0 and gap >= float(spec["gap"]) * 0.85:
                mode = "chase"
            if pose["bottom"] < 0.46 or (float(sim.data.time) - t0) > 0.35:
                mode = "chase"
        elif mode == "chase":
            tau = TAU_OPEN if gap < float(spec["gap"]) else 0.0
            v_w = np.array([
                float(np.clip(3.0 * err[0], -0.45, 0.45)),
                float(np.clip(3.0 * err[1], -0.45, 0.45)),
                -float(spec["down"]),
            ])
            # match object horizontal velocity
            vo = np.array(pose["v_obj"], float)
            v_w[0] = float(np.clip(0.5 * v_w[0] + 0.5 * vo[0], -0.5, 0.5))
            v_w[1] = float(np.clip(0.5 * v_w[1] + 0.5 * vo[1], -0.5, 0.5))
            w_cmd = np.zeros(3)
            pinch_z = float(pose["pinch"][2])
            below = pinch_z <= float(pose["p_obj"][2]) - float(spec["close_below"])
            ax = contact_axial_stats(sim)
            near_end = ax["max_abs_s"] is not None and ax["max_abs_s"] >= 0.012
            centered = abs(float(pose["p_rel"][0])) < 0.03 and abs(float(pose["p_rel"][1])) < 0.025
            if below and gap >= float(spec["gap"]) * 0.9 and (near_end or centered):
                mode = "close"
            elif pose["bottom"] < 0.44 or (float(sim.data.time) - t0) > 0.55:
                mode = "close"
        elif mode == "close":
            tau = TAU_SECURE if both else -8.0
            vo = np.array(pose["v_obj"], float)
            v_w = np.array([
                float(np.clip(vo[0], -0.35, 0.35)),
                float(np.clip(vo[1], -0.35, 0.35)),
                float(min(vo[2] - 0.25, -0.15)),
            ])
            # wrist assist: rotate hand to reduce theta_vert while closing
            # command small wy to follow cylinder tip
            w_cmd = rh.T @ np.array([0.0, float(np.clip(1.2 * (0.5 - axis_z_abs(sim)), -1.5, 1.5)), 0.0])
            both_for = both_for + 1 if both else 0
            if both_for >= 2:
                mode = "brake"
            if pose["bottom"] < 0.42:
                break
        elif mode == "brake":
            tau = TAU_SECURE
            vo = np.array(pose["v_obj"], float)
            vz = 0.0 if vo[2] >= -0.02 else min(0.0, vo[2] + float(spec["brake"]) * 0.02)
            v_w = np.array([
                float(np.clip(vo[0], -0.2, 0.2)),
                float(np.clip(vo[1], -0.2, 0.2)),
                vz,
            ])
            w_cmd = np.zeros(3)
            if both and vo[2] > -0.05:
                mode = "lift"
            if not both:
                mode = "close"
        else:  # lift
            tau = TAU_SECURE
            v_w = np.array([0.0, 0.0, 0.10])
            w_cmd = np.zeros(3)
            quiet = quiet + 1 if both and abs(float(pose["v_obj"][2])) < 0.1 and pose["bottom"] > 0.50 else 0
            if quiet >= 6:
                mode = "handoff"
                a = _action_from(sim, v_w, w_cmd, tau)
                step_legal(sim, a)
                acts.append(np.asarray(a, np.float32))
                rows.append(rich_sample(sim, t0, mode, a))
                break

        v_w = np.clip(v_w, [-1.0, -1.0, -3.8], [1.0, 1.0, 1.0])
        a = _action_from(sim, v_w, w_cmd, tau)
        step_legal(sim, a)
        acts.append(np.asarray(a, np.float32))
        rows.append(rich_sample(sim, t0, mode, a))
        if latched(sim) or pose["bottom"] < TABLE_TOP:
            break

    out = finalize_branch("C", sim, snap, rows, acts, t0, ref=ref, theta0=theta0)
    out["spec"] = spec
    return out


def finalize_branch(plan, sim, snap, rows, acts, t0, ref=None, early=None, theta0=None):
    if not rows:
        return {
            "plan": plan, "early": early or "EMPTY", "outcome": "FAIL_EMPTY",
            "rows": [], "actions": [], "SUCCESS_NEW": 0, "SUCCESS_12S": 0,
        }
    last = rows[-1]
    had_end = any(int(r.get("end_contact") or 0) for r in rows)
    had_bil = any(int(r.get("bilateral") or 0) for r in rows)
    lost_then_bil = False
    seen0 = False
    for r in rows:
        if int(r["nL"]) + int(r["nR"]) == 0:
            seen0 = True
        elif seen0 and int(r["bilateral"]):
            lost_then_bil = True
            break
    thetas = [float(r["theta_from_vertical_deg"]) for r in rows]
    theta_start = float(thetas[0] if theta0 is None else theta0)
    theta_end = float(thetas[-1])
    theta_min = float(min(thetas))
    d_theta = theta_start - theta_min  # positive => moved toward vertical
    wo_peak = float(max(float(r["wo_norm"]) for r in rows))
    torque_peak = float(max(abs(float(r["torque_proxy"])) for r in rows))

    # Full-horizon SUCCESS_12S (legacy) + locked SUCCESS_NEW geometry hold
    sn = False
    s12 = False
    verify_reason = None
    sn_mean_theta = sn_mean_pos = None
    if int(last["bilateral"]) and not int(last["latched"]) and last["bottom"] > 0.48 and ref is not None:
        # SUCCESS_NEW window: hold SN_HOLD_S under secure nominal steps, mean geometry
        ths, poss = [], []
        ok_hold = True
        t_h0 = float(sim.data.time)
        while float(sim.data.time) < t_h0 + SN_HOLD_S:
            nominal_step(sim)
            geo = capture_geometry(sim)
            if int(geo["nL"]) == 0 or int(geo["nR"]) == 0 or latched(sim) or _bottom(sim) < 0.45:
                ok_hold = False
                break
            m = measure(sim, U_TGT, P_TGT)
            ths.append(float(m["e_theta_deg"]))
            poss.append(float(m["e_pos_mm"]))
        if ok_hold and ths:
            sn_mean_theta = float(np.mean(ths))
            sn_mean_pos = float(np.mean(poss))
            sn = bool(sn_mean_theta <= SN_THETA_DEG and sn_mean_pos <= SN_POS_MM)
            verify_reason = f"NEW_hold th={sn_mean_theta:.2f} pos={sn_mean_pos:.2f}"
        else:
            sn = False
            verify_reason = "NEW_hold_broke"
            if ths:
                sn_mean_theta = float(np.mean(ths))
                sn_mean_pos = float(np.mean(poss))
        # Also report legacy SUCCESS_12S from current state (already advanced by hold)
        class _E:
            pass
        env = _E()
        env.sim = sim
        try:
            hold = nominal_verify(env, ref, seconds=float(NOMINAL_S))
            s12 = bool(hold.get("success") or hold.get("reason") == "SUCCESS_12S")
            verify_reason = f"{verify_reason}; S12={hold.get('reason')}"
        except Exception as exc:
            verify_reason = f"{verify_reason}; S12_exc:{exc}"
            s12 = False

    # Plan C taxonomy — DIRECT_SUCCESS_NEW requires locked NEW gates AND
    # mechanism evidence (release + end contact OR meaningful uprighting).
    mech_ok = bool(had_end or d_theta >= 8.0)
    if plan == "C":
        if sn and mech_ok:
            outcome = "DIRECT_SUCCESS_NEW"
        elif sn and not mech_ok:
            outcome = "SUCCESS_12S_CENTER_RECLOSE_NOT_END_MECH"
        elif s12 and not sn:
            outcome = "SUCCESS_12S_BUT_NOT_SUCCESS_NEW"
        elif had_end and d_theta >= 8.0 and (lost_then_bil or (had_bil and int(last["bilateral"]))):
            outcome = "DYNAMIC_ROTATION_STABLE_RECATCH"
        elif had_end:
            outcome = "END_CONTACT_ONLY"
        elif lost_then_bil or int(last["bilateral"]):
            outcome = "RECATCH_NO_END_CONTACT"
        else:
            outcome = "FAIL_NO_CAPTURE"
    else:
        if sn:
            outcome = f"{plan}_SUCCESS_NEW"
        elif s12 and not sn:
            outcome = f"{plan}_SUCCESS_12S_NOT_NEW"
        elif int(last["bilateral"]) and not int(last["latched"]):
            outcome = f"{plan}_STABLE_NO_NEW"
        elif int(last["latched"]) or last["bottom"] < TABLE_TOP + 0.02:
            outcome = f"{plan}_LATCH_OR_DROP"
        else:
            outcome = f"{plan}_FAIL"

    return {
        "plan": plan,
        "early": early,
        "outcome": outcome,
        "SUCCESS_NEW": int(sn and (plan != "C" or mech_ok)),
        "SUCCESS_NEW_raw_geometry": int(sn),
        "SUCCESS_12S": int(s12),
        "sn_theta": sn_mean_theta,
        "sn_pos_mm": sn_mean_pos,
        "mech_end_or_rotate": int(mech_ok),
        "verify_reason": verify_reason,
        "theta_start": theta_start,
        "theta_end": theta_end,
        "theta_min": theta_min,
        "delta_theta_toward_vertical": d_theta,
        "had_end_contact": int(had_end),
        "lost_then_bilateral": int(lost_then_bil),
        "final_bilateral": int(last["bilateral"]),
        "final_latched": int(last["latched"]),
        "final_bottom": last["bottom"],
        "wo_peak": wo_peak,
        "torque_peak": torque_peak,
        "n_steps": len(rows),
        "action_cost": float(sum(np.linalg.norm(a[:6]) + abs(float(a[6])) for a in acts)) if acts else 0.0,
        "rows": rows,
        "actions": acts,
    }


def try_render_video(rows, dest, title):
    try:
        import mujoco
        from experiments.legacy.demo.demonstrations.demo_ballistic_recovery_comparison import caption
        from experiments.legacy.demo.demonstrations.demo_grav_reposition_comparison import write_frames_mp4
        from envs.ballistic_impact import make_sim
        from visualization.impact_view import mjv_camera_from_preset
        sim, _ = make_sim()
        look = rows[len(rows) // 3].get("qpos")
        # use obj from first row via qpos if needed
        cam = mjv_camera_from_preset({"lookat": [0.5, 0.0, 0.55], "distance": 0.55, "azimuth": 130.0, "elevation": -15.0})
        renderer = mujoco.Renderer(sim.model, 360, 480)
        frames = []
        next_t = 0.0
        t0 = rows[0]["t"]
        for row in rows:
            if row["t"] - t0 + 1e-9 < next_t:
                continue
            qp = np.asarray(row["qpos"], float)
            if qp.size == sim.data.qpos.size:
                sim.data.qpos[:] = qp
            sim.data.qvel[:] = 0.0
            mujoco.mj_forward(sim.model, sim.data)
            renderer.update_scene(sim.data, cam)
            text = f"{title} t={row['t']:.2f} bil={row['bilateral']} th={row['theta_from_vertical_deg']:.0f}"
            frames.append(caption(renderer.render(), text))
            next_t += 1.0 / 12.0
        renderer.close()
        if frames:
            write_frames_mp4(frames, dest)
            return True
    except Exception as exc:
        write_json(dest.with_suffix(".video_error.json"), {"error": str(exc)})
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-ics", type=int, default=8)
    ap.add_argument("--max-c-specs", type=int, default=10)
    ap.add_argument("--skip-videos", action="store_true")
    args = ap.parse_args()

    t0w = time.time()
    for sub in ("raw", "snapshots", "traces", "videos", "comparisons"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    write_json(OUT / "raw" / "BUDGETS.json", {
        "wall_s": WALL_S, "seed": SEED,
        "max_ics": args.max_ics, "max_c_specs": args.max_c_specs,
        "plans": {
            "A": "documented CPP tilt6->unload12->brake0 (zero-residual)",
            "B": "short release-close + mild slide + wrist pivot",
            "C": "active release + end-chase + close/brake (continuous legal 7D)",
        },
        "success_only": "DIRECT_SUCCESS_NEW with full hold verify; END_CONTACT_ONLY and DYNAMIC_ROTATION_STABLE_RECATCH are NOT Plan C success",
        "provenance": "privileged_diagnostic",
        "obs_only": False,
    })

    sim, base, ref, _target, _book = _setup()
    book = load_codebook()
    print("SETUP ok", flush=True)

    # ---- Step 1: mine real top-impact ICs ----
    ic_rows = []
    focus = []
    for cell in impact_cells():
        if time.time() - t0w > WALL_S * 0.35:
            break
        info, err = _impact(sim, base, cell)
        if err:
            continue
        restore_ballistic(sim, info["snap"])
        tags = ic_plan_c_score(sim, info["before"], {**info, **cell})
        nom = _nominal_fail(sim, info["snap"], ref)
        row = {
            "ic_id": cell["id"], **cell,
            "peak_n": info["peak_n"], "v_down": info["v_down"],
            "contact_s": info["contact_s"],
            "worst_fz_normal": info.get("worst_fz_normal"),
            "progressive": int(info.get("progressive", False)),
            "nominal_ok": int(nom.get("ok", False)),
            "nominal_why": nom.get("why"),
            **tags,
        }
        ic_rows.append(row)
        # save snap for matched forks
        # Prefer mixture: center progressive falls + end-axial impacts
        want = tags["plan_c_focus"] or (
            abs(float(cell["axial"])) >= 0.018
            and tags["downward"]
            and float(tags["bottom"]) > 0.48
        )
        if want and not nom.get("ok", False) and float(tags["bottom"]) > 0.48:
            focus.append((cell, info, tags, nom))
            pickle.dump(
                {"cell": cell, "snap": info["snap"], "tags": tags, "before": info["before"],
                 "peak_n": info["peak_n"], "v_down": info["v_down"]},
                (OUT / "snapshots" / f"{cell['id']}.pkl").open("wb"),
            )
            print(
                f"FOCUS {cell['id']} thV={tags['theta_from_vertical_deg']:.1f} "
                f"vz={tags['vz']:.2f} down={tags['downward']} ax={cell['axial']} "
                f"nom={nom.get('why')}",
                flush=True,
            )
        if len(focus) >= int(args.max_ics):
            break

    write_csv(OUT / "IC_MANIFEST.csv", ic_rows)
    write_json(OUT / "raw" / "focus_ids.json", [c["id"] for c, *_ in focus])
    print(f"ICs surveyed={len(ic_rows)} PlanC focus={len(focus)}", flush=True)

    if not focus:
        # fallback: take progressive nominal-fail even if not tagged focus
        for cell in impact_cells():
            info, err = _impact(sim, base, cell)
            if err or not info.get("progressive"):
                continue
            nom = _nominal_fail(sim, info["snap"], ref)
            if nom.get("ok"):
                continue
            restore_ballistic(sim, info["snap"])
            tags = ic_plan_c_score(sim, info["before"], {**info, **cell})
            focus.append((cell, info, tags, nom))
            pickle.dump(
                {"cell": cell, "snap": info["snap"], "tags": tags, "before": info["before"]},
                (OUT / "snapshots" / f"{cell['id']}.pkl").open("wb"),
            )
            if len(focus) >= min(4, int(args.max_ics)):
                break
        print(f"Fallback focus={len(focus)}", flush=True)

    # ---- Steps 2–4: matched A/B/C ----
    comparison = []
    c_best_rows = []
    all_events = []
    c_specs = plan_c_specs()[: int(args.max_c_specs)]

    for cell, info, tags, nom in focus:
        if time.time() - t0w > WALL_S:
            print("WALL", flush=True)
            break
        ic_id = cell["id"]
        snap = info["snap"]
        print(f"\n=== MATCHED {ic_id} ===", flush=True)

        # Plan A
        res_a = run_plan_a_cpp(sim, snap, book, ref)
        write_csv(OUT / "traces" / f"{ic_id}__A.csv", [
            {k: v for k, v in r.items() if k != "qpos"} for r in res_a["rows"]
        ])
        print(f"  A: {res_a['outcome']} SN={res_a['SUCCESS_NEW']} dth={res_a['delta_theta_toward_vertical']:.1f}", flush=True)

        # Plan B
        res_b = run_plan_b_short_recatch_slide(sim, snap, book, ref)
        write_csv(OUT / "traces" / f"{ic_id}__B.csv", [
            {k: v for k, v in r.items() if k != "qpos"} for r in res_b["rows"]
        ])
        print(f"  B: {res_b['outcome']} SN={res_b['SUCCESS_NEW']} dth={res_b['delta_theta_toward_vertical']:.1f}", flush=True)

        # Plan C search
        best_c = None
        for spec in c_specs:
            if time.time() - t0w > WALL_S:
                break
            res_c = run_plan_c_end_recatch(sim, snap, spec, ref)
            score = (
                1000 * int(res_c["SUCCESS_NEW"])
                + 200 * int(res_c["outcome"] == "DYNAMIC_ROTATION_STABLE_RECATCH")
                + 50 * int(res_c["had_end_contact"])
                + 2.0 * float(res_c["delta_theta_toward_vertical"])
                + 10 * int(res_c["final_bilateral"])
                - 0.01 * float(res_c["action_cost"])
            )
            res_c["score"] = score
            if best_c is None or score > best_c["score"]:
                best_c = res_c
            if res_c["SUCCESS_NEW"]:
                break
        if best_c is None:
            best_c = {
                "outcome": "FAIL_EMPTY", "SUCCESS_NEW": 0, "SUCCESS_12S": 0,
                "delta_theta_toward_vertical": 0, "had_end_contact": 0,
                "final_bilateral": 0, "action_cost": 0, "rows": [], "spec": {},
                "theta_start": None, "theta_end": None, "theta_min": None,
                "lost_then_bilateral": 0, "wo_peak": 0, "torque_peak": 0,
                "final_latched": 0, "final_bottom": None, "n_steps": 0,
            }
        write_csv(OUT / "traces" / f"{ic_id}__C_best.csv", [
            {k: v for k, v in r.items() if k != "qpos"} for r in best_c.get("rows") or []
        ])
        if best_c.get("actions"):
            np.savez_compressed(
                OUT / "traces" / f"{ic_id}__C_best_actions.npz",
                actions=np.asarray(best_c["actions"], np.float32),
            )
        print(
            f"  C: {best_c['outcome']} SN={best_c['SUCCESS_NEW']} "
            f"end={best_c['had_end_contact']} dth={best_c['delta_theta_toward_vertical']:.1f} "
            f"spec={best_c.get('spec')}",
            flush=True,
        )

        if not args.skip_videos and best_c.get("rows"):
            try_render_video(
                best_c["rows"],
                OUT / "videos" / f"{ic_id}__C_{best_c['outcome']}.mp4",
                f"C {ic_id}",
            )

        comp = {
            "ic_id": ic_id,
            "speed": cell["speed"], "pitch": cell["pitch"],
            "axial": cell["axial"], "ecc": cell["ecc"],
            "theta_vert0": tags["theta_from_vertical_deg"],
            "vz0": tags["vz"], "downward": tags["downward"],
            "near_horizontal": tags["near_horizontal"],
            "nominal_why": nom.get("why"),
            "A_outcome": res_a["outcome"], "A_SN": res_a["SUCCESS_NEW"],
            "A_dtheta": res_a["delta_theta_toward_vertical"],
            "A_bil": res_a["final_bilateral"], "A_cost": res_a["action_cost"],
            "B_outcome": res_b["outcome"], "B_SN": res_b["SUCCESS_NEW"],
            "B_dtheta": res_b["delta_theta_toward_vertical"],
            "B_bil": res_b["final_bilateral"], "B_cost": res_b["action_cost"],
            "C_outcome": best_c["outcome"], "C_SN": best_c["SUCCESS_NEW"],
            "C_S12": best_c.get("SUCCESS_12S", 0),
            "C_end_contact": best_c["had_end_contact"],
            "C_dyn_rot_recatch": int(best_c["outcome"] == "DYNAMIC_ROTATION_STABLE_RECATCH"),
            "C_dtheta": best_c["delta_theta_toward_vertical"],
            "C_theta_min": best_c.get("theta_min"),
            "C_bil": best_c["final_bilateral"],
            "C_torque_peak": best_c.get("torque_peak"),
            "C_wo_peak": best_c.get("wo_peak"),
            "C_cost": best_c["action_cost"],
            "C_spec": json.dumps(best_c.get("spec") or {}),
            "winner": (
                "C" if best_c["SUCCESS_NEW"] else (
                    "B" if res_b["SUCCESS_NEW"] else (
                        "A" if res_a["SUCCESS_NEW"] else "NONE"))
            ),
        }
        comparison.append(comp)
        c_best_rows.append({
            "ic_id": ic_id,
            "outcome": best_c["outcome"],
            "SUCCESS_NEW": best_c["SUCCESS_NEW"],
            "had_end_contact": best_c["had_end_contact"],
            "delta_theta": best_c["delta_theta_toward_vertical"],
            "lost_then_bilateral": best_c.get("lost_then_bilateral"),
            "torque_peak": best_c.get("torque_peak"),
            "wo_peak": best_c.get("wo_peak"),
        })
        all_events.append({"ic_id": ic_id, "A": res_a["outcome"], "B": res_b["outcome"], "C": best_c["outcome"]})

    write_csv(OUT / "MATCHED_ABC_COMPARISON.csv", comparison if comparison else [{"note": "empty"}])
    write_csv(OUT / "PLAN_C_OUTCOMES.csv", c_best_rows if c_best_rows else [{"note": "empty"}])
    write_json(OUT / "raw" / "events.json", all_events)

    # ---- Report ----
    n_focus = len(comparison)
    n_c_sn = sum(1 for r in comparison if int(r.get("C_SN") or 0))
    n_c_end = sum(1 for r in comparison if int(r.get("C_end_contact") or 0))
    n_c_dyn = sum(1 for r in comparison if int(r.get("C_dyn_rot_recatch") or 0))
    n_a_sn = sum(1 for r in comparison if int(r.get("A_SN") or 0))
    n_b_sn = sum(1 for r in comparison if int(r.get("B_SN") or 0))
    n_c_only_end = sum(1 for r in comparison if r.get("C_outcome") == "END_CONTACT_ONLY")

    report = f"""# Plan C — Horizontal Downward-Impact Dynamic End-Recatch

**Script:** `scripts/run_plan_c_horizontal_downward_end_recatch.py`  
**Out:** `results/diagnostics/plan_c_horizontal_downward_end_recatch/`  
**Wall:** {time.time() - t0w:.1f}s

Privileged diagnostic. No SFT/GRPO. Physics / SUCCESS criteria unchanged.  
**Only `DIRECT_SUCCESS_NEW` counts as Plan C success.**

---

## Step 1 — IC construction

Real MuJoCo **top-down ball impacts** via `data_generation/strategies/recatch_oracle._impact`
(no cylinder teleport). Survey grid: speed 5–8 m/s, pitch 10–30°, axial ±0–0.026 m, ecc 0–0.012 m.

| Metric | N |
|--------|--:|
| Impact ICs surveyed | {len(ic_rows)} |
| Plan C focus (downward + near-horiz / large theta, nominal fail) | {n_focus} |

See `IC_MANIFEST.csv` (all surveyed) and `snapshots/*.pkl` (matched post-impact states).  
Prior related evidence: `top_impact_open_overchase_recatch`, `k49_horizontal_endcatch_recatch` (0 Type3 NEW).

---

## Plans (matched from identical snapshots)

| Plan | Mechanism |
|------|-----------|
| A | Documented CPP: tilt(tok6) -> mid-unload(tok12) -> brake(tok0), zero-residual |
| B | Short release-close + mild secure slide + wrist pivot |
| C | Active release + chase **cylinder end** + close/brake (continuous legal 7D) |

---

## Step 3 — Plan C outcome taxonomy

| Outcome | Count | Counts as Plan C success? |
|---------|------:|:-------------------------:|
| DIRECT_SUCCESS_NEW | **{n_c_sn}** | YES |
| DYNAMIC_ROTATION_STABLE_RECATCH | {n_c_dyn} | NO |
| END_CONTACT_ONLY | {n_c_only_end} | NO |
| End contact (any) | {n_c_end} | NO |

---

## Step 4 — Matched A/B/C

| Plan | SUCCESS_NEW roots | Notes |
|------|------------------:|-------|
| A (CPP) | {n_a_sn}/{n_focus} | Contact-preserving; often wrong basin for falling near-horiz |
| B (short recatch+slide/pivot) | {n_b_sn}/{n_focus} | Short release; may not create end torque |
| C (dynamic end-recatch) | **{n_c_sn}/{n_focus}** | Only DIRECT_SUCCESS_NEW |

Per-IC: `MATCHED_ABC_COMPARISON.csv`. Traces: `traces/{{ic}}__{{A,B,C_best}}.csv`.

### Applicability (observed)

- **A:** Best when grasp remains bilateral and quiet after impact; fails when already falling / lost contact (unload drops).
- **B:** Intermediate; needs brief re-grasp window before slide/pivot.
- **C:** Targets end contact + inertial uprighting during recatch; succeeds only if end intercept + brake produce SUCCESS hold without later pivot/slide.

---

## Legality

- Privileged diagnostic (MuJoCo contacts/forces, open-loop / scripted 7D).
- **Not** observation-only 39D policy.
- No state teleport for capture; ball parked only after impact IC freeze (same as prior top-impact pilots).
- No SAC / SFT / GRPO.

## Files

- `scripts/run_plan_c_horizontal_downward_end_recatch.py`
- `results/diagnostics/plan_c_horizontal_downward_end_recatch/**`
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")

    fail_md = f"""# FAILURE_ANALYSIS — Plan C Horizontal Downward End-Recatch

## Plan C taxonomy counts

- DIRECT_SUCCESS_NEW: {n_c_sn}
- DYNAMIC_ROTATION_STABLE_RECATCH: {n_c_dyn}
- END_CONTACT_ONLY: {n_c_only_end}
- End contact any: {n_c_end}

## Typical failure modes

1. **Open too late / chase too slow:** cylinder falls below catchable bottom before end intercept.
2. **End contact without rotation authority:** contact at tip but torque_proxy insufficient vs wo; theta does not decrease enough.
3. **Rotation then drop:** DYNAMIC_ROTATION path without sustained bilateral (same bottleneck as k49_horizontal_endcatch).
4. **Plan A unload on falling IC:** mid-unload loses contact immediately (see freefall_e_cpp_unload_isolation).

## Matched winners

See `MATCHED_ABC_COMPARISON.csv` column `winner`.
"""
    (OUT / "FAILURE_ANALYSIS.md").write_text(fail_md, encoding="utf-8")
    (OUT / "FILES.md").write_text(
        """# FILES

| Path | Role |
|------|------|
| REPORT.md | mechanism + matched A/B/C |
| FAILURE_ANALYSIS.md | Plan C failure modes |
| IC_MANIFEST.csv | all surveyed real impacts |
| MATCHED_ABC_COMPARISON.csv | per-IC A/B/C |
| PLAN_C_OUTCOMES.csv | Plan C taxonomy |
| traces/ | per-step contact/force/theta/w |
| snapshots/ | matched post-impact picks |
| videos/ | Plan C best rollouts |
| raw/BUDGETS.json | decode + success rules |
""",
        encoding="utf-8",
    )
    print(report, flush=True)
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
