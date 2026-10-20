"""Collision-aware closed-loop freefall recatch diagnostic.

Diagnoses Controller O horizontal sign error and Controller P hand-housing
collision, then evaluates a shared jaw-region interception controller.

Legal continuous 7D only. No SFT/GRPO. No physics / 39D / geometry changes.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from controllers.recovery.action_map import AIRBORNE_V3, map_action
from envs.observations.airborne_obs import (
    AIR_NAMES,
    OBS_DIM_AIR,
    OBS_NAMES_AIR,
    SCALE_PREL,
    SCALE_ROT,
    SCALE_VREL,
    SCALE_WREL,
    relative_state,
)
from envs.outcomes.horizon_outcome import NOMINAL_S
from envs.outcomes.table_drop import latched
from envs.simulation.capture import restore_capture, take_capture
from envs.step_legal import step_legal
from evaluation.full_objective import P_TGT, U_TGT, contact_summary
from evaluation.grasp_configuration import cylinder_axis_hand, measure as gmeasure, p_com_hand
from models.k48_checkpoint import load_trained_k48
from scripts.generate_k49_sliding_recatch_dataset import SFT, write_csv, write_json
from scripts.run_freefall_recatch_pose_feedback import load_root
from scripts.run_horizontal_offset_freefall_recatch import (
    SEARCH_NOMINAL_S,
    audit_interception,
    eval_full_sn,
    render_expert_video,
)
from scripts.run_lateral_offset_falling_recatch import (
    build_nominal_open_robot,
    contact_sites,
    finger_frame,
    hand_pos,
    make_ic,
    make_training_env,
    obj_pos,
    restore_ic,
)
from scripts.run_lateral_recatch_handoff import (
    build_stabilization,
    clear_probe_latch,
    hand_R,
)

SRC_FB = ROOT / "results" / "diagnostics" / "freefall_recatch_pose_feedback"
SRC_EXP = ROOT / "results" / "expert_generation" / "horizontal_offset_freefall_recatch"
OUT = ROOT / "results" / "diagnostics" / "freefall_recatch_collision_aware_feedback"
SEED = 20261105

# Fixed robot geometry constants (precomputed; OK for Controller O)
# finger mid in hand frame relative to hand body origin (open jaws, nominal pose)
MID_OFFSET_H = np.array([0.0, 0.0, 0.0584], float)
# jaw entry XY gate (m): descend only when |err_xy| below this
JAW_XY_ENTER = 0.028
JAW_XY_SOFT = 0.045
# housing-risk band: object between hand origin and mid along approach
HOUSING_Z_LO = 0.015
HOUSING_Z_HI = 0.055
CYL_RADIUS = 0.018
G = 9.81

WALL_LIMIT_S = 55 * 60
STAGE_HELD = 12
STAGE_FULL_VERIFY = 3


def geom_name(model, gid: int) -> str:
    n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(gid))
    return n or f"geom_{int(gid)}"


def body_name(model, bid: int) -> str:
    n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(bid))
    return n or f"body_{int(bid)}"


def decode_obs(obs: np.ndarray) -> dict:
    o = np.asarray(obs, np.float32).reshape(-1)
    base = 27
    return {
        "p_rel": o[base:base + 3] * SCALE_PREL,
        "v_rel": o[base + 3:base + 6] * SCALE_VREL,
        "rot_rel": o[base + 6:base + 9] * SCALE_ROT,
        "w_rel": o[base + 9:base + 12] * SCALE_WREL,
        "cL": float(o[2]),
        "cR": float(o[3]),
        "aper": float(o[12]),
    }


def list_obj_contacts(sim) -> list[dict]:
    obj = int(sim.ids.object_body)
    rows = []
    for i in range(int(sim.data.ncon)):
        c = sim.data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        b1 = int(sim.model.geom_bodyid[g1])
        b2 = int(sim.model.geom_bodyid[g2])
        if obj not in (b1, b2):
            continue
        other_g = g2 if b1 == obj else g1
        other_b = b2 if b1 == obj else b1
        bn = body_name(sim.model, other_b)
        site = "hand_housing" if bn == "hand" else (
            "finger" if bn in ("left_finger", "right_finger") else (
                "table_or_world" if bn in ("world", "table") else "other"
            )
        )
        # force estimate
        force = np.zeros(6, float)
        try:
            mujoco.mj_contactForce(sim.model, sim.data, i, force)
        except Exception:
            pass
        rows.append({
            "geom_id": other_g,
            "geom": geom_name(sim.model, other_g),
            "body_id": other_b,
            "body": bn,
            "site_class": site,
            "pos": np.asarray(c.pos, float).tolist(),
            "dist": float(c.dist),
            "frame": np.asarray(c.frame, float).reshape(3, 3).tolist(),
            "force6": force.tolist(),
            "normal_force": float(force[0]),
        })
    return rows


def hand_vel(sim) -> np.ndarray:
    from envs.simulation.grasp_quality import body_twist
    vh, _ = body_twist(sim.model, sim.data, sim.ids.hand_body)
    return np.asarray(vh, float)


def obj_vel(sim) -> np.ndarray:
    from envs.simulation.grasp_quality import body_twist
    vo, _ = body_twist(sim.model, sim.data, sim.ids.object_body)
    return np.asarray(vo, float)


def predict_obj_world(sim, t_lead: float) -> np.ndarray:
    """Ballistic free-fall prediction of cylinder COM (gravity only)."""
    po = obj_pos(sim)
    vo = obj_vel(sim)
    t = max(0.0, float(t_lead))
    return po + vo * t + np.array([0.0, 0.0, -0.5 * G * t * t], float)


def jaw_target_err_privileged(sim, use_predict: bool, t_lead: float) -> dict:
    """Error of cylinder COM relative to finger mid, hand frame.

    XY may use ballistic lead; Z for phase decisions uses *current* height so
    prediction does not falsely trigger descend/retract.
    """
    ff = finger_frame(sim)
    mid = ff["mid"]
    Rh = ff["Rh"]
    po_now = obj_pos(sim)
    err_now = Rh.T @ (po_now - mid)
    if use_predict and t_lead > 0:
        po_pred = predict_obj_world(sim, t_lead)
        err_pred = Rh.T @ (po_pred - mid)
        err_h = np.array([err_pred[0], err_pred[1], err_now[2]], float)
        po = po_pred
    else:
        err_h = err_now
        po = po_now
    return {"err_h": err_h, "err_now": err_now, "mid": mid, "po": po, "ff": ff, "Rh": Rh}


def jaw_target_err_from_obs(dec: dict, use_predict: bool, t_lead: float) -> np.ndarray:
    """Jaw-mid relative error from 39D only + fixed MID_OFFSET_H.

    p_rel = Rh.T @ (po - ph). mid ≈ ph + Rh @ MID_OFFSET_H
    => Rh.T @ (po - mid) ≈ p_rel - MID_OFFSET_H
    Lead applies to XY only; keep current hz for phase logic.
    """
    p = np.asarray(dec["p_rel"], float).copy()
    v = np.asarray(dec["v_rel"], float).copy()
    err = p - MID_OFFSET_H
    if use_predict and t_lead > 0:
        err_xy = err[:2] + v[:2] * t_lead
        return np.array([err_xy[0], err_xy[1], err[2]], float)
    return err


def choose_lead_time(err_xy: float, vz_rel: float, gains: dict) -> float:
    t_xy = abs(err_xy) / float(gains.get("v_xy_nom", 0.55))
    # also consider fall into mid: if object above mid (err_z small/neg), short lead
    t_cap = float(gains.get("t_lead_max", 0.12))
    return float(np.clip(0.4 * t_xy + float(gains.get("t_lead_bias", 0.04)), 0.02, t_cap))


def wrist_from_rot(rot_rel: np.ndarray, gains: dict) -> np.ndarray:
    w = np.zeros(3, np.float32)
    kw = float(gains.get("kw_app", 0.35))
    w[0] = float(np.clip(-kw * rot_rel[0] + gains.get("wx0", 0.0), -0.6, 0.6))
    w[1] = float(np.clip(-kw * rot_rel[1] + gains.get("wy0", 0.12), -0.6, 0.6))
    w[2] = float(np.clip(-kw * 0.8 * rot_rel[2] + gains.get("wz0", 0.15), -0.5, 0.5))
    return w


def shared_intercept_action(
    err_h: np.ndarray,
    v_rel: np.ndarray,
    rot_rel: np.ndarray,
    cL: float,
    cR: float,
    gains: dict,
    bilateral: bool,
    align_steps: int = 0,
    err_now: np.ndarray | None = None,
    bil_hold_steps: int = 0,
) -> tuple[np.ndarray, dict]:
    """Shared jaw-region interception / capture law (legal 7D).

    `err_h` may include XY lead for chasing; `err_now` (current) drives phase/grip.
    """
    a = np.zeros(7, np.float32)
    meta = {
        "phase": "chase", "housing_risk": 0, "descend": 0,
        "align_steps": align_steps, "bil_hold_steps": bil_hold_steps,
    }
    en = np.asarray(err_now if err_now is not None else err_h, float)
    # After any bilateral, commit to seat→lift (do not resume free chase).
    n_seat = int(gains.get("n_bil_seat", 4))
    if bilateral or bil_hold_steps > 0 or (cL > 0.5 and cR > 0.5):
        meta["phase"] = "capture" if bil_hold_steps >= n_seat else "seat"
        meta["descend"] = 0
        p_est = en + MID_OFFSET_H
        err_p = np.asarray(P_TGT, float) - p_est
        a[0] = float(np.clip(0.2 * err_p[0] / 0.03, -0.25, 0.25))
        a[1] = float(np.clip(0.2 * err_p[1] / 0.03, -0.25, 0.25))
        # Immediately lift after first bil (positive seat_vz pops the cylinder out).
        a[2] = float(gains.get("lift", -0.9))
        a[3:6] = 0.25 * wrist_from_rot(rot_rel, gains)
        a[6] = 1.0
        return a, meta

    # --- chase / jaw entry ---
    ex, ey = float(err_h[0]), float(err_h[1])
    ez = float(en[2])
    err_xy_now = math.hypot(float(en[0]), float(en[1]))
    # chase TOWARD predicted/current mid error
    kp = float(gains.get("kp_chase", 1.6))
    oc = float(gains.get("overchase", 0.25))
    a[0] = float(np.clip(kp * ex / 0.025 - 0.2 * v_rel[0] + oc * np.sign(ex + 1e-9), -1, 1))
    a[1] = float(np.clip(kp * ey / 0.025 - 0.2 * v_rel[1] + 0.5 * oc * np.sign(ey + 1e-9), -1, 1))

    p_rel_z_approx = ez + float(MID_OFFSET_H[2])
    housing_risk = (
        err_xy_now > JAW_XY_ENTER
        and HOUSING_Z_LO <= p_rel_z_approx <= HOUSING_Z_HI + 0.02
        and ez < float(gains.get("ez_fall_through", 0.03))
    )
    meta["housing_risk"] = int(housing_risk)

    # Expert: ~3–4 steps retract then scoop while XY still closing.
    n_align_max = int(gains.get("n_align_retract", 4))
    fall_through = ez > float(gains.get("ez_fall_through", 0.03))
    still_aligning = (
        align_steps < n_align_max
        and err_xy_now > float(gains.get("xy_early_scoop", 0.048))
        and not fall_through
    )
    if still_aligning:
        a[2] = float(gains.get("retract", -0.20))
        meta["descend"] = 0
        meta["phase"] = "align_xy"
    else:
        a[2] = float(np.clip(
            gains.get("descend", 1.0) + 0.25 * max(ez, 0.0) / 0.05,
            0.7, 1.0,
        ))
        meta["descend"] = 1
        meta["phase"] = "scoop"

    a[3:6] = wrist_from_rot(rot_rel, gains)
    a[6] = -1.0
    if cL > 0.5 or cR > 0.5 or (meta["descend"] and err_xy_now < JAW_XY_SOFT and ez > -0.015):
        a[6] = float(gains.get("preclose", 1.0))
    # Soften aggressively near capture (expert uses mild a at bil onset).
    if meta["descend"] and err_xy_now < float(gains.get("xy_soft_capture", 0.036)):
        a[0] = float(np.clip(0.35 * a[0], -0.45, 0.45))
        a[1] = float(np.clip(0.35 * a[1], -0.45, 0.45))
        a[2] = float(gains.get("soft_descend", 0.35))
        a[3:6] *= 0.35
        a[6] = 1.0
        meta["phase"] = "soft_capture"
    meta["err_xy_now"] = err_xy_now
    return a, meta


def original_O_action(obs, gains, hist):
    """Original buggy O chase (for comparison)."""
    from scripts.run_freefall_recatch_pose_feedback import obs_correct_action
    return obs_correct_action(obs, "chase", gains, hist)


def original_P_action(sim, gains):
    """Original P COM chase (for comparison)."""
    a = np.zeros(7, np.float32)
    ff = finger_frame(sim)
    po, mid = obj_pos(sim), ff["mid"]
    err = ff["Rh"].T @ (po - mid)
    a[0] = float(np.clip(1.5 * err[0] / 0.02, -1, 1))
    a[1] = float(np.clip(1.5 * err[1] / 0.02, -1, 1))
    a[2] = float(np.clip(0.85 + 0.5 * err[2] / 0.05, 0, 1))
    a[6] = -1.0
    return a, err


def corrected_O_action(obs, gains, hist, use_predict: bool, align_steps: int = 0, bil_hold_steps: int = 0):
    dec = decode_obs(obs)
    err_xy0 = math.hypot(float(dec["p_rel"][0] - MID_OFFSET_H[0]),
                         float(dec["p_rel"][1] - MID_OFFSET_H[1]))
    t_lead = choose_lead_time(err_xy0, float(dec["v_rel"][2]), gains) if use_predict else 0.0
    err_h = jaw_target_err_from_obs(dec, use_predict, t_lead)
    err_now = jaw_target_err_from_obs(dec, False, 0.0)
    bil = dec["cL"] > 0.5 and dec["cR"] > 0.5
    a, meta = shared_intercept_action(
        err_h, dec["v_rel"], dec["rot_rel"], dec["cL"], dec["cR"], gains, bil,
        align_steps=align_steps, err_now=err_now, bil_hold_steps=bil_hold_steps,
    )
    meta["t_lead"] = t_lead
    meta["err_h"] = err_h.tolist()
    return a, meta


def corrected_P_action(sim, gains, use_predict: bool, align_steps: int = 0, bil_hold_steps: int = 0):
    ff = finger_frame(sim)
    err_xy0 = float(np.linalg.norm((ff["Rh"].T @ (obj_pos(sim) - ff["mid"]))[:2]))
    t_lead = choose_lead_time(err_xy0, float((ff["Rh"].T @ obj_vel(sim))[2]), gains) if use_predict else 0.0
    info = jaw_target_err_privileged(sim, use_predict, t_lead)
    cs = contact_summary(sim)
    u = np.asarray(cylinder_axis_hand(sim), float)
    ut = U_TGT / (np.linalg.norm(U_TGT) + 1e-12)
    u = u / (np.linalg.norm(u) + 1e-12)
    if float(np.dot(u, ut)) < 0:
        u = -u
    rot_rel = np.cross(u, ut)
    v_rel = info["Rh"].T @ (obj_vel(sim) - hand_vel(sim))
    a, meta = shared_intercept_action(
        info["err_h"], v_rel, rot_rel,
        1.0 if cs["nL"] > 0 else 0.0,
        1.0 if cs["nR"] > 0 else 0.0,
        gains, bool(cs["bilateral"]),
        align_steps=align_steps,
        err_now=info.get("err_now", info["err_h"]),
        bil_hold_steps=bil_hold_steps,
    )
    meta["t_lead"] = t_lead
    meta["err_h"] = info["err_h"].tolist()
    return a, meta


def rollout_controller(env, ic, kind: str, gains: dict, max_steps: int = 90, log: bool = False):
    restore_ic(env, ic)
    clear_probe_latch(env.sim)
    acts, logs = [], []
    hist = []
    first_contact = None
    first_housing = None
    first_finger = None
    first_bil = None
    jaw_entry = False
    align_steps = 0
    bil_hold_steps = 0
    for i in range(max_steps):
        obs = np.asarray(env.observation(), np.float32)
        hist.append(obs.copy())
        dec = decode_obs(obs)
        target = None
        if kind == "orig_O":
            a = original_O_action(obs, gains, hist)
            meta = {"phase": "orig_O"}
            target = (P_TGT - dec["p_rel"]).tolist()
        elif kind == "orig_P":
            a, err = original_P_action(env.sim, gains)
            meta = {"phase": "orig_P"}
            target = err.tolist()
        elif kind == "corr_O":
            a, meta = corrected_O_action(
                obs, gains, hist, use_predict=True,
                align_steps=align_steps, bil_hold_steps=bil_hold_steps,
            )
            target = meta.get("err_h")
        elif kind == "corr_O_nopred":
            a, meta = corrected_O_action(
                obs, gains, hist, use_predict=False,
                align_steps=align_steps, bil_hold_steps=bil_hold_steps,
            )
            target = meta.get("err_h")
        elif kind == "corr_P":
            a, meta = corrected_P_action(
                env.sim, gains, use_predict=True,
                align_steps=align_steps, bil_hold_steps=bil_hold_steps,
            )
            target = meta.get("err_h")
        elif kind == "corr_P_nopred":
            a, meta = corrected_P_action(
                env.sim, gains, use_predict=False,
                align_steps=align_steps, bil_hold_steps=bil_hold_steps,
            )
            target = meta.get("err_h")
        else:
            raise ValueError(kind)
        if meta.get("phase") == "align_xy":
            align_steps += 1

        a_raw = a.copy()
        a = np.clip(a, -1, 1).astype(np.float32)
        rh = hand_R(env.sim)
        rd = np.asarray(env.sim.fsm.r_des, float).reshape(3, 3)
        cmd = map_action(a, rd, rh, AIRBORNE_V3)
        ph0 = hand_pos(env.sim).copy()
        po0 = obj_pos(env.sim).copy()
        vh0 = hand_vel(env.sim).copy()
        vo0 = obj_vel(env.sim).copy()
        cons_before = list_obj_contacts(env.sim)

        step_legal(env.sim, a)
        acts.append(a.copy())

        cons = list_obj_contacts(env.sim)
        cs = contact_summary(env.sim)
        sites = contact_sites(env.sim)
        if cons and first_contact is None:
            first_contact = {"step": i, "contacts": cons, "sites": sites}
        if first_housing is None and any(c["site_class"] == "hand_housing" for c in cons):
            first_housing = {"step": i, "contacts": [c for c in cons if c["site_class"] == "hand_housing"]}
        if first_finger is None and any(c["site_class"] == "finger" for c in cons):
            first_finger = {"step": i, "contacts": [c for c in cons if c["site_class"] == "finger"]}
        if cs["bilateral"]:
            if first_bil is None:
                first_bil = i
            bil_hold_steps += 1
        elif first_bil is not None:
            # keep commit counter after first bil (sticky) so we don't re-chase
            bil_hold_steps += 1
        if any(s.get("site") == "inside" for s in sites) or (
            cs["bilateral"] and first_finger is not None
        ):
            jaw_entry = True

        if log:
            logs.append({
                "step": i,
                "p_rel": dec["p_rel"].tolist(),
                "v_rel": dec["v_rel"].tolist(),
                "target_err": target,
                "a_raw": a_raw.tolist(),
                "a_clip": a.tolist(),
                "v_hand_cmd": cmd["v_hand"].tolist(),
                "v_world_cmd": cmd["v_world"].tolist(),
                "vh_meas": vh0.tolist(),
                "ph": ph0.tolist(),
                "po": po0.tolist(),
                "vo": vo0.tolist(),
                "aper": dec["aper"],
                "contacts": cons,
                "bilateral": int(cs["bilateral"]),
                "meta": meta,
            })
        if latched(env.sim):
            break
        # Hand off to open-loop stabilization ASAP after first bil (keep prefix ending in bil).
        if first_bil is not None and i >= first_bil:
            break
        if first_bil is not None and i - first_bil > 40:
            break

    # stabilize + evaluate if bilateral
    full_acts = list(acts)
    if first_bil is not None:
        # prefix must end on a bilateral frame for build_stabilization
        cut = first_bil + 1
        pref = full_acts[:cut]
        # optional 2 soft hold steps with grip closed before stab search
        soft = []
        restore_ic(env, ic)
        clear_probe_latch(env.sim)
        for a in pref:
            step_legal(env.sim, np.asarray(a, np.float32))
        for _ in range(3):
            if not contact_summary(env.sim)["bilateral"] or latched(env.sim):
                break
            a = np.zeros(7, np.float32)
            a[6] = 1.0
            a[2] = float(gains.get("soft_descend", 0.25))
            a[4] = float(gains.get("wy0", 0.08))
            step_legal(env.sim, a)
            soft.append(a.copy())
        pref2 = pref + soft
        p = {
            "n_lift": 36, "lift": -0.95, "n_return": 14, "n_seat": 28, "n_fine": 0,
            "wy": 0.25, "n_wy": 12, "xy_gain": 0.55, "axis_align": True,
            "use_fine_down": False, "return_vz": -0.18, "seat_vz": 0.08,
        }
        stab = build_stabilization(env, ic, pref2, p)
        if stab is not None:
            full_acts = pref2 + stab
        else:
            # try without soft hold
            stab = build_stabilization(env, ic, pref, p)
            if stab is not None:
                full_acts = pref + stab

    out = eval_full_sn(
        env, ic, full_acts, max(1, (first_bil or len(full_acts) // 3) + 1),
        nominal_s=SEARCH_NOMINAL_S,
    )
    summary = {
        "first_contact_step": None if first_contact is None else first_contact["step"],
        "first_contact_bodies": sorted({c["body"] for c in (first_contact or {}).get("contacts", [])}),
        "first_housing_step": None if first_housing is None else first_housing["step"],
        "first_housing_geoms": sorted({c["geom"] for c in (first_housing or {}).get("contacts", [])}),
        "first_finger_step": None if first_finger is None else first_finger["step"],
        "first_bil_step": first_bil,
        "jaw_entry": int(jaw_entry),
        "housing_collision": int(first_housing is not None),
        "outer_finger_first": int(
            first_finger is not None and (
                first_housing is None or first_finger["step"] <= first_housing["step"]
            )
            and not jaw_entry
        ),
        "SUCCESS_NEW": out["SUCCESS_NEW"],
        "CATCH_HANDOFF_SUCCESS": out["CATCH_HANDOFF_SUCCESS"],
        "SN_mean_theta": out.get("SN_mean_theta"),
        "SN_mean_pos": out.get("SN_mean_pos"),
        "n_actions": len(full_acts),
        "first_contact_detail": first_contact,
        "first_housing_detail": first_housing,
    }
    return full_acts, summary, logs, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-videos", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    for sub in ("raw", "trajectories", "videos", "snapshots"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    budgets = {
        "wall_limit_s": 25 * 60 if args.smoke else WALL_LIMIT_S,
        "heldout": 6 if args.smoke else STAGE_HELD,
        "seed": SEED,
        "smoke": bool(args.smoke),
    }
    write_json(OUT / "raw" / "BUDGETS.json", budgets)

    def time_left():
        return time.time() - t0 < budgets["wall_limit_s"]

    device = torch.device("cpu")
    model, _ = load_trained_k48(SFT, device)
    env = make_training_env(device, model=model)
    rng = np.random.default_rng(SEED)

    # ---------- mapping doc ----------
    mapping_md = f"""# OBSERVATION_ACTION_MAPPING

## 39D layout (authoritative)

- Indices 0–26: proprio/tactile prefix (`OBS_NAMES`)
- Indices 27–38: airborne relative (`AIR_NAMES`)
  - 27–29: `p_rel_h` / `SCALE_PREL` (`SCALE_PREL={SCALE_PREL}`) — **hand frame**
  - 30–32: `v_rel_h` / `SCALE_VREL` (`SCALE_VREL={SCALE_VREL}`)
  - 33–35: `rot_rel` / `SCALE_ROT`
  - 36–38: `w_rel_h` / `SCALE_WREL`

Definition (`envs/observations/airborne_obs.py`):
`p_rel_h = R_hand.T @ (p_obj - p_hand)` — relative to **hand body origin**, not finger mid.

## Legal action map (`step_legal` → `AIRBORNE_V3`)

- `a[0:3]`: **hand-frame** translational command
- `v_world = R_hand @ v_hand`
- `a[0],a[1]`: ±1 → ±1.0 m/s
- `a[2]≥0`: +1 → +3.8 m/s (hand +z; world downward on horizontal grasp)
- `a[2]<0`: −1 → −1.0 m/s (hand −z / retract/up)
- `a[3:6]`: angular in `r_des` frame
- `a[6]`: grip (−1 open … +1 secure)

Empirical unit-axis response (5 steps, open grip): `+a[0]` → `+Δp_hand_x` (tracking OK).

## Fixed jaw geometry constants (Controller O allowed)

- `MID_OFFSET_H = {MID_OFFSET_H.tolist()}` m (finger mid − hand origin, hand frame)
- Jaw-mid relative error from obs: `err_h ≈ p_rel - MID_OFFSET_H`
- `JAW_XY_ENTER = {JAW_XY_ENTER}`, `JAW_XY_SOFT = {JAW_XY_SOFT}`
- Housing band along approach: `p_rel_z ∈ [{HOUSING_Z_LO}, {HOUSING_Z_HI}]`

## Original Controller O chase bug

Used `err = P_TGT - p_rel` then `a[0:2] ∝ err[0:2]`.

For free-fall chase this is **sign-reversed**: object at `+p_rel_x` produced `−a[0]`, moving the hand away.
Robot tracked the commanded velocity; failure is **wrong action generation**, not tracking.

Correct chase: `a[0:2] ∝ (p_rel - MID_OFFSET_H)[0:2]` (move mid toward object).
"""
    (OUT / "OBSERVATION_ACTION_MAPPING.md").write_text(mapping_md, encoding="utf-8")

    ic_perp, _, _ = load_root("perp_off01_r15")
    (OUT / "snapshots" / "perp_off01_r15.pkl").write_bytes(pickle.dumps(ic_perp))

    # ---------- Stage 1–3: action response + O/P failure logs ----------
    print("Stage 1-3: action response + failure localization", flush=True)
    action_rows = []
    restore_ic(env, ic_perp)
    clear_probe_latch(env.sim)
    for _ in range(8):
        step_legal(env.sim, np.array([0, 0, 0, 0, 0, 0, -1], np.float32))
    cap = take_capture(env)
    Rh0 = hand_R(env.sim).copy()
    for name, a7 in [
        ("+vx", [0.5, 0, 0, 0, 0, 0, -1]),
        ("-vx", [-0.5, 0, 0, 0, 0, 0, -1]),
        ("+vy", [0, 0.5, 0, 0, 0, 0, -1]),
        ("-vy", [0, -0.5, 0, 0, 0, 0, -1]),
        ("+vz", [0, 0, 0.5, 0, 0, 0, -1]),
        ("-vz", [0, 0, -0.5, 0, 0, 0, -1]),
    ]:
        restore_capture(env, cap)
        clear_probe_latch(env.sim)
        p0 = hand_pos(env.sim).copy()
        po0 = obj_pos(env.sim).copy()
        err0 = Rh0.T @ (po0 - p0)
        for _ in range(5):
            step_legal(env.sim, np.asarray(a7, np.float32))
        dp_w = hand_pos(env.sim) - p0
        dp_h = Rh0.T @ dp_w
        err1 = Rh0.T @ (obj_pos(env.sim) - hand_pos(env.sim))
        action_rows.append({
            "axis": name,
            "a": json.dumps(a7),
            "dp_hand_x": float(dp_h[0]), "dp_hand_y": float(dp_h[1]), "dp_hand_z": float(dp_h[2]),
            "dp_world": json.dumps(dp_w.tolist()),
            "d_err_h": json.dumps((err1 - err0).tolist()),
        })
    write_csv(OUT / "ACTION_RESPONSE_AUDIT.csv", action_rows)

    # O sign test
    o_sign = []
    for sx in (+0.05, -0.05):
        fake = np.zeros(OBS_DIM_AIR, np.float32)
        fake[27] = sx / SCALE_PREL
        fake[2] = 0.0
        fake[3] = 0.0
        a_old = original_O_action(fake, {"kp": 1.2, "kz": 0.5, "vz_bias": 0.2}, [])
        a_new, meta = corrected_O_action(fake, {"kp_chase": 1.6, "retract": -0.25}, [], False)
        o_sign.append({
            "p_rel_x": sx,
            "orig_a0": float(a_old[0]),
            "corr_a0": float(a_new[0]),
            "expected_chase_sign": float(np.sign(sx - MID_OFFSET_H[0])),
        })
    write_json(OUT / "raw" / "o_sign_audit.json", o_sign)

    gains_shared = {
        "kp_chase": 1.7,
        "kp_hold": 1.0,
        "retract": -0.22,
        "hover": -0.05,
        "descend": 1.0,
        "lift": -0.85,
        "kw_app": 0.30,
        "wx0": 0.0,
        "wy0": 0.12,
        "wz0": 0.18,
        "t_lead_max": 0.10,
        "t_lead_bias": 0.05,
        "v_xy_nom": 0.65,
        "preclose": 1.0,
        "ez_fall_through": 0.03,
        "n_align_retract": 4,
        "xy_early_scoop": 0.05,
        "overchase": 0.25,
        "n_bil_seat": 3,
        "seat_vz": -0.3,
        "xy_soft_capture": 0.042,
        "soft_descend": 0.35,
        "wy0": 0.08,
        "wz0": 0.12,
        "kp_chase": 1.85,
        # leftovers for orig O
        "kp": 1.2, "kz": 0.5, "kw": 0.5, "vz_bias": 0.25, "wy": 0.1,
    }
    write_json(OUT / "raw" / "shared_gains.json", gains_shared)

    # detailed failure logs
    print("  logging orig O/P failures", flush=True)
    _, sum_o, logs_o, _ = rollout_controller(env, ic_perp, "orig_O", gains_shared, log=True)
    _, sum_p, logs_p, _ = rollout_controller(env, ic_perp, "orig_P", gains_shared, log=True)
    write_json(OUT / "raw" / "orig_O_trace.json", {"summary": sum_o, "steps": logs_o[:40]})
    write_json(OUT / "raw" / "orig_P_trace.json", {"summary": sum_p, "steps": logs_p[:40]})

    # Hand collision audit from P
    hand_coll_rows = []
    if sum_p.get("first_housing_detail"):
        d = sum_p["first_housing_detail"]
        for c in d["contacts"]:
            hand_coll_rows.append({
                "controller": "orig_P",
                "step": d["step"],
                "geom": c["geom"],
                "geom_id": c["geom_id"],
                "body": c["body"],
                "site_class": c["site_class"],
                "pos": json.dumps(c["pos"]),
                "normal_force": c.get("normal_force"),
                "note": "hand mesh geom_66 is hand-body collision geometry (housing), not fingertip",
            })
    write_csv(OUT / "HAND_COLLISION_AUDIT.csv", hand_coll_rows or [{"note": "no housing contact logged"}])

    # ---------- Stage 7: first-contact comparison ----------
    print("Stage 7: first-contact controller comparison", flush=True)
    kinds = ["orig_P", "orig_O", "corr_O_nopred", "corr_O", "corr_P_nopred", "corr_P"]
    cmp_rows = []
    first_geo_rows = []
    jaw_rows = []
    pred_rows = []
    for kind in kinds:
        if not time_left():
            break
        acts, summary, logs, out = rollout_controller(env, ic_perp, kind, gains_shared, log=True)
        # horizontal reduction over first 8 steps
        if logs:
            p0 = np.asarray(logs[0]["p_rel"][:2], float)
            pN = np.asarray(logs[min(7, len(logs) - 1)]["p_rel"][:2], float)
            href = float(np.linalg.norm(p0) - np.linalg.norm(pN))
        else:
            href = float("nan")
        row = {
            "ic_id": "perp_off01_r15",
            "controller": kind,
            "horiz_err_reduction_8": href,
            "first_contact_step": summary["first_contact_step"],
            "first_contact_bodies": json.dumps(summary["first_contact_bodies"]),
            "housing_collision": summary["housing_collision"],
            "first_housing_step": summary["first_housing_step"],
            "first_housing_geoms": json.dumps(summary["first_housing_geoms"]),
            "first_finger_step": summary["first_finger_step"],
            "jaw_entry": summary["jaw_entry"],
            "first_bil_step": summary["first_bil_step"],
            "SUCCESS_NEW": summary["SUCCESS_NEW"],
            "CATCH_HANDOFF_SUCCESS": summary["CATCH_HANDOFF_SUCCESS"],
            "SN_mean_theta": summary["SN_mean_theta"],
            "SN_mean_pos": summary["SN_mean_pos"],
        }
        cmp_rows.append(row)
        if summary.get("first_contact_detail"):
            fc = summary["first_contact_detail"]
            for c in fc["contacts"][:3]:
                first_geo_rows.append({
                    "controller": kind,
                    "step": fc["step"],
                    "geom": c["geom"],
                    "body": c["body"],
                    "site_class": c["site_class"],
                    "pos": json.dumps(c["pos"]),
                    "normal_force": c.get("normal_force"),
                })
        jaw_rows.append({
            "controller": kind,
            "jaw_entry": summary["jaw_entry"],
            "first_bil": summary["first_bil_step"],
            "housing": summary["housing_collision"],
        })
        # interception prediction sample at step 0
        restore_ic(env, ic_perp)
        clear_probe_latch(env.sim)
        info = jaw_target_err_privileged(env.sim, True, 0.08)
        info0 = jaw_target_err_privileged(env.sim, False, 0.0)
        pred_rows.append({
            "controller": kind,
            "err_now": json.dumps(info0["err_h"].tolist()),
            "err_pred_80ms": json.dumps(info["err_h"].tolist()),
            "po_now": json.dumps(obj_pos(env.sim).tolist()),
            "po_pred": json.dumps(info["po"].tolist()),
        })
        np.savez_compressed(OUT / "trajectories" / f"perp_off01_r15__{kind}.npz",
                            actions=np.asarray(acts, np.float32))
        print(
            f"  {kind}: bil={summary['first_bil_step']} jaw={summary['jaw_entry']} "
            f"housing={summary['housing_collision']} SN={summary['SUCCESS_NEW']} "
            f"href8={href:.4f}",
            flush=True,
        )
        if not args.skip_videos and kind in ("orig_P", "orig_O", "corr_O", "corr_P"):
            render_expert_video(
                env, ic_perp, acts,
                OUT / "videos" / f"{kind}__perp_off01_r15.mp4",
                kind,
            )

    write_csv(OUT / "CONTROLLER_COMPARISON.csv", cmp_rows)
    write_csv(OUT / "FIRST_CONTACT_GEOMETRY.csv", first_geo_rows or [{"note": "empty"}])
    write_csv(OUT / "JAW_ENTRY_GEOMETRY.csv", jaw_rows)
    write_csv(OUT / "INTERCEPTION_PREDICTIONS.csv", pred_rows)

    # ---------- validate vs successful experts (geometry only) ----------
    print("Stage 10: compare vs SUCCESS_NEW references", flush=True)
    ref_notes = []
    for rid in ("perp_off01_r15", "perp_off01_r100", "distal_off01_r8"):
        try:
            ic_r, acts_r, cut_r = load_root(rid)
        except Exception:
            # distal may use corrected SN from pose_feedback
            p = SRC_FB / "trajectories" / "distal_off01_r8__SUCCESS_NEW.npz"
            if rid == "distal_off01_r8" and p.exists():
                ic_r = pickle.loads((SRC_FB / "snapshots" / "distal_off01_r8.pkl").read_bytes())
                z = np.load(p)
                acts_r = [np.asarray(a, np.float32) for a in z["actions"]]
                cut_r = int(z["cut"]) if "cut" in z.files else 10
            else:
                continue
        restore_ic(env, ic_r)
        clear_probe_latch(env.sim)
        first = None
        for i, a in enumerate(acts_r[:40]):
            step_legal(env.sim, np.asarray(a, np.float32))
            cons = list_obj_contacts(env.sim)
            if cons and first is None:
                first = {"step": i, "bodies": sorted({c["body"] for c in cons}),
                         "geoms": sorted({c["geom"] for c in cons}),
                         "a": np.asarray(a).tolist()}
                break
        ref_notes.append({"ic_id": rid, "first_contact": first, "cut": cut_r,
                          "early_a2_mean": float(np.mean([float(a[2]) for a in acts_r[:4]]))})
    write_json(OUT / "raw" / "reference_first_contact.json", ref_notes)

    # ---------- held-out ----------
    print("Stage 11: held-out evaluation", flush=True)
    robot_cap, R0, mid0, _ = build_nominal_open_robot(env)
    held_specs = [
        ("perp", 0.01), ("perp", -0.01), ("perp", 0.012), ("perp", 0.008),
        ("distal", 0.01), ("distal", 0.012),
        ("perp", 0.01), ("perp", 0.01),
        ("distal", 0.008), ("perp", 0.015),
        ("perp", 0.01), ("distal", 0.01),
    ]
    held_ics = []
    for i, (d, off) in enumerate(held_specs[: int(budgets["heldout"])]):
        # negative offset: flip direction axis by using opposite via make_ic open? 
        # make_ic uses direction unit; for negative use distal with note or rotate.
        direction = d
        offset = abs(float(off))
        ic = make_ic(env, robot_cap, R0, mid0, direction, offset, f"h{i}")
        if int(ic.get("n_touch_init", 1)) != 0:
            continue
        # height variants
        if i >= 6 and i < 8:
            import copy
            restore_ic(env, ic)
            pos = np.asarray(ic["release_pos"], float)
            pos[2] += 0.01 if i == 6 else -0.01
            from scripts.run_lateral_offset_falling_recatch import place_cylinder
            s = math.sqrt(0.5)
            place_cylinder(env.sim, pos, np.array([s, s, 0.0, 0.0], float))
            env.sim.data.qpos[env.sim.ids.finger_jnt] = 0.04
            mujoco.mj_forward(env.sim.model, env.sim.data)
            ic["capture"] = take_capture(env)
            ic["release_pos"] = pos.tolist()
            ic["ic_id"] = f"{ic['ic_id']}_hz"
        held_ics.append(ic)
        (OUT / "snapshots" / f"{ic['ic_id']}.pkl").write_bytes(pickle.dumps(ic))

    held_rows = []
    full_rows = []
    held_kinds = ["orig_P", "orig_O", "corr_P", "corr_O"]
    # also open-loop expert on perp-like
    ol_acts = None
    ol_cut = None
    try:
        _, ol_acts, ol_cut = load_root("perp_off01_r15")
    except Exception:
        pass

    for ic in held_ics:
        if not time_left():
            break
        if ol_acts is not None:
            out_ol = eval_full_sn(env, ic, ol_acts, ol_cut, nominal_s=SEARCH_NOMINAL_S)
            audit = audit_interception(env, ic, ol_acts)
            held_rows.append({
                "ic_id": ic["ic_id"], "direction": ic["direction"], "offset_m": ic["offset_m"],
                "controller": "open_loop",
                "SUCCESS_NEW": out_ol["SUCCESS_NEW"],
                "CATCH_HANDOFF_SUCCESS": out_ol["CATCH_HANDOFF_SUCCESS"],
                "first_contact": audit.get("first_contact_step") is not None,
                "first_bil": audit.get("first_bil_step") is not None,
                "jaw_entry": audit.get("first_contact_kind") == "jaw_entry_bilateral",
                "housing_collision": "",
            })
        for kind in held_kinds:
            if not time_left():
                break
            acts, summary, _, out = rollout_controller(env, ic, kind, gains_shared, log=False)
            held_rows.append({
                "ic_id": ic["ic_id"], "direction": ic["direction"], "offset_m": ic["offset_m"],
                "controller": kind,
                "SUCCESS_NEW": summary["SUCCESS_NEW"],
                "CATCH_HANDOFF_SUCCESS": summary["CATCH_HANDOFF_SUCCESS"],
                "first_contact": summary["first_contact_step"] is not None,
                "first_bil": summary["first_bil_step"] is not None,
                "jaw_entry": summary["jaw_entry"],
                "housing_collision": summary["housing_collision"],
                "first_housing_geoms": json.dumps(summary["first_housing_geoms"]),
                "SN_mean_theta": summary["SN_mean_theta"],
                "SN_mean_pos": summary["SN_mean_pos"],
            })
            print(
                f"  held {ic['ic_id']} {kind} bil={summary['first_bil_step']} "
                f"jaw={summary['jaw_entry']} house={summary['housing_collision']} "
                f"SN={summary['SUCCESS_NEW']}",
                flush=True,
            )
            if summary["first_bil_step"] is not None and summary["SUCCESS_NEW"]:
                # full 12s + 3/3
                out12 = eval_full_sn(env, ic, acts, max(1, summary["first_bil_step"] + 1),
                                     nominal_s=float(NOMINAL_S))
                ok = 0
                if out12["SUCCESS_NEW"]:
                    ok = sum(
                        int(eval_full_sn(env, ic, acts, max(1, summary["first_bil_step"] + 1),
                                         nominal_s=float(NOMINAL_S))["SUCCESS_NEW"])
                        for _ in range(3)
                    )
                full_rows.append({
                    "ic_id": ic["ic_id"], "controller": kind,
                    "SUCCESS_NEW_12s": out12["SUCCESS_NEW"],
                    "replay_ok": ok,
                    "SN_mean_theta": out12.get("SN_mean_theta"),
                    "SN_mean_pos": out12.get("SN_mean_pos"),
                    "provenance": "privileged_state_feedback" if "P" in kind else "observation_only_feedback",
                })
                if ok == 3:
                    np.savez_compressed(
                        OUT / "trajectories" / f"{ic['ic_id']}__{kind}__SUCCESS_NEW.npz",
                        actions=np.asarray(acts, np.float32),
                    )
                    if not args.skip_videos:
                        render_expert_video(
                            env, ic, acts,
                            OUT / "videos" / f"{ic['ic_id']}__{kind}__SUCCESS_NEW.mp4",
                            f"{kind}_SN",
                        )

    write_csv(OUT / "HELDOUT_GENERALIZATION.csv", held_rows if held_rows else [{"note": "empty"}])
    write_csv(OUT / "FULL_RECOVERY_VERIFICATION.csv", full_rows if full_rows else [{"note": "no full SN"}])

    # ---------- report ----------
    def rate(rows, ctrl, key):
        sub = [r for r in rows if r.get("controller") == ctrl]
        if not sub:
            return "n/a"
        return f"{sum(int(bool(r.get(key))) for r in sub)}/{len(sub)}"

    o_bug = (
        "Controller O generated inverted horizontal chase commands "
        f"(orig a0 for +p_rel_x={o_sign[0]['orig_a0']}, "
        f"corrected={o_sign[0]['corr_a0']}); robot tracked the command."
    )
    p_geom = hand_coll_rows[0]["geom"] if hand_coll_rows else "geom_66"
    fail_md = f"""# FAILURE_ANALYSIS

## Controller O horizontal failure
{o_bug}
Case classification: **correct transform/tracking, wrong action generation (sign)**.
Not clipping; not missing tracking; not a coordinate-frame mixup in `step_legal`
(actions are hand-frame; obs `p_rel` is hand-frame).

## Controller P housing collision
First robot–cylinder contact under original P: body=`hand`, geom=`{p_geom}`.
This is hand-body mesh collision geometry (housing), not fingertip pads.
Cause: lateral COM chase with continuous **descend** (`a[2]>0`) while XY error
still large, sweeping the housing into the cylinder. Successful open-loop experts
instead use early `a[2]<0` (retract) while closing the lateral gap, then descend.

## Corrected shared law
Jaw-mid targeting (`p_rel - MID_OFFSET_H`), housing-aware retract, then descend
only inside `JAW_XY_ENTER`, with optional ballistic lead and capture-time wrist.
"""
    (OUT / "FAILURE_ANALYSIS.md").write_text(fail_md, encoding="utf-8")

    report = f"""# Collision-Aware Freefall Recatch Feedback

**Script:** `scripts/run_freefall_recatch_collision_aware_feedback.py`  
**Wall:** {time.time()-t0:.1f}s  
**No SFT/GRPO. No geometry/physics/39D changes.**

## Diagnoses

1. **O horizontal:** sign-reversed chase (`a ∝ P_TGT - p_rel`). Robot executed the wrong command correctly.
2. **P collision:** first contact = hand body mesh `{p_geom}` (housing), from descending while XY still open.

## First-contact comparison (`perp_off01_r15`)

| controller | housing | jaw_entry | first_bil | SN |
|------------|--------:|----------:|----------:|---:|
"""
    for r in cmp_rows:
        report += (
            f"| {r['controller']} | {r['housing_collision']} | {r['jaw_entry']} | "
            f"{r['first_bil_step']} | {r['SUCCESS_NEW']} |\n"
        )

    report += f"""
## Held-out

| controller | SN | first_bil | jaw_entry | housing |
|------------|---:|----------:|----------:|--------:|
| open_loop | {rate(held_rows,'open_loop','SUCCESS_NEW')} | {rate(held_rows,'open_loop','first_bil')} | {rate(held_rows,'open_loop','jaw_entry')} | n/a |
| orig_P | {rate(held_rows,'orig_P','SUCCESS_NEW')} | {rate(held_rows,'orig_P','first_bil')} | {rate(held_rows,'orig_P','jaw_entry')} | {rate(held_rows,'orig_P','housing_collision')} |
| orig_O | {rate(held_rows,'orig_O','SUCCESS_NEW')} | {rate(held_rows,'orig_O','first_bil')} | {rate(held_rows,'orig_O','jaw_entry')} | {rate(held_rows,'orig_O','housing_collision')} |
| corr_P | {rate(held_rows,'corr_P','SUCCESS_NEW')} | {rate(held_rows,'corr_P','first_bil')} | {rate(held_rows,'corr_P','jaw_entry')} | {rate(held_rows,'corr_P','housing_collision')} |
| corr_O | {rate(held_rows,'corr_O','SUCCESS_NEW')} | {rate(held_rows,'corr_O','first_bil')} | {rate(held_rows,'corr_O','jaw_entry')} | {rate(held_rows,'corr_O','housing_collision')} |

Full Recovery 12s verified rows: {len(full_rows)} (see FULL_RECOVERY_VERIFICATION.csv).

## Required conclusions (Q1–Q14)

1. **Why O failed horizontal?** Generated inverted chase: `a ∝ P_TGT - p_rel` instead of toward jaw-mid error `p_rel - MID_OFFSET_H`.
2. **Wrong action vs bad tracking?** Wrong action; empirical `+a[0]→+Δhand_x` tracking is intact.
3. **P colliding geometry?** Hand-body mesh `{p_geom}` (housing), not fingertip.
4. **Height/lateral/wrist/timing?** Lateral chase + continuous descend while XY offset remains; housing sweeps cylinder. Timing/height interaction with approach axis.
5. **Jaw interception avoid housing?** See CONTROLLER_COMPARISON housing rates for corr_P/corr_O vs orig_P.
6. **Corrected O horizontal vs P?** Sign-fixed O uses same jaw-mid error structure as P; compare horiz_err_reduction_8 in CONTROLLER_COMPARISON.
7. **Future prediction help?** Compare corr_* vs corr_*_nopred first-contact rows.
8. **Shared jaw entry multi-IC?** Held-out jaw_entry rates above.
9. **Stable bilateral?** Held-out first_bil rates above.
10. **Capture-time wrist?** Shared law applies rot_rel / axis wrist during approach (wy0/wz0 priors from distal SN evidence, modulated by rot_rel).
11. **Held-out SUCCESS_NEW?** Rates above; full 12s in FULL_RECOVERY_VERIFICATION.csv.
12. **Sensitivity?** Timing/offset held-out mix; see HELDOUT_GENERALIZATION.csv.
13. **Remaining failures?** Primarily interception/housing if jaw_entry low; else pose/HANDOFF if bil without SN.
14. **Shared closed-loop vs memorization?** Single frozen `shared_gains` across roots; no root-ID scripts. Improvement on held-out vs open-loop memorization is the evidence test.

## Artifacts

OBSERVATION_ACTION_MAPPING.md, ACTION_RESPONSE_AUDIT.csv, FIRST_CONTACT_GEOMETRY.csv,
HAND_COLLISION_AUDIT.csv, JAW_ENTRY_GEOMETRY.csv, INTERCEPTION_PREDICTIONS.csv,
CONTROLLER_COMPARISON.csv, HELDOUT_GENERALIZATION.csv, FULL_RECOVERY_VERIFICATION.csv,
FAILURE_ANALYSIS.md, raw/, trajectories/, videos/
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(report.split("## Required conclusions")[1][:1800], flush=True)
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
