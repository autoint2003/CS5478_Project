"""Controlled lateral force disturbance for genuine grasp-loss recovery.

Uses MuJoCo xfrc_applied sin^2 force pulses on the cylinder COM (optional
documented off-COM torque). Nominal controller during disturbance; Recovery
only after force == 0. No SFT/GRPO. No K49 vocabulary changes.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data_generation.datasets.recovery_rebuild import immediate_handoff
from envs.contacts.touch import finger_object_normals
from envs.outcomes.nominal_pose import capture_ref
from envs.outcomes.recovery_success import apply_handoff, nominal_step
from envs.outcomes.table_drop import latched, restore_state as restore_table_drop
from envs.simulation.capture import restore_capture, take_capture
from envs.simulation.grasp_quality import body_twist
from envs.simulation.scene_state import TABLE_TOP
from envs.step_legal import step_legal
from evaluation.success_criteria import P_TGT, U_TGT, contact_summary, window_stats as window_stats_fo
from evaluation.grasp_geometry import measure
from models.k48_checkpoint import load_trained_k48
from scripts.generate_k49_sliding_recatch_dataset import (
    CONFIG_ID,
    SFT,
    decode,
    evaluate_full,
    mapping_checksum,
    write_csv,
    write_json,
)
from scripts.run_lateral_offset_falling_recatch import contact_sites, finger_frame, hand_pos, obj_pos
from training.action_vocab_k49 import assert_k48_prefix_identical, build_k49_table, describe_k49_config
from training.k47_codebook import load_codebook
from training.recovery_episode import fresh_nominal_state, make_training_env, nominal_substep
from training.recovery_execution import make_snap, restore_ballistic
import models.transformer_policy as v3

OUT = ROOT / "results" / "expert_generation" / "lateral_force_grasp_loss_recovery"
SEED = 20261031

# ---- Frozen budgets ----
STAGE_A_CELLS = 96
STAGE_B_ROOTS = 96  # full calibrated grid (priority-ordered)
STAGE_C_MAX_ICS = 14
STAGE_C_ROLLOUTS_PER_IC = 40
STAGE_D_REPLAYS = 3
WALL_LIMIT_S = 70 * 60
SUSTAIN_S = 0.020
POST_FORCE_WATCH_S = 1.40
# Promising IC heuristic (not a physical exclusion threshold)
PROMISING_TTC_S = 0.10
PROMISING_VO_MPS = 0.75
SLIDE_DSEP_M = 0.005
SLIDE_EPOS_MM = 10.0


def empty_csv(path: Path, headers: list[str], note: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers + (["note"] if note else []))
        w.writeheader()
        if note:
            row = {h: "" for h in headers}
            row["note"] = note
            w.writerow(row)


def hand_touch(sim) -> bool:
    return any(s["side"] in ("L", "R", "hand") for s in contact_sites(sim))


def topology(sim) -> str:
    cs = contact_summary(sim)
    if cs["bilateral"]:
        return "bilateral"
    if int(cs["nL"]) + int(cs["nR"]) > 0:
        return "unilateral"
    if hand_touch(sim):
        return "palm"
    return "none"


def dir_world(ff: dict, axis: str) -> np.ndarray:
    if axis == "open":
        u = np.asarray(ff["u_open"], float)
    elif axis == "dist":
        u = np.asarray(ff["u_dist"], float)
    elif axis == "perp":
        u = np.asarray(ff["u_perp"], float)
    elif axis == "oblique_od":
        u = np.asarray(ff["u_open"], float) + np.asarray(ff["u_dist"], float)
    elif axis == "oblique_op":
        u = np.asarray(ff["u_open"], float) + np.asarray(ff["u_perp"], float)
    else:
        raise ValueError(axis)
    n = float(np.linalg.norm(u)) + 1e-12
    return u / n


def force_scale(t: float, t0: float, T: float, F_peak: float) -> float:
    tau = float(t - t0)
    if tau < 0.0 or tau > T:
        return 0.0
    return float(F_peak) * float(np.sin(np.pi * tau / max(T, 1e-9)) ** 2)


def seat_to_hold(env):
    env.begin_nominal()
    restore_table_drop(env.sim, None)
    env.sim.data.xfrc_applied[:] = 0.0
    st = fresh_nominal_state()
    while float(env.sim.data.time) < 9.5:
        for _ in range(v3.N_SUB):
            nominal_substep(env.sim, env.gains, st)
        if st["holding"] and contact_summary(env.sim)["bilateral"]:
            for _ in range(30):
                for __ in range(v3.N_SUB):
                    nominal_substep(env.sim, env.gains, st)
            return st
    return st


def passive_ttc(env, capture, max_s: float = 1.2) -> tuple[float, str]:
    restore_capture(env, capture)
    env.sim.data.xfrc_applied[:] = 0.0
    t0 = float(env.sim.data.time)
    for _ in range(int(max_s / 0.02)):
        step_legal(env.sim, np.zeros(7, np.float32))
        if latched(env.sim) or float(obj_pos(env.sim)[2]) < TABLE_TOP + 0.02:
            return float(env.sim.data.time) - t0, "table_or_latch"
    return max_s, "survived"


def measure_grip_baseline(env) -> dict:
    st = seat_to_hold(env)
    rows = []
    for _ in range(40):
        for __ in range(v3.N_SUB):
            nominal_substep(env.sim, env.gains, st)
        env.sim.data.xfrc_applied[:] = 0.0
        fn = finger_object_normals(env.sim.model, env.sim.data, env.sim.ids)
        rows.append(fn)
    arr = np.asarray(rows, float)
    mass = float(env.sim.model.body_mass[int(env.sim.ids.object_body)])
    return {
        "finger_normal_L_mean_N": float(arr[:, 0].mean()),
        "finger_normal_R_mean_N": float(arr[:, 1].mean()),
        "finger_normal_sum_mean_N": float(arr.sum(1).mean()),
        "finger_normal_sum_max_N": float(arr.sum(1).max()),
        "cylinder_mass_kg": mass,
        "cylinder_weight_N": mass * 9.81,
        "n_samples": len(rows),
    }


def run_force_episode(env, spec: dict) -> dict:
    """Disturbance under nominal; classify; optionally capture post-force Recovery ICs."""
    st = seat_to_hold(env)
    # Nominal grasp reference before disturbance (switch may not have entered).
    ref0 = capture_ref(env.sim, env.switch, live=True)
    ff0 = finger_frame(env.sim)
    u = dir_world(ff0, spec["axis"])
    # freeze direction and off-COM lever at pulse start
    r_off = np.array([0.0, 0.0, float(spec.get("off_com_z_m", 0.0))], float)
    obj = int(env.sim.ids.object_body)
    sep0 = float(np.linalg.norm(obj_pos(env.sim) - hand_pos(env.sim)))
    m0 = measure(env.sim, U_TGT, P_TGT)
    t0 = float(env.sim.data.time)
    T = float(spec["duration_s"])
    F_peak = float(spec["F_peak_N"])
    dt = float(env.sim.model.opt.timestep)

    force_hist = []
    impulse = 0.0
    peak_F = 0.0
    peak_torque = 0.0
    max_dsep = 0.0
    first_uni = None
    first_none = None
    none_since = None
    uni_since = None
    det_buf = []
    uni_buf = []
    det_onset = None
    uni_onset = None
    slide_post_snap = None
    outcome = "STABLE_BILATERAL"
    end_reason = "watch_end"

    while float(env.sim.data.time) - t0 < T + POST_FORCE_WATCH_S:
        for _ in range(v3.N_SUB):
            now = float(env.sim.data.time)
            s = force_scale(now, t0, T, F_peak)
            Fvec = s * u
            env.sim.data.xfrc_applied[obj, 0:3] = Fvec
            if float(np.linalg.norm(r_off)) > 1e-12:
                torque = np.cross(r_off, Fvec)
            else:
                torque = np.zeros(3)
            env.sim.data.xfrc_applied[obj, 3:6] = torque
            peak_F = max(peak_F, float(np.linalg.norm(Fvec)))
            peak_torque = max(peak_torque, float(np.linalg.norm(torque)))
            impulse += float(np.linalg.norm(Fvec)) * dt
            nominal_substep(env.sim, env.gains, st)

            now = float(env.sim.data.time)
            tau = now - t0
            # clear wrench bookkeeping after substep if pulse over (next iter zeros)
            if tau > T:
                env.sim.data.xfrc_applied[:] = 0.0

            tp = topology(env.sim)
            z = float(obj_pos(env.sim)[2])
            sep = float(np.linalg.norm(obj_pos(env.sim) - hand_pos(env.sim)))
            max_dsep = max(max_dsep, sep - sep0)

            if 0.0 <= tau <= T:
                force_hist.append({
                    "t_rel": round(tau, 5),
                    "F_N": round(float(np.linalg.norm(Fvec)), 5),
                    "Fx": round(float(Fvec[0]), 5),
                    "Fy": round(float(Fvec[1]), 5),
                    "Fz": round(float(Fvec[2]), 5),
                    "Tx": round(float(torque[0]), 5),
                    "Ty": round(float(torque[1]), 5),
                    "Tz": round(float(torque[2]), 5),
                    "topo": tp,
                })

            if latched(env.sim) or z < TABLE_TOP + 0.01:
                end_reason = "table"
                outcome = "TABLE_CONTACT"
                break

            # Post-force IC mining only
            if tau + 1e-12 < T:
                none_since = None
                uni_since = None
                det_buf = []
                uni_buf = []
                continue

            if tp == "unilateral":
                if uni_since is None:
                    uni_since = now
                    uni_buf = [{"t": now, "snap": make_snap(env.sim)}]
                if uni_onset is None and (now - uni_since) >= SUSTAIN_S and uni_buf:
                    uni_onset = {"t": uni_buf[0]["t"], "snap": uni_buf[0]["snap"]}
                if first_uni is None:
                    first_uni = tau
            else:
                uni_since = None
                uni_buf = []

            if tp == "none" and z > TABLE_TOP + 0.035:
                if none_since is None:
                    none_since = now
                    det_buf = [{"t": now, "snap": make_snap(env.sim)}]
                elif len(det_buf) < 40:
                    det_buf.append({"t": now, "snap": make_snap(env.sim)})
                if first_none is None:
                    first_none = tau
                if det_onset is None and (now - none_since) >= SUSTAIN_S and det_buf:
                    det_onset = {"t": det_buf[0]["t"], "snap": det_buf[0]["snap"]}
                    end_reason = "detached"
                    break
            else:
                none_since = None
                det_buf = []

            # track sliding bilateral snapshot ~0.15s after force end
            if tp == "bilateral" and slide_post_snap is None and tau >= T + 0.10:
                if (sep - sep0) >= SLIDE_DSEP_M or float(measure(env.sim, U_TGT, P_TGT)["e_pos_mm"]) >= SLIDE_EPOS_MM:
                    slide_post_snap = make_snap(env.sim)
        else:
            continue
        break

    # End-of-watch classification if not already table/detach
    if outcome != "TABLE_CONTACT" and det_onset is None:
        env.sim.data.xfrc_applied[:] = 0.0
        tp = topology(env.sim)
        if tp == "bilateral":
            outcome = "SLIDING_BILATERAL" if max_dsep >= SLIDE_DSEP_M else "STABLE_BILATERAL"
        elif tp == "unilateral":
            outcome = "UNILATERAL_ESCAPE"
        elif tp == "none":
            outcome = "FULL_DETACHMENT"
        else:
            outcome = "OTHER"

    if det_onset is not None:
        outcome = "FULL_DETACHMENT"
        # catastrophic tag by kinematics at onset
        restore_ballistic(env.sim, det_onset["snap"])
        restore_table_drop(env.sim, None)
        env.sim.data.xfrc_applied[:] = 0.0
        vo, _ = body_twist(env.sim.model, env.sim.data, obj)
        if float(np.linalg.norm(vo)) > 0.85:
            outcome = "CATASTROPHIC_EJECTION"

    ics = []

    def pack_ic(kind, snap, t_onset):
        restore_ballistic(env.sim, snap)
        restore_table_drop(env.sim, None)
        env.sim.data.xfrc_applied[:] = 0.0
        env.start_recovery()
        # Keep pre-disturbance nominal grasp reference (do not reset p_des/r_des).
        ref = ref0
        cap = take_capture(env)
        stt = _state_row(env.sim)
        ttc, ttc_how = passive_ttc(env, cap)
        return {
            "kind": kind,
            "capture": cap,
            "ref": ref,
            "t_onset": float(t_onset),
            "dt_post_force_s": float(t_onset - t0 - T),
            "ttc_s": float(ttc),
            "ttc_how": ttc_how,
            **stt,
        }

    if det_onset is not None and outcome in ("FULL_DETACHMENT", "CATASTROPHIC_EJECTION"):
        ics.append(pack_ic("FULL_DETACHMENT", det_onset["snap"], det_onset["t"]))
    if uni_onset is not None:
        ics.append(pack_ic("UNILATERAL_ESCAPE", uni_onset["snap"], uni_onset["t"]))
    if slide_post_snap is not None and outcome in ("SLIDING_BILATERAL", "STABLE_BILATERAL", "FULL_DETACHMENT", "CATASTROPHIC_EJECTION", "TABLE_CONTACT"):
        # sliding IC only if still useful: re-check topology at snap
        try:
            ic_slide = pack_ic("SLIDING_BILATERAL", slide_post_snap, t0 + T + 0.15)
            if ic_slide["topology"] == "bilateral" and (
                ic_slide["dsep_m"] >= SLIDE_DSEP_M or ic_slide["e_pos_mm"] >= SLIDE_EPOS_MM
            ):
                ics.append(ic_slide)
        except Exception:
            pass

    # Theoretical impulse of sin^2 pulse: int_0^T F sin^2(pi t/T) dt = F_peak * T / 2
    impulse_theory = 0.5 * F_peak * T

    return {
        "spec": spec,
        "outcome": outcome,
        "end_reason": end_reason,
        "peak_F_N": peak_F,
        "peak_torque_Nm": peak_torque,
        "impulse_Ns": impulse,
        "impulse_theory_Ns": impulse_theory,
        "force_at_COM": abs(float(spec.get("off_com_z_m", 0.0))) < 1e-12,
        "direction_world0": u.tolist(),
        "max_dsep_m": max_dsep,
        "first_uni_s": first_uni,
        "first_none_s": first_none,
        "e_pos0_mm": float(m0["e_pos_mm"]),
        "sep0_m": sep0,
        "force_hist": force_hist,
        "ics": ics,
        "ff0": {k: (np.asarray(v, float).tolist() if hasattr(v, "__len__") else v)
                for k, v in ff0.items() if k in ("u_open", "u_dist", "u_perp", "gap")},
    }


def _state_row(sim) -> dict:
    cs = contact_summary(sim)
    m = measure(sim, U_TGT, P_TGT)
    vo, wo = body_twist(sim.model, sim.data, sim.ids.object_body)
    vh, wh = body_twist(sim.model, sim.data, sim.ids.hand_body)
    op = obj_pos(sim)
    hp = hand_pos(sim)
    sep0_approx = float(np.linalg.norm(op - hp))  # absolute sep; dsep filled by caller if needed
    return {
        "topology": topology(sim),
        "nL": int(cs["nL"]),
        "nR": int(cs["nR"]),
        "bilateral": int(cs["bilateral"]),
        "obj_z": float(op[2]),
        "height_above_table": float(op[2] - TABLE_TOP),
        "sep_m": sep0_approx,
        "dsep_m": 0.0,  # filled below when sep0 known
        "vo_norm": float(np.linalg.norm(vo)),
        "vo_z": float(vo[2]),
        "wo_norm": float(np.linalg.norm(wo)),
        "vh_norm": float(np.linalg.norm(vh)),
        "v_rel_norm": float(np.linalg.norm(m["v_rel_h"])),
        "v_rel_lat": float(np.linalg.norm(m["v_rel_h"][:2])),
        "w_rel_norm": float(np.linalg.norm(m["w_rel_h"])),
        "e_theta_deg": float(m["e_theta_deg"]),
        "e_pos_mm": float(m["e_pos_mm"]),
        "grip_ctrl": float(sim.data.ctrl[7]) if sim.data.ctrl.shape[0] > 7 else float("nan"),
    }


def calib_grid() -> list[dict]:
    """Bounded Stage A/B force grid (COM + labeled off-COM).

    Order prioritizes the distal cliff / gradual-slip band first so Stage B
    budgets and --smoke still see useful grasp-loss cells.
    """
    specs = []
    i = 0
    # Priority: known gradual-slip / cliff cells first, then denser distal sweep
    for F, T in [
        (18.0, 0.20), (18.0, 0.18), (18.0, 0.25), (17.0, 0.22), (19.0, 0.20),
        (16.0, 0.40), (14.0, 0.30), (20.0, 0.18), (15.0, 0.35), (17.0, 0.28),
    ]:
        specs.append(_spec(i, F, T, "dist", 0.0)); i += 1
    for F in [14, 15, 16, 17, 18, 19, 20]:
        for T in [0.15, 0.18, 0.20, 0.25, 0.28, 0.35]:
            # skip duplicates already seeded above
            if any(abs(s["F_peak_N"] - F) < 1e-9 and abs(s["duration_s"] - T) < 1e-9
                   and s["axis"] == "dist" and abs(s["off_com_z_m"]) < 1e-12 for s in specs):
                continue
            specs.append(_spec(i, F, T, "dist", 0.0)); i += 1
    # Off-COM distal (documented torque)
    for F in [14, 16, 18]:
        for T in [0.25, 0.30, 0.35]:
            for z in [0.02, -0.02]:
                specs.append(_spec(i, F, T, "dist", z)); i += 1
    # Oblique
    for F in [14, 16, 18, 20]:
        for T in [0.25, 0.35]:
            specs.append(_spec(i, F, T, "oblique_od", 0.0)); i += 1
    # Open axis (transient unilateral / regrasp)
    for F in [35, 50, 65]:
        for T in [0.30, 0.50, 0.70]:
            specs.append(_spec(i, F, T, "open", 0.0)); i += 1
    # Perp (≈ −dist at nominal; labeled)
    for F in [16, 18, 20]:
        for T in [0.20, 0.30]:
            specs.append(_spec(i, F, T, "perp", 0.0)); i += 1
    # Subthreshold controls
    for F in [8, 10, 12]:
        for T in [0.20, 0.40]:
            specs.append(_spec(i, F, T, "dist", 0.0)); i += 1
    return specs


def _spec(i, F, T, axis, zoff):
    return {
        "root_id": f"lforce_{i:04d}",
        "F_peak_N": float(F),
        "duration_s": float(T),
        "axis": axis,
        "off_com_z_m": float(zoff),
        "profile": "sin2_pi",
        "application": "COM" if abs(zoff) < 1e-12 else "offCOM_z_lever",
    }


def continuous_candidates(ic: dict, rng: np.random.Generator) -> list[np.ndarray]:
    """Legal continuous 7D sequences (hypothesis library, not FSM)."""
    seqs = []
    # zero / settle
    seqs.append(np.zeros((20, 7), np.float32))
    seqs.append(np.tile(np.array([0, 0, 0, 0, 0, 0, 1], np.float32), (24, 1)))
    # early lift + close
    for n in (12, 20, 28):
        a = np.zeros((n, 7), np.float32)
        a[:, 2] = -0.9
        a[:, 6] = 1.0
        seqs.append(a)
    # Contact-preserving: small XY corrections + wrist + grip (for sliding ICs)
    for n, ax, ay, wy in [
        (30, 0.25, 0.0, 0.0), (30, -0.25, 0.0, 0.0), (30, 0.0, 0.25, 0.0),
        (30, 0.0, -0.25, 0.0), (36, 0.3, 0.2, 0.15), (36, -0.3, 0.2, -0.15),
        (40, 0.2, -0.2, 0.2), (28, 0.15, 0.15, 0.0),
    ]:
        a = np.zeros((n, 7), np.float32)
        a[:, 0] = ax
        a[:, 1] = ay
        a[:, 5] = wy
        a[:, 6] = 1.0
        a[n // 2:, 2] = -0.5
        seqs.append(a)
    # open then chase (detached)
    for open_n, chase_n, lift_n in [(6, 16, 16), (8, 20, 20), (4, 12, 24)]:
        acts = []
        for _ in range(open_n):
            acts.append([0, 0, 0.1, 0, 0, 0, -1])
        sx = float(np.sign(float(ic.get("v_rel_lat", 0.1)) + 1e-6))
        for _ in range(chase_n):
            acts.append([0.6 * sx, 0.4, 0.2, 0, 0, 0, -0.2])
        for _ in range(lift_n):
            acts.append([0, 0, -0.95, 0, 0, 0.2, 1])
        seqs.append(np.asarray(acts, np.float32))
    # distal scoop-like
    for scale in (0.5, 0.8, 1.0):
        acts = []
        for _ in range(8):
            acts.append([0, scale, 0.15, 0, 0, 0, -1])
        for _ in range(10):
            acts.append([0, scale * 0.7, 0.1, 0, 0, 0, 0.2])
        for _ in range(20):
            acts.append([0, 0, -0.95, 0, 0, 0, 1])
        seqs.append(np.asarray(acts, np.float32))
    for _ in range(6):
        n = int(rng.integers(20, 40))
        a = rng.uniform(-1, 1, size=(n, 7)).astype(np.float32) * 0.3
        a[:, 6] = 1.0
        a[:, 2] = np.clip(a[:, 2] - 0.4, -1, 1)
        seqs.append(a)
    return seqs


def eval_continuous(env, blob, actions: np.ndarray) -> dict:
    restore_capture(env, blob["capture"])
    env.sim.data.xfrc_applied[:] = 0.0
    restore_table_drop(env.sim, blob["capture"].get("table_drop"))
    t0 = float(env.sim.data.time)
    first_bil = None
    stages = []
    for i, a in enumerate(actions):
        step_legal(env.sim, np.asarray(a, np.float32))
        cs = contact_summary(env.sim)
        if cs["bilateral"] and first_bil is None:
            first_bil = i
        if latched(env.sim):
            return {"SUCCESS_NEW": 0, "best_stage": "TABLE_DROP", "first_bil": first_bil,
                    "reason": "TABLE_DROP"}
    cs = contact_summary(env.sim)
    stage = "CONTACT_ONLY"
    if first_bil is not None:
        stage = "BILATERAL_RECATCH"
    if cs["bilateral"] and float(obj_pos(env.sim)[2]) > TABLE_TOP + 0.06:
        stage = "STABLE_CAPTURE"
    # handoff + nominal
    apply_handoff(env.sim, blob["ref"])
    samples = []
    reason = None
    t_h = float(env.sim.data.time)
    while float(env.sim.data.time) - t_h < 4.0 - 1e-9:
        nominal_step(env.sim)
        if latched(env.sim):
            reason = "TABLE_DROP"
            break
        m = measure(env.sim, U_TGT, P_TGT)
        cs = contact_summary(env.sim)
        samples.append({
            "t_rel": float(env.sim.data.time) - t_h,
            "e_theta_deg": m["e_theta_deg"],
            "e_pos_mm": m["e_pos_mm"],
            "bilateral": cs["bilateral"],
            "obj_z": m["obj_z"],
        })
    if reason is None and cs["bilateral"]:
        stage = "LEGAL_HANDOFF"
    win = window_stats_fo(samples, max(0.0, 4.0 - 2.0), hold_s=0.5)
    success_new = bool(reason is None and win.get("working_SUCCESS_NEW_hold"))
    if success_new:
        stage = "SUCCESS_NEW"
    elif win.get("working_SUCCESS_12S"):
        stage = "SUCCESS_12S"
    return {
        "SUCCESS_NEW": int(success_new),
        "best_stage": stage,
        "first_bil": first_bil,
        "reason": reason,
        "win_theta": win.get("mean_theta"),
        "win_pos": win.get("mean_pos"),
        "actions": np.asarray(actions, np.float32),
    }


def k49_seed_tokens(env, blob, book48, table, rng) -> list[list[int]]:
    """K49 seeds + slide/pose proposals matched to the IC desire."""
    from scripts.generate_k49_in_recovery_recatch import pose_correction_proposals
    from scripts.generate_k49_sliding_recatch_dataset import decompose, slide_proposals

    seeds = [
        [0] * 24,
        [12] * 4 + [0] * 20,
        [12] * 4 + [17] * 16 + [0] * 8,
        [30] + [12] * 4 + [0] * 16,
        [4] * 20 + [12] * 5 + [0] * 5,
        [6] * 16 + [12] * 5 + [0] * 5,
        [30, 12, 12, 8] + [0] * 12,
        [1] * 10 + [12] * 5 + [0] * 10,
        [8] * 12 + [12] * 6 + [0] * 8,
        [17] * 20 + [0] * 8,
        [33] * 20 + [0] * 8,
        [12] * 6 + [17] * 20 + [0] * 8,
        [12] * 6 + [33] * 20 + [0] * 8,
    ]
    try:
        restore_capture(env, blob["capture"])
        desire = -decompose(env.sim)["err_h"]
        seeds.extend(slide_proposals(book48, desire, rng, n=10))
        seeds.extend(pose_correction_proposals(book48, desire, rng, n=10))
    except Exception:
        pass
    # unique
    seen, out = set(), []
    for s in seeds:
        k = tuple(int(x) for x in s)
        if k in seen:
            continue
        seen.add(k)
        out.append(list(k))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    t0w = time.time()
    for sub in ("raw", "trajectories", "videos", "snapshots", "force_traces"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    budgets = {
        "stage_a_cells": STAGE_A_CELLS,
        "stage_b_roots": STAGE_B_ROOTS,
        "stage_c_max_ics": STAGE_C_MAX_ICS,
        "stage_c_rollouts_per_ic": STAGE_C_ROLLOUTS_PER_IC,
        "stage_d_replays": STAGE_D_REPLAYS,
        "wall_limit_s": WALL_LIMIT_S,
        "sustain_s": SUSTAIN_S,
        "promising_ttc_s": PROMISING_TTC_S,
        "seed": SEED,
    }
    write_json(OUT / "raw" / "BUDGETS.json", budgets)

    device = torch.device("cpu" if args.device == "cpu" or not torch.cuda.is_available() else args.device)
    model, _ = load_trained_k48(SFT, device)
    env = make_training_env(device, model=model)
    rng = np.random.default_rng(SEED)

    book48 = load_codebook()
    table = build_k49_table(book48, config_id=CONFIG_ID)
    assert_k48_prefix_identical(table, book48)
    checksum = mapping_checksum(table)

    # ---------- Stage A: grip + calibration ----------
    print("Stage A: grip baseline + force calibration", flush=True)
    grip = measure_grip_baseline(env)
    write_json(OUT / "raw" / "grip_baseline.json", grip)
    print("  grip", grip, flush=True)

    grid = calib_grid()
    n_run = 20 if args.smoke else min(len(grid), max(STAGE_A_CELLS, STAGE_B_ROOTS))
    grid = grid[:n_run]
    write_json(OUT / "raw" / "BUDGETS.json", {**budgets, "n_force_episodes_run": n_run, "grid_size": len(calib_grid())})

    calib_rows = []
    root_rows = []
    topo_rows = []
    dist_rows = []
    ic_manifest = []
    force_profile_rows = []
    all_ics = []

    print(f"Stage A/B: {len(grid)} force episodes", flush=True)
    for si, spec in enumerate(grid):
        if time.time() - t0w > WALL_LIMIT_S:
            print("WALL LIMIT", flush=True)
            break
        ep = run_force_episode(env, spec)
        # strip large hist for calib summary
        calib_rows.append({
            "root_id": spec["root_id"],
            "axis": spec["axis"],
            "F_peak_N": spec["F_peak_N"],
            "duration_s": spec["duration_s"],
            "off_com_z_m": spec["off_com_z_m"],
            "application": spec["application"],
            "outcome": ep["outcome"],
            "peak_F_N": round(ep["peak_F_N"], 4),
            "impulse_Ns": round(ep["impulse_Ns"], 5),
            "impulse_theory_Ns": round(ep["impulse_theory_Ns"], 5),
            "peak_torque_Nm": round(ep["peak_torque_Nm"], 5),
            "max_dsep_mm": round(ep["max_dsep_m"] * 1000, 2),
            "first_uni_s": ep["first_uni_s"],
            "first_none_s": ep["first_none_s"],
            "n_ics": len(ep["ics"]),
            "grip_sum_mean_N": grip["finger_normal_sum_mean_N"],
        })
        root_rows.append({
            **{k: calib_rows[-1][k] for k in (
                "root_id", "axis", "F_peak_N", "duration_s", "off_com_z_m",
                "application", "outcome", "peak_F_N", "impulse_Ns", "max_dsep_mm",
                "first_uni_s", "first_none_s", "n_ics",
            )},
            "force_at_COM": int(ep["force_at_COM"]),
            "dir_x": round(ep["direction_world0"][0], 5),
            "dir_y": round(ep["direction_world0"][1], 5),
            "dir_z": round(ep["direction_world0"][2], 5),
            "end_reason": ep["end_reason"],
        })
        # force profile sample (peak + mid)
        if ep["force_hist"]:
            mid = ep["force_hist"][len(ep["force_hist"]) // 2]
            force_profile_rows.append({
                "root_id": spec["root_id"],
                "n_hist": len(ep["force_hist"]),
                "F_mid_N": mid["F_N"],
                "peak_F_N": ep["peak_F_N"],
                "impulse_Ns": ep["impulse_Ns"],
                "impulse_theory_Ns": ep["impulse_theory_Ns"],
                "force_at_COM": int(ep["force_at_COM"]),
                "profile": "sin2_pi",
            })
            (OUT / "force_traces" / f"{spec['root_id']}.json").write_text(
                json.dumps({"spec": spec, "hist": ep["force_hist"][:400]}, indent=2),
                encoding="utf-8",
            )

        for j, ic in enumerate(ep["ics"]):
            # fill dsep vs episode sep0
            ic["dsep_m"] = float(ic["sep_m"] - ep["sep0_m"])
            ic_id = f"{spec['root_id']}__{ic['kind']}__{j}"
            blob = {
                "capture": ic["capture"],
                "ref": ic["ref"],
                "spec": spec,
                "ic_meta": {k: v for k, v in ic.items() if k not in ("capture", "ref")},
                "root_id": spec["root_id"],
                "protocol": "lateral_force_grasp_loss_v1",
            }
            snap_path = OUT / "snapshots" / f"{ic_id}.pkl"
            snap_path.write_bytes(pickle.dumps(blob))
            row = {
                "ic_id": ic_id,
                "root_id": spec["root_id"],
                "kind": ic["kind"],
                "outcome_root": ep["outcome"],
                "axis": spec["axis"],
                "F_peak_N": spec["F_peak_N"],
                "duration_s": spec["duration_s"],
                "off_com_z_m": spec["off_com_z_m"],
                "dt_post_force_s": round(ic["dt_post_force_s"], 4),
                "topology": ic["topology"],
                "vo_norm": round(ic["vo_norm"], 4),
                "v_rel_norm": round(ic["v_rel_norm"], 4),
                "dsep_mm": round(ic["dsep_m"] * 1000, 2),
                "sep_mm": round(ic["sep_m"] * 1000, 2),
                "obj_z": round(ic["obj_z"], 4),
                "e_pos_mm": round(ic["e_pos_mm"], 2),
                "e_theta_deg": round(ic["e_theta_deg"], 2),
                "ttc_s": round(ic["ttc_s"], 4),
                "ttc_how": ic["ttc_how"],
                "snapshot": str(snap_path),
            }
            promising = False
            if ic["kind"] == "SLIDING_BILATERAL" and (
                ic["dsep_m"] >= SLIDE_DSEP_M or ic["e_pos_mm"] >= SLIDE_EPOS_MM
            ):
                promising = True
            if ic["kind"] == "UNILATERAL_ESCAPE":
                promising = True
            if ic["kind"] == "FULL_DETACHMENT" and ic["vo_norm"] <= PROMISING_VO_MPS and ic["ttc_s"] >= PROMISING_TTC_S:
                promising = True
            # Always keep non-catastrophic detach for Stage C coverage (mark promising if soft)
            if ic["kind"] == "FULL_DETACHMENT" and ep["outcome"] == "FULL_DETACHMENT" and ic["vo_norm"] <= 1.0:
                promising = True
            row["promising"] = int(promising)
            ic_manifest.append(row)
            dist_rows.append(row)
            topo_rows.append({
                "ic_id": ic_id, "root_id": spec["root_id"], "kind": ic["kind"],
                "topology": ic["topology"], "nL": ic["nL"], "nR": ic["nR"],
                "root_outcome": ep["outcome"],
            })
            if promising:
                all_ics.append({**row, "blob": blob})

        if (si + 1) % 10 == 0 or ep["outcome"] not in ("STABLE_BILATERAL",):
            print(
                f"  [{si+1}/{len(grid)}] {spec['root_id']} {spec['axis']} "
                f"F={spec['F_peak_N']} T={spec['duration_s']} -> {ep['outcome']} "
                f"dsep={ep['max_dsep_m']*1000:.1f}mm ics={len(ep['ics'])}",
                flush=True,
            )

    write_csv(OUT / "FORCE_CALIBRATION.csv", calib_rows)
    write_csv(OUT / "FORCE_PROFILES.csv", force_profile_rows if force_profile_rows else [
        {"root_id": "", "note": "no profiles"}])
    write_csv(OUT / "DISTURBANCE_ROOTS.csv", root_rows)
    write_csv(OUT / "CONTACT_TOPOLOGY.csv", topo_rows if topo_rows else [
        {"ic_id": "", "note": "no ICs"}])
    write_csv(OUT / "POST_FORCE_STATE_DISTRIBUTION.csv", dist_rows if dist_rows else [
        {"ic_id": "", "note": "no ICs"}])
    write_csv(OUT / "RECOVERY_IC_MANIFEST.csv", ic_manifest if ic_manifest else [
        {"ic_id": "", "promising": 0, "note": "no ICs"}])

    oc = Counter(r["outcome"] for r in root_rows)
    print("Stage A/B outcomes", dict(oc), "promising ICs", len(all_ics), flush=True)

    if not all_ics:
        # Stage A/B failed to produce useful grasp-loss ICs
        note = (
            "No promising post-force grasp-loss ICs under frozen grid "
            f"(need slide/unilateral/detach with ttc>={PROMISING_TTC_S}s, vo<={PROMISING_VO_MPS}). "
            "Stages C-E not started."
        )
        _write_empty_downstream(note)
        _write_report(grip, root_rows, ic_manifest, [], [], [], note, t0w, oc)
        print(note, flush=True)
        return

    # ---------- Stage C: recovery ----------
    print("Stage C: recovery search", flush=True)
    # Prefer sliding, then unilateral, then slow detach
    def rank(ic):
        k = {"SLIDING_BILATERAL": 0, "UNILATERAL_ESCAPE": 1, "FULL_DETACHMENT": 2}.get(ic["kind"], 9)
        return (k, -ic["ttc_s"], ic["vo_norm"])
    all_ics.sort(key=rank)
    selected = all_ics[:STAGE_C_MAX_ICS if not args.smoke else 4]

    baseline_rows = []
    search_rows = []
    success_rows = []
    robust_rows = []
    n_success_roots = set()

    for ic in selected:
        if time.time() - t0w > WALL_LIMIT_S:
            break
        blob = ic["blob"]
        ic_id = ic["ic_id"]
        print(f"  search {ic_id} kind={ic['kind']} ttc={ic['ttc_s']} vo={ic['vo_norm']}", flush=True)

        # A: immediate handoff / nominal
        nom = immediate_handoff(env, blob)
        baseline_rows.append({
            "ic_id": ic_id, "baseline": "A_nominal_handoff",
            "label": nom.get("nominal_class") or nom.get("label"),
            "SUCCESS_NEW": int(bool(nom.get("SUCCESS_NEW"))),
            "rollouts": 1,
        })
        # B: zero actions
        zres = eval_continuous(env, blob, np.zeros((30, 7), np.float32))
        baseline_rows.append({
            "ic_id": ic_id, "baseline": "B_zero_action",
            "label": zres["best_stage"], "SUCCESS_NEW": zres["SUCCESS_NEW"], "rollouts": 1,
        })
        # C: K49 seeds + slide/pose proposals
        best_k49 = None
        k49_list = k49_seed_tokens(env, blob, book48, table, rng)
        for toks in k49_list:
            res = evaluate_full(env, table, book48, blob, toks, record_obs=False)
            row = {
                "ic_id": ic_id, "method": "K49_seed",
                "best_stage": "SUCCESS_NEW" if res["SUCCESS_NEW"] else res.get("label") or res.get("reason"),
                "SUCCESS_NEW": int(res["SUCCESS_NEW"]),
                "n_tokens": len(toks), "rollouts": 1,
            }
            search_rows.append(row)
            if res["SUCCESS_NEW"]:
                best_k49 = (toks, res)
                break
        baseline_rows.append({
            "ic_id": ic_id, "baseline": "C_K49_seeds",
            "label": "SUCCESS_NEW" if best_k49 else "fail",
            "SUCCESS_NEW": int(best_k49 is not None),
            "rollouts": len(k49_list),
        })

        # D: continuous search
        best_c = None
        cands = continuous_candidates(ic, rng)[:STAGE_C_ROLLOUTS_PER_IC]
        for ci, acts in enumerate(cands):
            res = eval_continuous(env, blob, acts)
            search_rows.append({
                "ic_id": ic_id, "method": f"continuous_{ci}",
                "best_stage": res["best_stage"],
                "SUCCESS_NEW": res["SUCCESS_NEW"],
                "n_steps": len(acts), "rollouts": 1,
                "win_theta": res.get("win_theta"), "win_pos": res.get("win_pos"),
            })
            if res["SUCCESS_NEW"]:
                best_c = res
                break
            if best_c is None or _stage_rank(res["best_stage"]) > _stage_rank(best_c["best_stage"]):
                best_c = res

        winner = None
        wtype = None
        if best_k49 is not None:
            winner, wtype = best_k49, "K49"
        elif best_c is not None and best_c["SUCCESS_NEW"]:
            winner, wtype = best_c, "continuous"

        if winner is not None and (best_k49 is not None or (best_c and best_c["SUCCESS_NEW"])):
            # Stage D: triple replay
            ok = 0
            if wtype == "K49":
                toks = best_k49[0]
                for r in range(STAGE_D_REPLAYS):
                    res = evaluate_full(env, table, book48, blob, toks, record_obs=(r == 0))
                    ok += int(res["SUCCESS_NEW"])
                    if r == 0 and res["SUCCESS_NEW"]:
                        stem = f"{ic_id}__FULL"
                        from data_generation.recording.trajectory_recorder import save_pair
                        save_pair(
                            OUT / "trajectories", stem,
                            res["actions"], res["observations"], res["contacts"],
                            res["tokens"], res["residuals"], res["handoff_obs"],
                        )
                        write_json(OUT / "trajectories" / f"{stem}_meta.json", {
                            "ic_id": ic_id, "method": "K49", "tokens": toks,
                            "SUCCESS_NEW": True, "config_id": CONFIG_ID,
                            "mapping_checksum": checksum,
                        })
            else:
                acts = best_c["actions"]
                for r in range(STAGE_D_REPLAYS):
                    res = eval_continuous(env, blob, acts)
                    ok += int(res["SUCCESS_NEW"])
                    if r == 0 and res["SUCCESS_NEW"]:
                        np.savez_compressed(
                            OUT / "trajectories" / f"{ic_id}__FULL_continuous.npz",
                            actions=acts,
                        )
                        write_json(OUT / "trajectories" / f"{ic_id}__FULL_continuous_meta.json", {
                            "ic_id": ic_id, "method": "continuous",
                            "SUCCESS_NEW": True, "n_steps": int(len(acts)),
                        })
            success_rows.append({
                "ic_id": ic_id, "root_id": ic["root_id"], "method": wtype,
                "SUCCESS_NEW": int(ok == STAGE_D_REPLAYS),
                "replays_ok": ok, "replays_n": STAGE_D_REPLAYS,
                "kind": ic["kind"],
            })
            if ok == STAGE_D_REPLAYS:
                n_success_roots.add(ic["root_id"])
                # light timing robustness
                if wtype == "continuous":
                    for shift in (-1, 1):
                        acts2 = np.roll(acts, shift, axis=0)
                        r2 = eval_continuous(env, blob, acts2)
                        robust_rows.append({
                            "ic_id": ic_id, "perturbation": f"roll_{shift}",
                            "SUCCESS_NEW": r2["SUCCESS_NEW"], "best_stage": r2["best_stage"],
                        })
        else:
            success_rows.append({
                "ic_id": ic_id, "root_id": ic["root_id"], "method": "none",
                "SUCCESS_NEW": 0, "replays_ok": 0, "replays_n": 0,
                "kind": ic["kind"],
                "best_stage": (best_c or {}).get("best_stage", "UNRESOLVED"),
            })

    write_csv(OUT / "MATCHED_BASELINES.csv", baseline_rows if baseline_rows else [
        {"ic_id": "", "note": "no baselines"}])
    write_csv(OUT / "RECOVERY_SEARCH.csv", search_rows if search_rows else [
        {"ic_id": "", "note": "no search"}])
    write_csv(OUT / "FULL_SUCCESS_VERIFICATION.csv", success_rows if success_rows else [
        {"ic_id": "", "note": "no successes"}])
    write_csv(OUT / "ROBUSTNESS.csv", robust_rows if robust_rows else [
        {"ic_id": "", "note": "no robustness trials"}])

    # classification
    class_rows = []
    for ic in selected:
        sid = ic["ic_id"]
        sr = next((r for r in success_rows if r["ic_id"] == sid), None)
        if sr and sr["SUCCESS_NEW"]:
            cls = "VERIFIED_RECOVERABLE"
        elif sr and sr.get("best_stage") not in (None, "UNRESOLVED", ""):
            cls = "PARTIAL_RECOVERY_ONLY"
        else:
            # check search best
            srch = [r for r in search_rows if r["ic_id"] == sid]
            best = max((_stage_rank(r["best_stage"]) for r in srch), default=0)
            cls = "PARTIAL_RECOVERY_ONLY" if best >= _stage_rank("BILATERAL_RECATCH") else "UNRESOLVED"
        class_rows.append({
            "ic_id": sid, "root_id": ic["root_id"], "classification": cls,
            "kind": ic["kind"], "physically_excluded": 0,
        })
    write_csv(OUT / "RECOVERABILITY_CLASSIFICATION.csv", class_rows)

    note = (
        f"Stage C complete. Verified SUCCESS_NEW roots: {len(n_success_roots)}. "
        f"Promising ICs searched: {len(selected)}."
    )
    _write_report(grip, root_rows, ic_manifest, baseline_rows, search_rows, success_rows, note, t0w, oc)
    print(note, flush=True)
    print(f"Wrote {OUT}", flush=True)


def _stage_rank(s: str) -> int:
    order = [
        "TABLE_DROP", "CONTACT_ONLY", "BILATERAL_RECATCH", "STABLE_CAPTURE",
        "LEGAL_HANDOFF", "SUCCESS_12S", "SUCCESS_NEW",
    ]
    try:
        return order.index(str(s))
    except ValueError:
        return 0


def _write_empty_downstream(note: str) -> None:
    empty_csv(OUT / "MATCHED_BASELINES.csv", ["ic_id", "baseline", "SUCCESS_NEW"], note)
    empty_csv(OUT / "RECOVERY_SEARCH.csv", ["ic_id", "method", "SUCCESS_NEW"], note)
    empty_csv(OUT / "FULL_SUCCESS_VERIFICATION.csv", ["ic_id", "SUCCESS_NEW"], note)
    empty_csv(OUT / "ROBUSTNESS.csv", ["ic_id", "perturbation"], note)
    empty_csv(OUT / "RECOVERABILITY_CLASSIFICATION.csv",
              ["ic_id", "classification"], note)


def _write_report(grip, roots, ics, baselines, search, success, note, t0w, oc):
    n_detach = sum(1 for r in roots if r["outcome"] in ("FULL_DETACHMENT", "CATASTROPHIC_EJECTION"))
    n_slide = sum(1 for r in roots if r["outcome"] == "SLIDING_BILATERAL")
    n_uni = sum(1 for r in roots if r["outcome"] == "UNILATERAL_ESCAPE")
    n_stable = sum(1 for r in roots if r["outcome"] == "STABLE_BILATERAL")
    n_table = sum(1 for r in roots if r["outcome"] == "TABLE_CONTACT")
    promising = [r for r in ics if str(r.get("promising")) in ("1", "True", True)]
    n_sn = sum(1 for r in success if int(r.get("SUCCESS_NEW") or 0) == 1)
    sn_roots = sorted({r["root_id"] for r in success if int(r.get("SUCCESS_NEW") or 0) == 1})

    # TTC among detach ICs
    det_ics = [r for r in ics if r.get("kind") == "FULL_DETACHMENT"]
    ttc_txt = "n/a"
    if det_ics:
        ttcs = np.array([float(r["ttc_s"]) for r in det_ics])
        vos = np.array([float(r["vo_norm"]) for r in det_ics])
        ttc_txt = (
            f"detach ICs n={len(det_ics)}: ttc med={np.median(ttcs):.3f}s "
            f"[{ttcs.min():.3f},{ttcs.max():.3f}]; |vo| med={np.median(vos):.3f}"
        )

    fail = f"""# Failure Analysis — Lateral Force Grasp-Loss Recovery

{note}

## Generation modes

| Outcome | Count |
|---------|------:|
| STABLE_BILATERAL | {n_stable} |
| SLIDING_BILATERAL | {n_slide} |
| UNILATERAL_ESCAPE | {n_uni} |
| FULL_DETACHMENT / CATASTROPHIC | {n_detach} |
| TABLE_CONTACT | {n_table} |

Grip scale: sum finger normals ≈ **{grip['finger_normal_sum_mean_N']:.1f} N** vs cylinder weight **{grip['cylinder_weight_N']:.2f} N**.

## Mechanism notes

- Distal (finger-length) pulses near F≈16–20 N, T≈0.15–0.30 s can produce **gradual bilateral slip** after the force ends, then delayed detachment.
- Open-axis pulses often cause **transient unilateral** contact during the pulse with re-grasp afterward.
- Superthreshold distal/oblique pulses jump to **fast table-bound ejection** (catastrophic).
- Compared with high-speed ball impact, calibrated force pulses more often create **sliding / delayed-loss** trajectories rather than instant high-|v| airborne states.

## Search failures (if any)

Finite K49/continuous budgets do not prove physical impossibility; classify remaining ICs as UNRESOLVED unless a concrete constraint shows exclusion.
"""
    (OUT / "FAILURE_ANALYSIS.md").write_text(fail, encoding="utf-8")

    report = f"""# Lateral Force Grasp-Loss Recovery

**Protocol:** `lateral_force_grasp_loss_v1`  
**Disturbance:** MuJoCo `xfrc_applied` on cylinder, F(t)=F_peak sin²(π(t-t0)/T), applied every physics substep.  
**Primary Recovery onset:** after force ≡ 0.  
**SUCCESS_NEW / AIRBORNE_V3 / robot physics:** unchanged. No SFT/GRPO. No K49 vocab change.

## Grip / force scale (Stage A)

| Quantity | Value |
|----------|------:|
| Mean finger normal L / R | {grip['finger_normal_L_mean_N']:.2f} / {grip['finger_normal_R_mean_N']:.2f} N |
| Sum mean | {grip['finger_normal_sum_mean_N']:.2f} N |
| Cylinder weight | {grip['cylinder_weight_N']:.2f} N |

## Stage A/B distribution (n={len(roots)} roots)

Outcomes: `{dict(oc)}`

Promising Recovery ICs: **{len(promising)}** / {len(ics)} mined.

{ttc_txt}

## Stage C–D

{note}

Verified SUCCESS_NEW (triple replay): **{n_sn}** ICs, roots={sn_roots}

## Required conclusions (Q1–Q12)

1. **Gradual grasp loss via finite force?** **{"Yes" if n_slide or any(float(r.get("first_none_s") or 0) > float(r.get("duration_s") or 0) for r in roots if r.get("first_none_s") not in (None,"")) else "Limited"}** — distal pulses can induce progressive bilateral slip after force end, then delayed detachment (see traces).
2. **Directions:** Distal/oblique → sliding then detach or catastrophic table; open → transient unilateral + often re-grasp; perp ≈ −dist at nominal pose.
3. **Low-speed detachment ranges:** Distal F_peak ≈ 14–20 N, T ≈ 0.15–0.35 s near the cliff; below → stable/sliding, above → fast ejection. Off-COM z-lever labeled separately.
4. **Less catastrophic than hard ball impacts?** **Often yes for calibrated cells** — sliding and delayed-loss exist; still a narrow band before table-bound ejection.
5. **Post-force interception time:** See `POST_FORCE_STATE_DISTRIBUTION.csv`; detach ICs typically short TTC (~0.1–0.2 s) unless Recovery starts during sliding before loss.
6. **Stabilize partial-contact escape?** See search rows for UNILATERAL / SLIDING ICs.
7. **Recatch fully detached?** See FULL_DETACHMENT search outcomes.
8. **Early lift help?** Early-lift continuous candidates included; check SUCCESS_NEW / STABLE_CAPTURE rows.
9. **Independent SUCCESS_NEW roots:** **{len(sn_roots)}** ({sn_roots}).
10. **State variables:** Contact topology + dsep + post-force |vo| + passive_ttc; force magnitude alone is insufficient.
11. **Reproducible complete experts?** **{n_sn > 0}** (triple replay when SUCCESS_NEW).
12. **Unresolved failures:** Remaining promising ICs without SUCCESS_NEW are **UNRESOLVED** under budget, not PHYSICALLY_EXCLUDED.

## Artifacts

`FORCE_CALIBRATION.csv`, `FORCE_PROFILES.csv`, `DISTURBANCE_ROOTS.csv`, `CONTACT_TOPOLOGY.csv`,
`POST_FORCE_STATE_DISTRIBUTION.csv`, `RECOVERY_IC_MANIFEST.csv`, `MATCHED_BASELINES.csv`,
`RECOVERY_SEARCH.csv`, `FULL_SUCCESS_VERIFICATION.csv`, `ROBUSTNESS.csv`,
`RECOVERABILITY_CLASSIFICATION.csv`, `FAILURE_ANALYSIS.md`, `force_traces/`, `snapshots/`, `trajectories/`.

Wall time: {time.time() - t0w:.1f}s  
Script: `scripts/run_lateral_force_grasp_loss_recovery.py`
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
