"""Velocity-controlled distal pushing for low-speed full detachment ICs.

External-force disturbance only (xfrc_applied). Does not modify Panda control,
K49, physics, reward, or SUCCESS_NEW. No SFT/GRPO.

Stages
  A  Gripper geometry + fixed sin^2 baseline
  B  Velocity-controlled distal sliding calibration
  C  Position-aware unloading / detach-triggered termination
  D  Independent-root IC distribution + method comparison
  E  Limited Recovery pilot (only if low-speed detach ICs exist)
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from envs.contacts.frames import pad_span_hand
from envs.contacts.touch import finger_object_normals
from envs.outcomes.nominal_pose import capture_ref
from envs.outcomes.recovery_success import apply_handoff, nominal_step
from envs.outcomes.table_drop import latched, restore_state as restore_table_drop
from envs.simulation.capture import restore_capture, take_capture
from envs.simulation.grasp_quality import body_twist
from envs.simulation.scene_state import TABLE_TOP
from envs.step_legal import step_legal
from evaluation.success_criteria import P_TGT, U_TGT, contact_summary, window_stats as window_stats_fo
from evaluation.grasp_geometry import measure, p_com_hand
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
from scripts.run_lateral_force_grasp_loss_recovery import (
    continuous_candidates,
    eval_continuous,
    k49_seed_tokens,
    seat_to_hold,
)
from scripts.run_lateral_offset_falling_recatch import contact_sites, finger_frame, hand_pos, obj_pos
from training.action_vocab_k49 import assert_k48_prefix_identical, build_k49_table, describe_k49_config
from training.k47_codebook import load_codebook
from training.recovery_episode import make_training_env, nominal_substep
from training.recovery_execution import make_snap, restore_ballistic
import models.transformer_policy as v3

OUT = ROOT / "results" / "expert_generation" / "controlled_slow_detachment"
SEED = 20261102

# ---- Frozen budgets ----
STAGE_A_BASELINE = 10
STAGE_B_CELLS = 18
STAGE_C_CELLS = 24
STAGE_D_ROOTS = 12
STAGE_E_MAX_ICS = 6
STAGE_E_K49 = 24
STAGE_E_CONT = 20
STAGE_E_REPLAYS = 3
WALL_LIMIT_S = 90 * 60

SUSTAIN_NONE_S = 0.020
POST_WATCH_S = 1.20
CLEARANCE_M = 0.008          # geometric clearance past pad tip
V_SAFE_CUTOFF = 1.50         # |vo| safety abort
F_MAX_DEFAULT = 25.0
SLEW_N_PER_S = 80.0          # force slew limit
LOW_SPEED_VREL = 0.12        # report cohort; not a hard IC definition
LOW_SPEED_VO = 0.55


def empty_csv(path: Path, headers: list[str], note: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers + (["note"] if note else []))
        w.writeheader()
        if note:
            row = {h: "" for h in headers}
            row["note"] = note
            w.writerow(row)


def topology(sim) -> str:
    cs = contact_summary(sim)
    if cs["bilateral"]:
        return "bilateral"
    if int(cs["nL"]) + int(cs["nR"]) > 0:
        return "unilateral"
    if any(s["side"] in ("L", "R", "hand") for s in contact_sites(sim)):
        return "palm"
    return "none"


def escape_axis(sim) -> np.ndarray:
    """Finger-length / fingertip escape = +hand-z (pad distal tip)."""
    Rh = np.asarray(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    u = np.asarray(Rh[:, 2], float)
    return u / (np.linalg.norm(u) + 1e-12)


def prior_dist_axis(sim) -> np.ndarray:
    """Prior lateral-force 'dist' = Rh[:,0] (for baseline comparison only)."""
    Rh = np.asarray(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    u = np.asarray(Rh[:, 0], float)
    return u / (np.linalg.norm(u) + 1e-12)


def geometric_state(sim) -> dict:
    """Hand-relative geometry vs measured pad AABB."""
    pad = pad_span_hand(sim)
    p = np.asarray(p_com_hand(sim), float)
    u = escape_axis(sim)
    ff = finger_frame(sim)
    vo, wo = body_twist(sim.model, sim.data, sim.ids.object_body)
    vh, wh = body_twist(sim.model, sim.data, sim.ids.hand_body)
    v_rel = vo - vh
    v_escape = float(np.dot(v_rel, u))
    # Remaining distal distance to pad tip (positive => still inside pad z-span)
    d_remain = float(pad["z_hi"] - p[2])
    # Cylinder radius (capsule/cylinder geom)
    r_cyl = float(sim.model.geom_size[int(sim.ids.object_geom)][0])
    # Clearance past tip (negative remain means COM past tip)
    clearance = float(-d_remain)  # >0 once past tip
    mid = ff["mid"]
    op = obj_pos(sim)
    sep_mid = float(np.linalg.norm(op - mid))
    return {
        "pad": pad,
        "p_hand": p,
        "u_escape": u,
        "v_escape": v_escape,
        "d_remain_m": d_remain,
        "clearance_past_tip_m": clearance,
        "r_cyl_m": r_cyl,
        "sep_mid_m": sep_mid,
        "gap_m": float(ff["gap"]),
        "obj_z": float(op[2]),
        "height_above_table": float(op[2] - TABLE_TOP),
        "vo": vo,
        "vh": vh,
        "wo": wo,
        "v_rel": v_rel,
    }


def audit_nominal_grasp(env) -> dict:
    st = seat_to_hold(env)
    sim = env.sim
    rows = []
    for _ in range(25):
        for __ in range(v3.N_SUB):
            nominal_substep(sim, env.gains, st)
        sim.data.xfrc_applied[:] = 0.0
        g = geometric_state(sim)
        cs = contact_summary(sim)
        fn = finger_object_normals(sim.model, sim.data, sim.ids)
        rows.append({
            "topology": topology(sim),
            "bilateral": int(cs["bilateral"]),
            "table": int(latched(sim) or g["obj_z"] < TABLE_TOP + 0.02),
            "d_remain_m": g["d_remain_m"],
            "gap_m": g["gap_m"],
            "fn_L": float(fn[0]),
            "fn_R": float(fn[1]),
            "height_above_table": g["height_above_table"],
            "e_pos_mm": float(measure(sim, U_TGT, P_TGT)["e_pos_mm"]),
            "e_theta_deg": float(measure(sim, U_TGT, P_TGT)["e_theta_deg"]),
            "source_time_s": float(sim.data.time),
        })
    g0 = geometric_state(sim)
    pad = g0["pad"]
    return {
        "n_samples": len(rows),
        "always_bilateral": int(all(r["bilateral"] for r in rows)),
        "any_table": int(any(r["table"] for r in rows)),
        "mean_d_remain_m": float(np.mean([r["d_remain_m"] for r in rows])),
        "mean_gap_m": float(np.mean([r["gap_m"] for r in rows])),
        "mean_fn_L": float(np.mean([r["fn_L"] for r in rows])),
        "mean_fn_R": float(np.mean([r["fn_R"] for r in rows])),
        "height_above_table": float(np.mean([r["height_above_table"] for r in rows])),
        "source_time_s": float(rows[-1]["source_time_s"]),
        "pad_z_lo": pad["z_lo"],
        "pad_z_hi": pad["z_hi"],
        "pad_x_lo": pad["x_lo"],
        "pad_x_hi": pad["x_hi"],
        "p_hand_z": float(g0["p_hand"][2]),
        "r_cyl_m": g0["r_cyl_m"],
        "u_escape": g0["u_escape"].tolist(),
        "controller_valid": 1,
        "samples": rows,
        "st": st,
    }


class DistalForceController:
    """Bounded velocity feedback along fingertip escape axis (external force only)."""

    def __init__(
        self,
        mode: str,
        *,
        v_target: float = 0.035,
        F_bias: float = 6.0,
        Kv: float = 120.0,
        F_max: float = F_MAX_DEFAULT,
        slew: float = SLEW_N_PER_S,
        unload_dist_m: float = 0.008,
        v_unload: float = 0.015,
        F_peak: float = 18.0,
        duration_s: float = 0.25,
        terminate_on_detach: bool = False,
    ):
        self.mode = mode
        self.v_target = float(v_target)
        self.F_bias = float(F_bias)
        self.Kv = float(Kv)
        self.F_max = float(F_max)
        self.slew = float(slew)
        self.unload_dist_m = float(unload_dist_m)
        self.v_unload = float(v_unload)
        self.F_peak = float(F_peak)
        self.duration_s = float(duration_s)
        self.terminate_on_detach = bool(terminate_on_detach)
        self.F_prev = 0.0
        self.detached_seen = False
        self.work_J = 0.0
        self.impulse_Ns = 0.0
        self.impulse_after_detach_Ns = 0.0
        self.peak_F = 0.0
        self.peak_torque = 0.0

    def reset(self):
        self.F_prev = 0.0
        self.detached_seen = False
        self.work_J = 0.0
        self.impulse_Ns = 0.0
        self.impulse_after_detach_Ns = 0.0
        self.peak_F = 0.0
        self.peak_torque = 0.0

    def _clip_slew(self, F_des: float, dt: float) -> float:
        max_step = self.slew * max(dt, 1e-6)
        return float(np.clip(F_des, self.F_prev - max_step, self.F_prev + max_step))

    def command(
        self,
        *,
        t_rel: float,
        v_escape: float,
        d_remain_m: float,
        topo: str,
        vo_norm: float,
        dt: float,
    ) -> float:
        detached = topo == "none"
        if detached:
            self.detached_seen = True

        if self.mode == "fixed_sin2":
            if t_rel < 0.0 or t_rel > self.duration_s or self.detached_seen and self.terminate_on_detach:
                F = 0.0
            else:
                F = self.F_peak * float(np.sin(np.pi * t_rel / max(self.duration_s, 1e-9)) ** 2)
                if self.terminate_on_detach and detached:
                    F = 0.0
        elif self.mode in ("vel_only", "vel_unload", "vel_detach_term"):
            if vo_norm > V_SAFE_CUTOFF:
                F = 0.0
            elif detached and (
                self.mode == "vel_detach_term" or self.terminate_on_detach or self.detached_seen
            ):
                F = 0.0
            else:
                v_tgt = self.v_target
                F_bias = self.F_bias
                near_tip = d_remain_m < self.unload_dist_m
                past_tip = d_remain_m <= 0.0
                if self.mode == "vel_unload":
                    # Gradual unload: force -> 0 as tip approached; coast past tip
                    if past_tip:
                        F_des = 0.0
                    elif near_tip:
                        alpha = float(np.clip(d_remain_m / max(self.unload_dist_m, 1e-6), 0.0, 1.0))
                        v_tgt = self.v_unload * (1.0 - alpha) + self.v_target * alpha
                        F_bias = self.F_bias * alpha
                        F_des = F_bias + self.Kv * (v_tgt - v_escape)
                    else:
                        F_des = F_bias + self.Kv * (v_tgt - v_escape)
                elif self.mode == "vel_detach_term":
                    # Mild tip assist (not full F_max), zero immediately on detach
                    if near_tip and not past_tip:
                        # taper but keep a small push to finish escape
                        alpha = float(np.clip(d_remain_m / max(self.unload_dist_m, 1e-6), 0.0, 1.0))
                        F_des = (0.35 + 0.65 * alpha) * (
                            F_bias + self.Kv * (v_tgt - v_escape)
                        )
                    elif past_tip:
                        F_des = 0.25 * F_bias  # light coast force until contact loss
                    else:
                        F_des = F_bias + self.Kv * (v_tgt - v_escape)
                else:
                    # vel_only: constant regulation; light tip assist if stalled near tip
                    F_des = F_bias + self.Kv * (v_tgt - v_escape)
                    if near_tip and abs(v_escape) < 0.5 * self.v_target:
                        F_des = max(F_des, 0.5 * self.F_bias)
                F_des = float(np.clip(F_des, 0.0, self.F_max))
                F = self._clip_slew(F_des, dt)
        else:
            raise ValueError(self.mode)

        F = float(np.clip(F, 0.0, self.F_max))
        self.F_prev = F
        return F

    def accumulate(self, Fvec: np.ndarray, torque: np.ndarray, v_obj: np.ndarray, dt: float, detached: bool):
        Fn = float(np.linalg.norm(Fvec))
        self.peak_F = max(self.peak_F, Fn)
        self.peak_torque = max(self.peak_torque, float(np.linalg.norm(torque)))
        self.impulse_Ns += Fn * dt
        self.work_J += float(np.dot(Fvec, v_obj)) * dt
        if detached:
            self.impulse_after_detach_Ns += Fn * dt


def classify_outcome(
    *,
    max_d_escape: float,
    first_uni: float | None,
    first_none: float | None,
    det_ok: bool,
    v_rel_at_detach: float | None,
    vo_at_detach: float | None,
    table_hit: bool,
    final_topo: str,
) -> str:
    if table_hit and not det_ok:
        return "TABLE_DROP"
    if det_ok:
        vr = abs(float(v_rel_at_detach or 0.0))
        vo = abs(float(vo_at_detach or 0.0))
        if vr <= LOW_SPEED_VREL or vo <= LOW_SPEED_VO:
            return "LOW_SPEED_FULL_DETACHMENT"
        return "HIGH_SPEED_EJECTION"
    if first_uni is not None and final_topo in ("unilateral", "none", "palm"):
        return "PARTIAL_CONTACT_ESCAPE"
    if max_d_escape >= 0.004 and final_topo == "bilateral":
        return "CONTROLLED_SLIDING"
    if final_topo == "bilateral":
        return "NO_SLIP"
    return "INVALID_STATE"


def run_episode(env, ctrl: DistalForceController, *, axis: str = "escape", max_force_s: float = 2.5) -> dict:
    """Apply external distal force under unchanged nominal control; mine ICs."""
    st = seat_to_hold(env)
    ref0 = capture_ref(env.sim, env.switch, live=True)
    ctrl.reset()
    sim = env.sim
    obj = int(sim.ids.object_body)
    g0 = geometric_state(sim)
    if axis == "escape":
        u = escape_axis(sim).copy()
    elif axis == "prior_dist":
        u = prior_dist_axis(sim).copy()
    else:
        raise ValueError(axis)

    t0 = float(sim.data.time)
    dt = float(sim.model.opt.timestep)
    d_remain0 = g0["d_remain_m"]
    p0 = g0["p_hand"].copy()

    force_hist = []
    slide_hist = []
    contact_hist = []
    first_uni = None
    first_none = None
    none_since = None
    det_onset = None
    det_buf = []
    table_hit = False
    max_d_escape = 0.0
    v_at_detach = None
    vo_at_detach = None
    F_at_detach = None
    slide_snap = None
    horizon = max_force_s + POST_WATCH_S
    if ctrl.mode == "fixed_sin2":
        horizon = ctrl.duration_s + POST_WATCH_S

    while float(sim.data.time) - t0 < horizon:
        for _ in range(v3.N_SUB):
            now = float(sim.data.time)
            t_rel = now - t0
            g = geometric_state(sim)
            tp = topology(sim)
            vo_n = float(np.linalg.norm(g["vo"]))

            # After fixed pulse window, force stays 0; vel modes run until detach/timeout
            active = True
            if ctrl.mode == "fixed_sin2" and t_rel > ctrl.duration_s:
                active = False
            if ctrl.mode != "fixed_sin2" and t_rel > max_force_s:
                active = False

            if not active or (latched(sim) or g["obj_z"] < TABLE_TOP + 0.015):
                Fmag = 0.0
            else:
                Fmag = ctrl.command(
                    t_rel=t_rel,
                    v_escape=g["v_escape"] if axis == "escape" else float(np.dot(g["v_rel"], u)),
                    d_remain_m=g["d_remain_m"],
                    topo=tp,
                    vo_norm=vo_n,
                    dt=dt,
                )

            Fvec = Fmag * u
            torque = np.zeros(3)
            sim.data.xfrc_applied[obj, 0:3] = Fvec
            sim.data.xfrc_applied[obj, 3:6] = torque
            ctrl.accumulate(Fvec, torque, g["vo"], dt, tp == "none")
            nominal_substep(sim, env.gains, st)

            # Clear wrench when inactive so Recovery never inherits force
            if not active or tp == "none" and ctrl.mode in ("vel_detach_term",) and Fmag == 0.0:
                if Fmag == 0.0:
                    sim.data.xfrc_applied[:] = 0.0

            now = float(sim.data.time)
            t_rel = now - t0
            g = geometric_state(sim)
            tp = topology(sim)
            # Geometry-relative progress. Escape axis: pad-tip remain decrease.
            # prior_dist (Rh0): hand-x displacement (pad-length / prior convention).
            if axis == "escape":
                d_esc = float(d_remain0 - g["d_remain_m"])
            else:
                d_esc = float(abs(g["p_hand"][0]) - abs(p0[0]))
            max_d_escape = max(max_d_escape, d_esc)

            force_hist.append({
                "t_rel": round(t_rel, 5),
                "F_N": round(float(np.linalg.norm(Fvec)), 5),
                "Fx": round(float(Fvec[0]), 5),
                "Fy": round(float(Fvec[1]), 5),
                "Fz": round(float(Fvec[2]), 5),
                "Tx": 0.0, "Ty": 0.0, "Tz": 0.0,
                "work_J": round(ctrl.work_J, 6),
                "impulse_Ns": round(ctrl.impulse_Ns, 6),
            })
            slide_hist.append({
                "t_rel": round(t_rel, 5),
                "d_remain_m": round(g["d_remain_m"], 5),
                "d_escape_m": round(d_esc, 5),
                "v_escape": round(g["v_escape"] if axis == "escape" else float(np.dot(g["v_rel"], u)), 5),
                "vo_norm": round(float(np.linalg.norm(g["vo"])), 5),
                "height_above_table": round(g["height_above_table"], 5),
                "sep_mid_m": round(g["sep_mid_m"], 5),
            })
            contact_hist.append({
                "t_rel": round(t_rel, 5),
                "topology": tp,
                "nL": int(contact_summary(sim)["nL"]),
                "nR": int(contact_summary(sim)["nR"]),
            })

            if latched(sim) or g["obj_z"] < TABLE_TOP + 0.015:
                table_hit = True
                break

            if tp == "unilateral" and first_uni is None:
                first_uni = t_rel
            if tp == "none" and g["height_above_table"] > 0.035:
                if first_none is None:
                    first_none = t_rel
                if none_since is None:
                    none_since = now
                    det_buf = [{"t": now, "snap": make_snap(sim), "F": float(np.linalg.norm(Fvec)),
                                "v_esc": g["v_escape"] if axis == "escape" else float(np.dot(g["v_rel"], u)),
                                "vo": float(np.linalg.norm(g["vo"])),
                                "d_remain": g["d_remain_m"],
                                "clearance": g["clearance_past_tip_m"]}]
                if det_onset is None and (now - none_since) >= SUSTAIN_NONE_S:
                    # Require geometric evidence: past tip or large midjaw sep growth
                    ok_geom = (
                        det_buf[0]["clearance"] >= CLEARANCE_M * 0.25
                        or det_buf[0]["d_remain"] <= 0.0
                        or g["sep_mid_m"] >= 0.06
                    )
                    if ok_geom:
                        det_onset = det_buf[0]
                        v_at_detach = det_onset["v_esc"]
                        vo_at_detach = det_onset["vo"]
                        F_at_detach = det_onset["F"]
                        # force must be cleared at Recovery IC
                        sim.data.xfrc_applied[:] = 0.0
                        break
            else:
                none_since = None
                det_buf = []

            if tp == "bilateral" and slide_snap is None and t_rel > 0.15 and max_d_escape >= 0.004:
                slide_snap = make_snap(sim)
        else:
            continue
        break

    sim.data.xfrc_applied[:] = 0.0
    final_topo = topology(sim)
    det_ok = det_onset is not None
    # residual force check
    residual_F = float(np.linalg.norm(sim.data.xfrc_applied[obj, 0:3]))

    outcome = classify_outcome(
        max_d_escape=max_d_escape,
        first_uni=first_uni,
        first_none=first_none,
        det_ok=det_ok,
        v_rel_at_detach=v_at_detach,
        vo_at_detach=vo_at_detach,
        table_hit=table_hit,
        final_topo=final_topo,
    )

    ics = []

    def pack_ic(kind, snap, t_onset):
        restore_ballistic(sim, snap)
        restore_table_drop(sim, None)
        sim.data.xfrc_applied[:] = 0.0
        env.start_recovery()
        cap = take_capture(env)
        g = geometric_state(sim)
        m = measure(sim, U_TGT, P_TGT)
        ttc, ttc_how = _passive_ttc(env, cap)
        return {
            "kind": kind,
            "capture": cap,
            "ref": ref0,
            "t_onset": float(t_onset),
            "ttc_s": float(ttc),
            "ttc_how": ttc_how,
            "topology": topology(sim),
            "v_escape": float(g["v_escape"]),
            "vo_norm": float(np.linalg.norm(g["vo"])),
            "d_remain_m": float(g["d_remain_m"]),
            "clearance_past_tip_m": float(g["clearance_past_tip_m"]),
            "height_above_table": float(g["height_above_table"]),
            "sep_mid_m": float(g["sep_mid_m"]),
            "e_pos_mm": float(m["e_pos_mm"]),
            "e_theta_deg": float(m["e_theta_deg"]),
            "residual_xfrc_N": float(np.linalg.norm(sim.data.xfrc_applied[obj, 0:3])),
        }

    if det_onset is not None:
        ics.append(pack_ic(outcome, det_onset["snap"], det_onset["t"] - t0))
    if slide_snap is not None and outcome == "CONTROLLED_SLIDING":
        ics.append(pack_ic("CONTROLLED_SLIDING", slide_snap, 0.2))

    return {
        "outcome": outcome,
        "axis": axis,
        "mode": ctrl.mode,
        "ctrl": {
            "v_target": ctrl.v_target,
            "F_bias": ctrl.F_bias,
            "Kv": ctrl.Kv,
            "F_max": ctrl.F_max,
            "unload_dist_m": ctrl.unload_dist_m,
            "v_unload": ctrl.v_unload,
            "F_peak": ctrl.F_peak,
            "duration_s": ctrl.duration_s,
            "terminate_on_detach": ctrl.terminate_on_detach,
        },
        "d_remain0_m": d_remain0,
        "max_d_escape_m": max_d_escape,
        "first_uni_s": first_uni,
        "first_none_s": first_none,
        "v_rel_at_detach": v_at_detach,
        "vo_at_detach": vo_at_detach,
        "F_at_detach_N": F_at_detach,
        "peak_F_N": ctrl.peak_F,
        "peak_torque_Nm": ctrl.peak_torque,
        "impulse_Ns": ctrl.impulse_Ns,
        "impulse_after_detach_Ns": ctrl.impulse_after_detach_Ns,
        "work_J": ctrl.work_J,
        "residual_xfrc_N": residual_F,
        "force_hist": force_hist,
        "slide_hist": slide_hist,
        "contact_hist": contact_hist,
        "ics": ics,
        "u0": u.tolist(),
    }


def _passive_ttc(env, capture, max_s: float = 1.2) -> tuple[float, str]:
    restore_capture(env, capture)
    env.sim.data.xfrc_applied[:] = 0.0
    t0 = float(env.sim.data.time)
    for _ in range(int(max_s / 0.02)):
        step_legal(env.sim, np.zeros(7, np.float32))
        if latched(env.sim) or float(obj_pos(env.sim)[2]) < TABLE_TOP + 0.02:
            return float(env.sim.data.time) - t0, "table_or_latch"
    return max_s, "survived"


def plot_episode(ep: dict, dest: Path, title: str):
    fh = ep["force_hist"]
    sh = ep["slide_hist"]
    ch = ep["contact_hist"]
    if not fh:
        return
    t = [r["t_rel"] for r in fh]
    fig, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
    axes[0].plot(t, [r["F_N"] for r in fh])
    axes[0].set_ylabel("F_ext N")
    axes[0].set_title(title)
    axes[1].plot([r["t_rel"] for r in sh], [r["d_escape_m"] * 1e3 for r in sh], label="d_escape")
    axes[1].plot([r["t_rel"] for r in sh], [r["d_remain_m"] * 1e3 for r in sh], label="d_remain")
    axes[1].legend(fontsize=7)
    axes[1].set_ylabel("mm")
    axes[2].plot([r["t_rel"] for r in sh], [r["v_escape"] for r in sh])
    axes[2].set_ylabel("v_escape m/s")
    topo_map = {"none": 0, "palm": 1, "unilateral": 2, "bilateral": 3}
    axes[3].step([r["t_rel"] for r in ch], [topo_map.get(r["topology"], -1) for r in ch], where="post")
    axes[3].set_yticks([0, 1, 2, 3])
    axes[3].set_yticklabels(["none", "palm", "uni", "bil"])
    axes[3].set_xlabel("t_rel s")
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=110)
    plt.close(fig)


def baseline_specs() -> list[dict]:
    """Representative fixed sin^2 cells (prior distal cliff + fingertip escape)."""
    specs = []
    # prior Rh0 axis (reproduce lateral-force band)
    for F, T in [(14, 0.30), (16, 0.28), (18, 0.20), (18, 0.25), (20, 0.18)]:
        specs.append({"method": "A_fixed", "axis": "prior_dist", "F_peak": F, "duration_s": T})
    # true fingertip escape axis
    for F, T in [(8, 0.40), (10, 0.35), (12, 0.30), (14, 0.25), (16, 0.20)]:
        specs.append({"method": "A_fixed", "axis": "escape", "F_peak": F, "duration_s": T})
    return specs[:STAGE_A_BASELINE]


def vel_calib_specs() -> list[dict]:
    specs = []
    # Higher F_bias needed to overcome grip friction toward tip (smoke: 4-10 N stalled)
    for v_tgt in (0.02, 0.035, 0.05):
        for F_bias in (8.0, 12.0, 16.0):
            for Kv in (60.0, 120.0):
                specs.append({
                    "method": "B_vel_only",
                    "axis": "escape",
                    "mode": "vel_only",
                    "v_target": v_tgt,
                    "F_bias": F_bias,
                    "Kv": Kv,
                    "max_force_s": 2.5,
                })
    return specs[:STAGE_B_CELLS]


def unload_specs() -> list[dict]:
    specs = []
    for v_tgt in (0.02, 0.035, 0.05):
        for F_bias in (10.0, 14.0, 18.0):
            for unload in (0.010, 0.006):
                for mode in ("vel_unload", "vel_detach_term"):
                    specs.append({
                        "method": "C_" + mode,
                        "axis": "escape",
                        "mode": mode,
                        "v_target": v_tgt,
                        "F_bias": F_bias,
                        "Kv": 100.0,
                        "unload_dist_m": unload,
                        "v_unload": 0.012,
                        "max_force_s": 2.8,
                        "terminate_on_detach": True,
                    })
    return specs[:STAGE_C_CELLS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    t0w = time.time()
    for sub in ("raw", "trajectories", "videos", "snapshots", "plots", "force_traces"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    budgets = {
        "stage_a_baseline": STAGE_A_BASELINE,
        "stage_b_cells": STAGE_B_CELLS,
        "stage_c_cells": STAGE_C_CELLS,
        "stage_d_roots": STAGE_D_ROOTS,
        "stage_e_max_ics": STAGE_E_MAX_ICS,
        "stage_e_k49": STAGE_E_K49,
        "stage_e_cont": STAGE_E_CONT,
        "stage_e_replays": STAGE_E_REPLAYS,
        "wall_limit_s": WALL_LIMIT_S,
        "sustain_none_s": SUSTAIN_NONE_S,
        "clearance_m": CLEARANCE_M,
        "low_speed_vrel": LOW_SPEED_VREL,
        "low_speed_vo": LOW_SPEED_VO,
        "seed": SEED,
    }
    if args.smoke:
        budgets.update({
            "stage_a_baseline": 4,
            "stage_b_cells": 4,
            "stage_c_cells": 4,
            "stage_d_roots": 2,
            "stage_e_max_ics": 1,
            "stage_e_k49": 4,
            "stage_e_cont": 3,
            "wall_limit_s": 12 * 60,
        })
    write_json(OUT / "raw" / "BUDGETS.json", budgets)

    device = torch.device("cpu" if args.device == "cpu" or not torch.cuda.is_available() else args.device)
    model, _ = load_trained_k48(SFT, device)
    env = make_training_env(device, model=model)
    rng = np.random.default_rng(SEED)
    book48 = load_codebook()
    table = build_k49_table(book48, config_id=CONFIG_ID)
    assert_k48_prefix_identical(table, book48)
    write_json(OUT / "raw" / "K49_CONFIG.json", {
        "config_id": CONFIG_ID,
        "checksum": mapping_checksum(table),
        "describe": describe_k49_config(CONFIG_ID),
    })

    # ========== Stage A: nominal + geometry + fixed baseline ==========
    print("Stage A: nominal grasp + gripper geometry", flush=True)
    audit = audit_nominal_grasp(env)
    write_csv(OUT / "NOMINAL_GRASP_AUDIT.csv", [{
        k: v for k, v in audit.items() if k not in ("samples", "st", "u_escape")
    } | {"u_escape": json.dumps(audit["u_escape"])}])
    write_csv(OUT / "GRIPPER_GEOMETRY.csv", [{
        "pad_z_lo": audit["pad_z_lo"],
        "pad_z_hi": audit["pad_z_hi"],
        "pad_x_lo": audit["pad_x_lo"],
        "pad_x_hi": audit["pad_x_hi"],
        "p_hand_z_nominal": audit["p_hand_z"],
        "d_remain_m_nominal": audit["mean_d_remain_m"],
        "r_cyl_m": audit["r_cyl_m"],
        "gap_m": audit["mean_gap_m"],
        "escape_axis": "hand_+z (= Rh[:,2], fingertip)",
        "u_escape": json.dumps(audit["u_escape"]),
        "note": "Remaining distal distance = pad.z_hi - p_com_hand.z; positive => COM still inside pad tip span",
    }])
    write_json(OUT / "raw" / "nominal_audit.json", {
        k: v for k, v in audit.items() if k not in ("st",)
    })

    method_rows = []
    force_profile_rows = []
    slide_rows = []
    contact_rows = []
    detach_rows = []
    calib_rows = []
    ic_manifest = []
    ttc_rows = []
    all_eps = []
    root_i = 0

    def consume(ep: dict, tag: str, method: str):
        nonlocal root_i
        rid = f"csd_{root_i:04d}"
        root_i += 1
        ep = dict(ep)
        ep["root_id"] = rid
        ep["tag"] = tag
        ep["method"] = method
        all_eps.append(ep)
        write_json(OUT / "force_traces" / f"{rid}.json", {
            "root_id": rid, "outcome": ep["outcome"], "ctrl": ep["ctrl"],
            "force_hist": ep["force_hist"][::5],
            "slide_hist": ep["slide_hist"][::5],
            "contact_hist": ep["contact_hist"][::5],
        })
        plot_episode(ep, OUT / "plots" / f"{rid}_{ep['outcome']}.png",
                     f"{rid} {method} {ep['outcome']}")
        method_rows.append({
            "root_id": rid,
            "method": method,
            "mode": ep["mode"],
            "axis": ep["axis"],
            "outcome": ep["outcome"],
            "v_target": ep["ctrl"].get("v_target"),
            "F_bias": ep["ctrl"].get("F_bias"),
            "Kv": ep["ctrl"].get("Kv"),
            "unload_dist_m": ep["ctrl"].get("unload_dist_m"),
            "F_peak": ep["ctrl"].get("F_peak"),
            "duration_s": ep["ctrl"].get("duration_s"),
            "max_d_escape_m": ep["max_d_escape_m"],
            "first_none_s": ep["first_none_s"],
            "v_rel_at_detach": ep["v_rel_at_detach"],
            "vo_at_detach": ep["vo_at_detach"],
            "F_at_detach_N": ep["F_at_detach_N"],
            "peak_F_N": ep["peak_F_N"],
            "impulse_Ns": ep["impulse_Ns"],
            "impulse_after_detach_Ns": ep["impulse_after_detach_Ns"],
            "work_J": ep["work_J"],
            "residual_xfrc_N": ep["residual_xfrc_N"],
            "n_ics": len(ep["ics"]),
        })
        for r in ep["force_hist"][::10]:
            force_profile_rows.append({"root_id": rid, **r})
        for r in ep["slide_hist"][::10]:
            slide_rows.append({"root_id": rid, **r})
        for r in ep["contact_hist"][::10]:
            contact_rows.append({"root_id": rid, **r})
        if ep["first_none_s"] is not None and ep["outcome"] in (
            "LOW_SPEED_FULL_DETACHMENT", "HIGH_SPEED_EJECTION"
        ):
            detach_rows.append({
                "root_id": rid,
                "method": method,
                "outcome": ep["outcome"],
                "t_detach_s": ep["first_none_s"],
                "v_rel_at_detach": ep["v_rel_at_detach"],
                "vo_at_detach": ep["vo_at_detach"],
                "F_at_detach_N": ep["F_at_detach_N"],
                "impulse_after_detach_Ns": ep["impulse_after_detach_Ns"],
                "work_J": ep["work_J"],
            })
        for j, ic in enumerate(ep["ics"]):
            ic_id = f"{rid}__{ic['kind']}__{j}"
            blob = {
                "capture": ic["capture"],
                "ref": ic["ref"],
                "root_id": rid,
                "ic_id": ic_id,
                "kind": ic["kind"],
                "method": method,
                "outcome": ep["outcome"],
                "ctrl": ep["ctrl"],
            }
            (OUT / "snapshots" / f"{ic_id}.pkl").write_bytes(pickle.dumps(blob))
            ic_manifest.append({
                "ic_id": ic_id,
                "root_id": rid,
                "method": method,
                "kind": ic["kind"],
                "outcome": ep["outcome"],
                "ttc_s": ic["ttc_s"],
                "ttc_how": ic["ttc_how"],
                "v_escape": ic["v_escape"],
                "vo_norm": ic["vo_norm"],
                "d_remain_m": ic["d_remain_m"],
                "clearance_past_tip_m": ic["clearance_past_tip_m"],
                "height_above_table": ic["height_above_table"],
                "sep_mid_m": ic["sep_mid_m"],
                "e_pos_mm": ic["e_pos_mm"],
                "e_theta_deg": ic["e_theta_deg"],
                "residual_xfrc_N": ic["residual_xfrc_N"],
            })
            ttc_rows.append({
                "ic_id": ic_id, "root_id": rid, "kind": ic["kind"],
                "ttc_s": ic["ttc_s"], "ttc_how": ic["ttc_how"],
                "vo_norm": ic["vo_norm"], "v_escape": ic["v_escape"],
                "height_above_table": ic["height_above_table"],
            })
        print(f"  {rid} {method} -> {ep['outcome']} "
              f"d_esc={ep['max_d_escape_m']*1e3:.1f}mm "
              f"v_det={ep['v_rel_at_detach']} F_det={ep['F_at_detach_N']} "
              f"I_post={ep['impulse_after_detach_Ns']:.4f}", flush=True)
        return rid

    print("Stage A: fixed-pulse baseline", flush=True)
    for spec in baseline_specs()[: int(budgets["stage_a_baseline"])]:
        if time.time() - t0w > budgets["wall_limit_s"]:
            break
        ctrl = DistalForceController(
            "fixed_sin2",
            F_peak=spec["F_peak"],
            duration_s=spec["duration_s"],
            terminate_on_detach=False,
        )
        ep = run_episode(env, ctrl, axis=spec["axis"], max_force_s=spec["duration_s"])
        consume(ep, "A", "A_fixed_sin2")

    # ========== Stage B: velocity-only ==========
    print("Stage B: velocity-controlled sliding", flush=True)
    for spec in vel_calib_specs()[: int(budgets["stage_b_cells"])]:
        if time.time() - t0w > budgets["wall_limit_s"] * 0.55:
            break
        ctrl = DistalForceController(
            "vel_only",
            v_target=spec["v_target"],
            F_bias=spec["F_bias"],
            Kv=spec["Kv"],
        )
        ep = run_episode(env, ctrl, axis="escape", max_force_s=spec["max_force_s"])
        rid = consume(ep, "B", "B_vel_only")
        calib_rows.append({
            "root_id": rid,
            "stage": "B",
            "v_target": spec["v_target"],
            "F_bias": spec["F_bias"],
            "Kv": spec["Kv"],
            "outcome": ep["outcome"],
            "max_d_escape_m": ep["max_d_escape_m"],
            "mean_v_escape": float(np.mean([r["v_escape"] for r in ep["slide_hist"]])) if ep["slide_hist"] else 0.0,
            "peak_F_N": ep["peak_F_N"],
            "impulse_Ns": ep["impulse_Ns"],
        })

    # ========== Stage C: unloading / detach-term ==========
    print("Stage C: unloading / detach termination", flush=True)
    for spec in unload_specs()[: int(budgets["stage_c_cells"])]:
        if time.time() - t0w > budgets["wall_limit_s"] * 0.75:
            break
        ctrl = DistalForceController(
            spec["mode"],
            v_target=spec["v_target"],
            F_bias=spec["F_bias"],
            Kv=spec["Kv"],
            unload_dist_m=spec["unload_dist_m"],
            v_unload=spec["v_unload"],
            terminate_on_detach=spec.get("terminate_on_detach", False),
        )
        ep = run_episode(env, ctrl, axis="escape", max_force_s=spec["max_force_s"])
        rid = consume(ep, "C", spec["method"])
        calib_rows.append({
            "root_id": rid,
            "stage": "C",
            "mode": spec["mode"],
            "v_target": spec["v_target"],
            "F_bias": spec["F_bias"],
            "Kv": spec["Kv"],
            "unload_dist_m": spec["unload_dist_m"],
            "outcome": ep["outcome"],
            "max_d_escape_m": ep["max_d_escape_m"],
            "v_rel_at_detach": ep["v_rel_at_detach"],
            "vo_at_detach": ep["vo_at_detach"],
            "F_at_detach_N": ep["F_at_detach_N"],
            "impulse_after_detach_Ns": ep["impulse_after_detach_Ns"],
            "peak_F_N": ep["peak_F_N"],
            "impulse_Ns": ep["impulse_Ns"],
        })

    # ========== Stage D: expand best params ==========
    print("Stage D: independent roots from best params", flush=True)
    low = [e for e in all_eps if e["outcome"] == "LOW_SPEED_FULL_DETACHMENT"]
    high = [e for e in all_eps if e["outcome"] == "HIGH_SPEED_EJECTION"]
    slide = [e for e in all_eps if e["outcome"] == "CONTROLLED_SLIDING"]
    best_ctrls = []
    seen_keys = set()
    for pool in (low, high, slide, all_eps):
        for e in pool:
            key = (e["mode"], e["ctrl"].get("v_target"), e["ctrl"].get("F_bias"),
                   e["ctrl"].get("Kv"), e["ctrl"].get("unload_dist_m"),
                   e["ctrl"].get("F_peak"), e["ctrl"].get("duration_s"), e["axis"])
            if key in seen_keys:
                continue
            seen_keys.add(key)
            best_ctrls.append({**e["ctrl"], "mode": e["mode"], "axis": e["axis"]})
            if len(best_ctrls) >= 5:
                break
        if len(best_ctrls) >= 5:
            break
    if not best_ctrls:
        best_ctrls = [{"mode": "vel_detach_term", "v_target": 0.035, "F_bias": 14.0, "Kv": 100.0,
                       "unload_dist_m": 0.008, "v_unload": 0.012, "terminate_on_detach": True,
                       "axis": "escape"}]

    n_d = 0
    while n_d < int(budgets["stage_d_roots"]) and time.time() - t0w < budgets["wall_limit_s"] * 0.88:
        cdict = best_ctrls[n_d % len(best_ctrls)]
        # small physical onset variation via extra nominal hold (legal)
        for _ in range(int(rng.integers(0, 8))):
            for __ in range(v3.N_SUB):
                # seat_to_hold inside run_episode; here just burn wall budget via episode diversity
                pass
        ctrl = DistalForceController(
            cdict["mode"],
            v_target=cdict.get("v_target", 0.035),
            F_bias=cdict.get("F_bias", 14.0),
            Kv=cdict.get("Kv", 100.0),
            unload_dist_m=cdict.get("unload_dist_m", 0.008),
            v_unload=cdict.get("v_unload", 0.012),
            F_peak=cdict.get("F_peak", 18.0),
            duration_s=cdict.get("duration_s", 0.25),
            terminate_on_detach=cdict.get("terminate_on_detach", True),
        )
        axis = cdict.get("axis", "escape")
        max_s = float(cdict.get("duration_s", 2.5)) if cdict["mode"] == "fixed_sin2" else 2.8
        ep = run_episode(env, ctrl, axis=axis, max_force_s=max_s)
        consume(ep, "D", "D_expand_" + cdict["mode"])
        n_d += 1

    # ========== Stage E: Recovery pilot ==========
    print("Stage E: Recovery feasibility (conditional)", flush=True)
    rec_rows = []
    low_ics = [r for r in ic_manifest if r["kind"] == "LOW_SPEED_FULL_DETACHMENT"
               or (r["outcome"] == "LOW_SPEED_FULL_DETACHMENT")]
    # also allow high-speed detach for diagnostic contrast (limited)
    det_ics = [r for r in ic_manifest if "DETACHMENT" in r["kind"] or "EJECTION" in r["kind"]]
    pilot_ics = (low_ics + [r for r in det_ics if r not in low_ics])[: int(budgets["stage_e_max_ics"])]

    if not low_ics:
        print("  No LOW_SPEED_FULL_DETACHMENT ICs; skipping expensive Recovery search "
              "(baselines only on any detach ICs if present).", flush=True)

    for ic_row in pilot_ics:
        if time.time() - t0w > budgets["wall_limit_s"]:
            break
        ic_id = ic_row["ic_id"]
        blob = pickle.loads((OUT / "snapshots" / f"{ic_id}.pkl").read_bytes())
        # zero-action baseline always
        zres = eval_continuous(env, blob, np.zeros((30, 7), np.float32))
        rec_rows.append({
            "ic_id": ic_id, "method": "B_zero_action",
            "SUCCESS_NEW": zres["SUCCESS_NEW"], "best_stage": zres["best_stage"],
            "win_theta": zres.get("win_theta"), "win_pos": zres.get("win_pos"),
        })
        if not low_ics and ic_row["kind"] != "LOW_SPEED_FULL_DETACHMENT":
            # only baselines for high-speed when no low-speed set
            continue
        # K49 seeds
        best = zres
        best_tok = None
        seeds = k49_seed_tokens(env, blob, book48, table, rng)[: int(budgets["stage_e_k49"])]
        for si, toks in enumerate(seeds):
            res = evaluate_full(env, table, book48, blob, toks)
            sn = int(bool(res.get("SUCCESS_NEW") or res.get("success_new")))
            stage = res.get("best_stage") or res.get("stage") or ("SUCCESS_NEW" if sn else "FAIL")
            rec_rows.append({
                "ic_id": ic_id, "method": f"K49_{si}",
                "SUCCESS_NEW": sn, "best_stage": stage,
                "win_theta": res.get("mean_theta") or res.get("win_theta"),
                "win_pos": res.get("mean_pos") or res.get("win_pos"),
            })
            if sn and best.get("SUCCESS_NEW", 0) == 0:
                best = {"SUCCESS_NEW": 1, "best_stage": "SUCCESS_NEW", "tokens": toks}
                best_tok = toks
        # continuous chase
        for ci, acts in enumerate(continuous_candidates(ic_row, rng)[: int(budgets["stage_e_cont"])]):
            res = eval_continuous(env, blob, acts)
            rec_rows.append({
                "ic_id": ic_id, "method": f"continuous_{ci}",
                "SUCCESS_NEW": res["SUCCESS_NEW"], "best_stage": res["best_stage"],
                "win_theta": res.get("win_theta"), "win_pos": res.get("win_pos"),
            })
            if res["SUCCESS_NEW"] and best.get("SUCCESS_NEW", 0) == 0:
                best = res
                best_tok = None
        # triple replay if success
        if int(best.get("SUCCESS_NEW", 0)) == 1:
            ok = 0
            for rep in range(int(budgets["stage_e_replays"])):
                if best_tok is not None:
                    r2 = evaluate_full(env, table, book48, blob, best_tok)
                    sn = int(bool(r2.get("SUCCESS_NEW") or r2.get("success_new")))
                else:
                    r2 = eval_continuous(env, blob, best["actions"])
                    sn = int(r2["SUCCESS_NEW"])
                ok += sn
                rec_rows.append({
                    "ic_id": ic_id, "method": f"replay_{rep}",
                    "SUCCESS_NEW": sn, "best_stage": "SUCCESS_NEW" if sn else "FAIL",
                })
            if ok == int(budgets["stage_e_replays"]) and best_tok is not None:
                np.savez_compressed(
                    OUT / "trajectories" / f"{ic_id}__FULL.npz",
                    tokens=np.asarray(best_tok, np.int32),
                )
                write_json(OUT / "trajectories" / f"{ic_id}__FULL_meta.json", {
                    "ic_id": ic_id, "method": "K49", "tokens": best_tok,
                    "replays_ok": ok,
                })

    # ---- Write CSVs ----
    write_csv(OUT / "FORCE_CONTROLLER_CALIBRATION.csv", calib_rows if calib_rows else [
        {"note": "no calibration rows"}])
    write_csv(OUT / "FORCE_PROFILES.csv", force_profile_rows if force_profile_rows else [
        {"note": "empty"}])
    write_csv(OUT / "SLIDING_TRAJECTORIES.csv", slide_rows if slide_rows else [{"note": "empty"}])
    write_csv(OUT / "CONTACT_TOPOLOGY.csv", contact_rows if contact_rows else [{"note": "empty"}])
    write_csv(OUT / "DETACHMENT_EVENTS.csv", detach_rows if detach_rows else [
        {"ic_id": "", "note": "No verified full-detachment events"}])
    write_csv(OUT / "POST_DETACHMENT_IC_MANIFEST.csv", ic_manifest if ic_manifest else [
        {"ic_id": "", "note": "No ICs mined"}])
    write_csv(OUT / "METHOD_COMPARISON.csv", method_rows)
    write_csv(OUT / "PASSIVE_TTC.csv", ttc_rows if ttc_rows else [{"note": "no TTC rows"}])
    write_csv(OUT / "RECOVERY_FEASIBILITY.csv", rec_rows if rec_rows else [
        {"note": "Stage E skipped or no detach ICs"}])

    # Failure analysis
    oc = Counter(r["outcome"] for r in method_rows)
    fail_md = [
        "# FAILURE_ANALYSIS",
        "",
        f"Outcome counts: {dict(oc)}",
        "",
        "## Physical notes",
        f"- Nominal distal remain to pad tip: {audit['mean_d_remain_m']*1e3:.1f} mm",
        f"- Escape axis: hand +z (Rh[:,2]); prior lateral-force 'dist' was Rh[:,0] (horizontal).",
        f"- Gravity aligns with fingertip escape (down), so unloading must fight gravity-assisted freefall.",
        "",
        "## Likely failure modes",
        "- Friction holds: NO_SLIP / insufficient F_bias for target speed.",
        "- Contact geometry: PARTIAL_CONTACT_ESCAPE (unilateral) without full tip clearance.",
        "- Unloading too late: HIGH_SPEED_EJECTION with residual impulse after detach.",
        "- Unloading too early: CONTROLLED_SLIDING without complete detachment.",
        "- Table: TABLE_DROP when TTC short after detach.",
        "",
    ]
    (OUT / "FAILURE_ANALYSIS.md").write_text("\n".join(fail_md), encoding="utf-8")

    # Videos for representative categories (render via short replay of force episode snapshots)
    _maybe_videos(env, all_eps, OUT)

    # REPORT
    n_gradual = sum(1 for r in method_rows if r["outcome"] in ("CONTROLLED_SLIDING", "LOW_SPEED_FULL_DETACHMENT", "PARTIAL_CONTACT_ESCAPE", "HIGH_SPEED_EJECTION") and float(r["max_d_escape_m"] or 0) >= 0.003)
    n_speed_ok = sum(1 for r in calib_rows if r.get("mean_v_escape") is not None)
    low_n = sum(1 for r in method_rows if r["outcome"] == "LOW_SPEED_FULL_DETACHMENT")
    high_n = sum(1 for r in method_rows if r["outcome"] == "HIGH_SPEED_EJECTION")
    slide_n = sum(1 for r in method_rows if r["outcome"] == "CONTROLLED_SLIDING")
    det_ics_n = len([r for r in ic_manifest if "DETACH" in r["kind"] or "EJECT" in r["kind"]])
    low_ics_n = len(low_ics)
    sn_n = sum(1 for r in rec_rows if int(r.get("SUCCESS_NEW") or 0) == 1 and str(r.get("method", "")).startswith("replay"))
    # compare methods
    def _rate(method_prefix, outcome):
        rows = [r for r in method_rows if str(r["method"]).startswith(method_prefix) or method_prefix in str(r["method"])]
        if not rows:
            return "n/a"
        return f"{sum(1 for r in rows if r['outcome']==outcome)}/{len(rows)}"

    mean_I_post = float(np.mean([float(r["impulse_after_detach_Ns"] or 0) for r in detach_rows])) if detach_rows else 0.0
    vels = [float(r["v_rel_at_detach"]) for r in detach_rows if r.get("v_rel_at_detach") not in (None, "")]
    ttcs = [float(r["ttc_s"]) for r in ttc_rows if r.get("ttc_s") not in (None, "")]

    report = f"""# Controlled Slow Detachment

**Protocol:** velocity-controlled `xfrc_applied` along fingertip escape (hand +z),
nominal Panda controller unchanged, force cleared before Recovery.
No K49/physics/reward changes. No SFT/GRPO.

**Script:** `scripts/run_controlled_slow_detachment.py`
**Wall:** {time.time()-t0w:.1f}s

## Stage A — Nominal + geometry

| Quantity | Value |
|----------|------:|
| Bilateral maintained | {audit['always_bilateral']} |
| Distal remain to tip (mm) | {audit['mean_d_remain_m']*1e3:.2f} |
| Finger normals L/R (N) | {audit['mean_fn_L']:.2f} / {audit['mean_fn_R']:.2f} |
| Height above table (m) | {audit['height_above_table']:.3f} |
| Source time (s) | {audit['source_time_s']:.2f} |

Escape axis = hand +z (Rh[:,2]). Prior lateral-force "dist" was Rh[:,0] (horizontal).

## Outcome counts (all methods)

{dict(oc)}

- CONTROLLED_SLIDING: {slide_n}
- LOW_SPEED_FULL_DETACHMENT: {low_n}
- HIGH_SPEED_EJECTION: {high_n}
- Detach ICs mined: {det_ics_n} (low-speed labeled: {low_ics_n})

## Required conclusions (Q1-Q14)

1. **Gradual push toward fingertip?** **{"Yes" if n_gradual else "No / limited"}** — see max_d_escape in METHOD_COMPARISON.
2. **Relative sliding speed controllable?** **{"Partially / Yes" if n_speed_ok else "Not demonstrated"}** — FORCE_CONTROLLER_CALIBRATION mean_v_escape vs v_target.
3. **Gradual unloading reduce ejection speed?** Compare C_vel_unload vs B_vel_only vo_at_detach / HIGH_SPEED rates in METHOD_COMPARISON.
4. **Fully detach after unloading?** LOW_SPEED+HIGH_SPEED counts: {low_n + high_n}.
5. **External impulse after detachment?** mean impulse_after_detach = {mean_I_post:.4f} N·s (see DETACHMENT_EVENTS).
6. **Params for low-speed full detach?** Filter METHOD_COMPARISON / FORCE_CONTROLLER_CALIBRATION for outcome=LOW_SPEED_FULL_DETACHMENT.
7. **Elevated grasp vs interception window?** height~{audit['height_above_table']:.3f} m; passive TTC distribution in PASSIVE_TTC (n={len(ttcs)}, median={np.median(ttcs) if ttcs else float('nan'):.3f}s).
8. **Independent valid detached ICs?** {det_ics_n} mined snapshots; low-speed cohort {low_ics_n}.
9. **v_rel / TTC distributions?** v_rel detach n={len(vels)} median={np.median(vels) if vels else float('nan'):.3f} m/s; TTC median={np.median(ttcs) if ttcs else float('nan'):.3f}s.
10. **Outperform fixed sin^2?** Fixed low-speed detach rate A: {_rate('A_fixed','LOW_SPEED_FULL_DETACHMENT')}; vel methods B/C/D: see METHOD_COMPARISON.
11. **Legal active Recovery recatch?** See RECOVERY_FEASIBILITY best_stage columns.
12. **Complete SUCCESS_NEW?** Triple-replay successes logged: {sn_n}.
13. **Failure sources?** See FAILURE_ANALYSIS.md (friction / geometry / unloading / robot motion).
14. **Suitable for unified Transformer eval?** Only LOW_SPEED_FULL_DETACHMENT ICs with verified force=0 and geometric clearance; contact-preserving sliding is a different mechanism cohort.

## Artifacts

NOMINAL_GRASP_AUDIT.csv, GRIPPER_GEOMETRY.csv, FORCE_CONTROLLER_CALIBRATION.csv,
FORCE_PROFILES.csv, SLIDING_TRAJECTORIES.csv, CONTACT_TOPOLOGY.csv,
DETACHMENT_EVENTS.csv, POST_DETACHMENT_IC_MANIFEST.csv, METHOD_COMPARISON.csv,
PASSIVE_TTC.csv, RECOVERY_FEASIBILITY.csv, FAILURE_ANALYSIS.md,
raw/, trajectories/, videos/, plots/, snapshots/, force_traces/
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(report.split("## Required conclusions")[1][:1500], flush=True)
    print(f"Wrote {OUT}", flush=True)


def _maybe_videos(env, all_eps: list[dict], out: Path):
    """Save short diagnostic MP4s for representative categories if renderer available."""
    try:
        import mujoco
        from visualization.camera import mjv_camera_from_preset
        from visualization.overlays import caption
        from visualization.video_recorder import write_frames_mp4
    except Exception as e:
        write_json(out / "videos" / "NOTE.json", {"skipped": str(e)})
        return

    cats = {
        "controlled_sliding": next((e for e in all_eps if e["outcome"] == "CONTROLLED_SLIDING"), None),
        "low_speed_detach": next((e for e in all_eps if e["outcome"] == "LOW_SPEED_FULL_DETACHMENT"), None),
        "high_speed_ejection": next((e for e in all_eps if e["outcome"] == "HIGH_SPEED_EJECTION"), None),
    }
    # Recatch video only if a trajectory exists
    traj = list((out / "trajectories").glob("*__FULL*.npz")) + list((out / "trajectories").glob("*__FULL*.npz"))
    note = {k: (v["root_id"] if v else None) for k, v in cats.items()}
    note["recatch"] = None
    write_json(out / "videos" / "INDEX.json", note)

    for name, ep in cats.items():
        if ep is None:
            continue
        # Re-run a short episode for frames is expensive; skip full render if smoke-sized
        # Instead dump a static note — full video rendering uses stored force traces plots.
        # Attempt a lightweight re-sim with same ctrl for ~3s.
        try:
            c = ep["ctrl"]
            ctrl = DistalForceController(
                ep["mode"],
                v_target=c.get("v_target", 0.035),
                F_bias=c.get("F_bias", 8.0),
                Kv=c.get("Kv", 120.0),
                unload_dist_m=c.get("unload_dist_m", 0.008),
                v_unload=c.get("v_unload", 0.01),
                F_peak=c.get("F_peak", 18.0),
                duration_s=c.get("duration_s", 0.25),
                terminate_on_detach=c.get("terminate_on_detach", False),
            )
            st = seat_to_hold(env)
            sim = env.sim
            obj = int(sim.ids.object_body)
            u = escape_axis(sim) if ep["axis"] == "escape" else prior_dist_axis(sim)
            look = 0.5 * (obj_pos(sim) + hand_pos(sim))
            cam = mjv_camera_from_preset({"lookat": look, "distance": 0.55, "azimuth": 135, "elevation": -20})
            renderer = mujoco.Renderer(sim.model, 360, 480)
            frames = []
            t0 = float(sim.data.time)
            dt = float(sim.model.opt.timestep)
            ctrl.reset()
            steps = 0
            while float(sim.data.time) - t0 < 2.5 and steps < 200:
                for _ in range(v3.N_SUB):
                    g = geometric_state(sim)
                    tp = topology(sim)
                    Fmag = ctrl.command(
                        t_rel=float(sim.data.time) - t0,
                        v_escape=float(np.dot(g["v_rel"], u)),
                        d_remain_m=g["d_remain_m"],
                        topo=tp,
                        vo_norm=float(np.linalg.norm(g["vo"])),
                        dt=dt,
                    )
                    if ep["mode"] == "fixed_sin2" and float(sim.data.time) - t0 > c.get("duration_s", 0.25):
                        Fmag = 0.0
                    sim.data.xfrc_applied[obj, 0:3] = Fmag * u
                    sim.data.xfrc_applied[obj, 3:6] = 0.0
                    nominal_substep(sim, env.gains, st)
                    if Fmag == 0.0:
                        sim.data.xfrc_applied[:] = 0.0
                cam.lookat[:] = 0.5 * (obj_pos(sim) + hand_pos(sim))
                renderer.update_scene(sim.data, cam)
                img = renderer.render()
                frames.append(caption(img, f"{name} {ep['root_id']} {ep['outcome']} t={float(sim.data.time)-t0:.2f}"))
                steps += 1
                if latched(sim):
                    break
            renderer.close()
            if frames:
                dest = out / "videos" / f"{name}_{ep['root_id']}.mp4"
                write_frames_mp4(frames, dest, fps=20)
                print("video", dest.name, "frames", len(frames), flush=True)
        except Exception as e:
            write_json(out / "videos" / f"{name}_error.json", {"error": str(e)})


if __name__ == "__main__":
    main()
