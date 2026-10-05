"""Audited free-flight ballistic impact during nominal LIFT (noslip=1). ZERO only.

Does not run recovery, RULE, SAC, recatch, or retune the gravitational heuristic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.jacobian_controller import gains_from_cfg
from envs.config_util import load_yaml, merge_sim_config
from envs.deterioration import body_twist
from envs.dynamics import set_finger_object_sliding_mu
from envs.grasp_sim import GraspSim, apply_solver_from_cfg, reset_home
from envs.ids import resolve_ids
from envs.physical_recovery import TABLE_DROP, Z_AIR, physical_pack
from envs.xml_build import write_panda_torque
from training.demo_grav_reposition_recovery import TAU_SEC, V_LIFT, dropped, live_opt
from training.grav_reposition_v2_viz import PRE_VIEW_PAUSE_S, apply_camera_preset, make_reset_cam_callback
from training.impact_ball import apply_ball_free_state, incoming_hand_x
from training.impact_visualization_utils import mjv_camera_from_preset
from training.replay_core import tick_vw
from training.vertical_slip_contact_mechanics_audit import extract_contacts
from training.write_ballistic_impact_scene import (
    BALL_I,
    BALL_M,
    BALL_R,
    CYL_IXX,
    CYL_IZZ,
    CYL_M,
    write_ballistic_scene,
)

OUT = ROOT / "results" / "diagnostics" / "ballistic_impact_noslip1"
RAW = OUT / "raw"
FIG = OUT / "figures"
VID = OUT / "videos"
SNAP = RAW / "snapshots"

PAIR_MU = 1.0
HOLD_S = 12.0
AFTER_DROP_S = 2.0
HAND_RISE_M = 0.05
D_LAUNCH = 0.18
LOG_DT = 0.010
FINE_WIN = 0.20
IMMEDIATE_S = 0.10
STABLE_EX_MM = 3.0
SWEEP_V = (1.20, 1.80, 2.40, 2.70, 2.85, 2.95, 3.05, 3.20, 4.00)
VIDEO_FPS = 25
RENDER_W, RENDER_H = 640, 480
CAM = {"lookat": np.array([0.50, -0.08, 0.50]), "distance": 1.70, "azimuth": 125.0, "elevation": -18.0}


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return None
    return x


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(obj), indent=2), encoding="utf-8")


def assert_noslip(sim) -> None:
    n = int(sim.model.opt.noslip_iterations)
    if n != 1:
        raise RuntimeError(f"noslip_iterations={n}, expected 1")


def geom_name(model, gid: int) -> str:
    n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(gid))
    return str(n) if n else f"geom_{int(gid)}"


class BallisticSim(GraspSim):
    def __init__(self, cfg: dict):
        scene = write_ballistic_scene()
        write_panda_torque()
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        apply_solver_from_cfg(self.model, cfg)
        self.data = mujoco.MjData(self.model)
        self.ids = resolve_ids(self.model)
        from controllers.nominal import GraspFSM
        from controllers.residual import ResidualLimiter
        from envs.deterioration import DeteriorationMeter

        self.fsm = GraspFSM(cfg)
        self.meter = DeteriorationMeter(cfg)
        self.limiter = ResidualLimiter(cfg)
        self.n_sub = int(cfg.get("n_substeps", 10))
        self.dt_policy = float(self.model.opt.timestep) * self.n_sub
        self.mass = CYL_M
        self.friction = PAIR_MU
        self.captured = False
        self.ball_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "impact_ball")
        self.ball_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "impact_ball")
        self.ball_jnt = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_joint")
        self.ball_qadr = int(self.model.jnt_qposadr[self.ball_jnt])
        self.ball_dadr = int(self.model.jnt_dofadr[self.ball_jnt])
        self.guide_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball_guide")
        self.guide_eq = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "ball_guide_weld")
        self.guide_mocap = int(self.model.body_mocapid[self.guide_body])

    def set_guide_weld(self, on: bool) -> None:
        flag = int(bool(on))
        if hasattr(self.data, "eq_active"):
            self.data.eq_active[self.guide_eq] = flag
        if hasattr(self.model, "eq_active0"):
            self.model.eq_active0[self.guide_eq] = flag

    def set_guide_pos(self, p) -> None:
        self.data.mocap_pos[self.guide_mocap] = np.asarray(p, float).reshape(3)
        self.data.mocap_quat[self.guide_mocap] = np.array([1.0, 0.0, 0.0, 0.0])

    def eq_active(self) -> int:
        if hasattr(self.data, "eq_active"):
            return int(self.data.eq_active[self.guide_eq])
        return int(self.model.eq_active0[self.guide_eq])

    def reset(self, mass=None, friction=None, grasp_offset=None) -> None:
        self.mass = CYL_M
        self.friction = PAIR_MU
        reset_home(self.model, self.data)
        set_finger_object_sliding_mu(self.model, self.ids, PAIR_MU, data=None)
        self.set_guide_weld(True)
        hold = np.array([0.50, -0.40, 0.55])
        apply_ball_free_state(self, hold, np.zeros(3))
        self.set_guide_pos(hold)
        mujoco.mj_forward(self.model, self.data)
        off = np.zeros(3) if grasp_offset is None else np.asarray(grasp_offset, dtype=float)
        self.fsm.reset(off)
        self.meter.reset()
        self.limiter.reset()
        self.captured = False
        assert_noslip(self)


def make_sim() -> tuple[BallisticSim, dict]:
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    sim = BallisticSim(cfg)
    sim.reset()
    assert_noslip(sim)
    return sim, cfg


def xml_mass_declared() -> dict:
    text = write_ballistic_scene().read_text(encoding="utf-8")
    return {
        "scene": str(write_ballistic_scene()),
        "xml_contains_ball_0p05": 'inertial mass="0.05"' in text or "inertial mass=\"0.05\"" in text,
        "xml_contains_cyl_0p20": 'inertial mass="0.2"' in text or 'inertial mass="0.20"' in text,
        "declared_ball_m": BALL_M,
        "declared_ball_I": BALL_I,
        "declared_cyl_m": CYL_M,
        "declared_cyl_Ixx": CYL_IXX,
        "declared_cyl_Izz": CYL_IZZ,
        "ball_r": BALL_R,
    }


def qM_linear_mass(sim, body_id: int) -> float | None:
    """Diagonal qM entry for the first translational dof of a free joint, if sparse layout allows."""
    jnt = int(sim.model.body_jntadr[body_id])
    if jnt < 0:
        return None
    dof = int(sim.model.jnt_dofadr[jnt])
    mujoco.mj_forward(sim.model, sim.data)
    M = np.zeros((sim.model.nv, sim.model.nv))
    mujoco.mj_fullM(sim.model, sim.data, M)
    return float(M[dof, dof])


def audit_mass(sim) -> dict:
    xml = xml_mass_declared()
    bb = int(sim.ball_body)
    ob = int(sim.ids.object_body)
    out = {
        "xml": xml,
        "noslip_iterations": int(sim.model.opt.noslip_iterations),
        "ball": {
            "body_id": bb,
            "body_mass": float(sim.model.body_mass[bb]),
            "body_inertia": np.array(sim.model.body_inertia[bb], float).tolist(),
            "qM_lin0": qM_linear_mass(sim, bb),
        },
        "cylinder": {
            "body_id": ob,
            "body_mass": float(sim.model.body_mass[ob]),
            "body_inertia": np.array(sim.model.body_inertia[ob], float).tolist(),
            "qM_lin0": qM_linear_mass(sim, ob),
        },
        "runtime_set_ball_mass_used": False,
        "mj_setConst_used_for_ball": False,
    }
    out["ball_mass_ok"] = abs(out["ball"]["body_mass"] - BALL_M) < 1e-9
    out["cyl_mass_ok"] = abs(out["cylinder"]["body_mass"] - CYL_M) < 1e-9
    out["ball_qM_ok"] = out["ball"]["qM_lin0"] is not None and abs(out["ball"]["qM_lin0"] - BALL_M) < 1e-6
    if not out["ball_mass_ok"] or not out["cyl_mass_ok"]:
        raise RuntimeError(out)
    return out


def free_flight_drop_audit(sim) -> dict:
    """Weld off, ball at rest, measure gravitational acceleration."""
    sim.reset()
    sim.set_guide_weld(False)
    p0 = np.array([0.70, 0.40, 1.10])
    apply_ball_free_state(sim, p0, np.zeros(3))
    dt = float(sim.model.opt.timestep)
    t_run = 0.40
    n = int(round(t_run / dt))
    z = []
    for _ in range(n):
        mujoco.mj_step(sim.model, sim.data)
        z.append(float(sim.data.qpos[sim.ball_qadr + 2]))
    t = np.arange(1, n + 1) * dt
    z = np.array(z)
    # z = z0 - 0.5 g t^2
    pred = p0[2] + 0.5 * float(sim.model.opt.gravity[2]) * t * t
    err = float(np.max(np.abs(z - pred)))
    # finite-diff accel over last half
    i0 = n // 2
    a = (z[-1] - 2 * z[(n + i0) // 2] + z[i0]) / (((n - i0) // 2 * dt) ** 2)
    return {
        "t": t_run,
        "z0": float(p0[2]),
        "z_end": float(z[-1]),
        "z_pred_end": float(pred[-1]),
        "max_abs_z_error": err,
        "g_model": float(sim.model.opt.gravity[2]),
        "eq_active": sim.eq_active(),
        "ok": err < 1.0e-2 and sim.eq_active() == 0,
    }


def ball_force_on_ball(sim) -> tuple[np.ndarray, list]:
    """World force on the ball from contacts, using audited mj_contactForce semantics."""
    model, data = sim.model, sim.data
    wr = np.zeros(6)
    F = np.zeros(3)
    rows = []
    bg = int(sim.ball_geom)
    og = int(sim.ids.object_geom)
    for i in range(int(data.ncon)):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if bg not in (g1, g2):
            continue
        mujoco.mj_contactForce(model, data, i, wr)
        n = np.array(c.frame[0:3], float)
        t1 = np.array(c.frame[3:6], float)
        t2 = np.array(c.frame[6:9], float)
        fn, ft1, ft2 = float(wr[0]), float(wr[1]), float(wr[2])
        f_on_geom2 = n * fn + t1 * ft1 + t2 * ft2
        if g2 == bg:
            f_ball = f_on_geom2
            on = "geom2"
        else:
            f_ball = -f_on_geom2
            on = "geom1_negated"
        F += f_ball
        other = g2 if g1 == bg else g1
        kind = "ball-cylinder" if other == og else "ball-other"
        rows.append(
            {
                "kind": kind,
                "g1": geom_name(model, g1),
                "g2": geom_name(model, g2),
                "pos": np.array(c.pos, float).copy(),
                "normal": n.copy(),
                "fn": fn,
                "ft": float(np.hypot(ft1, ft2)),
                "f_ball_w": f_ball.copy(),
                "on": on,
            }
        )
    return F, rows


def pack_row(sim, tau: float) -> dict:
    o = physical_pack(sim)
    Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    Ro = np.array(sim.data.xmat[sim.ids.object_body].reshape(3, 3), float)
    vh, wh = body_twist(sim.model, sim.data, sim.ids.hand_body)
    vo, wo = body_twist(sim.model, sim.data, sim.ids.object_body)
    q = sim.ball_qadr
    d = sim.ball_dadr
    pb = np.array(sim.data.qpos[q : q + 3], float)
    vb = np.array(sim.data.qvel[d : d + 3], float)
    wb = np.array(sim.data.qvel[d + 3 : d + 6], float)
    Fb, bc = ball_force_on_ball(sim)
    fc, _ = extract_contacts(sim)
    return {
        "t": float(sim.data.time),
        "fsm": str(sim.fsm.phase),
        "p_ball": pb,
        "v_ball": vb,
        "w_ball": wb,
        "p_obj": np.array(sim.data.xpos[sim.ids.object_body], float).copy(),
        "R_obj": Ro.copy(),
        "v_obj": vo,
        "w_obj": wo,
        "p_hand": np.array(sim.data.xpos[sim.ids.hand_body], float).copy(),
        "R_hand": Rh.copy(),
        "v_hand": vh,
        "w_hand": wh,
        "rh": np.asarray(o["rh"], float).copy(),
        "R_rel": np.asarray(o["R_rel"], float).copy(),
        "v_rel_h": np.asarray(o["v_rel_h"], float).copy(),
        "w_rel_h": np.asarray(o["w_rel_h"], float).copy(),
        "cyl_h": (Rh.T @ Ro[:, 2]).copy(),
        "nL": int(o["nL"]),
        "nR": int(o["nR"]),
        "Fn_L": float(sum(abs(r["Fn"]) for r in fc if r["side"] == "L")),
        "Fn_R": float(sum(abs(r["Fn"]) for r in fc if r["side"] == "R")),
        "rho_max": float(np.nanmax([r["rho"] for r in fc])) if fc else float("nan"),
        "ball_n": len(bc),
        "ball_cyl": int(any(r["kind"] == "ball-cylinder" for r in bc)),
        "ball_other": int(any(r["kind"] != "ball-cylinder" for r in bc)),
        "F_ball_w": Fb,
        "p_des": np.asarray(sim.fsm.p_des, float).copy(),
        "v_cmd": np.asarray(sim.fsm.v_cmd, float).copy(),
        "tau": float(tau),
        "ctrl7": float(sim.data.ctrl[7]) if sim.data.ctrl.size > 7 else float(tau),
        "obj_z": float(o["obj_z"]),
        "clear": float(o.get("clear", np.nan)),
        "eq_active": sim.eq_active(),
        "bc": bc,
        "fc": fc,
    }


def slim_row(r: dict) -> dict:
    skip = {"bc", "fc", "R_obj", "R_hand", "R_rel"}
    out = {}
    for k, v in r.items():
        if k in skip:
            continue
        out[k] = v
    return out


def first_ball_kind(bc: list) -> str | None:
    if not bc:
        return None
    return str(bc[0]["kind"])


def classify(ep: dict) -> str:
    if not ep.get("impact_ok"):
        return "INVALID"
    td = ep.get("t_drop")
    ti = ep["t_impact"]
    if td is not None and (td - ti) < IMMEDIATE_S:
        return "IMMEDIATE_LOSS"
    if td is not None:
        return "DELAYED_FAILURE"
    if ep.get("hold_survived") and ep.get("max_ex_mm", 99) < STABLE_EX_MM and not ep.get("unilateral"):
        return "STABLE"
    if ep.get("hold_survived"):
        return "DISTURBED_CAPTURED"
    if td is None and not ep.get("hold_survived"):
        return "DISTURBED_CAPTURED" if ep.get("end_nL", 0) + ep.get("end_nR", 0) > 0 else "DELAYED_FAILURE"
    return "DISTURBED_CAPTURED"


def z_tgt_of(sim) -> float:
    return float(sim.fsm.grasp_z) + float(sim.cfg["fsm"]["lift_offset_z"])


def launch_state(sim, v_hit: float) -> dict:
    """Free-flight IC aimed at current object along +hand-x. Uses THIS scene's ball radius."""
    u = incoming_hand_x(sim)
    po = np.array(sim.data.xpos[sim.ids.object_body], float)
    r_ball = float(sim.model.geom_size[int(sim.ball_geom)][0])
    r_obj = float(sim.model.geom_size[int(sim.ids.object_geom)][0])
    r_surf = r_obj + r_ball
    g = np.array(sim.model.opt.gravity, float)
    p_hit = po - u * r_surf
    p_launch = po - u * float(D_LAUNCH)
    dist_hit = float(np.linalg.norm(p_hit - p_launch))
    T = dist_hit / max(float(v_hit), 1e-9)
    v0 = (p_hit - p_launch) / T - 0.5 * g * T
    return {
        "p_target": po.copy(),
        "p_hit": p_hit,
        "p_launch": p_launch,
        "v0": v0,
        "direction": u,
        "d_launch": float(D_LAUNCH),
        "T_hit_s": T,
        "r_surf": r_surf,
        "v_impact_requested": float(v_hit),
        "gravity": g.copy(),
        "v0_speed": float(np.linalg.norm(v0)),
        "hand_x_world": u.tolist(),
    }


def run_episode(sim, gains, v_hit: float, *, viz=None, record=False, label="ZERO") -> dict:
    assert_noslip(sim)
    sim.reset()
    dt = float(sim.model.opt.timestep)
    log_n = max(int(round(LOG_DT / dt)), 1)
    rows = []
    events = []
    hand_z0 = None
    t_lift0 = None
    released = False
    t_release = None
    t_impact = None
    launch = None
    impact_ok = False
    impact_reject = None
    first_c = None
    v_pre = None
    vo_pre = None
    Rh_imp = None
    J = np.zeros(3)
    in_contact = False
    t_last_c = None
    t_drop = None
    t_ul = None
    t_both = None
    t_lift_done = None
    snap_post = None
    snap_ul = None
    snap_loss = None
    max_ex = 0.0
    unilateral = False
    z_tgt = None
    holding = False
    pres = viz
    n_max = int(round(28.0 / dt))
    for i in range(n_max):
        if (not holding) and sim.fsm.phase == "lift" and z_tgt is not None:
            if float(sim.fsm.p_des[2]) >= z_tgt - 1e-9:
                holding = True
                t_lift_done = float(sim.data.time)

        if holding:
            tick_vw(sim, np.zeros(3), np.zeros(3), TAU_SEC, gains)
        else:
            sim.physics_step(None, in_recovery=False)
            sim.maybe_capture_reference()

        t = float(sim.data.time)
        if sim.fsm.phase == "lift" and hand_z0 is None:
            hand_z0 = float(sim.data.xpos[sim.ids.hand_body][2])
            t_lift0 = t
            z_tgt = z_tgt_of(sim)
            events.append({"event": "lift_start", "t": t})

        if released:
            Fb, bc = ball_force_on_ball(sim)
            has_cyl = any(r["kind"] == "ball-cylinder" for r in bc)
            has_other = any(r["kind"] != "ball-cylinder" for r in bc)
            if t_impact is None and has_other and not has_cyl:
                impact_reject = "first ball contact was not cylinder: " + ",".join(r["kind"] for r in bc)
                break
            if t_impact is None and has_cyl:
                if sim.eq_active() != 0:
                    impact_reject = f"eq_active={sim.eq_active()} at first contact"
                    break
                t_impact = t
                first_c = next(r for r in bc if r["kind"] == "ball-cylinder")
                Rh_imp = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
                impact_ok = True
                events.append({"event": "impact", "t": t_impact})
                snap_post = sim.snapshot()
            if has_cyl:
                J += Fb * dt
                t_last_c = t

        o = physical_pack(sim)
        hz = float(sim.data.xpos[sim.ids.hand_body][2])
        if (
            (not released)
            and sim.fsm.phase == "lift"
            and sim.captured
            and hand_z0 is not None
            and float(o["obj_z"]) >= Z_AIR
            and (hz - hand_z0) >= HAND_RISE_M
        ):
            launch = launch_state(sim, v_hit)
            sim.set_guide_weld(False)
            apply_ball_free_state(sim, launch["p_launch"], launch["v0"])
            released = True
            t_release = float(sim.data.time)
            v_pre = np.array(sim.data.qvel[sim.ball_dadr : sim.ball_dadr + 3], float).copy()
            vo_pre = body_twist(sim.model, sim.data, sim.ids.object_body)[0].copy()
            events.append(
                {
                    "event": "release",
                    "t": t_release,
                    "eq_active": sim.eq_active(),
                    "v0": launch["v0"].tolist(),
                    "p_launch": launch["p_launch"].tolist(),
                }
            )
            if sim.eq_active() != 0:
                impact_reject = "eq_active != 0 after release"
                break
        elif released and t_impact is None:
            v_pre = np.array(sim.data.qvel[sim.ball_dadr : sim.ball_dadr + 3], float).copy()
            vo_pre = body_twist(sim.model, sim.data, sim.ids.object_body)[0].copy()

        fine = released and (t_impact is None or abs(t - t_impact) <= FINE_WIN)
        if i % (1 if fine else log_n) == 0:
            r = pack_row(sim, TAU_SEC)
            if t_impact is not None:
                max_ex = max(max_ex, abs(float(r["rh"][0])) * 1e3)
                if r["nL"] == 0 or r["nR"] == 0:
                    unilateral = True
                    if t_ul is None:
                        t_ul = t
                        snap_ul = sim.snapshot()
                if r["nL"] == 0 and r["nR"] == 0 and t_both is None:
                    t_both = t
            rows.append(r)

        if t_impact is not None and t_drop is None and dropped(sim, o):
            t_drop = t
            events.append({"event": "drop", "t": t})
            snap_loss = sim.snapshot()

        if pres is not None:
            pres["phase"] = str(sim.fsm.phase).upper() if t_impact is None else (
                "IMPACT" if t_impact is not None and t < t_impact + 0.15 else ("DROP" if t_drop else ("HIGH HOLD" if holding else "LIFT"))
            )
            if released and t_impact is None:
                pres["phase"] = "FREE FLIGHT"
            if t_release is not None and t_impact is None and t >= t_release:
                pres["phase"] = "FREE FLIGHT"
            pres_show(pres, sim, label)

        if t_drop is not None and t >= t_drop + AFTER_DROP_S:
            break
        if t_lift_done is not None and t_drop is None and t >= t_lift_done + HOLD_S:
            break
        if t > 26.0:
            break

    v_post = None
    if t_last_c is not None:
        # last logged ball vel after contact window
        after = [r for r in rows if r["t"] >= t_last_c]
        if after:
            v_post = np.asarray(after[0]["v_ball"], float)
    if v_pre is None and rows:
        v_pre = np.asarray(rows[0]["v_ball"], float)
    dp = None
    if v_pre is not None and v_post is not None:
        dp = BALL_M * (np.asarray(v_post) - np.asarray(v_pre))
        dt_c = max(float(t_last_c) - float(t_impact) + dt, dt)
        dp_grav = BALL_M * np.array(sim.model.opt.gravity, float) * dt_c
        resid = dp - J - dp_grav
    else:
        dp_grav = None
        resid = None
    end = rows[-1] if rows else pack_row(sim, TAU_SEC)
    hold_survived = bool(t_lift_done is not None and t_drop is None and (float(sim.data.time) - t_lift_done) >= HOLD_S - 1e-6)
    vb_h = None
    if t_impact is not None and Rh_imp is not None:
        # pre-impact ball vel in hand: nearest row before impact
        pre_rows = [r for r in rows if r["t"] < t_impact]
        vb_w = np.asarray(pre_rows[-1]["v_ball"], float) if pre_rows else np.asarray(v_pre, float)
        vb_h = Rh_imp.T @ vb_w
        J_h = Rh_imp.T @ J
    else:
        J_h = None
        vb_w = None

    ep = {
        "v_hit": float(v_hit),
        "ball_mass": BALL_M,
        "momentum_mv": float(BALL_M * v_hit),
        "t_lift0": t_lift0,
        "t_release": t_release,
        "t_impact": t_impact,
        "t_last_contact": t_last_c,
        "free_flight_s": None if (t_release is None or t_impact is None) else t_impact - t_release,
        "eq_active_at_release": 0 if released else None,
        "impact_ok": bool(impact_ok and impact_reject is None),
        "impact_reject": impact_reject,
        "first_contact": None
        if first_c is None
        else {
            "kind": first_c["kind"],
            "g1": first_c["g1"],
            "g2": first_c["g2"],
            "pos": first_c["pos"].tolist(),
            "normal": first_c["normal"].tolist(),
            "fn": first_c["fn"],
        },
        "v_ball_w_pre": None if vb_w is None else np.asarray(vb_w, float).tolist(),
        "v_ball_h_pre": None if vb_h is None else np.asarray(vb_h, float).tolist(),
        "J_world": J.tolist(),
        "J_hand": None if J_h is None else np.asarray(J_h, float).tolist(),
        "J_mag": float(np.linalg.norm(J)),
        "dp_ball": None if dp is None else dp.tolist(),
        "dp_grav": None if dp_grav is None else np.asarray(dp_grav, float).tolist(),
        "dp_minus_J_minus_mgdt": None if resid is None else resid.tolist(),
        "impulse_consistency": None if resid is None else float(np.linalg.norm(resid)),
        "t_drop": t_drop,
        "t_unilateral": t_ul,
        "t_both": t_both,
        "t_lift_done": t_lift_done,
        "drop": t_drop is not None,
        "hold_survived": hold_survived,
        "max_ex_mm": float(max_ex),
        "unilateral": bool(unilateral),
        "end_nL": int(end["nL"]),
        "end_nR": int(end["nR"]),
        "end_obj_z": float(end["obj_z"]),
        "launch": None
        if launch is None
        else {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in launch.items()},
        "events": events,
        "live": live_opt(sim),
        "rows": rows,
        "snaps": {"post_impact": snap_post, "unilateral": snap_ul, "loss": snap_loss},
    }
    ep["class"] = classify(ep)
    return ep


def pres_show(pres, sim, label: str) -> None:
    viewer = pres.get("viewer")
    if viewer is None and not pres.get("record"):
        return
    rows = [
        (label, ""),
        ("phase", str(pres.get("phase", ""))),
        ("t", f"{float(sim.data.time):.2f} s"),
    ]
    if viewer is not None:
        pos = mujoco.mjtGridPos.mjGRID_TOPLEFT
        for k, v in rows:
            viewer.add_overlay(pos, k, v)
        if hasattr(viewer, "sync"):
            viewer.sync()
        spd = float(pres.get("speed", 0.55))
        dt = float(sim.model.opt.timestep)
        now = time.perf_counter()
        tgt = pres.get("_wall")
        if tgt is None:
            pres["_wall"] = now + dt / spd
        else:
            sl = tgt - now
            if sl > 0:
                time.sleep(min(sl, 0.05))
            pres["_wall"] = max(tgt, time.perf_counter()) + dt / spd
    if pres.get("record") and pres.get("renderer") is not None:
        interval = pres.get("speed", 0.55) / VIDEO_FPS
        t = float(sim.data.time)
        nxt = pres.get("_rt")
        if nxt is None or t + 1e-12 >= nxt:
            pres["renderer"].update_scene(sim.data, camera=pres["cam"])
            img = pres["renderer"].render()
            try:
                from PIL import Image, ImageDraw

                im = Image.fromarray(img)
                dr = ImageDraw.Draw(im, "RGBA")
                dr.rectangle((6, 6, 280, 70), fill=(0, 0, 0, 140))
                y = 10
                for k, v in rows:
                    dr.text((12, y), k if not v else f"{k}: {v}", fill=(255, 255, 255, 255))
                    y += 16
                img = np.asarray(im)
            except Exception:
                pass
            pres.setdefault("frames", []).append(img)
            pres["_rt"] = (t if nxt is None else nxt) + interval


def save_traj(path: Path, ep: dict) -> None:
    rows = ep["rows"]
    if not rows:
        return
    np.savez_compressed(
        path,
        t=np.array([r["t"] for r in rows]),
        fsm=np.array([r["fsm"] for r in rows]),
        p_ball=np.stack([r["p_ball"] for r in rows]),
        v_ball=np.stack([r["v_ball"] for r in rows]),
        w_ball=np.stack([r["w_ball"] for r in rows]),
        p_obj=np.stack([r["p_obj"] for r in rows]),
        v_obj=np.stack([r["v_obj"] for r in rows]),
        w_obj=np.stack([r["w_obj"] for r in rows]),
        p_hand=np.stack([r["p_hand"] for r in rows]),
        v_hand=np.stack([r["v_hand"] for r in rows]),
        w_hand=np.stack([r["w_hand"] for r in rows]),
        rh=np.stack([r["rh"] for r in rows]),
        v_rel_h=np.stack([r["v_rel_h"] for r in rows]),
        w_rel_h=np.stack([r["w_rel_h"] for r in rows]),
        cyl_h=np.stack([r["cyl_h"] for r in rows]),
        nL=np.array([r["nL"] for r in rows]),
        nR=np.array([r["nR"] for r in rows]),
        Fn_L=np.array([r["Fn_L"] for r in rows]),
        Fn_R=np.array([r["Fn_R"] for r in rows]),
        ball_cyl=np.array([r["ball_cyl"] for r in rows]),
        F_ball_w=np.stack([r["F_ball_w"] for r in rows]),
        p_des=np.stack([r["p_des"] for r in rows]),
        v_cmd=np.stack([r["v_cmd"] for r in rows]),
        tau=np.array([r["tau"] for r in rows]),
        obj_z=np.array([r["obj_z"] for r in rows]),
        eq_active=np.array([r["eq_active"] for r in rows]),
        t_impact=np.array([ep["t_impact"] if ep["t_impact"] is not None else np.nan]),
        v_hit=np.array([ep["v_hit"]]),
    )


def pub_ep(ep: dict) -> dict:
    return {k: v for k, v in ep.items() if k not in ("rows", "snaps", "launch")}


def pick_reps(sweep: list) -> dict:
    order = ["STABLE", "DISTURBED_CAPTURED", "DELAYED_FAILURE", "IMMEDIATE_LOSS"]
    keys = ["I0", "I1", "I2", "I3"]
    out = {}
    for k, cls in zip(keys, order):
        cands = [e for e in sweep if e["class"] == cls and e["impact_ok"]]
        if not cands:
            out[k] = None
            continue
        out[k] = min(cands, key=lambda e: e["v_hit"])
        if k == "I3":
            out[k] = max(cands, key=lambda e: e["v_hit"])
    return out


def figures(sweep: list) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    vs = [e["v_hit"] for e in sweep]
    cls = [e["class"] for e in sweep]
    mv = [e["momentum_mv"] for e in sweep]
    jm = [e["J_mag"] for e in sweep]
    td = [None if e["t_drop"] is None or e["t_impact"] is None else e["t_drop"] - e["t_impact"] for e in sweep]
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 3.2))
    ax.plot(vs, mv, "o-", label="m v_hit")
    ax.plot(vs, jm, "s-", label="|J|")
    ax.set_xlabel("requested along-track speed (m/s)")
    ax.set_ylabel("N·s")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_momentum_impulse.png", dpi=120)
    plt.close(fig)
    fig, ax = plt.subplots(1, 1, figsize=(8.0, 3.0))
    colors = {
        "STABLE": "C2",
        "DISTURBED_CAPTURED": "C0",
        "DELAYED_FAILURE": "C1",
        "IMMEDIATE_LOSS": "C3",
        "INVALID": "0.5",
    }
    ax.scatter(vs, [0] * len(vs), c=[colors.get(c, "k") for c in cls], s=80)
    for v, c in zip(vs, cls):
        ax.text(v, 0.04, c, rotation=40, fontsize=8, ha="left")
    ax.set_yticks([])
    ax.set_xlabel("v_hit (m/s)")
    ax.set_title("coarse severity labels")
    fig.tight_layout()
    fig.savefig(FIG / "fig_regimes.png", dpi=120)
    plt.close(fig)
    # delayed-failure rh if any
    for e in sweep:
        if e["class"] != "DELAYED_FAILURE":
            continue
        t = np.array([r["t"] for r in e["rows"]])
        rh = np.stack([r["rh"] for r in e["rows"]])
        fig, ax = plt.subplots(2, 1, figsize=(7.5, 5.0), sharex=True)
        ax[0].plot(t, 1e3 * rh[:, 0], label="r_h.x")
        ax[0].axvline(e["t_impact"], color="k", ls="--", lw=0.8)
        if e["t_drop"]:
            ax[0].axvline(e["t_drop"], color="r", ls=":", lw=0.8)
        ax[0].set_ylabel("mm")
        ax[0].legend()
        nL = [r["nL"] for r in e["rows"]]
        nR = [r["nR"] for r in e["rows"]]
        ax[1].plot(t, nL, label="nL")
        ax[1].plot(t, nR, label="nR")
        ax[1].set_xlabel("t (s)")
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(FIG / f"fig_delayed_v{e['v_hit']:.2f}.png", dpi=120)
        plt.close(fig)
        break


def save_snaps(rep: dict) -> None:
    SNAP.mkdir(parents=True, exist_ok=True)
    for key, ep in rep.items():
        if ep is None:
            continue
        snaps = ep.get("snaps") or {}
        for name, snap in snaps.items():
            if snap is None:
                continue
            p = SNAP / f"{key}_{name}.pkl"
            with p.open("wb") as f:
                pickle.dump(snap, f)
        dump_json(SNAP / f"{key}_meta.json", pub_ep(ep))


def encode_mp4(frame_dir: Path, dest: Path) -> str | None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    pattern = str(frame_dir / "frame_%06d.png")
    cmd_tail = ["-y", "-framerate", str(VIDEO_FPS), "-i", pattern, "-c:v", "libx264", "-pix_fmt", "yuv420p", str(dest)]
    cands = []
    try:
        import imageio_ffmpeg

        cands.append(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass
    cands.append("ffmpeg")
    import subprocess

    for exe in cands:
        try:
            subprocess.run([exe, *cmd_tail], check=True, capture_output=True)
            if dest.is_file() and dest.stat().st_size > 1000:
                return str(dest)
        except Exception:
            continue
    return None


def write_frames(frames, dest: Path) -> str | None:
    tmp = dest.parent / f"_frames_{dest.stem}"
    if tmp.exists():
        for p in tmp.glob("*.png"):
            p.unlink()
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio

        for i, fr in enumerate(frames):
            imageio.imwrite(str(tmp / f"frame_{i:06d}.png"), fr)
    except Exception:
        import matplotlib.image as mpimg

        for i, fr in enumerate(frames):
            mpimg.imsave(str(tmp / f"frame_{i:06d}.png"), fr)
    return encode_mp4(tmp, dest)


def write_report(meta: dict) -> None:
    p = OUT / "BALLISTIC_IMPACT_NOSLIP1.md"
    a = []
    ap = a.append
    ap("# Ballistic impact family under noslip=1")
    ap("")
    ap("ZERO only. No recovery, RULE, SAC, recatch, or heuristic retune.")
    ap("")
    ap("## USER VISUAL OBSERVATION: pending")
    ap("")
    ap("Until the representative I0–I3 animations are watched, do not treat")
    ap("the severity labels as visually confirmed.")
    ap("")
    ap("## 1. Diagnostic scene construction")
    ap("")
    ap("New diagnostic scene `assets/scene_ballistic_impact.xml` (not the official")
    ap("nominal scene). Cylinder `contype/conaffinity=3`, ball `=2` so the ball")
    ap("collides with the cylinder only (not fingers, arm, table, floor).")
    ap("Historical `scene_impact_diag.xml` / 0.5 kg claims are quarantined in")
    ap("`OLD_IMPACT_QUARANTINE.md`.")
    ap("")
    ap("## 2. Actual ball mass / inertia audit")
    ap("")
    ap("Masses are declared in XML before `MjModel` construction. No `set_ball_mass`.")
    ap("")
    ap("```json")
    ap(json.dumps(meta.get("mass_audit"), indent=2, default=str))
    ap("```")
    ap("")
    ap("Free-flight drop:")
    ap("")
    ap("```json")
    ap(json.dumps(meta.get("drop_audit"), indent=2, default=str))
    ap("```")
    ap("")
    ap("## 3. Free-flight / release audit")
    ap("")
    ap("Weld holds the ball at a visible pose until the lift trigger. At trigger:")
    ap("weld off (`eq_active=0`), one write of free-flight `(p,v)`, then no ball")
    ap("qpos/qvel overwrite through collision.")
    ap("")
    ap("## 4. Nominal-task impact timing")
    ap("")
    ap("Trigger (not tuned for recovery): `phase==lift` AND captured AND")
    ap(f"`obj_z>=Z_AIR ({Z_AIR})` AND hand rise since lift start `>= {HAND_RISE_M} m`.")
    ap("Impact occurs after a free-flight interval while LIFT is still integrating.")
    ap("")
    ap("## 5. Impact frame / direction")
    ap("")
    ap("Launch along **+hand-x** (`R_h[:,0]`), gravity-compensated so along-track")
    ap("speed at the intended hit ≈ `v_hit`. Report `v_ball,h = R_h^T v_ball,world`.")
    ap("")
    ap("## 6. Ball-cylinder contact audit")
    ap("")
    ap("Runs are rejected if the first ball contact is not the cylinder.")
    ap("")
    ap("## 7. Impulse and momentum cross-check")
    ap("")
    ap("`J = Σ F_ball dt` over ball-cylinder contacts using `mj_contactForce`")
    ap("(force-on-ball after geom1/geom2 sign). Cross-check `Δp_ball = m(v_after-v_before)`.")
    ap("")
    ap("## 8. Coarse severity sweep")
    ap("")
    ap("One mass `m=0.05 kg`. Sweep `v_hit` only.")
    ap("")
    ap("```json")
    ap(json.dumps(meta.get("sweep_table"), indent=2, default=str))
    ap("```")
    ap("")
    ap("## 9. Regime boundaries")
    ap("")
    ap("```json")
    ap(json.dumps(meta.get("boundaries"), indent=2, default=str))
    ap("```")
    ap("")
    ap("## 10. Representative I0–I3")
    ap("")
    ap("```json")
    ap(json.dumps(meta.get("reps"), indent=2, default=str))
    ap("```")
    ap("")
    ap("## 11. Delayed-failure trajectories")
    ap("")
    ap("See `figures/fig_delayed_*.png` and `raw/traj_v*.npz`.")
    ap("")
    ap("## 12. Repeatability")
    ap("")
    ap("```json")
    ap(json.dumps(meta.get("repeat"), indent=2, default=str))
    ap("```")
    ap("")
    ap("Deterministic identical-IC replays are not statistical robustness.")
    ap("")
    ap("## 13. Viewer / video commands")
    ap("")
    ap("```text")
    ap("python training/demo_ballistic_impact.py")
    ap("python training/demo_ballistic_impact.py --severity mild")
    ap("python training/demo_ballistic_impact.py --severity disturbed")
    ap("python training/demo_ballistic_impact.py --severity delayed_failure")
    ap("python training/demo_ballistic_impact.py --severity loss")
    ap("python training/demo_ballistic_impact.py --render-video")
    ap("```")
    ap("")
    ap("## 14. USER VISUAL OBSERVATION: pending")
    ap("")
    ap("## 15. Unresolved issues")
    ap("")
    ap(meta.get("unresolved", "-"))
    p.write_text("\n".join(a), encoding="utf-8")


def run_sweep() -> dict:
    RAW.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    VID.mkdir(parents=True, exist_ok=True)
    sim, cfg = make_sim()
    gains = gains_from_cfg(cfg)
    mass = audit_mass(sim)
    drop = free_flight_drop_audit(sim)
    dump_json(RAW / "mass_audit.json", mass)
    dump_json(RAW / "drop_audit.json", drop)
    if not drop.get("ok"):
        raise RuntimeError(f"free-flight drop audit failed: {drop}")

    sweep = []
    for v in SWEEP_V:
        sim, cfg = make_sim()
        gains = gains_from_cfg(cfg)
        ep = run_episode(sim, gains, v)
        save_traj(RAW / f"traj_v{v:.2f}.npz", ep)
        dump_json(RAW / f"ep_v{v:.2f}.json", pub_ep(ep))
        sweep.append(ep)
        print(f"v={v:.2f} class={ep['class']} impact_ok={ep['impact_ok']} J={ep['J_mag']:.4f} drop={ep['drop']} t_imp={ep['t_impact']}", flush=True)

    # refine one midpoint if a class is missing between neighbors
    classes = [e["class"] for e in sweep]
    extra = []
    for a, b in zip(sweep[:-1], sweep[1:]):
        if a["class"] != b["class"] and a["impact_ok"] and b["impact_ok"]:
            mid = 0.5 * (a["v_hit"] + b["v_hit"])
            if any(abs(mid - e["v_hit"]) < 0.05 for e in sweep + extra):
                continue
            sim, cfg = make_sim()
            gains = gains_from_cfg(cfg)
            ep = run_episode(sim, gains, mid)
            save_traj(RAW / f"traj_v{mid:.2f}.npz", ep)
            dump_json(RAW / f"ep_v{mid:.2f}.json", pub_ep(ep))
            extra.append(ep)
            print(f"refine v={mid:.2f} class={ep['class']}", flush=True)
    sweep.extend(extra)
    sweep.sort(key=lambda e: e["v_hit"])

    dist_vs = [e for e in sweep if e["class"] == "DISTURBED_CAPTURED" and e["impact_ok"]]
    loss_vs = [e for e in sweep if e["class"] == "IMMEDIATE_LOSS" and e["impact_ok"]]
    if dist_vs and loss_vs and not any(e["class"] == "DELAYED_FAILURE" for e in sweep):
        lo = max(e["v_hit"] for e in dist_vs)
        hi = min(e["v_hit"] for e in loss_vs)
        for v in np.linspace(lo, hi, 7)[1:-1]:
            v = float(np.round(v, 3))
            if any(abs(v - e["v_hit"]) < 0.008 for e in sweep):
                continue
            sim, cfg = make_sim()
            gains = gains_from_cfg(cfg)
            ep = run_episode(sim, gains, v)
            save_traj(RAW / f"traj_v{v:.3f}.npz", ep)
            dump_json(RAW / f"ep_v{v:.3f}.json", pub_ep(ep))
            sweep.append(ep)
            print(f"boundary v={v:.3f} class={ep['class']}", flush=True)
        sweep.sort(key=lambda e: e["v_hit"])

    table = []
    for e in sweep:
        cons = e.get("impulse_consistency")
        table.append(
            {
                "v_hit": e["v_hit"],
                "m_v": e["momentum_mv"],
                "class": e["class"],
                "t_release": e["t_release"],
                "t_impact": e["t_impact"],
                "free_flight_s": e["free_flight_s"],
                "v_ball_h_pre": e["v_ball_h_pre"],
                "J_mag": e["J_mag"],
                "impulse_consistency": cons,
                "max_ex_mm": e["max_ex_mm"],
                "unilateral": e["unilateral"],
                "t_drop": None if e["t_drop"] is None or e["t_impact"] is None else e["t_drop"] - e["t_impact"],
                "hold_survived": e["hold_survived"],
                "reject": e["impact_reject"],
            }
        )
    cons_vals = [e["impulse_consistency"] for e in sweep if e["impulse_consistency"] is not None]
    if cons_vals and max(cons_vals) > 0.05:
        raise RuntimeError(f"impulse vs Δp_ball-mgΔt grossly inconsistent: {max(cons_vals)}")

    reps_full = pick_reps(sweep)
    save_snaps(reps_full)
    reps = {k: None if v is None else {"v_hit": v["v_hit"], "class": v["class"], **{kk: pub_ep(v)[kk] for kk in ("t_impact", "J_mag", "max_ex_mm", "t_drop", "hold_survived", "v_ball_h_pre")}} for k, v in reps_full.items()}

    # repeatability on first available I2 else I1
    target = reps_full.get("I2") or reps_full.get("I1") or sweep[0]
    sim, cfg = make_sim()
    gains = gains_from_cfg(cfg)
    r2 = run_episode(sim, gains, target["v_hit"])
    dpos = None
    if target["rows"] and r2["rows"]:
        n = min(len(target["rows"]), len(r2["rows"]))
        dpos = float(np.max(np.abs(np.stack([x["p_obj"] for x in target["rows"][:n]]) - np.stack([x["p_obj"] for x in r2["rows"][:n]]))))
    repeat = {
        "v_hit": target["v_hit"],
        "class_a": target["class"],
        "class_b": r2["class"],
        "max_obj_pos_diff": dpos,
        "note": "identical IC, deterministic replay; not a statistical N-trial claim",
    }

    figures(sweep)
    bounds = {}
    prev = None
    for e in sweep:
        if e["class"] != prev and e["impact_ok"]:
            bounds[e["class"]] = {"first_v": e["v_hit"], "m_v": e["momentum_mv"]}
            prev = e["class"]

    unresolved = []
    for k, lab in [("I0", "STABLE"), ("I1", "DISTURBED_CAPTURED"), ("I2", "DELAYED_FAILURE"), ("I3", "IMMEDIATE_LOSS")]:
        if reps[k] is None:
            unresolved.append(f"no representative {k}/{lab} in the coarse sweep")
    if not unresolved:
        unresolved = ["none"]

    meta = {
        "mass_audit": mass,
        "drop_audit": drop,
        "sweep_table": table,
        "boundaries": bounds,
        "reps": reps,
        "repeat": repeat,
        "unresolved": "; ".join(unresolved),
        "trigger": {
            "definition": f"lift AND captured AND obj_z>=Z_AIR AND hand_rise>={HAND_RISE_M}",
            "Z_AIR": Z_AIR,
            "HAND_RISE_M": HAND_RISE_M,
            "D_LAUNCH": D_LAUNCH,
        },
    }
    dump_json(RAW / "summary.json", meta)
    dump_json(RAW / "rep_speeds.json", {k: (None if v is None else v["v_hit"]) for k, v in reps.items()})
    write_report(meta)
    return meta, reps_full, cfg


def interactive(severity: str, reps_speeds: dict | None = None) -> None:
    import mujoco.viewer

    key = {"mild": "I0", "disturbed": "I1", "delayed_failure": "I2", "loss": "I3"}[severity]
    if reps_speeds is None:
        p = RAW / "rep_speeds.json"
        if not p.is_file():
            raise SystemExit("run the sweep first (python training/demo_ballistic_impact.py)")
        reps_speeds = json.loads(p.read_text(encoding="utf-8"))
    v = reps_speeds.get(key)
    if v is None:
        raise SystemExit(f"no representative {key} / {severity} from the sweep")
    sim, cfg = make_sim()
    gains = gains_from_cfg(cfg)
    kh = {}

    def _key(kc):
        fn = kh.get("fn")
        if fn:
            fn(kc)

    with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=_key) as vwr:
        apply_camera_preset(vwr, CAM)
        kh["fn"] = make_reset_cam_callback({"viewer": vwr, "_cam_preset": CAM, "_cam_inited": True})
        t_end = time.perf_counter() + PRE_VIEW_PAUSE_S
        while time.perf_counter() < t_end:
            vwr.sync()
            time.sleep(0.03)
        run_episode(sim, gains, float(v), viz={"viewer": vwr, "speed": 0.45, "phase": "APPROACH"})
        t_end = time.perf_counter() + 2.0
        while time.perf_counter() < t_end:
            vwr.sync()
            time.sleep(0.03)


def render_videos(reps_full: dict) -> None:
    VID.mkdir(parents=True, exist_ok=True)
    names = {"I0": "mild", "I1": "disturbed", "I2": "delayed_failure", "I3": "loss"}
    sim0, cfg = make_sim()
    gains = gains_from_cfg(cfg)
    for key, name in names.items():
        ep = reps_full.get(key)
        if ep is None:
            continue
        sim, cfg = make_sim()
        gains = gains_from_cfg(cfg)
        renderer = mujoco.Renderer(sim.model, RENDER_H, RENDER_W)
        cam = mjv_camera_from_preset(CAM)
        pres = {"record": True, "renderer": renderer, "cam": cam, "speed": 0.55, "phase": "APPROACH", "frames": []}
        run_episode(sim, gains, ep["v_hit"], viz=pres, record=True)
        dest = VID / f"{name}.mp4"
        write_frames(pres["frames"], dest)
        renderer.close()
        print("wrote", dest, flush=True)


def main():
    p = argparse.ArgumentParser(description="Audited ballistic impact family (ZERO only).")
    p.add_argument("--severity", default="", help="mild | disturbed | delayed_failure | loss")
    p.add_argument("--case", default="", help="interactive disturbance-map case (no MP4)")
    p.add_argument("--render-video", action="store_true")
    p.add_argument("--sweep-only", action="store_true")
    args = p.parse_args()
    if args.case:
        from training.map_ballistic_disturbance import interactive as map_interactive, parse_case

        site, v = parse_case(args.case)
        map_interactive(site, v, args.case)
        return
    if args.severity:
        interactive(args.severity)
        return
    if args.sweep_only:
        meta, reps_full, cfg = run_sweep()
        print(json.dumps({"reps": meta["reps"], "unresolved": meta["unresolved"], "repeat": meta["repeat"]}, indent=2, default=str))
        return
    print(
        "Use python training/map_ballistic_disturbance.py for the disturbance map,\n"
        "or python training/demo_ballistic_impact.py --case <name> for the viewer.\n"
        "Refusing to auto-render MP4."
    )


if __name__ == "__main__":
    main()
