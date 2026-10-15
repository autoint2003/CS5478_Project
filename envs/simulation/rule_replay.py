"""Frozen RULE baseline replay.

The hashed controller is controllers/recovery/rule_based.py. Scene constants,
ball parking, and obs_from_sim live in neighboring modules.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import mujoco
import numpy as np
import yaml

from controllers.nominal import GraspFSM, grasp_orientation
from controllers.recovery.residual_limits import ResidualLimiter
from controllers.recovery.rule_based import RuleBasedRecovery, RuleParams
from envs.simulation.grasp_quality import DeteriorationMeter
from envs.simulation.object_dynamics import set_object_dynamics
from envs.simulation.grasp_sim import GraspSim, apply_solver_from_cfg, reset_home
from envs.simulation.mujoco_ids import resolve_ids
from envs.simulation.panda_xml import write_panda_torque
from envs.simulation.impact_diag_scene import write_impact_scene
from controllers.cartesian.velocity_step import freeze, tick_vw
from envs.observations.privileged_obs import obs_from_sim
from envs.simulation.scene_state import BALL_R, ball_is_parked, park_impact_ball

ROOT = Path(__file__).resolve().parents[2]
FROZEN_YAML = ROOT / "config" / "controllers" / "rule_based_recovery.yaml"
CTRL_PY = ROOT / "controllers" / "recovery" / "rule_based.py"
OFFSET_NPZ = ROOT / "results" / "eval_sets" / "airborne_offset_eval.npz"
IMPACT_NPZ = ROOT / "results" / "eval_sets" / "impact_severity_four.npz"

MASS, MU = 0.20, 1.00
G_HOLD = -18.0
KP, KD = 180.0, 28.0
EXPECTED_YAML = "6a0b8103e5b9d8c4a4a4e6634aca43adb0b37bd39242d39269f27785a4d574b8"
EXPECTED_PY = "778bd97db7ca19f64192d3d833abd05df77129a223a9683076e1a993701e858c"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def verify_frozen_hashes() -> dict:
    y = sha256_file(FROZEN_YAML)
    p = sha256_file(CTRL_PY)
    if y != EXPECTED_YAML or p != EXPECTED_PY:
        raise RuntimeError(f"frozen hash mismatch yaml={y} py={p}")
    return {"yaml": y, "controller_py": p}


def params_from_frozen() -> tuple[RuleParams, dict]:
    y = yaml.safe_load(FROZEN_YAML.read_text(encoding="utf-8"))
    if not y.get("frozen"):
        raise RuntimeError("config/controllers/rule_based_recovery.yaml is not frozen")
    p = RuleParams()
    for k, v in y.items():
        if k in ("frozen", "e_tol_note", "t_brake_justification", "kp_pos", "kd_pos", "kp_ori", "kd_ori"):
            continue
        if hasattr(p, k):
            setattr(p, k, v)
    p.e_tol = float(y["e_tol"])
    p.t_brake = float(y["t_brake"])
    p.tau_secure = float(y["tau_secure"])
    p.tau_slip = float(y["tau_slip"])
    p.tau_open = float(y["tau_open"])
    return p, y






def pack_replay(sim) -> dict:
    nact = int(getattr(sim.model, "na", 0) or 0)
    act = np.array(sim.data.act, float).copy() if nact else np.zeros(0)
    return {
        "qpos": np.array(sim.data.qpos, float).copy(),
        "qvel": np.array(sim.data.qvel, float).copy(),
        "act": act,
        "ctrl": np.array(sim.data.ctrl, float).copy(),
        "time": float(sim.data.time),
        "p_des": np.asarray(sim.fsm.p_des, float).copy(),
        "r_des": np.asarray(sim.fsm.r_des, float).copy(),
        "v_cmd": np.asarray(sim.fsm.v_cmd, float).copy(),
        "w_cmd": np.asarray(sim.fsm.w_cmd, float).copy(),
        "phase": str(sim.fsm.phase),
        "captured": bool(sim.captured),
    }


def restore_replay(sim, ic: dict) -> None:
    sim.data.qpos[:] = ic["qpos"]
    sim.data.qvel[:] = ic["qvel"]
    if len(ic.get("act", [])) :
        sim.data.act[:] = ic["act"]
    sim.data.ctrl[:] = ic["ctrl"]
    sim.data.time = float(ic["time"])
    sim.fsm.p_des = np.asarray(ic["p_des"], float).copy()
    sim.fsm.r_des = np.asarray(ic["r_des"], float).copy()
    sim.fsm.v_cmd = np.asarray(ic.get("v_cmd", np.zeros(3)), float).copy()
    sim.fsm.w_cmd = np.asarray(ic.get("w_cmd", np.zeros(3)), float).copy()
    sim.fsm.phase = ic.get("phase", "lift")
    sim.captured = bool(ic.get("captured", True))
    mujoco.mj_forward(sim.model, sim.data)




def prepare_replay(sim, ic: dict) -> None:
    """Restore snapshot, then park the impact ball. Never restore after park."""
    restore_replay(sim, ic)
    park_impact_ball(sim)






















def object_free_adr(sim) -> tuple[int, int]:
    j = int(sim.ids.object_jnt)
    return int(sim.model.jnt_qposadr[j]), int(sim.model.jnt_dofadr[j])


class ImpactSim(GraspSim):
    def __init__(self, cfg: dict):
        scene = write_impact_scene()
        write_panda_torque()
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        apply_solver_from_cfg(self.model, cfg)
        self.data = mujoco.MjData(self.model)
        self.ids = resolve_ids(self.model)
        self.fsm = GraspFSM(cfg)
        self.meter = DeteriorationMeter(cfg)
        self.limiter = ResidualLimiter(cfg)
        self.n_sub = int(cfg.get("n_substeps", 10))
        self.dt_policy = float(self.model.opt.timestep) * self.n_sub
        self.mass = MASS
        self.friction = MU
        self.captured = False
        self.ball_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "impact_ball")
        self.ball_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "impact_ball")
        self.ball_jnt = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_joint")
        self.ball_qadr = int(self.model.jnt_qposadr[self.ball_jnt])
        self.ball_dadr = int(self.model.jnt_dofadr[self.ball_jnt])
        self.guide_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball_guide")
        self.guide_eq = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "ball_guide_weld")
        self.guide_mocap = int(self.model.body_mocapid[self.guide_body]) if self.guide_body >= 0 else -1

    def set_guide_weld(self, on: bool) -> None:
        if self.guide_eq < 0:
            return
        flag = int(bool(on))
        if hasattr(self.data, "eq_active"):
            self.data.eq_active[self.guide_eq] = flag
        if hasattr(self.model, "eq_active0"):
            self.model.eq_active0[self.guide_eq] = flag

    def set_guide_pos(self, p) -> None:
        if self.guide_mocap < 0:
            return
        self.data.mocap_pos[self.guide_mocap] = np.asarray(p, float).reshape(3)
        self.data.mocap_quat[self.guide_mocap] = np.array([1.0, 0.0, 0.0, 0.0])

    def set_ball_mass(self, m: float) -> None:
        m = float(max(m, 1e-6))
        i = 0.4 * m * BALL_R * BALL_R
        self.model.body_mass[self.ball_body] = m
        self.model.body_inertia[self.ball_body] = np.array([i, i, i])
        # body_mass edits do not rebuild qM; without this, gravity accel is m_xml/m_qM * g.
        mujoco.mj_setConst(self.model, self.data)

    def park_ball(self) -> None:
        q = self.ball_qadr
        d = self.ball_dadr
        self.data.qpos[q : q + 3] = np.array([1.00, 0.80, 1.20])
        self.data.qpos[q + 3 : q + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        self.data.qvel[d : d + 6] = 0.0

    def reset(self, mass: float | None = None, friction: float | None = None, grasp_offset=None) -> None:
        self.mass = float(MASS if mass is None else mass)
        self.friction = float(MU if friction is None else friction)
        reset_home(self.model, self.data)
        set_object_dynamics(self.model, self.ids, self.mass, self.friction, self.data)
        self.set_guide_weld(False)
        self.park_ball()
        mujoco.mj_forward(self.model, self.data)
        off = np.zeros(3) if grasp_offset is None else np.asarray(grasp_offset, dtype=float)
        self.fsm.reset(off)
        self.meter.reset()
        self.limiter.reset()
        self.captured = False


def rollout_from_replay(
    sim,
    ic: dict,
    gains,
    params: RuleParams,
    zero: bool,
    timeout: float = 2.0,
    *,
    restore: bool = True,
):
    """If restore: snapshot then park ball (impact), then recover. No later restore."""
    if restore:
        prepare_replay(sim, ic)
    if hasattr(sim, "park_ball") and not ball_is_parked(sim):
        raise RuntimeError("impact ball not parked at recovery start")
    ctrl = RuleBasedRecovery(params)
    ctrl.reset(grasp_orientation())
    dt = float(sim.model.opt.timestep)
    o0 = obs_from_sim(sim)
    rows = []
    t = 0.0
    n = int(round(timeout / dt))
    last_mode = "STABILIZE"
    for _ in range(n):
        o = obs_from_sim(sim)
        if zero:
            cmd = {
                "v_world": np.zeros(3),
                "tau": params.tau_secure,
                "r_des": np.asarray(ic["r_des"], float).copy(),
                "freeze_p": True,
                "mode": "ZERO",
                "e_dot": 0.0,
            }
            ctrl.s.mode = "ZERO"
        else:
            cmd = ctrl.step(o, dt)
        if (not zero) and cmd["mode"] != last_mode:
            last_mode = cmd["mode"]
            if cmd["freeze_p"]:
                freeze(sim)
        if cmd["r_des"] is not None:
            sim.fsm.r_des = np.asarray(cmd["r_des"], float).copy()
        tick_vw(sim, cmd["v_world"], np.zeros(3), cmd["tau"], gains)
        t += dt
        o2 = obs_from_sim(sim)
        rows.append({"t": t, "mode": cmd["mode"], "e_x": o2["e_x"], "nL": o2["nL"], "nR": o2["nR"],
                     "v_rel": o2["v_rel"], "obj_z": o2["obj_z"], "scene": o2["scene"]})
        if (not zero) and ctrl.s.done:
            break
        if zero and t >= timeout:
            break
    oF = obs_from_sim(sim)
    rec_ok = (
        abs(oF["e_x"]) <= params.e_tol
        and oF["nL"] > 0 and oF["nR"] > 0
        and oF["v_rel"] < params.v_rel_tol
        and oF["scene"] == 0
    )
    task_ok = rec_ok and oF["obj_z"] > 0.45
    retained = oF["nL"] > 0 and oF["nR"] > 0 and oF["obj_z"] > 0.45 and oF["scene"] == 0
    return rows, {
        "e0_mm": 1e3 * o0["e_x"],
        "e_min_mm": 1e3 * min((abs(r["e_x"]) for r in rows), default=abs(o0["e_x"])),
        "e_final_mm": 1e3 * abs(oF["e_x"]),
        "modes": list(ctrl.s.modes) if not zero else ["ZERO"],
        "path": ("slip_open_regrasp" if ctrl.s.used_open else "slip_only") if not zero else "zero",
        "recovery_success": int(rec_ok),
        "task_success": int(task_ok),
        "grasp_retention": int(retained),
        "drop_or_contact_loss": int(not retained),
        "nL": oF["nL"], "nR": oF["nR"],
        "t": t,
    }


def load_offset_ic(idx: int = 0) -> dict:
    d = np.load(OFFSET_NPZ, allow_pickle=True)
    ic = d["ics"][idx]
    if not isinstance(ic, dict):
        ic = dict(ic)
    return ic["replay"]


def load_impact_ic(v: float = 8.5) -> dict:
    d = np.load(IMPACT_NPZ, allow_pickle=True)
    p = f"v{v:.1f}_"
    return {
        "qpos": d[p + "qpos"],
        "qvel": d[p + "qvel"],
        "act": d[p + "act"],
        "ctrl": d[p + "ctrl"],
        "time": float(d[p + "time"]),
        "p_des": d[p + "p_des"],
        "r_des": d[p + "r_des"],
        "v_cmd": d[p + "v_cmd"],
        "w_cmd": d[p + "w_cmd"],
        "phase": str(d[p + "phase"]),
        "captured": bool(d[p + "captured"]),
    }


def make_offset_sim(cfg) -> GraspSim:
    sim = GraspSim(cfg)
    sim.reset(MASS, MU, np.zeros(3))
    disable_object_table(sim)
    return sim


def make_impact_sim(cfg) -> ImpactSim:
    sim = ImpactSim(cfg)
    sim.reset(MASS, MU, np.zeros(3))
    disable_object_table(sim)
    isolate_ball_object_only(sim)
    sim.set_ball_mass(0.50)
    sim.park_ball()
    mujoco.mj_forward(sim.model, sim.data)
    return sim
