"""Nominal ballistic scene. noslip=1. Cylinder 0.20 kg, diagnostic ball 0.05 kg."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import mujoco
import numpy as np

from envs.simulation.yaml_config import load_yaml, merge_sim_config
from envs.simulation.object_dynamics import set_finger_object_sliding_mu
from envs.simulation.grasp_sim import GraspSim, apply_solver_from_cfg, reset_home
from envs.simulation.mujoco_ids import resolve_ids
from envs.simulation.panda_xml import write_panda_torque
from envs.simulation.impact_ball import apply_ball_free_state
from envs.simulation.ballistic_scene import CYL_M, write_ballistic_scene

ROOT = Path(__file__).resolve().parents[1]
PAIR_MU = 1.0
HAND_RISE_M = 0.05
TAU_SEC = -18.0
VIDEO_FPS = 25
HOLD_S = 12.0
AFTER_DROP_S = 2.0
LOG_DT = 0.010


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(obj), indent=2), encoding="utf-8")


def encode_mp4(frame_dir: Path, dest: Path) -> str | None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    pattern = str(frame_dir / "frame_%06d.png")
    cmd_tail = ["-y", "-framerate", str(VIDEO_FPS), "-i", pattern, "-c:v", "libx264", "-pix_fmt", "yuv420p", str(dest)]
    candidates = []
    try:
        import imageio_ffmpeg

        candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass
    candidates.append("ffmpeg")
    for exe in candidates:
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
        for path in tmp.glob("*.png"):
            path.unlink()
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio

        for index, frame in enumerate(frames):
            imageio.imwrite(str(tmp / f"frame_{index:06d}.png"), frame)
    except Exception:
        import matplotlib.image as mpimg

        for index, frame in enumerate(frames):
            mpimg.imsave(str(tmp / f"frame_{index:06d}.png"), frame)
    return encode_mp4(tmp, dest)


def live_opt(sim) -> dict:
    model = sim.model
    return {
        "mujoco": mujoco.__version__,
        "noslip_iterations": int(model.opt.noslip_iterations),
        "noslip_tolerance": float(model.opt.noslip_tolerance),
        "timestep": float(model.opt.timestep),
        "gravity": np.array(model.opt.gravity, float).tolist(),
    }


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
        from controllers.recovery.residual_limits import ResidualLimiter
        from envs.simulation.grasp_quality import DeteriorationMeter

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

    def park_ball(self) -> None:
        apply_ball_free_state(self, np.array([1.00, 0.80, 1.20]), np.zeros(3))

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
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "simulation" / "nominal.yaml"))
    sim = BallisticSim(cfg)
    sim.reset()
    assert_noslip(sim)
    return sim, cfg


def z_tgt_of(sim) -> float:
    return float(sim.fsm.grasp_z) + float(sim.cfg["fsm"]["lift_offset_z"])


def ball_force_on_ball(sim) -> tuple[np.ndarray, list]:
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
        else:
            f_ball = -f_on_geom2
        F += f_ball
        other = g2 if g1 == bg else g1
        kind = "ball-cylinder" if other == og else "ball-other"
        rows.append({"kind": kind, "pos": np.array(c.pos, float).copy(), "fn": fn})
    return F, rows


def pack_row(sim, tau: float) -> dict:
    from envs.simulation.grasp_quality import body_twist
    from envs.outcomes.physical_recovery import physical_pack
    from envs.contacts.frames import extract_contacts

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


def world():
    """Ballistic scene plus the nominal grasp and finger-pad records."""
    from controllers.cartesian.jacobian import gains_from_cfg
    from envs.simulation.object_dynamics import finger_pad_geom_ids
    from training.recovery_execution import capture_nominal, capture_pads

    sim, cfg = make_sim()
    gains = gains_from_cfg(cfg)
    nominal = capture_nominal(sim)
    pad_ids, pad_mu = capture_pads(sim)
    pads_l, pads_r = finger_pad_geom_ids(sim.model, sim.ids)
    return sim, gains, nominal, pad_ids, pad_mu, pads_l, pads_r
