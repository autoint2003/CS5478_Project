"""Interactive MuJoCo visualization of the existing impact → recovery sequence.

Replays ImpactSim physics and frozen RULE / ZERO. Does not change physics,
impact ball parameters, controller, reward, or the tactile estimator.
GT fields are overlay/evaluation only and are never passed to the controller
except through the existing frozen RULE obs_from_sim (unchanged).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.jacobian_controller import gains_from_cfg  # noqa: E402
from controllers.nominal import grasp_orientation  # noqa: E402
from controllers.residual import map_recovery4d  # noqa: E402
from controllers.rule_based_recovery import RuleBasedRecovery  # noqa: E402
from envs.config_util import load_yaml, merge_sim_config  # noqa: E402
from envs.observable_obs import ObservableObsState, observe_observable  # noqa: E402
from envs.observable_reward import read_tactile  # noqa: E402
from envs.physical_recovery import Z_AIR, observe_recovery4d, physical_pack  # noqa: E402
from training.impact_ball import ImpactBallDriver  # noqa: E402
from training.impact_visualization_utils import (  # noqa: E402
    LOG_DIR,
    VIDEO_DIR,
    EventTracker,
    HoldClocks,
    compare_integrity,
    draw_debug_geoms,
    enable_viewer_flags,
    format_overlay,
    overlay_dict,
    save_ex_plot,
    apply_viewer_camera,
    compute_camera_preset,
    mjv_camera_from_preset,
    snapshot_integrity,
    tick_vw_park,
    try_write_mp4,
)
from training.replay_core import (  # noqa: E402
    BALL_PARK_POS,
    BALL_R,
    G_HOLD,
    MASS,
    MU,
    ImpactSim,
    disable_object_table,
    freeze,
    isolate_ball_object_only,
    obs_from_sim,
    params_from_frozen,
    verify_frozen_hashes,
)

IMPACT_SPEEDS = (8.5, 9.0, 9.5, 10.0)


def make_live_impact_sim(cfg) -> ImpactSim:
    """Same pieces as make_impact_sim, table collisions left on until airborne.

    make_impact_sim() disables object-table immediately because eval ICs are
    already airborne. A live lift needs the table until capture.
    """
    sim = ImpactSim(cfg)
    sim.reset(MASS, MU, np.zeros(3))
    isolate_ball_object_only(sim)
    sim.set_ball_mass(0.50)
    sim.park_ball()
    mujoco.mj_forward(sim.model, sim.data)
    return sim


def _cmd_zero(params, r_des) -> dict:
    return {
        "v_world": np.zeros(3),
        "w_world": np.zeros(3),
        "w_hy": 0.0,
        "tau": float(params.tau_secure),
        "r_des": np.asarray(r_des, float).copy(),
        "freeze_p": True,
        "mode": "ZERO",
    }


def _try_load_sac(path: Path):
    from stable_baselines3 import SAC

    if not path.is_file():
        raise FileNotFoundError(f"SAC checkpoint not found: {path}")
    return SAC.load(str(path))


class ImpactRecoveryViz:
    def __init__(self, args):
        self.args = args
        verify_frozen_hashes()
        self.cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
        self.gains = gains_from_cfg(self.cfg)
        self.params, self.yaml = params_from_frozen()
        self.sim = make_live_impact_sim(self.cfg)
        self.dt = float(self.sim.model.opt.timestep)
        self.ball = ImpactBallDriver(
            self.sim,
            v_impact=float(args.impact_speed),
            d_launch=float(getattr(args, "ball_launch_distance", 0.30)),
            legacy=bool(getattr(args, "legacy_ball_launch", False)),
        )
        self.ctrl = RuleBasedRecovery(self.params)
        self.sac = None
        if args.controller == "sac":
            if not args.checkpoint:
                raise RuntimeError("--controller sac requires --checkpoint PATH (no fabricated policy)")
            self.sac = _try_load_sac(Path(args.checkpoint))
        self.phase = "grasp"
        self.ball_parked = True
        self.park_after_clear = 0
        self.t_launch = None
        self.t_first_contact = None
        self.t_last_contact = None
        self.t_park = None
        self.t_recovery = None
        self.launch_meta = {}
        self.modes: list[str] = []
        self.events = EventTracker()
        self.clocks = HoldClocks(dt=self.dt)
        self.hist = defaultdict(list)
        self.event_times = defaultdict(list)
        self.trail_h: list[np.ndarray] = []
        self.trail_o: list[np.ndarray] = []
        self.last_cmd = _cmd_zero(self.params, grasp_orientation())
        self.lift_extra = 0.0
        self.settle_logged = False
        self.overlay_last_print = -1.0
        self._obs_hist = None
        self.renderer = None
        self.frame_i = 0
        self.frame_dir = None
        self._cam_init = None
        self._cam_shot_done = False
        self._viewer = None
        if args.record:
            self.frame_dir = VIDEO_DIR / f"_frames_v{args.impact_speed}_{args.controller}"
            self.frame_dir.mkdir(parents=True, exist_ok=True)
            self.renderer = mujoco.Renderer(self.sim.model, 720, 480)

    def _tactile(self):
        return read_tactile(self.sim.model, self.sim.data, self.sim.ids)

    def _pack(self):
        return physical_pack(self.sim)

    def _record_hist(self, d, t, mode, first_c, parked):
        for k, v in d.items():
            if isinstance(v, (bool, int, float, np.floating, np.integer)):
                self.hist[k].append(float(v) if not isinstance(v, bool) else float(v))
        if first_c and "impact" not in self.event_times:
            self.event_times["impact"].append(t)
        if parked and "recovery_start" not in self.event_times and self.phase == "recovery":
            pass
        if self.phase == "recovery" and self.t_recovery is not None:
            if not self.event_times["recovery_start"]:
                self.event_times["recovery_start"].append(self.t_recovery)
        if d.get("success_GT") or d.get("success_obs"):
            if t not in self.event_times["success"]:
                self.event_times["success"].append(t)
        if d.get("failure_GT") or d.get("failure_obs"):
            if t not in self.event_times["failure"]:
                self.event_times["failure"].append(t)

    def _init_camera_once(self, viewer) -> None:
        preset = str(getattr(self.args, "camera", "impact"))
        d_launch = float(getattr(self.args, "ball_launch_distance", 0.30))
        self._cam_init = compute_camera_preset(self.sim, preset, d_launch)
        if viewer is not None:
            apply_viewer_camera(viewer, self._cam_init)

    def reset_camera_to_preset(self, viewer=None) -> None:
        apply_viewer_camera(viewer if viewer is not None else self._viewer, self._cam_init)

    def _maybe_framing_screenshot(self) -> None:
        if self._cam_shot_done or self._cam_init is None:
            return
        if self.t_launch is None or self.ball.state != "FLYING":
            return
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = LOG_DIR / "camera_impact_initial.png"
        mujoco.mj_forward(self.sim.model, self.sim.data)
        cam = mjv_camera_from_preset(self._cam_init)
        r = mujoco.Renderer(self.sim.model, 480, 640)
        r.update_scene(self.sim.data, camera=cam)
        _ = r.render()
        r.update_scene(self.sim.data, camera=cam)
        img = r.render()
        r.close()
        try:
            import imageio.v2 as imageio

            imageio.imwrite(path, img)
        except Exception:
            import matplotlib.image as mpimg

            mpimg.imsave(str(path), img)
        self._cam_shot_done = True
        self._cam_shot_path = str(path)
        p = self._cam_init
        po = np.array(self.sim.data.xpos[self.sim.ids.object_body], float)
        pb = np.array(self.sim.data.qpos[int(self.sim.ball_qadr) : int(self.sim.ball_qadr) + 3], float)
        print(
            f"[camera] preset={p['preset']} lookat={np.asarray(p['lookat'])} "
            f"az={p['azimuth']} el={p['elevation']} dist={p['distance']:.3f} "
            f"p_obj={po} p_ball={pb} shot={path}",
            flush=True,
        )

    def _maybe_frames(self, viewer):
        if self.renderer is None:
            return
        n_sub = max(1, int(round(0.02 / self.dt)))
        if self.frame_i % n_sub != 0:
            self.frame_i += 1
            return
        cam = getattr(viewer, "cam", None)
        if cam is not None:
            self.renderer.update_scene(self.sim.data, camera=cam)
        else:
            self.renderer.update_scene(self.sim.data)
        img = self.renderer.render()
        try:
            import imageio.v2 as imageio

            imageio.imwrite(self.frame_dir / f"frame_{self.frame_i // n_sub:06d}.png", img)
        except Exception:
            import matplotlib.image as mpimg

            mpimg.imsave(str(self.frame_dir / f"frame_{self.frame_i // n_sub:06d}.png"), img)
        self.frame_i += 1

    def _pace(self, t0: float) -> None:
        if self.args.speed <= 0:
            return
        target = self.dt / float(self.args.speed)
        leftover = target - (time.perf_counter() - t0)
        if leftover > 0:
            time.sleep(leftover)

    def _pause_wall(self, viewer, seconds: float) -> None:
        t_end = time.perf_counter() + seconds
        while time.perf_counter() < t_end:
            if viewer is not None and hasattr(viewer, "sync"):
                if hasattr(viewer, "is_running") and not viewer.is_running():
                    break
                viewer.sync()
            time.sleep(0.02)

    def step_once(self, viewer) -> bool:
        """One physics step. Returns False when the scenario is finished."""
        sim = self.sim
        t_wall = time.perf_counter()
        park_now = bool(self.ball_parked)
        reading, tac = self._tactile()
        o = self._pack()
        t = float(sim.data.time)
        cmd = self.last_cmd
        mode = str(cmd.get("mode", self.phase))

        if self.phase == "grasp":
            sim.physics_step(None, in_recovery=False)
            sim.maybe_capture_reference()
            z = float(sim.data.xpos[sim.ids.object_body][2])
            if sim.fsm.phase == "lift" and sim.captured and z >= Z_AIR:
                self.lift_extra += self.dt
                if self.lift_extra >= 0.25:
                    freeze(sim)
                    disable_object_table(sim)
                    isolate_ball_object_only(sim)
                    self.phase = "hold"
                    print(
                        f"[t={sim.data.time:.3f}] HOLD freeze + disable_object_table + re-isolate ball",
                        flush=True,
                    )
            elif sim.dropped() or sim.data.time > self.args.lift_timeout:
                print(f"[t={sim.data.time:.3f}] GRASP FAILED phase={sim.fsm.phase}", flush=True)
                return False
            park_now = True
            self.ball.keep_parked()
            cmd = {
                "v_world": np.asarray(sim.fsm.v_cmd, float),
                "w_world": np.zeros(3),
                "w_hy": 0.0,
                "tau": float(sim.data.ctrl[7]) if sim.data.ctrl.size > 7 else G_HOLD,
                "mode": f"NOMINAL_{sim.fsm.phase}",
            }
            self.last_cmd = cmd
            mode = cmd["mode"]

        elif self.phase == "hold":
            tick_vw_park(sim, np.zeros(3), np.zeros(3), G_HOLD, self.gains, park=False)
            self.ball.keep_parked()
            if not hasattr(self, "_hold_t0"):
                self._hold_t0 = float(sim.data.time)
            if float(sim.data.time) - self._hold_t0 >= 0.20:
                self.launch_meta = self.ball.launch()
                self.t_launch = self.ball.t_launch
                self.ball_parked = False
                self.phase = "ball_flight"
                print(
                    f"[t={self.t_launch:.3f}] BALL LAUNCH state={self.ball.state}\n"
                    f"    d_launch={self.launch_meta['d_launch']:.3f} m  T_hit~{self.launch_meta['T_hit_s']:.4f} s\n"
                    f"    pos={self.launch_meta['p_launch']}\n"
                    f"    v0={self.launch_meta['v0']}\n"
                    f"    hand_x={self.launch_meta['direction']}",
                    flush=True,
                )
                self.events.emit(
                    self.t_launch,
                    "BALL LAUNCH",
                    f"    speed={self.args.impact_speed} m/s  d_launch={self.launch_meta['d_launch']:.3f} m\n"
                    f"    e_x_GT = {o['e_x']*1e3:+.1f} mm",
                )
            cmd = _cmd_zero(self.params, sim.fsm.r_des)
            self.last_cmd = cmd
            mode = "HOLD"

        elif self.phase == "ball_flight":
            tick_vw_park(sim, np.zeros(3), np.zeros(3), G_HOLD, self.gains, park=False)
            self.ball.note_after_step()
            if self.ball.t_first_contact is not None and self.t_first_contact is None:
                self.t_first_contact = self.ball.t_first_contact
                vpre = self.ball.pre_contact_vel
                spd = float(np.linalg.norm(vpre)) if vpre is not None else float("nan")
                print(
                    f"[t={self.t_first_contact:.3f}] first ball-object contact\n"
                    f"    requested={self.args.impact_speed} m/s  measured_pre={spd:.4f} m/s\n"
                    f"    v_pre={vpre}",
                    flush=True,
                )
                if self.args.pause_at_impact and viewer is not None:
                    print("PAUSED AT IMPACT (wall-clock; physics frozen)", flush=True)
                    self._pause_wall(viewer, 1.5)
            if self.ball.t_last_contact is not None:
                self.t_last_contact = self.ball.t_last_contact
            if self.ball.maybe_park_after_impact():
                self.ball_parked = True
                self.t_park = self.ball.t_park
                print(
                    f"[t={self.t_park:.3f}] ball park AFTER impact (state={self.ball.state})\n"
                    f"    first={self.t_first_contact} last={self.t_last_contact}",
                    flush=True,
                )
                self.phase = "recovery"
                self.t_recovery = float(sim.data.time)
                self.ctrl.reset(grasp_orientation())
                freeze(sim)
                self.clocks = HoldClocks(dt=self.dt)
                if self.args.pause_at_recovery and viewer is not None:
                    print("PAUSED AT RECOVERY (wall-clock; physics frozen)", flush=True)
                    self._pause_wall(viewer, 1.5)
            tmax = 0.5 + 4.0 * float(self.launch_meta.get("T_hit_s", 0.05))
            if self.t_launch is not None and float(sim.data.time) - self.t_launch > tmax:
                print("[warn] impact not completed in time; parking ball", flush=True)
                self.ball.park_now(after_impact=True)
                self.ball_parked = True
                self.t_park = self.ball.t_park
                self.phase = "recovery"
                self.t_recovery = float(sim.data.time)
                self.ctrl.reset(grasp_orientation())
                freeze(sim)
                self.clocks = HoldClocks(dt=self.dt)
            cmd = _cmd_zero(self.params, sim.fsm.r_des)
            self.last_cmd = cmd
            mode = "IMPACT"
            park_now = self.ball_parked

        elif self.phase == "recovery":
            dt = self.dt
            o_rule = obs_from_sim(sim)
            if self.args.controller == "zero":
                cmd = _cmd_zero(self.params, sim.fsm.r_des)
                self.ctrl.s.mode = "ZERO"
            elif self.args.controller == "rule":
                cmd = self.ctrl.step(o_rule, dt)
                cmd["w_world"] = np.zeros(3)
                cmd["w_hy"] = 0.0
                if cmd["mode"] != (self.modes[-1] if self.modes else None):
                    if cmd.get("freeze_p"):
                        freeze(sim)
                if cmd.get("r_des") is not None:
                    sim.fsm.r_des = np.asarray(cmd["r_des"], float).copy()
            else:
                Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
                if int(self.sac.observation_space.shape[0]) == 27:
                    _, tac_p = self._tactile()
                    if self._obs_hist is None:
                        self._obs_hist = ObservableObsState(sim.dt_policy)
                    obs = observe_observable(
                        sim.model, sim.data, sim.ids, sim.fsm, tac_p, self._obs_hist
                    )
                else:
                    obs = observe_recovery4d(sim)
                act, _ = self.sac.predict(obs, deterministic=True)
                mapped = map_recovery4d(act, sim.fsm.r_des, Rh)
                cmd = {
                    "v_world": mapped["v_world"],
                    "w_world": mapped["w_world"],
                    "w_hy": mapped["w_hy"],
                    "tau": mapped["tau"],
                    "r_des": sim.fsm.r_des,
                    "freeze_p": False,
                    "mode": "SAC",
                }
            tick_vw_park(
                sim,
                cmd["v_world"],
                cmd.get("w_world", np.zeros(3)),
                cmd["tau"],
                self.gains,
                park=True,
            )
            self.last_cmd = cmd
            mode = str(cmd["mode"])
            park_now = True
            if self.t_recovery is not None and float(sim.data.time) - self.t_recovery >= self.args.recovery_timeout:
                return False
            if self.args.controller == "rule" and self.ctrl.s.done:
                self.phase = "done"

        if mode and (not self.modes or self.modes[-1] != mode):
            self.modes.append(mode)

        reading, tac = self._tactile()
        o = self._pack()
        t = float(sim.data.time)
        if self.phase in ("recovery", "done"):
            clock = self.clocks.update(o, tac)
        else:
            clock = {
                "success_GT": False,
                "failure_GT": False,
                "fail_kind_GT": "",
                "success_obs": False,
                "failure_obs": False,
            }
        first_c = self.t_first_contact is not None and abs(t - self.t_first_contact) < self.dt * 1.5
        parked_evt = self.t_park is not None and abs(t - self.t_park) < self.dt * 1.5
        self.events.update(t, self.phase, mode, o, tac, clock, first_c, parked_evt)
        ov = overlay_dict(sim, self.phase, mode, o, tac, clock, self.last_cmd, t)
        self._record_hist(ov, t, mode, first_c, parked_evt)
        if abs(t - self.overlay_last_print) >= 0.10:
            print(format_overlay(ov), flush=True)
            self.overlay_last_print = t

        ph = np.array(sim.data.xpos[sim.ids.hand_body], float)
        po = np.array(sim.data.xpos[sim.ids.object_body], float)
        self.trail_h.append(ph.copy())
        self.trail_o.append(po.copy())
        if len(self.trail_h) > 200:
            self.trail_h = self.trail_h[-200:]
            self.trail_o = self.trail_o[-200:]

        if viewer is not None:
            scn = getattr(viewer, "user_scn", None)
            draw_debug_geoms(
                scn,
                sim,
                self.trail_h,
                self.trail_o,
                reading,
                show_raw_contacts=bool(self.args.show_raw_contacts),
                show_cop=True,
                show_trails=True,
                show_hand_x=True,
            )
            if hasattr(viewer, "sync"):
                viewer.sync()

        self._maybe_framing_screenshot()
        self._maybe_frames(viewer)
        if viewer is not None:
            self._pace(t_wall)

        if self.phase == "done":
            return False
        return True

    def run(self, viewer) -> dict:
        self._viewer = viewer
        self._init_camera_once(viewer)
        if viewer is not None:
            enable_viewer_flags(
                viewer,
                show_contact_points=bool(self.args.show_contact_points),
                show_contact_forces=bool(self.args.show_contact_forces),
            )
        running = True
        while running:
            if viewer is not None and hasattr(viewer, "is_running") and not viewer.is_running():
                break
            running = self.step_once(viewer)
        reading, tac = self._tactile()
        o = self._pack()
        clock = {
            "success_GT": self.clocks.ok_gt >= 0.20,
            "failure_GT": self.clocks.lost_gt >= 0.15,
            "success_obs": self.clocks.ok_obs >= 0.20,
            "failure_obs": self.clocks.lost_obs >= 0.15,
        }
        # Recompute from last overlay if hist exists
        if self.hist.get("success_GT"):
            clock["success_GT"] = bool(self.hist["success_GT"][-1])
            clock["failure_GT"] = bool(self.hist["failure_GT"][-1])
            clock["success_obs"] = bool(self.hist["success_obs"][-1])
            clock["failure_obs"] = bool(self.hist["failure_obs"][-1])
        for line in self.events.lines:
            if "CENTER CROSSING" in line and line.startswith("[t="):
                try:
                    self.event_times["center_crossing"].append(float(line.split("]")[0][3:]))
                except ValueError:
                    pass
        integ = snapshot_integrity(self.sim, self.modes, clock)
        summary = {
            "controller": self.args.controller,
            "impact_speed": self.args.impact_speed,
            "mass": MASS,
            "mu": MU,
            "offset": 0.0,
            "ball_mass": 0.50,
            "ball_radius": float(BALL_R),
            "ball_park_pos": BALL_PARK_POS.tolist(),
            "t_launch": self.t_launch,
            "t_first_ball_object_contact": self.t_first_contact,
            "t_last_ball_object_contact": self.t_last_contact,
            "t_park": self.t_park,
            "t_recovery": self.t_recovery,
            "ball": self.ball.measured(),
            "launch": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.launch_meta.items()},
            "final_e_x_GT": float(o["e_x"]),
            "final_e_hat_x": float(tac["e_hat_x"]) if tac["estimate_valid"] else None,
            "success_GT": clock["success_GT"],
            "failure_GT": clock["failure_GT"],
            "success_obs": clock["success_obs"],
            "failure_obs": clock["failure_obs"],
            "modes": self.modes,
            "events": self.events.lines,
            "camera": None
            if self._cam_init is None
            else {
                "preset": self._cam_init["preset"],
                "lookat": np.asarray(self._cam_init["lookat"], float).tolist(),
                "distance": float(self._cam_init["distance"]),
                "azimuth": float(self._cam_init["azimuth"]),
                "elevation": float(self._cam_init["elevation"]),
                "screenshot": getattr(self, "_cam_shot_path", None),
            },
            "integrity_snapshot": {
                "e_x_GT": integ["e_x_GT"],
                "time": integ["time"],
                "success_GT": integ["success_GT"],
                "failure_GT": integ["failure_GT"],
                "success_obs": integ["success_obs"],
                "failure_obs": integ["failure_obs"],
                "modes": integ["modes"],
            },
        }
        return {"summary": summary, "integrity": integ, "hist": dict(self.hist), "event_times": dict(self.event_times)}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Interactive impact+recovery visualization")
    p.add_argument("--impact-speed", type=float, default=9.0)
    p.add_argument("--controller", default="rule", choices=("zero", "rule", "sac"))
    p.add_argument("--checkpoint", default="", help="SAC checkpoint; required for --controller sac")
    p.add_argument("--speed", type=float, default=0.25, help="Playback speed; physics dt unchanged")
    p.add_argument(
        "--ball-launch-distance",
        type=float,
        default=0.30,
        help="Free-flight distance along -hand-x before the target (meters). Engineering, not calibrated.",
    )
    p.add_argument(
        "--legacy-ball-launch",
        action="store_true",
        help="Old near-surface teleport launch (diagnostics/comparison only).",
    )
    p.add_argument("--pause-at-impact", action="store_true")
    p.add_argument("--pause-at-recovery", action="store_true")
    p.add_argument("--record", action="store_true")
    p.add_argument(
        "--preset",
        default="normal",
        choices=("normal", "contact_debug"),
        help="normal: motion-first (default). contact_debug: MuJoCo contact disks/forces on.",
    )
    p.add_argument(
        "--show-contact-points",
        action="store_true",
        help="Enable MuJoCo mjVIS_CONTACTPOINT gold disks (off by default).",
    )
    p.add_argument(
        "--show-contact-forces",
        action="store_true",
        help="Enable MuJoCo mjVIS_CONTACTFORCE (off by default).",
    )
    p.add_argument(
        "--show-raw-contacts",
        action="store_true",
        help="Draw custom yellow spheres at contact.pos (off by default).",
    )
    p.add_argument("--headless", action="store_true")
    p.add_argument(
        "--camera",
        default="impact",
        choices=("impact", "grasp", "free"),
        help="Initial view only; mouse orbit/pan/zoom stay enabled. R resets impact/grasp preset.",
    )
    p.add_argument("--integrity-test", action="store_true")
    p.add_argument("--lift-timeout", type=float, default=12.0)
    p.add_argument("--recovery-timeout", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if args.impact_speed not in IMPACT_SPEEDS:
        raise SystemExit(f"--impact-speed must be one of {IMPACT_SPEEDS}, got {args.impact_speed}")
    if args.preset == "contact_debug":
        args.show_contact_points = True
        args.show_contact_forces = True
    return args


def _write_outputs(args, result) -> dict[str, str]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"impact_v{args.impact_speed}_{args.controller}"
    paths = {}
    log_path = LOG_DIR / f"{tag}.json"
    log_path.write_text(json.dumps(result["summary"], indent=2, default=str), encoding="utf-8")
    paths["log"] = str(log_path)
    ev = LOG_DIR / f"{tag}_events.txt"
    ev.write_text("\n\n".join(result["summary"]["events"]), encoding="utf-8")
    paths["events"] = str(ev)
    if result["hist"].get("t"):
        fig = LOG_DIR / f"{tag}_ex.png"
        save_ex_plot(result["hist"], result["event_times"], fig)
        paths["plot"] = str(fig)
    return paths


def run_headless(args) -> dict:
    np.random.seed(args.seed)
    viz = ImpactRecoveryViz(args)
    return viz.run(None)


def run_viewer(args) -> dict:
    np.random.seed(args.seed)
    viz = ImpactRecoveryViz(args)
    import mujoco.viewer

    def _on_key(key: int) -> None:
        if int(key) in (82, ord("R"), ord("r")):
            viz.reset_camera_to_preset()

    with mujoco.viewer.launch_passive(
        viz.sim.model, viz.sim.data, key_callback=_on_key
    ) as handle:
        return viz.run(handle)


def integrity_test(args) -> dict:
    a = argparse.Namespace(**vars(args))
    a.headless = True
    a.pause_at_impact = False
    a.pause_at_recovery = False
    a.record = False
    a.speed = 0.0
    r1 = run_headless(a)
    r2 = run_headless(a)
    cmp_ = compare_integrity(r1["integrity"], r2["integrity"])
    return {"compare": cmp_, "run0": r1["summary"], "run1": r2["summary"]}


def main(argv=None) -> int:
    args = parse_args(argv)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    if args.integrity_test:
        args.speed = 0.0
        args.headless = True
        out = integrity_test(args)
        path = LOG_DIR / "integrity_headless_repeat.json"
        path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
        print(json.dumps(out["compare"], indent=2))
        print(f"wrote {path}")
        print(
            "Viewer instrumentation: overlay/trails/events are read-only. "
            "viewer.sync() is display-only. Headless vs visualized use the same "
            "step_once(); pause flags are disabled for this test."
        )
        return 0 if out["compare"]["match"] else 1

    if args.headless:
        result = run_headless(args)
    else:
        result = run_viewer(args)
    paths = _write_outputs(args, result)
    if args.record and result.get("hist"):
        tag = f"impact_v{args.impact_speed}_{args.controller}"
        frames = VIDEO_DIR / f"_frames_v{args.impact_speed}_{args.controller}"
        mp4 = try_write_mp4(frames, VIDEO_DIR / f"{tag}.mp4")
        if mp4:
            paths["video"] = mp4
        else:
            paths["frames"] = str(frames)
    print("outputs:", json.dumps(paths, indent=2))
    print("final:", json.dumps({k: result["summary"][k] for k in (
        "t_launch", "t_first_ball_object_contact", "t_last_ball_object_contact",
        "t_park", "t_recovery", "final_e_x_GT", "success_GT", "failure_GT",
        "success_obs", "failure_obs", "modes",
    )}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
