"""Gymnasium recovery environment (starts from deteriorating lift states)."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from controllers.residual import ResidualCommand
from envs.config_util import load_yaml, merge_sim_config
from envs.grasp_sim import GraspSim


class RecoveryEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: dict | None = None,
        config_path: str | Path = "config/randomized.yaml",
        snapshots: list[dict] | None = None,
        action_kind: str = "2d",
        use_contact_obs: bool = True,
    ):
        super().__init__()
        if cfg is None:
            cfg = merge_sim_config(load_yaml(config_path))
        self.cfg = cfg
        self.action_kind = action_kind
        self.use_contact_obs = bool(use_contact_obs and cfg.get("use_contact", True))
        self.sim = GraspSim(cfg)
        self.limiter = self.sim.limiter
        self.snapshots = snapshots or []
        self._i = 0
        n_act = {"2d": 2, "3d": 3}.get(action_kind, 7)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n_act,), dtype=np.float32)
        self.sim.reset()
        self.sim.meter.compute(
            self.sim.model, self.sim.data, self.sim.ids, self.sim.dt_policy
        )
        obs0 = self.sim.observe(self.use_contact_obs)
        self.observation_space = spaces.Box(
            low=-1e6, high=1e6, shape=obs0.shape, dtype=np.float32
        )
        self.max_steps = int(
            round(float(cfg["recovery"]["timeout"]) / self.sim.dt_policy)
        )
        self.n_steps = 0
        self._last_D = 0.0
        self._recovered_steps = 0

    def _reward(self, D: float, cmd: ResidualCommand, terminated_kind: str | None) -> float:
        rw = self.cfg["reward"]
        lam_d = float(rw.get("lambda_D", 1.0))
        gamma = float(rw.get("gamma", self.cfg.get("sac", {}).get("gamma", 0.99)))
        phi = -lam_d * D
        phi_prev = -lam_d * self._last_D
        r = gamma * phi - phi_prev
        r -= float(rw.get("lambda_f", 0.0)) * (self.sim.fsm.fg_cmd ** 2)
        r -= float(rw.get("lambda_c", 0.0)) * (
            float(np.dot(cmd.dv, cmd.dv) + np.dot(cmd.dw, cmd.dw))
        )
        r -= float(rw.get("lambda_t", 0.0))
        if terminated_kind == "recovered":
            r += float(rw.get("recovered", 8.0))
        elif terminated_kind == "drop":
            r -= float(rw.get("drop", 8.0))
        return float(r)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        snap = options.get("snapshot")
        if snap is None and self.snapshots:
            if seed is not None:
                self._i = int(seed) % len(self.snapshots)
            snap = self.snapshots[self._i % len(self.snapshots)]
            self._i += 1
        if snap is None:
            from envs.dynamics import sample_uncertainty
            from envs.grasp_sim import run_to_lift_or_deterioration

            rng = np.random.default_rng(seed)
            mass, friction, offset = sample_uncertainty(rng, self.cfg, heldout=False)
            self.sim.reset(mass, friction, offset)
            outcome = run_to_lift_or_deterioration(self.sim)
            if outcome != "deteriorate":
                self.sim.reset(mass, friction, offset)
                self.sim.fsm.phase = "lift"
                self.sim.maybe_capture_reference()
        else:
            self.sim.load_snapshot(snap)
        self.limiter.reset()
        self.n_steps = 0
        self._recovered_steps = 0
        d = self.sim.meter.compute(
            self.sim.model, self.sim.data, self.sim.ids, self.sim.dt_policy
        )
        self.sim.meter.in_recovery = True
        self._last_D = d.D
        obs = self.sim.observe(self.use_contact_obs)
        info = {"D": d.D, "Ddot": d.Ddot, "mass": self.sim.mass, "friction": self.sim.friction}
        return obs, info

    def step(self, action):
        raw = self.limiter.from_action(action, self.action_kind)
        cmd = self.limiter.apply(raw, self.sim.dt_policy)
        self.sim.policy_tick(cmd, in_recovery=True)
        d = self.sim.meter.compute(
            self.sim.model, self.sim.data, self.sim.ids, self.sim.dt_policy
        )
        self.n_steps += 1
        kind = None
        terminated = False
        truncated = False
        if d.D < self.sim.meter.D_exit:
            self._recovered_steps += 1
        else:
            self._recovered_steps = 0
        if self._recovered_steps >= self.sim.meter.N_stable:
            kind = "recovered"
            terminated = True
        elif self.sim.dropped() or self.sim.bad_state():
            kind = "drop"
            terminated = True
        elif self.n_steps >= self.max_steps:
            truncated = True
            kind = "timeout"
        reward = self._reward(d.D, cmd, kind)
        self._last_D = d.D
        obs = self.sim.observe(self.use_contact_obs)
        pref = self.sim.meter.prel_ref
        slip = float(np.linalg.norm(d.p_rel - (d.p_rel if pref is None else pref)))
        info = {
            "D": d.D,
            "Ddot": d.Ddot,
            "fg": self.sim.fsm.fg_cmd,
            "vz": float(self.sim.fsm.v_cmd[2]),
            "dvy_requested": float(raw.dv[1]),
            "dvy": float(cmd.dv[1]),
            "dvz": float(cmd.dv[2]),
            "dfg": float(cmd.dfg),
            "v_cmd_y": float(self.sim.fsm.v_cmd[1]),
            "v_cmd_z": float(self.sim.fsm.v_cmd[2]),
            "residual_mag": float(np.linalg.norm(cmd.dv) + np.linalg.norm(cmd.dw) + abs(cmd.dfg)),
            "slip": slip,
            "term": kind or "",
        }
        return obs, reward, terminated, truncated, info


def make_env_from_buffer(
    buffer_path: Path,
    cfg: dict,
    action_kind: str = "2d",
    use_contact_obs: bool = True,
) -> RecoveryEnv:
    snaps = load_buffer(buffer_path)
    return RecoveryEnv(
        cfg=cfg,
        snapshots=snaps,
        action_kind=action_kind,
        use_contact_obs=use_contact_obs,
    )


def save_buffer(path: Path, snapshots: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, snapshots=np.array(snapshots, dtype=object))


def load_buffer(path: Path) -> list[dict]:
    data = np.load(path, allow_pickle=True)
    snaps = data["snapshots"]
    return list(snaps)
