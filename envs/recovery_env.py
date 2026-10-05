"""Gymnasium recovery environment. Final SAC mode: recovery4d.

D_t has no control authority (not entry, success, termination, or masking).
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from controllers.residual import ResidualCommand, map_recovery4d
from envs.config_util import load_yaml, merge_sim_config
from envs.grasp_sim import GraspSim
from envs.observable_obs import OBS_DIM_OBSERVABLE, ObservableObsState, observe_observable
from envs.observable_reward import from_cfg as observable_from_cfg
from envs.observable_reward import read_tactile
from envs.physical_recovery import (
    CONTACT_LOSS_HOLD,
    E_SCALE,
    OBS_DIM,
    SUCCESS_HOLD,
    fail_kind,
    observe_recovery4d,
    physical_pack,
    recovered,
)
from training.replay_core import disable_object_table


class RecoveryEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: dict | None = None,
        config_path: str | Path = "config/randomized.yaml",
        snapshots: list[dict] | None = None,
        action_kind: str = "recovery4d",
        use_contact_obs: bool = True,
        reward_mode: str | None = None,
    ):
        super().__init__()
        if cfg is None:
            cfg = merge_sim_config(load_yaml(config_path))
        self.cfg = cfg
        mode = reward_mode if reward_mode is not None else cfg.get("reward", {}).get("mode", "oracle")
        self.reward_mode = str(mode)
        if self.reward_mode not in ("oracle", "observable_tactile"):
            raise ValueError(f"reward_mode must be oracle|observable_tactile, got {self.reward_mode}")
        self.action_kind = action_kind
        self.use_contact_obs = bool(use_contact_obs and cfg.get("use_contact", True))
        self.sim = GraspSim(cfg)
        if action_kind == "recovery4d":
            disable_object_table(self.sim)
        self.limiter = self.sim.limiter
        self.snapshots = snapshots or []
        self._i = 0
        n_act = {"2d": 2, "3d": 3, "recovery4d": 4}.get(action_kind, 7)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n_act,), dtype=np.float32)
        self.sim.reset()
        self._obs_rw = observable_from_cfg(cfg, self.sim.dt_policy)
        self._obs_hist = ObservableObsState(self.sim.dt_policy)
        if action_kind == "recovery4d":
            disable_object_table(self.sim)
            obs0 = self._policy_obs()
        else:
            self.sim.meter.compute(
                self.sim.model, self.sim.data, self.sim.ids, self.sim.dt_policy
            )
            obs0 = self.sim.observe(self.use_contact_obs)
        self.observation_space = spaces.Box(
            low=-1.0 if action_kind == "recovery4d" else -1e6,
            high=1.0 if action_kind == "recovery4d" else 1e6,
            shape=obs0.shape,
            dtype=np.float32,
        )
        if action_kind == "recovery4d":
            want = OBS_DIM_OBSERVABLE if self.reward_mode == "observable_tactile" else OBS_DIM
            assert obs0.shape[0] == want, (obs0.shape, want, self.reward_mode)
        self.max_steps = int(
            round(float(cfg["recovery"]["timeout"]) / self.sim.dt_policy)
        )
        self.n_steps = 0
        self._last_ex = 0.0
        self._last_D = 0.0
        self._ok_s = 0.0
        self._lost_s = 0.0

    def _policy_obs(self) -> np.ndarray:
        if self.action_kind != "recovery4d":
            return self.sim.observe(self.use_contact_obs)
        if self.reward_mode == "observable_tactile":
            _, tac = read_tactile(self.sim.model, self.sim.data, self.sim.ids)
            return observe_observable(
                self.sim.model, self.sim.data, self.sim.ids, self.sim.fsm, tac, self._obs_hist
            )
        return observe_recovery4d(self.sim)

    def _reward(self, o: dict, a_used: np.ndarray, kind: str | None) -> float:
        """ORACLE dense+terminal reward. Uses GT e_x / nL nR / recovered-fail kinds."""
        rw = self.cfg.get("reward", {})
        e_scale = float(rw.get("e_scale", E_SCALE))
        phi = -abs(float(o["e_x"])) / max(e_scale, 1e-9)
        phi_prev = -abs(self._last_ex) / max(e_scale, 1e-9)
        r = phi - phi_prev
        bilat = int(o["nL"]) > 0 and int(o["nR"]) > 0
        r += float(rw.get("lambda_cap", 0.02)) * (1.0 if bilat else -0.5)
        r -= float(rw.get("lambda_t", 0.01))
        a = np.asarray(a_used, float).reshape(-1)
        lam_a = float(rw.get("lambda_a", 0.01))
        if self.action_kind == "recovery4d" and a.size >= 3:
            r -= lam_a * float(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)
        else:
            r -= lam_a * float(np.dot(a, a))
        lam_d = float(rw.get("lambda_D", 0.0))
        if lam_d:
            r += lam_d * (self._last_D - float(o.get("D", self._last_D)))
        if kind == "recovered":
            r += float(rw.get("recovered", 8.0))
        elif kind in ("drop", "scene", "contact_loss"):
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
            from training.offset_construct import make_one_training_ic

            rng = np.random.default_rng(None if seed is None else seed)
            snap = make_one_training_ic(self.cfg, rng)
        self.sim.load_snapshot(snap)
        if self.action_kind == "recovery4d":
            disable_object_table(self.sim)
            mujoco_fwd = __import__("mujoco")
            mujoco_fwd.mj_forward(self.sim.model, self.sim.data)
        self.limiter.reset()
        self.n_steps = 0
        self._ok_s = 0.0
        self._lost_s = 0.0
        o = physical_pack(self.sim)
        self._last_ex = float(o["e_x"])
        d = self.sim.meter.compute(
            self.sim.model, self.sim.data, self.sim.ids, self.sim.dt_policy
        )
        self._last_D = float(d.D)
        _, tac = read_tactile(self.sim.model, self.sim.data, self.sim.ids)
        self._obs_rw.seed_from_reading(tac)
        self._obs_hist.reset()
        obs = self._policy_obs()
        info = {
            "e_x": float(o["e_x"]),
            "D": d.D,
            "nL": o["nL"],
            "nR": o["nR"],
            "mass": self.sim.mass,
            "friction": self.sim.friction,
            "reward_mode": self.reward_mode,
            "e_hat_x": tac["e_hat_x"],
            "estimate_valid": tac["estimate_valid"],
            "success_GT": False,
            "failure_GT": False,
            "success_obs": False,
            "failure_obs": False,
        }
        return obs, info

    def step(self, action):
        a = np.asarray(action, dtype=float).reshape(-1)
        mapped = None
        if self.action_kind == "recovery4d":
            r_h = np.array(self.sim.data.xmat[self.sim.ids.hand_body].reshape(3, 3), float)
            mapped = map_recovery4d(a, self.sim.fsm.r_des, r_h)
            self.sim.recovery4d_tick(mapped["v_world"], mapped["w_world"], mapped["tau"])
            a_used = mapped["a"]
        else:
            raw = self.limiter.from_action(a, self.action_kind)
            cmd = self.limiter.apply(raw, self.sim.dt_policy)
            self.sim.policy_tick(cmd, in_recovery=True)
            a_used = np.clip(a, -1.0, 1.0)
        o = physical_pack(self.sim)
        d = self.sim.meter.compute(
            self.sim.model, self.sim.data, self.sim.ids, self.sim.dt_policy
        )
        o["D"] = d.D
        self.n_steps += 1
        dt = self.sim.dt_policy
        _, tac = read_tactile(self.sim.model, self.sim.data, self.sim.ids)
        obs_terms = self._obs_rw.terms(tac, a_used)
        if recovered(o):
            self._ok_s += dt
        else:
            self._ok_s = 0.0
        if int(o["nL"]) == 0 and int(o["nR"]) == 0:
            self._lost_s += dt
        else:
            self._lost_s = 0.0
        kind_gt = None
        if self._ok_s >= SUCCESS_HOLD:
            kind_gt = "recovered"
        else:
            fk = fail_kind(o, self._lost_s)
            if fk:
                kind_gt = fk
            elif self.n_steps >= self.max_steps:
                kind_gt = "timeout"
        terminated = False
        truncated = False
        kind = None
        if self.reward_mode == "oracle":
            if kind_gt == "recovered":
                kind = "recovered"
                terminated = True
            elif kind_gt in ("drop", "scene", "contact_loss"):
                kind = kind_gt
                terminated = True
            elif kind_gt == "timeout":
                truncated = True
                kind = "timeout"
            reward = self._reward(o, a_used, kind)
        else:
            if obs_terms["success_obs"]:
                kind = "success_obs"
                terminated = True
            elif obs_terms["failure_obs"]:
                kind = "failure_obs"
                terminated = True
            elif self.n_steps >= self.max_steps:
                truncated = True
                kind = "timeout"
            reward = float(obs_terms["r"])
            if not terminated:
                # timeout has no ±8; terms() only adds terminal when success/fail
                if kind == "timeout":
                    pass
        self._last_ex = float(o["e_x"])
        self._last_D = float(d.D)
        obs = self._policy_obs()
        info = {
            "e_x": float(o["e_x"]),
            "D": d.D,
            "nL": o["nL"],
            "nR": o["nR"],
            "v_rel": o["v_rel"],
            "w_rel": o["w_rel"],
            "tau": o["tau"],
            "obj_z": o["obj_z"],
            "clear": o["clear"],
            "scene": o["scene"],
            "term": kind or "",
            "term_GT": kind_gt or "",
            "v_cmd": np.asarray(self.sim.fsm.v_cmd, float).copy(),
            "w_cmd": np.asarray(self.sim.fsm.w_cmd, float).copy(),
            "reward_mode": self.reward_mode,
            "e_hat_x": tac["e_hat_x"],
            "estimate_valid": tac["estimate_valid"],
            "bilateral_valid": tac["bilateral_valid"],
            "progress_valid": obs_terms["progress_valid"],
            "success_obs": obs_terms["success_obs"],
            "failure_obs": obs_terms["failure_obs"],
            "success_GT": kind_gt == "recovered",
            "failure_GT": kind_gt in ("drop", "scene", "contact_loss"),
            "recovered_GT_now": bool(recovered(o)),
            "r_progress": obs_terms["r_progress"],
            "r_contact": obs_terms["r_contact"],
            "r_time": obs_terms["r_time"],
            "r_action": obs_terms["r_action"],
            "r_term": obs_terms["r_term"] if self.reward_mode == "observable_tactile" else 0.0,
            "obs_dim": int(obs.shape[0]),
        }
        if mapped is not None:
            info["v_hx"] = mapped["v_hx"]
            info["v_z"] = mapped["v_z"]
            info["w_hy"] = mapped["w_hy"]
        return obs, reward, terminated, truncated, info


def make_env_from_buffer(
    buffer_path: Path,
    cfg: dict,
    action_kind: str = "recovery4d",
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
