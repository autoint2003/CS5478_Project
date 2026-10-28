"""Plan C curriculum environments: tracking → catching → full-horizon.

No Oracle actions/phase labels in observations. Physical events only.
Does not mid-abort on TABLE_TOP/clear (Plan C catch geometry).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from envs.observations.airborne_obs import observe_airborne
from envs.observations.observable_obs import ObservableObsState
from envs.outcomes.ground_drop import arm, latched
from envs.outcomes.physical_recovery import physical_pack
from envs.step_legal import step_legal
from evaluation.axis_aware_success import evaluate_handoff_with_axis
from evaluation.grasp_geometry import measure
from evaluation.success_criteria import P_TGT, U_TGT
from experiments.precision.p18_discrete_vocab_audit.replay_harness import make_env
from experiments.precision.p21_full_stage1_experts.run_p21 import restore_start
from experiments.precision.p23_c_d2_sft.run_p23 import contact_flags
from models.m111_checkpoint import HANDOFF, decode_m111_action
from models.transformer_policy import DT_POLICY
from training.recovery_execution import make_snap, restore_ballistic

Stage = Literal["tracking", "catching", "full"]


def grasp_metrics(sim) -> dict:
    m = measure(sim, U_TGT, P_TGT)
    o = physical_pack(sim)
    cf = contact_flags(o)
    vrel = float(np.linalg.norm(o.get("v_rel_h") if "v_rel_h" in o else o.get("v_rel", 0.0)))
    if isinstance(o.get("v_rel_h"), np.ndarray):
        vrel = float(np.linalg.norm(o["v_rel_h"]))
    elif "v_rel" in o:
        vrel = float(o["v_rel"])
    return {
        "e_pos_m": float(m["e_pos_m"]),
        "e_theta_deg": float(m["e_theta_deg"]),
        "obj_z": float(o.get("obj_z") or 0.0),
        "bilateral": int(cf["bilateral"]),
        "any_contact": int(cf["any_contact"]),
        "nL": int(cf["nL"]),
        "nR": int(cf["nR"]),
        "vrel": vrel,
        "scene": int(o.get("scene") or 0),
        "clear": float(o.get("clear") or 0.0),
        "latched": int(latched(sim)),
    }


@dataclass
class EpisodeStart:
    kind: str  # recovery_ic | snap
    ep: dict | None = None
    snap: dict | None = None
    ref: Any = None
    difficulty: float = 1.0  # 0 easy .. 1 hard
    label: str = ""


class PlanCCurriculumEnv:
    def __init__(self, book, rho, *, stage: Stage = "tracking", max_steps: int = 100):
        self.book = book
        self.rho = rho
        self.stage = stage
        self.max_steps = int(max_steps)
        self.env = make_env()
        self.obs_seq: list = []
        self.prev_seq: list = []
        self.t = 0
        self.prev_tok = None
        self.switches = 0
        self._prev_metrics = None
        self._had_contact = False
        self._had_bilateral = False
        self._bilateral_hold = 0
        self._lifted = False
        self._z_at_bil = None
        self.start: EpisodeStart | None = None
        self.last_info: dict = {}

    def _observe(self) -> np.ndarray:
        hist = self.env._hist
        return np.asarray(
            observe_airborne(
                self.env.sim.model, self.env.sim.data, self.env.sim.ids, self.env.sim.fsm, hist
            ),
            np.float32,
        )

    def reset(self, start: EpisodeStart) -> np.ndarray:
        self.start = start
        self.t = 0
        self.prev_tok = None
        self.switches = 0
        self._had_contact = False
        self._had_bilateral = False
        self._bilateral_hold = 0
        self._lifted = False
        self._z_at_bil = None
        if start.kind == "snap":
            restore_ballistic(self.env.sim, start.snap)
            from envs.outcomes.ground_drop import restore_state as restore_drop

            restore_drop(self.env.sim, None)
            arm(self.env.sim)
            self.env._hist = ObservableObsState(DT_POLICY)
            self.env._prev = np.zeros(7, np.float32)
            self.env.ref = start.ref
        else:
            restore_start(self.env, start.ep)
            arm(self.env.sim)
            self.env.ref = start.ep["ref"]
        self.obs_seq = []
        self.prev_seq = []
        obs = self._observe()
        self.obs_seq.append(obs)
        self.prev_seq.append(self.env._prev.copy())
        self._prev_metrics = grasp_metrics(self.env.sim)
        return obs

    def step(self, token: int, resid: np.ndarray) -> tuple[float, bool, dict]:
        """Apply one M111 token (+ residual). Returns reward, done, info."""
        m0 = self._prev_metrics
        if token == HANDOFF:
            action = None
        else:
            action = decode_m111_action(self.book, int(token), resid, rho=self.rho)

        if self.prev_tok is not None and int(token) != int(self.prev_tok):
            self.switches += 1
        self.prev_tok = int(token)

        terminated = False
        truncated = False
        reason = None
        if action is None:
            # HANDOFF only meaningful in full stage near success
            if self.stage == "full" and self._had_bilateral and self._lifted:
                hand = evaluate_handoff_with_axis(self.env.sim, self.env.ref, stabilize_n=8)
                info = {
                    "SUCCESS_NEW": bool(hand.get("SUCCESS_NEW")),
                    "SUCCESS_12S": bool(hand.get("SUCCESS_12S")),
                    "event": "HANDOFF",
                    "reason": hand.get("std_reason") or hand.get("reason"),
                }
                r = 5.0 if hand.get("SUCCESS_NEW") else -0.5
                self.last_info = info
                return r, True, info
            reason = "EARLY_HANDOFF"
            terminated = True
            r = -1.0
            m1 = grasp_metrics(self.env.sim)
        else:
            step_legal(self.env.sim, np.asarray(action, np.float32))
            self.env._prev = np.asarray(action, np.float32)
            m1 = grasp_metrics(self.env.sim)
            self.t += 1
            if m1["latched"] or m1["scene"] > 0:
                terminated = True
                reason = "PHYSICAL_DROP"
            if self.t >= self.max_steps:
                truncated = True
                reason = reason or "TIMEOUT"

        # Event tracking
        if m1["any_contact"]:
            self._had_contact = True
        if m1["bilateral"]:
            if not self._had_bilateral:
                self._z_at_bil = m1["obj_z"]
            self._had_bilateral = True
            self._bilateral_hold += 1
            if self._z_at_bil is not None and m1["obj_z"] > self._z_at_bil + 0.08:
                self._lifted = True
        else:
            self._bilateral_hold = 0

        r = self._reward(m0, m1, token, terminated, truncated, reason)
        done = terminated or truncated

        # Stage-specific success termination
        if self.stage == "tracking" and self._had_contact and m1["e_pos_m"] < 0.06 and m1["vrel"] < 0.35:
            done = True
            reason = reason or "INTERCEPT_OK"
            r += 2.0
        if self.stage == "catching" and self._had_bilateral and self._bilateral_hold >= 8:
            if self._lifted:
                done = True
                reason = reason or "CATCH_LIFT_OK"
                r += 3.0
            elif self._bilateral_hold >= 15:
                done = True
                reason = reason or "BILATERAL_STABLE"
                r += 2.0

        if not done:
            obs = self._observe()
            self.obs_seq.append(obs)
            self.prev_seq.append(self.env._prev.copy())
        self._prev_metrics = m1
        info = {
            "metrics": m1,
            "reason": reason,
            "switches": self.switches,
            "had_contact": self._had_contact,
            "had_bilateral": self._had_bilateral,
            "lifted": self._lifted,
            "t": self.t,
            "SUCCESS_NEW": False,
        }
        self.last_info = info
        return float(r), bool(done), info

    def _reward(self, m0, m1, token, terminated, truncated, reason) -> float:
        r = 0.0
        # progress toward grasp-feasible relative pose (not COM-only)
        # use e_pos to locked P_TGT (grasp geometry)
        de = float(m0["e_pos_m"] - m1["e_pos_m"])
        if self.stage in ("tracking", "catching", "full"):
            r += 1.5 * np.clip(de, -0.05, 0.05) / 0.05
            r += 0.5 * np.clip(m0["vrel"] - m1["vrel"], -0.2, 0.2) / 0.2
            r += 0.3 * np.clip((m0["e_theta_deg"] - m1["e_theta_deg"]) / 30.0, -0.5, 0.5)
        # contact / bilateral events
        if m1["any_contact"] and not m0["any_contact"]:
            r += 0.8 if self.stage != "full" else 0.4
        if m1["bilateral"] and not m0["bilateral"]:
            r += 1.5
        if m1["bilateral"]:
            r += 0.05
        if self._lifted and m1["bilateral"]:
            r += 0.1
        # penalties
        r -= 0.01  # step cost
        if self.prev_tok is not None and token != self.prev_tok:
            # mild switch penalty only if oscillating (handled via switches density)
            pass
        if self.switches > 0 and self.t > 0 and (self.switches / max(1, self.t)) > 0.7:
            r -= 0.02
        if reason == "PHYSICAL_DROP":
            r -= 3.0
        if reason == "EARLY_HANDOFF":
            r -= 1.0
        # Do NOT reward incidental table support / clear alone
        return float(r)
