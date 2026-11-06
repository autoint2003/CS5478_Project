"""StabilizeGrpoEnv + optional C-only grasp event rewards (P57-A base unchanged)."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv
from experiments.precision.p63_transformer_ppo_c_grasp.contact_events import CGraspEventTracker
from experiments.precision.p63_transformer_ppo_c_grasp.paths import MAX_EP_STEPS, T_MAX


class CGraspEnv(StabilizeGrpoEnv):
    """P57-A physics/reward; C episodes may add once-only grasp event rewards."""

    def __init__(
        self,
        book,
        rho,
        *,
        enable_c_grasp_reward: bool = False,
        max_steps: int = MAX_EP_STEPS,
        t_max: int = T_MAX,
        weights: PoseRewardWeights | None = None,
    ):
        super().__init__(
            book,
            rho,
            max_steps=max_steps,
            t_max=t_max,
            weights=weights,
            use_stabilization=False,
        )
        self.enable_c_grasp_reward = bool(enable_c_grasp_reward)
        self.c_tracker = CGraspEventTracker()
        self.r_raw_sum = 0.0
        self.r_c_sum = 0.0

    def reset(self, start):
        obs = super().reset(start)
        self.c_tracker = CGraspEventTracker()
        self.r_raw_sum = 0.0
        self.r_c_sum = 0.0
        return obs

    def step(self, token: int, resid: np.ndarray) -> tuple[float, bool, dict]:
        r_raw, done, info = super().step(token, resid)
        self.r_raw_sum += float(r_raw)
        r_c = 0.0
        flags: dict[str, Any] = {}
        is_c = str(getattr(self.start, "skill", "") or "") == "C"
        if self.enable_c_grasp_reward and is_c:
            reason = info.get("handoff_reason")
            r3f_early = bool(info.get("handoff")) and str(reason) == "R3_EARLY"
            # Also fire when dwell just became ready on this step without requiring
            # SUCCESS path — handoff True + R3_EARLY is the gate.
            r_c, flags = self.c_tracker.step(
                self.env.sim,
                metrics=info.get("metrics") or {},
                r3f_early_ready=r3f_early,
                dropped=bool(info.get("drop")),
            )
            self.r_c_sum += float(r_c)

        total = float(r_raw) + float(r_c)
        bd = dict(info.get("reward_breakdown") or {})
        bd["r_raw"] = float(r_raw)
        bd["r_c_grasp"] = float(r_c)
        bd["r_raw_sum"] = float(self.r_raw_sum)
        bd["r_c_sum"] = float(self.r_c_sum)
        bd["c_events"] = list(self.c_tracker.log)
        info["reward_breakdown"] = bd
        info["c_grasp_flags"] = flags
        info["r_raw"] = float(r_raw)
        info["r_c_grasp"] = float(r_c)
        # SUCCESS_RELAXED_12S must be unchanged by reward path
        return total, bool(done), info


def make_env(book, rho, *, enable_c_grasp_reward: bool = False) -> CGraspEnv:
    return CGraspEnv(
        book,
        rho,
        enable_c_grasp_reward=enable_c_grasp_reward,
        weights=PoseRewardWeights(),
    )
